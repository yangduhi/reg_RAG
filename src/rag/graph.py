from typing import List, TypedDict, Literal

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, StateGraph

from src.core.logging import logger
from src.core.config import settings

# --- State Definition ---
class GraphState(TypedDict):
    """
    RAG 그래프의 상태를 저장하는 딕셔너리입니다.
    """
    question: str           # 사용자 질문
    documents: List[Document] # 검색된 문서 리스트
    generation: str         # 생성된 답변
    search_count: int       # 재검색 횟수 (무한 루프 방지)

# --- Data Models for LLM Structure Output ---
class GradeDocuments(BaseModel):
    """문서의 관련성 평가를 위한 바이너리 스코어 모델"""
    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

class RAGGraph:
    def __init__(self, rag_engine):
        """
        RAGGraph 초기화
        
        Args:
            rag_engine: 기존 RAGEngine 인스턴스 (검색 및 LLM 기능 활용)
        """
        self.engine = rag_engine
        self.app = self._build_graph()

    def _build_graph(self):
        """LangGraph 워크플로우 정의 및 컴파일"""
        workflow = StateGraph(GraphState)

        # 노드 등록
        workflow.add_node("retrieve", self.retrieve)
        workflow.add_node("grade_documents", self.grade_documents)
        workflow.add_node("transform_query", self.transform_query)
        workflow.add_node("generate", self.generate)

        # 엣지 연결
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "grade_documents")
        
        # 조건부 엣지: 문서 평가 후 분기
        workflow.add_conditional_edges(
            "grade_documents",
            self.decide_to_generate,
            {
                "transform_query": "transform_query",
                "generate": "generate",
            },
        )
        
        workflow.add_edge("transform_query", "retrieve")
        workflow.add_edge("generate", END)

        return workflow.compile()

    # --- Nodes ---

    async def retrieve(self, state: GraphState):
        """검색 노드: 하이브리드 검색 수행"""
        question = state["question"]
        search_count = state.get("search_count", 0)
        
        logger.info(f"--- [Graph: Retrieve] 검색 수행 (시도: {search_count+1}) ---")
        
        # RAGEngine의 검색 로직 재사용
        # 주의: engine.chat() 내부 로직을 쪼개서 가져와야 함.
        # 여기서는 engine.get_retrievers()를 사용하여 직접 검색함.
        
        retrievers = self.engine.get_retrievers(use_mmr=True)
        tasks = [retriever.ainvoke(question) for retriever in retrievers]
        
        import asyncio
        results = await asyncio.gather(*tasks)
        
        # 중복 제거
        unique_docs = {}
        for doc_list in results:
            for doc in doc_list:
                if doc.page_content not in unique_docs:
                    unique_docs[doc.page_content] = doc
        
        documents = list(unique_docs.values())
        logger.info(f"--- [Graph: Retrieve] 검색된 문서 수: {len(documents)} ---")
        
        return {"documents": documents, "question": question, "search_count": search_count + 1}

    async def grade_documents(self, state: GraphState):
        """평가 노드: 문서와 질문의 관련성 평가"""
        question = state["question"]
        documents = state["documents"]
        
        # [최적화] 너무 많은 문서를 평가하면 느리므로 상위 10개만 평가 (또는 설정에 따름)
        # 검색 단계에서 이미 점수순으로 정렬되어 있으므로 상위 문서가 가장 중요함.
        documents_to_grade = documents[:10] 
        logger.info(f"--- [Graph: Grade] 상위 {len(documents_to_grade)}개 문서 평가 시작 ---")
        
        # 평가용 LLM 체인 설정
        llm_with_tool = self.engine.llm.with_structured_output(GradeDocuments)
        
        prompt = ChatPromptTemplate.from_template(
            """You are a grader assessing relevance of a retrieved document to a user question. \n 
            Here is the retrieved document: \n\n {context} \n\n
            Here is the user question: {question} \n
            If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
            Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
        )
        
        chain = prompt | llm_with_tool
        
        # [최적화] 비동기 병렬 처리
        import asyncio
        
        async def grade_doc(doc):
            try:
                score = await chain.ainvoke({"question": question, "context": doc.page_content})
                return doc if score.binary_score == "yes" else None
            except Exception:
                return doc # 에러 시 안전하게 포함

        # 모든 문서에 대해 병렬로 grade 실행
        results = await asyncio.gather(*[grade_doc(d) for d in documents_to_grade])
        
        # None(관련 없음) 제외
        filtered_docs = [doc for doc in results if doc is not None]
        
        logger.info(f"--- [Graph: Grade] 관련 있는 문서: {len(filtered_docs)} / {len(documents_to_grade)} (Total Retrieved: {len(documents)}) ---")
        return {"documents": filtered_docs, "question": question}

    async def transform_query(self, state: GraphState):
        """쿼리 변환 노드: 검색 결과가 없을 때 질문 재작성"""
        question = state["question"]
        logger.info(f"--- [Graph: Transform] 쿼리 재작성 중... (원본: {question}) ---")
        
        # RAGEngine의 transform_query 재사용
        better_question = await self.engine.transform_query(question)
        
        return {"question": better_question}

    async def generate(self, state: GraphState):
        """생성 노드: 최종 답변 생성"""
        logger.info("--- [Graph: Generate] 답변 생성 ---")
        question = state["question"]
        documents = state["documents"]
        
        # RAGEngine의 generate 로직 (chat 메서드의 후반부) 재사용을 위해
        # 여기서 직접 프롬프트 체인을 구성해야 함. (engine.py의 코드를 모듈화하는 것이 좋음)
        
        template = """
        You are an expert in Automotive Safety Regulations.
        Based on the provided [Context], write an accurate and professional answer to the [Question].

        [Answer Guidelines]
        1. **Fact-Based:** Use only the information contained in the [Context]. Do not use external knowledge or speculation.
        2. **Handle Unknowns:** If the answer is not in the [Context], honestly state, "I could not find the relevant information in the provided documents."
        3. **Accuracy:** Quote numerical values (voltage, length, angle, etc.) and table data exactly as they appear.
        4. **Cite Sources:** At the end of your answer, you MUST cite the regulation ID or section number that is the basis for your answer. (e.g., [Source: FMVSS 108 S7.3])
        5. **Language:** Respond politely and clearly in Korean.

        [Context]
        {context}

        [Question]
        {question}

        [Answer]
        """
        
        def format_docs(docs):
            return "\n\n".join(
                f"--- Document Start ({d.metadata.get('standard_id', 'Unknown')}) ---\\n{d.page_content}\\n--- Document End ---"
                for d in docs
            )
            
        context_text = format_docs(documents)
        
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.engine.llm | StrOutputParser()
        
        response = await chain.ainvoke({"context": context_text, "question": question})
        
        return {"generation": response}

    # --- Conditional Logic ---

    def decide_to_generate(self, state: GraphState) -> Literal["transform_query", "generate"]:
        """
        다음 단계 결정: 문서가 충분하면 Generate, 없으면 Transform Query
        """
        filtered_documents = state["documents"]
        search_count = state.get("search_count", 0)
        
        if not filtered_documents:
            # 문서가 없지만, 너무 많이 재검색했다면 그냥 포기하고 생성(또는 종료)
            if search_count > 1: # 최대 1회 재검색 허용
                logger.info("--- [Graph: Decision] 재검색 한도 초과 -> 강제 생성 ---")
                return "generate"
            
            logger.info("--- [Graph: Decision] 관련 문서 없음 -> 쿼리 변환 및 재검색 ---")
            return "transform_query"
        else:
            logger.info("--- [Graph: Decision] 문서 확보 완료 -> 답변 생성 ---")
            return "generate"

    async def run(self, question: str):
        """그래프 실행 진입점"""
        inputs = {"question": question, "search_count": 0}
        config = {"recursion_limit": 10} # 무한 루프 안전장치
        
        final_state = await self.app.ainvoke(inputs, config=config)
        return final_state["generation"]
