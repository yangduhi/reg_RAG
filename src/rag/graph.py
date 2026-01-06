from typing import List, TypedDict, Literal, Optional, Any

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, StateGraph

from src.core.logging import logger
from src.core.config import settings

# --- State Definition (상태 정의) ---
class GraphState(TypedDict):
    """
    RAG 그래프의 상태를 저장하는 딕셔너리입니다.
    """
    question: str               # 사용자 질문 (현재 턴)
    chat_history: List[dict]    # 이전 대화 기록 (Streamlit 형식)
    multi_queries: List[str]    # 분석을 통해 생성된 멀티 쿼리 리스트
    documents: List[Document]   # 검색 및 재순위화된 문서 리스트
    generation: str             # 최종 생성된 답변
    search_count: int           # 재검색 시도 횟수

# --- Data Models for LLM Structure Output (구조화된 출력 모델) ---
class GradeDocuments(BaseModel):
    """
    문서 적합성 평가(Grader)의 출력 스키마입니다.
    LLM이 문서를 평가할 때 'yes' 또는 'no'로 명확한 판단을 내리도록 강제합니다.
    """
    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

class RAGGraph:
    """
    [LangGraph 워크플로우 정의]
    RAG 파이프라인의 실행 흐름(Directed Cyclic Graph)을 정의합니다.
    
    [흐름 요약]
    1. transform_query: 사용자 질문 분석 및 다국어 쿼리 확장
    2. retrieve: 하이브리드 검색 수행 (BM25 + Vector)
    3. rerank: 언어별 특화 재순위화 (영어: Cross-Encoder, 한국어: 원본 유지)
    4. grade_documents: 문서 적합성 평가 및 필터링
    5. decide_to_generate: 재검색 여부 결정 (조건부 엣지)
    6. generate: 최종 답변 생성
    """
    def __init__(self, rag_engine):
        """
        RAGGraph 초기화
        Args:
            rag_engine: RAGEngine 인스턴스 (LLM, Retriever 등 핵심 리소스 접근용)
        """
        self.engine = rag_engine
        self.app = self._build_graph()

    def _build_graph(self):
        """LangGraph 워크플로우 정의 및 컴파일"""
        workflow = StateGraph(GraphState)

        # 노드 등록
        workflow.add_node("transform_query", self.transform_query)
        workflow.add_node("retrieve", self.retrieve)
        workflow.add_node("rerank", self.rerank)
        workflow.add_node("grade_documents", self.grade_documents)
        workflow.add_node("generate", self.generate)

        # 엣지 연결 (흐름 제어)
        workflow.set_entry_point("transform_query")
        workflow.add_edge("transform_query", "retrieve")
        workflow.add_edge("retrieve", "rerank")
        workflow.add_edge("rerank", "grade_documents")
        
        # 조건부 엣지: 문서 품질에 따른 분기
        workflow.add_conditional_edges(
            "grade_documents",
            self.decide_to_generate,
            {
                "transform_query": "transform_query", # 관련 문서 없으면 쿼리 재구성 후 재검색
                "generate": "generate",               # 관련 문서 있으면 답변 생성
            },
        )
        
        workflow.add_edge("generate", END)

        return workflow.compile()

    # --- Nodes ---

    async def transform_query(self, state: GraphState):
        """
        [Node 1: 질의 변환 및 확장] (Query Transformation)
        - 사용자 질문과 대화 맥락을 분석합니다.
        - 규정 검색에 최적화된 '다국어 쿼리'를 생성합니다. (Query Expansion)
        - 한국어 규정 검색을 위해 '별표/시험방법' 키워드를 추가하고, 영어 규정을 위해 전문용어를 번역합니다.
        """
        question = state["question"]
        chat_history = state.get("chat_history", [])
        search_count = state.get("search_count", 0)
        search_regions = state.get("search_regions", [])
        
        logger.info(f"--- [Graph: Transform] 쿼리 분석 시작 (시도: {search_count+1}) - Regions: {search_regions} ---")
        
        # 최근 5개 대화 요약
        history_str = "\n".join([f"{m['role']}: {m['content']}" for m in chat_history[-5:]]) if chat_history else "No history."
        
        # 지역 필터링 지침 생성
        region_instruction = ""
        if search_regions:
            region_instruction = f"FOCUS ONLY on the following regions: {', '.join(search_regions)}."
            if "KMVSS" not in search_regions:
                region_instruction += " DO NOT generate Korean queries for KMVSS."
            if "FMVSS" not in search_regions and "ECE" not in search_regions:
                region_instruction += " DO NOT generate English queries for FMVSS/ECE."
        
        prompt = ChatPromptTemplate.from_template(
            """
            You are an expert in Automotive Safety Regulations (FMVSS, KMVSS, ECE).
            Your task is to generate search queries to retrieve relevant documents.
            {region_instruction}

            **Conversation History:**
            {history}

            **User Question:** {question}
            
            **Instructions:**
            1. **Analyze:** specific topics (e.g., "head injury", "braking", "seatbelt").
            2. **Translate & Expand (Conditional):**
               - **Query 1 (Korean):** Optimized for KMVSS. Use specific Korean technical terms. **MANDATORY: Append "충격시험방법" (Test Procedure) and "HIC" (if head injury).** (Only if KMVSS is targeted)
               - **Query 2 (English):** Optimized for ECE/FMVSS. **TRANSLATE** technical terms. **CRITICAL: If "US/America" is mentioned, include "FMVSS". If "Europe" is mentioned, include "ECE".** (Only if ECE/FMVSS are targeted)
               - **Query 3 (Keyword/ID):** Use specific Regulation IDs (e.g., "ECE R127", "FMVSS 208", "KMVSS 102").
            
            **Output Format:**
            - Output ONE query per line.
            - NO numbering, NO labels.
            - Just the raw query text.
            """
        )
        
        chain = prompt | self.engine.llm | StrOutputParser()
        try:
            response = await chain.ainvoke({
                "history": history_str, 
                "question": question,
                "region_instruction": region_instruction
            })
            # 라인 단위 분리 및 빈 줄 제거
            queries = [q.strip() for q in response.split("\n") if q.strip()]
            # 3개 미만일 경우 원본 질문 추가
            if not queries:
                 queries = [question]
        except Exception as e:
            logger.error(f"Transform Query Error: {e}")
            queries = [question]
            
        # 원본 질문이 포함되지 않았으면 추가 (선택사항, 하지만 검색 범위 확장을 위해 좋음)
        if question not in queries:
            queries.append(question)
            
        logger.info(f"--- [Graph: Transform] 생성된 멀티 쿼리: {queries} ---")
        return {"multi_queries": queries, "search_count": search_count}

    async def retrieve(self, state: GraphState):
        """
        [Node 2: 문서 검색] (Retrieval)
        - 생성된 멀티 쿼리를 사용하여 병렬 검색을 수행합니다.
        - BM25(키워드)와 Vector(의미) 검색 결과를 결합합니다. (Ensemble)
        - 검색 후 사용자 설정에 따라 지역(Region) 필터링을 수행합니다.
        """
        queries = state["multi_queries"]
        search_regions = state.get("search_regions", [])
        
        logger.info(f"--- [Graph: Retrieve] {len(queries)}개 쿼리로 병렬 검색 중... ---")
        
        import asyncio
        retrievers = self.engine.get_retrievers(use_mmr=True)
        
        # 모든 쿼리에 대해 병렬 검색 실행
        tasks = []
        for q in queries:
            for retriever in retrievers:
                tasks.append(retriever.ainvoke(q))
        
        results = await asyncio.gather(*tasks)
        
        # 중복 제거 및 필터링
        unique_docs = {}
        for doc_list in results:
            for doc in doc_list:
                if len(doc.page_content) < 50: # 너무 짧은 정보 제외
                    continue
                
                # 지역 필터링 (선택된 지역만 포함)
                if search_regions:
                    std_id = str(doc.metadata.get("standard_id", "")).upper()
                    source = str(doc.metadata.get("source", "")).upper()
                    region_match = False
                    
                    if "FMVSS" in search_regions and ("FMVSS" in std_id or "571" in std_id or "US" in source):
                        region_match = True
                    if "KMVSS" in search_regions and ("KMVSS" in std_id or "KR" in source or "KOREA" in source):
                        region_match = True
                    if "ECE" in search_regions and ("ECE" in std_id or "R" in std_id or "EU" in source):
                        region_match = True
                        
                    if not region_match:
                        continue

                if doc.page_content not in unique_docs:
                    unique_docs[doc.page_content] = doc
        
        documents = list(unique_docs.values())
        logger.info(f"--- [Graph: Retrieve] 총 검색 결과: {len(documents)}개 (Region Filtered) ---")
        return {"documents": documents}

    async def rerank(self, state: GraphState):
        """
        [Node 3: 언어별 분리 재순위화] (Language-Specific Reranking)
        [핵심 알고리즘]
        - 문제점: 일반적인 Reranker(ms-marco 등)는 영어 중심이라 한국어 문서 점수를 낮게 평가하여 탈락시킴.
        - 해결책: 
            1. 문서를 한국어/영어로 분류합니다.
            2. 영어 문서는 Reranker(FlashRank)를 사용하여 정밀하게 재순위화합니다.
            3. 한국어 문서는 Reranker를 거치지 않고, 1차 검색(Retrieval)의 순위를 그대로 신뢰합니다. (Skip)
            4. 두 리스트를 1:1 비율로 교차 병합(Interleaving)하여 상위권에 골고루 분포시킵니다.
        """
        question = state["question"]
        documents = state["documents"]
        multi_queries = state.get("multi_queries", [])
        
        if not documents or not self.engine.reranker:
            return {"documents": documents}

        logger.info(f"--- [Graph: Rerank] {len(documents)}개 문서 재순위화 시작 ---")
        
        # 1. 문서 및 쿼리 분류
        korean_docs = []
        english_docs = []
        
        for doc in documents:
            # 메타데이터 기반 분류 (보조적으로 텍스트 분석)
            std_id = str(doc.metadata.get("standard_id", "")).upper()
            source = str(doc.metadata.get("source", "")).upper()
            content = doc.page_content
            
            is_korean = "KMVSS" in std_id or "KMVSS" in source or any(ord(c) > 12592 for c in content[:50])
            
            if is_korean:
                korean_docs.append(doc)
            else:
                english_docs.append(doc)
        
        # 쿼리 매핑 (transform_query의 순서: 1.Korean, 2.English, 3.Keyword)
        korean_query = multi_queries[0] if len(multi_queries) > 0 else question
        english_query = multi_queries[1] if len(multi_queries) > 1 else (question if not korean_query else multi_queries[0])
        
        # 2. 개별 재순위화 수행 (한국어는 원본 순서 유지, 영어는 재순위화)
        from flashrank import RerankRequest
        import asyncio
        
        # 한국어 문서는 Reranker(ms-marco)가 한국어를 잘 이해하지 못하므로 
        # 검색 단계의 점수(BM25/Vector)를 신뢰하여 원본 순서를 유지합니다.
        reranked_ko = [{"text": d.page_content, "meta": d.metadata} for d in korean_docs]

        # 영어 문서는 Reranker를 사용하여 정확도 향상
        reranked_en = []
        if english_docs:
            req_en = RerankRequest(
                query=english_query,
                passages=[{"id": i, "text": d.page_content, "meta": d.metadata} for i, d in enumerate(english_docs)]
            )
            # 비동기 실행
            reranked_results = await asyncio.to_thread(self.engine.reranker.rerank, req_en)
            reranked_en = reranked_results
            
        # 3. 결과 병합 (Interleaving)
        final_docs = []
        
        # 문서 객체로 변환
        ko_objs = [Document(page_content=r["text"], metadata=r["meta"]) for r in reranked_ko]
        en_objs = [Document(page_content=r["text"], metadata=r["meta"]) for r in reranked_en]
        
        # 최대 20개까지 1:1로 섞어서 추가
        import itertools
        for k, e in itertools.zip_longest(ko_objs[:10], en_objs[:10]):
            if k: final_docs.append(k)
            if e: final_docs.append(e)
            
        logger.info(f"--- [Graph: Rerank] 완료 (상위 {len(final_docs)}개 선택 - Ko:{len(ko_objs)}, En:{len(en_objs)}) ---")
        return {"documents": final_docs}
        final_docs = []
        
        # 문서 객체로 변환
        ko_objs = [Document(page_content=r["text"], metadata=r["meta"]) for r in reranked_ko]
        en_objs = [Document(page_content=r["text"], metadata=r["meta"]) for r in reranked_en]
        
        # 최대 RETRIEVER_K개까지 1:1로 섞어서 추가
        import itertools
        target_k = settings.RETRIEVER_K
        
        # Interleaving: 각 리스트에서 번갈아가며 선택
        # ko_objs, en_objs는 이미 정렬되어 있다고 가정
        for k, e in itertools.zip_longest(ko_objs, en_objs):
            if k and len(final_docs) < target_k: final_docs.append(k)
            if e and len(final_docs) < target_k: final_docs.append(e)
            
            if len(final_docs) >= target_k:
                break
            
        logger.info(f"--- [Graph: Rerank] 완료 (상위 {len(final_docs)}개 선택 - Ko:{len(ko_objs)}, En:{len(en_objs)}) ---")
        return {"documents": final_docs}

    async def grade_documents(self, state: GraphState):
        """
        [Node 4: 문서 적합성 평가] (Relevance Grading)
        - 검색된 문서가 사용자의 질문에 실질적으로 도움이 되는지 LLM을 통해 검증합니다.
        - 사용자 UI에서 설정한 '검색 정확도(Similarity Threshold)'에 따라 평가 기준(엄격도)을 조절합니다.
        - 다국어 호환성 고려: 질문이 한국어고 문서가 영어라도 내용이 맞으면 통과시킵니다.
        """
        question = state["question"]
        documents = state["documents"]
        similarity_threshold = state.get("similarity_threshold", 0.5)
        
        # 설정된 개수만큼 평가 (이미 rerank에서 잘렸겠지만 안전장치)
        k = settings.RETRIEVER_K
        docs_to_grade = documents[:k]
        
        logger.info(f"--- [Graph: Grade] 상위 {len(docs_to_grade)}개 문서 평가 시작 (Threshold: {similarity_threshold}) ---")
        
        llm_with_tool = self.engine.llm.with_structured_output(GradeDocuments)
        
        # 임계값에 따른 엄격도 조정 지침
        strictness = "Be very lenient."
        if similarity_threshold > 0.7:
            strictness = "Be STRICT. The document must be directly relevant."
        elif similarity_threshold < 0.3:
            strictness = "Be EXTREMELY lenient. If it's remotely related, say 'yes'."
            
        prompt = ChatPromptTemplate.from_template(
            """You are a grader assessing relevance of a retrieved document to a user question.
            
            **Context:** The user is asking about Automotive Regulations (KMVSS, FMVSS, ECE).
            **Scenario:** The user often asks in **Korean**, but the answer requires **English documents** (ECE/FMVSS).
            
            Document:
            {context}
            
            Question: {question}
            
            **Grading Rules (CRITICAL):**
            1. **Cross-Language Relevance:** If the Question is in Korean (e.g., "보행자" - Pedestrian) and the Document is in English (e.g., "Pedestrian Safety"), it is **RELEVANT**. Grade 'yes'.
            2. **Keyword Matching:** Look for technical terms (Head, Leg, Impact, Speed, HIC, WAD). If they match (even if translated), Grade 'yes'.
            3. **Be Lenient with Annexes:** If the document is an **Annex (별표)** or **Test Procedure (시험방법)**, it contains the detailed CRITERIA. Grade 'yes' even if it looks like just technical steps.
            4. **Be Lenient:** If the document belongs to the correct Standard (ECE/FMVSS/KMVSS) and Topic, Grade 'yes'. Do not require exact sentence matching.
            
            **Strictness Level:** {strictness}
            
            Give binary score 'yes' or 'no'."""
        )
        chain = prompt | llm_with_tool
        
        import asyncio
        async def grade_doc(doc):
            try:
                res = await chain.ainvoke({
                    "question": question, 
                    "context": doc.page_content,
                    "strictness": strictness
                })
                return doc if res.binary_score == "yes" else None
            except:
                return doc

        graded_results = await asyncio.gather(*[grade_doc(d) for d in docs_to_grade])
        filtered_docs = [d for d in graded_results if d is not None]
        
        logger.info(f"--- [Graph: Grade] 관련 문서 수: {len(filtered_docs)} / {len(docs_to_grade)} ---")
        return {"documents": filtered_docs}

    async def generate(self, state: GraphState):
        """
        [Node 5: 최종 답변 생성] (Answer Generation)
        - 검증된 문서들을 문맥(Context)으로 하여 사용자 질문에 답변합니다.
        - 규정 비교를 위해 표 형식을 권장하고, 출처 표기를 강제합니다.
        """
        logger.info("--- [Graph: Generate] 답변 생성 중... ---")
        question = state["question"]
        documents = state["documents"]
        chat_history = state.get("chat_history", [])
        
        history_str = "\n".join([f"{m['role']}: {m['content']}" for m in chat_history[-5:]]) if chat_history else "None"
        
        # 컨텍스트 정보량 최대화
        docs_to_use = documents[:10]
        context_text = "\n\n".join([
            f"--- Document ({d.metadata.get('standard_id', 'N/A')}, Region: {d.metadata.get('region', 'Unknown')}) ---\n{d.page_content}" 
            for d in docs_to_use
        ])

        template = """
        You are a senior expert in Automotive Safety Regulations (FMVSS, KMVSS, ECE).
        Answer the [Question] based on the provided [Context] (which may include English and Korean documents).

        [Rules]
        1. **Multi-source:** You MUST use information from both Korean (KMVSS) and Foreign (ECE/FMVSS) documents if available.
        2. **Translation:** If the context is in English, translate the relevant parts into Korean for the final answer.
        3. **Accuracy:** Use specific numbers and values from the context.
        4. **Citations:** Cite the Regulation ID (e.g., [Source: ECE R127]) for every key point.
        5. **Structure:** If comparing multiple regions, use a Markdown table.
        6. **Language:** Respond in polite Korean.

        [Conversation History]
        {history}

        [Context]
        {context}

        [Question]
        {question}

        [Answer]
        """
        
        prompt = ChatPromptTemplate.from_template(template)
        # 답변 생성 단계에서는 고성능 모델(gemini-2.5-pro) 사용
        chain = prompt | self.engine.llm_smart | StrOutputParser()
        
        response = await chain.ainvoke({
            "history": history_str, 
            "context": context_text, 
            "question": question
        })
        
        return {"generation": response}

    def decide_to_generate(self, state: GraphState) -> Literal["transform_query", "generate"]:
        """
        [Edge Condition] 재검색 여부 결정
        - 검색된 관련 문서가 하나도 없으면 쿼리를 수정하여 재검색을 시도합니다. (최대 2회)
        - 문서가 있으면 바로 생성 단계로 넘어갑니다.
        """
        if not state["documents"] and state.get("search_count", 0) < 2:
            logger.info("--- [Graph: Decision] No relevant docs found. Retrying... ---")
            return "transform_query"
        return "generate"

    async def run(self, question: str, chat_history: List[dict] = None, search_regions: List[str] = None, similarity_threshold: float = 0.5):
        """
        [그래프 실행 진입점]
        - 초기 상태(Inputs)를 설정하고 그래프를 실행합니다.
        - UI로부터 받은 설정값(search_regions, similarity_threshold)을 상태에 주입합니다.
        """
        inputs = {
            "question": question, 
            "chat_history": chat_history or [],
            "search_count": 0,
            "search_regions": search_regions,
            "similarity_threshold": similarity_threshold
        }
        # 초기화 상태 보장을 위해 search_count는 여기서 초기화
        final_state = await self.app.ainvoke(inputs, config={"recursion_limit": 20})
        return {
            "generation": final_state.get("generation", "답변을 생성하지 못했습니다."),
            "documents": final_state.get("documents", []),
        }
