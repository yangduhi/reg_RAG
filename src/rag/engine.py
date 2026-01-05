import asyncio
import json
import re
from typing import Coroutine, List, Dict, Optional, Union, Any

from flashrank import Ranker, RerankRequest
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_google_genai import ChatGoogleGenerativeAI

from src.core.config import settings
from src.core.logging import logger
from src.ingestion.pipeline import IngestionPipeline
from src.rag.vectorstore import VectorStoreManager


class DummyRetriever(BaseRetriever):
    """
    벡터 데이터베이스가 비어있을 때 시스템 충돌을 방지하기 위한 더미 검색기(Retriever) 구현체입니다.

    초기화 실패나 데이터 부재 시 예외를 발생시키는 대신 빈 리스트를 반환하여
    RAG 파이프라인이 안전하게 동작하도록 보장하는 안전장치 역할을 합니다.
    """

    def _get_relevant_documents(
        self, query: str, *, run_manager: Any = None
    ) -> List[Document]:
        """
        동기 방식 문서 검색 구현 (항상 빈 리스트 반환).

        Args:
            query (str): 검색 쿼리.
            run_manager (Any, optional): 실행 콜백 관리자.

        Returns:
            List[Document]: 항상 빈 리스트를 반환합니다.
        """
        return []

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: Any = None
    ) -> List[Document]:
        """
        비동기 방식 문서 검색 구현 (항상 빈 리스트 반환).

        Args:
            query (str): 검색 쿼리.
            run_manager (Any, optional): 실행 콜백 관리자.

        Returns:
            List[Document]: 항상 빈 리스트를 반환합니다.
        """
        return []


class RAGEngine:
    """
    핵심 RAG(Retrieval-Augmented Generation, 검색 증강 생성) 엔진 클래스입니다.

    이 클래스는 사용자 인터페이스(UI)와 비즈니스 로직을 연결하는 파사드(Facade) 역할을 수행합니다.
    다음과 같은 전체 RAG 파이프라인을 조율합니다:
    1. 쿼리 처리 및 변환 (Query Transformation)
    2. 하이브리드 검색 (벡터 검색 + 키워드 검색)
    3. 재순위화 (Reranking, Cross-encoder/FlashRank 활용)
    4. 답변 생성 (LLM 활용)

    Attributes:
        llm (ChatGoogleGenerativeAI): 언어 모델 인스턴스 (Gemini).
        vstore_manager (VectorStoreManager): 벡터 데이터베이스 관리자.
        metadata_cache (Dict[str, dict]): 규정 메타데이터의 인메모리 캐시.
        reranker (Optional[Ranker]): 재순위화 모델 인스턴스 (FlashRank).
        bm25_retriever (Optional[BM25Retriever]): 키워드 기반 검색기 인스턴스.
        is_initialized (bool): 검색기 초기화 완료 여부 플래그.
    """

    def __init__(self) -> None:
        """
        RAGEngine을 초기화합니다.

        LLM, 벡터 저장소 관리자, 메타데이터 캐시를 설정합니다.
        검색기(BM25, Reranker)의 비동기 초기화 작업을 시작합니다.
        """
        # 1. LLM 초기화 (Gemini)
        self.llm = ChatGoogleGenerativeAI(
            model=settings.LLM_MODEL_NAME,
            google_api_key=settings.GOOGLE_API_KEY,
            temperature=settings.LLM_TEMPERATURE,
        )

        # 2. 벡터 DB 매니저 연결
        self.vstore_manager = VectorStoreManager()

        # 3. 메타데이터 캐시 로드
        self.metadata_cache = self._load_all_metadata()

        # 4. 하이브리드 검색 컴포넌트 초기화
        self.reranker: Optional[Ranker] = None
        self.bm25_retriever: Optional[BM25Retriever] = None
        self.is_initialized: bool = False
        self.initialization_lock = asyncio.Lock()
        
        # 백그라운드 초기화 작업 시작
        self.initialization_task: Coroutine = self._initialize_retrievers()

    async def _initialize_retrievers(self) -> None:
        """
        검색 컴포넌트(BM25, Reranker)를 비동기적으로 초기화합니다.

        벡터 데이터베이스에 문서가 존재하는지 확인합니다. 데이터가 있는 경우,
        Non-blocking 스레드에서 BM25 인덱스를 생성하고 Reranker 모델을 로드합니다.
        
        Raises:
            Exception: 초기화 중 발생하는 모든 오류를 로그로 기록합니다.
        """
        try:
            logger.info("⏳ 검색기(Retriever) 초기화 중...")
            # 블로킹 DB 호출을 별도 스레드에서 실행
            all_docs = await asyncio.to_thread(self.vstore_manager.get_all_documents)

            if all_docs:
                # BM25 검색기 초기화 (Non-blocking)
                logger.info("🛠️ BM25 검색기 인덱싱 시작...")
                self.bm25_retriever = await asyncio.to_thread(
                    BM25Retriever.from_documents, all_docs
                )
                logger.info(
                    f"✅ BM25 인덱싱 완료 (문서 수: {len(all_docs)}개)"
                )

                # Reranker 초기화 (Non-blocking)
                logger.info(f"🚀 Reranker 모델 초기화 중: {settings.RERANKER_MODEL}")
                self.reranker = await asyncio.to_thread(
                    Ranker, model_name=settings.RERANKER_MODEL, cache_dir="/tmp/flashrank_cache"
                )
                logger.info("✅ Reranker 초기화 완료.")
                self.is_initialized = True
            else:
                logger.warning("⚠️ DB가 비어있습니다. 검색기를 초기화할 수 없습니다.")
                self.bm25_retriever = None
                self.reranker = None

        except Exception as e:
            logger.error(f"❌ 검색기 초기화 실패: {e}")
            self.bm25_retriever = None
            self.reranker = None
    
    @property
    def vector_store(self):
        """내부 VectorStore 인스턴스에 접근합니다."""
        return self.vstore_manager.db

    def get_retrievers(
        self, k: Optional[int] = None, use_mmr: bool = False, filter_std: str = "All"
    ) -> List[BaseRetriever]:
        """
        하이브리드 검색에 사용할 검색기 목록을 생성합니다.

        Args:
            k (Optional[int]): 검색할 문서 수. 기본값은 settings.RETRIEVER_K.
            use_mmr (bool): 다양성 확보를 위한 MMR(Maximal Marginal Relevance) 사용 여부.
            filter_std (str): 필터링할 규정 ID (예: "FMVSS 108"). 기본값은 "All".

        Returns:
            List[BaseRetriever]: [BM25Retriever, VectorRetriever] 리스트를 반환합니다.
        """
        _k = k if k is not None else settings.RETRIEVER_K

        if not self.bm25_retriever:
            logger.error("❌ 검색기가 준비되지 않았습니다.")
            return [DummyRetriever()]

        # 1. 벡터 검색 설정 (후보군 확장)
        # 재순위화(Reranking)를 위해 최종 개수보다 더 많은 후보를 1차로 검색합니다.
        search_type = "mmr" if use_mmr else "similarity"
        candidate_k = _k * 3

        search_kwargs = {"k": candidate_k}
        if use_mmr:
            search_kwargs["fetch_k"] = candidate_k * 2

        if filter_std and filter_std != "All":
            search_kwargs["filter"] = {"standard_id": filter_std}

        vector_retriever = self.vstore_manager.db.as_retriever(
            search_type=search_type, search_kwargs=search_kwargs
        )

        return [self.bm25_retriever, vector_retriever]

    def get_ensemble_retriever(
        self, k: Optional[int] = None, use_mmr: bool = False, filter_std: str = "All"
    ) -> BaseRetriever:
        """
        사전 설정된 앙상블 검색기(EnsembleRetriever, BM25 + Vector)를 생성합니다.

        Args:
            k (Optional[int]): 검색할 문서 수.
            use_mmr (bool): MMR 사용 여부.
            filter_std (str): 필터링할 규정 ID.

        Returns:
            BaseRetriever: 키워드와 의미 기반 검색이 결합된 EnsembleRetriever 인스턴스.
        """
        _k = k if k is not None else settings.RETRIEVER_K

        if not self.bm25_retriever:
            logger.error("❌ 검색기가 준비되지 않았습니다.")
            return DummyRetriever()

        # 1. 벡터 검색 설정
        search_type = "mmr" if use_mmr else "similarity"
        candidate_k = _k * 3

        search_kwargs = {"k": candidate_k}
        if use_mmr:
            search_kwargs["fetch_k"] = candidate_k * 2

        if filter_std and filter_std != "All":
            search_kwargs["filter"] = {"standard_id": filter_std}

        vector_retriever = self.vstore_manager.db.as_retriever(
            search_type=search_type, search_kwargs=search_kwargs
        )

        # 2. 앙상블 생성 (BM25 50%, Vector 50% 가중치)
        ensemble_retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, vector_retriever], weights=[0.5, 0.5]
        )

        return ensemble_retriever

    async def transform_query(self, original_query: str) -> str:
        """
        사용자 쿼리를 변환하여 검색 최적화를 수행합니다.

        이 메서드는 "쿼리 확장(Query Expansion)" 단계로 동작합니다.
        LLM을 사용하여 사용자의 질문(특히 한국어 기술 용어)을 영어 키워드로 변환하고,
        이를 원본 쿼리에 추가하여 검색 정확도를 높입니다.

        Args:
            original_query (str): 사용자의 원본 질문.

        Returns:
            str: 확장된 쿼리 문자열 (원본 + 영어 키워드).
        """
        try:
            prompt = ChatPromptTemplate.from_template(
                """
                Translate the technical terms in the user's question into English keywords.
                Output ONLY the English keywords separated by spaces.

                User Question: {question}
                English Keywords:"""
            )
            chain = prompt | self.llm | StrOutputParser()
            english_keywords = await chain.ainvoke({"question": original_query})
            
            # 원본 쿼리와 추출된 키워드 결합
            final_query = f"{original_query} {english_keywords.strip()}"
            logger.info(f"🔥 [쿼리 확장] 최종: '{final_query}'")
            return final_query
        except Exception as e:
            logger.error(f"쿼리 변환 오류: {e}")
            return original_query

    async def chat(self, user_question: str) -> str:
        """
        RAG 파이프라인의 메인 진입점입니다.

        실행 단계:
        1. 초기화 대기 (필요 시).
        2. 쿼리 변환 (번역/확장).
        3. 하이브리드 검색 수행 (BM25 + Vector).
        4. 검색 결과 중복 제거.
        5. 재순위화 (FlashRank Cross-Encoder 사용).
        6. 답변 생성 (LLM, 출처 포함).

        Args:
            user_question (str): 사용자의 질문.

        Returns:
            str: LLM이 생성한 답변.
        """
        async with self.initialization_lock:
            if not self.is_initialized:
                await self.initialization_task

        # 1. 쿼리 변환
        optimized_query = await self.transform_query(user_question)
        logger.info(f"🔍 [검색] 원본: '{user_question}' -> 변환됨: '{optimized_query}'")

        # 2. 하이브리드 검색
        retrievers = self.get_retrievers(use_mmr=True)
        
        # 모든 검색기에서 병렬로 문서 검색
        tasks = [retriever.ainvoke(optimized_query) for retriever in retrievers]
        results = await asyncio.gather(*tasks)
        
        # 결과 통합 및 중복 제거
        unique_docs: Dict[str, Document] = {}
        for doc_list in results:
            for doc in doc_list:
                if doc.page_content not in unique_docs:
                    unique_docs[doc.page_content] = doc
        
        retrieved_docs = list(unique_docs.values())

        if not retrieved_docs:
            return "죄송합니다. 관련 규정을 찾을 수 없습니다. 검색어를 변경하거나 데이터베이스를 확인해 주세요."

        # 3. FlashRank를 이용한 재순위화 (Reranking)
        if self.reranker:
            passages = [
                {"id": i, "text": doc.page_content, "meta": doc.metadata}
                for i, doc in enumerate(retrieved_docs)
            ]
            
            logger.info(f"🧠 {len(passages)}개 문서에 대해 FlashRank 재순위화 수행 중...")
            rerank_request = RerankRequest(query=user_question, passages=passages)
            reranked_passages = await asyncio.to_thread(
                self.reranker.rerank, rerank_request
            )
            
            # LangChain Document 형식으로 변환 및 상위 K개 추출
            reranked_docs = [
                Document(page_content=p["text"], metadata=p["meta"])
                for p in reranked_passages
            ][:settings.RETRIEVER_K]
            logger.info(f"✨ 재순위화 완료. 상위 {len(reranked_docs)}개 문서 선택됨.")
            final_docs = reranked_docs
        else:
            logger.warning("⚠️ Reranker를 사용할 수 없습니다. 재순위화 단계를 건너뜁니다.")
            final_docs = retrieved_docs[:settings.RETRIEVER_K]


        if not final_docs:
            return "죄송합니다. 재순위화 후 적절한 관련 규정을 찾지 못했습니다."


        # 4. 답변 생성 (프롬프트 엔지니어링)
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

        def format_docs(docs: List[Document]) -> str:
            return "\n\n".join(
                f"--- Document Start ({d.metadata.get('standard_id', 'Unknown')}) ---\n{d.page_content}\n--- Document End ---"
                for d in docs
            )

        chain = ChatPromptTemplate.from_template(template) | self.llm | StrOutputParser()
        context_text = format_docs(final_docs)
        response = await chain.ainvoke(
            {"context": context_text, "question": user_question}
        )

        return response

    async def run_pipeline(self, force_refresh: bool = False) -> str:
        """
        데이터 수집 파이프라인을 수동으로 실행합니다.

        Args:
            force_refresh (bool): True일 경우, 변경 사항과 관계없이 모든 파일을 다시 처리합니다.

        Returns:
            str: 완료 상태 메시지.

        Raises:
            Exception: 파이프라인 실행 중 오류 발생 시 예외를 전파합니다.
        """
        try:
            logger.info("🔄 데이터 파이프라인 실행 시작...")
            pipeline = IngestionPipeline()
            await pipeline.run(force_refresh=force_refresh)

            self.metadata_cache = self._load_all_metadata()
            logger.info("🔄 데이터 변경 감지: 검색 엔진을 다시 로드합니다.")
            # 새 데이터를 반영하기 위해 검색기 재초기화
            self.initialization_task = self._initialize_retrievers()
            await self.initialization_task

            return "✅ 데이터 처리 및 엔진 업데이트 완료!"
        except Exception as e:
            logger.error(f"파이프라인 오류: {e}", exc_info=True)
            raise

    def get_available_standards(self) -> List[str]:
        """
        사용 가능한 규정 ID 목록을 정렬하여 반환합니다 (예: "108", "201").

        Returns:
            List[str]: 정렬된 ID 리스트.
        """
        ids = list(self.metadata_cache.keys())

        def sort_key(k: str) -> int:
            nums = re.findall(r"(\d+)", str(k))
            return int(nums[0]) if nums else 9999

        return sorted(list(set(ids)), key=sort_key)

    def get_metadata_title(self, standard_id: str) -> str:
        """주어진 규정 ID에 해당하는 제목을 반환합니다."""
        return self.metadata_cache.get(str(standard_id), {}).get("title", "")

    def get_web_url(self, standard_id: str) -> Optional[str]:
        """주어진 규정 ID의 원문 웹 URL을 반환합니다."""
        return self.metadata_cache.get(str(standard_id), {}).get("url")

    def _load_all_metadata(self) -> Dict[str, dict]:
        """
        여러 JSON 소스에서 메타데이터를 로드하고 병합합니다.

        Returns:
            Dict[str, dict]: 규정 ID를 키로 하는 메타데이터 딕셔너리.
        """
        merged = {}
        candidates = [
            settings.DATA_DIR / "metadata_kmvss.json",
            settings.DATA_DIR / "crawled_metadata_ece.json",
            settings.DATA_DIR / "crawled_metadata.json",
            settings.METADATA_FILE,
        ]
        for path in candidates:
            if path.exists():
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            for item in data:
                                key = str(item.get("id", item.get("num", "")))
                                if key:
                                    merged[key] = {
                                        "title": item.get("title", ""),
                                        "url": item.get("web_url")
                                        or item.get("source_url"),
                                    }
                        elif isinstance(data, dict):
                            merged.update(data)
                except Exception:
                    # 견고성을 위해 개별 메타데이터 파일 오류는 무시합니다.
                    pass
        return merged
