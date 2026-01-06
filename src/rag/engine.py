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
from src.rag.graph import RAGGraph


class DummyRetriever(BaseRetriever):
    """
    [안전 장치] 더미 검색기 (Dummy Retriever)
    - 벡터 데이터베이스가 비어있거나 초기화 실패 시 시스템 충돌(Crash)을 방지합니다.
    - 예외를 발생시키는 대신 빈 리스트를 반환하여 RAG 파이프라인이 안전하게 종료되도록 합니다.
    """

    def _get_relevant_documents(
        self, query: str, *, run_manager: Any = None
    ) -> List[Document]:
        """동기 호출: 항상 빈 리스트 반환"""
        return []

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: Any = None
    ) -> List[Document]:
        """비동기 호출: 항상 빈 리스트 반환"""
        return []


class RAGEngine:
    """
    [RAG 엔진 코어] (Retrieval-Augmented Generation Engine)
    
    이 클래스는 시스템의 두뇌 역할을 하며, UI와 비즈니스 로직(검색 및 생성)을 연결하는 파사드(Facade)입니다.
    
    [주요 기능]
    1. LLM 및 벡터 저장소 초기화
    2. 하이브리드 검색기(BM25 + Vector) 구성 및 관리
    3. LangGraph 워크플로우 실행 (Chat)
    4. 데이터 파이프라인(수집/가공) 제어
    
    Attributes:
        llm (ChatGoogleGenerativeAI): 답변 생성을 위한 Gemini 모델 인스턴스.
        vstore_manager (VectorStoreManager): 벡터 데이터베이스(ChromaDB) 접근 관리자.
        metadata_cache (Dict): 규정 ID, 제목, URL 등의 메타데이터를 인메모리에 캐싱하여 빠른 접근 지원.
        reranker (Ranker): 검색 결과의 정확도를 높이기 위한 재순위화(Reranking) 모델.
        bm25_retriever (BM25Retriever): 키워드 기반 검색기 (정확한 용어 매칭용).
        graph (RAGGraph): LangGraph 기반의 검색-생성 워크플로우 정의.
    """

    def __init__(self) -> None:
        """
        엔진 초기화
        - 동기적으로 LLM, DB 매니저를 설정하고,
        - 비동기적으로 검색기(Retriever) 초기화 태스크를 백그라운드에서 시작합니다. (UI 로딩 지연 방지)
        """
        # 1. LLM 초기화 (Google Gemini API 사용)
        # settings.LLM_MODEL_NAME (예: gemini-2.0-flash) 모델 사용 -> 일반/고속 작업용
        self.llm = ChatGoogleGenerativeAI(
            model=settings.LLM_MODEL_NAME,
            google_api_key=settings.GOOGLE_API_KEY,
            temperature=settings.LLM_TEMPERATURE,
        )
        
        # settings.LLM_MODEL_SMART (예: gemini-2.5-pro) 모델 사용 -> 고성능 답변 생성용
        self.llm_smart = ChatGoogleGenerativeAI(
            model=settings.LLM_MODEL_SMART,
            google_api_key=settings.GOOGLE_API_KEY,
            temperature=settings.LLM_TEMPERATURE,
        )

        # 2. 벡터 DB 매니저 연결 (데이터 저장소)
        self.vstore_manager = VectorStoreManager()

        # 3. 메타데이터 캐시 로드 (규정 정보)
        self.metadata_cache = self._load_all_metadata()

        # 4. 하이브리드 검색 컴포넌트 (초기값 None, 비동기 로딩)
        self.reranker: Optional[Ranker] = None
        self.bm25_retriever: Optional[BM25Retriever] = None
        self.is_initialized: bool = False
        self.initialization_lock = asyncio.Lock()
        
        # 5. LangGraph 워크플로우 생성 (self를 전달하여 엔진의 리소스 공유)
        self.graph = RAGGraph(self)
        
        # [성능 최적화] 검색기 초기화는 시간이 걸리므로(인덱싱 등) 백그라운드 태스크로 실행
        self.initialization_task = asyncio.create_task(self._initialize_retrievers())

    async def _initialize_retrievers(self) -> None:
        """
        [비동기 초기화] 검색기 구성 (BM25 & Reranker)
        
        - 벡터 DB의 모든 문서를 로드하여 BM25(TF-IDF) 인덱스를 생성합니다.
        - Reranker 모델을 메모리에 로드합니다.
        - 이 과정이 완료되어야 정상적인 검색이 가능합니다.
        """
        try:
            logger.info("⏳ 검색기(Retriever) 초기화 및 인덱싱 시작...")
            # DB I/O는 블로킹 작업이므로 별도 스레드에서 실행하여 이벤트 루프 차단 방지
            all_docs = await asyncio.to_thread(self.vstore_manager.get_all_documents)

            if all_docs:
                # 1. BM25 검색기 초기화 (키워드 검색용 역색인 생성)
                logger.info(f"🛠️ BM25 검색기 인덱싱 중... (문서 수: {len(all_docs)}개)")
                self.bm25_retriever = await asyncio.to_thread(
                    BM25Retriever.from_documents, all_docs
                )
                logger.info("✅ BM25 인덱싱 완료.")

                # 2. Reranker 모델 로드 (재순위화용 Cross-Encoder)
                # FlashRank 라이브러리 사용 (경량화된 BERT 모델)
                logger.info(f"🚀 Reranker 모델 로딩: {settings.RERANKER_MODEL}")
                self.reranker = await asyncio.to_thread(
                    Ranker, model_name=settings.RERANKER_MODEL, cache_dir="/tmp/flashrank_cache"
                )
                logger.info("✅ Reranker 초기화 완료.")
                self.is_initialized = True
            else:
                logger.warning("⚠️ DB가 비어있습니다. 검색기를 초기화할 수 없습니다. (데이터 수집 필요)")
                self.bm25_retriever = None
                self.reranker = None

        except Exception as e:
            logger.error(f"❌ 검색기 초기화 실패: {e}")
            self.bm25_retriever = None
            self.reranker = None
    
    @property
    def vector_store(self):
        """내부 VectorStore(ChromaDB) 객체 접근자"""
        return self.vstore_manager.db

    def get_retrievers(
        self, k: Optional[int] = None, use_mmr: bool = False, filter_std: str = "All"
    ) -> List[BaseRetriever]:
        """
        [검색기 팩토리] 상황에 맞는 검색기 목록 반환
        
        하이브리드 검색(Hybrid Search)을 위해 BM25와 Vector Retriever를 리스트로 반환합니다.
        LangGraph의 'retrieve' 노드에서 이를 병렬로 실행합니다.

        Args:
            k (int): 검색할 문서 수 (기본값: 설정파일의 RETRIEVER_K = 25)
            use_mmr (bool): MMR(Maximal Marginal Relevance) 알고리즘 사용 여부.
                            (유사하면서도 다양한 내용을 찾기 위해 True 권장)
            filter_std (str): 특정 규정 ID로 필터링할 경우 사용 (현재는 "All"로 전체 검색)

        Returns:
            List[BaseRetriever]: [BM25Retriever, VectorRetriever]
        """
        _k = k if k is not None else settings.RETRIEVER_K

        if not self.bm25_retriever:
            logger.error("❌ 검색기가 준비되지 않았습니다.")
            return [DummyRetriever()]

        # 1. 벡터 검색 설정 (Semantic Search)
        # Reranking 효과를 극대화하기 위해 최종 필요한 개수(K)보다 더 많은 후보군(3배수)을 1차로 검색합니다.
        search_type = "mmr" if use_mmr else "similarity"
        candidate_k = _k * 3

        search_kwargs = {"k": candidate_k}
        if use_mmr:
            search_kwargs["fetch_k"] = candidate_k * 2 # MMR 후보풀 크기

        # (선택사항) 메타데이터 필터링
        if filter_std and filter_std != "All":
            search_kwargs["filter"] = {"standard_id": filter_std}

        vector_retriever = self.vstore_manager.db.as_retriever(
            search_type=search_type, search_kwargs=search_kwargs
        )

        return [self.bm25_retriever, vector_retriever]

    # ... (get_ensemble_retriever는 현재 사용되지 않지만 하위 호환성을 위해 유지) ...

    async def chat(
        self, 
        user_question: str, 
        chat_history: List[Dict] = None,
        search_regions: List[str] = None,
        similarity_threshold: float = 0.5
    ) -> Dict:
        """
        [메인 실행 함수] 사용자 질문 처리 및 답변 생성
        
        UI에서 호출되는 진입점으로, 다음 과정을 수행합니다:
        1. 엔진 초기화 대기 (검색기 준비 확인)
        2. LangGraph 워크플로우(Graph) 실행
        3. 결과(답변 및 근거 문서) 반환

        Args:
            user_question (str): 사용자 질문.
            chat_history (List[Dict]): 이전 대화 내역 (멀티턴 대화 지원용, 현재는 주로 단일턴).
            search_regions (List[str]): 검색할 지역 필터 (예: ["FMVSS", "KMVSS"]). UI 체크박스와 연동.
            similarity_threshold (float): 검색 민감도(정확도) 설정. UI 슬라이더와 연동.

        Returns:
            Dict: {"generation": 답변 텍스트, "documents": [참고 문서 리스트]}
        """
        # 검색기 초기화가 완료될 때까지 대기 (Thread-safe)
        async with self.initialization_lock:
            if not self.is_initialized:
                await self.initialization_task

        # LangGraph 워크플로우 실행
        try:
            return await self.graph.run(
                user_question, 
                chat_history=chat_history,
                search_regions=search_regions,
                similarity_threshold=similarity_threshold
            )
        except Exception as e:
            logger.error(f"LangGraph 실행 중 치명적 오류: {e}", exc_info=True)
            return {
                "generation": "죄송합니다. 시스템 오류가 발생하여 답변을 생성할 수 없습니다. 관리자에게 문의해주세요.",
                "documents": [],
            }

    async def run_pipeline(self, force_refresh: bool = False) -> str:
        """
        [데이터 파이프라인 트리거] 데이터 수집 및 가공 프로세스 실행
        
        UI의 '데이터베이스 관리' 버튼을 통해 호출됩니다.
        1. IngestionPipeline 실행 (크롤링 -> 파싱 -> 청킹 -> 임베딩 -> 저장)
        2. 변경 사항이 있을 경우 검색기(BM25/Reranker) 재초기화

        Args:
            force_refresh (bool): True일 경우 기존 DB를 삭제하고 처음부터 다시 구축.
        """
        try:
            logger.info("🔄 데이터 파이프라인 실행 시작...")
            pipeline = IngestionPipeline()
            await pipeline.run(force_refresh=force_refresh)

            # 메타데이터 캐시 갱신
            self.metadata_cache = self._load_all_metadata()
            
            logger.info("🔄 데이터 변경 감지: 검색 엔진을 다시 로드합니다.")
            # 새 데이터를 반영하기 위해 검색기 재초기화 태스크 시작
            self.initialization_task = asyncio.create_task(self._initialize_retrievers())
            await self.initialization_task

            return "✅ 데이터 처리 및 엔진 업데이트가 성공적으로 완료되었습니다!"
        except Exception as e:
            logger.error(f"파이프라인 실행 오류: {e}", exc_info=True)
            raise

    # ... (Helper 메서드들: 메타데이터 조회 등) ...
    def get_available_standards(self) -> List[str]:
        """DB에 존재하는 규정 ID 목록 반환"""
        ids = list(self.metadata_cache.keys())
        # 숫자 기준 정렬 로직 (예: FMVSS 108 -> 108 추출)
        def sort_key(k: str) -> int:
            nums = re.findall(r"(\d+)", str(k))
            return int(nums[0]) if nums else 9999
        return sorted(list(set(ids)), key=sort_key)

    def _load_all_metadata(self) -> Dict[str, dict]:
        """여러 소스(JSON)에서 메타데이터를 로드하여 통합"""
        merged = {}
        # 우선순위: KMVSS -> ECE -> FMVSS -> 통합 레지스트리
        candidates = [
            settings.DATA_DIR / "metadata_kmvss.json",
            settings.DATA_DIR / "metadata_ece.json",
            settings.DATA_DIR / "metadata_fmvss.json",
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
                                        "url": item.get("web_url") or item.get("source_url"),
                                    }
                        elif isinstance(data, dict):
                            merged.update(data)
                except Exception:
                    pass
        return merged
