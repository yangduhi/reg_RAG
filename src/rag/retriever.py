from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from src.core.config import settings
from src.core.logging import logger
from src.rag.vectorstore import VectorStoreManager


def get_retriever(
    filter_std: str = "All",
    k: int = settings.RETRIEVER_K
) -> ContextualCompressionRetriever:
    """
    LangChain v1 기반 리랭킹 검색기 반환
    """
    vstore_mgr = VectorStoreManager()

    # 1. 기본 검색기 (Vector Store)
    fetch_k = k * 4
    search_kwargs = {"k": fetch_k}

    if filter_std and filter_std != "All":
        search_kwargs["filter"] = {"standard_id": filter_std}

    base_retriever = vstore_mgr.db.as_retriever(
        search_type="similarity",
        search_kwargs=search_kwargs
    )

    # 2. 리랭커 (Cross Encoder)
    # HuggingFaceCrossEncoder는 커뮤니티 패키지에 유지됨
    logger.info(f"🚀 리랭커 초기화 중: {settings.RERANKER_MODEL}")

    model = HuggingFaceCrossEncoder(
        model_name=settings.RERANKER_MODEL,
        model_kwargs={"device": settings.DEVICE}
    )

    # 3. 압축기 연결
    compressor = CrossEncoderReranker(model=model, top_n=k)

    # 4. 최종 파이프라인
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )

    return compression_retriever
