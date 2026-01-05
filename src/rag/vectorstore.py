import shutil

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from src.core.config import settings
from src.core.logging import logger


class VectorStoreManager:
    """
    벡터 데이터베이스(ChromaDB) 관리 클래스
    - 문서 임베딩 및 저장
    - DB 초기화 및 로드
    """
    def __init__(self):
        self.persist_directory = str(settings.VECTOR_DB_PATH)

        # 임베딩 모델 초기화 (다국어 모델 사용)
        self.embedding_fn = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            model_kwargs={"device": settings.DEVICE},
            encode_kwargs={"normalize_embeddings": True}
        )

        self._db = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embedding_fn,
            collection_name="regulations_store"
        )

    @property
    def db(self) -> Chroma:
        return self._db

    def add_documents(self, documents: list[Document], batch_size: int = 100):
        """문서를 배치 단위로 DB에 추가"""
        if not documents:
            return

        total = len(documents)
        logger.info(f"💾 벡터 DB 저장 시작: 총 {total}개 문서 청크")

        for i in range(0, total, batch_size):
            batch = documents[i : i + batch_size]
            self._db.add_documents(batch)
            logger.debug(f"   -> 진행률: {min(i + batch_size, total)}/{total}")

        logger.info("✅ 벡터 DB 저장 완료")

    def clear(self):
        """DB 초기화 (데이터 삭제)"""
        logger.warning("⚠️ 벡터 DB를 초기화(삭제)합니다.")
        self._db = None
        if settings.VECTOR_DB_PATH.exists():
            shutil.rmtree(settings.VECTOR_DB_PATH)
        settings.VECTOR_DB_PATH.mkdir(parents=True, exist_ok=True)
        # 재연결
        self.__init__()

    def get_all_documents(self) -> list[Document]:
        """DB에서 모든 문서를 LangChain Document 형태로 가져옵니다."""
        logger.info("📚 DB에서 모든 문서를 로드하는 중... (BM25 인덱싱용)")

        # .get()은 metadatas와 documents를 포함한 dict를 반환합니다.
        raw_docs = self._db.get()

        # Document 객체 재구성
        all_documents = []
        if raw_docs and 'documents' in raw_docs and raw_docs['documents']:
            for i, page_content in enumerate(raw_docs['documents']):
                metadata = raw_docs['metadatas'][i] if raw_docs['metadatas'] and i < len(raw_docs['metadatas']) else {}
                all_documents.append(Document(page_content=page_content, metadata=metadata))

        logger.info(f"✅ 총 {len(all_documents)}개 문서 로드 완료.")
        return all_documents
