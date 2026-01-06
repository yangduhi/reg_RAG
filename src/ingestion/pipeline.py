import asyncio
import hashlib
import shutil
from pathlib import Path
from typing import List, Tuple, Optional, Dict

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rich.traceback import install

from src.core.config import settings
from src.core.logging import logger
from src.ingestion.db_state import DatabaseStateManager, StatusEnum
from src.ingestion.loaders import LoaderFactory
from src.rag.vectorstore import VectorStoreManager

# Rich traceback 활성화로 디버깅 가시성 향상
install(show_locals=True, width=120)


class IngestionPipeline:
    """
    RAG 시스템의 데이터 수집 파이프라인을 조정하는 클래스입니다.

    이 클래스는 다음과 같은 엔드 투 엔드 프로세스를 처리합니다:
    1. 구성된 디렉토리에서 대상 규정 파일(XML, PDF) 식별.
    2. 증분 업데이트(Incremental Update)를 지원하기 위해 파일 변경 사항(SHA256 해시) 추적.
    3. 전용 로더(Loader)를 사용한 파일 로딩 및 파싱.
    4. 텍스트를 구조적 청크(Recursive Character Chunk)로 분할.
    5. 청크에 문맥 메타데이터(규정 ID, 제목 등) 주입.
    6. 벡터 데이터베이스에 청크 인덱싱.
    7. 수집 상태 데이터베이스 업데이트.

    Attributes:
        vstore (VectorStoreManager): 벡터 데이터베이스 인터페이스.
        db_state_manager (DatabaseStateManager): 파일 처리 상태 관리자.
        splitter (RecursiveCharacterTextSplitter): 텍스트 구조 기반의 분할기.
    """

    def __init__(self) -> None:
        """
        IngestionPipeline을 초기화합니다.
        
        데이터베이스 상태 관리자를 설정하고, 텍스트 구조를 보존하는
        RecursiveCharacterTextSplitter를 초기화합니다.
        """
        self.vstore: Optional[VectorStoreManager] = None
        self.db_state_manager = DatabaseStateManager()

        logger.info(f"🛠️ RecursiveCharacterTextSplitter 초기화 (Size: {settings.CHUNK_SIZE}, Overlap: {settings.CHUNK_OVERLAP})")
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""], # 법규 구조 보존을 위한 구분자 순서
            length_function=len,
        )

    def _calculate_hash(self, file_path: Path) -> str:
        """
        파일의 SHA256 해시를 효율적으로 계산합니다.

        대용량 파일을 처리할 때 메모리 사용량을 최소화하기 위해 파일을 청크 단위로 읽습니다.

        Args:
            file_path (Path): 파일 경로.

        Returns:
            str: 16진수 SHA256 해시 문자열.
        """
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            while True:
                chunk = f.read(8192)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()

    async def run(self, force_refresh: bool = False) -> None:
        """
        메인 수집 파이프라인 로직을 실행합니다.

        Args:
            force_refresh (bool): True일 경우, 기존 벡터 데이터베이스와 처리 상태를 초기화하고
                                  모든 데이터를 처음부터 다시 인덱싱합니다.
        """
        if force_refresh:
            logger.warning("🔄 강제 새로고침 모드: 모든 데이터를 다시 처리합니다.")
            if settings.VECTOR_DB_PATH.exists():
                logger.warning(f"🗑️ 기존 벡터 DB 삭제 중: {settings.VECTOR_DB_PATH}")
                try:
                    shutil.rmtree(settings.VECTOR_DB_PATH)
                except PermissionError:
                    logger.error("❌ 권한 거부됨! Streamlit 서버를 중지하고 다시 시도하세요.")
                    return
            self.db_state_manager.clear_all_status()

        # 벡터 저장소 초기화 (여기서 연결 수립)
        self.vstore = VectorStoreManager()

        # 1. Scan for Target Files
        target_files: List[Path] = []
        
        # [FMVSS]: Use pre-processed JSON files (Data quality is better than raw XML)
        json_dir = settings.DATA_DIR / "processed_json_for_rag"
        if json_dir.exists():
            target_files.extend(json_dir.glob("*.json"))

        # [KMVSS]: Use raw XML files
        if settings.RAW_XML_KMVSS_PATH.exists():
            target_files.extend(settings.RAW_XML_KMVSS_PATH.glob("*.xml"))
            
        # [ECE]: Use PDF files
        if settings.RAW_PDF_ECE_PATH.exists():
            target_files.extend(settings.RAW_PDF_ECE_PATH.glob("*.pdf"))
        
        # Filter out archive files and sort for deterministic processing order
        target_files = sorted([f for f in target_files if "archive" not in str(f)])

        if not target_files:
            logger.warning("❌ 처리할 데이터 파일이 없습니다.")
            return

        # 2. 파일 상태 확인 (증분 업데이트 로직)
        last_states = self.db_state_manager.get_files_status(target_files)
        to_process: List[Tuple[Path, str]] = []

        for f in target_files:
            current_hash = self._calculate_hash(f)
            last_record = last_states.get(str(f))

            # 처리 대상: 새 파일 OR 해시 변경됨 OR 이전 시도 실패
            if not last_record or \
               last_record.file_hash != current_hash or \
               last_record.status == StatusEnum.FAIL:
                to_process.append((f, current_hash))
        
        if not to_process:
            logger.info("✨ 모든 데이터가 최신입니다.")
            return

        logger.info(f"🔄 처리 대상 파일 수: {len(to_process)}개")
        
        # 3. 파일 처리 (로드 -> 청킹 -> 인덱싱)
        documents_to_add: List[Document] = []
        processed_files: List[Tuple[Path, str, StatusEnum, Optional[str]]] = []

        for file_path, file_hash in to_process:
            error_msg: Optional[str] = None
            try:
                # 팩토리 패턴을 사용하여 파일 확장자에 맞는 로더 생성
                loader = LoaderFactory.create(file_path)
                ingested_docs = await loader.load(file_path)
                
                if not ingested_docs:
                    logger.warning(f"⚠️ 문서가 추출되지 않음 (Skipped): {file_path.name}")
                    # 실패가 아닌 'SKIPPED' 상태로 기록하거나, 일단 성공으로 처리하되 내용은 없음
                    processed_files.append((file_path, file_hash, StatusEnum.SUCCESS, "Skipped (No Content)"))
                    continue

                # 내부 문서 형식을 LangChain 형식으로 변환
                langchain_docs = [i_doc.to_langchain_format() for i_doc in ingested_docs]
                documents_to_add.extend(langchain_docs)
                
                processed_files.append((file_path, file_hash, StatusEnum.SUCCESS, None))
                logger.info(f"✅ 로드 성공: {file_path.name}")

            except Exception as e:
                error_msg = f"{type(e).__name__}: {e}"
                logger.error(f"❌ 처리 실패 ({file_path.name}): {error_msg}")
                processed_files.append((file_path, file_hash, StatusEnum.FAIL, error_msg))

        if documents_to_add:
            # 3-1. 의미론적 청킹 및 문맥 주입
            logger.info(f"✂️ {len(documents_to_add)}개 문서 분할 중...")
            chunks = self.splitter.split_documents(documents_to_add)
            logger.info(f"🧩 {len(chunks)}개 청크 생성됨.")
            
            enriched_chunks = self._enrich_chunks_context(chunks)

            # 3-2. 벡터 DB에 인덱싱
            self.vstore.add_documents(enriched_chunks)

        # 4. 상태 데이터베이스 업데이트 (원자적 업데이트)
        for file_path, file_hash, status, error_msg in processed_files:
            self.db_state_manager.update_status(file_path, file_hash, status, error_msg)

        logger.info("✅ 파이프라인 실행이 성공적으로 완료되었습니다!")

    def _enrich_chunks_context(self, chunks: List[Document]) -> List[Document]:
        """
        각 청크의 페이지 콘텐츠에 메타데이터 문맥을 주입합니다.
        
        청크 텍스트 자체가 일반적이더라도 규정 ID 및 제목과 같은 
        중요한 문맥 정보를 임베딩 벡터에 포함시켜 검색 정확도를 향상시킵니다.

        Args:
            chunks (List[Document]): 문서 청크 리스트.

        Returns:
            List[Document]: 문맥이 주입된 문서 청크 리스트.
        """
        enriched = []
        for chunk in chunks:
            source = chunk.metadata.get("source", "")
            title = chunk.metadata.get("title", "")
            std_id = chunk.metadata.get("standard_id", "")

            # 대체 로직: 메타데이터에 ID가 없으면 파일명에서 추출
            if not std_id and source:
                path_obj = Path(source)
                std_id = path_obj.stem

            # 문맥 헤더 생성
            context_header = f"[Standard: {std_id}]"
            if title:
                context_header += f" [Title: {title}]"

            # 콘텐츠 앞에 문맥 추가
            chunk.page_content = f"{context_header}\n{chunk.page_content}"
            enriched.append(chunk)

        logger.info(f"🧬 {len(enriched)}개 청크에 메타데이터 문맥 주입 완료.")
        return enriched


if __name__ == "__main__":
    pipeline = IngestionPipeline()
    asyncio.run(pipeline.run(force_refresh=True))