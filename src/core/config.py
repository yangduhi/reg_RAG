from pathlib import Path

import torch
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """
    애플리케이션 전역 설정 관리 (Singleton Pattern)
    환경변수(.env) 로드 및 기본값 설정을 담당합니다.
    """
    # --- Project Info ---
    PROJECT_NAME: str = "Global Auto Regulations AI"
    VERSION: str = "2.0.0"
    DEBUG: bool = False

    # --- Cloud Settings (for GCS integration) ---
    GCS_BUCKET_NAME: str | None = Field(default=None, description="Google Cloud Storage 버킷 이름")
    GCS_DATA_DIR: str = "data" # GCS 버킷 내 데이터가 저장된 최상위 폴더

    # --- Paths (자동으로 절대 경로 변환) ---
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent

    @property
    def DATA_DIR(self) -> Path:
        """GCS 사용 여부에 따라 데이터 디렉토리를 동적으로 결정"""
        if self.GCS_BUCKET_NAME:
            # 클라우드 환경 (GCS 연동) 시 컨테이너 내 임시 로컬 경로 사용
            return Path("/tmp") / self.GCS_DATA_DIR
        else:
            # 로컬 환경 시 프로젝트 내 data 폴더 사용
            return self.BASE_DIR / self.GCS_DATA_DIR

    # Raw Data Paths
    @property
    def RAW_XML_FMVSS_PATH(self) -> Path: return self.DATA_DIR / "raw" / "xml_fmvss"
    @property
    def RAW_XML_KMVSS_PATH(self) -> Path: return self.DATA_DIR / "raw" / "xml_kmvss"
    @property
    def RAW_PDF_ECE_PATH(self) -> Path: return self.DATA_DIR / "raw" / "pdf_ece"

    # Processed / DB Paths
    @property
    def VECTOR_DB_PATH(self) -> Path: return self.DATA_DIR / "vector_db"
    @property
    def METADATA_FILE(self) -> Path: return self.DATA_DIR / "metadata_registry.json"
    @property
    def DB_STATE_PATH(self) -> Path: return self.DATA_DIR / "ingestion_state.sqlite"
    @property
    def LOG_DIR(self) -> Path: return self.BASE_DIR / "logs"

    # --- Model Settings ---
    GOOGLE_API_KEY: str = Field(..., description="Google Gemini API Key is required")
    LLM_MODEL_NAME: str = "gemini-2.5-flash"
    LLM_TEMPERATURE: float = 0.0

    # 다국어(한국어/영어) 지원 임베딩 모델
    EMBEDDING_MODEL: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    RERANKER_MODEL: str = "ms-marco-MiniLM-L-12-v2"

    # [중요] PyTorch Device 자동 감지 (auto 대신 명시적 cuda/cpu 사용)
    DEVICE: str = Field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )

    # --- RAG Parameters ---
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    RETRIEVER_K: int = 15
    USE_MMR: bool = True

    # --- Configuration ---
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"  # 정의되지 않은 환경변수는 무시
    )

    def create_dirs(self):
        """필요한 데이터 디렉토리 자동 생성 (GCS 사용 시에는 호출되지 않음)"""
        if self.GCS_BUCKET_NAME:
            # GCS 사용 시에는 gcs_utils가 동기화 과정에서 디렉토리를 생성하므로 별도 처리 안 함
            self.DATA_DIR.mkdir(parents=True, exist_ok=True)
            return

        dirs = [
            self.RAW_XML_FMVSS_PATH,
            self.RAW_XML_KMVSS_PATH,
            self.RAW_PDF_ECE_PATH,
            self.VECTOR_DB_PATH,
            self.LOG_DIR
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

# 전역 설정 인스턴스
settings = AppSettings()
# GCS를 사용하지 않을 때만 로컬 디렉토리 생성
if not settings.GCS_BUCKET_NAME:
    settings.create_dirs()
