from pathlib import Path

import torch
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """
    애플리케이션 전역 설정 관리 (Singleton Pattern)
    - 환경변수(.env) 로드 및 기본값 설정을 담당합니다.
    - Pydantic을 사용하여 타입 안전성을 보장합니다.
    - 주요 설정: 프로젝트 정보, 클라우드/로컬 경로, 모델 설정, RAG 파라미터 등
    """

    # --- 프로젝트 기본 정보 ---
    PROJECT_NAME: str = "Global Auto Regulations AI"
    VERSION: str = "2.0.0"
    DEBUG: bool = False

    # --- 클라우드 설정 (GCS 연동용) ---
    # GCS_BUCKET_NAME이 설정되면 클라우드 환경으로 동작하며, 데이터 경로가 변경됩니다.
    GCS_BUCKET_NAME: str | None = Field(
        default=None, description="Google Cloud Storage 버킷 이름"
    )
    GCS_DATA_DIR: str = "data"  # GCS 버킷 내 데이터가 저장된 최상위 폴더

    # --- 경로 설정 (Path Configuration) ---
    # 현재 파일 위치를 기준으로 프로젝트 루트 경로를 자동 계산합니다.
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent

    @property
    def DATA_DIR(self) -> Path:
        """
        데이터 저장 디렉토리를 동적으로 결정합니다.
        - GCS 사용 시: 클라우드 컨테이너의 임시 경로(/tmp) 사용
        - 로컬 사용 시: 프로젝트 루트의 'data' 폴더 사용
        """
        if self.GCS_BUCKET_NAME:
            return Path("/tmp") / self.GCS_DATA_DIR
        else:
            return self.BASE_DIR / self.GCS_DATA_DIR

    # [원시 데이터 경로]
    # 각 규정(FMVSS, KMVSS, ECE)의 원본 파일(XML, PDF)이 저장되는 위치
    @property
    def RAW_XML_FMVSS_PATH(self) -> Path:
        return self.DATA_DIR / "raw" / "xml_fmvss"

    @property
    def RAW_XML_KMVSS_PATH(self) -> Path:
        return self.DATA_DIR / "raw" / "xml_kmvss"

    @property
    def RAW_PDF_ECE_PATH(self) -> Path:
        return self.DATA_DIR / "raw" / "pdf_ece"

    # [가공 데이터 및 DB 경로]
    @property
    def VECTOR_DB_PATH(self) -> Path:
        return self.DATA_DIR / "vector_db"  # ChromaDB 저장 경로

    @property
    def METADATA_FILE(self) -> Path:
        return self.DATA_DIR / "metadata_registry.json"  # 통합 메타데이터 파일

    @property
    def DB_STATE_PATH(self) -> Path:
        return self.DATA_DIR / "ingestion_state.sqlite"  # 데이터 수집 상태 관리 DB

    @property
    def LOG_DIR(self) -> Path:
        return self.BASE_DIR / "logs"  # 로그 파일 저장 경로

    # --- AI 모델 설정 (Model Settings) ---
    # Google Gemini API 키 (환경변수에서 로드)
    GOOGLE_API_KEY: str = Field(description="Google Gemini API Key")
    
    # 사용할 LLM 모델명 (비용 및 성능을 고려하여 2.0 Flash/Pro 등 선택)
    LLM_MODEL_NAME: str = "gemini-2.0-flash"  # 고속 처리용 (질의 변환, 평가)
    LLM_MODEL_SMART: str = "gemini-2.5-pro"   # 고성능 추론용 (최종 답변 생성)
    LLM_TEMPERATURE: float = 0.0 # 0.0으로 설정하여 답변의 일관성 및 사실성 확보 (창의성 억제)

    # [임베딩 및 재순위화 모델]
    # 다국어(한국어/영어) 지원이 우수한 모델 선정
    EMBEDDING_MODEL: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    # 검색 결과 재순위화를 위한 Cross-Encoder 모델 (영어 성능 우수, 한국어는 별도 로직으로 보완)
    RERANKER_MODEL: str = "ms-marco-MiniLM-L-12-v2"

    # [하드웨어 가속] PyTorch Device 자동 감지 (GPU/CPU)
    DEVICE: str = Field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )

    # --- RAG 파라미터 (RAG Parameters) ---
    CHUNK_SIZE: int = 2000  # 청킹 크기: 규정 문서의 특성상 긴 문맥(표, 조항)을 유지하기 위해 크게 설정
    CHUNK_OVERLAP: int = 400 # 중복 크기: 문맥 단절 방지
    RETRIEVER_K: int = 25    # 검색 문서 수: Annex(별표) 등 세부 문서를 놓치지 않기 위해 넉넉하게 설정 (15 -> 25 상향)
    USE_MMR: bool = True     # MMR(Maximal Marginal Relevance) 사용 여부: 검색 결과의 다양성 확보

    # --- Pydantic 설정 ---
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # 정의되지 않은 환경변수는 무시하여 에러 방지
    )

    def create_dirs(self):
        """
        필요한 데이터 디렉토리 자동 생성
        - 로컬 환경에서 실행 시 data 폴더 구조를 자동으로 만듭니다.
        """
        if self.GCS_BUCKET_NAME:
            # GCS 사용 시에는 gcs_utils가 동기화 과정에서 디렉토리를 생성하므로 별도 처리 안 함
            self.DATA_DIR.mkdir(parents=True, exist_ok=True)
            return

        dirs = [
            self.RAW_XML_FMVSS_PATH,
            self.RAW_XML_KMVSS_PATH,
            self.RAW_PDF_ECE_PATH,
            self.VECTOR_DB_PATH,
            self.LOG_DIR,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)


# 전역 설정 인스턴스 (다른 모듈에서 import하여 사용)
settings = AppSettings()

# 로컬 환경인 경우 디렉토리 자동 생성 수행
if not settings.GCS_BUCKET_NAME:
    settings.create_dirs()
