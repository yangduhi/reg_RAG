from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class RegulationRegion(str, Enum):
    """규정 적용 지역/국가"""
    FMVSS = "FMVSS"   # 미국
    ECE = "ECE"       # 유럽
    KMVSS = "KMVSS"   # 한국

class DocumentMetadata(BaseModel):
    """
    RAG 문서의 메타데이터 스키마
    - 답변의 근거(Citation)를 추적하기 위한 필수 정보
    """
    source_file: str = Field(..., description="원본 파일명")
    region: RegulationRegion = Field(..., description="규정 지역")
    standard_id: str = Field(..., description="규정 식별자 (예: 108, R13, 제2조)")
    title: str = Field(default="No Title", description="규정 제목")
    page: int | None = Field(None, description="PDF 페이지 번호 (PDF인 경우)")
    url: str | None = Field(None, description="웹 원문 링크")
    crawled_at: datetime = Field(default_factory=datetime.now)

class IngestedDocument(BaseModel):
    """
    로더(Loader)가 반환하는 표준 문서 객체
    """
    content: str = Field(..., description="전처리된 본문 텍스트 (Markdown 권장)")
    metadata: DocumentMetadata

    def to_langchain_format(self):
        """LangChain 호환 객체로 변환"""
        from langchain_core.documents import Document
        # LangChain은 metadata 값으로 dict만 허용
        return Document(page_content=self.content, metadata=self.metadata.model_dump(mode='json'))
