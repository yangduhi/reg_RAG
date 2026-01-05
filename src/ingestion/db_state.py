import enum
from datetime import datetime
from pathlib import Path

from sqlalchemy import (
    create_engine,
    DateTime,
    Enum,
    func,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from src.core.config import settings
from src.core.logging import logger

# DB 엔진 및 세션 설정
# check_same_thread=False는 SQLite가 여러 스레드에서 접근될 때 필요합니다.
# SQLAlchemy의 세션 관리와 함께 사용하면 스레드 안전성을 보장할 수 있습니다.
engine = create_engine(
    f"sqlite:///{settings.DB_STATE_PATH}",
    echo=False,
    connect_args={"check_same_thread": False},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# ORM을 위한 기본 클래스
class Base(DeclarativeBase):
    pass


# Enum 정의
class StatusEnum(enum.Enum):
    SUCCESS = "Success"
    FAIL = "Fail"


# 테이블 스키마 정의
class IngestionStatus(Base):
    __tablename__ = "ingestion_status"

    file_path: Mapped[str] = mapped_column(String, primary_key=True)
    file_hash: Mapped[str] = mapped_column(String, nullable=False)
    last_processed: Mapped[datetime] = mapped_column(
        DateTime, default=func.now(), onupdate=func.now()
    )
    status: Mapped[StatusEnum] = mapped_column(Enum(StatusEnum), nullable=False)
    error_message: Mapped[str] = mapped_column(Text, nullable=True)

    def __repr__(self):
        return f"<IngestionStatus(path='{self.file_path}', status='{self.status.value}')>"


# DB 상태 관리를 위한 클래스
class DatabaseStateManager:
    def __init__(self):
        self.engine = engine
        self.SessionLocal = SessionLocal
        Base.metadata.create_all(self.engine)
        logger.info("🗃️ DB 상태 관리자 초기화 완료.")

    def get_files_status(self, file_paths: list[Path]) -> dict[str, IngestionStatus]:
        """주어진 파일 경로 목록에 대한 IngestionStatus를 DB에서 조회합니다."""
        if not file_paths:
            return {}
        
        session = self.SessionLocal()
        try:
            path_strs = [str(p) for p in file_paths]
            results = session.query(IngestionStatus).filter(IngestionStatus.file_path.in_(path_strs)).all()
            return {result.file_path: result for result in results}
        finally:
            session.close()

    def update_status(
        self,
        file_path: Path,
        file_hash: str,
        status: StatusEnum,
        error_message: str | None = None,
    ):
        """특정 파일의 처리 상태를 DB에 업데이트하거나 새로 생성합니다. (Atomic)"""
        session = self.SessionLocal()
        try:
            # 기존 레코드 조회
            record = session.query(IngestionStatus).filter_by(file_path=str(file_path)).first()
            if record:
                # 업데이트
                record.file_hash = file_hash
                record.status = status
                record.error_message = error_message
                record.last_processed = func.now()
            else:
                # 새로 생성
                record = IngestionStatus(
                    file_path=str(file_path),
                    file_hash=file_hash,
                    status=status,
                    error_message=error_message,
                )
                session.add(record)
            
            session.commit()
            logger.debug(f"💾 상태 저장 완료: {record}")
        except Exception as e:
            session.rollback()
            logger.error(f"❌ DB 상태 저장 실패 ({file_path.name}): {e}")
        finally:
            session.close()

    def clear_all_status(self):
        """DB의 모든 상태 기록을 삭제합니다."""
        session = self.SessionLocal()
        try:
            session.query(IngestionStatus).delete()
            session.commit()
            logger.warning("🗑️ 모든 처리 상태 기록을 DB에서 삭제했습니다.")
        except Exception as e:
            session.rollback()
            logger.error(f"❌ DB 상태 초기화 실패: {e}")
        finally:
            session.close()

