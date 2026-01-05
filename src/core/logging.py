import logging

from rich.logging import RichHandler

from src.core.config import settings


def setup_logger(name: str = "FMVSS_AI") -> logging.Logger:
    """
    Rich 기반의 구조화된 로거 설정
    - Console: RichHandler (컬러, 가독성)
    - File: Standard FileHandler (상세 기록)
    """
    logger = logging.getLogger(name)

    # 중복 핸들러 방지
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    # 1. Console Handler (Rich)
    console_handler = RichHandler(
        rich_tracebacks=True,
        markup=True,
        show_time=True,
        show_path=False
    )
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(console_handler)

    # 2. File Handler
    settings.LOG_DIR.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(
        settings.LOG_DIR / "app.log",
        encoding="utf-8"
    )
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")
    )
    logger.addHandler(file_handler)

    return logger

# 기본 로거 인스턴스
logger = setup_logger()
