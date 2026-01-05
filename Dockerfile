# ==============================================================================
# STAGE 1: Builder - 빌드 의존성 설치 및 패키지 빌드
# ==============================================================================
FROM python:3.11-slim-bookworm AS builder

# 시스템 패키지 업데이트 및 빌드에 필요한 도구 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python 가상 환경 생성
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# PyTorch CPU 버전 우선 설치 (이미지 용량 최적화)
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# requirements.txt를 복사하고 모든 Python 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# ==============================================================================
# STAGE 2: Runner - 최종 실행 환경 구성
# ==============================================================================
FROM python:3.11-slim-bookworm AS runner

# 보안을 위해 non-root 사용자 생성
RUN groupadd --system app && useradd --system --gid app app
USER app

# 작업 디렉토리 설정
WORKDIR /app

# Builder 스테이지에서 빌드된 가상 환경(Python 패키지)만 복사
COPY --from=builder /opt/venv /opt/venv

# 필요한 소스 코드만 복사
COPY --chown=app:app src/ ./src/

# 가상 환경 경로 설정
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Cloud Run이 주입하는 PORT 환경 변수(기본 8080)를 사용하도록 설정
# ENTRYPOINT와 CMD를 분리하여 유연성 확보
ENTRYPOINT ["streamlit", "run"]
CMD ["src/interface/streamlit_app.py", "--server.port=$PORT", "--server.address=0.0.0.0", "--server.enableCORS=false"]