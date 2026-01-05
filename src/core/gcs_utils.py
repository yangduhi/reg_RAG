"""
Google Cloud Storage (GCS) 유틸리티 모듈
- GCS 버킷과의 파일 및 디렉토리 동기화를 처리합니다.
- Cloud Run 환경에서는 서비스 계정의 권한을 통해 자동으로 인증됩니다.
- 로컬 개발 환경에서는 `gcloud auth application-default login` 명령을 통해 인증해야 합니다.
"""
import os
from pathlib import Path
from loguru import logger
from google.cloud import storage
from google.api_core.exceptions import NotFound

def get_gcs_client() -> storage.Client:
    """GCS 클라이언트를 생성하고 반환합니다."""
    try:
        client = storage.Client()
        return client
    except Exception as e:
        logger.error(f"GCS 클라이언트 생성 실패: {e}")
        logger.info("Cloud Run 환경이 아닌 경우, 'gcloud auth application-default login' 명령으로 인증했는지 확인하세요.")
        raise

def sync_folder_from_gcs(
    client: storage.Client,
    bucket_name: str,
    gcs_folder_path: str,
    local_folder_path: Path
):
    """GCS의 폴더를 로컬 폴더로 재귀적으로 다운로드(동기화)합니다."""
    logger.info(f"GCS 동기화 시작: gs://{bucket_name}/{gcs_folder_path} -> {local_folder_path}")
    
    bucket = client.bucket(bucket_name)
    blobs = client.list_blobs(bucket_name, prefix=gcs_folder_path)
    
    downloaded_count = 0
    for blob in blobs:
        # GCS 경로에서 로컬 파일 경로 생성
        relative_path = os.path.relpath(blob.name, gcs_folder_path)
        local_file_path = local_folder_path / relative_path

        # 로컬에 디렉토리 생성
        local_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 파일 다운로드
        try:
            blob.download_to_filename(str(local_file_path))
            downloaded_count += 1
        except Exception as e:
            logger.error(f"파일 다운로드 실패 {blob.name}: {e}")
            
    if downloaded_count == 0:
        logger.warning("GCS에서 다운로드할 파일이 없습니다. 버킷/경로가 올바른지 확인하세요.")
    else:
        logger.success(f"GCS 동기화 완료: 총 {downloaded_count}개 파일 다운로드")


def sync_folder_to_gcs(
    client: storage.Client,
    bucket_name: str,
    local_folder_path: Path,
    gcs_folder_path: str,
):
    """로컬 폴더의 내용을 GCS 폴더로 재귀적으로 업로드합니다."""
    logger.info(f"GCS 업로드 시작: {local_folder_path} -> gs://{bucket_name}/{gcs_folder_path}")

    bucket = client.bucket(bucket_name)
    
    uploaded_count = 0
    for local_file in local_folder_path.rglob('*'):
        if local_file.is_file():
            # 로컬 파일 경로에서 GCS blob 이름 생성
            gcs_blob_name = f"{gcs_folder_path}/{local_file.relative_to(local_folder_path)}"
            
            blob = bucket.blob(gcs_blob_name)
            try:
                blob.upload_from_filename(str(local_file))
                uploaded_count += 1
            except Exception as e:
                logger.error(f"파일 업로드 실패 {local_file}: {e}")

    logger.success(f"GCS 업로드 완료: 총 {uploaded_count}개 파일 업로드")

def check_gcs_folder_exists(client: storage.Client, bucket_name: str, gcs_folder_path: str) -> bool:
    """GCS에 지정된 폴더(prefix)에 파일이 하나라도 있는지 확인합니다."""
    bucket = client.bucket(bucket_name)
    blobs = client.list_blobs(bucket_name, prefix=gcs_folder_path, max_results=1)
    return len(list(blobs)) > 0
