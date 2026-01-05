from src.core.config import settings


def test_project_settings_load():
    """기본 프로젝트 설정이 올바르게 로드되는지 테스트"""
    assert settings.PROJECT_NAME == "Global Auto Regulations AI"
    assert settings.VERSION == "2.0.0"

def test_paths_exist():
    """중요 데이터 경로들이 실제로 존재하는지 테스트"""
    # config.py가 초기화될 때 create_dirs()가 돌면서 폴더를 만드므로 존재해야 함
    assert settings.DATA_DIR.exists()
    assert settings.RAW_XML_FMVSS_PATH.exists()
    assert settings.LOG_DIR.exists()

def test_api_key_loaded():
    """API 키가 .env에서 잘 읽혔는지 테스트"""
    # 실제 키 값을 출력하면 보안상 위험하므로, None이나 빈 값이 아닌지만 체크
    assert settings.GOOGLE_API_KEY is not None
    assert len(settings.GOOGLE_API_KEY) > 0
