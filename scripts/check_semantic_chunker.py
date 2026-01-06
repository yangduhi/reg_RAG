import asyncio
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[1]))

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from src.core.config import settings

def check_chunking():
    print("🛠️ SemanticChunker 점검 시작...")
    
    # 1. 임베딩 모델 로드 확인
    print(f"📦 임베딩 모델 로드 중: {settings.EMBEDDING_MODEL}")
    try:
        embedding_fn = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"}, # 테스트용으로 CPU 사용
            encode_kwargs={"normalize_embeddings": True},
        )
        print("✅ 임베딩 모델 로드 성공!")
    except Exception as e:
        print(f"❌ 임베딩 모델 로드 실패: {e}")
        return

    # 2. SemanticChunker 초기화
    splitter = SemanticChunker(embedding_fn)
    
    # 3. 테스트 데이터 (법규 문서 시뮬레이션)
    test_text = """
    제1조(목적) 이 규칙은 자동차관리법 제29조의 규정에 의하여 자동차의 안전기준에 관한 사항을 정함을 목적으로 한다.
    
    제2조(정의) 이 규칙에서 사용하는 용어의 정의는 다음과 같다.
    1. "자동차"라 함은 자동차관리법 제2조제1호의 규정에 의한 자동차를 말한다.
    2. "승용자동차"라 함은 10인 이하를 운송하기에 적합하게 제작된 자동차를 말한다.
    
    제3조(적용범위) 이 규칙은 국내에서 운행하는 모든 자동차에 적용한다. 다만, 군용차량에 대해서는 예외로 한다.
    """
    
    doc = Document(page_content=test_text)
    
    # 4. 청킹 실행
    print("\n✂️ 텍스트 분할 실행 중...")
    chunks = splitter.split_documents([doc])
    
    print(f"\n📊 분할 결과: {len(chunks)}개 청크 생성됨")
    for i, chunk in enumerate(chunks):
        print(f"\n[Chunk {i+1}] Length: {len(chunk.page_content)}")
        print("-" * 40)
        print(chunk.page_content.strip())
        print("-" * 40)

if __name__ == "__main__":
    check_chunking()
