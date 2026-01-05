import asyncio
import sys
from pathlib import Path

# 프로젝트 루트 경로 추가
sys.path.append(str(Path(__file__).parents[1]))

from src.rag.engine import RAGEngine
from src.core.logging import logger

async def main():
    print("🚀 LangGraph 기반 RAG 시스템 테스트 시작...")
    
    try:
        # 엔진 초기화 (여기서 그래프도 초기화됨)
        engine = RAGEngine()
        
        # 비동기 초기화 대기 (엔진 내부적으로 처리하지만 확실하게 하기 위해)
        print("⏳ 초기화 대기 중...")
        await engine.initialization_task
        
        # 질문 던지기
        question = "한국과 유럽의 보행자보호 다리 상해 기준을 비교해줘"
        print(f"\n❓ 질문: {question}")
        
        # 답변 생성
        answer = await engine.chat(question)
        
        print("\n✅ 답변:")
        print(answer)
        
    except Exception as e:
        logger.error(f"❌ 테스트 실패: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(main())

