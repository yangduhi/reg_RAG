import asyncio
import sys
import os

# 프로젝트 루트 경로를 sys.path에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.rag.engine import RAGEngine
from src.core.config import settings

async def main():
    print(f"🔹 Config Check:")
    print(f"   - Fast Model (Transform/Grade): {settings.LLM_MODEL_NAME}")
    print(f"   - Smart Model (Generate): {settings.LLM_MODEL_SMART}")
    
    print("\n🚀 RAGEngine initializing...")
    engine = RAGEngine()
    
    # Wait for initialization
    print("⏳ Waiting for retriever initialization...")
    async with engine.initialization_lock:
        if not engine.is_initialized:
            await engine.initialization_task
    print("✅ Initialization complete.")

    question = "FMVSS 208의 주요 내용은 무엇인가요?"
    print(f"\n❓ Test Question: {question}")
    
    print("🏃 Running Chat Workflow...")
    result = await engine.chat(question)
    
    print("\n[Generated Answer]")
    print("=" * 60)
    print(result["generation"])
    print("=" * 60)
    
    print(f"\n📚 Source Documents: {len(result['documents'])}")
    for i, doc in enumerate(result['documents'][:3]):
        print(f"   {i+1}. [{doc.metadata.get('standard_id')}] {doc.page_content[:50]}...")

if __name__ == "__main__":
    asyncio.run(main())
