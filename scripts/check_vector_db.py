# check_db.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.rag.vectorstore import VectorStoreManager

def check(query=None):
    print("🔍 데이터베이스 검사 중...")
    try:
        vm = VectorStoreManager()
        collection = vm.db._collection

        total = collection.count()
        print(f"📊 총 문서 수: {total}개")

        if total == 0:
            print("❌ DB가 비어있습니다!")
            return

        if query:
            print(f"\n🔎 검색 테스트: '{query}'")
            # 간단한 텍스트 검색 (contains)
            results = collection.get(where_document={"$contains": query}, limit=5)
            if not results['ids']:
                print("❌ 검색 결과가 없습니다.")
            else:
                print(f"✅ {len(results['ids'])}개 문서 발견:")
                for i, doc in enumerate(results['documents']):
                    meta = results['metadatas'][i]
                    print(f"[{i+1}] {meta.get('source_file')} (ID: {meta.get('standard_id')})")
                    print(f"    {doc[:100]}...\n")
            return

        # 한국어 데이터 샘플링
        results = collection.get(limit=1000, include=['metadatas', 'documents'])
        korean_count = 0

        for meta, doc in zip(results['metadatas'], results['documents']):
            # 파일명이나 내용에 한글/KMVSS가 있는지 확인
            if "KMVSS" in str(meta) or "제" in doc[:10] or "조" in doc[:10]:
                korean_count += 1
                if korean_count == 1:
                    print(f"\n✅ 한국 데이터 발견 예시:\n- 파일: {meta.get('source_file')}\n- 내용: {doc[:50]}...")

        print(f"\n🇰🇷 한국어 추정 문서 수: {korean_count} / 1000 (샘플링)")

        if korean_count == 0:
            print("\n🚨 결론: DB에 한국 데이터가 하나도 없습니다. 로더(Loader)가 XML을 못 읽고 있습니다.")
        else:
            print("\n✅ 결론: 데이터는 있습니다. 검색 로직 문제입니다.")

    except Exception as e:
        print(f"오류: {e}")

if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else None
    check(query)
