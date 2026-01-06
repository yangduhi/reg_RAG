from src.rag.vectorstore import VectorStoreManager

def inspect_ece():
    vm = VectorStoreManager()
    # region이 'ECE'인 문서 필터링
    results = vm.db._collection.get(where={"region": "ECE"}, limit=10)
    
    print(f"--- ECE Documents in DB: {len(results['documents'])} ---")
    for i in range(len(results['documents'])):
        doc = results['documents'][i]
        meta = results['metadatas'][i]
        print(f"[{i+1}] ID: {meta.get('standard_id')}")
        print(f"Source: {meta.get('source_file')}")
        print(f"Content Length: {len(doc)}")
        print(f"Preview: {doc[:300]}...")
        print("-" * 50)

if __name__ == "__main__":
    inspect_ece()
