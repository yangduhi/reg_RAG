# 🚗 자동차 안전 법규 RAG 시스템 (Regulatory RAG System)

복잡하고 방대한 자동차 안전 법규(FMVSS, KMVSS, ECE)를 효과적으로 탐색할 수 있는 **검색 증강 생성(Retrieval-Augmented Generation, RAG)** 시스템입니다. 최신 NLP 기술을 활용하여 법규 문서를 수집, 인덱싱하고, 사용자의 질의에 대해 정확한 문맥을 파악하여 전문적인 답변을 제공합니다.

---

## 📖 개요

자동차 안전 법규는 기술적 밀도가 높고 상호 참조가 많아 이해하기 어렵습니다. 본 시스템은 이러한 문제를 해결하기 위해 개발되었습니다:
1.  **수집 (Ingestion)**: XML 및 PDF 형식의 원문 법규 데이터를 수집합니다.
2.  **인덱싱 (Indexing)**: 의미론적 청킹(Semantic Chunking)과 메타데이터 강화(Enrichment)를 통해 벡터 DB에 저장합니다.
3.  **검색 (Retrieval)**: 하이브리드 검색(키워드 + 의미 기반)을 통해 가장 관련성 높은 조항을 찾습니다.
4.  **생성 (Generation)**: 검색된 근거 자료를 바탕으로 LLM(Google Gemini)이 정확하고 신뢰할 수 있는 답변을 생성합니다.

## 🏗️ 시스템 아키텍처 및 데이터 흐름 (Node Flow)

본 프로젝트의 데이터 처리 및 질의응답 흐름은 다음과 같습니다.

```mermaid
graph TD
    %% 스타일 정의
    classDef source fill:#e1f5fe,stroke:#01579b,stroke-width:2px;
    classDef process fill:#fff9c4,stroke:#fbc02d,stroke-width:2px;
    classDef db fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,stroke-dasharray: 5 5;
    classDef ai fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px;

    subgraph Data_Sources [데이터 원본]
        A[FMVSS (미국, XML)]:::source
        B[KMVSS (한국, XML)]:::source
        C[ECE (유럽/UN, PDF)]:::source
    end

    subgraph Ingestion_Pipeline [데이터 수집 파이프라인]
        Loader[Data Loader<br/>(파일 파싱)]:::process
        Splitter[Semantic Chunker<br/>(의미 단위 분할)]:::process
        Enrich[Metadata Enrichment<br/>(문맥/제목 주입)]:::process
        Embed[Embedding Model<br/>(HuggingFace)]:::ai
        VectorDB[(Vector Store<br/>ChromaDB)]:::db
    end

    subgraph RAG_Engine [RAG 엔진 (질의응답)]
        User[사용자 질문]:::source
        QueryTrans[Query Transformation<br/>(번역 및 확장)]:::process
        Retriever{Hybrid Retriever<br/>(BM25 + Vector)}:::process
        Reranker[FlashRank Reranker<br/>(재순위화)]:::ai
        Generator[LLM Generator<br/>(Google Gemini)]:::ai
        Answer[최종 답변]:::process
    end

    %% 연결선 (데이터 흐름)
    A & B & C --> Loader
    Loader --> Splitter
    Splitter --> Enrich
    Enrich --> Embed
    Embed --> VectorDB

    User --> QueryTrans
    QueryTrans --> Retriever
    VectorDB <--> Retriever
    Retriever -->|후보 문서 추출| Reranker
    Reranker -->|상위 K개 문서| Generator
    Generator --> Answer
```

## ✨ 주요 기능

-   **멀티 소스 수집**: FMVSS(미국), KMVSS(한국), ECE(유럽) 등 다양한 규격의 문서를 통합 처리합니다.
-   **하이브리드 검색 (Hybrid Search)**: **BM25(키워드 매칭)**와 **Vector Search(의미적 유사도)**를 결합하여 검색 정확도를 극대화했습니다.
-   **고급 재순위화 (Reranking)**: **FlashRank(Cross-Encoder)**를 사용하여 1차 검색된 문서들의 연관성을 다시 정밀하게 평가합니다.
-   **메타데이터 강화**: 텍스트 분할 시 문맥이 손실되지 않도록, 각 청크에 '규정 ID', '제목' 등의 정보를 자동으로 주입합니다.
-   **증분 업데이트**: 파일 해시(SHA256)를 추적하여 변경된 법규 파일만 지능적으로 재처리합니다.
-   **출처 기반 답변**: LLM이 답변 생성 시 반드시 근거가 되는 규정 조항(예: `[Source: FMVSS 108 S7.3]`)을 인용하도록 설계되었습니다.

## 🛠️ 기술 스택 (Tech Stack)

-   **언어**: Python 3.10+
-   **LLM**: Google Gemini 2.5 (via `langchain-google-genai`)
-   **프레임워크**: LangChain
-   **벡터 저장소**: ChromaDB (Local)
-   **임베딩**: HuggingFace (`sentence-transformers`)
-   **재순위화**: FlashRank
-   **UI**: Streamlit
-   **크롤링**: Selenium & BeautifulSoup

## 🚀 설치 및 실행 방법

### 사전 요구사항
-   Python 3.10 이상
-   Git
-   Google Cloud API Key (Gemini 사용 목적)

### 설치 단계

1.  **리포지토리 복제 (Clone)**
    ```bash
    git clone https://github.com/yangduhi/reg_RAG.git
    cd reg_RAG
    ```

2.  **가상환경 생성 및 활성화**
    ```bash
    # Windows
    python -m venv venv
    .\venv\Scripts\activate

    # Mac/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **의존성 패키지 설치**
    ```bash
    pip install -r requirements.txt
    ```

4.  **환경 설정 (.env)**
    프로젝트 루트 경로에 `.env` 파일을 생성하고 아래 내용을 입력하세요.
    ```ini
    GOOGLE_API_KEY=your_google_api_key_here
    # 선택 사항 (기본값 사용 시 생략 가능)
    LLM_MODEL_NAME=gemini-2.5-flash
    EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
    ```

## 💻 사용법 (Usage)

### 1. 데이터 수집 및 DB 구축
제공된 법규 데이터(XML, PDF)를 벡터 DB에 적재해야 합니다. 파이프라인은 자동으로 변경된 파일만 처리합니다.

```bash
python -m src.ingestion.pipeline
```

### 2. 애플리케이션 실행
Streamlit 웹 인터페이스를 실행하여 RAG 시스템과 대화합니다.

## 📂 프로젝트 구조

```
reg_RAG/
├── config/              # 설정 파일 디렉토리
├── data/                # 원본 및 처리된 데이터 (Git 제외됨)
├── scripts/             # 유틸리티 스크립트
│   └── check_vector_db.py # DB 인덱싱 상태 확인
├── src/                 # 소스 코드
│   ├── core/            # 핵심 설정 및 로깅
│   ├── ingestion/       # 데이터 수집 파이프라인
│   │   ├── loaders.py   # 파일 파서 (XML, PDF)
│   │   └── pipeline.py  # 수집 로직 (Loader -> Splitter -> DB)
│   ├── rag/             # 검색 및 생성 로직
│   │   ├── engine.py    # RAG 엔진 파사드 (Main Logic)
│   │   └── vectorstore.py
│   └── interface/       # Streamlit UI 코드
├── .gitignore           # Git 제외 규칙
├── requirements.txt     # Python 라이브러리 목록
└── README.md            # 프로젝트 문서
```

## 🤝 기여 (Contribution)

이 프로젝트에 기여하고 싶다면 다음 절차를 따라주세요:
1.  Fork를 생성합니다.
2.  기능 브랜치를 만듭니다 (`git checkout -b feature/NewFeature`).
3.  변경 사항을 커밋합니다 (`git commit -m 'Add NewFeature'`).
4.  브랜치에 푸시합니다 (`git push origin feature/NewFeature`).
5.  Pull Request를 생성합니다.

## 📄 라이선스 (License)

이 프로젝트는 MIT 라이선스 하에 배포됩니다.