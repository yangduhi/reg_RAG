import asyncio
import os
import re
import shutil
import sys
import time
import uuid
from pathlib import Path

# [경로 설정] 프로젝트 루트 디렉토리를 sys.path에 추가하여 모듈 import를 지원
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

import streamlit as st
import streamlit.components.v1 as components
from tornado.websocket import WebSocketClosedError

from src.core.config import settings
from src.rag.engine import RAGEngine

# ---------------------------------------------------------
# [1] UI 및 로깅 설정 (UI & Logging Configuration)
# ---------------------------------------------------------
st.set_page_config(
    page_title="Global Auto Regulations AI (v3.0 - Modular RAG)",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 커스텀 CSS 적용: 채팅 입력창 하단 여백, 테이블 스타일 등
st.markdown(
    """
<style>
    .stChatInput { padding-bottom: 2rem; }
    table { width: 100%; border-collapse: collapse; margin-bottom: 1rem; }
    th, td { padding: 10px; text-align: left; border-bottom: 1px solid #444; line-height: 1.6; }
    th { background-color: rgba(255, 255, 255, 0.05); font-weight: bold; }
    .stRadio > div { background-color: rgba(255, 255, 255, 0.1); padding: 10px; border-radius: 5px; margin-bottom: 10px; }
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# [2] 엔진 초기화 (Engine Initialization)
# ---------------------------------------------------------
@st.cache_resource
def get_engine():
    """
    RAGEngine을 초기화하고 반환합니다.
    Streamlit의 캐싱(@st.cache_resource)을 사용하여 앱 재실행 시에도 인스턴스를 유지합니다.
    이로 인해 불필요한 모델 로딩 시간을 줄일 수 있습니다.
    """
    try:
        # 엔진 생성 시 내부적으로 비동기 초기화 태스크가 시작됨
        engine = RAGEngine()
        return engine
    except Exception as e:
        st.error(f"❌ 엔진 초기화 중 치명적 오류 발생: {e}")
        return None

async def ensure_engine_initialized(engine):
    """
    엔진의 비동기 초기화 작업(검색기 로딩 등)이 완료될 때까지 대기합니다.
    앱 실행 초기 단계에서 필수 리소스가 준비되었는지 확인합니다.
    """
    if engine and not engine.is_initialized:
        with st.spinner("🚀 검색 엔진 초기화 중... (데이터 로딩)"):
            await engine.initialization_task


# ---------------------------------------------------------
# [3] 메인 애플리케이션 (Main Application)
# ---------------------------------------------------------
async def main():
    # [세션 상태 초기화] 채팅 기록 저장소
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "안녕하세요! 🇺🇸FMVSS, 🇪🇺ECE, 🇰🇷KMVSS 자동차 규정 전문가 AI입니다.\n\n궁금한 규정이나 비교하고 싶은 내용을 물어보세요. (LangGraph 기반 Adaptive RAG 적용)",
            }
        ]

    engine = get_engine()
    if not engine:
        st.stop()
    
    # 초기화 대기 (비동기)
    await ensure_engine_initialized(engine)

    # ==========================================
    # [사이드바] 설정 및 도구 (Sidebar)
    # ==========================================
    with st.sidebar:
        st.header("⚙️ Regulations Tool")
        
        # [검색 설정] 사용자 맞춤형 검색 옵션 제공
        st.subheader("🌍 검색 범위 설정 (Region)")
        region_cols = st.columns(3)
        with region_cols[0]:
            search_us = st.checkbox("🇺🇸 US", value=True)
        with region_cols[1]:
            search_kr = st.checkbox("🇰🇷 KR", value=True)
        with region_cols[2]:
            search_eu = st.checkbox("🇪🇺 EU", value=True)
            
        st.subheader("🎯 검색 정확도 설정")
        keyword_accuracy = st.slider(
            "유사성 임계값 (Similarity Threshold)", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.5, 
            step=0.1,
            help="낮을수록 더 많은 문서를 검색하지만 관련성이 떨어질 수 있습니다."
        )
        st.caption("✨ 설정은 다음 질문부터 적용됩니다.")

        if len(st.session_state.messages) > 1:
            chat_str = "\n\n".join(
                [f"[{m['role'].upper()}]\n{m['content']}" for m in st.session_state.messages]
            )
            st.download_button("💾 대화 내용 저장 (.txt)", chat_str, "chat_history.txt", "text/plain")

        st.markdown("---")
        
        # [관리 도구] 데이터 파이프라인 제어
        st.subheader("💾 데이터베이스 관리")
        
        # 증분 갱신 핸들러 (새로운 파일만 추가)
        async def handle_incremental_update():
            status = st.status("갱신 진행 중...", expanded=True)
            try:
                msg = await engine.run_pipeline(force_refresh=False)
                status.update(label="갱신 완료!", state="complete", expanded=False)
                st.success(msg)
                await asyncio.sleep(1)
                st.rerun()
            except Exception as e:
                status.update(label="오류 발생", state="error")
                st.error(f"실패: {e}")

        # 전체 재구축 핸들러 (DB 초기화 후 재생성)
        async def handle_full_rebuild():
            status = st.status("전체 재구축 중... (잠시만 기다려주세요)", expanded=True)
            try:
                # 1. DB 락 해제를 위해 엔진 리소스 해제 시도
                if engine.vstore_manager:
                    engine.vstore_manager = None
                if engine.bm25_retriever:
                    engine.bm25_retriever = None
                
                # Streamlit 캐시 초기화 (중요: 기존 연결을 끊기 위함)
                st.cache_resource.clear()
                
                # 잠시 대기하여 파일 핸들이 반환되도록 유도
                await asyncio.sleep(1)

                # 2. 파이프라인 강제 실행 (내부적으로 삭제 및 재생성)
                from src.ingestion.pipeline import IngestionPipeline
                pipeline = IngestionPipeline()
                await pipeline.run(force_refresh=True)
                
                status.update(label="재구축 완료!", state="complete", expanded=False)
                st.success("DB가 성공적으로 재구축되었습니다. 페이지를 새로고침합니다.")
                await asyncio.sleep(2)
                st.rerun()
                
            except Exception as e:
                status.update(label="오류 발생", state="error")
                st.error(f"실패 (서버를 껐다 켜주세요): {e}")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 증분 갱신"):
                await handle_incremental_update()

        with col2:
            # 체크박스 상태를 먼저 확인하고 버튼을 활성화하는 것이 UX상 좋음 (실수 방지)
            confirm = st.checkbox("삭제 확인", key="confirm_reset")
            if st.button("💥 전체 재구축", type="primary", disabled=not confirm):
                await handle_full_rebuild()

        if engine.vector_store:
            try:
                st.caption(f"📚 학습된 문서 청크: {engine.vector_store._collection.count():,}개")
            except:
                pass
        else:
            st.error("⚠️ DB 없음")

        if st.button("🗑️ 대화 내용 지우기", type="secondary"):
            st.session_state.messages = []
            st.rerun()

        st.markdown("---")
        st.subheader("🕸️ 지식 그래프 (Knowledge Graph)")
        
        # [지식 그래프] RAG 워크플로우 시각화 파일 로드
        graph_path = Path("graph.html")
        if graph_path.exists():
            with st.expander("그래프 보기", expanded=False):
                try:
                    with open(graph_path, 'r', encoding='utf-8') as f:
                        html_content = f.read()
                    components.html(html_content, height=600, scrolling=True)
                    st.caption("마우스 휠로 확대/축소, 드래그로 이동 가능합니다.")
                except Exception as e:
                    st.error(f"그래프 로드 실패: {e}")
        else:
            st.info("생성된 그래프 파일(graph.html)이 없습니다.")
            if st.button("그래프 생성 (예시)"):
                try:
                    # 워크플로우 그래프 생성 스크립트 실행
                    from scripts.visualize_kg import visualize_knowledge_graph
                    # ... (데이터 정의 생략) ...
                    # 실제로는 스크립트 파일 내용을 실행하거나 함수를 호출해야 함
                    # 여기서는 편의상 생략, 실제 구현은 scripts/visualize_kg.py 참조
                    st.warning("scripts/visualize_kg.py를 직접 실행해주세요.")
                except ImportError:
                    st.error("scripts.visualize_kg 모듈을 찾을 수 없습니다.")
                except Exception as e:
                    st.error(f"생성 실패: {e}")

    # --- 메인 채팅 화면 (Chat Interface) ---
    st.title("⚖️ Global Auto Regulations AI")
    
    # [Helper] 문서 그룹화 및 렌더링 함수
    def render_grouped_documents(documents):
        """
        검색된 문서를 지역별(US, KR, EU) 탭으로 분류하여 카드 형태로 표시합니다.
        가독성을 위해 메타데이터 배지와 접이식 본문(Expander)을 사용합니다.
        """
        if not documents:
            return
            
        # 그룹화 로직
        groups = {"All": [], "🇺🇸 US (FMVSS)": [], "🇰🇷 KR (KMVSS)": [], "🇪🇺 EU (ECE)": []}
        for doc in documents:
            groups["All"].append(doc)
            std_id = str(doc.metadata.get('standard_id', '')).upper()
            src = str(doc.metadata.get('source', '')).upper()
            
            if "FMVSS" in std_id or "571" in std_id or "US" in src:
                groups["🇺🇸 US (FMVSS)"].append(doc)
            elif "KMVSS" in std_id or "KR" in src or "KOREA" in src:
                groups["🇰🇷 KR (KMVSS)"].append(doc)
            elif "ECE" in std_id or "R" in std_id or "EU" in src:
                groups["🇪🇺 EU (ECE)"].append(doc)
        
        # 탭 생성
        tabs = st.tabs(list(groups.keys()))
        
        for i, (label, docs) in enumerate(groups.items()):
            with tabs[i]:
                if not docs:
                    st.info("이 영역에 해당하는 관련 문서가 없습니다.")
                    continue
                    
                for idx, doc in enumerate(docs):
                    meta = doc.metadata
                    std_id = meta.get('standard_id', 'N/A')
                    title = meta.get('title', '')
                    region = meta.get('region', 'Unknown')
                    source = meta.get('source', 'N/A')
                    source_file = meta.get('source_file', 'Unknown File')
                    
                    # 문서 카드 스타일링 (st.container)
                    with st.container(border=True):
                        c1, c2 = st.columns([3, 1])
                        with c1:
                            st.markdown(f"#### 📄 **{std_id}**")
                            if title:
                                st.markdown(f"**{title}**")
                        with c2:
                            st.caption(f"🌍 {region}")
                            st.caption(f"🏛️ {source}")

                        st.divider()
                        
                        # 파일명 및 내용
                        st.caption(f"📂 파일: `{source_file}`")
                        
                        # 긴 본문 내용은 Expander로 숨김 처리
                        with st.expander("📖 원문 내용 보기 (Click to expand)", expanded=False):
                            st.markdown(doc.page_content)

    # [채팅 메시지 표시]
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            content = msg["content"]
            
            # [Source: ...] 하이라이팅 (Regex 활용하여 붉은색 배지 스타일 적용)
            # 예: [Source: KMVSS 102] -> <span ...>[Source: KMVSS 102]</span>
            highlighted_content = re.sub(
                r"(\[Source:.*?\])", 
                r"<span style='color:#ff4b4b; font-weight:bold; background-color:rgba(255, 75, 75, 0.1); padding:2px 6px; border-radius:4px;'>\1</span>", 
                content
            )
            
            st.markdown(highlighted_content, unsafe_allow_html=True)
            
            # [참고 자료 표시] (답변 생성 시 저장된 문서 목록)
            if msg["role"] == "assistant" and "documents" in msg and msg["documents"]:
                with st.expander("📚 참고 자료 확인 (출처)", expanded=False):
                    render_grouped_documents(msg["documents"])

    # [사용자 입력 처리]
    if prompt := st.chat_input("질문을 입력하세요..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        if not engine.vector_store:
            st.error("데이터베이스가 준비되지 않았습니다. 사이드바에서 [전체 재구축]을 먼저 실행해주세요.")
            return

        with st.chat_message("assistant"):
            container = st.empty()
            
            # [상태 표시] LangGraph 실행 과정을 시각적으로 보여줌
            with st.status("🧠 LangGraph 실행 중... (분석 -> 검색 -> 평가 -> 생성)", expanded=True) as status:
                try:
                    # 선택된 리전 리스트 생성 (체크박스 값 기반)
                    selected_regions = []
                    if search_us: selected_regions.append("FMVSS")
                    if search_kr: selected_regions.append("KMVSS")
                    if search_eu: selected_regions.append("ECE")
                    
                    # RAGEngine 호출 (비동기)
                    response_data = await engine.chat(
                        user_question=prompt, 
                        chat_history=st.session_state.messages,
                        search_regions=selected_regions,
                        similarity_threshold=keyword_accuracy
                    )
                    
                    status.update(label="답변 생성 완료", state="complete", expanded=False)
                    
                    # response_data는 {"generation": str, "documents": List[Document]} 형식
                    generation = response_data.get("generation", "")
                    documents = response_data.get("documents", [])
                    
                    # 답변 표시
                    container.markdown(generation, unsafe_allow_html=True)
                    
                    # 상태창 내부에 참고자료 즉시 렌더링 (사용자 피드백 반영)
                    with status:
                        st.markdown("---")
                        st.subheader("📚 답변에 사용된 출처")
                        render_grouped_documents(documents)
                    
                    # 채팅 기록 저장 (documents 포함하여 이력 유지)
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": generation,
                        "documents": documents  # 소스 문서 정보 추가
                    })
                    
                except Exception as e:
                    status.update(label="실행 오류", state="error")
                    st.error(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
