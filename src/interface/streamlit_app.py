import asyncio
import os
import re
import shutil
import sys
import time
from pathlib import Path

# [경로 설정]
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

import streamlit as st
from tornado.websocket import WebSocketClosedError

from src.core.config import settings
from src.rag.engine import RAGEngine

# ---------------------------------------------------------
# [1] UI 및 로깅 설정
# ---------------------------------------------------------
st.set_page_config(
    page_title="Global Auto Regulations AI (v3.0 - Modular RAG)",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
# [2] 엔진 초기화 (비동기 지원)
# ---------------------------------------------------------
@st.cache_resource
def get_engine():
    """
    RAGEngine을 초기화하고 반환합니다.
    Streamlit의 캐싱을 사용하여 재실행 시에도 인스턴스를 유지합니다.
    """
    try:
        # 엔진 생성 시 내부적으로 비동기 태스크가 시작됨
        engine = RAGEngine()
        return engine
    except Exception as e:
        st.error(f"❌ 엔진 초기화 중 치명적 오류 발생: {e}")
        return None

async def ensure_engine_initialized(engine):
    """엔진의 비동기 초기화 작업이 완료될 때까지 대기"""
    if engine and not engine.is_initialized:
        with st.spinner("🚀 검색 엔진 초기화 중... (데이터 로딩)"):
            await engine.initialization_task


# ---------------------------------------------------------
# [3] 메인 애플리케이션
# ---------------------------------------------------------
async def main():
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
    # [사이드바] 설정 및 도구
    # ==========================================
    with st.sidebar:
        st.header("⚙️ Regulations Tool")

        if len(st.session_state.messages) > 1:
            chat_str = "\n\n".join(
                [f"[{m['role'].upper()}]\n{m['content']}" for m in st.session_state.messages]
            )
            st.download_button("💾 대화 내용 저장 (.txt)", chat_str, "chat_history.txt", "text/plain")

        st.markdown("---")
        
        # [관리 도구]
        st.subheader("💾 데이터베이스 관리")
        
        # 증분 갱신 핸들러
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

        # 전체 재구축 핸들러
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
                # 주의: engine 인스턴스가 불안정할 수 있으므로, 새로운 파이프라인 인스턴스로 실행 권장
                # 하지만 여기서는 engine 메서드를 사용해야 하므로 시도.
                # 만약 실패하면 pipeline 모듈을 직접 import해서 실행해야 함.
                
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
            # 체크박스 상태를 먼저 확인하고 버튼을 활성화하는 것이 UX상 좋음
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

    # --- 메인 채팅 ---
    st.title("⚖️ Global Auto Regulations AI")
    
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"], unsafe_allow_html=True)

    if prompt := st.chat_input("질문을 입력하세요..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        if not engine.vector_store:
            st.error("데이터베이스가 준비되지 않았습니다. 사이드바에서 [전체 재구축]을 먼저 실행해주세요.")
            return

        with st.chat_message("assistant"):
            container = st.empty()
            
            with st.status("🧠 LangGraph 실행 중... (검색 -> 평가 -> 생성)", expanded=True) as status:
                try:
                    # LangGraph 실행 (엔진에게 위임)
                    response = await engine.chat(prompt)
                    
                    status.update(label="답변 생성 완료", state="complete", expanded=False)
                    container.markdown(response, unsafe_allow_html=True)
                    
                    # 채팅 기록 저장
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                except Exception as e:
                    status.update(label="실행 오류", state="error")
                    st.error(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
