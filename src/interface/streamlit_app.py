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
# [1] UI 및 로깅 설정 (v2 스타일 복구)
# ---------------------------------------------------------
st.set_page_config(
    page_title="Global Auto Regulations AI (v2.2)",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .stChatInput { padding-bottom: 2rem; }

    /* Markdown 테이블 스타일 개선 */
    table {
        width: 100%;
        border-collapse: collapse;
        margin-bottom: 1rem;
    }
    th, td {
        padding: 10px;
        text-align: left;
        border-bottom: 1px solid #444; /* 다크모드 대응 */
        line-height: 1.6;
    }
    th {
        background-color: rgba(255, 255, 255, 0.05);
        font-weight: bold;
    }
    /* 사이드바 라디오 버튼 박스 스타일 */
    .stRadio > div {
        background-color: rgba(255, 255, 255, 0.1);
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# [2] 엔진 초기화 (캐싱)
# ---------------------------------------------------------
@st.cache_resource
def get_engine():
    try:
        return RAGEngine()
    except Exception as e:
        st.error(f"❌ 엔진 초기화 중 치명적 오류 발생: {e}")
        return None


# ---------------------------------------------------------
# [3] 메인 애플리케이션
# ---------------------------------------------------------
def main():
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "안녕하세요! 🇺🇸FMVSS, 🇪🇺ECE, 🇰🇷KMVSS 자동차 규정 전문가 AI입니다.\n\n궁금한 규정이나 비교하고 싶은 내용을 물어보세요.",
            }
        ]

    engine = get_engine()
    if not engine:
        st.stop()

    # ==========================================
    # [사이드바] 설정 및 도구 (v2 기능 100% 복구)
    # ==========================================
    with st.sidebar:
        st.header("⚙️ Regulations Tool")

        if len(st.session_state.messages) > 1:
            chat_str = "\n\n".join(
                [
                    f"[{m['role'].upper()}]\n{m['content']}"
                    for m in st.session_state.messages
                ]
            )
            st.download_button(
                "💾 대화 내용 저장 (.txt)", chat_str, "chat_history.txt", "text/plain"
            )

        st.markdown("---")

        # 필터 로직
        st.subheader("🎯 검색 범위 설정")
        all_stds = engine.get_available_standards()

        usa_stds, eu_stds, kr_stds = [], [], []
        for std in all_stds:
            s_str = str(std)
            s_str_lower = s_str.lower()

            if 'kmvss' in s_str_lower or ('제' in s_str and '조' in s_str):
                kr_stds.append(std)
            elif s_str_lower.startswith('ece') or "un r" in s_str_lower:
                eu_stds.append(std)
            elif re.match(r'^\d', s_str): # FMVSS starts with a digit
                usa_stds.append(std)
        
        # 정렬 로직 (한국 규정은 숫자 기준 정렬 시도)
        kr_stds.sort(
            key=lambda x: int(re.findall(r"\d+", str(x))[0])
            if re.findall(r"\d+", str(x))
            else 0
        )
        usa_stds.sort()
        eu_stds.sort()

        region_filter = st.radio(
            "1️⃣ 지역 선택",
            ["전체 (All)", "🇰🇷 한국 (KMVSS)", "🇺🇸 북미 (FMVSS)", "🇪🇺 유럽 (ECE)"],
            index=0,
        )

        filtered_list = all_stds
        if "한국" in region_filter:
            filtered_list = kr_stds
        elif "북미" in region_filter:
            filtered_list = usa_stds
        elif "유럽" in region_filter:
            filtered_list = eu_stds

        def format_func_dynamic(x):
            if x == "All":
                return (
                    f"{region_filter.split(' ')[1]} 전체 검색"
                    if region_filter != "전체 (All)"
                    else "전체 규정 검색"
                )
            title = engine.get_metadata_title(str(x))
            display_title = title if len(title) <= 30 else title[:30] + "..."
            if not display_title or display_title == "No Title":
                display_title = ""
            else:
                display_title = f": {display_title}"
            return f"{x} {display_title}"

        selected_std = st.selectbox(
            "2️⃣ 세부 규정 선택", ["All"] + filtered_list, format_func=format_func_dynamic
        )

        st.markdown("---")

        st.subheader("💾 데이터베이스 관리")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 증분 갱신"):
                status = st.status("갱신 진행 중...", expanded=True)
                try:
                    import asyncio
                    msg = asyncio.run(engine.run_pipeline(force_refresh=False))
                    status.update(label="갱신 완료!", state="complete", expanded=False)
                    st.success(msg)
                    time.sleep(1)
                    st.rerun()
                except Exception as e:
                    status.update(label="오류 발생", state="error")
                    st.error(f"실패: {e}")

        with col2:
            if st.button("💥 전체 재구축", type="primary"):
                if st.checkbox("데이터 삭제 확인", key="confirm_reset"):
                    status = st.status("전체 재구축 중...", expanded=True)
                    try:
                        if settings.VECTOR_DB_PATH.exists():
                            shutil.rmtree(settings.VECTOR_DB_PATH)
                        if settings.DB_STATE_PATH.exists():
                            os.remove(settings.DB_STATE_PATH)
                        import asyncio
                        msg = asyncio.run(engine.run_pipeline(force_refresh=True))
                        status.update(
                            label="재구축 완료!", state="complete", expanded=False
                        )
                        st.success("DB가 재구축되었습니다.")
                        time.sleep(1)
                        st.rerun()
                    except Exception as e:
                        status.update(label="오류 발생", state="error")
                        st.error(f"실패: {e}")
                else:
                    st.warning("체크박스를 선택해주세요.")

        if engine.vector_store:
            try:
                st.caption(
                    f"📚 학습된 문서 청크: {engine.vector_store._collection.count():,}개"
                )
            except:
                pass
        else:
            st.error("⚠️ DB 없음")

        st.markdown("---")
        with st.expander("🛠️ 고급 검색 옵션"):
            retriever_k = st.slider("참고 문서 수 (K)", 3, 20, settings.RETRIEVER_K)
            use_mmr = st.toggle("MMR 검색 (다양성)", value=settings.USE_MMR)

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
            st.error(
                "데이터베이스가 준비되지 않았습니다. 사이드바에서 [전체 재구축]을 먼저 실행해주세요."
            )
            return

        with st.chat_message("assistant"):
            container = st.empty()
            full_res = ""
            history = [m for m in st.session_state.messages if m["role"] != "system"][
                -6:
            ]
            hist_str = "\n".join([f"{m['role']}: {m['content']}" for m in history])

            with st.status("🧠 질의 분석 및 검색 중...", expanded=False) as status:
                try:
                    # [수정된 부분] 동적으로 검색어를 생성하는 새로운 프롬프트
                    q_prompt = f'''
                    You are an expert search query generator for a vector database of automotive regulations.
                    Analyze the user's question and generate a single line of space-separated keywords.

                    **User Question:** "{prompt}"

                    **Instructions:**
                    1.  **Identify Regions/Regulations:** From the user's question, identify which regulations are relevant (e.g., "한국", "미국", "유럽", "KMVSS", "FMVSS", "ECE").
                    2.  **Identify Core Topics:** Extract the main technical subjects (e.g., "정면 충돌", "상해 기준", "pedestrian protection").
                    3.  **Translate & Expand:**
                        - For Korea/KMVSS: Include Korean terms like "자동차안전기준" and the topic in Korean.
                        - For US/FMVSS: Include English terms like "FMVSS" and the topic in English.
                        - For Europe/ECE: Include English terms like "ECE Regulation" and the topic in English.
                    4.  **Combine:** Create a single line of keywords. Add synonyms or related terms if it helps.
                    5.  **Output Format:** Output **only** the keywords, separated by spaces. No labels, no explanations.

                    **Example:**
                    *User Question:* "한국과 미국의 정면 충돌 상해 기준을 알려줘."
                    *Your Output:* 정면충돌 상해기준 KMVSS 자동차안전기준 FMVSS frontal impact injury criteria

                    **Your Output:**
                    '''

                    search_q = engine.llm.invoke(q_prompt).content.strip().replace('"', '')
                    st.write(f"🔍 생성된 검색어: `{search_q}`")

                    retriever = engine.get_ensemble_retriever(k=retriever_k, use_mmr=use_mmr, filter_std=selected_std)
                    docs = retriever.invoke(search_q)
                    st.write(f"📄 관련 문서 {len(docs)}건 확보")

                    if not docs:
                        context_text = ""; st.warning("검색 결과가 없습니다.")
                    else:
                        context_text = "\n\n".join([f"[[{d.metadata.get('standard_id')}]] {d.page_content}" for d in docs])

                    status.update(label="검색 완료", state="complete")
                except Exception as e: status.update(label="검색 오류", state="error"); st.error(f"Error: {e}"); return

            # [답변 프롬프트 유지]
            if not context_text:
                full_res = "죄송합니다. 관련 문서를 찾을 수 없습니다."
                container.markdown(full_res)
            else:
                qa_prompt = f"""
                You are a senior expert in Automotive Regulations.
                Answer the user's question based on the Context and Conversation History.

                IMPORTANT RULES:
                1. **DIRECT ANSWER:** Start with the answer immediately.
                2. **Language:** Answer in **Korean**.
                3. **Citations:** Explicitly cite the regulation number.
                4. **Format (CRITICAL):**
                   - Use standard Markdown Tables for lists or comparisons.
                   - **Structure:** `| Header 1 | Header 2 |` followed by `|---|---|` separator.
                   - **Newlines:** Ensure EACH ROW is on a NEW LINE. Do NOT collapse rows.
                   - **Cells:** Use `<br>` for line breaks inside a cell. Do NOT use HTML list tags.

                Conversation History:
                {hist_str}

                Context:
                {context_text}

                Question: {prompt}
                Answer:"""

                try:
                    for chunk in engine.llm.stream(qa_prompt):
                        full_res += chunk.content
                        container.markdown(full_res + "▌", unsafe_allow_html=True)
                    container.markdown(full_res, unsafe_allow_html=True)
                except WebSocketClosedError:
                    pass
                except Exception as e:
                    st.error(f"Generation Error: {e}")

            with st.expander("📚 답변 근거 문서 (Source)"):
                if docs:
                    seen = set()
                    for d in docs:
                        h = hash(d.page_content)
                        if h in seen:
                            continue
                        seen.add(h)
                        std = d.metadata.get("standard_id", "Unknown")
                        src = d.metadata.get("source_file", "")
                        header = f"**{std}**"
                        if "ECE" in src:
                            header = f"🇪🇺 **ECE {std}**"
                        elif "KMVSS" in src or "제" in str(std):
                            header = f"🇰🇷 **KMVSS {std}**"
                        else:
                            header = f"🇺🇸 **FMVSS {std}**"
                        st.markdown(f"**{header}**")
                        st.caption(d.page_content[:400].replace("\n", " ") + "...")
                        if url := engine.get_web_url(std):
                            st.link_button("🌐 원문 보기", url)
                        else:
                            st.caption(f"파일명: {src}")
                        st.divider()
                else:
                    st.info("근거 문서를 찾지 못했습니다.")

            st.session_state.messages.append({"role": "assistant", "content": full_res})


if __name__ == "__main__":
    main()
