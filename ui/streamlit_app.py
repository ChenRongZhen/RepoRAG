"""RepoRAG 网页演示：和 CLI 共用同一张图，只负责展示。

运行：python -m streamlit run ui/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import ConfigError, get_settings  # noqa: E402
from app.graph import default_graph  # noqa: E402
from app.prompts import REFUSAL_TEXT  # noqa: E402

st.set_page_config(page_title="RepoRAG · CoreCoder 文档问答", page_icon="📚", layout="wide")


@st.cache_resource(show_spinner="正在加载索引与工作流……")
def load_graph(no_github: bool, fake: bool, top_k: int):
    return default_graph(no_github=no_github, fake=fake, top_k=top_k)


def render_sources(sources: list[str]) -> None:
    if not sources:
        return
    with st.expander(f"来源（{len(sources)}）"):
        for source in sources:
            if source.startswith("http"):
                st.markdown(f"- [Issue]({source})")
            else:
                st.markdown(f"- `{source}`")


def render_trace(trace: list[str]) -> None:
    with st.expander("执行轨迹"):
        st.code("\n".join(trace) or "（无）")


with st.sidebar:
    st.header("设置")
    fake = st.toggle("离线演示（不调用模型）", value=False)
    no_github = st.toggle("关闭 GitHub Issue 检索", value=False)
    top_k = st.slider("检索片段数", 1, 5, 3)
    st.caption("密钥只从项目根目录的 .env 读取。")

try:
    settings = get_settings()
    if not fake:
        settings.validate_for_llm()
except ConfigError as exc:
    st.error(f"配置不完整：{exc}")
    st.stop()

graph = load_graph(no_github, fake, top_k)

st.title("RepoRAG")
st.caption("面向 CoreCoder 文档的问答：答案带出处；资料不足时查 Issue；没依据就拒答。")

if "history" not in st.session_state:
    st.session_state.history = []

for turn in st.session_state.history:
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        st.markdown(turn["answer"])
        render_sources(turn["sources"])
        render_trace(turn["trace"])

question = st.chat_input("问 CoreCoder 的循环、工具、上下文压缩……")
if question:
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"), st.spinner("检索与生成中……"):
        try:
            state = graph.invoke({"question": question, "trace": []})
        except Exception as exc:  # noqa: BLE001
            st.error(f"运行失败：{exc}")
            st.stop()
        answer = state.get("answer", REFUSAL_TEXT)
        st.markdown(answer)
        sources = state.get("sources", [])
        render_sources(sources)
        if st.toggle("显示执行轨迹", value=False, key=f"trace-{len(st.session_state.history)}"):
            render_trace(state.get("trace", []))
        st.session_state.history.append(
            {"question": question, "answer": answer,
             "sources": sources, "trace": state.get("trace", [])}
        )
