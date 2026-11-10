"""Streamlit interface for Signal Search."""

from __future__ import annotations

import os
import uuid
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import streamlit as st
from dotenv import load_dotenv

from agent import SearchAgent, ThreadRegistry, open_checkpointer


load_dotenv()

ROOT = Path(__file__).resolve().parent
DATABASE_PATH = ROOT / "data" / "agent.db"
APP_TIMEZONE = os.getenv("APP_TIMEZONE", "Asia/Kolkata")


st.set_page_config(
    page_title="Signal Search",
    page_icon="◉",
    layout="centered",
    initial_sidebar_state="auto",
)

st.markdown(
    """
    <style>
    .block-container { max-width: 850px; padding-top: 2rem; }
    [data-testid="stSidebar"] { border-right: 1px solid rgba(128,128,128,.18); }
    .signal-kicker { color: #6c7cff; font-weight: 700; letter-spacing: .08em; font-size: .75rem; }
    .signal-subtitle { color: #7b8190; margin-top: -.6rem; margin-bottom: 1.5rem; }
    .timestamp { color: #8b91a1; font-size: .72rem; margin-top: -.4rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def _secret(name: str) -> str | None:
    value = os.getenv(name)
    if value:
        return value
    try:
        return st.secrets.get(name)
    except Exception:
        return None


def _now() -> str:
    return datetime.now(ZoneInfo(APP_TIMEZONE)).isoformat(timespec="seconds")


def _display_time(value: str | None) -> str:
    if not value:
        return ""
    try:
        parsed = datetime.fromisoformat(value)
        return parsed.strftime("%d %b %Y, %I:%M %p %Z").strip()
    except ValueError:
        return value


@st.cache_resource
def _registry() -> ThreadRegistry:
    return ThreadRegistry(DATABASE_PATH)


@st.cache_resource
def _agent(api_key: str, model: str) -> SearchAgent:
    checkpointer, _connection = open_checkpointer(DATABASE_PATH)
    return SearchAgent.create(checkpointer=checkpointer, api_key=api_key, model=model)


registry = _registry()
api_key = _secret("GROQ_API_KEY")
model = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

if "thread_id" not in st.session_state:
    st.session_state.thread_id = uuid.uuid4().hex

with st.sidebar:
    st.markdown("### Conversations")
    if st.button("＋ New chat", use_container_width=True, type="primary"):
        st.session_state.thread_id = uuid.uuid4().hex
        st.rerun()

    for item in registry.list_threads():
        label = item["title"][:36]
        if st.button(label, key=f"thread-{item['thread_id']}", use_container_width=True):
            st.session_state.thread_id = item["thread_id"]
            st.rerun()

    st.divider()
    st.caption("One tool: Groq browser search")
    st.caption(f"Model: {model}")

st.markdown('<div class="signal-kicker">ONE-TOOL RESEARCH AGENT</div>', unsafe_allow_html=True)
st.title("Signal Search")
st.markdown(
    '<div class="signal-subtitle">Fast web answers with sources — and a memory for what you asked and when.</div>',
    unsafe_allow_html=True,
)

if not api_key:
    st.warning(
        "Add `GROQ_API_KEY` to a local `.env` file or Streamlit secrets, then restart the app. "
        "Copy `.env.example` to get started."
    )

agent = _agent(api_key, model) if api_key else None
history = agent.history(st.session_state.thread_id) if agent else []

if not history:
    st.info("Try: **What changed in AI this week? Cite the most reliable sources.**")

for message in history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "user" and message.get("asked_at"):
            st.markdown(
                f'<div class="timestamp">Asked {_display_time(message["asked_at"])}</div>',
                unsafe_allow_html=True,
            )

question = st.chat_input("Ask anything…", disabled=agent is None)
if question and agent:
    asked_at = _now()
    title = question.strip().replace("\n", " ")[:55] or "New conversation"
    registry.touch(thread_id=st.session_state.thread_id, title=title, timestamp=asked_at)

    with st.chat_message("user"):
        st.markdown(question)
        st.markdown(
            f'<div class="timestamp">Asked {_display_time(asked_at)}</div>',
            unsafe_allow_html=True,
        )

    with st.chat_message("assistant"):
        with st.spinner("Searching and checking sources…"):
            try:
                answer = agent.ask(
                    thread_id=st.session_state.thread_id,
                    question=question,
                    asked_at=asked_at,
                )
            except Exception as exc:
                error_text = str(exc).lower()
                if "429" in error_text or "rate" in error_text:
                    st.error("Groq's free-tier rate limit was reached. Wait briefly and try again.")
                else:
                    st.error("The request failed. Check the API key and try again.")
                st.stop()
        st.markdown(answer)

