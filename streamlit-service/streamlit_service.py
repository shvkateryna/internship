import os
import uuid
import json
import asyncio
import base64
import mimetypes
import html as htmlesc
from pathlib import Path
from typing import List, Dict

import streamlit as st
from langchain_mcp_adapters.client import MultiServerMCPClient

PINK_BG = "#ffd6d9"
PINK_ACCENT = "#ef8d93"
CREAM_STAR = "#F6D67A"
CARD_BG = "rgba(255,255,255,0.78)"

CLIENT_MCP_URL = os.getenv("CLIENT_MCP_URL", "http://client-service:8000/mcp/")
APP_TITLE = os.getenv("APP_TITLE", "Tasia-Perekladasia")

BOT_AVATAR_PATH = os.getenv("BOT_AVATAR", "assets/tasia.png")
USER_AVATAR_PATH = os.getenv("USER_AVATAR", "assets/user.png")

st.set_page_config(page_title = APP_TITLE, page_icon = "⭐", layout = "centered")

def load_css(path: str) -> None:
    """
    load styles
    """
    css = Path(path).read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

def file_to_data_uri(path: str) -> str:
    """
    show avatar uri's
    """
    try:
        bytes_file = Path(path).read_bytes()
        mime = mimetypes.guess_type(path)[0] or "image/png"
        return f"data:{mime};base64,{base64.b64encode(bytes_file).decode('ascii')}"
    except Exception:
        return ""

def render_chat_html(messages: List[Dict[str, str]], user_avatar_src: str, bot_avatar_src: str) -> str:
    """
    show the chat
    """
    parts = []
    for message in messages:
        role = "user" if message.get("role") == "user" else "assistant"
        avatar = user_avatar_src if role == "user" else bot_avatar_src
        text = message.get("content", "")
        text = text if isinstance(text, str) else json.dumps(text, ensure_ascii = False, indent = 2)
        safe_text = htmlesc.escape(text).replace("\n", "<br>")
        parts.append(
            f'<div class="msg {role}"><img class="avatar" src="{avatar}" alt="{role}"><div class="bubble">{safe_text}</div></div>'
        )
    return '<div class="container-narrow"><div class="chat-wrapper">' + "".join(parts) + "</div></div>"

load_css("styles.css")

st.markdown(
    "<style>"
    ".chat-wrapper{max-height:60vh;overflow-y:auto;padding-right:6px;scroll-behavior:smooth}"
    ".msg{display:flex;align-items:flex-start;gap:10px;margin:10px 0}"
    ".msg .avatar{width:32px;height:32px;border-radius:50%;object-fit:cover;border:1px solid #ffffffaa}"
    ".msg .bubble{background:var(--card-bg);border:1px solid #ffffffaa;border-radius:18px;padding:8px 12px;"
    "box-shadow:0 8px 20px rgba(0,0,0,0.05);max-width:100%;word-wrap:break-word}"
    ".msg.user .bubble{border-color:#ef8d9333}"
    "</style>",
    unsafe_allow_html=True,
)

if "chat" not in st.session_state:
    st.session_state.chat: List[Dict[str, str]] = []
if "session_id" not in st.session_state:
    st.session_state.session_id = f"web-{uuid.uuid4().hex[:8]}"

USER_AVATAR = file_to_data_uri(USER_AVATAR_PATH)
BOT_AVATAR  = file_to_data_uri(BOT_AVATAR_PATH)

@st.cache_resource(show_spinner = False)
def get_mcp_client(mcp_url: str) -> MultiServerMCPClient:
    """
    get MCP client
    """
    servers = {"client": {"transport": "streamable_http", "url": mcp_url}}
    return MultiServerMCPClient(servers)

@st.cache_resource(show_spinner = False)
def get_ask_tool(mcp_url: str):
    """
    get ask tool
    """
    client = get_mcp_client(mcp_url)
    tools = asyncio.run(client.get_tools())
    return next((t for t in tools if t.name == "ask"), None)

async def ask_async(text: str, sid: str) -> str:
    ask_tool = get_ask_tool(CLIENT_MCP_URL)
    if ask_tool is None:
        return "Помилка: інструмент 'ask' не знайдено на client-service."
    try:
        res = await ask_tool.ainvoke({"input": text, "session_id": sid})
        return res if isinstance(res, str) else json.dumps(res, ensure_ascii=False)
    except Exception as error:
        return f"Помилка виклику агента: {error}"

def ask_via_mcp(text: str, sid: str) -> str:
    try:
        return asyncio.run(ask_async(text, sid))
    except RuntimeError:
        try:
            import nest_asyncio
            nest_asyncio.apply()
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(ask_async(text, sid))
        except Exception as error:
            return f"Помилка виклику агента: {error}"

st.markdown('<div class="container-narrow card">'
            '<div style="display:flex;align-items:center;gap:10px;justify-content:center;margin-bottom:8px;">'
            '<span style="font-size:26px;">⭐</span>'
            '<h1 style="margin:0;font-size:28px;">Tasia-Perekladаsia</h1>'
            '<span style="font-size:26px;">⭐</span>'
            '</div></div>', unsafe_allow_html = True)

st.markdown(render_chat_html(st.session_state.chat, USER_AVATAR, BOT_AVATAR), unsafe_allow_html=True)

st.markdown(
    "<script>"
    "const boxes=window.parent.document.getElementsByClassName('chat-wrapper');"
    "if(boxes&&boxes.length){const box=boxes[boxes.length-1];box.scrollTop=box.scrollHeight;}"
    "</script>",
    unsafe_allow_html=True,
)

st.markdown('<div class="container-narrow card" style="margin-top:16px;">', unsafe_allow_html=True)
with st.form("ask-form", clear_on_submit=True):
    q = st.text_area("Запит", placeholder="Напиши сюди свій запит…")
    submitted = st.form_submit_button("Надіслати", type="primary")
    if submitted:
        user_text = (q or "").strip()
        if not user_text:
            st.warning("Введи запит, будь ласка.")
        else:
            st.session_state.chat.append({"role": "user", "content": user_text})
            with st.spinner("Думаю над відповіддю…"):
                answer = ask_via_mcp(user_text, st.session_state.session_id)
            st.session_state.chat.append({"role": "assistant", "content": str(answer)})
            st.rerun()
st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="container-narrow" style="margin-top:18px;">', unsafe_allow_html=True)
with st.expander("Налаштування", expanded=False):
    st.caption(f"MCP endpoint: {CLIENT_MCP_URL}")
    st.session_state.session_id = st.text_input(
        "Session ID",
        value=st.session_state.session_id,
        help="Ідентифікатор діалогу для збереження контексту",
    )
    if st.button("Очистити чат"):
        st.session_state.chat.clear()
        st.toast("Історію очищено.", icon="🧹")
        st.rerun()
st.markdown("</div>", unsafe_allow_html=True)
