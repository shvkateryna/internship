"""
telegram service
"""

import os
import logging
import asyncio
from typing import Any, Dict
import httpx
from fastapi import FastAPI, Request, Response
from langchain_mcp_adapters.client import MultiServerMCPClient

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_API = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"
CLIENT_URL = os.getenv("CLIENT_URL", "http://client-service:8000/mcp/")
PORT = 8003
HTTP_TIMEOUT = httpx.Timeout(10.0, connect = 3.0)
HTTP_LIMITS = httpx.Limits(max_connections = 50, max_keepalive_connections = 10)

app = FastAPI(title = "telegram-service")

logging.basicConfig(level = os.getenv("LOG_LEVEL","INFO"))
logger = logging.getLogger("telegram-service")

@app.on_event("startup")
async def on_startup():
    """
    initialize the app
    """

    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set")

    app.state.http = httpx.AsyncClient(timeout=HTTP_TIMEOUT, limits=HTTP_LIMITS)

    servers = {"client": {"transport": "streamable_http", "url": CLIENT_URL}}
    app.state.mcp = MultiServerMCPClient(servers)

    tools = await app.state.mcp.get_tools()
    app.state.ask = next((tool for tool in tools if tool.name == "ask"), None)

    if app.state.ask is None:
        raise RuntimeError("client-service MCP tool 'ask' not found")

    app.state.ask_sema = asyncio.Semaphore(20)

@app.on_event("shutdown")
async def on_shutdown():
    """
    close the app
    """

    try:
        await app.state.http.aclose()
    except Exception:
        pass

@app.get("/health")
async def health():
    """
    healthcheck
    """

    try:
        assert app.state.ask is not None
        return {"status": "ok"}
    except Exception as error:
        return {"status": "error", "detail": str(error)}

async def post_telegram(method: str, message: Dict[str, Any], attempts: int = 2) -> Dict[str, Any]:
    """
    handle posts from users and answers from the bot
    """

    if not TELEGRAM_BOT_TOKEN:
        return {"ok": False, "error": "bot_token_missing"}

    url = f"{TELEGRAM_API}/{method}"

    delay = 0.8

    for i in range(attempts):

        try:
            result = await app.state.http.post(url, json = message)

            if result.status_code == 200:
                return result.json()

            if result.status_code in (429, 500, 502, 503, 504):
                await asyncio.sleep(delay)
                delay *= 2
                continue

            return {"ok": False, "status": result.status_code, "text": result.text}

        except httpx.RequestError as error:

            if i == attempts - 1:
                return {"ok": False, "error": f"http_error: {error}"}

            await asyncio.sleep(delay)
            delay *= 2

    return {"ok": False, "error": "unknown"}

async def send_telegram_message(chat_id: int | str, text: str) -> None:
    """
    send message
    """

    await post_telegram("sendMessage", {"chat_id": chat_id, "text": text})

async def ask_agent(text: str, session_id: str) -> str:
    """
    ask agent user's question
    """

    async with app.state.ask_sema:
        try:
            result = await app.state.ask.ainvoke({"input": text, "session_id": session_id})
            return result if isinstance(result, str) else str(result)

        except Exception as error:
            logger.exception("[agent_error] %s", error)
            return "Вибач, сталася помилка. Спробуй, будь ласка, ще раз трохи пізніше."

@app.post("/telegram/webhook")
async def telegram_webhook(request: Request):
    """
    get message from telegram
    """
    try:
        data = await request.json()

    except Exception:
        return Response(status_code = 200)

    message = data.get("message") or data.get("edited_message") or {}
    chat = message.get("chat") or {}
    chat_id = chat.get("id")
    text = (message.get("text") or "").strip()

    if isinstance(chat_id, int) and text.startswith("/start"):
        welcome_text = (
            "Привіт, я Тася! 👋\n\n"
            "Я можу:\n"
            "• Перекладати текст з англійської на українську, якщо ти чітко напишеш, що потрібен переклад.\n"
            "• Відповідати на питання про Катерину.\n\n"
            "Спробуй написати мені будь-що! 😉"
        )
        await send_telegram_message(chat_id, welcome_text)
        return Response(status_code=200)

    if isinstance(chat_id, int) and not text:
        await send_telegram_message(chat_id, "Поки що я розумію лише текстові повідомлення.")
        return Response(status_code = 200)

    if not isinstance(chat_id, int) or not text:
        return Response(status_code = 200)

    reply = await ask_agent(text, session_id = str(chat_id))

    try:
        await send_telegram_message(chat_id, reply)

    except Exception as error:
        logger.warning("[send_error] %s", error)

    return Response(status_code = 200)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host = "0.0.0.0", port = PORT)
