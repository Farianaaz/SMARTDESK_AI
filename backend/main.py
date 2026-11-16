import logging
import os
import threading
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

from agent import chat, reset

load_dotenv()
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if os.getenv("USE_MOCK_JIRA", "true").lower() != "true":
        import jira_mcp

        threading.Thread(target=jira_mcp.warm_up, daemon=True).start()
    yield


app = FastAPI(title="SmartDesk AI", lifespan=lifespan)


class ChatRequest(BaseModel):
    session_id: str
    message: str


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/chat")
def chat_endpoint(req: ChatRequest):
    try:
        return chat(req.session_id, req.message)
    except Exception:
        log.exception("Chat failed")
        return {"reply": "Sorry, something went wrong on my side. Please try again.",
                "intent": None, "awaiting_input": False}


@app.post("/api/reset/{session_id}")
def reset_endpoint(session_id: str):
    reset(session_id)
    return {"status": "reset"}