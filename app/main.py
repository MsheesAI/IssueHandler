"""FastAPI app: single /chat endpoint for the GitHub Issue Chatbot."""

import logging
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from langchain_core.messages import HumanMessage

from app.agent import get_agent
from app.config import get_settings
from app.db import init_db
from app.schemas import ChatRequest, ChatResponse

settings = get_settings()
logging.basicConfig(level=settings.log_level)
log = logging.getLogger("chat")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    app.state.agent = await get_agent()
    yield


app = FastAPI(
    title="GitHub Issue Chatbot",
    version="1.0.0",
    description="Single endpoint AI chatbot that reads, explains, creates, and closes "
    "GitHub issues using an LLM and FastMCP.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


async def verify_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    if settings.api_key and x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key header")


def extract_text(content) -> str:
    """Message content can be a string or a list of parts depending on the model."""
    if isinstance(content, str):
        return content
    parts = []
    for p in content or []:
        if isinstance(p, str):
            parts.append(p)
        elif isinstance(p, dict):
            parts.append(p.get("text", ""))
    return "".join(parts)


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")


@app.post(
    "/chat",
    response_model=ChatResponse,
    summary="Chat with the GitHub Issue Assistant",
    dependencies=[Depends(verify_api_key)],
)
async def chat_endpoint(req: ChatRequest) -> ChatResponse:
    """
    Chat with the AI assistant. Capabilities:

    - **Read issues:** "List open issues in owner/repo" or "Show issue #10 in owner/repo"
    - **Count issues:** "How many issues are in my default repo?"
    - **Explain issues:** "Explain issue #10" (summary, root cause, fix recommendations)
    - **Create issues:** "Create an issue: Login button broken on iOS"
    - **Close issues:** "Close issue #10"
    """
    thread_id = req.thread_id or str(uuid.uuid4())
    try:
        res = await app.state.agent.ainvoke(
            {"messages": [HumanMessage(content=req.message)]},
            config={"configurable": {"thread_id": thread_id}},
        )
    except Exception as e:
        log.exception("chat failed")
        raise HTTPException(status_code=502, detail=f"Agent error: {type(e).__name__}: {e}")

    reply = extract_text(res["messages"][-1].content) or "(The assistant returned an empty reply.)"
    return ChatResponse(reply=reply, thread_id=thread_id)
