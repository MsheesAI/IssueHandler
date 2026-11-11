"""Request and response schemas for the Chatbot API."""

from typing import Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        description="Your request, e.g. 'Show issues in owner/repo', 'Explain issue #5', "
        "'Create an issue', or 'Close issue #5'",
    )
    thread_id: Optional[str] = Field(
        None, description="Optional conversation thread ID to continue context"
    )


class ChatResponse(BaseModel):
    reply: str = Field(..., description="Chatbot response")
    thread_id: str = Field(..., description="Conversation thread ID")
