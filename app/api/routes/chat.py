"""API Route for context-aware findings chatbot."""

from __future__ import annotations

from fastapi import APIRouter
from app.schemas.chat import ChatRequest, ChatResponse
from chatbot.assistant import FindingsChatbot

router = APIRouter(tags=["Findings Chatbot"])
_chatbot_instance = FindingsChatbot()


@router.post("/chat", response_model=ChatResponse, summary="Chat with AI Findings Assistant")
async def chat_endpoint(payload: ChatRequest) -> ChatResponse:
    """Answer questions regarding current model findings and XAI explanations."""
    result = _chatbot_instance.ask(
        message=payload.message,
        analysis_result=payload.analysis_result,
        session_id=payload.session_id,
    )
    return ChatResponse(
        answer=result["answer"],
        session_id=result["session_id"],
        research_disclaimer=result["research_disclaimer"],
    )
