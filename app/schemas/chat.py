"""Pydantic schemas for chatbot endpoint."""

from __future__ import annotations

from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., description="User question regarding the model findings.")
    analysis_result: Optional[Dict[str, Any]] = Field(None, description="Current prediction and explanation findings JSON.")
    session_id: str = Field("default_session", description="Session identifier for stateful chat.")


class ChatResponse(BaseModel):
    answer: str = Field(..., description="Chatbot response text.")
    session_id: str = Field(..., description="Session identifier.")
    research_disclaimer: str = Field(..., description="Mandatory research notice.")
