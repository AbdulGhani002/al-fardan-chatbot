"""Pydantic schemas for the HTTP API."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional
from pydantic import BaseModel, EmailStr, Field, field_validator


# ─── /chat ────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    """One incoming message from the widget."""

    session_token: str = Field(..., min_length=8, max_length=64)
    message: str = Field(..., min_length=1, max_length=2000)
    # Optional — populated if the user is already signed in
    user_id: Optional[str] = None
    user_email: Optional[str] = None

    @field_validator("message")
    @classmethod
    def strip_message(cls, v: str) -> str:
        return v.strip()


class ChatMessage(BaseModel):
    role: Literal["user", "bot", "system"]
    text: str
    created_at: datetime


MatchType = Literal[
    "kb_hit",
    "low_confidence",
    "fallback",
    "intent_signup",
    "intent_greeting",
    "intent_goodbye",
    "system_error",
]


class ChatResponse(BaseModel):
    reply: str
    match_type: MatchType
    # Top KB entry id + score, if any — useful for debugging the
    # retriever via the Swagger UI.
    matched_entry_id: Optional[str] = None
    match_score: Optional[float] = None
    # When match_type == intent_signup, the bot returns a form
    # scaffold so the widget can render collection fields inline.
    signup_fields_needed: Optional[list[str]] = None
    # Session id for the client to persist so future messages stay
    # threaded with this conversation.
    session_token: str


# ─── admin + integration endpoints ────────────────────────────────────

class LeadCaptureRequest(BaseModel):
    """Chatbot hands a completed signup off to the CRM."""

    first_name: str = Field(..., min_length=1, max_length=50)
    last_name: str = Field(..., min_length=1, max_length=50)
    email: EmailStr
    phone: str = Field(..., min_length=6, max_length=25)
    service: Optional[Literal["custody", "staking", "otc", "lending"]] = None
    consent_timestamp: datetime


class ReindexResponse(BaseModel):
    built: bool
    entries: int
    vocab_size: int
    took_ms: int


class UnansweredQueryOut(BaseModel):
    id: int
    session_token: str
    user_text: str
    best_score: float
    status: Literal["new", "promoted", "resolved", "dismissed"]
    user_email: Optional[str]
    created_at: datetime


class HealthResponse(BaseModel):
    ok: bool
    index_loaded: bool
    entries: int
    uptime_seconds: float
