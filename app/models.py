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
    "intent_navigate_staking",
    "intent_navigate_lending",
    "intent_navigate_custody",
    "intent_navigate_otc",
    "intent_navigate_portfolio",
    "intent_navigate_settings",
    "intent_navigate_wallets",
    "intent_balance",
    "intent_contact_support",
    "intent_withdraw",
    "intent_deposit",
    "intent_affirmation",
    "intent_negation",
    # Specific staking sub-topics added post-QA — each has a bespoke
    # scripted reply so "what is slashing?" no longer gets the generic
    # "which network would you like to stake?" mis-routing.
    "intent_slashing",
    "intent_tax_reporting",
    "intent_network_outage",
    "intent_withdrawal_delay",
    "intent_staking_fees",
    "intent_staking_frequency",
    "intent_unauthorized_login",
    "intent_insurance_claim",
    "intent_auto_reinvest",
    "intent_transfer_from_exchange",
    "intent_otc_quote_validity",
    # Set 4
    "intent_interest_calculation",
    "intent_whitelist_address",
    "intent_account_closure",
    "intent_early_repayment",
    "intent_validator_choice",
    "intent_security_incident",
    # Set 5
    "intent_loan_extension",
    "intent_internal_transfer",
    "intent_interest_rate_change",
    "intent_staking_network_fees",
    "intent_custody_minimum_balance",
    "intent_otc_counterparty",
    "intent_api_key_management",
    "intent_staking_rewards_withdraw",
    # Set 6
    "intent_ltv_calculation",
    "intent_rewards_history",
    "intent_session_timeout",
    # Sets 7/8/9
    "intent_claim_rewards",
    "intent_interest_payment",
    "intent_shared_account",
    "intent_cancel_otc",
    "intent_otc_max_size",
    "intent_tin_required",
    "intent_add_collateral",
    "intent_ip_whitelisting",
    "intent_negotiate_rate",
    "intent_loan_default",
    "intent_validator_diversification",
    "intent_monthly_statements",
    "intent_liquidation_details",
    # Sets 11/12/13/14
    "intent_partial_repayment",
    "intent_hardware_wallet",
    "intent_rewards_in_different_asset",
    "intent_missing_rewards",
    "intent_user_to_user_transfer",
    "intent_otc_after_hours",
    "intent_login_history",
    "intent_interest_source",
    "intent_rewards_expiration",
    "intent_withdrawal_time",
    "intent_active_sessions",
    "intent_loan_transfer",
    "intent_validator_locations",
    "intent_historical_snapshot",
    "system_error",
]


class ChatAction(BaseModel):
    """A tappable button the widget renders below a bot message.

    kind="link"    → widget navigates the parent page to `url`
    kind="prefill" → widget pre-fills the chat input with `url` as a
                     follow-up message (for multi-turn flows)
    """

    label: str = Field(..., min_length=1, max_length=60)
    url: str = Field(..., min_length=1, max_length=200)
    kind: Literal["link", "prefill"] = "link"


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
    # Interactive affordances — the widget renders these as buttons
    # below the bot message. Used to deep-link users directly to the
    # right CRM page (or kick off a follow-up multi-turn flow).
    actions: Optional[list[ChatAction]] = None
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
