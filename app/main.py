"""FastAPI entrypoint for the Al-Fardan Q9 chatbot."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .db import (
    capture_unanswered,
    ensure_db,
    get_history,
    list_unanswered,
    record_message,
    update_query_status,
)
from .integrations.crm import post_lead_to_crm
from .models import (
    ChatAction,
    ChatRequest,
    ChatResponse,
    HealthResponse,
    LeadCaptureRequest,
    ReindexResponse,
    UnansweredQueryOut,
)
from .retrieval.intent import (
    classify,
    match_type_for,
    scripted_actions,
    scripted_reply,
)
from .retrieval.tfidf import (
    KbEntry,
    TfidfRetriever,
    build_retriever_from_kb,
)


# ─── Category → default action mapping ────────────────────────────────
# When a KB hit has no explicit `actions`, we auto-derive a button based
# on the entry's category so answers about staking get an [Open Staking]
# button, answers about lending get [Open Lending], etc. This is what
# makes the bot feel agent-ish even for entries we haven't manually
# annotated with action buttons.

_CATEGORY_ACTIONS: dict[str, list[dict]] = {
    "staking": [
        {"label": "Open Staking", "url": "/staking", "kind": "link"},
    ],
    "lending": [
        {"label": "Open Lending", "url": "/lending", "kind": "link"},
        {"label": "Loan Calculator", "url": "/lending#calculator", "kind": "link"},
    ],
    "custody": [
        {"label": "Open Wallets", "url": "/wallets", "kind": "link"},
    ],
    "otc": [
        {"label": "Open OTC Desk", "url": "/otc", "kind": "link"},
    ],
    "crm": [
        # Auto-link for CRM nav answers depends on the question — leave
        # the answer text to provide the specific URL; return a Portfolio
        # button as a reasonable default fallback.
        {"label": "Open Portfolio", "url": "/portfolio", "kind": "link"},
    ],
    "pricing": [
        {"label": "Open Pricing Details", "url": "/pricing", "kind": "link"},
    ],
    "security": [
        {"label": "Security Settings", "url": "/settings#security", "kind": "link"},
    ],
    "support": [
        {"label": "Open Support", "url": "/settings?tab=support", "kind": "link"},
    ],
    "onboarding": [
        {"label": "Start Signup", "url": "/auth/signup", "kind": "link"},
    ],
    # Crypto / bitcoin / general → no auto action (pure info)
}


def _actions_for_entry(entry: KbEntry) -> list[ChatAction]:
    """Explicit entry-level actions override category defaults.

    Uses getattr with a default so entries unpickled from an older
    schema (pre-`actions` field) don't crash — they just fall through
    to the category-default action set.
    """
    explicit = getattr(entry, "actions", None) or []
    category = getattr(entry, "category", "") or ""
    raw = explicit or _CATEGORY_ACTIONS.get(category, [])
    return [
        ChatAction(
            label=a["label"],
            url=a["url"],
            kind=a.get("kind", "link"),
        )
        for a in raw
    ]


def _actions_from_list(raw: list[dict] | None) -> list[ChatAction] | None:
    if not raw:
        return None
    return [
        ChatAction(
            label=a["label"],
            url=a["url"],
            kind=a.get("kind", "link"),
        )
        for a in raw
    ]


# ─── App bootstrap ─────────────────────────────────────────────────────

app = FastAPI(
    title="Al-Fardan Q9 Chatbot",
    version="0.1.0",
    description="Local-only FAQ + signup chatbot for the Al-Fardan Q9 portal.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-Chatbot-Secret"],
)

_retriever: Optional[TfidfRetriever] = None
_start_time = time.time()


def _load_retriever() -> TfidfRetriever:
    """Load the prebuilt pickle if present, otherwise build from KB.

    First-boot + after an admin edit we fall through to the in-memory
    build. This keeps dev friction low — no separate `build_index`
    step required unless you want to skip the cold-start work.

    Schema-drift guard: if a pickle is from an older schema (e.g.
    missing the newer `actions` field on KbEntry) we detect it by
    sampling one entry and rebuild from source. This prevents the
    new code from silently loading stale pickles and crashing on
    attribute access downstream.
    """
    if settings.index_path.exists():
        try:
            r = TfidfRetriever.load(settings.index_path)
            # Sample one entry — if the schema drifted (e.g. missing
            # attribute added in a later release), trigger a rebuild.
            if r.entries:
                sample = r.entries[0]
                if not hasattr(sample, "actions"):
                    print(
                        "[main] pickled index predates `actions` field — "
                        "rebuilding from KB source"
                    )
                    return build_retriever_from_kb(settings.kb_dir)
            return r
        except Exception as err:  # noqa: BLE001
            print(f"[main] failed to load prebuilt index ({err}); rebuilding")
    return build_retriever_from_kb(settings.kb_dir)


@app.on_event("startup")
async def _on_startup() -> None:
    global _retriever
    ensure_db(settings.db_path)
    _retriever = _load_retriever()
    print(
        f"[main] retriever ready — {_retriever.size} entries, "
        f"{_retriever.vocab_size} tokens"
    )


def require_admin_secret(
    x_chatbot_secret: str = Header(default=""),
) -> None:
    expected = settings.chatbot_secret
    if not expected or x_chatbot_secret != expected:
        raise HTTPException(status_code=401, detail="invalid secret")


def retriever() -> TfidfRetriever:
    if _retriever is None:
        raise HTTPException(status_code=503, detail="retriever not loaded")
    return _retriever


# ─── Routes ───────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    r = _retriever
    return HealthResponse(
        ok=True,
        index_loaded=r is not None,
        entries=r.size if r else 0,
        uptime_seconds=time.time() - _start_time,
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(
    req: ChatRequest,
    request: Request,
    r: TfidfRetriever = Depends(retriever),
) -> ChatResponse:
    """Main chat endpoint — public (widget calls it directly)."""
    # Record user message first so we have context even if the reply fails
    record_message(
        settings.db_path,
        session_token=req.session_token,
        role="user",
        text=req.message,
        user_id=req.user_id,
        user_email=req.user_email,
    )

    intent = classify(req.message)

    # Scripted replies beat retrieval for well-defined intents
    scripted = scripted_reply(intent)
    if scripted is not None:
        actions = _actions_from_list(scripted_actions(intent))
        reply = ChatResponse(
            reply=scripted,
            match_type=match_type_for(intent),
            session_token=req.session_token,
            signup_fields_needed=(
                ["firstName", "lastName", "email", "phone", "service"]
                if intent == "signup"
                else None
            ),
            actions=actions,
        )
        record_message(
            settings.db_path,
            session_token=req.session_token,
            role="bot",
            text=reply.reply,
            match_type=reply.match_type,
        )
        return reply

    # Fall through to TF-IDF retrieval
    hits = r.search(req.message, top_k=settings.top_k)

    if not hits or hits[0].score < settings.confidence_threshold:
        best_score = hits[0].score if hits else 0.0
        nearest = hits[0].entry.id if hits else None
        capture_unanswered(
            settings.db_path,
            session_token=req.session_token,
            user_text=req.message,
            best_score=best_score,
            nearest_entry=nearest,
            user_id=req.user_id,
            user_email=req.user_email,
        )
        reply_text = (
            "I don't have a confident answer for that yet — I've logged your "
            "question for our team and a human will follow up. In the "
            "meantime, here are some things I can help with."
        )
        # Even on fallback, give the user quick-access buttons to the
        # main service pages so the conversation has somewhere to go.
        fallback_actions = [
            ChatAction(label="Open Portfolio", url="/portfolio", kind="link"),
            ChatAction(label="Open Staking", url="/staking", kind="link"),
            ChatAction(label="Open Lending", url="/lending", kind="link"),
            ChatAction(label="Contact Support", url="/settings?tab=support", kind="link"),
        ]
        response = ChatResponse(
            reply=reply_text,
            match_type="low_confidence" if hits else "fallback",
            matched_entry_id=nearest,
            match_score=best_score,
            session_token=req.session_token,
            actions=fallback_actions,
        )
        record_message(
            settings.db_path,
            session_token=req.session_token,
            role="bot",
            text=reply_text,
            match_type=response.match_type,
            matched_entry=nearest,
            match_score=best_score,
        )
        return response

    top = hits[0]
    entry_actions = _actions_for_entry(top.entry)
    response = ChatResponse(
        reply=top.entry.answer,
        match_type="kb_hit",
        matched_entry_id=top.entry.id,
        match_score=top.score,
        session_token=req.session_token,
        actions=entry_actions or None,
    )
    record_message(
        settings.db_path,
        session_token=req.session_token,
        role="bot",
        text=top.entry.answer,
        match_type="kb_hit",
        matched_entry=top.entry.id,
        match_score=top.score,
    )
    return response


@app.get("/chat/history")
async def chat_history(session_token: str, limit: int = 50):
    return {"data": get_history(settings.db_path, session_token, limit)}


# ─── Admin / integration ──────────────────────────────────────────────

@app.post("/admin/reindex", response_model=ReindexResponse)
async def admin_reindex(_: None = Depends(require_admin_secret)) -> ReindexResponse:
    """Rebuild the TF-IDF index — call after editing the KB."""
    global _retriever
    t0 = time.time()
    _retriever = build_retriever_from_kb(settings.kb_dir)
    _retriever.save(settings.index_path)
    return ReindexResponse(
        built=True,
        entries=_retriever.size,
        vocab_size=_retriever.vocab_size,
        took_ms=int((time.time() - t0) * 1000),
    )


@app.get("/admin/queries", response_model=list[UnansweredQueryOut])
async def admin_queries(
    status: str = "new",
    limit: int = 100,
    _: None = Depends(require_admin_secret),
):
    rows = list_unanswered(settings.db_path, status=status, limit=limit)
    return [
        UnansweredQueryOut(
            id=r["id"],
            session_token=r["session_token"],
            user_text=r["user_text"],
            best_score=r["best_score"],
            status=r["status"],
            user_email=r.get("user_email"),
            created_at=r["created_at"],
        )
        for r in rows
    ]


@app.post("/admin/queries/{qid}")
async def admin_update_query(
    qid: int,
    status: str,
    review_notes: str = "",
    _: None = Depends(require_admin_secret),
):
    if status not in {"new", "promoted", "resolved", "dismissed"}:
        raise HTTPException(status_code=400, detail="invalid status")
    ok = update_query_status(settings.db_path, qid, status, review_notes)
    if not ok:
        raise HTTPException(status_code=404, detail="query not found")
    return {"success": True}


@app.post("/admin/lead")
async def admin_lead_handoff(
    lead: LeadCaptureRequest,
    _: None = Depends(require_admin_secret),
):
    """Bot gathered a qualified lead — hand off to the CRM.

    The widget collects this via the signup flow after the user gives
    consent. We POST it to the CRM, which creates the user and sends
    the OTP.
    """
    crm_response = await post_lead_to_crm(lead)
    return {"success": True, "crm": crm_response}
