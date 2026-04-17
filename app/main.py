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
from .retrieval import dense as dense_retrieval


# ─── Category → default action mapping ────────────────────────────────
# When a KB hit has no explicit `actions`, we auto-derive a button based
# on the entry's category so answers about staking get an [Open Staking]
# button, answers about lending get [Open Lending], etc. This is what
# makes the bot feel agent-ish even for entries we haven't manually
# annotated with action buttons.

_CATEGORY_ACTIONS: dict[str, list[dict]] = {
    # Real CRM routes — every URL here MUST map to an existing page in
    # al-fardan-crm/src/app/<path>/page.tsx. Wrong paths produce 404s
    # in production because the widget navigates users to them directly.
    "staking": [
        {"label": "Open Staking", "url": "/dashboard/staking", "kind": "link"},
    ],
    "lending": [
        {"label": "Open Loans", "url": "/dashboard/loans", "kind": "link"},
        {"label": "Lending Service", "url": "/dashboard/services/lending", "kind": "link"},
    ],
    "custody": [
        {"label": "Open Wallets", "url": "/dashboard/wallets", "kind": "link"},
    ],
    "otc": [
        {"label": "Open OTC Desk", "url": "/dashboard/services/otc", "kind": "link"},
    ],
    "crm": [
        {"label": "Open Dashboard", "url": "/dashboard", "kind": "link"},
    ],
    "pricing": [
        {"label": "Open Dashboard", "url": "/dashboard", "kind": "link"},
    ],
    "security": [
        {"label": "Security Settings", "url": "/dashboard/settings", "kind": "link"},
    ],
    "support": [
        {"label": "Email Support", "url": "mailto:institutional@alfardanq9.com", "kind": "link"},
    ],
    "onboarding": [
        {"label": "Create Account", "url": "/auth/signup", "kind": "link"},
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

# The retriever can be either a DenseRetriever (preferred, semantic) or
# a TfidfRetriever (fallback). Both expose the same public surface
# (size, vocab_size, search, entries) so call sites are retriever-agnostic.
_retriever: Optional[object] = None
_retriever_kind: str = "none"  # "dense" | "tfidf" | "none" — for /health
_start_time = time.time()


def _load_retriever() -> object:
    """Preference order for producing a ready-to-serve retriever:

    1. If a prebuilt pickle exists and declares itself 'dense', load it.
       Dense pickles are O(100ms) to mmap and serve.
    2. If fastembed is available, build a fresh dense retriever from the
       KB dir. First boot downloads the embedding model (~130MB); every
       subsequent boot uses the cached model.
    3. Fall back to TF-IDF — either loading a legacy pickle or building
       from KB. The bot stays functional if fastembed/onnxruntime is
       somehow broken (missing lib, OOM, etc.).

    Every transition logs clearly so ops can see which tier is serving
    traffic at any given time.
    """
    global _retriever_kind

    # ─── Try dense pickle load ────────────────────────────────────
    if settings.index_path.exists() and dense_retrieval.available():
        try:
            r = dense_retrieval.DenseRetriever.load(settings.index_path)
            print(
                f"[main] dense retriever loaded from pickle "
                f"({r.size} entries, dim={r.vocab_size})"
            )
            _retriever_kind = "dense"
            return r
        except Exception as err:  # noqa: BLE001
            # Could be: pickle is a TF-IDF variant, pickle corrupted, or
            # schema drifted. Fall through to next strategy.
            print(f"[main] dense pickle load failed ({err}); trying next")

    # ─── Build fresh dense retriever from KB ──────────────────────
    if dense_retrieval.available():
        try:
            print("[main] building dense retriever from KB (first boot)...")
            r = dense_retrieval.build_dense_from_kb(settings.kb_dir)
            # Cache for next boot — saves ~seconds on cold-start once models
            # are downloaded but KB re-embedding still takes time.
            try:
                r.save(settings.index_path)
            except Exception as err:  # noqa: BLE001
                print(f"[main] could not save dense pickle ({err}); in-memory only")
            print(
                f"[main] dense retriever built "
                f"({r.size} entries, dim={r.vocab_size})"
            )
            _retriever_kind = "dense"
            return r
        except Exception as err:  # noqa: BLE001
            print(f"[main] dense build failed ({err}); falling back to TF-IDF")
    else:
        print(
            f"[main] fastembed unavailable — using TF-IDF fallback. "
            f"Reason: {dense_retrieval.unavailable_reason()}"
        )

    # ─── Fall back to TF-IDF ──────────────────────────────────────
    if settings.index_path.exists():
        try:
            r = TfidfRetriever.load(settings.index_path)
            if r.entries and not hasattr(r.entries[0], "actions"):
                print("[main] TF-IDF pickle predates `actions` — rebuilding")
                r = build_retriever_from_kb(settings.kb_dir)
            _retriever_kind = "tfidf"
            return r
        except Exception as err:  # noqa: BLE001
            print(f"[main] TF-IDF pickle load failed ({err}); rebuilding from KB")

    r = build_retriever_from_kb(settings.kb_dir)
    _retriever_kind = "tfidf"
    return r


@app.on_event("startup")
async def _on_startup() -> None:
    global _retriever
    ensure_db(settings.db_path)
    _retriever = _load_retriever()
    print(
        f"[main] retriever ready — kind={_retriever_kind}, "
        f"{_retriever.size} entries, "  # type: ignore[attr-defined]
        f"{_retriever.vocab_size} "  # type: ignore[attr-defined]
        f"{'dims' if _retriever_kind == 'dense' else 'tokens'}"
    )


def require_admin_secret(
    x_chatbot_secret: str = Header(default=""),
) -> None:
    expected = settings.chatbot_secret
    if not expected or x_chatbot_secret != expected:
        raise HTTPException(status_code=401, detail="invalid secret")


def retriever() -> object:
    """Retriever dependency — duck-typed (dense or TF-IDF). Both
    expose .search(query, top_k) returning list[SearchResult]."""
    if _retriever is None:
        raise HTTPException(status_code=503, detail="retriever not loaded")
    return _retriever


# ─── Routes ───────────────────────────────────────────────────────────

# Per-retriever confidence threshold. Dense cosine sims live in a very
# different range than TF-IDF TF-IDF cosine — same 0.15 threshold would
# make dense answer anything faintly related. Tune separately.
_CONFIDENCE_BY_KIND = {
    "dense": 0.50,   # bge-small returns 0.55-0.90 for good matches
    "tfidf": 0.15,   # legacy setting — bag-of-words cosine is tighter
}


def _confidence_threshold() -> float:
    """Kind-aware threshold — fallback to the config default if the
    retriever kind is unrecognised (shouldn't happen in practice)."""
    return _CONFIDENCE_BY_KIND.get(
        _retriever_kind, settings.confidence_threshold
    )


# ─── Lightweight response cache ──────────────────────────────────────
# Embedding + retrieval for a freshly-typed query is ~10-15ms with
# bge-small. That's fast, but we still see the same popular questions
# asked many times ("tell me about al fardan", "what is LTV"). Caching
# the {retriever_kind, normalised_query} → ChatResponse tuple turns
# the 2nd+ hit into a sub-ms dict lookup — meaningful savings under
# load, and it caps model CPU usage.
#
# Why a TTL? KB reindexes (via /admin/reindex) should invalidate the
# cache. Simple TTL eviction achieves that without wiring explicit
# invalidation hooks into every admin path.

from collections import OrderedDict
import threading

_CACHE_MAX_ENTRIES = 1000
_CACHE_TTL_SECONDS = 3600  # 1 hour
_response_cache: "OrderedDict[str, tuple[float, dict]]" = OrderedDict()
_cache_lock = threading.Lock()


def _cache_key(query: str) -> str:
    """Normalise queries so minor typography variations share entries."""
    return f"{_retriever_kind}|{' '.join(query.lower().split())}"


def _cache_get(query: str) -> dict | None:
    key = _cache_key(query)
    now = time.time()
    with _cache_lock:
        hit = _response_cache.get(key)
        if hit is None:
            return None
        cached_at, payload = hit
        if now - cached_at > _CACHE_TTL_SECONDS:
            _response_cache.pop(key, None)
            return None
        # LRU bump
        _response_cache.move_to_end(key)
        return payload


def _cache_put(query: str, payload: dict) -> None:
    key = _cache_key(query)
    with _cache_lock:
        _response_cache[key] = (time.time(), payload)
        _response_cache.move_to_end(key)
        while len(_response_cache) > _CACHE_MAX_ENTRIES:
            _response_cache.popitem(last=False)


def _cache_invalidate_all() -> None:
    with _cache_lock:
        _response_cache.clear()


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    r = _retriever
    return HealthResponse(
        ok=True,
        index_loaded=r is not None,
        entries=r.size if r else 0,  # type: ignore[attr-defined]
        uptime_seconds=time.time() - _start_time,
    )


@app.get("/debug")
async def debug_info() -> dict:
    """Ops-friendly introspection — which retriever tier is live, how
    hot the cache is, whether fastembed imported cleanly. Handy when
    diagnosing a slow or stale chatbot without pulling container logs."""
    r = _retriever
    return {
        "retriever_kind": _retriever_kind,
        "entries": r.size if r else 0,  # type: ignore[attr-defined]
        "dim_or_vocab": r.vocab_size if r else 0,  # type: ignore[attr-defined]
        "confidence_threshold": _confidence_threshold(),
        "fastembed_available": dense_retrieval.available(),
        "fastembed_error": dense_retrieval.unavailable_reason() or None,
        "cache_entries": len(_response_cache),
        "cache_max": _CACHE_MAX_ENTRIES,
        "uptime_seconds": time.time() - _start_time,
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(
    req: ChatRequest,
    request: Request,
    r=Depends(retriever),
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

    # ─── Response cache ──────────────────────────────────────────
    # Skip recording-cache logic for super-short messages (likely noise
    # or greetings handled by intent); cache only substantive queries.
    cached = _cache_get(req.message) if len(req.message) >= 3 else None
    if cached is not None:
        # Inject the caller's session token so history stays consistent
        # while everything else (reply + actions + match metadata) is
        # served from cache.
        cached_reply = ChatResponse(**{**cached, "session_token": req.session_token})
        record_message(
            settings.db_path,
            session_token=req.session_token,
            role="bot",
            text=cached_reply.reply,
            match_type=cached_reply.match_type,
            matched_entry=cached_reply.matched_entry_id,
            match_score=cached_reply.match_score,
        )
        return cached_reply

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

    # Fall through to semantic retrieval (dense) or TF-IDF (fallback)
    hits = r.search(req.message, top_k=settings.top_k)
    threshold = _confidence_threshold()

    if not hits or hits[0].score < threshold:
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
            ChatAction(label="Open Dashboard", url="/dashboard", kind="link"),
            ChatAction(label="Open Staking", url="/dashboard/staking", kind="link"),
            ChatAction(label="Open Loans", url="/dashboard/loans", kind="link"),
            ChatAction(label="Email Support", url="mailto:institutional@alfardanq9.com", kind="link"),
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
    # Cache the confident answer — subsequent identical queries are
    # served from memory in sub-ms rather than re-embedding.
    if len(req.message) >= 3:
        _cache_put(
            req.message,
            response.model_dump(exclude={"session_token"}),
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
    """Rebuild the index — call after editing the KB.

    Prefers the dense retriever if fastembed is available; falls back
    to TF-IDF. Clears the response cache since answers may have changed.
    """
    global _retriever, _retriever_kind
    t0 = time.time()
    if dense_retrieval.available():
        try:
            _retriever = dense_retrieval.build_dense_from_kb(settings.kb_dir)
            _retriever_kind = "dense"
        except Exception as err:  # noqa: BLE001
            print(f"[reindex] dense rebuild failed ({err}); falling back to TF-IDF")
            _retriever = build_retriever_from_kb(settings.kb_dir)
            _retriever_kind = "tfidf"
    else:
        _retriever = build_retriever_from_kb(settings.kb_dir)
        _retriever_kind = "tfidf"
    try:
        _retriever.save(settings.index_path)  # type: ignore[attr-defined]
    except Exception as err:  # noqa: BLE001
        print(f"[reindex] save failed ({err}); in-memory only")
    _cache_invalidate_all()
    return ReindexResponse(
        built=True,
        entries=_retriever.size,  # type: ignore[attr-defined]
        vocab_size=_retriever.vocab_size,  # type: ignore[attr-defined]
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
