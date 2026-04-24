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
    list_all_messages,
    list_unanswered,
    message_stats,
    record_message,
    training_turn_pairs,
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
from .compose.entities import extract_entities
from .compose.composer import compose as compose_reply
from .refine.typos import correct_typos, preview_corrections
from .refine.extract import extract_best_sentences
from .refine.emotion import detect as detect_mood, acknowledgment, escalation_actions
from .rag.generator import (
    Generator,
    build_generator,
    probe_generator,
)
from .rag.prompt import (
    DEFAULT_GEN_OPTIONS,  # noqa: F401 — kept for downstream tuning
    MAX_REFERENCES,
    SYSTEM_PROMPT,
    build_user_prompt,
)
from .refine.synonyms import expand as expand_synonyms
from .integrations import platform_settings as _platform_settings


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


# ─── Response post-processor ─────────────────────────────────────────
# Scraped content from Wikipedia / ethereum.org / bitcoin.org is often
# 500-1200 chars long — fine for a reference doc, but robotic inside a
# chat bubble next to Safiya's photo. This post-processor trims scraped
# answers to the first couple of sentences and appends a contextual
# follow-up question if one isn't already there. Curated KB entries are
# left untouched — the human voice is already in the source.
import re as _re

_SCRAPED_ID_PREFIX = "scraped:"
_MAX_SCRAPED_SENTENCES = 2
_MAX_SCRAPED_CHARS = 500


def _is_scraped(entry) -> bool:
    eid = getattr(entry, "id", "")
    return isinstance(eid, str) and eid.startswith(_SCRAPED_ID_PREFIX)


def _shorten_scraped(text: str) -> str:
    """Take the first 1-2 sentences of a scraped answer so the bot
    doesn't paste a Wikipedia paragraph into a chat bubble."""
    if len(text) <= _MAX_SCRAPED_CHARS:
        return text
    # Split on sentence terminators that are followed by space + capital —
    # better than a naive split on "." which mangles "U.S." and "e.g."
    sentences = _re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    if len(sentences) <= _MAX_SCRAPED_SENTENCES:
        return text[:_MAX_SCRAPED_CHARS].rsplit(" ", 1)[0] + "…"
    short = " ".join(sentences[:_MAX_SCRAPED_SENTENCES])
    if len(short) > _MAX_SCRAPED_CHARS:
        short = short[:_MAX_SCRAPED_CHARS].rsplit(" ", 1)[0] + "…"
    return short


# Follow-up prompts per category. Institutional Voice Protocol: no
# retail CTAs ("want me to…", "shall I…"), no emotional closers. Only
# category-specific continuation prompts that invite a precise follow-up
# the client has a right to ask. Empty = no follow-up appended.
#
# Behavior Manual §8 and §9 — the chatbot should not chase engagement,
# it should wait to be asked. Prefer silence over retail warmth.
_FOLLOWUP_BY_CATEGORY: dict[str, str] = {
    # Product topics where a single clarifying parameter is genuinely
    # needed to move a conversation forward.
    "lending":    "For an indicative quote, please confirm collateral asset and loan size.",
    "staking":    "Please indicate which network is relevant to your mandate.",
    "otc":        "Please share indicative trade size so we can route to the desk.",
    # Everything else: no appended prompt. The answer stands on its own.
}

_CATEGORY_ALIASES = {
    # Categories used by the scrapers map to curated categories for
    # follow-up selection.
    "bitcoin_org": "crypto",
    "ethereum_org": "crypto",
    "wikipedia_org": "crypto",
    "bitcoin": "crypto",
}


def _ends_with_question(text: str) -> bool:
    """True if the text ends with '?' (allowing trailing whitespace /
    closing quotes / period-after-question)."""
    stripped = text.rstrip(' ."\'""…')
    return stripped.endswith("?")


# ─── Language fidelity ────────────────────────────────────────────────
# Cheap heuristic script detector: catches the most common
# language-drift failures (English question → French/Spanish/Arabic
# reply) without pulling in langdetect + its 50MB of models. Good
# enough for a pre-LLM sanity check; if it misclassifies edge cases we
# err on the side of serving the LLM output (not the KB fallback).

_FR_MARKERS = (
    "c'est", "le client", "les références", "veuillez", "nous sommes",
    "je vous invite", "il est", "vous pouvez", "n'est pas", "notre équipe",
    "en savoir plus", "pour plus", "il s'agit", "cependant", "toutefois",
)
_ES_MARKERS = (
    "el cliente", "para más", "se recomienda", "no se", "por favor",
    "nuestro equipo", "en cuanto", "usted puede", "si usted", "podemos",
)


def _quick_lang(text: str) -> str:
    """Return 'arabic' | 'french' | 'spanish' | 'english' based on a
    cheap character + marker-phrase sniff. Defaults to 'english' when
    unsure — English is the dominant query language and that keeps the
    mismatch detector conservative."""
    if not text:
        return "english"
    # Script-based: Arabic block is unambiguous.
    arabic_chars = sum(
        1 for c in text if 0x0600 <= ord(c) <= 0x06FF
    )
    if arabic_chars >= 3:
        return "arabic"
    low = text.lower()
    if any(m in low for m in _FR_MARKERS):
        return "french"
    if any(m in low for m in _ES_MARKERS):
        return "spanish"
    return "english"


def _is_language_mismatch(user_msg: str, llm_reply: str) -> bool:
    """Return True when the LLM reply looks like a different language
    than the user's message. Only signals on confident mismatches so
    we don't reject correct responses."""
    u = _quick_lang(user_msg)
    r = _quick_lang(llm_reply)
    return u == "english" and r in {"french", "spanish", "arabic"}


def _followup_for(category: str) -> str | None:
    cat = (category or "").lower()
    cat = _CATEGORY_ALIASES.get(cat, cat)
    return _FOLLOWUP_BY_CATEGORY.get(cat)


def _humanize_answer(entry, answer: str, query: str = "") -> str:
    """Trim long content + append a follow-up question when missing.

    If we have the original user query, we use sentence-level
    extraction to keep only the 1-2 sentences most relevant to what
    they asked (rather than dumping the whole KB entry). Scraped
    content + long curated entries both get this treatment.

    Curated short entries (< 300 chars and ≤ 3 sentences) pass
    through unchanged — they're already in Safiya's voice.
    """
    out = answer
    # Only aggressively extract sentences from SCRAPED content or genuinely
    # long curated answers (>600 chars). Shorter institutional-register
    # answers (300-600 chars, which is most of the KB) pass through
    # untouched — sentence extraction on them was causing truncation at
    # honorific boundaries (e.g. "Sheikh Dr.") and losing meaningful
    # context ("MoU signed 10 Dec 2023" dropped from MoU entries).
    if _is_scraped(entry):
        if query:
            out = extract_best_sentences(
                out, query, n=3, max_chars=500, always_keep_first=True
            )
        else:
            out = _shorten_scraped(out)
    elif query and len(out) > 600:
        out = extract_best_sentences(
            out, query, n=3, max_chars=520, always_keep_first=True
        )

    if not _ends_with_question(out):
        fu = _followup_for(getattr(entry, "category", ""))
        if fu:
            out = f"{out.rstrip()} {fu}"
    return out


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

# Twilio voice webhook routes — /voice/incoming, /voice/respond,
# /voice/status. Mounted here so the routes + OpenAPI docs are
# always visible; they're no-ops until Twilio is wired to the
# same base URL.
from .voice.twilio_routes import router as voice_router  # noqa: E402

app.include_router(voice_router)

# The retriever can be either a DenseRetriever (preferred, semantic) or
# a TfidfRetriever (fallback). Both expose the same public surface
# (size, vocab_size, search, entries) so call sites are retriever-agnostic.
_retriever: Optional[object] = None
_retriever_kind: str = "none"  # "dense" | "tfidf" | "none" — for /health
_start_time = time.time()

# RAG (retrieval-augmented generation) layer.
# Populated on startup when a generator backend is configured
# (Ollama, Genspark, OpenAI, Groq, …). Kept optional so the bot
# still works in pure-retrieval mode when no LLM is available.
_generator: Optional[Generator] = None
_generator_ready: bool = False


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
    global _retriever, _generator, _generator_ready
    ensure_db(settings.db_path)
    _retriever = _load_retriever()
    print(
        f"[main] retriever ready — kind={_retriever_kind}, "
        f"{_retriever.size} entries, "  # type: ignore[attr-defined]
        f"{_retriever.vocab_size} "  # type: ignore[attr-defined]
        f"{'dims' if _retriever_kind == 'dense' else 'tokens'}"
    )

    # Optional RAG generator. Stays None if not configured → bot
    # serves KB answers verbatim (prior behaviour).
    _generator = build_generator()
    if _generator is not None:
        _generator_ready = await probe_generator(_generator)
        print(
            f"[main] generator backend={_generator.backend} "
            f"model={_generator.model} ready={_generator_ready}"
        )
    else:
        print("[main] no generator configured — retrieval-only mode")


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


# ─── Per-session conversation memory ────────────────────────────────
# In-process ring buffer of the last ~3 turn-pairs per session_token.
# Used to:
#   1. Continue a topic when a user sends a short affirmation ("yeah")
#      — we know what the bot just asked and can extend the answer.
#   2. Disambiguate follow-up questions ("how much?") by concatenating
#      the previous user query with the current one for retrieval.
#
# Why in-process + not SQLite-backed? Latency. The SQLite chat_messages
# table already persists everything for training export. This cache is
# just a hot read-path for retrieval-time context — stale entries
# eviction is automatic via the 30-minute TTL.

_SESSION_CACHE_MAX = 5_000          # cap on unique active sessions
_SESSION_TURNS_PER = 3              # keep last 3 turn-pairs per session
_SESSION_TTL_SECONDS = 1_800        # 30 minutes of idle = forget

# Session cache shape:
#   session_token -> {
#       "last_touch_ts": 1.7e9,
#       "turns": [
#           {"user": "...", "bot": "...", "matched_entry": "bw011", ...},
#           ...
#       ],
#   }
_session_cache: "OrderedDict[str, dict]" = OrderedDict()
_session_lock = threading.Lock()


def _session_record_turn(
    session_token: str,
    user_text: str,
    bot_text: str,
    match_type: str,
    matched_entry: str | None,
    match_score: float | None,
) -> None:
    """Push a user+bot turn pair into the session's ring buffer."""
    now = time.time()
    with _session_lock:
        entry = _session_cache.get(session_token)
        if entry is None:
            entry = {"last_touch_ts": now, "turns": []}
            _session_cache[session_token] = entry
        turns = entry["turns"]
        turns.append(
            {
                "user": user_text,
                "bot": bot_text,
                "match_type": match_type,
                "matched_entry": matched_entry,
                "match_score": match_score,
                "ts": now,
            }
        )
        # Keep only the last N turns
        if len(turns) > _SESSION_TURNS_PER:
            del turns[: len(turns) - _SESSION_TURNS_PER]
        entry["last_touch_ts"] = now
        _session_cache.move_to_end(session_token)
        # Evict expired + cap size
        _session_evict_locked(now)


def _session_evict_locked(now: float) -> None:
    """Remove expired + over-cap entries. Caller holds the lock."""
    expired = [
        k for k, v in _session_cache.items()
        if now - v["last_touch_ts"] > _SESSION_TTL_SECONDS
    ]
    for k in expired:
        _session_cache.pop(k, None)
    while len(_session_cache) > _SESSION_CACHE_MAX:
        _session_cache.popitem(last=False)


def _session_get_turns(session_token: str) -> list[dict]:
    """Return up to the last 3 turn-pairs for a session (empty if none)."""
    with _session_lock:
        entry = _session_cache.get(session_token)
        if entry is None:
            return []
        if time.time() - entry["last_touch_ts"] > _SESSION_TTL_SECONDS:
            _session_cache.pop(session_token, None)
            return []
        return list(entry["turns"])


def _session_previous_bot_turn(session_token: str) -> dict | None:
    """Just the most recent bot turn for this session, or None."""
    turns = _session_get_turns(session_token)
    return turns[-1] if turns else None


def _build_contextual_query(session_token: str, current_message: str) -> str:
    """Augment VERY short follow-ups ('yes', 'how much?', 'more details')
    with the previous user turn so the retriever has enough signal.

    Only 1-3 word messages qualify. A 4+ word message is already a
    specific question — augmenting it can cause the previous topic to
    dominate retrieval and lock the bot onto the prior entry. Classic
    bug: user asks "Is Al-Fardan Q9 Sharia-compliant?" then "What is
    your Proof of Reserves?" — with the old 6-word threshold, the
    second question was augmented into "Sharia … Proof of Reserves"
    and retrieval re-served the Sharia entry.
    """
    if len(current_message.split()) > 3:
        return current_message
    turns = _session_get_turns(session_token)
    if not turns:
        return current_message
    prev_user = turns[-1].get("user") or ""
    if prev_user and prev_user.strip().lower() != current_message.strip().lower():
        return f"{prev_user} {current_message}".strip()
    return current_message


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
    hot the cache is, whether fastembed imported cleanly, plus a peek
    at the live CRM settings + recent typo-correction examples. Handy
    when diagnosing without pulling container logs."""
    r = _retriever
    return {
        "retriever_kind": _retriever_kind,
        "entries": r.size if r else 0,  # type: ignore[attr-defined]
        "dim_or_vocab": r.vocab_size if r else 0,  # type: ignore[attr-defined]
        "confidence_threshold": _confidence_threshold(),
        "fastembed_available": dense_retrieval.available(),
        "fastembed_error": dense_retrieval.unavailable_reason() or None,
        "response_cache_entries": len(_response_cache),
        "response_cache_max": _CACHE_MAX_ENTRIES,
        "session_cache_sessions": len(_session_cache),
        "session_cache_max": _SESSION_CACHE_MAX,
        "platform_settings": _platform_settings.debug_snapshot(),
        "uptime_seconds": time.time() - _start_time,
    }


@app.get("/debug/settings")
async def debug_settings() -> dict:
    """What the chatbot currently reads as the source-of-truth for LTV,
    APY, fees etc. — pulled from the CRM /api/settings every 5 min.
    Edit the CRM admin page and this should reflect within that window
    (or instantly after POST /admin/reindex)."""
    return {
        "live": _platform_settings.get_settings(),
        "snapshot": _platform_settings.debug_snapshot(),
    }


@app.get("/debug/typos")
async def debug_typos(q: str) -> dict:
    """Show what typo corrections would apply to `q` without actually
    routing the message. Lets you sanity-check the typo dictionary
    against real inbound messages from /admin/conversations/export."""
    fixes = preview_corrections(q)
    return {
        "original": q,
        "corrected": correct_typos(q),
        "fixes": fixes,
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

    # ─── Typo correction — runs BEFORE everything else ───────────
    # Common crypto-term misspellings ("lonas", "etherium", "stakign")
    # get normalised so intent classification + retrieval + composer
    # all see a clean query. No-op for correctly-spelled inputs.
    normalised_message = correct_typos(req.message)
    if normalised_message != req.message:
        # Swap on a lightweight copy so the original is still logged
        # to SQLite below (useful for debugging what users actually
        # typed vs what the bot received).
        req = req.model_copy(update={"message": normalised_message})

    # ─── Emotion / mood detection ────────────────────────────────
    # Used AFTER retrieval/compose to (a) add an acknowledgement
    # prefix + (b) surface human-handoff buttons if the user is
    # frustrated or reporting something urgent.
    mood = detect_mood(req.message)

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

    # Scripted replies beat retrieval for well-defined intents.
    # For affirmations ("yeah", "sure"), we try to continue the topic
    # using session memory BEFORE falling back to the generic reply.
    scripted = scripted_reply(intent)
    if scripted is not None:
        # ─── Conversation-memory enhancement for affirmations ──────
        # If the user just said "yes/yeah/sure" and we have a previous
        # bot turn in this session, we give a topic-aware continuation
        # instead of the generic "what would you like to do next?"
        if intent == "affirmation":
            prev_bot = _session_previous_bot_turn(req.session_token)
            if prev_bot and prev_bot.get("matched_entry"):
                # Try to re-search the previous user query to fetch a
                # deeper follow-up answer in the same topic
                prev_user = prev_bot.get("user", "")
                if prev_user:
                    follow_hits = r.search(
                        f"{prev_user} tell me more details",
                        top_k=settings.top_k,
                    )
                    # Take a different entry than the previous one so
                    # the user gets fresh information, not a repeat.
                    picked = next(
                        (
                            h for h in (follow_hits or [])
                            if h.entry.id != prev_bot.get("matched_entry")
                            and h.score >= _confidence_threshold()
                        ),
                        None,
                    )
                    if picked:
                        follow_actions = _actions_for_entry(picked.entry)
                        follow_text = _humanize_answer(
                            picked.entry, picked.entry.answer, req.message
                        )
                        reply = ChatResponse(
                            reply=follow_text,
                            match_type="kb_hit",
                            matched_entry_id=picked.entry.id,
                            match_score=picked.score,
                            session_token=req.session_token,
                            actions=follow_actions or None,
                        )
                        record_message(
                            settings.db_path,
                            session_token=req.session_token,
                            role="bot",
                            text=reply.reply,
                            match_type=reply.match_type,
                            matched_entry=picked.entry.id,
                            match_score=picked.score,
                        )
                        _session_record_turn(
                            req.session_token, req.message, reply.reply,
                            reply.match_type, picked.entry.id, picked.score,
                        )
                        return reply

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
        _session_record_turn(
            req.session_token, req.message, reply.reply,
            reply.match_type, None, None,
        )
        return reply

    # ─── Fact composer — no-LLM generative pass ─────────────────
    # Extract structured entities (amounts, assets, actions, crisis/
    # beginner flags) and try to COMPOSE a fresh answer from the
    # business-facts in app/compose/facts.py. If the composer is
    # confident (covered pattern), we use its reply and skip retrieval
    # entirely. If not, fall through to the semantic retriever.
    #
    # This scales better than hardcoding every phrasing: one fact
    # answers every variant of "I have $X can I join".
    entities = extract_entities(req.message)
    composed = compose_reply(entities)
    if composed is not None:
        composed_text, composed_actions = composed
        composed_text = composed_text.strip()
        composed_action_list = [
            ChatAction(label=a["label"], url=a["url"], kind=a.get("kind", "link"))
            for a in (composed_actions or [])
        ] or None
        response = ChatResponse(
            reply=composed_text,
            match_type="kb_hit",
            matched_entry_id="composer",
            match_score=0.99,
            session_token=req.session_token,
            actions=composed_action_list,
        )
        record_message(
            settings.db_path,
            session_token=req.session_token,
            role="bot",
            text=composed_text,
            match_type="kb_hit",
            matched_entry="composer",
            match_score=0.99,
        )
        _session_record_turn(
            req.session_token, req.message, composed_text,
            "kb_hit", "composer", 0.99,
        )
        # Cache the composed answer so repeated identical queries are
        # served sub-ms.
        if len(req.message) >= 3:
            _cache_put(
                req.message,
                response.model_dump(exclude={"session_token"}),
            )
        return response

    # Fall through to semantic retrieval (dense) or TF-IDF (fallback)
    # For short follow-up queries we augment with the previous user turn
    # so the retriever has enough context to resolve "how much?" etc.
    # Then expand the query with domain synonyms (borrow → loan/credit/
    # lending/murabaha) so the retriever can match entries using
    # different wording than the user chose.
    search_query = _build_contextual_query(req.session_token, req.message)
    expanded_query = expand_synonyms(search_query)
    hits = r.search(expanded_query, top_k=settings.top_k)
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
        _session_record_turn(
            req.session_token, req.message, reply_text,
            response.match_type, nearest, best_score,
        )
        return response

    top = hits[0]
    entry_actions = _actions_for_entry(top.entry)
    humanized = _humanize_answer(top.entry, top.entry.answer, req.message)

    # ─── RAG (generative) layer ───────────────────────────────────
    # When a generator is configured AND healthy, rewrite the canned
    # KB answer into a natural-language reply grounded on the top-N
    # references. Retains the KB answer as a safety net — if the
    # LLM call fails, times out, or produces an empty string we fall
    # straight back to `humanized` and the user sees no difference.
    match_type_used = "kb_hit"
    if _generator_ready and _generator is not None:
        refs = [
            {
                "id": h.entry.id,
                "category": h.entry.category,
                "question": h.entry.question,
                "answer": h.entry.answer,
            }
            for h in hits[:MAX_REFERENCES]
        ]
        try:
            history = get_history(
                settings.db_path, req.session_token, limit=8
            )
        except Exception:  # noqa: BLE001
            history = []
        user_prompt = build_user_prompt(
            question=req.message,
            references=refs,
            session_history=history,
        )
        try:
            gen_result = await _generator.generate(
                system=SYSTEM_PROMPT,
                user=user_prompt,
            )
            if gen_result.text and len(gen_result.text) >= 20:
                # Language-fidelity sanity check — catches llama3.2:3b
                # drifting into French / Spanish on English queries. If
                # the user's message looks English but the reply has
                # telltale non-English phrases, fall back to the KB
                # answer instead of shipping a wrong-language response.
                if _is_language_mismatch(req.message, gen_result.text):
                    print(
                        f"[rag] language drift detected — "
                        f"user={_quick_lang(req.message)}, "
                        f"reply looks non-{_quick_lang(req.message)}; "
                        f"falling back to KB answer for {top.entry.id}"
                    )
                else:
                    humanized = gen_result.text
                    match_type_used = f"rag_{gen_result.backend}"
                    print(
                        f"[rag] {gen_result.backend} ok — "
                        f"{gen_result.latency_ms}ms, "
                        f"{len(gen_result.text)} chars, "
                        f"top_ref={top.entry.id} score={top.score:.3f}"
                    )
        except Exception as err:  # noqa: BLE001
            # Never crash /chat on LLM failure — fall back silently.
            print(f"[rag] generation failed: {err!r} — using KB answer")

    # ─── Apply mood-aware adjustments ─────────────────────────────
    # Prepend a short acknowledgement if the user sounded frustrated,
    # urgent, confused, or excited. Surface human-handoff buttons as
    # the FIRST actions when mood warrants (frustrated/urgent).
    if mood.has_any:
        ack = acknowledgment(mood)
        if ack:
            humanized = ack + humanized
    mood_actions = [
        ChatAction(label=a["label"], url=a["url"], kind=a.get("kind", "link"))
        for a in escalation_actions(mood)
    ]
    final_actions = (mood_actions + (entry_actions or [])) or None

    response = ChatResponse(
        reply=humanized,
        match_type=match_type_used,
        matched_entry_id=top.entry.id,
        match_score=top.score,
        session_token=req.session_token,
        actions=final_actions,
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
        text=humanized,
        match_type=match_type_used,
        matched_entry=top.entry.id,
        match_score=top.score,
    )
    _session_record_turn(
        req.session_token, req.message, humanized,
        match_type_used, top.entry.id, top.score,
    )
    return response


@app.get("/chat/history")
async def chat_history(session_token: str, limit: int = 50):
    """Return a session's chat history so the widget can restore the
    conversation UI after a page reload. Public endpoint (the session
    token is effectively the capability — only the holder of the token
    can read that conversation)."""
    return {"data": get_history(settings.db_path, session_token, limit)}


# ─── Shared helper — used by /chat AND /voice/respond ─────────────────

async def _handle_chat_text(
    session_token: str,
    message: str,
    source: str = "chat",
    user_id: str | None = None,
    user_email: str | None = None,
) -> str:
    """Lightweight wrapper for non-HTTP callers (e.g. the voice
    webhook). Returns just the reply text — no TwiML, no action
    buttons, no intent metadata. The voice layer strips formatting
    before calling TTS, so we deliberately keep the output plain.

    We build a synthetic ChatRequest, invoke the same `chat` handler
    the widget uses, and unwrap the text. This guarantees the phone
    agent's answers are identical to the widget's for the same
    question — same KB, same RAG, same Safiya persona.
    """
    from fastapi import Request as _Req  # local import to avoid loop

    req = ChatRequest(
        session_token=session_token,
        message=message,
        user_id=user_id,
        user_email=user_email,
    )

    # Build a minimal scope so the `chat()` endpoint's Request param
    # is present but we never read it for anything voice-relevant
    # (rate-limiting keys on IP which we don't have for a phone call).
    scope = {
        "type": "http",
        "headers": [(b"x-forwarded-for", b"voice-agent")],
        "method": "POST",
        "path": "/chat",
        "query_string": b"",
    }
    fake_request = _Req(scope)

    response: ChatResponse = await chat(  # type: ignore[misc]
        req,
        fake_request,
        r=_retriever,
    )
    # For voice, strip any leading sales-y bridge words the widget
    # prepends (e.g. mood acknowledgements like "I hear you —").
    text = response.reply or ""
    # Replace Markdown-ish list bullets with pauses; TTS reads "*" as
    # "asterisk" which sounds awful.
    text = (
        text.replace("\n\n", ". ")
        .replace("\n", ". ")
        .replace(" • ", ". ")
        .replace(" * ", ". ")
        .replace(" — ", ", ")
    )
    # Log source for debugging (voice vs widget use the same brain
    # but we want to separate metrics later).
    print(
        f"[handle_chat_text] source={source} session={session_token[:8]} "
        f"len={len(text)} match={response.match_type}"
    )
    return text


# ─── Admin / integration ──────────────────────────────────────────────

@app.post("/admin/reindex", response_model=ReindexResponse)
async def admin_reindex(_: None = Depends(require_admin_secret)) -> ReindexResponse:
    """Rebuild the index — call after editing the KB. Also forces a
    refresh of the live CRM platform-settings cache so an admin who
    just edited /admin/settings doesn't have to wait up to 5 minutes
    for the chatbot to pick up their changes.

    Prefers the dense retriever if fastembed is available; falls back
    to TF-IDF. Clears the response cache since answers may have changed.
    """
    global _retriever, _retriever_kind
    t0 = time.time()
    # Force-refresh the live settings cache — admins often reindex
    # right after editing a number in the CRM admin page.
    try:
        _platform_settings.force_refresh()
    except Exception as err:  # noqa: BLE001
        print(f"[reindex] platform_settings refresh failed: {err}")
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


# ─── Conversation storage + training export ───────────────────────────
# Every user + bot message is already written to SQLite in chat.db
# (/app/app/state/chat.db on the VPS, persisted across container
# restarts via the chatbot-state volume). These endpoints surface that
# data for training / KB curation work without exposing anything over
# the public /chat route.

@app.get("/admin/conversations/stats")
async def admin_conversation_stats(
    _: None = Depends(require_admin_secret),
) -> dict:
    """Quick numbers: how many messages, sessions, unanswered queries."""
    return message_stats(settings.db_path)


@app.get("/admin/conversations/export")
async def admin_conversations_export(
    since: Optional[str] = None,
    limit: int = 10_000,
    _: None = Depends(require_admin_secret),
) -> dict:
    """All chat messages since `since` (ISO datetime), ready for
    offline training / analysis. Default limit 10k — raise via query
    param if you need more.

    Output shape: {count: N, messages: [...]} — each message has
    role, text, matched_entry, score, session_token, created_at.
    """
    rows = list_all_messages(settings.db_path, since_iso=since, limit=limit)
    return {"count": len(rows), "messages": rows}


@app.get("/admin/conversations/pairs")
async def admin_conversation_pairs(
    since: Optional[str] = None,
    limit: int = 5_000,
    _: None = Depends(require_admin_secret),
) -> dict:
    """User+bot turn pairs — the format you'd hand to a curation
    workflow. Each pair has the raw user question, the bot's reply,
    the match type + score + matched KB entry.

    Filter client-side on match_type='low_confidence' to find the
    queries that need new KB entries, or match_type='kb_hit' with
    low scores to find answers that should be tuned.
    """
    pairs = training_turn_pairs(settings.db_path, since_iso=since, limit=limit)
    return {"count": len(pairs), "pairs": pairs}
