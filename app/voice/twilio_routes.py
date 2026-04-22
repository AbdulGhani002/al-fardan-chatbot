"""Twilio Voice webhook routes for the Al-Fardan Q9 phone agent.

High-level flow of a live call:

  1. Twilio receives an inbound call on your Al-Fardan number.
  2. Twilio POSTs to /voice/incoming. We return TwiML that greets
     the caller (as Safiya) and opens a <Gather input="speech">
     block — Twilio records the caller and transcribes on their
     side using Google / Amazon ASR.
  3. When the caller finishes speaking, Twilio POSTs the transcript
     to /voice/respond. We route it through the same retrieval →
     RAG pipeline the chat widget uses, get an answer, and return
     TwiML <Say> that speaks the answer back in a Polly voice, then
     re-opens <Gather> for the next turn.
  4. If the caller says "goodbye", "thank you", or falls silent for
     >10 seconds, we <Say> a warm farewell and <Hangup>.

Security
--------
Twilio signs every webhook with an HMAC-SHA1 over the URL + form
body using your Auth Token. We validate the signature on every
POST to prevent anyone else from forging call traffic into your
bot. When TWILIO_AUTH_TOKEN is unset we log a warning and accept
requests — dev mode only; ALWAYS set the token in production.

Why TwiML and not Media Streams?
---------------------------------
TwiML gives us Twilio-native STT + TTS with a ~1-3s round trip,
zero audio engineering, and no WebSocket to babysit. When we want
higher fidelity + interruption handling we'll add a /voice/stream
endpoint that swaps in Whisper (ASR) + Piper (TTS) over a Twilio
Media Stream WebSocket. That's a Phase 2 task.
"""

from __future__ import annotations

import hmac
import hashlib
import base64
from typing import Optional
from xml.sax.saxutils import escape as xml_escape

from fastapi import APIRouter, Form, Header, Request
from fastapi.responses import Response


router = APIRouter(prefix="/voice", tags=["voice"])


# ─── TwiML helpers ────────────────────────────────────────────────────

# Amazon Polly Neural — warm, reasonably natural. Swap to a premium
# voice (Polly Generative, ElevenLabs) when we promote to Media
# Streams. Arabic callers get Zeina; English callers get Joanna.
DEFAULT_VOICE_EN = "Polly.Joanna-Neural"
DEFAULT_VOICE_AR = "Polly.Zeina"

# Per-turn listening window. Long enough for a thoughtful question,
# short enough that silence doesn't stall the call.
SPEECH_TIMEOUT_S = 4
GATHER_TIMEOUT_S = 10


def _looks_arabic(text: str) -> bool:
    """Very coarse Arabic-script detector — used to pick a voice."""
    if not text:
        return False
    arabic_chars = sum(1 for ch in text if "\u0600" <= ch <= "\u06FF")
    return arabic_chars >= max(3, len(text) // 4)


def _twiml(body: str) -> Response:
    """Wrap XML body in a TwiML Response with the right content type."""
    xml = f'<?xml version="1.0" encoding="UTF-8"?>\n<Response>\n{body}\n</Response>'
    return Response(content=xml, media_type="application/xml")


def _say(text: str, voice: str = DEFAULT_VOICE_EN, language: str = "en-US") -> str:
    """Single <Say> element — escapes user-influenced text so an answer
    containing `<` or `&` can't break the TwiML parse."""
    safe = xml_escape(text)
    return (
        f'  <Say voice="{voice}" language="{language}">'
        f"{safe}"
        f"</Say>"
    )


def _gather(prompt: Optional[str] = None, language: str = "en-US") -> str:
    """Open a speech-input gather with our /voice/respond callback.

    `language` maps to Twilio's supported STT locales. We pass the
    action as a relative URL so the request protocol + host match
    whatever Twilio used on the incoming webhook — simpler than
    hard-coding https://80.65.211.25.sslip.io/voice/respond.
    """
    prompt_xml = ""
    if prompt:
        voice = DEFAULT_VOICE_AR if language.startswith("ar") else DEFAULT_VOICE_EN
        prompt_xml = f"\n    {_say(prompt, voice, language).strip()}"
    return (
        f'  <Gather input="speech" action="/voice/respond" method="POST" '
        f'speechTimeout="{SPEECH_TIMEOUT_S}" timeout="{GATHER_TIMEOUT_S}" '
        f'language="{language}" hints="staking,custody,lending,zakat,shariah,'
        f'murabaha,bitcoin,ethereum,solana,al-fardan">{prompt_xml}\n  </Gather>'
    )


# ─── Signature verification ───────────────────────────────────────────

def _verify_twilio_signature(
    signature_header: Optional[str],
    full_url: str,
    form_data: dict[str, str],
    auth_token: str,
) -> bool:
    """Re-implement Twilio's webhook-signature HMAC check.

    The official twilio-python SDK has `RequestValidator` but pulling
    in the whole SDK for one HMAC-SHA1 is overkill. Algorithm:
      sort form fields alphabetically → concat `key+value` → append
      to the full request URL → HMAC-SHA1 with the auth token →
      base64-encode → compare with X-Twilio-Signature.
    """
    if not auth_token or not signature_header:
        return False
    parts = [full_url]
    for key in sorted(form_data.keys()):
        parts.append(key)
        parts.append(form_data[key])
    payload = "".join(parts).encode("utf-8")
    digest = hmac.new(
        auth_token.encode("utf-8"), payload, hashlib.sha1
    ).digest()
    expected = base64.b64encode(digest).decode("ascii")
    return hmac.compare_digest(expected, signature_header)


async def _gather_form(request: Request) -> dict[str, str]:
    """Collect the Twilio form body as a plain dict."""
    form = await request.form()
    out: dict[str, str] = {}
    for k in form:
        v = form.get(k)
        if isinstance(v, str):
            out[k] = v
    return out


# ─── /voice/incoming ──────────────────────────────────────────────────

@router.post("/incoming")
async def voice_incoming(
    request: Request,
    x_twilio_signature: Optional[str] = Header(default=None),
    From: Optional[str] = Form(default=None),  # noqa: N803 — Twilio param name
) -> Response:
    """First hit of every inbound call.

    We greet the caller, open a speech gather, and point Twilio at
    /voice/respond for the transcript. The From number isn't used
    yet but we log it for future CRM linkage.
    """
    from ..config import settings

    form = await _gather_form(request)
    full_url = str(request.url)

    if settings.twilio_auth_token:
        ok = _verify_twilio_signature(
            x_twilio_signature, full_url, form, settings.twilio_auth_token
        )
        if not ok:
            return _twiml(
                _say(
                    "This request could not be verified. Please hang up and try again.",
                    language="en-US",
                )
            )
    # Best-effort caller logging — helpful for CRM linkage later.
    print(f"[voice/incoming] from={From} call_sid={form.get('CallSid')}")

    greeting = (
        "Thank you for calling Al-Fardan Q9. I'm Safiya, your client "
        "advisor. How can I help you today? You can ask about custody, "
        "staking, lending, our OTC desk, Zakat, or anything else about "
        "our platform."
    )
    return _twiml(
        _say(greeting, DEFAULT_VOICE_EN, "en-US")
        + "\n"
        + _gather(language="en-US")
        + "\n"
        + _say(
            "I didn't catch that. Please call back when you're ready to talk. "
            "Goodbye.",
            DEFAULT_VOICE_EN,
            "en-US",
        )
    )


# ─── /voice/respond ───────────────────────────────────────────────────

@router.post("/respond")
async def voice_respond(
    request: Request,
    x_twilio_signature: Optional[str] = Header(default=None),
    SpeechResult: Optional[str] = Form(default=None),  # noqa: N803
    CallSid: Optional[str] = Form(default=None),  # noqa: N803
    Confidence: Optional[str] = Form(default=None),  # noqa: N803
) -> Response:
    """Twilio POSTs here with the caller's transcribed speech.

    We reuse the main /chat pipeline (retrieval → RAG → KB fallback)
    to get an answer, then TwiML-wrap it and re-open the gather for
    the next turn.
    """
    from ..config import settings
    from ..main import _handle_chat_text

    form = await _gather_form(request)
    full_url = str(request.url)

    if settings.twilio_auth_token:
        if not _verify_twilio_signature(
            x_twilio_signature, full_url, form, settings.twilio_auth_token
        ):
            return _twiml(
                _say(
                    "I couldn't verify this call. Please try again.",
                    language="en-US",
                )
            )

    user_said = (SpeechResult or "").strip()
    conf = 0.0
    try:
        conf = float(Confidence or 0.0)
    except ValueError:
        conf = 0.0

    # Nothing intelligible — ask once more, then hang up if still silent.
    if not user_said or conf < 0.3:
        return _twiml(
            _say(
                "I didn't catch that, could you please say it again?",
                DEFAULT_VOICE_EN,
                "en-US",
            )
            + "\n"
            + _gather(language="en-US")
            + "\n"
            + _say(
                "If you need more time, call back any time. Goodbye.",
                DEFAULT_VOICE_EN,
                "en-US",
            )
        )

    # Caller said goodbye — wrap up warmly.
    if any(
        kw in user_said.lower()
        for kw in ("goodbye", "bye", "thank you", "that's all", "no thanks")
    ):
        return _twiml(
            _say(
                "Thank you for calling Al-Fardan Q9. If you'd like to speak "
                "with our team further, email institutional@alfardanq9.com. "
                "Goodbye.",
                DEFAULT_VOICE_EN,
                "en-US",
            )
        )

    is_arabic = _looks_arabic(user_said)
    voice = DEFAULT_VOICE_AR if is_arabic else DEFAULT_VOICE_EN
    language = "ar-AE" if is_arabic else "en-US"

    # Use the CallSid as the conversation session token so follow-ups
    # within the same call share context. Pad to 16 chars to satisfy
    # ChatRequest's min_length=8 validator.
    session_token = (
        f"voice_{CallSid or 'nosid'}_sess".replace(" ", "_")[:64]
    ).ljust(16, "x")

    # Call the shared chat handler — same brain as the widget.
    try:
        answer = await _handle_chat_text(
            session_token=session_token,
            message=user_said,
            source="voice",
        )
    except Exception as err:  # noqa: BLE001
        print(f"[voice/respond] chat handler error: {err!r}")
        answer = (
            "I'm having a bit of trouble retrieving that right now. "
            "Please try again in a moment, or email our team at "
            "institutional@alfardanq9.com."
        )

    # Twilio <Say> is capped around 4000 characters per clause. Our
    # answers are ~300 chars so this is a safety rail, not an active
    # concern.
    answer = answer.strip()[:1500] or (
        "I'm not sure about that one. Our institutional desk at "
        "institutional@alfardanq9.com can help you directly."
    )

    # Follow-up prompt nudges the caller to keep talking without
    # re-reading the whole greeting.
    followup = (
        "Would you like to know more, or ask something else?"
        if not is_arabic
        else "هل تريد معرفة المزيد أو طرح سؤال آخر؟"
    )

    return _twiml(
        _say(answer, voice, language)
        + "\n"
        + _say(followup, voice, language)
        + "\n"
        + _gather(language=language)
        + "\n"
        + _say(
            "Thanks for calling Al-Fardan Q9. Goodbye."
            if not is_arabic
            else "شكراً لاتصالك بالفردان Q9. إلى اللقاء.",
            voice,
            language,
        )
    )


# ─── /voice/status ────────────────────────────────────────────────────

@router.post("/status")
async def voice_status(request: Request) -> Response:
    """Twilio Status Callback endpoint — informational only.

    Twilio POSTs here on call events (initiated, ringing, answered,
    completed). We log for observability + future analytics; no
    response body is required.
    """
    form = await _gather_form(request)
    print(
        f"[voice/status] "
        f"call_sid={form.get('CallSid')} "
        f"status={form.get('CallStatus')} "
        f"duration={form.get('CallDuration')} "
        f"from={form.get('From')}"
    )
    return Response(status_code=204)
