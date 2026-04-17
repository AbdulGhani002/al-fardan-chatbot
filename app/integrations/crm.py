"""HTTP client for calls FROM the chatbot TO the Al-Fardan CRM."""

from __future__ import annotations

import httpx

from ..config import settings
from ..models import LeadCaptureRequest


async def post_lead_to_crm(lead: LeadCaptureRequest) -> dict:
    """POST /api/chatbot/lead — creates user + sends OTP.

    The CRM endpoint accepts the same fields as /api/auth/register
    plus a consent_timestamp + source="chatbot". It returns the new
    user id or an error.

    Auth via shared secret header so the CRM knows the request came
    from our trusted chatbot container and not a spoofed caller.
    """
    if not settings.crm_api_key:
        # No secret configured — in dev we just log and noop.
        return {"skipped": True, "reason": "CRM_API_KEY unset"}

    body = {
        "firstName": lead.first_name,
        "lastName": lead.last_name,
        "email": lead.email,
        "phone": lead.phone,
        "service": lead.service,
        "consentTimestamp": lead.consent_timestamp.isoformat(),
        "source": "chatbot",
    }

    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.post(
            f"{settings.crm_base_url.rstrip('/')}/api/chatbot/lead",
            json=body,
            headers={"x-chatbot-secret": settings.chatbot_secret},
        )
        try:
            payload = resp.json()
        except Exception:  # noqa: BLE001
            payload = {"raw": resp.text}
        return {"status": resp.status_code, **(payload or {})}
