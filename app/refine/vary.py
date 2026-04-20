"""Response variety for scripted intent replies — institutional register.

Every variant follows the Institutional Voice Protocol:
  - Direct, factual, minimal framing
  - No hype tokens (perfect, great, happy to help, awesome, absolutely,
    fully insured, guaranteed, best)
  - No emojis, no exclamation marks
  - No retail CTAs ("want me to…", "shall I…", "ping me")
  - No upsell unless the user explicitly asked

Persona: Safiya Al Suwaidi — Institutional Client Advisor at
Al-Fardan Q9. Register: private banking / DIFC advisor.
"""

from __future__ import annotations

import random


# Per-intent template pools. Each variant is self-contained — the
# caller picks one at random per reply.

GREETING_VARIANTS = (
    "Assalamu alaikum. Safiya Al Suwaidi, Institutional Client Advisor "
    "at Al-Fardan Q9. How can I help today?",

    "Good day. Safiya Al Suwaidi here, Institutional Client Advisor, "
    "Al-Fardan Q9. What would you like to discuss?",

    "Welcome to Al-Fardan Q9. I'm Safiya Al Suwaidi, Institutional "
    "Client Advisor. Please go ahead with your question.",

    "Safiya Al Suwaidi, Al-Fardan Q9 Institutional Client Advisor. "
    "How may I assist.",
)

GOODBYE_VARIANTS = (
    "Thank you. If you need further assistance, "
    "institutional@alfardanq9.com is staffed 24/7.",

    "Noted. Our team is available at institutional@alfardanq9.com "
    "should you require further information.",

    "Thank you for your time. Al-Fardan Q9's client desk remains "
    "available at institutional@alfardanq9.com.",
)

AFFIRMATION_VARIANTS = (
    "Understood. Please specify which product or policy you would "
    "like me to address.",

    "Noted. Which area would you like to cover — Custody, Staking, "
    "OTC, Lending, or compliance?",

    "Acknowledged. Please indicate the specific topic.",
)

NEGATION_VARIANTS = (
    "Understood. If you have further questions, please proceed.",

    "Noted. I remain available should you require information on "
    "products, fees, or our regulatory framework.",

    "Acknowledged.",
)

SIGNUP_VARIANTS = (
    "Account onboarding follows our standard KYC process — identity "
    "verification, source-of-funds declaration, and tier assignment. "
    "Please indicate which product you intend to use (Custody, Staking, "
    "OTC, or Lending) and I will route accordingly.",

    "To open an account, we require standard KYC documentation. "
    "Individual clients: government ID, proof of address, and "
    "source-of-funds declaration. Institutional clients: additional "
    "entity documents. Which category applies.",

    "Onboarding typically completes within 24-48 hours for individual "
    "accounts, 3-5 business days for institutional accounts. Please "
    "indicate which service you intend to use.",
)

CONTACT_SUPPORT_VARIANTS = (
    "Our client desk is reachable at institutional@alfardanq9.com. "
    "Response time is typically within 2 UAE business hours. For "
    "urgent matters, the 24/7 line is listed in your dashboard.",

    "Please contact institutional@alfardanq9.com for non-urgent "
    "matters, or the 24/7 line in your dashboard for time-sensitive "
    "issues. For OTC execution, Layla Al-Fayez, Head of Institutional "
    "OTC Desk, handles bespoke requests directly.",

    "For direct human assistance, email institutional@alfardanq9.com. "
    "Urgent security or withdrawal matters should use the emergency "
    "line at +971 4 123 4568.",
)

BALANCE_VARIANTS = (
    "Account balances are accessible through your Portfolio and "
    "Wallets pages within the client portal.",

    "Holdings are displayed on the Portfolio page (consolidated view) "
    "and Wallets page (per-asset breakdown) within the client portal.",

    "Please access your client portal to review holdings. The "
    "Portfolio page provides consolidated positioning; Wallets provides "
    "per-asset detail.",
)

# Mapping intent name → tuple of variants. Keep in sync with intent.py.
_INTENT_POOLS = {
    "greeting": GREETING_VARIANTS,
    "goodbye": GOODBYE_VARIANTS,
    "affirmation": AFFIRMATION_VARIANTS,
    "negation": NEGATION_VARIANTS,
    "signup": SIGNUP_VARIANTS,
    "contact_support": CONTACT_SUPPORT_VARIANTS,
    "balance_check": BALANCE_VARIANTS,
}


def pick_variant(intent: str) -> str | None:
    """Return one random variant for the intent, or None if no pool
    exists (caller falls back to its own scripted reply)."""
    pool = _INTENT_POOLS.get(intent)
    if not pool:
        return None
    return random.choice(pool)
