"""Response variety for scripted intent replies.

Human conversations don't repeat verbatim. When a user says "hi"
three times in a row they shouldn't get the same 40-word intro each
time. This module returns a randomly-picked phrasing from a small
pool per intent.

Stateless — no session awareness at this layer. The session cache
in main.py handles continuity; this just breaks up the "every reply
feels identical" tell that makes chatbots feel canned.
"""

from __future__ import annotations

import random


# Per-intent template pools. Each variant is self-contained — the
# caller picks one at random per reply. Keep each under ~50 words so
# the bot stays concise.

GREETING_VARIANTS = (
    "Assalamu alaikum — I'm Safiya Al Suwaidi, Client Acquisition & "
    "Growth Manager at Al-Fardan Q9. I help new investors get started "
    "with Custody, Staking, OTC, and Lending. What brings you here today?",

    "Hi — Safiya Al Suwaidi here. I'm Client Acquisition & Growth "
    "Manager at Al-Fardan Q9. Whether it's custody, staking, OTC, or "
    "lending, happy to help. What are you looking into?",

    "Hey — I'm Safiya, Client Acquisition & Growth Manager at Al-Fardan "
    "Q9. I help new investors navigate the platform. What can I help "
    "with today?",

    "Welcome to Al-Fardan Q9. I'm Safiya Al Suwaidi, your starting "
    "point — Custody, Staking, OTC, Lending, or something else on your "
    "mind?",
)

GOODBYE_VARIANTS = (
    "Take care — I'm here whenever you need. Urgent? "
    "institutional@alfardanq9.com covers you 24/7.",

    "Talk soon. Ping me anytime — or reach the desk directly at "
    "institutional@alfardanq9.com.",

    "All the best. I'll be right here when you're back; the 24/7 desk "
    "is at institutional@alfardanq9.com.",

    "Thanks for stopping by. Ping anytime — institutional@alfardanq9.com "
    "is always staffed if it's urgent.",
)

AFFIRMATION_VARIANTS = (
    "Excellent — happy to help. What would you like to do next? "
    "Open an account, review our products, or discuss something specific?",

    "Perfect. How may I assist from here? I can walk you through signup, "
    "explain a product in detail, or connect you with our desk.",

    "Understood — let's continue. What would you like to cover next? "
    "A specific product, a number, or the onboarding flow?",

    "Of course. Would you like me to walk you through the first step, "
    "or is there a specific aspect you would like me to explain?",
)

NEGATION_VARIANTS = (
    "No concern — please take your time. What else may I help you with? "
    "I can explain products, fees, or how we are regulated.",

    "Of course. Is there anything else I can clarify? Products, fees, "
    "or our operating framework.",

    "Understood — no rush. If anything else comes to mind, I am here. "
    "Would you like to explore another topic?",

    "Absolutely. When you are ready, please ask. I can cover any "
    "product, pricing structure, or compliance detail.",
)

SIGNUP_VARIANTS = (
    "Happy to help you open an account — takes about 5 minutes. "
    "Which service are you most interested in first?",

    "Great — let's get you set up. Name, email, phone, password, OTP "
    "verification, then KYC. Which product drew you in?",

    "Welcome aboard. Account creation is 5 minutes, KYC is 24-48 hours. "
    "Which service should we centre the onboarding on?",
)

CONTACT_SUPPORT_VARIANTS = (
    "I'll route you to a human. Fastest options: email "
    "institutional@alfardanq9.com (response under 2 UAE business hours) "
    "or a ticket from Settings → Support.",

    "I can connect you with our team. Email institutional@alfardanq9.com "
    "gets a reply within 2 UAE business hours. Urgent? The 24/7 line is "
    "in your dashboard.",

    "Let's get you a human. Email is fastest: institutional@alfardanq9.com. "
    "Or file a ticket from Settings → Support. I'm also happy to take a "
    "message for Layla if it's OTC-related.",
)

BALANCE_VARIANTS = (
    "To see your balances, open your Portfolio or Wallets page in the "
    "portal. Quick-access buttons below — tap one to jump straight "
    "there.",

    "Balances live on your Portfolio and Wallets pages. Links below.",

    "Your holdings are on the Portfolio page (aggregated) and Wallets "
    "(per asset). Tap below to jump in.",
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
