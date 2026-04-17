"""Rule-based intent classifier — runs BEFORE TF-IDF search.

Why rules and not a model?
  - Greetings, goodbyes, and signup intents are tiny, closed sets with
    obvious phrasings. A regex matcher is more reliable + auditable
    than a fine-tuned classifier at this volume.
  - Zero training data required.
  - Easy for an admin to extend.
"""

from __future__ import annotations

import re
from typing import Literal


Intent = Literal[
    "greeting",
    "goodbye",
    "signup",
    "balance_check",
    "loan_question",
    "staking_question",
    "unknown",
]


_GREETING_RE = re.compile(
    r"\b(hi|hello|hey|salaam|assalamu|hola|bonjour|good\s+(morning|afternoon|evening))\b",
    re.I,
)
_GOODBYE_RE = re.compile(
    r"\b(bye|goodbye|thanks|thank\s+you|shukran|cya|see\s+you)\b", re.I
)
_SIGNUP_RE = re.compile(
    r"\b("
    r"sign\s*up|signup|register|create\s+(an?\s+)?(account|profile)|"
    r"open\s+(an?\s+)?(account|wallet)|get\s+started|how\s+do\s+i\s+start|"
    r"i\s+want\s+to\s+(invest|join|start|sign\s+up)"
    r")\b",
    re.I,
)
_BALANCE_RE = re.compile(
    r"\b(balance|my\s+(portfolio|holdings?|assets?|wallet)|how\s+much\s+(do\s+)?i\s+have)\b",
    re.I,
)
_LOAN_RE = re.compile(
    r"\b(loan|borrow|lending|ltv|interest\s+rate|apr|credit\s+line|murabaha)\b",
    re.I,
)
_STAKING_RE = re.compile(
    r"\b(stake|staking|yield|apy|rewards?|validator)\b", re.I
)


def classify(text: str) -> Intent:
    """Return the first matching intent or 'unknown'.

    Order matters: signup is checked before the broader greeting so
    "hi i want to sign up" routes as signup, not greeting.
    """
    if not text or not text.strip():
        return "unknown"

    if _SIGNUP_RE.search(text):
        return "signup"
    if _BALANCE_RE.search(text):
        return "balance_check"
    if _LOAN_RE.search(text):
        return "loan_question"
    if _STAKING_RE.search(text):
        return "staking_question"
    # Greetings + goodbyes matched last so specific intents win
    if _GREETING_RE.search(text):
        return "greeting"
    if _GOODBYE_RE.search(text):
        return "goodbye"
    return "unknown"


def scripted_reply(intent: Intent) -> str | None:
    """Canned responses for greeting / goodbye / signup prompts."""
    if intent == "greeting":
        return (
            "Assalamu alaikum! I'm Al-Fardan Q9's assistant. I can help you with "
            "Custody, Staking, OTC Desk, or Lending — or answer questions about "
            "crypto in general. What brings you here today?"
        )
    if intent == "goodbye":
        return (
            "Thank you for reaching out. If you need anything else, you can reach "
            "your relationship manager through the portal, or just open this chat "
            "again. Have a great day."
        )
    if intent == "signup":
        return (
            "Happy to help you open an account. We offer four services — Custody, "
            "Staking, OTC Desk, and Lending. Which one interests you most? "
            "(You can say \"all four\" if you'd like to keep your options open.)"
        )
    if intent == "balance_check":
        return (
            "I can't pull live balances here for privacy reasons — please sign in "
            "to your portal and check the Wallets page, or contact your "
            "relationship manager."
        )
    return None
