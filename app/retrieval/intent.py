"""Rule-based intent classifier — runs BEFORE TF-IDF search.

Why rules and not a model?
  - Greetings, goodbyes, and signup intents are tiny, closed sets with
    obvious phrasings. A regex matcher is more reliable + auditable
    than a fine-tuned classifier at this volume.
  - Zero training data required.
  - Easy for an admin to extend.

Intents with "navigate_*" return a scripted reply PLUS interactive
action buttons that deep-link the user to the right CRM page. This
lets the bot feel agent-ish — asking "how do I stake?" returns an
answer AND a clickable [Open Staking →] button.
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
    "navigate_staking",
    "navigate_lending",
    "navigate_custody",
    "navigate_otc",
    "navigate_portfolio",
    "navigate_settings",
    "navigate_wallets",
    "withdraw_request",
    "deposit_request",
    "contact_support",
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
    r"\b(balance|my\s+(portfolio|holdings?|assets?|wallet)|"
    r"how\s+much\s+(do\s+)?i\s+have|net\s+worth|total\s+value)\b",
    re.I,
)

# ─── Action intents — "I want to DO X on your platform" ─────────────
# These routes trigger navigation buttons so the bot feels agent-ish.

_START_STAKING_RE = re.compile(
    r"\b(start\s+staking|begin\s+staking|i\s+want\s+to\s+stake|"
    r"let\s+me\s+stake|take\s+me\s+to\s+staking|go\s+to\s+staking|"
    r"open\s+staking|how\s+(do|to)\s+stake|stake\s+(now|my|some|"
    r"for\s+me|some\s+eth|some\s+sol|some\s+bitcoin|some\s+crypto)|"
    r"stake\s+(eth|sol|ethereum|solana))\b",
    re.I,
)
_START_LOAN_RE = re.compile(
    r"\b(apply\s+for\s+(a\s+)?loan|i\s+want\s+(a|to\s+get\s+a)\s+loan|"
    r"take\s+me\s+to\s+lending|go\s+to\s+lending|open\s+lending|"
    r"how\s+(do|to)\s+(get\s+a\s+loan|borrow|apply)|start\s+a\s+loan|"
    r"borrow\s+(money|aed|usd|against|crypto)|"
    r"new\s+loan|loan\s+application|get\s+credit|credit\s+line)\b",
    re.I,
)
_START_WITHDRAW_RE = re.compile(
    r"\b(withdraw|i\s+want\s+to\s+withdraw|take\s+out|cash\s+out|"
    r"send\s+(my\s+)?crypto\s+(out|to\s+my\s+wallet)|move\s+(my\s+)?funds?\s+out|"
    r"how\s+(do|to)\s+withdraw)\b",
    re.I,
)
_START_DEPOSIT_RE = re.compile(
    r"\b(deposit|fund\s+my\s+account|top\s+up|add\s+crypto|"
    r"send\s+(you|crypto\s+to\s+you)|how\s+(do|to)\s+deposit|"
    r"deposit\s+address|where\s+to\s+send)\b",
    re.I,
)
_GOTO_CUSTODY_RE = re.compile(
    r"\b(custody|take\s+me\s+to\s+custody|go\s+to\s+custody|"
    r"open\s+custody|see\s+my\s+vault)\b",
    re.I,
)
_GOTO_OTC_RE = re.compile(
    r"\b(otc|otc\s+(desk|quote|trade)|take\s+me\s+to\s+otc|"
    r"go\s+to\s+otc|open\s+otc|block\s+trade|large\s+trade|"
    r"request\s+(a\s+)?quote|get\s+a\s+quote)\b",
    re.I,
)
_GOTO_PORTFOLIO_RE = re.compile(
    r"\b(portfolio|take\s+me\s+to\s+portfolio|go\s+to\s+portfolio|"
    r"open\s+portfolio|see\s+my\s+portfolio|show\s+me\s+my\s+(portfolio|assets))\b",
    re.I,
)
_GOTO_SETTINGS_RE = re.compile(
    r"\b(settings|take\s+me\s+to\s+settings|go\s+to\s+settings|"
    r"open\s+settings|change\s+(password|email|phone)|"
    r"enable\s+2fa|security\s+settings|profile\s+settings)\b",
    re.I,
)
_GOTO_WALLETS_RE = re.compile(
    r"\b(wallets?|my\s+wallet|take\s+me\s+to\s+wallets?|go\s+to\s+wallets?|"
    r"open\s+wallets?|show\s+me\s+my\s+wallet)\b",
    re.I,
)

# Generic "staking question" / "loan question" — kept from original
_LOAN_Q_RE = re.compile(
    r"\b(loan|borrow|lending|ltv|interest\s+rate|apr|credit\s+line|"
    r"murabaha|lonas?|lones)\b",
    re.I,
)
_STAKING_Q_RE = re.compile(
    r"\b(stak(e|ing)|yield|apy|rewards?|validator|stakign|stacking)\b",
    re.I,
)
_CONTACT_RE = re.compile(
    r"\b(talk\s+to\s+(a\s+)?human|speak\s+to\s+(someone|a\s+person|a\s+human)|"
    r"human\s+support|real\s+person|customer\s+service|phone\s+support|"
    r"call\s+support|i\s+need\s+help\s+from\s+a\s+human|escalate)\b",
    re.I,
)


def classify(text: str) -> Intent:
    """Return the most specific matching intent or 'unknown'.

    Order matters: action intents (I want to DO X) win over generic
    topic intents (I have a question ABOUT X). Within action intents,
    verbs of "start / go / take me to" win over pure navigation words.
    """
    if not text or not text.strip():
        return "unknown"

    # ─── Action / navigation intents — highest priority ────────────
    if _START_STAKING_RE.search(text):
        return "navigate_staking"
    if _START_LOAN_RE.search(text):
        return "navigate_lending"
    if _START_WITHDRAW_RE.search(text):
        return "withdraw_request"
    if _START_DEPOSIT_RE.search(text):
        return "deposit_request"
    if _GOTO_PORTFOLIO_RE.search(text):
        return "navigate_portfolio"
    if _GOTO_OTC_RE.search(text):
        return "navigate_otc"
    if _GOTO_CUSTODY_RE.search(text):
        return "navigate_custody"
    if _GOTO_SETTINGS_RE.search(text):
        return "navigate_settings"
    if _GOTO_WALLETS_RE.search(text):
        return "navigate_wallets"
    if _CONTACT_RE.search(text):
        return "contact_support"

    # ─── Signup flow — user wants to create an account ────────────
    if _SIGNUP_RE.search(text):
        return "signup"

    # ─── Info-question intents ────────────────────────────────────
    if _BALANCE_RE.search(text):
        return "balance_check"
    if _LOAN_Q_RE.search(text):
        return "loan_question"
    if _STAKING_Q_RE.search(text):
        return "staking_question"

    # ─── Social intents — matched last so specific intents win ────
    if _GREETING_RE.search(text):
        return "greeting"
    if _GOODBYE_RE.search(text):
        return "goodbye"
    return "unknown"


# ─── Scripted replies + action buttons per intent ────────────────────

def scripted_reply(intent: Intent) -> str | None:
    """Canned response text for intent-matched messages.

    Returns None for intents where we fall through to KB retrieval
    instead (loan_question, staking_question, unknown).
    """
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
            "Staking, OTC Desk, and Lending. Pick one from the buttons below (or "
            "tell me which you'd like to start with)."
        )
    if intent == "balance_check":
        return (
            "To see your balances, open your Portfolio or Wallets page in the "
            "portal. I've added quick-access buttons below — tap one to jump "
            "straight there."
        )
    if intent == "navigate_staking":
        return (
            "Great — staking earns you institutional-grade yield (~4% APY on ETH, "
            "~6-7% on SOL) with no lock-up. Minimum is 1 ETH or 1 SOL. Tap "
            "\"Open Staking\" below to go straight to the Staking page and start "
            "a new position."
        )
    if intent == "navigate_lending":
        return (
            "Our lending is a Shariah-compliant Murabaha credit line backed by "
            "your BTC or ETH. Starting from 3.25% APR for 5-year terms, min $25K "
            "loan. Tap \"Open Lending\" to start an application or \"Loan "
            "Calculator\" to see what you qualify for first."
        )
    if intent == "navigate_custody":
        return (
            "Custody is our segregated, Lloyd's-insured Fireblocks vault — 1:1 "
            "cold storage with no monthly fees. Tap \"Open Custody\" to see your "
            "custody balances or start a deposit."
        )
    if intent == "navigate_otc":
        return (
            "Our OTC Desk handles block trades above USD 100K with tight spreads "
            "(10-20 bps typical) and T+0 crypto / T+1 fiat settlement. Tap "
            "\"Open OTC Desk\" to request a quote."
        )
    if intent == "navigate_portfolio":
        return (
            "Your Portfolio page consolidates every asset you hold across Custody, "
            "Staking, OTC, and Lending collateral — with live USD values. Tap "
            "\"Open Portfolio\" to see it."
        )
    if intent == "navigate_settings":
        return (
            "In Settings you can update profile, change password, enable 2FA, "
            "whitelist addresses, and download tax statements. Tap \"Open "
            "Settings\" below."
        )
    if intent == "navigate_wallets":
        return (
            "Your Wallets page is where you deposit, withdraw, and see "
            "transaction history per asset. Tap \"Open Wallets\" to jump there."
        )
    if intent == "withdraw_request":
        return (
            "Withdrawals are initiated from your Wallets page. Enter the "
            "destination address and amount, confirm with 2FA — admin reviews "
            "within 1 hour during UAE business hours. Tap \"Open Wallets\" below "
            "to start a withdrawal."
        )
    if intent == "deposit_request":
        return (
            "Deposits: go to Wallets → pick your asset → copy the deposit address "
            "(or scan the QR) and send from your external wallet. Auto-credits "
            "after 3 network confirmations. Tap \"Open Wallets\" below."
        )
    if intent == "contact_support":
        return (
            "I'll route you to a human. The fastest options: email "
            "support@alfardanq9.com (response under 2 UAE business hours), or "
            "submit a ticket from Settings → Support. For urgent security issues "
            "call +971-4-xxx-xxxx. Tap \"Open Support\" below to raise a ticket."
        )
    return None


def scripted_actions(intent: Intent) -> list[dict] | None:
    """Interactive action buttons per intent.

    These deep-link into the CRM's routes. The widget renders them as
    clickable chips beneath the bot's message.
    """
    # URL map — these are the ACTUAL CRM routes. Keep this table as
    # the single source of truth; do not invent new paths here without
    # confirming the page exists at src/app/<path>/page.tsx in the CRM.
    if intent == "signup":
        return [
            {"label": "Create Account", "url": "/auth/signup", "kind": "link"},
            {"label": "Already have an account? Log in", "url": "/auth/login", "kind": "link"},
        ]
    if intent == "balance_check":
        return [
            {"label": "Open Portfolio", "url": "/dashboard/portfolio", "kind": "link"},
            {"label": "Open Wallets", "url": "/dashboard/wallets", "kind": "link"},
        ]
    if intent == "navigate_staking":
        return [
            {"label": "Open Staking", "url": "/dashboard/staking", "kind": "link"},
            {"label": "Staking Service", "url": "/dashboard/services/staking", "kind": "link"},
        ]
    if intent == "navigate_lending":
        return [
            {"label": "Open Loans", "url": "/dashboard/loans", "kind": "link"},
            {"label": "Lending Service", "url": "/dashboard/services/lending", "kind": "link"},
        ]
    if intent == "navigate_custody":
        return [
            {"label": "Open Wallets", "url": "/dashboard/wallets", "kind": "link"},
            {"label": "Custody Service", "url": "/dashboard/services/custody", "kind": "link"},
        ]
    if intent == "navigate_otc":
        return [
            {"label": "Open OTC Desk", "url": "/dashboard/services/otc", "kind": "link"},
        ]
    if intent == "navigate_portfolio":
        return [
            {"label": "Open Portfolio", "url": "/dashboard/portfolio", "kind": "link"},
        ]
    if intent == "navigate_settings":
        return [
            {"label": "Open Settings", "url": "/dashboard/settings", "kind": "link"},
        ]
    if intent == "navigate_wallets":
        return [
            {"label": "Open Wallets", "url": "/dashboard/wallets", "kind": "link"},
        ]
    if intent == "withdraw_request":
        return [
            {"label": "Open Wallets", "url": "/dashboard/wallets", "kind": "link"},
        ]
    if intent == "deposit_request":
        return [
            {"label": "Open Wallets", "url": "/dashboard/wallets", "kind": "link"},
        ]
    if intent == "contact_support":
        return [
            {"label": "Email Support", "url": "mailto:institutional@alfardanq9.com", "kind": "link"},
            {"label": "Open Settings", "url": "/dashboard/settings", "kind": "link"},
        ]
    return None


# ─── Map raw intent → MatchType literal for ChatResponse ─────────────

_INTENT_TO_MATCH_TYPE = {
    "greeting": "intent_greeting",
    "goodbye": "intent_goodbye",
    "signup": "intent_signup",
    "balance_check": "intent_balance",
    "navigate_staking": "intent_navigate_staking",
    "navigate_lending": "intent_navigate_lending",
    "navigate_custody": "intent_navigate_custody",
    "navigate_otc": "intent_navigate_otc",
    "navigate_portfolio": "intent_navigate_portfolio",
    "navigate_settings": "intent_navigate_settings",
    "navigate_wallets": "intent_navigate_wallets",
    "withdraw_request": "intent_withdraw",
    "deposit_request": "intent_deposit",
    "contact_support": "intent_contact_support",
}


def match_type_for(intent: Intent) -> str:
    """Return the MatchType literal for a scripted-reply intent, or
    'kb_hit' for intents that fall through to retrieval."""
    return _INTENT_TO_MATCH_TYPE.get(intent, "kb_hit")
