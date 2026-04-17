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
    "affirmation",   # "yes / yeah / sure / ok / sounds good"
    "negation",      # "no / nope / not really"
    "unknown",
]


# A short list of substantive tokens we look for to decide if a message
# is more than just a greeting. If the user typed "hi" they probably
# want the greeting flow; if they typed "hi, i am new to crypto" the
# "new to crypto" part is what matters — the greeting intent should
# NOT shadow the substantive ask.
_SUBSTANTIVE_RE = re.compile(
    r"\b("
    r"crypto|btc|bitcoin|eth|ethereum|sol|solana|coin|token|stak\w*|"
    r"invest\w*|lend\w*|borrow\w*|loan\w*|yield|apy|rewards?|"
    r"wallet|custody|portfolio|account|balance|otc|desk|trade|trading|"
    r"signup|sign\s*up|register|start|begin|kyc|verify|"
    r"deposit|withdraw|transfer|buy|sell|price|fee|"
    r"sharia|halal|murabaha|safe|scam|trust|hack|insurance|"
    r"sheikh|khalid|fatima|layla|hamdan|omar|nadia|"
    r"new\s+to|don't\s+know|need\s+help|how\s+(do|can)\s+i|"
    r"starter|family|office|institution"
    r")\b",
    re.I,
)


# Short affirmations / negations. We only match when the WHOLE message
# is one of these — otherwise we let retrieval handle richer replies.
_AFFIRMATION_ONLY_RE = re.compile(
    r"^\s*("
    r"yes+|yeah+|yep+|yup+|ya|sure|ok+|okay+|"
    r"absolutely|definitely|of\s+course|sounds?\s+good|"
    r"please|please\s+do|go\s+ahead|great|perfect|"
    r"great\.?|perfect\.?|nice|cool|awesome|wonderful|"
    r"lets\s+go|let's\s+go|lets\s+do\s+it|let's\s+do\s+it|"
    r"👍+|✅+"
    r")\s*[.!?]*\s*$",
    re.I,
)

_NEGATION_ONLY_RE = re.compile(
    r"^\s*("
    r"no+|nope|nah|not\s+really|not\s+now|"
    r"no\s+thanks?|no\s+thank\s+you|maybe\s+later|"
    r"not\s+yet|skip|pass|"
    r"👎+|❌+"
    r")\s*[.!?]*\s*$",
    re.I,
)


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
# Balance check fires ONLY when the user is asking to look at/check a
# balance — not when they just happen to mention "my assets" in a
# different context (e.g. "are my assets protected?" which is about
# safety, not balance). Require an action verb or a balance-specific noun.
_BALANCE_RE = re.compile(
    r"\b("
    r"(show|check|see|view|look\s+at|display|open|where\s+is)\s+"
    r"my\s+(balance|portfolio|holdings?|wallet|assets?)|"
    r"how\s+much\s+(do\s+)?i\s+have|"
    r"what('s|\s+is)\s+my\s+(balance|net\s+worth|total)|"
    r"my\s+balance|current\s+balance"
    r")\b",
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
# navigate_otc fires only when the user wants to OPEN the OTC page, not
# when they're asking a specific quantitative question about OTC (those
# should flow to KB so we answer the specific question, not just drop
# a navigation button).
_GOTO_OTC_RE = re.compile(
    r"\b("
    r"take\s+me\s+to\s+otc|go\s+to\s+otc|open\s+otc|"
    r"new\s+quote|request\s+(a\s+)?quote|get\s+a\s+quote|place\s+an?\s+otc|"
    r"otc\s+page|otc\s+desk\s+page"
    r")\b",
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

    Greeting / goodbye intents are DELIBERATELY downranked — they only
    fire when the message is essentially JUST a greeting (no crypto
    content, short length). This prevents "Hi, I'm new to crypto" from
    being misread as a bare greeting — the substantive content wins.
    """
    if not text or not text.strip():
        return "unknown"

    # ─── Pure short affirmations / negations — handled first ──────
    # Matched ONLY when the ENTIRE message is one of these; a longer
    # "yes I do want to stake" would fall through to the retriever.
    if _AFFIRMATION_ONLY_RE.match(text):
        return "affirmation"
    if _NEGATION_ONLY_RE.match(text):
        return "negation"

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
    # Skip signup intent if the message also contains a specific money
    # amount or minimum/qualify keyword — those are quantitative
    # questions best answered by retrieval (e.g. "I have $5,000, can
    # I open an account?" should return the $5,000-specific answer,
    # not the generic signup reply).
    has_amount = bool(re.search(r"(\$\s?\d|\d+\s*(dollar|usd|aed|eur|eth|btc|sol|k\b|thousand|million))", text, re.I))
    has_minimum_question = bool(
        re.search(r"\b(minimum|min\b|enough|qualify|at\s+least)\b", text, re.I)
    )
    if _SIGNUP_RE.search(text) and not (has_amount or has_minimum_question):
        return "signup"

    # ─── Info-question intents ────────────────────────────────────
    if _BALANCE_RE.search(text):
        return "balance_check"
    if _LOAN_Q_RE.search(text):
        return "loan_question"
    if _STAKING_Q_RE.search(text):
        return "staking_question"

    # ─── Social intents — only if message is mostly just a greeting
    # This fixes the "Hi, I am new to crypto" bug. We refuse to treat
    # a message as a bare greeting if it also carries substantive
    # crypto / signup / product content — let retrieval take it.
    has_substantive_content = bool(_SUBSTANTIVE_RE.search(text))
    is_short = len(text.split()) <= 4

    if _GREETING_RE.search(text) and is_short and not has_substantive_content:
        return "greeting"
    if _GOODBYE_RE.search(text) and is_short and not has_substantive_content:
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
            "Assalamu alaikum — I'm Safiya Al Suwaidi, Client Acquisition & "
            "Growth Manager at Al-Fardan Q9. I help new investors get started "
            "with our Custody, Staking, OTC, and Lending services. "
            "What brings you here today?"
        )
    if intent == "goodbye":
        return (
            "Take care — I'm here whenever you need. Urgent? "
            "institutional@alfardanq9.com covers you 24/7."
        )
    if intent == "signup":
        return (
            "Happy to help you open an account — takes about 5 minutes. "
            "Which service are you most interested in first?"
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
    if intent == "affirmation":
        # We don't track conversation state yet, so we can't know what
        # the user is affirming. Give a warm continuation prompt + quick
        # paths into the common flows.
        return (
            "Great — happy to help. What would you like to do next? "
            "Open an account, see our products, or ask something specific?"
        )
    if intent == "negation":
        return (
            "No worries — take your time. What else can I help with? "
            "I can explain products, fees, or how we're regulated. "
            "Whatever's useful."
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
    if intent == "affirmation":
        # Offer the most common next-steps any new visitor might want.
        return [
            {"label": "Create Account", "url": "/auth/signup", "kind": "link"},
            {"label": "Open Dashboard", "url": "/dashboard", "kind": "link"},
            {"label": "Email Support", "url": "mailto:institutional@alfardanq9.com", "kind": "link"},
        ]
    if intent == "negation":
        return None  # Don't push links when they just said no
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
    # Short yes/no — map to generic intent types so the widget can
    # still render actions the usual way.
    "affirmation": "intent_affirmation",
    "negation": "intent_negation",
}


def match_type_for(intent: Intent) -> str:
    """Return the MatchType literal for a scripted-reply intent, or
    'kb_hit' for intents that fall through to retrieval."""
    return _INTENT_TO_MATCH_TYPE.get(intent, "kb_hit")
