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
    # Specific staking sub-topics that MUST fire before the generic
    # staking_question intent (which asks "which network?" and otherwise
    # shadows real questions about slashing, tax reporting, outages).
    "slashing_question",
    "tax_reporting",
    "network_outage",
    "withdrawal_delay",
    "staking_fees",
    "staking_frequency",
    "unauthorized_login",
    "insurance_claim",
    "auto_reinvest",
    "transfer_from_exchange",
    "otc_quote_validity",
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

# ─── Specific staking sub-topics that must outrank staking_question ──
# Each of these needs a dedicated answer. If we fall through to
# staking_question the bot just asks "which network would you like to
# stake?" — which is the wrong reply for all of these.

_SLASHING_RE = re.compile(
    r"\b("
    r"slashing|slashed|"
    # "validator penalty", "validator misbehav…", "validator gets penalized",
    # "validator will be punished", etc — allow a few filler words so
    # natural phrasings still land on this intent rather than falling
    # through to staking_question. `\w*` on each keyword consumes the
    # full word so the trailing `\b` in the outer group matches.
    r"validator(\s+\w+){0,3}\s+(penalt\w*|penaliz\w*|misbehav\w*|punish\w*|slash\w*)|"
    r"(penalt\w*|penaliz\w*|punish\w*)(\s+\w+){0,3}\s+validator|"
    r"lose\s+(my\s+)?(staked|stake|assets)|"
    r"losing\s+(my\s+)?(stake|staked|assets)|"
    r"risk\s+(of\s+)?(losing|loss).*(stake|staked)|"
    r"slashing\s+protection"
    r")\b",
    re.I,
)
_TAX_REPORTING_RE = re.compile(
    r"\b("
    r"tax\s+(document|report|statement|form|season|purpose|year)|"
    r"tax\s+(filing|return|preparation|advisor|adviser|authority|authorities)|"
    r"(report|declare)\s+(my\s+)?(staking|rewards|yield|income|earnings)|"
    r"how.*(report|declare).*(tax|irs|crs|fatca)|"
    r"do\s+you\s+provide\s+(a\s+)?(statement|report|tax|csv|export)|"
    r"tax\s+statement|tax\s+report|tax\s+documents|tax\s+purposes|"
    r"report\s+to\s+(tax|the\s+tax|irs|hmrc|tax\s+authorit)"
    r")\b",
    re.I,
)
# "Network outage" — user is asking about blockchain or platform
# availability, NOT trying to access balances. Distinct from a
# balance_check because the framing is about the chain itself going
# down or funds being inaccessible during an incident.
_NETWORK_OUTAGE_RE = re.compile(
    r"\b("
    r"(ethereum|eth|solana|sol|bitcoin|btc|polygon|polkadot|dot|avalanche|avax|"
    r"cosmos|atom|blockchain|network|chain)\s+(goes?\s+down|went\s+down|"
    r"is\s+down|outage|offline|unavailable|halted|stopped|crashed|crash|"
    r"is\s+experiencing|has\s+issues)|"
    r"(network|blockchain|chain|validator)\s+(outage|down|offline|unavailable|halt|incident)|"
    r"what\s+(if|happens\s+when)\s+(the\s+)?(network|blockchain|chain|"
    r"ethereum|solana|bitcoin).*(down|outage|offline|stop|halt|fail)|"
    r"can(no|')?t\s+access\s+(my\s+)?(funds|account|assets|money)"
    r")\b",
    re.I,
)
# "My withdrawal is pending / stuck / taking forever" — the user
# already submitted and is anxious. We want to acknowledge the delay
# before offering the standard troubleshooting prompt.
_WITHDRAWAL_DELAY_RE = re.compile(
    r"\b("
    r"(my\s+)?withdrawal\s+(is\s+)?(pending|stuck|delayed|taking\s+(too\s+)?long|"
    r"still\s+pending|not\s+(processed|done|complete)|"
    r"been\s+(pending|waiting|sitting|stuck))|"
    r"withdrawal\s+.*\b\d+\s*(hour|hr|day|minute|min)s?\b|"
    r"(pending|waiting|stuck)\s+(for|since)\s+\d+\s*(hour|hr|day|minute|min)|"
    r"waiting.*withdraw|withdraw.*still\s+waiting|"
    r"whats?\s+(wrong|happening|up)\s+with\s+my\s+withdraw"
    r")\b",
    re.I,
)

# ─── Staking sub-topics part 2 (James QA, 19 Apr) ────────────────────

# Fees / commission on staking — "what's the fee?", "do you take a cut?"
_STAKING_FEES_RE = re.compile(
    r"\b("
    r"(fee|fees|commission|charge|cost|cut|percentage)\s+(for|on|from|off)\s+(staking|rewards?|validator)|"
    r"(staking|validator)\s+(fee|fees|commission|charge|cost)|"
    r"how\s+much\s+do\s+you\s+(charge|take|keep)\s+(for|from|on|off)\s+(staking|rewards?)|"
    r"do\s+you\s+(take|keep|charge)\s+(a\s+)?(percentage|cut|commission|fee)\s+(of|from|on|off)\s+(my\s+)?rewards?|"
    r"validator\s+commission|percent\s+of\s+rewards?"
    r")\b",
    re.I,
)

# Frequency / schedule of staking payouts
_STAKING_FREQUENCY_RE = re.compile(
    r"\b("
    r"how\s+(often|frequently)\s+(do\s+you\s+)?(pay|distribute)\s+(staking\s+)?rewards?|"
    r"when\s+(are|do)\s+(staking\s+)?rewards?\s+(paid|distributed|credited)|"
    r"(staking\s+)?rewards?\s+(paid|distributed|credited)\s+(daily|weekly|hourly)|"
    r"(staking\s+)?payout\s+(schedule|frequency|interval)|"
    r"rewards?\s+(schedule|frequency|interval)|"
    r"daily\s+or\s+weekly|hourly\s+or\s+daily"
    r")\b",
    re.I,
)

# Unauthorized / suspicious login — user is asking about security, not
# reporting an incident (incidents should route to contact_support).
_UNAUTHORIZED_LOGIN_RE = re.compile(
    r"\b("
    r"(someone|somebody|hacker)\s+(tries?|tried|attempts?|attempted|logs?|logged|logging)\s+"
    r"(in|into)\s+(my\s+)?(account|portal|dashboard)|"
    r"(unauthori[sz]ed|suspicious)\s+(login|access|sign[\s-]?in|attempt|activity)|"
    r"login\s+from\s+(another|different|unknown|unfamiliar|foreign)\s+(country|location|ip|device|place)|"
    r"sign[\s-]?in\s+from\s+(another|different|unknown|unfamiliar|foreign)\s+(country|location|ip|device)|"
    r"what\s+(if|happens).*(someone|hacker|attacker).*(log|sign)|"
    r"protect.*against.*(unauthori[sz]ed|account\s+takeover)|"
    r"detect.*suspicious\s+(login|sign[\s-]?in|activity)"
    r")\b",
    re.I,
)

# How to file an insurance claim — user is asking PROCESS, not policy
# details. If matched, we route to a dedicated scripted reply that
# includes the claim email + phone + needed info.
_INSURANCE_CLAIM_RE = re.compile(
    r"\b("
    r"(file|filing|submit|raise|make)\s+(an?\s+)?(insurance\s+)?claim|"
    r"how\s+(do|to)\s+(i\s+)?(file|submit|raise|make)\s+(an?\s+)?(insurance\s+)?claim|"
    r"claim\s+(process|procedure|steps|instructions)|"
    r"insurance\s+claim\s+process"
    r")\b",
    re.I,
)

# ─── Staking sub-topics part 3 (James QA, Set 3) ─────────────────────

# Auto-reinvest / compounding — user wants to know if rewards compound
# automatically. Routes to a dedicated scripted reply so we don't get
# mis-routed to "which network would you like to stake?" again.
_AUTO_REINVEST_RE = re.compile(
    r"\b("
    r"auto[\s-]?(re[\s-]?invest|compound|compounding|stake|staking)|"
    r"automatic(ally)?\s+(re[\s-]?invest|compound|compounding|restake|restaking)|"
    r"automatic\s+(re[\s-]?invest|compound|compounding)|"
    r"(re[\s-]?invest|restake)\s+(my\s+)?(staking\s+)?rewards?\s+(automatic|by\s+default)|"
    r"compound\s+(my\s+)?(staking\s+)?rewards?|"
    r"compound(ing)?\s+interest|"
    r"re[\s-]?invest\s+(my\s+)?rewards?"
    r")\b",
    re.I,
)

# Transferring from another exchange (Binance, Coinbase, Kraken) into
# custody. Distinct from the generic deposit_request because the user
# explicitly has funds on another venue — the reply needs "copy our
# deposit address + initiate a withdrawal on their side".
_TRANSFER_FROM_EXCHANGE_RE = re.compile(
    r"\b("
    # "transfer from Binance", "move from Coinbase", "send from Kraken"
    r"(transfer|move|send)(\s+(my\s+)?(crypto|btc|eth|sol|coins?|funds|assets?|balance))?\s+"
    r"from\s+(another\s+)?(exchange|wallet|platform|"
    r"binance|coinbase|kraken|bybit|okx|bitget|kucoin|bitfinex|ftx|gemini|crypto\.?com|"
    r"bitstamp|huobi|mexc|bingx|upbit|coincheck|bitflyer)|"
    # "deposit from Binance"
    r"deposit\s+from\s+(another\s+)?(exchange|wallet|platform|"
    r"binance|coinbase|kraken|bybit|okx|bitget|kucoin|gemini|crypto\.?com)|"
    # "how do I transfer from …"
    r"how\s+(do|to)\s+i?\s*(transfer|move|send|deposit)\s+from\s+(another\s+)?(exchange|wallet|platform|binance|coinbase|kraken)"
    r")\b",
    re.I,
)

# OTC quote validity / firm-vs-indicative + price lock. Routes away
# from generic navigate_otc (which returns the generic OTC intro).
_OTC_QUOTE_VALIDITY_RE = re.compile(
    r"\b("
    r"how\s+long\s+is\s+(an?\s+|the\s+)?(otc\s+)?quote\s+valid|"
    r"(otc\s+)?quote\s+(valid(ity)?|expir|window|lock|lifetime)|"
    r"(lock|freeze|guarantee|secure)\s+(in\s+)?(the\s+|my\s+)?price|"
    r"(price|quote)\s+lock|"
    r"firm\s+(quote|price)|indicative\s+(quote|price)|"
    r"can\s+i\s+lock\s+(in\s+)?(the\s+|my\s+)?price"
    r")\b",
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
    # transfer_from_exchange must outrank the plain deposit_request
    # because "deposit from Binance" is a FROM-exchange question that
    # needs the detailed reply (copy address + withdraw on their side
    # + send a test amount), not the generic deposit blurb.
    if _TRANSFER_FROM_EXCHANGE_RE.search(text):
        return "transfer_from_exchange"
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
    # Withdrawal-delay check comes BEFORE withdraw_request navigation
    # because phrases like "my withdrawal is pending" shouldn't be
    # re-routed to "Open Wallets" — the user wants status, not a link.
    # Moved here to win over navigate_* if somehow both matched, but
    # in practice the regex is specific enough to not overlap.
    if _WITHDRAWAL_DELAY_RE.search(text):
        return "withdrawal_delay"
    # These MUST fire BEFORE staking_question / loan_question —
    # otherwise "what is slashing?" / "what are your staking fees?" /
    # "how often are rewards paid?" all get answered with "which
    # network would you like to stake?" (mis-routes reported by
    # James in two QA passes).
    if _NETWORK_OUTAGE_RE.search(text):
        return "network_outage"
    if _TAX_REPORTING_RE.search(text):
        return "tax_reporting"
    if _SLASHING_RE.search(text):
        return "slashing_question"
    if _STAKING_FEES_RE.search(text):
        return "staking_fees"
    if _STAKING_FREQUENCY_RE.search(text):
        return "staking_frequency"
    if _UNAUTHORIZED_LOGIN_RE.search(text):
        return "unauthorized_login"
    if _INSURANCE_CLAIM_RE.search(text):
        return "insurance_claim"
    # Set 3 (auto-reinvest / quote-validity) — must fire BEFORE the
    # generic topic routes so we don't answer "can I auto-reinvest?"
    # with "which network would you like to stake?" (James's Set 3 QA).
    # transfer_from_exchange is already handled earlier in the action
    # block so it outranks the plain deposit_request.
    if _AUTO_REINVEST_RE.search(text):
        return "auto_reinvest"
    if _OTC_QUOTE_VALIDITY_RE.search(text):
        return "otc_quote_validity"
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

    For intents that have multiple phrasings (greeting, goodbye,
    affirmation, negation, signup, contact_support, balance_check) we
    pick a random variant from refine.vary — the bot doesn't say the
    exact same thing every time.

    Returns None for intents where we fall through to KB retrieval
    instead (loan_question, staking_question, unknown).
    """
    # Try the variant pool first — gives the bot human-sounding variety
    from ..refine.vary import pick_variant  # local import to avoid cycles
    varied = pick_variant(intent)
    if varied is not None:
        return varied

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
            "your BTC or ETH, with up to 75% LTV — among the highest in "
            "institutional crypto lending. Rates start at 3.25% APR for 5-year "
            "terms (min $25K). If the collateral price drops, you receive a "
            "margin call at 85% LTV with 24 hours to top up or partially repay "
            "before any liquidation. Tap \"Open Lending\" to start, or \"Loan "
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
            "Our OTC Desk handles block trades from USD 100,000 to USD 50M+. "
            "Typical spreads are 5 to 25 bps depending on asset and size. "
            "Crypto settlement completes within 1-4 hours; fiat wires within "
            "24 hours — same-day crypto is standard. Tap \"Open OTC Desk\" to "
            "request a firm quote."
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
            "within 1 hour during UAE business hours. Typical processing: "
            "1-4 hours for crypto, 24 hours for fiat wires. Tap \"Open Wallets\" "
            "below to start a withdrawal."
        )
    if intent == "withdrawal_delay":
        return (
            "I understand your concern — let me help. Our typical processing "
            "is 1 to 4 hours for crypto. Anything significantly longer may be "
            "a compliance review or a temporary technical delay. Would you "
            "please share the asset and withdrawal ID so I can check the "
            "status for you? You can also reach our team directly at "
            "institutional@alfardanq9.com — they can escalate if needed."
        )
    if intent == "slashing_question":
        return (
            "Slashing is a penalty imposed by proof-of-stake networks when "
            "validators misbehave — typically through double-signing or "
            "prolonged downtime. However, you do not lose your assets with us. "
            "We maintain a dedicated slashing-protection reserve that fully "
            "indemnifies clients against validator penalties; we have had zero "
            "slashing events since inception. Would you like me to explain "
            "how our slashing protection works in more detail?"
        )
    if intent == "tax_reporting":
        return (
            "We provide full transaction history in CSV format — you can "
            "export your staking rewards, OTC trades, and loan cost records "
            "directly from your dashboard (Settings → Statements). We do not "
            "provide tax advice; tax treatment varies by jurisdiction and "
            "personal circumstances, so we recommend consulting a qualified "
            "tax adviser. Would you like me to walk you through exporting "
            "your rewards report?"
        )
    if intent == "network_outage":
        return (
            "If the underlying blockchain experiences an outage, certain "
            "operations — unstaking, on-chain transfers, withdrawals on that "
            "network — are temporarily paused at the protocol level. However, "
            "your assets remain safe in segregated Fireblocks MPC custody. "
            "Full functionality resumes automatically once the network "
            "recovers. Major outages are rare and historically resolved within "
            "hours. Would you like me to explain our contingency procedures "
            "in more detail?"
        )
    if intent == "staking_fees":
        return (
            "We take 15% of staking rewards only — your principal is never "
            "touched. No lock-up fees, no withdrawal fees, no platform fees. "
            "The published APY on every network is already NET of our "
            "commission, so the rate you see is the rate you earn. Would you "
            "like me to calculate your net yield for a specific network?"
        )
    if intent == "staking_frequency":
        return (
            "Staking rewards are paid daily. Every day at 00:00 UTC rewards "
            "are automatically calculated and distributed to your segregated "
            "custodial account — no manual claim required. For Ethereum "
            "rewards auto-compound into your staked principal; Solana credits "
            "as a separate balance. Would you like me to help you track your "
            "rewards on a specific network?"
        )
    if intent == "unauthorized_login":
        return (
            "If someone attempts to sign in from an unrecognised device or "
            "location, our system blocks the attempt and emails you a "
            "security alert immediately. Every sensitive action "
            "(withdrawals, settings changes, loan approvals) additionally "
            "requires 2FA. We also offer withdrawal-address whitelisting "
            "with a 24-hour cooldown to stop address-swap attacks. Would you "
            "like me to walk you through enabling 2FA and reviewing your "
            "security settings?"
        )
    if intent == "insurance_claim":
        return (
            "To file an insurance claim: (1) email claims@alfardanq9.com with "
            "your account details and the transaction hash or incident "
            "summary, or (2) call our 24/7 emergency line at +971 4 123 4568. "
            "Our claims team acknowledges within 2 UAE business hours and "
            "opens a case with the Lloyd's of London syndicate directly. "
            "Would you like me to connect you with a human on the claims "
            "team right now?"
        )
    if intent == "auto_reinvest":
        return (
            "Yes — automatic compounding is on by default. Your Ethereum "
            "staking rewards compound directly into your staked principal "
            "(no manual claim), and Solana rewards can auto-restake at each "
            "epoch. This lifts your effective APY above the headline rate. "
            "You can toggle auto-compounding off per position from your "
            "dashboard if you prefer to take rewards as liquid balance. "
            "Would you like me to walk you through configuring "
            "auto-compounding for a specific network?"
        )
    if intent == "transfer_from_exchange":
        return (
            "To move crypto from another exchange into Al-Fardan Q9: (1) go "
            "to Wallets → pick the asset → Deposit to see your unique "
            "Fireblocks deposit address (and a QR code). (2) On the other "
            "exchange, initiate a withdrawal to that exact address on the "
            "matching network (e.g. BTC mainnet, ETH ERC-20). (3) Send a "
            "small test amount first — crypto transactions are irreversible. "
            "Auto-credit happens after 3 on-chain confirmations. Would you "
            "like me to walk you through this step by step?"
        )
    if intent == "otc_quote_validity":
        return (
            "A firm OTC quote is valid for 15 minutes — within that window "
            "the price is locked, regardless of market movement. An "
            "indicative quote is an estimate that isn't locked, meant for "
            "sizing only. Once you're ready, you can ask the dealer to "
            "convert it to a firm quote at the current mid plus our spread, "
            "then you have 15 minutes to accept. Would you like me to "
            "connect you with Layla on the OTC desk for a firm quote?"
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
    # Specific staking sub-topics added after James's QA passes.
    "slashing_question": "intent_slashing",
    "tax_reporting": "intent_tax_reporting",
    "network_outage": "intent_network_outage",
    "withdrawal_delay": "intent_withdrawal_delay",
    "staking_fees": "intent_staking_fees",
    "staking_frequency": "intent_staking_frequency",
    "unauthorized_login": "intent_unauthorized_login",
    "insurance_claim": "intent_insurance_claim",
    "auto_reinvest": "intent_auto_reinvest",
    "transfer_from_exchange": "intent_transfer_from_exchange",
    "otc_quote_validity": "intent_otc_quote_validity",
    # Short yes/no — map to generic intent types so the widget can
    # still render actions the usual way.
    "affirmation": "intent_affirmation",
    "negation": "intent_negation",
}


def match_type_for(intent: Intent) -> str:
    """Return the MatchType literal for a scripted-reply intent, or
    'kb_hit' for intents that fall through to retrieval."""
    return _INTENT_TO_MATCH_TYPE.get(intent, "kb_hit")
