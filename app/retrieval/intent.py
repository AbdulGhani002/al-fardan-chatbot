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
    # ─── Set 4 (wrong-intent / missing-info fixes, James 19 Apr) ─
    "interest_calculation",
    "whitelist_address",
    "account_closure",
    "early_repayment",
    "validator_choice",
    "security_incident",
    # ─── Set 5 ──────────────────────────────────────────────────
    "loan_extension",
    "internal_transfer",
    "interest_rate_change",
    "staking_network_fees",
    "custody_minimum_balance",
    "otc_counterparty",
    "api_key_management",
    "staking_rewards_withdraw",
    # ─── Set 6 ──────────────────────────────────────────────────
    "ltv_calculation",
    "rewards_history",
    "session_timeout",
    # ─── Sets 7 + 8 + 9 (13 more) ───────────────────────────────
    "claim_rewards",
    "interest_payment",
    "shared_account",
    "cancel_otc",
    "otc_max_size",
    "tin_required",
    "add_collateral",
    "ip_whitelisting",
    "negotiate_rate",
    "loan_default",
    "validator_diversification",
    "monthly_statements",
    "liquidation_details",
    # Sets 11 + 12 + 13 + 14
    "partial_repayment",
    "hardware_wallet",
    "rewards_in_different_asset",
    "missing_rewards",
    "user_to_user_transfer",
    "otc_after_hours",
    "login_history",
    "interest_source",
    "rewards_expiration",
    "withdrawal_time",
    "active_sessions",
    "loan_transfer",
    "validator_locations",
    "historical_snapshot",
    # Set 15
    "otc_settlement_currency",
    "temporary_deactivation",
    "collateral_revaluation",
    "tax_withholding",
    "account_recovery",
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
    # Scoped "send ... to you" only when the user is clearly talking
    # about crypto/funds — not "send you my passport" (James QA).
    r"send\s+(crypto|btc|eth|sol|usdt|usdc|funds|assets?|coins?|tokens?)\s+to\s+you|"
    r"how\s+(do|to)\s+deposit|"
    r"deposit\s+address|"
    r"where\s+to\s+send\s+(my\s+|the\s+)?(crypto|btc|eth|sol|funds|usdt|usdc|coins?|tokens?|assets?))\b",
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
    r"\b("
    # Explicit requests for a human
    r"talk\s+to\s+(a\s+)?(human|someone|a\s+person|an?\s+agent|support|the\s+team)|"
    r"speak\s+to\s+(someone|a\s+person|a\s+human|an?\s+agent|support|the\s+team)|"
    r"chat\s+with\s+(someone|a\s+person|a\s+human|an?\s+agent|support|the\s+team)|"
    r"human\s+support|real\s+person|customer\s+(service|support)|phone\s+support|"
    r"call\s+support|i\s+need\s+help\s+from\s+a\s+human|escalate|"
    # Generic "I'm stuck / need help" — user wants assistance, not
    # necessarily an agent but the reply covers it.
    r"(i\s+)?(get|got|am|'m|feel)\s+stuck|"
    r"need\s+help(\s+with)?|"
    r"contact\s+(support|the\s+team|someone)|"
    r"(is\s+there|anyone|anybody)\s+(a\s+|an\s+)?(person|human|someone|agent).*\b(speak|talk|chat|help)|"
    r"(speak|talk|chat|help)\s+(to|with)\s+(someone|anyone|a\s+human|a\s+person)"
    r")\b",
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
    r"report\s+to\s+(tax|the\s+tax|irs|hmrc|tax\s+authorit)|"
    # James Set 4 gap — "are rewards taxed?" was falling through because
    # we only matched "tax <noun>" and not "tax(ed|able)" as a single word.
    r"(are|is)\s+(my\s+|the\s+)?(staking\s+|my\s+)?rewards?\s+(taxed|taxable)|"
    r"(rewards?|income|yield|earnings?)\s+(taxed|taxable)|"
    r"tax(able|ed|ation)\s+(on|of|for)\s+(staking|rewards?|income|yield|earnings)"
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

# Frequency / schedule of staking payouts. Also absorbs James's
# Set 8 "what time are rewards paid?" phrasing — the reply already
# says "00:00 UTC daily" which answers both.
_STAKING_FREQUENCY_RE = re.compile(
    r"\b("
    r"how\s+(often|frequently)\s+(do\s+you\s+)?(pay|distribute)\s+(staking\s+)?rewards?|"
    r"how\s+(often|frequently)\s+(are\s+)?(staking\s+)?rewards?\s+(paid|distributed|credited)|"
    r"how\s+(often|frequently)\s+(are\s+)?(staking\s+)?(paid|distributed|credited)|"
    r"when\s+(are|do)\s+(staking\s+)?rewards?\s+(paid|distributed|credited)|"
    r"what\s+time\s+(are\s+|do\s+)?(staking\s+)?rewards?\s+(paid|distributed|credited)|"
    r"(staking\s+)?rewards?\s+(paid|distributed|credited)\s+(daily|weekly|hourly|at\s+\d|same\s+time)|"
    r"(staking\s+)?payout\s+(schedule|frequency|interval|time)|"
    r"rewards?\s+(schedule|frequency|interval)|"
    r"daily\s+or\s+weekly|hourly\s+or\s+daily|"
    r"same\s+time\s+every\s+day"
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
    # Added "start|open|initiate|begin|lodge" to verb list so
    # "how do I start an insurance claim" (James Set 15 Q2) also
    # routes correctly. Also added "stolen|theft|loss|incident|hack"
    # as qualifiers so phrasings that skip the word "insurance"
    # still land here.
    r"(file|filing|submit|raise|make|start|open|initiate|begin|lodge)\s+(an?\s+)?(insurance\s+|stolen[\s-]asset\s+|theft\s+|loss\s+)?claim|"
    r"how\s+(do|to)\s+(i\s+)?(file|submit|raise|make|start|open|initiate|begin|lodge)\s+(an?\s+)?(insurance\s+|stolen[\s-]asset\s+|theft\s+)?claim|"
    r"claim\s+(process|procedure|steps|instructions)|"
    r"insurance\s+claim\s+process|"
    r"claim\s+for\s+(stolen|lost|theft|hacked|missing)\s+(assets?|funds|crypto|btc|eth|sol|money)"
    r")\b",
    re.I,
)

# ─── Set 15 intents ───────────────────────────────────────────────────

# "Can I settle OTC in a different currency than quoted?" — YES.
# Distinct from navigate_otc (generic intro) and otc_quote_validity.
_OTC_SETTLEMENT_CURRENCY_RE = re.compile(
    r"\b("
    r"settle\s+(an?\s+|my\s+|the\s+)?otc\s+(trade|quote|order|deal)?\s*in\s+(a\s+)?(different|another|alternative|other)\s+currency|"
    r"(different|another|alternative|other)\s+(settlement\s+)?currency\s+(than|from|to)\s+(quoted|the\s+quote)|"
    r"change\s+(the\s+|my\s+)?(otc\s+)?settlement\s+currency|"
    r"otc\s+(settlement\s+)?currenc(y|ies)\s+(options?|choices?|support)"
    r")\b",
    re.I,
)

# "Can I temporarily deactivate my account?" — NO.
_TEMPORARY_DEACTIVATION_RE = re.compile(
    r"\b("
    r"temporar(il)?y\s+(deactivate|pause|suspend|freeze|disable|lock|hold)\s+(my\s+|the\s+)?account|"
    r"(pause|suspend|freeze|put\s+on\s+hold)\s+(my\s+|the\s+)?account|"
    r"(deactivate|disable)\s+(my\s+|the\s+)?account\s+(temporar|for\s+a\s+while|for\s+now)|"
    r"account\s+(hibernation|dormancy|hold|pause|suspension|freeze)|"
    r"go\s+dormant"
    r")\b",
    re.I,
)

# "How often is my collateral revalued?" — continuously.
_COLLATERAL_REVALUATION_RE = re.compile(
    r"\b("
    # Each verb has \w* so the `\b` at the outer group can still match
    # after "revalued" / "repriced" / "updated" etc.
    r"(how\s+often|when)\s+(is|are)\s+(my\s+|the\s+)?collateral\s+(revalu\w*|re[\s-]?pric\w*|updat\w*|mark\w*|valu\w*|assess\w*)|"
    r"collateral\s+(revaluation|re[\s-]?pric\w*|update\s+frequency|valuation\s+frequency|mark[\s-]to[\s-]market)|"
    r"(ltv|collateral\s+value)\s+(update|refresh|recalc)\s+(frequency|interval|how\s+often)|"
    r"(price|value)\s+(update|refresh)\s+(of|for)\s+(my\s+)?collateral"
    r")\b",
    re.I,
)

# "Do you withhold taxes from my staking rewards?" — NO.
_TAX_WITHHOLDING_RE = re.compile(
    r"\b("
    r"(do|does)\s+(you|al[\s.-]?fardan)\s+withhold\s+(any\s+|my\s+)?(taxes|tax)|"
    r"(tax|taxes)\s+(withheld|withhold|deducted|deduction)\s+(from|on)\s+(my\s+|the\s+)?(staking\s+)?rewards?|"
    r"(tax|taxes)\s+withholding\s+on\s+(my\s+|staking\s+)?rewards?|"
    r"tax\s+deduction\s+(from|on)\s+(my\s+)?(staking\s+)?rewards?|"
    r"automatic\s+tax\s+(withhold|deduction)"
    r")\b",
    re.I,
)

# "What is the process to recover my account if I lose access?"
# Distinct from unauthorized_login (security alerts) + the forgot-
# password self-service flow — this is the full identity-recovery
# process with video KYC.
_ACCOUNT_RECOVERY_RE = re.compile(
    r"\b("
    r"recover\s+(my\s+|the\s+)?account|"
    r"account\s+recovery|"
    r"(lost|lose|can'?t|cannot|no)\s+access\s+to\s+(my\s+)?account|"
    r"regain\s+access\s+to\s+(my\s+)?account|"
    r"(process|how\s+to|how\s+do\s+i)\s+(recover|get\s+back\s+into)\s+(my\s+)?account|"
    r"lost\s+(my\s+)?(password|email|2fa|device|phone|everything).*account"
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

# OTC quote validity / firm-vs-indicative + price lock + price-move-
# during-settlement. Routes away from generic navigate_otc (which
# returns the generic OTC intro) and also absorbs the "do you guarantee
# the price?" variant from James Set 5.
_OTC_QUOTE_VALIDITY_RE = re.compile(
    r"\b("
    r"how\s+long\s+is\s+(an?\s+|the\s+)?(otc\s+)?quote\s+valid|"
    r"(otc\s+)?quote\s+(valid(ity)?|expir|window|lock|lifetime)|"
    r"(lock|freeze|guarantee|secure)\s+(in\s+)?(the\s+|my\s+)?price|"
    r"(price|quote)\s+lock|"
    r"firm\s+(quote|price)|indicative\s+(quote|price)|"
    r"can\s+i\s+lock\s+(in\s+)?(the\s+|my\s+)?price|"
    r"(do\s+you\s+|does\s+al.?fardan\s+)?guarantee\s+(the\s+|my\s+)?(otc\s+)?price|"
    r"market\s+moves?\s+(during|before)\s+settlement|"
    r"what\s+if\s+(the\s+)?(market|price)\s+moves?"
    r")\b",
    re.I,
)

# ═════════════════════════════════════════════════════════════════════
#  James QA Sets 4 + 5 (13 more intents)
# ═════════════════════════════════════════════════════════════════════

# Interest calc — daily accrual / monthly charge vs fixed / variable.
_INTEREST_CALCULATION_RE = re.compile(
    r"\b("
    r"how\s+is\s+(the\s+|my\s+)?(interest|profit|cost|murabaha)\s+(calculated|computed|accrued|charged|worked\s+out)|"
    r"(daily|monthly|hourly|annually)\s+(interest|accrual|charge)|"
    r"when\s+(is|does)\s+(the\s+|my\s+)?(interest|profit)\s+(charged|paid|accrue|accrued)|"
    r"interest\s+(calculation|accrual|schedule)|"
    r"(daily|monthly)\s+vs\s+(monthly|daily)"
    r")\b",
    re.I,
)

# Whitelist withdrawal addresses — must outrank generic withdraw_request
# because the word "withdrawal" is in the question.
_WHITELIST_ADDRESS_RE = re.compile(
    r"\b("
    r"(whitelist|whitelisting|white[\s-]listed?)\s+(a\s+|my\s+|an?\s+|new\s+)?(withdrawal\s+)?addresses?|"
    r"(add|register)\s+(a\s+|new\s+|an?\s+)?(whitelist|whitelisted)\s+addresses?|"
    r"(withdrawal\s+)?addresses?\s+whitelist|"
    r"whitelist\s+(how\s+long|duration|cooldown)|"
    r"approved\s+addresses?\s+list"
    r")\b",
    re.I,
)

# Close account + GDPR-style data deletion.
_ACCOUNT_CLOSURE_RE = re.compile(
    r"\b("
    r"(close|delete|deactivate|cancel|terminate)\s+(my\s+|the\s+)?account|"
    r"how\s+(do|to)\s+(i\s+)?(close|delete|deactivate|terminate)\s+(my\s+)?account|"
    r"delete\s+(my\s+|all\s+)?(data|personal\s+data|information)|"
    r"data\s+(deletion|retention|privacy|policy|erasure)|"
    r"right\s+to\s+(be\s+forgotten|erasure|deletion)|"
    r"gdpr\s+(request|rights?)"
    r")\b",
    re.I,
)

# Early loan repayment / prepayment penalty.
_EARLY_REPAYMENT_RE = re.compile(
    r"\b("
    r"(repay|pay\s+(off|back))\s+(my\s+|the\s+)?loan\s+(early|earlier|ahead)|"
    r"early\s+(repayment|loan\s+payment|loan\s+repayment|payoff)|"
    r"prepay(ment)?\s+(penalty|fee|charge|cost)|"
    r"(repay|pay\s+off)\s+early|"
    r"pay\s+back\s+(my\s+)?loan\s+(early|before|ahead)"
    r")\b",
    re.I,
)

# Validator choice / selection.
_VALIDATOR_CHOICE_RE = re.compile(
    r"\b("
    # Adds "switch|change|swap|move" to the verb list — James Set 12 Q3
    # "can I switch to a different validator if I'm not happy with
    # performance" was falling through otherwise. Allow up to 3 adjective
    # words between the verb and "validator" (e.g. "switch to a different").
    r"(choose|select|pick|specify|nominate|switch|change|swap|move)\s+(to\s+)?((a|my|which|the|another|different|other)\s+){0,3}validator|"
    r"which\s+validator.*(stake|eth|sol|my)|"
    r"can\s+i\s+(pick|select|choose|specify|switch|change)\s+(to\s+)?((a|the|my|another|different|other)\s+){0,3}validator|"
    r"my\s+own\s+validator|"
    r"validator\s+(selection|switch|change)|"
    r"byo\s+validator"
    r")\b",
    re.I,
)

# Security incident — user reports suspicious activity, panic wording.
# Fires BEFORE contact_support / unauthorized_login because the user is
# ALREADY in trouble and needs emergency steps, not a generic "talk to
# a human" link.
_SECURITY_INCIDENT_RE = re.compile(
    r"\b("
    r"suspicious\s+(activity|transaction|withdrawal|behavior|behaviour|action|stuff)|"
    r"unauthori[sz]ed\s+(transaction|withdrawal|transfer|activity|charge)|"
    # Allow natural "account is/was/has been hacked" phrasing, not just
    # the strict "account hacked" adjacent form.
    r"(my\s+)?account\s+(is\s+|was\s+|has\s+been\s+|got\s+|just\s+got\s+)?"
    r"(compromised|hacked|breached|taken\s+over|stolen|drained|emptied)|"
    r"i\s+(was|got|been|think\s+i('ve|\s+have)\s+been|feel\s+i('ve|\s+have)\s+been)\s+hacked|"
    r"someone\s+(stole|took|drained|moved).*my\s+(funds|crypto|balance|assets?|btc|eth)|"
    r"i\s+see\s+suspicious|"
    r"(report|flag)\s+(a\s+)?(suspicious|unauthori[sz]ed|fraudulent)"
    r")\b",
    re.I,
)

# Loan extension / renewal.
_LOAN_EXTENSION_RE = re.compile(
    r"\b("
    r"extend\s+(my\s+|the\s+)?loan(\s+(term|period|duration|length))?|"
    r"(renew|roll[\s-]?over)\s+(my\s+|the\s+)?loan|"
    r"loan\s+(extension|renewal|rollover|roll[\s-]?over)|"
    r"can\s+i\s+extend\s+(my\s+)?loan"
    r")\b",
    re.I,
)

# Internal wallet-to-wallet transfer within Al-Fardan.
_INTERNAL_TRANSFER_RE = re.compile(
    r"\b("
    # "transfer crypto between my (own) wallets" — tolerate "own"
    r"(transfer|move|send)\s+(crypto|btc|eth|sol|funds|balance|assets?)\s+between\s+(my\s+)?(own\s+)?wallets?|"
    r"transfer\s+between\s+(my\s+)?(own\s+)?wallets?|"
    r"internal\s+transfer|"
    r"wallet[\s-]to[\s-]wallet\s+transfer|"
    r"move\s+(crypto|funds|balance).*between.*(my\s+)?(own\s+)?wallets?|"
    r"transfer\s+within\s+al.?fardan|"
    # James Set 7 — custody<->staking / lending / otc buckets are
    # also internal transfers. Answer is same: instant + free.
    r"(transfer|move|send).*between\s+(custody|staking|lending|otc)\s+(and|&|to|-|,)\s+(custody|staking|lending|otc)|"
    r"how\s+long.*(transfer|move).*between\s+(custody|staking|lending|otc)|"
    r"(custody|staking|lending|otc)\s+to\s+(custody|staking|lending|otc)"
    r")\b",
    re.I,
)

# Interest rate change after loan origination.
_INTEREST_RATE_CHANGE_RE = re.compile(
    r"\b("
    # Tolerant of filler words between "rate" and the change verb.
    r"(interest\s+|loan\s+)?rate(\s+\w+){0,5}\s+(change|changes|vary|varies|float|floats|adjust|adjusts|move|moves)|"
    r"(can|does|will)\s+(my\s+|the\s+)?(interest\s+|loan\s+)?rate\s+(change|vary|float|adjust)|"
    r"fixed\s+or\s+variable(\s+\w+){0,3}|"
    r"variable\s+or\s+fixed(\s+\w+){0,3}|"
    # James Set 12 Q1 — "is the interest rate fixed or variable?"
    r"is\s+(my\s+|the\s+|your\s+)?(interest\s+|loan\s+)?rate\s+(fixed|variable|floating)|"
    r"(interest\s+|loan\s+)?rate\s+(fixed|variable|floating)"
    r")\b",
    re.I,
)

# Who pays the staking network (gas) fees?
_STAKING_NETWORK_FEES_RE = re.compile(
    r"\b("
    r"(network|gas|transaction)\s+fees?\s+(for|on|in|from|during)\s+staking|"
    r"(who|does\s+al[\s-]?fardan|do\s+you)\s+pays?\s+(the\s+)?(gas|network|transaction)\s+(fees?|costs?)|"
    r"network\s+costs?\s+(for|on)\s+stak|"
    r"gas\s+costs?\s+(for|on)\s+stak"
    r")\b",
    re.I,
)

# Minimum balance for custody.
_CUSTODY_MINIMUM_BALANCE_RE = re.compile(
    r"\b("
    r"minimum\s+balance\s+(for\s+|in\s+)?custody|"
    r"custody\s+(minimum|min)\s+balance|"
    r"minimum.*(custody|cold\s+storage|fireblocks)|"
    r"(do\s+you|does\s+al[\s-]?fardan)\s+(require|have)\s+(a\s+)?minimum\s+balance|"
    r"is\s+there\s+(a\s+)?minimum.*custody|"
    r"what\s+if\s+my\s+balance\s+drops?"
    r")\b",
    re.I,
)

# Who is the OTC counterparty?
_OTC_COUNTERPARTY_RE = re.compile(
    r"\b("
    r"who\s+is\s+(the\s+|my\s+)?(counterparty|counter[\s-]?party|principal|dealer)|"
    r"counter[\s-]?party\s+(for|in|of|on)\s+(otc|my\s+trade|the\s+trade)|"
    r"otc\s+counter[\s-]?party|"
    r"(are|is)\s+you\s+(the\s+)?(counterparty|principal|dealer)|"
    r"al[\s-]?fardan\s+(counterparty|principal|dealer)"
    r")\b",
    re.I,
)

# API key create / manage / revoke.
_API_KEY_MANAGEMENT_RE = re.compile(
    r"\b("
    r"(create|generate|make|get|issue)\s+(an?\s+|new\s+|my\s+)?api\s+keys?|"
    r"(delete|revoke|remove|cancel|rotate)\s+(an?\s+|my\s+|the\s+)?api\s+keys?|"
    r"(manage|manages|managing)\s+(my\s+|the\s+)?api\s+keys?|"
    r"how\s+(do|to)\s+(i\s+)?(create|generate|make|get|use)\s+(an?\s+)?api\s+key|"
    r"api\s+key\s+(management|create|creation|delete|revoke|scope|scoping|permission)"
    r")\b",
    re.I,
)

# Withdraw staking rewards specifically (must outrank generic withdraw).
_STAKING_REWARDS_WITHDRAW_RE = re.compile(
    r"\b("
    r"withdraw\s+(my\s+)?(staking\s+)?rewards?(\s+anytime)?|"
    r"(pull|cash)\s+(out|down)\s+(my\s+)?(staking\s+)?rewards?|"
    r"take\s+out\s+(my\s+)?(staking\s+)?rewards?|"
    r"(staking\s+)?rewards?\s+(withdraw|withdrawal)"
    r")\b",
    re.I,
)

# ═════════════════════════════════════════════════════════════════════
#  James QA Set 6 (3 more intents)
# ═════════════════════════════════════════════════════════════════════

# LTV formula / calculation basis.
_LTV_CALCULATION_RE = re.compile(
    r"\b("
    r"how\s+is\s+(the\s+|my\s+)?ltv\s+(calculated|computed|worked\s+out|derived)|"
    r"ltv\s+(formula|calculation|based\s+on|formula\s+is)|"
    r"loan[\s-]to[\s-]value\s+(formula|calculation|ratio|calculated|computed)|"
    r"how\s+do\s+you\s+calculate\s+ltv|"
    r"ltv\s+(current\s+|real[\s-]time\s+|live\s+)?(market\s+)?price"
    r")\b",
    re.I,
)

# Staking rewards history / export.
_REWARDS_HISTORY_RE = re.compile(
    r"\b("
    r"(staking\s+)?rewards?\s+(history|log|record|archive|tape)|"
    r"(history|log)\s+of\s+(my\s+)?(staking\s+)?rewards?|"
    r"how\s+far\s+back\s+(do\s+)?(rewards?|my\s+rewards?)|"
    r"export\s+(my\s+)?(staking\s+)?rewards?|"
    r"see\s+(my\s+)?(staking\s+)?rewards?\s+history|"
    r"view\s+(my\s+)?(past|historical)\s+(staking\s+)?rewards?"
    r")\b",
    re.I,
)

# Session timeout / auto-logout.
_SESSION_TIMEOUT_RE = re.compile(
    r"\b("
    r"session\s+(timeout|time\s+out|times?\s+out|expir|expires|length|duration|limit|time\s+limit)|"
    r"(how\s+long\s+before|when\s+does)\s+(my\s+)?session\s+(times?[\s-]?out|expire|end|terminate|log\s+me\s+out)|"
    r"auto[\s-]?(logout|log[\s-]?out|sign[\s-]?out)|"
    r"automatic(ally)?\s+logged?\s+out|"
    r"change\s+(my\s+|the\s+)?session\s+(timeout|duration|length)|"
    r"extend\s+(my\s+|the\s+)?session"
    r")\b",
    re.I,
)

# ═════════════════════════════════════════════════════════════════════
#  James QA Sets 7 + 8 + 9 (13 more intents)
# ═════════════════════════════════════════════════════════════════════

# "Is there a minimum amount to claim staking rewards?"
_CLAIM_REWARDS_RE = re.compile(
    r"\b("
    r"(claim|claiming|collect)\s+(my\s+|staking\s+)?rewards?|"
    r"minimum.*(to\s+)?claim\s+(staking\s+)?rewards?|"
    r"(how\s+do\s+i|when\s+can\s+i)\s+(claim|collect)\s+(staking\s+)?rewards?"
    r")\b",
    re.I,
)

# "How do I pay the interest on my loan?" — existing loan management.
# Outranks the generic loan_question / _START_LOAN_RE which ask about
# opening a new loan.
_INTEREST_PAYMENT_RE = re.compile(
    r"\b("
    r"(how\s+do\s+i|how\s+to)\s+pay\s+(the\s+|my\s+)?interest|"
    r"pay(ing)?\s+(the\s+|my\s+)?(loan\s+)?interest|"
    r"interest\s+payment|"
    r"pay\s+(my\s+|the\s+)?(loan\s+)?(interest|profit|cost)\s+(with|in)\s+(crypto|btc|eth|usdt|usdc|dai|stablecoins?)|"
    r"make\s+(a\s+|my\s+)?(loan\s+)?interest\s+payment"
    r")\b",
    re.I,
)

# "Can I have a shared custody account?" — must outrank navigate_custody.
_SHARED_ACCOUNT_RE = re.compile(
    r"\b("
    r"shared\s+(custody\s+)?account|"
    r"joint\s+(custody\s+)?account|"
    r"account\s+(with\s+)?(another\s+person|my\s+(wife|husband|partner|spouse|kids?|children|family))|"
    r"(add|register)\s+(another\s+|a\s+second\s+)?owner|"
    r"co[\s-]?own(er|ership)\s+(account|custody)"
    r")\b",
    re.I,
)

# "Can I cancel an OTC trade after confirming?"
_CANCEL_OTC_RE = re.compile(
    r"\b("
    r"cancel\s+(an?\s+|my\s+|the\s+)?otc\s+(trade|order|quote)|"
    r"reverse\s+(an?\s+|my\s+|the\s+)?otc\s+(trade|order)|"
    r"otc\s+(trade|order)\s+cancel|"
    r"undo\s+(an?\s+|my\s+|the\s+)?otc\s+(trade|order)|"
    r"back\s+out\s+(of\s+)?(an?\s+|my\s+|the\s+)?(otc\s+)?(trade|order)"
    r")\b",
    re.I,
)

# "Is there a maximum OTC trade size?"
_OTC_MAX_SIZE_RE = re.compile(
    r"\b("
    r"maximum\s+otc\s+(trade|ticket|size|limit)|"
    r"(largest|biggest|max)\s+(otc\s+)?(trade|ticket|order)|"
    r"otc\s+(max|maximum|largest|biggest|cap|ceiling)\s+(size|limit|trade)?|"
    r"trade\s+more\s+than\s+(\$\s?\d|\d+\s*(m|million))"
    r")\b",
    re.I,
)

# "Do I need to provide a tax identification number?"
_TIN_REQUIRED_RE = re.compile(
    r"\b("
    r"(tax\s+identification|tax\s+id|tin)\s+(required|needed|necessary|mandatory)|"
    r"(provide|submit|give)\s+(my\s+|a\s+)?(tax\s+identification|tax\s+id|tin)|"
    r"(need|require).*(tax\s+identification|tax\s+id|tin)|"
    r"(tax\s+identification|tax\s+id|tin)\s+(to\s+open|for\s+opening|for\s+kyc)"
    r")\b",
    re.I,
)

# "How do I add more collateral to an existing loan?" — existing
# loan management, MUST outrank _START_LOAN_RE.
_ADD_COLLATERAL_RE = re.compile(
    r"\b("
    r"(add|top\s*up|topup|increase)\s+(more\s+)?(my\s+|the\s+)?collateral|"
    r"add\s+collateral\s+to\s+(an?\s+|my\s+|the\s+)?(existing\s+)?loan|"
    r"increase\s+(my\s+|the\s+)?(loan\s+)?collateral|"
    r"post\s+more\s+collateral|"
    r"reduce\s+(my\s+|the\s+)?ltv\s+(by|with|using)\s+(more\s+)?collateral"
    r")\b",
    re.I,
)

# "Can I restrict logins to specific IP addresses?" — distinct from
# the withdrawal-address whitelist (which is for outbound addresses).
_IP_WHITELISTING_RE = re.compile(
    r"\b("
    r"(restrict|limit|lock)\s+(my\s+)?(logins?|sign[\s-]?ins?|access)\s+(to\s+)?(specific\s+|certain\s+)?ip(\s+address)?|"
    r"ip\s+(whitelist|whitelisting|white[\s-]?listed?|restrict|restriction|lock|filter)|"
    r"(allow|accept)\s+logins?\s+only\s+from\s+(specific\s+|certain\s+)?ip|"
    r"(login|sign[\s-]?in|access)\s+ip\s+(whitelist|restrict|lock)"
    r")\b",
    re.I,
)

# "Can I negotiate the interest rate on a large loan?"
_NEGOTIATE_RATE_RE = re.compile(
    r"\b("
    r"negotiate\s+(the\s+|my\s+|a\s+)?(interest\s+)?rate|"
    r"(better|custom|bespoke|tailored|preferential|lower)\s+(interest\s+)?rate|"
    r"(can|will)\s+you\s+(offer|give|provide)\s+(me\s+)?(a\s+)?(better|lower|custom|preferential)\s+(interest\s+)?rate|"
    r"rate\s+negotiation"
    r")\b",
    re.I,
)

# "What happens if I don't repay my loan at all?"
_LOAN_DEFAULT_RE = re.compile(
    r"\b("
    r"(don't|do\s+not|never|fail\s+to|can'?t|cannot|unable\s+to)\s+repay\s+(my\s+)?loan|"
    r"loan\s+(default|defaults|defaulting|unpaid|non[\s-]?repayment)|"
    r"what\s+(if|happens\s+if)\s+(i\s+)?(don't|do\s+not|can'?t|never|fail)\s+(pay|repay)|"
    r"default\s+on\s+(my\s+|the\s+)?loan|"
    r"miss\s+(a\s+|my\s+|the\s+)?loan\s+payment"
    r")\b",
    re.I,
)

# "Are my staked assets spread across multiple validators?"
_VALIDATOR_DIVERSIFICATION_RE = re.compile(
    r"\b("
    r"(multiple|many|several|different)\s+validators|"
    r"(spread|distributed|diversified)\s+(my\s+)?(staked\s+)?(stake|assets|funds)|"
    r"validator\s+(diversif|spread|distribution)|"
    r"how\s+many\s+validators|"
    r"single\s+validator\s+(risk|concentration|expose)"
    r")\b",
    re.I,
)

# "Do you provide monthly custody statements?"
_MONTHLY_STATEMENTS_RE = re.compile(
    r"\b("
    r"(monthly|weekly|quarterly|annual|yearly)\s+(custody\s+|account\s+)?statements?|"
    r"(custody\s+|account\s+)?statements?\s+(monthly|weekly|quarterly|annual|yearly)|"
    r"(provide|get|download|receive)\s+(my\s+|a\s+)?(monthly\s+|weekly\s+|quarterly\s+|custody\s+|account\s+)?statement|"
    r"custody\s+statement"
    r")\b",
    re.I,
)

# "What happens during liquidation?" — distinct from margin-call.
# Focuses on the ACT of liquidation: how much, how it's executed.
_LIQUIDATION_DETAILS_RE = re.compile(
    r"\b("
    r"what\s+happens\s+during\s+liquidation|"
    r"liquidation\s+(process|fee|cost|amount|how|executes?|steps)|"
    r"how\s+much.*(do\s+i|will\s+i)\s+lose\s+(during|on|in)\s+liquidation|"
    r"(process|steps)\s+of\s+liquidation|"
    r"partial\s+liquidation|"
    r"how\s+does\s+liquidation\s+work"
    r")\b",
    re.I,
)

# ═════════════════════════════════════════════════════════════════════
#  James QA Sets 11 + 12 + 13 (11 more intents)
# ═════════════════════════════════════════════════════════════════════

# "Can I repay PART of my loan early?" — distinct from early_repayment
# because the user explicitly says "part" or "partial", and the reply
# should emphasise that partial is allowed.
_PARTIAL_REPAYMENT_RE = re.compile(
    r"\b("
    r"(repay|pay|settle)\s+(part|partial|partially|some|a\s+portion|a\s+bit|a\s+little)\s+(of\s+)?(my\s+|the\s+)?loan|"
    r"partial\s+(loan\s+)?(repayment|payment|payoff)|"
    r"(pay|repay)\s+(a\s+)?portion|"
    r"part[\s-]?pay(ment)?\s+loan"
    r")\b",
    re.I,
)

# "Can I connect my Ledger / Trezor?" — answer NO with explanation.
_HARDWARE_WALLET_RE = re.compile(
    r"\b("
    r"(connect|use|integrate|link|pair)\s+(my\s+|a\s+)?(ledger|trezor|hardware\s+wallet|keepkey|coldcard|cold\s+card|bitbox|lattice)|"
    r"(ledger|trezor|hardware\s+wallet)\s+(to|with|in|into)\s+(custody|my\s+account|al.?fardan|q9)|"
    r"hardware\s+wallet\s+(support|integration|connect|connection|compat)|"
    r"does\s+(al.?fardan|q9)\s+support\s+(ledger|trezor|hardware)"
    r")\b",
    re.I,
)

# "Can I receive my rewards in another crypto?" — NO, paid in-kind.
_REWARDS_IN_DIFFERENT_ASSET_RE = re.compile(
    r"\b("
    r"(receive|get|take|paid)\s+(my\s+|the\s+)?(staking\s+)?rewards?\s+in\s+(a\s+|an\s+)?(different|another|other|alternative|stable|stablecoin|stablecoins|usdc|usdt|dai)\b|"
    r"(receive|get|take|paid)\s+(my\s+|the\s+)?(staking\s+)?rewards?\s+in\s+(a\s+|an\s+)?(different|another|other|alternative)\s+(crypto|coin|asset|currency|token|stablecoin)|"
    r"rewards?\s+(paid|distributed)\s+in\s+(another|different|other|alternative|stablecoins?|usdc|usdt|dai)|"
    r"(eth|sol|dot|atom|ada|avax|pol|matic)\s+rewards?\s+(as|in|paid\s+in)\s+(usdc|usdt|dai|stablecoin|cash|fiat|usd)|"
    r"swap\s+(my\s+|the\s+)?staking\s+rewards?"
    r")\b",
    re.I,
)

# "Didn't receive my staking rewards today."
_MISSING_REWARDS_RE = re.compile(
    r"\b("
    r"(didn'?t|did\s+not|don'?t|do\s+not|haven'?t|have\s+not)\s+(receive|get|see)\s+(my\s+|any\s+)?(staking\s+)?rewards?|"
    r"(missing|missed|lost|not\s+getting|not\s+seeing)\s+(my\s+|some\s+|any\s+)?(staking\s+)?rewards?|"
    r"(staking\s+)?rewards?\s+(not|never)\s+(paid|distributed|credited|arrived|received|shown|visible|appearing)|"
    r"where\s+are\s+my\s+(staking\s+)?rewards?|"
    r"expected\s+(my\s+|some\s+)?rewards?\s+(today|yesterday|this)"
    r")\b",
    re.I,
)

# "Can I send crypto directly to ANOTHER Al-Fardan user?" — distinct
# from the existing internal_transfer which is between a SINGLE user's
# own wallets. This one is between DIFFERENT users → answer NO.
_USER_TO_USER_TRANSFER_RE = re.compile(
    r"\b("
    # Allow brand-name filler like "al fardan q9 user" — the q9 (plus
    # any short token) can sit between "fardan" and the user noun.
    r"send\s+(crypto|btc|eth|sol|funds|assets?|money|coins?)\s+(directly\s+)?to\s+another\s+(al[\s.-]?fardan(\s+\w+){0,2}\s+)?(user|person|account|client|customer|member)|"
    r"transfer\s+(crypto|funds|assets?|money|balance)\s+between\s+(users|accounts|al[\s.-]?fardan(\s+\w+){0,2}\s+clients|different\s+accounts)|"
    r"send\s+(crypto|btc|eth|sol|funds|balance)\s+to\s+(my\s+)?(friend|wife|husband|partner|colleague|brother|sister|son|daughter|family)|"
    r"(peer[\s-]?to[\s-]?peer|p2p|user[\s-]to[\s-]user)\s+(transfer|send|payment)|"
    r"send.*to\s+another\s+al[\s.-]?fardan"
    r")\b",
    re.I,
)

# "Can I get an OTC quote outside business hours?"
_OTC_AFTER_HOURS_RE = re.compile(
    r"\b("
    # Generous: "outside [of] [the] [UAE] [business] hours" — allow
    # up to 3 qualifier words between the preposition and "hours".
    r"otc\s+(quote|trade|price|deal)\s+(after|outside|beyond|past|late|before)(\s+\w+){0,4}\s+hours|"
    r"(24/7|24x7|weekend|night|nighttime|overnight|late[\s-]?night)\s+otc|"
    r"quote\s+(outside|after)(\s+\w+){0,4}\s+hours|"
    r"urgent\s+(otc\s+)?(quote|trade|deal)|"
    r"otc\s+(weekend|sunday|saturday|friday)"
    r")\b",
    re.I,
)

# "Can I see my login history?"
_LOGIN_HISTORY_RE = re.compile(
    r"\b("
    r"(see|view|check|display|show)\s+(my\s+|the\s+)?login\s+history|"
    r"login\s+(history|log|activity|records?|audit)|"
    r"(history|log)\s+of\s+(my\s+)?logins?|"
    r"recent\s+logins?|"
    r"list\s+of\s+(my\s+)?logins?"
    r")\b",
    re.I,
)

# "Where is the interest taken from?"
_INTEREST_SOURCE_RE = re.compile(
    r"\b("
    r"where\s+is\s+(the\s+|my\s+)?interest\s+(taken|deducted|charged|paid|drawn|pulled)|"
    r"interest\s+(come|comes|taken|paid|deducted|drawn)\s+from.*(wallet|collateral|balance|custody)|"
    r"interest.*(deducted|taken)\s+from.*(wallet|collateral|balance)|"
    r"wallet\s+or\s+(my\s+)?collateral"
    r")\b",
    re.I,
)

# "Do staking rewards expire?"
_REWARDS_EXPIRATION_RE = re.compile(
    r"\b("
    r"(do|does)\s+(staking\s+|my\s+)?rewards?\s+expire|"
    r"(staking\s+)?rewards?\s+(expire|expiration|expiry)|"
    r"(claim|unclaimed)\s+rewards?\s+(expir|time\s+limit|deadline|lose|dropped)|"
    r"rewards?\s+(time\s+limit|deadline|ttl)"
    r")\b",
    re.I,
)

# "How long does it take to withdraw to an EXTERNAL wallet?" —
# distinct from internal_transfer (instant) and staking unbonding.
_WITHDRAWAL_TIME_RE = re.compile(
    r"\b("
    r"how\s+long.*(withdraw|withdrawal|transfer|move|send)\s+(crypto|btc|eth|sol|funds|assets?)?.*(external|my\s+(personal\s+)?wallet|outside|off[\s-]?platform)|"
    r"how\s+long.*(move|transfer|send)\s+(crypto|funds|assets?)\s+to\s+(external|outside|my\s+wallet|ledger|trezor)|"
    r"(withdrawal|withdraw)\s+time.*(external|outside|off[\s-]?platform)|"
    r"how\s+(fast|quick|long).*withdraw\s+(my\s+)?(crypto|funds|assets?|btc|eth|sol)"
    r")\b",
    re.I,
)

# "Can I see all devices logged into my account?"
_ACTIVE_SESSIONS_RE = re.compile(
    r"\b("
    r"(see|view|check|list|show|display)\s+(all\s+)?(my\s+)?(active\s+)?(sessions?|devices|logins?)\s+(logged|signed)\s+(in|into)|"
    r"(logged|signed)\s+in\s+(devices|sessions?)|"
    r"active\s+(sessions?|devices|logins?)|"
    r"where\s+am\s+i\s+(logged|signed)\s+in|"
    r"(list|show|see)\s+(my\s+|all\s+)?devices|"
    r"log\s+out\s+(all|other|remote)\s+(devices?|sessions?)|"
    r"remote\s+(log[\s-]?out|sign[\s-]?out)"
    r")\b",
    re.I,
)

# ─── Set 14 intents ───────────────────────────────────────────────────

# "Can I transfer my loan to another person or entity?" — NO.
_LOAN_TRANSFER_RE = re.compile(
    r"\b("
    r"transfer\s+(my\s+|the\s+)?loan\s+to\s+(another|someone|a\s+different|a\s+new)|"
    r"(assign|sell|reassign|hand\s+over|pass\s+on|pass\s+over|novate)\s+(my\s+|the\s+)?loan|"
    r"loan\s+(transfer|assignment|novation|novate)\b|"
    r"(transfer|give|pass)\s+(my\s+)?loan\s+(to|over)"
    r")\b",
    re.I,
)

# "Where are your validators physically located?"
_VALIDATOR_LOCATIONS_RE = re.compile(
    r"\b("
    r"where\s+are\s+(your|our|the)\s+validators?\s+(located|based|hosted|physically|geograph)|"
    r"validator\s+(location|locations|geography|data\s+cent(er|re)s?|region|regions|country|countries)|"
    r"(physical|geographic)\s+(location|locations|diversit)\s+of\s+validators?|"
    r"which\s+country.*validators?|"
    r"where\s+do\s+you\s+(host|run)\s+validators?"
    r")\b",
    re.I,
)

# "Snapshot of my assets at a specific date and time." — distinct
# from monthly_statements which is just the default monthly cut.
_HISTORICAL_SNAPSHOT_RE = re.compile(
    r"\b("
    r"snapshot\s+of\s+(my\s+|all\s+)?(assets|holdings|balances|portfolio)|"
    r"historical\s+(snapshot|holdings|balance|balances|portfolio)|"
    r"statement\s+for\s+(a\s+)?(specific|custom|particular)\s+(date|time|period|window)|"
    r"balance.*on\s+(a\s+)?(specific|particular)\s+date|"
    r"what\s+did\s+i\s+(hold|own)\s+on\s+\d"
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
    # These MUST fire BEFORE _START_WITHDRAW_RE so that the words
    # "withdraw" / "wallet" don't hijack a specific question about:
    #   - whitelisting (mentions "withdrawal addresses")
    #   - internal transfers (mentions "wallets")
    #   - withdrawing staking rewards
    #   - user-to-user transfers (mentions "send crypto")
    #   - withdrawal time to external wallet
    if _WHITELIST_ADDRESS_RE.search(text):
        return "whitelist_address"
    if _IP_WHITELISTING_RE.search(text):
        return "ip_whitelisting"
    # user-to-user transfer BEFORE internal_transfer — former is
    # "to another user", latter is "between my own wallets".
    if _USER_TO_USER_TRANSFER_RE.search(text):
        return "user_to_user_transfer"
    # Withdrawal time to EXTERNAL wallet also beats _INTERNAL_TRANSFER_RE
    # because phrasings like "move assets between custody and external
    # wallet" otherwise match internal_transfer's "custody ... to ..."
    # alternative with "external" as the target service.
    if _WITHDRAWAL_TIME_RE.search(text):
        return "withdrawal_time"
    if _INTERNAL_TRANSFER_RE.search(text):
        return "internal_transfer"
    if _STAKING_REWARDS_WITHDRAW_RE.search(text):
        return "staking_rewards_withdraw"
    if _CLAIM_REWARDS_RE.search(text):
        return "claim_rewards"
    # Security incident must outrank every nav — "my account is hacked"
    # shouldn't just drop the user on /settings, they need the
    # emergency flow.
    if _SECURITY_INCIDENT_RE.search(text):
        return "security_incident"
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
    # OTC-specific asks beat navigate_otc because they need bespoke
    # answers, not a generic OTC intro.
    if _CANCEL_OTC_RE.search(text):
        return "cancel_otc"
    if _OTC_MAX_SIZE_RE.search(text):
        return "otc_max_size"
    if _OTC_AFTER_HOURS_RE.search(text):
        return "otc_after_hours"
    if _GOTO_OTC_RE.search(text):
        return "navigate_otc"
    # custody_minimum_balance must outrank _GOTO_CUSTODY_RE — "minimum
    # balance for custody" contains "custody" and would otherwise get
    # the generic navigate_custody reply instead of the specific answer.
    if _CUSTODY_MINIMUM_BALANCE_RE.search(text):
        return "custody_minimum_balance"
    # These also mention "custody" or "account" and would hijack the
    # nav route otherwise.
    if _SHARED_ACCOUNT_RE.search(text):
        return "shared_account"
    if _MONTHLY_STATEMENTS_RE.search(text):
        return "monthly_statements"
    if _HARDWARE_WALLET_RE.search(text):
        return "hardware_wallet"
    if _GOTO_CUSTODY_RE.search(text):
        return "navigate_custody"
    # Settings-specific asks beat navigate_settings.
    if _LOGIN_HISTORY_RE.search(text):
        return "login_history"
    if _ACTIVE_SESSIONS_RE.search(text):
        return "active_sessions"
    if _GOTO_SETTINGS_RE.search(text):
        return "navigate_settings"
    # interest_source contains "wallet" and would lose to navigate_wallets
    # otherwise.
    if _INTEREST_SOURCE_RE.search(text):
        return "interest_source"
    if _GOTO_WALLETS_RE.search(text):
        return "navigate_wallets"
    if _CONTACT_RE.search(text):
        return "contact_support"

    # ─── Insurance-claim check must beat the signup regex because
    #     "how do I start an insurance claim" matches "how do i start"
    #     in _SIGNUP_RE. Set 15 Q2.
    if _INSURANCE_CLAIM_RE.search(text):
        return "insurance_claim"

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
    # "Network fees for staking" must outrank staking_fees — it's a
    # distinct question about who pays gas, not about our commission.
    if _STAKING_NETWORK_FEES_RE.search(text):
        return "staking_network_fees"
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
    # ─── Set 4 + 5 specifics — must beat generic loan_question /
    #     staking_question which would otherwise shadow these.
    if _EARLY_REPAYMENT_RE.search(text):
        return "early_repayment"
    if _LOAN_EXTENSION_RE.search(text):
        return "loan_extension"
    if _INTEREST_RATE_CHANGE_RE.search(text):
        return "interest_rate_change"
    if _INTEREST_CALCULATION_RE.search(text):
        return "interest_calculation"
    if _VALIDATOR_CHOICE_RE.search(text):
        return "validator_choice"
    # staking_network_fees + custody_minimum_balance already fire
    # earlier (above the generic nav routes), so we don't need to
    # re-check them here.
    # temporary_deactivation must beat account_closure because phrases
    # like "temporarily deactivate instead of closing" contain "close"
    # which account_closure would otherwise match.
    if _TEMPORARY_DEACTIVATION_RE.search(text):
        return "temporary_deactivation"
    if _ACCOUNT_CLOSURE_RE.search(text):
        return "account_closure"
    if _OTC_COUNTERPARTY_RE.search(text):
        return "otc_counterparty"
    # Set 6 — session_timeout must beat the generic navigate_settings
    # route (which would catch "session timeout" via "settings").
    # Actually navigate_settings already fired in the action block;
    # if we reach here it didn't match, so we're safe to just check.
    if _LTV_CALCULATION_RE.search(text):
        return "ltv_calculation"
    if _REWARDS_HISTORY_RE.search(text):
        return "rewards_history"
    if _SESSION_TIMEOUT_RE.search(text):
        return "session_timeout"
    # ─── Sets 7/8/9 — all must beat the generic topic routes ─────
    if _CLAIM_REWARDS_RE.search(text):
        return "claim_rewards"
    if _INTEREST_PAYMENT_RE.search(text):
        return "interest_payment"
    if _ADD_COLLATERAL_RE.search(text):
        return "add_collateral"
    if _NEGOTIATE_RATE_RE.search(text):
        return "negotiate_rate"
    if _LOAN_DEFAULT_RE.search(text):
        return "loan_default"
    if _TIN_REQUIRED_RE.search(text):
        return "tin_required"
    if _VALIDATOR_DIVERSIFICATION_RE.search(text):
        return "validator_diversification"
    if _LIQUIDATION_DETAILS_RE.search(text):
        return "liquidation_details"
    # ─── Sets 11-14 — specific Q&A that must beat generic topic
    #     routes. Loan-management + staking-management + info-lookup.
    # (interest_source / hardware_wallet already handled above, in the
    # nav-outrank sections)
    if _PARTIAL_REPAYMENT_RE.search(text):
        return "partial_repayment"
    if _LOAN_TRANSFER_RE.search(text):
        return "loan_transfer"
    if _REWARDS_IN_DIFFERENT_ASSET_RE.search(text):
        return "rewards_in_different_asset"
    if _MISSING_REWARDS_RE.search(text):
        return "missing_rewards"
    if _REWARDS_EXPIRATION_RE.search(text):
        return "rewards_expiration"
    if _VALIDATOR_LOCATIONS_RE.search(text):
        return "validator_locations"
    if _HISTORICAL_SNAPSHOT_RE.search(text):
        return "historical_snapshot"
    # ─── Set 15 ──────────────────────────────────────────────────
    # (insurance_claim + temporary_deactivation already handled above,
    # earlier in classify() — they needed to outrank signup /
    # account_closure respectively.)
    if _OTC_SETTLEMENT_CURRENCY_RE.search(text):
        return "otc_settlement_currency"
    if _COLLATERAL_REVALUATION_RE.search(text):
        return "collateral_revaluation"
    if _TAX_WITHHOLDING_RE.search(text):
        return "tax_withholding"
    if _ACCOUNT_RECOVERY_RE.search(text):
        return "account_recovery"
    if _API_KEY_MANAGEMENT_RE.search(text):
        return "api_key_management"
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


# ─── Intents migrated to KB (app/data/kb/20_intent_migrations.jsonl)
#
# These info-only intents used to have hardcoded scripted replies in
# this file — one `if intent == "..."` branch each. Every round of QA
# feedback forced me to patch the regex classifier to catch new user
# phrasings, producing the "intent whack-a-mole" pattern James flagged.
#
# The fix is to treat answers as DATA, not code:
#   - Each former intent now has a KB entry with rich aliases covering
#     every phrasing we've seen.
#   - The classify() regex still fires (harmless — it just returns an
#     intent name). `scripted_reply()` returns None for these, so we
#     fall through to the composer → semantic retriever → KB hit.
#   - The semantic retriever catches paraphrases automatically — new
#     wordings don't need new regex.
#
# Adding new Q&A coverage is now a KB edit (JSONL file), not a code
# change. The bot gets smarter by data, not by more `if` branches.
#
# KEEP intent + scripted_reply for action/navigation intents that need
# to render specific CRM buttons (navigate_*, withdraw_request,
# deposit_request, transfer_from_exchange, signup, balance_check,
# contact_support, greeting, goodbye, affirmation, negation,
# security_incident).
_MIGRATED_TO_KB: set[str] = {
    "slashing_question", "tax_reporting", "network_outage",
    "withdrawal_delay", "staking_fees", "staking_frequency",
    "unauthorized_login", "insurance_claim", "auto_reinvest",
    "otc_quote_validity", "interest_calculation", "whitelist_address",
    "account_closure", "early_repayment", "validator_choice",
    "loan_extension", "internal_transfer", "interest_rate_change",
    "staking_network_fees", "custody_minimum_balance",
    "otc_counterparty", "api_key_management", "staking_rewards_withdraw",
    "ltv_calculation", "rewards_history", "session_timeout",
    "claim_rewards", "interest_payment", "shared_account", "cancel_otc",
    "otc_max_size", "tin_required", "add_collateral", "ip_whitelisting",
    "negotiate_rate", "loan_default", "validator_diversification",
    "monthly_statements", "liquidation_details", "partial_repayment",
    "hardware_wallet", "rewards_in_different_asset", "missing_rewards",
    "user_to_user_transfer", "otc_after_hours", "login_history",
    "interest_source", "rewards_expiration", "withdrawal_time",
    "active_sessions", "loan_transfer", "validator_locations",
    "historical_snapshot", "otc_settlement_currency",
    "temporary_deactivation", "collateral_revaluation",
    "tax_withholding", "account_recovery",
}


# ─── Scripted replies + action buttons per intent ────────────────────

def scripted_reply(intent: Intent) -> str | None:
    """Canned response text for intent-matched messages.

    For intents that have multiple phrasings (greeting, goodbye,
    affirmation, negation, signup, contact_support, balance_check) we
    pick a random variant from refine.vary — the bot doesn't say the
    exact same thing every time.

    Returns None for intents where we fall through to KB retrieval
    instead (loan_question, staking_question, unknown, and every
    intent listed in _MIGRATED_TO_KB below).
    """
    # Intents whose answers moved to KB entries in 20_intent_migrations
    # .jsonl — the semantic retriever handles them now. Adding new Q&A
    # coverage is a KB edit, not a code change. DO NOT add info-only
    # intents back to this file; extend the KB entry's `aliases` list
    # instead so the retriever picks up new phrasings.
    if intent in _MIGRATED_TO_KB:
        return None

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
            "Our claims team acknowledges within 2 UAE business hours; full "
            "Lloyd's syndicate processing typically takes 60 to 90 days from "
            "the date all required documents are submitted. Required: account "
            "details, the transaction hash, and any supporting evidence. "
            "Would you like me to connect you with the claims team right now?"
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
    # ─── Set 4 replies ───────────────────────────────────────────
    if intent == "interest_calculation":
        return (
            "Interest accrues daily on your outstanding loan balance and is "
            "charged monthly. The Murabaha cost is pre-calculated at signing "
            "and fixed for the full term — so total cost is known day one. "
            "You can see your accrued interest live in the Lending module of "
            "your dashboard. Would you like me to calculate the interest for "
            "a specific loan amount?"
        )
    if intent == "whitelist_address":
        return (
            "Yes — you can whitelist withdrawal addresses in Settings → "
            "Security → Whitelisted Addresses. Adding a new address enters a "
            "24-hour cooldown before it becomes active — this is a security "
            "measure that prevents an attacker from adding their own address "
            "and withdrawing immediately. You can whitelist up to 20 "
            "addresses per asset. Would you like me to guide you through "
            "adding one?"
        )
    if intent == "account_closure":
        return (
            "Yes, you can close your account once balances are zero and any "
            "active loans / staking positions are settled. Email "
            "support@alfardanq9.com to request closure and we'll process it "
            "within 3 UAE business days. No — a closed account cannot be "
            "reopened; the email is retired and you would need to create a "
            "new account if you want to come back later. Data retention: "
            "most personal data is deleted on closure, but transaction "
            "records are retained for 5-7 years per UAE AML regulations. "
            "Would you like me to send you our full data retention policy?"
        )
    if intent == "early_repayment":
        return (
            "Yes, you can repay your loan early at any time — there is no "
            "prepayment penalty. Interest is calculated pro-rata for the "
            "actual duration of the loan, so you save on the remaining "
            "scheduled cost. Full repayment releases all collateral back to "
            "your custody vault within 24 hours. Would you like me to "
            "calculate your current payoff amount?"
        )
    if intent == "validator_choice":
        return (
            "No, you cannot select a specific validator. Our institutional "
            "staking team runs a proprietary validator fleet chosen for "
            "maximum uptime (99.95% SLA), security, and our slashing-"
            "protection guarantee. For clients staking above $5M per network "
            "we can discuss dedicated validator routing with your "
            "relationship manager. Would you like me to explain how our "
            "validator selection process works?"
        )
    if intent == "security_incident":
        return (
            "I hear you — this is urgent. Take these steps immediately: "
            "(1) change your password in Settings → Security, (2) revoke any "
            "API keys you don't recognise, (3) enable 2FA if it isn't "
            "already, (4) email security@alfardanq9.com or call our 24/7 "
            "emergency line at +971 4 123 4568 right now. Please do NOT "
            "dismiss any unfamiliar alerts. Would you like me to connect you "
            "with the security team this minute?"
        )
    # ─── Set 5 replies ───────────────────────────────────────────
    if intent == "loan_extension":
        return (
            "Yes, you can extend your loan term. Contact your relationship "
            "manager at least 30 days before maturity — extension terms are "
            "re-priced at current rates against current collateral valuation. "
            "No extension fee, but the new-term Murabaha cost applies from "
            "the extension date forward. Would you like me to connect you "
            "with your relationship manager to start the conversation?"
        )
    if intent == "internal_transfer":
        return (
            "Yes, you can transfer crypto between your own wallets instantly "
            "and for free. Go to Wallets → Transfer, select the source "
            "wallet, the destination wallet, the asset, and the amount. "
            "Confirm with 2FA and the transfer executes on the backend — no "
            "on-chain gas, no admin review. Would you like me to guide you "
            "through the process?"
        )
    if intent == "interest_rate_change":
        return (
            "No — your interest rate is fixed at loan origination and does "
            "not change during the term. The Murabaha structure we use "
            "locks the total cost at signing, so even if market rates move, "
            "your rate stays where it was. Would you like me to provide a "
            "fixed-rate quote for a new loan?"
        )
    if intent == "staking_network_fees":
        return (
            "We cover all network gas and transaction costs for staking — "
            "you pay zero network fees on stake, unstake, or reward claims. "
            "The APY shown on our Staking page is already NET of both our "
            "15% commission and the network fees, so what you see is what "
            "you earn. Would you like me to explain how our fee structure "
            "works for a specific network?"
        )
    if intent == "custody_minimum_balance":
        return (
            "No, there is no minimum balance for custody. You can hold any "
            "amount — large or small — in your segregated Fireblocks vault, "
            "with no monthly or per-asset fee. If your balance drops, "
            "nothing changes: your assets stay fully insured under our "
            "Lloyd's of London policy. The $100K minimum you may have seen "
            "applies to institutional staking / OTC, not to custody itself. "
            "Would you like me to explain our custody fee structure?"
        )
    if intent == "otc_counterparty":
        return (
            "Al-Fardan Q9 acts as the principal counterparty on every OTC "
            "trade — your agreement is with us directly, not with the third-"
            "party liquidity providers we source from in the background. "
            "That gives you a single point of contact, one settlement leg, "
            "and one legal entity on the other side of the deal. Would you "
            "like me to connect you with Layla for a firm quote?"
        )
    if intent == "api_key_management":
        # Flagged by James Set 11 Q10 — the self-service API-key UI
        # doesn't exist yet. Do NOT claim features we don't ship.
        return (
            "API keys are available only to institutional clients and are "
            "issued manually by our team — there is no self-service UI in "
            "the dashboard today. Please email institutional@alfardanq9.com "
            "with your account details and intended use case (read-only "
            "balances, trade execution, etc.) and we'll scope + issue a "
            "key with HMAC-SHA256 signing. Would you like me to connect "
            "you with our API support team?"
        )
    if intent == "staking_rewards_withdraw":
        return (
            "Yes, you can withdraw your staking rewards at any time. Go to "
            "Wallets → Withdraw, select the asset (ETH / SOL), enter the "
            "destination address and amount, and confirm with 2FA. Admin "
            "review completes within 1 hour during UAE business hours; "
            "Fireblocks broadcasts the transaction after approval. No "
            "lock-up beyond each network's standard unbonding (ETH ~27h, "
            "SOL ~2d) if you're withdrawing freshly-unstaked principal. "
            "Would you like me to help you submit a withdrawal?"
        )
    # ─── Set 6 replies ───────────────────────────────────────────
    if intent == "ltv_calculation":
        return (
            "LTV (Loan-to-Value) is calculated as (Loan Amount ÷ Collateral "
            "Value) × 100, and yes — it is based on the current market "
            "price of your collateral, updated in real time. Example: pledge "
            "1 BTC at $70,000 and borrow $35,000 → LTV is 50%. Our cap is "
            "75% (among the highest institutional), margin call triggers at "
            "85%, liquidation at 90%. Would you like me to calculate LTV "
            "for your specific collateral?"
        )
    if intent == "rewards_history":
        return (
            "Yes, you can see your full staking rewards history in the "
            "Staking module of your dashboard — Staking → Reward History. "
            "You can view daily, monthly, and yearly breakdowns per network, "
            "and export to CSV for your accountant. History goes back to "
            "the start of your first stake on each network. Would you like "
            "me to help you export your rewards history?"
        )
    # ─── Sets 7/8/9 replies ──────────────────────────────────────
    if intent == "claim_rewards":
        return (
            "No, there is no minimum amount to claim staking rewards. "
            "Rewards of any size — even fractional amounts — are eligible. "
            "They're also distributed automatically daily at 00:00 UTC to "
            "your segregated custodial account, so you don't have to claim "
            "them manually. Would you like me to help you check your "
            "available rewards?"
        )
    if intent == "interest_payment":
        return (
            "Interest is deducted automatically each month from your "
            "available custody balance — no manual action needed for the "
            "standard cycle. To make an ad-hoc payment, go to Lending → "
            "Active Loans → Pay Interest. Yes, you can pay in crypto — "
            "USDC, USDT, BTC, and ETH are accepted at the current spot "
            "rate. Would you like me to help you make an interest payment?"
        )
    if intent == "shared_account":
        return (
            "No, custody accounts are individual and can't be jointly "
            "owned. For family-office / corporate setups, we support sub-"
            "users under a single entity account — each with scoped "
            "permissions (view-only, trade, withdraw) so spouses / "
            "directors / finance staff can operate without giving anyone "
            "full control. Would you like me to explain how to set up "
            "sub-user access?"
        )
    if intent == "cancel_otc":
        return (
            "No, once you confirm an OTC trade it's binding and cannot be "
            "cancelled — we've already hedged with market makers on your "
            "acceptance. Before confirming you get a firm quote valid for "
            "15 minutes where the price is locked; use that window to "
            "double-check size and direction. Would you like me to walk "
            "you through the confirmation flow so nothing slips?"
        )
    if intent == "otc_max_size":
        return (
            "Our standard maximum OTC trade size is USD 50 million per "
            "ticket. For trades above $50M we can absolutely accommodate "
            "— please contact your relationship manager or the OTC desk "
            "24-48 hours in advance so we can pre-stage liquidity and "
            "give you a firm (not indicative) quote. Our largest single "
            "execution to date is over $50M, and we've done blocks up to "
            "several hundred million with advance notice. What size are "
            "you looking at?"
        )
    if intent == "tin_required":
        return (
            "Yes, for institutional / corporate accounts we require a tax "
            "identification number (TIN) or equivalent as part of KYC — "
            "the exact form depends on the entity's jurisdiction (EIN for "
            "US, trade licence for UAE, CRN for UK, etc.). For individual "
            "accounts your passport or Emirates ID number is sufficient. "
            "Would you like me to explain the full KYC document list for "
            "your account type?"
        )
    if intent == "add_collateral":
        return (
            "To add more collateral to an existing loan: go to Lending → "
            "Active Loans → click your loan → 'Add Collateral'. Choose "
            "the asset (BTC or ETH) and amount, confirm with 2FA, and the "
            "additional collateral pledges to the Fireblocks escrow "
            "immediately — your LTV drops instantly. No fee for top-ups. "
            "Would you like me to help you add collateral right now?"
        )
    if intent == "ip_whitelisting":
        return (
            "Yes, you can restrict logins to specific IP addresses. Go to "
            "Settings → Security → IP Whitelist, add the IPs or CIDR "
            "ranges you want to allow, and save with 2FA. From then on, "
            "sign-ins from any IP outside the list are blocked and you get "
            "an email alert. We also offer device-fingerprint anomaly "
            "detection even without IP whitelist enabled. Would you like "
            "me to help you set this up?"
        )
    if intent == "negotiate_rate":
        return (
            "Yes, interest rates are negotiable for larger loans. Our "
            "published tiers: Starter (<$1M) 4.9% APR, Core 4.5%, Pro "
            "4.2%, Institutional ($1M+, 5-year) 3.25-3.9%. For loans "
            "above $5M we run custom pricing through our credit committee "
            "— bespoke rates tied to collateral quality, loan purpose, "
            "and relationship depth. Would you like me to connect you "
            "with our lending desk to discuss your specific requirements?"
        )
    if intent == "loan_default":
        return (
            "If you don't repay at maturity we follow a staged process: "
            "(1) You receive reminders at 30, 14, and 7 days before "
            "maturity. (2) At maturity we attempt a collateral-level "
            "margin call via email + phone. (3) If no action within "
            "the cure window, we liquidate collateral through our OTC "
            "desk to recover the outstanding principal + accrued cost + "
            "a 1% default fee. (4) Any excess collateral returns to your "
            "custody vault; any shortfall is logged as a personal "
            "liability. Default also flags your future-borrowing "
            "eligibility with us. Would you like me to explain the "
            "liquidation mechanics in more detail?"
        )
    if intent == "validator_diversification":
        return (
            "Yes, your staked assets are distributed across multiple "
            "validators — we never concentrate more than 10% of any "
            "client's position with a single validator. This reduces "
            "correlated slashing risk and keeps uptime high if one "
            "validator has an issue. For clients staking above $5M per "
            "network we can discuss dedicated validator routing with "
            "your relationship manager. Would you like me to explain our "
            "validator selection methodology in more detail?"
        )
    if intent == "monthly_statements":
        return (
            "Yes, we provide monthly custody statements automatically. "
            "Go to Wallets → Statements, pick the month (or a custom "
            "range), and download as PDF or CSV. Each statement includes "
            "opening and closing balances per asset, all transactions for "
            "the period, staking rewards, fees, and our custody "
            "attestation stamp. Would you like me to help you download "
            "your latest statement?"
        )
    if intent == "liquidation_details":
        return (
            "Liquidation is PARTIAL — we never auto-liquidate your full "
            "collateral, only the minimum needed to bring LTV back to a "
            "safe level (60%). Execution happens through our OTC desk, "
            "not a retail exchange, which minimises slippage on large "
            "positions. A 1% liquidation fee applies to the liquidated "
            "portion. Example: on a $30K loan against 1 BTC at $42K "
            "(LTV 71%), if BTC falls further to trigger liquidation, we "
            "might sell ~0.15 BTC — you keep the rest plus any remaining "
            "equity after loan payoff. Would you like me to calculate a "
            "liquidation scenario for your position?"
        )
    # ─── Sets 11-14 replies ──────────────────────────────────────
    if intent == "partial_repayment":
        return (
            "Yes, you can make partial repayments at any time. There is "
            "no prepayment penalty — interest is calculated pro-rata on "
            "the outstanding balance, so each partial repayment reduces "
            "the remaining cost proportionally. Go to Lending → Active "
            "Loans → Repay, enter the amount, and the balance updates "
            "immediately. Would you like me to calculate how much "
            "interest you would save with a specific partial repayment?"
        )
    if intent == "hardware_wallet":
        return (
            "No, you cannot connect a Ledger or Trezor directly to your "
            "custody account. Our custody runs on Fireblocks MPC-CMP — "
            "keys are split into shards across multiple HSMs, so there "
            "isn't a single private key that a hardware wallet could "
            "pair with. However, you can withdraw your assets to your "
            "Ledger / Trezor at any time — the withdrawal address is "
            "the one your hardware wallet generates. Would you like me "
            "to walk you through withdrawing to a hardware wallet?"
        )
    if intent == "rewards_in_different_asset":
        return (
            "No, staking rewards are paid in-kind — ETH rewards in ETH, "
            "SOL rewards in SOL, etc. This is a protocol-level rule, "
            "not a policy of ours. However, you can swap your rewards "
            "for stablecoins (USDC / USDT) or any other supported asset "
            "via our OTC desk, or convert smaller amounts through our "
            "trading platform. Would you like me to help you swap your "
            "rewards into stablecoins?"
        )
    if intent == "missing_rewards":
        return (
            "Staking rewards are distributed daily at 00:00 UTC. If you "
            "don't see today's rewards by ~02:00 UTC, first check the "
            "Staking page — the reward counter updates as soon as the "
            "epoch closes. If they're still not showing after that, "
            "please email support@alfardanq9.com with your account "
            "email, the network (ETH / SOL / etc.), and the specific "
            "date — we investigate and credit any missed rewards plus "
            "a good-faith top-up for the inconvenience. Would you like "
            "me to help you check your recent rewards?"
        )
    if intent == "user_to_user_transfer":
        return (
            "No, Al-Fardan Q9 doesn't support direct user-to-user "
            "transfers on the platform — we're a custodian, not a "
            "payment network. If you need to send crypto to another "
            "Al-Fardan client, you'd withdraw to their external wallet "
            "address (they can find it on their Wallets → Deposit "
            "page) as a normal on-chain transfer. Network fees apply, "
            "same as any withdrawal. Would you like me to explain how "
            "to set up the withdrawal?"
        )
    if intent == "otc_after_hours":
        return (
            "No, standard OTC quotes only run during UAE business "
            "hours (Sunday-Thursday, 9am-6pm GST). For urgent trades "
            "outside those hours, email otc@alfardanq9.com or call our "
            "24/7 emergency line at +971 4 123 4568 — the on-call "
            "dealer can provide a firm quote within 15 minutes for "
            "qualifying tickets (≥ $5M). Would you like me to flag an "
            "out-of-hours request for you?"
        )
    if intent == "login_history":
        return (
            "Yes, you can review your login history. Go to Settings → "
            "Security → Login History. You'll see each recent login "
            "with date, time, device fingerprint, and approximate "
            "location. If any entry looks unfamiliar, tap 'Flag as "
            "suspicious' on that row and our security team will "
            "follow up within 2 UAE business hours. Would you like me "
            "to help you review your login history now?"
        )
    if intent == "interest_source":
        return (
            "Interest is deducted from your available custody balance "
            "— never from the pledged collateral. On the scheduled "
            "monthly date we debit the cost amount from your custody "
            "wallet (in USD, AED, or EUR as chosen at signing, or "
            "you can pre-elect stablecoin). If your balance is short, "
            "we'll notify you 5 days ahead so you can top up. Would "
            "you like me to explain how to set up auto-top-up from "
            "a bank transfer?"
        )
    if intent == "rewards_expiration":
        return (
            "No, staking rewards never expire. They accumulate to "
            "your staked principal (ETH auto-compounds) or sit as "
            "liquid rewards in your custody (SOL / ATOM / DOT), and "
            "you can claim or re-deploy them whenever you want — "
            "years later is fine. There's no sweep, no loss, no "
            "time-decay. Would you like me to help you check your "
            "accumulated rewards?"
        )
    if intent == "withdrawal_time":
        return (
            "Withdrawals to an external wallet typically complete in "
            "1-4 hours during UAE business hours (most are done "
            "within 1 hour). Breakdown: you submit → admin reviews "
            "(≤ 1 hour) → Fireblocks broadcasts → blockchain confirms "
            "(10-30 minutes for BTC, a few minutes for ETH, seconds "
            "for SOL). Off-hours withdrawals process the next "
            "business morning. Whitelisted addresses skip the admin "
            "queue and go in minutes. Would you like me to walk you "
            "through submitting a withdrawal?"
        )
    if intent == "active_sessions":
        return (
            "Yes, you can see every device that's signed into your "
            "account. Go to Settings → Security → Active Sessions — "
            "each row shows device, browser, approximate location, "
            "and last-active time. Tap 'Sign Out' on any row to "
            "terminate that session remotely, or 'Sign Out All' to "
            "end every session including the current one. Would you "
            "like me to help you review your active sessions now?"
        )
    if intent == "loan_transfer":
        return (
            "No, loans cannot be transferred to another person or "
            "entity — the agreement is specific to the original "
            "borrower and the collateral pledged. If someone else "
            "needs the liquidity, the cleanest path is: you repay "
            "early (no prepayment penalty, interest pro-rata on "
            "actual duration), your collateral releases, and they "
            "apply for their own loan against their own collateral. "
            "Would you like me to calculate your current payoff "
            "amount?"
        )
    if intent == "validator_locations":
        return (
            "Our validators run across three continents in Tier-IV "
            "data centres: Dubai (UAE), Singapore, and Frankfurt "
            "(Germany). Geographic + provider diversity means no "
            "single outage, jurisdiction change, or AZ failure takes "
            "validators offline together — we maintain 99.95% uptime "
            "SLA with automatic failover. Would you like me to share "
            "more detail on our infrastructure setup?"
        )
    if intent == "historical_snapshot":
        return (
            "Yes, you can generate a statement for any historical "
            "date or date range. Go to Wallets → Statements → Custom "
            "Range, pick the start and end dates, and download the "
            "PDF / CSV. The report includes opening + closing "
            "balances per asset, every transaction in the window, "
            "and the fiat valuation at end-of-day. Useful for audits, "
            "tax filings, and board reports. Would you like me to "
            "help you generate a statement for a specific date?"
        )
    # ─── Set 15 replies ──────────────────────────────────────────
    if intent == "otc_settlement_currency":
        return (
            "Yes, you can settle in a currency other than the quote "
            "currency. We support 8 settlement currencies: AED, SAR, "
            "KWD, BHD, QAR, OMR, USD, and EUR. Tell the dealer your "
            "preferred settlement currency when requesting the quote "
            "and we'll lock it in (cross-currency quotes use our "
            "daily spot at execution time). Would you like me to "
            "connect you with our OTC desk for a quote in your "
            "preferred currency?"
        )
    if intent == "temporary_deactivation":
        return (
            "No, we don't offer temporary deactivation of an account. "
            "Your options are: (1) keep it active with zero balance "
            "— no fees, no monthly charges, nothing happens until you "
            "come back; or (2) permanently close it via support "
            "(irreversible, see our account-closure policy for data "
            "retention). Most clients on a pause simply leave the "
            "account idle. Would you like me to explain the permanent "
            "closure process instead?"
        )
    if intent == "collateral_revaluation":
        return (
            "Your collateral is revalued in real time — every price "
            "tick from our live CoinGecko feed updates your LTV "
            "immediately in the dashboard. There's no daily cut-off "
            "or delayed refresh. The margin-call + liquidation "
            "triggers fire on the same live prices, so there's no "
            "stale-valuation risk either way. Would you like me to "
            "explain how LTV is calculated?"
        )
    if intent == "tax_withholding":
        return (
            "No, we don't withhold any tax from your staking rewards "
            "or OTC gains. Rewards are paid to you in full; tax "
            "reporting + payment is your responsibility based on "
            "your jurisdiction. We provide full transaction history "
            "(Settings → Statements, CSV + PDF) so you or your "
            "accountant has everything needed. Would you like me to "
            "help you export your rewards report?"
        )
    if intent == "account_recovery":
        return (
            "Account recovery process: (1) Email "
            "support@alfardanq9.com from ANY address (your account "
            "email is ideal but not required). (2) Our team sends a "
            "video-KYC booking link — show your government-issued ID "
            "on the call. (3) Once we verify your identity against "
            "your KYC records, we reset your 2FA and issue a "
            "password-reset link. End-to-end typically 24-48 hours "
            "for standard cases, same-day during UAE business hours. "
            "Would you like me to connect you with support to start "
            "the recovery process now?"
        )
    if intent == "session_timeout":
        return (
            "Your session times out after 30 minutes of inactivity — for "
            "security reasons this is fixed and cannot be extended per-"
            "user. JWTs themselves expire after 7 days (you'll be re-asked "
            "to sign in then even with activity). If you need a longer-"
            "lived connection for programmatic access, use an API key from "
            "Settings → API Keys instead of a browser session. Would you "
            "like me to help you set up an API key?"
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
    # Migrated intents' buttons come from their KB entry's `actions`
    # field (main.py's `_actions_for_entry` picks those up). Returning
    # None here means no scripted-action override — retrieval wins.
    if intent in _MIGRATED_TO_KB:
        return None

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
    # Set 4
    "interest_calculation": "intent_interest_calculation",
    "whitelist_address": "intent_whitelist_address",
    "account_closure": "intent_account_closure",
    "early_repayment": "intent_early_repayment",
    "validator_choice": "intent_validator_choice",
    "security_incident": "intent_security_incident",
    # Set 5
    "loan_extension": "intent_loan_extension",
    "internal_transfer": "intent_internal_transfer",
    "interest_rate_change": "intent_interest_rate_change",
    "staking_network_fees": "intent_staking_network_fees",
    "custody_minimum_balance": "intent_custody_minimum_balance",
    "otc_counterparty": "intent_otc_counterparty",
    "api_key_management": "intent_api_key_management",
    "staking_rewards_withdraw": "intent_staking_rewards_withdraw",
    # Set 6
    "ltv_calculation": "intent_ltv_calculation",
    "rewards_history": "intent_rewards_history",
    "session_timeout": "intent_session_timeout",
    # Sets 7/8/9
    "claim_rewards": "intent_claim_rewards",
    "interest_payment": "intent_interest_payment",
    "shared_account": "intent_shared_account",
    "cancel_otc": "intent_cancel_otc",
    "otc_max_size": "intent_otc_max_size",
    "tin_required": "intent_tin_required",
    "add_collateral": "intent_add_collateral",
    "ip_whitelisting": "intent_ip_whitelisting",
    "negotiate_rate": "intent_negotiate_rate",
    "loan_default": "intent_loan_default",
    "validator_diversification": "intent_validator_diversification",
    "monthly_statements": "intent_monthly_statements",
    "liquidation_details": "intent_liquidation_details",
    # Sets 11/12/13/14
    "partial_repayment": "intent_partial_repayment",
    "hardware_wallet": "intent_hardware_wallet",
    "rewards_in_different_asset": "intent_rewards_in_different_asset",
    "missing_rewards": "intent_missing_rewards",
    "user_to_user_transfer": "intent_user_to_user_transfer",
    "otc_after_hours": "intent_otc_after_hours",
    "login_history": "intent_login_history",
    "interest_source": "intent_interest_source",
    "rewards_expiration": "intent_rewards_expiration",
    "withdrawal_time": "intent_withdrawal_time",
    "active_sessions": "intent_active_sessions",
    "loan_transfer": "intent_loan_transfer",
    "validator_locations": "intent_validator_locations",
    "historical_snapshot": "intent_historical_snapshot",
    # Set 15
    "otc_settlement_currency": "intent_otc_settlement_currency",
    "temporary_deactivation": "intent_temporary_deactivation",
    "collateral_revaluation": "intent_collateral_revaluation",
    "tax_withholding": "intent_tax_withholding",
    "account_recovery": "intent_account_recovery",
    # Short yes/no — map to generic intent types so the widget can
    # still render actions the usual way.
    "affirmation": "intent_affirmation",
    "negation": "intent_negation",
}


def match_type_for(intent: Intent) -> str:
    """Return the MatchType literal for a scripted-reply intent, or
    'kb_hit' for intents that fall through to retrieval."""
    return _INTENT_TO_MATCH_TYPE.get(intent, "kb_hit")
