"""Entity extraction from user queries.

Zero ML. Pure regex + ordered-pattern matching. Returns a structured
dict the composer can act on:

    {
      "amounts": [
        {"raw": "$5,000", "value": 5000.0, "currency": "USD"},
        {"raw": "0.3 BTC", "value": 0.3, "currency": "BTC"},
      ],
      "assets": ["BTC"],
      "actions": ["borrow"],
      "question_type": "amount_check" | "comparison" | "crisis" | ...,
      "affirmative": False,
      "beginner": True,   # user self-identified as new
    }

Extractors run in order. Each sets optional fields on the result
dict; missing fields default to safe null/empty.
"""

from __future__ import annotations

import re
from typing import Any


# ─── Patterns ─────────────────────────────────────────────────────────

# Fiat currency-BEFORE-number: "AED 50,000", "EUR 1,000", "SAR 500k"
_CURRENCY_FIRST_RE = re.compile(
    r"""
    \b
    (?P<curr>usd|aed|eur|gbp|bhd|sar|kwd|qar|omr)
    \s+
    (?P<num>\d{1,3}(?:[,\s]?\d{3})*(?:\.\d+)?)
    \s*
    (?P<mult>k|m|million|thousand|b|billion)?
    \b
    """,
    re.I | re.VERBOSE,
)

# USD-ish: $1,000  $5k  $10 million  5,000 dollars  5k USD  100k  $1M
_USD_RE = re.compile(
    r"""
    \$?\s*
    (?P<num>\d{1,3}(?:[,\s]?\d{3})*(?:\.\d+)?)
    \s*
    (?P<mult>k|m|million|thousand|b|billion)?
    \s*
    (?P<curr>usd|aed|eur|gbp|dollars?|dirhams?|euros?|bhd|sar|kwd|qar|omr)?
    \b
    """,
    re.I | re.VERBOSE,
)

# Crypto amount: 0.3 BTC  10 ETH  1,000 SOL  0.5BTC
_CRYPTO_RE = re.compile(
    r"""
    (?P<num>\d{1,5}(?:[,\s]?\d{3})*(?:\.\d+)?)
    \s*
    (?P<asset>btc|bitcoin|eth|ethereum|sol|solana|usdt|usdc|bnb|xrp|ada|avax|dot|matic|atom|link)
    \b
    """,
    re.I | re.VERBOSE,
)

# Percent: 75%  5.2 percent
_PERCENT_RE = re.compile(
    r"(?P<num>\d{1,3}(?:\.\d+)?)\s*(?:%|percent|pct)\b", re.I
)

# Asset-only mention (no amount): "stake ETH"
_ASSET_ONLY_RE = re.compile(
    r"\b(btc|bitcoin|eth|ethereum|sol|solana|usdt|usdc|bnb|xrp|ada|avax|dot|matic|atom|link)\b",
    re.I,
)

# Actions (action verbs — what they want to do)
_ACTION_WORDS: dict[str, tuple[str, ...]] = {
    "borrow": ("borrow", "loan", "take out", "take a loan"),
    "stake": ("stake", "staking", "delegate"),
    "buy": ("buy", "purchase", "acquire", "get some"),
    "sell": ("sell", "liquidate", "dispose"),
    "deposit": ("deposit", "fund", "send in", "add crypto", "top up"),
    "withdraw": ("withdraw", "send out", "pull out", "cash out"),
    "open_account": ("open an account", "sign up", "register", "create account"),
    "verify": ("verify", "kyc", "prove identity"),
    "trade": ("trade", "execute", "otc"),
    "earn": ("earn", "yield", "returns", "rewards"),
    "protect": ("protect", "protected", "safe", "safety", "secure", "security"),
    "compare": ("vs", "versus", "compared to", "better than", "difference"),
    "start": ("start", "begin", "how to start", "get started"),
}

# Sentiment / self-description
_BEGINNER_RE = re.compile(
    r"\b(new\s+to\s+crypto|first\s+time|noob|never\s+used|"
    r"don't\s+know|dont\s+know|no\s+idea|beginner|starting\s+out)\b",
    re.I,
)
_CONCERN_RE = re.compile(
    r"\b(scared|worried|afraid|anxious|concerned|nervous|safe\?|trust)\b",
    re.I,
)
_CRISIS_RE = re.compile(
    r"\b("
    r"bank\s+run|market\s+crash|what\s+if|bankrupt|insolven|"
    r"hack(ed)?|go\s+down|outage|disaster|scam|steal|fraud|"
    r"lose\s+(everything|all|my\s+(money|crypto))|rug\s+pull|"
    r"shut\s*down|freeze|seize"
    r")\b",
    re.I,
)

# Multipliers
_MULTIPLIERS = {
    "k": 1_000, "thousand": 1_000,
    "m": 1_000_000, "million": 1_000_000,
    "b": 1_000_000_000, "billion": 1_000_000_000,
}

# Normalise currency tokens
_CURRENCY_NORM = {
    "usd": "USD", "dollar": "USD", "dollars": "USD",
    "aed": "AED", "dirham": "AED", "dirhams": "AED",
    "eur": "EUR", "euro": "EUR", "euros": "EUR",
    "gbp": "GBP",
    "bhd": "BHD", "sar": "SAR", "kwd": "KWD", "qar": "QAR", "omr": "OMR",
}

# Assets — canonical symbols
_ASSET_NORM = {
    "btc": "BTC", "bitcoin": "BTC",
    "eth": "ETH", "ethereum": "ETH",
    "sol": "SOL", "solana": "SOL",
    "usdt": "USDT", "usdc": "USDC",
    "bnb": "BNB", "xrp": "XRP", "ada": "ADA",
    "avax": "AVAX", "dot": "DOT", "matic": "MATIC",
    "atom": "ATOM", "link": "LINK",
}


# ─── Extractor ───────────────────────────────────────────────────────

def _parse_number(raw: str) -> float:
    """Handle '1,000', '1 000', '5k' → 1000 / 5000 etc."""
    s = raw.replace(",", "").replace(" ", "")
    return float(s)


def extract_entities(text: str) -> dict[str, Any]:
    """Return a structured view of what the user typed."""
    result: dict[str, Any] = {
        "text": text,
        "amounts": [],     # [{raw, value, currency|asset}]
        "assets": [],      # ["BTC", "ETH"] unique, order of appearance
        "actions": [],     # ["borrow", "stake", ...]
        "percents": [],    # [75.0, 5.2]
        "beginner": bool(_BEGINNER_RE.search(text)),
        "concerned": bool(_CONCERN_RE.search(text)),
        "crisis": bool(_CRISIS_RE.search(text)),
    }

    # Crypto amounts come first so we don't double-count "5 BTC" as "$5"
    seen_spans: list[tuple[int, int]] = []
    for m in _CRYPTO_RE.finditer(text):
        try:
            v = _parse_number(m.group("num"))
        except Exception:  # noqa: BLE001
            continue
        asset = _ASSET_NORM.get(m.group("asset").lower(), m.group("asset").upper())
        result["amounts"].append(
            {"raw": m.group(0).strip(), "value": v, "currency": asset}
        )
        seen_spans.append(m.span())
        if asset not in result["assets"]:
            result["assets"].append(asset)

    # Currency-before-number matches ("AED 50,000")
    for m in _CURRENCY_FIRST_RE.finditer(text):
        try:
            v = _parse_number(m.group("num"))
        except Exception:  # noqa: BLE001
            continue
        mult = m.group("mult")
        if mult:
            v *= _MULTIPLIERS.get(mult.lower(), 1)
        curr = _CURRENCY_NORM.get(m.group("curr").lower(), m.group("curr").upper())
        result["amounts"].append(
            {"raw": m.group(0).strip(), "value": v, "currency": curr}
        )
        seen_spans.append(m.span())

    # USD / fiat amounts — skip spans already consumed by crypto-amount regex
    for m in _USD_RE.finditer(text):
        if any(s <= m.start() < e for s, e in seen_spans):
            continue
        raw_num = m.group("num")
        try:
            v = _parse_number(raw_num)
        except Exception:  # noqa: BLE001
            continue
        mult = m.group("mult")
        if mult:
            v *= _MULTIPLIERS.get(mult.lower(), 1)
        curr_raw = (m.group("curr") or "USD").lower()
        curr = _CURRENCY_NORM.get(curr_raw, curr_raw.upper())
        # Dollar-sign in the original OR explicit currency token required
        # so that '3 layers' doesn't become $3.
        had_dollar = "$" in m.group(0) or m.group("curr") is not None
        if not had_dollar and not mult:
            continue
        result["amounts"].append(
            {"raw": m.group(0).strip(), "value": v, "currency": curr}
        )

    # Percents
    for m in _PERCENT_RE.finditer(text):
        try:
            result["percents"].append(float(m.group("num")))
        except Exception:  # noqa: BLE001
            pass

    # Asset-only mentions (even if no amount)
    for m in _ASSET_ONLY_RE.finditer(text):
        asset = _ASSET_NORM.get(m.group(0).lower(), m.group(0).upper())
        if asset not in result["assets"]:
            result["assets"].append(asset)

    # Actions
    lower = text.lower()
    for action, phrases in _ACTION_WORDS.items():
        if any(p in lower for p in phrases):
            result["actions"].append(action)

    return result
