"""Typo correction for common crypto/finance terms.

We fix about 80 frequently-misspelled terms BEFORE the intent
classifier and retriever see the query. This dramatically improves
match quality for users who type casually — and James's QA runs
have shown typos like "lonas", "etherium", "stakign" cost us real
hits.

Simple word-boundary replacement. Case-preserving for the first
letter (so "Etherium" → "Ethereum", "lonas" → "loans").

Not using a general-purpose spellchecker (like SymSpell) because the
vocabulary is tiny + domain-specific, and we don't want to "correct"
valid words that look like typos. Dictionary is the safest.
"""

from __future__ import annotations

import re


# ─── Common misspellings dictionary ─────────────────────────────────
# Left side: the typo (lowercased). Right side: the correction.
# Keep additions short + specific. Don't add pairs where the typo
# is also a valid English word.

_TYPO_MAP: dict[str, str] = {
    # Assets
    "btcoin": "bitcoin",
    "biticoin": "bitcoin",
    "bitocin": "bitcoin",
    "bitcion": "bitcoin",
    "bictoin": "bitcoin",
    "etherium": "ethereum",
    "etheruem": "ethereum",
    "ehtereum": "ethereum",
    "solanaa": "solana",
    "slana": "solana",
    "tethr": "tether",
    "teher": "tether",
    "polkadog": "polkadot",
    "avaacs": "avalanche",
    "avalance": "avalanche",

    # Actions
    "stak": "stake",
    "stakign": "staking",
    "stacking": "staking",   # common muscle-memory typo (not the pallet kind)
    "stke": "stake",
    "stkae": "stake",
    "witdraw": "withdraw",
    "withdarw": "withdraw",
    "withdraws": "withdraw",
    "wothdraw": "withdraw",
    "depsoit": "deposit",
    "deopsit": "deposit",
    "borow": "borrow",
    "borrrow": "borrow",
    "lend": "lending",

    # Nouns
    "lonas": "loans",
    "lones": "loans",
    "lons": "loans",
    "credti": "credit",
    "acount": "account",
    "accoount": "account",
    "acconut": "account",
    "balnce": "balance",
    "balnace": "balance",
    "porfolio": "portfolio",
    "portolfio": "portfolio",
    "custdy": "custody",
    "custoddy": "custody",
    "wallt": "wallet",
    "walet": "wallet",
    "depsit": "deposit",
    "staing": "staking",

    # Platform
    "cryoto": "crypto",
    "crytpo": "crypto",
    "crypo": "crypto",
    "cryto": "crypto",
    "corpto": "crypto",
    "blokchain": "blockchain",
    "blckchain": "blockchain",

    # Regulatory / products
    "maraba": "murabaha",
    "murabha": "murabaha",
    "sharia": "shariah",   # either spelling is fine, normalise
    "shari": "shariah",
    "halal?": "halal",
    "fireblock": "fireblocks",
    "lloyd": "lloyd's",
    "lloyds": "lloyd's",

    # Finance
    "aprs": "apr",
    "intereset": "interest",
    "interst": "interest",
    "insurace": "insurance",
    "insurence": "insurance",
    "polcy": "policy",
    "coverag": "coverage",
    "covrage": "coverage",

    # Common English typos that matter here
    "wht": "what",
    "waht": "what",
    "hte": "the",
    "teh": "the",
    "adn": "and",
    "nad": "and",
    "yoru": "your",
    "ot": "to",
    "ot ": "to ",  # leave trailing space so 'ot' inside 'cotton' is safe — word-boundary pattern does the rest
}


# Pre-compile one big alternation regex for speed. Matches the typos
# as whole words only (case-insensitive) and uses a capture group so
# we can preserve the first-letter case of the original.
_TYPO_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in sorted(_TYPO_MAP.keys(), key=len, reverse=True)) + r")\b",
    re.I,
)


def _preserve_case(original: str, replacement: str) -> str:
    """Apply the case style of `original` to `replacement`."""
    if original.isupper():
        return replacement.upper()
    if original[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def correct_typos(text: str) -> str:
    """Return `text` with known crypto-finance typos fixed.

    No-op for text without a known typo — cheap regex + dict lookup,
    safe to call on every inbound message.
    """
    if not text:
        return text

    def _repl(m: re.Match) -> str:
        original = m.group(1)
        replacement = _TYPO_MAP.get(original.lower())
        if replacement is None:
            return original
        return _preserve_case(original, replacement)

    return _TYPO_RE.sub(_repl, text)


def preview_corrections(text: str) -> dict[str, str]:
    """Handy for debug endpoints / tests — show every correction made."""
    out: dict[str, str] = {}
    for m in _TYPO_RE.finditer(text):
        original = m.group(1)
        replacement = _TYPO_MAP.get(original.lower())
        if replacement and replacement.lower() != original.lower():
            out[original] = _preserve_case(original, replacement)
    return out
