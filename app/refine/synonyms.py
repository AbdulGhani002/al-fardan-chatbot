"""Domain-specific synonym expansion for retrieval recall.

Users type "borrow" but the matching KB entry talks about "loan".
Users type "stake" but the entry talks about "yield". Users type
"safe" but the entry uses "secure" and "insured".

We expand the user's query with known synonyms BEFORE it hits the
retriever. More matching surface → better recall, without any extra
data. Stays zero-ML.

Two public functions:

  expand(text) -> str
    Append domain synonyms to the text. "I want to borrow" becomes
    "I want to borrow loan credit lending financing". The original
    text is preserved first, so semantic retrieval keeps its focus
    on the user's actual phrasing; the appended synonyms boost recall
    when the KB uses different wording.

  expansion_score(text) -> int
    How many synonym groups the text touched — useful for telemetry
    and deciding whether to use the expanded query.
"""

from __future__ import annotations

import re


# Synonym groups — keys are one representative term. When ANY key's
# trigger pattern matches, ALL of the group's synonyms get appended.
# Keep groups tight — adding unrelated words pollutes retrieval.

_GROUPS: list[tuple[str, tuple[str, ...]]] = [
    # lending / borrowing
    (r"\b(borrow|loan|credit|lending|financ|murabaha|pledge)",
     ("borrow", "loan", "credit", "lending", "financing", "murabaha", "pledge")),
    # staking
    (r"\b(stake|staking|yield|apy|rewards?|validator|delegate)",
     ("stake", "staking", "yield", "apy", "rewards", "validator", "delegate")),
    # custody / storage
    (r"\b(custody|custodial|storage|cold\s+storage|fireblocks|vault|mpc)",
     ("custody", "storage", "vault", "fireblocks", "mpc", "cold")),
    # safety / protection
    (r"\b(safe|safety|secur(e|ity)|protect(ed|ion)?|insured|insurance)",
     ("safe", "secure", "security", "protected", "protection",
      "insured", "insurance", "lloyds")),
    # withdrawal
    (r"\b(withdraw|withdrawal|send\s+out|cash\s+out|pull\s+out|take\s+out)",
     ("withdraw", "withdrawal", "exit", "transfer", "send")),
    # deposit / fund
    (r"\b(deposit|fund|top\s+up|add\s+(crypto|funds)|transfer\s+in)",
     ("deposit", "fund", "topup", "transfer")),
    # onboarding
    (r"\b(sign\s*up|register|open\s+(an?\s+)?account|onboard|kyc|verify)",
     ("signup", "register", "open account", "onboard", "kyc", "verify")),
    # shariah
    (r"\b(sharia|shariah|halal|islamic|riba|murabaha|aaoifi)",
     ("sharia", "shariah", "halal", "islamic", "murabaha", "aaoifi", "riba")),
    # regulation
    (r"\b(regulat|licens|compliant|difc|vara|cbuae|fsra|legal)",
     ("regulated", "licensed", "compliant", "vara", "difc", "cbuae", "legal")),
    # OTC / trading
    (r"\b(otc|trade|trading|buy|sell|block\s+trade|execution)",
     ("otc", "trade", "trading", "execution", "buy", "sell", "block")),
    # fees / cost / pricing
    (r"\b(fee|cost|price|spread|commission|rate|apr|charges?)",
     ("fee", "cost", "price", "spread", "commission", "rate", "apr")),
    # portfolio / balance / holdings
    (r"\b(portfolio|balance|holdings?|my\s+assets?|my\s+wallet|my\s+account)",
     ("portfolio", "balance", "holdings", "assets", "wallet", "account")),
    # minimum thresholds
    (r"\b(minimum|min\b|threshold|enough|qualify|floor)",
     ("minimum", "threshold", "floor", "qualify", "eligibility")),
    # family office / institutional
    (r"\b(family\s+office|institution|hedge\s+fund|fund\s+manager|sovereign|endowment)",
     ("institutional", "family office", "fund", "hedge", "sovereign", "endowment")),
    # comparisons
    (r"\b(vs|versus|compared?\s+to|better\s+than|different\s+from|instead\s+of)",
     ("vs", "versus", "compared", "comparison", "alternative")),
]

_COMPILED_TRIGGERS: list[tuple[re.Pattern, tuple[str, ...]]] = [
    (re.compile(pat, re.I), syns) for pat, syns in _GROUPS
]


def expand(text: str) -> str:
    """Append relevant synonym groups to `text`. Original text comes
    first — retrieval should still weight it heavier."""
    if not text:
        return text
    extras: list[str] = []
    seen: set[str] = set()
    for pattern, syns in _COMPILED_TRIGGERS:
        if pattern.search(text):
            for s in syns:
                if s not in seen:
                    extras.append(s)
                    seen.add(s)
    if not extras:
        return text
    return f"{text} {' '.join(extras)}"


def expansion_score(text: str) -> int:
    if not text:
        return 0
    return sum(1 for pattern, _ in _COMPILED_TRIGGERS if pattern.search(text))
