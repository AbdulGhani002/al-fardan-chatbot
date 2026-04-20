"""Template-based response composer — the no-LLM answer generator.

Also handles CLARIFICATION: if the user mentions an action (borrow,
stake, withdraw) but leaves out the critical info (amount, asset),
the composer asks back instead of guessing. Good bots ask good
questions.

Takes an entity dict (from entities.extract_entities) and synthesises
a Safiya-voiced reply from the structured facts in facts.py.

Return value: (reply_text, action_list) OR None if the composer isn't
confident — caller falls back to retrieval in that case. Keeps
responsibility focused: the composer only fires on patterns it can
handle well. Everything else flows to the semantic retriever as before.

Patterns covered today:
  1. Fiat amount + signup/minimum context   → tier routing
  2. Crypto amount + borrow/loan context    → pledge-minimum check
  3. Crypto amount + stake context          → staking-minimum check
  4. Crypto amount + no action              → generic "what can I do with X?"
  5. Crisis/safety + no specific entity     → protection framing
"""

from __future__ import annotations

from typing import Optional

from . import facts as F


# ─── Reusable follow-up prompts — institutional register ──────────────
# No retail CTAs. The chatbot does not solicit engagement. When a
# follow-up is appended, it names a concrete next step the client can
# take if they wish; otherwise the answer stands alone.
_NEXT_STEP_ASKS = {
    "ask_starter_or_topup": "Please indicate whether you would prefer Starter Access at the current allocation or to increase to the individual tier.",
    "ask_firm_quote":       "A firm quote can be provided by our OTC desk on request.",
    "ask_walk_through":     "A relationship manager can guide the onboarding steps.",
    "ask_specific_concern": "",
    "ask_schedule_call":    "A call with the relationship manager can be arranged on request.",
}


def _format_btc_loan_capacity(btc: float, ltv_percent: int = 75) -> str:
    usd = F.estimate_loan_capacity_usd("BTC", btc, ltv_percent)
    return F.fmt_usd(usd)


def _format_eth_loan_capacity(eth: float, ltv_percent: int = 75) -> str:
    usd = F.estimate_loan_capacity_usd("ETH", eth, ltv_percent)
    return F.fmt_usd(usd)


# ─── Template: fiat amount + signup context ────────────────────────

def _compose_fiat_tier_check(amount_usd: float) -> Optional[str]:
    """User mentioned a fiat amount and is asking about opening an
    account or minimums. Route them to the right tier."""
    starter_min = F.MINIMUMS["starter_usd"]
    indiv_min = F.MINIMUMS["individual_usd"]
    inst_min = F.MINIMUMS["institutional_usd"]
    fmt = F.fmt_usd

    if amount_usd < starter_min:
        return (
            f"{fmt(amount_usd)} is below our Starter Access floor of "
            f"{fmt(starter_min)}. For smaller amounts a retail exchange is "
            f"more economic. The relationship manager can discuss options if you intend to scale."
        )
    if amount_usd < indiv_min:
        short = indiv_min - amount_usd
        return (
            f"{fmt(amount_usd)} qualifies for Starter Access "
            f"({fmt(starter_min)}–{fmt(indiv_min - 1)}). "
            f"You are {fmt(short)} from the individual tier."
        )
    if amount_usd < inst_min:
        short = inst_min - amount_usd
        return (
            f"{fmt(amount_usd)} is in the individual tier "
            f"(minimum {fmt(indiv_min)}). {fmt(short)} away from full "
            f"institutional ({fmt(inst_min)})."
        )
    if amount_usd < 1_000_000:
        return (
            f"{fmt(amount_usd)} qualifies for institutional access — "
            f"dedicated relationship manager, tighter OTC spreads, priority onboarding."
        )
    if amount_usd < 5_000_000:
        return (
            f"{fmt(amount_usd)} places you in the institutional tier — "
            f"enhanced staking rates and VIP OTC coverage. A call with "
            f"Layla Al-Fayez (Head of OTC) or Khalid Al Fardan (CEO) can be arranged on request."
        )
    if amount_usd < 25_000_000:
        return (
            f"{fmt(amount_usd)} qualifies for Institutional tier with "
            f"custom terms — dedicated validator and bespoke OTC spreads. "
            f"A tailored proposal can be prepared by the relationship manager."
        )
    return (
        f"{fmt(amount_usd)} is Sovereign-tier territory. +0.6% enhanced "
        f"staking rates, direct desk access, and bespoke structuring. "
        f"An introduction to Khalid Al Fardan (CEO) can be arranged on request."
    )


# ─── Template: crypto amount + borrow ──────────────────────────────

def _compose_btc_borrow(btc: float) -> str:
    pledge_min = F.pledge_minimum("BTC") or F.MINIMUMS["pledge_btc"]
    if btc < pledge_min:
        diff = pledge_min - btc
        return (
            f"{btc:g} BTC is below the {pledge_min:g} BTC minimum pledge. "
            f"Shortfall: {diff:g} BTC. Starter Access may accommodate smaller positions."
        )
    cap = _format_btc_loan_capacity(btc, 75)
    cap_safe = _format_btc_loan_capacity(btc, 50)
    rate = F.LENDING_RATES[1]
    return (
        f"With {btc:g} BTC as collateral at the 75% LTV maximum, "
        f"indicative borrow capacity is up to {cap}. At 50% LTV: {cap_safe}. "
        f"Rates from {rate:.2f}% APR for 1-year. Firm quote available from the desk."
    )


def _compose_eth_borrow(eth: float) -> str:
    pledge_min = F.pledge_minimum("ETH") or F.MINIMUMS["pledge_eth"]
    if eth < pledge_min:
        diff = pledge_min - eth
        return (
            f"{eth:g} ETH is below the {pledge_min:g} ETH minimum pledge. "
            f"Shortfall: {diff:g} ETH. Starter Access may accommodate smaller positions."
        )
    cap = _format_eth_loan_capacity(eth, 75)
    cap_safe = _format_eth_loan_capacity(eth, 50)
    return (
        f"With {eth:g} ETH as collateral at 75% LTV, indicative borrow "
        f"capacity is up to {cap}. At 50% LTV: {cap_safe}. No prepayment "
        f"penalty on any term. Firm quote available from the desk."
    )


# ─── Template: crypto amount + stake ───────────────────────────────

def _compose_crypto_stake(asset: str, amount: float) -> Optional[str]:
    apy = F.STAKING_APY.get(asset)
    if apy is None:
        # BTC can't stake
        if asset == "BTC":
            return (
                "Bitcoin does not support staking — it is proof-of-work. "
                "BTC can be used as loan collateral (up to 75% LTV) and the "
                "borrowed fiat deployed into a yield strategy. "
                "The relationship manager can structure this approach."
            )
        return None
    unbonding = F.STAKING_UNBONDING_DAYS.get(asset, 0)
    price = F.REFERENCE_PRICES_USD.get(asset, 0)
    usd_value = amount * price
    stake_min = F.MINIMUMS["stake_per_network_usd"]
    if usd_value < stake_min:
        short = stake_min - usd_value
        return (
            f"{amount:g} {asset} (~{F.fmt_usd(usd_value)}) is below the "
            f"{F.fmt_usd(stake_min)} per-network staking minimum. "
            f"Shortfall: ~{F.fmt_usd(short)}. "
            f"Starter Access may accommodate smaller positions."
        )
    # Above minimum — give a real projection
    yearly = amount * (apy / 100)
    unbonding_note = (
        f"no lock-up" if unbonding == 0 else f"{unbonding:g}-day unbonding"
    )
    return (
        f"At {apy:.1f}% APY on {amount:g} {asset}, indicative annual "
        f"reward is {yearly:g} {asset} (~{F.fmt_usd(yearly * price)}), "
        f"paid daily. {unbonding_note.capitalize()}. Rates vary with network conditions."
    )


# ─── Template: crisis / safety ──────────────────────────────────────

def _compose_crisis(text: str) -> str:
    ins = F.INSURANCE
    covers = ", ".join(ins["covers"])
    return (
        f"Client assets are segregated in {F.CUSTODY['provider']} vaults — "
        f"legally separate from our balance sheet and not rehypothecated. "
        f"An {F.fmt_usd(ins['amount_usd'])} {ins['underwriter']} policy "
        f"covers {covers}, subject to policy terms and exclusions. "
        f"Market price movements remain the client's own risk."
    )


# ─── Clarification: ask back when key info is missing ─────────────

import re as _re

# Question-form detector — a message is a QUESTION (not an intent-to-do)
# when it either ends with "?" or opens with a question word. Questions
# should flow through to the semantic retriever, NOT into the "ask for
# amount+asset" clarification branches below (those are meant only for
# users who EXPRESS an intent to do an action).
_QUESTION_OPENER_RE = _re.compile(
    r"^\s*("
    r"can|could|will|would|should|shall|may|might|must|do|does|did|is|"
    r"are|was|were|am|have|has|had|what|whats|when|whens|where|wheres|"
    r"which|who|whos|whose|whom|why|whys|how|hows|"
    # Also treat leading "please" / "help me understand" / "tell me"
    # as question-form so "tell me how staking works" doesn't get
    # mis-clarified.
    r"please\s+(tell|explain|help)|tell\s+me|explain|help\s+me\s+understand"
    r")\b",
    _re.I,
)

# A question opener anywhere in the sentence (after a greeting prefix).
# Catches "Hi, what is staking" — the literal "Hi," prefix previously
# hid the "what is" signal from the start-anchored _QUESTION_OPENER_RE
# and the composer fired clarification on the word "staking".
_QUESTION_SIGNAL_RE = _re.compile(
    r"\b("
    r"what\s+(is|are|does|do|was|were)|"
    r"how\s+(do|does|is|are|can|could|long|much|many)|"
    r"why\s+(is|are|do|does|would|should)|"
    r"when\s+(is|are|was|were|did|do|does|can|will)|"
    r"where\s+(is|are|can|do|does)|"
    r"who\s+(is|are|do|does)|"
    r"which\s+(is|are|one|ones|of)|"
    r"can\s+(you|i|we)|do\s+(you|i|we)|does\s+(the|your|al|q9)|"
    r"tell\s+me|explain\s+|please\s+(tell|explain|help)"
    r")\b",
    _re.I,
)

# Greeting/filler prefixes to strip before question-form detection.
# Catches "Hi, hello hey what is staking" without requiring the
# question opener to be the very first token.
_GREETING_PREFIX_RE = _re.compile(
    r"^\s*("
    r"hi|hello|hey|yo|greetings|assalamu\s+alaikum|salaam|salam|"
    r"good\s+(morning|afternoon|evening|day)|"
    r"okay|ok|alright|so|umm|uhh|well|actually|sorry|excuse\s+me"
    r")"
    r"[\s,.\-:!]+",
    _re.I,
)

# Intent-to-do markers. Even inside a longer message these indicate the
# user WANTS to do the thing right now, not ask about it. If we see
# these + an action word, the clarification branch is appropriate.
_INTENT_TO_DO_RE = _re.compile(
    r"\b("
    r"i\s+(want|wanna|need|would\s+like|wish|plan)\s+to|"
    r"i'?d\s+like\s+to|"
    r"let\s+me|help\s+me|let's|lets|"
    r"please\s+(help|let|allow)\s+me|"
    r"take\s+me\s+to|go\s+to|i'?m\s+(ready|looking)\s+to"
    r")\b",
    _re.I,
)


def _strip_greeting_prefix(text: str) -> str:
    """Remove leading greetings/fillers so ``Hi, what is staking`` is
    seen as ``what is staking`` by the question-form detector."""
    stripped = text
    # Strip up to two layers (e.g. "hi, hello, what is X")
    for _ in range(2):
        m = _GREETING_PREFIX_RE.match(stripped)
        if not m:
            break
        stripped = stripped[m.end():]
    return stripped.strip() or text


def _is_question_form(text: str) -> bool:
    """True if the message reads as a question rather than an
    intent-to-do statement. Questions should flow to retrieval.

    Logic:
      - Explicit intent-to-do markers ("I want to...", "let me...")
        trump everything — even if the message ends in "?" we treat
        it as action ("I want to stake, is that possible?").
      - Otherwise: ends in "?", or opens with a question word
        (after stripping greeting prefixes), or contains an obvious
        question signal anywhere ("what is staking" inside a longer
        greeting + question compound).
    """
    if not text:
        return False
    if _INTENT_TO_DO_RE.search(text):
        return False
    if text.rstrip().endswith("?"):
        return True
    cleaned = _strip_greeting_prefix(text)
    if _QUESTION_OPENER_RE.match(cleaned):
        return True
    if _QUESTION_SIGNAL_RE.search(text):
        return True
    return False


def _compose_clarification(entities: dict) -> Optional[tuple[str, list[dict]]]:
    """If the user stated an intent but left out the critical info,
    ask for it instead of guessing.

    CRITICAL: this branch fires ONLY on clear intent-to-do statements
    ("I want to borrow", "help me stake") — NEVER on questions about
    the products ("Can I negotiate the interest rate?"). Questions
    are routed to the semantic retriever instead. Previously we only
    skipped very short questions; that was the root cause of repeated
    "wrong intent" mis-routings across James's QA sets.
    """
    actions = entities.get("actions") or []
    amounts = entities.get("amounts") or []
    assets = entities.get("assets") or []
    text = (entities.get("text") or "").lower()

    # Skip clarification entirely for questions. The retriever handles
    # those — they should land on a specific KB entry or scripted
    # intent reply, not a generic "ask for amount + asset" prompt.
    if _is_question_form(text):
        return None

    # REQUIRE an explicit intent-to-do marker ("I want to stake", "help
    # me borrow", "let me deposit"). Just having an action word in the
    # sentence (e.g. the word "staking" inside "Hi, what is staking")
    # is NOT enough — that word-presence heuristic was the root cause
    # of repeated mis-routings where pure questions got the "please
    # indicate the network" clarification reply instead of a KB answer.
    if not _INTENT_TO_DO_RE.search(text):
        return None

    # Short fragments (≤ 3 words) — still skip. Retriever handles these.
    is_short_fragment = len(text.split()) <= 3
    if is_short_fragment:
        return None

    # "I want to borrow" / "can i take a loan" — no amount, no asset
    if "borrow" in actions and not amounts and not assets:
        return (
            "Please specify the loan size (USD, AED, or EUR) and the "
            "collateral asset and amount (BTC or ETH). An indicative quote can then be provided.",
            [{"label": "Loan Calculator", "url": "/dashboard/loans", "kind": "link"}],
        )

    # "I want to stake" — no amount, no asset
    if "stake" in actions and not amounts and not assets:
        return (
            "Please indicate the network (ETH, SOL, POL, ADA, DOT, AVAX, "
            "or ATOM) and the approximate size. Rates and unbonding terms vary by network.",
            [{"label": "Open Staking", "url": "/dashboard/staking", "kind": "link"}],
        )

    # "I want to withdraw" — no asset/amount
    if "withdraw" in actions and not amounts and not assets:
        return (
            "Please specify the asset and amount. Indicative processing: "
            "1-4 hours for crypto, 24 hours for fiat wire.",
            [{"label": "Open Wallets", "url": "/dashboard/wallets", "kind": "link"}],
        )

    # "I want to deposit" — no asset
    if "deposit" in actions and not assets:
        return (
            "Please specify the asset (BTC, ETH, SOL, USDT, or USDC — "
            "40+ networks supported). A deposit address will be generated accordingly.",
            [{"label": "Open Wallets", "url": "/dashboard/wallets", "kind": "link"}],
        )

    # "how much can i get/earn/borrow" WITHOUT asset or amount.
    # Kept deliberately OUTSIDE the question-form guard because even
    # though "how much" is technically a question, the user needs
    # specific input (asset + amount) before we can compute anything.
    if ("borrow" in actions or "earn" in actions) and not amounts and not assets:
        if "how much" in text:
            return (
                "Please specify the asset (BTC or ETH) and the amount "
                "held. Indicative capacity can then be computed.",
                [{"label": "Loan Calculator", "url": "/dashboard/loans", "kind": "link"}],
            )

    return None


# ─── Top-level compose() ───────────────────────────────────────────

def compose(entities: dict) -> Optional[tuple[str, list[dict]]]:
    """Return (reply, actions) if the composer is confident. Otherwise
    return None so the caller falls back to the retriever."""
    amounts = entities.get("amounts") or []
    actions = entities.get("actions") or []
    crisis = entities.get("crisis", False)

    # Crisis / safety questions — handle before anything else
    if crisis:
        reply = _compose_crisis(entities.get("text", ""))
        return (
            reply,
            [
                {"label": "Proof of Reserves", "url": "/dashboard/portfolio", "kind": "link"},
                {"label": "Email Support", "url": "mailto:institutional@alfardanq9.com", "kind": "link"},
            ],
        )

    # Clarification — if user stated an action but left out required
    # info, ask back before giving a generic reply or falling to the
    # retriever.
    if not amounts:
        clarification = _compose_clarification(entities)
        if clarification is not None:
            return clarification
        return None

    # Pick the first/most-relevant amount
    primary = amounts[0]
    val = primary["value"]
    curr = primary["currency"]

    is_fiat = curr in ("USD", "AED", "EUR", "GBP", "BHD", "SAR", "KWD", "QAR", "OMR")
    is_crypto = curr in F.STAKING_APY or curr in ("BTC", "USDT", "USDC", "XRP", "LINK", "BNB", "UNI", "AAVE", "ARB")

    wants_borrow = "borrow" in actions
    wants_stake  = "stake" in actions
    wants_open   = "open_account" in actions or "start" in actions
    mentions_min = any(w in (entities.get("text") or "").lower()
                       for w in ["minimum", "min ", "enough", "qualify"])

    # ─── Crypto amount + borrow intent ────────────────────────────
    if is_crypto and wants_borrow:
        if curr == "BTC":
            reply = _compose_btc_borrow(val)
            return (
                reply,
                [
                    {"label": "Open Loans", "url": "/dashboard/loans", "kind": "link"},
                    {"label": "Lending Service", "url": "/dashboard/services/lending", "kind": "link"},
                ],
            )
        if curr == "ETH":
            reply = _compose_eth_borrow(val)
            return (
                reply,
                [
                    {"label": "Open Loans", "url": "/dashboard/loans", "kind": "link"},
                    {"label": "Lending Service", "url": "/dashboard/services/lending", "kind": "link"},
                ],
            )
        # Other crypto assets aren't loan-collateral today
        return (
            f"{val:g} {curr} is not currently accepted as loan collateral "
            f"— only BTC and ETH. A swap to BTC via our OTC desk, followed "
            f"by a loan against that, can be arranged on request.",
            [{"label": "Open OTC Desk", "url": "/dashboard/services/otc", "kind": "link"}],
        )

    # ─── Crypto amount + stake intent ─────────────────────────────
    if is_crypto and wants_stake:
        reply = _compose_crypto_stake(curr, val)
        if reply:
            return (
                reply,
                [{"label": "Open Staking", "url": "/dashboard/staking", "kind": "link"}],
            )

    # ─── Fiat amount + open-account / minimum context ─────────────
    if is_fiat and (wants_open or mentions_min or entities.get("beginner")):
        # Normalise AED/EUR to USD for tier check
        usd_value = val
        if curr == "AED":
            usd_value = val / F.AED_USD_PEG
        elif curr == "EUR":
            usd_value = val / F.EUR_USD
        reply = _compose_fiat_tier_check(usd_value)
        if reply:
            actions_out = [
                {"label": "Create Account", "url": "/auth/signup", "kind": "link"},
                {"label": "Email Support", "url": "mailto:institutional@alfardanq9.com", "kind": "link"},
            ]
            return reply, actions_out

    # ─── Crypto amount + no specific action: generic "what can I do" ─
    if is_crypto and not (wants_borrow or wants_stake):
        if curr == "BTC":
            borrow_cap = _format_btc_loan_capacity(val, 75)
            return (
                f"With {val:g} BTC, available activities are: insured "
                f"custody (fee 0.10-0.50% p.a.), or borrowing up to "
                f"{borrow_cap} at 75% LTV. BTC does not support staking.",
                [
                    {"label": "Open Wallets", "url": "/dashboard/wallets", "kind": "link"},
                    {"label": "Open Loans", "url": "/dashboard/loans", "kind": "link"},
                ],
            )
        if curr == "ETH":
            apy = F.STAKING_APY.get("ETH", 5.2)
            return (
                f"With {val:g} ETH, available activities are: custody, "
                f"staking (~{apy:.1f}% APY, daily payouts), or borrowing at 75% LTV.",
                [
                    {"label": "Open Staking", "url": "/dashboard/staking", "kind": "link"},
                    {"label": "Open Loans", "url": "/dashboard/loans", "kind": "link"},
                ],
            )
        if curr in F.STAKING_APY:
            apy = F.STAKING_APY[curr]
            return (
                f"With {val:g} {curr}, available activities include "
                f"custody or staking at ~{apy:.1f}% APY. Other options depend on the asset.",
                [{"label": "Open Staking", "url": "/dashboard/staking", "kind": "link"}],
            )

    # ─── Fiat amount without action context — tier hint only ──────
    if is_fiat:
        usd_value = val if curr == "USD" else (
            val / F.AED_USD_PEG if curr == "AED" else val / F.EUR_USD if curr == "EUR" else val
        )
        reply = _compose_fiat_tier_check(usd_value)
        if reply:
            return reply, [
                {"label": "Create Account", "url": "/auth/signup", "kind": "link"},
            ]

    # ─── Clarification — ask back instead of guessing ───────────
    # If the user expressed an intent (borrow / stake / withdraw) but
    # left out the specifics, ask for them instead of dumping a
    # generic answer or falling through to the retriever.
    clarification = _compose_clarification(entities)
    if clarification is not None:
        return clarification

    # ─── Composer isn't confident — fall back to retrieval ───────
    return None
