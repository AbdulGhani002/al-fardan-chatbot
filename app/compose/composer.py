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


# ─── Reusable ask-for-next-step phrases ──────────────────────────────
_NEXT_STEP_ASKS = {
    "ask_starter_or_topup": "Which path works better for you — top up, or Starter Access?",
    "ask_firm_quote":       "Want a firm quote for your trade?",
    "ask_walk_through":     "Want me to walk you through the first step?",
    "ask_specific_concern": "Any specific concern I can go deeper on?",
    "ask_schedule_call":    "Want me to schedule a call with our team?",
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
            f"more economic. Want me to suggest one, or top up?"
        )
    if amount_usd < indiv_min:
        short = indiv_min - amount_usd
        return (
            f"{fmt(amount_usd)} qualifies for Starter Access "
            f"({fmt(starter_min)}–{fmt(indiv_min - 1)}). "
            f"You're {fmt(short)} from the individual tier. "
            f"Want the Starter Access application, or top up to individual?"
        )
    if amount_usd < inst_min:
        short = inst_min - amount_usd
        return (
            f"{fmt(amount_usd)} is in the individual tier "
            f"(minimum {fmt(indiv_min)}). You're {fmt(short)} from full "
            f"institutional ({fmt(inst_min)}). Happy where you are?"
        )
    if amount_usd < 1_000_000:
        return (
            f"{fmt(amount_usd)} qualifies for institutional access — "
            f"dedicated RM, tighter OTC spreads, priority onboarding. "
            f"Want me to start your application?"
        )
    if amount_usd < 5_000_000:
        return (
            f"{fmt(amount_usd)} puts you in our institutional tier. "
            f"Enhanced staking rates, VIP OTC coverage. Want a call "
            f"with Layla (OTC) or Khalid (CEO)?"
        )
    if amount_usd < 25_000_000:
        return (
            f"{fmt(amount_usd)} qualifies for Institutional tier with "
            f"custom terms — dedicated validator, bespoke OTC spreads. "
            f"Shall I set up a bespoke proposal?"
        )
    return (
        f"{fmt(amount_usd)} is Sovereign-tier territory. +0.6% enhanced "
        f"staking rates, direct desk access, bespoke everything. Want "
        f"me to arrange an introduction to Khalid (CEO)?"
    )


# ─── Template: crypto amount + borrow ──────────────────────────────

def _compose_btc_borrow(btc: float) -> str:
    pledge_min = F.pledge_minimum("BTC") or F.MINIMUMS["pledge_btc"]
    if btc < pledge_min:
        diff = pledge_min - btc
        return (
            f"{btc:g} BTC is below our {pledge_min:g} BTC minimum pledge. "
            f"You're {diff:g} short. Want to add {diff:g} BTC, or look "
            f"at Starter Access for smaller positions?"
        )
    cap = _format_btc_loan_capacity(btc, 75)
    cap_safe = _format_btc_loan_capacity(btc, 50)
    rate = F.LENDING_RATES[1]
    return (
        f"With {btc:g} BTC as collateral at our max 75% LTV you can "
        f"borrow up to {cap}. At a safer 50% LTV: {cap_safe}. Rates "
        f"from {rate:.2f}% APR for 1-year. Want a firm quote?"
    )


def _compose_eth_borrow(eth: float) -> str:
    pledge_min = F.pledge_minimum("ETH") or F.MINIMUMS["pledge_eth"]
    if eth < pledge_min:
        diff = pledge_min - eth
        return (
            f"{eth:g} ETH is below our {pledge_min:g} ETH minimum pledge. "
            f"You're {diff:g} short. Want to add {diff:g} ETH, or try "
            f"Starter Access for smaller positions?"
        )
    cap = _format_eth_loan_capacity(eth, 75)
    cap_safe = _format_eth_loan_capacity(eth, 50)
    return (
        f"With {eth:g} ETH as collateral at 75% LTV you can borrow "
        f"up to {cap}. At a safer 50% LTV: {cap_safe}. No prepayment "
        f"penalty on any term. Want a firm quote?"
    )


# ─── Template: crypto amount + stake ───────────────────────────────

def _compose_crypto_stake(asset: str, amount: float) -> Optional[str]:
    apy = F.STAKING_APY.get(asset)
    if apy is None:
        # BTC can't stake
        if asset == "BTC":
            return (
                "Bitcoin doesn't support staking — it's proof-of-work. "
                "You can use BTC as loan collateral (up to 75% LTV) and "
                "deploy the borrowed fiat into a yield strategy. Want to "
                "explore that path?"
            )
        return None
    unbonding = F.STAKING_UNBONDING_DAYS.get(asset, 0)
    price = F.REFERENCE_PRICES_USD.get(asset, 0)
    usd_value = amount * price
    stake_min = F.MINIMUMS["stake_per_network_usd"]
    if usd_value < stake_min:
        short = stake_min - usd_value
        return (
            f"{amount:g} {asset} (~{F.fmt_usd(usd_value)}) is below our "
            f"{F.fmt_usd(stake_min)} staking minimum per network. You're "
            f"about {F.fmt_usd(short)} short. Starter Access may open "
            f"lower tiers — want me to check eligibility?"
        )
    # Above minimum — give a real projection
    yearly = amount * (apy / 100)
    unbonding_note = (
        f"no lock-up" if unbonding == 0 else f"{unbonding:g}-day unbonding"
    )
    return (
        f"At {apy:.1f}% APY on {amount:g} {asset} you'd earn about "
        f"{yearly:g} {asset}/year (~{F.fmt_usd(yearly * price)}), paid "
        f"daily. {unbonding_note.capitalize()}. Ready to stake?"
    )


# ─── Template: crisis / safety ──────────────────────────────────────

def _compose_crisis(text: str) -> str:
    ins = F.INSURANCE
    covers = ", ".join(ins["covers"])
    return (
        f"Your assets are segregated in {F.CUSTODY['provider']} vaults — "
        f"legally separate from our balance sheet. We never lend them "
        f"out. Plus {F.fmt_usd(ins['amount_usd'])} {ins['underwriter']} "
        f"insurance covering {covers}. Market price swings are your own "
        f"risk though. Any specific scenario worrying you?"
    )


# ─── Clarification: ask back when key info is missing ─────────────

def _compose_clarification(entities: dict) -> Optional[tuple[str, list[dict]]]:
    """If the user stated an intent but left out the critical info,
    ask for it instead of guessing."""
    actions = entities.get("actions") or []
    amounts = entities.get("amounts") or []
    assets = entities.get("assets") or []
    text = (entities.get("text") or "").lower()

    # Short question — don't clarify on casual questions
    is_short_question = len(text.split()) <= 3

    # "I want to borrow" / "can i take a loan" — no amount, no asset
    if "borrow" in actions and not amounts and not assets and not is_short_question:
        return (
            "Happy to help with a loan. Two things: how much do you want "
            "to borrow (USD, AED, or EUR), and how much BTC or ETH can "
            "you pledge as collateral?",
            [{"label": "Loan Calculator", "url": "/dashboard/loans", "kind": "link"}],
        )

    # "I want to stake" — no amount, no asset
    if "stake" in actions and not amounts and not assets and not is_short_question:
        return (
            "Good choice — which network would you like to stake? We "
            "support ETH, SOL, POL, ADA, DOT, AVAX, and ATOM. And "
            "roughly how much?",
            [{"label": "Open Staking", "url": "/dashboard/staking", "kind": "link"}],
        )

    # "I want to withdraw" — no asset/amount
    if "withdraw" in actions and not amounts and not assets and not is_short_question:
        return (
            "Sure — which asset are you withdrawing, and how much? "
            "Typical processing: 1-4 hours for crypto, 24 hours for fiat wire.",
            [{"label": "Open Wallets", "url": "/dashboard/wallets", "kind": "link"}],
        )

    # "I want to deposit" — no asset
    if "deposit" in actions and not assets and not is_short_question:
        return (
            "Pick an asset and I'll get you a deposit address. BTC, ETH, "
            "SOL, USDT, or USDC? (We support 40+ networks — which one?)",
            [{"label": "Open Wallets", "url": "/dashboard/wallets", "kind": "link"}],
        )

    # "how much can i get/earn/borrow" without asset or amount
    if ("borrow" in actions or "earn" in actions) and not amounts and not assets:
        if "how much" in text:
            return (
                "Happy to run the numbers. Quick question: which asset "
                "(BTC or ETH), and how much of it do you have?",
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
            f"{val:g} {curr} isn't currently accepted as loan collateral "
            f"— we accept BTC and ETH only. Want to swap to BTC via our "
            f"OTC desk and borrow against that?",
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
                f"With {val:g} BTC you can: custody (insured cold storage, "
                f"fee 0.10-0.50%/yr), or borrow up to {borrow_cap} at 75% "
                f"LTV. BTC doesn't stake. Which direction sounds useful?",
                [
                    {"label": "Open Wallets", "url": "/dashboard/wallets", "kind": "link"},
                    {"label": "Open Loans", "url": "/dashboard/loans", "kind": "link"},
                ],
            )
        if curr == "ETH":
            apy = F.STAKING_APY.get("ETH", 5.2)
            return (
                f"With {val:g} ETH you can: custody, stake (~{apy:.1f}% "
                f"APY daily payouts), or borrow at 75% LTV. Which angle "
                f"interests you?",
                [
                    {"label": "Open Staking", "url": "/dashboard/staking", "kind": "link"},
                    {"label": "Open Loans", "url": "/dashboard/loans", "kind": "link"},
                ],
            )
        if curr in F.STAKING_APY:
            apy = F.STAKING_APY[curr]
            return (
                f"With {val:g} {curr} you can custody it or stake at "
                f"~{apy:.1f}% APY. Other options depend on the asset. "
                f"Which interests you?",
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
