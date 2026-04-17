"""Structured business facts — single source of truth.

Instead of writing a prose answer for every phrasing, we define the
underlying FACTS once here. The composer layer turns these facts
into natural-sounding answers at runtime.

Facts that change frequently (APYs, spot prices, LTV) are defined here
once and referenced everywhere. Editing a single number ripples through
every answer that uses it.
"""

from __future__ import annotations

# ─── Product minimums + tiers ───────────────────────────────────────

MINIMUMS = {
    # Individual / HNW account
    "individual_usd": 10_000,
    # Institutional account
    "institutional_usd": 100_000,
    # Starter Access (special-consideration programme)
    "starter_usd": 3_000,

    # Lending
    "pledge_btc": 0.5,
    "pledge_eth": 10,

    # Staking
    "stake_per_network_usd": 100_000,

    # OTC
    "otc_ticket_usd": 100_000,

    # RWA offerings
    "airbus_a320_usd": 20_000,
    "gold_bars_usd": 88_000,
}

MAXIMUMS = {
    "otc_ticket_usd": 50_000_000,  # $50M+ standard; larger by request
    "ltv_percent": 75,
    "reserve_coverage_percent": 215,
}

# ─── Lending ────────────────────────────────────────────────────────

LTV = {
    "max_percent": 75,
    "margin_call_percent": 85,
    "liquidation_percent": 90,
}

LENDING_RATES = {
    # APR by term (years) — floor for institutional is 3.25% on 5y
    1: 4.25,
    2: 3.95,
    3: 3.50,
    5: 3.25,
}

LENDING_TIERS = {
    "starter":       {"range_btc": (0.5, 1),   "apr": 4.9,  "label": "Starter"},
    "core":          {"range_btc": (1, 5),     "apr": 4.5,  "label": "Core"},
    "pro":           {"range_btc": (5, 10),    "apr": 4.2,  "label": "Pro"},
    "institutional": {"range_btc": (10, 9999), "apr": 3.9,  "label": "Institutional"},
}

# ─── Staking ────────────────────────────────────────────────────────

STAKING_APY = {
    "ETH":  5.2,
    "SOL":  7.1,
    "POL":  5.8,
    "MATIC": 5.8,   # alias
    "ADA":  4.5,
    "DOT":  14.2,
    "AVAX": 8.5,
    "ATOM": 18.5,
}

STAKING_UNBONDING_DAYS = {
    "ETH":  1.125,   # ~27 hours
    "SOL":  2,
    "DOT":  28,
    "AVAX": 14,
    "ATOM": 21,
    "POL":  0,       # flexible
    "MATIC": 0,
    "ADA":  0,
}

STAKING_COMMISSION_PERCENT = 15  # we take 15% of rewards, not principal

# ─── OTC ────────────────────────────────────────────────────────────

OTC_SPREAD_BPS = {"min": 5, "max": 25}  # 0.05%-0.25%
OTC_SETTLEMENT = {
    "crypto_hours": "1-4",
    "fiat_hours": "24",
}
OTC_SUPPORTED_ASSETS = [
    "BTC", "ETH", "SOL", "USDT", "USDC", "BNB", "XRP", "ADA",
    "AVAX", "DOT", "MATIC", "LINK", "ATOM", "UNI", "AAVE", "ARB",
]
OTC_SETTLEMENT_CURRENCIES = ["AED", "SAR", "KWD", "BHD", "QAR", "OMR", "USD", "EUR"]

# ─── Custody / security ─────────────────────────────────────────────

INSURANCE = {
    "amount_usd": 250_000_000,
    "underwriter": "Lloyd's of London",
    "syndicates": ["Apollo", "Arch", "Ark", "Ascot", "Aspen"],
    "policy_number": "SY-2025-49881",
    "covers": ["theft", "hacks", "insider collusion", "loss of private keys"],
    "does_not_cover": ["market price movements (your own market risk)"],
}

CUSTODY = {
    "provider": "Fireblocks MPC-CMP",
    "cold_percent": 95,
    "warm_percent": 5,
    "segregated": True,
    "rehypothecation": False,
    "data_center_tier": "IV",
}

# ─── Regulatory ─────────────────────────────────────────────────────

REGULATION = {
    "vara_license": "VL/23/10/002",
    "vara_partner": "Fuze (Morpheus Software Technology FZE)",
    "difc_entities": ["Alfardan Holdings Limited (DIFC Reg #5605)",
                      "Alfardan PTC Holding Limited (DIFC Reg #5598)"],
    "cbuae_since_year": 1971,
    "certifications": ["SOC 2 Type II", "ISO 27001", "ISO 27017", "ISO 9001"],
}

# ─── Team (high-signal names + titles) ──────────────────────────────

TEAM = {
    "founder_chairman": {"name": "Sheikh Rashid Al Fardan", "title": "Founder & Chairman"},
    "ceo":              {"name": "Khalid Al Fardan", "title": "CEO & Co-Founder"},
    "vice_chair":       {"name": "Dr. Fatima Al Fardan", "title": "Vice-Chairman"},
    "cio":              {"name": "Hamdan Al Naheri", "title": "CIO"},
    "cto":              {"name": "Omar Al Mahmoud", "title": "CTO"},
    "gc":               {"name": "Nadia Hassan", "title": "General Counsel"},
    "sharia_chair":     {"name": "Sheikh Dr. Tariq Al-Mahrouqi",
                         "title": "Sharia Supervisory Board Chairman (AAOIFI)"},
    "head_otc":         {"name": "Layla Al-Fayez", "title": "Head of Institutional OTC Desk"},
}

# ─── Heritage ────────────────────────────────────────────────────────

HERITAGE = {
    "group_years": 50,
    "exchange_founded_year": 1971,
    "q9_launched_year": 2023,
    "pearl_trading_origin": "1800s",
    "headquarters": "The Onyx Tower 2, The Greens, Sheikh Zayed Road, Dubai",
    "locations": ["Dubai (DIFC HQ)", "London (service)", "Kuala Lumpur (APAC)"],
}

# ─── Contacts ────────────────────────────────────────────────────────

CONTACTS = {
    "general": "institutional@alfardanq9.com",
    "otc":     "otc@alfardanq9.com",
    "shariah": "shariah@alfardanq9.com",
    "compliance": "compliance@alfardanq9.com",
    "security": "security@alfardanq9.com",
    "complaints": "complaints@alfardanq9.com",
    "feedback": "feedback@alfardanq9.com",
}

# ─── Pricing / FX ───────────────────────────────────────────────────

AED_USD_PEG = 3.67  # fixed UAE peg
EUR_USD = 0.94      # admin-published, ish

# Spot-ish reference prices (used only for sizing estimates in composed
# answers — live prices come from priceMap in the CRM, not here).
REFERENCE_PRICES_USD = {
    "BTC": 70_000,
    "ETH": 2_300,
    "SOL": 160,
    "USDT": 1,
    "USDC": 1,
    "BNB": 600,
    "AVAX": 35,
    "DOT": 7,
    "ATOM": 8,
    "MATIC": 0.8,
    "ADA": 0.55,
    "LINK": 13,
}

# ─── Helpers ────────────────────────────────────────────────────────

def estimate_loan_capacity_usd(asset: str, amount: float, ltv_percent: float | None = None) -> float:
    """Loan capacity in USD given collateral amount + asset + LTV %."""
    price = REFERENCE_PRICES_USD.get(asset.upper(), 0)
    ltv = (ltv_percent if ltv_percent is not None else LTV["max_percent"]) / 100
    return amount * price * ltv


def pledge_minimum(asset: str) -> float | None:
    """Minimum collateral for a loan, in the asset's own units."""
    m = {"BTC": MINIMUMS["pledge_btc"], "ETH": MINIMUMS["pledge_eth"]}
    return m.get(asset.upper())


def fmt_usd(v: float) -> str:
    """Pretty USD string — $1,234,567 or $1.23M."""
    if v >= 1_000_000:
        return f"${v/1_000_000:.2f}M".replace(".00M", "M")
    if v >= 1_000:
        return f"${v:,.0f}"
    return f"${v:.2f}"
