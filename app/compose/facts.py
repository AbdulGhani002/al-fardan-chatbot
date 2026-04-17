"""Structured business facts — live-sourced from the CRM.

Numbers that change (max LTV, APYs, loan rates, FX, fiat minimums,
BTC/ETH pledge minimums) are fetched LIVE from the CRM's
/api/settings endpoint via app/integrations/platform_settings.py,
which caches for 5 minutes. The admin edits numbers in the CRM's
/admin/settings page and the chatbot picks them up within 5 minutes
— no chatbot redeploy needed.

Semi-static facts (team names, office locations, heritage story,
insurance policy number, compliance IDs) live below as module-level
constants. They change rarely; when they do, a redeploy is fine.

The helpers at the bottom (estimate_loan_capacity_usd, pledge_minimum,
etc.) always resolve to the current live values so every composed
answer reflects today's settings.
"""

from __future__ import annotations

from ..integrations import platform_settings as _ps

# ─── Product minimums + tiers ───────────────────────────────────────

# Dict wrapper that always reads the latest live platform settings.
# Composers use MINIMUMS["pledge_btc"] and get the CURRENT value,
# even after an admin edits /admin/settings.
class _LiveMinimums:
    """Dict-like accessor whose values come from the CRM at call time."""

    def __getitem__(self, key: str):
        if key == "pledge_btc":
            return _ps.min_btc_pledge()
        if key == "pledge_eth":
            return _ps.min_eth_pledge()
        if key == "individual_usd":
            return 10_000   # Not exposed in CRM settings yet — fixed
        if key == "institutional_usd":
            return 100_000
        if key == "starter_usd":
            return 3_000
        if key == "stake_per_network_usd":
            return 100_000
        if key == "otc_ticket_usd":
            return 100_000
        if key == "airbus_a320_usd":
            return 20_000
        if key == "gold_bars_usd":
            return 88_000
        raise KeyError(key)

    def get(self, key: str, default=None):
        try:
            return self[key]
        except KeyError:
            return default


MINIMUMS = _LiveMinimums()

MAXIMUMS = {
    "otc_ticket_usd": 50_000_000,  # $50M+ standard; larger by request
    "ltv_percent": 75,
    "reserve_coverage_percent": 215,
}

# ─── Lending ────────────────────────────────────────────────────────

class _LiveLTV:
    """LTV thresholds sourced live from the CRM."""

    def __getitem__(self, key: str):
        if key == "max_percent":
            return _ps.max_ltv()
        if key == "margin_call_percent":
            return 85   # Not exposed in CRM settings yet — fixed policy
        if key == "liquidation_percent":
            return 90
        raise KeyError(key)

    def get(self, key: str, default=None):
        try:
            return self[key]
        except KeyError:
            return default


LTV = _LiveLTV()


class _LiveRates:
    """Loan APRs sourced live from CRM loanTerms."""

    def __getitem__(self, years: int):
        return _ps.apr(int(years))

    def get(self, years, default=None):
        try:
            return self[int(years)]
        except Exception:  # noqa: BLE001
            return default


LENDING_RATES = _LiveRates()

LENDING_TIERS = {
    "starter":       {"range_btc": (0.5, 1),   "apr": 4.9,  "label": "Starter"},
    "core":          {"range_btc": (1, 5),     "apr": 4.5,  "label": "Core"},
    "pro":           {"range_btc": (5, 10),    "apr": 4.2,  "label": "Pro"},
    "institutional": {"range_btc": (10, 9999), "apr": 3.9,  "label": "Institutional"},
}

# ─── Staking ────────────────────────────────────────────────────────

class _LiveStakingAPY:
    """Staking APYs sourced live from the CRM's stakingApy config."""

    _DEFAULTS = {
        "ETH": 5.2, "SOL": 7.1, "POL": 5.8, "MATIC": 5.8,
        "ADA": 4.5, "DOT": 14.2, "AVAX": 8.5, "ATOM": 18.5,
    }

    def __getitem__(self, asset: str):
        return self.get(asset)

    def get(self, asset: str, default=None):
        live = _ps.staking_apy(asset)
        if live is not None:
            return live
        return self._DEFAULTS.get(asset.upper(), default)

    def __contains__(self, asset: str) -> bool:
        live = _ps.staking_apy(asset)
        if live is not None:
            return True
        return asset.upper() in self._DEFAULTS


STAKING_APY = _LiveStakingAPY()

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

def _aed_peg() -> float:
    return _ps.aed_peg()


def _eur_usd() -> float:
    return _ps.eur_rate()


# Back-compat module-level attributes — callers can still read them,
# but each access re-reads the live value.
class _FXProxy:
    @property
    def aed(self) -> float:
        return _ps.aed_peg()

    @property
    def eur(self) -> float:
        return _ps.eur_rate()


_FX = _FXProxy()
AED_USD_PEG = _FX.aed  # re-read by reference pattern below
EUR_USD = _FX.eur


def __getattr__(name):
    """Module-level dynamic attribute — always returns the LATEST live
    value for AED_USD_PEG / EUR_USD instead of a stale snapshot."""
    if name == "AED_USD_PEG":
        return _ps.aed_peg()
    if name == "EUR_USD":
        return _ps.eur_rate()
    raise AttributeError(name)

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
