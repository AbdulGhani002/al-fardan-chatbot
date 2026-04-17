"""Live CRM settings fetcher — sourced from /api/settings.

The CRM's PlatformSettings collection is the canonical source of
truth for every number that changes over time: max LTV, staking APYs,
loan APRs, OTC spread, FX rates, compliance IDs, fiat minimums, BTC/
ETH pledge minimums.

Admin edits them at /admin/settings; within 5 minutes every bot reply
reflects the new numbers. No duplicated constants in Python.

If the CRM is unreachable we serve whatever we last cached, and as a
final fallback use the hardcoded defaults at module load — so the bot
degrades gracefully rather than going silent.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Optional

import httpx

from ..config import settings


# ─── Hard defaults — only used if CRM has never responded ────────────
# These are deliberately identical to what the PlatformSettings model
# defaults ship with today. If you need to change one, edit the CRM
# admin settings page — THAT is the source of truth.
_DEFAULTS: dict[str, Any] = {
    "loanTerms": {
        "years1Apr": 4.25,
        "years2Apr": 3.95,
        "years3Apr": 3.5,
        "years5Apr": 3.25,
        "maxLtv": 75,
        "minBtc": 0.5,
        "minEth": 10,
    },
    "stakingApy": {
        "ETH": 5.2, "SOL": 7.1, "POL": 5.8, "MATIC": 5.8,
        "ADA": 4.5, "DOT": 14.2, "AVAX": 8.5, "ATOM": 18.5,
    },
    "otcSpreadBps": 15,
    "fxRates": {"AED": 3.67, "EUR": 0.94, "USD": 1.0},
    "complianceIds": {
        "fireblocks": "Fireblocks MPC",
        "lloyds": "SY-2025-49881",
        "vara": "VL/23/10/002",
        "difc": "DIFC #5605",
        "entity": "Alfardan Holdings",
    },
}


_TTL_SECONDS = 300   # 5 minutes
_FETCH_TIMEOUT = 5.0


class _LiveSettingsCache:
    """Thread-safe cached view of the CRM platform settings."""

    def __init__(self) -> None:
        self._data: Optional[dict[str, Any]] = None
        self._fetched_at: float = 0.0
        self._lock = threading.Lock()
        self._last_error: str = ""

    @property
    def crm_url(self) -> str:
        return settings.crm_base_url.rstrip("/") + "/api/settings"

    def _merge_with_defaults(self, remote: dict[str, Any]) -> dict[str, Any]:
        """Fill missing keys from _DEFAULTS so composers never KeyError
        on older CRM settings shapes (e.g. before we added more APYs)."""
        out: dict[str, Any] = {}
        for section, default_val in _DEFAULTS.items():
            if isinstance(default_val, dict):
                merged = dict(default_val)
                merged.update(remote.get(section) or {})
                out[section] = merged
            else:
                out[section] = remote.get(section, default_val)
        # pass through any extra fields unchanged
        for k, v in remote.items():
            out.setdefault(k, v)
        return out

    def _fetch_remote(self) -> Optional[dict[str, Any]]:
        """One-shot synchronous fetch. Returns None on failure."""
        try:
            with httpx.Client(timeout=_FETCH_TIMEOUT) as c:
                r = c.get(self.crm_url)
                if r.status_code != 200:
                    self._last_error = f"HTTP {r.status_code}"
                    return None
                payload = r.json()
                if not payload.get("success"):
                    self._last_error = "success=false"
                    return None
                return payload.get("data") or {}
        except Exception as err:  # noqa: BLE001
            self._last_error = f"{type(err).__name__}: {err}"
            return None

    def get(self) -> dict[str, Any]:
        """Return the current settings. Blocking fetch on first call;
        lock-free cache-hit on every subsequent call within the TTL."""
        with self._lock:
            now = time.time()
            if self._data is not None and now - self._fetched_at < _TTL_SECONDS:
                return self._data
        # Fetch outside the lock to avoid serialising slow HTTP calls
        remote = self._fetch_remote()
        with self._lock:
            if remote is not None:
                self._data = self._merge_with_defaults(remote)
                self._fetched_at = time.time()
            elif self._data is None:
                # First-ever call + fetch failed → use defaults so the
                # chatbot still answers, just with stale values.
                self._data = self._merge_with_defaults({})
                self._fetched_at = time.time()
            return self._data

    def force_refresh(self) -> bool:
        """Blocking refresh — used by /admin/reindex to propagate a
        settings edit without waiting for the TTL."""
        remote = self._fetch_remote()
        if remote is None:
            return False
        with self._lock:
            self._data = self._merge_with_defaults(remote)
            self._fetched_at = time.time()
        return True

    def debug_snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "url": self.crm_url,
                "last_fetched_at": self._fetched_at,
                "stale_seconds": (
                    round(time.time() - self._fetched_at, 1)
                    if self._fetched_at else None
                ),
                "last_error": self._last_error or None,
                "has_cached_data": self._data is not None,
            }


_cache = _LiveSettingsCache()


# ─── Public surface used by composer + templates ───────────────────

def get_settings() -> dict[str, Any]:
    return _cache.get()


def force_refresh() -> bool:
    return _cache.force_refresh()


def debug_snapshot() -> dict[str, Any]:
    return _cache.debug_snapshot()


# Convenience readers — each one is a single-purpose function composers
# can call without having to know the settings schema.

def max_ltv() -> float:
    return float(get_settings().get("loanTerms", {}).get("maxLtv", 75))


def apr(years: int) -> float:
    """APR for a given loan term in years (1, 2, 3, 5)."""
    terms = get_settings().get("loanTerms", {})
    key = f"years{years}Apr"
    default = {1: 4.25, 2: 3.95, 3: 3.5, 5: 3.25}.get(years, 4.25)
    return float(terms.get(key, default))


def min_btc_pledge() -> float:
    return float(get_settings().get("loanTerms", {}).get("minBtc", 0.5))


def min_eth_pledge() -> float:
    return float(get_settings().get("loanTerms", {}).get("minEth", 10))


def staking_apy(asset: str) -> Optional[float]:
    return get_settings().get("stakingApy", {}).get(asset.upper())


def otc_spread_bps() -> int:
    return int(get_settings().get("otcSpreadBps", 15))


def aed_peg() -> float:
    return float(get_settings().get("fxRates", {}).get("AED", 3.67))


def eur_rate() -> float:
    return float(get_settings().get("fxRates", {}).get("EUR", 0.94))


def compliance_id(key: str) -> str:
    return str(get_settings().get("complianceIds", {}).get(key, ""))
