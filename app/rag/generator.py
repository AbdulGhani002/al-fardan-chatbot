"""LLM generation backends for the RAG layer.

Two backends, configured via env:

  GENERATOR_BACKEND=ollama           (default, open-source, on-VPS)
  GENERATOR_BACKEND=openai_compatible (Genspark / OpenAI / Groq / …)

Both implement the same `Generator.generate(system, user)` interface
so the rest of the code (main.py /chat) doesn't care which is active.

Fallback chain at runtime:
  1. Configured backend (Ollama or OpenAI-compatible)
  2. If it errors / times out → the caller returns the top-matched
     KB entry verbatim (existing TF-IDF behaviour)
  3. If retrieval also empty → fallback apology + human-handoff CTAs.

Nothing here is Anthropic-SDK or OpenAI-SDK specific — we speak the
raw REST protocols via httpx, so we don't need to ship vendor SDKs.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Optional

import httpx


# ─── Result ────────────────────────────────────────────────────────────

@dataclass
class GenerationResult:
    text: str
    # Latency in milliseconds — useful for p50/p99 observability.
    latency_ms: int
    # Which backend produced this answer (for match_type / debugging).
    backend: str
    # Model identifier the backend actually used.
    model: str


# ─── Base interface ────────────────────────────────────────────────────

class Generator:
    """All generator implementations expose this interface."""

    backend: str
    model: str

    async def generate(self, system: str, user: str) -> GenerationResult:
        raise NotImplementedError

    async def health(self) -> bool:
        """Quick liveness probe. Return True if the backend looks
        ready to serve. Called once on startup + periodically.
        """
        raise NotImplementedError


# ─── Ollama (default — runs on the VPS alongside the chatbot) ──────────

class OllamaGenerator(Generator):
    """Talks to a local Ollama HTTP server.

    Ollama is a single Go binary that serves quantised open-source
    LLMs over HTTP. Zero external API dependency, fully self-hosted.
    Recommended default: Qwen 2.5 7B Instruct Q4_K_M (~4.5GB RAM,
    multilingual, Apache-2.0). Swap via OLLAMA_MODEL env var.
    """

    backend = "ollama"

    def __init__(
        self,
        host: str,
        model: str,
        timeout_s: float = 30.0,
    ) -> None:
        self._base = host.rstrip("/")
        self.model = model
        self._timeout = timeout_s

    async def health(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5.0) as c:
                r = await c.get(f"{self._base}/api/tags")
                return r.status_code == 200
        except Exception:  # noqa: BLE001
            return False

    async def generate(self, system: str, user: str) -> GenerationResult:
        # Ollama's /api/chat endpoint is the OpenAI-style messages API.
        # We prefer it over /api/generate because it handles the
        # system-prompt framing correctly for most instruct models.
        import time

        payload = {
            "model": self.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "options": {
                "temperature": 0.3,
                "top_p": 0.9,
                "num_predict": 320,
                "repeat_penalty": 1.05,
            },
        }
        t0 = time.time()
        async with httpx.AsyncClient(timeout=self._timeout) as c:
            r = await c.post(f"{self._base}/api/chat", json=payload)
            r.raise_for_status()
            data = r.json()
        latency_ms = int((time.time() - t0) * 1000)
        text = str(
            data.get("message", {}).get("content", "") or ""
        ).strip()
        return GenerationResult(
            text=text,
            latency_ms=latency_ms,
            backend=self.backend,
            model=self.model,
        )


# ─── OpenAI-compatible (Genspark / OpenAI / Groq / Together / etc.) ────

class OpenAICompatibleGenerator(Generator):
    """Works with any provider that speaks the OpenAI /v1/chat/completions
    protocol — including Genspark's API, Groq, Together.ai, Fireworks,
    and self-hosted vLLM.

    Configure via env:
      OPENAI_API_BASE  (e.g. https://api.genspark.ai/v1)
      OPENAI_API_KEY
      OPENAI_MODEL     (provider-specific, e.g. 'qwen/qwen2.5-7b-instruct')
    """

    backend = "openai_compatible"

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout_s: float = 30.0,
        provider_label: str = "openai_compatible",
    ) -> None:
        self._base = base_url.rstrip("/")
        self._key = api_key
        self.model = model
        self._timeout = timeout_s
        self.backend = provider_label

    async def health(self) -> bool:
        # Most OpenAI-compatible providers expose /v1/models but some
        # require auth. Probe it cheaply — if it returns anything other
        # than a network error we consider the endpoint reachable.
        try:
            async with httpx.AsyncClient(timeout=5.0) as c:
                r = await c.get(
                    f"{self._base}/models",
                    headers={"Authorization": f"Bearer {self._key}"},
                )
                return r.status_code < 500
        except Exception:  # noqa: BLE001
            return False

    async def generate(self, system: str, user: str) -> GenerationResult:
        import time

        payload = {
            "model": self.model,
            "temperature": 0.3,
            "top_p": 0.9,
            "max_tokens": 320,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        t0 = time.time()
        async with httpx.AsyncClient(timeout=self._timeout) as c:
            r = await c.post(
                f"{self._base}/chat/completions",
                json=payload,
                headers={
                    "Authorization": f"Bearer {self._key}",
                    "Content-Type": "application/json",
                },
            )
            r.raise_for_status()
            data = r.json()
        latency_ms = int((time.time() - t0) * 1000)
        text = str(
            (data.get("choices") or [{}])[0]
            .get("message", {})
            .get("content", "")
            or ""
        ).strip()
        return GenerationResult(
            text=text,
            latency_ms=latency_ms,
            backend=self.backend,
            model=self.model,
        )


# ─── Factory ───────────────────────────────────────────────────────────

def build_generator() -> Optional[Generator]:
    """Construct the configured generator based on settings, or
    return None if RAG is not configured (caller falls back to
    plain retrieval)."""
    from ..config import settings

    kind = (settings.generator_backend or "").strip().lower()

    if kind == "ollama":
        if not settings.ollama_host or not settings.ollama_model:
            return None
        return OllamaGenerator(
            host=settings.ollama_host,
            model=settings.ollama_model,
            timeout_s=settings.rag_timeout_s,
        )

    if kind in {"openai_compatible", "openai", "genspark", "groq"}:
        if not settings.openai_api_base or not settings.openai_api_key:
            return None
        # Use a friendlier backend label for the common providers
        # so it shows up nicely in logs / match_type.
        label_by_kind = {
            "openai": "openai",
            "genspark": "genspark",
            "groq": "groq",
        }
        label = label_by_kind.get(kind, "openai_compatible")
        return OpenAICompatibleGenerator(
            base_url=settings.openai_api_base,
            api_key=settings.openai_api_key,
            model=settings.openai_model,
            timeout_s=settings.rag_timeout_s,
            provider_label=label,
        )

    return None


async def probe_generator(gen: Generator, max_wait_s: float = 2.0) -> bool:
    """Short-timeout health check used at startup. Never raises."""
    try:
        return await asyncio.wait_for(gen.health(), timeout=max_wait_s)
    except Exception:  # noqa: BLE001
        return False
