"""Dense semantic retriever — ONNX-compiled sentence-transformers via fastembed.

Why semantic over TF-IDF?
  TF-IDF only matches surface tokens. "How much credit against my btc?"
  doesn't share many words with "What's the maximum LTV you offer?" —
  yet they're asking the same question. A sentence-embedding model
  encodes both into the same neighbourhood of a 384-dim vector space,
  so cosine similarity finds the match even with no shared words.

Why fastembed and not sentence-transformers + torch?
  Stack weight. sentence-transformers pulls pytorch (~2GB), CUDA libs,
  and tokenizers — blows up container size + cold-start. fastembed
  ships BAAI/bge-small-en-v1.5 as an optimised ONNX graph with
  onnxruntime (no torch). ~130MB model, ~10ms/query on CPU, identical
  embedding quality for our use case.

Interface parity with TfidfRetriever:
  size, vocab_size, search(query, top_k), save(path), load(path),
  entries attribute. main.py doesn't care which retriever it's holding
  — both return a list[SearchResult].
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .tfidf import KbEntry, SearchResult


# fastembed is an optional dependency — if it can't be imported (e.g.
# model download failed during build, memory pressure, onnxruntime
# missing), main.py falls back to the TF-IDF retriever. The import is
# wrapped so the whole module still loads cleanly and the `available()`
# check below tells callers whether to instantiate.
try:
    from fastembed import TextEmbedding  # type: ignore
    _FASTEMBED_AVAILABLE = True
    _FASTEMBED_ERR: Optional[Exception] = None
except Exception as err:  # noqa: BLE001
    TextEmbedding = None  # type: ignore
    _FASTEMBED_AVAILABLE = False
    _FASTEMBED_ERR = err


# Small, fast, English-focused. Upgrade to bge-small-multilingual when
# we ship the multi-language rollout (Arabic/Chinese/etc.).
DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


def available() -> bool:
    """True iff fastembed imported cleanly and the model is usable."""
    return _FASTEMBED_AVAILABLE


def unavailable_reason() -> str:
    if _FASTEMBED_AVAILABLE:
        return ""
    return f"fastembed import failed: {_FASTEMBED_ERR!r}"


class DenseRetriever:
    """Sentence-embedding retriever with the same public surface as
    TfidfRetriever. Pickle-serialisable so we cache the embedding
    matrix alongside the index."""

    def __init__(
        self,
        entries: list[KbEntry],
        model_name: str = DEFAULT_MODEL,
        embedder: object | None = None,
    ) -> None:
        if not _FASTEMBED_AVAILABLE:
            raise RuntimeError(unavailable_reason())

        self.entries = list(entries)
        self._model_name = model_name
        # Cache the embedder instance across calls. Initialising it
        # downloads the model on first run (~130MB); subsequent calls
        # are cheap.
        self._embedder = embedder or TextEmbedding(model_name=model_name)  # type: ignore
        self._matrix: Optional[np.ndarray] = None

        if self.entries:
            docs = [self._searchable_for(e) for e in self.entries]
            # fastembed returns a generator of np arrays; materialise +
            # L2-normalise so we can use dot-product as cosine sim.
            vecs = list(self._embedder.embed(docs))  # type: ignore[attr-defined]
            matrix = np.asarray(vecs, dtype=np.float32)
            matrix /= np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12
            self._matrix = matrix

    @staticmethod
    def _searchable_for(e: KbEntry) -> str:
        """Build the string we embed.

        Unlike TF-IDF we don't need aggressive keyword repetition —
        dense models handle phrasing naturally. Just: question +
        aliases (different phrasings of the same question) + a short
        slice of the answer (so the answer body contributes to
        matching when the question is terse).
        """
        parts: list[str] = [e.question]
        parts.extend(e.aliases)
        if e.answer:
            # Keep the first ~400 chars of the answer — enough to
            # anchor semantically without drowning the question.
            parts.append(e.answer[:400])
        return " ".join(parts)

    # ─── Public API — mirrors TfidfRetriever exactly ───────────────

    @property
    def size(self) -> int:
        return len(self.entries)

    @property
    def vocab_size(self) -> int:
        """Embedding dimensionality (384 for bge-small). Naming
        preserved for parity with TF-IDF; it's not actually a 'vocab'."""
        if self._matrix is None:
            return 0
        return int(self._matrix.shape[1])

    def search(self, query: str, top_k: int = 3) -> list[SearchResult]:
        if self._matrix is None or not query.strip() or self._embedder is None:
            return []
        q_vecs = list(self._embedder.embed([query]))  # type: ignore[attr-defined]
        q = np.asarray(q_vecs[0], dtype=np.float32)
        q /= np.linalg.norm(q) + 1e-12
        sims = self._matrix @ q  # cosine because both are L2-normalised
        if top_k >= len(self.entries):
            order = np.argsort(-sims)
        else:
            top_idx = np.argpartition(-sims, top_k)[:top_k]
            order = top_idx[np.argsort(-sims[top_idx])]
        results: list[SearchResult] = []
        for idx in order[:top_k]:
            score = float(sims[idx])
            if score <= 0:
                continue
            results.append(SearchResult(entry=self.entries[idx], score=score))
        return results

    # ─── Persistence ───────────────────────────────────────────────

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(
                {
                    "entries": self.entries,
                    "matrix": self._matrix,
                    "model_name": self._model_name,
                    "variant": "dense",
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    @classmethod
    def load(cls, path: Path) -> "DenseRetriever":
        if not _FASTEMBED_AVAILABLE:
            raise RuntimeError(unavailable_reason())
        with path.open("rb") as f:
            blob = pickle.load(f)
        if blob.get("variant") != "dense":
            raise ValueError(
                "index.pkl is not a dense index — expected variant='dense'"
            )
        obj = cls.__new__(cls)
        obj.entries = blob["entries"]
        obj._matrix = blob["matrix"]
        obj._model_name = blob.get("model_name", DEFAULT_MODEL)
        obj._embedder = TextEmbedding(model_name=obj._model_name)  # type: ignore
        return obj


def build_dense_from_kb(kb_dir: Path) -> DenseRetriever:
    """Parallel of build_retriever_from_kb, returns a dense retriever."""
    from .tfidf import load_kb  # import here to avoid circular at module load

    entries = load_kb(kb_dir)
    return DenseRetriever(entries)
