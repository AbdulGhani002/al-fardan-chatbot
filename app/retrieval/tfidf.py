"""TF-IDF retriever for the curated knowledge base.

Why TF-IDF and not a transformer embedding?
  - Zero model weights, zero GPU, fits in <50 MB RAM.
  - Deterministic and interpretable — we can log exactly which tokens
    matched and why.
  - Fast to rebuild when the admin edits the KB (<1s for 1000 entries).
  - Good enough for FAQ-style retrieval where queries and KB entries
    share domain vocabulary.

When we outgrow TF-IDF (e.g., users phrase questions with words the KB
never uses), we swap this module for a sentence-transformers embedding
retriever behind the same `.search()` interface — no consumer changes.
"""

from __future__ import annotations

import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ─── Data shapes ──────────────────────────────────────────────────────

@dataclass
class KbEntry:
    """One Q&A pair in the corpus."""

    id: str
    category: str
    question: str
    answer: str
    aliases: list[str]
    keywords: list[str]

    @classmethod
    def from_dict(cls, d: dict) -> "KbEntry":
        return cls(
            id=str(d["id"]),
            category=str(d.get("category", "general")).lower(),
            question=str(d["question"]).strip(),
            answer=str(d["answer"]).strip(),
            aliases=[str(a).strip() for a in d.get("aliases", []) if a],
            keywords=[str(k).strip().lower() for k in d.get("keywords", []) if k],
        )

    def searchable_text(self) -> str:
        """Concatenate every phrasing that should match this entry.

        The question itself is weighted heaviest by repetition; aliases
        and keywords come once each. The answer is NOT included — we
        want matches against how users *ask*, not what we *answer*.
        """
        parts = [self.question, self.question]  # 2x weight on canonical
        parts.extend(self.aliases)
        parts.extend(self.keywords)
        return " ".join(parts)


@dataclass
class SearchResult:
    entry: KbEntry
    score: float


# ─── KB loading ───────────────────────────────────────────────────────

def load_kb(kb_dir: Path) -> list[KbEntry]:
    """Walk `kb_dir` and load every .jsonl file.

    Each line must be a JSON object matching KbEntry.from_dict. Lines
    that don't parse are skipped with a warning — we never want a
    malformed entry to bring the whole service down.
    """
    entries: list[KbEntry] = []
    seen_ids: set[str] = set()
    for path in sorted(kb_dir.rglob("*.jsonl")):
        with path.open(encoding="utf-8") as f:
            for n, line in enumerate(f, start=1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    data = json.loads(line)
                    entry = KbEntry.from_dict(data)
                except Exception as err:  # noqa: BLE001
                    print(
                        f"[kb] skipping {path.name}:{n} — {err}"
                    )
                    continue
                if entry.id in seen_ids:
                    print(f"[kb] duplicate id {entry.id} in {path.name}:{n}")
                    continue
                seen_ids.add(entry.id)
                entries.append(entry)
    return entries


# ─── Retriever ────────────────────────────────────────────────────────

# Light tokeniser: lowercase, strip punctuation, split on whitespace.
# sklearn's defaults already lowercase + token-pattern-split, but we
# normalise a few common crypto unicode chars first.
_NORMALISE = str.maketrans(
    {
        "'": "'",
        "'": "'",
        "—": "-",
        "–": "-",
    }
)


def _preprocess(text: str) -> str:
    t = text.translate(_NORMALISE).lower()
    # Collapse repeated whitespace
    t = re.sub(r"\s+", " ", t).strip()
    return t


class TfidfRetriever:
    """Fit-once, search-many TF-IDF retriever."""

    def __init__(
        self,
        entries: list[KbEntry],
        vectorizer: TfidfVectorizer | None = None,
    ) -> None:
        self.entries = list(entries)
        if not self.entries:
            # Empty corpus is a valid state (nothing to match). Build a
            # trivial vectorizer so .search() returns empty rather than
            # crashing.
            self._vectorizer = TfidfVectorizer()
            self._matrix = None
            return
        self._vectorizer = vectorizer or TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
            sublinear_tf=True,
            token_pattern=r"[a-zA-Z][a-zA-Z0-9\-]{1,}",
        )
        docs = [_preprocess(e.searchable_text()) for e in self.entries]
        self._matrix = self._vectorizer.fit_transform(docs)

    # ─── Public API ───────────────────────────────────────────────

    @property
    def size(self) -> int:
        return len(self.entries)

    @property
    def vocab_size(self) -> int:
        try:
            return len(self._vectorizer.vocabulary_)  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            return 0

    def search(self, query: str, top_k: int = 3) -> list[SearchResult]:
        if self._matrix is None or not query.strip():
            return []
        q_vec = self._vectorizer.transform([_preprocess(query)])
        sims = cosine_similarity(q_vec, self._matrix)[0]
        # argpartition for top-k without full sort — cheap at small N
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

    # ─── Persistence ──────────────────────────────────────────────

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(
                {
                    "entries": self.entries,
                    "vectorizer": self._vectorizer,
                    "matrix": self._matrix,
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    @classmethod
    def load(cls, path: Path) -> "TfidfRetriever":
        with path.open("rb") as f:
            blob = pickle.load(f)
        obj = cls.__new__(cls)
        obj.entries = blob["entries"]
        obj._vectorizer = blob["vectorizer"]
        obj._matrix = blob["matrix"]
        return obj


def build_retriever_from_kb(kb_dir: Path) -> TfidfRetriever:
    entries = load_kb(kb_dir)
    return TfidfRetriever(entries)
