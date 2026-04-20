"""Sentence-level extraction — keep answers punchy.

Long KB entries (especially scraped Wikipedia / ethereum.org content)
are paragraph-length. Dumping the whole paragraph into a chat bubble
next to Safiya's photo feels robotic. This module picks the 1-2
sentences most relevant to what the user actually asked and returns
those, preserving original order.

No ML — just token overlap scoring + a tiny stopword list. Good
enough for FAQ-style domains where query terms usually appear in
the relevant sentence.
"""

from __future__ import annotations

import re


# Lightweight stopword list — keep only the ones that frequently steal
# signal (articles, fillers). Domain terms (crypto, custody, etc.)
# remain.
_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "of", "to", "in", "is", "are",
    "for", "on", "with", "as", "at", "by", "it", "be", "that", "this",
    "from", "not", "can", "i", "you", "your", "our", "we", "they",
    "them", "their", "have", "has", "was", "were", "will", "would",
    "but", "which", "who", "when", "what", "where", "why", "how",
    "any", "some", "into", "per", "its", "do", "does", "did", "am",
    "me", "my", "us", "he", "she", "if", "so", "also", "just", "only",
})

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-']{1,}")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")

# Abbreviations whose trailing period must NOT be treated as a sentence
# boundary. Critical for institutional copy that routinely names
# "Sheikh Dr. Tariq", "Mr. Al Fardan", "e.g.", "U.S.", "Ltd.", etc.
# We temporarily swap the period for a sentinel, split, then restore.
_ABBREV_TOKENS = (
    "Dr.", "Mr.", "Mrs.", "Ms.", "Prof.", "Sr.", "Jr.", "St.",
    "Ltd.", "Inc.", "Co.", "Corp.", "LLC.",
    "vs.", "etc.", "e.g.", "i.e.", "approx.", "No.",
    "U.S.", "U.A.E.", "U.K.",
    "L.L.C.",
)
_PERIOD_SENTINEL = "\u0001"  # ASCII SOH — never appears in real copy


def _protect_abbrevs(text: str) -> str:
    """Swap the period of common abbreviations with a sentinel so the
    sentence splitter doesn't mistake them for sentence boundaries."""
    for token in _ABBREV_TOKENS:
        if token in text:
            text = text.replace(token, token.replace(".", _PERIOD_SENTINEL))
    return text


def _unprotect_abbrevs(text: str) -> str:
    return text.replace(_PERIOD_SENTINEL, ".")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _WORD_RE.findall(text)]


def _content_tokens(text: str) -> set[str]:
    return {t for t in _tokenize(text) if t not in _STOPWORDS and len(t) > 1}


def split_sentences(text: str) -> list[str]:
    """Split on sentence terminators followed by capital / digit.

    Protects common abbreviations (Dr., Mr., e.g., U.S., Ltd.) so
    ``Sheikh Dr. Tariq Al-Mahrouqi`` stays in one sentence instead of
    being cut off at ``Sheikh Dr.``.
    """
    protected = _protect_abbrevs(text.strip())
    parts = _SENTENCE_SPLIT_RE.split(protected)
    return [_unprotect_abbrevs(p).strip() for p in parts if p.strip()]


def _score_sentence(sentence: str, query_tokens: set[str]) -> float:
    """Salience = fraction of query terms present + small length penalty.

    We slightly prefer mid-length sentences (10-25 words) — short
    ones are often filler, very long ones dilute the answer.
    """
    s_tokens = _content_tokens(sentence)
    if not query_tokens:
        # No signal from the query — just prefer earlier sentences
        return 0.0
    overlap = len(query_tokens & s_tokens)
    if overlap == 0:
        return 0.0
    # Base score: overlap fraction against the query
    base = overlap / max(1, len(query_tokens))
    # Length bonus: smooth function peaking around 12-18 words
    n_words = len(sentence.split())
    if n_words < 4:
        length_factor = 0.5
    elif n_words < 8:
        length_factor = 0.8
    elif n_words <= 25:
        length_factor = 1.0
    elif n_words <= 40:
        length_factor = 0.85
    else:
        length_factor = 0.65
    return base * length_factor


def extract_best_sentences(
    answer: str,
    query: str,
    *,
    n: int = 2,
    max_chars: int = 360,
    always_keep_first: bool = True,
) -> str:
    """Return the best `n` sentences for `query` from `answer`.

    Rules:
      - If answer is already short (≤ max_chars AND ≤ 3 sentences),
        return unchanged — nothing to trim.
      - Otherwise score each sentence, take the top `n`, preserve
        their original order.
      - If `always_keep_first` is True (default), we always include
        the first sentence — it usually sets the topic.
      - If no sentence has positive score (query shares no tokens
        with the answer), we fall back to the first `n` sentences
        so the user still gets something coherent.
    """
    sentences = split_sentences(answer)
    if not sentences:
        return answer.strip()
    if len(answer) <= max_chars and len(sentences) <= 3:
        return answer.strip()

    q_tokens = _content_tokens(query)

    scored = [
        (i, _score_sentence(s, q_tokens), s)
        for i, s in enumerate(sentences)
    ]

    # Rank by score desc, index asc (tiebreak — earlier sentences win)
    ranked = sorted(scored, key=lambda row: (-row[1], row[0]))

    # Pick top-n unique, with optional first-sentence guarantee
    picked_idx: set[int] = set()
    if always_keep_first:
        picked_idx.add(0)
    for idx, score, _ in ranked:
        if score <= 0:
            break  # no more useful matches
        picked_idx.add(idx)
        if len(picked_idx) >= n:
            break

    # If we still have fewer than `n`, top up by index order (so the
    # bot doesn't suddenly answer with a 1-sentence stub when the
    # scoring couldn't separate options).
    if len(picked_idx) < min(n, len(sentences)):
        for i in range(len(sentences)):
            picked_idx.add(i)
            if len(picked_idx) >= n:
                break

    # Emit in original order
    out = [sentences[i] for i in sorted(picked_idx)]
    joined = " ".join(out).strip()

    # Hard character cap as a final safety net
    if len(joined) > max_chars + 40:
        cut = joined[:max_chars]
        # Don't cut mid-word
        if " " in cut:
            cut = cut[: cut.rfind(" ")]
        joined = cut.rstrip(" .,;:") + "…"

    return joined
