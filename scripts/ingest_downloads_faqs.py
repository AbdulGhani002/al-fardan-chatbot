"""Ingest the staged Downloads/chatbot.zip + the 1000 Q&A docx +
the Gold Reserve docx into the KB.

Sources land in `scripts/incoming_faqs/` (staged manually so the
script is deterministic). For each file we:

  1. Parse Q&A pairs out of the source (txt patterns / docx
     paragraphs).
  2. Normalise each pair into the KB entry shape used by the rest
     of the bot:
       { id, category, question, answer, aliases?, keywords? }
  3. Deduplicate against the existing KB by canonicalised question
     hash so we don't pollute the index with re-imports.
  4. Emit a single sharded JSONL per source under `app/data/kb/`
     so the existing index builder picks it up on next run.

The script is idempotent — re-running it with the same incoming
files produces the same output and skips already-imported entries
(everything is tagged by source id).

After this script finishes, run:
    python -m app.train.build_index
…to rebuild the dense/TF-IDF index.
"""

from __future__ import annotations

import json
import re
import sys
from hashlib import sha1
from pathlib import Path
from typing import Iterable

# Defensive: `docx` (python-docx) is optional. We hard-fail with a
# clear message if a .docx source is staged but the lib isn't on
# the venv.
try:
    import docx  # type: ignore[import-not-found]
except Exception:  # noqa: BLE001
    docx = None  # type: ignore[assignment]


REPO = Path(__file__).resolve().parent.parent
INCOMING = REPO / "scripts" / "incoming_faqs"
KB_DIR = REPO / "app" / "data" / "kb"


def canonicalize_question(q: str) -> str:
    """Lowercase + strip punctuation + collapse whitespace so we can
    spot near-duplicates without semantic search."""
    q = q.lower()
    q = re.sub(r"[^a-z0-9\s]+", " ", q)
    q = re.sub(r"\s+", " ", q).strip()
    return q


def question_hash(q: str) -> str:
    return sha1(canonicalize_question(q).encode("utf-8")).hexdigest()[:12]


def load_existing_question_hashes() -> set[str]:
    """Read every existing KB jsonl and build a set of canonical
    question hashes so we can skip duplicates on import."""
    seen: set[str] = set()
    for f in sorted(KB_DIR.glob("*.jsonl")):
        with f.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                q = obj.get("question")
                if isinstance(q, str):
                    seen.add(question_hash(q))
    return seen


# ────────────────────────────────────────────────────────────────
# Source parsers


# Matches Q&A in the chatbot.zip .txt files. Tries three common
# patterns: "Qn: question\n\nanswer", "n. question\n\nanswer",
# and "Q: ... A: ...".
_TXT_PATTERNS = [
    # "Q1: question text\n\nanswer text\n\nQ2: ..." — the most common
    # shape across chatbot-training-1.txt, chatbot-advanced q&a.txt,
    # chatbot-better then.txt, etc.
    re.compile(
        r"Q\d+[:.]\s*(?P<q>.+?)\n+(?P<a>.+?)(?=(?:\n+Q\d+[:.])|\Z)",
        re.DOTALL,
    ),
    # "1. question text\n\nanswer text\n\n2. ..." — used by the
    # "200 additional Q&A.txt" file. The lookahead handles the
    # SECTION HEADERS that separate the topical groups in that file.
    re.compile(
        r"(?:^|\n)(?P<num>\d+)\.\s*(?P<q>.+?)\n+(?P<a>.+?)(?=(?:\n+\d+\.\s)|\Z)",
        re.DOTALL,
    ),
    # "Q: ...\nA: ..." — used by a couple of the fix files
    re.compile(
        r"(?:^|\n)Q[:.]\s*(?P<q>.+?)\n+A[:.]\s*(?P<a>.+?)(?=(?:\n+Q[:.])|\Z)",
        re.DOTALL,
    ),
]


def parse_txt(path: Path) -> list[tuple[str, str]]:
    """Extract (question, answer) pairs from a txt file. Tries each
    pattern in order and keeps the result with the most pairs — some
    files mix formats and we don't want to pick the loser."""
    text = path.read_text(encoding="utf-8", errors="ignore")
    best: list[tuple[str, str]] = []
    for pat in _TXT_PATTERNS:
        pairs: list[tuple[str, str]] = []
        for m in pat.finditer(text):
            q = m.group("q").strip()
            a = m.group("a").strip()
            # Pre-filter junk — headings, separators, empty answers.
            if len(q) < 6 or len(a) < 6:
                continue
            # Section headers like "ACCOUNT & ONBOARDING (1-50)" can
            # otherwise sneak in as answers when the regex matches a
            # boundary.
            if q.isupper() and len(q.split()) < 8:
                continue
            pairs.append((q, a))
        if len(pairs) > len(best):
            best = pairs
    return best


def parse_docx(path: Path) -> list[tuple[str, str]]:
    """Walk paragraphs in a docx file and pull out Q/A pairs.

    Three layouts we accept:

      (A) Q on one paragraph, A on the next paragraph (Q123:/A:
          shape). The Gold Reserve docx and some training docs.
      (B) Question text and answer in the SAME paragraph, split by
          a literal " Answer:" delimiter. The "1_000 Q_A pairs"
          docx uses this throughout.
      (C) Numbered list shape "1. What is X?\\nanswer paragraph"
          — handled like (A) with a numeric Q opener.
    """
    if docx is None:
        raise RuntimeError(
            "python-docx is not installed. Install with: pip install python-docx"
        )
    doc = docx.Document(str(path))
    paras = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    out: list[tuple[str, str]] = []
    pending_q: str | None = None

    q_pat = re.compile(r"^(?:Q\d+|\d+)\s*[:.\-]\s*(?P<q>.+)$", re.IGNORECASE)
    a_pat = re.compile(r"^A\s*[:.\-]\s*(?P<a>.+)$", re.IGNORECASE)
    # Layout B — a question ends in `?` and is followed inline by
    # ` Answer: ...` (case-insensitive, allow a few separators).
    inline_split_pat = re.compile(
        r"^(?P<q>.+?\?)\s*[Aa]nswer\s*[:\-]\s*(?P<a>.+)$",
        re.DOTALL,
    )

    for p in paras:
        # Layout B FIRST — single-paragraph Q+A is unambiguous and
        # cheap to spot.
        bm = inline_split_pat.match(p)
        if bm:
            q = bm.group("q").strip()
            a = bm.group("a").strip()
            if len(q) >= 6 and len(a) >= 6:
                out.append((q, a))
                pending_q = None
                continue

        # Skip section dividers (all-caps headers, short)
        if re.match(r"^[A-Z\s&(\-)0-9]+$", p) and len(p) < 80:
            pending_q = None
            continue
        m = q_pat.match(p)
        if m:
            pending_q = m.group("q").strip().rstrip(":")
            continue
        am = a_pat.match(p)
        if am and pending_q:
            out.append((pending_q, am.group("a").strip()))
            pending_q = None
            continue
        # Bare paragraph after a Q is treated as the answer (layout A)
        if pending_q:
            out.append((pending_q, p.strip()))
            pending_q = None
    return out


# ────────────────────────────────────────────────────────────────
# Categorisation heuristic — assign a coarse bucket based on the
# question's word content so the entry surfaces under a sensible
# `category` field for downstream re-ranking.


_CATEGORY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "account": ("sign up", "kyc", "onboarding", "account", "password", "2fa", "verify", "login"),
    "staking": ("stake", "stak", "unstake", "validator", "apy", "yield"),
    "lending": ("loan", "lend", "borrow", "ltv", "collateral", "murabaha", "interest"),
    "custody": ("custody", "vault", "wallet", "fireblocks", "cold storage"),
    "otc": ("otc", "block trade", "quote", "spread", "dealer"),
    "security": ("security", "secur", "insurance", "lloyd", "encrypt", "hack", "audit"),
    "compliance": ("aml", "kyc", "regulator", "vara", "difc", "licence", "license"),
    "fees": ("fee", "pricing", "cost", "commission"),
    "support": ("contact", "support", "help", "human", "agent", "hours"),
    "withdrawal": ("withdraw", "send", "transfer out", "payout"),
    "gold": ("gold", "xau", "precious metal"),
    "sharia": ("sharia", "halal", "islamic", "riba", "gharar", "aaoifi"),
}


def categorize(q: str) -> str:
    q_lower = q.lower()
    for cat, kws in _CATEGORY_KEYWORDS.items():
        for kw in kws:
            if kw in q_lower:
                return cat
    return "general"


# ────────────────────────────────────────────────────────────────
# Output emission


def write_jsonl(out_path: Path, entries: Iterable[dict]) -> int:
    n = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for e in entries:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")
            n += 1
    return n


def build_entries(
    source_id: str,
    pairs: list[tuple[str, str]],
    seen_hashes: set[str],
) -> list[dict]:
    """Convert raw (q, a) pairs into KB entries, skipping anything
    whose canonical hash already exists in the index. Mutates
    `seen_hashes` so duplicates within the same import are also
    caught."""
    out: list[dict] = []
    for i, (q, a) in enumerate(pairs):
        h = question_hash(q)
        if h in seen_hashes:
            continue
        seen_hashes.add(h)
        out.append(
            {
                "id": f"{source_id}-{i+1:04d}",
                "category": categorize(q),
                "question": q,
                "answer": a,
                "source": source_id,
            }
        )
    return out


# ────────────────────────────────────────────────────────────────
# Driver


def main() -> int:
    if not INCOMING.exists():
        print(f"[ingest] {INCOMING} does not exist. Nothing to do.")
        return 0

    seen = load_existing_question_hashes()
    print(f"[ingest] loaded {len(seen)} existing question hashes for dedup")

    total_in = 0
    total_kept = 0

    # Bucket by source-prefix so each input file produces one JSONL.
    # Prefix is sortable so the file names land in alphabetical order
    # under app/data/kb/ — convenient for inspection.
    for src in sorted(INCOMING.iterdir()):
        if src.is_dir():
            continue
        suffix = src.suffix.lower()
        try:
            if suffix == ".txt":
                pairs = parse_txt(src)
            elif suffix == ".docx":
                pairs = parse_docx(src)
            else:
                print(f"[ingest] skipping {src.name} (unsupported suffix)")
                continue
        except Exception as err:  # noqa: BLE001
            print(f"[ingest] FAILED to parse {src.name}: {err}")
            continue

        total_in += len(pairs)
        # Slug the source file name into a stable id prefix.
        slug = re.sub(r"[^a-z0-9]+", "-", src.stem.lower()).strip("-")[:40]
        entries = build_entries(slug, pairs, seen)
        if not entries:
            print(f"[ingest] {src.name}: 0 new (all duplicates)")
            continue

        out_path = KB_DIR / f"50_downloads_{slug}.jsonl"
        n = write_jsonl(out_path, entries)
        total_kept += n
        print(
            f"[ingest] {src.name}: parsed {len(pairs)}, kept {n}, "
            f"wrote {out_path.relative_to(REPO)}"
        )

    print(
        f"\n[ingest] done. parsed {total_in} pairs across staged files, "
        f"kept {total_kept} new entries (deduped against existing KB)."
    )
    print(
        "[ingest] next: python -m app.train.build_index"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
