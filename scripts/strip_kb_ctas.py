"""One-shot cleanup: strip trailing sales-CTA questions from KB answers.

Patterns like "Want it?", "Interested?", "Sound good?" are unprofessional
for an institutional client-facing bot. The LLM layer would normally
rewrite these away, but:
  1. High-confidence hits (score >= 0.55) bypass the LLM and serve the
     KB answer verbatim.
  2. If the LLM is down, fallback is also the verbatim KB answer.

So we clean the KB at source.

Run with: python -m scripts.strip_kb_ctas
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


# Conservative — only strips patterns we've observed as salesy tails.
# Must appear at the END of the answer, optionally preceded by whitespace.
# Order matters: longer/more-specific patterns first so they take priority.
TAIL_CTA_PATTERNS = [
    r"\s*Want one scheduled\?\s*$",
    r"\s*Want us to [^?]{1,60}\?\s*$",
    r"\s*Want the full [^?]{1,60}\?\s*$",
    r"\s*Want it delivered\?\s*$",
    r"\s*Want to see [^?]{1,60}\?\s*$",
    r"\s*Want a [a-zA-Z ]{1,30}\?\s*$",
    r"\s*Want it\?\s*$",
    r"\s*Want one\?\s*$",
    r"\s*Want this\?\s*$",
    r"\s*Still interested\?\s*$",
    r"\s*Interested\?\s*$",
    r"\s*Sound good\?\s*$",
    r"\s*Sound workable\?\s*$",
    r"\s*Ready to [a-zA-Z ]{1,30}\?\s*$",
    r"\s*Shall we [a-zA-Z ]{1,30}\?\s*$",
]


def clean_answer(text: str) -> str:
    """Strip all matching CTA tails, iterating until stable."""
    prev = None
    while prev != text:
        prev = text
        for pat in TAIL_CTA_PATTERNS:
            text = re.sub(pat, "", text, flags=re.IGNORECASE)
    return text.strip()


def process_file(path: Path) -> tuple[int, int]:
    """Returns (entries_changed, total_entries)."""
    raw = path.read_text(encoding="utf-8")
    out_lines: list[str] = []
    changed = 0
    total = 0
    for line in raw.splitlines():
        if not line.strip():
            out_lines.append(line)
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            # Preserve malformed lines as-is — we don't want to drop data.
            out_lines.append(line)
            continue
        total += 1
        ans = obj.get("answer", "")
        if isinstance(ans, str):
            new_ans = clean_answer(ans)
            if new_ans != ans:
                obj["answer"] = new_ans
                changed += 1
        out_lines.append(json.dumps(obj, ensure_ascii=False))
    # Preserve trailing newline if the original file had one.
    trailing = "\n" if raw.endswith("\n") else ""
    path.write_text("\n".join(out_lines) + trailing, encoding="utf-8")
    return changed, total


def main() -> int:
    kb_dir = Path(__file__).resolve().parent.parent / "app" / "data" / "kb"
    if not kb_dir.is_dir():
        print(f"[strip] KB dir not found: {kb_dir}", file=sys.stderr)
        return 1
    grand_changed = 0
    grand_total = 0
    for p in sorted(kb_dir.glob("*.jsonl")):
        changed, total = process_file(p)
        grand_changed += changed
        grand_total += total
        if changed:
            print(f"[strip] {p.name}: {changed}/{total} entries cleaned")
    print(f"[strip] done — {grand_changed}/{grand_total} entries modified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
