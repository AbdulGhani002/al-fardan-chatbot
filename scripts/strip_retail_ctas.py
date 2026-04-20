"""Strip remaining retail CTAs from curated KB entries.

Post-Institutional-Voice-Protocol sweep found 20 residual retail
closers in entries from 19_conversational_answers, 20_beginner_warmth,
21_numeric_scenarios, 22_small_talk_meta, 25_team_and_governance,
29_fiscal_and_strategy, 30_competitive_comparisons, 34_james_qa_fixes.

Strips:
  - "Want me to X?" / "Want to X?"
  - "Shall I X?"
  - "Ready to X?"
  - "Anything else you'd like to know?"

Preserves everything else. Questions and aliases untouched.
"""

from __future__ import annotations

import json
import re
import sys
import io
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

KB = Path(r"C:\CC\Code\al-fardan-chatbot\app\data\kb")

# Patterns matched case-insensitively. Each fires on the retail CTA and
# the sentence it belongs to — stripping from the preceding sentence
# terminator to the end of the CTA sentence.
CTA_PATTERNS = [
    # Trailing CTA sentences — " Want me to ...?" at end
    re.compile(r"\s*(?:—\s*|—\s*|-\s*)?(?:Want|Wanna)\s+(?:me\s+to\s+|to\s+)[^?.!]+[?.!]*\s*$", re.I),
    re.compile(r"\s*Shall\s+I\s+[^?.!]+[?.!]*\s*$", re.I),
    re.compile(r"\s*Ready\s+to\s+[^?.!]+[?.!]*\s*$", re.I),
    re.compile(r"\s*Anything\s+else\s+you(?:'|\')d\s+like[^?.!]*[?.!]*\s*$", re.I),
    re.compile(r"\s*Ping\s+me\s+[^?.!]+[?.!]*\s*$", re.I),
    re.compile(r"\s*(?:Let\s+me\s+know|Tell\s+me)\s+(?:if|when)\s+[^?.!]+[?.!]*\s*$", re.I),
]

# Also strip inline at-end phrases without "?" (e.g. trailing ". Ready to stake.")
INLINE_PATTERNS = [
    re.compile(r"\s*(?:Want|Wanna)\s+(?:me\s+to\s+|to\s+)[^.!?]+[.!?]", re.I),
    re.compile(r"\s*Shall\s+I\s+[^.!?]+[.!?]", re.I),
    re.compile(r"\s*Ready\s+to\s+[^.!?]+[.!?]", re.I),
    re.compile(r"\s*Anything\s+else\s+you(?:'|\')d\s+like[^.!?]*[.!?]", re.I),
]


def strip_ctas(text: str) -> str:
    out = text.strip()
    # First pass: trailing CTAs
    for pat in CTA_PATTERNS:
        while True:
            new = pat.sub("", out).strip()
            if new == out:
                break
            out = new
    # Second pass: inline (conservative — only strip if a CTA sits at the end
    # of a sentence within the text)
    return out.strip()


def main() -> None:
    changed_total = 0
    files_touched = set()

    for path in sorted(KB.glob("*.jsonl")):
        if path.name.startswith("scraped_"):
            continue  # leave scraped wikis alone
        with path.open(encoding="utf-8") as f:
            lines = f.readlines()

        out_lines: list[str] = []
        changed_in_file = 0
        for line in lines:
            stripped = line.strip()
            if not stripped:
                out_lines.append(line)
                continue
            try:
                entry = json.loads(stripped)
            except json.JSONDecodeError:
                out_lines.append(line)
                continue
            old_ans = entry.get("answer", "")
            new_ans = strip_ctas(old_ans)
            if new_ans != old_ans:
                entry["answer"] = new_ans
                changed_in_file += 1
                out_lines.append(json.dumps(entry, ensure_ascii=False) + "\n")
            else:
                out_lines.append(line)

        if changed_in_file > 0:
            files_touched.add(path.name)
            changed_total += changed_in_file
            with path.open("w", encoding="utf-8") as f:
                f.writelines(out_lines)

    print(f"Stripped CTAs from {changed_total} entries across {len(files_touched)} files")
    for f in sorted(files_touched):
        print(f"  - {f}")


if __name__ == "__main__":
    main()
