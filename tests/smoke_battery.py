"""Chatbot smoke test battery.

Hits the public chatbot endpoint with a representative set of institutional
questions and flags failures: wrong-intent routes, truncated answers,
retail CTAs, or fallback/low-confidence replies.

Usage:
    python tests/smoke_battery.py [BASE_URL]

Defaults to https://80.65.211.25.sslip.io/ (the VPS deployment).
"""

from __future__ import annotations

import io
import json
import re
import sys
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import Optional

# Force UTF-8 stdout on Windows so check marks / arrows don't explode
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "https://80.65.211.25.sslip.io"


# ─── Test definitions ────────────────────────────────────────────────────

@dataclass
class TestCase:
    name: str
    message: str
    # At least one of these must appear in the reply for PASS:
    must_contain: list[str] = field(default_factory=list)
    # If any of these appear in the reply, it's a FAIL:
    must_not_contain: list[str] = field(default_factory=list)
    # Optional — expected matched_entry_id prefix (e.g. "gen-", "jq014", "im-")
    expected_entry_prefix: Optional[str] = None
    # If True, match_type must be "kb_hit" (not fallback / low_confidence)
    require_kb_hit: bool = True


# Battery grouped by topic
BATTERY: list[TestCase] = [
    # ─── Pure-question routing (the bug James hit) ─────────────────────
    TestCase(
        name="pure-question: what is staking",
        message="What is staking?",
        must_contain=["staking", "proof"],
        must_not_contain=["Please indicate the network"],
    ),
    TestCase(
        name="pure-question with greeting: Hi, what is staking",
        message="Hi, what is staking",
        must_contain=["staking"],
        must_not_contain=["Please indicate the network", "ETH, SOL, POL"],
    ),
    TestCase(
        name="pure-question: what is lending",
        message="What is lending?",
        must_contain=["lend"],
        must_not_contain=["Please specify the loan size"],
    ),
    TestCase(
        name="pure-question: what is OTC",
        message="What is OTC?",
        must_contain=["OTC"],
        must_not_contain=["Please specify the asset"],
    ),

    # ─── Heritage / company identity ──────────────────────────────────
    TestCase(
        name="heritage: how long in business",
        message="How long have you been in business?",
        must_contain=["1971"],
        must_not_contain=["Standard individual KYC", "24-48 hours"],
    ),
    TestCase(
        name="heritage: when founded",
        message="When was Al-Fardan Q9 founded?",
        must_contain=["December", "2023"],
    ),
    TestCase(
        name="heritage: who are you",
        message="Who is Al-Fardan Q9?",
        must_contain=["Al-Fardan"],
    ),

    # ─── Sharia (was truncated at "Sheikh Dr.") ───────────────────────
    TestCase(
        name="sharia-compliant",
        message="Is Al Fardan Q9 Sharia-compliant?",
        must_contain=["Tariq Al-Mahrouqi"],  # Full name, NOT truncated
        must_not_contain=[
            "Sheikh Dr. Anything",
            "Sheikh Dr.\n",
            "chaired by Sheikh Dr. Tariq Al-Mahrouqi —",  # ok if present
        ],
    ),
    TestCase(
        name="sharia: halal",
        message="Is this halal?",
        must_contain=["Murabaha"],
    ),

    # ─── Proof of Reserves ────────────────────────────────────────────
    TestCase(
        name="proof of reserves - direct",
        message="What is your Proof of Reserves?",
        must_contain=["215"],
        must_not_contain=["Sheikh", "Sharia Supervisory Board"],
    ),
    TestCase(
        name="proof of reserves - verify",
        message="How do I verify your Proof of Reserves?",
        must_contain=["reserves"],
    ),

    # ─── Sequential context bug (PoR after Sharia) ────────────────────
    # (Two requests on the same session — second must NOT leak from first.)
    TestCase(
        name="session-context sharia first",
        message="Is Al Fardan Q9 Sharia-compliant?",
        must_contain=["Tariq"],
    ),
    # Next one uses same session token (set below in run())

    # ─── Regulatory framework ─────────────────────────────────────────
    TestCase(
        name="vara-license",
        message="What is your VARA license number?",
        must_contain=["VL/23/10/002"],
    ),
    TestCase(
        name="dfsa-question (negative — we are NOT DFSA)",
        message="Are you DFSA licensed?",
        must_not_contain=["Yes, we hold a DFSA license"],
    ),
    TestCase(
        name="difc-registration",
        message="Are you DIFC registered?",
        must_contain=["5605"],
    ),
    TestCase(
        name="cbuae-connection",
        message="Are you CBUAE regulated?",
        must_contain=["1971"],
    ),

    # ─── Retail-CTA regression (must NOT appear anywhere) ─────────────
    TestCase(
        name="retail-cta: what are fees",
        message="What are the custody fees?",
        must_contain=["fee"],
        must_not_contain=[
            "Anything else you'd like to know",
            "Want me to",
            "Shall I",
            "Ready to",
            "Happy to help",
        ],
    ),

    # ─── Composer: EXPLICIT intent-to-do (should fire clarification) ──
    TestCase(
        name="composer: i want to stake",
        message="I want to stake ETH",
        must_contain=["network", "size"],  # Should ask clarifier
        require_kb_hit=False,  # scripted / composer reply
    ),

    # ─── Greeting / intent classification ─────────────────────────────
    TestCase(
        name="greeting: hello",
        message="hello",
        must_contain=["Safiya"],
        require_kb_hit=False,
    ),
    TestCase(
        name="greeting: assalamu alaikum",
        message="assalamu alaikum",
        must_contain=["Safiya"],
        require_kb_hit=False,
    ),

    # ─── Complex institutional questions ──────────────────────────────
    TestCase(
        name="custody location",
        message="Where are my assets stored?",
        must_contain=["Fireblocks"],
    ),
    TestCase(
        name="insurance coverage",
        message="What does your Lloyd's insurance cover?",
        must_contain=["Lloyd"],
    ),
    TestCase(
        name="minimum investment",
        message="What is the minimum investment?",
        must_contain=["$3,000"],
    ),
    TestCase(
        name="OTC fees",
        message="What are your OTC fees?",
        must_contain=["OTC"],
    ),
    TestCase(
        name="staking rewards frequency",
        message="When are staking rewards paid?",
        must_contain=["daily"],
    ),
    TestCase(
        name="segregated custody",
        message="Are my assets segregated?",
        must_contain=["segregat"],
    ),

    # ─── Edge: short follow-ups that should hit affirmation path ──────
    TestCase(
        name="affirmation: yes",
        message="yes",
        require_kb_hit=False,
        must_not_contain=["Please indicate the network"],
    ),

    # ─── Abbreviation regression (Sheikh Dr., Mr., U.S.) ──────────────
    TestCase(
        name="sharia chair name not truncated",
        message="Who chairs your Sharia board?",
        must_contain=["Tariq"],
        must_not_contain=["Sheikh Dr. Anything"],
    ),
]


# ─── Runner ────────────────────────────────────────────────────────────

def post_chat(session_token: str, message: str, timeout: int = 30) -> dict:
    url = f"{BASE_URL}/chat"
    payload = {"session_token": session_token, "message": message}
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        return {"error": f"HTTP {e.code}", "body": body}
    except Exception as e:
        return {"error": str(e)}


def run() -> None:
    # Session 1: most cases
    base_session = f"smoke-{int(time.time())}"
    results: list[tuple[TestCase, dict, str, list[str]]] = []

    # Separate session for sequence-dependent tests so state doesn't pollute
    session_counter = 0

    for tc in BATTERY:
        session_counter += 1
        session_token = f"{base_session}-{session_counter:03d}"
        resp = post_chat(session_token, tc.message)
        verdict = "PASS"
        issues: list[str] = []

        if "error" in resp:
            verdict = "FAIL"
            issues.append(f"Request error: {resp['error']}")
        else:
            reply = resp.get("reply", "") or ""
            mt = resp.get("match_type", "")
            # require_kb_hit check
            if tc.require_kb_hit and mt not in ("kb_hit", "scripted"):
                # scripted is OK if the case doesn't strictly need kb
                verdict = "FAIL"
                issues.append(f"match_type={mt} (expected kb_hit)")

            # must_contain
            for phrase in tc.must_contain:
                if phrase.lower() not in reply.lower():
                    verdict = "FAIL"
                    issues.append(f"missing required phrase: {phrase!r}")

            # must_not_contain
            for phrase in tc.must_not_contain:
                if phrase.lower() in reply.lower():
                    verdict = "FAIL"
                    issues.append(f"unwanted phrase: {phrase!r}")

            # Smell check: truncation / bad sentence boundaries
            truncation_patterns = [
                r"\bDr\.\s*$",  # ends with "Dr."
                r"\bMr\.\s*$",
                r"\bMrs\.\s*$",
                r"\be\.g\.\s*$",
                r"\bi\.e\.\s*$",
            ]
            for pat in truncation_patterns:
                if re.search(pat, reply):
                    verdict = "FAIL"
                    issues.append(f"truncation detected: reply ends with {pat!r}")

        results.append((tc, resp, verdict, issues))

    # Sequence test: Sharia followed by PoR on SAME session
    seq_session = f"{base_session}-seq"
    r1 = post_chat(seq_session, "Is Al Fardan Q9 Sharia-compliant?")
    r2 = post_chat(seq_session, "What is your Proof of Reserves?")
    seq_tc = TestCase(
        name="sequence: PoR after Sharia (same session)",
        message="What is your Proof of Reserves?",
        must_contain=["reserves"],
        must_not_contain=["Sharia", "Tariq"],
    )
    seq_verdict = "PASS"
    seq_issues: list[str] = []
    if "error" in r2:
        seq_verdict = "FAIL"
        seq_issues.append(f"Request error: {r2['error']}")
    else:
        reply2 = (r2.get("reply") or "")
        for p in seq_tc.must_contain:
            if p.lower() not in reply2.lower():
                seq_verdict = "FAIL"
                seq_issues.append(f"missing: {p!r}")
        for p in seq_tc.must_not_contain:
            if p.lower() in reply2.lower():
                seq_verdict = "FAIL"
                seq_issues.append(f"leaked from previous turn: {p!r}")
    results.append((seq_tc, r2, seq_verdict, seq_issues))

    # ─── Report ────────────────────────────────────────────────────────
    passed = sum(1 for _, _, v, _ in results if v == "PASS")
    failed = sum(1 for _, _, v, _ in results if v == "FAIL")
    print(f"\n{'='*72}")
    print(f"CHATBOT SMOKE BATTERY — {passed}/{len(results)} PASS, {failed} FAIL")
    print(f"Endpoint: {BASE_URL}")
    print(f"{'='*72}")

    for tc, resp, verdict, issues in results:
        status = "✓" if verdict == "PASS" else "✗"
        print(f"\n{status} {verdict}: {tc.name}")
        print(f"   Q: {tc.message!r}")
        reply_snippet = (resp.get("reply") or "[no reply]")[:160]
        match_info = ""
        if "matched_entry_id" in resp:
            score = resp.get("match_score")
            score_s = f"{score:.3f}" if isinstance(score, (int, float)) else "n/a"
            match_info = f" [{resp.get('match_type','?')}/{resp.get('matched_entry_id','?')}/{score_s}]"
        print(f"   A:{match_info} {reply_snippet}")
        if issues:
            for issue in issues:
                print(f"   ! {issue}")

    # Exit code reflects pass/fail
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    run()
