"""RAG prompt templates.

We keep the system prompt narrow and factual so the open-source LLM
(Qwen / Llama / Phi / etc.) stays grounded on the curated KB instead
of inventing facts about Al-Fardan Q9.

Design rules — do not relax without a Sharia / compliance review:

  1. The LLM answers ONLY from the REFERENCE block below the system
     instructions. If the references don't cover the question, the
     instructions force the model to say so and offer a human handoff.
     This prevents the bot from making up policy, LTV numbers, or
     Sharia rulings that Sheikh Dr. Al-Mahrouqi hasn't reviewed.

  2. Tone: warm but factual; no sales theatrics. The persona is
     Safiya Al Suwaidi (Client Acquisition & Growth Manager). If
     the user switches languages (Arabic, French, Spanish, etc.)
     Safiya replies in that language using the same references.

  3. Length cap: 3-5 sentences, ~120 words max. Institutional clients
     hate walls of text.

  4. No speculation on price movement, regulatory changes, or
     anything else forward-looking that isn't in the references.
"""

from __future__ import annotations

from typing import Iterable


SYSTEM_PROMPT = (
    "You are Safiya Al Suwaidi, Client Acquisition & Growth Manager at "
    "Al-Fardan Q9 — an institutional digital-asset platform based in "
    "DIFC, Dubai. You help family offices, institutional investors, and "
    "private clients understand our Custody, Staking, OTC, Lending, and "
    "Zakat services.\n\n"
    "ANSWER STYLE\n"
    "• Warm, factual, professional. Never salesy or hype-driven.\n"
    "• 3-5 sentences, ~120 words maximum.\n"
    "• If the client writes in Arabic, French, Spanish, or any other "
    "language, reply in the same language.\n"
    "• Do not say 'Assalamu alaikum' unless the client greeted you "
    "in Arabic first.\n\n"
    "GROUNDING RULES (important)\n"
    "• Use ONLY the facts in the REFERENCE block. If the references "
    "don't cover the question, say so honestly and offer a human "
    "handoff (institutional@alfardanq9.com or the Contact page).\n"
    "• Never invent LTV ratios, APYs, Sharia rulings, compliance "
    "thresholds, team members, or office locations. If a specific "
    "number isn't in the references, don't include a number.\n"
    "• Never speculate on future prices, regulatory changes, or "
    "product roadmap.\n"
    "• Don't recommend competing platforms.\n\n"
    "OUTPUT FORMAT\n"
    "• Plain prose. No bullet points unless the reference itself "
    "uses them.\n"
    "• No markdown headings.\n"
    "• No 'As Safiya' or 'As an AI' framing — just answer."
)


def build_user_prompt(
    question: str,
    references: Iterable[dict],
    session_history: list[dict] | None = None,
) -> str:
    """Assemble the RAG-grounded user turn.

    `references` is an iterable of KB entries — each a dict with
    `question`, `answer`, `id`, `category`. We format them as an
    XML-like block the model can scan easily.

    `session_history` is optional — a list of prior {role, text}
    dicts. Giving the model 2-4 turns of context helps coherence on
    follow-up questions ("what about Solana?" after an ETH question)
    without blowing the context window.
    """
    parts: list[str] = []

    # Reference block — numbered for traceability in debug logs.
    parts.append("<REFERENCES>")
    for i, ref in enumerate(references, start=1):
        ref_id = str(ref.get("id", f"ref-{i}"))
        category = str(ref.get("category", "general"))
        q = str(ref.get("question", "")).strip()
        a = str(ref.get("answer", "")).strip()
        parts.append(f"[{i}] id={ref_id} category={category}")
        if q:
            parts.append(f"    Q: {q}")
        parts.append(f"    A: {a}")
    parts.append("</REFERENCES>\n")

    # Short session history — last few turns only.
    if session_history:
        recent = session_history[-6:]
        if recent:
            parts.append("<RECENT_CHAT>")
            for msg in recent:
                role = str(msg.get("role", "user"))
                text = str(msg.get("text", "")).strip()
                if not text:
                    continue
                if role == "user":
                    parts.append(f"  Client: {text}")
                elif role == "bot":
                    parts.append(f"  Safiya: {text}")
            parts.append("</RECENT_CHAT>\n")

    parts.append(f"Client's question: {question.strip()}")
    parts.append("")
    parts.append(
        "Write Safiya's reply now. Use only the references above. "
        "If the references don't cover the question, say so honestly "
        "and offer a human contact."
    )

    return "\n".join(parts)


# Maximum references we'll include in the prompt — more = better
# grounding but bigger prompt = slower inference on CPU. 4 is a
# sweet spot for 7B models on commodity hardware.
MAX_REFERENCES = 4

# Conservative generation settings for factual answering.
DEFAULT_GEN_OPTIONS: dict = {
    "temperature": 0.3,
    "top_p": 0.9,
    "num_predict": 320,
    "repeat_penalty": 1.05,
    "stop": ["Client's question:", "<REFERENCES>", "</REFERENCES>"],
}
