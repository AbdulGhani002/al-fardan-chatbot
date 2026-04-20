"""Emotion + urgency detection from the raw user query.

Rule-based, zero ML — a small set of regexes that classify the
message's affect. The composer + intent layer use the classification
to:

  * detect frustration → soften tone + offer human handoff
  * detect urgency    → surface the 24/7 support line immediately
  * detect excitement → mirror the energy (short, enthusiastic reply)
  * detect confusion  → slow down, simplify, avoid jargon

Returns a Mood dataclass so callers can branch cleanly without
parsing free-form strings.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Mood:
    frustrated: bool = False
    urgent: bool = False
    excited: bool = False
    confused: bool = False

    @property
    def has_any(self) -> bool:
        return self.frustrated or self.urgent or self.excited or self.confused

    def primary(self) -> str:
        """Return the single most-salient tag, in priority order."""
        if self.frustrated:
            return "frustrated"
        if self.urgent:
            return "urgent"
        if self.confused:
            return "confused"
        if self.excited:
            return "excited"
        return "neutral"


# ─── Pattern bank ───────────────────────────────────────────────────
# Kept conservative — we want high precision; we'd rather miss a
# subtle frustration than falsely flag a normal enthusiastic question.

_FRUSTRATED_RE = re.compile(
    r"""
    \b(
      wtf|fuck|shit|damn|bullshit|
      this\ (sucks|is\ stupid|is\ broken|is\ useless|is\ annoying|is\ a\ joke|is\ trash)|
      not\ working|doesn'?t\ work|
      garbage|waste\ of\ time|useless|terrible|awful|horrible|
      ridiculous|hate\ (this|it|you)|
      are\ you\ (serious|kidding)|
      for\ the\ love\ of|
      fed\ up|frustrat(ed|ing)|annoyed|angry|
      why\ (are\ you|is\ this)\ so|
      pissed|
      \?{3,}                 # "??? " is annoyed, "!!" alone is excited
    )\b
    """,
    re.I | re.VERBOSE,
)

_URGENT_RE = re.compile(
    r"""
    \b(
      urgent|urgently|asap|right\ now|immediately|
      emergency|critical|right\ away|
      help\ me\ now|stuck|locked\ out|
      can'?t\ (access|login|withdraw)|
      account\ (hacked|stolen|compromised)|
      transaction\ (stuck|failed|missing)|
      money\ (missing|gone|lost)
    )\b
    """,
    re.I | re.VERBOSE,
)

_EXCITED_RE = re.compile(
    r"""
    \b(
      amazing|awesome|brilliant|fantastic|excellent|
      love\ (it|this)|perfect|
      let'?s\ (go|do\ it)|ready\ to\ (go|start|invest)|
      can'?t\ wait|looking\ forward
    )\b|
    ❤️|🔥|🚀|💯|👏|🙌|✨
    """,
    re.I | re.VERBOSE,
)

_CONFUSED_RE = re.compile(
    r"""
    \b(
      i\ don'?t\ (understand|get\ it|follow|know)|
      confused|confusing|complicated|
      what\ does\ that\ mean|
      this\ is\ (hard|complex)|
      too\ (much|complicated|technical)|
      explain\ (again|slowly|simply)|
      can\ you\ (rephrase|simplify|break\ this\ down)|
      layman|eli5|like\ i'?m\ 5|in\ plain\ english
    )\b
    """,
    re.I | re.VERBOSE,
)


def detect(text: str) -> Mood:
    """Classify the affect in `text`. Cheap; safe to call on every
    inbound message."""
    if not text:
        return Mood()
    return Mood(
        frustrated=bool(_FRUSTRATED_RE.search(text)),
        urgent=bool(_URGENT_RE.search(text)),
        excited=bool(_EXCITED_RE.search(text)),
        confused=bool(_CONFUSED_RE.search(text)),
    )


# ─── Pre-reply tone adjustments ─────────────────────────────────────

def acknowledgment(mood: Mood) -> str:
    """A short acknowledgement prepended to the main reply so the user
    feels heard. Empty string if mood is neutral. Institutional
    register — measured, not effusive, no retail warmth."""
    if mood.frustrated:
        return (
            "Noted — apologies for the inconvenience. "
        )
    if mood.urgent:
        return (
            "Understood — this appears time-sensitive. "
            "The 24/7 desk (institutional@alfardanq9.com) is available for immediate escalation. "
        )
    if mood.confused:
        return "Understood. "
    if mood.excited:
        return ""
    return ""


def escalation_actions(mood: Mood) -> list[dict]:
    """If the mood suggests the bot can't help alone (frustrated,
    urgent), surface human handoff buttons first."""
    if mood.frustrated or mood.urgent:
        return [
            {
                "label": "Email Support",
                "url": "mailto:institutional@alfardanq9.com",
                "kind": "link",
            },
            {
                "label": "Talk to Layla (OTC)",
                "url": "mailto:otc@alfardanq9.com",
                "kind": "link",
            },
        ]
    return []
