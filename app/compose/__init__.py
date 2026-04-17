"""Algorithmic response composition — no LLM, no external AI.

The retriever finds CLOSEST pre-written answers. The composer
*generates* answers by (1) extracting structured entities from the
user's question (amounts, assets, actions), (2) looking up the
relevant business facts, and (3) filling a response template shaped
by the question type.

Why this exists:
  A pure retriever fails on wording variants ("I have 0.3 BTC" vs
  "Can I borrow against 0.3 BTC" vs "Is 0.3 enough?" all need the
  same answer). Hardcoding every variant doesn't scale.

  The composer sidesteps this: extract the underlying numbers + ask,
  compare against structured facts, and compose a fresh reply. The
  same fact can render into dozens of wordings without an LLM.
"""
