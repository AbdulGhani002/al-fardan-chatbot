"""Post-retrieval refinements that make replies feel human.

Three layers:
  - typos.py  — correct common crypto-term misspellings BEFORE intent
                classification + retrieval. ("lonas" → "loans",
                "etherium" → "ethereum", etc.)
  - extract.py — when a matched KB entry is long, pick only the 1-2
                 sentences most relevant to the actual user query
                 instead of dumping the whole paragraph.
  - vary.py    — multiple phrasings for scripted intents (greeting,
                 affirmation, negation, etc.) so the bot doesn't sound
                 identical every time.

All three are zero-cost at runtime — simple string/regex operations.
"""
