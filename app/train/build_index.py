"""Build the TF-IDF index from the KB + any scraped content.

Run this after editing the KB or re-running scrapers:
    python -m app.train.build_index
"""

from __future__ import annotations

import time

from ..config import settings
from ..retrieval.tfidf import build_retriever_from_kb


def main() -> None:
    t0 = time.time()
    retriever = build_retriever_from_kb(settings.kb_dir)
    retriever.save(settings.index_path)
    took_ms = int((time.time() - t0) * 1000)
    print(
        f"[build_index] {retriever.size} entries, "
        f"{retriever.vocab_size} tokens, "
        f"saved to {settings.index_path} in {took_ms}ms"
    )


if __name__ == "__main__":
    main()
