"""Build the retrieval index from the KB.

Preference order:
  1. Dense (fastembed / BAAI/bge-small-en-v1.5) — best semantic quality
  2. TF-IDF — fallback if fastembed isn't available

Run after editing the KB or re-running scrapers:
    python -m app.train.build_index
"""

from __future__ import annotations

import time

from ..config import settings
from ..retrieval import dense as dense_retrieval
from ..retrieval.tfidf import build_retriever_from_kb


def main() -> None:
    t0 = time.time()

    if dense_retrieval.available():
        try:
            retriever = dense_retrieval.build_dense_from_kb(settings.kb_dir)
            retriever.save(settings.index_path)
            took_ms = int((time.time() - t0) * 1000)
            print(
                f"[build_index] DENSE index built: {retriever.size} entries, "
                f"dim={retriever.vocab_size}, "
                f"saved to {settings.index_path} in {took_ms}ms"
            )
            return
        except Exception as err:  # noqa: BLE001
            print(
                f"[build_index] dense build failed ({err}); "
                f"falling back to TF-IDF"
            )

    retriever = build_retriever_from_kb(settings.kb_dir)
    retriever.save(settings.index_path)
    took_ms = int((time.time() - t0) * 1000)
    print(
        f"[build_index] TF-IDF index built: {retriever.size} entries, "
        f"{retriever.vocab_size} tokens, "
        f"saved to {settings.index_path} in {took_ms}ms"
    )


if __name__ == "__main__":
    main()
