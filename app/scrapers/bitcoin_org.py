"""Scraper for bitcoin.org — Bitcoin basics, how-it-works, FAQ, brochure.

bitcoin.org content is released under the Bitcoin License (effectively
public-domain / CC0). Safe for FAQ retrieval use.
See https://github.com/bitcoin-dot-org/Bitcoin.org/blob/master/LICENSE.
"""

from pathlib import Path

from .common import (
    chunks_to_kb_entries,
    crawl_chunked,
    write_jsonl,
)


SEEDS = [
    "https://bitcoin.org/en/how-it-works",
    "https://bitcoin.org/en/faq",
    "https://bitcoin.org/en/getting-started",
    "https://bitcoin.org/en/you-need-to-know",
    "https://bitcoin.org/en/secure-your-wallet",
    "https://bitcoin.org/en/protect-your-privacy",
    "https://bitcoin.org/en/choose-your-wallet",
    "https://bitcoin.org/en/bitcoin-for-individuals",
    "https://bitcoin.org/en/bitcoin-for-businesses",
    "https://bitcoin.org/en/vocabulary",
    "https://bitcoin.org/en/innovation",
]
ALLOW = ["https://bitcoin.org/en/"]
MAX_PAGES = 30


def run(out_dir: Path) -> int:
    chunks = crawl_chunked(
        SEEDS,
        ALLOW,
        source="bitcoin.org",
        max_pages=MAX_PAGES,
    )
    entries = chunks_to_kb_entries(chunks)
    write_jsonl(entries, out_dir / "scraped_bitcoin_org.jsonl")
    return len(entries)


if __name__ == "__main__":
    n = run(Path(__file__).resolve().parent.parent / "data" / "kb")
    print(f"[bitcoin_org] wrote {n} chunks")
