"""Scraper for ethereum.org — basics, staking, DeFi, developer docs.

ethereum.org content is published under the MIT License (content)
with Creative Commons Attribution on most pages — safe for FAQ
retrieval use. See https://ethereum.org/en/contributing/#license.
"""

from pathlib import Path

from .common import (
    chunks_to_kb_entries,
    crawl_chunked,
    write_jsonl,
)


# Multiple seed URLs so we hit many topic hubs directly. Single-seed
# BFS previously stalled because /learn/ redirects to root.
SEEDS = [
    "https://ethereum.org/en/what-is-ethereum/",
    "https://ethereum.org/en/eth/",
    "https://ethereum.org/en/wallets/",
    "https://ethereum.org/en/staking/",
    "https://ethereum.org/en/defi/",
    "https://ethereum.org/en/stablecoins/",
    "https://ethereum.org/en/nft/",
    "https://ethereum.org/en/dao/",
    "https://ethereum.org/en/layer-2/",
    "https://ethereum.org/en/security/",
    "https://ethereum.org/en/developers/docs/intro-to-ethereum/",
    "https://ethereum.org/en/developers/docs/smart-contracts/",
    "https://ethereum.org/en/developers/docs/gas/",
    "https://ethereum.org/en/developers/docs/consensus-mechanisms/pos/",
    "https://ethereum.org/en/energy-consumption/",
    "https://ethereum.org/en/roadmap/",
]
ALLOW = ["https://ethereum.org/en/"]
MAX_PAGES = 60


def run(out_dir: Path) -> int:
    chunks = crawl_chunked(
        SEEDS,
        ALLOW,
        source="ethereum.org",
        max_pages=MAX_PAGES,
    )
    entries = chunks_to_kb_entries(chunks)
    write_jsonl(entries, out_dir / "scraped_ethereum_org.jsonl")
    return len(entries)


if __name__ == "__main__":
    n = run(Path(__file__).resolve().parent.parent / "data" / "kb")
    print(f"[ethereum_org] wrote {n} chunks")
