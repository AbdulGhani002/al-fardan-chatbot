"""Scraper for ethereum.org/en/ — basics, staking, DeFi, developer docs."""

from pathlib import Path

from .common import crawl, write_scraped_jsonl

SEED = "https://ethereum.org/en/learn/"
ALLOW = "https://ethereum.org/en/"
MAX_PAGES = 25


def run(out_dir: Path) -> int:
    pages = crawl(SEED, ALLOW, max_pages=MAX_PAGES)
    write_scraped_jsonl(pages, out_dir / "scraped_ethereum_org.jsonl")
    return len(pages)


if __name__ == "__main__":
    n = run(Path(__file__).resolve().parent.parent / "data" / "kb")
    print(f"[ethereum_org] wrote {n} pages")
