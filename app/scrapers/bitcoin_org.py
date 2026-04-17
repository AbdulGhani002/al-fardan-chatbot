"""Scraper for bitcoin.org/en/ — fundamentals, FAQ, developer guides."""

from pathlib import Path

from .common import crawl, write_scraped_jsonl

SEED = "https://bitcoin.org/en/how-it-works"
ALLOW = "https://bitcoin.org/en/"
MAX_PAGES = 25


def run(out_dir: Path) -> int:
    pages = crawl(SEED, ALLOW, max_pages=MAX_PAGES)
    write_scraped_jsonl(pages, out_dir / "scraped_bitcoin_org.jsonl")
    return len(pages)


if __name__ == "__main__":
    n = run(Path(__file__).resolve().parent.parent / "data" / "kb")
    print(f"[bitcoin_org] wrote {n} pages")
