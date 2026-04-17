"""Scraper for the Al-Fardan demo site (dev-al-fardan.vercel.app)."""

from pathlib import Path

from .common import crawl, write_scraped_jsonl

SEED = "https://dev-al-fardan.vercel.app/"
ALLOW = "https://dev-al-fardan.vercel.app/"
MAX_PAGES = 40


def run(out_dir: Path) -> int:
    pages = crawl(SEED, ALLOW, max_pages=MAX_PAGES)
    write_scraped_jsonl(pages, out_dir / "scraped_al_fardan.jsonl")
    return len(pages)


if __name__ == "__main__":
    n = run(Path(__file__).resolve().parent.parent / "data" / "kb")
    print(f"[al_fardan] wrote {n} pages")
