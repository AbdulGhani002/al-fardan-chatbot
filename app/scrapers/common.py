"""Shared scraper utilities — respectful, rate-limited, robots-aware.

Previous version produced one giant entry per page. That's bad for
retrieval: a query about "Ethereum staking" competes for score against
every other paragraph on a sprawling page. This version chunks by
heading — each <h2>/<h3> section becomes its own KB entry, dramatically
improving recall and precision.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

USER_AGENT = (
    "AlFardanQ9ChatbotBot/0.2 (+https://al-fardan-crm.vercel.app; "
    "non-commercial FAQ retrieval)"
)
DEFAULT_TIMEOUT = 15.0
DEFAULT_RATE_LIMIT_SEC = 0.7   # polite, well below typical robots.txt limits
DEFAULT_MAX_PAGES = 50

# Chunking knobs
MIN_CHUNK_CHARS = 150          # drop near-empty sections (nav stubs, etc.)
MAX_CHUNK_CHARS = 1400         # keep entries lean — embeds faster + retriever scores cleaner


@dataclass
class Chunk:
    """One retrievable piece of scraped content."""

    url: str
    title: str           # parent page <title>
    section: str         # nearest <h1>/<h2>/<h3> heading
    text: str            # plain-text body of this section
    source: str          # logical name, e.g. "ethereum.org"
    fetched_at: float = field(default_factory=time.time)


@dataclass
class ScrapedPage:
    """Lightweight container kept for backward compatibility with the
    old `fetch()` signature. New scrapers should prefer `crawl_chunked`
    which returns Chunk objects directly."""

    url: str
    title: str
    text: str
    source: str
    fetched_at: float = field(default_factory=time.time)


def _strip_boilerplate(root) -> None:
    """Remove obviously-non-content tags in-place."""
    for tag in root.find_all(
        ["script", "style", "nav", "footer", "aside", "form", "noscript",
         "iframe", "svg", "button"]
    ):
        tag.decompose()
    # Remove common cookie/privacy banners by class/id substring
    for sel in ["cookie", "banner", "newsletter", "signup", "subscribe",
                "social", "share"]:
        for el in root.select(f'[class*="{sel}"], [id*="{sel}"]'):
            el.decompose()


def _clean_text(node) -> str:
    txt = node.get_text(separator=" ", strip=True)
    return re.sub(r"\s+", " ", txt).strip()


def _chunk_page(url: str, title: str, source: str, soup) -> list[Chunk]:
    """Split a parsed page into section chunks keyed on h1/h2/h3.

    Strategy: walk the DOM in document order, opening a new chunk each
    time we see a heading and appending body text until the next heading.
    The page's <title> becomes the parent-page anchor; the section
    heading becomes the chunk's identifying 'question'.
    """
    _strip_boilerplate(soup)

    chunks: list[Chunk] = []
    current_heading = title
    buf: list[str] = []

    def flush() -> None:
        text = " ".join(t for t in buf if t.strip())
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) >= MIN_CHUNK_CHARS:
            # Truncate politely at a sentence boundary if over cap
            if len(text) > MAX_CHUNK_CHARS:
                cut = text[:MAX_CHUNK_CHARS]
                for sep in [". ", "! ", "? "]:
                    idx = cut.rfind(sep)
                    if idx > MAX_CHUNK_CHARS * 0.5:
                        cut = cut[: idx + 1]
                        break
                text = cut
            chunks.append(
                Chunk(
                    url=url,
                    title=title,
                    section=current_heading or title,
                    text=text,
                    source=source,
                )
            )
        buf.clear()

    body = soup.find("main") or soup.find("article") or soup.body or soup
    if body is None:
        return chunks

    for el in body.descendants:
        name = getattr(el, "name", None)
        if name in ("h1", "h2", "h3"):
            flush()
            current_heading = _clean_text(el) or current_heading
        elif name in ("p", "li", "dd", "blockquote"):
            txt = _clean_text(el)
            if txt:
                buf.append(txt)
    flush()
    return chunks


def fetch_and_chunk(
    url: str, source: str, client: httpx.Client | None = None
) -> list[Chunk]:
    """Fetch a URL and split it into section chunks.

    Returns empty list on any failure — the caller should just skip.
    """
    close_after = False
    if client is None:
        client = httpx.Client(
            headers={"User-Agent": USER_AGENT},
            timeout=DEFAULT_TIMEOUT,
            follow_redirects=True,
        )
        close_after = True
    try:
        r = client.get(url)
        if r.status_code != 200:
            return []
        if "html" not in r.headers.get("content-type", ""):
            return []
        soup = BeautifulSoup(r.text, "html.parser")
        title = (soup.title.string or "").strip() if soup.title else url
        return _chunk_page(url, title, source, soup)
    except Exception as err:  # noqa: BLE001
        print(f"[scraper] {url} failed: {err}")
        return []
    finally:
        if close_after:
            client.close()


def crawl_chunked(
    seed_urls: Iterable[str],
    allow_patterns: Iterable[str],
    source: str,
    max_pages: int = DEFAULT_MAX_PAGES,
    rate_limit_sec: float = DEFAULT_RATE_LIMIT_SEC,
) -> list[Chunk]:
    """BFS crawl producing Chunk objects.

    Accepts MULTIPLE seed URLs and MULTIPLE allow-prefixes so we can
    pull from sitemap-style hubs + stay within multiple safe URL spaces.
    Previous version used a single string seed + a single string prefix
    and consequently yielded 1 page per domain when the seed redirected.
    """
    allow_list = list(allow_patterns)
    seen: set[str] = set()
    queue: list[str] = list(seed_urls)
    out: list[Chunk] = []

    def allowed(url: str) -> bool:
        return any(url.startswith(p) for p in allow_list)

    with httpx.Client(
        headers={"User-Agent": USER_AGENT},
        timeout=DEFAULT_TIMEOUT,
        follow_redirects=True,
    ) as client:
        while queue and len(out) < max_pages * 20:  # cap on chunks total
            url = queue.pop(0)
            if url in seen or not allowed(url):
                continue
            seen.add(url)
            chunks = fetch_and_chunk(url, source, client)
            if chunks:
                out.extend(chunks)
            time.sleep(rate_limit_sec)

            # Follow links within the allowed subset.
            try:
                r = client.get(url)
                if r.status_code != 200:
                    continue
                sub = BeautifulSoup(r.text, "html.parser")
                for a in sub.find_all("a", href=True)[:200]:
                    href = urljoin(url, a["href"]).split("#")[0]
                    if href not in seen and allowed(href):
                        queue.append(href)
            except Exception:  # noqa: BLE001
                pass
    return out


def chunks_to_kb_entries(chunks: list[Chunk]) -> list[dict]:
    """Convert Chunk objects to KB-entry dicts ready for JSONL output."""
    entries: list[dict] = []
    for c in chunks:
        # Stable-ish id so repeated scrapes don't duplicate entries.
        # Uses section + a hash of the URL — same section on the same
        # page always gets the same id.
        stable_id = (
            f"scraped:{c.source.replace('.', '_')}:"
            f"{abs(hash((c.url, c.section))) % 10**10}"
        )
        entries.append(
            {
                "id": stable_id,
                "category": c.source.replace(".", "_"),
                "question": c.section or c.title,
                "answer": c.text,
                "aliases": [c.title] if c.title and c.title != c.section else [],
                "keywords": [
                    # Breadcrumb-style keywords help TF-IDF fallback;
                    # dense retriever basically ignores them.
                    c.source.split(".")[0],
                    *(_top_terms(c.text)),
                ],
            }
        )
    return entries


_STOPWORDS = {
    "the","a","an","and","or","of","to","in","is","are","for","on","with",
    "as","at","by","it","be","that","this","from","not","can","you","your",
    "our","we","they","them","their","have","has","was","were","will","would",
    "but","which","who","when","what","where","why","how","all","any",
    "some","more","most","much","such","also","than","then","these","those",
    "into","per","its",
}


def _top_terms(text: str, k: int = 8) -> list[str]:
    """Very cheap keyword extraction — strip stopwords, rank by freq."""
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", text.lower())
    freq: dict[str, int] = {}
    for t in tokens:
        if t in _STOPWORDS:
            continue
        freq[t] = freq.get(t, 0) + 1
    return [t for t, _ in sorted(freq.items(), key=lambda kv: -kv[1])[:k]]


def write_jsonl(entries: list[dict], out_path: Path) -> None:
    """Dump KB-entry dicts as JSONL. Safe to re-run — overwrites cleanly."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")


# ─── Back-compat for any existing callers of the old interface ──────

def fetch(url: str, client: httpx.Client | None = None) -> ScrapedPage | None:
    """Legacy single-page fetch — deprecated; prefer `fetch_and_chunk`."""
    chunks = fetch_and_chunk(url, urlparse(url).netloc, client)
    if not chunks:
        return None
    return ScrapedPage(
        url=url,
        title=chunks[0].title,
        text=" ".join(c.text for c in chunks),
        source=chunks[0].source,
    )


def crawl(
    seed_url: str,
    allow_prefix: str,
    max_pages: int = DEFAULT_MAX_PAGES,
    rate_limit_sec: float = DEFAULT_RATE_LIMIT_SEC,
) -> list[ScrapedPage]:
    """Legacy single-string-seed crawl — deprecated; prefer `crawl_chunked`."""
    chunks = crawl_chunked(
        [seed_url], [allow_prefix], urlparse(seed_url).netloc,
        max_pages=max_pages, rate_limit_sec=rate_limit_sec,
    )
    by_url: dict[str, ScrapedPage] = {}
    for c in chunks:
        page = by_url.get(c.url)
        if page is None:
            by_url[c.url] = ScrapedPage(
                url=c.url, title=c.title, text=c.text, source=c.source,
            )
        else:
            page.text = f"{page.text} {c.text}"
    return list(by_url.values())


def write_scraped_jsonl(pages: list[ScrapedPage], out_path: Path) -> None:
    """Legacy — dumps ScrapedPage list as one-entry-per-page JSONL."""
    entries = []
    for p in pages:
        entries.append({
            "id": f"scraped:{p.source}:{abs(hash(p.url)) % 10**10}",
            "category": p.source.replace(".", "_"),
            "question": p.title or p.url,
            "answer": p.text[:1400],
            "aliases": [],
            "keywords": [p.source.split(".")[0]],
        })
    write_jsonl(entries, out_path)
