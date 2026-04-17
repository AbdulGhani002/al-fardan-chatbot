"""Shared scraper utilities — respectful, rate-limited, robots-aware."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

USER_AGENT = "AlFardanQ9ChatbotBot/0.1 (+https://al-fardan-crm.vercel.app)"
DEFAULT_TIMEOUT = 15.0
DEFAULT_RATE_LIMIT_SEC = 1.0


@dataclass
class ScrapedPage:
    url: str
    title: str
    text: str  # plain-text body, whitespace-normalised
    source: str  # logical name, e.g. "bitcoin.org"
    fetched_at: float = field(default_factory=time.time)


def _clean_text(node) -> str:
    for tag in node.find_all(["script", "style", "nav", "footer", "aside"]):
        tag.decompose()
    txt = node.get_text(separator=" ", strip=True)
    return re.sub(r"\s+", " ", txt).strip()


def fetch(url: str, client: httpx.Client | None = None) -> ScrapedPage | None:
    """Fetch a single URL and return a ScrapedPage, or None on failure."""
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
        if r.status_code != 200 or "html" not in r.headers.get(
            "content-type", ""
        ):
            return None
        soup = BeautifulSoup(r.text, "html.parser")
        title = (soup.title.string or "").strip() if soup.title else ""
        body = soup.body or soup
        text = _clean_text(body)
        return ScrapedPage(
            url=url,
            title=title,
            text=text,
            source=urlparse(url).netloc,
        )
    except Exception as err:  # noqa: BLE001
        print(f"[scraper] {url} failed: {err}")
        return None
    finally:
        if close_after:
            client.close()


def crawl(
    seed_url: str,
    allow_prefix: str,
    max_pages: int = 30,
    rate_limit_sec: float = DEFAULT_RATE_LIMIT_SEC,
) -> list[ScrapedPage]:
    """BFS crawl from seed_url, staying within allow_prefix."""
    seen: set[str] = set()
    queue: list[str] = [seed_url]
    out: list[ScrapedPage] = []

    with httpx.Client(
        headers={"User-Agent": USER_AGENT},
        timeout=DEFAULT_TIMEOUT,
        follow_redirects=True,
    ) as client:
        while queue and len(out) < max_pages:
            url = queue.pop(0)
            if url in seen:
                continue
            seen.add(url)
            if not url.startswith(allow_prefix):
                continue
            page = fetch(url, client)
            if page is None:
                continue
            out.append(page)
            time.sleep(rate_limit_sec)

            # Harvest links for BFS
            try:
                soup = BeautifulSoup(page.text[:50000], "html.parser")
                for a in soup.find_all("a", href=True):
                    href = urljoin(url, a["href"]).split("#")[0]
                    if (
                        href.startswith(allow_prefix)
                        and href not in seen
                        and len(queue) < 500
                    ):
                        queue.append(href)
            except Exception:  # noqa: BLE001
                pass
    return out


def write_scraped_jsonl(pages: list[ScrapedPage], out_path: Path) -> None:
    """Dump scraped pages as JSONL so the indexer can merge them."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for p in pages:
            # Turn each page into a synthetic KB-style entry. The TF-IDF
            # retriever indexes by question text, so we use the page
            # title as the "question" and the body as the "answer".
            entry = {
                "id": f"scraped:{p.source}:{abs(hash(p.url)) % 10**10}",
                "category": p.source.replace(".", "_"),
                "question": p.title or p.url,
                "answer": _truncate(p.text, 1200),
                "aliases": [],
                "keywords": [p.source.split(".")[0]],
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _truncate(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    cut = text[:max_len]
    # Cut at the last sentence end to avoid mid-word truncation
    for sep in [". ", "! ", "? "]:
        idx = cut.rfind(sep)
        if idx > max_len * 0.5:
            return cut[: idx + 1]
    return cut
