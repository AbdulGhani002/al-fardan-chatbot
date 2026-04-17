"""Unit tests for the TF-IDF retriever."""

from __future__ import annotations

from pathlib import Path

from app.retrieval.tfidf import KbEntry, TfidfRetriever, load_kb


def _entry(eid: str, question: str, answer: str, **kw):
    return KbEntry(
        id=eid,
        category=kw.get("category", "test"),
        question=question,
        answer=answer,
        aliases=kw.get("aliases", []),
        keywords=kw.get("keywords", []),
    )


def test_empty_corpus():
    r = TfidfRetriever([])
    assert r.size == 0
    assert r.search("anything") == []


def test_exact_match_wins():
    r = TfidfRetriever(
        [
            _entry("a", "How does staking work?", "Stake ETH/SOL."),
            _entry("b", "How does lending work?", "Pledge BTC for USD."),
        ]
    )
    hits = r.search("how does staking work")
    assert hits
    assert hits[0].entry.id == "a"
    assert hits[0].score > 0


def test_alias_match():
    r = TfidfRetriever(
        [
            _entry(
                "a",
                "How do I sign up?",
                "Click signup.",
                aliases=["register account", "create profile", "open account"],
            ),
        ]
    )
    hits = r.search("where do i register an account")
    assert hits
    assert hits[0].entry.id == "a"


def test_keyword_weight():
    r = TfidfRetriever(
        [
            _entry(
                "a",
                "What is VARA?",
                "UAE crypto regulator.",
                keywords=["vara", "regulation", "uae"],
            ),
            _entry("b", "What is KYC?", "Know your customer."),
        ]
    )
    hits = r.search("tell me about vara regulation")
    assert hits[0].entry.id == "a"


def test_unrelated_query_low_score():
    r = TfidfRetriever(
        [
            _entry("a", "How does staking work?", "Stake ETH/SOL."),
        ]
    )
    hits = r.search("pineapple pizza recipe")
    # Might still match zero-score tokens; ensure we either got nothing
    # or a meaningfully low score.
    if hits:
        assert hits[0].score < 0.1


def test_save_and_load(tmp_path: Path):
    r = TfidfRetriever(
        [
            _entry("a", "How does staking work?", "Stake ETH/SOL."),
            _entry("b", "How does lending work?", "Pledge BTC."),
        ]
    )
    idx_path = tmp_path / "idx.pkl"
    r.save(idx_path)
    r2 = TfidfRetriever.load(idx_path)
    assert r2.size == 2
    assert r2.search("staking")[0].entry.id == "a"


def test_load_real_kb():
    """Smoke test that the shipped seed KB parses + builds."""
    kb_dir = Path(__file__).resolve().parent.parent / "app" / "data" / "kb"
    if not kb_dir.exists():
        return  # running outside the repo — skip
    entries = load_kb(kb_dir)
    # Should have at least the 4 seed files we wrote
    assert len(entries) >= 40
    r = TfidfRetriever(entries)
    # Sanity — the bot knows about Al-Fardan
    hits = r.search("who are you")
    assert hits
    assert "al-fardan" in hits[0].entry.answer.lower()
