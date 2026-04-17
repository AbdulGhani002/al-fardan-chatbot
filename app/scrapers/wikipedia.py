"""Scraper for Wikipedia — broad crypto knowledge via curated topic list.

Wikipedia content is CC-BY-SA — requires attribution in any produced
artifact. For chatbot retrieval we cite Wikipedia in each KB entry's
'keywords' field and add a 'wikipedia:' id prefix so answers can be
traced back to the source page.

We don't BFS Wikipedia because link density is enormous and off-topic
wandering is guaranteed. Instead we hit a curated list of crypto-adjacent
articles directly. Add topics here to expand coverage.
"""

from pathlib import Path

from .common import (
    chunks_to_kb_entries,
    crawl_chunked,
    write_jsonl,
)


# Curated topic list — each slug is the last path segment of the EN
# Wikipedia URL. Article-level crawl only; we don't follow blue links.
TOPICS = [
    # Money + payments foundations
    "Cryptocurrency", "Bitcoin", "Ethereum", "Stablecoin", "Smart_contract",
    "Blockchain", "Distributed_ledger_technology", "Cryptographic_hash_function",
    "Public-key_cryptography", "Digital_signature",
    # PoW + PoS + consensus
    "Proof_of_work", "Proof_of_stake", "Consensus_(computer_science)",
    "Byzantine_fault",
    # Institutional / custody
    "Cryptocurrency_wallet", "Hardware_wallet", "Cryptocurrency_exchange",
    "Custodian_bank", "Multi-party_computation",
    # Core protocols + projects
    "Bitcoin_network", "Ethereum_Virtual_Machine", "Solidity",
    "Decentralized_finance", "Decentralized_exchange",
    # Tokens + categories
    "Non-fungible_token", "ERC-20", "Initial_coin_offering",
    # Regulation + compliance
    "Know_your_customer", "Anti-money_laundering",
    "Financial_Action_Task_Force", "Cryptocurrency_and_crime",
    # Sharia + Islamic finance
    "Islamic_banking_and_finance", "Murabaha", "Riba",
    # Networks beyond BTC/ETH
    "Solana_(blockchain_platform)", "Cardano_(blockchain_platform)",
    "Polkadot_(cryptocurrency)", "Avalanche_(blockchain_platform)",
    "Cosmos_(network)", "Polygon_(blockchain)", "Binance",
    # Market structure / history
    "History_of_Bitcoin", "Cryptocurrency_bubble", "Mt._Gox",
    "FTX", "Luna_Foundation_Guard",
]

BASE = "https://en.wikipedia.org/wiki/"
ALLOW = ["https://en.wikipedia.org/wiki/"]


def run(out_dir: Path) -> int:
    seeds = [BASE + t for t in TOPICS]
    # max_pages matches len(seeds) so crawl_chunked doesn't wander —
    # the queue starts with exactly these urls; internal allow-prefix
    # permits any en.wikipedia.org/wiki/ link but max_pages * 20 limit
    # keeps total chunks bounded.
    chunks = crawl_chunked(
        seeds,
        ALLOW,
        source="wikipedia.org",
        max_pages=len(seeds),
        # Wikipedia asks for 1 req/sec for bots — be conservative
        rate_limit_sec=1.2,
    )
    entries = chunks_to_kb_entries(chunks)
    write_jsonl(entries, out_dir / "scraped_wikipedia.jsonl")
    return len(entries)


if __name__ == "__main__":
    n = run(Path(__file__).resolve().parent.parent / "data" / "kb")
    print(f"[wikipedia] wrote {n} chunks")
