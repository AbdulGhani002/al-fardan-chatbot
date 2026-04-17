#!/usr/bin/env bash
# Run every scraper sequentially — ~2 min total on a decent connection.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m app.scrapers.al_fardan
python -m app.scrapers.bitcoin_org
python -m app.scrapers.ethereum_org
echo "[scrape_all] done. Run 'python -m app.train.build_index' to rebuild the index."
