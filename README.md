# Al-Fardan Q9 — Chatbot Service

A local-only, no-external-AI-API chatbot for the Al-Fardan Q9 client
portal. Runs as a FastAPI service on James's VPS alongside the Redis +
MongoDB containers. The Next.js CRM calls this service over HTTPS (via
Caddy / nginx reverse proxy) to power the floating chat widget.

## Design principles

1. **Zero external AI APIs.** No OpenAI, Anthropic, Gemini, Cohere. All
   inference happens on the VPS using open-source primitives.
2. **Private by default.** Chat history persists on the VPS. User data
   flowing through the bot never leaves the Al-Fardan environment.
3. **Incrementally upgradeable.** v1 uses TF-IDF + cosine similarity
   (lightweight, fast, interpretable). v2 can swap in
   `sentence-transformers` for semantic search without changing the
   consumer API.
4. **Honest about what it knows.** When confidence is below threshold,
   the bot captures the question + user context and emails the admin so
   James can answer directly and seed the KB.

## Architecture

```
┌──────────────────────────┐          ┌─────────────────────────────┐
│ Next.js CRM              │          │ al-fardan-chatbot (FastAPI) │
│ (Vercel)                 │          │ (VPS 80.65.211.25:8001)     │
│                          │          │                             │
│  ┌────────────────────┐  │  HTTPS   │  ┌─────────────────────┐   │
│  │ <ChatbotWidget />  ├──┼─────────→│  │ POST /chat          │   │
│  └────────────────────┘  │          │  │ GET /health         │   │
│                          │          │  │ POST /admin/reindex │   │
│  ┌────────────────────┐  │  ←───────│  │ POST /admin/lead    │   │
│  │ /api/chatbot/lead  │  │  HTTPS   │  └─────────┬───────────┘   │
│  └────────────────────┘  │          │            ↓                │
└──────────────────────────┘          │  ┌─────────────────────┐   │
                                       │  │ TF-IDF retriever    │   │
                                       │  │ + KB corpus (JSONL) │   │
                                       │  │ + chat history      │   │
                                       │  └─────────┬───────────┘   │
                                       │            ↓                │
                                       │  ┌─────────────────────┐   │
                                       │  │ SQLite (chat_db)    │   │
                                       │  └─────────────────────┘   │
                                       └─────────────────────────────┘
```

## Capabilities

- **FAQ retrieval** over a curated knowledge base covering:
  - Al-Fardan Q9 company + four services (Custody, Staking, OTC, Lending)
  - How lending / staking / custody / OTC actually work on the platform
  - Bitcoin + Ethereum + Solana fundamentals
  - General crypto concepts (wallets, blockchains, KYC, AML)
- **Account-creation intent**: when the bot detects signup intent, it
  walks the user through consent + collects name/email/phone/service,
  then calls the CRM's `/api/chatbot/lead` endpoint to create the user.
- **Unknown-query capture**: below-threshold queries go into
  `unanswered_queries` with full context, triaged via `/admin/queries`.
- **Chat history**: every message persisted, retrievable by session.

## Quick start (local dev)

```bash
cd al-fardan-chatbot
python3.11 -m venv .venv
source .venv/bin/activate   # on Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Optional: scrape training content from the public sources
bash scripts/scrape_all.sh

# Build the TF-IDF index (seed KB + scraped content)
python -m app.train.build_index

# Run the server
uvicorn app.main:app --reload --port 8001
```

Open http://localhost:8001/docs for the FastAPI Swagger UI.

## Deploying to VPS

One-command rollout via Docker compose (added to the al-fardan stack):

```bash
# On the VPS:
cd /root/al-fardan
nano ticker.env   # set CHATBOT_BASE_URL, CRM_API_KEY, CRM_BASE_URL
docker compose --env-file redis.env up -d chatbot
```

See `docker-compose.yml` in this repo for the container spec.

## Training pipeline

1. **Seed KB** — `app/data/kb/*.jsonl`, hand-curated entries. Covers
   Al-Fardan services + crypto basics (~100 entries to start).
2. **Scrapers** — `app/scrapers/` pulls public pages from:
   - `https://dev-al-fardan.vercel.app` (company copy)
   - `https://bitcoin.org/en/` (bitcoin fundamentals)
   - `https://ethereum.org/en/` (ethereum fundamentals)
   Each scraper respects `robots.txt` and rate-limits requests.
3. **Build index** — `app/train/build_index.py` tokenises + fits a
   `TfidfVectorizer` over the combined corpus and pickles the vectoriser
   + matrix to `app/data/index.pkl`.
4. **Serve** — `app/main.py` loads the index on startup and routes
   `/chat` through the retriever.

## API contract with the CRM

| From | To | Method | Path | Purpose |
|------|----|--------|------|---------|
| CRM widget | chatbot | POST | `/chat` | User message → bot reply |
| chatbot | CRM | POST | `/api/chatbot/lead` | Bot-qualified signup → create user + send OTP |
| CRM admin | chatbot | POST | `/admin/reindex` | Rebuild TF-IDF index after KB edit |
| CRM admin | chatbot | GET  | `/admin/queries?status=new` | Unanswered queries for triage |

All CRM ↔ chatbot calls authenticated via a shared secret header
(`x-chatbot-secret`).

## Repo layout

```
app/
  main.py                    # FastAPI entry
  config.py                  # env + settings
  models.py                  # Pydantic request/response schemas
  db.py                      # SQLite helpers (chat history, queries)
  retrieval/
    tfidf.py                 # TF-IDF retriever
    intent.py                # rule-based intent classifier
  scrapers/
    common.py                # requests + BeautifulSoup + rate limit
    bitcoin_org.py
    ethereum_org.py
    al_fardan.py
  integrations/
    crm.py                   # HTTP client for CRM /api/chatbot/lead
  data/
    kb/                      # hand-curated JSONL knowledge base
    index.pkl                # built TF-IDF index (gitignored)
    chat.db                  # SQLite (gitignored)
  train/
    build_index.py           # scrape+tokenise+fit
scripts/
  scrape_all.sh
  seed_kb.py                 # one-shot KB loader
tests/
  test_tfidf.py
  test_intent.py
  test_kb_integrity.py
Dockerfile
docker-compose.yml           # standalone (merged into /root/al-fardan/docker-compose.yml on deploy)
pyproject.toml
requirements.txt
.env.example
```
