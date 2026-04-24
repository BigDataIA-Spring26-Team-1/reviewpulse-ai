# ReviewPulse AI
 
**Cross-Platform Product Review Intelligence**
 
DAMG 7245 - Big Data and Intelligent Analytics | Northeastern University | Spring 2026
 
| Member | Role |
|--------|------|
| Ayush Patil | Pipeline Engineer - eBay pipeline, iFixit pipeline, YouTube extractor, Airflow DAGs |
| Piyush Kunjilwar | Spark + LLM Engineer - Spark normalization, Ollama extraction, sentiment scoring, embeddings, ChromaDB, FastAPI |
| Raghavendra Prasath Sridhar | Frontend + QA - Streamlit dashboard, chatbot UI, charts, tests, CI, documentation |
 
---
 
## What This Project Does
 
ReviewPulse AI is a data engineering platform that pulls product reviews from multiple sources - Amazon, eBay, Yelp, iFixit, YouTube, and optional Reddit POC data - normalizes them into a single queryable schema using Spark, enriches them with sentiment scores and semantic embeddings, and exposes the result through a FastAPI service and Streamlit dashboard.
 
The system answers questions like:
- "What are the top complaints about Sony WH-1000XM5 headphones across sources?"
- "How does the Fairphone 5 compare to the iPhone 15 on repairability?"
- "What do reviews say about customer service?"
 
This project is data-first. The ingestion, schema normalization, cleaning, sentiment scoring, and retrieval pipeline are the core of the system. The LLM-backed answer generation is optional and sits on top of that pipeline.
 
---
 
## Current Architecture
 
The checked-in repository currently implements this flow:
 
| Layer | Components |
|-------|------------|
| Data Sources | Amazon Reviews 2023, eBay pipeline, Yelp dataset, iFixit pipeline, YouTube transcripts, optional Reddit connector |
| Raw / Intermediate Storage | S3-backed raw/current prefixes plus local files under `data/` |
| Processing | PySpark normalization, cleaning, and sentiment enrichment |
| ML / NLP | Ollama aspect extraction POC, sentence-transformers embeddings, ChromaDB retrieval |
| Structured / Vector Access | Local parquet files, Snowflake warehouse tables, and ChromaDB |
| Application Layer | FastAPI endpoints for health, stats, search, grounded chat, guardrails, fake-review filtering, and HITL queue access |
| Frontend | Streamlit analytics app with dashboard, review explorer, sentiment trends, product comparison, RAG chat, aspect intelligence, HITL queue, and pipeline status |
 
### Data Flow
 
```text
Amazon/eBay/Yelp/iFixit/YouTube/Reddit
    -> POC ingestion scripts and dataset loaders
    -> S3 raw/current prefixes and local staged files in data/
    -> Spark normalization and cleaning
    -> sentiment scoring parquet output
    -> Snowflake normalized/sentiment tables
    -> embeddings + ChromaDB index
    -> FastAPI
    -> Streamlit
```
 
### Note on the project diagrams
 
The repository now wires the core S3 data lake, Snowflake warehouse path, RAG chatbot, guardrails, deterministic fake-review filtering, and a local HITL queue. Redis remains optional as a response cache, while BigQuery and Gemini remain optional/planned architecture components.
 
---
 
## Data Sources
 
### Sources We Build Custom Pipelines For
 
| Source | What We Build | Volume | Access |
|--------|--------------|--------|--------|
| eBay | Scrapy-style pipeline with pagination, seller ratings, item condition, buyer feedback | 200K+ listings target | Public listings |
| iFixit | Pipeline for teardown guides, repairability scores, and user repair reviews | 50K+ reviews target | Public pages |
| YouTube | `youtube-transcript-api` based transcript extraction for review videos | 5K-10K transcripts target | Public transcripts |
| Reddit | OAuth-based connector for review-oriented subreddits | POC / stretch | Free API tier |
 
### Large Academic Datasets We Integrate
 
| Source | Volume | What It Requires | Access |
|--------|--------|-----------------|--------|
| McAuley Amazon Reviews 2023 | 571M reviews, 750 GB, 33 categories | Streaming, schema mapping, category selection | Free, academic |
| Yelp Open Dataset | 8.65 GB | Download, JSON parsing, normalization | Free, educational |
 
---
 
## Schema Normalization
 
Every source has a different format:
 
| Field | Amazon | Yelp | eBay | iFixit | YouTube |
|-------|--------|------|------|--------|---------|
| Rating | float 1.0-5.0 | int 1-5 | seller percent 0-100 | repairability 0-10 | no rating |
| Date format | epoch ms | YYYY-MM-DD | ISO 8601 | ISO 8601 | ISO 8601 from API or transcript metadata |
| Product ID | ASIN | business_id | item_id | guide_id | video_id |
| Verified purchase | Boolean | N/A | N/A | N/A | N/A |
 
Unified output schema:
 
```text
review_id | product_name | product_category | source | rating_normalized
review_text | review_date | reviewer_id | verified_purchase
helpful_votes | source_url
```
 
Normalization formulas:
- Amazon: `(rating - 1) / 4`
- Yelp: `(stars - 1) / 4`
- eBay: `seller_rating / 100`
- iFixit: `(repairability_score - 1) / 9`
- YouTube: `NULL`
 
---
 
## Technology Stack
 
| Tool | Why This |
|------|----------|
| PySpark | Handles normalization and parquet processing consistently across larger datasets |
| S3 + local parquet | Stores run/current raw and processed artifacts while preserving local reruns |
| Snowflake | Warehouse load target for normalized and sentiment parquet |
| Ollama + Llama 3.1 | Free local option for aspect extraction POC |
| Anthropic API (optional) | Improves grounded chat answers when configured |
| sentence-transformers | Local embeddings for semantic retrieval |
| ChromaDB | Persistent local vector store |
| FastAPI | Simple typed API layer |
| Streamlit | Fast Python-native dashboard |
| Airflow | Pipeline orchestration DAGs |
 
---
 
## Repository Structure
 
```text
reviewpulse-ai/
|-- docs/
|-- poc/
|   |-- aspect_extraction.py
|   |-- eda_amazon.py
|   |-- normalize_schema.py
|   |-- reddit_connector.py
|   `-- youtube_extractor.py
|-- src/
|   |-- api/
|   |   `-- main.py
|   |-- common/
|   |   `-- settings.py
|   |-- frontend/
|   |   `-- app.py
|   |-- ml/
|   |   `-- sentiment_scoring.py
|   |-- normalization/
|   |   `-- core.py
|   |-- retrieval/
|   |   |-- build_embeddings.py
|   |   `-- query_reviews.py
|   |-- spark/
|   |   `-- normalize_reviews_spark.py
|   `-- warehouse/
|       `-- snowflake_loader.py
|-- tests/
|-- data/
|-- results/
|-- .env.example
|-- pyproject.toml
`-- README.md
```
 
---
 
## Setup
 
### 1. Clone and install
 
```bash
git clone https://github.com/<your-org>/reviewpulse-ai.git
cd reviewpulse-ai
poetry install --no-root
```
 
Poetry is the supported dependency manager for this repository. There is no separate `requirements.txt`.
 
### 2. Environment variables
 
```bash
cp .env.example .env
```
 
Current variables in `.env.example`:
- `ANTHROPIC_API_KEY` and `ANTHROPIC_MODEL` for optional LLM-backed `/chat` answers
- `S3_BUCKET_NAME`, `S3_RAW_PREFIX`, and `S3_PROCESSED_PREFIX` for S3-backed pipeline artifacts and detailed health checks
- `SNOWFLAKE_ACCOUNT`, `SNOWFLAKE_USER`, `SNOWFLAKE_PASSWORD`, `SNOWFLAKE_WAREHOUSE`, `SNOWFLAKE_DATABASE`, and `SNOWFLAKE_SCHEMA` for warehouse loading
- `REDIS_URL` for optional API response caching and Redis health checks
- `HEALTHCHECK_TIMEOUT_SECONDS` to cap external dependency checks
- `YELP_DATASET_PATH` for a local Yelp Open Dataset directory, or `YELP_DATASET_S3_URI` for an S3 prefix containing the review and business JSON files
- `OLLAMA_HOST` for `poc/aspect_extraction.py`
- `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`, and `REDDIT_USER_AGENT` for `poc/reddit_connector.py`
 
The FastAPI app and the POC scripts that use environment variables now load `.env` automatically.
 
If no Anthropic key is configured, the chat endpoint falls back to a grounded extractive answer built from retrieved reviews.
 
### 3. Run the pipeline end to end
 
```bash
# Step 1: Ingest raw source data
# Yelp requires either YELP_DATASET_PATH or YELP_DATASET_S3_URI.
poetry run python src/ingestion/yelp.py
 
# Amazon full-batch ingestion
poetry run python src/ingestion/amazon.py
 
# Optional demo/POC sources
poetry run python poc/youtube_extractor.py
poetry run python poc/reddit_connector.py
 
# Step 2: Build the local JSONL preview
poetry run python poc/normalize_schema.py
 
# Step 3: Build Spark parquet output
poetry run python -m src.spark.normalize_reviews_spark
 
# Step 4: Add sentiment labels and scores
poetry run python -m src.ml.sentiment_scoring
 
# Step 5: Load processed parquet into Snowflake when configured
poetry run python -m src.warehouse.snowflake_loader --dataset all
 
# Step 6: Build embeddings and ChromaDB index
poetry run python -m src.retrieval.build_embeddings
 
# Step 7: Run tests
poetry run pytest tests/test_normalization.py -v
 
# Step 8: Start FastAPI
poetry run uvicorn src.api.main:app --reload
 
# Step 9: Start Streamlit in a new terminal
poetry run streamlit run src/frontend/app.py
```
 
FastAPI docs: `http://127.0.0.1:8000/docs`
 
If port `8000` is already occupied, run FastAPI on another port and point the
frontend at it:
 
```bash
poetry run uvicorn src.api.main:app --host 127.0.0.1 --port 8001
# .env
API_BASE_URL=http://127.0.0.1:8001
```
 
---
 
## API Endpoints
 
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Lightweight health check |
| `/health/detailed` | GET | Detailed app, S3, Snowflake, and Redis health checks |
| `/dashboard/summary` | GET | Fast dashboard KPIs and artifact freshness |
| `/reviews/options` | GET | Product/source/sentiment filter options from parquet data |
| `/reviews/explore` | GET | Filtered normalized/enriched review rows for the explorer UI |
| `/stats/sources` | GET | Source counts and sentiment breakdown |
| `/analytics/sentiment` | GET | Sentiment distribution, rating buckets, source sentiment, trends, and top aspects |
| `/products/compare` | GET | Product-level comparison aggregates for selected products |
| `/aspects/summary` | GET | Aspect-level positive/negative sentiment summary |
| `/pipeline/status` | GET | Local data artifacts, DAG files, logs, and vector-store status |
| `/search/semantic` | GET | Semantic search over embedded reviews |
| `/chat` | GET | Grounded answer generation with citations |
| `/hitl/queue` | GET | Pending or all human-review queue items |
 
Examples:
 
```bash
curl "http://localhost:8000/search/semantic?query=battery+life+headphones&n_results=5"
curl "http://localhost:8000/chat?query=What+do+people+say+about+Sony+XM5+noise+canceling"
curl "http://localhost:8000/reviews/explore?query=battery&source=amazon&limit=10"
curl "http://localhost:8000/products/compare?products=Sony%20WH-1000XM5,Fairphone%205"
```
 
## Streamlit Application
 
Run the UI:
 
```bash
poetry run streamlit run src/frontend/app.py
```
 
The app provides:
 
- Dashboard Home: KPIs, API/vector/cache/data freshness, and artifact readiness
- Review Explorer: filtered review rows with source, sentiment, rating, aspects, dates, helpful votes, and URLs
- Sentiment Analytics: sentiment distribution, rating buckets, source-wise sentiment, trend charts, and top aspects
- Product Comparison: average rating, average sentiment, review volume, source mix, and top positive/negative aspects
- RAG Chatbot: grounded answer generation with citations and optional supporting retrieved reviews
- Aspect Intelligence: aspect-level positive and negative sentiment breakdowns
- HITL Queue: read-only human-review queue with guardrail reason/confidence metadata
- Pipeline & Data Ops: local source coverage, parquet/vector artifacts, Airflow DAG files, and recent logs
- Settings / About: active backend URL, architecture summary, supported feature matrix, and run commands
 
Slow aggregate pages use explicit load buttons because local parquet/vector artifacts can be several GB.
 
---
 
## Testing
 
```bash
poetry run pytest tests/test_normalization.py -v
```
 
Tests cover normalization behavior across the supported sources and validate the unified schema contract.
 
---
 
## LLM Integration
 
LLMs are used in two bounded ways. The data pipeline still works without them.
 
### Batch Aspect Extraction (Ollama, offline POC)
 
`poc/aspect_extraction.py` supports:
- heuristic extraction by default
- Ollama-backed extraction when `OLLAMA_HOST` is configured
 
### Grounded Chat (FastAPI + optional Anthropic)
 
The chat flow is:
 
```text
query
-> guardrail evaluation
-> embed
-> retrieve relevant reviews from ChromaDB
-> optionally filter likely fake reviews
-> optionally send retrieved context to Anthropic
-> return grounded answer with citations
```
 
If Anthropic is not configured, the API returns a grounded extractive summary instead of an LLM-generated answer.
 
What the LLM does not do:
- build the sentiment parquet
- generate embeddings
- compute source counts
- replace the retrieval pipeline
 
---
 
## Guardrails and HITL
 
| Guardrail | How It Works |
|-----------|-------------|
| Typed API contracts | FastAPI + Pydantic validate requests and responses |
| Prompt and data-safety filters | `/chat` blocks prompt injection, secret extraction, and review-manipulation requests when `ENABLE_GUARDRAILS=true` |
| HITL queue | Ambiguous low-confidence queries are written to `.runtime/hitl_queue.jsonl` when `ENABLE_HITL=true` |
| Fake-review filtering | Retrieved reviews can be annotated and likely fake items removed when `ENABLE_FAKE_DETECTION=true` |
| Retrieval grounding | Answers are constrained to retrieved review context |
| Fallback mode | `/chat` still works without an external LLM key |
 
---
 
## Scalability
 
| Layer | Dev (now) | Production (scaled) |
|-------|-----------|-------------------|
| Scraping | Sequential on laptop | Parallel orchestrated jobs |
| Spark | PySpark local | Dataproc or managed Spark cluster |
| LLM extraction | Ollama single machine | Multiple workers or batch API |
| Vector store | ChromaDB local | Managed vector database |
| Structured storage | Local parquet files + Snowflake | Warehouse/lakehouse automation |
| API | FastAPI localhost | Containerized deployment |
| Frontend | Streamlit local | Streamlit Cloud or containerized deployment |
 
---
 
## Current Limitations
 
- eBay and iFixit pipelines are still POC-oriented rather than production crawlers
- Sentiment scoring is a lightweight baseline rather than a production classifier
- Retrieval quality depends on source text and metadata quality
- Redis, BigQuery, and Gemini remain optional/planned components
 
---
 
## References
 
1. Hou et al. (2024). Bridging Language and Items for Retrieval and Recommendation. arXiv:2403.03952.
2. Yelp Open Dataset. https://business.yelp.com/data/resources/open-dataset/
3. Lewis et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. NeurIPS.
4. Apache Spark. https://spark.apache.org/
5. FastAPI. https://fastapi.tiangolo.com/
6. ChromaDB. https://docs.trychroma.com/
7. sentence-transformers. https://www.sbert.net/
8. Ollama. https://ollama.com/
9. Streamlit. https://docs.streamlit.io/
 
---
 
## License
 
MIT
 
GitHub · Change is constant. GitHub keeps you ahead.
Join the world's most widely adopted, AI-powered developer platform where millions of developers, businesses, and the largest open source community build software that advances humanity.
 