# ReviewPulse AI

**Cross-Platform Product Review Intelligence**

DAMG 7245 — Big Data and Intelligent Analytics | Northeastern University | Spring 2026

| Member | Role |
|--------|------|
| Ayush Patil | Pipeline Engineer — eBay pipeline, iFixit pipeline, YouTube extractor, Airflow DAGs, GCS setup |
| Piyush Kunjilwar | Spark + LLM Engineer — Spark normalization, Ollama extraction, DistilBERT sentiment, embeddings, ChromaDB, RAG pipeline, FastAPI |
| Raghavendra Prasath Sridhar | Frontend + QA — Streamlit dashboard, chatbot UI, trend charts, HITL interface, golden set, tests, CI, documentation |

---

## Attestation

> WE ATTEST THAT WE HAVEN'T USED ANY OTHER STUDENTS' WORK IN OUR ASSIGNMENT AND ABIDE BY THE POLICIES LISTED IN THE STUDENT HANDBOOK.
>
> Ayush Patil: 33.3% · Piyush Kunjilwar: 33.3% · Raghavendra Prasath Sridhar: 33.3%

---

## What This Project Does

ReviewPulse AI is a data engineering platform that pulls product reviews from five real sources — Amazon (571M reviews), eBay, Yelp (8.65 GB), iFixit, and YouTube — normalizes them into a single queryable schema using Spark, enriches them with sentiment scores and semantic embeddings, and exposes the result through a RAG chatbot and analytics dashboard.

The system answers questions like:
- "What are the top complaints about Sony WH-1000XM5 headphones across all platforms?"
- "How does the Fairphone 5 compare to the iPhone 15 on repairability?"
- "Show me sentiment trends for noise-canceling headphones over the past year."

This project is **data-first**. Over 30% of the work is ingesting, cleaning, normalizing, and processing heterogeneous data from sources with different schemas, rating systems, and formats. The LLM features sit on top of this data foundation — they do not replace it.

---

## Architecture

### System Architecture (7 Layers)

The system follows a layered architecture orchestrated by Airflow with GitHub Actions CI/CD:

| Layer | Components |
|-------|------------|
| **Data Sources** | Amazon 571M (pre-built datasets), eBay (custom pipelines), Yelp 8.65GB (pre-built datasets), iFixit (custom pipelines), YouTube (custom pipelines) |
| **Storage** | GCS Raw Data Lake — partitioned by source, date, and category |
| **Spark Processing** | Normalize → Deduplicate → Clean → Feature Engineer |
| **ML/NLP Analysis** | Aspect Extraction (Ollama/Llama 3.1), Sentiment (DistilBERT), Embeddings (sentence-transformers) |
| **Data Storage & Access** | BigQuery (structured queries), ChromaDB (vector search), Redis Cache (query + vector cache) |
| **Application Logic** | RAG Chatbot (Gemini free tier), Fake Detection, Guardrails + HITL |
| **Frontends** | FastAPI REST API, Streamlit Dashboard |

### Data Flow

```
Amazon ──┐
eBay   ──┤                                  ┌─ Aspect Extraction (Ollama) ──→ BigQuery ──→ Redis
Yelp   ──┼→ Ingestion Layer → GCS Raw Lake → Spark Processing ─┼─ Sentiment (DistilBERT) ──→ ChromaDB
iFixit ──┤   (Airflow DAGs /   (partitioned:   (Normalize →     └─ Embeddings ──→ ChromaDB
YouTube ─┘    Scrapy / API)     source/date)     Dedup → Clean                        │
                                                  → Feature Eng)     Guardrails ──→ FastAPI ──→ Streamlit
                                                                      + HITL           Dashboard
```

---

## Data Sources

We confirmed every source is free, legal, and accessible for a student project.

### Sources We Build Custom Pipelines For

| Source | What We Build | Volume | Access |
|--------|--------------|--------|--------|
| eBay | Scrapy pipeline with pagination, rate limiting, seller ratings, item condition, buyer feedback | 200K+ listings | Public listings |
| iFixit | Scrapy pipeline for teardown guides, repairability scores (0–10), and user repair reviews | 50K+ reviews | Public pages |
| YouTube | YouTube Data API + youtube-transcript-api for auto-generated transcripts from review videos | 5K–10K transcripts | Free library |

### Large Academic Datasets We Integrate

| Source | Volume | What It Requires | Access |
|--------|--------|-----------------|--------|
| McAuley Amazon Reviews 2023 | 571M reviews, 750 GB, 33 categories | Streaming via Hugging Face. Schema mapping. Category selection. | Free, academic |
| Yelp Open Dataset | 8.65 GB, millions of business reviews | Download, parse JSON, normalize star ratings and date formats | Free, educational |

**Total:** 580M+ reviews, 750+ GB raw. We process 5–10 Amazon categories (50–100M reviews) plus all Yelp + eBay + iFixit + YouTube data.

---

## Schema Normalization

Every source has a different format. This table shows why normalization is a real engineering problem:

| Field | Amazon | Yelp | eBay | iFixit | YouTube |
|-------|--------|------|------|--------|---------|
| Rating | float 1.0–5.0 | int 1–5 | seller % (0–100) | repairability 0–10 | No rating |
| Date format | epoch ms | YYYY-MM-DD string | ISO 8601 | ISO 8601 | ISO 8601 from API |
| Product ID | ASIN | business_id | item_id | guide_id | video_id |
| Verified purchase | Boolean | N/A | N/A | N/A | N/A |

**Unified output schema:**

```
review_id | product_name | product_category | source | rating_normalized (0–1)
review_text | review_date (ISO 8601) | reviewer_id | verified_purchase (nullable)
helpful_votes (nullable) | source_url
```

**Normalization formulas:**
- Amazon: `(rating - 1) / 4`
- Yelp: `(stars - 1) / 4`
- eBay: `seller_rating / 100`
- iFixit: `repairability_score / 10`
- YouTube: `NULL` (no rating system — sentiment extracted by LLM)

---

## Technology Stack

| Tool | Why This | Why Not Alternatives |
|------|----------|---------------------|
| PySpark / Dataproc | 750 GB doesn't fit in pandas | Dask: less mature. Pandas: memory limit. |
| GCP ($300 credit) | Best free tier for students | AWS: S3 only 5 GB free. Azure: $200 but Databricks extra. |
| BigQuery + DuckDB | 1 TB free queries/month + zero-cost local dev | Snowflake: 30-day trial. Postgres: not columnar. |
| Ollama + Llama 3.1 | Free, unlimited, runs locally | OpenAI: $5 runs out. Claude: no free batch tier. |
| Gemini free tier | 15 RPM, 1.5M tokens/day free | OpenAI: insufficient for ongoing chat. |
| DistilBERT | Free, ~100 reviews/sec on CPU | VADER: rule-based. TextBlob: low accuracy. |
| sentence-transformers | Free, 384-dim vectors, local | OpenAI ada-002: costs at scale. |
| ChromaDB | Open-source, no storage limits | Pinecone: 2 GB cap free. FAISS: no persistence. |
| Airflow | DAG scheduling, dependency management | Cron: no monitoring. Prefect: less ecosystem. |
| FastAPI | Async, auto OpenAPI docs, Pydantic | Flask: no async. Django: too heavy for API-only. |
| Streamlit | Free hosting, Python-native | React: more work, team is Python-focused. |
| Scrapy | Distributed spiders, rate limiting | Selenium: slower. Requests+BS4: no scheduling. |

**Total cost: $0.** All free tiers, student credits, or open-source.

---

## Repository Structure

```
reviewpulse-ai/
├── docs/                          # Proposal, Codelabs, diagrams
│   ├── FinalProjectProposal.docx
│   ├── Architecture_Diagram.png
│   └── Data_Flow_Diagram.png
├── poc/                           # Proof of concept scripts
│   ├── eda_amazon.py              # Amazon EDA + chart generation
│   ├── ebay_pipeline.py           # eBay custom Scrapy-style pipeline
│   ├── ifixit_pipeline.py         # iFixit custom Scrapy-style pipeline
│   ├── youtube_extractor.py       # YouTube transcript extraction
│   ├── normalize_schema.py        # Python normalization (all 5 sources)
│   └── aspect_extraction.py       # Ollama aspect extraction demo
├── src/
│   ├── api/
│   │   └── main.py                # FastAPI — /health, /stats, /search, /chat
│   ├── frontend/
│   │   └── app.py                 # Streamlit dashboard
│   ├── ml/
│   │   └── sentiment_scoring.py   # DistilBERT sentiment pipeline
│   ├── retrieval/
│   │   ├── build_embeddings.py    # sentence-transformers → ChromaDB
│   │   ├── query_reviews.py       # Semantic search
│   │   └── query_reviews_filtered.py  # Filtered search by source
│   └── spark/
│       └── normalize_reviews_spark.py  # PySpark normalization job
├── dags/
│   └── dag_ingestion.py           # Airflow DAG — parallel ingestion
├── tests/
│   └── test_normalization.py      # 30+ unit tests across all sources
├── data/                          # Generated/downloaded data (gitignored)
├── results/                       # EDA charts and visualizations
│   ├── eda_rating_distribution.png
│   ├── eda_review_length.png
│   ├── eda_temporal_distribution.png
│   ├── eda_top_products.png
│   └── eda_verified_vs_rating.png
├── .github/workflows/             # GitHub Actions CI
├── pyproject.toml                 # Poetry dependencies
├── requirements.txt
├── .env.example
└── README.md
```

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/<your-org>/reviewpulse-ai.git
cd reviewpulse-ai
poetry install --no-root
```

### 2. Environment variables (optional)

```bash
cp .env.example .env
# Edit .env with your keys if using Gemini/Anthropic for chat
export ANTHROPIC_API_KEY="your_key_here"       # optional — fallback to extractive answers
export GOOGLE_API_KEY="your_gemini_key_here"    # optional — for RAG chatbot
```

If no LLM API is configured, the chat endpoint falls back to a grounded extractive answer from retrieved reviews.

### 3. Run the pipeline end-to-end

```bash
# Step 1: Generate/ingest source data
poetry run python poc/ebay_pipeline.py
poetry run python poc/ifixit_pipeline.py
poetry run python poc/youtube_extractor.py
poetry run python poc/eda_amazon.py

# Step 2: Normalize all sources into unified schema
poetry run python poc/normalize_schema.py

# Step 3: Spark parquet output
poetry run python src/spark/normalize_reviews_spark.py

# Step 4: Sentiment scoring
poetry run python src/ml/sentiment_scoring.py

# Step 5: Build embeddings + ChromaDB index
poetry run python src/retrieval/build_embeddings.py

# Step 6: Run tests
poetry run pytest tests/test_normalization.py -v

# Step 7: Start FastAPI
poetry run uvicorn src.api.main:app --reload
# → http://127.0.0.1:8000/docs

# Step 8: Start Streamlit (new terminal)
poetry run streamlit run src/frontend/app.py
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/stats/sources` | GET | Source counts and sentiment breakdown |
| `/search/semantic` | GET | Semantic search over embedded reviews. Params: `query`, `source_filter`, `n_results` |
| `/chat` | GET | Grounded answer generation with citations. Params: `query`, `source_filter`, `n_results` |

Example:
```bash
curl "http://localhost:8000/search/semantic?query=battery+life+headphones&n_results=5"
curl "http://localhost:8000/chat?query=What+do+people+say+about+Sony+XM5+noise+canceling"
```

---

## Testing

```bash
poetry run pytest tests/test_normalization.py -v
```

Tests cover:
- **Amazon normalization** — float rating → 0–1, epoch ms → ISO 8601, null text handling, verified_purchase passthrough
- **Yelp normalization** — integer stars → 0–1, date string parsing, null verified_purchase/helpful_votes
- **eBay normalization** — seller percentage → 0–1, ISO date preservation, condition metadata
- **iFixit normalization** — repairability 0–10 → 0–1, device category mapping
- **YouTube normalization** — null rating handling, long transcript preservation
- **Unified schema** — all required fields present, rating range 0–1, null ratings allowed

CI runs on every push via GitHub Actions: pytest, ruff linting, mypy type checking.

---

## LLM Integration

LLMs are used in two bounded ways. The data platform works without them.

### Batch Aspect Extraction (Ollama, offline)

Runs overnight on 500K–1M reviews using Llama 3.1 8B locally. Extracts structured aspects and sentiment from unstructured text. ~3% malformed JSON rate — caught, logged, skipped.

### RAG Chatbot (Gemini, live)

Query → embed → retrieve top 50 reviews from ChromaDB → pass to Gemini with grounding prompt → check for hallucinations → return answer with citations.

**What the LLM does NOT do:** Average sentiment, rating trends, cross-platform comparisons — those come from structured BigQuery queries. The LLM adds a natural language interface; it does not replace the data engineering.

---

## Guardrails & HITL

| Guardrail | How It Works |
|-----------|-------------|
| Pydantic schema enforcement | All API requests/responses validated against typed models |
| Hallucination detection | Each chatbot claim checked against retrieved context. Flag if overlap < 0.7 |
| Confidence scoring | 0–1 score based on review count, source consistency, faithfulness. < 0.5 → flagged |
| Input moderation | Queries screened for abuse/off-topic before RAG pipeline |
| HITL: Fake reviews (stretch) | Reviews with fake_score > 0.7 enter human review queue |
| LLM output validation | Malformed JSON from Ollama caught, logged, skipped. Tested with deliberate bad input |

---

## Scalability

| Layer | Dev (now) | Production (scaled) |
|-------|-----------|-------------------|
| Scraping | Sequential on laptop | Scrapy cluster + Airflow parallelization |
| Spark | PySpark local, 4 cores | Dataproc 10–50 nodes (no code change) |
| LLM extraction | Ollama single machine | Multiple containers or batch API |
| Vector store | ChromaDB local | Pinecone/Weaviate managed cluster |
| Warehouse | DuckDB / BigQuery free tier | BigQuery partitioned tables |
| API | FastAPI localhost | Cloud Run (0–100 instances, auto-scaling) |
| Frontend | Streamlit local | Streamlit Cloud or Cloud Run |

---

## Project Timeline

| Phase | Days | What | Deliverables |
|-------|------|------|-------------|
| M1 | 1–4 | Data Ingestion | eBay + iFixit pipelines working. Amazon streaming tested. Yelp parsed. YouTube transcripts extracted. Raw data in GCS. EDA complete. |
| M2 | 3–6 | Spark Pipeline | Schema normalization for all 5 sources. Dedup logic. Cleaning. Feature engineering. Parquet output. Airflow DAGs. |
| M3 | 5–9 | LLM + Embeddings | Ollama aspect extraction. DistilBERT sentiment. sentence-transformers embeddings. ChromaDB loaded. |
| M4 | 7–10 | Backend + Chatbot | FastAPI endpoints. RAG chatbot. Guardrails. Pydantic models. Redis caching. |
| M5 | 9–12 | Frontend | Streamlit dashboard: chatbot, trend charts, product comparison. HITL UI. |
| M6 | 11–14 | Test + Deploy | Golden set eval. Unit/integration tests. CI. Deploy to Streamlit Cloud. Video. Docs. |

---

## EDA Results

Exploratory analysis on 100K Amazon Electronics reviews (streamed via Hugging Face):

| Finding | Detail |
|---------|--------|
| Rating distribution | Heavily skewed — 58% are 5-star |
| Review length | Average 85 words, long tail distribution |
| Temporal coverage | 80% of reviews are post-2015 |
| Empty text | 12% of reviews have no text (rating-only) |
| Verified purchase | 85% are verified purchases |

Charts saved in `results/` directory.

---

## Current Limitations

- Amazon data in current MVP uses a sample subset, not the full 571M dataset
- eBay and iFixit pipelines use generated sample data; production would use live Scrapy spiders
- Sentiment pipeline is a lightweight DistilBERT baseline
- Retrieval quality depends on source text and metadata quality
- YouTube transcripts require LLM to extract structured product mentions

---

## References

1. Hou et al. (2024). Bridging Language and Items for Retrieval and Recommendation. arXiv:2403.03952. Dataset: huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023
2. Yelp Open Dataset. business.yelp.com/data/resources/open-dataset/
3. World Economic Forum (2021). Fake online reviews cost $152 billion annually.
4. Lewis et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. NeurIPS.
5. Apache Spark: spark.apache.org | FastAPI: fastapi.tiangolo.com | ChromaDB: docs.trychroma.com
6. sentence-transformers: sbert.net | Ollama: ollama.com | Gemini API: ai.google.dev
7. Scrapy: scrapy.org | Airflow: airflow.apache.org | Streamlit: docs.streamlit.io

---

## License

MIT
