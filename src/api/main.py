"""
ReviewPulse AI — MVP FastAPI Backend
===================================
Provides endpoints for health, source stats, semantic search, and grounded chat.

Run:
    poetry run uvicorn src.api.main:app --reload --reload-dir src
"""

import os
import re
from typing import Optional

import chromadb
from fastapi import FastAPI, Query
from pydantic import BaseModel
from pyspark.sql import SparkSession
from sentence_transformers import SentenceTransformer

try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
CHROMA_DIR = os.path.join(DATA_DIR, "chromadb_reviews")
PARQUET_PATH = os.path.join(DATA_DIR, "reviews_with_sentiment_parquet")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-sonnet-20240229")

app = FastAPI(title="ReviewPulse AI API", version="0.1.0")


class SearchResponse(BaseModel):
    source: str
    product_name: str
    product_category: str
    display_name: str
    display_category: str
    entity_type: str
    sentiment_label: str
    sentiment_score: float
    source_url: str
    distance: float
    review_text: str


class Citation(BaseModel):
    source: str
    display_name: str
    display_category: str
    entity_type: str
    source_url: str
    sentiment_label: str
    sentiment_score: float


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation]


def get_spark():
    return (
        SparkSession.builder
        .appName("ReviewPulse-API")
        .master("local[*]")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )


def get_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_collection(name="reviewpulse_reviews")


def looks_like_machine_id(value: str) -> bool:
    if not value:
        return True
    value = str(value).strip()
    return bool(re.fullmatch(r"[A-Za-z0-9_-]{16,}", value))


def clean_display_name(meta: dict) -> str:
    source = str(meta.get("source", "")).strip()
    display_name = str(meta.get("display_name", "")).strip()
    product_name = str(meta.get("product_name", "")).strip()

    if source == "amazon":
        if display_name and not looks_like_machine_id(display_name.replace("Amazon Electronics Item ", "")):
            return display_name
        if display_name:
            return display_name
        if product_name and product_name.lower() != "unknown":
            return f"Amazon Electronics Item {product_name}"
        return "Amazon Product Review"

    if source == "yelp":
        if display_name and not looks_like_machine_id(display_name):
            return display_name
        return "Yelp Business Review"

    if source == "reddit":
        if display_name and display_name.lower() != "unknown":
            return display_name
        return "Reddit Discussion"

    if source == "youtube":
        if display_name and display_name.lower() != "unknown":
            return display_name
        return "YouTube Review"

    return display_name or product_name or "Unknown Item"


def clean_display_category(meta: dict) -> str:
    source = str(meta.get("source", "")).strip()
    display_category = str(meta.get("display_category", "")).strip()
    product_category = str(meta.get("product_category", "")).strip()

    if source == "amazon":
        return display_category or "Electronics Product"

    if source == "yelp":
        if display_category and not looks_like_machine_id(display_category):
            return display_category
        return "Local Business"

    if source == "reddit":
        return display_category or product_category or "Forum Discussion"

    if source == "youtube":
        return display_category or "Video Review"

    return display_category or product_category or "Unknown Category"


def clean_entity_type(meta: dict) -> str:
    source = str(meta.get("source", "")).strip()
    entity_type = str(meta.get("entity_type", "")).strip()

    if entity_type:
        return entity_type

    defaults = {
        "amazon": "product_review",
        "yelp": "business_review",
        "reddit": "forum_post",
        "youtube": "video_transcript",
    }
    return defaults.get(source, "unknown")


def retrieve_reviews(query: str, source_filter: Optional[str] = None, n_results: int = 5):
    if not os.path.exists(CHROMA_DIR):
        return []

    model = get_model()
    collection = get_collection()

    query_embedding = model.encode([query])[0].tolist()

    where = None
    if source_filter:
        where = {"source": source_filter.lower()}

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    output = []
    for doc, meta, dist in zip(documents, metadatas, distances):
        output.append(
            {
                "source": str(meta.get("source", "")),
                "product_name": str(meta.get("product_name", "")),
                "product_category": str(meta.get("product_category", "")),
                "display_name": clean_display_name(meta),
                "display_category": clean_display_category(meta),
                "entity_type": clean_entity_type(meta),
                "sentiment_label": str(meta.get("sentiment_label", "")),
                "sentiment_score": float(meta.get("sentiment_score", 0.0)),
                "source_url": str(meta.get("source_url", "")),
                "distance": float(dist),
                "review_text": str(doc),
            }
        )

    return output


def fallback_answer(query: str, retrieved: list[dict]) -> str:
    if not retrieved:
        return "I could not find relevant reviews for that query."

    positives = []
    negatives = []
    neutrals = []

    for item in retrieved[:5]:
        summary_line = (
            f"{item['display_name']} ({item['display_category']}, {item['source']})"
        )
        if item["sentiment_label"] == "positive":
            positives.append(summary_line)
        elif item["sentiment_label"] == "negative":
            negatives.append(summary_line)
        else:
            neutrals.append(summary_line)

    parts = [f"Based on the retrieved reviews for '{query}', here is a grounded summary."]

    if positives:
        parts.append(
            "Positive evidence appears in reviews such as "
            + ", ".join(positives[:2])
            + "."
        )

    if negatives:
        parts.append(
            "Negative evidence appears in reviews such as "
            + ", ".join(negatives[:2])
            + "."
        )

    if neutrals and not negatives:
        parts.append(
            "Some results are neutral or mixed, including "
            + ", ".join(neutrals[:2])
            + "."
        )

    parts.append("This answer is based only on the retrieved review text and citations below.")

    return " ".join(parts)


def generate_grounded_answer(query: str, retrieved: list[dict]) -> str:
    if not retrieved:
        return "I could not find relevant reviews for that query."

    if not (ANTHROPIC_AVAILABLE and ANTHROPIC_API_KEY):
        return fallback_answer(query, retrieved)

    try:
        client = Anthropic(api_key=ANTHROPIC_API_KEY)

        context_blocks = []
        for i, item in enumerate(retrieved, start=1):
            context_blocks.append(
                f"[Review {i}]\n"
                f"Source: {item['source']}\n"
                f"Display Name: {item['display_name']}\n"
                f"Display Category: {item['display_category']}\n"
                f"Entity Type: {item['entity_type']}\n"
                f"Sentiment: {item['sentiment_label']} ({item['sentiment_score']})\n"
                f"URL: {item['source_url']}\n"
                f"Text: {item['review_text']}\n"
            )

        prompt = f"""
You are answering a user question ONLY from the retrieved reviews below.

Rules:
- Use only the provided reviews.
- Do not invent product facts.
- If evidence is weak or mixed, say so clearly.
- Keep the answer concise and useful.
- Use the display name and display category naturally when referring to retrieved items.
- Do not claim certainty when the evidence is limited.

User query:
{query}

Retrieved reviews:
{chr(10).join(context_blocks)}

Now answer the user query using only this evidence.
"""

        response = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=400,
            temperature=0.2,
            messages=[
                {"role": "user", "content": prompt}
            ],
        )

        parts = response.content
        if parts and hasattr(parts[0], "text"):
            return parts[0].text.strip()

        return fallback_answer(query, retrieved)

    except Exception as e:
        print(f"Anthropic call failed: {e}")
        return fallback_answer(query, retrieved)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/stats/sources")
def source_stats():
    if not os.path.exists(PARQUET_PATH):
        return {"error": "Sentiment parquet not found. Run the earlier pipeline steps first."}

    spark = get_spark()
    df = spark.read.parquet(PARQUET_PATH)

    counts = df.groupBy("source").count().collect()
    sentiment = df.groupBy("source", "sentiment_label").count().collect()

    spark.stop()

    return {
        "source_counts": [
            {"source": row["source"], "count": row["count"]}
            for row in counts
        ],
        "sentiment_breakdown": [
            {
                "source": row["source"],
                "sentiment_label": row["sentiment_label"],
                "count": row["count"],
            }
            for row in sentiment
        ],
    }


@app.get("/search/semantic", response_model=list[SearchResponse])
def semantic_search(
    query: str = Query(..., description="Natural-language query"),
    source_filter: Optional[str] = Query(None, description="amazon/yelp/reddit/youtube"),
    n_results: int = Query(5, ge=1, le=20),
):
    results = retrieve_reviews(query=query, source_filter=source_filter, n_results=n_results)
    return [SearchResponse(**item) for item in results]


@app.get("/chat", response_model=ChatResponse)
def chat(
    query: str = Query(..., description="User question"),
    source_filter: Optional[str] = Query(None, description="Optional source filter"),
    n_results: int = Query(5, ge=1, le=10),
):
    retrieved = retrieve_reviews(query=query, source_filter=source_filter, n_results=n_results)
    answer = generate_grounded_answer(query, retrieved)

    citations = [
        Citation(
            source=item["source"],
            display_name=item["display_name"],
            display_category=item["display_category"],
            entity_type=item["entity_type"],
            source_url=item["source_url"],
            sentiment_label=item["sentiment_label"],
            sentiment_score=item["sentiment_score"],
        )
        for item in retrieved
    ]

    return ChatResponse(answer=answer, citations=citations)