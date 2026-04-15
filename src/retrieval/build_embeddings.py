"""
ReviewPulse AI — Build Embeddings and Load ChromaDB
==================================================
Reads sentiment-enriched reviews, embeds review_text, and stores
them in ChromaDB for semantic retrieval.

Run:
    poetry run python src/retrieval/build_embeddings.py
"""

import os
import shutil
from pyspark.sql import SparkSession
from sentence_transformers import SentenceTransformer
import chromadb

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
INPUT_PATH = os.path.join(DATA_DIR, "reviews_with_sentiment_parquet")
CHROMA_DIR = os.path.join(DATA_DIR, "chromadb_reviews")


def build_spark():
    return (
        SparkSession.builder
        .appName("ReviewPulse-BuildEmbeddings")
        .master("local[*]")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )


def main():
    print("=" * 60)
    print("REVIEWPULSE AI — BUILD EMBEDDINGS")
    print("=" * 60)

    if not os.path.exists(INPUT_PATH):
        print(f"Input parquet not found: {INPUT_PATH}")
        print("Run sentiment scoring first.")
        return

    spark = build_spark()
    df = spark.read.parquet(INPUT_PATH)
    df = df.filter(df.review_text.isNotNull())

    amazon_df = df.filter(df.source == "amazon").limit(2000)
    yelp_df = df.filter(df.source == "yelp").limit(2000)
    reddit_df = df.filter(df.source == "reddit").limit(500)
    youtube_df = df.filter(df.source == "youtube").limit(100)

    df = amazon_df.unionByName(yelp_df).unionByName(reddit_df).unionByName(youtube_df)

    print("\nEmbedding subset counts by source:")
    df.groupBy("source").count().show(truncate=False)

    rows = df.select(
        "review_id",
        "review_text",
        "source",
        "product_name",
        "product_category",
        "display_name",
        "display_category",
        "entity_type",
        "sentiment_label",
        "sentiment_score",
        "review_date",
        "source_url",
    ).collect()

    spark.stop()

    if not rows:
        print("No rows found for embedding.")
        return

    documents = []
    ids = []
    metadatas = []

    for row in rows:
        review_text = row["review_text"]
        if not review_text or len(review_text.strip()) < 20:
            continue

        ids.append(str(row["review_id"]))
        documents.append(review_text)
        metadatas.append({
            "source": str(row["source"] or ""),
            "product_name": str(row["product_name"] or ""),
            "product_category": str(row["product_category"] or ""),
            "display_name": str(row["display_name"] or ""),
            "display_category": str(row["display_category"] or ""),
            "entity_type": str(row["entity_type"] or ""),
            "sentiment_label": str(row["sentiment_label"] or ""),
            "sentiment_score": float(row["sentiment_score"] or 0.0),
            "review_date": str(row["review_date"] or ""),
            "source_url": str(row["source_url"] or ""),
        })

    print(f"Documents to embed: {len(documents)}")

    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    print("Generating embeddings...")
    embeddings = model.encode(documents, batch_size=64, show_progress_bar=True)

    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)

    print("Creating ChromaDB collection...")
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(name="reviewpulse_reviews")

    batch_size = 500
    for i in range(0, len(documents), batch_size):
        collection.add(
            ids=ids[i:i + batch_size],
            documents=documents[i:i + batch_size],
            embeddings=embeddings[i:i + batch_size].tolist(),
            metadatas=metadatas[i:i + batch_size],
        )
        print(f"Loaded batch {i} to {min(i + batch_size, len(documents))}")

    print("\nDone.")
    print(f"ChromaDB stored at: {CHROMA_DIR}")
    print(f"Collection count: {collection.count()}")


if __name__ == "__main__":
    main()