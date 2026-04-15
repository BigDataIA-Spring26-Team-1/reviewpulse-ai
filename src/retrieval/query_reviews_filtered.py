"""
ReviewPulse AI — Filtered Query Retrieval
=========================================
Improves retrieval quality by allowing metadata filters.

Run:
    poetry run python src/retrieval/query_reviews_filtered.py
"""

import os
from sentence_transformers import SentenceTransformer
import chromadb


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
CHROMA_DIR = os.path.join(DATA_DIR, "chromadb_reviews")


def build_where_clause(source_filter: str, category_filter: str):
    clauses = []

    if source_filter and source_filter.lower() != "all":
        clauses.append({"source": source_filter.lower()})

    if category_filter and category_filter.lower() != "all":
        clauses.append({"product_category": category_filter})

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def print_results(results):
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    if not documents:
        print("\nNo results found.\n")
        return

    print("\nTop results:\n")
    for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances), start=1):
        print(f"Result {i}")
        print(f"  Source: {meta.get('source', '')}")
        print(f"  Product: {meta.get('product_name', '')}")
        print(f"  Category: {meta.get('product_category', '')}")
        print(f"  Sentiment: {meta.get('sentiment_label', '')} ({meta.get('sentiment_score', '')})")
        print(f"  Distance: {round(dist, 4)}")
        print(f"  URL: {meta.get('source_url', '')}")
        print(f"  Text: {doc[:400]}...")
        print("-" * 60)


def main():
    print("=" * 60)
    print("REVIEWPULSE AI — FILTERED QUERY REVIEWS")
    print("=" * 60)

    if not os.path.exists(CHROMA_DIR):
        print(f"ChromaDB directory not found: {CHROMA_DIR}")
        print("Run build_embeddings.py first.")
        return

    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(name="reviewpulse_reviews")

    while True:
        query = input("\nEnter a query (or type 'exit'): ").strip()
        if query.lower() in {"exit", "quit"}:
            break
        if not query:
            continue

        source_filter = input("Source filter (amazon/yelp/reddit/youtube/all): ").strip() or "all"
        category_filter = input("Category filter (or 'all'): ").strip() or "all"

        where_clause = build_where_clause(source_filter, category_filter)
        query_embedding = model.encode([query])[0].tolist()

        if where_clause:
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=5,
                where=where_clause,
                include=["documents", "metadatas", "distances"],
            )
        else:
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=5,
                include=["documents", "metadatas", "distances"],
            )

        print_results(results)


if __name__ == "__main__":
    main()