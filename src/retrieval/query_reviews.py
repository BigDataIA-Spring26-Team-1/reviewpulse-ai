"""
ReviewPulse AI — Query Embedded Reviews
======================================
Takes a natural language query, embeds it, and retrieves the most
relevant reviews from ChromaDB.

Run:
    poetry run python src/retrieval/query_reviews.py
"""

import os
from sentence_transformers import SentenceTransformer
import chromadb


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
CHROMA_DIR = os.path.join(DATA_DIR, "chromadb_reviews")


def main():
    print("=" * 60)
    print("REVIEWPULSE AI — QUERY REVIEWS")
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

        query_embedding = model.encode([query])[0].tolist()

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=5,
            include=["documents", "metadatas", "distances"],
        )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

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


if __name__ == "__main__":
    main()