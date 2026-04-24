"""
ReviewPulse AI interactive semantic search.
 
Run:
    poetry run python src/retrieval/query_reviews.py
"""
 
from __future__ import annotations
 
import os
import sys
from pathlib import Path
 
import chromadb
 
 
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
 
from src.common.settings import get_settings
from src.retrieval.embedding_backend import (
    DEFAULT_EMBEDDING_MODEL,
    encode_texts,
    load_embedding_backend,
)
 
 
def main() -> None:
    settings = get_settings()
 
    print("=" * 60)
    print("REVIEWPULSE AI QUERY REVIEWS")
    print("=" * 60)
 
    if not settings.chroma_path.exists():
        print(f"ChromaDB directory not found: {settings.chroma_path}")
        print("Run build_embeddings.py first.")
        return
 
    model_name = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    embedding_backend = load_embedding_backend(
        chroma_path=settings.chroma_path,
        model_name=model_name,
    )
    print(
        "Loading embedding backend: "
        f"{embedding_backend.backend_name} ({embedding_backend.model_name})"
    )
 
    client = chromadb.PersistentClient(path=str(settings.chroma_path))
    collection = client.get_collection(
        name=os.getenv("CHROMA_COLLECTION_NAME", "reviewpulse_reviews")
    )
 
    while True:
        query = input("\nEnter a query (or type 'exit'): ").strip()
        if query.lower() in {"exit", "quit"}:
            break
        if not query:
            continue
 
        query_embedding = encode_texts(embedding_backend, [query])[0]
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=5,
            include=["documents", "metadatas", "distances"],
        )
 
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
 
        print("\nTop results:\n")
        for index, (document, metadata, distance) in enumerate(
            zip(documents, metadatas, distances),
            start=1,
        ):
            print(f"Result {index}")
            print(f"  Source: {metadata.get('source', '')}")
            print(f"  Product: {metadata.get('product_name', '')}")
            print(f"  Category: {metadata.get('product_category', '')}")
            if metadata.get("aspect_labels"):
                print(
                    f"  Aspects: {metadata.get('aspect_labels', '')} "
                    f"({int(metadata.get('aspect_count', 0) or 0)})"
                )
            print(
                f"  Sentiment: {metadata.get('sentiment_label', '')} "
                f"({metadata.get('sentiment_score', '')})"
            )
            print(f"  Distance: {round(distance, 4)}")
            print(f"  URL: {metadata.get('source_url', '')}")
            print(f"  Text: {document[:400]}...")
            print("-" * 60)
 
 
if __name__ == "__main__":
    main()
 
 