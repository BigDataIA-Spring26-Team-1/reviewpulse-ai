from __future__ import annotations

import json
import re
import sqlite3
from array import array
from pathlib import Path
from typing import Any, Sequence

from src.retrieval.embedding_backend import EmbeddingBackend, encode_texts


SQLITE_STORE_FILENAME = "reviewpulse_sqlite_store.sqlite3"
VECTOR_STORE_METADATA_FILENAME = "_VECTOR_STORE_BACKEND.json"
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9']+")


def sqlite_store_path(store_dir: Path) -> Path:
    return store_dir / SQLITE_STORE_FILENAME


def sqlite_store_exists(store_dir: Path) -> bool:
    return sqlite_store_path(store_dir).exists()


def write_vector_store_metadata(
    store_dir: Path,
    *,
    backend_name: str,
    embedding_dimensions: int | None = None,
) -> Path:
    store_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = store_dir / VECTOR_STORE_METADATA_FILENAME
    metadata_path.write_text(
        json.dumps(
            {
                "backend_name": backend_name,
                "embedding_dimensions": embedding_dimensions,
            },
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return metadata_path


def _vector_to_blob(vector: Sequence[float]) -> bytes:
    return array("f", (float(value) for value in vector)).tobytes()


def _blob_to_vector(blob: bytes | None) -> list[float]:
    if not blob:
        return []
    values = array("f")
    values.frombytes(blob)
    return list(values)


def _build_fts_query(query: str) -> str | None:
    tokens = TOKEN_PATTERN.findall(query.lower())
    if not tokens:
        return None
    unique_tokens = list(dict.fromkeys(tokens))[:16]
    return " OR ".join(f'"{token.replace(chr(34), "")}"' for token in unique_tokens)


def _cosine_distance(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 1.0
    score = sum(float(a) * float(b) for a, b in zip(left, right))
    return float(1.0 - score)


class SQLiteReviewVectorStore:
    """Persistent, native-extension-free fallback for review search."""

    def __init__(self, store_dir: Path) -> None:
        self.store_dir = store_dir
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = sqlite_store_path(store_dir)
        self.connection = sqlite3.connect(str(self.db_path))
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.execute("PRAGMA temp_store=MEMORY")
        self.connection.execute("PRAGMA cache_size=-200000")

    def initialize(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS reviews (
                rowid INTEGER PRIMARY KEY,
                id TEXT NOT NULL UNIQUE,
                source TEXT NOT NULL DEFAULT '',
                document TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                embedding BLOB
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS reviews_fts
            USING fts5(document, content='reviews', content_rowid='rowid');
            CREATE TRIGGER IF NOT EXISTS reviews_ai AFTER INSERT ON reviews BEGIN
                INSERT INTO reviews_fts(rowid, document) VALUES (new.rowid, new.document);
            END;
            CREATE TRIGGER IF NOT EXISTS reviews_ad AFTER DELETE ON reviews BEGIN
                INSERT INTO reviews_fts(reviews_fts, rowid, document)
                VALUES('delete', old.rowid, old.document);
            END;
            CREATE TRIGGER IF NOT EXISTS reviews_au AFTER UPDATE ON reviews BEGIN
                INSERT INTO reviews_fts(reviews_fts, rowid, document)
                VALUES('delete', old.rowid, old.document);
                INSERT INTO reviews_fts(rowid, document) VALUES (new.rowid, new.document);
            END;
            """
        )
        self.connection.commit()

    def upsert(
        self,
        payloads: Sequence[tuple[str, str, dict[str, object]]],
        embeddings: Sequence[Sequence[float]],
    ) -> int:
        rows = [
            (
                review_id,
                str(metadata.get("source", "") or ""),
                document,
                json.dumps(metadata, ensure_ascii=True, sort_keys=True),
                _vector_to_blob(embedding),
            )
            for (review_id, document, metadata), embedding in zip(payloads, embeddings)
        ]
        if not rows:
            return 0

        self.connection.executemany(
            """
            INSERT INTO reviews(id, source, document, metadata_json, embedding)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                source=excluded.source,
                document=excluded.document,
                metadata_json=excluded.metadata_json,
                embedding=excluded.embedding
            """,
            rows,
        )
        self.connection.commit()
        return len(rows)

    def count(self) -> int:
        row = self.connection.execute("SELECT COUNT(*) FROM reviews").fetchone()
        return int(row[0] if row else 0)

    def search(
        self,
        *,
        query: str,
        embedding_backend: EmbeddingBackend,
        source_filter: str | None = None,
        n_results: int = 5,
        candidate_limit: int | None = None,
    ) -> list[dict[str, object]]:
        fts_query = _build_fts_query(query)
        if not fts_query:
            return []

        resolved_candidate_limit = candidate_limit or max(250, n_results * 100)
        params: list[object] = [fts_query]
        source_clause = ""
        if source_filter:
            source_clause = "AND r.source = ?"
            params.append(source_filter.lower())
        params.append(resolved_candidate_limit)

        try:
            rows = self.connection.execute(
                f"""
                SELECT r.id, r.document, r.metadata_json, r.embedding, bm25(reviews_fts) AS rank_score
                FROM reviews_fts
                JOIN reviews r ON r.rowid = reviews_fts.rowid
                WHERE reviews_fts MATCH ? {source_clause}
                ORDER BY rank_score ASC
                LIMIT ?
                """,
                params,
            ).fetchall()
        except sqlite3.OperationalError:
            return []

        query_embedding = encode_texts(embedding_backend, [query])[0]
        ranked: list[dict[str, object]] = []
        for _review_id, document, metadata_json, embedding_blob, _rank_score in rows:
            metadata = json.loads(str(metadata_json or "{}"))
            embedding = _blob_to_vector(embedding_blob)
            if not embedding:
                embedding = encode_texts(embedding_backend, [str(document)])[0]
            distance = _cosine_distance(query_embedding, embedding)
            ranked.append(
                {
                    "source": str(metadata.get("source", "")),
                    "product_name": str(metadata.get("product_name", "")),
                    "product_category": str(metadata.get("product_category", "")),
                    "display_name": str(metadata.get("display_name", "")),
                    "display_category": str(metadata.get("display_category", "")),
                    "entity_type": str(metadata.get("entity_type", "")),
                    "aspect_labels": str(metadata.get("aspect_labels", "")),
                    "aspect_count": int(metadata.get("aspect_count", 0) or 0),
                    "sentiment_label": str(metadata.get("sentiment_label", "")),
                    "sentiment_score": float(metadata.get("sentiment_score", 0.0) or 0.0),
                    "source_url": str(metadata.get("source_url", "")),
                    "distance": distance,
                    "review_text": str(document),
                }
            )

        ranked.sort(key=lambda item: float(item["distance"]))
        return ranked[:n_results]

    def close(self) -> None:
        self.connection.commit()
        self.connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        self.connection.close()
