from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.build_embeddings import (
    EMBEDDING_COLUMNS,
    _iter_embedding_batches_with_arrow,
    _load_embedding_rows_with_arrow,
    _prepare_embedding_dataframe,
)
from src.retrieval.embedding_backend import (
    EmbeddingBackend,
    HashingEmbeddingModel,
    encode_texts,
    load_embedding_backend,
    write_embedding_backend_metadata,
)


class FakeEmbeddingDataFrame:
    def __init__(self) -> None:
        self.operations: list[tuple[str, object]] = []

    def filter(self, expression: object) -> "FakeEmbeddingDataFrame":
        self.operations.append(("filter", expression))
        return self

    def select(self, *columns: str) -> "FakeEmbeddingDataFrame":
        self.operations.append(("select", columns))
        return self

    def limit(self, _count: int) -> "FakeEmbeddingDataFrame":
        raise AssertionError("Embedding preparation should not limit per-source rows.")


def test_prepare_embedding_dataframe_selects_all_embedding_columns_without_sampling():
    dataframe = FakeEmbeddingDataFrame()

    prepared = _prepare_embedding_dataframe(dataframe)

    assert prepared is dataframe
    assert dataframe.operations == [
        ("filter", "review_text IS NOT NULL"),
        ("select", EMBEDDING_COLUMNS),
    ]


def test_load_embedding_rows_with_arrow_reads_all_embedding_columns(tmp_path: Path):
    input_path = tmp_path / "reviews_with_sentiment_parquet"
    input_path.mkdir(parents=True, exist_ok=True)

    table = pa.table(
        {
            "review_id": ["r1"],
            "review_text": ["A long enough review body to embed cleanly."],
            "source": ["ebay"],
            "product_name": ["Widget"],
            "product_category": ["Gadgets"],
            "display_name": ["Widget Review"],
            "display_category": ["Product Review"],
            "entity_type": ["product_review"],
            "aspect_labels": ["battery, price"],
            "aspect_count": [2],
            "sentiment_label": ["positive"],
            "sentiment_score": [0.9],
            "review_date": ["2026-04-22T00:00:00"],
            "source_url": ["https://example.com/review/r1"],
        }
    )
    pq.write_table(table, input_path / "part-00000.parquet", compression="snappy")

    rows = _load_embedding_rows_with_arrow(input_path)

    assert rows == [
        {
            "review_id": "r1",
            "review_text": "A long enough review body to embed cleanly.",
            "source": "ebay",
            "product_name": "Widget",
            "product_category": "Gadgets",
            "display_name": "Widget Review",
            "display_category": "Product Review",
            "entity_type": "product_review",
            "aspect_labels": "battery, price",
            "aspect_count": 2,
            "sentiment_label": "positive",
            "sentiment_score": 0.9,
            "review_date": "2026-04-22T00:00:00",
            "source_url": "https://example.com/review/r1",
        }
    ]


def test_iter_embedding_batches_with_arrow_streams_rows(tmp_path: Path):
    input_path = tmp_path / "reviews_with_sentiment_parquet"
    input_path.mkdir(parents=True, exist_ok=True)

    table = pa.table(
        {
            "review_id": ["r1", "r2", "r3"],
            "review_text": [
                "A long enough review body to embed cleanly.",
                "Another long enough review body to embed cleanly.",
                "A third long enough review body to embed cleanly.",
            ],
            "source": ["ebay", "youtube", "ifixit"],
        }
    )
    pq.write_table(table, input_path / "part-00000.parquet", compression="snappy")

    batches = list(_iter_embedding_batches_with_arrow(input_path, row_batch_size=2))

    assert [len(batch) for batch in batches] == [2, 1]
    assert batches[0][0]["review_id"] == "r1"
    assert batches[0][0]["aspect_count"] == 0
    assert batches[0][0]["sentiment_score"] == 0.0


def test_hashing_embedding_model_is_deterministic():
    model = HashingEmbeddingModel(dimensions=32)

    first = model.encode(["same text", "different text"])
    second = model.encode(["same text", "different text"])

    assert first == second
    assert len(first[0]) == 32
    assert first[0] != first[1]


def test_load_embedding_backend_falls_back_when_sentence_transformers_fails(monkeypatch: pytest.MonkeyPatch):
    original_import_module = importlib.import_module

    def fake_import_module(name: str, package: str | None = None):
        if name == "sentence_transformers":
            raise OSError("c10.dll failed")
        return original_import_module(name, package)

    monkeypatch.setattr("src.retrieval.embedding_backend.importlib.import_module", fake_import_module)
    monkeypatch.setattr("src.retrieval.embedding_backend._probe_sentence_transformers_import", lambda: (True, None))
    monkeypatch.setenv("EMBEDDING_FALLBACK_DIM", "24")

    backend = load_embedding_backend(model_name="sentence-transformers/all-MiniLM-L6-v2")
    encoded = encode_texts(backend, ["fallback text"])

    assert backend.backend_name == "hashing"
    assert backend.model_name == "hashing-24"
    assert backend.fallback_reason is not None
    assert len(encoded[0]) == 24


def test_load_embedding_backend_can_force_hashing_model():
    backend = load_embedding_backend(model_name="hashing-32")

    assert backend.backend_name == "hashing"
    assert backend.model_name == "hashing-32"
    assert len(encode_texts(backend, ["query text"])[0]) == 32


def test_load_embedding_backend_uses_hashing_metadata(tmp_path: Path):
    chroma_path = tmp_path / "chromadb_reviews"
    fallback_backend = EmbeddingBackend(
        model=HashingEmbeddingModel(48),
        backend_name="hashing",
        model_name="hashing-48",
        fallback_reason="forced-for-test",
    )

    write_embedding_backend_metadata(chroma_path, fallback_backend)

    loaded = load_embedding_backend(chroma_path=chroma_path, model_name="sentence-transformers/all-MiniLM-L6-v2")

    assert loaded.backend_name == "hashing"
    assert loaded.model_name == fallback_backend.model_name
    assert len(encode_texts(loaded, ["query text"])[0]) == getattr(loaded.model, "dimensions")
