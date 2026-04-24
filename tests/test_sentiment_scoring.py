from __future__ import annotations

import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.sentiment_backend import LexiconSentimentModel, SentimentBackend, load_sentiment_backend
from src.ml.sentiment_scoring import _score_sentiment_with_arrow


def test_load_sentiment_backend_can_force_lexicon():
    backend = load_sentiment_backend("lexicon")

    assert backend.backend_name == "lexicon"
    assert backend.model_name == "lexicon"


def test_score_sentiment_with_arrow_writes_scored_parquet(tmp_path: Path):
    input_path = tmp_path / "normalized_reviews_parquet"
    input_path.mkdir(parents=True, exist_ok=True)
    output_path = tmp_path / "reviews_with_sentiment_parquet"

    source_table = pa.table(
        {
            "review_id": ["r1", "r2"],
            "review_text": [
                "Great battery life and excellent reliability.",
                "Terrible support and bad refund experience.",
            ],
            "source": ["ebay", "ebay"],
        }
    )
    pq.write_table(source_table, input_path / "part-00000.parquet", compression="snappy")

    backend = SentimentBackend(
        model=LexiconSentimentModel(),
        backend_name="lexicon",
        model_name="lexicon",
    )
    record_count = _score_sentiment_with_arrow(
        input_path,
        output_path,
        sentiment_backend=backend,
        ollama_available=False,
        row_batch_size=1,
    )

    written_parts = sorted(output_path.glob("*.parquet"))
    written = pq.read_table(output_path).to_pylist()
    assert record_count == 2
    assert len(written_parts) == 2
    assert written[0]["sentiment_label"] == "positive"
    assert written[1]["sentiment_label"] == "negative"
    assert "sentiment_score" in written[0]
    assert written[0]["aspect_labels"] == "battery"
    assert written[0]["aspect_count"] == 1
    assert json.loads(written[0]["aspect_details_json"])[0]["aspect"] == "battery"
    assert written[1]["aspect_labels"] == "customer service"
