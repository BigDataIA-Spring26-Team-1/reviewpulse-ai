from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_FALLBACK_DIMENSIONS = 384
BACKEND_METADATA_FILENAME = "_EMBEDDING_BACKEND.json"
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9']+")


@dataclass(frozen=True, slots=True)
class EmbeddingBackend:
    model: Any
    backend_name: str
    model_name: str
    fallback_reason: str | None = None


class HashingEmbeddingModel:
    def __init__(self, dimensions: int = DEFAULT_FALLBACK_DIMENSIONS) -> None:
        self.dimensions = max(16, int(dimensions))

    def encode(
        self,
        texts: Sequence[str],
        batch_size: int = 64,
        show_progress_bar: bool = False,
    ) -> list[list[float]]:
        del batch_size, show_progress_bar
        return [self._encode_text(text) for text in texts]

    def _encode_text(self, text: str | None) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = TOKEN_PATTERN.findall((text or "").lower())
        if not tokens:
            return vector

        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=16).digest()
            primary_index = int.from_bytes(digest[0:4], "little") % self.dimensions
            secondary_index = int.from_bytes(digest[4:8], "little") % self.dimensions
            primary_sign = 1.0 if digest[8] % 2 == 0 else -1.0
            secondary_sign = 1.0 if digest[9] % 2 == 0 else -1.0
            vector[primary_index] += primary_sign
            vector[secondary_index] += 0.5 * secondary_sign

        norm = math.sqrt(sum(value * value for value in vector))
        if norm <= 0.0:
            return vector
        return [value / norm for value in vector]


def _fallback_dimensions() -> int:
    raw_value = os.getenv("EMBEDDING_FALLBACK_DIM", str(DEFAULT_FALLBACK_DIMENSIONS)).strip()
    try:
        return max(16, int(raw_value))
    except ValueError:
        return DEFAULT_FALLBACK_DIMENSIONS


def _metadata_path(chroma_path: Path) -> Path:
    return chroma_path / BACKEND_METADATA_FILENAME


def _probe_sentence_transformers_import() -> tuple[bool, str | None]:
    if os.name != "nt":
        return True, None

    result = subprocess.run(
        [sys.executable, "-c", "import sentence_transformers"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode == 0:
        return True, None

    reason = result.stderr.strip() or result.stdout.strip() or f"subprocess exited with code {result.returncode}"
    return False, reason


def load_embedding_backend_metadata(chroma_path: Path) -> dict[str, Any] | None:
    metadata_path = _metadata_path(chroma_path)
    if not metadata_path.exists():
        return None
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def write_embedding_backend_metadata(chroma_path: Path, backend: EmbeddingBackend) -> Path:
    chroma_path.mkdir(parents=True, exist_ok=True)
    metadata = {
        "backend_name": backend.backend_name,
        "model_name": backend.model_name,
        "dimensions": getattr(backend.model, "dimensions", None),
    }
    if backend.fallback_reason:
        metadata["fallback_reason"] = backend.fallback_reason
    metadata_path = _metadata_path(chroma_path)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")
    return metadata_path


def load_embedding_backend(
    *,
    chroma_path: Path | None = None,
    model_name: str | None = None,
    allow_hashing_fallback: bool = True,
) -> EmbeddingBackend:
    metadata: dict[str, Any] | None = None
    if chroma_path is not None:
        metadata = load_embedding_backend_metadata(chroma_path)
        if metadata and metadata.get("backend_name") == "hashing":
            dimensions = int(metadata.get("dimensions") or _fallback_dimensions())
            model_name = str(metadata.get("model_name") or f"hashing-{dimensions}")
            return EmbeddingBackend(
                model=HashingEmbeddingModel(dimensions),
                backend_name="hashing",
                model_name=model_name,
                fallback_reason=metadata.get("fallback_reason"),
            )

    selected_model = model_name or (
        str(metadata.get("model_name"))
        if metadata and metadata.get("model_name")
        else os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    )

    try:
        available, probe_reason = _probe_sentence_transformers_import()
        if not available:
            raise RuntimeError(f"sentence_transformers import probe failed: {probe_reason}")
        sentence_transformers = importlib.import_module("sentence_transformers")
        sentence_transformer_cls = sentence_transformers.SentenceTransformer
        return EmbeddingBackend(
            model=sentence_transformer_cls(selected_model),
            backend_name="sentence-transformers",
            model_name=selected_model,
        )
    except Exception as exc:
        if metadata and metadata.get("backend_name") == "sentence-transformers":
            raise RuntimeError(
                f"Embedding backend `{metadata.get('model_name')}` could not be loaded. "
                "Fix the sentence-transformers/torch environment or rebuild embeddings with the hashing fallback."
            ) from exc
        if not allow_hashing_fallback:
            raise
        dimensions = _fallback_dimensions()
        return EmbeddingBackend(
            model=HashingEmbeddingModel(dimensions),
            backend_name="hashing",
            model_name=f"hashing-{dimensions}",
            fallback_reason=f"{type(exc).__name__}: {exc}",
        )


def encode_texts(
    backend: EmbeddingBackend,
    texts: Sequence[str],
    *,
    batch_size: int = 64,
    show_progress_bar: bool = False,
) -> list[list[float]]:
    encoded = backend.model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
    )
    if hasattr(encoded, "tolist"):
        return encoded.tolist()
    return [list(vector) for vector in encoded]
