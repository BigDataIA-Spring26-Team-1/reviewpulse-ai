from __future__ import annotations

import shutil
from dataclasses import replace

import src.api.main as api_main
from src.retrieval.embedding_backend import EmbeddingBackend, HashingEmbeddingModel, write_embedding_backend_metadata
from tests.test_source_ingestion import build_test_settings, make_workspace


def test_api_query_embedding_uses_chroma_backend_metadata():
    workspace = make_workspace("api_query_embedding_backend")
    try:
        chroma_path = workspace / "chromadb_reviews"
        backend = EmbeddingBackend(
            model=HashingEmbeddingModel(32),
            backend_name="hashing",
            model_name="hashing-32",
        )
        write_embedding_backend_metadata(chroma_path, backend)

        api_main.settings = replace(
            build_test_settings(workspace),
            chroma_path=chroma_path,
        )
        api_main.EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
        api_main.get_embedding_backend.cache_clear()

        query_embedding = api_main.encode_query_text("battery life complaints")

        assert len(query_embedding) == 32
        assert api_main.get_embedding_backend().backend_name == "hashing"
    finally:
        api_main.get_embedding_backend.cache_clear()
        shutil.rmtree(workspace, ignore_errors=True)
