"""Helpers for resolving Amazon parent ASINs to product titles."""
 
from __future__ import annotations
 
import json
import logging
import os
from pathlib import Path
from typing import Iterable
 
from src.common.settings import Settings, get_settings
 
 
logger = logging.getLogger(__name__)
_TITLE_CACHE: dict[str, str] | None = None
_CACHE_FILENAME = "product_title_cache.jsonl"
 
 
def _title_lookup_enabled() -> bool:
    raw_value = os.getenv("REVIEWPULSE_AMAZON_TITLE_LOOKUP", "").strip().lower()
    if not raw_value:
        return True
    return raw_value not in {"0", "false", "no", "off"}
 
 
def _cache_path(settings: Settings) -> Path:
    return (settings.data_dir / "amazon" / _CACHE_FILENAME).resolve()
 
 
def _load_cache(path: Path) -> dict[str, str]:
    global _TITLE_CACHE
    if _TITLE_CACHE is not None:
        return _TITLE_CACHE
 
    cache: dict[str, str] = {}
    if path.exists() and path.is_file():
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                payload = line.strip()
                if not payload:
                    continue
                try:
                    row = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                parent_asin = str(row.get("parent_asin") or "").strip()
                title = str(row.get("title") or "").strip()
                if parent_asin and title:
                    cache[parent_asin] = title
 
    _TITLE_CACHE = cache
    return _TITLE_CACHE
 
 
def _append_cache(path: Path, titles: dict[str, str]) -> None:
    global _TITLE_CACHE
    if not titles:
        return
 
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for parent_asin, title in sorted(titles.items()):
            handle.write(
                json.dumps(
                    {"parent_asin": parent_asin, "title": title},
                    ensure_ascii=True,
                )
                + "\n"
            )
 
    cache = _load_cache(path)
    cache.update(titles)
    _TITLE_CACHE = cache
 
 
def _metadata_config(category: str) -> str:
    normalized = str(category).strip().replace(" ", "_")
    if not normalized:
        raise RuntimeError("AMAZON_CATEGORY must be configured.")
    if normalized.startswith("raw_meta_"):
        return normalized
    if normalized.startswith("raw_review_"):
        normalized = normalized.removeprefix("raw_review_")
    return f"raw_meta_{normalized}"
 
 
def _load_dataset_module():
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: `datasets`. Install project dependencies before resolving Amazon product titles."
        ) from exc
    return load_dataset
 
 
def resolve_amazon_product_titles(
    product_names: Iterable[str],
    *,
    settings: Settings | None = None,
) -> dict[str, str]:
    targets = {
        str(product_name).strip()
        for product_name in product_names
        if str(product_name or "").strip()
    }
    if not targets or not _title_lookup_enabled():
        return {}
 
    resolved_settings = settings or get_settings()
    cache_path = _cache_path(resolved_settings)
    cache = _load_cache(cache_path)
    remaining = {product_name for product_name in targets if product_name not in cache}
    if not remaining:
        return {product_name: cache[product_name] for product_name in targets if product_name in cache}
 
    found: dict[str, str] = {}
    try:
        load_dataset = _load_dataset_module()
        dataset = load_dataset(
            resolved_settings.amazon_dataset_name,
            name=_metadata_config(resolved_settings.amazon_category),
            split="full",
            streaming=True,
            trust_remote_code=True,
            token=resolved_settings.huggingface_token or None,
        )
        for row in dataset:
            parent_asin = str(row.get("parent_asin") or "").strip()
            if parent_asin not in remaining:
                continue
            title = str(row.get("title") or "").strip()
            if not title:
                continue
            found[parent_asin] = title
            remaining.remove(parent_asin)
            if not remaining:
                break
    except Exception as exc:
        logger.warning("Amazon title lookup skipped: %s", exc)
        found = {}
 
    if found:
        _append_cache(cache_path, found)
        cache = _load_cache(cache_path)
        cache.update(found)
 
    return {product_name: cache[product_name] for product_name in targets if product_name in cache}
 
 