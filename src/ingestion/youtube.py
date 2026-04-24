"""YouTube transcript ingestion runner."""
 
from __future__ import annotations
 
import logging
import re
import time
from typing import Any, Iterable
 
import requests
 
from src.common.run_context import PipelineRunContext, build_run_context
from src.common.settings import Settings, get_settings
from src.common.storage import S3StorageManager
from src.common.structured_logging import configure_structured_logging, get_logger, log_event
from src.ingestion.common import SourceIngestionResult, build_output_path, publish_source_files, write_jsonl
 
 
VIDEO_ID_PATTERNS = (
    r"v=([A-Za-z0-9_-]{11})",
    r"youtu\.be/([A-Za-z0-9_-]{11})",
    r"^([A-Za-z0-9_-]{11})$",
)
 
 
def normalize_video_id(value: str) -> str:
    candidate = str(value).strip()
    for pattern in VIDEO_ID_PATTERNS:
        match = re.search(pattern, candidate)
        if match:
            return match.group(1)
    raise RuntimeError(f"Invalid YouTube video identifier: {value}")
 
 
def resolve_video_ids(values: Iterable[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        video_id = normalize_video_id(value)
        if video_id not in seen:
            seen.add(video_id)
            normalized.append(video_id)
    if not normalized:
        raise RuntimeError("YOUTUBE_VIDEO_IDS must include at least one valid video ID or URL.")
    return tuple(normalized)


def search_video_ids(
    query: str,
    *,
    api_key: str,
    max_results: int,
    session: requests.Session | None = None,
) -> list[str]:
    if not api_key:
        raise RuntimeError("YOUTUBE_API_KEY must be configured when YOUTUBE_SEARCH_QUERIES is used.")
    if max_results <= 0:
        raise RuntimeError("YOUTUBE_MAX_VIDEOS_PER_QUERY must be greater than zero.")

    active_session = session or requests.Session()
    response = active_session.get(
        "https://www.googleapis.com/youtube/v3/search",
        params={
            "part": "snippet",
            "q": query,
            "type": "video",
            "maxResults": min(max_results, 50),
            "key": api_key,
        },
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    video_ids: list[str] = []
    for item in payload.get("items") or []:
        item_id = item.get("id") or {}
        video_id = str(item_id.get("videoId") or "").strip()
        if video_id:
            video_ids.append(video_id)
    return video_ids


def resolve_configured_video_ids(
    settings: Settings,
    *,
    session: requests.Session | None = None,
) -> tuple[str, ...]:
    raw_video_ids: list[str] = list(settings.youtube_video_ids)
    if settings.youtube_search_queries:
        active_session = session or requests.Session()
        for query in settings.youtube_search_queries:
            raw_video_ids.extend(
                search_video_ids(
                    query,
                    api_key=settings.youtube_api_key,
                    max_results=settings.youtube_max_videos_per_query,
                    session=active_session,
                )
            )
    return resolve_video_ids(raw_video_ids)
 
 
def fetch_transcript_segments(video_id: str, languages: tuple[str, ...]) -> list[dict[str, Any]]:
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: `youtube-transcript-api`. Run `poetry install` before YouTube ingestion."
        ) from exc
 
    if hasattr(YouTubeTranscriptApi, "fetch"):
        api = YouTubeTranscriptApi()
        transcript = api.fetch(video_id, languages=languages)
        return transcript.to_raw_data()
 
    if hasattr(YouTubeTranscriptApi, "get_transcript"):
        return list(YouTubeTranscriptApi.get_transcript(video_id, languages=list(languages)))
 
    raise RuntimeError("Unsupported youtube-transcript-api version. Expected `fetch` or `get_transcript`.")
 
 
def fetch_oembed_metadata(video_id: str, session: requests.Session | None = None) -> dict[str, Any]:
    active_session = session or requests.Session()
    response = active_session.get(
        "https://www.youtube.com/oembed",
        params={
            "url": f"https://www.youtube.com/watch?v={video_id}",
            "format": "json",
        },
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    return {
        "title": payload.get("title"),
        "channel": payload.get("author_name"),
    }
 
 
def fetch_api_metadata(video_id: str, api_key: str, session: requests.Session | None = None) -> dict[str, Any]:
    active_session = session or requests.Session()
    response = active_session.get(
        "https://www.googleapis.com/youtube/v3/videos",
        params={
            "part": "snippet,statistics",
            "id": video_id,
            "key": api_key,
        },
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    items = payload.get("items") or []
    if not items:
        return {}
    item = items[0]
    snippet = item.get("snippet") or {}
    statistics = item.get("statistics") or {}
    return {
        "title": snippet.get("title"),
        "channel": snippet.get("channelTitle"),
        "channel_id": snippet.get("channelId"),
        "published_date": snippet.get("publishedAt"),
        "like_count": statistics.get("likeCount"),
    }
 
 
def build_record(
    *,
    video_id: str,
    segments: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    full_text = " ".join(str(segment.get("text") or "").strip() for segment in segments).strip()
    duration_seconds = sum(float(segment.get("duration") or 0.0) for segment in segments)
 
    return {
        "source": "youtube",
        "source_id": video_id,
        "video_id": video_id,
        "title": metadata.get("title"),
        "product_name": metadata.get("title"),
        "product_category": "video_review",
        "text": full_text,
        "duration_seconds": round(duration_seconds, 2),
        "segment_count": len(segments),
        "url": f"https://youtube.com/watch?v={video_id}",
        "rating": None,
        "verified_purchase": None,
        "created_utc": None,
        "published_date": metadata.get("published_date"),
        "channel": metadata.get("channel"),
        "channel_id": metadata.get("channel_id"),
        "like_count": metadata.get("like_count"),
    }
 
 
def run(
    *,
    settings: Settings | None = None,
    run_context: PipelineRunContext | None = None,
    logger: logging.Logger | None = None,
    storage_manager: S3StorageManager | None = None,
    session: requests.Session | None = None,
) -> SourceIngestionResult:
    settings = settings or get_settings()
    run_context = run_context or build_run_context(stage="ingest_youtube", source="youtube")
    logger = logger or get_logger("ingestion.youtube")
    storage_manager = storage_manager or S3StorageManager.from_settings(settings)
    started_at = time.perf_counter()
 
    log_event(logger, "pipeline_run_started", **run_context.as_log_fields(), status="started")
 
    try:
        active_session = session or requests.Session()
        video_ids = resolve_configured_video_ids(settings, session=active_session)
        log_event(
            logger,
            "source_fetch_started",
            **run_context.as_log_fields(),
            record_count=len(video_ids),
            status="started",
        )
        records: list[dict[str, Any]] = []
 
        for video_id in video_ids:
            video_started_at = time.perf_counter()
            segments = fetch_transcript_segments(video_id, settings.youtube_transcript_languages)
            metadata = fetch_oembed_metadata(video_id, active_session)
            if settings.youtube_api_key:
                metadata.update(fetch_api_metadata(video_id, settings.youtube_api_key, active_session))
            records.append(build_record(video_id=video_id, segments=segments, metadata=metadata))
            log_event(
                logger,
                "source_fetch_completed",
                **run_context.as_log_fields(),
                input_path=video_id,
                record_count=1,
                duration_ms=round((time.perf_counter() - video_started_at) * 1000, 2),
                status="success",
            )
 
        if not records:
            raise RuntimeError("YouTube ingestion returned zero transcript records.")
 
        output_path = build_output_path(settings, "youtube", "youtube_reviews.jsonl")
        write_jsonl(output_path, records)
        result = publish_source_files(
            source="youtube",
            local_paths=[output_path],
            record_count=len(records),
            storage_manager=storage_manager,
            run_context=run_context,
            logger=logger,
        )
        log_event(
            logger,
            "pipeline_run_completed",
            **run_context.as_log_fields(),
            output_path=str(output_path),
            record_count=len(records),
            duration_ms=round((time.perf_counter() - started_at) * 1000, 2),
            status="success",
        )
        return result
    except Exception as error:
        log_event(
            logger,
            "source_fetch_failed",
            level=logging.ERROR,
            **run_context.as_log_fields(),
            duration_ms=round((time.perf_counter() - started_at) * 1000, 2),
            status="failed",
            error_type=type(error).__name__,
            error_message=str(error),
        )
        log_event(
            logger,
            "pipeline_run_failed",
            level=logging.ERROR,
            **run_context.as_log_fields(),
            duration_ms=round((time.perf_counter() - started_at) * 1000, 2),
            status="failed",
            error_type=type(error).__name__,
            error_message=str(error),
        )
        raise
 
 
def main() -> None:
    settings = get_settings()
    configure_structured_logging(settings.log_level)
    run()
 
 
if __name__ == "__main__":
    main()
 
 
