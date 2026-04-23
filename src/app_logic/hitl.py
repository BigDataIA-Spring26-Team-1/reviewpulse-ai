"""Local human-in-the-loop review queue."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.app_logic.guardrails import GuardrailDecision


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_QUEUE_PATH = PROJECT_ROOT / ".runtime" / "hitl_queue.jsonl"


@dataclass(frozen=True, slots=True)
class HitlQueueItem:
    request_id: str
    created_at: str
    status: str
    query: str
    source_filter: str | None
    n_results: int
    reason: str
    flags: tuple[str, ...]
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["flags"] = list(self.flags)
        return payload


def enqueue_hitl_request(
    *,
    query: str,
    decision: GuardrailDecision,
    source_filter: str | None,
    n_results: int,
    queue_path: Path | None = None,
) -> HitlQueueItem:
    path = queue_path or DEFAULT_QUEUE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    item = HitlQueueItem(
        request_id=uuid.uuid4().hex,
        created_at=datetime.now(UTC).isoformat(),
        status="pending",
        query=str(query),
        source_filter=source_filter,
        n_results=int(n_results),
        reason=decision.reason,
        flags=decision.flags,
        confidence=decision.confidence,
    )

    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(item.to_dict(), ensure_ascii=True, sort_keys=True) + "\n"
        )
    return item


def load_hitl_queue(
    *,
    queue_path: Path | None = None,
    status: str | None = "pending",
    limit: int = 100,
) -> list[dict[str, Any]]:
    path = queue_path or DEFAULT_QUEUE_PATH
    if not path.exists():
        return []

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if status and payload.get("status") != status:
                continue
            rows.append(payload)

    rows.sort(key=lambda item: str(item.get("created_at", "")), reverse=True)
    return rows[: max(0, limit)]
