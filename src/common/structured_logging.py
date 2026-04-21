"""Structured JSON logging helpers for ReviewPulse AI."""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any, TextIO


DEFAULT_LOGGER_NAME = "reviewpulse"


def _normalize_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


class JsonFormatter(logging.Formatter):
    """Formats log records as single-line JSON for batch observability."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname.lower(),
            "logger": record.name,
            "event_name": getattr(record, "event_name", record.getMessage()),
        }

        event_fields = getattr(record, "event_fields", {})
        for key, value in dict(event_fields).items():
            normalized = _normalize_value(value)
            if normalized is not None:
                payload[key] = normalized

        if record.exc_info:
            exc_type = record.exc_info[0]
            payload["error_type"] = exc_type.__name__ if exc_type else "Exception"
            payload["error_message"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=True, sort_keys=True)


def configure_structured_logging(
    level: str = "INFO",
    *,
    logger_name: str = DEFAULT_LOGGER_NAME,
    stream: TextIO | None = None,
) -> logging.Logger:
    logger = logging.getLogger(logger_name)
    logger.setLevel(getattr(logging, str(level).upper(), logging.INFO))
    logger.propagate = False

    handler = logging.StreamHandler(stream or sys.stdout)
    handler.setFormatter(JsonFormatter())
    logger.handlers = [handler]
    return logger


def get_logger(name: str = DEFAULT_LOGGER_NAME) -> logging.Logger:
    if name.startswith(f"{DEFAULT_LOGGER_NAME}.") or name == DEFAULT_LOGGER_NAME:
        return logging.getLogger(name)
    return logging.getLogger(f"{DEFAULT_LOGGER_NAME}.{name}")


def log_event(
    logger: logging.Logger,
    event_name: str,
    *,
    level: int = logging.INFO,
    **fields: Any,
) -> None:
    logger.log(
        level,
        event_name,
        extra={
            "event_name": event_name,
            "event_fields": fields,
        },
    )
