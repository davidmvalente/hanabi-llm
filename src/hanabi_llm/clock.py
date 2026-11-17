"""One format for every timestamp in the logs: UTC, ISO 8601, millisecond precision, trailing Z."""

from __future__ import annotations

from datetime import UTC, datetime


def now_iso() -> str:
    """The current time, for example ``2026-11-17T10:15:30.123Z``."""
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def parse_iso(text: str) -> datetime:
    """The inverse of ``now_iso`` (also accepts any ISO 8601 with an offset)."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))
