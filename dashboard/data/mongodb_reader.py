"""
dashboard/data/mongodb_reader.py — Local JSONL event reader.

Reads events from tempfiles/events.jsonl instead of MongoDB Atlas.
The function signature is unchanged so dashboard/app.py needs no edits.
"""
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_EVENTS_FILE = Path(__file__).resolve().parents[2] / "tempfiles" / "events.jsonl"
_cached_data: list[dict] = []


def fetch_latest_events(n: int = 50) -> tuple[list[dict], bool]:
    """Read the most recent n events from the local JSONL store."""
    global _cached_data
    try:
        if not _EVENTS_FILE.exists():
            return ([], False)

        lines = _EVENTS_FILE.read_text(encoding="utf-8").splitlines()
        events: list[dict] = []
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
            if len(events) >= n:
                break

        _cached_data = events
        return (events, False)
    except Exception as e:
        logger.error("Local event read failed: %s", e)
        return (_cached_data, True)


def get_mongo_client():
    """Compatibility shim — MongoDB replaced by local JSONL store."""
    raise RuntimeError(
        "MongoDB is not used. Read events via fetch_latest_events() instead."
    )
