"""
services/brain/services/mongodb.py — Local JSONL event store.

Replaces MongoDB Atlas with a local append-only JSONL file so the stack
runs without any network dependency.

Events are written to tempfiles/events.jsonl (one JSON object per line).
image_b64 is never stored — excluded at the EventRecord model level.
"""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

from services.brain.models import EventRecord

logger = logging.getLogger(__name__)

_EVENTS_FILE = Path(__file__).resolve().parents[3] / "tempfiles" / "events.jsonl"
_write_lock = threading.Lock()


# ── Compatibility shims so Brain main.py needs no changes ────────────────────

class _LocalClient:
    """Minimal stand-in for AsyncIOMotorClient used in app.state."""
    def close(self):
        pass

    def __getitem__(self, db_name: str):
        return _LocalDB()


class _LocalDB:
    def __getitem__(self, col_name: str):
        return _LocalCollection()

    async def command(self, cmd: str):
        return {"ok": 1}


class _LocalCollection:
    async def insert_one(self, doc: dict):
        pass

    async def count_documents(self, query: dict) -> int:
        return _count_local_events()


def init_motor(uri: str) -> _LocalClient:
    """Return a local client (no network connection)."""
    logger.info("Local event store initialised: %s", _EVENTS_FILE)
    return _LocalClient()


async def verify_mongodb(client) -> bool:
    """Always returns True — local store is always available."""
    return True


# ── Core write function ───────────────────────────────────────────────────────

async def write_event_record(
    record: EventRecord,
    client,
    db: str,
    collection: str,
) -> bool:
    """Append an EventRecord to the local JSONL store."""
    try:
        document = record.model_dump()
        assert "image_b64" not in document, "image_b64 must not be in EventRecord"
        _append_local(document)
        logger.info("EventRecord written locally (event_id=%s)", record.event_id)
        return True
    except Exception as exc:
        logger.error("Local write failed (event_id=%s): %s", record.event_id, exc)
        return False


# ── Helpers ───────────────────────────────────────────────────────────────────

def _append_local(doc: dict) -> None:
    _EVENTS_FILE.parent.mkdir(exist_ok=True)
    with _write_lock:
        with _EVENTS_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(doc) + "\n")


def _count_local_events() -> int:
    if not _EVENTS_FILE.exists():
        return 0
    return sum(1 for line in _EVENTS_FILE.read_text().splitlines() if line.strip())
