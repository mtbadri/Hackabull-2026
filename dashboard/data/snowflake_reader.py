"""Health trends derived from local JSONL event store."""
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_EVENTS_FILE = Path(__file__).resolve().parents[2] / "tempfiles" / "events.jsonl"


def fetch_health_trends(hours: int = 24) -> tuple[pd.DataFrame | None, bool]:
    try:
        if not _EVENTS_FILE.exists():
            return (pd.DataFrame(columns=["hour", "subtype", "count"]), False)

        since = datetime.now(timezone.utc) - timedelta(hours=hours)
        counts: dict[tuple[str, str], int] = {}

        for line in _EVENTS_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            if event.get("type") != "health":
                continue

            processed_at = event.get("processed_at", "")
            try:
                ts = datetime.fromisoformat(processed_at)
                if ts < since:
                    continue
                hour = ts.strftime("%Y-%m-%d %H:00")
            except ValueError:
                continue

            subtype = event.get("subtype", "unknown")
            counts[(hour, subtype)] = counts.get((hour, subtype), 0) + 1

        if not counts:
            return (pd.DataFrame(columns=["hour", "subtype", "count"]), False)

        df = pd.DataFrame(
            [{"hour": h, "subtype": s, "count": c} for (h, s), c in sorted(counts.items())]
        )
        return (df, False)

    except Exception as e:
        logger.error("Health trends fetch failed: %s", e)
        return (None, True)
