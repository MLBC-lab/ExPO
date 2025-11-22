from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


class JsonlLogger:
    """Append-only JSONL logger for training metrics and events."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: Dict[str, Any]) -> None:
        """Append a raw record to the log file."""
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def log_epoch(self, epoch: int, **metrics: Any) -> None:
        """Convenience wrapper for logging per-epoch metrics."""
        rec = {"epoch": int(epoch)}
        rec.update(metrics)
        self.log(rec)

    def iter_records(self) -> Iterable[Dict[str, Any]]:
        """Yield all records currently stored in the log file."""
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue  # skip malformed lines


def load_history(path: str | Path) -> List[Dict[str, Any]]:
    """Load all JSONL records from *path* into memory."""
    p = Path(path)
    if not p.exists():
        return []
    out: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out
