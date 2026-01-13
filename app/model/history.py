from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


HISTORY_PATH = _project_root() / "artifacts" / "models" / "history.json"
HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)


def _load_history() -> list[dict[str, Any]]:
    if not HISTORY_PATH.exists():
        return []
    try:
        data = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        return []
    return []


def _save_history(records: list[dict[str, Any]]) -> None:
    HISTORY_PATH.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


def append_record(record: dict[str, Any]) -> None:
    history = _load_history()
    history.append(record)
    _save_history(history)


def get_history(model_type: str | None = None, limit: int | None = None) -> list[dict[str, Any]]:
    history = _load_history()
    if model_type:
        history = [r for r in history if r.get("model_type") == model_type]
    # сортировка по времени (если есть), иначе как есть
    def _key(r: dict[str, Any]):
        ts = r.get("timestamp")
        try:
            return datetime.fromisoformat(ts)
        except Exception:
            return datetime.min

    history.sort(key=_key, reverse=True)
    if limit and limit > 0:
        history = history[:limit]
    return history

