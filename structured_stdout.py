"""Structured stdout helpers for CLI scripts."""

from __future__ import annotations

import json
from typing import Any


def _normalize_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, (int, str)):
        text = str(value)
    else:
        text = json.dumps(value, separators=(",", ":"), sort_keys=True)
    return " ".join(text.splitlines()).strip() or "null"


def emit(event: str, /, **fields: Any) -> None:
    parts = [event]
    for key, value in fields.items():
        parts.append(f"{key}={_normalize_value(value)}")
    print(" ".join(parts), flush=True)


def emit_start(**fields: Any) -> None:
    emit("START", **fields)


def emit_step(**fields: Any) -> None:
    emit("STEP", **fields)


def emit_end(**fields: Any) -> None:
    emit("END", **fields)
