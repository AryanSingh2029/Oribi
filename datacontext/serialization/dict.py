"""JSON-compatible projection of the canonical model."""

import json
import math
from dataclasses import fields, is_dataclass
from datetime import date, datetime
from typing import Any


def _normalize(value: Any) -> Any:
    if is_dataclass(value):
        return {field.name: _normalize(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, dict):
        return {str(k): _normalize(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_normalize(v) for v in value]
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return _normalize(value.item())
        except (ValueError, TypeError):
            pass
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def to_dict(context: Any) -> dict[str, Any]:
    """Return a stable, JSON-compatible representation."""
    return _normalize(context)


def to_json(context: Any, *, indent: int | None = 2) -> str:
    """Serialize a context using the standard JSON encoder."""
    return json.dumps(to_dict(context), indent=indent, sort_keys=True, ensure_ascii=False)
