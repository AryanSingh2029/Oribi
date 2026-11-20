"""Format observed categories consistently in text reports."""

import json
from collections.abc import Callable

from ..models.evidence import CategoricalContext


def format_categories(
    categories: CategoricalContext,
    distinct_count: int,
    *,
    escape: Callable[[str], str] = str,
) -> str:
    items = []
    for value, count in categories.values:
        label = json.dumps(value, ensure_ascii=False) if isinstance(value, str) else str(value)
        items.append(f"{escape(label)} ({count})")
    values = ", ".join(items)
    if categories.is_complete:
        return f"Categories (all {distinct_count:,}): {values}"
    return f"Top categories (top {len(categories.values):,} of {distinct_count:,}): {values}"
