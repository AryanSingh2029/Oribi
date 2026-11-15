"""Conservative temporal cadence and gap observations."""

from collections import Counter

import pandas as pd

from ..models.evidence import Evidence, Warning

# Four timestamps provide three intervals, enough to support the supplied
# weekly example while keeping three-point sequences out of cadence inference.
_MIN_UNIQUE_TIMESTAMPS = 4
_MIN_CADENCE_INTERVALS = 2
_MIN_CADENCE_SHARE = 0.6


def temporal_gap_warnings(series: pd.Series, column_name: str) -> tuple[Warning, ...]:
    """Return qualified gap warnings when a dominant fixed cadence is evident."""
    timestamps = pd.DatetimeIndex(series.dropna().drop_duplicates().sort_values())
    if len(timestamps) < _MIN_UNIQUE_TIMESTAMPS:
        return ()

    deltas = [timestamps[i] - timestamps[i - 1] for i in range(1, len(timestamps))]
    counts = Counter(deltas)
    ranked = counts.most_common()
    cadence, dominant_count = ranked[0]
    if len(ranked) > 1 and ranked[1][1] == dominant_count:
        return ()
    if dominant_count < _MIN_CADENCE_INTERVALS or dominant_count / len(deltas) < _MIN_CADENCE_SHARE:
        return ()

    # Every observed interval must be an integer number of cadence steps.
    # This rejects dispersed event streams while allowing real multi-step gaps.
    for delta in deltas:
        steps = delta / cadence
        if steps < 1 or steps != int(steps):
            return ()

    cadence_steps: list[int] = [int(delta / cadence) for delta in deltas]
    if not any(steps > 1 for steps in cadence_steps):
        return ()

    warnings: list[Warning] = []
    for index, steps in enumerate(cadence_steps):
        missing_count = steps - 1
        if not missing_count:
            continue
        before, after = timestamps[index], timestamps[index + 1]
        expected = [before + cadence * step for step in range(1, steps)]
        cadence_text = _format_timedelta(cadence)
        expected_text = [value.isoformat() for value in expected]
        evidence = Evidence(
            "temporal_gap",
            f"{missing_count} timestamp(s) may be absent relative to an inferred {cadence_text} cadence",
            {
                "cadence": cadence_text,
                "gap_start": expected[0].isoformat(),
                "gap_end": expected[-1].isoformat(),
                "missing_intervals": missing_count,
                "expected_timestamps": expected_text,
                "observed_before": before.isoformat(),
                "observed_after": after.isoformat(),
            },
        )
        warnings.append(Warning(
            "potential_temporal_gap",
            "low",
            "Potential temporal gap",
            f"{column_name} has a possible temporal gap relative to an inferred {cadence_text} cadence.",
            (evidence,),
            "The interval may reflect missing observations or a normal pause in data collection.",
            "Confirm the expected sampling schedule before treating the interval as missing data.",
        ))
    return tuple(warnings)


def _format_timedelta(value: pd.Timedelta) -> str:
    """Format fixed cadence durations in concise, stable terms."""
    seconds = value.total_seconds()
    units = ((86400, "day"), (3600, "hour"), (60, "minute"), (1, "second"))
    for size, name in units:
        if seconds % size == 0:
            amount = int(seconds / size)
            return f"{amount} {name}{'' if amount == 1 else 's'}"
    return str(value)
