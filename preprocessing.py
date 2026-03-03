"""
Holiday interpolation for TimesFM context windows.

For each public holiday day within a series, all data points for that day
are replaced with a point-by-point average of 3 randomly sampled non-holiday
days from the same context window.

Raises ValueError if fewer than 3 non-holiday days are available.
"""

from __future__ import annotations

import random
from collections import defaultdict
from datetime import date, datetime

import numpy as np

from datasources.base import TimeSeries


def interpolate_holidays(series: TimeSeries, holiday_dates: set[date]) -> TimeSeries:
    """
    Replace public holiday data points with a point-by-point average of
    3 randomly sampled non-holiday days from the same series.

    Parameters
    ----------
    series:
        The time series to process. Assumes uniform step spacing.
    holiday_dates:
        Set of calendar dates to treat as public holidays.

    Returns
    -------
    A new TimeSeries with holiday days substituted. The original timestamps
    are preserved — only values change.

    Raises
    ------
    ValueError
        If fewer than 3 non-holiday days are present in the series, since
        a reliable average cannot be computed.
    """
    if not holiday_dates:
        return series

    # Group data-point indices by calendar day
    day_indices: dict[date, list[int]] = defaultdict(list)
    for i, ts in enumerate(series.timestamps):
        d = datetime.fromtimestamp(ts).date()
        day_indices[d].append(i)

    all_days = sorted(day_indices.keys())
    holiday_days = [d for d in all_days if d in holiday_dates]
    normal_days  = [d for d in all_days if d not in holiday_dates]

    if not holiday_days:
        return series  # no holidays fall within this series' window

    # Determine the standard number of points in a full day (most common count)
    from collections import Counter
    counts = Counter(len(idx) for idx in day_indices.values())
    points_per_day = counts.most_common(1)[0][0]

    # Only sample from non-holiday days that are complete (full day of data)
    full_normal_days = [
        d for d in normal_days if len(day_indices[d]) == points_per_day
    ]

    if len(full_normal_days) < 3:
        raise ValueError(
            f"Series '{series.name}': forecast cannot be produced — only "
            f"{len(full_normal_days)} full non-holiday day(s) in the context window, "
            f"need at least 3 to interpolate holidays."
        )

    # Randomly sample 3 non-holiday days
    sampled_days = random.sample(full_normal_days, 3)
    print(
        f"  [{series.name}] Interpolating {len(holiday_days)} holiday day(s) "
        f"using sampled days: {[str(d) for d in sampled_days]}"
    )

    # Point-by-point average across the 3 sampled days
    # All sampled days are full days so shapes are guaranteed to match
    avg_values = np.mean(
        [series.values[day_indices[d]] for d in sampled_days],
        axis=0,
    )

    # Substitute values for every holiday day
    new_values = series.values.copy()
    for d in holiday_days:
        indices = day_indices[d]
        # Guard against partial days (e.g. series starts/ends mid-day)
        n = min(len(indices), points_per_day)
        new_values[indices[:n]] = avg_values[:n]

    return TimeSeries(
        name=series.name,
        timestamps=series.timestamps,
        values=new_values,
    )
