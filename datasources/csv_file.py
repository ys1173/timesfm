"""CSV file data source — example stub for developers."""

from __future__ import annotations

import csv
import glob as _glob
import os

import numpy as np

from .base import TimeSeries, TimeSeriesSource


class CsvFileSource(TimeSeriesSource):
    """
    Load time series from CSV files matching a directory or glob pattern.

    Expected CSV format: a header row followed by data rows.  The timestamp
    column should contain Unix epoch seconds; the value column should be
    numeric.

    Parameters
    ----------
    path:
        Directory path or glob pattern.  Defaults to the TIMESFM_SOURCE_PATH
        environment variable, then "data".
    timestamp_col:
        Name of the timestamp column (default "timestamp").
    value_col:
        Name of the value column (default "value").
    """

    def __init__(
        self,
        path: str | None = None,
        timestamp_col: str = "timestamp",
        value_col: str = "value",
    ) -> None:
        self.path = path or os.environ.get("TIMESFM_SOURCE_PATH", "samples/input")
        self.timestamp_col = timestamp_col
        self.value_col = value_col

    def load(self) -> list[TimeSeries]:
        if os.path.isdir(self.path):
            pattern = os.path.join(self.path, "*.csv")
        else:
            pattern = self.path

        files = sorted(_glob.glob(pattern))
        if not files:
            raise FileNotFoundError(f"No CSV files found matching: {pattern!r}")

        series_list: list[TimeSeries] = []
        for filepath in files:
            name = os.path.splitext(os.path.basename(filepath))[0]
            timestamps: list[float] = []
            values: list[float] = []
            with open(filepath, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    timestamps.append(float(row[self.timestamp_col]))
                    values.append(float(row[self.value_col]))
            series_list.append(
                TimeSeries(
                    name=name,
                    timestamps=np.array(timestamps, dtype=np.float64),
                    values=np.array(values, dtype=np.float64),
                )
            )

        return series_list
