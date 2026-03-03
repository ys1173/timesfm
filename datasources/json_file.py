"""JSON file data source — reads Grafana-style JSON exports."""

from __future__ import annotations

import glob as _glob
import json
import os

import numpy as np

from .base import TimeSeries, TimeSeriesSource


class JsonFileSource(TimeSeriesSource):
    """
    Load time series from JSON files matching a directory or glob pattern.

    Expected JSON format (Grafana query result)::

        {"result": [{"values": [[timestamp, "value"], ...]}]}

    Parameters
    ----------
    path:
        Directory path or glob pattern.  Defaults to the TIMESFM_SOURCE_PATH
        environment variable, then "data".
    """

    def __init__(self, path: str | None = None) -> None:
        self.path = path or os.environ.get("TIMESFM_SOURCE_PATH", "samples/input")

    def load(self) -> list[TimeSeries]:
        # If path is a directory, glob all *.json inside it; otherwise treat
        # it as a glob pattern directly.
        if os.path.isdir(self.path):
            pattern = os.path.join(self.path, "*.json")
        else:
            pattern = self.path

        files = sorted(_glob.glob(pattern))
        if not files:
            raise FileNotFoundError(f"No JSON files found matching: {pattern!r}")

        series_list: list[TimeSeries] = []
        for filepath in files:
            basename = os.path.splitext(os.path.basename(filepath))[0]
            with open(filepath) as f:
                data = json.load(f)
            results = data["result"]
            for i, result in enumerate(results):
                # Name: use a label value if present, otherwise fall back to
                # filename (single series) or filename_N (multiple series).
                labels = result.get("labels", {})
                if labels:
                    # join all label values, e.g. {"service": "auth"} → "auth"
                    label_part = "_".join(str(v) for v in labels.values())
                    name = f"{basename}_{label_part}" if len(results) > 1 else label_part
                else:
                    name = basename if len(results) == 1 else f"{basename}_{i}"
                raw = result["values"]
                timestamps = np.array([v[0] for v in raw], dtype=np.float64)
                values = np.array([float(v[1]) for v in raw], dtype=np.float64)
                series_list.append(TimeSeries(name=name, timestamps=timestamps, values=values))

        return series_list
