"""
Base types for TimesFM data sources.

To create a custom data source:
1. Subclass TimeSeriesSource and implement load().
2. Register it: from datasources import register; register("mydb", MyDbSource)
3. Set TIMESFM_SOURCE=mydb before running forecast.py.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass

import numpy as np


@dataclass
class TimeSeries:
    """A single named time series with uniformly-spaced observations."""

    name: str
    timestamps: np.ndarray  # Unix epoch seconds, float64
    values: np.ndarray      # float64

    @property
    def n(self) -> int:
        return len(self.values)

    @property
    def step(self) -> int:
        """Seconds between consecutive observations."""
        return int(self.timestamps[1] - self.timestamps[0])


class TimeSeriesSource(abc.ABC):
    """
    Abstract base class for pluggable time-series data sources.

    Contract
    --------
    Implement load() to return a list of TimeSeries objects.  Each series
    must have at least 2 observations so that step can be computed.

    Registration pattern
    --------------------
    After defining your subclass, register it with the global registry so
    forecast.py can discover it via the TIMESFM_SOURCE environment variable::

        from datasources import register
        register("mydb", MyDbSource)

    Then set TIMESFM_SOURCE=mydb (and TIMESFM_SOURCE_PATH if your source
    needs a path/connection string) before running forecast.py.
    """

    @abc.abstractmethod
    def load(self) -> list[TimeSeries]:
        """Load and return all available time series."""
