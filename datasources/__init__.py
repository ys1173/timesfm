"""
datasources — pluggable time-series ingestion for TimesFM.

Built-in sources
----------------
- "json"  → JsonFileSource   (Grafana-style JSON)
- "csv"   → CsvFileSource    (plain CSV)

Adding a custom source
----------------------
1. Create a module (e.g. datasources/mydb.py) with a class that extends
   TimeSeriesSource and implements load().
2. Register it before forecast.py runs, for example at the top of forecast.py::

       from datasources import register
       from datasources.mydb import MyDbSource
       register("mydb", MyDbSource)

3. Set TIMESFM_SOURCE=mydb (and TIMESFM_SOURCE_PATH if needed).
"""

from .base import TimeSeries, TimeSeriesSource
from .csv_file import CsvFileSource
from .json_file import JsonFileSource

_REGISTRY: dict[str, type[TimeSeriesSource]] = {
    "json": JsonFileSource,
    "csv": CsvFileSource,
}


def register(name: str, cls: type[TimeSeriesSource]) -> None:
    """Add (or replace) a source class in the registry under *name*."""
    _REGISTRY[name] = cls


def get_source_class(name: str) -> type[TimeSeriesSource]:
    """Return the registered source class for *name*, or raise KeyError."""
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown data source {name!r}. "
            f"Available: {sorted(_REGISTRY)}. "
            "Use register() to add custom sources."
        )
    return _REGISTRY[name]


__all__ = [
    "TimeSeries",
    "TimeSeriesSource",
    "JsonFileSource",
    "CsvFileSource",
    "register",
    "get_source_class",
]
