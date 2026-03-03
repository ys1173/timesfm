# TimesFM Forecasting

A modular time-series forecasting program built on [Google's TimesFM 2.5](https://huggingface.co/google/timesfm-2.5-200m-pytorch) — a 200M-parameter foundation model for time-series prediction. The program ingests historical data from pluggable sources, runs inference, and writes both a PNG chart and a structured JSON file per series.

---

## Table of Contents

- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Running the Forecast](#running-the-forecast)
- [Environment Variables](#environment-variables)
- [Output Files](#output-files)
- [Built-in Data Sources](#built-in-data-sources)
  - [JSON (default)](#json-source)
  - [CSV](#csv-source)
- [Adding a Custom Data Source](#adding-a-custom-data-source)
  - [Example: Prometheus / Grafana Cloud](#example-prometheus--grafana-cloud)
- [Data Source Contract](#data-source-contract)

---

## How It Works

1. A **data source** loads one or more named time series (historical observations).
2. The program auto-detects the context length from each series (capped at the model's max of 16 384 steps).
3. **TimesFM 2.5** is loaded once and compiled with a `ForecastConfig` sized to the largest series in the batch.
4. For each series the model produces a **point forecast** plus a **quantile forecast** (10 quantile levels). The chosen percentile determines the upper/lower confidence band.
5. Results are written as a **JSON file** per series (PNG chart output is available but disabled by default — see [Output Files](#output-files)).

---

## Project Structure

```
timesfm/
├── samples/
│   ├── input/                # Sample input files (Grafana-style JSON)
│   │   ├── data1.json
│   │   ├── data2.json
│   │   └── data3.json
│   └── output/               # Sample output files produced by a reference run
│       ├── forecast_data1.json
│       ├── forecast_data1.png
│       ├── forecast_data2.json
│       ├── forecast_data2.png
│       ├── forecast_data3.json
│       └── forecast_data3.png
├── datasources/              # Pluggable ingestion layer
│   ├── __init__.py           # Registry: register(), get_source_class()
│   ├── base.py               # TimeSeries dataclass + TimeSeriesSource ABC
│   ├── json_file.py          # Built-in JSON source
│   └── csv_file.py           # Built-in CSV source (example stub)
├── .env.example              # Environment variable reference
└── forecast.py               # Main program
```

---

## Setup

> Requires Python 3.10+.

```bash
# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install timesfm matplotlib numpy
```

The TimesFM model weights (~800 MB) are downloaded automatically from HuggingFace on first run.

---

## Running the Forecast

```bash
# Default: JSON source, samples/input/ directory, 90th percentile, horizon = context length
.venv/bin/python forecast.py

# Narrow the confidence band to the 70th percentile
TIMESFM_PERCENTILE=70 .venv/bin/python forecast.py

# Forecast only the next 50 steps
TIMESFM_HORIZON=50 .venv/bin/python forecast.py

# Combine options
TIMESFM_PERCENTILE=80 TIMESFM_HORIZON=100 .venv/bin/python forecast.py

# Use a different data directory
TIMESFM_SOURCE_PATH=/path/to/my/data .venv/bin/python forecast.py

# Write outputs to a specific directory
TIMESFM_OUTPUT_DIR=/tmp/results .venv/bin/python forecast.py
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `TIMESFM_SOURCE` | `json` | Data source name in the registry (`json`, `csv`, or any custom name) |
| `TIMESFM_SOURCE_PATH` | `samples/input` | Directory path or glob pattern passed to the source |
| `TIMESFM_HORIZON` | *(unset)* | Forecast horizon in steps. If unset, defaults to the context length of each series |
| `TIMESFM_PERCENTILE` | `90` | Percentile for the confidence band. Must be a multiple of 10 between 10 and 90 |
| `TIMESFM_OUTPUT_DIR` | `output/` | Where PNG and JSON output files are written |

### Percentile behaviour

The model outputs 9 quantile levels (10th through 90th percentile). Setting `TIMESFM_PERCENTILE=P` selects:

- **Upper band** → `P`th percentile
- **Lower band** → `(100 - P)`th percentile  *(symmetric)*

| `TIMESFM_PERCENTILE` | Band |
|---|---|
| `90` (default) | 10th – 90th (wide) |
| `80` | 20th – 80th |
| `70` | 30th – 70th |
| `50` | 50th – 50th (point only) |

---

## Output Files

For each series named `<name>`, the following is written to `TIMESFM_OUTPUT_DIR` (defaults to `output/`, which is excluded from version control via `.gitignore`).

> **PNG charts are disabled by default.** To re-enable them, uncomment the PNG chart block and its imports in `forecast.py`. Sample charts from a reference run are available in `samples/output/`.

### `forecast_<name>.json`

```json
{
  "name": "data1",
  "context_len": 169,
  "horizon": 100,
  "percentile": 90,
  "forecast": [
    {"timestamp": 1772512297, "value": 12.22},
    ...
  ],
  "forecast_upper": [
    {"timestamp": 1772512297, "value": 13.10},
    ...
  ],
  "forecast_lower": [
    {"timestamp": 1772512297, "value": 11.50},
    ...
  ]
}
```

All timestamps are Unix epoch seconds. Arrays have `horizon` entries each.

---

## Built-in Data Sources

### JSON source

**Name:** `json` (default)

Reads files matching `TIMESFM_SOURCE_PATH/*.json`. Each entry in the `result` array becomes a separate named series. A single file can contain one or many series.

**Single series per file** — series name is the filename without extension:

```json
{
  "result": [
    {
      "values": [
        [1771903897, "12.096"],
        [1771907497, "12.193"],
        ...
      ]
    }
  ]
}
```

**Multiple series per file** — each result entry must have a `labels` object whose values are used to name the series. The series name is composed as `{filename}_{label_value}`:

```json
{
  "result": [
    {
      "labels": {"service": "external-auth"},
      "values": [[1772253967, "99.42"], ...]
    },
    {
      "labels": {"service": "internal-auth"},
      "values": [[1772253967, "99.34"], ...]
    }
  ]
}
```

This would produce two series named `data4_external-auth` and `data4_internal-auth`, and two corresponding output files.

Each entry in `values` is a `[unix_timestamp, "numeric_string"]` pair.

```bash
# Use default samples/input/ directory
TIMESFM_SOURCE=json .venv/bin/python forecast.py

# Use a different directory
TIMESFM_SOURCE=json TIMESFM_SOURCE_PATH=/exports/grafana .venv/bin/python forecast.py
```

---

### CSV source

**Name:** `csv`

Reads files matching `TIMESFM_SOURCE_PATH/*.csv`. Each file becomes one named series. The CSV must have a header row with (at minimum) a timestamp column and a value column.

**Expected format:**

```
timestamp,value
1771903897,12.096
1771907497,12.193
...
```

Column names default to `timestamp` and `value`. To use different column names, instantiate `CsvFileSource` directly in `forecast.py`:

```python
from datasources.csv_file import CsvFileSource
from datasources import register
register("mycsv", lambda: CsvFileSource(timestamp_col="ts", value_col="metric"))
```

```bash
TIMESFM_SOURCE=csv TIMESFM_SOURCE_PATH=/data/csvfiles .venv/bin/python forecast.py
```

---

## Adding a Custom Data Source

Any data source — a database, a REST API, a message queue — can be plugged in without modifying the core program. There are only three steps.

### Step 1 — Create `datasources/<yourname>.py`

Subclass `TimeSeriesSource` and implement `load()`. The method must return a `list[TimeSeries]`. Each `TimeSeries` needs:

- `name` — a unique string identifier for the series
- `timestamps` — a `numpy` array of Unix epoch seconds (float64), uniformly spaced
- `values` — a `numpy` array of float64 observations, same length as `timestamps`

```python
# datasources/mydb.py
import os
import numpy as np
from .base import TimeSeries, TimeSeriesSource

class MyDbSource(TimeSeriesSource):
    def __init__(self):
        self.connection_string = os.environ.get("TIMESFM_SOURCE_PATH")

    def load(self) -> list[TimeSeries]:
        # connect to your database, fetch data ...
        return [
            TimeSeries(
                name="my_metric",
                timestamps=np.array([...], dtype=np.float64),
                values=np.array([...], dtype=np.float64),
            )
        ]
```

### Step 2 — Register it in `forecast.py`

Add two lines near the top of `forecast.py`, before the source is loaded:

```python
from datasources.mydb import MyDbSource
from datasources import register
register("mydb", MyDbSource)
```

### Step 3 — Set the environment variable and run

```bash
TIMESFM_SOURCE=mydb .venv/bin/python forecast.py
```

---

### Example: Prometheus / Grafana Cloud

```python
# datasources/prometheus.py
import os
import numpy as np
import requests
from .base import TimeSeries, TimeSeriesSource

class PrometheusSource(TimeSeriesSource):
    """Pull a range query from a Prometheus-compatible endpoint (e.g. Grafana Cloud)."""

    def __init__(self):
        self.base_url = os.environ.get("TIMESFM_SOURCE_PATH")   # e.g. https://prometheus-prod-xx.grafana.net
        self.token    = os.environ.get("PROMETHEUS_TOKEN")       # Grafana Cloud API key
        self.query    = os.environ.get("PROMETHEUS_QUERY")       # PromQL expression
        self.start    = os.environ.get("PROMETHEUS_START")       # Unix timestamp or RFC3339
        self.end      = os.environ.get("PROMETHEUS_END")
        self.step_s   = os.environ.get("PROMETHEUS_STEP", "3600")

    def load(self) -> list[TimeSeries]:
        url = f"{self.base_url}/api/v1/query_range"
        resp = requests.get(
            url,
            params={
                "query": self.query,
                "start": self.start,
                "end":   self.end,
                "step":  self.step_s,
            },
            headers={"Authorization": f"Bearer {self.token}"},
            timeout=30,
        )
        resp.raise_for_status()
        results = resp.json()["data"]["result"]

        series_list = []
        for i, result in enumerate(results):
            name = result["metric"].get("__name__", f"series_{i}")
            values = result["values"]          # [[timestamp, "value"], ...]
            timestamps = np.array([v[0] for v in values], dtype=np.float64)
            data       = np.array([float(v[1]) for v in values], dtype=np.float64)
            series_list.append(TimeSeries(name=name, timestamps=timestamps, values=data))

        return series_list
```

Register and run:

```python
# top of forecast.py
from datasources.prometheus import PrometheusSource
from datasources import register
register("prometheus", PrometheusSource)
```

```bash
export TIMESFM_SOURCE=prometheus
export TIMESFM_SOURCE_PATH=https://prometheus-prod-01-eu-west-0.grafana.net/api/prom
export PROMETHEUS_TOKEN=glc_eyJ...
export PROMETHEUS_QUERY='rate(http_requests_total[5m])'
export PROMETHEUS_START=1770000000
export PROMETHEUS_END=1772500000
export PROMETHEUS_STEP=3600

.venv/bin/python forecast.py
```

---

## Data Source Contract

| Requirement | Detail |
|---|---|
| Minimum observations | At least 2 per series (needed to compute the step interval) |
| Timestamp spacing | Must be uniform (equal step between all consecutive timestamps) |
| Return type | `list[TimeSeries]` — empty list raises an error |
| Dependencies | Only import what your source needs; unused sources have no cost |
