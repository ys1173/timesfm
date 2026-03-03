"""
TimesFM 2.5 Time Series Forecasting
Uses google/timesfm-2.5-200m-pytorch from HuggingFace to predict future
data points using historical data as context.

Environment variables
---------------------
TIMESFM_SOURCE       Source type in the registry (default: "json")
TIMESFM_SOURCE_PATH  Path/glob passed to the source   (default: "samples/input")
TIMESFM_HORIZON      Override forecast horizon; if unset, defaults to the
                     context length of each series
TIMESFM_PERCENTILE   Percentile for confidence band, multiple of 10 in
                     [10, 90] (default: 90)
TIMESFM_OUTPUT_DIR   Directory for PNG and JSON output (default: script dir)
"""

from __future__ import annotations

import json
import os
import sys
# from datetime import datetime  # only needed for PNG chart

import numpy as np
# import matplotlib.dates as mdates
# import matplotlib.pyplot as plt
import timesfm

from datasources import get_source_class

# ---------------------------------------------------------------------------
# Configuration from environment
# ---------------------------------------------------------------------------
SOURCE_NAME = os.environ.get("TIMESFM_SOURCE", "json")
HORIZON_OVERRIDE = os.environ.get("TIMESFM_HORIZON")          # None → auto
OUTPUT_DIR = os.environ.get(
    "TIMESFM_OUTPUT_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "output"),
)

_pct_raw = int(os.environ.get("TIMESFM_PERCENTILE", "90"))
if _pct_raw % 10 != 0 or not (10 <= _pct_raw <= 90):
    print(f"ERROR: TIMESFM_PERCENTILE={_pct_raw} must be a multiple of 10 in [10, 90].")
    sys.exit(1)
PERCENTILE = _pct_raw

# quantile_forecast last dim: [mean, q0.1, q0.2, ..., q0.9]  (10 entries)
UPPER_IDX = PERCENTILE // 10        # e.g. 90 → 9
LOWER_IDX = 10 - UPPER_IDX         # e.g. 90 → 1  (symmetric lower tail)

MODEL_MAX_CONTEXT = 16384

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
SourceClass = get_source_class(SOURCE_NAME)
source = SourceClass()
all_series = source.load()

if not all_series:
    print("ERROR: no series loaded.")
    sys.exit(1)

for s in all_series:
    ctx = min(s.n, MODEL_MAX_CONTEXT)
    print(f"{s.name}: {s.n} data points, step={s.step}s ({s.step // 3600}h), context={ctx}")

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------
# max_context for ForecastConfig: largest series (capped at model limit)
cfg_max_context = min(max(s.n for s in all_series), MODEL_MAX_CONTEXT)

# Determine the largest horizon we will request across all series
def _horizon_for(series_n: int) -> int:
    if HORIZON_OVERRIDE is not None:
        return int(HORIZON_OVERRIDE)
    return min(series_n, MODEL_MAX_CONTEXT)

max_raw_horizon = max(_horizon_for(s.n) for s in all_series)
# Round up to next multiple of 128 (model internal stride)
cfg_max_horizon = ((max_raw_horizon + 127) // 128) * 128

print(f"\nForecastConfig: max_context={cfg_max_context}, max_horizon={cfg_max_horizon}")
print(f"Percentile band: {10 * LOWER_IDX}th–{10 * UPPER_IDX}th  (PERCENTILE={PERCENTILE})")

print("\nLoading google/timesfm-2.5-200m-pytorch ...")
model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
    "google/timesfm-2.5-200m-pytorch",
    torch_compile=False,
)

model.compile(
    timesfm.ForecastConfig(
        max_context=cfg_max_context,
        max_horizon=cfg_max_horizon,
        normalize_inputs=True,
        use_continuous_quantile_head=True,
        force_flip_invariance=True,
        infer_is_positive=True,
        fix_quantile_crossing=True,
    )
)
print("Model ready.\n")

# ---------------------------------------------------------------------------
# Forecast + output each series
# ---------------------------------------------------------------------------

# Used by PNG chart (commented out below)
# def _to_datetimes(timestamps: np.ndarray) -> list[datetime]:
#     return [datetime.fromtimestamp(t) for t in timestamps]


for series in all_series:
    context_len = min(series.n, MODEL_MAX_CONTEXT)
    horizon = _horizon_for(series.n)

    # Use only the most recent context_len observations
    ctx_ts = series.timestamps[-context_len:]
    ctx_vals = series.values[-context_len:]

    print(f"Forecasting {series.name}: context={context_len}, horizon={horizon} ...")

    # forecast() → (point_forecast, quantile_forecast)
    # point_forecast:    (batch, horizon)
    # quantile_forecast: (batch, horizon, 10)  layout [mean, q0.1, ..., q0.9]
    point_forecast, quantile_forecast = model.forecast(
        horizon=horizon,
        inputs=[ctx_vals],
    )

    pred_mean = point_forecast[0]                         # (horizon,)
    pred_upper = quantile_forecast[0, :, UPPER_IDX]       # e.g. q0.9
    pred_lower = quantile_forecast[0, :, LOWER_IDX]       # e.g. q0.1

    # Future timestamps
    fut_ts = ctx_ts[-1] + series.step * np.arange(1, horizon + 1)

    # --- PNG chart (commented out — uncomment to re-enable) ---
    # ctx_dt = _to_datetimes(ctx_ts)
    # fut_dt = _to_datetimes(fut_ts)
    # fig, ax = plt.subplots(figsize=(16, 6))
    # ax.plot(ctx_dt, ctx_vals, color="steelblue", linewidth=1.5,
    #         label=f"Historical ({context_len} pts)")
    # ax.plot(fut_dt, pred_mean, color="tomato", linewidth=1.5,
    #         label=f"Forecast ({horizon} pts)")
    # ax.fill_between(
    #     fut_dt, pred_lower, pred_upper,
    #     color="tomato", alpha=0.2,
    #     label=f"{10 * LOWER_IDX}–{10 * UPPER_IDX}th percentile band",
    # )
    # ax.axvline(x=ctx_dt[-1], color="gray", linestyle="--", linewidth=1,
    #            label="Forecast start")
    # ax.set_title(
    #     f"{series.name}  —  TimesFM 2.5  |  {context_len} context → {horizon} forecast steps",
    #     fontsize=14,
    # )
    # ax.set_xlabel("Time")
    # ax.set_ylabel("Value")
    # ax.legend(loc="upper left")
    # ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    # plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    # plt.tight_layout()
    # png_path = os.path.join(OUTPUT_DIR, f"forecast_{series.name}.png")
    # plt.savefig(png_path, dpi=150)
    # plt.close(fig)
    # print(f"  PNG → {png_path}")

    # --- JSON output ---
    def _to_records(ts_arr: np.ndarray, val_arr: np.ndarray) -> list[dict]:
        return [
            {"timestamp": int(t), "value": float(v)}
            for t, v in zip(ts_arr, val_arr)
        ]

    output = {
        "name": series.name,
        "context_len": context_len,
        "horizon": horizon,
        "percentile": PERCENTILE,
        "forecast":       _to_records(fut_ts, pred_mean),
        "forecast_upper": _to_records(fut_ts, pred_upper),
        "forecast_lower": _to_records(fut_ts, pred_lower),
    }

    json_path = os.path.join(OUTPUT_DIR, f"forecast_{series.name}.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  JSON → {json_path}")

print("\nDone.")
