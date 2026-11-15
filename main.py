#!/usr/bin/env python3
"""Forecast Bitcoin's price 240 minutes ahead, one case per run.

Reads every *.in file in $MAKEFUN_INPUT_DIR (else /input/) and writes one
/output/<id>.out per case: {"forecasts": [{"case": <id>, "price": <guess>}]}

Method: work in log prices. Take the last close P0, the linear-regression
slope of log price over the last 240 minutes (short trend), and the mean log
price over the last 1440 minutes (24h mean reversion). The forecast is

    log(P) = log(P0) + A * (slope * 240) + B * (mean24h_log - log(P0))

with A and B damped (tuned on practice backtests). When the series is too
short or flat the forecast falls back to the last close, which is the
persistence baseline. Deterministic, standard library only, no network.
"""

import json
import math
import os
import sys

# Tuned damping weights (backtested on the 6 practice cases: 270 synthetic
# checkpoints plus the 5 evaluable real checkpoints; this pair beats the
# persistence baseline on both sets). A is NEGATIVE: the 240-minute trend
# mean-reverts at this horizon, so the forecast fades it. B pulls the
# forecast toward the 24-hour mean.
TREND_WEIGHT = -0.30
MEAN_REVERSION_WEIGHT = 0.10

TREND_MINUTES = 240
MEAN_MINUTES = 1440
HORIZON_MINUTES = 240


def forecast(prices):
    """Predict the price 240 minutes after the last close in `prices`."""
    n = len(prices)
    if n == 0:
        return 0.0
    p0 = float(prices[-1])
    if n < 2 or p0 <= 0.0:
        return p0
    logs = [math.log(p) if p > 0.0 else math.log(p0) for p in prices]
    log_p0 = logs[-1]

    # Short trend: linear-regression slope of log price over the last
    # TREND_MINUTES minutes (or fewer if the series is short).
    m = min(TREND_MINUTES, n)
    seg = logs[n - m:]
    xs = list(range(m))
    mean_x = (m - 1) / 2.0
    mean_y = sum(seg) / m
    num = 0.0
    den = 0.0
    for x, y in zip(xs, seg):
        dx = x - mean_x
        num += dx * (y - mean_y)
        den += dx * dx
    slope = num / den if den > 0.0 else 0.0

    # Longer mean: mean log price over the last MEAN_MINUTES minutes.
    k = min(MEAN_MINUTES, n)
    mean24 = sum(logs[n - k:]) / k

    log_pred = log_p0 + TREND_WEIGHT * slope * HORIZON_MINUTES \
        + MEAN_REVERSION_WEIGHT * (mean24 - log_p0)
    # Guard against overflow on wild inputs; keep the guess in range.
    if log_pred > 16.0:
        log_pred = 16.0
    elif log_pred < 0.0:
        log_pred = 0.0
    price = math.exp(log_pred)
    if not (price > 0.0) or price >= 10000000.0:
        return p0
    return price


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input/"
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output/"
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(f for f in os.listdir(in_dir) if f.endswith(".in"))
    for name in names:
        case_id = name[:-3]
        with open(os.path.join(in_dir, name), "r", encoding="utf-8") as f:
            case = json.load(f)
        markets = case.get("markets") or []
        prices = []
        if markets:
            prices = markets[0].get("prices") or []
        price = forecast(prices)
        out = {"forecasts": [{"case": case.get("case", case_id),
                              "price": price}]}
        out_path = os.path.join(out_dir, case_id + ".out")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, separators=(",", ":"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
