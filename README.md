# BTC Cent Forecast

Forecast Bitcoin's US-dollar price 240 minutes (4 hours) after each
checkpoint. The program reads one case per run from `/input/<id>.in`
(4,320 one-minute closes over the 72 hours up to the checkpoint) and writes
one guess to `/output/<id>.out`.

## Method

Work in log prices. Take the last close, the linear-regression slope of log
price over the last 240 minutes, and the mean log price over the last 1,440
minutes (24 hours). The forecast is

    log(P) = log(P0) + A * (slope * 240) + B * (mean24h - log(P0))

with damped weights A and B tuned on practice backtests. When the series is
too short or flat the forecast falls back to the last close (persistence).

Deterministic, Python 3.13 standard library only, no network, well under the
8-second limit per case.

## Run

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```
