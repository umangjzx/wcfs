# Colab v2 — 3-target model (archived, NOT served)

Trained on Colab (T4, CPU-bound LightGBM) over **2025-10-01 → 2026-02-15**,
`--num-boost 400 --stride 3`, targets **pm25 + pm10 + no2** → real multi-pollutant
CPCB AQI. Delivered 2026-08-29.

The API serves `../registry/` (pm25-only). This copy is kept for the record and
as the starting point for a corrected retrain — it is **not** wired into serving.

## Why it is not the served model

Its walk-forward backtest (`backtest_metrics.json` here) regressed on calibration
versus the shipped pm25-only model:

| | served (`../registry/`) | this (colab v2) |
| --- | --- | --- |
| overall MAE | 37.2 | 30.8 |
| overall RMSE | 54.8 | 38.6 |
| **bias** | **+4.2** | **+21.6** |
| **P10–P90 coverage** (target 0.80) | **0.72** | **0.44** |
| AQI category accuracy | 0.48 | 0.50 |
| Very Poor CSI | 0.55 | 0.54 |
| Severe CSI | 0.17 | **0.31** |
| skill vs persistence | positive at every horizon | negative at h=1, 2, 24 |

The two runs used different feature matrices (the Colab ingest pulled a fuller
Jan–Feb observation set — persistence-baseline MAE 54→41, severe-event count
22k→5.8k), so MAE is not directly comparable. The **+21.6 bias** (dashboard AQI
runs ~1 category high) and **0.44 interval coverage** (bands too tight — the
conformal margins came out at ~0.1–0.75 µg/m³) are real and would degrade the
demo. `severe` event skill did improve.

Probable causes: `--num-boost 400` (vs ~220 for the shipped model) overfit the
Oct–Nov pollution level; the random-holdout conformal split is over-optimistic
under the Oct→Feb distribution shift.

## To promote a fixed v2

Re-run the Colab pipeline with:

- `--num-boost ~220` (match the shipped model's fit)
- a **time-blocked** conformal calibration split instead of the random holdout
  (`models/baseline_lgbm.py`, `_CAL_FRACTION` / `train_target`), so calibration
  sees the same distribution shift as the test folds

then confirm the backtest lands near **bias ≈ +4** and **coverage ≈ 0.7** for all
three pollutants before copying the files into `../registry/` and updating
`registry_index.json` there to `["pm25", "pm10", "no2"]`.
