# Financial Market Stability under Manipulation & Crisis Conditions

ICEF coursework — Grigory Belyy.

## What's in this repo

- `project_files/LPPL_ML_Trading.ipynb` — main notebook. End-to-end pipeline:
  - **Part A** — single-asset LPPL exploration on NVDA (parameter fits, nested
    windows, bubble confidence indicators)
  - **Part B** — 20-ticker multi-asset pipeline (yfinance OHLCV → multi-scale
    LPPL → DS LPPLS Confidence feature extraction → walk-forward XGB →
    per-ticker dose-response policy)
  - **Part C** — per-asset equity curves, drawdowns, CAPM alpha
- `generate_lppl_pkls.py` — pre-generates per-ticker LPPL nested-window fits
  (slow: ~1–2 h per ticker per scale; run once, results are cached as
  `project_files/lppl_<TICKER>.pkl`)
- `build_ds_confidence_features.py` — reaggregates the cached LPPL fits into
  DS LPPLS Confidence-style features (Sornette filter + cross-scale
  confluence). Fast: ~1 minute for all 20 tickers.
- `Belyy Grigory Project ICEF28.pdf` — project proposal
- `general direction (not structured).txt` — initial direction notes (Russian)
- `project_files/data/NVDA_.csv` — sample CSV used by Part A

## Strategy headline

Per-ticker dose-response policy on multi-scale LPPL signals, with vol-target
overlay. Full period 2015-09 → 2025-11, equal-weight portfolio of 20 tickers,
10 bps round-trip transaction cost.

| Strategy                       | Sharpe | CAGR  | maxDD   | CAPM α (t-stat) |
|--------------------------------|--------|-------|---------|-----------------|
| Buy & Hold (20-asset EW)       | 1.166  | 37.3% | -49.4%  | 10.1% (3.49)    |
| Buy & Hold + Vol-target @ 0.20 | 1.402  | 38.1% | -35.9%  | —               |
| **Per-ticker policy + VT@0.20**| **1.437** | **39.6%** | **-35.0%** | **14.25% (4.33)** |

Out-of-sample test (2021-10 → 2025-11) Sharpe + VT: **1.177** vs B&H + VT 1.107.

## Method summary

1. Fit LPPL log-periodic power law in nested sub-windows at 3 horizons (60,
   120, 240 trading days) for each ticker (`generate_lppl_pkls.py`).
2. Aggregate the 62 sub-window fits per day into DS LPPLS Confidence-style
   features (`build_ds_confidence_features.py`): fraction of fits passing
   Sornette's filter (`m ∈ [0.1, 0.9]`, `ω ∈ [6, 13]`, damping ≥ 0.8,
   ≥ 2.5 oscillations); cross-scale confluence; median bubble parameters.
3. Train walk-forward XGBoost on `days_to_next_15%_drawdown`. (The XGB output
   itself is not used in the production policy — see note below.)
4. Per-ticker dose-response policy: each ticker picks its preferred
   positive-bubble signal feed (`max3`, `confluence`, `strict`, or `mixed`)
   and an anti-bubble lever; per-ticker thresholds + strengths fitted on the
   first 60% of dates and validated on the remaining 40%.
5. Equal-weight portfolio with vol-target overlay (target annualized vol 20%,
   capped at 2× leverage).

## Method note: XGB regression doesn't drive Sharpe

A walk-forward XGB regressor was trained to predict the number of days until
the next 15% drawdown. Per-ticker R² of those predictions is **negative on
all 20 tickers** (range -6.4 to -0.16, median -0.35). Forcing the policy's
tc-channel to zero everywhere lifts OOS Sharpe by ~0.003 and is the cleanest
methodology. The strategy's alpha comes from the LPPL/DS Confidence signals
themselves, plus the vol-target overlay — not from any ML head on top.

The `w_med` feature (median LPPL angular log-frequency ω across qualified
sub-window fits) ranks #2 in XGB feature importance, behind `vol_20d`, so the
LPPL fits are clearly informative; the regression target framing just doesn't
suit them.

## Reproducing the results

```bash
pip install lppls yfinance xgboost scikit-learn ta pandas numpy matplotlib

# 1. Generate LPPL nested-window fits (SLOW: hours per ticker — overnight job)
python generate_lppl_pkls.py

# 2. Aggregate to DS LPPLS Confidence features (~1 minute total)
python build_ds_confidence_features.py

# 3. Open and run the notebook
jupyter notebook project_files/LPPL_ML_Trading.ipynb
```

Walk-forward XGB inside the notebook takes ~1 minute on 8 cores. The notebook
itself takes ~5 minutes if LPPL pkls are already cached.

## Universe

```
NVDA AAPL MSFT AMD TSLA           (US large-cap tech)
SPY  QQQ  IWM  EEM                (broad equity ETFs)
XLF  XLE  XLV                     (sector ETFs)
JPM  JNJ  XOM                     (defensive single names)
GLD  TLT                          (commodities, bonds)
BTC-USD ETH-USD COIN              (crypto)
```
