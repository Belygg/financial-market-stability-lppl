"""Aggregate the cached _fits across all 3 scales into DS LPPLS Confidence-style
features. No LPPL re-fitting required — the existing pkls contain 8 (short) +
18 (mid) + 36 (long) = 62 sub-window fits per day per ticker.

New features per day (in addition to existing pos_conf_{scale} etc.):

  Fraction-based (across ALL 62 sub-window fits, Sornette filter):
    frac_pos_strict   -- fraction qualifying as positive bubble
    frac_neg_strict   -- fraction qualifying as negative bubble
    frac_tc_30d       -- fraction of qualified fits predicting tc within 30 days
    frac_tc_60d       -- fraction predicting tc within 60 days
    n_qualified       -- total qualified fits (signal strength)

  TC distribution (positive-bubble fits only):
    tc_med_pos        -- median predicted tc-days (more robust than XGB pred)
    tc_iqr_pos        -- IQR of tc (disagreement among fits)

  Bubble parameter distribution:
    m_med             -- median m across qualified fits
    w_med             -- median w across qualified fits

  Cross-scale confluence:
    confluence_pos    -- pos_conf_short * pos_conf_mid * pos_conf_long
    confluence_neg    -- same for negative

Writes per-ticker `lppl_features_<ticker>.pkl` with a DataFrame[Date -> features].
"""
from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent
CACHE   = PROJECT / "project_files"
OUTDIR  = PROJECT / "project_files" / "ds_features"
OUTDIR.mkdir(exist_ok=True)

TICKERS = [
    'NVDA', 'AAPL', 'MSFT', 'AMD', 'TSLA', 'SPY', 'QQQ', 'BTC-USD',
    'XLF', 'XLE', 'XLV', 'GLD', 'TLT', 'JPM', 'JNJ', 'XOM',
    'COIN', 'ETH-USD', 'IWM', 'EEM',
]

# Sornette filter conditions
M_LO, M_HI = 0.10, 0.90
W_LO, W_HI = 6.0,  13.0
DAMP_MIN   = 0.8     # damping D >= 0.8 (loosened from Sornette's 1.0 — empirical robustness)
OSC_MIN    = 2.5     # at least ~2.5 log-periodic oscillations
TC_MAX_DAYS = 252.0  # only count tc predictions within next trading year

NUM_AGG = ('frac_pos_strict','frac_neg_strict','frac_tc_30d','frac_tc_60d',
           'n_qualified','tc_med_pos','tc_iqr_pos','m_med','w_med',
           'confluence_pos','confluence_neg')


def fit_qualifies(fit, t_now):
    """Sornette filter. Returns (qualifies, is_positive_bubble, tc_days)."""
    try:
        m, w, b, tc, t2 = fit['m'], fit['w'], fit['b'], fit['tc'], fit['t2']
        O, D = fit.get('O', np.nan), fit.get('D', np.nan)
    except (KeyError, TypeError):
        return False, False, np.nan
    if not (np.isfinite(m) and np.isfinite(w) and np.isfinite(tc) and np.isfinite(b)):
        return False, False, np.nan
    if not (M_LO <= m <= M_HI and W_LO <= w <= W_HI):
        return False, False, np.nan
    # tc must be in (t2, t2 + TC_MAX_DAYS] — a real forward forecast within the year
    tc_days = float(tc - t2)
    if not (0.0 < tc_days <= TC_MAX_DAYS):
        return False, False, np.nan
    # tc must also be > t_now (current day) — no backward-looking fits
    if tc < t_now:
        return False, False, np.nan
    # Sornette damping + oscillation filters (use cached values if present)
    if np.isfinite(D) and abs(D) < DAMP_MIN:
        return False, False, np.nan
    if np.isfinite(O) and abs(O) < OSC_MIN:
        return False, False, np.nan
    # Positive bubble: B<0  (log-price acceleration upward toward crash)
    # Negative bubble: B>0
    is_pos = b < 0
    return True, is_pos, tc_days


def aggregate_day(t_now, all_fits, pos_confs_3scale, neg_confs_3scale):
    """Aggregate all sub-window fits for one (day, ticker) into features."""
    n_total = len(all_fits)
    if n_total == 0:
        return dict.fromkeys(NUM_AGG, 0.0) | {'frac_pos_strict': 0.0,
                                              'frac_neg_strict': 0.0,
                                              'n_qualified': 0,
                                              'tc_med_pos': 252.0,
                                              'tc_iqr_pos': 0.0}
    n_pos = n_neg = n_qual = 0
    tc_pos = []; tc_all = []
    ms = []; ws = []
    n_tc30 = n_tc60 = 0
    for fit in all_fits:
        q, is_pos, tc_days = fit_qualifies(fit, t_now)
        if not q:
            continue
        n_qual += 1
        ms.append(fit['m']); ws.append(fit['w'])
        tc_all.append(tc_days)
        if tc_days <= 30: n_tc30 += 1
        if tc_days <= 60: n_tc60 += 1
        if is_pos:
            n_pos += 1
            tc_pos.append(tc_days)
        else:
            n_neg += 1
    # Fractions across the WHOLE pool of fits (n_total), not just qualified —
    # this is the DS LPPLS Confidence definition (low n_qual = low confidence).
    out = dict(
        frac_pos_strict = n_pos / n_total,
        frac_neg_strict = n_neg / n_total,
        frac_tc_30d     = n_tc30 / n_total,
        frac_tc_60d     = n_tc60 / n_total,
        n_qualified     = int(n_qual),
        tc_med_pos      = float(np.median(tc_pos)) if tc_pos else 252.0,
        tc_iqr_pos      = float(np.percentile(tc_pos, 75) - np.percentile(tc_pos, 25)) if len(tc_pos) >= 2 else 0.0,
        m_med           = float(np.median(ms)) if ms else 0.0,
        w_med           = float(np.median(ws)) if ws else 0.0,
        confluence_pos  = float(pos_confs_3scale[0] * pos_confs_3scale[1] * pos_confs_3scale[2]),
        confluence_neg  = float(neg_confs_3scale[0] * neg_confs_3scale[1] * neg_confs_3scale[2]),
    )
    return out


def process_ticker(ticker):
    pkl = CACHE / f"lppl_{ticker}.pkl"
    if not pkl.exists():
        return False, f"  {ticker}: no lppl pkl"
    with open(pkl, 'rb') as f:
        lppl = pickle.load(f)

    # Build per-day union of fits across scales, using ordinal `time` as key.
    # Each scale's DataFrame is indexed by integer trading-day ordinal in `time`.
    scales = ['short', 'mid', 'long']
    if not all(s in lppl for s in scales):
        return False, f"  {ticker}: missing scale(s) {set(scales) - set(lppl.keys())}"

    # Index each by time (already int trading-day ordinal as float)
    by_scale = {}
    for s in scales:
        df = lppl[s].copy()
        df = df.set_index(df['time'].astype(int))
        by_scale[s] = df

    # Use union of all times
    all_times = sorted(set().union(*(set(by_scale[s].index) for s in scales)))

    rows = []
    for t in all_times:
        # Concatenate all sub-window fits across scales for this day
        all_fits = []
        pcs = []; ncs = []
        for s in scales:
            df = by_scale[s]
            if t in df.index:
                row = df.loc[t]
                f = row.get('_fits', [])
                if isinstance(f, list):
                    all_fits.extend(f)
                pcs.append(float(row.get('pos_conf', 0.0)))
                ncs.append(float(row.get('neg_conf', 0.0)))
            else:
                pcs.append(0.0); ncs.append(0.0)
        feats = aggregate_day(float(t), all_fits, pcs, ncs)
        feats['time'] = t
        rows.append(feats)
    out = pd.DataFrame(rows).set_index('time')
    # Convert ordinal index to Timestamp
    out.index = pd.to_datetime([pd.Timestamp.fromordinal(int(t)) for t in out.index])
    out.index.name = 'Date'

    # Portability: coerce pandas 3.x StringDtype columns/index to plain object so
    # the pkl unpickles on older pandas (notebook kernel can be < 3.0).
    if isinstance(out.columns.dtype, pd.StringDtype) or str(out.columns.dtype) == 'str':
        out.columns = pd.Index(out.columns.to_numpy(dtype=object), dtype=object)
    if isinstance(out.index.dtype, pd.StringDtype) or str(out.index.dtype) == 'str':
        out.index = pd.Index(out.index.to_numpy(dtype=object), dtype=object)

    out_path = OUTDIR / f"lppl_features_{ticker}.pkl"
    with open(out_path, 'wb') as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    return True, f"  {ticker}: {len(out)} rows  cols={list(out.columns)}  -> {out_path.name}"


def main():
    print(f"DS LPPLS Confidence feature extractor")
    print(f"  output dir: {OUTDIR}")
    print(f"  tickers   : {len(TICKERS)}")
    print()
    t0 = time.time()
    for i, tk in enumerate(TICKERS, 1):
        t1 = time.time()
        ok, msg = process_ticker(tk)
        print(f"[{i:2d}/{len(TICKERS)}] {tk:8s}  {msg}  ({time.time()-t1:.1f}s)", flush=True)
    print(f"\nTotal: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    sys.exit(main())
