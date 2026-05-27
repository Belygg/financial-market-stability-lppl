"""Generate `lppl_<TICKER>.pkl` files for additional tickers, matching the
schema of the existing 8 (NVDA, AAPL, MSFT, AMD, TSLA, SPY, QQQ, BTC-USD).

Schema (verified against existing files):
    {
        'short': DataFrame[time, price, pos_conf, neg_conf, _fits],
        'mid'  : DataFrame[time, price, pos_conf, neg_conf, _fits],
        'long' : DataFrame[time, price, pos_conf, neg_conf, _fits],
    }

`time` is a float trading-day ordinal (date.toordinal()).
Each per-day row in `_fits` is a list of LPPL fit dicts for nested sub-windows.

Defaults expand the universe by 12 diverse tickers:
    XLF, XLE, XLV, GLD, TLT, JPM, JNJ, XOM, COIN, ETH-USD, IWM, EEM

Run from the project root:
    python generate_lppl_pkls.py                # all 12, skip files that exist
    python generate_lppl_pkls.py --tickers XLF XLE
    python generate_lppl_pkls.py --force        # overwrite existing pkls
    python generate_lppl_pkls.py --workers 4    # tune mp parallelism

Compute cost: roughly 1-2 hours per ticker per scale at workers=8. For all 12
new tickers across 3 scales (short/mid/long) plan for an overnight run.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
import traceback
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
CACHE_DIR   = PROJECT_DIR / "project_files"

DEFAULT_NEW_TICKERS = [
    "XLF", "XLE", "XLV", "GLD", "TLT",
    "JPM", "JNJ", "XOM",
    "COIN", "ETH-USD",
    "IWM", "EEM",
]

START = "2010-01-01"
END   = "2025-12-31"

# Window configurations for each scale. window_size = largest nested window,
# smallest_window_size = smallest. outer_increment = step between anchoring
# days (1 = one anchor per day). inner_increment = step between sub-windows.
SCALES = {
    "short": dict(window_size=60,  smallest_window_size=30, outer_increment=1, inner_increment=5),
    "mid"  : dict(window_size=120, smallest_window_size=30, outer_increment=1, inner_increment=5),
    "long" : dict(window_size=240, smallest_window_size=60, outer_increment=1, inner_increment=5),
}

MAX_SEARCHES = 25     # annealing restarts per fit (same as the original Part A)


def fmt_dt(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}h{m:02d}m{s:02d}s" if h else f"{m:d}m{s:02d}s"


def fit_one_scale(log_price, time_ord, scale_name, params, workers):
    """Fit LPPL at one scale and return the `compute_indicators` DataFrame."""
    from lppls import lppls
    obs = __import__("numpy").array([time_ord, log_price])
    model = lppls.LPPLS(observations=obs)
    res = model.mp_compute_nested_fits(
        workers=workers,
        max_searches=MAX_SEARCHES,
        **params,
    )
    res_df = model.compute_indicators(res)
    return res_df


def process_ticker(ticker: str, workers: int, force: bool) -> tuple[bool, str]:
    import numpy as np
    import pandas as pd
    import yfinance as yf

    pkl_path = CACHE_DIR / f"lppl_{ticker}.pkl"
    if pkl_path.exists() and not force:
        return False, f"  {ticker}: pkl already exists at {pkl_path.name} - skipping (use --force to overwrite)"

    # -- Download OHLCV ----------------------------------------------------
    print(f"  {ticker}: downloading {START} to {END} ...", flush=True)
    df = yf.download(ticker, start=START, end=END, auto_adjust=True, progress=False)
    if df is None or len(df) == 0:
        return False, f"  {ticker}: no data returned by yfinance - skipping"

    if hasattr(df.columns, "levels"):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower).dropna()
    if "close" not in df.columns:
        return False, f"  {ticker}: missing 'close' column after download - skipping"

    print(f"  {ticker}: {len(df)} rows  ({df.index[0].date()} to {df.index[-1].date()})", flush=True)

    # -- Cache OHLCV in the notebook's schema ------------------------------
    # The notebook (LPPL_ML_Trading_hedge_work) loads ohlcv_<TICKER>.pkl as a
    # DataFrame[open,high,low,close,volume] with a DatetimeIndex named 'Date'
    # and drops any ticker missing one. We already have the data here, so save
    # it (unless it exists) -- otherwise the notebook re-downloads from Yahoo
    # and silently drops throttled tickers.
    ohlcv_cols = ["open", "high", "low", "close", "volume"]
    ohlcv_path = CACHE_DIR / f"ohlcv_{ticker}.pkl"
    if set(ohlcv_cols).issubset(df.columns) and (force or not ohlcv_path.exists()):
        ohlcv = df[ohlcv_cols].dropna().copy()
        ohlcv.index = pd.DatetimeIndex(ohlcv.index).as_unit("s")
        ohlcv.index.name = "Date"
        # Downcast pandas-3.0 StringDtype labels to plain object so the pkl
        # unpickles on older pandas (same portability fix applied to the
        # LPPL frames below; NDArrayBacked.__setstate__ raises otherwise).
        ohlcv.columns = pd.Index(ohlcv.columns.to_numpy(dtype=object), dtype=object)
        with open(ohlcv_path, "wb") as fh:
            pickle.dump(ohlcv, fh, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  {ticker}: cached OHLCV -> {ohlcv_path.name}", flush=True)

    log_price = np.log(df["close"].astype(float).values)
    time_ord  = np.array([d.toordinal() for d in df.index], dtype=float)

    # -- Fit each scale ----------------------------------------------------
    out = {}
    for scale_name, scale_params in SCALES.items():
        if len(df) < scale_params["window_size"] + 10:
            print(f"  {ticker}: skipping '{scale_name}' - only {len(df)} rows "
                  f"(needs {scale_params['window_size']}+).", flush=True)
            continue
        t_scale = time.time()
        print(f"  {ticker}: fitting '{scale_name}' "
              f"(window={scale_params['window_size']}, "
              f"smallest={scale_params['smallest_window_size']}, "
              f"workers={workers}) ...", flush=True)
        try:
            res_df = fit_one_scale(log_price, time_ord, scale_name, scale_params, workers)
        except Exception as e:
            print(f"  {ticker}/{scale_name}: FIT FAILED - {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            continue
        out[scale_name] = res_df
        elapsed = time.time() - t_scale
        print(f"  {ticker}: '{scale_name}' done in {fmt_dt(elapsed)} "
              f"-> {len(res_df)} rows", flush=True)

    if not out:
        return False, f"  {ticker}: no scales produced output - pkl NOT written"

    # -- Make pkl portable across pandas versions --------------------------
    # pandas 3.0 stores string column labels as the new "str" StringDtype,
    # which older pandas (<3.0) cannot unpickle (NotImplementedError in
    # NDArrayBacked.__setstate__). Downcast string Indexes to plain object so
    # the pkl loads on any pandas version.
    for _df in out.values():
        if isinstance(_df.columns.dtype, pd.StringDtype) or _df.columns.dtype == "str":
            _df.columns = pd.Index(_df.columns.to_numpy(dtype=object), dtype=object)
        if isinstance(_df.index.dtype, pd.StringDtype) or _df.index.dtype == "str":
            _df.index = pd.Index(_df.index.to_numpy(dtype=object), dtype=object)

    # -- Atomic write ------------------------------------------------------
    tmp_path = pkl_path.with_suffix(".pkl.tmp")
    with open(tmp_path, "wb") as fh:
        pickle.dump(out, fh)
    tmp_path.replace(pkl_path)
    return True, f"  {ticker}: wrote {pkl_path.name}  scales={list(out.keys())}"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_NEW_TICKERS,
                        help="Tickers to process (default: the curated 12-ticker expansion).")
    parser.add_argument("--workers", type=int, default=8,
                        help="Multiprocessing workers for mp_compute_nested_fits (default 8).")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing pkl files.")
    args = parser.parse_args(argv)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    overall_start = time.time()

    print(f"LPPL pkl generator")
    print(f"  cache dir : {CACHE_DIR}")
    print(f"  tickers   : {args.tickers}")
    print(f"  workers   : {args.workers}")
    print(f"  force     : {args.force}")
    print()

    summary = []
    for i, ticker in enumerate(args.tickers, 1):
        print(f"[{i}/{len(args.tickers)}] {ticker}  "
              f"(elapsed {fmt_dt(time.time() - overall_start)})", flush=True)
        try:
            ok, msg = process_ticker(ticker, args.workers, args.force)
        except KeyboardInterrupt:
            print("\nInterrupted by user.", flush=True)
            return 130
        except Exception as e:
            print(f"  {ticker}: UNEXPECTED ERROR - {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            ok, msg = False, f"  {ticker}: errored"
        summary.append((ticker, ok, msg))
        print(msg, flush=True)
        print()

    print("=" * 60)
    print(f"DONE in {fmt_dt(time.time() - overall_start)}")
    written  = [t for t, ok, _ in summary if ok]
    skipped  = [t for t, ok, m in summary if not ok and "skipping" in m]
    failed   = [t for t, ok, m in summary if not ok and "skipping" not in m]
    print(f"  wrote   : {written}")
    if skipped:
        print(f"  skipped : {skipped}")
    if failed:
        print(f"  failed  : {failed}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
