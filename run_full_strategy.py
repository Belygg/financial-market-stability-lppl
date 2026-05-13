#!/usr/bin/env python3
"""
Complete LPPL-ML Trading Strategy Pipeline
Runs without dependency on Part A or CSV files
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import yfinance as yf
import ta
from datetime import datetime
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report
from lppls import lppls as lppls_lib

print("\n" + "="*80)
print("LPPL-ML TRADING STRATEGY - FULL PIPELINE")
print("="*80)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

TICKERS = ['NVDA', 'AAPL', 'MSFT', 'AMD', 'TSLA', 'SPY', 'QQQ', 'BTC-USD']
START = '2010-01-01'
END = '2025-12-31'
PROJECT_DIR = './project_files'
CACHE_DIR = PROJECT_DIR

# Strategy parameters (UPDATED)
HORIZON = 30
BUY_THRESH = 0.01
SELL_THRESH = -0.01
MAX_LEVERAGE = 6.0
TARGET_VOL = 0.15
MAX_EXPOSURE = 1.0

# Walk-forward parameters
TRAIN_DAYS = 1500
TEST_STEP = 60
MAX_SEARCHES = 25

WINDOW_CONFIGS = [
    {'name': 'short', 'window_size': 60, 'smallest_window_size': 20},
    {'name': 'mid', 'window_size': 120, 'smallest_window_size': 30},
    {'name': 'long', 'window_size': 240, 'smallest_window_size': 60},
]

FEATURE_COLS = [
    'pos_conf_short', 'neg_conf_short',
    'pos_conf_mid', 'neg_conf_mid',
    'pos_conf_long', 'neg_conf_long',
    'net_conf_short', 'net_conf_mid', 'net_conf_long',
    'conf_mom_short', 'conf_mom_mid', 'conf_mom_long',
    'pos_roll5_short', 'pos_roll5_mid', 'pos_roll5_long',
    'scale_agree_bull', 'scale_agree_bear',
    'ret_1d', 'ret_5d', 'ret_20d', 'vol_20d',
    'rsi_14', 'macd', 'bb_pct', 'atr_14', 'vol_ratio'
]

os.makedirs(CACHE_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Download Price Data
# ─────────────────────────────────────────────────────────────────────────────

print("\n[STEP 1/7] Downloading price data for 8 assets...")
asset_data = {}
for ticker in TICKERS:
    try:
        print(f"  Downloading {ticker}...", end=" ")
        df = yf.download(ticker, start=START, end=END, auto_adjust=True, progress=False)

        # Handle both single and multi-column formats
        if isinstance(df.columns, pd.MultiIndex):
            df = df.xs(ticker, axis=1, level=1)

        df.columns = [col.lower() for col in df.columns]
        df = df[['open', 'high', 'low', 'close', 'volume']].dropna()
        asset_data[ticker] = df
        print(f"✓ {len(df)} rows ({df.index[0].date()} to {df.index[-1].date()})")
    except Exception as e:
        print(f"✗ ERROR: {e}")
        sys.exit(1)

print(f"\n✓ Successfully loaded {len(asset_data)} assets")
if len(asset_data) != len(TICKERS):
    print(f"✗ ERROR: Expected {len(TICKERS)} assets, got {len(asset_data)}")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Load or Compute LPPL Cache
# ─────────────────────────────────────────────────────────────────────────────

print("\n[STEP 2/7] Loading LPPL cache from project_files/...")
lppl_cache = {}
missing_cache = []

for ticker in TICKERS:
    cache_path = os.path.join(CACHE_DIR, f'lppl_{ticker}.pkl')
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as fh:
                lppl_cache[ticker] = pickle.load(fh)
            print(f"  {ticker}: ✓ loaded")
        except Exception as e:
            print(f"  {ticker}: ✗ ERROR loading cache: {e}")
            missing_cache.append(ticker)
    else:
        print(f"  {ticker}: ✗ cache file not found at {cache_path}")
        missing_cache.append(ticker)

if missing_cache:
    print(f"\n✗ ERROR: Missing LPPL cache for: {missing_cache}")
    print(f"   Make sure lppl_*.pkl files exist in {CACHE_DIR}/")
    sys.exit(1)

print(f"\n✓ Loaded LPPL cache for all {len(lppl_cache)} tickers")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Build Feature Matrix
# ─────────────────────────────────────────────────────────────────────────────

print("\n[STEP 3/7] Building feature matrix...")

def build_asset_df(ticker, ohlcv_df, lppl_results):
    """Joins LPPL confidence columns onto OHLCV data"""
    df = ohlcv_df.copy()
    df['ticker'] = ticker

    for scale, rdf in lppl_results.items():
        r = rdf.copy()
        # Convert time ordinals to datetime
        r.index = pd.to_datetime(r['time'].astype(int).apply(pd.Timestamp.fromordinal))
        # Reindex to match ohlcv dates with forward-fill
        df[f'pos_conf_{scale}'] = r['pos_conf'].reindex(df.index, fill_value=0.0)
        df[f'neg_conf_{scale}'] = r['neg_conf'].reindex(df.index, fill_value=0.0)

    return df

try:
    merged = {}
    for ticker in TICKERS:
        df = build_asset_df(ticker, asset_data[ticker], lppl_cache[ticker])
        merged[ticker] = df
        print(f"  {ticker}: ✓ {len(df)} rows with LPPL features")

    all_data = pd.concat(merged.values(), axis=0).sort_index()
    print(f"\n✓ Combined dataset: {len(all_data)} rows")
except Exception as e:
    print(f"\n✗ ERROR building features: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: Add Technical Features
# ─────────────────────────────────────────────────────────────────────────────

print("\n[STEP 4/7] Adding technical features...")

def add_features(df):
    """Add LPPL-derived and technical features"""
    df = df.copy().sort_index()

    # Multi-scale LPPL features
    for scale in ['short', 'mid', 'long']:
        p = f'pos_conf_{scale}'
        n = f'neg_conf_{scale}'
        df[f'net_conf_{scale}'] = df[p] - df[n]
        df[f'conf_mom_{scale}'] = df[p].diff(5)
        df[f'pos_roll5_{scale}'] = df[p].rolling(5).mean()

    # Cross-scale agreement
    df['scale_agree_bull'] = (
        (df['pos_conf_short'] > 0.1) & (df['pos_conf_mid'] > 0.1) & (df['pos_conf_long'] > 0.1)
    ).astype(int)
    df['scale_agree_bear'] = (
        (df['neg_conf_short'] > 0.1) & (df['neg_conf_mid'] > 0.1) & (df['neg_conf_long'] > 0.1)
    ).astype(int)

    # Returns and volatility
    df['ret_1d'] = df['close'].pct_change(1)
    df['ret_5d'] = df['close'].pct_change(5)
    df['ret_20d'] = df['close'].pct_change(20)
    df['vol_20d'] = df['ret_1d'].rolling(20).std() * np.sqrt(252)

    # Technical indicators
    df['rsi_14'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    df['macd'] = ta.trend.MACD(df['close']).macd_diff()
    df['bb_pct'] = ta.volatility.BollingerBands(df['close']).bollinger_pband()
    df['atr_14'] = ta.volatility.AverageTrueRange(
        df['high'], df['low'], df['close'], window=14).average_true_range()
    df['vol_ratio'] = df['volume'] / df['volume'].rolling(20).mean()

    return df

try:
    featured = pd.concat([add_features(merged[t]) for t in TICKERS], axis=0).sort_index()
    featured = featured.dropna(subset=FEATURE_COLS)
    print(f"✓ Feature matrix: {len(featured)} rows × {len(FEATURE_COLS)} features")
except Exception as e:
    print(f"✗ ERROR adding features: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: Create Labels
# ─────────────────────────────────────────────────────────────────────────────

print("\n[STEP 5/7] Creating labels with updated thresholds...")

def add_labels(df):
    """Add forward-looking labels"""
    df = df.copy().sort_index()
    df['forward_ret'] = df['close'].pct_change(HORIZON).shift(-HORIZON)
    return df

try:
    featured = pd.concat([add_labels(featured[featured['ticker'] == t]) for t in TICKERS], axis=0).sort_index()
    featured = featured.dropna(subset=['forward_ret'])

    featured['target'] = featured['forward_ret'].apply(
        lambda r: 2 if r > BUY_THRESH else (0 if r < SELL_THRESH else 1)
    )

    label_dist = featured['target'].value_counts().sort_index()
    print(f"✓ Label distribution (HORIZON={HORIZON}d, BUY={BUY_THRESH*100:.1f}%, SELL={SELL_THRESH*100:.1f}%):")
    print(f"    SELL (0): {label_dist.get(0, 0):>6d}")
    print(f"    HOLD (1): {label_dist.get(1, 0):>6d}")
    print(f"    BUY  (2): {label_dist.get(2, 0):>6d}")
except Exception as e:
    print(f"✗ ERROR creating labels: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: Walk-Forward Training
# ─────────────────────────────────────────────────────────────────────────────

print("\n[STEP 6/7] Walk-forward training (this may take several minutes)...")

try:
    featured_sorted = featured.sort_index()
    dates_all = featured_sorted.index.unique().sort_values()
    all_records = []
    fold_count = 0

    for i in range(0, len(dates_all) - TRAIN_DAYS - TEST_STEP, TEST_STEP):
        train_dates = dates_all[i : i + TRAIN_DAYS]
        test_dates = dates_all[i + TRAIN_DAYS : i + TRAIN_DAYS + TEST_STEP]

        train = featured_sorted[featured_sorted.index.isin(train_dates)]
        test = featured_sorted[featured_sorted.index.isin(test_dates)]

        if len(train) < 100 or len(test) == 0:
            continue

        fold_count += 1
        print(f"  Fold {fold_count}: Training on {len(train)} rows, testing on {len(test)} rows...", end=" ")

        # Train model
        base = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric='mlogloss', random_state=42
        )
        model = CalibratedClassifierCV(base, method='isotonic', cv=3)
        model.fit(train[FEATURE_COLS].values, train['target'].values)

        # Predict
        proba = model.predict_proba(test[FEATURE_COLS].values)
        rec = test[['ticker', 'close', 'vol_20d', 'target']].copy()
        rec[['p_sell', 'p_hold', 'p_buy']] = proba
        all_records.append(rec)
        print("✓")

    val_df = pd.concat(all_records).sort_index()
    y_pred = val_df[['p_sell', 'p_hold', 'p_buy']].values.argmax(axis=1)

    print(f"\n✓ Completed {fold_count} folds with {len(val_df)} validation records")
    print("\nClassification Report:")
    print(classification_report(val_df['target'], y_pred, target_names=['SELL', 'HOLD', 'BUY']))

except Exception as e:
    print(f"\n✗ ERROR in walk-forward training: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7: Generate Signals & Backtest
# ─────────────────────────────────────────────────────────────────────────────

print("\n[STEP 7/7] Generating signals and backtesting...")

def generate_signal(row):
    """Convert model probabilities to trading signals"""
    idx = int(np.argmax([row['p_sell'], row['p_hold'], row['p_buy']]))
    signal = {0: 'sell', 1: 'hold', 2: 'buy'}[idx]
    confidence = float(max(row['p_sell'], row['p_hold'], row['p_buy']))

    if signal == 'hold':
        return pd.Series({'signal': 'hold', 'confidence': round(confidence, 4),
                          'suggested_exposure': 0.0, 'max_leverage': 1.0})

    vol = row['vol_20d'] if row['vol_20d'] > 0 else TARGET_VOL
    vol_scalar = min(1.0, TARGET_VOL / vol)
    exposure = round(confidence * MAX_EXPOSURE * vol_scalar, 4)

    edge = (confidence - 0.5) * 2
    leverage = round(1.0 + edge * (MAX_LEVERAGE - 1.0), 2)

    return pd.Series({'signal': signal, 'confidence': round(confidence, 4),
                      'suggested_exposure': exposure, 'max_leverage': leverage})

try:
    sig_cols = val_df.apply(generate_signal, axis=1)
    val_df = pd.concat([val_df, sig_cols], axis=1)

    print("\n✓ Latest Signals by Ticker:")
    latest = val_df.groupby('ticker')[['signal', 'confidence', 'suggested_exposure', 'max_leverage']].last()
    print(latest.to_string())

    # Backtest
    val_df = val_df.sort_index()
    val_df['ret_1d_actual'] = val_df.groupby('ticker')['close'].pct_change(1)
    val_df['strat_ret'] = 0.0

    buy_mask = val_df['signal'] == 'buy'
    sell_mask = val_df['signal'] == 'sell'
    val_df.loc[buy_mask, 'strat_ret'] = (
        val_df.loc[buy_mask, 'ret_1d_actual'] * val_df.loc[buy_mask, 'suggested_exposure'])
    val_df.loc[sell_mask, 'strat_ret'] = (
        -val_df.loc[sell_mask, 'ret_1d_actual'] * val_df.loc[sell_mask, 'suggested_exposure'])

    daily = val_df.groupby(level=0)[['strat_ret', 'ret_1d_actual']].mean()
    cum_s = (1 + daily['strat_ret']).cumprod()
    cum_bah = (1 + daily['ret_1d_actual']).cumprod()

    sharpe = (daily['strat_ret'].mean() / daily['strat_ret'].std()) * np.sqrt(252) if daily['strat_ret'].std() > 0 else 0
    max_dd = (cum_s / cum_s.cummax() - 1).min()

    print("\n" + "="*80)
    print("BACKTEST RESULTS")
    print("="*80)
    print(f"Sharpe Ratio:        {sharpe:>8.2f}")
    print(f"Max Drawdown:        {max_dd:>7.1%}")
    print(f"Strategy Return:     {cum_s.iloc[-1]:>8.2f}x")
    print(f"Buy & Hold Return:   {cum_bah.iloc[-1]:>8.2f}x")
    print(f"Outperformance:      {(cum_s.iloc[-1] / cum_bah.iloc[-1] - 1):>7.1%}")
    print("="*80)

    # Save results
    val_df.to_csv('strategy_results_updated.csv')
    print(f"\n✓ Results saved to: strategy_results_updated.csv")
    print(f"   Total signal rows: {len(val_df)}")
    print(f"   BUY signals:  {(val_df['signal'] == 'buy').sum()}")
    print(f"   SELL signals: {(val_df['signal'] == 'sell').sum()}")
    print(f"   HOLD signals: {(val_df['signal'] == 'hold').sum()}")

    print("\n✓ PIPELINE COMPLETE - ALL STEPS SUCCESSFUL!")

except Exception as e:
    print(f"\n✗ ERROR in signal generation/backtest: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
