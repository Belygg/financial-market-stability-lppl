"""
LPPL-ML Trading Strategy with updated parameters:
- HORIZON: 30 days
- BUY_THRESH: +1%, SELL_THRESH: -1%
- MAX_LEVERAGE: 6.0
"""

import pickle
import os
import numpy as np
import pandas as pd
from datetime import datetime as dt
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report
import yfinance as yf
import ta
from lppls import lppls as lppls_lib

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

TICKERS = ['NVDA', 'AAPL', 'MSFT', 'AMD', 'TSLA', 'SPY', 'QQQ', 'BTC-USD']
START = '2010-01-01'
END = '2025-12-31'
CACHE_DIR = './lppl_cache'
os.makedirs(CACHE_DIR, exist_ok=True)

# Strategy parameters (UPDATED)
HORIZON = 30              # 30-day forward return
BUY_THRESH = 0.01        # +1% for BUY
SELL_THRESH = -0.01      # -1% for SELL
MAX_LEVERAGE = 6.0        # Increased from 3.0
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

# ─────────────────────────────────────────────────────────────────────────────
# Load or compute LPPL features
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 80)
print("LPPL-ML TRADING STRATEGY (Updated Parameters)")
print("=" * 80)
print(f"\nParameters:")
print(f"  HORIZON: {HORIZON} days")
print(f"  BUY_THRESH: +{BUY_THRESH*100:.1f}%")
print(f"  SELL_THRESH: {SELL_THRESH*100:.1f}%")
print(f"  MAX_LEVERAGE: {MAX_LEVERAGE}x")
print("\n" + "=" * 80)

print("\n[1/5] Downloading price data...")
raw = yf.download(TICKERS, start=START, end=END, auto_adjust=True, progress=False)

asset_data = {}
for ticker in TICKERS:
    try:
        df = raw.xs(ticker, axis=1, level=1)[['Open','High','Low','Close','Volume']].copy()
    except KeyError:
        df = raw[['Open','High','Low','Close','Volume']].copy()
    df.columns = ['open','high','low','close','volume']
    df = df.dropna()
    asset_data[ticker] = df
    print(f'  {ticker}: {len(df)} rows  ({df.index[0].date()} to {df.index[-1].date()})')

print("\n[2/5] Computing/loading LPPL features (this may take a while if not cached)...")

def run_lppl_all_windows(df):
    """Fits LPPL at three window scales."""
    time_ord = [pd.Timestamp.toordinal(t) for t in df.index]
    price = np.log(df['close'].values)
    obs = np.array([time_ord, price])
    model = lppls_lib.LPPLS(observations=obs)
    results = {}
    for cfg in WINDOW_CONFIGS:
        res = model.mp_compute_nested_fits(
            workers=8,
            window_size=cfg['window_size'],
            smallest_window_size=cfg['smallest_window_size'],
            outer_increment=1,
            inner_increment=5,
            max_searches=MAX_SEARCHES
        )
        res_df = model.compute_indicators(res)
        results[cfg['name']] = res_df
    return results

lppl_cache = {}
for ticker, df in asset_data.items():
    cache_path = os.path.join(CACHE_DIR, f'lppl_{ticker}.pkl')
    if os.path.exists(cache_path):
        print(f'  {ticker}: loaded from cache')
        with open(cache_path, 'rb') as fh:
            lppl_cache[ticker] = pickle.load(fh)
    else:
        print(f'  {ticker}: computing LPPL (~90 min)...')
        lppl_cache[ticker] = run_lppl_all_windows(df)
        with open(cache_path, 'wb') as fh:
            pickle.dump(lppl_cache[ticker], fh)

print("\n[3/5] Building feature matrix...")

def build_asset_df(ticker, ohlcv_df, lppl_results):
    df = ohlcv_df.copy()
    df['ticker'] = ticker
    for scale, rdf in lppl_results.items():
        r = rdf.copy()
        r.index = pd.to_datetime(r['time'].astype(int).apply(pd.Timestamp.fromordinal))
        df[f'pos_conf_{scale}'] = r['pos_conf'].reindex(df.index, fill_value=0.0)
        df[f'neg_conf_{scale}'] = r['neg_conf'].reindex(df.index, fill_value=0.0)
    return df

merged = {t: build_asset_df(t, asset_data[t], lppl_cache[t]) for t in TICKERS}
all_data = pd.concat(merged.values(), axis=0).sort_index()
print(f"  Total rows: {len(all_data)}")

def add_features(df):
    df = df.copy().sort_index()
    for scale in ['short', 'mid', 'long']:
        p = f'pos_conf_{scale}'
        n = f'neg_conf_{scale}'
        df[f'net_conf_{scale}'] = df[p] - df[n]
        df[f'conf_mom_{scale}'] = df[p].diff(5)
        df[f'pos_roll5_{scale}'] = df[p].rolling(5).mean()

    df['scale_agree_bull'] = (
        (df['pos_conf_short'] > 0.1) & (df['pos_conf_mid'] > 0.1) & (df['pos_conf_long'] > 0.1)
    ).astype(int)
    df['scale_agree_bear'] = (
        (df['neg_conf_short'] > 0.1) & (df['neg_conf_mid'] > 0.1) & (df['neg_conf_long'] > 0.1)
    ).astype(int)

    df['ret_1d'] = df['close'].pct_change(1)
    df['ret_5d'] = df['close'].pct_change(5)
    df['ret_20d'] = df['close'].pct_change(20)
    df['vol_20d'] = df['ret_1d'].rolling(20).std() * np.sqrt(252)

    df['rsi_14'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    df['macd'] = ta.trend.MACD(df['close']).macd_diff()
    df['bb_pct'] = ta.volatility.BollingerBands(df['close']).bollinger_pband()
    df['atr_14'] = ta.volatility.AverageTrueRange(
        df['high'], df['low'], df['close'], window=14).average_true_range()
    df['vol_ratio'] = df['volume'] / df['volume'].rolling(20).mean()

    return df

featured = pd.concat([add_features(merged[t]) for t in TICKERS], axis=0).sort_index()
featured = featured.dropna(subset=FEATURE_COLS)
print(f"  Feature matrix: {len(featured)} rows")

print("\n[4/5] Creating labels with updated thresholds...")

def add_labels(df):
    df = df.copy().sort_index()
    df['forward_ret'] = df['close'].pct_change(HORIZON).shift(-HORIZON)
    return df

featured = pd.concat(
    [add_labels(featured[featured['ticker'] == t]) for t in TICKERS]
).sort_index()
featured = featured.dropna(subset=['forward_ret'])

featured['target'] = featured['forward_ret'].apply(
    lambda r: 2 if r > BUY_THRESH else (0 if r < SELL_THRESH else 1)
)

print(f"\n  Label distribution (HORIZON={HORIZON}d, BUY={BUY_THRESH*100:.1f}%, SELL={SELL_THRESH*100:.1f}%):")
label_counts = featured['target'].value_counts().sort_index()
print(f"    SELL (0): {label_counts.get(0, 0):6d}")
print(f"    HOLD (1): {label_counts.get(1, 0):6d}")
print(f"    BUY  (2): {label_counts.get(2, 0):6d}")

print("\n[5/5] Walk-forward training and backtesting...")

featured_sorted = featured.sort_index()
dates_all = featured_sorted.index.unique().sort_values()
all_records = []

for i in range(0, len(dates_all) - TRAIN_DAYS - TEST_STEP, TEST_STEP):
    train_dates = dates_all[i : i + TRAIN_DAYS]
    test_dates = dates_all[i + TRAIN_DAYS : i + TRAIN_DAYS + TEST_STEP]

    train = featured_sorted[featured_sorted.index.isin(train_dates)]
    test = featured_sorted[featured_sorted.index.isin(test_dates)]
    if len(train) < 100 or len(test) == 0:
        continue

    base = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        use_label_encoder=False, eval_metric='mlogloss', random_state=42
    )
    model = CalibratedClassifierCV(base, method='isotonic', cv=3)
    model.fit(train[FEATURE_COLS].values, train['target'].values)

    proba = model.predict_proba(test[FEATURE_COLS].values)
    rec = test[['ticker', 'close', 'vol_20d', 'target']].copy()
    rec[['p_sell', 'p_hold', 'p_buy']] = proba
    all_records.append(rec)

val_df = pd.concat(all_records).sort_index()
y_pred = val_df[['p_sell', 'p_hold', 'p_buy']].values.argmax(axis=1)

print("\nClassification Report:")
print(classification_report(val_df['target'], y_pred, target_names=['sell', 'hold', 'buy']))

# ─────────────────────────────────────────────────────────────────────────────
# Generate signals with updated MAX_LEVERAGE
# ─────────────────────────────────────────────────────────────────────────────

def generate_signal(row):
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

sig_cols = val_df.apply(generate_signal, axis=1)
val_df = pd.concat([val_df, sig_cols], axis=1)

print("\nLatest Signals by Ticker:")
latest = val_df.groupby('ticker')[['signal', 'confidence', 'suggested_exposure', 'max_leverage']].last()
print(latest.to_string())

# ─────────────────────────────────────────────────────────────────────────────
# Backtest
# ─────────────────────────────────────────────────────────────────────────────

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

sharpe = (daily['strat_ret'].mean() / daily['strat_ret'].std()) * np.sqrt(252)
max_dd = (cum_s / cum_s.cummax() - 1).min()

print("\n" + "=" * 80)
print("BACKTEST RESULTS")
print("=" * 80)
print(f"Sharpe Ratio:    {sharpe:.2f}")
print(f"Max Drawdown:    {max_dd:.1%}")
print(f"Total Return:    {cum_s.iloc[-1]:.2f}x")
print(f"B&H Return:      {cum_bah.iloc[-1]:.2f}x")
print(f"Outperformance:  {(cum_s.iloc[-1] / cum_bah.iloc[-1] - 1):.1%}")
print("=" * 80)

# Save results
val_df.to_csv('strategy_results_updated.csv')
print(f"\nResults saved to: strategy_results_updated.csv")
print(f"Total signal rows: {len(val_df)}")
print(f"  BUY:  {(val_df['signal'] == 'buy').sum()}")
print(f"  SELL: {(val_df['signal'] == 'sell').sum()}")
print(f"  HOLD: {(val_df['signal'] == 'hold').sum()}")
