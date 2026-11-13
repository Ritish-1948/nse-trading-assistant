"""
Phase 1 — Historical Data & Feature Engineering (data_features.py)
==================================================================
Downloads 5 to 10 years of daily OHLCV data via `yfinance` (defaulting to NSE
tickers such as RELIANCE.NS, TCS.NS, INFY.NS, and ^NSEI for Nifty 50) and
computes 30+ clean mathematical features across four pillars:
  1. Candlestick Geometry (body-to-range, upper/lower wick ratios, gap %, CLV)
  2. Trend & Moving Averages (20/50/200 EMA distances, slopes, alignment ratios)
  3. Momentum & Volatility (RSI-14, Stochastics, MACD hist, Bollinger width/squeeze, ATR-14, ADX-14)
  4. Volume Dynamics (20-day volume ratio, volume Z-score, OBV slope, PV momentum)

Note: This module is strictly a mathematical feature preparation tool for
probabilistic decision support and risk management.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    yf = None  # Handled gracefully with a clear error message or offline synthetic mode

logger = logging.getLogger(__name__)

# Default NSE universe
DEFAULT_NSE_TICKERS: List[str] = ["RELIANCE.NS", "TCS.NS", "INFY.NS"]
DEFAULT_INDEX_TICKER: str = "^NSEI"

# Canonical list of 32 mathematical features used by the ML model
FEATURE_COLUMNS: List[str] = [
    # 1. Candlestick Geometry (6 features)
    "body_to_range",
    "signed_body_to_range",
    "upper_wick_ratio",
    "lower_wick_ratio",
    "gap_pct",
    "close_location_value",
    # 2. Trend & Moving Averages (11 features)
    "dist_ema_20_pct",
    "dist_ema_50_pct",
    "dist_ema_200_pct",
    "slope_ema_20_5d",
    "slope_ema_50_5d",
    "slope_ema_200_5d",
    "ema_20_50_spread_pct",
    "ema_50_200_spread_pct",
    "return_3d_pct",
    "return_5d_pct",
    "return_20d_pct",
    # 3. Momentum & Volatility (11 features)
    "rsi_14",
    "rsi_14_slope_3d",
    "stoch_k_14",
    "stoch_d_3",
    "macd_hist_norm",
    "macd_hist_slope_3d",
    "bb_width_pct",
    "bb_pct_b",
    "bb_squeeze_ratio",
    "atr_pct",
    "volatility_20d_ann",
    # 4. Regime & Volume Dynamics (4 features)
    "adx_14",
    "di_spread_14",
    "vol_ratio_20",
    "vol_zscore_20",
    "obv_slope_10d",
    "pv_momentum_5d",
]


def _flatten_yfinance_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes DataFrame columns from older and newer versions of `yfinance`,
    which may return a MultiIndex `(Price, Ticker)` even for a single ticker.
    """
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = out.columns.get_level_values(0)
    # Standardize column capitalization
    col_map = {c: str(c).strip().title() for c in out.columns}
    out = out.rename(columns=col_map)
    return out


def generate_synthetic_ohlcv(
    ticker: str = "RELIANCE.NS",
    years: int = 8,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generates realistic geometric Brownian motion + regime-switching daily OHLCV
    data. Used only when `--offline-demo` is requested or when testing the
    pipeline in an air-gapped environment without internet access.
    """
    rng = np.random.default_rng(seed + sum(ord(c) for c in ticker))
    n_days = int(years * 252)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_days)

    # Simulate alternating bull, bear, and choppy regimes
    drift = np.zeros(n_days)
    vol = np.full(n_days, 0.014)
    for start in range(0, n_days, 120):
        end = min(start + 120, n_days)
        regime_draw = rng.choice(["bull", "choppy", "bear"], p=[0.50, 0.30, 0.20])
        if regime_draw == "bull":
            drift[start:end] = 0.0009
            vol[start:end] = 0.012
        elif regime_draw == "bear":
            drift[start:end] = -0.0007
            vol[start:end] = 0.018
        else:
            drift[start:end] = 0.0001
            vol[start:end] = 0.011

    # Ensure recent window is in a constructive uptrend for realistic inspection
    drift[-60:] = 0.0012
    vol[-60:] = 0.011

    rets = drift + vol * rng.standard_normal(n_days)
    base_price = 18000.0 if ticker.startswith("^") else 1250.0
    close = base_price * np.exp(np.cumsum(rets))

    gap = rng.normal(0.0, 0.003, size=n_days)
    open_p = np.roll(close, 1) * (1.0 + gap)
    open_p[0] = close[0] * (1.0 - 0.002)

    intraday_span = np.abs(rng.normal(0.012, 0.005, size=n_days)) * close
    high = np.maximum(open_p, close) + rng.uniform(0.25, 0.75, size=n_days) * intraday_span
    low = np.minimum(open_p, close) - rng.uniform(0.25, 0.75, size=n_days) * intraday_span
    low = np.maximum(low, 1.0)

    base_vol = 250_000 if ticker.startswith("^") else 6_500_000
    vol_mult = np.exp(rng.normal(0.0, 0.35, size=n_days)) * (1.0 + 18.0 * np.abs(rets))
    volume = (base_vol * vol_mult).astype(np.int64)

    df = pd.DataFrame(
        {
            "Open": open_p,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        },
        index=dates,
    )
    df.index.name = "Date"
    return df


def download_ohlcv(
    ticker: str = "RELIANCE.NS",
    years: int = 8,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    allow_synthetic_fallback: bool = False,
) -> pd.DataFrame:
    """
    Downloads 5 to 10 years of daily OHLCV history for `ticker` using `yfinance`.

    Parameters
    ----------
    ticker : str
        Yahoo Finance symbol (e.g. 'RELIANCE.NS', 'TCS.NS', 'INFY.NS', '^NSEI').
    years : int
        Number of years of daily data to fetch (clamped between 5 and 10 by default).
    start_date : str, optional
        Explicit ISO start date 'YYYY-MM-DD'. Overrides `years` if supplied.
    end_date : str, optional
        Explicit ISO end date 'YYYY-MM-DD'. Defaults to today.
    allow_synthetic_fallback : bool
        If True and network/yfinance is unavailable, returns synthetic OHLCV data
        for offline testing.

    Returns
    -------
    pd.DataFrame
        Clean DataFrame indexed by `Date` with columns `['Open', 'High', 'Low', 'Close', 'Volume']`.
    """
    years_clamped = int(np.clip(years, 5, 10))
    if end_date is None:
        end_ts = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)
        end_date = end_ts.strftime("%Y-%m-%d")
    if start_date is None:
        start_ts = pd.Timestamp(end_date) - pd.DateOffset(years=years_clamped)
        start_date = start_ts.strftime("%Y-%m-%d")

    if yf is not None:
        try:
            raw = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                interval="1d",
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            if raw is not None and not raw.empty:
                df = _flatten_yfinance_columns(raw)
                required = ["Open", "High", "Low", "Close", "Volume"]
                missing = [c for c in required if c not in df.columns]
                if not missing:
                    df = df[required].copy()
                    df = df.dropna(subset=["Open", "High", "Low", "Close"])
                    df["Volume"] = df["Volume"].fillna(0).astype(float)
                    df = df[df["Close"] > 0].sort_index()
                    df.index = pd.to_datetime(df.index).tz_localize(None)
                    df.index.name = "Date"
                    if len(df) >= 250:
                        return df
        except Exception as exc:
            logger.warning("yfinance download failed for %s: %s", ticker, exc)

    if allow_synthetic_fallback:
        logger.warning(
            "Using synthetic offline OHLCV data for %s (%d years).",
            ticker,
            years_clamped,
        )
        return generate_synthetic_ohlcv(ticker=ticker, years=years_clamped)

    raise RuntimeError(
        f"Failed to download OHLCV data for '{ticker}' via yfinance. "
        "Verify your internet connection, ticker symbol, or run with `--offline-demo`."
    )


def compute_wilder_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Computes Wilder's Average True Range (ATR)."""
    high = df["High"]
    low = df["Low"]
    prev_close = df["Close"].shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = true_range.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    return atr


def compute_wilder_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Computes Wilder's Relative Strength Index (RSI)."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def compute_adx_and_di(
    df: pd.DataFrame, period: int = 14
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Computes Wilder's Average Directional Index (ADX), +DI, and -DI.

    Returns
    -------
    Tuple[pd.Series, pd.Series, pd.Series]
        (adx_14, plus_di_14, minus_di_14)
    """
    high = df["High"]
    low = df["Low"]
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = df["Close"].shift(1)

    up_move = high - prev_high
    down_move = prev_low - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    plus_dm_s = pd.Series(plus_dm, index=df.index)
    minus_dm_s = pd.Series(minus_dm, index=df.index)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr_smooth = tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    plus_dm_smooth = plus_dm_s.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    minus_dm_smooth = minus_dm_s.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    safe_atr = atr_smooth.replace(0.0, np.nan)
    plus_di = 100.0 * (plus_dm_smooth / safe_atr)
    minus_di = 100.0 * (minus_dm_smooth / safe_atr)

    di_sum = (plus_di + minus_di).replace(0.0, np.nan)
    dx = 100.0 * ((plus_di - minus_di).abs() / di_sum)
    adx = dx.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    return adx.fillna(0.0), plus_di.fillna(0.0), minus_di.fillna(0.0)


def engineer_features(df: pd.DataFrame, drop_na: bool = True) -> pd.DataFrame:
    """
    Computes 34 mathematical technical & statistical features on daily OHLCV data
    using strictly causal (past-and-present only) window operations.

    Parameters
    ----------
    df : pd.DataFrame
        Input OHLCV DataFrame with columns `['Open', 'High', 'Low', 'Close', 'Volume']`.
    drop_na : bool
        Whether to drop warm-up rows containing NaN values (e.g., first 200 bars).

    Returns
    -------
    pd.DataFrame
        DataFrame containing original OHLCV columns, intermediate indicators
        (`EMA_20`, `EMA_50`, `EMA_200`, `ATR_14`, `PLUS_DI_14`, `MINUS_DI_14`),
        and all 34 `FEATURE_COLUMNS`.
    """
    feat = df.copy()
    eps = 1e-9

    open_p = feat["Open"]
    high = feat["High"]
    low = feat["Low"]
    close = feat["Close"]
    volume = feat["Volume"].astype(float)
    prev_close = close.shift(1)

    # -------------------------------------------------------------------------
    # 1. Candlestick Geometry (6 features)
    # -------------------------------------------------------------------------
    candle_range = (high - low).replace(0.0, np.nan)
    real_body = (close - open_p).abs()
    upper_wick = high - np.maximum(open_p, close)
    lower_wick = np.minimum(open_p, close) - low

    feat["body_to_range"] = (real_body / (candle_range + eps)).fillna(0.0).clip(0.0, 1.0)
    feat["signed_body_to_range"] = ((close - open_p) / (candle_range + eps)).fillna(0.0).clip(-1.0, 1.0)
    feat["upper_wick_ratio"] = (upper_wick / (candle_range + eps)).fillna(0.0).clip(0.0, 1.0)
    feat["lower_wick_ratio"] = (lower_wick / (candle_range + eps)).fillna(0.0).clip(0.0, 1.0)
    feat["gap_pct"] = ((open_p - prev_close) / (prev_close + eps)) * 100.0
    feat["close_location_value"] = (((close - low) - (high - close)) / (candle_range + eps)).fillna(0.0).clip(-1.0, 1.0)

    # -------------------------------------------------------------------------
    # 2. Trend & Moving Averages (11 features + core EMAs)
    # -------------------------------------------------------------------------
    ema_20 = close.ewm(span=20, adjust=False).mean()
    ema_50 = close.ewm(span=50, adjust=False).mean()
    ema_200 = close.ewm(span=200, adjust=False).mean()

    feat["EMA_20"] = ema_20
    feat["EMA_50"] = ema_50
    feat["EMA_200"] = ema_200

    feat["dist_ema_20_pct"] = ((close - ema_20) / (ema_20 + eps)) * 100.0
    feat["dist_ema_50_pct"] = ((close - ema_50) / (ema_50 + eps)) * 100.0
    feat["dist_ema_200_pct"] = ((close - ema_200) / (ema_200 + eps)) * 100.0

    feat["slope_ema_20_5d"] = ema_20.pct_change(5) * 100.0
    feat["slope_ema_50_5d"] = ema_50.pct_change(5) * 100.0
    feat["slope_ema_200_5d"] = ema_200.pct_change(5) * 100.0

    feat["ema_20_50_spread_pct"] = ((ema_20 - ema_50) / (ema_50 + eps)) * 100.0
    feat["ema_50_200_spread_pct"] = ((ema_50 - ema_200) / (ema_200 + eps)) * 100.0

    feat["return_3d_pct"] = close.pct_change(3) * 100.0
    feat["return_5d_pct"] = close.pct_change(5) * 100.0
    feat["return_20d_pct"] = close.pct_change(20) * 100.0

    # -------------------------------------------------------------------------
    # 3. Momentum & Volatility (11 features)
    # -------------------------------------------------------------------------
    rsi_14 = compute_wilder_rsi(close, period=14)
    feat["rsi_14"] = rsi_14
    feat["rsi_14_slope_3d"] = rsi_14.diff(3)

    low_14 = low.rolling(window=14, min_periods=14).min()
    high_14 = high.rolling(window=14, min_periods=14).max()
    stoch_k = 100.0 * ((close - low_14) / ((high_14 - low_14) + eps))
    stoch_d = stoch_k.rolling(window=3, min_periods=3).mean()
    feat["stoch_k_14"] = stoch_k.clip(0.0, 100.0)
    feat["stoch_d_3"] = stoch_d.clip(0.0, 100.0)

    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - macd_signal
    # Normalize MACD histogram by close price (%) so it is scale-invariant across stocks
    feat["macd_hist_norm"] = (macd_hist / (close + eps)) * 100.0
    feat["macd_hist_slope_3d"] = feat["macd_hist_norm"].diff(3)

    bb_mid = close.rolling(window=20, min_periods=20).mean()
    bb_std = close.rolling(window=20, min_periods=20).std(ddof=0)
    bb_upper = bb_mid + 2.0 * bb_std
    bb_lower = bb_mid - 2.0 * bb_std
    bb_width_pct = ((bb_upper - bb_lower) / (bb_mid + eps)) * 100.0
    bb_pct_b = (close - bb_lower) / ((bb_upper - bb_lower) + eps)
    bb_width_mean_50 = bb_width_pct.rolling(window=50, min_periods=20).mean()

    feat["bb_width_pct"] = bb_width_pct
    feat["bb_pct_b"] = bb_pct_b
    feat["bb_squeeze_ratio"] = bb_width_pct / (bb_width_mean_50 + eps)

    atr_14 = compute_wilder_atr(feat, period=14)
    feat["ATR_14"] = atr_14
    feat["atr_pct"] = (atr_14 / (close + eps)) * 100.0

    daily_ret = close.pct_change()
    feat["volatility_20d_ann"] = daily_ret.rolling(window=20, min_periods=20).std(ddof=0) * np.sqrt(252.0) * 100.0

    # -------------------------------------------------------------------------
    # 4. Regime & Volume Dynamics (6 features + +DI/-DI columns)
    # -------------------------------------------------------------------------
    adx_14, plus_di_14, minus_di_14 = compute_adx_and_di(feat, period=14)
    feat["PLUS_DI_14"] = plus_di_14
    feat["MINUS_DI_14"] = minus_di_14
    feat["adx_14"] = adx_14
    feat["di_spread_14"] = plus_di_14 - minus_di_14

    vol_ma_20 = volume.rolling(window=20, min_periods=10).mean()
    vol_std_20 = volume.rolling(window=20, min_periods=10).std(ddof=0)

    feat["vol_ratio_20"] = (volume / (vol_ma_20 + eps)).fillna(1.0)
    feat["vol_zscore_20"] = ((volume - vol_ma_20) / (vol_std_20 + eps)).fillna(0.0).clip(-5.0, 5.0)

    # On-Balance Volume (OBV) normalized 10-day slope
    direction = np.sign(close.diff()).fillna(0.0)
    obv = (direction * volume).cumsum()
    obv_diff_10 = obv.diff(10)
    feat["obv_slope_10d"] = (obv_diff_10 / (vol_ma_20 * 10.0 + eps)).fillna(0.0).clip(-3.0, 3.0)

    # Price-Volume Momentum (5-day return weighted by relative volume)
    feat["pv_momentum_5d"] = feat["return_5d_pct"] * feat["vol_ratio_20"]

    # Replace any inf values with NaN before optional dropna
    feat = feat.replace([np.inf, -np.inf], np.nan)

    if drop_na:
        # Warm-up period is ~50 bars (since 200 EMA uses ewm with adjust=False)
        feat = feat.iloc[55:].dropna(subset=FEATURE_COLUMNS).copy()

    return feat


if __name__ == "__main__":
    print("=== Phase 1 Self-Test: Data & Feature Engineering ===")
    sample_df = download_ohlcv("RELIANCE.NS", years=8, allow_synthetic_fallback=True)
    featured_df = engineer_features(sample_df)
    print(f"Rows after feature engineering: {len(featured_df)}")
    print(f"Total mathematical features engineered: {len(FEATURE_COLUMNS)}")
    print("Latest feature snapshot (last row):")
    print(featured_df[FEATURE_COLUMNS].tail(1).T.round(4))
