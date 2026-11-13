"""
Phase 4 — Chart Image Generator for Vision Check (chart_plotter.py)
===================================================================
Uses `mplfinance` (with an automatic high-DPI `matplotlib` candlestick fallback
if `mplfinance` is not installed in the current environment) to plot clean
candlestick charts with:
  - 20 EMA, 50 EMA, and 200 EMA overlays
  - Volume sub-panel with 20-day Volume Moving Average
  - Horizontal Entry, ATR Stop-Loss, and Profit Target reference levels

You can upload the saved PNG chart directly into Gemini for a second-opinion
visual inspection of horizontal support/resistance zones, chart patterns, and
trendline structures.
"""

from __future__ import annotations

import os
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

try:
    import mplfinance as mpf
except ImportError:
    mpf = None


def plot_candlestick_chart(
    featured_df: pd.DataFrame,
    ticker: str = "RELIANCE.NS",
    lookback_bars: int = 120,
    output_path: str = "charts/RELIANCE_NS_vision_check.png",
    entry_price: Optional[float] = None,
    stop_loss: Optional[float] = None,
    target_price: Optional[float] = None,
    regime_label: Optional[str] = None,
    ml_win_prob: Optional[float] = None,
) -> str:
    """
    Renders a candlestick chart with 20/50/200 EMAs and Volume over the most
    recent `lookback_bars` trading days and saves it to `output_path`.

    Parameters
    ----------
    featured_df : pd.DataFrame
        DataFrame produced by `engineer_features` containing `Open`, `High`,
        `Low`, `Close`, `Volume`, `EMA_20`, `EMA_50`, and `EMA_200`.
    ticker : str
        Stock symbol for chart title.
    lookback_bars : int
        Number of recent daily candles to plot (default 120, ~6 months).
    output_path : str
        Destination PNG file path.
    entry_price, stop_loss, target_price : float, optional
        If provided, draws dashed horizontal reference lines for the proposed
        Entry, Stop-Loss, and Target levels.
    regime_label : str, optional
        Market regime label shown in the subtitle.
    ml_win_prob : float, optional
        ML probability score (0..1) shown in the subtitle.

    Returns
    -------
    str
        Resolved absolute path to the saved PNG chart.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    sub = featured_df.tail(lookback_bars).copy()

    title_parts = [f"{ticker} — Daily Candlestick & EMA Structure ({len(sub)} Bars)"]
    if regime_label:
        title_parts.append(f"Regime: {regime_label}")
    if ml_win_prob is not None:
        title_parts.append(f"ML Win Prob: {ml_win_prob * 100:.1f}%")
    chart_title = " | ".join(title_parts)

    # Primary Path: mplfinance
    if mpf is not None:
        add_plots = [
            mpf.make_addplot(sub["EMA_20"], color="#2563EB", width=1.3, label="20 EMA"),
            mpf.make_addplot(sub["EMA_50"], color="#D97706", width=1.5, label="50 EMA"),
            mpf.make_addplot(sub["EMA_200"], color="#7C3AED", width=1.8, label="200 EMA"),
        ]

        hlines_vals = []
        hlines_colors = []
        if entry_price is not None:
            hlines_vals.append(float(entry_price))
            hlines_colors.append("#0284C7")
        if stop_loss is not None:
            hlines_vals.append(float(stop_loss))
            hlines_colors.append("#DC2626")
        if target_price is not None:
            hlines_vals.append(float(target_price))
            hlines_colors.append("#16A34A")

        mc = mpf.make_marketcolors(
            up="#16A34A",
            down="#DC2626",
            edge="inherit",
            wick="inherit",
            volume="in",
        )
        style = mpf.make_mpf_style(
            base_mpf_style="yahoo",
            marketcolors=mc,
            gridstyle="--",
            gridcolor="#E2E8F0",
            facecolor="#FFFFFF",
            figcolor="#F8FAFC",
        )

        plot_kwargs = dict(
            type="candle",
            style=style,
            addplot=add_plots,
            volume=True,
            title=chart_title,
            ylabel="Price",
            ylabel_lower="Volume",
            figsize=(14, 8),
            panel_ratios=(3, 1),
            tight_layout=True,
            savefig=dict(fname=output_path, dpi=180, bbox_inches="tight"),
        )
        if hlines_vals:
            plot_kwargs["hlines"] = dict(
                hlines=hlines_vals,
                colors=hlines_colors,
                linestyle="--",
                linewidths=1.2,
            )

        mpf.plot(sub, **plot_kwargs)
        return os.path.abspath(output_path)

    # Clean Matplotlib Fallback (used if mplfinance is not installed in a test container)
    fig, (ax_price, ax_vol) = plt.subplots(
        2,
        1,
        figsize=(14, 8),
         sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1.0], "hspace": 0.06},
    )
    fig.patch.set_facecolor("#F8FAFC")
    ax_price.set_facecolor("#FFFFFF")
    ax_vol.set_facecolor("#FFFFFF")

    x = np.arange(len(sub))
    opens = sub["Open"].to_numpy(dtype=float)
    highs = sub["High"].to_numpy(dtype=float)
    lows = sub["Low"].to_numpy(dtype=float)
    closes = sub["Close"].to_numpy(dtype=float)
    vols = sub["Volume"].to_numpy(dtype=float)

    width = 0.62
    for i in range(len(sub)):
        is_bull = closes[i] >= opens[i]
        color = "#16A34A" if is_bull else "#DC2626"
        ax_price.plot([i, i], [lows[i], highs[i]], color=color, linewidth=1.0, zorder=2)
        body_bottom = min(opens[i], closes[i])
        body_height = max(abs(closes[i] - opens[i]), (highs[i] - lows[i]) * 0.02, 0.01)
        rect = Rectangle(
            (i - width / 2.0, body_bottom),
            width,
            body_height,
            facecolor=color,
            edgecolor=color,
            linewidth=0.8,
            zorder=3,
        )
        ax_price.add_patch(rect)
        ax_vol.bar(i, vols[i], width=width, color=color, alpha=0.75)

    ax_price.plot(x, sub["EMA_20"].to_numpy(), color="#2563EB", linewidth=1.4, label="20 EMA")
    ax_price.plot(x, sub["EMA_50"].to_numpy(), color="#D97706", linewidth=1.6, label="50 EMA")
    ax_price.plot(x, sub["EMA_200"].to_numpy(), color="#7C3AED", linewidth=1.9, label="200 EMA")

    if entry_price is not None:
        ax_price.axhline(
            entry_price, color="#0284C7", linestyle="--", linewidth=1.2, label=f"Entry: {entry_price:.2f}"
        )
    if stop_loss is not None:
        ax_price.axhline(
            stop_loss, color="#DC2626", linestyle="--", linewidth=1.2, label=f"Stop-Loss (-1 ATR): {stop_loss:.2f}"
        )
    if target_price is not None:
        ax_price.axhline(
            target_price, color="#16A34A", linestyle="--", linewidth=1.2, label=f"Target (+2 ATR): {target_price:.2f}"
        )

    vol_ma20 = pd.Series(vols).rolling(20, min_periods=5).mean().to_numpy()
    ax_vol.plot(x, vol_ma20, color="#334155", linewidth=1.2, label="20d Avg Vol")

    ax_price.set_title(chart_title, fontsize=12, fontweight="bold", color="#0F172A", pad=12)
    ax_price.set_ylabel("Price", fontsize=10, fontweight="bold", color="#1E293B")
    ax_vol.set_ylabel("Volume", fontsize=10, fontweight="bold", color="#1E293B")

    ax_price.grid(True, linestyle="--", alpha=0.45, color="#CBD5E1")
    ax_vol.grid(True, linestyle="--", alpha=0.45, color="#CBD5E1")
    ax_price.legend(loc="upper left", frameon=True, facecolor="#FFFFFF", edgecolor="#E2E8F0", fontsize=9)
    ax_vol.legend(loc="upper left", frameon=True, facecolor="#FFFFFF", edgecolor="#E2E8F0", fontsize=8)

    tick_step = max(len(sub) // 10, 1)
    tick_positions = list(range(0, len(sub), tick_step))
    tick_labels = [sub.index[idx].strftime("%Y-%m-%d") for idx in tick_positions]
    ax_vol.set_xticks(tick_positions)
    ax_vol.set_xticklabels(tick_labels, rotation=30, ha="right", fontsize=8.5)

    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return os.path.abspath(output_path)


if __name__ == "__main__":
    from data_features import download_ohlcv, engineer_features

    print("=== Phase 4 Self-Test: Chart Image Generator ===")
    df_feat = engineer_features(download_ohlcv("RELIANCE.NS", years=8, allow_synthetic_fallback=True))
    last_close = float(df_feat["Close"].iloc[-1])
    last_atr = float(df_feat["ATR_14"].iloc[-1])
    saved = plot_candlestick_chart(
        df_feat,
        ticker="RELIANCE.NS",
        output_path="charts/RELIANCE_NS_vision_check.png",
        entry_price=last_close,
        stop_loss=last_close - 1.0 * last_atr,
        target_price=last_close + 2.0 * last_atr,
        regime_label="Strong Uptrend",
        ml_win_prob=0.64,
    )
    print(f"Chart saved to: {saved}")
