"""
Phase 6 — Assistant Briefing & Paper-Trading Logger (main.py)
=============================================================
Single-command CLI entrypoint (`python main.py --ticker RELIANCE.NS`) that
orchestrates all 6 phases of the Hybrid AI Trading Assistant:
  1. Downloads 5–10 years of daily OHLCV data for the stock and `^NSEI` and
     engineers 34 mathematical candlestick, trend, momentum, volatility & volume features.
  2. Evaluates the Market Regime Filter (Strong Uptrend / Strong Downtrend /
     Sideways-Choppy) on both the stock and the Nifty 50 index.
  3. Labels historical data with the 10-day Triple-Barrier target (+2.0 ATR vs
     -1.0 ATR), trains the time-split XGBoost/LightGBM classifier (80% oldest train,
     20% newest unseen test), prints honest out-of-sample metrics, and scores the latest bar.
  4. Renders a clean candlestick + 20/50/200 EMA + Volume chart PNG for visual
     second-opinion checks.
  5. Runs the Hard-Coded Risk & Position Gatekeeper (<60% ML win probability or
     <1:2 Risk-to-Reward or choppy regime -> automatic `WAIT (No Trade)`).
  6. Prints a structured decision-support briefing and appends the signal record
     to `paper_trades.csv` for forward paper-trading verification.

Disclaimer: This tool is strictly for probabilistic research, educational
analysis, and paper-trading decision support. No model has 100% accuracy.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import os
import sys
from typing import Optional

from chart_plotter import plot_candlestick_chart
from data_features import (
    DEFAULT_INDEX_TICKER,
    DEFAULT_NSE_TICKERS,
    FEATURE_COLUMNS,
    download_ohlcv,
    engineer_features,
)
from regime_filter import evaluate_market_regime
from risk_engine import TradeDecisionPlan, evaluate_risk_and_position
from train_model import (
    ModelEvaluationMetrics,
    format_metrics_report,
    train_and_evaluate_model,
)


PAPER_TRADES_COLUMNS = [
    "timestamp_logged",
    "as_of_date",
    "ticker",
    "index_ticker",
    "action_signal",
    "ml_win_probability_pct",
    "min_prob_threshold_pct",
    "entry_price",
    "stop_loss_price",
    "target_price",
    "atr_14",
    "risk_reward_ratio",
    "recommended_shares",
    "position_notional_value",
    "max_capital_at_risk",
    "stock_regime",
    "stock_adx_14",
    "index_regime",
    "index_adx_14",
    "oos_roc_auc",
    "oos_precision_at_60_pct",
    "oos_profit_factor_at_60",
    "chart_path",
    "gatekeeper_summary",
]


def append_to_paper_trades_csv(
    csv_path: str,
    plan: TradeDecisionPlan,
    metrics: ModelEvaluationMetrics,
    stock_adx: float,
    index_ticker: str,
    index_adx: float,
    chart_path: str,
) -> str:
    """Appends the latest signal evaluation to `paper_trades.csv`."""
    abs_csv = os.path.abspath(csv_path)
    parent = os.path.dirname(abs_csv)
    if parent:
        os.makedirs(parent, exist_ok=True)

    file_exists = os.path.isfile(abs_csv) and os.path.getsize(abs_csv) > 0

    if plan.gatekeeper_passed:
        summary_text = " | ".join(plan.acceptance_notes)
    else:
        summary_text = " | ".join(plan.rejection_reasons)

    row = {
        "timestamp_logged": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "as_of_date": plan.as_of_date,
        "ticker": plan.ticker,
        "index_ticker": index_ticker,
        "action_signal": plan.action_signal,
        "ml_win_probability_pct": f"{plan.ml_win_probability * 100:.2f}",
        "min_prob_threshold_pct": f"{plan.min_required_probability * 100:.2f}",
        "entry_price": f"{plan.entry_price:.2f}",
        "stop_loss_price": f"{plan.stop_loss_price:.2f}",
        "target_price": f"{plan.target_price:.2f}",
        "atr_14": f"{plan.atr_14:.2f}",
        "risk_reward_ratio": f"{plan.risk_reward_ratio:.2f}",
        "recommended_shares": plan.recommended_shares,
        "position_notional_value": f"{plan.position_notional_value:.2f}",
        "max_capital_at_risk": f"{plan.max_capital_at_risk:.2f}",
        "stock_regime": plan.stock_regime,
        "stock_adx_14": f"{stock_adx:.2f}",
        "index_regime": plan.index_regime,
        "index_adx_14": f"{index_adx:.2f}",
        "oos_roc_auc": f"{metrics.roc_auc:.4f}",
        "oos_precision_at_60_pct": f"{metrics.precision_at_60 * 100:.2f}",
        "oos_profit_factor_at_60": f"{metrics.simulated_profit_factor_at_60:.2f}",
        "chart_path": chart_path,
        "gatekeeper_summary": summary_text,
    }

    with open(abs_csv, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PAPER_TRADES_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    return abs_csv


def print_assistant_briefing(
    ticker: str,
    index_ticker: str,
    featured_df,
    regime_gate,
    metrics: ModelEvaluationMetrics,
    plan: TradeDecisionPlan,
    chart_path: str,
    model_path: str,
    csv_path: str,
) -> None:
    """Prints a structured, human-readable trading decision-support briefing."""
    last = featured_df.iloc[-1]
    sep = "=" * 78

    print("\n" + sep)
    print(f" HYBRID AI TRADING ASSISTANT — DECISION SUPPORT BRIEFING ({ticker})")
    print(sep)
    print(f" As-of Market Close : {plan.as_of_date}")
    print(f" Latest Close Price : {plan.entry_price:.2f}  |  ATR(14): {plan.atr_14:.2f} ({float(last['atr_pct']):.2f}%)")
    print(
        f" Key Indicators     : RSI(14)={float(last['rsi_14']):.1f} | "
        f"MACD Hist={float(last['macd_hist_norm']):+.3f}% | "
        f"BB Squeeze={float(last['bb_squeeze_ratio']):.2f}x | "
        f"Vol Ratio={float(last['vol_ratio_20']):.2f}x"
    )
    print(
        f" Moving Averages    : 20-EMA={float(last['EMA_20']):.2f} | "
        f"50-EMA={float(last['EMA_50']):.2f} | "
        f"200-EMA={float(last['EMA_200']):.2f}"
    )

    print("\n[PHASE 2] MARKET & STOCK REGIME FILTER")
    print("-" * 78)
    print(f" Stock Regime ({ticker:<11}) : {regime_gate.stock_assessment.summary}")
    print(f" Index Regime ({index_ticker:<11}) : {regime_gate.index_assessment.summary}")
    print(f" Regime Gate Status         : {regime_gate.gate_status}")
    print(f" Regime Gate Explanation    : {regime_gate.gate_reason}")

    print("\n[PHASE 3] TRIPLE-BARRIER ML PROBABILITY ENGINE (80/20 TIME SPLIT)")
    print("-" * 78)
    for line in format_metrics_report(metrics).splitlines():
        print(" " + line)
    top_feat_str = ", ".join(f"{name} ({weight * 100:.1f}%)" for name, weight in metrics.top_features[:5])
    print(f" Top Predictive Features   : {top_feat_str}")

    print("\n[PHASE 5] HARD-CODED RISK & POSITION GATEKEEPER")
    print("-" * 78)
    print(f" FINAL VERDICT             : >>> {plan.action_signal} <<<")
    print(f" ML Win Probability        : {plan.ml_win_probability * 100:.2f}% (Required: >= {plan.min_required_probability * 100:.0f}%)")
    print(f" Reference Entry Price     : {plan.entry_price:.2f}")
    print(f" ATR Stop-Loss (-1.0 ATR)  : {plan.stop_loss_price:.2f} (Risk/Share: {plan.risk_per_share:.2f})")
    print(f" Profit Target (+2.0 ATR)  : {plan.target_price:.2f} (Reward/Share: {plan.reward_per_share:.2f})")
    print(f" Reward-to-Risk Ratio      : 1 : {plan.risk_reward_ratio:.2f} (Required: >= 1 : {plan.min_required_rr:.1f})")
    print(
        f" Position Sizing           : {plan.recommended_shares} shares "
        f"(Notional: {plan.position_notional_value:,.2f} | Max Risk @ {plan.max_risk_pct:.1f}%: {plan.max_capital_at_risk:,.2f})"
    )

    if plan.gatekeeper_passed:
        print("\n Gatekeeper Checks Passed:")
        for note in plan.acceptance_notes:
            print(f"   [+] {note}")
    else:
        print("\n Gatekeeper Rejection Reasons (Why Signal is WAIT):")
        for reason in plan.rejection_reasons:
            print(f"   [!] {reason}")

    print("\n[PHASE 4 & 6] ARTIFACTS & VISION CHECK")
    print("-" * 78)
    print(f" Vision-Check Chart PNG    : {chart_path}")
    print(f" Saved Trained Model       : {os.path.abspath(model_path)}")
    print(f" Paper-Trading Log CSV     : {csv_path}")
    print(
        " Tip: Upload the Vision-Check Chart PNG to Gemini to verify horizontal "
        "support/resistance, trendlines, and price structure before placing any paper trade."
    )
    print(sep + "\n")


def run_pipeline_for_ticker(
    ticker: str,
    index_ticker: str = DEFAULT_INDEX_TICKER,
    years: int = 8,
    account_capital: float = 100_000.0,
    max_risk_pct: float = 1.0,
    min_win_prob: float = 0.60,
    min_rr_ratio: float = 2.0,
    adx_threshold: float = 20.0,
    lookback_bars: int = 120,
    output_dir: str = ".",
    offline_demo: bool = False,
) -> TradeDecisionPlan:
    """Executes all 6 phases for a single ticker symbol."""
    # Phase 1: Download & engineer 34 mathematical features for stock and index
    stock_raw = download_ohlcv(ticker, years=years, allow_synthetic_fallback=offline_demo)
    index_raw = download_ohlcv(index_ticker, years=years, allow_synthetic_fallback=offline_demo)

    stock_feat = engineer_features(stock_raw)
    index_feat = engineer_features(index_raw)

    # Phase 2: Market Regime Filter
    regime_gate = evaluate_market_regime(
        stock_df=stock_feat,
        index_df=index_feat,
        stock_symbol=ticker,
        index_symbol=index_ticker,
        adx_threshold=adx_threshold,
    )

    # Phase 3: Triple-Barrier Labeling & Chronological 80/20 ML Training
    safe_symbol = ticker.replace("^", "").replace(".", "_")
    model_path = os.path.join(output_dir, "models", f"{safe_symbol}_xgb_model.pkl")
    _, metrics, _ = train_and_evaluate_model(
        featured_df=stock_feat,
        ticker=ticker,
        train_ratio=0.80,
        profit_atr_mult=2.0,
        stop_atr_mult=1.0,
        horizon_days=10,
        model_save_path=model_path,
    )

    # Phase 5: Hard-Coded Risk & Position Gatekeeper
    plan = evaluate_risk_and_position(
        featured_df=stock_feat,
        regime_gate=regime_gate,
        ml_win_probability=metrics.latest_win_probability,
        ticker=ticker,
        account_capital=account_capital,
        max_risk_pct=max_risk_pct,
        stop_atr_mult=1.0,
        target_atr_mult=2.0,
        min_win_prob=min_win_prob,
        min_rr_ratio=min_rr_ratio,
    )

    # Phase 4: Chart Image Generator for Vision Check
    chart_file = os.path.join(output_dir, "charts", f"{safe_symbol}_vision_check.png")
    chart_path = plot_candlestick_chart(
        featured_df=stock_feat,
        ticker=ticker,
        lookback_bars=lookback_bars,
        output_path=chart_file,
        entry_price=plan.entry_price,
        stop_loss=plan.stop_loss_price,
        target_price=plan.target_price,
        regime_label=plan.stock_regime,
        ml_win_prob=plan.ml_win_probability,
    )

    # Phase 6: Paper-Trading CSV Logger & Console Briefing
    csv_file = os.path.join(output_dir, "paper_trades.csv")
    csv_path = append_to_paper_trades_csv(
        csv_path=csv_file,
        plan=plan,
        metrics=metrics,
        stock_adx=regime_gate.stock_assessment.adx_14,
        index_ticker=index_ticker,
        index_adx=regime_gate.index_assessment.adx_14,
        chart_path=chart_path,
    )

    print_assistant_briefing(
        ticker=ticker,
        index_ticker=index_ticker,
        featured_df=stock_feat,
        regime_gate=regime_gate,
        metrics=metrics,
        plan=plan,
        chart_path=chart_path,
        model_path=model_path,
        csv_path=csv_path,
    )
    return plan


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hybrid AI Trading Assistant — Probabilistic ML & Risk Gatekeeper CLI"
    )
    parser.add_argument(
        "--ticker",
        type=str,
        default="RELIANCE.NS",
        help="Yahoo Finance ticker symbol (e.g., RELIANCE.NS, TCS.NS, INFY.NS) or 'ALL' for default NSE universe.",
    )
    parser.add_argument(
        "--index",
        type=str,
        default=DEFAULT_INDEX_TICKER,
        help="Broader market index ticker for regime filtering (default: ^NSEI).",
    )
    parser.add_argument(
        "--years",
        type=int,
        default=8,
        help="Years of daily historical OHLCV data to download (5 to 10, default: 8).",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=100_000.0,
        help="Account capital for position sizing (default: 100000.0).",
    )
    parser.add_argument(
        "--risk-pct",
        type=float,
        default=1.0,
        help="Max percentage of capital risked per trade (default: 1.0%%).",
    )
    parser.add_argument(
        "--min-prob",
        type=float,
        default=0.60,
        help="Minimum ML win probability required to pass gatekeeper (default: 0.60).",
    )
    parser.add_argument(
        "--min-rr",
        type=float,
        default=2.0,
        help="Minimum Reward-to-Risk ratio required (default: 2.0 for 1:2 R:R).",
    )
    parser.add_argument(
        "--adx-threshold",
        type=float,
        default=20.0,
        help="Minimum ADX(14) threshold to classify a Strong Trend vs Sideways/Choppy (default: 20.0).",
    )
    parser.add_argument(
        "--lookback-bars",
        type=int,
        default=120,
        help="Number of recent daily candles to render on the Vision-Check chart (default: 120).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Directory for saving models/, charts/, and paper_trades.csv (default: current directory).",
    )
    parser.add_argument(
        "--offline-demo",
        action="store_true",
        help="Use synthetic regime-switching OHLCV data if running offline without internet.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list] = None) -> int:
    args = parse_args(argv)
    tickers = DEFAULT_NSE_TICKERS if args.ticker.upper() == "ALL" else [args.ticker]

    for sym in tickers:
        run_pipeline_for_ticker(
            ticker=sym,
            index_ticker=args.index,
            years=args.years,
            account_capital=args.capital,
            max_risk_pct=args.risk_pct,
            min_win_prob=args.min_prob,
            min_rr_ratio=args.min_rr,
            adx_threshold=args.adx_threshold,
            lookback_bars=args.lookback_bars,
            output_dir=args.output_dir,
            offline_demo=args.offline_demo,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
