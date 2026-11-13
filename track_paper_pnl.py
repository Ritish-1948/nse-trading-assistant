"""
Paper-Trade Accuracy & Profit Tracker (track_paper_pnl.py)
==========================================================
Reads all logged `BUY` signals from `paper_trades.csv`, fetches subsequent
daily OHLCV bars from `yfinance` (up to the 10-trading-day Triple-Barrier
horizon), and automatically checks whether each trade:
  - Hit its +2.0 ATR Profit Target (WIN, +2.0R)
  - Hit its -1.0 ATR Stop-Loss (LOSS, -1.0R)
  - Expired after 10 trading days (TIME EXIT, mark-to-market P&L)
  - Is still OPEN (in progress)

Outputs your real-world forward Win Rate %, Total Net P&L (₹), and Profit Factor
so you can verify profitability before risking any real capital.

Usage:
------
  python track_paper_pnl.py
  python track_paper_pnl.py --csv paper_trades.csv --horizon 10
"""

from __future__ import annotations

import argparse
import csv
import os
from typing import Any, Dict, List

import pandas as pd

try:
    import yfinance as yf
except ImportError:
    yf = None


def evaluate_paper_trade_outcome(
    row: Dict[str, str],
    horizon_days: int = 10,
) -> Dict[str, Any]:
    """Evaluates a single BUY row from paper_trades.csv against subsequent market bars."""
    ticker = row["ticker"]
    entry_date = pd.Timestamp(row["as_of_date"])
    entry_price = float(row["entry_price"])
    stop_loss = float(row["stop_loss_price"])
    target_price = float(row["target_price"])
    shares = int(float(row["recommended_shares"]))
    risk_per_share = max(entry_price - stop_loss, 1e-6)

    result: Dict[str, Any] = {
        "as_of_date": row["as_of_date"],
        "ticker": ticker,
        "shares": shares,
        "entry": entry_price,
        "stop": stop_loss,
        "target": target_price,
        "status": "OPEN (Awaiting Next Session)",
        "exit_date": "-",
        "exit_price": entry_price,
        "pnl_inr": 0.0,
        "r_multiple": 0.0,
    }

    if yf is None:
        result["status"] = "ERROR (yfinance not installed)"
        return result

    start_fetch = (entry_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    end_fetch = (pd.Timestamp.today().normalize() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    if start_fetch >= end_fetch:
        return result

    try:
        df = yf.download(
            ticker,
            start=start_fetch,
            end=end_fetch,
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
        )
    except Exception:
        return result

    if df is None or df.empty:
        return result

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.sort_index().head(horizon_days)

    for dt, bar in df.iterrows():
        low_p = float(bar["Low"])
        high_p = float(bar["High"])
        close_p = float(bar["Close"])
        dt_str = dt.strftime("%Y-%m-%d")

        # Conservative tie-break: check stop-loss first
        if low_p <= stop_loss:
            pnl = (stop_loss - entry_price) * shares
            result.update(
                status="STOP-LOSS HIT",
                exit_date=dt_str,
                exit_price=stop_loss,
                pnl_inr=round(pnl, 2),
                r_multiple=-1.0,
            )
            return result

        if high_p >= target_price:
            pnl = (target_price - entry_price) * shares
            result.update(
                status="TARGET HIT (WIN)",
                exit_date=dt_str,
                exit_price=target_price,
                pnl_inr=round(pnl, 2),
                r_multiple=2.0,
            )
            return result

    # Neither barrier hit yet; check if 10 trading days have elapsed
    last_dt = df.index[-1].strftime("%Y-%m-%d")
    last_close = float(df["Close"].iloc[-1])
    mtm_pnl = (last_close - entry_price) * shares
    mtm_r = (last_close - entry_price) / risk_per_share

    if len(df) >= horizon_days:
        status_label = "TIME EXIT (10d)"
    else:
        status_label = f"OPEN (Day {len(df)}/{horizon_days})"

    result.update(
        status=status_label,
        exit_date=last_dt,
        exit_price=round(last_close, 2),
        pnl_inr=round(mtm_pnl, 2),
        r_multiple=round(mtm_r, 2),
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Track Forward Paper-Trading Accuracy & P&L")
    parser.add_argument("--csv", type=str, default="paper_trades.csv", help="Path to paper_trades.csv")
    parser.add_argument("--horizon", type=int, default=10, help="Triple-barrier holding horizon in trading days")
    args = parser.parse_args()

    if not os.path.isfile(args.csv):
        print(f"[!] Could not find '{args.csv}'. Run main.py first to log signals.")
        return

    with open(args.csv, mode="r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    buy_rows = [r for r in rows if str(r.get("action_signal", "")).startswith("BUY")]
    if not buy_rows:
        print("No BUY trades logged in paper_trades.csv yet (all entries are WAIT).")
        return

    outcomes: List[Dict[str, Any]] = [evaluate_paper_trade_outcome(r, horizon_days=args.horizon) for r in buy_rows]

    print("\n" + "=" * 92)
    print(" FORWARD PAPER-TRADE ACCURACY & PROFIT TRACKER")
    print("=" * 92)
    print(f"{'Date':<12} {'Ticker':<13} {'Shares':>6} {'Entry':>9} {'Stop':>9} {'Target':>9} {'Status':<26} {'P&L (₹)':>10} {'R':>6}")
    print("-" * 92)

    total_pnl = 0.0
    gross_profit = 0.0
    gross_loss = 0.0
    closed_wins = 0
    closed_total = 0

    for o in outcomes:
        total_pnl += o["pnl_inr"]
        if o["pnl_inr"] > 0:
            gross_profit += o["pnl_inr"]
        elif o["pnl_inr"] < 0:
            gross_loss += abs(o["pnl_inr"])

        if not o["status"].startswith("OPEN"):
            closed_total += 1
            if o["pnl_inr"] > 0:
                closed_wins += 1

        print(
            f"{o['as_of_date']:<12} {o['ticker']:<13} {o['shares']:>6} "
            f"{o['entry']:>9.2f} {o['stop']:>9.2f} {o['target']:>9.2f} "
            f"{o['status']:<26} {o['pnl_inr']:>+10.2f} {o['r_multiple']:>+6.2f}"
        )

    win_rate = (closed_wins / closed_total * 100.0) if closed_total > 0 else 0.0
    pf = (gross_profit / gross_loss) if gross_loss > 0 else (gross_profit if gross_profit > 0 else 0.0)

    print("-" * 92)
    print(f" Total Logged BUY Trades : {len(outcomes)} (Closed: {closed_total}, Open: {len(outcomes) - closed_total})")
    print(f" Realized Win Rate       : {win_rate:.2f}%")
    print(f" Net Paper P&L           : ₹{total_pnl:+,.2f}  |  Profit Factor: {pf:.2f}")
    print("=" * 92 + "\n")


if __name__ == "__main__":
    main()