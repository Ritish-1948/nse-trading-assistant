"""
24/7 Live Autonomous Watchlist Scanner & Agent Feed Daemon (live_agent_daemon.py)
=================================================================================
Eliminates manual CSV uploads completely. This daemon:
  1. Pulls live daily/intraday OHLCV bars directly via `yfinance` across an
     8-stock NSE Nifty 50 watchlist (`RELIANCE.NS`, `TCS.NS`, `INFY.NS`,
     `HDFCBANK.NS`, `ICICIBANK.NS`, `SBIN.NS`, `BHARTIARTL.NS`, `LT.NS`) and
     the broader `^NSEI` index on a continuous 24/7 schedule.
  2. Runs all 6 phases (34 mathematical features, ADX/EMA regime filter,
     10-day Triple-Barrier ML probability scoring, and the strict 60% / 1:2
     Risk Gatekeeper) for every stock in the watchlist.
  3. Automatically filters out all `WAIT (No Trade)` symbols and selects the
     #1 Best Qualifying Trade (`best_selected_trade`) ranked by ML Win
     Probability and Out-of-Sample Expectancy.
  4. Writes a live machine-readable JSON feed (`live_agent_feed.json`) AND
     serves a lightweight local HTTP REST API (`http://127.0.0.1:8765/api/v1/best-trade`
     and `http://127.0.0.1:8765/api/v1/signals`) plus optional Webhook push so
     any external AI agent, Telegram bot, or broker execution bridge can read
     the selected trade details 24/7 without touching a CSV file.

Usage:
------
  # Run a single scan and start the 24/7 HTTP + polling daemon (every 300s):
  python live_agent_daemon.py --interval 300 --port 8765

  # Run one immediate scan and exit (writes live_agent_feed.json):
  python live_agent_daemon.py --once
"""

from __future__ import annotations

import argparse
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
import threading
import time
from typing import Any, Dict, List, Optional
import urllib.request

from data_features import download_ohlcv, engineer_features
from regime_filter import evaluate_market_regime
from risk_engine import evaluate_risk_and_position
from train_model import train_and_evaluate_model

DEFAULT_WATCHLIST: List[str] = [
    "RELIANCE.NS",
    "TCS.NS",
    "INFY.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "SBIN.NS",
    "BHARTIARTL.NS",
    "LT.NS",
]

LATEST_AGENT_FEED: Dict[str, Any] = {}


def scan_watchlist_and_select_best_trade(
    watchlist: List[str] = DEFAULT_WATCHLIST,
    index_ticker: str = "^NSEI",
    years: int = 5,
    account_capital: float = 100_000.0,
    max_risk_pct: float = 1.0,
    min_win_prob: float = 0.60,
    min_rr_ratio: float = 2.0,
    output_json_path: str = "live_agent_feed.json",
    webhook_url: Optional[str] = None,
    offline_demo: bool = False,
) -> Dict[str, Any]:
    """
    Pulls live OHLCV data for every symbol in `watchlist` (no CSV files required),
    evaluates all 6 phases, and automatically selects the #1 qualifying trade.
    """
    index_raw = download_ohlcv(index_ticker, years=years, allow_synthetic_fallback=offline_demo)
    index_feat = engineer_features(index_raw)

    evaluated_signals: List[Dict[str, Any]] = []

    for ticker in watchlist:
        stock_raw = download_ohlcv(ticker, years=years, allow_synthetic_fallback=offline_demo)
        stock_feat = engineer_features(stock_raw)

        regime_gate = evaluate_market_regime(
            stock_df=stock_feat,
            index_df=index_feat,
            stock_symbol=ticker,
            index_symbol=index_ticker,
            adx_threshold=20.0,
        )

        _, metrics, _ = train_and_evaluate_model(
            featured_df=stock_feat,
            ticker=ticker,
            train_ratio=0.80,
            profit_atr_mult=2.0,
            stop_atr_mult=1.0,
            horizon_days=10,
            model_save_path=None,
        )

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

        last_row = stock_feat.iloc[-1]
        signal_entry = {
            "ticker": ticker,
            "as_of_date": plan.as_of_date,
            "action_signal": plan.action_signal,
            "gatekeeper_passed": plan.gatekeeper_passed,
            "ml_win_probability": round(plan.ml_win_probability, 4),
            "ml_win_probability_pct": round(plan.ml_win_probability * 100.0, 2),
            "entry_price": plan.entry_price,
            "stop_loss_price": plan.stop_loss_price,
            "target_price": plan.target_price,
            "atr_14": plan.atr_14,
            "risk_per_share": plan.risk_per_share,
            "reward_per_share": plan.reward_per_share,
            "risk_reward_ratio": plan.risk_reward_ratio,
            "recommended_shares": plan.recommended_shares,
            "position_notional_value": plan.position_notional_value,
            "max_capital_at_risk": plan.max_capital_at_risk,
            "stock_regime": plan.stock_regime,
            "stock_adx_14": round(regime_gate.stock_assessment.adx_14, 2),
            "index_regime": plan.index_regime,
            "index_adx_14": round(regime_gate.index_assessment.adx_14, 2),
            "rsi_14": round(float(last_row["rsi_14"]), 2),
            "ema_20": round(float(last_row["EMA_20"]), 2),
            "ema_50": round(float(last_row["EMA_50"]), 2),
            "ema_200": round(float(last_row["EMA_200"]), 2),
            "oos_roc_auc": round(metrics.roc_auc, 4),
            "oos_precision_at_60_pct": round(metrics.precision_at_60 * 100.0, 2),
            "oos_profit_factor_at_60": round(metrics.simulated_profit_factor_at_60, 2),
            "oos_expectancy_r_at_60": round(metrics.expectancy_r_at_60, 2),
            "gatekeeper_reasons": plan.acceptance_notes if plan.gatekeeper_passed else plan.rejection_reasons,
        }
        evaluated_signals.append(signal_entry)

    # Sort all evaluated stocks by: (1) gatekeeper_passed descending, (2) ml_win_probability descending
    evaluated_signals.sort(
        key=lambda x: (1 if x["gatekeeper_passed"] else 0, x["ml_win_probability"], x["oos_profit_factor_at_60"]),
        reverse=True,
    )

    qualifying_buys = [s for s in evaluated_signals if s["gatekeeper_passed"]]
    best_trade = qualifying_buys[0] if qualifying_buys else None

    feed_payload = {
        "agent_feed_version": "1.0",
        "generated_at_ist": datetime.now().strftime("%Y-%m-%d %H:%M:%S IST"),
        "source_mode": "DIRECT_YFINANCE_LIVE_POLLING (Zero CSV)",
        "index_benchmark": {
            "symbol": index_ticker,
            "close": round(float(index_feat["Close"].iloc[-1]), 2),
            "regime": evaluated_signals[0]["index_regime"] if evaluated_signals else "Unknown",
            "adx_14": evaluated_signals[0]["index_adx_14"] if evaluated_signals else 0.0,
        },
        "selection_status": (
            f"ACTIONABLE_BUY_FOUND ({len(qualifying_buys)} of {len(evaluated_signals)} stocks passed all gates)"
            if best_trade
            else f"WAIT_NO_TRADE (0 of {len(evaluated_signals)} stocks passed strict 60% & Regime gates)"
        ),
        "best_selected_trade": best_trade,
        "qualifying_buy_candidates": qualifying_buys,
        "all_watchlist_signals": evaluated_signals,
    }

    global LATEST_AGENT_FEED
    LATEST_AGENT_FEED = feed_payload

    if output_json_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_json_path)), exist_ok=True)
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(feed_payload, f, indent=2)

    if webhook_url:
        try:
            req = urllib.request.Request(
                webhook_url,
                data=json.dumps(feed_payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=10)
        except Exception as exc:
            print(f"[WARN] Webhook delivery failed: {exc}")

    return feed_payload


class AgentFeedHTTPHandler(BaseHTTPRequestHandler):
    """Lightweight JSON REST server so any AI agent can read live trade details 24/7."""

    def do_GET(self) -> None:
        if self.path in ("/", "/api/v1/signals"):
            body = json.dumps(LATEST_AGENT_FEED, indent=2).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/v1/best-trade":
            payload = {
                "generated_at_ist": LATEST_AGENT_FEED.get("generated_at_ist"),
                "selection_status": LATEST_AGENT_FEED.get("selection_status"),
                "best_selected_trade": LATEST_AGENT_FEED.get("best_selected_trade"),
            }
            body = json.dumps(payload, indent=2).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format: str, *args: Any) -> None:
        return  # Keep console clean


def main() -> None:
    parser = argparse.ArgumentParser(description="24/7 Live Watchlist Scanner & Agent JSON Feed Daemon")
    parser.add_argument("--interval", type=int, default=300, help="Scan interval in seconds (default: 300s = 5m)")
    parser.add_argument("--port", type=int, default=8765, help="Local REST API port for agent queries (default: 8765)")
    parser.add_argument("--output-json", type=str, default="live_agent_feed.json", help="Path to write live JSON feed")
    parser.add_argument("--webhook-url", type=str, default=None, help="Optional Webhook URL to POST trade updates")
    parser.add_argument("--once", action="store_true", help="Run one full watchlist scan, write JSON, and exit")
    parser.add_argument("--offline-demo", action="store_true", help="Use offline anchored data if no internet")
    args = parser.parse_args()

    print(f"[*] Running initial 8-stock NSE watchlist scan (Zero CSV mode)...")
    feed = scan_watchlist_and_select_best_trade(
        output_json_path=args.output_json,
        webhook_url=args.webhook_url,
        offline_demo=args.offline_demo,
    )
    print(f"[*] Selection Status : {feed['selection_status']}")
    if feed["best_selected_trade"]:
        bt = feed["best_selected_trade"]
        print(
            f"[+] AUTO-SELECTED #1 TRADE: {bt['ticker']} | Signal: {bt['action_signal']} | "
            f"ML Win Prob: {bt['ml_win_probability_pct']:.2f}% | Entry: ₹{bt['entry_price']:.2f} | "
            f"Stop: ₹{bt['stop_loss_price']:.2f} | Target: ₹{bt['target_price']:.2f} | Shares: {bt['recommended_shares']}"
        )
    print(f"[*] Wrote machine-readable agent feed to: {os.path.abspath(args.output_json)}")

    if args.once:
        return

    server = HTTPServer(("127.0.0.1", args.port), AgentFeedHTTPHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    print(f"[*] 24/7 Agent HTTP JSON Endpoint active at:")
    print(f"    -> Full Watchlist Feed : http://127.0.0.1:{args.port}/api/v1/signals")
    print(f"    -> Best Selected Trade : http://127.0.0.1:{args.port}/api/v1/best-trade")

    try:
        while True:
            time.sleep(max(args.interval, 15))
            scan_watchlist_and_select_best_trade(
                output_json_path=args.output_json,
                webhook_url=args.webhook_url,
                offline_demo=args.offline_demo,
            )
    except KeyboardInterrupt:
        print("\n[*] Stopping 24/7 Live Agent Daemon.")


if __name__ == "__main__":
    main()
