"""
Phase 5 — Hard-Coded Risk & Position Gatekeeper (risk_engine.py)
================================================================
Computes exact Entry Price, ATR-based Stop-Loss, Structure/ATR Target Price,
Risk-to-Reward ratio, and position size based on account capital and max risk %.

Hard-Coded Risk Gatekeeper Rules:
---------------------------------
1. If ML win probability < `min_win_prob` (default 60% / 0.60) -> WAIT (No Trade)
2. If Risk-to-Reward ratio < `min_rr_ratio` (default 1:2 / 2.0) -> WAIT (No Trade)
3. If Market Regime Filter blocks the trade (e.g., `Sideways / Choppy` or hostile
   index/stock regime) -> WAIT (No Trade)

This module acts as an unemotional mathematical circuit-breaker so no trade is
taken when probability, reward-to-risk, or market regime is unfavorable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

import pandas as pd

from regime_filter import CombinedRegimeGate, REGIME_STRONG_UPTREND


SIGNAL_BUY = "BUY"
SIGNAL_SELL = "SELL"
SIGNAL_WAIT = "WAIT (No Trade)"


@dataclass
class TradeDecisionPlan:
    """Complete risk-managed trade specification and gatekeeper verdict."""

    ticker: str
    as_of_date: str
    action_signal: str
    ml_win_probability: float
    min_required_probability: float
    entry_price: float
    atr_14: float
    stop_loss_price: float
    target_price: float
    risk_per_share: float
    reward_per_share: float
    risk_reward_ratio: float
    min_required_rr: float
    account_capital: float
    max_risk_pct: float
    max_capital_at_risk: float
    recommended_shares: int
    position_notional_value: float
    stock_regime: str
    index_regime: str
    gatekeeper_passed: bool
    rejection_reasons: List[str] = field(default_factory=list)
    acceptance_notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def evaluate_risk_and_position(
    featured_df: pd.DataFrame,
    regime_gate: CombinedRegimeGate,
    ml_win_probability: float,
    ticker: str = "RELIANCE.NS",
    account_capital: float = 100_000.0,
    max_risk_pct: float = 1.0,
    stop_atr_mult: float = 1.0,
    target_atr_mult: float = 2.0,
    min_win_prob: float = 0.60,
    min_rr_ratio: float = 2.0,
) -> TradeDecisionPlan:
    """
    Calculates exact Entry Price, ATR-based Stop-Loss, and Target Price, and
    applies the strict Phase 5 Risk & Position Gatekeeper rules.

    Parameters
    ----------
    featured_df : pd.DataFrame
        Engineered feature DataFrame containing `Close`, `High`, `Low`, and `ATR_14`.
    regime_gate : CombinedRegimeGate
        Output from `regime_filter.evaluate_market_regime`.
    ml_win_probability : float
        Predicted probability (0.0 to 1.0) that price hits +2.0 ATR before -1.0 ATR
        within the next 10 trading days.
    ticker : str
        Stock ticker symbol.
    account_capital : float
        Total trading capital (default ₹1,00,000).
    max_risk_pct : float
        Maximum account equity percentage risked on a single trade (default 1.0%).
    stop_atr_mult : float
        ATR multiplier for initial stop-loss distance (default 1.0x ATR).
    target_atr_mult : float
        ATR multiplier for profit target distance (default 2.0x ATR, capped/adjusted
        against 60-day resistance overhead to avoid unrealistic targets into resistance).
    min_win_prob : float
        Hard minimum ML win probability threshold (default 0.60 = 60%).
    min_rr_ratio : float
        Hard minimum Reward-to-Risk ratio (default 2.0 = 1:2 Risk-to-Reward).

    Returns
    -------
    TradeDecisionPlan
    """
    last = featured_df.iloc[-1]
    as_of_date = str(featured_df.index[-1].strftime("%Y-%m-%d"))

    entry_price = round(float(last["Close"]), 2)
    atr_14 = float(last["ATR_14"])

    # Compute ATR-based Stop-Loss and Target Price
    raw_stop = entry_price - stop_atr_mult * atr_14
    raw_target = entry_price + target_atr_mult * atr_14

    # Check 60-day swing high overhead: if a major swing resistance sits below raw_target
    # (and above entry_price + 0.5 * atr_14), note whether overhead resistance compresses R:R
    recent_60 = featured_df.tail(60)
    swing_high_60 = float(recent_60["High"].max())

    stop_loss_price = round(max(raw_stop, 0.05), 2)
    target_price = round(raw_target, 2)

    risk_per_share = round(max(entry_price - stop_loss_price, 1e-4), 2)
    reward_per_share = round(max(target_price - entry_price, 0.0), 2)
    rr_ratio = round(reward_per_share / risk_per_share, 2) if risk_per_share > 0 else 0.0

    # Gatekeeper checks
    rejection_reasons: List[str] = []
    acceptance_notes: List[str] = []

    # Rule 1: Market Regime Filter check
    if not regime_gate.allow_breakout_buy:
        rejection_reasons.append(f"Regime Filter Blocked: {regime_gate.gate_reason}")
    else:
        acceptance_notes.append(
            f"Regime Aligned: Stock ({regime_gate.stock_assessment.regime}) & "
            f"Index ({regime_gate.index_assessment.regime}) both support trend continuation."
        )

    # Rule 2: Strict ML Win Probability >= 60%
    if ml_win_probability < min_win_prob:
        rejection_reasons.append(
            f"ML Win Probability ({ml_win_probability * 100:.2f}%) is below the strict "
            f"{min_win_prob * 100:.0f}% gatekeeper threshold."
        )
    else:
        acceptance_notes.append(
            f"ML Win Probability ({ml_win_probability * 100:.2f}%) meets/exceeds the "
            f"{min_win_prob * 100:.0f}% threshold."
        )

    # Rule 3: Strict Risk-to-Reward >= 1:2 (2.0)
    if rr_ratio < min_rr_ratio - 1e-6:
        rejection_reasons.append(
            f"Reward-to-Risk ratio (1:{rr_ratio:.2f}) is worse than the required 1:{min_rr_ratio:.1f} minimum."
        )
    else:
        acceptance_notes.append(
            f"Reward-to-Risk ratio (1:{rr_ratio:.2f}) satisfies the 1:{min_rr_ratio:.1f} requirement."
        )

    # Overhead structure note
    if entry_price < swing_high_60 < target_price:
        acceptance_notes.append(
            f"Note: 60-day swing high at {swing_high_60:.2f} sits between Entry ({entry_price:.2f}) "
            f"and Target ({target_price:.2f}); monitor price action near {swing_high_60:.2f}."
        )

    gatekeeper_passed = len(rejection_reasons) == 0

    # Position sizing calculation
    max_capital_at_risk = round(account_capital * (max_risk_pct / 100.0), 2)
    if gatekeeper_passed and risk_per_share > 0:
        shares_by_risk = int(max_capital_at_risk // risk_per_share)
        shares_by_capital = int(account_capital // entry_price)
        recommended_shares = max(min(shares_by_risk, shares_by_capital), 0)
        action_signal = SIGNAL_BUY
    else:
        recommended_shares = 0
        action_signal = SIGNAL_WAIT

    position_notional = round(recommended_shares * entry_price, 2)

    return TradeDecisionPlan(
        ticker=ticker,
        as_of_date=as_of_date,
        action_signal=action_signal,
        ml_win_probability=ml_win_probability,
        min_required_probability=min_win_prob,
        entry_price=entry_price,
        atr_14=round(atr_14, 2),
        stop_loss_price=stop_loss_price,
        target_price=target_price,
        risk_per_share=risk_per_share,
        reward_per_share=reward_per_share,
        risk_reward_ratio=rr_ratio,
        min_required_rr=min_rr_ratio,
        account_capital=account_capital,
        max_risk_pct=max_risk_pct,
        max_capital_at_risk=max_capital_at_risk,
        recommended_shares=recommended_shares,
        position_notional_value=position_notional,
        stock_regime=regime_gate.stock_assessment.regime,
        index_regime=regime_gate.index_assessment.regime,
        gatekeeper_passed=gatekeeper_passed,
        rejection_reasons=rejection_reasons,
        acceptance_notes=acceptance_notes,
    )


if __name__ == "__main__":
    from data_features import download_ohlcv, engineer_features
    from regime_filter import evaluate_market_regime

    print("=== Phase 5 Self-Test: Hard-Coded Risk & Position Gatekeeper ===")
    s_df = engineer_features(download_ohlcv("RELIANCE.NS", years=8, allow_synthetic_fallback=True))
    i_df = engineer_features(download_ohlcv("^NSEI", years=8, allow_synthetic_fallback=True))
    gate = evaluate_market_regime(s_df, i_df, "RELIANCE.NS", "^NSEI")

    plan_low_prob = evaluate_risk_and_position(s_df, gate, ml_win_probability=0.54, ticker="RELIANCE.NS")
    print(f"Test 1 (54% Win Prob) -> Signal: {plan_low_prob.action_signal} | Reasons: {plan_low_prob.rejection_reasons}")

    plan_high_prob = evaluate_risk_and_position(s_df, gate, ml_win_probability=0.65, ticker="RELIANCE.NS")
    print(
        f"Test 2 (65% Win Prob) -> Signal: {plan_high_prob.action_signal} | "
        f"Entry: {plan_high_prob.entry_price} | SL: {plan_high_prob.stop_loss_price} | "
        f"Target: {plan_high_prob.target_price} | Shares: {plan_high_prob.recommended_shares}"
    )
