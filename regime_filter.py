"""
Phase 2 — Market Regime Filter (regime_filter.py)
=================================================
Classifies both the broader market index (default: `^NSEI` Nifty 50) and the
individual stock into one of three regimes using ADX (14) and 50/200 EMA alignment:
  - Strong Uptrend
  - Strong Downtrend
  - Sideways / Choppy

Hard Gatekeeper Rule:
Automatically blocks breakout Buy signals when either the stock regime is
`Sideways / Choppy` (or `Strong Downtrend`) OR the broader market index is
in a hostile `Strong Downtrend` / `Sideways / Choppy` state.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Any

import pandas as pd


REGIME_STRONG_UPTREND = "Strong Uptrend"
REGIME_STRONG_DOWNTREND = "Strong Downtrend"
REGIME_SIDEWAYS_CHOPPY = "Sideways / Choppy"


@dataclass
class RegimeAssessment:
    """Structured output for a single asset's regime evaluation."""

    symbol: str
    as_of_date: str
    close: float
    ema_50: float
    ema_200: float
    adx_14: float
    plus_di_14: float
    minus_di_14: float
    regime: str
    ema_bullish_aligned: bool
    ema_bearish_aligned: bool
    trend_strength_ok: bool
    summary: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CombinedRegimeGate:
    """Combined stock + broader market index regime gatekeeper decision."""

    stock_assessment: RegimeAssessment
    index_assessment: RegimeAssessment
    allow_breakout_buy: bool
    allow_breakdown_sell: bool
    gate_status: str
    gate_reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stock_assessment": self.stock_assessment.to_dict(),
            "index_assessment": self.index_assessment.to_dict(),
            "allow_breakout_buy": self.allow_breakout_buy,
            "allow_breakdown_sell": self.allow_breakdown_sell,
            "gate_status": self.gate_status,
            "gate_reason": self.gate_reason,
        }


def classify_single_regime(
    featured_df: pd.DataFrame,
    symbol: str,
    adx_threshold: float = 20.0,
) -> RegimeAssessment:
    """
    Classifies the latest bar of `featured_df` into Strong Uptrend,
    Strong Downtrend, or Sideways / Choppy using ADX(14) and 50/200 EMA alignment.

    Classification Rules:
    ---------------------
    1. If ADX(14) < `adx_threshold` (default 20.0):
         -> `Sideways / Choppy` (trend strength is too weak; high whip-saw risk).
    2. If ADX(14) >= `adx_threshold` AND Close > EMA_50 AND Close > EMA_200 AND
       (EMA_50 >= EMA_200 or slope_ema_50_5d > 0):
         -> `Strong Uptrend`
    3. If ADX(14) >= `adx_threshold` AND Close < EMA_50 AND Close < EMA_200 AND
       (EMA_50 <= EMA_200 or slope_ema_50_5d < 0):
         -> `Strong Downtrend`
    4. Otherwise (conflicting price/EMA alignment such as price trapped between
       50-EMA and 200-EMA):
         -> `Sideways / Choppy`
    """
    if featured_df.empty:
        raise ValueError(f"Cannot classify regime: featured_df for {symbol} is empty.")

    last = featured_df.iloc[-1]
    as_of_date = str(featured_df.index[-1].strftime("%Y-%m-%d"))

    close = float(last["Close"])
    ema_50 = float(last["EMA_50"])
    ema_200 = float(last["EMA_200"])
    adx_14 = float(last["adx_14"])
    plus_di = float(last.get("PLUS_DI_14", 0.0))
    minus_di = float(last.get("MINUS_DI_14", 0.0))
    slope_50 = float(last.get("slope_ema_50_5d", 0.0))

    ema_bullish = (close > ema_50) and (close > ema_200) and ((ema_50 >= ema_200) or (slope_50 > 0.15))
    ema_bearish = (close < ema_50) and (close < ema_200) and ((ema_50 <= ema_200) or (slope_50 < -0.15))
    trend_strength_ok = adx_14 >= adx_threshold

    if trend_strength_ok and ema_bullish:
        regime = REGIME_STRONG_UPTREND
        summary = (
            f"{symbol} is in a {REGIME_STRONG_UPTREND} "
            f"(ADX={adx_14:.1f} >= {adx_threshold:.0f}, Close {close:.2f} > 50-EMA {ema_50:.2f} & 200-EMA {ema_200:.2f})."
        )
    elif trend_strength_ok and ema_bearish:
        regime = REGIME_STRONG_DOWNTREND
        summary = (
            f"{symbol} is in a {REGIME_STRONG_DOWNTREND} "
            f"(ADX={adx_14:.1f} >= {adx_threshold:.0f}, Close {close:.2f} < 50-EMA {ema_50:.2f} & 200-EMA {ema_200:.2f})."
        )
    else:
        regime = REGIME_SIDEWAYS_CHOPPY
        if not trend_strength_ok:
            reason_detail = f"weak trend strength (ADX={adx_14:.1f} < {adx_threshold:.0f})"
        else:
            reason_detail = (
                f"mixed EMA alignment (Close={close:.2f}, 50-EMA={ema_50:.2f}, 200-EMA={ema_200:.2f})"
            )
        summary = f"{symbol} is classified as {REGIME_SIDEWAYS_CHOPPY} due to {reason_detail}."

    return RegimeAssessment(
        symbol=symbol,
        as_of_date=as_of_date,
        close=close,
        ema_50=ema_50,
        ema_200=ema_200,
        adx_14=adx_14,
        plus_di_14=plus_di,
        minus_di_14=minus_di,
        regime=regime,
        ema_bullish_aligned=ema_bullish,
        ema_bearish_aligned=ema_bearish,
        trend_strength_ok=trend_strength_ok,
        summary=summary,
    )


def evaluate_market_regime(
    stock_df: pd.DataFrame,
    index_df: pd.DataFrame,
    stock_symbol: str = "RELIANCE.NS",
    index_symbol: str = "^NSEI",
    adx_threshold: float = 20.0,
    require_index_uptrend_for_buys: bool = False,
) -> CombinedRegimeGate:
    """
    Evaluates both the individual stock and the broader market index (`^NSEI`)
    and enforces the Phase 2 regime filter rule:
      - Automatically block breakout Buy signals when the stock regime is
        `Sideways / Choppy` (or `Strong Downtrend`), OR when the broader index
        is in `Sideways / Choppy` or `Strong Downtrend`.
    """
    stock_assess = classify_single_regime(stock_df, symbol=stock_symbol, adx_threshold=adx_threshold)
    index_assess = classify_single_regime(index_df, symbol=index_symbol, adx_threshold=adx_threshold)

    blocking_reasons = []

    if stock_assess.regime == REGIME_SIDEWAYS_CHOPPY:
        blocking_reasons.append(f"{stock_symbol} regime is {REGIME_SIDEWAYS_CHOPPY} (ADX={stock_assess.adx_14:.1f})")
    elif stock_assess.regime == REGIME_STRONG_DOWNTREND:
        blocking_reasons.append(f"{stock_symbol} is in a {REGIME_STRONG_DOWNTREND}")

    if index_assess.regime == REGIME_SIDEWAYS_CHOPPY:
        blocking_reasons.append(f"Broader index ({index_symbol}) is {REGIME_SIDEWAYS_CHOPPY} (ADX={index_assess.adx_14:.1f})")
    elif index_assess.regime == REGIME_STRONG_DOWNTREND:
        blocking_reasons.append(f"Broader index ({index_symbol}) is in a {REGIME_STRONG_DOWNTREND}")

    if require_index_uptrend_for_buys and index_assess.regime != REGIME_STRONG_UPTREND:
        if not any(index_symbol in r for r in blocking_reasons):
            blocking_reasons.append(f"Index {index_symbol} is not in {REGIME_STRONG_UPTREND}")

    allow_buy = (
        len(blocking_reasons) == 0
        and stock_assess.regime == REGIME_STRONG_UPTREND
        and index_assess.regime == REGIME_STRONG_UPTREND
    )

    allow_sell = (
        stock_assess.regime == REGIME_STRONG_DOWNTREND
        and index_assess.regime != REGIME_SIDEWAYS_CHOPPY
    )

    if allow_buy:
        gate_status = "PASS (Bullish Trend Aligned)"
        gate_reason = (
            f"Both {stock_symbol} (ADX={stock_assess.adx_14:.1f}) and {index_symbol} "
            f"(ADX={index_assess.adx_14:.1f}) are in a confirmed {REGIME_STRONG_UPTREND}."
        )
    elif allow_sell:
        gate_status = "BEARISH_ONLY (Long Breakouts Blocked)"
        gate_reason = (
            f"Long breakouts blocked: {'; '.join(blocking_reasons)}. "
            f"{stock_symbol} is aligned for short/defensive positioning."
        )
    else:
        gate_status = "BLOCKED (Regime Filter Active)"
        gate_reason = "Breakout Buy blocked — " + "; ".join(blocking_reasons) + "."

    return CombinedRegimeGate(
        stock_assessment=stock_assess,
        index_assessment=index_assess,
        allow_breakout_buy=allow_buy,
        allow_breakdown_sell=allow_sell,
        gate_status=gate_status,
        gate_reason=gate_reason,
    )
