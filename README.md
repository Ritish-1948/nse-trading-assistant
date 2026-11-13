# Hybrid AI Trading Assistant (6-Phase Probabilistic & Risk-Managed Engine)

A modular Python decision-support and paper-trading system designed for NSE equities (`RELIANCE.NS`, `TCS.NS`, `INFY.NS`, and custom Yahoo Finance symbols) benchmarked against the Nifty 50 (`^NSEI`).

> **Important Risk Disclaimer**: Financial markets are noisy and non-stationary. No statistical or machine learning model achieves 100% accuracy. This codebase is built strictly as an unemotional **probability, regime-filtering, and risk-management decision-support tool** for research and paper trading.

## Project Structure

| File | Phase | Responsibility | 
| ----- | ----- | ----- | 
| `requirements.txt` | Setup | Pinned Python dependencies (`yfinance`, `pandas`, `numpy`, `scikit-learn`, `xgboost`, `lightgbm`, `matplotlib`, `mplfinance`, `joblib`) | 
| `data_features.py` | Phase 1 | Downloads 5–10 years of daily OHLCV via `yfinance` and engineers **34 causal mathematical features** across candlestick geometry, 20/50/200 EMAs, momentum/volatility (RSI-14, Stochastics, MACD hist, Bollinger width & squeeze, ATR-14), and ADX/OBV volume dynamics | 
| `regime_filter.py` | Phase 2 | Classifies both the broader index (`^NSEI`) and the target stock into **Strong Uptrend**, **Strong Downtrend**, or **Sideways / Choppy** using ADX(14) and 50/200 EMA alignment; automatically blocks breakout Buy signals in choppy or bearish regimes | 
| `train_model.py` | Phase 3 | Labels historical candles using a **10-day Triple-Barrier target** (*did price hit +2.0 ATR before -1.0 ATR within 10 trading days?*), trains an **XGBoost** classifier on the oldest 80% with a 10-day embargo purge before the unseen newest 20% test split, and reports honest out-of-sample Win Rate, Precision, ROC-AUC, and simulated Profit Factor | 
| `chart_plotter.py` | Phase 4 | Plots high-resolution candlestick charts with 20/50/200 EMAs, volume bars, and proposed Entry / Stop-Loss / Target horizontal lines via `mplfinance` for visual second-opinion inspection | 
| `risk_engine.py` | Phase 5 | Calculates exact Entry Price, `-1.0 ATR` Stop-Loss, `+2.0 ATR` Target Price, and position sizing, enforcing the strict gatekeeper rule: **if ML win probability < 60% OR Risk-to-Reward < 1:2 OR regime is blocked, output `WAIT (No Trade)`** | 
| `main.py` | Phase 6 | Single-command CLI (`python main.py --ticker RELIANCE.NS`) that runs all phases, prints a full briefing, saves the chart and trained model, and logs every evaluation to `paper_trades.csv` | 

## Step-by-Step Local Setup Instructions

### 1. Create and Activate a Virtual Environment

Open a terminal (Command Prompt, PowerShell, or Bash) inside the folder where you saved these files:

```
# Create virtual environment
python -m venv .venv

# Activate on Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# Activate on Windows (CMD)
.venv\Scripts\activate.bat

# Activate on macOS / Linux
source .venv/bin/activate

```

### 2. Install Dependencies

```
pip install --upgrade pip
pip install -r requirements.txt

```

### 3. Run the Assistant on Any NSE Ticker

Run a single stock (`RELIANCE.NS` by default):

```
python main.py --ticker RELIANCE.NS

```

Run on `TCS.NS` or `INFY.NS` with custom capital and risk parameters:

```
python main.py --ticker TCS.NS --capital 250000 --risk-pct 1.0 --years 10

```

Scan the entire default NSE universe (`RELIANCE.NS`, `TCS.NS`, `INFY.NS`) in one command:

```
python main.py --ticker ALL

```

Run in offline verification mode (uses synthetic regime-switching OHLCV data when testing without an internet connection):

```
python main.py --ticker RELIANCE.NS --offline-demo

```

## How to Use the Phase 4 Chart for a Second-Opinion Visual Check

Every run saves a high-resolution chart to `charts/<TICKER>_vision_check.png` showing the last 120 daily candles, 20/50/200 EMAs, volume, and the exact Entry, Stop-Loss (`-1.0 ATR`), and Target (`+2.0 ATR`) lines.

Before acting on any paper-trade signal, upload `charts/<TICKER>_vision_check.png` into Gemini with a prompt such as:

> *"Here is the generated chart for RELIANCE.NS with the proposed Entry, -1 ATR Stop-Loss, and +2 ATR Target. Do you see any major horizontal resistance zones, supply zones, or trendline obstacles sitting right above the entry before the +2 ATR target?"*

## Documented Engineering Assumptions & Safeguards

1. **Zero Data Leakage in Triple-Barrier Labeling & Splitting**:

   * The forward-looking triple-barrier window (`horizon_days=10`) inspects `t+1 ... t+10`.

   * The final 10 rows of the dataset (whose future 10-day outcomes are not yet fully known) are set to `NaN` and excluded from supervised training/testing.

   * An explicit 10-trading-day embargo gap is purged between the end of the 80% training split and the start of the 20% unseen test split so training labels never overlap with test period prices.

2. **Conservative Same-Bar Tie-Breaking**:

   * In daily OHLCV bars, if a future candle's `Low` touches `-1.0 ATR` and its `High` also touches `+2.0 ATR` on the same day, the labeler conservatively counts it as a **stop-loss (`0` / `-1.0R`)** rather than assuming the target was hit first.

3. **Scale-Invariant Features**:

   * All price-distance and MACD features are normalized as percentages of Close price (`dist_ema_20_pct`, `macd_hist_norm`, `atr_pct`, `bb_width_pct`) so tree splits remain valid as a stock's nominal price level grows over an 8–10 year window.

4. **Classifier Fallback Resilience**:

   * `train_model.py` uses `xgboost.XGBClassifier` by default and automatically falls back to `lightgbm.LGBMClassifier` or `sklearn.ensemble.HistGradientBoostingClassifier` if an environment lacks native OpenMP/XGBoost binaries.