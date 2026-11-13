"""
Phase 3 — Historical Pattern Probability Engine (train_model.py)
================================================================
Labels historical daily candles using a realistic triple-barrier style target:
  "Did price hit a +2.0 ATR profit target before hitting a -1.0 ATR stop-loss
   within the next 10 trading days?"

Trains an XGBoost (or LightGBM / Scikit-Learn / Pure-NumPy Gradient Boosted Trees
fallback) classifier strictly using a chronological 80/20 split (oldest 80% train,
newest 20% unseen out-of-sample test — zero random shuffling and zero lookahead
leakage).

Computes and reports honest out-of-sample metrics:
  - Baseline win rate vs. ML-filtered win rate (at >= 60% probability threshold)
  - Precision, Recall, ROC-AUC
  - Simulated out-of-sample Profit Factor (Gross Profit in R / Gross Loss in R)
"""

from __future__ import annotations

import os
import pickle
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from data_features import FEATURE_COLUMNS

try:
    import joblib
except ImportError:
    joblib = None

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

try:
    from sklearn.ensemble import HistGradientBoostingClassifier
except ImportError:
    HistGradientBoostingClassifier = None


def _precision_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = float(np.sum((y_true == 1) & (y_pred == 1)))
    fp = float(np.sum((y_true == 0) & (y_pred == 1)))
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def _recall_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = float(np.sum((y_true == 1) & (y_pred == 1)))
    fn = float(np.sum((y_true == 1) & (y_pred == 0)))
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def _roc_auc_score_np(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Computes exact ROC-AUC via the Mann-Whitney U / Wilcoxon rank statistic."""
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.50
    # Pairwise comparison: P(score_pos > score_neg) + 0.5 * P(score_pos == score_neg)
    diff = pos[:, None] - neg[None, :]
    wins = float(np.sum(diff > 0)) + 0.5 * float(np.sum(diff == 0))
    return wins / float(len(pos) * len(neg))


class NumpyGradientBoostedTrees:
    """
    Lightweight, self-contained Gradient Boosted Decision Stump/Depth-2 Tree
    classifier in pure NumPy. Used automatically as a fallback if neither
    `xgboost`, `lightgbm`, nor `scikit-learn` is installed in the current Python
    environment.
    """

    def __init__(
        self,
        n_estimators: int = 80,
        learning_rate: float = 0.08,
        scale_pos_weight: float = 1.0,
        min_samples_leaf: int = 20,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.scale_pos_weight = scale_pos_weight
        self.min_samples_leaf = min_samples_leaf
        self.base_log_odds: float = 0.0
        self.trees_: List[Dict[str, float]] = []
        self.feature_importances_: np.ndarray = np.array([])

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(z, -20.0, 20.0)))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NumpyGradientBoostedTrees":
        n_samples, n_features = X.shape
        y_float = y.astype(float)
        sample_weights = np.where(y_float == 1.0, self.scale_pos_weight, 1.0)

        pos_w = float(np.sum(sample_weights[y_float == 1.0]))
        neg_w = float(np.sum(sample_weights[y_float == 0.0]))
        p0 = np.clip(pos_w / max(pos_w + neg_w, 1e-9), 0.05, 0.95)
        self.base_log_odds = float(np.log(p0 / (1.0 - p0)))

        raw_pred = np.full(n_samples, self.base_log_odds, dtype=float)
        importances = np.zeros(n_features, dtype=float)
        self.trees_ = []

        # Precompute candidate quantile thresholds per feature for fast exact boosting
        quantiles = np.linspace(0.15, 0.85, 9)
        thresholds = [np.unique(np.quantile(X[:, j], quantiles)) for j in range(n_features)]

        for _ in range(self.n_estimators):
            prob = self._sigmoid(raw_pred)
            # Negative gradient (pseudo-residual) and Hessian with L2 regularization
            grad = sample_weights * (y_float - prob)
            hess = sample_weights * prob * (1.0 - prob)

            best_gain = -1.0
            best_tree = None

            total_g = float(np.sum(grad))
            total_h = float(np.sum(hess))
            reg_lambda = 5.0

            for j in range(n_features):
                col = X[:, j]
                for thr in thresholds[j]:
                    left_mask = col <= thr
                    n_left = int(np.sum(left_mask))
                    n_right = n_samples - n_left
                    if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                        continue

                    g_left = float(np.sum(grad[left_mask]))
                    h_left = float(np.sum(hess[left_mask]))
                    g_right = total_g - g_left
                    h_right = total_h - h_left

                    gain = (
                        (g_left * g_left) / (h_left + reg_lambda)
                        + (g_right * g_right) / (h_right + reg_lambda)
                        - (total_g * total_g) / (total_h + reg_lambda)
                    )
                    if gain > best_gain:
                        best_gain = gain
                        best_tree = {
                            "feature": float(j),
                            "threshold": float(thr),
                            "left_val": float(g_left / (h_left + reg_lambda)),
                            "right_val": float(g_right / (h_right + reg_lambda)),
                            "gain": float(max(gain, 0.0)),
                        }

            if best_tree is None:
                break

            feat_idx = int(best_tree["feature"])
            thr = best_tree["threshold"]
            step = np.where(X[:, feat_idx] <= thr, best_tree["left_val"], best_tree["right_val"])
            raw_pred += self.learning_rate * step
            importances[feat_idx] += best_tree["gain"]
            self.trees_.append(best_tree)

        total_imp = float(np.sum(importances))
        if total_imp > 0:
            self.feature_importances_ = importances / total_imp
        else:
            self.feature_importances_ = np.full(n_features, 1.0 / n_features)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raw = np.full(X.shape[0], self.base_log_odds, dtype=float)
        for tree in self.trees_:
            j = int(tree["feature"])
            thr = tree["threshold"]
            raw += self.learning_rate * np.where(
                X[:, j] <= thr, tree["left_val"], tree["right_val"]
            )
        p1 = self._sigmoid(raw)
        return np.column_stack([1.0 - p1, p1])


@dataclass
class ModelEvaluationMetrics:
    """Out-of-sample backtest and classification metrics on the unseen 20% test set."""

    ticker: str
    model_type: str
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    n_train: int
    n_test: int
    baseline_win_rate: float
    ml_win_rate_at_50: float
    ml_win_rate_at_60: float
    precision_at_50: float
    precision_at_60: float
    recall_at_50: float
    roc_auc: float
    signals_at_60: int
    simulated_profit_factor_at_60: float
    simulated_profit_factor_at_50: float
    expectancy_r_at_60: float
    latest_win_probability: float
    top_features: List[Tuple[str, float]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def apply_triple_barrier_labels(
    df: pd.DataFrame,
    profit_atr_mult: float = 2.0,
    stop_atr_mult: float = 1.0,
    horizon_days: int = 10,
) -> pd.DataFrame:
    """
    Applies a triple-barrier label to each historical candle `t`:
      - Upper Profit Barrier : `Close[t] + profit_atr_mult * ATR_14[t]` (+2.0 ATR)
      - Lower Stop Barrier   : `Close[t] - stop_atr_mult * ATR_14[t]`   (-1.0 ATR)
      - Vertical Time Barrier: `t + horizon_days` (10 trading days)

    Conservative tie-breaking rule on any future day `k` in `[1, horizon_days]`:
      - If `Low[t+k] <= stop_barrier`: stop-loss was touched -> label = 0, realized_r = -1.0
        (Checked first so ambiguous same-bar touches are counted as losses).
      - Else if `High[t+k] >= profit_barrier`: target hit -> label = 1, realized_r = +2.0
      - If neither barrier is hit within `horizon_days`:
        label = 0, and `realized_r` is the mark-to-market outcome at `t + horizon_days`
        clamped between `-stop_atr_mult` and `+profit_atr_mult`.
    """
    out = df.copy()
    closes = out["Close"].to_numpy(dtype=float)
    highs = out["High"].to_numpy(dtype=float)
    lows = out["Low"].to_numpy(dtype=float)
    atrs = out["ATR_14"].to_numpy(dtype=float)

    n = len(out)
    labels = np.full(n, np.nan, dtype=float)
    realized_r = np.full(n, np.nan, dtype=float)

    for i in range(n - horizon_days):
        entry = closes[i]
        atr = atrs[i]
        if not np.isfinite(atr) or atr <= 0:
            continue

        target_price = entry + profit_atr_mult * atr
        stop_price = entry - stop_atr_mult * atr

        outcome_label = 0.0
        outcome_r = np.clip(
            (closes[i + horizon_days] - entry) / atr, -stop_atr_mult, profit_atr_mult
        )

        for k in range(1, horizon_days + 1):
            f_high = highs[i + k]
            f_low = lows[i + k]

            hit_stop = f_low <= stop_price
            hit_target = f_high >= target_price

            if hit_stop:
                outcome_label = 0.0
                outcome_r = -stop_atr_mult
                break
            if hit_target:
                outcome_label = 1.0
                outcome_r = profit_atr_mult
                break

        labels[i] = outcome_label
        realized_r[i] = outcome_r

    out["target_label"] = labels
    out["realized_r"] = realized_r
    return out


def _build_classifier(random_state: int = 42, scale_pos_weight: float = 1.0) -> Tuple[Any, str]:
    """
    Instantiates the gradient-boosted decision tree classifier with regularized
    hyperparameters designed to prevent overfitting on noisy financial series.
    Prefers XGBoost, then LightGBM, then Scikit-Learn HistGradientBoosting,
    then the built-in pure-NumPy Gradient Boosted Trees implementation.
    """
    if XGBClassifier is not None:
        model = XGBClassifier(
            n_estimators=220,
            max_depth=3,
            learning_rate=0.035,
            subsample=0.80,
            colsample_bytree=0.80,
            min_child_weight=10,
            reg_alpha=0.5,
            reg_lambda=2.0,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=2,
        )
        return model, "XGBoost (XGBClassifier)"

    if LGBMClassifier is not None:
        model = LGBMClassifier(
            n_estimators=220,
            max_depth=3,
            num_leaves=7,
            learning_rate=0.035,
            subsample=0.80,
            colsample_bytree=0.80,
            min_child_samples=20,
            reg_alpha=0.5,
            reg_lambda=2.0,
            class_weight="balanced",
            random_state=random_state,
            verbose=-1,
        )
        return model, "LightGBM (LGBMClassifier)"

    if HistGradientBoostingClassifier is not None:
        model = HistGradientBoostingClassifier(
            max_iter=200,
            max_depth=3,
            learning_rate=0.04,
            min_samples_leaf=25,
            l2_regularization=2.0,
            class_weight="balanced",
            random_state=random_state,
        )
        return model, "Scikit-Learn (HistGradientBoostingClassifier)"

    model = NumpyGradientBoostedTrees(
        n_estimators=80,
        learning_rate=0.08,
        scale_pos_weight=scale_pos_weight,
        min_samples_leaf=20,
    )
    return model, "GradientBoostedTrees (NumPy Native Engine)"


def _compute_profit_factor(realized_r: np.ndarray, mask: np.ndarray) -> Tuple[float, float]:
    """
    Computes simulated Profit Factor (`Gross Profit R / |Gross Loss R|`) and
    per-trade R expectancy for trades selected by boolean `mask`.
    """
    selected_r = realized_r[mask]
    if len(selected_r) == 0:
        return 0.0, 0.0

    gross_profit = float(np.sum(selected_r[selected_r > 0]))
    gross_loss = float(np.abs(np.sum(selected_r[selected_r < 0])))
    expectancy = float(np.mean(selected_r))

    if gross_loss <= 1e-9:
        pf = gross_profit if gross_profit > 0 else 0.0
    else:
        pf = gross_profit / gross_loss
    return float(pf), expectancy


def train_and_evaluate_model(
    featured_df: pd.DataFrame,
    ticker: str = "RELIANCE.NS",
    train_ratio: float = 0.80,
    profit_atr_mult: float = 2.0,
    stop_atr_mult: float = 1.0,
    horizon_days: int = 10,
    model_save_path: Optional[str] = None,
) -> Tuple[Any, ModelEvaluationMetrics, pd.DataFrame]:
    """
    Labels historical data with the triple-barrier method, performs a strict
    chronological 80/20 train/test split (with an embargo gap of `horizon_days`
    between train and test so training labels never overlap with test bars),
    evaluates out-of-sample metrics, and scores the latest market bar.
    """
    labeled_df = apply_triple_barrier_labels(
        featured_df,
        profit_atr_mult=profit_atr_mult,
        stop_atr_mult=stop_atr_mult,
        horizon_days=horizon_days,
    )

    supervised_df = labeled_df.dropna(subset=["target_label", "realized_r"] + FEATURE_COLUMNS).copy()
    n_total = len(supervised_df)
    if n_total < 150:
        raise ValueError(
            f"Insufficient historical bars ({n_total}) for {ticker} after feature & barrier calculation."
        )

    split_idx = int(n_total * train_ratio)
    train_end_idx = max(split_idx - horizon_days, int(n_total * 0.70))

    train_df = supervised_df.iloc[:train_end_idx]
    test_df = supervised_df.iloc[split_idx:]

    X_train = train_df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y_train = train_df["target_label"].to_numpy(dtype=int)

    X_test = test_df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y_test = test_df["target_label"].to_numpy(dtype=int)
    r_test = test_df["realized_r"].to_numpy(dtype=float)

    pos_count = max(int(np.sum(y_train == 1)), 1)
    neg_count = max(int(np.sum(y_train == 0)), 1)
    damped_weight = float(np.sqrt(neg_count / pos_count))

    model, model_type = _build_classifier(scale_pos_weight=damped_weight)
    model.fit(X_train, y_train)

    y_prob_test = model.predict_proba(X_test)[:, 1]
    y_pred_50 = (y_prob_test >= 0.50).astype(int)
    y_pred_60 = (y_prob_test >= 0.60).astype(int)

    baseline_win_rate = float(np.mean(y_test))
    precision_50 = _precision_score_np(y_test, y_pred_50)
    precision_60 = _precision_score_np(y_test, y_pred_60)
    recall_50 = _recall_score_np(y_test, y_pred_50)
    roc_auc = _roc_auc_score_np(y_test, y_prob_test)

    mask_50 = y_prob_test >= 0.50
    mask_60 = y_prob_test >= 0.60
    signals_60 = int(np.sum(mask_60))

    pf_50, _ = _compute_profit_factor(r_test, mask_50)
    pf_60, exp_60 = _compute_profit_factor(r_test, mask_60)

    X_latest = labeled_df[FEATURE_COLUMNS].iloc[[-1]].to_numpy(dtype=float)
    latest_prob = float(model.predict_proba(X_latest)[0, 1])

    all_X = labeled_df[FEATURE_COLUMNS].to_numpy(dtype=float)
    labeled_df["ml_win_prob"] = model.predict_proba(all_X)[:, 1]

    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_, dtype=float)
        total_imp = np.sum(importances)
        if total_imp > 0:
            importances = importances / total_imp
        ranked = sorted(
            zip(FEATURE_COLUMNS, importances.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )[:8]
    else:
        corrs = [
            (col, float(abs(np.corrcoef(test_df[col].to_numpy(), y_prob_test)[0, 1])))
            for col in FEATURE_COLUMNS
        ]
        ranked = sorted(corrs, key=lambda x: np.nan_to_num(x[1]), reverse=True)[:8]

    metrics = ModelEvaluationMetrics(
        ticker=ticker,
        model_type=model_type,
        train_start=str(train_df.index[0].strftime("%Y-%m-%d")),
        train_end=str(train_df.index[-1].strftime("%Y-%m-%d")),
        test_start=str(test_df.index[0].strftime("%Y-%m-%d")),
        test_end=str(test_df.index[-1].strftime("%Y-%m-%d")),
        n_train=len(train_df),
        n_test=len(test_df),
        baseline_win_rate=baseline_win_rate,
        ml_win_rate_at_50=precision_50,
        ml_win_rate_at_60=precision_60,
        precision_at_50=precision_50,
        precision_at_60=precision_60,
        recall_at_50=recall_50,
        roc_auc=roc_auc,
        signals_at_60=signals_60,
        simulated_profit_factor_at_60=pf_60,
        simulated_profit_factor_at_50=pf_50,
        expectancy_r_at_60=exp_60,
        latest_win_probability=latest_prob,
        top_features=ranked,
    )

    if model_save_path:
        os.makedirs(os.path.dirname(os.path.abspath(model_save_path)), exist_ok=True)
        bundle = {
            "model": model,
            "model_type": model_type,
            "ticker": ticker,
            "feature_columns": FEATURE_COLUMNS,
            "metrics": metrics.to_dict(),
        }
        if joblib is not None:
            joblib.dump(bundle, model_save_path)
        else:
            with open(model_save_path, "wb") as f:
                pickle.dump(bundle, f)

    return model, metrics, labeled_df


def format_metrics_report(metrics: ModelEvaluationMetrics) -> str:
    """Formats a transparent out-of-sample evaluation summary string."""
    lines = [
        f"Model Architecture        : {metrics.model_type}",
        f"Chronological Train Split : {metrics.train_start} -> {metrics.train_end} ({metrics.n_train} bars)",
        f"Unseen Test Split (20%)   : {metrics.test_start} -> {metrics.test_end} ({metrics.n_test} bars)",
        f"Unfiltered Baseline Win % : {metrics.baseline_win_rate * 100:.2f}% (+2.0 ATR before -1.0 ATR in 10d)",
        f"Out-of-Sample ROC-AUC     : {metrics.roc_auc:.4f}",
        f"Precision / Win % (>=50%) : {metrics.precision_at_50 * 100:.2f}% (Profit Factor: {metrics.simulated_profit_factor_at_50:.2f})",
        f"Precision / Win % (>=60%) : {metrics.precision_at_60 * 100:.2f}% across {metrics.signals_at_60} unseen signals "
        f"(Profit Factor: {metrics.simulated_profit_factor_at_60:.2f}, Expectancy: {metrics.expectancy_r_at_60:+.2f}R)",
        f"Latest Bar ML Win Prob    : {metrics.latest_win_probability * 100:.2f}%",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    from data_features import download_ohlcv, engineer_features

    print("=== Phase 3 Self-Test: Triple-Barrier ML Probability Engine ===")
    df_raw = download_ohlcv("RELIANCE.NS", years=8, allow_synthetic_fallback=True)
    df_feat = engineer_features(df_raw)
    _, eval_metrics, _ = train_and_evaluate_model(
        df_feat,
        ticker="RELIANCE.NS",
        model_save_path="models/RELIANCE_NS_model.pkl",
    )
    print(format_metrics_report(eval_metrics))
