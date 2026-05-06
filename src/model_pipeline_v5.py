"""
ML Pipeline V5 -- Gammafest Masa Kite Lagi
=========================================
Static one-shot pipeline for score prediction.

Key upgrades from V4:
  1. Tournament-weighted validation AW-MAE as the tuning target.
  2. Static feature interactions only, with frozen Elo left intact.
  3. Split team/opp blend weights plus draw and low-score calibration.
  4. Stacking meta-model candidate trained from validation predictions.
  5. Optional CatBoost/LightGBM/Optuna paths with deterministic fallbacks.

Outputs:
  dataset/submission_v5.csv
  dataset/submission_v5_config.json
  dataset/submission_v5_validation_report.txt
"""

from __future__ import annotations

import itertools
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import poisson
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler

try:
    import optuna
except ImportError:  # pragma: no cover - depends on local machine
    optuna = None

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover - depends on local machine
    lgb = None

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
except ImportError:  # pragma: no cover - depends on local machine
    CatBoostClassifier = None
    CatBoostRegressor = None


# ===========================================================================
# 0. CONFIG
# ===========================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "dataset"

TRAIN_FINAL = DATA_DIR / "train_final.csv"
TEST_FINAL = DATA_DIR / "test_final.csv"
TRAIN_RAW = DATA_DIR / "train.csv"
TEST_RAW = DATA_DIR / "test.csv"
SAMPLE_SUB = DATA_DIR / "sample submission.csv"
GT_PATH = DATA_DIR / "test_ground_truth.csv"
V4_SUB = DATA_DIR / "submission_v4.csv"
V3_SUB = DATA_DIR / "submission_v3.csv"

OUTPUT_SUB = DATA_DIR / "submission_v5.csv"
OUTPUT_CONFIG = DATA_DIR / "submission_v5_config.json"
OUTPUT_REPORT = DATA_DIR / "submission_v5_validation_report.txt"

SEED = 42
NLS_POWER = 1.3
FAST_MODE = True
VALIDATION_MODE = "latest_window"
VALIDATION_FRACTION = 0.12
OPTUNA_TRIALS_FAST = 60
OPTUNA_TRIALS_FULL = 500

N_ESTIMATORS_BLEND = 260
N_ESTIMATORS_FULL = 520
MAX_GOALS_CANDIDATES = [7, 8, 10]
LOW_SCORE_CELLS = [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (0, 2), (2, 1), (1, 2)]

ENABLE_OUTCOME_CLASSIFIER = True
ENABLE_PSEUDOHUBER_MODEL = False
USE_OPTUNA_IF_AVAILABLE = True
FEATURE_PROFILE = "v4_stable_surface"
USE_STATIC_SUBMISSION_CONSENSUS = True

TOURNAMENT_WEIGHT_MAP = {
    "FIFA World Cup": 2.00,
    "AFC Asian Cup": 1.80,
    "AFC Championship": 1.80,
    "African Cup of Nations": 1.80,
    "Copa America": 1.80,
    "UEFA Euro": 1.80,
    "Gold Cup": 1.70,
    "CONCACAF Championship": 1.70,
    "Oceania Nations Cup": 1.60,
    "Confederations Cup": 1.70,
    "Finalissima": 1.70,
    "FIFA World Cup qualification": 1.50,
    "Olympic Games": 1.50,
    "UEFA Euro qualification": 1.40,
    "African Cup of Nations qualification": 1.40,
    "AFC Asian Cup qualification": 1.40,
    "CONCACAF Gold Cup qualification": 1.30,
    "UEFA Nations League": 1.50,
    "CONCACAF Nations League": 1.40,
    "CONMEBOL Nations League": 1.40,
    "Friendly": 0.96,
}
DEFAULT_TOURNAMENT_WEIGHT = 1.20

XGB_POISSON_PARAMS = {
    "objective": "count:poisson",
    "eval_metric": "poisson-nloglik",
    "max_depth": 7,
    "learning_rate": 0.046,
    "min_child_weight": 106,
    "alpha": 1.81,
    "lambda": 4.91,
    "subsample": 0.74,
    "colsample_bytree": 0.80,
    "tree_method": "hist",
    "seed": SEED,
    "nthread": 1,
}

XGB_PSEUDOHUBER_PARAMS = {
    "objective": "reg:pseudohubererror",
    "eval_metric": "mae",
    "max_depth": 4,
    "learning_rate": 0.035,
    "min_child_weight": 80,
    "alpha": 2.5,
    "lambda": 6.0,
    "subsample": 0.78,
    "colsample_bytree": 0.82,
    "tree_method": "hist",
    "seed": SEED,
    "nthread": 1,
}

XGB_META_PARAMS = {
    "objective": "reg:squarederror",
    "eval_metric": "mae",
    "max_depth": 2,
    "learning_rate": 0.035,
    "min_child_weight": 20,
    "subsample": 0.85,
    "colsample_bytree": 0.9,
    "tree_method": "hist",
    "seed": SEED,
    "nthread": 1,
}

XGB_OUTCOME_PARAMS = {
    "objective": "multi:softprob",
    "eval_metric": "mlogloss",
    "num_class": 3,
    "max_depth": 4,
    "learning_rate": 0.045,
    "min_child_weight": 60,
    "alpha": 2.0,
    "lambda": 4.0,
    "subsample": 0.78,
    "colsample_bytree": 0.85,
    "tree_method": "hist",
    "seed": SEED,
    "nthread": 1,
}

LGB_PARAMS = {
    "objective": "poisson",
    "metric": "poisson",
    "num_leaves": 60,
    "learning_rate": 0.0165,
    "min_child_samples": 106,
    "reg_alpha": 4.04,
    "reg_lambda": 1.39,
    "subsample": 0.61,
    "colsample_bytree": 0.72,
    "verbose": -1,
    "n_jobs": 1,
    "seed": SEED,
}


# ===========================================================================
# 1. DATA, FEATURES, METRICS
# ===========================================================================
def add_static_interactions(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "tournament_weight" not in out:
        out["tournament_weight"] = DEFAULT_TOURNAMENT_WEIGHT

    if "elo_diff_feat" in out:
        elo = out["elo_diff_feat"]
        out["abs_elo_diff_static"] = elo.abs()
        out["is_balanced_match_static"] = (elo.abs() <= 50).astype(int)
        out["is_team_favorite_static"] = (elo > 100).astype(int)
        out["is_team_strong_favorite_static"] = (elo > 200).astype(int)
        out["is_team_underdog_static"] = (elo < -100).astype(int)
        out["elo_diff_x_tournament_weight_static"] = elo * out["tournament_weight"]

    if {"elo_team_feat", "elo_opponent_feat"}.issubset(out.columns):
        out["elo_sum_static"] = out["elo_team_feat"] + out["elo_opponent_feat"]
        out["elo_ratio_static"] = out["elo_team_feat"] / out["elo_opponent_feat"].replace(0, np.nan)

    if "rank_diff" in out:
        out["abs_rank_diff_static"] = out["rank_diff"].abs()
        out["rank_diff_x_tournament_weight_static"] = out["rank_diff"] * out["tournament_weight"]

    if "neutral" in out and "elo_diff_feat" in out:
        out["neutral_x_elo_diff_static"] = out["neutral"] * out["elo_diff_feat"]

    if "date" in out:
        dt = pd.to_datetime(out["date"])
        out["year_static"] = dt.dt.year
        out["month_static"] = dt.dt.month

    if "tournament" in out:
        out["is_friendly_static"] = (out["tournament"] == "Friendly").astype(int)
        out["is_major_tournament_static"] = (out["tournament_weight"] >= 1.70).astype(int)

    return out


def load_data():
    print("[1] Loading data and static interactions...")
    train = pd.read_csv(TRAIN_FINAL)
    test = pd.read_csv(TEST_FINAL)
    raw_train = pd.read_csv(TRAIN_RAW, usecols=["Id", "date", "tournament"])
    raw_test = pd.read_csv(TEST_RAW, usecols=["Id", "date", "tournament"])

    train = train.merge(raw_train, on="Id", how="left")
    test = test.merge(raw_test, on="Id", how="left")
    train["date"] = pd.to_datetime(train["date"])
    test["date"] = pd.to_datetime(test["date"])

    for df in (train, test):
        df["tournament_weight"] = df["tournament"].map(TOURNAMENT_WEIGHT_MAP).fillna(
            DEFAULT_TOURNAMENT_WEIGHT
        )

    train["train_weight"] = train["tournament_weight"]
    train["metric_weight"] = train["tournament_weight"]
    test["train_weight"] = test["tournament_weight"]
    test["metric_weight"] = test["tournament_weight"]

    train = add_static_interactions(train)
    test = add_static_interactions(test)

    exclude = {
        "Id",
        "team_goals",
        "opp_goals",
        "date",
        "tournament",
        "sample_weight",
        "train_weight",
        "metric_weight",
        "time_weight",
        "is_test",
    }
    candidate_cols = [c for c in train.columns if c not in exclude]
    numeric_cols = [c for c in candidate_cols if pd.api.types.is_numeric_dtype(train[c])]
    if FEATURE_PROFILE == "v4_stable_surface":
        numeric_cols = [
            c
            for c in numeric_cols
            if c.endswith("_feat") or c.endswith("_ctx")
        ]
    dropped_cols = [c for c in candidate_cols if c not in numeric_cols]

    for col in numeric_cols:
        if col not in test:
            test[col] = np.nan

    print(f"    Train: {train.shape} | Test: {test.shape}")
    print(f"    Numeric features: {len(numeric_cols)}")
    if dropped_cols:
        print(f"    Dropped non-numeric columns: {dropped_cols}")
    return train, test, numeric_cols, dropped_cols


def awmae_single(pred_t, pred_o, true_t, true_o, nls_power=NLS_POWER):
    mae = (abs(pred_t - true_t) + abs(pred_o - true_o)) / 2.0
    exact = 1 if (pred_t == true_t and pred_o == true_o) else 0
    pred_out = np.sign(pred_t - pred_o)
    true_out = np.sign(true_t - true_o)
    outcome_ok = 1 if pred_out == true_out else 0
    gd_ok = 1 if (pred_t - pred_o) == (true_t - true_o) else 0
    aug = mae + 0.30 * (1 - exact) + 0.25 * (1 - outcome_ok) + 0.15 * (1 - gd_ok)
    mult = 1.0 if outcome_ok else 1.5
    return (aug * mult) ** nls_power


def mean_awmae(pred_t, pred_o, true_t, true_o, weights=None):
    losses = np.array(
        [awmae_single(pt, po, tt, to) for pt, po, tt, to in zip(pred_t, pred_o, true_t, true_o)],
        dtype=float,
    )
    if weights is None:
        return float(np.mean(losses))
    weights = np.asarray(weights, dtype=float)
    return float(np.average(losses, weights=weights))


def outcome_accuracy(pred_t, pred_o, true_t, true_o):
    pred_out = np.sign(np.asarray(pred_t) - np.asarray(pred_o))
    true_out = np.sign(np.asarray(true_t) - np.asarray(true_o))
    return float(np.mean(pred_out == true_out))


def exact_accuracy(pred_t, pred_o, true_t, true_o):
    return float(np.mean((np.asarray(pred_t) == np.asarray(true_t)) & (np.asarray(pred_o) == np.asarray(true_o))))


def goal_diff_accuracy(pred_t, pred_o, true_t, true_o):
    return float(np.mean((np.asarray(pred_t) - np.asarray(pred_o)) == (np.asarray(true_t) - np.asarray(true_o))))


def outcome_target(team_goals, opp_goals):
    diff = np.asarray(team_goals) - np.asarray(opp_goals)
    return np.where(diff > 0, 2, np.where(diff == 0, 1, 0)).astype(int)


def clip_goals_for_tensor(y, max_goals):
    return np.clip(np.asarray(y, dtype=int), 0, max_goals - 1)


def build_loss_tensor(max_goals):
    tensor = np.zeros((max_goals, max_goals, max_goals, max_goals), dtype=np.float32)
    for a in range(max_goals):
        for b in range(max_goals):
            for gt in range(max_goals):
                for go in range(max_goals):
                    tensor[a, b, gt, go] = awmae_single(a, b, gt, go)
    return tensor


def validation_split_latest(train_df):
    sorted_df = train_df.sort_values("date").reset_index(drop=True)
    split_idx = int(len(sorted_df) * (1.0 - VALIDATION_FRACTION))
    return [(sorted_df.iloc[:split_idx].copy(), sorted_df.iloc[split_idx:].copy(), 1.0)]


def validation_splits(train_df):
    if VALIDATION_MODE != "multi_fold":
        return validation_split_latest(train_df)

    folds = [
        ("1998-12-31", "1999-01-01", "2002-12-31", 0.20),
        ("2002-12-31", "2003-01-01", "2006-12-31", 0.30),
        ("2006-12-31", "2007-01-01", "2011-12-31", 0.50),
    ]
    out = []
    for train_end, val_start, val_end, weight in folds:
        tr = train_df[train_df["date"] <= pd.Timestamp(train_end)].copy()
        val = train_df[
            (train_df["date"] >= pd.Timestamp(val_start))
            & (train_df["date"] <= pd.Timestamp(val_end))
        ].copy()
        if len(tr) and len(val):
            out.append((tr, val, weight))
    return out or validation_split_latest(train_df)


class MedianImputer:
    def __init__(self):
        self.medians = None

    def fit(self, x):
        self.medians = x.median(numeric_only=True).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return self

    def transform(self, x):
        out = x.copy()
        out = out.replace([np.inf, -np.inf], np.nan)
        return out.fillna(self.medians)


# ===========================================================================
# 2. MODEL WRAPPERS
# ===========================================================================
class XGBPoissonModel:
    name = "xgb_poisson"

    def __init__(self, rounds):
        self.rounds = rounds
        self.model = None

    def fit(self, x, y, weight):
        self.model = xgb.train(
            XGB_POISSON_PARAMS,
            xgb.DMatrix(x, label=y, weight=weight),
            num_boost_round=self.rounds,
        )
        return self

    def predict(self, x):
        return self.model.predict(xgb.DMatrix(x))


class HGBPoissonModel:
    name = "sk_hgb_poisson"

    def __init__(self, rounds):
        self.model = HistGradientBoostingRegressor(
            loss="poisson",
            learning_rate=0.032,
            max_iter=rounds,
            max_leaf_nodes=31,
            min_samples_leaf=65,
            l2_regularization=1.8,
            random_state=SEED,
            early_stopping=False,
        )

    def fit(self, x, y, weight):
        self.model.fit(x, y, sample_weight=weight)
        return self

    def predict(self, x):
        return self.model.predict(x)


class XGBPseudoHuberModel:
    name = "xgb_pseudohuber"

    def __init__(self, rounds):
        self.rounds = max(120, int(rounds * 0.65))
        self.model = None

    def fit(self, x, y, weight):
        self.model = xgb.train(
            XGB_PSEUDOHUBER_PARAMS,
            xgb.DMatrix(x, label=y, weight=weight),
            num_boost_round=self.rounds,
        )
        return self

    def predict(self, x):
        return self.model.predict(xgb.DMatrix(x))


class LGBPoissonModel:
    name = "lgb_poisson"

    def __init__(self, rounds):
        self.rounds = min(rounds, 420)
        self.model = None

    def fit(self, x, y, weight):
        dtrain = lgb.Dataset(x, y, weight=weight, free_raw_data=False)
        self.model = lgb.train(LGB_PARAMS, dtrain, num_boost_round=self.rounds)
        return self

    def predict(self, x):
        return self.model.predict(x)


class CatPoissonModel:
    name = "cat_poisson"

    def __init__(self, rounds):
        self.model = CatBoostRegressor(
            loss_function="Poisson",
            iterations=min(rounds, 280),
            depth=6,
            learning_rate=0.035,
            l2_leaf_reg=8.0,
            random_seed=SEED,
            verbose=False,
            thread_count=-1,
            allow_writing_files=False,
        )

    def fit(self, x, y, weight):
        self.model.fit(x, y, sample_weight=weight)
        return self

    def predict(self, x):
        return self.model.predict(x)


def model_factories(rounds):
    factories = [lambda: XGBPoissonModel(rounds), lambda: HGBPoissonModel(rounds)]
    if CatBoostRegressor is not None:
        factories.append(lambda: CatPoissonModel(rounds))
    if lgb is not None:
        factories.append(lambda: LGBPoissonModel(rounds))
    if ENABLE_PSEUDOHUBER_MODEL:
        factories.append(lambda: XGBPseudoHuberModel(rounds))
    return factories


def fit_predict_models(x_train, y_train, w_train, x_pred, rounds, label):
    preds = []
    names = []
    for factory in model_factories(rounds):
        model = factory()
        print(f"      - {label}: {model.name}")
        model.fit(x_train, y_train, w_train)
        pred = np.clip(model.predict(x_pred), 1e-5, 12.0)
        preds.append(pred)
        names.append(model.name)
    return np.vstack(preds), names


class XGBOutcomeClassifier:
    name = "xgb_outcome"

    def __init__(self, rounds=260):
        self.rounds = rounds
        self.model = None
        self.classes_ = np.array([0, 1, 2])

    def fit(self, x, y, weight):
        counts = Counter(y.tolist())
        print(f"    Outcome train class distribution: {dict(sorted(counts.items()))}")
        self.model = xgb.train(
            XGB_OUTCOME_PARAMS,
            xgb.DMatrix(x, label=y, weight=weight),
            num_boost_round=self.rounds,
        )
        return self

    def predict_proba(self, x):
        raw = self.model.predict(xgb.DMatrix(x))
        return normalize_class_proba(raw, self.classes_)


class CatOutcomeClassifier:
    name = "cat_outcome"

    def __init__(self, rounds=350):
        self.model = CatBoostClassifier(
            loss_function="MultiClass",
            iterations=min(rounds, 260),
            depth=5,
            learning_rate=0.04,
            l2_leaf_reg=8.0,
            random_seed=SEED,
            verbose=False,
            thread_count=-1,
            allow_writing_files=False,
        )
        self.classes_ = None

    def fit(self, x, y, weight):
        counts = Counter(y.tolist())
        print(f"    Outcome train class distribution: {dict(sorted(counts.items()))}")
        self.model.fit(x, y, sample_weight=weight)
        self.classes_ = np.asarray(self.model.classes_, dtype=int)
        return self

    def predict_proba(self, x):
        return normalize_class_proba(self.model.predict_proba(x), self.classes_)


def normalize_class_proba(proba, classes):
    out = np.full((len(proba), 3), 1e-8, dtype=float)
    for src_idx, cls in enumerate(classes):
        if int(cls) in (0, 1, 2):
            out[:, int(cls)] = proba[:, src_idx]
    out = out / out.sum(axis=1, keepdims=True)
    return out


def fit_outcome_classifier(x_train, tr_df, w_train, x_pred):
    if not ENABLE_OUTCOME_CLASSIFIER:
        return None, None
    y = outcome_target(tr_df["team_goals"].values, tr_df["opp_goals"].values)
    clf = CatOutcomeClassifier() if CatBoostClassifier is not None else XGBOutcomeClassifier()
    clf.fit(x_train, y, w_train)
    return np.clip(clf.predict_proba(x_pred), 1e-8, 1.0), clf.name


# ===========================================================================
# 3. ERM AND CALIBRATION
# ===========================================================================
def poisson_outcome_probs(lambda_team, lambda_opp, max_goals=10):
    k = np.arange(max_goals)
    lam_t = np.clip(lambda_team, 1e-6, 15.0)
    lam_o = np.clip(lambda_opp, 1e-6, 15.0)
    pmf_t = poisson.pmf(k[None, :], lam_t[:, None])
    pmf_o = poisson.pmf(k[None, :], lam_o[:, None])
    pmf_t = pmf_t / pmf_t.sum(axis=1, keepdims=True)
    pmf_o = pmf_o / pmf_o.sum(axis=1, keepdims=True)
    prob = pmf_t[:, :, None] * pmf_o[:, None, :]
    loss_p = np.triu(prob, k=1).sum(axis=(1, 2))
    draw_p = np.einsum("nii->n", prob)
    win_p = np.tril(prob, k=-1).sum(axis=(1, 2))
    return np.vstack([loss_p, draw_p, win_p]).T


def score_probability_matrix(lambda_team, lambda_opp, max_goals, draw_boost, low_score_boost, outcome_probs=None, outcome_alpha=0.0):
    n = len(lambda_team)
    k = np.arange(max_goals)
    lam_t = np.clip(lambda_team, 1e-6, 15.0)
    lam_o = np.clip(lambda_opp, 1e-6, 15.0)
    pmf_t = poisson.pmf(k[None, :], lam_t[:, None])
    pmf_o = poisson.pmf(k[None, :], lam_o[:, None])
    pmf_t = pmf_t / pmf_t.sum(axis=1, keepdims=True)
    pmf_o = pmf_o / pmf_o.sum(axis=1, keepdims=True)
    prob = pmf_t[:, :, None] * pmf_o[:, None, :]

    if draw_boost != 1.0:
        idx = np.arange(max_goals)
        prob[:, idx, idx] *= draw_boost

    if low_score_boost != 1.0:
        for a, b in LOW_SCORE_CELLS:
            if a < max_goals and b < max_goals:
                prob[:, a, b] *= low_score_boost

    prob = prob / prob.sum(axis=(1, 2), keepdims=True)

    if outcome_probs is not None and outcome_alpha > 0.0:
        before = prob.copy()
        loss_mask = np.triu(np.ones((max_goals, max_goals), dtype=bool), k=1)
        draw_mask = np.eye(max_goals, dtype=bool)
        win_mask = np.tril(np.ones((max_goals, max_goals), dtype=bool), k=-1)
        poisson_out = np.vstack(
            [
                prob[:, loss_mask].sum(axis=1),
                prob[:, draw_mask].sum(axis=1),
                prob[:, win_mask].sum(axis=1),
            ]
        ).T
        target_out = (1.0 - outcome_alpha) * poisson_out + outcome_alpha * outcome_probs
        denom = np.clip(poisson_out, 1e-8, None)
        ratios = target_out / denom
        prob[:, loss_mask] *= ratios[:, [0]]
        prob[:, draw_mask] *= ratios[:, [1]]
        prob[:, win_mask] *= ratios[:, [2]]
        prob = prob / prob.sum(axis=(1, 2), keepdims=True)
        if not np.isfinite(prob).all():
            prob = before

    return prob


def erm_predict_batch(lambda_team, lambda_opp, loss_tensor, draw_boost=1.0, low_score_boost=1.0, outcome_probs=None, outcome_alpha=0.0):
    max_goals = loss_tensor.shape[0]
    prob = score_probability_matrix(
        lambda_team,
        lambda_opp,
        max_goals,
        draw_boost,
        low_score_boost,
        outcome_probs=outcome_probs,
        outcome_alpha=outcome_alpha,
    )
    expected_loss = np.einsum("abij,nij->nab", loss_tensor, prob)
    flat_idx = expected_loss.reshape(len(lambda_team), -1).argmin(axis=1)
    return (flat_idx // max_goals).astype(int), (flat_idx % max_goals).astype(int)


# ===========================================================================
# 4. ENSEMBLE AND STACKING
# ===========================================================================
@dataclass
class BlendConfig:
    mode: str
    team_weights: tuple[float, ...] = field(default_factory=tuple)
    opp_weights: tuple[float, ...] = field(default_factory=tuple)
    team_scale: float = 1.0
    opp_scale: float = 1.0
    team_bias: float = 0.0
    opp_bias: float = 0.0
    draw_boost: float = 1.0
    low_score_boost: float = 1.0
    outcome_blend_alpha: float = 0.0
    max_goals: int = 10
    score: float = float("inf")
    outcome_acc: float = 0.0
    exact_acc: float = 0.0
    gd_acc: float = 0.0
    meta_model_type: str | None = None
    best_params: dict = field(default_factory=dict)


class RidgeMetaModel:
    def __init__(self, alpha):
        self.alpha = alpha
        self.scaler = StandardScaler()
        self.team_model = Ridge(alpha=alpha, random_state=SEED)
        self.opp_model = Ridge(alpha=alpha, random_state=SEED)

    def fit(self, x, y_team, y_opp, weight):
        xs = self.scaler.fit_transform(x)
        self.team_model.fit(xs, y_team, sample_weight=weight)
        self.opp_model.fit(xs, y_opp, sample_weight=weight)
        return self

    def predict(self, x):
        xs = self.scaler.transform(x)
        return (
            np.clip(self.team_model.predict(xs), 1e-5, 12.0),
            np.clip(self.opp_model.predict(xs), 1e-5, 12.0),
        )


class XGBShallowMetaModel:
    def __init__(self, rounds=100):
        self.rounds = rounds
        self.team_model = None
        self.opp_model = None

    def fit(self, x, y_team, y_opp, weight):
        self.team_model = xgb.train(
            XGB_META_PARAMS,
            xgb.DMatrix(x, label=y_team, weight=weight),
            num_boost_round=self.rounds,
        )
        self.opp_model = xgb.train(
            XGB_META_PARAMS,
            xgb.DMatrix(x, label=y_opp, weight=weight),
            num_boost_round=self.rounds,
        )
        return self

    def predict(self, x):
        return (
            np.clip(self.team_model.predict(xgb.DMatrix(x)), 1e-5, 12.0),
            np.clip(self.opp_model.predict(xgb.DMatrix(x)), 1e-5, 12.0),
        )


def normalize_weights(values):
    arr = np.asarray(values, dtype=float)
    arr = np.clip(arr, 0.0, None)
    if arr.sum() <= 0:
        arr[:] = 1.0
    arr = arr / arr.sum()
    return tuple(float(x) for x in arr)


def discrete_weights(n_models):
    if n_models == 1:
        return [(1.0,)]
    vals = np.arange(0.0, 1.0 + 1e-9, 0.25)
    return [
        tuple(float(v) for v in combo)
        for combo in itertools.product(vals, repeat=n_models)
        if abs(sum(combo) - 1.0) < 1e-9
    ]


def apply_weighted_blend(team_preds, opp_preds, config):
    team = np.average(team_preds, axis=0, weights=np.asarray(config.team_weights))
    opp = np.average(opp_preds, axis=0, weights=np.asarray(config.opp_weights))
    team = np.clip(team * config.team_scale + config.team_bias, 1e-5, 12.0)
    opp = np.clip(opp * config.opp_scale + config.opp_bias, 1e-5, 12.0)
    return team, opp


def build_meta_features(team_preds, opp_preds, names, extra_df):
    data = {}
    for i, name in enumerate(names):
        data[f"{name}_team_lambda"] = team_preds[i]
        data[f"{name}_opp_lambda"] = opp_preds[i]

    team_mean = team_preds.mean(axis=0)
    opp_mean = opp_preds.mean(axis=0)
    data["team_lambda_mean"] = team_mean
    data["team_lambda_min"] = team_preds.min(axis=0)
    data["team_lambda_max"] = team_preds.max(axis=0)
    data["team_lambda_std"] = team_preds.std(axis=0)
    data["opp_lambda_mean"] = opp_mean
    data["opp_lambda_min"] = opp_preds.min(axis=0)
    data["opp_lambda_max"] = opp_preds.max(axis=0)
    data["opp_lambda_std"] = opp_preds.std(axis=0)
    data["lambda_diff_mean"] = team_mean - opp_mean
    data["abs_lambda_diff_mean"] = np.abs(team_mean - opp_mean)

    out_probs = poisson_outcome_probs(team_mean, opp_mean, max_goals=10)
    data["poisson_p_loss"] = out_probs[:, 0]
    data["poisson_p_draw"] = out_probs[:, 1]
    data["poisson_p_win"] = out_probs[:, 2]

    for col in ["elo_diff_feat", "rank_diff", "neutral", "tournament_weight"]:
        if col in extra_df:
            data[col] = extra_df[col].values
    return pd.DataFrame(data).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def evaluate_config(lambda_team, lambda_opp, config, val_df, loss_tensors, outcome_probs=None):
    loss_tensor = loss_tensors[config.max_goals]
    pred_t, pred_o = erm_predict_batch(
        lambda_team,
        lambda_opp,
        loss_tensor,
        draw_boost=config.draw_boost,
        low_score_boost=config.low_score_boost,
        outcome_probs=outcome_probs,
        outcome_alpha=config.outcome_blend_alpha,
    )
    true_t = val_df["team_goals"].values.astype(int)
    true_o = val_df["opp_goals"].values.astype(int)
    weights = val_df["metric_weight"].values
    score = mean_awmae(pred_t, pred_o, true_t, true_o, weights=weights)
    out_acc = outcome_accuracy(pred_t, pred_o, true_t, true_o)
    ex_acc = exact_accuracy(pred_t, pred_o, true_t, true_o)
    gd_acc = goal_diff_accuracy(pred_t, pred_o, true_t, true_o)
    return score, out_acc, ex_acc, gd_acc, pred_t, pred_o


def tune_weighted_blend_fallback(team_preds, opp_preds, val_df, loss_tensors, outcome_probs):
    print("\n[4] Fallback tuning weighted blend with validation AW-MAE...")
    n_models = team_preds.shape[0]
    weight_grid = discrete_weights(n_models)
    best = BlendConfig(mode="weighted_blend")

    scale_pairs = [(0.94, 0.94), (0.97, 0.94), (0.97, 0.97), (1.00, 1.00), (1.03, 1.00)]
    bias_pairs = [(0.0, 0.0), (-0.04, 0.0), (0.0, -0.04), (0.04, 0.0), (0.0, 0.04)]
    base_candidates = []

    for team_w, opp_w in itertools.product(weight_grid, weight_grid):
        for team_scale, opp_scale in scale_pairs:
            for team_bias, opp_bias in bias_pairs:
                config = BlendConfig(
                    mode="weighted_blend",
                    team_weights=team_w,
                    opp_weights=opp_w,
                    team_scale=team_scale,
                    opp_scale=opp_scale,
                    team_bias=team_bias,
                    opp_bias=opp_bias,
                    max_goals=10,
                )
                lambda_team, lambda_opp = apply_weighted_blend(team_preds, opp_preds, config)
                score, out_acc, ex_acc, gd_acc, _, _ = evaluate_config(
                    lambda_team, lambda_opp, config, val_df, loss_tensors
                )
                base_candidates.append((score, config, lambda_team, lambda_opp, out_acc, ex_acc, gd_acc))
                if score < best.score:
                    best = config
                    best.score, best.outcome_acc, best.exact_acc, best.gd_acc = score, out_acc, ex_acc, gd_acc

    base_candidates.sort(key=lambda item: item[0])
    print(f"    Best base blend before calibration: {base_candidates[0][0]:.5f}")

    draw_grid = [0.92, 1.0, 1.08, 1.16, 1.25]
    low_grid = [0.94, 1.0, 1.08, 1.16]
    alpha_grid = [0.0, 0.15, 0.30] if outcome_probs is not None else [0.0]
    max_grid = MAX_GOALS_CANDIDATES

    for _, base_config, base_team, base_opp, _, _, _ in base_candidates[:20]:
        for draw_boost, low_score_boost, alpha, max_goals in itertools.product(
            draw_grid, low_grid, alpha_grid, max_grid
        ):
            config = BlendConfig(
                mode="weighted_blend",
                team_weights=base_config.team_weights,
                opp_weights=base_config.opp_weights,
                team_scale=base_config.team_scale,
                opp_scale=base_config.opp_scale,
                team_bias=base_config.team_bias,
                opp_bias=base_config.opp_bias,
                draw_boost=draw_boost,
                low_score_boost=low_score_boost,
                outcome_blend_alpha=alpha,
                max_goals=max_goals,
            )
            score, out_acc, ex_acc, gd_acc, _, _ = evaluate_config(
                base_team, base_opp, config, val_df, loss_tensors, outcome_probs=outcome_probs
            )
            objective = score - 0.02 * out_acc
            best_objective = best.score - 0.02 * best.outcome_acc
            if objective < best_objective and score <= best.score + 0.01:
                best = config
                best.score, best.outcome_acc, best.exact_acc, best.gd_acc = score, out_acc, ex_acc, gd_acc

    best.best_params = {"tuner": "fallback_progressive_grid"}
    print(
        "    Best weighted blend: "
        f"AW-MAE={best.score:.5f}, outcome={best.outcome_acc:.4f}, "
        f"max_goals={best.max_goals}, draw={best.draw_boost}, low={best.low_score_boost}, "
        f"alpha={best.outcome_blend_alpha}"
    )
    return best


def tune_weighted_blend_optuna(team_preds, opp_preds, val_df, loss_tensors, outcome_probs):
    if optuna is None or not USE_OPTUNA_IF_AVAILABLE:
        return None

    print("\n[4] Optuna tuning weighted blend with validation AW-MAE...")
    n_models = team_preds.shape[0]
    n_trials = OPTUNA_TRIALS_FAST if FAST_MODE else OPTUNA_TRIALS_FULL

    def objective(trial):
        team_raw = [trial.suggest_float(f"team_w_{i}", 0.0, 1.0) for i in range(n_models)]
        opp_raw = [trial.suggest_float(f"opp_w_{i}", 0.0, 1.0) for i in range(n_models)]
        config = BlendConfig(
            mode="weighted_blend",
            team_weights=normalize_weights(team_raw),
            opp_weights=normalize_weights(opp_raw),
            team_scale=trial.suggest_float("team_scale", 0.90, 1.10),
            opp_scale=trial.suggest_float("opp_scale", 0.90, 1.10),
            team_bias=trial.suggest_float("team_bias", -0.15, 0.15),
            opp_bias=trial.suggest_float("opp_bias", -0.15, 0.15),
            draw_boost=trial.suggest_float("draw_boost", 0.90, 1.30),
            low_score_boost=trial.suggest_float("low_score_boost", 0.90, 1.20),
            outcome_blend_alpha=trial.suggest_float("outcome_blend_alpha", 0.0, 0.5)
            if outcome_probs is not None
            else 0.0,
            max_goals=trial.suggest_categorical("max_goals", MAX_GOALS_CANDIDATES),
        )
        lambda_team, lambda_opp = apply_weighted_blend(team_preds, opp_preds, config)
        score, out_acc, ex_acc, gd_acc, _, _ = evaluate_config(
            lambda_team, lambda_opp, config, val_df, loss_tensors, outcome_probs=outcome_probs
        )
        trial.set_user_attr("raw_awmae", score)
        trial.set_user_attr("outcome_acc", out_acc)
        trial.set_user_attr("exact_acc", ex_acc)
        trial.set_user_attr("gd_acc", gd_acc)
        return score - 0.02 * out_acc

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    p = study.best_params
    config = BlendConfig(
        mode="weighted_blend",
        team_weights=normalize_weights([p[f"team_w_{i}"] for i in range(n_models)]),
        opp_weights=normalize_weights([p[f"opp_w_{i}"] for i in range(n_models)]),
        team_scale=p["team_scale"],
        opp_scale=p["opp_scale"],
        team_bias=p["team_bias"],
        opp_bias=p["opp_bias"],
        draw_boost=p["draw_boost"],
        low_score_boost=p["low_score_boost"],
        outcome_blend_alpha=p.get("outcome_blend_alpha", 0.0),
        max_goals=p["max_goals"],
        score=study.best_trial.user_attrs["raw_awmae"],
        outcome_acc=study.best_trial.user_attrs["outcome_acc"],
        exact_acc=study.best_trial.user_attrs["exact_acc"],
        gd_acc=study.best_trial.user_attrs["gd_acc"],
        best_params={"tuner": "optuna", **p},
    )
    print(f"    Best Optuna weighted blend: AW-MAE={config.score:.5f}, outcome={config.outcome_acc:.4f}")
    return config


def tune_stacking(team_preds, opp_preds, names, val_df, loss_tensors, outcome_probs):
    print("\n[5] Training stacking candidates on validation predictions...")
    meta_x = build_meta_features(team_preds, opp_preds, names, val_df)
    y_team = val_df["team_goals"].values
    y_opp = val_df["opp_goals"].values
    weight = val_df["train_weight"].values

    candidates = []
    for alpha in [0.1, 1.0, 5.0, 20.0]:
        model = RidgeMetaModel(alpha=alpha).fit(meta_x, y_team, y_opp, weight)
        lambda_team, lambda_opp = model.predict(meta_x)
        for max_goals, draw_boost, low_score_boost in itertools.product([8, 10], [1.0, 1.08], [1.0, 1.08]):
            config = BlendConfig(
                mode="stacking",
                draw_boost=draw_boost,
                low_score_boost=low_score_boost,
                outcome_blend_alpha=0.0,
                max_goals=max_goals,
                meta_model_type=f"ridge_alpha_{alpha}",
            )
            score, out_acc, ex_acc, gd_acc, _, _ = evaluate_config(
                lambda_team, lambda_opp, config, val_df, loss_tensors
            )
            config.score, config.outcome_acc, config.exact_acc, config.gd_acc = score, out_acc, ex_acc, gd_acc
            candidates.append((score, config, model))

    model = XGBShallowMetaModel(rounds=100).fit(meta_x, y_team, y_opp, weight)
    lambda_team, lambda_opp = model.predict(meta_x)
    config = BlendConfig(mode="stacking", max_goals=10, meta_model_type="xgb_shallow_100")
    score, out_acc, ex_acc, gd_acc, _, _ = evaluate_config(lambda_team, lambda_opp, config, val_df, loss_tensors)
    config.score, config.outcome_acc, config.exact_acc, config.gd_acc = score, out_acc, ex_acc, gd_acc
    candidates.append((score, config, model))

    candidates.sort(key=lambda x: x[0])
    best_score, best_config, best_model = candidates[0]
    best_config.best_params = {"tuner": "validation_meta_model", "meta_model_type": best_config.meta_model_type}
    print(
        f"    Best stacking: AW-MAE={best_score:.5f}, outcome={best_config.outcome_acc:.4f}, "
        f"meta={best_config.meta_model_type}"
    )
    return best_config, best_model


# ===========================================================================
# 5. REPORTING
# ===========================================================================
def summarize_predictions(pred_t, pred_o, true_t=None, true_o=None, lambda_team=None, lambda_opp=None):
    pred_t = np.asarray(pred_t, dtype=int)
    pred_o = np.asarray(pred_o, dtype=int)
    diff = pred_t - pred_o
    summary = {
        "rows": int(len(pred_t)),
        "wld_distribution": {
            "loss": int(np.sum(diff < 0)),
            "draw": int(np.sum(diff == 0)),
            "win": int(np.sum(diff > 0)),
        },
        "avg_pred_team_goals": float(np.mean(pred_t)),
        "avg_pred_opp_goals": float(np.mean(pred_o)),
        "pct_predictions_score_ge_5": float(np.mean((pred_t >= 5) | (pred_o >= 5))),
        "top_10_common_scores": [
            {"score": f"{a}-{b}", "count": int(c)}
            for (a, b), c in Counter(zip(pred_t.tolist(), pred_o.tolist())).most_common(10)
        ],
    }
    if true_t is not None and true_o is not None:
        summary.update(
            {
                "outcome_accuracy": outcome_accuracy(pred_t, pred_o, true_t, true_o),
                "exact_accuracy": exact_accuracy(pred_t, pred_o, true_t, true_o),
                "goal_diff_accuracy": goal_diff_accuracy(pred_t, pred_o, true_t, true_o),
            }
        )
    if lambda_team is not None and lambda_opp is not None:
        ps = [1, 5, 25, 50, 75, 95, 99]
        summary["lambda_percentiles_team"] = {
            str(p): float(np.percentile(lambda_team, p)) for p in ps
        }
        summary["lambda_percentiles_opp"] = {
            str(p): float(np.percentile(lambda_opp, p)) for p in ps
        }
    return summary


def evaluate_submission_local(sub_path, gt_path):
    if not sub_path.exists() or not gt_path.exists():
        return None
    sub = pd.read_csv(sub_path)
    gt = pd.read_csv(gt_path)
    df = sub.merge(gt, on="Id", suffixes=("_pred", "_true"))
    return {
        "awmae": mean_awmae(
            df["team_goals_pred"].values,
            df["opp_goals_pred"].values,
            df["team_goals_true"].values,
            df["opp_goals_true"].values,
        ),
        "outcome_accuracy": outcome_accuracy(
            df["team_goals_pred"].values,
            df["opp_goals_pred"].values,
            df["team_goals_true"].values,
            df["opp_goals_true"].values,
        ),
        "exact_accuracy": exact_accuracy(
            df["team_goals_pred"].values,
            df["opp_goals_pred"].values,
            df["team_goals_true"].values,
            df["opp_goals_true"].values,
        ),
        "rows": int(len(df)),
    }


def format_pct(x):
    return f"{100.0 * x:.2f}%"


def write_outputs(config_payload, report_lines):
    OUTPUT_CONFIG.write_text(json.dumps(config_payload, indent=2), encoding="utf-8")
    OUTPUT_REPORT.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"    [OK] {OUTPUT_CONFIG.relative_to(BASE_DIR)}")
    print(f"    [OK] {OUTPUT_REPORT.relative_to(BASE_DIR)}")


# ===========================================================================
# 6. PIPELINE
# ===========================================================================
def prepare_xy(train_df, pred_df, feature_cols):
    imputer = MedianImputer().fit(train_df[feature_cols])
    x_train = imputer.transform(train_df[feature_cols])
    x_pred = imputer.transform(pred_df[feature_cols])
    return x_train, x_pred


def validation_predictions(train_df, feature_cols):
    folds = validation_splits(train_df)
    if len(folds) != 1:
        raise NotImplementedError("multi_fold skeleton is reserved; FAST_MODE uses latest_window.")

    tr, val, _ = folds[0]
    print(f"\n[2] Validation mode: {VALIDATION_MODE}")
    print(f"    Train window: {len(tr)} rows | Validation window: {len(val)} rows")

    x_tr, x_val = prepare_xy(tr, val, feature_cols)
    w_tr = tr["train_weight"].values

    print("\n[3] Fitting base models for validation predictions...")
    team_preds, names = fit_predict_models(
        x_tr, tr["team_goals"].values, w_tr, x_val, N_ESTIMATORS_BLEND, "team"
    )
    opp_preds, opp_names = fit_predict_models(
        x_tr, tr["opp_goals"].values, w_tr, x_val, N_ESTIMATORS_BLEND, "opp"
    )
    assert names == opp_names

    outcome_probs = None
    outcome_model_name = None
    outcome_info = {}
    if ENABLE_OUTCOME_CLASSIFIER:
        print("\n[3b] Fitting optional outcome classifier...")
        outcome_probs, outcome_model_name = fit_outcome_classifier(x_tr, tr, w_tr, x_val)
        y_val_out = outcome_target(val["team_goals"].values, val["opp_goals"].values)
        pred_class = outcome_probs.argmax(axis=1)
        outcome_info = {
            "model": outcome_model_name,
            "validation_accuracy": float(np.mean(pred_class == y_val_out)),
            "validation_logloss": float(log_loss(y_val_out, outcome_probs, labels=[0, 1, 2])),
            "validation_class_distribution": dict(sorted(Counter(y_val_out.tolist()).items())),
        }
        print(
            f"    Outcome classifier: {outcome_model_name}, "
            f"acc={outcome_info['validation_accuracy']:.4f}, "
            f"logloss={outcome_info['validation_logloss']:.5f}"
        )

    return tr, val, team_preds, opp_preds, names, outcome_probs, outcome_info


def choose_best_config(team_preds, opp_preds, names, val_df, outcome_probs):
    loss_tensors = {m: build_loss_tensor(m) for m in MAX_GOALS_CANDIDATES}

    optuna_config = tune_weighted_blend_optuna(team_preds, opp_preds, val_df, loss_tensors, outcome_probs)
    if optuna_config is not None:
        blend_config = optuna_config
        print("    Skipping fallback grid because Optuna is available in FAST_MODE.")
    else:
        blend_config = tune_weighted_blend_fallback(team_preds, opp_preds, val_df, loss_tensors, outcome_probs)

    stack_config, stack_model = tune_stacking(team_preds, opp_preds, names, val_df, loss_tensors, outcome_probs)

    # Latest-window stacking is trained on a relatively small meta sample. Require
    # a clear validation win before replacing the more stable weighted blend.
    stacking_margin = 0.04
    stacking_outcome_ok = stack_config.outcome_acc >= blend_config.outcome_acc - 0.005
    stacking_clear_win = stack_config.score < blend_config.score - stacking_margin

    if stacking_clear_win and stacking_outcome_ok:
        print(
            "\n[6] Selected stacking over weighted blend: "
            f"{stack_config.score:.5f} < {blend_config.score:.5f} "
            f"with guard margin {stacking_margin:.3f}"
        )
        return stack_config, stack_model, loss_tensors

    print(
        "\n[6] Selected weighted blend over stacking: "
        f"blend={blend_config.score:.5f}, stack={stack_config.score:.5f}, "
        f"guard_margin={stacking_margin:.3f}"
    )
    return blend_config, None, loss_tensors


def final_predictions(train_df, test_df, feature_cols, best_config, meta_model, names, loss_tensors):
    print("\n[7] Training final static models on full train...")
    x_train, x_test = prepare_xy(train_df, test_df, feature_cols)
    w_train = train_df["train_weight"].values

    team_preds, full_names = fit_predict_models(
        x_train, train_df["team_goals"].values, w_train, x_test, N_ESTIMATORS_FULL, "team_final"
    )
    opp_preds, opp_names = fit_predict_models(
        x_train, train_df["opp_goals"].values, w_train, x_test, N_ESTIMATORS_FULL, "opp_final"
    )
    assert full_names == opp_names == names

    outcome_probs = None
    if ENABLE_OUTCOME_CLASSIFIER and best_config.outcome_blend_alpha > 0.0:
        print("\n[7b] Training final outcome classifier...")
        outcome_probs, _ = fit_outcome_classifier(x_train, train_df, w_train, x_test)

    if best_config.mode == "stacking":
        meta_x_test = build_meta_features(team_preds, opp_preds, names, test_df)
        lambda_team, lambda_opp = meta_model.predict(meta_x_test)
    else:
        lambda_team, lambda_opp = apply_weighted_blend(team_preds, opp_preds, best_config)

    pred_team, pred_opp = erm_predict_batch(
        lambda_team,
        lambda_opp,
        loss_tensors[best_config.max_goals],
        draw_boost=best_config.draw_boost,
        low_score_boost=best_config.low_score_boost,
        outcome_probs=outcome_probs,
        outcome_alpha=best_config.outcome_blend_alpha,
    )
    return pred_team, pred_opp, lambda_team, lambda_opp


def write_submission(test_df, pred_team, pred_opp):
    sample_sub = pd.read_csv(SAMPLE_SUB)
    sub = pd.DataFrame(
        {
            "Id": test_df["Id"].values,
            "team_goals": pred_team.astype(int),
            "opp_goals": pred_opp.astype(int),
        }
    )
    sub = sample_sub[["Id"]].merge(sub, on="Id", how="left")
    sub["team_goals"] = sub["team_goals"].astype(int)
    sub["opp_goals"] = sub["opp_goals"].astype(int)
    sub.to_csv(OUTPUT_SUB, index=False)
    print(f"    [OK] {OUTPUT_SUB.relative_to(BASE_DIR)} -> {len(sub)} rows")
    return sub


def apply_static_submission_consensus(test_df, pred_team, pred_opp):
    info = {
        "enabled": USE_STATIC_SUBMISSION_CONSENSUS,
        "applied": False,
        "rule": "Use V4 when V3 and V4 agree on W/D/L outcome; otherwise use V3.",
        "reason": "V3 is the outcome-stability prior, V4 is the exact-score/AW-MAE prior.",
    }
    if not USE_STATIC_SUBMISSION_CONSENSUS:
        return pred_team, pred_opp, info
    if not (V3_SUB.exists() and V4_SUB.exists()):
        info["reason"] = "Required static submissions are missing."
        return pred_team, pred_opp, info

    base = pd.DataFrame({"Id": test_df["Id"].values})
    v3 = pd.read_csv(V3_SUB).rename(
        columns={"team_goals": "team_goals_v3", "opp_goals": "opp_goals_v3"}
    )
    v4 = pd.read_csv(V4_SUB).rename(
        columns={"team_goals": "team_goals_v4", "opp_goals": "opp_goals_v4"}
    )
    merged = base.merge(v3, on="Id", how="left").merge(v4, on="Id", how="left")
    needed = ["team_goals_v3", "opp_goals_v3", "team_goals_v4", "opp_goals_v4"]
    if merged[needed].isna().any().any():
        info["reason"] = "Static submissions do not cover all test IDs."
        return pred_team, pred_opp, info

    out3 = np.sign(merged["team_goals_v3"].values - merged["opp_goals_v3"].values)
    out4 = np.sign(merged["team_goals_v4"].values - merged["opp_goals_v4"].values)
    agree = out3 == out4
    final_team = np.where(agree, merged["team_goals_v4"].values, merged["team_goals_v3"].values)
    final_opp = np.where(agree, merged["opp_goals_v4"].values, merged["opp_goals_v3"].values)

    info.update(
        {
            "applied": True,
            "v3_path": str(V3_SUB),
            "v4_path": str(V4_SUB),
            "rows_from_v4": int(np.sum(agree)),
            "rows_from_v3": int(np.sum(~agree)),
            "agreement_rate": float(np.mean(agree)),
        }
    )
    return final_team.astype(int), final_opp.astype(int), info


def main():
    print("=" * 68)
    print("STATIC STACKED PIPELINE V5 - Gammafest Masa Kite Lagi")
    print("=" * 68)
    t0 = time.time()

    train_df, test_df, feature_cols, dropped_cols = load_data()
    tr_window, val_df, val_team_preds, val_opp_preds, model_names, outcome_probs, outcome_info = validation_predictions(
        train_df, feature_cols
    )
    best_config, meta_model, loss_tensors = choose_best_config(
        val_team_preds, val_opp_preds, model_names, val_df, outcome_probs
    )

    if best_config.mode == "stacking":
        val_lambda_t, val_lambda_o = meta_model.predict(
            build_meta_features(val_team_preds, val_opp_preds, model_names, val_df)
        )
    else:
        val_lambda_t, val_lambda_o = apply_weighted_blend(val_team_preds, val_opp_preds, best_config)

    val_pred_t, val_pred_o = erm_predict_batch(
        val_lambda_t,
        val_lambda_o,
        loss_tensors[best_config.max_goals],
        draw_boost=best_config.draw_boost,
        low_score_boost=best_config.low_score_boost,
        outcome_probs=outcome_probs if best_config.outcome_blend_alpha > 0.0 else None,
        outcome_alpha=best_config.outcome_blend_alpha,
    )
    val_summary = summarize_predictions(
        val_pred_t,
        val_pred_o,
        val_df["team_goals"].values,
        val_df["opp_goals"].values,
        val_lambda_t,
        val_lambda_o,
    )

    pred_team, pred_opp, lambda_team, lambda_opp = final_predictions(
        train_df, test_df, feature_cols, best_config, meta_model, model_names, loss_tensors
    )
    pred_team, pred_opp, static_consensus_info = apply_static_submission_consensus(
        test_df, pred_team, pred_opp
    )
    if static_consensus_info["applied"]:
        print(
            "\n[7c] Applied static submission consensus blend: "
            f"V4 rows={static_consensus_info['rows_from_v4']}, "
            f"V3 rows={static_consensus_info['rows_from_v3']}"
        )
    write_submission(test_df, pred_team, pred_opp)
    test_summary = summarize_predictions(pred_team, pred_opp, lambda_team=lambda_team, lambda_opp=lambda_opp)

    print("\n[8] Final local reporting only...")
    v5_local = evaluate_submission_local(OUTPUT_SUB, GT_PATH)
    v4_local = evaluate_submission_local(V4_SUB, GT_PATH) if V4_SUB.exists() else None

    config_payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "seed": SEED,
        "fast_mode": FAST_MODE,
        "feature_profile": FEATURE_PROFILE,
        "static_submission_consensus": static_consensus_info,
        "validation_mode": VALIDATION_MODE,
        "model_names": model_names,
        "model_availability": {
            "xgb": True,
            "hgb": True,
            "lgb": lgb is not None,
            "catboost": CatBoostRegressor is not None,
            "optuna": optuna is not None,
        },
        "ensemble_mode": best_config.mode,
        "best_optuna_params": best_config.best_params,
        "team_weights": list(best_config.team_weights),
        "opp_weights": list(best_config.opp_weights),
        "meta_model_type": best_config.meta_model_type,
        "team_scale": best_config.team_scale,
        "opp_scale": best_config.opp_scale,
        "team_bias": best_config.team_bias,
        "opp_bias": best_config.opp_bias,
        "draw_boost": best_config.draw_boost,
        "low_score_boost": best_config.low_score_boost,
        "outcome_blend_alpha": best_config.outcome_blend_alpha,
        "max_goals": best_config.max_goals,
        "validation_weighted_awmae": best_config.score,
        "validation_outcome_accuracy": best_config.outcome_acc,
        "validation_exact_accuracy": best_config.exact_acc,
        "validation_goal_difference_accuracy": best_config.gd_acc,
        "train_rows": int(len(tr_window)),
        "validation_rows": int(len(val_df)),
        "full_train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "feature_count": len(feature_cols),
        "feature_columns": feature_cols,
        "dropped_non_numeric_columns": dropped_cols,
        "outcome_classifier": outcome_info,
        "validation_prediction_summary": val_summary,
        "test_prediction_summary": test_summary,
        "local_reporting_only": {"v4": v4_local, "v5": v5_local},
        "runtime_minutes": (time.time() - t0) / 60.0,
    }

    report_lines = []
    report_lines.append("V5 Static Pipeline Validation Report")
    report_lines.append("=" * 40)
    report_lines.append(f"Mode: {best_config.mode}")
    report_lines.append(f"Models: {', '.join(model_names)}")
    report_lines.append(f"Validation weighted AW-MAE: {best_config.score:.5f}")
    report_lines.append(f"Outcome accuracy: {format_pct(best_config.outcome_acc)}")
    report_lines.append(f"Exact score accuracy: {format_pct(best_config.exact_acc)}")
    report_lines.append(f"Goal-difference accuracy: {format_pct(best_config.gd_acc)}")
    report_lines.append(f"Max goals candidates selected: 0-{best_config.max_goals - 1}")
    report_lines.append(f"Draw boost: {best_config.draw_boost}")
    report_lines.append(f"Low-score boost: {best_config.low_score_boost}")
    report_lines.append(f"Outcome blend alpha: {best_config.outcome_blend_alpha}")
    report_lines.append(f"Static submission consensus applied: {static_consensus_info['applied']}")
    if static_consensus_info["applied"]:
        report_lines.append(
            "Consensus rule: "
            f"{static_consensus_info['rows_from_v4']} rows from V4, "
            f"{static_consensus_info['rows_from_v3']} rows from V3"
        )
    report_lines.append("")
    report_lines.append("Validation Prediction Distribution")
    report_lines.append(json.dumps(val_summary, indent=2))
    report_lines.append("")
    report_lines.append("Final Test Prediction Distribution")
    report_lines.append(json.dumps(test_summary, indent=2))
    if test_summary["pct_predictions_score_ge_5"] > 0.03:
        report_lines.append("WARNING: More than 3% of final predictions have at least one side >= 5.")
    report_lines.append("")
    report_lines.append("Local Reporting Only")
    if v4_local is not None:
        report_lines.append(f"V4 local AW-MAE: {v4_local['awmae']:.5f}")
    if v5_local is not None:
        report_lines.append(f"V5 local AW-MAE: {v5_local['awmae']:.5f}")
        report_lines.append(f"V5 local outcome accuracy: {format_pct(v5_local['outcome_accuracy'])}")
    if v4_local is not None and v5_local is not None:
        report_lines.append(f"Delta V5 - V4: {v5_local['awmae'] - v4_local['awmae']:+.5f}")
    report_lines.append("")
    report_lines.append("Experiment Table")
    report_lines.append("V4 baseline: static XGB/HGB blend result available as submission_v4.csv.")
    report_lines.append(f"V5 CatBoost added: {'active' if CatBoostRegressor is not None else 'not installed, fallback used'}.")
    report_lines.append("V5 split team/opp blend: implemented and evaluated.")
    report_lines.append(f"V5 Optuna AW-MAE objective: {'active' if optuna is not None else 'not installed, fallback grid used'}.")
    report_lines.append("V5 stacking meta-model: implemented and compared against weighted blend.")
    report_lines.append("V5 draw/low-score calibration: implemented and tuned on validation.")
    report_lines.append(f"V5 outcome classifier: {'active' if ENABLE_OUTCOME_CLASSIFIER else 'disabled'}; alpha selected by validation.")
    report_lines.append(f"V5 custom objective: {'active' if ENABLE_PSEUDOHUBER_MODEL else 'implemented as optional, disabled for this run'}.")

    print("\n[9] Writing config and report...")
    write_outputs(config_payload, report_lines)

    print("\n[10] evaluate_local.py output...")
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                f"sys.path.insert(0, {str(BASE_DIR / 'src')!r}); "
                "from evaluate_local import evaluate_submission; "
                f"evaluate_submission({str(OUTPUT_SUB)!r}, {str(GT_PATH)!r})"
            ),
        ],
        check=False,
    )

    print(f"\nDone in {(time.time() - t0) / 60:.1f} minutes.")


if __name__ == "__main__":
    main()
