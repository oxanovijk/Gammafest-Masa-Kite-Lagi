"""
ML Pipeline V8 Anchor Safe -- Gammafest Masa Kite Lagi
=========================================
V5-centered bounded selective correction.

Layer 1: feature/state construction
  Uses full-era frozen features from train_final.csv/test_final.csv. Elo, form,
  H2H, and context features are never updated on the test period.

Layer 2: supervised mapping to score
  Trains recent-era supervised models only to produce a small, bounded
  correction around the V5/static anchor. If the correction is not clearly
  better on train-only time validation, the final submission falls back to V5.

Outputs:
  dataset/submission_v8_anchor_safe.csv
  dataset/submission_v8_anchor_safe_config.json
  dataset/submission_v8_anchor_safe_validation_report.txt
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import warnings
from collections import Counter
from dataclasses import asdict, dataclass
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

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover - optional local dependency
    lgb = None

try:
    from catboost import CatBoostRegressor
except ImportError:  # pragma: no cover - optional local dependency
    CatBoostRegressor = None

warnings.filterwarnings("ignore")


# ===========================================================================
# 1. CONFIG
# ===========================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "dataset"
CACHE_DIR = DATA_DIR / "cache_v6"

TRAIN_FINAL = DATA_DIR / "train_final.csv"
TEST_FINAL = DATA_DIR / "test_final.csv"
TRAIN_RAW = DATA_DIR / "train.csv"
TEST_RAW = DATA_DIR / "test.csv"
SAMPLE_SUB = DATA_DIR / "sample submission.csv"
GT_PATH = DATA_DIR / "test_ground_truth.csv"
V5_SUB = DATA_DIR / "submission_v5.csv"

OUTPUT_SUB = DATA_DIR / "submission_v8_anchor_safe.csv"
OUTPUT_CONFIG = DATA_DIR / "submission_v8_anchor_safe_config.json"
OUTPUT_REPORT = DATA_DIR / "submission_v8_anchor_safe_validation_report.txt"

SEED = 42
FAST_MODE = True
USE_CACHE = True
CACHE_VERSION = "v6_v5_centered_bounded_selective_correction_2026_04_30_a"

STRATEGY_NAME = "v5_centered_bounded_selective_correction"
AWMAE_POWER = 1.5
LEGACY_LOCAL_POWER = 1.3
MAX_LAMBDA = 12.0
MIN_LAMBDA = 1e-5

N_ROUNDS_FAST = 160
N_ROUNDS_FULL = 420
N_ROUNDS_ANCHOR_FAST = 190
N_ROUNDS_ANCHOR_FULL = 520

ENABLE_LGB = False
ENABLE_CATBOOST = False
ACTIVE_CUTOFFS = ["1990", "2000"]
OPTIONAL_CUTOFFS = ["2002"]

RECENT_PRESETS = [
    {"1990": 0.70, "2000": 0.30, "2002": 0.00},
    {"1990": 0.60, "2000": 0.30, "2002": 0.10},
    {"1990": 0.50, "2000": 0.40, "2002": 0.10},
    {"1990": 0.80, "2000": 0.20, "2002": 0.00},
]

FOLDS = [
    {
        "name": "fold_2003_2005",
        "train_end": "2002-12-31",
        "valid_start": "2003-01-01",
        "valid_end": "2005-12-31",
        "weight": 0.20,
    },
    {
        "name": "fold_2006_2008",
        "train_end": "2005-12-31",
        "valid_start": "2006-01-01",
        "valid_end": "2008-12-31",
        "weight": 0.30,
    },
    {
        "name": "fold_2009_2011",
        "train_end": "2008-12-31",
        "valid_start": "2009-01-01",
        "valid_end": "2011-08-04",
        "weight": 0.50,
    },
]

FAST_TUNING_GRID = {
    "beta": [0.15, 0.25, 0.35],
    "cap": [0.15, 0.25, 0.35],
    "anchor_offset": [0.10, 0.20, 0.30],
    "gate_threshold": [0.00, 0.45],
    "max_goals": [7, 8, 10],
    "draw_boost": [1.00, 1.06],
    "low_score_boost": [1.00, 1.04],
}

FULL_TUNING_GRID = {
    "beta": [0.10, 0.15, 0.25, 0.35],
    "cap": [0.15, 0.25, 0.35, 0.45],
    "anchor_offset": [0.05, 0.10, 0.20, 0.30],
    "gate_threshold": [0.00, 0.42, 0.45, 0.50],
    "max_goals": [7, 8, 10],
    "draw_boost": [0.98, 1.00, 1.06, 1.12],
    "low_score_boost": [0.98, 1.00, 1.04, 1.08],
}

LOW_SCORE_CELLS = [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (0, 2), (2, 1), (1, 2)]
COMMON_LOW_SCORES = [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (0, 2), (2, 1), (1, 2)]
LOSS_TENSOR_CACHE: dict[int, np.ndarray] = {}

MIN_AWMAE_IMPROVEMENT = 0.001
MAX_OUTCOME_DROP = 0.003
MAX_EXACT_DROP = 0.005
MAX_COMMON_LOW_SCORE_SHARE_INCREASE = 0.03
MAX_TOP3_SCORE_SHARE_INCREASE = 0.03
MAX_DRAW_SHARE_SHIFT = 0.03
MAX_AVG_TOTAL_GOALS_DROP = 0.10

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
    "max_depth": 6,
    "learning_rate": 0.045,
    "min_child_weight": 85,
    "alpha": 1.9,
    "lambda": 5.0,
    "subsample": 0.78,
    "colsample_bytree": 0.82,
    "tree_method": "hist",
    "seed": SEED,
    "nthread": 1,
}

LGB_POISSON_PARAMS = {
    "objective": "poisson",
    "metric": "poisson",
    "num_leaves": 48,
    "learning_rate": 0.025,
    "min_child_samples": 90,
    "reg_alpha": 3.0,
    "reg_lambda": 2.0,
    "subsample": 0.72,
    "colsample_bytree": 0.78,
    "verbose": -1,
    "n_jobs": 1,
    "seed": SEED,
}


@dataclass(frozen=True)
class CandidateConfig:
    preset: dict[str, float]
    beta_team: float
    beta_opp: float
    cap_team: float
    cap_opp: float
    anchor_offset: float
    gate_threshold: float
    max_goals: int
    draw_boost: float
    low_score_boost: float


@dataclass
class CandidateResult:
    config: CandidateConfig
    weighted_awmae: float
    unweighted_awmae: float
    outcome_accuracy: float
    exact_accuracy: float
    goal_diff_accuracy: float
    fold_scores: list[dict]
    distribution: dict
    anchor_metrics: dict
    recent_only_metrics: dict
    selection_score: float
    score_collapse_penalty: float
    stability_penalty: float
    safety_passed: bool
    safety_reasons: list[str]


# ===========================================================================
# 2. AW-MAE HELPERS
# ===========================================================================
def awmae_loss_array(pred_t, pred_o, true_t, true_o, power=AWMAE_POWER) -> np.ndarray:
    pred_t = np.asarray(pred_t, dtype=float)
    pred_o = np.asarray(pred_o, dtype=float)
    true_t = np.asarray(true_t, dtype=float)
    true_o = np.asarray(true_o, dtype=float)
    mae = (np.abs(pred_t - true_t) + np.abs(pred_o - true_o)) / 2.0
    exact = ((pred_t == true_t) & (pred_o == true_o)).astype(float)
    pred_out = np.sign(pred_t - pred_o)
    true_out = np.sign(true_t - true_o)
    outcome = (pred_out == true_out).astype(float)
    gd = ((pred_t - pred_o) == (true_t - true_o)).astype(float)
    penalty = 0.30 * (1.0 - exact) + 0.25 * (1.0 - outcome) + 0.15 * (1.0 - gd)
    multiplier = np.where(outcome == 1.0, 1.0, 1.5)
    return ((mae + penalty) * multiplier) ** power


def mean_awmae(pred_t, pred_o, true_t, true_o, weights=None, power=AWMAE_POWER) -> float:
    losses = awmae_loss_array(pred_t, pred_o, true_t, true_o, power=power)
    if weights is None:
        return float(np.mean(losses))
    weights = np.asarray(weights, dtype=float)
    return float(np.average(losses, weights=weights))


def outcome_accuracy(pred_t, pred_o, true_t, true_o) -> float:
    return float(np.mean(np.sign(np.asarray(pred_t) - np.asarray(pred_o)) == np.sign(np.asarray(true_t) - np.asarray(true_o))))


def exact_accuracy(pred_t, pred_o, true_t, true_o) -> float:
    return float(np.mean((np.asarray(pred_t) == np.asarray(true_t)) & (np.asarray(pred_o) == np.asarray(true_o))))


def goal_diff_accuracy(pred_t, pred_o, true_t, true_o) -> float:
    return float(np.mean((np.asarray(pred_t) - np.asarray(pred_o)) == (np.asarray(true_t) - np.asarray(true_o))))


def metrics_dict(pred_t, pred_o, true_t, true_o, weights=None, power=AWMAE_POWER) -> dict:
    return {
        "weighted_awmae": mean_awmae(pred_t, pred_o, true_t, true_o, weights=weights, power=power),
        "unweighted_awmae": mean_awmae(pred_t, pred_o, true_t, true_o, weights=None, power=power),
        "outcome_accuracy": outcome_accuracy(pred_t, pred_o, true_t, true_o),
        "exact_accuracy": exact_accuracy(pred_t, pred_o, true_t, true_o),
        "goal_diff_accuracy": goal_diff_accuracy(pred_t, pred_o, true_t, true_o),
    }


def build_loss_tensor(max_goals: int, power=AWMAE_POWER) -> np.ndarray:
    tensor = np.zeros((max_goals, max_goals, max_goals, max_goals), dtype=np.float32)
    for pred_t in range(max_goals):
        for pred_o in range(max_goals):
            for true_t in range(max_goals):
                for true_o in range(max_goals):
                    tensor[pred_t, pred_o, true_t, true_o] = awmae_loss_array(
                        [pred_t], [pred_o], [true_t], [true_o], power=power
                    )[0]
    return tensor


def get_loss_tensor(max_goals: int) -> np.ndarray:
    if max_goals not in LOSS_TENSOR_CACHE:
        LOSS_TENSOR_CACHE[max_goals] = build_loss_tensor(max_goals, power=AWMAE_POWER)
    return LOSS_TENSOR_CACHE[max_goals]


# ===========================================================================
# 3. DATA LOADING AND FEATURES
# ===========================================================================
def raw_usecols(path: Path) -> list[str]:
    available = pd.read_csv(path, nrows=0).columns.tolist()
    wanted = ["Id", "date", "tournament", "neutral", "rank_diff", "rank_team", "rank_opponent"]
    return [c for c in wanted if c in available]


def add_static_interactions(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "tournament_weight" not in out:
        out["tournament_weight"] = DEFAULT_TOURNAMENT_WEIGHT

    if {"elo_team_feat", "elo_opponent_feat"}.issubset(out.columns):
        out["elo_diff"] = out["elo_team_feat"] - out["elo_opponent_feat"]
    elif "elo_diff_feat" in out.columns:
        out["elo_diff"] = out["elo_diff_feat"]

    if "elo_diff" in out.columns:
        out["abs_elo_diff"] = out["elo_diff"].abs()
        out["is_balanced_match"] = (out["abs_elo_diff"] <= 50).astype(int)
        out["is_team_favorite"] = (out["elo_diff"] > 100).astype(int)
        out["is_team_strong_favorite"] = (out["elo_diff"] > 200).astype(int)
        out["is_team_underdog"] = (out["elo_diff"] < -100).astype(int)
        out["elo_diff_x_tournament_weight"] = out["elo_diff"] * out["tournament_weight"]

    if {"rank_team", "rank_opponent"}.issubset(out.columns):
        out["rank_diff"] = out["rank_opponent"] - out["rank_team"]
    if "rank_diff" in out.columns:
        out["abs_rank_diff"] = out["rank_diff"].abs()
        out["rank_diff_x_tournament_weight"] = out["rank_diff"] * out["tournament_weight"]

    if "neutral" in out.columns and "elo_diff" in out.columns:
        out["neutral_x_elo_diff"] = out["neutral"].fillna(0) * out["elo_diff"]

    if "date" in out.columns:
        dt = pd.to_datetime(out["date"])
        out["year"] = dt.dt.year
        out["month"] = dt.dt.month

    if "tournament" in out.columns:
        out["is_friendly"] = (out["tournament"] == "Friendly").astype(int)
    out["is_major_tournament"] = (out["tournament_weight"] >= 1.50).astype(int)
    return out


def load_data():
    print("[1] Loading data...")
    train = pd.read_csv(TRAIN_FINAL)
    test = pd.read_csv(TEST_FINAL)
    raw_train = pd.read_csv(TRAIN_RAW, usecols=raw_usecols(TRAIN_RAW))
    raw_test = pd.read_csv(TEST_RAW, usecols=raw_usecols(TEST_RAW))

    train = train.merge(raw_train, on="Id", how="left")
    test = test.merge(raw_test, on="Id", how="left")
    train["date"] = pd.to_datetime(train["date"])
    test["date"] = pd.to_datetime(test["date"])
    train["tournament_weight"] = train["tournament"].map(TOURNAMENT_WEIGHT_MAP).fillna(DEFAULT_TOURNAMENT_WEIGHT)
    test["tournament_weight"] = test["tournament"].map(TOURNAMENT_WEIGHT_MAP).fillna(DEFAULT_TOURNAMENT_WEIGHT)
    train["train_weight"] = train["tournament_weight"]
    train["metric_weight"] = train["tournament_weight"]
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
    feature_cols = [c for c in train.columns if c not in exclude and c in test.columns]
    numeric_cols = []
    dropped = []
    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(train[col]) and pd.api.types.is_numeric_dtype(test[col]):
            numeric_cols.append(col)
        else:
            dropped.append(col)
    feature_cols = numeric_cols
    train[feature_cols] = train[feature_cols].replace([np.inf, -np.inf], np.nan)
    test[feature_cols] = test[feature_cols].replace([np.inf, -np.inf], np.nan)

    print(f"    Train: {train.shape} | Test: {test.shape} | Features: {len(feature_cols)}")
    if dropped:
        print(f"    Dropped non-numeric features: {dropped}")
    return train, test, feature_cols, dropped


def cutoff_mask(df: pd.DataFrame, cutoff: str, train_end: str | None = None) -> np.ndarray:
    mask = np.ones(len(df), dtype=bool)
    if cutoff != "full":
        mask &= df["date"] >= pd.Timestamp(f"{cutoff}-01-01")
    if train_end is not None:
        mask &= df["date"] <= pd.Timestamp(train_end)
    return mask


# ===========================================================================
# 4. MODEL WRAPPERS
# ===========================================================================
class XGBPoissonModel:
    name = "xgb_poisson"

    def __init__(self, rounds: int):
        self.rounds = rounds
        self.model = None

    def fit(self, x, y, weight):
        dtrain = xgb.DMatrix(x, label=y, weight=weight)
        self.model = xgb.train(XGB_POISSON_PARAMS, dtrain, num_boost_round=self.rounds, verbose_eval=False)
        return self

    def predict(self, x):
        return self.model.predict(xgb.DMatrix(x))


class HGBPoissonModel:
    name = "hgb_poisson"

    def __init__(self, rounds: int):
        self.model = HistGradientBoostingRegressor(
            loss="poisson",
            learning_rate=0.035,
            max_iter=rounds,
            max_leaf_nodes=31,
            min_samples_leaf=65,
            l2_regularization=2.0,
            random_state=SEED,
            early_stopping=False,
        )

    def fit(self, x, y, weight):
        self.model.fit(x, y, sample_weight=weight)
        return self

    def predict(self, x):
        return self.model.predict(x)


class LGBPoissonModel:
    name = "lgb_poisson"

    def __init__(self, rounds: int):
        self.rounds = rounds
        self.model = None

    def fit(self, x, y, weight):
        dtrain = lgb.Dataset(x, y, weight=weight, free_raw_data=False)
        self.model = lgb.train(LGB_POISSON_PARAMS, dtrain, num_boost_round=self.rounds)
        return self

    def predict(self, x):
        return self.model.predict(x)


class CatPoissonModel:
    name = "cat_poisson"

    def __init__(self, rounds: int):
        self.model = CatBoostRegressor(
            loss_function="Poisson",
            iterations=rounds,
            depth=6,
            learning_rate=0.035,
            l2_leaf_reg=8.0,
            random_seed=SEED,
            verbose=False,
            allow_writing_files=False,
        )

    def fit(self, x, y, weight):
        self.model.fit(x, y, sample_weight=weight)
        return self

    def predict(self, x):
        return self.model.predict(x)


def model_factories(rounds: int, include_optional=True):
    factories = [lambda: XGBPoissonModel(rounds), lambda: HGBPoissonModel(rounds)]
    if include_optional and ENABLE_LGB and lgb is not None:
        factories.append(lambda: LGBPoissonModel(rounds))
    if include_optional and ENABLE_CATBOOST and CatBoostRegressor is not None:
        factories.append(lambda: CatPoissonModel(rounds))
    return factories


def cache_key(parts: dict) -> str:
    raw = json.dumps(parts, sort_keys=True, default=str).encode("utf-8")
    return hashlib.md5(raw).hexdigest()


def fit_predict_models(x_train, y_train, w_train, x_pred, rounds: int, tag: dict, include_optional=True):
    preds = []
    names = []
    for factory in model_factories(rounds, include_optional=include_optional):
        model = factory()
        key = cache_key(
            {
                "version": CACHE_VERSION,
                "model": model.name,
                "target_tag": tag,
                "rounds": rounds,
                "params": XGB_POISSON_PARAMS if model.name == "xgb_poisson" else model.name,
            }
        )
        path = CACHE_DIR / f"{key}.npz"
        if USE_CACHE and path.exists():
            pred = np.load(path)["pred"]
        else:
            print(f"      - fitting {model.name} [{tag.get('scope', '')} {tag.get('target', '')}]")
            model.fit(x_train, y_train, w_train)
            pred = np.clip(model.predict(x_pred), MIN_LAMBDA, MAX_LAMBDA)
            if USE_CACHE:
                CACHE_DIR.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(path, pred=pred)
        preds.append(np.clip(pred, MIN_LAMBDA, MAX_LAMBDA))
        names.append(model.name)
    return np.vstack(preds), names


# ===========================================================================
# 5. PROBABILITY, ERM, AND DIAGNOSTICS
# ===========================================================================
def poisson_prob_matrix(lambda_team, lambda_opp, max_goals: int, draw_boost=1.0, low_score_boost=1.0) -> np.ndarray:
    lambda_team = np.clip(np.asarray(lambda_team, dtype=float), 1e-6, 15.0)
    lambda_opp = np.clip(np.asarray(lambda_opp, dtype=float), 1e-6, 15.0)
    k = np.arange(max_goals)
    pmf_t = poisson.pmf(k[None, :], lambda_team[:, None])
    pmf_o = poisson.pmf(k[None, :], lambda_opp[:, None])
    pmf_t = pmf_t / np.maximum(pmf_t.sum(axis=1, keepdims=True), 1e-12)
    pmf_o = pmf_o / np.maximum(pmf_o.sum(axis=1, keepdims=True), 1e-12)
    prob = pmf_t[:, :, None] * pmf_o[:, None, :]
    if draw_boost != 1.0:
        idx = np.arange(max_goals)
        prob[:, idx, idx] *= draw_boost
    if low_score_boost != 1.0:
        for a, b in LOW_SCORE_CELLS:
            if a < max_goals and b < max_goals:
                prob[:, a, b] *= low_score_boost
    prob = prob / np.maximum(prob.sum(axis=(1, 2), keepdims=True), 1e-12)
    return prob


def erm_from_prob(prob: np.ndarray, loss_tensor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    expected_loss = np.einsum("abij,nij->nab", loss_tensor, prob)
    n = expected_loss.shape[0]
    max_goals = expected_loss.shape[1]
    flat = expected_loss.reshape(n, -1).argmin(axis=1)
    return (flat // max_goals).astype(int), (flat % max_goals).astype(int)


def predict_from_lambdas(lambda_team, lambda_opp, max_goals: int, draw_boost=1.0, low_score_boost=1.0):
    prob = poisson_prob_matrix(lambda_team, lambda_opp, max_goals, draw_boost, low_score_boost)
    loss_tensor = get_loss_tensor(max_goals)
    return erm_from_prob(prob, loss_tensor)


def outcome_probs_from_lambdas(lambda_team, lambda_opp, max_goals: int = 8) -> np.ndarray:
    prob = poisson_prob_matrix(lambda_team, lambda_opp, max_goals=max_goals)
    loss = np.triu(prob, k=1).sum(axis=(1, 2))
    draw = np.array([np.trace(p) for p in prob])
    win = np.tril(prob, k=-1).sum(axis=(1, 2))
    return np.vstack([loss, draw, win]).T


def score_distribution(pred_t, pred_o) -> dict:
    pred_t = np.asarray(pred_t, dtype=int)
    pred_o = np.asarray(pred_o, dtype=int)
    n = len(pred_t)
    scores = list(zip(pred_t, pred_o))
    counts = Counter(scores)
    top = counts.most_common(10)
    common = sum(counts.get(s, 0) for s in COMMON_LOW_SCORES) / max(n, 1)
    draw = float(np.mean(pred_t == pred_o))
    win = float(np.mean(pred_t > pred_o))
    loss = float(np.mean(pred_t < pred_o))
    total_goals = pred_t + pred_o
    top_counts = [c for _, c in counts.most_common(5)] + [0, 0, 0, 0, 0]
    return {
        "win_share": win,
        "draw_share": draw,
        "loss_share": loss,
        "avg_team_goals": float(np.mean(pred_t)),
        "avg_opp_goals": float(np.mean(pred_o)),
        "avg_total_goals": float(np.mean(total_goals)),
        "score_ge_5_share": float(np.mean((pred_t >= 5) | (pred_o >= 5))),
        "common_low_score_share": float(common),
        "top_1_score_share": float(top_counts[0] / max(n, 1)),
        "top_3_score_share": float(sum(top_counts[:3]) / max(n, 1)),
        "top_5_score_share": float(sum(top_counts[:5]) / max(n, 1)),
        "top_10_scores": [{"score": f"{a}-{b}", "count": int(c), "share": float(c / max(n, 1))} for (a, b), c in top],
    }


def score_collapse_penalty(anchor_dist: dict, hybrid_dist: dict) -> float:
    penalty = 0.0
    common_inc = hybrid_dist["common_low_score_share"] - anchor_dist["common_low_score_share"]
    top3_inc = hybrid_dist["top_3_score_share"] - anchor_dist["top_3_score_share"]
    draw_shift = abs(hybrid_dist["draw_share"] - anchor_dist["draw_share"])
    goals_drop = anchor_dist["avg_total_goals"] - hybrid_dist["avg_total_goals"]
    if common_inc > MAX_COMMON_LOW_SCORE_SHARE_INCREASE:
        penalty += 0.10 * (common_inc - MAX_COMMON_LOW_SCORE_SHARE_INCREASE)
    if top3_inc > MAX_TOP3_SCORE_SHARE_INCREASE:
        penalty += 0.10 * (top3_inc - MAX_TOP3_SCORE_SHARE_INCREASE)
    if draw_shift > MAX_DRAW_SHARE_SHIFT:
        penalty += 0.05 * (draw_shift - MAX_DRAW_SHARE_SHIFT)
    if goals_drop > MAX_AVG_TOTAL_GOALS_DROP:
        penalty += 0.05 * (goals_drop - MAX_AVG_TOTAL_GOALS_DROP)
    return float(penalty)


# ===========================================================================
# 6. ANCHOR AND RECENT EXPERTS
# ===========================================================================
def build_anchor_for_fold(train_df, val_df, feature_cols, fold_name: str, rounds: int):
    mask = np.ones(len(train_df), dtype=bool)
    x_train = train_df.loc[mask, feature_cols]
    w_train = train_df.loc[mask, "train_weight"].values
    x_val = val_df[feature_cols]
    tag_base = {"scope": "anchor", "fold": fold_name}
    team_preds, names = fit_predict_models(
        x_train,
        train_df.loc[mask, "team_goals"].values,
        w_train,
        x_val,
        rounds,
        {**tag_base, "target": "team"},
        include_optional=False,
    )
    opp_preds, opp_names = fit_predict_models(
        x_train,
        train_df.loc[mask, "opp_goals"].values,
        w_train,
        x_val,
        rounds,
        {**tag_base, "target": "opp"},
        include_optional=False,
    )
    assert names == opp_names
    lambda_team = np.mean(team_preds, axis=0)
    lambda_opp = np.mean(opp_preds, axis=0)
    pred_t, pred_o = predict_from_lambdas(lambda_team, lambda_opp, max_goals=8, draw_boost=1.0, low_score_boost=1.0)
    return {
        "lambda_team": lambda_team,
        "lambda_opp": lambda_opp,
        "pred_team": pred_t,
        "pred_opp": pred_o,
        "model_names": names,
    }


def build_recent_cutoff_for_fold(train_df, val_df, feature_cols, cutoff: str, fold_name: str, train_end: str, rounds: int):
    mask = cutoff_mask(train_df, cutoff, train_end=train_end)
    x_train = train_df.loc[mask, feature_cols]
    y_team = train_df.loc[mask, "team_goals"].values
    y_opp = train_df.loc[mask, "opp_goals"].values
    w_train = train_df.loc[mask, "train_weight"].values
    x_val = val_df[feature_cols]
    tag_base = {"scope": "recent_validation", "fold": fold_name, "cutoff": cutoff}
    team_preds, names = fit_predict_models(
        x_train, y_team, w_train, x_val, rounds, {**tag_base, "target": "team"}
    )
    opp_preds, opp_names = fit_predict_models(
        x_train, y_opp, w_train, x_val, rounds, {**tag_base, "target": "opp"}
    )
    assert names == opp_names
    return {
        "lambda_team": np.mean(team_preds, axis=0),
        "lambda_opp": np.mean(opp_preds, axis=0),
        "model_names": names,
        "rows": int(mask.sum()),
    }


def combine_recent_predictions(recent_by_cutoff: dict, preset: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    available = {k: v for k, v in preset.items() if v > 0 and k in recent_by_cutoff}
    if not available:
        first = next(iter(recent_by_cutoff.values()))
        return first["lambda_team"], first["lambda_opp"]
    total = sum(available.values())
    weights = {k: v / total for k, v in available.items()}
    team = None
    opp = None
    for cutoff, weight in weights.items():
        pred = recent_by_cutoff[cutoff]
        team = pred["lambda_team"] * weight if team is None else team + pred["lambda_team"] * weight
        opp = pred["lambda_opp"] * weight if opp is None else opp + pred["lambda_opp"] * weight
    return np.clip(team, MIN_LAMBDA, MAX_LAMBDA), np.clip(opp, MIN_LAMBDA, MAX_LAMBDA)


def correction_gate(anchor_team, anchor_opp, recent_team, recent_opp, df: pd.DataFrame, threshold: float) -> np.ndarray:
    if threshold <= 0:
        return np.ones(len(df), dtype=bool)
    probs = outcome_probs_from_lambdas(recent_team, recent_opp, max_goals=8)
    confidence = probs.max(axis=1)
    recent_outcome = np.argmax(probs, axis=1) - 1
    anchor_outcome = np.sign(anchor_team - anchor_opp).astype(int)
    agree = recent_outcome == anchor_outcome
    elo_signal = np.zeros(len(df), dtype=bool)
    if "abs_elo_diff" in df.columns:
        elo_signal = df["abs_elo_diff"].fillna(0).values >= 150
    major_signal = df["is_major_tournament"].fillna(0).values.astype(bool) if "is_major_tournament" in df.columns else np.zeros(len(df), dtype=bool)
    return (confidence >= threshold) & (agree | elo_signal | major_signal)


def apply_bounded_correction(
    anchor_pred_t,
    anchor_pred_o,
    recent_team,
    recent_opp,
    df: pd.DataFrame,
    config: CandidateConfig,
):
    anchor_team = np.maximum(np.asarray(anchor_pred_t, dtype=float) + config.anchor_offset, MIN_LAMBDA)
    anchor_opp = np.maximum(np.asarray(anchor_pred_o, dtype=float) + config.anchor_offset, MIN_LAMBDA)
    delta_team = np.clip(recent_team - anchor_team, -config.cap_team, config.cap_team)
    delta_opp = np.clip(recent_opp - anchor_opp, -config.cap_opp, config.cap_opp)
    gate = correction_gate(anchor_team, anchor_opp, recent_team, recent_opp, df, config.gate_threshold)
    lambda_team = anchor_team + config.beta_team * delta_team * gate.astype(float)
    lambda_opp = anchor_opp + config.beta_opp * delta_opp * gate.astype(float)
    return np.clip(lambda_team, MIN_LAMBDA, MAX_LAMBDA), np.clip(lambda_opp, MIN_LAMBDA, MAX_LAMBDA), gate


# ===========================================================================
# 7. VALIDATION AND TUNING
# ===========================================================================
def fold_data(train: pd.DataFrame, fold: dict):
    train_mask = train["date"] <= pd.Timestamp(fold["train_end"])
    val_mask = (train["date"] >= pd.Timestamp(fold["valid_start"])) & (train["date"] <= pd.Timestamp(fold["valid_end"]))
    return train.loc[train_mask].copy(), train.loc[val_mask].copy()


def build_validation_artifacts(train: pd.DataFrame, feature_cols: list[str]):
    rounds = N_ROUNDS_FAST if FAST_MODE else N_ROUNDS_FULL
    anchor_rounds = N_ROUNDS_ANCHOR_FAST if FAST_MODE else N_ROUNDS_ANCHOR_FULL
    artifacts = []
    active_cutoffs = ACTIVE_CUTOFFS + ([] if FAST_MODE else OPTIONAL_CUTOFFS)

    print("\n[2] Building train-only validation artifacts...")
    for fold in FOLDS:
        fold_train, fold_val = fold_data(train, fold)
        print(f"    {fold['name']}: train={len(fold_train)} valid={len(fold_val)}")
        anchor = build_anchor_for_fold(fold_train, fold_val, feature_cols, fold["name"], anchor_rounds)
        recent = {}
        for cutoff in active_cutoffs:
            recent[cutoff] = build_recent_cutoff_for_fold(
                fold_train, fold_val, feature_cols, cutoff, fold["name"], fold["train_end"], rounds
            )
        artifacts.append({"fold": fold, "val": fold_val, "anchor": anchor, "recent": recent})
    return artifacts


def aggregate_fold_metrics(fold_metrics: list[dict], key: str) -> float:
    weights = np.array([m["fold_weight"] for m in fold_metrics], dtype=float)
    values = np.array([m[key] for m in fold_metrics], dtype=float)
    return float(np.average(values, weights=weights))


def evaluate_candidate(config: CandidateConfig, artifacts: list[dict]) -> CandidateResult:
    all_pred_t = []
    all_pred_o = []
    all_true_t = []
    all_true_o = []
    all_weights = []
    all_anchor_t = []
    all_anchor_o = []
    all_recent_t_pred = []
    all_recent_o_pred = []
    fold_scores = []

    for artifact in artifacts:
        val = artifact["val"]
        anchor = artifact["anchor"]
        recent_team, recent_opp = combine_recent_predictions(artifact["recent"], config.preset)
        lambda_team, lambda_opp, gate = apply_bounded_correction(
            anchor["pred_team"], anchor["pred_opp"], recent_team, recent_opp, val, config
        )
        pred_t, pred_o = predict_from_lambdas(
            lambda_team, lambda_opp, config.max_goals, config.draw_boost, config.low_score_boost
        )
        pred_t = np.where(gate, pred_t, anchor["pred_team"])
        pred_o = np.where(gate, pred_o, anchor["pred_opp"])

        recent_pred_t, recent_pred_o = predict_from_lambdas(
            recent_team, recent_opp, config.max_goals, config.draw_boost, config.low_score_boost
        )

        y_t = val["team_goals"].values.astype(int)
        y_o = val["opp_goals"].values.astype(int)
        weights = val["metric_weight"].values

        fold_metric = metrics_dict(pred_t, pred_o, y_t, y_o, weights=weights)
        fold_metric["fold_name"] = artifact["fold"]["name"]
        fold_metric["fold_weight"] = artifact["fold"]["weight"]
        fold_metric["gate_share"] = float(np.mean(gate))
        fold_scores.append(fold_metric)

        all_pred_t.append(pred_t)
        all_pred_o.append(pred_o)
        all_true_t.append(y_t)
        all_true_o.append(y_o)
        all_weights.append(weights)
        all_anchor_t.append(anchor["pred_team"])
        all_anchor_o.append(anchor["pred_opp"])
        all_recent_t_pred.append(recent_pred_t)
        all_recent_o_pred.append(recent_pred_o)

    pred_t = np.concatenate(all_pred_t)
    pred_o = np.concatenate(all_pred_o)
    true_t = np.concatenate(all_true_t)
    true_o = np.concatenate(all_true_o)
    weights = np.concatenate(all_weights)
    anchor_t = np.concatenate(all_anchor_t)
    anchor_o = np.concatenate(all_anchor_o)
    recent_t_pred = np.concatenate(all_recent_t_pred)
    recent_o_pred = np.concatenate(all_recent_o_pred)

    hybrid_metrics = metrics_dict(pred_t, pred_o, true_t, true_o, weights=weights)
    anchor_metrics = metrics_dict(anchor_t, anchor_o, true_t, true_o, weights=weights)
    recent_only_metrics = metrics_dict(recent_t_pred, recent_o_pred, true_t, true_o, weights=weights)
    anchor_dist = score_distribution(anchor_t, anchor_o)
    hybrid_dist = score_distribution(pred_t, pred_o)
    collapse_penalty = score_collapse_penalty(anchor_dist, hybrid_dist)
    fold_awmaes = np.array([m["weighted_awmae"] for m in fold_scores], dtype=float)
    stability_penalty = 0.03 * float(np.std(fold_awmaes)) + 0.02 * float(np.max(fold_awmaes) - hybrid_metrics["weighted_awmae"])
    selection_score = hybrid_metrics["weighted_awmae"] + stability_penalty + collapse_penalty

    safety_reasons = []
    if anchor_metrics["weighted_awmae"] - hybrid_metrics["weighted_awmae"] < MIN_AWMAE_IMPROVEMENT:
        safety_reasons.append("weighted_awmae_improvement_too_small")
    if anchor_metrics["outcome_accuracy"] - hybrid_metrics["outcome_accuracy"] > MAX_OUTCOME_DROP:
        safety_reasons.append("outcome_drop_too_large")
    if anchor_metrics["exact_accuracy"] - hybrid_metrics["exact_accuracy"] > MAX_EXACT_DROP:
        safety_reasons.append("exact_drop_too_large")
    if hybrid_dist["common_low_score_share"] - anchor_dist["common_low_score_share"] > MAX_COMMON_LOW_SCORE_SHARE_INCREASE:
        safety_reasons.append("common_low_score_share_increase")
    if hybrid_dist["top_3_score_share"] - anchor_dist["top_3_score_share"] > MAX_TOP3_SCORE_SHARE_INCREASE:
        safety_reasons.append("top3_score_concentration_increase")
    if abs(hybrid_dist["draw_share"] - anchor_dist["draw_share"]) > MAX_DRAW_SHARE_SHIFT:
        safety_reasons.append("draw_share_shift")
    if anchor_dist["avg_total_goals"] - hybrid_dist["avg_total_goals"] > MAX_AVG_TOTAL_GOALS_DROP:
        safety_reasons.append("avg_total_goals_drop")

    return CandidateResult(
        config=config,
        weighted_awmae=hybrid_metrics["weighted_awmae"],
        unweighted_awmae=hybrid_metrics["unweighted_awmae"],
        outcome_accuracy=hybrid_metrics["outcome_accuracy"],
        exact_accuracy=hybrid_metrics["exact_accuracy"],
        goal_diff_accuracy=hybrid_metrics["goal_diff_accuracy"],
        fold_scores=fold_scores,
        distribution={"anchor": anchor_dist, "hybrid": hybrid_dist},
        anchor_metrics=anchor_metrics,
        recent_only_metrics=recent_only_metrics,
        selection_score=float(selection_score),
        score_collapse_penalty=float(collapse_penalty),
        stability_penalty=float(stability_penalty),
        safety_passed=len(safety_reasons) == 0,
        safety_reasons=safety_reasons,
    )


def candidate_grid():
    grid = FAST_TUNING_GRID if FAST_MODE else FULL_TUNING_GRID
    for preset in RECENT_PRESETS:
        active = {k: v for k, v in preset.items() if k in ACTIVE_CUTOFFS + OPTIONAL_CUTOFFS}
        if sum(active.values()) <= 0:
            continue
        total = sum(active.values())
        active = {k: v / total for k, v in active.items()}
        for beta in grid["beta"]:
            for cap in grid["cap"]:
                for offset in grid["anchor_offset"]:
                    for threshold in grid["gate_threshold"]:
                        for max_goals in grid["max_goals"]:
                            for draw_boost in grid["draw_boost"]:
                                for low_score_boost in grid["low_score_boost"]:
                                    yield CandidateConfig(
                                        preset=active,
                                        beta_team=beta,
                                        beta_opp=beta,
                                        cap_team=cap,
                                        cap_opp=cap,
                                        anchor_offset=offset,
                                        gate_threshold=threshold,
                                        max_goals=max_goals,
                                        draw_boost=draw_boost,
                                        low_score_boost=low_score_boost,
                                    )


def tune_candidates(artifacts: list[dict]) -> CandidateResult:
    print("\n[3] Tuning bounded selective correction...")
    best = None
    best_safe = None
    for idx, config in enumerate(candidate_grid(), start=1):
        result = evaluate_candidate(config, artifacts)
        if best is None or result.selection_score < best.selection_score:
            best = result
        if result.safety_passed and (best_safe is None or result.selection_score < best_safe.selection_score):
            best_safe = result
        if idx % 100 == 0:
            safe_note = "safe" if best_safe is not None else "no-safe-yet"
            print(f"    tried={idx} best={best.weighted_awmae:.5f} selection={best.selection_score:.5f} {safe_note}")
    chosen = best_safe if best_safe is not None else best
    print(
        f"    Chosen validation AW-MAE={chosen.weighted_awmae:.5f} "
        f"safe={chosen.safety_passed} reasons={chosen.safety_reasons}"
    )
    return chosen


# ===========================================================================
# 8. FINAL TRAIN/PREDICT
# ===========================================================================
def fit_recent_final(train, test, feature_cols, chosen: CandidateResult):
    rounds = N_ROUNDS_FAST if FAST_MODE else N_ROUNDS_FULL
    recent = {}
    for cutoff in chosen.config.preset.keys():
        if chosen.config.preset[cutoff] <= 0:
            continue
        mask = cutoff_mask(train, cutoff)
        tag_base = {"scope": "recent_test", "cutoff": cutoff}
        team_preds, names = fit_predict_models(
            train.loc[mask, feature_cols],
            train.loc[mask, "team_goals"].values,
            train.loc[mask, "train_weight"].values,
            test[feature_cols],
            rounds,
            {**tag_base, "target": "team"},
        )
        opp_preds, opp_names = fit_predict_models(
            train.loc[mask, feature_cols],
            train.loc[mask, "opp_goals"].values,
            train.loc[mask, "train_weight"].values,
            test[feature_cols],
            rounds,
            {**tag_base, "target": "opp"},
        )
        assert names == opp_names
        recent[cutoff] = {
            "lambda_team": np.mean(team_preds, axis=0),
            "lambda_opp": np.mean(opp_preds, axis=0),
            "model_names": names,
            "rows": int(mask.sum()),
        }
    return recent


def make_final_submission(train, test, feature_cols, chosen: CandidateResult, fallback_to_v5: bool):
    sample = pd.read_csv(SAMPLE_SUB)
    v5 = pd.read_csv(V5_SUB)
    if fallback_to_v5:
        sub = sample[["Id"]].merge(v5[["Id", "team_goals", "opp_goals"]], on="Id", how="left")
        sub.to_csv(OUTPUT_SUB, index=False)
        return sub, score_distribution(sub["team_goals"].values, sub["opp_goals"].values), {}

    print("\n[4] Training final recent correction expert...")
    recent = fit_recent_final(train, test, feature_cols, chosen)
    recent_team, recent_opp = combine_recent_predictions(recent, chosen.config.preset)
    v5_aligned = test[["Id"]].merge(v5[["Id", "team_goals", "opp_goals"]], on="Id", how="left")
    if v5_aligned[["team_goals", "opp_goals"]].isna().any().any():
        raise ValueError("submission_v5.csv is missing test Id rows.")

    lambda_team, lambda_opp, gate = apply_bounded_correction(
        v5_aligned["team_goals"].values,
        v5_aligned["opp_goals"].values,
        recent_team,
        recent_opp,
        test,
        chosen.config,
    )
    pred_t, pred_o = predict_from_lambdas(
        lambda_team,
        lambda_opp,
        chosen.config.max_goals,
        chosen.config.draw_boost,
        chosen.config.low_score_boost,
    )
    pred_t = np.where(gate, pred_t, v5_aligned["team_goals"].values.astype(int))
    pred_o = np.where(gate, pred_o, v5_aligned["opp_goals"].values.astype(int))

    sub = pd.DataFrame({"Id": test["Id"].values, "team_goals": pred_t.astype(int), "opp_goals": pred_o.astype(int)})
    sub = sample[["Id"]].merge(sub, on="Id", how="left")
    sub["team_goals"] = sub["team_goals"].astype(int)
    sub["opp_goals"] = sub["opp_goals"].astype(int)
    sub.to_csv(OUTPUT_SUB, index=False)
    return sub, score_distribution(sub["team_goals"].values, sub["opp_goals"].values), {
        "gate_share_test": float(np.mean(gate)),
        "recent_cutoff_rows": {k: v["rows"] for k, v in recent.items()},
        "recent_model_names": sorted(set(n for v in recent.values() for n in v["model_names"])),
    }


# ===========================================================================
# 9. REPORTING
# ===========================================================================
def local_metrics(sub_path: Path, power=AWMAE_POWER):
    if not GT_PATH.exists() or not sub_path.exists():
        return None
    sub = pd.read_csv(sub_path)
    gt = pd.read_csv(GT_PATH)
    raw_test = pd.read_csv(TEST_RAW, usecols=["Id", "tournament"])
    raw_test["metric_weight"] = raw_test["tournament"].map(TOURNAMENT_WEIGHT_MAP).fillna(DEFAULT_TOURNAMENT_WEIGHT)
    df = sub.merge(gt, on="Id", suffixes=("_pred", "_true")).merge(raw_test[["Id", "metric_weight"]], on="Id", how="left")
    return {
        **metrics_dict(
            df["team_goals_pred"].values,
            df["opp_goals_pred"].values,
            df["team_goals_true"].values,
            df["opp_goals_true"].values,
            weights=df["metric_weight"].values,
            power=power,
        ),
        "rows": int(len(df)),
        "distribution": score_distribution(df["team_goals_pred"].values, df["opp_goals_pred"].values),
    }


def segment_metrics(pred_t, pred_o, true_t, true_o, val_df):
    segments = {}
    masks = {
        "major_tournaments": val_df["tournament_weight"].values >= 1.50,
        "friendlies": (val_df["tournament"].values == "Friendly") if "tournament" in val_df else np.zeros(len(val_df), dtype=bool),
        "qualifiers": val_df["tournament"].astype(str).str.contains("qualification", case=False, na=False).values
        if "tournament" in val_df
        else np.zeros(len(val_df), dtype=bool),
        "neutral": val_df["neutral"].fillna(0).astype(bool).values if "neutral" in val_df else np.zeros(len(val_df), dtype=bool),
        "non_neutral": ~val_df["neutral"].fillna(0).astype(bool).values if "neutral" in val_df else np.ones(len(val_df), dtype=bool),
    }
    weights = val_df["metric_weight"].values
    for name, mask in masks.items():
        if int(mask.sum()) == 0:
            continue
        segments[name] = {
            "rows": int(mask.sum()),
            **metrics_dict(
                np.asarray(pred_t)[mask],
                np.asarray(pred_o)[mask],
                np.asarray(true_t)[mask],
                np.asarray(true_o)[mask],
                weights=weights[mask],
            ),
        }
    return segments


def build_segment_summary(chosen: CandidateResult, artifacts: list[dict]):
    pred_t_all = []
    pred_o_all = []
    true_t_all = []
    true_o_all = []
    val_all = []
    for artifact in artifacts:
        val = artifact["val"]
        anchor = artifact["anchor"]
        recent_team, recent_opp = combine_recent_predictions(artifact["recent"], chosen.config.preset)
        lambda_team, lambda_opp, gate = apply_bounded_correction(
            anchor["pred_team"], anchor["pred_opp"], recent_team, recent_opp, val, chosen.config
        )
        pred_t, pred_o = predict_from_lambdas(
            lambda_team, lambda_opp, chosen.config.max_goals, chosen.config.draw_boost, chosen.config.low_score_boost
        )
        pred_t = np.where(gate, pred_t, anchor["pred_team"])
        pred_o = np.where(gate, pred_o, anchor["pred_opp"])
        pred_t_all.append(pred_t)
        pred_o_all.append(pred_o)
        true_t_all.append(val["team_goals"].values.astype(int))
        true_o_all.append(val["opp_goals"].values.astype(int))
        val_all.append(val)
    val_df = pd.concat(val_all, ignore_index=True)
    return segment_metrics(
        np.concatenate(pred_t_all),
        np.concatenate(pred_o_all),
        np.concatenate(true_t_all),
        np.concatenate(true_o_all),
        val_df,
    )


def write_outputs(
    train,
    test,
    feature_cols,
    dropped_features,
    chosen: CandidateResult,
    fallback_to_v5: bool,
    final_distribution: dict,
    final_extra: dict,
    segment_summary: dict,
    elapsed_minutes: float,
):
    v5_local = local_metrics(V5_SUB, power=AWMAE_POWER)
    v8_local = local_metrics(OUTPUT_SUB, power=AWMAE_POWER)
    v5_legacy = local_metrics(V5_SUB, power=LEGACY_LOCAL_POWER)
    v8_legacy = local_metrics(OUTPUT_SUB, power=LEGACY_LOCAL_POWER)
    config = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "seed": SEED,
        "strategy_name": STRATEGY_NAME,
        "awmae_power": AWMAE_POWER,
        "legacy_local_power": LEGACY_LOCAL_POWER,
        "fast_mode": FAST_MODE,
        "fallback_to_v5": fallback_to_v5,
        "fallback_reasons": chosen.safety_reasons if fallback_to_v5 else [],
        "validation_mode": "multi_fold_modern",
        "active_cutoffs": ACTIVE_CUTOFFS + ([] if FAST_MODE else OPTIONAL_CUTOFFS),
        "active_base_models": ["xgb_poisson", "hgb_poisson"]
        + (["lgb_poisson"] if ENABLE_LGB and lgb is not None else [])
        + (["cat_poisson"] if ENABLE_CATBOOST and CatBoostRegressor is not None else []),
        "recent_cutoff_weights": chosen.config.preset,
        "beta_team": chosen.config.beta_team,
        "beta_opp": chosen.config.beta_opp,
        "cap_team": chosen.config.cap_team,
        "cap_opp": chosen.config.cap_opp,
        "anchor_offset": chosen.config.anchor_offset,
        "gate_threshold": chosen.config.gate_threshold,
        "max_goals": chosen.config.max_goals,
        "draw_boost": chosen.config.draw_boost,
        "low_score_boost": chosen.config.low_score_boost,
        "anti_score_collapse_thresholds": {
            "max_common_low_score_share_increase": MAX_COMMON_LOW_SCORE_SHARE_INCREASE,
            "max_top3_score_share_increase": MAX_TOP3_SCORE_SHARE_INCREASE,
            "max_draw_share_shift": MAX_DRAW_SHARE_SHIFT,
            "max_avg_total_goals_drop": MAX_AVG_TOTAL_GOALS_DROP,
        },
        "validation_anchor_metrics": chosen.anchor_metrics,
        "validation_recent_only_metrics": chosen.recent_only_metrics,
        "validation_hybrid_metrics": {
            "weighted_awmae": chosen.weighted_awmae,
            "unweighted_awmae": chosen.unweighted_awmae,
            "outcome_accuracy": chosen.outcome_accuracy,
            "exact_accuracy": chosen.exact_accuracy,
            "goal_diff_accuracy": chosen.goal_diff_accuracy,
            "selection_score": chosen.selection_score,
            "score_collapse_penalty": chosen.score_collapse_penalty,
            "stability_penalty": chosen.stability_penalty,
            "safety_passed": chosen.safety_passed,
            "safety_reasons": chosen.safety_reasons,
        },
        "fold_level_summary": chosen.fold_scores,
        "distribution_diagnostics": {**chosen.distribution, "final_test": final_distribution},
        "segment_diagnostics_summary": segment_summary,
        "local_v5_metrics_power_1_5": v5_local,
        "local_v8_anchor_safe_metrics_power_1_5": v8_local,
        "local_v5_metrics_legacy_power_1_3": v5_legacy,
        "local_v8_anchor_safe_metrics_legacy_power_1_3": v8_legacy,
        "local_delta_awmae_power_1_5": None
        if v5_local is None or v8_local is None
        else v8_local["weighted_awmae"] - v5_local["weighted_awmae"],
        "local_delta_awmae_legacy_power_1_3": None
        if v5_legacy is None or v8_legacy is None
        else v8_legacy["weighted_awmae"] - v5_legacy["weighted_awmae"],
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "feature_count": int(len(feature_cols)),
        "feature_columns": feature_cols,
        "dropped_features": dropped_features,
        "final_extra": final_extra,
        "elapsed_minutes": elapsed_minutes,
    }

    OUTPUT_CONFIG.write_text(json.dumps(config, indent=2, default=float), encoding="utf-8")

    lines = []
    lines.append("V8 Anchor Safe Validation Report - V5-centered bounded selective correction")
    lines.append("=" * 72)
    lines.append(f"AW-MAE formula power: {AWMAE_POWER}")
    lines.append(f"Fallback to V5: {fallback_to_v5}")
    if fallback_to_v5:
        lines.append(f"Fallback reasons: {', '.join(chosen.safety_reasons) if chosen.safety_reasons else 'none'}")
    lines.append("")
    lines.append("Validation anchor vs recent-only vs hybrid")
    lines.append(f"  Anchor weighted AW-MAE : {chosen.anchor_metrics['weighted_awmae']:.5f}")
    lines.append(f"  Recent weighted AW-MAE : {chosen.recent_only_metrics['weighted_awmae']:.5f}")
    lines.append(f"  Hybrid weighted AW-MAE : {chosen.weighted_awmae:.5f}")
    lines.append(f"  Hybrid outcome accuracy: {chosen.outcome_accuracy:.4f}")
    lines.append(f"  Hybrid exact accuracy  : {chosen.exact_accuracy:.4f}")
    lines.append(f"  Hybrid goal-diff acc   : {chosen.goal_diff_accuracy:.4f}")
    lines.append(f"  Selection score        : {chosen.selection_score:.5f}")
    lines.append(f"  Collapse penalty       : {chosen.score_collapse_penalty:.5f}")
    lines.append("")
    lines.append("Chosen correction config")
    lines.append(f"  recent weights: {chosen.config.preset}")
    lines.append(f"  beta: team={chosen.config.beta_team}, opp={chosen.config.beta_opp}")
    lines.append(f"  cap: team={chosen.config.cap_team}, opp={chosen.config.cap_opp}")
    lines.append(f"  anchor_offset={chosen.config.anchor_offset}, gate_threshold={chosen.config.gate_threshold}")
    lines.append(f"  max_goals={chosen.config.max_goals}, draw_boost={chosen.config.draw_boost}, low_score_boost={chosen.config.low_score_boost}")
    lines.append("")
    lines.append("Fold summary")
    for fold in chosen.fold_scores:
        lines.append(
            f"  {fold['fold_name']}: weighted={fold['weighted_awmae']:.5f}, "
            f"outcome={fold['outcome_accuracy']:.4f}, exact={fold['exact_accuracy']:.4f}, gate={fold['gate_share']:.3f}"
        )
    lines.append("")
    lines.append("Anti-score-collapse diagnostics")
    for label in ["anchor", "hybrid"]:
        dist = chosen.distribution[label]
        lines.append(
            f"  validation {label}: common_low={dist['common_low_score_share']:.4f}, "
            f"top3={dist['top_3_score_share']:.4f}, draw={dist['draw_share']:.4f}, avg_total={dist['avg_total_goals']:.3f}"
        )
    lines.append(
        f"  final test: common_low={final_distribution['common_low_score_share']:.4f}, "
        f"top3={final_distribution['top_3_score_share']:.4f}, draw={final_distribution['draw_share']:.4f}, "
        f"avg_total={final_distribution['avg_total_goals']:.3f}"
    )
    lines.append("")
    lines.append("Top final test scores")
    for item in final_distribution["top_10_scores"]:
        lines.append(f"  {item['score']}: {item['count']} ({item['share']:.4f})")
    lines.append("")
    lines.append("Segment diagnostics")
    for name, metric in segment_summary.items():
        lines.append(
            f"  {name}: rows={metric['rows']}, weighted={metric['weighted_awmae']:.5f}, "
            f"outcome={metric['outcome_accuracy']:.4f}, exact={metric['exact_accuracy']:.4f}"
        )
    lines.append("")
    lines.append("Local final reporting")
    if v5_local and v8_local:
        lines.append(f"  V5 local weighted AW-MAE power 1.5: {v5_local['weighted_awmae']:.5f}")
        lines.append(f"  V8 Anchor Safe local weighted AW-MAE power 1.5: {v8_local['weighted_awmae']:.5f}")
        lines.append(f"  Delta power 1.5                           : {v8_local['weighted_awmae'] - v5_local['weighted_awmae']:.5f}")
        lines.append(f"  V5 outcome/exact              : {v5_local['outcome_accuracy']:.4f} / {v5_local['exact_accuracy']:.4f}")
        lines.append(f"  V8 Anchor Safe outcome/exact              : {v8_local['outcome_accuracy']:.4f} / {v8_local['exact_accuracy']:.4f}")
    if v5_legacy and v8_legacy:
        lines.append(f"  V5 local weighted AW-MAE legacy power 1.3: {v5_legacy['weighted_awmae']:.5f}")
        lines.append(f"  V8 Anchor Safe local weighted AW-MAE legacy power 1.3: {v8_legacy['weighted_awmae']:.5f}")
        lines.append(f"  Delta legacy power 1.3                           : {v8_legacy['weighted_awmae'] - v5_legacy['weighted_awmae']:.5f}")
    lines.append("")
    lines.append(f"Done in {elapsed_minutes:.1f} minutes.")
    OUTPUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ===========================================================================
# 10. MAIN
# ===========================================================================
def main():
    t0 = time.time()
    print("=" * 72)
    print("STATIC V8 ANCHOR SAFE - V5-CENTERED BOUNDED SELECTIVE CORRECTION")
    print("=" * 72)
    train, test, feature_cols, dropped_features = load_data()
    artifacts = build_validation_artifacts(train, feature_cols)
    chosen = tune_candidates(artifacts)
    segment_summary = build_segment_summary(chosen, artifacts)

    fallback_to_v5 = not chosen.safety_passed
    print(f"\n[4] Safety decision: fallback_to_v5={fallback_to_v5}")
    sub, final_distribution, final_extra = make_final_submission(train, test, feature_cols, chosen, fallback_to_v5)
    elapsed = (time.time() - t0) / 60.0
    write_outputs(
        train,
        test,
        feature_cols,
        dropped_features,
        chosen,
        fallback_to_v5,
        final_distribution,
        final_extra,
        segment_summary,
        elapsed,
    )
    print(f"\n[OK] Wrote {OUTPUT_SUB.relative_to(BASE_DIR)} ({len(sub)} rows)")
    print(f"[OK] Wrote {OUTPUT_CONFIG.relative_to(BASE_DIR)}")
    print(f"[OK] Wrote {OUTPUT_REPORT.relative_to(BASE_DIR)}")
    print(f"Done in {elapsed:.1f} minutes.")


if __name__ == "__main__":
    main()
