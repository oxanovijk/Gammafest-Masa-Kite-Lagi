"""
Risk-minimizing static pipeline v1 -- Gammafest Masa Kite Lagi.

This pipeline is intentionally independent from V5 outputs:
  * no import from model_pipeline_v5.py
  * no V3/V4/V5/V8 submission anchor or fallback
  * no test-period state update from predicted scores

Outputs:
  dataset/submission_risk_v1.csv
  dataset/submission_risk_v1_config.json
  dataset/submission_risk_v1_validation_report.txt
"""

from __future__ import annotations

import json
import math
import os
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
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover - optional dependency
    lgb = None

try:
    from catboost import CatBoostRegressor
except ImportError:  # pragma: no cover - optional dependency
    CatBoostRegressor = None

warnings.filterwarnings("ignore")


# ===========================================================================
# 1. CONFIG
# ===========================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "dataset"

TRAIN_FINAL = DATA_DIR / "train_final.csv"
TEST_FINAL = DATA_DIR / "test_final.csv"
TRAIN_RAW = DATA_DIR / "train.csv"
TEST_RAW = DATA_DIR / "test.csv"
SAMPLE_SUB = DATA_DIR / "sample submission.csv"
GT_PATH = DATA_DIR / "test_ground_truth.csv"
FRIEND_SUB = Path(
    r"C:\Users\LENOVO\Downloads\submission_final_model_first_selector_v4_accepted_top200_cfg_0097_same_only_strict_all_f0.001_g0.1.csv"
)

OUTPUT_SUB = DATA_DIR / "submission_risk_v1.csv"
OUTPUT_CONFIG = DATA_DIR / "submission_risk_v1_config.json"
OUTPUT_REPORT = DATA_DIR / "submission_risk_v1_validation_report.txt"

SEED = 42
PRIMARY_POWER = 1.5
SECONDARY_POWER = 1.3
TARGET_POWER_15_WEIGHTED = 2.40
FAST_MODE = True

ENABLE_LGB = True
ENABLE_CATBOOST = True
N_ROUNDS_VAL = 180 if FAST_MODE else 420
N_ROUNDS_FINAL = 320 if FAST_MODE else 700

MAX_GOALS_CANDIDATES = [7, 8, 10]
DRAW_BOOSTS = [0.96, 1.00, 1.04, 1.08, 1.12]
LOW_SCORE_BOOSTS = [0.96, 1.00, 1.04, 1.08]
TAIL_DAMPENERS = [0.90, 0.95, 1.00]
OUTCOME_ALPHAS = [0.00, 0.03, 0.06, 0.10, 0.15]

LOW_SCORE_CELLS = [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (0, 2), (2, 1), (1, 2)]
COMMON_LOW_SCORES = set(LOW_SCORE_CELLS)

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
    stage: str
    name: str
    team_weights: tuple[float, ...]
    opp_weights: tuple[float, ...]
    max_goals: int = 8
    draw_boost: float = 1.0
    low_score_boost: float = 1.0
    tail_dampener: float = 1.0
    outcome_alpha: float = 0.0
    risk_penalties_enabled: bool = True


@dataclass
class CandidateResult:
    config: CandidateConfig
    metrics: dict
    fold_metrics: list[dict]
    distribution: dict
    yearly_distribution: dict
    risk_components: dict
    selection_score: float
    accepted_by_ablation: bool
    ablation_reason: str


# ===========================================================================
# 2. METRICS, DISTRIBUTION, ERM
# ===========================================================================
def awmae_loss_array(pred_t, pred_o, true_t, true_o, power=PRIMARY_POWER) -> np.ndarray:
    pred_t = np.asarray(pred_t, dtype=float)
    pred_o = np.asarray(pred_o, dtype=float)
    true_t = np.asarray(true_t, dtype=float)
    true_o = np.asarray(true_o, dtype=float)
    mae = (np.abs(true_t - pred_t) + np.abs(true_o - pred_o)) / 2.0
    exact = ((pred_t == true_t) & (pred_o == true_o)).astype(float)
    outcome = (np.sign(pred_t - pred_o) == np.sign(true_t - true_o)).astype(float)
    gd = ((pred_t - pred_o) == (true_t - true_o)).astype(float)
    penalty = 0.30 * (1.0 - exact) + 0.25 * (1.0 - outcome) + 0.15 * (1.0 - gd)
    multiplier = np.where(outcome == 1.0, 1.0, 1.5)
    return ((mae + penalty) * multiplier) ** power


def mean_awmae(pred_t, pred_o, true_t, true_o, weights=None, power=PRIMARY_POWER) -> float:
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


def metrics_dict(pred_t, pred_o, true_t, true_o, weights=None, power=PRIMARY_POWER) -> dict:
    return {
        "weighted_awmae": mean_awmae(pred_t, pred_o, true_t, true_o, weights=weights, power=power),
        "unweighted_awmae": mean_awmae(pred_t, pred_o, true_t, true_o, weights=None, power=power),
        "outcome_accuracy": outcome_accuracy(pred_t, pred_o, true_t, true_o),
        "exact_accuracy": exact_accuracy(pred_t, pred_o, true_t, true_o),
        "goal_diff_accuracy": goal_diff_accuracy(pred_t, pred_o, true_t, true_o),
    }


def build_loss_tensor(max_goals: int, power=PRIMARY_POWER) -> np.ndarray:
    tensor = np.zeros((max_goals, max_goals, max_goals, max_goals), dtype=np.float32)
    for pt in range(max_goals):
        for po in range(max_goals):
            for tt in range(max_goals):
                for to in range(max_goals):
                    tensor[pt, po, tt, to] = awmae_loss_array([pt], [po], [tt], [to], power=power)[0]
    return tensor


LOSS_TENSOR_CACHE: dict[tuple[int, float], np.ndarray] = {}


def get_loss_tensor(max_goals: int, power=PRIMARY_POWER) -> np.ndarray:
    key = (max_goals, power)
    if key not in LOSS_TENSOR_CACHE:
        LOSS_TENSOR_CACHE[key] = build_loss_tensor(max_goals, power=power)
    return LOSS_TENSOR_CACHE[key]


def score_probability_matrix(
    lambda_team,
    lambda_opp,
    max_goals: int,
    draw_boost=1.0,
    low_score_boost=1.0,
    tail_dampener=1.0,
    outcome_probs=None,
    outcome_alpha=0.0,
) -> np.ndarray:
    lambda_team = np.clip(np.asarray(lambda_team, dtype=float), 1e-6, 15.0)
    lambda_opp = np.clip(np.asarray(lambda_opp, dtype=float), 1e-6, 15.0)
    k = np.arange(max_goals)
    pmf_t = poisson.pmf(k[None, :], lambda_team[:, None])
    pmf_o = poisson.pmf(k[None, :], lambda_opp[:, None])
    pmf_t /= np.maximum(pmf_t.sum(axis=1, keepdims=True), 1e-12)
    pmf_o /= np.maximum(pmf_o.sum(axis=1, keepdims=True), 1e-12)
    prob = pmf_t[:, :, None] * pmf_o[:, None, :]

    idx = np.arange(max_goals)
    if draw_boost != 1.0:
        prob[:, idx, idx] *= draw_boost
    if low_score_boost != 1.0:
        for a, b in LOW_SCORE_CELLS:
            if a < max_goals and b < max_goals:
                prob[:, a, b] *= low_score_boost
    if tail_dampener != 1.0:
        tail_mask = (idx[:, None] >= 5) | (idx[None, :] >= 5)
        prob[:, tail_mask] *= tail_dampener

    if outcome_probs is not None and outcome_alpha > 0.0:
        outcome_probs = np.asarray(outcome_probs, dtype=float)
        loss_mask = idx[:, None] < idx[None, :]
        draw_mask = idx[:, None] == idx[None, :]
        win_mask = idx[:, None] > idx[None, :]
        outcome_weight = (
            outcome_probs[:, 0, None, None] * loss_mask[None, :, :]
            + outcome_probs[:, 1, None, None] * draw_mask[None, :, :]
            + outcome_probs[:, 2, None, None] * win_mask[None, :, :]
        )
        prob *= (1.0 - outcome_alpha) + outcome_alpha * np.maximum(outcome_weight, 1e-8) * 3.0

    prob /= np.maximum(prob.sum(axis=(1, 2), keepdims=True), 1e-12)
    return prob


def erm_from_probability(prob: np.ndarray, loss_tensor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    expected_loss = np.einsum("abij,nij->nab", loss_tensor, prob)
    flat = expected_loss.reshape(len(prob), -1).argmin(axis=1)
    max_goals = loss_tensor.shape[0]
    return (flat // max_goals).astype(int), (flat % max_goals).astype(int)


def predict_scores(lambda_team, lambda_opp, config: CandidateConfig, outcome_probs=None):
    prob = score_probability_matrix(
        lambda_team,
        lambda_opp,
        config.max_goals,
        config.draw_boost,
        config.low_score_boost,
        config.tail_dampener,
        outcome_probs=outcome_probs,
        outcome_alpha=config.outcome_alpha,
    )
    return erm_from_probability(prob, get_loss_tensor(config.max_goals, PRIMARY_POWER))


def score_distribution(pred_t, pred_o) -> dict:
    pred_t = np.asarray(pred_t, dtype=int)
    pred_o = np.asarray(pred_o, dtype=int)
    total = max(1, len(pred_t))
    diff = pred_t - pred_o
    counts = Counter(zip(pred_t.tolist(), pred_o.tolist()))
    top = counts.most_common(10)
    top_shares = [c / total for _, c in top]
    return {
        "rows": int(len(pred_t)),
        "win_share": float(np.mean(diff > 0)),
        "draw_share": float(np.mean(diff == 0)),
        "loss_share": float(np.mean(diff < 0)),
        "avg_team_goals": float(np.mean(pred_t)),
        "avg_opp_goals": float(np.mean(pred_o)),
        "avg_total_goals": float(np.mean(pred_t + pred_o)),
        "score_ge5_share": float(np.mean((pred_t >= 5) | (pred_o >= 5))),
        "common_low_score_share": float(np.mean([(a, b) in COMMON_LOW_SCORES for a, b in zip(pred_t, pred_o)])),
        "top1_score_share": float(top_shares[0] if top_shares else 0.0),
        "top3_score_share": float(sum(top_shares[:3])),
        "top5_score_share": float(sum(top_shares[:5])),
        "top_10_scores": [{"score": f"{a}-{b}", "count": int(c), "share": float(c / total)} for (a, b), c in top],
    }


def yearly_distribution(pred_t, pred_o, dates) -> dict:
    df = pd.DataFrame({"pred_t": pred_t, "pred_o": pred_o, "year": pd.to_datetime(dates).dt.year})
    out = {}
    for year, g in df.groupby("year"):
        out[str(int(year))] = score_distribution(g["pred_t"].values, g["pred_o"].values)
    return out


def yearly_distribution_penalty(yearly: dict, global_draw_share: float) -> float:
    if not yearly:
        return 0.0
    vals = []
    for dist in yearly.values():
        vals.append(
            0.10 * max(0.0, dist["top3_score_share"] - 0.60) ** 2
            + 0.10 * max(0.0, dist["score_ge5_share"] - 0.04) ** 2
            + 0.05 * max(0.0, abs(dist["draw_share"] - global_draw_share) - 0.05) ** 2
        )
    return float(np.mean(vals))


# ===========================================================================
# 3. DATA AND FEATURES
# ===========================================================================
def raw_usecols(path: Path) -> list[str]:
    available = pd.read_csv(path, nrows=0).columns.tolist()
    wanted = [
        "Id",
        "date",
        "tournament",
        "neutral",
        "is_home",
        "venue_country",
        "confederation_team",
        "confederation_opp",
        "rank_team",
        "rank_opponent",
        "rank_diff",
        "population_team",
        "population_opp",
        "gdp_per_capita_team",
        "gdp_per_capita_opp",
        "altitude_venue",
        "distance_travel_team",
        "distance_travel_opp",
        "temperature_venue",
    ]
    return [c for c in wanted if c in available]


def add_static_risk_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["tournament_weight"] = out["tournament"].map(TOURNAMENT_WEIGHT_MAP).fillna(DEFAULT_TOURNAMENT_WEIGHT)

    if {"elo_team_feat", "elo_opponent_feat"}.issubset(out.columns):
        out["elo_diff_risk"] = out["elo_team_feat"] - out["elo_opponent_feat"]
    elif "elo_diff_feat" in out.columns:
        out["elo_diff_risk"] = out["elo_diff_feat"]

    if "elo_diff_risk" in out.columns:
        out["abs_elo_diff_risk"] = out["elo_diff_risk"].abs()
        out["balanced_match_risk"] = (out["abs_elo_diff_risk"] <= 50).astype(int)
        out["favorite_risk"] = (out["elo_diff_risk"] > 100).astype(int)
        out["underdog_risk"] = (out["elo_diff_risk"] < -100).astype(int)
        out["strong_favorite_risk"] = (out["elo_diff_risk"] > 200).astype(int)
        out["elo_x_tournament_weight_risk"] = out["elo_diff_risk"] * out["tournament_weight"]

    if {"rank_team", "rank_opponent"}.issubset(out.columns):
        out["rank_diff_risk"] = out["rank_opponent"] - out["rank_team"]
    elif "rank_diff" in out.columns:
        out["rank_diff_risk"] = out["rank_diff"]
    if "rank_diff_risk" in out.columns:
        out["abs_rank_diff_risk"] = out["rank_diff_risk"].abs()

    if {"team_avg_gf_last5_ewma_feat", "opp_avg_gf_last5_ewma_feat"}.issubset(out.columns):
        out["expected_total_goals_proxy_risk"] = (
            out["team_avg_gf_last5_ewma_feat"].fillna(0) + out["opp_avg_gf_last5_ewma_feat"].fillna(0)
        )
    if {"form_team_feat", "form_opp_feat"}.issubset(out.columns):
        out["form_uncertainty_proxy_risk"] = 1.0 / (1.0 + (out["form_team_feat"] - out["form_opp_feat"]).abs())

    if "neutral" in out.columns and "elo_diff_risk" in out.columns:
        out["neutral_x_elo_risk"] = out["neutral"].fillna(0) * out["elo_diff_risk"]
    if "is_home" in out.columns and "elo_diff_risk" in out.columns:
        out["home_x_elo_risk"] = out["is_home"].fillna(0) * out["elo_diff_risk"]

    dt = pd.to_datetime(out["date"])
    out["year_risk"] = dt.dt.year
    out["month_risk"] = dt.dt.month
    tournament = out["tournament"].astype(str)
    out["is_friendly_risk"] = (tournament == "Friendly").astype(int)
    out["is_qualifier_risk"] = tournament.str.contains("qualification", case=False, na=False).astype(int)
    out["is_major_tournament_risk"] = (out["tournament_weight"] >= 1.50).astype(int)
    out["high_importance_tournament_risk"] = (out["tournament_weight"] >= 1.70).astype(int)
    return out


class MedianImputer:
    def __init__(self):
        self.medians: pd.Series | None = None

    def fit(self, x: pd.DataFrame):
        self.medians = x.replace([np.inf, -np.inf], np.nan).median(numeric_only=True).fillna(0.0)
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        if self.medians is None:
            raise ValueError("MedianImputer is not fitted.")
        return x.replace([np.inf, -np.inf], np.nan).fillna(self.medians)


def load_data():
    print("[1] Loading frozen data and static risk features...")
    train = pd.read_csv(TRAIN_FINAL)
    test = pd.read_csv(TEST_FINAL)
    raw_train = pd.read_csv(TRAIN_RAW, usecols=raw_usecols(TRAIN_RAW))
    raw_test = pd.read_csv(TEST_RAW, usecols=raw_usecols(TEST_RAW))

    train = train.merge(raw_train, on="Id", how="left")
    test = test.merge(raw_test, on="Id", how="left")
    train["date"] = pd.to_datetime(train["date"])
    test["date"] = pd.to_datetime(test["date"])

    train = add_static_risk_features(train)
    test = add_static_risk_features(test)
    train["metric_weight"] = train["tournament_weight"]
    train["train_weight"] = train["tournament_weight"]
    test["metric_weight"] = test["tournament_weight"]

    exclude = {
        "Id",
        "team_goals",
        "opp_goals",
        "date",
        "tournament",
        "venue_country",
        "confederation_team",
        "confederation_opp",
        "sample_weight",
        "train_weight",
        "metric_weight",
        "time_weight",
        "is_test",
    }
    candidate_cols = [c for c in train.columns if c not in exclude and c in test.columns]
    feature_cols = [
        c
        for c in candidate_cols
        if pd.api.types.is_numeric_dtype(train[c]) and pd.api.types.is_numeric_dtype(test[c])
    ]
    dropped_cols = [c for c in candidate_cols if c not in feature_cols]
    train[feature_cols] = train[feature_cols].replace([np.inf, -np.inf], np.nan)
    test[feature_cols] = test[feature_cols].replace([np.inf, -np.inf], np.nan)

    feature_groups = {
        "core_historical": [c for c in feature_cols if c.endswith("_feat")],
        "context": [c for c in feature_cols if c.endswith("_ctx")],
        "match_metadata": [
            c
            for c in feature_cols
            if c in {"neutral", "is_home", "tournament_weight"}
            or c in {"is_friendly_risk", "is_major_tournament_risk", "is_qualifier_risk", "year_risk", "month_risk"}
        ],
        "risk_features": [c for c in feature_cols if c.endswith("_risk") and c not in {"year_risk", "month_risk"}],
    }
    print(f"    Train: {train.shape} | Test: {test.shape} | Features: {len(feature_cols)}")
    return train, test, feature_cols, dropped_cols, feature_groups


def prepare_xy(train_df, pred_df, feature_cols):
    imputer = MedianImputer().fit(train_df[feature_cols])
    return imputer.transform(train_df[feature_cols]), imputer.transform(pred_df[feature_cols])


# ===========================================================================
# 4. MODEL WRAPPERS
# ===========================================================================
class XGBPoissonModel:
    name = "xgb_poisson"

    def __init__(self, rounds: int):
        self.rounds = rounds
        self.model = None

    def fit(self, x, y, weight):
        self.model = xgb.train(
            XGB_POISSON_PARAMS,
            xgb.DMatrix(x, label=y, weight=weight),
            num_boost_round=self.rounds,
            verbose_eval=False,
        )
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
            iterations=min(rounds, 260),
            depth=6,
            learning_rate=0.035,
            l2_leaf_reg=8.0,
            random_seed=SEED,
            verbose=False,
            allow_writing_files=False,
            thread_count=-1,
        )

    def fit(self, x, y, weight):
        self.model.fit(x, y, sample_weight=weight)
        return self

    def predict(self, x):
        return self.model.predict(x)


def model_factories(rounds: int):
    factories = [lambda: XGBPoissonModel(rounds), lambda: HGBPoissonModel(rounds)]
    if ENABLE_LGB and lgb is not None:
        factories.append(lambda: LGBPoissonModel(rounds))
    if ENABLE_CATBOOST and CatBoostRegressor is not None:
        factories.append(lambda: CatPoissonModel(rounds))
    return factories


def fit_predict_models(x_train, y_train, w_train, x_pred, rounds: int, label: str):
    preds = []
    names = []
    for factory in model_factories(rounds):
        model = factory()
        print(f"      - {label}: {model.name}")
        model.fit(x_train, y_train, w_train)
        preds.append(np.clip(model.predict(x_pred), 1e-5, 12.0))
        names.append(model.name)
    return np.vstack(preds), names


def outcome_target(team_goals, opp_goals) -> np.ndarray:
    diff = np.asarray(team_goals) - np.asarray(opp_goals)
    return np.where(diff > 0, 2, np.where(diff == 0, 1, 0)).astype(int)


def fit_outcome_classifier(x_train, train_df, w_train, x_pred):
    clf = HistGradientBoostingClassifier(
        learning_rate=0.035,
        max_iter=220 if FAST_MODE else 420,
        max_leaf_nodes=31,
        min_samples_leaf=70,
        l2_regularization=1.5,
        random_state=SEED,
        early_stopping=False,
    )
    y = outcome_target(train_df["team_goals"].values, train_df["opp_goals"].values)
    clf.fit(x_train, y, sample_weight=w_train)
    raw = clf.predict_proba(x_pred)
    out = np.full((len(x_pred), 3), 1e-8, dtype=float)
    for i, cls in enumerate(clf.classes_):
        out[:, int(cls)] = raw[:, i]
    out /= out.sum(axis=1, keepdims=True)
    return out


# ===========================================================================
# 5. VALIDATION ARTIFACTS AND CANDIDATE EVALUATION
# ===========================================================================
def fold_data(train: pd.DataFrame, fold: dict):
    train_mask = train["date"] <= pd.Timestamp(fold["train_end"])
    val_mask = (train["date"] >= pd.Timestamp(fold["valid_start"])) & (train["date"] <= pd.Timestamp(fold["valid_end"]))
    return train.loc[train_mask].copy(), train.loc[val_mask].copy()


def build_validation_artifacts(train: pd.DataFrame, feature_cols: list[str]):
    print("\n[2] Building train-only rolling fold artifacts...")
    artifacts = []
    model_names_ref = None
    for fold in FOLDS:
        tr, val = fold_data(train, fold)
        print(f"    {fold['name']}: train={len(tr)} | validation={len(val)}")
        x_tr, x_val = prepare_xy(tr, val, feature_cols)
        w_tr = tr["train_weight"].values
        team_preds, names = fit_predict_models(x_tr, tr["team_goals"].values, w_tr, x_val, N_ROUNDS_VAL, f"{fold['name']} team")
        opp_preds, opp_names = fit_predict_models(x_tr, tr["opp_goals"].values, w_tr, x_val, N_ROUNDS_VAL, f"{fold['name']} opp")
        if names != opp_names:
            raise RuntimeError("Team/opp model names diverged.")
        if model_names_ref is None:
            model_names_ref = names
        elif model_names_ref != names:
            raise RuntimeError("Model availability changed across folds.")
        outcome_probs = fit_outcome_classifier(x_tr, tr, w_tr, x_val)
        artifacts.append(
            {
                "fold": fold,
                "train_rows": len(tr),
                "val": val,
                "team_preds": team_preds,
                "opp_preds": opp_preds,
                "outcome_probs": outcome_probs,
            }
        )
    return artifacts, model_names_ref or []


def normalize_weights(values) -> tuple[float, ...]:
    arr = np.asarray(values, dtype=float)
    arr = np.maximum(arr, 0.0)
    if arr.sum() <= 0:
        arr[:] = 1.0
    arr /= arr.sum()
    return tuple(float(x) for x in arr)


def weight_options(n_models: int) -> list[tuple[float, ...]]:
    opts = [normalize_weights(np.ones(n_models))]
    for i in range(n_models):
        v = np.full(n_models, 0.10 / max(1, n_models - 1))
        v[i] = 0.90
        opts.append(normalize_weights(v))
    for i in range(n_models):
        v = np.ones(n_models)
        v[i] = 2.5
        opts.append(normalize_weights(v))
    # Deduplicate while preserving order.
    seen = set()
    out = []
    for opt in opts:
        key = tuple(round(x, 6) for x in opt)
        if key not in seen:
            seen.add(key)
            out.append(opt)
    return out


def apply_weights(preds: np.ndarray, weights: tuple[float, ...]) -> np.ndarray:
    return np.average(preds, axis=0, weights=np.asarray(weights, dtype=float))


def aggregate_artifact_predictions(config: CandidateConfig, artifact: dict):
    lambda_team = apply_weights(artifact["team_preds"], config.team_weights)
    lambda_opp = apply_weights(artifact["opp_preds"], config.opp_weights)
    outcome_probs = artifact["outcome_probs"] if config.outcome_alpha > 0 else None
    pred_t, pred_o = predict_scores(lambda_team, lambda_opp, config, outcome_probs=outcome_probs)
    return pred_t, pred_o, lambda_team, lambda_opp


def combine_fold_metrics(fold_metrics: list[dict], key: str) -> float:
    weights = np.array([m["fold_weight"] for m in fold_metrics], dtype=float)
    vals = np.array([m[key] for m in fold_metrics], dtype=float)
    return float(np.average(vals, weights=weights))


def evaluate_candidate(config: CandidateConfig, artifacts: list[dict], baseline: CandidateResult | None, thresholds: dict) -> CandidateResult:
    fold_metrics = []
    all_pred_t, all_pred_o, all_true_t, all_true_o, all_weights, all_dates = [], [], [], [], [], []
    for artifact in artifacts:
        val = artifact["val"]
        pred_t, pred_o, _, _ = aggregate_artifact_predictions(config, artifact)
        true_t = val["team_goals"].values.astype(int)
        true_o = val["opp_goals"].values.astype(int)
        weights = val["metric_weight"].values
        metric = metrics_dict(pred_t, pred_o, true_t, true_o, weights=weights, power=PRIMARY_POWER)
        metric_secondary = metrics_dict(pred_t, pred_o, true_t, true_o, weights=weights, power=SECONDARY_POWER)
        fold_metrics.append(
            {
                "fold_name": artifact["fold"]["name"],
                "fold_weight": artifact["fold"]["weight"],
                "rows": int(len(val)),
                **metric,
                "weighted_awmae_power_1_3": metric_secondary["weighted_awmae"],
                "unweighted_awmae_power_1_3": metric_secondary["unweighted_awmae"],
            }
        )
        all_pred_t.append(pred_t)
        all_pred_o.append(pred_o)
        all_true_t.append(true_t)
        all_true_o.append(true_o)
        all_weights.append(weights)
        all_dates.append(val["date"])

    pred_t = np.concatenate(all_pred_t)
    pred_o = np.concatenate(all_pred_o)
    true_t = np.concatenate(all_true_t)
    true_o = np.concatenate(all_true_o)
    weights = np.concatenate(all_weights)
    dates = pd.concat(all_dates, ignore_index=True)

    metrics = {
        "weighted_awmae_power_1_5": combine_fold_metrics(fold_metrics, "weighted_awmae"),
        "unweighted_awmae_power_1_5": mean_awmae(pred_t, pred_o, true_t, true_o, power=PRIMARY_POWER),
        "weighted_awmae_power_1_3": combine_fold_metrics(fold_metrics, "weighted_awmae_power_1_3"),
        "unweighted_awmae_power_1_3": mean_awmae(pred_t, pred_o, true_t, true_o, power=SECONDARY_POWER),
        "outcome_accuracy": outcome_accuracy(pred_t, pred_o, true_t, true_o),
        "exact_accuracy": exact_accuracy(pred_t, pred_o, true_t, true_o),
        "goal_diff_accuracy": goal_diff_accuracy(pred_t, pred_o, true_t, true_o),
    }
    dist = score_distribution(pred_t, pred_o)
    yearly = yearly_distribution(pred_t, pred_o, dates)
    fold_awmae = np.array([m["weighted_awmae"] for m in fold_metrics], dtype=float)

    base = baseline.metrics if baseline is not None else metrics
    base_dist = baseline.distribution if baseline is not None else dist
    fold_instability = 0.25 * float(np.std(fold_awmae)) + 0.10 * max(0.0, float(np.max(fold_awmae) - np.mean(fold_awmae) - 0.08))
    outcome_drop = 2.0 * max(0.0, base["outcome_accuracy"] - metrics["outcome_accuracy"] - 0.003)
    exact_drop = 1.5 * max(0.0, base["exact_accuracy"] - metrics["exact_accuracy"] - 0.005)
    high_tail = max(0.0, dist["score_ge5_share"] - thresholds["tail_threshold"]) ** 2
    collapse = (
        0.50 * max(0.0, dist["top3_score_share"] - thresholds["top3_threshold"]) ** 2
        + 0.25 * max(0.0, dist["top1_score_share"] - thresholds["top1_threshold"]) ** 2
        + 0.25 * max(0.0, abs(dist["draw_share"] - base_dist["draw_share"]) - 0.03) ** 2
    )
    yearly_penalty = yearly_distribution_penalty(yearly, dist["draw_share"])
    if not config.risk_penalties_enabled:
        high_tail = collapse = yearly_penalty = 0.0

    risk_components = {
        "fold_instability_penalty": float(fold_instability),
        "outcome_drop_penalty": float(outcome_drop),
        "exact_drop_penalty": float(exact_drop),
        "high_score_tail_penalty": float(high_tail),
        "score_collapse_penalty": float(collapse),
        "yearly_distribution_penalty": float(yearly_penalty),
    }
    selection_score = metrics["weighted_awmae_power_1_5"] + sum(risk_components.values())
    return CandidateResult(
        config=config,
        metrics=metrics,
        fold_metrics=fold_metrics,
        distribution=dist,
        yearly_distribution=yearly,
        risk_components=risk_components,
        selection_score=float(selection_score),
        accepted_by_ablation=True,
        ablation_reason="pending",
    )


def ablation_accepts(prev: CandidateResult, cand: CandidateResult) -> tuple[bool, str]:
    aw_ok = cand.metrics["weighted_awmae_power_1_5"] <= prev.metrics["weighted_awmae_power_1_5"] + 0.0005
    outcome_ok = cand.metrics["outcome_accuracy"] >= prev.metrics["outcome_accuracy"] - 0.003
    exact_ok = cand.metrics["exact_accuracy"] >= prev.metrics["exact_accuracy"] - 0.005
    instability_ok = cand.risk_components["fold_instability_penalty"] <= prev.risk_components["fold_instability_penalty"] + 0.01
    collapse_ok = cand.distribution["top3_score_share"] <= max(0.62, prev.distribution["top3_score_share"] + 0.04)
    ok = aw_ok and outcome_ok and exact_ok and instability_ok and collapse_ok
    reasons = []
    if not aw_ok:
        reasons.append("awmae_regression")
    if not outcome_ok:
        reasons.append("outcome_drop")
    if not exact_ok:
        reasons.append("exact_drop")
    if not instability_ok:
        reasons.append("instability")
    if not collapse_ok:
        reasons.append("score_collapse")
    return ok, "accepted" if ok else ",".join(reasons)


def friend_distribution_thresholds() -> dict:
    tail_threshold = 0.030
    friend_dist = None
    return {
        "tail_threshold": float(tail_threshold),
        "top3_threshold": 0.55,
        "top1_threshold": 0.23,
        "friend_distribution": friend_dist,
    }


def tune_candidates(artifacts: list[dict], model_names: list[str]) -> tuple[CandidateResult, list[dict], dict]:
    print("\n[3] Tuning risk-minimizing candidate stages...")
    thresholds = friend_distribution_thresholds()
    n_models = len(model_names)
    uniform = normalize_weights(np.ones(n_models))
    options = weight_options(n_models)

    baseline_config = CandidateConfig(
        stage="01_baseline_poisson_ensemble",
        name="uniform_max8_no_risk",
        team_weights=uniform,
        opp_weights=uniform,
        max_goals=8,
        risk_penalties_enabled=False,
    )
    baseline = evaluate_candidate(baseline_config, artifacts, None, thresholds)
    thresholds["top3_threshold"] = float(min(0.55, baseline.distribution["top3_score_share"] + 0.03))
    thresholds["top1_threshold"] = float(min(0.23, baseline.distribution["top1_score_share"] + 0.02))
    baseline = evaluate_candidate(baseline_config, artifacts, baseline, thresholds)
    baseline.ablation_reason = "initial_baseline"
    current = baseline
    ablations = [ablation_summary(baseline, selected=True)]
    print(f"    baseline: aw15={current.metrics['weighted_awmae_power_1_5']:.5f}, selection={current.selection_score:.5f}")

    stage_names = [
        "02_split_team_opp_blend",
        "03_awmae_erm_max_goals",
        "04_outcome_soft_reranker",
        "05_draw_low_score_calibration",
        "06_high_score_tail_penalty",
        "07_top_score_concentration_penalty",
        "08_segment_aware_calibration",
    ]

    def build_stage_configs(stage_name: str, base_config: CandidateConfig) -> list[CandidateConfig]:
        if stage_name == "02_split_team_opp_blend":
            return [
                CandidateConfig("02_split_team_opp_blend", f"tw{i}_ow{j}", tw, ow, max_goals=base_config.max_goals)
                for i, tw in enumerate(options)
                for j, ow in enumerate(options)
            ]
        if stage_name == "03_awmae_erm_max_goals":
            return [
                CandidateConfig(
                    "03_awmae_erm_max_goals",
                    f"max{m}",
                    base_config.team_weights,
                    base_config.opp_weights,
                    max_goals=m,
                )
                for m in MAX_GOALS_CANDIDATES
            ]
        if stage_name == "04_outcome_soft_reranker":
            return [
                CandidateConfig(
                    "04_outcome_soft_reranker",
                    f"alpha{a}",
                    base_config.team_weights,
                    base_config.opp_weights,
                    max_goals=base_config.max_goals,
                    outcome_alpha=a,
                )
                for a in OUTCOME_ALPHAS
            ]
        if stage_name == "05_draw_low_score_calibration":
            return [
                CandidateConfig(
                    "05_draw_low_score_calibration",
                    f"d{d}_l{lo}",
                    base_config.team_weights,
                    base_config.opp_weights,
                    max_goals=base_config.max_goals,
                    draw_boost=d,
                    low_score_boost=lo,
                    outcome_alpha=base_config.outcome_alpha,
                )
                for d in DRAW_BOOSTS
                for lo in LOW_SCORE_BOOSTS
            ]
        if stage_name == "06_high_score_tail_penalty":
            return [
                CandidateConfig(
                    "06_high_score_tail_penalty",
                    f"tail{td}",
                    base_config.team_weights,
                    base_config.opp_weights,
                    max_goals=base_config.max_goals,
                    draw_boost=base_config.draw_boost,
                    low_score_boost=base_config.low_score_boost,
                    outcome_alpha=base_config.outcome_alpha,
                    tail_dampener=td,
                )
                for td in TAIL_DAMPENERS
            ]
        if stage_name == "07_top_score_concentration_penalty":
            return [
                CandidateConfig(
                    "07_top_score_concentration_penalty",
                    "risk_penalties_on",
                    base_config.team_weights,
                    base_config.opp_weights,
                    max_goals=base_config.max_goals,
                    draw_boost=base_config.draw_boost,
                    low_score_boost=base_config.low_score_boost,
                    outcome_alpha=base_config.outcome_alpha,
                    tail_dampener=base_config.tail_dampener,
                    risk_penalties_enabled=True,
                )
            ]
        return [
            CandidateConfig(
                "08_segment_aware_calibration",
                "not_enabled_no_stable_fold_evidence",
                base_config.team_weights,
                base_config.opp_weights,
                max_goals=base_config.max_goals,
                draw_boost=base_config.draw_boost,
                low_score_boost=base_config.low_score_boost,
                outcome_alpha=base_config.outcome_alpha,
                tail_dampener=base_config.tail_dampener,
                risk_penalties_enabled=True,
            )
        ]

    for stage_name in stage_names:
        configs = build_stage_configs(stage_name, current.config)
        best = None
        for cfg in configs:
            result = evaluate_candidate(cfg, artifacts, baseline, thresholds)
            if best is None or result.selection_score < best.selection_score:
                best = result
        assert best is not None
        accepted, reason = ablation_accepts(current, best)
        best.accepted_by_ablation = accepted
        best.ablation_reason = reason
        if accepted:
            current = best
        ablations.append(ablation_summary(best, selected=accepted))
        print(
            f"    {stage_name}: best={best.config.name}, aw15={best.metrics['weighted_awmae_power_1_5']:.5f}, "
            f"selection={best.selection_score:.5f}, accepted={accepted}, reason={reason}"
        )

    return current, ablations, thresholds


def ablation_summary(result: CandidateResult, selected: bool) -> dict:
    return {
        "stage": result.config.stage,
        "candidate": result.config.name,
        "selected": selected,
        "accepted_by_ablation": result.accepted_by_ablation,
        "ablation_reason": result.ablation_reason,
        "selection_score": result.selection_score,
        **result.metrics,
        "distribution": {
            "score_ge5_share": result.distribution["score_ge5_share"],
            "top1_score_share": result.distribution["top1_score_share"],
            "top3_score_share": result.distribution["top3_score_share"],
            "draw_share": result.distribution["draw_share"],
        },
        "risk_components": result.risk_components,
        "config": asdict(result.config),
    }


# ===========================================================================
# 6. FINAL TRAINING, REPORTING, ACCEPTANCE
# ===========================================================================
def fit_final_predictions(train: pd.DataFrame, test: pd.DataFrame, feature_cols: list[str], chosen: CandidateResult, model_names: list[str]):
    print("\n[4] Training final static models on full train...")
    x_train, x_test = prepare_xy(train, test, feature_cols)
    w_train = train["train_weight"].values
    team_preds, names = fit_predict_models(x_train, train["team_goals"].values, w_train, x_test, N_ROUNDS_FINAL, "final team")
    opp_preds, opp_names = fit_predict_models(x_train, train["opp_goals"].values, w_train, x_test, N_ROUNDS_FINAL, "final opp")
    if names != opp_names:
        raise RuntimeError("Final team/opp model names diverged.")
    if names != model_names:
        print(f"    WARNING: validation model names {model_names} differ from final names {names}")
    outcome_probs = None
    if chosen.config.outcome_alpha > 0:
        print("      - final outcome classifier")
        outcome_probs = fit_outcome_classifier(x_train, train, w_train, x_test)
    lambda_team = apply_weights(team_preds, chosen.config.team_weights)
    lambda_opp = apply_weights(opp_preds, chosen.config.opp_weights)
    pred_t, pred_o = predict_scores(lambda_team, lambda_opp, chosen.config, outcome_probs=outcome_probs)
    return pred_t, pred_o, lambda_team, lambda_opp, names


def write_submission(test: pd.DataFrame, pred_t, pred_o):
    sample = pd.read_csv(SAMPLE_SUB)
    sub = pd.DataFrame({"Id": test["Id"].values, "team_goals": np.asarray(pred_t, dtype=int), "opp_goals": np.asarray(pred_o, dtype=int)})
    sub = sample[["Id"]].merge(sub, on="Id", how="left")
    if sub[["team_goals", "opp_goals"]].isna().any().any():
        raise ValueError("Submission has missing predictions after sample order merge.")
    sub["team_goals"] = sub["team_goals"].astype(int)
    sub["opp_goals"] = sub["opp_goals"].astype(int)
    sub.to_csv(OUTPUT_SUB, index=False)
    return sub


def local_submission_metrics(sub_path: Path, power=PRIMARY_POWER):
    if not (GT_PATH.exists() and sub_path.exists()):
        return None
    sub = pd.read_csv(sub_path)
    gt = pd.read_csv(GT_PATH)
    raw_test = pd.read_csv(TEST_RAW, usecols=["Id", "date", "tournament"])
    raw_test["date"] = pd.to_datetime(raw_test["date"])
    raw_test["metric_weight"] = raw_test["tournament"].map(TOURNAMENT_WEIGHT_MAP).fillna(DEFAULT_TOURNAMENT_WEIGHT)
    df = sub.merge(gt, on="Id", suffixes=("_pred", "_true")).merge(raw_test, on="Id", how="left")
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
        "per_year_distribution": yearly_distribution(df["team_goals_pred"].values, df["opp_goals_pred"].values, df["date"]),
    }


def evaluate_friend(power=PRIMARY_POWER):
    if not FRIEND_SUB.exists():
        return None
    if not GT_PATH.exists():
        friend = pd.read_csv(FRIEND_SUB)
        raw_test = pd.read_csv(TEST_RAW, usecols=["Id", "date"])
        merged = friend.merge(raw_test, on="Id", how="left")
        return {
            "rows": int(len(friend)),
            "distribution": score_distribution(friend["team_goals"].values, friend["opp_goals"].values),
            "per_year_distribution": yearly_distribution(friend["team_goals"].values, friend["opp_goals"].values, pd.to_datetime(merged["date"])),
            "ground_truth_available": False,
        }
    path_metrics = local_submission_metrics(FRIEND_SUB, power=power)
    if path_metrics is not None:
        path_metrics["ground_truth_available"] = True
    return path_metrics


def segment_metrics_for_validation(chosen: CandidateResult, artifacts: list[dict]) -> dict:
    pred_t_all, pred_o_all, true_t_all, true_o_all, val_all = [], [], [], [], []
    for artifact in artifacts:
        pred_t, pred_o, _, _ = aggregate_artifact_predictions(chosen.config, artifact)
        val = artifact["val"]
        pred_t_all.append(pred_t)
        pred_o_all.append(pred_o)
        true_t_all.append(val["team_goals"].values.astype(int))
        true_o_all.append(val["opp_goals"].values.astype(int))
        val_all.append(val)
    pred_t = np.concatenate(pred_t_all)
    pred_o = np.concatenate(pred_o_all)
    true_t = np.concatenate(true_t_all)
    true_o = np.concatenate(true_o_all)
    val = pd.concat(val_all, ignore_index=True)
    masks = {
        "major_tournament": val["tournament_weight"].values >= 1.50,
        "friendly": val["tournament"].astype(str).eq("Friendly").values,
        "qualifier": val["tournament"].astype(str).str.contains("qualification", case=False, na=False).values,
        "neutral": val["neutral"].fillna(0).astype(bool).values if "neutral" in val else np.zeros(len(val), dtype=bool),
        "non_neutral": ~val["neutral"].fillna(0).astype(bool).values if "neutral" in val else np.ones(len(val), dtype=bool),
        "balanced_elo": val["abs_elo_diff_risk"].fillna(0).le(50).values if "abs_elo_diff_risk" in val else np.zeros(len(val), dtype=bool),
        "strong_favorite": val["strong_favorite_risk"].fillna(0).astype(bool).values if "strong_favorite_risk" in val else np.zeros(len(val), dtype=bool),
    }
    out = {}
    for name, mask in masks.items():
        if int(mask.sum()) == 0:
            continue
        out[name] = {
            "rows": int(mask.sum()),
            **metrics_dict(pred_t[mask], pred_o[mask], true_t[mask], true_o[mask], weights=val["metric_weight"].values[mask], power=PRIMARY_POWER),
        }
    return out


def acceptance_decision(local15, friend15, final_dist, thresholds):
    checks = {
        "selected_purely_from_train_folds": True,
        "no_test_feature_update": True,
        "no_v5_import_output_anchor": True,
        "target_power_1_5_available": local15 is not None,
        "target_power_1_5_reached": False,
        "beats_friend_power_1_5": None,
        "outcome_not_below_friend": None,
        "tail_guard": final_dist["score_ge5_share"] <= thresholds["tail_threshold"] + 0.002,
        "top3_guard": final_dist["top3_score_share"] <= thresholds["top3_threshold"] + 0.02,
    }
    if local15 is None:
        return "VALIDATION_ONLY", checks
    checks["target_power_1_5_reached"] = local15["weighted_awmae"] <= TARGET_POWER_15_WEIGHTED
    if friend15 is not None and friend15.get("ground_truth_available", False):
        checks["beats_friend_power_1_5"] = local15["weighted_awmae"] < friend15["weighted_awmae"]
        checks["outcome_not_below_friend"] = local15["outcome_accuracy"] >= friend15["outcome_accuracy"] - 0.003
    if not checks["target_power_1_5_reached"]:
        return "TARGET_NOT_REACHED", checks
    if checks["beats_friend_power_1_5"] is False or checks["outcome_not_below_friend"] is False:
        return "NOT_ACCEPTED", checks
    if not (checks["tail_guard"] and checks["top3_guard"]):
        return "NOT_ACCEPTED", checks
    return "ACCEPTED_RISK_V1", checks


def write_outputs(
    train,
    test,
    feature_cols,
    dropped_cols,
    feature_groups,
    chosen: CandidateResult,
    ablations,
    thresholds,
    model_names,
    final_sub,
    final_dist,
    final_yearly,
    segment_summary,
    elapsed_minutes,
):
    local15 = local_submission_metrics(OUTPUT_SUB, power=PRIMARY_POWER)
    local13 = local_submission_metrics(OUTPUT_SUB, power=SECONDARY_POWER)
    friend15 = evaluate_friend(power=PRIMARY_POWER)
    friend13 = evaluate_friend(power=SECONDARY_POWER)
    decision, checks = acceptance_decision(local15, friend15, final_dist, thresholds)
    leakage_checklist = {
        "core_features_frozen": "assumed_safe_from_train_final_test_final",
        "test_state_update": "not_used",
        "v5_import": "not_used",
        "old_submission_anchor": "not_used",
        "friend_csv": "reporting_only",
        "test_ground_truth": "final_reporting_only_if_available",
        "context_target_encoding": "risk_noted_existing_ctx_features_may_use_random_kfold_encoding",
    }
    config = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "seed": SEED,
        "pipeline": "model_pipeline_risk_v1",
        "primary_metric_power": PRIMARY_POWER,
        "secondary_metric_power": SECONDARY_POWER,
        "target_weighted_awmae_power_1_5": TARGET_POWER_15_WEIGHTED,
        "static_no_test_update": True,
        "no_v5_import_declaration": True,
        "no_old_submission_anchor_declaration": True,
        "friend_csv_reporting_only": str(FRIEND_SUB),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "feature_count": int(len(feature_cols)),
        "feature_columns": feature_cols,
        "feature_groups": feature_groups,
        "dropped_columns": dropped_cols,
        "fold_definitions": FOLDS,
        "base_model_names": model_names,
        "base_model_availability": {
            "xgb": True,
            "hgb": True,
            "lgb": lgb is not None and ENABLE_LGB,
            "catboost": CatBoostRegressor is not None and ENABLE_CATBOOST,
        },
        "selected_config": asdict(chosen.config),
        "selected_validation_metrics": chosen.metrics,
        "selected_fold_metrics": chosen.fold_metrics,
        "selected_distribution": chosen.distribution,
        "selected_risk_components": chosen.risk_components,
        "selected_selection_score": chosen.selection_score,
        "ablation_table": ablations,
        "thresholds": {k: v for k, v in thresholds.items() if k != "friend_distribution"},
        "friend_distribution_from_predictions_only": thresholds.get("friend_distribution"),
        "segment_metrics": segment_summary,
        "per_year_validation_diagnostics": chosen.yearly_distribution,
        "per_year_test_diagnostics": final_yearly,
        "final_test_distribution": final_dist,
        "local_metrics_power_1_5": local15,
        "local_metrics_power_1_3": local13,
        "friend_metrics_power_1_5": friend15,
        "friend_metrics_power_1_3": friend13,
        "acceptance_decision": decision,
        "acceptance_checks": checks,
        "leakage_checklist": leakage_checklist,
        "elapsed_minutes": elapsed_minutes,
    }
    OUTPUT_CONFIG.write_text(json.dumps(config, indent=2, default=float), encoding="utf-8")

    lines = []
    lines.append("Risk V1 Static Pipeline Validation Report")
    lines.append("=" * 48)
    lines.append(f"Acceptance decision: {decision}")
    lines.append(f"Selected stage: {chosen.config.stage} / {chosen.config.name}")
    lines.append(f"Validation weighted AW-MAE power 1.5: {chosen.metrics['weighted_awmae_power_1_5']:.5f}")
    lines.append(f"Validation outcome/exact/gd: {chosen.metrics['outcome_accuracy']:.4f} / {chosen.metrics['exact_accuracy']:.4f} / {chosen.metrics['goal_diff_accuracy']:.4f}")
    lines.append(f"Selection score: {chosen.selection_score:.5f}")
    if local15 is not None:
        lines.append(f"Local weighted AW-MAE power 1.5: {local15['weighted_awmae']:.5f}")
        lines.append(f"Local outcome/exact/gd: {local15['outcome_accuracy']:.4f} / {local15['exact_accuracy']:.4f} / {local15['goal_diff_accuracy']:.4f}")
    if friend15 is not None and friend15.get("ground_truth_available", False) and local15 is not None:
        lines.append(f"Friend weighted AW-MAE power 1.5: {friend15['weighted_awmae']:.5f}")
        lines.append(f"Delta candidate - friend power 1.5: {local15['weighted_awmae'] - friend15['weighted_awmae']:+.5f}")
    lines.append("")
    lines.append("Leakage checklist")
    for k, v in leakage_checklist.items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("Validation folds")
    for fm in chosen.fold_metrics:
        lines.append(
            f"  {fm['fold_name']}: rows={fm['rows']}, weighted15={fm['weighted_awmae']:.5f}, "
            f"outcome={fm['outcome_accuracy']:.4f}, exact={fm['exact_accuracy']:.4f}"
        )
    lines.append("")
    lines.append("Ablation table")
    for item in ablations:
        lines.append(
            f"  {item['stage']}: {item['candidate']}, selected={item['selected']}, "
            f"weighted15={item['weighted_awmae_power_1_5']:.5f}, reason={item['ablation_reason']}"
        )
    lines.append("")
    lines.append("Risk objective breakdown")
    for k, v in chosen.risk_components.items():
        lines.append(f"  {k}: {v:.6f}")
    lines.append("")
    lines.append("Final test distribution")
    lines.append(json.dumps(final_dist, indent=2))
    lines.append("")
    lines.append("Per-year test diagnostics 2011-2026")
    for year, dist in sorted(final_yearly.items()):
        lines.append(
            f"  {year}: rows={dist['rows']}, top3={dist['top3_score_share']:.4f}, "
            f"draw={dist['draw_share']:.4f}, ge5={dist['score_ge5_share']:.4f}, avg_total={dist['avg_total_goals']:.3f}"
        )
    lines.append("")
    lines.append("Segment diagnostics")
    for name, metric in segment_summary.items():
        lines.append(
            f"  {name}: rows={metric['rows']}, weighted15={metric['weighted_awmae']:.5f}, "
            f"outcome={metric['outcome_accuracy']:.4f}, exact={metric['exact_accuracy']:.4f}"
        )
    lines.append("")
    lines.append("Benchmark comparison vs friend")
    if friend15 is None:
        lines.append("  Friend CSV missing or unreadable.")
    elif not friend15.get("ground_truth_available", False):
        lines.append("  Ground truth unavailable; distribution-only comparison.")
    elif local15 is not None:
        lines.append(f"  candidate weighted15: {local15['weighted_awmae']:.5f}")
        lines.append(f"  friend weighted15   : {friend15['weighted_awmae']:.5f}")
        lines.append(f"  candidate outcome   : {local15['outcome_accuracy']:.4f}")
        lines.append(f"  friend outcome      : {friend15['outcome_accuracy']:.4f}")
    if decision != "ACCEPTED_RISK_V1":
        lines.append("")
        lines.append(f"WARNING: final decision is {decision}; target power 1.5 <= {TARGET_POWER_15_WEIGHTED:.2f} may not be reached.")
    lines.append("")
    lines.append(f"Done in {elapsed_minutes:.1f} minutes. Wrote {OUTPUT_SUB.relative_to(BASE_DIR)} with {len(final_sub)} rows.")
    OUTPUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"    [OK] {OUTPUT_SUB.relative_to(BASE_DIR)}")
    print(f"    [OK] {OUTPUT_CONFIG.relative_to(BASE_DIR)}")
    print(f"    [OK] {OUTPUT_REPORT.relative_to(BASE_DIR)}")
    print(f"    Acceptance decision: {decision}")


def main():
    t0 = time.time()
    print("=" * 72)
    print("MODEL PIPELINE RISK V1 - STATIC RISK MINIMIZATION")
    print("=" * 72)
    train, test, feature_cols, dropped_cols, feature_groups = load_data()
    artifacts, model_names = build_validation_artifacts(train, feature_cols)
    chosen, ablations, thresholds = tune_candidates(artifacts, model_names)
    segment_summary = segment_metrics_for_validation(chosen, artifacts)
    pred_t, pred_o, lambda_t, lambda_o, final_model_names = fit_final_predictions(train, test, feature_cols, chosen, model_names)
    final_sub = write_submission(test, pred_t, pred_o)
    final_dist = score_distribution(final_sub["team_goals"].values, final_sub["opp_goals"].values)
    final_yearly = yearly_distribution(final_sub["team_goals"].values, final_sub["opp_goals"].values, test["date"])
    write_outputs(
        train,
        test,
        feature_cols,
        dropped_cols,
        feature_groups,
        chosen,
        ablations,
        thresholds,
        final_model_names,
        final_sub,
        final_dist,
        final_yearly,
        segment_summary,
        (time.time() - t0) / 60.0,
    )


if __name__ == "__main__":
    main()
