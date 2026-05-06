"""
Risk-minimizing static pipeline v4 static drift -- Gammafest Masa Kite Lagi.

This pipeline is intentionally independent from V5 outputs:
  * no import from model_pipeline_v5.py
  * no V3/V4/V5/V8 submission anchor or fallback
  * no test-period state update from predicted scores

Outputs:
  dataset/submission_risk_v4_static_drift.csv
  dataset/submission_risk_v4_static_drift_config.json
  dataset/submission_risk_v4_static_drift_validation_report.txt
"""

from __future__ import annotations

import json
import hashlib
import math
import os
import pickle
import time
import warnings
from collections import Counter
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import xgboost as xgb
import sklearn
from scipy.optimize import nnls
from scipy.stats import poisson
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover - optional dependency
    lgb = None

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
except ImportError:  # pragma: no cover - optional dependency
    CatBoostClassifier = None
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

OUTPUT_SUB = DATA_DIR / "submission_risk_v4_static_drift.csv"
OUTPUT_CONFIG = DATA_DIR / "submission_risk_v4_static_drift_config.json"
OUTPUT_REPORT = DATA_DIR / "submission_risk_v4_static_drift_validation_report.txt"
CACHE_DIR = DATA_DIR / "risk_v4_static_drift_cache"

SEED = 42
PRIMARY_POWER = 1.5
SECONDARY_POWER = 1.3
TARGET_POWER_15_WEIGHTED = 2.99883
TARGET_POWER_13_WEIGHTED = 2.47838
TARGET_OUTCOME = 0.59164
TARGET_EXACT = 0.10011
FAST_MODE = True

ENABLE_LGB = True
ENABLE_CATBOOST = False
N_ROUNDS_VAL = 80 if FAST_MODE else 420
N_ROUNDS_FINAL = 220 if FAST_MODE else 700

MAX_GOALS_CANDIDATES = [7, 8, 10]
TEAM_SCALES = [0.96, 1.00, 1.02]
OPP_SCALES = [0.96, 1.00, 1.02]
TEAM_BIASES = [-0.04, 0.00, 0.04]
OPP_BIASES = [-0.04, 0.00, 0.04]
TOTAL_LAMBDA_SCALES = [1.00, 1.02, 1.04, 1.06, 1.08]
DRAW_BOOSTS = [1.00]
LOW_DRAW_BOOSTS = [1.00]
LOW_DECISIVE_BOOSTS = [1.00, 1.10]
COMMON_SCORE_BOOSTS = [1.00, 1.05, 1.10]
HIGH_MARGIN_DAMPENERS = [0.90, 0.95, 1.00]
TAIL_DAMPENERS = [0.90, 0.95, 1.00]
OUTCOME_ALPHAS = [0.00, 0.10]
OUTCOME_THRESHOLDS = [0.0, 0.42, 0.50]
OUTCOME_PENALTY_ALPHAS = [0.00, 0.10]
EMPIRICAL_PRIOR_BETAS = [0.00, 0.10, 0.20, 0.30]
TOTAL_GOAL_DELTAS = [0.00, 0.10, 0.20]
GOAL_DIFF_ETAS = [0.00, 0.10, 0.20]
TAU_TOTALS = [1.2]
TAU_GDS = [1.2]
BALANCED_DRAW_BOOSTS = [1.00, 1.16]
BALANCED_TOTAL_SCALES = [0.94, 1.00]
NEUTRAL_TOTAL_SCALES = [0.96, 1.00]
NEUTRAL_DRAW_BOOSTS = [1.00, 1.12]
NEUTRAL_MARGIN_DAMPENERS = [0.90, 1.00]
FAST_FEATURE_PROFILES = ["stable_core", "static_drift", "static_priors"]
CACHE_SCHEMA_VERSION = "risk_v4_static_drift_hybrid_matrix_v2"
PRIOR_SMOOTHING = 50.0
PRIOR_MIN_COUNT = 5
TAU_STALENESS = 10.0
PRIOR_FEATURES = [
    "team_attack_prior",
    "team_defense_prior",
    "opp_attack_prior",
    "opp_defense_prior",
    "team_win_prior",
    "team_draw_prior",
    "team_loss_prior",
    "opp_win_prior",
    "opp_draw_prior",
    "opp_loss_prior",
    "tournament_avg_total_prior",
    "tournament_draw_prior",
    "confed_avg_total_prior",
    "confed_draw_prior",
    "gender_avg_total_prior",
    "gender_draw_prior",
    "gender_score_ge5_prior",
    "prior_trust_team",
    "prior_trust_opp",
    "frozen_feature_trust",
]

LOW_SCORE_CELLS = [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (0, 2), (2, 1), (1, 2)]
COMMON_LOW_SCORES = set(LOW_SCORE_CELLS)

FOLDS = [
    {
        "name": "fold_2003_2005",
        "train_end": "2002-12-31",
        "valid_start": "2003-01-01",
        "valid_end": "2005-12-31",
        "weight": 0.05,
    },
    {
        "name": "fold_2006_2008",
        "train_end": "2005-12-31",
        "valid_start": "2006-01-01",
        "valid_end": "2008-12-31",
        "weight": 0.15,
    },
    {
        "name": "fold_2009_2010",
        "train_end": "2008-12-31",
        "valid_start": "2009-01-01",
        "valid_end": "2010-12-31",
        "weight": 0.30,
    },
    {
        "name": "fold_2011_latest",
        "train_end": "2010-12-31",
        "valid_start": "2011-01-01",
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
    feature_profile: str = "all_features_safe"
    blend_method: str = "convex_grid"
    weight_cap: float = 0.70
    max_goals: int = 8
    team_scale: float = 1.0
    opp_scale: float = 1.0
    team_bias: float = 0.0
    opp_bias: float = 0.0
    total_lambda_scale: float = 1.0
    draw_boost: float = 1.0
    low_draw_boost: float = 1.0
    low_decisive_boost: float = 1.0
    tail_dampener: float = 1.0
    common_score_boost: float = 1.0
    high_margin_dampener: float = 1.0
    outcome_alpha: float = 0.0
    outcome_threshold: float = 0.0
    outcome_penalty_alpha: float = 0.0
    empirical_prior_beta: float = 0.0
    total_goal_delta: float = 0.0
    goal_diff_eta: float = 0.0
    tau_total: float = 1.2
    tau_gd: float = 1.2
    use_score_reranker: bool = False
    balanced_draw_boost: float = 1.0
    balanced_total_scale: float = 1.0
    neutral_total_scale: float = 1.0
    neutral_draw_boost: float = 1.0
    neutral_margin_dampener: float = 1.0
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
    low_draw_boost=1.0,
    low_decisive_boost=1.0,
    tail_dampener=1.0,
    common_score_boost=1.0,
    high_margin_dampener=1.0,
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
    if low_draw_boost != 1.0:
        for a, b in [(0, 0), (1, 1), (2, 2)]:
            if a < max_goals and b < max_goals:
                prob[:, a, b] *= low_draw_boost
    if low_decisive_boost != 1.0:
        for a, b in [(1, 0), (0, 1), (2, 1), (1, 2)]:
            if a < max_goals and b < max_goals:
                prob[:, a, b] *= low_decisive_boost
    if common_score_boost != 1.0:
        for a, b in [(1, 0), (0, 1), (1, 1), (2, 1), (1, 2)]:
            if a < max_goals and b < max_goals:
                prob[:, a, b] *= common_score_boost
    if high_margin_dampener != 1.0:
        margin_mask = np.abs(idx[:, None] - idx[None, :]) >= 3
        prob[:, margin_mask] *= high_margin_dampener
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


def apply_hybrid_matrix_weights(
    prob: np.ndarray,
    lambda_team: np.ndarray,
    lambda_opp: np.ndarray,
    config: CandidateConfig,
    score_priors: np.ndarray | None = None,
) -> np.ndarray:
    """Controlled score-cell reweighting from static priors and lambda-derived total/GD."""
    n, max_goals, _ = prob.shape
    idx = np.arange(max_goals)
    goals_sum = idx[:, None] + idx[None, :]
    goals_diff = idx[:, None] - idx[None, :]

    if score_priors is not None and config.empirical_prior_beta > 0:
        prior = np.asarray(score_priors, dtype=float)
        if prior.shape[1] < max_goals or prior.shape[2] < max_goals:
            raise ValueError("Score prior matrix is smaller than candidate max_goals.")
        prior = prior[:, :max_goals, :max_goals]
        prior = np.maximum(prior, 1e-8)
        prior /= np.maximum(prior.sum(axis=(1, 2), keepdims=True), 1e-12)
        prob *= np.power(prior, config.empirical_prior_beta)

    if config.total_goal_delta > 0:
        pred_total = np.clip(np.asarray(lambda_team) + np.asarray(lambda_opp), 0.2, 8.0)
        total_weight = np.exp(-np.abs(goals_sum[None, :, :] - pred_total[:, None, None]) / max(config.tau_total, 1e-6))
        prob *= np.power(total_weight, config.total_goal_delta)

    if config.goal_diff_eta > 0:
        pred_gd = np.clip(np.asarray(lambda_team) - np.asarray(lambda_opp), -6.0, 6.0)
        gd_weight = np.exp(-np.abs(goals_diff[None, :, :] - pred_gd[:, None, None]) / max(config.tau_gd, 1e-6))
        prob *= np.power(gd_weight, config.goal_diff_eta)

    prob /= np.maximum(prob.sum(axis=(1, 2), keepdims=True), 1e-12)
    return prob


def erm_from_probability(prob: np.ndarray, loss_tensor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    expected_loss = np.einsum("abij,nij->nab", loss_tensor, prob)
    flat = expected_loss.reshape(len(prob), -1).argmin(axis=1)
    max_goals = loss_tensor.shape[0]
    return (flat // max_goals).astype(int), (flat % max_goals).astype(int)


def erm_from_probability_with_outcome(
    prob: np.ndarray,
    loss_tensor: np.ndarray,
    outcome_probs,
    config: CandidateConfig,
) -> tuple[np.ndarray, np.ndarray]:
    expected_loss = np.einsum("abij,nij->nab", loss_tensor, prob)
    max_goals = loss_tensor.shape[0]
    idx = np.arange(max_goals)
    outcome_grid = np.where(idx[:, None] > idx[None, :], 2, np.where(idx[:, None] == idx[None, :], 1, 0))

    if outcome_probs is not None and config.outcome_penalty_alpha > 0:
        mismatch = 1.0 - np.take_along_axis(outcome_probs[:, None, None, :], outcome_grid[None, :, :, None], axis=3)[
            :, :, :, 0
        ]
        expected_loss = expected_loss + config.outcome_penalty_alpha * mismatch

    if outcome_probs is not None and config.outcome_threshold > 0:
        top_class = outcome_probs.argmax(axis=1)
        confidence = outcome_probs.max(axis=1)
        active = confidence >= config.outcome_threshold
        if np.any(active):
            allowed = outcome_grid[None, :, :] == top_class[:, None, None]
            expected_loss = np.where(active[:, None, None] & ~allowed, expected_loss + 1e6, expected_loss)

    flat = expected_loss.reshape(len(prob), -1).argmin(axis=1)
    return (flat // max_goals).astype(int), (flat % max_goals).astype(int)


def predict_scores(
    lambda_team,
    lambda_opp,
    config: CandidateConfig,
    outcome_probs=None,
    df: pd.DataFrame | None = None,
    score_priors=None,
):
    lambda_team = (np.asarray(lambda_team, dtype=float) * config.team_scale + config.team_bias) * config.total_lambda_scale
    lambda_opp = (np.asarray(lambda_opp, dtype=float) * config.opp_scale + config.opp_bias) * config.total_lambda_scale
    lambda_team = np.clip(lambda_team, 1e-5, 12.0)
    lambda_opp = np.clip(lambda_opp, 1e-5, 12.0)
    prob = score_probability_matrix(
        lambda_team,
        lambda_opp,
        config.max_goals,
        config.draw_boost,
        config.low_draw_boost,
        config.low_decisive_boost,
        config.tail_dampener,
        common_score_boost=config.common_score_boost,
        high_margin_dampener=config.high_margin_dampener,
        outcome_probs=outcome_probs,
        outcome_alpha=config.outcome_alpha,
    )
    prob = apply_hybrid_matrix_weights(prob, lambda_team, lambda_opp, config, score_priors=score_priors)
    loss_tensor = get_loss_tensor(config.max_goals, PRIMARY_POWER)
    pred_t, pred_o = erm_from_probability_with_outcome(prob, loss_tensor, outcome_probs, config)

    if df is None:
        return pred_t, pred_o

    if "abs_elo_diff_risk" in df.columns and (config.balanced_draw_boost != 1.0 or config.balanced_total_scale != 1.0):
        mask = df["abs_elo_diff_risk"].fillna(9999).values <= 50
        if np.any(mask):
            bt, bo = predict_segment_scores(
                lambda_team[mask] * config.balanced_total_scale,
                lambda_opp[mask] * config.balanced_total_scale,
                config,
                None if outcome_probs is None else outcome_probs[mask],
                draw_boost=config.draw_boost * config.balanced_draw_boost,
                allowed_cells=[(0, 0), (1, 1), (2, 2), (1, 0), (0, 1), (2, 1), (1, 2)],
                score_priors=None if score_priors is None else score_priors[mask],
            )
            pred_t[mask] = bt
            pred_o[mask] = bo

    if "neutral" in df.columns and (
        config.neutral_total_scale != 1.0 or config.neutral_draw_boost != 1.0 or config.neutral_margin_dampener != 1.0
    ):
        mask = df["neutral"].fillna(0).astype(bool).values
        if np.any(mask):
            nt, no = predict_segment_scores(
                lambda_team[mask] * config.neutral_total_scale,
                lambda_opp[mask] * config.neutral_total_scale,
                config,
                None if outcome_probs is None else outcome_probs[mask],
                draw_boost=config.draw_boost * config.neutral_draw_boost,
                high_margin_dampener=config.high_margin_dampener * config.neutral_margin_dampener,
                score_priors=None if score_priors is None else score_priors[mask],
            )
            pred_t[mask] = nt
            pred_o[mask] = no
    return pred_t, pred_o


def predict_segment_scores(
    lambda_team,
    lambda_opp,
    config: CandidateConfig,
    outcome_probs=None,
    draw_boost=1.0,
    high_margin_dampener=1.0,
    allowed_cells: list[tuple[int, int]] | None = None,
    score_priors=None,
):
    prob = score_probability_matrix(
        lambda_team,
        lambda_opp,
        config.max_goals,
        draw_boost,
        config.low_draw_boost,
        config.low_decisive_boost,
        config.tail_dampener,
        common_score_boost=config.common_score_boost,
        high_margin_dampener=high_margin_dampener,
        outcome_probs=outcome_probs,
        outcome_alpha=config.outcome_alpha,
    )
    prob = apply_hybrid_matrix_weights(prob, lambda_team, lambda_opp, config, score_priors=score_priors)
    loss_tensor = get_loss_tensor(config.max_goals, PRIMARY_POWER)
    expected_loss = np.einsum("abij,nij->nab", loss_tensor, prob)
    if allowed_cells is not None:
        allowed = np.zeros((config.max_goals, config.max_goals), dtype=bool)
        for a, b in allowed_cells:
            if a < config.max_goals and b < config.max_goals:
                allowed[a, b] = True
        expected_loss = np.where(allowed[None, :, :], expected_loss, expected_loss + 1e6)
    if outcome_probs is not None and config.outcome_penalty_alpha > 0:
        idx = np.arange(config.max_goals)
        outcome_grid = np.where(idx[:, None] > idx[None, :], 2, np.where(idx[:, None] == idx[None, :], 1, 0))
        mismatch = 1.0 - np.take_along_axis(outcome_probs[:, None, None, :], outcome_grid[None, :, :, None], axis=3)[
            :, :, :, 0
        ]
        expected_loss = expected_loss + config.outcome_penalty_alpha * mismatch
    flat = expected_loss.reshape(len(prob), -1).argmin(axis=1)
    return (flat // config.max_goals).astype(int), (flat % config.max_goals).astype(int)


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
            0.05 * max(0.0, dist["top3_score_share"] - 0.68) ** 2
            + 0.10 * max(0.0, dist["score_ge5_share"] - 0.04) ** 2
            + 0.08 * max(0.0, dist["avg_total_goals"] - 2.85) ** 2
        )
    return float(np.mean(vals))


def validation_segment_snapshot(pred_t, pred_o, true_t, true_o, weights, df: pd.DataFrame) -> dict:
    out = {}
    masks = {
        "balanced_elo": df["abs_elo_diff_risk"].fillna(9999).le(50).values
        if "abs_elo_diff_risk" in df.columns
        else np.zeros(len(df), dtype=bool),
        "neutral": df["neutral"].fillna(0).astype(bool).values if "neutral" in df.columns else np.zeros(len(df), dtype=bool),
        "non_neutral": ~df["neutral"].fillna(0).astype(bool).values if "neutral" in df.columns else np.ones(len(df), dtype=bool),
    }
    for name, mask in masks.items():
        if int(mask.sum()) == 0:
            continue
        out[name] = metrics_dict(
            np.asarray(pred_t)[mask],
            np.asarray(pred_o)[mask],
            np.asarray(true_t)[mask],
            np.asarray(true_o)[mask],
            weights=np.asarray(weights)[mask],
            power=PRIMARY_POWER,
        )
        out[name]["rows"] = int(mask.sum())
    return out


# ===========================================================================
# 3. DATA AND FEATURES
# ===========================================================================
def raw_usecols(path: Path) -> list[str]:
    available = pd.read_csv(path, nrows=0).columns.tolist()
    wanted = [
        "Id",
        "date",
        "gender",
        "team",
        "opponent",
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
    gender = out.get("gender", pd.Series("M", index=out.index)).fillna("M").astype(str).str.upper().str.strip()
    out["is_women_match"] = (gender == "W").astype(int)
    out["is_men_match"] = (gender == "M").astype(int)

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
    out["year_since_2011_drift"] = out["year_risk"] - 2011
    out["year_since_2000_drift"] = out["year_risk"] - 2000
    out["post_2018_drift"] = (out["year_risk"] >= 2018).astype(int)
    out["post_2020_drift"] = (out["year_risk"] >= 2020).astype(int)
    out["post_2022_drift"] = (out["year_risk"] >= 2022).astype(int)
    out["era_2012_2017_drift"] = ((out["year_risk"] >= 2012) & (out["year_risk"] <= 2017)).astype(int)
    out["era_2018_plus_drift"] = (out["year_risk"] >= 2018).astype(int)
    out["frozen_strength_staleness"] = np.maximum(0, out["year_risk"] - 2011)
    tournament = out["tournament"].astype(str)
    out["is_friendly_risk"] = (tournament == "Friendly").astype(int)
    out["is_qualifier_risk"] = tournament.str.contains("qualification", case=False, na=False).astype(int)
    out["is_major_tournament_risk"] = (out["tournament_weight"] >= 1.50).astype(int)
    out["high_importance_tournament_risk"] = (out["tournament_weight"] >= 1.70).astype(int)
    lower_tournament = tournament.str.lower()
    out["tournament_type_friendly"] = out["is_friendly_risk"]
    out["tournament_type_qualifier"] = out["is_qualifier_risk"]
    out["tournament_type_major"] = out["is_major_tournament_risk"]
    out["tournament_type_nations_league"] = lower_tournament.str.contains("nations league", na=False).astype(int)
    out["tournament_type_continental"] = lower_tournament.str.contains("euro|copa|asian|african|concacaf|oceania|gold cup", na=False).astype(int)
    out["tournament_type_olympic"] = lower_tournament.str.contains("olympic", na=False).astype(int)
    out["tournament_type_youth"] = lower_tournament.str.contains("u-?17|u-?20|under", regex=True, na=False).astype(int)
    out["tournament_type_women_major"] = out["is_women_match"] * out["tournament_type_major"]
    out["tournament_type_women_qualifier"] = out["is_women_match"] * out["tournament_type_qualifier"]
    out["tournament_type_women_friendly"] = out["is_women_match"] * out["tournament_type_friendly"]
    out["tournament_type_women_continental"] = out["is_women_match"] * out["tournament_type_continental"]
    if "abs_elo_diff_risk" in out.columns:
        out["women_x_abs_elo_diff"] = out["is_women_match"] * out["abs_elo_diff_risk"]
        out["abs_elo_diff_x_year_since_2011"] = out["abs_elo_diff_risk"] * out["year_since_2011_drift"]
    out["women_x_year_since_2011"] = out["is_women_match"] * out["year_since_2011_drift"]
    out["women_x_post_2018"] = out["is_women_match"] * out["post_2018_drift"]
    out["women_x_post_2020"] = out["is_women_match"] * out["post_2020_drift"]
    out["women_x_tournament_weight"] = out["is_women_match"] * out["tournament_weight"]
    out["women_x_neutral"] = out["is_women_match"] * out.get("neutral", 0)
    out["women_x_is_home"] = out["is_women_match"] * out.get("is_home", 0)
    out["women_x_friendly"] = out["is_women_match"] * out["is_friendly_risk"]
    out["women_x_qualifier"] = out["is_women_match"] * out["is_qualifier_risk"]
    out["women_x_major"] = out["is_women_match"] * out["is_major_tournament_risk"]
    out["women_x_strong_favorite"] = out["is_women_match"] * out.get("strong_favorite_risk", 0)
    out["women_x_balanced_match"] = out["is_women_match"] * out.get("balanced_match_risk", 0)
    out["women_x_frozen_strength_staleness"] = out["is_women_match"] * out["frozen_strength_staleness"]
    if "expected_total_goals_proxy_risk" in out.columns:
        out["women_x_expected_total_goals_proxy"] = out["is_women_match"] * out["expected_total_goals_proxy_risk"]
    out["neutral_x_year_since_2011"] = out.get("neutral", 0) * out["year_since_2011_drift"]
    out["neutral_x_post_2018"] = out.get("neutral", 0) * out["post_2018_drift"]
    out["qualifier_x_year_since_2011"] = out["is_qualifier_risk"] * out["year_since_2011_drift"]
    out["friendly_x_year_since_2011"] = out["is_friendly_risk"] * out["year_since_2011_drift"]
    out["major_x_year_since_2011"] = out["is_major_tournament_risk"] * out["year_since_2011_drift"]
    out["tournament_weight_x_year_since_2011"] = out["tournament_weight"] * out["year_since_2011_drift"]
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


def normalize_name_series(s: pd.Series) -> pd.Series:
    return (
        s.fillna("unknown")
        .astype(str)
        .str.lower()
        .str.replace("'", "", regex=False)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def smooth_mean_map(df: pd.DataFrame, key: str, value: str, global_mean: float, smoothing=PRIOR_SMOOTHING):
    agg = df.groupby(key)[value].agg(["sum", "count"])
    return ((agg["sum"] + global_mean * smoothing) / (agg["count"] + smoothing)).to_dict(), agg["count"].to_dict()


def smooth_rate_map(df: pd.DataFrame, key: str, value: str, global_mean: float, smoothing=PRIOR_SMOOTHING):
    return smooth_mean_map(df, key, value, global_mean, smoothing)


def add_static_priors(train_source: pd.DataFrame, apply_df: pd.DataFrame) -> pd.DataFrame:
    src = train_source.copy()
    out = apply_df.copy()
    for frame in (src, out):
        gender = frame.get("gender", pd.Series("M", index=frame.index)).fillna("M").astype(str).str.upper().str.strip()
        frame["_team_key"] = gender + "::" + normalize_name_series(frame["team"])
        frame["_opp_key"] = gender + "::" + normalize_name_series(frame["opponent"])
        frame["_gender_key"] = gender
        frame["_gender_tournament_key"] = gender + "::" + frame["tournament"].fillna("unknown").astype(str).str.lower().str.strip()
        frame["_gender_confed_key"] = gender + "::" + frame.get("confederation_team", pd.Series("unknown", index=frame.index)).fillna("unknown").astype(str).str.lower().str.strip()

    src["total_goals_prior_target"] = src["team_goals"] + src["opp_goals"]
    src["is_draw_prior_target"] = (src["team_goals"] == src["opp_goals"]).astype(float)
    src["is_win_prior_target"] = (src["team_goals"] > src["opp_goals"]).astype(float)
    src["is_loss_prior_target"] = (src["team_goals"] < src["opp_goals"]).astype(float)
    src["score_ge5_prior_target"] = ((src["team_goals"] >= 5) | (src["opp_goals"] >= 5)).astype(float)

    global_for = float(src["team_goals"].mean())
    global_against = float(src["opp_goals"].mean())
    global_total = float(src["total_goals_prior_target"].mean())
    global_draw = float(src["is_draw_prior_target"].mean())
    global_win = float(src["is_win_prior_target"].mean())
    global_loss = float(src["is_loss_prior_target"].mean())
    global_ge5 = float(src["score_ge5_prior_target"].mean())

    attack_map, attack_count = smooth_mean_map(src, "_team_key", "team_goals", global_for)
    defense_map, defense_count = smooth_mean_map(src, "_team_key", "opp_goals", global_against)
    win_map, _ = smooth_rate_map(src, "_team_key", "is_win_prior_target", global_win)
    draw_map, _ = smooth_rate_map(src, "_team_key", "is_draw_prior_target", global_draw)
    loss_map, _ = smooth_rate_map(src, "_team_key", "is_loss_prior_target", global_loss)
    tournament_total_map, _ = smooth_mean_map(src, "_gender_tournament_key", "total_goals_prior_target", global_total)
    tournament_draw_map, _ = smooth_rate_map(src, "_gender_tournament_key", "is_draw_prior_target", global_draw)
    confed_total_map, _ = smooth_mean_map(src, "_gender_confed_key", "total_goals_prior_target", global_total)
    confed_draw_map, _ = smooth_rate_map(src, "_gender_confed_key", "is_draw_prior_target", global_draw)
    gender_total_map, _ = smooth_mean_map(src, "_gender_key", "total_goals_prior_target", global_total)
    gender_draw_map, _ = smooth_rate_map(src, "_gender_key", "is_draw_prior_target", global_draw)
    gender_ge5_map, _ = smooth_rate_map(src, "_gender_key", "score_ge5_prior_target", global_ge5)

    def mapped(key_col, mapping, default):
        return out[key_col].map(mapping).fillna(default).astype(float)

    out["team_attack_prior"] = mapped("_team_key", attack_map, global_for).clip(global_for * 0.65, global_for * 1.60)
    out["team_defense_prior"] = mapped("_team_key", defense_map, global_against).clip(global_against * 0.65, global_against * 1.60)
    out["opp_attack_prior"] = mapped("_opp_key", attack_map, global_for).clip(global_for * 0.65, global_for * 1.60)
    out["opp_defense_prior"] = mapped("_opp_key", defense_map, global_against).clip(global_against * 0.65, global_against * 1.60)
    out["team_win_prior"] = mapped("_team_key", win_map, global_win)
    out["team_draw_prior"] = mapped("_team_key", draw_map, global_draw).clip(0.05, 0.35)
    out["team_loss_prior"] = mapped("_team_key", loss_map, global_loss)
    out["opp_win_prior"] = mapped("_opp_key", win_map, global_win)
    out["opp_draw_prior"] = mapped("_opp_key", draw_map, global_draw).clip(0.05, 0.35)
    out["opp_loss_prior"] = mapped("_opp_key", loss_map, global_loss)
    out["tournament_avg_total_prior"] = mapped("_gender_tournament_key", tournament_total_map, global_total).clip(1.8, 3.6)
    out["tournament_draw_prior"] = mapped("_gender_tournament_key", tournament_draw_map, global_draw).clip(0.05, 0.35)
    out["confed_avg_total_prior"] = mapped("_gender_confed_key", confed_total_map, global_total).clip(1.8, 3.6)
    out["confed_draw_prior"] = mapped("_gender_confed_key", confed_draw_map, global_draw).clip(0.05, 0.35)
    out["gender_avg_total_prior"] = mapped("_gender_key", gender_total_map, global_total).clip(1.8, 3.6)
    out["gender_draw_prior"] = mapped("_gender_key", gender_draw_map, global_draw).clip(0.05, 0.35)
    out["gender_score_ge5_prior"] = mapped("_gender_key", gender_ge5_map, global_ge5)
    out["prior_trust_team"] = out["_team_key"].map(attack_count).fillna(0).astype(float) / (out["_team_key"].map(attack_count).fillna(0).astype(float) + PRIOR_SMOOTHING)
    out["prior_trust_opp"] = out["_opp_key"].map(attack_count).fillna(0).astype(float) / (out["_opp_key"].map(attack_count).fillna(0).astype(float) + PRIOR_SMOOTHING)
    out["frozen_feature_trust"] = np.exp(-out.get("frozen_strength_staleness", 0) / TAU_STALENESS)
    out.drop(columns=[c for c in out.columns if c.startswith("_")], inplace=True, errors="ignore")
    return out


def score_prior_segment_key(df: pd.DataFrame) -> pd.Series:
    gender = np.where(df.get("is_women_match", 0).astype(int).values == 1, "W", "M")
    tournament_type = np.select(
        [
            df.get("tournament_type_friendly", 0).astype(bool).values,
            df.get("tournament_type_qualifier", 0).astype(bool).values,
            df.get("tournament_type_major", 0).astype(bool).values,
            df.get("tournament_type_continental", 0).astype(bool).values,
            df.get("tournament_type_youth", 0).astype(bool).values,
        ],
        ["friendly", "qualifier", "major", "continental", "youth"],
        default="other",
    )
    neutral = np.where(df.get("neutral", 0).fillna(0).astype(int).values == 1, "neutral", "nonneutral")
    abs_elo = df.get("abs_elo_diff_risk", pd.Series(np.nan, index=df.index)).fillna(9999).astype(float).values
    strength = np.where(abs_elo <= 50, "balanced", np.where(abs_elo >= 250, "mismatch", "middle"))
    return pd.Series(gender + "|" + tournament_type + "|" + neutral + "|" + strength, index=df.index)


def build_score_prior_lookup(
    train_source: pd.DataFrame,
    max_goals: int = 10,
    smoothing: float = 60.0,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    score_t = np.clip(train_source["team_goals"].astype(int).values, 0, max_goals - 1)
    score_o = np.clip(train_source["opp_goals"].astype(int).values, 0, max_goals - 1)
    global_counts = np.ones((max_goals, max_goals), dtype=float) * 0.25
    for a, b in zip(score_t, score_o):
        global_counts[a, b] += 1.0
    global_prior = global_counts / global_counts.sum()

    segment_keys = score_prior_segment_key(train_source)
    lookup: dict[str, np.ndarray] = {}
    for key, idx in segment_keys.groupby(segment_keys).groups.items():
        counts = np.zeros((max_goals, max_goals), dtype=float)
        rows = np.asarray(list(idx), dtype=int)
        for a, b in zip(score_t[rows], score_o[rows]):
            counts[a, b] += 1.0
        prior = (counts + global_prior * smoothing) / (len(rows) + smoothing)
        prior = np.maximum(prior, 1e-8)
        prior /= prior.sum()
        lookup[str(key)] = prior
    return global_prior, lookup


def score_prior_matrices(train_source: pd.DataFrame, apply_df: pd.DataFrame, max_goals: int = 10) -> np.ndarray:
    global_prior, lookup = build_score_prior_lookup(train_source, max_goals=max_goals)
    keys = score_prior_segment_key(apply_df).astype(str).values
    priors = np.empty((len(apply_df), max_goals, max_goals), dtype=np.float32)
    for i, key in enumerate(keys):
        priors[i] = lookup.get(key, global_prior)
    priors /= np.maximum(priors.sum(axis=(1, 2), keepdims=True), 1e-12)
    return priors


def is_existing_te_column(col: str) -> bool:
    lower = col.lower()
    return lower.endswith("_te_ctx") or "_te_" in lower or lower.endswith("_target_enc")


def build_feature_profiles(train: pd.DataFrame, test: pd.DataFrame, candidate_cols: list[str]) -> tuple[dict, dict, dict]:
    numeric_cols = [
        c
        for c in candidate_cols
        if pd.api.types.is_numeric_dtype(train[c]) and pd.api.types.is_numeric_dtype(test[c])
    ]
    numeric_set = set(numeric_cols)
    dropped_base = {c: "non_numeric_or_missing_from_test" for c in candidate_cols if c not in numeric_set}

    core = sorted([c for c in numeric_cols if c.endswith("_feat")])
    metadata_names = {
        "neutral",
        "is_home",
        "tournament_weight",
        "year_risk",
        "month_risk",
        "is_friendly_risk",
        "is_qualifier_risk",
        "is_major_tournament_risk",
        "high_importance_tournament_risk",
        "is_women_match",
        "is_men_match",
    }
    metadata = sorted([c for c in numeric_cols if c in metadata_names])
    context_no_te_names = {
        "travel_stress_diff_ctx",
        "altitude_shock_team_ctx",
        "altitude_shock_opp_ctx",
        "temp_stress_ctx",
        "log_gdp_diff_ctx",
        "log_pop_diff_ctx",
        "venue_country_freq_ctx",
        "population_team",
        "population_opp",
        "gdp_per_capita_team",
        "gdp_per_capita_opp",
        "altitude_venue",
        "distance_travel_team",
        "distance_travel_opp",
        "temperature_venue",
    }
    context_no_te = sorted([c for c in numeric_cols if c in context_no_te_names and not is_existing_te_column(c)])
    risk_cols = sorted([c for c in numeric_cols if c.endswith("_risk") and c not in {"year_risk", "month_risk"}])
    drift_cols = sorted([c for c in numeric_cols if c.endswith("_drift") or c.startswith("women_x_") or c.startswith("tournament_type_") or c.endswith("_x_year_since_2011") or c in {"frozen_strength_staleness", "neutral_x_post_2018"}])
    prior_cols = sorted([c for c in PRIOR_FEATURES if c in numeric_cols])
    existing_te = sorted([c for c in numeric_cols if is_existing_te_column(c)])

    safe_excluded = {"sample_weight", "train_weight", "metric_weight", "time_weight", "is_test"}
    all_safe = sorted([c for c in numeric_cols if c not in safe_excluded])
    all_safe_no_te = sorted([c for c in all_safe if not is_existing_te_column(c)])

    profiles = {
        "stable_core": sorted(set(core)),
        "static_drift": sorted(set(core + metadata + risk_cols + drift_cols)),
        "static_drift_plus_context": sorted(set(core + metadata + context_no_te + risk_cols + drift_cols)),
        "static_priors": sorted(set(core + metadata + context_no_te + risk_cols + drift_cols + prior_cols)),
        "all_safe_no_random_te": sorted(set(all_safe_no_te + prior_cols)),
    }
    if FAST_MODE:
        profiles = {k: v for k, v in profiles.items() if k in FAST_FEATURE_PROFILES}

    dropped_by_profile = {}
    for name, cols in profiles.items():
        drop = dict(dropped_base)
        for col in numeric_cols:
            if col not in cols:
                if name.endswith("_no_te") and is_existing_te_column(col):
                    drop[col] = "excluded_existing_target_encoding_for_temporal_safety"
                elif name == "stable_core":
                    drop[col] = "excluded_by_stable_core_profile"
                elif is_existing_te_column(col):
                    drop[col] = "excluded_existing_target_encoding_for_temporal_safety"
                else:
                    drop[col] = "excluded_by_feature_profile"
        dropped_by_profile[name] = drop

    feature_groups_by_profile = {}
    for name, cols in profiles.items():
        feature_groups_by_profile[name] = {
            "core_historical": [c for c in cols if c.endswith("_feat")],
            "context_no_te": [c for c in cols if c in context_no_te],
            "existing_te": [c for c in cols if c in existing_te],
            "match_metadata": [c for c in cols if c in metadata],
            "risk_features": [c for c in cols if c in risk_cols],
            "drift_features": [c for c in cols if c in drift_cols],
            "static_priors": [c for c in cols if c in prior_cols],
        }
    return profiles, dropped_by_profile, feature_groups_by_profile


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
    # Full-train priors are placeholders for column discovery and final inference.
    # Fold validation recomputes these priors from each fold train split.
    train = add_static_priors(train, train)
    test = add_static_priors(train, test)
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
    feature_profiles, dropped_by_profile, feature_groups_by_profile = build_feature_profiles(train, test, candidate_cols)
    all_profile_cols = sorted(set().union(*[set(v) for v in feature_profiles.values()]))
    train[all_profile_cols] = train[all_profile_cols].replace([np.inf, -np.inf], np.nan)
    test[all_profile_cols] = test[all_profile_cols].replace([np.inf, -np.inf], np.nan)

    print(f"    Train: {train.shape} | Test: {test.shape} | Profiles: {', '.join(feature_profiles)}")
    for name, cols in feature_profiles.items():
        print(f"      - {name}: {len(cols)} features")
    return train, test, feature_profiles, dropped_by_profile, feature_groups_by_profile


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


class XGBLogGoalModel:
    name = "xgb_log1p_sq"

    def __init__(self, rounds: int):
        self.rounds = rounds
        self.model = None

    def fit(self, x, y, weight):
        params = dict(XGB_POISSON_PARAMS)
        params.update({"objective": "reg:squarederror", "eval_metric": "rmse", "max_depth": 5, "learning_rate": 0.035})
        self.model = xgb.train(params, xgb.DMatrix(x, label=np.log1p(y), weight=weight), num_boost_round=self.rounds, verbose_eval=False)
        return self

    def predict(self, x):
        return np.expm1(self.model.predict(xgb.DMatrix(x)))


class HGBSquaredGoalModel:
    name = "hgb_squared"

    def __init__(self, rounds: int):
        self.model = HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.035,
            max_iter=rounds,
            max_leaf_nodes=31,
            min_samples_leaf=75,
            l2_regularization=1.2,
            random_state=SEED,
            early_stopping=False,
        )

    def fit(self, x, y, weight):
        self.model.fit(x, np.log1p(y), sample_weight=weight)
        return self

    def predict(self, x):
        return np.expm1(self.model.predict(x))


def model_factories(rounds: int):
    factories = [
        lambda: XGBPoissonModel(rounds),
        lambda: HGBPoissonModel(rounds),
        lambda: XGBLogGoalModel(rounds),
        lambda: HGBSquaredGoalModel(rounds),
    ]
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
    y = outcome_target(train_df["team_goals"].values, train_df["opp_goals"].values)
    probas = []

    clf = HistGradientBoostingClassifier(
        learning_rate=0.035,
        max_iter=220 if FAST_MODE else 420,
        max_leaf_nodes=31,
        min_samples_leaf=70,
        l2_regularization=1.5,
        random_state=SEED,
        early_stopping=False,
    )
    clf.fit(x_train, y, sample_weight=w_train)
    raw = clf.predict_proba(x_pred)
    out = np.full((len(x_pred), 3), 1e-8, dtype=float)
    for i, cls in enumerate(clf.classes_):
        out[:, int(cls)] = raw[:, i]
    out /= out.sum(axis=1, keepdims=True)
    probas.append(out)

    try:
        xgb_params = {
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
        xgb_clf = xgb.train(xgb_params, xgb.DMatrix(x_train, label=y, weight=w_train), num_boost_round=180, verbose_eval=False)
        probas.append(np.clip(xgb_clf.predict(xgb.DMatrix(x_pred)), 1e-8, 1.0))
    except Exception:
        pass

    if ENABLE_CATBOOST and CatBoostClassifier is not None:
        try:
            cat = CatBoostClassifier(
                loss_function="MultiClass",
                iterations=180 if FAST_MODE else 320,
                depth=5,
                learning_rate=0.04,
                l2_leaf_reg=8.0,
                random_seed=SEED,
                verbose=False,
                allow_writing_files=False,
            )
            cat.fit(x_train, y, sample_weight=w_train)
            raw = cat.predict_proba(x_pred)
            cat_out = np.full((len(x_pred), 3), 1e-8, dtype=float)
            for i, cls in enumerate(cat.classes_):
                cat_out[:, int(cls)] = raw[:, i]
            cat_out /= cat_out.sum(axis=1, keepdims=True)
            probas.append(cat_out)
        except Exception:
            pass

    out = np.mean(probas, axis=0)
    out /= out.sum(axis=1, keepdims=True)
    return np.clip(out, 1e-8, 1.0)


# ===========================================================================
# 5. VALIDATION ARTIFACTS AND CANDIDATE EVALUATION
# ===========================================================================
def fold_data(train: pd.DataFrame, fold: dict):
    train_mask = train["date"] <= pd.Timestamp(fold["train_end"])
    val_mask = (train["date"] >= pd.Timestamp(fold["valid_start"])) & (train["date"] <= pd.Timestamp(fold["valid_end"]))
    return train.loc[train_mask].copy(), train.loc[val_mask].copy()


def bucket_series(df: pd.DataFrame, name: str) -> pd.Series:
    if name == "gender":
        return df.get("gender", pd.Series("M", index=df.index)).fillna("M").astype(str).str.upper()
    if name == "tournament_type":
        vals = np.select(
            [
                df.get("is_friendly_risk", 0).astype(bool),
                df.get("is_qualifier_risk", 0).astype(bool),
                df.get("is_major_tournament_risk", 0).astype(bool),
                df.get("tournament_type_nations_league", 0).astype(bool),
            ],
            ["friendly", "qualifier", "major", "nations_league"],
            default="other",
        )
        return pd.Series(vals, index=df.index)
    if name == "neutral":
        return df.get("neutral", pd.Series(0, index=df.index)).fillna(0).astype(int).astype(str)
    if name == "weight_bucket":
        return pd.cut(df.get("tournament_weight", pd.Series(DEFAULT_TOURNAMENT_WEIGHT, index=df.index)), bins=[0, 1.0, 1.4, 1.7, 10], labels=["low", "mid", "high", "elite"], include_lowest=True).astype(str)
    if name == "elo_bucket":
        vals = df.get("abs_elo_diff_risk", pd.Series(0, index=df.index)).fillna(0)
        return pd.cut(vals, bins=[-1, 50, 150, 300, 10000], labels=["balanced", "edge", "favorite", "mismatch"]).astype(str)
    if name == "confed":
        return df.get("confederation_team", pd.Series("unknown", index=df.index)).fillna("unknown").astype(str)
    return pd.Series("all", index=df.index)


def compute_drift_weights(val: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    dims = ["gender", "tournament_type", "neutral", "weight_bucket", "elo_bucket", "confed"]
    weights = np.ones(len(val), dtype=float)
    for dim in dims:
        val_bin = bucket_series(val, dim)
        test_bin = bucket_series(test, dim)
        val_share = val_bin.value_counts(normalize=True).to_dict()
        test_share = test_bin.value_counts(normalize=True).to_dict()
        ratios = {k: np.clip(test_share.get(k, 0.0) / max(val_share.get(k, 0.0), 1e-6), 0.5, 2.0) for k in set(val_share) | set(test_share)}
        weights *= val_bin.map(ratios).fillna(1.0).values
    weights = np.clip(weights, 0.4, 2.5)
    weights /= max(weights.mean(), 1e-12)
    return weights


def file_mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


def feature_cache_key(profile_name: str, feature_cols: list[str], train: pd.DataFrame, test: pd.DataFrame) -> tuple[str, dict]:
    payload = {
        "pipeline_version": CACHE_SCHEMA_VERSION,
        "feature_profile_name": profile_name,
        "feature_cols": feature_cols,
        "fold_definitions": FOLDS,
        "model_parameters": {
            "xgb": XGB_POISSON_PARAMS,
            "lgb": LGB_POISSON_PARAMS,
            "hgb_rounds": N_ROUNDS_VAL,
            "fast_mode": FAST_MODE,
            "enable_lgb": ENABLE_LGB,
            "enable_catboost": ENABLE_CATBOOST,
        },
        "seed": SEED,
        "train_final_rows": int(len(train)),
        "test_final_rows": int(len(test)),
        "train_final_mtime": file_mtime(TRAIN_FINAL),
        "test_final_mtime": file_mtime(TEST_FINAL),
        "dependency_availability": {
            "xgboost": getattr(xgb, "__version__", "unknown"),
            "sklearn": getattr(sklearn, "__version__", "unknown"),
            "lightgbm": getattr(lgb, "__version__", None) if lgb is not None else None,
            "catboost": CatBoostRegressor is not None,
        },
        "gender_distribution": {
            "train_w_share": float((train["gender"].astype(str).str.upper() == "W").mean()) if "gender" in train else None,
            "test_w_share": float((test["gender"].astype(str).str.upper() == "W").mean()) if "gender" in test else None,
        },
    }
    raw = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16], payload


def build_validation_artifacts(train: pd.DataFrame, test: pd.DataFrame, feature_cols: list[str], profile_name: str):
    print(f"\n[2] Building train-only rolling fold artifacts for profile={profile_name}...")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_key, cache_payload = feature_cache_key(profile_name, feature_cols, train, test)
    cache_path = CACHE_DIR / f"{profile_name}_{cache_key}.pkl"
    cache_meta = {
        "cache_used": False,
        "cache_key": cache_key,
        "cache_path": str(cache_path),
        "cache_rebuild_reason": "cache_missing",
        "cache_payload": cache_payload,
        "artifact_rows_per_fold": {},
    }
    if cache_path.exists():
        try:
            with cache_path.open("rb") as f:
                cached = pickle.load(f)
            cache_meta.update(cached.get("cache_meta", {}))
            cache_meta["cache_used"] = True
            cache_meta["cache_rebuild_reason"] = "valid_cache_hit"
            print(f"    cache hit: {cache_path.name}")
            return cached["artifacts"], cached["model_names"], cache_meta
        except Exception as exc:
            cache_meta["cache_rebuild_reason"] = f"cache_read_failed:{exc.__class__.__name__}"

    artifacts = []
    model_names_ref = None
    for fold in FOLDS:
        tr, val = fold_data(train, fold)
        if any(c in feature_cols for c in PRIOR_FEATURES):
            tr = add_static_priors(tr, tr)
            val = add_static_priors(tr, val)
        val["drift_weight"] = compute_drift_weights(val, test)
        val["eval_weight"] = val["metric_weight"] * val["drift_weight"]
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
        score_priors = score_prior_matrices(tr, val, max_goals=max(MAX_GOALS_CANDIDATES))
        artifacts.append(
            {
                "fold": fold,
                "feature_profile": profile_name,
                "train_rows": len(tr),
                "val": val,
                "team_preds": team_preds,
                "opp_preds": opp_preds,
                "outcome_probs": outcome_probs,
                "score_priors": score_priors,
            }
        )
        cache_meta["artifact_rows_per_fold"][fold["name"]] = int(len(val))
    with cache_path.open("wb") as f:
        pickle.dump({"artifacts": artifacts, "model_names": model_names_ref or [], "cache_meta": cache_meta}, f)
    return artifacts, model_names_ref or [], cache_meta


def normalize_weights(values) -> tuple[float, ...]:
    arr = np.asarray(values, dtype=float)
    arr = np.maximum(arr, 0.0)
    if arr.sum() <= 0:
        arr[:] = 1.0
    arr /= arr.sum()
    return tuple(float(x) for x in arr)


def weight_options(n_models: int) -> list[tuple[float, ...]]:
    opts = [normalize_weights(np.ones(n_models))]
    caps = [0.55, 0.65, 0.70]
    for cap in caps:
        for i in range(n_models):
            v = np.full(n_models, (1.0 - cap) / max(1, n_models - 1))
            v[i] = cap
            opts.append(normalize_weights(v))
    for i in range(n_models):
        for j in range(i + 1, n_models):
            v = np.full(n_models, 0.05)
            v[i] = 0.45
            v[j] = 0.45
            opts.append(normalize_weights(v))
    # A compact 0.1-grid sample that enforces diversity and max weight <= 0.70.
    for vals in np.ndindex(*(11 for _ in range(n_models))):
        if sum(vals) != 10:
            continue
        v = np.array(vals, dtype=float) / 10.0
        if v.max() <= 0.70 and np.sum(v >= 0.10) >= 2:
            opts.append(tuple(float(x) for x in v))
        if len(opts) >= 12:
            break
    # Deduplicate while preserving order.
    seen = set()
    out = []
    for opt in opts:
        key = tuple(round(x, 6) for x in opt)
        if key not in seen:
            seen.add(key)
            out.append(opt)
    return out


def apply_weight_cap(weights: tuple[float, ...], cap: float) -> tuple[float, ...]:
    arr = np.asarray(weights, dtype=float)
    arr = np.maximum(arr, 0.0)
    if arr.sum() <= 0:
        arr[:] = 1.0
    arr /= arr.sum()
    for _ in range(10):
        over = arr > cap
        if not np.any(over):
            break
        excess = float((arr[over] - cap).sum())
        arr[over] = cap
        under = ~over
        if not np.any(under):
            break
        under_sum = float(arr[under].sum())
        if under_sum <= 1e-12:
            arr[under] += excess / under.sum()
        else:
            arr[under] += excess * arr[under] / under_sum
    arr = np.minimum(arr, cap)
    arr /= max(arr.sum(), 1e-12)
    return tuple(float(x) for x in arr)


def meta_blend_weight_options(artifacts: list[dict], cap: float = 0.70) -> list[tuple[str, tuple[float, ...], tuple[float, ...]]]:
    team_x = np.vstack([a["team_preds"].T for a in artifacts])
    opp_x = np.vstack([a["opp_preds"].T for a in artifacts])
    team_y = np.concatenate([a["val"]["team_goals"].values for a in artifacts]).astype(float)
    opp_y = np.concatenate([a["val"]["opp_goals"].values for a in artifacts]).astype(float)
    out = []
    try:
        team_nnls = apply_weight_cap(normalize_weights(nnls(team_x, team_y)[0]), cap)
        opp_nnls = apply_weight_cap(normalize_weights(nnls(opp_x, opp_y)[0]), cap)
        out.append(("nnls_cap", team_nnls, opp_nnls))
    except Exception:
        pass
    try:
        ridge = Ridge(alpha=0.25, fit_intercept=False, positive=False)
        team_coef = ridge.fit(team_x, team_y).coef_
        opp_coef = ridge.fit(opp_x, opp_y).coef_
        out.append(("ridge_nonnegative_cap", apply_weight_cap(normalize_weights(np.maximum(team_coef, 0.0)), cap), apply_weight_cap(normalize_weights(np.maximum(opp_coef, 0.0)), cap)))
    except Exception:
        pass
    return out


def apply_weights(preds: np.ndarray, weights: tuple[float, ...]) -> np.ndarray:
    return np.average(preds, axis=0, weights=np.asarray(weights, dtype=float))


def aggregate_artifact_predictions(config: CandidateConfig, artifact: dict):
    lambda_team = apply_weights(artifact["team_preds"], config.team_weights)
    lambda_opp = apply_weights(artifact["opp_preds"], config.opp_weights)
    use_outcome = config.outcome_alpha > 0 or config.outcome_threshold > 0 or config.outcome_penalty_alpha > 0
    outcome_probs = artifact["outcome_probs"] if use_outcome else None
    score_priors = artifact.get("score_priors") if config.empirical_prior_beta > 0 else None
    pred_t, pred_o = predict_scores(lambda_team, lambda_opp, config, outcome_probs=outcome_probs, df=artifact["val"], score_priors=score_priors)
    return pred_t, pred_o, lambda_team, lambda_opp


def combine_fold_metrics(fold_metrics: list[dict], key: str) -> float:
    weights = np.array([m["fold_weight"] for m in fold_metrics], dtype=float)
    vals = np.array([m[key] for m in fold_metrics], dtype=float)
    return float(np.average(vals, weights=weights))


def evaluate_candidate(
    config: CandidateConfig,
    artifacts: list[dict],
    baseline: CandidateResult | None,
    thresholds: dict,
    precomputed_fold_preds: list[tuple[np.ndarray, np.ndarray]] | None = None,
) -> CandidateResult:
    if artifacts and config.feature_profile != artifacts[0].get("feature_profile", config.feature_profile):
        config = replace(config, feature_profile=artifacts[0]["feature_profile"])
    fold_metrics = []
    all_pred_t, all_pred_o, all_true_t, all_true_o, all_weights, all_dates, all_val = [], [], [], [], [], [], []
    for fold_idx, artifact in enumerate(artifacts):
        val = artifact["val"]
        if precomputed_fold_preds is None:
            pred_t, pred_o, _, _ = aggregate_artifact_predictions(config, artifact)
        else:
            pred_t, pred_o = precomputed_fold_preds[fold_idx]
        true_t = val["team_goals"].values.astype(int)
        true_o = val["opp_goals"].values.astype(int)
        weights = val["eval_weight"].values if "eval_weight" in val.columns else val["metric_weight"].values
        raw_weights = val["metric_weight"].values
        metric = metrics_dict(pred_t, pred_o, true_t, true_o, weights=weights, power=PRIMARY_POWER)
        raw_metric = metrics_dict(pred_t, pred_o, true_t, true_o, weights=raw_weights, power=PRIMARY_POWER)
        metric_secondary = metrics_dict(pred_t, pred_o, true_t, true_o, weights=weights, power=SECONDARY_POWER)
        fold_metrics.append(
            {
                "fold_name": artifact["fold"]["name"],
                "fold_weight": artifact["fold"]["weight"],
                "rows": int(len(val)),
                **metric,
                "raw_weighted_awmae": raw_metric["weighted_awmae"],
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
        all_val.append(val)

    pred_t = np.concatenate(all_pred_t)
    pred_o = np.concatenate(all_pred_o)
    true_t = np.concatenate(all_true_t)
    true_o = np.concatenate(all_true_o)
    weights = np.concatenate(all_weights)
    dates = pd.concat(all_dates, ignore_index=True)
    val_all = pd.concat(all_val, ignore_index=True)

    metrics = {
        "weighted_awmae_power_1_5": combine_fold_metrics(fold_metrics, "weighted_awmae"),
        "raw_weighted_awmae_power_1_5": combine_fold_metrics(fold_metrics, "raw_weighted_awmae"),
        "unweighted_awmae_power_1_5": mean_awmae(pred_t, pred_o, true_t, true_o, power=PRIMARY_POWER),
        "weighted_awmae_power_1_3": combine_fold_metrics(fold_metrics, "weighted_awmae_power_1_3"),
        "unweighted_awmae_power_1_3": mean_awmae(pred_t, pred_o, true_t, true_o, power=SECONDARY_POWER),
        "outcome_accuracy": outcome_accuracy(pred_t, pred_o, true_t, true_o),
        "exact_accuracy": exact_accuracy(pred_t, pred_o, true_t, true_o),
        "goal_diff_accuracy": goal_diff_accuracy(pred_t, pred_o, true_t, true_o),
    }
    dist = score_distribution(pred_t, pred_o)
    yearly = yearly_distribution(pred_t, pred_o, dates)
    segments = validation_segment_snapshot(pred_t, pred_o, true_t, true_o, weights, val_all)
    fold_awmae = np.array([m["weighted_awmae"] for m in fold_metrics], dtype=float)

    base = baseline.metrics if baseline is not None else metrics
    base_dist = baseline.distribution if baseline is not None else dist
    latest_fold_awmae = fold_metrics[-1]["weighted_awmae"]
    fold_instability = 0.15 * float(np.std(fold_awmae)) + 0.10 * max(0.0, float(latest_fold_awmae - np.mean(fold_awmae) - 0.06))
    outcome_drop = 2.0 * max(0.0, base["outcome_accuracy"] - metrics["outcome_accuracy"] - 0.003)
    exact_drop = 1.5 * max(0.0, base["exact_accuracy"] - metrics["exact_accuracy"] - 0.004)
    distribution_guard = (
        0.30 * max(0.0, dist["score_ge5_share"] - 0.030) ** 2
        + 0.20 * max(0.0, dist["avg_total_goals"] - 3.05) ** 2
        + 1.00 * max(0.0, 2.75 - dist["avg_total_goals"]) ** 2
        + 0.80 * max(0.0, dist["draw_share"] - 0.220) ** 2
    )
    balanced = segments.get("balanced_elo", {})
    neutral = segments.get("neutral", {})
    non_neutral = segments.get("non_neutral", {})
    balanced_penalty = 0.0
    if balanced:
        balanced_penalty = max(0.0, balanced["weighted_awmae"] - metrics["weighted_awmae_power_1_5"] - 0.35)
    neutral_penalty = 0.0
    if neutral and non_neutral:
        neutral_penalty = max(0.0, neutral["weighted_awmae"] - non_neutral["weighted_awmae"] - 0.25)
    collapse = 0.0
    yearly_penalty = yearly_distribution_penalty(yearly, dist["draw_share"])
    if not config.risk_penalties_enabled:
        distribution_guard = collapse = balanced_penalty = neutral_penalty = yearly_penalty = outcome_drop = 0.0

    risk_components = {
        "fold_instability_penalty": float(fold_instability),
        "outcome_gap_penalty": float(outcome_drop),
        "exact_drop_penalty": float(exact_drop),
        "distribution_guard_penalty": float(distribution_guard),
        "high_score_tail_penalty": float(0.0),
        "score_collapse_penalty": float(collapse),
        "balanced_elo_penalty": float(balanced_penalty),
        "neutral_segment_penalty": float(neutral_penalty),
        "yearly_distribution_penalty": float(yearly_penalty),
    }
    selection_score = (
        0.70 * metrics["weighted_awmae_power_1_5"]
        + 0.15 * balanced_penalty
        + 0.10 * neutral_penalty
        + 0.05 * outcome_drop
        + fold_instability
        + distribution_guard
    )
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


def conditional_score_candidate_configs(base_config: CandidateConfig) -> list[CandidateConfig]:
    base = replace(base_config, use_score_reranker=False)
    variants = [
        replace(base, stage="10_conditional_score_reranker", name="base_erm"),
        replace(
            base,
            stage="10_conditional_score_reranker",
            name="outcome_soft",
            outcome_alpha=max(base.outcome_alpha, 0.10),
            outcome_threshold=max(base.outcome_threshold, 0.42),
            outcome_penalty_alpha=max(base.outcome_penalty_alpha, 0.10),
        ),
        replace(
            base,
            stage="10_conditional_score_reranker",
            name="outcome_strict",
            outcome_alpha=max(base.outcome_alpha, 0.20),
            outcome_threshold=0.38,
            outcome_penalty_alpha=max(base.outcome_penalty_alpha, 0.15),
        ),
        replace(
            base,
            stage="10_conditional_score_reranker",
            name="higher_total",
            total_lambda_scale=max(base.total_lambda_scale, 1.06),
            total_goal_delta=max(base.total_goal_delta, 0.20),
        ),
        replace(
            base,
            stage="10_conditional_score_reranker",
            name="common_scores",
            common_score_boost=max(base.common_score_boost, 1.10),
            high_margin_dampener=min(base.high_margin_dampener, 0.95),
        ),
        replace(
            base,
            stage="10_conditional_score_reranker",
            name="women_total_push",
            total_lambda_scale=max(base.total_lambda_scale, 1.08),
            outcome_alpha=max(base.outcome_alpha, 0.10),
            total_goal_delta=max(base.total_goal_delta, 0.20),
        ),
        replace(
            base,
            stage="10_conditional_score_reranker",
            name="segment_safe",
            balanced_draw_boost=max(base.balanced_draw_boost, 1.16),
            neutral_total_scale=max(base.neutral_total_scale, 1.00),
            neutral_draw_boost=base.neutral_draw_boost,
        ),
    ]
    seen = set()
    out = []
    for cfg in variants:
        key = (
            cfg.max_goals,
            cfg.total_lambda_scale,
            cfg.common_score_boost,
            cfg.high_margin_dampener,
            cfg.outcome_alpha,
            cfg.outcome_threshold,
            cfg.outcome_penalty_alpha,
            cfg.empirical_prior_beta,
            cfg.total_goal_delta,
            cfg.goal_diff_eta,
            cfg.balanced_draw_boost,
            cfg.neutral_total_scale,
        )
        if key not in seen:
            seen.add(key)
            out.append(cfg)
    return out


def reranker_row_features(df: pd.DataFrame, outcome_probs: np.ndarray, lambda_team: np.ndarray, lambda_opp: np.ndarray) -> np.ndarray:
    n = len(df)
    outcome_probs = np.asarray(outcome_probs, dtype=float)
    confidence = outcome_probs.max(axis=1)
    margin = np.sort(outcome_probs, axis=1)[:, -1] - np.sort(outcome_probs, axis=1)[:, -2]
    cols = [
        df.get("is_women_match", pd.Series(0, index=df.index)).fillna(0).astype(float).values,
        df.get("neutral", pd.Series(0, index=df.index)).fillna(0).astype(float).values,
        df.get("is_friendly_risk", pd.Series(0, index=df.index)).fillna(0).astype(float).values,
        df.get("is_qualifier_risk", pd.Series(0, index=df.index)).fillna(0).astype(float).values,
        df.get("is_major_tournament_risk", pd.Series(0, index=df.index)).fillna(0).astype(float).values,
        df.get("abs_elo_diff_risk", pd.Series(0, index=df.index)).fillna(0).astype(float).values / 500.0,
        df.get("tournament_weight", pd.Series(1.0, index=df.index)).fillna(1.0).astype(float).values,
        df.get("year_since_2011_drift", pd.Series(0, index=df.index)).fillna(0).astype(float).values / 15.0,
        df.get("frozen_strength_staleness", pd.Series(0, index=df.index)).fillna(0).astype(float).values / 15.0,
        np.asarray(lambda_team, dtype=float),
        np.asarray(lambda_opp, dtype=float),
        np.asarray(lambda_team, dtype=float) + np.asarray(lambda_opp, dtype=float),
        np.asarray(lambda_team, dtype=float) - np.asarray(lambda_opp, dtype=float),
        outcome_probs[:, 0],
        outcome_probs[:, 1],
        outcome_probs[:, 2],
        confidence,
        margin,
    ]
    return np.column_stack(cols).reshape(n, -1)


def candidate_score_table(
    base_config: CandidateConfig,
    artifact: dict,
    candidate_configs: list[CandidateConfig] | None = None,
    include_target: bool = True,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray, np.ndarray]:
    candidate_configs = candidate_configs or conditional_score_candidate_configs(base_config)
    df = artifact["val"]
    lambda_team = apply_weights(artifact["team_preds"], base_config.team_weights)
    lambda_opp = apply_weights(artifact["opp_preds"], base_config.opp_weights)
    outcome_probs = artifact["outcome_probs"]
    score_priors_all = artifact.get("score_priors")

    pred_t_cols = []
    pred_o_cols = []
    for cfg in candidate_configs:
        use_outcome = cfg.outcome_alpha > 0 or cfg.outcome_threshold > 0 or cfg.outcome_penalty_alpha > 0
        score_priors = score_priors_all if cfg.empirical_prior_beta > 0 else None
        pt, po = predict_scores(
            lambda_team,
            lambda_opp,
            cfg,
            outcome_probs=outcome_probs if use_outcome else None,
            df=df,
            score_priors=score_priors,
        )
        pred_t_cols.append(pt)
        pred_o_cols.append(po)

    pred_t = np.vstack(pred_t_cols).T.astype(float)
    pred_o = np.vstack(pred_o_cols).T.astype(float)
    n, k = pred_t.shape
    row = reranker_row_features(df, outcome_probs, lambda_team, lambda_opp)
    row_rep = np.repeat(row, k, axis=0)
    flat_t = pred_t.reshape(-1)
    flat_o = pred_o.reshape(-1)
    score_outcome = np.sign(flat_t - flat_o)
    candidate_idx = np.tile(np.arange(k, dtype=float) / max(1, k - 1), n)
    score_total = flat_t + flat_o
    score_diff = flat_t - flat_o
    score_abs_diff = np.abs(score_diff)
    score_is_draw = (score_diff == 0).astype(float)
    score_is_win = (score_diff > 0).astype(float)
    score_is_loss = (score_diff < 0).astype(float)
    meta = np.column_stack(
        [
            row_rep,
            candidate_idx,
            flat_t,
            flat_o,
            score_total,
            score_diff,
            score_abs_diff,
            score_is_loss,
            score_is_draw,
            score_is_win,
            (flat_t >= 5).astype(float),
            (flat_o >= 5).astype(float),
            (score_total <= 2).astype(float),
            (score_total >= 4).astype(float),
            score_outcome,
        ]
    )

    y = None
    weights = None
    if include_target:
        true_t = np.repeat(df["team_goals"].values.astype(int), k)
        true_o = np.repeat(df["opp_goals"].values.astype(int), k)
        y = awmae_loss_array(flat_t, flat_o, true_t, true_o, power=PRIMARY_POWER)
        weights = np.repeat(df.get("eval_weight", df["metric_weight"]).values.astype(float), k)
    return meta, y, weights, pred_t.astype(int), pred_o.astype(int)


def fit_score_reranker(x, y, w):
    reg = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.045,
        max_iter=160 if FAST_MODE else 260,
        max_leaf_nodes=31,
        min_samples_leaf=90,
        l2_regularization=2.5,
        random_state=SEED,
        early_stopping=False,
    )
    reg.fit(x, y, sample_weight=w)
    return reg


def reranker_oof_predictions(base_config: CandidateConfig, artifacts: list[dict]) -> list[tuple[np.ndarray, np.ndarray]]:
    candidate_configs = conditional_score_candidate_configs(base_config)
    tables = [candidate_score_table(base_config, artifact, candidate_configs, include_target=True) for artifact in artifacts]
    out = []
    for i, (x_val, _, _, pred_t, pred_o) in enumerate(tables):
        train_parts = [tables[j] for j in range(len(tables)) if j != i]
        x_train = np.vstack([p[0] for p in train_parts])
        y_train = np.concatenate([p[1] for p in train_parts])
        w_train = np.concatenate([p[2] for p in train_parts])
        reg = fit_score_reranker(x_train, y_train, w_train)
        pred_loss = reg.predict(x_val).reshape(pred_t.shape)
        choice = pred_loss.argmin(axis=1)
        rows = np.arange(len(choice))
        out.append((pred_t[rows, choice].astype(int), pred_o[rows, choice].astype(int)))
    return out


def fit_score_reranker_from_artifacts(base_config: CandidateConfig, artifacts: list[dict]):
    candidate_configs = conditional_score_candidate_configs(base_config)
    tables = [candidate_score_table(base_config, artifact, candidate_configs, include_target=True) for artifact in artifacts]
    x_train = np.vstack([p[0] for p in tables])
    y_train = np.concatenate([p[1] for p in tables])
    w_train = np.concatenate([p[2] for p in tables])
    return fit_score_reranker(x_train, y_train, w_train), candidate_configs


def apply_score_reranker_to_artifact(base_config: CandidateConfig, artifact: dict, reranker, candidate_configs: list[CandidateConfig]):
    x, _, _, pred_t, pred_o = candidate_score_table(base_config, artifact, candidate_configs, include_target=False)
    pred_loss = reranker.predict(x).reshape(pred_t.shape)
    choice = pred_loss.argmin(axis=1)
    rows = np.arange(len(choice))
    return pred_t[rows, choice].astype(int), pred_o[rows, choice].astype(int)


def evaluate_score_reranker_candidate(
    base_config: CandidateConfig,
    artifacts: list[dict],
    baseline: CandidateResult | None,
    thresholds: dict,
) -> CandidateResult:
    cfg = replace(base_config, stage="10_conditional_score_reranker", name="oof_loss_reranker", use_score_reranker=True)
    preds = reranker_oof_predictions(base_config, artifacts)
    return evaluate_candidate(cfg, artifacts, baseline, thresholds, precomputed_fold_preds=preds)


def ablation_accepts(prev: CandidateResult, cand: CandidateResult) -> tuple[bool, str]:
    aw_delta = prev.metrics["weighted_awmae_power_1_5"] - cand.metrics["weighted_awmae_power_1_5"]
    aw_ok = cand.metrics["weighted_awmae_power_1_5"] <= prev.metrics["weighted_awmae_power_1_5"] + 0.0005
    outcome_ok = cand.metrics["outcome_accuracy"] >= prev.metrics["outcome_accuracy"] - 0.007
    exact_ok = cand.metrics["exact_accuracy"] >= prev.metrics["exact_accuracy"] - 0.005
    instability_ok = cand.risk_components["fold_instability_penalty"] <= prev.risk_components["fold_instability_penalty"] + 0.025
    collapse_ok = cand.distribution["top3_score_share"] <= 0.86
    avg_total_ok = 2.55 <= cand.distribution["avg_total_goals"] <= 3.10
    draw_ok = cand.distribution["draw_share"] <= 0.235
    # AW-MAE already includes outcome and exact penalties; let clear AW wins pass
    # unless they cause a severe explicit guard failure.
    clear_aw_win = aw_delta >= 0.003 and exact_ok and cand.distribution["avg_total_goals"] <= 3.05
    outcome_guard_win = (
        cand.config.stage == "05_outcome_first_erm"
        and cand.metrics["outcome_accuracy"] >= prev.metrics["outcome_accuracy"] + 0.004
        and cand.metrics["weighted_awmae_power_1_5"] <= prev.metrics["weighted_awmae_power_1_5"] + 0.012
        and exact_ok
        and avg_total_ok
        and draw_ok
    )
    ok = (
        (aw_ok and outcome_ok and exact_ok and instability_ok and collapse_ok and avg_total_ok and draw_ok)
        or (clear_aw_win and draw_ok)
        or outcome_guard_win
    )
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
    if not avg_total_ok:
        reasons.append("avg_total_out_of_band")
    if not draw_ok:
        reasons.append("draw_share_too_high")
    return ok, "accepted" if ok else ",".join(reasons)


def friend_distribution_thresholds() -> dict:
    tail_threshold = 0.030
    friend_dist = None
    return {
        "tail_threshold": 0.0275,
        "top3_threshold": 0.68,
        "top1_threshold": 0.30,
        "friend_distribution": friend_dist,
    }


def tune_candidates(artifacts: list[dict], model_names: list[str], profile_name: str) -> tuple[CandidateResult, list[dict], dict]:
    print("\n[3] Tuning risk-minimizing candidate stages...")
    thresholds = friend_distribution_thresholds()
    n_models = len(model_names)
    uniform = normalize_weights(np.ones(n_models))
    options = weight_options(n_models)
    meta_options = meta_blend_weight_options(artifacts, cap=0.70)

    baseline_config = CandidateConfig(
        stage="01_baseline_poisson_ensemble",
        name="uniform_max8_no_risk",
        team_weights=uniform,
        opp_weights=uniform,
        feature_profile=profile_name,
        max_goals=8,
        risk_penalties_enabled=False,
    )
    baseline = evaluate_candidate(baseline_config, artifacts, None, thresholds)
    baseline = evaluate_candidate(baseline_config, artifacts, baseline, thresholds)
    baseline.ablation_reason = "initial_baseline"
    current = baseline
    ablations = [ablation_summary(baseline, selected=True)]
    print(f"    baseline: aw15={current.metrics['weighted_awmae_power_1_5']:.5f}, selection={current.selection_score:.5f}")

    stage_names = [
        "02_oof_blend_selection",
        "03_scale_bias_total_goals",
        "04_awmae_erm_max_goals",
        "05_outcome_first_erm",
        "06_draw_low_score_prior_calibration",
        "07_segment_balanced_neutral_calibration",
        "08_hybrid_matrix_static_prior",
        "09_risk_objective_v4",
        "10_conditional_score_reranker",
    ]

    def build_stage_configs(stage_name: str, base_config: CandidateConfig) -> list[CandidateConfig]:
        if stage_name == "02_oof_blend_selection":
            configs = [
                CandidateConfig("02_oof_blend_selection", f"convex_tw{i}_ow{j}", tw, ow, feature_profile=profile_name, blend_method="convex_grid", max_goals=base_config.max_goals)
                for i, tw in enumerate(options)
                for j, ow in enumerate(options)
            ]
            configs.extend(
                CandidateConfig("02_oof_blend_selection", name, tw, ow, feature_profile=profile_name, blend_method=name, max_goals=base_config.max_goals)
                for name, tw, ow in meta_options
            )
            return configs
        if stage_name == "03_awmae_erm_max_goals":
            return []
        if stage_name == "03_scale_bias_total_goals":
            configs = []
            for total_scale in TOTAL_LAMBDA_SCALES:
                configs.append(
                    CandidateConfig(
                        "03_scale_bias_total_goals",
                        f"total{total_scale}",
                        base_config.team_weights,
                        base_config.opp_weights,
                        max_goals=base_config.max_goals,
                        total_lambda_scale=total_scale,
                    )
                )
            for ts in [0.92, 0.96, 1.00]:
                for os in [0.92, 0.96, 1.00]:
                    configs.append(
                        CandidateConfig(
                            "03_scale_bias_total_goals",
                            f"team{ts}_opp{os}",
                            base_config.team_weights,
                            base_config.opp_weights,
                            max_goals=base_config.max_goals,
                            team_scale=ts,
                            opp_scale=os,
                            total_lambda_scale=base_config.total_lambda_scale,
                        )
                    )
            for tb in TEAM_BIASES:
                for ob in OPP_BIASES:
                    configs.append(
                        CandidateConfig(
                            "03_scale_bias_total_goals",
                            f"tb{tb}_ob{ob}",
                            base_config.team_weights,
                            base_config.opp_weights,
                            max_goals=base_config.max_goals,
                            team_scale=base_config.team_scale,
                            opp_scale=base_config.opp_scale,
                            team_bias=tb,
                            opp_bias=ob,
                            total_lambda_scale=base_config.total_lambda_scale,
                        )
                    )
            return configs
        if stage_name == "04_awmae_erm_max_goals":
            return [
                CandidateConfig(
                    "04_awmae_erm_max_goals",
                    f"max{m}",
                    base_config.team_weights,
                    base_config.opp_weights,
                    max_goals=m,
                    team_scale=base_config.team_scale,
                    opp_scale=base_config.opp_scale,
                    team_bias=base_config.team_bias,
                    opp_bias=base_config.opp_bias,
                    total_lambda_scale=base_config.total_lambda_scale,
                )
                for m in MAX_GOALS_CANDIDATES
            ]
        if stage_name == "05_outcome_first_erm":
            return [
                CandidateConfig(
                    "05_outcome_first_erm",
                    f"alpha{a}_thr{thr}_pen{pen}",
                    base_config.team_weights,
                    base_config.opp_weights,
                    max_goals=base_config.max_goals,
                    team_scale=base_config.team_scale,
                    opp_scale=base_config.opp_scale,
                    team_bias=base_config.team_bias,
                    opp_bias=base_config.opp_bias,
                    total_lambda_scale=base_config.total_lambda_scale,
                    outcome_alpha=a,
                    outcome_threshold=thr,
                    outcome_penalty_alpha=pen,
                )
                for a in OUTCOME_ALPHAS
                for thr in OUTCOME_THRESHOLDS
                for pen in OUTCOME_PENALTY_ALPHAS
            ]
        if stage_name == "06_draw_low_score_prior_calibration":
            return [
                CandidateConfig(
                    "06_draw_low_score_prior_calibration",
                    f"d{d}_ld{ld}_lc{lc}_cb{cb}_hm{hm}",
                    base_config.team_weights,
                    base_config.opp_weights,
                    max_goals=base_config.max_goals,
                    team_scale=base_config.team_scale,
                    opp_scale=base_config.opp_scale,
                    team_bias=base_config.team_bias,
                    opp_bias=base_config.opp_bias,
                    total_lambda_scale=base_config.total_lambda_scale,
                    draw_boost=d,
                    low_draw_boost=ld,
                    low_decisive_boost=lc,
                    common_score_boost=cb,
                    high_margin_dampener=hm,
                    outcome_alpha=base_config.outcome_alpha,
                    outcome_threshold=base_config.outcome_threshold,
                    outcome_penalty_alpha=base_config.outcome_penalty_alpha,
                )
                for d in DRAW_BOOSTS
                for ld in LOW_DRAW_BOOSTS
                for lc in LOW_DECISIVE_BOOSTS
                for cb in COMMON_SCORE_BOOSTS
                for hm in HIGH_MARGIN_DAMPENERS
            ]
        if stage_name == "07_segment_balanced_neutral_calibration":
            return [
                CandidateConfig(
                    "07_segment_balanced_neutral_calibration",
                    f"bd{bd}_bs{bs}_ns{ns}_nd{nd}_nm{nm}",
                    base_config.team_weights,
                    base_config.opp_weights,
                    max_goals=base_config.max_goals,
                    team_scale=base_config.team_scale,
                    opp_scale=base_config.opp_scale,
                    team_bias=base_config.team_bias,
                    opp_bias=base_config.opp_bias,
                    total_lambda_scale=base_config.total_lambda_scale,
                    draw_boost=base_config.draw_boost,
                    low_draw_boost=base_config.low_draw_boost,
                    low_decisive_boost=base_config.low_decisive_boost,
                    outcome_alpha=base_config.outcome_alpha,
                    outcome_threshold=base_config.outcome_threshold,
                    outcome_penalty_alpha=base_config.outcome_penalty_alpha,
                    tail_dampener=base_config.tail_dampener,
                    common_score_boost=base_config.common_score_boost,
                    high_margin_dampener=base_config.high_margin_dampener,
                    balanced_draw_boost=bd,
                    balanced_total_scale=bs,
                    neutral_total_scale=ns,
                    neutral_draw_boost=nd,
                    neutral_margin_dampener=nm,
                )
                for bd in BALANCED_DRAW_BOOSTS
                for bs in BALANCED_TOTAL_SCALES
                for ns in NEUTRAL_TOTAL_SCALES
                for nd in NEUTRAL_DRAW_BOOSTS
                for nm in NEUTRAL_MARGIN_DAMPENERS
            ]
        if stage_name == "08_hybrid_matrix_static_prior":
            return [
                replace(
                    base_config,
                    stage="08_hybrid_matrix_static_prior",
                    name=f"beta{beta}_td{td}_gd{eta}_tt{tau_t}_tg{tau_g}",
                    empirical_prior_beta=beta,
                    total_goal_delta=td,
                    goal_diff_eta=eta,
                    tau_total=tau_t,
                    tau_gd=tau_g,
                )
                for beta in EMPIRICAL_PRIOR_BETAS
                for td in TOTAL_GOAL_DELTAS
                for eta in GOAL_DIFF_ETAS
                for tau_t in TAU_TOTALS
                for tau_g in TAU_GDS
            ]
        if stage_name == "09_risk_objective_v4":
            return [
                CandidateConfig(
                    "09_risk_objective_v4",
                    "risk_penalties_on",
                    base_config.team_weights,
                    base_config.opp_weights,
                    max_goals=base_config.max_goals,
                    team_scale=base_config.team_scale,
                    opp_scale=base_config.opp_scale,
                    team_bias=base_config.team_bias,
                    opp_bias=base_config.opp_bias,
                    total_lambda_scale=base_config.total_lambda_scale,
                    draw_boost=base_config.draw_boost,
                    low_draw_boost=base_config.low_draw_boost,
                    low_decisive_boost=base_config.low_decisive_boost,
                    outcome_alpha=base_config.outcome_alpha,
                    outcome_threshold=base_config.outcome_threshold,
                    outcome_penalty_alpha=base_config.outcome_penalty_alpha,
                    empirical_prior_beta=base_config.empirical_prior_beta,
                    total_goal_delta=base_config.total_goal_delta,
                    goal_diff_eta=base_config.goal_diff_eta,
                    tau_total=base_config.tau_total,
                    tau_gd=base_config.tau_gd,
                    tail_dampener=base_config.tail_dampener,
                    common_score_boost=base_config.common_score_boost,
                    high_margin_dampener=base_config.high_margin_dampener,
                    balanced_draw_boost=base_config.balanced_draw_boost,
                    balanced_total_scale=base_config.balanced_total_scale,
                    neutral_total_scale=base_config.neutral_total_scale,
                    neutral_draw_boost=base_config.neutral_draw_boost,
                    neutral_margin_dampener=base_config.neutral_margin_dampener,
                    risk_penalties_enabled=True,
                )
            ]
        if stage_name == "10_conditional_score_reranker":
            return [
                replace(
                    base_config,
                    stage="10_conditional_score_reranker",
                    name="oof_loss_reranker",
                    use_score_reranker=True,
                )
            ]
        return []

    for stage_name in stage_names:
        configs = build_stage_configs(stage_name, current.config)
        best = None
        for cfg in configs:
            if stage_name != "02_oof_blend_selection":
                cfg = replace(
                    cfg,
                    feature_profile=current.config.feature_profile,
                    blend_method=current.config.blend_method,
                    weight_cap=current.config.weight_cap,
            )
            if cfg.use_score_reranker:
                result = evaluate_score_reranker_candidate(current.config, artifacts, baseline, thresholds)
            else:
                result = evaluate_candidate(cfg, artifacts, baseline, thresholds)
            if stage_name == "05_outcome_first_erm":
                eligible = (
                    result.metrics["weighted_awmae_power_1_5"] <= current.metrics["weighted_awmae_power_1_5"] + 0.012
                    and result.metrics["exact_accuracy"] >= current.metrics["exact_accuracy"] - 0.006
                    and result.distribution["draw_share"] <= 0.180
                    and result.distribution["avg_total_goals"] >= 2.50
                )
                result_key = (
                    0 if eligible else 1,
                    -result.metrics["outcome_accuracy"],
                    result.selection_score,
                )
                best_key = (
                    0 if (
                        best is not None
                        and best.metrics["weighted_awmae_power_1_5"] <= current.metrics["weighted_awmae_power_1_5"] + 0.012
                        and best.metrics["exact_accuracy"] >= current.metrics["exact_accuracy"] - 0.006
                        and best.distribution["draw_share"] <= 0.180
                        and best.distribution["avg_total_goals"] >= 2.50
                    ) else 1,
                    -best.metrics["outcome_accuracy"] if best is not None else 0,
                    best.selection_score if best is not None else float("inf"),
                )
                if best is None or result_key < best_key:
                    best = result
            elif best is None or result.selection_score < best.selection_score:
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


def select_profile_result(results: list[dict]) -> dict:
    ordered = sorted(
        results,
        key=lambda r: (
            r["chosen"].selection_score,
            r["chosen"].fold_metrics[-1]["weighted_awmae"],
            -r["chosen"].metrics["outcome_accuracy"],
            r["chosen"].risk_components.get("balanced_elo_penalty", 0.0),
            len(r["feature_cols"]),
        ),
    )
    best = ordered[0]
    for item in ordered:
        delta = item["chosen"].selection_score - best["chosen"].selection_score
        if 0 <= delta < 0.005:
            current_key = (
                item["chosen"].fold_metrics[-1]["weighted_awmae"],
                -item["chosen"].metrics["outcome_accuracy"],
                item["chosen"].risk_components.get("balanced_elo_penalty", 0.0),
                len(item["feature_cols"]),
            )
            best_key = (
                best["chosen"].fold_metrics[-1]["weighted_awmae"],
                -best["chosen"].metrics["outcome_accuracy"],
                best["chosen"].risk_components.get("balanced_elo_penalty", 0.0),
                len(best["feature_cols"]),
            )
            if current_key < best_key:
                best = item
    return best


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
def fit_final_predictions(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
    chosen: CandidateResult,
    model_names: list[str],
    artifacts: list[dict] | None = None,
):
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
    if chosen.config.outcome_alpha > 0 or chosen.config.outcome_threshold > 0 or chosen.config.outcome_penalty_alpha > 0:
        print("      - final outcome classifier")
        outcome_probs = fit_outcome_classifier(x_train, train, w_train, x_test)
    lambda_team = apply_weights(team_preds, chosen.config.team_weights)
    lambda_opp = apply_weights(opp_preds, chosen.config.opp_weights)
    score_priors = score_prior_matrices(train, test, max_goals=max(MAX_GOALS_CANDIDATES)) if chosen.config.empirical_prior_beta > 0 else None
    if chosen.config.use_score_reranker:
        if artifacts is None:
            raise ValueError("Score reranker final inference requires validation artifacts.")
        if outcome_probs is None:
            outcome_probs = fit_outcome_classifier(x_train, train, w_train, x_test)
        print("      - final conditional score reranker")
        reranker, candidate_configs = fit_score_reranker_from_artifacts(chosen.config, artifacts)
        final_artifact = {
            "val": test,
            "team_preds": team_preds,
            "opp_preds": opp_preds,
            "outcome_probs": outcome_probs,
            "score_priors": score_prior_matrices(train, test, max_goals=max(MAX_GOALS_CANDIDATES)),
        }
        pred_t, pred_o = apply_score_reranker_to_artifact(chosen.config, final_artifact, reranker, candidate_configs)
    else:
        pred_t, pred_o = predict_scores(lambda_team, lambda_opp, chosen.config, outcome_probs=outcome_probs, df=test, score_priors=score_priors)
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


def gender_diagnostics(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    def summarize(df: pd.DataFrame):
        gender = df.get("gender", pd.Series("M", index=df.index)).fillna("M").astype(str).str.upper().str.strip()
        years = pd.to_datetime(df["date"]).dt.year
        by_year = gender.eq("W").groupby(years).mean().to_dict()
        top_w_tournaments = df.loc[gender.eq("W"), "tournament"].astype(str).value_counts().head(20).to_dict()
        return {
            "rows": int(len(df)),
            "unique_gender_values": sorted(gender.unique().tolist()),
            "women_rows": int(gender.eq("W").sum()),
            "women_share": float(gender.eq("W").mean()),
            "missing_gender_rows": int(df.get("gender", pd.Series(index=df.index)).isna().sum()),
            "women_share_by_year": {str(k): float(v) for k, v in by_year.items()},
            "top_women_tournaments": top_w_tournaments,
        }
    return {
        "train": summarize(train),
        "test": summarize(test),
        "expected_train_w_share": 0.1118,
        "expected_test_w_share": 0.3290,
    }


def segment_metrics_for_validation(chosen: CandidateResult, artifacts: list[dict]) -> dict:
    pred_t_all, pred_o_all, true_t_all, true_o_all, val_all = [], [], [], [], []
    precomputed = reranker_oof_predictions(chosen.config, artifacts) if chosen.config.use_score_reranker else None
    for idx, artifact in enumerate(artifacts):
        if precomputed is None:
            pred_t, pred_o, _, _ = aggregate_artifact_predictions(chosen.config, artifact)
        else:
            pred_t, pred_o = precomputed[idx]
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


def acceptance_decision(local15, local13, friend15, friend13, final_dist, thresholds):
    checks = {
        "selected_purely_from_train_folds": True,
        "no_test_feature_update": True,
        "no_v5_import_output_anchor": True,
        "target_power_1_5_available": local15 is not None,
        "target_power_1_5_reached": False,
        "target_power_1_3_reached": None,
        "target_outcome_reached": None,
        "target_exact_reached": None,
        "target_goal_diff_reached": None,
        "beats_friend_power_1_5": None,
        "beats_friend_power_1_3": None,
        "outcome_not_below_friend": None,
        "tail_guard": final_dist["score_ge5_share"] <= 0.030,
        "avg_total_guard": 2.50 <= final_dist["avg_total_goals"] <= 2.75,
    }
    if local15 is None:
        return "VALIDATION_ONLY", checks
    checks["target_power_1_5_reached"] = local15["weighted_awmae"] <= TARGET_POWER_15_WEIGHTED
    if local13 is not None:
        checks["target_power_1_3_reached"] = local13["weighted_awmae"] <= TARGET_POWER_13_WEIGHTED
    checks["target_outcome_reached"] = local15["outcome_accuracy"] >= TARGET_OUTCOME
    checks["target_exact_reached"] = local15["exact_accuracy"] >= TARGET_EXACT
    checks["target_goal_diff_reached"] = local15["goal_diff_accuracy"] >= 0.21838
    if friend15 is not None and friend15.get("ground_truth_available", False):
        checks["beats_friend_power_1_5"] = local15["weighted_awmae"] < friend15["weighted_awmae"]
        checks["outcome_not_below_friend"] = local15["outcome_accuracy"] >= friend15["outcome_accuracy"] - 0.003
    if local13 is not None and friend13 is not None and friend13.get("ground_truth_available", False):
        checks["beats_friend_power_1_3"] = local13["weighted_awmae"] < friend13["weighted_awmae"]
    if not checks["target_power_1_5_reached"]:
        return "TARGET_NOT_REACHED", checks
    if checks["target_power_1_3_reached"] is False or checks["target_outcome_reached"] is False or checks["target_exact_reached"] is False or checks["target_goal_diff_reached"] is False:
        return "TARGET_NOT_REACHED", checks
    if checks["beats_friend_power_1_5"] is False or checks["beats_friend_power_1_3"] is False or checks["outcome_not_below_friend"] is False:
        return "NOT_ACCEPTED", checks
    if not (checks["tail_guard"] and checks["avg_total_guard"]):
        return "NOT_ACCEPTED", checks
    return "ACCEPTED_STATIC_DRIFT_V4", checks


def write_outputs(
    train,
    test,
    feature_cols,
    dropped_cols,
    feature_groups,
    profile_comparison,
    cache_metadata,
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
    decision, checks = acceptance_decision(local15, local13, friend15, friend13, final_dist, thresholds)
    gender_diag = gender_diagnostics(train, test)
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
        "pipeline": "model_pipeline_risk_v4_static_drift",
        "primary_metric_power": PRIMARY_POWER,
        "secondary_metric_power": SECONDARY_POWER,
        "target_weighted_awmae_power_1_5": TARGET_POWER_15_WEIGHTED,
        "static_no_test_update": True,
        "no_v5_import_declaration": True,
        "no_old_submission_anchor_declaration": True,
        "friend_csv_reporting_only": str(FRIEND_SUB),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "feature_profile": chosen.config.feature_profile,
        "feature_profiles_evaluated": profile_comparison,
        "feature_count": int(len(feature_cols)),
        "feature_columns": feature_cols,
        "feature_groups": feature_groups,
        "dropped_columns": dropped_cols,
        "cache_metadata": cache_metadata,
        "gender_diagnostics": gender_diag,
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
    lines.append("Risk V4 Static Drift Pipeline Validation Report")
    lines.append("=" * 48)
    lines.append(f"Acceptance decision: {decision}")
    lines.append(f"Selected stage: {chosen.config.stage} / {chosen.config.name}")
    lines.append(f"Selected feature profile: {chosen.config.feature_profile}")
    lines.append(
        f"Gender W share train/test: {gender_diag['train']['women_share']:.4f} / {gender_diag['test']['women_share']:.4f}"
    )
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
    lines.append("Gender diagnostics")
    lines.append(json.dumps(gender_diag, indent=2)[:4000])
    lines.append("")
    lines.append("Feature profile comparison")
    for item in profile_comparison:
        lines.append(
            f"  {item['profile']}: selected={item['selected']}, features={item['feature_count']}, "
            f"selection={item['selection_score']:.5f}, weighted15={item['weighted_awmae_power_1_5']:.5f}, "
            f"outcome={item['outcome_accuracy']:.4f}, latest={item['latest_fold_awmae']:.5f}"
        )
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
    if decision != "ACCEPTED_STATIC_DRIFT_V4":
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
    print("MODEL PIPELINE RISK V4 STATIC DRIFT - GENDER AWARE")
    print("=" * 72)
    train, test, feature_profiles, dropped_by_profile, feature_groups_by_profile = load_data()
    profile_results = []
    for profile_name, cols in feature_profiles.items():
        artifacts, model_names, cache_meta = build_validation_artifacts(train, test, cols, profile_name)
        chosen, ablations, thresholds = tune_candidates(artifacts, model_names, profile_name)
        profile_results.append(
            {
                "profile": profile_name,
                "feature_cols": cols,
                "artifacts": artifacts,
                "model_names": model_names,
                "chosen": chosen,
                "ablations": ablations,
                "thresholds": thresholds,
                "cache_meta": cache_meta,
            }
        )
        print(
            f"    profile={profile_name}: selection={chosen.selection_score:.5f}, "
            f"weighted15={chosen.metrics['weighted_awmae_power_1_5']:.5f}, outcome={chosen.metrics['outcome_accuracy']:.4f}"
        )

    selected = select_profile_result(profile_results)
    feature_cols = selected["feature_cols"]
    artifacts = selected["artifacts"]
    model_names = selected["model_names"]
    chosen = selected["chosen"]
    ablations = selected["ablations"]
    thresholds = selected["thresholds"]
    cache_meta = selected["cache_meta"]
    dropped_cols = dropped_by_profile[selected["profile"]]
    feature_groups = feature_groups_by_profile[selected["profile"]]
    profile_comparison = []
    for item in profile_results:
        res = item["chosen"]
        profile_comparison.append(
            {
                "profile": item["profile"],
                "selected": item is selected,
                "feature_count": len(item["feature_cols"]),
                "selection_score": res.selection_score,
                "weighted_awmae_power_1_5": res.metrics["weighted_awmae_power_1_5"],
                "outcome_accuracy": res.metrics["outcome_accuracy"],
                "exact_accuracy": res.metrics["exact_accuracy"],
                "goal_diff_accuracy": res.metrics["goal_diff_accuracy"],
                "latest_fold_awmae": res.fold_metrics[-1]["weighted_awmae"],
                "cache_used": item["cache_meta"].get("cache_used", False),
            }
        )
    print(f"\n[profile selection] selected={selected['profile']} stage={chosen.config.stage} candidate={chosen.config.name}")

    segment_summary = segment_metrics_for_validation(chosen, artifacts)
    pred_t, pred_o, lambda_t, lambda_o, final_model_names = fit_final_predictions(
        train,
        test,
        feature_cols,
        chosen,
        model_names,
        artifacts=artifacts,
    )
    final_sub = write_submission(test, pred_t, pred_o)
    final_dist = score_distribution(final_sub["team_goals"].values, final_sub["opp_goals"].values)
    final_yearly = yearly_distribution(final_sub["team_goals"].values, final_sub["opp_goals"].values, test["date"])
    write_outputs(
        train,
        test,
        feature_cols,
        dropped_cols,
        feature_groups,
        profile_comparison,
        cache_meta,
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
