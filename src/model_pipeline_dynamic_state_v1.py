"""
Leakage-safe probabilistic dynamic-state pipeline v1.

This file is intentionally standalone:
  * it does not import model_pipeline_v5.py;
  * it does not consume V3/V4/V5/V8 submission anchors, blends, or selectors;
  * test_ground_truth.csv and any friend submission are read only after the
    candidate lock has been written.

Outputs:
  dataset/submission_dynamic_state_v1.csv
  dataset/submission_dynamic_state_v1_config.json
  dataset/submission_dynamic_state_v1_validation_report.txt
  dataset/submission_dynamic_state_v1_candidate_lock.json
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import pickle
import re
import sys
import time
import warnings
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge

try:  # Optional, reported only in FAST_MODE.
    import xgboost as xgb  # noqa: F401
except Exception:  # pragma: no cover - optional dependency
    xgb = None

try:  # Optional FULL_MODE dependency.
    import lightgbm as lgb  # noqa: F401
except Exception:  # pragma: no cover - optional dependency
    lgb = None

try:  # Optional FULL_MODE dependency.
    import catboost  # noqa: F401
except Exception:  # pragma: no cover - optional dependency
    catboost = None

warnings.filterwarnings("ignore")


# ============================================================================
# Paths and constants
# ============================================================================
PIPELINE_VERSION = "dynamic_state_v1"
SEED = 42
PRIMARY_POWER = 1.5
SECONDARY_POWER = 1.3
DEFAULT_FAST_MODE = True

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "dataset"

TRAIN_FINAL = DATA_DIR / "train_final.csv"
TEST_FINAL = DATA_DIR / "test_final.csv"
TRAIN_RAW = DATA_DIR / "train.csv"
TEST_RAW = DATA_DIR / "test.csv"
SAMPLE_SUB = DATA_DIR / "sample submission.csv"
GT_PATH = DATA_DIR / "test_ground_truth.csv"

OUTPUT_SUB = DATA_DIR / "submission_dynamic_state_v1.csv"
OUTPUT_CONFIG = DATA_DIR / "submission_dynamic_state_v1_config.json"
OUTPUT_REPORT = DATA_DIR / "submission_dynamic_state_v1_validation_report.txt"
OUTPUT_LOCK = DATA_DIR / "submission_dynamic_state_v1_candidate_lock.json"
CACHE_DIR = DATA_DIR / "dynamic_state_v1_cache"

PRIMARY_FOLDS = [
    {"name": "F1_2003_2005", "train_end_year": 2002, "valid_start_year": 2003, "valid_end_year": 2005, "weight": 0.05},
    {"name": "F2_2006_2008", "train_end_year": 2005, "valid_start_year": 2006, "valid_end_year": 2008, "weight": 0.15},
    {"name": "F3_2009_2010", "train_end_year": 2008, "valid_start_year": 2009, "valid_end_year": 2010, "weight": 0.30},
    {"name": "F4_2011", "train_end_year": 2010, "valid_start_year": 2011, "valid_end_year": 2011, "weight": 0.50},
]

STRESS_FOLDS = [
    {"name": "H1_2003_end", "train_end_year": 2002, "valid_start_year": 2003, "valid_end_year": None, "weight": 1.0 / 3.0},
    {"name": "H2_2006_end", "train_end_year": 2005, "valid_start_year": 2006, "valid_end_year": None, "weight": 1.0 / 3.0},
    {"name": "H3_2009_end", "train_end_year": 2008, "valid_start_year": 2009, "valid_end_year": None, "weight": 1.0 / 3.0},
]

TRACK_LR = {
    "static": 0.000,
    "slow": 0.015,
    "medium": 0.035,
    "fast": 0.070,
    "aggressive": 0.100,
}

SMOOTHING_CANDIDATES = [25.0, 50.0, 100.0, 200.0]

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

TARGET_NAMES = [
    "attack_state",
    "defense_state",
    "outcome_strength",
    "draw_tendency",
    "total_goals_tendency",
    "goal_diff_tendency",
]

DYNAMIC_DIRECT_FEATURES = [
    "attack_state",
    "defense_state",
    "outcome_strength",
    "draw_tendency",
    "total_goals_tendency",
    "goal_diff_tendency",
    "uncertainty",
    "support_count",
    "staleness_years",
]

STATE_FEATURES = [
    "dyn_team_attack_state",
    "dyn_opp_attack_state",
    "dyn_attack_diff",
    "dyn_team_defense_state",
    "dyn_opp_defense_state",
    "dyn_defense_diff",
    "dyn_team_outcome_strength",
    "dyn_opp_outcome_strength",
    "dyn_outcome_strength_diff",
    "dyn_team_draw_tendency",
    "dyn_opp_draw_tendency",
    "dyn_draw_tendency_diff",
    "dyn_team_total_goals_tendency",
    "dyn_opp_total_goals_tendency",
    "dyn_total_goals_tendency_avg",
    "dyn_team_goal_diff_tendency",
    "dyn_opp_goal_diff_tendency",
    "dyn_goal_diff_tendency_diff",
    "dyn_team_uncertainty",
    "dyn_opp_uncertainty",
    "dyn_uncertainty_avg",
    "dyn_team_support_count",
    "dyn_opp_support_count",
    "dyn_support_min",
    "dyn_team_staleness_years",
    "dyn_opp_staleness_years",
    "dyn_staleness_max",
]

PRIOR_FEATURES = [
    "prior_team_attack",
    "prior_opp_attack",
    "prior_attack_diff",
    "prior_team_defense",
    "prior_opp_defense",
    "prior_defense_diff",
    "prior_team_outcome",
    "prior_opp_outcome",
    "prior_outcome_diff",
    "prior_team_draw",
    "prior_opp_draw",
    "prior_draw_avg",
    "prior_team_total",
    "prior_opp_total",
    "prior_total_avg",
    "prior_team_gd",
    "prior_opp_gd",
    "prior_gd_diff",
    "prior_team_count",
    "prior_opp_count",
    "prior_count_min",
]

DERIVED_NUMERIC_FEATURES = [
    "year",
    "month",
    "year_since_2000",
    "year_since_2011",
    "is_women_match",
    "is_men_match",
    "neutral",
    "is_home",
    "tournament_weight",
    "is_friendly",
    "is_qualifier",
    "is_major_tournament",
    "is_elite_tournament",
    "women_x_year_since_2011",
    "women_x_tournament_weight",
    "friendly_x_year_since_2011",
    "major_x_year_since_2011",
]

CATEGORICAL_FEATURES = [
    "gender",
    "team_norm",
    "opponent_norm",
    "tournament",
    "venue_country",
    "confederation_team",
    "confederation_opp",
]

EXCLUDE_FEATURES = {
    "Id",
    "match_id",
    "date",
    "team",
    "opponent",
    "team_goals",
    "opp_goals",
    "metric_weight",
    "train_weight",
    "state_key",
    "team_state_key",
    "opp_state_key",
}


# ============================================================================
# Dataclasses
# ============================================================================
@dataclass(frozen=True)
class CandidateConfig:
    name: str
    track: str
    base_lr: float
    smoothing: float
    max_goals: int
    poisson_alpha: float = 1.00
    outcome_gamma: float = 0.16
    total_delta: float = 0.10
    gd_eta: float = 0.12
    empirical_beta: float = 0.22
    tau_total: float = 1.25
    tau_gd: float = 1.20
    use_gender_state: bool = True
    use_confidence_gate: bool = True
    outcome_aware_update: bool = True
    total_goals_state: bool = True
    pair_consistency: bool = True
    static_fallback: bool = True
    ablation_group: str = "core"
    full_mode_only: bool = False


@dataclass
class StateRecord:
    attack_state: float
    defense_state: float
    outcome_strength: float
    draw_tendency: float
    total_goals_tendency: float
    goal_diff_tendency: float
    uncertainty: float
    support_count: float
    last_update_year: float
    prior_source: str = "global"


@dataclass
class CandidateResult:
    config: dict[str, Any]
    metrics: dict[str, float]
    fold_metrics: list[dict[str, Any]]
    stress_metrics: dict[str, Any] | None
    stress_pass: bool | None
    selection_score: float
    acceptance: dict[str, Any]
    risk_components: dict[str, float]
    distribution: dict[str, float]
    segment_metrics: dict[str, dict[str, Any]]
    state_diagnostics: dict[str, Any]
    pair_diagnostics: dict[str, Any]
    cache_used: bool = False


@dataclass
class PredictionWalkResult:
    pred_team: np.ndarray
    pred_opp: np.ndarray
    exp_team: np.ndarray
    exp_opp: np.ndarray
    frames: pd.DataFrame
    state_diagnostics: dict[str, Any]
    pair_diagnostics: dict[str, Any]


# ============================================================================
# General helpers
# ============================================================================
def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_hash(obj: Any) -> str:
    payload = json.dumps(to_jsonable(obj), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        if math.isnan(val) or math.isinf(val):
            return None
        return val
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (float, int, str, bool)) or obj is None:
        return obj
    return str(obj)


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        val = float(x)
        if math.isnan(val) or math.isinf(val):
            return default
        return val
    except Exception:
        return default


def normalize_team_name(value: Any) -> str:
    text = "" if pd.isna(value) else str(value)
    text = text.lower().strip()
    text = re.sub(r"[`'’‘ʼ]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


MATCH_RE = re.compile(r"^(M\d+)")


def extract_match_id(value: Any) -> str:
    text = "" if pd.isna(value) else str(value)
    m = MATCH_RE.search(text)
    if m:
        return m.group(1)
    if "_" in text:
        return text.rsplit("_", 1)[0]
    return text


def clip_array(arr: np.ndarray, low: float, high: float) -> np.ndarray:
    return np.minimum(np.maximum(arr, low), high)


def dependency_versions() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
        "xgboost_available": xgb is not None,
        "lightgbm_available": lgb is not None,
        "catboost_available": catboost is not None,
    }


def source_file_mtimes(include_test: bool = True) -> dict[str, float]:
    paths = [TRAIN_FINAL, TRAIN_RAW, SAMPLE_SUB, Path(__file__).resolve()]
    if include_test:
        paths += [TEST_FINAL, TEST_RAW]
    return {str(p.relative_to(BASE_DIR)): p.stat().st_mtime for p in paths if p.exists()}


# ============================================================================
# Data loading and feature preparation
# ============================================================================
def merge_final_with_raw(final_df: pd.DataFrame, raw_df: pd.DataFrame, is_train: bool) -> pd.DataFrame:
    raw_cols = [
        "Id",
        "match_id",
        "date",
        "gender",
        "team",
        "opponent",
        "is_home",
        "neutral",
        "tournament",
        "venue_country",
        "confederation_team",
        "confederation_opp",
    ]
    raw_cols = [c for c in raw_cols if c in raw_df.columns]
    raw_meta = raw_df[raw_cols].copy()
    out = final_df.merge(raw_meta, on="Id", how="left", validate="one_to_one")
    if "match_id" not in out.columns or out["match_id"].isna().any():
        extracted = out["Id"].map(extract_match_id)
        if "match_id" in out.columns:
            out["match_id"] = out["match_id"].fillna(extracted)
        else:
            out["match_id"] = extracted
    out["match_id"] = out["match_id"].astype(str)
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    if out["date"].isna().any():
        raise ValueError("Date merge produced missing dates.")
    out["gender"] = out.get("gender", pd.Series("M", index=out.index)).fillna("M").astype(str).str.upper().str.strip()
    out["team"] = out.get("team", pd.Series("", index=out.index)).fillna("").astype(str)
    out["opponent"] = out.get("opponent", pd.Series("", index=out.index)).fillna("").astype(str)
    out["team_norm"] = out["team"].map(normalize_team_name)
    out["opponent_norm"] = out["opponent"].map(normalize_team_name)
    out["year"] = out["date"].dt.year.astype(int)
    out["month"] = out["date"].dt.month.astype(int)
    out["year_since_2000"] = out["year"] - 2000
    out["year_since_2011"] = out["year"] - 2011
    for col in ["is_home", "neutral"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(float)
        else:
            out[col] = 0.0
    for col in ["tournament", "venue_country", "confederation_team", "confederation_opp"]:
        if col in out.columns:
            out[col] = out[col].fillna("UNK").astype(str)
        else:
            out[col] = "UNK"
    out["tournament_weight"] = out["tournament"].map(TOURNAMENT_WEIGHT_MAP).fillna(DEFAULT_TOURNAMENT_WEIGHT).astype(float)
    out["metric_weight"] = out["tournament_weight"]
    out["train_weight"] = out["tournament_weight"]
    out["is_women_match"] = (out["gender"] == "W").astype(float)
    out["is_men_match"] = (out["gender"] == "M").astype(float)
    tournament_lower = out["tournament"].str.lower()
    out["is_friendly"] = (out["tournament"] == "Friendly").astype(float)
    out["is_qualifier"] = tournament_lower.str.contains("qualification", case=False, na=False).astype(float)
    out["is_major_tournament"] = (out["tournament_weight"] >= 1.50).astype(float)
    out["is_elite_tournament"] = (out["tournament_weight"] >= 1.70).astype(float)
    out["women_x_year_since_2011"] = out["is_women_match"] * out["year_since_2011"]
    out["women_x_tournament_weight"] = out["is_women_match"] * out["tournament_weight"]
    out["friendly_x_year_since_2011"] = out["is_friendly"] * out["year_since_2011"]
    out["major_x_year_since_2011"] = out["is_major_tournament"] * out["year_since_2011"]
    if is_train:
        out["team_goals"] = pd.to_numeric(out["team_goals"], errors="coerce").fillna(0).astype(float)
        out["opp_goals"] = pd.to_numeric(out["opp_goals"], errors="coerce").fillna(0).astype(float)
    return out


def load_core_data(read_test: bool = True) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
    train_final = pd.read_csv(TRAIN_FINAL)
    raw_train = pd.read_csv(TRAIN_RAW)
    train = merge_final_with_raw(train_final, raw_train, is_train=True)
    test = None
    sample = None
    if read_test:
        test_final = pd.read_csv(TEST_FINAL)
        raw_test = pd.read_csv(TEST_RAW)
        sample = pd.read_csv(SAMPLE_SUB)
        test = merge_final_with_raw(test_final, raw_test, is_train=False)
    train = train.sort_values(["date", "match_id", "Id"]).reset_index(drop=True)
    if test is not None:
        test = test.sort_values(["date", "match_id", "Id"]).reset_index(drop=True)
    return train, test, sample


# ============================================================================
# Metrics and probability matrices
# ============================================================================
def awmae_loss_array(pred_t: Any, pred_o: Any, true_t: Any, true_o: Any, power: float = PRIMARY_POWER) -> np.ndarray:
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


def mean_awmae(pred_t: Any, pred_o: Any, true_t: Any, true_o: Any, weights: Any = None, power: float = PRIMARY_POWER) -> float:
    losses = awmae_loss_array(pred_t, pred_o, true_t, true_o, power=power)
    if weights is None:
        return float(np.mean(losses))
    weights = np.asarray(weights, dtype=float)
    return float(np.average(losses, weights=weights))


def outcome_accuracy(pred_t: Any, pred_o: Any, true_t: Any, true_o: Any) -> float:
    return float(np.mean(np.sign(np.asarray(pred_t) - np.asarray(pred_o)) == np.sign(np.asarray(true_t) - np.asarray(true_o))))


def exact_accuracy(pred_t: Any, pred_o: Any, true_t: Any, true_o: Any) -> float:
    return float(np.mean((np.asarray(pred_t) == np.asarray(true_t)) & (np.asarray(pred_o) == np.asarray(true_o))))


def goal_diff_accuracy(pred_t: Any, pred_o: Any, true_t: Any, true_o: Any) -> float:
    return float(np.mean((np.asarray(pred_t) - np.asarray(pred_o)) == (np.asarray(true_t) - np.asarray(true_o))))


def metrics_dict(pred_t: Any, pred_o: Any, true_t: Any, true_o: Any, weights: Any = None) -> dict[str, float]:
    return {
        "weighted_awmae_p15": mean_awmae(pred_t, pred_o, true_t, true_o, weights=weights, power=PRIMARY_POWER),
        "unweighted_awmae_p15": mean_awmae(pred_t, pred_o, true_t, true_o, weights=None, power=PRIMARY_POWER),
        "weighted_awmae_p13": mean_awmae(pred_t, pred_o, true_t, true_o, weights=weights, power=SECONDARY_POWER),
        "unweighted_awmae_p13": mean_awmae(pred_t, pred_o, true_t, true_o, weights=None, power=SECONDARY_POWER),
        "outcome_accuracy": outcome_accuracy(pred_t, pred_o, true_t, true_o),
        "exact_accuracy": exact_accuracy(pred_t, pred_o, true_t, true_o),
        "goal_diff_accuracy": goal_diff_accuracy(pred_t, pred_o, true_t, true_o),
    }


LOSS_TENSOR_CACHE: dict[tuple[int, float], np.ndarray] = {}


def loss_tensor(max_goals: int, power: float = PRIMARY_POWER) -> np.ndarray:
    key = (max_goals, power)
    if key in LOSS_TENSOR_CACHE:
        return LOSS_TENSOR_CACHE[key]
    size = max_goals + 1
    tensor = np.zeros((size, size, size, size), dtype=float)
    for pt in range(size):
        for po in range(size):
            for tt in range(size):
                for to in range(size):
                    tensor[pt, po, tt, to] = awmae_loss_array([pt], [po], [tt], [to], power=power)[0]
    LOSS_TENSOR_CACHE[key] = tensor
    return tensor


def poisson_probs(lam: float, max_goals: int) -> np.ndarray:
    lam = float(np.clip(lam, 0.03, 9.5))
    probs = np.zeros(max_goals + 1, dtype=float)
    probs[0] = math.exp(-lam)
    for k in range(1, max_goals + 1):
        probs[k] = probs[k - 1] * lam / k
    total = probs.sum()
    if total <= 0 or not np.isfinite(total):
        probs[:] = 1.0 / len(probs)
    else:
        probs /= total
    return probs


def empirical_score_prior(train_df: pd.DataFrame, max_goals: int) -> np.ndarray:
    counts = np.full((max_goals + 1, max_goals + 1), 0.35, dtype=float)
    a = np.minimum(train_df["team_goals"].astype(int).values, max_goals)
    b = np.minimum(train_df["opp_goals"].astype(int).values, max_goals)
    for x, y in zip(a, b):
        counts[x, y] += 1.0
    counts /= counts.sum()
    return counts


def score_distribution(pred_t: Any, pred_o: Any) -> dict[str, float]:
    pred_t = np.asarray(pred_t, dtype=int)
    pred_o = np.asarray(pred_o, dtype=int)
    totals = pred_t + pred_o
    if len(pred_t) == 0:
        return {}
    pairs = pd.Series([f"{a}-{b}" for a, b in zip(pred_t, pred_o)]).value_counts(normalize=True)
    return {
        "rows": int(len(pred_t)),
        "avg_team_goals": float(np.mean(pred_t)),
        "avg_opp_goals": float(np.mean(pred_o)),
        "avg_total_goals": float(np.mean(totals)),
        "draw_share": float(np.mean(pred_t == pred_o)),
        "home_or_team_win_share": float(np.mean(pred_t > pred_o)),
        "score_ge5": float(np.mean((pred_t >= 5) | (pred_o >= 5))),
        "score_ge6": float(np.mean((pred_t >= 6) | (pred_o >= 6))),
        "top1_score_share": float(pairs.iloc[0]) if len(pairs) else 0.0,
        "top3_score_share": float(pairs.iloc[:3].sum()) if len(pairs) else 0.0,
    }


def matrix_from_heads(
    lambda_team: float,
    lambda_opp: float,
    outcome_probs: np.ndarray,
    total_pred: float,
    gd_pred: float,
    empirical_prior: np.ndarray,
    config: CandidateConfig,
) -> np.ndarray:
    max_goals = config.max_goals
    size = max_goals + 1
    p_team = poisson_probs(lambda_team, max_goals)
    p_opp = poisson_probs(lambda_opp, max_goals)
    matrix = np.outer(p_team, p_opp)
    matrix = np.power(np.maximum(matrix, 1e-12), config.poisson_alpha)

    outcome_probs = np.asarray(outcome_probs, dtype=float)
    if outcome_probs.shape[0] != 3 or not np.all(np.isfinite(outcome_probs)):
        outcome_probs = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=float)
    outcome_probs = np.maximum(outcome_probs, 1e-4)
    outcome_probs = outcome_probs / outcome_probs.sum()

    total_prob = poisson_probs(max(total_pred, 0.05), max_goals * 2)
    gd_values = np.arange(-max_goals, max_goals + 1)
    gd_weight_lookup = np.exp(-np.abs(gd_values - gd_pred) / max(config.tau_gd, 1e-3))
    gd_weight_lookup = np.maximum(gd_weight_lookup, 1e-8)

    for a in range(size):
        for b in range(size):
            cls_idx = 2 if a > b else 1 if a == b else 0
            matrix[a, b] *= outcome_probs[cls_idx] ** config.outcome_gamma
            matrix[a, b] *= total_prob[a + b] ** config.total_delta
            matrix[a, b] *= gd_weight_lookup[(a - b) + max_goals] ** config.gd_eta
            matrix[a, b] *= max(empirical_prior[a, b], 1e-12) ** config.empirical_beta

    matrix = np.maximum(matrix, 1e-12)
    s = matrix.sum()
    if s <= 0 or not np.isfinite(s):
        matrix[:] = 1.0 / matrix.size
    else:
        matrix /= s
    return matrix


def matrix_expected_signals(matrix: np.ndarray) -> dict[str, float]:
    size = matrix.shape[0]
    goals = np.arange(size)
    exp_team = float((matrix * goals[:, None]).sum())
    exp_opp = float((matrix * goals[None, :]).sum())
    p_win = float(np.tril(matrix, -1).sum())  # row goal > col goal after transpose? corrected below
    p_loss = float(np.triu(matrix, 1).sum())
    # np.tril(matrix, -1) is a > b because rows are team goals.
    p_draw = float(np.trace(matrix))
    return {
        "expected_team_goals": exp_team,
        "expected_opp_goals": exp_opp,
        "expected_total_goals": exp_team + exp_opp,
        "expected_gd": exp_team - exp_opp,
        "p_win": p_win,
        "p_draw": p_draw,
        "p_loss": p_loss,
        "confidence": max(p_win, p_draw, p_loss),
    }


def choose_erm_score(matrix: np.ndarray, max_goals: int) -> tuple[int, int, float]:
    expected_loss = np.tensordot(loss_tensor(max_goals, PRIMARY_POWER), matrix, axes=([2, 3], [0, 1]))
    idx = np.unravel_index(int(np.argmin(expected_loss)), expected_loss.shape)
    return int(idx[0]), int(idx[1]), float(expected_loss[idx])


def choose_joint_pair_score(matrix_a: np.ndarray, matrix_b: np.ndarray, max_goals: int) -> tuple[int, int, float]:
    tensor = loss_tensor(max_goals, PRIMARY_POWER)
    loss_a = np.tensordot(tensor, matrix_a, axes=([2, 3], [0, 1]))
    loss_b_raw = np.tensordot(tensor, matrix_b, axes=([2, 3], [0, 1]))
    joint = loss_a + loss_b_raw.T
    idx = np.unravel_index(int(np.argmin(joint)), joint.shape)
    return int(idx[0]), int(idx[1]), float(joint[idx])


# ============================================================================
# Prior and dynamic state
# ============================================================================
class StatePriors:
    def __init__(self, train_df: pd.DataFrame, smoothing: float, use_gender_state: bool = True):
        self.smoothing = float(smoothing)
        self.use_gender_state = bool(use_gender_state)
        self.global_mean: dict[str, float] = {}
        self.tables: dict[str, dict[Any, dict[str, Any]]] = {}
        self._fit(train_df)

    @staticmethod
    def _state_targets(df: pd.DataFrame) -> pd.DataFrame:
        work = df[["gender", "team_norm", "tournament", "year", "team_goals", "opp_goals"]].copy()
        work["confederation_team"] = df.get("confederation_team", pd.Series("UNK", index=df.index)).fillna("UNK").astype(str)
        work["attack_state"] = df["team_goals"].astype(float)
        work["defense_state"] = df["opp_goals"].astype(float)
        gd = df["team_goals"].astype(float) - df["opp_goals"].astype(float)
        work["outcome_strength"] = np.sign(gd).astype(float)
        work["draw_tendency"] = (gd == 0).astype(float)
        work["total_goals_tendency"] = df["team_goals"].astype(float) + df["opp_goals"].astype(float)
        work["goal_diff_tendency"] = gd
        if not work.empty:
            work["gender"] = work["gender"].fillna("M").astype(str).str.upper().str.strip()
            work["team_norm"] = work["team_norm"].map(normalize_team_name)
            work["tournament"] = work["tournament"].fillna("UNK").astype(str)
        return work

    def _fit(self, train_df: pd.DataFrame) -> None:
        work = self._state_targets(train_df)
        for target in TARGET_NAMES:
            self.global_mean[target] = float(work[target].mean())
        self.global_count = int(len(work))
        self.global_last_year = int(work["year"].max()) if len(work) else 2000

        self.tables["global"] = {
            "__global__": {
                "count": self.global_count,
                "last_year": self.global_last_year,
                **{target: self.global_mean[target] for target in TARGET_NAMES},
                "source": "global",
            }
        }
        group_specs = {
            "team_gender": ["gender", "team_norm"],
            "team_global": ["team_norm"],
            "confed_gender": ["gender", "confederation_team"],
            "tournament_gender": ["gender", "tournament"],
            "gender_global": ["gender"],
        }
        for name, keys in group_specs.items():
            self.tables[name] = self._build_group_table(work, keys, name)

    def _build_group_table(self, work: pd.DataFrame, keys: list[str], source: str) -> dict[Any, dict[str, Any]]:
        if work.empty:
            return {}
        agg_dict = {target: "sum" for target in TARGET_NAMES}
        agg_dict["year"] = "max"
        grouped = work.groupby(keys, dropna=False).agg(agg_dict)
        counts = work.groupby(keys, dropna=False).size()
        table: dict[Any, dict[str, Any]] = {}
        for key, row in grouped.iterrows():
            count = int(counts.loc[key])
            if not isinstance(key, tuple):
                key = (key,)
            values = {
                target: float((row[target] + self.global_mean[target] * self.smoothing) / (count + self.smoothing))
                for target in TARGET_NAMES
            }
            values["count"] = count
            values["last_year"] = int(row["year"])
            values["source"] = source
            table[tuple(str(v) for v in key)] = values
        return table

    def _gender_key(self, gender: Any) -> str:
        return str(gender if self.use_gender_state else "ALL").upper().strip()

    def lookup_values(self, gender: Any, team_norm: Any, confed: Any, tournament: Any) -> dict[str, Any]:
        gender_key = self._gender_key(gender)
        raw_gender = str(gender).upper().strip()
        team_key = normalize_team_name(team_norm)
        confed_key = "UNK" if pd.isna(confed) else str(confed)
        tournament_key = "UNK" if pd.isna(tournament) else str(tournament)
        candidates = [
            ("team_gender", (gender_key if self.use_gender_state else "ALL", team_key)),
            ("team_global", (team_key,)),
            ("confed_gender", (raw_gender, confed_key)),
            ("tournament_gender", (raw_gender, tournament_key)),
            ("gender_global", (raw_gender,)),
            ("global", "__global__"),
        ]
        for table_name, key in candidates:
            if table_name == "team_gender" and not self.use_gender_state:
                # The table was trained by true gender; fallback to team_global for gender-off ablation.
                continue
            if table_name == "global":
                return self.tables["global"]["__global__"].copy()
            table = self.tables.get(table_name, {})
            lookup_key = key if isinstance(key, tuple) else (key,)
            lookup_key = tuple(str(v) for v in lookup_key)
            if lookup_key in table:
                return table[lookup_key].copy()
        return self.tables["global"]["__global__"].copy()

    def make_state(self, gender: Any, team_norm: Any, confed: Any, tournament: Any, current_year: float) -> StateRecord:
        vals = self.lookup_values(gender, team_norm, confed, tournament)
        count = float(vals.get("count", 0.0))
        last_year = float(vals.get("last_year", current_year - 8.0))
        uncertainty = float(np.clip(1.15 / math.sqrt(max(count, 1.0)), 0.05, 1.35))
        return StateRecord(
            attack_state=float(np.clip(vals["attack_state"], 0.0, 7.0)),
            defense_state=float(np.clip(vals["defense_state"], 0.0, 7.0)),
            outcome_strength=float(np.clip(vals["outcome_strength"], -1.0, 1.0)),
            draw_tendency=float(np.clip(vals["draw_tendency"], 0.0, 0.8)),
            total_goals_tendency=float(np.clip(vals["total_goals_tendency"], 0.0, 9.0)),
            goal_diff_tendency=float(np.clip(vals["goal_diff_tendency"], -6.0, 6.0)),
            uncertainty=uncertainty,
            support_count=count,
            last_update_year=last_year,
            prior_source=str(vals.get("source", "global")),
        )


class DynamicStateManager:
    def __init__(self, priors: StatePriors, config: CandidateConfig):
        self.priors = priors
        self.config = config
        self.states: dict[str, StateRecord] = {}
        self.diag: dict[str, Any] = {
            "updates": 0,
            "truth_updates": 0,
            "prediction_updates": 0,
            "decay_applications": 0,
            "total_abs_delta": 0.0,
            "max_abs_delta": 0.0,
            "lr_eff_sum": 0.0,
            "confidence_gate_sum": 0.0,
            "tournament_factor_sum": 0.0,
            "gender_support_factor_sum": 0.0,
            "staleness_factor_sum": 0.0,
            "minor_team_safety_factor_sum": 0.0,
            "feature_rows": 0,
            "high_staleness_feature_rows": 0,
            "low_support_feature_rows": 0,
            "prior_sources": defaultdict(int),
        }

    def clone(self) -> "DynamicStateManager":
        other = DynamicStateManager(self.priors, self.config)
        other.states = copy.deepcopy(self.states)
        other.diag = copy.deepcopy(self.diag)
        return other

    def state_key(self, gender: Any, team_norm: Any) -> str:
        gender_key = str(gender if self.config.use_gender_state else "ALL").upper().strip()
        return f"{gender_key}::{normalize_team_name(team_norm)}"

    def _ensure_state(self, row: pd.Series, side: str) -> StateRecord:
        if side == "team":
            team_norm = row["team_norm"]
            confed = row.get("confederation_team", "UNK")
        else:
            team_norm = row["opponent_norm"]
            confed = row.get("confederation_opp", "UNK")
        key = self.state_key(row["gender"], team_norm)
        if key not in self.states:
            self.states[key] = self.priors.make_state(row["gender"], team_norm, confed, row.get("tournament", "UNK"), row["year"])
            self.diag["prior_sources"][self.states[key].prior_source] += 1
        self._decay_to_prior(key, row, side)
        return self.states[key]

    def _decay_to_prior(self, key: str, row: pd.Series, side: str) -> None:
        if self.config.base_lr <= 0.0:
            return
        state = self.states[key]
        current_year = float(row["year"])
        stale = max(0.0, current_year - state.last_update_year)
        if stale < 3.0 and state.support_count >= 5:
            return
        if side == "team":
            team_norm = row["team_norm"]
            confed = row.get("confederation_team", "UNK")
        else:
            team_norm = row["opponent_norm"]
            confed = row.get("confederation_opp", "UNK")
        prior = self.priors.make_state(row["gender"], team_norm, confed, row.get("tournament", "UNK"), row["year"])
        decay = min(0.35, max(0.0, 0.025 * stale) + (0.05 if state.support_count < 5 else 0.0))
        if decay <= 0.0:
            return
        for name in TARGET_NAMES:
            old = getattr(state, name)
            setattr(state, name, float((1.0 - decay) * old + decay * getattr(prior, name)))
        state.uncertainty = float(np.clip((1.0 - decay) * state.uncertainty + decay * prior.uncertainty, 0.03, 1.5))
        self.diag["decay_applications"] += 1

    def feature_dict(self, row: pd.Series) -> dict[str, float]:
        team = self._ensure_state(row, "team")
        opp = self._ensure_state(row, "opp")
        current_year = float(row["year"])
        team_stale = max(0.0, current_year - team.last_update_year)
        opp_stale = max(0.0, current_year - opp.last_update_year)
        self.diag["feature_rows"] += 1
        if max(team_stale, opp_stale) >= 6:
            self.diag["high_staleness_feature_rows"] += 1
        if min(team.support_count, opp.support_count) < 6:
            self.diag["low_support_feature_rows"] += 1
        total_team = team.total_goals_tendency if self.config.total_goals_state else 0.0
        total_opp = opp.total_goals_tendency if self.config.total_goals_state else 0.0
        return {
            "dyn_team_attack_state": team.attack_state,
            "dyn_opp_attack_state": opp.attack_state,
            "dyn_attack_diff": team.attack_state - opp.attack_state,
            "dyn_team_defense_state": team.defense_state,
            "dyn_opp_defense_state": opp.defense_state,
            "dyn_defense_diff": opp.defense_state - team.defense_state,
            "dyn_team_outcome_strength": team.outcome_strength,
            "dyn_opp_outcome_strength": opp.outcome_strength,
            "dyn_outcome_strength_diff": team.outcome_strength - opp.outcome_strength,
            "dyn_team_draw_tendency": team.draw_tendency,
            "dyn_opp_draw_tendency": opp.draw_tendency,
            "dyn_draw_tendency_diff": team.draw_tendency - opp.draw_tendency,
            "dyn_team_total_goals_tendency": total_team,
            "dyn_opp_total_goals_tendency": total_opp,
            "dyn_total_goals_tendency_avg": 0.5 * (total_team + total_opp),
            "dyn_team_goal_diff_tendency": team.goal_diff_tendency,
            "dyn_opp_goal_diff_tendency": opp.goal_diff_tendency,
            "dyn_goal_diff_tendency_diff": team.goal_diff_tendency + opp.goal_diff_tendency,
            "dyn_team_uncertainty": team.uncertainty,
            "dyn_opp_uncertainty": opp.uncertainty,
            "dyn_uncertainty_avg": 0.5 * (team.uncertainty + opp.uncertainty),
            "dyn_team_support_count": team.support_count,
            "dyn_opp_support_count": opp.support_count,
            "dyn_support_min": min(team.support_count, opp.support_count),
            "dyn_team_staleness_years": team_stale,
            "dyn_opp_staleness_years": opp_stale,
            "dyn_staleness_max": max(team_stale, opp_stale),
        }

    def _effective_lr(self, state: StateRecord, row: pd.Series, signals: dict[str, float], is_truth: bool) -> float:
        base_lr = self.config.base_lr
        if base_lr <= 0.0:
            return 0.0
        confidence = 1.0
        if self.config.use_confidence_gate and not is_truth:
            raw_conf = safe_float(signals.get("confidence", 1.0 / 3.0), 1.0 / 3.0)
            confidence = float(np.clip((raw_conf - 1.0 / 3.0) / 0.45, 0.05, 1.0))
        tournament_weight = safe_float(row.get("tournament_weight", DEFAULT_TOURNAMENT_WEIGHT), DEFAULT_TOURNAMENT_WEIGHT)
        tournament_factor = float(np.clip(0.5 + 0.55 * (tournament_weight / DEFAULT_TOURNAMENT_WEIGHT - 0.7), 0.5, 1.2))
        gender = str(row.get("gender", "M")).upper().strip()
        gender_support_factor = 1.0
        if gender == "W":
            gender_support_factor = float(np.clip(0.5 + state.support_count / 80.0, 0.5, 1.0))
        staleness = max(0.0, safe_float(row.get("year", 0), 0) - state.last_update_year)
        staleness_factor = float(np.clip(1.0 - staleness / 18.0, 0.4, 1.0))
        minor_team_safety = float(np.clip(0.4 + state.support_count / 25.0, 0.4, 1.0))
        eff = base_lr * confidence * tournament_factor * gender_support_factor * staleness_factor * minor_team_safety
        eff = float(np.clip(eff, 0.0, 0.18))
        self.diag["lr_eff_sum"] += eff
        self.diag["confidence_gate_sum"] += confidence
        self.diag["tournament_factor_sum"] += tournament_factor
        self.diag["gender_support_factor_sum"] += gender_support_factor
        self.diag["staleness_factor_sum"] += staleness_factor
        self.diag["minor_team_safety_factor_sum"] += minor_team_safety
        return eff

    @staticmethod
    def _clamp_state_value(name: str, value: float) -> float:
        if name in {"attack_state", "defense_state"}:
            return float(np.clip(value, 0.0, 7.0))
        if name == "outcome_strength":
            return float(np.clip(value, -1.0, 1.0))
        if name == "draw_tendency":
            return float(np.clip(value, 0.0, 0.8))
        if name == "total_goals_tendency":
            return float(np.clip(value, 0.0, 9.0))
        if name == "goal_diff_tendency":
            return float(np.clip(value, -6.0, 6.0))
        return float(value)

    def update_key(self, key: str, row: pd.Series, signals: dict[str, float], is_truth: bool) -> None:
        state = self.states.get(key)
        if state is None:
            state = self.priors.make_state(row["gender"], key.split("::", 1)[-1], row.get("confederation_team", "UNK"), row.get("tournament", "UNK"), row["year"])
            self.states[key] = state
        eff = self._effective_lr(state, row, signals, is_truth)
        if eff <= 0.0:
            return
        update_names = ["attack_state", "defense_state", "goal_diff_tendency"]
        if self.config.total_goals_state:
            update_names.append("total_goals_tendency")
        if self.config.outcome_aware_update:
            update_names.extend(["outcome_strength", "draw_tendency"])
        max_delta = 0.30 if self.config.track in {"slow", "medium"} else 0.45
        for name in update_names:
            if name not in signals:
                continue
            old = getattr(state, name)
            desired = self._clamp_state_value(name, safe_float(signals[name], old))
            delta = float(np.clip(eff * (desired - old), -max_delta, max_delta))
            setattr(state, name, self._clamp_state_value(name, old + delta))
            self.diag["total_abs_delta"] += abs(delta)
            self.diag["max_abs_delta"] = max(float(self.diag["max_abs_delta"]), abs(delta))
        state.uncertainty = float(np.clip((1.0 - eff) * state.uncertainty + eff * (1.0 - min(signals.get("confidence", 0.65), 0.98)), 0.03, 1.5))
        state.support_count += 1.0
        state.last_update_year = float(row["year"])
        self.diag["updates"] += 1
        if is_truth:
            self.diag["truth_updates"] += 1
        else:
            self.diag["prediction_updates"] += 1

    def update_truth_match(self, group: pd.DataFrame) -> None:
        seen: set[str] = set()
        for _, row in group.iterrows():
            key = self.state_key(row["gender"], row["team_norm"])
            if key in seen:
                continue
            self._ensure_state(row, "team")
            gd = safe_float(row["team_goals"]) - safe_float(row["opp_goals"])
            signals = {
                "attack_state": safe_float(row["team_goals"]),
                "defense_state": safe_float(row["opp_goals"]),
                "outcome_strength": float(np.sign(gd)),
                "draw_tendency": float(gd == 0),
                "total_goals_tendency": safe_float(row["team_goals"]) + safe_float(row["opp_goals"]),
                "goal_diff_tendency": gd,
                "confidence": 1.0,
            }
            self.update_key(key, row, signals, is_truth=True)
            seen.add(key)
        if len(seen) == 1 and len(group) == 1:
            row = group.iloc[0]
            opp_key = self.state_key(row["gender"], row["opponent_norm"])
            self._ensure_state(row, "opp")
            gd = safe_float(row["opp_goals"]) - safe_float(row["team_goals"])
            rev = row.copy()
            rev["team_norm"] = row["opponent_norm"]
            rev["opponent_norm"] = row["team_norm"]
            rev["confederation_team"] = row.get("confederation_opp", "UNK")
            signals = {
                "attack_state": safe_float(row["opp_goals"]),
                "defense_state": safe_float(row["team_goals"]),
                "outcome_strength": float(np.sign(gd)),
                "draw_tendency": float(gd == 0),
                "total_goals_tendency": safe_float(row["team_goals"]) + safe_float(row["opp_goals"]),
                "goal_diff_tendency": gd,
                "confidence": 1.0,
            }
            self.update_key(opp_key, rev, signals, is_truth=True)

    def update_prediction_signals(self, row: pd.Series, signals: dict[str, float], side: str = "team") -> None:
        if side == "team":
            team_norm = row["team_norm"]
            key = self.state_key(row["gender"], team_norm)
            self._ensure_state(row, "team")
            sig = {
                "attack_state": signals["expected_team_goals"],
                "defense_state": signals["expected_opp_goals"],
                "outcome_strength": signals["p_win"] - signals["p_loss"],
                "draw_tendency": signals["p_draw"],
                "total_goals_tendency": signals["expected_total_goals"],
                "goal_diff_tendency": signals["expected_gd"],
                "confidence": signals["confidence"],
            }
            self.update_key(key, row, sig, is_truth=False)
        else:
            opp_norm = row["opponent_norm"]
            key = self.state_key(row["gender"], opp_norm)
            self._ensure_state(row, "opp")
            rev = row.copy()
            rev["team_norm"] = row["opponent_norm"]
            rev["opponent_norm"] = row["team_norm"]
            rev["confederation_team"] = row.get("confederation_opp", "UNK")
            sig = {
                "attack_state": signals["expected_opp_goals"],
                "defense_state": signals["expected_team_goals"],
                "outcome_strength": signals["p_loss"] - signals["p_win"],
                "draw_tendency": signals["p_draw"],
                "total_goals_tendency": signals["expected_total_goals"],
                "goal_diff_tendency": -signals["expected_gd"],
                "confidence": signals["confidence"],
            }
            self.update_key(key, rev, sig, is_truth=False)

    def diagnostics(self) -> dict[str, Any]:
        out = dict(self.diag)
        out["prior_sources"] = dict(out["prior_sources"])
        updates = max(1, int(out["updates"]))
        for key in [
            "lr_eff_sum",
            "confidence_gate_sum",
            "tournament_factor_sum",
            "gender_support_factor_sum",
            "staleness_factor_sum",
            "minor_team_safety_factor_sum",
        ]:
            out[key.replace("_sum", "_avg")] = float(out[key]) / updates
        out["avg_abs_delta_per_update"] = float(out["total_abs_delta"]) / updates
        feature_rows = max(1, int(out["feature_rows"]))
        out["high_staleness_feature_rate"] = float(out["high_staleness_feature_rows"]) / feature_rows
        out["low_support_feature_rate"] = float(out["low_support_feature_rows"]) / feature_rows
        return to_jsonable(out)


def add_prior_features(df: pd.DataFrame, priors: StatePriors) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        team_vals = priors.lookup_values(row["gender"], row["team_norm"], row.get("confederation_team", "UNK"), row.get("tournament", "UNK"))
        opp_vals = priors.lookup_values(row["gender"], row["opponent_norm"], row.get("confederation_opp", "UNK"), row.get("tournament", "UNK"))
        rows.append(
            {
                "prior_team_attack": team_vals["attack_state"],
                "prior_opp_attack": opp_vals["attack_state"],
                "prior_attack_diff": team_vals["attack_state"] - opp_vals["attack_state"],
                "prior_team_defense": team_vals["defense_state"],
                "prior_opp_defense": opp_vals["defense_state"],
                "prior_defense_diff": opp_vals["defense_state"] - team_vals["defense_state"],
                "prior_team_outcome": team_vals["outcome_strength"],
                "prior_opp_outcome": opp_vals["outcome_strength"],
                "prior_outcome_diff": team_vals["outcome_strength"] - opp_vals["outcome_strength"],
                "prior_team_draw": team_vals["draw_tendency"],
                "prior_opp_draw": opp_vals["draw_tendency"],
                "prior_draw_avg": 0.5 * (team_vals["draw_tendency"] + opp_vals["draw_tendency"]),
                "prior_team_total": team_vals["total_goals_tendency"],
                "prior_opp_total": opp_vals["total_goals_tendency"],
                "prior_total_avg": 0.5 * (team_vals["total_goals_tendency"] + opp_vals["total_goals_tendency"]),
                "prior_team_gd": team_vals["goal_diff_tendency"],
                "prior_opp_gd": opp_vals["goal_diff_tendency"],
                "prior_gd_diff": team_vals["goal_diff_tendency"] + opp_vals["goal_diff_tendency"],
                "prior_team_count": team_vals.get("count", 0),
                "prior_opp_count": opp_vals.get("count", 0),
                "prior_count_min": min(team_vals.get("count", 0), opp_vals.get("count", 0)),
            }
        )
    prior_df = pd.DataFrame(rows, index=df.index)
    return pd.concat([df.reset_index(drop=True), prior_df.reset_index(drop=True)], axis=1)


def build_training_dynamic_features(train_df: pd.DataFrame, priors: StatePriors, config: CandidateConfig) -> tuple[pd.DataFrame, DynamicStateManager]:
    manager = DynamicStateManager(priors, config)
    frames = []
    ordered = train_df.sort_values(["date", "match_id", "Id"]).reset_index(drop=True)
    for _, group in ordered.groupby("match_id", sort=False):
        feature_rows = [manager.feature_dict(row) for _, row in group.iterrows()]
        dyn = pd.DataFrame(feature_rows, index=group.index)
        frames.append(pd.concat([group.reset_index(drop=True), dyn.reset_index(drop=True)], axis=1))
        manager.update_truth_match(group)
    return pd.concat(frames, ignore_index=True), manager


# ============================================================================
# Preprocessing and models
# ============================================================================
class StablePreprocessor:
    def __init__(self, numeric_cols: list[str], categorical_cols: list[str]):
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols
        self.medians: dict[str, float] = {}
        self.category_maps: dict[str, dict[str, int]] = {}
        self.feature_names: list[str] = []

    def fit(self, df: pd.DataFrame) -> "StablePreprocessor":
        self.medians = {}
        for col in self.numeric_cols:
            vals = pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(np.nan, index=df.index)
            med = float(vals.median()) if vals.notna().any() else 0.0
            self.medians[col] = med
        self.category_maps = {}
        for col in self.categorical_cols:
            vals = df[col].fillna("__MISSING__").astype(str) if col in df.columns else pd.Series("__MISSING__", index=df.index)
            cats = sorted(vals.unique().tolist())
            self.category_maps[col] = {cat: i for i, cat in enumerate(cats)}
        self.feature_names = self.numeric_cols + [f"cat_{c}" for c in self.categorical_cols]
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        arrays = []
        for col in self.numeric_cols:
            vals = pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(np.nan, index=df.index)
            vals = vals.fillna(self.medians.get(col, 0.0)).astype(float).values
            arrays.append(vals)
        for col in self.categorical_cols:
            vals = df[col].fillna("__MISSING__").astype(str) if col in df.columns else pd.Series("__MISSING__", index=df.index)
            mapping = self.category_maps.get(col, {})
            enc = vals.map(mapping).fillna(-1).astype(float).values
            arrays.append(enc)
        if not arrays:
            return np.zeros((len(df), 0), dtype=np.float32)
        return np.vstack(arrays).T.astype(np.float32)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        self.fit(df)
        return self.transform(df)


def infer_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric_cols = []
    for col in df.columns:
        if col in EXCLUDE_FEATURES or col in CATEGORICAL_FEATURES:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
    for col in DERIVED_NUMERIC_FEATURES + PRIOR_FEATURES + STATE_FEATURES:
        if col in df.columns and col not in numeric_cols:
            numeric_cols.append(col)
    categorical_cols = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    return sorted(set(numeric_cols)), categorical_cols


def make_hgb_regressor(loss: str, fast_mode: bool) -> Any:
    max_iter = 80 if fast_mode else 180
    try:
        return HistGradientBoostingRegressor(
            loss=loss,
            learning_rate=0.055,
            max_iter=max_iter,
            max_leaf_nodes=31,
            min_samples_leaf=35,
            l2_regularization=0.08,
            random_state=SEED,
        )
    except Exception:
        return RandomForestRegressor(n_estimators=160 if fast_mode else 320, min_samples_leaf=8, random_state=SEED, n_jobs=1)


def make_hgb_classifier(fast_mode: bool) -> Any:
    try:
        return HistGradientBoostingClassifier(
            learning_rate=0.055,
            max_iter=90 if fast_mode else 200,
            max_leaf_nodes=31,
            min_samples_leaf=35,
            l2_regularization=0.08,
            random_state=SEED,
        )
    except Exception:
        return RandomForestClassifier(n_estimators=180 if fast_mode else 360, min_samples_leaf=8, random_state=SEED, n_jobs=1)


def align_proba(proba: np.ndarray, classes: np.ndarray) -> np.ndarray:
    aligned = np.full((proba.shape[0], 3), 1e-6, dtype=float)
    for idx, cls in enumerate(classes):
        target_idx = int(cls)
        if 0 <= target_idx <= 2:
            aligned[:, target_idx] = proba[:, idx]
    aligned = np.maximum(aligned, 1e-6)
    aligned /= aligned.sum(axis=1, keepdims=True)
    return aligned


def apply_temperature(proba: np.ndarray, temperature: float) -> np.ndarray:
    temperature = max(float(temperature), 1e-3)
    logits = np.log(np.maximum(proba, 1e-8)) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def weighted_log_loss(y: np.ndarray, proba: np.ndarray, weights: np.ndarray) -> float:
    idx = np.arange(len(y))
    p = np.maximum(proba[idx, y.astype(int)], 1e-8)
    return float(np.average(-np.log(p), weights=weights))


def fit_temperature_from_internal_split(base_classifier: Any, X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
    if len(y) < 1200 or len(np.unique(y)) < 3:
        return 1.0
    split = int(len(y) * 0.82)
    if split <= 100 or len(y) - split <= 100:
        return 1.0
    y_fit, y_cal = y[:split], y[split:]
    if len(np.unique(y_fit)) < 3 or len(np.unique(y_cal)) < 3:
        return 1.0
    clf = clone(base_classifier)
    try:
        clf.fit(X[:split], y_fit, sample_weight=weights[:split])
        proba = align_proba(clf.predict_proba(X[split:]), clf.classes_)
    except Exception:
        return 1.0
    best_temp = 1.0
    best_loss = weighted_log_loss(y_cal, proba, weights[split:])
    for temp in np.linspace(0.70, 1.80, 23):
        loss = weighted_log_loss(y_cal, apply_temperature(proba, temp), weights[split:])
        if loss < best_loss:
            best_loss = loss
            best_temp = float(temp)
    return best_temp


class ModelBundle:
    def __init__(self, fast_mode: bool):
        self.fast_mode = fast_mode
        self.preprocessor: StablePreprocessor | None = None
        self.numeric_cols: list[str] = []
        self.categorical_cols: list[str] = []
        self.model_team: Any = None
        self.model_opp: Any = None
        self.model_total: Any = None
        self.model_gd: Any = None
        self.model_outcome: Any = None
        self.outcome_temperature: float = 1.0
        self.backend: str = "sklearn_hgb"

    def fit(self, train_df: pd.DataFrame) -> "ModelBundle":
        self.numeric_cols, self.categorical_cols = infer_feature_columns(train_df)
        self.preprocessor = StablePreprocessor(self.numeric_cols, self.categorical_cols)
        X = self.preprocessor.fit_transform(train_df)
        weights = train_df.get("train_weight", pd.Series(1.0, index=train_df.index)).astype(float).values
        y_team = train_df["team_goals"].astype(float).values
        y_opp = train_df["opp_goals"].astype(float).values
        y_total = y_team + y_opp
        y_gd = y_team - y_opp
        y_outcome = (np.sign(y_gd) + 1).astype(int)

        self.model_team = make_hgb_regressor("poisson", self.fast_mode)
        self.model_opp = make_hgb_regressor("poisson", self.fast_mode)
        self.model_total = make_hgb_regressor("squared_error", self.fast_mode)
        self.model_gd = make_hgb_regressor("squared_error", self.fast_mode)
        self.model_outcome = make_hgb_classifier(self.fast_mode)

        try:
            self.model_team.fit(X, y_team, sample_weight=weights)
            self.model_opp.fit(X, y_opp, sample_weight=weights)
            self.model_total.fit(X, y_total, sample_weight=weights)
            self.model_gd.fit(X, y_gd, sample_weight=weights)
        except Exception:
            self.backend = "sklearn_fallback"
            self.model_team = Ridge(alpha=2.5, random_state=SEED)
            self.model_opp = Ridge(alpha=2.5, random_state=SEED)
            self.model_total = Ridge(alpha=2.5, random_state=SEED)
            self.model_gd = Ridge(alpha=2.5, random_state=SEED)
            self.model_team.fit(X, y_team, sample_weight=weights)
            self.model_opp.fit(X, y_opp, sample_weight=weights)
            self.model_total.fit(X, y_total, sample_weight=weights)
            self.model_gd.fit(X, y_gd, sample_weight=weights)

        self.outcome_temperature = fit_temperature_from_internal_split(self.model_outcome, X, y_outcome, weights)
        try:
            self.model_outcome.fit(X, y_outcome, sample_weight=weights)
        except Exception:
            self.backend = "sklearn_fallback"
            self.model_outcome = LogisticRegression(max_iter=1000, random_state=SEED, multi_class="auto")
            self.model_outcome.fit(X, y_outcome, sample_weight=weights)
        return self

    def predict_heads(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        if self.preprocessor is None:
            raise RuntimeError("ModelBundle must be fitted before prediction.")
        X = self.preprocessor.transform(df)
        lambda_team = clip_array(np.asarray(self.model_team.predict(X), dtype=float), 0.03, 8.5)
        lambda_opp = clip_array(np.asarray(self.model_opp.predict(X), dtype=float), 0.03, 8.5)
        total_pred = clip_array(np.asarray(self.model_total.predict(X), dtype=float), 0.05, 12.0)
        gd_pred = clip_array(np.asarray(self.model_gd.predict(X), dtype=float), -7.0, 7.0)
        try:
            proba = align_proba(self.model_outcome.predict_proba(X), self.model_outcome.classes_)
        except Exception:
            proba = np.tile(np.array([[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]]), (len(df), 1))
        proba = apply_temperature(proba, self.outcome_temperature)
        return {
            "lambda_team": lambda_team,
            "lambda_opp": lambda_opp,
            "total_pred": total_pred,
            "gd_pred": gd_pred,
            "outcome_probs": proba,
        }


# ============================================================================
# Prediction walk and pair handling
# ============================================================================
def reciprocal_pair(group: pd.DataFrame) -> bool:
    if len(group) != 2:
        return False
    a = group.iloc[0]
    b = group.iloc[1]
    return bool(a["team_norm"] == b["opponent_norm"] and a["opponent_norm"] == b["team_norm"])


def update_pair_diagnostics(diag: dict[str, Any], key: str, amount: int = 1) -> None:
    diag[key] = int(diag.get(key, 0)) + amount


def walk_predict(
    df: pd.DataFrame,
    manager: DynamicStateManager,
    model: ModelBundle,
    empirical_prior: np.ndarray,
    config: CandidateConfig,
    truth_available: bool,
) -> PredictionWalkResult:
    ordered = df.sort_values(["date", "match_id", "Id"]).reset_index(drop=True)
    ordered["_walk_pos"] = np.arange(len(ordered), dtype=int)
    pred_team = np.zeros(len(ordered), dtype=int)
    pred_opp = np.zeros(len(ordered), dtype=int)
    exp_team = np.zeros(len(ordered), dtype=float)
    exp_opp = np.zeros(len(ordered), dtype=float)
    pair_diag: dict[str, Any] = {
        "match_groups": 0,
        "two_row_matches": 0,
        "single_row_matches": 0,
        "duplicate_or_multirow_matches": 0,
        "reciprocal_pairs": 0,
        "inconsistent_pairs": 0,
        "pair_consistency_applied": 0,
        "pair_consistency_disabled": 0,
        "independent_rows": 0,
    }
    frame_parts = []

    for _, group in ordered.groupby("match_id", sort=False):
        update_pair_diagnostics(pair_diag, "match_groups")
        if len(group) == 1:
            update_pair_diagnostics(pair_diag, "single_row_matches")
        elif len(group) == 2:
            update_pair_diagnostics(pair_diag, "two_row_matches")
        else:
            update_pair_diagnostics(pair_diag, "duplicate_or_multirow_matches")

        dyn_rows = [manager.feature_dict(row) for _, row in group.iterrows()]
        group_features = pd.concat([group.reset_index(drop=True), pd.DataFrame(dyn_rows)], axis=1)
        heads = model.predict_heads(group_features)
        matrices = []
        signals_list = []
        raw_scores = []
        for i in range(len(group_features)):
            matrix = matrix_from_heads(
                heads["lambda_team"][i],
                heads["lambda_opp"][i],
                heads["outcome_probs"][i],
                heads["total_pred"][i],
                heads["gd_pred"][i],
                empirical_prior,
                config,
            )
            matrices.append(matrix)
            signals_list.append(matrix_expected_signals(matrix))
            raw_scores.append(choose_erm_score(matrix, config.max_goals))

        reciprocal = reciprocal_pair(group_features)
        if len(group_features) == 2 and reciprocal:
            update_pair_diagnostics(pair_diag, "reciprocal_pairs")
            if config.pair_consistency:
                update_pair_diagnostics(pair_diag, "pair_consistency_applied")
                a, b, _ = choose_joint_pair_score(matrices[0], matrices[1], config.max_goals)
                scores = [(a, b), (b, a)]
                combined_matrix = 0.5 * (matrices[0] + matrices[1].T)
                combined_matrix = combined_matrix / combined_matrix.sum()
                sig0 = matrix_expected_signals(combined_matrix)
                sig1 = {
                    "expected_team_goals": sig0["expected_opp_goals"],
                    "expected_opp_goals": sig0["expected_team_goals"],
                    "expected_total_goals": sig0["expected_total_goals"],
                    "expected_gd": -sig0["expected_gd"],
                    "p_win": sig0["p_loss"],
                    "p_draw": sig0["p_draw"],
                    "p_loss": sig0["p_win"],
                    "confidence": sig0["confidence"],
                }
                update_signals = [sig0, sig1]
            else:
                update_pair_diagnostics(pair_diag, "pair_consistency_disabled")
                scores = [(raw_scores[0][0], raw_scores[0][1]), (raw_scores[1][0], raw_scores[1][1])]
                update_signals = signals_list
        else:
            if len(group_features) == 2:
                update_pair_diagnostics(pair_diag, "inconsistent_pairs")
            update_pair_diagnostics(pair_diag, "independent_rows", len(group_features))
            scores = [(x[0], x[1]) for x in raw_scores]
            update_signals = signals_list

        for local_i, (score_t, score_o) in enumerate(scores):
            global_i = int(group_features.loc[local_i, "_walk_pos"])
            pred_team[global_i] = score_t
            pred_opp[global_i] = score_o
            exp_team[global_i] = update_signals[local_i]["expected_team_goals"]
            exp_opp[global_i] = update_signals[local_i]["expected_opp_goals"]

        seen: set[str] = set()
        for local_i, (_, row) in enumerate(group_features.iterrows()):
            key = manager.state_key(row["gender"], row["team_norm"])
            if key not in seen:
                manager.update_prediction_signals(row, update_signals[local_i], "team")
                seen.add(key)
        if len(group_features) == 1:
            row = group_features.iloc[0]
            opp_key = manager.state_key(row["gender"], row["opponent_norm"])
            if opp_key not in seen:
                manager.update_prediction_signals(row, update_signals[0], "opp")

        local = group_features.copy()
        local["pred_team_goals"] = [s[0] for s in scores]
        local["pred_opp_goals"] = [s[1] for s in scores]
        local["expected_team_goals"] = [s["expected_team_goals"] for s in update_signals]
        local["expected_opp_goals"] = [s["expected_opp_goals"] for s in update_signals]
        frame_parts.append(local)

    pred_frame = pd.concat(frame_parts, ignore_index=True) if frame_parts else ordered.copy()
    if truth_available:
        pred_frame["truth_team_goals"] = pred_frame["team_goals"]
        pred_frame["truth_opp_goals"] = pred_frame["opp_goals"]
    return PredictionWalkResult(pred_team, pred_opp, exp_team, exp_opp, pred_frame, manager.diagnostics(), pair_diag)


# ============================================================================
# Candidate evaluation, cache, and selection
# ============================================================================
def fold_split(train: pd.DataFrame, fold: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_mask = train["year"] <= int(fold["train_end_year"])
    valid_end = fold.get("valid_end_year")
    valid_mask = train["year"] >= int(fold["valid_start_year"])
    if valid_end is not None:
        valid_mask &= train["year"] <= int(valid_end)
    return train.loc[train_mask].copy(), train.loc[valid_mask].copy()


def cache_key_for_candidate(
    config: CandidateConfig,
    folds: list[dict[str, Any]],
    script_hash: str,
    fast_mode: bool,
    label: str,
) -> str:
    payload = {
        "pipeline_version": PIPELINE_VERSION,
        "label": label,
        "script_hash": script_hash,
        "config_hash": json_hash(asdict(config)),
        "feature_columns": {
            "derived": DERIVED_NUMERIC_FEATURES,
            "prior": PRIOR_FEATURES,
            "state": STATE_FEATURES,
            "categorical": CATEGORICAL_FEATURES,
        },
        "fold_definitions": folds,
        "model_params": {
            "backend": "sklearn_hgb",
            "seed": SEED,
            "fast_mode": fast_mode,
        },
        "lr_config": asdict(config),
        "dependency_versions": dependency_versions(),
        "source_file_mtimes": source_file_mtimes(include_test=False),
    }
    return json_hash(payload)


def load_candidate_cache(key: str) -> CandidateResult | None:
    path = CACHE_DIR / f"{key}.pkl"
    if not path.exists():
        return None
    try:
        with path.open("rb") as f:
            payload = pickle.load(f)
        result = CandidateResult(**payload)
        result.cache_used = True
        return result
    except Exception:
        return None


def save_candidate_cache(key: str, result: CandidateResult) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{key}.pkl"
    payload = asdict(result)
    payload["cache_used"] = False
    with path.open("wb") as f:
        pickle.dump(to_jsonable(payload), f, protocol=pickle.HIGHEST_PROTOCOL)


def combine_fold_metrics(fold_metrics: list[dict[str, Any]], metric_name: str) -> float:
    weights = np.asarray([safe_float(m.get("fold_weight", 1.0), 1.0) for m in fold_metrics], dtype=float)
    values = np.asarray([safe_float(m.get(metric_name), np.nan) for m in fold_metrics], dtype=float)
    mask = np.isfinite(values)
    if not mask.any():
        return float("nan")
    return float(np.average(values[mask], weights=weights[mask]))


def segment_summary(pred_frame: pd.DataFrame, pred_t: np.ndarray, pred_o: np.ndarray) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    masks = {
        "men": pred_frame["gender"].astype(str).str.upper().eq("M").values,
        "women": pred_frame["gender"].astype(str).str.upper().eq("W").values,
        "high_staleness": pred_frame.get("dyn_staleness_max", pd.Series(0, index=pred_frame.index)).astype(float).values >= 6.0,
        "low_support": pred_frame.get("dyn_support_min", pd.Series(99, index=pred_frame.index)).astype(float).values < 6.0,
        "major_tournament": pred_frame.get("tournament_weight", pd.Series(1.0, index=pred_frame.index)).astype(float).values >= 1.50,
        "friendly": pred_frame["tournament"].astype(str).eq("Friendly").values,
    }
    for name, mask in masks.items():
        if int(mask.sum()) < 50:
            out[name] = {"rows": int(mask.sum()), "skipped": True}
            continue
        true_t = pred_frame.loc[mask, "team_goals"].values
        true_o = pred_frame.loc[mask, "opp_goals"].values
        weights = pred_frame.loc[mask, "metric_weight"].values
        out[name] = {"rows": int(mask.sum()), **metrics_dict(pred_t[mask], pred_o[mask], true_t, true_o, weights)}
    return out


def summarize_state_diags(diags: list[dict[str, Any]]) -> dict[str, Any]:
    if not diags:
        return {}
    keys_num = set()
    for diag in diags:
        for k, v in diag.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                keys_num.add(k)
    out = {}
    for key in sorted(keys_num):
        vals = [safe_float(d.get(key), 0.0) for d in diags]
        out[key] = float(np.mean(vals))
    prior_sources: defaultdict[str, int] = defaultdict(int)
    for diag in diags:
        for k, v in diag.get("prior_sources", {}).items():
            prior_sources[k] += int(v)
    out["prior_sources"] = dict(prior_sources)
    return to_jsonable(out)


def summarize_pair_diags(diags: list[dict[str, Any]]) -> dict[str, Any]:
    out: defaultdict[str, int] = defaultdict(int)
    for diag in diags:
        for k, v in diag.items():
            out[k] += int(v)
    out_dict = dict(out)
    pairs = max(1, out_dict.get("two_row_matches", 0))
    out_dict["pair_consistency_pass"] = bool(out_dict.get("inconsistent_pairs", 0) == 0)
    out_dict["inconsistent_pair_rate"] = float(out_dict.get("inconsistent_pairs", 0)) / pairs
    return out_dict


def evaluate_candidate_on_folds(
    train: pd.DataFrame,
    config: CandidateConfig,
    folds: list[dict[str, Any]],
    fast_mode: bool,
    script_hash: str,
    label: str,
    use_cache: bool = True,
    baseline: CandidateResult | None = None,
) -> CandidateResult:
    key = cache_key_for_candidate(config, folds, script_hash, fast_mode, label)
    if use_cache:
        cached = load_candidate_cache(key)
        if cached is not None:
            return cached

    fold_metrics: list[dict[str, Any]] = []
    state_diags: list[dict[str, Any]] = []
    pair_diags: list[dict[str, Any]] = []
    all_frames: list[pd.DataFrame] = []
    all_pred_t: list[np.ndarray] = []
    all_pred_o: list[np.ndarray] = []

    for fold in folds:
        fold_train, fold_valid = fold_split(train, fold)
        if fold_train.empty or fold_valid.empty:
            continue
        priors = StatePriors(fold_train, config.smoothing, config.use_gender_state)
        fold_train_p = add_prior_features(fold_train, priors)
        fold_valid_p = add_prior_features(fold_valid, priors)
        train_dyn, manager_after_train = build_training_dynamic_features(fold_train_p, priors, config)
        model = ModelBundle(fast_mode=fast_mode).fit(train_dyn)
        empirical_prior = empirical_score_prior(fold_train, config.max_goals)
        walk = walk_predict(fold_valid_p, manager_after_train.clone(), model, empirical_prior, config, truth_available=True)

        valid_sorted = walk.frames
        pred_t = valid_sorted["pred_team_goals"].astype(int).values
        pred_o = valid_sorted["pred_opp_goals"].astype(int).values
        true_t = valid_sorted["team_goals"].values
        true_o = valid_sorted["opp_goals"].values
        weights = valid_sorted["metric_weight"].values
        metric = metrics_dict(pred_t, pred_o, true_t, true_o, weights)
        metric.update(
            {
                "fold_name": fold["name"],
                "fold_weight": float(fold.get("weight", 1.0)),
                "rows": int(len(valid_sorted)),
                "train_rows": int(len(fold_train)),
                "valid_year_min": int(valid_sorted["year"].min()),
                "valid_year_max": int(valid_sorted["year"].max()),
                "score_ge5": score_distribution(pred_t, pred_o)["score_ge5"],
            }
        )
        fold_metrics.append(metric)
        state_diags.append(walk.state_diagnostics)
        pair_diags.append(walk.pair_diagnostics)
        all_frames.append(valid_sorted)
        all_pred_t.append(pred_t)
        all_pred_o.append(pred_o)

    if not fold_metrics:
        raise RuntimeError(f"No validation folds could be evaluated for {config.name}.")

    pred_t_all = np.concatenate(all_pred_t)
    pred_o_all = np.concatenate(all_pred_o)
    frame_all = pd.concat(all_frames, ignore_index=True)
    metrics = {
        "weighted_awmae_p15": combine_fold_metrics(fold_metrics, "weighted_awmae_p15"),
        "unweighted_awmae_p15": combine_fold_metrics(fold_metrics, "unweighted_awmae_p15"),
        "weighted_awmae_p13": combine_fold_metrics(fold_metrics, "weighted_awmae_p13"),
        "unweighted_awmae_p13": combine_fold_metrics(fold_metrics, "unweighted_awmae_p13"),
        "outcome_accuracy": combine_fold_metrics(fold_metrics, "outcome_accuracy"),
        "exact_accuracy": combine_fold_metrics(fold_metrics, "exact_accuracy"),
        "goal_diff_accuracy": combine_fold_metrics(fold_metrics, "goal_diff_accuracy"),
    }
    distribution = score_distribution(pred_t_all, pred_o_all)
    segments = segment_summary(frame_all, pred_t_all, pred_o_all)
    state_summary = summarize_state_diags(state_diags)
    pair_summary = summarize_pair_diags(pair_diags)
    risk, selection_score, acceptance = compute_selection_components(
        config=config,
        metrics=metrics,
        fold_metrics=fold_metrics,
        distribution=distribution,
        segments=segments,
        state_diagnostics=state_summary,
        pair_diagnostics=pair_summary,
        baseline=baseline,
    )
    result = CandidateResult(
        config=asdict(config),
        metrics=to_jsonable(metrics),
        fold_metrics=to_jsonable(fold_metrics),
        stress_metrics=None,
        stress_pass=None,
        selection_score=float(selection_score),
        acceptance=to_jsonable(acceptance),
        risk_components=to_jsonable(risk),
        distribution=to_jsonable(distribution),
        segment_metrics=to_jsonable(segments),
        state_diagnostics=to_jsonable(state_summary),
        pair_diagnostics=to_jsonable(pair_summary),
        cache_used=False,
    )
    if use_cache:
        save_candidate_cache(key, result)
    return result


def compute_selection_components(
    config: CandidateConfig,
    metrics: dict[str, float],
    fold_metrics: list[dict[str, Any]],
    distribution: dict[str, float],
    segments: dict[str, dict[str, Any]],
    state_diagnostics: dict[str, Any],
    pair_diagnostics: dict[str, Any],
    baseline: CandidateResult | None,
) -> tuple[dict[str, float], float, dict[str, Any]]:
    if baseline is None:
        base_metrics = metrics
        base_segments = segments
        base_folds = fold_metrics
    else:
        base_metrics = baseline.metrics
        base_segments = baseline.segment_metrics
        base_folds = baseline.fold_metrics

    outcome_drop = max(0.0, base_metrics["outcome_accuracy"] - metrics["outcome_accuracy"])
    gd_drop = max(0.0, base_metrics["goal_diff_accuracy"] - metrics["goal_diff_accuracy"])
    exact_drop = max(0.0, base_metrics["exact_accuracy"] - metrics["exact_accuracy"])
    outcome_drop_penalty = max(0.0, outcome_drop - 0.003)
    gd_drop_penalty = max(0.0, gd_drop - 0.004)
    exact_drop_penalty = max(0.0, exact_drop - 0.004)
    fold_aw = np.asarray([m["weighted_awmae_p15"] for m in fold_metrics], dtype=float)
    fold_instability_penalty = 0.04 * float(np.std(fold_aw)) + 0.10 * max(0.0, float(fold_aw[-1] - np.mean(fold_aw) - 0.06))

    def seg_delta(seg_name: str, metric_name: str, tolerance: float) -> float:
        seg = segments.get(seg_name, {})
        base = base_segments.get(seg_name, {}) if isinstance(base_segments, dict) else {}
        if seg.get("skipped") or base.get("skipped") or seg.get("rows", 0) < 50 or base.get("rows", 0) < 50:
            return 0.0
        return max(0.0, safe_float(seg.get(metric_name)) - safe_float(base.get(metric_name)) - tolerance)

    women_segment_penalty = seg_delta("women", "weighted_awmae_p15", 0.006)
    men = segments.get("men", {})
    base_men = base_segments.get("men", {}) if isinstance(base_segments, dict) else {}
    men_outcome_penalty = 0.0
    if men.get("rows", 0) >= 1000 and base_men.get("rows", 0) >= 1000:
        men_outcome_penalty = max(0.0, safe_float(base_men.get("outcome_accuracy")) - safe_float(men.get("outcome_accuracy")) - 0.003)
    high_staleness_penalty = seg_delta("high_staleness", "weighted_awmae_p15", 0.010)
    tail_penalty = max(0.0, distribution.get("score_ge5", 0.0) - 0.030) * 8.0 + max(0.0, distribution.get("score_ge6", 0.0) - 0.006) * 10.0
    dynamic_instability_penalty = (
        max(0.0, safe_float(state_diagnostics.get("max_abs_delta")) - 0.42) * 0.20
        + max(0.0, safe_float(state_diagnostics.get("high_staleness_feature_rate")) - 0.25) * 0.03
    )
    pair_inconsistency_penalty = safe_float(pair_diagnostics.get("inconsistent_pair_rate")) * 0.50
    risk = {
        "outcome_drop_penalty": outcome_drop_penalty,
        "gd_drop_penalty": gd_drop_penalty,
        "exact_drop_penalty": exact_drop_penalty,
        "fold_instability_penalty": fold_instability_penalty,
        "women_segment_penalty": women_segment_penalty,
        "men_outcome_penalty": men_outcome_penalty,
        "high_staleness_penalty": high_staleness_penalty,
        "tail_penalty": tail_penalty,
        "dynamic_instability_penalty": dynamic_instability_penalty,
        "pair_inconsistency_penalty": pair_inconsistency_penalty,
    }
    selection_score = (
        metrics["weighted_awmae_p15"]
        + 3.0 * outcome_drop_penalty
        + 1.5 * gd_drop_penalty
        + 1.0 * exact_drop_penalty
        + fold_instability_penalty
        + women_segment_penalty
        + men_outcome_penalty
        + high_staleness_penalty
        + tail_penalty
        + dynamic_instability_penalty
        + pair_inconsistency_penalty
    )
    base_fold_map = {m["fold_name"]: m for m in base_folds}
    fold_improvements = 0
    any_fold_worse_gt_020 = False
    for fm in fold_metrics:
        bm = base_fold_map.get(fm["fold_name"], fm)
        if fm["weighted_awmae_p15"] < bm["weighted_awmae_p15"]:
            fold_improvements += 1
        if fm["weighted_awmae_p15"] > bm["weighted_awmae_p15"] + 0.020:
            any_fold_worse_gt_020 = True
    f4_worse = fold_metrics[-1]["weighted_awmae_p15"] - base_folds[-1]["weighted_awmae_p15"] if baseline is not None else 0.0
    weighted_improvement = base_metrics["weighted_awmae_p15"] - metrics["weighted_awmae_p15"]
    acceptance = {
        "weighted_improvement": weighted_improvement,
        "fold_improvements": int(fold_improvements),
        "outcome_drop": outcome_drop,
        "goal_diff_drop": gd_drop,
        "exact_drop": exact_drop,
        "f4_worse": f4_worse,
        "score_ge5": distribution.get("score_ge5", 0.0),
        "pair_consistency_pass": bool(pair_diagnostics.get("pair_consistency_pass", True)),
        "any_fold_worse_gt_020": bool(any_fold_worse_gt_020),
        "primary_accept": bool(
            config.track != "static"
            and weighted_improvement > 0.0
            and outcome_drop <= 0.003
            and gd_drop <= 0.004
            and exact_drop <= 0.004
            and f4_worse <= 0.008
            and distribution.get("score_ge5", 0.0) <= 0.030
            and config.pair_consistency
            and bool(pair_diagnostics.get("pair_consistency_pass", True))
            and (fold_improvements >= 3 or weighted_improvement >= 0.015)
        ),
    }
    return risk, float(selection_score), acceptance


def stress_pass(candidate: CandidateResult, baseline_stress: CandidateResult) -> tuple[bool, dict[str, Any]]:
    stress = candidate.metrics
    base = baseline_stress.metrics
    agg_aw_worse = stress["weighted_awmae_p15"] - base["weighted_awmae_p15"]
    outcome_drop = base["outcome_accuracy"] - stress["outcome_accuracy"]
    fold_worse = []
    base_map = {m["fold_name"]: m for m in baseline_stress.fold_metrics}
    for fm in candidate.fold_metrics:
        bm = base_map.get(fm["fold_name"])
        if bm is None:
            continue
        fold_worse.append({"fold": fm["fold_name"], "delta": fm["weighted_awmae_p15"] - bm["weighted_awmae_p15"]})
    any_fold = any(x["delta"] > 0.020 for x in fold_worse)
    passed = bool(agg_aw_worse <= 0.010 and outcome_drop <= 0.005 and not any_fold)
    return passed, {"aggregate_awmae_delta": agg_aw_worse, "outcome_drop": outcome_drop, "fold_deltas": fold_worse, "passed": passed}


def build_candidate_registry(fast_mode: bool) -> tuple[list[CandidateConfig], list[dict[str, Any]]]:
    max_goals = 8 if fast_mode else 10
    registry: list[CandidateConfig] = []
    for smoothing in SMOOTHING_CANDIDATES:
        registry.append(
            CandidateConfig(
                name=f"baseline_static_s{int(smoothing)}",
                track="static",
                base_lr=0.0,
                smoothing=smoothing,
                max_goals=max_goals,
                ablation_group="static",
            )
        )
    active_tracks = ["slow", "medium"] if fast_mode else ["slow", "medium", "fast", "aggressive"]
    for track in active_tracks:
        for smoothing in SMOOTHING_CANDIDATES:
            registry.append(
                CandidateConfig(
                    name=f"{track}_s{int(smoothing)}",
                    track=track,
                    base_lr=TRACK_LR[track],
                    smoothing=smoothing,
                    max_goals=max_goals,
                    ablation_group="speed_smoothing",
                )
            )
    # Focused validation-only ablations around the middle dynamic setting.
    mid = CandidateConfig(name="medium_s50_gender_state_off", track="medium", base_lr=TRACK_LR["medium"], smoothing=50.0, max_goals=max_goals, use_gender_state=False, ablation_group="gender_state_off")
    registry.append(mid)
    registry.append(CandidateConfig(name="medium_s50_confidence_gate_off", track="medium", base_lr=TRACK_LR["medium"], smoothing=50.0, max_goals=max_goals, use_confidence_gate=False, ablation_group="confidence_gate_off"))
    registry.append(CandidateConfig(name="medium_s50_outcome_update_off", track="medium", base_lr=TRACK_LR["medium"], smoothing=50.0, max_goals=max_goals, outcome_aware_update=False, ablation_group="outcome_aware_update_off"))
    registry.append(CandidateConfig(name="medium_s50_total_state_off", track="medium", base_lr=TRACK_LR["medium"], smoothing=50.0, max_goals=max_goals, total_goals_state=False, ablation_group="total_goals_state_off"))
    registry.append(CandidateConfig(name="medium_s50_pair_consistency_off", track="medium", base_lr=TRACK_LR["medium"], smoothing=50.0, max_goals=max_goals, pair_consistency=False, ablation_group="pair_consistency_off"))
    registry.append(CandidateConfig(name="medium_s50_static_fallback_off", track="medium", base_lr=TRACK_LR["medium"], smoothing=50.0, max_goals=max_goals, static_fallback=False, ablation_group="static_fallback_off"))

    skipped = []
    if fast_mode:
        skipped.append({"item": "fast/aggressive tracks", "reason": "FULL_MODE optional; FAST_MODE default evaluates static/slow/medium only."})
        skipped.append({"item": "multi-speed ensemble", "reason": "Skipped in FAST_MODE runtime budget; single-speed candidates remain apples-to-apples with baseline."})
        skipped.append({"item": "LGB/CatBoost heads", "reason": "FULL_MODE optional; HGB backend used."})
    else:
        skipped.append({"item": "multi-speed ensemble", "reason": "Not implemented in v1 registry; reported as skipped, not used for rescue."})
    return registry, skipped


def select_candidate(
    train: pd.DataFrame,
    candidates: list[CandidateConfig],
    fast_mode: bool,
    script_hash: str,
    use_cache: bool,
) -> tuple[CandidateResult, CandidateResult, list[CandidateResult], CandidateResult | None, list[dict[str, Any]]]:
    print(f"[selection] evaluating {len(candidates)} pre-registered candidates")
    static_configs = [c for c in candidates if c.track == "static"]
    dynamic_configs = [c for c in candidates if c.track != "static"]
    static_results: list[CandidateResult] = []
    for config in static_configs:
        print(f"  static_baseline={config.name}")
        static_results.append(evaluate_candidate_on_folds(train, config, PRIMARY_FOLDS, fast_mode, script_hash, "primary", use_cache, baseline=None))
    baseline_by_smoothing = {float(r.config["smoothing"]): r for r in static_results}
    best_static = min(static_results, key=lambda r: (r.metrics["weighted_awmae_p15"], -r.metrics["outcome_accuracy"]))

    results = list(static_results)
    for config in dynamic_configs:
        print(f"  candidate={config.name}")
        apples_baseline = baseline_by_smoothing.get(float(config.smoothing), best_static)
        res = evaluate_candidate_on_folds(train, config, PRIMARY_FOLDS, fast_mode, script_hash, "primary", use_cache, baseline=apples_baseline)
        results.append(res)

    primary_accepted = [r for r in results if r.config.get("track") != "static" and r.acceptance.get("primary_accept", False)]
    primary_accepted.sort(key=lambda r: (r.selection_score, r.metrics["weighted_awmae_p15"], -r.metrics["outcome_accuracy"]))

    print("[selection] evaluating stress veto for best static fallback")
    best_static_config = CandidateConfig(**best_static.config)
    best_static_stress = evaluate_candidate_on_folds(train, best_static_config, STRESS_FOLDS, fast_mode, script_hash, "stress", use_cache, baseline=None)
    stress_records = []
    selected = best_static
    selected_baseline = best_static
    selected_stress = best_static_stress
    stress_cache_by_smoothing: dict[float, CandidateResult] = {float(best_static_config.smoothing): best_static_stress}
    for candidate in primary_accepted:
        cfg = CandidateConfig(**candidate.config)
        apples_baseline = baseline_by_smoothing.get(float(cfg.smoothing), best_static)
        apples_baseline_cfg = CandidateConfig(**apples_baseline.config)
        if float(cfg.smoothing) in stress_cache_by_smoothing:
            baseline_stress = stress_cache_by_smoothing[float(cfg.smoothing)]
        else:
            baseline_stress = evaluate_candidate_on_folds(
                train,
                apples_baseline_cfg,
                STRESS_FOLDS,
                fast_mode,
                script_hash,
                "stress",
                use_cache,
                baseline=None,
            )
            stress_cache_by_smoothing[float(cfg.smoothing)] = baseline_stress
        print(f"[selection] stress veto candidate={cfg.name}")
        cand_stress = evaluate_candidate_on_folds(train, cfg, STRESS_FOLDS, fast_mode, script_hash, "stress", use_cache, baseline=baseline_stress)
        passed, stress_detail = stress_pass(cand_stress, baseline_stress)
        candidate.stress_metrics = cand_stress.metrics | {"detail": stress_detail, "fold_metrics": cand_stress.fold_metrics}
        candidate.stress_pass = passed
        stress_records.append({"candidate": cfg.name, "baseline": apples_baseline.config["name"], **stress_detail})
        if passed:
            selected = candidate
            selected_baseline = apples_baseline
            selected_stress = cand_stress
            break
    if selected.config.get("track") == "static":
        selected.stress_metrics = best_static_stress.metrics | {"fold_metrics": best_static_stress.fold_metrics}
        selected.stress_pass = True
    return selected, selected_baseline, results, selected_stress, stress_records


def sensitivity_diagnostics(train: pd.DataFrame, selected_config: CandidateConfig, fast_mode: bool, script_hash: str, use_cache: bool) -> list[dict[str, Any]]:
    diagnostics = []
    for delta in [-1, 1]:
        shifted = []
        for fold in PRIMARY_FOLDS:
            start = int(fold["valid_start_year"]) + delta
            end = int(fold["valid_end_year"]) + delta
            if start <= int(fold["train_end_year"]):
                continue
            shifted.append({**fold, "name": f"{fold['name']}_shift_{delta:+d}", "valid_start_year": start, "valid_end_year": end})
        if not shifted:
            continue
        row_count = 0
        for fold in shifted:
            _, val = fold_split(train, fold)
            row_count += len(val)
        if row_count < 500:
            diagnostics.append({"shift": delta, "rows": row_count, "skipped": True, "reason": "row_count_below_500"})
            continue
        result = evaluate_candidate_on_folds(train, selected_config, shifted, fast_mode, script_hash, f"sensitivity_{delta:+d}", use_cache, baseline=None)
        diagnostics.append({"shift": delta, "rows": row_count, "skipped": False, "metrics": result.metrics, "fold_metrics": result.fold_metrics})
    return diagnostics


# ============================================================================
# Final inference and post-lock reporting
# ============================================================================
def fit_final_and_predict(train: pd.DataFrame, test: pd.DataFrame, sample: pd.DataFrame, config: CandidateConfig, fast_mode: bool) -> tuple[pd.DataFrame, dict[str, Any]]:
    priors = StatePriors(train, config.smoothing, config.use_gender_state)
    train_p = add_prior_features(train, priors)
    test_p = add_prior_features(test, priors)
    train_dyn, manager_after_train = build_training_dynamic_features(train_p, priors, config)
    model = ModelBundle(fast_mode=fast_mode).fit(train_dyn)
    empirical_prior = empirical_score_prior(train, config.max_goals)
    walk = walk_predict(test_p, manager_after_train.clone(), model, empirical_prior, config, truth_available=False)
    pred = walk.frames[["Id", "pred_team_goals", "pred_opp_goals"]].rename(
        columns={"pred_team_goals": "team_goals", "pred_opp_goals": "opp_goals"}
    )
    pred["team_goals"] = pred["team_goals"].astype(int)
    pred["opp_goals"] = pred["opp_goals"].astype(int)
    submission = sample[["Id"]].merge(pred, on="Id", how="left")
    if submission[["team_goals", "opp_goals"]].isna().any().any():
        missing = submission.loc[submission["team_goals"].isna(), "Id"].head().tolist()
        raise RuntimeError(f"Missing predictions for sample submission ids: {missing}")
    submission["team_goals"] = submission["team_goals"].astype(int)
    submission["opp_goals"] = submission["opp_goals"].astype(int)
    diagnostics = {
        "model_backend": model.backend,
        "feature_count": len(model.numeric_cols) + len(model.categorical_cols),
        "numeric_feature_count": len(model.numeric_cols),
        "categorical_feature_count": len(model.categorical_cols),
        "state_diagnostics": walk.state_diagnostics,
        "pair_diagnostics": walk.pair_diagnostics,
        "distribution": score_distribution(submission["team_goals"].values, submission["opp_goals"].values),
    }
    return submission, diagnostics


def write_candidate_lock(selected: CandidateResult, validation_payload: dict[str, Any], script_hash: str, config_hash: str) -> str:
    lock_payload = {
        "pipeline_version": PIPELINE_VERSION,
        "timestamp_utc": now_utc_iso(),
        "selected_config": selected.config,
        "validation_metrics": selected.metrics,
        "validation_payload_hash": json_hash(validation_payload),
        "script_hash": script_hash,
        "config_hash": config_hash,
        "constraints": {
            "selected_validation_only": True,
            "friend_csv_read_before_lock": False,
            "test_ground_truth_read_before_lock": False,
            "old_submission_anchor_used": False,
            "validation_or_test_state_updated_with_actual_results": False,
        },
    }
    OUTPUT_LOCK.write_text(json.dumps(to_jsonable(lock_payload), indent=2), encoding="utf-8")
    return file_sha256(OUTPUT_LOCK)


def post_lock_test_metadata_distribution(test: pd.DataFrame, submission: pd.DataFrame) -> dict[str, Any]:
    merged = test[["Id", "date", "year", "gender", "tournament", "tournament_weight"]].merge(submission, on="Id", how="left")
    per_year = {}
    for year, g in merged.groupby("year"):
        per_year[str(int(year))] = {
            "rows": int(len(g)),
            "women_share": float((g["gender"].astype(str).str.upper() == "W").mean()),
            "avg_tournament_weight": float(g["tournament_weight"].mean()),
            **score_distribution(g["team_goals"].values, g["opp_goals"].values),
        }
    return {
        "rows": int(len(merged)),
        "date_min": str(merged["date"].min().date()),
        "date_max": str(merged["date"].max().date()),
        "gender_counts": merged["gender"].value_counts(dropna=False).to_dict(),
        "top_tournaments": merged["tournament"].value_counts().head(12).to_dict(),
        "submission_distribution": score_distribution(merged["team_goals"].values, merged["opp_goals"].values),
        "per_year": per_year,
    }


def local_submission_metrics(submission_path: Path, test: pd.DataFrame, power: float = PRIMARY_POWER) -> dict[str, Any] | None:
    if not submission_path.exists() or not GT_PATH.exists():
        return None
    sub = pd.read_csv(submission_path)
    gt = pd.read_csv(GT_PATH)
    merged = sub.merge(gt, on="Id", suffixes=("_pred", "_true")).merge(test[["Id", "metric_weight", "year", "gender"]], on="Id", how="left")
    if len(merged) != len(sub):
        return {"available": False, "reason": "id_mismatch", "rows": int(len(merged))}
    return {
        "available": True,
        "rows": int(len(merged)),
        "weighted_awmae": mean_awmae(
            merged["team_goals_pred"].values,
            merged["opp_goals_pred"].values,
            merged["team_goals_true"].values,
            merged["opp_goals_true"].values,
            weights=merged["metric_weight"].values,
            power=power,
        ),
        "unweighted_awmae": mean_awmae(
            merged["team_goals_pred"].values,
            merged["opp_goals_pred"].values,
            merged["team_goals_true"].values,
            merged["opp_goals_true"].values,
            weights=None,
            power=power,
        ),
        "outcome_accuracy": outcome_accuracy(
            merged["team_goals_pred"].values,
            merged["opp_goals_pred"].values,
            merged["team_goals_true"].values,
            merged["opp_goals_true"].values,
        ),
        "exact_accuracy": exact_accuracy(
            merged["team_goals_pred"].values,
            merged["opp_goals_pred"].values,
            merged["team_goals_true"].values,
            merged["opp_goals_true"].values,
        ),
        "goal_diff_accuracy": goal_diff_accuracy(
            merged["team_goals_pred"].values,
            merged["opp_goals_pred"].values,
            merged["team_goals_true"].values,
            merged["opp_goals_true"].values,
        ),
        "distribution": score_distribution(merged["team_goals_pred"].values, merged["opp_goals_pred"].values),
    }


def find_friend_csv_after_lock() -> Path | None:
    candidates = []
    for root in [DATA_DIR, BASE_DIR]:
        for path in root.glob("**/*.csv"):
            lower = path.name.lower()
            if "friend" in lower and path.resolve() != OUTPUT_SUB.resolve():
                candidates.append(path)
    if not candidates:
        return None
    candidates.sort(key=lambda p: (p.stat().st_mtime, str(p)), reverse=True)
    return candidates[0]


def evaluate_friend_after_lock(test: pd.DataFrame) -> dict[str, Any] | None:
    friend = find_friend_csv_after_lock()
    if friend is None:
        return None
    metrics15 = local_submission_metrics(friend, test, PRIMARY_POWER)
    metrics13 = local_submission_metrics(friend, test, SECONDARY_POWER)
    if metrics15 is None:
        try:
            sub = pd.read_csv(friend)
            return {"path": str(friend), "ground_truth_available": False, "rows": int(len(sub)), "distribution": score_distribution(sub["team_goals"].values, sub["opp_goals"].values)}
        except Exception as exc:
            return {"path": str(friend), "available": False, "error": str(exc)}
    return {"path": str(friend), "ground_truth_available": GT_PATH.exists(), "power_1_5": metrics15, "power_1_3": metrics13}


def acceptance_decision(
    selected: CandidateResult,
    baseline: CandidateResult,
    lock_exists: bool,
    local15: dict[str, Any] | None,
    friend_report: dict[str, Any] | None,
    final_distribution: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    checks = {
        "selected_validation_only": True,
        "candidate_lock_exists_before_gt_friend_read": lock_exists,
        "no_old_anchor": True,
        "no_friend_influence": True,
        "no_gt_influence": True,
        "no_actual_test_update": True,
        "dynamic_selected": selected.config.get("track") != "static",
        "validation_improved_vs_static": selected.metrics["weighted_awmae_p15"] < baseline.metrics["weighted_awmae_p15"],
        "validation_gd_guard": selected.metrics["goal_diff_accuracy"] >= baseline.metrics["goal_diff_accuracy"] - 0.002,
        "score_ge5_guard": final_distribution.get("score_ge5", 1.0) <= 0.030,
        "stress_veto_pass": bool(selected.stress_pass) if selected.stress_pass is not None else selected.config.get("track") == "static",
    }
    if local15 is not None and local15.get("available"):
        checks["gt_available"] = True
        checks["gt_score_ge5_guard"] = local15.get("distribution", {}).get("score_ge5", 1.0) <= 0.030
    else:
        checks["gt_available"] = False
    if friend_report and friend_report.get("ground_truth_available") and local15 and local15.get("available"):
        friend15 = friend_report["power_1_5"]
        checks["friend_available"] = True
        checks["beats_friend_p15"] = local15["weighted_awmae"] < friend15["weighted_awmae"]
        checks["friend_outcome_guard"] = local15["outcome_accuracy"] >= friend15["outcome_accuracy"] - 0.003
    else:
        checks["friend_available"] = False
    accepted_core = all(
        bool(checks[k])
        for k in [
            "selected_validation_only",
            "candidate_lock_exists_before_gt_friend_read",
            "no_old_anchor",
            "no_friend_influence",
            "no_gt_influence",
            "no_actual_test_update",
            "dynamic_selected",
            "validation_improved_vs_static",
            "validation_gd_guard",
            "score_ge5_guard",
            "stress_veto_pass",
        ]
    )
    if checks.get("friend_available"):
        accepted_core = accepted_core and checks.get("beats_friend_p15", False) and checks.get("friend_outcome_guard", False)
        if not checks.get("beats_friend_p15", False):
            return "TARGET_NOT_REACHED", checks
    if accepted_core:
        return "ACCEPTED_DYNAMIC_STATE_V1", checks
    if checks.get("gt_available"):
        return "TARGET_NOT_REACHED", checks
    return "VALIDATION_ONLY", checks


# ============================================================================
# Reports
# ============================================================================
def short_metrics(metrics: dict[str, Any]) -> str:
    return (
        f"w15={metrics.get('weighted_awmae_p15', metrics.get('weighted_awmae', float('nan'))):.5f}, "
        f"u15={metrics.get('unweighted_awmae_p15', metrics.get('unweighted_awmae', float('nan'))):.5f}, "
        f"w13={metrics.get('weighted_awmae_p13', float('nan')):.5f}, "
        f"out={metrics.get('outcome_accuracy', float('nan')):.4f}, "
        f"exact={metrics.get('exact_accuracy', float('nan')):.4f}, "
        f"gd={metrics.get('goal_diff_accuracy', float('nan')):.4f}"
    )


def build_report(
    selected: CandidateResult,
    baseline: CandidateResult,
    all_results: list[CandidateResult],
    skipped_items: list[dict[str, Any]],
    sensitivity: list[dict[str, Any]],
    stress_records: list[dict[str, Any]],
    final_diagnostics: dict[str, Any],
    test_distribution: dict[str, Any],
    local15: dict[str, Any] | None,
    local13: dict[str, Any] | None,
    friend_report: dict[str, Any] | None,
    lock_hash: str,
    decision: str,
    checks: dict[str, Any],
    max_train_date: str,
    fast_mode: bool,
    script_hash: str,
    config_hash: str,
) -> str:
    lines: list[str] = []
    lines.append("Dynamic State V1 Validation Report")
    lines.append("=" * 40)
    lines.append(f"Decision: {decision}")
    lines.append("")
    lines.append("Constraints declaration")
    lines.append("- Standalone script; no import from model_pipeline_v5.py.")
    lines.append("- No V3/V4/V5/V8 submission anchor/input/blend/selector is read or used.")
    lines.append("- Friend CSV and test_ground_truth.csv are final-reporting-only.")
    lines.append("- Candidate lock was written before post-lock test diagnostics and before GT/friend reads.")
    lines.append("- Validation/test state updates use predicted signals only; train history uses train truth only.")
    lines.append("- Gender drift uses metadata/state keys only, never friend/GT/test labels.")
    lines.append("")
    lines.append("Dependency availability")
    lines.append(json.dumps(dependency_versions(), indent=2))
    lines.append("")
    lines.append(f"Seed and mode: seed={SEED}, mode={'FAST_MODE' if fast_mode else 'FULL_MODE'}")
    lines.append(f"Max train date: {max_train_date}")
    lines.append(f"Script hash: {script_hash}")
    lines.append(f"Selected config hash: {config_hash}")
    lines.append(f"Candidate lock hash: {lock_hash}")
    lines.append("")
    lines.append("Fold definitions")
    lines.append(json.dumps({"primary": PRIMARY_FOLDS, "stress": STRESS_FOLDS}, indent=2))
    lines.append("")
    lines.append("Selected config")
    lines.append(json.dumps(selected.config, indent=2))
    lines.append("")
    lines.append("Static vs dynamic comparison")
    lines.append(f"  baseline_static: {short_metrics(baseline.metrics)}")
    lines.append(f"  selected       : {short_metrics(selected.metrics)}")
    lines.append(f"  delta weighted p1.5 selected-static: {selected.metrics['weighted_awmae_p15'] - baseline.metrics['weighted_awmae_p15']:+.5f}")
    lines.append("")
    lines.append("Ablation table")
    for res in sorted(all_results, key=lambda r: (r.config.get("ablation_group", ""), r.selection_score)):
        selected_mark = "*" if res.config["name"] == selected.config["name"] else " "
        lines.append(
            f"{selected_mark} {res.config['name']:<38} group={res.config.get('ablation_group',''):<24} "
            f"selection={res.selection_score:.5f} {short_metrics(res.metrics)} "
            f"primary_accept={res.acceptance.get('primary_accept')}"
        )
    if skipped_items:
        lines.append("Skipped ablations/items")
        for item in skipped_items:
            lines.append(f"  - {item['item']}: {item['reason']}")
    lines.append("")
    lines.append("Primary fold metrics")
    for fm in selected.fold_metrics:
        lines.append(f"  {fm['fold_name']}: rows={fm['rows']}, {short_metrics(fm)}, score_ge5={fm['score_ge5']:.4f}")
    lines.append("")
    lines.append("Stress metrics")
    if selected.stress_metrics:
        lines.append(json.dumps(selected.stress_metrics, indent=2)[:6000])
    if stress_records:
        lines.append("Stress veto records")
        lines.append(json.dumps(stress_records, indent=2)[:6000])
    lines.append("")
    lines.append("Sensitivity diagnostics")
    lines.append(json.dumps(sensitivity, indent=2)[:6000])
    lines.append("")
    lines.append("Men/women diagnostics")
    lines.append(json.dumps({k: selected.segment_metrics.get(k) for k in ["men", "women"]}, indent=2)[:6000])
    lines.append("")
    lines.append("State update diagnostics")
    lines.append(json.dumps(selected.state_diagnostics, indent=2)[:6000])
    lines.append("")
    lines.append("Pair diagnostics")
    lines.append(json.dumps(selected.pair_diagnostics, indent=2))
    lines.append("")
    lines.append("Final inference diagnostics")
    lines.append(json.dumps(final_diagnostics, indent=2)[:6000])
    lines.append(f"score_ge5 rate: {final_diagnostics['distribution'].get('score_ge5', float('nan')):.5f}")
    lines.append("")
    lines.append("Post-lock per-year test metadata distribution")
    lines.append(json.dumps(test_distribution, indent=2)[:9000])
    lines.append("")
    lines.append("Final read-only GT/friend reporting")
    if local15 is None:
        lines.append("  test_ground_truth.csv unavailable or not evaluated.")
    else:
        lines.append(f"  candidate GT p1.5: {json.dumps(local15, indent=2)[:4000]}")
    if local13 is not None:
        lines.append(f"  candidate GT p1.3: {json.dumps(local13, indent=2)[:4000]}")
    if friend_report is None:
        lines.append("  friend CSV not found.")
    else:
        lines.append(f"  friend report: {json.dumps(friend_report, indent=2)[:5000]}")
    lines.append("")
    lines.append("Leakage checklist")
    for k, v in checks.items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("Acceptance decision")
    lines.append(decision)
    return "\n".join(lines) + "\n"


# ============================================================================
# Main
# ============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Leakage-safe dynamic-state pipeline v1")
    parser.add_argument("--full-mode", action="store_true", help="Enable optional fast/aggressive tracks and larger score matrix.")
    parser.add_argument("--no-cache", action="store_true", help="Disable validation result cache.")
    parser.add_argument("--skip-sensitivity", action="store_true", help="Skip cutoff sensitivity diagnostics.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fast_mode = not args.full_mode if DEFAULT_FAST_MODE else False
    use_cache = not args.no_cache
    np.random.seed(SEED)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    script_hash = file_sha256(Path(__file__).resolve())
    print(f"[dynamic_state_v1] loading core data from {DATA_DIR}")
    train, test, sample = load_core_data(read_test=True)
    assert test is not None and sample is not None
    max_train_date = str(train["date"].max().date())
    print(f"[dynamic_state_v1] train rows={len(train)} test rows={len(test)} max_train_date={max_train_date}")

    candidates, skipped_items = build_candidate_registry(fast_mode)
    registry_hash = json_hash([asdict(c) for c in candidates])
    print(f"[dynamic_state_v1] pre-registered candidate registry hash={registry_hash}")

    selected, baseline, all_results, selected_stress, stress_records = select_candidate(train, candidates, fast_mode, script_hash, use_cache)
    _ = selected_stress

    selected_config = CandidateConfig(**selected.config)
    print(f"[dynamic_state_v1] selected={selected_config.name} track={selected_config.track} validation={short_metrics(selected.metrics)}")

    sensitivity = []
    if args.skip_sensitivity:
        sensitivity = [{"skipped": True, "reason": "command_line_skip"}]
    else:
        print("[dynamic_state_v1] running sensitivity diagnostics")
        sensitivity = sensitivity_diagnostics(train, selected_config, fast_mode, script_hash, use_cache)

    validation_payload = {
        "selected": asdict(selected),
        "baseline": asdict(baseline),
        "all_results": [asdict(r) for r in all_results],
        "sensitivity": sensitivity,
        "stress_records": stress_records,
        "registry_hash": registry_hash,
    }
    config_hash = json_hash(selected.config)

    # Candidate lock is written before final inference diagnostics, aggregate
    # test metadata diagnostics, GT reads, or friend-submission reads.
    lock_hash = write_candidate_lock(selected, validation_payload, script_hash, config_hash)
    print(f"[dynamic_state_v1] candidate lock written hash={lock_hash}")

    print("[dynamic_state_v1] fitting final model and writing submission")
    submission, final_diagnostics = fit_final_and_predict(train, test, sample, selected_config, fast_mode)
    submission.to_csv(OUTPUT_SUB, index=False)

    config_payload = {
        "pipeline_version": PIPELINE_VERSION,
        "timestamp_utc": now_utc_iso(),
        "mode": "FAST_MODE" if fast_mode else "FULL_MODE",
        "seed": SEED,
        "max_train_date": max_train_date,
        "script_hash": script_hash,
        "selected_config_hash": json_hash(selected.config),
        "candidate_registry_hash": registry_hash,
        "candidate_lock_hash": lock_hash,
        "selected_result": asdict(selected),
        "baseline_result": asdict(baseline),
        "all_candidate_results": [asdict(r) for r in all_results],
        "final_inference_diagnostics": final_diagnostics,
        "skipped_items": skipped_items,
        "constraints": {
            "no_model_pipeline_v5_import": True,
            "no_v3_v4_v5_v8_anchor": True,
            "friend_and_gt_final_reporting_only": True,
            "prediction_only_validation_test_updates": True,
            "no_true_test_scores_for_selection": True,
        },
    }

    # Everything below is post-lock, read-only reporting.
    test_distribution = post_lock_test_metadata_distribution(test, submission)
    local15 = local_submission_metrics(OUTPUT_SUB, test, PRIMARY_POWER) if GT_PATH.exists() else None
    local13 = local_submission_metrics(OUTPUT_SUB, test, SECONDARY_POWER) if GT_PATH.exists() else None
    friend_report = evaluate_friend_after_lock(test)
    decision, checks = acceptance_decision(selected, baseline, OUTPUT_LOCK.exists(), local15, friend_report, final_diagnostics["distribution"])
    report = build_report(
        selected=selected,
        baseline=baseline,
        all_results=all_results,
        skipped_items=skipped_items,
        sensitivity=sensitivity,
        stress_records=stress_records,
        final_diagnostics=final_diagnostics,
        test_distribution=test_distribution,
        local15=local15,
        local13=local13,
        friend_report=friend_report,
        lock_hash=lock_hash,
        decision=decision,
        checks=checks,
        max_train_date=max_train_date,
        fast_mode=fast_mode,
        script_hash=script_hash,
        config_hash=config_hash,
    )
    OUTPUT_REPORT.write_text(report, encoding="utf-8")

    # Persist final decision/checks after report-only diagnostics without altering inference.
    config_payload["candidate_lock_hash"] = lock_hash
    config_payload["decision"] = decision
    config_payload["acceptance_checks"] = checks
    config_payload["post_lock_test_metadata_distribution"] = test_distribution
    config_payload["post_lock_gt_metrics_p15"] = local15
    config_payload["post_lock_gt_metrics_p13"] = local13
    config_payload["post_lock_friend_report"] = friend_report
    OUTPUT_CONFIG.write_text(json.dumps(to_jsonable(config_payload), indent=2), encoding="utf-8")

    print(f"[dynamic_state_v1] wrote {OUTPUT_SUB}")
    print(f"[dynamic_state_v1] wrote {OUTPUT_CONFIG}")
    print(f"[dynamic_state_v1] wrote {OUTPUT_REPORT}")
    print(f"[dynamic_state_v1] decision={decision}")


if __name__ == "__main__":
    main()
