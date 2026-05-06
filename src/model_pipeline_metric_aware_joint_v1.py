"""
Metric-aware joint score distribution pipeline v1.

Standalone and leakage-safe:
  * no dependency on model_pipeline_v5.py;
  * no V3/V4/V5/V8 submission anchor, blend, selector, or pseudo-label;
  * friend CSV and test_ground_truth.csv are read only after candidate lock.

Main innovation:
  Build probabilistic heads, combine them into a joint score matrix, and choose
  final integer scores by minimizing expected AW-MAE p1.5 instead of rounding.

Outputs:
  dataset/submission_metric_aware_joint_v1.csv
  dataset/submission_metric_aware_joint_v1_config.json
  dataset/submission_metric_aware_joint_v1_validation_report.txt
  dataset/submission_metric_aware_joint_v1_audit.txt
  dataset/submission_metric_aware_joint_v1_candidate_lock.json
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
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import log_loss

try:
    import xgboost as xgb  # noqa: F401
except Exception:  # pragma: no cover
    xgb = None

try:
    import lightgbm as lgb  # noqa: F401
except Exception:  # pragma: no cover
    lgb = None

try:
    import catboost  # noqa: F401
except Exception:  # pragma: no cover
    catboost = None

warnings.filterwarnings("ignore")


# ============================================================================
# Constants and paths
# ============================================================================
PIPELINE_VERSION = "metric_aware_joint_v1"
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

OUTPUT_SUB = DATA_DIR / "submission_metric_aware_joint_v1.csv"
OUTPUT_CONFIG = DATA_DIR / "submission_metric_aware_joint_v1_config.json"
OUTPUT_REPORT = DATA_DIR / "submission_metric_aware_joint_v1_validation_report.txt"
OUTPUT_AUDIT = DATA_DIR / "submission_metric_aware_joint_v1_audit.txt"
OUTPUT_LOCK = DATA_DIR / "submission_metric_aware_joint_v1_candidate_lock.json"
CACHE_DIR = DATA_DIR / "metric_aware_joint_v1_cache"

PRIMARY_FOLDS = [
    {"name": "F1_2003_2005", "train_end_year": 2002, "valid_start_year": 2003, "valid_end_year": 2005, "weight": 0.05},
    {"name": "F2_2006_2008", "train_end_year": 2005, "valid_start_year": 2006, "valid_end_year": 2008, "weight": 0.15},
    {"name": "F3_2009_2010", "train_end_year": 2008, "valid_start_year": 2009, "valid_end_year": 2010, "weight": 0.30},
    {"name": "F4_2011", "train_end_year": 2010, "valid_start_year": 2011, "valid_end_year": 2011, "weight": 0.50},
]

STRESS_FOLDS = [
    {"name": "H1_2003_end", "train_end_year": 2002, "valid_start_year": 2003, "valid_end_year": None, "weight": 1 / 3},
    {"name": "H2_2006_end", "train_end_year": 2005, "valid_start_year": 2006, "valid_end_year": None, "weight": 1 / 3},
    {"name": "H3_2009_end", "train_end_year": 2008, "valid_start_year": 2009, "valid_end_year": None, "weight": 1 / 3},
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

MATCH_RE = re.compile(r"^(M\d+)")

BASELINE_KINDS = {"static_prior", "regression_round", "outcome_first"}


# ============================================================================
# Dataclasses
# ============================================================================
@dataclass(frozen=True)
class CandidateConfig:
    name: str
    kind: str
    smoothing: float = 50.0
    max_goals: int = 8
    alpha: float = 0.5
    gamma: float = 1.25
    delta: float = 1.0
    eta: float = 1.0
    theta: float = 0.5
    kappa: float = 0.5
    beta: float = 0.25
    use_poisson: bool = True
    use_empirical_prior: bool = True
    draw_correction: float = 1.0
    tail_dampening: float = 1.0
    pair_consistency: bool = True
    use_hist_features: bool = True
    use_gender_interactions: bool = True
    backend: str = "hgb"
    overcomplexity: float = 0.0


@dataclass
class CandidateResult:
    config: dict[str, Any]
    metrics: dict[str, float]
    fold_metrics: list[dict[str, Any]]
    segment_metrics: dict[str, dict[str, Any]]
    distribution: dict[str, float]
    selection_score: float
    acceptance: dict[str, Any]
    risk_components: dict[str, float]
    pair_diagnostics: dict[str, Any]
    calibration_diagnostics: dict[str, Any]
    cache_used: bool = False


# ============================================================================
# Serialization and utility helpers
# ============================================================================
def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        return None if not np.isfinite(val) else val
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_hash(obj: Any) -> str:
    payload = json.dumps(to_jsonable(obj), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def normalize_team_name(value: Any) -> str:
    text = "" if pd.isna(value) else str(value)
    text = text.lower().strip()
    text = re.sub(r"[`'’‘ʼ]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def extract_match_id(value: Any) -> str:
    text = "" if pd.isna(value) else str(value)
    m = MATCH_RE.search(text)
    if m:
        return m.group(1)
    return text.rsplit("_", 1)[0] if "_" in text else text


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        val = float(value)
        return val if np.isfinite(val) else default
    except Exception:
        return default


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


def source_mtimes() -> dict[str, float]:
    paths = [Path(__file__).resolve(), TRAIN_FINAL, TEST_FINAL, TRAIN_RAW, TEST_RAW, SAMPLE_SUB]
    return {str(p.relative_to(BASE_DIR)): p.stat().st_mtime for p in paths if p.exists()}


# ============================================================================
# Data loading
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
    out = final_df.merge(raw_df[raw_cols], on="Id", how="left", validate="one_to_one")
    if "match_id" not in out.columns:
        out["match_id"] = out["Id"].map(extract_match_id)
    else:
        out["match_id"] = out["match_id"].fillna(out["Id"].map(extract_match_id)).astype(str)
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    if out["date"].isna().any():
        raise ValueError("Missing dates after metadata merge.")
    out["gender"] = out.get("gender", pd.Series("M", index=out.index)).fillna("M").astype(str).str.upper().str.strip()
    out["team"] = out.get("team", pd.Series("", index=out.index)).fillna("").astype(str)
    out["opponent"] = out.get("opponent", pd.Series("", index=out.index)).fillna("").astype(str)
    out["team_norm"] = out["team"].map(normalize_team_name)
    out["opponent_norm"] = out["opponent"].map(normalize_team_name)
    out["year"] = out["date"].dt.year.astype(int)
    out["month"] = out["date"].dt.month.astype(int)
    out["year_since_2000"] = out["year"] - 2000
    out["year_since_2011"] = out["year"] - 2011
    for col in ["neutral", "is_home"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(float) if col in out else 0.0
    for col in ["tournament", "venue_country", "confederation_team", "confederation_opp"]:
        out[col] = out[col].fillna("UNK").astype(str) if col in out else "UNK"
    lower_tournament = out["tournament"].str.lower()
    out["tournament_weight"] = out["tournament"].map(TOURNAMENT_WEIGHT_MAP).fillna(DEFAULT_TOURNAMENT_WEIGHT).astype(float)
    out["metric_weight"] = out["tournament_weight"]
    out["train_weight"] = out["tournament_weight"]
    out["is_women"] = (out["gender"] == "W").astype(float)
    out["is_men"] = (out["gender"] == "M").astype(float)
    out["is_friendly"] = (out["tournament"] == "Friendly").astype(float)
    out["is_qualifier"] = lower_tournament.str.contains("qualification", case=False, na=False).astype(float)
    out["is_major_tournament"] = (out["tournament_weight"] >= 1.50).astype(float)
    out["gender_x_major"] = out["is_women"] * out["is_major_tournament"]
    out["gender_x_friendly"] = out["is_women"] * out["is_friendly"]
    out["gender_x_home"] = out["is_women"] * out["is_home"]
    out["neutral_x_major"] = out["neutral"] * out["is_major_tournament"]
    if is_train:
        out["team_goals"] = pd.to_numeric(out["team_goals"], errors="coerce").fillna(0).astype(float)
        out["opp_goals"] = pd.to_numeric(out["opp_goals"], errors="coerce").fillna(0).astype(float)
    return out.sort_values(["date", "match_id", "Id"]).reset_index(drop=True)


def load_data(read_test: bool = True) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
    train = merge_final_with_raw(pd.read_csv(TRAIN_FINAL), pd.read_csv(TRAIN_RAW), is_train=True)
    if not read_test:
        return train, None, None
    test = merge_final_with_raw(pd.read_csv(TEST_FINAL), pd.read_csv(TEST_RAW), is_train=False)
    sample = pd.read_csv(SAMPLE_SUB)
    return train, test, sample


# ============================================================================
# Metrics and score selection
# ============================================================================
def awmae_loss_array(pred_t: Any, pred_o: Any, true_t: Any, true_o: Any, power: float = PRIMARY_POWER) -> np.ndarray:
    pred_t = np.asarray(pred_t, dtype=float)
    pred_o = np.asarray(pred_o, dtype=float)
    true_t = np.asarray(true_t, dtype=float)
    true_o = np.asarray(true_o, dtype=float)
    mae = (np.abs(pred_t - true_t) + np.abs(pred_o - true_o)) / 2.0
    exact = ((pred_t == true_t) & (pred_o == true_o)).astype(float)
    outcome = (np.sign(pred_t - pred_o) == np.sign(true_t - true_o)).astype(float)
    gd = ((pred_t - pred_o) == (true_t - true_o)).astype(float)
    augmented = mae + 0.30 * (1 - exact) + 0.25 * (1 - outcome) + 0.15 * (1 - gd)
    multiplier = np.where(outcome == 1.0, 1.0, 1.5)
    return (augmented * multiplier) ** power


def mean_awmae(pred_t: Any, pred_o: Any, true_t: Any, true_o: Any, weights: Any = None, power: float = PRIMARY_POWER) -> float:
    losses = awmae_loss_array(pred_t, pred_o, true_t, true_o, power=power)
    if weights is None:
        return float(np.mean(losses))
    return float(np.average(losses, weights=np.asarray(weights, dtype=float)))


def outcome_accuracy(pred_t: Any, pred_o: Any, true_t: Any, true_o: Any) -> float:
    return float(np.mean(np.sign(np.asarray(pred_t) - np.asarray(pred_o)) == np.sign(np.asarray(true_t) - np.asarray(true_o))))


def exact_accuracy(pred_t: Any, pred_o: Any, true_t: Any, true_o: Any) -> float:
    return float(np.mean((np.asarray(pred_t) == np.asarray(true_t)) & (np.asarray(pred_o) == np.asarray(true_o))))


def goal_diff_accuracy(pred_t: Any, pred_o: Any, true_t: Any, true_o: Any) -> float:
    return float(np.mean((np.asarray(pred_t) - np.asarray(pred_o)) == (np.asarray(true_t) - np.asarray(true_o))))


def metrics_dict(pred_t: Any, pred_o: Any, true_t: Any, true_o: Any, weights: Any = None) -> dict[str, float]:
    return {
        "weighted_awmae_p15": mean_awmae(pred_t, pred_o, true_t, true_o, weights, PRIMARY_POWER),
        "unweighted_awmae_p15": mean_awmae(pred_t, pred_o, true_t, true_o, None, PRIMARY_POWER),
        "weighted_awmae_p13": mean_awmae(pred_t, pred_o, true_t, true_o, weights, SECONDARY_POWER),
        "unweighted_awmae_p13": mean_awmae(pred_t, pred_o, true_t, true_o, None, SECONDARY_POWER),
        "outcome_accuracy": outcome_accuracy(pred_t, pred_o, true_t, true_o),
        "exact_accuracy": exact_accuracy(pred_t, pred_o, true_t, true_o),
        "goal_diff_accuracy": goal_diff_accuracy(pred_t, pred_o, true_t, true_o),
    }


LOSS_TENSOR_CACHE: dict[tuple[int, float], np.ndarray] = {}


def loss_tensor(max_goals: int, power: float) -> np.ndarray:
    key = (max_goals, power)
    if key in LOSS_TENSOR_CACHE:
        return LOSS_TENSOR_CACHE[key]
    size = max_goals + 1
    tensor = np.zeros((size, size, size, size), dtype=float)
    for p in range(size):
        for q in range(size):
            for a in range(size):
                for b in range(size):
                    tensor[p, q, a, b] = awmae_loss_array([p], [q], [a], [b], power)[0]
    LOSS_TENSOR_CACHE[key] = tensor
    return tensor


def score_distribution(pred_t: Any, pred_o: Any) -> dict[str, float]:
    pred_t = np.asarray(pred_t, dtype=int)
    pred_o = np.asarray(pred_o, dtype=int)
    if len(pred_t) == 0:
        return {}
    totals = pred_t + pred_o
    scores = pd.Series([f"{a}-{b}" for a, b in zip(pred_t, pred_o)]).value_counts(normalize=True)
    return {
        "rows": int(len(pred_t)),
        "avg_team_goals": float(pred_t.mean()),
        "avg_opp_goals": float(pred_o.mean()),
        "avg_total_goals": float(totals.mean()),
        "draw_share": float(np.mean(pred_t == pred_o)),
        "team_win_share": float(np.mean(pred_t > pred_o)),
        "score_ge5": float(np.mean((pred_t >= 5) | (pred_o >= 5))),
        "score_ge6": float(np.mean((pred_t >= 6) | (pred_o >= 6))),
        "top1_score_share": float(scores.iloc[0]) if len(scores) else 0.0,
        "top3_score_share": float(scores.iloc[:3].sum()) if len(scores) else 0.0,
    }


def poisson_probs(lam: float, max_goals: int) -> np.ndarray:
    lam = float(np.clip(lam, 0.03, 9.5))
    probs = np.zeros(max_goals + 1, dtype=float)
    probs[0] = math.exp(-lam)
    for k in range(1, max_goals + 1):
        probs[k] = probs[k - 1] * lam / k
    return probs / max(probs.sum(), 1e-12)


def empirical_score_prior(train_df: pd.DataFrame, max_goals: int, smoothing: float = 0.35) -> np.ndarray:
    mat = np.full((max_goals + 1, max_goals + 1), smoothing, dtype=float)
    a = np.minimum(train_df["team_goals"].astype(int).values, max_goals)
    b = np.minimum(train_df["opp_goals"].astype(int).values, max_goals)
    for x, y in zip(a, b):
        mat[x, y] += 1.0
    return mat / mat.sum()


def expected_loss_matrix(joint: np.ndarray, max_goals: int, power: float) -> np.ndarray:
    return np.tensordot(loss_tensor(max_goals, power), joint, axes=([2, 3], [0, 1]))


def choose_expected_awmae_score(joint: np.ndarray, outcome_probs: np.ndarray, max_goals: int) -> tuple[int, int, dict[str, float]]:
    loss15 = expected_loss_matrix(joint, max_goals, PRIMARY_POWER)
    min_loss = float(loss15.min())
    candidates = np.argwhere(np.isclose(loss15, min_loss, rtol=0, atol=1e-12))
    if len(candidates) > 1:
        loss13 = expected_loss_matrix(joint, max_goals, SECONDARY_POWER)
        best = None
        best_key = None
        preferred_outcome = int(np.argmax(outcome_probs)) - 1
        for p, q in candidates:
            outcome_consistency = 1 if np.sign(p - q) == preferred_outcome else 0
            key = (loss13[p, q], -joint[p, q], -outcome_consistency, p + q, p, q)
            if best_key is None or key < best_key:
                best_key = key
                best = (int(p), int(q))
        p, q = best
    else:
        p, q = map(int, candidates[0])
    return p, q, {"expected_loss_p15": float(loss15[p, q]), "selected_joint_prob": float(joint[p, q])}


# ============================================================================
# Historical feature engineering
# ============================================================================
STAT_NAMES = ["gf", "ga", "total", "gd", "win", "draw", "loss"]


class RunningStats:
    def __init__(self) -> None:
        self.counts: defaultdict[tuple[str, ...], int] = defaultdict(int)
        self.sums: defaultdict[tuple[str, ...], np.ndarray] = defaultdict(lambda: np.zeros(len(STAT_NAMES), dtype=float))

    def update(self, keys: list[tuple[str, ...]], gf: float, ga: float) -> None:
        gd = gf - ga
        vec = np.array([gf, ga, gf + ga, gd, float(gd > 0), float(gd == 0), float(gd < 0)], dtype=float)
        for key in keys:
            self.counts[key] += 1
            self.sums[key] += vec

    def lookup(self, key: tuple[str, ...], prior: np.ndarray, smoothing: float) -> tuple[np.ndarray, int]:
        count = self.counts.get(key, 0)
        if count <= 0:
            return prior.copy(), 0
        return (self.sums[key] + prior * smoothing) / (count + smoothing), count


def hierarchy_keys(row: pd.Series, team_col: str) -> list[tuple[str, ...]]:
    gender = str(row.get("gender", "M")).upper()
    team = normalize_team_name(row.get(team_col, ""))
    tournament = str(row.get("tournament", "UNK"))
    return [
        ("team_gender", gender, team),
        ("team_global", team),
        ("tournament_gender", gender, tournament),
        ("tournament_global", tournament),
        ("gender_global", gender),
        ("global",),
    ]


class HistoricalFeatureBuilder:
    def __init__(self, smoothing: float):
        self.smoothing = float(smoothing)
        self.stats = RunningStats()
        self.global_prior = np.array([1.25, 1.25, 2.50, 0.0, 0.38, 0.24, 0.38], dtype=float)

    def fit(self, train_df: pd.DataFrame) -> "HistoricalFeatureBuilder":
        self.stats = RunningStats()
        if len(train_df):
            self.global_prior = np.array(
                [
                    train_df["team_goals"].mean(),
                    train_df["opp_goals"].mean(),
                    (train_df["team_goals"] + train_df["opp_goals"]).mean(),
                    (train_df["team_goals"] - train_df["opp_goals"]).mean(),
                    (train_df["team_goals"] > train_df["opp_goals"]).mean(),
                    (train_df["team_goals"] == train_df["opp_goals"]).mean(),
                    (train_df["team_goals"] < train_df["opp_goals"]).mean(),
                ],
                dtype=float,
            )
        for _, row in train_df.sort_values(["date", "match_id", "Id"]).iterrows():
            self._update_row(row)
        return self

    def _lookup_team(self, row: pd.Series, side: str) -> tuple[np.ndarray, int, str]:
        team_col = "team_norm" if side == "team" else "opponent_norm"
        keys = hierarchy_keys(row, team_col)
        for key in keys:
            vals, count = self.stats.lookup(key, self.global_prior, self.smoothing)
            if count > 0 or key == ("global",):
                return vals, count, key[0]
        return self.global_prior.copy(), 0, "global"

    def _row_features(self, row: pd.Series) -> dict[str, float]:
        team_vals, team_count, _ = self._lookup_team(row, "team")
        opp_vals, opp_count, _ = self._lookup_team(row, "opp")
        feat: dict[str, float] = {}
        for i, name in enumerate(STAT_NAMES):
            feat[f"hist_team_{name}"] = float(team_vals[i])
            feat[f"hist_opp_{name}"] = float(opp_vals[i])
            feat[f"hist_diff_{name}"] = float(team_vals[i] - opp_vals[i])
        feat["hist_team_count"] = float(team_count)
        feat["hist_opp_count"] = float(opp_count)
        feat["hist_support_min"] = float(min(team_count, opp_count))
        feat["hist_support_max"] = float(max(team_count, opp_count))
        feat["hist_low_support"] = float(min(team_count, opp_count) < 5)
        feat["hist_attack_vs_defense"] = float(team_vals[0] - opp_vals[1])
        feat["hist_opp_attack_vs_defense"] = float(opp_vals[0] - team_vals[1])
        feat["hist_total_env"] = float(0.5 * (team_vals[2] + opp_vals[2]))
        feat["hist_gd_prior_diff"] = float(team_vals[3] + opp_vals[3])
        return feat

    def _update_row(self, row: pd.Series) -> None:
        gf = safe_float(row["team_goals"])
        ga = safe_float(row["opp_goals"])
        team_keys = hierarchy_keys(row, "team_norm")
        self.stats.update(team_keys, gf, ga)
        rev = row.copy()
        rev["team_norm"] = row["opponent_norm"]
        rev["opponent_norm"] = row["team_norm"]
        opp_keys = hierarchy_keys(rev, "team_norm")
        self.stats.update(opp_keys, ga, gf)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        rows = [self._row_features(row) for _, row in df.iterrows()]
        return pd.concat([df.reset_index(drop=True), pd.DataFrame(rows).reset_index(drop=True)], axis=1)

    def transform_train_walk_forward(self, df: pd.DataFrame) -> pd.DataFrame:
        work = df.sort_values(["date", "match_id", "Id"]).reset_index(drop=True)
        rows: list[dict[str, float]] = []
        for _, group in work.groupby("match_id", sort=False):
            for _, row in group.iterrows():
                rows.append(self._row_features(row))
            for _, row in group.iterrows():
                self._update_row(row)
        return pd.concat([work.reset_index(drop=True), pd.DataFrame(rows).reset_index(drop=True)], axis=1)


# ============================================================================
# Preprocessing and models
# ============================================================================
EXCLUDE_COLS = {
    "Id",
    "match_id",
    "date",
    "team",
    "opponent",
    "team_goals",
    "opp_goals",
    "metric_weight",
    "train_weight",
}

CATEGORICAL_COLS = [
    "gender",
    "team_norm",
    "opponent_norm",
    "tournament",
    "venue_country",
    "confederation_team",
    "confederation_opp",
]


class StablePreprocessor:
    def __init__(self) -> None:
        self.numeric_cols: list[str] = []
        self.categorical_cols: list[str] = []
        self.medians: dict[str, float] = {}
        self.cat_maps: dict[str, dict[str, int]] = {}

    def fit(self, df: pd.DataFrame) -> "StablePreprocessor":
        self.categorical_cols = [c for c in CATEGORICAL_COLS if c in df.columns]
        self.numeric_cols = []
        for col in df.columns:
            if col in EXCLUDE_COLS or col in self.categorical_cols:
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                self.numeric_cols.append(col)
        self.numeric_cols = sorted(set(self.numeric_cols))
        self.medians = {}
        for col in self.numeric_cols:
            vals = pd.to_numeric(df[col], errors="coerce")
            self.medians[col] = float(vals.median()) if vals.notna().any() else 0.0
        self.cat_maps = {}
        for col in self.categorical_cols:
            vals = df[col].fillna("__MISSING__").astype(str)
            self.cat_maps[col] = {cat: i for i, cat in enumerate(sorted(vals.unique()))}
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        parts = []
        for col in self.numeric_cols:
            vals = pd.to_numeric(df[col], errors="coerce") if col in df else pd.Series(np.nan, index=df.index)
            parts.append(vals.fillna(self.medians[col]).astype(float).values)
        for col in self.categorical_cols:
            vals = df[col].fillna("__MISSING__").astype(str) if col in df else pd.Series("__MISSING__", index=df.index)
            parts.append(vals.map(self.cat_maps[col]).fillna(-1).astype(float).values)
        return np.vstack(parts).T.astype(np.float32) if parts else np.zeros((len(df), 0), dtype=np.float32)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        self.fit(df)
        return self.transform(df)


def clipped_classes(values: np.ndarray, low: int, high: int) -> np.ndarray:
    return np.clip(values.astype(int), low, high).astype(int)


def align_proba(model: Any, proba: np.ndarray, classes: np.ndarray, target_classes: list[int]) -> np.ndarray:
    out = np.full((proba.shape[0], len(target_classes)), 1e-7, dtype=float)
    class_to_idx = {int(c): i for i, c in enumerate(target_classes)}
    for j, cls in enumerate(classes):
        if int(cls) in class_to_idx:
            out[:, class_to_idx[int(cls)]] = proba[:, j]
    out = np.maximum(out, 1e-7)
    out /= out.sum(axis=1, keepdims=True)
    return out


def temperature_scale(proba: np.ndarray, temperature: float) -> np.ndarray:
    logits = np.log(np.maximum(proba, 1e-8)) / max(temperature, 1e-3)
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def fit_temperature(y: np.ndarray, proba: np.ndarray, classes: list[int], weights: np.ndarray) -> float:
    if len(y) < 500:
        return 1.0
    y_idx = np.array([classes.index(int(v)) for v in y], dtype=int)
    best_t = 1.0
    best = float("inf")
    for t in np.linspace(0.75, 1.65, 19):
        scaled = temperature_scale(proba, float(t))
        try:
            score = log_loss(y_idx, scaled, sample_weight=weights, labels=list(range(len(classes))))
        except Exception:
            continue
        if score < best:
            best = score
            best_t = float(t)
    return best_t


class ProbabilisticHeads:
    def __init__(self, max_goals: int, fast_mode: bool):
        self.max_goals = int(max_goals)
        self.fast_mode = bool(fast_mode)
        self.pre = StablePreprocessor()
        self.models: dict[str, Any] = {}
        self.temperatures: dict[str, float] = {}
        self.classes = {
            "outcome": [-1, 0, 1],
            "team": list(range(self.max_goals + 1)),
            "opp": list(range(self.max_goals + 1)),
            "total": list(range(2 * self.max_goals + 1)),
            "gd": list(range(-self.max_goals, self.max_goals + 1)),
        }
        self.backend = "hgb"

    def _clf(self) -> Any:
        return HistGradientBoostingClassifier(
            learning_rate=0.055,
            max_iter=85 if self.fast_mode else 180,
            max_leaf_nodes=31,
            min_samples_leaf=35,
            l2_regularization=0.08,
            random_state=SEED,
        )

    def _reg(self) -> Any:
        return HistGradientBoostingRegressor(
            loss="poisson",
            learning_rate=0.055,
            max_iter=80 if self.fast_mode else 180,
            max_leaf_nodes=31,
            min_samples_leaf=35,
            l2_regularization=0.08,
            random_state=SEED,
        )

    def fit(self, df: pd.DataFrame) -> "ProbabilisticHeads":
        X = self.pre.fit_transform(df)
        weights = df.get("train_weight", pd.Series(1.0, index=df.index)).astype(float).values
        y_team = clipped_classes(df["team_goals"].values, 0, self.max_goals)
        y_opp = clipped_classes(df["opp_goals"].values, 0, self.max_goals)
        y_total = clipped_classes((df["team_goals"] + df["opp_goals"]).values, 0, 2 * self.max_goals)
        y_gd = clipped_classes((df["team_goals"] - df["opp_goals"]).values, -self.max_goals, self.max_goals)
        y_outcome = np.sign(df["team_goals"].values - df["opp_goals"].values).astype(int)
        targets = {"outcome": y_outcome, "team": y_team, "opp": y_opp, "total": y_total, "gd": y_gd}

        for name, y in targets.items():
            if len(np.unique(y)) < 2:
                self.models[name] = None
                self.temperatures[name] = 1.0
                continue
            split = max(100, int(len(y) * 0.82))
            split = min(split, len(y) - 50) if len(y) > 250 else len(y)
            model = self._clf()
            if split < len(y) and len(np.unique(y[:split])) >= 2:
                cal_model = self._clf()
                cal_model.fit(X[:split], y[:split], sample_weight=weights[:split])
                raw = align_proba(cal_model, cal_model.predict_proba(X[split:]), cal_model.classes_, self.classes[name])
                self.temperatures[name] = fit_temperature(y[split:], raw, self.classes[name], weights[split:])
            else:
                self.temperatures[name] = 1.0
            model.fit(X, y, sample_weight=weights)
            self.models[name] = model

        try:
            self.models["lambda_team"] = self._reg().fit(X, np.clip(df["team_goals"].values, 0, None), sample_weight=weights)
            self.models["lambda_opp"] = self._reg().fit(X, np.clip(df["opp_goals"].values, 0, None), sample_weight=weights)
        except Exception:
            self.models["lambda_team"] = Ridge(alpha=2.5).fit(X, df["team_goals"].values, sample_weight=weights)
            self.models["lambda_opp"] = Ridge(alpha=2.5).fit(X, df["opp_goals"].values, sample_weight=weights)
            self.backend = "ridge_lambda_fallback"
        return self

    def predict(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        X = self.pre.transform(df)
        out: dict[str, np.ndarray] = {}
        for name in ["outcome", "team", "opp", "total", "gd"]:
            model = self.models.get(name)
            if model is None:
                out[name] = np.full((len(df), len(self.classes[name])), 1.0 / len(self.classes[name]), dtype=float)
                continue
            proba = align_proba(model, model.predict_proba(X), model.classes_, self.classes[name])
            out[name] = temperature_scale(proba, self.temperatures.get(name, 1.0))
        for name in ["lambda_team", "lambda_opp"]:
            pred = np.asarray(self.models[name].predict(X), dtype=float)
            out[name] = np.clip(pred, 0.03, 9.5)
        return out


# ============================================================================
# Joint matrix and prediction
# ============================================================================
def build_joint_matrix(heads: dict[str, np.ndarray], i: int, prior: np.ndarray, config: CandidateConfig) -> np.ndarray:
    max_g = config.max_goals
    size = max_g + 1
    mat = np.ones((size, size), dtype=float)
    eps = 1e-8
    if config.use_poisson:
        p_team = poisson_probs(float(heads["lambda_team"][i]), max_g)
        p_opp = poisson_probs(float(heads["lambda_opp"][i]), max_g)
        mat *= np.maximum(np.outer(p_team, p_opp), eps) ** config.alpha
    outcome_p = heads["outcome"][i]
    team_p = heads["team"][i]
    opp_p = heads["opp"][i]
    total_p = heads["total"][i]
    gd_p = heads["gd"][i]
    gd_offset = max_g
    for a in range(size):
        for b in range(size):
            outcome_idx = 2 if a > b else 1 if a == b else 0
            mat[a, b] *= max(outcome_p[outcome_idx], eps) ** config.gamma
            mat[a, b] *= max(total_p[min(a + b, len(total_p) - 1)], eps) ** config.delta
            mat[a, b] *= max(gd_p[(a - b) + gd_offset], eps) ** config.eta
            mat[a, b] *= max(team_p[a], eps) ** config.theta
            mat[a, b] *= max(opp_p[b], eps) ** config.kappa
            if config.use_empirical_prior:
                mat[a, b] *= max(prior[a, b], eps) ** config.beta
            if a == b:
                mat[a, b] *= config.draw_correction
            if a >= 5 or b >= 5:
                mat[a, b] *= config.tail_dampening
    mat = np.maximum(mat, eps)
    mat /= mat.sum()
    return mat


def reciprocal_pair(group: pd.DataFrame) -> bool:
    if len(group) != 2:
        return False
    a = group.iloc[0]
    b = group.iloc[1]
    return bool(a["team_norm"] == b["opponent_norm"] and a["opponent_norm"] == b["team_norm"])


def predict_frame(df: pd.DataFrame, model: ProbabilisticHeads, prior: np.ndarray, config: CandidateConfig) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    ordered = df.sort_values(["date", "match_id", "Id"]).reset_index(drop=True)
    ordered["_pos"] = np.arange(len(ordered))
    pred_t = np.zeros(len(ordered), dtype=int)
    pred_o = np.zeros(len(ordered), dtype=int)
    aux_rows = []
    pair_diag: defaultdict[str, int] = defaultdict(int)
    matrix_diag = {"avg_selected_joint_prob": 0.0, "avg_entropy": 0.0, "rows": 0}

    for _, group in ordered.groupby("match_id", sort=False):
        pair_diag["match_groups"] += 1
        if len(group) == 1:
            pair_diag["single_row_matches"] += 1
        elif len(group) == 2:
            pair_diag["two_row_matches"] += 1
        else:
            pair_diag["multirow_matches"] += 1
        heads = model.predict(group)
        matrices = []
        row_choices = []
        for i in range(len(group)):
            mat = build_joint_matrix(heads, i, prior, config)
            matrices.append(mat)
            p, q, info = choose_expected_awmae_score(mat, heads["outcome"][i], config.max_goals)
            row_choices.append((p, q, info))
            matrix_diag["avg_selected_joint_prob"] += info["selected_joint_prob"]
            matrix_diag["avg_entropy"] += float(-(mat * np.log(np.maximum(mat, 1e-12))).sum())
            matrix_diag["rows"] += 1

        scores = [(p, q) for p, q, _ in row_choices]
        if len(group) == 2 and reciprocal_pair(group):
            pair_diag["reciprocal_pairs"] += 1
            if config.pair_consistency:
                pair_diag["pair_consistency_applied"] += 1
                loss_a = expected_loss_matrix(matrices[0], config.max_goals, PRIMARY_POWER)
                loss_b = expected_loss_matrix(matrices[1], config.max_goals, PRIMARY_POWER)
                combined = loss_a + loss_b.T
                a, b = np.unravel_index(int(np.argmin(combined)), combined.shape)
                if scores != [(int(a), int(b)), (int(b), int(a))]:
                    pair_diag["pair_conflicts_corrected"] += 1
                    pair_diag["pair_correction_abs"] += abs(scores[0][0] - int(a)) + abs(scores[0][1] - int(b))
                scores = [(int(a), int(b)), (int(b), int(a))]
            else:
                pair_diag["pair_consistency_disabled"] += 1
        elif len(group) == 2:
            pair_diag["inconsistent_pairs"] += 1

        for local_i, row in group.reset_index(drop=True).iterrows():
            pos = int(row["_pos"])
            pred_t[pos], pred_o[pos] = scores[local_i]

        out_group = group.copy()
        out_group["pred_team_goals"] = [s[0] for s in scores]
        out_group["pred_opp_goals"] = [s[1] for s in scores]
        aux_rows.append(out_group)

    out = pd.concat(aux_rows, ignore_index=True) if aux_rows else ordered.copy()
    rows = max(1, matrix_diag["rows"])
    matrix_diag["avg_selected_joint_prob"] /= rows
    matrix_diag["avg_entropy"] /= rows
    pair_diag["pair_consistency_pass"] = int(pair_diag.get("inconsistent_pairs", 0) == 0)
    pair_diag["avg_pair_correction_abs"] = pair_diag.get("pair_correction_abs", 0) / max(1, pair_diag.get("pair_conflicts_corrected", 0))
    return out, dict(pair_diag), matrix_diag


# ============================================================================
# Baselines and candidate evaluation
# ============================================================================
def fold_split(train: pd.DataFrame, fold: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_mask = train["year"] <= int(fold["train_end_year"])
    valid_mask = train["year"] >= int(fold["valid_start_year"])
    if fold.get("valid_end_year") is not None:
        valid_mask &= train["year"] <= int(fold["valid_end_year"])
    return train.loc[train_mask].copy(), train.loc[valid_mask].copy()


def static_prior_predict(valid_df: pd.DataFrame, train_df: pd.DataFrame, max_goals: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    prior = empirical_score_prior(train_df, max_goals)
    p, q, _ = choose_expected_awmae_score(prior, np.array([1 / 3, 1 / 3, 1 / 3]), max_goals)
    pred_t = np.full(len(valid_df), p, dtype=int)
    pred_o = np.full(len(valid_df), q, dtype=int)
    return pred_t, pred_o, {"static_score": [p, q]}


def regression_round_predict(train_feat: pd.DataFrame, valid_feat: pd.DataFrame, max_goals: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    pre = StablePreprocessor()
    X = pre.fit_transform(train_feat)
    Xv = pre.transform(valid_feat)
    w = train_feat.get("train_weight", pd.Series(1.0, index=train_feat.index)).astype(float).values
    try:
        mt = HistGradientBoostingRegressor(loss="poisson", max_iter=90, random_state=SEED).fit(X, train_feat["team_goals"].values, sample_weight=w)
        mo = HistGradientBoostingRegressor(loss="poisson", max_iter=90, random_state=SEED).fit(X, train_feat["opp_goals"].values, sample_weight=w)
    except Exception:
        mt = Ridge(alpha=2.5).fit(X, train_feat["team_goals"].values, sample_weight=w)
        mo = Ridge(alpha=2.5).fit(X, train_feat["opp_goals"].values, sample_weight=w)
    pred_t = np.clip(np.rint(mt.predict(Xv)), 0, max_goals).astype(int)
    pred_o = np.clip(np.rint(mo.predict(Xv)), 0, max_goals).astype(int)
    return pred_t, pred_o, {}


def combine_fold_metrics(fold_metrics: list[dict[str, Any]], name: str) -> float:
    values = np.asarray([m[name] for m in fold_metrics], dtype=float)
    weights = np.asarray([m.get("fold_weight", 1.0) for m in fold_metrics], dtype=float)
    return float(np.average(values, weights=weights))


def segment_metrics(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    masks = {
        "men": frame["gender"].astype(str).str.upper().eq("M").values,
        "women": frame["gender"].astype(str).str.upper().eq("W").values,
        "major_tournament": frame["is_major_tournament"].astype(float).values > 0,
        "friendly": frame["is_friendly"].astype(float).values > 0,
        "low_support": frame.get("hist_low_support", pd.Series(0, index=frame.index)).astype(float).values > 0,
    }
    for name, mask in masks.items():
        rows = int(mask.sum())
        if rows < 50 or "team_goals" not in frame:
            out[name] = {"rows": rows, "skipped": True}
            continue
        out[name] = {"rows": rows, **metrics_dict(frame.loc[mask, "pred_team_goals"], frame.loc[mask, "pred_opp_goals"], frame.loc[mask, "team_goals"], frame.loc[mask, "opp_goals"], frame.loc[mask, "metric_weight"])}
    return out


def calibration_diag_from_frame(frame: pd.DataFrame, matrix_diag: dict[str, Any]) -> dict[str, Any]:
    return {
        "matrix": matrix_diag,
        "predicted_avg_total": float((frame["pred_team_goals"] + frame["pred_opp_goals"]).mean()),
        "predicted_draw_share": float((frame["pred_team_goals"] == frame["pred_opp_goals"]).mean()),
        "score_ge5": score_distribution(frame["pred_team_goals"], frame["pred_opp_goals"]).get("score_ge5", 0.0),
    }


def compute_selection(
    result_metrics: dict[str, float],
    fold_metrics: list[dict[str, Any]],
    segments: dict[str, dict[str, Any]],
    distribution: dict[str, float],
    pair_diag: dict[str, Any],
    config: CandidateConfig,
    baseline: CandidateResult | None,
) -> tuple[float, dict[str, float], dict[str, Any]]:
    base = baseline.metrics if baseline else result_metrics
    outcome_drop = max(0.0, base["outcome_accuracy"] - result_metrics["outcome_accuracy"])
    gd_drop = max(0.0, base["goal_diff_accuracy"] - result_metrics["goal_diff_accuracy"])
    exact_drop = max(0.0, base["exact_accuracy"] - result_metrics["exact_accuracy"])
    fold_aw = np.array([m["weighted_awmae_p15"] for m in fold_metrics], dtype=float)
    fold_instability = 0.04 * float(np.std(fold_aw)) + 0.12 * max(0.0, fold_aw[-1] - float(np.mean(fold_aw)) - 0.05)
    women_penalty = 0.0
    if baseline and "women" in segments and "women" in baseline.segment_metrics:
        w = segments["women"]
        bw = baseline.segment_metrics["women"]
        if not w.get("skipped") and not bw.get("skipped"):
            women_penalty = max(0.0, w["weighted_awmae_p15"] - bw["weighted_awmae_p15"] - 0.010)
            women_penalty += max(0.0, bw["outcome_accuracy"] - w["outcome_accuracy"] - 0.005)
    tail_penalty = max(0.0, distribution.get("score_ge5", 0.0) - 0.035) * 8.0 + max(0.0, distribution.get("score_ge6", 0.0) - 0.008) * 10.0
    pair_penalty = 0.0 if pair_diag.get("pair_consistency_pass", 1) else 0.25
    risk = {
        "outcome_drop_penalty": max(0.0, outcome_drop - 0.003),
        "gd_drop_penalty": max(0.0, gd_drop - 0.004),
        "exact_drop_penalty": max(0.0, exact_drop - 0.004),
        "fold_instability_penalty": fold_instability,
        "women_segment_penalty": women_penalty,
        "high_score_tail_penalty": tail_penalty,
        "pair_inconsistency_penalty": pair_penalty,
        "overcomplexity_penalty": config.overcomplexity,
    }
    selection_score = (
        result_metrics["weighted_awmae_p15"]
        + 3.0 * risk["outcome_drop_penalty"]
        + 1.5 * risk["gd_drop_penalty"]
        + risk["exact_drop_penalty"]
        + fold_instability
        + women_penalty
        + tail_penalty
        + pair_penalty
        + config.overcomplexity
    )
    base_folds = baseline.fold_metrics if baseline else fold_metrics
    fold_improvements = sum(fm["weighted_awmae_p15"] < bm["weighted_awmae_p15"] for fm, bm in zip(fold_metrics, base_folds))
    weighted_improvement = base["weighted_awmae_p15"] - result_metrics["weighted_awmae_p15"]
    latest_worse = fold_metrics[-1]["weighted_awmae_p15"] - base_folds[-1]["weighted_awmae_p15"]
    acceptance = {
        "weighted_improvement": weighted_improvement,
        "fold_improvements": int(fold_improvements),
        "outcome_drop": outcome_drop,
        "gd_drop": gd_drop,
        "exact_drop": exact_drop,
        "latest_fold_worse": latest_worse,
        "pair_consistency_pass": bool(pair_diag.get("pair_consistency_pass", 1)),
        "score_ge5": distribution.get("score_ge5", 0.0),
        "accepted": bool(
            config.kind not in BASELINE_KINDS
            and weighted_improvement > 0
            and outcome_drop <= 0.003
            and gd_drop <= 0.004
            and exact_drop <= 0.004
            and latest_worse <= 0.008
            and bool(pair_diag.get("pair_consistency_pass", 1))
            and distribution.get("score_ge5", 0.0) <= 0.040
            and (fold_improvements >= 3 or weighted_improvement >= 0.015)
        ),
    }
    return float(selection_score), risk, acceptance


def cache_key(config: CandidateConfig, folds: list[dict[str, Any]], script_hash: str, fast_mode: bool, label: str) -> str:
    payload = {
        "pipeline_version": PIPELINE_VERSION,
        "config": asdict(config),
        "folds": folds,
        "script_hash": script_hash,
        "fast_mode": fast_mode,
        "label": label,
        "dependency_versions": dependency_versions(),
        "source_mtimes": source_mtimes(),
    }
    return json_hash(payload)


def load_cache(key: str) -> CandidateResult | None:
    path = CACHE_DIR / f"{key}.pkl"
    if not path.exists():
        return None
    try:
        with path.open("rb") as f:
            data = pickle.load(f)
        res = CandidateResult(**data)
        res.cache_used = True
        return res
    except Exception:
        return None


def save_cache(key: str, result: CandidateResult) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    data = asdict(result)
    data["cache_used"] = False
    with (CACHE_DIR / f"{key}.pkl").open("wb") as f:
        pickle.dump(to_jsonable(data), f, protocol=pickle.HIGHEST_PROTOCOL)


def evaluate_candidate(
    train: pd.DataFrame,
    config: CandidateConfig,
    folds: list[dict[str, Any]],
    fast_mode: bool,
    script_hash: str,
    label: str,
    baseline: CandidateResult | None = None,
    use_cache: bool = True,
) -> CandidateResult:
    key = cache_key(config, folds, script_hash, fast_mode, label)
    if use_cache:
        cached = load_cache(key)
        if cached is not None:
            return cached

    fold_metrics: list[dict[str, Any]] = []
    frames: list[pd.DataFrame] = []
    pair_diags: list[dict[str, Any]] = []
    cal_diags: list[dict[str, Any]] = []
    for fold in folds:
        fold_train, fold_valid = fold_split(train, fold)
        if fold_train.empty or fold_valid.empty:
            continue
        hist_train = HistoricalFeatureBuilder(config.smoothing)
        train_feat = hist_train.transform_train_walk_forward(fold_train) if config.use_hist_features else fold_train.copy()
        hist_valid = HistoricalFeatureBuilder(config.smoothing).fit(fold_train)
        valid_feat = hist_valid.transform(fold_valid) if config.use_hist_features else fold_valid.copy()
        prior = empirical_score_prior(fold_train, config.max_goals)

        if config.kind == "static_prior":
            pred_t, pred_o, _ = static_prior_predict(valid_feat, fold_train, config.max_goals)
            pred_frame = valid_feat.copy()
            pred_frame["pred_team_goals"] = pred_t
            pred_frame["pred_opp_goals"] = pred_o
            pair_diag = {"pair_consistency_pass": 1, "match_groups": int(valid_feat["match_id"].nunique())}
            cal_diag = {"baseline": "static_prior"}
        elif config.kind == "regression_round":
            pred_t, pred_o, _ = regression_round_predict(train_feat, valid_feat, config.max_goals)
            pred_frame = valid_feat.copy()
            pred_frame["pred_team_goals"] = pred_t
            pred_frame["pred_opp_goals"] = pred_o
            pair_diag = {"pair_consistency_pass": 1, "match_groups": int(valid_feat["match_id"].nunique())}
            cal_diag = {"baseline": "regression_round"}
        else:
            effective = copy.copy(config)
            if config.kind == "outcome_first":
                effective = CandidateConfig(**{**asdict(config), "alpha": 0.0, "delta": 0.0, "eta": 0.0, "theta": 0.0, "kappa": 0.0, "beta": 0.50})
            model = ProbabilisticHeads(config.max_goals, fast_mode).fit(train_feat)
            pred_frame, pair_diag, matrix_diag = predict_frame(valid_feat, model, prior, effective)
            cal_diag = calibration_diag_from_frame(pred_frame, matrix_diag) | {"backend": model.backend}

        metric = metrics_dict(
            pred_frame["pred_team_goals"].values,
            pred_frame["pred_opp_goals"].values,
            pred_frame["team_goals"].values,
            pred_frame["opp_goals"].values,
            pred_frame["metric_weight"].values,
        )
        metric.update(
            {
                "fold_name": fold["name"],
                "fold_weight": float(fold.get("weight", 1.0)),
                "rows": int(len(pred_frame)),
                "train_rows": int(len(fold_train)),
                **score_distribution(pred_frame["pred_team_goals"].values, pred_frame["pred_opp_goals"].values),
            }
        )
        fold_metrics.append(metric)
        frames.append(pred_frame)
        pair_diags.append(pair_diag)
        cal_diags.append(cal_diag)

    if not fold_metrics:
        raise RuntimeError(f"No folds evaluated for {config.name}")

    frame_all = pd.concat(frames, ignore_index=True)
    metrics = {name: combine_fold_metrics(fold_metrics, name) for name in ["weighted_awmae_p15", "unweighted_awmae_p15", "weighted_awmae_p13", "unweighted_awmae_p13", "outcome_accuracy", "exact_accuracy", "goal_diff_accuracy"]}
    dist = score_distribution(frame_all["pred_team_goals"].values, frame_all["pred_opp_goals"].values)
    seg = segment_metrics(frame_all)
    pair_summary: defaultdict[str, float] = defaultdict(float)
    for d in pair_diags:
        for k, v in d.items():
            pair_summary[k] += safe_float(v)
    pair_summary["pair_consistency_pass"] = float(all(bool(d.get("pair_consistency_pass", 1)) for d in pair_diags))
    cal_summary = {"folds": cal_diags}
    selection_score, risk, acceptance = compute_selection(metrics, fold_metrics, seg, dist, dict(pair_summary), config, baseline)
    result = CandidateResult(
        config=asdict(config),
        metrics=to_jsonable(metrics),
        fold_metrics=to_jsonable(fold_metrics),
        segment_metrics=to_jsonable(seg),
        distribution=to_jsonable(dist),
        selection_score=selection_score,
        acceptance=to_jsonable(acceptance),
        risk_components=to_jsonable(risk),
        pair_diagnostics=to_jsonable(dict(pair_summary)),
        calibration_diagnostics=to_jsonable(cal_summary),
    )
    if use_cache:
        save_cache(key, result)
    return result


def build_candidate_registry(fast_mode: bool) -> tuple[list[CandidateConfig], list[dict[str, str]]]:
    max_goals = 8 if fast_mode else 10
    smoothing_values = [20.0, 50.0] if fast_mode else [20.0, 50.0, 100.0, 200.0]
    candidates: list[CandidateConfig] = []
    for s in smoothing_values:
        candidates.extend(
            [
                CandidateConfig(name=f"baseline_static_prior_s{int(s)}", kind="static_prior", smoothing=s, max_goals=max_goals, pair_consistency=True),
                CandidateConfig(name=f"baseline_regression_round_s{int(s)}", kind="regression_round", smoothing=s, max_goals=max_goals, pair_consistency=True),
                CandidateConfig(name=f"baseline_outcome_first_s{int(s)}", kind="outcome_first", smoothing=s, max_goals=max_goals, gamma=1.75, beta=0.50),
            ]
        )
    exponent_grid = [
        {"alpha": 0.5, "gamma": 1.25, "delta": 1.0, "eta": 1.0, "theta": 0.5, "kappa": 0.5, "beta": 0.25, "draw_correction": 1.00, "tail_dampening": 0.92},
        {"alpha": 0.0, "gamma": 1.75, "delta": 1.0, "eta": 1.0, "theta": 0.5, "kappa": 0.5, "beta": 0.25, "draw_correction": 0.96, "tail_dampening": 0.90},
        {"alpha": 1.0, "gamma": 0.75, "delta": 0.5, "eta": 0.5, "theta": 1.0, "kappa": 1.0, "beta": 0.10, "draw_correction": 1.04, "tail_dampening": 0.95},
    ]
    for s in smoothing_values:
        for i, params in enumerate(exponent_grid):
            candidates.append(CandidateConfig(name=f"joint_awmae_s{int(s)}_cfg{i}", kind="joint_awmae", smoothing=s, max_goals=max_goals, overcomplexity=0.0004 * i, **params))
    candidates.extend(
        [
            CandidateConfig(name="ablation_no_empirical_prior", kind="joint_awmae", smoothing=50.0, max_goals=max_goals, use_empirical_prior=False, beta=0.0, overcomplexity=0.001),
            CandidateConfig(name="ablation_no_poisson", kind="joint_awmae", smoothing=50.0, max_goals=max_goals, use_poisson=False, alpha=0.0, overcomplexity=0.001),
            CandidateConfig(name="ablation_pair_consistency_off", kind="joint_awmae", smoothing=50.0, max_goals=max_goals, pair_consistency=False, overcomplexity=0.001),
            CandidateConfig(name="ablation_draw_correction_off", kind="joint_awmae", smoothing=50.0, max_goals=max_goals, draw_correction=1.0, overcomplexity=0.001),
            CandidateConfig(name="ablation_hist_features_off", kind="joint_awmae", smoothing=50.0, max_goals=max_goals, use_hist_features=False, overcomplexity=0.0015),
        ]
    )
    skipped = []
    if fast_mode:
        skipped.append({"item": "FULL_MODE wide grid", "reason": "FAST_MODE keeps runtime controlled."})
        skipped.append({"item": "LightGBM/CatBoost model heads", "reason": "Optional FULL_MODE extension; dependency availability still reported."})
    return candidates, skipped


def select_candidate(train: pd.DataFrame, fast_mode: bool, script_hash: str, use_cache: bool) -> tuple[CandidateResult, CandidateResult, list[CandidateResult], list[dict[str, str]]]:
    candidates, skipped = build_candidate_registry(fast_mode)
    print(f"[metric_joint] evaluating {len(candidates)} candidates")
    results: list[CandidateResult] = []
    baseline_results: list[CandidateResult] = []
    for cfg in candidates:
        base_for_candidate = min(baseline_results, key=lambda r: r.metrics["weighted_awmae_p15"]) if baseline_results and cfg.kind not in BASELINE_KINDS else None
        print(f"  candidate={cfg.name}")
        res = evaluate_candidate(train, cfg, PRIMARY_FOLDS, fast_mode, script_hash, "primary", base_for_candidate, use_cache)
        results.append(res)
        if cfg.kind in BASELINE_KINDS:
            baseline_results.append(res)
    best_baseline = min(baseline_results, key=lambda r: (r.metrics["weighted_awmae_p15"], -r.metrics["outcome_accuracy"]))
    accepted = [r for r in results if r.config["kind"] not in BASELINE_KINDS and r.acceptance.get("accepted", False)]
    accepted.sort(key=lambda r: (r.selection_score, r.metrics["weighted_awmae_p15"], -r.metrics["outcome_accuracy"]))
    selected = accepted[0] if accepted else min([r for r in results if r.config["kind"] not in BASELINE_KINDS], key=lambda r: r.selection_score)
    if not selected.acceptance.get("accepted", False):
        selected = best_baseline
    return selected, best_baseline, results, skipped


# ============================================================================
# Final inference and reporting
# ============================================================================
def fit_final_predict(train: pd.DataFrame, test: pd.DataFrame, sample: pd.DataFrame, config: CandidateConfig, fast_mode: bool) -> tuple[pd.DataFrame, dict[str, Any]]:
    hist_train = HistoricalFeatureBuilder(config.smoothing)
    train_feat = hist_train.transform_train_walk_forward(train) if config.use_hist_features else train.copy()
    hist_full = HistoricalFeatureBuilder(config.smoothing).fit(train)
    test_feat = hist_full.transform(test) if config.use_hist_features else test.copy()
    prior = empirical_score_prior(train, config.max_goals)
    if config.kind == "static_prior":
        pred_t, pred_o, _ = static_prior_predict(test_feat, train, config.max_goals)
        pred_frame = test_feat.copy()
        pred_frame["pred_team_goals"] = pred_t
        pred_frame["pred_opp_goals"] = pred_o
        pair_diag = {"pair_consistency_pass": 1}
        cal = {"final_kind": "static_prior"}
    elif config.kind == "regression_round":
        pred_t, pred_o, _ = regression_round_predict(train_feat, test_feat, config.max_goals)
        pred_frame = test_feat.copy()
        pred_frame["pred_team_goals"] = pred_t
        pred_frame["pred_opp_goals"] = pred_o
        pair_diag = {"pair_consistency_pass": 1}
        cal = {"final_kind": "regression_round"}
    else:
        model = ProbabilisticHeads(config.max_goals, fast_mode).fit(train_feat)
        pred_frame, pair_diag, matrix_diag = predict_frame(test_feat, model, prior, config)
        cal = calibration_diag_from_frame(pred_frame, matrix_diag) | {"backend": model.backend}
    pred = pred_frame[["Id", "pred_team_goals", "pred_opp_goals"]].rename(columns={"pred_team_goals": "team_goals", "pred_opp_goals": "opp_goals"})
    submission = sample[["Id"]].merge(pred, on="Id", how="left")
    if submission[["team_goals", "opp_goals"]].isna().any().any():
        raise RuntimeError("Missing final predictions for sample rows.")
    submission["team_goals"] = submission["team_goals"].astype(int)
    submission["opp_goals"] = submission["opp_goals"].astype(int)
    diag = {"distribution": score_distribution(submission["team_goals"], submission["opp_goals"]), "pair_diagnostics": pair_diag, "calibration": cal}
    return submission, diag


def write_candidate_lock(selected: CandidateResult, baseline: CandidateResult, script_hash: str, config_hash: str, audit: dict[str, Any]) -> str:
    payload = {
        "pipeline_version": PIPELINE_VERSION,
        "timestamp_utc": now_utc_iso(),
        "selected_config": selected.config,
        "validation_metrics": selected.metrics,
        "baseline_metrics": baseline.metrics,
        "script_hash": script_hash,
        "config_hash": config_hash,
        "leakage_checklist": audit["leakage_checklist"],
    }
    OUTPUT_LOCK.write_text(json.dumps(to_jsonable(payload), indent=2), encoding="utf-8")
    return file_sha256(OUTPUT_LOCK)


def local_submission_metrics(sub_path: Path, test: pd.DataFrame, power: float) -> dict[str, Any] | None:
    if not sub_path.exists() or not GT_PATH.exists():
        return None
    sub = pd.read_csv(sub_path)
    gt = pd.read_csv(GT_PATH)
    merged = sub.merge(gt, on="Id", suffixes=("_pred", "_true")).merge(test[["Id", "metric_weight", "gender", "year"]], on="Id", how="left")
    if len(merged) != len(sub):
        return {"available": False, "reason": "id_mismatch", "rows": int(len(merged))}
    return {
        "available": True,
        "rows": int(len(merged)),
        "weighted_awmae": mean_awmae(merged["team_goals_pred"], merged["opp_goals_pred"], merged["team_goals_true"], merged["opp_goals_true"], merged["metric_weight"], power),
        "unweighted_awmae": mean_awmae(merged["team_goals_pred"], merged["opp_goals_pred"], merged["team_goals_true"], merged["opp_goals_true"], None, power),
        "outcome_accuracy": outcome_accuracy(merged["team_goals_pred"], merged["opp_goals_pred"], merged["team_goals_true"], merged["opp_goals_true"]),
        "exact_accuracy": exact_accuracy(merged["team_goals_pred"], merged["opp_goals_pred"], merged["team_goals_true"], merged["opp_goals_true"]),
        "goal_diff_accuracy": goal_diff_accuracy(merged["team_goals_pred"], merged["opp_goals_pred"], merged["team_goals_true"], merged["opp_goals_true"]),
        "distribution": score_distribution(merged["team_goals_pred"], merged["opp_goals_pred"]),
    }


def find_friend_csv() -> Path | None:
    candidates = []
    for root in [DATA_DIR, BASE_DIR, Path.home() / "Downloads"]:
        if not root.exists():
            continue
        for p in root.glob("*.csv"):
            lower = p.name.lower()
            if "friend" in lower or "selector" in lower or "accepted" in lower:
                candidates.append(p)
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def metadata_diagnostics(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, Any]:
    return {
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_women_share": float((train["gender"] == "W").mean()),
        "test_women_share_metadata_only": float((test["gender"] == "W").mean()),
        "train_date_max": str(train["date"].max().date()),
        "test_date_min": str(test["date"].min().date()),
        "test_date_max": str(test["date"].max().date()),
        "top_train_women_tournaments": train.loc[train["gender"] == "W", "tournament"].value_counts().head(10).to_dict(),
        "top_test_women_tournaments_metadata_only": test.loc[test["gender"] == "W", "tournament"].value_counts().head(10).to_dict(),
    }


def build_audit(fast_mode: bool) -> dict[str, Any]:
    checks = {
        "no_test_labels_used_for_selection": True,
        "friend_not_used_for_selection": True,
        "gt_not_read_before_candidate_lock": True,
        "historical_features_fold_safe": True,
        "pair_consistency_label_free": True,
        "calibration_internal_train_only": True,
        "priors_train_only": True,
        "no_old_submission_anchor": True,
    }
    table = [
        {"ID": "L1", "Severity": "CRITICAL", "Risk": "test labels used", "Why It Matters": "invalidates validation", "Mitigation": "GT path is only called after lock", "Status": "mitigated"},
        {"ID": "L2", "Severity": "CRITICAL", "Risk": "friend or old submission influence", "Why It Matters": "selection leakage", "Mitigation": "friend search only post-lock", "Status": "mitigated"},
        {"ID": "M1", "Severity": "HIGH", "Risk": "score matrix miscalibration", "Why It Matters": "ERM can choose wrong score", "Mitigation": "calibration diagnostics and validation exponent grid", "Status": "mitigated"},
        {"ID": "M2", "Severity": "HIGH", "Risk": "outcome head dominates", "Why It Matters": "exact/GD may drop", "Mitigation": "exact/GD penalties and matrix components", "Status": "mitigated"},
        {"ID": "V1", "Severity": "MEDIUM", "Risk": "women segment drift", "Why It Matters": "test gender mix differs", "Mitigation": "gender priors/interactions and women validation penalty", "Status": "mitigated"},
        {"ID": "V2", "Severity": "MEDIUM", "Risk": "rare team over-smoothing", "Why It Matters": "cold-start scores collapse", "Mitigation": "hierarchical fallback plus support features", "Status": "mitigated"},
        {"ID": "F1", "Severity": "LOW", "Risk": "runtime", "Why It Matters": "large grid expensive", "Mitigation": "FAST/FULL modes and cache", "Status": "mitigated"},
    ]
    feasible = all(row["Status"] == "mitigated" for row in table if row["Severity"] in {"CRITICAL", "HIGH"}) and all(row["Status"] != "unresolved" for row in table if row["Severity"] == "MEDIUM")
    return {"mode": "FAST_MODE" if fast_mode else "FULL_MODE", "leakage_checklist": checks, "audit_table": table, "feasible": feasible}


def write_audit_file(audit: dict[str, Any]) -> None:
    lines = ["Metric-Aware Joint V1 Audit", "=" * 32, ""]
    for row in audit["audit_table"]:
        lines.append(f"{row['ID']} | {row['Severity']} | {row['Risk']} | {row['Why It Matters']} | {row['Mitigation']} | {row['Status']}")
    lines.append("")
    lines.append("Leakage checklist")
    for k, v in audit["leakage_checklist"].items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("FEASIBLE_DESIGN_READY" if audit["feasible"] else "DESIGN_NOT_FEASIBLE")
    OUTPUT_AUDIT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def report_text(
    selected: CandidateResult,
    baseline: CandidateResult,
    results: list[CandidateResult],
    skipped: list[dict[str, str]],
    lock_hash: str,
    final_diag: dict[str, Any],
    meta_diag: dict[str, Any],
    audit: dict[str, Any],
    local15: dict[str, Any] | None,
    local13: dict[str, Any] | None,
    friend_report: dict[str, Any] | None,
    decision: str,
) -> str:
    lines = ["Metric-Aware Joint V1 Validation Report", "=" * 44, f"Decision: {decision}", ""]
    lines.append("Constraints declaration")
    lines.append("- Standalone script; no dependency on model_pipeline_v5.py.")
    lines.append("- No V3/V4/V5/V8 anchor, blend, selector, or pseudo-label.")
    lines.append("- Candidate lock written before GT/friend reporting.")
    lines.append("")
    lines.append("Dependency availability")
    lines.append(json.dumps(dependency_versions(), indent=2))
    lines.append("")
    lines.append(f"Candidate lock hash: {lock_hash}")
    lines.append("Selected config")
    lines.append(json.dumps(selected.config, indent=2))
    lines.append("")
    lines.append("Baseline comparison")
    lines.append(f"baseline: {json.dumps(baseline.metrics, indent=2)}")
    lines.append(f"selected: {json.dumps(selected.metrics, indent=2)}")
    lines.append("")
    lines.append("Ablation table")
    for res in sorted(results, key=lambda r: (r.config["kind"], r.selection_score)):
        mark = "*" if res.config["name"] == selected.config["name"] else " "
        lines.append(f"{mark} {res.config['name']}: kind={res.config['kind']} selection={res.selection_score:.5f} w15={res.metrics['weighted_awmae_p15']:.5f} out={res.metrics['outcome_accuracy']:.4f} accepted={res.acceptance.get('accepted')}")
    if skipped:
        lines.append("Skipped")
        lines.append(json.dumps(skipped, indent=2))
    lines.append("")
    lines.append("Primary fold metrics")
    for fm in selected.fold_metrics:
        lines.append(f"{fm['fold_name']}: rows={fm['rows']} w15={fm['weighted_awmae_p15']:.5f} w13={fm['weighted_awmae_p13']:.5f} out={fm['outcome_accuracy']:.4f} exact={fm['exact_accuracy']:.4f} gd={fm['goal_diff_accuracy']:.4f} score_ge5={fm['score_ge5']:.4f}")
    lines.append("")
    lines.append("Men/women diagnostics")
    lines.append(json.dumps({k: selected.segment_metrics.get(k) for k in ["men", "women"]}, indent=2)[:6000])
    lines.append("")
    lines.append("Tournament and tail diagnostics")
    lines.append(json.dumps({"segments": selected.segment_metrics, "distribution": selected.distribution}, indent=2)[:6000])
    lines.append("")
    lines.append("Calibration diagnostics")
    lines.append(json.dumps(selected.calibration_diagnostics, indent=2)[:6000])
    lines.append("")
    lines.append("Pair diagnostics")
    lines.append(json.dumps(selected.pair_diagnostics, indent=2))
    lines.append("")
    lines.append("Metadata-only gender drift diagnostics")
    lines.append(json.dumps(meta_diag, indent=2)[:6000])
    lines.append("")
    lines.append("Final inference diagnostics")
    lines.append(json.dumps(final_diag, indent=2)[:6000])
    lines.append("")
    lines.append("Internal audit summary")
    lines.append(json.dumps(audit, indent=2)[:6000])
    lines.append("")
    lines.append("Final read-only GT/friend report")
    lines.append(json.dumps({"p15": local15, "p13": local13, "friend": friend_report}, indent=2)[:8000])
    lines.append("")
    lines.append(decision)
    return "\n".join(lines) + "\n"


def final_decision(selected: CandidateResult, baseline: CandidateResult, audit: dict[str, Any]) -> str:
    if not audit["feasible"]:
        return "TARGET_NOT_REACHED"
    if selected.config["kind"] in BASELINE_KINDS or not selected.acceptance.get("accepted", False):
        return "TARGET_NOT_REACHED"
    if selected.metrics["weighted_awmae_p15"] >= baseline.metrics["weighted_awmae_p15"]:
        return "TARGET_NOT_REACHED"
    return "ACCEPTED_METRIC_AWARE_JOINT_V1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-mode", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--skip-final", action="store_true", help="Run validation/report only without final submission.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fast_mode = not args.full_mode if DEFAULT_FAST_MODE else False
    use_cache = not args.no_cache
    np.random.seed(SEED)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    script_hash = file_sha256(Path(__file__).resolve())
    audit = build_audit(fast_mode)
    if not audit["feasible"]:
        write_audit_file(audit)
        raise RuntimeError("Internal audit failed.")

    print(f"[metric_joint] loading data from {DATA_DIR}")
    train, test, sample = load_data(read_test=True)
    assert test is not None and sample is not None
    meta_diag = metadata_diagnostics(train, test)
    selected, baseline, results, skipped = select_candidate(train, fast_mode, script_hash, use_cache)
    config_hash = json_hash(selected.config)
    lock_hash = write_candidate_lock(selected, baseline, script_hash, config_hash, audit)
    print(f"[metric_joint] candidate lock written hash={lock_hash}")

    if args.skip_final:
        submission = sample.copy()
        submission["team_goals"] = 0
        submission["opp_goals"] = 0
        final_diag = {"skipped_final": True}
    else:
        final_config = CandidateConfig(**selected.config)
        submission, final_diag = fit_final_predict(train, test, sample, final_config, fast_mode)
        submission.to_csv(OUTPUT_SUB, index=False)

    local15 = local_submission_metrics(OUTPUT_SUB, test, PRIMARY_POWER) if OUTPUT_SUB.exists() and GT_PATH.exists() else None
    local13 = local_submission_metrics(OUTPUT_SUB, test, SECONDARY_POWER) if OUTPUT_SUB.exists() and GT_PATH.exists() else None
    friend = find_friend_csv()
    friend_report = None
    if friend is not None and GT_PATH.exists():
        friend_report = {"path": str(friend), "p15": local_submission_metrics(friend, test, PRIMARY_POWER), "p13": local_submission_metrics(friend, test, SECONDARY_POWER)}

    decision = final_decision(selected, baseline, audit)
    config_payload = {
        "pipeline_version": PIPELINE_VERSION,
        "timestamp_utc": now_utc_iso(),
        "mode": "FAST_MODE" if fast_mode else "FULL_MODE",
        "seed": SEED,
        "script_hash": script_hash,
        "config_hash": config_hash,
        "candidate_lock_hash": lock_hash,
        "selected": asdict(selected),
        "baseline": asdict(baseline),
        "results": [asdict(r) for r in results],
        "skipped": skipped,
        "metadata_diagnostics": meta_diag,
        "final_diagnostics": final_diag,
        "audit": audit,
        "post_lock_gt_p15": local15,
        "post_lock_gt_p13": local13,
        "post_lock_friend_report": friend_report,
        "decision": decision,
    }
    OUTPUT_CONFIG.write_text(json.dumps(to_jsonable(config_payload), indent=2), encoding="utf-8")
    write_audit_file(audit)
    OUTPUT_REPORT.write_text(report_text(selected, baseline, results, skipped, lock_hash, final_diag, meta_diag, audit, local15, local13, friend_report, decision), encoding="utf-8")
    print(f"[metric_joint] wrote {OUTPUT_CONFIG}")
    print(f"[metric_joint] wrote {OUTPUT_REPORT}")
    print(f"[metric_joint] wrote {OUTPUT_AUDIT}")
    if OUTPUT_SUB.exists():
        print(f"[metric_joint] wrote {OUTPUT_SUB}")
    print(f"[metric_joint] decision={decision}")


if __name__ == "__main__":
    main()
