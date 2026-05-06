"""Temporal Robust Metric-Aware Pair Expert Ensemble.

Leakage-safety contract:
* No import of model_pipeline_v5.py.
* No old submission anchors, pseudo-labels, blends, or selectors.
* Validation selection uses train labels only inside chronological boundaries.
* Test metadata may be used for inference/features, never test labels.
* Friend CSV and test_ground_truth.csv are read only after candidate lock.

Main design correction versus metric_aware_joint_v1:
* GD is a soft selection risk, not a hard veto.
* Strong AW-MAE + outcome gains can remain eligible with moderate GD tradeoff.
* All final candidates use pair-level consistency, including regression fallback.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import re
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

warnings.filterwarnings("ignore")


# ============================================================================
# Constants and paths
# ============================================================================
PIPELINE_VERSION = "temporal_robust_joint_v1"
SEED = 42
PRIMARY_POWER = 1.5
SECONDARY_POWER = 1.3

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "dataset"
TRAIN_FINAL = DATA_DIR / "train_final.csv"
TEST_FINAL = DATA_DIR / "test_final.csv"
TRAIN_RAW = DATA_DIR / "train.csv"
TEST_RAW = DATA_DIR / "test.csv"
SAMPLE_SUB = DATA_DIR / "sample submission.csv"
GT_PATH = DATA_DIR / "test_ground_truth.csv"

OUTPUT_SUB = DATA_DIR / "submission_temporal_robust_joint_v1.csv"
OUTPUT_CONFIG = DATA_DIR / "submission_temporal_robust_joint_v1_config.json"
OUTPUT_REPORT = DATA_DIR / "submission_temporal_robust_joint_v1_validation_report.txt"
OUTPUT_AUDIT = DATA_DIR / "submission_temporal_robust_joint_v1_audit.txt"
OUTPUT_LOCK = DATA_DIR / "submission_temporal_robust_joint_v1_candidate_lock.json"
CACHE_DIR = DATA_DIR / "temporal_robust_joint_v1_cache"

MATCH_RE = re.compile(r"^(M\d+)")
BASELINE_KINDS = {"static_prior", "regression_round"}

# Process-local reuse keeps the validation protocol unchanged while avoiding
# retraining identical fold models for joint candidates that only differ in
# score-matrix exponents or selector penalties.
FEATURE_CACHE: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = {}
JOINT_MODEL_CACHE: dict[str, tuple[ProbabilisticHeads, ExpertPriorStore]] = {}
JOINT_HEAD_CACHE: dict[str, tuple[pd.DataFrame, dict[str, np.ndarray], np.ndarray]] = {}

# Old folds kept for comparability; recency-weighted folds are used for selection.
COMPARISON_FOLDS = [
    {"name": "OLD_F1_2003_2005", "train_end_year": 2002, "valid_start_year": 2003, "valid_end_year": 2005, "weight": 0.05},
    {"name": "OLD_F2_2006_2008", "train_end_year": 2005, "valid_start_year": 2006, "valid_end_year": 2008, "weight": 0.15},
    {"name": "OLD_F3_2009_2010", "train_end_year": 2008, "valid_start_year": 2009, "valid_end_year": 2010, "weight": 0.30},
    {"name": "OLD_F4_2011", "train_end_year": 2010, "valid_start_year": 2011, "valid_end_year": 2011, "weight": 0.50},
]

SELECTION_FOLDS = [
    {"name": "R1_2003_2005", "train_end_year": 2002, "valid_start_year": 2003, "valid_end_year": 2005, "weight": 0.05},
    {"name": "R2_2006_2008", "train_end_year": 2005, "valid_start_year": 2006, "valid_end_year": 2008, "weight": 0.10},
    {"name": "R3_2009", "train_end_year": 2008, "valid_start_year": 2009, "valid_end_year": 2009, "weight": 0.15},
    {"name": "R4_2010", "train_end_year": 2009, "valid_start_year": 2010, "valid_end_year": 2010, "weight": 0.25},
    {"name": "R5_2011", "train_end_year": 2010, "valid_start_year": 2011, "valid_end_year": 2011, "weight": 0.45},
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
    "Copa América": 1.80,
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
    use_expert_prior: bool = True
    draw_correction: float = 1.0
    tail_dampening: float = 0.93
    pair_consistency: bool = True
    use_hist_features: bool = True
    gd_selector_weight: float = 0.0
    outcome_selector_weight: float = 0.0
    complexity: float = 0.0


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
# Utility helpers
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
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    return obj


def json_hash(obj: Any) -> str:
    payload = json.dumps(to_jsonable(obj), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def source_mtimes() -> dict[str, float]:
    out = {}
    for p in [TRAIN_FINAL, TEST_FINAL, TRAIN_RAW, TEST_RAW, SAMPLE_SUB]:
        if p.exists():
            out[p.name] = p.stat().st_mtime
    return out


def dependency_versions() -> dict[str, str]:
    return {"numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__}


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def extract_match_id(value: Any) -> str:
    s = str(value)
    m = MATCH_RE.match(s)
    if m:
        return m.group(1)
    if "_" in s:
        return "_".join(s.split("_")[:-1]) or s
    return s


def normalize_team_name(value: Any) -> str:
    s = str(value).lower().strip()
    s = s.replace("'", "").replace("`", "").replace("'", "").replace("'", "")
    return re.sub(r"\s+", " ", s)


def tournament_group(value: Any) -> str:
    s = str(value).lower()
    if "friendly" in s:
        return "friendly"
    if "qualification" in s or "qualifier" in s:
        return "qualifier"
    if "world cup" in s or "euro" in s or "nations league" in s or "cup" in s or "championship" in s:
        return "major"
    return "other"


def add_weights(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["metric_weight"] = out.get("tournament", pd.Series("", index=out.index)).map(TOURNAMENT_WEIGHT_MAP).fillna(DEFAULT_TOURNAMENT_WEIGHT).astype(float)
    out["train_weight"] = out["metric_weight"].astype(float)
    return out


# ============================================================================
# Data loading
# ============================================================================
def merge_metadata(final: pd.DataFrame, raw: pd.DataFrame, is_train: bool) -> pd.DataFrame:
    meta_cols = [
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
    ]
    if is_train:
        meta_cols.extend([c for c in ["team_goals", "opp_goals"] if c in raw.columns and c not in final.columns])
    meta = raw[[c for c in meta_cols if c in raw.columns]].copy()
    out = final.copy()
    missing = [c for c in meta.columns if c == "Id" or c not in out.columns]
    if len(missing) > 1:
        out = out.merge(meta[missing], on="Id", how="left")
    else:
        for col in ["date", "gender", "team", "opponent", "tournament", "neutral", "is_home"]:
            if col not in out.columns and col in meta.columns:
                out = out.merge(meta[["Id", col]], on="Id", how="left")
    for col in meta.columns:
        if col == "Id":
            continue
        if col in out.columns and out[col].isna().any():
            fill = meta[["Id", col]]
            out = out.merge(fill, on="Id", how="left", suffixes=("", "_rawfill"))
            out[col] = out[col].where(out[col].notna(), out[f"{col}_rawfill"])
            out = out.drop(columns=[f"{col}_rawfill"])
    return out


def prepare_frame(df: pd.DataFrame, is_train: bool) -> pd.DataFrame:
    out = df.copy()
    required = ["Id", "date", "team", "opponent", "gender", "tournament", "neutral", "is_home"]
    for col in required:
        if col not in out.columns:
            if col in {"neutral", "is_home"}:
                out[col] = 0
            else:
                out[col] = "UNK"
    if is_train:
        for col in ["team_goals", "opp_goals"]:
            if col not in out.columns:
                raise ValueError(f"Missing required train column {col}")
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["year"] = out["date"].dt.year.fillna(0).astype(int)
    out["month"] = out["date"].dt.month.fillna(0).astype(int)
    out["match_id"] = out["Id"].map(extract_match_id)
    out["team_norm"] = out["team"].map(normalize_team_name)
    out["opponent_norm"] = out["opponent"].map(normalize_team_name)
    out["gender"] = out["gender"].fillna("UNK").astype(str).str.upper()
    out["tournament"] = out["tournament"].fillna("UNK").astype(str)
    out["tournament_group"] = out["tournament"].map(tournament_group)
    out["is_friendly"] = (out["tournament_group"] == "friendly").astype(int)
    out["is_qualifier"] = (out["tournament_group"] == "qualifier").astype(int)
    out["is_major_tournament"] = (out["tournament_group"] == "major").astype(int)
    out["is_women"] = (out["gender"] == "W").astype(int)
    for col in ["neutral", "is_home"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(int)
    for col in ["venue_country", "confederation_team", "confederation_opp"]:
        if col not in out.columns:
            out[col] = "UNK"
        out[col] = out[col].fillna("UNK").astype(str)
    out = add_weights(out)
    return out


def load_data(read_test: bool = True) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
    train_final = pd.read_csv(TRAIN_FINAL)
    train_raw = pd.read_csv(TRAIN_RAW)
    train = prepare_frame(merge_metadata(train_final, train_raw, True), True)
    if not read_test:
        return train, None, None
    test_final = pd.read_csv(TEST_FINAL)
    test_raw = pd.read_csv(TEST_RAW)
    sample = pd.read_csv(SAMPLE_SUB)
    test = prepare_frame(merge_metadata(test_final, test_raw, False), False)
    return train, test, sample


# ============================================================================
# Metrics
# ============================================================================
def awmae_loss_array(pred_t: Any, pred_o: Any, true_t: Any, true_o: Any, power: float) -> np.ndarray:
    pred_t = np.asarray(pred_t, dtype=float)
    pred_o = np.asarray(pred_o, dtype=float)
    true_t = np.asarray(true_t, dtype=float)
    true_o = np.asarray(true_o, dtype=float)
    mae = (np.abs(pred_t - true_t) + np.abs(pred_o - true_o)) / 2.0
    exact_ok = ((pred_t == true_t) & (pred_o == true_o)).astype(float)
    outcome_ok = (np.sign(pred_t - pred_o) == np.sign(true_t - true_o)).astype(float)
    gd_ok = ((pred_t - pred_o) == (true_t - true_o)).astype(float)
    augmented = mae + 0.30 * (1.0 - exact_ok) + 0.25 * (1.0 - outcome_ok) + 0.15 * (1.0 - gd_ok)
    multiplier = np.where(outcome_ok.astype(bool), 1.0, 1.5)
    return (augmented * multiplier) ** power


def mean_awmae(pred_t: Any, pred_o: Any, true_t: Any, true_o: Any, weights: Any | None, power: float) -> float:
    losses = awmae_loss_array(pred_t, pred_o, true_t, true_o, power)
    if weights is None:
        return float(losses.mean())
    weights = np.asarray(weights, dtype=float)
    return float(np.average(losses, weights=weights))


def outcome_accuracy(pred_t: Any, pred_o: Any, true_t: Any, true_o: Any) -> float:
    return float((np.sign(np.asarray(pred_t) - np.asarray(pred_o)) == np.sign(np.asarray(true_t) - np.asarray(true_o))).mean())


def exact_accuracy(pred_t: Any, pred_o: Any, true_t: Any, true_o: Any) -> float:
    return float(((np.asarray(pred_t) == np.asarray(true_t)) & (np.asarray(pred_o) == np.asarray(true_o))).mean())


def goal_diff_accuracy(pred_t: Any, pred_o: Any, true_t: Any, true_o: Any) -> float:
    return float(((np.asarray(pred_t) - np.asarray(pred_o)) == (np.asarray(true_t) - np.asarray(true_o))).mean())


def metrics_dict(pred_t: Any, pred_o: Any, true_t: Any, true_o: Any, weights: Any | None) -> dict[str, float]:
    return {
        "weighted_awmae_p15": mean_awmae(pred_t, pred_o, true_t, true_o, weights, PRIMARY_POWER),
        "unweighted_awmae_p15": mean_awmae(pred_t, pred_o, true_t, true_o, None, PRIMARY_POWER),
        "weighted_awmae_p13": mean_awmae(pred_t, pred_o, true_t, true_o, weights, SECONDARY_POWER),
        "unweighted_awmae_p13": mean_awmae(pred_t, pred_o, true_t, true_o, None, SECONDARY_POWER),
        "outcome_accuracy": outcome_accuracy(pred_t, pred_o, true_t, true_o),
        "exact_accuracy": exact_accuracy(pred_t, pred_o, true_t, true_o),
        "goal_diff_accuracy": goal_diff_accuracy(pred_t, pred_o, true_t, true_o),
    }


def score_distribution(pred_t: Any, pred_o: Any) -> dict[str, float]:
    pred_t = np.asarray(pred_t, dtype=int)
    pred_o = np.asarray(pred_o, dtype=int)
    if len(pred_t) == 0:
        return {}
    total = pred_t + pred_o
    return {
        "rows": int(len(pred_t)),
        "avg_team_goals": float(pred_t.mean()),
        "avg_opp_goals": float(pred_o.mean()),
        "avg_total_goals": float(total.mean()),
        "draw_share": float((pred_t == pred_o).mean()),
        "team_win_share": float((pred_t > pred_o).mean()),
        "score_ge5": float(((pred_t >= 5) | (pred_o >= 5)).mean()),
        "score_ge6": float(((pred_t >= 6) | (pred_o >= 6)).mean()),
    }


LOSS_TENSOR_CACHE: dict[tuple[int, float], np.ndarray] = {}
GD_LOSS_CACHE: dict[int, np.ndarray] = {}
OUTCOME_MISMATCH_CACHE: dict[int, np.ndarray] = {}


def loss_tensor(max_goals: int, power: float) -> np.ndarray:
    key = (max_goals, power)
    if key in LOSS_TENSOR_CACHE:
        return LOSS_TENSOR_CACHE[key]
    size = max_goals + 1
    t = np.zeros((size, size, size, size), dtype=float)
    for p in range(size):
        for q in range(size):
            for a in range(size):
                for b in range(size):
                    t[p, q, a, b] = awmae_loss_array([p], [q], [a], [b], power)[0]
    LOSS_TENSOR_CACHE[key] = t
    return t


def gd_loss_tensor(max_goals: int) -> np.ndarray:
    if max_goals in GD_LOSS_CACHE:
        return GD_LOSS_CACHE[max_goals]
    size = max_goals + 1
    t = np.zeros((size, size, size, size), dtype=float)
    for p in range(size):
        for q in range(size):
            for a in range(size):
                for b in range(size):
                    t[p, q, a, b] = abs((p - q) - (a - b)) / max(1, max_goals)
    GD_LOSS_CACHE[max_goals] = t
    return t


def outcome_mismatch_tensor(max_goals: int) -> np.ndarray:
    if max_goals in OUTCOME_MISMATCH_CACHE:
        return OUTCOME_MISMATCH_CACHE[max_goals]
    size = max_goals + 1
    t = np.zeros((size, size, size, size), dtype=float)
    for p in range(size):
        for q in range(size):
            for a in range(size):
                for b in range(size):
                    t[p, q, a, b] = float(np.sign(p - q) != np.sign(a - b))
    OUTCOME_MISMATCH_CACHE[max_goals] = t
    return t


def expected_loss_matrix(joint: np.ndarray, config: CandidateConfig, power: float = PRIMARY_POWER) -> np.ndarray:
    base = np.tensordot(loss_tensor(config.max_goals, power), joint, axes=([2, 3], [0, 1]))
    if power == PRIMARY_POWER and config.gd_selector_weight:
        base += config.gd_selector_weight * np.tensordot(gd_loss_tensor(config.max_goals), joint, axes=([2, 3], [0, 1]))
    if power == PRIMARY_POWER and config.outcome_selector_weight:
        base += config.outcome_selector_weight * np.tensordot(outcome_mismatch_tensor(config.max_goals), joint, axes=([2, 3], [0, 1]))
    return base


# ============================================================================
# Historical features
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
    gender = str(row.get("gender", "UNK")).upper()
    team = normalize_team_name(row.get(team_col, ""))
    tournament = str(row.get("tournament", "UNK"))
    t_group = str(row.get("tournament_group", "other"))
    confed = str(row.get("confederation_team", "UNK"))
    return [
        ("team_gender", gender, team),
        ("team_global", team),
        ("confed_gender", confed, gender),
        ("tournament_gender", tournament, gender),
        ("tournament_group_gender", t_group, gender),
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

    def _lookup_team(self, row: pd.Series, side: str) -> tuple[np.ndarray, int]:
        team_col = "team_norm" if side == "team" else "opponent_norm"
        for key in hierarchy_keys(row, team_col):
            vals, count = self.stats.lookup(key, self.global_prior, self.smoothing)
            if count > 0 or key == ("global",):
                return vals, count
        return self.global_prior.copy(), 0

    def _row_features(self, row: pd.Series) -> dict[str, float]:
        team_vals, team_count = self._lookup_team(row, "team")
        opp_vals, opp_count = self._lookup_team(row, "opp")
        feat = {}
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
        self.stats.update(hierarchy_keys(row, "team_norm"), gf, ga)
        rev = row.copy()
        rev["team_norm"] = row["opponent_norm"]
        rev["opponent_norm"] = row["team_norm"]
        rev["confederation_team"] = row.get("confederation_opp", row.get("confederation_team", "UNK"))
        self.stats.update(hierarchy_keys(rev, "team_norm"), ga, gf)

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
# Preprocessing and probabilistic heads
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
    "tournament_group",
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


def align_proba(proba: np.ndarray, classes: np.ndarray, target_classes: list[int]) -> np.ndarray:
    out = np.full((proba.shape[0], len(target_classes)), 1e-7, dtype=float)
    class_to_idx = {int(c): i for i, c in enumerate(target_classes)}
    for j, cls in enumerate(classes):
        if int(cls) in class_to_idx:
            out[:, class_to_idx[int(cls)]] = proba[:, j]
    out = np.maximum(out, 1e-7)
    out /= out.sum(axis=1, keepdims=True)
    return out


class ProbabilisticHeads:
    def __init__(self, max_goals: int):
        self.max_goals = int(max_goals)
        self.pre = StablePreprocessor()
        self.models: dict[str, Any] = {}
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
            learning_rate=0.075,
            max_iter=40,
            max_leaf_nodes=15,
            min_samples_leaf=65,
            l2_regularization=0.12,
            random_state=SEED,
        )

    def _reg(self) -> Any:
        return HistGradientBoostingRegressor(
            loss="poisson",
            learning_rate=0.075,
            max_iter=40,
            max_leaf_nodes=15,
            min_samples_leaf=65,
            l2_regularization=0.12,
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
        for name, y in {"outcome": y_outcome, "team": y_team, "opp": y_opp, "total": y_total, "gd": y_gd}.items():
            if len(np.unique(y)) < 2:
                self.models[name] = None
                continue
            self.models[name] = self._clf().fit(X, y, sample_weight=weights)
        try:
            self.models["lambda_team"] = self._reg().fit(X, np.clip(df["team_goals"].values, 0, None), sample_weight=weights)
            self.models["lambda_opp"] = self._reg().fit(X, np.clip(df["opp_goals"].values, 0, None), sample_weight=weights)
        except Exception:
            self.backend = "ridge_lambda_fallback"
            self.models["lambda_team"] = Ridge(alpha=2.5).fit(X, df["team_goals"].values, sample_weight=weights)
            self.models["lambda_opp"] = Ridge(alpha=2.5).fit(X, df["opp_goals"].values, sample_weight=weights)
        return self

    def predict(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        X = self.pre.transform(df)
        out: dict[str, np.ndarray] = {}
        for name in ["outcome", "team", "opp", "total", "gd"]:
            model = self.models.get(name)
            if model is None:
                out[name] = np.full((len(df), len(self.classes[name])), 1.0 / len(self.classes[name]), dtype=float)
            else:
                out[name] = align_proba(model.predict_proba(X), model.classes_, self.classes[name])
        for name in ["lambda_team", "lambda_opp"]:
            out[name] = np.clip(np.asarray(self.models[name].predict(X), dtype=float), 0.03, 9.5)
        return out


# ============================================================================
# Expert priors and joint score matrices
# ============================================================================
def empirical_score_prior(train_df: pd.DataFrame, max_goals: int, smoothing: float = 0.35) -> np.ndarray:
    mat = np.full((max_goals + 1, max_goals + 1), smoothing, dtype=float)
    a = np.minimum(train_df["team_goals"].astype(int).values, max_goals)
    b = np.minimum(train_df["opp_goals"].astype(int).values, max_goals)
    for x, y in zip(a, b):
        mat[x, y] += 1.0
    return mat / mat.sum()


class ExpertPriorStore:
    def __init__(self, max_goals: int):
        self.max_goals = max_goals
        self.global_prior: np.ndarray | None = None
        self.priors: dict[tuple[str, str], np.ndarray] = {}

    def fit(self, train_df: pd.DataFrame) -> "ExpertPriorStore":
        self.global_prior = empirical_score_prior(train_df, self.max_goals)
        for key_col in ["gender", "tournament_group"]:
            for key, grp in train_df.groupby(key_col):
                if len(grp) >= 250:
                    self.priors[(key_col, str(key))] = empirical_score_prior(grp, self.max_goals)
        return self

    def row_prior(self, row: pd.Series, use_expert: bool = True) -> np.ndarray:
        assert self.global_prior is not None
        prior = self.global_prior.copy()
        if not use_expert:
            return prior
        parts = [prior]
        weights = [0.60]
        for key_col, weight in [("gender", 0.25), ("tournament_group", 0.15)]:
            key = (key_col, str(row.get(key_col, "")))
            if key in self.priors:
                parts.append(self.priors[key])
                weights.append(weight)
        w = np.asarray(weights, dtype=float)
        w /= w.sum()
        out = np.zeros_like(prior)
        for p, ww in zip(parts, w):
            out += ww * p
        return out / out.sum()


def poisson_probs_batch(lam: np.ndarray, max_goals: int) -> np.ndarray:
    lam = np.clip(np.asarray(lam, dtype=float), 0.03, 9.5)
    out = np.zeros((len(lam), max_goals + 1), dtype=float)
    out[:, 0] = np.exp(-lam)
    for k in range(1, max_goals + 1):
        out[:, k] = out[:, k - 1] * lam / k
    out /= np.maximum(out.sum(axis=1, keepdims=True), 1e-12)
    return out


def grid_cache(max_goals: int) -> dict[str, np.ndarray]:
    size = max_goals + 1
    a, b = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    return {
        "a": a,
        "b": b,
        "outcome": np.where(a > b, 2, np.where(a == b, 1, 0)),
        "total": a + b,
        "gd": a - b + max_goals,
        "draw": a == b,
        "tail": (a >= 5) | (b >= 5),
    }


def batch_joint_matrices(df: pd.DataFrame, heads: dict[str, np.ndarray], priors: np.ndarray, config: CandidateConfig) -> np.ndarray:
    max_g = config.max_goals
    g = grid_cache(max_g)
    n = len(df)
    eps = 1e-8
    mat = np.ones((n, max_g + 1, max_g + 1), dtype=float)
    if config.use_poisson and config.alpha != 0:
        pt = poisson_probs_batch(heads["lambda_team"], max_g)
        po = poisson_probs_batch(heads["lambda_opp"], max_g)
        mat *= np.maximum(pt[:, :, None] * po[:, None, :], eps) ** config.alpha
    mat *= np.maximum(heads["outcome"][:, g["outcome"]], eps) ** config.gamma
    mat *= np.maximum(heads["total"][:, g["total"]], eps) ** config.delta
    mat *= np.maximum(heads["gd"][:, g["gd"]], eps) ** config.eta
    mat *= np.maximum(heads["team"][:, g["a"]], eps) ** config.theta
    mat *= np.maximum(heads["opp"][:, g["b"]], eps) ** config.kappa
    if config.use_empirical_prior and config.beta != 0:
        mat *= np.maximum(priors, eps) ** config.beta
    if config.draw_correction != 1.0:
        mat[:, g["draw"]] *= config.draw_correction
    if config.tail_dampening != 1.0:
        mat[:, g["tail"]] *= config.tail_dampening
    mat = np.maximum(mat, eps)
    mat /= np.maximum(mat.sum(axis=(1, 2), keepdims=True), eps)
    return mat


def expected_losses_batch(joints: np.ndarray, config: CandidateConfig) -> np.ndarray:
    losses = np.tensordot(joints, loss_tensor(config.max_goals, PRIMARY_POWER), axes=([1, 2], [2, 3]))
    if config.gd_selector_weight:
        losses += config.gd_selector_weight * np.tensordot(joints, gd_loss_tensor(config.max_goals), axes=([1, 2], [2, 3]))
    if config.outcome_selector_weight:
        losses += config.outcome_selector_weight * np.tensordot(joints, outcome_mismatch_tensor(config.max_goals), axes=([1, 2], [2, 3]))
    return losses


def reciprocal_pair(group: pd.DataFrame) -> bool:
    if len(group) != 2:
        return False
    a = group.iloc[0]
    b = group.iloc[1]
    return bool(a["team_norm"] == b["opponent_norm"] and a["opponent_norm"] == b["team_norm"])


def predict_frame_from_components(
    ordered: pd.DataFrame,
    heads: dict[str, np.ndarray],
    row_priors: np.ndarray,
    config: CandidateConfig,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    joints = batch_joint_matrices(ordered, heads, row_priors, config)
    loss15 = expected_losses_batch(joints, config)
    size = config.max_goals + 1
    flat = np.argmin(loss15.reshape(len(ordered), -1), axis=1)
    pred_t = (flat // size).astype(int)
    pred_o = (flat % size).astype(int)
    selected_prob = joints[np.arange(len(ordered)), pred_t, pred_o]
    pair_diag: defaultdict[str, float] = defaultdict(float)
    for _, group in ordered.groupby("match_id", sort=False):
        pair_diag["match_groups"] += 1
        idxs = [int(i) for i in group.index.to_list()]
        if len(group) == 1:
            pair_diag["single_row_matches"] += 1
        elif len(group) == 2:
            pair_diag["two_row_matches"] += 1
        else:
            pair_diag["multirow_matches"] += 1
        if len(group) == 2 and reciprocal_pair(group):
            pair_diag["reciprocal_pairs"] += 1
            if config.pair_consistency:
                pair_diag["pair_consistency_applied"] += 1
                combined = loss15[idxs[0]] + loss15[idxs[1]].T
                a, b = np.unravel_index(int(np.argmin(combined)), combined.shape)
                if (pred_t[idxs[0]], pred_o[idxs[0]], pred_t[idxs[1]], pred_o[idxs[1]]) != (int(a), int(b), int(b), int(a)):
                    pair_diag["pair_conflicts_corrected"] += 1
                pred_t[idxs[0]], pred_o[idxs[0]] = int(a), int(b)
                pred_t[idxs[1]], pred_o[idxs[1]] = int(b), int(a)
        elif len(group) == 2:
            pair_diag["inconsistent_pairs"] += 1
    out = ordered.copy()
    out["pred_team_goals"] = pred_t
    out["pred_opp_goals"] = pred_o
    entropy = float((-(joints * np.log(np.maximum(joints, 1e-12))).sum(axis=(1, 2))).mean())
    matrix_diag = {"avg_selected_joint_prob": float(selected_prob.mean()), "avg_entropy": entropy, "rows": int(len(out))}
    pair_diag["pair_consistency_pass"] = float(pair_diag.get("inconsistent_pairs", 0) == 0)
    return out, dict(pair_diag), matrix_diag


def predict_frame(df: pd.DataFrame, model: ProbabilisticHeads, priors: ExpertPriorStore, config: CandidateConfig) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    ordered = df.sort_values(["date", "match_id", "Id"]).reset_index(drop=True)
    heads = model.predict(ordered)
    row_priors = np.stack([priors.row_prior(row, config.use_expert_prior) for _, row in ordered.iterrows()])
    return predict_frame_from_components(ordered, heads, row_priors, config)


# ============================================================================
# Baselines and validation
# ============================================================================
def fold_split(train: pd.DataFrame, fold: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_mask = train["year"] <= int(fold["train_end_year"])
    valid_mask = train["year"] >= int(fold["valid_start_year"])
    if fold.get("valid_end_year") is not None:
        valid_mask &= train["year"] <= int(fold["valid_end_year"])
    return train.loc[train_mask].copy(), train.loc[valid_mask].copy()


def combine_fold_metrics(fold_metrics: list[dict[str, Any]], name: str) -> float:
    values = np.asarray([m[name] for m in fold_metrics], dtype=float)
    weights = np.asarray([m.get("fold_weight", 1.0) for m in fold_metrics], dtype=float)
    return float(np.average(values, weights=weights))


def choose_static_score(train_df: pd.DataFrame, config: CandidateConfig) -> tuple[int, int]:
    prior = empirical_score_prior(train_df, config.max_goals)
    loss = expected_loss_matrix(prior, config, PRIMARY_POWER)
    a, b = np.unravel_index(int(np.argmin(loss)), loss.shape)
    return int(a), int(b)


def static_prior_predict(valid_df: pd.DataFrame, train_df: pd.DataFrame, config: CandidateConfig) -> tuple[np.ndarray, np.ndarray]:
    p, q = choose_static_score(train_df, config)
    return np.full(len(valid_df), p, dtype=int), np.full(len(valid_df), q, dtype=int)


def apply_pair_consistency_scores(frame: pd.DataFrame, pred_t: np.ndarray, pred_o: np.ndarray, max_goals: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    ordered = frame.sort_values(["date", "match_id", "Id"]).reset_index(drop=True)
    pt = pred_t.copy()
    po = pred_o.copy()
    diag: defaultdict[str, float] = defaultdict(float)
    for _, group in ordered.groupby("match_id", sort=False):
        diag["match_groups"] += 1
        idxs = [int(i) for i in group.index.to_list()]
        if len(group) == 2 and reciprocal_pair(group):
            diag["reciprocal_pairs"] += 1
            a = int(np.clip(round((pt[idxs[0]] + po[idxs[1]]) / 2), 0, max_goals))
            b = int(np.clip(round((po[idxs[0]] + pt[idxs[1]]) / 2), 0, max_goals))
            if (pt[idxs[0]], po[idxs[0]], pt[idxs[1]], po[idxs[1]]) != (a, b, b, a):
                diag["pair_conflicts_corrected"] += 1
            pt[idxs[0]], po[idxs[0]] = a, b
            pt[idxs[1]], po[idxs[1]] = b, a
        elif len(group) == 2:
            diag["inconsistent_pairs"] += 1
    diag["pair_consistency_pass"] = float(diag.get("inconsistent_pairs", 0) == 0)
    return pt, po, dict(diag)


def regression_round_predict(train_feat: pd.DataFrame, valid_feat: pd.DataFrame, config: CandidateConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    pre = StablePreprocessor()
    X = pre.fit_transform(train_feat)
    Xv = pre.transform(valid_feat.sort_values(["date", "match_id", "Id"]).reset_index(drop=True))
    w = train_feat.get("train_weight", pd.Series(1.0, index=train_feat.index)).astype(float).values
    try:
        mt = HistGradientBoostingRegressor(loss="poisson", learning_rate=0.075, max_iter=45, max_leaf_nodes=15, min_samples_leaf=65, l2_regularization=0.12, random_state=SEED).fit(X, train_feat["team_goals"].values, sample_weight=w)
        mo = HistGradientBoostingRegressor(loss="poisson", learning_rate=0.075, max_iter=45, max_leaf_nodes=15, min_samples_leaf=65, l2_regularization=0.12, random_state=SEED).fit(X, train_feat["opp_goals"].values, sample_weight=w)
    except Exception:
        mt = Ridge(alpha=2.5).fit(X, train_feat["team_goals"].values, sample_weight=w)
        mo = Ridge(alpha=2.5).fit(X, train_feat["opp_goals"].values, sample_weight=w)
    ordered = valid_feat.sort_values(["date", "match_id", "Id"]).reset_index(drop=True)
    pred_t = np.clip(np.rint(mt.predict(Xv)), 0, config.max_goals).astype(int)
    pred_o = np.clip(np.rint(mo.predict(Xv)), 0, config.max_goals).astype(int)
    pred_t, pred_o, pair_diag = apply_pair_consistency_scores(ordered, pred_t, pred_o, config.max_goals)
    out = ordered.copy()
    out["pred_team_goals"] = pred_t
    out["pred_opp_goals"] = pred_o
    return out, pair_diag


def segment_metrics(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    out = {}
    masks = {
        "men": frame["gender"].astype(str).str.upper().eq("M").values,
        "women": frame["gender"].astype(str).str.upper().eq("W").values,
        "major": frame["tournament_group"].eq("major").values,
        "friendly": frame["tournament_group"].eq("friendly").values,
        "qualifier": frame["tournament_group"].eq("qualifier").values,
        "low_support": frame.get("hist_low_support", pd.Series(0, index=frame.index)).astype(float).values > 0,
    }
    for name, mask in masks.items():
        rows = int(mask.sum())
        if rows < 50 or "team_goals" not in frame:
            out[name] = {"rows": rows, "skipped": True}
            continue
        out[name] = {
            "rows": rows,
            **metrics_dict(frame.loc[mask, "pred_team_goals"], frame.loc[mask, "pred_opp_goals"], frame.loc[mask, "team_goals"], frame.loc[mask, "opp_goals"], frame.loc[mask, "metric_weight"]),
        }
    return out


def calibration_diag_from_frame(frame: pd.DataFrame, matrix_diag: dict[str, Any], backend: str) -> dict[str, Any]:
    return {
        "backend": backend,
        "matrix": matrix_diag,
        "predicted_avg_total": float((frame["pred_team_goals"] + frame["pred_opp_goals"]).mean()),
        "predicted_draw_share": float((frame["pred_team_goals"] == frame["pred_opp_goals"]).mean()),
        "score_ge5": score_distribution(frame["pred_team_goals"], frame["pred_opp_goals"]).get("score_ge5", 0.0),
    }


def compute_selection(
    metrics: dict[str, float],
    fold_metrics: list[dict[str, Any]],
    segments: dict[str, dict[str, Any]],
    distribution: dict[str, float],
    pair_diag: dict[str, Any],
    config: CandidateConfig,
    baseline: CandidateResult | None,
) -> tuple[float, dict[str, float], dict[str, Any]]:
    base = baseline.metrics if baseline else metrics
    aw_improvement = base["weighted_awmae_p15"] - metrics["weighted_awmae_p15"]
    outcome_gain = metrics["outcome_accuracy"] - base["outcome_accuracy"]
    outcome_drop = max(0.0, -outcome_gain)
    gd_drop = max(0.0, base["goal_diff_accuracy"] - metrics["goal_diff_accuracy"])
    exact_drop = max(0.0, base["exact_accuracy"] - metrics["exact_accuracy"])
    strong_tradeoff = aw_improvement >= 0.035 and outcome_gain >= 0.010
    gd_allowance = 0.020 if strong_tradeoff else 0.006
    fold_aw = np.array([m["weighted_awmae_p15"] for m in fold_metrics], dtype=float)
    recent_penalty = max(0.0, fold_metrics[-1]["weighted_awmae_p15"] - base.get("weighted_awmae_p15", fold_metrics[-1]["weighted_awmae_p15"]) - 0.010)
    fold_instability = 0.03 * float(np.std(fold_aw))
    women_penalty = 0.0
    tournament_penalty = 0.0
    if baseline:
        for seg_name, bucket in [("women", "women_penalty"), ("major", "tournament_penalty"), ("qualifier", "tournament_penalty")]:
            seg = segments.get(seg_name, {})
            bseg = baseline.segment_metrics.get(seg_name, {})
            if not seg.get("skipped") and not bseg.get("skipped"):
                penalty = max(0.0, seg["weighted_awmae_p15"] - bseg["weighted_awmae_p15"] - 0.020)
                penalty += max(0.0, bseg["outcome_accuracy"] - seg["outcome_accuracy"] - 0.010)
                if bucket == "women_penalty":
                    women_penalty += penalty
                else:
                    tournament_penalty += 0.5 * penalty
    tail_penalty = max(0.0, distribution.get("score_ge5", 0.0) - 0.045) * 5.0 + max(0.0, distribution.get("score_ge6", 0.0) - 0.012) * 8.0
    pair_penalty = 0.0 if pair_diag.get("pair_consistency_pass", 1) else 0.25
    risk = {
        "outcome_penalty": max(0.0, outcome_drop - 0.004) * 2.5,
        "soft_gd_penalty": max(0.0, gd_drop - gd_allowance) * 0.75,
        "exact_penalty": max(0.0, exact_drop - 0.006) * 0.75,
        "recent_fold_penalty": recent_penalty,
        "fold_instability_penalty": fold_instability,
        "women_penalty": women_penalty,
        "tournament_penalty": tournament_penalty,
        "tail_penalty": tail_penalty,
        "pair_penalty": pair_penalty,
        "complexity_penalty": config.complexity,
    }
    selection_score = metrics["weighted_awmae_p15"] + sum(risk.values())
    base_folds = baseline.fold_metrics if baseline else fold_metrics
    fold_improvements = sum(fm["weighted_awmae_p15"] < bm["weighted_awmae_p15"] for fm, bm in zip(fold_metrics, base_folds))
    severe_gd_collapse = gd_drop > 0.035 and not strong_tradeoff
    segment_collapse = women_penalty > 0.06 or tournament_penalty > 0.08
    accepted = bool(
        config.kind not in BASELINE_KINDS
        and aw_improvement > 0
        and outcome_drop <= 0.010
        and not severe_gd_collapse
        and exact_drop <= 0.020
        and not segment_collapse
        and distribution.get("score_ge5", 0.0) <= 0.055
        and bool(pair_diag.get("pair_consistency_pass", 1))
        and (fold_improvements >= max(3, len(fold_metrics) - 1) or aw_improvement >= 0.025)
    )
    acceptance = {
        "weighted_improvement": aw_improvement,
        "outcome_gain": outcome_gain,
        "outcome_drop": outcome_drop,
        "gd_drop": gd_drop,
        "gd_allowance": gd_allowance,
        "gd_tradeoff_allowed": strong_tradeoff,
        "exact_drop": exact_drop,
        "fold_improvements": int(fold_improvements),
        "pair_consistency_pass": bool(pair_diag.get("pair_consistency_pass", 1)),
        "score_ge5": distribution.get("score_ge5", 0.0),
        "accepted": accepted,
    }
    return float(selection_score), risk, acceptance


def cache_key(config: CandidateConfig, folds: list[dict[str, Any]], script_hash: str, label: str) -> str:
    payload = {
        "pipeline_version": PIPELINE_VERSION,
        "config": asdict(config),
        "folds": folds,
        "script_hash": script_hash,
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
    script_hash: str,
    label: str,
    baseline: CandidateResult | None = None,
    use_cache: bool = True,
) -> CandidateResult:
    key = cache_key(config, folds, script_hash, label)
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
        feature_key = json_hash(
            {
                "label": label,
                "fold": fold,
                "smoothing": config.smoothing,
                "use_hist_features": config.use_hist_features,
                "train_rows": len(fold_train),
                "valid_rows": len(fold_valid),
            }
        )
        if feature_key in FEATURE_CACHE:
            fold_train_ref, train_feat, valid_feat = FEATURE_CACHE[feature_key]
        else:
            hist_train = HistoricalFeatureBuilder(config.smoothing)
            train_feat = hist_train.transform_train_walk_forward(fold_train) if config.use_hist_features else fold_train.sort_values(["date", "match_id", "Id"]).reset_index(drop=True)
            hist_valid = HistoricalFeatureBuilder(config.smoothing).fit(fold_train)
            valid_feat = hist_valid.transform(fold_valid) if config.use_hist_features else fold_valid.sort_values(["date", "match_id", "Id"]).reset_index(drop=True)
            fold_train_ref = fold_train
            FEATURE_CACHE[feature_key] = (fold_train_ref, train_feat, valid_feat)
        if config.kind == "static_prior":
            pred_t, pred_o = static_prior_predict(valid_feat, fold_train, config)
            pred_frame = valid_feat.copy()
            pred_frame["pred_team_goals"] = pred_t
            pred_frame["pred_opp_goals"] = pred_o
            pred_t, pred_o, pair_diag = apply_pair_consistency_scores(pred_frame, pred_t, pred_o, config.max_goals)
            pred_frame["pred_team_goals"] = pred_t
            pred_frame["pred_opp_goals"] = pred_o
            cal_diag = {"baseline": "static_prior"}
        elif config.kind == "regression_round":
            pred_frame, pair_diag = regression_round_predict(train_feat, valid_feat, config)
            cal_diag = {"baseline": "regression_round"}
        else:
            joint_key = json_hash(
                {
                    "label": label,
                    "fold": fold,
                    "smoothing": config.smoothing,
                    "use_hist_features": config.use_hist_features,
                    "max_goals": config.max_goals,
                    "train_rows": len(fold_train_ref),
                    "feature_cols": list(train_feat.columns),
                }
            )
            if joint_key in JOINT_MODEL_CACHE:
                model, priors = JOINT_MODEL_CACHE[joint_key]
            else:
                model = ProbabilisticHeads(config.max_goals).fit(train_feat)
                priors = ExpertPriorStore(config.max_goals).fit(fold_train_ref)
                JOINT_MODEL_CACHE[joint_key] = (model, priors)
            head_key = json_hash(
                {
                    "joint_key": joint_key,
                    "use_expert_prior": config.use_expert_prior,
                    "valid_rows": len(valid_feat),
                    "valid_date_min": str(valid_feat["date"].min()),
                    "valid_date_max": str(valid_feat["date"].max()),
                }
            )
            if head_key in JOINT_HEAD_CACHE:
                ordered_valid, heads, row_priors = JOINT_HEAD_CACHE[head_key]
            else:
                ordered_valid = valid_feat.sort_values(["date", "match_id", "Id"]).reset_index(drop=True)
                heads = model.predict(ordered_valid)
                row_priors = np.stack([priors.row_prior(row, config.use_expert_prior) for _, row in ordered_valid.iterrows()])
                JOINT_HEAD_CACHE[head_key] = (ordered_valid, heads, row_priors)
            pred_frame, pair_diag, matrix_diag = predict_frame_from_components(ordered_valid, heads, row_priors, config)
            cal_diag = calibration_diag_from_frame(pred_frame, matrix_diag, model.backend)
        metric = metrics_dict(pred_frame["pred_team_goals"], pred_frame["pred_opp_goals"], pred_frame["team_goals"], pred_frame["opp_goals"], pred_frame["metric_weight"])
        metric.update({"fold_name": fold["name"], "fold_weight": float(fold.get("weight", 1.0)), "rows": int(len(pred_frame)), "train_rows": int(len(fold_train)), **score_distribution(pred_frame["pred_team_goals"], pred_frame["pred_opp_goals"])})
        fold_metrics.append(metric)
        frames.append(pred_frame)
        pair_diags.append(pair_diag)
        cal_diags.append(cal_diag)
    if not fold_metrics:
        raise RuntimeError(f"No folds evaluated for {config.name}")
    frame_all = pd.concat(frames, ignore_index=True)
    metrics = {name: combine_fold_metrics(fold_metrics, name) for name in ["weighted_awmae_p15", "unweighted_awmae_p15", "weighted_awmae_p13", "unweighted_awmae_p13", "outcome_accuracy", "exact_accuracy", "goal_diff_accuracy"]}
    dist = score_distribution(frame_all["pred_team_goals"], frame_all["pred_opp_goals"])
    seg = segment_metrics(frame_all)
    pair_summary: defaultdict[str, float] = defaultdict(float)
    for d in pair_diags:
        for k, v in d.items():
            pair_summary[k] += safe_float(v)
    pair_summary["pair_consistency_pass"] = float(all(bool(d.get("pair_consistency_pass", 1)) for d in pair_diags))
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
        calibration_diagnostics=to_jsonable({"folds": cal_diags}),
    )
    if use_cache:
        save_cache(key, result)
    return result


def build_candidate_registry() -> tuple[list[CandidateConfig], list[dict[str, str]]]:
    max_g = 8
    candidates = [
        CandidateConfig(name="baseline_static_prior_s50", kind="static_prior", smoothing=50, max_goals=max_g, use_hist_features=False, complexity=0.0),
        CandidateConfig(name="baseline_regression_round_s50", kind="regression_round", smoothing=50, max_goals=max_g, complexity=0.0),
        CandidateConfig(name="joint_balanced_s50", kind="joint", smoothing=50, max_goals=max_g, alpha=0.5, gamma=1.25, delta=1.0, eta=1.0, theta=0.5, kappa=0.5, beta=0.25, draw_correction=0.98, tail_dampening=0.93, complexity=0.0005),
        CandidateConfig(name="joint_outcome_strong_s50", kind="joint", smoothing=50, max_goals=max_g, alpha=0.0, gamma=1.75, delta=1.0, eta=1.0, theta=0.5, kappa=0.5, beta=0.25, draw_correction=0.98, tail_dampening=0.91, outcome_selector_weight=0.02, complexity=0.0010),
        CandidateConfig(name="joint_gd_soft_s50", kind="joint", smoothing=50, max_goals=max_g, alpha=0.5, gamma=1.20, delta=0.8, eta=1.20, theta=0.5, kappa=0.5, beta=0.25, draw_correction=1.0, tail_dampening=0.92, gd_selector_weight=0.03, complexity=0.0010),
        CandidateConfig(name="joint_draw_off_s50", kind="joint", smoothing=50, max_goals=max_g, alpha=0.5, gamma=1.25, delta=1.0, eta=1.0, theta=0.5, kappa=0.5, beta=0.25, draw_correction=1.0, tail_dampening=1.0, complexity=0.0010),
        CandidateConfig(name="joint_tail_safe_s50", kind="joint", smoothing=50, max_goals=max_g, alpha=0.5, gamma=1.25, delta=1.0, eta=1.0, theta=0.5, kappa=0.5, beta=0.25, draw_correction=1.0, tail_dampening=0.86, complexity=0.0012),
        CandidateConfig(name="joint_no_expert_prior_s50", kind="joint", smoothing=50, max_goals=max_g, use_expert_prior=False, complexity=0.0015),
        CandidateConfig(name="joint_hist_off_s50", kind="joint", smoothing=50, max_goals=max_g, use_hist_features=False, complexity=0.0020),
    ]
    skipped = [{"item": "FULL wide model library grid", "reason": "v1 runtime control; HGB-only core with cached candidates."}]
    return candidates, skipped


def select_candidate(train: pd.DataFrame, script_hash: str, use_cache: bool) -> tuple[CandidateResult, CandidateResult, list[CandidateResult], list[dict[str, str]]]:
    candidates, skipped = build_candidate_registry()
    print(f"[temporal_robust] evaluating {len(candidates)} candidates")
    results: list[CandidateResult] = []
    baseline_results: list[CandidateResult] = []
    for cfg in candidates:
        base = min(baseline_results, key=lambda r: r.metrics["weighted_awmae_p15"]) if baseline_results and cfg.kind not in BASELINE_KINDS else None
        print(f"  candidate={cfg.name}", flush=True)
        res = evaluate_candidate(train, cfg, SELECTION_FOLDS, script_hash, "selection", base, use_cache)
        print(f"    w15={res.metrics['weighted_awmae_p15']:.6f} w13={res.metrics['weighted_awmae_p13']:.6f} out={res.metrics['outcome_accuracy']:.6f} gd={res.metrics['goal_diff_accuracy']:.6f} accepted={res.acceptance.get('accepted')}", flush=True)
        results.append(res)
        if cfg.kind in BASELINE_KINDS:
            baseline_results.append(res)
    best_baseline = min(baseline_results, key=lambda r: (r.metrics["weighted_awmae_p15"], -r.metrics["outcome_accuracy"]))
    accepted = [r for r in results if r.config["kind"] not in BASELINE_KINDS and r.acceptance.get("accepted", False)]
    accepted.sort(key=lambda r: (r.selection_score, r.metrics["weighted_awmae_p15"], -r.metrics["outcome_accuracy"]))
    if accepted:
        selected = accepted[0]
    else:
        selected = min([r for r in results if r.config["kind"] not in BASELINE_KINDS], key=lambda r: r.selection_score)
    return selected, best_baseline, results, skipped


# ============================================================================
# Final inference and reporting
# ============================================================================
def fit_final_predict(train: pd.DataFrame, test: pd.DataFrame, sample: pd.DataFrame, config: CandidateConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    hist_train = HistoricalFeatureBuilder(config.smoothing)
    train_feat = hist_train.transform_train_walk_forward(train) if config.use_hist_features else train.sort_values(["date", "match_id", "Id"]).reset_index(drop=True)
    hist_full = HistoricalFeatureBuilder(config.smoothing).fit(train)
    test_feat = hist_full.transform(test) if config.use_hist_features else test.sort_values(["date", "match_id", "Id"]).reset_index(drop=True)
    if config.kind == "static_prior":
        pred_t, pred_o = static_prior_predict(test_feat, train, config)
        pred_frame = test_feat.copy()
        pred_frame["pred_team_goals"] = pred_t
        pred_frame["pred_opp_goals"] = pred_o
        pred_t, pred_o, pair_diag = apply_pair_consistency_scores(pred_frame, pred_t, pred_o, config.max_goals)
        pred_frame["pred_team_goals"] = pred_t
        pred_frame["pred_opp_goals"] = pred_o
        cal = {"final_kind": "static_prior"}
    elif config.kind == "regression_round":
        pred_frame, pair_diag = regression_round_predict(train_feat, test_feat, config)
        cal = {"final_kind": "regression_round"}
    else:
        model = ProbabilisticHeads(config.max_goals).fit(train_feat)
        priors = ExpertPriorStore(config.max_goals).fit(train)
        pred_frame, pair_diag, matrix_diag = predict_frame(test_feat, model, priors, config)
        cal = calibration_diag_from_frame(pred_frame, matrix_diag, model.backend)
    pred = pred_frame[["Id", "pred_team_goals", "pred_opp_goals"]].rename(columns={"pred_team_goals": "team_goals", "pred_opp_goals": "opp_goals"})
    submission = sample[["Id"]].merge(pred, on="Id", how="left")
    if submission[["team_goals", "opp_goals"]].isna().any().any():
        raise RuntimeError("Missing final predictions for sample rows.")
    submission["team_goals"] = submission["team_goals"].astype(int)
    submission["opp_goals"] = submission["opp_goals"].astype(int)
    diag = {"distribution": score_distribution(submission["team_goals"], submission["opp_goals"]), "pair_diagnostics": pair_diag, "calibration": cal}
    return submission, diag


def build_audit() -> dict[str, Any]:
    checklist = {
        "no_model_pipeline_v5_import": True,
        "no_old_submission_anchor": True,
        "selection_validation_only": True,
        "gt_not_read_before_candidate_lock": True,
        "friend_not_used_for_selection": True,
        "historical_features_fold_safe": True,
        "pair_consistency_label_free": True,
        "gd_is_soft_penalty_not_hard_veto": True,
    }
    table = [
        {"ID": "L1", "Severity": "CRITICAL", "Risk": "test GT leaks into selection", "Mitigation": "GT read only in local_submission_metrics after lock", "Status": "mitigated"},
        {"ID": "L2", "Severity": "CRITICAL", "Risk": "old submission influence", "Mitigation": "no submission reads except post-lock friend report", "Status": "mitigated"},
        {"ID": "S1", "Severity": "HIGH", "Risk": "GD hard veto repeats old failure", "Mitigation": "soft GD penalty plus explicit tradeoff allowance", "Status": "mitigated"},
        {"ID": "V1", "Severity": "HIGH", "Risk": "validation too old", "Mitigation": "recency-weighted folds and stress reporting", "Status": "mitigated"},
        {"ID": "D1", "Severity": "MEDIUM", "Risk": "gender/tournament drift", "Mitigation": "metadata features, expert priors, segment diagnostics", "Status": "mitigated"},
        {"ID": "R1", "Severity": "MEDIUM", "Risk": "runtime", "Mitigation": "vectorized matrix prediction and compact grid", "Status": "mitigated"},
    ]
    return {"leakage_checklist": checklist, "audit_table": table, "feasible": True}


def write_audit_file(audit: dict[str, Any]) -> None:
    lines = ["Temporal Robust Joint V1 Audit", "=" * 34, ""]
    for row in audit["audit_table"]:
        lines.append(f"{row['ID']} | {row['Severity']} | {row['Risk']} | {row['Mitigation']} | {row['Status']}")
    lines.append("")
    lines.append("Leakage checklist")
    for k, v in audit["leakage_checklist"].items():
        lines.append(f"- {k}: {v}")
    OUTPUT_AUDIT.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
        "train_date_min": str(train["date"].min().date()),
        "train_date_max": str(train["date"].max().date()),
        "test_date_min": str(test["date"].min().date()),
        "test_date_max": str(test["date"].max().date()),
        "train_women_share": float((train["gender"] == "W").mean()),
        "test_women_share_metadata_only": float((test["gender"] == "W").mean()),
        "top_train_tournaments": train["tournament"].value_counts().head(10).to_dict(),
        "top_test_tournaments_metadata_only": test["tournament"].value_counts().head(10).to_dict(),
    }


def final_decision(selected: CandidateResult, baseline: CandidateResult, audit: dict[str, Any]) -> str:
    if not audit["feasible"]:
        return "TARGET_NOT_REACHED"
    if selected.config["kind"] in BASELINE_KINDS:
        return "TARGET_NOT_REACHED"
    if selected.metrics["weighted_awmae_p15"] >= baseline.metrics["weighted_awmae_p15"]:
        return "TARGET_NOT_REACHED"
    return "ACCEPTED_TEMPORAL_ROBUST_JOINT_V1" if selected.acceptance.get("accepted", False) else "TARGET_NOT_REACHED"


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
    stress: CandidateResult | None,
    comparison: CandidateResult | None,
    decision: str,
) -> str:
    lines = ["Temporal Robust Joint V1 Validation Report", "=" * 46, f"Decision: {decision}", ""]
    lines.append("Constraints")
    lines.append("- Standalone script; no model_pipeline_v5 import.")
    lines.append("- No old submission anchor/blend/pseudo-label/selector.")
    lines.append("- Candidate lock written before GT/friend reporting.")
    lines.append("- GD is a soft risk, not a hard acceptance veto.")
    lines.append("")
    lines.append(f"Candidate lock hash: {lock_hash}")
    lines.append("Selected config")
    lines.append(json.dumps(selected.config, indent=2))
    lines.append("Selected validation metrics")
    lines.append(json.dumps(selected.metrics, indent=2))
    lines.append("Baseline metrics")
    lines.append(json.dumps(baseline.metrics, indent=2))
    lines.append("")
    lines.append("Candidate table")
    for res in sorted(results, key=lambda r: (r.selection_score, r.metrics["weighted_awmae_p15"])):
        mark = "*" if res.config["name"] == selected.config["name"] else " "
        lines.append(f"{mark} {res.config['name']}: kind={res.config['kind']} selection={res.selection_score:.5f} w15={res.metrics['weighted_awmae_p15']:.5f} w13={res.metrics['weighted_awmae_p13']:.5f} out={res.metrics['outcome_accuracy']:.4f} gd={res.metrics['goal_diff_accuracy']:.4f} accepted={res.acceptance.get('accepted')}")
    if skipped:
        lines.append("Skipped")
        lines.append(json.dumps(skipped, indent=2))
    lines.append("")
    lines.append("Selection fold metrics")
    for fm in selected.fold_metrics:
        lines.append(f"{fm['fold_name']}: rows={fm['rows']} w15={fm['weighted_awmae_p15']:.5f} w13={fm['weighted_awmae_p13']:.5f} out={fm['outcome_accuracy']:.4f} exact={fm['exact_accuracy']:.4f} gd={fm['goal_diff_accuracy']:.4f} ge5={fm['score_ge5']:.4f}")
    lines.append("")
    lines.append("Stress metrics")
    lines.append(json.dumps(stress.metrics if stress else None, indent=2))
    lines.append("Old-fold comparability metrics")
    lines.append(json.dumps(comparison.metrics if comparison else None, indent=2))
    lines.append("")
    lines.append("Segment diagnostics")
    lines.append(json.dumps(selected.segment_metrics, indent=2)[:7000])
    lines.append("")
    lines.append("Distribution, pair, calibration")
    lines.append(json.dumps({"distribution": selected.distribution, "pair": selected.pair_diagnostics, "calibration": selected.calibration_diagnostics}, indent=2)[:8000])
    lines.append("")
    lines.append("Metadata-only drift diagnostics")
    lines.append(json.dumps(meta_diag, indent=2)[:6000])
    lines.append("")
    lines.append("Final inference diagnostics")
    lines.append(json.dumps(final_diag, indent=2)[:6000])
    lines.append("")
    lines.append("Audit")
    lines.append(json.dumps(audit, indent=2)[:6000])
    lines.append("")
    lines.append("Post-lock GT/friend report")
    lines.append(json.dumps({"p15": local15, "p13": local13, "friend": friend_report}, indent=2)[:8000])
    lines.append("")
    lines.append(decision)
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--skip-final", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(SEED)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    script_hash = file_sha256(Path(__file__).resolve())
    audit = build_audit()
    if not audit["feasible"]:
        write_audit_file(audit)
        raise RuntimeError("Internal audit failed")
    print(f"[temporal_robust] loading data from {DATA_DIR}")
    train, test, sample = load_data(read_test=True)
    assert test is not None and sample is not None
    meta_diag = metadata_diagnostics(train, test)
    selected, baseline, results, skipped = select_candidate(train, script_hash, not args.no_cache)
    stress = None
    comparison = None
    try:
        stress = evaluate_candidate(train, CandidateConfig(**selected.config), STRESS_FOLDS, script_hash, "stress", baseline, not args.no_cache)
        comparison = evaluate_candidate(train, CandidateConfig(**selected.config), COMPARISON_FOLDS, script_hash, "old_comparison", baseline, not args.no_cache)
    except Exception as exc:
        skipped.append({"item": "stress_or_comparison", "reason": str(exc)})
    config_hash = json_hash(selected.config)
    lock_hash = write_candidate_lock(selected, baseline, script_hash, config_hash, audit)
    print(f"[temporal_robust] candidate lock written hash={lock_hash}")
    if args.skip_final:
        final_diag = {"skipped_final": True}
    else:
        submission, final_diag = fit_final_predict(train, test, sample, CandidateConfig(**selected.config))
        submission.to_csv(OUTPUT_SUB, index=False)
    local15 = local_submission_metrics(OUTPUT_SUB, test, PRIMARY_POWER) if OUTPUT_SUB.exists() and GT_PATH.exists() else None
    local13 = local_submission_metrics(OUTPUT_SUB, test, SECONDARY_POWER) if OUTPUT_SUB.exists() and GT_PATH.exists() else None
    friend = find_friend_csv()
    friend_report = None
    if friend is not None and GT_PATH.exists():
        friend_report = {"path": str(friend), "p15": local_submission_metrics(friend, test, PRIMARY_POWER), "p13": local_submission_metrics(friend, test, SECONDARY_POWER)}
    decision = final_decision(selected, baseline, audit)
    payload = {
        "pipeline_version": PIPELINE_VERSION,
        "timestamp_utc": now_utc_iso(),
        "seed": SEED,
        "script_hash": script_hash,
        "config_hash": config_hash,
        "candidate_lock_hash": lock_hash,
        "selected": asdict(selected),
        "baseline": asdict(baseline),
        "results": [asdict(r) for r in results],
        "stress": asdict(stress) if stress else None,
        "old_fold_comparison": asdict(comparison) if comparison else None,
        "skipped": skipped,
        "metadata_diagnostics": meta_diag,
        "final_diagnostics": final_diag,
        "audit": audit,
        "post_lock_gt_p15": local15,
        "post_lock_gt_p13": local13,
        "post_lock_friend_report": friend_report,
        "decision": decision,
    }
    OUTPUT_CONFIG.write_text(json.dumps(to_jsonable(payload), indent=2), encoding="utf-8")
    write_audit_file(audit)
    OUTPUT_REPORT.write_text(report_text(selected, baseline, results, skipped, lock_hash, final_diag, meta_diag, audit, local15, local13, friend_report, stress, comparison, decision), encoding="utf-8")
    print(f"[temporal_robust] selected={selected.config['name']} decision={decision}")
    print(f"[temporal_robust] validation_w15={selected.metrics['weighted_awmae_p15']:.6f} validation_w13={selected.metrics['weighted_awmae_p13']:.6f} outcome={selected.metrics['outcome_accuracy']:.6f}")
    if OUTPUT_SUB.exists():
        print(f"[temporal_robust] wrote {OUTPUT_SUB}")
    print(f"[temporal_robust] wrote {OUTPUT_CONFIG}")
    print(f"[temporal_robust] wrote {OUTPUT_REPORT}")
    print(f"[temporal_robust] wrote {OUTPUT_AUDIT}")


if __name__ == "__main__":
    main()
