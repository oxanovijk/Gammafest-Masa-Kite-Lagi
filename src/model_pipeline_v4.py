"""
ML Pipeline V4 -- Gammafest Masa Kite Lagi
=========================================
Static one-shot ensemble experiment.

Design goals:
  1. Keep 2011 Elo/features frozen. No iterative prediction, no Elo decay.
  2. Add a time-validation blender instead of fixed 50/50 averaging.
  3. Use optional third brains when installed (LightGBM/CatBoost), while still
     runnable with the local fallback stack (XGBoost + sklearn Poisson HGB).

Output:
  dataset/submission_v4.csv
"""

from __future__ import annotations

import itertools
import os
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import poisson
from sklearn.ensemble import HistGradientBoostingRegressor

warnings.filterwarnings("ignore")

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover - depends on local machine
    lgb = None

try:
    from catboost import CatBoostRegressor
except ImportError:  # pragma: no cover - depends on local machine
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
OUTPUT_SUB = DATA_DIR / "submission_v4.csv"

MAX_GOALS = 10
NLS_POWER = 1.3
VALIDATION_FRACTION = 0.12
SEED = 42

N_ESTIMATORS_FULL = 800
N_ESTIMATORS_BLEND = 450

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

XGB_PARAMS = {
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
    "n_jobs": -1,
    "seed": SEED,
}


# ===========================================================================
# 1. DATA AND LOSS
# ===========================================================================
def load_data():
    print("[1] Loading data...")
    train = pd.read_csv(TRAIN_FINAL)
    test = pd.read_csv(TEST_FINAL)

    raw_train = pd.read_csv(TRAIN_RAW, usecols=["Id", "date", "tournament"])
    raw_test = pd.read_csv(TEST_RAW, usecols=["Id", "date", "tournament"])

    train = train.merge(raw_train, on="Id", how="left")
    test = test.merge(raw_test, on="Id", how="left")
    train["date"] = pd.to_datetime(train["date"])
    test["date"] = pd.to_datetime(test["date"])
    train["sample_weight"] = train["tournament"].map(TOURNAMENT_WEIGHT_MAP).fillna(
        DEFAULT_TOURNAMENT_WEIGHT
    )

    exclude = {
        "Id",
        "team_goals",
        "opp_goals",
        "date",
        "tournament",
        "sample_weight",
        "is_test",
    }
    feature_cols = [c for c in train.columns if c not in exclude]
    print(f"    Train: {train.shape} | Test: {test.shape} | Features: {len(feature_cols)}")
    return train, test, feature_cols


def awmae_single(pred_t, pred_o, true_t, true_o, nls_power=NLS_POWER):
    mae = (abs(pred_t - true_t) + abs(pred_o - true_o)) / 2.0
    exact = int(pred_t == true_t and pred_o == true_o)
    pred_out = np.sign(pred_t - pred_o)
    true_out = np.sign(true_t - true_o)
    outcome_ok = int(pred_out == true_out)
    gd_ok = int((pred_t - pred_o) == (true_t - true_o))
    aug = mae + 0.30 * (1 - exact) + 0.25 * (1 - outcome_ok) + 0.15 * (1 - gd_ok)
    mult = 1.0 if outcome_ok else 1.5
    return (aug * mult) ** nls_power


def build_loss_tensor(max_goals=MAX_GOALS):
    tensor = np.zeros((max_goals, max_goals, max_goals, max_goals), dtype=np.float32)
    for a in range(max_goals):
        for b in range(max_goals):
            for gt in range(max_goals):
                for go in range(max_goals):
                    tensor[a, b, gt, go] = awmae_single(a, b, gt, go)
    return tensor


def erm_predict_batch(lambdas_team, lambdas_opp, loss_tensor):
    n = len(lambdas_team)
    max_goals = loss_tensor.shape[0]
    k = np.arange(max_goals)
    lam_t = np.clip(lambdas_team, 1e-6, 15.0)
    lam_o = np.clip(lambdas_opp, 1e-6, 15.0)

    pmf_team = poisson.pmf(k[None, :], lam_t[:, None])
    pmf_opp = poisson.pmf(k[None, :], lam_o[:, None])
    pmf_team = pmf_team / pmf_team.sum(axis=1, keepdims=True)
    pmf_opp = pmf_opp / pmf_opp.sum(axis=1, keepdims=True)

    prob = pmf_team[:, :, None] * pmf_opp[:, None, :]
    expected_loss = np.einsum("abij,nij->nab", loss_tensor, prob)
    flat_idx = expected_loss.reshape(n, -1).argmin(axis=1)
    return (flat_idx // max_goals).astype(int), (flat_idx % max_goals).astype(int)


def mean_awmae(pred_t, pred_o, true_t, true_o):
    losses = [
        awmae_single(pt, po, tt, to)
        for pt, po, tt, to in zip(pred_t, pred_o, true_t, true_o)
    ]
    return float(np.mean(losses))


# ===========================================================================
# 2. MODEL WRAPPERS
# ===========================================================================
@dataclass(frozen=True)
class BlendConfig:
    weights: tuple[float, ...]
    team_scale: float
    opp_scale: float
    team_bias: float
    opp_bias: float
    score: float


class XGBPoissonModel:
    name = "xgb_poisson"

    def __init__(self, rounds):
        self.rounds = rounds
        self.model = None

    def fit(self, x, y, weight):
        dtrain = xgb.DMatrix(x, label=y, weight=weight)
        self.model = xgb.train(XGB_PARAMS, dtrain, num_boost_round=self.rounds)
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


class LGBPoissonModel:
    name = "lgb_poisson"

    def __init__(self, rounds):
        self.rounds = rounds
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


def model_factories(rounds):
    factories = [lambda: XGBPoissonModel(rounds), lambda: HGBPoissonModel(rounds)]
    if lgb is not None:
        factories.append(lambda: LGBPoissonModel(rounds))
    if CatBoostRegressor is not None:
        factories.append(lambda: CatPoissonModel(rounds))
    return factories


def fit_predict_models(x_train, y_train, w_train, x_pred, rounds):
    preds = []
    names = []
    for factory in model_factories(rounds):
        model = factory()
        print(f"      - {model.name}")
        model.fit(x_train, y_train, w_train)
        pred = np.clip(model.predict(x_pred), 1e-5, 12.0)
        preds.append(pred)
        names.append(model.name)
    return np.vstack(preds), names


# ===========================================================================
# 3. BLENDER
# ===========================================================================
def convex_weight_grid(n_models, step=0.25):
    if n_models == 1:
        return [(1.0,)]
    values = np.arange(0.0, 1.0 + 1e-9, step)
    out = []
    for combo in itertools.product(values, repeat=n_models):
        if abs(sum(combo) - 1.0) < 1e-9:
            out.append(tuple(float(v) for v in combo))
    return out


def apply_blend(pred_matrix, weights, scale, bias):
    blended = np.average(pred_matrix, axis=0, weights=np.array(weights))
    return np.clip(blended * scale + bias, 1e-5, 12.0)


def tune_blender(train_df, feature_cols, loss_tensor):
    print("\n[3] Tuning static blender on latest train window...")
    sorted_df = train_df.sort_values("date").reset_index(drop=True)
    split_idx = int(len(sorted_df) * (1.0 - VALIDATION_FRACTION))
    tr = sorted_df.iloc[:split_idx]
    val = sorted_df.iloc[split_idx:]

    x_tr = tr[feature_cols]
    x_val = val[feature_cols]
    w_tr = tr["sample_weight"].values

    print(f"    Train window: {len(tr)} | Validation window: {len(val)}")
    print("    Fitting team-goals blend models:")
    val_team_preds, names = fit_predict_models(
        x_tr, tr["team_goals"].values, w_tr, x_val, N_ESTIMATORS_BLEND
    )
    print("    Fitting opp-goals blend models:")
    val_opp_preds, opp_names = fit_predict_models(
        x_tr, tr["opp_goals"].values, w_tr, x_val, N_ESTIMATORS_BLEND
    )
    assert names == opp_names
    print(f"    Blend candidates: {', '.join(names)}")

    best = BlendConfig(
        weights=tuple([1.0 / len(names)] * len(names)),
        team_scale=1.0,
        opp_scale=1.0,
        team_bias=0.0,
        opp_bias=0.0,
        score=float("inf"),
    )
    scales = [0.94, 0.97, 1.0, 1.03, 1.06]
    biases = [-0.04, 0.0, 0.04]
    weights_grid = convex_weight_grid(len(names), step=0.25)

    for weights in weights_grid:
        base_team = np.average(val_team_preds, axis=0, weights=np.array(weights))
        base_opp = np.average(val_opp_preds, axis=0, weights=np.array(weights))
        for team_scale, opp_scale, team_bias, opp_bias in itertools.product(
            scales, scales, biases, biases
        ):
            lambda_team = np.clip(base_team * team_scale + team_bias, 1e-5, 12.0)
            lambda_opp = np.clip(base_opp * opp_scale + opp_bias, 1e-5, 12.0)
            pred_t, pred_o = erm_predict_batch(lambda_team, lambda_opp, loss_tensor)
            score = mean_awmae(
                pred_t,
                pred_o,
                val["team_goals"].values.astype(int),
                val["opp_goals"].values.astype(int),
            )
            if score < best.score:
                best = BlendConfig(weights, team_scale, opp_scale, team_bias, opp_bias, score)

    print(
        "    Best validation AW-MAE: "
        f"{best.score:.5f} | weights={dict(zip(names, best.weights))} | "
        f"team_scale={best.team_scale}, opp_scale={best.opp_scale}, "
        f"team_bias={best.team_bias}, opp_bias={best.opp_bias}"
    )
    return best, names


# ===========================================================================
# 4. FINAL PIPELINE
# ===========================================================================
def run_full_pipeline(train_df, test_df, feature_cols, loss_tensor, blend_config, names):
    print("\n[4] Training full static ensemble...")
    x_train = train_df[feature_cols]
    x_test = test_df[feature_cols]
    w_train = train_df["sample_weight"].values

    print("    Training team-goals full models:")
    team_preds, full_names = fit_predict_models(
        x_train, train_df["team_goals"].values, w_train, x_test, N_ESTIMATORS_FULL
    )
    print("    Training opp-goals full models:")
    opp_preds, opp_names = fit_predict_models(
        x_train, train_df["opp_goals"].values, w_train, x_test, N_ESTIMATORS_FULL
    )
    assert full_names == opp_names == names

    lambda_team = apply_blend(
        team_preds, blend_config.weights, blend_config.team_scale, blend_config.team_bias
    )
    lambda_opp = apply_blend(
        opp_preds, blend_config.weights, blend_config.opp_scale, blend_config.opp_bias
    )
    print(
        "    Lambda stats: "
        f"team mean={lambda_team.mean():.3f}, opp mean={lambda_opp.mean():.3f}"
    )

    print("\n[5] Applying ERM...")
    pred_team, pred_opp = erm_predict_batch(lambda_team, lambda_opp, loss_tensor)

    print("\n[6] Writing submission_v4.csv...")
    sample_sub = pd.read_csv(SAMPLE_SUB)
    sub = pd.DataFrame(
        {
            "Id": test_df["Id"].values,
            "team_goals": pred_team,
            "opp_goals": pred_opp,
        }
    )
    sub = sample_sub[["Id"]].merge(sub, on="Id", how="left")
    sub["team_goals"] = sub["team_goals"].astype(int)
    sub["opp_goals"] = sub["opp_goals"].astype(int)
    sub.to_csv(OUTPUT_SUB, index=False)
    print(f"    [OK] {OUTPUT_SUB.relative_to(BASE_DIR)} -> {len(sub)} rows")


def main():
    print("=" * 62)
    print("STATIC BLENDED PIPELINE V4 - Gammafest Masa Kite Lagi")
    print("=" * 62)
    t0 = time.time()

    train_df, test_df, feature_cols = load_data()
    print("\n[2] Building AW-MAE tensor...")
    loss_tensor = build_loss_tensor(MAX_GOALS)
    blend_config, names = tune_blender(train_df, feature_cols, loss_tensor)
    run_full_pipeline(train_df, test_df, feature_cols, loss_tensor, blend_config, names)

    print("\n[7] Local validation...")
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                f"sys.path.insert(0, {str(BASE_DIR / 'src')!r}); "
                "from evaluate_local import evaluate_submission; "
                f"evaluate_submission({str(OUTPUT_SUB)!r}, "
                f"{str(DATA_DIR / 'test_ground_truth.csv')!r})"
            ),
        ]
    )
    print(f"\nDone in {(time.time() - t0) / 60:.1f} minutes.")


if __name__ == "__main__":
    main()
