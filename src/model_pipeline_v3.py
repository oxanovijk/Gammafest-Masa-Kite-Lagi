"""
ML Pipeline V3 -- Gammafest Masa Kite Lagi
========================================
Arsitektur:
  1. Static Decayed-Ensemble (LGBM + XGBoost Poisson Regression)
  2. Elo Time-Decay untuk memerangi open-loop drift (menggantikan iterative pipeline)
  3. Averaging Lambda Ensemble
  4. Full Expected Risk Minimization (Bivariate Poisson -> AW-MAE)
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from scipy.stats import poisson
from pathlib import Path
import warnings
import time

warnings.filterwarnings("ignore")

# ===========================================================================
# 0. KONFIGURASI
# ===========================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "dataset"

TRAIN_FINAL = DATA_DIR / "train_final.csv"
TEST_FINAL  = DATA_DIR / "test_final.csv"
TRAIN_RAW   = DATA_DIR / "train.csv"
TEST_RAW    = DATA_DIR / "test.csv"
SAMPLE_SUB  = DATA_DIR / "sample submission.csv"
OUTPUT_SUB  = DATA_DIR / "submission_v3.csv"

# --- ERM Config ---
MAX_GOALS = 10        # Matriks skor 0-9
NLS_POWER = 1.3       # Non-linear scaling exponent

# --- Decay Config ---
DECAY_HALF_LIFE_DAYS = 1500 # Poin Elo akan memudar 50% setiap 4 tahun.

# --- LightGBM Config ---
LGB_PARAMS = {
    "objective":        "poisson",
    "metric":           "poisson",
    "num_leaves":       60,
    "learning_rate":    0.0165,
    "min_child_samples": 106,
    "reg_alpha":        4.04,
    "reg_lambda":       1.39,
    "subsample":        0.61,
    "colsample_bytree": 0.72,
    "verbose":          -1,
    "n_jobs":           -1,
    "seed":             42,
}

# --- XGBoost Config ---
XGB_PARAMS = {
    "objective":        "count:poisson",
    "eval_metric":      "poisson-nloglik",
    "max_depth":        7,
    "learning_rate":    0.046,
    "min_child_weight": 106,
    "alpha":            1.81,
    "lambda":           4.91,
    "subsample":        0.74,
    "colsample_bytree": 0.80,
    "tree_method":      "hist",
    "seed":             42,
}

N_ESTIMATORS    = 800
EARLY_STOPPING  = 100

# --- Tournament Weights (AW-MAE) ---
TOURNAMENT_WEIGHT_MAP = {
    "FIFA World Cup":                           2.00,
    "AFC Asian Cup":                            1.80,
    "AFC Championship":                         1.80,
    "African Cup of Nations":                   1.80,
    "Copa America":                             1.80,
    "UEFA Euro":                                1.80,
    "Gold Cup":                                 1.70,
    "CONCACAF Championship":                    1.70,
    "Oceania Nations Cup":                      1.60,
    "Confederations Cup":                       1.70,
    "Finalissima":                              1.70,
    "FIFA World Cup qualification":             1.50,
    "Olympic Games":                            1.50,
    "UEFA Euro qualification":                  1.40,
    "African Cup of Nations qualification":     1.40,
    "AFC Asian Cup qualification":              1.40,
    "CONCACAF Gold Cup qualification":          1.30,
    "UEFA Nations League":                      1.50,
    "CONCACAF Nations League":                  1.40,
    "CONMEBOL Nations League":                  1.40,
    "Friendly":                                 0.96,
}
DEFAULT_TOURNAMENT_WEIGHT = 1.20

# ===========================================================================
# 1. DATA LOADING & TIME-DECAY
# ===========================================================================
def load_data():
    print("[1] Loading data...")
    train = pd.read_csv(TRAIN_FINAL)
    test  = pd.read_csv(TEST_FINAL)

    raw_train = pd.read_csv(TRAIN_RAW, usecols=["Id", "date", "tournament"])
    raw_test  = pd.read_csv(TEST_RAW,  usecols=["Id", "date", "tournament"])

    train = train.merge(raw_train, on="Id", how="left")
    test  = test.merge(raw_test, on="Id", how="left")

    train["date"] = pd.to_datetime(train["date"])
    test["date"]  = pd.to_datetime(test["date"])
    
    # [OBSERVATION]: Elo decay removed. 2011 Elo is a persistent indicator of pedigree.

    train["sample_weight"] = train["tournament"].map(
        TOURNAMENT_WEIGHT_MAP
    ).fillna(DEFAULT_TOURNAMENT_WEIGHT)

    exclude = {"Id", "team_goals", "opp_goals", "date", "tournament",
               "sample_weight", "is_test"}
    feature_cols = [c for c in train.columns if c not in exclude]

    print(f"    Train: {train.shape}  |  Test: {test.shape}")
    print(f"    Fitur: {len(feature_cols)} kolom")

    return train, test, feature_cols

# ===========================================================================
# 2. AW-MAE METRIC
# ===========================================================================
def awmae_single(pred_t, pred_o, true_t, true_o):
    mae = (abs(pred_t - true_t) + abs(pred_o - true_o)) / 2.0
    exact = 1 if (pred_t == true_t and pred_o == true_o) else 0
    pred_outcome = np.sign(pred_t - pred_o)
    true_outcome = np.sign(true_t - true_o)
    outcome_ok = 1 if pred_outcome == true_outcome else 0
    gd_ok = 1 if (pred_t - pred_o) == (true_t - true_o) else 0
    augmented = mae + 0.30 * (1 - exact) + 0.25 * (1 - outcome_ok) + 0.15 * (1 - gd_ok)
    multiplier = 1.0 if outcome_ok else 1.5
    scaled = (augmented * multiplier) ** NLS_POWER
    return scaled

def build_loss_tensor(max_goals=MAX_GOALS):
    tensor = np.zeros((max_goals, max_goals, max_goals, max_goals))
    for a in range(max_goals):
        for b in range(max_goals):
            for gt in range(max_goals):
                for go in range(max_goals):
                    tensor[a, b, gt, go] = awmae_single(a, b, gt, go)
    return tensor

def erm_predict_batch(lambdas_team, lambdas_opp, loss_tensor):
    N = len(lambdas_team)
    M = loss_tensor.shape[0]
    k = np.arange(M)
    lam_t = np.clip(lambdas_team, 1e-6, 15.0)
    lam_o = np.clip(lambdas_opp,  1e-6, 15.0)
    pmf_team = poisson.pmf(k[None, :], lam_t[:, None])
    pmf_opp  = poisson.pmf(k[None, :], lam_o[:, None])
    pmf_team = pmf_team / pmf_team.sum(axis=1, keepdims=True)
    pmf_opp  = pmf_opp  / pmf_opp.sum(axis=1, keepdims=True)
    prob = pmf_team[:, :, None] * pmf_opp[:, None, :]
    expected_loss = np.einsum("abij,nij->nab", loss_tensor, prob)
    flat_idx = expected_loss.reshape(N, -1).argmin(axis=1)
    return (flat_idx // M).astype(int), (flat_idx % M).astype(int)

# ===========================================================================
# 3. ENSEMBLE TRAINING
# ===========================================================================
def train_lgb_xgb(X_train, y_train, X_val, y_val, w_train=None, w_val=None):
    # Train LGB
    dtrain_l = lgb.Dataset(X_train, y_train, weight=w_train, free_raw_data=False)
    dval_l = lgb.Dataset(X_val, y_val, weight=w_val, reference=dtrain_l, free_raw_data=False)
    callbacks = [lgb.early_stopping(EARLY_STOPPING, verbose=False), lgb.log_evaluation(0)]
    model_lgb = lgb.train(LGB_PARAMS, dtrain_l, num_boost_round=N_ESTIMATORS, valid_sets=[dval_l], callbacks=callbacks)
    
    # Train XGB
    dtrain_x = xgb.DMatrix(X_train, label=y_train, weight=w_train)
    dval_x = xgb.DMatrix(X_val, label=y_val, weight=w_val)
    model_xgb = xgb.train(XGB_PARAMS, dtrain_x, num_boost_round=N_ESTIMATORS, evals=[(dval_x, "val")], early_stopping_rounds=EARLY_STOPPING, verbose_eval=False)
    
    return model_lgb, model_xgb

def predict_ensemble(model_lgb, model_xgb, X):
    pred_l = model_lgb.predict(X)
    dtest = xgb.DMatrix(X)
    pred_x = model_xgb.predict(dtest)
    return (pred_l + pred_x) / 2.0

# ===========================================================================
# 4. FULL PIPELINE
# ===========================================================================
def train_lgb_xgb_full(X_train, y_train, w_train):
    # Train LGB (No early stopping, 100% data)
    dtrain_l = lgb.Dataset(X_train, y_train, weight=w_train, free_raw_data=False)
    model_lgb = lgb.train(LGB_PARAMS, dtrain_l, num_boost_round=N_ESTIMATORS)
    
    # Train XGB (No early stopping, 100% data)
    dtrain_x = xgb.DMatrix(X_train, label=y_train, weight=w_train)
    model_xgb = xgb.train(XGB_PARAMS, dtrain_x, num_boost_round=N_ESTIMATORS)
    
    return model_lgb, model_xgb

def run_full_pipeline(train_df, test_df, feature_cols, loss_tensor):
    print("\n[4] Training final models on full data (100% Train, NO Early Stopping)...")
    X_train = train_df[feature_cols]
    X_test  = test_df[feature_cols]
    w_train = train_df["sample_weight"].values

    print("    Training team_goals ensemble...")
    lgb_t, xgb_t = train_lgb_xgb_full(X_train, train_df["team_goals"].values, w_train)

    print("    Training opp_goals ensemble...")
    lgb_o, xgb_o = train_lgb_xgb_full(X_train, train_df["opp_goals"].values, w_train)

    print("\n[5] Predicting test set with Ensemble...")
    lambda_team = predict_ensemble(lgb_t, xgb_t, X_test)
    lambda_opp  = predict_ensemble(lgb_o, xgb_o, X_test)

    print("\n[6] Applying Expected Risk Minimization...")
    pred_team, pred_opp = erm_predict_batch(lambda_team, lambda_opp, loss_tensor)

    print("\n[7] Generating submission_v3.csv...")
    sample_sub = pd.read_csv(SAMPLE_SUB)
    sub = pd.DataFrame({"Id": test_df["Id"].values, "team_goals": pred_team, "opp_goals":  pred_opp})
    sub = sample_sub[["Id"]].merge(sub, on="Id", how="left")
    
    sub["team_goals"] = sub["team_goals"].astype(int)
    sub["opp_goals"]  = sub["opp_goals"].astype(int)
    sub.to_csv(OUTPUT_SUB, index=False)
    
    print("    [OK] Selesai! Validating local...")
    import subprocess
    subprocess.run(["python", "src/evaluate_local.py", "dataset/submission_v3.csv"])

def main():
    train_df, test_df, feature_cols = load_data()
    loss_tensor = build_loss_tensor(MAX_GOALS)
    run_full_pipeline(train_df, test_df, feature_cols, loss_tensor)

if __name__ == "__main__":
    main()
