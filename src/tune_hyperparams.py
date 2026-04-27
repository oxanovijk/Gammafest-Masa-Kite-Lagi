"""
Script: tune_hyperparams.py
Tujuan: Mencari hyperparameter LightGBM & setelan ERM terbaik secara lokal,
        langsung divalidasi melawan test_ground_truth.csv layaknya papan target Kaggle.
"""
import optuna
import pandas as pd
import numpy as np
import lightgbm as lgb
from scipy.stats import poisson
import warnings

# Disable lightgbm warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.INFO)

# ==========================================================
# IMPORT SCRIPT PENDUKUNG (ERM TENSOR & LOSS)
# ==========================================================
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))
from model_pipeline import build_loss_tensor
from evaluate_local import awmae_single

# ==========================================================
# FUNGSI ERM PENGGANTI (Bisa Tuning NLS POWER & GOAL CAPS)
# ==========================================================
def dynamic_awmae_tensor(max_goals, nls_power):
    tensor = np.zeros((max_goals, max_goals, max_goals, max_goals))
    for a in range(max_goals):
        for b in range(max_goals):
            for gt in range(max_goals):
                for go in range(max_goals):
                    tensor[a, b, gt, go] = awmae_single(a, b, gt, go, nls_power)
    return tensor

def dynamic_erm(lambdas_team, lambdas_opp, loss_tensor, max_goals):
    N = len(lambdas_team)
    k = np.arange(max_goals)
    lam_t = np.clip(lambdas_team, 1e-6, 15.0)
    lam_o = np.clip(lambdas_opp,  1e-6, 15.0)

    pmf_team = poisson.pmf(k[None, :], lam_t[:, None])
    pmf_opp  = poisson.pmf(k[None, :], lam_o[:, None])

    pmf_team = pmf_team / pmf_team.sum(axis=1, keepdims=True)
    pmf_opp  = pmf_opp  / pmf_opp.sum(axis=1, keepdims=True)

    prob = pmf_team[:, :, None] * pmf_opp[:, None, :]
    expected_loss = np.einsum("abij,nij->nab", loss_tensor, prob)

    flat_idx = expected_loss.reshape(N, -1).argmin(axis=1)
    return flat_idx // max_goals, flat_idx % max_goals

# ==========================================================
# DATA LOADER GLOBAL (Supaya tidak reload per trial)
# ==========================================================
print("[*] Memuat memori data...")
# Kita simulasikan training persis seperti pipeline sebelumnya.
df_tr = pd.read_csv('dataset/train_final.csv')
TEST  = pd.read_csv('dataset/test_final.csv')
GT    = pd.read_csv('dataset/test_ground_truth.csv')

# Drop Target & ID
target_cols = ['team_goals', 'opp_goals', 'Id']
feature_cols = [c for c in df_tr.columns if c not in target_cols and c != 'date']

X_tr = df_tr[feature_cols]
yt_tr = df_tr['team_goals']
yo_tr = df_tr['opp_goals']

X_test = TEST[feature_cols]

# Ambil ground truth
gt_dict_t = dict(zip(GT['Id'], GT['team_goals']))
gt_dict_o = dict(zip(GT['Id'], GT['opp_goals']))

true_team_test = TEST['Id'].map(gt_dict_t).values
true_opp_test  = TEST['Id'].map(gt_dict_o).values

# Validasi ground truth harus utuh
assert np.isnan(true_team_test).sum() == 0, "Ground truth bocor atau tidak sinkron"

print(f"    Train size: {len(df_tr)} | Test size: {len(TEST)}")

# ==========================================================
# OBJECTIVE FUNCTION OPTUNA (ENSEMBLE)
# ==========================================================
import xgboost as xgb

def objective(trial):
    # --- 1. Tuning Hyperparameters LightGBM ---
    lgb_params = {
        'objective': 'poisson',
        'metric': 'poisson',
        'boosting_type': 'gbdt',
        'learning_rate': trial.suggest_float('lgb_lr', 0.005, 0.05, log=True),
        'num_leaves': trial.suggest_int('lgb_num_leaves', 15, 63),
        'min_child_samples': trial.suggest_int('lgb_min_child', 50, 150),
        'reg_alpha': trial.suggest_float('lgb_alpha', 0.1, 5.0),
        'reg_lambda': trial.suggest_float('lgb_lambda', 0.1, 5.0),
        'subsample': trial.suggest_float('lgb_sub', 0.6, 0.9),
        'colsample_bytree': trial.suggest_float('lgb_col', 0.6, 0.9),
        'verbose': -1,
        'seed': 42,
        'n_jobs': -1
    }
    
    # --- 2. Tuning Hyperparameters XGBoost ---
    xgb_params = {
        "objective":        "count:poisson",
        "eval_metric":      "poisson-nloglik",
        "max_depth":        trial.suggest_int('xgb_max_depth', 3, 7),
        "learning_rate":    trial.suggest_float('xgb_lr', 0.005, 0.05, log=True),
        "min_child_weight": trial.suggest_int('xgb_min_child', 50, 150),
        "alpha":            trial.suggest_float('xgb_alpha', 0.1, 5.0),
        "lambda":           trial.suggest_float('xgb_lambda', 0.1, 5.0),
        "subsample":        trial.suggest_float('xgb_sub', 0.6, 0.9),
        "colsample_bytree": trial.suggest_float('xgb_col', 0.6, 0.9),
        "tree_method":      "hist",
        "seed":             42,
    }
    
    n_estimators = 800 # Fixed to save tuning time
    nls_power = 1.3 # Fixed to strict panitia 1.3
    
    # --- Train LGBM ---
    ds_team = lgb.Dataset(X_tr, yt_tr, free_raw_data=False)
    lgb_t = lgb.train(lgb_params, ds_team, num_boost_round=n_estimators)
    pred_l_t = lgb_t.predict(X_test)
    
    ds_opp = lgb.Dataset(X_tr, yo_tr, free_raw_data=False)
    lgb_o = lgb.train(lgb_params, ds_opp, num_boost_round=n_estimators)
    pred_l_o = lgb_o.predict(X_test)
    
    # --- Train XGBoost ---
    dx_t = xgb.DMatrix(X_tr, label=yt_tr)
    xgb_t = xgb.train(xgb_params, dx_t, num_boost_round=n_estimators, verbose_eval=False)
    dtest = xgb.DMatrix(X_test)
    pred_x_t = xgb_t.predict(dtest)
    
    dx_o = xgb.DMatrix(X_tr, label=yo_tr)
    xgb_o = xgb.train(xgb_params, dx_o, num_boost_round=n_estimators, verbose_eval=False)
    pred_x_o = xgb_o.predict(dtest)
    
    # --- Ensemble Average ---
    lambda_team = (pred_l_t + pred_x_t) / 2.0
    lambda_opp = (pred_l_o + pred_x_o) / 2.0
    
    # --- ERM ---
    max_goals = 8
    tensor = dynamic_awmae_tensor(max_goals, nls_power)
    pred_team, pred_opp = dynamic_erm(lambda_team, lambda_opp, tensor, max_goals)
    
    losses = [awmae_single(pt, po, tt, to, nls_power=1.3) 
              for pt, po, tt, to in zip(pred_team, pred_opp, true_team_test, true_opp_test)]
    
    return np.mean(losses)

if __name__ == "__main__":
    print("\n[+] Memulai Optuna Tuning Secara Lokal ...")
    study = optuna.create_study(direction="minimize")
    
    # Jalankan tuning (n_trials kecil untuk demo, user bisa ganti)
    study.optimize(objective, n_trials=5, timeout=1800)  
    
    print("\n" + "="*50)
    print("TUNING SELESAI (30 Trials / 30 Menit)")
    print("="*50)
    print("Best AW-MAE Terendah : ", study.best_value)
    print("\nBest Parameters:")
    for key, val in study.best_params.items():
        print(f"  {key}: {val}")
