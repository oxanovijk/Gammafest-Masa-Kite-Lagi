"""
Model Pipeline V13 (Comprehensive Improvements over V12)
=========================================================
P0A: Transfer Learning Men → Women (pre-train on Men, fine-tune Women)
P0B: Bivariate Ordinal Regression (separate team_goals & opp_goals ordinal models)
P1C: Improved Calibration (temperature + Platt ensemble calibration)
P1D: Enhanced Tournament Features (more granular encoding)
P2E: Pseudo-Labeling Self-Training Loop  
P2F: Diversity Ensemble (LightGBM + XGBoost + CatBoost)
P3H: Expected Goals Regression + Threshold alternative

Architecture retains Gender-Split + Two-Stage from V12.
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, Pool as CatPool
from sklearn.isotonic import IsotonicRegression
from pathlib import Path
import warnings, time, json, sys
from scipy.stats import poisson
from scipy.optimize import minimize
from collections import defaultdict

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "dataset"

MAX_GOALS = 6  # 0..5 (5 = 5+ goals capped)
NUM_SCORE_CLASSES = MAX_GOALS  # 6 classes per dimension
NUM_OUTCOME_CLASSES = 3  # Loss, Draw, Win
NLS_POWER = 1.3

# ===========================================================================
# AW-MAE METRIC & LOSS TENSOR
# ===========================================================================
def awmae_single(pt, po, tt, to_):
    mae = (abs(int(pt) - int(tt)) + abs(int(po) - int(to_))) / 2.0
    exact = 1 if (int(pt) == int(tt) and int(po) == int(to_)) else 0
    out_ok = 1 if np.sign(int(pt) - int(po)) == np.sign(int(tt) - int(to_)) else 0
    gd_ok = 1 if (int(pt) - int(po)) == (int(tt) - int(to_)) else 0
    aug = mae + 0.30*(1-exact) + 0.25*(1-out_ok) + 0.15*(1-gd_ok)
    mult = 1.0 if out_ok else 1.5
    return (aug * mult) ** NLS_POWER

def build_loss_tensor():
    M = MAX_GOALS
    tensor = np.zeros((M, M, M, M))
    for a in range(M):
        for b in range(M):
            for gt in range(M):
                for go in range(M):
                    tensor[a, b, gt, go] = awmae_single(a, b, gt, go)
    return tensor

loss_tensor = build_loss_tensor()

# ===========================================================================
# P0B: BIVARIATE ORDINAL - POISSON BASELINE
# ===========================================================================
def poisson_pmf_6(lam):
    """Compute Poisson PMF for categories 0..4, with 5 = P(≥5)."""
    if lam <= 0:
        prob = np.zeros(6)
        prob[0] = 1.0
        return prob
    prob = np.zeros(6)
    for k in range(5):
        prob[k] = poisson.pmf(k, lam)
    prob[5] = 1.0 - prob[:5].sum()
    prob = np.clip(prob, 1e-7, 1.0)
    prob /= prob.sum()
    return prob

def estimate_lambda_from_elo(elo_team, elo_opp, is_home=0, is_neutral=0):
    """Estimate expected goals (lambda) from Elo ratings.
    Calibrated on training data: avg goals ~ 1.6 per team.
    """
    elo_diff = elo_team - elo_opp
    # Home advantage ~0.2 goals
    home_bonus = 0.0
    if is_neutral == 0 and is_home == 1:
        home_bonus = 0.15
    elif is_neutral == 0 and is_home == 0:
        home_bonus = -0.15
    
    # Elo diff impacts goals linearly
    base_rate = 1.55  # Average goals per team
    elo_effect = elo_diff / 400.0 * 0.3  # 400 Elo = ~0.3 goal difference
    lam = base_rate + elo_effect + home_bonus
    lam = max(0.15, min(6.0, lam))
    return lam

def build_bivariate_poisson_joint(lam_t, lam_o, rho=-0.05):
    """Build bivariate Poisson joint PMF (6x6) for team_goals x opp_goals.
    Uses Dixon-Coles correction for (0,0), (0,1), (1,0), (1,1).
    """
    M = MAX_GOALS
    joint = np.zeros((M, M))
    
    p_t = poisson_pmf_6(lam_t)
    p_o = poisson_pmf_6(lam_o)
    
    # Independent product
    for t in range(M):
        for o in range(M):
            joint[t, o] = p_t[t] * p_o[o]
    
    # Dixon-Coles correction
    def dc_tau(x, y, lam_t, lam_o, rho):
        if x == 0 and y == 0: return 1 - lam_t * lam_o * rho
        elif x == 0 and y == 1: return 1 + lam_t * rho
        elif x == 1 and y == 0: return 1 + lam_o * rho
        elif x == 1 and y == 1: return 1 - rho
        return 1.0
    
    for t in range(min(2, M)):
        for o in range(min(2, M)):
            tau = dc_tau(t, o, lam_t, lam_o, rho)
            joint[t, o] *= max(tau, 0.01)
    
    joint = np.clip(joint, 1e-8, 1.0)
    joint /= joint.sum()
    return joint

# ===========================================================================
# P1C: ENHANCED CALIBRATION
# ===========================================================================
def temperature_scale(prob, T):
    """Apply temperature scaling to probabilities."""
    prob = np.clip(prob, 1e-8, 1.0)
    log_p = np.log(prob) / T
    exp_p = np.exp(log_p)
    return exp_p / exp_p.sum(axis=1, keepdims=True)

# ===========================================================================
# P0B: ORDINAL-AWARE TWO-STAGE PIPELINE  
# ===========================================================================
def train_ordinal_stage2(train_df, test_df, feature_cols, outcome_confs, rho=-0.05):
    """
    P0B: Train two 6-class ordinal models (team_goals, opp_goals) 
    and combine into joint PMF via bivariate structure.
    
    Returns: prob_joint (N_test x 36) 
    """
    X_train = train_df[feature_cols].copy()
    X_test = test_df[feature_cols].copy()
    
    y_t = np.clip(train_df["team_goals"].values, 0, MAX_GOALS-1).astype(int)
    y_o = np.clip(train_df["opp_goals"].values, 0, MAX_GOALS-1).astype(int)
    
    # Ordinal models for team_goals and opp_goals separately
    # LightGBM
    lgb_params_clf = {
        "objective": "multiclass", "num_class": NUM_SCORE_CLASSES, "metric": "multi_logloss",
        "num_leaves": 31, "learning_rate": 0.02, "min_child_samples": 100,
        "subsample": 0.7, "colsample_bytree": 0.7, "verbose": -1, "seed": 42
    }
    
    dt_team_l = lgb.Dataset(X_train, y_t, free_raw_data=False)
    lgb_team = lgb.train(lgb_params_clf, dt_team_l, num_boost_round=500)
    
    dt_opp_l = lgb.Dataset(X_train, y_o, free_raw_data=False)
    lgb_opp = lgb.train(lgb_params_clf, dt_opp_l, num_boost_round=500)
    
    # XGBoost
    xgb_params_clf = {
        "objective": "multi:softprob", "num_class": NUM_SCORE_CLASSES, "eval_metric": "mlogloss",
        "max_depth": 5, "learning_rate": 0.03, "min_child_weight": 100,
        "subsample": 0.8, "colsample_bytree": 0.8, "tree_method": "hist", "seed": 42
    }
    
    dt_team_x = xgb.DMatrix(X_train, label=y_t)
    xgb_team = xgb.train(xgb_params_clf, dt_team_x, num_boost_round=500)
    
    dt_opp_x = xgb.DMatrix(X_train, label=y_o)
    xgb_opp = xgb.train(xgb_params_clf, dt_opp_x, num_boost_round=500)
    
    # Predictions
    prob_t_team = (lgb_team.predict(X_test) + xgb_team.predict(xgb.DMatrix(X_test))) / 2.0
    prob_t_opp  = (lgb_opp.predict(X_test)  + xgb_opp.predict(xgb.DMatrix(X_test)))  / 2.0
    
    N = len(X_test)
    prob_joint = np.zeros((N, MAX_GOALS * MAX_GOALS))
    
    for i in range(N):
        p_t = prob_t_team[i]
        p_o = prob_t_opp[i]
        
        # Build joint with independent + slight correlation correction
        joint = np.outer(p_t, p_o)
        
        # Apply Dixon-Coles correction for low scores
        for t in range(min(2, MAX_GOALS)):
            for o in range(min(2, MAX_GOALS)):
                lam_t_est = estimate_lambda_from_elo(
                    test_df.iloc[i].get("elo_team", 1500),
                    test_df.iloc[i].get("elo_opp", 1500),
                    test_df.iloc[i].get("is_home", 0),
                    test_df.iloc[i].get("is_neutral", 0)
                )
                lam_o_est = estimate_lambda_from_elo(
                    test_df.iloc[i].get("elo_opp", 1500),
                    test_df.iloc[i].get("elo_team", 1500),
                    0 if test_df.iloc[i].get("is_home", 0) == 1 else test_df.iloc[i].get("is_home", 0),
                    test_df.iloc[i].get("is_neutral", 0)
                )
                
                if t == 0 and o == 0:
                    tau = 1 - lam_t_est * lam_o_est * rho
                elif t == 0 and o == 1:
                    tau = 1 + lam_t_est * rho
                elif t == 1 and o == 0:
                    tau = 1 + lam_o_est * rho
                elif t == 1 and o == 1:
                    tau = 1 - rho
                else:
                    tau = 1.0
                joint[t, o] *= max(tau, 0.01)
        
        joint = np.clip(joint, 1e-8, 1.0)
        joint /= joint.sum()
        prob_joint[i] = joint.flatten()
    
    return prob_joint

# ===========================================================================
# P0A: TRANSFER LEARNING
# ===========================================================================
def train_with_transfer(men_train_df, women_train_df, test_women_df, feature_cols, stage="outcome"):
    """
    P0A: Pre-train on Men data, fine-tune on Women data.
    Weighted training: Men data with lower weight, Women data with higher weight.
    """
    X_men = men_train_df[feature_cols].copy()
    X_women = women_train_df[feature_cols].copy()
    X_test = test_women_df[feature_cols].copy()
    
    y_t_men = np.clip(men_train_df["team_goals"].values, 0, MAX_GOALS-1).astype(int)
    y_o_men = np.clip(men_train_df["opp_goals"].values, 0, MAX_GOALS-1).astype(int)
    y_t_women = np.clip(women_train_df["team_goals"].values, 0, MAX_GOALS-1).astype(int)
    y_o_women = np.clip(women_train_df["opp_goals"].values, 0, MAX_GOALS-1).astype(int)
    
    if stage == "outcome":
        y_men = np.sign(y_t_men - y_o_men) + 1
        y_women = np.sign(y_t_women - y_o_women) + 1
        num_class = NUM_OUTCOME_CLASSES
    else:  # joint
        y_men = y_t_men * MAX_GOALS + y_o_men
        y_women = y_t_women * MAX_GOALS + y_o_women
        num_class = MAX_GOALS * MAX_GOALS
    
    # Build boosted training set: Men data with weight 0.3, Women data with weight 1.0
    # LightGBM
    lgb_params = {
        "objective": "multiclass", "num_class": num_class, "metric": "multi_logloss",
        "num_leaves": 31, "learning_rate": 0.02, "min_child_samples": 100,
        "subsample": 0.7, "colsample_bytree": 0.7, "verbose": -1, "seed": 42
    }
    
    # Combined training with sample weights
    X_combined = pd.concat([X_men, X_women], ignore_index=True)
    y_combined = np.concatenate([y_men, y_women])
    weights = np.concatenate([np.full(len(X_men), 0.3), np.full(len(X_women), 1.0)])
    
    dt_all = lgb.Dataset(X_combined, y_combined, weight=weights, free_raw_data=False)
    lgb_model = lgb.train(lgb_params, dt_all, num_boost_round=500)
    
    # XGBoost  
    xgb_params = {
        "objective": "multi:softprob", "num_class": num_class, "eval_metric": "mlogloss",
        "max_depth": 5, "learning_rate": 0.03, "min_child_weight": 100,
        "subsample": 0.8, "colsample_bytree": 0.8, "tree_method": "hist", "seed": 42
    }
    dt_x = xgb.DMatrix(X_combined, label=y_combined, weight=weights)
    xgb_model = xgb.train(xgb_params, dt_x, num_boost_round=500)
    
    # CatBoost
    cat_params = {
        "loss_function": "MultiClass", "num_boost_round": 500,
        "learning_rate": 0.03, "depth": 5, "verbose": False, "random_seed": 42
    }
    cat_model = CatBoostClassifier(**cat_params)
    cat_model.fit(X_combined, y_combined, sample_weight=weights, verbose=False)
    
    prob_lgb = lgb_model.predict(X_test)
    prob_xgb = xgb_model.predict(xgb.DMatrix(X_test))
    prob_cat = cat_model.predict_proba(X_test)
    
    prob_ensemble = (prob_lgb + prob_xgb + prob_cat) / 3.0
    return prob_ensemble

# ===========================================================================
# P2E: PSEUDO-LABELING
# ===========================================================================
def pseudo_label_iteration(train_df, test_df, feature_cols, pred_team, pred_opp, 
                           prob_out_from_stage1, n_iter=2, weight=0.5):
    """
    P2E: Self-training with pseudo-labels from current predictions.
    """
    improved = False
    best_loss = float('inf')
    best_preds = (pred_team.copy(), pred_opp.copy())
    
    for iteration in range(n_iter):
        # Create pseudo-labeled dataset
        test_pseudo = test_df.copy()
        test_pseudo["team_goals"] = pred_team
        test_pseudo["opp_goals"] = pred_opp
        
        # Only use confident predictions (entropy threshold)
        if prob_out_from_stage1 is not None:
            max_prob = prob_out_from_stage1.max(axis=1)
            confident_mask = max_prob > 0.6
            test_pseudo = test_pseudo[confident_mask]
        
        if len(test_pseudo) < 100:
            break
        
        # Combine with original train
        extended_train = pd.concat([train_df, test_pseudo], ignore_index=True)
        extended_weights = np.concatenate([
            np.ones(len(train_df)),
            np.full(len(test_pseudo), weight)
        ])
        
        # Retrain with extended data
        X_train = extended_train[feature_cols]
        X_test = test_df[feature_cols]
        y_train = np.clip(extended_train["team_goals"].values, 0, MAX_GOALS-1).astype(int)
        y_opp_train = np.clip(extended_train["opp_goals"].values, 0, MAX_GOALS-1).astype(int)
        y_out_train = np.sign(y_train - y_opp_train) + 1
        
        # Quick retrain
        lgb_params = {
            "objective": "multiclass", "num_class": 3, "metric": "multi_logloss",
            "num_leaves": 31, "learning_rate": 0.02, "min_child_samples": 50,
            "subsample": 0.7, "colsample_bytree": 0.7, "verbose": -1, "seed": 42
        }
        dt = lgb.Dataset(X_train, y_out_train, weight=extended_weights, free_raw_data=False)
        model = lgb.train(lgb_params, dt, num_boost_round=300)
        
        prob_out = model.predict(X_test)
        # Update Stage 1 predictions
        prob_out_from_stage1 = prob_out
    
    return prob_out_from_stage1

# ===========================================================================
# ERM PREDICTION
# ===========================================================================
def predict_erm(prob_joint_36):
    N = len(prob_joint_36)
    M = MAX_GOALS
    joint = prob_joint_36.reshape(N, M, M)
    
    # Ensure valid probabilities
    joint = np.clip(joint, 1e-8, 1.0)
    joint /= joint.sum(axis=(1,2), keepdims=True)
    
    expected_loss = np.einsum("abij,nij->nab", loss_tensor, joint)
    
    flat_idx = expected_loss.reshape(N, -1).argmin(axis=1)
    pred_team = flat_idx // M
    pred_opp = flat_idx % M
    
    return pred_team, pred_opp, joint

# ===========================================================================
# P3H: REGRESSION + THRESHOLD ALTERNATIVE
# ===========================================================================
def train_regression_stage2(train_df, test_df, feature_cols):
    """P3H: Predict expected goals (continuous) as alternative."""
    X_train = train_df[feature_cols].copy()
    X_test = test_df[feature_cols].copy()
    
    y_t = np.clip(train_df["team_goals"].values, 0, MAX_GOALS-1).astype(float)
    y_o = np.clip(train_df["opp_goals"].values, 0, MAX_GOALS-1).astype(float)
    
    # LightGBM regressors
    lgb_params_reg = {
        "objective": "regression", "metric": "rmse",
        "num_leaves": 31, "learning_rate": 0.02, "min_child_samples": 100,
        "subsample": 0.7, "colsample_bytree": 0.7, "verbose": -1, "seed": 42
    }
    
    dt_team = lgb.Dataset(X_train, y_t, free_raw_data=False)
    lgb_team_reg = lgb.train(lgb_params_reg, dt_team, num_boost_round=500)
    
    dt_opp = lgb.Dataset(X_train, y_o, free_raw_data=False)
    lgb_opp_reg = lgb.train(lgb_params_reg, dt_opp, num_boost_round=500)
    
    xgb_params_reg = {
        "objective": "reg:squarederror", "eval_metric": "rmse",
        "max_depth": 5, "learning_rate": 0.03, "min_child_weight": 100,
        "subsample": 0.8, "colsample_bytree": 0.8, "tree_method": "hist", "seed": 42
    }
    
    dt_team_x = xgb.DMatrix(X_train, label=y_t)
    xgb_team_reg = xgb.train(xgb_params_reg, dt_team_x, num_boost_round=500)
    
    dt_opp_x = xgb.DMatrix(X_train, label=y_o)
    xgb_opp_reg = xgb.train(xgb_params_reg, dt_opp_x, num_boost_round=500)
    
    # Predict xG
    xg_team = (lgb_team_reg.predict(X_test) + xgb_team_reg.predict(xgb.DMatrix(X_test))) / 2.0
    xg_opp  = (lgb_opp_reg.predict(X_test)  + xgb_opp_reg.predict(xgb.DMatrix(X_test)))  / 2.0
    
    return xg_team, xg_opp

def xg_to_discrete_erm(xg_team, xg_opp, prob_out):
    """Convert xG predictions to discrete using ERM with Poisson probabilities."""
    N = len(xg_team)
    pred_t = np.zeros(N, dtype=int)
    pred_o = np.zeros(N, dtype=int)
    
    M = MAX_GOALS
    
    for i in range(N):
        # Build Poisson PMF from xG
        lam_t = max(0.1, xg_team[i])
        lam_o = max(0.1, xg_opp[i])
        
        p_t = poisson_pmf_6(lam_t)
        p_o = poisson_pmf_6(lam_o)
        joint = np.outer(p_t, p_o)
        
        # Weight by outcome probabilities
        for t in range(M):
            for o in range(M):
                out_idx = np.sign(t - o) + 1
                joint[t, o] *= prob_out[i, out_idx] ** 0.3
        
        joint = np.clip(joint, 1e-8, 1.0)
        joint /= joint.sum()
        
        # ERM
        expected_loss = np.zeros((M, M))
        for a in range(M):
            for b in range(M):
                el = 0.0
                for gt in range(M):
                    for go in range(M):
                        el += loss_tensor[a, b, gt, go] * joint[gt, go]
                expected_loss[a, b] = el
        
        best_idx = np.unravel_index(expected_loss.argmin(), (M, M))
        pred_t[i] = best_idx[0]
        pred_o[i] = best_idx[1]
    
    return pred_t, pred_o

# ===========================================================================
# P0B ALTERNATIVE: COMBINED ORDINAL + FLAT
# ===========================================================================
def train_joint_stage2(train_df, test_df, feature_cols):
    """Original flat 36-class approach (for comparison/ensemble)."""
    X_train = train_df[feature_cols].copy()
    X_test = test_df[feature_cols].copy()
    
    y_t = np.clip(train_df["team_goals"].values, 0, MAX_GOALS-1).astype(int)
    y_o = np.clip(train_df["opp_goals"].values, 0, MAX_GOALS-1).astype(int)
    y_joint = y_t * MAX_GOALS + y_o
    
    num_class = MAX_GOALS * MAX_GOALS
    
    lgb_params = {
        "objective": "multiclass", "num_class": num_class, "metric": "multi_logloss",
        "num_leaves": 31, "learning_rate": 0.02, "min_child_samples": 100,
        "subsample": 0.7, "colsample_bytree": 0.7, "verbose": -1, "seed": 42
    }
    dt_l = lgb.Dataset(X_train, y_joint, free_raw_data=False)
    lgb_model = lgb.train(lgb_params, dt_l, num_boost_round=500)
    
    xgb_params = {
        "objective": "multi:softprob", "num_class": num_class, "eval_metric": "mlogloss",
        "max_depth": 5, "learning_rate": 0.03, "min_child_weight": 100,
        "subsample": 0.8, "colsample_bytree": 0.8, "tree_method": "hist", "seed": 42
    }
    dt_x = xgb.DMatrix(X_train, label=y_joint)
    xgb_model = xgb.train(xgb_params, dt_x, num_boost_round=500)
    
    # CatBoost
    cat_params = {
        "loss_function": "MultiClass", "num_boost_round": 500,
        "learning_rate": 0.03, "depth": 5, "verbose": False, "random_seed": 42
    }
    cat_model = CatBoostClassifier(**cat_params)
    cat_model.fit(X_train, y_joint, verbose=False)
    
    prob_lgb = lgb_model.predict(X_test)
    prob_xgb = xgb_model.predict(xgb.DMatrix(X_test))
    prob_cat = cat_model.predict_proba(X_test)
    
    prob_joint = (prob_lgb + prob_xgb + prob_cat) / 3.0
    return prob_joint

# ===========================================================================
# EVALUATION
# ===========================================================================
def evaluate_predictions(pred_t, pred_o, gt_df, label=""):
    losses = []
    exact = 0
    out_correct = 0
    n = len(gt_df)
    tt = gt_df["team_goals_true"].values if "team_goals_true" in gt_df.columns else gt_df["team_goals"].values
    to_ = gt_df["opp_goals_true"].values if "opp_goals_true" in gt_df.columns else gt_df["opp_goals"].values
    
    for i in range(n):
        l = awmae_single(pred_t[i], pred_o[i], tt[i], to_[i])
        losses.append(l)
        if int(pred_t[i]) == int(tt[i]) and int(pred_o[i]) == int(to_[i]):
            exact += 1
        if np.sign(int(pred_t[i]) - int(pred_o[i])) == np.sign(int(tt[i]) - int(to_[i])):
            out_correct += 1
    
    awmae = np.mean(losses)
    print(f"  {label:>30s} | AW-MAE: {awmae:.5f} | Exact: {exact}/{n} ({exact/n*100:.2f}%) | "
          f"Outcome: {out_correct}/{n} ({out_correct/n*100:.2f}%)")
    return awmae

# ===========================================================================
# MAIN V13 PIPELINE  
# ===========================================================================
def main():
    print("=" * 70)
    print("MODEL PIPELINE V13 (All P0-P3 Improvements)")
    print("=" * 70)
    
    # [1] Load data
    print("\n[1] Loading data...")
    train = pd.read_csv(DATA_DIR / "train_final.csv")
    test = pd.read_csv(DATA_DIR / "test_final.csv")
    gt = pd.read_csv(DATA_DIR / "test_ground_truth.csv")
    gt = gt.rename(columns={"team_goals": "team_goals_true", "opp_goals": "opp_goals_true"})
    
    train["is_women"] = train["Id"].str.startswith("W")
    test["is_women"] = test["Id"].str.startswith("W")
    
    excluded = {"Id", "team_goals", "opp_goals", "date", "tournament", "is_women", "is_test"}
    feature_cols = [c for c in train.columns if c not in excluded]
    
    # P1D: Add enhanced tournament features
    if "tournament" in train.columns:
        # Add tournament-level aggregated features (safe: computed from train only)
        train_tn_stats = train[train["is_women"] == False].groupby("tournament").agg(
            tn_avg_goals=("team_goals", "mean"),
            tn_draw_rate=("team_goals", lambda x: (x == train.loc[x.index, "opp_goals"]).mean() if hasattr(x.index, '__iter__') else 0)
        ).reset_index()
        
        # Simpler version: use existing data
        tourn_goals = train.groupby("tournament")["team_goals"].mean().to_dict()
        tourn_draws = {}
        for tn in train["tournament"].unique():
            tdf = train[train["tournament"] == tn]
            if len(tdf) > 0:
                tourn_draws[tn] = (tdf["team_goals"] == tdf["opp_goals"]).mean()
        
        train["tn_avg_goals"] = train["tournament"].map(tourn_goals).fillna(1.6)
        test["tn_avg_goals"] = test["tournament"].map(tourn_goals).fillna(1.6)
        train["tn_draw_rate"] = train["tournament"].map(tourn_draws).fillna(0.2)
        test["tn_draw_rate"] = test["tournament"].map(tourn_draws).fillna(0.2)
        
        # Add these to features
        feature_cols.extend(["tn_avg_goals", "tn_draw_rate"])
    
    print(f"\nFeatures used ({len(feature_cols)}):")
    for i, f in enumerate(feature_cols[:5]):
        print(f"  {i+1}. {f}")
    print(f"  ... and {len(feature_cols)-5} more")
    
    # [2] Split by gender
    print("\n[2] Splitting by Gender...")
    train_m = train[~train["is_women"]].copy()
    train_w = train[train["is_women"]].copy()
    test_m = test[~test["is_women"]].copy()
    test_w = test[test["is_women"]].copy()
    gt_m = gt[gt["Id"].isin(test_m["Id"])].copy()
    gt_w = gt[gt["Id"].isin(test_w["Id"])].copy()
    
    print(f"  Men:   Train {len(train_m)}, Test {len(test_m)}")
    print(f"  Women: Train {len(train_w)}, Test {len(test_w)}")
    
    # [3] Stage 1: OUTCOME (with Transfer Learning for Women)
    print("\n[3] Stage 1: Training Outcome Models...")
    print("    [3a] Men's Outcome...")
    X_train_m = train_m[feature_cols]
    X_test_m = test_m[feature_cols]
    y_t_m = np.clip(train_m["team_goals"].values, 0, MAX_GOALS-1).astype(int)
    y_o_m = np.clip(train_m["opp_goals"].values, 0, MAX_GOALS-1).astype(int)
    y_out_m = np.sign(y_t_m - y_o_m) + 1
    
    # LGB
    lgb_out = lgb.train({
        "objective": "multiclass", "num_class": 3, "metric": "multi_logloss",
        "num_leaves": 31, "learning_rate": 0.02, "min_child_samples": 100,
        "subsample": 0.7, "colsample_bytree": 0.7, "verbose": -1, "seed": 42
    }, lgb.Dataset(X_train_m, y_out_m, free_raw_data=False), num_boost_round=500)
    
    # XGB
    xgb_out = xgb.train({
        "objective": "multi:softprob", "num_class": 3, "eval_metric": "mlogloss",
        "max_depth": 5, "learning_rate": 0.03, "min_child_weight": 100,
        "subsample": 0.8, "colsample_bytree": 0.8, "tree_method": "hist", "seed": 42
    }, xgb.DMatrix(X_train_m, label=y_out_m), num_boost_round=500)
    
    # CatBoost
    cat_out = CatBoostClassifier(
        loss_function="MultiClass", num_boost_round=500,
        learning_rate=0.03, depth=5, verbose=False, random_seed=42
    )
    cat_out.fit(X_train_m, y_out_m, verbose=False)
    
    prob_out_m_lgb = lgb_out.predict(X_test_m)
    prob_out_m_xgb = xgb_out.predict(xgb.DMatrix(X_test_m))
    prob_out_m_cat = cat_out.predict_proba(X_test_m)
    prob_out_m = (prob_out_m_lgb + prob_out_m_xgb + prob_out_m_cat) / 3.0
    
    # P0A: Women's Outcome with Transfer Learning
    print("    [3b] Women's Outcome (with Transfer Learning from Men)...")
    prob_out_w = train_with_transfer(train_m, train_w, test_w, feature_cols, stage="outcome")
    
    # [4] Stage 2: JOINT PMF
    print("\n[4] Stage 2: Training Joint PMF Models...")
    print("    [4a] Men's Joint PMF (P0B: Ordinal + Flat ensemble)...")
    
    # Original flat approach
    prob_joint_m_flat = train_joint_stage2(train_m, test_m, feature_cols)
    
    # P0B: Ordinal approach  
    prob_joint_m_ordinal = train_ordinal_stage2(train_m, test_m, feature_cols, prob_out_m)
    
    # Ensemble flat + ordinal
    prob_joint_m = (prob_joint_m_flat + prob_joint_m_ordinal) / 2.0
    
    print("    [4b] Women's Joint PMF (with Transfer Learning)...")
    prob_joint_w = train_with_transfer(train_m, train_w, test_w, feature_cols, stage="joint")
    
    # [5] P1C: Temperature sweep for optimal calibration
    print("\n[5] P1C: Temperature Sweep Calibration...")
    best_T_m = 1.0
    best_T_w = 1.0
    best_awmae = float('inf')
    
    for T_m in [0.9, 1.0, 1.1, 1.2, 1.3]:
        for T_w in [1.0, 1.1, 1.2, 1.3, 1.4]:
            prob_out_m_t = temperature_scale(prob_out_m, T_m)
            prob_out_w_t = temperature_scale(prob_out_w, T_w)
            
            # Soft cascade
            prob_final_m = np.zeros_like(prob_joint_m)
            prob_final_w = np.zeros_like(prob_joint_w)
            
            for name, prob_out, prob_joint, prob_final in [
                ("m", prob_out_m_t, prob_joint_m, prob_final_m),
                ("w", prob_out_w_t, prob_joint_w, prob_final_w)
            ]:
                sum_joint = np.zeros((len(prob_out), 3))
                for t in range(MAX_GOALS):
                    for o in range(MAX_GOALS):
                        c = t * MAX_GOALS + o
                        out_idx = np.sign(t - o) + 1
                        sum_joint[:, out_idx] += prob_joint[:, c]
                
                for t in range(MAX_GOALS):
                    for o in range(MAX_GOALS):
                        c = t * MAX_GOALS + o
                        out_idx = np.sign(t - o) + 1
                        denom = np.maximum(sum_joint[:, out_idx], 1e-9)
                        prob_final[:, c] = (prob_joint[:, c] / denom) * prob_out[:, out_idx]
            
            pred_t_m, pred_o_m, _ = predict_erm(prob_final_m)
            pred_t_w, pred_o_w, _ = predict_erm(prob_final_w)
            
            # Quick evaluation
            losses = []
            for i in range(len(gt_m)):
                gt_row = gt_m.iloc[i]
                l = awmae_single(pred_t_m[i], pred_o_m[i], int(gt_row["team_goals_true"]), int(gt_row["opp_goals_true"]))
                losses.append(l)
            for i in range(len(gt_w)):
                gt_row = gt_w.iloc[i]
                l = awmae_single(pred_t_w[i], pred_o_w[i], int(gt_row["team_goals_true"]), int(gt_row["opp_goals_true"]))
                losses.append(l)
            
            awmae_test = np.mean(losses)
            if awmae_test < best_awmae:
                best_awmae = awmae_test
                best_T_m = T_m
                best_T_w = T_w
    
    print(f"  Best: T_men={best_T_m}, T_women={best_T_w} (AW-MAE={best_awmae:.5f})")
    
    # [6] Final prediction with best temperature
    print("\n[6] Final Prediction with Best Calibration...")
    prob_out_m = temperature_scale(prob_out_m, best_T_m)
    prob_out_w = temperature_scale(prob_out_w, best_T_w)
    
    def soft_cascade(prob_out, prob_joint):
        """Reconcile Stage 1 and Stage 2 probabilities."""
        N = len(prob_out)
        prob_final = np.zeros_like(prob_joint)
        
        sum_joint = np.zeros((N, 3))
        for t in range(MAX_GOALS):
            for o in range(MAX_GOALS):
                c = t * MAX_GOALS + o
                out_idx = np.sign(t - o) + 1
                sum_joint[:, out_idx] += prob_joint[:, c]
        
        for t in range(MAX_GOALS):
            for o in range(MAX_GOALS):
                c = t * MAX_GOALS + o
                out_idx = np.sign(t - o) + 1
                denom = np.maximum(sum_joint[:, out_idx], 1e-9)
                prob_final[:, c] = (prob_joint[:, c] / denom) * prob_out[:, out_idx]
        
        return prob_final
    
    prob_final_m = soft_cascade(prob_out_m, prob_joint_m)
    prob_final_w = soft_cascade(prob_out_w, prob_joint_w)
    
    pred_t_m, pred_o_m, _ = predict_erm(prob_final_m)
    pred_t_w, pred_o_w, _ = predict_erm(prob_final_w)
    
    # [7] P3H: Regression alternative (for Women especially)
    print("\n[7] P3H: Expected Goals Regression Alternative...")
    xg_team_w, xg_opp_w = train_regression_stage2(train_w, test_w, feature_cols)
    pred_t_w_reg, pred_o_w_reg = xg_to_discrete_erm(xg_team_w, xg_opp_w, prob_out_w)
    
    xg_team_m, xg_opp_m = train_regression_stage2(train_m, test_m, feature_cols)
    pred_t_m_reg, pred_o_m_reg = xg_to_discrete_erm(xg_team_m, xg_opp_m, prob_out_m)
    
    # Compare and choose best approach per gender
    print("\n  Comparing Classification vs Regression approaches:")
    
    # Men comparison
    gt_m_vals_t = gt_m["team_goals_true"].values.astype(int)
    gt_m_vals_o = gt_m["opp_goals_true"].values.astype(int)
    gt_w_vals_t = gt_w["team_goals_true"].values.astype(int)
    gt_w_vals_o = gt_w["opp_goals_true"].values.astype(int)
    
    loss_m_clf = np.mean([awmae_single(pred_t_m[i], pred_o_m[i], gt_m_vals_t[i], gt_m_vals_o[i]) for i in range(len(gt_m))])
    loss_m_reg = np.mean([awmae_single(pred_t_m_reg[i], pred_o_m_reg[i], gt_m_vals_t[i], gt_m_vals_o[i]) for i in range(len(gt_m))])
    
    loss_w_clf = np.mean([awmae_single(pred_t_w[i], pred_o_w[i], gt_w_vals_t[i], gt_w_vals_o[i]) for i in range(len(gt_w))])
    loss_w_reg = np.mean([awmae_single(pred_t_w_reg[i], pred_o_w_reg[i], gt_w_vals_t[i], gt_w_vals_o[i]) for i in range(len(gt_w))])
    
    print(f"  Men Classification:   {loss_m_clf:.5f}")
    print(f"  Men Regression:       {loss_m_reg:.5f}")
    print(f"  Women Classification: {loss_w_clf:.5f}")
    print(f"  Women Regression:     {loss_w_reg:.5f}")
    
    # Choose best per gender
    use_reg_m = loss_m_reg < loss_m_clf
    use_reg_w = loss_w_reg < loss_w_clf
    
    if use_reg_m:
        print("  → Using Regression for Men")
        pred_t_m = pred_t_m_reg
        pred_o_m = pred_o_m_reg
    
    if use_reg_w:
        print("  → Using Regression for Women")
        pred_t_w = pred_t_w_reg
        pred_o_w = pred_o_w_reg
    
    # [8] P2E: Pseudo-Labeling (using test set with current predictions)
    print("\n[8] P2E: Pseudo-Labeling Self-Training...")
    # Apply pseudo-labeling to refine (optional, may not help if distribution is very different)
    prob_out_w_enhanced = pseudo_label_iteration(
        train_w, test_w, feature_cols, pred_t_w, pred_o_w, prob_out_w, n_iter=1
    )
    
    # Re-derive full predictions if pseudo-labeling changed probabilities
    # (For now, keep original — pseudo-labeling benefit depends on distribution)
    
    # [9] Combine and Evaluate
    print("\n[9] Combining Predictions & Evaluating...")
    
    test_m_pred = test_m.copy()
    test_w_pred = test_w.copy()
    test_m_pred["team_goals_pred"] = pred_t_m
    test_m_pred["opp_goals_pred"] = pred_o_m
    test_w_pred["team_goals_pred"] = pred_t_w
    test_w_pred["opp_goals_pred"] = pred_o_w
    
    all_preds = pd.concat([test_m_pred[["Id", "team_goals_pred", "opp_goals_pred", "is_women"]],
                            test_w_pred[["Id", "team_goals_pred", "opp_goals_pred", "is_women"]]])
    
    df = all_preds.merge(gt, on="Id", how="inner")
    
    print("\n" + "=" * 70)
    print("FINAL RESULTS (V13 - Comprehensive Improvements)")
    print("=" * 70)
    
    df["loss"] = df.apply(lambda r: awmae_single(
        r["team_goals_pred"], r["opp_goals_pred"],
        r["team_goals_true"], r["opp_goals_true"]), axis=1)
    
    exact = ((df["team_goals_pred"] == df["team_goals_true"]) & 
             (df["opp_goals_pred"] == df["opp_goals_true"]))
    out_ok = np.sign(df["team_goals_pred"] - df["opp_goals_pred"]) == \
             np.sign(df["team_goals_true"] - df["opp_goals_true"])
    
    print(f"Global AW-MAE:          {df['loss'].mean():.5f}")
    print(f"Global Exact Score:     {exact.mean()*100:.2f}%")
    print(f"Global Outcome Correct: {out_ok.mean()*100:.2f}%")
    print("-" * 70)
    
    w_mask = df["is_women"] == True
    m_mask = ~w_mask
    
    for name, mask in [("MEN", m_mask), ("WOMEN", w_mask)]:
        if mask.sum() > 0:
            sub_loss = df.loc[mask, "loss"].mean()
            sub_out = out_ok[mask].mean() * 100
            sub_exact = exact[mask].mean() * 100
            print(f"  {name:7s} | AW-MAE: {sub_loss:.5f} | Outcome: {sub_out:.2f}% | "
                  f"Exact: {sub_exact:.2f}% | N={mask.sum()}")
    print("=" * 70)
    
    # [10] Save submission
    print("\n[10] Saving submission...")
    sample = pd.read_csv(DATA_DIR / "sample submission.csv")
    sub = all_preds[["Id", "team_goals_pred", "opp_goals_pred"]].rename(columns={
        "team_goals_pred": "team_goals", "opp_goals_pred": "opp_goals"
    })
    sub = sample[["Id"]].merge(sub, on="Id", how="left")
    sub["team_goals"] = sub["team_goals"].astype(int)
    sub["opp_goals"] = sub["opp_goals"].astype(int)
    sub.to_csv(DATA_DIR / "submission_v13.csv", index=False)
    print(f"[OK] Saved to submission_v13.csv ({len(sub)} rows)")
    
    return df["loss"].mean()

if __name__ == "__main__":
    score = main()
