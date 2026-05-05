"""
Model Pipeline V8 (Two-Stage Objective + Robustness)
======================================================
Strategy C from LLM: 
1. Predict Outcome (Win/Draw/Loss) first.
2. Predict Score PMF.
3. Decision Rule: Select the Exact Score that minimizes AW-MAE ONLY WITHIN the predicted outcome bucket.
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from pathlib import Path
import warnings, time

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "dataset"

MAX_GOALS = 6
NLS_POWER = 1.3
TEMPERATURE = 1.2  # Temperature scaling for PMF

# ===========================================================================
# HYPERPARAMETERS (Conservative / Reduced Complexity)
# ===========================================================================
LGB_PARAMS_OUTCOME = {
    "objective": "multiclass", "num_class": 3, "metric": "multi_logloss",
    "num_leaves": 24, "learning_rate": 0.02, "min_child_samples": 200,
    "subsample": 0.6, "colsample_bytree": 0.6, "verbose": -1, "seed": 42
}
XGB_PARAMS_OUTCOME = {
    "objective": "multi:softprob", "num_class": 3, "eval_metric": "mlogloss",
    "max_depth": 4, "learning_rate": 0.03, "min_child_weight": 200,
    "subsample": 0.65, "colsample_bytree": 0.65, "tree_method": "hist", "seed": 42
}

LGB_PARAMS_SCORE = {
    "objective": "multiclass", "num_class": MAX_GOALS, "metric": "multi_logloss",
    "num_leaves": 24, "learning_rate": 0.02, "min_child_samples": 200,
    "subsample": 0.6, "colsample_bytree": 0.6, "verbose": -1, "seed": 42
}
XGB_PARAMS_SCORE = {
    "objective": "multi:softprob", "num_class": MAX_GOALS, "eval_metric": "mlogloss",
    "max_depth": 4, "learning_rate": 0.03, "min_child_weight": 200,
    "subsample": 0.65, "colsample_bytree": 0.65, "tree_method": "hist", "seed": 42
}

N_ESTIMATORS = 600

# ===========================================================================
# AW-MAE METRIC
# ===========================================================================
def awmae_single(pt, po, tt, to_):
    mae = (abs(pt - tt) + abs(po - to_)) / 2.0
    exact = 1 if (pt == tt and po == to_) else 0
    out_ok = 1 if np.sign(pt - po) == np.sign(tt - to_) else 0
    gd_ok = 1 if (pt - po) == (tt - to_) else 0
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
# TWO-STAGE DECISION RULE
# ===========================================================================
def apply_temperature(pmf, T=1.0):
    """Calibrate PMF by softening extreme probabilities."""
    pmf = np.clip(pmf, 1e-7, 1.0)
    log_pmf = np.log(pmf) / T
    exp_pmf = np.exp(log_pmf)
    return exp_pmf / exp_pmf.sum(axis=1, keepdims=True)

def two_stage_predict(pmf_t, pmf_o, prob_outcome, loss_tensor, T=1.2):
    """
    pmf_t, pmf_o: (N, 6)
    prob_outcome: (N, 3) where 0=Loss, 1=Draw, 2=Win
    """
    N = len(pmf_t)
    M = MAX_GOALS
    
    # Calibrate PMF
    pmf_t = apply_temperature(pmf_t, T)
    pmf_o = apply_temperature(pmf_o, T)
    
    # Joint Probability Matrix
    prob_joint = pmf_t[:, :, None] * pmf_o[:, None, :] # (N, 6, 6)
    
    # Calculate Expected AW-MAE for all possible predictions
    # expected_loss shape: (N, 6, 6)
    expected_loss = np.einsum("abij,nij->nab", loss_tensor, prob_joint)
    
    # Predicted Outcome based on Argmax
    pred_outcome = np.argmax(prob_outcome, axis=1) # 0, 1, or 2
    
    pred_team = np.zeros(N, dtype=int)
    pred_opp = np.zeros(N, dtype=int)
    
    for i in range(N):
        o = pred_outcome[i]
        
        # Create a mask of valid scores that MATCH the predicted outcome
        mask = np.zeros((M, M), dtype=bool)
        for a in range(M):
            for b in range(M):
                sign = np.sign(a - b)
                # target_outcome = np.sign(team_goals - opp_goals) + 1
                # so: -1 -> 0 (Loss), 0 -> 1 (Draw), 1 -> 2 (Win)
                if sign == -1 and o == 0: mask[a, b] = True
                elif sign == 0 and o == 1: mask[a, b] = True
                elif sign == 1 and o == 2: mask[a, b] = True
                
        # Find the minimum expected loss ONLY within the allowed mask
        masked_loss = np.where(mask, expected_loss[i], np.inf)
        flat_idx = np.argmin(masked_loss)
        pred_team[i] = flat_idx // M
        pred_opp[i] = flat_idx % M
        
    return pred_team, pred_opp

# ===========================================================================
# MAIN PIPELINE
# ===========================================================================
def main():
    print("=" * 60)
    print("MODEL PIPELINE V8 (Two-Stage Objective)")
    print("=" * 60)

    print("[1] Loading data...")
    train = pd.read_csv(DATA_DIR / "train_final.csv")
    test  = pd.read_csv(DATA_DIR / "test_final.csv")
    raw_train = pd.read_csv(DATA_DIR / "train.csv")
    raw_test = pd.read_csv(DATA_DIR / "test.csv")
    
    train = train.merge(raw_train[["Id", "tournament"]], on="Id", how="left")
    test  = test.merge(raw_test[["Id", "tournament"]], on="Id", how="left")
    
    # Prepare Outcome Target
    # np.sign(team - opp) -> -1, 0, 1
    # + 1 -> 0, 1, 2 (0=Loss, 1=Draw, 2=Win)
    train["outcome"] = np.sign(train["team_goals"] - train["opp_goals"]) + 1
    
    # Prepare subsets for testing
    train["is_women"] = train["Id"].str.startswith("W")
    test["is_women"] = test["Id"].str.startswith("W")
    
    train["is_friendly"] = train["tournament"].str.contains("Friendly", case=False, na=False)
    test["is_friendly"] = test["tournament"].str.contains("Friendly", case=False, na=False)
    
    feature_cols = [c for c in train.columns if c not in {"Id", "team_goals", "opp_goals", "date", "tournament", "outcome", "is_women", "is_friendly", "is_test"}]
    
    X_train = train[feature_cols]
    y_out_tr = train["outcome"].values.astype(int)
    y_t_tr = np.clip(train["team_goals"].values, 0, MAX_GOALS-1).astype(int)
    y_o_tr = np.clip(train["opp_goals"].values, 0, MAX_GOALS-1).astype(int)
    
    X_test = test[feature_cols]
    
    print("\n[2] Training Stage 1: Outcome Classifier...")
    t0 = time.time()
    dt_l_out = lgb.Dataset(X_train, y_out_tr, free_raw_data=False)
    lgb_out = lgb.train(LGB_PARAMS_OUTCOME, dt_l_out, num_boost_round=N_ESTIMATORS)
    dt_x_out = xgb.DMatrix(X_train, label=y_out_tr)
    xgb_out = xgb.train(XGB_PARAMS_OUTCOME, dt_x_out, num_boost_round=N_ESTIMATORS)
    
    prob_out_test = (lgb_out.predict(X_test) + xgb_out.predict(xgb.DMatrix(X_test))) / 2.0
    print(f"    Done in {time.time()-t0:.1f}s")
    
    print("\n[3] Training Stage 2: PMF Score Model...")
    t0 = time.time()
    dt_l_t = lgb.Dataset(X_train, y_t_tr, free_raw_data=False)
    lgb_t = lgb.train(LGB_PARAMS_SCORE, dt_l_t, num_boost_round=N_ESTIMATORS)
    dt_x_t = xgb.DMatrix(X_train, label=y_t_tr)
    xgb_t = xgb.train(XGB_PARAMS_SCORE, dt_x_t, num_boost_round=N_ESTIMATORS)
    
    dt_l_o = lgb.Dataset(X_train, y_o_tr, free_raw_data=False)
    lgb_o = lgb.train(LGB_PARAMS_SCORE, dt_l_o, num_boost_round=N_ESTIMATORS)
    dt_x_o = xgb.DMatrix(X_train, label=y_o_tr)
    xgb_o = xgb.train(XGB_PARAMS_SCORE, dt_x_o, num_boost_round=N_ESTIMATORS)
    
    pmf_t_test = (lgb_t.predict(X_test) + xgb_t.predict(xgb.DMatrix(X_test))) / 2.0
    pmf_o_test = (lgb_o.predict(X_test) + xgb_o.predict(xgb.DMatrix(X_test))) / 2.0
    print(f"    Done in {time.time()-t0:.1f}s")
    
    print("\n[4] Applying Two-Stage Decision Rule (Temperature=1.2)...")
    pred_t, pred_o = two_stage_predict(pmf_t_test, pmf_o_test, prob_out_test, loss_tensor, T=TEMPERATURE)
    
    print("\n[5] Evaluating against Ground Truth...")
    gt_path = DATA_DIR / "test_ground_truth.csv"
    if gt_path.exists():
        gt = pd.read_csv(gt_path)
        gt.rename(columns={"team_goals": "team_goals_true", "opp_goals": "opp_goals_true"}, inplace=True)
        sub = pd.DataFrame({"Id": test["Id"], "team_goals_pred": pred_t, "opp_goals_pred": pred_o})
        df = sub.merge(gt, on="Id", how="inner")
        
        # Add metadata for sub-group evaluation
        df = df.merge(test[["Id", "is_women", "is_friendly"]], on="Id", how="left")
        
        df["loss"] = df.apply(lambda r: awmae_single(
            r["team_goals_pred"], r["opp_goals_pred"],
            r["team_goals_true"], r["opp_goals_true"]), axis=1)
            
        exact = ((df["team_goals_pred"]==df["team_goals_true"]) & 
                 (df["opp_goals_pred"]==df["opp_goals_true"]))
        out_ok = np.sign(df["team_goals_pred"]-df["opp_goals_pred"]) == \
                 np.sign(df["team_goals_true"]-df["opp_goals_true"])
                 
        print("=" * 60)
        print("RESULTS (TWO-STAGE OBJECTIVE)")
        print("=" * 60)
        print(f"Global AW-MAE:          {df['loss'].mean():.5f}")
        print(f"Global Exact Score:     {exact.mean()*100:.2f}%")
        print(f"Global Outcome Correct: {out_ok.mean()*100:.2f}%")
        print("-" * 60)
        
        # Subgroup analysis
        w_mask = df["is_women"] == True
        m_mask = ~w_mask
        f_mask = df["is_friendly"] == True
        nf_mask = ~f_mask
        
        for name, mask in [("MEN", m_mask), ("WOMEN", w_mask), ("FRIENDLY", f_mask), ("COMPETITIVE", nf_mask)]:
            if mask.sum() > 0:
                sub_loss = df.loc[mask, "loss"].mean()
                sub_out = out_ok[mask].mean() * 100
                print(f"  {name:11s} | AW-MAE: {sub_loss:.5f} | Outcome: {sub_out:.2f}% | N={mask.sum()}")
        print("=" * 60)
    
    sub_final = pd.DataFrame({"Id": test["Id"], "team_goals": pred_t, "opp_goals": pred_o})
    sub_final.to_csv(DATA_DIR / "submission_v8.csv", index=False)
    print("Saved to submission_v8.csv")

if __name__ == "__main__":
    main()
