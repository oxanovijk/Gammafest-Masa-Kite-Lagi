"""
Model Pipeline V11 (36-Class Joint PMF)
=======================================
Directly predicts the Exact Score Joint PMF (36 classes) to natively capture
all goal dependencies, correlation, and match dynamics.
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
NUM_CLASSES = MAX_GOALS * MAX_GOALS
NLS_POWER = 1.3
TEMPERATURE = 1.1

# ===========================================================================
# HYPERPARAMETERS
# ===========================================================================
LGB_PARAMS = {
    "objective": "multiclass", "num_class": NUM_CLASSES, "metric": "multi_logloss",
    "num_leaves": 31, "learning_rate": 0.02, "min_child_samples": 150,
    "subsample": 0.7, "colsample_bytree": 0.7, "verbose": -1, "seed": 42
}
XGB_PARAMS = {
    "objective": "multi:softprob", "num_class": NUM_CLASSES, "eval_metric": "mlogloss",
    "max_depth": 5, "learning_rate": 0.03, "min_child_weight": 150,
    "subsample": 0.75, "colsample_bytree": 0.75, "tree_method": "hist", "seed": 42
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
# JOINT PMF ERM
# ===========================================================================
def joint_pmf_predict_erm(prob_36, T=1.1):
    N = len(prob_36)
    M = MAX_GOALS
    
    # Apply Temperature Scaling
    prob_36 = np.clip(prob_36, 1e-7, 1.0)
    log_p = np.log(prob_36) / T
    exp_p = np.exp(log_p)
    prob_36 = exp_p / exp_p.sum(axis=1, keepdims=True)
    
    # Reshape back to (N, 6, 6)
    joint = prob_36.reshape(N, M, M)
    
    # Expected Risk Minimization (ERM)
    expected_loss = np.einsum("abij,nij->nab", loss_tensor, joint)
    
    flat_idx = expected_loss.reshape(N, -1).argmin(axis=1)
    pred_team = flat_idx // M
    pred_opp = flat_idx % M
    
    return pred_team, pred_opp

# ===========================================================================
# TRAIN SUB-POPULATION
# ===========================================================================
def train_and_predict_subpop(train_df, test_df, feature_cols, T):
    if len(train_df) == 0 or len(test_df) == 0:
        return np.array([]), np.array([])
        
    X_train = train_df[feature_cols]
    
    # Create 36-class target
    y_t = np.clip(train_df["team_goals"].values, 0, MAX_GOALS-1).astype(int)
    y_o = np.clip(train_df["opp_goals"].values, 0, MAX_GOALS-1).astype(int)
    y_class = y_t * MAX_GOALS + y_o
    
    X_test = test_df[feature_cols]
    
    dt_l = lgb.Dataset(X_train, y_class, free_raw_data=False)
    lgb_model = lgb.train(LGB_PARAMS, dt_l, num_boost_round=N_ESTIMATORS)
    
    dt_x = xgb.DMatrix(X_train, label=y_class)
    xgb_model = xgb.train(XGB_PARAMS, dt_x, num_boost_round=N_ESTIMATORS)
    
    prob_36 = (lgb_model.predict(X_test) + xgb_model.predict(xgb.DMatrix(X_test))) / 2.0
    
    pred_t, pred_o = joint_pmf_predict_erm(prob_36, T=T)
    return pred_t, pred_o

# ===========================================================================
# MAIN PIPELINE
# ===========================================================================
def main():
    print("=" * 60)
    print("MODEL PIPELINE V11 (36-Class Joint PMF + Gender-Split)")
    print("=" * 60)

    print("[1] Loading data...")
    train = pd.read_csv(DATA_DIR / "train_final.csv")
    test  = pd.read_csv(DATA_DIR / "test_final.csv")
    
    train["is_women"] = train["Id"].str.startswith("W")
    test["is_women"] = test["Id"].str.startswith("W")
    
    feature_cols = [c for c in train.columns if c not in {"Id", "team_goals", "opp_goals", "date", "tournament", "is_women", "is_friendly", "is_test"}]
    
    print("[2] Splitting Datasets...")
    train_m = train[~train["is_women"]].copy()
    train_w = train[train["is_women"]].copy()
    
    test_m = test[~test["is_women"]].copy()
    test_w = test[test["is_women"]].copy()
    
    print(f"    Men: Train {len(train_m)}, Test {len(test_m)}")
    print(f"    Women: Train {len(train_w)}, Test {len(test_w)}")
    
    print("\n[3] Training and Predicting MEN's Models (T=1.1)...")
    t0 = time.time()
    pred_t_m, pred_o_m = train_and_predict_subpop(train_m, test_m, feature_cols, T=1.1)
    print(f"    Men's pipeline done in {time.time()-t0:.1f}s")
    
    print("\n[4] Training and Predicting WOMEN's Models (T=1.2)...")
    t0 = time.time()
    pred_t_w, pred_o_w = train_and_predict_subpop(train_w, test_w, feature_cols, T=1.2)
    print(f"    Women's pipeline done in {time.time()-t0:.1f}s")
    
    print("\n[5] Reconciling Submissions...")
    test_m["team_goals_pred"] = pred_t_m
    test_m["opp_goals_pred"] = pred_o_m
    
    test_w["team_goals_pred"] = pred_t_w
    test_w["opp_goals_pred"] = pred_o_w
    
    final_preds = pd.concat([test_m, test_w])
    
    gt_path = DATA_DIR / "test_ground_truth.csv"
    if gt_path.exists():
        gt = pd.read_csv(gt_path)
        gt.rename(columns={"team_goals": "team_goals_true", "opp_goals": "opp_goals_true"}, inplace=True)
        
        df = final_preds[["Id", "team_goals_pred", "opp_goals_pred", "is_women"]].merge(gt, on="Id", how="inner")
        
        df["loss"] = df.apply(lambda r: awmae_single(
            r["team_goals_pred"], r["opp_goals_pred"],
            r["team_goals_true"], r["opp_goals_true"]), axis=1)
            
        exact = ((df["team_goals_pred"]==df["team_goals_true"]) & 
                 (df["opp_goals_pred"]==df["opp_goals_true"]))
        out_ok = np.sign(df["team_goals_pred"]-df["opp_goals_pred"]) == \
                 np.sign(df["team_goals_true"]-df["opp_goals_true"])
                 
        print("=" * 60)
        print("RESULTS (V11 36-CLASS JOINT PMF)")
        print("=" * 60)
        print(f"Global AW-MAE:          {df['loss'].mean():.5f}")
        print(f"Global Exact Score:     {exact.mean()*100:.2f}%")
        print(f"Global Outcome Correct: {out_ok.mean()*100:.2f}%")
        print("-" * 60)
        
        # Subgroup analysis
        w_mask = df["is_women"] == True
        m_mask = ~w_mask
        
        for name, mask in [("MEN", m_mask), ("WOMEN", w_mask)]:
            if mask.sum() > 0:
                sub_loss = df.loc[mask, "loss"].mean()
                sub_out = out_ok[mask].mean() * 100
                print(f"  {name:7s} | AW-MAE: {sub_loss:.5f} | Outcome: {sub_out:.2f}% | N={mask.sum()}")
        print("=" * 60)
        
        # Distribution check
        print("\nPrediction Distribution:")
        print(df.groupby(["team_goals_pred", "opp_goals_pred"]).size())
    
    sample = pd.read_csv(DATA_DIR / "sample submission.csv")
    sub = final_preds[["Id", "team_goals_pred", "opp_goals_pred"]].rename(columns={
        "team_goals_pred": "team_goals", "opp_goals_pred": "opp_goals"
    })
    sub = sample[["Id"]].merge(sub, on="Id", how="left")
    sub.to_csv(DATA_DIR / "submission_v11.csv", index=False)
    print("Saved to submission_v11.csv")

if __name__ == "__main__":
    main()
