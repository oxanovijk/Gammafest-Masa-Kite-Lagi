"""
Model Pipeline V10 (Ultimate Poisson-Boosting + Dixon-Coles ERM)
================================================================
1. Drops toxic target encoding features.
2. Uses Poisson objective on XGB/LGB to output expected goals (smooth).
3. Applies manual Dixon-Coles correction to boost Draws (0-0, 1-1).
4. Direct ERM to find the AW-MAE optimal score.
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from scipy.stats import poisson
from pathlib import Path
import warnings, time

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "dataset"

MAX_GOALS = 6
NLS_POWER = 1.3

# ===========================================================================
# HYPERPARAMETERS (Heavily Regularized for Poisson)
# ===========================================================================
LGB_PARAMS = {
    "objective": "poisson", "metric": "poisson",
    "num_leaves": 15, "learning_rate": 0.02, "min_child_samples": 200,
    "subsample": 0.6, "colsample_bytree": 0.6, "verbose": -1, "seed": 42
}
XGB_PARAMS = {
    "objective": "count:poisson", "eval_metric": "poisson-nloglik",
    "max_depth": 3, "learning_rate": 0.03, "min_child_weight": 200,
    "subsample": 0.65, "colsample_bytree": 0.65, "tree_method": "hist", "seed": 42
}

N_ESTIMATORS = 500

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
# POISSON + DIXON COLES + ERM
# ===========================================================================
def poisson_predict_erm(lam_t, lam_o, rho=-0.15):
    N = len(lam_t)
    M = MAX_GOALS
    
    # 1. Build independent Poisson PMF
    pmf_t = np.zeros((N, M))
    pmf_o = np.zeros((N, M))
    for k in range(M):
        pmf_t[:, k] = poisson.pmf(k, lam_t)
        pmf_o[:, k] = poisson.pmf(k, lam_o)
        
    # Clip extreme small values
    pmf_t = np.clip(pmf_t, 1e-15, 1.0)
    pmf_o = np.clip(pmf_o, 1e-15, 1.0)
    
    # Make sure they sum to 1.0
    pmf_t /= pmf_t.sum(axis=1, keepdims=True)
    pmf_o /= pmf_o.sum(axis=1, keepdims=True)
    
    # Joint assuming independence
    joint = pmf_t[:, :, None] * pmf_o[:, None, :] # (N, M, M)
    
    # 2. Apply Dixon-Coles Correction
    tau = np.ones_like(joint)
    tau[:, 0, 0] = 1 - lam_t * lam_o * rho
    tau[:, 0, 1] = 1 + lam_t * rho
    tau[:, 1, 0] = 1 + lam_o * rho
    tau[:, 1, 1] = 1 - rho
    
    # Prevent negative probability if rho is too extreme
    tau = np.clip(tau, 0.0, None)
    
    joint = joint * tau
    joint = joint / joint.sum(axis=(1,2), keepdims=True)
    
    # 3. Expected Risk Minimization (ERM)
    expected_loss = np.einsum("abij,nij->nab", loss_tensor, joint)
    
    # Find argmin
    flat_idx = expected_loss.reshape(N, -1).argmin(axis=1)
    pred_team = flat_idx // M
    pred_opp = flat_idx % M
    
    return pred_team, pred_opp

# ===========================================================================
# TRAIN SUB-POPULATION
# ===========================================================================
def train_and_predict_subpop(train_df, test_df, feature_cols, rho):
    if len(train_df) == 0 or len(test_df) == 0:
        return np.array([]), np.array([])
        
    X_train = train_df[feature_cols]
    y_t_tr = train_df["team_goals"].values
    y_o_tr = train_df["opp_goals"].values
    
    X_test = test_df[feature_cols]
    
    # Predict Team Goals (Lambda)
    dt_l_t = lgb.Dataset(X_train, y_t_tr, free_raw_data=False)
    lgb_t = lgb.train(LGB_PARAMS, dt_l_t, num_boost_round=N_ESTIMATORS)
    dt_x_t = xgb.DMatrix(X_train, label=y_t_tr)
    xgb_t = xgb.train(XGB_PARAMS, dt_x_t, num_boost_round=N_ESTIMATORS)
    
    # Predict Opp Goals (Lambda)
    dt_l_o = lgb.Dataset(X_train, y_o_tr, free_raw_data=False)
    lgb_o = lgb.train(LGB_PARAMS, dt_l_o, num_boost_round=N_ESTIMATORS)
    dt_x_o = xgb.DMatrix(X_train, label=y_o_tr)
    xgb_o = xgb.train(XGB_PARAMS, dt_x_o, num_boost_round=N_ESTIMATORS)
    
    lam_t = (lgb_t.predict(X_test) + xgb_t.predict(xgb.DMatrix(X_test))) / 2.0
    lam_o = (lgb_o.predict(X_test) + xgb_o.predict(xgb.DMatrix(X_test))) / 2.0
    
    # Avoid predicting absolute 0 or absurd highs
    lam_t = np.clip(lam_t, 0.05, 5.0)
    lam_o = np.clip(lam_o, 0.05, 5.0)
    
    pred_t, pred_o = poisson_predict_erm(lam_t, lam_o, rho=rho)
    return pred_t, pred_o

# ===========================================================================
# MAIN PIPELINE
# ===========================================================================
def main():
    print("=" * 60)
    print("MODEL PIPELINE V10 (Poisson-Boosting + Dixon-Coles)")
    print("=" * 60)

    print("[1] Loading data...")
    train = pd.read_csv(DATA_DIR / "train_final.csv")
    test  = pd.read_csv(DATA_DIR / "test_final.csv")
    
    train["is_women"] = train["Id"].str.startswith("W")
    test["is_women"] = test["Id"].str.startswith("W")
    
    # DROP TOXIC FEATURES
    toxic_features = {"venue_country_te_ctx", "confederation_team_te_ctx"}
    feature_cols = [c for c in train.columns if c not in {"Id", "team_goals", "opp_goals", "date", "tournament", "is_women", "is_friendly", "is_test"} | toxic_features]
    
    print("[2] Splitting Datasets...")
    train_m = train[~train["is_women"]].copy()
    train_w = train[train["is_women"]].copy()
    
    test_m = test[~test["is_women"]].copy()
    test_w = test[test["is_women"]].copy()
    
    print(f"    Men: Train {len(train_m)}, Test {len(test_m)}")
    print(f"    Women: Train {len(train_w)}, Test {len(test_w)}")
    
    print("\n[3] Training and Predicting MEN's Models (rho=-0.15)...")
    t0 = time.time()
    pred_t_m, pred_o_m = train_and_predict_subpop(train_m, test_m, feature_cols, rho=-0.15)
    print(f"    Men's pipeline done in {time.time()-t0:.1f}s")
    
    print("\n[4] Training and Predicting WOMEN's Models (rho=-0.10)...")
    t0 = time.time()
    pred_t_w, pred_o_w = train_and_predict_subpop(train_w, test_w, feature_cols, rho=-0.10)
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
        print("RESULTS (V10 POISSON-BOOSTING)")
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
    
    # Generate exact format as sample_submission.csv
    sample = pd.read_csv(DATA_DIR / "sample submission.csv")
    sub = final_preds[["Id", "team_goals_pred", "opp_goals_pred"]].rename(columns={
        "team_goals_pred": "team_goals", "opp_goals_pred": "opp_goals"
    })
    sub = sample[["Id"]].merge(sub, on="Id", how="left")
    sub.to_csv(DATA_DIR / "submission_v10.csv", index=False)
    print("Saved to submission_v10.csv")

if __name__ == "__main__":
    main()
