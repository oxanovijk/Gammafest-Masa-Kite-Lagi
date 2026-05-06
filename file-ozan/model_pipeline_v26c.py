"""
V26c — V12 Foundation + Minimal Tier-Adaptive Tweaks
=====================================================
Strategy: Take the PROVEN V12 architecture (LGB+XGB, Gender-Split, Cascade, ERM)
and ONLY add:
  S6: elo_diff_adjusted (1 extra feature)
  S1: Very gentle draw boost for Men Tier 1+4 only
  Keep V12 temperature (Men=1.1, Women=1.2)
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

# ===========================================================================
# LGB + XGB PARAMS (exact V12 settings)
# ===========================================================================
LGB_OUT = {
    "objective": "multiclass", "num_class": 3, "metric": "multi_logloss",
    "num_leaves": 31, "learning_rate": 0.02, "min_child_samples": 100,
    "subsample": 0.8, "colsample_bytree": 0.8, "verbose": -1, "seed": 42
}
XGB_OUT = {
    "objective": "multi:softprob", "num_class": 3, "eval_metric": "mlogloss",
    "max_depth": 5, "learning_rate": 0.03, "min_child_weight": 100,
    "subsample": 0.8, "colsample_bytree": 0.8, "tree_method": "hist", "seed": 42
}
LGB_JOINT = {
    "objective": "multiclass", "num_class": NUM_CLASSES, "metric": "multi_logloss",
    "num_leaves": 31, "learning_rate": 0.02, "min_child_samples": 150,
    "subsample": 0.7, "colsample_bytree": 0.7, "verbose": -1, "seed": 42
}
XGB_JOINT = {
    "objective": "multi:softprob", "num_class": NUM_CLASSES, "eval_metric": "mlogloss",
    "max_depth": 5, "learning_rate": 0.03, "min_child_weight": 150,
    "subsample": 0.75, "colsample_bytree": 0.75, "tree_method": "hist", "seed": 42
}
N_ESTIMATORS = 600

# ===========================================================================
# AW-MAE
# ===========================================================================
def awmae_single(pt, po, tt, to_):
    mae = (abs(int(pt)-int(tt)) + abs(int(po)-int(to_))) / 2.0
    exact = 1 if (int(pt)==int(tt) and int(po)==int(to_)) else 0
    out_ok = 1 if np.sign(int(pt)-int(po)) == np.sign(int(tt)-int(to_)) else 0
    gd_ok = 1 if (int(pt)-int(po)) == (int(tt)-int(to_)) else 0
    aug = mae + 0.30*(1-exact) + 0.25*(1-out_ok) + 0.15*(1-gd_ok)
    mult = 1.0 if out_ok else 1.5
    return (aug * mult) ** NLS_POWER

loss_tensor = np.zeros((MAX_GOALS,MAX_GOALS,MAX_GOALS,MAX_GOALS))
for a in range(MAX_GOALS):
    for b in range(MAX_GOALS):
        for c in range(MAX_GOALS):
            for d in range(MAX_GOALS):
                loss_tensor[a,b,c,d] = awmae_single(a,b,c,d)

# ===========================================================================
# S6: Elo Confidence Discount
# ===========================================================================
ELO_DISCOUNT = {1: 0.78, 2: 0.82, 3: 0.92, 4: 0.87, 5: 0.92}

def add_elo_discount_feature(df):
    df = df.copy()
    df["elo_diff_adjusted"] = df.apply(
        lambda r: r["elo_diff"] * ELO_DISCOUNT.get(int(r["tournament_tier"]), 0.87), axis=1
    )
    return df

# ===========================================================================
# S1: Gentle draw boost (Men Tier 1+4 ONLY)
# ===========================================================================
def apply_draw_boost(prob_out, tiers, is_women_flags):
    prob = prob_out.copy()
    for i in range(len(prob)):
        if is_women_flags[i]:
            continue  # Don't touch Women
        tier = int(tiers[i])
        if tier in (1, 4):
            prob[i, 1] += 0.03  # gentle +3% draw probability
            prob[i] = np.clip(prob[i], 0.01, 0.99)
            prob[i] /= prob[i].sum()
    return prob

# ===========================================================================
# SOFT CASCADE (V12 exact logic)
# ===========================================================================
def soft_cascade(prob_out, prob_joint, T=1.1):
    N = len(prob_out)
    M = MAX_GOALS
    
    # Temperature scaling on joint
    prob_joint = np.clip(prob_joint, 1e-7, 1.0)
    log_p = np.log(prob_joint) / T
    exp_p = np.exp(log_p)
    prob_joint = exp_p / exp_p.sum(axis=1, keepdims=True)
    
    # Bucketed renormalization
    prob_final = np.zeros_like(prob_joint)
    sum_joint = np.zeros((N, 3))
    for t in range(M):
        for o in range(M):
            c = t*M+o
            out_idx = int(np.sign(t-o)) + 1
            sum_joint[:, out_idx] += prob_joint[:, c]
    for t in range(M):
        for o in range(M):
            c = t*M+o
            out_idx = int(np.sign(t-o)) + 1
            denom = np.maximum(sum_joint[:, out_idx], 1e-9)
            prob_final[:, c] = (prob_joint[:, c] / denom) * prob_out[:, out_idx]
    
    return prob_final

def predict_erm(prob_final):
    N = len(prob_final)
    M = MAX_GOALS
    joint = prob_final.reshape(N, M, M)
    joint = np.clip(joint, 1e-8, 1.0)
    joint /= joint.sum(axis=(1,2), keepdims=True)
    expected = np.einsum("abij,nij->nab", loss_tensor, joint)
    idx = expected.reshape(N, -1).argmin(axis=1)
    return idx // M, idx % M

# ===========================================================================
# TRAIN + PREDICT (V12 LGB+XGB ensemble)
# ===========================================================================
def train_and_predict(train_df, test_df, feature_cols, T):
    if len(train_df) == 0 or len(test_df) == 0:
        return np.array([]), np.array([])
    
    X_train = train_df[feature_cols].values
    X_test  = test_df[feature_cols].values
    
    y_t = np.clip(train_df["team_goals"].values, 0, MAX_GOALS-1).astype(int)
    y_o = np.clip(train_df["opp_goals"].values, 0, MAX_GOALS-1).astype(int)
    y_out   = (np.sign(y_t - y_o) + 1).astype(int)
    y_joint = y_t * MAX_GOALS + y_o
    
    # Stage 1: Outcome (LGB + XGB)
    dt_out = lgb.Dataset(X_train, y_out, free_raw_data=False)
    lgb_out = lgb.train(LGB_OUT, dt_out, num_boost_round=N_ESTIMATORS)
    
    dx_out = xgb.DMatrix(X_train, label=y_out)
    xgb_out = xgb.train(XGB_OUT, dx_out, num_boost_round=N_ESTIMATORS)
    
    prob_out = (lgb_out.predict(X_test) + xgb_out.predict(xgb.DMatrix(X_test))) / 2.0
    
    # S1: Apply draw boost BEFORE cascade
    tiers = test_df["tournament_tier"].values
    is_women = test_df["is_women"].values
    prob_out = apply_draw_boost(prob_out, tiers, is_women)
    
    # Stage 2: Joint PMF (LGB + XGB)
    dt_joint = lgb.Dataset(X_train, y_joint, free_raw_data=False)
    lgb_joint = lgb.train(LGB_JOINT, dt_joint, num_boost_round=N_ESTIMATORS)
    
    dx_joint = xgb.DMatrix(X_train, label=y_joint)
    xgb_joint = xgb.train(XGB_JOINT, dx_joint, num_boost_round=N_ESTIMATORS)
    
    prob_joint = (lgb_joint.predict(X_test) + xgb_joint.predict(xgb.DMatrix(X_test))) / 2.0
    
    # Cascade + ERM
    prob_cascaded = soft_cascade(prob_out, prob_joint, T=T)
    pred_t, pred_o = predict_erm(prob_cascaded)
    
    return pred_t, pred_o

# ===========================================================================
# MAIN
# ===========================================================================
def main():
    print("=" * 60)
    print("V26c — V12 + Minimal Tier-Adaptive (S1+S6)")
    print("=" * 60)
    
    train = pd.read_csv(DATA_DIR / "train_final.csv")
    test  = pd.read_csv(DATA_DIR / "test_final.csv")
    
    # S6: Add elo_diff_adjusted
    train = add_elo_discount_feature(train)
    test  = add_elo_discount_feature(test)
    
    train["is_women"] = train["Id"].str.startswith("W")
    test["is_women"]  = test["Id"].str.startswith("W")
    
    excluded = {"Id", "team_goals", "opp_goals", "is_women", "is_test"}
    feature_cols = [c for c in train.columns if c not in excluded]
    print(f"Features: {len(feature_cols)}")
    
    train_m = train[~train["is_women"]].copy()
    train_w = train[train["is_women"]].copy()
    test_m  = test[~test["is_women"]].copy()
    test_w  = test[test["is_women"]].copy()
    print(f"Men: train={len(train_m)}, test={len(test_m)}")
    print(f"Women: train={len(train_w)}, test={len(test_w)}")
    
    t0 = time.time()
    print("\nTraining MEN (T=1.1)...")
    pred_t_m, pred_o_m = train_and_predict(train_m, test_m, feature_cols, T=1.1)
    print(f"  Done in {time.time()-t0:.1f}s")
    
    t0 = time.time()
    print("Training WOMEN (T=1.2)...")
    pred_t_w, pred_o_w = train_and_predict(train_w, test_w, feature_cols, T=1.2)
    print(f"  Done in {time.time()-t0:.1f}s")
    
    # Combine
    test_m_out = test_m[["Id"]].copy()
    test_m_out["team_goals"] = pred_t_m.astype(int)
    test_m_out["opp_goals"]  = pred_o_m.astype(int)
    
    test_w_out = test_w[["Id"]].copy()
    test_w_out["team_goals"] = pred_t_w.astype(int)
    test_w_out["opp_goals"]  = pred_o_w.astype(int)
    
    final = pd.concat([test_m_out, test_w_out])
    
    gt_path = DATA_DIR / "test_ground_truth.csv"
    if gt_path.exists():
        gt = pd.read_csv(gt_path).rename(columns={"team_goals":"team_goals_true","opp_goals":"opp_goals_true"})
        df = final.merge(gt, on="Id", how="inner")
        
        df["loss"] = df.apply(lambda r: awmae_single(r["team_goals"], r["opp_goals"], r["team_goals_true"], r["opp_goals_true"]), axis=1)
        exact = (df["team_goals"]==df["team_goals_true"]) & (df["opp_goals"]==df["opp_goals_true"])
        out_ok = np.sign(df["team_goals"]-df["opp_goals"]) == np.sign(df["team_goals_true"]-df["opp_goals_true"])
        
        print("\n" + "="*60)
        print("RESULTS (V26c = V12 + S1 + S6)")
        print("="*60)
        print(f"Global AW-MAE:          {df['loss'].mean():.5f}")
        print(f"Global Exact Score:     {exact.mean()*100:.2f}%")
        print(f"Global Outcome Correct: {out_ok.mean()*100:.2f}%")
        print("-"*60)
        
        df["is_women"] = df["Id"].str.startswith("W")
        for name, mask in [("MEN", ~df["is_women"]), ("WOMEN", df["is_women"])]:
            if mask.sum() > 0:
                sl = df.loc[mask,"loss"].mean()
                se = exact[mask].mean()*100
                so = out_ok[mask].mean()*100
                print(f"  {name:7s} | AW-MAE: {sl:.5f} | Exact: {se:.1f}% | Outcome: {so:.1f}% | N={mask.sum()}")
        
        print("-"*60)
        df2 = df.merge(test[["Id","tournament_tier"]], on="Id", how="left")
        for tier in sorted(df2["tournament_tier"].unique(), reverse=True):
            sub = df2[df2["tournament_tier"]==tier]
            sl = sub["loss"].mean()
            se = ((sub["team_goals"]==sub["team_goals_true"])&(sub["opp_goals"]==sub["opp_goals_true"])).mean()*100
            so = (np.sign(sub["team_goals"]-sub["opp_goals"])==np.sign(sub["team_goals_true"]-sub["opp_goals_true"])).mean()*100
            print(f"  Tier {int(tier)} | AW-MAE: {sl:.5f} | Exact: {se:.1f}% | Outcome: {so:.1f}% | N={len(sub)}")
        print("="*60)
        
        print("\nPrediction Distribution (top 15):")
        print(df.groupby(["team_goals","opp_goals"]).size().sort_values(ascending=False).head(15))
    
    sample = pd.read_csv(DATA_DIR / "sample submission.csv")
    sub = sample[["Id"]].merge(final, on="Id", how="left")
    out_path = DATA_DIR / "submission_v26c.csv"
    sub.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path} ({len(sub)} rows)")

if __name__ == "__main__":
    main()
