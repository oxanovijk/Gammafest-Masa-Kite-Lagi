"""
V27 — Hybrid Hard Cascade (dari analisa_hasil_v26.md)
=====================================================
Akar masalah: ERM selalu memilih 2-1/1-2 karena menghitung expected loss 
di seluruh 36 skor. Draw (0-0, 1-1) kalah karena P(Draw) < P(Win/Loss).

Solusi: Hybrid Cascade
  - Confident (max P(outcome) > threshold): Hard Cascade
    → Putuskan outcome dulu, lalu ERM HANYA dalam bucket itu
  - Not confident: Soft Cascade (V12 fallback)
    
Base: V12 architecture (LGB+XGB, Gender-Split, proven best)
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

# Confidence threshold for hard cascade
HARD_CASCADE_THRESHOLD = 0.45

# ===========================================================================
# HYPERPARAMS (V12 exact)
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
# AW-MAE + LOSS TENSOR
# ===========================================================================
def awmae_single(pt, po, tt, to_):
    mae = (abs(int(pt)-int(tt)) + abs(int(po)-int(to_))) / 2.0
    exact = 1 if (int(pt)==int(tt) and int(po)==int(to_)) else 0
    out_ok = 1 if np.sign(int(pt)-int(po)) == np.sign(int(tt)-int(to_)) else 0
    gd_ok = 1 if (int(pt)-int(po)) == (int(tt)-int(to_)) else 0
    aug = mae + 0.30*(1-exact) + 0.25*(1-out_ok) + 0.15*(1-gd_ok)
    mult = 1.0 if out_ok else 1.5
    return (aug * mult) ** NLS_POWER

M = MAX_GOALS
loss_tensor = np.zeros((M, M, M, M))
for a in range(M):
    for b in range(M):
        for c in range(M):
            for d in range(M):
                loss_tensor[a,b,c,d] = awmae_single(a,b,c,d)

# Pre-compute outcome bucket masks for ERM
# outcome_idx: 0=Loss (team<opp), 1=Draw (team==opp), 2=Win (team>opp)
BUCKET_MASKS = {}
for out_idx in range(3):
    mask = np.zeros((M, M), dtype=bool)
    for t in range(M):
        for o in range(M):
            if int(np.sign(t - o)) + 1 == out_idx:
                mask[t, o] = True
    BUCKET_MASKS[out_idx] = mask

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
# HYBRID CASCADE: CORE LOGIC
# ===========================================================================
def hybrid_cascade_predict(prob_out, prob_joint, T=1.1, threshold=HARD_CASCADE_THRESHOLD):
    """
    For each match:
      if max(prob_out) > threshold → Hard Cascade (ERM within decided bucket)
      else → Soft Cascade (ERM across all 36, V12-style)
    """
    N = len(prob_out)
    pred_t = np.zeros(N, dtype=int)
    pred_o = np.zeros(N, dtype=int)
    
    # Temperature scaling on joint (same as V12)
    prob_joint_t = np.clip(prob_joint, 1e-7, 1.0)
    log_p = np.log(prob_joint_t) / T
    exp_p = np.exp(log_p)
    prob_joint_t = exp_p / exp_p.sum(axis=1, keepdims=True)
    
    # Pre-compute soft cascade result for fallback
    prob_soft = _soft_cascade(prob_out, prob_joint_t)
    
    n_hard = 0
    n_soft = 0
    
    for i in range(N):
        max_out_prob = prob_out[i].max()
        
        if max_out_prob > threshold:
            # === HARD CASCADE ===
            decided_outcome = prob_out[i].argmax()  # 0=Loss, 1=Draw, 2=Win
            
            # Get joint PMF for this match, masked to decided bucket
            joint_i = prob_joint_t[i].reshape(M, M)
            bucket_mask = BUCKET_MASKS[decided_outcome]
            
            # Zero out scores outside the decided outcome bucket
            joint_masked = joint_i * bucket_mask
            total = joint_masked.sum()
            if total > 1e-9:
                joint_masked /= total  # renormalize within bucket
            else:
                # Fallback: uniform within bucket
                joint_masked = bucket_mask.astype(float)
                joint_masked /= joint_masked.sum()
            
            # ERM within bucket
            expected_loss = np.zeros((M, M))
            for a in range(M):
                for b in range(M):
                    if bucket_mask[a, b]:
                        expected_loss[a, b] = (loss_tensor[a, b] * joint_masked).sum()
                    else:
                        expected_loss[a, b] = 1e9  # exclude
            
            best_idx = np.unravel_index(expected_loss.argmin(), (M, M))
            pred_t[i] = best_idx[0]
            pred_o[i] = best_idx[1]
            n_hard += 1
        else:
            # === SOFT CASCADE FALLBACK (V12) ===
            joint_i = prob_soft[i].reshape(M, M)
            joint_i = np.clip(joint_i, 1e-8, 1.0)
            joint_i /= joint_i.sum()
            
            expected_loss = np.zeros((M, M))
            for a in range(M):
                for b in range(M):
                    expected_loss[a, b] = (loss_tensor[a, b] * joint_i).sum()
            
            best_idx = np.unravel_index(expected_loss.argmin(), (M, M))
            pred_t[i] = best_idx[0]
            pred_o[i] = best_idx[1]
            n_soft += 1
    
    print(f"    Hard Cascade: {n_hard} ({n_hard/N*100:.1f}%)")
    print(f"    Soft Cascade: {n_soft} ({n_soft/N*100:.1f}%)")
    
    return pred_t, pred_o

def _soft_cascade(prob_out, prob_joint):
    """Standard V12 soft cascade (bucketed renormalization)"""
    N = len(prob_out)
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

# ===========================================================================
# TRAIN + PREDICT
# ===========================================================================
def train_and_predict(train_df, test_df, feature_cols, T, threshold):
    if len(test_df) == 0:
        return np.array([]), np.array([])
    
    X_train = train_df[feature_cols].values
    X_test  = test_df[feature_cols].values
    
    y_t = np.clip(train_df["team_goals"].values, 0, M-1).astype(int)
    y_o = np.clip(train_df["opp_goals"].values, 0, M-1).astype(int)
    y_out   = (np.sign(y_t - y_o) + 1).astype(int)
    y_joint = y_t * M + y_o
    
    # Stage 1: Outcome (LGB + XGB ensemble)
    t0 = time.time()
    dt_out = lgb.Dataset(X_train, y_out, free_raw_data=False)
    lgb_out = lgb.train(LGB_OUT, dt_out, num_boost_round=N_ESTIMATORS)
    dx_out = xgb.DMatrix(X_train, label=y_out)
    xgb_out = xgb.train(XGB_OUT, dx_out, num_boost_round=N_ESTIMATORS)
    prob_out = (lgb_out.predict(X_test) + xgb_out.predict(xgb.DMatrix(X_test))) / 2.0
    print(f"  Stage 1 done in {time.time()-t0:.1f}s")
    
    # Stage 2: Joint PMF (LGB + XGB ensemble)
    t0 = time.time()
    dt_joint = lgb.Dataset(X_train, y_joint, free_raw_data=False)
    lgb_joint = lgb.train(LGB_JOINT, dt_joint, num_boost_round=N_ESTIMATORS)
    dx_joint = xgb.DMatrix(X_train, label=y_joint)
    xgb_joint = xgb.train(XGB_JOINT, dx_joint, num_boost_round=N_ESTIMATORS)
    prob_joint = (lgb_joint.predict(X_test) + xgb_joint.predict(xgb.DMatrix(X_test))) / 2.0
    print(f"  Stage 2 done in {time.time()-t0:.1f}s")
    
    # Hybrid Cascade Decision
    pred_t, pred_o = hybrid_cascade_predict(prob_out, prob_joint, T=T, threshold=threshold)
    
    return pred_t, pred_o

# ===========================================================================
# MAIN
# ===========================================================================
def main():
    print("=" * 60)
    print(f"V27 — HYBRID HARD CASCADE (threshold={HARD_CASCADE_THRESHOLD})")
    print("=" * 60)
    
    train = pd.read_csv(DATA_DIR / "train_final.csv")
    test  = pd.read_csv(DATA_DIR / "test_final.csv")
    
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
    
    print("\n--- MEN (T=1.1) ---")
    pred_t_m, pred_o_m = train_and_predict(train_m, test_m, feature_cols, T=1.1, threshold=HARD_CASCADE_THRESHOLD)
    
    print("\n--- WOMEN (T=1.2) ---")
    pred_t_w, pred_o_w = train_and_predict(train_w, test_w, feature_cols, T=1.2, threshold=HARD_CASCADE_THRESHOLD)
    
    # Combine
    test_m_out = test_m[["Id"]].copy()
    test_m_out["team_goals"] = pred_t_m.astype(int)
    test_m_out["opp_goals"]  = pred_o_m.astype(int)
    test_w_out = test_w[["Id"]].copy()
    test_w_out["team_goals"] = pred_t_w.astype(int)
    test_w_out["opp_goals"]  = pred_o_w.astype(int)
    final = pd.concat([test_m_out, test_w_out])
    
    # Evaluate
    gt_path = DATA_DIR / "test_ground_truth.csv"
    if gt_path.exists():
        gt = pd.read_csv(gt_path).rename(columns={"team_goals":"team_goals_true","opp_goals":"opp_goals_true"})
        df = final.merge(gt, on="Id", how="inner")
        
        df["loss"] = df.apply(lambda r: awmae_single(r["team_goals"], r["opp_goals"], r["team_goals_true"], r["opp_goals_true"]), axis=1)
        exact = (df["team_goals"]==df["team_goals_true"]) & (df["opp_goals"]==df["opp_goals_true"])
        out_ok = np.sign(df["team_goals"]-df["opp_goals"]) == np.sign(df["team_goals_true"]-df["opp_goals_true"])
        
        print("\n" + "="*60)
        print(f"RESULTS (V27 Hybrid Cascade, threshold={HARD_CASCADE_THRESHOLD})")
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
        
        # Per-tier
        print("-"*60)
        df2 = df.merge(test[["Id","tournament_tier"]], on="Id", how="left")
        for tier in sorted(df2["tournament_tier"].unique(), reverse=True):
            sub = df2[df2["tournament_tier"]==tier]
            sl = sub["loss"].mean()
            se = ((sub["team_goals"]==sub["team_goals_true"])&(sub["opp_goals"]==sub["opp_goals_true"])).mean()*100
            so = (np.sign(sub["team_goals"]-sub["opp_goals"])==np.sign(sub["team_goals_true"]-sub["opp_goals_true"])).mean()*100
            print(f"  Tier {int(tier)} | AW-MAE: {sl:.5f} | Exact: {se:.1f}% | Outcome: {so:.1f}% | N={len(sub)}")
        print("="*60)
        
        # Draw prediction analysis
        draw_pred = (df["team_goals"] == df["opp_goals"]).sum()
        draw_gt   = (df["team_goals_true"] == df["opp_goals_true"]).sum()
        print(f"\nDraw predictions: {draw_pred} ({draw_pred/len(df)*100:.1f}%) vs GT: {draw_gt} ({draw_gt/len(df)*100:.1f}%)")
        
        # Score distribution
        print("\nPrediction Distribution (top 15):")
        print(df.groupby(["team_goals","opp_goals"]).size().sort_values(ascending=False).head(15))
    
    # Save
    sample = pd.read_csv(DATA_DIR / "sample submission.csv")
    sub = sample[["Id"]].merge(final, on="Id", how="left")
    out_path = DATA_DIR / "submission_v27.csv"
    sub.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path} ({len(sub)} rows)")

if __name__ == "__main__":
    main()
