"""
V26 — Tier-Adaptive Post-Processing (Built on V14 foundation)
=============================================================
Key changes:
  S1: Tier-Aware Temperature + Draw Boost per segment
  S5: Score Prior Injection per segment (Bayesian smoothing)
  S6: Elo Confidence Discount per tier (new feature)
  
  Base: LGB-only (ensemble diversity adds <0.01), Gender-Split, Two-Stage Cascade, ERM
  
  Philosophy: Stop changing the MODEL. Change the DECISION PARAMETERS per context.
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
import warnings, time
warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "dataset"
MAX_GOALS = 6
NUM_CLASSES_OUT = 3
NUM_CLASSES_JOINT = MAX_GOALS * MAX_GOALS
NLS_POWER = 1.3

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

loss_tensor = np.zeros((MAX_GOALS, MAX_GOALS, MAX_GOALS, MAX_GOALS))
for a in range(MAX_GOALS):
    for b in range(MAX_GOALS):
        for c in range(MAX_GOALS):
            for d in range(MAX_GOALS):
                loss_tensor[a,b,c,d] = awmae_single(a,b,c,d)

# ===========================================================================
# S6: ELO CONFIDENCE DISCOUNT (feature-level)
# ===========================================================================
ELO_DISCOUNT = {1: 0.75, 2: 0.80, 3: 0.90, 4: 0.85, 5: 0.90}

def add_elo_discount_feature(df):
    """Add elo_diff_adjusted = elo_diff * tier_discount"""
    df = df.copy()
    df["elo_diff_adjusted"] = df.apply(
        lambda r: r["elo_diff"] * ELO_DISCOUNT.get(int(r["tournament_tier"]), 0.85), axis=1
    )
    return df

# ===========================================================================
# S1: TIER-AWARE DRAW BOOST (per segment)
# ===========================================================================
# Parameters calibrated from TRAINING data distribution patterns.
# Key insight: Tier 4 (Continental) and Tier 1 (Friendly) have ~26% draw rate
# vs Tier 5 (WC) at ~19% and Tier 3 (Qualifiers) at ~21%.
DRAW_BOOST = {
    # (is_women, tier): draw_probability_additive_boost
    (False, 5): 0.00,   # Men WC: moderate draw rate, no boost
    (False, 4): 0.06,   # Men Continental: very high draw rate
    (False, 3): 0.01,   # Men Qualifiers: slightly above baseline
    (False, 2): 0.02,   # Men Regional: moderate
    (False, 1): 0.06,   # Men Friendly: very high draw rate
    (True, 5): -0.02,   # Women WC: lower draw rate than men
    (True, 4): -0.02,   # Women Continental: lower draw rate
    (True, 3): -0.06,   # Women Qualifiers: CHAOS, very few draws (12.6%)
    (True, 2): -0.03,   # Women Regional: fewer draws
    (True, 1): -0.01,   # Women Friendly: moderate
}

# S1: TIER-AWARE TEMPERATURE
TEMPERATURE = {
    (False, 5): 1.05,  # Men WC: slightly sharper
    (False, 4): 1.25,  # Men Continental: draw-heavy, spread more
    (False, 3): 1.10,  # Men Qualifiers: moderate
    (False, 2): 1.10,  # Men Regional: moderate
    (False, 1): 1.25,  # Men Friendly: draw-heavy, spread more
    (True, 5): 1.10,   # Women WC: moderate
    (True, 4): 1.10,   # Women Continental: moderate
    (True, 3): 0.95,   # Women Qualifiers: chaos, be more decisive
    (True, 2): 1.05,   # Women Regional: slightly sharp
    (True, 1): 1.15,   # Women Friendly: moderate spread
}

# ===========================================================================
# S5: SCORE PRIOR per segment (computed from TRAINING data, not GT)
# ===========================================================================
def compute_score_priors(train_df):
    """Compute empirical score distribution per (is_women, tier) from training data"""
    M = MAX_GOALS
    priors = {}
    train_df = train_df.copy()
    train_df["is_women"] = train_df["Id"].str.startswith("W")
    
    for is_w in [False, True]:
        for tier in range(1, 6):
            sub = train_df[(train_df["is_women"]==is_w) & (train_df["tournament_tier"]==tier)]
            prior = np.zeros(M * M)
            if len(sub) > 0:
                t = np.clip(sub["team_goals"].values, 0, M-1).astype(int)
                o = np.clip(sub["opp_goals"].values, 0, M-1).astype(int)
                for i in range(len(t)):
                    prior[t[i]*M + o[i]] += 1
                prior = prior / prior.sum()
                # Laplace smoothing
                prior = (prior + 1e-4)
                prior = prior / prior.sum()
            else:
                prior = np.ones(M*M) / (M*M)
            priors[(is_w, tier)] = prior
    return priors

PRIOR_ALPHA = 0.08  # mixing weight: 8% prior, 92% model

# ===========================================================================
# SOFT CASCADE + TIER-AWARE POST-PROCESSING
# ===========================================================================
def soft_cascade_tier_adaptive(prob_out, prob_joint, tiers, is_women_flags):
    """
    Full pipeline:
    1. Apply draw boost to outcome probabilities (S1)
    2. Apply temperature scaling to joint probabilities (S1)
    3. Bucketed renormalization (Soft Cascade)
    4. Apply score prior injection (S5)
    """
    N = len(prob_out)
    M = MAX_GOALS
    
    # --- S1: Draw Boost ---
    prob_out_adj = prob_out.copy()
    for i in range(N):
        key = (bool(is_women_flags[i]), int(tiers[i]))
        boost = DRAW_BOOST.get(key, 0.0)
        prob_out_adj[i, 1] += boost  # index 1 = Draw (sign(0)+1)
        prob_out_adj[i] = np.clip(prob_out_adj[i], 0.01, 0.99)
        prob_out_adj[i] /= prob_out_adj[i].sum()
    
    # --- S1: Tier-Aware Temperature on Joint ---
    prob_joint_adj = prob_joint.copy()
    for i in range(N):
        key = (bool(is_women_flags[i]), int(tiers[i]))
        T = TEMPERATURE.get(key, 1.1)
        p = np.clip(prob_joint_adj[i], 1e-7, 1.0)
        log_p = np.log(p) / T
        exp_p = np.exp(log_p - log_p.max())  # numerical stability
        prob_joint_adj[i] = exp_p / exp_p.sum()
    
    # --- Soft Cascade (bucketed renormalization) ---
    prob_final = np.zeros_like(prob_joint_adj)
    sum_joint = np.zeros((N, 3))
    for t in range(M):
        for o in range(M):
            c = t*M+o
            out_idx = int(np.sign(t-o)) + 1
            sum_joint[:, out_idx] += prob_joint_adj[:, c]
    for t in range(M):
        for o in range(M):
            c = t*M+o
            out_idx = int(np.sign(t-o)) + 1
            denom = np.maximum(sum_joint[:, out_idx], 1e-9)
            prob_final[:, c] = (prob_joint_adj[:, c] / denom) * prob_out_adj[:, out_idx]
    
    return prob_final

def inject_score_priors(prob_final, tiers, is_women_flags, priors, alpha=PRIOR_ALPHA):
    """S5: Bayesian smoothing with empirical score prior"""
    N = len(prob_final)
    prob_smoothed = prob_final.copy()
    for i in range(N):
        key = (bool(is_women_flags[i]), int(tiers[i]))
        prior = priors.get(key, np.ones(MAX_GOALS**2) / MAX_GOALS**2)
        prob_smoothed[i] = (1 - alpha) * prob_final[i] + alpha * prior
        prob_smoothed[i] /= prob_smoothed[i].sum()
    return prob_smoothed

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
# LGB PARAMS
# ===========================================================================
LGB_OUT = {
    "objective": "multiclass", "num_class": NUM_CLASSES_OUT, "metric": "multi_logloss",
    "num_leaves": 31, "learning_rate": 0.02, "min_child_samples": 100,
    "subsample": 0.8, "colsample_bytree": 0.8, "verbose": -1, "seed": 42
}
LGB_JOINT = {
    "objective": "multiclass", "num_class": NUM_CLASSES_JOINT, "metric": "multi_logloss",
    "num_leaves": 31, "learning_rate": 0.02, "min_child_samples": 150,
    "subsample": 0.7, "colsample_bytree": 0.7, "verbose": -1, "seed": 42
}
N_ESTIMATORS = 600

# ===========================================================================
# MAIN
# ===========================================================================
def main():
    print("=" * 60)
    print("V26 — TIER-ADAPTIVE POST-PROCESSING")
    print("  S1: Tier-Aware Temperature + Draw Boost")
    print("  S5: Score Prior Injection")
    print("  S6: Elo Confidence Discount")
    print("=" * 60)
    
    # --- Load & Prepare ---
    print("\n[1] Loading data...")
    train = pd.read_csv(DATA_DIR / "train_final.csv")
    test  = pd.read_csv(DATA_DIR / "test_final.csv")
    
    # S6: Add elo_diff_adjusted
    train = add_elo_discount_feature(train)
    test  = add_elo_discount_feature(test)
    
    train["is_women"] = train["Id"].str.startswith("W")
    test["is_women"]  = test["Id"].str.startswith("W")
    
    excluded = {"Id", "team_goals", "opp_goals", "is_women", "is_test"}
    feature_cols = [c for c in train.columns if c not in excluded]
    print(f"  Features: {len(feature_cols)} (includes elo_diff_adjusted)")
    
    # S5: Compute priors from TRAINING data only
    print("[2] Computing score priors from training data...")
    priors = compute_score_priors(train)
    for key, prior in priors.items():
        mode_idx = np.argmax(prior)
        mode_t, mode_o = mode_idx // MAX_GOALS, mode_idx % MAX_GOALS
        print(f"  {('Women' if key[0] else 'Men'):5s} Tier {key[1]}: mode={mode_t}-{mode_o} ({prior[mode_idx]*100:.1f}%)")
    
    # --- Gender Split ---
    print("\n[3] Splitting by gender...")
    train_m = train[~train["is_women"]].copy()
    train_w = train[train["is_women"]].copy()
    test_m  = test[~test["is_women"]].copy()
    test_w  = test[test["is_women"]].copy()
    print(f"  Men:   train={len(train_m)}, test={len(test_m)}")
    print(f"  Women: train={len(train_w)}, test={len(test_w)}")
    
    results = {}
    
    for gender_name, tr, te in [("MEN", train_m, test_m), ("WOMEN", train_w, test_w)]:
        if len(te) == 0:
            continue
        print(f"\n{'='*60}")
        print(f"  TRAINING: {gender_name}")
        print(f"{'='*60}")
        
        X_train = tr[feature_cols].values
        X_test  = te[feature_cols].values
        
        y_t = np.clip(tr["team_goals"].values, 0, MAX_GOALS-1).astype(int)
        y_o = np.clip(tr["opp_goals"].values, 0, MAX_GOALS-1).astype(int)
        y_out   = (np.sign(y_t - y_o) + 1).astype(int)  # 0=Loss, 1=Draw, 2=Win
        y_joint = y_t * MAX_GOALS + y_o
        
        # --- Stage 1: Outcome ---
        t0 = time.time()
        print(f"  Stage 1: Outcome (3-class)...")
        dt_out = lgb.Dataset(X_train, y_out, free_raw_data=False)
        model_out = lgb.train(LGB_OUT, dt_out, num_boost_round=N_ESTIMATORS)
        prob_out = model_out.predict(X_test)
        print(f"    Done in {time.time()-t0:.1f}s")
        
        # --- Stage 2: Joint PMF (36-class) ---
        t0 = time.time()
        print(f"  Stage 2: Joint PMF (36-class)...")
        dt_joint = lgb.Dataset(X_train, y_joint, free_raw_data=False)
        model_joint = lgb.train(LGB_JOINT, dt_joint, num_boost_round=N_ESTIMATORS)
        prob_joint = model_joint.predict(X_test)
        print(f"    Done in {time.time()-t0:.1f}s")
        
        # --- Tier-Adaptive Post-Processing ---
        print(f"  Applying tier-adaptive post-processing...")
        tiers = te["tournament_tier"].values
        is_women_flags = te["is_women"].values
        
        # S1 + Cascade
        prob_cascaded = soft_cascade_tier_adaptive(prob_out, prob_joint, tiers, is_women_flags)
        
        # S5: Score Prior Injection
        prob_final = inject_score_priors(prob_cascaded, tiers, is_women_flags, priors)
        
        # ERM Decision
        pred_t, pred_o = predict_erm(prob_final)
        
        results[gender_name] = (te["Id"].values, pred_t, pred_o)
    
    # --- Combine & Evaluate ---
    print(f"\n{'='*60}")
    print("COMBINING RESULTS")
    print(f"{'='*60}")
    
    all_ids = np.concatenate([results[g][0] for g in results])
    all_pred_t = np.concatenate([results[g][1] for g in results])
    all_pred_o = np.concatenate([results[g][2] for g in results])
    
    final = pd.DataFrame({"Id": all_ids, "team_goals": all_pred_t.astype(int), "opp_goals": all_pred_o.astype(int)})
    
    gt_path = DATA_DIR / "test_ground_truth.csv"
    if gt_path.exists():
        gt = pd.read_csv(gt_path).rename(columns={"team_goals":"team_goals_true","opp_goals":"opp_goals_true"})
        df = final.merge(gt, on="Id", how="inner")
        
        df["loss"] = df.apply(lambda r: awmae_single(r["team_goals"], r["opp_goals"], r["team_goals_true"], r["opp_goals_true"]), axis=1)
        exact = (df["team_goals"]==df["team_goals_true"]) & (df["opp_goals"]==df["opp_goals_true"])
        out_ok = np.sign(df["team_goals"]-df["opp_goals"]) == np.sign(df["team_goals_true"]-df["opp_goals_true"])
        gd_ok = (df["team_goals"]-df["opp_goals"]) == (df["team_goals_true"]-df["opp_goals_true"])
        
        print("\n" + "="*60)
        print("RESULTS (V26 Tier-Adaptive)")
        print("="*60)
        print(f"Global AW-MAE:          {df['loss'].mean():.5f}")
        print(f"Global Exact Score:     {exact.mean()*100:.2f}%")
        print(f"Global Outcome Correct: {out_ok.mean()*100:.2f}%")
        print(f"Global GD Correct:      {gd_ok.mean()*100:.2f}%")
        print("-"*60)
        
        df["is_women"] = df["Id"].str.startswith("W")
        for name, mask in [("MEN", ~df["is_women"]), ("WOMEN", df["is_women"])]:
            if mask.sum() > 0:
                sl = df.loc[mask,"loss"].mean()
                se = exact[mask].mean()*100
                so = out_ok[mask].mean()*100
                print(f"  {name:7s} | AW-MAE: {sl:.5f} | Exact: {se:.1f}% | Outcome: {so:.1f}% | N={mask.sum()}")
        
        # Per-tier breakdown
        print("-"*60)
        df2 = df.merge(test[["Id","tournament_tier"]], on="Id", how="left")
        for tier in sorted(df2["tournament_tier"].unique(), reverse=True):
            sub = df2[df2["tournament_tier"]==tier]
            sl = sub["loss"].mean()
            se = ((sub["team_goals"]==sub["team_goals_true"])&(sub["opp_goals"]==sub["opp_goals_true"])).mean()*100
            so = (np.sign(sub["team_goals"]-sub["opp_goals"])==np.sign(sub["team_goals_true"]-sub["opp_goals_true"])).mean()*100
            print(f"  Tier {int(tier)} | AW-MAE: {sl:.5f} | Exact: {se:.1f}% | Outcome: {so:.1f}% | N={len(sub)}")
        print("="*60)
        
        print("\nPrediction Distribution:")
        print(df.groupby(["team_goals","opp_goals"]).size().sort_values(ascending=False).head(15))
    
    # Save submission
    sample = pd.read_csv(DATA_DIR / "sample submission.csv")
    sub = sample[["Id"]].merge(final, on="Id", how="left")
    out_path = DATA_DIR / "submission_v26.csv"
    sub.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path} ({len(sub)} rows)")

if __name__ == "__main__":
    main()
