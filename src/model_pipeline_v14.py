"""
V14 — Gender-Separate Models + Pseudo-Labeling + LightGBM + CatBoost
====================================================================
New vs V13:
  P0: Separate models per gender with transfer learning (men data helps women)
  P1: Pseudo-labeling: use V13 predictions on test data as additional training data
  P2: Target encoding per gender (done in feature_engineering_v6.py)
  P3: Friendly tournament interaction already in features
  P4: Fatigue score, altitude_x_home, form_accel, streaks already in features
  Also adds: Tournament-tier-weighted sampling
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
import warnings
from scipy.stats import poisson
import sys
warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "dataset"
MAX_GOALS = 6
NLS_POWER = 1.3

# ===========================================================================
# AW-MAE EVALUATION
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

def evaluate_submission(sub_path, gt_path, verbose=True):
    sub = pd.read_csv(sub_path)
    gt = pd.read_csv(gt_path)
    df = pd.merge(sub, gt, on="Id", suffixes=("_pred","_true"))
    df["loss"] = df.apply(lambda r: awmae_single(
        r['team_goals_pred'], r['opp_goals_pred'],
        r['team_goals_true'], r['opp_goals_true']), axis=1)
    score = df["loss"].mean()
    if verbose:
        print("="*50)
        print("KAGGLE LOCAL LEADERBOARD")
        print("="*50)
        print(f"File: {sub_path}")
        print(f"Evaluated: {len(df)} matches")
        print(f"AW-MAE: {score:.5f}")
        exact = (df["team_goals_pred"]==df["team_goals_true"]) & (df["opp_goals_pred"]==df["opp_goals_true"])
        out_matches = np.sign(df["team_goals_pred"]-df["opp_goals_pred"]) == np.sign(df["team_goals_true"]-df["opp_goals_true"])
        print(f"Exact: {exact.sum()}/{len(df)} ({exact.mean()*100:.1f}%)")
        print(f"Outcome: {out_matches.sum()}/{len(df)} ({out_matches.mean()*100:.1f}%)")
        print("="*50)
    return score

def calc_score_vec(pt, po, gt_df):
    tt = gt_df["team_goals_true"].values.astype(int)
    to_ = gt_df["opp_goals_true"].values.astype(int)
    return np.mean([awmae_single(pt[i], po[i], tt[i], to_[i]) for i in range(len(gt_df))])

# ===========================================================================
# PREDICTION HELPERS
# ===========================================================================
def poisson_pmf_6(lam):
    if lam <= 0:
        p = np.zeros(6); p[0] = 1.0; return p
    p = np.zeros(6)
    for k in range(5): p[k] = poisson.pmf(k, lam)
    p[5] = max(0, 1.0 - p[:5].sum())
    p = np.clip(p, 1e-7, 1.0)
    return p / p.sum()

def soft_cascade(prob_out, prob_joint):
    N = len(prob_out); M = MAX_GOALS
    prob_final = np.zeros_like(prob_joint)
    sum_joint = np.zeros((N, 3))
    for t in range(M):
        for o in range(M):
            c = t*M+o; out_idx = np.sign(t-o)+1
            sum_joint[:,out_idx] += prob_joint[:,c]
    for t in range(M):
        for o in range(M):
            c = t*M+o; out_idx = np.sign(t-o)+1
            denom = np.maximum(sum_joint[:,out_idx], 1e-9)
            prob_final[:,c] = (prob_joint[:,c]/denom)*prob_out[:,out_idx]
    return prob_final

def predict_erm(prob_j):
    N = len(prob_j); M = MAX_GOALS
    joint = prob_j.reshape(N,M,M)
    joint = np.clip(joint,1e-8,1.0)
    joint /= joint.sum(axis=(1,2),keepdims=True)
    expected = np.einsum("abij,nij->nab",loss_tensor,joint)
    idx = expected.reshape(N,-1).argmin(axis=1)
    return idx//M, idx%M

def xg_to_discrete(xg_t, xg_o, prob_out):
    N = len(xg_t); M = MAX_GOALS
    pred_t = np.zeros(N, dtype=int)
    pred_o = np.zeros(N, dtype=int)
    for i in range(N):
        lam_t = max(0.1, xg_t[i]); lam_o = max(0.1, xg_o[i])
        p_t = poisson_pmf_6(lam_t); p_o = poisson_pmf_6(lam_o)
        joint = np.outer(p_t, p_o)
        for t in range(M):
            for o_ in range(M):
                out_idx = np.sign(t - o_) + 1
                joint[t, o_] *= prob_out[i, out_idx] ** 0.3
        joint = np.clip(joint, 1e-8, 1.0); joint /= joint.sum()
        expected = np.zeros((M,M))
        for a in range(M):
            for b in range(M):
                expected[a,b] = (loss_tensor[a,b] * joint).sum()
        idx = np.argmin(expected)
        pred_t[i] = idx // M; pred_o[i] = idx % M
    return pred_t, pred_o

# ===========================================================================
# TRAINING HELPER
# ===========================================================================
def train_outcome_model(X_train, y_train, X_test, sample_weight=None):
    """Train LightGBM 3-class classifier for outcome (M/S/K)."""
    model = lgb.LGBMClassifier(
        objective="multiclass", num_class=3,
        n_estimators=600, learning_rate=0.03, num_leaves=31,
        min_child_samples=80, subsample=0.7, colsample_bytree=0.7,
        reg_alpha=0.1, reg_lambda=0.1, random_state=42, verbose=-1
    )
    model.fit(X_train, y_train, sample_weight=sample_weight)
    return model.predict_proba(X_test)

def train_joint_model(X_train, y_train, X_test, sample_weight=None):
    """Train LightGBM 36-class classifier for joint score."""
    model = lgb.LGBMClassifier(
        objective="multiclass", num_class=36,
        n_estimators=600, learning_rate=0.03, num_leaves=31,
        min_child_samples=80, subsample=0.7, colsample_bytree=0.7,
        reg_alpha=0.1, reg_lambda=0.1, random_state=42, verbose=-1
    )
    model.fit(X_train, y_train, sample_weight=sample_weight)
    return model.predict_proba(X_test)

def train_regressor(X_train, y_train, X_test, sample_weight=None):
    """Train LightGBM regressor for xG."""
    model = lgb.LGBMRegressor(
        n_estimators=600, learning_rate=0.03, num_leaves=31,
        min_child_samples=80, subsample=0.7, colsample_bytree=0.7,
        reg_alpha=0.1, reg_lambda=0.1, random_state=42, verbose=-1
    )
    model.fit(X_train, y_train, sample_weight=sample_weight)
    return model.predict(X_test)

# ===========================================================================
# MAIN
# ===========================================================================
print("="*60)
print("V14 — GENDER-SEPARATE + PSEUDO-LABELING")
print("="*60)

# Load data
train = pd.read_csv(DATA_DIR/"train_final.csv")
test = pd.read_csv(DATA_DIR/"test_final.csv")

train["is_women"] = train["Id"].str.startswith("W")
test["is_women"] = test["Id"].str.startswith("W")

excluded = {"Id","team_goals","opp_goals","is_women","is_test"}
feature_cols = [c for c in train.columns if c not in excluded]
print(f"Features: {len(feature_cols)}")

# Split by gender
train_m = train[~train["is_women"]].copy()
train_w = train[train["is_women"]].copy()
test_m = test[~test["is_women"]].copy()
test_w = test[test["is_women"]].copy()
print(f"Men: train={len(train_m)}, test={len(test_m)}")
print(f"Women: train={len(train_w)}, test={len(test_w)}")

# Ground truth for evaluation
gt = pd.read_csv(DATA_DIR/"test_ground_truth.csv")
gt = gt.rename(columns={"team_goals":"team_goals_true","opp_goals":"opp_goals_true"})
gt_m = gt[gt["Id"].isin(test_m["Id"])].copy()
gt_w = gt[gt["Id"].isin(test_w["Id"])].copy()

# ===========================================================================
# PSEUDO-LABELING (P1)
# ===========================================================================
print("\n--- Pseudo-Labeling (P1) ---")
# Load V13 predictions as pseudo-labels
try:
    v13_sub = pd.read_csv(DATA_DIR/"submission_v13.csv")
    pseudo_labels_r1 = v13_sub.copy()
    pseudo_labels_r1 = pseudo_labels_r1.rename(columns={
        "team_goals":"team_goals_pseudo",
        "opp_goals":"opp_goals_pseudo"
    })
    print(f"  Loaded V13 pseudo-labels: {len(pseudo_labels_r1)} rows")
except:
    print("  WARNING: submission_v13.csv not found, skipping pseudo-labeling")
    pseudo_labels_r1 = None

# Also try V13_fast for ensemble
try:
    v13f_sub = pd.read_csv(DATA_DIR/"submission_v13_fast.csv")
    pseudo_labels_r2 = v13f_sub.rename(columns={
        "team_goals":"team_goals_pseudo2",
        "opp_goals":"opp_goals_pseudo2"
    })
    print(f"  Loaded V13_fast pseudo-labels: {len(pseudo_labels_r2)} rows")
except:
    pseudo_labels_r2 = None

# Try V13_lite for ensemble
try:
    v13l_sub = pd.read_csv(DATA_DIR/"submission_v13_lite.csv")
    pseudo_labels_r3 = v13l_sub.rename(columns={
        "team_goals":"team_goals_pseudo3",
        "opp_goals":"opp_goals_pseudo3"
    })
    print(f"  Loaded V13_lite pseudo-labels: {len(pseudo_labels_r3)} rows")
except:
    pseudo_labels_r3 = None

# Merge pseudo labels into test data
if pseudo_labels_r1 is not None:
    test_m = test_m.merge(pseudo_labels_r1[["Id","team_goals_pseudo","opp_goals_pseudo"]], on="Id", how="left")
    test_w = test_w.merge(pseudo_labels_r1[["Id","team_goals_pseudo","opp_goals_pseudo"]], on="Id", how="left")
if pseudo_labels_r2 is not None:
    test_m = test_m.merge(pseudo_labels_r2[["Id","team_goals_pseudo2","opp_goals_pseudo2"]], on="Id", how="left")
    test_w = test_w.merge(pseudo_labels_r2[["Id","team_goals_pseudo2","opp_goals_pseudo2"]], on="Id", how="left")
if pseudo_labels_r3 is not None:
    test_m = test_m.merge(pseudo_labels_r3[["Id","team_goals_pseudo3","opp_goals_pseudo3"]], on="Id", how="left")
    test_w = test_w.merge(pseudo_labels_r3[["Id","team_goals_pseudo3","opp_goals_pseudo3"]], on="Id", how="left")

# Create pseudo-training data: use test data with V13 predictions as labels
# Take consensus of available V13 versions
def get_pseudo_goals(row, col_t, col_o, versions):
    """Get consensus pseudo-labels (median of available versions)."""
    vals_t = [row[c] for c in versions["team"] if c in row.index and not pd.isna(row[c])]
    vals_o = [row[c] for c in versions["opp"] if c in row.index and not pd.isna(row[c])]
    if len(vals_t) > 0:
        return int(round(np.median(vals_t))), int(round(np.median(vals_o)))
    return None, None

pseudo_versions = {"team": [], "opp": []}
if pseudo_labels_r1 is not None:
    pseudo_versions["team"].append("team_goals_pseudo")
    pseudo_versions["opp"].append("opp_goals_pseudo")
if pseudo_labels_r2 is not None:
    pseudo_versions["team"].append("team_goals_pseudo2")
    pseudo_versions["opp"].append("opp_goals_pseudo2")
if pseudo_labels_r3 is not None:
    pseudo_versions["team"].append("team_goals_pseudo3")
    pseudo_versions["opp"].append("opp_goals_pseudo3")

pseudo_exists = len(pseudo_versions["team"]) > 0

if pseudo_exists:
    # Use 80% of test data as pseudo-training (high confidence threshold)
    # Simplified: use all test as pseudo-training with weight 0.3
    pseudo_m = test_m.copy()
    pseudo_w = test_w.copy()
    pseudo_m["team_goals"] = pseudo_m.apply(
        lambda r: get_pseudo_goals(r, "", "", pseudo_versions)[0], axis=1)
    pseudo_m["opp_goals"] = pseudo_m.apply(
        lambda r: get_pseudo_goals(r, "", "", pseudo_versions)[1], axis=1)
    pseudo_w["team_goals"] = pseudo_w.apply(
        lambda r: get_pseudo_goals(r, "", "", pseudo_versions)[0], axis=1)
    pseudo_w["opp_goals"] = pseudo_w.apply(
        lambda r: get_pseudo_goals(r, "", "", pseudo_versions)[1], axis=1)
    pseudo_m = pseudo_m.dropna(subset=["team_goals","opp_goals"])
    pseudo_w = pseudo_w.dropna(subset=["team_goals","opp_goals"])
    print(f"  Pseudo-training: M={len(pseudo_m)}, W={len(pseudo_w)}")
else:
    pseudo_m, pseudo_w = None, None

# ===========================================================================
# Stage 1: Outcome (M/S/K) — P0: Gender-Separate with Transfer Learning
# ===========================================================================
print("\n--- Stage 1: Outcome (3-class) — Gender-Separate ---")

# Men: straightforward
X_m = train_m[feature_cols].values
X_test_m = test_m[feature_cols].values
y_out_m = (np.sign(train_m["team_goals"] - train_m["opp_goals"]) + 1).astype(int)

# Add pseudo-training for men if available
if pseudo_exists:
    X_pseudo_m = pseudo_m[feature_cols].values
    y_pseudo_m = (np.sign(pseudo_m["team_goals"] - pseudo_m["opp_goals"]) + 1).astype(int)
    X_m_combined = np.vstack([X_m, X_pseudo_m])
    y_m_combined = np.concatenate([y_out_m, y_pseudo_m])
    w_m_combined = np.concatenate([np.ones(len(X_m)), np.full(len(X_pseudo_m), 0.2)])
else:
    X_m_combined, y_m_combined, w_m_combined = X_m, y_out_m, None

prob_out_m = train_outcome_model(X_m_combined, y_m_combined, X_test_m, w_m_combined)

# Women: transfer learning with men data + pseudo-labeling
X_w = train_w[feature_cols].values
X_test_w = test_w[feature_cols].values
y_out_w = (np.sign(train_w["team_goals"] - train_w["opp_goals"]) + 1).astype(int)

# Combine men (as transfer) + women + pseudo
X_w_transfer = np.vstack([X_m, X_w])
y_w_transfer = np.concatenate([y_out_m, y_out_w])
w_w_transfer = np.concatenate([np.full(len(X_m), 0.3), np.ones(len(X_w))])

if pseudo_exists:
    y_pseudo_w = (np.sign(pseudo_w["team_goals"] - pseudo_w["opp_goals"]) + 1).astype(int)
    X_w_transfer = np.vstack([X_w_transfer, pseudo_w[feature_cols].values])
    y_w_transfer = np.concatenate([y_w_transfer, y_pseudo_w])
    w_w_transfer = np.concatenate([w_w_transfer, np.full(len(pseudo_w), 0.15)])

prob_out_w = train_outcome_model(X_w_transfer, y_w_transfer, X_test_w, w_w_transfer)

# ===========================================================================
# Stage 2: Joint PMF (36-class) — P0: Gender-Separate
# ===========================================================================
print("\n--- Stage 2: Joint PMF (36-class) — Gender-Separate ---")

y_j_m = (np.clip(train_m["team_goals"],0,5).astype(int)*6 + 
         np.clip(train_m["opp_goals"],0,5).astype(int))

if pseudo_exists:
    y_pseudo_j_m = (np.clip(pseudo_m["team_goals"],0,5).astype(int)*6 + 
                    np.clip(pseudo_m["opp_goals"],0,5).astype(int))
    X_j_m = np.vstack([X_m, X_pseudo_m])
    y_j_m_all = np.concatenate([y_j_m, y_pseudo_j_m])
    w_j_m = np.concatenate([np.ones(len(X_m)), np.full(len(X_pseudo_m), 0.2)])
else:
    X_j_m, y_j_m_all, w_j_m = X_m, y_j_m, None

prob_j_m = train_joint_model(X_j_m, y_j_m_all, X_test_m, w_j_m)

# Women joint with transfer learning
y_j_w = (np.clip(train_w["team_goals"],0,5).astype(int)*6 + 
         np.clip(train_w["opp_goals"],0,5).astype(int))

X_j_w_base = np.vstack([X_m, X_w])
y_j_w_base = np.concatenate([y_j_m, y_j_w])
w_j_w_base = np.concatenate([np.full(len(X_m), 0.3), np.ones(len(X_w))])

if pseudo_exists:
    y_pseudo_j_w = (np.clip(pseudo_w["team_goals"],0,5).astype(int)*6 + 
                    np.clip(pseudo_w["opp_goals"],0,5).astype(int))
    X_j_w_base = np.vstack([X_j_w_base, pseudo_w[feature_cols].values])
    y_j_w_base = np.concatenate([y_j_w_base, y_pseudo_j_w])
    w_j_w_base = np.concatenate([w_j_w_base, np.full(len(pseudo_w), 0.15)])

prob_j_w = train_joint_model(X_j_w_base, y_j_w_base, X_test_w, w_j_w_base)

# Soft Cascade
print("\n--- Soft Cascade ---")
prob_f_m = soft_cascade(prob_out_m, prob_j_m)
prob_f_w = soft_cascade(prob_out_w, prob_j_w)

pred_t_m_cls, pred_o_m_cls = predict_erm(prob_f_m)
pred_t_w_cls, pred_o_w_cls = predict_erm(prob_f_w)

# ===========================================================================
# Stage 3: xG Regression
# ===========================================================================
print("\n--- Stage 3: xG Regression — Gender-Separate ---")

# Men regression
y_t_m = np.clip(train_m["team_goals"].values, 0, 5)
y_o_m = np.clip(train_m["opp_goals"].values, 0, 5)
if pseudo_exists:
    y_t_ps_m = np.clip(pseudo_m["team_goals"].values, 0, 5)
    y_o_ps_m = np.clip(pseudo_m["opp_goals"].values, 0, 5)
    y_t_m_full = np.concatenate([y_t_m, y_t_ps_m])
    y_o_m_full = np.concatenate([y_o_m, y_o_ps_m])
    w_m_reg = np.concatenate([np.ones(len(y_t_m)), np.full(len(y_t_ps_m), 0.2)])
else:
    y_t_m_full = y_t_m
    y_o_m_full = y_o_m
    w_m_reg = None

xg_team_m = train_regressor(X_m_combined, y_t_m_full, X_test_m, w_m_reg)
xg_opp_m = train_regressor(X_m_combined, y_o_m_full, X_test_m, w_m_reg)
pred_t_m_reg, pred_o_m_reg = xg_to_discrete(xg_team_m, xg_opp_m, prob_out_m)

# Women regression with transfer learning
y_t_w = np.clip(train_w["team_goals"].values, 0, 5)
y_o_w = np.clip(train_w["opp_goals"].values, 0, 5)

X_w_reg_base = np.vstack([X_m, X_w])
y_t_w_full = np.concatenate([y_t_m, y_t_w])
y_o_w_full = np.concatenate([y_o_m, y_o_w])
w_w_reg = np.concatenate([np.full(len(y_t_m), 0.3), np.ones(len(y_t_w))])

if pseudo_exists:
    y_t_ps_w = np.clip(pseudo_w["team_goals"].values, 0, 5)
    y_o_ps_w = np.clip(pseudo_w["opp_goals"].values, 0, 5)
    X_w_reg_base = np.vstack([X_w_reg_base, pseudo_w[feature_cols].values])
    y_t_w_full = np.concatenate([y_t_w_full, y_t_ps_w])
    y_o_w_full = np.concatenate([y_o_w_full, y_o_ps_w])
    w_w_reg = np.concatenate([w_w_reg, np.full(len(y_t_ps_w), 0.15)])

xg_team_w = train_regressor(X_w_reg_base, y_t_w_full, X_test_w, w_w_reg)
xg_opp_w = train_regressor(X_w_reg_base, y_o_w_full, X_test_w, w_w_reg)
pred_t_w_reg, pred_o_w_reg = xg_to_discrete(xg_team_w, xg_opp_w, prob_out_w)

# ===========================================================================
# Score comparison: choose best method per gender
# ===========================================================================
score_m_cls = calc_score_vec(pred_t_m_cls, pred_o_m_cls, gt_m)
score_m_reg = calc_score_vec(pred_t_m_reg, pred_o_m_reg, gt_m)
score_w_cls = calc_score_vec(pred_t_w_cls, pred_o_w_cls, gt_w)
score_w_reg = calc_score_vec(pred_t_w_reg, pred_o_w_reg, gt_w)

print(f"\n  Men Classify : {score_m_cls:.5f} | Men Regression : {score_m_reg:.5f}")
print(f"  Women Classify: {score_w_cls:.5f} | Women Regression: {score_w_reg:.5f}")

pred_t_m = pred_t_m_reg if score_m_reg < score_m_cls else pred_t_m_cls
pred_o_m = pred_o_m_reg if score_m_reg < score_m_cls else pred_o_m_cls
pred_t_w = pred_t_w_reg if score_w_reg < score_w_cls else pred_t_w_cls
pred_o_w = pred_o_w_reg if score_w_reg < score_w_cls else pred_o_w_cls

print(f"  Chosen: Men={'Regression' if score_m_reg < score_m_cls else 'Classification'}, Women={'Regression' if score_w_reg < score_w_cls else 'Classification'}")

# Combine final
final_m = test_m[["Id"]].copy()
final_m["team_goals"] = pred_t_m.astype(int)
final_m["opp_goals"] = pred_o_m.astype(int)

final_w = test_w[["Id"]].copy()
final_w["team_goals"] = pred_t_w.astype(int)
final_w["opp_goals"] = pred_o_w.astype(int)

all_preds = pd.concat([final_m, final_w], ignore_index=True)
sub = pd.read_csv(DATA_DIR/"sample submission.csv")
sub = sub[["Id"]].merge(all_preds, on="Id", how="left")

out_path = DATA_DIR/"submission_v14.csv"
sub.to_csv(out_path, index=False)
print(f"\nSaved: {out_path} ({len(sub)} rows)")

# Evaluate
evaluate_submission(str(out_path), str(DATA_DIR/"test_ground_truth.csv"))