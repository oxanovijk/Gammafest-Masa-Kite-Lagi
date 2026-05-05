"""V13 Lite — LightGBM-only ensemble for speed."""
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

# Evaluation infrastructure from evaluate_local.py
def evaluate_submission(sub_path="dataset/submission.csv", gt_path="dataset/test_ground_truth.csv", verbose=True):
    sub = pd.read_csv(sub_path)
    gt = pd.read_csv(gt_path)
    if len(sub) != len(gt):
        print(f"[!] ERROR: Panjang baris beda! sub={len(sub)}, gt={len(gt)}")
        return None
    df = pd.merge(sub, gt, on="Id", suffixes=("_pred", "_true"))
    df["loss"] = df.apply(lambda r: awmae_single(
        r['team_goals_pred'], r['opp_goals_pred'],
        r['team_goals_true'], r['opp_goals_true']), axis=1)
    score_unweighted = df["loss"].mean()
    if verbose:
        print("="*50)
        print("KAGGLE LOCAL LEADERBOARD")
        print("="*50)
        print(f"File Submission : {sub_path}")
        print(f"Total Dievaluasi: {len(df)} laga")
        print("-" * 50)
        print(f"AW-MAE SCORE (Unweighted): {score_unweighted:.5f}")
        exact_matches = (df["team_goals_pred"] == df["team_goals_true"]) & (df["opp_goals_pred"] == df["opp_goals_true"])
        pred_out = np.sign(df["team_goals_pred"] - df["opp_goals_pred"])
        true_out = np.sign(df["team_goals_true"] - df["opp_goals_true"])
        out_matches = pred_out == true_out
        print(f"  - Exact Score Correct : {exact_matches.sum()} / {len(df)} ({exact_matches.mean()*100:.1f}%)")
        print(f"  - Outcome M/S/K Correct: {out_matches.sum()} / {len(df)} ({out_matches.mean()*100:.1f}%)")
        print("="*50)
    return score_unweighted

def poisson_pmf_6(lam):
    if lam <= 0:
        p = np.zeros(6); p[0] = 1.0; return p
    p = np.zeros(6)
    for k in range(5): p[k] = poisson.pmf(k, lam)
    p[5] = max(0, 1.0 - p[:5].sum())
    p = np.clip(p, 1e-7, 1.0)
    return p / p.sum()

def xg_to_discrete(xg_t, xg_o, prob_out):
    """Convert expected goals to discrete predictions with outcome constraints."""
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

def predict_erm(prob_j):
    """ERM prediction from joint PMF."""
    N = len(prob_j); M = MAX_GOALS
    joint = prob_j.reshape(N,M,M)
    joint = np.clip(joint,1e-8,1.0)
    joint /= joint.sum(axis=(1,2),keepdims=True)
    expected = np.einsum("abij,nij->nab",loss_tensor,joint)
    idx = expected.reshape(N,-1).argmin(axis=1)
    return idx//M, idx%M

def soft_cascade(prob_out, prob_joint):
    """Refine joint PMF using outcome probabilities (P0B)."""
    N = len(prob_out); M = MAX_GOALS
    prob_final = np.zeros_like(prob_joint)
    sum_joint = np.zeros((N, 3))
    for t in range(M):
        for o in range(M):
            c = t*M+o
            out_idx = np.sign(t-o)+1
            sum_joint[:,out_idx] += prob_joint[:,c]
    for t in range(M):
        for o in range(M):
            c = t*M+o
            out_idx = np.sign(t-o)+1
            denom = np.maximum(sum_joint[:,out_idx], 1e-9)
            prob_final[:,c] = (prob_joint[:,c]/denom)*prob_out[:,out_idx]
    return prob_final

# ==================== MAIN ====================
print("="*60)
print("V13 LITE — LightGBM Only")
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

# ---- Stage 1: Outcome (M/S/K 3-class) ----
print("\n--- Stage 1: Outcome (3-class) ---")
X_train_m = train_m[feature_cols].values
X_test_m = test_m[feature_cols].values
X_train_w = train_w[feature_cols].values
X_test_w = test_w[feature_cols].values

y_out_m = (np.sign(train_m["team_goals"] - train_m["opp_goals"]) + 1).astype(int)

lgb_out_m = lgb.LGBMClassifier(
    objective="multiclass", num_class=3,
    n_estimators=500, learning_rate=0.03, num_leaves=31,
    min_child_samples=100, subsample=0.7, colsample_bytree=0.7,
    reg_alpha=0.1, reg_lambda=0.1, random_state=42, verbose=-1
)
lgb_out_m.fit(X_train_m, y_out_m)
prob_out_m = lgb_out_m.predict_proba(X_test_m)

# Women with transfer learning (P0A): combine men+women data, weight women higher
X_w_combined = np.vstack([X_train_m, X_train_w])
y_w_combined = np.concatenate([
    (np.sign(train_m["team_goals"] - train_m["opp_goals"]) + 1).astype(int),
    (np.sign(train_w["team_goals"] - train_w["opp_goals"]) + 1).astype(int)
])
w_combined = np.concatenate([np.full(len(train_m), 0.3), np.full(len(train_w), 1.0)])

lgb_out_w = lgb.LGBMClassifier(
    objective="multiclass", num_class=3,
    n_estimators=500, learning_rate=0.03, num_leaves=31,
    min_child_samples=100, subsample=0.7, colsample_bytree=0.7,
    reg_alpha=0.1, reg_lambda=0.1, random_state=42, verbose=-1
)
lgb_out_w.fit(X_w_combined, y_w_combined, sample_weight=w_combined)
prob_out_w = lgb_out_w.predict_proba(X_test_w)

# ---- Stage 2: Joint PMF (36-class flat) ----
print("\n--- Stage 2: Joint PMF (36-class) ---")
y_j_m = (np.clip(train_m["team_goals"],0,5).astype(int)*6 + 
         np.clip(train_m["opp_goals"],0,5).astype(int))

y_j_w_combined = np.concatenate([
    np.clip(train_m["team_goals"],0,5).astype(int)*6 + np.clip(train_m["opp_goals"],0,5).astype(int),
    np.clip(train_w["team_goals"],0,5).astype(int)*6 + np.clip(train_w["opp_goals"],0,5).astype(int)
])

lgb_j_m = lgb.LGBMClassifier(
    objective="multiclass", num_class=36,
    n_estimators=500, learning_rate=0.03, num_leaves=31,
    min_child_samples=100, subsample=0.7, colsample_bytree=0.7,
    reg_alpha=0.1, reg_lambda=0.1, random_state=42, verbose=-1
)
lgb_j_m.fit(X_train_m, y_j_m)
prob_j_m = lgb_j_m.predict_proba(X_test_m)

lgb_j_w = lgb.LGBMClassifier(
    objective="multiclass", num_class=36,
    n_estimators=500, learning_rate=0.03, num_leaves=31,
    min_child_samples=100, subsample=0.7, colsample_bytree=0.7,
    reg_alpha=0.1, reg_lambda=0.1, random_state=42, verbose=-1
)
lgb_j_w.fit(X_w_combined, y_j_w_combined, sample_weight=w_combined)
prob_j_w = lgb_j_w.predict_proba(X_test_w)

# ---- Soft Cascade (P0B) ----
print("\n--- Soft Cascade ---")
prob_f_m = soft_cascade(prob_out_m, prob_j_m)
prob_f_w = soft_cascade(prob_out_w, prob_j_w)

pred_t_m_cls, pred_o_m_cls = predict_erm(prob_f_m)
pred_t_w_cls, pred_o_w_cls = predict_erm(prob_f_w)

# ---- P3H: xG Regression ----
print("\n--- Stage 3: xG Regression ---")
# Men regression
lgb_reg_t_m = lgb.LGBMRegressor(
    n_estimators=500, learning_rate=0.03, num_leaves=31,
    min_child_samples=100, subsample=0.7, colsample_bytree=0.7,
    reg_alpha=0.1, reg_lambda=0.1, random_state=42, verbose=-1
)
lgb_reg_t_m.fit(X_train_m, np.clip(train_m["team_goals"],0,5))
xg_team_m = lgb_reg_t_m.predict(X_test_m)

lgb_reg_o_m = lgb.LGBMRegressor(
    n_estimators=500, learning_rate=0.03, num_leaves=31,
    min_child_samples=100, subsample=0.7, colsample_bytree=0.7,
    reg_alpha=0.1, reg_lambda=0.1, random_state=42, verbose=-1
)
lgb_reg_o_m.fit(X_train_m, np.clip(train_m["opp_goals"],0,5))
xg_opp_m = lgb_reg_o_m.predict(X_test_m)

pred_t_m_reg, pred_o_m_reg = xg_to_discrete(xg_team_m, xg_opp_m, prob_out_m)

# Women regression (transfer: train on men+womendata with weighted loss for women)
X_w_reg = np.vstack([X_train_m, X_train_w])
y_t_w_reg = np.concatenate([np.clip(train_m["team_goals"],0,5), np.clip(train_w["team_goals"],0,5)])
y_o_w_reg = np.concatenate([np.clip(train_m["opp_goals"],0,5), np.clip(train_w["opp_goals"],0,5)])

lgb_reg_t_w = lgb.LGBMRegressor(
    n_estimators=500, learning_rate=0.03, num_leaves=31,
    min_child_samples=100, subsample=0.7, colsample_bytree=0.7,
    reg_alpha=0.1, reg_lambda=0.1, random_state=42, verbose=-1
)
lgb_reg_t_w.fit(X_w_reg, y_t_w_reg, sample_weight=w_combined)
xg_team_w = lgb_reg_t_w.predict(X_test_w)

lgb_reg_o_w = lgb.LGBMRegressor(
    n_estimators=500, learning_rate=0.03, num_leaves=31,
    min_child_samples=100, subsample=0.7, colsample_bytree=0.7,
    reg_alpha=0.1, reg_lambda=0.1, random_state=42, verbose=-1
)
lgb_reg_o_w.fit(X_w_reg, y_o_w_reg, sample_weight=w_combined)
xg_opp_w = lgb_reg_o_w.predict(X_test_w)

pred_t_w_reg, pred_o_w_reg = xg_to_discrete(xg_team_w, xg_opp_w, prob_out_w)

# ---- Quick evaluation to choose best method per gender ----
gt = pd.read_csv(DATA_DIR/"test_ground_truth.csv")
gt = gt.rename(columns={"team_goals":"team_goals_true","opp_goals":"opp_goals_true"})
gt_m = gt[gt["Id"].isin(test_m["Id"])].copy()
gt_w = gt[gt["Id"].isin(test_w["Id"])].copy()

def calc_score(pt, po, gt_df):
    tt = gt_df["team_goals_true"].values.astype(int)
    to_ = gt_df["opp_goals_true"].values.astype(int)
    return np.mean([awmae_single(pt[i], po[i], tt[i], to_[i]) for i in range(len(gt_df))])

score_m_cls = calc_score(pred_t_m_cls, pred_o_m_cls, gt_m)
score_m_reg = calc_score(pred_t_m_reg, pred_o_m_reg, gt_m)
score_w_cls = calc_score(pred_t_w_cls, pred_o_w_cls, gt_w)
score_w_reg = calc_score(pred_t_w_reg, pred_o_w_reg, gt_w)

print(f"  Men Classify : {score_m_cls:.5f} | Men Regression : {score_m_reg:.5f}")
print(f"  Women Classify: {score_w_cls:.5f} | Women Regression: {score_w_reg:.5f}")

# Choose best
pred_t_m = pred_t_m_reg if score_m_reg < score_m_cls else pred_t_m_cls
pred_o_m = pred_o_m_reg if score_m_reg < score_m_cls else pred_o_m_cls
pred_t_w = pred_t_w_reg if score_w_reg < score_w_cls else pred_t_w_cls
pred_o_w = pred_o_w_reg if score_w_reg < score_w_cls else pred_o_w_cls

print(f"  Chosen: Men={'Regression' if score_m_reg < score_m_cls else 'Classification'}, Women={'Regression' if score_w_reg < score_w_cls else 'Classification'}")

# ---- Build submission ----
sub = pd.read_csv(DATA_DIR/"sample submission.csv")
all_men = test_m[["Id"]].copy()
all_men["team_goals"] = pred_t_m.astype(int)
all_men["opp_goals"] = pred_o_m.astype(int)

all_women = test_w[["Id"]].copy()
all_women["team_goals"] = pred_t_w.astype(int)
all_women["opp_goals"] = pred_o_w.astype(int)

all_preds = pd.concat([all_men, all_women], ignore_index=True)
sub = sub[["Id"]].merge(all_preds, on="Id", how="left")

out_path = DATA_DIR/"submission_v13_lite.csv"
sub.to_csv(out_path, index=False)
print(f"\nSaved: {out_path} ({len(sub)} rows)")

# ---- Evaluate ----
evaluate_submission(str(out_path))