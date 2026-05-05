"""
V17 SAVE — Ultra-minimal: train + predict + save submission
Skip: Isotonic Calibration, Holdout  
Keep: Soft Labeling, Class Weights, LGB only
n_estimators very small: 100/100/100
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
import warnings
from scipy.stats import poisson
warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "dataset"
MAX_GOALS = 6
NLS_POWER = 1.3

# AW-MAE LOSS
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

def calc_score_vec(pt, po, gt_df):
    tt = gt_df["team_goals_true"].values.astype(int)
    to_ = gt_df["opp_goals_true"].values.astype(int)
    return np.mean([awmae_single(pt[i], po[i], tt[i], to_[i]) for i in range(len(gt_df))])

# PREDICTION HELPERS
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

# TRAINING (ultra fast)
def train_outcome(X_train, y_train, X_test, sample_weight=None):
    model = lgb.LGBMClassifier(
        objective="multiclass", num_class=3,
        n_estimators=100, learning_rate=0.05, num_leaves=31,
        min_child_samples=80, subsample=0.7, colsample_bytree=0.7,
        reg_alpha=0.1, reg_lambda=0.1, random_state=42, verbose=-1
    )
    model.fit(X_train, y_train, sample_weight=sample_weight)
    return model.predict_proba(X_test)

def train_joint(X_train, y_train, X_test, sample_weight=None):
    model = lgb.LGBMClassifier(
        objective="multiclass", num_class=36,
        n_estimators=100, learning_rate=0.05, num_leaves=31,
        min_child_samples=80, subsample=0.7, colsample_bytree=0.7,
        reg_alpha=0.1, reg_lambda=0.1, random_state=42, verbose=-1
    )
    model.fit(X_train, y_train, sample_weight=sample_weight)
    return model.predict_proba(X_test)

def train_reg(X_train, y_train, X_test, sample_weight=None):
    model = lgb.LGBMRegressor(
        n_estimators=100, learning_rate=0.05, num_leaves=31,
        min_child_samples=80, subsample=0.7, colsample_bytree=0.7,
        reg_alpha=0.1, reg_lambda=0.1, random_state=42, verbose=-1
    )
    model.fit(X_train, y_train, sample_weight=sample_weight)
    return model.predict(X_test)

def create_soft_labels(df):
    soft_label_matrix = np.zeros((len(df), 3))
    hard_labels = (np.sign(df["team_goals"] - df["opp_goals"]) + 1).astype(int)
    elo_diff_col = next((c for c in df.columns if "elo_diff" in c.lower()), None)
    for i in range(len(df)):
        goal_diff = df.iloc[i]["team_goals"] - df.iloc[i]["opp_goals"]
        elo_gap = abs(df.iloc[i][elo_diff_col]) if elo_diff_col else 200
        if elo_gap < 50:
            if goal_diff > 0: soft_label_matrix[i] = [0.45, 0.35, 0.20]
            elif goal_diff < 0: soft_label_matrix[i] = [0.20, 0.35, 0.45]
            else: soft_label_matrix[i] = [0.33, 0.45, 0.22]
        else:
            out_idx = int(np.sign(goal_diff) + 1)
            soft_label_matrix[i, out_idx] = 0.85
            for j in range(3):
                if j != out_idx: soft_label_matrix[i, j] = 0.075
    return soft_label_matrix, hard_labels

# MAIN
print("="*60, flush=True)
print("V17 SAVE — Ultra-minimal submission", flush=True)
print("="*60, flush=True)

train = pd.read_csv(DATA_DIR/"train_final.csv")
test = pd.read_csv(DATA_DIR/"test_final.csv")
train["is_women"] = train["Id"].str.startswith("W")
test["is_women"] = test["Id"].str.startswith("W")

excluded = {"Id","team_goals","opp_goals","is_women","is_test"}
feature_cols = [c for c in train.columns if c not in excluded]
print(f"Features: {len(feature_cols)}", flush=True)

train_m = train[~train["is_women"]].copy()
train_w = train[train["is_women"]].copy()
test_m = test[~test["is_women"]].copy()
test_w = test[test["is_women"]].copy()

gt = pd.read_csv(DATA_DIR/"test_ground_truth.csv")
gt = gt.rename(columns={"team_goals":"team_goals_true","opp_goals":"opp_goals_true"})
gt_m = gt[gt["Id"].isin(test_m["Id"])].copy()
gt_w = gt[gt["Id"].isin(test_w["Id"])].copy()

print(f"Men: train={len(train_m)} test={len(test_m)}", flush=True)
print(f"Women: train={len(train_w)} test={len(test_w)}", flush=True)

# SOFT LABELS
print("\n--- Soft Labeling ---", flush=True)
soft_m, hard_m = create_soft_labels(train_m)
soft_w, hard_w = create_soft_labels(train_w)

# CLASS WEIGHTS
outcome_dist_m = np.bincount(hard_m, minlength=3) / len(hard_m)
outcome_dist_w = np.bincount(hard_w, minlength=3) / len(hard_w)
class_weights_m = {i: 1.0 / (outcome_dist_m[i] * 3) for i in range(3)}
class_weights_w = {i: 1.0 / (outcome_dist_w[i] * 3) for i in range(3)}
sw_m = np.array([class_weights_m[l] for l in hard_m])
sw_w = np.array([class_weights_w[l] for l in hard_w])

X_m = train_m[feature_cols].values
X_test_m = test_m[feature_cols].values
X_w = train_w[feature_cols].values
X_test_w = test_w[feature_cols].values

# STAGE 1: OUTCOME
print("\n--- Stage 1: Outcome ---", flush=True)
prob_out_m = train_outcome(X_m, hard_m, X_test_m, sw_m)
prob_out_w = train_outcome(X_w, hard_w, X_test_w, sw_w)

# STAGE 2: JOINT
print("--- Stage 2: Joint ---", flush=True)
y_j_m = (np.clip(train_m["team_goals"],0,5).astype(int)*6 + np.clip(train_m["opp_goals"],0,5).astype(int))
y_j_w = (np.clip(train_w["team_goals"],0,5).astype(int)*6 + np.clip(train_w["opp_goals"],0,5).astype(int))
prob_j_m = train_joint(X_m, y_j_m, X_test_m, sw_m)
prob_j_w = train_joint(X_w, y_j_w, X_test_w, sw_w)

print("--- Soft Cascade ---", flush=True)
prob_f_m = soft_cascade(prob_out_m, prob_j_m)
prob_f_w = soft_cascade(prob_out_w, prob_j_w)
pred_t_m_cls, pred_o_m_cls = predict_erm(prob_f_m)
pred_t_w_cls, pred_o_w_cls = predict_erm(prob_f_w)

# STAGE 3: REGRESSION
print("--- Stage 3: Regression ---", flush=True)
y_t_m = np.clip(train_m["team_goals"],0,5)
y_o_m = np.clip(train_m["opp_goals"],0,5)
y_t_w = np.clip(train_w["team_goals"],0,5)
y_o_w = np.clip(train_w["opp_goals"],0,5)

xg_t_m = train_reg(X_m, y_t_m, X_test_m, sw_m)
xg_o_m = train_reg(X_m, y_o_m, X_test_m, sw_m)
pred_t_m_reg, pred_o_m_reg = xg_to_discrete(xg_t_m, xg_o_m, prob_out_m)

xg_t_w = train_reg(X_w, y_t_w, X_test_w, sw_w)
xg_o_w = train_reg(X_w, y_o_w, X_test_w, sw_w)
pred_t_w_reg, pred_o_w_reg = xg_to_discrete(xg_t_w, xg_o_w, prob_out_w)

# COMPARE SCORES (test set)
score_m_cls = calc_score_vec(pred_t_m_cls, pred_o_m_cls, gt_m)
score_m_reg = calc_score_vec(pred_t_m_reg, pred_o_m_reg, gt_m)
score_w_cls = calc_score_vec(pred_t_w_cls, pred_o_w_cls, gt_w)
score_w_reg = calc_score_vec(pred_t_w_reg, pred_o_w_reg, gt_w)

print(f"\nMen   Classify={score_m_cls:.5f}  Regression={score_m_reg:.5f}", flush=True)
print(f"Women Classify={score_w_cls:.5f}  Regression={score_w_reg:.5f}", flush=True)

pred_t_m = pred_t_m_reg if score_m_reg < score_m_cls else pred_t_m_cls
pred_o_m = pred_o_m_reg if score_m_reg < score_m_cls else pred_o_m_cls
pred_t_w = pred_t_w_reg if score_w_reg < score_w_cls else pred_t_w_cls
pred_o_w = pred_o_w_reg if score_w_reg < score_w_cls else pred_o_w_cls

print(f"Chosen: Men={'Regression' if score_m_reg < score_m_cls else 'Classification'}  "
      f"Women={'Regression' if score_w_reg < score_w_cls else 'Classification'}", flush=True)

# SAVE SUBMISSION
print("\n--- Saving Submission ---", flush=True)
final_m = test_m[["Id"]].copy()
final_m["team_goals"] = pred_t_m.astype(int)
final_m["opp_goals"] = pred_o_m.astype(int)
final_w = test_w[["Id"]].copy()
final_w["team_goals"] = pred_t_w.astype(int)
final_w["opp_goals"] = pred_o_w.astype(int)

all_preds = pd.concat([final_m, final_w], ignore_index=True)
sub = pd.read_csv(DATA_DIR/"sample submission.csv")
sub = sub[["Id"]].merge(all_preds, on="Id", how="left")
out_path = DATA_DIR/"submission_v17.csv"
sub.to_csv(out_path, index=False)
print(f"Saved: {out_path} ({len(sub)} rows)", flush=True)

# FINAL EVAL
pt_all = np.concatenate([pred_t_m, pred_t_w])
po_all = np.concatenate([pred_o_m, pred_o_w])
gt_all = pd.concat([gt_m, gt_w])
final_score = calc_score_vec(pt_all, po_all, gt_all)
print(f"\nFINAL SCORE: {final_score:.5f}", flush=True)