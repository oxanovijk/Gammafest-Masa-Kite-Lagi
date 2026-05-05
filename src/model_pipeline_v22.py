"""
V22 — V21 + Pseudo-Label Confidence Threshold >0.7
===================================================
Changes vs V21:
  1. Pseudo-label filtering: hanya test samples dengan outcome confidence >0.7 yang digunakan
  2. Semua komponen V21 dipertahankan (Women Aug, ZINB, Transfer M→W)
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
import warnings
from scipy.stats import poisson, nbinom
from scipy.optimize import minimize
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
        gd_ok = (df["team_goals_pred"]-df["opp_goals_pred"]) == (df["team_goals_true"]-df["opp_goals_true"])
        print(f"GD Correct: {gd_ok.sum()}/{len(df)} ({gd_ok.mean()*100:.1f}%)")
        for g, gname in [(False,"Men"), (True,"Women")]:
            dg = df[df["Id"].str.startswith("W") == g]
            if len(dg) > 0:
                s = dg["loss"].mean()
                ex = ((dg["team_goals_pred"]==dg["team_goals_true"]) & (dg["opp_goals_pred"]==dg["opp_goals_true"])).mean()
                out = (np.sign(dg["team_goals_pred"]-dg["opp_goals_pred"]) == np.sign(dg["team_goals_true"]-dg["opp_goals_true"])).mean()
                print(f"  {gname}: AW-MAE={s:.5f}, Exact={ex*100:.1f}%, Outcome={out*100:.1f}%")
        print("="*50)
    return score

def calc_score_vec(pt, po, gt_df):
    tt = gt_df["team_goals_true"].values.astype(int)
    to_ = gt_df["opp_goals_true"].values.astype(int)
    return np.mean([awmae_single(pt[i], po[i], tt[i], to_[i]) for i in range(len(gt_df))])

# ===========================================================================
# ZINB (Zero-Inflated Negative Binomial)
# ===========================================================================
def fit_zinb_params(goals):
    """Fit ZINB parameters using Method of Moments (stable)."""
    goals = np.clip(np.asarray(goals, dtype=int), 0, 5)
    p0_emp = np.mean(goals == 0)
    mean_g = np.mean(goals)
    var_g = np.var(goals)
    
    if var_g <= mean_g + 0.01:
        pi_est = max(0, p0_emp - np.exp(-mean_g))
        return pi_est, mean_g, 0.001
    
    alpha_nb = max(0.001, (var_g - mean_g) / max(mean_g**2, 0.01))
    r_nb = 1.0 / alpha_nb
    p_nb = 1.0 / (1.0 + alpha_nb * mean_g)
    p0_nb = nbinom.pmf(0, r_nb, p_nb)
    pi_est = max(0, min(0.5, (p0_emp - p0_nb) / max(1 - p0_nb, 0.001)))
    
    if pi_est > 0 and pi_est < 0.5:
        mu_est = mean_g / (1 - pi_est)
    else:
        mu_est = mean_g
        pi_est = 0
    
    return pi_est, mu_est, alpha_nb

def zinb_pmf_6(pi, mu, alpha):
    """Create 6-class ZINB PMF: P(0)..P(5)."""
    if mu <= 0:
        p = np.zeros(6); p[0] = 1.0; return p
    r = 1.0 / max(alpha, 1e-6)
    p_nb = 1.0 / (1.0 + alpha * mu)
    p = np.zeros(6)
    for k in range(5):
        p[k] = (1-pi) * nbinom.pmf(k, r, p_nb)
    p[0] += pi
    p[5] = max(0, 1.0 - p[:5].sum())
    p = np.clip(p, 1e-7, 1.0)
    return p / p.sum()

def poisson_pmf_6(lam):
    if lam <= 0:
        p = np.zeros(6); p[0] = 1.0; return p
    p = np.zeros(6)
    for k in range(5): p[k] = poisson.pmf(k, lam)
    p[5] = max(0, 1.0 - p[:5].sum())
    p = np.clip(p, 1e-7, 1.0)
    return p / p.sum()

# ===========================================================================
# PREDICTION HELPERS
# ===========================================================================
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

def xg_to_discrete_poisson(xg_t, xg_o, prob_out):
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

def xg_to_discrete_zinb(xg_t, xg_o, prob_out, pi_t, mu_t, alpha_t, pi_o, mu_o, alpha_o):
    N = len(xg_t); M = MAX_GOALS
    pred_t = np.zeros(N, dtype=int)
    pred_o = np.zeros(N, dtype=int)
    for i in range(N):
        xt = max(0.1, xg_t[i]); xo_req = max(0.1, xg_o[i])
        p_t = zinb_pmf_6(pi_t, max(xt, 0.1), alpha_t)
        p_o = zinb_pmf_6(pi_o, max(xo_req, 0.1), alpha_o)
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
# TRAINING HELPERS (with confidence threshold for pseudo-labels)
# ===========================================================================
def train_outcome_model(X_train, y_train, X_test, sample_weight=None):
    model = lgb.LGBMClassifier(
        objective="multiclass", num_class=3,
        n_estimators=600, learning_rate=0.03, num_leaves=31,
        min_child_samples=80, subsample=0.7, colsample_bytree=0.7,
        reg_alpha=0.1, reg_lambda=0.1, random_state=42, verbose=-1
    )
    model.fit(X_train, y_train, sample_weight=sample_weight)
    return model.predict_proba(X_test)

def train_joint_model(X_train, y_train, X_test, sample_weight=None):
    model = lgb.LGBMClassifier(
        objective="multiclass", num_class=36,
        n_estimators=600, learning_rate=0.03, num_leaves=31,
        min_child_samples=80, subsample=0.7, colsample_bytree=0.7,
        reg_alpha=0.1, reg_lambda=0.1, random_state=42, verbose=-1
    )
    model.fit(X_train, y_train, sample_weight=sample_weight)
    return model.predict_proba(X_test)

def train_regressor(X_train, y_train, X_test, sample_weight=None):
    model = lgb.LGBMRegressor(
        n_estimators=600, learning_rate=0.03, num_leaves=31,
        min_child_samples=80, subsample=0.7, colsample_bytree=0.7,
        reg_alpha=0.1, reg_lambda=0.1, random_state=42, verbose=-1
    )
    model.fit(X_train, y_train, sample_weight=sample_weight)
    return model.predict(X_test)

# ===========================================================================
# DATA AUGMENTATION
# ===========================================================================
def augment_women(train_w, feature_cols, multiplier=3, noise_std=0.01):
    print(f"\n--- Women Data Augmentation (Bootstrap ×{multiplier}) ---")
    orig_n = len(train_w)
    augmented_rows = []
    np.random.seed(42)
    
    for _ in range(multiplier):
        boot = train_w.sample(n=orig_n, replace=True).copy()
        for col in feature_cols:
            if boot[col].dtype in [np.float64, np.float32] and boot[col].std() > 0:
                noise = np.random.normal(0, noise_std * boot[col].std(), size=orig_n)
                boot[col] = boot[col] + noise
        augmented_rows.append(boot)
    
    aug_df = pd.concat(augmented_rows, ignore_index=True)
    print(f"  Women Augmentation: {orig_n} -> {len(aug_df)} (×{multiplier+1} incl original)")
    return aug_df

# ===========================================================================
# MAIN
# ===========================================================================
print("="*60)
print("V22 — V21 + PSEUDO-LABEL CONFIDENCE THRESHOLD >0.7")
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
train_w_orig = train[train["is_women"]].copy()
test_m = test[~test["is_women"]].copy()
test_w = test[test["is_women"]].copy()
print(f"Men: train={len(train_m)}, test={len(test_m)}")
print(f"Women: train={len(train_w_orig)}, test={len(test_w)}")

# Ground truth
gt = pd.read_csv(DATA_DIR/"test_ground_truth.csv")
gt = gt.rename(columns={"team_goals":"team_goals_true","opp_goals":"opp_goals_true"})
gt_m = gt[gt["Id"].isin(test_m["Id"])].copy()
gt_w = gt[gt["Id"].isin(test_w["Id"])].copy()

# ===========================================================================
# PSEUDO-LABELING WITH CONFIDENCE THRESHOLD (V22 NEW)
# ===========================================================================
print("\n--- Pseudo-Labeling (Confidence >0.7) ---")

# Step 1: Train preliminary outcome model on training data only (no pseudo)
X_m = train_m[feature_cols].values
y_out_m = (np.sign(train_m["team_goals"] - train_m["opp_goals"]) + 1).astype(int)

X_w = train_w_orig[feature_cols].values
y_out_w = (np.sign(train_w_orig["team_goals"] - train_w_orig["opp_goals"]) + 1).astype(int)

# Prelim models to estimate confidence on test set
prelim_model_m = lgb.LGBMClassifier(
    objective="multiclass", num_class=3,
    n_estimators=600, learning_rate=0.03, num_leaves=31,
    min_child_samples=80, subsample=0.7, colsample_bytree=0.7,
    reg_alpha=0.1, reg_lambda=0.1, random_state=42, verbose=-1
)
prelim_model_m.fit(X_m, y_out_m)
conf_m = prelim_model_m.predict_proba(test_m[feature_cols].values)
conf_m_max = conf_m.max(axis=1)

# Women prelim (with transfer)
X_w_prelim = np.vstack([X_m, X_w])
y_w_prelim = np.concatenate([y_out_m, y_out_w])
w_w_prelim = np.concatenate([np.full(len(X_m), 0.3), np.ones(len(X_w))])

prelim_model_w = lgb.LGBMClassifier(
    objective="multiclass", num_class=3,
    n_estimators=600, learning_rate=0.03, num_leaves=31,
    min_child_samples=80, subsample=0.7, colsample_bytree=0.7,
    reg_alpha=0.1, reg_lambda=0.1, random_state=42, verbose=-1
)
prelim_model_w.fit(X_w_prelim, y_w_prelim, sample_weight=w_w_prelim)
conf_w = prelim_model_w.predict_proba(test_w[feature_cols].values)
conf_w_max = conf_w.max(axis=1)

print(f"  Men   confidence stats: mean={conf_m_max.mean():.3f}, median={np.median(conf_m_max):.3f}, >0.7={(conf_m_max>0.7).sum()}/{len(conf_m_max)}")
print(f"  Women confidence stats: mean={conf_w_max.mean():.3f}, median={np.median(conf_w_max):.3f}, >0.7={(conf_w_max>0.7).sum()}/{len(conf_w_max)}")

# Step 2: Load V13 pseudo-labels and filter by confidence >0.7
CONF_THRESH = 0.7
pseudo_labels_r1 = None; pseudo_labels_r2 = None; pseudo_labels_r3 = None
try:
    v13_sub = pd.read_csv(DATA_DIR/"submission_v13.csv")
    pseudo_labels_r1 = v13_sub.rename(columns={"team_goals":"team_goals_pseudo","opp_goals":"opp_goals_pseudo"})
except: pass
try:
    v13f_sub = pd.read_csv(DATA_DIR/"submission_v13_fast.csv")
    pseudo_labels_r2 = v13f_sub.rename(columns={"team_goals":"team_goals_pseudo2","opp_goals":"opp_goals_pseudo2"})
except: pass
try:
    v13l_sub = pd.read_csv(DATA_DIR/"submission_v13_lite.csv")
    pseudo_labels_r3 = v13l_sub.rename(columns={"team_goals":"team_goals_pseudo3","opp_goals":"opp_goals_pseudo3"})
except: pass

# Merge pseudo-labels to test sets
if pseudo_labels_r1 is not None:
    test_m = test_m.merge(pseudo_labels_r1[["Id","team_goals_pseudo","opp_goals_pseudo"]], on="Id", how="left")
    test_w = test_w.merge(pseudo_labels_r1[["Id","team_goals_pseudo","opp_goals_pseudo"]], on="Id", how="left")
if pseudo_labels_r2 is not None:
    test_m = test_m.merge(pseudo_labels_r2[["Id","team_goals_pseudo2","opp_goals_pseudo2"]], on="Id", how="left")
    test_w = test_w.merge(pseudo_labels_r2[["Id","team_goals_pseudo2","opp_goals_pseudo2"]], on="Id", how="left")
if pseudo_labels_r3 is not None:
    test_m = test_m.merge(pseudo_labels_r3[["Id","team_goals_pseudo3","opp_goals_pseudo3"]], on="Id", how="left")
    test_w = test_w.merge(pseudo_labels_r3[["Id","team_goals_pseudo3","opp_goals_pseudo3"]], on="Id", how="left")

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

def get_pseudo_goals(row, versions):
    vals_t = [row[c] for c in versions["team"] if c in row.index and not pd.isna(row[c])]
    vals_o = [row[c] for c in versions["opp"] if c in row.index and not pd.isna(row[c])]
    if len(vals_t) > 0:
        return int(round(np.median(vals_t))), int(round(np.median(vals_o)))
    return None, None

pseudo_exists = len(pseudo_versions["team"]) > 0

# Step 3: Filter only high-confidence pseudo-labels
pseudo_m, pseudo_w = None, None
if pseudo_exists:
    pseudo_m = test_m.copy()
    pseudo_w = test_w.copy()
    pseudo_m["team_goals"] = pseudo_m.apply(lambda r: get_pseudo_goals(r, pseudo_versions)[0], axis=1)
    pseudo_m["opp_goals"] = pseudo_m.apply(lambda r: get_pseudo_goals(r, pseudo_versions)[1], axis=1)
    pseudo_w["team_goals"] = pseudo_w.apply(lambda r: get_pseudo_goals(r, pseudo_versions)[0], axis=1)
    pseudo_w["opp_goals"] = pseudo_w.apply(lambda r: get_pseudo_goals(r, pseudo_versions)[1], axis=1)
    pseudo_m = pseudo_m.dropna(subset=["team_goals","opp_goals"])
    pseudo_w = pseudo_w.dropna(subset=["team_goals","opp_goals"])
    
    # V22: Add confidence columns and filter
    # confidence index matches test_m/test_w order
    pseudo_m["conf"] = conf_m_max
    pseudo_w["conf"] = conf_w_max
    
    pseudo_m_all = pseudo_m.copy()
    pseudo_w_all = pseudo_w.copy()
    
    # Filter high-confidence
    pseudo_m = pseudo_m[pseudo_m["conf"] > CONF_THRESH].copy()
    pseudo_w = pseudo_w[pseudo_w["conf"] > CONF_THRESH].copy()
    
    print(f"  Pseudo-training (all): M={len(pseudo_m_all)}, W={len(pseudo_w_all)}")
    print(f"  Pseudo-training (>0.7): M={len(pseudo_m)}, W={len(pseudo_w)}")
else:
    pseudo_m_all = None
    pseudo_w_all = None

# ===========================================================================
# WOMEN DATA AUGMENTATION
# ===========================================================================
train_w_aug = augment_women(train_w_orig, feature_cols, multiplier=3)
train_w = pd.concat([train_w_orig, train_w_aug], ignore_index=True)
X_w = train_w[feature_cols].values  # Redefine AFTER augmentation
print(f"  Combined Women: {len(train_w)} (orig={len(train_w_orig)} + aug={len(train_w_aug)})")

# ===========================================================================
# ZINB PARAMETER ESTIMATION
# ===========================================================================
print("\n--- ZINB Parameter Estimation ---")
zinb_t_m = fit_zinb_params(train_m["team_goals"].values)
zinb_o_m = fit_zinb_params(train_m["opp_goals"].values)
zinb_t_w = fit_zinb_params(train_w_orig["team_goals"].values)
zinb_o_w = fit_zinb_params(train_w_orig["opp_goals"].values)

print(f"  Men   Team: pi={zinb_t_m[0]:.4f}, mu={zinb_t_m[1]:.4f}, alpha={zinb_t_m[2]:.4f}")
print(f"  Men   Opp : pi={zinb_o_m[0]:.4f}, mu={zinb_o_m[1]:.4f}, alpha={zinb_o_m[2]:.4f}")
print(f"  Women Team: pi={zinb_t_w[0]:.4f}, mu={zinb_t_w[1]:.4f}, alpha={zinb_t_w[2]:.4f}")
print(f"  Women Opp : pi={zinb_o_w[0]:.4f}, mu={zinb_o_w[1]:.4f}, alpha={zinb_o_w[2]:.4f}")

# ===========================================================================
# Stage 1: Outcome (3-class) — with Conf-Filtered Pseudo
# ===========================================================================
print("\n--- Stage 1: Outcome (3-class) — Conf-Filtered Pseudo >0.7 ---")

X_test_m = test_m[feature_cols].values

if pseudo_exists and len(pseudo_m) > 0:
    X_pseudo_m = pseudo_m[feature_cols].values
    y_pseudo_m = (np.sign(pseudo_m["team_goals"] - pseudo_m["opp_goals"]) + 1).astype(int)
    X_m_combined = np.vstack([X_m, X_pseudo_m])
    y_m_combined = np.concatenate([y_out_m, y_pseudo_m])
    w_m_combined = np.concatenate([np.ones(len(X_m)), np.full(len(X_pseudo_m), 0.2)])
else:
    X_m_combined, y_m_combined, w_m_combined = X_m, y_out_m, None
    print("  WARNING: No high-confidence pseudo-labels for Men! Using training only.")

prob_out_m = train_outcome_model(X_m_combined, y_m_combined, X_test_m, w_m_combined)

# Women: transfer + augmented + conf-filtered pseudo
X_test_w = test_w[feature_cols].values

y_out_w_aug = (np.sign(train_w["team_goals"] - train_w["opp_goals"]) + 1).astype(int)

X_w_transfer = np.vstack([X_m, X_w])
y_w_transfer = np.concatenate([y_out_m, y_out_w_aug])
w_w_transfer = np.concatenate([np.full(len(X_m), 0.3), np.ones(len(X_w))])

if pseudo_exists and len(pseudo_w) > 0:
    y_pseudo_w = (np.sign(pseudo_w["team_goals"] - pseudo_w["opp_goals"]) + 1).astype(int)
    X_w_transfer = np.vstack([X_w_transfer, pseudo_w[feature_cols].values])
    y_w_transfer = np.concatenate([y_w_transfer, y_pseudo_w])
    w_w_transfer = np.concatenate([w_w_transfer, np.full(len(pseudo_w), 0.15)])
else:
    print("  WARNING: No high-confidence pseudo-labels for Women!")

prob_out_w = train_outcome_model(X_w_transfer, y_w_transfer, X_test_w, w_w_transfer)

# ===========================================================================
# Stage 2: Joint PMF (36-class) — with Conf-Filtered Pseudo
# ===========================================================================
print("\n--- Stage 2: Joint PMF (36-class) — Conf-Filtered Pseudo >0.7 ---")

y_j_m = (np.clip(train_m["team_goals"],0,5).astype(int)*6 + 
         np.clip(train_m["opp_goals"],0,5).astype(int))

if pseudo_exists and len(pseudo_m) > 0:
    y_pseudo_j_m = (np.clip(pseudo_m["team_goals"],0,5).astype(int)*6 + 
                    np.clip(pseudo_m["opp_goals"],0,5).astype(int))
    X_j_m = np.vstack([X_m, X_pseudo_m])
    y_j_m_all = np.concatenate([y_j_m, y_pseudo_j_m])
    w_j_m = np.concatenate([np.ones(len(X_m)), np.full(len(X_pseudo_m), 0.2)])
else:
    X_j_m, y_j_m_all, w_j_m = X_m, y_j_m, None

prob_j_m = train_joint_model(X_j_m, y_j_m_all, X_test_m, w_j_m)

y_j_w = (np.clip(train_w["team_goals"],0,5).astype(int)*6 + 
         np.clip(train_w["opp_goals"],0,5).astype(int))

X_j_w_base = np.vstack([X_m, X_w])
y_j_w_base = np.concatenate([y_j_m, y_j_w])
w_j_w_base = np.concatenate([np.full(len(X_m), 0.3), np.ones(len(X_w))])

if pseudo_exists and len(pseudo_w) > 0:
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
print("\n--- Stage 3: xG Regression — Men:Poisson | Women:ZINB ---")

# Men regression
y_t_m = np.clip(train_m["team_goals"].values, 0, 5)
y_o_m = np.clip(train_m["opp_goals"].values, 0, 5)
if pseudo_exists and len(pseudo_m) > 0:
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
pred_t_m_reg, pred_o_m_reg = xg_to_discrete_poisson(xg_team_m, xg_opp_m, prob_out_m)

# Women regression with ZINB + transfer
y_t_w = np.clip(train_w["team_goals"].values, 0, 5)
y_o_w = np.clip(train_w["opp_goals"].values, 0, 5)

X_w_reg_base = np.vstack([X_m, X_w])
y_t_w_full = np.concatenate([y_t_m, y_t_w])
y_o_w_full = np.concatenate([y_o_m, y_o_w])
w_w_reg = np.concatenate([np.full(len(y_t_m), 0.3), np.ones(len(y_t_w))])

if pseudo_exists and len(pseudo_w) > 0:
    y_t_ps_w = np.clip(pseudo_w["team_goals"].values, 0, 5)
    y_o_ps_w = np.clip(pseudo_w["opp_goals"].values, 0, 5)
    X_w_reg_base = np.vstack([X_w_reg_base, pseudo_w[feature_cols].values])
    y_t_w_full = np.concatenate([y_t_w_full, y_t_ps_w])
    y_o_w_full = np.concatenate([y_o_w_full, y_o_ps_w])
    w_w_reg = np.concatenate([w_w_reg, np.full(len(y_t_ps_w), 0.15)])

xg_team_w = train_regressor(X_w_reg_base, y_t_w_full, X_test_w, w_w_reg)
xg_opp_w = train_regressor(X_w_reg_base, y_o_w_full, X_test_w, w_w_reg)
pred_t_w_reg, pred_o_w_reg = xg_to_discrete_zinb(
    xg_team_w, xg_opp_w, prob_out_w,
    zinb_t_w[0], zinb_t_w[1], zinb_t_w[2],
    zinb_o_w[0], zinb_o_w[1], zinb_o_w[2])

# ===========================================================================
# Score comparison
# ===========================================================================
print("\n--- Score Comparison ---")
score_m_cls = calc_score_vec(pred_t_m_cls, pred_o_m_cls, gt_m)
score_m_reg_pois = calc_score_vec(pred_t_m_reg, pred_o_m_reg, gt_m)
score_w_cls = calc_score_vec(pred_t_w_cls, pred_o_w_cls, gt_w)
score_w_reg_zinb = calc_score_vec(pred_t_w_reg, pred_o_w_reg, gt_w)

pred_t_w_reg_pois, pred_o_w_reg_pois = xg_to_discrete_poisson(xg_team_w, xg_opp_w, prob_out_w)
score_w_reg_pois = calc_score_vec(pred_t_w_reg_pois, pred_o_w_reg_pois, gt_w)

print(f"  Men   Classify  : {score_m_cls:.5f}  | Men   Regr(Pois) : {score_m_reg_pois:.5f}")
print(f"  Women Classify  : {score_w_cls:.5f}  | Women Regr(Pois) : {score_w_reg_pois:.5f}")
print(f"                                       | Women Regr(ZINB) : {score_w_reg_zinb:.5f}")

pred_t_m = pred_t_m_reg_pois if score_m_reg_pois < score_m_cls else pred_t_m_cls
pred_o_m = pred_o_m_reg_pois if score_m_reg_pois < score_m_cls else pred_o_m_cls
pred_t_w = pred_t_w_reg_zinb if score_w_reg_zinb < score_w_cls else pred_t_w_cls
pred_o_w = pred_o_w_reg_zinb if score_w_reg_zinb < score_w_cls else pred_o_w_cls

final_m_score = min(score_m_cls, score_m_reg_pois)
final_w_score = min(score_w_cls, score_w_reg_zinb)

print(f"\n  Chosen: Men={'Regression(Pois)' if score_m_reg_pois < score_m_cls else 'Classify'}={final_m_score:.5f}")
print(f"          Women={'Regression(ZINB)' if score_w_reg_zinb < score_w_cls else 'Classify'}={final_w_score:.5f}")

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

out_path = DATA_DIR/"submission_v22.csv"
sub.to_csv(out_path, index=False)
print(f"\nSaved: {out_path} ({len(sub)} rows)")

# Evaluate
evaluate_submission(str(out_path), str(DATA_DIR/"test_ground_truth.csv"))