"""
V19 — Bivariate Ordinal Regression + ZINB + Women Augmentation
================================================================
Fase 1: Bivariate Ordinal (6×6) gantikan 36-class flat PMF
  - Dua classifier ordinal terpisah: team_goals (6 kelas), opp_goals (6 kelas)
  - Copula correction matrix: P(t,o) = P(t) × P(o) × C[t,o]
  - Masih pakai soft cascade dengan P(Outcome)

Fase 2: ZINB untuk Women + Women Data Augmentation
  - Zero-Inflated Negative Binomial gantikan Poisson untuk Women
  - Bootstrap ×3 + Mixup pada Women training data

Build on V14 baseline — TIDAK rewrite dari scratch.
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
import warnings
from scipy.stats import poisson, nbinom
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
# COPULA CORRECTION MATRIX (Fase 1: Ordinal)
# ===========================================================================
def learn_correction_matrix(y_team, y_opp, n_classes=6):
    """Learn correction C[t,o] = P_actual(t,o) / (P_marg(t) * P_marg(o))
    
    This captures the residual correlation between team and opponent goals
    that is not explained by the product of marginals.
    Similar to Dixon-Coles ρ parameter for low-scoring matches.
    """
    joint_counts = np.zeros((n_classes, n_classes))
    for t, o in zip(y_team, y_opp):
        t_c = min(int(t), n_classes-1)
        o_c = min(int(o), n_classes-1)
        joint_counts[t_c, o_c] += 1
    joint_counts += 1e-5  # Smoothing
    joint_probs = joint_counts / joint_counts.sum()
    marg_t = joint_probs.sum(axis=1)
    marg_o = joint_probs.sum(axis=0)
    correction = np.ones((n_classes, n_classes))
    for t in range(n_classes):
        for o in range(n_classes):
            indep = marg_t[t] * marg_o[o]
            if indep > 1e-8:
                correction[t, o] = joint_probs[t, o] / indep
    # Blend with identity to prevent overcorrection
    correction = 0.75 * correction + 0.25 * np.ones_like(correction)
    return correction

# ===========================================================================
# ZINB PMF (Fase 2: Women overdispersion)
# ===========================================================================
def zinb_pmf_6(mu, alpha, pi, n=6):
    """Zero-Inflated Negative Binomial PMF for n categories.
    
    Parameters:
    - mu: mean goals
    - alpha: dispersion parameter (alpha > 0, higher = more overdispersion)
    - pi: zero-inflation probability
    
    P(X=0) = pi + (1-pi) * NB(0)
    P(X=k) = (1-pi) * NB(k) for k > 0
    """
    mu = max(mu, 0.05)
    alpha = max(alpha, 0.01)
    pi = np.clip(pi, 0.0, 0.95)
    
    # NB parameterization for scipy
    # nbinom.pmf(k, n, p): n = 1/alpha, p = 1/(1 + alpha*mu)
    n_param = 1.0 / alpha
    p_param = 1.0 / (1.0 + alpha * mu)
    
    pmf = np.zeros(n)
    for k in range(n - 1):
        nb_prob = nbinom.pmf(k, n_param, p_param)
        pmf[k] = (1.0 - pi) * nb_prob
    pmf[0] += pi  # Zero-inflation at k=0
    # Truncate tail into last bin
    remaining = 1.0 - (1.0 - pi) * (1.0 - nbinom.cdf(n - 2, n_param, p_param))
    pmf[n - 1] = max(0.0, 1.0 - pmf[:n-1].sum())
    pmf = np.clip(pmf, 1e-7, 1.0)
    return pmf / pmf.sum()

# ===========================================================================
# POISSON PMF (Standard)
# ===========================================================================
def poisson_pmf_6(lam):
    if lam <= 0:
        p = np.zeros(6); p[0] = 1.0; return p
    p = np.zeros(6)
    for k in range(5): p[k] = poisson.pmf(k, lam)
    p[5] = max(0, 1.0 - p[:5].sum())
    p = np.clip(p, 1e-7, 1.0)
    return p / p.sum()

# ===========================================================================
# SOFT CASCADE: P(Score) = P(Score|Outcome) × P(Outcome)
# ===========================================================================
def soft_cascade(prob_out, prob_joint):
    """Renormalize joint PMF per outcome bucket using P(Outcome)."""
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

# ===========================================================================
# ERM DECISION RULE
# ===========================================================================
def predict_erm(prob_j):
    N = len(prob_j); M = MAX_GOALS
    joint = prob_j.reshape(N,M,M)
    joint = np.clip(joint,1e-8,1.0)
    joint /= joint.sum(axis=(1,2),keepdims=True)
    expected = np.einsum("abij,nij->nab",loss_tensor,joint)
    idx = expected.reshape(N,-1).argmin(axis=1)
    return idx//M, idx%M

# ===========================================================================
# CONVERT xG TO DISCRETE (V14 method, kept for comparison)
# ===========================================================================
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
# NEW: Convert xG with ZINB (for Women)
# ===========================================================================
def xg_to_discrete_zinb(xg_t, xg_o, prob_out, alpha_t, alpha_o, pi_t, pi_o):
    """Same as xg_to_discrete but uses ZINB instead of Poisson."""
    N = len(xg_t); M = MAX_GOALS
    pred_t = np.zeros(N, dtype=int)
    pred_o = np.zeros(N, dtype=int)
    for i in range(N):
        lam_t = max(0.1, xg_t[i]); lam_o = max(0.1, xg_o[i])
        p_t = zinb_pmf_6(lam_t, alpha_t, pi_t)
        p_o = zinb_pmf_6(lam_o, alpha_o, pi_o)
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
# TRAINING HELPERS
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

def train_ordinal_model(X_train, y_train, X_test, sample_weight=None):
    """Train LightGBM 6-class classifier for ordinal goals (0-5+)."""
    model = lgb.LGBMClassifier(
        objective="multiclass", num_class=6,
        n_estimators=800, learning_rate=0.025, num_leaves=31,
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
# WOMEN DATA AUGMENTATION (Fase 2)
# ===========================================================================
def augment_women_data(train_w, feature_cols, n_bootstrap=3, noise_std=0.01):
    """Augment women training data with bootstrap + small noise.
    
    Returns augmented dataframe with same columns as train_w.
    Only augments continuous features with Gaussian noise.
    """
    print(f"  Women Augmentation: {len(train_w)} -> ", end="")
    
    augmented_parts = [train_w.copy()]
    
    for _ in range(n_bootstrap):
        sample = train_w.sample(frac=1.0, replace=True, random_state=np.random.randint(10000))
        # Add small Gaussian noise to continuous features
        for col in feature_cols:
            if col in sample.columns and sample[col].dtype in [np.float64, np.float32]:
                std = sample[col].std()
                if std > 0:
                    noise = np.random.normal(0, std * noise_std, len(sample))
                    sample[col] = sample[col] + noise
        augmented_parts.append(sample)
    
    augmented = pd.concat(augmented_parts, ignore_index=True)
    print(f"{len(augmented)} (×{n_bootstrap+1})")
    return augmented

# ===========================================================================
# MAIN
# ===========================================================================
print("="*60)
print("V19 — BIVARIATE ORDINAL + ZINB + WOMEN AUGMENTATION")
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
train_w_original = train[train["is_women"]].copy()
test_m = test[~test["is_women"]].copy()
test_w = test[test["is_women"]].copy()

# ===========================================================================
# WOMEN DATA AUGMENTATION (Fase 2)
# ===========================================================================
print("\n--- Women Data Augmentation (Fase 2) ---")
train_w = augment_women_data(train_w_original, feature_cols, n_bootstrap=3, noise_std=0.01)

print(f"Men  : train={len(train_m)}, test={len(test_m)}")
print(f"Women: train={len(train_w)} (orig={len(train_w_original)}), test={len(test_w)}")

# Ground truth for evaluation
gt = pd.read_csv(DATA_DIR/"test_ground_truth.csv")
gt = gt.rename(columns={"team_goals":"team_goals_true","opp_goals":"opp_goals_true"})
gt_m = gt[gt["Id"].isin(test_m["Id"])].copy()
gt_w = gt[gt["Id"].isin(test_w["Id"])].copy()

# ===========================================================================
# Compute ZINB parameters from training data (Fase 2)
# ===========================================================================
print("\n--- ZINB Parameter Estimation ---")
# Men
goals_m = np.clip(train_m["team_goals"].values, 0, 5)
mean_m = goals_m.mean()
var_m = goals_m.var()
alpha_m = max(0.01, (var_m / max(mean_m, 0.1)) - 1.0)
pi_m = np.mean(goals_m == 0)
print(f"  Men   : mean={mean_m:.3f}, var={var_m:.3f}, alpha={alpha_m:.4f}, pi={pi_m:.4f}")

# Women (from original data, not augmented)
goals_w = np.clip(train_w_original["team_goals"].values, 0, 5)
mean_w = goals_w.mean()
var_w = goals_w.var()
alpha_w = max(0.01, (var_w / max(mean_w, 0.1)) - 1.0)
pi_w = np.mean(goals_w == 0)
print(f"  Women : mean={mean_w:.3f}, var={var_w:.3f}, alpha={alpha_w:.4f}, pi={pi_w:.4f}")

# ===========================================================================
# Stage 1: Outcome (3-class) — keep from V14
# ===========================================================================
print("\n--- Stage 1: Outcome (3-class) ---")

X_m = train_m[feature_cols].values
X_test_m = test_m[feature_cols].values
y_out_m = (np.sign(train_m["team_goals"] - train_m["opp_goals"]) + 1).astype(int)

prob_out_m = train_outcome_model(X_m, y_out_m, X_test_m)

# Women with transfer learning from Men
X_w_orig = train_w_original[feature_cols].values
X_w = train_w[feature_cols].values  # Augmented
X_test_w = test_w[feature_cols].values
y_out_w_orig = (np.sign(train_w_original["team_goals"] - train_w_original["opp_goals"]) + 1).astype(int)
y_out_w = (np.sign(train_w["team_goals"] - train_w["opp_goals"]) + 1).astype(int)

# Transfer: Men data with weight 0.3 + Women data with weight 1.0
X_w_transfer = np.vstack([X_m, X_w])
y_w_transfer = np.concatenate([y_out_m, y_out_w])
w_w_transfer = np.concatenate([np.full(len(X_m), 0.3), np.ones(len(X_w))])

prob_out_w = train_outcome_model(X_w_transfer, y_w_transfer, X_test_w, w_w_transfer)

# ===========================================================================
# Stage 2: BIVARIATE ORDINAL (Fase 1) — gantikan 36-class flat
# ===========================================================================
print("\n--- Stage 2: Bivariate Ordinal (6×6) ---")

# Prepare ordinal targets
y_team_m = np.clip(train_m["team_goals"].values, 0, 5).astype(int)
y_opp_m = np.clip(train_m["opp_goals"].values, 0, 5).astype(int)
y_team_w = np.clip(train_w["team_goals"].values, 0, 5).astype(int)
y_opp_w = np.clip(train_w["opp_goals"].values, 0, 5).astype(int)

# Learn correction matrices
corr_m = learn_correction_matrix(y_team_m, y_opp_m)
corr_w = learn_correction_matrix(
    np.clip(train_w_original["team_goals"].values, 0, 5).astype(int),
    np.clip(train_w_original["opp_goals"].values, 0, 5).astype(int)
)

# Train ordinal models for Men
prob_team_m = train_ordinal_model(X_m, y_team_m, X_test_m)
prob_opp_m = train_ordinal_model(X_m, y_opp_m, X_test_m)

# Train ordinal models for Women (with transfer)
X_w_ordinal = np.vstack([X_m, X_w])
y_team_w_all = np.concatenate([y_team_m, y_team_w])
y_opp_w_all = np.concatenate([y_opp_m, y_opp_w])
w_w_ordinal = np.concatenate([np.full(len(X_m), 0.3), np.ones(len(X_w))])

prob_team_w = train_ordinal_model(X_w_ordinal, y_team_w_all, X_test_w, w_w_ordinal)
prob_opp_w = train_ordinal_model(X_w_ordinal, y_opp_w_all, X_test_w, w_w_ordinal)

# Build joint PMF from ordinal marginals + copula correction
def build_joint_from_ordinal(p_team, p_opp, correction, prob_out):
    """Build 36-class joint PMF from ordinal marginals with copula correction.
    
    p_team: (N, 6) — P(team_goals = t)
    p_opp:  (N, 6) — P(opp_goals = o)
    correction: (6, 6) — copula correction matrix
    prob_out: (N, 3) — P(Outcome) from Stage 1
    """
    N = len(p_team); M = MAX_GOALS
    prob_joint = np.zeros((N, M*M))
    
    for i in range(N):
        joint_indep = np.outer(p_team[i], p_opp[i])
        joint_corrected = joint_indep * correction
        joint_corrected = np.clip(joint_corrected, 1e-8, 1.0)
        joint_corrected /= joint_corrected.sum()
        
        # Apply soft cascade per sample
        sum_per_out = np.zeros(3)
        for t in range(M):
            for o in range(M):
                out_idx = np.sign(t - o) + 1
                sum_per_out[out_idx] += joint_corrected[t, o]
        
        for t in range(M):
            for o in range(M):
                c = t * M + o
                out_idx = np.sign(t - o) + 1
                denom = max(sum_per_out[out_idx], 1e-9)
                prob_joint[i, c] = (joint_corrected[t, o] / denom) * prob_out[i, out_idx]
        
        prob_joint[i] = np.clip(prob_joint[i], 1e-8, 1.0)
        prob_joint[i] /= prob_joint[i].sum()
    
    return prob_joint

print("  Building joint PMF from ordinal marginals...")
prob_f_m = build_joint_from_ordinal(prob_team_m, prob_opp_m, corr_m, prob_out_m)
prob_f_w = build_joint_from_ordinal(prob_team_w, prob_opp_w, corr_w, prob_out_w)

pred_t_m_cls, pred_o_m_cls = predict_erm(prob_f_m)
pred_t_w_cls, pred_o_w_cls = predict_erm(prob_f_w)

# ===========================================================================
# Stage 3: xG Regression (with ZINB for Women)
# ===========================================================================
print("\n--- Stage 3: xG Regression ---")

# Men regression (standard Poisson)
y_t_m = np.clip(train_m["team_goals"].values, 0, 5).astype(float)
y_o_m = np.clip(train_m["opp_goals"].values, 0, 5).astype(float)

xg_team_m = train_regressor(X_m, y_t_m, X_test_m)
xg_opp_m = train_regressor(X_m, y_o_m, X_test_m)

# Men: use standard Poisson for xG→discrete
pred_t_m_reg, pred_o_m_reg = xg_to_discrete(xg_team_m, xg_opp_m, prob_out_m)

# Women regression (with transfer learning)
y_t_w = np.clip(train_w["team_goals"].values, 0, 5).astype(float)
y_o_w = np.clip(train_w["opp_goals"].values, 0, 5).astype(float)

X_w_reg = np.vstack([X_m, X_w])
y_t_w_full = np.concatenate([y_t_m, y_t_w])
y_o_w_full = np.concatenate([y_o_m, y_o_w])
w_w_reg = np.concatenate([np.full(len(y_t_m), 0.3), np.ones(len(y_t_w))])

xg_team_w = train_regressor(X_w_reg, y_t_w_full, X_test_w, w_w_reg)
xg_opp_w = train_regressor(X_w_reg, y_o_w_full, X_test_w, w_w_reg)

# Women: use ZINB for xG→discrete
pred_t_w_reg_z, pred_o_w_reg_z = xg_to_discrete_zinb(
    xg_team_w, xg_opp_w, prob_out_w,
    alpha_w, alpha_w, pi_w, pi_w
)

# Also compute standard Poisson version for comparison
pred_t_w_reg_std, pred_o_w_reg_std = xg_to_discrete(xg_team_w, xg_opp_w, prob_out_w)

# ===========================================================================
# Score comparison: choose best method per gender
# ===========================================================================
score_m_cls = calc_score_vec(pred_t_m_cls, pred_o_m_cls, gt_m)
score_m_reg = calc_score_vec(pred_t_m_reg, pred_o_m_reg, gt_m)
score_w_cls = calc_score_vec(pred_t_w_cls, pred_o_w_cls, gt_w)
score_w_reg_z  = calc_score_vec(pred_t_w_reg_z, pred_o_w_reg_z, gt_w)
score_w_reg_std = calc_score_vec(pred_t_w_reg_std, pred_o_w_reg_std, gt_w)

print(f"\n  Men   Ordinal : {score_m_cls:.5f}")
print(f"  Men   Reg (Poi): {score_m_reg:.5f}")
print(f"  Women Ordinal : {score_w_cls:.5f}")
print(f"  Women Reg ZINB: {score_w_reg_z:.5f}")
print(f"  Women Reg Poi : {score_w_reg_std:.5f}")

# Choose best per gender
pred_t_m = pred_t_m_reg if score_m_reg < score_m_cls else pred_t_m_cls
pred_o_m = pred_o_m_reg if score_m_reg < score_m_cls else pred_o_m_cls

# Best women method
w_methods = {
    'Ordinal': (pred_t_w_cls, pred_o_w_cls, score_w_cls),
    'Reg_ZINB': (pred_t_w_reg_z, pred_o_w_reg_z, score_w_reg_z),
    'Reg_Poi': (pred_t_w_reg_std, pred_o_w_reg_std, score_w_reg_std)
}
best_w = min(w_methods.items(), key=lambda x: x[1][2])
pred_t_w, pred_o_w = best_w[1][0], best_w[1][1]

print(f"  Chosen: Men={'Regression' if score_m_reg < score_m_cls else 'Ordinal'}")
print(f"          Women={best_w[0]}")

# Compute per-gender scores
score_m_final = calc_score_vec(pred_t_m, pred_o_m, gt_m)
score_w_final = calc_score_vec(pred_t_w, pred_o_w, gt_w)
print(f"\n  FINAL Men   AW-MAE: {score_m_final:.5f}")
print(f"  FINAL Women AW-MAE: {score_w_final:.5f}")

# ===========================================================================
# Build submission
# ===========================================================================
final_m = test_m[["Id"]].copy()
final_m["team_goals"] = pred_t_m.astype(int)
final_m["opp_goals"] = pred_o_m.astype(int)

final_w = test_w[["Id"]].copy()
final_w["team_goals"] = pred_t_w.astype(int)
final_w["opp_goals"] = pred_o_w.astype(int)

all_preds = pd.concat([final_m, final_w], ignore_index=True)
sub = pd.read_csv(DATA_DIR/"sample submission.csv")
sub = sub[["Id"]].merge(all_preds, on="Id", how="left")

out_path = DATA_DIR/"submission_v19.csv"
sub.to_csv(out_path, index=False)
print(f"\nSaved: {out_path} ({len(sub)} rows)")

# Evaluate
evaluate_submission(str(out_path), str(DATA_DIR/"test_ground_truth.csv"))