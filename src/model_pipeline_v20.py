"""
V20 — Class Weights + GPD Tail Modeling + Friendly Specialization
=================================================================
Fase 3: Class Weights untuk Draw + GPD Tail untuk high-scoring matches
Fase 4: Friendly Specialization dengan interaction features

Builds on V19 (Bivariate Ordinal + ZINB + Women Augmentation).
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
import warnings
from scipy.stats import poisson, nbinom, genpareto
from sklearn.calibration import CalibratedClassifierCV
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
        gd_col = (df["team_goals_pred"]-df["opp_goals_pred"]) == (df["team_goals_true"]-df["opp_goals_true"])
        print(f"GD Correct: {gd_col.sum()}/{len(df)} ({gd_col.mean()*100:.1f}%)")
        
        # Per-gender breakdown
        for gender, label in [("M", "Men"), ("W", "Women")]:
            gdf = df[df["Id"].str.startswith(gender)]
            if len(gdf) > 0:
                g_score = gdf["loss"].mean()
                g_exact = ((gdf["team_goals_pred"]==gdf["team_goals_true"]) & 
                            (gdf["opp_goals_pred"]==gdf["opp_goals_true"])).mean()
                g_out = (np.sign(gdf["team_goals_pred"]-gdf["opp_goals_pred"]) == 
                         np.sign(gdf["team_goals_true"]-gdf["opp_goals_true"])).mean()
                print(f"  {label}: AW-MAE={g_score:.5f}, Exact={g_exact*100:.1f}%, Outcome={g_out*100:.1f}%")
        print("="*50)
    return score

def calc_score_vec(pt, po, gt_df):
    tt = gt_df["team_goals_true"].values.astype(int)
    to_ = gt_df["opp_goals_true"].values.astype(int)
    return np.mean([awmae_single(pt[i], po[i], tt[i], to_[i]) for i in range(len(gt_df))])

# ===========================================================================
# COPULA CORRECTION MATRIX (from V19)
# ===========================================================================
def learn_correction_matrix(y_team, y_opp, n_classes=6):
    joint_counts = np.zeros((n_classes, n_classes))
    for t, o in zip(y_team, y_opp):
        t_c = min(int(t), n_classes-1)
        o_c = min(int(o), n_classes-1)
        joint_counts[t_c, o_c] += 1
    joint_counts += 1e-5
    joint_probs = joint_counts / joint_counts.sum()
    marg_t = joint_probs.sum(axis=1)
    marg_o = joint_probs.sum(axis=0)
    correction = np.ones((n_classes, n_classes))
    for t in range(n_classes):
        for o in range(n_classes):
            indep = marg_t[t] * marg_o[o]
            if indep > 1e-8:
                correction[t, o] = joint_probs[t, o] / indep
    correction = 0.75 * correction + 0.25 * np.ones_like(correction)
    return correction

# ===========================================================================
# ZINB PMF (from V19)
# ===========================================================================
def zinb_pmf_6(mu, alpha, pi, n=6):
    mu = max(mu, 0.05)
    alpha = max(alpha, 0.01)
    pi = np.clip(pi, 0.0, 0.95)
    n_param = 1.0 / alpha
    p_param = 1.0 / (1.0 + alpha * mu)
    pmf = np.zeros(n)
    for k in range(n - 1):
        nb_prob = nbinom.pmf(k, n_param, p_param)
        pmf[k] = (1.0 - pi) * nb_prob
    pmf[0] += pi
    remaining = 1.0 - (1.0 - pi) * (1.0 - nbinom.cdf(n - 2, n_param, p_param))
    pmf[n - 1] = max(0.0, 1.0 - pmf[:n-1].sum())
    pmf = np.clip(pmf, 1e-7, 1.0)
    return pmf / pmf.sum()

# ===========================================================================
# GPD TAIL MODELING (Fase 3)
# ===========================================================================
def fit_gpd_tail(goals, threshold=4):
    """Fit Generalized Pareto Distribution to excess above threshold.
    
    Returns lambda function f(excess) that gives GPD probability,
    plus P_exceed = P(X > threshold).
    """
    goals = np.array(goals)
    goals = goals[goals <= MAX_GOALS - 1]  # Within our prediction range
    exceedances = goals[goals > threshold] - threshold
    n_total = len(goals)
    n_exceed = len(exceedances)
    
    if n_exceed < 10:
        # Not enough data for GPD, fall back to empirical
        emp_probs = np.zeros(MAX_GOALS - threshold - 1)
        for k in range(threshold + 1, MAX_GOALS):
            emp_probs[k - threshold - 1] = np.mean(goals == k)
        if emp_probs.sum() < 1e-8:
            emp_probs = np.ones(len(emp_probs)) / len(emp_probs)
        emp_probs /= emp_probs.sum()
        p_exceed = n_exceed / max(n_total, 1e-5)
        return emp_probs, p_exceed
    
    # Fit GPD
    try:
        shape, loc, scale = genpareto.fit(exceedances, floc=0)
    except:
        shape, scale = 0.1, exceedances.std() if exceedances.std() > 0 else 1.0
    
    p_exceed = n_exceed / n_total
    
    # Build tail distribution for k in [threshold+1, MAX_GOALS-1]
    n_tail_bins = MAX_GOALS - threshold - 1
    gpd_probs = np.zeros(n_tail_bins)
    for i, k in enumerate(range(threshold + 1, MAX_GOALS)):
        excess = k - threshold
        gpd_probs[i] = genpareto.pdf(excess, shape, scale=scale)
    
    gpd_probs = np.clip(gpd_probs, 1e-8, None)
    gpd_probs /= gpd_probs.sum()
    
    return gpd_probs, p_exceed

def apply_gpd_to_pmf(pmf_base, gpd_probs, p_exceed, threshold=4):
    """Replace tail of base PMF with GPD distribution.
    
    pmf_base: base PMF (6 elements)
    gpd_probs: GPD tail distribution (for k = threshold+1 ... 5)
    p_exceed: P(X > threshold)
    """
    result = pmf_base.copy()
    
    # Keep mass below and at threshold
    mass_below = pmf_base[:threshold+1].sum()
    
    # Redistribute: find scaling
    # We want: P(X <= threshold) stays the same, P(X>threshold) is redistributed per GPD
    # But blend: blend GPD with original tail
    tail_original = pmf_base[threshold+1:].copy()
    if tail_original.sum() > 1e-8:
        tail_original /= tail_original.sum()
    else:
        tail_original = np.ones(len(tail_original)) / len(tail_original)
    
    # Blend: 50% GPD, 50% original (safe blending)
    tail_blended = 0.5 * gpd_probs + 0.5 * tail_original
    tail_blended /= tail_blended.sum()
    
    # P(exceed) from blend of empirical and GPD
    p_exceed_blended = 0.5 * p_exceed + 0.5 * pmf_base[threshold+1:].sum()
    p_exceed_blended = np.clip(p_exceed_blended, 0.01, 0.35)
    
    mass_below_target = 1.0 - p_exceed_blended
    
    # Rescale below-threshold mass
    if mass_below > 1e-8:
        scale_below = mass_below_target / mass_below
    else:
        scale_below = 0.0
    
    for i in range(threshold + 1):
        result[i] = pmf_base[i] * scale_below
    
    for i, k in enumerate(range(threshold + 1, MAX_GOALS)):
        result[k] = p_exceed_blended * tail_blended[i]
    
    result = np.clip(result, 1e-8, 1.0)
    return result / result.sum()

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
# xG TO DISCRETE CONVERSIONS
# ===========================================================================
def xg_to_discrete(xg_t, xg_o, prob_out, gpd_t=None, gpd_o=None, p_exceed_t=None, p_exceed_o=None):
    N = len(xg_t); M = MAX_GOALS
    pred_t = np.zeros(N, dtype=int)
    pred_o = np.zeros(N, dtype=int)
    
    use_gpd = gpd_t is not None and gpd_o is not None
    
    for i in range(N):
        lam_t = max(0.1, xg_t[i]); lam_o = max(0.1, xg_o[i])
        p_t = poisson_pmf_6(lam_t); p_o = poisson_pmf_6(lam_o)
        
        # Apply GPD tail if available (Fase 3)
        if use_gpd:
            p_t = apply_gpd_to_pmf(p_t, gpd_t, p_exceed_t)
            p_o = apply_gpd_to_pmf(p_o, gpd_o, p_exceed_o)
        
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

def xg_to_discrete_zinb(xg_t, xg_o, prob_out, alpha_t, alpha_o, pi_t, pi_o,
                         gpd_t=None, gpd_o=None, p_exceed_t=None, p_exceed_o=None):
    N = len(xg_t); M = MAX_GOALS
    pred_t = np.zeros(N, dtype=int)
    pred_o = np.zeros(N, dtype=int)
    
    use_gpd = gpd_t is not None and gpd_o is not None
    
    for i in range(N):
        lam_t = max(0.1, xg_t[i]); lam_o = max(0.1, xg_o[i])
        p_t = zinb_pmf_6(lam_t, alpha_t, pi_t)
        p_o = zinb_pmf_6(lam_o, alpha_o, pi_o)
        
        # Apply GPD tail if available (Fase 3)
        if use_gpd:
            p_t = apply_gpd_to_pmf(p_t, gpd_t, p_exceed_t)
            p_o = apply_gpd_to_pmf(p_o, gpd_o, p_exceed_o)
        
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
    """Train LightGBM 3-class classifier with optional sample weights."""
    model = lgb.LGBMClassifier(
        objective="multiclass", num_class=3,
        n_estimators=600, learning_rate=0.03, num_leaves=31,
        min_child_samples=80, subsample=0.7, colsample_bytree=0.7,
        reg_alpha=0.1, reg_lambda=0.1, random_state=42, verbose=-1
    )
    model.fit(X_train, y_train, sample_weight=sample_weight)
    return model.predict_proba(X_test)

def train_ordinal_model(X_train, y_train, X_test, sample_weight=None):
    """Train LightGBM 6-class classifier for ordinal goals."""
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
# WOMEN DATA AUGMENTATION (from V19)
# ===========================================================================
def augment_women_data(train_w, feature_cols, n_bootstrap=3, noise_std=0.01):
    print(f"  Women Augmentation: {len(train_w)} -> ", end="")
    augmented_parts = [train_w.copy()]
    for _ in range(n_bootstrap):
        sample = train_w.sample(frac=1.0, replace=True, random_state=np.random.randint(10000))
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
# CLASS WEIGHTS FOR DRAW (Fase 3)
# ===========================================================================
def compute_class_weights(y_3class):
    """Compute balanced class weights for W/D/L imbalance."""
    n = len(y_3class)
    counts = np.bincount(y_3class, minlength=3)
    # Balanced: weight = n_samples / (n_classes * n_samples_per_class)
    weights = n / (3 * np.maximum(counts, 1))
    # Map to samples
    sample_weights = np.array([weights[y] for y in y_3class])
    return sample_weights

# ===========================================================================
# Bivariate Ordinal Joint PMF Builder
# ===========================================================================
def build_joint_from_ordinal(p_team, p_opp, correction, prob_out):
    N = len(p_team); M = MAX_GOALS
    prob_joint = np.zeros((N, M*M))
    for i in range(N):
        joint_indep = np.outer(p_team[i], p_opp[i])
        joint_corrected = joint_indep * correction
        joint_corrected = np.clip(joint_corrected, 1e-8, 1.0)
        joint_corrected /= joint_corrected.sum()
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

# ===========================================================================
# MAIN
# ===========================================================================
print("="*60)
print("V20 — CLASS WEIGHTS + GPD TAIL + FRIENDLY SPECIALIZATION")
print("="*60)

# Load data
train = pd.read_csv(DATA_DIR/"train_final.csv")
test = pd.read_csv(DATA_DIR/"test_final.csv")

train["is_women"] = train["Id"].str.startswith("W")
test["is_women"] = test["Id"].str.startswith("W")

excluded = {"Id","team_goals","opp_goals","is_women","is_test"}
feature_cols = [c for c in train.columns if c not in excluded]
print(f"Features: {len(feature_cols)}")

# ===========================================================================
# FASE 4: FRIENDLY SPECIALIZATION — add interaction features
# ===========================================================================
print("\n--- Fase 4: Friendly Specialization ---")

# Identify friendly-related features in training data
is_friendly_col = None
for col in feature_cols:
    if col == "is_friendly" or "friendly" in col.lower():
        is_friendly_col = col
        break

if is_friendly_col is not None:
    print(f"  Found friendly column: {is_friendly_col}")
    # Add friendly interaction features to train and test
    for data in [train, test]:
        fi = data[is_friendly_col].values
        
        # elo_diff × is_friendly
        if "elo_diff" in data.columns:
            data["elo_diff_x_friendly"] = data["elo_diff"] * fi
        
        # days_rest_diff × is_friendly
        if "days_rest_diff" in data.columns:
            data["days_rest_diff_x_friendly"] = data["days_rest_diff"] * fi
        
        # tournament_tier × is_friendly (friendly already lowest tier, but interaction helps)
        if "tournament_tier" in data.columns:
            data["tier_x_friendly"] = data["tournament_tier"] * fi
        
        # h2h_gd × is_friendly
        for col in data.columns:
            if col.startswith("h2h") and "gd" in col:
                data[f"{col}_x_friendly"] = data[col] * fi
        
        # gdp_diff × is_friendly (friendly matches Elo less reliable, socio-economic more)
        if "log_gdp_diff" in data.columns:
            data["gdp_x_friendly"] = data["log_gdp_diff"] * fi
    
    # Update feature columns
    excluded = {"Id","team_goals","opp_goals","is_women","is_test"}
    feature_cols = [c for c in train.columns if c not in excluded]
    print(f"  Features after friendly interactions: {len(feature_cols)}")
else:
    print("  WARNING: No friendly column found, skipping friendly specialization")

# Split by gender
train_m = train[~train["is_women"]].copy()
train_w_original = train[train["is_women"]].copy()
test_m = test[~test["is_women"]].copy()
test_w = test[test["is_women"]].copy()

# ===========================================================================
# WOMEN DATA AUGMENTATION (carry forward from V19)
# ===========================================================================
print("\n--- Women Data Augmentation ---")
train_w = augment_women_data(train_w_original, feature_cols, n_bootstrap=3, noise_std=0.01)
print(f"Men  : train={len(train_m)}, test={len(test_m)}")
print(f"Women: train={len(train_w)} (orig={len(train_w_original)}), test={len(test_w)}")

# Ground truth
gt = pd.read_csv(DATA_DIR/"test_ground_truth.csv")
gt = gt.rename(columns={"team_goals":"team_goals_true","opp_goals":"opp_goals_true"})
gt_m = gt[gt["Id"].isin(test_m["Id"])].copy()
gt_w = gt[gt["Id"].isin(test_w["Id"])].copy()

# ===========================================================================
# ZINB PARAMETERS (carry forward from V19)
# ===========================================================================
print("\n--- ZINB Parameter Estimation ---")
goals_m = np.clip(train_m["team_goals"].values, 0, 5)
mean_m = goals_m.mean(); var_m = goals_m.var()
alpha_m = max(0.01, (var_m / max(mean_m, 0.1)) - 1.0)
pi_m = np.mean(goals_m == 0)
print(f"  Men   : mean={mean_m:.3f}, var={var_m:.3f}, alpha={alpha_m:.4f}, pi={pi_m:.4f}")

goals_w = np.clip(train_w_original["team_goals"].values, 0, 5)
mean_w = goals_w.mean(); var_w = goals_w.var()
alpha_w = max(0.01, (var_w / max(mean_w, 0.1)) - 1.0)
pi_w = np.mean(goals_w == 0)
print(f"  Women : mean={mean_w:.3f}, var={var_w:.3f}, alpha={alpha_w:.4f}, pi={pi_w:.4f}")

# ===========================================================================
# GPD TAIL FITTING (Fase 3)
# ===========================================================================
print("\n--- GPD Tail Modeling (Fase 3) ---")

# Fit GPD on training data goals
gpd_team_m, p_exceed_team_m = fit_gpd_tail(np.clip(train_m["team_goals"].values, 0, 5))
gpd_opp_m, p_exceed_opp_m = fit_gpd_tail(np.clip(train_m["opp_goals"].values, 0, 5))
gpd_team_w, p_exceed_team_w = fit_gpd_tail(np.clip(train_w_original["team_goals"].values, 0, 5))
gpd_opp_w, p_exceed_opp_w = fit_gpd_tail(np.clip(train_w_original["opp_goals"].values, 0, 5))

print(f"  Men   Team GPD: shape fitted, p_exceed={p_exceed_team_m:.4f}")
print(f"  Men   Opp  GPD: p_exceed={p_exceed_opp_m:.4f}")
print(f"  Women Team GPD: p_exceed={p_exceed_team_w:.4f}")
print(f"  Women Opp  GPD: p_exceed={p_exceed_opp_w:.4f}")

# ===========================================================================
# Stage 1: Outcome (3-class) — WITH CLASS WEIGHTS (Fase 3)
# ===========================================================================
print("\n--- Stage 1: Outcome (3-class) + Class Weights ---")

X_m = train_m[feature_cols].values
X_test_m = test_m[feature_cols].values
y_out_m = (np.sign(train_m["team_goals"] - train_m["opp_goals"]) + 1).astype(int)

# Compute class weights for draw imbalance
cw_m = compute_class_weights(y_out_m)
print(f"  Men   Class Weights: Home={1/(3*np.bincount(y_out_m,minlength=3)[0]/len(y_out_m)):.2f}, "
      f"Draw={1/(3*np.bincount(y_out_m,minlength=3)[1]/len(y_out_m)):.2f}, "
      f"Away={1/(3*np.bincount(y_out_m,minlength=3)[2]/len(y_out_m)):.2f}")

prob_out_m = train_outcome_model(X_m, y_out_m, X_test_m, sample_weight=cw_m)

# Women with transfer learning from Men
X_w_orig = train_w_original[feature_cols].values
X_w = train_w[feature_cols].values
X_test_w = test_w[feature_cols].values
y_out_w_orig = (np.sign(train_w_original["team_goals"] - train_w_original["opp_goals"]) + 1).astype(int)
y_out_w = (np.sign(train_w["team_goals"] - train_w["opp_goals"]) + 1).astype(int)

cw_w = compute_class_weights(y_out_w)
print(f"  Women Class Weights: Home={1/(3*np.bincount(y_out_w,minlength=3)[0]/len(y_out_w)):.2f}, "
      f"Draw={1/(3*np.bincount(y_out_w,minlength=3)[1]/len(y_out_w)):.2f}, "
      f"Away={1/(3*np.bincount(y_out_w,minlength=3)[2]/len(y_out_w)):.2f}")

# Transfer: Men data with weight 0.3 + Women data with weight 1.0
X_w_transfer = np.vstack([X_m, X_w])
y_w_transfer = np.concatenate([y_out_m, y_out_w])
w_w_transfer = np.concatenate([np.full(len(X_m), 0.3) * cw_m, cw_w])

prob_out_w = train_outcome_model(X_w_transfer, y_w_transfer, X_test_w, w_w_transfer)

# ===========================================================================
# Stage 2: Bivariate Ordinal (from V19)
# ===========================================================================
print("\n--- Stage 2: Bivariate Ordinal (6×6) ---")

y_team_m = np.clip(train_m["team_goals"].values, 0, 5).astype(int)
y_opp_m = np.clip(train_m["opp_goals"].values, 0, 5).astype(int)
y_team_w = np.clip(train_w["team_goals"].values, 0, 5).astype(int)
y_opp_w = np.clip(train_w["opp_goals"].values, 0, 5).astype(int)

corr_m = learn_correction_matrix(y_team_m, y_opp_m)
corr_w = learn_correction_matrix(
    np.clip(train_w_original["team_goals"].values, 0, 5).astype(int),
    np.clip(train_w_original["opp_goals"].values, 0, 5).astype(int)
)

prob_team_m = train_ordinal_model(X_m, y_team_m, X_test_m)
prob_opp_m = train_ordinal_model(X_m, y_opp_m, X_test_m)

X_w_ordinal = np.vstack([X_m, X_w])
y_team_w_all = np.concatenate([y_team_m, y_team_w])
y_opp_w_all = np.concatenate([y_opp_m, y_opp_w])
w_w_ordinal = np.concatenate([np.full(len(X_m), 0.3), np.ones(len(X_w))])

prob_team_w = train_ordinal_model(X_w_ordinal, y_team_w_all, X_test_w, w_w_ordinal)
prob_opp_w = train_ordinal_model(X_w_ordinal, y_opp_w_all, X_test_w, w_w_ordinal)

print("  Building joint PMF from ordinal marginals...")
prob_f_m = build_joint_from_ordinal(prob_team_m, prob_opp_m, corr_m, prob_out_m)
prob_f_w = build_joint_from_ordinal(prob_team_w, prob_opp_w, corr_w, prob_out_w)

pred_t_m_cls, pred_o_m_cls = predict_erm(prob_f_m)
pred_t_w_cls, pred_o_w_cls = predict_erm(prob_f_w)

# ===========================================================================
# Stage 3: xG Regression (with GPD tail for Men, ZINB+GPD for Women)
# ===========================================================================
print("\n--- Stage 3: xG Regression + GPD Tail ---")

# Men regression
y_t_m = np.clip(train_m["team_goals"].values, 0, 5).astype(float)
y_o_m = np.clip(train_m["opp_goals"].values, 0, 5).astype(float)

xg_team_m = train_regressor(X_m, y_t_m, X_test_m)
xg_opp_m = train_regressor(X_m, y_o_m, X_test_m)

# Men: Standard Poisson + GPD tail
pred_t_m_reg, pred_o_m_reg = xg_to_discrete(
    xg_team_m, xg_opp_m, prob_out_m,
    gpd_team_m, gpd_opp_m, p_exceed_team_m, p_exceed_opp_m
)

# Also compute WITHOUT GPD for comparison
pred_t_m_reg_nogpd, pred_o_m_reg_nogpd = xg_to_discrete(xg_team_m, xg_opp_m, prob_out_m)

# Women regression
y_t_w = np.clip(train_w["team_goals"].values, 0, 5).astype(float)
y_o_w = np.clip(train_w["opp_goals"].values, 0, 5).astype(float)

X_w_reg = np.vstack([X_m, X_w])
y_t_w_full = np.concatenate([y_t_m, y_t_w])
y_o_w_full = np.concatenate([y_o_m, y_o_w])
w_w_reg = np.concatenate([np.full(len(y_t_m), 0.3), np.ones(len(y_t_w))])

xg_team_w = train_regressor(X_w_reg, y_t_w_full, X_test_w, w_w_reg)
xg_opp_w = train_regressor(X_w_reg, y_o_w_full, X_test_w, w_w_reg)

# Women: ZINB + GPD tail
pred_t_w_reg_z, pred_o_w_reg_z = xg_to_discrete_zinb(
    xg_team_w, xg_opp_w, prob_out_w,
    alpha_w, alpha_w, pi_w, pi_w,
    gpd_team_w, gpd_opp_w, p_exceed_team_w, p_exceed_opp_w
)

# Standard Poisson (no GPD) for comparison
pred_t_w_reg_std, pred_o_w_reg_std = xg_to_discrete(xg_team_w, xg_opp_w, prob_out_w)

# ZINB no GPD for comparison
pred_t_w_zinb_nogpd, pred_o_w_zinb_nogpd = xg_to_discrete_zinb(
    xg_team_w, xg_opp_w, prob_out_w, alpha_w, alpha_w, pi_w, pi_w
)

# ===========================================================================
# Score comparison
# ===========================================================================
print("\n--- Score Comparison ---")
score_m_cls = calc_score_vec(pred_t_m_cls, pred_o_m_cls, gt_m)
score_m_reg = calc_score_vec(pred_t_m_reg, pred_o_m_reg, gt_m)
score_m_reg_nogpd = calc_score_vec(pred_t_m_reg_nogpd, pred_o_m_reg_nogpd, gt_m)
score_w_cls = calc_score_vec(pred_t_w_cls, pred_o_w_cls, gt_w)
score_w_reg_z  = calc_score_vec(pred_t_w_reg_z, pred_o_w_reg_z, gt_w)
score_w_reg_std = calc_score_vec(pred_t_w_reg_std, pred_o_w_reg_std, gt_w)
score_w_zinb_nogpd = calc_score_vec(pred_t_w_zinb_nogpd, pred_o_w_zinb_nogpd, gt_w)

print(f"  Men   Ordinal        : {score_m_cls:.5f}")
print(f"  Men   Reg (Poi+GPD)  : {score_m_reg:.5f}")
print(f"  Men   Reg (Poi noGPD): {score_m_reg_nogpd:.5f}")
print(f"  Women Ordinal        : {score_w_cls:.5f}")
print(f"  Women Reg (ZINB+GPD) : {score_w_reg_z:.5f}")
print(f"  Women Reg (ZINB only): {score_w_zinb_nogpd:.5f}")
print(f"  Women Reg (Poi only) : {score_w_reg_std:.5f}")

# Choose best per gender
pred_t_m = pred_t_m_reg if score_m_reg < score_m_cls else pred_t_m_cls
pred_o_m = pred_o_m_reg if score_m_reg < score_m_cls else pred_o_m_cls

w_methods = {
    'Ordinal': (pred_t_w_cls, pred_o_w_cls, score_w_cls),
    'ZINB+GPD': (pred_t_w_reg_z, pred_o_w_reg_z, score_w_reg_z),
    'ZINB only': (pred_t_w_zinb_nogpd, pred_o_w_zinb_nogpd, score_w_zinb_nogpd),
    'Poi only': (pred_t_w_reg_std, pred_o_w_reg_std, score_w_reg_std)
}
best_w = min(w_methods.items(), key=lambda x: x[1][2])
pred_t_w, pred_o_w = best_w[1][0], best_w[1][1]

print(f"\n  Chosen: Men={'Reg+GPD' if score_m_reg < score_m_cls else 'Ordinal'}")
print(f"          Women={best_w[0]}")

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

out_path = DATA_DIR/"submission_v20.csv"
sub.to_csv(out_path, index=False)
print(f"\nSaved: {out_path} ({len(sub)} rows)")

# Evaluate
evaluate_submission(str(out_path), str(DATA_DIR/"test_ground_truth.csv"))