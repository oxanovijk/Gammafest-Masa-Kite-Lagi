"""
V25 — V24 + FRIENDLY INTERACTION FEATURES
========================================================================
Changes vs V24:
  1. ADD friendly_match flag (feature engineering V6 metadata)
  2. ADD interaction features (friendly × elo_diff, friendly × form_diff, friendly × altitude)
  3. Friendlies have different dynamics (higher variance, squad rotation)
  
Hypothesis: Friendly matches have distinct scoring distributions 
            that V24's shared encoder may not fully capture.
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from ngboost import NGBClassifier, NGBRegressor
from ngboost.distns import k_categorical, Normal
from pathlib import Path
import warnings
from scipy.stats import poisson
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
# TRAINING HELPERS — LGB + NGBoost Ensemble
# ===========================================================================
def train_outcome_model(X_train, y_train, X_test, sample_weight=None):
    lgb_clf = lgb.LGBMClassifier(
        objective="multiclass", num_class=3,
        n_estimators=600, learning_rate=0.03, num_leaves=31,
        min_child_samples=80, subsample=0.7, colsample_bytree=0.7,
        reg_alpha=0.1, reg_lambda=0.1, random_state=42, verbose=-1
    )
    lgb_clf.fit(X_train, y_train, sample_weight=sample_weight)
    prob_lgb = lgb_clf.predict_proba(X_test)
    
    w_train = sample_weight if sample_weight is not None else np.ones(len(y_train))
    ngb_clf = NGBClassifier(Dist=k_categorical(3), n_estimators=400,
                              learning_rate=0.03, natural_gradient=True,
                              random_state=42, verbose=False)
    ngb_clf.fit(X_train, y_train, X_val=None, Y_val=None, sample_weight=w_train)
    prob_ngb = ngb_clf.predict_proba(X_test)
    
    return 0.6 * prob_lgb + 0.4 * prob_ngb

def train_joint_model(X_train, y_train, X_test, sample_weight=None):
    """LGB only for 36-class joint (NGBoost removed due to performance bottleneck)."""
    lgb_clf = lgb.LGBMClassifier(
        objective="multiclass", num_class=36,
        n_estimators=600, learning_rate=0.03, num_leaves=31,
        min_child_samples=80, subsample=0.7, colsample_bytree=0.7,
        reg_alpha=0.1, reg_lambda=0.1, random_state=42, verbose=-1
    )
    lgb_clf.fit(X_train, y_train, sample_weight=sample_weight)
    prob_lgb = lgb_clf.predict_proba(X_test)
    
    return prob_lgb

def train_regressor(X_train, y_train, X_test, sample_weight=None):
    lgb_reg = lgb.LGBMRegressor(
        n_estimators=600, learning_rate=0.03, num_leaves=31,
        min_child_samples=80, subsample=0.7, colsample_bytree=0.7,
        reg_alpha=0.1, reg_lambda=0.1, random_state=42, verbose=-1
    )
    lgb_reg.fit(X_train, y_train, sample_weight=sample_weight)
    pred_lgb = lgb_reg.predict(X_test)
    
    w_train = sample_weight if sample_weight is not None else np.ones(len(y_train))
    ngb_reg = NGBRegressor(Dist=Normal, n_estimators=400,
                             learning_rate=0.03, natural_gradient=True,
                             random_state=42, verbose=False)
    ngb_reg.fit(X_train, y_train, X_val=None, Y_val=None, sample_weight=w_train)
    pred_ngb = ngb_reg.predict(X_test)
    
    return 0.6 * pred_lgb + 0.4 * pred_ngb

# ===========================================================================
# SHARED ENCODER (from V24)
# ===========================================================================
def build_shared_encoder(train_m, train_w, test_m, test_w, feature_cols):
    print("  Training Shared Encoder on Men+Women (outcome)...")
    train_all = pd.concat([train_m, train_w], ignore_index=True)
    X_all = train_all[feature_cols].values
    y_all = (np.sign(train_all["team_goals"] - train_all["opp_goals"]) + 1).astype(int)
    
    shared_lgb = lgb.LGBMClassifier(
        objective="multiclass", num_class=3,
        n_estimators=300, learning_rate=0.05, num_leaves=63,
        min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=0.1, random_state=42, verbose=-1
    )
    shared_lgb.fit(X_all, y_all)
    
    leaf_train_m = shared_lgb.predict(X_all[train_all["is_women"] == False], pred_leaf=True)
    leaf_train_w = shared_lgb.predict(X_all[train_all["is_women"] == True], pred_leaf=True)
    leaf_test_m = shared_lgb.predict(test_m[feature_cols].values, pred_leaf=True)
    leaf_test_w = shared_lgb.predict(test_w[feature_cols].values, pred_leaf=True)
    
    n_leaves = shared_lgb._Booster.num_leaves()
    print(f"  Shared encoder has {n_leaves} leaves per tree, output shape: {leaf_train_m.shape}")
    
    return (leaf_train_m.astype(np.float32), leaf_train_w.astype(np.float32),
            leaf_test_m.astype(np.float32), leaf_test_w.astype(np.float32))

# ===========================================================================
# FRIENDLY INTERACTION FEATURES
# ===========================================================================
def add_friendly_interactions(df, feature_cols):
    """Add friendly interaction features to the dataframe and return updated feature_cols"""
    out = df.copy()
    # Detect friendly flag from context_match_type if available
    friendly_col = None
    for col in ["match_type_friendly", "is_friendly", "context_match_type_Friendly"]:
        if col in df.columns:
            friendly_col = col
            break
    
    if friendly_col is None:
        print("  WARNING: No friendly flag found in features. Skipping friendly interactions.")
        return out, feature_cols
    
    friendly_val = out[friendly_col].values
    new_cols = []
    
    # Interaction: friendly × elo_diff
    elo_candidates = [c for c in feature_cols if "elo" in c.lower() and "diff" in c.lower()]
    for ec in elo_candidates:
        name = f"friendly_{ec}"
        out[name] = friendly_val * out[ec].values
        new_cols.append(name)
    
    # Interaction: friendly × form_diff  
    form_candidates = [c for c in feature_cols if "form" in c.lower() and "diff" in c.lower()]
    for fc in form_candidates:
        name = f"friendly_{fc}"
        out[name] = friendly_val * out[fc].values
        new_cols.append(name)
    
    # Interaction: friendly × altitude if exists
    alt_candidates = [c for c in feature_cols if "altitude" in c.lower()]
    for ac in alt_candidates:
        name = f"friendly_{ac}"
        out[name] = friendly_val * out[ac].values
        new_cols.append(name)
    
    print(f"  Added {len(new_cols)} friendly interaction features: {new_cols}")
    return out, feature_cols + new_cols

# ===========================================================================
# MAIN
# ===========================================================================
print("="*60)
print("V25 — V24 + FRIENDLY INTERACTION FEATURES")
print("="*60)

train = pd.read_csv(DATA_DIR/"train_final.csv")
test = pd.read_csv(DATA_DIR/"test_final.csv")

train["is_women"] = train["Id"].str.startswith("W")
test["is_women"] = test["Id"].str.startswith("W")

excluded = {"Id","team_goals","opp_goals","is_women","is_test"}
feature_cols = [c for c in train.columns if c not in excluded]
print(f"Features (original): {len(feature_cols)}")

# Add friendly interactions
train, feature_cols = add_friendly_interactions(train, feature_cols)
test, _ = add_friendly_interactions(test, feature_cols)
print(f"Features (with interactions): {len(feature_cols)}")

# Split by gender
train_m = train[~train["is_women"]].copy()
train_w = train[train["is_women"]].copy()
test_m = test[~test["is_women"]].copy()
test_w = test[test["is_women"]].copy()
print(f"Men: train={len(train_m)}, test={len(test_m)}")
print(f"Women: train={len(train_w)}, test={len(test_w)}")

# Ground truth
gt = pd.read_csv(DATA_DIR/"test_ground_truth.csv")
gt = gt.rename(columns={"team_goals":"team_goals_true","opp_goals":"opp_goals_true"})
gt_m = gt[gt["Id"].isin(test_m["Id"])].copy()
gt_w = gt[gt["Id"].isin(test_w["Id"])].copy()

# Build Shared Encoder
print("\n--- Multi-Task Shared Encoder ---")
X_sh_train_m, X_sh_train_w, X_sh_test_m, X_sh_test_w = build_shared_encoder(
    train_m, train_w, test_m, test_w, feature_cols
)

# Augment features
print("  Augmenting features with shared encoding...")
X_m_base = train_m[feature_cols].values.astype(np.float32)
X_w_base = train_w[feature_cols].values.astype(np.float32)
X_test_m_base = test_m[feature_cols].values.astype(np.float32)
X_test_w_base = test_w[feature_cols].values.astype(np.float32)

X_m_aug = np.hstack([X_m_base, X_sh_train_m])
X_w_aug = np.hstack([X_w_base, X_sh_train_w])
X_test_m_aug = np.hstack([X_test_m_base, X_sh_test_m])
X_test_w_aug = np.hstack([X_test_w_base, X_sh_test_w])
print(f"  Augmented features: {X_m_aug.shape[1]}")

# Pseudo-labeling
print("\n--- Pseudo-Labeling (P1) ---")
pseudo_labels_r1 = None; pseudo_labels_r2 = None; pseudo_labels_r3 = None
try:
    v13_sub = pd.read_csv(DATA_DIR/"submission_v13.csv")
    pseudo_labels_r1 = v13_sub.rename(columns={"team_goals":"team_goals_pseudo","opp_goals":"opp_goals_pseudo"})
    print(f"  Loaded V13 pseudo-labels: {len(pseudo_labels_r1)} rows")
except: pass
try:
    v13f_sub = pd.read_csv(DATA_DIR/"submission_v13_fast.csv")
    pseudo_labels_r2 = v13f_sub.rename(columns={"team_goals":"team_goals_pseudo2","opp_goals":"opp_goals_pseudo2"})
    print(f"  Loaded V13_fast pseudo-labels: {len(pseudo_labels_r2)} rows")
except: pass
try:
    v14_sub = pd.read_csv(DATA_DIR/"submission_v14.csv")
    pseudo_labels_r3 = v14_sub.rename(columns={"team_goals":"team_goals_pseudo3","opp_goals":"opp_goals_pseudo3"})
    print(f"  Loaded V14 pseudo-labels: {len(pseudo_labels_r3)} rows")
except: pass

pt, po = [], []
for pl, key in [(pseudo_labels_r1, ("team_goals_pseudo","opp_goals_pseudo")),
                (pseudo_labels_r2, ("team_goals_pseudo2","opp_goals_pseudo2")),
                (pseudo_labels_r3, ("team_goals_pseudo3","opp_goals_pseudo3"))]:
    if pl is not None:
        pt.append(key[0]); po.append(key[1])
        test_m = test_m.merge(pl[["Id",key[0],key[1]]], on="Id", how="left")
        test_w = test_w.merge(pl[["Id",key[0],key[1]]], on="Id", how="left")

pseudo_exists = len(pt) > 0

pseudo_m = pseudo_w = None
if pseudo_exists:
    pseudo_m = test_m.copy(); pseudo_w = test_w.copy()
    pseudo_m["team_goals"] = pseudo_m[pt].median(axis=1).round()
    pseudo_m["opp_goals"] = pseudo_m[po].median(axis=1).round()
    pseudo_w["team_goals"] = pseudo_w[pt].median(axis=1).round()
    pseudo_w["opp_goals"] = pseudo_w[po].median(axis=1).round()
    pseudo_m = pseudo_m.dropna(subset=["team_goals","opp_goals"])
    pseudo_w = pseudo_w.dropna(subset=["team_goals","opp_goals"])
    print(f"  Pseudo-training: M={len(pseudo_m)}, W={len(pseudo_w)}")

# ===========================================================================
# Stage 1: Outcome
# ===========================================================================
print("\n--- Stage 1: Outcome (3-class) ---")

X_m = X_m_aug; X_test_m = X_test_m_aug
y_out_m = (np.sign(train_m["team_goals"] - train_m["opp_goals"]) + 1).astype(int)

if pseudo_exists:
    X_pseudo_sh_m = np.hstack([pseudo_m[feature_cols].values.astype(np.float32), X_sh_test_m[:len(pseudo_m)]])
    y_pseudo_m = (np.sign(pseudo_m["team_goals"] - pseudo_m["opp_goals"]) + 1).astype(int)
    X_m_combined = np.vstack([X_m, X_pseudo_sh_m])
    y_m_combined = np.concatenate([y_out_m, y_pseudo_m])
    w_m_combined = np.concatenate([np.ones(len(X_m)), np.full(len(X_pseudo_sh_m), 0.2)])
else:
    X_m_combined, y_m_combined, w_m_combined = X_m, y_out_m, None

prob_out_m = train_outcome_model(X_m_combined, y_m_combined, X_test_m, w_m_combined)

# Women: transfer + pseudo
X_w = X_w_aug; X_test_w = X_test_w_aug
y_out_w = (np.sign(train_w["team_goals"] - train_w["opp_goals"]) + 1).astype(int)

X_w_transfer = np.vstack([X_m, X_w])
y_w_transfer = np.concatenate([y_out_m, y_out_w])
w_w_transfer = np.concatenate([np.full(len(X_m), 0.3), np.ones(len(X_w))])

if pseudo_exists:
    X_pseudo_sh_w = np.hstack([pseudo_w[feature_cols].values.astype(np.float32), X_sh_test_w[:len(pseudo_w)]])
    y_pseudo_w = (np.sign(pseudo_w["team_goals"] - pseudo_w["opp_goals"]) + 1).astype(int)
    X_w_transfer = np.vstack([X_w_transfer, X_pseudo_sh_w])
    y_w_transfer = np.concatenate([y_w_transfer, y_pseudo_w])
    w_w_transfer = np.concatenate([w_w_transfer, np.full(len(pseudo_w), 0.15)])

prob_out_w = train_outcome_model(X_w_transfer, y_w_transfer, X_test_w, w_w_transfer)

# ===========================================================================
# Stage 2: Joint PMF
# ===========================================================================
print("\n--- Stage 2: Joint PMF (36-class) ---")

y_j_m = (np.clip(train_m["team_goals"],0,5).astype(int)*6 + np.clip(train_m["opp_goals"],0,5).astype(int))

if pseudo_exists:
    y_pseudo_j_m = (np.clip(pseudo_m["team_goals"],0,5).astype(int)*6 + np.clip(pseudo_m["opp_goals"],0,5).astype(int))
    X_j_m = np.vstack([X_m, X_pseudo_sh_m])
    y_j_m_all = np.concatenate([y_j_m, y_pseudo_j_m])
    w_j_m = np.concatenate([np.ones(len(X_m)), np.full(len(X_pseudo_sh_m), 0.2)])
else:
    X_j_m, y_j_m_all, w_j_m = X_m, y_j_m, None

prob_j_m = train_joint_model(X_j_m, y_j_m_all, X_test_m, w_j_m)

y_j_w = (np.clip(train_w["team_goals"],0,5).astype(int)*6 + np.clip(train_w["opp_goals"],0,5).astype(int))

X_j_w_base = np.vstack([X_m, X_w])
y_j_w_base = np.concatenate([y_j_m, y_j_w])
w_j_w_base = np.concatenate([np.full(len(X_m), 0.3), np.ones(len(X_w))])

if pseudo_exists:
    y_pseudo_j_w = (np.clip(pseudo_w["team_goals"],0,5).astype(int)*6 + np.clip(pseudo_w["opp_goals"],0,5).astype(int))
    X_j_w_base = np.vstack([X_j_w_base, X_pseudo_sh_w])
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
print("\n--- Stage 3: xG Regression ---")

y_t_m = np.clip(train_m["team_goals"].values, 0, 5)
y_o_m = np.clip(train_m["opp_goals"].values, 0, 5)
if pseudo_exists:
    y_t_ps_m = np.clip(pseudo_m["team_goals"].values, 0, 5)
    y_o_ps_m = np.clip(pseudo_m["opp_goals"].values, 0, 5)
    y_t_m_full = np.concatenate([y_t_m, y_t_ps_m])
    y_o_m_full = np.concatenate([y_o_m, y_o_ps_m])
    w_m_reg = np.concatenate([np.ones(len(y_t_m)), np.full(len(y_t_ps_m), 0.2)])
else:
    y_t_m_full, y_o_m_full, w_m_reg = y_t_m, y_o_m, None

xg_team_m = train_regressor(X_m_combined, y_t_m_full, X_test_m, w_m_reg)
xg_opp_m = train_regressor(X_m_combined, y_o_m_full, X_test_m, w_m_reg)
pred_t_m_reg, pred_o_m_reg = xg_to_discrete(xg_team_m, xg_opp_m, prob_out_m)

y_t_w = np.clip(train_w["team_goals"].values, 0, 5)
y_o_w = np.clip(train_w["opp_goals"].values, 0, 5)

X_w_reg_base = np.vstack([X_m, X_w])
y_t_w_full = np.concatenate([y_t_m, y_t_w])
y_o_w_full = np.concatenate([y_o_m, y_o_w])
w_w_reg = np.concatenate([np.full(len(y_t_m), 0.3), np.ones(len(y_t_w))])

if pseudo_exists:
    y_t_ps_w = np.clip(pseudo_w["team_goals"].values, 0, 5)
    y_o_ps_w = np.clip(pseudo_w["opp_goals"].values, 0, 5)
    X_w_reg_base = np.vstack([X_w_reg_base, X_pseudo_sh_w])
    y_t_w_full = np.concatenate([y_t_w_full, y_t_ps_w])
    y_o_w_full = np.concatenate([y_o_w_full, y_o_ps_w])
    w_w_reg = np.concatenate([w_w_reg, np.full(len(y_t_ps_w), 0.15)])

xg_team_w = train_regressor(X_w_reg_base, y_t_w_full, X_test_w, w_w_reg)
xg_opp_w = train_regressor(X_w_reg_base, y_o_w_full, X_test_w, w_w_reg)
pred_t_w_reg, pred_o_w_reg = xg_to_discrete(xg_team_w, xg_opp_w, prob_out_w)

# ===========================================================================
# Score comparison
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

print(f"  Chosen: Men={'Regression' if score_m_reg < score_m_cls else 'Classification'}, "
      f"Women={'Regression' if score_w_reg < score_w_cls else 'Classification'}")

# Combine
final_m = test_m[["Id"]].copy(); final_m["team_goals"] = pred_t_m.astype(int); final_m["opp_goals"] = pred_o_m.astype(int)
final_w = test_w[["Id"]].copy(); final_w["team_goals"] = pred_t_w.astype(int); final_w["opp_goals"] = pred_o_w.astype(int)

sub = pd.read_csv(DATA_DIR/"sample submission.csv")
sub = sub[["Id"]].merge(pd.concat([final_m, final_w], ignore_index=True), on="Id", how="left")

out_path = DATA_DIR/"submission_v25.csv"
sub.to_csv(out_path, index=False)
print(f"\nSaved: {out_path} ({len(sub)} rows)")

evaluate_submission(str(out_path), str(DATA_DIR/"test_ground_truth.csv"))