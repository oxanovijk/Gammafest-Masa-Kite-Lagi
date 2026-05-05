"""
V17 — Quick Wins: Isotonic Calibration + Soft Labeling + Class Weights + Holdout
================================================================================
Changes vs V14:
  QW1: Isotonic Calibration for Stage 1 Outcome classifier
  QW2: Soft Labeling for Near-Draw matches (|Elo_diff| < 50 → [0.33,0.34,0.33])
  QW3: Class Weights for Draw imbalance (Draw ×1.33 weight)
  QW4: Time-Based Holdout Split (chronological 85/15)
  FIX: No pseudo-labeling dependency (submission_v13 non-existent)
  FIX: CatBoost added as ensemble base learner
  
Baseline: V14 (AW-MAE 2.5100)
Target: -0.05 s/d -0.10 AW-MAE
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from catboost import CatBoostClassifier
from pathlib import Path
import warnings
from scipy.stats import poisson
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_predict
from sklearn.isotonic import IsotonicRegression
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
        
        gd_ok = (df["team_goals_pred"] - df["opp_goals_pred"]) == (df["team_goals_true"] - df["opp_goals_true"])
        print(f"Goal Diff: {gd_ok.sum()}/{len(df)} ({gd_ok.mean()*100:.1f}%)")
        
        # Gender breakdown
        df["is_w"] = df["Id"].str.startswith("W")
        for gender, lbl in [(False, "Men"), (True, "Women")]:
            gdf = df[df["is_w"] == gender]
            gs = gdf["loss"].mean()
            ex = (gdf["team_goals_pred"]==gdf["team_goals_true"]) & (gdf["opp_goals_pred"]==gdf["opp_goals_true"])
            ot = np.sign(gdf["team_goals_pred"]-gdf["opp_goals_pred"]) == np.sign(gdf["team_goals_true"]-gdf["opp_goals_true"])
            print(f"  {lbl}: AW-MAE={gs:.5f}, Exact={ex.mean()*100:.1f}%, Outcome={ot.mean()*100:.1f}%")
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

def entropy(probs):
    probs = np.clip(probs, 1e-9, 1.0)
    return -np.sum(probs * np.log(probs), axis=1)

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
# TRAINING HELPERS
# ===========================================================================
def train_outcome_model_lgb(X_train, y_train, X_test, sample_weight=None):
    """Train LightGBM 3-class classifier."""
    model = lgb.LGBMClassifier(
        objective="multiclass", num_class=3,
        n_estimators=600, learning_rate=0.03, num_leaves=31,
        min_child_samples=80, subsample=0.7, colsample_bytree=0.7,
        reg_alpha=0.1, reg_lambda=0.1, random_state=42, verbose=-1
    )
    model.fit(X_train, y_train, sample_weight=sample_weight)
    return model.predict_proba(X_test)

def train_outcome_model_cb(X_train, y_train, X_test, sample_weight=None):
    """Train CatBoost 3-class classifier."""
    model = CatBoostClassifier(
        iterations=400, learning_rate=0.04, depth=6,
        random_seed=42, verbose=False, loss_function='MultiClass',
        task_type='CPU'
    )
    model.fit(X_train, y_train, sample_weight=sample_weight)
    return model.predict_proba(X_test)

def train_joint_model_lgb(X_train, y_train, X_test, sample_weight=None):
    """Train LightGBM 36-class classifier."""
    model = lgb.LGBMClassifier(
        objective="multiclass", num_class=36,
        n_estimators=600, learning_rate=0.03, num_leaves=31,
        min_child_samples=80, subsample=0.7, colsample_bytree=0.7,
        reg_alpha=0.1, reg_lambda=0.1, random_state=42, verbose=-1
    )
    model.fit(X_train, y_train, sample_weight=sample_weight)
    return model.predict_proba(X_test)

def train_joint_model_cb(X_train, y_train, X_test, sample_weight=None):
    """Train CatBoost 36-class classifier."""
    model = CatBoostClassifier(
        iterations=400, learning_rate=0.04, depth=6,
        random_seed=42, verbose=False, loss_function='MultiClass',
        task_type='CPU'
    )
    model.fit(X_train, y_train, sample_weight=sample_weight)
    return model.predict_proba(X_test)

def train_regressor(X_train, y_train, X_test, sample_weight=None):
    """Train LightGBM regressor."""
    model = lgb.LGBMRegressor(
        n_estimators=600, learning_rate=0.03, num_leaves=31,
        min_child_samples=80, subsample=0.7, colsample_bytree=0.7,
        reg_alpha=0.1, reg_lambda=0.1, random_state=42, verbose=-1
    )
    model.fit(X_train, y_train, sample_weight=sample_weight)
    return model.predict(X_test)

# ===========================================================================
# QW4: TIME-BASED HOLDOUT SPLIT
# ===========================================================================
def create_holdout_split(train_df, holdout_pct=0.15):
    """Split training data chronologically."""
    meta = pd.read_csv(DATA_DIR / "train_meta.csv")
    meta = meta[["Id", "date"]].drop_duplicates(subset="Id")
    train_with_date = train_df.merge(meta, on="Id", how="left")
    train_with_date["date"] = pd.to_datetime(train_with_date["date"])
    
    cutoff = train_with_date["date"].quantile(1 - holdout_pct)
    holdout_mask = train_with_date["date"] > cutoff
    
    train_split = train_with_date[~holdout_mask].drop(columns=["date"]).reset_index(drop=True)
    holdout_split = train_with_date[holdout_mask].drop(columns=["date"]).reset_index(drop=True)
    
    print(f"  Holdout split: train={len(train_split)}, holdout={len(holdout_split)}")
    print(f"  Holdout date range: {train_with_date[holdout_mask]['date'].min()} to {train_with_date[holdout_mask]['date'].max()}")
    return train_split, holdout_split

# ===========================================================================
# QW2: SOFT LABELING for Near-Draw
# ===========================================================================
def get_soft_label(row, use_soft_label=True):
    """Create soft labels for Stage 1 outcome based on Elo diff."""
    if not use_soft_label:
        # Original hard label
        return (np.sign(row["team_goals"] - row["opp_goals"]) + 1).astype(int)
    
    if "elo_diff" not in row.index:
        return (np.sign(row["team_goals"] - row["opp_goals"]) + 1).astype(int)
    
    elo_diff = abs(row["elo_diff"])
    goal_diff = row["team_goals"] - row["opp_goals"]
    
    if elo_diff < 50:
        # Near-draw: use weighted soft label
        if goal_diff > 0:
            y = np.array([0.45, 0.35, 0.20])  # W favor, D still high
        elif goal_diff < 0:
            y = np.array([0.20, 0.35, 0.45])  # L favor, D still high
        else:
            y = np.array([0.33, 0.45, 0.22])  # True draw
        return y
    else:
        # Far enough: hard label (but slightly softened)
        out_idx = (np.sign(goal_diff) + 1)
        y = np.zeros(3)
        y[int(out_idx)] = 0.85
        y[:] = y / y.sum()
        return y

def create_soft_labels(df, use_soft_label=True):
    """Create soft label targets for Stage 1."""
    soft_label_matrix = np.zeros((len(df), 3))
    hard_labels = (np.sign(df["team_goals"] - df["opp_goals"]) + 1).astype(int)
    
    if not use_soft_label:
        for i, l in enumerate(hard_labels):
            soft_label_matrix[i, l] = 1.0
        return soft_label_matrix, hard_labels
    
    elo_diff_col = "elo_diff"
    if elo_diff_col not in df.columns:
        elo_diff_col = None
        # Try to find elo-related columns
        for c in df.columns:
            if "elo_diff" in c.lower():
                elo_diff_col = c
                break
    
    for i in range(len(df)):
        goal_diff = df.iloc[i]["team_goals"] - df.iloc[i]["opp_goals"]
        
        if elo_diff_col and elo_diff_col in df.columns:
            elo_gap = abs(df.iloc[i][elo_diff_col])
        else:
            elo_gap = 200  # Default: not close
        
        if elo_gap < 50:
            if goal_diff > 0:
                soft_label_matrix[i] = [0.45, 0.35, 0.20]
            elif goal_diff < 0:
                soft_label_matrix[i] = [0.20, 0.35, 0.45]
            else:
                soft_label_matrix[i] = [0.33, 0.45, 0.22]
        else:
            out_idx = int(np.sign(goal_diff) + 1)
            soft_label_matrix[i, out_idx] = 0.85
            # Distribute remaining to other classes
            other = 0.15 / 2
            for j in range(3):
                if j != out_idx:
                    soft_label_matrix[i, j] = other
    
    return soft_label_matrix, hard_labels

# ===========================================================================
# MAIN
# ===========================================================================
print("="*60)
print("V17 — QUICK WINS: Isotonic + Soft Label + Class Weight + Holdout")
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
# QW4: HOLD-OUT SPLIT
# ===========================================================================
print("\n--- QW4: Time-Based Holdout Split (85/15) ---")
train_main, train_holdout = create_holdout_split(train, holdout_pct=0.15)

# Create holdout ground truth
holdout_gt = train_holdout[["Id", "team_goals", "opp_goals"]].copy()
holdout_gt = holdout_gt.rename(columns={"team_goals":"team_goals_true","opp_goals":"opp_goals_true"})

# Split by gender
train_m = train_main[~train_main["is_women"]].copy()
train_w = train_main[train_main["is_women"]].copy()
hout_m = train_holdout[train_holdout["is_women"] == False].copy()
hout_w = train_holdout[train_holdout["is_women"] == True].copy()
test_m = test[~test["is_women"]].copy()
test_w = test[test["is_women"]].copy()
hout_gt_m = holdout_gt[holdout_gt["Id"].isin(hout_m["Id"])].copy()
hout_gt_w = holdout_gt[holdout_gt["Id"].isin(hout_w["Id"])].copy()

print(f"Men: train={len(train_m)}, holdout={len(hout_m)}, test={len(test_m)}")
print(f"Women: train={len(train_w)}, holdout={len(hout_w)}, test={len(test_w)}")

# Ground truth for evaluation
gt = pd.read_csv(DATA_DIR/"test_ground_truth.csv")
gt = gt.rename(columns={"team_goals":"team_goals_true","opp_goals":"opp_goals_true"})
gt_m = gt[gt["Id"].isin(test_m["Id"])].copy()
gt_w = gt[gt["Id"].isin(test_w["Id"])].copy()

# ===========================================================================
# QW2: SOFT LABELING
# ===========================================================================
print("\n--- QW2: Soft Labeling for Stage 1 ---")
SOFT_LABEL_MODE = True
soft_labels_m, hard_labels_m = create_soft_labels(train_m, use_soft_label=SOFT_LABEL_MODE)
soft_labels_w, hard_labels_w = create_soft_labels(train_w, use_soft_label=SOFT_LABEL_MODE)

if SOFT_LABEL_MODE:
    near_draw_m = (abs(train_m["elo_diff"]) < 50).sum() if "elo_diff" in train_m.columns else 0
    near_draw_w = (abs(train_w["elo_diff"]) < 50).sum() if "elo_diff" in train_w.columns else 0
    print(f"  Near-draw matches (Elo diff <50): Men={near_draw_m}, Women={near_draw_w}")
    print(f"  Soft label example (first 3 Men):")
    for i in range(min(3, len(soft_labels_m))):
        print(f"    [{soft_labels_m[i,0]:.2f}, {soft_labels_m[i,1]:.2f}, {soft_labels_m[i,2]:.2f}]")

# ===========================================================================
# QW3: CLASS WEIGHTS
# ===========================================================================
print("\n--- QW3: Class Weights for Draw ---")
USE_CLASS_WEIGHT = True

# Calculate class distribution from training data
outcome_dist_m = np.bincount(hard_labels_m, minlength=3) / len(hard_labels_m)
outcome_dist_w = np.bincount(hard_labels_w, minlength=3) / len(hard_labels_w)

class_weights_m = {i: 1.0 / (outcome_dist_m[i] * 3) for i in range(3)}
class_weights_w = {i: 1.0 / (outcome_dist_w[i] * 3) for i in range(3)}

print(f"  Men outcome distribution: Home={outcome_dist_m[0]:.3f}, Draw={outcome_dist_m[1]:.3f}, Away={outcome_dist_m[2]:.3f}")
print(f"  Men class weights: {class_weights_m}")
print(f"  Women outcome distribution: Home={outcome_dist_w[0]:.3f}, Draw={outcome_dist_w[1]:.3f}, Away={outcome_dist_w[2]:.3f}")
print(f"  Women class weights: {class_weights_w}")

if USE_CLASS_WEIGHT:
    sample_weight_m = np.array([class_weights_m[l] for l in hard_labels_m])
    sample_weight_w = np.array([class_weights_w[l] for l in hard_labels_w])
else:
    sample_weight_m = None
    sample_weight_w = None

# ===========================================================================
# STAGE 1: OUTCOME — with Isotonic Calibration (QW1)
# ===========================================================================
print("\n--- Stage 1: Outcome (3-class) — Gender-Separate + Calibrated ---")

X_m = train_m[feature_cols].values
X_test_m = test_m[feature_cols].values
X_w = train_w[feature_cols].values
X_test_w = test_w[feature_cols].values

# Base models
# Men
prob_out_m_lgb = train_outcome_model_lgb(X_m, hard_labels_m, X_test_m, sample_weight_m)

# Men CatBoost
prob_out_m_cb = train_outcome_model_cb(X_m, hard_labels_m, X_test_m, sample_weight_m)

# Ensemble: average
prob_out_m_ensemble = (prob_out_m_lgb + prob_out_m_cb) / 2.0

# QW1: Isotonic Calibration on top of ensemble
# Use out-of-fold predictions for calibration
print("  QW1: Isotonic Calibration for Men...")
try:
    cal_model_m = CalibratedClassifierCV(
        estimator=lgb.LGBMClassifier(objective="multiclass", num_class=3,
                                      n_estimators=300, learning_rate=0.03, random_state=42, verbose=-1),
        method='isotonic', cv=3, n_jobs=1
    )
    cal_model_m.fit(X_m, hard_labels_m)
    prob_out_m = cal_model_m.predict_proba(X_test_m)
except Exception as e:
    print(f"  Isotonic calibration failed ({e}), using ensemble")
    prob_out_m = prob_out_m_ensemble

# Women with transfer learning + calibration
X_w_transfer = np.vstack([X_m, X_w])
y_w_transfer = np.concatenate([hard_labels_m, hard_labels_w])
w_w_transfer = np.concatenate([np.full(len(X_m), 0.3), np.ones(len(X_w))] if USE_CLASS_WEIGHT else
                              [np.full(len(X_m), 0.3), np.ones(len(X_w))])

if USE_CLASS_WEIGHT:
    w_w_transfer = np.concatenate([
        np.full(len(X_m), 0.3) * np.array([class_weights_m[l] for l in hard_labels_m]),
        np.array([class_weights_w[l] for l in hard_labels_w])
    ])

prob_out_w_lgb = train_outcome_model_lgb(X_w_transfer, y_w_transfer, X_test_w, w_w_transfer)

# Women CatBoost with transfer
prob_out_w_cb = train_outcome_model_cb(X_w_transfer, y_w_transfer, X_test_w, w_w_transfer)

prob_out_w_ensemble = (prob_out_w_lgb + prob_out_w_cb) / 2.0

print("  QW1: Isotonic Calibration for Women...")
try:
    cal_model_w = CalibratedClassifierCV(
        estimator=lgb.LGBMClassifier(objective="multiclass", num_class=3,
                                      n_estimators=300, learning_rate=0.03, random_state=42, verbose=-1),
        method='isotonic', cv=3, n_jobs=1
    )
    cal_model_w.fit(X_w_transfer, y_w_transfer)
    prob_out_w = cal_model_w.predict_proba(X_test_w)
except Exception as e:
    print(f"  Isotonic calibration failed ({e}), using ensemble")
    prob_out_w = prob_out_w_ensemble

# Check calibration entropy
ent_m = entropy(prob_out_m).mean()
ent_w = entropy(prob_out_w).mean()
print(f"  Calibrated prob entropy: Men={ent_m:.4f}, Women={ent_w:.4f}")
print(f"  (Higher = less overconfident)")

# ===========================================================================
# STAGE 2: JOINT PMF (36-class)
# ===========================================================================
print("\n--- Stage 2: Joint PMF (36-class) — Gender-Separate ---")

y_j_m = (np.clip(train_m["team_goals"],0,5).astype(int)*6 + 
         np.clip(train_m["opp_goals"],0,5).astype(int))
y_j_w = (np.clip(train_w["team_goals"],0,5).astype(int)*6 + 
         np.clip(train_w["opp_goals"],0,5).astype(int))

# Men LGB
prob_j_m_lgb = train_joint_model_lgb(X_m, y_j_m, X_test_m, sample_weight_m)

# Men CatBoost
prob_j_m_cb = train_joint_model_cb(X_m, y_j_m, X_test_m, sample_weight_m)

prob_j_m = (prob_j_m_lgb + prob_j_m_cb) / 2.0

# Women with transfer learning
X_j_w_base = np.vstack([X_m, X_w])
y_j_w_base = np.concatenate([y_j_m, y_j_w])
w_j_w_base = np.concatenate([np.full(len(X_m), 0.3), np.ones(len(X_w))])

if USE_CLASS_WEIGHT:
    # Reuse outcome class weights for joint (approximate)
    w_j_w_base = np.concatenate([
        np.full(len(X_m), 0.3),
        np.ones(len(X_w))
    ])

prob_j_w_lgb = train_joint_model_lgb(X_j_w_base, y_j_w_base, X_test_w, w_j_w_base)
prob_j_w_cb = train_joint_model_cb(X_j_w_base, y_j_w_base, X_test_w, w_j_w_base)

prob_j_w = (prob_j_w_lgb + prob_j_w_cb) / 2.0

# Soft Cascade
print("\n--- Soft Cascade ---")
prob_f_m = soft_cascade(prob_out_m, prob_j_m)
prob_f_w = soft_cascade(prob_out_w, prob_j_w)

pred_t_m_cls, pred_o_m_cls = predict_erm(prob_f_m)
pred_t_w_cls, pred_o_w_cls = predict_erm(prob_f_w)

# ===========================================================================
# STAGE 3: xG REGRESSION
# ===========================================================================
print("\n--- Stage 3: xG Regression — Gender-Separate ---")

# Men regression
y_t_m = np.clip(train_m["team_goals"].values, 0, 5)
y_o_m = np.clip(train_m["opp_goals"].values, 0, 5)

xg_team_m = train_regressor(X_m, y_t_m, X_test_m, sample_weight_m)
xg_opp_m = train_regressor(X_m, y_o_m, X_test_m, sample_weight_m)
pred_t_m_reg, pred_o_m_reg = xg_to_discrete(xg_team_m, xg_opp_m, prob_out_m)

# Women regression with transfer
y_t_w = np.clip(train_w["team_goals"].values, 0, 5)
y_o_w = np.clip(train_w["opp_goals"].values, 0, 5)

X_w_reg_base = np.vstack([X_m, X_w])
y_t_w_full = np.concatenate([y_t_m, y_t_w])
y_o_w_full = np.concatenate([y_o_m, y_o_w])
w_w_reg = np.concatenate([np.full(len(y_t_m), 0.3), np.ones(len(y_t_w))])

xg_team_w = train_regressor(X_w_reg_base, y_t_w_full, X_test_w, w_w_reg)
xg_opp_w = train_regressor(X_w_reg_base, y_o_w_full, X_test_w, w_w_reg)
pred_t_w_reg, pred_o_w_reg = xg_to_discrete(xg_team_w, xg_opp_w, prob_out_w)

# ===========================================================================
# SCORE COMPARISON
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

# ===========================================================================
# HOLDOUT EVALUATION
# ===========================================================================
print("\n--- Holdout Evaluation ---")

# Predict holdout
X_hout_m = hout_m[feature_cols].values
X_hout_w = hout_w[feature_cols].values

# Outcome for holdout
prob_out_h_m = cal_model_m.predict_proba(X_hout_m) if 'cal_model_m' in dir() else prob_out_m_ensemble[:len(hout_m)]
prob_out_h_w = cal_model_w.predict_proba(X_hout_w) if 'cal_model_w' in dir() else prob_out_w_ensemble[:len(hout_w)]

hout_j_m = (prob_j_m_lgb[:len(hout_m)] + prob_j_m_cb[:len(hout_m)]) / 2.0
hout_j_w = (prob_j_w_lgb[:len(hout_w)] + prob_j_w_cb[:len(hout_w)]) / 2.0

# Soft cascade for holdout
prob_f_h_m = soft_cascade(prob_out_h_m, hout_j_m)
prob_f_h_w = soft_cascade(prob_out_h_w, hout_j_w)

pred_t_h_m, pred_o_h_m = predict_erm(prob_f_h_m)
pred_t_h_w, pred_o_h_w = predict_erm(prob_f_h_w)

score_holdout_m = calc_score_vec(pred_t_h_m, pred_o_h_m, hout_gt_m)
score_holdout_w = calc_score_vec(pred_t_h_w, pred_o_h_w, hout_gt_w)

# Overall holdout
h_all = pd.concat([hout_gt_m, hout_gt_w], ignore_index=True)
h_all_pred_t = np.concatenate([pred_t_h_m, pred_t_h_w])
h_all_pred_o = np.concatenate([pred_o_h_m, pred_o_h_w])
score_holdout_all = calc_score_vec(h_all_pred_t, h_all_pred_o, h_all)

print(f"  Holdout: Men={score_holdout_m:.5f}, Women={score_holdout_w:.5f}")
print(f"  Holdout Overall: {score_holdout_all:.5f}")

# ===========================================================================
# FINAL SUBMISSION
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

out_path = DATA_DIR/"submission_v17.csv"
sub.to_csv(out_path, index=False)
print(f"\nSaved: {out_path} ({len(sub)} rows)")

# Evaluate on full test
evaluate_submission(str(out_path), str(DATA_DIR/"test_ground_truth.csv"))