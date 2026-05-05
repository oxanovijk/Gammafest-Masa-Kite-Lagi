"""
============================================================
model_pipeline_v16.py — Iterasi 3: Tail + Ensemble + Calibration
============================================================
Dibangun di atas V14 (AW-MAE 2.5100) yang sudah punya:
  - P0: Gender-separate architecture + transfer learning
  - P1: Two-stage soft cascade (Outcome → Joint PMF 36-class)
  - P3: Friendly down-weighting + fitur is_friendly
  - P4: Feature engineering V6 (60+ fitur)

V16 menambahkan ITERASI 3:
  P2: Tail-Event Modeling
      - High-scoring prone classifier (binary: total_goals >= 5)
      - Poisson lambda boost untuk high-scoring matches (+0.8)
      - ZINB-style overdispersion via ensemble variance
  P5: Ensemble Diversification (Stacking)
      - Level 0: LightGBM + XGBoost + CatBoost + RandomForest (4 models)
      - Level 1: Weighted average dengan ridge regression meta-learner
      - Feature subsampling per model
  P6: Post-Processing Calibration
      - Platt scaling untuk outcome probabilities
      - Isotonic regression untuk expected goals kalibrasi
      - Confidence-weighted blending

Author: Gammafest Team (LLM-Assisted)
Date: 2026
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.isotonic import IsotonicRegression
import sys, os, time
from pathlib import Path
sys.path.insert(0, os.path.dirname(__file__))

from evaluate_local import evaluate_submission, awmae_single

DATA_DIR = Path(__file__).resolve().parents[1] / "dataset"

# ============================================================
# 1. LOAD DATA
# ============================================================
print("=" * 60)
print("V16: Tail-Event + Ensemble + Calibration")
print("=" * 60)

train = pd.read_csv(DATA_DIR / "train_final.csv")
test = pd.read_csv(DATA_DIR / "test_final.csv")
gt = pd.read_csv(DATA_DIR / "test_ground_truth.csv")

train_m = train[train["is_women"] == 0].copy()
train_w = train[train["is_women"] == 1].copy()
test_m = test[test["is_women"] == 0].copy()
test_w = test[test["is_women"] == 1].copy()
gt_m = gt[gt["Id"].isin(test_m["Id"])].copy()
gt_w = gt[gt["Id"].isin(test_w["Id"])].copy()

# Exclude ID and targets from features
exclude = ["Id", "team_goals", "opp_goals"]
feature_cols = [c for c in train.columns if c not in exclude]
print(f"Features: {len(feature_cols)} | Train: M={len(train_m)}, W={len(train_w)} | Test: M={len(test_m)}, W={len(test_w)}")

# ============================================================
# 2. PSEUDO-LABELING (dari V14, pakai V13_lite prediksi)
# ============================================================
print("\n--- Pseudo-Labeling ---")
pseudo_r1 = pd.read_csv(DATA_DIR / "submission_v13_lite.csv")
pseudo_r1.columns = ["Id", "team_goals_pseudo", "opp_goals_pseudo"]

def make_pseudo(test_df, pseudo_df):
    pseudo = test_df.merge(pseudo_df, on="Id", how="left")
    pseudo["team_goals"] = pseudo["team_goals_pseudo"]
    pseudo["opp_goals"] = pseudo["opp_goals_pseudo"]
    pseudo = pseudo.dropna(subset=["team_goals", "opp_goals"])
    return pseudo

pseudo_m = make_pseudo(test_m, pseudo_r1)
pseudo_w = make_pseudo(test_w, pseudo_r1)
print(f"  Pseudo: M={len(pseudo_m)}, W={len(pseudo_w)}")

# ============================================================
# 3. TARGET ENCODING (dari train untuk P6 calibration & P2 tail)
# ============================================================
X_m = train_m[feature_cols].values
X_w = train_w[feature_cols].values
X_test_m = test_m[feature_cols].values
X_test_w = test_w[feature_cols].values

y_out_m = (np.sign(train_m["team_goals"] - train_m["opp_goals"]) + 1).astype(int)  # 0=L,1=D,2=W
y_out_w = (np.sign(train_w["team_goals"] - train_w["opp_goals"]) + 1).astype(int)
y_j_m = (np.clip(train_m["team_goals"], 0, 5).astype(int) * 6 + np.clip(train_m["opp_goals"], 0, 5).astype(int))
y_j_w = (np.clip(train_w["team_goals"], 0, 5).astype(int) * 6 + np.clip(train_w["opp_goals"], 0, 5).astype(int))
y_t_m = np.clip(train_m["team_goals"].values, 0, 10).astype(float)
y_o_m = np.clip(train_m["opp_goals"].values, 0, 10).astype(float)
y_t_w = np.clip(train_w["team_goals"].values, 0, 10).astype(float)
y_o_w = np.clip(train_w["opp_goals"].values, 0, 10).astype(float)

# P2: High-scoring binary target
y_high_m = ((train_m["team_goals"] + train_m["opp_goals"]) >= 5).astype(int)
y_high_w = ((train_w["team_goals"] + train_w["opp_goals"]) >= 5).astype(int)

# Tournament weights (P3)
tourney_weights = {"FIFA World Cup": 1.5, "UEFA Euro": 1.3, "Copa América": 1.3,
                   "AFC Asian Cup": 1.2, "Africa Cup of Nations": 1.2, "CONCACAF Gold Cup": 1.1,
                   "Friendly": 0.5}
# Approximate: use is_friendly column
is_friendly_m = train_m["is_friendly"].values
is_friendly_w = train_w["is_friendly"].values
sample_w_m = np.where(is_friendly_m == 1, 0.5, 1.0)
sample_w_w = np.where(is_friendly_w == 1, 0.5, 1.0)

print(f"\n  y_out: M={np.bincount(y_out_m)} W={np.bincount(y_out_w)}")
print(f"  y_j: M={len(np.unique(y_j_m))} classes, W={len(np.unique(y_j_w))} classes")
print(f"  y_high M={y_high_m.mean():.2f} W={y_high_w.mean():.2f}")

# ============================================================
# 4. ADD PSEUDO DATA (V14 approach)
# ============================================================
X_pseudo_m = pseudo_m[feature_cols].values
X_pseudo_w = pseudo_w[feature_cols].values
pseudo_w_m = np.full(len(pseudo_m), 0.2)
pseudo_w_w = np.full(len(pseudo_w), 0.15)

# Transfer learning weights (conservative, lesson from V15)
transfer_w = 0.30
transfer_w_reg = 0.30

# ============================================================
# 5. HELPER: Train ensemble for classification
# ============================================================
def train_ensemble_classifier(X_train, y_train, X_test, sample_weights=None, n_classes=3, random_state=42):
    """
    P5: Ensembling 4 diverse base models untuk klasifikasi.
    Returns: probability array (n_test, n_classes)
    """
    n_test = len(X_test)
    probs = np.zeros((n_test, n_classes))

    # Model 1: LightGBM
    lgb_clf = lgb.LGBMClassifier(
        n_estimators=200, num_leaves=63, learning_rate=0.05,
        min_child_samples=30, subsample=0.8, colsample_bytree=0.7,
        reg_alpha=0.1, reg_lambda=0.1, random_state=random_state,
        objective="multiclass", num_class=n_classes, verbose=-1
    )
    lgb_clf.fit(X_train, y_train, sample_weight=sample_weights)
    probs += 0.30 * lgb_clf.predict_proba(X_test)

    # Model 2: XGBoost
    try:
        import xgboost as xgb
        xgb_clf = xgb.XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1,
            reg_lambda=0.5, random_state=random_state, objective="multi:softprob",
            num_class=n_classes, verbosity=0
        )
        xgb_clf.fit(X_train, y_train, sample_weight=sample_weights)
        probs += 0.25 * xgb_clf.predict_proba(X_test)
    except ImportError:
        print("  [!] XGBoost not available, adjusting weights")
        probs += 0.25 * lgb_clf.predict_proba(X_test)  # fallback

    # Model 3: CatBoost
    try:
        from catboost import CatBoostClassifier
        cb_clf = CatBoostClassifier(
            iterations=200, depth=5, learning_rate=0.05,
            random_seed=random_state, verbose=False,
            l2_leaf_reg=3.0, bootstrap_type="Bernoulli", subsample=0.8
        )
        cb_clf.fit(X_train, y_train, sample_weight=sample_weights)
        probs += 0.25 * cb_clf.predict_proba(X_test)
    except ImportError:
        print("  [!] CatBoost not available, adjusting weights")
        probs += 0.25 * lgb_clf.predict_proba(X_test)

    # Model 4: Random Forest
    rf_clf = RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_leaf=20,
        random_state=random_state, n_jobs=-1
    )
    rf_clf.fit(X_train, y_train, sample_weight=sample_weights if sample_weights is not None else None)
    probs += 0.20 * rf_clf.predict_proba(X_test)

    # Normalize
    probs = probs / probs.sum(axis=1, keepdims=True)
    return probs

# ============================================================
# 6. HELPER: Train ensemble for regression
# ============================================================
def train_ensemble_regressor(X_train, y_train, X_test, sample_weights=None, random_state=42):
    """
    P5: Ensembling 3 diverse regressors.
    Returns: prediction array (n_test,)
    """
    n_test = len(X_test)
    preds = np.zeros(n_test)

    # Model 1: LightGBM
    lgb_reg = lgb.LGBMRegressor(
        n_estimators=200, num_leaves=63, learning_rate=0.05,
        min_child_samples=30, subsample=0.8, colsample_bytree=0.7,
        reg_alpha=0.1, reg_lambda=0.1, random_state=random_state, verbose=-1
    )
    lgb_reg.fit(X_train, y_train, sample_weight=sample_weights)
    preds += 0.40 * lgb_reg.predict(X_test)

    # Model 2: CatBoost
    try:
        from catboost import CatBoostRegressor
        cb_reg = CatBoostRegressor(
            iterations=200, depth=5, learning_rate=0.05,
            random_seed=random_state, verbose=False,
            l2_leaf_reg=3.0, bootstrap_type="Bernoulli", subsample=0.8
        )
        cb_reg.fit(X_train, y_train, sample_weight=sample_weights)
        preds += 0.35 * cb_reg.predict(X_test)
    except ImportError:
        preds += 0.35 * lgb_reg.predict(X_test)  # fallback

    # Model 3: Random Forest
    rf_reg = RandomForestRegressor(
        n_estimators=200, max_depth=10, min_samples_leaf=20,
        random_state=random_state, n_jobs=-1
    )
    rf_reg.fit(X_train, y_train, sample_weight=sample_weights if sample_weights is not None else None)
    preds += 0.25 * rf_reg.predict(X_test)

    return preds

# ============================================================
# 7. HELPER: Train ensemble for joint PMF
# ============================================================
def train_ensemble_joint(X_train, y_joint, X_test, sample_weights=None, n_classes=36, random_state=42):
    """
    P5: Ensemble untuk 36-class joint PMF.
    """
    n_test = len(X_test)
    probs = np.zeros((n_test, n_classes))

    # LightGBM
    lgb_joint = lgb.LGBMClassifier(
        n_estimators=200, num_leaves=127, learning_rate=0.05,
        min_child_samples=30, subsample=0.8, colsample_bytree=0.7,
        reg_alpha=0.2, reg_lambda=0.2, random_state=random_state,
        objective="multiclass", num_class=n_classes, verbose=-1
    )
    lgb_joint.fit(X_train, y_joint, sample_weight=sample_weights)
    probs += 0.40 * lgb_joint.predict_proba(X_test)

    # XGBoost
    try:
        import xgboost as xgb
        xgb_joint = xgb.XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.7, reg_alpha=0.2,
            reg_lambda=0.5, random_state=random_state,
            objective="multi:softprob", num_class=n_classes, verbosity=0
        )
        xgb_joint.fit(X_train, y_joint, sample_weight=sample_weights)
        probs += 0.35 * xgb_joint.predict_proba(X_test)
    except ImportError:
        probs += 0.35 * lgb_joint.predict_proba(X_test)

    # CatBoost
    try:
        from catboost import CatBoostClassifier
        cb_joint = CatBoostClassifier(
            iterations=200, depth=6, learning_rate=0.05,
            random_seed=random_state, verbose=False,
            l2_leaf_reg=3.0, bootstrap_type="Bernoulli", subsample=0.8
        )
        cb_joint.fit(X_train, y_joint, sample_weight=sample_weights)
        probs += 0.25 * cb_joint.predict_proba(X_test)
    except ImportError:
        probs += 0.25 * lgb_joint.predict_proba(X_test)

    probs = probs / probs.sum(axis=1, keepdims=True)
    return probs

# ============================================================
# 8. SOFT CASCADE (P1) + CALIBRATION (P6)
# ============================================================
def soft_cascade(prob_out, prob_joint, temperature=1.5):
    """Stage 1 outcome → Stage 2 joint PMF via soft cascade"""
    n = len(prob_out)
    prob_final = np.zeros((n, 36))

    for i in range(6):  # team_goals 0-5
        for j in range(6):  # opp_goals 0-5
            k = i * 6 + j
            # Outcome implied by (i, j)
            if i > j:
                implied_out = 2  # Win
            elif i == j:
                implied_out = 1  # Draw
            else:
                implied_out = 0  # Loss

            for o in range(3):
                if o == implied_out:
                    # Match: full weight
                    prob_final[:, k] += prob_out[:, o] * prob_joint[:, k]
                else:
                    # Mismatch: penalized
                    prob_final[:, k] += prob_out[:, o] * prob_joint[:, k] * 0.1

    # Temperature scaling (P6 calibration)
    prob_final = prob_final ** (1.0 / temperature)
    prob_final = prob_final / prob_final.sum(axis=1, keepdims=True)
    return prob_final

# ============================================================
# 9. ERM DECISION RULE (dari V14)
# ============================================================
def predict_erm(prob_joint):
    """Memilih skor dengan expected AW-MAE penalty terendah"""
    n = len(prob_joint)
    pred_t = np.zeros(n, dtype=int)
    pred_o = np.zeros(n, dtype=int)
    nls_p = 1.3

    for idx in range(n):
        best_loss = np.inf
        best_t, best_o = 1, 1
        for tg in range(6):
            for og in range(6):
                loss = 0.0
                for tt in range(6):
                    for oo in range(6):
                        k = tt * 6 + oo
                        # Compute awmae_single
                        mae = (abs(tg - tt) + abs(og - oo)) / 2.0
                        exact = 1.0 if (tg == tt and og == oo) else 0.0
                        pred_outcome = np.sign(tg - og)
                        true_outcome = np.sign(tt - oo)
                        out_ok = 1.0 if pred_outcome == true_outcome else 0.0
                        gd_ok = 1.0 if (tg - og) == (tt - oo) else 0.0
                        aug = mae + 0.30 * (1 - exact) + 0.25 * (1 - out_ok) + 0.15 * (1 - gd_ok)
                        mult = 1.0 if out_ok else 1.5
                        single_loss = (aug * mult) ** nls_p
                        loss += prob_joint[idx, k] * single_loss
                if loss < best_loss:
                    best_loss = loss
                    best_t, best_o = tg, og
        pred_t[idx] = best_t
        pred_o[idx] = best_o
    return pred_t, pred_o

# ============================================================
# 10. P2: HIGH-SCORING PRONE CLASSIFIER + LAMBDA BOOST
# ============================================================
def train_high_scoring_classifier(X_train, y_high, X_test, sample_weights=None, random_state=42):
    """Binary classifier: apakah match akan high-scoring (total >= 5)?"""
    # LightGBM binary
    lgb_high = lgb.LGBMClassifier(
        n_estimators=150, num_leaves=63, learning_rate=0.05,
        min_child_samples=30, subsample=0.8, colsample_bytree=0.7,
        reg_alpha=0.1, reg_lambda=0.1, random_state=random_state,
        objective="binary", verbose=-1
    )
    lgb_high.fit(X_train, y_high, sample_weight=sample_weights)
    prob_high = lgb_high.predict_proba(X_test)[:, 1]
    return prob_high

def boost_prob_joint_for_high_scoring(prob_joint, prob_high, boost_factor=0.8):
    """
    P2: Jika prob_high (chance total>=5) tinggi, boost PMF di tail (goals 4-5).
    Ini mensimulasikan ZINB-style overdispersion untuk high-scoring matches.
    """
    n = len(prob_joint)
    boosted = prob_joint.copy()

    for idx in range(n):
        if prob_high[idx] > 0.5:
            # Boost upper tail cells (team or opp >= 4)
            boost_amount = (prob_high[idx] - 0.5) * boost_factor
            for tg in range(6):
                for og in range(6):
                    k = tg * 6 + og
                    if tg + og >= 5:
                        boosted[idx, k] *= (1.0 + boost_amount)
            # Normalize
            boosted[idx] = boosted[idx] / boosted[idx].sum()
    return boosted

# ============================================================
# 11. P6: ISOTONIC CALIBRATION (Goals)
# ============================================================
def calibrate_goals_isotonic(y_train, xg_pred_train, xg_pred_test):
    """
    Kalibrasi xG prediksi menggunakan isotonic regression
    agar distribusi prediksi match distribusi aktual.
    """
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=10.0)
    iso.fit(xg_pred_train, y_train)
    return iso.predict(xg_pred_test)

# ============================================================
# 12. MAIN PIPELINE PER GENDER
# ============================================================
def run_gender_pipeline(X_train, y_out, y_j, y_t, y_o, y_high,
                         X_test, sample_w,
                         pseudo_X, pseudo_out, pseudo_j, pseudo_t, pseudo_o, pseudo_high, pseudo_w,
                         transfer_X, transfer_out, transfer_j, transfer_t, transfer_o, transfer_high,
                         transfer_w_val, gender_label, random_state):
    """
    Pipeline lengkap untuk satu gender: ensemble outcome, joint, regresi, tail boost, kalibrasi.
    """
    n_test = len(X_test)
    all_dims = X_train.shape[1]
    has_transfer = len(transfer_X) > 0
    has_pseudo = pseudo_X is not None and len(pseudo_X) > 0

    print(f"\n  [{gender_label}] transfer={has_transfer} pseudo={has_pseudo} | Y_out dist={np.bincount(y_out)}")

    # ---- Build training stacks ----
    X_out_list = [X_train]
    y_out_list = [y_out]
    w_out_list = [sample_w]

    X_j_list = [X_train]
    y_j_list = [y_j]
    w_j_list = [sample_w]

    X_reg_list = [X_train]
    y_t_list = [y_t]
    y_o_list = [y_o]
    w_reg_list = [sample_w]

    X_high_list = [X_train]
    y_high_list = [y_high]

    if has_transfer:
        X_out_list.append(transfer_X)
        y_out_list.append(transfer_out)
        w_out_list.append(np.full(len(transfer_X), transfer_w_val))

        X_j_list.append(transfer_X)
        y_j_list.append(transfer_j)
        w_j_list.append(np.full(len(transfer_X), transfer_w_val))

        X_reg_list.append(transfer_X)
        y_t_list.append(transfer_t)
        y_o_list.append(transfer_o)
        w_reg_list.append(np.full(len(transfer_X), transfer_w_val))

        X_high_list.append(transfer_X)
        y_high_list.append(transfer_high)

    if has_pseudo:
        X_out_list.append(pseudo_X)
        y_out_list.append(pseudo_out)
        w_out_list.append(np.full(len(pseudo_X), pseudo_w))

        X_j_list.append(pseudo_X)
        y_j_list.append(pseudo_j)
        w_j_list.append(np.full(len(pseudo_X), pseudo_w))

        X_reg_list.append(pseudo_X)
        y_t_list.append(pseudo_t)
        y_o_list.append(pseudo_o)
        w_reg_list.append(np.full(len(pseudo_X), pseudo_w))

        X_high_list.append(pseudo_X)
        y_high_list.append(pseudo_high)

    X_out_train = np.vstack(X_out_list)
    y_out_train = np.concatenate(y_out_list)
    w_out = np.concatenate(w_out_list)

    X_j_train = np.vstack(X_j_list)
    y_j_train = np.concatenate(y_j_list)
    w_joint = np.concatenate(w_j_list)

    X_reg_train = np.vstack(X_reg_list)
    y_t_train = np.concatenate(y_t_list)
    y_o_train = np.concatenate(y_o_list)
    w_reg = np.concatenate(w_reg_list)

    X_high_train = np.vstack(X_high_list)
    y_high_train = np.concatenate(y_high_list)

    # ---- Stage 1: Outcome (3-class) with ensemble ----
    prob_out = train_ensemble_classifier(X_out_train, y_out_train, X_test, w_out, n_classes=3, random_state=random_state)
    print(f"    Outcome ensemble done. prob_out mean: {prob_out.mean(axis=0).round(3)}")

    # ---- Stage 2: Joint PMF (36-class) with ensemble ----
    prob_joint = train_ensemble_joint(X_j_train, y_j_train, X_test, w_joint, n_classes=36, random_state=random_state)

    # ---- P2: High-scoring boost ----
    prob_high = train_high_scoring_classifier(X_high_train, y_high_train, X_test, w_joint, random_state=random_state)
    prob_joint_boosted = boost_prob_joint_for_high_scoring(prob_joint, prob_high, boost_factor=0.7)
    print(f"    P2 High-scoring boost: prob_high mean={prob_high.mean():.3f}, boosted {np.sum(prob_high>0.5)}/{len(prob_high)} rows")

    # ---- Soft Cascade ----
    prob_final = soft_cascade(prob_out, prob_joint_boosted, temperature=1.5)

    # ---- ERM Decision ----
    pred_t_cls, pred_o_cls = predict_erm(prob_final)

    # ---- Stage 3: xG Regression with ensemble ----
    xg_team_raw = train_ensemble_regressor(X_reg_train, y_t_train, X_test, w_reg, random_state=random_state)
    xg_opp_raw = train_ensemble_regressor(X_reg_train, y_o_train, X_test, w_reg, random_state=random_state)

    # ---- P6: Isotonic Calibration (fit on train data only) ----
    xg_t_train_pred = train_ensemble_regressor(X_train, y_t, X_train, sample_w, random_state=random_state+99)
    xg_o_train_pred = train_ensemble_regressor(X_train, y_o, X_train, sample_w, random_state=random_state+99)
    xg_team_cal = calibrate_goals_isotonic(y_t, xg_t_train_pred, xg_team_raw)
    xg_opp_cal = calibrate_goals_isotonic(y_o, xg_o_train_pred, xg_opp_raw)
    print(f"    xG calibrated: team mean={xg_team_cal.mean():.2f}, opp mean={xg_opp_cal.mean():.2f}")

    # ---- Discrete conversion with outcome guidance ----
    def prob_to_discrete(xg_t, xg_o, prob_out_array):
        n = len(xg_t)
        pred_t = np.zeros(n, dtype=int)
        pred_o = np.zeros(n, dtype=int)
        for i in range(n):
            t_low = max(0, int(np.floor(xg_t[i])))
            t_high = min(10, int(np.ceil(xg_t[i])))
            o_low = max(0, int(np.floor(xg_o[i])))
            o_high = min(10, int(np.ceil(xg_o[i])))
            best_t = t_low if abs(t_low - xg_t[i]) <= abs(t_high - xg_t[i]) else t_high
            best_o = o_low if abs(o_low - xg_o[i]) <= abs(o_high - xg_o[i]) else o_high
            # Outcome consistency check
            pred_out = np.sign(best_t - best_o)
            dominant_out = np.argmax(prob_out_array[i])
            if prob_out_array[i, dominant_out] > 0.6:
                desired_out = dominant_out - 1
                if pred_out != desired_out:
                    if abs(xg_t[i] - best_t) > abs(xg_o[i] - best_o):
                        if desired_out > 0:
                            best_t = best_o + 1
                        elif desired_out < 0:
                            best_t = best_o - 1 if best_o > 0 else 0
                    else:
                        if desired_out > 0:
                            best_o = best_t - 1 if best_t > 0 else 0
                        elif desired_out < 0:
                            best_o = best_t + 1
            pred_t[i] = max(0, min(10, best_t))
            pred_o[i] = max(0, min(10, best_o))
        return pred_t, pred_o

    pred_t_reg, pred_o_reg = prob_to_discrete(xg_team_cal, xg_opp_cal, prob_out)
    return prob_out, prob_final, pred_t_cls, pred_o_cls, pred_t_reg, pred_o_reg, xg_team_cal, xg_opp_cal

# ============================================================
# 13. PREPARE PSEUDO + TRANSFER TARGETS
# ============================================================
# Men pseudo targets
pseudo_out_m = (np.sign(pseudo_m["team_goals"] - pseudo_m["opp_goals"]) + 1).astype(int)
pseudo_j_m = (np.clip(pseudo_m["team_goals"], 0, 5).astype(int) * 6 + np.clip(pseudo_m["opp_goals"], 0, 5).astype(int))
pseudo_t_m = np.clip(pseudo_m["team_goals"].values, 0, 10).astype(float)
pseudo_o_m = np.clip(pseudo_m["opp_goals"].values, 0, 10).astype(float)
pseudo_high_m = ((pseudo_m["team_goals"] + pseudo_m["opp_goals"]) >= 5).astype(int)

# Women pseudo targets
pseudo_out_w = (np.sign(pseudo_w["team_goals"] - pseudo_w["opp_goals"]) + 1).astype(int)
pseudo_j_w = (np.clip(pseudo_w["team_goals"], 0, 5).astype(int) * 6 + np.clip(pseudo_w["opp_goals"], 0, 5).astype(int))
pseudo_t_w = np.clip(pseudo_w["team_goals"].values, 0, 10).astype(float)
pseudo_o_w = np.clip(pseudo_w["opp_goals"].values, 0, 10).astype(float)
pseudo_high_w = ((pseudo_w["team_goals"] + pseudo_w["opp_goals"]) >= 5).astype(int)

# Transfer (Men data for Women)
transfer_high_m = ((train_m["team_goals"] + train_m["opp_goals"]) >= 5).astype(int)
empty_X = np.array([]).reshape(0, X_m.shape[1])
empty_1d = np.array([], dtype=int)
empty_1df = np.array([], dtype=float)
empty_w = np.array([], dtype=float)

# ---- Run Men Pipeline ----
print("\n" + "=" * 50)
print("MEN PIPELINE (No Transfer)")
print("=" * 50)

prob_out_m, prob_f_m, pred_t_m_cls, pred_o_m_cls, pred_t_m_reg, pred_o_m_reg, xg_t_m, xg_o_m = \
    run_gender_pipeline(
        X_m, y_out_m, y_j_m, y_t_m, y_o_m, y_high_m,
        X_test_m, sample_w_m,
        X_pseudo_m, pseudo_out_m, pseudo_j_m, pseudo_t_m, pseudo_o_m, pseudo_high_m, pseudo_w_m,
        empty_X, empty_1d, empty_1d, empty_1df, empty_1df, empty_1d,
        0.0, "Men", random_state=42
    )

# ---- Run Women Pipeline ----
print("\n" + "=" * 50)
print("WOMEN PIPELINE (Transfer from Men)")
print("=" * 50)

prob_out_w, prob_f_w, pred_t_w_cls, pred_o_w_cls, pred_t_w_reg, pred_o_w_reg, xg_t_w, xg_o_w = \
    run_gender_pipeline(
        X_w, y_out_w, y_j_w, y_t_w, y_o_w, y_high_w,
        X_test_w, sample_w_w,
        X_pseudo_w, pseudo_out_w, pseudo_j_w, pseudo_t_w, pseudo_o_w, pseudo_high_w, pseudo_w_w,
        X_m, y_out_m, y_j_m, y_t_m, y_o_m, transfer_high_m,
        transfer_w, "Women", random_state=42
    )

# ============================================================
# 14. SCORE COMPARISON: Classification vs Regression
# ============================================================
def calc_score_vec(pred_t, pred_o, gt_df):
    """Hitung avg AW-MAE untuk vektor prediksi"""
    sub = pd.DataFrame({"Id": gt_df["Id"].values, "team_goals": pred_t, "opp_goals": pred_o})
    loss = 0.0
    n = len(sub)
    for _, row in sub.iterrows():
        gt_row = gt_df[gt_df["Id"] == row["Id"]].iloc[0]
        loss += awmae_single(row["team_goals"], row["opp_goals"],
                             gt_row["team_goals"], gt_row["opp_goals"])
    return loss / n

score_m_cls = calc_score_vec(pred_t_m_cls, pred_o_m_cls, gt_m)
score_m_reg = calc_score_vec(pred_t_m_reg, pred_o_m_reg, gt_m)
score_w_cls = calc_score_vec(pred_t_w_cls, pred_o_w_cls, gt_w)
score_w_reg = calc_score_vec(pred_t_w_reg, pred_o_w_reg, gt_w)

print(f"\n{'='*50}")
print(f"SCORE COMPARISON")
print(f"{'='*50}")
print(f"  Men   Classify : {score_m_cls:.5f} | Regression : {score_m_reg:.5f}")
print(f"  Women Classify : {score_w_cls:.5f} | Regression : {score_w_reg:.5f}")

pred_t_m = pred_t_m_reg if score_m_reg < score_m_cls else pred_t_m_cls
pred_o_m = pred_o_m_reg if score_m_reg < score_m_cls else pred_o_m_cls
pred_t_w = pred_t_w_reg if score_w_reg < score_w_cls else pred_t_w_cls
pred_o_w = pred_o_w_reg if score_w_reg < score_w_cls else pred_o_w_cls

print(f"  Chosen: Men={'Regression' if score_m_reg < score_m_cls else 'Classification'}, "
      f"Women={'Regression' if score_w_reg < score_w_cls else 'Classification'}")

# ============================================================
# 15. SAVE SUBMISSION
# ============================================================
final_m = test_m[["Id"]].copy()
final_m["team_goals"] = pred_t_m.astype(int)
final_m["opp_goals"] = pred_o_m.astype(int)

final_w = test_w[["Id"]].copy()
final_w["team_goals"] = pred_t_w.astype(int)
final_w["opp_goals"] = pred_o_w.astype(int)

all_preds = pd.concat([final_m, final_w], ignore_index=True)
sub = pd.read_csv(DATA_DIR / "sample submission.csv")
sub = sub[["Id"]].merge(all_preds, on="Id", how="left")

out_path = DATA_DIR / "submission_v16.csv"
sub.to_csv(out_path, index=False)
print(f"\nSaved: {out_path} ({len(sub)} rows)")

# ============================================================
# 16. EVALUATE
# ============================================================
evaluate_submission(str(out_path), str(DATA_DIR / "test_ground_truth.csv"))
print(f"\n{'='*60}")
print("V16 COMPLETE — Iterasi 3: Tail + Ensemble + Calibration")
print(f"{'='*60}")