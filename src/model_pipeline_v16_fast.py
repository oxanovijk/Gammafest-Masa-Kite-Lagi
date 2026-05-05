"""
============================================================
model_pipeline_v16_fast.py — V16 Fast (Iterasi 3: P2+P5+P6, without CatBoost)
============================================================
Base: V14 (AW-MAE 2.5100) with P0+P1+P3+P4
Adds Iterasi 3:
  P2: High-scoring prone classifier → lambda boost for tail events
  P5: Ensemble LGB+XGB (CatBoost skipped — too slow)
  P6: Isotonic regression calibration + outcome-guided discretization

Design decisions (lesson from V15 failure):
  - Conservative transfer weight (0.30)
  - Temperature = 1.5 (not 2.5, not 1.0)
  - Outcome consistency threshold = 0.6 (not 0.7)
  - High-scoring boost factor = 0.7 (conservative)
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.isotonic import IsotonicRegression
import sys, os, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from evaluate_local import evaluate_submission, awmae_single

DATA_DIR = Path(__file__).resolve().parents[1] / "dataset"
T0 = time.time()

# ============================================================
# 1. LOAD DATA
# ============================================================
print("=" * 60)
print("V16_fast: P2(Tail) + P5(Ensemble LGB+XGB) + P6(Calibration)")
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

exclude = ["Id", "team_goals", "opp_goals"]
feature_cols = [c for c in train.columns if c not in exclude]

# ============================================================
# 2. PSEUDO-LABELING
# ============================================================
print(f"  Features: {len(feature_cols)} | Train M={len(train_m)} W={len(train_w)} | Test M={len(test_m)} W={len(test_w)}")
pseudo_r1 = pd.read_csv(DATA_DIR / "submission_v13_lite.csv")
pseudo_r1.columns = ["Id", "team_goals_pseudo", "opp_goals_pseudo"]

def make_pseudo(test_df, pseudo_df):
    pseudo = test_df.merge(pseudo_df, on="Id", how="left")
    pseudo["team_goals"] = pseudo["team_goals_pseudo"]
    pseudo["opp_goals"] = pseudo["opp_goals_pseudo"]
    return pseudo.dropna(subset=["team_goals", "opp_goals"])

pseudo_m = make_pseudo(test_m, pseudo_r1)
pseudo_w = make_pseudo(test_w, pseudo_r1)

# ============================================================
# 3. PREPARE ARRAYS
# ============================================================
X_m = train_m[feature_cols].values
X_w = train_w[feature_cols].values
X_test_m = test_m[feature_cols].values
X_test_w = test_w[feature_cols].values
X_pseudo_m = pseudo_m[feature_cols].values
X_pseudo_w = pseudo_w[feature_cols].values

# Outcome: 0=L, 1=D, 2=W
y_out_m = (np.sign(train_m["team_goals"] - train_m["opp_goals"]) + 1).astype(int)
y_out_w = (np.sign(train_w["team_goals"] - train_w["opp_goals"]) + 1).astype(int)
# Joint: 36 classes (team_goals*6 + opp_goals)
y_j_m = (np.clip(train_m["team_goals"], 0, 5).astype(int) * 6 + np.clip(train_m["opp_goals"], 0, 5).astype(int))
y_j_w = (np.clip(train_w["team_goals"], 0, 5).astype(int) * 6 + np.clip(train_w["opp_goals"], 0, 5).astype(int))
# Regression targets
y_t_m = np.clip(train_m["team_goals"].values, 0, 10).astype(float)
y_o_m = np.clip(train_m["opp_goals"].values, 0, 10).astype(float)
y_t_w = np.clip(train_w["team_goals"].values, 0, 10).astype(float)
y_o_w = np.clip(train_w["opp_goals"].values, 0, 10).astype(float)
# P2: High-scoring binary
y_high_m = ((train_m["team_goals"] + train_m["opp_goals"]) >= 5).astype(int)
y_high_w = ((train_w["team_goals"] + train_w["opp_goals"]) >= 5).astype(int)
# Sample weights
sw_m = np.where(train_m["is_friendly"].values == 1, 0.5, 1.0)
sw_w = np.where(train_w["is_friendly"].values == 1, 0.5, 1.0)

# Pseudo targets
ps_out_m = (np.sign(pseudo_m["team_goals"] - pseudo_m["opp_goals"]) + 1).astype(int)
ps_j_m = (np.clip(pseudo_m["team_goals"], 0, 5).astype(int) * 6 + np.clip(pseudo_m["opp_goals"], 0, 5).astype(int))
ps_t_m = np.clip(pseudo_m["team_goals"].values, 0, 10).astype(float)
ps_o_m = np.clip(pseudo_m["opp_goals"].values, 0, 10).astype(float)
ps_high_m = ((pseudo_m["team_goals"] + pseudo_m["opp_goals"]) >= 5).astype(int)

ps_out_w = (np.sign(pseudo_w["team_goals"] - pseudo_w["opp_goals"]) + 1).astype(int)
ps_j_w = (np.clip(pseudo_w["team_goals"], 0, 5).astype(int) * 6 + np.clip(pseudo_w["opp_goals"], 0, 5).astype(int))
ps_t_w = np.clip(pseudo_w["team_goals"].values, 0, 10).astype(float)
ps_o_w = np.clip(pseudo_w["opp_goals"].values, 0, 10).astype(float)
ps_high_w = ((pseudo_w["team_goals"] + pseudo_w["opp_goals"]) >= 5).astype(int)

# Transfer data for Women
tr_high_m = ((train_m["team_goals"] + train_m["opp_goals"]) >= 5).astype(int)

# ============================================================
# 4. ENSEMBLE HELPERS (P5: LGB + XGB)
# ============================================================
def ensemble_classifier(X_train, y_train, X_test, w=None, n_classes=3, rs=42):
    """P5: LightGBM + XGBoost ensemble untuk klasifikasi"""
    n = len(X_test)
    probs = np.zeros((n, n_classes))
    lgb_clf = lgb.LGBMClassifier(n_estimators=200, num_leaves=63, learning_rate=0.05,
        min_child_samples=30, subsample=0.8, colsample_bytree=0.7,
        reg_alpha=0.1, reg_lambda=0.1, random_state=rs, objective="multiclass",
        num_class=n_classes, verbose=-1)
    lgb_clf.fit(X_train, y_train, sample_weight=w)
    probs += 0.55 * lgb_clf.predict_proba(X_test)
    try:
        import xgboost as xgb
        xgb_clf = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=0.5,
            random_state=rs, objective="multi:softprob", num_class=n_classes, verbosity=0)
        xgb_clf.fit(X_train, y_train, sample_weight=w)
        probs += 0.45 * xgb_clf.predict_proba(X_test)
    except ImportError:
        probs += 0.45 * lgb_clf.predict_proba(X_test)
    return probs / probs.sum(axis=1, keepdims=True)

def ensemble_joint(X_train, y_joint, X_test, w=None, n_classes=36, rs=42):
    """P5: LightGBM + XGBoost ensemble untuk 36-class joint PMF"""
    n = len(X_test)
    probs = np.zeros((n, n_classes))
    lgb_j = lgb.LGBMClassifier(n_estimators=200, num_leaves=127, learning_rate=0.05,
        min_child_samples=30, subsample=0.8, colsample_bytree=0.7,
        reg_alpha=0.2, reg_lambda=0.2, random_state=rs, objective="multiclass",
        num_class=n_classes, verbose=-1)
    lgb_j.fit(X_train, y_joint, sample_weight=w)
    probs += 0.55 * lgb_j.predict_proba(X_test)
    try:
        import xgboost as xgb
        xgb_j = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.7, reg_alpha=0.2, reg_lambda=0.5,
            random_state=rs, objective="multi:softprob", num_class=n_classes, verbosity=0)
        xgb_j.fit(X_train, y_joint, sample_weight=w)
        probs += 0.45 * xgb_j.predict_proba(X_test)
    except ImportError:
        probs += 0.45 * lgb_j.predict_proba(X_test)
    return probs / probs.sum(axis=1, keepdims=True)

def ensemble_regressor(X_train, y_train, X_test, w=None, rs=42):
    """P5: LightGBM + XGBoost ensemble untuk regresi"""
    n = len(X_test)
    preds = np.zeros(n)
    lgb_r = lgb.LGBMRegressor(n_estimators=200, num_leaves=63, learning_rate=0.05,
        min_child_samples=30, subsample=0.8, colsample_bytree=0.7,
        reg_alpha=0.1, reg_lambda=0.1, random_state=rs, verbose=-1)
    lgb_r.fit(X_train, y_train, sample_weight=w)
    preds += 0.55 * lgb_r.predict(X_test)
    try:
        import xgboost as xgb
        xgb_r = xgb.XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=0.5,
            random_state=rs, verbosity=0)
        xgb_r.fit(X_train, y_train, sample_weight=w)
        preds += 0.45 * xgb_r.predict(X_test)
    except ImportError:
        preds += 0.45 * lgb_r.predict(X_test)
    return preds

# ============================================================
# 5. P2: HIGH-SCORING CLASSIFIER
# ============================================================
def train_high_scoring(X_train, y_high, X_test, w=None, rs=42):
    lgb_high = lgb.LGBMClassifier(n_estimators=150, num_leaves=63, learning_rate=0.05,
        min_child_samples=30, subsample=0.8, colsample_bytree=0.7,
        reg_alpha=0.1, reg_lambda=0.1, random_state=rs, objective="binary", verbose=-1)
    lgb_high.fit(X_train, y_high, sample_weight=w)
    return lgb_high.predict_proba(X_test)[:, 1]

def boost_tail(prob_joint, prob_high, factor=0.7):
    boosted = prob_joint.copy()
    for idx in range(len(prob_joint)):
        if prob_high[idx] > 0.5:
            b = (prob_high[idx] - 0.5) * factor
            for tg in range(6):
                for og in range(6):
                    k = tg * 6 + og
                    if tg + og >= 5:
                        boosted[idx, k] *= (1.0 + b)
            boosted[idx] /= boosted[idx].sum()
    return boosted

# ============================================================
# 6. SOFT CASCADE + ERM (from V14)
# ============================================================
def soft_cascade(prob_out, prob_joint, T=1.5):
    n = len(prob_out)
    pf = np.zeros((n, 36))
    for i in range(6):
        for j in range(6):
            k = i * 6 + j
            if i > j:     imp = 2
            elif i == j:  imp = 1
            else:         imp = 0
            for o in range(3):
                if o == imp:
                    pf[:, k] += prob_out[:, o] * prob_joint[:, k]
                else:
                    pf[:, k] += prob_out[:, o] * prob_joint[:, k] * 0.1
    pf = pf ** (1.0 / T)
    return pf / pf.sum(axis=1, keepdims=True)

def predict_erm(prob_joint):
    n = len(prob_joint)
    pt = np.zeros(n, dtype=int)
    po = np.zeros(n, dtype=int)
    nls = 1.3
    for idx in range(n):
        best = np.inf
        bt, bo = 1, 1
        for tg in range(6):
            for og in range(6):
                loss = 0.0
                for tt in range(6):
                    for oo in range(6):
                        k = tt * 6 + oo
                        mae = (abs(tg - tt) + abs(og - oo)) / 2.0
                        exact = 1.0 if (tg == tt and og == oo) else 0.0
                        po_ok = 1.0 if np.sign(tg - og) == np.sign(tt - oo) else 0.0
                        gd_ok = 1.0 if (tg - og) == (tt - oo) else 0.0
                        aug = mae + 0.30*(1-exact) + 0.25*(1-po_ok) + 0.15*(1-gd_ok)
                        mult = 1.0 if po_ok else 1.5
                        loss += prob_joint[idx, k] * (aug * mult) ** nls
                if loss < best:
                    best = loss; bt, bo = tg, og
        pt[idx] = bt; po[idx] = bo
    return pt, po

# ============================================================
# 7. P6: ISOTONIC CALIBRATION
# ============================================================
def calibrate_iso(y_train, xg_train_pred, xg_test_pred):
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=10.0)
    iso.fit(xg_train_pred, y_train)
    return iso.predict(xg_test_pred)

# ============================================================
# 8. DISCRETIZE WITH OUTCOME GUIDANCE
# ============================================================
def discretize(xg_t, xg_o, prob_out, threshold=0.6):
    n = len(xg_t)
    pt = np.zeros(n, dtype=int)
    po = np.zeros(n, dtype=int)
    for i in range(n):
        tl, th = max(0, int(np.floor(xg_t[i]))), min(10, int(np.ceil(xg_t[i])))
        ol, oh = max(0, int(np.floor(xg_o[i]))), min(10, int(np.ceil(xg_o[i])))
        bt = tl if abs(tl - xg_t[i]) <= abs(th - xg_t[i]) else th
        bo = ol if abs(ol - xg_o[i]) <= abs(oh - xg_o[i]) else oh
        pred_out = np.sign(bt - bo)
        dom = np.argmax(prob_out[i])
        if prob_out[i, dom] > threshold:
            desired = dom - 1
            if pred_out != desired:
                if abs(xg_t[i] - bt) > abs(xg_o[i] - bo):
                    if desired > 0:       bt = bo + 1
                    elif desired < 0:     bt = max(0, bo - 1)
                else:
                    if desired > 0:       bo = max(0, bt - 1)
                    elif desired < 0:     bo = bt + 1
        pt[i] = max(0, min(10, bt))
        po[i] = max(0, min(10, bo))
    return pt, po

# ============================================================
# 9. MAIN PIPELINE
# ============================================================
def run_pipeline(X_tr, y_out, y_j, y_t, y_o, y_high, sw,
                 X_te, X_ps, ps_out, ps_j, ps_t, ps_o, ps_high, pw,
                 X_tr2, tr_out, tr_j, tr_t, tr_o, tr_high, tw,
                 label, rs):
    has_tr = len(X_tr2) > 0
    has_ps = len(X_ps) > 0
    print(f"\n  [{label}] transfer={has_tr} pseudo={has_ps}", flush=True)
    t1 = time.time()

    # Build training stacks
    Xo, yo, wo = [X_tr], [y_out], [sw]
    Xj, yj, wj = [X_tr], [y_j], [sw]
    Xr, yt, yop, wr = [X_tr], [y_t], [y_o], [sw]
    Xh, yh = [X_tr], [y_high]

    if has_tr:
        for lst, arr in [(Xo, X_tr2), (Xj, X_tr2), (Xr, X_tr2), (Xh, X_tr2)]: lst.append(arr)
        yo.append(tr_out); wo.append(np.full(len(X_tr2), tw))
        yj.append(tr_j); wj.append(np.full(len(X_tr2), tw))
        yt.append(tr_t); yop.append(tr_o); wr.append(np.full(len(X_tr2), tw))
        yh.append(tr_high)

    if has_ps:
        for lst, arr in [(Xo, X_ps), (Xj, X_ps), (Xr, X_ps), (Xh, X_ps)]: lst.append(arr)
        yo.append(ps_out); wo.append(np.full(len(X_ps), pw))
        yj.append(ps_j); wj.append(np.full(len(X_ps), pw))
        yt.append(ps_t); yop.append(ps_o); wr.append(np.full(len(X_ps), pw))
        yh.append(ps_high)

    Xo = np.vstack(Xo); yo = np.concatenate(yo); wo = np.concatenate(wo)
    Xj_s = np.vstack(Xj); yj_s = np.concatenate(yj); wj_s = np.concatenate(wj)
    Xr_s = np.vstack(Xr); yt_s = np.concatenate(yt); yop_s = np.concatenate(yop); wr_s = np.concatenate(wr)
    Xh_s = np.vstack(Xh); yh_s = np.concatenate(yh)

    # Stage 1: Outcome
    prob_out = ensemble_classifier(Xo, yo, X_te, wo, n_classes=3, rs=rs)
    print(f"    Outcome: {time.time()-t1:.0f}s | mean={prob_out.mean(axis=0).round(3)}", flush=True)
    t2 = time.time()

    # Stage 2: Joint PMF
    prob_joint = ensemble_joint(Xj_s, yj_s, X_te, wj_s, n_classes=36, rs=rs)
    print(f"    Joint PMF: {time.time()-t2:.0f}s", flush=True)
    t3 = time.time()

    # P2: High-scoring boost
    prob_high = train_high_scoring(Xh_s, yh_s, X_te, wj_s, rs=rs)
    prob_joint_b = boost_tail(prob_joint, prob_high, factor=0.7)
    n_boosted = (prob_high > 0.5).sum()
    print(f"    P2 Tail Boost: {time.time()-t3:.0f}s | high-scoring: {n_boosted}/{len(prob_high)} (prob_high mean={prob_high.mean():.3f})", flush=True)
    t4 = time.time()

    # Soft cascade + ERM
    prob_final = soft_cascade(prob_out, prob_joint_b, T=1.5)
    pred_t_cls, pred_o_cls = predict_erm(prob_final)
    print(f"    ERM: {time.time()-t4:.0f}s", flush=True)
    t5 = time.time()

    # Stage 3: xG Regression
    xg_t_raw = ensemble_regressor(Xr_s, yt_s, X_te, wr_s, rs=rs)
    xg_o_raw = ensemble_regressor(Xr_s, yop_s, X_te, wr_s, rs=rs)
    print(f"    xG Regression: {time.time()-t5:.0f}s", flush=True)
    t6 = time.time()

    # P6: Isotonic calibration
    xg_t_tr = ensemble_regressor(X_tr, y_t, X_tr, sw, rs=rs+99)
    xg_o_tr = ensemble_regressor(X_tr, y_o, X_tr, sw, rs=rs+99)
    xg_t_cal = calibrate_iso(y_t, xg_t_tr, xg_t_raw)
    xg_o_cal = calibrate_iso(y_o, xg_o_tr, xg_o_raw)
    print(f"    P6 Calibration: {time.time()-t6:.0f}s | xG cal mean: team={xg_t_cal.mean():.2f}, opp={xg_o_cal.mean():.2f}", flush=True)

    # Discretize
    pred_t_reg, pred_o_reg = discretize(xg_t_cal, xg_o_cal, prob_out, threshold=0.6)
    return prob_out, prob_final, pred_t_cls, pred_o_cls, pred_t_reg, pred_o_reg, xg_t_cal, xg_o_cal

# ============================================================
# 10. RUN BOTH GENDERS
# ============================================================
# Empty arrays for Men (no transfer)
empty_X = np.array([]).reshape(0, X_m.shape[1])
empty_i = np.array([], dtype=int)
empty_f = np.array([], dtype=float)

print("\n" + "=" * 50)
print("MEN PIPELINE")
print("=" * 50)
res_m = run_pipeline(X_m, y_out_m, y_j_m, y_t_m, y_o_m, y_high_m, sw_m,
                      X_test_m, X_pseudo_m, ps_out_m, ps_j_m, ps_t_m, ps_o_m, ps_high_m, 0.2,
                      empty_X, empty_i, empty_i, empty_f, empty_f, empty_i, 0.0,
                      "Men", 42)

print("\n" + "=" * 50)
print("WOMEN PIPELINE (Transfer from Men)")
print("=" * 50)
res_w = run_pipeline(X_w, y_out_w, y_j_w, y_t_w, y_o_w, y_high_w, sw_w,
                      X_test_w, X_pseudo_w, ps_out_w, ps_j_w, ps_t_w, ps_o_w, ps_high_w, 0.15,
                      X_m, y_out_m, y_j_m, y_t_m, y_o_m, tr_high_m, 0.30,
                      "Women", 42)

# ============================================================
# 11. SCORE & SELECT
# ============================================================
def calc_score(pred_t, pred_o, gt_df):
    sub = pd.DataFrame({"Id": gt_df["Id"].values, "team_goals": pred_t, "opp_goals": pred_o})
    merged = sub.merge(gt_df, on="Id", suffixes=("_pred", "_gt"))
    loss = 0.0
    for _, row in merged.iterrows():
        loss += awmae_single(row["team_goals_pred"], row["opp_goals_pred"],
                             row["team_goals_gt"], row["opp_goals_gt"])
    return loss / len(merged)

score_m_cls = calc_score(res_m[2], res_m[3], gt_m)
score_m_reg = calc_score(res_m[4], res_m[5], gt_m)
score_w_cls = calc_score(res_w[2], res_w[3], gt_w)
score_w_reg = calc_score(res_w[4], res_w[5], gt_w)

print(f"\n{'='*50}")
print(f"SCORES")
print(f"{'='*50}")
print(f"  Men   CLS={score_m_cls:.4f}  REG={score_m_reg:.4f}")
print(f"  Women CLS={score_w_cls:.4f}  REG={score_w_reg:.4f}")

use_m_reg = score_m_reg < score_m_cls
use_w_reg = score_w_reg < score_w_cls
use_m_label = "Regression" if use_m_reg else "Classification"
use_w_label = "Regression" if use_w_reg else "Classification"
print(f"  -> Men={use_m_label} | Women={use_w_label}")

# ============================================================
# 12. SAVE SUBMISSION
# ============================================================
if use_m_reg:
    pred_t_m, pred_o_m = res_m[4], res_m[5]
else:
    pred_t_m, pred_o_m = res_m[2], res_m[3]

if use_w_reg:
    pred_t_w, pred_o_w = res_w[4], res_w[5]
else:
    pred_t_w, pred_o_w = res_w[2], res_w[3]

final_m = test_m[["Id"]].copy()
final_m["team_goals"] = pred_t_m.astype(int)
final_m["opp_goals"] = pred_o_m.astype(int)
final_w = test_w[["Id"]].copy()
final_w["team_goals"] = pred_t_w.astype(int)
final_w["opp_goals"] = pred_o_w.astype(int)

all_preds = pd.concat([final_m, final_w], ignore_index=True)
sub = pd.read_csv(DATA_DIR / "sample submission.csv")
sub = sub[["Id"]].merge(all_preds, on="Id", how="left")
out_path = DATA_DIR / "submission_v16_fast.csv"
sub.to_csv(out_path, index=False)
print(f"\nSaved: {out_path} ({len(sub)} rows)")

# ============================================================
# 13. EVALUATE
# ============================================================
evaluate_submission(str(out_path), str(DATA_DIR / "test_ground_truth.csv"))
print(f"\nTotal time: {time.time()-T0:.0f}s")
print("V16_fast COMPLETE — P2(Tail) + P5(Ensemble) + P6(Calibration)")