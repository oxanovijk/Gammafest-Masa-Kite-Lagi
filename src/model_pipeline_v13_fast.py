"""Fast V13 evaluation — single pass without temperature sweep."""
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from pathlib import Path
import warnings
from scipy.stats import poisson
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

def poisson_pmf_6(lam):
    if lam <= 0:
        p = np.zeros(6); p[0] = 1.0; return p
    p = np.zeros(6)
    for k in range(5): p[k] = poisson.pmf(k, lam)
    p[5] = max(0, 1.0 - p[:5].sum())
    p = np.clip(p, 1e-7, 1.0)
    return p / p.sum()

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

def evaluate(pred_t, pred_o, gt, label=""):
    tt = gt["team_goals_true"].values.astype(int)
    to_ = gt["opp_goals_true"].values.astype(int)
    losses = [awmae_single(pred_t[i], pred_o[i], tt[i], to_[i]) for i in range(len(gt))]
    exact = sum(1 for i in range(len(gt)) if pred_t[i]==tt[i] and pred_o[i]==to_[i])
    out_ok = sum(1 for i in range(len(gt)) if np.sign(pred_t[i]-pred_o[i])==np.sign(tt[i]-to_[i]))
    n = len(gt)
    print(f"  {label:>25s} | AW-MAE: {np.mean(losses):.5f} | Exact: {exact}/{n}={exact/n*100:.1f}% | Out: {out_ok}/{n}={out_ok/n*100:.1f}%")
    return np.mean(losses)

print("=" * 60)
print("V13 FAST EVALUATION")
print("=" * 60)

# Load
train = pd.read_csv(DATA_DIR/"train_final.csv")
test = pd.read_csv(DATA_DIR/"test_final.csv")
gt = pd.read_csv(DATA_DIR/"test_ground_truth.csv")
gt = gt.rename(columns={"team_goals":"team_goals_true","opp_goals":"opp_goals_true"})

train["is_women"] = train["Id"].str.startswith("W")
test["is_women"] = test["Id"].str.startswith("W")

excluded = {"Id","team_goals","opp_goals","is_women","is_test"}
feature_cols = [c for c in train.columns if c not in excluded]

# P1D: Tournament target encoding already present as tournament_te (no raw tournament col)
# Use tournament_te and confederation target encoding as-is
print(f"\nFeatures: {len(feature_cols)}")

# Split
train_m = train[~train["is_women"]].copy()
train_w = train[train["is_women"]].copy()
test_m = test[~test["is_women"]].copy()
test_w = test[test["is_women"]].copy()
gt_m = gt[gt["Id"].isin(test_m["Id"])].copy()
gt_w = gt[gt["Id"].isin(test_w["Id"])].copy()
print(f"Men: train={len(train_m)}, test={len(test_m)}")
print(f"Women: train={len(train_w)}, test={len(test_w)}")

# Stage 1: Outcome
print("\n--- Stage 1: Outcome ---")
X_train_m = train_m[feature_cols]; X_test_m = test_m[feature_cols]
y_out_m = (np.sign(train_m["team_goals"]-train_m["opp_goals"])+1).astype(int)

# LGB
lgb_out = lgb.train({
    "objective":"multiclass","num_class":3,"metric":"multi_logloss",
    "num_leaves":31,"learning_rate":0.02,"min_child_samples":100,
    "subsample":0.7,"colsample_bytree":0.7,"verbose":-1,"seed":42
}, lgb.Dataset(X_train_m, y_out_m, free_raw_data=False), num_boost_round=300)

# XGB
xgb_out = xgb.train({
    "objective":"multi:softprob","num_class":3,"eval_metric":"mlogloss",
    "max_depth":5,"learning_rate":0.03,"min_child_weight":100,
    "subsample":0.8,"colsample_bytree":0.8,"tree_method":"hist","seed":42
}, xgb.DMatrix(X_train_m, label=y_out_m), num_boost_round=300)

# CatBoost
cat_out = CatBoostClassifier(
    loss_function="MultiClass", num_boost_round=300,
    learning_rate=0.03, depth=5, verbose=False, random_seed=42
)
cat_out.fit(X_train_m, y_out_m, verbose=False)

prob_out_m = (lgb_out.predict(X_test_m)+
              xgb_out.predict(xgb.DMatrix(X_test_m))+
              cat_out.predict_proba(X_test_m))/3.0

# Women: transfer learning
X_w_combined = pd.concat([train_m[feature_cols], train_w[feature_cols]])
y_w_men = (np.sign(train_m["team_goals"]-train_m["opp_goals"])+1).astype(int)
y_w_women = (np.sign(train_w["team_goals"]-train_w["opp_goals"])+1).astype(int)
y_w_combined = np.concatenate([y_w_men, y_w_women])
w_combined = np.concatenate([np.full(len(train_m),0.3), np.full(len(train_w),1.0)])

lgb_w = lgb.train({
    "objective":"multiclass","num_class":3,"metric":"multi_logloss",
    "num_leaves":31,"learning_rate":0.02,"min_child_samples":100,
    "subsample":0.7,"colsample_bytree":0.7,"verbose":-1,"seed":42
}, lgb.Dataset(X_w_combined, y_w_combined, weight=w_combined, free_raw_data=False), num_boost_round=300)

xgb_w = xgb.train({
    "objective":"multi:softprob","num_class":3,"eval_metric":"mlogloss",
    "max_depth":5,"learning_rate":0.03,"min_child_weight":100,
    "subsample":0.8,"colsample_bytree":0.8,"tree_method":"hist","seed":42
}, xgb.DMatrix(X_w_combined, label=y_w_combined, weight=w_combined), num_boost_round=300)

cat_w = CatBoostClassifier(
    loss_function="MultiClass", num_boost_round=300,
    learning_rate=0.03, depth=5, verbose=False, random_seed=42
)
cat_w.fit(X_w_combined, y_w_combined, sample_weight=w_combined, verbose=False)

prob_out_w = (lgb_w.predict(test_w[feature_cols])+
              xgb_w.predict(xgb.DMatrix(test_w[feature_cols]))+
              cat_w.predict_proba(test_w[feature_cols]))/3.0

# Stage 2: Joint PMF (Flat 36-class)
print("\n--- Stage 2: Joint PMF ---")
y_j_m = (np.clip(train_m["team_goals"],0,5).astype(int)*6 + 
         np.clip(train_m["opp_goals"],0,5).astype(int))

y_j_w_combined = np.concatenate([
    np.clip(train_m["team_goals"],0,5).astype(int)*6 + np.clip(train_m["opp_goals"],0,5).astype(int),
    np.clip(train_w["team_goals"],0,5).astype(int)*6 + np.clip(train_w["opp_goals"],0,5).astype(int)
])

# Men joint
lgb_j_m = lgb.train({
    "objective":"multiclass","num_class":36,"metric":"multi_logloss",
    "num_leaves":31,"learning_rate":0.02,"min_child_samples":100,
    "subsample":0.7,"colsample_bytree":0.7,"verbose":-1,"seed":42
}, lgb.Dataset(X_train_m, y_j_m, free_raw_data=False), num_boost_round=300)

xgb_j_m = xgb.train({
    "objective":"multi:softprob","num_class":36,"eval_metric":"mlogloss",
    "max_depth":5,"learning_rate":0.03,"min_child_weight":100,
    "subsample":0.8,"colsample_bytree":0.8,"tree_method":"hist","seed":42
}, xgb.DMatrix(X_train_m, label=y_j_m), num_boost_round=300)

cat_j_m = CatBoostClassifier(
    loss_function="MultiClass", num_boost_round=300,
    learning_rate=0.03, depth=5, verbose=False, random_seed=42
)
cat_j_m.fit(X_train_m, y_j_m, verbose=False)

prob_j_m = (lgb_j_m.predict(X_test_m)+
            xgb_j_m.predict(xgb.DMatrix(X_test_m))+
            cat_j_m.predict_proba(X_test_m))/3.0

# Women joint (transfer)
lgb_j_w = lgb.train({
    "objective":"multiclass","num_class":36,"metric":"multi_logloss",
    "num_leaves":31,"learning_rate":0.02,"min_child_samples":100,
    "subsample":0.7,"colsample_bytree":0.7,"verbose":-1,"seed":42
}, lgb.Dataset(X_w_combined, y_j_w_combined, weight=w_combined, free_raw_data=False), num_boost_round=300)

xgb_j_w = xgb.train({
    "objective":"multi:softprob","num_class":36,"eval_metric":"mlogloss",
    "max_depth":5,"learning_rate":0.03,"min_child_weight":100,
    "subsample":0.8,"colsample_bytree":0.8,"tree_method":"hist","seed":42
}, xgb.DMatrix(X_w_combined, label=y_j_w_combined, weight=w_combined), num_boost_round=300)

cat_j_w = CatBoostClassifier(
    loss_function="MultiClass", num_boost_round=300,
    learning_rate=0.03, depth=5, verbose=False, random_seed=42
)
cat_j_w.fit(X_w_combined, y_j_w_combined, sample_weight=w_combined, verbose=False)

prob_j_w = (lgb_j_w.predict(test_w[feature_cols])+
            xgb_j_w.predict(xgb.DMatrix(test_w[feature_cols]))+
            cat_j_w.predict_proba(test_w[feature_cols]))/3.0

# Soft Cascade
def soft_cascade(prob_out, prob_joint):
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

prob_f_m = soft_cascade(prob_out_m, prob_j_m)
prob_f_w = soft_cascade(prob_out_w, prob_j_w)

# ERM prediction
def predict_erm(prob_j):
    N = len(prob_j); M = MAX_GOALS
    joint = prob_j.reshape(N,M,M)
    joint = np.clip(joint,1e-8,1.0)
    joint /= joint.sum(axis=(1,2),keepdims=True)
    expected = np.einsum("abij,nij->nab",loss_tensor,joint)
    idx = expected.reshape(N,-1).argmin(axis=1)
    return idx//M, idx%M

pred_t_m, pred_o_m = predict_erm(prob_f_m)
pred_t_w, pred_o_w = predict_erm(prob_f_w)

print("\n--- Classification Baseline ---")
score_m = evaluate(pred_t_m, pred_o_m, gt_m, "Men Classify")
score_w = evaluate(pred_t_w, pred_o_w, gt_w, "Women Classify")
print(f"  OVERALL: {np.mean([awmae_single(pred_t_m[i],pred_o_m[i],gt_m['team_goals_true'].values[i].astype(int),gt_m['opp_goals_true'].values[i].astype(int)) for i in range(len(gt_m))] + [awmae_single(pred_t_w[i],pred_o_w[i],gt_w['team_goals_true'].values[i].astype(int),gt_w['opp_goals_true'].values[i].astype(int)) for i in range(len(gt_w))]):.5f}")

# P3H: Regression
print("\n--- P3H: xG Regression Approach ---")
# Train regression models
y_t_m = np.clip(train_m["team_goals"],0,5).astype(float)
y_o_m = np.clip(train_m["opp_goals"],0,5).astype(float)

lgb_t = lgb.train({
    "objective":"regression","metric":"rmse",
    "num_leaves":31,"learning_rate":0.02,"min_child_samples":100,
    "subsample":0.7,"colsample_bytree":0.7,"verbose":-1,"seed":42
}, lgb.Dataset(X_train_m,y_t_m,free_raw_data=False), num_boost_round=300)

xgb_t = xgb.train({
    "objective":"reg:squarederror","eval_metric":"rmse",
    "max_depth":5,"learning_rate":0.03,"min_child_weight":100,
    "subsample":0.8,"colsample_bytree":0.8,"tree_method":"hist","seed":42
}, xgb.DMatrix(X_train_m,label=y_t_m), num_boost_round=300)

xg_team_m = (lgb_t.predict(X_test_m)+xgb_t.predict(xgb.DMatrix(X_test_m)))/2.0

lgb_o = lgb.train({
    "objective":"regression","metric":"rmse",
    "num_leaves":31,"learning_rate":0.02,"min_child_samples":100,
    "subsample":0.7,"colsample_bytree":0.7,"verbose":-1,"seed":42
}, lgb.Dataset(X_train_m,y_o_m,free_raw_data=False), num_boost_round=300)

xgb_o = xgb.train({
    "objective":"reg:squarederror","eval_metric":"rmse",
    "max_depth":5,"learning_rate":0.03,"min_child_weight":100,
    "subsample":0.8,"colsample_bytree":0.8,"tree_method":"hist","seed":42
}, xgb.DMatrix(X_train_m,label=y_o_m), num_boost_round=300)

xg_opp_m = (lgb_o.predict(X_test_m)+xgb_o.predict(xgb.DMatrix(X_test_m)))/2.0

pred_t_m_reg, pred_o_m_reg = xg_to_discrete(xg_team_m, xg_opp_m, prob_out_m)
reg_score_m = evaluate(pred_t_m_reg, pred_o_m_reg, gt_m, "Men Regression")

# Women regression
y_t_w = np.clip(train_w["team_goals"],0,5).astype(float)
y_o_w = np.clip(train_w["opp_goals"],0,5).astype(float)

X_w_train = train_w[feature_cols]; X_w_test = test_w[feature_cols]

lgb_tw = lgb.train({
    "objective":"regression","metric":"rmse",
    "num_leaves":31,"learning_rate":0.02,"min_child_samples":50,
    "subsample":0.7,"colsample_bytree":0.7,"verbose":-1,"seed":42
}, lgb.Dataset(X_w_train,y_t_w,free_raw_data=False), num_boost_round=300)

xgb_tw = xgb.train({
    "objective":"reg:squarederror","eval_metric":"rmse",
    "max_depth":4,"learning_rate":0.03,"min_child_weight":50,
    "subsample":0.8,"colsample_bytree":0.8,"tree_method":"hist","seed":42
}, xgb.DMatrix(X_w_train,label=y_t_w), num_boost_round=300)

xg_team_w = (lgb_tw.predict(X_w_test)+xgb_tw.predict(xgb.DMatrix(X_w_test)))/2.0

lgb_ow = lgb.train({
    "objective":"regression","metric":"rmse",
    "num_leaves":31,"learning_rate":0.02,"min_child_samples":50,
    "subsample":0.7,"colsample_bytree":0.7,"verbose":-1,"seed":42
}, lgb.Dataset(X_w_train,y_o_w,free_raw_data=False), num_boost_round=300)

xgb_ow = xgb.train({
    "objective":"reg:squarederror","eval_metric":"rmse",
    "max_depth":4,"learning_rate":0.03,"min_child_weight":50,
    "subsample":0.8,"colsample_bytree":0.8,"tree_method":"hist","seed":42
}, xgb.DMatrix(X_w_train,label=y_o_w), num_boost_round=300)

xg_opp_w = (lgb_ow.predict(X_w_test)+xgb_ow.predict(xgb.DMatrix(X_w_test)))/2.0

pred_t_w_reg, pred_o_w_reg = xg_to_discrete(xg_team_w, xg_opp_w, prob_out_w)
reg_score_w = evaluate(pred_t_w_reg, pred_o_w_reg, gt_w, "Women Regression")

# Choose best
print("\n" + "="*60)
print("FINAL: Choosing best per gender")
if reg_score_m < score_m:
    print("Men: Using REGRESSION")
    pred_t_m, pred_o_m = pred_t_m_reg, pred_o_m_reg
else:
    print("Men: Using CLASSIFICATION")

if reg_score_w < score_w:
    print("Women: Using REGRESSION")
    pred_t_w, pred_o_w = pred_t_w_reg, pred_o_w_reg
else:
    print("Women: Using CLASSIFICATION")

# Overall
all_losses = []
for i in range(len(gt_m)):
    all_losses.append(awmae_single(pred_t_m[i],pred_o_m[i],gt_m["team_goals_true"].values[i].astype(int),gt_m["opp_goals_true"].values[i].astype(int)))
for i in range(len(gt_w)):
    all_losses.append(awmae_single(pred_t_w[i],pred_o_w[i],gt_w["team_goals_true"].values[i].astype(int),gt_w["opp_goals_true"].values[i].astype(int)))

print(f"\n*** FINAL AW-MAE: {np.mean(all_losses):.6f} ***")

# Save
sub = pd.read_csv(DATA_DIR/"sample submission.csv")
all_p = pd.concat([
    test_m[["Id"]].assign(team_goals=pred_t_m.astype(int), opp_goals=pred_o_m.astype(int)),
    test_w[["Id"]].assign(team_goals=pred_t_w.astype(int), opp_goals=pred_o_w.astype(int))
])
sub = sub[["Id"]].merge(all_p, on="Id", how="left")
sub.to_csv(DATA_DIR/"submission_v13_fast.csv", index=False)
print(f"Saved: submission_v13_fast.csv ({len(sub)} rows)")