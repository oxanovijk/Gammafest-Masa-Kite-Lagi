"""
V30 — Decoupled Outcome-Score Architecture
============================================
Approach F dari analisa_komprehensif:
  1. Train outcome model (3-class W/D/L) — LGB+XGB
  2. Train 3 SEPARATE score models, each on filtered data:
     - Draw model: only draw matches → 6 classes {0-0,1-1,2-2,3-3,4-4,5-5}
     - Win model: only win matches → 15 classes
     - Loss model: only loss matches → 15 classes
  3. Inference: outcome model picks W/D/L → corresponding score model picks score
  
  KEY DIFFERENCE from V27 Hard Cascade:
  - V27 used the SAME joint model masked at inference
  - V30 trains DEDICATED models per outcome bucket at TRAINING time
"""
import pandas as pd, numpy as np, lightgbm as lgb, xgboost as xgb
from pathlib import Path
import warnings, time
warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent / "dataset"
M = 6; NLS = 1.3; N_EST = 600

LGB_O = {"objective":"multiclass","num_class":3,"metric":"multi_logloss","num_leaves":31,"learning_rate":0.02,"min_child_samples":100,"subsample":0.8,"colsample_bytree":0.8,"verbose":-1,"seed":42}
XGB_O = {"objective":"multi:softprob","num_class":3,"eval_metric":"mlogloss","max_depth":5,"learning_rate":0.03,"min_child_weight":100,"subsample":0.8,"colsample_bytree":0.8,"tree_method":"hist","seed":42}

def awmae(pt,po,tt,to_):
    mae=(abs(int(pt)-int(tt))+abs(int(po)-int(to_)))/2.0
    ex=1 if(int(pt)==int(tt)and int(po)==int(to_))else 0
    ok=1 if np.sign(int(pt)-int(po))==np.sign(int(tt)-int(to_))else 0
    gd=1 if(int(pt)-int(po))==(int(tt)-int(to_))else 0
    aug=mae+0.30*(1-ex)+0.25*(1-ok)+0.15*(1-gd)
    return(aug*(1.0 if ok else 1.5))**NLS

lt=np.zeros((M,M,M,M))
for a in range(M):
    for b in range(M):
        for c in range(M):
            for d in range(M): lt[a,b,c,d]=awmae(a,b,c,d)

# Score maps per outcome bucket
DRAW_SCORES = [(i,i) for i in range(M)]  # 0-0,1-1,...,5-5
WIN_SCORES  = [(t,o) for t in range(M) for o in range(M) if t>o]  # team>opp
LOSS_SCORES = [(t,o) for t in range(M) for o in range(M) if t<o]  # team<opp

def encode_score_in_bucket(team, opp, score_list):
    """Map (team,opp) to class index within a bucket"""
    for i,(t,o) in enumerate(score_list):
        if t==team and o==opp: return i
    return 0  # fallback

def train_score_model(train_df, fc, score_list, label):
    """Train LGB+XGB on a specific outcome bucket"""
    n_cls = len(score_list)
    yt = np.clip(train_df["team_goals"].values, 0, M-1).astype(int)
    yo = np.clip(train_df["opp_goals"].values, 0, M-1).astype(int)
    y = np.array([encode_score_in_bucket(t,o,score_list) for t,o in zip(yt,yo)])
    X = train_df[fc].values
    
    lgb_p = {"objective":"multiclass","num_class":n_cls,"metric":"multi_logloss",
             "num_leaves":31,"learning_rate":0.02,"min_child_samples":max(20,len(train_df)//100),
             "subsample":0.8,"colsample_bytree":0.8,"verbose":-1,"seed":42}
    xgb_p = {"objective":"multi:softprob","num_class":n_cls,"eval_metric":"mlogloss",
             "max_depth":5,"learning_rate":0.03,"min_child_weight":max(20,len(train_df)//100),
             "subsample":0.8,"colsample_bytree":0.8,"tree_method":"hist","seed":42}
    
    lgb_m = lgb.train(lgb_p, lgb.Dataset(X, y, free_raw_data=False), num_boost_round=N_EST)
    xgb_m = xgb.train(xgb_p, xgb.DMatrix(X, label=y), num_boost_round=N_EST)
    print(f"    {label} model: {len(train_df)} samples, {n_cls} classes")
    return lgb_m, xgb_m

def predict_score_erm(lgb_m, xgb_m, X_test, score_list):
    """ERM within a specific outcome bucket using dedicated model probabilities"""
    prob = (lgb_m.predict(X_test) + xgb_m.predict(xgb.DMatrix(X_test))) / 2.0
    prob = np.clip(prob, 1e-8, 1.0)
    prob /= prob.sum(axis=1, keepdims=True)
    
    N = len(X_test)
    n_cls = len(score_list)
    
    # Build mini loss tensor for this bucket
    mini_lt = np.zeros((n_cls, n_cls))
    for i,(pa,pb) in enumerate(score_list):
        for j,(ta,tb) in enumerate(score_list):
            mini_lt[i,j] = lt[pa,pb,ta,tb]
    
    # ERM: expected[i] = sum_j mini_lt[i,j] * prob[n,j]
    expected = prob @ mini_lt.T  # (N, n_cls)
    best_idx = expected.argmin(axis=1)
    
    pred_t = np.array([score_list[i][0] for i in best_idx])
    pred_o = np.array([score_list[i][1] for i in best_idx])
    return pred_t, pred_o

def train_predict(train_df, test_df, fc):
    if len(test_df)==0: return np.array([]),np.array([])
    X_tr = train_df[fc].values; X_te = test_df[fc].values
    yt=np.clip(train_df["team_goals"].values,0,M-1).astype(int)
    yo=np.clip(train_df["opp_goals"].values,0,M-1).astype(int)
    y_out=(np.sign(yt-yo)+1).astype(int)
    
    t0=time.time()
    # Stage 1: Outcome model
    lgb_o=lgb.train(LGB_O,lgb.Dataset(X_tr,y_out,free_raw_data=False),num_boost_round=N_EST)
    xgb_o=xgb.train(XGB_O,xgb.DMatrix(X_tr,label=y_out),num_boost_round=N_EST)
    prob_out=(lgb_o.predict(X_te)+xgb_o.predict(xgb.DMatrix(X_te)))/2.0
    outcome_pred = prob_out.argmax(axis=1)  # 0=Loss, 1=Draw, 2=Win
    print(f"  Stage 1 (Outcome): {time.time()-t0:.1f}s")
    print(f"    Outcome dist: Loss={np.sum(outcome_pred==0)}, Draw={np.sum(outcome_pred==1)}, Win={np.sum(outcome_pred==2)}")
    
    # Stage 2: Score models per bucket
    t0=time.time()
    # Filter training data by outcome
    train_draw = train_df[y_out==1]
    train_win  = train_df[y_out==2]
    train_loss = train_df[y_out==0]
    
    lgb_d, xgb_d = train_score_model(train_draw, fc, DRAW_SCORES, "Draw")
    lgb_w, xgb_w = train_score_model(train_win,  fc, WIN_SCORES,  "Win")
    lgb_l, xgb_l = train_score_model(train_loss, fc, LOSS_SCORES, "Loss")
    print(f"  Stage 2 (Score models): {time.time()-t0:.1f}s")
    
    # Stage 3: For each test sample, use the appropriate score model
    pred_t = np.zeros(len(test_df), dtype=int)
    pred_o = np.zeros(len(test_df), dtype=int)
    
    for out_idx, (lgb_s, xgb_s, score_list, name) in enumerate([
        (lgb_l, xgb_l, LOSS_SCORES, "Loss"),
        (lgb_d, xgb_d, DRAW_SCORES, "Draw"),
        (lgb_w, xgb_w, WIN_SCORES,  "Win")
    ]):
        mask = outcome_pred == out_idx
        if mask.sum() == 0: continue
        X_sub = X_te[mask]
        pt, po = predict_score_erm(lgb_s, xgb_s, X_sub, score_list)
        pred_t[mask] = pt
        pred_o[mask] = po
        print(f"    {name} bucket: {mask.sum()} matches")
    
    return pred_t, pred_o

def main():
    print("="*60); print("V30 — DECOUPLED OUTCOME-SCORE ARCHITECTURE"); print("="*60)
    train=pd.read_csv(BASE/"train_final.csv"); test=pd.read_csv(BASE/"test_final.csv")
    train["is_w"]=train["Id"].str.startswith("W"); test["is_w"]=test["Id"].str.startswith("W")
    exc={"Id","team_goals","opp_goals","is_w","is_test"}
    fc=[c for c in train.columns if c not in exc]
    
    trm,trw=train[~train["is_w"]],train[train["is_w"]]
    tem,tew=test[~test["is_w"]],test[test["is_w"]]
    print(f"Features: {len(fc)}, Men: {len(trm)}->{len(tem)}, Women: {len(trw)}->{len(tew)}")
    
    print("\n--- MEN ---")
    ptm,pom=train_predict(trm,tem,fc)
    print("\n--- WOMEN ---")
    ptw,pow_=train_predict(trw,tew,fc)
    
    r=pd.concat([tem[["Id"]].assign(team_goals=ptm,opp_goals=pom),tew[["Id"]].assign(team_goals=ptw,opp_goals=pow_)])
    
    gt_p=BASE/"test_ground_truth.csv"
    if gt_p.exists():
        gt=pd.read_csv(gt_p).rename(columns={"team_goals":"gt","opp_goals":"go"})
        df=r.merge(gt,on="Id")
        df["loss"]=df.apply(lambda rr:awmae(rr["team_goals"],rr["opp_goals"],rr["gt"],rr["go"]),axis=1)
        ex=(df["team_goals"]==df["gt"])&(df["opp_goals"]==df["go"])
        ok=np.sign(df["team_goals"]-df["opp_goals"])==np.sign(df["gt"]-df["go"])
        print("\n"+"="*60)
        print("RESULTS V30"); print("="*60)
        print(f"AW-MAE: {df['loss'].mean():.5f} | Exact: {ex.mean()*100:.2f}% | Outcome: {ok.mean()*100:.2f}%")
        df["is_w"]=df["Id"].str.startswith("W")
        for n,m in[("MEN",~df["is_w"]),("WOMEN",df["is_w"])]:
            print(f"  {n:7s}|AW:{df.loc[m,'loss'].mean():.5f}|Ex:{ex[m].mean()*100:.1f}%|Out:{ok[m].mean()*100:.1f}%|N={m.sum()}")
        df2=df.merge(test[["Id","tournament_tier"]],on="Id",how="left")
        for t in sorted(df2["tournament_tier"].unique(),reverse=True):
            s=df2[df2["tournament_tier"]==t]
            print(f"  T{int(t)}|AW:{s['loss'].mean():.5f}|Ex:{((s['team_goals']==s['gt'])&(s['opp_goals']==s['go'])).mean()*100:.1f}%|Out:{(np.sign(s['team_goals']-s['opp_goals'])==np.sign(s['gt']-s['go'])).mean()*100:.1f}%|N={len(s)}")
        dp=(df["team_goals"]==df["opp_goals"]).sum()
        dg=(df["gt"]==df["go"]).sum()
        print(f"\nDraw pred: {dp}({dp/len(df)*100:.1f}%) vs GT: {dg}({dg/len(df)*100:.1f}%)")
        print("\nTop 10 scores:")
        print(df.groupby(["team_goals","opp_goals"]).size().sort_values(ascending=False).head(10))
    
    sample=pd.read_csv(BASE/"sample submission.csv")
    sub=sample[["Id"]].merge(r,on="Id",how="left")
    sub.to_csv(BASE/"submission_v30.csv",index=False)
    print(f"\nSaved: submission_v30.csv ({len(sub)} rows)")

if __name__=="__main__": main()
