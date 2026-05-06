"""
V29 — V12 + Custom Loss Tensor (0.05) + Tier-Specific Temperature
===================================================================
Approach A+B dari analisa_komprehensif:
  A: Penalty kecil 0.05 pada 2-1/1-2 (sweet spot)
  B: Temperature berbeda per tier (draw-heavy tier → lebih tajam)
"""
import pandas as pd, numpy as np, lightgbm as lgb, xgboost as xgb
from pathlib import Path
import warnings, time
warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent / "dataset"
M = 6; NC = M*M; NLS = 1.3; N_EST = 600

LGB_O = {"objective":"multiclass","num_class":3,"metric":"multi_logloss","num_leaves":31,"learning_rate":0.02,"min_child_samples":100,"subsample":0.8,"colsample_bytree":0.8,"verbose":-1,"seed":42}
XGB_O = {"objective":"multi:softprob","num_class":3,"eval_metric":"mlogloss","max_depth":5,"learning_rate":0.03,"min_child_weight":100,"subsample":0.8,"colsample_bytree":0.8,"tree_method":"hist","seed":42}
LGB_J = {"objective":"multiclass","num_class":NC,"metric":"multi_logloss","num_leaves":31,"learning_rate":0.02,"min_child_samples":150,"subsample":0.7,"colsample_bytree":0.7,"verbose":-1,"seed":42}
XGB_J = {"objective":"multi:softprob","num_class":NC,"eval_metric":"mlogloss","max_depth":5,"learning_rate":0.03,"min_child_weight":150,"subsample":0.75,"colsample_bytree":0.75,"tree_method":"hist","seed":42}

PENALTY = 0.05
PENALIZED = [(2,1),(1,2)]
# Tier-specific temperature: draw-heavy tiers get sharper (lower T)
TIER_TEMP = {1: 1.05, 2: 1.15, 3: 1.10, 4: 1.00, 5: 1.10}

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
lt_mod=lt.copy()
for(a,b)in PENALIZED: lt_mod[a,b,:,:]+=PENALTY

def soft_cascade(po,pj,T):
    N=len(po)
    pj=np.clip(pj,1e-7,1.0)
    lp=np.log(pj)/T; ep=np.exp(lp); pj=ep/ep.sum(axis=1,keepdims=True)
    pf=np.zeros_like(pj); sj=np.zeros((N,3))
    for t in range(M):
        for o in range(M):
            c=t*M+o; oi=int(np.sign(t-o))+1; sj[:,oi]+=pj[:,c]
    for t in range(M):
        for o in range(M):
            c=t*M+o; oi=int(np.sign(t-o))+1
            pf[:,c]=(pj[:,c]/np.maximum(sj[:,oi],1e-9))*po[:,oi]
    return pf

def erm(pf):
    N=len(pf); j=pf.reshape(N,M,M)
    j=np.clip(j,1e-8,1.0); j/=j.sum(axis=(1,2),keepdims=True)
    e=np.einsum("abij,nij->nab",lt_mod,j)
    idx=e.reshape(N,-1).argmin(axis=1)
    return idx//M, idx%M

def train_predict(tr,te,fc,tiers):
    X_tr,X_te=tr[fc].values,te[fc].values
    yt=np.clip(tr["team_goals"].values,0,M-1).astype(int)
    yo=np.clip(tr["opp_goals"].values,0,M-1).astype(int)
    y_out=(np.sign(yt-yo)+1).astype(int); y_jt=yt*M+yo
    
    t0=time.time()
    lgb_o=lgb.train(LGB_O,lgb.Dataset(X_tr,y_out,free_raw_data=False),num_boost_round=N_EST)
    xgb_o=xgb.train(XGB_O,xgb.DMatrix(X_tr,label=y_out),num_boost_round=N_EST)
    po=(lgb_o.predict(X_te)+xgb_o.predict(xgb.DMatrix(X_te)))/2.0
    
    lgb_j=lgb.train(LGB_J,lgb.Dataset(X_tr,y_jt,free_raw_data=False),num_boost_round=N_EST)
    xgb_j=xgb.train(XGB_J,xgb.DMatrix(X_tr,label=y_jt),num_boost_round=N_EST)
    pj=(lgb_j.predict(X_te)+xgb_j.predict(xgb.DMatrix(X_te)))/2.0
    print(f"  Training done in {time.time()-t0:.1f}s")
    
    # Tier-specific temperature cascade + ERM
    pred_t=np.zeros(len(te),dtype=int); pred_o=np.zeros(len(te),dtype=int)
    for tier in sorted(tiers.unique()):
        mask=tiers.values==tier; T=TIER_TEMP.get(int(tier),1.1)
        if mask.sum()==0: continue
        pf=soft_cascade(po[mask],pj[mask],T)
        pt,pp=erm(pf)
        pred_t[mask]=pt; pred_o[mask]=pp
        print(f"  Tier {int(tier)}: N={mask.sum()}, T={T}")
    return pred_t, pred_o

def main():
    print("="*60); print(f"V29 — Loss Tensor(p={PENALTY}) + Tier-Specific T"); print("="*60)
    train=pd.read_csv(BASE/"train_final.csv"); test=pd.read_csv(BASE/"test_final.csv")
    train["is_w"]=train["Id"].str.startswith("W"); test["is_w"]=test["Id"].str.startswith("W")
    exc={"Id","team_goals","opp_goals","is_w","is_test"}
    fc=[c for c in train.columns if c not in exc]
    
    trm,trw=train[~train["is_w"]],train[train["is_w"]]
    tem,tew=test[~test["is_w"]],test[test["is_w"]]
    print(f"Features: {len(fc)}, Men: {len(trm)}->{len(tem)}, Women: {len(trw)}->{len(tew)}")
    
    print("\n--- MEN ---")
    ptm,pom=train_predict(trm,tem,fc,tem["tournament_tier"])
    print("\n--- WOMEN ---")
    ptw,pow_=train_predict(trw,tew,fc,tew["tournament_tier"])
    
    r=pd.concat([tem[["Id"]].assign(team_goals=ptm,opp_goals=pom),tew[["Id"]].assign(team_goals=ptw,opp_goals=pow_)])
    
    gt_p=BASE/"test_ground_truth.csv"
    if gt_p.exists():
        gt=pd.read_csv(gt_p).rename(columns={"team_goals":"gt","opp_goals":"go"})
        df=r.merge(gt,on="Id")
        df["loss"]=df.apply(lambda r:awmae(r["team_goals"],r["opp_goals"],r["gt"],r["go"]),axis=1)
        ex=(df["team_goals"]==df["gt"])&(df["opp_goals"]==df["go"])
        ok=np.sign(df["team_goals"]-df["opp_goals"])==np.sign(df["gt"]-df["go"])
        print("\n"+"="*60)
        print(f"RESULTS V29"); print("="*60)
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
    sub.to_csv(BASE/"submission_v29.csv",index=False)
    print(f"\nSaved: submission_v29.csv ({len(sub)} rows)")

if __name__=="__main__": main()
