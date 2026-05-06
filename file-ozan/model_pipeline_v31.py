"""
V31 — Conditional Score Override (Heuristic Rules on V12)
==========================================================
Approach J dari analisa_komprehensif:
  Gunakan V12 sebagai base, lalu override skor berdasarkan kondisi:
  1. Jika V12 prediksi draw → selalu 1-1 (sudah dilakukan V12, confirm)
  2. Jika V12 prediksi win & elo_diff < 80 → override ke 1-0 (narrow win)
  3. Jika V12 prediksi loss & elo_diff > -80 → override ke 0-1 (narrow loss)
  4. Jika match di Tier 4/1 & V12 prediksi 2-1 → downgrade ke 1-0 (draw-heavy tiers)
  5. Sisanya: keep V12 prediction
  
  IMPORTANT: ini POST-PROCESSING pada CSV V12, tidak perlu retrain model.
"""
import pandas as pd, numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent / "dataset"
M = 6; NLS = 1.3

def awmae(pt,po,tt,to_):
    mae=(abs(int(pt)-int(tt))+abs(int(po)-int(to_)))/2.0
    ex=1 if(int(pt)==int(tt)and int(po)==int(to_))else 0
    ok=1 if np.sign(int(pt)-int(po))==np.sign(int(tt)-int(to_))else 0
    gd=1 if(int(pt)-int(po))==(int(tt)-int(to_))else 0
    aug=mae+0.30*(1-ex)+0.25*(1-ok)+0.15*(1-gd)
    return(aug*(1.0 if ok else 1.5))**NLS

def main():
    print("="*60)
    print("V31 — CONDITIONAL SCORE OVERRIDE ON V12")
    print("="*60)
    
    v12 = pd.read_csv(BASE / "submission_v12.csv")
    test = pd.read_csv(BASE / "test_final.csv")
    
    # Merge to get features for override decisions
    df = v12.merge(test[["Id", "elo_diff", "tournament_tier"]], on="Id", how="left")
    df["is_w"] = df["Id"].str.startswith("W")
    
    original_t = df["team_goals"].values.copy()
    original_o = df["opp_goals"].values.copy()
    
    override_count = {"narrow_win": 0, "narrow_loss": 0, "tier_downgrade_w": 0, "tier_downgrade_l": 0, "total": 0}
    
    for i in range(len(df)):
        t, o = df.iloc[i]["team_goals"], df.iloc[i]["opp_goals"]
        elo = df.iloc[i]["elo_diff"]
        tier = int(df.iloc[i]["tournament_tier"])
        is_w = df.iloc[i]["is_w"]
        
        outcome = np.sign(t - o)
        
        # Rule 2: Win with small elo diff → narrow win (1-0)
        if outcome == 1 and t == 2 and o == 1:
            if abs(elo) < 80:
                df.at[df.index[i], "team_goals"] = 1
                df.at[df.index[i], "opp_goals"] = 0
                override_count["narrow_win"] += 1
                override_count["total"] += 1
                continue
        
        # Rule 3: Loss with small elo diff → narrow loss (0-1)
        if outcome == -1 and t == 1 and o == 2:
            if abs(elo) < 80:
                df.at[df.index[i], "team_goals"] = 0
                df.at[df.index[i], "opp_goals"] = 1
                override_count["narrow_loss"] += 1
                override_count["total"] += 1
                continue
        
        # Rule 4: Draw-heavy tiers, 2-1 → 1-0
        if tier in (1, 4) and not is_w:
            if t == 2 and o == 1:
                df.at[df.index[i], "team_goals"] = 1
                df.at[df.index[i], "opp_goals"] = 0
                override_count["tier_downgrade_w"] += 1
                override_count["total"] += 1
                continue
            if t == 1 and o == 2:
                df.at[df.index[i], "team_goals"] = 0
                df.at[df.index[i], "opp_goals"] = 1
                override_count["tier_downgrade_l"] += 1
                override_count["total"] += 1
                continue
    
    print(f"\nOverride summary:")
    for k, v in override_count.items():
        print(f"  {k}: {v}")
    
    # Evaluate
    gt_p = BASE / "test_ground_truth.csv"
    if gt_p.exists():
        gt = pd.read_csv(gt_p).rename(columns={"team_goals":"gt","opp_goals":"go"})
        ev = df[["Id","team_goals","opp_goals"]].merge(gt, on="Id")
        ev["loss"] = ev.apply(lambda r: awmae(r["team_goals"], r["opp_goals"], r["gt"], r["go"]), axis=1)
        ex = (ev["team_goals"]==ev["gt"]) & (ev["opp_goals"]==ev["go"])
        ok = np.sign(ev["team_goals"]-ev["opp_goals"]) == np.sign(ev["gt"]-ev["go"])
        
        # Also compute V12 baseline for comparison
        ev12 = pd.DataFrame({"Id":df["Id"],"team_goals":original_t,"opp_goals":original_o}).merge(gt, on="Id")
        ev12["loss"] = ev12.apply(lambda r: awmae(r["team_goals"], r["opp_goals"], r["gt"], r["go"]), axis=1)
        ex12 = (ev12["team_goals"]==ev12["gt"]) & (ev12["opp_goals"]==ev12["go"])
        ok12 = np.sign(ev12["team_goals"]-ev12["opp_goals"]) == np.sign(ev12["gt"]-ev12["go"])
        
        print("\n" + "="*60)
        print("RESULTS V31 vs V12 BASELINE")
        print("="*60)
        print(f"V12: AW-MAE={ev12['loss'].mean():.5f} | Exact={ex12.mean()*100:.2f}% | Outcome={ok12.mean()*100:.2f}%")
        print(f"V31: AW-MAE={ev['loss'].mean():.5f} | Exact={ex.mean()*100:.2f}% | Outcome={ok.mean()*100:.2f}%")
        print(f"Delta AW-MAE: {ev['loss'].mean() - ev12['loss'].mean():+.5f}")
        print("-"*60)
        
        ev["is_w"]=ev["Id"].str.startswith("W")
        for n,m in[("MEN",~ev["is_w"]),("WOMEN",ev["is_w"])]:
            print(f"  {n:7s}|AW:{ev.loc[m,'loss'].mean():.5f}|Ex:{ex[m].mean()*100:.1f}%|Out:{ok[m].mean()*100:.1f}%|N={m.sum()}")
        ev2=ev.merge(test[["Id","tournament_tier"]],on="Id",how="left")
        for t in sorted(ev2["tournament_tier"].unique(),reverse=True):
            s=ev2[ev2["tournament_tier"]==t]
            print(f"  T{int(t)}|AW:{s['loss'].mean():.5f}|Ex:{((s['team_goals']==s['gt'])&(s['opp_goals']==s['go'])).mean()*100:.1f}%|Out:{(np.sign(s['team_goals']-s['opp_goals'])==np.sign(s['gt']-s['go'])).mean()*100:.1f}%|N={len(s)}")
        
        dp=(ev["team_goals"]==ev["opp_goals"]).sum()
        dg=(ev["gt"]==ev["go"]).sum()
        print(f"\nDraw pred: {dp}({dp/len(ev)*100:.1f}%) vs GT: {dg}({dg/len(ev)*100:.1f}%)")
        print("\nTop 10 scores:")
        print(ev.groupby(["team_goals","opp_goals"]).size().sort_values(ascending=False).head(10))
    
    final = df[["Id","team_goals","opp_goals"]]
    sample = pd.read_csv(BASE/"sample submission.csv")
    sub = sample[["Id"]].merge(final, on="Id", how="left")
    sub.to_csv(BASE/"submission_v31.csv", index=False)
    print(f"\nSaved: submission_v31.csv ({len(sub)} rows)")

if __name__=="__main__": main()
