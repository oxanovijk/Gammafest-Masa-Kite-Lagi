import pandas as pd
import numpy as np
from pathlib import Path

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

def get_outcome(t, o):
    return np.sign(t - o)

def main():
    print("="*60)
    print("V28: OUTCOME-PRESERVING ENSEMBLE (V12 + V27c)")
    print("="*60)
    
    v12 = pd.read_csv(DATA_DIR / "submission_v12.csv").rename(columns={'team_goals':'t12', 'opp_goals':'o12'})
    v27c = pd.read_csv(DATA_DIR / "submission_v27c.csv").rename(columns={'team_goals':'t27', 'opp_goals':'o27'})
    
    merged = v12.merge(v27c, on='Id')
    
    def decide(row):
        out12 = get_outcome(row['t12'], row['o12'])
        out27 = get_outcome(row['t27'], row['o27'])
        
        # LOGIC:
        # 1. Jika outcome sama -> pakai V27c (lebih berani/exact tinggi)
        if out12 == out27:
            return row['t27'], row['o27'], "V27c (Same Outcome)"
        # 2. Jika outcome beda -> pakai V12 (outcome lebih akurat)
        else:
            return row['t12'], row['o12'], "V12 (Outcome Conflict)"

    results = merged.apply(decide, axis=1, result_type='expand')
    merged['team_goals'] = results[0].astype(int)
    merged['opp_goals'] = results[1].astype(int)
    merged['source'] = results[2]
    
    print(merged['source'].value_counts())
    
    # Validation
    gt_path = DATA_DIR / "test_ground_truth.csv"
    if gt_path.exists():
        gt = pd.read_csv(gt_path).rename(columns={'team_goals':'gt', 'opp_goals':'go'})
        df = merged.merge(gt, on='Id')
        
        df['loss'] = df.apply(lambda r: awmae_single(r['team_goals'], r['opp_goals'], r['gt'], r['go']), axis=1)
        exact = (df['team_goals']==df['gt']) & (df['opp_goals']==df['go'])
        out_ok = np.sign(df['team_goals']-df['opp_goals']) == np.sign(df['gt']-df['go'])
        
        print("\n" + "="*60)
        print("RESULTS: V28 ENSEMBLE")
        print("="*60)
        print(f"Global AW-MAE:          {df['loss'].mean():.5f}")
        print(f"Global Exact Score:     {exact.mean()*100:.2f}%")
        print(f"Global Outcome Correct: {out_ok.mean()*100:.2f}%")
        print("-" * 60)
        
        # Breakdown by source
        for src in merged['source'].unique():
            sub = df[df['source'] == src]
            print(f"Source: {src}")
            print(f"  N: {len(sub)}")
            print(f"  AW-MAE: {sub['loss'].mean():.5f}")
            print(f"  Exact: {((sub['team_goals']==sub['gt'])&(sub['opp_goals']==sub['go'])).mean()*100:.1f}%")
        
    final_sub = merged[['Id', 'team_goals', 'opp_goals']]
    final_sub.to_csv(DATA_DIR / "submission_v28.csv", index=False)
    print(f"\nSaved: dataset/submission_v28.csv")

if __name__ == "__main__":
    main()
