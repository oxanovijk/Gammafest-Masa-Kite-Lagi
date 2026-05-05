"""Quick evaluation of V21 and V22 submission CSVs."""
import pandas as pd
import numpy as np

def awmae_single(pred_t, pred_o, true_t, true_o):
    mae = (abs(int(pred_t)-int(true_t)) + abs(int(pred_o)-int(true_o))) / 2.0
    exact = 1 if (int(pred_t)==int(true_t) and int(pred_o)==int(true_o)) else 0
    pred_out = np.sign(int(pred_t)-int(pred_o))
    true_out = np.sign(int(true_t)-int(true_o))
    out_ok = 1 if pred_out == true_out else 0
    gd_ok = 1 if (int(pred_t)-int(pred_o)) == (int(true_t)-int(true_o)) else 0
    aug = mae + 0.30*(1-exact) + 0.25*(1-out_ok) + 0.15*(1-gd_ok)
    mult = 1.0 if out_ok else 1.5
    return (aug * mult) ** 1.3

gt = pd.read_csv("dataset/test_ground_truth.csv")
gt = gt.rename(columns={"team_goals":"team_goals_true","opp_goals":"opp_goals_true"})

for v in [20, 21, 22]:
    try:
        sub = pd.read_csv(f"dataset/submission_v{v}.csv")
        df = pd.merge(sub, gt, on="Id", suffixes=("","_true"))
        if len(df) == 0:
            print(f"V{v}: EMPTY MERGE - skipping")
            continue
        df["loss"] = df.apply(lambda r: awmae_single(
            r['team_goals'], r['opp_goals'],
            r['team_goals_true'], r['opp_goals_true']), axis=1)
        
        exact = (df["team_goals"]==df["team_goals_true"]) & (df["opp_goals"]==df["opp_goals_true"])
        out_ok = np.sign(df["team_goals"]-df["opp_goals"]) == np.sign(df["team_goals_true"]-df["opp_goals_true"])
        
        print(f"V{v}: AW-MAE={df['loss'].mean():.5f}  "
              f"Exact={exact.sum()}/{len(df)} ({exact.mean()*100:.1f}%)  "
              f"Outcome={out_ok.sum()}/{len(df)} ({out_ok.mean()*100:.1f}%)")
              
        # Per gender
        for g, gname in [(False,"Men"), (True,"Women")]:
            dg = df[df["Id"].str.startswith("W") == g]
            if len(dg) > 0:
                ex = ((dg["team_goals"]==dg["team_goals_true"]) & (dg["opp_goals"]==dg["opp_goals_true"])).mean()
                out = (np.sign(dg["team_goals"]-dg["opp_goals"]) == np.sign(dg["team_goals_true"]-dg["opp_goals_true"])).mean()
                print(f"  {gname}: AW-MAE={dg['loss'].mean():.5f}, Exact={ex*100:.1f}%, Outcome={out*100:.1f}%")
    except Exception as e:
        print(f"V{v}: ERROR - {e}")

# Also check V14 baseline
try:
    sub14 = pd.read_csv("dataset/submission_v14.csv")
    df14 = pd.merge(sub14, gt, on="Id", suffixes=("","_true"))
    df14["loss"] = df14.apply(lambda r: awmae_single(
        r['team_goals'], r['opp_goals'],
        r['team_goals_true'], r['opp_goals_true']), axis=1)
    exact14 = (df14["team_goals"]==df14["team_goals_true"]) & (df14["opp_goals"]==df14["opp_goals_true"])
    out14 = np.sign(df14["team_goals"]-df14["opp_goals"]) == np.sign(df14["team_goals_true"]-df14["opp_goals_true"])
    print(f"\nV14 (baseline): AW-MAE={df14['loss'].mean():.5f}  "
          f"Exact={exact14.sum()}/{len(df14)} ({exact14.mean()*100:.1f}%)  "
          f"Outcome={out14.sum()}/{len(df14)} ({out14.mean()*100:.1f}%)")
except Exception as e:
    print(f"V14: ERROR - {e}")