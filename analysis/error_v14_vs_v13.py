"""Error analysis: V14 vs V13_lite — breakdown by gender, outcome, goals."""
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

# Load data
gt = pd.read_csv(DATA_DIR / "test_ground_truth.csv")
gt.columns = ["Id", "team_goals_true", "opp_goals_true"]

sub_v13 = pd.read_csv(DATA_DIR / "submission_v13_lite.csv")
sub_v13.columns = ["Id", "team_goals_v13", "opp_goals_v13"]

sub_v14 = pd.read_csv(DATA_DIR / "submission_v14.csv")
sub_v14.columns = ["Id", "team_goals_v14", "opp_goals_v14"]

# Merge
df = gt.merge(sub_v13, on="Id").merge(sub_v14, on="Id")
df["is_women"] = df["Id"].str.startswith("W")
df["true_outcome"] = np.sign(df["team_goals_true"] - df["opp_goals_true"])
df["v13_outcome"] = np.sign(df["team_goals_v13"] - df["opp_goals_v13"])
df["v14_outcome"] = np.sign(df["team_goals_v14"] - df["opp_goals_v14"])

# Compute losses
df["loss_v13"] = df.apply(lambda r: awmae_single(
    r.team_goals_v13, r.opp_goals_v13,
    r.team_goals_true, r.opp_goals_true), axis=1)
df["loss_v14"] = df.apply(lambda r: awmae_single(
    r.team_goals_v14, r.opp_goals_v14,
    r.team_goals_true, r.opp_goals_true), axis=1)

df["delta"] = df["loss_v14"] - df["loss_v13"]  # positive = V14 worse
df["v13_exact"] = (df["team_goals_v13"] == df["team_goals_true"]) & (df["opp_goals_v13"] == df["opp_goals_true"])
df["v14_exact"] = (df["team_goals_v14"] == df["team_goals_true"]) & (df["opp_goals_v14"] == df["opp_goals_true"])

print("=" * 70)
print("ERROR ANALYSIS: V14 vs V13_lite")
print("=" * 70)

# Overall
print(f"\n{'Metric':<40} {'V13_lite':>12} {'V14':>12} {'Delta':>12}")
print("-" * 78)
print(f"{'AW-MAE (overall)':<40} {df['loss_v13'].mean():>12.5f} {df['loss_v14'].mean():>12.5f} {df['delta'].mean():>+12.5f}")

# By gender
for gender, label in [(False, "Men"), (True, "Women")]:
    d = df[df["is_women"] == gender]
    print(f"\n--- {label} ({len(d)} matches) ---")
    print(f"{'AW-MAE':<40} {d['loss_v13'].mean():>12.5f} {d['loss_v14'].mean():>12.5f} {d['delta'].mean():>+12.5f}")
    print(f"{'Exact Match Rate':<40} {d['v13_exact'].mean():>11.1%} {d['v14_exact'].mean():>11.1%} {d['v14_exact'].mean()-d['v13_exact'].mean():>+12.1%}")
    print(f"{'Outcome Correct Rate':<40} {(d['v13_outcome']==d['true_outcome']).mean():>11.1%} {(d['v14_outcome']==d['true_outcome']).mean():>11.1%} {(d['v14_outcome']==d['true_outcome']).mean()-(d['v13_outcome']==d['true_outcome']).mean():>+12.1%}")

# By true outcome
print(f"\n{'='*70}")
print("BY TRUE OUTCOME")
print(f"{'='*70}")
for out_val, out_label in [(-1, "Away Win"), (0, "Draw"), (1, "Home Win")]:
    d = df[df["true_outcome"] == out_val]
    print(f"\n--- {out_label} ({len(d)} matches) ---")
    print(f"{'AW-MAE':<40} {d['loss_v13'].mean():>12.5f} {d['loss_v14'].mean():>12.5f} {d['delta'].mean():>+12.5f}")
    print(f"{'Exact Match Rate':<40} {d['v13_exact'].mean():>11.1%} {d['v14_exact'].mean():>11.1%}")

# By gender × outcome
print(f"\n{'='*70}")
print("BY GENDER × OUTCOME")
print(f"{'='*70}")
for gender, g_label in [(False, "Men"), (True, "Women")]:
    for out_val, out_label in [(-1, "Away Win"), (0, "Draw"), (1, "Home Win")]:
        d = df[(df["is_women"] == gender) & (df["true_outcome"] == out_val)]
        if len(d) == 0: continue
        print(f"  {g_label:<6} {out_label:<10} ({len(d):>5})  V13: {d['loss_v13'].mean():.4f}  V14: {d['loss_v14'].mean():.4f}  Delta: {d['delta'].mean():+.4f}")

# Where V14 is better vs worse
v14_better = df[df["delta"] < -0.001]
v14_worse = df[df["delta"] > 0.001]
v14_same = df[abs(df["delta"]) <= 0.001]
print(f"\n{'='*70}")
print("V14 PERFORMANCE BREAKDOWN")
print(f"{'='*70}")
print(f"  Better than V13: {len(v14_better):>6} ({len(v14_better)/len(df)*100:.1f}%) — avg delta: {v14_better['delta'].mean():.4f}")
print(f"  Same as V13:    {len(v14_same):>6} ({len(v14_same)/len(df)*100:.1f}%)")
print(f"  Worse than V13: {len(v14_worse):>6} ({len(v14_worse)/len(df)*100:.1f}%) — avg delta: {v14_worse['delta'].mean():+.4f}")

# Top 20 worst degradations
print(f"\n{'='*70}")
print("TOP 20 WORST DEGRADATIONS (V14 >> V13)")
print(f"{'='*70}")
worst = df.nlargest(20, "delta")
for _, r in worst.iterrows():
    print(f"  {r['Id']:<12} True: {int(r['team_goals_true'])}-{int(r['opp_goals_true'])}  V13: {int(r['team_goals_v13'])}-{int(r['opp_goals_v13'])}  V14: {int(r['team_goals_v14'])}-{int(r['opp_goals_v14'])}  Delta: {r['delta']:+.4f}")

# Goal-level comparison
print(f"\n{'='*70}")
print("GOAL PREDICTION ACCURACY")
print(f"{'='*70}")
for label, col13, col14 in [("Team Goals", "team_goals_v13", "team_goals_v14"), ("Opp Goals", "opp_goals_v13", "opp_goals_v14")]:
    acc13 = (df[col13] == df[col13.replace("_v13","_true")]).mean()
    acc14 = (df[col14] == df[col14.replace("_v14","_true")]).mean()
    mae13 = abs(df[col13] - df[col13.replace("_v13","_true")]).mean()
    mae14 = abs(df[col14] - df[col14.replace("_v14","_true")]).mean()
    print(f"  {label:<15} Accuracy: V13={acc13:.3f} V14={acc14:.3f}  MAE: V13={mae13:.3f} V14={mae14:.3f}")

print(f"\n{'='*70}")
print("CONCLUSION")
print(f"{'='*70}")
# Check how many women matches V14 changed from V13
w = df[df["is_women"]]
w_changed = w[w["team_goals_v13"] != w["team_goals_v14"]]
print(f"Women matches where V14 != V13: {len(w_changed)}/{len(w)} ({len(w_changed)/len(w)*100:.1f}%)")
w_changed_better = w_changed[w_changed["delta"] < 0]
w_changed_worse = w_changed[w_changed["delta"] > 0]
print(f"  Of those, V14 better: {len(w_changed_better)}, V14 worse: {len(w_changed_worse)}")

m = df[~df["is_women"]]
m_changed = m[m["team_goals_v13"] != m["team_goals_v14"]]
print(f"Men matches where V14 != V13:   {len(m_changed)}/{len(m)} ({len(m_changed)/len(m)*100:.1f}%)")
m_changed_better = m_changed[m_changed["delta"] < 0]
m_changed_worse = m_changed[m_changed["delta"] > 0]
print(f"  Of those, V14 better: {len(m_changed_better)}, V14 worse: {len(m_changed_worse)}")