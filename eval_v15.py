import pandas as pd, numpy as np, sys

NLS_POWER = 1.3

def awmae_single(pt, po, tt, to_):
    mae = (abs(int(pt)-int(tt)) + abs(int(po)-int(to_))) / 2.0
    exact = 1 if (int(pt)==int(tt) and int(po)==int(to_)) else 0
    out_ok = 1 if np.sign(int(pt)-int(po)) == np.sign(int(tt)-int(to_)) else 0
    gd_ok = 1 if (int(pt)-int(po)) == (int(tt)-int(to_)) else 0
    aug = mae + 0.30*(1-exact) + 0.25*(1-out_ok) + 0.15*(1-gd_ok)
    mult = 1.0 if out_ok else 1.5
    return (aug * mult) ** NLS_POWER

gt = pd.read_csv('dataset/test_ground_truth.csv')
gt.columns = ['Id','tt','to']

subs = {}
for name, f in [('V13_lite','dataset/submission_v13_lite.csv'),('V14','dataset/submission_v14.csv'),('V15','dataset/submission_v15.csv')]:
    s = pd.read_csv(f)
    s.columns = ['Id',f'tg_{name}',f'og_{name}']
    subs[name] = s

df = gt
for name, s in subs.items():
    df = df.merge(s, on='Id')
    df[f'loss_{name}'] = df.apply(lambda r: awmae_single(r[f'tg_{name}'], r[f'og_{name}'], r['tt'], r['to']), axis=1)
    df[f'out_{name}'] = np.sign(df[f'tg_{name}'] - df[f'og_{name}']) == np.sign(df['tt'] - df['to'])
    df[f'exact_{name}'] = (df[f'tg_{name}']==df['tt']) & (df[f'og_{name}']==df['to'])

df['is_women'] = df['Id'].str.startswith('W')

# Generate report
lines = []
lines.append(f"{'Version':<12} {'AW-MAE':>10} {'Outcome':>10} {'Exact':>10}")
lines.append('-'*45)
for name in ['V13_lite','V14','V15']:
    lines.append(f"{name:<12} {df[f'loss_{name}'].mean():>10.4f} {df[f'out_{name}'].mean()*100:>9.1f}% {df[f'exact_{name}'].mean()*100:>9.1f}%")

lines.append(f'\n--- By Gender ---')
for name in ['V13_lite','V14','V15']:
    for g, label in [(False,'Men'),(True,'Women')]:
        d = df[df['is_women']==g]
        lines.append(f"{name:<12} {label:<6} {d[f'loss_{name}'].mean():>10.4f} {d[f'out_{name}'].mean()*100:>9.1f}% {d[f'exact_{name}'].mean()*100:>9.1f}%")

lines.append(f'\n--- Delta ---')
for a, b in [('V14','V13_lite'),('V15','V14'),('V15','V13_lite')]:
    delta = df[f'loss_{a}'].mean() - df[f'loss_{b}'].mean()
    lines.append(f"{a} vs {b}: {delta:+.4f} AW-MAE")

with open('eval_v15_result.txt', 'w') as f:
    f.write('\n'.join(lines))

print('\n'.join(lines))