"""
Script: evaluate_local.py
Tujuan: Mengukur akurasi submission.csv secara luring/offline menyerupai server Kaggle.
Aturan: DILARANG KERAS mengimpor script ini ke dalam feature_engineering.py.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

def awmae_single(pred_t, pred_o, true_t, true_o, nls_power=1.3):
    mae = (abs(pred_t - true_t) + abs(pred_o - true_o)) / 2.0
    exact = 1 if (pred_t == true_t and pred_o == true_o) else 0
    pred_out = np.sign(pred_t - pred_o)
    true_out = np.sign(true_t - true_o)
    out_ok = 1 if pred_out == true_out else 0
    gd_ok = 1 if (pred_t - pred_o) == (true_t - true_o) else 0
    aug = mae + 0.30*(1 - exact) + 0.25*(1 - out_ok) + 0.15*(1 - gd_ok)
    mult = 1.0 if out_ok else 1.5
    return (aug * mult) ** nls_power

def evaluate_submission(sub_path="dataset/submission.csv", gt_path="dataset/test_ground_truth.csv", verbose=True):
    sub = pd.read_csv(sub_path)
    gt = pd.read_csv(gt_path)

    if len(sub) != len(gt):
        print(f"[!] ERROR: Panjang baris beda! sub={len(sub)}, gt={len(gt)}")
        return None

    df = pd.merge(sub, gt, on="Id", suffixes=("_pred", "_true"))
    
    # Hitung per baris
    df["loss"] = df.apply(lambda r: awmae_single(r['team_goals_pred'], r['opp_goals_pred'], r['team_goals_true'], r['opp_goals_true']), axis=1)
    
    # Hitung Unweighted
    score_unweighted = df["loss"].mean()
    
    if verbose:
        print("="*50)
        print("KAGGLE LOCAL LEADERBOARD")
        print("="*50)
        print(f"File Submission : {sub_path}")
        print(f"Total Dievaluasi: {len(df)} laga")
        print("-" * 50)
        print(f"AW-MAE SCORE (Unweighted): {score_unweighted:.5f}")
        
        # Breakdown kesalahan
        exact_matches = (df["team_goals_pred"] == df["team_goals_true"]) & (df["opp_goals_pred"] == df["opp_goals_true"])
        pred_out = np.sign(df["team_goals_pred"] - df["opp_goals_pred"])
        true_out = np.sign(df["team_goals_true"] - df["opp_goals_true"])
        out_matches = pred_out == true_out
        
        print(f"  - Exact Score Correct : {exact_matches.sum()} / {len(df)} ({exact_matches.mean()*100:.1f}%)")
        print(f"  - Outcome M/S/K Correct: {out_matches.sum()} / {len(df)} ({out_matches.mean()*100:.1f}%)")
        print("="*50)
        
    return score_unweighted

if __name__ == "__main__":
    import sys
    sub_arg = sys.argv[1] if len(sys.argv) > 1 else "dataset/submission.csv"
    evaluate_submission(sub_arg)
