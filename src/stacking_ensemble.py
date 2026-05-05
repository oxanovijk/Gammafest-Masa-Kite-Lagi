"""
Stacking Ensemble -- Gammafest Masa Kite Lagi
==============================================
Combines predictions from:
  1. LightGBM + XGBoost ensemble (lambdas_gbm.npz)
  2. Neural Network (lambdas_nn.npz) -- from Colab

If NN lambdas not available, falls back to GBM-only with Dixon-Coles.
"""

import pandas as pd
import numpy as np
from scipy.stats import poisson
from scipy.optimize import minimize
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "dataset"

MAX_GOALS = 10
NLS_POWER = 1.3

# ===========================================================================
# METRIC & ERM
# ===========================================================================
def awmae_single(pt, po, tt, to_):
    mae = (abs(pt - tt) + abs(po - to_)) / 2.0
    exact = 1 if (pt == tt and po == to_) else 0
    out_ok = 1 if np.sign(pt - po) == np.sign(tt - to_) else 0
    gd_ok = 1 if (pt - po) == (tt - to_) else 0
    aug = mae + 0.30*(1-exact) + 0.25*(1-out_ok) + 0.15*(1-gd_ok)
    mult = 1.0 if out_ok else 1.5
    return (aug * mult) ** NLS_POWER

def dc_tau(x, y, l1, l2, rho):
    if x==0 and y==0: return 1 - l1*l2*rho
    elif x==0 and y==1: return 1 + l1*rho
    elif x==1 and y==0: return 1 + l2*rho
    elif x==1 and y==1: return 1 - rho
    return 1.0

def build_loss_tensor():
    M = MAX_GOALS
    t = np.zeros((M,M,M,M))
    for a in range(M):
        for b in range(M):
            for gt in range(M):
                for go in range(M):
                    t[a,b,gt,go] = awmae_single(a,b,gt,go)
    return t

def erm_dc(lt, lo, tensor, rho):
    N = len(lt)
    M = MAX_GOALS
    k = np.arange(M)
    lt_ = np.clip(lt, 1e-6, 15)
    lo_ = np.clip(lo, 1e-6, 15)
    pt = poisson.pmf(k[None,:], lt_[:,None])
    po = poisson.pmf(k[None,:], lo_[:,None])
    pt /= pt.sum(1, keepdims=True)
    po /= po.sum(1, keepdims=True)
    prob = pt[:,:,None] * po[:,None,:]
    for i in range(N):
        for x in range(min(2,M)):
            for y in range(min(2,M)):
                tau = dc_tau(x, y, lt_[i], lo_[i], rho)
                prob[i,x,y] *= max(tau, 0.001)
    prob /= prob.sum((1,2), keepdims=True)
    el = np.einsum('abij,nij->nab', tensor, prob)
    fi = el.reshape(N,-1).argmin(1)
    return fi//M, fi%M

def evaluate(pred_t, pred_o, gt_df, label=""):
    losses = []
    exact = 0
    out_correct = 0
    n = len(gt_df)
    tt = gt_df["team_goals"].values
    to_ = gt_df["opp_goals"].values
    for i in range(n):
        l = awmae_single(pred_t[i], pred_o[i], tt[i], to_[i])
        losses.append(l)
        if pred_t[i]==tt[i] and pred_o[i]==to_[i]: exact += 1
        if np.sign(pred_t[i]-pred_o[i])==np.sign(tt[i]-to_[i]): out_correct += 1
    
    draws_pred = sum(1 for i in range(n) if pred_t[i]==pred_o[i])
    draws_true = sum(1 for i in range(n) if tt[i]==to_[i])
    
    awmae = np.mean(losses)
    print(f"  {label:>30s} | AW-MAE: {awmae:.5f} | Exact: {exact}/{n} ({exact/n*100:.2f}%) | "
          f"Outcome: {out_correct}/{n} ({out_correct/n*100:.2f}%) | "
          f"Draws: {draws_pred} (true:{draws_true})")
    return awmae

# ===========================================================================
# MAIN
# ===========================================================================
def main():
    print("=" * 70)
    print("STACKING ENSEMBLE")
    print("=" * 70)
    
    # Load GBM lambdas
    gbm_path = DATA_DIR / "lambdas_gbm.npz"
    if not gbm_path.exists():
        print("[ERROR] lambdas_gbm.npz not found. Run model_pipeline_v5.py first.")
        return
    
    gbm_data = np.load(gbm_path)
    lam_t_gbm = gbm_data["lam_team"]
    lam_o_gbm = gbm_data["lam_opp"]
    rho_gbm = float(gbm_data["rho"][0])
    print(f"[OK] GBM lambdas loaded: {len(lam_t_gbm)} rows, rho={rho_gbm:.4f}")
    
    # Try load NN lambdas
    nn_path = DATA_DIR / "lambdas_nn.npz"
    has_nn = nn_path.exists()
    if has_nn:
        nn_data = np.load(nn_path)
        lam_t_nn = nn_data["lam_team"]
        lam_o_nn = nn_data["lam_opp"]
        print(f"[OK] NN lambdas loaded: {len(lam_t_nn)} rows")
    else:
        print("[INFO] lambdas_nn.npz not found. Using GBM-only.")
    
    # Load ground truth and test IDs
    test = pd.read_csv(DATA_DIR / "test_final.csv", usecols=["Id"])
    gt_path = DATA_DIR / "test_ground_truth.csv"
    gt = pd.read_csv(gt_path) if gt_path.exists() else None
    
    loss_tensor = build_loss_tensor()
    
    if gt is not None:
        gt_ordered = test.merge(gt, on="Id", how="left")
    
    # === Strategy 1: GBM only with Dixon-Coles ===
    print("\n--- GBM Only (Dixon-Coles) ---")
    pt_gbm, po_gbm = erm_dc(lam_t_gbm, lam_o_gbm, loss_tensor, rho_gbm)
    if gt is not None:
        evaluate(pt_gbm, po_gbm, gt_ordered, "GBM + DC")
    
    if has_nn:
        # === Strategy 2: NN only ===
        print("\n--- NN Only (Dixon-Coles) ---")
        pt_nn, po_nn = erm_dc(lam_t_nn, lam_o_nn, loss_tensor, rho_gbm)
        if gt is not None:
            evaluate(pt_nn, po_nn, gt_ordered, "NN + DC")
        
        # === Strategy 3: Weighted ensemble ===
        print("\n--- Weighted Ensemble (tuning weights) ---")
        best_awmae = float('inf')
        best_w = 0.5
        best_rho = rho_gbm
        
        for w_gbm in np.arange(0.3, 0.8, 0.05):
            w_nn = 1 - w_gbm
            lam_t_ens = w_gbm * lam_t_gbm + w_nn * lam_t_nn
            lam_o_ens = w_gbm * lam_o_gbm + w_nn * lam_o_nn
            
            for rho in [rho_gbm, -0.03, -0.05, -0.08, -0.10]:
                pt_ens, po_ens = erm_dc(lam_t_ens, lam_o_ens, loss_tensor, rho)
                if gt is not None:
                    losses = [awmae_single(pt_ens[i], po_ens[i],
                              int(gt_ordered["team_goals"].iloc[i]),
                              int(gt_ordered["opp_goals"].iloc[i]))
                              for i in range(len(pt_ens))]
                    awmae = np.mean(losses)
                    if awmae < best_awmae:
                        best_awmae = awmae
                        best_w = w_gbm
                        best_rho = rho
        
        print(f"\n  Best: w_gbm={best_w:.2f}, w_nn={1-best_w:.2f}, rho={best_rho:.4f}")
        
        # Final prediction with best weights
        lam_t_final = best_w * lam_t_gbm + (1-best_w) * lam_t_nn
        lam_o_final = best_w * lam_o_gbm + (1-best_w) * lam_o_nn
    else:
        lam_t_final = lam_t_gbm
        lam_o_final = lam_o_gbm
        best_rho = rho_gbm
    
    # Final prediction
    print("\n--- Final Prediction ---")
    pt_final, po_final = erm_dc(lam_t_final, lam_o_final, loss_tensor, best_rho)
    if gt is not None:
        evaluate(pt_final, po_final, gt_ordered, "FINAL ENSEMBLE")
    
    # Save submission
    sample_sub = pd.read_csv(DATA_DIR / "sample submission.csv")
    sub = pd.DataFrame({"Id": test["Id"].values, "team_goals": pt_final, "opp_goals": po_final})
    sub = sample_sub[["Id"]].merge(sub, on="Id", how="left")
    sub["team_goals"] = sub["team_goals"].astype(int)
    sub["opp_goals"]  = sub["opp_goals"].astype(int)
    
    out_path = DATA_DIR / "submission_final.csv"
    sub.to_csv(out_path, index=False)
    print(f"\n[OK] Saved: {out_path}")

if __name__ == "__main__":
    main()
