"""
ML Pipeline -- Gammafest Masa Kite Lagi
========================================
Peran  : Anggota 2 (ML Pipeline Architect)

Arsitektur:
  1. Dual LightGBM Poisson Regression (team_goals & opp_goals terpisah)
  2. Tournament-based Sample Weighting
  3. Time-Series Cross Validation (2 fold)
  4. Full Expected Risk Minimization (Bivariate Poisson -> AW-MAE minimization)

Input  : dataset/train_final.csv, dataset/test_final.csv
Output : dataset/submission.csv
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from scipy.stats import poisson
from pathlib import Path
import warnings
import time

warnings.filterwarnings("ignore")

# ===========================================================================
# 0. KONFIGURASI
# ===========================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "dataset"

TRAIN_FINAL = DATA_DIR / "train_final.csv"
TEST_FINAL  = DATA_DIR / "test_final.csv"
TRAIN_RAW   = DATA_DIR / "train.csv"
TEST_RAW    = DATA_DIR / "test.csv"
SAMPLE_SUB  = DATA_DIR / "sample submission.csv"
OUTPUT_SUB  = DATA_DIR / "submission.csv"

# --- ERM Config ---
MAX_GOALS = 10        # Matriks skor 0-9
NLS_POWER = 1.3       # Non-linear scaling exponent

# --- LightGBM Config ---
LGB_PARAMS = {
    "objective":        "poisson",
    "metric":           "poisson",
    "num_leaves":       15,        # [DRIFT CONTROL] Very small leaves
    "learning_rate":    0.015,     # [DRIFT CONTROL] Slow learning
    "min_child_samples": 120,      # [DRIFT CONTROL] Needs robust evidence
    "reg_alpha":        5.0,       # [DRIFT CONTROL] High L1 penalty
    "reg_lambda":       10.0,      # [DRIFT CONTROL] High L2 penalty
    "subsample":        0.7,
    "colsample_bytree": 0.6,
    "verbose":          -1,
    "n_jobs":           -1,
    "seed":             42,
}
N_ESTIMATORS    = 1200
EARLY_STOPPING  = 100

# --- Tournament Weights (AW-MAE) ---
# Sumber: deskripsi kompetisi + estimasi untuk turnamen lain
TOURNAMENT_WEIGHT_MAP = {
    # Tier 1 — Puncak prestisius
    "FIFA World Cup":                           2.00,
    # Tier 2 — Championship benua
    "AFC Asian Cup":                            1.80,
    "AFC Championship":                         1.80,
    "African Cup of Nations":                   1.80,
    "Copa America":                             1.80,
    "UEFA Euro":                                1.80,
    "Gold Cup":                                 1.70,
    "CONCACAF Championship":                    1.70,
    "Oceania Nations Cup":                      1.60,
    "Confederations Cup":                       1.70,
    "Finalissima":                              1.70,
    # Tier 3 — Kualifikasi besar & Olympic
    "FIFA World Cup qualification":             1.50,
    "Olympic Games":                            1.50,
    # Tier 4 — Kualifikasi benua & Nations League
    "UEFA Euro qualification":                  1.40,
    "African Cup of Nations qualification":     1.40,
    "AFC Asian Cup qualification":              1.40,
    "CONCACAF Gold Cup qualification":          1.30,
    "UEFA Nations League":                      1.50,
    "CONCACAF Nations League":                  1.40,
    "CONMEBOL Nations League":                  1.40,
    # Tier Friendly
    "Friendly":                                 0.96,
}
DEFAULT_TOURNAMENT_WEIGHT = 1.20

# --- CV Splits (time-series) ---
CV_SPLITS = [
    # (train_start, train_end, val_start, val_end)
    ("1872-01-01", "2005-12-31", "2006-01-01", "2008-12-31"),
    ("1872-01-01", "2008-12-31", "2009-01-01", "2011-08-31"),
]


# ===========================================================================
# 1. DATA LOADING
# ===========================================================================
def load_data():
    """
    Membaca train_final.csv, test_final.csv, dan metadata dari raw CSVs.
    Returns:
        train_df : DataFrame dengan fitur + target + date + tournament + weight
        test_df  : DataFrame dengan fitur
        feature_cols : list nama kolom fitur
    """
    print("[1] Loading data...")
    train = pd.read_csv(TRAIN_FINAL)
    test  = pd.read_csv(TEST_FINAL)

    # Ambil date & tournament dari raw CSV via Id
    raw_train = pd.read_csv(TRAIN_RAW, usecols=["Id", "date", "tournament"])
    raw_test  = pd.read_csv(TEST_RAW,  usecols=["Id", "date", "tournament"])

    train = train.merge(raw_train, on="Id", how="left")
    test  = test.merge(raw_test, on="Id", how="left")

    train["date"] = pd.to_datetime(train["date"])
    test["date"]  = pd.to_datetime(test["date"])

    # Mapping tournament -> sample weight
    train["sample_weight"] = train["tournament"].map(
        TOURNAMENT_WEIGHT_MAP
    ).fillna(DEFAULT_TOURNAMENT_WEIGHT)

    # Kolom fitur = semua kecuali Id, target, metadata
    exclude = {"Id", "team_goals", "opp_goals", "date", "tournament",
               "sample_weight", "is_test"}
    feature_cols = [c for c in train.columns if c not in exclude]

    print(f"    Train: {train.shape}  |  Test: {test.shape}")
    print(f"    Fitur: {len(feature_cols)} kolom")
    print(f"    Target mean: team_goals={train['team_goals'].mean():.3f}, "
          f"opp_goals={train['opp_goals'].mean():.3f}")

    return train, test, feature_cols


# ===========================================================================
# 2. AW-MAE METRIC
# ===========================================================================
def awmae_single(pred_t, pred_o, true_t, true_o):
    """Hitung komponen AW-MAE untuk satu pertandingan (tanpa tournament weight)."""
    # Base MAE
    mae = (abs(pred_t - true_t) + abs(pred_o - true_o)) / 2.0

    # Exact score
    exact = 1 if (pred_t == true_t and pred_o == true_o) else 0

    # Outcome (M/S/K)
    pred_outcome = np.sign(pred_t - pred_o)
    true_outcome = np.sign(true_t - true_o)
    outcome_ok = 1 if pred_outcome == true_outcome else 0

    # Goal difference
    gd_ok = 1 if (pred_t - pred_o) == (true_t - true_o) else 0

    # Augmented MAE
    augmented = mae + 0.30 * (1 - exact) + 0.25 * (1 - outcome_ok) + 0.15 * (1 - gd_ok)

    # Outcome multiplier
    multiplier = 1.0 if outcome_ok else 1.5

    # Non-linear scaling
    scaled = (augmented * multiplier) ** NLS_POWER

    return scaled


def compute_awmae(pred_team, pred_opp, true_team, true_opp, weights=None):
    """
    Hitung AW-MAE keseluruhan.
    pred/true: array integer.
    weights: tournament weights per match (optional).
    """
    n = len(pred_team)
    errors = np.zeros(n)
    for i in range(n):
        errors[i] = awmae_single(
            pred_team[i], pred_opp[i],
            true_team[i], true_opp[i]
        )
    if weights is not None:
        return np.average(errors, weights=weights)
    return np.mean(errors)


# ===========================================================================
# 3. LOSS TENSOR (Precomputed) untuk ERM
# ===========================================================================
def build_loss_tensor(max_goals=MAX_GOALS):
    """
    Precompute loss_tensor[a, b, gt, go] untuk semua kombinasi skor.
    a, b  = skor prediksi (kandidat)
    gt, go = skor ground-truth (yang mungkin terjadi)

    Ukuran: MAX_GOALS^4 entries (10^4 = 10.000 -> trivial).
    """
    tensor = np.zeros((max_goals, max_goals, max_goals, max_goals))
    for a in range(max_goals):
        for b in range(max_goals):
            for gt in range(max_goals):
                for go in range(max_goals):
                    tensor[a, b, gt, go] = awmae_single(a, b, gt, go)
    return tensor


# ===========================================================================
# 4. EXPECTED RISK MINIMIZATION (Vectorized)
# ===========================================================================
def erm_predict_batch(lambdas_team, lambdas_opp, loss_tensor):
    """
    Untuk setiap pertandingan, temukan skor integer (a, b) yang
    meminimalkan expected AW-MAE di bawah distribusi Poisson bivariat
    independen.

    Parameters
    ----------
    lambdas_team : array (N,) - prediksi lambda (rata-rata gol) tim
    lambdas_opp  : array (N,) - prediksi lambda (rata-rata gol) lawan
    loss_tensor  : array (M, M, M, M) - precomputed per-score loss

    Returns
    -------
    pred_team, pred_opp : array (N,) integer - skor optimal
    """
    N = len(lambdas_team)
    M = loss_tensor.shape[0]
    k = np.arange(M)

    # Clip lambda agar Poisson PMF stabil (hindari overflow)
    lam_t = np.clip(lambdas_team, 1e-6, 15.0)
    lam_o = np.clip(lambdas_opp,  1e-6, 15.0)

    # PMF per match: shape (N, M)
    pmf_team = poisson.pmf(k[None, :], lam_t[:, None])  # P(goals=k | lambda_team)
    pmf_opp  = poisson.pmf(k[None, :], lam_o[:, None])  # P(goals=k | lambda_opp)

    # Normalisasi (karena truncation di MAX_GOALS)
    pmf_team = pmf_team / pmf_team.sum(axis=1, keepdims=True)
    pmf_opp  = pmf_opp  / pmf_opp.sum(axis=1, keepdims=True)

    # Joint probability matrix per match: shape (N, M, M)
    # prob[n, gt, go] = P(team=gt) * P(opp=go)
    prob = pmf_team[:, :, None] * pmf_opp[:, None, :]  # (N, M, M)

    # Expected loss per kandidat (a, b) per match:
    # E[loss(a,b)] = sum_{gt,go} loss_tensor[a,b,gt,go] * prob[n,gt,go]
    # Menggunakan einsum: 'abij,nij->nab'
    expected_loss = np.einsum("abij,nij->nab", loss_tensor, prob)
    # Shape: (N, M, M) dimana [n, a, b] = expected AW-MAE jika prediksi (a, b)

    # Argmin per match
    flat_idx = expected_loss.reshape(N, -1).argmin(axis=1)
    pred_team = flat_idx // M
    pred_opp  = flat_idx % M

    return pred_team.astype(int), pred_opp.astype(int)


# ===========================================================================
# 5. MODEL TRAINING
# ===========================================================================
def train_lgb(X_train, y_train, X_val, y_val,
              sample_weight_train=None, sample_weight_val=None):
    """
    Melatih satu model LightGBM Poisson.
    Returns: trained model
    """
    dtrain = lgb.Dataset(
        X_train, y_train,
        weight=sample_weight_train,
        free_raw_data=False,
    )
    dval = lgb.Dataset(
        X_val, y_val,
        weight=sample_weight_val,
        reference=dtrain,
        free_raw_data=False,
    )

    callbacks = [
        lgb.early_stopping(EARLY_STOPPING, verbose=False),
        lgb.log_evaluation(period=0),  # suppress per-round logs
    ]

    model = lgb.train(
        LGB_PARAMS,
        dtrain,
        num_boost_round=N_ESTIMATORS,
        valid_sets=[dval],
        callbacks=callbacks,
    )
    return model


# ===========================================================================
# 6. TIME-SERIES CROSS VALIDATION
# ===========================================================================
def run_cv(train_df, feature_cols, loss_tensor):
    """
    Menjalankan Time-Series CV.
    Returns: list of AW-MAE scores per fold.
    """
    print("\n[3] Running Time-Series Cross Validation...")
    cv_scores = []

    for fold_i, (tr_start, tr_end, val_start, val_end) in enumerate(CV_SPLITS):
        print(f"\n    --- Fold {fold_i+1} ---")
        print(f"    Train: {tr_start} -> {tr_end}")
        print(f"    Val:   {val_start} -> {val_end}")

        mask_tr  = (train_df["date"] >= tr_start) & (train_df["date"] <= tr_end)
        mask_val = (train_df["date"] >= val_start) & (train_df["date"] <= val_end)

        tr  = train_df[mask_tr]
        val = train_df[mask_val]

        if len(val) == 0:
            print("    [SKIP] Tidak ada data validasi di rentang ini.")
            continue

        print(f"    Train size: {len(tr)} | Val size: {len(val)}")

        X_tr  = tr[feature_cols]
        X_val = val[feature_cols]

        # --- Model team_goals ---
        model_team = train_lgb(
            X_tr, tr["team_goals"],
            X_val, val["team_goals"],
            sample_weight_train=tr["sample_weight"].values,
            sample_weight_val=val["sample_weight"].values,
        )
        lambda_team = model_team.predict(X_val)
        print(f"    Model team_goals: best_iter={model_team.best_iteration}")

        # --- Model opp_goals ---
        model_opp = train_lgb(
            X_tr, tr["opp_goals"],
            X_val, val["opp_goals"],
            sample_weight_train=tr["sample_weight"].values,
            sample_weight_val=val["sample_weight"].values,
        )
        lambda_opp = model_opp.predict(X_val)
        print(f"    Model opp_goals:  best_iter={model_opp.best_iteration}")

        # --- ERM ---
        pred_t, pred_o = erm_predict_batch(lambda_team, lambda_opp, loss_tensor)

        # --- Hitung AW-MAE ---
        val_weights = val["sample_weight"].values
        score = compute_awmae(
            pred_t, pred_o,
            val["team_goals"].values.astype(int),
            val["opp_goals"].values.astype(int),
            weights=val_weights,
        )
        cv_scores.append(score)
        print(f"    AW-MAE (weighted): {score:.6f}")

        # --- Baseline comparison (selalu prediksi 1-1) ---
        baseline_score = compute_awmae(
            np.ones(len(val), dtype=int),
            np.ones(len(val), dtype=int),
            val["team_goals"].values.astype(int),
            val["opp_goals"].values.astype(int),
            weights=val_weights,
        )
        print(f"    Baseline (1-1):    {baseline_score:.6f}")

        # --- Distribusi prediksi ---
        unique_t, counts_t = np.unique(pred_t, return_counts=True)
        print(f"    Dist pred team_goals: "
              + ", ".join(f"{g}:{c}" for g, c in zip(unique_t, counts_t)))
        unique_o, counts_o = np.unique(pred_o, return_counts=True)
        print(f"    Dist pred opp_goals:  "
              + ", ".join(f"{g}:{c}" for g, c in zip(unique_o, counts_o)))

    if cv_scores:
        print(f"\n    === Rata-rata AW-MAE CV: {np.mean(cv_scores):.6f} ===")

    return cv_scores


# ===========================================================================
# 7. FULL TRAINING & SUBMISSION
# ===========================================================================
def run_full_pipeline(train_df, test_df, feature_cols, loss_tensor):
    """
    Melatih model pada seluruh data train, prediksi test, ERM, dan simpan.
    """
    print("\n[4] Training final models on full data...")

    X_train = train_df[feature_cols]
    X_test  = test_df[feature_cols]

    # Split: 95% train, 5% terakhir sebagai early stopping set
    n = len(train_df)
    split_idx = int(n * 0.95)
    # Urut berdasarkan date untuk time-series consistency
    sorted_idx = train_df["date"].argsort().values
    tr_idx  = sorted_idx[:split_idx]
    es_idx  = sorted_idx[split_idx:]

    X_tr  = X_train.iloc[tr_idx]
    X_es  = X_train.iloc[es_idx]
    w_tr  = train_df["sample_weight"].values[tr_idx]
    w_es  = train_df["sample_weight"].values[es_idx]

    # --- Model team_goals ---
    print("    Training model_team_goals...")
    model_team = train_lgb(
        X_tr, train_df["team_goals"].values[tr_idx],
        X_es, train_df["team_goals"].values[es_idx],
        sample_weight_train=w_tr,
        sample_weight_val=w_es,
    )
    print(f"    best_iteration: {model_team.best_iteration}")

    # --- Model opp_goals ---
    print("    Training model_opp_goals...")
    model_opp = train_lgb(
        X_tr, train_df["opp_goals"].values[tr_idx],
        X_es, train_df["opp_goals"].values[es_idx],
        sample_weight_train=w_tr,
        sample_weight_val=w_es,
    )
    print(f"    best_iteration: {model_opp.best_iteration}")

    # --- Predict ---
    print("\n[5] Predicting test set...")
    lambda_team = model_team.predict(X_test)
    lambda_opp  = model_opp.predict(X_test)

    print(f"    Lambda team: mean={lambda_team.mean():.3f}, "
          f"min={lambda_team.min():.3f}, max={lambda_team.max():.3f}")
    print(f"    Lambda opp:  mean={lambda_opp.mean():.3f}, "
          f"min={lambda_opp.min():.3f}, max={lambda_opp.max():.3f}")

    # --- ERM ---
    print("\n[6] Applying Expected Risk Minimization...")
    t0 = time.time()
    pred_team, pred_opp = erm_predict_batch(lambda_team, lambda_opp, loss_tensor)
    elapsed = time.time() - t0
    print(f"    ERM selesai dalam {elapsed:.1f} detik")

    # Distribusi prediksi
    unique_t, counts_t = np.unique(pred_team, return_counts=True)
    print(f"    Dist pred team_goals: "
          + ", ".join(f"{g}:{c}" for g, c in zip(unique_t, counts_t)))
    unique_o, counts_o = np.unique(pred_opp, return_counts=True)
    print(f"    Dist pred opp_goals:  "
          + ", ".join(f"{g}:{c}" for g, c in zip(unique_o, counts_o)))

    # --- Feature Importance ---
    print("\n    Top 15 Feature Importance (team_goals model):")
    imp_team = pd.Series(
        model_team.feature_importance(importance_type="gain"),
        index=feature_cols,
    ).sort_values(ascending=False)
    for feat, score in imp_team.head(15).items():
        print(f"      {feat:45s} {score:.1f}")

    print("\n    Top 15 Feature Importance (opp_goals model):")
    imp_opp = pd.Series(
        model_opp.feature_importance(importance_type="gain"),
        index=feature_cols,
    ).sort_values(ascending=False)
    for feat, score in imp_opp.head(15).items():
        print(f"      {feat:45s} {score:.1f}")

    # --- Submission ---
    print("\n[7] Generating submission.csv...")
    sample_sub = pd.read_csv(SAMPLE_SUB)

    sub = pd.DataFrame({
        "Id": test_df["Id"].values,
        "team_goals": pred_team,
        "opp_goals":  pred_opp,
    })

    # Pastikan urutan Id sesuai sample submission
    sub = sample_sub[["Id"]].merge(sub, on="Id", how="left")

    # Validasi
    assert len(sub) == len(sample_sub), \
        f"Jumlah baris tidak cocok: {len(sub)} vs {len(sample_sub)}"
    assert sub["team_goals"].notna().all(), "Ada NaN di team_goals!"
    assert sub["opp_goals"].notna().all(),  "Ada NaN di opp_goals!"

    sub["team_goals"] = sub["team_goals"].astype(int)
    sub["opp_goals"]  = sub["opp_goals"].astype(int)

    sub.to_csv(OUTPUT_SUB, index=False)
    print(f"    [OK] submission.csv -> {len(sub)} baris")
    print(f"    Preview:")
    print(sub.head(10).to_string(index=False))

    return model_team, model_opp


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    print("=" * 60)
    print("ML PIPELINE - Gammafest Masa Kite Lagi")
    print("=" * 60)

    # --- Load ---
    train_df, test_df, feature_cols = load_data()

    # --- Build loss tensor ---
    print("\n[2] Building AW-MAE loss tensor...")
    t0 = time.time()
    loss_tensor = build_loss_tensor(MAX_GOALS)
    print(f"    Loss tensor shape: {loss_tensor.shape} "
          f"({time.time()-t0:.1f}s)")

    # --- CV ---
    cv_scores = run_cv(train_df, feature_cols, loss_tensor)

    # --- Full training + submission ---
    run_full_pipeline(train_df, test_df, feature_cols, loss_tensor)

    print("\n" + "=" * 60)
    print("PIPELINE SELESAI")
    print("=" * 60)


if __name__ == "__main__":
    main()
