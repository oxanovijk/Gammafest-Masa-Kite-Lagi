"""
Feature Engineering Pipeline — Contextual & Socio-Economic Features
=====================================================================
Peran  : Anggota 2 (Feature Engineer - Contextual, Geo & Socio-Economic)
Tujuan : Membuat fitur kontekstual baru dari train_featured.csv dan test_featured.csv
         yang dihasilkan oleh Anggota 1.

Fitur yang dibuat (semua berakhiran _ctx):
  - Geo-Spatial & Physical Stress   : travel_stress_diff, altitude_shock_team,
                                       altitude_shock_opp, temp_stress
  - Socio-Economic Asymmetry        : log_gdp_diff, log_pop_diff
  - Categorical & Target Encoding   : venue_country_te (K-Fold Target Encoding),
                                       confederation_team_te, venue_country_freq
  - Tournament Importance           : tournament_weight

Metrik : Augmented Weighted Mean Absolute Error (AW-MAE)
Output : dataset/train_context_feat.csv  dan  dataset/test_context_feat.csv
         Kolom: Id + semua fitur baru berakhiran _ctx
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import KFold

# ===========================================================================
# 0. KONFIGURASI PATH
# ===========================================================================
BASE_DIR = Path(__file__).resolve().parent.parent   # naik dari src/ ke root project
DATA_DIR = BASE_DIR / "dataset"

TRAIN_PATH = DATA_DIR / "train_featured.csv"
TEST_PATH  = DATA_DIR / "test_featured.csv"

OUT_TRAIN = DATA_DIR / "train_context_feat.csv"
OUT_TEST  = DATA_DIR / "test_context_feat.csv"

# Nilai sentinel yang merepresentasikan missing value di dataset
MISSING_SENTINEL = -9999
MISSING_STR      = "Unknown"

# Suhu ideal bermain sepak bola (°C)
IDEAL_TEMP = 22.0

# Jumlah fold untuk K-Fold Target Encoding
N_FOLDS = 5

# Smoothing factor untuk Target Encoding (menghindari overfitting pada kategori jarang)
SMOOTH_ALPHA = 10.0

# Mapping bobot turnamen untuk AW-MAE
TOURNAMENT_WEIGHT_MAP = {
    "FIFA World Cup": 2.00,
    "AFC Asian Cup": 1.80,
    "UEFA Euro": 1.80,
    "Copa América": 1.80,
    "African Cup of Nations": 1.80,
    "FIFA World Cup qualification": 1.50,
    "UEFA Euro qualification": 1.50,
    "Friendly": 0.96,
}
DEFAULT_TOURNAMENT_WEIGHT = 1.20


# ===========================================================================
# 1. HELPER FUNCTIONS
# ===========================================================================

def replace_sentinel_with_nan(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Ganti nilai sentinel (-9999 dan 'Unknown') menjadi np.nan pada kolom yang ditentukan.
    Ini WAJIB dilakukan sebelum kalkulasi apapun.
    """
    for col in columns:
        if col not in df.columns:
            print(f"  [WARN] Kolom '{col}' tidak ditemukan — dilewati.")
            continue
        # Ganti string 'Unknown'
        df[col] = df[col].replace(MISSING_STR, np.nan)
        # Pastikan numerik, lalu ganti sentinel angka
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].replace(MISSING_SENTINEL, np.nan)
    return df


def impute_with_median(series: pd.Series, median_val: float) -> pd.Series:
    """Isi NaN dengan median yang diberikan."""
    return series.fillna(median_val)


# ===========================================================================
# 2. GEO-SPATIAL & PHYSICAL STRESS
# ===========================================================================

def build_geo_features(train: pd.DataFrame, test: pd.DataFrame,
                        combined: pd.DataFrame) -> tuple:
    """
    Membuat fitur geo-spasial dan stres fisik.

    Fitur yang dibuat:
      - travel_stress_diff_ctx   : selisih jarak tempuh (team - opp)
      - altitude_shock_team_ctx  : seberapa besar perubahan ketinggian bagi tim
      - altitude_shock_opp_ctx   : seberapa besar perubahan ketinggian bagi lawan
      - temp_stress_ctx          : penyimpangan suhu dari suhu ideal (22°C)
    """
    # Kolom yang perlu di-clean
    geo_cols = [
        "distance_travel_team", "distance_travel_opp",
        "temperature_venue", "altitude_venue"
    ]

    # Replace sentinel -> NaN pada SEMUA data (train + test)
    combined = replace_sentinel_with_nan(combined.copy(), geo_cols)
    train    = replace_sentinel_with_nan(train.copy(),    geo_cols)
    test     = replace_sentinel_with_nan(test.copy(),     geo_cols)

    # ---- Imputasi dengan median global (dari combined, agar train & test konsisten)
    medians = {}
    for col in geo_cols:
        if col in combined.columns:
            medians[col] = combined[col].median()
            print(f"  Median {col}: {medians[col]:.2f}")
        else:
            medians[col] = 0.0

    for col in geo_cols:
        if col in train.columns:
            train[col] = impute_with_median(train[col], medians[col])
        if col in test.columns:
            test[col]  = impute_with_median(test[col],  medians[col])

    # ---- (A) Travel Stress Diff ----
    train["travel_stress_diff_ctx"] = (
        train["distance_travel_team"] - train["distance_travel_opp"]
    )
    test["travel_stress_diff_ctx"] = (
        test["distance_travel_team"] - test["distance_travel_opp"]
    )

    # ---- (B) Home Altitude Proxy per Tim ----
    # Gabungkan combined (sudah di-clean) untuk menghitung proxy
    combined_clean = replace_sentinel_with_nan(combined.copy(), ["altitude_venue"])
    combined_clean["altitude_venue"] = impute_with_median(
        combined_clean["altitude_venue"], medians["altitude_venue"]
    )

    # Estimasi ketinggian "rumah" setiap tim = median altitude_venue saat is_home == 1
    home_rows = combined_clean[combined_clean["is_home"] == 1]

    # Proxy untuk tim (perspektif 'team')
    team_alt_proxy = (
        home_rows.groupby("team")["altitude_venue"]
        .median()
        .rename("team_home_alt_proxy")
        .reset_index()
    )
    # Proxy untuk lawan (perspektif 'opponent')
    opp_alt_proxy = (
        home_rows.groupby("team")["altitude_venue"]
        .median()
        .rename("opp_home_alt_proxy")
        .reset_index()
        .rename(columns={"team": "opponent"})
    )

    global_alt_median = combined_clean["altitude_venue"].median()

    train = _apply_altitude_shock(train, team_alt_proxy, opp_alt_proxy, global_alt_median)
    test  = _apply_altitude_shock(test,  team_alt_proxy, opp_alt_proxy, global_alt_median)

    # ---- (C) Temperature Stress ----
    train["temp_stress_ctx"] = abs(train["temperature_venue"] - IDEAL_TEMP)
    test["temp_stress_ctx"]  = abs(test["temperature_venue"]  - IDEAL_TEMP)

    return train, test


def _apply_altitude_shock(df: pd.DataFrame,
                           team_alt_proxy: pd.DataFrame,
                           opp_alt_proxy: pd.DataFrame,
                           global_median: float) -> pd.DataFrame:
    """Helper terpisah agar altitude shock bisa diterapkan dengan merge yang benar."""
    df = df.merge(team_alt_proxy, on="team", how="left")
    df["team_home_alt_proxy"] = df["team_home_alt_proxy"].fillna(global_median)

    df = df.merge(opp_alt_proxy, on="opponent", how="left")
    df["opp_home_alt_proxy"] = df["opp_home_alt_proxy"].fillna(global_median)

    df["altitude_shock_team_ctx"] = abs(df["altitude_venue"] - df["team_home_alt_proxy"])
    df["altitude_shock_opp_ctx"]  = abs(df["altitude_venue"] - df["opp_home_alt_proxy"])

    df.drop(columns=["team_home_alt_proxy", "opp_home_alt_proxy"], inplace=True)
    return df


# ===========================================================================
# 3. SOCIO-ECONOMIC ASYMMETRY
# ===========================================================================

def build_socio_features(train: pd.DataFrame, test: pd.DataFrame,
                          combined: pd.DataFrame) -> tuple:
    """
    Membuat fitur asimetri sosio-ekonomi.

    Fitur yang dibuat:
      - log_gdp_diff_ctx : log(GDP_team) - log(GDP_opp)
      - log_pop_diff_ctx : log(Pop_team) - log(Pop_opp)
    """
    socio_cols = [
        "gdp_per_capita_team", "gdp_per_capita_opp",
        "population_team", "population_opp"
    ]

    combined = replace_sentinel_with_nan(combined.copy(), socio_cols)
    train    = replace_sentinel_with_nan(train.copy(),    socio_cols)
    test     = replace_sentinel_with_nan(test.copy(),     socio_cols)

    # Hitung median dari combined untuk konsistensi
    medians = {}
    for col in socio_cols:
        if col in combined.columns:
            medians[col] = combined[col].median()
            print(f"  Median {col}: {medians[col]:.2f}")
        else:
            medians[col] = 1.0  # fallback agar log tidak error

    for col in socio_cols:
        med = medians.get(col, 1.0)
        if col in train.columns:
            train[col] = impute_with_median(train[col], med)
        if col in test.columns:
            test[col]  = impute_with_median(test[col],  med)

    # Log GDP Diff
    train["log_gdp_diff_ctx"] = (
        np.log1p(train["gdp_per_capita_team"]) - np.log1p(train["gdp_per_capita_opp"])
    )
    test["log_gdp_diff_ctx"] = (
        np.log1p(test["gdp_per_capita_team"]) - np.log1p(test["gdp_per_capita_opp"])
    )

    # Log Population Diff
    train["log_pop_diff_ctx"] = (
        np.log1p(train["population_team"]) - np.log1p(train["population_opp"])
    )
    test["log_pop_diff_ctx"] = (
        np.log1p(test["population_team"]) - np.log1p(test["population_opp"])
    )

    return train, test


# ===========================================================================
# 4. CATEGORICAL & TARGET ENCODING
# ===========================================================================

def smooth_target_encode(series: pd.Series, target: pd.Series,
                          global_mean: float, alpha: float = SMOOTH_ALPHA) -> pd.Series:
    """
    Target encoding dengan smoothing:
        encoded = (n * mean_cat + alpha * global_mean) / (n + alpha)

    Semakin sedikit sampel per kategori, semakin encoded mendekati global_mean.
    """
    stats = pd.DataFrame({"cat": series, "target": target})
    agg = stats.groupby("cat")["target"].agg(["mean", "count"])
    agg["smoothed"] = (
        (agg["count"] * agg["mean"] + alpha * global_mean) / (agg["count"] + alpha)
    )
    return series.map(agg["smoothed"])


def build_encoding_features(train: pd.DataFrame, test: pd.DataFrame) -> tuple:
    """
    Membuat fitur encoding:
      - venue_country_te_ctx        : K-Fold Target Encoding venue_country vs goal_diff
      - confederation_team_te_ctx   : K-Fold Target Encoding confederation_team vs goal_diff
      - venue_country_freq_ctx      : Frequency Encoding venue_country

    Untuk test: menggunakan rata-rata agregat dari train (+ smoothing).
    Target: team_goals - opp_goals (Goal Difference)
    """
    train = train.copy()
    test  = test.copy()

    # Pastikan kolom kategori ada; ganti 'Unknown' dengan NaN lalu 'UNKNOWN' sebagai fallback
    cat_cols = ["venue_country", "confederation_team", "confederation_opp"]
    for col in cat_cols:
        for df_part in [train, test]:
            if col in df_part.columns:
                df_part[col] = df_part[col].replace(MISSING_STR, "UNKNOWN")

    # Target: Goal Difference
    train["_goal_diff_temp"] = train["team_goals"] - train["opp_goals"]
    global_mean = train["_goal_diff_temp"].mean()

    # ---- (A) K-Fold Target Encoding: venue_country ----
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    train["venue_country_te_ctx"] = np.nan

    if "venue_country" in train.columns:
        for fold_idx, (trn_idx, val_idx) in enumerate(kf.split(train)):
            trn_fold = train.iloc[trn_idx]
            val_fold = train.iloc[val_idx]

            encoded = smooth_target_encode(
                trn_fold["venue_country"],
                trn_fold["_goal_diff_temp"],
                global_mean
            )
            mapping = dict(zip(trn_fold["venue_country"], encoded))
            train.loc[train.index[val_idx], "venue_country_te_ctx"] = (
                val_fold["venue_country"].map(mapping).values
            )

        # Isi NaN yang tersisa (kategori yang tidak muncul di fold training)
        train["venue_country_te_ctx"] = train["venue_country_te_ctx"].fillna(global_mean)

        # Mapping untuk test: agregat global dari seluruh train
        vc_map = smooth_target_encode(
            train["venue_country"], train["_goal_diff_temp"], global_mean
        )
        vc_global_map = dict(zip(train["venue_country"], vc_map))

        if "venue_country" in test.columns:
            test["venue_country_te_ctx"] = (
                test["venue_country"].map(vc_global_map).fillna(global_mean)
            )
        else:
            test["venue_country_te_ctx"] = global_mean
    else:
        train["venue_country_te_ctx"] = global_mean
        test["venue_country_te_ctx"]  = global_mean

    # ---- (B) K-Fold Target Encoding: confederation_team ----
    train["confederation_team_te_ctx"] = np.nan

    if "confederation_team" in train.columns:
        for fold_idx, (trn_idx, val_idx) in enumerate(kf.split(train)):
            trn_fold = train.iloc[trn_idx]
            val_fold = train.iloc[val_idx]

            encoded = smooth_target_encode(
                trn_fold["confederation_team"],
                trn_fold["_goal_diff_temp"],
                global_mean
            )
            mapping = dict(zip(trn_fold["confederation_team"], encoded))
            train.loc[train.index[val_idx], "confederation_team_te_ctx"] = (
                val_fold["confederation_team"].map(mapping).values
            )

        train["confederation_team_te_ctx"] = (
            train["confederation_team_te_ctx"].fillna(global_mean)
        )

        conf_map = smooth_target_encode(
            train["confederation_team"], train["_goal_diff_temp"], global_mean
        )
        conf_global_map = dict(zip(train["confederation_team"], conf_map))

        if "confederation_team" in test.columns:
            test["confederation_team_te_ctx"] = (
                test["confederation_team"].map(conf_global_map).fillna(global_mean)
            )
        else:
            test["confederation_team_te_ctx"] = global_mean
    else:
        train["confederation_team_te_ctx"] = global_mean
        test["confederation_team_te_ctx"]  = global_mean

    # ---- (C) Frequency Encoding: venue_country ----
    if "venue_country" in train.columns:
        vc_freq = train["venue_country"].value_counts(normalize=True)
        train["venue_country_freq_ctx"] = train["venue_country"].map(vc_freq).fillna(0.0)
        if "venue_country" in test.columns:
            test["venue_country_freq_ctx"] = test["venue_country"].map(vc_freq).fillna(0.0)
        else:
            test["venue_country_freq_ctx"] = 0.0
    else:
        train["venue_country_freq_ctx"] = 0.0
        test["venue_country_freq_ctx"]  = 0.0

    # Bersihkan kolom sementara
    train.drop(columns=["_goal_diff_temp"], inplace=True)

    return train, test


# ===========================================================================
# 5. TOURNAMENT IMPORTANCE
# ===========================================================================


# ===========================================================================
# 6. FINALISASI — Hanya simpan Id + kolom _ctx
# ===========================================================================

def finalize_and_save(train: pd.DataFrame, test: pd.DataFrame):
    """
    Pilih hanya kolom 'Id' dan kolom yang berakhiran '_ctx', lalu simpan.
    Validasi: jumlah baris output harus sama dengan input.
    """
    ctx_cols = [c for c in train.columns if c.endswith("_ctx")]

    if not ctx_cols:
        raise ValueError("Tidak ada kolom _ctx yang ditemukan! Periksa pipeline.")

    print(f"\n  Kolom _ctx yang akan disimpan ({len(ctx_cols)}):")
    for c in ctx_cols:
        nn_train = train[c].notna().sum()
        nn_test  = test[c].notna().sum()
        print(f"    {c:40s} | train non-null: {nn_train}/{len(train)} | test non-null: {nn_test}/{len(test)}")

    # Output train
    train_out = train[["Id"] + ctx_cols].copy()
    test_out  = test[["Id"] + ctx_cols].copy()

    train_out.to_csv(OUT_TRAIN, index=False)
    test_out.to_csv(OUT_TEST,   index=False)

    print(f"\n  [OK] train_context_feat.csv -> {len(train_out)} baris, {len(train_out.columns)} kolom")
    print(f"  [OK] test_context_feat.csv  -> {len(test_out)} baris, {len(test_out.columns)} kolom")


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    print("=" * 65)
    print("CONTEXTUAL & SOCIO-ECONOMIC FEATURE ENGINEERING PIPELINE")
    print("Anggota 2 | AW-MAE Optimization")
    print("=" * 65)

    # ---- Load Data ----
    print("\n[1/6] Loading data...")
    train = pd.read_csv(TRAIN_PATH)
    test  = pd.read_csv(TEST_PATH)

    n_train_orig = len(train)
    n_test_orig  = len(test)

    print(f"  train_featured.csv : {n_train_orig} baris, {len(train.columns)} kolom")
    print(f"  test_featured.csv  : {n_test_orig} baris, {len(test.columns)} kolom")

    # Validasi kolom Id ada
    for name, df_part in [("train", train), ("test", test)]:
        if "Id" not in df_part.columns:
            raise ValueError(f"Kolom 'Id' tidak ditemukan di {name}_featured.csv!")

    # Gabungkan untuk keperluan proxy global (hanya baca, tidak dimodifikasi output)
    # Test tidak punya target, tambahkan NaN agar concat rapi
    test_aug = test.copy()
    for col in ["team_goals", "opp_goals"]:
        if col not in test_aug.columns:
            test_aug[col] = np.nan
    for col in train.columns:
        if col not in test_aug.columns:
            test_aug[col] = np.nan

    combined = pd.concat([train, test_aug], ignore_index=True)

    # ---- Geo-Spatial Features ----
    print("\n[2/6] Building Geo-Spatial & Physical Stress features...")
    train, test = build_geo_features(train, test, combined)

    # ---- Socio-Economic Features ----
    print("\n[3/6] Building Socio-Economic Asymmetry features...")
    train, test = build_socio_features(train, test, combined)

    # ---- Encoding Features ----
    print("\n[4/6] Building Categorical & Target Encoding features...")
    train, test = build_encoding_features(train, test)


    # ---- Sanity check jumlah baris ----
    assert len(train) == n_train_orig, (
        f"Jumlah baris train berubah: {n_train_orig} -> {len(train)}"
    )
    assert len(test) == n_test_orig, (
        f"Jumlah baris test berubah: {n_test_orig} -> {len(test)}"
    )

    # ---- Simpan Output ----
    print("\n[6/6] Finalizing & saving output files...")
    finalize_and_save(train, test)

    print("\n" + "=" * 65)
    print("PIPELINE SELESAI")
    print(f"Output: {OUT_TRAIN}")
    print(f"        {OUT_TEST}")
    print("=" * 65)


if __name__ == "__main__":
    main()