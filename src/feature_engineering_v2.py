"""
Feature Engineering Pipeline V2 -- Gammafest Masa Kite Lagi
============================================================
Peran  : Anggota 1 (Core Historical Engineer)
Delta dari V1:
  1. Home Advantage di Elo  (+100 poin bayangan ke rumus ekspektasi)
  2. EWMA rolling stats     (bobot eksponensial berdasarkan jarak hari)
  3. Confederation K-Factor  (pengali kasta benua)
  4. Karantina kolom output  (hanya Id + fitur *_feat)

Output : dataset/train_core_v2.csv  dan  dataset/test_core_v2.csv
"""

import pandas as pd
import numpy as np
from collections import defaultdict, deque
from pathlib import Path
import math

# ===========================================================================
# 0. KONFIGURASI
# ===========================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "dataset"

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH  = DATA_DIR / "test.csv"

OUT_TRAIN = DATA_DIR / "train_core_v2.csv"
OUT_TEST  = DATA_DIR / "test_core_v2.csv"

# ===========================================================================
# 1. LOAD DATA & GABUNGKAN SECARA KRONOLOGIS
# ===========================================================================
def load_and_merge() -> pd.DataFrame:
    train = pd.read_csv(TRAIN_PATH)
    test  = pd.read_csv(TEST_PATH)

    train["is_test"] = False
    test["is_test"]  = True

    for col in ["team_goals", "opp_goals"]:
        if col not in test.columns:
            test[col] = np.nan
    for col in train.columns:
        if col not in test.columns:
            test[col] = np.nan

    df = pd.concat([train, test], ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "match_id", "Id"]).reset_index(drop=True)
    return df


# ===========================================================================
# 2. ELO RATING V2
#    - Home Advantage: +100 poin bayangan di rumus ekspektasi
#    - Confederation K-Factor multiplier
# ===========================================================================
ELO_INIT = 1500
ELO_K    = 32
ELO_HOME_ADVANTAGE = 100  # [BARU] Poin bayangan untuk tim kandang

# K-factor per turnamen (basis)
TOURNAMENT_K_WEIGHT = {
    "FIFA World Cup": 60,
    "FIFA World Cup qualification": 40,
    "Confederations Cup": 50,
    "Copa America": 50,
    "UEFA Euro": 50,
    "UEFA Euro qualification": 40,
    "African Cup of Nations": 50,
    "African Cup of Nations qualification": 40,
    "AFC Asian Cup": 50,
    "AFC Asian Cup qualification": 40,
    "Gold Cup": 45,
    "CONCACAF Championship": 45,
    "CONCACAF Gold Cup qualification": 40,
    "CONCACAF Nations League": 40,
    "UEFA Nations League": 45,
    "Oceania Nations Cup": 40,
    "Friendly": 20,
    "Olympic Games": 40,
    "Confederations Cup": 50,
    "Finalissima": 50,
    "CONMEBOL Nations League": 40,
}

# [BARU] Pengali K-factor berdasarkan konfederasi pertandingan
# Logika: pertandingan antar tim dari konfederasi kuat (UEFA, CONMEBOL)
# menghasilkan perpindahan Elo yang lebih bermakna;
# pertandingan di konfederasi lemah (OFC) sedikit diredam.
CONFEDERATION_K_MULTIPLIER = {
    "UEFA":     1.00,  # baseline
    "CONMEBOL": 1.00,
    "CAF":      0.95,
    "AFC":      0.90,
    "CONCACAF": 0.90,
    "OFC":      0.80,
    "Unknown":  0.85,
}


def get_k_factor(tournament: str, conf_a: str, conf_b: str) -> float:
    """
    K-factor = basis turnamen * rata-rata multiplier konfederasi kedua tim.
    Jika kedua tim dari konfederasi kuat, K tetap tinggi.
    Jika salah satu dari OFC, K sedikit turun.
    """
    base_k = TOURNAMENT_K_WEIGHT.get(tournament, ELO_K)
    mult_a = CONFEDERATION_K_MULTIPLIER.get(conf_a, 0.85)
    mult_b = CONFEDERATION_K_MULTIPLIER.get(conf_b, 0.85)
    conf_mult = (mult_a + mult_b) / 2.0
    return base_k * conf_mult


def calc_elo_change(elo_a: float, elo_b: float, score_a: float,
                    k: float, home_adv_a: float = 0.0) -> float:
    """
    Hitung perubahan Elo untuk tim A.
    home_adv_a: bonus bayangan untuk A (biasanya +100 jika kandang, 0 jika tidak).
    Bonus ini HANYA masuk ke rumus ekspektasi, TIDAK mengubah Elo permanen.
    """
    effective_elo_a = elo_a + home_adv_a
    expected = 1.0 / (1.0 + 10 ** ((elo_b - effective_elo_a) / 400.0))
    return k * (score_a - expected)


def compute_elo(df: pd.DataFrame) -> pd.DataFrame:
    elo = defaultdict(lambda: ELO_INIT)

    n = len(df)
    elo_team_vals = np.full(n, np.nan)
    elo_opp_vals  = np.full(n, np.nan)

    match_groups = df.groupby("match_id", sort=False)

    for match_id, group in match_groups:
        if len(group) != 2:
            continue

        idx = group.index.tolist()
        row_a = df.loc[idx[0]]
        row_b = df.loc[idx[1]]

        gender = row_a["gender"]
        team_a, team_b = row_a["team"], row_b["team"]
        tournament = row_a["tournament"]
        conf_a = str(row_a.get("confederation_team", "Unknown"))
        conf_b = str(row_b.get("confederation_team", "Unknown"))

        key_a = (team_a, gender)
        key_b = (team_b, gender)

        cur_elo_a = elo[key_a]
        cur_elo_b = elo[key_b]

        # Simpan Elo SEBELUM pertandingan (nilai permanen, tanpa bonus)
        elo_team_vals[idx[0]] = cur_elo_a
        elo_opp_vals[idx[0]]  = cur_elo_b
        elo_team_vals[idx[1]] = cur_elo_b
        elo_opp_vals[idx[1]]  = cur_elo_a

        # Update hanya jika skor tersedia
        goals_a = row_a["team_goals"]
        goals_b = row_b["team_goals"]

        if pd.notna(goals_a) and pd.notna(goals_b):
            if goals_a > goals_b:
                score_a = 1.0
            elif goals_a < goals_b:
                score_a = 0.0
            else:
                score_a = 0.5

            k = get_k_factor(tournament, conf_a, conf_b)

            # Goal difference adjustment
            gd = abs(goals_a - goals_b)
            if gd == 2:
                k *= 1.5
            elif gd == 3:
                k *= 1.75
            elif gd > 3:
                k *= (1.75 + (gd - 3) / 8.0)

            # [BARU] Home Advantage: tentukan siapa yang main di kandang
            # Bonus masuk ke rumus ekspektasi saja, bukan ke Elo permanen
            is_home_a = row_a.get("is_home", 0)
            is_home_b = row_b.get("is_home", 0)
            neutral   = row_a.get("neutral", 0)

            if neutral == 1:
                home_adv_a = 0.0
            elif is_home_a == 1:
                home_adv_a = ELO_HOME_ADVANTAGE
            elif is_home_b == 1:
                home_adv_a = -ELO_HOME_ADVANTAGE
            else:
                home_adv_a = 0.0

            delta = calc_elo_change(cur_elo_a, cur_elo_b, score_a, k,
                                    home_adv_a)
            elo[key_a] = cur_elo_a + delta
            elo[key_b] = cur_elo_b - delta

    df["elo_team_calc"]     = elo_team_vals
    df["elo_opponent_calc"] = elo_opp_vals

    return df


# ===========================================================================
# 3. EWMA ROLLING STATS
#    Bobot setiap laga lama di-decay secara eksponensial berdasarkan
#    jarak hari dari pertandingan saat ini.
#    w_i = exp(-alpha * days_gap_i)
#    alpha = ln(2) / half_life_days   (half-life = 90 hari)
# ===========================================================================
EWMA_HALF_LIFE_DAYS = 90
EWMA_ALPHA = math.log(2) / EWMA_HALF_LIFE_DAYS
ROLLING_WINDOW = 10  # simpan 10 laga terakhir

def _ewma_aggregate(history_list, current_date, last_n=None):
    """
    Hitung agregat EWMA dari daftar history.
    history_list: list of (points, gf, ga, match_date)
    current_date: tanggal pertandingan saat ini
    last_n: jika diberikan, ambil hanya N entri terakhir

    Returns dict:
      pts_ewma, gd_ewma, avg_gf_ewma, avg_ga_ewma,
      win_rate_ewma, pts_simple (untuk backward compat)
    """
    items = list(history_list)
    if last_n is not None:
        items = items[-last_n:]

    if len(items) == 0:
        return None

    weights = []
    for (pts, gf, ga, d) in items:
        days_gap = (current_date - d).days
        if days_gap < 0:
            days_gap = 0
        w = math.exp(-EWMA_ALPHA * days_gap)
        weights.append(w)

    total_w = sum(weights)
    if total_w == 0:
        total_w = 1e-9  # safety

    pts_ewma   = sum(w * x[0] for w, x in zip(weights, items)) / total_w
    gf_ewma    = sum(w * x[1] for w, x in zip(weights, items)) / total_w
    ga_ewma    = sum(w * x[2] for w, x in zip(weights, items)) / total_w
    gd_ewma    = sum(w * (x[1] - x[2]) for w, x in zip(weights, items)) / total_w
    wins_ewma  = sum(w * (1.0 if x[0] == 3 else 0.0)
                     for w, x in zip(weights, items)) / total_w

    # Simple (non-weighted) untuk backward compat / perbandingan
    pts_simple = sum(x[0] for x in items)

    return {
        "pts_ewma":    pts_ewma,
        "gd_ewma":     gd_ewma,
        "avg_gf_ewma": gf_ewma,
        "avg_ga_ewma": ga_ewma,
        "win_rate_ewma": wins_ewma,
        "pts_simple":  pts_simple,
    }


def compute_rolling_stats(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    col_names = [
        "pts_last5_ewma", "pts_last10_ewma",
        "gd_last5_ewma",  "gd_last10_ewma",
        "avg_gf_last5_ewma", "avg_ga_last5_ewma",
        "win_rate_last10_ewma",
        "days_since_last_calc",
        "pts_last5_simple", "pts_last10_simple",
    ]
    cols_out = {c: np.full(n, np.nan) for c in col_names}

    history = defaultdict(lambda: deque(maxlen=ROLLING_WINDOW))
    last_date = {}

    for i in range(n):
        row = df.iloc[i]
        team   = row["team"]
        gender = row["gender"]
        key    = (team, gender)
        date_i = row["date"]
        gf, ga = row["team_goals"], row["opp_goals"]

        hist = history[key]

        # ---- Fitur SEBELUM pertandingan ini ----
        if len(hist) > 0:
            agg5  = _ewma_aggregate(hist, date_i, last_n=5)
            agg10 = _ewma_aggregate(hist, date_i, last_n=10)

            if agg5:
                cols_out["pts_last5_ewma"][i]    = agg5["pts_ewma"]
                cols_out["gd_last5_ewma"][i]     = agg5["gd_ewma"]
                cols_out["avg_gf_last5_ewma"][i] = agg5["avg_gf_ewma"]
                cols_out["avg_ga_last5_ewma"][i] = agg5["avg_ga_ewma"]
                cols_out["pts_last5_simple"][i]   = agg5["pts_simple"]

            if agg10:
                cols_out["pts_last10_ewma"][i]      = agg10["pts_ewma"]
                cols_out["gd_last10_ewma"][i]        = agg10["gd_ewma"]
                cols_out["win_rate_last10_ewma"][i]  = agg10["win_rate_ewma"]
                cols_out["pts_last10_simple"][i]     = agg10["pts_simple"]

        # Days since last match
        if key in last_date:
            cols_out["days_since_last_calc"][i] = (date_i - last_date[key]).days

        # ---- Update history ----
        if pd.notna(gf) and pd.notna(ga):
            if gf > ga:
                pts = 3
            elif gf == ga:
                pts = 1
            else:
                pts = 0
            hist.append((pts, gf, ga, date_i))

        last_date[key] = date_i

    for c, arr in cols_out.items():
        df[c] = arr

    return df


# ===========================================================================
# 4. HEAD-TO-HEAD EWMA
# ===========================================================================
def compute_h2h(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    h2h_pts = np.full(n, np.nan)
    h2h_gd  = np.full(n, np.nan)
    h2h_pts_ewma = np.full(n, np.nan)
    h2h_gd_ewma  = np.full(n, np.nan)

    h2h_history = defaultdict(lambda: deque(maxlen=5))

    for i in range(n):
        row = df.iloc[i]
        team, opponent, gender = row["team"], row["opponent"], row["gender"]
        key = (team, opponent, gender)
        date_i = row["date"]

        hist = h2h_history[key]

        if len(hist) > 0:
            # Simple
            h2h_pts[i] = sum(x[0] for x in hist)
            h2h_gd[i]  = sum(x[1] - x[2] for x in hist)

            # EWMA
            agg = _ewma_aggregate(hist, date_i)
            if agg:
                h2h_pts_ewma[i] = agg["pts_ewma"]
                h2h_gd_ewma[i]  = agg["gd_ewma"]

        gf, ga = row["team_goals"], row["opp_goals"]
        if pd.notna(gf) and pd.notna(ga):
            if gf > ga:
                p = 3
            elif gf == ga:
                p = 1
            else:
                p = 0
            hist.append((p, gf, ga, date_i))

    df["h2h_pts_last5_simple"] = h2h_pts
    df["h2h_gd_last5_simple"]  = h2h_gd
    df["h2h_pts_last5_ewma"]   = h2h_pts_ewma
    df["h2h_gd_last5_ewma"]    = h2h_gd_ewma

    return df


# ===========================================================================
# 5. MIRROR OPPONENT FEATURES
# ===========================================================================
def mirror_opponent_features(df: pd.DataFrame) -> pd.DataFrame:
    team_rolling_cols = [
        "pts_last5_ewma", "pts_last10_ewma",
        "gd_last5_ewma", "gd_last10_ewma",
        "avg_gf_last5_ewma", "avg_ga_last5_ewma",
        "win_rate_last10_ewma",
        "days_since_last_calc",
        "pts_last5_simple", "pts_last10_simple",
    ]

    opp_col_names = ["opp_" + c for c in team_rolling_cols]
    for oc in opp_col_names:
        df[oc] = np.nan

    match_groups = df.groupby("match_id", sort=False)
    for match_id, group in match_groups:
        if len(group) != 2:
            continue
        idx = group.index.tolist()
        for col, opp_col in zip(team_rolling_cols, opp_col_names):
            df.at[idx[0], opp_col] = df.at[idx[1], col]
            df.at[idx[1], opp_col] = df.at[idx[0], col]

    return df


# ===========================================================================
# 6. DERIVED FEATURES
# ===========================================================================
def compute_derived(df: pd.DataFrame) -> pd.DataFrame:
    # Elo diff
    df["elo_diff_calc"] = df["elo_team_calc"] - df["elo_opponent_calc"]

    # EWMA diff (team - opp)
    df["pts_last5_ewma_diff"]  = df["pts_last5_ewma"]  - df["opp_pts_last5_ewma"]
    df["pts_last10_ewma_diff"] = df["pts_last10_ewma"] - df["opp_pts_last10_ewma"]
    df["gd_last5_ewma_diff"]   = df["gd_last5_ewma"]   - df["opp_gd_last5_ewma"]

    # Form index v2: menggabungkan pts + gd (EWMA)
    df["form_index_team"] = (
        df["pts_last5_ewma"].fillna(0) + df["gd_last5_ewma"].fillna(0) * 0.5
    )
    df["form_index_opp"] = (
        df["opp_pts_last5_ewma"].fillna(0) + df["opp_gd_last5_ewma"].fillna(0) * 0.5
    )
    df["form_index_diff"] = df["form_index_team"] - df["form_index_opp"]

    # Simple diff (backward compat)
    df["pts_last5_simple_diff"]  = df["pts_last5_simple"] - df["opp_pts_last5_simple"]
    df["pts_last10_simple_diff"] = df["pts_last10_simple"] - df["opp_pts_last10_simple"]

    return df


# ===========================================================================
# 7. FINALISASI -- Karantina: hanya Id + *_feat
# ===========================================================================
def finalize_and_save(df: pd.DataFrame):
    # Rename semua kolom kalkulasi kita menjadi bersufiks _feat
    rename_map = {
        # Elo
        "elo_team_calc":            "elo_team_feat",
        "elo_opponent_calc":        "elo_opponent_feat",
        "elo_diff_calc":            "elo_diff_feat",
        # Rolling EWMA -- team
        "pts_last5_ewma":           "team_pts_last5_ewma_feat",
        "pts_last10_ewma":          "team_pts_last10_ewma_feat",
        "gd_last5_ewma":            "team_gd_last5_ewma_feat",
        "gd_last10_ewma":           "team_gd_last10_ewma_feat",
        "avg_gf_last5_ewma":        "team_avg_gf_last5_ewma_feat",
        "avg_ga_last5_ewma":        "team_avg_ga_last5_ewma_feat",
        "win_rate_last10_ewma":     "team_win_rate_last10_ewma_feat",
        "days_since_last_calc":     "days_since_last_team_feat",
        "pts_last5_simple":         "team_pts_last5_simple_feat",
        "pts_last10_simple":        "team_pts_last10_simple_feat",
        # Rolling EWMA -- opponent (mirrored)
        "opp_pts_last5_ewma":       "opp_pts_last5_ewma_feat",
        "opp_pts_last10_ewma":      "opp_pts_last10_ewma_feat",
        "opp_gd_last5_ewma":        "opp_gd_last5_ewma_feat",
        "opp_gd_last10_ewma":       "opp_gd_last10_ewma_feat",
        "opp_avg_gf_last5_ewma":    "opp_avg_gf_last5_ewma_feat",
        "opp_avg_ga_last5_ewma":    "opp_avg_ga_last5_ewma_feat",
        "opp_win_rate_last10_ewma": "opp_win_rate_last10_ewma_feat",
        "opp_days_since_last_calc": "days_since_last_opp_feat",
        "opp_pts_last5_simple":     "opp_pts_last5_simple_feat",
        "opp_pts_last10_simple":    "opp_pts_last10_simple_feat",
        # H2H
        "h2h_pts_last5_simple":     "h2h_pts_last5_simple_feat",
        "h2h_gd_last5_simple":      "h2h_gd_last5_simple_feat",
        "h2h_pts_last5_ewma":       "h2h_pts_last5_ewma_feat",
        "h2h_gd_last5_ewma":        "h2h_gd_last5_ewma_feat",
        # Derived
        "pts_last5_ewma_diff":      "pts_last5_ewma_diff_feat",
        "pts_last10_ewma_diff":     "pts_last10_ewma_diff_feat",
        "gd_last5_ewma_diff":       "gd_last5_ewma_diff_feat",
        "form_index_team":          "form_team_feat",
        "form_index_opp":           "form_opp_feat",
        "form_index_diff":          "form_diff_feat",
        "pts_last5_simple_diff":    "pts_last5_simple_diff_feat",
        "pts_last10_simple_diff":   "pts_last10_simple_diff_feat",
    }

    df = df.rename(columns=rename_map)

    # ---- Karantina: hanya Id + kolom _feat + target (untuk train) ----
    feat_cols = sorted([c for c in df.columns if c.endswith("_feat")])
    keep_cols_train = ["Id"] + feat_cols + ["team_goals", "opp_goals"]
    keep_cols_test  = ["Id"] + feat_cols

    train_out = df[df["is_test"] == False][keep_cols_train].copy()
    test_out  = df[df["is_test"] == True][keep_cols_test].copy()

    train_out.to_csv(OUT_TRAIN, index=False)
    test_out.to_csv(OUT_TEST, index=False)

    print(f"[OK] train_core_v2.csv -> {len(train_out)} baris, {len(train_out.columns)} kolom")
    print(f"[OK] test_core_v2.csv  -> {len(test_out)} baris, {len(test_out.columns)} kolom")
    print(f"\nFitur ({len(feat_cols)} kolom):")
    for c in feat_cols:
        nn_train = train_out[c].notna().sum()
        nn_test  = test_out[c].notna().sum()
        print(f"  {c:45s} train:{nn_train:>6}/{len(train_out)}  test:{nn_test:>6}/{len(test_out)}")


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    print("=" * 60)
    print("FEATURE ENGINEERING V2 (Core Historical)")
    print("=" * 60)

    print("\n[1/7] Loading & merging data...")
    df = load_and_merge()
    print(f"      Total: {len(df)} baris | {df['date'].min()} -> {df['date'].max()}")

    print("\n[2/7] Computing Elo V2 (Home Adv + Confederation K)...")
    df = compute_elo(df)

    print("\n[3/7] Computing EWMA rolling stats...")
    df = compute_rolling_stats(df)

    print("\n[4/7] Computing H2H (simple + EWMA)...")
    df = compute_h2h(df)

    print("\n[5/7] Mirroring opponent features...")
    df = mirror_opponent_features(df)

    print("\n[6/7] Computing derived features...")
    df = compute_derived(df)

    print("\n[7/7] Finalizing & saving (quarantined output)...")
    finalize_and_save(df)

    print("\n" + "=" * 60)
    print("PIPELINE V2 SELESAI")
    print("=" * 60)


if __name__ == "__main__":
    main()
