"""
Feature Engineering Pipeline — Gammafest Masa Kite Lagi
=======================================================
Peran  : Anggota 1 (Feature Engineer)
Tujuan : Merekonstruksi semua fitur performa historis yang hilang di test.csv
         dengan menghitung ulang dari data gabungan (train + test) secara kronologis.

Fitur yang direkonstruksi:
  - Elo Rating             (elo_team, elo_opponent)
  - Poin 5 & 10 laga terakhir    (team_points_last5/10, opp_points_last5/10, dll.)
  - Goal Difference 5 laga       (team_gd_last5, opp_gd_last5, gd_last5_diff)
  - Rata-rata gol & kebobolan     (team_avg_goals_last5, team_avg_conceded_last5, dst.)
  - Win rate 10 laga terakhir     (team_win_rate_last10, opp_win_rate_last10)
  - Head-to-head 5 pertemuan      (h2h_points_last5, h2h_gd_last5)
  - Jeda hari antar laga          (days_since_last_match_team/opp)
  - Ranking relatif               (rank_team, rank_opponent, rank_diff)

Output : dataset/train_featured.csv  dan  dataset/test_featured.csv
         Kedua file ini siap digunakan oleh Anggota 2 (Modeling).
"""

import pandas as pd
import numpy as np
from collections import defaultdict, deque
from pathlib import Path

# ===========================================================================
# 0. KONFIGURASI PATH
# ===========================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "dataset"

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH  = DATA_DIR / "test.csv"

OUT_TRAIN = DATA_DIR / "train_featured.csv"
OUT_TEST  = DATA_DIR / "test_featured.csv"

# ===========================================================================
# 1. LOAD DATA & GABUNGKAN SECARA KRONOLOGIS
# ===========================================================================
def load_and_merge() -> pd.DataFrame:
    """
    Gabungkan train dan test menjadi satu DataFrame kronologis.
    Kolom target (team_goals, opp_goals) akan NaN untuk test rows.
    Tambahkan flag 'is_test' untuk memisahkan kembali nanti.
    """
    train = pd.read_csv(TRAIN_PATH)
    test  = pd.read_csv(TEST_PATH)

    train["is_test"] = False
    test["is_test"]  = True

    # Pastikan kolom target ada di test (diisi NaN)
    for col in ["team_goals", "opp_goals"]:
        if col not in test.columns:
            test[col] = np.nan

    # Pastikan semua kolom performa ada di test (diisi NaN) agar concat rapi
    for col in train.columns:
        if col not in test.columns:
            test[col] = np.nan

    df = pd.concat([train, test], ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "match_id", "Id"]).reset_index(drop=True)

    return df


# ===========================================================================
# 2. ELO RATING — Dihitung dari nol untuk seluruh history
# ===========================================================================
# Parameter Elo standar sepak bola (World Football Elo Ratings)
ELO_INIT = 1500
ELO_K    = 32   # K-factor dasar

# Bobot K-factor berdasarkan jenis turnamen (semakin prestisius → K lebih besar)
TOURNAMENT_K_WEIGHT = {
    "FIFA World Cup": 60,
    "FIFA World Cup qualification": 40,
    "Confederations Cup": 50,
    "Continental championship (UEFA)": 50,
    "Continental championship qualification (UEFA)": 40,
    "Copa América": 50,
    "AFC Asian Cup": 50,
    "African Cup of Nations": 50,
    "UEFA Euro": 50,
    "UEFA Euro qualification": 40,
    "AFC Asian Cup qualification": 40,
    "African Cup of Nations qualification": 40,
    "Gold Cup": 45,
    "CONCACAF Nations League": 40,
    "UEFA Nations League": 45,
    "Friendly": 20,
}

def get_k_factor(tournament: str) -> float:
    """Ambil K-factor berdasarkan turnamen. Default 32 jika tidak dikenali."""
    return TOURNAMENT_K_WEIGHT.get(tournament, ELO_K)


def calc_elo_change(elo_a: float, elo_b: float, score_a: float,
                    k: float) -> float:
    """
    Hitung perubahan Elo untuk tim A.
    score_a: 1.0 (menang), 0.5 (seri), 0.0 (kalah)
    """
    expected = 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))
    return k * (score_a - expected)


def compute_elo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Hitung Elo Rating secara kronologis untuk setiap tim.
    Setiap pertandingan (match_id) muncul 2 baris — kita proses sekali per match.

    Mengembalikan df dengan kolom 'elo_team_calc' dan 'elo_opponent_calc'.
    """
    elo = defaultdict(lambda: ELO_INIT)  # {(team, gender): elo}

    elo_team_vals = np.full(len(df), np.nan)
    elo_opp_vals  = np.full(len(df), np.nan)

    # Group per match — setiap match punya tepat 2 baris
    match_groups = df.groupby("match_id", sort=False)

    for match_id, group in match_groups:
        if len(group) != 2:
            continue  # skip data tidak valid

        idx = group.index.tolist()
        row_a = df.loc[idx[0]]
        row_b = df.loc[idx[1]]

        gender = row_a["gender"]
        team_a = row_a["team"]
        team_b = row_b["team"]
        tournament = row_a["tournament"]

        key_a = (team_a, gender)
        key_b = (team_b, gender)

        cur_elo_a = elo[key_a]
        cur_elo_b = elo[key_b]

        # Simpan Elo SEBELUM pertandingan
        elo_team_vals[idx[0]] = cur_elo_a
        elo_opp_vals[idx[0]]  = cur_elo_b  # lawan row_a = row_b's team
        elo_team_vals[idx[1]] = cur_elo_b
        elo_opp_vals[idx[1]]  = cur_elo_a

        # Update Elo hanya jika skor tersedia (train rows)
        goals_a = row_a["team_goals"]
        goals_b = row_b["team_goals"]

        if pd.notna(goals_a) and pd.notna(goals_b):
            if goals_a > goals_b:
                score_a = 1.0
            elif goals_a < goals_b:
                score_a = 0.0
            else:
                score_a = 0.5

            k = get_k_factor(tournament)

            # Sesuaikan K berdasarkan goal difference (standar World Football Elo)
            gd = abs(goals_a - goals_b)
            if gd == 2:
                k *= 1.5
            elif gd == 3:
                k *= 1.75
            elif gd > 3:
                k *= (1.75 + (gd - 3) / 8.0)

            delta = calc_elo_change(cur_elo_a, cur_elo_b, score_a, k)
            elo[key_a] = cur_elo_a + delta
            elo[key_b] = cur_elo_b - delta

    df["elo_team_calc"]     = elo_team_vals
    df["elo_opponent_calc"] = elo_opp_vals

    return df


# ===========================================================================
# 3. ROLLING STATS — Last N matches per tim
# ===========================================================================
def compute_rolling_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Untuk setiap baris, hitung statistik berdasarkan N pertandingan
    terakhir TIM TERSEBUT (bukan lawan).

    Fitur yang dihitung per tim (kemudian di-merge ke perspektif lawan):
      - points_last5, points_last10     : poin kumulatif (W=3, D=1, L=0)
      - gd_last5                        : goal difference
      - avg_goals_last5                 : rata-rata gol dicetak
      - avg_conceded_last5              : rata-rata gol kemasukan
      - win_rate_last10                 : persen kemenangan
      - days_since_last_match           : jeda hari
    """
    # ---- INISIALISASI KOLOM OUTPUT ----
    n = len(df)
    cols_out = {
        "points_last5_calc":       np.full(n, np.nan),
        "points_last10_calc":      np.full(n, np.nan),
        "gd_last5_calc":           np.full(n, np.nan),
        "avg_goals_last5_calc":    np.full(n, np.nan),
        "avg_conceded_last5_calc": np.full(n, np.nan),
        "win_rate_last10_calc":    np.full(n, np.nan),
        "days_since_last_calc":    np.full(n, np.nan),
    }

    # ---- STATE PER TIM ----
    # Menyimpan N pertandingan terakhir: deque of (points, gf, ga, date)
    # points: 3=win, 1=draw, 0=loss
    history = defaultdict(lambda: deque(maxlen=10))  # max 10 (superset of 5)
    last_date = {}  # {(team, gender): last match date}

    for i in range(n):
        row = df.iloc[i]
        team   = row["team"]
        gender = row["gender"]
        key    = (team, gender)

        date_i = row["date"]
        gf     = row["team_goals"]   # goals for
        ga     = row["opp_goals"]    # goals against

        hist = history[key]

        # ---- Hitung fitur SEBELUM memasukkan pertandingan ini ----
        if len(hist) > 0:
            # Last 5
            last5 = list(hist)[-5:]  # mengambil 5 terakhir
            pts5  = sum(x[0] for x in last5)
            gd5   = sum(x[1] - x[2] for x in last5)
            avg_g5  = np.mean([x[1] for x in last5])
            avg_c5  = np.mean([x[2] for x in last5])

            cols_out["points_last5_calc"][i]       = pts5
            cols_out["gd_last5_calc"][i]           = gd5
            cols_out["avg_goals_last5_calc"][i]    = avg_g5
            cols_out["avg_conceded_last5_calc"][i] = avg_c5

            # Last 10
            last10 = list(hist)[-10:]
            pts10  = sum(x[0] for x in last10)
            wins10 = sum(1 for x in last10 if x[0] == 3)
            wr10   = wins10 / len(last10)

            cols_out["points_last10_calc"][i]  = pts10
            cols_out["win_rate_last10_calc"][i] = wr10

        # Days since last match
        if key in last_date:
            delta_days = (date_i - last_date[key]).days
            cols_out["days_since_last_calc"][i] = delta_days

        # ---- Update history (hanya jika skor tersedia) ----
        if pd.notna(gf) and pd.notna(ga):
            if gf > ga:
                pts = 3
            elif gf == ga:
                pts = 1
            else:
                pts = 0
            hist.append((pts, gf, ga, date_i))

        last_date[key] = date_i

    for col_name, arr in cols_out.items():
        df[col_name] = arr

    return df


# ===========================================================================
# 4. HEAD-TO-HEAD — Performa di 5 pertemuan terakhir antar dua tim
# ===========================================================================
def compute_h2h(df: pd.DataFrame) -> pd.DataFrame:
    """
    Untuk setiap baris, hitung:
      - h2h_points_last5_calc : poin tim dalam 5 pertemuan terakhir vs lawan ini
      - h2h_gd_last5_calc     : goal difference dalam 5 pertemuan terakhir
    """
    n = len(df)
    h2h_pts = np.full(n, np.nan)
    h2h_gd  = np.full(n, np.nan)

    # Key: (team, opponent, gender) -> deque of (pts, gf, ga)
    # Kita simpan dari perspektif `team`
    h2h_history = defaultdict(lambda: deque(maxlen=5))

    for i in range(n):
        row = df.iloc[i]
        team     = row["team"]
        opponent = row["opponent"]
        gender   = row["gender"]
        key      = (team, opponent, gender)

        hist = h2h_history[key]

        # ---- Hitung fitur SEBELUM memasukkan pertandingan ini ----
        if len(hist) > 0:
            pts = sum(x[0] for x in hist)
            gd  = sum(x[1] - x[2] for x in hist)
            h2h_pts[i] = pts
            h2h_gd[i]  = gd

        # ---- Update ----
        gf = row["team_goals"]
        ga = row["opp_goals"]
        if pd.notna(gf) and pd.notna(ga):
            if gf > ga:
                p = 3
            elif gf == ga:
                p = 1
            else:
                p = 0
            hist.append((p, gf, ga))

    df["h2h_points_last5_calc"] = h2h_pts
    df["h2h_gd_last5_calc"]     = h2h_gd

    return df


# ===========================================================================
# 5. OPPONENT PERSPECTIVE — Mirror fitur rolling ke perspektif lawan
# ===========================================================================
def mirror_opponent_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Setiap pertandingan (match_id) muncul 2 baris:
      - Baris A: team=X, opponent=Y → rolling stats milik X
      - Baris B: team=Y, opponent=X → rolling stats milik Y

    Kita perlu mengisi kolom 'opp_*' di Baris A dengan rolling stats
    milik Y (yang ada di Baris B), dan sebaliknya.

    Kolom yang di-mirror:
      - points_last5_calc   → opp_points_last5_calc
      - points_last10_calc  → opp_points_last10_calc
      - gd_last5_calc       → opp_gd_last5_calc
      - avg_goals_last5_calc    → opp_avg_goals_last5_calc
      - avg_conceded_last5_calc → opp_avg_conceded_last5_calc
      - win_rate_last10_calc    → opp_win_rate_last10_calc
      - days_since_last_calc    → opp_days_since_last_calc
    """
    rolling_cols = [
        "points_last5_calc", "points_last10_calc", "gd_last5_calc",
        "avg_goals_last5_calc", "avg_conceded_last5_calc",
        "win_rate_last10_calc", "days_since_last_calc",
    ]

    # Buat kolom kosong untuk opponent
    opp_col_names = ["opp_" + c for c in rolling_cols]
    for oc in opp_col_names:
        df[oc] = np.nan

    # Proses per match_id
    match_groups = df.groupby("match_id", sort=False)
    for match_id, group in match_groups:
        if len(group) != 2:
            continue
        idx = group.index.tolist()
        for col, opp_col in zip(rolling_cols, opp_col_names):
            # Row A's opponent features = Row B's team features
            df.at[idx[0], opp_col] = df.at[idx[1], col]
            df.at[idx[1], opp_col] = df.at[idx[0], col]

    return df


# ===========================================================================
# 6. RANKING — Estimasi ranking sederhana berdasarkan Elo
# ===========================================================================
def compute_rank_from_elo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ranking FIFA asli tidak bisa direkonstruksi, tapi kita bisa membuat
    proxy rank dari Elo untuk setiap pertandingan.

    Pendekatan: Untuk setiap tanggal unik, urutkan semua tim berdasarkan
    Elo terkini → jadikan rank. Ini tidak sempurna tapi konsisten
    antara train dan test.

    Catatan: Karena mengurutkan per-tanggal sangat mahal, kita hitung
    rank diff saja: selisih Elo bisa menjadi proxy rank_diff yang
    lebih stabil.
    """
    df["elo_diff_calc"]  = df["elo_team_calc"] - df["elo_opponent_calc"]
    return df


# ===========================================================================
# 7. DERIVED FEATURES — Fitur turunan tambahan
# ===========================================================================
def compute_derived(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fitur gabungan / turunan yang mungkin berguna bagi model:
      - points_last5_diff_calc : selisih poin 5 laga terakhir (team - opp)
      - gd_last5_diff_calc     : selisih GD 5 laga terakhir
      - form_index             : (points_last5 + gd_last5 * 0.5) — indikator form
    """
    df["points_last5_diff_calc"] = (
        df["points_last5_calc"] - df["opp_points_last5_calc"]
    )
    df["gd_last5_diff_calc"] = (
        df["gd_last5_calc"] - df["opp_gd_last5_calc"]
    )

    # Form index: gabungan poin + gd untuk menangkap momentum
    df["form_index_team"] = (
        df["points_last5_calc"].fillna(0)
        + df["gd_last5_calc"].fillna(0) * 0.5
    )
    df["form_index_opp"] = (
        df["opp_points_last5_calc"].fillna(0)
        + df["opp_gd_last5_calc"].fillna(0) * 0.5
    )
    df["form_index_diff"] = df["form_index_team"] - df["form_index_opp"]

    return df


# ===========================================================================
# 8. FINALISASI — Pilih kolom final & pisahkan kembali train/test
# ===========================================================================
def finalize_and_save(df: pd.DataFrame):
    """
    Ganti kolom performa asli (dari train) dengan kolom kalkulasi kita (*_calc),
    lalu pisahkan kembali menjadi train_featured.csv dan test_featured.csv.

    Alasan mengganti: agar fitur yang digunakan model KONSISTEN antara
    train dan test (keduanya berasal dari perhitungan yang sama).
    """

    # Mapping: kolom asli → kolom kalkulasi kita
    rename_map = {
        "elo_team_calc":           "elo_team_feat",
        "elo_opponent_calc":       "elo_opponent_feat",
        "points_last5_calc":       "team_points_last5_feat",
        "opp_points_last5_calc":   "opp_points_last5_feat",
        "points_last5_diff_calc":  "points_last5_diff_feat",
        "points_last10_calc":      "team_points_last10_feat",
        "opp_points_last10_calc":  "opp_points_last10_feat",
        "gd_last5_calc":           "team_gd_last5_feat",
        "opp_gd_last5_calc":       "opp_gd_last5_feat",
        "gd_last5_diff_calc":      "gd_last5_diff_feat",
        "avg_goals_last5_calc":    "team_avg_goals_last5_feat",
        "avg_conceded_last5_calc": "team_avg_conceded_last5_feat",
        "opp_avg_goals_last5_calc":    "opp_avg_goals_last5_feat",
        "opp_avg_conceded_last5_calc": "opp_avg_conceded_last5_feat",
        "win_rate_last10_calc":      "team_win_rate_last10_feat",
        "opp_win_rate_last10_calc":  "opp_win_rate_last10_feat",
        "h2h_points_last5_calc":     "h2h_points_last5_feat",
        "h2h_gd_last5_calc":         "h2h_gd_last5_feat",
        "days_since_last_calc":      "days_since_last_match_team_feat",
        "opp_days_since_last_calc":  "days_since_last_match_opp_feat",
        "elo_diff_calc":             "elo_diff_feat",
        "form_index_team":           "form_index_team_feat",
        "form_index_opp":            "form_index_opp_feat",
        "form_index_diff":           "form_index_diff_feat",
    }

    df = df.rename(columns=rename_map)

    # Kolom yang akan dibuang (kolom performa asli dari train — sudah digantikan)
    original_perf_cols = [
        "team_points_last5", "opp_points_last5", "points_last5_diff",
        "team_gd_last5", "opp_gd_last5", "gd_last5_diff",
        "h2h_points_last5", "h2h_gd_last5",
        "days_since_last_match_team", "days_since_last_match_opp",
        "team_points_last10", "opp_points_last10",
        "team_avg_goals_last5", "team_avg_conceded_last5",
        "opp_avg_goals_last5", "opp_avg_conceded_last5",
        "team_win_rate_last10", "opp_win_rate_last10",
        "elo_team", "elo_opponent",
        "rank_team", "rank_opponent", "rank_diff",
        "rank_missing_team", "rank_missing_opp",
    ]
    cols_to_drop = [c for c in original_perf_cols if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    # ---- Pisahkan ----
    train_out = df[df["is_test"] == False].drop(columns=["is_test"]).copy()
    test_out  = df[df["is_test"] == True].drop(columns=["is_test"]).copy()

    # Test tidak memiliki target — pastikan drop
    test_out = test_out.drop(columns=["team_goals", "opp_goals"], errors="ignore")

    train_out.to_csv(OUT_TRAIN, index=False)
    test_out.to_csv(OUT_TEST, index=False)

    print(f"[OK] train_featured.csv -> {len(train_out)} baris, {len(train_out.columns)} kolom")
    print(f"[OK] test_featured.csv  -> {len(test_out)} baris, {len(test_out.columns)} kolom")
    print(f"\nKolom fitur baru (suffix '_feat'):")
    feat_cols = [c for c in train_out.columns if c.endswith("_feat")]
    for c in feat_cols:
        non_null = train_out[c].notna().sum()
        print(f"  {c:40s} - non-null: {non_null}/{len(train_out)}")


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    print("=" * 60)
    print("FEATURE ENGINEERING PIPELINE")
    print("=" * 60)

    print("\n[1/7] Loading & merging data...")
    df = load_and_merge()
    print(f"      Total baris gabungan: {len(df)}")
    print(f"      Rentang waktu: {df['date'].min()} -> {df['date'].max()}")

    print("\n[2/7] Computing Elo ratings...")
    df = compute_elo(df)
    print(f"      Elo computed. Sample: {df['elo_team_calc'].dropna().iloc[:3].tolist()}")

    print("\n[3/7] Computing rolling stats (last 5/10 matches)...")
    df = compute_rolling_stats(df)

    print("\n[4/7] Computing head-to-head stats...")
    df = compute_h2h(df)

    print("\n[5/7] Mirroring opponent features...")
    df = mirror_opponent_features(df)

    print("\n[6/7] Computing rank proxy & derived features...")
    df = compute_rank_from_elo(df)
    df = compute_derived(df)

    print("\n[7/7] Finalizing & saving...")
    finalize_and_save(df)

    print("\n" + "=" * 60)
    print("PIPELINE SELESAI")
    print("=" * 60)


if __name__ == "__main__":
    main()
