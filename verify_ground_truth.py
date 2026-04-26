import pandas as pd
import numpy as np

print("1. Loading raw files...")
test = pd.read_csv("dataset/test.csv")
res_men = pd.read_csv("dataset/results.csv")
res_women = pd.read_csv("dataset/women_result.csv")

# Tambah gender flag
res_men["gender"] = "M"
res_women["gender"] = "W"

# Gabung jadi satu ground truth master
res_master = pd.concat([res_men, res_women], ignore_index=True)

# Parse date
test["date"] = pd.to_datetime(test["date"])
res_master["date"] = pd.to_datetime(res_master["date"])

print(f"Total referensi matches di ground truth men:   {len(res_men)}")
print(f"Total referensi matches di ground truth women: {len(res_women)}")
print(f"Total test rows untuk di-mapping:              {len(test)}")

# Buat fungsi untuk bikin standard key tanpa pandang bulu home/away
# Format: YYYY-MM-DD_G_TeamA_TeamB (Team array disorted)
def make_key(d, g, t1, t2):
    teams = sorted([str(t1).strip(), str(t2).strip()])
    return f"{d.strftime('%Y-%m-%d')}_{g}_{teams[0]}_{teams[1]}"

print("\n2. Membuat relational keys...")
test["match_key"] = test.apply(lambda r: make_key(r["date"], r["gender"], r["team"], r["opponent"]), axis=1)
res_master["match_key"] = res_master.apply(lambda r: make_key(r["date"], r["gender"], r["home_team"], r["away_team"]), axis=1)

# Atasi duplikasi pada res_master (kadang ada match dicatat double)
res_master = res_master.drop_duplicates(subset=["match_key"], keep="last")

print("\n3. Mapping goals...")
# Lakukan left join pada test
test_joined = pd.merge(test, res_master[["match_key", "home_team", "away_team", "home_score", "away_score"]], on="match_key", how="left")

# Sekarang kita tentukan mana yang team_goals dan opp_goals
def assign_goals(row):
    if pd.isna(row["home_score"]): return np.nan, np.nan
    
    # Jika team == home_team
    if str(row["team"]).strip() == str(row["home_team"]).strip():
        return row["home_score"], row["away_score"]
    # Jika team == away_team
    elif str(row["team"]).strip() == str(row["away_team"]).strip():
        return row["away_score"], row["home_score"]
    else:
        # Menangani anomali penamaan yang lolos sorted key (sangat jarang jika exact match)
        return np.nan, np.nan

test_joined[["team_goals_gt", "opp_goals_gt"]] = test_joined.apply(
    lambda r: pd.Series(assign_goals(r)), axis=1
)

# Hitung coverage
found_mask = test_joined["team_goals_gt"].notna()
found_count = found_mask.sum()
missing_count = len(test_joined) - found_count
print(f"\n=== HASIL CAKUPAN GROUND TRUTH ===")
print(f"Ditemukan : {found_count} baris ({found_count/len(test_joined)*100:.2f}%)")
print(f"Hilang    : {missing_count} baris ({missing_count/len(test_joined)*100:.2f}%)")

if missing_count > 0:
    print("\nContoh yang masih gagal di-map (mungkin karena ejaan nama negara beda / data belum ada):")
    missing_df = test_joined[~found_mask]
    for i, r in missing_df.head(10).iterrows():
        print(f"  - {r['date'].strftime('%Y-%m-%d')} | {r['gender']} | {r['team']} vs {r['opponent']}")

# Simpan hasilnya ordered sesuai aslinya
# Kolom yang disiapkan: Id, team_goals, opp_goals
final_gt = test_joined[["Id", "team_goals_gt", "opp_goals_gt"]].rename(columns={
    "team_goals_gt": "team_goals",
    "opp_goals_gt": "opp_goals"
})

output_path = "dataset/test_ground_truth.csv"
final_gt.to_csv(output_path, index=False)
print(f"\n4. File CSV gabungan siap dan selaras dengan Test Final disimpan ke: {output_path}")
