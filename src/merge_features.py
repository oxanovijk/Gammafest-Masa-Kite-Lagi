"""
Merge Feature Pipeline -- Gammafest Masa Kite Lagi
============================================================
Tujuan: Menggabungkan hasil kerja Anggota 1 (Core Historical) 
        dan Anggota 2 (Context, Geo, Socio-Economic) menjadi dataset final.

Target merge:
  1. dataset/train_core_v2.csv   <-- Merge (on="Id") --> dataset/train_context_feat.csv
  2. dataset/test_core_v2.csv    <-- Merge (on="Id") --> dataset/test_context_feat.csv

Output:
  - dataset/train_final.csv
  - dataset/test_final.csv
"""

import pandas as pd
from pathlib import Path

# ===========================================================================
# KONFIGURASI PATH
# ===========================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "dataset"

TRAIN_CORE = DATA_DIR / "train_core_v2.csv"
TEST_CORE  = DATA_DIR / "test_core_v2.csv"

TRAIN_CTX = DATA_DIR / "train_context_feat.csv"
TEST_CTX  = DATA_DIR / "test_context_feat.csv"

OUT_TRAIN_FINAL = DATA_DIR / "train_final.csv"
OUT_TEST_FINAL  = DATA_DIR / "test_final.csv"

def merge_datasets(core_path, ctx_path, out_path, is_train=True):
    print(f"\n[+] Membaca {core_path.name}...")
    df_core = pd.read_csv(core_path)
    print(f"    - Baris: {len(df_core)} | Kolom: {len(df_core.columns)}")
    
    print(f"[+] Membaca {ctx_path.name}...")
    df_ctx = pd.read_csv(ctx_path)
    print(f"    - Baris: {len(df_ctx)} | Kolom: {len(df_ctx.columns)}")
    
    # Deteksi dan pelaporan duplikasi kolom (selain Id)
    core_cols = set(df_core.columns) - {"Id"}
    ctx_cols  = set(df_ctx.columns) - {"Id"}
    overlap   = core_cols.intersection(ctx_cols)
    
    if overlap:
        print(f"    [WARNING] Ditemukan kolom duplikat: {overlap}")
        # Jika ada duplikasi ringan selain target, hapus dari ctx
        if is_train and "team_goals" in overlap:
            df_ctx = df_ctx.drop(columns=["team_goals", "opp_goals"], errors="ignore")
            overlap = overlap - {"team_goals", "opp_goals"}
            
        df_ctx = df_ctx.drop(columns=list(overlap), errors="ignore")
    
    print(f"[+] Menjalankan penggabungan (Merge on='Id')...")
    df_final = pd.merge(df_core, df_ctx, on="Id", how="inner")
    
    # Validasi jumlah baris tidak berubah
    if len(df_final) != len(df_core):
        print(f"    [ERROR] Jumlah baris terdistorsi setelah merge! "
              f"({len(df_core)} -> {len(df_final)})")
        return
        
    print(f"[+] Menyimpan output ke {out_path.name}...")
    df_final.to_csv(out_path, index=False)
    print(f"    [OK] Selesai! Final Baris: {len(df_final)} | Kolom: {len(df_final.columns)}")

def main():
    print("=" * 60)
    print("PROSES MERGE DATASET CORE (A1) & CONTEXT (A2)")
    print("=" * 60)
    
    print("\n>>> MEMPROSES DATA TRAIN")
    merge_datasets(TRAIN_CORE, TRAIN_CTX, OUT_TRAIN_FINAL, is_train=True)
    
    print("\n>>> MEMPROSES DATA TEST")
    merge_datasets(TEST_CORE, TEST_CTX, OUT_TEST_FINAL, is_train=False)

    print("\n" + "=" * 60)
    print("SEMUA SELESAI")
    print("Dataset siap diserahkan ke Modeller.")
    print("=" * 60)

if __name__ == "__main__":
    main()
