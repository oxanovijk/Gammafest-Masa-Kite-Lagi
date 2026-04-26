"""
Script: iterative_pipeline.py
Tujuan: Melakukan prediski murni kronologis untuk Test Set (2011-2026).
        Elo & Momentum tim diperbarui SECARA OTOMATIS setelah setiap tebakan
        menggunakan hasil tebakan itu sendiri. Anti-leakage 100%.
"""

import pandas as pd
import numpy as np
from collections import defaultdict, deque
import math
import lightgbm as lgb
import time
from scipy.stats import poisson

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

import warnings
warnings.filterwarnings('ignore')

from feature_engineering_v2 import (
    ELO_INIT, ELO_K, ELO_HOME_ADVANTAGE, get_k_factor, calc_elo_change,
    EWMA_ALPHA, ROLLING_WINDOW, _ewma_aggregate
)
from model_pipeline import (
    train_lgb, build_loss_tensor, erm_predict_batch, 
    LGB_PARAMS, N_ESTIMATORS, EARLY_STOPPING, MAX_GOALS
)

class IterativeEngine:
    def __init__(self):
        self.elo = defaultdict(lambda: ELO_INIT)
        self.history = defaultdict(lambda: deque(maxlen=ROLLING_WINDOW))
        self.h2h_history = defaultdict(lambda: deque(maxlen=5))
        self.last_date = {}

    def extract_features(self, row):
        """Mengekstrak fitur SEBELUM pertandingan terjadi"""
        team, gender = row["team"], row["gender"]
        opponent = row["opponent"]
        date_i = row["date"]
        
        feat = {}
        
        # 1. Elo
        key_tm = (team, gender)
        key_opp = (opponent, gender)
        feat["elo_team_feat"] = self.elo[key_tm]
        feat["elo_opponent_feat"] = self.elo[key_opp]
        feat["elo_diff_feat"] = feat["elo_team_feat"] - feat["elo_opponent_feat"]

        # 2. History (Team)
        h_tm = self.history[key_tm]
        if len(h_tm) > 0:
            agg5 = _ewma_aggregate(h_tm, date_i, last_n=5)
            agg10 = _ewma_aggregate(h_tm, date_i, last_n=10)
            if agg5:
                feat["team_pts_last5_ewma_feat"] = agg5["pts_ewma"]
                feat["team_gd_last5_ewma_feat"] = agg5["gd_ewma"]
                feat["team_avg_gf_last5_ewma_feat"] = agg5["avg_gf_ewma"]
                feat["team_avg_ga_last5_ewma_feat"] = agg5["avg_ga_ewma"]
                feat["team_pts_last5_simple_feat"] = agg5["pts_simple"]
            if agg10:
                feat["team_pts_last10_ewma_feat"] = agg10["pts_ewma"]
                feat["team_gd_last10_ewma_feat"] = agg10["gd_ewma"]
                feat["team_win_rate_last10_ewma_feat"] = agg10["win_rate_ewma"]
                feat["team_pts_last10_simple_feat"] = agg10["pts_simple"]
                
        if key_tm in self.last_date:
            feat["days_since_last_team_feat"] = (date_i - self.last_date[key_tm]).days

        # 3. History (Opponent)
        h_op = self.history[key_opp]
        if len(h_op) > 0:
            agg5_op = _ewma_aggregate(h_op, date_i, last_n=5)
            agg10_op = _ewma_aggregate(h_op, date_i, last_n=10)
            if agg5_op:
                feat["opp_pts_last5_ewma_feat"] = agg5_op["pts_ewma"]
                feat["opp_gd_last5_ewma_feat"] = agg5_op["gd_ewma"]
                feat["opp_avg_gf_last5_ewma_feat"] = agg5_op["avg_gf_ewma"]
                feat["opp_avg_ga_last5_ewma_feat"] = agg5_op["avg_ga_ewma"]
                feat["opp_pts_last5_simple_feat"] = agg5_op["pts_simple"]
            if agg10_op:
                feat["opp_pts_last10_ewma_feat"] = agg10_op["pts_ewma"]
                feat["opp_gd_last10_ewma_feat"] = agg10_op["gd_ewma"]
                feat["opp_win_rate_last10_ewma_feat"] = agg10_op["win_rate_ewma"]
                feat["opp_pts_last10_simple_feat"] = agg10_op["pts_simple"]
                
        if key_opp in self.last_date:
            feat["days_since_last_opp_feat"] = (date_i - self.last_date[key_opp]).days

        # 4. H2H
        h2h_key = (team, opponent, gender)
        h2h = self.h2h_history[h2h_key]
        if len(h2h) > 0:
            feat["h2h_pts_last5_simple_feat"] = sum(x[0] for x in h2h)
            feat["h2h_gd_last5_simple_feat"] = sum(x[1] - x[2] for x in h2h)
            agg_h2h = _ewma_aggregate(h2h, date_i)
            if agg_h2h:
                feat["h2h_pts_last5_ewma_feat"] = agg_h2h["pts_ewma"]
                feat["h2h_gd_last5_ewma_feat"] = agg_h2h["gd_ewma"]

        # 5. Derived
        if "team_pts_last5_ewma_feat" in feat and "opp_pts_last5_ewma_feat" in feat:
            feat["pts_last5_ewma_diff_feat"] = feat["team_pts_last5_ewma_feat"] - feat["opp_pts_last5_ewma_feat"]
            feat["gd_last5_ewma_diff_feat"] = feat["team_gd_last5_ewma_feat"] - feat["opp_gd_last5_ewma_feat"]
            feat["form_team_feat"] = feat["team_pts_last5_ewma_feat"] + feat["team_gd_last5_ewma_feat"] * 0.5
            feat["form_opp_feat"] = feat["opp_pts_last5_ewma_feat"] + feat["opp_gd_last5_ewma_feat"] * 0.5
            feat["form_diff_feat"] = feat["form_team_feat"] - feat["form_opp_feat"]
            
        if "team_pts_last10_ewma_feat" in feat and "opp_pts_last10_ewma_feat" in feat:
            feat["pts_last10_ewma_diff_feat"] = feat["team_pts_last10_ewma_feat"] - feat["opp_pts_last10_ewma_feat"]
            
        if "team_pts_last5_simple_feat" in feat and "opp_pts_last5_simple_feat" in feat:
            feat["pts_last5_simple_diff_feat"] = feat["team_pts_last5_simple_feat"] - feat["opp_pts_last5_simple_feat"]
            
        if "team_pts_last10_simple_feat" in feat and "opp_pts_last10_simple_feat" in feat:
            feat["pts_last10_simple_diff_feat"] = feat["team_pts_last10_simple_feat"] - feat["opp_pts_last10_simple_feat"]

        return feat

    def update_state(self, row, goals_team, goals_opp):
        """Update Elo & History SETELAH pertandingan/prediksi terjadi"""
        team, opponent, gender = row["team"], row["opponent"], row["gender"]
        date_i = row["date"]
        
        key_tm = (team, gender)
        key_opp = (opponent, gender)
        
        # 1. Update Elo
        cur_elo_tm = self.elo[key_tm]
        cur_elo_opp = self.elo[key_opp]
        
        if goals_team > goals_opp: score_tm = 1.0
        elif goals_team < goals_opp: score_tm = 0.0
        else: score_tm = 0.5
        
        tournament = row["tournament"]
        conf_a = str(row.get("confederation_team", "Unknown"))
        conf_b = str(row.get("confederation_opponent", "Unknown"))
        k = get_k_factor(tournament, conf_a, conf_b)
        
        gd = abs(goals_team - goals_opp)
        if gd == 2: k *= 1.5
        elif gd == 3: k *= 1.75
        elif gd > 3: k *= (1.75 + (gd - 3) / 8.0)
        
        is_home_tm = row.get("is_home", 0)
        is_home_opp = row.get("is_home_opponent", 0)
        neutral = row.get("neutral", 0)
        
        if neutral == 1:
            home_adv = 0.0
        elif is_home_tm == 1:
            home_adv = ELO_HOME_ADVANTAGE
        elif is_home_opp == 1:
            home_adv = -ELO_HOME_ADVANTAGE
        else:
            home_adv = 0.0
            
        delta = calc_elo_change(cur_elo_tm, cur_elo_opp, score_tm, k, home_adv)
        self.elo[key_tm] = cur_elo_tm + delta
        self.elo[key_opp] = cur_elo_opp - delta
        
        # 2. Update EWMA
        if goals_team > goals_opp: pts_tm, pts_opp = 3, 0
        elif goals_team == goals_opp: pts_tm, pts_opp = 1, 1
        else: pts_tm, pts_opp = 0, 3
            
        self.history[key_tm].append((pts_tm, goals_team, goals_opp, date_i))
        self.history[key_opp].append((pts_opp, goals_opp, goals_team, date_i))
        self.last_date[key_tm] = date_i
        self.last_date[key_opp] = date_i

        # 3. Update H2H
        h2h_key1 = (team, opponent, gender)
        h2h_key2 = (opponent, team, gender)
        self.h2h_history[h2h_key1].append((pts_tm, goals_team, goals_opp, date_i))
        self.h2h_history[h2h_key2].append((pts_opp, goals_opp, goals_team, date_i))

def safe_merge_ctx(feat, ctx):
    for k, v in ctx.items():
        if k != "Id":
            feat[k] = v
    return feat

def fill_missing_cols(feat, all_cols):
    for c in all_cols:
        if c not in feat:
            feat[c] = np.nan
    return feat

def main():
    print("="*60)
    print("STRICT ITERATIVE PREDICTION PIPELINE")
    print("="*60)
    
    # 1. Melatih Model dengan data train_final
    from model_pipeline import load_data
    train_df, test_df_stale, feature_cols = load_data()
    
    # Gunakan 100% train untuk final model (no early stopping)
    # Ini memastikan kita pakai semua ilmu
    print("\n[1/4] Melatih LightGBM pada seluruh Train Set...")
    X_train = train_df[feature_cols]
    w_train = train_df['sample_weight'].values
    
    model_team = train_lgb(X_train, train_df["team_goals"], X_train, train_df["team_goals"], w_train, w_train)
    model_opp = train_lgb(X_train, train_df["opp_goals"], X_train, train_df["opp_goals"], w_train, w_train)
    loss_tensor = build_loss_tensor(MAX_GOALS)
    
    # 2. Menyiapkan Engine State
    print("\n[2/4] Menginisialisasi State Engine dari dataset Train...")
    train_raw = pd.read_csv("dataset/train.csv")
    train_raw["date"] = pd.to_datetime(train_raw["date"])
    # Kita butuh konfederasi opponent juga yang harus diderive jika tidak ada di raw
    # Tapi untuk efisiensi, jalankan update pass pada match unik
    train_matches = train_raw.drop_duplicates("match_id").sort_values("date")
    
    engine = IterativeEngine()
    for _, row in train_matches.iterrows():
        engine.update_state(row, row["team_goals"], row["opp_goals"])
        
    print(f"      State terisi. Total tim terekam: {len(engine.elo)}")
    
    # 3. Iterative Prediction di Test
    print("\n[3/4] Menjalankan PREDIKSI ITERATIF pada Test Set (2011-2026)...")
    test_raw = pd.read_csv("dataset/test.csv")
    test_raw["date"] = pd.to_datetime(test_raw["date"])
    
    test_final_df = pd.read_csv("dataset/test_final.csv")
    ctx_cols = [c for c in test_final_df.columns if c.endswith("_ctx")] + ["Id"]
    ctx_test = test_final_df[ctx_cols].set_index("Id").to_dict("index")
    
    # Sort kronologis mutlak
    test_raw = test_raw.sort_values(["date", "match_id"])
    
    submission_rows = []
    
    # Agar cepat, kita gunakan list comprehension & dictionary lookup
    # Kami update satu per satu secara runut karena butuh hasil tebak untuk besoknya
    
    buffer_feat = []
    buffer_ids = []
    buffer_rows = []
    current_date = None
    
    def process_buffer(bf, bids, brows):
        if not bf: return
        df_batch = pd.DataFrame(bf)[feature_cols]
        lam_t = model_team.predict(df_batch)
        lam_o = model_opp.predict(df_batch)
        
        # dynamic_erm butuh n ls tensor. pake ERM biasa dr model_pipeline yg vector
        pt, po = erm_predict_batch(lam_t, lam_o, loss_tensor)
        
        for i, (pred_t, pred_o) in enumerate(zip(pt, po)):
            Id = bids[i]
            row_dict = brows[i]
            
            # Update state engine PAKAI PREDIKSI!
            engine.update_state(row_dict, pred_t, pred_o)
            submission_rows.append({"Id": Id, "team_goals": pred_t, "opp_goals": pred_o})
            
    t0 = time.time()
    for i, row in test_raw.iterrows():
        date_i = row["date"]
        
        # Kalau ganti hati/tanggal, flush dan prediksikan secara batch biar cepat -> Update state
        if current_date is not None and date_i != current_date:
            process_buffer(buffer_feat, buffer_ids, buffer_rows)
            buffer_feat, buffer_ids, buffer_rows = [], [], []
            
        current_date = date_i
        
        # Ekstrak Fitur Murni Baru
        core_feat = engine.extract_features(row)
        Id = row["Id"]
        
        # Gabung dengan konteks
        if Id in ctx_test:
            core_feat.update(ctx_test[Id])
            
        core_feat = fill_missing_cols(core_feat, feature_cols)
        
        buffer_feat.append(core_feat)
        buffer_ids.append(Id)
        buffer_rows.append(row)
        
        if len(submission_rows) % 5000 == 0 and len(submission_rows) > 0:
            print(f"      Telah memprediksi {len(submission_rows)} / {len(test_raw)} laga...")
            
    # Sisa buffer akhir
    process_buffer(buffer_feat, buffer_ids, buffer_rows)
    elapsed = time.time() - t0
    print(f"      Selesai dalam {elapsed:.1f} detik.")
    
    # 4. Save & Evaluate
    print("\n[4/4] Finalisasi & Evaluasi...")
    sub_df = pd.DataFrame(submission_rows)
    
    # Samakan urutan test asli
    test_order = pd.read_csv("dataset/sample submission.csv")[["Id"]]
    sub_df = pd.merge(test_order, sub_df, on="Id", how="left")
    sub_df.to_csv("dataset/submission.csv", index=False)
    
    print("      Berhasil ditimpan ke dataset/submission.csv!")
    
    # Otomatis evaluasi (jika verify tersedia)
    import subprocess
    print("\nMemanggil Evaluator Lokal...\n")
    subprocess.run(["python", "src/evaluate_local.py"])

if __name__ == "__main__":
    main()
