# 💡 V14 Breakthrough Strategy: Dari Stagnasi 2.51 ke Target 2.30

> **Status**: V13 Lite stagnan di AW-MAE 2.514 — hanya +0.005 dari V12.  
> **Target**: AW-MAE 2.30 (Δ -0.21)  
> **Deadline**: Sebelum kompetisi berakhir

---

## 🔬 Root Cause Analysis: Kenapa V1–V13 Stagnan?

### 1. Bottleneck #1: Women's Football (AW-MAE 4.72!)

| Metrik | Men | Women | Gap |
|--------|-----|-------|-----|
| AW-MAE | 3.39 | **4.72** | +1.33 |
| Data Train | 69,966 | 8,806 | 8:1 ratio |
| Data Test | 28,464 | 13,958 | 2:1 ratio |
| Test Weight | 67% | **33%** | (overweighted in final score) |

**Dampak:** Meskipun women's football hanya 33% dari test set, error-nya 1.4× lebih besar. Ini berarti women's football menyumbang **~45% dari total AW-MAE**!

Rumus perkiraan: `AW-MAE_total ≈ 0.67 × 3.39 + 0.33 × 4.72 = 2.27 + 1.56 = 3.83` ❌ (tidak sesederhana ini karena NLS power)

Perhitungan yang lebih akurat (dari output V13):
- Men: 28,464 rows × 3.39 = ~96,500 loss units  
- Women: 13,958 rows × 4.72 = ~65,900 loss units  
- **Women menyumbang ~41% dari total loss meskipun hanya 33% data**

### 2. Bottleneck #2: Outcome Prediction (58.8%)

Metrik AW-MAE memberikan penalti **1.5×** jika outcome (M/S/K) salah diprediksi.

Dari 42,422 test match:
- Outcome benar: 24,960 (58.8%) — multiplier 1.0
- Outcome salah: 17,462 (41.2%) — multiplier 1.5

Jika outcome bisa dinaikkan ke 65%, perkiraan dampak:
- Loss rata-rata sekarang ≈ 2.514
- Dengan outcome 65%: ~2.35 → **Δ ≈ -0.16**

### 3. Bottleneck #3: Exact Score (hanya 10.6%)

Penalti exact match di AW-MAE = 0.30. Saat ini hanya 10.6% match diprediksi tepat.

Target: 15% exact → **Δ ≈ -0.04**

### 4. Bottleneck #4: No Prediction for High-Scoring Games (5+ goals)

V1–V13 memotong prediksi ke [0,5]. Tapi data nyata memiliki skor hingga 22 gol (Tonga 0-22 Australia). Model tidak pernah memprediksi >5 gol, padahal:
- 20% match men memiliki total gol ≥5
- 33% match women memiliki total gol ≥5

### 5. Bottleneck #5: Friendly Matches (Noise Besar)

Dari anomaly report: 35 dari 100 match paling anomali adalah **friendly**. Model saat ini tidak memiliki fitur `is_friendly`.

---

## 🎯 V14 Strategy: 7 Inovasi Bertarget

### Strategi disusun berdasarkan estimasi dampak terbesar → terkecil

---

### 🥇 P0: GENDER-AWARE ARCHITECTURE (Estimasi Δ: -0.15)

**Masalah**: Model saat ini **BUTA GENDER**. Fitur `is_women` sudah ada di train_final/test_final tapi tidak dimanfaatkan secara arsitektural. Pipeline melatih model terpisah untuk men dan women, tapi **proses feature engineering tidak membedakan keduanya**.

**Solusi:**

#### A. Tambah fitur interaksi gender ke feature engineering
```python
# Di feature_engineering_v6.py, tambahkan:
# 1. Elo_diff * is_women (women memiliki variansi Elo lebih besar)
df["elo_diff_x_women"] = df["elo_diff"] * df["is_women"].astype(float)
df["elo_team_x_women"] = df["elo_team"] * df["is_women"].astype(float)

# 2. Tournament TE terpisah per gender
# tournament_te_women dan tournament_te_men (bukan satu TE gabungan)

# 3. Confidence interval widening untuk women
df["uncertainty_mult"] = 1.0 + 0.5 * df["is_women"].astype(float) 
# (women punya 50% lebih uncertainty → lambda Poisson dikalikan)
```

#### B. Data augmentation untuk women's football
```python
# Bootstrap women's training data 3× dengan noise kecil
women_augmented = []
for _ in range(3):
    sample = train_w.sample(frac=1.0, replace=True)
    sample["elo_diff"] += np.random.normal(0, 10, len(sample))
    women_augmented.append(sample)
train_w_augmented = pd.concat(women_augmented)
# Hasil: 8,806 → 26,418 (+ noise)
```

#### C. Transfer learning dengan fine-tuning
```python
# Step 1: Train base model on men data (69K rows)
# Step 2: Fine-tune last N layers on women data (8.8K + augmented 26.4K)
# Step 3: Gunakan men model sebagai prior, women model sebagai posterior
```

**Estimasi**: Menurunkan women AW-MAE dari 4.72 → 3.80 → Δ total ≈ -0.15

---

### 🥈 P1: OUTCOME-CENTRIC TRAINING (Estimasi Δ: -0.12)

**Masalah**: Outcome hanya 58.8%. Ini adalah kelemahan terbesar karena multiplier 1.5×.

**Solusi:**

#### A. Two-stage focused training
```python
# Stage 1: Dedicated outcome classifier dengan class weights
# Karena M/S/K tidak seimbang (home wins > away wins > draws)
class_weights = compute_class_weight('balanced', classes=[0,1,2], y=y_outcome)
# Gunakan classifier terpisah dengan loss function yang memaksimalkan outcome accuracy

# Stage 2: Conditional score prediction
# Prediksi (team_goals, opp_goals) KONDISIONAL pada outcome yang sudah diprediksi
# Ini memaksa konsistensi outcome → menghilangkan possibility skor yang kontradiksi outcome
```

#### B. Soft labeling untuk near-draw matches
```python
# Jika Elo diff < 50 poin, label outcome menjadi "soft" (bukan hard 0/1/2)
# Probability outcome = [0.33, 0.34, 0.33] → memaksa model lebih hati-hati
elo_threshold = 50
mask = abs(df["elo_diff"]) < elo_threshold
# Gunakan label smoothing untuk matches dengan Elo diff kecil
```

#### C. Outcome-aware ensemble
```python
# Alih-alih memilih classification ATAU regression,
# GUNAKAN KEDUANYA dengan voting:
# - Jika classifier outcome yakin (prob > 0.7), gunakan classifier score
# - Jika tidak yakin (prob < 0.7), gunakan regression score
# Ini = adaptive ensemble berdasarkan confidence
```

**Estimasi**: Outcome 58.8% → 66% → Δ ≈ -0.12

---

### 🥉 P2: TAIL-EVENT MODELING (Estimasi Δ: -0.08)

**Masalah**: Model mengasumsikan max 6 gol (0-5), tapi data nyata memiliki skor ekstrem.  
**Dampak**: Prediksi selalu under-estimate untuk high-scoring matches.

**Solusi:**

#### A. Zero-inflated negative binomial (ZINB) sebagai ganti truncated Poisson
```python
# Poisson standar: lambda = rata-rata gol
# ZINB: lambda + overdispersion parameter + zero-inflation
# Lebih cocok untuk sepak bola karena:
# - Overdispersion (variance > mean) untuk high-scoring games
# - Zero-inflation untuk banyaknya skor 0-0

from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP
# atau implementasi manual
```

#### B. Prediksi "prone to high-scoring" sebagai binary classifier
```python
# Fitur tambahan: binary classifier memprediksi apakah match akan high-scoring (≥5 total gol)
# Jika ya → tambahkan boost ke lambda Poisson +0.5
# Ini membantu model memprediksi tail events tanpa merusak prediksi normal
```

#### C. Extreme value distribution untuk tail modeling
```python
# Generalized Pareto Distribution (GPD) untuk tail di atas threshold 4 gol
# Fit GPD pada residual untuk mengoreksi under-estimation di tail
from scipy.stats import genpareto
```

**Estimasi**: Δ ≈ -0.08 (dari memperbaiki under-estimation di high-scoring matches yang mencakup 20-33% data)

---

### 4️⃣ P3: FRIENDLY MATCH DOWN-WEIGHTING (Estimasi Δ: -0.04)

**Masalah**: Friendly matches adalah sumber noise terbesar (35/100 anomali teratas), tapi model memperlakukan semua match sama pentingnya.

**Solusi:**

#### A. Sample weight adjustment
```python
# Beri bobot lebih rendah ke friendly matches saat training
# Friendly = 0.5, Competitive = 1.0, World Cup = 1.5
tournament_weights = {
    'Friendly': 0.5,
    'FIFA World Cup': 1.5,
    'UEFA Euro': 1.3,
    # ... etc
}
sample_weights = df['tournament'].map(tournament_weights).fillna(1.0)
```

#### B. Fitur `is_friendly` + interaksi
```python
df["is_friendly"] = (df["tournament"] == "Friendly").astype(int)
df["elo_diff_x_friendly"] = df["elo_diff"] * df["is_friendly"]
# Model akan belajar bahwa Elo diff kurang prediktif di friendly matches
```

#### C. Separate friendly calibration
```python
# Kalibrasi probabilitas outcome terpisah untuk friendly vs competitive
# Friendly: Elo diff 300 poin → hanya 55% chance menang (bukan 70%)
```

**Estimasi**: Δ ≈ -0.04 (dari mengurangi noise friendly matches)

---

### 5️⃣ P4: FEATURE ENGINEERING V6 (Estimasi Δ: -0.05)

**Fitur baru yang belum ada:**

```python
# 1. Rest days interaction dengan travel distance
df["fatigue_score"] = (1.0 / (df["days_since_last_match"] + 1)) * df["travel_distance"] / 1000

# 2. Altitude x home advantage interaction
df["altitude_x_home"] = df["altitude_diff"] * df["is_home"].astype(float)

# 3. Form acceleration (derivative dari form)
df["form_accel"] = df["form_diff"] - df.get("form_diff_prev", 0)

# 4. Goal scoring streak vs conceding streak
df["scoring_streak"] = ... # consecutive matches with goals
df["clean_sheet_streak"] = ... # consecutive clean sheets

# 5. Tournament round importance
# (Group stage < Quarterfinal < Semifinal < Final)
# Fitur: round_importance = exp(round_number)

# 6. Confederation strength index
# UEFA = 1.0, CONMEBOL = 0.95, CONCACAF = 0.75, AFC = 0.70, CAF = 0.65, OFC = 0.40
```

**Estimasi**: Δ ≈ -0.05 (dari penambahan 6 fitur baru)

---

### 6️⃣ P5: ENSEMBLE DIVERSIFICATION (Estimasi Δ: -0.03)

**Masalah**: V13 menggunakan 3× LightGBM untuk 3 stage berbeda — diversity rendah.

**Solusi:**

#### A. Arsitektur stacked ensemble
```python
# Level 0: Diverse base models
# - LightGBM (gradient boosting)
# - XGBoost (gradient boosting, berbeda splitting)
# - CatBoost (ordered boosting)
# - Logistic Regression (linear baseline, calibrated)
# - Random Forest (bagging, uncorrelated errors)

# Level 1: Meta-learner (simple weighted average atau Logistic Regression)
```

#### B. Feature subsampling
```python
# Setiap base model menggunakan subset fitur berbeda
# LightGBM: Semua 41+6 fitur
# XGBoost: Remove tournament TE, keep raw features
# CatBoost: Categorical-focused (tournament, confederation, venue)
# Random Forest: Non-linear interactions (form × Elo, altitude × travel)
```

**Estimasi**: Δ ≈ -0.03 (dari ensemble diversity)

---

### 7️⃣ P6: POST-PROCESSING CALIBRATION (Estimasi Δ: -0.02)

**Solusi:**

#### A. Platt scaling untuk outcome probabilities
```python
# Kalibrasi prob_out menggunakan holdout validation set
from sklearn.calibration import CalibratedClassifierCV
calibrated = CalibratedClassifierCV(base_model, method='sigmoid', cv=5)
```

#### B. Isotonic regression untuk expected goals calibration
```python
# Setelah prediksi xG, kalibrasi dengan isotonic regression
# agar distribusi prediksi match distribusi aktual
from sklearn.isotonic import IsotonicRegression
```

#### C. Quantile-based clipping
```python
# Alih-alih clip ke [0,5], gunakan quantile-based threshold
# Jika prediksi > 95th percentile training → cap di 95th percentile, bukan 5
```

**Estimasi**: Δ ≈ -0.02 (dari kalibrasi probabilitas)

---

## 📊 Perkiraan Kumulatif Dampak

| Prioritas | Inovasi | Est. Δ AW-MAE | Akumulasi |
|-----------|---------|---------------|-----------|
| **V13 Baseline** | (saat ini) | **2.514** | 2.514 |
| P0 | Gender-Aware Architecture | -0.15 | 2.364 |
| P1 | Outcome-Centric Training | -0.12 | 2.244 |
| P2 | Tail-Event Modeling | -0.08 | 2.164 |
| P3 | Friendly Down-Weighting | -0.04 | 2.124 |
| P4 | Feature Engineering V6 | -0.05 | 2.074 |
| P5 | Ensemble Diversification | -0.03 | 2.044 |
| P6 | Post-Processing Calibration | -0.02 | **2.024** |

> ⚠️ Estimasi ini **optimistic upper bound** — asumsi setiap inovasi additive dan tidak ada overlap negatif.  
> Realistis: 2.15–2.25. **Target 2.30 seharusnya achievable dengan P0+P1.**

---

## 🗺️ Rencana Implementasi (Prioritas Waktu)

### Iteration 1 (Hari Ini) — Quick Wins (Δ -0.10)
- [x] ~~V13 Lite baseline~~ ✅
- [ ] P3: Friendly match down-weighting (30 menit)
- [ ] P4: Feature engineering V6 (45 menit)
- [ ] Evaluasi V14_quick

### Iteration 2 (Besok) — High Impact (Δ -0.15)
- [ ] P0: Gender-aware architecture + data augmentation (2 jam)
- [ ] P1: Outcome-centric two-stage training (1.5 jam)
- [ ] Evaluasi V14_core

### Iteration 3 (Lusa) — Refinement (Δ -0.05)
- [ ] P2: Tail-event modeling (ZINB/GPD)
- [ ] P5: Ensemble diversification
- [ ] P6: Calibration post-processing
- [ ] Evaluasi V14_full

### Iteration 4 (Opsional) — Hyperparameter Tuning
- [ ] Grid search learning rate, num_leaves, min_child_samples per gender
- [ ] Temperature sweep untuk soft cascade
- [ ] Pseudo-labeling untuk test set

---

## ⚡ Quick Start: V14_quick.py (P3 + P4)

Sebagai langkah pertama, implementasi **P3 (Friendly Down-Weighting) + P4 (6 Fitur Baru)** yang bisa langsung diuji:

```python
# Modifikasi pada feature_engineering_v5.py:
# 1. Tambah kolom is_friendly
# 2. Tambah fatigue_score, altitude_x_home, form_accel, scoring_streak
# 3. Tambah round_importance, confederation_strength

# Modifikasi pada model_pipeline:
# 4. Sample weight adjustment untuk friendly vs competitive
```

---

## 📈 Metrik yang Harus Dipantau

Setiap iterasi V14 harus melaporkan:

| Metrik | V13 | Target V14 |
|--------|-----|------------|
| AW-MAE Overall | 2.514 | **< 2.30** |
| AW-MAE Men | 3.39 | < 3.10 |
| AW-MAE Women | 4.72 | < 4.00 |
| Outcome Correct | 58.8% | > 65% |
| Exact Match | 10.6% | > 13% |
| Goals RMSE | ? | < 1.80 |

---

## 🧪 Eksperimen yang TIDAK Disarankan (Berdasarkan Kegagalan V13)

| Eksperimen | Kenapa Gagal |
|------------|-------------|
| ~~Temperature sweep global~~ | Tidak ada grid yang optimal (P1C gagal) |
| ~~Pseudo-labeling semua test~~ | Iterative lambat + overfit (P2E gagal) |
| ~~3 ensemble × multiple stages~~ | Training 9 model = 8 menit (P2F full gagal) |
| ~~Bivariate ordinal regression langsung~~ | Konvergensi lama, hasil mirip soft cascade (P0B redundant) |

---

## 📚 Referensi

- `analysis/anomaly_report.md` — 18,292 anomaly matches, friendly = biggest noise
- `analysis/gender_analysis.md` — Women's football: test weight 33%, data train 11%
- `src/model_pipeline_v12.py` — Baseline terbaik yang masih relevan
- `src/feature_engineering_v5.py` — 41 fitur yang menjadi basis V14
- `src/evaluate_local.py` — Metrik AW-MAE untuk benchmark lokal