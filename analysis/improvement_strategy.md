# 🚀 Strategi Perbaikan — Menembus Batas AW-MAE 2.50

> **Tanggal**: 2 Mei 2026
> **Baseline**: V14 (AW-MAE 2.5100, Outcome 58.9%, Exact 10.3%)
> **Target**: < 2.30 AW-MAE (minimal -0.21 dari baseline)
> **Prinsip**: TIDAK rewrite dari scratch — semua bangun di atas V14

---

## 📐 PRIORITAS STRATEGI

```
PRIORITAS 1 (KRITIS) → Bottleneck Women + Ordinal Structure
PRIORITAS 2 (TINGGI) → Outcome Accuracy + Kalibrasi
PRIORITAS 3 (MEDIUM) → Tail Handling + Ensemble Diversity
PRIORITAS 4 (VALIDASI) → Holdout Split + Ablation Framework
```

---

## 🔴 PRIORITAS 1 (KRITIS): Bottleneck Women + Ordinal Structure

### Strategi 1.1: Ganti 36-Class Flat PMF → Bivariate Ordinal Regression

**Masalah**: 36-class flat classification memperlakukan (2,0) dan (2,1) sebagai kelas terpisah, padahal struktur gol bersifat ORDINAL. Skor 2-1 lebih "dekat" ke 2-0 daripada ke 0-2. Model harus belajar dari nol adjacency matriks.

**Solusi**: Reformulasi Stage 2 dengan struktur ordinal:

```
Opsi A: Ordinal Logistic Regression
- Prediksi team_goals: 6 kelas ORDINAL (0,1,2,3,4,5+)
- Prediksi opp_goals: 6 kelas ORDINAL (0,1,2,3,4,5+)
- Copula / bivariate link untuk korelasi residual antara team & opp
- Output: P(team_goals=t, opp_goals=o) = P(t) × P(o) × copula_correction(t,o)

Opsi B: Bivariate Poisson (Dixon-Coles)
- λ_team = f(attack_team × defense_opp × home_advantage × ...)
- λ_opp = f(attack_opp × defense_team × ...)
- Dixon-Coles correction: ρ untuk korelasi skor rendah (0-0, 1-0, 0-1, 1-1)
- Output: distribusi probabilitas bivariate alami
```

**Implementasi rekomendasi**: Opsi A lebih mudah diintegrasikan dengan V14 karena tetap menggunakan LightGBM (ubah objective `multiclass 36` → `multiclass 6` + `multiclass 6` dengan copula post-processing). Opsi B lebih "benar" secara statistik tapi perlu rewrite signifikan.

**Estimasi dampak**: **-0.08 s/d -0.15 AW-MAE** (terutama dari sel matriks sparse di Women)

**File baru**: `src/model_pipeline_v17_ordinal.py`

---

### Strategi 1.2: Zero-Inflated Negative Binomial (ZINB) untuk Women

**Masalah**: Women's football memiliki:
- Overdispersion (variance >> mean): Std dev 2.58 vs Mean 2.65
- Inflasi zeros: banyak match 0-0, 1-0
- Poisson tidak cukup untuk menangkap overdispersion → under-estimate tail

**Solusi**: Gantikan Poisson PMF untuk Women dengan ZINB:
```
Stage 2 menghasilkan:
  - π (probabilitas zero-inflation, dari sigmoid)
  - μ (mean goals)
  - α (dispersion parameter, dari softplus)

P(X=k) = π × I(k=0) + (1-π) × NB(k; μ, α)

Implementasi:
1. Train Stage 2 seperti biasa → dapatkan raw logits
2. Post-process: hitung μ dari logits, π dari sigmoid(σ × elo_diff), α dari softplus(β × is_women)
3. Distribusi ZINB menggantikan truncated Poisson untuk prediksi goals
```

**Estimasi dampak**: **-0.05 s/d -0.10 AW-MAE** (Women-specific)

**File baru**: `src/zinb_head.py`

---

### Strategi 1.3: Data Augmentation Women (SMOTE + Bootstrap)

**Masalah**: 8,806 sampel Women tidak cukup untuk mempelajari 36 sel matriks. Transfer learning dari Men (w=0.3) tidak efektif karena distribusi gol fundamentally berbeda.

**Solusi**: Synthetic data augmentation khusus Women:

```
Opsi A: Bootstrap dengan noise
- Bootstrap 3× dari data Women
- Tambahkan Gaussian noise JITTER kecil pada fitur kontinyu (σ = 0.01 normalized)
- TIDAK mengubah target gol (label): ini adalah feature augmentation, bukan label augmentation
- Total: 8,806 × 3 = 26,418 sampel

Opsi B: SMOTE untuk outcome undersampled
- SMOTE pada kelas minoritas (Draw dan Away Win balance)
- Hanya augmentasi fitur kontinyu (Elo, EWMA, Pi, dll)
- Tidak menyentuh fitur kategorikal (venue, tournament)
- Total: seimbangkan ketiga outcome class

Opsi C: Mixup antara sampel Women
- x_new = λ × x_i + (1-λ) × x_j, λ ~ Beta(0.2, 0.2)
- y_new weighted average dari y_i dan y_j
- Ini adalah form of data augmentation yang terbukti di tabular
```

**Rekomendasi**: Mulai dengan Opsi A + B (bootstrap ×3 + SMOTE untuk draw) karena paling aman.

**Estimasi dampak**: **-0.03 s/d -0.07 AW-MAE** (Women-specific)

**Catatan**: Ini HANYA untuk Women model. Men sudah punya cukup data.

---

## 🟠 PRIORITAS 2 (TINGGI): Outcome Accuracy + Kalibrasi

### Strategi 2.1: Isotonic Calibration pada Outcome Classifier

**Masalah**: LightGBM menghasilkan probabilitas overconfident (tipikal boosting). Stage 1 (Outcome) output misal [0.85, 0.10, 0.05] padahal true probability seharusnya [0.60, 0.25, 0.15]. Ini merusak reconciliation di cascade.

**Solusi**: Kalibrasi Stage 1 dengan isotonic regression:

```python
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression

# Step 1: Train Stage 1 LGB seperti biasa
# Step 2: Kalibrasi dengan isotonic regression pada out-of-fold predictions
cal_outcome = CalibratedClassifierCV(
    base_estimator=outcome_model,
    method='isotonic',  # lebih fleksibel dari Platt
    cv=5  # 5-fold untuk kalibrasi
)
cal_outcome.fit(X_train, y_train_outcome)
prob_calibrated = cal_outcome.predict_proba(X_test)

# Step 3: Inject calibrated probabilities ke soft cascade
# P(Score) = P(Score|Outcome) × P_calibrated(Outcome)
```

**Estimasi dampak**: **-0.04 s/d -0.08 AW-MAE** (outcome accuracy +2-3%)

---

### Strategi 2.2: Soft Labeling untuk Near-Draw Matches

**Masalah**: Outcome classifier kehilangan banyak poin pada near-draw matches (Elo diff <50). Ini adalah match yang secara fundamental unpredictable — memaksa hard label [W/D/L] adalah overconfident.

**Solusi**: Soft label untuk training Stage 1:

```
Jika |Elo_diff| < 50:
    target = [0.33, 0.34, 0.33]  # W, D, L hampir seimbang
Jika |Elo_diff| >= 50:
    # Konvensional hard label (1,0,0), (0,1,0), atau (0,0,1)

Dampak:
- Model tidak overconfident pada matches yang inherently unpredictable
- Draw probability naik untuk match seimbang
- ERM akan lebih memilih draw untuk match ketat → outcome accuracy naik
```

**Implementasi**: Modifikasi label training Stage 1. Untuk Elo diff < 50, gunakan weighted cross-entropy dengan soft target.

**Estimasi dampak**: **-0.03 s/d -0.05 AW-MAE**

---

### Strategi 2.3: Class Weights for Draw Imbalance

**Masalah**: Draw hanya 20% data → Stage 1 under-predict draw → outcome accuracy rendah pada draw.

**Solusi**: Weighted loss untuk Stage 1:

```python
# Distribution di training Men: ~45% Home, ~25% Draw, ~30% Away
# Balanced weight:
class_weights = {
    0: 1 / 0.45 / 3,  # Home Win
    1: 1 / 0.25 / 3,  # Draw  
    2: 1 / 0.30 / 3   # Away Win
}
# Hasil: Draw weight = 1.33, Home = 0.74, Away = 1.11

model.fit(X, y, sample_weight=class_weights[y])
```

**Estimasi dampak**: **-0.02 s/d -0.04 AW-MAE**

---

### Strategi 2.4: Friendly Match Specialization

**Masalah**: 35/100 match paling anomali adalah friendly. Friendly punya pola berbeda: tim kuat rotasi, Elo gap tidak berkorelasi dengan hasil.

**Solusi**: Friendly-aware outcome prediction:

```
Opsi A: Separate Head for Friendly
- Stage 1 memiliki 2 output head: friendly_head dan competitive_head
- Saat inference, switch berdasarkan is_friendly

Opsi B: Interaction Feature yang Lebih Kuat
- elo_diff × is_friendly
- h2h × is_friendly
- days_rest × is_friendly
- Tournament tier = 1 → interaction: (elo_diff × (tournament_tier==1))

Opsi C: Friendly classifier + reweight
- Binary classifier: "is this friendly predictable by Elo?"
- Untuk unpredictable friendly: turunkan confidence outcome, biarkan ERM handle
```

**Rekomendasi**: Mulai dengan Opsi B (interaction features) yang paling murah implementasinya. Jika memberi signal, lanjut ke Opsi A.

**Estimasi dampak**: **-0.02 s/d -0.05 AW-MAE**

---

## 🟡 PRIORITAS 3 (MEDIUM): Tail Handling + Ensemble Diversity

### Strategi 3.1: Generalized Pareto Distribution (GPD) untuk High-Scoring Matches

**Masalah**: 20-33% match total goals ≥5. Truncated [0,5] under-estimate kronis.

**Solusi**: Extreme Value Theory — model tail dengan GPD:

```
Step 1: Threshold selection
- Threshold u = 4 (total goals ≥4 adalah "extreme")
- Fit GPD pada excess di atas threshold: y = X - u

Step 2: GPD parameters
- ξ (shape): mengontrol tail heaviness
- σ (scale): mengontrol spread di atas threshold
- Fit MLE pada training data

Step 3: Prediksi
- Untuk setiap match yang diprediksi high-scoring:
  - P(X > u) dari classifier (Stage 2 PMF tail)
  - Untuk X > u: distribusi dari GPD
  - Gabungkan: P(X=k) = P(X≤u) * PMF(k) + P(X>u) * GPD(k-u)
```

**Estimasi dampak**: **-0.04 s/d -0.07 AW-MAE** (terutama dari tail under-estimation)

---

### Strategi 3.2: NGBoost (Natural Gradient Boosting) — Output Distribusi

**Masalah**: Semua base learner saat ini (LGB, XGB, CatBoost) adalah point estimators → harus manually membuat PMF dari raw scores → loss of information.

**Solusi**: NGBoost untuk Stage 1 dan Stage 2:

```python
from ngboost import NGBClassifier, NGBRegressor
from ngboost.distns import k_categorical

# NGBoost output distribusi probabilistik asli (bukan point estimate)
# Stage 1: Distribusi kategorikal 3-class
ngb_outcome = NGBClassifier(
    Dist=k_categorical(3),
    n_estimators=500
)

# Stage 2: Bisa output Poisson atau Normal untuk goals
# Inference langsung memberikan P(goals) yang terkalibrasi
```

**Kelebihan NGBoost**:
- Output TRUE probability distribution (bukan pseudo-probability dari LGB)
- Built-in uncertainty quantification
- Tidak perlu Platt/Isotonic calibration tambahan
- Diversifikasi ensemble — NGBoost berbeda fundamental dari LGB+XGB

**Estimasi dampak**: **-0.03 s/d -0.06 AW-MAE**

---

### Strategi 3.3: Quantile Regression untuk Prediction Interval

**Masalah**: ERM saat ini hanya mencari satu best score. Tidak ada informasi tentang UNCERTAINTY prediction.

**Solusi**: Tambahkan quantile regression:

```
Model tambahan: LightGBM quantile regression
- q10: lower bound (under-predict boundary)
- q50: median prediction
- q90: upper bound (over-predict boundary)

Kegunaan:
1. Untuk ERM: batasi pencarian skor dalam interval [q10, q90]
2. Signal untuk model: jika q10-q90 lebar → match unpredictable
3. Confidence untuk pseudo-labeling: hanya label jika interval < 2
```

**Estimasi dampak**: **-0.02 s/d -0.04 AW-MAE** (improving ERM constraints)

---

## 🔵 PRIORITAS 4 (VALIDASI): Holdout Split + Ablation Framework

### Strategi 4.1: Time-Based Validation Split

**Masalah**: Semua evaluasi langsung pada full test set → tidak bisa mendeteksi overfitting sebelum submission.

**Solusi**:

```python
# Chronological split berdasarkan tanggal match
train_raw = pd.read_csv("dataset/train.csv")
train_raw["date"] = pd.to_datetime(train_raw["date"])

# Split terakhir 15% sebagai holdout
cutoff = train_raw["date"].quantile(0.85)
train = train_raw[train_raw["date"] <= cutoff]
holdout = train_raw[train_raw["date"] > cutoff]

# Training: 85% data earlier
# Holdout: 15% data terbaru → simulasi test set
# Evaluasi di holdout HARUS korrelasi dengan full test
```

**Penggunaan**: Setiap strategi baru dievaluasi di holdout DULU sebelum full test. Menghemat waktu dan mencegah overfitting.

**Estimasi dampak**: Tidak langsung pada AW-MAE, tapi mempercepat iterasi development.

---

### Strategi 4.2: Ablation Study Framework

**Masalah**: V15 gagal karena 3 perubahan sekaligus tanpa tau mana yang broken.

**Solusi**: Framework otomatis untuk menguji perubahan komponen:

```python
# ablation_study.py
# 1. Base: V14 (baseline)
# 2. Test: V14 + hanya ubah transfer_weight → V14_TF
# 3. Test: V14 + hanya ubah temperature → V14_T
# 4. Test: V14 + hanya ubah ERM threshold → V14_ERM
# 5. Test: V14 + hanya tambah ZINB → V14_ZINB
# Seharusnya: 2-5 dievaluasi satu per satu di holdout
```

**Aturan**: JANGAN gabungkan >1 perubahan sebelum masing-masing diverifikasi independen.

---

## 📊 ROADMAP & ESTIMASI DAMPAK KUMULATIF

### Fase 1 (V17): Ordinal + Kalibrasi — Target: -0.10 AW-MAE

| Strategi | Deskripsi | Δ AW-MAE |
|----------|-----------|----------|
| 1.1 | Bivariate Ordinal Regression (6×6) | -0.10 |
| 2.1 | Isotonic Calibration Outcome | -0.05 |
| 2.2 | Soft Labeling Near-Draw | -0.03 |
| **Fase 1 total** | | **-0.18 → AW-MAE ~2.33** |

### Fase 2 (V18): Women Focus — Target: -0.05 AW-MAE

| Strategi | Deskripsi | Δ AW-MAE |
|----------|-----------|----------|
| 1.2 | ZINB untuk Women | -0.05 |
| 1.3 | Women Augmentation (Bootstrap ×3) | -0.03 |
| **Fase 2 total (kumulatif)** | | **-0.26 → AW-MAE ~2.25** |

### Fase 3 (V19): Ensemble + Tail — Target: -0.05 AW-MAE

| Strategi | Deskripsi | Δ AW-MAE |
|----------|-----------|----------|
| 3.1 | GPD Tail Modeling | -0.05 |
| 3.2 | NGBoost diversification | -0.04 |
| 2.3 | Class Weights for Draw | -0.02 |
| **Fase 3 total (kumulatif)** | | **-0.37 → AW-MAE ~2.14** |

### Fase 4 (V20): Polishing — Target: -0.03 AW-MAE

| Strategi | Deskripsi | Δ AW-MAE |
|----------|-----------|----------|
| 2.4 | Friendly Specialization | -0.03 |
| 3.3 | Quantile Regression Constraints | -0.02 |
| **Fase 4 total (kumulatif)** | | **-0.42 → AW-MAE ~2.09** |

> ⚠️ **Disclaimer**: Estimasi ini optimistic dan asumsi Δ independen (tidak overlap). Realita kemungkinan -0.20 s/d -0.35.

---

## 🧪 EKSPERIMEN CEPAT (LOW-HANGING FRUIT)

Strategi yang bisa diimplementasikan dalam <1 jam dan diuji:

### Quick Win 1: Isotonic Calibration (Strategi 2.1)
```python
# Tambah 5 baris ke V14
from sklearn.calibration import CalibratedClassifierCV
outcome_model = CalibratedClassifierCV(outcome_model, cv=5, method='isotonic')
```
**Estimasi**: -0.04 AW-MAE, 15 menit

### Quick Win 2: Soft Labeling (Strategi 2.2)
```python
# Modifikasi pembuatan label di V14
if abs(elo_diff) < 50:
    y_stage1 = [0.33, 0.34, 0.33]  # soft label
```
**Estimasi**: -0.03 AW-MAE, 10 menit

### Quick Win 3: Class Weights (Strategi 2.3)
```python
# Tambah 2 baris ke LGB
class_weight = {0: 0.74, 1: 1.33, 2: 1.11}  # home, draw, away
lgb.Dataset(X_train, label=y_train, weight=class_weight_mapped)
```
**Estimasi**: -0.02 AW-MAE, 10 menit

### Quick Win 4: Time Holdout Split (Strategi 4.1)
```python
# Split 85/15 chronologically
cutoff = train_df['date'].quantile(0.85)
```
**Estimasi**: Tidak langsung, tapi fundamental untuk semua development. 5 menit.

---

## ❌ STRATEGI YANG TIDAK DIREKOMENDASIKAN (BERDASARKAN DATA)

| Strategi | Alasan Tidak Direkomendasikan |
|----------|-------------------------------|
| **Transfer weight >0.3** | V15 buktikan weight 0.5 overfit. Data Men dan Women terlalu berbeda distribusi. |
| **Temperature >2.0** | V15 buktikan T=2.5 terlalu flat → kehilangan signal. |
| **Threshold-aware ERM** | V15: outcome accuracy turun 58.9% → 56.1%. ERM sudah optimal tanpa constraint. |
| **Rewrite arsitektur** | V16: hasil >3.0. Jangan ulangi kesalahan yang sama. |
| **FitNais / Bayesian Hierarchical dengan data ini** | Data 78K row x 100+ turnamen — hierarchical model terlalu rumit, overfit pada kelompok kecil. |
| **Deep learning (NN from scratch)** | 78K sampel terlalu kecil untuk NN murni dengan 63 fitur. TabNet masih mungkin, tapi low priority. |

---

## 📝 IMPLEMENTASI PRAKTIS: NEXT STEP

### Langsung eksekusi setelah file ini:

1. **Implementasi Quick Win 1-4 ke V14** → buat V17_quickwins.py
2. **Run di holdout split** (bukan full test dulu)
3. **Jika holdout AW-MAE turun >0.03 → run full test**
4. **Update strategy_tracker.md**

### Yang saya butuhkan dari Anda:

- Konfirmasi untuk langsung mulai implementasi V17_quickwins (Isotonic + Soft Label + Class Weight + Holdout)
- Atau instruksi spesifik strategi mana yang ingin dicoba dulu

---

## 🔖 PEMBELAJARAN DARI KEGAGALAN SEBELUMNYA

| Kegagalan | Pelajaran | Implementasi di strategi baru |
|-----------|-----------|-------------------------------|
| V15: 3 perubahan sekaligus | Ablation study | Uji satu per satu di holdout |
| V15: Transfer weight 0.5 | Jangan >0.3 | Fokus ke augmentation, bukan transfer |
| V15: T=2.5 | Jangan >2.0 | Isotonic calibration gantikan temperature |
| V16: Rewrite dari scratch | JANGAN PERNAH | Semua strategi di atas V14 |
| V12→V14: Δ hanya +0.008 | Diminishing returns | Fokus pada perubahan FUNDAMENTAL (ordinal), bukan feature |
| V12→V14: Women AW-MAE stuck | Sparsity akut | ZINB + Augmentation + Ordinal (3 jalur paralel) |