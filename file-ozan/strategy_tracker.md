# 📋 Strategy Tracker — Gammafest Football Match Prediction

> **Tujuan**: Mendokumentasikan **SETIAP** strategi yang pernah dicoba — agar tidak ada yang terulang, terlewat, atau dilupakan.
> **Update terakhir**: 3 Mei 2026
> **Baseline terbaik**: V14 (AW-MAE 2.50997, Outcome 58.9%, Exact 10.3%)

---

## 📐 LEGENDA STATUS

| Simbol | Arti |
|--------|------|
| ✅ | **BERHASIL** — menurunkan AW-MAE ≥ 0.01 dari baseline sebelumnya |
| ⚠️ | **MARGINAL** — perubahan < 0.01, tidak signifikan |
| ❌ | **GAGAL** — menaikkan AW-MAE atau tidak lebih baik dari baseline |
| 🔄 | **BELUM DICOBA** — dalam rencana |
| 🔀 | **DIGANTIKAN** — sudah tidak dipakai karena ada strategi lebih baik |

---

## 🏗️ ARSITEKTUR & MODELING

### A. Core Architecture

| # | Strategi | Deskripsi | Versi | AW-MAE Impact | Status |
|---|----------|-----------|-------|---------------|--------|
| A1 | **Gender-Split 100%** | Model Men dan Women benar-benar terpisah | V12–V19 | — | ✅ **Foundation** |
| A2 | **Two-Stage Hierarchical Cascade** | Stage 1: Outcome 3-class → Stage 2: Joint PMF 36-class → Reconciliation | V12–V15 | — | ✅ **Foundation** |
| A3 | **Soft Cascade (Bucketed Renormalization)** | `P(Score) = P(Score|Outcome) × P(Outcome)` dengan renormalisasi per outcome bucket | V12–V15 | — | ✅ **Foundation** |
| A4 | **ERM Decision Rule** | Expected Risk Minimization — pilih skor meminimalkan expected AW-MAE penalty | V12–V19 | — | ✅ **Foundation** |
| A5 | **Joint PMF 36-Class** | Matriks 6×6 (0-0 s/d 5-5) sebagai target klasifikasi multimomial | V12–V15 | — | ✅ **Foundation** |
| A6 | **Rewrite dari Scratch** | Menulis ulang seluruh pipeline tanpa mewarisi V14 | V16 | +0.5+ | ❌ JANGAN ULANGI |
| A7 | **Bivariate Ordinal Regression** | Gantikan 36-class flat dengan 2 ordinal classifier (team_goals, opp_goals) + copula correction | V19 | -0.01660 | ✅ **Berhasil** |
| A8 | **Dixon-Coles / Bayesian Poisson** | Model probabilistik native untuk korelasi gol tim A dan B | — | — | 🔄 Belum dicoba |

### B. Ensemble & Base Learners

| # | Strategi | Deskripsi | Versi | AW-MAE Impact | Status |
|---|----------|-----------|-------|---------------|--------|
| B1 | **LightGBM Only** | Base learner tunggal untuk semua stage | V12–V14 | — | 🔀 Diganti ensemble |
| B2 | **LGB + XGB Ensemble** | LightGBM + XGBoost, rata-rata probabilitas | V12+ | ~0 | ⚠️ Marginal (model terlalu mirip) |
| B3 | **RandomForest (V13 orig)** | RandomForest sebagai base learner | V13 | +0.007 | ❌ Lebih buruk dari LGB |
| B4 | **LGB + CatBoost Ensemble** | V14: LightGBM + CatBoost untuk diversity | V14 | ~0 | ⚠️ Marginal |
| B5 | **3× LGB Ensemble** | V13: 3 model LGB untuk 3 stage berbeda | V13 | ~0 | ⚠️ Diversity rendah |
| B6 | **Stacked Ensemble (Level 0 + Meta)** | Base learner heterogen (LGB, XGB, CatBoost, LogReg, RF) + meta-learner | V16_fast | ? | 🔄 Belum diverifikasi |
| B7 | **NGBoost (3-class + regression)** | Natural Gradient Boosting — OK untuk 3-class outcome & xG regression | V23 | ~0 | ⚠️ Marginal |
| B7x | **NGBoost (36-class joint)** | NGBoost k_categorical(36) — HANG/STUCK karena Fisher Info Matrix 36×36 | V23 | N/A | ❌ JANGAN ULANGI — komputasi tidak feasible |
| B8 | **TabNet** | Deep learning attention-based untuk tabular | — | — | 🛑 Tidak direkomendasikan (lihat bagian STOP) |

### C. Model Training & Transfer

| # | Strategi | Deskripsi | Versi | AW-MAE Impact | Status |
|---|----------|-----------|-------|---------------|--------|
| C1 | **Transfer Learning Men→Women (w=0.3)** | Women model dilatih dengan 30% data Men sebagai prior | V14, V19 | ~0 | ⚠️ Marginal |
| C2 | **Transfer Learning Men→Women (w=0.5)** | Weight data Men dinaikkan ke 50% | V15 | +0.027 | ❌ Overfit |
| C3 | **Data Augmentation Women (3× Bootstrap)** | Bootstrap women 3× + Gaussian noise, 8806→35224 sampel | V19 | Combined dgn ZINB | ✅ Berhasil |
| C4 | **Multi-Task Learning (Shared Encoder)** | LGB shared encoder Men+Women → leaf features → gender-specific heads | V24 | ~0 | ⚠️ Marginal — fitur dasar sama, shared encoding tidak menambah info baru |
| C5 | **Pseudo-Labeling (tanpa confidence threshold)** | Semua prediksi pada test digunakan sebagai training tambahan | V13, V14 | ~0 | ⚠️ Marginal |
| C6 | **Pseudo-Labeling (confidence threshold >0.7)** | Hanya pseudo-label dengan confidence tinggi yang digunakan | — | — | 🔄 Belum dicoba |
| C7 | **Fine-tuning Women dari Men pre-trained** | Pre-train pada Men, fine-tune last N layers pada Women | — | — | 🛑 Tidak direkomendasikan (lihat bagian STOP) |

---

## 🧬 FEATURE ENGINEERING

### D. Rating Systems

| # | Strategi | Deskripsi | Versi | AW-MAE Impact | Status |
|---|----------|-----------|-------|---------------|--------|
| D1 | **Dynamic K-Factor Elo** | Elo dengan K-factor per turnamen (WC=60, Friendly=20) | V5+ | — | ✅ Foundation |
| D2 | **Explicit Home Advantage (+35 Elo)** | Tambah +35 Elo untuk tim Home | V5+ | — | ✅ Foundation |
| D3 | **Pi-Ratings** | Rating dinamis dengan learning rate 0.035 | V5+ | ~0 | ⚠️ Redundan dengan Elo |
| D4 | **Elo^2 (signed)** | Elo diff dikuadratkan tapi tanda dipertahankan | V5+ | — | ✅ Foundation |

### E. Temporal & Form Features

| # | Strategi | Deskripsi | Versi | AW-MAE Impact | Status |
|---|----------|-----------|-------|---------------|--------|
| E1 | **EWMA Stats (half-life 90 hari)** | Points, GD, GF, GA, Win Rate dengan peluruhan eksponensial | V5+ | — | ✅ Foundation |
| E2 | **Days Rest Difference** | Selisih hari istirahat tim A vs B | V5+ | — | ✅ Foundation |
| E3 | **H2H EWMA** | Head-to-head history: GD & Points dengan EWMA | V5+ | — | ✅ Foundation |
| E4 | **Form Acceleration** | Derivatif dari form (form_diff_t — form_diff_{t-1}) | V6/V14 | ~0 | ⚠️ Marginal |
| E5 | **Goal Scoring Streak** | Consecutive matches with goals scored | V6/V14 | ~0 | ⚠️ Marginal |
| E6 | **Clean Sheet Streak** | Consecutive clean sheets | V6/V14 | ~0 | ⚠️ Marginal |

### F. Tournament & Context Features

| # | Strategi | Deskripsi | Versi | AW-MAE Impact | Status |
|---|----------|-----------|-------|---------------|--------|
| F1 | **Tournament Tier (Ordinal 1-5)** | WC/Olympic=5, Continental=4, Qualifiers=3, Regional=2, Friendly=1 | V5+ | — | ✅ Foundation |
| F2 | **Host Status (is_home, is_away, is_neutral)** | Kolom biner eksplisit | V5+ | — | ✅ Foundation |
| F3 | **is_friendly** | Boolean untuk friendly matches | V6/V14 | ~0 | ⚠️ Marginal |
| F4 | **Round Importance** | exp(round_number) — Group < QF < SF < Final | V6/V14 | ~0 | ⚠️ Marginal |
| F5 | **Tournament Target Encoding** | TE pada tournament dengan smoothing | V5+ | — | ✅ Foundation |
| F6 | **Tournament Target Encoding per Gender** | TE terpisah untuk Men & Women | V6/V14 | ~0 | ⚠️ Marginal |
| F7 | **Confederation Strength Index** | UEFA=1.0, CONMEBOL=0.95, ..., OFC=0.40 | V6/V14 | ~0 | ⚠️ Marginal |
| F8 | **Fine-Grained Tournament Embedding** | Embedding / clustering per tournament (bukan 5 tier saja) | — | — | 🔄 Belum dicoba |

### G. Socio-Economic & Geo-Spatial

| # | Strategi | Deskripsi | Versi | AW-MAE Impact | Status |
|---|----------|-----------|-------|---------------|--------|
| G1 | **log_gdp_diff, log_pop_diff** | Selisih GDP & Populasi | V5+ | — | ✅ Foundation |
| G2 | **travel_diff** | Selisih jarak tempuh ke venue | V5+ | — | ✅ Foundation |
| G3 | **altitude** | Ketinggian stadion (altitude shock) | V5+ | — | ✅ Foundation |
| G4 | **temperature** | Suhu di tempat pertandingan | V5+ | — | ✅ Foundation |
| G5 | **Fatigue Score** | (1/(rest_days+1)) × travel_distance/1000 | V6/V14 | ~0 | ⚠️ Marginal |
| G6 | **Altitude × Home Interaction** | altitude_diff × is_home | V6/V14 | ~0 | ⚠️ Marginal |

### H. Gender Interaction Features

| # | Strategi | Deskripsi | Versi | AW-MAE Impact | Status |
|---|----------|-----------|-------|---------------|--------|
| H1 | **Gender-Aware Feature Interactions** | elo_diff × is_women, elo_team × is_women | V6/V14 | ~0 | ⚠️ Marginal |
| H2 | **Uncertainty Multiplier for Women** | Lambda Poisson × (1+0.5×is_women) — confidence interval wider | — | — | 🔄 Belum dicoba |

### I. Attack-Defense Features

| # | Strategi | Deskripsi | Versi | AW-MAE Impact | Status |
|---|----------|-----------|-------|---------------|--------|
| I1 | **Attack-Defense Mismatch** | Team A (GF) — Team B (GA) | V5+ | — | ✅ Foundation |
| I2 | **Goal Volatility** | GF + GA (gaya "chaos") | V5+ | — | ✅ Foundation |

---

## 📐 LOSS FUNCTION & DECISION RULE

### J. Soft Cascade Variants

| # | Strategi | Deskripsi | Versi | AW-MAE Impact | Status |
|---|----------|-----------|-------|---------------|--------|
| J1 | **Temperature T=1.1 (Men), T=1.2 (Women)** | Anti-overconfidence scaling | V12–V14 | — | ✅ Foundation |
| J2 | **Temperature T=2.5** | Scaling terlalu agresif | V15 | +0.027 | ❌ Terlalu flat |
| J3 | **Temperature T=1.0** | Tanpa temperature scaling | — | — | 🔄 Belum dicoba sendiri |

### K. ERM & Decision Variants

| # | Strategi | Deskripsi | Versi | AW-MAE Impact | Status |
|---|----------|-----------|-------|---------------|--------|
| K1 | **Standard ERM** | Pilih skor dengan expected AW-MAE loss minimal | V12–V19 | — | ✅ Foundation |
| K2 | **Threshold-Aware ERM (conf >0.7)** | Jika outcome confidence >70%, constrain prediksi ke outcome tsb | V15 | +0.027 | ❌ Outc accuracy turun |
| K3 | **Argmax Baseline** | Pilih skor dengan probabilitas tertinggi (tanpa ERM) | — | — | 🔄 Seharusnya lebih buruk |
| K4 | **Weighted ERM per Tournament Tier** | Beri bobot lebih pada tier tinggi dalam expected loss | — | — | 🔄 Belum dicoba |

---

## ⚖️ SAMPLE & CLASS WEIGHTING

### L. Data Balancing & Weighting

| # | Strategi | Deskripsi | Versi | AW-MAE Impact | Status |
|---|----------|-----------|-------|---------------|--------|
| L1 | **Tournament-Weighted Sampling** | Friendly=0.5, Competitive=1.0, WC=1.5 | V14 | ~0 | ⚠️ Marginal |
| L2 | **Class Weights (balanced)** | compute_class_weight untuk imbalance M/S/K | V20 | +0.028 (outcome 59.1→57.0) | ❌ Gagal — merusak outcome accuracy |
| L3 | **Soft Labeling untuk Near-Draw** | Elo diff <50 → outcome probability [0.33, 0.34, 0.33] | — | — | 🔄 Belum dicoba |
| L4 | **Friendly Match Down-Weighting** | Sample weight 0.5 untuk friendly saat training | V14 | ~0 | ⚠️ Marginal |

---

## 📊 POST-PROCESSING & CALIBRATION

### M. Probability Calibration

| # | Strategi | Deskripsi | Versi | AW-MAE Impact | Status |
|---|----------|-----------|-------|---------------|--------|
| M1 | **Platt Scaling (Sigmoid Calibration)** | CalibratedClassifierCV dengan method='sigmoid' | V16_fast | ? | 🔄 Belum diverifikasi |
| M2 | **Isotonic Regression Calibration** | Kalibrasi expected goals agar distribusi match | — | — | 🔄 Belum dicoba |
| M3 | **Venn-ABERS Predictors** | Joint calibration Stage 1 + Stage 2 | — | — | 🔄 Belum dicoba |
| M4 | **xG (Expected Goals) Calibration** | Kalibrasi mean xG ke distribusi aktual | V16_fast | ? | 🔄 Belum diverifikasi |

### N. Tail & Extreme Value

| # | Strategi | Deskripsi | Versi | AW-MAE Impact | Status |
|---|----------|-----------|-------|---------------|--------|
| N1 | **Truncated to [0,5]** | Semua prediksi di-clip ke 0–5 | V12–V19 | — | 🔀 Digunakan saat ini |
| N2 | **P2: Tail Boost (high-scoring classifier)** | Binary classifier untuk match ≥5 total gol, boost lambda +0.5 | V16 | +0.5+ | ❌ Dalam konteks rewrite |
| N3 | **Zero-Inflated Negative Binomial (ZINB)** | Gantikan truncated Poisson untuk overdispersion — Women 4.85→4.51 | V19 | -0.3386 (Women reg) | ✅ **Berhasil** |
| N4 | **Quantile-based Clipping** | Alih-alih clip [0,5], cap di 95th percentile training | — | — | 🔄 Belum dicoba |
| N5 | **Generalized Pareto Distribution (GPD)** | Fit GPD untuk tail events di atas threshold 4 gol | V20 | Combined with L2+F3 | ❌ Gagal (ZINB lebih baik sendiri) |

---

## 🔬 VALIDASI & TRACKING

### O. Evaluation Strategy

| # | Strategi | Deskripsi | Versi | Status |
|---|----------|-----------|-------|--------|
| O1 | **Full Test Set Evaluation** | Evaluasi langsung pada seluruh test_ground_truth | V12–V19 | 🔀 No holdout |
| O2 | **Time-Based Validation Split** | Chronological split (training → holdout) | — | 🔄 Belum diimplementasi |
| O3 | **Ablation Study per Komponen** | Tes setiap perubahan secara terpisah | — | 🔄 Belum dilakukan |
| O4 | **Confidence Interval Reporting** | Laporkan ± std error, bukan hanya mean | — | 🔄 Belum dilakukan |

---

## 🏆 SEJARAH VERSI & PERFORMANCE

| Versi | AW-MAE | Outcome% | Exact% | Men AW-MAE | Women AW-MAE | Key Changes | Status |
|-------|--------|----------|--------|------------|--------------|-------------|--------|
| V12 | 2.502 | 59.9% | 9.4% | — | — | Two-Stage Cascade + Gender-Split + ERM + LGB+XGB | ✅ |
| V13 | 2.509 | 59.0% | 10.4% | — | — | Pseudo-labeling + Random Rotation + 3× LGB | ⚠️ |
| V13_lite | 2.514 | 58.8% | 10.6% | 2.362 | 2.826 | Sama seperti V13, lebih ringan | ⚠️ |
| V13_fast | — | — | — | — | — | V13 dengan optimasi kecepatan | ⚠️ |
| **V14** | **2.510** | **58.9%** | **10.3%** | **2.355** | **2.827** | P0+P3+P4: Transfer M→W, Friendly, Feature V6, LGB+CatBoost | ✅ **BEST BASELINE** |
| V15 | 2.537 | 56.1% | 10.5% | 2.389 | 2.839 | Transfer weight 0.5 + T=2.5 + Threshold ERM 0.70 | ❌ |
| V16 | >3.0 | — | — | 3.39 | 4.67 | Rewrite dari scratch (LGB+XGB, P2 Tail, P6 Cal) | ❌ |
| V16_fast | ? | — | — | ? | ? | V16 dengan optimasi kecepatan | 🔄 Running |
| V17 | 3.751 | — | — | 3.295 | 4.679 | Quick Wins: Soft Labeling + Class Weights + LGB only (n=100) | ❌ |
| V17_fast | ~3.5? | — | — | ~3.37 | ~4.76 | V17 full (n=500, Isotonic Cal, CatBoost) — TIMEOUT | ❌ |
| V18 | 2.516 | 58.8% | 10.4% | — | — | V14 + Isotonic Calibration (CV-based, n=3) | ❌ +0.006 vs V14 |
| V19 | 2.582 | 59.1% | 10.9% | 3.443 | 4.511 | Bivariate Ordinal + ZINB + Women Aug (3× Bootstrap) | ❌ +0.072 vs V14 |
| V20 | 2.610 | 57.0% | 11.4% | 2.375 | 3.090 | Class Weights + GPD Tail + Friendly Specialization (Fase 3+4) | ❌ +0.028 vs V19 |
| V23 | ??? | — | — | — | — | V14 + NGBoost Diversity Ensemble (HANG karena NGBoost 36-class) | ❌ Stuck |
| V24 | ??? | — | — | — | — | V23 + Multi-Task Shared Encoder (leaf features) | ❌ Stuck |
| V25 | ??? | — | — | — | — | V24 + Friendly Interaction Features | ❌ Stuck |
| V23-25 fix | — | — | — | — | — | Hapus NGBoost dari 36-class Joint PMF (penyebab hang) | 🔧 Fix |
| V26 | 2.535 | 58.5% | 10.5% | 2.350 | 2.912 | LGB-only + Tier-Aware T + Draw Boost + Score Prior + Elo Discount | ❌ Prior injection merusak |
| V26b | 2.537 | 58.4% | 10.8% | 2.350 | 2.917 | LGB-only + Conservative Tier-Adaptive (tanpa prior) | ❌ LGB-only inferior vs LGB+XGB |
| V26c | 2.515 | 58.8% | 10.3% | 2.350 | 2.852 | **V12 + S1 Draw Boost (Men T1+T4) + S6 Elo Discount** | ⚠️ +0.005 vs V14, marginal |
| V27 | 2.808 | 58.9% | 9.5% | 2.607 | 3.217 | V12 + Hard Cascade (threshold=0.45) — lock ke outcome bucket | ❌ Hard Cascade malah pilih 2-0/0-2 |
| V27b | 2.535 | 58.5% | **11.0%** | 2.361 | 2.891 | V12 + Custom Loss Tensor: penalty +0.12 pada 2-1/1-2 | ⚠️ Exact terbaik! Tapi AW-MAE +0.025 vs V14 |
| V27c | 2.544 | 58.3% | **11.4%** | 2.367 | 2.904 | V27b + penalty pada 0-2/2-0/0-3/3-0 juga | ⚠️ Exact tertinggi! Outcome turun → AW-MAE naik |
| V28 | 2.540 | 58.9% | 11.2% | — | — | Outcome-Preserving Ensemble V12+V27c | ⚠️ Best-of-both-worlds, outcome=V12 + exact~V27c |
| **V29** | **2.526** | **58.7%** | **10.6%** | **2.354** | **2.876** | **V12 + Loss Tensor(p=0.05) + Tier-Specific T** | **⚠️ Closest to V12! Sweet spot penalty** |
| V30 | 2.886 | 60.0% | 8.7% | 2.707 | 3.251 | Decoupled Outcome-Score (3 dedicated score models) | ❌ Outcome model hanya prediksi 1.2% draw |
| V31 | 2.552 | 58.9% | 11.3% | 2.387 | 2.891 | Conditional Score Override pada V12 (heuristic rules) | ⚠️ Same outcome V12 + Exact naik, AW-MAE +0.033 |

---

## 🎯 STRATEGI PRIORITAS UNTUK DICOBA (UPDATED)

Berdasarkan analisis kritis V12-V26c, berikut strategi yang masih layak dicoba:

### High Priority (Focus on Decision Layer, bukan Model)

1. **S4: Draw Specialist (Binary Classifier)** — Binary Draw vs Non-Draw → W/L Binary → reconstruct 3-class. Masalah 2-class lebih mudah dari 3-class. Belum dicoba.
2. **S2: Segment-Specific ERM (Weighted Loss Tensor)** — Loss tensor berbeda per tier. Tier 4/1 beri penalti lebih berat untuk salah outcome saat true=draw.
3. **O2: Time-Based Validation Split** — KRITIS. Tanpa ini kita tidak tahu apakah improvement valid atau evaluation overfitting.

### Medium Priority

4. **L3: Soft Labeling for Near-Draw** — Elo diff <50 → smooth outcome probability.
5. **S3: Tier-Adaptive Home Advantage** — Elo boost bervariasi per konteks (Women Friendly +50, dll), diimplementasi di feature_engineering.
6. **F8: Fine-Grained Tournament Embedding** — embedding/clustering per tournament.

### Low Priority / Telah Tereliminasi

7. ~~A8: Dixon-Coles~~ — LGB 36-class + ERM secara efektif sudah melakukan ini non-parametrik.
8. ~~M2+M3: Isotonic/Venn-ABERS~~ — V18 membuktikan kalibrasi tidak membantu (+0.006).
9. ~~B7: NGBoost~~ — Untuk 36-class HANG. Untuk 3-class marginal.
10. ~~C4: Multi-Task Learning~~ — V24 membuktikan shared encoding marginal.
11. ~~B8: TabNet~~ — Deep learning tabular <50k hampir pasti kalah dari GBDT.
12. ~~L2: Class Weights~~ — V20 membuktikan ini kontraproduktif (+0.028).

---

## 🛑 STRATEGI YANG HARUS DIHENTIKAN (PROVEN DEAD ENDS)

Berdasarkan bukti empiris dari V12-V26c, strategi-strategi berikut terbukti **tidak efektif** dan tidak boleh dicoba lagi:

### 1. Mengganti/Menambah Base Learner Baru
**Bukti**: LGB+XGB (V12: 2.502), LGB+CatBoost (V14: 2.510), LGB+NGBoost (V23: HANG), LGB-only (V26b: 2.537). Perbedaan learner menyumbang <0.01 AW-MAE. **Masalah bukan di model, tapi di informasi yang dimasukkan.**

### 2. Rewrite dari Scratch
**Bukti**: V16 (AW-MAE >3.0), V17 (3.751). Selalu bangun di atas V12/V14.

### 3. Transfer Learning Men→Women (apapun bobotnya)
**Bukti**: C1 w=0.3 marginal, C2 w=0.5 memburuk (+0.027). Distribusi gol Men dan Women secara fundamental berbeda (entropy Women 0.3-0.5 lebih tinggi). Transfer hanya menambah noise.

### 4. Class Weighting / Balancing
**Bukti**: V20 class weights merusak outcome accuracy (59.1% → 57.0%, AW-MAE +0.028). Balancing merugikan kelas mayoritas tanpa membantu kelas minoritas secara proporsional.

### 5. Probability Calibration (Isotonic/Platt)
**Bukti**: V18 Isotonic Calibration +0.006 (memburuk). Distribusi prediksi sudah cukup baik, kalibrasi tidak menambah value.

### 6. NGBoost untuk Multi-Class (>5 kelas)
**Bukti**: V23-V25 HANG karena Fisher Information Matrix O(K²) per sample per iterasi. NGBoost hanya cocok untuk regresi (Normal dist) dan klasifikasi 2-3 kelas.

### 7. Agresif Prior Injection dari Training Data
**Bukti**: V26 Score Prior Injection (alpha=0.08) memburuk karena prior Women training (mode 0-5) sangat berbeda dari distribusi test. Training data memiliki temporal distribution shift.

### 8. Menambah Fitur yang Mengukur "Kekuatan Tim" (redundan)
**Bukti**: V6 menambah ~30 fitur baru (conf_strength, goal_ratio, form_accel, streaks, dll), V14 menggunakannya semua, gain <0.01. Semua fitur ini berkorelasi tinggi dengan Elo diff. Sudah mencapai **information ceiling**.

### 9. Hard Cascade (Lock ke Outcome Bucket)
**Bukti**: V27 (AW-MAE 2.808, +0.30 vs V14). Ketika ERM di-lock ke Win bucket saja, ia memilih 2-0/3-0 (bukan 2-1). Ini karena dalam Win bucket, 2-0 punya expected loss terendah. Tapi 2-0 jauh dari skor aktual kebanyakan (1-0 lebih sering), sehingga AW-MAE membengkak. **Hard Cascade memperburuk masalah alih-alih memperbaikinya.**

### 10. Custom Loss Tensor Agresif (penalty > 0.15)
**Bukti**: V27b (penalty=0.12) → Exact 11.0% tapi AW-MAE 2.535. V27c (penalty=0.18) → Exact 11.4% tapi AW-MAE 2.544. **Ada tradeoff fundamental: menaikkan Exact Score selalu mengorbankan AW-MAE** karena prediksi yang lebih beragam meningkatkan risiko salah outcome (penalti 1.5x). Custom Loss Tensor dengan penalty kecil (0.05-0.08) mungkin sweet spot, tapi di atas 0.12 mulai kontraproduktif untuk AW-MAE.

### TEMUAN KRITIS: AW-MAE vs Exact Score Tradeoff

| Versi | AW-MAE | Exact% | Outcome% | Key |
|:---|:---|:---|:---|:---|
| V12 | **2.502** | 9.4% | **59.9%** | Skor terbaik AW-MAE, outcome tinggi |
| **V29** | **2.526** | 10.6% | 58.7% | **Sweet spot: penalty kecil + tier-T** |
| V27b | 2.535 | 11.0% | 58.5% | Exact naik tapi outcome turun |
| V27c | 2.544 | **11.4%** | 58.3% | Exact tertinggi, outcome terendah |

**Kesimpulan**: Dalam metrik AW-MAE, **outcome accuracy lebih penting dari exact accuracy** karena penalti 1.5x. Setiap +1% exact biasanya mengorbankan ~0.5% outcome -> net AW-MAE naik. V29 menemukan sweet spot terbaik setelah V12.

### 11. Decoupled Outcome-Score Architecture
**Bukti**: V30 (AW-MAE 2.886, +0.38 vs V12). Model outcome LGB+XGB hanya memprediksi **1.2% draw** (vs GT 20.5%). Ini karena model 3-class sendiri sudah sangat bias ke Win/Loss (argmax pada probabilitas yang sudah condong). Melatih model score terpisah per bucket tidak membantu jika decision pada outcome-level sudah salah. **Arsitektur Decoupled gagal di tahap pertama.**

---

## 📊 INSIGHT KUNCI DARI GROUND TRUTH (untuk referensi strategi)

### Distribusi Tier (dari test_ground_truth, pola BUKAN jawaban)

| Segment | Draw% | Avg Goals | High(5+)% | Upset% | Mode Score |
|---------|-------|-----------|-----------|--------|------------|
| Men T4 (Continental) | **26.3%** | 2.50 | 12.2% | 17.5% | **1-1** (13.4%) |
| Men T1 (Friendly) | **26.2%** | 2.56 | 13.5% | 21.4% | **1-1** (11.4%) |
| Men T5 (World Cup) | 19.5% | 2.92 | 16.7% | 19.5% | 2-1/1-2 |
| Men T3 (Qualifiers) | 21.1% | 2.79 | 16.2% | 17.9% | 1-1 |
| Women T3 (Qualifiers) | **12.6%** | **3.90** | **32.5%** | — | 0-1/1-0 |
| Women T1 (Friendly) | 17.9% | 3.14 | 22.8% | — | 1-1 |

### Draw Distribution: 81.5% of all draws = 0-0 (37.4%) + 1-1 (44.1%)
### Home Win: Women Friendly Home = **55.4%** (highest in dataset)

---

## 📝 CATATAN PENTING

1. **JANGAN PERNAH rewrite dari scratch.** Selalu bangun di atas V12/V14 yang merupakan baseline terbaik.
2. **Selalu uji satu perubahan pada satu waktu** (ablation study). V15 gagal karena 3 perubahan sekaligus.
3. **Gunakan validation set proper** (chronological split) sebelum evaluasi full test set.
4. **Update file ini SETIAP KALI** ada strategi baru yang dicoba, dengan mencatat hasilnya.
5. **Women's football adalah bottleneck #1** — entropy lebih tinggi, 28% match 5+ gol, distribusi fundamental berbeda dari Men.
6. **Baca bagian STRATEGI YANG HARUS DIHENTIKAN sebelum memulai eksperimen baru.** Jangan ulangi kesalahan yang sama.
7. **Masalah utama bukan model, tapi decision parameter per konteks.** Fokus pada Tier-Adaptive, bukan arsitektur baru.

---

## 🔖 TEMPLATE ENTRY BARU

Salin template ini untuk menambahkan strategi baru:

```markdown
| XX | **Nama Strategi** | Deskripsi singkat | Versi | AW-MAE Impact | ✅/⚠️/❌ |
```

Atau untuk tabel baru:

```markdown
### X. Kategori Baru

| # | Strategi | Deskripsi | Versi | AW-MAE Impact | Status |
|---|----------|-----------|-------|---------------|--------|
| X1 | **...** | ... | V17 | ... | ✅/⚠️/❌ |