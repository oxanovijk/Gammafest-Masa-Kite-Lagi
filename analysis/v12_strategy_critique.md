# 🔬 Objective Critique: Strategi V12 (Two-Stage Hierarchical Soft-Cascade + Gender-Split)

> **Tanggal**: 1 Mei 2026  
> **State-of-the-Art Saat Ini**: AW-MAE ~2.502 (local), Outcome Acc 59.88%, Exact Score Acc 9.38%  
> **Target**: Menembus batas AW-MAE dan mengatasi Distribution Shift di Kaggle Leaderboard

---

## 📊 Ringkasan Strategi V12

| Komponen | Deskripsi |
|---|---|
| **Split Gender** | 100% terpisah: model Men dan Women masing-masing punya pipeline sendiri |
| **Stage 1** | LightGBM + XGBoost → prediksi Outcome (3-class: Win/Draw/Loss) |
| **Stage 2** | LightGBM + XGBoost → prediksi Joint PMF (36-class: matriks gol 6×6) |
| **Reconciliation** | `P(Score) = P(Score\|Outcome) × P(Outcome)` — bucketed renormalization |
| **Decision Rule** | ERM (Expected Risk Minimization), bukan argmax — memilih skor dengan penalty AW-MAE terendah |
| **Temperature** | T=1.1 (Men), T=1.2 (Women) — anti-overconfidence |
| **Fitur** | 40+ (Dynamic Elo, Pi-Ratings, EWMA, Tournament Tier, Geo-Socio, Target Encoding) |

---

## ✅ Kekuatan (What You Got Right)

### 1. Gender-Split adalah Keputusan Tepat dan Kritis
Data analysis menunjukkan perbedaan fundamental:
- **Volatilitas gol**: Std dev Women 2.58 vs Men 1.66 (57% lebih tinggi)
- **Distribusi Draw**: Women 13.0% vs Men 22.6% (perbedaan 9.6pp)
- **High-scoring**: Women 33.1% vs Men 20.1%
- **Distribution shift di test**: Women naik dari 11% → 33%

Satu model tunggal untuk kedua gender akan menghasilkan bias sistematis — under-predict gol untuk women dan over-predict draw. Keputusan split 100% adalah **satu-satunya cara benar** menangani perbedaan distribusi ini.

**Skor**: ★★★★★ (5/5)

---

### 2. Two-Stage Architecture Sangat Cocok dengan AW-MAE
Ini adalah insight paling brilian dari arsitektur V12. Metrik AW-MAE memberikan penalti **1.5x multiplier** jika Outcome salah. Dengan memisahkan prediksi Outcome (Stage 1) dari Exact Score (Stage 2), model bisa:

- Fokus pada **menghindari penalti terbesar** (outcome) terlebih dahulu
- Lalu mengoptimalkan exact score **dalam batasan outcome yang sudah diprediksi**
- Reconciliation memastikan konsistensi: jika model yakin team menang, distribusi skor harus memihak team tersebut

Ini jauh lebih elegant daripada pendekatan single-stage yang mencampur semua objective.

**Skor**: ★★★★★ (5/5)

---

### 3. ERM Decision Rule Lebih Superior dari Argmax
Dalam metrik AW-MAE, tidak semua error sama beratnya. Memilih skor dengan probabilitas tertinggi (argmax) tidak memperhitungkan **biaya asimetris** dari penalti — salah outcome didenda 1.5x, salah exact score didenda 0.30, salah GD didenda 0.15. ERM secara eksplisit menghitung expected loss untuk setiap kandidat skor dan memilih yang meminimalkan total expected penalty.

Ini adalah keputusan yang **benar secara matematis** dan jarang dilakukan kompetitor.

**Skor**: ★★★★★ (5/5)

---

### 4. Tournament Tier Ordinal (1–5) Menangkap Nuansa Penting
Turnamen bukan sekadar label kategorikal — ada hierarki prestise yang mempengaruhi perilaku tim:
- **Tier 5 (World Cup)**: Tim bermain konservatif, skor rendah, pertahanan ketat
- **Tier 1 (Friendly)**: Rotasi pemain, eksperimen taktik, hasil tak terduga

Encoding ordinal 1–5 lebih informatif daripada one-hot encoding yang akan menghasilkan 100+ kolom sparse.

**Skor**: ★★★★☆ (4/5)

---

### 5. Feature Engineering Komprehensif dan Anti-Leakage
40+ fitur yang mencakup:
- Rating dinamis (Elo, Pi-Ratings) — fundamental strength
- Rolling EWMA (half-life 90 hari) — momentum & form
- Geo-sosio (altitude, GDP, travel, temperature) — contextual factors
- H2H history — matchup-specific dynamics
- Target encoding — smooth prior

Semua fitur dihitung **tanpa data leakage** (hanya dari data sebelum tanggal pertandingan).

**Skor**: ★★★★☆ (4/5)

---

## ❌ Kelemahan (What's Holding You Back)

### 1. Data Women Sangat Sedikit untuk 36-Class Classification
**Ini adalah bottleneck paling serius.**

| Metrik | Men | Women |
|---|---|---|
| Training rows | 69,966 | 8,806 |
| Kelas target (PMF) | 36 | 36 |
| Rata-rata sample/kelas | ~1,943 | **~245** |
| Outlier: kelas paling jarang | <50 | **<5** |

Dengan hanya 8,806 sampel untuk mempelajari 36 kelas, model Women menderita **severe data sparsity**. Banyak sel matriks 6×6 (misalnya skor 5-4, 4-5) hampir tidak memiliki contoh di training. Akibatnya:
- Model Women **tidak bisa belajar** distribusi skor langka dengan baik
- Temperature scaling (T=1.2) hanya masking, bukan menyelesaikan akar masalah
- Saat test set memiliki 33% women, error dari model Women mendominasi total AW-MAE

**Dampak estimasi**: 0.15–0.30 poin AW-MAE (signifikan)

---

### 2. Independensi Stage 1 dan Stage 2 Menimbulkan Inkonsistensi Kalibrasi
Reconciliation formula `P(Score) = P(Score|Outcome) × P(Outcome)` mengasumsikan Stage 1 dan Stage 2 terkalibrasi sempurna. Kenyataannya:

- Stage 1 (Outcome) di-train dengan loss function classification (multi_logloss)
- Stage 2 (Score PMF) di-train dengan loss function classification (multi_logloss)
- Output probabilitas dari keduanya **tidak terkalibrasi** satu sama lain
- LightGBM cenderung menghasilkan probabilitas overconfident (terutama dengan banyak trees)

Bucketed renormalization membantu, tapi ini adalah **post-hoc hack**, bukan solusi struktural. Yang diperlukan adalah **joint calibration** yang memastikan:
```
Σ_{scores in outcome class} P(Score) = P(Outcome)
```
terpenuhi secara natural, bukan dipaksakan.

---

### 3. Outcome Accuracy 59.88% Masih Rendah
Angka ini berarti **40% prediksi terkena 1.5x penalty multiplier**. Di sinilah sebagian besar AW-MAE hilang.

Analisis lebih dalam: Outcome accuracy 62.17% di Tier 3 (Qualifiers) tapi lebih rendah di Tier 1 (Friendly) dan Tier 5 (World Cup). Artinya:
- Model belum bisa membedakan kapan tim kuat serius vs rotasi
- Friendly matches tetap menjadi "source of randomness" terbesar
- World Cup matches (high stakes) memiliki pola berbeda yang mungkin under-represented di training

Setiap 1% peningkatan outcome accuracy bisa mengurangi AW-MAE sekitar **0.04–0.06 poin** karena menghindari 1.5x multiplier.

---

### 4. Model LightGBM + XGBoost Saja Mungkin Tidak Cukup untuk Menangkap Interaksi Kompleks
Tree-based models bagus untuk:
- Non-linear threshold (altitude shock, elo gap besar)
- Feature interactions (home × elo, tournament × elo)

Tapi buruk untuk:
- **Low-rank structure**: Hubungan antar tim bisa direpresentasikan sebagai matriks embedding (tim_i × tim_j)
- **Smooth interpolation**: Tree model membuat prediksi piecewise-constant, tidak smooth
- **Uncertainty quantification**: Tidak ada confidence interval native

Dixon-Coles / Bayesian models secara natural menangkap struktur kovarians antar gol tim A dan tim B — sesuatu yang 36-class flat classification tidak lakukan secara eksplisit. PMF 36-class memperlakukan setiap sel matriks independen, padahal ada struktur spatial: gol(2,1) dan gol(1,2) berkorelasi.

---

### 5. Tidak Ada Mekanisme Test-Time Adaptation
Anda sudah mengidentifikasi distribution shift (proporsi women berubah, kemungkinan pola turnamen berbeda di private LB). Tapi model V12 tidak memiliki mekanisme untuk **beradaptasi** dengan distribusi test:

- Tidak ada pseudo-labeling atau self-training
- Tidak ada domain adaptation / importance weighting
- Tidak ada estimasi seberapa "out-of-distribution" sebuah prediksi

Akibatnya, model bisa sangat confident tapi salah pada sampel yang berbeda distribusinya.

---

### 6. Tournament Tier Terlalu Kasar
5 tier mengelompokkan ratusan turnamen berbeda. Contoh masalah:
- "AFF Championship" dan "Baltic Cup" sama-sama Tier 2 (Regional), tapi pola skornya mungkin sangat berbeda
- "FIFA World Cup qualification" (Tier 3) memiliki tingkat kompetitif yang berbeda antar konfederasi (UEFA qualifiers > OFC qualifiers)
- Beberapa turnamen memiliki format khusus (two-leg aggregate, group stage vs knockout) yang mempengaruhi strategi tim

Target encoding pada tournament membantu, tapi dengan 8,806 wanita vs 69,966 pria, TE untuk turnamen langka tidak reliable.

---

### 7. Pi-Ratings dan Elo Mungkin Redundant
Kedua sistem rating memiliki tujuan mirip: mengukur kekuatan tim. Dengan learning rate Pi-Ratings yang kecil (0.035), keduanya menghasilkan signal yang sangat berkorelasi. Ini bisa menyebabkan:
- **Multicollinearity** yang tidak masalah untuk prediksi tapi membuat feature importance tidak interpretable
- **Redundansi** — salah satu bisa di-drop tanpa kehilangan akurasi
- **Noise amplification** — error di salah satu sistem rating merambat ke prediksi

---

## 🔧 Rekomendasi Strategi Perbaikan

### Prioritas P0 (Kritis — Dampak Terbesar)

#### A. Transfer Learning dari Model Men ke Women
Mengingat data Women hanya 8,806 baris, manfaatkan 69,966 baris Men:

1. **Pre-train** model Stage 1 & Stage 2 pada data Men (tanpa gender)
2. **Fine-tune** pada data Women dengan learning rate lebih rendah
3. Atau: **Multi-task learning** — model shared encoder dengan head terpisah per gender

**Justifikasi**: Pola fundamental sepak bola (home advantage, form momentum, Elo correlation dengan hasil) berlaku universal lintas gender. Yang berbeda adalah skala (lebih banyak gol di women). Transfer learning memungkinkan model Women "meminjam" pengetahuan dari data Men yang 8x lebih banyak.

**Estimasi dampak**: -0.10 hingga -0.20 AW-MAE (dari perbaikan model Women)

---

#### B. Reformulasi Stage 2 sebagai Bivariate Ordinal Regression
Alih-alih 36-class flat classification, gunakan pendekatan yang menghormati struktur terurut gol:

1. **Ordinal logistic regression** untuk team_goals (0–5+) dan opp_goals (0–5+) secara terpisah
2. **Copula** untuk memodelkan korelasi residual antar dua dimensi gol
3. Atau: **Bivariate Poisson** dengan Dixon-Coles correction sebagai baseline, lalu LGBM sebagai meta-learner untuk residual

**Justifikasi**: Skor gol bersifat ordinal, bukan kategorikal. Skor 2-1 lebih "dekat" ke 2-0 daripada ke 0-2. Flat 36-class tidak menangkap proximity ini, sehingga model harus belajar dari nol bahwa (2,1) mirip dengan (2,0). Ordinal structure memberikan inductive bias yang mempercepat pembelajaran — kritis untuk data Women yang sparse.

**Estimasi dampak**: -0.08 hingga -0.15 AW-MAE (terutama dari sel-sel matriks yang jarang)

---

### Prioritas P1 (Tinggi)

#### C. Joint Calibration dengan Venn-ABERS Predictors
Gantikan reconciliation post-hoc dengan kalibrasi probabilistik yang ketat:

1. **Inductive Conformal Prediction** untuk mengkalibrasi probabilitas Outcome dari Stage 1
2. **Venn-ABERS** untuk mengkalibrasi probabilitas Score dari Stage 2
3. **Joint calibration** dengan constraint `Σ_{s ∈ outcome} P(s) = P(outcome)` dijamin secara matematis

**Justifikasi**: LightGBM menghasilkan probabilitas yang tidak terkalibrasi. Di metrik AW-MAE, probabilitas overconfident pada outcome yang salah sangat mahal (1.5x multiplier). Kalibrasi yang lebih baik → ERM decision rule bekerja lebih optimal → outcome accuracy naik.

**Estimasi dampak**: -0.05 hingga -0.10 AW-MAE

---

#### D. Fine-Grained Tournament Embedding + Cluster
Gantikan ordinal tier 1–5 dengan:

1. **Tournament-level target encoding** dengan strong smoothing (bayesian shrinkage)
2. **K-Means clustering** pada statistik historis per tournament (rata-rata gol, %draw, %home win)
3. **Embedding layer** (jika menggunakan NN) untuk setiap tournament

**Justifikasi**: Turnamen adalah proxy untuk "gaya bermain kolektif" dalam konteks tertentu — bukan hanya tingkat kepentingan, tapi juga karakteristik gol, agresivitas, dan pola hasil. FIFA World Cup dan Olympic sama-sama Tier 5 tapi memiliki karakteristik berbeda (Olympic = U-23, World Cup = senior).

**Estimasi dampak**: -0.03 hingga -0.07 AW-MAE

---

### Prioritas P2 (Medium)

#### E. Test-Time Pseudo-Labeling untuk Private LB
Setelah submission ke Kaggle mendapatkan public LB score:

1. Gunakan prediksi model V12 pada test set sebagai **pseudo-label**
2. Train ulang model dengan data train + test (pseudo-labeled) dengan bobot
3. Iterasi beberapa kali hingga konvergen

**Justifikasi**: Private test set memiliki distribusi berbeda. Pseudo-labeling adalah bentuk semi-supervised domain adaptation yang bisa membantu model menyesuaikan diri ke distribusi target tanpa melanggar aturan no-leakage (karena tidak menggunakan data eksternal).

**Estimasi dampak**: -0.05 hingga -0.12 AW-MAE (sulit diprediksi, tergantung seberapa besar distribution shift)

---

#### F. Diversity Ensemble Expansion
Tambahkan base learner yang **berbeda secara fundamental**:

1. **TabNet** (deep learning untuk tabular, attention-based)
2. **CatBoost** (ordered boosting, menangani categorical features lebih baik)
3. **NGBoost** (Natural Gradient Boosting — output distribusi, bukan point estimate)

**Justifikasi**: LightGBM + XGBoost adalah dua model gradient boosting yang mirip secara prinsip. Diversifikasi dengan attention-based (TabNet) dan probabilistic (NGBoost) mengurangi korelasi error antar base learner — meningkatkan gains dari stacking.

**Estimasi dampak**: -0.02 hingga -0.05 AW-MAE

---

### Prioritas P3 (Low — Eksperimen)

#### G. Bayesian Hierarchical Model sebagai Meta-Learner
Di atas semua prediksi base model, gunakan **Bayesian Hierarchical Model**:
- Group-level effects: gender, tournament, confederation
- Individual-level effects: specific team matchups
- Partial pooling untuk menangani kelompok dengan data sedikit (women, small confederations)

#### H. Konversi ke Regresi + Threshold
Daripada 36-class classification, prediksi **expected goals** (continuous) untuk masing-masing tim, lalu konversi ke discrete dengan threshold optimal yang di-tune terhadap AW-MAE.

---

## 📈 Proyeksi Dampak Kumulatif

| Prioritas | Strategi | Estimasi Δ AW-MAE |
|---|---|---|
| **P0** | Transfer Learning M→W | -0.10 s/d -0.20 |
| **P0** | Bivariate Ordinal Regression | -0.08 s/d -0.15 |
| **P1** | Joint Calibration (Venn-ABERS) | -0.05 s/d -0.10 |
| **P1** | Fine-Grained Tournament | -0.03 s/d -0.07 |
| **P2** | Pseudo-Labeling | -0.05 s/d -0.12 |
| **P2** | Diversity Ensemble | -0.02 s/d -0.05 |
| **Total (best case)** | | **-0.33 s/d -0.69** |
| **Target AW-MAE** | | **~1.80 s/d 2.17** |

> ⚠️ Estimasi bersifat proyeksi berdasarkan analisis bottleneck. Realita akan bergantung pada seberapa banyak improvement yang independen (tidak overlapping) dan seberapa akurat diagnosis distribution shift.

---

## 🎯 Kesimpulan Objektif

### Yang Sudah Benar (Jangan Diubah)
1. **Gender-Split** — foundation yang solid, dipertahankan
2. **Two-Stage Architecture** — tepat untuk AW-MAE, dipertahankan
3. **ERM Decision Rule** — optimal secara matematis, dipertahankan
4. **Feature Engineering V5** — komprehensif, minor additions only

### Yang Perlu Diperbaiki (Urut Prioritas)
1. **Data sparsity Women** → Transfer Learning dari model Men
2. **Flat 36-class PMF** → Bivariate Ordinal / struktur terurut
3. **Kalibrasi probabilitas** → Venn-ABERS / Conformal Prediction
4. **Tournament modeling** → Lebih granular dari 5 tier
5. **Distribution shift** → Pseudo-labeling / domain adaptation
6. **Ensemble diversity** → Tambah TabNet, CatBoost, NGBoost

### Verdict
Strategi V12 adalah fondasi yang **sangat solid** dan **arahnya benar**. Bottleneck saat ini bukan arsitektur yang salah, melainkan **eksekusi pada edge cases**: women data yang sparse, probabilitas yang tidak terkalibrasi, dan distribution shift. Fokus perbaikan seharusnya pada **memperkuat komponen terlemah** (model Women + kalibrasi) daripada merombak arsitektur yang sudah terbukti bekerja.

---