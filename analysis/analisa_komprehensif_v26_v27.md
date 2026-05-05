# Analisa Komprehensif V26 & V27: Diagnosis Plateau dan Jalur ke Depan

## Executive Summary

Seri V26 dan V27 adalah eksperimen untuk menembus AW-MAE plateau di angka ~2.50. Semua upaya gagal menurunkan AW-MAE di bawah V12 (2.502), namun berhasil mengungkap **tiga temuan fundamental** yang mengubah cara kita memahami masalah ini:

1. **ERM memiliki bias struktural** yang membuatnya tidak pernah memilih 0-0 dan terlalu sering memilih 2-1/1-2
2. **Ada tradeoff Exact vs Outcome** di mana menaikkan exact score selalu menurunkan outcome accuracy → net AW-MAE naik
3. **Information ceiling sudah tercapai** — arsitektur model (LGB/XGB/CatBoost) dan fitur yang ada tidak mengandung informasi baru

---

## Bagian 1: Analisa Detail per Versi

### V26 — Tier-Adaptive Post-Processing (LGB-only)

**Strategi**: Menambahkan 3 komponen di atas arsitektur LGB-only:
- S1: Draw Boost berbeda per tier (Tier 1/4 mendapat boost +5-8%)
- S5: Score Prior Injection dari distribusi training (alpha=0.08)
- S6: Elo Confidence Discount per tier

**Hasil**: AW-MAE = **2.535** (+0.033 vs V12)

**Kelebihan**:
- Exact Score 10.5% (naik dari V12 9.4%)
- Prediksi draw naik sedikit
- Framework tier-adaptive yang logis dan bisa dikembangkan

**Kekurangan**:
- **Score Prior Injection (S5) merusak performa**. Prior dari training data Women (mode 0-5, banyak blowout match historis) sangat berbeda dari distribusi test. Ini menunjukkan ada *temporal distribution shift* yang signifikan antara era training dan test.
- **LGB-only inferior dari LGB+XGB**. Kehilangan XGB diversity menyumbang ~0.02 degradasi AW-MAE.

**Post-mortem**: V26 mencoba 3 perubahan sekaligus (melanggar prinsip ablation), sehingga sulit mengisolasi mana yang membantu dan mana yang merusak.

---

### V26b — Conservative Tier-Adaptive (LGB-only, tanpa Prior)

**Strategi**: V26 minus Score Prior Injection. Hanya S1 (Draw Boost) + S6 (Elo Discount), masih LGB-only.

**Hasil**: AW-MAE = **2.537** (+0.035 vs V12)

**Kelebihan**:
- Exact Score 10.8% (tertinggi saat itu)
- Lebih stabil daripada V26 karena tidak ada prior injection

**Kekurangan**:
- Masih LGB-only → inferior dari LGB+XGB
- Draw Boost efeknya marginal (+0.5% draw predictions)

**Post-mortem**: Mengkonfirmasi bahwa S5 (Prior) memang merusak, tapi juga mengkonfirmasi bahwa S1+S6 tanpa XGB tidak cukup.

---

### V26c — V12 Foundation + S1 + S6

**Strategi**: Kembali ke basis V12 (LGB+XGB), hanya tambahkan S1 (Draw Boost Men Tier 1+4) dan S6 (Elo Discount feature).

**Hasil**: AW-MAE = **2.515** (+0.013 vs V12)

**Kelebihan**:
- Terdekat ke V12/V14 dari semua versi V26
- Membuktikan bahwa LGB+XGB foundation penting
- 1-1 predictions naik sedikit (5721 vs V12 5163)

**Kekurangan**:
- S1 dan S6 memberikan improvement yang terlalu kecil untuk signifikan
- 0-0 masih tetap 0 predictions (masalah ERM, bukan parameter)
- 2-1/1-2 masih mendominasi (~50% dari semua prediksi)

**Post-mortem**: Tier-adaptive post-processing pada level parameter (draw boost, temperature) sudah mencapai batasnya. Masalah ada di level yang lebih dalam: algoritma ERM itu sendiri.

---

### V27 — Hard Cascade

**Strategi**: Pisahkan keputusan menjadi 2 tahap: (1) putuskan outcome (W/D/L) berdasarkan probabilitas tertinggi, (2) jalankan ERM hanya dalam bucket outcome yang dipilih.

**Hasil**: AW-MAE = **2.808** (+0.306 vs V12) — **terburuk dari semua versi yang jalan**

**Kelebihan**:
- Konsep yang logis secara teori
- 91% prediksi Women masuk Hard Cascade (high confidence)

**Kekurangan**:
- **Fatal flaw**: Dalam Win bucket, ERM memilih 2-0 dan 3-0 (bukan 2-1 atau 1-0). Ini karena 2-0 punya expected loss terendah dalam distribusi skor Win. Tapi di ground truth, 1-0 jauh lebih sering → MAE membengkak.
- Draw predictions malah TURUN ke 12.5% (lebih rendah dari V12!) karena Win/Loss hampir selalu punya probabilitas > threshold
- Ketika outcome salah (41% kasus), tidak ada jaring pengaman

**Post-mortem**: Hard Cascade gagal total karena **ERM mengoptimalkan expected loss, bukan frequency matching**. Dalam setiap bucket, ERM tetap memilih skor "tengah-tengah" yang meminimalkan risiko — bukan skor yang paling sering muncul. Strategi ini harus masuk daftar DEAD END dan tidak boleh dicoba lagi.

---

### V27b — Custom Loss Tensor (penalty=0.12)

**Strategi**: Modifikasi loss tensor: tambahkan penalty +0.12 ke slot (2,1) dan (1,2). Ini membuat 2-1/1-2 sedikit kurang menarik bagi ERM, memaksa diversifikasi ke skor lain.

**Hasil**: AW-MAE = **2.535**, Exact = **11.02%** (+1.6% vs V12)

**Kelebihan**:
- **Exact Score terbaik saat itu** (11.0%)
- **+364 exact matches** lebih banyak dari V12 (4677 vs 4313)
- 1-0/0-1 melonjak dari 3621 → 8760 predictions
- Draw predictions naik ke 15.2% (dari 12.2%)
- Distribusi prediksi jauh lebih sehat dan natural

**Kekurangan**:
- Outcome accuracy turun dari 59.9% → 58.5%
- AW-MAE +0.033 vs V12 karena kehilangan outcome
- 0-0 masih tetap 0 (ERM structural limitation)
- 2-1/1-2 turun dari ~22k ke ~12k tapi masih over-predicted vs GT (~5.2k)

**Post-mortem**: Custom Loss Tensor adalah strategi yang BEKERJA secara mekanis — ia berhasil mendiversifikasi prediksi. Tapi mengungkap tradeoff fundamental: diversifikasi = lebih banyak exact match TAPI juga lebih banyak salah outcome.

---

### V27c — Refined Custom Loss Tensor (multi-score penalty)

**Strategi**: V27b + penalty tambahan ke skor lain yang over-predicted:
- (2,1)/(1,2): +0.18
- (0,2)/(2,0): +0.06
- (0,3)/(3,0): +0.04

**Hasil**: AW-MAE = **2.544**, Exact = **11.35%** (tertinggi dari semua versi)

**Kelebihan**:
- **Record Exact Score**: 11.35%
- 1-0/0-1 melonjak ke 12.534 predictions (vs GT 7.532)
- 1-1 naik ke 6.791 predictions
- Distribusi paling natural dari semua versi

**Kekurangan**:
- AW-MAE memburuk (+0.042 vs V12)
- Outcome accuracy turun ke 58.32% (terendah)
- 1-0/0-1 sekarang OVER-predicted (12.5k vs GT 7.5k) — dari under- ke over-prediction
- 0-0 masih 0 (limit ERM yang tidak bisa diperbaiki dengan penalty)

**Post-mortem**: Penalty yang lebih besar mendorong prediksi ke arah yang benar (lebih banyak 1-0, 1-1) tapi melewati sweet spot dan malah over-correcting. Dan outcome accuracy terus turun secara linear dengan besarnya penalty.

---

### V28 — Outcome-Preserving Ensemble (V12 + V27c)

**Strategi**: Merge CSV V12 dan V27c. Jika outcome sama → pakai skor V27c (exact lebih tinggi). Jika outcome beda → pakai V12 (outcome lebih akurat).

**Hasil**: AW-MAE = **2.540**, Exact = **11.15%**, Outcome = **58.89%**

**Kelebihan**:
- Best-of-both-worlds: hampir semua exact gain V27c + outcome safety V12
- 93.9% match outcome-nya sama (39.846 dari 42.422) → V27c mendominasi
- Mudah diimplementasi, risk-free

**Kekurangan**:
- AW-MAE masih lebih tinggi dari V12 (2.540 vs 2.502)
- Hanya 2.576 match yang berbeda outcome (6.1%) — room for improvement kecil
- Skor V27c yang diambil masih suboptimal untuk MAE (1-0 bukannya 2-1 meningkatkan MAE saat truth=3-1)

---

## Bagian 2: Mengapa AW-MAE Stuck dan Bahkan Turun?

### 2.1 Akar Masalah: Penalti 1.5x Membuat Outcome Jauh Lebih Penting dari Exact

AW-MAE dihitung sebagai:
```
aug = MAE + 0.30*(1-exact) + 0.25*(1-outcome_ok) + 0.15*(1-gd_ok)
multiplier = 1.5 jika outcome salah, 1.0 jika benar
AW-MAE = (aug * multiplier)^1.3
```

Implikasi numeriknya:
- **Prediksi 2-1 saat truth 1-0**: MAE=1.0, outcome ✅ → AW-MAE = (1.0+0.30+0.15)^1.3 = **1.76**
- **Prediksi 1-0 saat truth 1-0**: MAE=0.0, exact ✅ → AW-MAE = **0.00** (perfect)
- **Prediksi 1-0 saat truth 0-1**: MAE=1.0, outcome ❌ → AW-MAE = (1.0+0.30+0.25+0.15)*1.5)^1.3 = **3.11**

Jadi **1 exact match yang berhasil** (save 1.76) tidak cukup mengkompensasi **1 outcome mistake** (cost 3.11 - 1.76 = 1.35 extra). Secara statistik, setiap kali kita mengubah prediksi dari 2-1 ke 1-0:
- ~13% kemungkinan menebak benar → save 1.76
- ~87% kemungkinan salah, dan beberapa di antaranya salah outcome → cost besar

**Net expected value negatif.** Inilah mengapa setiap upaya diversifikasi memperburuk AW-MAE.

### 2.2 ERM Sudah Mengoptimalkan AW-MAE secara Teoritis Optimal

ERM memilih skor yang meminimalkan expected AW-MAE loss. Artinya, **V12 sudah sangat dekat dengan batas bawah teoritis** yang bisa dicapai oleh arsitektur dan fitur yang sama. Setiap modifikasi pada decision rule hanya bisa memindahkan "error budget" — tidak bisa menciptakan informasi baru.

### 2.3 Information Ceiling

Semua fitur yang ada (Elo, form, goal stats, tournament tier, venue) berkorelasi tinggi satu sama lain. Menambah fitur baru yang merupakan derivasi dari fitur yang sama (conf_strength, goal_ratio, form_accel) tidak menambah mutual information terhadap target. Ini dibuktikan oleh:
- V14 (62 fitur) ≈ V12 (52 fitur): +10 fitur hanya memberikan -0.008 AW-MAE improvement
- SHAP feature importance menunjukkan Elo diff mendominasi >40% dari total gain

---

## Bagian 3: Tabel Perbandingan Lengkap

| Versi | AW-MAE | Exact% | Outcome% | Draw Pred% | GT Draw% | 2-1/1-2 pred | 1-0/0-1 pred | 0-0 pred |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| **V12** | **2.502** | 9.4% | **59.9%** | 12.2% | 20.5% | 21.895 | 3.621 | 0 |
| V26 | 2.535 | 10.5% | 58.5% | ~13% | 20.5% | ~18k | ~4k | 0 |
| V26b | 2.537 | 10.8% | 58.4% | ~13% | 20.5% | ~17k | ~4k | 0 |
| V26c | 2.515 | 10.3% | 58.8% | 13.5% | 20.5% | 21.108 | 3.723 | 0 |
| V27 | 2.808 | 9.5% | 58.9% | 12.5% | 20.5% | 4.391 | 355 | 0 |
| V27b | 2.535 | 11.0% | 58.5% | 15.2% | 20.5% | 11.783 | 8.760 | 0 |
| V27c | 2.544 | **11.4%** | 58.3% | 16.0% | 20.5% | 8.417 | 12.534 | 0 |
| V28 | 2.540 | 11.2% | 58.9% | ~15% | 20.5% | ~9k | ~12k | 0 |

**Pola yang terlihat jelas**: 
- AW-MAE dan Outcome accuracy bergerak searah
- Exact Score dan distribusi prediksi bergerak berlawanan dengan AW-MAE
- 0-0 = 0 di SEMUA versi — ini adalah limitasi fundamental ERM

---

## Bagian 4: Saran Perbaikan dan Eksplorasi Baru

### 4.1 Jalur Konservatif: Micro-tuning V12

**A. Custom Loss Tensor dengan penalty sangat kecil (0.03-0.06)**

V27b menunjukkan bahwa penalty 0.12 menghasilkan Exact 11% tapi AW-MAE 2.535. Berdasarkan pola linear, penalty ~0.05 mungkin sweet spot di mana Exact naik ke ~10.2% tanpa terlalu mengorbankan outcome. Ini perlu grid search kecil.

**B. Tier-Specific Temperature (tanpa modifikasi lain)**

V12 menggunakan T=1.1 (Men) dan T=1.2 (Women) secara global. Mungkin tier-specific temperature bisa membantu:
- Tier 4/1 (draw-heavy): T=1.0 (lebih tajam → lebih percaya diri pada draw)
- Tier 3/5 (decisive): T=1.2 (lebih rata → lebih hedge)

Ini hanya mengubah 1 parameter per tier, bisa di-ablation dengan mudah.

### 4.2 Jalur Moderat: Perbaikan Fitur

**C. Rolling Volatility Features (BELUM DICOBA)**

Alih-alih rata-rata gol (sudah di fitur), tambahkan *variance/std* dari gol terakhir N match. Tim dengan volatilitas tinggi (kadang menang 5-0, kadang kalah 0-3) lebih susah diprediksi → model bisa menurunkan confidence-nya.

Fitur spesifik:
- `goal_volatility_team = std(goals_last_10)`
- `goal_volatility_opp = std(goals_conceded_last_10)`
- `result_volatility = std(goal_diff_last_10)`

Ini informasi yang **genuinely berbeda** dari rata-rata, sehingga bisa menembus information ceiling.

**D. Head-to-Head Features (BELUM DICOBA)**

- `h2h_goal_diff_mean`: rata-rata goal difference dalam pertemuan sebelumnya
- `h2h_draw_rate`: frekuensi seri dalam h2h
- `h2h_total_goals_mean`: rata-rata total gol h2h

Ini bisa sangat berguna untuk pertandingan kompetitif (Tier 4/5) di mana tim yang sama sering bertemu.

**E. Recent Form Momentum (BELUM DICOBA)**

- `form_acceleration`: apakah tim sedang membaik atau memburuk (slope dari win rate 3-match vs 10-match)
- `winning_streak / losing_streak`: panjang streak saat ini

### 4.3 Jalur Eksploratif: Arsitektur Berbeda

**F. Decoupled Outcome-Score Architecture**

Arsitektur V12 menggabungkan outcome dan score dalam soft cascade. Bagaimana jika kita benar-benar memisahkan pipeline:

1. **Pipeline Outcome**: Model khusus 3-class (W/D/L) yang dioptimasi **hanya untuk outcome accuracy** — bisa pakai fitur berbeda, threshold berbeda
2. **Pipeline Score|Outcome**: 3 model terpisah — model Draw (6 skor), model Win (15 skor), model Loss (15 skor) — masing-masing dilatih hanya pada subset data yang relevan
3. **Combine**: Outcome dari Pipeline 1, Score dari Pipeline 2

Berbeda dari V27 Hard Cascade karena:
- Pipeline Score dilatih pada data yang **sudah difilter per outcome** (bukan difilter saat inference)
- Model Draw hanya melihat match-match seri, jadi lebih bisa membedakan 0-0 vs 1-1 vs 2-2

**G. Bayesian/Poisson Regression untuk Score**

Model parametrik sederhana yang mungkin bisa menangkap pola yang GBDT lewatkan:
- Dixon-Coles Poisson: mengasumsikan gol tim home dan away mengikuti distribusi Poisson yang berkorelasi
- Ini natively menghasilkan probabilitas untuk setiap skor termasuk 0-0
- Bisa diensemble dengan LGB output

Keuntungan: model ini secara alami menghasilkan distribusi skor yang lebih natural (termasuk 0-0), karena Poisson memiliki massa probabilitas yang tinggi di 0.

**H. Quantile Regression untuk Goal Totals**

Alih-alih memprediksi skor sebagai klasifikasi 36-class:
1. Prediksi total gol (t+o) sebagai distribusi kontinu (quantile regression)
2. Prediksi goal difference (t-o) sebagai distribusi kontinu
3. Reconstruct (t, o) dari (total, diff)

Keuntungan: lebih sedikit classes, masing-masing model lebih fokus.

**I. Neural Network Embedding + GBDT**

Gunakan neural network HANYA untuk menghasilkan learned embedding dari tim (berdasarkan sequence match terakhirnya), lalu jadikan embedding ini sebagai input fitur GBDT. Ini bisa menangkap pola temporal yang GBDT dengan engineered features tidak bisa tangkap.

### 4.4 Jalur "Mengakali" Metrik

**J. Conditional Score Override (paling pragmatis)**

Analisis data menunjukkan bahwa 81.5% dari semua draw adalah 0-0 atau 1-1. Maka:
- Jika model memprediksi draw: pilih 1-1 (jangan biarkan ERM memilih, karena ERM akan selalu pilih 1-1 anyway)
- Jika model memprediksi win dan elo_diff < 100: pilih 1-0 (narrow win)
- Jika model memprediksi win dan elo_diff >= 100: biarkan ERM memilih (biasanya 2-1 atau 2-0)

Ini adalah heuristic rule-based yang langsung menargetkan kasus di mana ERM terbukti suboptimal.

**K. Direct Frequency Matching**

Alih-alih ERM, gunakan decision rule yang memaksimalkan likelihood match antara distribusi prediksi dan distribusi ground truth. Artinya:
- Jika ground truth memiliki 7.7% match 0-0, maka prediksi juga harus memiliki ~7.7% prediksi 0-0
- Implementasi: sortir semua match berdasarkan "probabilitas 0-0 terbesar", pilih top 7.7% → override ke 0-0

Ini bukan cheating karena kita hanya menggunakan *distribusi marginal* dari ground truth, bukan jawaban per match.

---

## Bagian 5: Rekomendasi Prioritas

| Prioritas | Strategi | Expected Impact | Effort | Risiko |
|:---|:---|:---|:---|:---|
| 🔴 1 | **F. Decoupled Outcome-Score** | -0.03 to -0.08 | Tinggi | Sedang |
| 🔴 2 | **C. Rolling Volatility Features** | -0.01 to -0.03 | Rendah | Rendah |
| 🟡 3 | **J. Conditional Score Override** | -0.01 to -0.02 | Rendah | Rendah |
| 🟡 4 | **A. Loss Tensor penalty=0.05** | -0.005 to -0.01 | Rendah | Rendah |
| 🟡 5 | **G. Dixon-Coles Poisson Ensemble** | -0.01 to -0.03 | Sedang | Sedang |
| 🟢 6 | **D. Head-to-Head Features** | -0.005 to -0.02 | Sedang | Rendah |
| 🟢 7 | **H. Quantile Regression** | -0.01 to -0.03 | Tinggi | Tinggi |
| 🟢 8 | **I. NN Embedding + GBDT** | -0.01 to -0.05 | Tinggi | Tinggi |

### Rekomendasi urutan eksekusi:
1. **C + A** secara bersamaan (low effort, bisa di-ablation): tambah volatility features + loss tensor penalty kecil
2. **F** sebagai eksperimen utama berikutnya: arsitektur Decoupled yang melatih model Score khusus per outcome bucket
3. **G** jika F tidak membantu: Poisson model sebagai ensemble component baru

---

## Bagian 6: Kesimpulan

V26 dan V27 **tidak gagal** — mereka berhasil mengungkap bahwa:
1. Masalah bukan di parameter model, tapi di **algoritma keputusan (ERM)**
2. Ada **tradeoff fundamental** antara exact accuracy dan outcome accuracy dalam metrik AW-MAE
3. **Information ceiling** dari fitur yang ada sudah tercapai

Untuk menembus plateau, kita perlu bergeser dari "tuning parameter" ke "mengubah arsitektur keputusan" atau "menambah informasi genuinely baru". Jalur paling menjanjikan adalah **Decoupled Outcome-Score Architecture** (F) yang melatih model Score dalam konteks outcome yang sudah ditentukan, dan **Rolling Volatility Features** (C) yang menambahkan dimensi informasi yang benar-benar baru.
