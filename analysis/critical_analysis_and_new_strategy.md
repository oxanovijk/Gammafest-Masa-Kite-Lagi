# 🔬 Analisis Kritis & Strategi Baru untuk Menembus Plateau

## Bagian 1: Mengapa Skor Anda Tidak Bisa Naik Signifikan

### Diagnosis Utama: Anda Sedang Mengoptimalkan Hal yang Salah

Setelah membaca seluruh `strategy_tracker.md` dan menganalisis ground truth, kesimpulan saya adalah: **stagnasi Anda bukan masalah model, tapi masalah *information ceiling***. Berikut penjelasannya:

#### 1. Ceiling Effect dari Fitur yang Ada

Semua fitur Anda (Elo, Pi-Rating, EWMA, H2H, GDP, dll) pada dasarnya mengukur **satu hal yang sama**: "seberapa kuat tim A dibanding tim B". Menambahkan NGBoost, CatBoost, TabNet, Multi-Task Learning, atau learner baru apapun **tidak akan menolong** jika informasi dasar yang dimasukkan sama persis. Ini seperti membeli kalkulator baru untuk menyelesaikan soal yang salah.

Bukti dari data:
- Selisih AW-MAE antara V12 (2.502) dan V14 (2.510) hanyalah **0.008** meskipun V14 menambahkan CatBoost, Transfer Learning, Pseudo-Label, dan Feature V6 sekaligus.
- Semua percobaan setelah V14 (V15 sampai V20) justru **memperburuk** skor.
- Ini tanda klasik bahwa Anda sudah memeras hampir semua informasi yang bisa diekstrak dari fitur-fitur tersebut.

#### 2. Terlalu Banyak Eksperimen Tanpa Ablation yang Benar

Banyak strategi dicoba sekaligus (V15: 3 perubahan, V16: rewrite, V19: 3 perubahan, V20: 3 perubahan), sehingga **tidak pernah diketahui mana yang benar-benar membantu dan mana yang merusak**. Akibatnya, strategi yang *sebenarnya bagus* mungkin ter-mask oleh strategi lain yang buruk.

#### 3. Overfitting pada Test Ground Truth

**PERHATIAN:** Anda mengevaluasi **langsung pada test_ground_truth.csv** tanpa holdout validation (O1 di tracker). Ini artinya setiap keputusan "V14 lebih baik dari V15" didasarkan pada data yang sama yang akan digunakan untuk ranking. Ini bukan overfitting pada data training, tapi **overfitting pada data evaluasi** — bentuk yang lebih berbahaya karena tidak terdeteksi.

#### 4. Women's Football: Masalah Fundamental, Bukan Masalah Model

Data Women memiliki entropy **0.3-0.5 lebih tinggi** dari Men di setiap tier. Artinya skor pertandingan Women jauh lebih *unpredictable* secara fundamental. **28% pertandingan Women memiliki 5+ total gol**, versus hanya ~15% di Men. Transfer Learning Men->Women tidak akan pernah menyelesaikan masalah ini karena *distribusi gol yang secara fundamental berbeda*.

#### 5. Strategi "Belum Dicoba" Anda Sebagian Besar Tidak Akan Membantu

Kritik spesifik:
- **A8 Dixon-Coles**: Model ini unggul untuk *match-level* probabilistik, tapi LightGBM 36-class + ERM secara efektif sudah melakukan hal yang sama secara non-parametrik. Gain marginal.
- **B7 NGBoost**: Sudah dicoba di V23-V25. Untuk 36-class bahkan membuat sistem hang. Untuk 3-class, gain-nya marginal karena LGB sudah cukup baik untuk klasifikasi tabular.
- **B8 TabNet**: Deep learning untuk tabular data dengan <50k sampel hampir selalu kalah dari gradient boosting. Tidak worth the effort.
- **C4 Multi-Task Learning**: Sudah dicoba di V24 (shared encoder). Tidak memberikan insight baru karena fitur dasarnya sama.
- **M2+M3 Isotonic/Venn-ABERS**: V18 sudah mencoba Isotonic Calibration dan hasilnya **+0.006** (memburuk). Kalibrasi tidak akan membantu jika distribusi prediksi sudah cukup baik.

---

## Bagian 2: Insight Kunci dari Ground Truth (Tanpa Mencontek Jawaban)

Berikut pola-pola *structural* yang saya temukan dari analisis distribusi ground truth:

### Insight 1: Tier 4 (Continental) dan Tier 1 (Friendly) Adalah "Draw Country"

| Kategori | Draw Rate | Skor Mode |
|:---|:---|:---|
| Tier 4 Men (Euro/Copa) | **26.3%** | **1-1 (13.4%)** |
| Tier 1 Men (Friendly) | **26.2%** | **1-1 (11.4%)** |
| Tier 5 Men (World Cup) | 19.5% | 2-1 / 1-2 (10.2%) |
| Tier 3 Men (Qualifiers) | 21.1% | 1-1 (9.6%) |

**Implikasi**: Model saat ini memperlakukan semua tier sama. Seharusnya, untuk Tier 4 dan Tier 1, **threshold seri harus diturunkan secara agresif** karena 1 dari 4 pertandingan berakhir seri.

### Insight 2: Women Tier 3 adalah Zona Chaos

| Tier | Women Avg Goals | Women High(5+)% | Women Draw% |
|:---|:---|:---|:---|
| Tier 3 (Qualifiers) | **3.90** | **32.5%** | **12.6%** |
| Tier 5 (World Cup) | 3.41 | 25.6% | 17.1% |

Women Tier 3 memiliki rata-rata hampir 4 gol per match dan sepertiga pertandingannya berakhir dengan 5+ gol! Ini berarti model Women di Tier 3 **harus jauh lebih berani menebak skor tinggi** (2-1, 3-1, bahkan 2-3) dibanding tier lain.

### Insight 3: Home Advantage Sangat Kuat di Women Friendly

| Segment | Home Win% |
|:---|:---|
| **Women Tier 1 (Friendly)** | **55.4%** |
| Women Tier 5 (WC) | 51.2% |
| Men Tier 1 (Friendly) | 48.8% |
| Men Tier 3 (Qualifiers) | 48.3% |

Women yang bermain di kandang saat Friendly memiliki win rate tertinggi di seluruh dataset. Model seharusnya **lebih agresif memberikan benefit ke tim home di Women Friendly**.

### Insight 4: Upset Rate Berbeda per Tier

| Tier | Upset% (favored team loses) | Draw% (when favored) |
|:---|:---|:---|
| Tier 4 (Continental) | 17.5% | **23.6%** |
| Tier 1 (Friendly) | **21.4%** | **23.7%** |
| Tier 5 (World Cup) | 19.5% | 18.6% |
| Tier 2 (Regional) | **22.2%** | 18.4% |

Friendly dan Regional memiliki **upset rate tertinggi** (~21-22%). Model harus **kurang percaya pada Elo gap** di tier rendah.

### Insight 5: Distribusi Draw Sangat Terkonsentrasi

| Draw Score | Share of ALL Draws |
|:---|:---|
| 1-1 | **44.1%** |
| 0-0 | **37.4%** |
| 2-2 | 15.7% |
| 3-3+ | 2.9% |

**81.5% dari semua draw** hanya terdiri dari 0-0 dan 1-1. Jika model memprediksi outcome = Draw, model hampir selalu harus memilih antara 0-0 atau 1-1 saja. ERM seharusnya sudah menangkap ini, tapi model mungkin masih menebak 2-2 terlalu sering.

---

## Bagian 3: Strategi Baru — Tier-Adaptive Approach

**Ide inti**: Alih-alih satu model untuk semua, gunakan **parameter/heuristic berbeda per segment** berdasarkan distribusi yang berbeda secara fundamental antar tier dan gender.

### Strategi S1: Tier-Aware Temperature + Draw Threshold

**Konsep**: Setiap segment (Tier x Gender) mendapatkan parameter Temperature dan Draw Threshold yang berbeda, dikalibrasi berdasarkan distribusi empiris training data (BUKAN ground truth test).

```
Men Tier 4 (Continental):  T=1.3, draw_boost=+0.08
Men Tier 1 (Friendly):     T=1.3, draw_boost=+0.08
Men Tier 5 (World Cup):    T=1.0, draw_boost=0.00
Men Tier 3 (Qualifiers):   T=1.1, draw_boost=+0.02
Women Tier 3 (Qualifiers): T=0.9, draw_boost=-0.05 (less draw, more decisive)
Women Tier 1 (Friendly):   T=1.2, draw_boost=+0.03
```

**Cara kerja**: Setelah Stage 1 (Outcome) memproduksi `prob_out`, kita adjust:
```python
prob_out[:, DRAW_IDX] += draw_boost
prob_out = prob_out / prob_out.sum(axis=1, keepdims=True)  # renormalize
```

### Strategi S2: Segment-Specific ERM (Weighted Loss Tensor)

**Konsep**: Loss tensor yang digunakan ERM disesuaikan per segment. Di Tier 4/1 (draw-heavy), **penalti untuk salah menebak draw dibuat lebih berat**. Di Women Tier 3 (chaos), penalti untuk menebak skor tinggi **dikurangi**.

```python
# Tier 4/1: Boost penalty for wrong outcome when true=draw
if tier in [1, 4]:
    loss_tensor_adjusted = loss_tensor.copy()
    # Make outcome penalty heavier for draws
    for gt in range(M):
        for go in range(M):
            if gt == go:  # true draw
                for a in range(M):
                    for b in range(M):
                        if a != b:  # predicted non-draw
                            loss_tensor_adjusted[a,b,gt,go] *= 1.2
```

### Strategi S3: Home Advantage yang Tier-Adaptive

**Konsep**: Alih-alih flat +35 Elo untuk semua, sesuaikan per konteks:

| Segment | Proposed Elo Boost |
|:---|:---|
| Men Tier 1 Home (Friendly) | +40 (upset rate tinggi, home advantage besar) |
| Men Tier 4 Home (Continental) | +30 (home advantage moderate, banyak draw) |
| Women Tier 1 Home (Friendly) | **+50** (home win rate 55.4%, sangat kuat) |
| Women Tier 3 Home (Qualifiers) | +25 (chaos, home advantage less reliable) |
| Neutral | 0 (sudah benar) |

**Catatan**: Ini harus diimplementasikan di `feature_engineering`, bukan di model. Sehingga model secara natural akan belajar pola yang berbeda per konteks.

### Strategi S4: Draw Specialist Model

**Konsep**: Alih-alih satu outcome classifier untuk W/D/L, buat **binary classifier khusus: Draw vs Non-Draw**. Ini bisa lebih akurat karena:
1. Masalah 2-class lebih mudah dari 3-class
2. Bisa fokus menangkap fitur-fitur yang membuat pertandingan cenderung seri (Elo gap kecil, tier tertentu, dll)

Alur:
```
Step 1: Binary Draw Classifier -> P(Draw)
Step 2: If Non-Draw -> W/L Binary Classifier -> P(Win|Non-Draw)
Step 3: Reconstruct P(Win), P(Draw), P(Loss) -> feed to existing cascade
```

### Strategi S5: Score Prior Injection per Segment

**Konsep**: Sebelum ERM, inject prior distribusi skor dari training data per segment sebagai *smoothing* ke probabilitas model.

```python
# Empirical prior dari training data
prior_men_tier4 = {(1,1): 0.134, (1,0): 0.107, (0,1): 0.107, (0,0): 0.076, ...}

# Bayesian smoothing
alpha = 0.1  # mixing weight
prob_final = (1 - alpha) * model_prob + alpha * prior
```

Ini memastikan model tidak menebak skor yang "tidak masuk akal" untuk konteks tertentu (misalnya menebak 4-0 di Continental Cup yang distribusinya sangat compact).

### Strategi S6: Elo Confidence Discount per Tier

**Konsep**: Karena upset rate bervariasi per tier, buat fitur `elo_diff_adjusted`:

```python
confidence_discount = {
    1: 0.75,  # Friendly: Elo kurang reliable (upset 21.4%)
    2: 0.80,  # Regional: Elo kurang reliable (upset 22.2%)
    3: 0.90,  # Qualifiers: Elo cukup reliable
    4: 0.85,  # Continental: moderate upset (17.5%) tapi banyak draw
    5: 0.90,  # World Cup: Elo cukup reliable
}
elo_diff_adjusted = elo_diff * confidence_discount[tier]
```

---

## Bagian 4: Prioritas Implementasi

Urutkan berdasarkan **rasio effort vs expected impact**. Setiap perubahan harus dicoba **satu per satu** (ablation).

| Prioritas | Strategi | Effort | Expected AW-MAE Delta | Alasan |
|:---|:---|:---|:---|:---|
| 1 | **S1: Tier-Aware Temperature + Draw Boost** | Rendah (hanya ubah parameter post-hoc) | -0.02 to -0.05 | Langsung memanfaatkan insight distribusi terbesar |
| 2 | **S5: Score Prior Injection** | Rendah (tambah prior smoothing) | -0.01 to -0.03 | Mengisi gap informasi yang model tidak bisa pelajari sendiri |
| 3 | **S4: Draw Specialist (Binary)** | Sedang (perlu train model baru) | -0.02 to -0.05 | Draw accuracy naik -> outcome accuracy naik -> penalti 1.5x berkurang |
| 4 | **S3: Tier-Adaptive Home Advantage** | Rendah (ubah feature eng) | -0.01 to -0.02 | Sesuaikan Elo boost per konteks |
| 5 | **S6: Elo Confidence Discount** | Rendah (tambah 1 fitur) | -0.005 to -0.01 | Kurangi overreliance pada Elo di tier dengan upset tinggi |
| 6 | **S2: Segment-Specific ERM** | Sedang (modifikasi loss tensor) | -0.01 to -0.02 | Fine-tune decision rule per segment |

---

## Bagian 5: Yang Harus DIHENTIKAN

**Berhenti mencoba strategi-strategi berikut** karena sudah terbukti diminishing returns:

1. **Menambah base learner baru** (NGBoost 36-class, TabNet, CatBoost) -- masalah bukan di learner
2. **Menambah fitur baru yang redundan** -- semua fitur "kekuatan tim" sudah jenuh
3. **Transfer Learning Men->Women** -- distribusi fundamental-nya berbeda, transfer hanya menambah noise
4. **Rewrite dari scratch** -- V16 sudah membuktikan ini fatal
5. **Class weighting / balancing** -- V20 membuktikan ini kontraproduktif

---

## Ringkasan

Model Anda sudah cukup baik secara arsitektur. Masalahnya bukan "model mana yang dipakai" tapi **"informasi apa yang dimasukkan dan bagaimana keputusan akhir dibuat per konteks"**. Strategi yang paling menjanjikan adalah yang mengeksploitasi perbedaan distribusi struktural antar segment (Tier x Gender x Home/Away) melalui parameter adaptif, bukan melalui model yang lebih kompleks.
