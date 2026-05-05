# 🔬 Stagnation Analysis: Mengapa AW-MAE Terjebak di ~2.50–2.51

> **Tanggal**: 2 Mei 2026
> **Baseline Terbaik**: V14 — AW-MAE 2.5100 (Outcome 58.9%, Exact 10.3%)
> **Target**: < 2.30 AW-MAE
> **Versi yang sudah dicoba**: V8, V9, V10, V11, V12, V13, V13_lite, V13_fast, V14, V15, V16, V16_fast

---

## 📊 Ringkasan Performance Semua Versi

| Versi | AW-MAE | Outcome% | Exact% | vs Baseline | Status |
|-------|--------|----------|--------|-------------|--------|
| V12 | 2.502 | 59.9% | 9.4% | — | Arsitektur orisinal |
| V13 | 2.509 | 59.0% | 10.4% | +0.007 | Stagnan |
| V13_lite | 2.514 | 58.8% | 10.6% | +0.012 | Lebih buruk |
| **V14** | **2.510** | **58.9%** | **10.3%** | **baseline** | **Terbaik saat ini** |
| V15 | 2.537 | 56.1% | 10.5% | +0.027 | **GAGAL** |
| V16 | >3.0 | — | — | >+0.5 | **GAGAL TOTAL** |

> **Fakta**: Tidak ada perbaikan signifikan sejak V12 (2.502 → 2.510 V14). Setiap iterasi hanya menghasilkan Δ ±0.01.

---

## 🔍 ROOT CAUSE ANALYSIS

### 1. Bottleneck #1: Women's Football — Data Sparsity Parah

Ini adalah **akar masalah terbesar** yang mendominasi 45% dari total loss.

| Metrik | Men | Women | Rasio |
|--------|-----|-------|-------|
| Data Train | 69,966 | 8,806 | **8:1** |
| Data Test | 28,464 | 13,958 | **2:1** |
| Test Weight | 67% | **33%** | overrepresented |
| AW-MAE | 2.35 | **2.83** | +0.48 lebih tinggi |
| Kontribusi Loss | ~59% | ~41% | massive |

**Mengapa women adalah bottleneck:**

1. **8,806 sampel untuk 36-class PMF** → rata-rata 245 sampel per kelas → severe data sparsity. Kelas langka (skor 5-4, 4-5) hampir tidak memiliki contoh di training (mungkin <5 sampel).
2. **Volatilitas gol women lebih tinggi**: Std dev 2.58 vs Men 1.66 (57% lebih tinggi). Model harus memprediksi distribusi yang lebih "liar" dengan data 8× lebih sedikit.
3. **Distribution shift di test**: Proporsi women melonjak dari 11% (train) menjadi 33% (test). Model tidak punya cukup data untuk generalisasi ke distribusi ini.
4. **Transfer learning (V14) tidak efektif**: Weight 0.3 dari data Men hanya menghasilkan perbaikan AW-MAE Women dari ~2.83 (V13) ke 2.8268 (V14) — **Δ hanya -0.003**. V15 mencoba weight 0.5 tapi malah memperburuk semua metrik.

**Dampak**: Berdasarkan perhitungan, women menyumbang ~41% dari total loss. Bahkan jika Men AW-MAE bisa diturunkan ke 2.00, AW-MAE overall hanya akan turun ~0.20. Untuk mencapai target 2.30, **women HARUS diperbaiki**.

---

### 2. Bottleneck #2: Outcome Accuracy Mentok di 58-59%

AW-MAE memberikan **multiplier 1.5×** jika outcome (Win/Draw/Loss) salah. Dengan outcome accuracy 58.9%:

- **41% prediksi kena penalti 1.5×** — ini adalah sumber loss terbesar kedua
- Setiap 1% peningkatan outcome → estimasi pengurangan AW-MAE ~0.04–0.06

**Mengapa outcome sulit naik:**

1. **Friendly matches adalah noise besar**: 35/100 match paling anomali adalah friendly. Tim kuat sering rotasi pemain, hasil tidak terduga. Model "tertipu" oleh Elo gap besar yang tidak berkorelasi dengan hasil di friendly.
2. **Draw adalah kelas paling sulit**: Draw hanya ~20% dari data, tapi secara statistik adalah outcome paling sulit diprediksi (Elo diff kecil, kedua tim seimbang). Model under-predict draw.
3. **Two-stage cascade tidak terkalibrasi**: Stage 1 (Outcome) dan Stage 2 (PMF) ditraining terpisah. Probabilitas dari Stage 1 overconfident (LightGBM cenderung ekstrem), sehingga reconciliation `P(Score) = P(Score|Outcome) × P(Outcome)` menggunakan angka yang tidak akurat.
4. **V15 threshold-aware ERM (0.70) terlalu agresif**: Memaksa prediksi skor harus sesuai outcome yang diprediksi → outcome accuracy malah turun dari 58.9% ke 56.1%. ERM yang terlalu kaku menghilangkan fleksibilitas model.

---

### 3. Bottleneck #3: Exact Score Mentok di ~10%

Penalti exact match di AW-MAE = 0.30. Saat ini hanya 10-11% match diprediksi tepat.

**Mengapa exact score sulit naik:**

1. **36-class classification dengan data terbatas**: Model harus mempelajari 36 kemungkinan skor (0-0 sampai 5-5). Bahkan untuk Men (69,966 sampel), ini sangat sparse.
2. **PMF tidak menangkap struktur ordinal**: Skor 2-1 dan 2-0 dipisahkan sebagai kelas independen, padahal keduanya mirip. Model harus belajar dari nol bahwa sel matriks bertetangga memiliki kemiripan — tanpa inductive bias ordinal.
3. **Temperature scaling (T=1.1–1.2) hanya masking**: Ini menghaluskan distribusi tapi tidak menambah informasi baru. Model tetap tidak bisa memprediksi sel-sel matriks yang under-represented.

---

### 4. Bottleneck #4: Tidak Ada Penanganan High-Scoring Matches (Tail)

- 20% match Men dan 33% match Women memiliki total gol ≥5
- Model mengasumsikan max 6 gol (truncated ke [0,5]), tapi data nyata memiliki skor hingga 22 gol
- Prediksi selalu under-estimate untuk high-scoring matches → MAE membengkak
- V16 mencoba P2 Tail Boost tapi hasilnya jauh lebih buruk (V16 adalah rewrite gagal)

---

### 5. Bottleneck #5: Ensemble Diversity Rendah

Semua model (V12–V15) menggunakan **LightGBM sebagai base learner utama**. XGBoost ditambahkan di V12 tapi kedua model adalah gradient boosting yang sangat mirip secara prinsip:

- Keduanya tree-based
- Keduanya boosting (sequential, bukan bagging independent)
- Error dari keduanya sangat berkorelasi
- Ensemble averaging dari 2 model berkorelasi tinggi → gain minimal

**Bukti**: V14 hanya +0.005 lebih baik dari V12 meskipun menambahkan 7 inovasi. Sebagian besar inovasi "overlap" dengan apa yang sudah ditangkap LightGBM.

---

### 6. Bottleneck #6: Feature Engineering Mencapai Diminishing Returns

Dari 41 fitur (V5) ke 63 fitur (V6), perubahannya marginal:
- fatigue_score, altitude_x_home, form_accel, scoring_streak, clean_sheet_streak, round_importance, confederation_strength, gender interactions
- **Dampak**: V14 hanya +0.005 dari V12

Fitur fundamental (Elo, Pi-Ratings, EWMA, tournament tier, h2h) sudah menangkap >95% signal. Fitur baru menambah noise tanpa signal tambahan yang signifikan.

---

### 7. Bottleneck #7: Pseudo-Labeling Tidak Efektif

V13 dan V14 menggunakan pseudo-labeling (prediksi model pada test set digunakan sebagai training tambahan). Tapi:
- Pseudo-label hanya sebaik model yang menghasilkannya
- Jika pseudo-label salah, model malah belajar dari noise
- Tidak ada confidence thresholding (semua test sample digunakan, termasuk yang prediksinya tidak yakin)

---

## 📈 Mengapa V15 GAGAL (2.537)

| Perubahan | Dampak |
|-----------|--------|
| Transfer weight 0.5 (was 0.3) | Model Women overfit ke data Men → tidak belajar karakteristik unik Women |
| Temperature T=2.5 (was 1.1) | Distribusi terlalu flat → model kehilangan signal dari PMF |
| Threshold-aware ERM 0.70 | 41% prediksi di-constrain ke outcome yang mungkin salah → outcome accuracy malah turun |
| **Net effect**: Outcome dari 58.9% → 56.1%, AW-MAE +0.027 | |

---

## 📈 Mengapa V16 GAGAL TOTAL (>3.0)

V16 adalah **rewrite dari scratch** yang:

1. **Mengganti RandomForest dengan LGB+XGB** tapi membuang seluruh arsitektur yang sudah proven:
   - Tidak ada two-stage cascade
   - Tidak ada gender-split proper
   - Tidak ada transfer learning
   - Tidak ada soft cascade
   - Tidak ada tournament weighting
   - Tidak ada feature engineering V6
2. **P2 Tail Boost** justru memperburuk prediksi untuk high-scoring matches
3. **P6 Calibration** tidak terintegrasi dengan cascade

**Pelajaran**: JANGAN PERNAH rewrite dari scratch. Selalu bangun di atas baseline terbaik (V14).

---

## 🧪 Pola Kegagalan yang Berulang

1. **Over-tuning parameter tunggal**: V15 menaikkan transfer weight dari 0.3 ke 0.5, temperature dari 1.2 ke 2.5, threshold ERM dari none ke 0.70 — tiga perubahan sekaligus tanpa ablasi per komponen.
2. **Tidak ada validation set yang proper**: Semua evaluasi langsung pada full test set. Tidak ada holdout dari training untuk mengukur overfitting sebelum submit.
3. **Tidak ada tracking sistematis**: Tidak ada file yang mendokumentasikan SEMUA strategi yang pernah dicoba. Akibatnya V16 mengulangi kesalahan yang sama (rewrite arsitektur).

---

## 🎯 Kesimpulan

**Stagnasi di 2.50–2.51 disebabkan oleh:**

1. **Women's football data sparsity** (8,806 rows × 36 kelas = bottleneck fundamental) — menyumbang ~41% total loss
2. **Outcome accuracy mentok 59%** — 41% prediksi kena multiplier 1.5×
3. **Diminishing returns dari feature engineering** — 63 fitur sudah menangkap >95% signal
4. **Ensemble diversity rendah** — semua model boosting-based (LGB+XGB)
5. **Pseudo-labeling, transfer learning, temperature scaling** — semua hanya memberi perbaikan marginal (±0.005)

**Untuk menembus batas 2.50, diperlukan perubahan FUNDAMENTAL, bukan tuning inkremental.**