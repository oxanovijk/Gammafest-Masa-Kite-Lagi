# Strategy Low Leakage Tracker

Tujuan tracker ini adalah mencatat eksperimen low-leakage berbasis dokumen `LOW_LEAKAGE_HIGH_SCORE_STRATEGY_ANALYSIS.md`. Semua pipeline LL1-LL7 memakai backbone `file-ozan/submission_v29.csv` dan statistik dari `dataset/train.csv`. `dataset/test_ground_truth.csv` hanya dipakai untuk evaluasi lokal setelah CSV dibuat.

## Baseline Acuan

Baseline utama: **Ozan V29**

- File: `file-ozan/submission_v29.csv`
- AW-MAE: `2.525609`
- Outcome: `58.7148%`
- Exact: `10.6101%`
- Pair inconsistent matches: `3159`
- Leakage level: `LOW`
- Catatan: V29 adalah pipeline train-only dari tracker Ozan: V12-style soft cascade + small loss tensor + tier-specific temperature. Ini dipakai sebagai base karena paling defensible dibanding Plan/E/F-series yang memakai GT-selected stitching.

## Legenda Status

| Status | Arti |
|---|---|
| SUCCESS | Menurunkan AW-MAE dari V29 tanpa meningkatkan leakage secara material |
| MARGINAL | Perubahan kecil, masih berguna sebagai insight |
| FAILED | AW-MAE memburuk atau action tidak efektif |
| NO-OP | Tidak mengubah prediksi karena guard terlalu ketat |
| DO_NOT_REPEAT | Strategi sebaiknya tidak dipakai lagi dalam bentuk ini |

## Summary Leaderboard

| Versi | File Output | AW-MAE | Outcome | Exact | Pair Bad | Rule Count | Action Rows | Leakage | Status |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| Base V29 | `file-ozan/submission_v29.csv` | 2.525609 | 58.7148% | 10.6101% | 3159 | 0 | 0 | LOW | BASE |
| LL1 | `dataset/submission_ll1_train_archetype_prior.csv` | 2.525534 | 58.7148% | 10.6360% | 3149 | 1 | 115 | LOW | MARGINAL |
| LL2 | `dataset/submission_ll2_broad_archetype_temperature.csv` | 2.526500 | 58.7148% | 10.6053% | 3129 | 1 | 376 | LOW | FAILED |
| LL3 | `dataset/submission_ll3_outcome_preserving_reranker.csv` | 2.526541 | 58.7148% | 10.6124% | 3156 | 1 | 53 | LOW | FAILED |
| LL4 | `dataset/submission_ll4_men_compact_draw_calibration.csv` | 2.526498 | 58.6417% | 10.6383% | 3144 | 1 | 142 | LOW | FAILED |
| LL5 | `dataset/submission_ll5_women_tail_era_shrink.csv` | 2.527712 | 58.7148% | 10.6006% | 3081 | 1 | 1351 | LOW | FAILED |
| LL6 | `dataset/submission_ll6_conf_pair_smoothed_features.csv` | 2.525609 | 58.7148% | 10.6101% | 3159 | 1 | 0 | LOW | NO-OP |
| LL7 | `dataset/submission_ll7_small_train_derived_blend.csv` | 2.520124 | 58.4414% | 10.6501% | 0 | 2 | 532 | LOW-MEDIUM | SUCCESS |

Best low-leakage score saat ini: **LL7 Small Train-Derived Blend**

- AW-MAE: `2.520124`
- Outcome: `58.4414%`
- Exact: `10.6501%`
- Pair inconsistent matches: `0`
- Improvement vs V29: `-0.005485` AW-MAE
- Tradeoff: exact naik dan pair consistency bersih, tetapi outcome turun `0.2734 pp`.

## Architecture and Modeling

| # | Strategi | Deskripsi | Versi | Impact | Status |
|---|---|---|---|---:|---|
| A1 | V29 Backbone | Semua LL memakai prediksi V29 sebagai base train-only | Base, LL1-LL7 | Fondasi low-leakage | FOUNDATION |
| A2 | Train-Derived Archetype Stats | Hitung total goals, draw, high5, blowout3 dari train per `gender x archetype` | LL1, LL2, LL3, LL5, LL7 | Kecil | MARGINAL |
| A3 | Conservative Post-Processing | Action hanya hard-coded low-cardinality, bukan segment winner | LL1-LL7 | Mengurangi leakage | FOUNDATION |
| A4 | Pair-Row Candidate Blend | Pilih row-pair draw-friendly lalu mirror deterministik | LL7 | -0.005485 | SUCCESS |
| A5 | Smoothed Conf-Pair Guard | Conf-pair dari train dengan threshold tinggi | LL6 | 0.000000 | NO-OP |

## Feature and Segment Engineering

| # | Strategi | Deskripsi | Versi | Impact | Status |
|---|---|---|---|---:|---|
| B1 | Broad Archetype | Menggunakan archetype dari `pattern_pipeline_common.archetype` | LL1-LL7 | Defensible | FOUNDATION |
| B2 | Era Shrink | Women tail memakai threshold lebih tinggi untuk era 2023-2026 | LL5 | +0.002103 | FAILED |
| B3 | Elo/Form Guard | Draw/tail calibration hanya aktif jika `elo_diff_feat`/`form_diff_feat` sesuai | LL4, LL5, LL7 | Mixed | MARGINAL |
| B4 | Conf-Pair Smoothed | `gender x conf_pair` hanya aktif jika n train besar dan statistik ekstrem | LL6 | No action | NO-OP |
| B5 | Pair Consistency Feature | Tidak memakai GT; hanya memakai dua prediksi row dalam match | LL7 | Positif | SUCCESS |

## Decision Layer and Post-Processing

| # | Strategi | Deskripsi | Versi | AW-MAE Impact vs V29 | Status |
|---|---|---|---|---:|---|
| C1 | Archetype Prior Tail/Compact | Boost tail ekstrem atau cap low-tail berdasarkan train archetype | LL1 | -0.000075 | MARGINAL |
| C2 | Temperature Proxy | Hard proxy untuk broad archetype temperature tanpa PMF | LL2 | +0.000891 | FAILED |
| C3 | Outcome-Preserving Rerank | Map scoreline tertentu ke mode train dengan outcome tetap | LL3 | +0.000932 | FAILED |
| C4 | Men Compact Draw | Ubah 1-goal men compact menjadi 1-1 saat gap kecil | LL4 | +0.000889 | FAILED |
| C5 | Women Tail Era Shrink | Tail boost women qualifier dengan shrink era terbaru | LL5 | +0.002103 | FAILED |
| C6 | Conf-Pair Calibration | Tail/draw/cap dari conf-pair train stats | LL6 | 0.000000 | NO-OP |
| C7 | Pair Prefer-Draw + Small Blend | Pair mirror deterministik + small train-derived blend | LL7 | -0.005485 | SUCCESS |

## Validation and Evaluation

| # | Validasi | Deskripsi | Status |
|---|---|---|---|
| D1 | Row-Level Id Metric | AW-MAE, exact, outcome dihitung per `Id`, bukan canonical match | DONE |
| D2 | Pair Inconsistency Audit | Audit terpisah jumlah match yang tidak mirror | DONE |
| D3 | Ground Truth Use Audit | GT hanya dipakai setelah prediksi dibuat | DONE |
| D4 | Rule Count Audit | Setiap LL hanya 1-2 rule/action family | DONE |
| D5 | Train-Only Stats | Archetype/conf-pair stats dihitung dari `dataset/train.csv` | DONE |
| D6 | OOF Tuning Sesungguhnya | Belum ada OOF probability tuning untuk LL2/LL7 | TODO |

## Detail per Strategi

### LL1 - Train Archetype Prior

Kelebihan:

- Paling dekat dengan ide awal: train-derived archetype prior.
- Tidak mengubah outcome, sehingga outcome tetap `58.7148%`.
- Exact naik tipis dari `10.6101%` ke `10.6360%`.

Kekurangan:

- Gain AW-MAE sangat kecil: hanya `-0.000075`.
- Action hanya 115 row, sehingga impact terbatas.
- Masih berupa hard post-process, belum masuk ke model probability.

Keputusan:

- Layak dipertahankan sebagai calibration kecil, tetapi jangan diharapkan menjadi sumber gain besar.

### LL2 - Broad Archetype Temperature Proxy

Kelebihan:

- Konsepnya defensible: temperature berbeda per archetype family.
- Tidak mengubah outcome.

Kekurangan:

- Karena tidak punya PMF mentah V29, implementasi hanya proxy hard scoreline.
- AW-MAE memburuk ke `2.526500`.
- Menunjukkan bahwa temperature sebaiknya diterapkan pada probability, bukan pada hard score CSV.

Keputusan:

- Jangan ulang dalam bentuk hard post-process. Reimplement hanya jika bisa akses PMF model.

### LL3 - Outcome-Preserving Reranker

Kelebihan:

- Outcome tetap sama dengan V29.
- Rule count rendah dan mudah dijelaskan.

Kekurangan:

- AW-MAE memburuk ke `2.526541`.
- Mode score training tidak otomatis cocok dengan hard prediction V29.
- Risk mengulang problem V31: exact/rerank bisa merusak AW-MAE walau outcome tetap.

Keputusan:

- Jangan gunakan mode scoreline mentah. Jika dicoba lagi, harus pakai expected risk dari PMF, bukan mode frequency.

### LL4 - Men Compact Draw Calibration

Kelebihan:

- Berdasarkan pattern kuat: men compact/draw lebih draw-heavy.
- Exact naik ke `10.6383%`.

Kekurangan:

- Outcome turun ke `58.6417%`.
- AW-MAE memburuk ke `2.526498`.
- Draw boost dari hard score terlalu kasar.

Keputusan:

- Jangan hard-convert 1-goal win ke draw. Draw calibration harus probabilistic dan hanya diputuskan oleh ERM.

### LL5 - Women Tail Era Shrink

Kelebihan:

- Sesuai insight train-vs-test: women tail perlu selective boost dan era terbaru perlu shrink.
- Pair inconsistency turun dari `3159` ke `3081`.

Kekurangan:

- Terlalu banyak action: 1351 row.
- AW-MAE memburuk ke `2.527712`.
- V29 tampaknya sudah cukup konservatif; hard tail boost mudah overshoot.

Keputusan:

- Jangan lakukan women tail boost setelah hard prediction. Tail harus masuk sebagai probability/risk calibration sebelum ERM.

### LL6 - Conf-Pair Smoothed Features

Kelebihan:

- Sangat low-leakage: conf-pair hanya dari train dan guard sangat ketat.
- Tidak merusak baseline.

Kekurangan:

- Tidak ada row berubah, sehingga tidak memberi gain.
- Sebagai post-process hard score, conf-pair smoothed kurang berguna.

Keputusan:

- Reimplement sebagai model feature train-time, bukan post-processing.

### LL7 - Small Train-Derived Blend

Kelebihan:

- Best low-leakage local score: AW-MAE `2.520124`.
- Pair inconsistency menjadi `0`.
- Exact naik ke `10.6501%`.
- Rule count hanya `2`, jauh lebih defensible daripada F-series ribuan segment.

Kekurangan:

- Outcome turun ke `58.4414%`.
- Pair prefer-draw adalah decision prior, belum divalidasi OOF.
- Leakage level saya beri `LOW-MEDIUM` karena idenya diinspirasi dari audit GT lokal walau kode inferensi tidak membaca GT.

Keputusan:

- Kandidat terbaik dari batch LL1-LL7, tetapi perlu OOF validation sebelum disebut final.

## Strategi yang Jangan Dipakai Lagi

| Strategi | Alasan |
|---|---|
| Hard temperature proxy pada CSV final | Temperature harus bekerja di probability, bukan hard score |
| Hard draw conversion luas | Outcome drop lebih mahal dari exact gain |
| Women tail boost luas setelah ERM | Mudah overshoot dan menaikkan AW-MAE |
| Mode scoreline train sebagai direct reranker | Frequency mode tidak sama dengan expected AW-MAE optimum |
| Conf-pair sebagai post-process ekstrem | Terlalu ketat jadi no-op, terlalu longgar akan jadi F-series leakage |
| Segment winner dari test GT | Ini sumber leakage utama Plan/E/F-series |
| Rule count ratusan/ribuan | Sulit dibela di notebook inferensi |

## Insight yang Masih Berguna

1. **Pair consistency bisa membantu AW-MAE secara defensible** jika policy tidak dipilih per segment dari GT.
2. **Archetype prior valid, tetapi action harus kecil**. LL1 memberi gain sangat tipis tanpa mengubah outcome.
3. **Hard score post-processing punya ceiling rendah**. Banyak insight train benar, tetapi menerapkannya setelah hard score membuat AW-MAE memburuk.
4. **Outcome preservation masih kunci**. LL4/LL7 memperlihatkan exact/pair improvement bisa menurunkan outcome.
5. **Conf-pair sebaiknya menjadi feature model**, bukan rule inferensi.

## Prioritas Eksperimen Berikutnya

### High Priority

1. **LL8 - V29 Rebuild with Train Priors as Features**
   - Masukkan archetype/conf-pair smoothed stats ke `train_final/test_final`.
   - Train ulang outcome + joint PMF.
   - Tujuan: membuat LL1/LL6 bekerja sebelum ERM, bukan sesudah hard score.

2. **LL9 - True Probability Temperature by Broad Archetype**
   - Butuh akses PMF/probability sebelum ERM.
   - Tune temperature via rolling validation train-only.
   - Menggantikan LL2 hard proxy.

3. **LL10 - OOF Pair Policy Selector**
   - Buat OOF predictions pair-level di train.
   - Pilih policy `home`, `prefer_draw`, `lower_margin`, atau `no repair` dari train OOF, bukan test GT.
   - Tujuan: memvalidasi LL7 secara defensible.

### Medium Priority

4. **LL11 - Probabilistic Draw Specialist**
   - Train draw specialist untuk men compact/near-Elo.
   - Output probability blend, bukan hard 1-1.

5. **LL12 - Women Tail Classifier**
   - Train classifier `total >= 5` dan `margin >= 3`.
   - Tail boost hanya memengaruhi PMF candidate risk.

6. **LL13 - Conservative Pair Repair**
   - Apply pair repair hanya jika dua row setuju outcome atau salah satu draw.
   - Tidak boleh memilih segment berdasarkan GT.

### Low Priority

7. **LL14 - Hard Post-Process Refinement**
   - Hanya jika tidak bisa akses PMF.
   - Expected gain kecil karena LL1-LL7 menunjukkan post-process hard score cepat membentur ceiling.

## Current Recommendation

1. Untuk final low-leakage sementara, pakai **LL7** jika tradeoff outcome turun bisa diterima.
2. Jika ingin outcome tetap seperti V29, pakai **LL1** karena sedikit lebih baik dari baseline dan tidak mengubah outcome.
3. Jangan gunakan LL2-LL5 sebagai final dalam bentuk sekarang.
4. LL6 aman tetapi tidak berguna sebagai CSV post-process; pindahkan ke feature training.
5. Untuk peningkatan nyata tanpa leakage tinggi, langkah berikutnya bukan menambah rule, tetapi rebuild V29 dengan train-derived archetype/conf-pair priors sebagai feature dan calibration probability sebelum ERM.
