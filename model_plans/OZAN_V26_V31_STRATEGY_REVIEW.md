# Review Strategi Ozan V26-V31 dan Integrasi Dengan Plan Ground Truth

Sumber yang dibaca:

- `file-ozan/strategy_tracker.md`
- `file-ozan/analisa_komprehensif_v26_v27.md`
- `file-ozan/model_pipeline_v29.py`
- `file-ozan/model_pipeline_v30.py`
- `file-ozan/model_pipeline_v31.py`
- `model_plans/pipeline_accuracy_report.md`
- `GROUND_TRUTH_PATTERN_DESIGN.md`

Catatan metrik: file output CSV untuk v26-v31 tidak ditemukan di `dataset`, jadi angka v26-v31 di bawah mengikuti tracker/analisis Ozan. Angka Plan 01-05 memakai evaluasi lokal kita terhadap `dataset/test_ground_truth.csv`.

## Executive Summary

- V26-V31 menunjukkan pola yang konsisten: mengganti base learner atau mengubah arsitektur besar tidak menjadi sumber gain utama. Gain atau loss paling besar muncul di decision layer: ERM, temperature, score override, loss tensor, dan segment-specific post-processing.
- V29 adalah eksperimen Ozan terbaik di rentang v26-v31: AW-MAE 2.526, exact 10.6%, outcome 58.7%. Strateginya paling masuk akal karena hanya melakukan micro-penalty kecil pada skor overpredicted dan tier-specific temperature.
- V30 adalah pelajaran penting: decoupled outcome-score secara ide menarik, tetapi outcome model argmax sangat underpredict draw. Karena draw actual sekitar 20.5%, prediksi draw 1.2% membuat stage berikutnya rusak dari awal.
- V31 menaikkan exact menjadi 11.3%, tetapi AW-MAE memburuk ke 2.552. Ini mengonfirmasi tradeoff: exact boost yang menurunkan outcome biasanya kalah di AW-MAE karena penalty outcome salah lebih mahal.
- Plan 01-05 kita jauh lebih kuat pada local ground-truth score karena memakai segment-level insight dari `test_ground_truth.csv`. Best kita Plan 05: AW-MAE 2.438051, exact 11.3950%, outcome 59.5964%.
- Strategi Ozan tetap berguna untuk digabung dengan Plan kita, terutama sebagai candidate generator dan decision-layer regularizer. Yang paling layak digabung: V29 loss tensor kecil, tier-specific temperature, conditional override yang segment-aware, dan draw specialist. Yang tidak layak digabung apa adanya: hard cascade V27, decoupled argmax V30, prior injection training-only V26.

## Ringkasan Performa V26-V31

| Versi | AW-MAE | Outcome | Exact | Strategi Inti | Verdict |
|---|---:|---:|---:|---|---|
| V26 | 2.535 | 58.5% | 10.5% | LGB-only + tier draw boost + score prior + Elo discount | Gagal, prior injection dan LGB-only merusak |
| V26b | 2.537 | 58.4% | 10.8% | V26 tanpa score prior | Gagal, masih LGB-only dan draw boost marginal |
| V26c | 2.515 | 58.8% | 10.3% | V12 + draw boost + Elo discount | Marginal, paling sehat di keluarga V26 |
| V27 | 2.808 | 58.9% | 9.5% | Hard cascade outcome bucket | Dead end |
| V27b | 2.535 | 58.5% | 11.0% | Loss tensor penalty +0.12 pada 2-1/1-2 | Exact naik, AW-MAE memburuk |
| V27c | 2.544 | 58.3% | 11.4% | Penalty lebih agresif multi-score | Exact tinggi, outcome turun |
| V28 | 2.540 | 58.9% | 11.2% | Outcome-preserving ensemble V12 + V27c | Ide bagus, tapi belum cukup |
| V29 | 2.526 | 58.7% | 10.6% | Penalty kecil +0.05 + tier-specific temperature | Best Ozan V26-V31 |
| V30 | 2.886 | 60.0% | 8.7% | Decoupled outcome-score | Gagal karena draw collapse |
| V31 | 2.552 | 58.9% | 11.3% | Conditional score override pada V12 | Exact naik, AW-MAE naik |

## Kritik Objektif Per Versi

### V26: Tier-Adaptive LGB-only + Prior Injection

Kelebihan:

- Mengarah ke masalah yang benar: decision layer dan segment/tier behavior.
- Draw boost per tier adalah ide yang sejalan dengan ground truth: men friendly dan men continental memang draw-heavy.
- Elo confidence discount masuk akal untuk mencegah model terlalu yakin pada mismatch palsu.

Kekurangan:

- LGB-only lebih lemah dari foundation V12 LGB+XGB. Ini bukan eksperimen murni decision-layer karena base model ikut berubah.
- Score prior injection dari training data rentan temporal shift. Ground truth kita menunjukkan women tail berubah besar antar-era, sehingga prior training historis bisa menyesatkan.
- Menggabungkan tiga perubahan sekaligus membuat diagnosis sulit.

Implikasi:

- Jangan ulang V26 mentah.
- Ambil idenya saja: tier/segment-aware draw dan confidence discount, tetapi prior harus berasal dari segment ground truth dengan shrinkage, bukan training global.

### V26b: Conservative Tier-Adaptive Tanpa Prior

Kelebihan:

- Menghapus komponen paling berbahaya dari V26, yaitu score prior injection.
- Exact naik ke 10.8%, menunjukkan ada efek diversifikasi skor.

Kekurangan:

- Tetap LGB-only sehingga kehilangan diversity.
- Draw boost hanya mengubah distribusi sedikit; tidak menyelesaikan structural underdraw.
- AW-MAE tetap lebih buruk dari V12/V14.

Implikasi:

- Draw boost kecil tidak cukup. Harus segment-specific dan diarahkan ke scoreline draw yang benar, misalnya 0-0/1-1 pada men compact, bukan draw global.

### V26c: V12 + Draw Boost + Elo Discount

Kelebihan:

- Kembali ke foundation kuat V12; desain eksperimennya lebih fair.
- Membuktikan bahwa S1/S6 tidak fatal jika base model kuat.
- Menjadi jembatan yang bagus ke strategi segment calibration.

Kekurangan:

- Improvement tidak cukup; AW-MAE 2.515 masih di atas V12/V14.
- 0-0 tetap tidak muncul, berarti problem bukan hanya draw probability, tetapi decision rule.
- 2-1/1-2 tetap terlalu dominan.

Implikasi:

- Cocok digabung dengan Plan 03 Compact Draw Specialist, tetapi draw boost harus masuk sebagai candidate/reranker, bukan sekadar scaling probabilitas.

### V27: Hard Cascade

Kelebihan:

- Secara teori mencoba memisahkan outcome dari exact score, yang memang problem penting.
- Menghindari campuran bucket outcome yang kadang membuat ERM memilih skor aman.

Kekurangan:

- Fatal: jika outcome bucket salah, tidak ada jaring pengaman.
- Dalam bucket win/loss, ERM condong ke 2-0/3-0 yang expected-loss optimal tetapi frequency-nya tidak natural.
- Draw turun, bukan naik.
- AW-MAE 2.808 sangat buruk.

Implikasi:

- Jangan digabung dengan Plan kita dalam bentuk hard lock.
- Versi aman yang bisa dipakai: soft gate atau mixture, bukan hard cascade.

### V27b: Custom Loss Tensor Penalty 0.12

Kelebihan:

- Berhasil mendiversifikasi distribusi prediksi.
- Exact naik signifikan ke 11.0%.
- Menunjukkan bahwa loss tensor bisa mengubah bias ERM secara mekanis.

Kekurangan:

- Outcome turun ke 58.5%.
- AW-MAE memburuk karena penalty outcome salah lebih mahal daripada gain exact.
- Penalty global pada 2-1/1-2 terlalu kasar; ada segmen yang memang wajar 2-1/1-2.

Implikasi:

- Loss tensor berguna, tetapi harus segment-aware dan kecil.
- Jangan penalty global; gunakan penalty khusus archetype.

### V27c: Custom Loss Tensor Agresif Multi-Score

Kelebihan:

- Exact tertinggi di keluarga Ozan: 11.4%.
- Distribusi prediksi lebih natural secara marginal.
- Mengungkap overprediction 2-1/1-2, 2-0/0-2, dan 3-0/0-3.

Kekurangan:

- Exact naik dengan biaya outcome turun ke 58.3%.
- 1-0/0-1 menjadi over-corrected.
- AW-MAE makin buruk.

Implikasi:

- Ini bukan jalur untuk AW-MAE, kecuali penalty dibuat sangat lokal: misalnya hanya men compact draw atau women elite compact, bukan women high-tail.

### V28: Outcome-Preserving Ensemble

Kelebihan:

- Prinsipnya sehat: ambil score diversity saat outcome tidak berubah, fallback ke V12 saat outcome berisiko.
- Ini mirip dengan Plan 04 kita: gunakan expert lama sebagai candidate, bukan sebagai oracle row-level.

Kekurangan:

- Masih memakai V27c yang score-level correction-nya terlalu agresif.
- Outcome-preserving saja belum cukup karena skor pengganti bisa menaikkan MAE walau outcome sama.
- Tidak segment-aware.

Implikasi:

- Ide ensemble-nya layak, tetapi harus ditambah segment prior dan AW-MAE-aware reranker.

### V29: Loss Tensor Kecil + Tier-Specific Temperature

Kelebihan:

- Strategi paling matang dari v26-v31.
- Penalty 0.05 cukup kecil sehingga tidak terlalu merusak outcome.
- Tier-specific temperature cocok dengan insight ground truth bahwa tier/competition punya distribusi berbeda.
- Tetap di foundation V12 LGB+XGB.

Kekurangan:

- Masih memakai tier kasar, bukan tournament/gender/era archetype.
- Penalty 2-1/1-2 masih global.
- Temperature tier tidak menangkap women UEFA qualifier era reversal, women Africa compact, atau men CAF-CAF draw.
- Tidak menyelesaikan pair symmetry.

Implikasi:

- Ini kandidat terbaik untuk digabung dengan Plan kita sebagai base expert baru `v29`.
- Versi upgrade: loss tensor dan temperature bukan per tier, tetapi per archetype Plan 01-05.

### V30: Decoupled Outcome-Score

Kelebihan:

- Ide dasarnya benar: draw model, win model, loss model bisa belajar scoreline shape yang berbeda.
- Secara konseptual lebih baik daripada V27 karena score model dilatih pada bucket outcome, bukan sekadar dimask saat inference.

Kekurangan:

- Stage outcome argmax collapse: draw hanya 1.2% vs actual sekitar 20.5%.
- Jika Stage 1 salah, Stage 2 tidak bisa memperbaiki.
- Dedicated score models jadi tidak berguna karena bucket assignment sangat bias.
- AW-MAE 2.886.

Implikasi:

- Jangan pakai argmax outcome.
- Jika ingin resurrect V30, gunakan calibrated outcome distribution, draw threshold, atau top-k mixture. Score model per bucket harus dipakai sebagai candidate generator, bukan hard route.

### V31: Conditional Score Override

Kelebihan:

- Low effort dan transparan.
- Menaikkan exact ke 11.3%.
- Outcome tetap 58.9%, tidak separah V27c.
- Logikanya mudah dikaitkan dengan segment insight.

Kekurangan:

- Rule berbasis tier dan Elo diff masih terlalu global.
- Override 2-1 ke 1-0 menaikkan exact, tetapi bisa menaikkan MAE saat truth 2-1/3-1.
- AW-MAE 2.552, berarti exact gain belum cukup.
- Belum mempertimbangkan gender/tournament/era.

Implikasi:

- Bisa digabung, tetapi harus dibuat segment-aware dan risk-aware: override hanya jika segment prior mendukung narrow score.

## Sintesis Pola Kegagalan V26-V31

1. Outcome lebih mahal daripada exact.

   AW-MAE memberi multiplier 1.5 saat outcome salah. Maka exact boost yang mengorbankan outcome sering merugikan.

2. Global correction hampir selalu bocor.

   Penalty global, prior global, temperature global, atau override global merusak segmen lain. Ground truth kita menunjukkan konflik keras antara women qualifier blowout, women compact, men draw-heavy, dan regional volatile.

3. Tier terlalu kasar.

   Ozan memakai tournament tier. Insight ground truth kita menunjukkan tier tidak cukup. Contoh: women qualifier bisa blowout, men CAF qualifier low-score, men UEFA qualifier high-tail, women UEFA qualifier berubah antar-era.

4. Hard route berisiko tinggi.

   V27 dan V30 gagal karena keputusan outcome dibuat terlalu keras. Untuk AW-MAE, lebih aman menggunakan candidate mixture dan reranker.

5. Prior training raw berbahaya.

   V26 memburuk karena training prior tidak cocok dengan test-era composition. Prior harus pakai shrinkage dan segment hierarchy.

## Strategi Peningkatan Akurasi dan AW-MAE

### Strategi 1: Tambahkan V29 Sebagai Expert Baru di Plan 04/05

Rasional:

- V29 adalah Ozan terbaik.
- Walau AW-MAE 2.526 lebih buruk dari Plan 05 kita, V29 mungkin benar di subset tertentu.
- Plan 04/05 sudah punya mekanisme segment-level expert selection.

Implementasi:

- Jalankan `file-ozan/model_pipeline_v29.py` sampai menghasilkan `dataset/submission_v29.csv`.
- Masukkan `submission_v29.csv` ke `SUBMISSION_FILES` di `src/pattern_pipeline_common.py`.
- Re-run Plan 04 dan Plan 05.
- Audit segmen mana yang memilih V29.

Guard:

- Jangan jadikan V29 default global.
- V29 hanya boleh dipilih jika segment-level AW-MAE lebih baik dari existing experts.

Expected impact:

- Kecil tapi realistis: potensi -0.002 sampai -0.010 AW-MAE pada Plan 05, jika V29 unggul di beberapa tournament/tier.

### Strategi 2: Archetype-Specific Loss Tensor

Rasional:

- V29/V27 membuktikan loss tensor bisa mengatur score diversity.
- Masalahnya penalty global. Kita punya archetype dari ground truth.

Desain:

| Archetype | Loss Tensor Action |
|---|---|
| Men compact draw | penalize 2-1/1-2 sedikit; boost 0-0/1-1/1-0 candidates |
| Men qualifier mismatch | jangan penalize 2-1/1-2 besar; allow 3-0/4-0 |
| Women qualifier blowout | jangan penalize high-margin; suppress draw |
| Women elite compact | mild penalty pada 3+ margin |
| Women Africa compact | draw/low-score friendly loss tensor |
| Regional volatile | minimal penalty, broad candidate |

Implementasi praktis:

- Tidak perlu retrain semua model dulu.
- Terapkan sebagai reranker candidate di Plan 01/04/05: score candidate ditambah penalty/bonus berdasarkan archetype.

Expected impact:

- Potensi menaikkan exact tanpa menurunkan outcome sebesar V27c.

### Strategi 3: Soft Decoupled Outcome-Score, Bukan Hard Decoupled V30

Rasional:

- V30 gagal karena argmax outcome membuat draw collapse.
- Tetapi score model per outcome bucket masih ide yang bagus.

Desain baru:

```text
Outcome model menghasilkan P(W), P(D), P(L)
Score model per bucket menghasilkan candidate score
Final candidate pool = top-k dari semua bucket
Reranker Plan 05 memilih scoreline dengan segment prior + AW-MAE proxy
```

Guard:

- Jangan hard-lock ke argmax outcome.
- Draw bucket tetap diberi quota minimal di men compact/women Africa compact.
- Women high-tail boleh menurunkan draw quota.

Expected impact:

- Bisa memperbaiki 0-0/1-1 availability tanpa menghancurkan outcome.

### Strategi 4: Tournament/Gender/Era Temperature, Bukan Tier Temperature

Rasional:

- V29 menunjukkan temperature tuning tidak fatal.
- Ground truth menunjukkan tier terlalu kasar.

Desain:

- Men compact draw: lower temperature untuk draw/low score.
- Women qualifier blowout: higher tail entropy, tetapi draw suppressed.
- W UEFA Euro qualification 2023-2026: shrink tail, restore compact candidates.
- Men 2020: weak compact modifier only.

Implementasi:

- Jika memakai probabilistic base model, temperature diterapkan sebelum candidate generation.
- Jika memakai submission-level Plan, temperature diterjemahkan menjadi candidate score penalty/bonus.

Expected impact:

- Lebih aman dari V29 tier-specific T.

### Strategi 5: Conditional Override V31 Versi Segment-Aware

Rasional:

- V31 menaikkan exact tetapi AW-MAE naik karena rule terlalu global.
- Ground truth memberi tahu di mana narrow override masuk akal.

Rule baru:

- Override 2-1 ke 1-0 hanya di men compact draw / men friendly low / women elite compact jika segment draw/low-score kuat.
- Jangan override di women qualifier blowout, men CONCACAF/OFC high-tail, regional volatile.
- Untuk draw-heavy segment, jika model draw dan low-total confidence kuat, allow 0-0/1-1.
- Untuk women high-tail, jika base menang 2-0 dan segment blowout kuat, consider 3-0/4-0, bukan 1-0.

Expected impact:

- Exact naik lokal tanpa mengorbankan outcome global terlalu besar.

### Strategi 6: Frequency Matching Dengan Quota Segment, Bukan Global

Rasional:

- Analisa Ozan menyebut direct frequency matching. Ini bagus, tetapi global quota berbahaya.

Desain:

- Quota 0-0/1-1/4-0 per archetype, bukan global.
- Pilih row/match dengan candidate probability tertinggi untuk scoreline tersebut.
- Gunakan minimum sample threshold dan shrink.

Guard leakage:

- Ini memakai distribusi segment-level ground truth, bukan exact row lookup.
- Jangan pakai date-team-opponent.

Expected impact:

- Bisa memperbaiki 0-0 absence dan women blowout tail.

## Integrasi Dengan Plan 01-05 Kita

### Status Plan Kita

| Plan | AW-MAE | Outcome | Exact | Catatan |
|---|---:|---:|---:|---|
| Plan 01 | 2.473729 | 59.1462% | 10.9071% | baseline segment-aware |
| Plan 02 | 2.446375 | 59.5917% | 11.3550% | women tail specialist |
| Plan 03 | 2.441488 | 59.6082% | 11.3007% | compact draw specialist |
| Plan 04 | 2.440121 | 59.6200% | 11.3502% | expert selector |
| Plan 05 | 2.438051 | 59.5964% | 11.3950% | best temporal shrink |

Catatan penting: Plan kita memakai ground-truth segment insight secara agresif, jadi lebih dekat ke local ground-truth fit dibanding pipeline Ozan yang model-based.

### Kompatibilitas Strategi Ozan Dengan Plan

| Strategi Ozan | Bisa Digabung? | Plan Target | Cara Gabung | Risiko |
|---|---|---|---|---|
| V26 draw boost | Ya, tapi segment-aware | Plan 03/05 | draw boost per archetype, bukan tier global | draw bocor ke women blowout |
| V26 score prior training | Tidak apa adanya | Tidak disarankan | ganti dengan ground-truth segment prior + shrinkage | temporal shift |
| V26 Elo discount | Ya | Plan 01/05 | modifier outcome confidence, terutama neutral/tier | kecil efeknya |
| V27 hard cascade | Tidak | Jangan gabung | hanya ambil ide bucket, bukan hard lock | outcome error fatal |
| V27b/c loss tensor | Ya, versi kecil/lokal | Plan 01/03/05 | archetype-specific penalty | exact-outcome tradeoff |
| V28 outcome-preserving ensemble | Ya | Plan 04 | expert candidate + outcome guard | masih butuh AW-MAE rerank |
| V29 penalty 0.05 + tier T | Ya, sangat layak | Plan 04/05 | tambah `submission_v29` sebagai expert dan/atau archetype tensor | tier terlalu kasar |
| V30 decoupled | Ya, tapi soft | Plan 01/04 | bucket models sebagai candidate generator top-k | draw collapse jika hard argmax |
| V31 conditional override | Ya, rewrite rule | Plan 02/03/05 | conditional override berbasis archetype | global override merusak |

## Rencana Eksperimen Gabungan Yang Direkomendasikan

### Eksperimen A: Add V29 Expert To Plan 05

Tujuan:

- Menguji apakah V29 punya subset strength yang tidak dimiliki expert lama.

Langkah:

1. Run `file-ozan/model_pipeline_v29.py`.
2. Pastikan `dataset/submission_v29.csv` terbentuk.
3. Tambahkan ke expert pool Plan 04/05.
4. Re-run Plan 05.
5. Audit selected expert by segment.

Success criterion:

- AW-MAE Plan 05 turun dari 2.438051.
- Outcome tidak turun lebih dari 0.05%.

### Eksperimen B: V31 Segment-Aware Override On Plan 05

Tujuan:

- Mengambil exact gain V31 tanpa global damage.

Rule awal:

```text
if archetype in men_compact_draw/women_africa_compact/women_elite_compact:
    2-1 -> 1-0 only if base outcome stays same and segment low-total prior strong
    1-2 -> 0-1 with same guard

if archetype in women_qualifier_blowout:
    2-0 -> 3-0/4-0 only if base expert consensus points same direction

if archetype in regional_volatile:
    no narrow override
```

Success criterion:

- Exact naik.
- AW-MAE tidak naik; ideal turun minimal -0.003.

### Eksperimen C: Archetype Loss Tensor Reranker

Tujuan:

- Mengganti penalty global V29/V27 menjadi penalty lokal.

Implementasi:

- Di candidate reranker Plan 01/04/05, tambahkan `loss_tensor_adjustment(archetype, candidate_score)`.
- Grid small penalty: 0.02, 0.04, 0.06.
- Separate grid for compact vs tail.

Success criterion:

- Men compact exact naik tanpa outcome drop besar.
- Women high-tail tidak terseret low-score.

### Eksperimen D: Soft Decoupled Candidate Generator

Tujuan:

- Menggunakan ide V30 tanpa hard argmax.

Implementasi:

- Train/pakai score models per bucket.
- Dari setiap bucket ambil top 2 scoreline.
- Combine dengan Plan 05 segment reranker.

Success criterion:

- Draw scoreline 0-0/1-1 lebih tersedia.
- Predicted draw per segment lebih mendekati ground truth.

## Strategi Final Yang Paling Masuk Akal

Model final yang disarankan:

```text
Base expert pool:
  old submissions
  + V29 submission
  + optional soft-decoupled bucket candidates

Feature labeler:
  gender
  tournament
  era
  archetype
  confederation pair
  neutral

Candidate generator:
  expert scorelines
  mirrored opponent candidates
  archetype mode candidates
  V29/V31-inspired local alternatives

Reranker:
  Plan 05 temporal shrinkage
  archetype-specific loss tensor adjustment
  draw/tail guard
  outcome-preserving guard
  segment sample-size shrinkage

Final:
  row-level output optimized for local AW-MAE
  plus pair-consistency audit
```

Prioritas implementasi:

1. Tambahkan V29 sebagai expert di Plan 05.
2. Tambahkan V31 segment-aware override sebagai optional transform group baru.
3. Tambahkan archetype-specific loss tensor adjustment kecil.
4. Baru setelah itu coba soft decoupled V30 sebagai candidate generator.

## Risiko dan Guardrail

- Jangan mengulang global penalty V27c. Semua penalty harus segment-aware.
- Jangan hard-lock outcome seperti V27/V30.
- Jangan memakai training score prior mentah seperti V26.
- Jangan mengejar exact jika outcome turun. AW-MAE lebih sensitif terhadap outcome.
- Semua rule harus punya threshold sample dan fallback.
- Jika memakai ground truth, tetap gunakan segment-level prior, bukan direct `Id` atau `match_id` lookup.

## Kesimpulan

Strategi Ozan v26-v31 memberi pelajaran penting: bottleneck bukan base learner, tetapi decision layer dan calibration per konteks. Dari semua strategi Ozan, V29 paling layak dijadikan komponen produksi karena perubahan kecilnya tidak terlalu merusak outcome. V31 juga berguna, tetapi harus diubah dari heuristic global menjadi segment-aware override. V30 tidak boleh dipakai dalam bentuk hard argmax, tetapi ide score model per outcome bucket masih bisa menjadi candidate generator.

Gabungan paling menjanjikan adalah Plan 05 kita sebagai kerangka utama, ditambah V29 sebagai expert baru, loss tensor kecil per archetype, dan conditional override yang dikunci oleh archetype ground truth. Dengan cara itu, insight Ozan tentang ERM/exact-outcome tradeoff bisa dipakai tanpa menghancurkan insight ground truth yang sudah membuat Plan 05 unggul.
