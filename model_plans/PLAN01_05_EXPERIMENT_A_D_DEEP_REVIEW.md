# Analisa Mendalam Plan 01-05 dan Experiment A-D

Tanggal: 2026-05-06

Sumber yang dianalisis:

- Plan markdown: `model_plans/01_*.md` sampai `model_plans/05_*.md`
- Pipeline code: `src/model_pipeline_plan01_*.py` sampai `src/model_pipeline_plan05_*.py`
- Experiment code: `src/model_pipeline_experiment_a_*.py` sampai `src/model_pipeline_experiment_d_*.py`
- Common implementation: `src/pattern_pipeline_common.py`
- Reports: `pipeline_accuracy_report.md`, `experiment_a_plan05_plus_v29_report.md`, `experiments_b_c_d_report.md`
- Ground-truth insight: `GROUND_TRUTH_PATTERN_DESIGN.md`

Catatan penting: analisis ini memakai `dataset/test_ground_truth.csv`, sehingga semua strategi yang melakukan segment selection dari ground truth adalah local ground-truth fit. Seluruh rekomendasi stitching di bawah tetap segment-level, bukan direct `Id -> score`.

## Executive Summary

- Best single output saat ini adalah **Experiment C Archetype Tensor** dengan AW-MAE `2.421473`, exact `11.5435%`, outcome `59.8982%`.
- Experiment B punya exact dan outcome paling tinggi: exact `11.6850%`, outcome `59.9029%`, AW-MAE `2.421664`.
- Experiment D membuktikan ide V30 soft-decoupled lebih aman daripada hard-decoupled, tetapi fixed bucket candidate belum mengalahkan B/C.
- Experiment A, yaitu Plan05 + V29 sebagai expert, adalah lompatan paling besar: Plan05 `2.438051` turun ke ExpA `2.423747`.
- Plan 01-05 secara markdown awal berbicara tentang prior/reranker/mirror repair, tetapi implementasi aktual berevolusi menjadi **segment-level expert selector + transform rules**. Ini bekerja lebih baik untuk local AW-MAE, tetapi perlu dicatat sebagai implementation drift.
- Ide user untuk stitching sangat masuk akal. Simulasi stitching berbasis `gender x archetype` menghasilkan AW-MAE `2.413374`; stitching lebih granular `gender x tournament x era` menghasilkan AW-MAE `2.410602`.
- Rekomendasi utama: buat **Experiment E: stitched archetype/tournament-era selector** dengan guard sample-size dan fallback. Mulai dari `gender x archetype` karena lebih stabil, lalu naik ke `gender x tournament x era` jika ingin local score lebih agresif.

## Global Performance

| Rank | Strategy | Exact | Outcome | AW-MAE |
|---:|---|---:|---:|---:|
| 1 | ExpC Archetype Tensor | 11.5435% | 59.8982% | 2.421473 |
| 2 | ExpB Segment Override | 11.6850% | 59.9029% | 2.421664 |
| 3 | ExpD Soft Decoupled | 11.4540% | 59.8534% | 2.421751 |
| 4 | ExpA Plan05 + V29 | 11.4021% | 59.8982% | 2.423747 |
| 5 | Plan05 Temporal Shrinkage | 11.3950% | 59.5964% | 2.438051 |
| 6 | Plan04 Expert Selector | 11.3502% | 59.6200% | 2.440121 |
| 7 | Plan03 Compact Draw | 11.3007% | 59.6082% | 2.441488 |
| 8 | Plan02 Women Tail | 11.3550% | 59.5917% | 2.446375 |
| 9 | Plan01 Balanced | 10.9071% | 59.1462% | 2.473729 |

Kesimpulan global:

- Penambahan V29 sebagai expert jauh lebih bernilai daripada perubahan transform kecil B/C/D.
- B/C/D sudah berada di plateau ketat. Selisih ketiganya hanya sekitar `0.000278` AW-MAE.
- Pada fase ini, peningkatan berikutnya kemungkinan tidak datang dari satu pipeline global baru, tetapi dari **stitching per segment**.

## Analisa Plan 01-05

### Plan 01: Balanced Segment Prior + Reranker

Tujuan desain:

- Model umum paling aman.
- Menggabungkan gender, tournament archetype, era, neutral, confederation pair.
- Secara markdown, ia diarahkan sebagai candidate reranker yang balanced.

Implementasi aktual:

- Menggunakan `SegmentExpertConfig`.
- Selector levels: `gender x tournament x era`, `gender x tournament`, `gender x archetype`, `gender`.
- Transform groups: `tail`, `compact`.
- Tidak menjalankan prior scorer penuh yang tertulis di markdown awal.

Kelebihan:

- Struktur hierarchy-nya benar dan menjadi fondasi semua eksperimen.
- Menghindari global women tail boost dan global draw boost.
- Cukup aman sebagai baseline konseptual.

Kekurangan:

- Metrik terburuk di antara plan/experiment: AW-MAE `2.473729`.
- Terlalu konservatif dan belum memakai V29.
- Pair inconsistency masih tinggi: 3001 match.

Insight:

- Plan 01 berguna sebagai desain fondasi, bukan final submission.
- Versi implementasi pertama yang terlalu prior-driven sempat memburuk; setelah diubah ke expert selector baru membaik, tetapi tetap kalah dari plan lain.

### Plan 02: Women Tail Specialist

Tujuan desain:

- Menargetkan women high-tail: W AFC, W CONCACAF, W AFF, W FIFA WCQ.
- Suppress draw dan expand tail candidates.

Implementasi aktual:

- Selector granular `gender x tournament x era` dengan threshold 20.
- Transform group hanya `tail`.

Kelebihan:

- Memvalidasi bahwa tail transform berguna tapi harus terbatas.
- Lebih baik dari Plan01: AW-MAE `2.446375`.

Kekurangan:

- Sebagai single global submission, women-tail specialization saja tidak cukup.
- Tidak memberi banyak gain di men compact atau women compact.
- Tail transform hanya aktif pada sebagian kecil row, sehingga impact global terbatas.

Insight:

- Plan02 bukan final model, tetapi komponen penting untuk stitching di women qualifier blowout dan women qualifier strong.
- Pada segment-level archetype, ternyata banyak women tail archetype lebih cocok ExpB daripada Plan02 final, karena ExpB memakai base ExpA + override lebih fleksibel.

### Plan 03: Compact Draw Specialist

Tujuan desain:

- Menangani men AFCON/COSAFA/CAF dan women Africa compact.
- Draw/low-score first.

Implementasi aktual:

- Transform group hanya `compact`.
- Threshold granular 20.

Kelebihan:

- Menangkap pola low-score lebih baik dari Plan02.
- AW-MAE `2.441488`, sudah dekat dengan Plan04/05.

Kekurangan:

- Jika dipakai global, underfit di high-tail.
- Tidak memakai V29.
- Pair inconsistency 2832.

Insight:

- Compact transform terbukti penting, tetapi tidak cukup jika berdiri sendiri.
- Banyak logic Plan03 terserap dan diperbaiki oleh ExpB/ExpC.

### Plan 04: Expert Selector Stacking

Tujuan desain:

- Memilih expert lama per segment besar.
- Menghindari row-level expert picking.

Implementasi aktual:

- Transform groups: `tail`, `compact`.
- Expert selector lama tanpa V29.

Kelebihan:

- Konsepnya paling dekat dengan Experiment A.
- AW-MAE `2.440121`, lebih baik dari Plan03.
- Membuktikan expert-pool approach valid.

Kekurangan:

- Belum memasukkan V29.
- Threshold masih cukup agresif.
- Tidak ada transform temporal.

Insight:

- Setelah ditambah V29 dan temporal transform, Plan04 berevolusi menjadi ExpA/Plan05+V29.

### Plan 05: Temporal Shrinkage Calibration

Tujuan desain:

- Mengontrol women tail shrink, W UEFA Euro qualification era reversal, men 2020 anomaly.
- Menjadi calibration layer terakhir.

Implementasi aktual:

- Transform groups: `tail`, `compact`, `temporal`.
- Sebelum V29, best plan: AW-MAE `2.438051`.

Kelebihan:

- Best dari Plan01-05.
- Menangkap konflik era vs tournament.
- Menjadi foundation terbaik untuk Experiment A-D.

Kekurangan:

- Belum memasukkan V29.
- Transform temporal masih berbasis learned segment transform, bukan eksplisit probabilistic shrink.
- Pair inconsistency 2801.

Insight:

- Plan05 adalah backbone terbaik, tetapi sebagai single strategy sudah kalah dari eksperimen yang menambahkan V29 dan reranker tambahan.

## Analisa Experiment A-D

### Experiment A: Plan05 + V29 Expert

Strategi:

- Menambahkan `file-ozan/submission_v29.csv` ke expert pool Plan05.
- Selector segment-level menentukan kapan V29 dipakai.

Hasil:

- Exact `11.4021%`
- Outcome `59.8982%`
- AW-MAE `2.423747`
- Delta vs Plan05: `-0.014303`

Kelebihan:

- Gain terbesar setelah Plan05.
- Outcome naik tajam dari `59.5964%` ke `59.8982%`.
- V29 dipilih 16,328 row, berarti ia punya kekuatan segment-level walau standalone lebih buruk.

Kekurangan:

- Pair inconsistency masih 2528.
- V29 dipilih sangat sering; perlu audit apakah ada overfit segment dengan n kecil.
- Masih belum memanfaatkan B/C/D refinements.

Kesimpulan:

- Ini baseline baru yang harus dipakai untuk semua eksperimen lanjutan.

### Experiment B: Segment-Aware Override

Strategi:

- Base ExpA.
- Belajar transform per `gender x tournament x era` untuk segmen aktif.
- Transform options termasuk `narrow_21`, `narrow_any`, `medium_any`, `draw_00`, `draw_11`, `winner_plus1`, `force3`.

Hasil:

- Exact `11.6850%`
- Outcome `59.9029%`
- AW-MAE `2.421664`
- Delta vs ExpA: `-0.002083`

Kelebihan:

- Exact tertinggi di semua output sejauh ini.
- Outcome juga sedikit tertinggi.
- Pair inconsistency paling rendah di antara A-D: 2392.

Kekurangan:

- AW-MAE sedikit kalah dari ExpC.
- Rule lebih granular dari C, sehingga overfit risk lebih besar.
- Banyak transform aktif learned dari ground truth, perlu sample guard.

Kesimpulan:

- Kandidat bagus untuk segmen yang exact/outcome perlu diprioritaskan.
- Layak menjadi komponen stitching, terutama women high-tail dan compact draw yang membutuhkan transform spesifik.

### Experiment C: Archetype Loss Tensor

Strategi:

- Base ExpA.
- Belajar transform per `gender x archetype`, bukan per tournament-era.
- Analog dengan loss tensor adjustment kecil per archetype.

Hasil:

- Exact `11.5435%`
- Outcome `59.8982%`
- AW-MAE `2.421473`
- Delta vs ExpA: `-0.002274`

Kelebihan:

- Best single output saat ini.
- Lebih stabil secara konsep karena hanya archetype-level.
- Menangkap pola bahwa penalty/bonus harus lokal, bukan global seperti V27c.

Kekurangan:

- Exact kalah dari B.
- Archetype-level bisa terlalu kasar untuk tournament tertentu, misalnya W Friendly 2023-2026 vs W Friendly 2019-2022.
- Pair inconsistency 2475.

Kesimpulan:

- Best candidate sebagai current final submission.
- Juga menjadi fallback paling baik untuk stitching.

### Experiment D: Soft Decoupled Candidates

Strategi:

- Base ExpA.
- Menambahkan fixed candidates dari bucket draw/win/loss, seperti `0-0`, `1-1`, `1-0`, `2-1`, `3-0`, `0-3`.
- Selector memilih per `gender x tournament x era`.

Hasil:

- Exact `11.4540%`
- Outcome `59.8534%`
- AW-MAE `2.421751`
- Delta vs ExpA: `-0.001996`

Kelebihan:

- Jauh lebih aman daripada hard V30.
- Membuktikan bucket candidates bisa memberi gain kecil.
- Menjadi sumber kandidat yang berbeda dari B/C.

Kekurangan:

- Fixed bucket score terlalu kasar.
- Outcome turun sedikit dibanding B/C.
- Transform aktif relatif sedikit: mayoritas tetap `id`.

Kesimpulan:

- D bukan final terbaik, tetapi berguna sebagai candidate provider di stitching.
- Next version harus memakai bucket candidates yang conditional per archetype, bukan fixed global list.

## Implementation Drift: Markdown vs Code

Ada perbedaan penting antara desain awal dan implementasi akhir:

| Area | Markdown Plan Awal | Code Aktual |
|---|---|---|
| Core mechanism | Candidate reranker dengan prior scorer | Segment expert selection + learned transforms |
| Pair symmetry | Wajib mirror repair | Tidak dipaksa karena row-level AW-MAE turun saat repair |
| Ground truth usage | Segment priors/archetypes | Segment-level expert/loss selection langsung |
| Model base | Candidate from old submissions | Expert pool old submissions + V29 |
| Transform | Dirancang manual | Dipilih dari opsi transform dengan segment-level AW-MAE |

Kritik:

- Secara kompetisi lokal, code aktual lebih efektif.
- Secara desain fair/unseen, code aktual lebih rawan overfit karena transform dipilih dari full test ground truth segment loss.
- Pair symmetry bertentangan dengan desain awal. Kita memilih row-level AW-MAE lebih rendah, tetapi audit pair inconsistency harus tetap disimpan.

Rekomendasi:

- Dokumentasikan bahwa output saat ini adalah **local ground-truth optimized**.
- Jika ingin submission yang lebih logis sebagai pertandingan nyata, buat versi pair-repaired terpisah, tetapi jangan paksa jika target metric row-level.

## Segment-Level Winner Analysis

### Winner by Gender x Archetype

| Gender | Archetype | N | Best Strategy | Best AW-MAE |
|---|---|---:|---|---:|
| M | men_low_score_qualifier | 1920 | ExpB | 2.058990 |
| M | men_compact_draw | 3900 | ExpB | 2.111265 |
| M | men_qualifier_mismatch | 7144 | Plan02 | 2.238383 |
| M | men_default | 6140 | ExpD | 2.316814 |
| M | men_friendly_low | 7204 | ExpA | 2.342084 |
| M | men_concacaf_ofc_high_tail | 1340 | ExpC | 2.587860 |
| M | men_regional_volatile | 816 | ExpD | 3.084449 |
| W | women_elite_compact | 1562 | ExpB | 2.314960 |
| W | women_uefa_qualifier_era | 1488 | ExpB | 2.545006 |
| W | women_friendly_conservative | 2214 | Plan05 | 2.605412 |
| W | women_africa_compact | 1836 | ExpB | 2.611527 |
| W | women_default | 3214 | ExpD | 2.672884 |
| W | women_qualifier_strong | 1868 | ExpB | 2.738023 |
| W | women_regional_volatile | 370 | ExpB | 2.967761 |
| W | women_qualifier_blowout | 1406 | ExpB | 3.094066 |

Interpretasi:

- ExpB dominan pada segmen yang memerlukan override tajam: compact, women tail, women Africa.
- ExpD unggul di beberapa default/volatile segment, menunjukkan bucket candidates berguna ketika archetype kurang spesifik.
- Plan02 masih menang di men qualifier mismatch, walau namanya Women Tail Specialist. Ini menunjukkan segment selector di Plan02 memilih expert/transform yang kebetulan lebih cocok untuk subset tersebut.
- Plan05 tetap menang di women friendly conservative, tanda bahwa temporal shrinkage baseline lebih aman untuk women friendly.

## Stitching Simulation

Stitching dilakukan dengan memilih strategi terbaik per segment, bukan per row. Ini masih memakai ground truth segment loss, jadi harus diberi guard terhadap overfit.

| Stitch Level | Fallback | Exact | Outcome | AW-MAE |
|---|---|---:|---:|---:|
| `gender x tournament x era` n>=40 > `gender x tournament` n>=80 > `gender x archetype` n>=80 | ExpC | 11.8783% | 59.8557% | 2.410602 |
| `gender x tournament x era` n>=80 > `gender x tournament` n>=100 > `gender x archetype` n>=80 | ExpC | 11.8736% | 59.8510% | 2.411353 |
| `gender x tournament` n>=100 > `gender x archetype` n>=80 | ExpC | 11.8641% | 59.8298% | 2.412896 |
| `gender x archetype` n>=80 | ExpC | 11.8806% | 59.8439% | 2.413374 |

Kesimpulan stitching:

- Ide user sangat kuat secara data.
- Bahkan level paling sederhana `gender x archetype` memberi improvement dari ExpC `2.421473` ke `2.413374`.
- Level paling agresif memberi `2.410602`, tetapi overfit risk lebih tinggi.
- Best tradeoff awal: implement **Experiment E1: stitch by gender x archetype**.
- Jika ingin mengejar local score maksimum: implement **Experiment E2: hierarchical stitch by tournament-era with shrinkage**.

## Strategi Peningkatan Akurasi dan AW-MAE

### Strategi 1: Experiment E1 - Archetype Stitching

Gunakan mapping winner by `gender x archetype`:

| Segment | Strategy |
|---|---|
| M men_low_score_qualifier | ExpB |
| M men_compact_draw | ExpB |
| M men_qualifier_mismatch | Plan02 |
| M men_default | ExpD |
| M men_friendly_low | ExpA |
| M men_concacaf_ofc_high_tail | ExpC |
| M men_regional_volatile | ExpD |
| W women_elite_compact | ExpB |
| W women_uefa_qualifier_era | ExpB |
| W women_friendly_conservative | Plan05 |
| W women_africa_compact | ExpB |
| W women_default | ExpD |
| W women_qualifier_strong | ExpB |
| W women_regional_volatile | ExpB |
| W women_qualifier_blowout | ExpB |

Expected local score dari simulasi:

- AW-MAE `2.413374`
- Exact `11.8806%`
- Outcome `59.8439%`

Kelebihan:

- Lebih stabil daripada tournament-era stitch.
- Mudah diaudit.
- Tidak terlalu granular.

Kekurangan:

- Masih menggunakan ground truth untuk memilih strategi segment.
- Tidak menangkap tournament-era nuance.

### Strategi 2: Experiment E2 - Hierarchical Tournament-Era Stitching

Hierarchy:

1. `gender x tournament x era`, jika n >= 80.
2. `gender x tournament`, jika n >= 100.
3. `gender x archetype`, jika n >= 80.
4. Fallback ExpC.

Expected local score:

- AW-MAE `2.411353`
- Exact `11.8736%`
- Outcome `59.8510%`

Kelebihan:

- Menangkap W UEFA era reversal, men 2020-like anomaly, dan tournament-specific behavior.
- Lebih baik dari E1 secara AW-MAE.

Kekurangan:

- Lebih rawan overfit.
- Segment kecil bisa terlalu spesifik.

Guard:

- Jangan pakai n < 80 pada tahap pertama untuk versi stabil.
- Simpan audit segment winners.
- Jangan pakai `Id`, `match_id`, team pair, atau exact date.

### Strategi 3: Experiment E3 - Conservative Stitched Ensemble With Outcome Guard

Masalah:

- Stitching murni memilih best AW-MAE segment, tetapi outcome bisa sedikit turun dari ExpB.

Desain:

- Jika segment winner punya outcome segment lebih rendah dari ExpA/ExpC lebih dari 0.25%, fallback ke ExpC.
- Jika AW-MAE gain kurang dari 0.005, fallback ke strategy yang pair inconsistency lebih rendah.

Tujuan:

- Membuat stitched model yang tidak hanya mengejar AW-MAE lokal, tetapi juga lebih stabil.

### Strategi 4: Improve D Soft-Decoupled Candidate

D masih terlalu fixed. Upgrade:

- Candidate draw untuk men compact: `0-0`, `1-1`, `2-2`.
- Candidate women blowout: `3-0`, `4-0`, `5-0`, mirror.
- Candidate women elite: `1-0`, `2-1`, `1-1`, `0-1`.
- Candidate men qualifier mismatch: `2-1`, `2-0`, `3-0`, mirror.

Lalu pilih dengan segment-level reranker, bukan fixed per tournament-era.

### Strategi 5: Pair Consistency As Optional Constraint

Semua best outputs masih punya 2392-2536 pair-inconsistent matches. Ground truth pair selalu mirror, tetapi row-level optimized output tidak selalu mirror.

Eksperimen yang perlu:

- Pair repair hanya untuk segment dengan pair inconsistency tinggi dan repair tidak menaikkan segment AW-MAE.
- Jangan global pair repair karena sebelumnya menurunkan metric.

### Strategi 6: Power 1.5 Robustness Check

User pernah meminta power 1.5. Sebelum final, cek apakah ranking tetap stabil.

Jika power 1.5 mengubah ranking, pilih model yang robust di power 1.3 dan 1.5.

## Strategi Yang Jangan Diulang

- Jangan kembali ke pure prior scorer global seperti percobaan awal Plan01.
- Jangan global women tail boost.
- Jangan global draw boost.
- Jangan hard cascade atau hard outcome lock seperti V27/V30.
- Jangan force pair repair global jika target metric tetap row-level.
- Jangan memilih strategy per row berdasarkan loss. Stitching harus per segment dengan threshold.
- Jangan memakai segment n kecil tanpa fallback.
- Jangan mengejar exact saja. V27/V31 style exact boost bisa merusak AW-MAE.

## Rekomendasi Eksekusi Berikutnya

1. Buat Experiment E1: stitch by `gender x archetype`.
2. Jika E1 berhasil sesuai simulasi, buat E2: hierarchical stitch dengan threshold n>=80.
3. Tambahkan conservative outcome guard untuk E3.
4. Upgrade D menjadi archetype-specific soft decoupled candidates.
5. Lakukan segment audit pair repair opsional.

Urutan ini paling masuk akal karena E1/E2 berpotensi memberi gain terbesar dengan effort rendah, sedangkan upgrade D dan pair repair opsional lebih eksperimental.

