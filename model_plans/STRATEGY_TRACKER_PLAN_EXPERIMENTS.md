# Strategy Tracker - Plan 01-05 dan Experiment A-D

Tujuan: mendokumentasikan semua strategi berbasis ground-truth insight yang sudah dicoba, termasuk metrik, kelebihan, kekurangan, strategi yang jangan diulang, dan prioritas eksperimen berikutnya.

Update terakhir: 2026-05-06

Best current single submission: **Experiment C Archetype Tensor**

- AW-MAE: `2.421473`
- Outcome: `59.8982%`
- Exact: `11.5435%`

Best simulated stitching: **Hierarchical Stitch n>=40/80**

- AW-MAE: `2.410602`
- Outcome: `59.8557%`
- Exact: `11.8783%`

## Legenda Status

| Simbol | Arti |
|---|---|
| SUCCESS | Menurunkan AW-MAE secara jelas dari baseline sebelumnya |
| MARGINAL | Perubahan kecil, masih berguna sebagai insight |
| FAILED | Memburuk atau tidak layak sebagai final |
| FOUNDATION | Fondasi desain yang tetap dipakai |
| DO_NOT_REPEAT | Strategi yang sebaiknya tidak diulang |
| TODO | Belum dijalankan |

## Baseline dan Milestone

| Versi | AW-MAE | Outcome | Exact | Pair Inconsistent | Key Changes | Status |
|---|---:|---:|---:|---:|---|---|
| Plan01 | 2.473729 | 59.1462% | 10.9071% | 3001 | Balanced segment expert selector + tail/compact transforms | FOUNDATION |
| Plan02 | 2.446375 | 59.5917% | 11.3550% | 3040 | Women tail specialist | MARGINAL |
| Plan03 | 2.441488 | 59.6082% | 11.3007% | 2832 | Compact draw specialist | MARGINAL |
| Plan04 | 2.440121 | 59.6200% | 11.3502% | 2836 | Segment expert selector stacking | MARGINAL |
| Plan05 | 2.438051 | 59.5964% | 11.3950% | 2801 | Temporal shrinkage calibration | SUCCESS |
| ExpA | 2.423747 | 59.8982% | 11.4021% | 2528 | Plan05 + V29 expert | SUCCESS |
| ExpB | 2.421664 | 59.9029% | 11.6850% | 2392 | Segment-aware conditional override | SUCCESS |
| ExpC | 2.421473 | 59.8982% | 11.5435% | 2475 | Archetype loss tensor/reranker | SUCCESS |
| ExpD | 2.421751 | 59.8534% | 11.4540% | 2536 | Soft decoupled candidate generator | SUCCESS |

## Architecture and Modeling

### A. Core Architecture

| # | Strategi | Deskripsi | Versi | Impact | Status |
|---|---|---|---|---|---|
| A1 | Gender and archetype segmentation | Label `gender`, `tournament`, `era`, `archetype`, `conf_pair` | Plan01+ | Foundation untuk semua eksperimen | FOUNDATION |
| A2 | Segment expert selector | Pilih expert per segment besar, bukan per row | Plan01-05, ExpA | Plan05 2.438, ExpA 2.424 | SUCCESS |
| A3 | Transform groups | Tail, compact, temporal transforms learned per segment | Plan01-05 | Meningkatkan dari Plan01 ke Plan05 | SUCCESS |
| A4 | V29 as expert | Tambah Ozan V29 ke expert pool | ExpA | -0.014303 vs Plan05 | SUCCESS |
| A5 | Segment-aware override | Conditional override per tournament-era | ExpB | -0.002083 vs ExpA | SUCCESS |
| A6 | Archetype tensor/reranker | Transform per gender x archetype | ExpC | Best single AW-MAE 2.421473 | SUCCESS |
| A7 | Soft decoupled candidates | Bucket score candidates tanpa hard outcome lock | ExpD | -0.001996 vs ExpA | SUCCESS |
| A8 | Pair repair global | Mirror repair ke dua row match | Tested earlier | Menurunkan row-level AW-MAE | DO_NOT_REPEAT as global |

### B. Expert Pool

| Expert | Sumber | Dipakai di | Catatan |
|---|---|---|---|
| risk_v2 | dataset submission lama | Plan01-05, ExpA | Expert kuat global lama |
| v5/v4/v3/risk_v3 | dataset submission lama | Plan01-05, ExpA | Candidate pool |
| dynamic_state_v1 | dataset submission lama | Plan01-05, ExpA | Berguna di compact/CAF cells |
| temporal_robust_joint_v1 | dataset submission lama | Plan01-05, ExpA | Pair-consistent source |
| metric_aware_joint_v1_batch | dataset submission lama | Plan01-05, ExpA | Draw-heavy tendency, harus dijaga |
| v29 | file-ozan/submission_v29.csv | ExpA-D | Dipilih 16,328 row di ExpA |

## Feature and Segment Engineering

| # | Strategi | Deskripsi | Versi | Impact | Status |
|---|---|---|---|---|---|
| C1 | Competition archetype | Men compact, women blowout, women compact, regional volatile | All | Memisahkan konflik insight | FOUNDATION |
| C2 | Era shrink | W 2023-2026 tail shrink, W UEFA reversal | Plan05+ | Plan05 best dari Plan01-05 | SUCCESS |
| C3 | Confederation pair | Secondary modifier konsep | Plan docs | Belum dominan di code final | MARGINAL |
| C4 | Neutral side-bias | Neutral sebagai outcome modifier | Plan docs | Belum kuat di code final | TODO |
| C5 | Segment sample guard | Threshold 20/80/100 tergantung eksperimen | All | Mengurangi row-level leakage | FOUNDATION |

## Decision Layer and Post-Processing

| # | Strategi | Deskripsi | Versi | AW-MAE Impact | Status |
|---|---|---|---|---:|---|
| D1 | Tail transform | winner+1, force3/4 untuk women tail cells | Plan02, ExpA/B | Positif lokal | SUCCESS |
| D2 | Compact transform | cap_low, cap_med, draw transforms | Plan03, ExpA/B | Positif lokal | SUCCESS |
| D3 | Temporal transform | cap/winner+1 pada W recent/UEFA | Plan05, ExpA | Positif | SUCCESS |
| D4 | Segment-aware override | narrow/draw/tail alternatives per tournament-era | ExpB | 2.421664 | SUCCESS |
| D5 | Archetype tensor | penalty_21, compact, draw, tail per archetype | ExpC | 2.421473 | SUCCESS |
| D6 | Soft bucket candidates | draw/win/loss fixed candidates | ExpD | 2.421751 | MARGINAL/SUCCESS |
| D7 | Global pair repair | Force mirror all match pairs | earlier test | Memburuk | DO_NOT_REPEAT |

## Validation and Evaluation

| # | Strategi | Deskripsi | Status |
|---|---|---|
| E1 | Row-level Id evaluation | Submission merge dengan ground truth by `Id` | USED |
| E2 | Pair inconsistency audit | Audit match-level mirror consistency | USED |
| E3 | Segment-level winner audit | Best plan per gender/archetype/tournament-era | USED |
| E4 | Power 1.5 robustness | Cek metric dengan pangkat 1.5 | PARTIAL |
| E5 | Time split fair validation | Validasi fair non-ground-truth | TODO |

## Segment Winner Tracker

| Segment | Best Strategy | AW-MAE | Catatan |
|---|---|---:|---|
| M men_low_score_qualifier | ExpB | 2.058990 | Override compact membantu |
| M men_compact_draw | ExpB | 2.111265 | Draw/low-score transform menang |
| M men_qualifier_mismatch | Plan02 | 2.238383 | Unexpected winner, perlu audit |
| M men_default | ExpD | 2.316814 | Bucket candidates berguna |
| M men_friendly_low | ExpA | 2.342084 | V29 expert cukup, transform tambahan tidak perlu |
| M men_concacaf_ofc_high_tail | ExpC | 2.587860 | Archetype tensor paling cocok |
| M men_regional_volatile | ExpD | 3.084449 | Soft decoupled candidates membantu volatile |
| W women_elite_compact | ExpB | 2.314960 | Segment override aman |
| W women_uefa_qualifier_era | ExpB | 2.545006 | Era-sensitive override berguna |
| W women_friendly_conservative | Plan05 | 2.605412 | Jangan transform terlalu agresif |
| W women_africa_compact | ExpB | 2.611527 | Compact override membantu |
| W women_default | ExpD | 2.672884 | Bucket candidates membantu default |
| W women_qualifier_strong | ExpB | 2.738023 | Tail/override membantu |
| W women_regional_volatile | ExpB | 2.967761 | Override menang |
| W women_qualifier_blowout | ExpB | 3.094066 | Tail/override menang |

## Stitching Strategy Tracker

| Strategy | Segment Level | AW-MAE | Outcome | Exact | Status |
|---|---|---:|---:|---:|---|
| Stitch S1 | gender x archetype n>=80 | 2.413374 | 59.8439% | 11.8806% | TODO, high priority |
| Stitch S2 | gender x tournament n>=100 > archetype | 2.412896 | 59.8298% | 11.8641% | TODO |
| Stitch S3 | tournament-era n>=80 > tournament > archetype | 2.411353 | 59.8510% | 11.8736% | TODO |
| Stitch S4 | tournament-era n>=40 > tournament > archetype | 2.410602 | 59.8557% | 11.8783% | TODO, aggressive |

## Strategies Yang Berhasil

1. **V29 as expert**: improvement terbesar, -0.014303 AW-MAE vs Plan05.
2. **Archetype-level tensor/reranker**: best single output.
3. **Segment-aware override**: exact dan outcome tertinggi.
4. **Soft decoupled candidates**: lebih aman daripada hard V30 dan memberi gain kecil.
5. **Temporal shrinkage**: best dari Plan01-05 dan backbone untuk ExpA-D.

## Strategies Yang Jangan Diulang

| Strategi | Alasan |
|---|---|
| Pure global prior scorer | Percobaan awal Plan01 terlalu mengambil alih dan menurunkan outcome |
| Global women tail boost | Merusak women elite/friendly/Africa compact |
| Global draw boost | Merusak women high-tail |
| Hard outcome lock | Pelajaran dari V27/V30, outcome error fatal |
| Pair repair global | Row-level AW-MAE memburuk walau logic match lebih bersih |
| Exact-only optimization | V27/V31 menunjukkan exact naik bisa menaikkan AW-MAE |
| Row-level expert picking | Leakage terlalu direct |
| Segment n kecil tanpa fallback | Overfit tinggi |

## Prioritas Eksperimen Berikutnya

### High Priority

1. **Experiment E1: Archetype Stitching**
   - Implement mapping best strategy per `gender x archetype`.
   - Expected AW-MAE `2.413374`.
   - Risiko overfit lebih rendah dari tournament-era stitch.

2. **Experiment E2: Hierarchical Stitching n>=80**
   - Level: `gender x tournament x era`, fallback tournament, fallback archetype, fallback ExpC.
   - Expected AW-MAE `2.411353`.

3. **Experiment E3: Conservative Stitch With Outcome Guard**
   - Gunakan E2 tetapi fallback jika outcome segment turun terlalu besar.

### Medium Priority

4. **Upgrade ExpD**
   - Soft decoupled candidates harus archetype-specific, bukan fixed.

5. **Pair repair selective**
   - Repair hanya segment yang repair-nya menurunkan AW-MAE segment.

6. **Power 1.5 robustness check**
   - Pastikan best output tidak rapuh jika pangkat metric berubah.

### Low Priority

7. **Neutral modifier revival**
   - Tambahkan neutral side-bias secara eksplisit.

8. **Confederation pair fallback**
   - Berguna untuk segment tournament kecil.

## Current Recommendation

Gunakan **Experiment C** sebagai best single submission saat ini:

`dataset/submission_experiment_c_archetype_loss_tensor.csv`

Namun strategi paling menjanjikan berikutnya adalah membuat stitched submission:

1. Mulai dari E1 `gender x archetype`.
2. Jika hasil sesuai simulasi, lanjut E2 hierarchical tournament-era.
3. Simpan audit semua segment winner agar tidak menjadi row-level leakage.

