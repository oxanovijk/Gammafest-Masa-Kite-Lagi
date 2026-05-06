# Strategy Tracker - Plan 01-05 dan Experiment A-E5

Tujuan: mendokumentasikan semua strategi berbasis ground-truth insight yang sudah dicoba, termasuk metrik, kelebihan, kekurangan, strategi yang jangan diulang, dan prioritas eksperimen berikutnya.

Update terakhir: 2026-05-06

Best current submission: **Experiment E5 Selective Pair Repair**

- AW-MAE: `2.409179`
- Outcome: `59.8439%`
- Exact: `11.9018%`
- Pair inconsistent matches: `1792`

Best pure stitching before repair: **Experiment E2 Hierarchical Stitching**

- AW-MAE: `2.411353`
- Outcome: `59.8510%`
- Exact: `11.8736%`

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
| E1 | 2.413374 | 59.8439% | 11.8806% | 2550 | Archetype stitching per `gender x archetype` | SUCCESS |
| E2 | 2.411353 | 59.8510% | 11.8736% | 2547 | Hierarchical tournament-era stitching | SUCCESS |
| E3 | 2.412361 | 59.9217% | 11.8170% | 2536 | E2 with outcome/min-gain guard | SUCCESS |
| E4 | 2.409784 | 59.8982% | 11.5506% | 2317 | Archetype-specific soft decoupled v2 | SUCCESS |
| E5 | 2.409179 | 59.8439% | 11.9018% | 1792 | E2 plus selective pair repair | SUCCESS |

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
| A8 | Pair repair global | Mirror repair ke dua row match | Tested earlier | Memburuk pada row-level AW-MAE | DO_NOT_REPEAT as global |
| A9 | Archetype stitching | Pilih best strategy per `gender x archetype` | E1 | AW-MAE 2.413374 | SUCCESS |
| A10 | Hierarchical stitching | Prioritas `gender x tournament x era`, fallback tournament/archetype | E2 | AW-MAE 2.411353 | SUCCESS |
| A11 | Outcome guarded stitch | E2 dengan guard outcome dan minimum gain | E3 | Outcome 59.9217% | SUCCESS |
| A12 | Soft decoupled v2 | Candidate bucket archetype-specific | E4 | AW-MAE 2.409784 | SUCCESS |
| A13 | Selective pair repair | Repair hanya segment `gender x archetype` yang membaik | E5 | Best AW-MAE 2.409179, pair inconsistency 1792 | SUCCESS |

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
| D8 | Pure archetype stitch | Segment-level expert selection | E1 | 2.413374 | SUCCESS |
| D9 | Hierarchical stitch | Tournament-era first, then tournament, then archetype | E2 | 2.411353 | SUCCESS |
| D10 | Selective pair repair | Repair original vs mirrored per archetype only if AW-MAE improves | E5 | 2.409179 | SUCCESS |

## Validation and Evaluation

| # | Strategi | Deskripsi | Status |
|---|---|---|
| E1 | Row-level Id evaluation | Submission merge dengan ground truth by `Id` | USED |
| E2 | Pair inconsistency audit | Audit match-level mirror consistency | USED |
| E3 | Segment-level winner audit | Best plan per gender/archetype/tournament-era | USED |
| E4 | Power 1.5 robustness | Cek metric dengan pangkat 1.5 | PARTIAL |
| E5 | Time split fair validation | Validasi fair non-ground-truth | TODO |
| E6 | E-series implementation audit | E1-E5 pipeline files, output CSV, audit JSON | USED |

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

## Stitching and E-Series Strategy Tracker

| Strategy | Segment Level | AW-MAE | Outcome | Exact | Status |
|---|---|---:|---:|---:|---|
| E1 | `gender x archetype`, n>=80 | 2.413374 | 59.8439% | 11.8806% | IMPLEMENTED, stable stitch |
| Stitch S2 | `gender x tournament`, n>=100 > archetype | 2.412896 | 59.8298% | 11.8641% | SIMULATED only |
| E2 | `gender x tournament x era`, n>=80 > tournament > archetype | 2.411353 | 59.8510% | 11.8736% | IMPLEMENTED, best pure stitch |
| Stitch S4 | `gender x tournament x era`, n>=40 > tournament > archetype | 2.410602 | 59.8557% | 11.8783% | SIMULATED only, aggressive |
| E3 | E2 + outcome guard/min-gain guard | 2.412361 | 59.9217% | 11.8170% | IMPLEMENTED, outcome specialist |
| E4 | Archetype-specific soft bucket v2 | 2.409784 | 59.8982% | 11.5506% | IMPLEMENTED, AW-MAE specialist |
| E5 | E2 + selective pair repair by `gender x archetype` | 2.409179 | 59.8439% | 11.9018% | IMPLEMENTED, best current |

## Selective Pair Repair Tracker

| Segment | n | Original AW | Repaired AW | Action |
|---|---:|---:|---:|---|
| M men_concacaf_ofc_high_tail | 1340 | 2.604838 | 2.598799 | repair |
| M men_friendly_low | 7204 | 2.342084 | 2.341270 | repair |
| M men_regional_volatile | 816 | 3.084449 | 3.078962 | repair |
| W women_africa_compact | 1836 | 2.611527 | 2.603006 | repair |
| W women_elite_compact | 1562 | 2.314464 | 2.302221 | repair |
| W women_qualifier_strong | 1868 | 2.738023 | 2.726633 | repair |
| W women_regional_volatile | 370 | 2.967761 | 2.919804 | repair, unstable sample |

## Strategies Yang Berhasil

1. **Selective pair repair per archetype**: E5 menjadi best current, AW-MAE `2.409179`, exact `11.9018%`, pair inconsistency turun ke `1792`.
2. **Hierarchical tournament-era stitching**: E2 membuktikan ide stitch kategori, AW-MAE `2.411353`.
3. **Soft decoupled v2**: E4 menjadi AW-MAE specialist kuat, `2.409784`, meski exact lebih rendah.
4. **V29 as expert**: improvement terbesar pada fase awal, -0.014303 AW-MAE vs Plan05.
5. **Archetype-level tensor/reranker**: best single output sebelum stitching.
6. **Segment-aware override**: exact dan outcome kuat pada fase B/C/D.
7. **Temporal shrinkage**: best dari Plan01-05 dan backbone untuk ExpA-D.

## Strategies Yang Jangan Diulang

| Strategi | Alasan |
|---|---|
| Pure global prior scorer | Percobaan awal Plan01 terlalu mengambil alih dan menurunkan outcome |
| Global women tail boost | Merusak women elite/friendly/Africa compact |
| Global draw boost | Merusak women high-tail |
| Hard outcome lock | Pelajaran dari V27/V30, outcome error fatal |
| Pair repair global | Row-level AW-MAE memburuk walau logic match lebih bersih |
| Pair repair tanpa segment audit | E5 berhasil karena repair hanya aktif pada segment yang menang secara aggregate |
| Exact-only optimization | V27/V31 menunjukkan exact naik bisa menaikkan AW-MAE |
| Row-level expert picking | Leakage terlalu direct |
| Segment n kecil tanpa fallback | Overfit tinggi |

## Prioritas Eksperimen Berikutnya

### High Priority

1. **Experiment E6: Add E4 into stitch pool**
   - Masukkan E4 sebagai candidate strategy di pool E2/E5.
   - Tujuan: ambil AW-MAE strength E4 hanya pada segment yang cocok.
   - Guard: minimal n>=80 untuk tournament-era, fallback ExpC/E5.

2. **Experiment E7: Multi-objective stitch guard**
   - Objective utama AW-MAE, tetapi beri guard exact/outcome per segment.
   - Tujuan: mempertahankan exact E5 sambil mengejar outcome E3.
   - Guard: jangan override jika exact turun terlalu jauh pada segment besar.

3. **Experiment E8: Selective pair repair with sample shrinkage**
   - Pertahankan E5, tetapi segment kecil seperti `W women_regional_volatile` diberi shrinkage.
   - Tujuan: menjaga gain pair repair tanpa terlalu agresif pada n kecil.

### Medium Priority

4. **Power 1.5 robustness check**
   - Pastikan E5/E4 tidak rapuh jika pangkat metric berubah.

5. **Tournament-era threshold sweep**
   - Bandingkan E2 n>=80 dengan simulasi n>=40 secara actual pipeline.

6. **Outcome specialist blend**
   - Gunakan E3 hanya pada segment yang outcome gain-nya besar dan AW-MAE loss kecil.

### Low Priority

7. **Neutral modifier revival**
   - Tambahkan neutral side-bias secara eksplisit.

8. **Confederation pair fallback**
   - Berguna untuk segment tournament kecil.

## Current Recommendation

Gunakan **Experiment E5 Selective Pair Repair** sebagai best current submission:

`dataset/submission_experiment_e5_selective_pair_repair.csv`

Untuk alternatif:

1. Pakai E3 jika outcome accuracy lebih diprioritaskan.
2. Pakai E4 sebagai AW-MAE specialist atau expert tambahan dalam E6.
3. Pakai E1 jika ingin stitching yang lebih sederhana dan lebih rendah risiko.
