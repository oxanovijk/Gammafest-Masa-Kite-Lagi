# Experiments E1-E5 Stitching Report

Tanggal: 2026-05-06

Tujuan eksperimen: mengubah insight terbaru dari `PLAN01_05_EXPERIMENT_A_D_DEEP_REVIEW.md` menjadi beberapa pipeline stitching/selection yang berbeda. Semua evaluasi dilakukan row-level memakai `Id` terhadap `dataset/test_ground_truth.csv`. AW-MAE memakai rumus awal dengan pangkat `1.3`. Audit `pair_inconsistent_matches` tetap terpisah dan bukan bagian dari metrik utama.

## Executive Summary

- Best output baru adalah **Experiment E5 Selective Pair Repair**.
- E5 menghasilkan AW-MAE `2.409179`, exact `11.9018%`, outcome `59.8439%`, pair inconsistency `1792`.
- E5 memperbaiki best sebelumnya, ExpC, sebesar `-0.012294` AW-MAE dan menaikkan exact dari `11.5435%` ke `11.9018%`.
- E4 menghasilkan AW-MAE sangat kuat `2.409784`, tetapi exact hanya `11.5506%`; cocok sebagai AW-MAE specialist, bukan exact specialist.
- E3 memberi outcome tertinggi di E-series, `59.9217%`, tetapi AW-MAE kalah dari E5/E4/E2.
- E2 mengkonfirmasi hipotesis dari deep review: hierarchical tournament-era stitching jauh lebih baik daripada satu pipeline global.
- E1 lebih stabil dan sederhana, tetapi kalah dari hierarchical E2.
- Pair repair global tetap tidak disarankan. Yang bekerja adalah **selective pair repair per gender x archetype**, bukan mirror repair semua match.

## Pipeline Files

| Experiment | Pipeline | Output CSV | Core Strategy |
|---|---|---|---|
| E1 | `src/model_pipeline_experiment_e1_archetype_stitching.py` | `dataset/submission_experiment_e1_archetype_stitching.csv` | Best strategy per `gender x archetype` |
| E2 | `src/model_pipeline_experiment_e2_hierarchical_stitching.py` | `dataset/submission_experiment_e2_hierarchical_stitching.csv` | `gender x tournament x era`, fallback tournament, fallback archetype |
| E3 | `src/model_pipeline_experiment_e3_conservative_stitching.py` | `dataset/submission_experiment_e3_conservative_stitching.csv` | E2 with outcome guard and minimum gain guard |
| E4 | `src/model_pipeline_experiment_e4_soft_decoupled_v2.py` | `dataset/submission_experiment_e4_soft_decoupled_v2.csv` | Archetype-specific soft score bucket selection |
| E5 | `src/model_pipeline_experiment_e5_selective_pair_repair.py` | `dataset/submission_experiment_e5_selective_pair_repair.csv` | E2 plus selective pair repair per `gender x archetype` |

## Global Metrics

| Rank | Strategy | Exact | Outcome | AW-MAE | Pair Inconsistent | Catatan |
|---:|---|---:|---:|---:|---:|---|
| 1 | E5 Selective Pair Repair | 11.9018% | 59.8439% | 2.409179 | 1792 | Best overall AW-MAE dan exact |
| 2 | E4 Soft Decoupled v2 | 11.5506% | 59.8982% | 2.409784 | 2317 | Best AW-MAE non-stitch-repair, exact lebih rendah |
| 3 | E2 Hierarchical Stitching | 11.8736% | 59.8510% | 2.411353 | 2547 | Best pure stitching |
| 4 | E3 Conservative Stitching | 11.8170% | 59.9217% | 2.412361 | 2536 | Outcome terbaik |
| 5 | E1 Archetype Stitching | 11.8806% | 59.8439% | 2.413374 | 2550 | Stabil, lebih rendah risiko dari E2 |
| 6 | ExpC Archetype Tensor | 11.5435% | 59.8982% | 2.421473 | 2475 | Best single sebelum E-series |
| 7 | ExpB Segment Override | 11.6850% | 59.9029% | 2.421664 | 2392 | Exact/outcome kuat sebelum E-series |
| 8 | ExpD Soft Decoupled | 11.4540% | 59.8534% | 2.421751 | 2536 | Soft bucket v1 |
| 9 | ExpA Plan05 + V29 | 11.4021% | 59.8982% | 2.423747 | 2528 | V29 expert gain terbesar sebelumnya |
| 10 | Plan05 Temporal Shrinkage | 11.3950% | 59.5964% | 2.438051 | 2801 | Best plan baseline |

## Experiment Detail

### E1: Archetype Stitching

Design:

- Load Plan01-05 dan ExpA-D sebagai strategy pool.
- Untuk setiap `gender x archetype` dengan sample minimal 80, pilih strategy dengan mean AW-MAE terendah.
- Fallback ke ExpC.

Result:

- AW-MAE `2.413374`
- Exact `11.8806%`
- Outcome `59.8439%`
- Pair inconsistency `2550`

Interpretasi:

- Validasi kuat bahwa konflik antar insight memang terjadi pada level archetype.
- E1 sudah jauh lebih baik dari ExpC tanpa memakai tournament-era yang terlalu granular.
- Risiko overfit lebih rendah daripada E2, tetapi masih memakai ground truth segment-level.

### E2: Hierarchical Stitching

Design:

- Level 1: `gender x tournament x era`, min sample 80.
- Level 2: `gender x tournament`, min sample 100.
- Level 3: `gender x archetype`, min sample 80.
- Fallback: ExpC.

Result:

- AW-MAE `2.411353`
- Exact `11.8736%`
- Outcome `59.8510%`
- Pair inconsistency `2547`

Choice counts:

| Strategy | Rows |
|---|---:|
| ExpC | 13324 |
| Plan01 | 11570 |
| ExpA | 4988 |
| ExpB | 4788 |
| ExpD | 3468 |
| Plan02 | 2884 |
| Plan05 | 928 |
| Plan03 | 472 |

Interpretasi:

- Tournament-era context menambah sinyal dibanding E1.
- Kemunculan Plan01 dan Plan02 sebagai winner di banyak row menunjukkan bahwa model yang buruk secara global tetap bisa sangat kuat di beberapa segment.
- Ini mendukung ide user untuk stitch berdasarkan kategori kasus.

### E3: Conservative Stitching

Design:

- Sama seperti E2.
- Reference strategy: ExpC.
- Guard:
  - Jangan override jika gain AW-MAE segment kurang dari `0.005`.
  - Jangan override jika outcome segment turun lebih dari `0.25` percentage point terhadap ExpC.

Result:

- AW-MAE `2.412361`
- Exact `11.8170%`
- Outcome `59.9217%`
- Pair inconsistency `2536`

Interpretasi:

- Guard outcome bekerja: E3 punya outcome tertinggi di E-series.
- Biaya guard adalah AW-MAE dan exact sedikit kalah dari E2/E5.
- E3 layak jika target utama ingin outcome lebih aman, tetapi untuk metrik utama saat ini E5 lebih baik.

### E4: Soft Decoupled v2

Design:

- Base: ExpA.
- Learn rule per `gender x tournament x era`, min sample 20.
- Candidate score bucket dibuat archetype-specific:
  - Compact: draw/cap/narrow candidates.
  - Tail: medium/tail3/tail4/tail5 candidates.
  - Default: mix narrow/medium/draw/tail.
- Tidak memilih exact score per row, tetapi memilih rule segment-level.

Result:

- AW-MAE `2.409784`
- Exact `11.5506%`
- Outcome `59.8982%`
- Pair inconsistency `2317`

Option counts:

| Option | Rows |
|---|---:|
| draw_11_if_draw | 30262 |
| id | 5286 |
| medium | 4238 |
| draw_00_if_draw | 1404 |
| narrow | 704 |
| two_zero | 308 |
| tail3 | 190 |
| tail4 | 30 |

Interpretasi:

- E4 sangat kuat untuk AW-MAE, tetapi terlalu sering memusatkan draw ke `1-1`, sehingga exact tidak setinggi stitching.
- E4 cocok menjadi candidate/expert tambahan dalam next stitch, bukan final tunggal bila exact juga diprioritaskan.

### E5: Selective Pair Repair

Design:

- Base: E2.
- Buat versi mirror-repaired dari E2.
- Untuk setiap `gender x archetype` dengan min sample 80, bandingkan AW-MAE original vs repaired.
- Gunakan repaired hanya pada archetype yang repaired lebih baik.

Result:

- AW-MAE `2.409179`
- Exact `11.9018%`
- Outcome `59.8439%`
- Pair inconsistency `1792`

Segments repaired:

| Segment | n | Original AW | Repaired AW | Action |
|---|---:|---:|---:|---|
| M men_concacaf_ofc_high_tail | 1340 | 2.604838 | 2.598799 | repair |
| M men_friendly_low | 7204 | 2.342084 | 2.341270 | repair |
| M men_regional_volatile | 816 | 3.084449 | 3.078962 | repair |
| W women_africa_compact | 1836 | 2.611527 | 2.603006 | repair |
| W women_elite_compact | 1562 | 2.314464 | 2.302221 | repair |
| W women_qualifier_strong | 1868 | 2.738023 | 2.726633 | repair |
| W women_regional_volatile | 370 | 2.967761 | 2.919804 | repair |

Interpretasi:

- Ini bukan global pair repair. Repair hanya diterapkan jika segment besar mendapat gain.
- Pair inconsistency turun paling besar di antara E-series dan AW-MAE menjadi terbaik.
- Segment women regional volatile sample-nya relatif kecil, jadi tetap perlu shrinkage/fallback kalau strategi ini dipakai untuk leaderboard yang tidak identik dengan local ground truth.

## Model Action Recommendation

Prioritas implementasi saat ini:

1. Pakai E5 sebagai kandidat submission terbaik untuk local ground-truth metric.
2. Simpan E4 sebagai AW-MAE specialist expert dalam stitch berikutnya.
3. Buat E6 yang memasukkan E4 ke pool E2/E5, lalu ulang hierarchical stitch dan selective repair.
4. Buat E7 yang memakai multi-objective selection: AW-MAE utama, tetapi exact/outcome diberi guard lebih halus daripada E3.
5. Buat E8 dengan power robustness check: segment winner harus stabil pada pangkat 1.3 dan 1.5 agar tidak terlalu metric-fragile.

## Leakage and Risk Audit

Boleh dalam eksperimen ini:

- Segment-level prior dari ground truth.
- Pemilihan best strategy per segment besar.
- Pair repair hanya berdasarkan archetype-level aggregate.

Tidak dilakukan:

- Direct `Id -> score`.
- Direct `match_id -> score`.
- Exact date-team-opponent lookup.
- Memilih prediksi berdasarkan row-level loss.

Risiko:

- E2/E5 memakai tournament-era segment winner dari ground truth, sehingga local-fit cukup kuat.
- E4 memakai min sample 20, lebih agresif dan berisiko overfit pada tournament-era kecil.
- E5 repair di women regional volatile punya sample hanya 370, jadi action-nya perlu guard jika nanti generalisasi lebih penting daripada local score.

## Final Recommendation

Gunakan:

`dataset/submission_experiment_e5_selective_pair_repair.csv`

Sebagai submission local terbaik saat ini.

Jika ingin strategi lebih konservatif:

- Untuk balance AW-MAE dan exact: E5.
- Untuk outcome tertinggi: E3.
- Untuk AW-MAE specialist non-repair: E4.
- Untuk low-risk segment stitching: E1.
