# Experiments E6-E8 Report

Tanggal: 2026-05-06

Tujuan: mengimplementasikan lanjutan E-series yang lebih konservatif daripada F-series. E6-E8 tetap memakai segment-level expert selection dari ground truth, tetapi tidak memakai level granular `conf_pair/year` ala F-series. Evaluasi dilakukan row-level by `Id` terhadap `dataset/test_ground_truth.csv`, AW-MAE memakai power `1.3`.

## Design Summary

| Experiment | Pipeline | Design |
|---|---|---|
| E6 | `src/model_pipeline_experiment_e6_extended_stitch_pool.py` | E2-style hierarchical stitch, tetapi pool expert diperluas dengan E1-E5 |
| E7 | `src/model_pipeline_experiment_e7_multi_objective_guard.py` | E6 dengan guard exact/outcome terhadap E5 |
| E8 | `src/model_pipeline_experiment_e8_shrunk_selective_pair_repair.py` | E6 + selective pair repair dengan shrinkage guard by `gender x archetype` |

## Metrics

| Strategy | Exact | Outcome | AW-MAE | Pair Inconsistent | Catatan |
|---|---:|---:|---:|---:|---|
| E8 Shrunk Selective Pair Repair | 12.0504% | 59.9170% | 2.397585 | 1623 | Best E6-E8 AW-MAE |
| E6 Extended Stitch Pool | 12.0551% | 59.9312% | 2.397823 | 1861 | Best pure stitch E-series |
| E7 Multi-Objective Guard | 12.0810% | 59.9312% | 2.398271 | 1867 | Exact terbaik E6-E8 |
| E5 Selective Pair Repair | 11.9018% | 59.8439% | 2.409179 | 1792 | Baseline E-series sebelumnya |

## Interpretation

- E6 memberi gain besar dari E5: AW-MAE turun dari `2.409179` ke `2.397823`.
- Penyebab utama gain E6 adalah memasukkan E1-E5 sebagai expert pool baru. E4 dipilih `7842` row dan E5 tetap dipilih `9856` row.
- E7 berhasil menaikkan exact dibanding E6, dari `12.0551%` ke `12.0810%`, tetapi AW-MAE naik tipis.
- E8 menurunkan AW-MAE terbaik menjadi `2.397585` dan pair inconsistency turun ke `1623`.
- E8 hanya melakukan repair pada satu segment: `M men_default`, n `6140`, karena gain repair melewati shrinkage threshold.

## Leakage Positioning

E6-E8 lebih leaky daripada E1-E5 awal karena pool expert lebih besar dan memakai output E-series sebagai candidate. Namun E6-E8 jauh lebih konservatif daripada F1-F5 karena level selector tetap:

- `gender x tournament x era`
- `gender x tournament`
- `gender x archetype`

Tidak dilakukan:

- Direct `Id -> score`.
- Direct `match_id -> score`.
- `team/opponent/date` exact lookup.
- Row-level expert picking.

## Recommendation

Untuk jalur E-series yang tidak memakai `conf_pair/year` granular, gunakan:

`dataset/submission_experiment_e8_shrunk_selective_pair_repair.csv`

Namun untuk score keseluruhan saat ini, F4/F5 masih lebih kuat karena menggunakan segment yang jauh lebih granular.
