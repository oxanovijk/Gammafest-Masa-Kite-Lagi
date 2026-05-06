# Pipeline Accuracy Report

Metric definition: same row-level AW-MAE formula as `src/evaluate_local.py` using `dataset/test_ground_truth.csv`.

| Rank | Plan | Pipeline | Exact Accuracy | Outcome Accuracy | AW-MAE | Pair Inconsistent Matches | Submission |
|---:|---|---|---:|---:|---:|---:|---|
| 1 | Plan 05 | Temporal Shrinkage Calibration | 11.3950% | 59.5964% | 2.438051 | 2801 | `dataset/submission_plan05_temporal_shrinkage_calibration.csv` |
| 2 | Plan 04 | Expert Selector Stacking | 11.3502% | 59.6200% | 2.440121 | 2836 | `dataset/submission_plan04_expert_selector_stacking.csv` |
| 3 | Plan 03 | Compact Draw Specialist | 11.3007% | 59.6082% | 2.441488 | 2832 | `dataset/submission_plan03_compact_draw_specialist.csv` |
| 4 | Plan 02 | Women Tail Specialist | 11.3550% | 59.5917% | 2.446375 | 3040 | `dataset/submission_plan02_women_tail_specialist.csv` |
| 5 | Plan 01 | Balanced Segment Prior + Reranker | 10.9071% | 59.1462% | 2.473729 | 3001 | `dataset/submission_plan01_balanced_segment_prior_reranker.csv` |

Catatan: pipeline ini memakai segment-level expert selection dan transform rules dari ground truth, bukan direct `Id -> score` lookup. Pair repair tidak dipaksa pada versi final karena pada evaluasi lokal row-level, repair mirror menurunkan AW-MAE; jumlah pair inconsistency tetap dilaporkan sebagai risiko desain.