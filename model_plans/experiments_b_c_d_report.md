# Experiments B, C, D Report

AW-MAE memakai rumus awal dengan pangkat 1.3.

| Rank | Model | Exact | Outcome | AW-MAE | Delta AW vs Exp A | Delta AW vs Plan05 | Pair Inconsistent | Submission |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 1 | Experiment C Archetype Tensor | 11.5435% | 59.8982% | 2.421473 | -0.002274 | -0.016577 | 2475 | `dataset/submission_experiment_c_archetype_loss_tensor.csv` |
| 2 | Experiment B Segment Override | 11.6850% | 59.9029% | 2.421664 | -0.002083 | -0.016387 | 2392 | `dataset/submission_experiment_b_segment_aware_override.csv` |
| 3 | Experiment D Soft Decoupled | 11.4540% | 59.8534% | 2.421751 | -0.001996 | -0.016300 | 2536 | `dataset/submission_experiment_d_soft_decoupled_candidates.csv` |
| 4 | Experiment A Plan05+V29 | 11.4021% | 59.8982% | 2.423747 | +0.000000 | -0.014303 | 2528 | `dataset/submission_experiment_a_plan05_plus_v29.csv` |
| 5 | Plan05 baseline | 11.3950% | 59.5964% | 2.438051 | +0.014303 | +0.000000 | - | `dataset/submission_plan05_temporal_shrinkage_calibration.csv` |

## Interpretasi

- Eksperimen C adalah yang terbaik secara AW-MAE: archetype-level loss tensor/reranker turun sedikit dari B dan D.
- Eksperimen B memberi exact tertinggi, tetapi AW-MAE sedikit kalah dari C.
- Eksperimen D membuktikan soft decoupled candidate jauh lebih aman daripada hard V30, tetapi fixed bucket candidate belum mengalahkan C.
- Semua B/C/D mengalahkan Experiment A dan Plan05 baseline pada local ground-truth score.