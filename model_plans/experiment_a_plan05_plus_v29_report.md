# Experiment A Report: Plan 05 + V29 Expert

Formula AW-MAE: rumus awal `src/evaluate_local.py`, power/pangkat 1.3.

| Model | Exact | Outcome | AW-MAE | Delta AW vs Plan05 | Delta Outcome | Delta Exact |
|---|---:|---:|---:|---:|---:|---:|
| Experiment A Plan05+V29 | 11.4021% | 59.8982% | 2.423747 | -0.014303 | +0.3017% | +0.0071% |
| Plan 05 baseline | 11.3950% | 59.5964% | 2.438051 | +0.000000 | +0.0000% | +0.0000% |
| V29 standalone | 10.6101% | 58.7148% | 2.525609 | +0.087559 | -0.8816% | -0.7850% |

Submission output: `dataset/submission_experiment_a_plan05_plus_v29.csv`
Audit output: `dataset/submission_experiment_a_plan05_plus_v29_audit.json`

## Selected Expert Counts

- `v29`: 16328
- `risk_v2`: 5108
- `v5`: 4876
- `dynamic_state_v1`: 3236
- `metric_aware_joint_v1_batch`: 3196
- `risk_v3`: 2760
- `v3`: 2108
- `temporal_robust_joint_v1`: 1870
- `v4`: 1332
- `risk_v5_outcome_experts`: 1056
- `risk_v4_static_drift`: 552