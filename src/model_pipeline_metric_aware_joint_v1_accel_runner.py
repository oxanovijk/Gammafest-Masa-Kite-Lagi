"""Accelerated runner for model_pipeline_metric_aware_joint_v1.

This keeps the same model/design semantics as the original script but replaces
the row prediction loop with a vectorized prediction pass. Output paths and cache
directory are suffixed with `_accel` so the long original run can continue safely.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import model_pipeline_metric_aware_joint_v1 as m


def predict_frame_vectorized(
    df: pd.DataFrame,
    model: m.ProbabilisticHeads,
    prior: np.ndarray,
    config: m.CandidateConfig,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    ordered = df.sort_values(["date", "match_id", "Id"]).reset_index(drop=True)
    ordered["_pos"] = np.arange(len(ordered))
    heads = model.predict(ordered)
    pred_t = np.zeros(len(ordered), dtype=int)
    pred_o = np.zeros(len(ordered), dtype=int)
    row_mats: dict[int, np.ndarray] = {}
    row_scores: dict[int, tuple[int, int]] = {}
    pair_diag: defaultdict[str, int] = defaultdict(int)
    matrix_diag = {"avg_selected_joint_prob": 0.0, "avg_entropy": 0.0, "rows": 0}

    for idx in range(len(ordered)):
        mat = m.build_joint_matrix(heads, idx, prior, config)
        row_mats[idx] = mat
        p, q, info = m.choose_expected_awmae_score(mat, heads["outcome"][idx], config.max_goals)
        row_scores[idx] = (p, q)
        matrix_diag["avg_selected_joint_prob"] += info["selected_joint_prob"]
        matrix_diag["avg_entropy"] += float(-(mat * np.log(np.maximum(mat, 1e-12))).sum())
        matrix_diag["rows"] += 1

    for _, group in ordered.groupby("match_id", sort=False):
        pair_diag["match_groups"] += 1
        idxs = [int(i) for i in group.index.to_list()]
        if len(group) == 1:
            pair_diag["single_row_matches"] += 1
        elif len(group) == 2:
            pair_diag["two_row_matches"] += 1
        else:
            pair_diag["multirow_matches"] += 1

        scores = [row_scores[i] for i in idxs]
        if len(group) == 2 and m.reciprocal_pair(group):
            pair_diag["reciprocal_pairs"] += 1
            if config.pair_consistency:
                pair_diag["pair_consistency_applied"] += 1
                loss_a = m.expected_loss_matrix(row_mats[idxs[0]], config.max_goals, m.PRIMARY_POWER)
                loss_b = m.expected_loss_matrix(row_mats[idxs[1]], config.max_goals, m.PRIMARY_POWER)
                combined = loss_a + loss_b.T
                a, b = np.unravel_index(int(np.argmin(combined)), combined.shape)
                target_scores = [(int(a), int(b)), (int(b), int(a))]
                if scores != target_scores:
                    pair_diag["pair_conflicts_corrected"] += 1
                    pair_diag["pair_correction_abs"] += abs(scores[0][0] - int(a)) + abs(scores[0][1] - int(b))
                scores = target_scores
            else:
                pair_diag["pair_consistency_disabled"] += 1
        elif len(group) == 2:
            pair_diag["inconsistent_pairs"] += 1

        for idx, score in zip(idxs, scores):
            pred_t[idx], pred_o[idx] = score

    out = ordered.copy()
    out["pred_team_goals"] = pred_t
    out["pred_opp_goals"] = pred_o
    rows = max(1, matrix_diag["rows"])
    matrix_diag["avg_selected_joint_prob"] /= rows
    matrix_diag["avg_entropy"] /= rows
    pair_diag["pair_consistency_pass"] = int(pair_diag.get("inconsistent_pairs", 0) == 0)
    pair_diag["avg_pair_correction_abs"] = pair_diag.get("pair_correction_abs", 0) / max(1, pair_diag.get("pair_conflicts_corrected", 0))
    return out, dict(pair_diag), matrix_diag


def configure_accel_paths() -> None:
    m.OUTPUT_SUB = m.DATA_DIR / "submission_metric_aware_joint_v1_accel.csv"
    m.OUTPUT_CONFIG = m.DATA_DIR / "submission_metric_aware_joint_v1_accel_config.json"
    m.OUTPUT_REPORT = m.DATA_DIR / "submission_metric_aware_joint_v1_accel_validation_report.txt"
    m.OUTPUT_AUDIT = m.DATA_DIR / "submission_metric_aware_joint_v1_accel_audit.txt"
    m.OUTPUT_LOCK = m.DATA_DIR / "submission_metric_aware_joint_v1_accel_candidate_lock.json"
    old_cache = m.CACHE_DIR
    m.CACHE_DIR = m.DATA_DIR / "metric_aware_joint_v1_accel_cache"
    m.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if old_cache.exists():
        for src in old_cache.glob("*.pkl"):
            dst = m.CACHE_DIR / src.name
            if not dst.exists():
                shutil.copy2(src, dst)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-mode", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--skip-final", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_accel_paths()
    m.predict_frame = predict_frame_vectorized
    fast_mode = not args.full_mode if m.DEFAULT_FAST_MODE else False
    use_cache = not args.no_cache
    np.random.seed(m.SEED)

    script_hash = m.file_sha256(Path(m.__file__).resolve())
    audit = m.build_audit(fast_mode)
    if not audit["feasible"]:
        m.write_audit_file(audit)
        raise RuntimeError("Internal audit failed.")

    print(f"[metric_joint_accel] loading data from {m.DATA_DIR}")
    train, test, sample = m.load_data(read_test=True)
    assert test is not None and sample is not None
    meta_diag = m.metadata_diagnostics(train, test)
    selected, baseline, results, skipped = m.select_candidate(train, fast_mode, script_hash, use_cache)
    config_hash = m.json_hash(selected.config)
    lock_hash = m.write_candidate_lock(selected, baseline, script_hash, config_hash, audit)
    print(f"[metric_joint_accel] candidate lock written hash={lock_hash}")

    if args.skip_final:
        final_diag = {"skipped_final": True}
    else:
        final_config = m.CandidateConfig(**selected.config)
        submission, final_diag = m.fit_final_predict(train, test, sample, final_config, fast_mode)
        submission.to_csv(m.OUTPUT_SUB, index=False)

    local15 = m.local_submission_metrics(m.OUTPUT_SUB, test, m.PRIMARY_POWER) if m.OUTPUT_SUB.exists() and m.GT_PATH.exists() else None
    local13 = m.local_submission_metrics(m.OUTPUT_SUB, test, m.SECONDARY_POWER) if m.OUTPUT_SUB.exists() and m.GT_PATH.exists() else None
    friend = m.find_friend_csv()
    friend_report = None
    if friend is not None and m.GT_PATH.exists():
        friend_report = {"path": str(friend), "p15": m.local_submission_metrics(friend, test, m.PRIMARY_POWER), "p13": m.local_submission_metrics(friend, test, m.SECONDARY_POWER)}

    decision = m.final_decision(selected, baseline, audit)
    config_payload = {
        "pipeline_version": m.PIPELINE_VERSION,
        "runner": "metric_aware_joint_v1_accel_runner",
        "mode": "FAST_MODE" if fast_mode else "FULL_MODE",
        "seed": m.SEED,
        "script_hash": script_hash,
        "config_hash": config_hash,
        "candidate_lock_hash": lock_hash,
        "selected": asdict(selected),
        "baseline": asdict(baseline),
        "results": [asdict(r) for r in results],
        "skipped": skipped,
        "metadata_diagnostics": meta_diag,
        "final_diagnostics": final_diag,
        "audit": audit,
        "post_lock_gt_p15": local15,
        "post_lock_gt_p13": local13,
        "post_lock_friend_report": friend_report,
        "decision": decision,
    }
    m.OUTPUT_CONFIG.write_text(json.dumps(m.to_jsonable(config_payload), indent=2), encoding="utf-8")
    m.write_audit_file(audit)
    m.OUTPUT_REPORT.write_text(m.report_text(selected, baseline, results, skipped, lock_hash, final_diag, meta_diag, audit, local15, local13, friend_report, decision), encoding="utf-8")
    print(f"[metric_joint_accel] selected={selected.config['name']} decision={decision}")
    print(f"[metric_joint_accel] validation_w15={selected.metrics['weighted_awmae_p15']:.6f} validation_w13={selected.metrics['weighted_awmae_p13']:.6f} outcome={selected.metrics['outcome_accuracy']:.6f}")
    if local15:
        print(f"[metric_joint_accel] post_lock_gt_w15={local15['weighted_awmae']:.6f} outcome={local15['outcome_accuracy']:.6f}")
    if local13:
        print(f"[metric_joint_accel] post_lock_gt_w13={local13['weighted_awmae']:.6f}")


if __name__ == "__main__":
    main()
