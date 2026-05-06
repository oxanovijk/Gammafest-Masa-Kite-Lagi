"""Diagnostic-only final inference for ablation_draw_correction_off.

This script is intentionally not part of the leakage-safe candidate-selection
flow. It runs after the official batch candidate lock and writes suffixed output
files for post-lock diagnosis only.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

import model_pipeline_metric_aware_joint_v1 as m
import model_pipeline_metric_aware_joint_v1_batch_runner as batch


def main() -> None:
    np.random.seed(m.SEED)
    m.predict_frame = batch.predict_frame_vectorized

    out_sub = m.DATA_DIR / "submission_metric_aware_joint_v1_diagnostic_draw_off.csv"
    out_config = m.DATA_DIR / "submission_metric_aware_joint_v1_diagnostic_draw_off_config.json"
    out_report = m.DATA_DIR / "submission_metric_aware_joint_v1_diagnostic_draw_off_report.txt"

    config = m.CandidateConfig(
        name="ablation_draw_correction_off_diagnostic_only",
        kind="joint_awmae",
        smoothing=50.0,
        max_goals=8,
        draw_correction=1.0,
        overcomplexity=0.001,
    )

    print("[draw_off_diag] loading data")
    train, test, sample = m.load_data(read_test=True)
    assert test is not None and sample is not None

    print("[draw_off_diag] fitting final model and predicting test")
    submission, final_diag = m.fit_final_predict(train, test, sample, config, fast_mode=True)
    submission.to_csv(out_sub, index=False)

    print("[draw_off_diag] computing post-lock diagnostics")
    local15 = m.local_submission_metrics(out_sub, test, m.PRIMARY_POWER) if m.GT_PATH.exists() else None
    local13 = m.local_submission_metrics(out_sub, test, m.SECONDARY_POWER) if m.GT_PATH.exists() else None
    friend = m.find_friend_csv()
    friend_report = None
    if friend is not None and m.GT_PATH.exists():
        friend_report = {
            "path": str(friend),
            "p15": m.local_submission_metrics(friend, test, m.PRIMARY_POWER),
            "p13": m.local_submission_metrics(friend, test, m.SECONDARY_POWER),
        }

    payload = {
        "diagnostic_only": True,
        "leakage_note": "Generated after official candidate lock and after post-lock GT reporting; not valid for selection or accepted-candidate replacement.",
        "config": asdict(config),
        "final_diagnostics": final_diag,
        "post_lock_gt_p15": local15,
        "post_lock_gt_p13": local13,
        "post_lock_friend_report": friend_report,
    }
    out_config.write_text(json.dumps(m.to_jsonable(payload), indent=2), encoding="utf-8")

    lines = [
        "Diagnostic draw-correction-off final inference",
        "=" * 48,
        "DIAGNOSTIC ONLY: generated after official lock/GT reporting; do not use as leakage-safe selection.",
        "",
        f"CSV: {out_sub}",
        f"Config: {json.dumps(asdict(config), indent=2)}",
        "",
        "Final diagnostics",
        json.dumps(m.to_jsonable(final_diag), indent=2)[:6000],
        "",
        "Post-lock GT p1.5",
        json.dumps(m.to_jsonable(local15), indent=2),
        "",
        "Post-lock GT p1.3",
        json.dumps(m.to_jsonable(local13), indent=2),
        "",
        "Friend report",
        json.dumps(m.to_jsonable(friend_report), indent=2)[:8000],
    ]
    out_report.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[draw_off_diag] wrote {out_sub}")
    if local15:
        print(f"[draw_off_diag] post_lock_gt_w15={local15['weighted_awmae']:.6f} outcome={local15['outcome_accuracy']:.6f}")
    if local13:
        print(f"[draw_off_diag] post_lock_gt_w13={local13['weighted_awmae']:.6f}")
    print(f"[draw_off_diag] wrote {out_report}")


if __name__ == "__main__":
    main()
