from __future__ import annotations

from pattern_pipeline_common import DATA_DIR
from e_series_extended_common import (
    DEFAULT_LEVELS,
    build_guarded_maps,
    build_submission_from_maps,
    load_extended_frame,
    print_audit,
    save_segment_map,
    write_audit,
)


OUTPUT = "submission_experiment_e7_multi_objective_guard.csv"
REFERENCE = "E5"
MIN_GAIN = 0.0
OUTCOME_GUARD_PP = 1.0
EXACT_GUARD_PP = 1.0


def main() -> None:
    test, gt, frame, strategies = load_extended_frame()
    maps = build_guarded_maps(
        frame,
        strategies,
        reference=REFERENCE,
        min_gain=MIN_GAIN,
        outcome_guard_pp=OUTCOME_GUARD_PP,
        exact_guard_pp=EXACT_GUARD_PP,
        levels=DEFAULT_LEVELS,
    )
    submission, choice_counts, level_counts = build_submission_from_maps(frame, test, maps, REFERENCE)
    submission.to_csv(DATA_DIR / OUTPUT, index=False)
    save_segment_map(maps, OUTPUT.replace(".csv", "_segment_map.csv"))
    audit = write_audit(
        "experiment_e7_multi_objective_guard",
        OUTPUT,
        submission,
        test,
        gt,
        {
            "levels": DEFAULT_LEVELS,
            "fallback": REFERENCE,
            "reference": REFERENCE,
            "min_gain": MIN_GAIN,
            "outcome_guard_pp": OUTCOME_GUARD_PP,
            "exact_guard_pp": EXACT_GUARD_PP,
            "strategies": strategies,
            "choice_counts": choice_counts,
            "level_counts": level_counts,
            "design": "E6 stitch with AW-MAE primary objective, but segment candidates must stay within exact/outcome guard versus E5.",
            "leakage_note": "Segment-level expert selection only; no Id/team/opponent/match_id lookup.",
        },
    )
    print_audit(audit)


if __name__ == "__main__":
    main()
