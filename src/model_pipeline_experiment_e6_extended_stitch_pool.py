from __future__ import annotations

from pattern_pipeline_common import DATA_DIR
from e_series_extended_common import (
    DEFAULT_LEVELS,
    build_aw_maps,
    build_submission_from_maps,
    load_extended_frame,
    print_audit,
    save_segment_map,
    write_audit,
)


OUTPUT = "submission_experiment_e6_extended_stitch_pool.csv"


def main() -> None:
    test, gt, frame, strategies = load_extended_frame()
    fallback = "E5"
    maps = build_aw_maps(frame, strategies, DEFAULT_LEVELS)
    submission, choice_counts, level_counts = build_submission_from_maps(frame, test, maps, fallback)
    submission.to_csv(DATA_DIR / OUTPUT, index=False)
    save_segment_map(maps, OUTPUT.replace(".csv", "_segment_map.csv"))
    audit = write_audit(
        "experiment_e6_extended_stitch_pool",
        OUTPUT,
        submission,
        test,
        gt,
        {
            "levels": DEFAULT_LEVELS,
            "fallback": fallback,
            "strategies": strategies,
            "choice_counts": choice_counts,
            "level_counts": level_counts,
            "design": "E2-style hierarchical stitch, but pool includes E1-E5 as additional experts.",
            "leakage_note": "Segment-level expert selection only; no Id/team/opponent/match_id lookup.",
        },
    )
    print_audit(audit)


if __name__ == "__main__":
    main()
