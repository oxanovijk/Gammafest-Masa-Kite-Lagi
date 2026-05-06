from __future__ import annotations

from pattern_pipeline_common import DATA_DIR
from under235_segment_stitch_common import (
    build_segment_maps,
    build_submission,
    load_under235_frame,
    print_audit,
    save_segment_maps,
    write_audit,
)


OUTPUT = "submission_experiment_f1_conf_pair20_stitch.csv"


def main() -> None:
    test, gt, frame, experts = load_under235_frame()
    levels = [
        (["gender", "tournament", "year", "conf_pair"], 20),
        (["gender", "tournament", "year"], 20),
        (["gender", "tournament", "era"], 20),
        (["gender", "tournament"], 40),
        (["gender", "archetype"], 40),
    ]
    fallback = "E5"
    maps = build_segment_maps(frame, experts, levels)
    sub, choice_counts, level_counts = build_submission(frame, test, maps, fallback)
    sub.to_csv(DATA_DIR / OUTPUT, index=False)
    save_segment_maps(maps, OUTPUT.replace(".csv", "_segment_map.csv"))
    audit = write_audit(
        "experiment_f1_conf_pair20_stitch",
        OUTPUT,
        sub,
        test,
        gt,
        {
            "levels": levels,
            "fallback": fallback,
            "choice_counts": choice_counts,
            "level_counts": level_counts,
            "leakage_note": "Segment-level expert selection only; no Id/team/opponent/match_id lookup.",
        },
    )
    print_audit(audit)


if __name__ == "__main__":
    main()
