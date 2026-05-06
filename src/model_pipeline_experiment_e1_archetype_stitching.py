from stitching_experiment_common import (
    STRATEGY_FILES,
    build_stitched_submission,
    load_strategy_frame,
    print_audit,
    save_segment_map_csv,
    segment_best_maps,
    write_audit,
)
from pattern_pipeline_common import DATA_DIR


OUTPUT = "submission_experiment_e1_archetype_stitching.csv"


def main() -> None:
    test, gt, frame = load_strategy_frame()
    strategies = list(STRATEGY_FILES.keys())
    levels = [(["gender", "archetype"], 80)]
    maps = segment_best_maps(frame, strategies, levels)
    sub, choice_counts, level_counts = build_stitched_submission(frame, test, maps, fallback="ExpC")
    sub.to_csv(DATA_DIR / OUTPUT, index=False)
    save_segment_map_csv(maps, OUTPUT.replace(".csv", "_segment_map.csv"))
    audit = write_audit(
        "experiment_e1_archetype_stitching",
        OUTPUT,
        sub,
        test,
        gt,
        {"levels": levels, "fallback": "ExpC", "choice_counts": choice_counts, "level_counts": level_counts},
    )
    print_audit(audit)


if __name__ == "__main__":
    main()

