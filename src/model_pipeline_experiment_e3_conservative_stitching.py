from __future__ import annotations

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


OUTPUT = "submission_experiment_e3_conservative_stitching.csv"
OUTCOME_GUARD = 0.25
MIN_GAIN = 0.005
REFERENCE = "ExpC"


def apply_conservative_guard(maps):
    guarded = []
    for level in maps:
        new_map = {}
        new_stats = {}
        for key, stats in level["stats"].items():
            best = stats["best"]
            ref_loss = stats["losses"][REFERENCE]
            best_loss = stats["losses"][best]
            ref_outcome = stats["outcomes"][REFERENCE]
            best_outcome = stats["outcomes"][best]
            selected = best
            reason = "best_aw"
            if (ref_loss - best_loss) < MIN_GAIN:
                selected = REFERENCE
                reason = "small_gain_fallback"
            if (best_outcome + OUTCOME_GUARD) < ref_outcome:
                selected = REFERENCE
                reason = "outcome_guard_fallback"
            new_map[key] = selected
            new_stats[key] = {**stats, "best": selected, "raw_best": best, "guard_reason": reason}
        guarded.append({**level, "map": new_map, "stats": new_stats})
    return guarded


def main() -> None:
    test, gt, frame = load_strategy_frame()
    strategies = list(STRATEGY_FILES.keys())
    levels = [
        (["gender", "tournament", "era"], 80),
        (["gender", "tournament"], 100),
        (["gender", "archetype"], 80),
    ]
    raw_maps = segment_best_maps(frame, strategies, levels)
    maps = apply_conservative_guard(raw_maps)
    sub, choice_counts, level_counts = build_stitched_submission(frame, test, maps, fallback=REFERENCE)
    sub.to_csv(DATA_DIR / OUTPUT, index=False)
    save_segment_map_csv(maps, OUTPUT.replace(".csv", "_segment_map.csv"))
    audit = write_audit(
        "experiment_e3_conservative_stitching",
        OUTPUT,
        sub,
        test,
        gt,
        {
            "levels": levels,
            "fallback": REFERENCE,
            "reference": REFERENCE,
            "outcome_guard": OUTCOME_GUARD,
            "min_gain": MIN_GAIN,
            "choice_counts": choice_counts,
            "level_counts": level_counts,
        },
    )
    print_audit(audit)


if __name__ == "__main__":
    main()

