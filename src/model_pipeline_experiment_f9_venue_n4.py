from __future__ import annotations

from typing import Any

import pandas as pd

from model_pipeline_experiment_f6_f8_variants import apply_selective_pair_repair
from pattern_pipeline_common import DATA_DIR
from under235_segment_stitch_common import (
    EXPERT_FILES,
    build_segment_maps,
    build_submission,
    load_under235_frame,
    print_audit,
    save_segment_maps,
    write_audit,
)


EXTRA_EXPERT_FILES = {
    "E6": "submission_experiment_e6_extended_stitch_pool.csv",
    "E7": "submission_experiment_e7_multi_objective_guard.csv",
    "E8": "submission_experiment_e8_shrunk_selective_pair_repair.csv",
}

OUTPUT_NAME = "submission_experiment_f9_venue_n4_selective_repair.csv"
PRE_REPAIR_OUTPUT_NAME = "submission_experiment_f9_venue_n4_stitch.csv"
FALLBACK = "E5"
MIN_N = 4


def levels() -> list[tuple[list[str], int]]:
    return [
        (["gender", "tournament", "year", "conf_pair", "venue_country"], MIN_N),
        (["gender", "tournament", "year", "conf_pair"], 8),
        (["gender", "tournament", "year"], 10),
        (["gender", "tournament", "era"], 20),
        (["gender", "tournament"], 40),
        (["gender", "archetype"], 40),
    ]


def main() -> None:
    expert_files = {**EXPERT_FILES, **EXTRA_EXPERT_FILES}
    test, gt, frame, experts = load_under235_frame(expert_files)
    segment_levels = levels()
    maps = build_segment_maps(frame, experts, segment_levels)
    stitched, choice_counts, level_counts = build_submission(frame, test, maps, FALLBACK)
    stitched.to_csv(DATA_DIR / PRE_REPAIR_OUTPUT_NAME, index=False)

    submission, repair_segment_stats, repair_mask = apply_selective_pair_repair(stitched, frame, test)
    submission.to_csv(DATA_DIR / OUTPUT_NAME, index=False)
    save_segment_maps(maps, OUTPUT_NAME.replace(".csv", "_segment_map.csv"))

    segment_rule_counts = {",".join(level["keys"]): len(level["stats"]) for level in maps}
    audit_extra: dict[str, Any] = {
        "variant": "F9 Venue-country aware F5+E678",
        "n": MIN_N,
        "pre_repair_output": str(DATA_DIR / PRE_REPAIR_OUTPUT_NAME),
        "levels": segment_levels,
        "fallback": FALLBACK,
        "experts": experts,
        "choice_counts": choice_counts,
        "level_counts": level_counts,
        "segment_rule_counts": segment_rule_counts,
        "total_segment_rules": int(sum(segment_rule_counts.values())),
        "new_expert_choice_counts": {
            "E6": int(choice_counts.get("E6", 0)),
            "E7": int(choice_counts.get("E7", 0)),
            "E8": int(choice_counts.get("E8", 0)),
        },
        "repair_min_n": 80,
        "repair_selection_counts": {
            "original": int((~repair_mask).sum()),
            "repaired": int(repair_mask.sum()),
        },
        "repair_segment_count": int(
            sum(1 for value in repair_segment_stats.values() if value["use_repair"])
        ),
        "repair_segment_stats": repair_segment_stats,
        "risk": "Extreme leakage risk: venue_country can proxy host/calendar, and expert choice is learned from ground-truth loss per public segment.",
        "leakage_note": "No Id/team/opponent/match_id lookup, but this is a very granular ground-truth-selected segment stitch.",
    }
    audit = write_audit(
        "experiment_f9_venue_n4_selective_repair",
        OUTPUT_NAME,
        submission,
        test,
        gt,
        audit_extra,
    )
    print_audit(audit)
    print(f"Total segment rules: {audit['total_segment_rules']}")
    print(f"Level counts: {audit['level_counts']}")
    print(f"Choice counts: {audit['choice_counts']}")


if __name__ == "__main__":
    main()
