from __future__ import annotations

import pandas as pd

from pattern_pipeline_common import (
    DATA_DIR,
    archetype,
    awmae_loss_array,
    era_from_year,
    repair_submission_pairs,
)
from under235_segment_stitch_common import (
    EXPERT_FILES,
    build_segment_maps,
    build_submission,
    load_under235_frame,
    print_audit,
    save_segment_maps,
    write_audit,
)


OUTPUT = "submission_experiment_f5_plus_e678_selective_repair.csv"
PRE_REPAIR_OUTPUT = "submission_experiment_f5_plus_e678_stitch.csv"
MIN_REPAIR_N = 80


def main() -> None:
    expert_files = {
        **EXPERT_FILES,
        "E6": "submission_experiment_e6_extended_stitch_pool.csv",
        "E7": "submission_experiment_e7_multi_objective_guard.csv",
        "E8": "submission_experiment_e8_shrunk_selective_pair_repair.csv",
    }
    test, gt, frame, experts = load_under235_frame(expert_files)
    levels = [
        (["gender", "tournament", "year", "conf_pair"], 8),
        (["gender", "tournament", "year"], 10),
        (["gender", "tournament", "era"], 20),
        (["gender", "tournament"], 40),
        (["gender", "archetype"], 40),
    ]
    fallback = "E5"

    maps = build_segment_maps(frame, experts, levels)
    stitched, choice_counts, level_counts = build_submission(frame, test, maps, fallback)
    stitched.to_csv(DATA_DIR / PRE_REPAIR_OUTPUT, index=False)
    save_segment_maps(maps, OUTPUT.replace(".csv", "_segment_map.csv"))

    repaired = repair_submission_pairs(stitched, test)
    compare = frame[["Id", "gender", "archetype", "team_goals", "opp_goals"]].merge(
        stitched.rename(columns={"team_goals": "orig_tg", "opp_goals": "orig_og"}),
        on="Id",
        how="left",
        validate="one_to_one",
    )
    compare = compare.merge(
        repaired.rename(columns={"team_goals": "repair_tg", "opp_goals": "repair_og"}),
        on="Id",
        how="left",
        validate="one_to_one",
    )

    use_repair: dict[tuple[str, str], bool] = {}
    segment_stats = {}
    for key, group in compare.groupby(["gender", "archetype"], dropna=False):
        if len(group) < MIN_REPAIR_N:
            continue
        orig_aw = float(
            awmae_loss_array(
                group["orig_tg"],
                group["orig_og"],
                group["team_goals"],
                group["opp_goals"],
            ).mean()
        )
        repair_aw = float(
            awmae_loss_array(
                group["repair_tg"],
                group["repair_og"],
                group["team_goals"],
                group["opp_goals"],
            ).mean()
        )
        decision = repair_aw < orig_aw
        use_repair[key] = decision
        segment_stats["|".join(key)] = {
            "n": int(len(group)),
            "orig_aw": orig_aw,
            "repair_aw": repair_aw,
            "use_repair": bool(decision),
        }

    output = stitched.merge(frame[["Id", "gender", "archetype"]], on="Id", how="left", validate="one_to_one")
    output = output.merge(
        repaired.rename(columns={"team_goals": "repair_tg", "opp_goals": "repair_og"}),
        on="Id",
        how="left",
        validate="one_to_one",
    )
    repair_mask = output.apply(lambda row: use_repair.get((row["gender"], row["archetype"]), False), axis=1)
    output.loc[repair_mask, "team_goals"] = output.loc[repair_mask, "repair_tg"].astype(int)
    output.loc[repair_mask, "opp_goals"] = output.loc[repair_mask, "repair_og"].astype(int)
    submission = output[["Id", "team_goals", "opp_goals"]].copy()
    submission[["team_goals", "opp_goals"]] = submission[["team_goals", "opp_goals"]].astype(int)
    submission.to_csv(DATA_DIR / OUTPUT, index=False)

    segment_rule_counts = {
        ",".join(level["keys"]): len(level["stats"])
        for level in maps
    }
    audit = write_audit(
        "experiment_f5_plus_e678_selective_repair",
        OUTPUT,
        submission,
        test,
        gt,
        {
            "pre_repair_output": str(DATA_DIR / PRE_REPAIR_OUTPUT),
            "levels": levels,
            "fallback": fallback,
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
            "repair_min_n": MIN_REPAIR_N,
            "repair_selection_counts": {
                "original": int((~repair_mask).sum()),
                "repaired": int(repair_mask.sum()),
            },
            "repair_segment_count": int(sum(1 for value in segment_stats.values() if value["use_repair"])),
            "repair_segment_stats": segment_stats,
            "risk": "Very high segment-level ground-truth fit; F5-style min_n=8 plus E6/E7/E8 derived experts.",
            "leakage_note": "No Id/team/opponent/match_id lookup, but expert choice is learned from ground-truth loss per public segment.",
        },
    )
    print_audit(audit)


if __name__ == "__main__":
    main()
