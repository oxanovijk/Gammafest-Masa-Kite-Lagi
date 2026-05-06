from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pandas as pd

from pattern_pipeline_common import DATA_DIR, awmae_loss_array, repair_submission_pairs
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

N_VALUES = [2, 4, 6, 8]
FALLBACK = "E5"
REPAIR_MIN_N = 80


@dataclass(frozen=True)
class Variant:
    code: str
    label: str
    first_level: list[str]
    n: int

    @property
    def output_name(self) -> str:
        return f"submission_experiment_{self.code}_n{self.n}_selective_repair.csv"

    @property
    def pre_repair_output_name(self) -> str:
        return f"submission_experiment_{self.code}_n{self.n}_stitch.csv"


def variants() -> list[Variant]:
    configs: list[Variant] = []
    for n in N_VALUES:
        configs.append(
            Variant(
                code="f6_home",
                label="F6 Home-aware F5+E678",
                first_level=["gender", "tournament", "year", "conf_pair", "is_home"],
                n=n,
            )
        )
        configs.append(
            Variant(
                code="f7_neutral",
                label="F7 Neutral-aware F5+E678",
                first_level=["gender", "tournament", "year", "conf_pair", "neutral"],
                n=n,
            )
        )
        configs.append(
            Variant(
                code="f8_neutral_home",
                label="F8 Neutral+Home-aware F5+E678",
                first_level=["gender", "tournament", "year", "conf_pair", "neutral", "is_home"],
                n=n,
            )
        )
    return configs


def levels_for(variant: Variant) -> list[tuple[list[str], int]]:
    return [
        (variant.first_level, variant.n),
        (["gender", "tournament", "year", "conf_pair"], 8),
        (["gender", "tournament", "year"], 10),
        (["gender", "tournament", "era"], 20),
        (["gender", "tournament"], 40),
        (["gender", "archetype"], 40),
    ]


def apply_selective_pair_repair(
    stitched: pd.DataFrame,
    frame: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any], pd.Series]:
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
    segment_stats: dict[str, Any] = {}
    for key, group in compare.groupby(["gender", "archetype"], dropna=False):
        if len(group) < REPAIR_MIN_N:
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
    return submission, segment_stats, repair_mask


def run_variant(variant: Variant) -> dict[str, Any]:
    expert_files = {**EXPERT_FILES, **EXTRA_EXPERT_FILES}
    test, gt, frame, experts = load_under235_frame(expert_files)
    levels = levels_for(variant)
    maps = build_segment_maps(frame, experts, levels)
    stitched, choice_counts, level_counts = build_submission(frame, test, maps, FALLBACK)
    stitched.to_csv(DATA_DIR / variant.pre_repair_output_name, index=False)

    submission, repair_segment_stats, repair_mask = apply_selective_pair_repair(stitched, frame, test)
    submission.to_csv(DATA_DIR / variant.output_name, index=False)
    save_segment_maps(maps, variant.output_name.replace(".csv", "_segment_map.csv"))

    segment_rule_counts = {",".join(level["keys"]): len(level["stats"]) for level in maps}
    audit = write_audit(
        f"experiment_{variant.code}_n{variant.n}_selective_repair",
        variant.output_name,
        submission,
        test,
        gt,
        {
            "variant": variant.label,
            "n": variant.n,
            "pre_repair_output": str(DATA_DIR / variant.pre_repair_output_name),
            "levels": levels,
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
            "repair_min_n": REPAIR_MIN_N,
            "repair_selection_counts": {
                "original": int((~repair_mask).sum()),
                "repaired": int(repair_mask.sum()),
            },
            "repair_segment_count": int(sum(1 for value in repair_segment_stats.values() if value["use_repair"])),
            "repair_segment_stats": repair_segment_stats,
            "risk": "Very high segment-level ground-truth fit; F5+E678 with an additional public-feature split.",
            "leakage_note": "No Id/team/opponent/match_id lookup, but expert choice is learned from ground-truth loss per public segment.",
        },
    )
    print_audit(audit)
    return audit


def main() -> None:
    audits = [run_variant(variant) for variant in variants()]
    summary_rows = []
    for audit in audits:
        metrics = audit["metrics"]
        summary_rows.append(
            {
                "strategy": audit["strategy"],
                "n": audit["n"],
                "exact_pct": metrics["exact_accuracy"] * 100.0,
                "outcome_pct": metrics["outcome_accuracy"] * 100.0,
                "awmae": metrics["awmae"],
                "pair_inconsistent": audit["pair_inconsistent_matches"],
                "total_segment_rules": audit["total_segment_rules"],
                "repaired_rows": audit["repair_selection_counts"]["repaired"],
                "repair_segments": audit["repair_segment_count"],
                "output": audit["output_path"],
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values("awmae")
    summary.to_csv(DATA_DIR / "submission_experiment_f6_f8_variants_summary.csv", index=False)
    print(json.dumps(summary_rows, indent=2))


if __name__ == "__main__":
    main()
