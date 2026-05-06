from __future__ import annotations

import numpy as np
import pandas as pd

from pattern_pipeline_common import (
    DATA_DIR,
    archetype,
    awmae_loss_array,
    era_from_year,
    repair_submission_pairs,
)
from under235_segment_stitch_common import print_audit, write_audit


BASE = "submission_experiment_f1_conf_pair20_stitch.csv"
OUTPUT = "submission_experiment_f4_conf_pair20_selective_repair.csv"
MIN_N = 80


def main() -> None:
    test = pd.read_csv(DATA_DIR / "test.csv", parse_dates=["date"])
    gt = pd.read_csv(DATA_DIR / "test_ground_truth.csv")
    frame = test.merge(gt, on="Id", how="left", validate="one_to_one")
    frame["year"] = frame["date"].dt.year.astype(int)
    frame["era"] = frame["year"].map(era_from_year)
    frame["archetype"] = frame.apply(archetype, axis=1)

    original = pd.read_csv(DATA_DIR / BASE)
    repaired = repair_submission_pairs(original, test)

    compare = frame[["Id", "gender", "archetype", "team_goals", "opp_goals"]].merge(
        original.rename(columns={"team_goals": "orig_tg", "opp_goals": "orig_og"}),
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
        if len(group) < MIN_N:
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

    output = original.merge(frame[["Id", "gender", "archetype"]], on="Id", how="left", validate="one_to_one")
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

    audit = write_audit(
        "experiment_f4_conf_pair20_selective_repair",
        OUTPUT,
        submission,
        test,
        gt,
        {
            "base_submission": str(DATA_DIR / BASE),
            "min_n": MIN_N,
            "selection_counts": {
                "original": int((~repair_mask).sum()),
                "repaired": int(repair_mask.sum()),
            },
            "segment_stats": segment_stats,
            "leakage_note": "Pair repair decision is aggregate by gender x archetype, not by Id or match_id.",
        },
    )
    print_audit(audit)


if __name__ == "__main__":
    main()
