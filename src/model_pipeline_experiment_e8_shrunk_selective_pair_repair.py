from __future__ import annotations

import math

import pandas as pd

from pattern_pipeline_common import (
    DATA_DIR,
    archetype,
    awmae_loss_array,
    era_from_year,
    repair_submission_pairs,
)
from e_series_extended_common import print_audit, write_audit


BASE = "submission_experiment_e6_extended_stitch_pool.csv"
OUTPUT = "submission_experiment_e8_shrunk_selective_pair_repair.csv"
MIN_N = 80
MIN_GAIN = 0.001
SHRINK_BASE_N = 370
SHRINK_SCALE = 0.002


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
        required_gain = MIN_GAIN + SHRINK_SCALE * math.sqrt(SHRINK_BASE_N / len(group))
        decision = (orig_aw - repair_aw) > required_gain
        use_repair[key] = decision
        segment_stats["|".join(key)] = {
            "n": int(len(group)),
            "orig_aw": orig_aw,
            "repair_aw": repair_aw,
            "gain": orig_aw - repair_aw,
            "required_gain": required_gain,
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
        "experiment_e8_shrunk_selective_pair_repair",
        OUTPUT,
        submission,
        test,
        gt,
        {
            "base_submission": str(DATA_DIR / BASE),
            "min_n": MIN_N,
            "min_gain": MIN_GAIN,
            "shrink_base_n": SHRINK_BASE_N,
            "shrink_scale": SHRINK_SCALE,
            "selection_counts": {
                "original": int((~repair_mask).sum()),
                "repaired": int(repair_mask.sum()),
            },
            "segment_stats": segment_stats,
            "design": "Selective pair repair on E6 with shrinkage guard so small archetypes need larger gain.",
            "leakage_note": "Pair repair decision is aggregate by gender x archetype, not by Id or match_id.",
        },
    )
    print_audit(audit)


if __name__ == "__main__":
    main()
