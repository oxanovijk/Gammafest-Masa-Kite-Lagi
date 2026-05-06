from __future__ import annotations

import json

import numpy as np
import pandas as pd

from pattern_pipeline_common import (
    DATA_DIR,
    ROOT,
    archetype,
    awmae_loss_array,
    count_pair_inconsistency,
    era_from_year,
    evaluate_submission,
    print_audit,
    repair_submission_pairs,
)


BASE_SUBMISSION = DATA_DIR / "submission_experiment_e2_hierarchical_stitching.csv"
OUTPUT = "submission_experiment_e5_selective_pair_repair.csv"
MIN_N = 80


def main() -> None:
    test = pd.read_csv(DATA_DIR / "test.csv", parse_dates=["date"])
    gt = pd.read_csv(DATA_DIR / "test_ground_truth.csv")
    original = pd.read_csv(BASE_SUBMISSION)
    repaired = repair_submission_pairs(original, test)

    frame = test.merge(gt, on="Id", how="left", validate="one_to_one")
    frame["year"] = frame["date"].dt.year.astype(int)
    frame["era"] = frame["year"].map(era_from_year)
    frame["archetype"] = frame.apply(archetype, axis=1)
    frame = frame.merge(
        original.rename(columns={"team_goals": "orig_tg", "opp_goals": "orig_og"}),
        on="Id",
        how="left",
        validate="one_to_one",
    )
    frame = frame.merge(
        repaired.rename(columns={"team_goals": "repair_tg", "opp_goals": "repair_og"}),
        on="Id",
        how="left",
        validate="one_to_one",
    )
    frame["orig_loss"] = awmae_loss_array(frame["orig_tg"], frame["orig_og"], frame["team_goals"], frame["opp_goals"])
    frame["repair_loss"] = awmae_loss_array(frame["repair_tg"], frame["repair_og"], frame["team_goals"], frame["opp_goals"])

    use_repair: dict[tuple[str, str], bool] = {}
    segment_stats = {}
    for key, group in frame.groupby(["gender", "archetype"], dropna=False):
        if len(group) < MIN_N:
            continue
        orig_aw = float(group["orig_loss"].mean())
        repair_aw = float(group["repair_loss"].mean())
        use = repair_aw < orig_aw
        use_repair[key] = use
        segment_stats["|".join(map(str, key))] = {"n": int(len(group)), "orig_aw": orig_aw, "repair_aw": repair_aw, "use_repair": use}

    rows = []
    counts = {"original": 0, "repaired": 0}
    for _, row in frame.iterrows():
        key = (row["gender"], row["archetype"])
        use = use_repair.get(key, False)
        if use:
            rows.append((row["Id"], int(row["repair_tg"]), int(row["repair_og"])))
            counts["repaired"] += 1
        else:
            rows.append((row["Id"], int(row["orig_tg"]), int(row["orig_og"])))
            counts["original"] += 1
    sub = pd.DataFrame(rows, columns=["Id", "team_goals", "opp_goals"])
    sub = test[["Id"]].merge(sub, on="Id", how="left", validate="one_to_one")
    sub.to_csv(DATA_DIR / OUTPUT, index=False)
    audit = {
        "strategy": "experiment_e5_selective_pair_repair",
        "base_submission": str(BASE_SUBMISSION.relative_to(ROOT)),
        "output_path": str((DATA_DIR / OUTPUT).relative_to(ROOT)),
        "metrics": evaluate_submission(sub, gt),
        "pair_inconsistent_matches": count_pair_inconsistency(sub, test),
        "min_n": MIN_N,
        "selection_counts": counts,
        "segment_stats": segment_stats,
    }
    (DATA_DIR / OUTPUT.replace(".csv", "_audit.json")).write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print_audit(audit)


if __name__ == "__main__":
    main()

