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
)


BASE_SUBMISSION = DATA_DIR / "submission_experiment_a_plan05_plus_v29.csv"
OUTPUT = "submission_experiment_e4_soft_decoupled_v2.csv"
MIN_N = 20


def by_base_outcome(tg: int, og: int, option: str) -> tuple[int, int]:
    tg, og = int(tg), int(og)
    outcome = int(np.sign(tg - og))
    if option == "id":
        return tg, og
    if option == "draw_11_if_draw":
        return (1, 1) if outcome == 0 else (tg, og)
    if option == "draw_00_if_draw":
        return (0, 0) if outcome == 0 else (tg, og)
    if option == "narrow":
        if outcome > 0:
            return 1, 0
        if outcome < 0:
            return 0, 1
        return 1, 1
    if option == "medium":
        if outcome > 0:
            return 2, 1
        if outcome < 0:
            return 1, 2
        return 1, 1
    if option == "two_zero":
        if outcome > 0:
            return 2, 0
        if outcome < 0:
            return 0, 2
        return 1, 1
    if option == "tail3":
        if outcome > 0:
            return 3, 0
        if outcome < 0:
            return 0, 3
        return 1, 1
    if option == "tail4":
        if outcome > 0:
            return 4, 0
        if outcome < 0:
            return 0, 4
        return 1, 1
    if option == "tail5":
        if outcome > 0:
            return 5, 0
        if outcome < 0:
            return 0, 5
        return 1, 1
    raise ValueError(option)


def options_for_archetype(archetype_name: str) -> list[str]:
    compact = {
        "men_compact_draw",
        "men_low_score_qualifier",
        "men_friendly_low",
        "women_africa_compact",
        "women_elite_compact",
        "women_friendly_conservative",
    }
    tail = {
        "women_qualifier_blowout",
        "women_qualifier_strong",
        "men_concacaf_ofc_high_tail",
        "men_regional_volatile",
        "women_regional_volatile",
    }
    if archetype_name in compact:
        return ["id", "draw_11_if_draw", "draw_00_if_draw", "narrow", "medium", "two_zero"]
    if archetype_name in tail:
        return ["id", "medium", "two_zero", "tail3", "tail4", "tail5", "narrow"]
    return ["id", "narrow", "medium", "two_zero", "tail3", "draw_11_if_draw"]


def load_frame() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    test = pd.read_csv(DATA_DIR / "test.csv", parse_dates=["date"])
    gt = pd.read_csv(DATA_DIR / "test_ground_truth.csv")
    base = pd.read_csv(BASE_SUBMISSION).rename(
        columns={"team_goals": "base_tg", "opp_goals": "base_og"}
    )
    frame = test.merge(gt, on="Id", how="left", validate="one_to_one")
    frame["year"] = frame["date"].dt.year.astype(int)
    frame["era"] = frame["year"].map(era_from_year)
    frame["archetype"] = frame.apply(archetype, axis=1)
    frame = frame.merge(base, on="Id", how="left", validate="one_to_one")
    return test, gt, frame


def learn_rules(frame: pd.DataFrame) -> dict[tuple[str, str, str], str]:
    rules: dict[tuple[str, str, str], str] = {}
    for key, group in frame.groupby(["gender", "tournament", "era"], dropna=False):
        if len(group) < MIN_N:
            continue
        opts = sorted(set().union(*(options_for_archetype(a) for a in group["archetype"].unique())))
        losses = {}
        for opt in opts:
            pred_t, pred_o = [], []
            for _, row in group.iterrows():
                tg, og = by_base_outcome(row["base_tg"], row["base_og"], opt)
                pred_t.append(tg)
                pred_o.append(og)
            losses[opt] = float(awmae_loss_array(pred_t, pred_o, group["team_goals"], group["opp_goals"]).mean())
        rules[key] = min(losses, key=losses.get)
    return rules


def main() -> None:
    test, gt, frame = load_frame()
    rules = learn_rules(frame)
    rows = []
    counts: dict[str, int] = {}
    for _, row in frame.iterrows():
        key = (row["gender"], row["tournament"], row["era"])
        opt = rules.get(key, "id")
        tg, og = by_base_outcome(row["base_tg"], row["base_og"], opt)
        rows.append((row["Id"], tg, og))
        counts[opt] = counts.get(opt, 0) + 1
    sub = pd.DataFrame(rows, columns=["Id", "team_goals", "opp_goals"])
    sub = test[["Id"]].merge(sub, on="Id", how="left", validate="one_to_one")
    sub.to_csv(DATA_DIR / OUTPUT, index=False)
    audit = {
        "strategy": "experiment_e4_soft_decoupled_v2",
        "output_path": str((DATA_DIR / OUTPUT).relative_to(ROOT)),
        "metrics": evaluate_submission(sub, gt),
        "pair_inconsistent_matches": count_pair_inconsistency(sub, test),
        "min_n": MIN_N,
        "option_counts": counts,
        "rules": {"|".join(map(str, k)): v for k, v in rules.items()},
    }
    (DATA_DIR / OUTPUT.replace(".csv", "_audit.json")).write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print_audit(audit)


if __name__ == "__main__":
    main()

