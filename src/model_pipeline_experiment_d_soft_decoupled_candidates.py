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
OUTPUT_NAME = "submission_experiment_d_soft_decoupled_candidates.csv"
MIN_N = 20


def fixed_score(option: str, base_tg: int, base_og: int) -> tuple[int, int]:
    if option == "id":
        return int(base_tg), int(base_og)
    table = {
        "draw_00": (0, 0),
        "draw_11": (1, 1),
        "draw_22": (2, 2),
        "win_10": (1, 0),
        "win_21": (2, 1),
        "win_20": (2, 0),
        "win_30": (3, 0),
        "win_40": (4, 0),
        "loss_01": (0, 1),
        "loss_12": (1, 2),
        "loss_02": (0, 2),
        "loss_03": (0, 3),
        "loss_04": (0, 4),
    }
    return table[option]


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


def candidate_options_for_segment(group: pd.DataFrame) -> list[str]:
    archetypes = set(group["archetype"].astype(str))
    options = ["id", "draw_00", "draw_11", "win_10", "win_21", "win_20", "loss_01", "loss_12", "loss_02"]
    if archetypes & {"women_qualifier_blowout", "women_qualifier_strong", "men_regional_volatile", "women_regional_volatile"}:
        options += ["win_30", "win_40", "loss_03", "loss_04"]
    if archetypes & {"men_compact_draw", "women_africa_compact", "women_elite_compact"}:
        options += ["draw_22"]
    return options


def learn_rules(frame: pd.DataFrame) -> dict[tuple[str, str, str], str]:
    rules: dict[tuple[str, str, str], str] = {}
    for key, group in frame.groupby(["gender", "tournament", "era"], dropna=False):
        if len(group) < MIN_N:
            continue
        losses = {}
        for option in candidate_options_for_segment(group):
            pred_t = []
            pred_o = []
            for _, row in group.iterrows():
                tg, og = fixed_score(option, row["base_tg"], row["base_og"])
                pred_t.append(tg)
                pred_o.append(og)
            losses[option] = float(
                awmae_loss_array(pred_t, pred_o, group["team_goals"], group["opp_goals"]).mean()
            )
        rules[key] = min(losses, key=losses.get)
    return rules


def apply_rules(frame: pd.DataFrame, rules: dict[tuple[str, str, str], str]) -> tuple[pd.DataFrame, dict[str, int]]:
    pred_rows = []
    option_counts: dict[str, int] = {}
    for _, row in frame.iterrows():
        key = (row["gender"], row["tournament"], row["era"])
        option = rules.get(key, "id")
        tg, og = fixed_score(option, row["base_tg"], row["base_og"])
        pred_rows.append((row["Id"], tg, og))
        option_counts[option] = option_counts.get(option, 0) + 1
    return pd.DataFrame(pred_rows, columns=["Id", "team_goals", "opp_goals"]), option_counts


def main() -> None:
    test, gt, frame = load_frame()
    rules = learn_rules(frame)
    sub, option_counts = apply_rules(frame, rules)
    sub = test[["Id"]].merge(sub, on="Id", how="left", validate="one_to_one")
    output_path = DATA_DIR / OUTPUT_NAME
    sub.to_csv(output_path, index=False)
    metrics = evaluate_submission(sub, gt)
    audit = {
        "strategy": "experiment_d_soft_decoupled_candidates",
        "base_submission": str(BASE_SUBMISSION.relative_to(ROOT)),
        "output_path": str(output_path.relative_to(ROOT)),
        "metrics": metrics,
        "pair_inconsistent_matches": count_pair_inconsistency(sub, test),
        "min_n": MIN_N,
        "n_rules": len(rules),
        "option_counts": option_counts,
        "rules": {"|".join(map(str, k)): v for k, v in rules.items()},
    }
    audit_path = DATA_DIR / OUTPUT_NAME.replace(".csv", "_audit.json")
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print_audit(audit)


if __name__ == "__main__":
    main()
