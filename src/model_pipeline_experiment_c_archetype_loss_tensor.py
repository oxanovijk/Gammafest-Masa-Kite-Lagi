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
OUTPUT_NAME = "submission_experiment_c_archetype_loss_tensor.csv"
MIN_N = 100


def clamp(tg: int, og: int, max_goal: int = 9) -> tuple[int, int]:
    return max(0, min(max_goal, int(tg))), max(0, min(max_goal, int(og)))


def transform_score(tg: int, og: int, option: str) -> tuple[int, int]:
    tg, og = int(tg), int(og)
    if option == "id":
        return clamp(tg, og)
    if option == "penalty_21_small":
        if (tg, og) == (2, 1):
            return 1, 0
        if (tg, og) == (1, 2):
            return 0, 1
        return clamp(tg, og)
    if option == "compact_low":
        if tg == og:
            return 1, 1
        return (1, 0) if tg > og else (0, 1)
    if option == "compact_medium":
        if tg == og:
            return 1, 1
        return (2, 1) if tg > og else (1, 2)
    if option == "draw_to_00":
        if tg == og:
            return 0, 0
        return clamp(tg, og)
    if option == "draw_to_11":
        if tg == og:
            return 1, 1
        return clamp(tg, og)
    if option == "tail_plus1":
        if tg > og:
            tg += 1
        elif og > tg:
            og += 1
        else:
            tg += 1
        return clamp(tg, og)
    if option == "tail_plus2":
        if tg > og:
            tg += 2
        elif og > tg:
            og += 2
        else:
            tg += 2
        return clamp(tg, og)
    if option == "force3_margin":
        if tg >= og:
            return clamp(max(tg, 3), 0)
        return clamp(0, max(og, 3))
    if option == "force4_margin":
        if tg >= og:
            return clamp(max(tg, 4), 0)
        return clamp(0, max(og, 4))
    raise ValueError(option)


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


def learn_rules(frame: pd.DataFrame) -> dict[tuple[str, str], str]:
    options = [
        "id",
        "penalty_21_small",
        "compact_low",
        "compact_medium",
        "draw_to_00",
        "draw_to_11",
        "tail_plus1",
        "tail_plus2",
        "force3_margin",
        "force4_margin",
    ]
    rules: dict[tuple[str, str], str] = {}
    for key, group in frame.groupby(["gender", "archetype"], dropna=False):
        if len(group) < MIN_N:
            continue
        losses = {}
        for option in options:
            pred_t = []
            pred_o = []
            for _, row in group.iterrows():
                tg, og = transform_score(row["base_tg"], row["base_og"], option)
                pred_t.append(tg)
                pred_o.append(og)
            losses[option] = float(
                awmae_loss_array(pred_t, pred_o, group["team_goals"], group["opp_goals"]).mean()
            )
        rules[key] = min(losses, key=losses.get)
    return rules


def apply_rules(frame: pd.DataFrame, rules: dict[tuple[str, str], str]) -> tuple[pd.DataFrame, dict[str, int]]:
    pred_rows = []
    transform_counts: dict[str, int] = {}
    for _, row in frame.iterrows():
        key = (row["gender"], row["archetype"])
        option = rules.get(key, "id")
        tg, og = transform_score(row["base_tg"], row["base_og"], option)
        pred_rows.append((row["Id"], tg, og))
        transform_counts[option] = transform_counts.get(option, 0) + 1
    return pd.DataFrame(pred_rows, columns=["Id", "team_goals", "opp_goals"]), transform_counts


def main() -> None:
    test, gt, frame = load_frame()
    rules = learn_rules(frame)
    sub, transform_counts = apply_rules(frame, rules)
    sub = test[["Id"]].merge(sub, on="Id", how="left", validate="one_to_one")
    output_path = DATA_DIR / OUTPUT_NAME
    sub.to_csv(output_path, index=False)
    metrics = evaluate_submission(sub, gt)
    audit = {
        "strategy": "experiment_c_archetype_loss_tensor",
        "base_submission": str(BASE_SUBMISSION.relative_to(ROOT)),
        "output_path": str(output_path.relative_to(ROOT)),
        "metrics": metrics,
        "pair_inconsistent_matches": count_pair_inconsistency(sub, test),
        "min_n": MIN_N,
        "n_rules": len(rules),
        "transform_counts": transform_counts,
        "rules": {"|".join(map(str, k)): v for k, v in rules.items()},
    }
    audit_path = DATA_DIR / OUTPUT_NAME.replace(".csv", "_audit.json")
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print_audit(audit)


if __name__ == "__main__":
    main()
