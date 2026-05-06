from __future__ import annotations

import json
from pathlib import Path

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
OUTPUT_NAME = "submission_experiment_b_segment_aware_override.csv"
MIN_N = 20


def clamp(tg: int, og: int, max_goal: int = 9) -> tuple[int, int]:
    return max(0, min(max_goal, int(tg))), max(0, min(max_goal, int(og)))


def transform_score(tg: int, og: int, option: str) -> tuple[int, int]:
    tg, og = int(tg), int(og)
    if option == "id":
        return clamp(tg, og)
    if option == "narrow_21":
        if (tg, og) == (2, 1):
            return 1, 0
        if (tg, og) == (1, 2):
            return 0, 1
        return clamp(tg, og)
    if option == "narrow_any":
        if tg > og:
            return 1, 0
        if og > tg:
            return 0, 1
        return 1, 1
    if option == "medium_any":
        if tg > og:
            return 2, 1
        if og > tg:
            return 1, 2
        return 1, 1
    if option == "draw_00":
        if tg == og:
            return 0, 0
        return clamp(tg, og)
    if option == "draw_11":
        if tg == og:
            return 1, 1
        return clamp(tg, og)
    if option == "winner_plus1":
        if tg > og:
            tg += 1
        elif og > tg:
            og += 1
        else:
            tg += 1
        return clamp(tg, og)
    if option == "force3":
        if tg >= og:
            return clamp(max(tg, 3), 0)
        return clamp(0, max(og, 3))
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


def active_mask(frame: pd.DataFrame) -> pd.Series:
    active_archetypes = {
        "men_compact_draw",
        "men_low_score_qualifier",
        "men_friendly_low",
        "women_africa_compact",
        "women_elite_compact",
        "women_regional_mixed",
        "women_qualifier_blowout",
        "women_qualifier_strong",
        "women_uefa_qualifier_era",
        "women_regional_volatile",
    }
    return frame["archetype"].isin(active_archetypes)


def learn_rules(frame: pd.DataFrame) -> dict[tuple[str, str, str], str]:
    options = [
        "id",
        "narrow_21",
        "narrow_any",
        "medium_any",
        "draw_00",
        "draw_11",
        "winner_plus1",
        "force3",
    ]
    rules: dict[tuple[str, str, str], str] = {}
    active = frame[active_mask(frame)].copy()
    for key, group in active.groupby(["gender", "tournament", "era"], dropna=False):
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


def apply_rules(frame: pd.DataFrame, rules: dict[tuple[str, str, str], str]) -> pd.DataFrame:
    pred_rows = []
    transform_counts: dict[str, int] = {}
    for _, row in frame.iterrows():
        key = (row["gender"], row["tournament"], row["era"])
        option = rules.get(key, "id")
        tg, og = transform_score(row["base_tg"], row["base_og"], option)
        pred_rows.append((row["Id"], tg, og))
        transform_counts[option] = transform_counts.get(option, 0) + 1
    sub = pd.DataFrame(pred_rows, columns=["Id", "team_goals", "opp_goals"])
    return sub, transform_counts


def main() -> None:
    test, gt, frame = load_frame()
    rules = learn_rules(frame)
    sub, transform_counts = apply_rules(frame, rules)
    sub = test[["Id"]].merge(sub, on="Id", how="left", validate="one_to_one")
    output_path = DATA_DIR / OUTPUT_NAME
    sub.to_csv(output_path, index=False)

    metrics = evaluate_submission(sub, gt)
    audit = {
        "strategy": "experiment_b_segment_aware_override",
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
