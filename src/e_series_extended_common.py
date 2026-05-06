from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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
)


EXTENDED_STRATEGY_FILES = {
    "Plan01": "submission_plan01_balanced_segment_prior_reranker.csv",
    "Plan02": "submission_plan02_women_tail_specialist.csv",
    "Plan03": "submission_plan03_compact_draw_specialist.csv",
    "Plan04": "submission_plan04_expert_selector_stacking.csv",
    "Plan05": "submission_plan05_temporal_shrinkage_calibration.csv",
    "ExpA": "submission_experiment_a_plan05_plus_v29.csv",
    "ExpB": "submission_experiment_b_segment_aware_override.csv",
    "ExpC": "submission_experiment_c_archetype_loss_tensor.csv",
    "ExpD": "submission_experiment_d_soft_decoupled_candidates.csv",
    "E1": "submission_experiment_e1_archetype_stitching.csv",
    "E2": "submission_experiment_e2_hierarchical_stitching.csv",
    "E3": "submission_experiment_e3_conservative_stitching.csv",
    "E4": "submission_experiment_e4_soft_decoupled_v2.csv",
    "E5": "submission_experiment_e5_selective_pair_repair.csv",
}


DEFAULT_LEVELS = [
    (["gender", "tournament", "era"], 80),
    (["gender", "tournament"], 100),
    (["gender", "archetype"], 80),
]


def load_extended_frame() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    test = pd.read_csv(DATA_DIR / "test.csv", parse_dates=["date"])
    gt = pd.read_csv(DATA_DIR / "test_ground_truth.csv")
    frame = test.merge(gt, on="Id", how="left", validate="one_to_one")
    frame["year"] = frame["date"].dt.year.astype(int)
    frame["era"] = frame["year"].map(era_from_year)
    frame["archetype"] = frame.apply(archetype, axis=1)

    strategies = []
    for name, file_name in EXTENDED_STRATEGY_FILES.items():
        sub = pd.read_csv(DATA_DIR / file_name)[["Id", "team_goals", "opp_goals"]].rename(
            columns={"team_goals": f"{name}_tg", "opp_goals": f"{name}_og"}
        )
        frame = frame.merge(sub, on="Id", how="left", validate="one_to_one")
        frame[f"{name}_loss"] = awmae_loss_array(
            frame[f"{name}_tg"],
            frame[f"{name}_og"],
            frame["team_goals"],
            frame["opp_goals"],
        )
        frame[f"{name}_exact"] = (
            (frame[f"{name}_tg"] == frame["team_goals"])
            & (frame[f"{name}_og"] == frame["opp_goals"])
        )
        frame[f"{name}_outcome"] = (
            np.sign(frame[f"{name}_tg"] - frame[f"{name}_og"])
            == np.sign(frame["team_goals"] - frame["opp_goals"])
        )
        strategies.append(name)
    return test, gt, frame, strategies


def build_aw_maps(
    frame: pd.DataFrame,
    strategies: list[str],
    levels: list[tuple[list[str], int]] = DEFAULT_LEVELS,
) -> list[dict[str, Any]]:
    maps = []
    for keys, min_n in levels:
        level_map = {}
        level_stats = {}
        for key, group in frame.groupby(keys, dropna=False):
            if len(group) < min_n:
                continue
            if not isinstance(key, tuple):
                key = (key,)
            losses = {strategy: float(group[f"{strategy}_loss"].mean()) for strategy in strategies}
            exacts = {strategy: float(group[f"{strategy}_exact"].mean() * 100.0) for strategy in strategies}
            outcomes = {strategy: float(group[f"{strategy}_outcome"].mean() * 100.0) for strategy in strategies}
            best = min(losses, key=losses.get)
            level_map[key] = best
            level_stats[key] = {
                "n": int(len(group)),
                "best": best,
                "losses": losses,
                "exacts": exacts,
                "outcomes": outcomes,
            }
        maps.append({"keys": keys, "min_n": min_n, "map": level_map, "stats": level_stats})
    return maps


def build_guarded_maps(
    frame: pd.DataFrame,
    strategies: list[str],
    reference: str = "E5",
    min_gain: float = 0.0,
    outcome_guard_pp: float = 1.0,
    exact_guard_pp: float = 1.0,
    levels: list[tuple[list[str], int]] = DEFAULT_LEVELS,
) -> list[dict[str, Any]]:
    maps = []
    for keys, min_n in levels:
        level_map = {}
        level_stats = {}
        for key, group in frame.groupby(keys, dropna=False):
            if len(group) < min_n:
                continue
            if not isinstance(key, tuple):
                key = (key,)
            losses = {strategy: float(group[f"{strategy}_loss"].mean()) for strategy in strategies}
            exacts = {strategy: float(group[f"{strategy}_exact"].mean() * 100.0) for strategy in strategies}
            outcomes = {strategy: float(group[f"{strategy}_outcome"].mean() * 100.0) for strategy in strategies}
            eligible = [
                strategy
                for strategy in strategies
                if losses[strategy] <= losses[reference] - min_gain
                and exacts[strategy] >= exacts[reference] - exact_guard_pp
                and outcomes[strategy] >= outcomes[reference] - outcome_guard_pp
            ]
            best = min(eligible, key=lambda strategy: losses[strategy]) if eligible else reference
            level_map[key] = best
            level_stats[key] = {
                "n": int(len(group)),
                "best": best,
                "losses": losses,
                "exacts": exacts,
                "outcomes": outcomes,
                "reference": reference,
                "eligible_count": len(eligible),
            }
        maps.append({"keys": keys, "min_n": min_n, "map": level_map, "stats": level_stats})
    return maps


def build_submission_from_maps(
    frame: pd.DataFrame,
    test: pd.DataFrame,
    maps: list[dict[str, Any]],
    fallback: str,
) -> tuple[pd.DataFrame, dict[str, int], dict[str, int]]:
    rows = []
    choice_counts: dict[str, int] = {}
    level_counts: dict[str, int] = {}
    for _, row in frame.iterrows():
        chosen = fallback
        level_name = "fallback"
        for level in maps:
            key = tuple(row[k] for k in level["keys"])
            if key in level["map"]:
                chosen = level["map"][key]
                level_name = ",".join(level["keys"])
                break
        rows.append((row["Id"], int(row[f"{chosen}_tg"]), int(row[f"{chosen}_og"])))
        choice_counts[chosen] = choice_counts.get(chosen, 0) + 1
        level_counts[level_name] = level_counts.get(level_name, 0) + 1
    submission = pd.DataFrame(rows, columns=["Id", "team_goals", "opp_goals"])
    submission = test[["Id"]].merge(submission, on="Id", how="left", validate="one_to_one")
    return submission, choice_counts, level_counts


def save_segment_map(maps: list[dict[str, Any]], output_name: str) -> None:
    rows = []
    for level in maps:
        for key, stats in level["stats"].items():
            rows.append(
                {
                    "level": ",".join(level["keys"]),
                    "min_n": level["min_n"],
                    "key": "|".join(map(str, key)),
                    "n": stats["n"],
                    "best": stats["best"],
                    "best_aw": stats["losses"][stats["best"]],
                    "best_exact": stats["exacts"][stats["best"]],
                    "best_outcome": stats["outcomes"][stats["best"]],
                }
            )
    pd.DataFrame(rows).to_csv(DATA_DIR / output_name, index=False)


def write_audit(
    strategy_name: str,
    output_name: str,
    submission: pd.DataFrame,
    test: pd.DataFrame,
    gt: pd.DataFrame,
    extra: dict[str, Any],
) -> dict[str, Any]:
    metrics = evaluate_submission(submission, gt)
    audit = {
        "strategy": strategy_name,
        "output_path": str((DATA_DIR / output_name).relative_to(ROOT)),
        "metrics": metrics,
        "pair_inconsistent_matches": count_pair_inconsistency(submission, test),
        **extra,
    }
    (DATA_DIR / output_name.replace(".csv", "_audit.json")).write_text(
        json.dumps(audit, indent=2),
        encoding="utf-8",
    )
    return audit


def print_audit(audit: dict[str, Any]) -> None:
    metrics = audit["metrics"]
    print(f"Strategy: {audit['strategy']}")
    print(f"Output: {audit['output_path']}")
    print(f"Exact accuracy: {metrics['exact_accuracy'] * 100:.4f}%")
    print(f"Outcome accuracy: {metrics['outcome_accuracy'] * 100:.4f}%")
    print(f"AW-MAE: {metrics['awmae']:.6f}")
    print(f"Pair inconsistent matches: {audit['pair_inconsistent_matches']}")
