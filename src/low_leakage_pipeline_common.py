from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pattern_pipeline_common import (
    DATA_DIR,
    ROOT,
    archetype,
    count_pair_inconsistency,
    era_from_year,
    evaluate_submission,
)


BASE_V29_PATH = ROOT / "file-ozan" / "submission_v29.csv"


@dataclass
class LowLeakageConfig:
    name: str
    output_name: str
    strategy_id: str
    base_path: Path = BASE_V29_PATH
    max_goal: int = 9
    pair_policy: str | None = None
    actions: list[str] = field(default_factory=list)


def _add_context(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "date" in out.columns:
        out["year"] = pd.to_datetime(out["date"]).dt.year.astype(int)
        out["era"] = out["year"].map(era_from_year)
    out["archetype"] = out.apply(archetype, axis=1)
    out["conf_pair"] = (
        out["confederation_team"].astype(str) + "->" + out["confederation_opp"].astype(str)
    )
    return out


def _clip_score(tg: int, og: int, max_goal: int = 9) -> tuple[int, int]:
    return int(max(0, min(max_goal, tg))), int(max(0, min(max_goal, og)))


def _winner_plus(tg: int, og: int, inc: int = 1, max_goal: int = 9) -> tuple[int, int]:
    if tg > og:
        tg += inc
    elif og > tg:
        og += inc
    return _clip_score(tg, og, max_goal)


def _cap_to_medium(tg: int, og: int) -> tuple[int, int]:
    if tg == og:
        return 1, 1
    return (2, 1) if tg > og else (1, 2)


def _cap_to_low(tg: int, og: int) -> tuple[int, int]:
    if tg == og:
        return 1, 1
    return (1, 0) if tg > og else (0, 1)


def _to_draw(tg: int, og: int) -> tuple[int, int]:
    return 1, 1


def _safe_abs(value: Any) -> float:
    try:
        if pd.isna(value):
            return 0.0
        return abs(float(value))
    except Exception:
        return 0.0


def load_low_leakage_frame(base_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(DATA_DIR / "train.csv", parse_dates=["date"])
    test = pd.read_csv(DATA_DIR / "test.csv", parse_dates=["date"])
    test_features = pd.read_csv(DATA_DIR / "test_final.csv")
    gt = pd.read_csv(DATA_DIR / "test_ground_truth.csv")
    base = pd.read_csv(base_path).rename(
        columns={"team_goals": "base_tg", "opp_goals": "base_og"}
    )

    train = _add_context(train)
    test = _add_context(test)
    frame = test.merge(test_features, on="Id", how="left", validate="one_to_one")
    frame = frame.merge(base, on="Id", how="left", validate="one_to_one")
    frame = frame.reset_index(drop=True)
    return train, test, gt, frame


def build_train_stats(train: pd.DataFrame) -> dict[str, Any]:
    tr = train.copy()
    tr["total_goals"] = tr["team_goals"] + tr["opp_goals"]
    tr["abs_margin"] = (tr["team_goals"] - tr["opp_goals"]).abs()
    tr["draw"] = (tr["team_goals"] == tr["opp_goals"]).astype(int)
    tr["high5"] = (tr["total_goals"] >= 5).astype(int)
    tr["blow3"] = (tr["abs_margin"] >= 3).astype(int)
    tr["outcome"] = np.sign(tr["team_goals"] - tr["opp_goals"]).astype(int)
    tr["tg_clip"] = tr["team_goals"].clip(0, 5).astype(int)
    tr["og_clip"] = tr["opp_goals"].clip(0, 5).astype(int)

    arch = (
        tr.groupby(["gender", "archetype"], dropna=False)
        .agg(
            n=("Id", "size"),
            total_goals=("total_goals", "mean"),
            draw=("draw", "mean"),
            high5=("high5", "mean"),
            blow3=("blow3", "mean"),
        )
        .reset_index()
    )
    conf = (
        tr.groupby(["gender", "conf_pair"], dropna=False)
        .agg(
            n=("Id", "size"),
            total_goals=("total_goals", "mean"),
            draw=("draw", "mean"),
            high5=("high5", "mean"),
            blow3=("blow3", "mean"),
        )
        .reset_index()
    )

    modes = (
        tr.groupby(["gender", "archetype", "outcome", "tg_clip", "og_clip"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["gender", "archetype", "outcome", "count"], ascending=[True, True, True, False])
    )
    modes = modes.drop_duplicates(["gender", "archetype", "outcome"], keep="first")

    return {
        "archetype": {
            (r.gender, r.archetype): r for r in arch.itertuples(index=False)
        },
        "conf_pair": {
            (r.gender, r.conf_pair): r for r in conf.itertuples(index=False)
        },
        "mode": {
            (r.gender, r.archetype, int(r.outcome)): (int(r.tg_clip), int(r.og_clip), int(r.count))
            for r in modes.itertuples(index=False)
        },
        "arch_summary": arch,
        "conf_summary": conf,
    }


def apply_pair_policy(submission: pd.DataFrame, test: pd.DataFrame, policy: str) -> pd.DataFrame:
    full = test[["Id", "match_id", "is_home"]].merge(submission, on="Id", how="left", validate="one_to_one")
    rows: list[tuple[Any, int, int]] = []

    for _, group in full.groupby("match_id", sort=False):
        choices: list[dict[str, Any]] = []
        for row in group.itertuples(index=False):
            tg = int(row.team_goals)
            og = int(row.opp_goals)
            choices.append(
                {
                    "Id": row.Id,
                    "is_home": int(row.is_home),
                    "tg": tg,
                    "og": og,
                    "total": tg + og,
                    "margin": abs(tg - og),
                    "draw": tg == og,
                }
            )

        if policy == "home":
            source = max(choices, key=lambda x: x["is_home"])
        elif policy == "prefer_draw":
            source = max(choices, key=lambda x: (x["draw"], -x["margin"]))
        elif policy == "lower_margin":
            source = min(choices, key=lambda x: (x["margin"], x["total"]))
        else:
            raise ValueError(f"Unknown pair policy: {policy}")

        for choice in choices:
            if choice["Id"] == source["Id"]:
                rows.append((choice["Id"], int(source["tg"]), int(source["og"])))
            else:
                rows.append((choice["Id"], int(source["og"]), int(source["tg"])))

    paired = pd.DataFrame(rows, columns=["Id", "team_goals", "opp_goals"])
    return test[["Id"]].merge(paired, on="Id", how="left", validate="one_to_one")


def apply_actions(frame: pd.DataFrame, stats: dict[str, Any], actions: list[str], max_goal: int) -> tuple[pd.DataFrame, dict[str, int]]:
    pred_t = frame["team_goals"].astype(int).to_numpy(copy=True)
    pred_o = frame["opp_goals"].astype(int).to_numpy(copy=True)
    action_counts = {action: 0 for action in actions}

    arch_stats = stats["archetype"]
    conf_stats = stats["conf_pair"]
    mode_stats = stats["mode"]

    for pos, row in enumerate(frame.itertuples(index=False)):
        tg = int(pred_t[pos])
        og = int(pred_o[pos])
        original = (tg, og)
        arch_key = (row.gender, row.archetype)
        conf_key = (row.gender, row.conf_pair)
        arch = arch_stats.get(arch_key)
        conf = conf_stats.get(conf_key)
        abs_elo = _safe_abs(getattr(row, "elo_diff_feat", 0.0))
        abs_form = _safe_abs(getattr(row, "form_diff_feat", 0.0))

        if "archetype_prior" in actions and arch is not None:
            if arch.high5 > 0.43 and tg != og and (tg + og) <= 3 and abs(tg - og) >= 2 and abs_elo > 230:
                tg, og = _winner_plus(tg, og, 1, max_goal)
                action_counts["archetype_prior"] += int((tg, og) != original)
            elif arch.high5 < 0.13 and (tg + og) >= 4:
                tg, og = _cap_to_medium(tg, og)
                action_counts["archetype_prior"] += int((tg, og) != original)

        if "temperature_proxy" in actions and arch is not None:
            before = (tg, og)
            compact = row.archetype in {
                "men_compact_draw",
                "men_low_score_qualifier",
                "men_friendly_low",
                "women_elite_compact",
                "women_friendly_conservative",
            }
            tail = row.archetype in {
                "women_qualifier_blowout",
                "women_qualifier_strong",
                "women_uefa_qualifier_era",
                "men_regional_volatile",
                "men_concacaf_ofc_high_tail",
            }
            if compact and arch.high5 < 0.18 and (tg + og) >= 5:
                tg, og = _cap_to_medium(tg, og)
            elif tail and arch.high5 > 0.38 and tg != og and (tg + og) <= 3 and abs(tg - og) >= 2 and abs_elo > 260:
                tg, og = _winner_plus(tg, og, 1, max_goal)
            action_counts["temperature_proxy"] += int((tg, og) != before)

        if "outcome_preserving_rerank" in actions:
            before = (tg, og)
            outcome = int(np.sign(tg - og))
            mode = mode_stats.get((row.gender, row.archetype, outcome))
            if mode is not None and mode[2] >= 100:
                mtg, mog, _ = mode
                if outcome == 0 and (tg + og) >= 4 and row.archetype in {"men_compact_draw", "men_low_score_qualifier"}:
                    tg, og = mtg, mog
                elif outcome != 0 and row.archetype in {"men_compact_draw", "men_low_score_qualifier"} and (tg + og) >= 4:
                    tg, og = mtg, mog
                elif outcome != 0 and row.archetype == "women_qualifier_blowout" and abs(tg - og) >= 3 and (tg + og) <= 4:
                    tg, og = _winner_plus(tg, og, 1, max_goal)
            action_counts["outcome_preserving_rerank"] += int((tg, og) != before)

        if "draw_calibration" in actions:
            before = (tg, og)
            if (
                row.gender == "M"
                and row.archetype == "men_compact_draw"
                and abs(tg - og) == 1
                and abs_elo < 45
                and abs_form < 0.8
            ):
                tg, og = _to_draw(tg, og)
            action_counts["draw_calibration"] += int((tg, og) != before)

        if "women_tail_era_shrink" in actions and row.gender == "W" and arch is not None:
            before = (tg, og)
            is_tail_arch = row.archetype in {
                "women_qualifier_blowout",
                "women_qualifier_strong",
                "women_uefa_qualifier_era",
            }
            if is_tail_arch and tg != og and arch.high5 > 0.30:
                threshold_elo = 300 if row.era == "2023-2026" else 220
                threshold_form = 2.0 if row.era == "2023-2026" else 1.5
                strong_mismatch = abs_elo > threshold_elo or abs_form > threshold_form
                if strong_mismatch and (tg + og) <= 3 and abs(tg - og) >= 2:
                    tg, og = _winner_plus(tg, og, 1, max_goal)
                if (
                    row.archetype == "women_qualifier_blowout"
                    and strong_mismatch
                    and abs(tg - og) >= 3
                    and (tg + og) <= 4
                ):
                    tg, og = _winner_plus(tg, og, 1, max_goal)
            if row.archetype in {"women_friendly_conservative", "women_elite_compact"} and row.era == "2023-2026" and (tg + og) >= 5:
                tg, og = _cap_to_medium(tg, og)
            action_counts["women_tail_era_shrink"] += int((tg, og) != before)

        if "conf_pair_smoothed" in actions and conf is not None and conf.n >= 300:
            before = (tg, og)
            if conf.draw > 0.32 and row.gender == "M" and abs(tg - og) == 1 and abs_elo < 35:
                tg, og = _to_draw(tg, og)
            elif conf.high5 > 0.45 and row.gender == "W" and tg != og and (tg + og) <= 3 and abs(tg - og) >= 2 and abs_elo > 300:
                tg, og = _winner_plus(tg, og, 1, max_goal)
            elif conf.high5 < 0.08 and (tg + og) >= 5:
                tg, og = _cap_to_medium(tg, og)
            action_counts["conf_pair_smoothed"] += int((tg, og) != before)

        if "small_expert_blend" in actions:
            before = (tg, og)
            if arch is not None and arch.high5 > 0.43 and tg != og and (tg + og) <= 3 and abs(tg - og) >= 2 and abs_elo > 230:
                tg, og = _winner_plus(tg, og, 1, max_goal)
            if arch is not None and arch.high5 < 0.13 and (tg + og) >= 4:
                tg, og = _cap_to_medium(tg, og)
            if row.gender == "M" and row.archetype == "men_compact_draw" and abs(tg - og) == 1 and abs_elo < 50:
                tg, og = _to_draw(tg, og)
            action_counts["small_expert_blend"] += int((tg, og) != before)

        pred_t[pos], pred_o[pos] = _clip_score(tg, og, max_goal)

    submission = pd.DataFrame(
        {"Id": frame["Id"].to_numpy(), "team_goals": pred_t, "opp_goals": pred_o}
    )
    return submission, action_counts


def run_low_leakage_pipeline(config: LowLeakageConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    train, test, gt, frame = load_low_leakage_frame(config.base_path)
    stats = build_train_stats(train)

    working = frame[["Id", "base_tg", "base_og"]].rename(
        columns={"base_tg": "team_goals", "base_og": "opp_goals"}
    )
    pair_counts: dict[str, Any] = {}
    if config.pair_policy:
        before_bad = count_pair_inconsistency(working, test)
        working = apply_pair_policy(working, test, config.pair_policy)
        after_bad = count_pair_inconsistency(working, test)
        pair_counts = {
            "policy": config.pair_policy,
            "pair_inconsistent_before_policy": before_bad,
            "pair_inconsistent_after_policy": after_bad,
        }

    action_frame = frame.drop(columns=["base_tg", "base_og"]).merge(
        working.rename(columns={"team_goals": "base_tg", "opp_goals": "base_og"}),
        on="Id",
        how="left",
        validate="one_to_one",
    )
    action_frame = action_frame.rename(columns={"base_tg": "team_goals", "base_og": "opp_goals"})
    submission, action_counts = apply_actions(action_frame, stats, config.actions, config.max_goal)
    submission = test[["Id"]].merge(submission, on="Id", how="left", validate="one_to_one")

    output_path = DATA_DIR / config.output_name
    submission.to_csv(output_path, index=False)
    metrics = evaluate_submission(submission, gt)
    audit = {
        "strategy": config.name,
        "strategy_id": config.strategy_id,
        "output_path": str(output_path.relative_to(ROOT)),
        "base_path": str(config.base_path.relative_to(ROOT)) if config.base_path.is_relative_to(ROOT) else str(config.base_path),
        "leakage_policy": "train-derived stats only; test_ground_truth used only in evaluation",
        "actions": config.actions,
        "action_counts": action_counts,
        "pair_policy": pair_counts,
        "metrics": metrics,
        "pair_inconsistent_matches": count_pair_inconsistency(submission, test),
        "rule_count": len(config.actions) + (1 if config.pair_policy else 0),
    }
    audit_path = DATA_DIR / config.output_name.replace(".csv", "_audit.json")
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return submission, audit


def print_low_leakage_audit(audit: dict[str, Any]) -> None:
    m = audit["metrics"]
    print(f"Strategy: {audit['strategy']}")
    print(f"Output: {audit['output_path']}")
    print(f"Exact accuracy: {m['exact_accuracy'] * 100:.4f}%")
    print(f"Outcome accuracy: {m['outcome_accuracy'] * 100:.4f}%")
    print(f"AW-MAE: {m['awmae']:.6f}")
    print(f"Pair inconsistent matches: {audit['pair_inconsistent_matches']}")
    print(f"Rule count: {audit['rule_count']}")
    print(f"Action counts: {audit['action_counts']}")
