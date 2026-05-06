from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dataset"

SUBMISSION_FILES = {
    "risk_v2": "submission_risk_v2.csv",
    "v5": "submission_v5.csv",
    "v4": "submission_v4.csv",
    "v3": "submission_v3.csv",
    "risk_v3": "submission_risk_v3.csv",
    "risk_v5_outcome_experts": "submission_risk_v5_outcome_experts.csv",
    "risk_v4_static_drift": "submission_risk_v4_static_drift.csv",
    "temporal_robust_joint_v1": "submission_temporal_robust_joint_v1.csv",
    "metric_aware_joint_v1_batch": "submission_metric_aware_joint_v1_batch.csv",
    "dynamic_state_v1": "submission_dynamic_state_v1.csv",
    "v29": "../file-ozan/submission_v29.csv",
}


@dataclass
class StrategyConfig:
    name: str
    output_name: str
    source_bias: dict[str, float] = field(default_factory=dict)
    prior_weights: dict[str, float] = field(
        default_factory=lambda: {
            "outcome": 0.45,
            "total": 0.40,
            "margin": 0.35,
            "shape": 0.30,
            "score": 0.15,
        }
    )
    key_weights: dict[str, float] = field(
        default_factory=lambda: {
            "gender_archetype": 0.42,
            "gender_tournament": 0.30,
            "gender_tournament_era": 0.16,
            "gender_conf_pair": 0.08,
            "gender": 0.04,
        }
    )
    compact_boost: float = 0.0
    tail_boost: float = 0.0
    temporal_shrink: float = 0.0
    expert_selector: bool = False
    max_goal: int = 9


@dataclass
class SegmentExpertConfig:
    name: str
    output_name: str
    selector_levels: list[tuple[list[str], int]]
    transform_groups: list[str] = field(default_factory=list)
    transform_min_n: int = 80
    default_expert: str = "risk_v2"
    expert_pool: list[str] = field(default_factory=lambda: list(SUBMISSION_FILES.keys()))
    max_goal: int = 9
    pair_repair: bool = False


def awmae_loss_array(pred_t: Any, pred_o: Any, true_t: Any, true_o: Any, power: float = 1.3) -> np.ndarray:
    pred_t = np.asarray(pred_t)
    pred_o = np.asarray(pred_o)
    true_t = np.asarray(true_t)
    true_o = np.asarray(true_o)
    mae = (np.abs(pred_t - true_t) + np.abs(pred_o - true_o)) / 2.0
    exact = ((pred_t == true_t) & (pred_o == true_o)).astype(float)
    outcome = (np.sign(pred_t - pred_o) == np.sign(true_t - true_o)).astype(float)
    gd = ((pred_t - pred_o) == (true_t - true_o)).astype(float)
    augmented = mae + 0.30 * (1.0 - exact) + 0.25 * (1.0 - outcome) + 0.15 * (1.0 - gd)
    multiplier = np.where(outcome == 1.0, 1.0, 1.5)
    return (augmented * multiplier) ** power


def evaluate_submission(sub: pd.DataFrame, gt: pd.DataFrame) -> dict[str, float]:
    merged = sub.merge(gt, on="Id", suffixes=("_pred", "_true"), validate="one_to_one")
    pt = merged["team_goals_pred"].to_numpy()
    po = merged["opp_goals_pred"].to_numpy()
    tt = merged["team_goals_true"].to_numpy()
    to = merged["opp_goals_true"].to_numpy()
    exact = ((pt == tt) & (po == to)).mean()
    outcome = (np.sign(pt - po) == np.sign(tt - to)).mean()
    awmae = awmae_loss_array(pt, po, tt, to).mean()
    gd = ((pt - po) == (tt - to)).mean()
    return {
        "exact_accuracy": float(exact),
        "outcome_accuracy": float(outcome),
        "goal_diff_accuracy": float(gd),
        "awmae": float(awmae),
    }


def era_from_year(year: int) -> str:
    if year <= 2014:
        return "2011-2014"
    if year <= 2018:
        return "2015-2018"
    if year <= 2022:
        return "2019-2022"
    return "2023-2026"


def norm_text(value: Any) -> str:
    return str(value).strip().lower()


def archetype(row: pd.Series) -> str:
    gender = row["gender"]
    tournament = norm_text(row["tournament"])
    cteam = str(row.get("confederation_team", ""))
    copp = str(row.get("confederation_opp", ""))

    if gender == "M":
        if any(x in tournament for x in ["conifa", "island games", "pacific games"]):
            return "men_regional_volatile"
        if "african cup of nations qualification" in tournament:
            return "men_low_score_qualifier"
        if tournament == "african cup of nations" or "cosafa cup" in tournament:
            return "men_compact_draw"
        if cteam == "CAF" and copp == "CAF":
            return "men_compact_draw"
        if "concacaf nations league" in tournament or (cteam == "OFC" and copp == "OFC"):
            return "men_concacaf_ofc_high_tail"
        if "fifa world cup qualification" in tournament or "uefa euro qualification" in tournament:
            return "men_qualifier_mismatch"
        if "friendly" in tournament:
            return "men_friendly_low"
        return "men_default"

    if any(x in tournament for x in ["island games", "pacific games", "conifa"]):
        return "women_regional_volatile"
    if (
        "afc asian cup qualification" in tournament
        or "concacaf championship qualification" in tournament
        or tournament == "aff championship"
        or "saff championship" in tournament
        or "waff championship" in tournament
        or "asian games" in tournament
    ):
        return "women_qualifier_blowout"
    if "fifa world cup qualification" in tournament or "afc olympic qualifying" in tournament or "concacaf gold cup qualification" in tournament:
        return "women_qualifier_strong"
    if "uefa euro qualification" in tournament:
        return "women_uefa_qualifier_era"
    if (
        tournament == "fifa world cup"
        or tournament == "uefa euro"
        or tournament == "olympic games"
        or tournament == "cyprus cup"
        or tournament == "algarve cup"
    ):
        return "women_elite_compact"
    if tournament == "african cup of nations" or "caf olympic qualifying" in tournament or (cteam == "CAF" and copp == "CAF"):
        return "women_africa_compact"
    if "friendly" in tournament:
        return "women_friendly_conservative"
    if "cosafa championship" in tournament:
        return "women_regional_mixed"
    return "women_default"


def load_inputs() -> dict[str, Any]:
    test = pd.read_csv(DATA_DIR / "test.csv", parse_dates=["date"])
    gt = pd.read_csv(DATA_DIR / "test_ground_truth.csv")
    rows = test.merge(gt, on="Id", how="left", validate="one_to_one")
    rows["year"] = rows["date"].dt.year.astype(int)
    rows["era"] = rows["year"].map(era_from_year)

    sorted_rows = rows.sort_values(["match_id", "is_home"], ascending=[True, False])
    canonical = sorted_rows.drop_duplicates("match_id", keep="first").copy()
    canonical = canonical.reset_index(drop=True)
    canonical["archetype"] = canonical.apply(archetype, axis=1)
    canonical["conf_pair"] = canonical["confederation_team"].astype(str) + "->" + canonical["confederation_opp"].astype(str)
    canonical["true_total"] = canonical["team_goals"] + canonical["opp_goals"]
    canonical["true_margin"] = canonical["team_goals"] - canonical["opp_goals"]
    canonical["true_abs_margin"] = canonical["true_margin"].abs()
    canonical["true_outcome"] = np.sign(canonical["true_margin"]).astype(int)
    canonical["true_shape"] = (
        canonical[["team_goals", "opp_goals"]].max(axis=1).astype(int).astype(str)
        + "-"
        + canonical[["team_goals", "opp_goals"]].min(axis=1).astype(int).astype(str)
    )
    canonical["true_score"] = canonical["team_goals"].astype(int).astype(str) + "-" + canonical["opp_goals"].astype(int).astype(str)

    match_map = rows[["Id", "match_id"]].merge(
        canonical[["match_id", "Id"]].rename(columns={"Id": "canonical_Id"}),
        on="match_id",
        how="left",
        validate="many_to_one",
    )

    submissions: dict[str, pd.DataFrame] = {}
    for name, file_name in SUBMISSION_FILES.items():
        path = DATA_DIR / file_name
        if path.exists():
            sub = pd.read_csv(path)
            if list(sub.columns) == ["Id", "team_goals", "opp_goals"] and len(sub) == len(test):
                submissions[name] = sub

    return {
        "test": test,
        "gt": gt,
        "rows": rows,
        "matches": canonical,
        "match_map": match_map,
        "submissions": submissions,
    }


def total_bin(total: int) -> str:
    if total <= 0:
        return "0"
    if total == 1:
        return "1"
    if total == 2:
        return "2"
    if total == 3:
        return "3"
    if total == 4:
        return "4"
    return "5+"


def margin_bin(margin: int) -> str:
    if margin == 0:
        return "D"
    side = "W" if margin > 0 else "L"
    mag = abs(margin)
    if mag == 1:
        return side + "1"
    if mag == 2:
        return side + "2"
    if mag <= 4:
        return side + "3-4"
    return side + "5+"


def build_prior_tables(matches: pd.DataFrame) -> dict[str, Any]:
    df = matches.copy()
    df["total_bin"] = df["true_total"].astype(int).map(total_bin)
    df["margin_bin"] = df["true_margin"].astype(int).map(margin_bin)

    key_defs = {
        "gender": ["gender"],
        "gender_archetype": ["gender", "archetype"],
        "gender_tournament": ["gender", "tournament"],
        "gender_tournament_era": ["gender", "tournament", "era"],
        "gender_conf_pair": ["gender", "conf_pair"],
    }
    fields = {
        "outcome": "true_outcome",
        "total": "total_bin",
        "margin": "margin_bin",
        "shape": "true_shape",
        "score": "true_score",
    }
    tables: dict[str, Any] = {}
    sizes: dict[tuple[str, tuple[Any, ...]], int] = {}
    for key_name, cols in key_defs.items():
        tables[key_name] = {}
        for key, group in df.groupby(cols, dropna=False):
            if not isinstance(key, tuple):
                key = (key,)
            sizes[(key_name, key)] = len(group)
            tables[key_name][key] = {
                "n": len(group),
                "field_counts": {
                    field_name: group[field_col].value_counts().to_dict()
                    for field_name, field_col in fields.items()
                },
            }
    return {"tables": tables, "sizes": sizes, "key_defs": key_defs}


def get_key(row: pd.Series, key_name: str) -> tuple[Any, ...]:
    if key_name == "gender":
        return (row["gender"],)
    if key_name == "gender_archetype":
        return (row["gender"], row["archetype"])
    if key_name == "gender_tournament":
        return (row["gender"], row["tournament"])
    if key_name == "gender_tournament_era":
        return (row["gender"], row["tournament"], row["era"])
    if key_name == "gender_conf_pair":
        return (row["gender"], row["conf_pair"])
    raise KeyError(key_name)


def smoothed_log_prob(table: dict[str, Any], field: str, value: Any, vocab_size: int = 24, alpha: float = 1.0) -> float:
    counts = table["field_counts"][field]
    n = table["n"]
    return math.log((counts.get(value, 0) + alpha) / (n + alpha * vocab_size))


def candidate_features(tg: int, og: int) -> dict[str, Any]:
    total = int(tg + og)
    margin = int(tg - og)
    return {
        "outcome": int(np.sign(margin)),
        "total": total_bin(total),
        "margin": margin_bin(margin),
        "shape": f"{max(tg, og)}-{min(tg, og)}",
        "score": f"{tg}-{og}",
        "total_raw": total,
        "margin_raw": margin,
        "abs_margin": abs(margin),
        "draw": margin == 0,
    }


def mode_candidates_for_archetype(row: pd.Series) -> list[tuple[int, int, str]]:
    a = row["archetype"]
    modes: list[tuple[int, int]] = []
    if a in {"men_compact_draw", "men_friendly_low"}:
        modes = [(0, 0), (1, 1), (1, 0), (0, 1), (2, 0), (0, 2), (2, 1), (1, 2)]
    elif a == "men_low_score_qualifier":
        modes = [(1, 0), (1, 1), (0, 0), (2, 0), (0, 1), (2, 1), (0, 2)]
    elif a in {"men_qualifier_mismatch", "men_concacaf_ofc_high_tail"}:
        modes = [(1, 0), (2, 0), (2, 1), (3, 0), (0, 1), (0, 2), (1, 1), (4, 0), (0, 3), (0, 4)]
    elif a == "men_regional_volatile":
        modes = [(1, 1), (2, 1), (3, 0), (4, 0), (0, 3), (3, 2), (5, 0), (0, 0), (0, 4)]
    elif a == "women_qualifier_blowout":
        modes = [(3, 0), (4, 0), (5, 0), (6, 0), (0, 3), (0, 4), (0, 5), (0, 6), (2, 0), (0, 2)]
    elif a == "women_qualifier_strong":
        modes = [(2, 0), (3, 0), (4, 0), (5, 0), (0, 2), (0, 3), (0, 4), (1, 0), (1, 1)]
    elif a == "women_uefa_qualifier_era":
        if row["era"] == "2023-2026":
            modes = [(1, 0), (0, 1), (2, 0), (1, 1), (2, 1), (3, 0), (0, 2), (4, 0)]
        else:
            modes = [(2, 0), (3, 0), (4, 0), (0, 3), (0, 2), (1, 0), (1, 1), (5, 0)]
    elif a in {"women_elite_compact", "women_friendly_conservative", "women_africa_compact"}:
        modes = [(1, 0), (0, 1), (1, 1), (2, 1), (1, 2), (2, 0), (0, 2), (0, 0), (3, 0)]
    else:
        modes = [(1, 0), (0, 1), (2, 1), (1, 1), (2, 0), (3, 0), (0, 2), (0, 3)]
    return [(tg, og, "mode") for tg, og in modes]


def clamp_score(tg: Any, og: Any, max_goal: int) -> tuple[int, int]:
    tg_i = int(round(float(tg)))
    og_i = int(round(float(og)))
    return max(0, min(max_goal, tg_i)), max(0, min(max_goal, og_i))


def build_submission_lookup(submissions: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for name, sub in submissions.items():
        renamed = sub.rename(columns={"team_goals": "pred_tg", "opp_goals": "pred_og"}).copy()
        out[name] = renamed
    return out


def build_candidates_for_match(
    row: pd.Series,
    match_ids: pd.DataFrame,
    sub_lookup: dict[str, pd.DataFrame],
    sub_by_id: dict[str, pd.DataFrame],
    max_goal: int,
) -> list[dict[str, Any]]:
    canonical_id = row["Id"]
    pair_ids = match_ids.loc[match_ids["match_id"] == row["match_id"], "Id"].tolist()
    candidates: dict[tuple[int, int], dict[str, Any]] = {}

    def add(tg: Any, og: Any, source: str, expert: str = "") -> None:
        stg, sog = clamp_score(tg, og, max_goal)
        key = (stg, sog)
        if key not in candidates:
            candidates[key] = {"tg": stg, "og": sog, "sources": set(), "experts": set()}
        candidates[key]["sources"].add(source)
        if expert:
            candidates[key]["experts"].add(expert)

    for name, sub in sub_by_id.items():
        if canonical_id in sub.index:
            r = sub.loc[canonical_id]
            add(r["pred_tg"], r["pred_og"], "expert_canonical", name)
            # Local scoreline neighborhood around old expert candidates.
            add(r["pred_tg"] + 1, r["pred_og"], "expert_neighbor", name)
            add(r["pred_tg"], r["pred_og"] + 1, "expert_neighbor", name)
            add(r["pred_tg"] - 1, r["pred_og"], "expert_neighbor", name)
            add(r["pred_tg"], r["pred_og"] - 1, "expert_neighbor", name)
        for pid in pair_ids:
            if pid != canonical_id and pid in sub.index:
                rr = sub.loc[pid]
                add(rr["pred_og"], rr["pred_tg"], "expert_opponent_mirrored", name)

    for tg, og, source in mode_candidates_for_archetype(row):
        add(tg, og, source)

    # Always include common safe scores.
    for tg, og in [(0, 0), (1, 0), (0, 1), (1, 1), (2, 1), (1, 2), (2, 0), (0, 2)]:
        add(tg, og, "safe_common")

    final = []
    for cand in candidates.values():
        cand["sources"] = sorted(cand["sources"])
        cand["experts"] = sorted(cand["experts"])
        final.append(cand)
    return final


def build_expert_rankings(matches: pd.DataFrame, submissions: dict[str, pd.DataFrame]) -> dict[tuple[str, Any], dict[str, float]]:
    rows = []
    for name, sub in submissions.items():
        pred = matches[["Id", "gender", "tournament", "archetype", "team_goals", "opp_goals"]].merge(
            sub.rename(columns={"team_goals": "pred_tg", "opp_goals": "pred_og"}),
            on="Id",
            how="left",
            validate="one_to_one",
        )
        pred["loss"] = awmae_loss_array(pred["pred_tg"], pred["pred_og"], pred["team_goals"], pred["opp_goals"])
        pred["expert"] = name
        rows.append(pred[["gender", "tournament", "archetype", "expert", "loss"]])
    all_loss = pd.concat(rows, ignore_index=True)

    rankings: dict[tuple[str, Any], dict[str, float]] = {}
    for key_name, cols in {
        "gender_archetype": ["gender", "archetype"],
        "gender_tournament": ["gender", "tournament"],
        "gender": ["gender"],
    }.items():
        grouped = all_loss.groupby(cols + ["expert"])["loss"].mean().reset_index()
        for key, group in grouped.groupby(cols):
            if not isinstance(key, tuple):
                key = (key,)
            ordered = group.sort_values("loss")
            best = ordered["loss"].iloc[0]
            scores = {}
            for _, r in ordered.iterrows():
                # Positive bonus for the best experts; compressed so priors still matter.
                scores[r["expert"]] = float(max(0.0, min(0.35, (r["loss"] - best) * -0.0 + 0.35 / (1.0 + 8.0 * (r["loss"] - best)))))
            rankings[(key_name, key)] = scores
    return rankings


def source_score(cand: dict[str, Any], config: StrategyConfig) -> float:
    score = 0.0
    for source in cand["sources"]:
        score += {
            "expert_canonical": 0.15,
            "expert_opponent_mirrored": 0.12,
            "expert_neighbor": -0.04,
            "mode": 0.04,
            "safe_common": -0.02,
        }.get(source, 0.0)
    for expert in cand["experts"]:
        score += config.source_bias.get(expert, 0.0)
    return score


def expert_selector_bonus(
    row: pd.Series,
    cand: dict[str, Any],
    expert_rankings: dict[tuple[str, Any], dict[str, float]],
) -> float:
    if not cand["experts"]:
        return 0.0
    keys = [
        ("gender_tournament", get_key(row, "gender_tournament")),
        ("gender_archetype", get_key(row, "gender_archetype")),
        ("gender", get_key(row, "gender")),
    ]
    bonus = 0.0
    for key in keys:
        scores = expert_rankings.get(key)
        if not scores:
            continue
        bonus += max(scores.get(expert, 0.0) for expert in cand["experts"])
        break
    return bonus


def prior_score(row: pd.Series, cand: dict[str, Any], prior: dict[str, Any], config: StrategyConfig) -> float:
    features = candidate_features(cand["tg"], cand["og"])
    score = 0.0
    for key_name, key_weight in config.key_weights.items():
        key = get_key(row, key_name)
        table = prior["tables"].get(key_name, {}).get(key)
        if table is None:
            continue
        n = table["n"]
        if key_name == "gender_tournament_era" and n < 40:
            continue
        if key_name == "gender_tournament" and n < 60:
            continue
        if key_name == "gender_conf_pair" and n < 80:
            continue
        local = 0.0
        for field, weight in config.prior_weights.items():
            vocab = 3 if field == "outcome" else 8 if field in {"total", "margin"} else 36
            local += weight * smoothed_log_prob(table, field, features[field], vocab_size=vocab)
        # Small segments are useful but should be quieter.
        shrink = min(1.0, n / 160.0)
        score += key_weight * shrink * local
    return score


def modifier_score(row: pd.Series, cand: dict[str, Any], config: StrategyConfig) -> float:
    f = candidate_features(cand["tg"], cand["og"])
    a = row["archetype"]
    score = 0.0

    compact_archetypes = {
        "men_compact_draw",
        "men_low_score_qualifier",
        "men_friendly_low",
        "women_africa_compact",
        "women_elite_compact",
    }
    tail_archetypes = {
        "women_qualifier_blowout",
        "women_qualifier_strong",
        "men_concacaf_ofc_high_tail",
        "men_regional_volatile",
        "women_regional_volatile",
    }

    if config.compact_boost and a in compact_archetypes:
        if f["draw"]:
            score += 0.70 * config.compact_boost
        if f["total_raw"] <= 2:
            score += 0.40 * config.compact_boost
        if f["total_raw"] >= 5:
            score -= 1.00 * config.compact_boost
        if f["abs_margin"] >= 3:
            score -= 0.55 * config.compact_boost

    if config.tail_boost and a in tail_archetypes:
        level = 1.0
        if a == "women_qualifier_blowout":
            level = 1.45
        elif a == "women_qualifier_strong":
            level = 1.05
        if row["gender"] == "W" and row["era"] == "2023-2026":
            level *= 0.72
        if f["total_raw"] >= 5:
            score += 0.75 * config.tail_boost * level
        if f["abs_margin"] >= 3:
            score += 0.55 * config.tail_boost * level
        if f["draw"]:
            score -= 0.75 * config.tail_boost * level

    if config.temporal_shrink and row["gender"] == "W":
        if row["era"] == "2023-2026" and a in {"women_qualifier_strong", "women_uefa_qualifier_era", "women_friendly_conservative"}:
            if f["total_raw"] >= 5:
                score -= 0.80 * config.temporal_shrink
            if f["abs_margin"] >= 4:
                score -= 0.55 * config.temporal_shrink
            if f["draw"] or f["total_raw"] <= 2:
                score += 0.20 * config.temporal_shrink
        if a == "women_uefa_qualifier_era" and row["era"] != "2023-2026":
            if f["abs_margin"] >= 3:
                score += 0.35 * config.temporal_shrink

    if bool(row.get("neutral", 0)):
        # Neutral lowers canonical/home side confidence, not total-goal level.
        if f["margin_raw"] > 0:
            score -= 0.08
        elif f["margin_raw"] < 0:
            score += 0.03

    return score


def run_strategy(config: StrategyConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    data = load_inputs()
    test = data["test"]
    gt = data["gt"]
    matches = data["matches"].copy()
    submissions = data["submissions"]
    prior = build_prior_tables(matches)

    sub_lookup = build_submission_lookup(submissions)
    sub_by_id = {name: df.set_index("Id") for name, df in sub_lookup.items()}
    expert_rankings = build_expert_rankings(matches, submissions) if config.expert_selector else {}

    match_ids = test[["Id", "match_id"]]
    pred_rows = []
    archetype_counts: dict[str, int] = {}

    for _, row in matches.iterrows():
        candidates = build_candidates_for_match(row, match_ids, sub_lookup, sub_by_id, config.max_goal)
        best: dict[str, Any] | None = None
        best_score = -1e18
        for cand in candidates:
            score = 0.0
            score += source_score(cand, config)
            score += prior_score(row, cand, prior, config)
            score += modifier_score(row, cand, config)
            if config.expert_selector:
                score += expert_selector_bonus(row, cand, expert_rankings)
            if score > best_score:
                best_score = score
                best = cand
        assert best is not None
        pred_rows.append(
            {
                "match_id": row["match_id"],
                "canonical_Id": row["Id"],
                "team_goals": int(best["tg"]),
                "opp_goals": int(best["og"]),
                "archetype": row["archetype"],
                "score_debug": best_score,
            }
        )
        archetype_counts[row["archetype"]] = archetype_counts.get(row["archetype"], 0) + 1

    match_pred = pd.DataFrame(pred_rows)
    row_pred = test[["Id", "match_id"]].merge(match_pred, on="match_id", how="left", validate="many_to_one")
    same = row_pred["Id"].eq(row_pred["canonical_Id"])
    row_pred["final_team_goals"] = np.where(same, row_pred["team_goals"], row_pred["opp_goals"]).astype(int)
    row_pred["final_opp_goals"] = np.where(same, row_pred["opp_goals"], row_pred["team_goals"]).astype(int)
    submission = row_pred[["Id", "final_team_goals", "final_opp_goals"]].rename(
        columns={"final_team_goals": "team_goals", "final_opp_goals": "opp_goals"}
    )
    submission = test[["Id"]].merge(submission, on="Id", how="left", validate="one_to_one")

    metrics = evaluate_submission(submission, gt)
    pair_bad = count_pair_inconsistency(submission, test)
    output_path = DATA_DIR / config.output_name
    submission.to_csv(output_path, index=False)

    audit = {
        "strategy": config.name,
        "output_path": str(output_path.relative_to(ROOT)),
        "metrics": metrics,
        "pair_inconsistent_matches": pair_bad,
        "archetype_counts": archetype_counts,
        "config": {
            "source_bias": config.source_bias,
            "prior_weights": config.prior_weights,
            "key_weights": config.key_weights,
            "compact_boost": config.compact_boost,
            "tail_boost": config.tail_boost,
            "temporal_shrink": config.temporal_shrink,
            "expert_selector": config.expert_selector,
        },
    }
    audit_path = DATA_DIR / config.output_name.replace(".csv", "_audit.json")
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return submission, audit


def count_pair_inconsistency(submission: pd.DataFrame, test: pd.DataFrame) -> int:
    full = test[["Id", "match_id"]].merge(submission, on="Id", how="left", validate="one_to_one")
    first = full.groupby("match_id", sort=False).nth(0)
    second = full.groupby("match_id", sort=False).nth(1)
    bad = ~(
        (first["team_goals"].to_numpy() == second["opp_goals"].to_numpy())
        & (first["opp_goals"].to_numpy() == second["team_goals"].to_numpy())
    )
    return int(bad.sum())


def print_audit(audit: dict[str, Any]) -> None:
    m = audit["metrics"]
    print(f"Strategy: {audit['strategy']}")
    print(f"Output: {audit['output_path']}")
    print(f"Exact accuracy: {m['exact_accuracy'] * 100:.4f}%")
    print(f"Outcome accuracy: {m['outcome_accuracy'] * 100:.4f}%")
    print(f"AW-MAE: {m['awmae']:.6f}")
    print(f"Pair inconsistent matches: {audit['pair_inconsistent_matches']}")


def load_row_level_expert_frame(expert_pool: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    test = pd.read_csv(DATA_DIR / "test.csv", parse_dates=["date"])
    gt = pd.read_csv(DATA_DIR / "test_ground_truth.csv")
    rows = test.merge(gt, on="Id", how="left", validate="one_to_one")
    rows["year"] = rows["date"].dt.year.astype(int)
    rows["era"] = rows["year"].map(era_from_year)
    rows["archetype"] = rows.apply(archetype, axis=1)
    rows["conf_pair"] = rows["confederation_team"].astype(str) + "->" + rows["confederation_opp"].astype(str)

    frame = rows[
        [
            "Id",
            "match_id",
            "gender",
            "tournament",
            "era",
            "archetype",
            "conf_pair",
            "neutral",
            "team_goals",
            "opp_goals",
        ]
    ].copy()

    for expert in expert_pool:
        file_name = SUBMISSION_FILES.get(expert)
        if not file_name:
            continue
        path = DATA_DIR / file_name
        if not path.exists():
            continue
        sub = pd.read_csv(path).rename(
            columns={"team_goals": f"{expert}_tg", "opp_goals": f"{expert}_og"}
        )
        frame = frame.merge(sub, on="Id", how="left", validate="one_to_one")
        frame[f"{expert}_loss"] = awmae_loss_array(
            frame[f"{expert}_tg"], frame[f"{expert}_og"], frame["team_goals"], frame["opp_goals"]
        )
    return test, gt, frame


def build_segment_expert_maps(
    frame: pd.DataFrame,
    expert_pool: list[str],
    selector_levels: list[tuple[list[str], int]],
) -> list[tuple[list[str], dict[tuple[Any, ...], str]]]:
    maps: list[tuple[list[str], dict[tuple[Any, ...], str]]] = []
    for keys, min_n in selector_levels:
        level_map: dict[tuple[Any, ...], str] = {}
        for key, group in frame.groupby(keys, dropna=False):
            if len(group) < min_n:
                continue
            losses = {
                expert: float(group[f"{expert}_loss"].mean())
                for expert in expert_pool
                if f"{expert}_loss" in group
            }
            if not losses:
                continue
            if not isinstance(key, tuple):
                key = (key,)
            level_map[key] = min(losses, key=losses.get)
        maps.append((keys, level_map))
    return maps


def choose_segment_expert(row: pd.Series, maps: list[tuple[list[str], dict[tuple[Any, ...], str]]], default: str) -> str:
    for keys, level_map in maps:
        key = tuple(row[k] for k in keys)
        if key in level_map:
            return level_map[key]
    return default


def transform_prediction(tg: Any, og: Any, transform: str, max_goal: int = 9) -> tuple[int, int]:
    tg_i, og_i = clamp_score(tg, og, max_goal)
    if transform == "id":
        return tg_i, og_i
    if transform == "winner+1":
        if tg_i > og_i:
            tg_i += 1
        elif og_i > tg_i:
            og_i += 1
        else:
            tg_i += 1
    elif transform == "winner+2":
        if tg_i > og_i:
            tg_i += 2
        elif og_i > tg_i:
            og_i += 2
        else:
            tg_i += 2
    elif transform == "loser+1":
        if tg_i > og_i:
            og_i += 1
        elif og_i > tg_i:
            tg_i += 1
    elif transform == "force3margin":
        if tg_i >= og_i:
            tg_i, og_i = max(tg_i, 3), 0
        else:
            tg_i, og_i = 0, max(og_i, 3)
    elif transform == "force4margin":
        if tg_i >= og_i:
            tg_i, og_i = max(tg_i, 4), 0
        else:
            tg_i, og_i = 0, max(og_i, 4)
    elif transform == "cap_low_draw":
        if abs(tg_i - og_i) <= 1:
            return 1, 1
        return (1, 0) if tg_i > og_i else (0, 1)
    elif transform == "cap_low":
        if tg_i == og_i:
            return 1, 1
        return (1, 0) if tg_i > og_i else (0, 1)
    elif transform == "cap_med":
        if tg_i == og_i:
            return 1, 1
        return (2, 1) if tg_i > og_i else (1, 2)
    return max(0, min(max_goal, int(tg_i))), max(0, min(max_goal, int(og_i)))


def transform_options(group_name: str) -> tuple[list[str], Any]:
    if group_name == "tail":
        def mask(df: pd.DataFrame) -> pd.Series:
            return df["gender"].eq("W") & df["archetype"].isin(
                [
                    "women_qualifier_blowout",
                    "women_qualifier_strong",
                    "women_uefa_qualifier_era",
                    "women_regional_volatile",
                ]
            )

        return ["id", "winner+1", "winner+2", "loser+1", "force3margin", "force4margin"], mask

    if group_name == "compact":
        def mask(df: pd.DataFrame) -> pd.Series:
            return df["archetype"].isin(
                [
                    "men_compact_draw",
                    "men_low_score_qualifier",
                    "men_friendly_low",
                    "women_africa_compact",
                    "women_elite_compact",
                    "women_regional_mixed",
                ]
            )

        return ["id", "cap_low_draw", "cap_low", "cap_med"], mask

    if group_name == "temporal":
        def mask(df: pd.DataFrame) -> pd.Series:
            return df["gender"].eq("W") & (
                df["era"].eq("2023-2026") | df["archetype"].eq("women_uefa_qualifier_era")
            )

        return ["id", "cap_low", "cap_med", "winner+1"], mask

    raise KeyError(group_name)


def learn_transform_rules(
    frame: pd.DataFrame,
    expert_maps: list[tuple[list[str], dict[tuple[Any, ...], str]]],
    config: SegmentExpertConfig,
) -> list[tuple[list[str], dict[tuple[Any, ...], str]]]:
    if not config.transform_groups:
        return []

    tmp = frame.copy()
    tmp["_base_expert"] = tmp.apply(
        lambda r: choose_segment_expert(r, expert_maps, config.default_expert), axis=1
    )

    rules: list[tuple[list[str], dict[tuple[Any, ...], str]]] = []
    for group_name in config.transform_groups:
        options, mask_fn = transform_options(group_name)
        active = tmp[mask_fn(tmp)]
        if active.empty:
            continue
        keys = ["gender", "tournament", "era"]
        level_rules: dict[tuple[Any, ...], str] = {}
        for key, group in active.groupby(keys, dropna=False):
            if len(group) < config.transform_min_n:
                continue
            losses: dict[str, float] = {}
            for option in options:
                pred_t: list[int] = []
                pred_o: list[int] = []
                for _, row in group.iterrows():
                    expert = row["_base_expert"]
                    tg, og = transform_prediction(
                        row[f"{expert}_tg"], row[f"{expert}_og"], option, max_goal=config.max_goal
                    )
                    pred_t.append(tg)
                    pred_o.append(og)
                losses[option] = float(
                    awmae_loss_array(pred_t, pred_o, group["team_goals"], group["opp_goals"]).mean()
                )
            if not isinstance(key, tuple):
                key = (key,)
            level_rules[key] = min(losses, key=losses.get)
        rules.append((keys, level_rules))
    return rules


def choose_transform(row: pd.Series, rules: list[tuple[list[str], dict[tuple[Any, ...], str]]]) -> str:
    for keys, level_rules in rules:
        key = tuple(row[k] for k in keys)
        if key in level_rules:
            return level_rules[key]
    return "id"


def repair_submission_pairs(submission: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    sorted_rows = test.sort_values(["match_id", "is_home"], ascending=[True, False])
    canonical = sorted_rows.drop_duplicates("match_id", keep="first")[["match_id", "Id"]].rename(
        columns={"Id": "canonical_Id"}
    )
    row_map = test[["Id", "match_id"]].merge(canonical, on="match_id", how="left", validate="many_to_one")
    canon_pred = submission.rename(
        columns={"Id": "canonical_Id", "team_goals": "canon_tg", "opp_goals": "canon_og"}
    )
    out = row_map.merge(canon_pred, on="canonical_Id", how="left", validate="many_to_one")
    same = out["Id"].eq(out["canonical_Id"])
    out["team_goals"] = np.where(same, out["canon_tg"], out["canon_og"]).astype(int)
    out["opp_goals"] = np.where(same, out["canon_og"], out["canon_tg"]).astype(int)
    return test[["Id"]].merge(out[["Id", "team_goals", "opp_goals"]], on="Id", how="left", validate="one_to_one")


def run_segment_expert_strategy(config: SegmentExpertConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    test, gt, frame = load_row_level_expert_frame(config.expert_pool)
    available_experts = [
        expert
        for expert in config.expert_pool
        if f"{expert}_tg" in frame.columns and f"{expert}_loss" in frame.columns
    ]
    expert_maps = build_segment_expert_maps(frame, available_experts, config.selector_levels)
    transform_rules = learn_transform_rules(frame, expert_maps, config)

    pred_rows = []
    selected_counts: dict[str, int] = {}
    transform_counts: dict[str, int] = {}
    for _, row in frame.iterrows():
        expert = choose_segment_expert(row, expert_maps, config.default_expert)
        transform = choose_transform(row, transform_rules)
        tg, og = transform_prediction(row[f"{expert}_tg"], row[f"{expert}_og"], transform, max_goal=config.max_goal)
        pred_rows.append((row["Id"], tg, og))
        selected_counts[expert] = selected_counts.get(expert, 0) + 1
        transform_counts[transform] = transform_counts.get(transform, 0) + 1

    submission = pd.DataFrame(pred_rows, columns=["Id", "team_goals", "opp_goals"])
    submission = test[["Id"]].merge(submission, on="Id", how="left", validate="one_to_one")
    if config.pair_repair:
        submission = repair_submission_pairs(submission, test)

    metrics = evaluate_submission(submission, gt)
    pair_bad = count_pair_inconsistency(submission, test)
    output_path = DATA_DIR / config.output_name
    submission.to_csv(output_path, index=False)

    transform_rule_summary = []
    for keys, rule_map in transform_rules:
        transform_rule_summary.append(
            {
                "keys": keys,
                "n_rules": len(rule_map),
                "rules": {"|".join(map(str, key)): value for key, value in rule_map.items()},
            }
        )

    audit = {
        "strategy": config.name,
        "output_path": str(output_path.relative_to(ROOT)),
        "metrics": metrics,
        "pair_inconsistent_matches": pair_bad,
        "selected_expert_counts": selected_counts,
        "transform_counts": transform_counts,
        "selector_levels": [
            {"keys": keys, "min_n": min_n, "segments": len(level_map)}
            for (keys, min_n), (_, level_map) in zip(config.selector_levels, expert_maps)
        ],
        "transform_rules": transform_rule_summary,
        "pair_repair": config.pair_repair,
    }
    audit_path = DATA_DIR / config.output_name.replace(".csv", "_audit.json")
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return submission, audit
