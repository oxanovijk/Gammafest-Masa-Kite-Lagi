"""Batch runner for metric-aware joint v1.

This runner keeps the same leakage boundaries and candidate registry as
model_pipeline_metric_aware_joint_v1, but evaluates joint-score candidates in
batches by reusing fold models and vectorizing score-matrix ERM. It writes
`*_batch*` outputs so the original long-running process can continue safely.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import pickle
import shutil
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import model_pipeline_metric_aware_joint_v1 as m


MODEL_CACHE: dict[tuple[str, float, bool, int, bool], dict[str, Any]] = {}


def configure_paths() -> None:
    m.OUTPUT_SUB = m.DATA_DIR / "submission_metric_aware_joint_v1_batch.csv"
    m.OUTPUT_CONFIG = m.DATA_DIR / "submission_metric_aware_joint_v1_batch_config.json"
    m.OUTPUT_REPORT = m.DATA_DIR / "submission_metric_aware_joint_v1_batch_validation_report.txt"
    m.OUTPUT_AUDIT = m.DATA_DIR / "submission_metric_aware_joint_v1_batch_audit.txt"
    m.OUTPUT_LOCK = m.DATA_DIR / "submission_metric_aware_joint_v1_batch_candidate_lock.json"
    old_cache = m.CACHE_DIR
    m.CACHE_DIR = m.DATA_DIR / "metric_aware_joint_v1_batch_cache"
    m.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for src_dir in [old_cache, m.DATA_DIR / "metric_aware_joint_v1_accel_cache"]:
        if src_dir.exists():
            for src in src_dir.glob("*.pkl"):
                dst = m.CACHE_DIR / src.name
                if not dst.exists():
                    shutil.copy2(src, dst)


def poisson_probs_batch(lam: np.ndarray, max_goals: int) -> np.ndarray:
    lam = np.clip(np.asarray(lam, dtype=float), 0.03, 9.5)
    out = np.zeros((len(lam), max_goals + 1), dtype=float)
    out[:, 0] = np.exp(-lam)
    for k in range(1, max_goals + 1):
        out[:, k] = out[:, k - 1] * lam / k
    out /= np.maximum(out.sum(axis=1, keepdims=True), 1e-12)
    return out


def grid_cache(max_goals: int) -> dict[str, np.ndarray]:
    size = max_goals + 1
    a, b = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    return {
        "a": a,
        "b": b,
        "outcome": np.where(a > b, 2, np.where(a == b, 1, 0)),
        "total": a + b,
        "gd": a - b + max_goals,
        "draw": a == b,
        "tail": (a >= 5) | (b >= 5),
    }


def batch_joint_matrices(heads: dict[str, np.ndarray], prior: np.ndarray, config: m.CandidateConfig) -> np.ndarray:
    max_g = config.max_goals
    g = grid_cache(max_g)
    n = len(heads["outcome"])
    eps = 1e-8
    mat = np.ones((n, max_g + 1, max_g + 1), dtype=float)
    if config.use_poisson and config.alpha != 0:
        pt = poisson_probs_batch(heads["lambda_team"], max_g)
        po = poisson_probs_batch(heads["lambda_opp"], max_g)
        mat *= np.maximum(pt[:, :, None] * po[:, None, :], eps) ** config.alpha
    if config.gamma != 0:
        mat *= np.maximum(heads["outcome"][:, g["outcome"]], eps) ** config.gamma
    if config.delta != 0:
        mat *= np.maximum(heads["total"][:, g["total"]], eps) ** config.delta
    if config.eta != 0:
        mat *= np.maximum(heads["gd"][:, g["gd"]], eps) ** config.eta
    if config.theta != 0:
        mat *= np.maximum(heads["team"][:, g["a"]], eps) ** config.theta
    if config.kappa != 0:
        mat *= np.maximum(heads["opp"][:, g["b"]], eps) ** config.kappa
    if config.use_empirical_prior and config.beta != 0:
        mat *= np.maximum(prior, eps)[None, :, :] ** config.beta
    if config.draw_correction != 1.0:
        mat[:, g["draw"]] *= config.draw_correction
    if config.tail_dampening != 1.0:
        mat[:, g["tail"]] *= config.tail_dampening
    mat = np.maximum(mat, eps)
    mat /= np.maximum(mat.sum(axis=(1, 2), keepdims=True), eps)
    return mat


def expected_losses_batch(joints: np.ndarray, max_goals: int, power: float) -> np.ndarray:
    return np.tensordot(joints, m.loss_tensor(max_goals, power), axes=([1, 2], [2, 3]))


def predict_from_heads(
    valid_feat: pd.DataFrame,
    heads: dict[str, np.ndarray],
    prior: np.ndarray,
    config: m.CandidateConfig,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    ordered = valid_feat.sort_values(["date", "match_id", "Id"]).reset_index(drop=True)
    joints = batch_joint_matrices(heads, prior, config)
    loss15 = expected_losses_batch(joints, config.max_goals, m.PRIMARY_POWER)
    size = config.max_goals + 1
    flat = np.argmin(loss15.reshape(len(ordered), -1), axis=1)
    pred_t = (flat // size).astype(int)
    pred_o = (flat % size).astype(int)
    selected_prob = joints[np.arange(len(ordered)), pred_t, pred_o]
    pair_diag: defaultdict[str, float] = defaultdict(float)

    for _, group in ordered.groupby("match_id", sort=False):
        pair_diag["match_groups"] += 1
        idxs = [int(i) for i in group.index.to_list()]
        if len(group) == 1:
            pair_diag["single_row_matches"] += 1
        elif len(group) == 2:
            pair_diag["two_row_matches"] += 1
        else:
            pair_diag["multirow_matches"] += 1
        if len(group) == 2 and m.reciprocal_pair(group):
            pair_diag["reciprocal_pairs"] += 1
            if config.pair_consistency:
                pair_diag["pair_consistency_applied"] += 1
                combined = loss15[idxs[0]] + loss15[idxs[1]].T
                a, b = np.unravel_index(int(np.argmin(combined)), combined.shape)
                if (pred_t[idxs[0]], pred_o[idxs[0]], pred_t[idxs[1]], pred_o[idxs[1]]) != (int(a), int(b), int(b), int(a)):
                    pair_diag["pair_conflicts_corrected"] += 1
                    pair_diag["pair_correction_abs"] += abs(int(pred_t[idxs[0]]) - int(a)) + abs(int(pred_o[idxs[0]]) - int(b))
                pred_t[idxs[0]], pred_o[idxs[0]] = int(a), int(b)
                pred_t[idxs[1]], pred_o[idxs[1]] = int(b), int(a)
            else:
                pair_diag["pair_consistency_disabled"] += 1
        elif len(group) == 2:
            pair_diag["inconsistent_pairs"] += 1

    out = ordered.copy()
    out["pred_team_goals"] = pred_t
    out["pred_opp_goals"] = pred_o
    ent = float((-(joints * np.log(np.maximum(joints, 1e-12))).sum(axis=(1, 2))).mean())
    matrix_diag = {
        "avg_selected_joint_prob": float(selected_prob.mean()),
        "avg_entropy": ent,
        "rows": int(len(ordered)),
    }
    pair_diag["pair_consistency_pass"] = float(pair_diag.get("inconsistent_pairs", 0) == 0)
    pair_diag["avg_pair_correction_abs"] = pair_diag.get("pair_correction_abs", 0.0) / max(1.0, pair_diag.get("pair_conflicts_corrected", 0.0))
    return out, dict(pair_diag), matrix_diag


def fit_fold_model(train: pd.DataFrame, fold: dict[str, Any], config: m.CandidateConfig, fast_mode: bool) -> dict[str, Any]:
    key = (fold["name"], config.smoothing, config.use_hist_features, config.max_goals, fast_mode)
    if key in MODEL_CACHE:
        return MODEL_CACHE[key]
    fold_train, fold_valid = m.fold_split(train, fold)
    hist_train = m.HistoricalFeatureBuilder(config.smoothing)
    train_feat = hist_train.transform_train_walk_forward(fold_train) if config.use_hist_features else fold_train.copy()
    hist_valid = m.HistoricalFeatureBuilder(config.smoothing).fit(fold_train)
    valid_feat = hist_valid.transform(fold_valid) if config.use_hist_features else fold_valid.copy()
    prior = m.empirical_score_prior(fold_train, config.max_goals)
    model = m.ProbabilisticHeads(config.max_goals, fast_mode).fit(train_feat)
    ordered = valid_feat.sort_values(["date", "match_id", "Id"]).reset_index(drop=True)
    heads = model.predict(ordered)
    data = {
        "fold_train": fold_train,
        "fold_valid": fold_valid,
        "valid_feat": valid_feat,
        "prior": prior,
        "heads": heads,
        "backend": model.backend,
    }
    MODEL_CACHE[key] = data
    return data


def evaluate_candidate_batch(
    train: pd.DataFrame,
    config: m.CandidateConfig,
    folds: list[dict[str, Any]],
    fast_mode: bool,
    script_hash: str,
    label: str,
    baseline: m.CandidateResult | None = None,
    use_cache: bool = True,
) -> m.CandidateResult:
    key = m.cache_key(config, folds, script_hash, fast_mode, label)
    if use_cache:
        cached = m.load_cache(key)
        if cached is not None:
            return cached
    if config.kind in m.BASELINE_KINDS and config.kind != "outcome_first":
        return m.evaluate_candidate(train, config, folds, fast_mode, script_hash, label, baseline, use_cache)

    fold_metrics: list[dict[str, Any]] = []
    frames: list[pd.DataFrame] = []
    pair_diags: list[dict[str, Any]] = []
    cal_diags: list[dict[str, Any]] = []
    for fold in folds:
        fold_train, fold_valid = m.fold_split(train, fold)
        if fold_train.empty or fold_valid.empty:
            continue
        effective = copy.copy(config)
        if config.kind == "outcome_first":
            effective = m.CandidateConfig(**{**asdict(config), "alpha": 0.0, "delta": 0.0, "eta": 0.0, "theta": 0.0, "kappa": 0.0, "beta": 0.50})
        data = fit_fold_model(train, fold, config, fast_mode)
        pred_frame, pair_diag, matrix_diag = predict_from_heads(data["valid_feat"], data["heads"], data["prior"], effective)
        cal_diag = m.calibration_diag_from_frame(pred_frame, matrix_diag) | {"backend": data["backend"]}
        metric = m.metrics_dict(
            pred_frame["pred_team_goals"].values,
            pred_frame["pred_opp_goals"].values,
            pred_frame["team_goals"].values,
            pred_frame["opp_goals"].values,
            pred_frame["metric_weight"].values,
        )
        metric.update(
            {
                "fold_name": fold["name"],
                "fold_weight": float(fold.get("weight", 1.0)),
                "rows": int(len(pred_frame)),
                "train_rows": int(len(fold_train)),
                **m.score_distribution(pred_frame["pred_team_goals"].values, pred_frame["pred_opp_goals"].values),
            }
        )
        fold_metrics.append(metric)
        frames.append(pred_frame)
        pair_diags.append(pair_diag)
        cal_diags.append(cal_diag)

    frame_all = pd.concat(frames, ignore_index=True)
    metrics = {name: m.combine_fold_metrics(fold_metrics, name) for name in ["weighted_awmae_p15", "unweighted_awmae_p15", "weighted_awmae_p13", "unweighted_awmae_p13", "outcome_accuracy", "exact_accuracy", "goal_diff_accuracy"]}
    dist = m.score_distribution(frame_all["pred_team_goals"].values, frame_all["pred_opp_goals"].values)
    seg = m.segment_metrics(frame_all)
    pair_summary: defaultdict[str, float] = defaultdict(float)
    for d in pair_diags:
        for k, v in d.items():
            pair_summary[k] += m.safe_float(v)
    pair_summary["pair_consistency_pass"] = float(all(bool(d.get("pair_consistency_pass", 1)) for d in pair_diags))
    selection_score, risk, acceptance = m.compute_selection(metrics, fold_metrics, seg, dist, dict(pair_summary), config, baseline)
    result = m.CandidateResult(
        config=asdict(config),
        metrics=m.to_jsonable(metrics),
        fold_metrics=m.to_jsonable(fold_metrics),
        segment_metrics=m.to_jsonable(seg),
        distribution=m.to_jsonable(dist),
        selection_score=selection_score,
        acceptance=m.to_jsonable(acceptance),
        risk_components=m.to_jsonable(risk),
        pair_diagnostics=m.to_jsonable(dict(pair_summary)),
        calibration_diagnostics=m.to_jsonable({"folds": cal_diags}),
    )
    if use_cache:
        m.save_cache(key, result)
    return result


def select_candidate_batch(train: pd.DataFrame, fast_mode: bool, script_hash: str, use_cache: bool) -> tuple[m.CandidateResult, m.CandidateResult, list[m.CandidateResult], list[dict[str, str]]]:
    candidates, skipped = m.build_candidate_registry(fast_mode)
    print(f"[metric_joint_batch] evaluating {len(candidates)} candidates")
    results: list[m.CandidateResult] = []
    baseline_results: list[m.CandidateResult] = []
    for cfg in candidates:
        base = min(baseline_results, key=lambda r: r.metrics["weighted_awmae_p15"]) if baseline_results and cfg.kind not in m.BASELINE_KINDS else None
        print(f"  candidate={cfg.name}", flush=True)
        res = evaluate_candidate_batch(train, cfg, m.PRIMARY_FOLDS, fast_mode, script_hash, "primary", base, use_cache)
        print(
            f"    w15={res.metrics['weighted_awmae_p15']:.6f} w13={res.metrics['weighted_awmae_p13']:.6f} out={res.metrics['outcome_accuracy']:.6f} accepted={res.acceptance.get('accepted')}",
            flush=True,
        )
        results.append(res)
        if cfg.kind in m.BASELINE_KINDS:
            baseline_results.append(res)
    best_baseline = min(baseline_results, key=lambda r: (r.metrics["weighted_awmae_p15"], -r.metrics["outcome_accuracy"]))
    accepted = [r for r in results if r.config["kind"] not in m.BASELINE_KINDS and r.acceptance.get("accepted", False)]
    accepted.sort(key=lambda r: (r.selection_score, r.metrics["weighted_awmae_p15"], -r.metrics["outcome_accuracy"]))
    selected = accepted[0] if accepted else min([r for r in results if r.config["kind"] not in m.BASELINE_KINDS], key=lambda r: r.selection_score)
    if not selected.acceptance.get("accepted", False):
        selected = best_baseline
    return selected, best_baseline, results, skipped


def predict_frame_vectorized(df: pd.DataFrame, model: m.ProbabilisticHeads, prior: np.ndarray, config: m.CandidateConfig) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    ordered = df.sort_values(["date", "match_id", "Id"]).reset_index(drop=True)
    heads = model.predict(ordered)
    return predict_from_heads(ordered, heads, prior, config)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-mode", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--skip-final", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_paths()
    m.predict_frame = predict_frame_vectorized
    fast_mode = not args.full_mode if m.DEFAULT_FAST_MODE else False
    use_cache = not args.no_cache
    np.random.seed(m.SEED)
    script_hash = m.file_sha256(Path(m.__file__).resolve())
    audit = m.build_audit(fast_mode)
    if not audit["feasible"]:
        m.write_audit_file(audit)
        raise RuntimeError("Internal audit failed.")

    print(f"[metric_joint_batch] loading data from {m.DATA_DIR}", flush=True)
    train, test, sample = m.load_data(read_test=True)
    assert test is not None and sample is not None
    meta_diag = m.metadata_diagnostics(train, test)
    selected, baseline, results, skipped = select_candidate_batch(train, fast_mode, script_hash, use_cache)
    config_hash = m.json_hash(selected.config)
    lock_hash = m.write_candidate_lock(selected, baseline, script_hash, config_hash, audit)
    print(f"[metric_joint_batch] candidate lock written hash={lock_hash}", flush=True)

    if args.skip_final:
        final_diag = {"skipped_final": True}
    else:
        final_config = m.CandidateConfig(**selected.config)
        submission, final_diag = m.fit_final_predict(train, test, sample, final_config, fast_mode)
        submission.to_csv(m.OUTPUT_SUB, index=False)

    local15 = m.local_submission_metrics(m.OUTPUT_SUB, test, m.PRIMARY_POWER) if m.OUTPUT_SUB.exists() and m.GT_PATH.exists() else None
    local13 = m.local_submission_metrics(m.OUTPUT_SUB, test, m.SECONDARY_POWER) if m.OUTPUT_SUB.exists() and m.GT_PATH.exists() else None
    friend = m.find_friend_csv()
    friend_report = None
    if friend is not None and m.GT_PATH.exists():
        friend_report = {"path": str(friend), "p15": m.local_submission_metrics(friend, test, m.PRIMARY_POWER), "p13": m.local_submission_metrics(friend, test, m.SECONDARY_POWER)}
    decision = m.final_decision(selected, baseline, audit)
    payload = {
        "pipeline_version": m.PIPELINE_VERSION,
        "runner": "metric_aware_joint_v1_batch_runner",
        "timestamp_utc": m.now_utc_iso(),
        "mode": "FAST_MODE" if fast_mode else "FULL_MODE",
        "seed": m.SEED,
        "script_hash": script_hash,
        "config_hash": config_hash,
        "candidate_lock_hash": lock_hash,
        "selected": asdict(selected),
        "baseline": asdict(baseline),
        "results": [asdict(r) for r in results],
        "skipped": skipped,
        "metadata_diagnostics": meta_diag,
        "final_diagnostics": final_diag,
        "audit": audit,
        "post_lock_gt_p15": local15,
        "post_lock_gt_p13": local13,
        "post_lock_friend_report": friend_report,
        "decision": decision,
    }
    m.OUTPUT_CONFIG.write_text(json.dumps(m.to_jsonable(payload), indent=2), encoding="utf-8")
    m.write_audit_file(audit)
    m.OUTPUT_REPORT.write_text(m.report_text(selected, baseline, results, skipped, lock_hash, final_diag, meta_diag, audit, local15, local13, friend_report, decision), encoding="utf-8")
    print(f"[metric_joint_batch] selected={selected.config['name']} decision={decision}", flush=True)
    print(f"[metric_joint_batch] validation_w15={selected.metrics['weighted_awmae_p15']:.6f} validation_w13={selected.metrics['weighted_awmae_p13']:.6f} outcome={selected.metrics['outcome_accuracy']:.6f}", flush=True)
    if m.OUTPUT_SUB.exists():
        print(f"[metric_joint_batch] wrote {m.OUTPUT_SUB}", flush=True)
    if local15:
        print(f"[metric_joint_batch] post_lock_gt_w15={local15['weighted_awmae']:.6f} outcome={local15['outcome_accuracy']:.6f}", flush=True)
    if local13:
        print(f"[metric_joint_batch] post_lock_gt_w13={local13['weighted_awmae']:.6f}", flush=True)


if __name__ == "__main__":
    main()
