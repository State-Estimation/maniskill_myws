#!/usr/bin/env python
"""Evaluate ActProbe with official and SAFE-comparable detector metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
ACTPROBE_ROOT = ROOT / "third_party" / "actprobe"
ACTPROBE_PI0_CODE = ACTPROBE_ROOT / "benchmarks" / "pi0_libero" / "code"
EXPECTED_ACTPROBE_COMMIT = "d5dfbcc98ee5e5766f8aa548c657d3446e40e272"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ACTPROBE_ROOT))
sys.path.insert(0, str(ACTPROBE_PI0_CODE))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--target-episode-fpr", type=float, default=0.05)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    if not 0.0 <= args.target_episode_fpr < 1.0:
        parser.error("--target-episode-fpr must lie in [0, 1)")
    return args


def _require_official_checkout() -> None:
    commit = subprocess.run(
        ["git", "-C", str(ACTPROBE_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != EXPECTED_ACTPROBE_COMMIT:
        raise RuntimeError(
            f"Official ActProbe commit {commit} != supported {EXPECTED_ACTPROBE_COMMIT}"
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while block := file.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _wilson(successes: int, total: int) -> list[float]:
    if total <= 0:
        return [float("nan"), float("nan")]
    z = 1.959963984540054
    probability = successes / total
    denominator = 1 + z * z / total
    center = (probability + z * z / (2 * total)) / denominator
    half = (
        z
        * math.sqrt(
            probability * (1 - probability) / total
            + z * z / (4 * total * total)
        )
        / denominator
    )
    return [max(0.0, center - half), min(1.0, center + half)]


def _threshold_for_success_fpr(scores: list[float], cap: float) -> float:
    values = np.sort(np.asarray(scores, dtype=np.float64))[::-1]
    if not len(values):
        raise ValueError("Cannot calibrate ActProbe without successful episodes")
    allowed = math.floor(cap * len(values))
    if allowed >= len(values):
        return float("-inf")
    return float(np.nextafter(np.float32(values[allowed]), np.float32(np.inf)))


def _operating_point(results, threshold: float, chunk_len: int) -> dict[str, Any]:
    labels = np.asarray([result["label"] for result in results], dtype=np.int64)
    alerts = np.asarray(
        [bool(np.any(result["scores"] >= threshold)) for result in results]
    )
    failures = labels == 1
    successes = labels == 0
    first_alerts = []
    relative_detection_times = []
    for result, is_failure in zip(results, failures, strict=True):
        if not is_failure:
            continue
        hits = np.flatnonzero(result["scores"] >= threshold)
        if len(hits):
            first_alerts.append(int(hits[0]))
            relative_detection_times.append(float((hits[0] + 1) / result["length"]))
        else:
            relative_detection_times.append(1.0)
    true_positives = int(np.sum(alerts & failures))
    false_positives = int(np.sum(alerts & successes))
    failure_count = int(np.sum(failures))
    success_count = int(np.sum(successes))
    return {
        "threshold": threshold,
        "tp": true_positives,
        "failures": failure_count,
        "tpr": true_positives / failure_count,
        "tpr_wilson95": _wilson(true_positives, failure_count),
        "fp": false_positives,
        "successes": success_count,
        "fpr": false_positives / success_count,
        "fpr_wilson95": _wilson(false_positives, success_count),
        "mean_first_alert_chunk_zero_based_detected_only": (
            float(np.mean(first_alerts)) if first_alerts else None
        ),
        "mean_first_alert_step_detected_only": (
            float(chunk_len * (np.mean(first_alerts) + 1))
            if first_alerts
            else None
        ),
        "mean_relative_detection_time_all_failures": float(
            np.mean(relative_detection_times)
        ),
    }


def main() -> None:
    args = _parse_args()
    _require_official_checkout()

    import torch
    from sklearn.metrics import average_precision_score, roc_auc_score

    from lib.methods.actprobe import LANG_DIM, load_ckpt, score_episodes
    from lib_shared.metrics import q_auc
    from maniskill_myws.rlt.actprobe import load_actprobe_dataset

    summaries = []
    for raw_checkpoint in args.checkpoints:
        checkpoint = Path(raw_checkpoint).resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"ActProbe checkpoint not found: {checkpoint}")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        config = payload.get("config")
        if not isinstance(config, dict):
            raise ValueError(f"ActProbe checkpoint has no config: {checkpoint}")
        dataset_path = (
            Path(args.dataset).resolve()
            if args.dataset is not None
            else Path(config["dataset_path"]).resolve()
        )
        if not dataset_path.is_file():
            raise FileNotFoundError(f"ActProbe dataset not found: {dataset_path}")
        if _sha256(dataset_path) != config["dataset_sha256"]:
            raise ValueError("ActProbe dataset content does not match checkpoint")
        dataset = load_actprobe_dataset(dataset_path)
        if dataset.metadata != config["dataset_metadata"]:
            raise ValueError("ActProbe dataset metadata does not match checkpoint")
        split_indices = config["split_indices"]
        task_id = config["task_id"]
        task_embeddings = {
            task_id: np.zeros((LANG_DIM,), dtype=np.float32)
        }

        model, norm_mean, norm_std = load_ckpt(str(checkpoint), device=args.device)
        split_results = {}
        split_metrics = {}
        for split, indices in split_indices.items():
            episodes = [
                {
                    "raw": dataset.episodes[int(index)].features,
                    "label": 0 if dataset.episodes[int(index)].success else 1,
                    "task_id": task_id,
                    "length": len(dataset.episodes[int(index)].features),
                    "episode_id": dataset.episodes[int(index)].seed,
                }
                for index in indices
            ]
            results = score_episodes(
                model,
                episodes,
                task_embeddings,
                norm_mean,
                norm_std,
                device=args.device,
            )
            split_results[split] = results
            labels = np.asarray([result["label"] for result in results])
            max_scores = np.asarray(
                [float(np.max(result["scores"])) for result in results]
            )
            split_metrics[split] = {
                "episodes": len(results),
                "successes": int(np.sum(labels == 0)),
                "failures": int(np.sum(labels == 1)),
                "full_episode_roc_auc": float(roc_auc_score(labels, max_scores)),
                "full_episode_average_precision": float(
                    average_precision_score(labels, max_scores)
                ),
                "official_q025_taskmax_roc_auc": float(
                    q_auc(results, mode="taskmax", q=0.25)
                ),
            }

        val_success_max = [
            float(np.max(result["scores"]))
            for result in split_results["val_seen"]
            if result["label"] == 0
        ]
        threshold = _threshold_for_success_fpr(
            val_success_max, args.target_episode_fpr
        )
        operating_point = _operating_point(
            split_results["val_unseen"],
            threshold,
            int(dataset.metadata["chunk_len"]),
        )
        summary = {
            "schema": "maniskill_actprobe_eval_v1",
            "checkpoint": str(checkpoint),
            "dataset": str(dataset_path),
            "seed": int(config["seed"]),
            "official_actprobe_commit": EXPECTED_ACTPROBE_COMMIT,
            "task_embedding": config["task_embedding"],
            "calibration": "val_seen successful-episode full max score",
            "test": "val_unseen only",
            "target_episode_fpr": args.target_episode_fpr,
            "split_metrics": split_metrics,
            "calibrated_operating_point": operating_point,
        }
        summary_path = checkpoint.with_name("eval_summary.json")
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        summaries.append(summary)

    aggregate = {}
    for key, extractor in {
        "val_unseen_full_episode_roc_auc": lambda item: item["split_metrics"][
            "val_unseen"
        ]["full_episode_roc_auc"],
        "val_unseen_average_precision": lambda item: item["split_metrics"][
            "val_unseen"
        ]["full_episode_average_precision"],
        "val_unseen_official_q025_taskmax_roc_auc": lambda item: item[
            "split_metrics"
        ]["val_unseen"]["official_q025_taskmax_roc_auc"],
        "calibrated_fpr": lambda item: item["calibrated_operating_point"]["fpr"],
        "calibrated_recall": lambda item: item["calibrated_operating_point"]["tpr"],
    }.items():
        values = np.asarray([extractor(item) for item in summaries], dtype=np.float64)
        aggregate[key] = {
            "mean": float(np.mean(values)),
            "sample_std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        }
    output = {
        "runs": summaries,
        "aggregate": aggregate,
    }
    output_path = (
        Path(args.output).resolve()
        if args.output is not None
        else Path.cwd() / "actprobe_eval_aggregate.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
