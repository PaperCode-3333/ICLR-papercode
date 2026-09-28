from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.benchmarks import bbeh, ifeval, ruler, socialeval, socialeval_iae
from src.common.config import B025_ROOT, model_config
from src.common.io import atomic_write_json
from src.common.vllm_server import serve_model
from src.preflight import build_manifest

PRIMARY = {
    "sotopia": ("SOTOPIA", "official_goal_mean", 14),
    "agentsense": ("AgentSense", "official_goal_judge_majority", 1225),
    "socket_trustworthiness": (
        "SocKET_trustworthiness",
        "macro_over_official_trustworthy_tasks",
        None,
    ),
    "socialeval": ("SocialEval", "official_GAE_overall", 1210),
    "ruler_aggregation_4k": (
        "RULER_aggregation_4k",
        "mean_cwe_fwe_accuracy",
        1000,
    ),
    "ifeval": ("IFEval", "prompt_level_strict_accuracy", 541),
    "bbeh": ("BBEH", "official_full_harmonic_mean", 4520),
}
STATIC_BLOCKERS = {
    "sotopia": (
        "Pinned official hard environment data link is unavailable; OPENAI judge "
        "and fixed Together partner endpoints are not configured."
    ),
    "agentsense": (
        "The frozen three-judge panel (GPT-4o, Qwen2.5-72B, Llama3-70B) is not "
        "fully available and verified."
    ),
    "socket_trustworthiness": ("Pinned official Hugging Face data revision is not materialized."),
}


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def status_path(run_id: str, benchmark: str, model: str) -> Path:
    return B025_ROOT / "runs" / run_id / "status" / benchmark / f"{model}.json"


def write_status(
    run_id: str,
    benchmark: str,
    model: str,
    status: str,
    reason: str | None = None,
    **extra: Any,
) -> None:
    display, metric, expected = PRIMARY.get(benchmark, (benchmark, None, None))
    payload = {
        "run_id": run_id,
        "benchmark": display,
        "benchmark_directory": benchmark,
        "model": model,
        "status": status,
        "reason": reason,
        "primary_metric": metric,
        "expected_items": expected,
        "timestamp_utc": utc_now(),
        **extra,
    }
    atomic_write_json(status_path(run_id, benchmark, model), payload)


def initialize(run_id: str, mode: str) -> None:
    run = B025_ROOT / "runs" / run_id
    run.mkdir(parents=True, exist_ok=True)
    snapshot = run / "snapshot"
    snapshot.mkdir(parents=True, exist_ok=True)
    for source in (
        B025_ROOT / "configs/models.yaml",
        B025_ROOT / "configs/runtime.yaml",
        B025_ROOT / "manifests/benchmark_item_manifests/SUMMARY.json",
    ):
        if source.exists():
            shutil.copy2(source, snapshot / source.name)
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "mode": mode,
        "created_at_utc": utc_now(),
        "models": list(model_config()["models"]),
        "benchmarks": [value[0] for value in PRIMARY.values()],
        "expected_primary_cells": len(model_config()["models"]) * len(PRIMARY),
        "formal_forbids_subsets": True,
    }
    atomic_write_json(run / "run_manifest.json", manifest)
    environment = build_manifest()
    atomic_write_json(run / "preflight.json", environment)
    atomic_write_json(B025_ROOT / "manifests/environment.json", environment)


def mark_static_blockers(run_id: str) -> None:
    for model in model_config()["models"]:
        for benchmark, reason in STATIC_BLOCKERS.items():
            write_status(run_id, benchmark, model, "blocked", reason)


def _already_complete(run_id: str, benchmark: str, model: str) -> bool:
    path = B025_ROOT / "runs" / run_id / "scored" / benchmark / model / "metrics.json"
    if not path.exists():
        return False
    metrics = json.loads(path.read_text(encoding="utf-8"))
    return metrics.get("status") == "complete" and metrics.get("primary_value") is not None


def _execute(
    module: Any,
    *,
    run_id: str,
    model: str,
    endpoint: str | None,
    benchmark_dir: str,
    limit: int | None,
    mini: bool = False,
    primary: bool = True,
) -> None:
    if primary and _already_complete(run_id, benchmark_dir, model):
        write_status(run_id, benchmark_dir, model, "complete", "resume_skip_existing_complete")
        return
    kwargs = {
        "run_id": run_id,
        "model_key": model,
        "endpoint_override": endpoint,
        "limit": limit,
    }
    if module is bbeh:
        kwargs["mini"] = mini
    counts = module.generate(**kwargs)
    metrics = module.score(run_id=run_id, model_key=model)
    if primary:
        write_status(
            run_id,
            benchmark_dir,
            model,
            str(metrics.get("status", "incomplete")),
            None if metrics.get("status") == "complete" else "completion_gate_not_met",
            generation_counts=counts,
            completed_items=metrics.get("completed_items"),
            metrics_path=(
                B025_ROOT / "runs" / run_id / "scored" / benchmark_dir / model / "metrics.json"
            ).as_posix(),
        )


def run_with_endpoint(
    run_id: str,
    model: str,
    endpoint: str | None,
    mode: str,
) -> None:
    formal = mode == "formal"
    small = None if formal else 2
    plans = [
        (ifeval, "ifeval", small, False, True),
        (bbeh, "bbeh", small, not formal, True),
        (socialeval, "socialeval", None if formal else 1, False, True),
        (socialeval_iae, "socialeval_iae", small, False, False),
    ]
    for module, directory, limit, mini, primary in plans:
        try:
            _execute(
                module,
                run_id=run_id,
                model=model,
                endpoint=endpoint,
                benchmark_dir=directory,
                limit=limit,
                mini=mini,
                primary=primary,
            )
        except Exception as exc:
            write_status(
                run_id,
                directory,
                model,
                "failed",
                f"{type(exc).__name__}: {str(exc)[:500]}",
                traceback=traceback.format_exc(limit=8),
            )
    spec = model_config()["models"][model]
    if spec.get("checkpoint"):
        try:
            _execute(
                ruler,
                run_id=run_id,
                model=model,
                endpoint=endpoint,
                benchmark_dir="ruler_aggregation_4k",
                limit=small,
                primary=True,
            )
        except Exception as exc:
            write_status(
                run_id,
                "ruler_aggregation_4k",
                model,
                "failed",
                f"{type(exc).__name__}: {str(exc)[:500]}",
                traceback=traceback.format_exc(limit=8),
            )
    else:
        write_status(
            run_id,
            "ruler_aggregation_4k",
            model,
            "blocked",
            "No verified tokenizer matching the API model revision.",
        )


def run_model(run_id: str, model: str, mode: str) -> None:
    spec = model_config()["models"][model]
    if spec["backend"] == "vllm":
        try:
            with serve_model(model, run_id) as endpoint:
                run_with_endpoint(run_id, model, endpoint, mode)
        except Exception as exc:
            reason = f"{type(exc).__name__}: {str(exc)[:500]}"
            for directory in (
                "ifeval",
                "bbeh",
                "socialeval",
                "socialeval_iae",
                "ruler_aggregation_4k",
            ):
                write_status(run_id, directory, model, "blocked_resource_gate", reason)
    else:
        run_with_endpoint(run_id, model, None, mode)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--mode", choices=["smoke", "formal"], required=True)
    args = parser.parse_args()
    initialize(args.run_id, args.mode)
    mark_static_blockers(args.run_id)
    for model in model_config()["models"]:
        run_model(args.run_id, model, args.mode)
    if args.mode == "formal":
        subprocess.run(
            [
                sys.executable,
                "-m",
                "src.analysis.collect_scores",
                "--run-id",
                args.run_id,
            ],
            cwd=B025_ROOT,
            check=False,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
