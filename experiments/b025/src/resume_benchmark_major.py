#!/usr/bin/env python3
"""Run the formal panel benchmark-by-benchmark with an association barrier."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import threading
import traceback
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.benchmarks import bbeh, ifeval, ruler, socialeval, socialeval_iae
from src.common.config import (
    B025_ROOT,
    configure_cache_environment,
    model_config,
    runtime_config,
)
from src.common.io import atomic_write_json, atomic_write_text, read_jsonl
from src.common.vllm_server import TP_SIZE, serve_model
from src.orchestrate import initialize, mark_static_blockers, write_status


@dataclass(frozen=True)
class TaskSpec:
    key: str
    directory: str
    module: Any
    expected_items: int
    primary: bool
    mini: bool = False


@dataclass(frozen=True)
class StageSpec:
    key: str
    display_name: str
    association_benchmark: str
    tasks: tuple[TaskSpec, ...]


IFEVAL = TaskSpec("ifeval", "ifeval", ifeval, 541, True)
RULER = TaskSpec("ruler", "ruler_aggregation_4k", ruler, 1000, True)
SOCIALEVAL_IAE = TaskSpec(
    "socialeval_iae", "socialeval_iae", socialeval_iae, 1988, False
)
SOCIALEVAL = TaskSpec("socialeval", "socialeval", socialeval, 1210, True)
BBEH = TaskSpec("bbeh", "bbeh", bbeh, 4520, True)

# Estimated short-to-long order. SocialEval IAE is generated first inside the
# SocialEval stage, but only the frozen SocialEval primary metric is associated.
STAGES: tuple[StageSpec, ...] = (
    StageSpec("ifeval", "IFEval", "IFEval", (IFEVAL,)),
    StageSpec(
        "ruler_aggregation_4k",
        "RULER aggregation 4k",
        "RULER_aggregation_4k",
        (RULER,),
    ),
    StageSpec(
        "socialeval",
        "SocialEval",
        "SocialEval",
        (SOCIALEVAL_IAE, SOCIALEVAL),
    ),
    StageSpec("bbeh", "BBEH", "BBEH", (BBEH,)),
)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _raw_path(run_id: str, task: TaskSpec, model: str) -> Path:
    return B025_ROOT / "runs" / run_id / "raw" / task.directory / f"{model}.jsonl"


def _latest_done_count(path: Path) -> int:
    if not path.exists():
        return 0
    latest: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        request_id = row.get("request_id")
        if request_id:
            latest[str(request_id)] = row
    return sum(row.get("status") == "done" for row in latest.values())


def generation_complete(run_id: str, task: TaskSpec, model: str) -> bool:
    return _latest_done_count(_raw_path(run_id, task, model)) == task.expected_items


class Progress:
    def __init__(self, run_id: str, models: list[str]) -> None:
        self.path = B025_ROOT / "runs" / run_id / "benchmark_major_progress.json"
        self.heartbeat = B025_ROOT / "runs" / run_id / "benchmark_major_heartbeat.txt"
        self.lock = threading.Lock()
        self.data: dict[str, Any] = {
            "schema_version": 1,
            "run_id": run_id,
            "execution_order": "benchmark_major_short_to_long",
            "stage_order": [stage.key for stage in STAGES],
            "models": models,
            "started_at_utc": utc_now(),
            "status": "running",
            "current_stage": None,
            "events": [],
            "stages": {},
        }
        self._write("initialized")

    def _write(self, heartbeat: str) -> None:
        atomic_write_json(self.path, self.data)
        atomic_write_text(self.heartbeat, f"{utc_now()} {heartbeat}\n")

    def event(self, stage: str, phase: str, **details: Any) -> None:
        with self.lock:
            self.data["current_stage"] = stage
            event = {"timestamp_utc": utc_now(), "stage": stage, "phase": phase, **details}
            self.data["events"].append(event)
            self.data["events"] = self.data["events"][-500:]
            self._write(f"stage={stage} phase={phase}")

    def finish_stage(self, stage: str, summary: dict[str, Any]) -> None:
        with self.lock:
            self.data["stages"][stage] = summary
            self._write(f"stage={stage} phase=associated")

    def finish(self) -> None:
        with self.lock:
            self.data["status"] = "finished"
            self.data["current_stage"] = None
            self.data["finished_at_utc"] = utc_now()
            self._write("finished")


class GpuPool:
    """Allocate disjoint physical GPUs to concurrent local model servers."""

    def __init__(self, gpu_ids: list[int]) -> None:
        if not gpu_ids:
            raise ValueError("GpuPool requires at least one GPU")
        self._available = sorted(gpu_ids)
        self._condition = threading.Condition()

    @contextmanager
    def acquire(self, count: int) -> Iterator[tuple[int, ...]]:
        if count < 1:
            raise ValueError(f"Invalid GPU count: {count}")
        with self._condition:
            self._condition.wait_for(lambda: len(self._available) >= count)
            allocated = tuple(self._available[:count])
            del self._available[:count]
        try:
            yield allocated
        finally:
            with self._condition:
                self._available.extend(allocated)
                self._available.sort()
                self._condition.notify_all()


def _generate_one(
    run_id: str,
    model: str,
    endpoint: str | None,
    task: TaskSpec,
) -> dict[str, Any]:
    if generation_complete(run_id, task, model):
        return {
            "model": model,
            "task": task.key,
            "status": "skipped_generation_complete",
            "completed_items": task.expected_items,
        }
    kwargs: dict[str, Any] = {
        "run_id": run_id,
        "model_key": model,
        "endpoint_override": endpoint,
        "limit": None,
    }
    if task.module is bbeh:
        kwargs["mini"] = task.mini
    counts = task.module.generate(**kwargs)
    return {
        "model": model,
        "task": task.key,
        "status": "generation_finished",
        "counts": counts,
        "completed_items": _latest_done_count(_raw_path(run_id, task, model)),
    }


def _generate_model(
    run_id: str,
    stage: StageSpec,
    model: str,
    local: bool,
    progress: Progress,
    gpu_pool: GpuPool | None = None,
    port_by_gpu: dict[int, int] | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    group = "local" if local else "api"
    spec = model_config()["models"][model]
    runnable_tasks = list(stage.tasks)
    if stage.key == "ruler_aggregation_4k" and not spec.get("checkpoint"):
        write_status(
            run_id,
            "ruler_aggregation_4k",
            model,
            "blocked",
            "No verified tokenizer matching the API model revision.",
        )
        return [
            {
                "model": model,
                "task": "ruler",
                "status": "blocked_no_verified_tokenizer",
            }
        ]

    pending = [
        task for task in runnable_tasks if not generation_complete(run_id, task, model)
    ]
    if not pending:
        return [
            {
                "model": model,
                "task": task.key,
                "status": "skipped_generation_complete",
                "completed_items": task.expected_items,
            }
            for task in runnable_tasks
        ]

    def run_tasks(endpoint: str | None) -> None:
        for task in pending:
            progress.event(
                stage.key,
                "task_generation_started",
                group=group,
                model=model,
                task=task.key,
            )
            try:
                records.append(_generate_one(run_id, model, endpoint, task))
            except Exception as exc:
                if task.primary:
                    write_status(
                        run_id,
                        task.directory,
                        model,
                        "failed",
                        f"{type(exc).__name__}: {str(exc)[:500]}",
                        traceback=traceback.format_exc(limit=8),
                    )
                records.append(
                    {
                        "model": model,
                        "task": task.key,
                        "status": "generation_failed",
                        "reason": f"{type(exc).__name__}: {str(exc)[:500]}",
                    }
                )

    if local:
        if gpu_pool is None or port_by_gpu is None:
            raise ValueError("Local generation requires a GPU pool and port map")
        with gpu_pool.acquire(TP_SIZE[model]) as physical_gpus:
            progress.event(
                stage.key,
                "model_generation_started",
                group=group,
                model=model,
                physical_gpus=list(physical_gpus),
                request_concurrency=runtime_config()["request_concurrency"]["vllm"],
            )
            try:
                with serve_model(
                    model,
                    run_id,
                    port=port_by_gpu[physical_gpus[0]],
                    physical_gpus=physical_gpus,
                ) as endpoint:
                    run_tasks(endpoint)
            except Exception as exc:
                reason = f"{type(exc).__name__}: {str(exc)[:500]}"
                for task in pending:
                    if task.primary:
                        write_status(
                            run_id,
                            task.directory,
                            model,
                            "blocked_resource_gate",
                            reason,
                        )
                    records.append(
                        {
                            "model": model,
                            "task": task.key,
                            "status": "model_service_failed",
                            "reason": reason,
                        }
                    )
    else:
        progress.event(
            stage.key,
            "model_generation_started",
            group=group,
            model=model,
            request_concurrency=runtime_config()["request_concurrency"][spec["backend"]],
        )
        run_tasks(None)
    progress.event(stage.key, "model_generation_finished", group=group, model=model)
    return records


def _generate_group(
    run_id: str,
    stage: StageSpec,
    models: list[str],
    local: bool,
    progress: Progress,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    runtime = runtime_config()
    gpu_pool: GpuPool | None = None
    port_by_gpu: dict[int, int] | None = None
    if local:
        gpu_ids = [int(index) for index in runtime["allowed_gpus"]]
        ports = [int(port) for port in runtime["local_vllm"]["ports"]]
        if len(ports) != len(gpu_ids):
            raise ValueError("A distinct vLLM port is required for every allowed GPU")
        gpu_pool = GpuPool(gpu_ids)
        port_by_gpu = dict(zip(gpu_ids, ports, strict=True))
        workers = len(models)
    else:
        workers = min(len(models), int(runtime["api"]["model_concurrency"]))
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = [
            executor.submit(
                _generate_model,
                run_id,
                stage,
                model,
                local,
                progress,
                gpu_pool,
                port_by_gpu,
            )
            for model in models
        ]
        # Resolve in panel order for deterministic summaries; execution is concurrent.
        for future in futures:
            records.extend(future.result())
    return records


def _score_stage(
    run_id: str,
    stage: StageSpec,
    models: list[str],
    progress: Progress,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for task in stage.tasks:
        for model in models:
            spec = model_config()["models"][model]
            if task is RULER and not spec.get("checkpoint"):
                records.append(
                    {
                        "model": model,
                        "task": task.key,
                        "status": "blocked_no_verified_tokenizer",
                    }
                )
                continue
            progress.event(stage.key, "scoring", model=model, task=task.key)
            try:
                metrics = task.module.score(run_id=run_id, model_key=model)
                status = str(metrics.get("status", "incomplete"))
                write_status(
                    run_id,
                    task.directory,
                    model,
                    status,
                    None if status == "complete" else "completion_gate_not_met",
                    completed_items=metrics.get("completed_items"),
                    metrics_path=(
                        B025_ROOT
                        / "runs"
                        / run_id
                        / "scored"
                        / task.directory
                        / model
                        / "metrics.json"
                    ).as_posix(),
                )
                metric = metrics.get("primary_metric") or metrics.get("secondary_metric")
                records.append(
                    {
                        "model": model,
                        "task": task.key,
                        "status": status,
                        "completed_items": metrics.get("completed_items"),
                        "expected_items": metrics.get("expected_items"),
                        "metric": metric,
                        "value": metrics.get("primary_value")
                        if task.primary
                        else metrics.get(str(metric or "")),
                    }
                )
            except Exception as exc:
                write_status(
                    run_id,
                    task.directory,
                    model,
                    "failed",
                    f"{type(exc).__name__}: {str(exc)[:500]}",
                    traceback=traceback.format_exc(limit=8),
                )
                records.append(
                    {
                        "model": model,
                        "task": task.key,
                        "status": "scoring_failed",
                        "reason": f"{type(exc).__name__}: {str(exc)[:500]}",
                    }
                )
    return records


def _run_analysis(command: list[str], log_path: Path) -> int:
    result = subprocess.run(
        command,
        cwd=B025_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"$ {' '.join(command)}\n")
        handle.write(result.stdout)
        handle.write(f"\nexit_code={result.returncode}\n")
    return result.returncode


def _associate_stage(
    run_id: str,
    stage: StageSpec,
    score_records: list[dict[str, Any]],
) -> dict[str, Any]:
    run = B025_ROOT / "runs" / run_id
    association_dir = run / "associations" / stage.key
    association_dir.mkdir(parents=True, exist_ok=True)
    log_path = association_dir / "analysis.log"
    collect_code = _run_analysis(
        [sys.executable, "-m", "src.analysis.collect_scores", "--run-id", run_id],
        log_path,
    )
    extract_behavior_code = _run_analysis(
        [sys.executable, "-m", "src.analysis.extract_b024"],
        log_path,
    )
    correlate_code = _run_analysis(
        [sys.executable, "-m", "src.analysis.correlate"],
        log_path,
    )
    source = B025_ROOT / "reports" / "tables" / "correlation_matrix_long.csv"
    selected: list[dict[str, str]] = []
    if source.exists():
        with source.open(newline="", encoding="utf-8") as handle:
            selected = [
                row
                for row in csv.DictReader(handle)
                if row.get("benchmark") == stage.association_benchmark
            ]
    association_csv = association_dir / "association_rows.csv"
    if selected:
        with association_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(selected[0]))
            writer.writeheader()
            writer.writerows(selected)
    else:
        association_csv.write_text("", encoding="utf-8")
    capability = B025_ROOT / "derived" / "capability_scores.csv"
    if capability.exists():
        shutil.copy2(capability, association_dir / "capability_scores_snapshot.csv")
    behavior = B025_ROOT / "derived" / "behavior_metrics.csv"
    if behavior.exists():
        shutil.copy2(behavior, association_dir / "behavior_metrics_B024_snapshot.csv")
    behavior_provenance = behavior.with_suffix(".provenance.json")
    if behavior_provenance.exists():
        shutil.copy2(
            behavior_provenance,
            association_dir / "behavior_metrics_B024_snapshot.provenance.json",
        )
    complete = sum(row.get("status") == "complete" for row in score_records)
    incomplete = len(score_records) - complete
    summary = {
        "schema_version": 1,
        "run_id": run_id,
        "stage": stage.key,
        "benchmark": stage.association_benchmark,
        "created_at_utc": utc_now(),
        "score_records": score_records,
        "complete_score_records": complete,
        "incomplete_or_blocked_score_records": incomplete,
        "collect_scores_exit_code": collect_code,
        "extract_b024_exit_code": extract_behavior_code,
        "correlate_exit_code": correlate_code,
        "association_rows": len(selected),
        "association_path": association_csv.as_posix(),
        "policy": (
            "Generate all runnable models, score all models, snapshot association results, "
            "then advance to the next benchmark."
        ),
    }
    atomic_write_json(association_dir / "stage_summary.json", summary)
    return summary


def _ensure_run(run_id: str) -> None:
    manifest = B025_ROOT / "runs" / run_id / "run_manifest.json"
    if not manifest.exists():
        initialize(run_id, "formal")
        mark_static_blockers(run_id)
        return
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if payload.get("mode") != "formal" or payload.get("expected_primary_cells") != 63:
        raise RuntimeError("benchmark-major recovery requires the amended 63-cell formal run")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    configure_cache_environment()
    _ensure_run(args.run_id)
    models = list(model_config()["models"])
    specs = model_config()["models"]
    local_models = [model for model in models if specs[model]["backend"] == "vllm"]
    api_models = [model for model in models if model not in local_models]
    progress = Progress(args.run_id, models)
    try:
        for stage in STAGES:
            progress.event(stage.key, "generation_barrier_started")
            with ThreadPoolExecutor(max_workers=2) as executor:
                local_future = executor.submit(
                    _generate_group,
                    args.run_id,
                    stage,
                    local_models,
                    True,
                    progress,
                )
                api_future = executor.submit(
                    _generate_group,
                    args.run_id,
                    stage,
                    api_models,
                    False,
                    progress,
                )
                generation_records = local_future.result() + api_future.result()
            progress.event(stage.key, "generation_barrier_finished")
            score_records = _score_stage(args.run_id, stage, models, progress)
            progress.event(stage.key, "association_started")
            association = _associate_stage(args.run_id, stage, score_records)
            progress.finish_stage(
                stage.key,
                {
                    "display_name": stage.display_name,
                    "generation_records": generation_records,
                    "score_records": score_records,
                    "association": association,
                    "finished_at_utc": utc_now(),
                },
            )
        progress.finish()
        return 0
    except BaseException:
        progress.event(
            str(progress.data.get("current_stage") or "unknown"),
            "runner_failed",
            traceback=traceback.format_exc(limit=20),
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
