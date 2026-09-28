from __future__ import annotations

import importlib.util
import json
import statistics
from pathlib import Path
from typing import Any

from src.common.config import B025_ROOT
from src.common.io import atomic_write_json, atomic_write_text, read_jsonl, sha256_file
from src.common.run_items import BenchmarkItem, run_items

BENCHMARK = "BBEH"
SOURCE = B025_ROOT / "benchmark_sources/bbeh"
TASK_ROOT = SOURCE / "bbeh/benchmark_tasks"
UPSTREAM_SHA = "80d12ca916b7158f22293fcf3144f4d3d854d4be"


def _task_files() -> list[Path]:
    return sorted(TASK_ROOT.glob("*/task.json"))


def load_items(limit: int | None = None, mini: bool = False) -> list[BenchmarkItem]:
    if mini:
        path = SOURCE / "bbeh/mini/data.json"
        rows = json.loads(path.read_text(encoding="utf-8"))["examples"]
        pairs = [("mini", index, row, path) for index, row in enumerate(rows)]
    else:
        pairs = []
        for path in _task_files():
            rows = json.loads(path.read_text(encoding="utf-8"))["examples"]
            pairs.extend((path.parent.name, index, row, path) for index, row in enumerate(rows))
    if limit is not None:
        pairs = pairs[:limit]
    return [
        BenchmarkItem(
            item_id=f"{task}:{index:04d}",
            subtask=task,
            prompt=row["input"],
            metadata={
                "official_task": task,
                "official_index": index,
                "source_file": str(path.relative_to(SOURCE)),
                "source_sha256": sha256_file(path),
            },
        )
        for task, index, row, path in pairs
    ]


def generate(
    *,
    run_id: str,
    model_key: str,
    endpoint_override: str | None,
    limit: int | None,
    mini: bool,
) -> dict[str, int]:
    output = B025_ROOT / "runs" / run_id / "raw" / "bbeh" / f"{model_key}.jsonl"
    data_revision = "mini" if mini else "full-4520"
    return run_items(
        run_id=run_id,
        benchmark=BENCHMARK,
        benchmark_commit=UPSTREAM_SHA,
        data_revision=data_revision,
        model_key=model_key,
        items=load_items(limit=limit, mini=mini),
        generation_config={"temperature": 0.0, "top_p": 1.0, "max_tokens": 2048},
        output_path=output,
        endpoint_override=endpoint_override,
    )


def _official_evaluator():
    path = SOURCE / "bbeh/evaluate.py"
    spec = importlib.util.spec_from_file_location("bbeh_official_evaluate", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import official evaluator at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _targets() -> dict[str, str]:
    values: dict[str, str] = {}
    for path in _task_files():
        rows = json.loads(path.read_text(encoding="utf-8"))["examples"]
        for index, row in enumerate(rows):
            values[f"{path.parent.name}:{index:04d}"] = row["target"]
    mini_path = SOURCE / "bbeh/mini/data.json"
    mini_rows = json.loads(mini_path.read_text(encoding="utf-8"))["examples"]
    for index, row in enumerate(mini_rows):
        values[f"mini:{index:04d}"] = row["target"]
    return values


def score(*, run_id: str, model_key: str) -> dict[str, Any]:
    raw_path = B025_ROOT / "runs" / run_id / "raw" / "bbeh" / f"{model_key}.jsonl"
    rows = [row for row in read_jsonl(raw_path) if row["status"] == "done"]
    targets = _targets()
    official = _official_evaluator()
    scored = []
    for row in rows:
        target = targets.get(row["item_id"])
        correct = (
            official.evaluate_correctness(row["parsed_response"], target)
            if target is not None
            else None
        )
        scored.append(
            {
                "item_id": row["item_id"],
                "subtask": row["subtask"],
                "prediction": row["parsed_response"],
                "target": target,
                "correct": correct,
            }
        )
    by_task: dict[str, list[bool]] = {}
    for row in scored:
        if row["correct"] is not None:
            by_task.setdefault(row["subtask"], []).append(bool(row["correct"]))
    per_task = {
        task: {"accuracy": sum(values) / len(values), "n": len(values)}
        for task, values in sorted(by_task.items())
    }
    valid = [bool(row["correct"]) for row in scored if row["correct"] is not None]
    task_accuracies = [entry["accuracy"] for entry in per_task.values()]
    harmonic = statistics.harmonic_mean(task_accuracies) if task_accuracies else None
    micro = sum(valid) / len(valid) if valid else None
    complete = len(valid) == 4520 and len(per_task) == 23
    metrics = {
        "benchmark": BENCHMARK,
        "model": model_key,
        "expected_items": 4520,
        "completed_items": len(valid),
        "expected_subtasks": 23,
        "completed_subtasks": len(per_task),
        "status": "complete" if complete else "incomplete",
        "source_commit": UPSTREAM_SHA,
        "official_full_harmonic_mean": harmonic,
        "official_full_micro_average": micro,
        "primary_metric": "official_full_harmonic_mean",
        "primary_value": harmonic if complete else None,
        "subtasks": per_task,
    }
    scored_dir = B025_ROOT / "runs" / run_id / "scored" / "bbeh" / model_key
    scored_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        scored_dir / "item_scores.jsonl",
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in scored),
    )
    atomic_write_json(scored_dir / "metrics.json", metrics)
    return metrics
