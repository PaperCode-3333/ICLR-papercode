from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from src.common.config import B025_ROOT, model_config
from src.common.io import atomic_write_json, read_jsonl, sha256_file
from src.common.run_items import BenchmarkItem, run_items

BENCHMARK = "RULER_aggregation_4k"
SOURCE = B025_ROOT / "benchmark_sources/ruler"
PREPARE = SOURCE / "scripts/data/prepare.py"
UPSTREAM_SHA = "c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a"
TASKS = ("cwe", "fwe")
EXPECTED_PER_TASK = 500
CONTEXT_LENGTH = 4096
RANDOM_SEED = 42


def _checkpoint(model_key: str) -> Path:
    spec = model_config()["models"][model_key]
    path = spec.get("checkpoint")
    if not path:
        raise RuntimeError(f"{model_key} has no verified local tokenizer; RULER cell is blocked")
    checkpoint = Path(path)
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"checkpoint/tokenizer not found: {checkpoint}")
    return checkpoint


def _tokenizer_provenance(checkpoint: Path) -> dict[str, Any]:
    records = {}
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "sentencepiece.bpe.model",
        "tokenizer.model",
    ):
        path = checkpoint / name
        if path.is_file():
            records[name] = sha256_file(path)
    if not records:
        raise RuntimeError(f"no tokenizer artifacts found in {checkpoint}")
    return {"path": checkpoint.as_posix(), "files": records}


def _data_root(model_key: str) -> Path:
    return B025_ROOT / "cache" / "ruler_data" / model_key / "4k"


def _generator_tokenizer(model_key: str, checkpoint: Path) -> tuple[Path, dict[str, Any] | None]:
    config_path = checkpoint / "tokenizer_config.json"
    if not config_path.is_file():
        return checkpoint, None
    config = json.loads(config_path.read_text(encoding="utf-8"))
    extra = config.get("extra_special_tokens")
    if not isinstance(extra, list):
        return checkpoint, None
    overlay = B025_ROOT / "cache" / "tokenizer_overlays" / model_key
    overlay.mkdir(parents=True, exist_ok=True)
    for name in (
        "config.json",
        "tokenizer.json",
        "special_tokens_map.json",
        "tokenizer.model",
        "sentencepiece.bpe.model",
    ):
        source = checkpoint / name
        if source.is_file():
            shutil.copy2(source, overlay / name)
    sanitized = dict(config)
    removed = sanitized.pop("extra_special_tokens")
    atomic_write_json(overlay / "tokenizer_config.json", sanitized)
    return overlay, {
        "reason": "transformers_4_57_rejects_list_valued_extra_special_tokens",
        "removed_unused_field": {"extra_special_tokens": removed},
        "sanitized_config_sha256": sha256_file(overlay / "tokenizer_config.json"),
    }


def prepare_data(model_key: str, num_samples: int = EXPECTED_PER_TASK) -> dict[str, Any]:
    checkpoint = _checkpoint(model_key)
    generator_tokenizer, tokenizer_overlay = _generator_tokenizer(model_key, checkpoint)
    target = _data_root(model_key)
    target.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    # Upstream prepare.py downloads Punkt when its marker is absent, although
    # the registered cwe/fwe generators never import or use it. A local sentinel
    # avoids an irrelevant network side effect; future Punkt use still fails.
    punkt_sentinel = B025_ROOT / "cache" / "nltk" / "tokenizers" / "punkt"
    punkt_sentinel.mkdir(parents=True, exist_ok=True)
    env["NLTK_DATA"] = str(B025_ROOT / "cache" / "nltk")
    env["HF_HOME"] = str(B025_ROOT / "cache" / "huggingface")
    env["TRANSFORMERS_CACHE"] = str(B025_ROOT / "cache" / "huggingface")
    env["PATH"] = f"{Path(sys.executable).parent}:{env.get('PATH', '')}"
    for task in TASKS:
        command = [
            sys.executable,
            str(PREPARE),
            "--save_dir",
            str(target),
            "--benchmark",
            "synthetic",
            "--task",
            task,
            "--tokenizer_path",
            str(generator_tokenizer),
            "--tokenizer_type",
            "hf",
            "--max_seq_length",
            str(CONTEXT_LENGTH),
            "--model_template_type",
            "base",
            "--num_samples",
            str(num_samples),
            "--random_seed",
            str(RANDOM_SEED),
        ]
        subprocess.run(
            command,
            cwd=SOURCE / "scripts/data",
            env=env,
            check=True,
        )
        output = target / task / "validation.jsonl"
        if not output.exists():
            raise RuntimeError(f"official RULER generator did not create {output}")
        count = sum(1 for line in output.read_text(encoding="utf-8").splitlines() if line.strip())
        if count != num_samples:
            raise RuntimeError(f"{task}: expected {num_samples} rows, found {count}")
    provenance = {
        "model": model_key,
        "source_commit": UPSTREAM_SHA,
        "context_length": CONTEXT_LENGTH,
        "random_seed": RANDOM_SEED,
        "samples_per_task": num_samples,
        "tokenizer": _tokenizer_provenance(checkpoint),
        "tokenizer_overlay": tokenizer_overlay,
        "registered_patch": (
            "local empty Punkt sentinel prevents unused upstream NLTK download; "
            "cwe/fwe generators do not import NLTK"
        ),
        "files": {
            task: {
                "path": (target / task / "validation.jsonl").as_posix(),
                "sha256": sha256_file(target / task / "validation.jsonl"),
            }
            for task in TASKS
        },
    }
    atomic_write_json(target / "provenance.json", provenance)
    return provenance


def load_items(model_key: str, limit: int | None = None) -> list[BenchmarkItem]:
    items: list[BenchmarkItem] = []
    root = _data_root(model_key)
    for task in TASKS:
        path = root / task / "validation.jsonl"
        if not path.exists():
            raise RuntimeError(f"RULER data missing for {model_key}; call prepare_data first")
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if limit is not None:
            rows = rows[:limit]
        for row in rows:
            prompt = str(row["input"]) + str(row.get("answer_prefix", ""))
            items.append(
                BenchmarkItem(
                    item_id=f"{task}:{int(row['index']):04d}",
                    subtask=task,
                    prompt=prompt,
                    seed=RANDOM_SEED,
                    metadata={
                        "official_index": row["index"],
                        "outputs": row["outputs"],
                        "length": row.get("length"),
                        "length_w_model_temp": row.get("length_w_model_temp"),
                        "data_file_sha256": sha256_file(path),
                    },
                )
            )
    return items


def generate(
    *,
    run_id: str,
    model_key: str,
    endpoint_override: str | None,
    limit: int | None,
) -> dict[str, int]:
    expected = limit if limit is not None else EXPECTED_PER_TASK
    prepare_data(model_key, num_samples=expected)
    output = B025_ROOT / "runs" / run_id / "raw" / "ruler_aggregation_4k" / f"{model_key}.jsonl"
    provenance_path = _data_root(model_key) / "provenance.json"
    json.loads(provenance_path.read_text(encoding="utf-8"))
    return run_items(
        run_id=run_id,
        benchmark=BENCHMARK,
        benchmark_commit=UPSTREAM_SHA,
        data_revision=f"tokenizer_specific:{sha256_file(provenance_path)}",
        model_key=model_key,
        items=load_items(model_key, limit=limit),
        generation_config={"temperature": 0.0, "top_p": 1.0, "max_tokens": 120},
        output_path=output,
        endpoint_override=endpoint_override,
    )


def _string_match_all(prediction: str, references: list[str]) -> float:
    lowered = prediction.lower()
    if not references:
        return 0.0
    return sum(reference.lower() in lowered for reference in references) / len(references)


def score(*, run_id: str, model_key: str) -> dict[str, Any]:
    raw_path = B025_ROOT / "runs" / run_id / "raw" / "ruler_aggregation_4k" / f"{model_key}.jsonl"
    latest: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(raw_path):
        latest[str(row["request_id"])] = row
    done = [row for row in latest.values() if row.get("status") == "done"]
    task_scores: dict[str, dict[str, Any]] = {}
    item_scores = []
    for task in TASKS:
        rows = [row for row in done if row["subtask"] == task]
        scores = [
            _string_match_all(str(row["parsed_response"]), row["metadata"]["outputs"])
            for row in rows
        ]
        task_scores[task] = {
            "score": sum(scores) / len(scores) * 100 if scores else None,
            "completed_items": len(rows),
            "expected_items": EXPECTED_PER_TASK,
        }
        for row, value in zip(rows, scores, strict=False):
            item_scores.append(
                {
                    "item_id": row["item_id"],
                    "task": task,
                    "score": value,
                    "prediction": row["parsed_response"],
                    "references": row["metadata"]["outputs"],
                }
            )
    complete = all(task_scores[task]["completed_items"] == EXPECTED_PER_TASK for task in TASKS)
    diagnostic = (
        sum(task_scores[task]["score"] for task in TASKS) / len(TASKS)
        if all(task_scores[task]["score"] is not None for task in TASKS)
        else None
    )
    provenance_path = _data_root(model_key) / "provenance.json"
    metrics = {
        "benchmark": BENCHMARK,
        "model": model_key,
        "expected_items": EXPECTED_PER_TASK * len(TASKS),
        "completed_items": len(done),
        "status": "complete" if complete else "incomplete",
        "source_commit": UPSTREAM_SHA,
        "data_revision": f"sha256:{sha256_file(provenance_path)}",
        "primary_metric": "mean_cwe_fwe_accuracy",
        "mean_cwe_fwe_accuracy": diagnostic,
        "primary_value": diagnostic if complete else None,
        "task_scores": task_scores,
    }
    scored_dir = B025_ROOT / "runs" / run_id / "scored" / "ruler_aggregation_4k" / model_key
    scored_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(scored_dir / "metrics.json", metrics)
    with (scored_dir / "item_scores.jsonl").open("w", encoding="utf-8") as handle:
        for row in item_scores:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return metrics
