from __future__ import annotations

import traceback
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.adapters.unified import UnifiedAdapter
from src.common.config import runtime_config
from src.common.io import append_jsonl, completed_request_ids, sha256_bytes


@dataclass(frozen=True)
class BenchmarkItem:
    item_id: str
    subtask: str
    prompt: str
    language: str = "en"
    seed: int = 20260916
    metadata: dict[str, Any] | None = None


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def run_items(
    *,
    run_id: str,
    benchmark: str,
    benchmark_commit: str,
    data_revision: str,
    model_key: str,
    items: Iterable[BenchmarkItem],
    generation_config: dict[str, Any],
    output_path: Path,
    endpoint_override: str | None = None,
) -> dict[str, int]:
    adapter = UnifiedAdapter(model_key, endpoint_override=endpoint_override)
    completed = completed_request_ids(output_path)
    pending: list[BenchmarkItem] = []
    skipped_done = 0
    for item in list(items):
        request_id = f"{run_id}:{benchmark}:{model_key}:{item.item_id}"
        if request_id in completed:
            skipped_done += 1
            continue
        pending.append(item)

    def execute(item: BenchmarkItem) -> dict[str, Any]:
        request_id = f"{run_id}:{benchmark}:{model_key}:{item.item_id}"
        messages = [{"role": "user", "content": item.prompt}]
        started = utc_now()
        base = {
            "run_id": run_id,
            "benchmark": benchmark,
            "benchmark_commit": benchmark_commit,
            "benchmark_version": data_revision,
            "model_display_name": adapter.spec["display_name"],
            "model_backend": adapter.spec["backend"],
            "model_actual_id": adapter.spec["actual_id"],
            "item_id": item.item_id,
            "subtask": item.subtask,
            "language": item.language,
            "seed": item.seed,
            "messages": messages,
            "prompt_sha256": sha256_bytes(item.prompt.encode("utf-8")),
            "generation_config": generation_config,
            "request_id": request_id,
            "timestamp_start": started,
            "metadata": item.metadata or {},
        }
        try:
            result = adapter.generate(
                messages=messages,
                generation_config=generation_config,
                request_id=request_id,
            )
            record = {
                **base,
                "raw_response": result.raw_response,
                "parsed_response": result.text,
                "score_raw": None,
                "score_metric_name": None,
                "judge_model_id": None,
                "judge_raw_response": None,
                "retry_count": result.retry_count,
                "latency": result.latency_seconds,
                "status": "done",
                "error_type": None,
                "error_message": None,
                "timestamp_end": utc_now(),
                "provider_endpoint": result.endpoint,
                "request_sha256": result.request_sha256,
            }
        except Exception as exc:
            record = {
                **base,
                "raw_response": None,
                "parsed_response": None,
                "score_raw": None,
                "score_metric_name": None,
                "judge_model_id": None,
                "judge_raw_response": None,
                "retry_count": None,
                "latency": None,
                "status": "failed",
                "error_type": type(exc).__name__,
                "error_message": str(exc)[:1000],
                "traceback": traceback.format_exc(limit=5),
                "timestamp_end": utc_now(),
            }
        return record

    backend = str(adapter.spec["backend"])
    configured = runtime_config().get("request_concurrency", {})
    workers = max(1, int(configured.get(backend, 1)))
    workers = min(workers, len(pending)) if pending else 1
    counts = {"done": 0, "failed": 0, "skipped_done": skipped_done}

    if workers == 1:
        for record in (execute(item) for item in pending):
            append_jsonl(output_path, record)
            counts["done" if record["status"] == "done" else "failed"] += 1
        return counts

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(execute, item) for item in pending]
        for future in as_completed(futures):
            record = future.result()
            # Only this coordinator thread writes JSONL, preventing interleaved records.
            append_jsonl(output_path, record)
            counts["done" if record["status"] == "done" else "failed"] += 1
    return counts
