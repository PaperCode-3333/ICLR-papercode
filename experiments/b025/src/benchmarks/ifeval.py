from __future__ import annotations

import json
import sys
from typing import Any

from src.common.config import B025_ROOT
from src.common.io import atomic_write_json, atomic_write_text, read_jsonl, sha256_file
from src.common.run_items import BenchmarkItem, run_items

BENCHMARK = "IFEval"
SOURCE = B025_ROOT / "benchmark_sources/ifeval"
INPUT_DATA = SOURCE / "data/input_data.jsonl"
UPSTREAM_SHA = "e6890f85757dd84e27ca6df2dd30651dafad28e0"


def load_items(limit: int | None = None) -> list[BenchmarkItem]:
    rows = [
        json.loads(line)
        for line in INPUT_DATA.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if limit is not None:
        rows = rows[:limit]
    return [
        BenchmarkItem(
            item_id=str(row["key"]),
            subtask="official_full",
            prompt=row["prompt"],
            metadata={
                "official_key": row["key"],
                "instruction_id_list": row["instruction_id_list"],
                "kwargs": row["kwargs"],
            },
        )
        for row in rows
    ]


def generate(
    *,
    run_id: str,
    model_key: str,
    endpoint_override: str | None,
    limit: int | None,
) -> dict[str, int]:
    output = B025_ROOT / "runs" / run_id / "raw" / "ifeval" / f"{model_key}.jsonl"
    return run_items(
        run_id=run_id,
        benchmark=BENCHMARK,
        benchmark_commit=UPSTREAM_SHA,
        data_revision=f"sha256:{sha256_file(INPUT_DATA)}",
        model_key=model_key,
        items=load_items(limit),
        generation_config={"temperature": 0.0, "top_p": 1.0, "max_tokens": 2048},
        output_path=output,
        endpoint_override=endpoint_override,
    )


def _official_module():
    sources_root = B025_ROOT / "benchmark_sources"
    if str(sources_root) not in sys.path:
        sys.path.insert(0, str(sources_root))
    from instruction_following_eval import evaluation_lib

    return evaluation_lib


def score(*, run_id: str, model_key: str) -> dict[str, Any]:
    raw_path = B025_ROOT / "runs" / run_id / "raw" / "ifeval" / f"{model_key}.jsonl"
    rows = [row for row in read_jsonl(raw_path) if row["status"] == "done"]
    prompt_by_hash = {row["prompt_sha256"]: row["messages"][0]["content"] for row in rows}
    response_by_prompt = {
        prompt_by_hash[row["prompt_sha256"]]: row["parsed_response"] for row in rows
    }
    official = _official_module()
    inputs = official.read_prompt_list(str(INPUT_DATA))
    selected = [inp for inp in inputs if inp.prompt in response_by_prompt]
    scored_dir = B025_ROOT / "runs" / run_id / "scored" / "ifeval" / model_key
    scored_dir.mkdir(parents=True, exist_ok=True)
    response_path = scored_dir / "official_input_response.jsonl"
    atomic_write_text(
        response_path,
        "".join(
            json.dumps({"prompt": prompt, "response": response}, ensure_ascii=False) + "\n"
            for prompt, response in response_by_prompt.items()
        ),
    )
    metric_payload: dict[str, Any] = {
        "benchmark": BENCHMARK,
        "model": model_key,
        "expected_items": len(inputs),
        "completed_items": len(selected),
        "data_revision": f"sha256:{sha256_file(INPUT_DATA)}",
        "source_commit": UPSTREAM_SHA,
    }
    for label, function in [
        ("strict", official.test_instruction_following_strict),
        ("loose", official.test_instruction_following_loose),
    ]:
        outputs = [function(inp, response_by_prompt) for inp in selected]
        official.write_outputs(str(scored_dir / f"eval_results_{label}.jsonl"), outputs)
        prompt_correct = sum(output.follow_all_instructions for output in outputs)
        instruction_total = sum(len(output.follow_instruction_list) for output in outputs)
        instruction_correct = sum(sum(output.follow_instruction_list) for output in outputs)
        metric_payload[f"prompt_level_{label}_accuracy"] = (
            prompt_correct / len(outputs) if outputs else None
        )
        metric_payload[f"instruction_level_{label}_accuracy"] = (
            instruction_correct / instruction_total if instruction_total else None
        )
    complete = len(selected) == len(inputs) == 541
    metric_payload["status"] = "complete" if complete else "incomplete"
    metric_payload["primary_metric"] = "prompt_level_strict_accuracy"
    metric_payload["primary_value"] = (
        metric_payload["prompt_level_strict_accuracy"] if complete else None
    )
    atomic_write_json(scored_dir / "metrics.json", metric_payload)
    return metric_payload
