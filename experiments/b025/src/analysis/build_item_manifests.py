#!/usr/bin/env python3
"""Build immutable item manifests from the pinned, locally materialized sources."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SOCKET_TASKS = [
    "bragging#brag_achievement",
    "bragging#brag_action",
    "bragging#brag_possession",
    "bragging#brag_trait",
    "hypo-l",
    "neutralizing-bias-pairs",
    "propaganda-span",
    "rumor#rumor_bool",
    "two-to-lie#receiver_truth",
    "two-to-lie#sender_truth",
]


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    sources = root / "benchmark_sources"
    target = root / "manifests" / "benchmark_item_manifests"
    target.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "policy": "Pinned official sources only; missing official data remains blocked.",
        "benchmarks": {},
    }

    ifeval_source = sources / "ifeval" / "data" / "input_data.jsonl"
    ifeval_rows = load_jsonl(ifeval_source)
    ifeval_count = write_jsonl(
        target / "ifeval.jsonl",
        (
            {
                "item_id": str(row["key"]),
                "prompt_sha256": canonical_hash(row["prompt"]),
                "instruction_ids": row["instruction_id_list"],
                "item_sha256": canonical_hash(row),
            }
            for row in ifeval_rows
        ),
    )
    summary["benchmarks"]["ifeval"] = {
        "status": "ready",
        "items": ifeval_count,
        "expected_items": 541,
        "source": ifeval_source.as_posix(),
        "source_sha256": file_hash(ifeval_source),
    }

    bbeh_root = sources / "bbeh" / "bbeh" / "benchmark_tasks"
    bbeh_rows: list[dict[str, Any]] = []
    bbeh_tasks: dict[str, int] = {}
    for path in sorted(bbeh_root.glob("*/task.json")):
        task = path.parent.name
        payload = json.loads(path.read_text(encoding="utf-8"))
        examples = payload["examples"]
        bbeh_tasks[task] = len(examples)
        for index, example in enumerate(examples):
            bbeh_rows.append(
                {
                    "item_id": f"{task}:{index:04d}",
                    "task": task,
                    "index": index,
                    "input_sha256": canonical_hash(example.get("input")),
                    "target_sha256": canonical_hash(example.get("target")),
                    "item_sha256": canonical_hash(example),
                    "source_file_sha256": file_hash(path),
                }
            )
    bbeh_count = write_jsonl(target / "bbeh.jsonl", bbeh_rows)
    summary["benchmarks"]["bbeh"] = {
        "status": "ready" if bbeh_count == 4520 and len(bbeh_tasks) == 23 else "invalid",
        "items": bbeh_count,
        "expected_items": 4520,
        "tasks": bbeh_tasks,
    }

    agent_source = sources / "agentsense" / "SENSE" / "data" / "final_data.jsonl"
    agent_rows = load_jsonl(agent_source)
    agent_count = write_jsonl(
        target / "agentsense.jsonl",
        (
            {
                "item_id": str(row["sample_idx"]),
                "template_idx": row["template_idx"],
                "characters": len(row.get("characters", [])),
                "item_sha256": canonical_hash(row),
            }
            for row in agent_rows
        ),
    )
    summary["benchmarks"]["agentsense"] = {
        "status": "data_ready_judges_blocked",
        "items": agent_count,
        "expected_items": 1225,
        "unique_templates": len({row["template_idx"] for row in agent_rows}),
        "source": agent_source.as_posix(),
        "source_sha256": file_hash(agent_source),
        "blocker": "All three fixed official judge endpoints are not locally verified.",
    }

    social_root = sources / "socialeval"
    world_path = social_root / "worldtree_data" / "worldtree_data.json"
    iae_path = social_root / "interpersonal_abilities_data" / "interpersonal_abilities_data.json"
    world_rows = json.loads(world_path.read_text(encoding="utf-8"))
    iae_rows = json.loads(iae_path.read_text(encoding="utf-8"))
    world_count = write_jsonl(
        target / "socialeval_worldtree.jsonl",
        (
            {
                "item_id": row["data_id"],
                "item_sha256": canonical_hash(row),
            }
            for row in world_rows
        ),
    )
    iae_count = write_jsonl(
        target / "socialeval_iae.jsonl",
        (
            {
                "item_id": row["data_id"],
                "item_sha256": canonical_hash(row),
            }
            for row in iae_rows
        ),
    )
    summary["benchmarks"]["socialeval"] = {
        "status": "data_ready_wrapper_patch_required",
        "gae_scenarios": world_count,
        "gae_episodes_per_scenario": 10,
        "gae_expected_episodes": world_count * 10,
        "iae_records": iae_count,
        "sources": {
            world_path.as_posix(): file_hash(world_path),
            iae_path.as_posix(): file_hash(iae_path),
        },
    }

    socket_root = sources / "socket"
    socket_data_files = sorted(socket_root.glob("**/*.parquet"))
    summary["benchmarks"]["socket_trustworthiness"] = {
        "status": "ready" if len(socket_data_files) == len(SOCKET_TASKS) else "blocked",
        "tasks": SOCKET_TASKS,
        "expected_task_count": 10,
        "local_parquet_files": [path.as_posix() for path in socket_data_files],
        "blocker": None
        if len(socket_data_files) == len(SOCKET_TASKS)
        else "Pinned official Hugging Face data revision is not materialized.",
    }
    write_jsonl(
        target / "socket_trustworthiness_tasks.jsonl",
        (
            {
                "item_id": task,
                "data_status": "materialized"
                if any(task.split("#")[0] in path.name for path in socket_data_files)
                else "blocked",
            }
            for task in SOCKET_TASKS
        ),
    )

    summary["benchmarks"]["ruler_aggregation_4k"] = {
        "status": "generation_pending_tokenizer_validation",
        "tasks": {
            "cwe": {
                "samples": 500,
                "seed": 42,
                "context_length": 4096,
                "freq_cw": 30,
                "freq_ucw": 3,
                "num_cw": 10,
            },
            "fwe": {
                "samples": 500,
                "seed": 42,
                "context_length": 4096,
                "alpha": 2.0,
            },
        },
        "blocker": "Per-model official tokenizer must be verified before item generation.",
    }
    write_jsonl(
        target / "ruler_aggregation_4k_plan.jsonl",
        (
            {
                "item_id": f"{task}:{index:04d}",
                "task": task,
                "index": index,
                "seed": 42,
                "context_length": 4096,
                "status": "planned_not_generated",
            }
            for task in ("cwe", "fwe")
            for index in range(500)
        ),
    )

    summary["benchmarks"]["sotopia"] = {
        "status": "blocked",
        "environment_list": "01HAK34YPB1H1RWXQDASDKHSNS",
        "expected_official_combinations": 14,
        "blocker": (
            "Pinned official hard environment data link is unavailable and "
            "the fixed partner/evaluator endpoints are not configured."
        ),
    }
    write_jsonl(
        target / "sotopia_plan.jsonl",
        [
            {
                "item_id": "official_environment_list:01HAK34YPB1H1RWXQDASDKHSNS",
                "status": "blocked_not_materialized",
            }
        ],
    )

    for name, record in summary["benchmarks"].items():
        count_fields = {
            key: value
            for key, value in record.items()
            if key in {"items", "expected_items", "gae_scenarios", "iae_records"}
        }
        if "expected_items" in count_fields and "items" in count_fields:
            if count_fields["items"] != count_fields["expected_items"]:
                raise ValueError(f"{name} item count mismatch: {count_fields}")

    summary_path = target / "SUMMARY.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
