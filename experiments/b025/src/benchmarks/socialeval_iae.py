from __future__ import annotations

import hashlib
import json
import random
import re
import sys
from collections import defaultdict
from typing import Any

from src.common.config import B025_ROOT
from src.common.io import atomic_write_json, read_jsonl, sha256_file
from src.common.run_items import BenchmarkItem, run_items

BENCHMARK = "SocialEval_IAE"
SOURCE = B025_ROOT / "benchmark_sources/socialeval"
DATA_PATH = SOURCE / "interpersonal_abilities_data/interpersonal_abilities_data.json"
UPSTREAM_SHA = "ac37a4341f6cb4871e1598978b2682121d269071"
BASE_SEED = 20260916


def _prompt_template() -> str:
    if str(SOURCE) not in sys.path:
        sys.path.insert(0, str(SOURCE))
    from evalprompt import Skill_Evaluation_Prompt_zhou_en

    return Skill_Evaluation_Prompt_zhou_en


def _profile(profile: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": profile.get("name"),
        "public": profile.get("public profile"),
        "private": profile.get("private profile"),
        "goal": profile.get("goal"),
    }


def _dialogue(rows: list[dict[str, Any]]) -> str:
    return "\n".join(f"{row.get('role', 'system')}: {row['content']}" for row in rows)


def _stable_seed(item_id: str) -> int:
    digest = hashlib.sha256(f"{BASE_SEED}:{item_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def load_items(limit: int | None = None) -> list[BenchmarkItem]:
    rows = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    if limit is not None:
        rows = rows[:limit]
    template = _prompt_template()
    items: list[BenchmarkItem] = []
    for row in rows:
        item_id = str(row["data_id"])
        data = row["en_data"]
        profiles = data["profile"]
        main = _profile(profiles[0])
        others = [
            {"name": profile.get("name"), "info": profile.get("public profile")}
            for profile in profiles[1:]
        ]
        question = data["question"]
        choices = [dict(choice) for choice in data["choices"]]
        random.Random(_stable_seed(item_id)).shuffle(choices)
        correct_indexes = [
            index for index, choice in enumerate(choices) if choice.get("type") == "skill choice"
        ]
        if len(correct_indexes) != 1:
            raise ValueError(f"{item_id} has {len(correct_indexes)} skill choices")
        answer = chr(65 + correct_indexes[0])
        choices_text = "\n".join(
            f"{chr(65 + index)}. {choice['content']}" for index, choice in enumerate(choices)
        )
        prompt = template.format(
            character_name=main["name"],
            public=main["public"] or "",
            private=main["private"] or "",
            goal=main["goal"] or "",
            user_profile=others,
            dialogue_context=_dialogue(data.get("content", [])),
            question=question["text"],
            choices=choices_text,
        )
        items.append(
            BenchmarkItem(
                item_id=item_id,
                subtask="IAE",
                prompt=prompt,
                seed=BASE_SEED,
                metadata={
                    "answer": answer,
                    "skills": question.get("skills", []),
                    "choice_order": [choice.get("label") for choice in choices],
                    "shuffle_seed": _stable_seed(item_id),
                    "registered_patch": "English prompt and deterministic local shuffle",
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
    output = B025_ROOT / "runs" / run_id / "raw" / "socialeval_iae" / f"{model_key}.jsonl"
    return run_items(
        run_id=run_id,
        benchmark=BENCHMARK,
        benchmark_commit=UPSTREAM_SHA,
        data_revision=f"sha256:{sha256_file(DATA_PATH)}",
        model_key=model_key,
        items=load_items(limit),
        generation_config={"temperature": 1.0, "top_p": 1.0, "max_tokens": 512},
        output_path=output,
        endpoint_override=endpoint_override,
    )


def _choice(text: str) -> str | None:
    fence = chr(96) * 3
    cleaned = text.strip().replace(fence + "json", "").replace(fence, "").strip()
    try:
        value = json.loads(cleaned)
        if isinstance(value, dict):
            choice = str(value.get("choice", "")).strip().upper()
            return choice if re.fullmatch(r"[A-Z]", choice) else None
    except json.JSONDecodeError:
        pass
    match = re.search(r'["\']choice["\']\s*:\s*["\']([A-Za-z])["\']', cleaned)
    return match.group(1).upper() if match else None


def score(*, run_id: str, model_key: str) -> dict[str, Any]:
    raw_path = B025_ROOT / "runs" / run_id / "raw" / "socialeval_iae" / f"{model_key}.jsonl"
    latest: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(raw_path):
        latest[str(row["request_id"])] = row
    rows = [row for row in latest.values() if row.get("status") == "done"]
    skill_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})
    item_scores = []
    for row in rows:
        predicted = _choice(str(row["parsed_response"]))
        correct = predicted == row["metadata"]["answer"]
        item_scores.append(
            {
                "item_id": row["item_id"],
                "prediction": predicted,
                "answer": row["metadata"]["answer"],
                "correct": correct,
            }
        )
        for skill in row["metadata"].get("skills", []):
            skill_stats[skill]["total"] += 1
            skill_stats[skill]["correct"] += int(correct)
    expected = 1988
    complete = len(rows) == expected
    accuracy = (
        sum(row["correct"] for row in item_scores) / len(item_scores) if item_scores else None
    )
    metrics = {
        "benchmark": BENCHMARK,
        "model": model_key,
        "expected_items": expected,
        "completed_items": len(rows),
        "status": "complete" if complete else "incomplete",
        "source_commit": UPSTREAM_SHA,
        "data_revision": f"sha256:{sha256_file(DATA_PATH)}",
        "secondary_metric": "iae_overall_accuracy",
        "iae_overall_accuracy": accuracy,
        "skill_scores": {
            skill: {
                **counts,
                "accuracy": counts["correct"] / counts["total"] if counts["total"] else None,
            }
            for skill, counts in sorted(skill_stats.items())
        },
        "registered_patch": "English prompt and deterministic shuffle; scoring unchanged",
    }
    scored_dir = B025_ROOT / "runs" / run_id / "scored" / "socialeval_iae" / model_key
    scored_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(scored_dir / "metrics.json", metrics)
    with (scored_dir / "item_scores.jsonl").open("w", encoding="utf-8") as handle:
        for row in item_scores:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return metrics
