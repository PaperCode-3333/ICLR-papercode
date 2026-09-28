from __future__ import annotations

import json
import re
import sys
import traceback
from collections import defaultdict
from datetime import UTC, datetime
from typing import Any

from src.adapters.unified import UnifiedAdapter
from src.common.config import B025_ROOT
from src.common.io import (
    append_jsonl,
    atomic_write_json,
    completed_request_ids,
    read_jsonl,
    sha256_bytes,
    sha256_file,
)

BENCHMARK = "SocialEval"
SOURCE = B025_ROOT / "benchmark_sources/socialeval"
DATA_PATH = SOURCE / "worldtree_data/worldtree_data.json"
UPSTREAM_SHA = "ac37a4341f6cb4871e1598978b2682121d269071"
EPISODES_PER_SCENARIO = 10
BASE_SEED = 20260916
MAX_STEPS = 30


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _prompt_template() -> str:
    if str(SOURCE) not in sys.path:
        sys.path.insert(0, str(SOURCE))
    from evalprompt import Ending_Evaluation_Prompt_zhou_en

    return Ending_Evaluation_Prompt_zhou_en


def _simple_profile(profile: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": profile.get("name"),
        "public": profile.get("public profile"),
        "private": profile.get("private profile"),
        "goal": profile.get("goal"),
    }


def _simple_others(profiles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"name": profile.get("name"), "public": profile.get("public profile", "")}
        for profile in profiles
    ]


def _state(data: dict[str, Any], path: list[int]) -> dict[str, Any]:
    plots = {plot["cid"]: plot for plot in data["interactive_plot"]}
    main = _simple_profile(data["predefined_profiles"][0])
    others = _simple_others(data["predefined_profiles"][1:])
    dialogue: list[str] = []
    for cid in path:
        current = plots.get(cid)
        if current is None:
            raise ValueError(f"unknown cid {cid}")
        for turn in current.get("dialog", []):
            if "profile" in turn:
                others.append(_simple_profile(turn["profile"]))
            if "content" in turn:
                dialogue.append(f"{turn.get('role', 'Narrator')}: {turn['content']}")
    current = plots[path[-1]]
    goal = current.get("goal achievement", -1) if current.get("type") == "ending" else -1
    choices = []
    if goal == -1:
        for choice in current.get("choices", []):
            choices.append(
                {
                    "cid": choice["cid"],
                    "text": f"{choice.get('content', {}).get('role', '')}: "
                    f"{choice.get('content', {}).get('content', '')}",
                }
            )
    return {
        "main": main,
        "others": others,
        "dialogue": "\n".join(dialogue),
        "choices": choices,
        "goal_achievement": goal,
        "category": str(data["predefined_profiles"][0]["orientation"]),
    }


def _parse_choice(text: str, choice_count: int) -> tuple[str, int]:
    fence = chr(96) * 3
    cleaned = text.strip().replace(fence + "json", "").replace(fence, "").strip()
    payload: dict[str, Any] | None = None
    try:
        value = json.loads(cleaned)
        if isinstance(value, dict):
            payload = value
    except json.JSONDecodeError:
        match = re.search(r"\{.*?\}", cleaned, flags=re.DOTALL)
        if match:
            try:
                value = json.loads(match.group(0))
                if isinstance(value, dict):
                    payload = value
            except json.JSONDecodeError:
                payload = None
    if payload is None:
        raise ValueError("response is not a JSON object")
    choice = str(payload.get("choice", "")).strip().upper()
    if not re.fullmatch(r"[A-Z]", choice):
        raise ValueError(f"invalid choice value {choice!r}")
    index = ord(choice) - ord("A")
    if index >= choice_count:
        raise ValueError(f"choice {choice} outside {choice_count} options")
    return choice, index


def _items(limit: int | None = None) -> list[tuple[str, dict[str, Any], int]]:
    rows = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    if limit is not None:
        rows = rows[:limit]
    result = []
    for row in rows:
        if "en_data" not in row:
            raise ValueError(f"missing en_data for {row.get('data_id')}")
        for episode in range(EPISODES_PER_SCENARIO):
            result.append((str(row["data_id"]), row["en_data"], episode))
    return result


def generate(
    *,
    run_id: str,
    model_key: str,
    endpoint_override: str | None,
    limit: int | None,
) -> dict[str, int]:
    output = B025_ROOT / "runs" / run_id / "raw" / "socialeval" / f"{model_key}.jsonl"
    adapter = UnifiedAdapter(model_key, endpoint_override=endpoint_override)
    done = completed_request_ids(output)
    template = _prompt_template()
    counts = {"done": 0, "failed": 0, "skipped_done": 0}
    generation_config = {
        "temperature": 1.0,
        "top_p": 1.0,
        "max_tokens": 512,
    }
    for data_id, data, episode in _items(limit):
        item_id = f"{data_id}:episode-{episode:02d}"
        request_id = f"{run_id}:{BENCHMARK}:{model_key}:{item_id}"
        if request_id in done:
            counts["skipped_done"] += 1
            continue
        base = {
            "run_id": run_id,
            "benchmark": BENCHMARK,
            "benchmark_commit": UPSTREAM_SHA,
            "benchmark_version": f"sha256:{sha256_file(DATA_PATH)}",
            "model_display_name": adapter.spec["display_name"],
            "model_backend": adapter.spec["backend"],
            "model_actual_id": adapter.spec["actual_id"],
            "item_id": item_id,
            "scenario_id": data_id,
            "episode": episode,
            "language": "en",
            "seed": BASE_SEED,
            "generation_config": generation_config,
            "request_id": request_id,
            "timestamp_start": utc_now(),
            "registered_patch": "use_upstream_English_prompt_constant",
        }
        path = [0]
        turns: list[dict[str, Any]] = []
        try:
            final_state: dict[str, Any] | None = None
            for step in range(MAX_STEPS):
                state = _state(data, path)
                if state["goal_achievement"] != -1:
                    final_state = state
                    break
                choices = state["choices"]
                if not choices:
                    raise ValueError("non-ending node has no choices")
                choices_text = "\n".join(
                    f"{chr(65 + index)}: {choice['text']}" for index, choice in enumerate(choices)
                )
                prompt = template.format(
                    character_name=state["main"]["name"],
                    main_profile=json.dumps(state["main"], ensure_ascii=False),
                    user_profile=json.dumps(state["others"], ensure_ascii=False),
                    dialogue_context=state["dialogue"],
                    choices=choices_text,
                )
                result = adapter.generate(
                    messages=[{"role": "user", "content": prompt}],
                    generation_config=generation_config,
                    request_id=f"{request_id}:turn-{step:02d}",
                )
                choice, choice_index = _parse_choice(result.text, len(choices))
                next_cid = choices[choice_index]["cid"]
                turns.append(
                    {
                        "step": step,
                        "prompt": prompt,
                        "prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
                        "choice": choice,
                        "next_cid": next_cid,
                        "parsed_response": result.text,
                        "raw_response": result.raw_response,
                        "retry_count": result.retry_count,
                        "latency": result.latency_seconds,
                        "request_sha256": result.request_sha256,
                    }
                )
                path.append(next_cid)
            if final_state is None:
                raise RuntimeError(f"no ending within {MAX_STEPS} steps")
            record = {
                **base,
                "path": path,
                "turns": turns,
                "goal_achievement": final_state["goal_achievement"],
                "category": final_state["category"],
                "status": "done",
                "error_type": None,
                "error_message": None,
                "timestamp_end": utc_now(),
            }
            counts["done"] += 1
        except Exception as exc:
            record = {
                **base,
                "path": path,
                "turns": turns,
                "goal_achievement": None,
                "category": None,
                "status": "failed",
                "error_type": type(exc).__name__,
                "error_message": str(exc)[:1000],
                "traceback": traceback.format_exc(limit=5),
                "timestamp_end": utc_now(),
            }
            counts["failed"] += 1
        append_jsonl(output, record)
    return counts


def score(*, run_id: str, model_key: str) -> dict[str, Any]:
    raw_path = B025_ROOT / "runs" / run_id / "raw" / "socialeval" / f"{model_key}.jsonl"
    latest: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(raw_path):
        latest[str(row["request_id"])] = row
    rows = [row for row in latest.values() if row.get("status") == "done"]
    category_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"count": 0, "success": 0})
    for row in rows:
        category = str(row["category"])
        category_stats[category]["count"] += 1
        if row["goal_achievement"] in (1, 2):
            category_stats[category]["success"] += 1
    category_scores = {
        category: {
            **values,
            "success_rate": values["success"] / values["count"] if values["count"] else None,
        }
        for category, values in sorted(category_stats.items())
    }
    diagnostic_overall = (
        sum(value["success_rate"] for value in category_scores.values()) / len(category_scores)
        if category_scores
        else None
    )
    expected = 121 * EPISODES_PER_SCENARIO
    complete = len(rows) == expected
    metrics = {
        "benchmark": BENCHMARK,
        "model": model_key,
        "expected_items": expected,
        "completed_items": len(rows),
        "status": "complete" if complete else "incomplete",
        "source_commit": UPSTREAM_SHA,
        "data_revision": f"sha256:{sha256_file(DATA_PATH)}",
        "primary_metric": "official_GAE_overall",
        "official_GAE_overall": diagnostic_overall,
        "primary_value": diagnostic_overall if complete else None,
        "category_scores": category_scores,
        "registered_patch": "upstream English prompt constant selected; no scoring change",
    }
    scored_dir = B025_ROOT / "runs" / run_id / "scored" / "socialeval" / model_key
    scored_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(scored_dir / "metrics.json", metrics)
    return metrics
