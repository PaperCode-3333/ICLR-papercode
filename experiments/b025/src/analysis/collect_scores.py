#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from src.common.config import B025_ROOT, model_config

BENCHMARK_DIRS = {
    "SOTOPIA": "sotopia",
    "AgentSense": "agentsense",
    "SocKET_trustworthiness": "socket_trustworthiness",
    "SocialEval": "socialeval",
    "RULER_aggregation_4k": "ruler_aggregation_4k",
    "IFEval": "ifeval",
    "BBEH": "bbeh",
}


def _load(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    run = B025_ROOT / "runs" / args.run_id
    models = list(model_config()["models"])
    rows = []
    for benchmark, directory in BENCHMARK_DIRS.items():
        for model in models:
            metric_path = run / "scored" / directory / model / "metrics.json"
            status_path = run / "status" / directory / f"{model}.json"
            metrics = _load(metric_path)
            cell = _load(status_path)
            if metrics is not None:
                status = str(metrics.get("status", "incomplete"))
                score = metrics.get("primary_value")
                primary_metric = metrics.get("primary_metric")
                completed = metrics.get("completed_items")
                expected = metrics.get("expected_items")
                reason = None if status == "complete" else "completion_gate_not_met"
            elif cell is not None:
                status = str(cell.get("status", "blocked"))
                score = None
                primary_metric = cell.get("primary_metric")
                completed = cell.get("completed_items", 0)
                expected = cell.get("expected_items")
                reason = cell.get("reason")
            else:
                status = "not_run"
                score = None
                primary_metric = None
                completed = 0
                expected = None
                reason = "no metrics or cell status file"
            rows.append(
                {
                    "run_id": args.run_id,
                    "model": model,
                    "benchmark": benchmark,
                    "primary_metric": primary_metric or "",
                    "primary_score": "" if score is None else score,
                    "status": status,
                    "completed_items": "" if completed is None else completed,
                    "expected_items": "" if expected is None else expected,
                    "reason": reason or "",
                    "metrics_path": metric_path.as_posix() if metrics else "",
                    "status_path": status_path.as_posix() if cell else "",
                }
            )
    output = B025_ROOT / "derived" / "capability_scores.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    complete = sum(row["status"] == "complete" for row in rows)
    summary = {
        "run_id": args.run_id,
        "expected_cells": len(rows),
        "complete_cells": complete,
        "blocked_or_incomplete_cells": len(rows) - complete,
        "output": output.as_posix(),
    }
    (B025_ROOT / "derived" / "capability_scores.summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if complete == len(rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
