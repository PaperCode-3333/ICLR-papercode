#!/usr/bin/env python3
"""Prepare tokenizer-specific official RULER cwe/fwe data on CPU only."""

from __future__ import annotations

import json
import traceback
from datetime import UTC, datetime

from src.benchmarks.ruler import EXPECTED_PER_TASK, prepare_data
from src.common.config import B025_ROOT, model_config
from src.common.io import atomic_write_json


def main() -> int:
    records = []
    for model_key, spec in model_config()["models"].items():
        if not spec.get("checkpoint"):
            records.append(
                {
                    "model": model_key,
                    "status": "blocked",
                    "reason": "no verified tokenizer for API model revision",
                }
            )
            continue
        try:
            provenance = prepare_data(model_key, num_samples=EXPECTED_PER_TASK)
            records.append(
                {
                    "model": model_key,
                    "status": "ready",
                    "samples_per_task": provenance["samples_per_task"],
                    "provenance": (
                        B025_ROOT / "cache/ruler_data" / model_key / "4k/provenance.json"
                    ).as_posix(),
                }
            )
        except Exception as exc:
            records.append(
                {
                    "model": model_key,
                    "status": "failed",
                    "reason": f"{type(exc).__name__}: {str(exc)[:500]}",
                    "traceback": traceback.format_exc(limit=8),
                }
            )
    output = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "records": records,
    }
    atomic_write_json(B025_ROOT / "manifests/ruler_preparation.json", output)
    print(json.dumps(output, indent=2))
    local_records = [row for row in records if row["status"] != "blocked"]
    return 0 if all(row["status"] == "ready" for row in local_records) else 2


if __name__ == "__main__":
    raise SystemExit(main())
