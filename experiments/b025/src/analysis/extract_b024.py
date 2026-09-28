#!/usr/bin/env python3
"""Extract the amended B024 behavior outcomes without modifying B024."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from src.common.config import runtime_config
from datetime import UTC, datetime
from pathlib import Path

MODEL_ORDER = [
    "gemma-4-31b-it",
    "qwen3-14b",
    "qwen3-32b",
    "qwen3.5-4b",
    "qwen3.5-9b",
    "qwen3.5-27b",
    "qwen3.8-27b",
    "deepseek-v4.1-flash",
    "deepseek-v4-pro",
]
DEFAULT_RUN_ID = os.environ.get("B024_RUN_ID", "SOURCE_RUN")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite(value: str, *, field: str, model: str, bridge: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"non-finite {field} for {model}/{bridge}")
    return parsed


def read_fixed_rows(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    selected = [
        row
        for row in rows
        if row.get("return_condition") == "fixed"
        and row.get("bridge") in {"bridge_0", "bridge_3"}
    ]
    indexed: dict[tuple[str, str], dict[str, str]] = {}
    for row in selected:
        key = (row["model_name"], row["bridge"])
        if key in indexed:
            raise ValueError(f"duplicate B024 T4 row: {key}")
        indexed[key] = row
    expected = {(model, bridge) for model in MODEL_ORDER for bridge in ("bridge_0", "bridge_3")}
    missing = sorted(expected - set(indexed))
    if missing:
        raise ValueError(f"B024 T4 fixed rows missing: {missing}")
    return indexed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--b024", type=Path)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    b024 = (args.b024 or Path(runtime_config()["b024_root"])).resolve()
    run = b024 / "runs" / args.run_id
    output = (args.output or root / "derived" / "behavior_metrics.csv").resolve()
    table_path = run / "analysis" / "tables" / "T4_social_gradient_dynamics.csv"
    manifest_path = run / "analysis" / "analysis_manifest.json"
    run_result_path = run / "run_result.json"
    for path in (table_path, manifest_path, run_result_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("run_id") != args.run_id:
        raise ValueError(
            f"analysis manifest run_id={manifest.get('run_id')!r}, expected {args.run_id!r}"
        )
    rows = read_fixed_rows(table_path)

    fields = [
        "model",
        "source_run_id",
        "return_condition",
        "Y1_b0_G1",
        "Y2_b3_G1",
        "Y3_G1_b3_minus_b0",
        "Y4_gradient_auc_signed_b3_minus_b0",
        "n_valid_trials_b0",
        "n_valid_trials_b3",
        "metric_status",
    ]
    result: list[dict[str, object]] = []
    for model in MODEL_ORDER:
        b0 = rows[(model, "bridge_0")]
        b3 = rows[(model, "bridge_3")]
        y1 = finite(b0["G1"], field="G1", model=model, bridge="bridge_0")
        y2 = finite(b3["G1"], field="G1", model=model, bridge="bridge_3")
        auc_b0 = finite(
            b0["Gradient_AUC_signed"],
            field="Gradient_AUC_signed",
            model=model,
            bridge="bridge_0",
        )
        auc_b3 = finite(
            b3["Gradient_AUC_signed"],
            field="Gradient_AUC_signed",
            model=model,
            bridge="bridge_3",
        )
        result.append(
            {
                "model": model,
                "source_run_id": args.run_id,
                "return_condition": "fixed",
                "Y1_b0_G1": y1,
                "Y2_b3_G1": y2,
                "Y3_G1_b3_minus_b0": y2 - y1,
                "Y4_gradient_auc_signed_b3_minus_b0": auc_b3 - auc_b0,
                "n_valid_trials_b0": int(b0["n_valid_trials"]),
                "n_valid_trials_b3": int(b3["n_valid_trials"]),
                "metric_status": "observed",
            }
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(result)

    provenance = {
        "schema_version": 2,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "amendment_id": "B025-behavior-source-20260918-B024",
        "policy": (
            "B024 read-only; official fixed-return T4 point estimates only; "
            "nine-model panel; no imputation"
        ),
        "post_results_amendment": True,
        "b024_root": b024.as_posix(),
        "b024_run_id": args.run_id,
        "definitions": {
            "G_mbt": (
                "For model m, bridge b and round t, fit investment on honesty within each "
                "paired trial, then average the trial-level honesty slopes."
            ),
            "Y1_b0_G1": "G_(m,bridge_0,round_1,fixed)",
            "Y2_b3_G1": "G_(m,bridge_3,round_1,fixed)",
            "Y3_G1_b3_minus_b0": "Y2_b3_G1 - Y1_b0_G1",
            "Y4_gradient_auc_signed_b3_minus_b0": (
                "mean_t=1..20 G_(m,bridge_3,t,fixed) - "
                "mean_t=1..20 G_(m,bridge_0,t,fixed)"
            ),
        },
        "units": "investment tokens per 1.0 honesty unit",
        "inputs": {
            table_path.as_posix(): sha256(table_path),
            manifest_path.as_posix(): sha256(manifest_path),
            run_result_path.as_posix(): sha256(run_result_path),
        },
        "output": output.as_posix(),
        "row_count": len(result),
        "missing_metrics": [],
    }
    output.with_suffix(".provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(provenance, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
