#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
from pathlib import Path

from b029_backends import atomic_json
from b029_stimuli import ROOT
from b029_validate import pairing_audit, reuse_audit, static_audit, write_frozen_seeds


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(ROOT / "conditions" / "audit"))
    parser.add_argument("--skip-full-reuse-validation", action="store_true")
    args = parser.parse_args()
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    seeds = write_frozen_seeds()
    static = static_audit()
    pairing = pairing_audit(output / "seed_and_material_audit.csv")
    reuse = reuse_audit(
        validate_events=not args.skip_full_reuse_validation,
        output=output / "phase1_reuse_coverage.json",
    )
    result = {
        "static": static,
        "pairing": pairing,
        "phase1_reuse": reuse,
        "seed_manifest": str(seeds),
        "status": "pass" if static["status"] == pairing["status"] == reuse["status"] == "pass" else "fail",
    }
    atomic_json(output / "audit_summary.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
