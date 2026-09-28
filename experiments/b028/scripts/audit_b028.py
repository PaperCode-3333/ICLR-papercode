#!/usr/bin/env python3
from __future__ import annotations
import json
from b028_backends import atomic_json
from b028_stimuli import ROOT
from b028_validate import full_audit

def main() -> int:
    result=full_audit(ROOT/"conditions"/"audit"/"seed_and_material_audit.csv")
    atomic_json(ROOT/"conditions"/"audit"/"audit_summary.json",result)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return 0 if result["status"]=="pass" else 2

if __name__=="__main__":
    raise SystemExit(main())
