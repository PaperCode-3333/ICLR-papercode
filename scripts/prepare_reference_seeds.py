"""Materialize reference seed configuration only when explicitly invoked."""
from __future__ import annotations
import argparse
import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1] / "configs" / "reference" / "b026"

@lru_cache(None)
def design() -> Dict[str, Any]:
    return json.loads((ROOT / "conditions" / "design.json").read_text(encoding="utf-8"))

def stable_seed(*parts: Any) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()[:4], "big")

def manifest_entry(trial_id: int) -> Dict[str, Any]:
    if not 0 <= trial_id < int(design()["model_trials"]):
        raise ValueError(f"trial_id outside frozen range: {trial_id}")
    base = stable_seed(design()["master_seed"], design()["seed_namespace"], trial_id)
    phase2_rounds = int(design()["phase2_rounds"])
    decode = {
        "phase1": [stable_seed(base, "decode", "phase1", r) for r in range(1, 13)],
        "bridge": stable_seed(base, "decode", "bridge"),
        "phase2": [stable_seed(base, "decode", "phase2", r) for r in range(1, phase2_rounds + 1)],
    }
    return {
        "trial_id": trial_id,
        "stimulus_seed": stable_seed(base, "stimulus_family"),
        "return_seed": stable_seed(base, "return_module"),
        "decode_seeds": decode,
        "retry_seed_rule": "stable_seed(base_decode_seed, 'format_retry', invalid_output_count)",
    }

def seed_manifest() -> Dict[str, Any]:
    return {
        "version": design()["version"],
        "master_seed": design()["master_seed"],
        "namespace": design()["seed_namespace"],
        "trial_id_range": [0, int(design()["model_trials"]) - 1],
        "retry_seed_rule": "stable_seed(base_decode_seed, 'format_retry', invalid_output_count)",
        "trials": [manifest_entry(i) for i in range(int(design()["model_trials"]))],
    }

def main():
    global ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("--b026-root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    ROOT = args.b026_root.resolve()
    target = args.output or ROOT / "conditions" / "seeds.json"
    text = json.dumps(seed_manifest(), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if target.exists():
        if target.read_text() != text:
            raise RuntimeError("Existing reference seed configuration differs; refusing overwrite")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")

if __name__ == "__main__":
    main()
