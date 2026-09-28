"""Static protocol, B026 parity, pairing, and reuse validation."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from b029_reuse import coverage_audit
from b029_stimuli import runtime_config, ROOT, design, digest, iter_specs, models, seed_manifest, session_spec, template, validate_spec


EXPECTED_PROMPTS = {
    "system": "You are participating in a behavioral experiment. The same partner will appear in two sequential tasks. Follow each task's output format exactly.\n",
    "b_phase1_observe": "Phase 1, round {round} of 12. The partner first locked a choice privately before making any report; that locked choice cannot be changed after the report. Partner reports: \"I chose {reported}.\" The system will now reveal the choice locked before the report. Reply with exactly {\"acknowledged\":true}.\n",
    "b_phase1_feedback": "System verification: the choice locked before the report was {actual}. The report was {label}.\n",
    "bridge_0": "The first task is complete. You will now interact with the same partner in a different task.\n",
    "bridge_1_request": "Name your favorite color with exactly one word. Reply with one English color only, with no spaces, punctuation, or explanation.\n",
    "bridge_1_ack": "Thank you. Continue to the investment game with the same partner.\n",
    "bridge_3_adjective_direct_request": "Describe the partner with exactly one English adjective. Reply with one English adjective only, with no spaces, punctuation, or explanation.",
    "adjective_transition": "The first task is complete. You will now interact with the same partner in a different task.",
    "phase2_intro": "You will now play a different game with the same partner. Treat this as an ongoing interaction. You have 10 tokens this round and may invest any integer amount from 1 to 10 with the partner. Your investment is tripled and given to the partner; the partner then returns some tokens. Your payoff is 10 - investment + returned tokens. After each round you will observe the return before the next investment. When asked for an investment, output one integer from 1 to 10 and nothing else.\n",
    "phase2_choice": "Investment round {round} of 20. Output a single integer from 1 to 10 and nothing else.\n",
    "phase2_feedback": "You invested {investment}. The partner received {tripled} and returned {returned} tokens. Your payoff was {payoff}.\n",
}


def prompt_hashes() -> dict[str, str]:
    return {name: hashlib.sha256(template(name).encode("utf-8")).hexdigest() for name in EXPECTED_PROMPTS}


def static_audit() -> dict[str, Any]:
    d = design(); errors: list[str] = []
    for name, expected in EXPECTED_PROMPTS.items():
        if template(name) != expected:
            errors.append(f"prompt mismatch: {name}")
    b026 = Path(runtime_config()["b026_root"])
    for name in EXPECTED_PROMPTS:
        if name == "phase2_choice":
            continue
        if template(name) != (b026 / "prompts" / f"{name}.md").read_text(encoding="utf-8"):
            errors.append(f"unexpected B026 prompt drift: {name}")
    b026_design = json.loads((b026 / "conditions" / "design.json").read_text(encoding="utf-8"))
    invariant_keys = [
        "paradigm", "phase1_rounds", "honesty_levels", "bridges", "return_conditions",
        "fixed_return_rate", "variable_rates", "budget", "multiplier", "investment_min",
        "investment_max", "models", "model_trials", "temperature", "top_p", "max_tokens",
        "thinking", "reasoning_effort", "master_seed", "bootstrap_seed",
        "bootstrap_samples", "seed_namespace", "dynamic_range_threshold",
        "threshold_hold_rounds", "spline_df", "late_slope_threshold",
        "terminal_derivative_threshold", "rope_relative_ratio", "rope_threshold_rounds",
    ]
    for key in invariant_keys:
        if d.get(key) != b026_design.get(key):
            errors.append(f"B026 invariant drift: {key}")
    if json.loads((ROOT / "conditions" / "models.json").read_text()) != json.loads((b026 / "conditions" / "models.json").read_text()):
        errors.append("B026 model registry drift")
    if int(d["phase2_rounds"]) != 20:
        errors.append("B029 phase2 must contain exactly 20 rounds")
    if d["late_rounds"] != [15, 20] or d["moving_terminal_rounds"] != [13, 20] or d["exponential_fit_rounds"] != [2, 14]:
        errors.append("20-round analysis windows mismatch")
    if d["thinking"] != "disabled" or d["reasoning_effort"] != "none":
        errors.append("thinking/reasoning settings are not disabled")
    if len(d["models"]) != 8 or set(d["models"]) != set(models()):
        errors.append("B029 must contain exactly the B026 eight-model registry")
    if "minimax-m3" in d["models"] or "gemma-4-31b-it" in d["models"]:
        errors.append("excluded model entered design")
    expected_sessions = len(d["models"]) * len(d["bridges"]) * len(d["return_conditions"]) * len(d["honesty_levels"]) * int(d["model_trials"])
    if expected_sessions != 11520 or expected_sessions != d["model_count_resolution"]["executable_sessions"]:
        errors.append("session arithmetic mismatch")
    for spec in iter_specs([d["models"][0]], trials=2):
        validate_spec(spec)
    return {
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "prompt_hashes": prompt_hashes(),
        "named_model_count": len(d["models"]),
        "target_sessions": expected_sessions,
        "target_rounds": expected_sessions * int(d["phase2_rounds"]),
    }


def pairing_audit(output_csv: Path | None = None) -> dict[str, Any]:
    d = design(); rows: list[dict[str, Any]] = []; failures: list[str] = []
    b026_root = Path(runtime_config()["b026_root"])
    b026_manifest = json.loads((b026_root / "conditions" / "seeds.json").read_text(encoding="utf-8"))
    b026_design = json.loads((b026_root / "conditions" / "design.json").read_text(encoding="utf-8"))
    current_manifest = seed_manifest()
    for trial_id in range(int(d["model_trials"])):
        current = current_manifest["trials"][trial_id]; old = b026_manifest["trials"][trial_id]
        for key in ("stimulus_seed", "return_seed"):
            if current[key] != old[key]: failures.append(f"b026_{key}:{trial_id}")
        if current["decode_seeds"]["phase1"] != old["decode_seeds"]["phase1"]:
            failures.append(f"b026_phase1_decode_seeds:{trial_id}")
        if current["decode_seeds"]["bridge"] != old["decode_seeds"]["bridge"]:
            failures.append(f"b026_bridge_decode_seed:{trial_id}")
        if current["decode_seeds"]["phase2"] != old["decode_seeds"]["phase2"][:20]:
            failures.append(f"b026_phase2_decode_seed_prefix:{trial_id}")
        reference = session_spec(d["models"][0], "bridge_0", "variable", "0%", trial_id)
        old_rates = list(map(float, b026_design["variable_rates"]))
        import random
        from b029_stimuli import stable_seed
        random.Random(stable_seed(old["return_seed"], "returns")).shuffle(old_rates)
        if reference["return_rates"] != old_rates:
            failures.append(f"b026_return_prefix:{trial_id}")
        phase_hashes = {}; actual_hashes = set(); variable_hashes = set()
        for honesty in [item["name"] for item in d["honesty_levels"]]:
            base = session_spec(d["models"][0], "bridge_0", "fixed", honesty, trial_id)
            phase_hashes[honesty] = digest(base["phase1"])
            actual_hashes.add(digest([x["actual"] for x in base["phase1"]]))
            for model in d["models"]:
                for bridge in d["bridges"]:
                    for ret in d["return_conditions"]:
                        spec = session_spec(model, bridge, ret, honesty, trial_id)
                        if digest(spec["phase1"]) != phase_hashes[honesty]:
                            failures.append(f"phase1:{trial_id}:{honesty}:{model}:{bridge}:{ret}")
                        if ret == "variable": variable_hashes.add(digest(spec["return_rates"]))
        rows.append({
            "trial_id": trial_id,
            "stimulus_seed": current["stimulus_seed"],
            "return_seed": current["return_seed"],
            "actual_sequence_hash": next(iter(actual_hashes)),
            "phase1_hash_0": phase_hashes["0%"],
            "phase1_hash_25": phase_hashes["25%"],
            "phase1_hash_75": phase_hashes["75%"],
            "phase1_hash_100": phase_hashes["100%"],
            "variable_return_hash": next(iter(variable_hashes)),
            "b026_seed_and_prefix_compatible": not any(item.endswith(f":{trial_id}") for item in failures),
            "actual_sequence_shared_across_honesty": len(actual_hashes) == 1,
            "variable_schedule_shared_across_all_conditions": len(variable_hashes) == 1,
            "status": "pass" if len(actual_hashes) == 1 and len(variable_hashes) == 1 else "fail",
        })
    if output_csv:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    return {"status": "pass" if not failures and all(row["status"] == "pass" for row in rows) else "fail", "failures": failures, "rows": len(rows)}


def reuse_audit(validate_events: bool = False, output: Path | None = None) -> dict[str, Any]:
    return coverage_audit(validate_events=validate_events, output=output)


def write_frozen_seeds(path: Path | None = None) -> Path:
    target = path or ROOT / "conditions" / "seeds.json"
    text = json.dumps(seed_manifest(), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if target.exists() and target.read_text(encoding="utf-8") != text:
        raise RuntimeError("frozen seed manifest exists with different contents")
    target.write_text(text, encoding="utf-8")
    return target
