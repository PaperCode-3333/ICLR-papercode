"""Frozen B028 conditions, B026 seed/pairing semantics, and parsers."""
from __future__ import annotations
import hashlib
import json
import random
import re
from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List

import sys
from pathlib import Path
_RELEASE_SRC = Path(__file__).resolve().parents[3] / "src"
if str(_RELEASE_SRC) not in sys.path:
    sys.path.insert(0, str(_RELEASE_SRC))
from release_common.config import resolve_runtime

ROOT = Path(__file__).resolve().parents[1]

@lru_cache(None)
def design() -> Dict[str, Any]:
    return json.loads((ROOT / "conditions" / "design.json").read_text(encoding="utf-8"))

@lru_cache(None)
def models() -> Dict[str, Any]:
    return json.loads((ROOT / "conditions" / "models.json").read_text(encoding="utf-8"))

@lru_cache(None)
def runtime_config() -> Dict[str, Any]:
    return resolve_runtime(json.loads((ROOT / "conditions" / "runtime.json").read_text(encoding="utf-8")), ROOT)

@lru_cache(None)
def registry() -> Dict[str, Dict[str, Any]]:
    rows = json.loads((ROOT / "conditions" / "condition_registry.json").read_text(encoding="utf-8"))
    return {row["id"]: row for row in rows}

@lru_cache(None)
def b026_seed_manifest() -> Dict[str, Any]:
    path = Path(runtime_config()["b026_root"]) / "conditions" / "seeds.json"
    return json.loads(path.read_text(encoding="utf-8"))

@lru_cache(None)
def template(name: str) -> str:
    return (ROOT / "prompts" / f"{name}.md").read_text(encoding="utf-8")

def prompt(name: str, **values: Any) -> str:
    text = template(name)
    for key, value in values.items():
        text = text.replace("{" + key + "}", str(value))
    return text

def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)

def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()

def file_digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def stable_seed(*parts: Any) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()[:4], "big")

def manifest_entry(trial_id: int) -> Dict[str, Any]:
    rows = b026_seed_manifest()["trials"]
    if not 0 <= trial_id < len(rows):
        raise ValueError(f"trial_id outside frozen range: {trial_id}")
    return rows[trial_id]

def return_schedule(condition: str, return_seed: int) -> List[float]:
    rounds = int(design()["expansion_phase2_rounds"])
    if condition == "fixed":
        return [float(design()["fixed_return_rate"])] * rounds
    if condition != "variable":
        raise ValueError(f"unknown expansion return condition: {condition}")
    rates = list(map(float, design()["variable_rates"]))
    random.Random(stable_seed(return_seed, "returns")).shuffle(rates)
    if len(rates) != rounds or abs(sum(rates) / rounds - 0.60) > 1e-12:
        raise ValueError("invalid B026-compatible 20-round return schedule")
    return rates

def retry_decode_seed(base_decode_seed: int, invalid_output_count: int) -> int:
    return stable_seed(base_decode_seed, "format_retry", invalid_output_count)

def intervention_seed(bridge_seed: int, condition_id: str, call_tag: str) -> int:
    return stable_seed(bridge_seed, "B028-intervention", condition_id, call_tag)

def honesty_value(name: str) -> float:
    return float(next(row["value"] for row in design()["honesty_levels"] if row["name"] == name))

def session_id(spec: Dict[str, Any]) -> str:
    h = int(round(100 * float(spec["honesty_value"])))
    if spec["phase"] == "screen":
        return f"screen__{spec['model_name']}__{spec['condition_id']}__h{h:03d}__t{int(spec['trial_id']):03d}"
    return f"expanded__{spec['model_name']}__{spec['condition_id']}__{spec['return_condition']}__h{h:03d}__t{int(spec['trial_id']):03d}"

def session_spec(model: str, condition_id: str, honesty: str, trial_id: int, phase: str = "screen", return_condition: str = "fixed") -> Dict[str, Any]:
    if model not in models():
        raise ValueError(f"model outside B028: {model}")
    if condition_id not in registry():
        raise ValueError(f"unknown B028 condition: {condition_id}")
    if phase not in ("screen", "expanded"):
        raise ValueError(f"unknown phase: {phase}")
    if phase == "screen":
        if honesty not in design()["screening_honesty"]:
            raise ValueError("screening only uses 0% and 100%")
        return_condition = "fixed_linked" if condition_id == "I10" else "fixed"
        rates = [float(design()["linked_rates"][honesty] if condition_id == "I10" else design()["fixed_return_rate"])]
    else:
        if condition_id == "I10":
            raise ValueError("I10 is not expansion-eligible")
        if return_condition not in design()["return_conditions"]:
            raise ValueError("invalid expansion return condition")
        rates = return_schedule(return_condition, int(manifest_entry(trial_id)["return_seed"]))
    entry = manifest_entry(trial_id)
    condition = registry()[condition_id]
    rounds = 1 if phase == "screen" else int(design()["expansion_phase2_rounds"])
    spec = {
        "phase": phase,
        "trial_id": int(trial_id),
        "model_name": model,
        "backend": models()[model]["backend"],
        "condition_id": condition_id,
        "condition_name": condition["name"],
        "condition_family": condition["family"],
        "bridge": condition["bridge"],
        "honesty": honesty,
        "honesty_value": honesty_value(honesty),
        "return_condition": return_condition,
        "return_rates": rates,
        "stimulus_seed": int(entry["stimulus_seed"]),
        "return_seed": int(entry["return_seed"]),
        "decode_seeds": {
            "phase1": list(entry["decode_seeds"]["phase1"]),
            "bridge": int(entry["decode_seeds"]["bridge"]),
            "phase2": list(entry["decode_seeds"]["phase2"][:rounds]),
        },
        "seed_supported": bool(models()[model]["seed_supported"]),
        "prompt_version": design()["prompt_version"],
    }
    spec["session_id"] = session_id(spec)
    validate_spec(spec)
    return spec

def iter_screen_specs(selected_models: Iterable[str] | None = None, conditions: Iterable[str] | None = None, trials: int | None = None) -> Iterable[Dict[str, Any]]:
    chosen_models = list(selected_models or design()["models"])
    chosen_conditions = list(conditions or design()["core_conditions"])
    n = int(design()["model_trials"] if trials is None else trials)
    for trial_id in range(n):
        cells = [(c, h) for c in chosen_conditions for h in design()["screening_honesty"]]
        random.Random(stable_seed(design()["seed_namespace"], "B028-screen-order", trial_id)).shuffle(cells)
        for model in chosen_models:
            for condition_id, honesty in cells:
                yield session_spec(model, condition_id, honesty, trial_id, "screen")

def iter_expansion_specs(selection: Dict[str, List[str]], selected_models: Iterable[str] | None = None, trials: int | None = None) -> Iterable[Dict[str, Any]]:
    chosen_models = list(selected_models or design()["models"])
    n = int(design()["model_trials"] if trials is None else trials)
    for trial_id in range(n):
        for model in chosen_models:
            interventions = [x for x in selection.get(model, []) if x and x != "I10"]
            conditions = ["B3"] + sorted(set(interventions))
            cells = [(c, r, h["name"]) for c in conditions for r in design()["return_conditions"] for h in design()["honesty_levels"]]
            random.Random(stable_seed(design()["seed_namespace"], "B028-expanded-order", model, trial_id)).shuffle(cells)
            for condition_id, return_condition, honesty in cells:
                yield session_spec(model, condition_id, honesty, trial_id, "expanded", return_condition)

def validate_spec(spec: Dict[str, Any]) -> None:
    assert spec["model_name"] in design()["models"]
    assert spec["condition_id"] in registry()
    assert len(spec["decode_seeds"]["phase1"]) == 12
    expected_rounds = 1 if spec["phase"] == "screen" else int(design()["expansion_phase2_rounds"])
    assert len(spec["decode_seeds"]["phase2"]) == expected_rounds
    assert len(spec["return_rates"]) == expected_rounds
    if spec["phase"] == "expanded":
        assert spec["return_condition"] in design()["return_conditions"]
        assert abs(sum(spec["return_rates"]) / expected_rounds - 0.60) < 1e-12

def settle(investment: int, return_rate: float) -> Dict[str, Any]:
    returned = Decimal(3 * investment) * Decimal(str(return_rate))
    payoff = Decimal(10 - investment) + returned
    return {"investment": int(investment), "return_rate": float(return_rate), "tripled": int(3 * investment), "returned": float(returned), "payoff": float(payoff)}

def parse_investment(raw: Any) -> int | None:
    match = re.fullmatch(r"(?:\x60{3})?\s*([1-9]|10)\s*(?:\x60{3})?", str(raw).strip())
    return int(match.group(1)) if match else None

def parse_single_word(raw: Any) -> str | None:
    text = raw.strip() if isinstance(raw, str) else ""
    return text if re.fullmatch(r"[A-Za-z]+", text) else None

def parse_caution(raw: Any) -> str | None:
    text = raw.strip() if isinstance(raw, str) else ""
    return text if text in ("MORE", "EQUAL", "LESS") else None

def parse_output(raw: Any, phase: str) -> Any:
    parsers = {"phase2": parse_investment, "caution": parse_caution, "bridge_1": parse_single_word, "bridge_3": parse_single_word}
    return parsers[phase](raw)
