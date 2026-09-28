"""Frozen B029 stimulus, seed, prompt, and settlement semantics.

The B_precommit generator is a direct semantic copy of B022.  No model name,
bridge, return condition, or honesty level enters the family seed.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import sys
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
LOCAL_DEPS = ROOT / ".deps"
if LOCAL_DEPS.exists() and str(LOCAL_DEPS) not in sys.path:
    sys.path.insert(0, str(LOCAL_DEPS))


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


def truth_schedule(honesty: str, stimulus_seed: int) -> List[bool]:
    # B022 used 3 blocks of 4 with q={0,1,3,4}; this gives 0/3/9/12 TRUE.
    q = {"0%": 0, "25%": 1, "75%": 3, "100%": 4}[honesty]
    rng = random.Random(stable_seed(stimulus_seed, "truth"))
    result: List[bool] = []
    for _ in range(3):
        block = [True] * q + [False] * (4 - q)
        rng.shuffle(block)
        result.extend(block)
    return result


def actual_sequence(stimulus_seed: int) -> List[str]:
    rng = random.Random(stable_seed(stimulus_seed, "B_precommit", "actual"))
    return [rng.choice("ABC") for _ in range(12)]


def phase1_material(honesty: str, stimulus_seed: int) -> List[Dict[str, Any]]:
    actual = actual_sequence(stimulus_seed)
    truthful = truth_schedule(honesty, stimulus_seed)
    rng = random.Random(stable_seed(stimulus_seed, "B_precommit", "false"))
    rows = []
    for round_no, (choice, is_true) in enumerate(zip(actual, truthful), 1):
        report = choice if is_true else rng.choice([x for x in "ABC" if x != choice])
        rows.append({"round": round_no, "actual": choice, "reported": report, "truthful": is_true})
    return rows


def return_schedule(condition: str, return_seed: int) -> List[float]:
    phase2_rounds = int(design()["phase2_rounds"])
    if condition == "fixed":
        return [float(design()["fixed_return_rate"])] * phase2_rounds
    if condition != "variable":
        raise ValueError(f"unknown return condition: {condition}")
    # B029 is the exact 20-round prefix of B026/B024.
    rates = list(map(float, design()["variable_rates"]))
    random.Random(stable_seed(return_seed, "returns")).shuffle(rates)
    if len(rates) != phase2_rounds:
        raise ValueError("variable return schedule length does not match phase2_rounds")
    return rates


def session_spec(model: str, bridge: str, return_condition: str, honesty: str, trial_id: int) -> Dict[str, Any]:
    entry = manifest_entry(trial_id)
    spec = {
        "trial_id": trial_id,
        "model_name": model,
        "backend": models()[model]["backend"],
        "bridge": bridge,
        "honesty": honesty,
        "honesty_value": next(x["value"] for x in design()["honesty_levels"] if x["name"] == honesty),
        "return_condition": return_condition,
        "stimulus_seed": entry["stimulus_seed"],
        "return_seed": entry["return_seed"],
        "decode_seeds": entry["decode_seeds"],
        "seed_supported": bool(models()[model]["seed_supported"]),
        "phase1": phase1_material(honesty, entry["stimulus_seed"]),
        "return_rates": return_schedule(return_condition, entry["return_seed"]),
        "prompt_version": design()["prompt_version"],
    }
    spec["session_id"] = session_id(spec)
    return spec


def session_id(spec: Dict[str, Any]) -> str:
    h = int(round(float(spec.get("honesty_value", 0)) * 100))
    return f"{spec['model_name']}__{spec['bridge']}__{spec['return_condition']}__h{h:03d}__t{int(spec['trial_id']):03d}"


def iter_specs(selected_models: Iterable[str] | None = None, trials: int | None = None) -> Iterable[Dict[str, Any]]:
    d = design()
    chosen = list(selected_models or d["models"])
    unknown = set(chosen) - set(d["models"])
    if unknown:
        raise ValueError(f"models outside frozen design: {sorted(unknown)}")
    n = int(d["model_trials"] if trials is None else trials)
    if not 1 <= n <= int(d["model_trials"]):
        raise ValueError("trials must be between 1 and the frozen 60")
    for trial_id in range(n):
        # The order is randomized reproducibly within trial without changing pairing.
        cells = [(b, r, h["name"]) for b in d["bridges"] for r in d["return_conditions"] for h in d["honesty_levels"]]
        random.Random(stable_seed(d["seed_namespace"], "job_order", trial_id)).shuffle(cells)
        for model in chosen:
            for bridge, return_condition, honesty in cells:
                yield session_spec(model, bridge, return_condition, honesty, trial_id)


def retry_decode_seed(base_decode_seed: int, invalid_output_count: int) -> int:
    return stable_seed(base_decode_seed, "format_retry", invalid_output_count)


def settle(investment: int, return_rate: float) -> Dict[str, Any]:
    returned = Decimal(3 * investment) * Decimal(str(return_rate))
    payoff = Decimal(10 - investment) + returned
    return {
        "investment": int(investment),
        "return_rate": float(return_rate),
        "tripled": int(3 * investment),
        "returned": float(returned),
        "payoff": float(payoff),
    }


def parse_investment(raw: Any) -> int | None:
    match = re.fullmatch(r"(?:`{3})?\s*([1-9]|10)\s*(?:`{3})?", str(raw).strip())
    return int(match.group(1)) if match else None


def parse_acknowledgement(raw: Any) -> bool | None:
    try:
        value = json.loads(str(raw).strip())
    except (ValueError, TypeError):
        return None
    return True if isinstance(value, dict) and list(value) == ["acknowledged"] and value["acknowledged"] is True else None


def parse_single_english_word(raw: Any) -> str | None:
    text = raw.strip() if isinstance(raw, str) else ""
    return text if re.fullmatch(r"[A-Za-z]+", text) else None


def parse_adjective(raw: Any) -> str | None:
    return parse_single_english_word(raw)


def parse_color(raw: Any) -> str | None:
    return parse_single_english_word(raw)


def parse_summary(raw: Any) -> str | None:
    text = raw.strip() if isinstance(raw, str) else ""
    return text if text and "\n" not in text and len(text.split()) <= 100 else None


def parse_output(raw: Any, phase: str) -> Any:
    return {
        "phase1": parse_acknowledgement,
        "phase2": parse_investment,
        "bridge_1": parse_color,
        "bridge_3": parse_adjective,
    }[phase](raw)


def validate_spec(spec: Dict[str, Any]) -> None:
    d = design()
    assert spec["model_name"] in d["models"]
    assert spec["bridge"] in d["bridges"]
    assert spec["return_condition"] in d["return_conditions"]
    phase1_rounds = int(d["phase1_rounds"])
    phase2_rounds = int(d["phase2_rounds"])
    assert len(spec["phase1"]) == phase1_rounds and len(spec["return_rates"]) == phase2_rounds
    expected = {x["name"]: x["truthful"] for x in d["honesty_levels"]}[spec["honesty"]]
    assert sum(x["truthful"] for x in spec["phase1"]) == expected
    for block in range(3):
        assert sum(x["truthful"] for x in spec["phase1"][block * 4:block * 4 + 4]) == expected // 3
    assert all((x["actual"] == x["reported"]) == x["truthful"] for x in spec["phase1"])
    assert abs(sum(spec["return_rates"]) / phase2_rounds - 0.60) < 1e-12
