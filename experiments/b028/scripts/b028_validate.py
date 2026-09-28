"""B028 protocol, source-reuse, and pairing audits."""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List
from b028_sources import source_audit
from b028_stimuli import ROOT, b026_seed_manifest, design, digest, iter_screen_specs, manifest_entry, models, registry, return_schedule, runtime_config, session_spec, stable_seed, template

def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def prompt_audit() -> Dict[str, Any]:
    errors: List[str] = []
    b026 = Path(runtime_config()["b026_root"]) / "prompts"
    names = ["system","b_phase1_observe","b_phase1_feedback","bridge_0","bridge_1_request","bridge_1_ack","bridge_3_adjective_direct_request","adjective_transition","phase2_intro","phase2_feedback"]
    for name in names:
        if (ROOT / "prompts" / f"{name}.md").read_bytes() != (b026 / f"{name}.md").read_bytes():
            errors.append(f"B026_prompt_mismatch:{name}")
    expected_choice = "Investment round {round} of 20. Output a single integer from 1 to 10 and nothing else.\n"
    if template("phase2_choice") != expected_choice:
        errors.append("phase2_choice_not_20_round_version")
    b023_rows = {row["id"]: row for row in json.loads((Path(runtime_config()["b023_root"]) / "conditions" / "condition_registry.json").read_text())}
    for condition_id, row in registry().items():
        source = row.get("b023_source")
        if source in b023_rows and condition_id not in ("B0","B1_COLOR","B3"):
            if row.get("extra") != b023_rows[source].get("extra"):
                errors.append(f"B023_intervention_prompt_mismatch:{condition_id}")
    return {"status":"pass" if not errors else "fail","errors":errors,"prompt_hashes":{p.stem:_sha(p) for p in sorted((ROOT/"prompts").glob("*.md"))}}

def static_audit() -> Dict[str, Any]:
    errors: List[str] = []
    d = design()
    if d["models"] != ["qwen3.5-4b","qwen3.5-9b","qwen3.5-27b"]:
        errors.append("model_scope")
    if d["screening_honesty"] != ["0%","100%"] or d["model_trials"] != 60:
        errors.append("screening_scope")
    if d["screening_phase2_rounds"] != 1 or d["expansion_phase2_rounds"] != 20:
        errors.append("round_scope")
    if d["temperature"] != 0.7 or d["top_p"] != 1 or d["max_tokens"] != 128:
        errors.append("generation_parameters")
    if d["thinking"] != "disabled" or d["reasoning_effort"] != "none":
        errors.append("reasoning_not_disabled")
    expected = {"B0","B1_COLOR","B3","I03","I04","I05","I06","I07","I08","I10","I11","I12","I13","I14","I15"}
    if set(registry()) != expected:
        errors.append("condition_registry")
    if registry()["B1_COLOR"]["b023_source"] != "C09":
        errors.append("color_control_not_merged")
    if not all(registry()[x]["pre_intro"] for x in ("I07","I10","I12")):
        errors.append("pre_intro_placement")
    if any(registry()[x].get("pre_intro") for x in ("I03","I04","I05","I06","I08","I11","I13","I14","I15")):
        errors.append("post_intro_placement")
    if not registry()["I08"].get("caution") or not registry()["I13"].get("caution"):
        errors.append("caution_conditions")
    prompt_result = prompt_audit()
    errors.extend(prompt_result["errors"])
    target = len(d["models"]) * len(d["core_conditions"]) * len(d["screening_honesty"]) * d["model_trials"]
    if target != 3600:
        errors.append("core_target")
    return {"status":"pass" if not errors else "fail","errors":errors,"screening_target":target,"prompt":prompt_result}

def pairing_audit(output_csv: Path | None = None) -> Dict[str, Any]:
    failures: List[str] = []
    rows = []
    manifest = b026_seed_manifest()
    if manifest["master_seed"] != design()["master_seed"] or manifest["namespace"] != design()["seed_namespace"]:
        failures.append("manifest_identity")
    for trial_id in range(int(design()["model_trials"])):
        entry = manifest_entry(trial_id)
        variable = return_schedule("variable", int(entry["return_seed"]))
        if len(variable) != 20 or abs(sum(variable)/20-0.60)>1e-12:
            failures.append(f"return_schedule:{trial_id}")
        signatures = set()
        for model in design()["models"]:
            for condition_id in registry():
                if condition_id == "I10":
                    continue
                for honesty in ("0%","25%","75%","100%"):
                    for ret in ("fixed","variable"):
                        spec=session_spec(model,condition_id,honesty,trial_id,"expanded",ret)
                        signatures.add(digest({
                            "stimulus_seed":spec["stimulus_seed"],
                            "return_seed":spec["return_seed"],
                            "phase1":spec["decode_seeds"]["phase1"],
                            "bridge":spec["decode_seeds"]["bridge"],
                            "phase2":spec["decode_seeds"]["phase2"],
                            "rates":spec["return_rates"] if ret=="variable" else None,
                        }))
        # One signature per return condition; fixed omits rates and variable includes the shared schedule.
        if len(signatures) != 2:
            failures.append(f"cross_condition_pairing:{trial_id}:{len(signatures)}")
        rows.append({"trial_id":trial_id,"stimulus_seed":entry["stimulus_seed"],"return_seed":entry["return_seed"],"phase2_seed_prefix_hash":digest(entry["decode_seeds"]["phase2"][:20]),"variable_return_hash":digest(variable),"status":"pass"})
    if output_csv:
        output_csv.parent.mkdir(parents=True,exist_ok=True)
        with output_csv.open("w",newline="",encoding="utf-8") as handle:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
            writer.writeheader();writer.writerows(rows)
    return {"status":"pass" if not failures else "fail","failures":failures,"rows":len(rows)}

def full_audit(output_csv: Path | None = None) -> Dict[str, Any]:
    static=static_audit()
    pairing=pairing_audit(output_csv)
    sources=source_audit()
    return {"status":"pass" if all(x["status"]=="pass" for x in (static,pairing,sources)) else "fail","static":static,"pairing":pairing,"sources":sources}
