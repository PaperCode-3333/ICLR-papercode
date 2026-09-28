"""Durable B028 session runtime with read-only B026 prefix reuse."""
from __future__ import annotations
import csv
import fcntl
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List

from b028_backends import DataFailure, Endpoint, IntegrityFailure, PermanentFailure, ServiceFailure, atomic_json, audit_attempts, durable_call, immutable_json, load_json, now
from b028_sources import make_source_manifest, source_bundle, verify_source_manifest
from b028_stimuli import ROOT, design, digest, intervention_seed, models, prompt, registry, runtime_config, settle, validate_spec

CAUTION_PROMPT = "Relative to a partner with no prior history, should you be more cautious, equally cautious, or less cautious with this partner? Reply with exactly one of: MORE, EQUAL, LESS."
CAUTION_CONTINUE = "Continue to the investment decision."

def session_directory(run_dir: Path, session: Dict[str, Any]) -> Path:
    return Path(run_dir) / "sessions" / session["session_id"]

def _format_number(value: float | int) -> str:
    return str(value)

def _source_reference(source: Dict[str, Any]) -> Dict[str, Any]:
    return {key: source[key] for key in ("path", "file_sha256", "session_id", "phase1_prefix_hash") if key in source}

def run_session(endpoint: Endpoint, run_dir: Path, session: Dict[str, Any]) -> str:
    validate_spec(session)
    directory = session_directory(run_dir, session)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "session.lock").open("w", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        immutable_json(directory / "spec.json", session)
        if (directory / "session.json").exists():
            return "complete"
        if (directory / "excluded.json").exists():
            return "excluded"
        accepted = load_json(directory / "accepted.json", {})
        condition = registry()[session["condition_id"]]
        sources = source_bundle(session["model_name"], session["honesty"], session["trial_id"], condition["bridge"])
        phase_source = sources["phase1"]
        bridge_source = sources["bridge"]
        messages: List[Dict[str, str]] = [dict(item) for item in phase_source["phase1_messages"]]
        bridge_raw = None
        if condition["bridge"] == "bridge_0":
            messages.append({"role": "user", "content": prompt("bridge_0")})
        else:
            if not bridge_source or len(bridge_source["bridge_messages"]) != 3:
                raise IntegrityFailure("required B026 bridge transcript is incomplete")
            messages.extend(dict(item) for item in bridge_source["bridge_messages"])
            bridge_raw = bridge_source["bridge_raw_output"]
        extra = condition.get("extra")
        if extra:
            extra = extra.replace("{self_adjective}", bridge_raw or "")
            extra = extra.replace("{matched_adjective}", sources["matched_adjective"])
            extra = extra.replace("{n}", str(round(12 * float(session["honesty_value"]))))
        if extra and condition.get("pre_intro"):
            messages.append({"role": "user", "content": extra})
        messages.append({"role": "user", "content": prompt("phase2_intro")})
        if extra and not condition.get("pre_intro"):
            messages.append({"role": "user", "content": extra})
        started = now()
        rounds: List[Dict[str, Any]] = []
        caution_event = None

        def call(logical_call: str, text: str, phase: str, decode_seed: int) -> Dict[str, Any]:
            request_messages = messages + [{"role": "user", "content": text}]
            return durable_call(endpoint, directory, accepted, request_messages, logical_call, phase, session, decode_seed)

        try:
            if condition.get("caution"):
                seed = intervention_seed(session["decode_seeds"]["bridge"], session["condition_id"], "policy_bridge")
                caution_event = call("policy_bridge", CAUTION_PROMPT, "caution", seed)
                messages.extend([
                    {"role": "user", "content": CAUTION_PROMPT},
                    {"role": "assistant", "content": str(caution_event["raw_model_output"])},
                    {"role": "user", "content": CAUTION_CONTINUE},
                ])
            for round_no, return_rate in enumerate(session["return_rates"], 1):
                text = prompt("phase2_choice", round=round_no)
                event = call(f"phase2_{round_no:02d}", text, "phase2", session["decode_seeds"]["phase2"][round_no - 1])
                investment = int(event["parsed"])
                economics = settle(investment, return_rate)
                rounds.append({
                    "run_id": Path(run_dir).name,
                    "session_id": session["session_id"],
                    "phase": session["phase"],
                    "trial_id": session["trial_id"],
                    "model_name": session["model_name"],
                    "condition_id": session["condition_id"],
                    "honesty": session["honesty"],
                    "honesty_value": session["honesty_value"],
                    "return_condition": session["return_condition"],
                    "round": round_no,
                    **economics,
                    "raw_model_output": event["raw_model_output"],
                    "parse_status": event["parse_status"],
                    "decode_seed": session["decode_seeds"]["phase2"][round_no - 1],
                    "effective_decode_seed": event.get("effective_decode_seed"),
                    "format_retry_count": event["format_retry_count"],
                    "request_retry_count": event["request_retry_count"],
                    "first_pass": event["format_retry_count"] == 0 and event["request_retry_count"] == 0,
                })
                messages.extend([{"role": "user", "content": text}, {"role": "assistant", "content": str(investment)}])
                if session["phase"] == "expanded":
                    messages.append({"role": "user", "content": prompt(
                        "phase2_feedback",
                        investment=investment,
                        tripled=economics["tripled"],
                        returned=_format_number(economics["returned"]),
                        payoff=_format_number(economics["payoff"]),
                    )})
            audit = audit_attempts(directory)
            result = {
                **{k: session[k] for k in (
                    "phase", "session_id", "trial_id", "model_name", "backend", "condition_id",
                    "condition_name", "condition_family", "bridge", "honesty", "honesty_value",
                    "return_condition", "stimulus_seed", "return_seed", "seed_supported",
                )},
                "run_id": Path(run_dir).name,
                "session_status": "valid",
                "bridge_raw_output": bridge_raw,
                "self_adjective": bridge_raw if condition["bridge"] == "bridge_3" else None,
                "external_adjective_if_any": sources["matched_adjective"] if session["condition_id"] == "I06" else None,
                "external_count_if_any": int(round(12 * float(session["honesty_value"]))) if session["condition_id"] in ("I07", "I12") else None,
                "caution_raw_output_if_any": caution_event["raw_model_output"] if caution_event else None,
                "caution_class_if_any": caution_event["parsed"] if caution_event else None,
                "phase1_reused": True,
                "bridge_reused": condition["bridge"] in ("bridge_1", "bridge_3"),
                "phase1_source": _source_reference(phase_source),
                "bridge_source": _source_reference(bridge_source) if bridge_source else None,
                "phase1": phase_source["phase1"],
                "rounds": rounds,
                "messages": messages,
                "format_retry_count": audit["format_retry_count"],
                "request_retry_count": audit["request_retry_count"],
                "format_first_pass": audit["first_attempt_valid"] == audit["logical_calls_with_output"],
                "investment_first_pass": bool(rounds and rounds[0]["first_pass"]),
                "start_time": started,
                "end_time": now(),
                "audit": audit,
                "runtime_profile": endpoint.profile,
                "spec_hash": digest(session),
                "synthetic": endpoint.synthetic,
            }
            atomic_json(directory / "session.json", result)
            failure = directory / "service_failure.json"
            if failure.exists():
                failure.unlink()
            return "complete"
        except DataFailure as exc:
            audit = audit_attempts(directory)
            atomic_json(directory / "excluded.json", {
                "run_id": Path(run_dir).name,
                "session_id": session["session_id"],
                "phase": session["phase"],
                "trial_id": session["trial_id"],
                "model_name": session["model_name"],
                "condition_id": session["condition_id"],
                "honesty": session["honesty"],
                "return_condition": session["return_condition"],
                "session_status": "excluded",
                "reason": str(exc),
                "no_imputation": True,
                "messages_at_failure": messages,
                "accepted_calls": len(accepted),
                "audit": audit,
                "start_time": started,
                "end_time": now(),
            })
            return "excluded"
        except (PermanentFailure, ServiceFailure) as exc:
            atomic_json(directory / "service_failure.json", {
                "reason": str(exc), "timestamp": now(), "accepted_calls": len(accepted),
                "retryable": isinstance(exc, ServiceFailure),
            })
            raise

def _atomic_bytes(path: Path, value: bytes) -> None:
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)

def materialize_raw(run_dir: Path) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    sessions = [load_json(path) for path in sorted((run_dir / "sessions").glob("*/session.json"))]
    excluded = [load_json(path) for path in sorted((run_dir / "sessions").glob("*/excluded.json"))]
    rows = []
    rounds = []
    conversations = []
    for item in sessions:
        expected = 1 if item["phase"] == "screen" else int(design()["expansion_phase2_rounds"])
        if len(item.get("rounds", [])) != expected:
            raise IntegrityFailure(f"round count mismatch: {item['session_id']}")
        rows.append({k: v for k, v in item.items() if k not in ("rounds", "messages", "phase1")})
        rounds.extend(item["rounds"])
        conversations.append({"run_id": item["run_id"], "session_id": item["session_id"], "messages": item["messages"], "phase1_source": item["phase1_source"]})
    def lines(values: Iterable[Dict[str, Any]]) -> bytes:
        return b"".join((json.dumps(x, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8") for x in values)
    _atomic_bytes(run_dir / "raw_sessions.jsonl", lines(rows))
    _atomic_bytes(run_dir / "excluded.jsonl", lines(excluded))
    _atomic_bytes(run_dir / "conversations.jsonl", lines(conversations))
    path = run_dir / "raw_rounds.csv"
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    if rounds:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rounds[0]))
            writer.writeheader()
            writer.writerows(rounds)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    else:
        _atomic_bytes(path, b"")
    return {"sessions": len(sessions), "excluded": len(excluded), "rounds": len(rounds)}

def progress(run_dir: Path, specs: Iterable[Dict[str, Any]], label: str) -> Dict[str, Any]:
    specs = list(specs)
    by_model: Dict[str, Dict[str, int]] = {}
    total = {"valid": 0, "excluded": 0, "service_failure": 0, "missing": 0}
    for spec in specs:
        directory = session_directory(run_dir, spec)
        if (directory / "session.json").exists():
            status = "valid"
        elif (directory / "excluded.json").exists():
            status = "excluded"
        elif (directory / "service_failure.json").exists():
            status = "service_failure"
        else:
            status = "missing"
        total[status] += 1
        bucket = by_model.setdefault(spec["model_name"], {"target": 0, "valid": 0, "excluded": 0, "service_failure": 0, "missing": 0})
        bucket["target"] += 1
        bucket[status] += 1
    value = {"timestamp": now(), "label": label, "target": len(specs), **total, "by_model": by_model}
    atomic_json(Path(run_dir) / f"progress_{label}.json", value)
    atomic_json(Path(run_dir) / "progress.json", value)
    return value

def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def prepare_run(run_dir: Path, synthetic: bool, dry_run: bool) -> None:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    shared_root = ROOT.parents[1] / "src" / "release_common"
    immutable_json(run_dir / "shared_source_hashes.json", {
        path.name: _hash_file(path) for path in sorted(shared_root.glob("*.py"))
    })
    snapshot = run_dir / "snapshot"
    if not snapshot.exists():
        for folder in ("scripts", "conditions", "prompts"):
            shutil.copytree(ROOT / folder, snapshot / folder, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        for name in ("README.md", "B028_中文实验设计.md", "B028_分析与验收清单.md"):
            path = ROOT / name
            if path.exists():
                shutil.copy2(path, snapshot / name)
    hashes = {str(path.relative_to(snapshot)): _hash_file(path) for path in sorted(snapshot.rglob("*")) if path.is_file()}
    immutable_json(run_dir / "source_hashes.json", hashes)
    for relative, expected in hashes.items():
        current = ROOT / relative
        if not current.exists() or _hash_file(current) != expected:
            raise IntegrityFailure(f"current B028 source differs from frozen snapshot: {relative}")
    manifest_path = run_dir / "source_manifest_before.json"
    if not manifest_path.exists():
        atomic_json(manifest_path, make_source_manifest())
    check = verify_source_manifest(load_json(manifest_path))
    if check["status"] != "pass":
        raise IntegrityFailure("B023/B026 read-only source changed after B028 run creation")
    immutable_json(run_dir / "run_identity.json", {
        "run_id": run_dir.name,
        "synthetic": bool(synthetic),
        "dry_run": bool(dry_run),
        "design_hash": _hash_file(ROOT / "conditions" / "design.json"),
        "b026_seed_manifest_hash": _hash_file(Path(runtime_config()["b026_root"]) / "conditions" / "seeds.json"),
    })
