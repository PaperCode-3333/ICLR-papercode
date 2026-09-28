"""Session runtime and immutable/raw-table materialization for B029."""
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
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List

from b029_backends import DataFailure, Endpoint, IntegrityFailure, PermanentFailure, ServiceFailure, atomic_json, audit_attempts, durable_call, immutable_json, load_json, now
from b029_stimuli import ROOT, design, digest, models, prompt, runtime_config, settle, validate_spec
from b029_validate import prompt_hashes
from b029_reuse import ReuseUnavailable, load_phase1_reuse


def session_directory(run_dir: Path, session: Dict[str, Any]) -> Path:
    return Path(run_dir) / "sessions" / session["session_id"]


def _format_number(value: float | int) -> str:
    # B022 inserted Python's compact numeric string; retain its observable format.
    return str(value)


def run_session(endpoint: Endpoint, run_dir: Path, session: Dict[str, Any]) -> str:
    validate_spec(session)
    directory = session_directory(run_dir, session); directory.mkdir(parents=True, exist_ok=True)
    with (directory / "session.lock").open("w", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        immutable_json(directory / "spec.json", session)
        if (directory / "session.json").exists(): return "complete"
        if (directory / "excluded.json").exists(): return "excluded"
        accepted = load_json(directory / "accepted.json", {})
        messages: List[Dict[str, str]] = [{"role": "system", "content": prompt("system")}]
        phase1_events = []; rounds = []; bridge_raw = None; bridge_status = "not_applicable"
        phase1_reuse: Dict[str, Any]
        started = now()

        def call(key: str, text: str, phase: str, decode_seed: int) -> Dict[str, Any]:
            request_messages = messages + [{"role": "user", "content": text}]
            return durable_call(endpoint, directory, accepted, request_messages, key, phase, session, decode_seed)

        try:
            try:
                reused_events, phase1_reuse = load_phase1_reuse(session)
            except ReuseUnavailable as exc:
                phase1_reuse = {
                    "mode": "live_fallback",
                    "reason": str(exc),
                    "source_run_id": None,
                    "source_session_id": None,
                    "exact_session_match": False,
                    "verified_calls": 0,
                }
                reused_events = None
            for index, row in enumerate(session["phase1"]):
                text = prompt("b_phase1_observe", round=row["round"], reported=row["reported"])
                if reused_events is None:
                    event = call(f"phase1_{row['round']:02d}", text, "phase1", session["decode_seeds"]["phase1"][index])
                    phase1_events.append({**row, **event, "reused": False})
                else:
                    phase1_events.append(reused_events[index])
                messages.extend([
                    {"role": "user", "content": text},
                    {"role": "assistant", "content": '{"acknowledged":true}'},
                    {"role": "user", "content": prompt("b_phase1_feedback", actual=row["actual"], label="TRUE" if row["truthful"] else "FALSE")},
                ])
            if session["bridge"] == "bridge_0":
                messages.append({"role": "user", "content": prompt("bridge_0")})
            elif session["bridge"] == "bridge_1":
                text = prompt("bridge_1_request")
                event = call("bridge", text, "bridge_1", session["decode_seeds"]["bridge"])
                bridge_raw = event["raw_model_output"]; bridge_status = event["parse_status"]
                messages.extend([{"role": "user", "content": text}, {"role": "assistant", "content": bridge_raw}, {"role": "user", "content": prompt("adjective_transition")}])
            else:
                text = prompt("bridge_3_adjective_direct_request")
                event = call("bridge", text, "bridge_3", session["decode_seeds"]["bridge"])
                bridge_raw = event["raw_model_output"]; bridge_status = event["parse_status"]
                messages.extend([{"role": "user", "content": text}, {"role": "assistant", "content": bridge_raw}, {"role": "user", "content": prompt("adjective_transition")}])
            messages.append({"role": "user", "content": prompt("phase2_intro")})
            for round_no, return_rate in enumerate(session["return_rates"], 1):
                text = prompt("phase2_choice", round=round_no)
                event = call(f"phase2_{round_no:02d}", text, "phase2", session["decode_seeds"]["phase2"][round_no - 1])
                investment = int(event["parsed"]); economics = settle(investment, return_rate)
                rounds.append({
                    "run_id": Path(run_dir).name, "session_id": session["session_id"], "trial_id": session["trial_id"],
                    "model_name": session["model_name"], "bridge": session["bridge"], "honesty": session["honesty"],
                    "honesty_value": session["honesty_value"], "return_condition": session["return_condition"],
                    "round": round_no, **economics, "raw_model_output": event["raw_model_output"],
                    "parse_status": event["parse_status"], "decode_seed": session["decode_seeds"]["phase2"][round_no - 1],
                    "effective_decode_seed": event.get("effective_decode_seed"), "format_retry_count": event["format_retry_count"],
                    "request_retry_count": event["request_retry_count"],
                })
                messages.extend([
                    {"role": "user", "content": text}, {"role": "assistant", "content": str(investment)},
                    {"role": "user", "content": prompt("phase2_feedback", investment=investment, tripled=economics["tripled"], returned=_format_number(economics["returned"]), payoff=_format_number(economics["payoff"]))},
                ])
            audit = audit_attempts(directory)
            result = {
                "run_id": Path(run_dir).name, "session_id": session["session_id"], "trial_id": session["trial_id"],
                "model_name": session["model_name"], "backend": session["backend"], "bridge": session["bridge"],
                "honesty": session["honesty"], "honesty_value": session["honesty_value"], "return_condition": session["return_condition"],
                "stimulus_seed": session["stimulus_seed"], "return_seed": session["return_seed"], "seed_supported": session["seed_supported"],
                "bridge_raw_output": bridge_raw, "bridge_parse_status": bridge_status, "session_status": "valid",
                "format_retry_count": audit["format_retry_count"], "request_retry_count": audit["request_retry_count"],
                "format_first_pass": audit["first_attempt_valid"] == audit["logical_calls_with_output"],
                "start_time": started, "end_time": now(), "phase1": phase1_events, "phase1_reuse": phase1_reuse, "rounds": rounds,
                "messages": messages, "audit": audit, "runtime_profile": endpoint.profile,
                "spec_hash": digest(session), "synthetic": endpoint.synthetic,
            }
            atomic_json(directory / "session.json", result)
            return "complete"
        except DataFailure as exc:
            audit = audit_attempts(directory)
            atomic_json(directory / "excluded.json", {
                "run_id": Path(run_dir).name, "session_id": session["session_id"], **{k: session[k] for k in ("trial_id", "model_name", "backend", "bridge", "honesty", "honesty_value", "return_condition", "stimulus_seed", "return_seed", "seed_supported")},
                "bridge_raw_output": bridge_raw, "bridge_parse_status": bridge_status, "session_status": "excluded",
                "format_retry_count": audit["format_retry_count"], "request_retry_count": audit["request_retry_count"],
                "start_time": started, "end_time": now(), "reason": str(exc), "no_imputation": True,
                "accepted_calls": len(accepted), "audit": audit, "phase1_reuse": phase1_reuse, "messages_at_failure": messages,
            })
            return "excluded"
        except (PermanentFailure, ServiceFailure) as exc:
            atomic_json(directory / "service_failure.json", {"reason": str(exc), "timestamp": now(), "accepted_calls": len(accepted), "phase1_reuse": phase1_reuse, "retryable": isinstance(exc, ServiceFailure)})
            raise


def _jsonl_bytes(items: Iterable[Dict[str, Any]]) -> bytes:
    return b"".join((json.dumps(x, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8") for x in items)


def _atomic_bytes(path: Path, value: bytes) -> None:
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temporary.open("wb") as handle: handle.write(value); handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)


def materialize_raw(run_dir: Path) -> Dict[str, Any]:
    """Rebuild aggregate raw tables from immutable per-session source files."""
    run_dir = Path(run_dir); lock_path = run_dir / "materialize.lock"
    with lock_path.open("w", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        sessions = [load_json(path) for path in sorted((run_dir / "sessions").glob("*/session.json"))]
        excluded = [load_json(path) for path in sorted((run_dir / "sessions").glob("*/excluded.json"))]
        phase2_rounds = int(design()["phase2_rounds"])
        if any(len(item.get("rounds", [])) != phase2_rounds for item in sessions):
            raise IntegrityFailure(f"valid session does not contain exactly {phase2_rounds} rounds")
        session_rows = [{k: v for k, v in item.items() if k not in ("rounds", "messages", "phase1")} for item in sessions]
        _atomic_bytes(run_dir / "raw_sessions.jsonl", _jsonl_bytes(session_rows))
        _atomic_bytes(run_dir / "conversations.jsonl", _jsonl_bytes({"run_id": x["run_id"], "session_id": x["session_id"], "messages": x["messages"], "phase1": x["phase1"]} for x in sessions))
        _atomic_bytes(run_dir / "excluded.jsonl", _jsonl_bytes(excluded))
        rounds = [row for item in sessions for row in item["rounds"]]
        csv_path = run_dir / "raw_rounds.csv"
        temporary = csv_path.with_name(csv_path.name + "." + uuid.uuid4().hex + ".tmp")
        if rounds:
            with temporary.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rounds[0])); writer.writeheader(); writer.writerows(rounds)
                handle.flush(); os.fsync(handle.fileno())
            os.replace(temporary, csv_path)
        else:
            _atomic_bytes(csv_path, b"")
        parquet_status = "not_written"
        try:
            import pandas as pd
            frame = pd.DataFrame(rounds)
            parquet_tmp = run_dir / ("raw_rounds." + uuid.uuid4().hex + ".parquet.tmp")
            frame.to_parquet(parquet_tmp, index=False)
            os.replace(parquet_tmp, run_dir / "raw_rounds.parquet"); parquet_status = "written"
            unavailable = run_dir / "raw_rounds.parquet.unavailable.json"
            if unavailable.exists(): unavailable.unlink()
        except (ImportError, ValueError, OSError) as exc:
            atomic_json(run_dir / "raw_rounds.parquet.unavailable.json", {"reason": f"{type(exc).__name__}: {exc}", "csv_fallback": "raw_rounds.csv", "timestamp": now()})
            parquet_status = "csv_fallback"
        return {"sessions": len(sessions), "excluded": len(excluded), "rounds": len(rounds), "parquet": parquet_status}


def progress(run_dir: Path, target_specs: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    specs = list(target_specs); completed = excluded = failed = 0
    by_model: Dict[str, Dict[str, int]] = {}
    for spec in specs:
        directory = session_directory(run_dir, spec); status = "missing"
        if (directory / "session.json").exists(): completed += 1; status = "valid"
        elif (directory / "excluded.json").exists(): excluded += 1; status = "excluded"
        elif (directory / "service_failure.json").exists(): failed += 1; status = "service_failure"
        bucket = by_model.setdefault(spec["model_name"], {"target": 0, "valid": 0, "excluded": 0, "service_failure": 0, "missing": 0})
        bucket["target"] += 1; bucket[status] += 1
    value = {"timestamp": now(), "target": len(specs), "valid": completed, "excluded": excluded, "service_failure": failed, "missing": len(specs) - completed - excluded, "by_model": by_model}
    atomic_json(Path(run_dir) / "progress.json", value); return value


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_run(run_dir: Path, selected_models: List[str], trials: int, synthetic: bool, dry_run: bool = False, target_sessions: int | None = None) -> None:
    run_dir = Path(run_dir); run_dir.mkdir(parents=True, exist_ok=True)
    shared_root = ROOT.parents[1] / "src" / "release_common"
    immutable_json(run_dir / "shared_source_hashes.json", {
        path.name: _hash_file(path) for path in sorted(shared_root.glob("*.py"))
    })
    snapshot = run_dir / "snapshot"
    if not snapshot.exists():
        for folder in ("scripts", "conditions", "prompts"):
            shutil.copytree(ROOT / folder, snapshot / folder, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        for name in ("README.md", "B029_实验设计.md", "B029_中文实验报告分析清单.md"):
            if (ROOT / name).exists(): shutil.copy2(ROOT / name, snapshot / name)
    source_hashes = {str(path.relative_to(snapshot)): _hash_file(path) for path in sorted(snapshot.rglob("*")) if path.is_file()}
    immutable_json(run_dir / "source_hashes.json", source_hashes)
    # The executable root must remain byte-identical to the frozen snapshot.
    # This lets resume use ordinary imports without silently changing protocol.
    for relative, expected_hash in source_hashes.items():
        current = ROOT / relative
        if not current.exists() or _hash_file(current) != expected_hash:
            raise IntegrityFailure(f"current source differs from frozen snapshot: {relative}")
    d = design(); expected = int(target_sessions if target_sessions is not None else len(selected_models) * len(d["bridges"]) * len(d["return_conditions"]) * len(d["honesty_levels"]) * trials)
    existing_spec = load_json(run_dir / "spec.json", {})
    spec = {
        "run_id": run_dir.name, "created_at": existing_spec.get("created_at", now()), "synthetic": synthetic, "dry_run": dry_run, "models": selected_models, "trials": trials,
        "target_sessions": expected, "target_rounds": expected * d["phase2_rounds"], "full_named_design_target_sessions": d["model_count_resolution"]["executable_sessions"],
        "instruction_claimed_target_sessions": d["model_count_resolution"]["instruction_claimed_sessions"], "model_count_resolution": d["model_count_resolution"],
        "temperature": d["temperature"], "thinking": d["thinking"], "reasoning_effort": d["reasoning_effort"],
        "top_p": d["top_p"], "max_tokens": d["max_tokens"],
        "format_attempts": runtime_config()["format_attempts"], "request_attempts": runtime_config()["request_attempts"],
        "bootstrap_seed": d["bootstrap_seed"], "bootstrap_samples": d["bootstrap_samples"], "gpu": runtime_config()["gpu"], "cpu_affinity": runtime_config()["cpu_affinity"],
        "prompt_hashes": prompt_hashes(), "design_hash": _hash_file(ROOT / "conditions" / "design.json"), "seed_manifest_hash": _hash_file(ROOT / "conditions" / "seeds.json"),
    }
    immutable_json(run_dir / "spec.json", spec)
    packages = {}
    for package in ("httpx", "numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "vllm", "transformers", "pyarrow"):
        try: packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: packages[package] = None
    def output(command: List[str]) -> str | None:
        try: return subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.SubprocessError): return None
    existing_environment = load_json(run_dir / "environment.json", {})
    model_paths = {name: (str(Path(runtime_config()["model_root"]) / models()[name]["id"]) if models()[name]["backend"] == "vllm" else None) for name in selected_models}
    model_metadata_hashes = {}
    for name, path_text in model_paths.items():
        if path_text:
            path = Path(path_text)
            model_metadata_hashes[name] = {item.name: _hash_file(item) for item in sorted(path.glob("*.json"))}
    environment = {
        "timestamp": existing_environment.get("timestamp", now()), "python": sys.version, "platform": platform.platform(), "packages": packages,
        "git_commit": output(["git", "-C", runtime_config()["project"], "rev-parse", "HEAD"]),
        "cuda_version": output(["nvcc", "--version"]) or output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]),
        "gpu_name": output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]),
        "model_specs": {name: models()[name] for name in selected_models},
        "model_paths": model_paths, "model_metadata_hashes": model_metadata_hashes,
        "cache_root": runtime_config()["cache_root"],
    }
    immutable_json(run_dir / "environment.json", environment)
