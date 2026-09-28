"""Verified reuse of B026 phase-1 interactions for B029."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from b029_stimuli import design, digest, models, prompt, runtime_config, session_spec


class ReuseUnavailable(RuntimeError):
    pass


def source_run() -> Path:
    config = runtime_config()
    return Path(config["b026_run"])


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _events_from_file(path: Path) -> list[dict[str, Any]]:
    data = _read_json(path)
    if path.name == "session.json":
        events = data.get("phase1", [])
    else:
        events = [data.get(f"phase1_{round_no:02d}") for round_no in range(1, 13)]
    if len(events) != 12 or any(not isinstance(item, dict) for item in events):
        raise ReuseUnavailable(f"incomplete phase1 source: {path}")
    return events


def _candidate_files(session: dict[str, Any]) -> list[tuple[Path, bool]]:
    root = source_run() / "sessions"
    exact = root / session["session_id"]
    candidates: list[tuple[Path, bool]] = []
    for name in ("session.json", "accepted.json"):
        path = exact / name
        if path.exists():
            candidates.append((path, True))
    h = int(round(float(session["honesty_value"]) * 100))
    pattern = f"{session['model_name']}__*__*__h{h:03d}__t{int(session['trial_id']):03d}"
    for directory in sorted(root.glob(pattern)):
        if directory == exact:
            continue
        for name in ("session.json", "accepted.json"):
            path = directory / name
            if path.exists():
                candidates.append((path, False))
    return candidates


def _verify_events(session: dict[str, Any], path: Path, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    messages: list[dict[str, str]] = [{"role": "system", "content": prompt("system")}]
    verified: list[dict[str, Any]] = []
    model_id = models()[session["model_name"]]["model"]
    for index, (row, event) in enumerate(zip(session["phase1"], events), 1):
        text = prompt("b_phase1_observe", round=row["round"], reported=row["reported"])
        request_messages = messages + [{"role": "user", "content": text}]
        expected_hash = digest({
            "messages": request_messages,
            "phase": "phase1",
            "model": model_id,
            "temperature": design()["temperature"],
        })
        if event.get("prompt_hash") != expected_hash:
            raise ReuseUnavailable(f"phase1 prompt hash mismatch at round {index}: {path}")
        if event.get("decode_seed") != session["decode_seeds"]["phase1"][index - 1]:
            raise ReuseUnavailable(f"phase1 decode seed mismatch at round {index}: {path}")
        if event.get("parse_status") != "valid" or event.get("parsed") is not True:
            raise ReuseUnavailable(f"phase1 event not accepted at round {index}: {path}")
        for key in ("round", "actual", "reported", "truthful"):
            if key in event and event[key] != row[key]:
                raise ReuseUnavailable(f"phase1 material mismatch for {key} at round {index}: {path}")
        verified.append({
            **row,
            **event,
            "reused": True,
            "source_run_id": source_run().name,
            "source_session_id": path.parent.name,
            "source_file": str(path),
        })
        messages.extend([
            {"role": "user", "content": text},
            {"role": "assistant", "content": '{"acknowledged":true}'},
            {"role": "user", "content": prompt(
                "b_phase1_feedback",
                actual=row["actual"],
                label="TRUE" if row["truthful"] else "FALSE",
            )},
        ])
    return verified


def load_phase1_reuse(session: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    failures: list[str] = []
    for path, exact in _candidate_files(session):
        try:
            events = _verify_events(session, path, _events_from_file(path))
            metadata = {
                "mode": "reused_exact_session" if exact else "reused_family_fallback",
                "source_run_id": source_run().name,
                "source_session_id": path.parent.name,
                "source_file": str(path),
                "source_file_sha256": _sha256(path),
                "exact_session_match": exact,
                "verified_calls": 12,
                "events_digest": digest(events),
            }
            return events, metadata
        except (OSError, ValueError, KeyError, json.JSONDecodeError, ReuseUnavailable) as exc:
            failures.append(f"{path}:{type(exc).__name__}:{exc}")
    raise ReuseUnavailable("; ".join(failures[:8]) or "no B026 phase1 source candidate")


def coverage_audit(validate_events: bool = False, output: Path | None = None) -> dict[str, Any]:
    run = source_run()
    raw = run / "raw_sessions.jsonl"
    families: set[tuple[str, str, int]] = set()
    exact_valid = 0
    with raw.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            families.add((row["model_name"], row["honesty"], int(row["trial_id"])))
            exact_valid += 1
    expected = {
        (model, honesty, trial)
        for model in design()["models"]
        for honesty in ("0%", "25%", "75%", "100%")
        for trial in range(int(design()["model_trials"]))
    }
    missing = sorted(expected - families)
    validation_failures: list[str] = []
    if validate_events:
        for model, honesty, trial in sorted(expected):
            spec = session_spec(model, "bridge_0", "fixed", honesty, trial)
            try:
                load_phase1_reuse(spec)
            except ReuseUnavailable as exc:
                validation_failures.append(f"{model}:{honesty}:{trial}:{exc}")
    result = {
        "status": "pass" if not missing and not validation_failures else "fail",
        "source_run": str(run),
        "source_raw_sessions_sha256": _sha256(raw),
        "expected_families": len(expected),
        "covered_families": len(families & expected),
        "missing_families": len(missing),
        "valid_source_sessions": exact_valid,
        "validated_family_sources": len(expected) if validate_events else 0,
        "validation_failures": validation_failures[:50],
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result
