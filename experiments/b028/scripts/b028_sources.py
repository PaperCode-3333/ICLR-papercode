"""Read-only B026 phase-one/bridge source index for B028."""
from __future__ import annotations
import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable
from b028_stimuli import design, digest, models, runtime_config

def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def _honesty_code(honesty: str) -> str:
    return f"h{int(float(honesty[:-1])):03d}"

def source_path(model: str, bridge: str, honesty: str, trial_id: int) -> Path:
    run = Path(runtime_config()["b026_run"])
    session_id = f"{model}__{bridge}__fixed__{_honesty_code(honesty)}__t{trial_id:03d}"
    return run / "sessions" / session_id / "session.json"

@lru_cache(maxsize=4096)
def load_source(model: str, bridge: str, honesty: str, trial_id: int) -> Dict[str, Any]:
    path = source_path(model, bridge, honesty, trial_id)
    if not path.exists():
        raise FileNotFoundError(f"required B026 source missing: {path}")
    item = json.loads(path.read_text(encoding="utf-8"))
    if item.get("session_status") != "valid" or len(item.get("phase1", [])) != 12:
        raise ValueError(f"invalid B026 source session: {path}")
    messages = item.get("messages", [])
    if len(messages) < 40:
        raise ValueError(f"incomplete B026 source conversation: {path}")
    prefix = messages[:37]
    if len(prefix) != 37 or prefix[0].get("role") != "system":
        raise ValueError(f"invalid B026 phase1 prefix: {path}")
    return {
        "path": str(path),
        "file_sha256": _sha(path),
        "session_id": item["session_id"],
        "phase1": item["phase1"],
        "phase1_messages": prefix,
        "phase1_prefix_hash": digest(prefix),
        "bridge_raw_output": item.get("bridge_raw_output"),
        "bridge_parse_status": item.get("bridge_parse_status"),
        "bridge_messages": messages[37:40] if bridge in ("bridge_1", "bridge_3") else [],
    }

def source_bundle(model: str, honesty: str, trial_id: int, bridge: str) -> Dict[str, Any]:
    phase = load_source(model, "bridge_3", honesty, trial_id)
    chosen = phase if bridge == "bridge_3" else load_source(model, bridge, honesty, trial_id) if bridge == "bridge_1" else phase
    if chosen["phase1_prefix_hash"] != phase["phase1_prefix_hash"]:
        raise ValueError(f"B026 phase1 prefix drift: {model}/{honesty}/{trial_id}")
    adjective = phase["bridge_raw_output"]
    if not isinstance(adjective, str) or not adjective.isalpha():
        raise ValueError(f"invalid B026 adjective source: {model}/{honesty}/{trial_id}")
    return {
        "phase1": phase,
        "bridge": chosen if bridge in ("bridge_1", "bridge_3") else None,
        "matched_adjective": adjective,
    }

def source_audit() -> Dict[str, Any]:
    errors = []
    rows = 0
    prefix_keys = set()
    bridge_keys = set()
    for model in design()["models"]:
        for honesty_row in design()["honesty_levels"]:
            honesty = honesty_row["name"]
            for trial_id in range(int(design()["model_trials"])):
                try:
                    b3 = load_source(model, "bridge_3", honesty, trial_id)
                    color = load_source(model, "bridge_1", honesty, trial_id)
                    if b3["phase1_prefix_hash"] != color["phase1_prefix_hash"]:
                        errors.append(f"prefix_mismatch:{model}:{honesty}:{trial_id}")
                    if color["bridge_parse_status"] != "valid" or b3["bridge_parse_status"] != "valid":
                        errors.append(f"bridge_invalid:{model}:{honesty}:{trial_id}")
                    prefix_keys.add((model, honesty, trial_id))
                    bridge_keys.add((model, "bridge_1", honesty, trial_id))
                    bridge_keys.add((model, "bridge_3", honesty, trial_id))
                    rows += 1
                except Exception as exc:
                    errors.append(f"{model}:{honesty}:{trial_id}:{type(exc).__name__}:{exc}")
    return {
        "status": "pass" if not errors else "fail",
        "expected_prefix_keys": len(design()["models"]) * len(design()["honesty_levels"]) * int(design()["model_trials"]),
        "prefix_keys": len(prefix_keys),
        "bridge_keys": len(bridge_keys),
        "rows": rows,
        "errors": errors[:100],
    }

def source_files(root: Path) -> Iterable[Path]:
    for folder in ("conditions", "prompts", "scripts"):
        base = root / folder
        if base.exists():
            for path in sorted(base.rglob("*")):
                if path.is_file() and "__pycache__" not in path.parts and path.suffix not in (".pyc", ".orig"):
                    yield path
    for name in ("README.md", "B023_中文实验设计.md", "B026_实验设计.md"):
        path = root / name
        if path.exists():
            yield path

def make_source_manifest() -> Dict[str, Any]:
    roots = {"B023": Path(runtime_config()["b023_root"]), "B026": Path(runtime_config()["b026_root"])}
    files = {}
    for label, root in roots.items():
        for path in source_files(root):
            files[f"{label}/{path.relative_to(root)}"] = _sha(path)
    return {"roots": {k: str(v) for k, v in roots.items()}, "files": files}

def verify_source_manifest(manifest: Dict[str, Any]) -> Dict[str, Any]:
    errors = []
    for key, expected in manifest["files"].items():
        label, relative = key.split("/", 1)
        path = Path(manifest["roots"][label]) / relative
        if not path.exists() or _sha(path) != expected:
            errors.append(key)
    return {"status": "pass" if not errors else "fail", "checked": len(manifest["files"]), "errors": errors}
