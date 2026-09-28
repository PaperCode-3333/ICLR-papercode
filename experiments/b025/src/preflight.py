from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dulwich import porcelain
from dulwich.repo import Repo

from src.common.config import B025_ROOT, configure_cache_environment, model_config, runtime_config
from src.common.io import atomic_write_json
from src.common.resource_guard import assert_b025_path, b023_is_active, b023_observation


def command_output(command: list[str]) -> dict[str, Any]:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        return {
            "available": True,
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "error": type(exc).__name__}


def package_versions() -> dict[str, str | None]:
    names = [
        "dulwich",
        "httpx",
        "numpy",
        "pandas",
        "scipy",
        "matplotlib",
        "torch",
        "transformers",
        "datasets",
        "vllm",
    ]
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions




def repo_snapshot(name: str, path: Path) -> dict[str, Any]:
    archive_marker = path / ".b025_source_manifest.json"
    if archive_marker.exists():
        marker = json.loads(archive_marker.read_text(encoding="utf-8"))
        marker.update(
            {"path": str(path), "exists": True, "git_metadata_present": (path / ".git").exists()}
        )
        return marker
    snapshot_marker = path / ".source_snapshot.json"
    if snapshot_marker.exists():
        marker = json.loads(snapshot_marker.read_text(encoding="utf-8"))
        marker.update({"name": name, "path": str(path), "exists": True, "git_repo": False})
        return marker
    if not (path / ".git").exists():
        return {"name": name, "path": str(path), "exists": path.exists(), "git_repo": False}
    try:
        repo = Repo(str(path))
        status = porcelain.status(repo)
        config = repo.get_config()
        remote = None
        try:
            remote = config.get((b"remote", b"origin"), b"url").decode()
        except KeyError:
            pass
        return {
            "name": name,
            "path": str(path),
            "exists": True,
            "git_repo": True,
            "head": repo.head().decode(),
            "remote_url": remote,
            "dirty": bool(status.staged or status.unstaged or status.untracked),
            "staged": {
                str(key): [item.decode(errors="replace") for item in value]
                for key, value in status.staged.items()
            },
            "unstaged": [item.decode(errors="replace") for item in status.unstaged],
            "untracked": [item.decode(errors="replace") for item in status.untracked],
        }
    except Exception as exc:
        return {
            "name": name,
            "path": str(path),
            "exists": path.exists(),
            "git_repo": True,
            "audit_error": type(exc).__name__,
        }


def url_reachable(url: str) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            ["curl", "-I", "-L", "--max-time", "12", "-A", "B025-preflight/1", url],
            capture_output=True,
            text=True,
            timeout=15,
        )
        return {
            "url": url,
            "reachable": completed.returncode == 0,
            "returncode": completed.returncode,
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"url": url, "reachable": False, "error": type(exc).__name__}


def build_manifest() -> dict[str, Any]:
    assert_b025_path()
    cache = configure_cache_environment()
    runtime = runtime_config()
    models = model_config()["models"]
    disk = shutil.disk_usage(B025_ROOT)
    source_urls = {
        "sotopia": "https://github.com/sotopia-lab/sotopia",
        "agentsense": "https://github.com/ljcleo/agent_sense",
        "socket": "https://github.com/minjechoi/SOCKET",
        "socialeval": "https://github.com/thu-coai/SocialEval",
        "ruler": "https://github.com/NVIDIA/RULER",
        "ifeval": "https://github.com/google-research/google-research/tree/master/instruction_following_eval",
        "bbeh": "https://github.com/google-deepmind/bbeh",
    }
    gpu_summary = command_output(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader",
        ]
    )
    gpu_processes = command_output(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader",
        ]
    )
    checkpoints = {
        key: {
            "backend": spec["backend"],
            "actual_id": spec["actual_id"],
            "checkpoint": spec.get("checkpoint"),
            "checkpoint_exists": Path(spec["checkpoint"]).exists()
            if spec.get("checkpoint")
            else None,
        }
        for key, spec in models.items()
    }
    source_reachability = {
        name: {
            "url": url,
            "network_checked": False,
            "local_pinned_source_exists": (B025_ROOT / "benchmark_sources" / name).exists(),
            "note": "Network probing is separated from preflight to avoid blocking local runs.",
        }
        for name, url in source_urls.items()
    }
    b023_state = b023_observation()

    return {
        "schema_version": 1,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "cwd": os.getcwd(),
        "b025_root": str(B025_ROOT),
        "b025_root_exists": B025_ROOT.is_dir(),
        "disk": {"total": disk.total, "used": disk.used, "free": disk.free},
        "python": {"executable": sys.executable, "version": sys.version},
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "allowed_cpus": runtime["allowed_cpus"],
        "commands": {
            "git": command_output(["git", "--version"]),
            "uv": command_output(["uv", "--version"]),
            "nvidia_smi": command_output(["nvidia-smi", "--version"]),
        },
        "packages": package_versions(),
        "gpu_summary": gpu_summary,
        "gpu_processes": gpu_processes,
        "allowed_gpus": runtime["allowed_gpus"],
        "b023_observation": b023_state,
        "b024_exists": Path(runtime["b024_root"]).exists(),
        "b024_readable": Path(runtime["b024_root"]).is_dir(),
        "models": checkpoints,
        "credentials_present": {name: bool(os.environ.get(name)) for name in ("OPENAI_API_KEY", "DEEPSEEK_API_KEY")},
        "cache_environment": cache,
        "source_reachability": source_reachability,
        "source_repositories": {
            name: repo_snapshot(name, B025_ROOT / "benchmark_sources" / name)
            for name in source_urls
        },
        "b025_repository_status": repo_snapshot("b025", B025_ROOT),
        "warnings": [
            (
                "GPU launch remains forbidden while any B023 PID is running "
                "or allowed GPUs are occupied."
            ),
            "Credential booleans record presence only; secret values are never serialized.",
            "B024 is a read-only behavior input; B025 never writes into the B024 tree.",
            (
                "Missing system git is recorded; repository operations use "
                "pinned Dulwich in the B025 environment."
            ),
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id")
    args = parser.parse_args()
    manifest = build_manifest()
    atomic_write_json(B025_ROOT / "manifests/environment.json", manifest)
    if args.run_id:
        atomic_write_json(B025_ROOT / "runs" / args.run_id / "preflight.json", manifest)
    print(
        json.dumps(
            {
                "environment": str(B025_ROOT / "manifests/environment.json"),
                "gpu_gate_open": not b023_is_active(manifest["b023_observation"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
