from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

from .config import B025_ROOT, runtime_config

PROJECT = Path(runtime_config()["project_root"])
B023 = Path(runtime_config()["b023_root"])


class ResourceGuardError(RuntimeError):
    pass


def _gpu_rows() -> list[dict[str, int]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    rows = []
    for line in completed.stdout.splitlines():
        index, used, free, utilization = [int(field.strip()) for field in line.split(",")]
        rows.append(
            {
                "index": index,
                "memory_used_mib": used,
                "memory_free_mib": free,
                "utilization": utilization,
            }
        )
    return rows


def _pid_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def b023_observation() -> dict[str, Any]:
    runs_root = B023 / "runs"
    candidates = (
        sorted(runs_root.glob("*.pid")) + sorted(runs_root.glob("*/*.pid"))
        if runs_root.exists()
        else []
    )
    pids: list[dict[str, Any]] = []
    for path in candidates:
        try:
            pid = int(path.read_text().strip())
            pids.append({"path": str(path), "pid": pid, "running": _pid_running(pid)})
        except (OSError, ValueError):
            pids.append({"path": str(path), "pid": None, "running": None})
    matching_processes: list[dict[str, Any]] = []
    try:
        completed = subprocess.run(
            ["ps", "-eo", "pid=,args="],
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        )
        for line in completed.stdout.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            pid_text, _, command = stripped.partition(" ")
            if (
                str(B023) in command
                or "orchestrate_b023.py" in command
                or "expand_b023.py" in command
            ):
                matching_processes.append({"pid": int(pid_text), "command": command})
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return {
        "path": str(B023),
        "exists": B023.exists(),
        "pid_files": pids,
        "matching_processes": matching_processes,
    }


def b023_is_active(observation: dict[str, Any] | None = None) -> bool:
    observation = observation or b023_observation()
    return any(item.get("running") for item in observation["pid_files"]) or bool(
        observation["matching_processes"]
    )


def assert_b025_path() -> None:
    expected = runtime_config()["b025_root"]
    if str(B025_ROOT) != str(Path(expected).resolve()):
        raise ResourceGuardError(f"B025 path mismatch: {B025_ROOT} != {expected}")
    if not B025_ROOT.is_dir():
        raise ResourceGuardError(f"B025 root does not exist: {B025_ROOT}")


def _parse_cpu_set(value: str) -> set[int]:
    cpus: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = (int(item) for item in part.split("-", 1))
            cpus.update(range(start, end + 1))
        else:
            cpus.add(int(part))
    return cpus


def assert_cpu_affinity() -> None:
    allowed = _parse_cpu_set(str(runtime_config()["allowed_cpus"]))
    affinity = set(os.sched_getaffinity(0))
    if not affinity.issubset(allowed):
        raise ResourceGuardError(
            f"CPU affinity {sorted(affinity)} is outside {sorted(allowed)}"
        )


def assert_gpu_launch_allowed(
    gpu_indices: list[int] | tuple[int, ...] | None = None,
    max_foreign_memory_mib: int = 1500,
) -> list[dict[str, int]]:
    assert_b025_path()
    runtime = runtime_config()
    allowed = set(runtime["allowed_gpus"])
    requested = set(gpu_indices) if gpu_indices is not None else allowed
    if not requested:
        raise ResourceGuardError("No GPUs requested")
    if not requested.issubset(allowed):
        raise ResourceGuardError(
            f"Requested GPUs {sorted(requested)} are outside {sorted(allowed)}"
        )
    observation = b023_observation()
    if b023_is_active(observation):
        raise ResourceGuardError("B023 still has a running PID; B025 GPU launch is forbidden")
    rows = _gpu_rows()
    forbidden = [
        row
        for row in rows
        if row["index"] in requested
        and row["memory_used_mib"] > max_foreign_memory_mib
    ]
    if forbidden:
        raise ResourceGuardError(f"Requested GPUs are occupied: {json.dumps(forbidden)}")
    return rows
