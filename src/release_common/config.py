"""Resolve deployment paths without embedding host-specific configuration."""
import os
import sys
import tempfile
from pathlib import Path

def environment_credentials():
    return dict(os.environ)

def resolve_path(value, base):
    path = Path(os.path.expandvars(str(value))).expanduser()
    return str((path if path.is_absolute() else Path(base) / path).resolve())

def resolve_runtime(config, root):
    config = dict(config)
    env_names = {
        "project": "RELEASE_ROOT", "project_root": "RELEASE_ROOT",
        "b029_root": "B029_ROOT", "b028_root": "B028_ROOT", "b025_root": "B025_ROOT",
        "b026_root": "B026_ROOT", "b024_root": "B024_ROOT", "b023_root": "B023_ROOT",
        "model_root": "MODEL_ROOT", "cache_root": "EXPERIMENT_CACHE_ROOT",
    }
    for key, env_name in env_names.items():
        if key in config:
            config[key] = resolve_path(os.environ.get(env_name, config[key]), root)
    if "b026_run_id" in config:
        config["b026_run_id"] = os.environ.get("B026_RUN_ID", config["b026_run_id"])
    if "b026_run" in config:
        config["b026_run"] = resolve_path(os.environ.get("B026_RUN", config["b026_run"]), root)
    if "tmp_root" in config:
        config["tmp_root"] = os.environ.get("EXPERIMENT_TMP_ROOT", str(Path(tempfile.gettempdir()) / root.name))
    affinity = ",".join(map(str, sorted(os.sched_getaffinity(0)))) if hasattr(os, "sched_getaffinity") else "0"
    if config.get("cpu_affinity") == "auto":
        config["cpu_affinity"] = os.environ.get("EXPERIMENT_CPU_AFFINITY", affinity)
    if config.get("allowed_cpus") == "auto":
        config["allowed_cpus"] = os.environ.get("EXPERIMENT_CPU_AFFINITY", affinity)
    if "gpu_slots" in config:
        config["gpu_slots"] = [dict(s, cpu_affinity=os.environ.get("EXPERIMENT_CPU_AFFINITY", affinity) if s["cpu_affinity"] == "auto" else s["cpu_affinity"]) for s in config["gpu_slots"]]
    for key in ("forbidden_paths", "read_only_inputs"):
        if key in config:
            config[key] = [resolve_path(p, root) for p in config[key]]
    if "local_vllm" in config:
        config["local_vllm"] = dict(config["local_vllm"], python=os.environ.get("VLLM_PYTHON", sys.executable))
    return config
