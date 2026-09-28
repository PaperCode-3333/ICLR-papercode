from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from release_common.config import resolve_runtime, resolve_path

B025_ROOT = Path(os.environ.get("B025_ROOT", Path(__file__).resolve().parents[2])).resolve()


def load_yaml(relative_path: str | Path) -> dict[str, Any]:
    path = B025_ROOT / relative_path
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


def runtime_config() -> dict[str, Any]:
    return resolve_runtime(load_yaml("configs/runtime.yaml"), B025_ROOT)


def model_config() -> dict[str, Any]:
    config = load_yaml("configs/models.yaml")
    config["model_root"] = resolve_path(os.environ.get("MODEL_ROOT", config["model_root"]), B025_ROOT)
    for spec in config["models"].values():
        if spec.get("checkpoint"):
            spec["checkpoint"] = str(Path(config["model_root"]) / Path(spec["checkpoint"]).name)
    return config


def configure_cache_environment() -> dict[str, str]:
    runtime = runtime_config()
    applied: dict[str, str] = {}
    for name, relative in runtime["cache"].items():
        path = (B025_ROOT / relative).resolve()
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)
        applied[name] = str(path)
    return applied
