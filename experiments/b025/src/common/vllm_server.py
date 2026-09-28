from __future__ import annotations

import json
import os
import signal
import subprocess
import time
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from src.common.config import B025_ROOT, model_config, runtime_config
from src.common.resource_guard import assert_gpu_launch_allowed

TP_SIZE = {
    "gemma-4-31b-it": 2,
    "qwen3-14b": 1,
    "qwen3-32b": 2,
    "qwen3.5-4b": 1,
    "qwen3.5-9b": 1,
    "qwen3.5-27b": 1,
    "qwen3.8-27b": 1,
}


def _ready(endpoint: str, actual_id: str) -> bool:
    try:
        with urllib.request.urlopen(endpoint + "/models", timeout=5) as response:
            payload = json.loads(response.read())
        return any(item.get("id") == actual_id for item in payload.get("data", []))
    except Exception:
        return False


def _short_tmp_path() -> str:
    target = (B025_ROOT / "tmp").resolve()
    target.mkdir(parents=True, exist_ok=True)
    link = Path(os.environ.get("B025_TMP_LINK", "/tmp/b025_release"))
    if os.path.lexists(link):
        if not link.is_symlink() or link.resolve() != target:
            raise RuntimeError(f"short vLLM temp path is unsafe: {link}")
    else:
        link.symlink_to(target, target_is_directory=True)
    return str(link)


@contextmanager
def serve_model(
    model_key: str,
    run_id: str,
    port: int = 8000,
    physical_gpus: list[int] | tuple[int, ...] | None = None,
) -> Iterator[str]:
    runtime = runtime_config()
    spec = model_config()["models"][model_key]
    if spec["backend"] != "vllm":
        raise ValueError(f"{model_key} is not a local vLLM model")
    tp = TP_SIZE[model_key]
    allowed = [int(index) for index in runtime["allowed_gpus"]]
    physical = list(physical_gpus) if physical_gpus is not None else allowed[:tp]
    if len(physical) != tp:
        raise ValueError(
            f"{model_key} requires {tp} GPUs, received {len(physical)}: {physical}"
        )
    if len(set(physical)) != len(physical):
        raise ValueError(f"Duplicate physical GPU allocation: {physical}")
    if not set(physical).issubset(set(allowed)):
        raise ValueError(f"GPU allocation {physical} is outside allowed set {allowed}")
    assert_gpu_launch_allowed(tuple(physical))
    endpoint = f"http://localhost:{port}/v1"
    service_dir = B025_ROOT / "runs" / run_id / "services"
    service_dir.mkdir(parents=True, exist_ok=True)
    log_path = service_dir / f"{model_key}.log"
    pid_path = service_dir / f"{model_key}.pid"
    command = [
        runtime["local_vllm"]["python"],
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        spec["checkpoint"],
        "--served-model-name",
        spec["actual_id"],
        "--host",
        "localhost",
        "--port",
        str(port),
        "--tensor-parallel-size",
        str(tp),
        "--dtype",
        runtime["local_vllm"]["dtype"],
        "--max-model-len",
        str(runtime["local_vllm"]["max_model_len"]),
        "--gpu-memory-utilization",
        str(runtime["local_vllm"]["gpu_memory_utilization"]),
        "--max-num-seqs",
        str(runtime["local_vllm"]["max_num_seqs"]),
        "--max-num-batched-tokens",
        str(runtime["local_vllm"]["max_num_batched_tokens"]),
        "--generation-config",
        "vllm",
        "--enable-prefix-caching",
        "--no-enable-log-requests",
    ]
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(index) for index in physical)
    env["HF_HOME"] = str(B025_ROOT / "cache" / "huggingface")
    env["OMP_NUM_THREADS"] = str(runtime["local_vllm"]["cpu_threads_per_model"])
    env["MKL_NUM_THREADS"] = str(runtime["local_vllm"]["cpu_threads_per_model"])
    env["TOKENIZERS_PARALLELISM"] = "true"
    env["XDG_CACHE_HOME"] = str(B025_ROOT / "cache" / "xdg")
    # vLLM uses AF_UNIX sockets and Linux limits their path to 107 bytes.
    # This short symlink still stores all socket files in B025/tmp.
    env["TMPDIR"] = _short_tmp_path()
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            env=env,
            cwd=B025_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    pid_path.write_text(str(process.pid) + "\n", encoding="utf-8")
    deadline = time.monotonic() + runtime["local_vllm"]["startup_timeout_seconds"]
    try:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"vLLM exited with code {process.returncode}; see {log_path}")
            if _ready(endpoint, spec["actual_id"]):
                yield endpoint + "/chat/completions"
                return
            time.sleep(5)
        raise TimeoutError(f"vLLM startup timed out; see {log_path}")
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=90)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=30)
        pid_path.unlink(missing_ok=True)
