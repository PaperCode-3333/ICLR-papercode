"""B028 model backends with durable, auditable retries and owned-server safety."""
from __future__ import annotations

import contextlib
import gzip
import json
import os
import signal
import socket
import subprocess
import threading
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import httpx

from b028_stimuli import canonical, design, digest, models, parse_output, retry_decode_seed, runtime_config
from release_common.io import now, atomic_json, load_json
from release_common.config import environment_credentials as _credentials
import sys


class IntegrityFailure(RuntimeError): pass
class DataFailure(RuntimeError): pass
class ServiceFailure(RuntimeError): pass
class PermanentFailure(RuntimeError): pass








def immutable_json(path: Path, value: Any) -> None:
    if path.exists():
        if load_json(path) != value:
            raise IntegrityFailure(f"immutable object mismatch: {path}")
    else:
        atomic_json(path, value)




def _extract_response(response: Dict[str, Any], phase: str, model_spec: Dict[str, Any], synthetic: bool) -> Dict[str, Any]:
    choices = response.get("choices") or []
    choice = choices[0] if choices else {}
    message = choice.get("message") or {}
    raw = message.get("content")
    provider_model = response.get("model")
    if not synthetic and provider_model != model_spec["model"]:
        raise PermanentFailure(f"provider_identity_mismatch:{provider_model}")
    reasoning = bool(message.get("reasoning_content") or message.get("reasoning") or message.get("reasoning_details") or message.get("thinking"))
    reasoning |= bool(((response.get("usage") or {}).get("completion_tokens_details") or {}).get("reasoning_tokens"))
    reasoning |= isinstance(raw, str) and any(marker in raw.lower() for marker in ("<think", "</think", "<analysis", "[channel]analysis", "[start]thought"))
    parsed = parse_output(raw, phase)
    if reasoning or choice.get("finish_reason") in ("length", "content_filter"):
        parsed = None
    return {
        "raw_model_output": raw,
        "parsed": parsed,
        "parse_status": "valid" if parsed is not None else "invalid",
        "reasoning_detected": reasoning,
        "finish_reason": choice.get("finish_reason"),
        "provider_model": provider_model,
        "system_fingerprint": response.get("system_fingerprint"),
        "usage": response.get("usage") or {},
    }


_RATE_LOCK = threading.Lock()
_LAST_REQUEST: Dict[str, float] = {}


class Endpoint:
    def __init__(self, model_name: str, url: str | None = None, synthetic: bool = False):
        self.model_name = model_name
        self.spec = models()[model_name]
        self.backend = self.spec["backend"]
        self.synthetic = synthetic
        self.profile = "synthetic" if synthetic else None
        credentials = _credentials()
        if self.backend == "vllm":
            self.url = url or os.getenv("B028_VLLM_URL", "")
            self.key = ""
        else:
            default = "https://api.deepseek.com/v1/chat/completions"
            self.url = url or os.getenv("B028_DEEPSEEK_URL", credentials.get("B022_DEEPSEEK_URL", default))
            self.key = os.getenv("DEEPSEEK_API_KEY", credentials.get("DEEPSEEK_API_KEY", ""))

    def body(self, messages: List[Dict[str, str]], phase: str, decode_seed: int, format_retry: int) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "model": self.spec["model"],
            "messages": messages,
            "temperature": design()["temperature"],
            "top_p": design()["top_p"],
            "max_tokens": design()["max_tokens"],
            "stream": False,
        }
        if phase == "phase2":
            body["stop"] = ["\n"]
        if self.backend == "vllm":
            body["chat_template_kwargs"] = {"enable_thinking": False}
        else:
            body["thinking"] = {"type": "disabled"}
        if self.spec["seed_supported"]:
            body["seed"] = retry_decode_seed(decode_seed, format_retry)
        return body

    def post(self, body: Dict[str, Any], phase: str, session: Dict[str, Any]) -> tuple[Dict[str, Any], int]:
        if self.synthetic:
            h = float(session["honesty_value"])
            bridge_shift = {"bridge_0": 0.0, "bridge_1": 0.3, "bridge_3": 0.8}[session["bridge"]]
            if phase == "phase1": raw = '{"acknowledged":true}'
            elif phase == "bridge_1": raw = "blue"
            elif phase == "bridge_3": raw = "trustworthy" if h >= 0.5 else "untrustworthy"
            elif phase == "caution": raw = "MORE" if h < 0.5 else "LESS"
            else:
                text = body["messages"][-1]["content"]
                round_no = int(text.split("round ", 1)[1].split(" of", 1)[0])
                target = 4.0 + 4.0 * h + bridge_shift * (2.0 * h - 1.0)
                start = 2.0 + 6.0 * h + 1.5 * bridge_shift * (2.0 * h - 1.0)
                value = target + (start - target) * (0.72 ** (round_no - 1))
                raw = str(max(1, min(10, int(round(value)))))
            return {"model": "SYNTHETIC_TEST_ONLY", "choices": [{"message": {"content": raw}, "finish_reason": "stop"}]}, 200
        if self.backend != "vllm" and not self.key:
            raise PermanentFailure("missing_api_key")
        if self.backend != "vllm":
            interval = float(runtime_config()["deepseek_request_interval"])
            with _RATE_LOCK:
                delay = interval - (time.monotonic() - _LAST_REQUEST.get(self.backend, 0.0))
                if delay > 0: time.sleep(delay)
                _LAST_REQUEST[self.backend] = time.monotonic()
        headers = {"Content-Type": "application/json"}
        if self.key: headers["Authorization"] = "Bearer " + self.key
        timeout = httpx.Timeout(float(runtime_config()["timeout_seconds"]), connect=20)
        with httpx.Client(trust_env=False, timeout=timeout) as client:
            result = client.post(self.url, headers=headers, json=body)
        text = result.text.replace(self.key, "[REDACTED]") if self.key else result.text
        try: payload = json.loads(text)
        except ValueError: payload = {"non_json_response": text[:2000]}
        return payload, result.status_code


def _save_attempt(path: Path, value: Dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as raw_handle:
        with gzip.GzipFile(fileobj=raw_handle, mode="wb") as zipped:
            zipped.write(canonical(value).encode("utf-8"))
        raw_handle.flush(); os.fsync(raw_handle.fileno())
    os.replace(temporary, path)


def audit_attempts(directory: Path) -> Dict[str, Any]:
    counts: Counter[str] = Counter(); first: Dict[str, str] = {}
    for path in sorted((directory / "attempts").glob("*.json.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as handle: attempt = json.load(handle)
        counts[attempt["status"]] += 1
        if attempt["status"] in ("accepted", "invalid_output"):
            first.setdefault(attempt["logical_call"], attempt["status"])
    return {
        "attempt_counts": dict(counts),
        "logical_calls_with_output": len(first),
        "first_attempt_valid": sum(status == "accepted" for status in first.values()),
        "format_retry_count": counts["invalid_output"],
        "request_retry_count": counts["transport_error"] + counts["provider_error"],
    }


def durable_call(endpoint: Endpoint, directory: Path, accepted: Dict[str, Any], messages: List[Dict[str, str]], logical_call: str, phase: str, session: Dict[str, Any], decode_seed: int) -> Dict[str, Any]:
    prompt_hash = digest({"messages": messages, "phase": phase, "model": endpoint.spec["model"], "temperature": design()["temperature"]})
    if logical_call in accepted:
        if accepted[logical_call]["prompt_hash"] != prompt_hash:
            raise IntegrityFailure(f"cached request hash mismatch: {logical_call}")
        return accepted[logical_call]
    attempts_dir = directory / "attempts"; attempts_dir.mkdir(exist_ok=True)
    previous = sorted(attempts_dir.glob(logical_call + "_*.json.gz"))
    invalid_count = request_errors = 0
    for path in previous:
        with gzip.open(path, "rt", encoding="utf-8") as handle: item = json.load(handle)
        if item["prompt_hash"] != prompt_hash:
            raise IntegrityFailure(f"previous attempt hash mismatch: {logical_call}")
        if item["status"] == "accepted":
            accepted[logical_call] = item["accepted"]
            atomic_json(directory / "accepted.json", accepted)
            return accepted[logical_call]
        invalid_count += item["status"] == "invalid_output"
        # Transport/provider errors are retained for audit but each supervisor resume receives a fresh retry budget.
    if invalid_count >= int(runtime_config()["format_attempts"]):
        raise DataFailure(f"format_attempts_exhausted:{logical_call}")
    next_index = max([int(x.name.split("_")[-1].split(".")[0]) for x in previous], default=-1) + 1
    while invalid_count < int(runtime_config()["format_attempts"]) and request_errors < int(runtime_config()["request_attempts"]):
        index = next_index; next_index += 1
        body = endpoint.body(messages, phase, decode_seed, invalid_count)
        attempt: Dict[str, Any] = {
            "logical_call": logical_call,
            "phase": phase,
            "attempt_index": index,
            "format_attempt_index": invalid_count,
            "request_error_index": request_errors,
            "timestamp": now(),
            "request": body,
            "prompt_hash": prompt_hash,
            "request_hash": digest(body),
            "decode_seed": decode_seed,
            "effective_decode_seed": body.get("seed"),
            "seed_supported": bool(endpoint.spec["seed_supported"]),
            "endpoint": endpoint.url,
            "synthetic": endpoint.synthetic,
        }
        started = time.monotonic(); permanent = None
        try:
            payload, status = endpoint.post(body, phase, session)
            attempt.update(response=payload, http_status=status)
            provider_code = (payload.get("base_resp") or {}).get("status_code", 0) if isinstance(payload, dict) else 0
            error_obj = payload.get("error") if isinstance(payload, dict) else None
            if status in (401, 403, 404):
                raise PermanentFailure(f"provider_http_error:{status}")
            if status != 200 or provider_code or error_obj or not (payload.get("choices") if isinstance(payload, dict) else None):
                attempt.update(status="provider_error", provider_code=provider_code, error_message=str(error_obj)[:500] if error_obj else None, retryable=True)
                request_errors += 1
            else:
                fields = _extract_response(payload, phase, endpoint.spec, endpoint.synthetic)
                attempt["status"] = "accepted" if fields["parse_status"] == "valid" else "invalid_output"
                if attempt["status"] == "accepted":
                    event = {
                        **fields,
                        "prompt_hash": prompt_hash,
                        "request_hash": attempt["request_hash"],
                        "attempt_file": f"attempts/{logical_call}_{index:06d}.json.gz",
                        "decode_seed": decode_seed,
                        "effective_decode_seed": body.get("seed"),
                        "format_retry_count": invalid_count,
                        "request_retry_count": request_errors,
                        "timestamp": now(),
                    }
                    attempt["accepted"] = event
                else:
                    invalid_count += 1
        except PermanentFailure as exc:
            permanent = exc; attempt.update(status="permanent_error", error=str(exc))
        except (httpx.HTTPError, OSError, ValueError) as exc:
            request_errors += 1; attempt.update(status="transport_error", error_type=type(exc).__name__, error=str(exc)[:500])
        attempt["elapsed_seconds"] = time.monotonic() - started
        _save_attempt(attempts_dir / f"{logical_call}_{index:06d}.json.gz", attempt)
        if permanent: raise permanent
        if attempt["status"] == "accepted":
            accepted[logical_call] = attempt["accepted"]
            atomic_json(directory / "accepted.json", accepted)
            return accepted[logical_call]
        if attempt["status"] in ("transport_error", "provider_error") and not endpoint.synthetic:
            time.sleep(min(60, 2 ** min(request_errors, 6)))
    if invalid_count >= int(runtime_config()["format_attempts"]):
        raise DataFailure(f"format_attempts_exhausted:{logical_call}")
    raise ServiceFailure(f"request_attempts_exhausted:{logical_call}")


def _b023_active() -> bool:
    # Only actual B023 runners reserve GPUs; read-only scans do not.
    try:
        output = subprocess.check_output(['ps', '-eo', 'args'], text=True)
    except (OSError, subprocess.SubprocessError):
        return True
    runner_markers = ('orchestrate_b023.py', 'run_b023.py', 'expand_b023.py')
    return any(any(marker in line for marker in runner_markers) for line in output.splitlines())


class LocalServer:
    """Owns exactly one B028 vLLM process and never touches external PIDs."""
    def __init__(self, model_name: str, run_dir: Path, gpu: int | None = None, port: int | None = None, cpu_affinity: str | None = None):
        self.model_name = model_name; self.spec = models()[model_name]
        self.run_dir = Path(run_dir); self.process: subprocess.Popen | None = None; self.log_handle = None
        config = runtime_config()
        self.gpu = int(config["gpu"] if gpu is None else gpu)
        self.port = int(config["port"] if port is None else port)
        self.cpu_affinity = str(config["cpu_affinity"] if cpu_affinity is None else cpu_affinity)

    def __enter__(self) -> Endpoint:
        try: return self.start()
        except BaseException:
            self.stop(); raise

    def start(self) -> Endpoint:
        config = runtime_config(); port = self.port
        if _b023_active():
            raise ServiceFailure("B023_active_GPU_use_is_forbidden")
        with socket.socket() as check:
            if check.connect_ex(("localhost", port)) == 0:
                raise PermanentFailure(f"B028_port_in_use:{port}")
        memory = [int(v) for v in subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True).split()]
        visible_text = str(self.gpu)
        try:
            visible_gpus = [int(value.strip()) for value in visible_text.split(",") if value.strip()]
            tensor_parallel_size = 1
        except ValueError as exc:
            raise PermanentFailure("invalid_B028_multi_gpu_configuration") from exc
        if not visible_gpus or tensor_parallel_size != len(visible_gpus):
            raise PermanentFailure("tensor_parallel_size_must_match_visible_GPU_count")
        if any(gpu < 0 or gpu >= len(memory) for gpu in visible_gpus):
            raise PermanentFailure(f"GPU_index_out_of_range:{visible_text}")
        busy = {gpu: memory[gpu] for gpu in visible_gpus if memory[gpu] > int(config["gpu_busy_threshold_mib"])}
        if busy:
            raise ServiceFailure("GPU_busy:" + ",".join(f"GPU{gpu}={used}MiB" for gpu, used in busy.items()))
        model_path = Path(config["model_root"]) / self.spec["id"]
        if not model_path.exists(): raise PermanentFailure(f"model_path_missing:{model_path}")
        cache = Path(config["cache_root"]); cache.mkdir(parents=True, exist_ok=True)
        # Keep vLLM's ZeroMQ IPC endpoint below Linux's 107-byte
        # sockaddr_un.sun_path limit.  The project-local cache path is too
        # deep once vLLM appends its UUID.  This remains on the mounted disk,
        # is isolated by run id, and can be overridden operationally.
        tmp_root = Path(config["tmp_root"])
        tmp_dir = tmp_root / self.run_dir.name / f"gpu{self.gpu}"
        environment = os.environ.copy()
        environment.update({
            "CUDA_VISIBLE_DEVICES": visible_text,
            "B028_OWNED_SERVICE": str(self.run_dir),
            "TOKENIZERS_PARALLELISM": "false",
            "HF_HOME": str(cache / "hf"),
            "TRANSFORMERS_CACHE": str(cache / "transformers"),
            "XDG_CACHE_HOME": str(cache / "xdg"),
            "VLLM_CACHE_ROOT": str(cache / "vllm"),
            "TMPDIR": str(tmp_dir),
        })
        for path in (cache / "hf", cache / "transformers", cache / "xdg", cache / "vllm", tmp_dir): path.mkdir(parents=True, exist_ok=True)
        python = os.environ.get("VLLM_PYTHON", sys.executable)
        base = [
            "taskset", "-c", self.cpu_affinity, python, "-m", "vllm.entrypoints.openai.api_server",
            "--model", str(model_path), "--served-model-name", self.spec["model"], "--host", "localhost",
            "--port", str(port), "--tensor-parallel-size", str(tensor_parallel_size), "--dtype", "bfloat16", "--max-model-len", str(config["max_model_len"]),
            "--gpu-memory-utilization", "0.85", "--max-num-seqs", "16", "--max-num-batched-tokens", "4096",
            "--generation-config", "vllm", "--enable-prefix-caching", "--no-enable-log-requests",
        ]
        service_dir = self.run_dir / "services"; service_dir.mkdir(parents=True, exist_ok=True)
        for eager in (False, True):
            self.log_handle = (service_dir / f"{self.model_name}_{time.time_ns()}.log").open("w", encoding="utf-8")
            command = base + (["--enforce-eager"] if eager else [])
            self.process = subprocess.Popen(command, env=environment, stdout=self.log_handle, stderr=subprocess.STDOUT, start_new_session=True)
            owned_name = f"owned_server_{self.model_name}.json"
            atomic_json(service_dir / owned_name, {
                "pid": self.process.pid, "run": str(self.run_dir), "model": self.model_name,
                "port": port, "started": now(), "host": socket.gethostname(),
                "visible_gpus": visible_gpus, "tensor_parallel_size": tensor_parallel_size,
                "cpu_affinity": self.cpu_affinity, "max_model_len": int(config["max_model_len"]),
            })
            deadline = time.monotonic() + float(config["server_startup_timeout_seconds"])
            while time.monotonic() < deadline and self.process.poll() is None:
                try:
                    with httpx.Client(trust_env=False, timeout=5) as client: answer = client.get(f"http://localhost:{port}/v1/models")
                    if answer.status_code == 200 and any(item["id"] == self.spec["model"] for item in answer.json()["data"]):
                        endpoint = Endpoint(self.model_name, f"http://localhost:{port}/v1/chat/completions")
                        endpoint.profile = "eager" if eager else "compiled"
                        return endpoint
                except (httpx.HTTPError, ValueError, KeyError): pass
                time.sleep(5)
            self.stop()
        raise ServiceFailure("vllm_startup_failed")

    def stop(self) -> None:
        # Only the process group created by this object is eligible for termination.
        if self.process:
            with contextlib.suppress(ProcessLookupError): os.killpg(self.process.pid, signal.SIGTERM)
            try: self.process.wait(timeout=40)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError): os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait(timeout=20)
            self.process = None
        if self.log_handle:
            self.log_handle.close(); self.log_handle = None

    def __exit__(self, *args: Any) -> None:
        self.stop()
