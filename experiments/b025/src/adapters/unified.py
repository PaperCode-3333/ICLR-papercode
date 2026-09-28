from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from release_common.config import environment_credentials as _credentials

from src.common.config import model_config, runtime_config


class AdapterError(RuntimeError):
    pass


class ProviderIdentityError(AdapterError):
    pass


@dataclass(frozen=True)
class GenerationResult:
    model_key: str
    model_actual_id: str
    raw_response: dict[str, Any]
    text: str
    finish_reason: str | None
    latency_seconds: float
    retry_count: int
    endpoint: str
    request_sha256: str




class UnifiedAdapter:
    def __init__(self, model_key: str, endpoint_override: str | None = None) -> None:
        models = model_config()["models"]
        if model_key not in models:
            raise KeyError(f"Unknown model key: {model_key}")
        self.model_key = model_key
        self.spec = models[model_key]
        self.endpoint = endpoint_override or self.spec.get("endpoint")
        if self.spec["backend"] == "vllm" and not self.endpoint:
            raise AdapterError("Local vLLM adapter requires an explicit endpoint")
        credentials = _credentials()
        key_name = self.spec.get("api_key_env")
        self.api_key = credentials.get(key_name, "") if key_name else ""
        if self.spec["backend"] != "vllm" and not self.api_key:
            raise AdapterError(f"Missing credential environment variable {key_name}")

    def generate(
        self,
        messages: list[dict[str, str]],
        generation_config: dict[str, Any],
        request_id: str,
    ) -> GenerationResult:
        body: dict[str, Any] = {
            "model": self.spec["actual_id"],
            "messages": messages,
            "stream": False,
            **generation_config,
        }
        if self.spec["backend"] == "vllm":
            body.setdefault("chat_template_kwargs", {"enable_thinking": False})
        else:
            body.setdefault("thinking", {"type": "disabled"})
        request_sha = hashlib.sha256(
            str({"request_id": request_id, "body": body}).encode("utf-8")
        ).hexdigest()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        runtime = runtime_config()["api"]
        retryable = {408, 409, 425, 429, 500, 502, 503, 504}
        last_error: Exception | None = None
        for attempt in range(runtime["request_attempts"]):
            start = time.monotonic()
            try:
                with httpx.Client(
                    trust_env=False,
                    timeout=httpx.Timeout(runtime["timeout_seconds"], connect=30),
                ) as client:
                    response = client.post(self.endpoint, headers=headers, json=body)
                if response.status_code in retryable:
                    raise AdapterError(f"Retryable HTTP {response.status_code}")
                response.raise_for_status()
                payload = response.json()
                choices = payload.get("choices") or []
                if not choices:
                    raise AdapterError("Provider response has no choices")
                provider_model = payload.get("model")
                if provider_model != self.spec["actual_id"]:
                    raise ProviderIdentityError(
                        f"Provider returned {provider_model!r}; expected {self.spec['actual_id']!r}"
                    )
                message = choices[0].get("message") or {}
                text = message.get("content")
                if not isinstance(text, str):
                    raise AdapterError("Provider response content is not text")
                return GenerationResult(
                    model_key=self.model_key,
                    model_actual_id=provider_model,
                    raw_response=payload,
                    text=text,
                    finish_reason=choices[0].get("finish_reason"),
                    latency_seconds=time.monotonic() - start,
                    retry_count=attempt,
                    endpoint=self.endpoint,
                    request_sha256=request_sha,
                )
            except ProviderIdentityError:
                raise
            except (httpx.HTTPError, ValueError, AdapterError) as exc:
                last_error = exc
                if attempt + 1 >= runtime["request_attempts"]:
                    break
                time.sleep(min(runtime["max_backoff_seconds"], 2 ** (attempt + 1)))
        raise AdapterError(f"Bounded retries exhausted: {type(last_error).__name__}")
