#!/usr/bin/env python3
"""Idempotent B029 initializer, synthetic runner, and model worker."""
from __future__ import annotations

import argparse
import concurrent.futures as futures
import contextlib
import fcntl
import json
import os
from pathlib import Path

from b029_backends import Endpoint, LocalServer, PermanentFailure, ServiceFailure, atomic_json, now
from b029_runtime import materialize_raw, prepare_run, progress, run_session
from b029_stimuli import ROOT, design, iter_specs, models, session_spec, runtime_config
from b029_validate import pairing_audit, reuse_audit, static_audit, write_frozen_seeds


def run_group(endpoint, run_dir, specs, workers):
    counts = {"complete": 0, "excluded": 0, "failed": 0, "failure_reasons": []}

    def one(spec):
        try:
            return run_session(endpoint, run_dir, spec)
        except (PermanentFailure, ServiceFailure) as exc:
            return ("failed", spec["session_id"], str(exc))

    iterator = iter(specs)
    with futures.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        active = {}
        for _ in range(max(1, workers)):
            try:
                spec = next(iterator)
            except StopIteration:
                break
            active[pool.submit(one, spec)] = spec
        stop_submitting = False
        while active:
            done, _ = futures.wait(active, return_when=futures.FIRST_COMPLETED)
            for future in done:
                spec = active.pop(future); result = future.result()
                key = result if isinstance(result, str) else result[0]; counts[key] += 1
                if not isinstance(result, str):
                    counts["failure_reasons"].append({"session_id": result[1], "reason": result[2]})
                    stop_submitting = True
                if not stop_submitting:
                    try:
                        next_spec = next(iterator)
                    except StopIteration:
                        continue
                    active[pool.submit(one, next_spec)] = next_spec
            if stop_submitting:
                for future in active:
                    future.cancel()
    return counts


def _model_terminal(run_dir: Path, model_name: str) -> int:
    root = run_dir / "sessions"
    valid = sum(1 for _ in root.glob(f"{model_name}__*/session.json"))
    excluded = sum(1 for _ in root.glob(f"{model_name}__*/excluded.json"))
    return valid + excluded


def execute_model(model_name, run_dir, model_specs, synthetic, slot=None, real_canary=False):
    config = runtime_config()
    model_spec = models()[model_name]
    lock_handle = None
    if not synthetic:
        worker_dir = run_dir / "workers"; worker_dir.mkdir(parents=True, exist_ok=True)
        lock_handle = (worker_dir / f"{model_name}.lock").open("w", encoding="utf-8")
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock_handle.close()
            return {"model": model_name, "already_running": True}
    if synthetic:
        endpoint_context = contextlib.nullcontext(Endpoint(model_name, synthetic=True)); workers = 4
    elif model_spec["backend"] == "vllm":
        if slot is None:
            raise ValueError("local model requires a GPU slot")
        endpoint_context = LocalServer(model_name, run_dir, gpu=slot["gpu"], port=slot["port"], cpu_affinity=slot["cpu_affinity"])
        workers = int(config["local_workers"])
    else:
        endpoint_context = contextlib.nullcontext(Endpoint(model_name)); workers = int(config["api_workers"])
        workers = int(os.getenv("B029_API_WORKERS_OVERRIDE", workers))
    try:
        with endpoint_context as endpoint:
            canary_status = None
            if real_canary:
                canary_dir = run_dir / "canaries" / model_name
                canary = session_spec(model_name, "bridge_3", "fixed", "75%", 0)
                canary_status = run_session(endpoint, canary_dir, canary)
                atomic_json(run_dir / "canaries" / f"{model_name}.json", {
                    "timestamp": now(), "model": model_name, "status": canary_status,
                    "session_id": canary["session_id"], "bridge": "bridge_3", "honesty": "75%",
                    "return_condition": "fixed", "trial_id": 0,
                })
                if canary_status != "complete":
                    raise PermanentFailure(f"real_canary_not_valid:{model_name}:{canary_status}")
            counts = run_group(endpoint, run_dir, model_specs, workers)
            return {"model": model_name, "canary": canary_status, **counts}
    except (PermanentFailure, ServiceFailure) as exc:
        (run_dir / "logs").mkdir(parents=True, exist_ok=True)
        atomic_json(run_dir / "logs" / f"{model_name}_blocked.json", {"timestamp": now(), "reason": str(exc), "model": model_name})
        return {"model": model_name, "blocked": str(exc)}
    finally:
        if lock_handle is not None:
            lock_handle.close()


def _preflight() -> None:
    write_frozen_seeds()
    static = static_audit()
    pairing = pairing_audit(ROOT / "conditions" / "audit" / "seed_and_material_audit.csv")
    reuse = reuse_audit(validate_events=False)
    if static["status"] != "pass" or pairing["status"] != "pass" or reuse["status"] != "pass":
        raise RuntimeError("B029 preflight audit failed")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "conditions" / "design.json"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--models", nargs="*")
    parser.add_argument("--trials", type=int)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--initialize-only", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--slot-gpu", type=int)
    parser.add_argument("--slot-port", type=int)
    parser.add_argument("--slot-cpu")
    parser.add_argument("--skip-canary", action="store_true")
    args = parser.parse_args()
    if Path(args.config).resolve() != (ROOT / "conditions" / "design.json").resolve():
        raise ValueError("B029 uses the frozen conditions/design.json")
    _preflight()
    d = design(); full_models = list(d["models"]); full_trials = int(d["model_trials"])
    run_dir = ROOT / "runs" / args.run_id
    if args.initialize_only:
        if args.synthetic or args.worker:
            raise ValueError("initialize-only cannot be combined with a runner mode")
        prepare_run(run_dir, full_models, full_trials, False, False, target_sessions=11520)
        value = progress(run_dir, iter_specs())
        print(json.dumps(value, ensure_ascii=False, indent=2)); return 0
    selected = args.models or full_models
    unknown = set(selected) - set(full_models)
    if unknown:
        raise ValueError(f"models outside frozen design: {sorted(unknown)}")
    if args.synthetic:
        trials = args.trials or 1
        specs = list(iter_specs(selected, trials))
        prepare_run(run_dir, selected, trials, True, False, target_sessions=len(specs))
    elif args.worker:
        if args.trials not in (None, full_trials):
            raise ValueError("formal workers require all 60 trials")
        prepare_run(run_dir, full_models, full_trials, False, False, target_sessions=11520)
        specs = list(iter_specs(selected, full_trials))
    else:
        raise ValueError("real execution requires --initialize-only or --worker")
    by_model = {name: [item for item in specs if item["model_name"] == name] for name in selected}
    local_names = [name for name in selected if models()[name]["backend"] == "vllm"]
    api_names = [name for name in selected if models()[name]["backend"] != "vllm"]
    if args.worker and local_names:
        if len(local_names) != 1 or api_names:
            raise ValueError("a local worker owns exactly one local model")
        if args.slot_gpu is None or args.slot_port is None or not args.slot_cpu:
            raise ValueError("local worker requires slot GPU, port, and CPU affinity")
        slot = {"gpu": args.slot_gpu, "port": args.slot_port, "cpu_affinity": args.slot_cpu}
        result = execute_model(local_names[0], run_dir, by_model[local_names[0]], False, slot, real_canary=not args.skip_canary)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    else:
        jobs = []
        with futures.ThreadPoolExecutor(max_workers=max(1, len(selected))) as pool:
            for name in selected:
                jobs.append(pool.submit(execute_model, name, run_dir, by_model[name], args.synthetic, None, args.worker and not args.skip_canary))
            for future in futures.as_completed(jobs):
                print(json.dumps(future.result(), ensure_ascii=False), flush=True)
    materialize_raw(run_dir)
    global_specs = list(iter_specs()) if args.worker else specs
    final = progress(run_dir, global_specs)
    atomic_json(run_dir / "run_result.json", {"timestamp": now(), **final})
    if args.worker:
        missing = sum(max(0, 1440 - _model_terminal(run_dir, name)) for name in selected)
        return 0 if missing == 0 else 2
    return 0 if final["missing"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
