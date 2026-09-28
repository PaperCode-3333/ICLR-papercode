#!/usr/bin/env python3
"""Idempotent B028 screening/expansion runner."""
from __future__ import annotations
import argparse
import concurrent.futures as futures
import contextlib
import json
from pathlib import Path
from b028_backends import Endpoint, LocalServer, PermanentFailure, ServiceFailure, atomic_json, immutable_json, now
from b028_runtime import materialize_raw, prepare_run, progress, run_session
from b028_stimuli import ROOT, design, iter_expansion_specs, iter_screen_specs, models, registry, runtime_config
from b028_validate import full_audit

def run_group(endpoint, run_dir, specs, workers):
    counts={"complete":0,"excluded":0,"failed":0,"failure_reasons":[]}
    def one(spec):
        try:
            return run_session(endpoint,run_dir,spec)
        except (PermanentFailure,ServiceFailure) as exc:
            return ("failed",spec["session_id"],str(exc))
    iterator=iter(specs)
    with futures.ThreadPoolExecutor(max_workers=max(1,workers)) as pool:
        active={}
        for _ in range(max(1,workers)):
            try: spec=next(iterator)
            except StopIteration: break
            active[pool.submit(one,spec)]=spec
        stop=False
        while active:
            done,_=futures.wait(active,return_when=futures.FIRST_COMPLETED)
            for future in done:
                spec=active.pop(future)
                result=future.result()
                key=result if isinstance(result,str) else result[0]
                counts[key]+=1
                if not isinstance(result,str):
                    counts["failure_reasons"].append({"session_id":result[1],"reason":result[2]})
                    stop=True
                if not stop:
                    try: next_spec=next(iterator)
                    except StopIteration: continue
                    active[pool.submit(one,next_spec)]=next_spec
            if stop:
                for future in active:
                    future.cancel()
    return counts

def execute_model(model_name, run_dir, specs, synthetic, slot=None):
    config=runtime_config()
    if synthetic:
        context=contextlib.nullcontext(Endpoint(model_name,synthetic=True))
        workers=4
    else:
        if slot is None:
            raise ValueError("local model requires GPU slot")
        context=LocalServer(model_name,run_dir,gpu=slot["gpu"],port=slot["port"],cpu_affinity=slot["cpu_affinity"])
        workers=int(config["local_workers"])
    try:
        with context as endpoint:
            return {"model":model_name,**run_group(endpoint,run_dir,specs,workers)}
    except (PermanentFailure,ServiceFailure) as exc:
        atomic_json(run_dir/"logs"/f"{model_name}_blocked.json",{"timestamp":now(),"model":model_name,"reason":str(exc)})
        return {"model":model_name,"blocked":str(exc)}

def main() -> int:
    parser=argparse.ArgumentParser()
    parser.add_argument("--run-id",required=True)
    parser.add_argument("--phase",choices=("screen","expanded"),default="screen")
    parser.add_argument("--models",nargs="*")
    parser.add_argument("--conditions",nargs="*")
    parser.add_argument("--selection")
    parser.add_argument("--trials",type=int)
    parser.add_argument("--synthetic",action="store_true")
    parser.add_argument("--dry-run",action="store_true")
    parser.add_argument("--canary",action="store_true")
    parser.add_argument("--label")
    args=parser.parse_args()
    audit=full_audit()
    if audit["status"]!="pass":
        raise RuntimeError("B028 preflight audit failed")
    if args.synthetic and args.dry_run:
        raise ValueError("synthetic and dry-run are mutually exclusive")
    if args.canary and not args.dry_run:
        raise ValueError("--canary requires --dry-run")
    if args.dry_run and (args.trials is None or not 1<=args.trials<=3):
        raise ValueError("real-model dry runs require 1..3 trials")
    if not args.synthetic and not args.dry_run and args.trials not in (None,60):
        raise ValueError("formal B028 runs require 60 trials")
    selected=args.models or list(design()["models"])
    trials=args.trials or int(design()["model_trials"])
    if args.phase=="screen":
        conditions=args.conditions or list(design()["core_conditions"])
        unknown=set(conditions)-set(registry())
        if unknown:
            raise ValueError(f"unknown conditions: {sorted(unknown)}")
        specs=list(iter_screen_specs(selected,conditions,trials))
    else:
        if not args.selection:
            raise ValueError("--selection is required for expansion")
        selection=json.loads(Path(args.selection).read_text(encoding="utf-8"))
        specs=list(iter_expansion_specs(selection,selected,trials))
    if args.canary:
        paths=("I03","I06","I08")
        chosen=[]
        for index,model in enumerate(selected):
            wanted=paths[index%len(paths)]
            chosen.append(next(x for x in specs if x["model_name"]==model and x["condition_id"]==wanted and x["honesty"]=="100%"))
        specs=chosen
    label=args.label or ("expanded" if args.phase=="expanded" else "screen")
    run_dir=ROOT/"runs"/args.run_id
    prepare_run(run_dir,args.synthetic,args.dry_run)
    immutable_json(run_dir/"phase_manifests"/f"{label}.json",{
        "label":label,"phase":args.phase,"models":selected,"trials":trials,
        "target_sessions":len(specs),"session_ids":[x["session_id"] for x in specs],
    })
    print(json.dumps({"run_id":args.run_id,"label":label,"phase":args.phase,"target_sessions":len(specs)},ensure_ascii=False),flush=True)
    by_model={name:[x for x in specs if x["model_name"]==name] for name in selected}
    def terminal(spec):
        directory=run_dir/"sessions"/spec["session_id"]
        return (directory/"session.json").exists() or (directory/"excluded.json").exists()
    pending_by_model={name:[x for x in by_model[name] if not terminal(x)] for name in selected}
    config=runtime_config()
    jobs=[]
    with futures.ThreadPoolExecutor(max_workers=max(1,len(config["gpu_slots"]))) as pool:
        if args.synthetic:
            for model in selected:
                if by_model[model]:
                    jobs.append(pool.submit(execute_model,model,run_dir,by_model[model],True,None))
        else:
            queues=[[] for _ in config["gpu_slots"]]
            if args.phase=="expanded" and len(queues)>1 and selected:
                split_model=selected[-1]
                regular=[name for name in selected[:-1] if pending_by_model[name]]
                for index,name in enumerate(regular):
                    queues[index%len(queues)].append((name,pending_by_model[name]))
                split_specs=pending_by_model[split_model]
                shards=[split_specs[index::len(queues)] for index in range(len(queues))]
                for queue,shard in zip(queues,shards):
                    if shard:
                        queue.append((split_model,shard))
            else:
                active=[name for name in selected if pending_by_model[name]]
                for index,name in enumerate(active):
                    queues[index%len(queues)].append((name,pending_by_model[name]))
            def lane(slot,batches):
                return [execute_model(name,run_dir,batch,False,slot) for name,batch in batches if batch]
            for slot,batches in zip(config["gpu_slots"],queues):
                if batches:
                    jobs.append(pool.submit(lane,slot,batches))
        for future in futures.as_completed(jobs):
            result=future.result()
            for item in (result if isinstance(result,list) else [result]):
                print(json.dumps(item,ensure_ascii=False),flush=True)
            materialize_raw(run_dir)
            progress(run_dir,specs,label)
    final=progress(run_dir,specs,label)
    materialize_raw(run_dir)
    atomic_json(run_dir/f"run_result_{label}.json",{"timestamp":now(),**final})
    return 0 if final["missing"]==0 and final["service_failure"]==0 else 2

if __name__=="__main__":
    raise SystemExit(main())
