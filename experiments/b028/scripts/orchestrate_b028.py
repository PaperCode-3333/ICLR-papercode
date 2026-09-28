#!/usr/bin/env python3
"""Persistent, gated, adaptive, unattended B028 supervisor."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from b028_backends import atomic_json, now
from b028_runtime import progress
from b028_sources import make_source_manifest, verify_source_manifest
from b028_stimuli import ROOT, design, iter_screen_specs

def call_logged(command,log_path):
    log_path.parent.mkdir(parents=True,exist_ok=True)
    with log_path.open("a",encoding="utf-8") as handle:
        handle.write(f"\n[{now()}] command={json.dumps(command,ensure_ascii=False)}\n")
        handle.flush()
        return subprocess.call(command,cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT)

def read_json(path,default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default

def write_state(path,state,**updates):
    state.update(updates)
    state["updated"]=now()
    atomic_json(path,state)

def main() -> int:
    parser=argparse.ArgumentParser()
    parser.add_argument("--run-id",required=True)
    parser.add_argument("--poll-seconds",type=int,default=60)
    parser.add_argument("--skip-gates",action="store_true")
    args=parser.parse_args()
    run=ROOT/"runs"/args.run_id;run.mkdir(parents=True,exist_ok=True)
    logs=run/"logs";logs.mkdir(exist_ok=True)
    lock=(run/"orchestrator.lock").open("w",encoding="utf-8")
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("another B028 orchestrator already owns this run")
    state={"pid":os.getpid(),"started":now(),"run_id":args.run_id,"status":"starting","poll_seconds":args.poll_seconds,"triggered_conditions":[]}
    state_path=run/"orchestrator.json";atomic_json(state_path,state)
    python=sys.executable

    if not args.skip_gates:
        smoke_selection=run/"gate_expansion_selection.json"
        atomic_json(smoke_selection,{model:["I05"] for model in design()["models"]})
        all_conditions=list(json.loads((ROOT/"conditions"/"condition_registry.json").read_text()))
        all_ids=[x["id"] for x in all_conditions]
        gates=[
            ("protocol_audit",[python,str(ROOT/"scripts"/"audit_b028.py")]),
            ("synthetic_all_conditions",[python,str(ROOT/"scripts"/"run_b028.py"),"--run-id",args.run_id+"_synthetic_smoke","--phase","screen","--trials","1","--synthetic","--conditions",*all_ids]),
            ("synthetic_analysis",[python,str(ROOT/"scripts"/"analyze_b028.py"),"--run-id",args.run_id+"_synthetic_smoke"]),
            ("real_canary",[python,str(ROOT/"scripts"/"run_b028.py"),"--run-id",args.run_id+"_real_canary","--phase","screen","--trials","1","--dry-run","--canary"]),
        ]
        gate_results={}
        for name,command in gates:
            write_state(state_path,state,status="gate_"+name,gate_results=gate_results)
            code=call_logged(command,logs/f"gate_{name}.log")
            gate_results[name]=code
            if code!=0:
                write_state(state_path,state,status="gate_failed",failed_gate=name,gate_results=gate_results)
                return 4
        atomic_json(run/"gates_passed.json",{"timestamp":now(),"results":gate_results})
        state["gate_results"]=gate_results

    def run_until(command,label):
        cycle=0
        while True:
            cycle+=1
            write_state(state_path,state,status="running_"+label,current_label=label,current_cycle=cycle)
            code=call_logged(command,logs/f"{label}.log")
            current=read_json(run/f"progress_{label}.json",{})
            if code==0 and current.get("missing")==0 and current.get("service_failure")==0 and current.get("valid",0)+current.get("excluded",0)==current.get("target",-1):
                write_state(state_path,state,status="completed_"+label,current_progress=current)
                return current
            write_state(state_path,state,status="retry_wait_"+label,current_progress=current,runner_return_code=code)
            time.sleep(max(10,args.poll_seconds))

    core=list(design()["core_conditions"])
    core_command=[python,str(ROOT/"scripts"/"run_b028.py"),"--run-id",args.run_id,"--phase","screen","--label","core","--conditions",*core]
    core_progress=run_until(core_command,"core")
    write_state(state_path,state,status="analyzing_core",current_progress=core_progress)
    if call_logged([python,str(ROOT/"scripts"/"analyze_b028.py"),"--run-id",args.run_id],logs/"analysis_core.log")!=0:
        write_state(state_path,state,status="analysis_failed",failed_stage="core")
        return 5
    summary=read_json(run/"analysis"/"summary.json",{})
    robust=set(summary.get("robust_rescues",[]))
    triggered=[]

    weak=[]
    if not ({"I03","I05"} & robust):
        weak.append("I11")
    if not ({"I07","I05"} & robust):
        weak.append("I12")
    caution=summary.get("mechanism_status",{}).get("caution_direction",{})
    if any(caution.get(model,{}).get("direction",False) for model in design()["models"][:2]) and "I08" not in robust:
        weak.append("I13")
    for condition in weak:
        command=[python,str(ROOT/"scripts"/"run_b028.py"),"--run-id",args.run_id,"--phase","screen","--label","screen_"+condition,"--conditions",condition]
        run_until(command,"screen_"+condition);triggered.append(condition)
    if weak:
        call_logged([python,str(ROOT/"scripts"/"analyze_b028.py"),"--run-id",args.run_id],logs/"analysis_after_weak.log")
        summary=read_json(run/"analysis"/"summary.json",{})
        robust=set(summary.get("robust_rescues",[]))

    if not robust:
        condition="I14"
        run_until([python,str(ROOT/"scripts"/"run_b028.py"),"--run-id",args.run_id,"--phase","screen","--label","screen_I14","--conditions",condition],"screen_I14")
        triggered.append(condition)
        call_logged([python,str(ROOT/"scripts"/"analyze_b028.py"),"--run-id",args.run_id],logs/"analysis_after_I14.log")
        summary=read_json(run/"analysis"/"summary.json",{})
        robust=set(summary.get("robust_rescues",[]))
        if "I14" not in robust:
            condition="I15"
            run_until([python,str(ROOT/"scripts"/"run_b028.py"),"--run-id",args.run_id,"--phase","screen","--label","screen_I15","--conditions",condition],"screen_I15")
            triggered.append(condition)
            call_logged([python,str(ROOT/"scripts"/"analyze_b028.py"),"--run-id",args.run_id],logs/"analysis_after_I15.log")
            summary=read_json(run/"analysis"/"summary.json",{})
            robust=set(summary.get("robust_rescues",[]))
    state["triggered_conditions"]=triggered

    observed=core+triggered
    all_screen=list(iter_screen_specs(conditions=observed))
    screening_progress=progress(run,all_screen,"screening_all")
    write_state(state_path,state,status="screening_complete",screening_progress=screening_progress)
    if screening_progress["missing"] or screening_progress["service_failure"]:
        write_state(state_path,state,status="screening_integrity_failed")
        return 6

    small=design()["models"][:2]
    selection={model:[] for model in design()["models"]}
    for model in small:
        value=summary.get("minimal_rescue_by_model",{}).get(model)
        if value and value!="I10":
            selection[model]=[value]
    positive=sorted(set(summary.get("robust_rescues",[]))|{x for model in small for x in selection[model]})
    selection[design()["models"][2]]=positive
    atomic_json(run/"expansion_selection.json",selection)
    selected_models=[model for model,conditions in selection.items() if conditions]
    expanded_progress=None
    if selected_models:
        command=[python,str(ROOT/"scripts"/"run_b028.py"),"--run-id",args.run_id,"--phase","expanded","--label","expanded","--selection",str(run/"expansion_selection.json"),"--models",*selected_models]
        expanded_progress=run_until(command,"expanded")
        write_state(state_path,state,status="analyzing_expanded",expanded_progress=expanded_progress)
        if call_logged([python,str(ROOT/"scripts"/"analyze_expanded.py"),"--run-id",args.run_id],logs/"analysis_expanded.log")!=0:
            write_state(state_path,state,status="analysis_failed",failed_stage="expanded")
            return 7
    else:
        atomic_json(run/"analysis"/"expanded"/"summary.json",{"status":"not_run","reason":"no registered rescue"})

    write_state(state_path,state,status="building_report")
    report_code=0  # Manuscript/report rendering is not part of this release.
    before=read_json(run/"source_manifest_before.json",{})
    after=make_source_manifest();atomic_json(run/"source_manifest_after.json",after)
    source_check=verify_source_manifest(before)
    atomic_json(run/"source_readonly_verification.json",source_check)
    if source_check["status"]!="pass":
        write_state(state_path,state,status="source_integrity_failed",source_verification=source_check)
        return 8
    if report_code!=0:
        write_state(state_path,state,status="report_failed",report_return_code=report_code)
        return 9
    write_state(state_path,state,status="complete",screening_progress=screening_progress,expanded_progress=expanded_progress,selection=selection,robust_rescues=summary.get("robust_rescues",[]),report_return_code=report_code,source_verification=source_check)
    return 0

if __name__=="__main__":
    raise SystemExit(main())
