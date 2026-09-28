#!/usr/bin/env python3
"""B028 20-round dose-gradient and paired expansion analysis."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from b028_backends import atomic_json
from b028_stimuli import ROOT, design

HONESTY=np.asarray([0.0,0.25,0.75,1.0])
DEN=float(np.sum((HONESTY-HONESTY.mean())**2))

def ci(values, seed_offset=0):
    values=np.asarray(values,dtype=float)
    values=values[np.isfinite(values)]
    if not len(values):
        return (np.nan,np.nan)
    rng=np.random.default_rng(int(design()["bootstrap_seed"])+int(seed_offset))
    draws=values[rng.integers(0,len(values),(int(design()["bootstrap_samples"]),len(values)))].mean(axis=1)
    return tuple(np.percentile(draws,[2.5,97.5]))

def load_expanded(run: Path):
    rows=[]
    for path in sorted((run/"sessions").glob("expanded__*/session.json")):
        item=json.loads(path.read_text(encoding="utf-8"))
        if item.get("phase")!="expanded" or item.get("session_status")!="valid" or len(item.get("rounds",[]))!=20:
            continue
        for row in item["rounds"]:
            rows.append(row)
    return pd.DataFrame(rows)

def cube(frame,model,condition,ret):
    data=frame[(frame.model_name==model)&(frame.condition_id==condition)&(frame.return_condition==ret)]
    if data.empty:
        return np.asarray([]),np.asarray([])
    trials=[]
    arrays=[]
    for trial,group in data.groupby("trial_id"):
        pivot=group.pivot_table(index="honesty_value",columns="round",values="investment",aggfunc="first")
        if list(pivot.index)==list(HONESTY) and list(pivot.columns)==list(range(1,21)):
            trials.append(int(trial));arrays.append(pivot.to_numpy())
    return np.asarray(trials),np.asarray(arrays,dtype=float)

def slopes(values):
    return np.sum((HONESTY[None,:,None]-HONESTY.mean())*values,axis=1)/DEN

def main(run: Path) -> dict:
    frame=load_expanded(run)
    out=run/"analysis"/"expanded";out.mkdir(parents=True,exist_ok=True)
    if frame.empty:
        result={"status":"not_run","reason":"no expanded sessions"}
        atomic_json(out/"summary.json",result);print(json.dumps(result));return result
    round_rows=[];summary_rows=[];contrast_rows=[]
    combos=frame[["model_name","condition_id","return_condition"]].drop_duplicates().itertuples(index=False,name=None)
    caches={}
    for index,(model,condition,ret) in enumerate(sorted(combos)):
        trials,values=cube(frame,model,condition,ret)
        if not len(trials):
            continue
        slope=slopes(values)
        caches[(model,condition,ret)]=(trials,values,slope)
        for round_index in range(20):
            lo,hi=ci(slope[:,round_index],index*100+round_index)
            round_rows.append({"model":model,"condition_id":condition,"return_condition":ret,"round":round_index+1,"n_complete_trials":len(trials),"gradient":float(slope[:,round_index].mean()),"ci_lo":lo,"ci_hi":hi})
        per_trial_auc=slope.mean(axis=1)
        metrics={"G1":slope[:,0],"G20":slope[:,-1],"gradient_auc":per_trial_auc,"endpoint_gap_r1":values[:,3,0]-values[:,0,0],"endpoint_gap_mean":values[:,3,:].mean(axis=1)-values[:,0,:].mean(axis=1)}
        row={"model":model,"condition_id":condition,"return_condition":ret,"n_complete_trials":len(trials)}
        for offset,(name,values1) in enumerate(metrics.items()):
            lo,hi=ci(values1,index*1000+offset)
            row[name]=float(values1.mean());row[name+"_ci_lo"]=lo;row[name+"_ci_hi"]=hi
        summary_rows.append(row)
    for index,(key,value) in enumerate(sorted(caches.items())):
        model,condition,ret=key
        if condition=="B3":
            continue
        base=caches.get((model,"B3",ret))
        if base is None:
            continue
        trials,values,slope=value
        bt,bv,bs=base
        common=sorted(set(trials).intersection(bt))
        if not common:
            continue
        pos={int(t):i for i,t in enumerate(trials)};bpos={int(t):i for i,t in enumerate(bt)}
        left=np.asarray([pos[t] for t in common]);right=np.asarray([bpos[t] for t in common])
        metrics={
            "G1_DiD":slope[left,0]-bs[right,0],
            "G20_DiD":slope[left,-1]-bs[right,-1],
            "gradient_auc_DiD":slope[left].mean(axis=1)-bs[right].mean(axis=1),
            "endpoint_gap_mean_DiD":(values[left,3,:].mean(axis=1)-values[left,0,:].mean(axis=1))-(bv[right,3,:].mean(axis=1)-bv[right,0,:].mean(axis=1)),
        }
        row={"model":model,"condition_id":condition,"return_condition":ret,"n_common_trials":len(common)}
        for offset,(name,metric) in enumerate(metrics.items()):
            lo,hi=ci(metric,500000+index*10+offset)
            row[name]=float(metric.mean());row[name+"_ci_lo"]=lo;row[name+"_ci_hi"]=hi
        contrast_rows.append(row)
    pd.DataFrame(round_rows).to_csv(out/"expanded_rounds.csv",index=False)
    pd.DataFrame(summary_rows).to_csv(out/"expanded_summary.csv",index=False)
    pd.DataFrame(contrast_rows).to_csv(out/"expanded_contrasts.csv",index=False)
    completeness=frame.groupby(["model_name","condition_id","return_condition","honesty"]).session_id.nunique().reset_index(name="valid_sessions")
    completeness.to_csv(out/"expanded_completeness.csv",index=False)
    result={"status":"complete","valid_sessions":int(frame.session_id.nunique()),"round_rows":len(frame),"complete_metric_cells":len(summary_rows),"paired_contrasts":len(contrast_rows)}
    atomic_json(out/"summary.json",result)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return result

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--run-id",required=True)
    args=parser.parse_args();main(ROOT/"runs"/args.run_id)
