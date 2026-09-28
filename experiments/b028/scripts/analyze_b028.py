#!/usr/bin/env python3
"""Registered B028 first-round paired analysis and decision trace."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from b028_backends import atomic_json
from b028_stimuli import ROOT, design, models, registry

GATE=design()["action_range_gate"]
TREE=json.loads((ROOT/"conditions"/"decision_tree.json").read_text())
LEX=json.loads((ROOT/"conditions"/"adjective_valence.json").read_text())

def ci(values, seed=None):
    array=np.asarray(values,dtype=float)
    if not len(array):
        return (float("nan"),float("nan"))
    rng=np.random.default_rng(design()["bootstrap_seed"] if seed is None else seed)
    draws=array[rng.integers(0,len(array),(int(design()["bootstrap_samples"]),len(array)))].mean(axis=1)
    return tuple(np.percentile(draws,[2.5,97.5]))

def load_screen(run: Path) -> pd.DataFrame:
    rows=[]
    for path in sorted((run/"sessions").glob("screen__*/session.json")):
        item=json.loads(path.read_text(encoding="utf-8"))
        if item.get("phase")!="screen" or item.get("session_status")!="valid":
            continue
        row={k:v for k,v in item.items() if k not in ("messages","phase1","rounds","audit")}
        row["investment_r1"]=item["rounds"][0]["investment"]
        row["investment_first_pass"]=bool(item.get("investment_first_pass",item["rounds"][0].get("first_pass",False)))
        rows.append(row)
    return pd.DataFrame(rows)

def paired(frame, model, condition):
    data=frame[(frame.model_name==model)&(frame.condition_id==condition)]
    if data.empty:
        return pd.DataFrame()
    return data.pivot_table(index="trial_id",columns="honesty",values="investment_r1",aggfunc="first").dropna(subset=["0%","100%"])

def summarize(run: Path) -> dict:
    out=run/"analysis"
    figures=out/"figures"
    out.mkdir(parents=True,exist_ok=True)
    figures.mkdir(exist_ok=True)
    frame=load_screen(run)
    cells=[];gaps=[];dids=[];cautions=[];adjectives=[]
    condition_order=list(registry())
    for model in design()["models"]:
        for condition in condition_order:
            data=frame[(frame.model_name==model)&(frame.condition_id==condition)]
            if data.empty:
                continue
            pair=paired(frame,model,condition)
            diff=(pair["100%"]-pair["0%"]).to_numpy() if len(pair) else np.asarray([])
            gate=True
            for _,group in data.groupby("honesty"):
                gate &= len(group)>2
                gate &= float(group.investment_r1.std(ddof=1))>float(GATE["min_sd_exclusive"])
            low,high=ci(diff)
            pooled=np.sqrt((data[data.honesty=="0%"].investment_r1.var()+data[data.honesty=="100%"].investment_r1.var())/2)
            half90=student_t.ppf(.95,len(diff)-1)*float(diff.std(ddof=1))/np.sqrt(len(diff)) if len(diff)>2 else np.nan
            gaps.append({"model":model,"condition_id":condition,"n_paired":len(pair),"delta":float(diff.mean()) if len(diff) else np.nan,"ci_lo":low,"ci_hi":high,"gz":float(diff.mean()/pooled) if pooled>0 and len(diff) else np.nan,"gate":bool(gate),"investment_first_pass":float(data.investment_first_pass.mean()),"equivalent_zero":bool(len(diff)>2 and abs(float(diff.mean()))+half90<float(design()["sesoi"])),"ci90_lo":float(diff.mean()-half90) if len(diff)>2 else np.nan,"ci90_hi":float(diff.mean()+half90) if len(diff)>2 else np.nan})
            for honesty in design()["screening_honesty"]:
                group=data[data.honesty==honesty]
                values=group.investment_r1.to_numpy()
                lo,hi=ci(values)
                cells.append({"model":model,"condition_id":condition,"honesty":honesty,"n":len(values),"mean":float(values.mean()) if len(values) else np.nan,"ci_lo":lo,"ci_hi":hi,"r1_sd":float(values.std(ddof=1)) if len(values)>1 else np.nan,"p_inv_1":float((values==1).mean()) if len(values) else np.nan,"p_inv_10":float((values==10).mean()) if len(values) else np.nan,"investment_first_pass":float(group.investment_first_pass.mean()) if len(group) else np.nan})
            if condition in ("I08","I13"):
                for honesty in design()["screening_honesty"]:
                    group=data[data.honesty==honesty]
                    counts=group.caution_class_if_any.value_counts()
                    cautions.append({"model":model,"condition_id":condition,"honesty":honesty,"n":len(group),"MORE":int(counts.get("MORE",0)),"EQUAL":int(counts.get("EQUAL",0)),"LESS":int(counts.get("LESS",0))})
            if condition=="B3":
                for honesty in design()["screening_honesty"]:
                    words=data[data.honesty==honesty].self_adjective.dropna().str.lower()
                    for word,count in words.value_counts().items():
                        adjectives.append({"model":model,"honesty":honesty,"adjective":word,"count":int(count),"valence":LEX.get(word,np.nan)})
    for model in design()["models"]:
        base=paired(frame,model,"B3")
        for condition in condition_order:
            if condition=="B3":
                continue
            other=paired(frame,model,condition)
            common=base.index.intersection(other.index)
            if not len(common):
                continue
            values=(other.loc[common,"100%"]-other.loc[common,"0%"]-base.loc[common,"100%"]+base.loc[common,"0%"]).to_numpy()
            low,high=ci(values)
            dids.append({"model":model,"condition_id":condition,"n_paired":len(values),"rescue_gain":float(values.mean()),"ci_lo":low,"ci_hi":high})
    tables={"first_round_cells":cells,"paired_gaps":gaps,"rescue_did":dids,"caution_metrics":cautions,"adjective_frequency":adjectives}
    for name,rows in tables.items():
        pd.DataFrame(rows).to_csv(out/f"{name}.csv",index=False)
    gap=pd.DataFrame(gaps);did=pd.DataFrame(dids)
    rescues={};minimal={}
    for model in design()["models"][:2]:
        good=[]
        for condition in TREE["strength"]:
            row=gap[(gap.model==model)&(gap.condition_id==condition)]
            gain=did[(did.model==model)&(did.condition_id==condition)]
            control=gap[(gap.model==model)&(gap.condition_id=="B1_COLOR")]
            if row.empty or gain.empty:
                continue
            x=row.iloc[0];y=gain.iloc[0]
            generic=not control.empty and control.iloc[0].ci_lo>0 and control.iloc[0].delta>=x.delta
            if x.n_paired>=50 and bool(x.gate) and x.investment_first_pass>=.98 and x.delta>0 and x.ci_lo>0 and y.rescue_gain>0 and y.ci_lo>0 and not generic:
                good.append(condition)
        rescues[model]=good
        minimal[model]=next((condition for condition in TREE["strength"] if condition in good),None)
    robust=[]
    for condition in sorted(set(rescues.get(design()["models"][0],[])).intersection(rescues.get(design()["models"][1],[]))):
        row=gap[(gap.model==design()["models"][2])&(gap.condition_id==condition)]
        if not row.empty and row.iloc[0].delta>0 and row.iloc[0].investment_first_pass>=.98:
            robust.append(condition)
    caution_direction={}
    for model in design()["models"]:
        data=frame[(frame.model_name==model)&(frame.condition_id=="I08")].copy()
        if data.empty:
            continue
        data["score"]=data.caution_class_if_any.map({"MORE":1,"EQUAL":0,"LESS":-1})
        pair=data.pivot_table(index="trial_id",columns="honesty",values="score",aggfunc="first").dropna(subset=["0%","100%"])
        if len(pair):
            diff=(pair["0%"]-pair["100%"]).to_numpy()
            lo,hi=ci(diff)
            caution_direction[model]={"direction":bool(diff.mean()>0 and lo>0),"difference":float(diff.mean()),"ci_lo":lo,"ci_hi":hi,"n_paired":len(diff)}
    instrument={}
    for model in design()["models"]:
        row=gap[(gap.model==model)&(gap.condition_id=="I10")]
        if not row.empty:
            x=row.iloc[0]
            instrument[model]={"delta":float(x.delta),"ci_lo":float(x.ci_lo),"ci_hi":float(x.ci_hi)}
    summary={
        "baseline_replicated":{model:{condition:float(gap[(gap.model==model)&(gap.condition_id==condition)].iloc[0].delta) for condition in ("B0","B1_COLOR","B3") if not gap[(gap.model==model)&(gap.condition_id==condition)].empty} for model in design()["models"]},
        "model_level_rescues":rescues,
        "robust_rescues":robust,
        "minimal_rescue_by_model":minimal,
        "instrumental_relevance_result":instrument,
        "mechanism_status":{"caution_direction":caution_direction},
        "action_range_gate":GATE,
        "counts":{"complete":len(frame),"conditions_observed":sorted(frame.condition_id.unique().tolist()) if len(frame) else []},
        "expanded_conditions":{},
        "run_id":run.name,
    }
    atomic_json(out/"summary.json",summary)
    trace=[{"condition":condition,"robust":condition in robust,"model_rescues":{model:condition in rescues.get(model,[]) for model in design()["models"][:2]},"source_run":run.name} for condition in condition_order]
    atomic_json(run/"decision_trace.json",trace)
    (run/"decision_trace.md").write_text("# B028 决策轨迹\n\n"+"\n".join(f"- {row['condition']}: robust={row['robust']}; model-level={row['model_rescues']}" for row in trace)+"\n",encoding="utf-8")
    if len(gap):
        ordered=gap.sort_values(["model","condition_id"])
        y=np.arange(len(ordered))
        fig,ax=plt.subplots(figsize=(11,max(5,len(ordered)*.2)))
        ax.errorbar(ordered.delta,y,xerr=[ordered.delta-ordered.ci_lo,ordered.ci_hi-ordered.delta],fmt="o",capsize=2)
        ax.set_yticks(y,[f"{row.model} {row.condition_id}" for row in ordered.itertuples()],fontsize=7)
        ax.axvline(0,color="gray")
        ax.set_xlabel("Δ investment (100% - 0%), paired bootstrap 95% CI")
        fig.tight_layout();fig.savefig(figures/"delta_forest.png",dpi=150);plt.close(fig)
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    return summary

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--run-id",required=True)
    args=parser.parse_args();summarize(ROOT/"runs"/args.run_id)
