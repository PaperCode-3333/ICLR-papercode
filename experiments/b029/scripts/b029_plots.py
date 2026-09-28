"""B029 figure generation from derived, read-only analysis tables."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from b029_dynamics import PHASE2_ROUNDS, spline_fit

COLORS={0.0:"#7f0000",0.25:"#d95f02",0.75:"#1b9e77",1.0:"#084081"}
BRIDGE_LABELS={"bridge_0":"B0 no readout","bridge_1":"B1 favorite color","bridge_3":"B3 adjective"}

def _safe(value: str) -> str: return re.sub(r"[^A-Za-z0-9_.-]+","_",value)

def _save(fig, path: Path) -> None:
    fig.savefig(path,dpi=180,bbox_inches="tight"); plt.close(fig)

def investment_trajectories(rounds: pd.DataFrame, out: Path) -> None:
    for model,frame in rounds.groupby("model_name"):
        fig,axes=plt.subplots(3,2,figsize=(12,11),sharex=True,sharey=True)
        for i,bridge in enumerate(["bridge_0","bridge_1","bridge_3"]):
            for j,ret in enumerate(["fixed","variable"]):
                ax=axes[i,j]; part=frame[(frame.bridge==bridge)&(frame.return_condition==ret)]
                for honesty,grp in part.groupby("honesty_value"):
                    stats=grp.groupby("round").investment.agg(["mean","sem"]); x=stats.index.to_numpy(); y=stats["mean"].to_numpy(); e=stats["sem"].fillna(0).to_numpy()
                    ax.plot(x,y,color=COLORS[float(honesty)],label=f"h={honesty:g}"); ax.fill_between(x,y-1.96*e,y+1.96*e,color=COLORS[float(honesty)],alpha=.12)
                ax.set_title(f"{BRIDGE_LABELS[bridge]} | {ret} | n={part.trial_id.nunique()}"); ax.set_ylim(.5,10.5); ax.grid(alpha=.2)
        axes[-1,0].set_xlabel("round"); axes[-1,1].set_xlabel("round")
        for ax in axes[:,0]: ax.set_ylabel("investment")
        handles,labels=axes[0,0].get_legend_handles_labels(); fig.legend(handles,labels,loc="upper center",ncol=4); fig.suptitle(f"B029 investment trajectories — {model}",y=.995)
        _save(fig,out/f"F1_investment_trajectories__{_safe(model)}.png")

def gradient_plots(t3: pd.DataFrame, out: Path) -> None:
    for model,frame in t3.groupby("model_name"):
        for variable,column,prefix,ylabel in (("G_t","G_t","F2_honesty_gradient","honesty dose slope G_t"),("D_t","D_t","F3_endpoint_gap","D_t = I(h=1)-I(h=0)")):
            fig,axes=plt.subplots(1,2,figsize=(12,4.5),sharey=True)
            for ax,ret in zip(axes,["fixed","variable"]):
                part=frame[frame.return_condition==ret]
                for bridge,color in zip(["bridge_0","bridge_1","bridge_3"],["#666666","#377eb8","#e41a1c"]):
                    data=part[part.bridge==bridge].sort_values("round"); x=data["round"].to_numpy(); y=data[column].to_numpy(); lo=data[column.replace("_t","")+"_ci_low"].to_numpy(); hi=data[column.replace("_t","")+"_ci_high"].to_numpy()
                    ax.plot(x,y,label=BRIDGE_LABELS[bridge],color=color); ax.fill_between(x,lo,hi,color=color,alpha=.13)
                ax.axhline(0,color="black",lw=.8); ax.set_title(f"{ret} | n={int(part.n_valid_trials.max())}"); ax.set_xlabel("round"); ax.grid(alpha=.2)
            axes[0].set_ylabel(ylabel); axes[0].legend(fontsize=8); fig.suptitle(f"{model} — {variable}")
            _save(fig,out/f"{prefix}__{_safe(model)}.png")

def forest_plots(t6: pd.DataFrame, out: Path) -> None:
    selections=[("social_gradient","Gradient_AUC_distance","F4a_forest_gradient_AUC_distance"),("allocation_adaptation","Allocation_AUC_distance","F4b_forest_allocation_AUC_distance")]
    for family,metric,name in selections:
        data=t6[(t6.metric_family==family)&(t6.metric==metric)&(t6.contrast.isin(["B3-B0","B3-B1"]))].copy()
        if not len(data): continue
        data["label"]=data.model_name+" | "+data.return_condition+" | "+data.contrast; data=data.sort_values("label")
        y=np.arange(len(data)); fig,ax=plt.subplots(figsize=(9,max(4,len(data)*.28)))
        errors=np.maximum(0,np.vstack([data.estimate-data.ci_low,data.ci_high-data.estimate]))
        ax.errorbar(data.estimate,y,xerr=errors,fmt="o",color="#4c78a8",capsize=2); ax.axvline(0,color="black",lw=.8)
        ax.set_yticks(y,labels=data.label); ax.set_xlabel(f"paired contrast in {metric} (negative = faster relaxation)"); ax.grid(axis="x",alpha=.2)
        _save(fig,out/f"{name}.png")

def social_comparison(t4: pd.DataFrame, out: Path) -> None:
    if not len(t4): return
    fig,axes=plt.subplots(1,2,figsize=(14,max(5,len(t4.model_name.unique())*.45)))
    models=sorted(t4.model_name.unique()); y=np.arange(len(models)); offsets={"bridge_0":-.22,"bridge_1":0,"bridge_3":.22}; colors={"bridge_0":"#777777","bridge_1":"#377eb8","bridge_3":"#e41a1c"}
    for ax,ret in zip(axes,["fixed","variable"]):
        for bridge in offsets:
            data=t4[(t4.return_condition==ret)&(t4.bridge==bridge)].set_index("model_name").reindex(models)
            ax.scatter(data.Gradient_AUC_signed,y+offsets[bridge],label=BRIDGE_LABELS[bridge],color=colors[bridge]);
            errors=np.maximum(0,np.vstack([data.Gradient_AUC_signed-data.Gradient_AUC_signed_ci_low,data.Gradient_AUC_signed_ci_high-data.Gradient_AUC_signed]))
            ax.errorbar(data.Gradient_AUC_signed,y+offsets[bridge],xerr=errors,fmt="none",ecolor=colors[bridge],alpha=.8)
        ax.axvline(0,color="black",lw=.8); ax.set_yticks(y,labels=models); ax.set_title(ret); ax.set_xlabel("Gradient AUC signed"); ax.grid(axis="x",alpha=.2)
    axes[0].legend(fontsize=8); fig.suptitle("Social-gradient persistence comparison")
    _save(fig,out/"F5_social_gradient_AUC_model_comparison.png")

def allocation_comparison(t5: pd.DataFrame, out: Path) -> None:
    if not len(t5): return
    summary=t5.groupby(["model_name","bridge","return_condition"],as_index=False).Allocation_AUC_distance.mean()
    fig,axes=plt.subplots(1,2,figsize=(14,max(5,len(summary.model_name.unique())*.45)),sharey=True); models=sorted(summary.model_name.unique()); y=np.arange(len(models)); offsets={"bridge_0":-.22,"bridge_1":0,"bridge_3":.22}
    for ax,ret in zip(axes,["fixed","variable"]):
        for bridge,color in zip(offsets,["#777777","#377eb8","#e41a1c"]):
            data=summary[(summary.return_condition==ret)&(summary.bridge==bridge)].set_index("model_name").reindex(models); ax.scatter(data.Allocation_AUC_distance,y+offsets[bridge],label=BRIDGE_LABELS[bridge],color=color)
        ax.set_yticks(y,labels=models); ax.set_title(ret); ax.set_xlabel("Allocation AUC-distance (smaller=faster)"); ax.grid(axis="x",alpha=.2)
    axes[0].legend(fontsize=8); fig.suptitle("Allocation adaptation comparison; honesty levels equally weighted")
    _save(fig,out/"F6_allocation_AUC_distance_model_comparison.png")

def feedback_plot(t7: pd.DataFrame, out: Path) -> None:
    data=t7[(t7.estimand=="beta_return_at_honesty_0.5_per_0.01")].copy()
    if not len(data): return
    models=sorted(data.model_name.unique()); y=np.arange(len(models)); offsets={"bridge_0":-.22,"bridge_1":0,"bridge_3":.22}; fig,ax=plt.subplots(figsize=(10,max(5,len(models)*.5)))
    for bridge,color in zip(offsets,["#777777","#377eb8","#e41a1c"]):
        part=data[data.bridge==bridge].set_index("model_name").reindex(models); yy=y+offsets[bridge]
        errors=np.maximum(0,np.vstack([part.estimate-part.ci_low,part.ci_high-part.estimate]))
        ax.errorbar(part.estimate,yy,xerr=errors,fmt="o",label=BRIDGE_LABELS[bridge],color=color,capsize=2)
    ax.axvline(0,color="black",lw=.8); ax.set_yticks(y,labels=models); ax.set_xlabel("change in next investment per +0.01 return rate"); ax.legend(); ax.grid(axis="x",alpha=.2)
    _save(fig,out/"F7_variable_feedback_sensitivity.png")

def convergence_diagnostics(rounds: pd.DataFrame, t5: pd.DataFrame, out: Path) -> None:
    for model,frame in rounds.groupby("model_name"):
        fig,axes=plt.subplots(3,2,figsize=(12,11),sharex=True,sharey=True)
        for i,bridge in enumerate(["bridge_0","bridge_1","bridge_3"]):
            for j,ret in enumerate(["fixed","variable"]):
                ax=axes[i,j]; part=frame[(frame.bridge==bridge)&(frame.return_condition==ret)]
                mean=part.groupby("round").investment.mean().reindex(range(1,PHASE2_ROUNDS+1)).to_numpy(); fit=spline_fit(mean)["fitted"]
                ax.plot(range(1,PHASE2_ROUNDS+1),mean,"o-",ms=3,label="raw mean"); ax.plot(range(1,PHASE2_ROUNDS+1),fit,"--",label="fixed-df spline")
                classes=t5[(t5.model_name==model)&(t5.bridge==bridge)&(t5.return_condition==ret)].trajectory_class.value_counts().to_dict(); ax.set_title(f"{bridge}/{ret}\n{classes}",fontsize=9); ax.grid(alpha=.2)
        axes[0,0].legend(fontsize=8); fig.suptitle(f"Convergence diagnostics — {model}")
        _save(fig,out/f"F8_convergence_diagnostics__{_safe(model)}.png")

def make_all(rounds: pd.DataFrame, tables: Dict[str,pd.DataFrame], out: Path) -> None:
    out.mkdir(parents=True,exist_ok=True)
    investment_trajectories(rounds,out); gradient_plots(tables["T3"],out); forest_plots(tables["T6"],out)
    social_comparison(tables["T4"],out); allocation_comparison(tables["T5"],out); feedback_plot(tables["T7"],out); convergence_diagnostics(rounds,tables["T5"],out)
