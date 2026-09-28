#!/usr/bin/env python3
from __future__ import annotations
import csv,hashlib,json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; RUN=None
OUT=None
MODELS=["qwen3.5-4b","qwen3.5-9b","qwen3.5-27b"]; ML=dict(zip(MODELS,["Qwen3.5-4B","Qwen3.5-9B","Qwen3.5-27B"]))
CONDS=[]; RETS=["fixed","variable"]; HS=[0.,.25,.75,1.]
HL={0.:"0%",.25:"25%",.75:"75%",1.:"100%"}; HC={0.:"#6A51A3",.25:"#3775BA",.75:"#42949E",1.:"#E28E2C"}
CC={"I03":"#3775BA","I05":"#42949E","I11":"#B64342"}; NBOOT=10000; BSEED=20260916024
mpl.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","DejaVu Sans","Liberation Sans"],"svg.fonttype":"none","pdf.fonttype":42,"font.size":8,"axes.spines.right":False,"axes.spines.top":False,"axes.linewidth":.8,"legend.frameon":False})
def offset(*x): return int.from_bytes(hashlib.sha256("|".join(map(str,x)).encode()).digest()[:4],"big")
def load():
 rows=[]; wanted={"B3",*CONDS}
 for p in sorted((RUN/"sessions").glob("expanded__*/session.json")):
  s=json.loads(p.read_text())
  if s.get("session_status")!="valid" or s.get("condition_id") not in wanted or s.get("model_name") not in MODELS or len(s.get("rounds",[]))!=20: continue
  for r in s["rounds"]: rows.append({"model":s["model_name"],"condition_id":s["condition_id"],"return_condition":s["return_condition"],"honesty_value":float(s["honesty_value"]),"trial_id":int(s["trial_id"]),"round":int(r["round"]),"investment":float(r["investment"])})
 f=pd.DataFrame(rows)
 if len(f)!=len(MODELS)*(len(CONDS)+1)*2*4*60*20: raise RuntimeError(f"row mismatch {len(f)}")
 return f
def cube(f,m,c,ret):
 d=f[(f.model==m)&(f.condition_id==c)&(f.return_condition==ret)]; trials=sorted(d.trial_id.unique())
 a=np.full((len(trials),4,20),np.nan); tp={t:i for i,t in enumerate(trials)}; hp={h:i for i,h in enumerate(HS)}
 for r in d.itertuples(index=False): a[tp[r.trial_id],hp[r.honesty_value],r.round-1]=r.investment
 if trials!=list(range(60)) or not np.isfinite(a).all(): raise RuntimeError(f"incomplete {m}/{c}/{ret}")
 return np.asarray(trials),a
def bootcube(a,*k):
 rng=np.random.default_rng(BSEED+offset(*k)); ix=rng.integers(0,len(a),(NBOOT,len(a))); draws=a[ix].mean(1); lo,hi=np.percentile(draws,[2.5,97.5],axis=0)
 return a.mean(0),lo,hi
def bootvec(a,*k):
 rng=np.random.default_rng(BSEED+offset(*k)); ix=rng.integers(0,len(a),(NBOOT,len(a))); lo,hi=np.percentile(a[ix].mean(1),[2.5,97.5])
 return float(a.mean()),float(lo),float(hi)
def source(f):
 cr=[]; fr=[]; cache={}
 for m in MODELS:
  for c in ["B3",*CONDS]:
   for ret in RETS:
    t,a=cube(f,m,c,ret); cache[m,c,ret]=(t,a)
    if c in CONDS:
     mean,lo,hi=bootcube(a,"curve",m,c,ret)
     for j,h in enumerate(HS):
      for q in range(20): cr.append({"model":m,"condition_id":c,"return_condition":ret,"honesty":HL[h],"honesty_value":h,"round":q+1,"n_trials":60,"mean_investment":mean[j,q],"ci_lo":lo[j,q],"ci_hi":hi[j,q]})
 for m in MODELS:
  for c in CONDS:
   for ret in RETS:
    t,a=cache[m,c,ret]; bt,b=cache[m,"B3",ret]
    if not np.array_equal(t,bt): raise RuntimeError("pairing mismatch")
    di=a[:,3,0]-a[:,0,0]; db=b[:,3,0]-b[:,0,0]; est,lo,hi=bootvec(di-db,"r1did",m,c,ret)
    fr.append({"model":m,"condition_id":c,"return_condition":ret,"n_paired_trials":60,"B3_delta_100_minus_0":db.mean(),"intervention_delta_100_minus_0":di.mean(),"change_vs_B3":est,"ci_lo":lo,"ci_hi":hi})
 OUT.mkdir(parents=True,exist_ok=True); curves=pd.DataFrame(cr); first=pd.DataFrame(fr)
 curves.to_csv(OUT/"round_trajectory_source_data.csv",index=False); first.to_csv(OUT/"first_round_endpoint_penalty_vs_B3.csv",index=False)
 return curves,first
def save(fig,stem):
 fig.savefig(Path(str(stem)+".svg"),bbox_inches="tight"); fig.savefig(Path(str(stem)+".pdf"),bbox_inches="tight"); fig.savefig(Path(str(stem)+".png"),dpi=300,bbox_inches="tight"); plt.close(fig)
def plots(curves,first):
 made=[]; x=np.arange(1,21)
 for c in CONDS:
  for m in MODELS:
   fig,axs=plt.subplots(1,2,figsize=(7.2,3.05),sharex=True,sharey=True)
   for ax,ret in zip(axs,RETS):
    d=curves[(curves.model==m)&(curves.condition_id==c)&(curves.return_condition==ret)]
    for h in HS:
     z=d[d.honesty_value==h].sort_values("round"); ax.plot(x,z.mean_investment,color=HC[h],lw=1.8,label=f"Honesty {HL[h]}"); ax.fill_between(x,z.ci_lo,z.ci_hi,color=HC[h],alpha=.12,linewidth=0)
    ax.set(title=ret.capitalize(),xlim=(1,20),ylim=(.8,10.2),xlabel="Investment round"); ax.set_xticks([1,5,10,15,20]); ax.set_yticks([1,4,7,10]); ax.grid(axis="y",color="#D8D8D8",lw=.6,alpha=.7)
   axs[0].set_ylabel("Mean investment (tokens)"); h,l=axs[1].get_legend_handles_labels(); fig.legend(h,l,loc="upper center",ncol=4,bbox_to_anchor=(.5,1.02))
   fig.suptitle(f"{ML[m]} | {c}",y=1.12,fontsize=10,fontweight="bold"); fig.text(.5,-.02,"Lines: trial mean; shading: paired trial-bootstrap 95% CI; n=60 trials per honesty level.",ha="center",fontsize=7)
   fig.tight_layout(rect=[0,.04,1,.96],w_pad=1.2); stem=OUT/f"{c}_{m}_round_trajectories"; save(fig,stem); made.append(Path(str(stem)+".png"))
 labels=[f"{ML[m]}  {c}" for m in MODELS for c in CONDS]; y=np.arange(len(MODELS)*len(CONDS))[::-1]; fig,axs=plt.subplots(1,2,figsize=(7.2,4.05),sharey=True)
 b=max(abs(first.ci_lo.min()),abs(first.ci_hi.max())); lim=(-max(1,b*.18),b*1.08)
 for ax,ret in zip(axs,RETS):
  d=first[first.return_condition==ret].set_index(["model","condition_id"])
  for i,(m,c) in enumerate((m,c) for m in MODELS for c in CONDS):
   z=d.loc[m,c]; ax.plot([z.ci_lo,z.ci_hi],[y[i],y[i]],color=CC[c],lw=1.7); ax.plot(z.change_vs_B3,y[i],"o",color=CC[c],ms=4.8)
  ax.axvline(0,color="#767676",ls="--",lw=1); ax.set(xlim=lim,title=ret.capitalize(),xlabel="First-round endpoint penalty change vs B3 (tokens)"); ax.grid(axis="x",color="#D8D8D8",lw=.6,alpha=.7)
  for q in range(0,len(y),2*len(CONDS)): ax.axhspan(y[min(q+len(CONDS)-1,len(y)-1)]-.5,y[q]+.5,color="#F4F4F4",zorder=-2)
 axs[0].set_yticks(y); axs[0].set_yticklabels(labels); axs[1].tick_params(labelleft=False)
 fig.legend(handles=[plt.Line2D([0],[0],marker="o",color=CC[c],lw=1.7,label=c) for c in CONDS],loc="upper center",ncol=3,bbox_to_anchor=(.5,1.01))
 fig.suptitle("First-round low-honesty penalty change relative to B3",y=1.08,fontsize=10,fontweight="bold"); fig.text(.5,-.01,"Estimate = [(I100 - I0)intervention - (I100 - I0)B3]; paired trial-bootstrap 95% CI; n=60.",ha="center",fontsize=7)
 fig.tight_layout(rect=[0,.04,1,.96],w_pad=1.5); stem=OUT/"first_round_endpoint_penalty_vs_B3"; save(fig,stem); made.append(Path(str(stem)+".png")); return made

def main():
    import argparse
    global RUN, OUT, CONDS
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--conditions", nargs="+", required=True, choices=list(CC))
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    RUN = ROOT / "runs" / args.run_id
    OUT = args.out_dir or RUN / "analysis" / "intervention_figures"
    CONDS = args.conditions
    curves, first = source(load())
    plots(curves, first)

if __name__ == "__main__":
    main()
