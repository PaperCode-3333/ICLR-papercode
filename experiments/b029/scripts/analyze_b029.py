#!/usr/bin/env python3
"""Read-only analysis: raw run data are opened but never modified."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from b029_backends import atomic_json, now
from b029_plots import make_all
from b029_stats import analyze_dynamics, feedback_sensitivity, read_run, table_behavior_gates, table_completeness, trajectory_model
from b029_stimuli import ROOT, design

def main() -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--run-id",required=True); parser.add_argument("--bootstrap-samples",type=int); args=parser.parse_args()
    run_dir=ROOT/"runs"/args.run_id; spec,sessions,excluded,rounds=read_run(run_dir)
    if rounds.empty: raise RuntimeError("no valid raw rounds to analyze")
    samples=args.bootstrap_samples or int(design()["bootstrap_samples"])
    if not spec.get("synthetic") and not spec.get("dry_run") and samples!=10000: raise ValueError("formal analysis requires the frozen 10,000 bootstrap samples")
    analysis=run_dir/"analysis"; tables_dir=analysis/"tables"; figures_dir=analysis/"figures"; bootstrap_dir=analysis/"bootstrap"; report_inputs=analysis/"report_inputs"
    for path in (tables_dir,figures_dir,bootstrap_dir,report_inputs): path.mkdir(parents=True,exist_ok=True)
    progress=json.loads((run_dir/"progress.json").read_text()) if (run_dir/"progress.json").exists() else {}
    dynamics=analyze_dynamics(rounds,samples)
    tables={
        "T1":table_completeness(spec,sessions,excluded,progress),
        "T2":table_behavior_gates(sessions,rounds),
        "T3":dynamics["T3"],"T4":dynamics["T4"],"T5":dynamics["T5"],"T6":dynamics["T6"],
        "T7":feedback_sensitivity(rounds,samples),
    }
    tables["T5b"]=tables["T5"].groupby(["model_name","bridge","return_condition"],as_index=False).agg(n_valid_trials=("n_valid_trials","min"),identified_fraction=("speed_identifiable","mean"),exponential_eligible_fraction=("exponential_eligible","mean"),Allocation_AUC_distance=("Allocation_AUC_distance","mean"),Allocation_AUC_abs=("Allocation_AUC_abs","mean"),T50_restricted=("T50_restricted","mean"),T80_restricted=("T80_restricted","mean"),s_early=("s_early","mean"),k=("k","mean"),half_life_allocation=("half_life_allocation","mean"))
    tables["T8"]=tables["T5"][["model_name","bridge","return_condition","honesty","n_valid_trials","trajectory_class","classification_reason","s_late","terminal_derivative","spline_r2","late_volatility","exponential_eligible","exponential_r2"]].copy()
    tables["T9"]=trajectory_model(rounds)
    names={"T1":"data_completeness","T2":"behavior_gates","T3":"round_honesty_gradient","T4":"social_gradient_dynamics","T5":"allocation_adaptation","T5b":"allocation_bridge_summary","T6":"bridge_contrasts","T7":"feedback_sensitivity","T8":"convergence_diagnostics","T9":"trajectory_spline_interaction"}
    for key,frame in tables.items(): frame.to_csv(tables_dir/f"{key}_{names[key]}.csv",index=False)
    np.savez_compressed(bootstrap_dir/"paired_trial_bootstrap_trajectories.npz",**dynamics["bootstrap"])
    make_all(rounds,tables,figures_dir)
    metadata={"run_id":args.run_id,"generated_at":now(),"bootstrap_samples":samples,"bootstrap_seed":design()["bootstrap_seed"],"trial_cluster_resampling":True,"raw_data_modified":False,"valid_sessions":len(sessions),"valid_rounds":len(rounds),"tables":names,"figure_count":len(list(figures_dir.glob('*.png')))}
    atomic_json(analysis/"analysis_manifest.json",metadata); print(json.dumps(metadata,ensure_ascii=False,indent=2)); return 0
if __name__=="__main__": raise SystemExit(main())
