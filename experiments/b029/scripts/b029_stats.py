"""Read-only B029 statistical analysis with trial-cluster paired bootstrap."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from b029_dynamics import allocation_metrics, gradient_metrics, honesty_slope, vectorized_allocation_draws, vectorized_gradient_draws, _natural_cubic_basis
from b029_stimuli import design, stable_seed


HONESTY = [0.0, 0.25, 0.75, 1.0]
BRIDGES = ["bridge_0", "bridge_1", "bridge_3"]
RETURNS = ["fixed", "variable"]
CONTRASTS = [("B3-B0", "bridge_3", "bridge_0"), ("B3-B1", "bridge_3", "bridge_1"), ("B1-B0", "bridge_1", "bridge_0")]
PHASE2_ROUNDS = int(design()["phase2_rounds"])


def _read_jsonl(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0: return pd.DataFrame()
    return pd.DataFrame(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def read_run(run_dir: Path) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    run_dir = Path(run_dir); spec = json.loads((run_dir / "spec.json").read_text(encoding="utf-8"))
    sessions = _read_jsonl(run_dir / "raw_sessions.jsonl"); excluded = _read_jsonl(run_dir / "excluded.jsonl")
    if (run_dir / "raw_rounds.parquet").exists(): rounds = pd.read_parquet(run_dir / "raw_rounds.parquet")
    else: rounds = pd.read_csv(run_dir / "raw_rounds.csv") if (run_dir / "raw_rounds.csv").exists() and (run_dir / "raw_rounds.csv").stat().st_size else pd.DataFrame()
    for frame in (sessions, excluded, rounds):
        if "trial_id" in frame: frame["trial_id"] = frame["trial_id"].astype(int)
    if "round" in rounds: rounds["round"] = rounds["round"].astype(int)
    return spec, sessions, excluded, rounds


def ci(values: np.ndarray) -> Tuple[float, float]:
    finite = np.asarray(values, dtype=float); finite = finite[np.isfinite(finite)]
    return (float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975))) if len(finite) else (float("nan"), float("nan"))


def finite_mean(values: np.ndarray, axis: int | None = None) -> np.ndarray | float:
    array=np.asarray(values,dtype=float); count=np.isfinite(array).sum(axis=axis); total=np.nansum(array,axis=axis)
    result=np.divide(total,count,out=np.full(np.shape(total),np.nan,dtype=float),where=count>0)
    return float(result) if np.ndim(result)==0 else result


def _counts(model: str, n: int, samples: int) -> np.ndarray:
    rng = np.random.default_rng(int(design()["bootstrap_seed"]) + stable_seed("bootstrap", model))
    return rng.multinomial(n, np.full(n, 1 / n), size=samples).astype(np.float64)


def common_trials(rounds: pd.DataFrame, model: str) -> List[int]:
    part = rounds[rounds.model_name == model]
    expected_sessions = len(BRIDGES) * len(RETURNS) * len(HONESTY)
    counts = part.groupby("trial_id").agg(sessions=("session_id", "nunique"), rows=("round", "size"))
    return sorted(int(index) for index, row in counts.iterrows() if row.sessions == expected_sessions and row.rows == expected_sessions * PHASE2_ROUNDS)


def group_matrix(rounds: pd.DataFrame, model: str, bridge: str, return_condition: str, honesty: float, trials: List[int]) -> np.ndarray:
    part = rounds[(rounds.model_name == model) & (rounds.bridge == bridge) & (rounds.return_condition == return_condition) & (np.isclose(rounds.honesty_value.astype(float), honesty))]
    pivot = part.pivot(index="trial_id", columns="round", values="investment").reindex(index=trials, columns=range(1, PHASE2_ROUNDS + 1))
    values = pivot.to_numpy(dtype=float)
    if values.shape != (len(trials), PHASE2_ROUNDS) or not np.isfinite(values).all(): raise ValueError(f"incomplete matrix: {model}/{bridge}/{return_condition}/{honesty}")
    return values


def table_completeness(spec: Dict[str, Any], sessions: pd.DataFrame, excluded: pd.DataFrame, progress: Dict[str, Any] | None = None) -> pd.DataFrame:
    rows = []
    target_per_model = int(spec["trials"]) * 24
    for model in spec["models"]:
        valid = sessions[sessions.model_name == model] if len(sessions) else sessions
        exc = excluded[excluded.model_name == model] if len(excluded) else excluded
        failures = int((progress or {}).get("by_model", {}).get(model, {}).get("service_failure", 0))
        completed = len(valid) + len(exc)
        rows.append({
            "model_name": model, "target_sessions": target_per_model, "completed_sessions": completed,
            "valid_sessions": len(valid), "excluded_sessions": len(exc), "completion_rate": completed / target_per_model,
            "format_first_pass_rate": float(valid.format_first_pass.mean()) if len(valid) and "format_first_pass" in valid else float("nan"),
            "api_or_generation_failures": failures,
        })
    return pd.DataFrame(rows)


def table_behavior_gates(sessions: pd.DataFrame, rounds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["model_name", "bridge", "return_condition", "honesty", "honesty_value"]
    for key, group in rounds.groupby(keys, sort=True):
        by_round = group.groupby("round").investment.mean().reindex(range(1, PHASE2_ROUNDS + 1)).to_numpy(dtype=float)
        metrics = allocation_metrics(by_round)
        first = group[group["round"] == 1].investment.astype(float)
        session_ids = set(group.session_id.unique()); session_part = sessions[sessions.session_id.isin(session_ids)]
        rows.append({
            **dict(zip(keys, key)), "n_valid_trials": group.trial_id.nunique(), "r1_mean": first.mean(), "r1_sd": first.std(ddof=1),
            "p_inv1_r1": (first == 1).mean(), "p_inv10_r1": (first == 10).mean(),
            "p_inv1_all": (group.investment == 1).mean(), "p_inv10_all": (group.investment == 10).mean(),
            "format_first_pass_rate": session_part.format_first_pass.mean() if len(session_part) else float("nan"),
            "A0": metrics["A0"], "total_variation": metrics["total_variation"], "speed_identifiable": metrics["speed_identifiable"],
        })
    return pd.DataFrame(rows)


def _summarize_draw(point: float, draws: np.ndarray, prefix: str = "") -> Dict[str, float]:
    low, high = ci(draws); return {prefix + "estimate": float(point), prefix + "ci_low": low, prefix + "ci_high": high}


def analyze_dynamics(rounds: pd.DataFrame, samples: int | None = None) -> Dict[str, Any]:
    samples = int(samples or design()["bootstrap_samples"])
    t3_rows: List[Dict[str, Any]] = []; t4_rows: List[Dict[str, Any]] = []; t5_rows: List[Dict[str, Any]] = []; t6_rows: List[Dict[str, Any]] = []
    gradient_draw_store: Dict[Tuple[str, str, str], Dict[str, np.ndarray]] = {}
    allocation_bridge_draw_store: Dict[Tuple[str, str, str], Dict[str, np.ndarray]] = {}
    allocation_bridge_eligibility: Dict[Tuple[str, str, str, str], float] = {}
    bootstrap_artifacts: Dict[str, np.ndarray] = {}
    for model in sorted(rounds.model_name.unique()):
        trials = common_trials(rounds, model)
        if not trials: continue
        counts = _counts(model, len(trials), samples); denominator = counts.sum(axis=1, keepdims=True)
        allocation_draw_by_return_bridge: Dict[Tuple[str, str], Dict[str, List[np.ndarray]]] = {}
        for return_condition in RETURNS:
            for bridge in BRIDGES:
                honesty_arrays = [group_matrix(rounds, model, bridge, return_condition, h, trials) for h in HONESTY]
                cube = np.stack(honesty_arrays, axis=1)
                trial_slopes = honesty_slope(cube)
                trial_endpoint = cube[:, 3, :] - cube[:, 0, :]
                gradient = trial_slopes.mean(axis=0); endpoint = trial_endpoint.mean(axis=0)
                gradient_draws = (counts @ trial_slopes) / denominator
                endpoint_draws = (counts @ trial_endpoint) / denominator
                for round_no in range(1, PHASE2_ROUNDS + 1):
                    gl, gh = ci(gradient_draws[:, round_no - 1]); dl, dh = ci(endpoint_draws[:, round_no - 1])
                    t3_rows.append({"model_name": model, "bridge": bridge, "return_condition": return_condition, "round": round_no, "n_valid_trials": len(trials), "G_t": gradient[round_no - 1], "G_ci_low": gl, "G_ci_high": gh, "D_t": endpoint[round_no - 1], "D_ci_low": dl, "D_ci_high": dh})
                point_gradient = gradient_metrics(gradient, fit_exponential=True)
                draws_gradient = vectorized_gradient_draws(gradient_draws)
                if not point_gradient["gradient_exponential_eligible"]:
                    draws_gradient["lambda"] = np.full(samples, np.nan)
                    draws_gradient["half_life_gradient"] = np.full(samples, np.nan)
                row = {"model_name": model, "bridge": bridge, "return_condition": return_condition, "n_valid_trials": len(trials), **point_gradient}
                for metric, values in draws_gradient.items():
                    low, high = ci(values); row[metric + "_ci_low"] = low; row[metric + "_ci_high"] = high
                t4_rows.append(row); gradient_draw_store[(model, bridge, return_condition)] = draws_gradient
                bootstrap_artifacts[f"gradient__{model}__{bridge}__{return_condition}"] = gradient_draws.astype(np.float32)
                for honesty, matrix in zip(HONESTY, honesty_arrays):
                    mean_trajectory = matrix.mean(axis=0); point = allocation_metrics(mean_trajectory, fit_exponential=True)
                    mean_draws = (counts @ matrix) / denominator; metric_draws = vectorized_allocation_draws(mean_draws)
                    if not point["exponential_eligible"]:
                        metric_draws["k"] = np.full(samples, np.nan)
                        metric_draws["half_life_allocation"] = np.full(samples, np.nan)
                    row = {"model_name": model, "bridge": bridge, "return_condition": return_condition, "honesty_value": honesty, "honesty": f"{int(honesty*100)}%", "n_valid_trials": len(trials), **point}
                    for metric, values in metric_draws.items():
                        low, high = ci(values); row[metric + "_ci_low"] = low; row[metric + "_ci_high"] = high
                    t5_rows.append(row)
                    target = allocation_draw_by_return_bridge.setdefault((return_condition, bridge), {})
                    for metric, values in metric_draws.items(): target.setdefault(metric, []).append(values)
        # Honesty-equal bridge summaries and paired bridge contrasts use exactly the same cluster draws.
        for (return_condition, bridge), metrics in allocation_draw_by_return_bridge.items():
            for metric, four in metrics.items():
                allocation_bridge_draw_store[(model, bridge, return_condition, metric)] = finite_mean(np.stack(four), axis=0)
                allocation_bridge_eligibility[(model, bridge, return_condition, metric)] = sum(np.isfinite(values).any() for values in four) / len(four)
        gradient_metrics_for_contrast = ["G1", "Gradient_AUC_signed", "Gradient_AUC_abs", "Gradient_AUC_distance", "T50_gradient_restricted", "T80_gradient_restricted", "lambda", "half_life_gradient"]
        allocation_metrics_for_contrast = ["Allocation_AUC_distance", "Allocation_AUC_abs", "T50_restricted", "T80_restricted", "abs_s_early_normalized", "k", "half_life_allocation"]
        for return_condition in RETURNS:
            for contrast, high_bridge, low_bridge in CONTRASTS:
                for metric in gradient_metrics_for_contrast:
                    high = gradient_draw_store[(model, high_bridge, return_condition)][metric]; low = gradient_draw_store[(model, low_bridge, return_condition)][metric]
                    values = high - low; lower, upper = ci(values)
                    t6_rows.append({"model_name": model, "return_condition": return_condition, "contrast": contrast, "metric_family": "social_gradient", "metric": metric, "estimate": finite_mean(values), "ci_low": lower, "ci_high": upper, "n_valid_trials": len(trials), "eligible_fraction_high":float(np.isfinite(high).any()),"eligible_fraction_low":float(np.isfinite(low).any()),"paired_trial_bootstrap": True})
                for metric in allocation_metrics_for_contrast:
                    high = allocation_bridge_draw_store[(model, high_bridge, return_condition, metric)]; low = allocation_bridge_draw_store[(model, low_bridge, return_condition, metric)]
                    values = high - low; lower, upper = ci(values)
                    t6_rows.append({"model_name": model, "return_condition": return_condition, "contrast": contrast, "metric_family": "allocation_adaptation", "metric": metric, "estimate": finite_mean(values), "ci_low": lower, "ci_high": upper, "n_valid_trials": len(trials), "eligible_fraction_high":allocation_bridge_eligibility[(model,high_bridge,return_condition,metric)],"eligible_fraction_low":allocation_bridge_eligibility[(model,low_bridge,return_condition,metric)],"paired_trial_bootstrap": True})
    return {"T3": pd.DataFrame(t3_rows), "T4": pd.DataFrame(t4_rows), "T5": pd.DataFrame(t5_rows), "T6": pd.DataFrame(t6_rows), "bootstrap": bootstrap_artifacts}


def _feedback_design(frame: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
    frame = frame.copy(); frame["centered_return"] = (frame.return_rate.astype(float) - 0.60) / 0.01
    frame["round_centered"] = (frame["round"].astype(float) - (PHASE2_ROUNDS + 1.0) / 2.0) / (PHASE2_ROUNDS / 2.0)
    b1 = (frame.bridge == "bridge_1").astype(float).to_numpy(); b3 = (frame.bridge == "bridge_3").astype(float).to_numpy()
    ret = frame.centered_return.to_numpy(); h = frame.honesty_value.astype(float).to_numpy()
    columns = [np.ones(len(frame)), frame.investment.astype(float).to_numpy(), ret, h, frame.round_centered.to_numpy(), b1, b3, ret*b1, ret*b3, ret*h, ret*b1*h, ret*b3*h]
    names = ["Intercept", "I_t", "return", "honesty", "round", "B1", "B3", "return:B1", "return:B3", "return:honesty", "return:B1:honesty", "return:B3:honesty"]
    return np.column_stack(columns), frame.next_investment.astype(float).to_numpy(), names, frame.trial_id.astype(int).to_numpy()


def feedback_sensitivity(rounds: pd.DataFrame, samples: int | None = None) -> pd.DataFrame:
    samples = int(samples or design()["bootstrap_samples"]); rows = []
    variable = rounds[rounds.return_condition == "variable"].copy().sort_values(["session_id", "round"])
    variable["next_investment"] = variable.groupby("session_id").investment.shift(-1); variable = variable[variable["round"] < PHASE2_ROUNDS].dropna(subset=["next_investment"])
    for model, frame in variable.groupby("model_name"):
        trials = common_trials(rounds, model); frame = frame[frame.trial_id.isin(trials)]
        if not len(frame): continue
        X, y, names, cluster = _feedback_design(frame); beta = np.linalg.lstsq(X, y, rcond=None)[0]
        xtx_by=[]; xty_by=[]
        for trial in trials:
            mask=cluster==trial; xtx_by.append(X[mask].T@X[mask]); xty_by.append(X[mask].T@y[mask])
        xtx_by=np.stack(xtx_by); xty_by=np.stack(xty_by); counts=_counts(model+"feedback",len(trials),samples)
        draws=[]
        for start in range(0,samples,500):
            c=counts[start:start+500]; xx=np.einsum("bi,ijk->bjk",c,xtx_by); xy=np.einsum("bi,ij->bj",c,xty_by)
            draws.append(np.asarray([np.linalg.lstsq(a,b,rcond=None)[0] for a,b in zip(xx,xy)]))
        draws=np.vstack(draws); index={name:i for i,name in enumerate(names)}
        # Report feedback sensitivity at h=.5, plus the honesty interaction.
        for bridge in BRIDGES:
            point=beta[index["return"]] + .5*beta[index["return:honesty"]]
            sampled=draws[:,index["return"]] + .5*draws[:,index["return:honesty"]]
            if bridge=="bridge_1": point += beta[index["return:B1"]]+.5*beta[index["return:B1:honesty"]]; sampled += draws[:,index["return:B1"]]+.5*draws[:,index["return:B1:honesty"]]
            if bridge=="bridge_3": point += beta[index["return:B3"]]+.5*beta[index["return:B3:honesty"]]; sampled += draws[:,index["return:B3"]]+.5*draws[:,index["return:B3:honesty"]]
            low,high=ci(sampled); rows.append({"model_name":model,"bridge":bridge,"estimand":"beta_return_at_honesty_0.5_per_0.01","estimate":point,"ci_low":low,"ci_high":high,"n_valid_trials":len(trials)})
        for label,term,term_h in (("B3-B0","return:B3","return:B3:honesty"),("B1-B0","return:B1","return:B1:honesty"),("B3-B1",None,None)):
            if label=="B3-B1": point=(beta[index["return:B3"]]-beta[index["return:B1"]])+.5*(beta[index["return:B3:honesty"]]-beta[index["return:B1:honesty"]]); sampled=(draws[:,index["return:B3"]]-draws[:,index["return:B1"]])+.5*(draws[:,index["return:B3:honesty"]]-draws[:,index["return:B1:honesty"]])
            else: point=beta[index[term]]+.5*beta[index[term_h]]; sampled=draws[:,index[term]]+.5*draws[:,index[term_h]]
            low,high=ci(sampled); rows.append({"model_name":model,"bridge":label,"estimand":"beta_return_bridge_contrast_at_honesty_0.5","estimate":point,"ci_low":low,"ci_high":high,"n_valid_trials":len(trials)})
        for bridge, term in (("bridge_0","return:honesty"),("bridge_1","return:B1:honesty"),("bridge_3","return:B3:honesty")):
            point=beta[index["return:honesty"]] + (0 if bridge=="bridge_0" else beta[index[term]])
            sampled=draws[:,index["return:honesty"]] + (0 if bridge=="bridge_0" else draws[:,index[term]])
            low,high=ci(sampled); rows.append({"model_name":model,"bridge":bridge,"estimand":"return_by_honesty_interaction","estimate":point,"ci_low":low,"ci_high":high,"n_valid_trials":len(trials)})
    return pd.DataFrame(rows)


def trajectory_model(rounds: pd.DataFrame) -> pd.DataFrame:
    """Per-model OLS for honesty * bridge * natural-spline(round) * return.

    The tensor-product coding is saturated for these factors. Inference uses a
    trial-cluster sandwich covariance, so rounds are never treated as independent.
    """
    output=[]
    spline=_natural_cubic_basis(np.arange(1.0,PHASE2_ROUNDS+1.0)); spline_names=["round_linear","ns2","ns3","ns4"]
    for model,frame in rounds.groupby("model_name"):
        trials=common_trials(rounds,model); frame=frame[frame.trial_id.isin(trials)].copy()
        if not len(frame): continue
        basis=spline[frame["round"].astype(int).to_numpy()-1]
        columns=[]; names=[]
        for rvalue,rname in ((0.0,"fixed"),(1.0,"variable")):
            rind=(frame.return_condition.eq("variable").astype(float).to_numpy() if rvalue else frame.return_condition.eq("fixed").astype(float).to_numpy())
            for bridge in BRIDGES:
                bind=frame.bridge.eq(bridge).astype(float).to_numpy()
                for hpower,hname in ((0,"intercept"),(1,"honesty")):
                    h=np.ones(len(frame)) if hpower==0 else frame.honesty_value.astype(float).to_numpy()
                    for j,bname in enumerate(["intercept",*spline_names]):
                        b=np.ones(len(frame)) if j==0 else basis[:,j-1]
                        columns.append(rind*bind*h*b); names.append(f"{rname}:{bridge}:{hname}:{bname}")
        X=np.column_stack(columns); y=frame.investment.astype(float).to_numpy(); beta=np.linalg.lstsq(X,y,rcond=None)[0]
        bread=np.linalg.pinv(X.T@X); residual=y-X@beta; meat=np.zeros((X.shape[1],X.shape[1]))
        for trial in trials:
            mask=frame.trial_id.to_numpy()==trial; score=X[mask].T@residual[mask]; meat+=np.outer(score,score)
        covariance=bread@meat@bread; se=np.sqrt(np.maximum(np.diag(covariance),0))
        for name,estimate,error in zip(names,beta,se): output.append({"model_name":model,"term":name,"estimate":estimate,"cluster_se":error,"ci_low":estimate-1.96*error,"ci_high":estimate+1.96*error,"n_trial_clusters":len(trials),"spline_df":4})
    return pd.DataFrame(output)
