#!/usr/bin/env python3
"""Model-level B025 association analysis under the B024 source amendment."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

BENCHMARKS = [
    "SOTOPIA",
    "AgentSense",
    "SocKET_trustworthiness",
    "SocialEval",
    "RULER_aggregation_4k",
    "IFEval",
    "BBEH",
]
OUTCOMES = {
    "Y1": "Y1_b0_G1",
    "Y2": "Y2_b3_G1",
    "Y3": "Y3_G1_b3_minus_b0",
    "Y4": "Y4_gradient_auc_signed_b3_minus_b0",
}
CONFIRMATORY = {
    ("RULER_aggregation_4k", "Y1"): "H1",
    ("SOTOPIA", "Y2"): "H3a",
    ("AgentSense", "Y2"): "H3b",
    ("SocialEval", "Y2"): "H4",
    ("IFEval", "Y3"): "H5",
}
GROUPS = {
    "gemma-4-31b-it": ("Gemma", "Google"),
    "qwen3-14b": ("Qwen3", "Alibaba-Qwen"),
    "qwen3-32b": ("Qwen3", "Alibaba-Qwen"),
    "qwen3.5-4b": ("Qwen3.5", "Alibaba-Qwen"),
    "qwen3.5-9b": ("Qwen3.5", "Alibaba-Qwen"),
    "qwen3.5-27b": ("Qwen3.5", "Alibaba-Qwen"),
    "qwen3.8-27b": ("Qwen3.8", "Alibaba-Qwen"),
    "deepseek-v4.1-flash": ("DeepSeek-V4", "DeepSeek"),
    "deepseek-v4-pro": ("DeepSeek-V4", "DeepSeek"),
}
PERMUTATION_SEED = 20260916
BOOTSTRAP_SEED = 20260917
MONTE_CARLO_ITERATIONS = 100_000
BOOTSTRAP_ITERATIONS = 10_000
MIN_INFERENTIAL_N = 5


def finite_float(value: str | None) -> float | None:
    try:
        result = float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None
    return result if result is not None and math.isfinite(result) else None


def read_behavior(path: Path) -> dict[str, dict[str, float | None]]:
    result: dict[str, dict[str, float | None]] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            result[row["model"]] = {
                key: finite_float(row.get(column)) for key, column in OUTCOMES.items()
            }
    return result


def read_capability(path: Path) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if row.get("status") != "complete":
                continue
            value = finite_float(row.get("primary_score"))
            if value is None:
                continue
            result.setdefault(row["model"], {})[row["benchmark"]] = value
    return result


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    value = stats.spearmanr(x, y).statistic
    return float(value) if np.isfinite(value) else math.nan


def permutation_p(x: np.ndarray, y: np.ndarray, observed: float) -> tuple[float, str, int]:
    n = len(x)
    if not np.isfinite(observed):
        return math.nan, "undefined", 0
    threshold = abs(observed) - 1e-15
    if math.factorial(n) <= MONTE_CARLO_ITERATIONS:
        total = 0
        extreme = 0
        for perm in itertools.permutations(y.tolist()):
            candidate = spearman(x, np.asarray(perm, dtype=float))
            total += 1
            if np.isfinite(candidate) and abs(candidate) >= threshold:
                extreme += 1
        return extreme / total, "exact", total
    rng = np.random.default_rng(PERMUTATION_SEED)
    extreme = 0
    for _ in range(MONTE_CARLO_ITERATIONS):
        candidate = spearman(x, rng.permutation(y))
        if np.isfinite(candidate) and abs(candidate) >= threshold:
            extreme += 1
    return (extreme + 1) / (MONTE_CARLO_ITERATIONS + 1), "monte_carlo", MONTE_CARLO_ITERATIONS


def bootstrap_ci(x: np.ndarray, y: np.ndarray) -> tuple[float, float, int]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    values: list[float] = []
    n = len(x)
    for _ in range(BOOTSTRAP_ITERATIONS):
        indexes = rng.integers(0, n, size=n)
        candidate = spearman(x[indexes], y[indexes])
        if np.isfinite(candidate):
            values.append(candidate)
    if not values:
        return math.nan, math.nan, 0
    low, high = np.quantile(np.asarray(values), [0.025, 0.975])
    return float(low), float(high), len(values)


def group_sensitivity(
    models: list[str],
    x: np.ndarray,
    y: np.ndarray,
    label: str,
    getter: Callable[[str], str],
) -> list[dict[str, object]]:
    rows = []
    for group in sorted({getter(model) for model in models}):
        keep = np.asarray([getter(model) != group for model in models])
        value = spearman(x[keep], y[keep]) if int(keep.sum()) >= 3 else math.nan
        rows.append(
            {
                "analysis": label,
                "left_out": group,
                "n": int(keep.sum()),
                "spearman_rho": value,
            }
        )
    return rows


def analyze_pair(
    benchmark: str,
    outcome: str,
    behavior: dict[str, dict[str, float | None]],
    capability: dict[str, dict[str, float]],
) -> tuple[dict[str, object], list[dict[str, object]], tuple[list[str], np.ndarray, np.ndarray]]:
    models = [
        model
        for model in GROUPS
        if behavior.get(model, {}).get(outcome) is not None
        and benchmark in capability.get(model, {})
    ]
    x = np.asarray([capability[model][benchmark] for model in models], dtype=float)
    y = np.asarray([behavior[model][outcome] for model in models], dtype=float)
    rho = spearman(x, y) if len(models) >= 3 else math.nan
    tau = float(stats.kendalltau(x, y, variant="b").statistic) if len(models) >= 3 else math.nan
    pearson = (
        float(stats.pearsonr(x, y).statistic)
        if len(models) >= 3 and np.std(x) > 0 and np.std(y) > 0
        else math.nan
    )
    if len(models) >= MIN_INFERENTIAL_N:
        perm_p, perm_method, perm_n = permutation_p(x, y, rho)
        ci_low, ci_high, bootstrap_valid = bootstrap_ci(x, y)
    else:
        perm_p, perm_method, perm_n = math.nan, "not_run_n_lt_5", 0
        ci_low, ci_high, bootstrap_valid = math.nan, math.nan, 0

    robustness: list[dict[str, object]] = []
    if len(models) >= 4:
        for index, model in enumerate(models):
            keep = np.arange(len(models)) != index
            robustness.append(
                {
                    "analysis": "leave_one_model_out",
                    "left_out": model,
                    "n": int(keep.sum()),
                    "spearman_rho": spearman(x[keep], y[keep]),
                }
            )
        robustness.extend(
            group_sensitivity(models, x, y, "leave_one_family_out", lambda m: GROUPS[m][0])
        )
        robustness.extend(
            group_sensitivity(models, x, y, "leave_one_vendor_out", lambda m: GROUPS[m][1])
        )
    for row in robustness:
        row["benchmark"] = benchmark
        row["outcome"] = outcome

    hypothesis = CONFIRMATORY.get((benchmark, outcome))
    result = {
        "benchmark": benchmark,
        "outcome": outcome,
        "hypothesis": hypothesis or "",
        "analysis_class": "amendment_confirmatory" if hypothesis else "exploratory",
        "n_models": len(models),
        "models": ";".join(models),
        "spearman_rho": rho,
        "spearman_ci_low": ci_low,
        "spearman_ci_high": ci_high,
        "bootstrap_valid": bootstrap_valid,
        "permutation_p": perm_p,
        "permutation_method": perm_method,
        "permutation_iterations": perm_n,
        "kendall_tau_b": tau,
        "pearson_r": pearson,
        "inferential_status": "complete"
        if len(models) >= MIN_INFERENTIAL_N
        else "descriptive_n_lt_5",
    }
    return result, robustness, (models, x, y)


def bh_adjust(rows: list[dict[str, object]]) -> None:
    valid = [
        (index, float(row["permutation_p"]))
        for index, row in enumerate(rows)
        if row["analysis_class"] == "exploratory" and math.isfinite(float(row["permutation_p"]))
    ]
    valid.sort(key=lambda item: item[1])
    m = len(valid)
    adjusted = [math.nan] * len(rows)
    running = 1.0
    for rank_from_end, (index, p_value) in enumerate(reversed(valid), start=1):
        rank = m - rank_from_end + 1
        running = min(running, p_value * m / rank)
        adjusted[index] = min(1.0, running)
    for index, row in enumerate(rows):
        row["exploratory_bh_q"] = adjusted[index]
        row["exploratory_fdr_0_10"] = (
            bool(adjusted[index] <= 0.10) if math.isfinite(adjusted[index]) else ""
        )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_pair(
    path: Path,
    benchmark: str,
    outcome: str,
    models: list[str],
    x: np.ndarray,
    y: np.ndarray,
    rho: float,
) -> None:
    if len(models) < 3:
        return
    figure, axis = plt.subplots(figsize=(6.2, 4.8), constrained_layout=True)
    axis.scatter(x, y, color="#2C6EBA", s=52)
    for model, x_value, y_value in zip(models, x, y, strict=False):
        axis.annotate(
            model, (x_value, y_value), xytext=(4, 4), textcoords="offset points", fontsize=7
        )
    axis.set_xlabel(benchmark)
    axis.set_ylabel(outcome)
    axis.set_title(f"{benchmark} vs {outcome} (Spearman rho={rho:.3f})")
    axis.grid(alpha=0.25)
    figure.savefig(path, dpi=180)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capability-scores", type=Path)
    parser.add_argument("--behavior-metrics", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    cap_path = (args.capability_scores or root / "derived" / "capability_scores.csv").resolve()
    beh_path = (args.behavior_metrics or root / "derived" / "behavior_metrics.csv").resolve()
    table_dir = root / "reports" / "tables"
    figure_dir = root / "reports" / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    status = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "capability_scores": cap_path.as_posix(),
        "behavior_metrics": beh_path.as_posix(),
        "frozen_settings": {
            "primary": "spearman",
            "permutation_iterations": MONTE_CARLO_ITERATIONS,
            "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
            "minimum_inferential_n": MIN_INFERENTIAL_N,
            "exploratory_fdr_q": 0.10,
        },
    }
    if not cap_path.exists():
        status.update({"status": "blocked", "blocker": "capability_scores.csv does not exist"})
        (root / "derived" / "analysis_status.json").write_text(
            json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(status, ensure_ascii=False, indent=2))
        return 2

    behavior = read_behavior(beh_path)
    capability = read_capability(cap_path)
    results: list[dict[str, object]] = []
    robustness: list[dict[str, object]] = []
    for benchmark in BENCHMARKS:
        for outcome in OUTCOMES:
            row, sensitivity, plotting = analyze_pair(benchmark, outcome, behavior, capability)
            results.append(row)
            robustness.extend(sensitivity)
            models, x, y = plotting
            plot_pair(
                figure_dir / f"{benchmark}__{outcome}.png",
                benchmark,
                outcome,
                models,
                x,
                y,
                float(row["spearman_rho"]),
            )
    bh_adjust(results)
    write_csv(table_dir / "correlation_matrix_long.csv", results)
    write_csv(
        table_dir / "confirmatory_tests.csv",
        [row for row in results if row["analysis_class"] == "amendment_confirmatory"],
    )
    write_csv(
        table_dir / "exploratory_tests.csv",
        [row for row in results if row["analysis_class"] == "exploratory"],
    )
    write_csv(table_dir / "robustness.csv", robustness)
    inferential_pairs = sum(row["inferential_status"] == "complete" for row in results)
    status.update(
        {
            "status": (
                "complete"
                if inferential_pairs == len(results)
                else "blocked_incomplete_capability_matrix"
            ),
            "pairs": len(results),
            "inferential_pairs": inferential_pairs,
            "behavior_source": "B024 (see input provenance)",
            "analysis_design_status": "post_results_user_requested_amendment",
            "h2_status": "not_tested_no_prespecified_B024_explicit_social_discrimination_metric",
        }
    )
    if inferential_pairs != len(results):
        status["blocker"] = (
            "At least one benchmark-outcome pair has fewer than five complete models."
        )
    (root / "derived" / "analysis_status.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return 0 if status["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
