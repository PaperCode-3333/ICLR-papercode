#!/usr/bin/env python3
"""ICLR-ready B029 allocation trajectories for all models.

Creates six 4x2 quantitative-grid figures (three bridges x two return
conditions). Each model panel shows the four honesty conditions with paired
trial-cluster bootstrap 95% confidence bands. Raw run files are read-only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from PIL import Image

from b029_stats import _counts, common_trials, group_matrix, read_run
from b029_stimuli import ROOT, design


MODELS = [
    "qwen3-14b",
    "qwen3-32b",
    "qwen3.5-4b",
    "qwen3.5-9b",
    "qwen3.5-27b",
    "qwen3.8-27b",
    "deepseek-v4.1-flash",
    "deepseek-v4-pro",
]
MODEL_LABELS = {
    "qwen3-14b": "Qwen3-14B",
    "qwen3-32b": "Qwen3-32B",
    "qwen3.5-4b": "Qwen3.5-4B",
    "qwen3.5-9b": "Qwen3.5-9B",
    "qwen3.5-27b": "Qwen3.5-27B",
    "qwen3.8-27b": "Qwen3.8-27B",
    "deepseek-v4.1-flash": "DeepSeek V4.1 Flash",
    "deepseek-v4-pro": "DeepSeek V4 Pro",
}
BRIDGES = ["bridge_0", "bridge_1", "bridge_3"]
RETURNS = ["fixed", "variable"]
HONESTY = [0.0, 0.25, 0.75, 1.0]
HONESTY_LABELS = {0.0: "0%", 0.25: "25%", 0.75: "75%", 1.0: "100%"}

# Okabe-Ito-derived cool-to-warm sequence plus redundant line/marker coding.
COLORS = {0.0: "#0072B2", 0.25: "#56B4E9", 0.75: "#E69F00", 1.0: "#D55E00"}
LINESTYLES = {0.0: "-", 0.25: "--", 0.75: "-.", 1.0: ":"}
MARKERS = {0.0: "o", 0.25: "s", 0.75: "^", 1.0: "D"}
PANEL_LETTERS = list("abcdefgh")
ROUNDS = np.arange(1, 21)
BOOTSTRAP_SAMPLES = 10_000
FIGURE_WIDTH_IN = 5.5  # ICLR full text width.
FIGURE_HEIGHT_IN = 7.15
PNG_DPI = 600


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans", "sans-serif"],
            "font.size": 7.0,
            "axes.titlesize": 7.5,
            "axes.labelsize": 7.2,
            "xtick.labelsize": 6.4,
            "ytick.labelsize": 6.4,
            "legend.fontsize": 6.6,
            "axes.linewidth": 0.65,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_formal_run(run_dir: Path, spec: dict[str, Any], rounds: pd.DataFrame) -> None:
    progress = json.loads((run_dir / "progress.json").read_text(encoding="utf-8"))
    if progress.get("missing") != 0 or progress.get("service_failure") != 0:
        raise RuntimeError(f"formal run is incomplete: {progress}")
    if progress.get("valid", 0) + progress.get("excluded", 0) != progress.get("target"):
        raise RuntimeError("terminal-session count does not match target")
    if spec.get("synthetic") or spec.get("dry_run"):
        raise RuntimeError("synthetic or dry-run data cannot be used for the paper figure")
    if set(rounds["model_name"].unique()) != set(MODELS):
        raise RuntimeError("formal round data do not contain exactly the eight frozen models")
    if not (rounds.groupby("session_id").size() == 20).all():
        raise RuntimeError("at least one valid session does not contain exactly 20 rounds")


def calculate_source_data(rounds: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    rows: list[dict[str, Any]] = []
    trial_counts: dict[str, int] = {}
    for model in MODELS:
        trials = common_trials(rounds, model)
        if not trials:
            raise RuntimeError(f"no complete paired trials for {model}")
        trial_counts[model] = len(trials)
        weights = _counts(model, len(trials), BOOTSTRAP_SAMPLES)
        denominator = weights.sum(axis=1, keepdims=True)
        for bridge in BRIDGES:
            for return_condition in RETURNS:
                for honesty in HONESTY:
                    matrix = group_matrix(rounds, model, bridge, return_condition, honesty, trials)
                    mean = matrix.mean(axis=0)
                    draws = (weights @ matrix) / denominator
                    low, high = np.quantile(draws, [0.025, 0.975], axis=0)
                    for index, round_no in enumerate(ROUNDS):
                        rows.append(
                            {
                                "model_name": model,
                                "model_label": MODEL_LABELS[model],
                                "bridge": bridge,
                                "return_condition": return_condition,
                                "honesty_value": honesty,
                                "honesty": HONESTY_LABELS[honesty],
                                "round": int(round_no),
                                "mean_allocation": float(mean[index]),
                                "ci_low": float(low[index]),
                                "ci_high": float(high[index]),
                                "n_complete_paired_trials": len(trials),
                                "bootstrap_samples": BOOTSTRAP_SAMPLES,
                                "bootstrap_seed": int(design()["bootstrap_seed"]),
                                "interval": "paired trial-cluster percentile 95% CI",
                            }
                        )
    frame = pd.DataFrame(rows)
    expected = len(MODELS) * len(BRIDGES) * len(RETURNS) * len(HONESTY) * len(ROUNDS)
    if len(frame) != expected or frame.isna().any().any():
        raise RuntimeError(f"invalid source-data table: rows={len(frame)}, expected={expected}")
    return frame, trial_counts


def legend_handles() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            color=COLORS[h],
            linestyle=LINESTYLES[h],
            marker=MARKERS[h],
            markersize=3.0,
            linewidth=1.25,
            label=f"Honesty {HONESTY_LABELS[h]}",
        )
        for h in HONESTY
    ]


def plot_one(
    source: pd.DataFrame,
    bridge: str,
    return_condition: str,
    out_dir: Path,
    trial_counts: dict[str, int],
) -> list[Path]:
    fig, axes = plt.subplots(
        4,
        2,
        figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN),
        sharex=True,
        sharey=True,
        constrained_layout=False,
    )
    for index, (model, ax) in enumerate(zip(MODELS, axes.flat)):
        panel = source[
            (source["model_name"] == model)
            & (source["bridge"] == bridge)
            & (source["return_condition"] == return_condition)
        ]
        for honesty in HONESTY:
            data = panel[panel["honesty_value"] == honesty].sort_values("round")
            x = data["round"].to_numpy()
            mean = data["mean_allocation"].to_numpy()
            low = data["ci_low"].to_numpy()
            high = data["ci_high"].to_numpy()
            ax.fill_between(x, low, high, color=COLORS[honesty], alpha=0.11, linewidth=0)
            ax.plot(
                x,
                mean,
                color=COLORS[honesty],
                linestyle=LINESTYLES[honesty],
                marker=MARKERS[honesty],
                markevery=[0, 4, 9, 14, 19],
                markersize=2.5,
                markeredgewidth=0.35,
                linewidth=1.15,
                solid_capstyle="round",
            )
        ax.set_title(f"{MODEL_LABELS[model]}  ($n$={trial_counts[model]})", pad=3.0, fontweight="bold")
        ax.text(
            -0.13,
            1.06,
            PANEL_LETTERS[index],
            transform=ax.transAxes,
            fontsize=8.0,
            fontweight="bold",
            va="bottom",
            ha="left",
        )
        ax.set_xlim(1, 20)
        ax.set_ylim(0.7, 10.3)
        ax.set_xticks([1, 5, 10, 15, 20])
        ax.set_yticks([1, 4, 7, 10])
        ax.tick_params(direction="out", pad=1.5)
        if index % 2 == 0:
            ax.set_ylabel("Allocation")
        if index >= 6:
            ax.set_xlabel("Round")

    fig.legend(
        handles=legend_handles(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=4,
        columnspacing=1.25,
        handlelength=2.5,
        handletextpad=0.45,
    )
    fig.text(
        0.5,
        0.018,
        "Lines show means; shading shows paired trial-cluster bootstrap 95% CIs (10,000 resamples).",
        ha="center",
        va="bottom",
        fontsize=5.8,
        color="#4D4D4D",
    )
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.075, top=0.925, hspace=0.38, wspace=0.18)
    stem = f"B029_{bridge}_{return_condition}_allocation_trajectories_8models"
    outputs = [out_dir / f"{stem}.svg", out_dir / f"{stem}.pdf", out_dir / f"{stem}.png"]
    fig.savefig(outputs[0], bbox_inches=None, metadata={"Creator": "B029 Python figure workflow"})
    fig.savefig(outputs[1], bbox_inches=None, metadata={"Creator": "B029 Python figure workflow"})
    fig.savefig(
        outputs[2],
        dpi=PNG_DPI,
        bbox_inches=None,
        metadata={
            "Creator": "B029 Python figure workflow",
            "Description": f"{bridge}, {return_condition}; mean allocation and paired-bootstrap 95% CI",
        },
    )
    plt.close(fig)
    return outputs


def image_qa(paths: list[Path]) -> dict[str, dict[str, Any]]:
    qa: dict[str, dict[str, Any]] = {}
    expected_size = (round(FIGURE_WIDTH_IN * PNG_DPI), round(FIGURE_HEIGHT_IN * PNG_DPI))
    for path in paths:
        if path.suffix != ".png":
            continue
        with Image.open(path) as image:
            array = np.asarray(image.convert("RGB"))
            nonwhite = np.any(array < 250, axis=2)
            rows, cols = np.where(nonwhite)
            edge_ink = bool(nonwhite[0, :].any() or nonwhite[-1, :].any() or nonwhite[:, 0].any() or nonwhite[:, -1].any())
            qa[path.name] = {
                "pixel_size": list(image.size),
                "expected_pixel_size": list(expected_size),
                "mode": image.mode,
                "nonwhite_bbox_px": [int(cols.min()), int(rows.min()), int(cols.max()), int(rows.max())],
                "ink_touches_canvas_edge": edge_ink,
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            if image.size != expected_size or edge_ink:
                raise RuntimeError(f"PNG QA failed for {path.name}: {qa[path.name]}")
    return qa


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--out-dir")
    args = parser.parse_args()
    configure_matplotlib()
    run_dir = ROOT / "runs" / args.run_id
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "analysis" / "figures_iclr_bridge_trajectories"
    out_dir.mkdir(parents=True, exist_ok=True)
    spec, _sessions, _excluded, rounds = read_run(run_dir)
    validate_formal_run(run_dir, spec, rounds)
    source, trial_counts = calculate_source_data(rounds)
    source_path = out_dir / "B029_bridge_allocation_trajectories_source_data.csv"
    source.to_csv(source_path, index=False)
    output_paths: list[Path] = []
    for bridge in BRIDGES:
        for return_condition in RETURNS:
            output_paths.extend(plot_one(source, bridge, return_condition, out_dir, trial_counts))
    qa = image_qa(output_paths)
    contract = {
        "core_conclusion": "The figures show how honesty-conditioned allocation trajectories evolve over 20 rounds across all eight models under each bridge and return condition.",
        "figure_archetype": "quantitative grid",
        "target": "ICLR full-width figure",
        "backend": "Python/matplotlib only",
        "final_size_inches": [FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN],
        "png_dpi": PNG_DPI,
        "panels_per_figure": 8,
        "statistics": "mean and paired trial-cluster percentile 95% CI; 10,000 synchronized resamples per model",
        "source_data": source_path.name,
        "trial_counts": trial_counts,
        "excluded_sessions_are_not_imputed": True,
        "model_order": MODELS,
        "honesty_encoding": {
            HONESTY_LABELS[h]: {"color": COLORS[h], "linestyle": LINESTYLES[h], "marker": MARKERS[h]}
            for h in HONESTY
        },
        "files": [path.name for path in output_paths],
        "png_qa": qa,
    }
    (out_dir / "figure_manifest_and_qa.json").write_text(
        json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(contract, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
