#!/usr/bin/env python3
"""Nature-style first-20-round allocation trajectories for B029 or B027."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT.name
MODELS_PATH = ROOT / "conditions" / "models.json"
DESIGN_PATH = ROOT / "conditions" / "design.json"

BRIDGES = ("bridge_0", "bridge_1", "bridge_3")
RETURNS = ("fixed", "variable")
HONESTY = (0.0, 0.25, 0.75, 1.0)
PLOT_ROUNDS = 20
HONESTY_LABELS = {
    0.0: "0% · completely dishonest",
    0.25: "25% · mostly dishonest",
    0.75: "75% · mostly honest",
    1.0: "100% · completely honest",
}
COLORS = {
    0.0: "#9E3D34",
    0.25: "#D28B40",
    0.75: "#4D8FA8",
    1.0: "#2F4B7C",
}
LINESTYLES = {0.0: "-", 0.25: "--", 0.75: "-.", 1.0: ":"}
MARKERS = {0.0: "o", 0.25: "s", 0.75: "^", 1.0: "D"}
BRIDGE_TITLES = {
    "bridge_0": "Bridge 0",
    "bridge_1": "Bridge 1",
    "bridge_3": "Bridge 3",
}
RETURN_TITLES = {"fixed": "Fixed returns", "variable": "Variable returns"}
PANEL_LABELS = tuple("abcdefgh")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=None)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_condition_seed(base_seed: int, parts: Iterable[object]) -> int:
    payload = "|".join(map(str, parts)).encode("utf-8")
    offset = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return (int(base_seed) + offset) % (2**63 - 1)


def validate_rounds(
    rounds: pd.DataFrame,
    expected_models: Tuple[str, ...],
    phase2_rounds: int,
) -> dict:
    required = {
        "session_id", "trial_id", "model_name", "bridge", "honesty",
        "honesty_value", "return_condition", "round", "investment",
        "parse_status",
    }
    missing = sorted(required - set(rounds.columns))
    if missing:
        raise ValueError(f"raw_rounds.csv missing columns: {missing}")
    if set(rounds["model_name"]) != set(expected_models):
        raise ValueError("model set differs from the frozen experiment design")
    if set(rounds["bridge"]) != set(BRIDGES):
        raise ValueError("bridge set differs from the frozen experiment design")
    if set(rounds["return_condition"]) != set(RETURNS):
        raise ValueError("return-condition set differs from the frozen experiment design")
    if set(rounds["honesty_value"].astype(float)) != set(HONESTY):
        raise ValueError("honesty set differs from the frozen experiment design")
    if not rounds["parse_status"].eq("valid").all():
        raise ValueError("raw_rounds.csv contains non-valid rows")
    if not rounds["investment"].between(1, 10).all():
        raise ValueError("investment outside the experimental range [1, 10]")
    if rounds.duplicated(["session_id", "round"]).any():
        raise ValueError("duplicate session-round rows detected")

    session_sizes = rounds.groupby("session_id", sort=False)["round"].agg(
        ["count", "nunique", "min", "max"]
    )
    complete = (
        session_sizes["count"].eq(phase2_rounds)
        & session_sizes["nunique"].eq(phase2_rounds)
        & session_sizes["min"].eq(1)
        & session_sizes["max"].eq(phase2_rounds)
    )
    if not complete.all():
        raise ValueError(
            f"{int((~complete).sum())} incomplete sessions found in raw_rounds.csv"
        )
    expected_cells = len(expected_models) * len(BRIDGES) * len(RETURNS) * len(HONESTY)
    cells = rounds[
        ["model_name", "bridge", "return_condition", "honesty_value"]
    ].drop_duplicates()
    if len(cells) != expected_cells:
        raise ValueError(f"expected {expected_cells} condition cells, found {len(cells)}")
    return {
        "valid_round_rows": int(len(rounds)),
        "complete_sessions": int(rounds["session_id"].nunique()),
        "cell_count": int(len(cells)),
        "full_session_rounds": int(phase2_rounds),
        "plotted_rounds": PLOT_ROUNDS,
    }


def bootstrap_trajectories(
    rounds: pd.DataFrame,
    model_order: Tuple[str, ...],
    samples: int,
    seed: int,
) -> pd.DataFrame:
    plot_data = rounds[rounds["round"].between(1, PLOT_ROUNDS)].copy()
    rows = []
    keys = ["model_name", "bridge", "return_condition", "honesty_value"]
    grouped = plot_data.groupby(keys, sort=False, observed=True)
    for model in model_order:
        for bridge in BRIDGES:
            for return_condition in RETURNS:
                for honesty_value in HONESTY:
                    key = (model, bridge, return_condition, honesty_value)
                    group = grouped.get_group(key)
                    pivot = (
                        group.pivot(index="trial_id", columns="round", values="investment")
                        .reindex(columns=range(1, PLOT_ROUNDS + 1))
                        .dropna(axis=0, how="any")
                        .sort_index()
                    )
                    if pivot.empty:
                        raise ValueError(f"no complete first-20-round trials for {key}")
                    values = pivot.to_numpy(dtype=np.float64)
                    n_trials = values.shape[0]
                    rng = np.random.default_rng(stable_condition_seed(seed, key))
                    indices = rng.integers(
                        0, n_trials, size=(samples, n_trials), endpoint=False
                    )
                    draws = values[indices].mean(axis=1)
                    means = values.mean(axis=0)
                    lower, upper = np.quantile(draws, [0.025, 0.975], axis=0)
                    honesty_text = str(group["honesty"].iloc[0])
                    for round_number, mean, lo, hi in zip(
                        range(1, PLOT_ROUNDS + 1), means, lower, upper
                    ):
                        rows.append(
                            {
                                "experiment": EXPERIMENT,
                                "model_name": model,
                                "bridge": bridge,
                                "return_condition": return_condition,
                                "honesty": honesty_text,
                                "honesty_value": float(honesty_value),
                                "round": int(round_number),
                                "n_trials": int(n_trials),
                                "mean_investment": float(mean),
                                "ci_low": float(lo),
                                "ci_high": float(hi),
                            }
                        )
    result = pd.DataFrame(rows)
    expected_rows = (
        len(model_order) * len(BRIDGES) * len(RETURNS)
        * len(HONESTY) * PLOT_ROUNDS
    )
    if len(result) != expected_rows:
        raise AssertionError(f"expected {expected_rows} rows, found {len(result)}")
    return result


def apply_style() -> None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [
        "Arial", "DejaVu Sans", "Liberation Sans", "sans-serif"
    ]
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["font.size"] = 8
    plt.rcParams["axes.titlesize"] = 9
    plt.rcParams["axes.labelsize"] = 9
    plt.rcParams["xtick.labelsize"] = 7
    plt.rcParams["ytick.labelsize"] = 7
    plt.rcParams["legend.fontsize"] = 8
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["legend.frameon"] = False
    plt.rcParams["savefig.facecolor"] = "white"


def make_figure(
    summary: pd.DataFrame,
    model_order: Tuple[str, ...],
    model_labels: Dict[str, str],
    bridge: str,
    return_condition: str,
    out_dir: Path,
    dpi: int,
    samples: int,
) -> Tuple[Path, Path]:
    subset = summary[
        summary["bridge"].eq(bridge)
        & summary["return_condition"].eq(return_condition)
    ].copy()
    fig, axes = plt.subplots(3, 3, figsize=(9.0, 8.1), sharex=True, sharey=True)
    axes_flat = axes.ravel()

    for index, model in enumerate(model_order):
        ax = axes_flat[index]
        part = subset[subset["model_name"].eq(model)]
        for honesty_value in HONESTY:
            line = part[
                part["honesty_value"].eq(honesty_value)
            ].sort_values("round")
            x = line["round"].to_numpy()
            mean = line["mean_investment"].to_numpy()
            lower = line["ci_low"].to_numpy()
            upper = line["ci_high"].to_numpy()
            ax.fill_between(
                x, lower, upper, color=COLORS[honesty_value],
                alpha=0.14, linewidth=0, zorder=1,
            )
            ax.plot(
                x, mean, color=COLORS[honesty_value],
                linestyle=LINESTYLES[honesty_value], linewidth=1.65,
                marker=MARKERS[honesty_value], markevery=(4, 5),
                markersize=2.7, markeredgewidth=0, zorder=2,
            )

        n_values = sorted(part["n_trials"].unique())
        n_text = str(n_values[0]) if len(n_values) == 1 else f"{n_values[0]}–{n_values[-1]}"
        ax.set_title(model_labels[model], fontweight="bold", pad=5)
        ax.text(
            0.98, 0.96, f"n={n_text}", transform=ax.transAxes,
            ha="right", va="top", fontsize=6.8, color="#666666",
        )
        ax.text(
            -0.11, 1.04, PANEL_LABELS[index], transform=ax.transAxes,
            ha="left", va="bottom", fontsize=9, fontweight="bold",
        )
        ax.set_xlim(1, PLOT_ROUNDS)
        ax.set_ylim(0.5, 10.5)
        ax.set_xticks([1, 5, 10, 15, 20])
        ax.set_yticks([1, 3, 5, 7, 9])
        ax.tick_params(length=3, width=0.7, color="#555555")
        ax.spines["left"].set_color("#555555")
        ax.spines["bottom"].set_color("#555555")

    legend_ax = axes_flat[-1]
    legend_ax.set_axis_off()
    handles = [
        Line2D(
            [0], [0], color=COLORS[h], linestyle=LINESTYLES[h],
            marker=MARKERS[h], linewidth=1.8, markersize=4,
            label=HONESTY_LABELS[h],
        )
        for h in HONESTY
    ]
    legend_ax.legend(
        handles=handles, loc="upper left", bbox_to_anchor=(0.05, 0.92),
        title="Partner honesty", title_fontsize=9,
        handlelength=3.0, labelspacing=0.9,
    )
    min_n = int(subset["n_trials"].min())
    max_n = int(subset["n_trials"].max())
    legend_ax.text(
        0.06, 0.27,
        f"Shading: 95% trial-cluster\nbootstrap CI ({samples:,} resamples)\n"
        f"Complete 20-round trials only\nCurve-level n={min_n}–{max_n}",
        transform=legend_ax.transAxes, ha="left", va="top",
        fontsize=7.4, color="#4D4D4D", linespacing=1.45,
    )

    fig.suptitle(
        f"{EXPERIMENT} allocation trajectories (rounds 1–20) · "
        f"{BRIDGE_TITLES[bridge]} · {RETURN_TITLES[return_condition]}",
        x=0.5, y=0.985, fontsize=12, fontweight="bold",
    )
    fig.supxlabel("Round", x=0.49, y=0.035, fontsize=9.5)
    fig.supylabel("Allocation to partner", x=0.035, y=0.49, fontsize=9.5)
    fig.subplots_adjust(
        left=0.085, right=0.985, bottom=0.085, top=0.925,
        wspace=0.25, hspace=0.34,
    )

    stem = (
        f"{EXPERIMENT}_{bridge}_{return_condition}_"
        "allocation_trajectories_rounds01-20"
    )
    svg_path = out_dir / f"{stem}.svg"
    png_path = out_dir / f"{stem}.png"
    fig.savefig(svg_path, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return svg_path, png_path


def main() -> int:
    args = parse_args()
    design = load_json(DESIGN_PATH)
    models = load_json(MODELS_PATH)
    model_order = tuple(design["models"])
    model_labels = {name: models[name]["display_name"] for name in model_order}
    phase2_rounds = int(design["phase2_rounds"])
    if phase2_rounds < PLOT_ROUNDS:
        raise ValueError("experiment has fewer than 20 phase-two rounds")
    samples = int(args.bootstrap_samples or design["bootstrap_samples"])
    if samples != int(design["bootstrap_samples"]):
        raise ValueError("formal figure generation must use the frozen bootstrap count")
    seed = int(design["bootstrap_seed"])

    run_dir = ROOT / "runs" / args.run_id
    raw_path = run_dir / "raw_rounds.csv"
    if not raw_path.exists():
        raise FileNotFoundError(raw_path)
    out_dir = run_dir / "analysis" / "figures_bridge_trajectories_first20"
    out_dir.mkdir(parents=True, exist_ok=True)

    rounds = pd.read_csv(raw_path)
    audit = validate_rounds(rounds, model_order, phase2_rounds)
    summary = bootstrap_trajectories(rounds, model_order, samples, seed)
    source_path = out_dir / f"{EXPERIMENT}_first20_bridge_trajectory_source_data.csv"
    summary.to_csv(source_path, index=False)

    apply_style()
    svg_paths = []
    png_paths = []
    for bridge in BRIDGES:
        for return_condition in RETURNS:
            svg_path, png_path = make_figure(
                summary, model_order, model_labels, bridge,
                return_condition, out_dir, args.dpi, samples,
            )
            svg_paths.append(svg_path)
            png_paths.append(png_path)

    image_checks = {}
    for path in png_paths:
        with Image.open(path) as image:
            image_checks[path.name] = {
                "width_px": int(image.width),
                "height_px": int(image.height),
                "mode": image.mode,
                "dpi": list(image.info.get("dpi", ())),
                "sha256": sha256_file(path),
                "bytes": int(path.stat().st_size),
            }

    qa = {
        "experiment": EXPERIMENT,
        "run_id": args.run_id,
        "figure_contract": {
            "core_conclusion": (
                "Partner honesty produces model- and bridge-specific allocation "
                "trajectories during the first 20 rounds."
            ),
            "archetype": "quantitative grid",
            "backend": "Python/matplotlib",
            "panel_map": "eight model panels plus one shared legend/statistics panel",
            "output": "six bridge-by-return figures; SVG primary and PNG at 300 dpi",
        },
        "statistics": {
            "unit": "complete 20-round trial/session",
            "center": "mean investment at each plotted round",
            "interval": "two-sided percentile 95% trial-cluster bootstrap CI",
            "bootstrap_samples": samples,
            "bootstrap_seed_base": seed,
            "plotted_rounds": [1, PLOT_ROUNDS],
            "rounds_are_independent_samples": False,
            "missing_or_excluded_imputed": False,
        },
        "data_audit": audit,
        "source_data": {
            "path": str(source_path),
            "rows": int(len(summary)),
            "sha256": sha256_file(source_path),
            "raw_rounds_path": str(raw_path),
            "raw_rounds_sha256": sha256_file(raw_path),
        },
        "code": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "outputs": {
            "png": [str(path) for path in png_paths],
            "svg": [str(path) for path in svg_paths],
            "image_checks": image_checks,
        },
    }
    qa_path = out_dir / f"{EXPERIMENT}_first20_bridge_trajectory_QA.json"
    qa_path.write_text(
        json.dumps(qa, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(qa, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
