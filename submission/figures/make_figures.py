"""Figures for the README and the pitch. Run: uv run python figures/make_figures.py

Numbers come from our runs (official `score_compiler`, our benchmark table, the VQE comparison).
Style: white background, one accent colour (ours), one grey (baselines), values labelled directly.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
# Palette: deep navy (dominant), laser orange (ours / highlights), grey (baselines). Backgrounds transparent.
NAVY = "#14283D"
OURS = "#D9622B"
BASE = "#9AA0A6"
INK = NAVY
MUTED = "#5B6B7A"
GRID = "#E3E7EB"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 13,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "none",
        "text.color": INK,
        "axes.facecolor": "none",
    }
)


def _save(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", dpi=200, transparent=True)
    fig.savefig(OUT / f"{name}.svg", transparent=True)
    plt.close(fig)


def score_progression() -> None:
    versions = ["v0", "v1", "v2", "v3", "v4", "v5"]
    notes = [
        "valid\npipeline",
        "merged moves,\natoms stay up",
        "layouts,\nshifts, fusion",
        "ZZ grouping",
        "2D layout",
        "lookahead",
    ]
    score = [65.4, 76.0, 76.9, 77.3, 77.5, 77.7]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(range(len(score)), score, color=OURS, linewidth=2, marker="o", markersize=8, zorder=3)
    for i, s in enumerate(score):
        ax.annotate(
            f"{s:.1f}%", (i, s), textcoords="offset points", xytext=(0, 10), ha="center", color=INK, fontsize=12
        )
    ax.set_xticks(range(len(score)), [f"{v}\n{n}" for v, n in zip(versions, notes, strict=True)], fontsize=10)
    ax.set_ylim(60, 80)
    ax.set_ylabel("% better than the naive compiler")
    ax.yaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    ax.set_title("Official score per compiler version", loc="left", color=INK, fontsize=15)
    _save(fig, "score_progression")


def moves_before_after() -> None:
    circuits = ["12-qubit iHVA", "50-qubit HEA", "12-qubit HEA"]
    first = [232, 196, 44]
    final = [114, 15, 6]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    y = range(len(circuits))
    h = 0.36
    ax.barh([i + h / 2 + 0.02 for i in y], first, height=h, color=BASE, label="first valid version (v0)")
    ax.barh([i - h / 2 - 0.02 for i in y], final, height=h, color=OURS, label="final version")
    for i in y:
        ax.text(first[i] + 3, i + h / 2 + 0.02, str(first[i]), va="center", color=INK, fontsize=12)
        ax.text(final[i] + 3, i - h / 2 - 0.02, str(final[i]), va="center", color=INK, fontsize=12)
    ax.set_yticks(list(y), circuits)
    ax.set_xlabel("atom moves")
    ax.xaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("Fewer, fuller moves", loc="left", color=INK, fontsize=15)
    _save(fig, "moves_before_after")


def vqe_success() -> None:
    labels = ["No entanglement (Rx only)", "Our 8-qubit ansatz"]
    values = [9, 19]
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.barh([0, 1], values, height=0.5, color=[BASE, OURS])
    for i, v in enumerate(values):
        ax.text(v + 0.3, i, f"{v} / 20", va="center", color=INK, fontsize=13)
    ax.set_yticks([0, 1], labels)
    ax.set_xlim(0, 21)
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.set_xlabel("random starts that reach the optimum (cost 14), out of 20")
    ax.xaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    _save(fig, "vqe_success")


def compiled_cost_of_ansatze() -> None:
    labels = ["12-qubit iHVA (given)", "12-qubit HEA (given)", "Our 12-qubit ansatz", "Our 8-qubit ansatz"]
    values = [284.8, 12.4, 16.5, 5.3]
    colors = [BASE, BASE, OURS, OURS]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(range(len(labels)), values, height=0.55, color=colors)
    for i, v in enumerate(values):
        ax.text(v + 3, i, f"{v:g}", va="center", color=INK, fontsize=12)
    ax.set_yticks(range(len(labels)), labels)
    ax.invert_yaxis()
    ax.set_xlabel("total move distance after compilation (same compiler)")
    ax.xaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    ax.set_title("What each circuit costs to run", loc="left", color=INK, fontsize=15)
    _save(fig, "compiled_cost_of_ansatze")


if __name__ == "__main__":
    score_progression()
    moves_before_after()
    vqe_success()
    compiled_cost_of_ansatze()
    print("figures written to", OUT)
