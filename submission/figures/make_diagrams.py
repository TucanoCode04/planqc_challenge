"""Explanatory diagrams for the pitch: the bus-stop graph, the device zones, and one merged move.

Run: uv run python figures/make_diagrams.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent
OURS = "#1A6FA8"
BASE = "#9AA0A6"
INK = "#222222"
MUTED = "#666666"
STORAGE_BG = "#EAF2F8"
ENTANGLE_BG = "#FBEADF"
plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": "white"})

# --- bus-stop graph (positions follow the figure in the challenge brief) -------------------------------------------
POS = {
    1: (1.75, 4.9), 2: (2.5, 4.9), 3: (2.12, 4.45),
    10: (2.13, 3.85), 11: (1.8, 3.3), 12: (2.45, 3.3),
    4: (0.45, 3.0), 6: (1.25, 2.9), 5: (0.95, 2.1),
    9: (2.95, 2.9), 7: (3.8, 3.0), 8: (3.3, 2.1),
}  # fmt: skip
TRIANGLES = [(1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12)]
BRIDGES = [(1, 4), (2, 7), (3, 10), (5, 8), (6, 11), (9, 12)]
KEPT = {1, 2, 5, 6, 7, 8, 10, 12}
CODES = {(1, 2, 3): "00", (4, 5, 6): "10", (7, 8, 9): "00", (10, 11, 12): "01"}
CODE_POS = {(1, 2, 3): (3.05, 4.75), (4, 5, 6): (0.2, 2.25), (7, 8, 9): (3.95, 2.25), (10, 11, 12): (2.95, 3.65)}


def bus_stop_graph() -> None:
    fig, ax = plt.subplots(figsize=(6, 5.2))
    for a, b in BRIDGES:
        (x1, y1), (x2, y2) = POS[a], POS[b]
        ax.plot([x1, x2], [y1, y2], color=BASE, lw=1.5, ls=(0, (4, 3)), zorder=1)
    for t in TRIANGLES:
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[0], t[2])):
            (x1, y1), (x2, y2) = POS[a], POS[b]
            ax.plot([x1, x2], [y1, y2], color=INK, lw=2, zorder=2)
    for s, (x, y) in POS.items():
        kept = s in KEPT
        ax.scatter(x, y, s=620, color=OURS if kept else "white", edgecolor=OURS if kept else BASE, lw=2, zorder=3)
        ax.text(x, y, str(s), ha="center", va="center", fontsize=12, color="white" if kept else MUTED, zorder=4)
    for t, (x, y) in CODE_POS.items():
        ax.text(x, y, CODES[t], ha="center", va="center", fontsize=13, family="DejaVu Sans Mono", color=OURS,
                bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": OURS, "lw": 1.2})  # fmt: skip
    ax.scatter([], [], s=120, color=OURS, label="stop kept")
    ax.scatter([], [], s=120, color="white", edgecolor=BASE, lw=2, label="stop dropped")
    ax.plot([], [], color=INK, lw=2, label="triangle edge")
    ax.plot([], [], color=BASE, lw=1.5, ls=(0, (4, 3)), label="bridge")
    ax.legend(loc="lower center", ncol=4, frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.06), handletextpad=0.4)
    ax.text(
        4.25, 5.25, "2 qubits per triangle\n(code shown in the box)", ha="right", va="top", fontsize=10, color=MUTED
    )
    ax.set_xlim(-0.2, 4.35)
    ax.set_ylim(1.75, 5.35)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "bus_stop_graph.png", dpi=200)
    fig.savefig(OUT / "bus_stop_graph.svg")
    plt.close(fig)


# --- device helpers (physical coordinates: entangling columns at 0, 0.3, 2, 2.3, ...) -----------------------------
def ent_x(c: int) -> float:
    return 2 * (c // 2) + 0.3 * (c % 2)


def _zone(ax, x0: float, y0: float, w: float, h: float, color: str, label: str) -> None:
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0.02,rounding_size=0.25", fc=color, ec="none"))
    ax.text(x0 + w + 0.35, y0 + h / 2, label, va="center", ha="left", fontsize=12, color=INK)


def device_zones() -> None:
    fig, ax = plt.subplots(figsize=(6, 6.5))
    _zone(ax, -0.6, -0.6, 9.9, 10.3, STORAGE_BG, "storage\nRx, Rz")
    _zone(ax, -0.6, 9.9, 9.9, 10.2, ENTANGLE_BG, "entangling\nCZ")
    atoms = {(1, 9), (2, 9), (3, 9), (4, 9), (6, 8), (7, 8), (2, 7), (5, 6), (8, 5)}
    for r in range(10):
        for c in range(10):
            filled = (c, r) in atoms
            ax.scatter(
                c, r, s=55, color=OURS if filled else "white", edgecolor=OURS if filled else BASE, lw=1, zorder=3
            )
    for r in range(10, 20):
        for c in range(10):
            ax.scatter(ent_x(c) + 0.35, r + 0.5, s=40, color="white", edgecolor=BASE, lw=1, zorder=3)
    for c in (2, 3):
        ax.scatter(ent_x(c) + 0.35, 12.5, s=55, color=OURS, edgecolor=OURS, zorder=4)
    ax.set_xlim(-1, 12.6)
    ax.set_ylim(-1, 20.6)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "device_zones.png", dpi=200)
    fig.savefig(OUT / "device_zones.svg")
    plt.close(fig)


def row_move() -> None:
    fig, ax = plt.subplots(figsize=(9, 4.6))
    _zone(ax, -0.5, 6.55, 10.0, 3.0, STORAGE_BG, "storage")
    _zone(ax, -0.5, 9.75, 10.0, 1.9, ENTANGLE_BG, "entangling")
    for r in (7, 8, 9):
        for c in range(10):
            ax.scatter(c, r, s=70, color="white", edgecolor=BASE, lw=1, zorder=2)
    for r in (10, 11):
        for c in range(10):
            ax.scatter(ent_x(c), r + 0.2, s=70, color="white", edgecolor=BASE, lw=1, zorder=2)
    for c in range(6):
        ax.scatter(c, 9, s=110, color=OURS, zorder=4)
        ax.scatter(ent_x(c), 10.2, s=110, color="white", edgecolor=OURS, lw=2, ls="--", zorder=4)
        arrow = FancyArrowPatch(
            (c, 9.12), (ent_x(c), 10.08), arrowstyle="-|>", mutation_scale=12, color=OURS, lw=1.4, zorder=3
        )
        ax.add_patch(arrow)
    for k in range(3):
        x = ent_x(2 * k)
        box = FancyBboxPatch(
            (x - 0.22, 10.0), 0.74, 0.4, boxstyle="round,pad=0.02,rounding_size=0.18", fc="none", ec=OURS, lw=1, ls=":"
        )
        ax.add_patch(box)
    caption = "One move carries all 6 atoms. Cost = the longest single trip (1.22), not the sum of 6 trips."
    ax.text(-0.5, 6.05, caption, ha="left", va="center", fontsize=12, color=INK)
    ax.set_xlim(-0.8, 12.4)
    ax.set_ylim(5.7, 11.8)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "row_move.png", dpi=200)
    fig.savefig(OUT / "row_move.svg")
    plt.close(fig)


if __name__ == "__main__":
    bus_stop_graph()
    device_zones()
    row_move()
    print("diagrams written to", OUT)
