"""Where atoms live: the home layout (initial loading) and the pair site of every CZ in a round."""

from __future__ import annotations

from collections import Counter, deque

import networkx as nx
import numpy as np
from scipy.optimize import linear_sum_assignment

from pushquantum.compiler.config import HomeMethod
from pushquantum.compiler.geometry import Device
from pushquantum.compiler.ir import CZRound, Layout, Schedule
from pushquantum.move import Site


def interaction_counts(schedule: Schedule) -> Counter[tuple[int, int]]:
    return Counter(pair for step in schedule.steps if isinstance(step, CZRound) for pair in step.pairs)


def _bfs_order(schedule: Schedule) -> list[int]:
    """v0: BFS from the busiest qubit, neighbours by how many CZs they share."""
    n = schedule.num_qubits
    neighbours: dict[int, Counter[int]] = {q: Counter() for q in range(n)}
    for (a, b), c in interaction_counts(schedule).items():
        neighbours[a][b] += c
        neighbours[b][a] += c
    degree = {q: sum(neighbours[q].values()) for q in range(n)}
    order: list[int] = []
    seen: set[int] = set()
    for start in sorted(range(n), key=lambda q: -degree[q]):
        if start in seen:
            continue
        seen.add(start)
        todo = deque([start])
        while todo:
            q = todo.popleft()
            order.append(q)
            for nb, _ in neighbours[q].most_common():
                if nb not in seen:
                    seen.add(nb)
                    todo.append(nb)
    return order


def _cuthill_mckee_order(schedule: Schedule) -> list[int]:
    """v2a: BFS from a qubit on the edge of the interaction graph (lowest degree): connected qubits get nearby
    positions in the order. For a chain (HEA) this is exactly 0, 1, 2, ... from one end."""
    graph = nx.Graph()
    graph.add_nodes_from(range(schedule.num_qubits))
    graph.add_edges_from(interaction_counts(schedule))
    return list(nx.utils.cuthill_mckee_ordering(graph))


def _snake_sites(device: Device) -> list[Site]:
    """Storage sites row by row (nearest the entangling zone first), alternating direction: row 9 left->right,
    row 8 right->left, ... so consecutive qubits of the order are always neighbours, also across rows."""
    rows: dict[int, list[Site]] = {}
    for site in device.storage_sites:
        rows.setdefault(site[1], []).append(site)
    return [s for i, row in enumerate(rows.values()) for s in (row if i % 2 == 0 else row[::-1])]


def home_layout(schedule: Schedule, device: Device, method: HomeMethod = "bfs") -> Layout:
    """Where every atom starts (the initial loading) and returns to."""
    n = schedule.num_qubits
    if n > len(device.storage_sites):
        msg = f"{n} qubits do not fit in {len(device.storage_sites)} storage sites."
        raise ValueError(msg)
    if method == "snake":
        order, sites = _cuthill_mckee_order(schedule), _snake_sites(device)
    else:
        order, sites = _bfs_order(schedule), device.storage_sites
    return {q: sites[i] for i, q in enumerate(order)}


def round_targets(rnd: CZRound, layout: Layout, device: Device, occupied: set[Site]) -> Layout:
    """Hungarian matching: each pair -> a free pair site; cost = the longer of the two atoms' trips."""
    free = [p for p in device.pair_sites if not occupied.intersection(device.pair_slots(p))]
    if len(free) < len(rnd.pairs):
        msg = "Not enough free pair sites for this round."
        raise ValueError(msg)
    cost = np.empty((len(rnd.pairs), len(free)))
    flip = np.zeros_like(cost, dtype=bool)
    for i, (a, b) in enumerate(rnd.pairs):
        for j, p in enumerate(free):
            left, right = device.pair_slots(p)
            straight = max(device.dist(layout[a], left), device.dist(layout[b], right))
            crossed = max(device.dist(layout[b], left), device.dist(layout[a], right))
            cost[i, j] = min(straight, crossed)
            flip[i, j] = crossed < straight
    rows, cols = linear_sum_assignment(cost)
    targets: Layout = {}
    for i, j in zip(rows, cols, strict=True):
        a, b = rnd.pairs[i]
        left, right = device.pair_slots(free[j])
        if flip[i, j]:
            a, b = b, a
        targets[a], targets[b] = left, right
    return targets


def shift_targets(rnd: CZRound, layout: Layout, device: Device, occupied: set[Site]) -> Layout:
    """v2b: pairs sitting side by side in the same storage row all get the SAME column shift d (chosen so every left
    atom lands on a left slot), and storage rows map in order onto the nearest entangling rows. One shared row map +
    one shared column map = the router can lift them all with a single move. Other pairs: Hungarian on what's left."""
    if not all(device.in_storage(layout[q]) for q in rnd.qubits):
        return round_targets(rnd, layout, device, occupied)
    x0, x1 = device.entangling.x
    side_by_side: list[tuple[int, int]] = []
    for a, b in rnd.pairs:
        (xa, ya), (xb, yb) = layout[a], layout[b]
        if ya == yb and abs(xa - xb) == 1:
            side_by_side.append((a, b) if xa < xb else (b, a))  # (left atom, right atom)

    def fits(pair: tuple[int, int], d: int) -> bool:
        left_x = layout[pair[0]][0] + d
        return (left_x - x0) % 2 == 0 and x0 <= left_x and left_x + 1 <= x1

    shifts = sorted(range(x0 - x1, x1 - x0 + 1), key=abs)
    d = max(shifts, key=lambda s: sum(fits(p, s) for p in side_by_side))  # ties: smallest |d| (max keeps the first)
    placed = [p for p in side_by_side if fits(p, d)]

    storage_rows = sorted({layout[a][1] for a, _ in placed})
    nearest_rows: list[int] = []
    for row, _ in device.pair_sites:
        if row not in nearest_rows:
            nearest_rows.append(row)
    row_map = dict(zip(storage_rows, sorted(nearest_rows[: len(storage_rows)]), strict=True))

    targets: Layout = {}
    for a, b in placed:
        (x, y) = layout[a]
        left, right = (x + d, row_map[y]), (x + d + 1, row_map[y])
        if left in occupied or right in occupied:
            continue  # leave it to the Hungarian fallback
        targets[a], targets[b] = left, right
    rest = [p for p in rnd.pairs if p[0] not in targets]
    if rest:
        taken = occupied | set(targets.values())
        targets |= round_targets(CZRound(rest, rnd.block), layout, device, taken)
    return targets
