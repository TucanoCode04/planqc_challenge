"""Where atoms live: the home layout (initial loading) and the pair site of every CZ in a round."""

from __future__ import annotations

import math
import random
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


def round_targets(
    rnd: CZRound,
    layout: Layout,
    device: Device,
    occupied: set[Site],
    next_partner: dict[int, int] | None = None,
    lookahead: float = 0.0,
) -> Layout:
    """Hungarian matching: each pair -> a free pair site; cost = the longer of the two atoms' trips.
    v5 (`lookahead` > 0): plus lookahead x distance from each slot to the current site of that atom's next
    partner, so pairs are placed towards where they will be needed next (ZAC-style lookahead)."""
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
            if lookahead and next_partner:
                straight += lookahead * (
                    _ahead(a, left, layout, next_partner, device) + _ahead(b, right, layout, next_partner, device)
                )
                crossed += lookahead * (
                    _ahead(b, left, layout, next_partner, device) + _ahead(a, right, layout, next_partner, device)
                )
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


def _ahead(q: int, slot: Site, layout: Layout, next_partner: dict[int, int], device: Device) -> float:
    """Lookahead term: how far `slot` is from where q's next partner currently is (0 if q has none)."""
    p = next_partner.get(q)
    return device.dist(slot, layout[p]) if p is not None and p in layout else 0.0


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


def grid_layout(
    edges: set[tuple[int, int]], num_qubits: int, device: Device, seed: int = 0, steps: int = 20000
) -> Layout:
    """v4 (2D): simulated annealing over storage positions. Cost = for each pair, 0 if the two atoms are grid
    neighbours (left/right or up/down: both can reach a pair site with merged moves), else their Manhattan distance;
    plus a small penalty per row away from the entangling zone. Move: put a random qubit on a random site of the rows
    nearest the entangling zone (swapping with the qubit there, if any). Starts from the Cuthill-McKee snake."""
    rng = random.Random(seed)
    width = device.storage.x[1] - device.storage.x[0] + 1
    rows_needed = math.ceil(num_qubits / width) + 1
    snake = _snake_sites(device)
    sites = [s for s in snake if device._rows_to_entangling(s[1]) <= rows_needed]
    graph = nx.Graph()
    graph.add_nodes_from(range(num_qubits))
    graph.add_edges_from(edges)
    order = list(nx.utils.cuthill_mckee_ordering(graph))
    pos: dict[int, Site] = {q: snake[i] for i, q in enumerate(order)}
    nbrs = {q: list(graph.neighbors(q)) for q in range(num_qubits)}

    def pair_cost(a: Site, b: Site) -> float:
        d = abs(a[0] - b[0]) + abs(a[1] - b[1])
        return 0.0 if d == 1 else float(d)

    def node_cost(q: int, at: Site) -> float:
        return sum(pair_cost(at, pos[n]) for n in nbrs[q] if n in pos) + 0.05 * device._rows_to_entangling(at[1])

    occupant: dict[Site, int] = {site: q for q, site in pos.items()}
    for step in range(steps):
        temp = 2.0 * (1 - step / steps) + 0.01
        q = rng.randrange(num_qubits)
        target = rng.choice(sites)
        src = pos[q]
        if target == src:
            continue
        other = occupant.get(target)
        old = node_cost(q, src) + (node_cost(other, target) if other is not None else 0.0)
        pos[q] = target
        if other is not None:
            pos[other] = src
        new = node_cost(q, target) + (node_cost(other, src) if other is not None else 0.0)
        if new <= old or rng.random() < math.exp((old - new) / temp):
            occupant.pop(src, None)
            occupant[target] = q
            if other is not None:
                occupant[src] = other
        else:
            pos[q] = src
            if other is not None:
                pos[other] = target
    return pos


def grid_targets(rnd: CZRound, layout: Layout, device: Device, occupied: set[Site]) -> Layout:
    """v4 (2D): vertical neighbours are interleaved onto one entangling row (upper atom -> even column, lower atom ->
    odd column, columns spread x2; storage columns of the right half use a second row), so each storage row moves as
    one block. Horizontal neighbours: shift_targets. Anything else: Hungarian (inside shift_targets)."""
    if not all(device.in_storage(layout[q]) for q in rnd.qubits):
        return round_targets(rnd, layout, device, occupied)
    x0 = device.entangling.x[0]
    half = device.pairs_per_row
    rows_by_distance: list[int] = []
    for row, _ in device.pair_sites:
        if row not in rows_by_distance:
            rows_by_distance.append(row)
    vertical = [
        (a, b)
        for a, b in rnd.pairs
        if layout[a][0] == layout[b][0] and abs(layout[a][1] - layout[b][1]) == 1 and layout[a][0] - x0 < 2 * half
    ]
    row_for: dict[tuple[int, int], int] = {}
    free_rows = iter(rows_by_distance)
    for low in sorted({min(layout[a][1], layout[b][1]) for a, b in vertical}, reverse=True):
        for side in (0, 1):
            row = next(free_rows, None)
            if row is not None:
                row_for[(low, side)] = row
    targets: Layout = {}
    taken = set(occupied)
    for a, b in vertical:
        upper, lower = (a, b) if layout[a][1] > layout[b][1] else (b, a)
        x, low = layout[upper][0] - x0, layout[lower][1]
        side = 0 if x < half else 1
        if (low, side) not in row_for:
            continue
        col = x0 + 2 * (x - side * half)
        left, right = (col, row_for[(low, side)]), (col + 1, row_for[(low, side)])
        if left in taken or right in taken:
            continue
        targets[upper], targets[lower] = left, right
        taken |= {left, right}
    rest = [pair for pair in rnd.pairs if pair[0] not in targets]
    if rest:
        targets |= shift_targets(CZRound(rest, rnd.block), layout, device, taken)
    return targets
