"""Where atoms live: the home layout (initial loading) and the pair site of every CZ in a round."""

from __future__ import annotations

from collections import Counter, deque

import numpy as np
from scipy.optimize import linear_sum_assignment

from pushquantum.compiler.geometry import Device
from pushquantum.compiler.ir import CZRound, Layout, Schedule
from pushquantum.move import Site


def interaction_counts(schedule: Schedule) -> Counter[tuple[int, int]]:
    return Counter(pair for step in schedule.steps if isinstance(step, CZRound) for pair in step.pairs)


def home_layout(schedule: Schedule, device: Device) -> Layout:
    """BFS order from the busiest qubit, so interacting qubits get neighbouring sites, nearest the entangling zone."""
    n = schedule.num_qubits
    if n > len(device.storage_sites):
        msg = f"{n} qubits do not fit in {len(device.storage_sites)} storage sites."
        raise ValueError(msg)
    counts = interaction_counts(schedule)
    neighbours: dict[int, Counter[int]] = {q: Counter() for q in range(n)}
    for (a, b), c in counts.items():
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
    return {q: device.storage_sites[i] for i, q in enumerate(order)}


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
