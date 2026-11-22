"""Layout -> layout as legal moves.

Scored distance = sum over `move` instructions of the longest atom trip in that move (checked with the validator), so
merging two moves always helps: cost a + b becomes max(a, b). `route_merged` packs atoms into as few moves as possible;
`route_sequential` (v0) moves one atom at a time and is the fallback when a merged move isn't legal.
"""

from __future__ import annotations

import itertools

from pushquantum.compiler.geometry import Device
from pushquantum.compiler.ir import Layout, MoveOp, MoveSet
from pushquantum.move import Site


def single_move(q: int, src: Site, dst: Site) -> MoveOp:
    return MoveOp((q,), (src[1],), (src[0],), (dst[1],), (dst[0],))


def _increasing(values: list[int]) -> bool:
    return all(a < b for a, b in itertools.pairwise(values))


def build_move(group: list[int], layout: Layout, targets: Layout, device: Device) -> MoveOp | None:
    """One AOD move carrying every atom of `group` to its target, or None if that isn't legal right now."""
    row_map: dict[int, int] = {}
    col_map: dict[int, int] = {}
    for q in group:
        (x, y), (nx, ny) = layout[q], targets[q]
        if row_map.setdefault(y, ny) != ny or col_map.setdefault(x, nx) != nx:
            return None  # one source line would have to split into two destination lines
    rows, cols = sorted(row_map), sorted(col_map)
    new_rows, new_cols = [row_map[r] for r in rows], [col_map[c] for c in cols]
    if not (_increasing(new_rows) and _increasing(new_cols)):
        return None  # lines would cross (AOD order must be preserved)
    if len({device.zone_of(layout[q]) for q in group}) > 1 or len({device.zone_of(targets[q]) for q in group}) > 1:
        return None  # pickup rows in one zone, drop rows in one zone
    members = set(group)
    for q, (x, y) in layout.items():
        if q not in members and rows[0] <= y <= rows[-1] and cols[0] <= x <= cols[-1]:
            return None  # ghost: an outside atom sits inside the pickup rectangle
    taken = {site for q, site in layout.items() if q not in members}
    if any(targets[q] in taken for q in group):
        return None  # would drop onto an atom that isn't part of this move
    return MoveOp(
        tuple(sorted(group)),
        tuple(range(rows[0], rows[-1] + 1)),
        tuple(range(cols[0], cols[-1] + 1)),
        tuple(new_rows),
        tuple(new_cols),
    )


def _apply(move: MoveOp, layout: Layout, targets: Layout) -> None:
    for q in move.qubits:
        layout[q] = targets[q]


def route_merged(layout: Layout, targets: Layout, device: Device) -> list[MoveSet]:
    """Greedy: first-fit every atom into an existing group if the group stays a legal move, then try merging groups.
    Groups are executed when legal (drop sites free); anything left falls back to one-atom moves. Updates `layout`."""
    pending = sorted(
        (q for q, dst in targets.items() if layout[q] != dst),
        key=lambda q: (device.zone_of(layout[q]) or "", layout[q][1], layout[q][0]),
    )
    groups: list[list[int]] = []
    for q in pending:
        for g in groups:
            if build_move([*g, q], layout, targets, device) is not None:
                g.append(q)
                break
        else:
            groups.append([q])

    merged = True
    while merged:
        merged = False
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                if build_move(groups[i] + groups[j], layout, targets, device) is not None:
                    groups[i] += groups.pop(j)
                    merged = True
                    break
            if merged:
                break

    sets: list[MoveSet] = []
    progress = True
    while groups and progress:
        progress = False
        for g in list(groups):
            move = build_move(g, layout, targets, device)
            if move is not None:
                sets.append([move])
                _apply(move, layout, targets)
                groups.remove(g)
                progress = True
    leftover = {q: targets[q] for g in groups for q in g}
    return sets + route_sequential(layout, leftover, device)


def route_sequential(layout: Layout, targets: Layout, device: Device) -> list[MoveSet]:
    """Moves every atom in `targets` there, one by one, never dropping onto an occupied site. Updates `layout`."""
    pending = {q: dst for q, dst in targets.items() if layout[q] != dst}
    occupied = set(layout.values())
    sets: list[MoveSet] = []

    def go(q: int, dst: Site) -> None:
        sets.append([single_move(q, layout[q], dst)])
        occupied.discard(layout[q])
        occupied.add(dst)
        layout[q] = dst

    while pending:
        movable = [q for q, dst in pending.items() if dst not in occupied]
        for q in movable:
            go(q, pending.pop(q))
        if not movable:
            # every target is taken by another pending atom (a cycle): park one atom on a free site first
            q = next(iter(pending))
            zone = device.zone_of(pending[q])
            sites = (
                device.storage_sites
                if zone == "storage"
                else [s for p in device.pair_sites for s in device.pair_slots(p)]
            )
            park = min((s for s in sites if s not in occupied), key=lambda s: device.dist(layout[q], s))
            go(q, park)
    return sets


def route(layout: Layout, targets: Layout, device: Device, merge: bool) -> list[MoveSet]:
    return route_merged(layout, targets, device) if merge else route_sequential(layout, targets, device)
