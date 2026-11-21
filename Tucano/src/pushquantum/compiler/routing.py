"""Layout -> layout as legal moves. v0: one atom per move, one move per parallel set (always ghost-free)."""

from __future__ import annotations

from pushquantum.compiler.geometry import Device
from pushquantum.compiler.ir import Layout, MoveOp, MoveSet
from pushquantum.move import Site


def single_move(q: int, src: Site, dst: Site) -> MoveOp:
    return MoveOp((q,), (src[1],), (src[0],), (dst[1],), (dst[0],))


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
