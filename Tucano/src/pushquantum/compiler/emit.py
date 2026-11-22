"""Schedule -> final circuit with moves.

v0: before each pulse the round's atoms go up, after it they go home.
v1b (`reuse`): after a pulse an atom stays up only if it is in the NEXT pulse and has no single-qubit gate before it.
Validator rule (found by experiment): at every pulse, every atom in the entangling zone must take part in a CZ, so an
atom can never wait up there during a pulse it isn't part of.
"""

from __future__ import annotations

from qiskit import QuantumCircuit

import pushquantum.move  # noqa: F401 -- registers QuantumCircuit.move
from pushquantum.compiler.config import Config
from pushquantum.compiler.geometry import Device
from pushquantum.compiler.ir import CZRound, Layout, MoveSet, Schedule, StoragePhase
from pushquantum.compiler.placement import home_layout, round_targets, shift_targets
from pushquantum.compiler.routing import route


def _write_moves(qc: QuantumCircuit, sets: list[MoveSet]) -> None:
    for move_set in sets:
        for m in move_set:
            qc.move(list(m.qubits), rows=m.rows, cols=m.cols, new_rows=m.new_rows, new_cols=m.new_cols)


def _in_next_pulse(schedule: Schedule) -> list[set[int]]:
    """For each CZ round i: qubits of round i also in the next CZ round, with no storage gate in between."""
    steps = schedule.steps
    result: list[set[int]] = [set() for _ in steps]
    for i, step in enumerate(steps):
        if not isinstance(step, CZRound):
            continue
        busy: set[int] = set()
        for later in steps[i + 1 :]:
            if isinstance(later, CZRound):
                result[i] = (step.qubits & later.qubits) - busy
                break
            busy |= later.qubits
    return result


def _next_partner(schedule: Schedule) -> list[dict[int, int]]:
    """For CZ round i and each qubit q in it: q's partner in the next CZ round that uses q (if any)."""
    steps = schedule.steps
    result: list[dict[int, int]] = [{} for _ in steps]
    upcoming: dict[int, int] = {}
    for i in range(len(steps) - 1, -1, -1):
        step = steps[i]
        if not isinstance(step, CZRound):
            continue
        result[i] = {q: upcoming[q] for q in step.qubits if q in upcoming}
        for a, b in step.pairs:
            upcoming[a], upcoming[b] = b, a
    return result


def _return_sites(
    returning: list[int], partner: dict[int, int], home: Layout, layout: Layout, device: Device
) -> Layout:
    """v2b dynamic return: a returning atom parks on a free storage site right next to (same row) the home of its next
    CZ partner, so next time the two are side by side and go up in one move. Otherwise it goes back to its home."""
    taken = set(layout.values()) | set(home.values())
    anchored: set[int] = set()
    targets: Layout = {}
    for q in sorted(returning):
        p = partner.get(q)
        if p is not None and q not in anchored:
            px, py = home[p]
            spots = [(px + dx, py) for dx in (-1, 1) if device.in_storage((px + dx, py)) and (px + dx, py) not in taken]
            if spots:
                site = min(spots, key=lambda s: device.dist(layout[q], s))
                taken.add(site)
                home[q] = site
                anchored.add(p)
        targets[q] = home[q]
    return targets


def emit(
    original: QuantumCircuit, global_phase: float, schedule: Schedule, device: Device, config: Config
) -> tuple[QuantumCircuit, Layout]:
    initial = home_layout(schedule, device, config.home)
    home = dict(initial)  # where each atom returns to; changes with dynamic_return
    layout = dict(home)
    qc = original.copy_empty_like()
    qc.global_phase = global_phase
    stay_up = _in_next_pulse(schedule)
    partners = _next_partner(schedule)

    def move_to(targets: Layout) -> None:
        if targets:
            _write_moves(qc, route(layout, targets, device, config.merge_moves))

    for i, step in enumerate(schedule.steps):
        if isinstance(step, StoragePhase):
            move_to({q: home[q] for q in step.qubits if not device.in_storage(layout[q])})
            for op in step.ops:
                getattr(qc, op.name)(op.angle, op.qubit)
            continue

        assert isinstance(step, CZRound)
        in_round = step.qubits
        # safety net: nobody outside this pulse may be in the entangling zone when it fires
        move_to({q: home[q] for q, site in layout.items() if q not in in_round and device.in_entangling(site)})

        occupied = {site for q, site in layout.items() if q not in in_round}
        place = shift_targets if config.targets == "shift" else round_targets
        move_to(place(step, layout, device, occupied))
        for a, b in step.pairs:
            qc.cz(a, b)
        returning = [q for q in in_round if not (config.reuse and q in stay_up[i])]
        if config.dynamic_return:
            move_to(_return_sites(returning, partners[i], home, layout, device))
        else:
            move_to({q: home[q] for q in returning})

    for q, c in schedule.measurements:
        qc.measure(q, c)
    return qc, initial
