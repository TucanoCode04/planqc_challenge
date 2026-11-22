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
from pushquantum.compiler.placement import home_layout, round_targets
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


def emit(
    original: QuantumCircuit, global_phase: float, schedule: Schedule, device: Device, config: Config
) -> tuple[QuantumCircuit, Layout]:
    home = home_layout(schedule, device, config.home)
    layout = dict(home)
    qc = original.copy_empty_like()
    qc.global_phase = global_phase
    stay_up = _in_next_pulse(schedule)

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
        move_to(round_targets(step, layout, device, occupied))
        for a, b in step.pairs:
            qc.cz(a, b)
        move_to({q: home[q] for q in in_round if not (config.reuse and q in stay_up[i])})

    for q, c in schedule.measurements:
        qc.measure(q, c)
    return qc, home
