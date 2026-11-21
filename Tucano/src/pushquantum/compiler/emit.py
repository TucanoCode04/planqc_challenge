"""Schedule -> final circuit with moves. v0: before each pulse the round's atoms go up, after it they go home."""

from __future__ import annotations

from qiskit import QuantumCircuit

import pushquantum.move  # noqa: F401 -- registers QuantumCircuit.move
from pushquantum.compiler.config import Config
from pushquantum.compiler.geometry import Device
from pushquantum.compiler.ir import CZRound, Layout, MoveSet, Schedule, StoragePhase
from pushquantum.compiler.placement import home_layout, round_targets
from pushquantum.compiler.routing import route_sequential


def _write_moves(qc: QuantumCircuit, sets: list[MoveSet]) -> None:
    for move_set in sets:
        for m in move_set:
            qc.move(list(m.qubits), rows=m.rows, cols=m.cols, new_rows=m.new_rows, new_cols=m.new_cols)


def emit(
    original: QuantumCircuit, global_phase: float, schedule: Schedule, device: Device, config: Config
) -> tuple[QuantumCircuit, Layout]:
    home = home_layout(schedule, device)
    layout = dict(home)
    qc = original.copy_empty_like()
    qc.global_phase = global_phase
    for step in schedule.steps:
        if isinstance(step, StoragePhase):
            for op in step.ops:
                getattr(qc, op.name)(op.angle, op.qubit)
            continue
        assert isinstance(step, CZRound)
        targets = round_targets(step, layout, device, set(layout.values()))
        _write_moves(qc, route_sequential(layout, targets, device))
        for a, b in step.pairs:
            qc.cz(a, b)
        back = {q: home[q] for q in step.qubits}
        _write_moves(qc, route_sequential(layout, back, device))
    for q, c in schedule.measurements:
        qc.measure(q, c)
    return qc, home
