"""Our own copy of the scored distance: sum over `move` instructions of the longest atom trip in that move.

Used to choose between compiler variants without calling the validator. Matches the validator's numbers (checked).
"""

from __future__ import annotations

from qiskit import QuantumCircuit

from pushquantum.compiler.geometry import Device
from pushquantum.compiler.ir import Layout


def move_distance(compiled: QuantumCircuit, loading: Layout, device: Device) -> float:
    pos = dict(loading)
    total = 0.0
    for inst in compiled.data:
        if inst.operation.name != "move":
            continue
        qubits = [compiled.find_bit(q).index for q in inst.qubits]
        _, _, new_rows, new_cols = inst.operation.params
        rows = sorted({pos[q][1] for q in qubits})
        cols = sorted({pos[q][0] for q in qubits})
        row_map, col_map = dict(zip(rows, new_rows, strict=True)), dict(zip(cols, new_cols, strict=True))
        longest = 0.0
        for q in qubits:
            x, y = pos[q]
            pos[q] = (col_map[x], row_map[y])
            longest = max(longest, device.dist((x, y), pos[q]))
        total += longest
    return total
