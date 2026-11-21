"""Input circuit -> native circuit (rx, rz, cz only) + the final measurements, kept aside."""

from __future__ import annotations

from qiskit import QuantumCircuit, transpile

NATIVE = ["rx", "rz", "cz"]


def split_measurements(circuit: QuantumCircuit) -> tuple[QuantumCircuit, list[tuple[int, int]]]:
    """Removes barriers and final measurements; returns the body and the (qubit, clbit) measurements."""
    body = circuit.copy_empty_like()
    measurements: list[tuple[int, int]] = []
    measured: set[int] = set()
    for inst in circuit.data:
        name = inst.operation.name
        qubits = [circuit.find_bit(q).index for q in inst.qubits]
        if name == "barrier":
            continue
        if name == "measure":
            measurements.append((qubits[0], circuit.find_bit(inst.clbits[0]).index))
            measured.add(qubits[0])
            continue
        if measured.intersection(qubits):
            msg = "Only final measurements are supported."
            raise NotImplementedError(msg)
        body.append(inst.operation, inst.qubits, inst.clbits)
    return body, measurements


def to_native(body: QuantumCircuit) -> QuantumCircuit:
    """Native gates only. Level 1 also fuses runs of single-qubit gates and cancels adjacent inverse gates."""
    return transpile(body, basis_gates=NATIVE, optimization_level=1)
