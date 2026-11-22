"""Entry point called by the validator. The real work is in `pushquantum.compiler`."""

from __future__ import annotations

from qiskit import QuantumCircuit

import pushquantum.move  # noqa: F401 -- registers QuantumCircuit.move
from pushquantum.architecture import Architecture, ArchitectureSpec
from pushquantum.compiler import compile_circuit as _compile


def compile_circuit(
    circuit: QuantumCircuit, architecture_spec: ArchitectureSpec
) -> tuple[QuantumCircuit, Architecture]:
    """Returns the compiled circuit and the `Architecture` (zones of `architecture_spec`, your initial loading)."""
    return _compile(circuit, architecture_spec)
