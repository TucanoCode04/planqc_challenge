"""The pipeline: preprocess -> schedule -> placement + routing (inside emit) -> circuit + initial loading."""

from __future__ import annotations

from qiskit import QuantumCircuit

from pushquantum.architecture import Architecture, ArchitectureSpec
from pushquantum.compiler.config import DEFAULT, Config
from pushquantum.compiler.emit import emit
from pushquantum.compiler.geometry import Device
from pushquantum.compiler.preprocess import split_measurements, to_native
from pushquantum.compiler.schedule import build_schedule


def compile_circuit(
    circuit: QuantumCircuit, architecture_spec: ArchitectureSpec | None = None, config: Config = DEFAULT
) -> tuple[QuantumCircuit, Architecture]:
    device = Device.from_spec(architecture_spec)
    body, measurements = split_measurements(circuit)
    native = to_native(body)
    schedule = build_schedule(native, measurements, circuit.num_clbits)
    compiled, home = emit(circuit, native.global_phase, schedule, device, config)
    return compiled, Architecture(home, spec=architecture_spec)
