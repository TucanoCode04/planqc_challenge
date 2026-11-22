"""The pipeline: preprocess -> schedule -> placement + routing (inside emit) -> circuit + initial loading.

With `config.pick_best`, the emit step runs once per variant (reuse off/on x home bfs/snake) and the cheapest is kept,
ranked like the score: move distance first, then depth. Each variant takes milliseconds.
"""

from __future__ import annotations

from dataclasses import replace

from qiskit import QuantumCircuit

from pushquantum.architecture import Architecture, ArchitectureSpec
from pushquantum.compiler.config import DEFAULT, Config
from pushquantum.compiler.cost import move_distance
from pushquantum.compiler.emit import emit
from pushquantum.compiler.geometry import Device
from pushquantum.compiler.preprocess import split_measurements, to_native
from pushquantum.compiler.schedule import build_schedule


def _depth(qc: QuantumCircuit) -> int:
    return qc.depth(filter_function=lambda inst: inst.operation.name not in ("move", "barrier"))


def compile_circuit(
    circuit: QuantumCircuit, architecture_spec: ArchitectureSpec | None = None, config: Config = DEFAULT
) -> tuple[QuantumCircuit, Architecture]:
    device = Device.from_spec(architecture_spec)
    body, measurements = split_measurements(circuit)
    native = to_native(body)
    schedule = build_schedule(native, measurements, circuit.num_clbits)

    if config.pick_best:
        variants = [replace(config, reuse=r, home=h) for r in (False, True) for h in ("bfs", "snake")]
    else:
        variants = [config]
    best = None
    for variant in variants:
        compiled, home = emit(circuit, native.global_phase, schedule, device, variant)
        key = (round(move_distance(compiled, home, device), 9), _depth(compiled))
        if best is None or key < best[0]:
            best = (key, compiled, home)
    assert best is not None
    _, compiled, home = best
    return compiled, Architecture(home, spec=architecture_spec)
