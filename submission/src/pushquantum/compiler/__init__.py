"""The pipeline: preprocess -> schedule -> placement + routing (inside emit) -> circuit + initial loading.

With `config.pick_best`, emit runs once per variant (zz mode x fuse_rz x reuse x home x targets x dynamic_return =
96) and the cheapest is kept. `rank`: "distance_first" (distance, then depth) or "balanced" (relative distance +
relative depth, w.r.t. the first variant). Each variant takes milliseconds.
"""

from __future__ import annotations

from dataclasses import replace

from qiskit import QuantumCircuit

from pushquantum.architecture import Architecture, ArchitectureSpec
from pushquantum.compiler.config import DEFAULT, Config
from pushquantum.compiler.cost import move_distance
from pushquantum.compiler.emit import emit
from pushquantum.compiler.geometry import Device
from pushquantum.compiler.placement import grid_layout
from pushquantum.compiler.preprocess import group_zz, split_measurements, to_native
from pushquantum.compiler.schedule import build_schedule


def _depth(qc: QuantumCircuit) -> int:
    return qc.depth(filter_function=lambda inst: inst.operation.name not in ("move", "barrier"))


def compile_circuit(
    circuit: QuantumCircuit, architecture_spec: ArchitectureSpec | None = None, config: Config = DEFAULT
) -> tuple[QuantumCircuit, Architecture]:
    device = Device.from_spec(architecture_spec)
    body, measurements = split_measurements(circuit)
    # ZZ modes: (group_zz, fanout) = off / grouped / grouped + hub fan-out on extra atoms
    zz_modes = [(False, False), (True, False), (True, True)] if config.pick_best else [(config.group_zz, config.fanout)]
    fuse_options = (False, True) if config.pick_best else (config.fuse_rz,)
    max_qubits = len(device.storage_sites)
    natives = {m: to_native(group_zz(body, m[1], max_qubits) if m[0] else body) for m in zz_modes}

    # v4 (2D): pair graph -> 2D layout -> circuit coloured by grid position
    grid_modes = [(True, False), (True, True)] if config.pick_best and config.grid2d else []
    grid_natives, grid_layouts = {}, {}
    for m in grid_modes:
        pairs: set[tuple[int, int]] = set()
        first = to_native(group_zz(body, m[1], max_qubits, collect=pairs))
        pairs |= {
            tuple(sorted(first.find_bit(q).index for q in inst.qubits))
            for inst in first.data
            if inst.operation.name == "cz"
        }
        grid_layouts[m] = grid_layout(pairs, first.num_qubits, device, config.seed)
        grid_natives[m] = to_native(group_zz(body, m[1], max_qubits, layout=grid_layouts[m]))
    schedules = {
        (m, f): build_schedule(natives[m], measurements, circuit.num_clbits, f) for m in zz_modes for f in fuse_options
    }

    if config.pick_best:
        variants = [
            replace(config, group_zz=m[0], fanout=m[1], fuse_rz=f, reuse=r, home=h, targets=t, dynamic_return=d)
            for m in zz_modes
            for f in fuse_options
            for r in (False, True)
            for h in ("bfs", "snake")
            for t in ("hungarian", "shift")
            for d in (False, True)
        ]
        # v5: lookahead only changes the Hungarian targets, so only those variants get a lookahead copy
        variants += [replace(v, lookahead=config.lookahead_try) for v in variants if v.targets == "hungarian"]
    else:
        variants = [config]
    for m in grid_modes:
        for f in fuse_options:
            schedules[(("grid", *m), f)] = build_schedule(grid_natives[m], measurements, circuit.num_clbits, f)
        variants += [
            replace(
                config,
                group_zz=True,
                fanout=m[1],
                fuse_rz=f,
                reuse=r,
                home="grid",
                targets="grid",
                dynamic_return=False,
            )
            for f in fuse_options
            for r in (False, True)
        ]
    best = None
    ref: tuple[float, int] | None = None
    for v in variants:
        if v.home == "grid":
            m = (v.group_zz, v.fanout)
            native, schedule, given = grid_natives[m], schedules[(("grid", *m), v.fuse_rz)], grid_layouts[m]
        else:
            native, given = natives[(v.group_zz, v.fanout)], None
            schedule = schedules[((v.group_zz, v.fanout), v.fuse_rz)]
        compiled, home = emit(native, native.global_phase, schedule, device, v, given)
        dist, depth = move_distance(compiled, home, device), _depth(compiled)
        ref = ref or (max(dist, 1e-9), max(depth, 1))
        if config.rank == "balanced":
            key: tuple[float, ...] = (round(dist / ref[0] + config.depth_weight * depth / ref[1], 9), dist)
        else:
            key = (round(dist, 9), depth)
        if best is None or key < best[0]:
            best = (key, compiled, home)
    assert best is not None
    _, compiled, home = best
    return compiled, Architecture(home, spec=architecture_spec)
