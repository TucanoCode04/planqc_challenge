"""Native circuit -> Schedule: alternating storage phases (1q gates) and CZ rounds (pulses).

Each qubit gets a queue of tokens: a `_Chunk` (consecutive 1q gates) or a CZ id. Rz commutes with CZ, so Rz-only
chunks slide forward into the next chunk that has an Rx. Then we loop: pop every chunk at a queue head (one storage
phase), take every CZ with no chunk before it on either qubit (one diagonal block: all commute), and split the block
into pulses with edge colouring.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import RXGate, RZGate
from qiskit.synthesis import OneQubitEulerDecomposer

from pushquantum.compiler.ir import CZRound, OneQOp, Schedule, StoragePhase


@dataclass
class _Chunk:
    ops: list[OneQOp] = field(default_factory=list)

    @property
    def diagonal(self) -> bool:
        return all(op.name == "rz" for op in self.ops)


Token = _Chunk | int


def _tokenize(native: QuantumCircuit) -> tuple[list[list[Token]], list[tuple[int, int]]]:
    queues: list[list[Token]] = [[] for _ in range(native.num_qubits)]
    czs: list[tuple[int, int]] = []
    for inst in native.data:
        name = inst.operation.name
        qubits = [native.find_bit(q).index for q in inst.qubits]
        if name in ("rx", "rz"):
            q = qubits[0]
            if not (queues[q] and isinstance(queues[q][-1], _Chunk)):
                queues[q].append(_Chunk())
            queues[q][-1].ops.append(OneQOp(name, inst.operation.params[0], q))
        elif name == "cz":
            a, b = sorted(qubits)
            for q in (a, b):
                queues[q].append(len(czs))
            czs.append((a, b))
        else:
            msg = f"Unexpected gate after preprocessing: {name}"
            raise ValueError(msg)
    return queues, czs


def _float_rz(queue: list[Token]) -> list[Token]:
    """Moves Rz-only chunks forward (past CZs) into the next chunk with an Rx, or to the end."""
    out: list[Token] = []
    carry: list[OneQOp] = []
    for token in queue:
        if isinstance(token, _Chunk):
            if token.diagonal:
                carry += token.ops
            else:
                out.append(_Chunk(carry + token.ops))
                carry = []
        else:
            out.append(token)
    if carry:
        out.append(_Chunk(carry))
    return out


_ZXZ = OneQubitEulerDecomposer("ZXZ")
_EPS = 1e-9


def _wrap(angle: float) -> float:
    """Angle in (-pi, pi]; Rz(a + 2pi) = -Rz(a) differs only by a global phase."""
    return math.remainder(angle, 2 * math.pi)


def _matrix(ops: list[OneQOp]) -> np.ndarray:
    u = np.eye(2, dtype=complex)
    for op in ops:
        gate = RXGate(op.angle) if op.name == "rx" else RZGate(op.angle)
        u = gate.to_matrix() @ u
    return u


def _fuse_rz(queue: list[Token], q: int) -> list[Token]:
    """v2c (depth): every chunk becomes Rz(lam)·Rx(theta), and its trailing Rz(phi) slides forward past the CZs
    (they commute) into the next chunk. Chunks that end up diagonal vanish into the carried Rz. The last chunk keeps
    Rz·Rx·Rz. Result: 2 gates per storage visit instead of 3. Falls back to `_float_rz` for symbolic angles."""
    ops_all = [op for t in queue if isinstance(t, _Chunk) for op in t.ops]
    if not all(isinstance(op.angle, (int, float)) for op in ops_all):
        return _float_rz(queue)
    out: list[Token] = []
    carry = 0.0
    last_chunk = max((i for i, t in enumerate(queue) if isinstance(t, _Chunk)), default=-1)
    for i, token in enumerate(queue):
        if not isinstance(token, _Chunk):
            out.append(token)
            continue
        u = _matrix([OneQOp("rz", carry, q), *token.ops])
        if abs(u[0, 1]) < _EPS:  # diagonal: a pure Rz, keep carrying it
            carry = float(np.angle(u[1, 1] / u[0, 0]))
            continue
        theta, phi, lam, _ = _ZXZ.angles_and_phase(u)
        ops = [OneQOp("rz", lam, q)] if abs(_wrap(lam)) > _EPS else []
        ops.append(OneQOp("rx", theta, q))
        if i == last_chunk:
            if abs(_wrap(phi)) > _EPS:
                ops.append(OneQOp("rz", phi, q))
            carry = 0.0
        else:
            carry = phi
        out.append(_Chunk(ops))
    if abs(_wrap(carry)) > _EPS:
        out.append(_Chunk([OneQOp("rz", carry, q)]))
    return out


def greedy_colouring(pairs: list[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    """Each colour = one pulse: no qubit twice. Busiest pairs first."""
    degree = Counter(q for pair in pairs for q in pair)
    rounds: list[list[tuple[int, int]]] = []
    used: list[set[int]] = []
    for a, b in sorted(pairs, key=lambda p: -(degree[p[0]] + degree[p[1]])):
        for i, qs in enumerate(used):
            if a not in qs and b not in qs:
                rounds[i].append((a, b))
                qs.update((a, b))
                break
        else:
            rounds.append([(a, b)])
            used.append({a, b})
    return rounds


def _cancel_duplicates(pairs: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Inside a block CZ·CZ on the same pair is the identity: keep a pair only if it appears an odd number of times."""
    counts = Counter(pairs)
    return [pair for pair, n in counts.items() if n % 2]


def build_schedule(
    native: QuantumCircuit, measurements: list[tuple[int, int]], num_clbits: int, fuse: bool = True
) -> Schedule:
    queues, czs = _tokenize(native)
    queues = [_fuse_rz(queue, q) if fuse else _float_rz(queue) for q, queue in enumerate(queues)]
    schedule = Schedule(native.num_qubits, num_clbits, measurements=measurements)
    block = 0
    while any(queues):
        phase = StoragePhase()
        for queue in queues:
            while queue and isinstance(queue[0], _Chunk):
                phase.ops += queue.pop(0).ops
        if phase.ops:
            schedule.steps.append(phase)

        prefixes: list[set[int]] = []
        for queue in queues:
            prefix: set[int] = set()
            for token in queue:
                if isinstance(token, _Chunk):
                    break
                prefix.add(token)
            prefixes.append(prefix)
        ready = {g for g in set().union(*prefixes) if all(g in prefixes[q] for q in czs[g])}
        if not ready:
            continue
        for queue in queues:
            queue[:] = [t for t in queue if isinstance(t, _Chunk) or t not in ready]
        pairs = _cancel_duplicates([czs[g] for g in sorted(ready)])
        for pulse in greedy_colouring(pairs):
            schedule.steps.append(CZRound(pulse, block))
        block += 1
    return schedule


def schedule_to_circuit(schedule: Schedule) -> QuantumCircuit:
    """Replays the schedule as a plain circuit (no moves), to test equivalence."""
    qc = QuantumCircuit(schedule.num_qubits, schedule.num_clbits)
    for step in schedule.steps:
        if isinstance(step, StoragePhase):
            for op in step.ops:
                getattr(qc, op.name)(op.angle, op.qubit)
        else:
            for a, b in step.pairs:
                qc.cz(a, b)
    return qc
