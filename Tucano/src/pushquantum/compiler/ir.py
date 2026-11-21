"""Data passed between the compiler stages: preprocess -> schedule -> placement -> routing -> emit."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from qiskit.circuit import ParameterExpression

from pushquantum.move import Site

Layout = dict[int, Site]
Angle = float | ParameterExpression


@dataclass(frozen=True)
class OneQOp:
    name: Literal["rx", "rz"]
    angle: Angle
    qubit: int


@dataclass
class StoragePhase:
    """Single-qubit gates, done while the involved atoms are in storage. Order matters per qubit."""

    ops: list[OneQOp] = field(default_factory=list)

    @property
    def qubits(self) -> set[int]:
        return {op.qubit for op in self.ops}


@dataclass
class CZRound:
    """One CZ pulse. Rounds with the same block id need no storage visit in between."""

    pairs: list[tuple[int, int]]
    block: int

    def __post_init__(self) -> None:
        # One pulse = one moment in time. An atom sits on one site, so it can be in only one pair per pulse:
        # a round like [(0, 1), (1, 2)] would need atom 1 in two places at once. If this fires, scheduling is wrong.
        flat = [q for pair in self.pairs for q in pair]
        if len(flat) != len(set(flat)):
            msg = f"A qubit appears twice in one round: {self.pairs}"
            raise ValueError(msg)

    @property
    def qubits(self) -> set[int]:
        return {q for pair in self.pairs for q in pair}


@dataclass
class Schedule:
    num_qubits: int
    num_clbits: int = 0
    steps: list[StoragePhase | CZRound] = field(default_factory=list)
    measurements: list[tuple[int, int]] = field(default_factory=list)  # (qubit, clbit), applied at the end


@dataclass(frozen=True)
class MoveOp:
    """One AOD move, exactly the arguments of `QuantumCircuit.move`."""

    qubits: tuple[int, ...]
    rows: tuple[int, ...]
    cols: tuple[int, ...]
    new_rows: tuple[int, ...]
    new_cols: tuple[int, ...]


MoveSet = list[MoveOp]  # consecutive moves meant to run in parallel
