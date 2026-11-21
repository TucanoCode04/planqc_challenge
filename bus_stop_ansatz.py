"""Hardware-native variational ansatze and VQE optimization for the PushQuantum 2026
planqc 12-bus-stop Minimum-Weight Vertex Cover problem.

Scope:
    This module provides problem definitions, exact cost evaluators, decoders,
    hardware-native Qiskit circuit builders ({rx, rz, cz}), fast exact-gradient
    statevector VQE training, OpenQASM 2.0 export, and the submission entry point
    ``get_bus_stop_circuit``. Neutral-atom routing/compilation is intentionally
    kept separate (implemented by ``compile_circuit``).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit import qasm2
from qiskit.quantum_info import Statevector
from scipy.optimize import minimize

# ---------------------------------------------------------------------------
# 1. Ground-Truth Problem Definition (12-stop, 18-edge truncated tetrahedron)
# ---------------------------------------------------------------------------

NUM_STOPS: int = 12
PENALTY: float = 5.0

STOP_COSTS: dict[int, int] = {
    1: 2,
    2: 1,
    3: 3,
    4: 4,
    5: 2,
    6: 1,
    7: 3,
    8: 2,
    9: 4,
    10: 1,
    11: 3,
    12: 2,
}

# Four triangles (12 intra-triangle edges)
TRIANGLES: tuple[tuple[int, int, int], ...] = (
    (1, 2, 3),    # T1 -> 8q qubits (q0, q1)
    (4, 5, 6),    # T2 -> 8q qubits (q2, q3)
    (7, 8, 9),    # T3 -> 8q qubits (q4, q5)
    (10, 11, 12), # T4 -> 8q qubits (q6, q7)
)

INTRA_TRIANGLE_EDGES: tuple[tuple[int, int], ...] = tuple(
    (a, b)
    for t in TRIANGLES
    for a, b in ((t[0], t[1]), (t[1], t[2]), (t[0], t[2]))
)

# Six inter-triangle bridges
BRIDGES: tuple[tuple[int, int], ...] = (
    (1, 4),
    (2, 7),
    (3, 10),
    (5, 8),
    (6, 11),
    (9, 12),
)

ALL_EDGES: tuple[tuple[int, int], ...] = INTRA_TRIANGLE_EDGES + BRIDGES

OPTIMAL_KEPT_STOPS: frozenset[int] = frozenset({1, 2, 5, 6, 7, 8, 10, 12})
OPTIMAL_DROPPED_STOPS: frozenset[int] = frozenset({3, 4, 9, 11})
OPTIMAL_COST: float = 14.0
OPTIMAL_8Q_BITS: str = "00100001"  # q0..q7 order: T1=00(drop 3), T2=10(drop 4), T3=00(drop 9), T4=01(drop 11)
OPTIMAL_12Q_BITS: str = "".join(
    "1" if s in OPTIMAL_KEPT_STOPS else "0" for s in range(1, NUM_STOPS + 1)
)

# 2-qubit-per-triangle code table: 2-bit string (q_{2t}, q_{2t+1}) -> (x_a, x_b, x_c)
TRIANGLE_CODE_MAP: dict[str, tuple[int, int, int]] = {
    "11": (1, 1, 1),  # keep all 3 stops
    "10": (0, 1, 1),  # drop 1st stop in triangle
    "01": (1, 0, 1),  # drop 2nd stop in triangle
    "00": (1, 1, 0),  # drop 3rd stop in triangle
}

# Bridge violation conditions in 8-qubit triangle code:
# (bridge_stops, triangle_idx_a, code_a, triangle_idx_b, code_b)
BRIDGE_8Q_PENALTIES: tuple[tuple[tuple[int, int], int, str, int, str], ...] = (
    ((1, 4), 0, "10", 1, "10"),
    ((2, 7), 0, "01", 2, "10"),
    ((3, 10), 0, "00", 3, "10"),
    ((5, 8), 1, "01", 2, "01"),
    ((6, 11), 1, "00", 3, "01"),
    ((9, 12), 2, "00", 3, "00"),
)

# Entangling rounds for Candidate A (8q single-row) and Candidate B (12q 2x6 real-edges)
ROUNDS_8Q: tuple[tuple[tuple[int, int], ...], ...] = (
    ((0, 1), (2, 3), (4, 5), (6, 7)),  # Round 1: intra-triangle pairs
    ((1, 2), (3, 4), (5, 6)),          # Round 2: inter-triangle pairs (1-col AOD shift)
)

ROUNDS_12Q: tuple[tuple[tuple[int, int], ...], ...] = (
    ((1, 2), (0, 3), (4, 5), (6, 7), (8, 11), (9, 10)),  # Round 1: 6 real edges
    ((2, 0), (3, 4), (7, 8), (11, 9)),                   # Round 2: 4 real edges (1-col shift)
    ((1, 6), (5, 10)),                                   # Round 3: 2 vertical bridges
)

# Recommended storage-zone initial placements (qubit -> (x, y)) for the compiler teammate
RECOMMENDED_LAYOUT_8Q: dict[int, tuple[int, int]] = {k: (k, 9) for k in range(8)}

# Row 9 stops: [2, 3, 1, 4, 5, 6] -> qubits [1, 2, 0, 3, 4, 5]
# Row 8 stops: [7, 8, 9, 12, 10, 11] -> qubits [6, 7, 8, 11, 9, 10]
RECOMMENDED_LAYOUT_12Q: dict[int, tuple[int, int]] = {
    **{q: (col, 9) for col, q in enumerate((1, 2, 0, 3, 4, 5))},
    **{q: (col, 8) for col, q in enumerate((6, 7, 8, 11, 9, 10))},
}


# ---------------------------------------------------------------------------
# 2. Exact Cost Evaluators, Decoders & 8-Qubit Pauli-Z Hamiltonian
# ---------------------------------------------------------------------------

def evaluate_stops_cost(x: Sequence[int]) -> float:
    """Evaluate exact MWVC objective F(x) for binary keep-vector x[0..11] (stops 1..12)."""
    if len(x) != NUM_STOPS:
        raise ValueError(f"Expected length-{NUM_STOPS} binary sequence, got {len(x)}.")
    maint = sum(STOP_COSTS[i + 1] * int(x[i]) for i in range(NUM_STOPS))
    penalties = PENALTY * sum(
        (1 - int(x[u - 1])) * (1 - int(x[v - 1])) for u, v in ALL_EDGES
    )
    return float(maint + penalties)


def decode_8q_bitstring(bits_q0_to_q7: str) -> dict[str, Any]:
    """Decode an 8-qubit bitstring in q0..q7 order into 12-stop decisions and exact cost."""
    if len(bits_q0_to_q7) != 8 or any(ch not in "01" for ch in bits_q0_to_q7):
        raise ValueError(f"Expected 8-char binary string in q0..q7 order, got {bits_q0_to_q7!r}.")

    tri_codes = [bits_q0_to_q7[2 * t : 2 * t + 2] for t in range(4)]
    x = [0] * NUM_STOPS
    triangle_decisions: dict[str, dict[str, Any]] = {}
    for t_idx, (stops, code) in enumerate(zip(TRIANGLES, tri_codes, strict=True)):
        vals = TRIANGLE_CODE_MAP[code]
        for s, val in zip(stops, vals, strict=True):
            x[s - 1] = val
        dropped = [s for s, val in zip(stops, vals, strict=True) if val == 0]
        triangle_decisions[f"T{t_idx + 1}"] = {
            "code": code,
            "stops": stops,
            "values": vals,
            "dropped": dropped[0] if dropped else None,
        }

    violated_bridges = [
        bridge
        for bridge, t_a, code_a, t_b, code_b in BRIDGE_8Q_PENALTIES
        if tri_codes[t_a] == code_a and tri_codes[t_b] == code_b
    ]
    cost = evaluate_stops_cost(x)
    kept_stops = [i + 1 for i, val in enumerate(x) if val == 1]
    dropped_stops = [i + 1 for i, val in enumerate(x) if val == 0]
    return {
        "bits_q0_to_q7": bits_q0_to_q7,
        "bits_q7_to_q0": bits_q0_to_q7[::-1],
        "x": tuple(x),
        "kept_stops": kept_stops,
        "dropped_stops": dropped_stops,
        "triangles": triangle_decisions,
        "violated_bridges": violated_bridges,
        "is_feasible": len(violated_bridges) == 0,
        "cost": cost,
    }


def decode_12q_bitstring(bits_q0_to_q11: str) -> dict[str, Any]:
    """Decode a 12-qubit bitstring in q0..q11 (stops 1..12) order into stop decisions."""
    if len(bits_q0_to_q11) != 12 or any(ch not in "01" for ch in bits_q0_to_q11):
        raise ValueError(f"Expected 12-char binary string in q0..q11 order, got {bits_q0_to_q11!r}.")
    x = tuple(int(ch) for ch in bits_q0_to_q11)
    violated_edges = [(u, v) for u, v in ALL_EDGES if x[u - 1] == 0 and x[v - 1] == 0]
    return {
        "bits_q0_to_q11": bits_q0_to_q11,
        "bits_q11_to_q0": bits_q0_to_q11[::-1],
        "x": x,
        "kept_stops": [i + 1 for i, val in enumerate(x) if val == 1],
        "dropped_stops": [i + 1 for i, val in enumerate(x) if val == 0],
        "violated_edges": violated_edges,
        "is_feasible": len(violated_edges) == 0,
        "cost": evaluate_stops_cost(x),
    }


def _build_costs_8q() -> np.ndarray:
    """Precompute exact 256-entry cost vector indexed by Qiskit little-endian state index."""
    costs = np.zeros(1 << 8, dtype=np.float64)
    for idx in range(1 << 8):
        bits = f"{idx:08b}"[::-1]  # q0..q7 order
        costs[idx] = decode_8q_bitstring(bits)["cost"]
    return costs


def _build_costs_12q() -> np.ndarray:
    """Precompute exact 4096-entry cost vector indexed by Qiskit little-endian state index."""
    costs = np.zeros(1 << 12, dtype=np.float64)
    for idx in range(1 << 12):
        bits = f"{idx:012b}"[::-1]  # q0..q11 = stops 1..12
        costs[idx] = evaluate_stops_cost([int(ch) for ch in bits])
    return costs


COSTS_8Q: np.ndarray = _build_costs_8q()
COSTS_12Q: np.ndarray = _build_costs_12q()

OPTIMAL_IDX_8Q: int = int(np.argmin(COSTS_8Q))
OPTIMAL_IDX_12Q: int = int(np.argmin(COSTS_12Q))


def build_8q_pauli_hamiltonian() -> dict[tuple[int, ...], float]:
    """Derive exact Pauli-Z expansion of H_8q as {sorted_Z_qubits_tuple: coeff}.

    Uses projectors P_0(k) = (I + Z_k)/2 and P_1(k) = (I - Z_k)/2 so that
    H_8q = sum_t H_maint^(t) + 5 * sum_bridges Pi_{c_a}^(t_a) Pi_{c_b}^(t_b).
    """
    terms: dict[tuple[int, ...], float] = {}

    def add_term(qubits: Sequence[int], coeff: float) -> None:
        key = tuple(sorted(qubits))
        terms[key] = terms.get(key, 0.0) + coeff

    def projector_2q(q_a: int, q_b: int, code: str) -> list[tuple[tuple[int, ...], float]]:
        s_a = 1.0 if code[0] == "0" else -1.0
        s_b = 1.0 if code[1] == "0" else -1.0
        return [
            ((), 0.25),
            ((q_a,), 0.25 * s_a),
            ((q_b,), 0.25 * s_b),
            ((q_a, q_b), 0.25 * s_a * s_b),
        ]

    # Triangle maintenance costs
    for t_idx, (a, b, c) in enumerate(TRIANGLES):
        u, v = 2 * t_idx, 2 * t_idx + 1
        ca, cb, cc = float(STOP_COSTS[a]), float(STOP_COSTS[b]), float(STOP_COSTS[c])
        add_term((), 0.75 * (ca + cb + cc))
        add_term((u,), 0.25 * (ca - cb - cc))
        add_term((v,), 0.25 * (-ca + cb - cc))
        add_term((u, v), 0.25 * (ca + cb - cc))

    # Bridge penalty 4-local projectors
    for _, t_a, code_a, t_b, code_b in BRIDGE_8Q_PENALTIES:
        pa = projector_2q(2 * t_a, 2 * t_a + 1, code_a)
        pb = projector_2q(2 * t_b, 2 * t_b + 1, code_b)
        for qa, wa in pa:
            for qb, wb in pb:
                add_term(qa + qb, PENALTY * wa * wb)

    return {k: v for k, v in terms.items() if abs(v) > 1e-12}


def evaluate_pauli_z_diagonal(terms: dict[tuple[int, ...], float], num_qubits: int) -> np.ndarray:
    """Evaluate diagonal of a Pauli-Z Hamiltonian in Qiskit little-endian basis."""
    indices = np.arange(1 << num_qubits, dtype=np.int64)
    diag = np.zeros(1 << num_qubits, dtype=np.float64)
    for qubits, coeff in terms.items():
        if not qubits:
            diag += coeff
        else:
            parity = np.zeros(1 << num_qubits, dtype=np.int64)
            for q in qubits:
                parity ^= (indices >> q) & 1
            diag += coeff * np.where(parity == 0, 1.0, -1.0)
    return diag


# ---------------------------------------------------------------------------
# 3. Parametric Qiskit Circuit Builders (Native {rx, rz, cz} + HEA Baseline)
# ---------------------------------------------------------------------------

def num_params_8q(layers: int = 1) -> int:
    if layers < 1:
        raise ValueError("layers must be >= 1")
    return 8 * (2 * layers + 1)


def num_params_12q(layers: int = 1) -> int:
    if layers < 1:
        raise ValueError("layers must be >= 1")
    return 12 * (2 * layers + 1)


def clean_hardware_params(
    params: Sequence[float],
    zero_tol: float = 1e-4,
    pi_tol: float = 1e-4,
    decimals: int | None = 4,
) -> np.ndarray:
    """Snap near-zero / near-±pi angles and quantize remaining pulses to realistic laser precision.

    Parameters
    ----------
    params : Sequence[float]
        Raw continuous VQE angles (radians).
    zero_tol : float, default=1e-4
        Angles with ``|theta| < zero_tol`` (mod 2*pi) snap to exact ``0.0``.
    pi_tol : float, default=1e-4
        Angles within ``pi_tol`` of ``+pi`` or ``-pi`` snap to exact ``±pi``.
    decimals : int | None, default=4
        Decimal places for remaining non-special angles (default ``4`` -> 0.1 mrad).
        Pass ``None`` to skip rounding.
    """
    raw = np.asarray(params, dtype=np.float64)
    # Wrap into (-pi, pi]
    wrapped = ((raw + math.pi) % (2.0 * math.pi)) - math.pi
    cleaned = np.empty_like(wrapped)
    for i, val in enumerate(wrapped):
        if abs(val) < zero_tol:
            cleaned[i] = 0.0
        elif abs(val - math.pi) < pi_tol:
            cleaned[i] = math.pi
        elif abs(val + math.pi) < pi_tol:
            cleaned[i] = -math.pi
        else:
            r = round(float(val), decimals) if decimals is not None else float(val)
            cleaned[i] = 0.0 if abs(r) < zero_tol else r
    return cleaned


def build_8q_ansatz(
    params: Sequence[float],
    layers: int = 1,
    measure: bool = False,
    prune_zeros: bool = True,
    zero_tol: float = 1e-7,
) -> QuantumCircuit:
    """Build Candidate A: 8-qubit triangle-compressed hardware-native {rx, rz, cz} ansatz.

    Designed for a single row (y=9 <-> y=10, cols 0..7 -> 1..8):
      - Odd layers execute Round 1 (intra-triangle) then Round 2 (inter-triangle).
      - Even layers execute Round 2 then Round 1 to avoid resetting atom positions.
      - Final layer uses rx only (omits unobservable terminal rz before Z measurement).
      - When ``prune_zeros=True`` (default), omits any rx/rz gate with ``|angle| <= zero_tol``.
    """
    expected = num_params_8q(layers)
    p = np.asarray(params, dtype=np.float64)
    if p.shape != (expected,):
        raise ValueError(f"8q ansatz (layers={layers}) expects {expected} params, got {p.shape}.")

    qc = QuantumCircuit(8, 8) if measure else QuantumCircuit(8)
    idx = 0
    for layer in range(layers):
        for k in range(8):
            ang = float(p[idx + k])
            if not prune_zeros or abs(ang) > zero_tol:
                qc.rx(ang, k)
        idx += 8
        for k in range(8):
            ang = float(p[idx + k])
            if not prune_zeros or abs(ang) > zero_tol:
                qc.rz(ang, k)
        idx += 8

        round_order = ROUNDS_8Q if (layer % 2 == 0) else tuple(reversed(ROUNDS_8Q))
        for entangling_round in round_order:
            for q_a, q_b in entangling_round:
                qc.cz(q_a, q_b)

    for k in range(8):
        ang = float(p[idx + k])
        if not prune_zeros or abs(ang) > zero_tol:
            qc.rx(ang, k)

    if measure:
        qc.measure(range(8), range(8))
    return qc


def build_12q_ansatz(
    params: Sequence[float],
    layers: int = 1,
    measure: bool = False,
    prune_zeros: bool = True,
    zero_tol: float = 1e-7,
) -> QuantumCircuit:
    """Build Candidate B: 12-qubit 2x6 real-edges hardware-native {rx, rz, cz} ansatz.

    Uses 12 CZ gates per layer acting exclusively on real problem edges in 3 parallel rounds.
    When ``prune_zeros=True`` (default), omits any rx/rz gate with ``|angle| <= zero_tol``.
    """
    expected = num_params_12q(layers)
    p = np.asarray(params, dtype=np.float64)
    if p.shape != (expected,):
        raise ValueError(f"12q ansatz (layers={layers}) expects {expected} params, got {p.shape}.")

    qc = QuantumCircuit(12, 12) if measure else QuantumCircuit(12)
    idx = 0
    for _ in range(layers):
        for k in range(12):
            ang = float(p[idx + k])
            if not prune_zeros or abs(ang) > zero_tol:
                qc.rx(ang, k)
        idx += 12
        for k in range(12):
            ang = float(p[idx + k])
            if not prune_zeros or abs(ang) > zero_tol:
                qc.rz(ang, k)
        idx += 12
        for entangling_round in ROUNDS_12Q:
            for q_a, q_b in entangling_round:
                qc.cz(q_a, q_b)

    for k in range(12):
        ang = float(p[idx + k])
        if not prune_zeros or abs(ang) > zero_tol:
            qc.rx(ang, k)

    if measure:
        qc.measure(range(12), range(12))
    return qc


def build_hea_linear_12q(
    params: Sequence[float],
    layers: int = 1,
    measure: bool = False,
) -> QuantumCircuit:
    """Build baseline 12-qubit linear-chain HEA matching `bus_stop_12q_hea_linear_p1.qasm`."""
    expected = num_params_12q(layers)
    p = np.asarray(params, dtype=np.float64)
    if p.shape != (expected,):
        raise ValueError(f"hea_linear (layers={layers}) expects {expected} params, got {p.shape}.")

    qc = QuantumCircuit(12, 12) if measure else QuantumCircuit(12)
    idx = 0
    for _ in range(layers):
        for k in range(12):
            qc.ry(float(p[idx + k]), k)
        idx += 12
        for k in range(12):
            qc.rz(float(p[idx + k]), k)
        idx += 12
        for k in range(0, 12, 2):
            qc.cx(k, k + 1)
        for k in range(1, 11, 2):
            qc.cx(k, k + 1)

    for k in range(12):
        qc.ry(float(p[idx + k]), k)

    if measure:
        qc.measure(range(12), range(12))
    return qc


# ---------------------------------------------------------------------------
# 4. Fast Vectorized Statevector + Exact Adjoint-Gradient VQE Engine
# ---------------------------------------------------------------------------

def _apply_rx_inplace(psi: np.ndarray, qubit: int, theta: float, n: int) -> None:
    axis = n - 1 - qubit
    c = math.cos(0.5 * theta)
    s = -1j * math.sin(0.5 * theta)
    view = np.moveaxis(psi, axis, 0)
    v0 = view[0].copy()
    v1 = view[1].copy()
    view[0] = c * v0 + s * v1
    view[1] = s * v0 + c * v1


def _apply_ry_inplace(psi: np.ndarray, qubit: int, theta: float, n: int) -> None:
    axis = n - 1 - qubit
    c = math.cos(0.5 * theta)
    s = math.sin(0.5 * theta)
    view = np.moveaxis(psi, axis, 0)
    v0 = view[0].copy()
    v1 = view[1].copy()
    view[0] = c * v0 - s * v1
    view[1] = s * v0 + c * v1


def _apply_rz_inplace(psi: np.ndarray, qubit: int, theta: float, n: int) -> None:
    axis = n - 1 - qubit
    e0 = complex(math.cos(0.5 * theta), -math.sin(0.5 * theta))
    e1 = complex(math.cos(0.5 * theta), math.sin(0.5 * theta))
    view = np.moveaxis(psi, axis, 0)
    view[0] *= e0
    view[1] *= e1


def _apply_cz_inplace(psi: np.ndarray, q_a: int, q_b: int, n: int) -> None:
    sl = [slice(None)] * n
    sl[n - 1 - q_a] = 1
    sl[n - 1 - q_b] = 1
    psi[tuple(sl)] *= -1.0


def _apply_cx_inplace(psi: np.ndarray, ctrl: int, tgt: int, n: int) -> None:
    ax_c = n - 1 - ctrl
    ax_t = n - 1 - tgt
    view = np.moveaxis(psi, (ax_c, ax_t), (0, 1))
    tmp = view[1, 0].copy()
    view[1, 0] = view[1, 1]
    view[1, 1] = tmp


def _pauli_inner_imag(lam: np.ndarray, psi: np.ndarray, qubit: int, pauli: str, n: int) -> float:
    """Compute Im(<lam | P_qubit | psi>) for P in {'X', 'Y', 'Z'}."""
    axis = n - 1 - qubit
    lv = np.moveaxis(lam, axis, 0)
    pv = np.moveaxis(psi, axis, 0)
    if pauli == "X":
        val = np.vdot(lv[0], pv[1]) + np.vdot(lv[1], pv[0])
    elif pauli == "Y":
        val = -1j * np.vdot(lv[0], pv[1]) + 1j * np.vdot(lv[1], pv[0])
    elif pauli == "Z":
        val = np.vdot(lv[0], pv[0]) - np.vdot(lv[1], pv[1])
    else:
        raise ValueError(f"Unknown Pauli {pauli!r}")
    return float(val.imag)


def _get_gate_program(ansatz_type: str, layers: int) -> tuple[int, np.ndarray, list[tuple]]:
    """Return (num_qubits, cost_vector, gate_ops) for fast simulation and adjoint gradients."""
    ops: list[tuple] = []
    if ansatz_type == "8q":
        n = 8
        costs = COSTS_8Q
        idx = 0
        for layer in range(layers):
            for k in range(8):
                ops.append(("rx", k, idx + k))
            idx += 8
            for k in range(8):
                ops.append(("rz", k, idx + k))
            idx += 8
            round_order = ROUNDS_8Q if (layer % 2 == 0) else tuple(reversed(ROUNDS_8Q))
            for ent_round in round_order:
                for q_a, q_b in ent_round:
                    ops.append(("cz", q_a, q_b))
        for k in range(8):
            ops.append(("rx", k, idx + k))
        return n, costs, ops

    if ansatz_type == "12q":
        n = 12
        costs = COSTS_12Q
        idx = 0
        for _ in range(layers):
            for k in range(12):
                ops.append(("rx", k, idx + k))
            idx += 12
            for k in range(12):
                ops.append(("rz", k, idx + k))
            idx += 12
            for ent_round in ROUNDS_12Q:
                for q_a, q_b in ent_round:
                    ops.append(("cz", q_a, q_b))
        for k in range(12):
            ops.append(("rx", k, idx + k))
        return n, costs, ops

    if ansatz_type == "hea_linear":
        n = 12
        costs = COSTS_12Q
        idx = 0
        for _ in range(layers):
            for k in range(12):
                ops.append(("ry", k, idx + k))
            idx += 12
            for k in range(12):
                ops.append(("rz", k, idx + k))
            idx += 12
            for k in range(0, 12, 2):
                ops.append(("cx", k, k + 1))
            for k in range(1, 11, 2):
                ops.append(("cx", k, k + 1))
        for k in range(12):
            ops.append(("ry", k, idx + k))
        return n, costs, ops

    raise ValueError(f"Unsupported ansatz_type {ansatz_type!r}. Choose '8q', '12q', or 'hea_linear'.")


def simulate_statevector_fast(
    ansatz_type: str, params: Sequence[float], layers: int = 1
) -> np.ndarray:
    """Compute exact 1D Qiskit-ordered statevector using vectorized NumPy."""
    n, _, ops = _get_gate_program(ansatz_type, layers)
    p = np.asarray(params, dtype=np.float64)
    psi = np.zeros((2,) * n, dtype=np.complex128)
    psi[(0,) * n] = 1.0
    for op in ops:
        kind = op[0]
        if kind == "rx":
            _apply_rx_inplace(psi, op[1], float(p[op[2]]), n)
        elif kind == "ry":
            _apply_ry_inplace(psi, op[1], float(p[op[2]]), n)
        elif kind == "rz":
            _apply_rz_inplace(psi, op[1], float(p[op[2]]), n)
        elif kind == "cz":
            _apply_cz_inplace(psi, op[1], op[2], n)
        elif kind == "cx":
            _apply_cx_inplace(psi, op[1], op[2], n)
    return psi.reshape(-1)


def _energy_and_grad(
    p: np.ndarray,
    n: int,
    costs_tensor: np.ndarray,
    ops: list[tuple],
) -> tuple[float, np.ndarray]:
    """Compute exact <psi|C|psi> and exact analytic gradient via adjoint differentiation."""
    psi = np.zeros((2,) * n, dtype=np.complex128)
    psi[(0,) * n] = 1.0
    for op in ops:
        kind = op[0]
        if kind == "rx":
            _apply_rx_inplace(psi, op[1], float(p[op[2]]), n)
        elif kind == "ry":
            _apply_ry_inplace(psi, op[1], float(p[op[2]]), n)
        elif kind == "rz":
            _apply_rz_inplace(psi, op[1], float(p[op[2]]), n)
        elif kind == "cz":
            _apply_cz_inplace(psi, op[1], op[2], n)
        elif kind == "cx":
            _apply_cx_inplace(psi, op[1], op[2], n)

    lam = costs_tensor * psi
    energy = float(np.vdot(psi, lam).real)

    grad = np.zeros_like(p, dtype=np.float64)
    for op in reversed(ops):
        kind = op[0]
        if kind == "rx":
            q, p_idx = op[1], op[2]
            grad[p_idx] += _pauli_inner_imag(lam, psi, q, "X", n)
            theta = -float(p[p_idx])
            _apply_rx_inplace(psi, q, theta, n)
            _apply_rx_inplace(lam, q, theta, n)
        elif kind == "ry":
            q, p_idx = op[1], op[2]
            grad[p_idx] += _pauli_inner_imag(lam, psi, q, "Y", n)
            theta = -float(p[p_idx])
            _apply_ry_inplace(psi, q, theta, n)
            _apply_ry_inplace(lam, q, theta, n)
        elif kind == "rz":
            q, p_idx = op[1], op[2]
            grad[p_idx] += _pauli_inner_imag(lam, psi, q, "Z", n)
            theta = -float(p[p_idx])
            _apply_rz_inplace(psi, q, theta, n)
            _apply_rz_inplace(lam, q, theta, n)
        elif kind == "cz":
            _apply_cz_inplace(psi, op[1], op[2], n)
            _apply_cz_inplace(lam, op[1], op[2], n)
        elif kind == "cx":
            _apply_cx_inplace(psi, op[1], op[2], n)
            _apply_cx_inplace(lam, op[1], op[2], n)

    return energy, grad


def train_vqe(
    ansatz_type: str = "8q",
    layers: int = 1,
    n_seeds: int = 20,
    method: str = "BFGS",
    base_seed: int = 42,
    maxiter: int = 500,
    success_tol: float = 14.01,
) -> dict[str, Any]:
    """Run multi-start exact-gradient statevector VQE for the chosen ansatz."""
    n, costs, ops = _get_gate_program(ansatz_type, layers)
    costs_tensor = costs.reshape((2,) * n)
    n_params = n * (2 * layers + 1)
    opt_idx = OPTIMAL_IDX_8Q if n == 8 else OPTIMAL_IDX_12Q

    seed_results: list[dict[str, Any]] = []
    best_result: dict[str, Any] | None = None

    for s in range(n_seeds):
        rng = np.random.default_rng(base_seed + s)
        x0 = rng.uniform(-math.pi, math.pi, size=n_params)
        history: list[float] = []

        def objective(params_vec: np.ndarray) -> tuple[float, np.ndarray]:
            e, g = _energy_and_grad(params_vec, n, costs_tensor, ops)
            return e, g

        def callback(xk: np.ndarray) -> None:
            sv = simulate_statevector_fast(ansatz_type, xk, layers=layers)
            history.append(float(np.dot(np.abs(sv) ** 2, costs)))

        # Record initial energy
        sv0 = simulate_statevector_fast(ansatz_type, x0, layers=layers)
        history.append(float(np.dot(np.abs(sv0) ** 2, costs)))

        options: dict[str, Any] = {"maxiter": maxiter}
        if method == "BFGS":
            options["gtol"] = 1e-8
        elif method == "L-BFGS-B":
            options["ftol"] = 1e-13
            options["gtol"] = 1e-8

        res = minimize(
            objective,
            x0,
            method=method,
            jac=True,
            callback=callback,
            options=options,
        )
        # Wrap angles into (-pi, pi] for clean QASM readability
        opt_params = ((res.x + math.pi) % (2.0 * math.pi)) - math.pi
        sv = simulate_statevector_fast(ansatz_type, opt_params, layers=layers)
        probs = np.abs(sv) ** 2
        energy = float(np.dot(probs, costs))
        opt_prob = float(probs[opt_idx])
        top_idx = int(np.argmax(probs))
        top_bits = f"{top_idx:0{n}b}"[::-1]

        record = {
            "seed": base_seed + s,
            "energy": energy,
            "optimal_probability": opt_prob,
            "top_bitstring": top_bits,
            "top_probability": float(probs[top_idx]),
            "nit": int(res.nit),
            "nfev": int(res.nfev),
            "success": energy < success_tol,
            "params": opt_params,
            "history": history,
        }
        seed_results.append(record)
        if best_result is None or energy < best_result["energy"]:
            best_result = record

    assert best_result is not None
    success_count = sum(1 for r in seed_results if r["success"])
    energies = [r["energy"] for r in seed_results]

    return {
        "ansatz_type": ansatz_type,
        "layers": layers,
        "num_qubits": n,
        "num_params": n_params,
        "n_seeds": n_seeds,
        "method": method,
        "best_energy": best_result["energy"],
        "median_energy": float(np.median(energies)),
        "mean_energy": float(np.mean(energies)),
        "best_optimal_probability": best_result["optimal_probability"],
        "success_count": success_count,
        "success_rate": success_count / n_seeds,
        "best_params": best_result["params"],
        "best_seed": best_result["seed"],
        "best_history": best_result["history"],
        "seed_results": seed_results,
    }


# ---------------------------------------------------------------------------
# 5. Circuit Metrics, QASM Export & Submission Entry Point
# ---------------------------------------------------------------------------

def count_cz_rounds(circuit: QuantumCircuit) -> int:
    """Count minimum parallel CZ layers (depth of the CZ-only subcircuit)."""
    cz_only = QuantumCircuit(circuit.num_qubits)
    for inst in circuit.data:
        if inst.operation.name == "cz":
            cz_only.append(inst.operation, inst.qubits)
    return int(cz_only.depth())


def count_storage_entangling_visits(circuit: QuantumCircuit, commute_rz: bool = True) -> int:
    """Count minimum number of distinct entangling-zone visits forced by non-commuting 1Q gates.

    Models alternating Storage (even stages 0, 2, 4, ...) and Entangling (odd stages
    1, 3, 5, ...) phases on the circuit DAG.
    When ``commute_rz=True`` (default), exploits ``[Rz, CZ] = 0``: diagonal Z-axis
    rotations (``rz``, ``z``, ``s``, ``sdg``, ``t``, ``tdg``, ``p``) commute past ``cz``
    blocks into the adjacent Storage stage and therefore do NOT force an extra Storage
    round-trip. Only non-diagonal 1Q gates (e.g., ``rx``) separate consecutive ``cz``
    blocks into distinct entangling-zone visits.
    """
    from qiskit.converters import circuit_to_dag

    diagonal_1q = frozenset({"rz", "z", "s", "sdg", "t", "tdg", "p"})
    dag = circuit_to_dag(circuit.remove_final_measurements(inplace=False))
    qubit_stage: dict[Any, int] = {q: -1 for q in circuit.qubits}
    max_odd_stage = -1

    for node in dag.topological_op_nodes():
        if node.name in ("barrier", "measure"):
            continue
        preds = [qubit_stage[q] for q in node.qargs]
        m = max(preds) if preds else -1
        if node.name == "cz":
            # Stay in current odd entangling stage if m is odd; else advance to next odd stage
            stage = m if (m >= 1 and m % 2 == 1) else (m + 1 if m >= 0 else 1)
            max_odd_stage = max(max_odd_stage, stage)
        elif commute_rz and node.name in diagonal_1q:
            # Diagonal Z gates commute with CZ: do not force leaving an odd Entangling stage
            stage = 0 if m < 0 else m
        else:
            # Non-diagonal 1Q gate (e.g. rx) requires Storage (even stage 0, 2, 4, ...)
            stage = 0 if m < 0 else (m if m % 2 == 0 else m + 1)
        for q in node.qargs:
            qubit_stage[q] = stage

    return 0 if max_odd_stage < 0 else (max_odd_stage + 1) // 2


def inspect_native_metrics(circuit: QuantumCircuit, force_transpile: bool = False) -> dict[str, Any]:
    """Measure native {rx, rz, cz} gate counts, depth, CZ rounds, and commutation-aware zone visits.

    Non-native circuits (containing e.g. ``cx``, ``ry``, ``h``) are transpiled to
    ``["rx", "rz", "cz"]`` at ``optimization_level=3``. Circuits already strictly in
    ``{"rx", "rz", "cz"}`` are measured directly unless ``force_transpile=True``, avoiding
    zone-blind peephole passes that commute ``Rx(pi)`` into the middle of a CZ block.
    """
    nom = circuit.remove_final_measurements(inplace=False)
    already_native = set(nom.count_ops().keys()) <= {"rx", "rz", "cz"}
    native = (
        nom
        if (already_native and not force_transpile)
        else transpile(nom, basis_gates=["rx", "rz", "cz"], optimization_level=3)
    )
    ops = dict(native.count_ops())
    return {
        "qubits": int(native.num_qubits),
        "native_cz": int(ops.get("cz", 0)),
        "native_rx": int(ops.get("rx", 0)),
        "native_rz": int(ops.get("rz", 0)),
        "native_total_gates": int(native.size()),
        "native_depth": int(native.depth()),
        "cz_rounds": count_cz_rounds(native),
        "entangling_visits": count_storage_entangling_visits(native, commute_rz=True),
    }


def evaluate_circuit_exact(circuit: QuantumCircuit, encoding: str) -> dict[str, Any]:
    """Evaluate exact statevector expectation <F> and optimal probability via Qiskit Statevector."""
    nom = circuit.remove_final_measurements(inplace=False)
    sv = Statevector.from_instruction(nom)
    probs = np.abs(sv.data) ** 2

    if encoding == "8q":
        costs = COSTS_8Q
        opt_idx = OPTIMAL_IDX_8Q
        decoder = decode_8q_bitstring
        n = 8
    elif encoding == "12q":
        costs = COSTS_12Q
        opt_idx = OPTIMAL_IDX_12Q
        decoder = decode_12q_bitstring
        n = 12
    else:
        raise ValueError(f"Unknown encoding {encoding!r}")

    expected_cost = float(np.dot(probs, costs))
    opt_prob = float(probs[opt_idx])
    top_idx = int(np.argmax(probs))
    top_bits = f"{top_idx:0{n}b}"[::-1]
    return {
        "expected_cost": expected_cost,
        "optimal_probability": opt_prob,
        "top_bitstring": top_bits,
        "top_probability": float(probs[top_idx]),
        "decoded_top": decoder(top_bits),
    }


def export_qasm(circuit: QuantumCircuit, path: str | Path) -> Path:
    """Export a Qiskit QuantumCircuit to an OpenQASM 2.0 file."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        qasm2.dump(circuit, fh)
    return out_path


def verify_qasm_file(path: str | Path, encoding: str) -> dict[str, Any]:
    """Reload an OpenQASM 2.0 file, verify native {rx, rz, cz} gate set, and compute exact <F>."""
    qasm_str = Path(path).read_text(encoding="utf-8")
    qc = QuantumCircuit.from_qasm_str(qasm_str)
    qc_nom = qc.remove_final_measurements(inplace=False)
    op_names = set(qc_nom.count_ops().keys())
    allowed = {"rx", "rz", "cz"}
    illegal = op_names - allowed
    exact = evaluate_circuit_exact(qc_nom, encoding=encoding)
    return {
        "path": str(path),
        "num_qubits": qc.num_qubits,
        "has_measurements": "measure" in qc.count_ops(),
        "op_counts": dict(qc_nom.count_ops()),
        "illegal_gates": sorted(illegal),
        "is_strictly_native": len(illegal) == 0,
        "raw_depth": int(qc_nom.depth()),
        **exact,
    }


# Pre-optimized deterministic VQE parameters (base_seed=42, n_seeds=20, BFGS)
# Enables instant retrieval via get_bus_stop_circuit() while remaining 100% reproducible.
PRECOMPUTED_VQE_SOLUTIONS: dict[tuple[str, int], dict[str, Any]] = {
    ("8q", 1): {
        "best_params": np.array([
            1.012692736217318, -3.141592629321011, 1.6762301e-08, -0.598510980984999,
            2.8146939e-08, 5.019375e-09, 3.141592645381002, -0.118089166516368,
            3.141592630724222, 0.223387960968027, -0.345197843404026, 2.6353675e-08,
            2.709292050218851, -1.931091280206692, 1.423179157458717, -3.141592625457793,
            -1.012692736205678, 3.141592626689018, -3.141592642417526, 0.598510984538961,
            1.6301925e-08, 7.133092e-09, 3.141592651785549, -3.023503480497688,
        ], dtype=np.float64),
        "success_count": 20,
        "n_seeds": 20,
        "success_rate": 1.0,
        "best_seed": 55,
    },
    ("8q", 2): {
        "best_params": np.array([
            2.930060640346362, -3.141592652964015, -3.141592570653079, 0.929116123195875,
            -3.141592652501326, -2.558043092356482, -6.25018e-10, 2.001304716498336,
            0.398160234004888, 0.895034614858249, -2.989621818822258, 3.2092863e-08,
            0.856520506459423, -2.477424132952573, 2.010810322251931, 2.223703752956222,
            -2.889704475417041, -1.467607e-09, -0.339733193602279, -0.929116304630212,
            1.79087e-09, 2.702698088460487, 3.141592653224492, -1.390375631124446,
            0.186632662585461, -0.051155312065688, 2.212663e-08, -2.998643896870201,
            -2.967845256190818, 1.459002609975352, 0.46268476067521, 1.978182138577196,
            0.454177412391155, 3.141592652484272, -0.3397331444694, -1.95041405e-07,
            -3.141592652502184, -0.348795353758398, 3.141592652710906, 0.90460892283147,
        ], dtype=np.float64),
        "success_count": 19,
        "n_seeds": 20,
        "success_rate": 0.95,
        "best_seed": 46,
    },
    ("12q", 1): {
        "best_params": np.array([
            1.663238525801781, 2.815736013366589, -3.141592652739591, 5.32042e-10,
            4.455725e-09, 7.5778e-11, -6.07769e-10, -2.178867213184463,
            -9.602591e-09, 5.132086e-09, 1.15294821569905, -8.035712e-09,
            3.141592653547187, 3.141592652054038, 2.161208767921558, 1.311156080852704,
            -0.944876606187978, 1.535478734591345, -1.64129221769661, -4.8084e-11,
            -0.285669764227309, -0.269476195888939, 1.312044e-09, -0.20464519414333,
            1.478354129002351, 0.325856640046227, -3.141592650552497, 1.408357e-09,
            -3.141592645521049, -3.141592653131232, -3.14159264424578, -0.962725442016999,
            -6.867297e-09, 3.141592648907841, -1.152948218085283, -3.14159264592216,
        ], dtype=np.float64),
        "success_count": 4,
        "n_seeds": 20,
        "success_rate": 0.20,
        "best_seed": 56,
    },
    ("12q", 2): {
        "best_params": np.array([
            3.141592653122704, 3.141592650807305, -1.603254758056317, 1.972429036625062,
            -3.141592653510726, -2.374643111051228, 0.386442927825718, 1.166314219757294,
            -3.28518e-10, -8.08954e-10, 3.141592653209647, 0.018636364625634,
            1.493664647872937, 2.908273158366796, 1.460460990903586, 1.303377580715183,
            -2.926793315861827, 3.141592653558764, -0.589921387090356, -0.334281862309593,
            -1.752561480808046, -2.96429912338462, -3.100490321324285, -1.472255174668847,
            4.3242e-11, 2.24394e-10, -0.681605983027374, 0.964171241356861,
            3.141592653383221, 0.766949542546889, 1.550922e-09, -3.141592648958146,
            2.460239e-09, -3.14159265305384, -0.937737926968518, -3.071282926956027,
            -2.991858936094839, -2.231290597697271, -1.505440410648187, 1.368019153377293,
            -0.348880667970266, -1.929084673833753, 0.589921385937204, 2.807310792407134,
            -0.372909374139034, 2.587068616323577, -4.7416e-10, -0.251859060621394,
            -6.59236e-10, 3.28152e-09, 1.665477562151555, -2.007200910056481,
            -3.141592652927647, 1.45487e-10, -2.755149727188461, 1.166314223856548,
            -1.719479e-09, 1.89041e-10, -2.203854726167146, 0.074485082010307,
        ], dtype=np.float64),
        "success_count": 8,
        "n_seeds": 20,
        "success_rate": 0.40,
        "best_seed": 46,
    },
}

_OPTIMIZED_CACHE: dict[tuple[str, int], dict[str, Any]] = dict(PRECOMPUTED_VQE_SOLUTIONS)


def get_bus_stop_circuit(
    encoding: str = "8q",
    layers: int = 1,
    measure: bool = True,
    retrain: bool = False,
    clean_params: bool = True,
    prune_zeros: bool = True,
    decimals: int | None = 4,
    n_seeds: int = 20,
    base_seed: int = 42,
) -> tuple[QuantumCircuit, dict[str, Any]]:
    """Main submission entry point returning an optimized QuantumCircuit and metadata.

    Parameters
    ----------
    encoding : {"8q", "12q"}
        ``"8q"`` returns Candidate A (8-qubit triangle-compressed hardware-native ansatz).
        ``"12q"`` returns Candidate B (12-qubit 2x6 real-edges hardware-native ansatz).
    layers : int, default=1
        Number of variational layers (1 or 2).
    measure : bool, default=True
        Whether to append Z-basis measurements on all qubits.
    retrain : bool, default=False
        If True, re-runs multi-start VQE instead of using cached deterministic parameters.
    clean_params : bool, default=True
        If True, snaps near-zero / near-±pi angles and rounds remaining pulses via
        ``clean_hardware_params(..., decimals=decimals)``.
    prune_zeros : bool, default=True
        If True, omits zero-angle ``rx`` / ``rz`` gates from the returned circuit.
    decimals : int | None, default=4
        Decimal precision for ``clean_hardware_params`` (4 -> 0.1 mrad).
    n_seeds : int, default=20
        Number of multi-start VQE initializations if retraining or uncached.
    base_seed : int, default=42
        Base RNG seed for reproducibility.

    Returns
    -------
    tuple[QuantumCircuit, dict[str, Any]]
        Optimized circuit ready for ``compile_circuit(circuit, architecture_spec)`` and
        a metadata dictionary with exact energy, optimal probability, decoded stops,
        recommended initial layout, and native gate statistics.
    """
    if encoding not in ("8q", "12q"):
        raise ValueError(f"encoding must be '8q' or '12q', got {encoding!r}.")

    key = (encoding, layers)
    if retrain or key not in _OPTIMIZED_CACHE:
        _OPTIMIZED_CACHE[key] = train_vqe(
            ansatz_type=encoding,
            layers=layers,
            n_seeds=n_seeds,
            method="BFGS",
            base_seed=base_seed,
        )

    vqe_res = _OPTIMIZED_CACHE[key]
    raw_params = vqe_res["best_params"]
    best_params = (
        clean_hardware_params(raw_params, decimals=decimals)
        if clean_params
        else np.asarray(raw_params, dtype=np.float64)
    )

    if encoding == "8q":
        qc = build_8q_ansatz(best_params, layers=layers, measure=measure, prune_zeros=prune_zeros)
        layout = dict(RECOMMENDED_LAYOUT_8Q)
    else:
        qc = build_12q_ansatz(best_params, layers=layers, measure=measure, prune_zeros=prune_zeros)
        layout = dict(RECOMMENDED_LAYOUT_12Q)

    exact_eval = evaluate_circuit_exact(qc, encoding=encoding)
    native_metrics = inspect_native_metrics(qc)

    metadata: dict[str, Any] = {
        "encoding": encoding,
        "layers": layers,
        "num_qubits": qc.num_qubits,
        "num_params": len(best_params),
        "optimal_energy": exact_eval["expected_cost"],
        "optimal_probability": exact_eval["optimal_probability"],
        "optimal_bitstring": exact_eval["top_bitstring"],
        "decoded_stops": exact_eval["decoded_top"],
        "kept_stops": exact_eval["decoded_top"]["kept_stops"],
        "dropped_stops": exact_eval["decoded_top"]["dropped_stops"],
        "vqe_success_rate": vqe_res["success_rate"],
        "vqe_success_count": vqe_res["success_count"],
        "vqe_n_seeds": vqe_res["n_seeds"],
        "recommended_layout": layout,
        "native_metrics": native_metrics,
        "params": best_params.tolist(),
    }
    return qc, metadata


if __name__ == "__main__":
    # Quick CLI entry point: optimize, export QASM files, and print summary
    root = Path(__file__).resolve().parent
    for enc in ("8q", "12q"):
        qc_meas, meta = get_bus_stop_circuit(encoding=enc, layers=1, measure=True)
        qc_nom, _ = get_bus_stop_circuit(encoding=enc, layers=1, measure=False)
        export_qasm(qc_meas, root / f"ansatz_{enc}_p1_optimized.qasm")
        export_qasm(qc_nom, root / f"ansatz_{enc}_p1_nom.qasm")
        print(
            f"[{enc} p=1] <F>={meta['optimal_energy']:.8f}, "
            f"P(x*)={meta['optimal_probability']:.6f}, "
            f"success={meta['vqe_success_count']}/{meta['vqe_n_seeds']}, "
            f"kept={meta['kept_stops']}, native={meta['native_metrics']}"
        )
