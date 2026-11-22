"""Hardware-native variational ansatze, problem encodings, and Hamiltonians for the
PushQuantum 2026 planqc 12-bus-stop Minimum-Weight Vertex Cover problem.

Scope:
    This module owns the problem definitions, exact cost evaluators, decoders,
    Qiskit ``SparsePauliOp`` Hamiltonians, hardware-native Qiskit circuit builders
    (``{rx, rz, cz}``), laser-precision angle pruning, native circuit metrics,
    OpenQASM 2.0 export, cached fallback solutions, and the submission entry point
    ``get_bus_stop_circuit``.

    Live VQE optimization is cleanly separated into an external teammate module
    (default ``bus_stop_vqe.py``) invoked via ``run_external_vqe(...)``.
    Neutral-atom routing/compilation is kept separate (``compile_circuit``).
"""

from __future__ import annotations

import importlib
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit import qasm2
from qiskit.quantum_info import SparsePauliOp, Statevector

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

# Entangling rounds for Candidate A (8q single-row) and Candidate B (12q 5+5 real-edges)
ROUNDS_8Q: tuple[tuple[tuple[int, int], ...], ...] = (
    ((0, 1), (2, 3), (4, 5), (6, 7)),  # Round 1: intra-triangle pairs
    ((1, 2), (3, 4), (5, 6)),          # Round 2: inter-triangle pairs (1-col AOD shift)
)

# 7-stop chain 3-2-1-4-5-6-11 + 5-stop chain 10-12-9-8-7:
# 10 real bus-stop edges (incl. 3 bridges), max matching = 5, 2 pulses of 5 pairs each.
ROUNDS_12Q: tuple[tuple[tuple[int, int], ...], ...] = (
    ((2, 1), (0, 3), (4, 5), (9, 11), (8, 7)),   # Round 1: 5 real edges (<= 5 pairs/row!)
    ((1, 0), (3, 4), (5, 10), (11, 8), (7, 6)),  # Round 2: 5 real edges (<= 5 pairs/row!)
)

# Recommended storage-zone initial placements (qubit -> (x, y)) for the compiler teammate
RECOMMENDED_LAYOUT_8Q: dict[int, tuple[int, int]] = {k: (k, 9) for k in range(8)}

# Row 9 cols 0..6 (stops 3-2-1-4-5-6-11) -> qubits [2, 1, 0, 3, 4, 5, 10]
# Row 8 cols 0..4 (stops 10-12-9-8-7)    -> qubits [9, 11, 8, 7, 6]
RECOMMENDED_LAYOUT_12Q: dict[int, tuple[int, int]] = {
    **{q: (col, 9) for col, q in enumerate((2, 1, 0, 3, 4, 5, 10))},
    **{q: (col, 8) for col, q in enumerate((9, 11, 8, 7, 6))},
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


def _z_pauli_label(qubits: Sequence[int], num_qubits: int) -> str:
    """Format a Qiskit little-endian Pauli-Z string (qubit q at index num_qubits - 1 - q)."""
    chars = ["I"] * num_qubits
    for q in qubits:
        chars[num_qubits - 1 - q] = "Z"
    return "".join(chars)


def build_sparse_pauli_hamiltonian(encoding: str = "8q") -> SparsePauliOp:
    """Construct the exact Qiskit ``SparsePauliOp`` for objective F(x).

    Matches ``COSTS_8Q`` (for ``encoding="8q"``) or ``COSTS_12Q`` (for ``encoding="12q"``)
    exactly along the computational-basis diagonal in Qiskit little-endian qubit order.

    Parameters
    ----------
    encoding : {"8q", "12q"}, default="8q"
        - ``"8q"``: exact 8-qubit triangle-compressed projector Hamiltonian H_8q.
        - ``"12q"``: rescaled 12-qubit Ising Hamiltonian F(x) = (H_12q + 146) / 4.
    """
    if encoding == "8q":
        terms = build_8q_pauli_hamiltonian()
        pauli_list = [(_z_pauli_label(qs, 8), float(coeff)) for qs, coeff in terms.items()]
        return SparsePauliOp.from_list(pauli_list).simplify()

    if encoding == "12q":
        pauli_list: list[tuple[str, float]] = [(_z_pauli_label((), 12), 146.0 / 4.0)]
        for stop_id in range(1, NUM_STOPS + 1):
            coeff = (15.0 - 2.0 * float(STOP_COSTS[stop_id])) / 4.0
            if abs(coeff) > 1e-12:
                pauli_list.append((_z_pauli_label((stop_id - 1,), 12), coeff))
        for u, v in ALL_EDGES:
            pauli_list.append((_z_pauli_label((u - 1, v - 1), 12), PENALTY / 4.0))
        return SparsePauliOp.from_list(pauli_list).simplify()

    raise ValueError(f"encoding must be '8q' or '12q', got {encoding!r}.")


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
    """Build Candidate B: 12-qubit 5+5 real-edges hardware-native {rx, rz, cz} ansatz.

    Uses 10 CZ gates per layer acting exclusively on real problem edges in 2 parallel
    pulses of 5 pairs each (respecting the 5-pairs-per-row Entangling zone limit).
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
# 4. External VQE Delegation Interface (Plug-In Hook for bus_stop_vqe.py)
# ---------------------------------------------------------------------------

_VQE_CONTRACT_DOC = (
    "Expected signature in `{module}.py`:\n"
    "    train_vqe(ansatz_type: str = '8q', layers: int = 1, n_seeds: int = 20, "
    "base_seed: int = 42, **kwargs) -> dict\n"
    "Expected return keys:\n"
    "    'best_params', 'best_energy', 'best_optimal_probability', "
    "'success_count', 'n_seeds', 'success_rate' (and optional 'best_history')."
)


def run_external_vqe(
    ansatz_type: str = "8q",
    layers: int = 1,
    n_seeds: int = 20,
    base_seed: int = 42,
    vqe_module: str = "bus_stop_vqe",
    **kwargs: Any,
) -> dict[str, Any]:
    """Delegate live VQE optimization to an external teammate module (default ``bus_stop_vqe``).

    Parameters
    ----------
    ansatz_type : {"8q", "12q", "hea_linear"}, default="8q"
        Target ansatz family to optimize.
    layers : int, default=1
        Number of variational layers.
    n_seeds : int, default=20
        Number of multi-start random initializations.
    base_seed : int, default=42
        Base RNG seed for reproducibility.
    vqe_module : str, default="bus_stop_vqe"
        Python module path exporting ``train_vqe(...)``.
    **kwargs : Any
        Additional optimizer keyword arguments forwarded directly to ``train_vqe``.

    Returns
    -------
    dict[str, Any]
        Result dictionary adhering to the teammate VQE contract:
        - ``best_params`` (Sequence[float] | np.ndarray): best variational parameter vector
        - ``best_energy`` (float): minimum achieved expectation value ``<F>``
        - ``best_optimal_probability`` (float): probability ``P(x*)`` of the unique optimum
        - ``success_count`` (int): number of seeds reaching ``F < 14.01``
        - ``n_seeds`` (int): total seeds evaluated
        - ``success_rate`` (float): ``success_count / n_seeds``
        - ``best_history`` (Sequence[float], optional): convergence trace of the best seed

    Raises
    ------
    ImportError
        If ``vqe_module`` cannot be imported.
    AttributeError
        If ``vqe_module`` does not define a callable ``train_vqe`` function.
    """
    try:
        mod = importlib.import_module(vqe_module)
    except ImportError as exc:
        raise ImportError(
            f"Could not import external VQE module '{vqe_module}'.\n"
            + _VQE_CONTRACT_DOC.format(module=vqe_module)
        ) from exc

    train_fn = getattr(mod, "train_vqe", None)
    if not callable(train_fn):
        raise AttributeError(
            f"External VQE module '{vqe_module}' does not expose a callable `train_vqe`.\n"
            + _VQE_CONTRACT_DOC.format(module=vqe_module)
        )

    result = train_fn(
        ansatz_type=ansatz_type,
        layers=layers,
        n_seeds=n_seeds,
        base_seed=base_seed,
        **kwargs,
    )
    if not isinstance(result, dict) or "best_params" not in result:
        raise ValueError(
            f"`{vqe_module}.train_vqe` must return a dict containing 'best_params'.\n"
            + _VQE_CONTRACT_DOC.format(module=vqe_module)
        )
    return result


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


# Cached fallback VQE parameters (n_seeds=20, base_seed=42) enabling instant
# retrieval via get_bus_stop_circuit() before/without live bus_stop_vqe.py execution.
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
        "best_energy": 14.0,
        "best_optimal_probability": 1.0,
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
        "best_energy": 14.0,
        "best_optimal_probability": 1.0,
        "success_count": 19,
        "n_seeds": 20,
        "success_rate": 0.95,
        "best_seed": 46,
    },
    ("12q", 1): {
        "best_params": np.array([
            -5.77185e-09, 3.141592647082327, 0.957056836975628, 3.86624e-09,
            -1.155719154752306, -3.141592653515239, -2.517406559663601, 3.589063e-09,
            -1.23066241418578, -3.141592649470937, -2.62667020462947, -7.17163e-10,
            -2.741165738741704, 3.081896784282815, 3.141592649653476, 2.942918053699405,
            7.077729e-09, 2.639867854174749, 3.141592587328574, 0.591574331662315,
            -9.761553e-09, 2.014228918351368, -1.07060835e-07, -2.564997976684237,
            -3.141592645736606, 5.808641e-09, -0.957056838737416, 9.442449e-09,
            1.985873502862624, 9.44727e-10, 0.62418608142989, 3.141592640774485,
            1.230662413678975, 5.039134e-09, -2.626670212408607, -3.141592647956318,
        ], dtype=np.float64),
        "best_energy": 14.0,
        "best_optimal_probability": 1.0,
        "success_count": 5,
        "n_seeds": 20,
        "success_rate": 0.25,
        "best_seed": 50,
    },
    ("12q", 2): {
        "best_params": np.array([
            3.141591177055134, 3.14159259425041, -2.118967635343743, 0.1143795379633,
            -2.173993e-09, 1.369449501692786, 0.71808069106508, -3.141592651687265,
            -2.709136087492346, 0.603246572136573, -7.105816e-09, -2.3175012e-08,
            3.136852879320613, -0.558973061276772, -3.08601391507093, 2.356663968298937,
            0.550199441618191, 0.011050311849217, 3.119551372596149, -0.781483247516652,
            -0.612854742796328, 0.020718539846368, -2.147192465695299, -2.506265343791525,
            -3.141591170994395, -3.141592614898599, 0.961861516699855, -1.496617806360465,
            -1.2494303e-08, 1.769462201720983, -0.71839024739158, 3.069851e-09,
            -0.869146801260586, 2.788194745504887, -2.230419e-08, 3.478622e-09,
            -2.263015872734264, 2.678484099241744, -0.051775081084317, 3.059861557438116,
            1.719139974142275, 1.814560792468365, 1.60043433640042, -1.855316452076576,
            0.257676684046999, -3.094072249978197, -3.041475403316037, 2.465769349979253,
            -3.141592499465728, -3.141592601610388, 1.158286390948855, -1.41603485040255,
            -3.1415926468733, -0.011156921728893, -3.127085088856042, -1.13183e-09,
            1.900971810518015, 0.250019035129232, 3.1161943e-08, 3.141592634936017,
        ], dtype=np.float64),
        "best_energy": 14.0,
        "best_optimal_probability": 1.0,
        "success_count": 10,
        "n_seeds": 20,
        "success_rate": 0.50,
        "best_seed": 55,
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
    vqe_module: str = "bus_stop_vqe",
) -> tuple[QuantumCircuit, dict[str, Any]]:
    """Main submission entry point returning an optimized QuantumCircuit and metadata.

    Parameters
    ----------
    encoding : {"8q", "12q"}
        ``"8q"`` returns Candidate A (8-qubit triangle-compressed hardware-native ansatz).
        ``"12q"`` returns Candidate B (12-qubit 5+5 real-edges hardware-native ansatz).
    layers : int, default=1
        Number of variational layers (1 or 2).
    measure : bool, default=True
        Whether to append Z-basis measurements on all qubits.
    retrain : bool, default=False
        If True, delegates live VQE training to ``vqe_module`` via ``run_external_vqe``
        instead of using cached fallback parameters.
    clean_params : bool, default=True
        If True, snaps near-zero / near-±pi angles and rounds remaining pulses via
        ``clean_hardware_params(..., decimals=decimals)``.
    prune_zeros : bool, default=True
        If True, omits zero-angle ``rx`` / ``rz`` gates from the returned circuit.
    decimals : int | None, default=4
        Decimal precision for ``clean_hardware_params`` (4 -> 0.1 mrad).
    n_seeds : int, default=20
        Number of multi-start VQE initializations forwarded to ``run_external_vqe``.
    base_seed : int, default=42
        Base RNG seed forwarded to ``run_external_vqe``.
    vqe_module : str, default="bus_stop_vqe"
        External VQE teammate module used when retraining or when ``(encoding, layers)``
        is not present in ``PRECOMPUTED_VQE_SOLUTIONS``.

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
        _OPTIMIZED_CACHE[key] = run_external_vqe(
            ansatz_type=encoding,
            layers=layers,
            n_seeds=n_seeds,
            base_seed=base_seed,
            vqe_module=vqe_module,
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
        "vqe_success_rate": vqe_res.get("success_rate"),
        "vqe_success_count": vqe_res.get("success_count"),
        "vqe_n_seeds": vqe_res.get("n_seeds"),
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
