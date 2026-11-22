"""Standalone 6-qubit encoding ablation study for PushQuantum 2026 (planqc track).

Scientifically proves why our 8-qubit triangle-compressed encoding is the optimal
("Goldilocks") trade-off between uncompressed 12-qubit and sub-8-qubit encodings:

1. Exact Information-Theoretic Lower Bound:
   Enumerates all feasible vertex covers (163 -> ceil(log2(163)) = 8 qubits lossless
   lower bound) and all 34 minimal vertex covers (30 four-drop + 4 three-drop covers),
   plus verifies an explicit valid-cost counterexample (c_i in {1, 4} < P=5) where
   6-qubit drop-1-per-triangle compression fails to capture the true global optimum.
2. 6-Qubit Pair-of-Triangles Encoding (3 + 3 = 6 qubits, 64 states, zero padding):
   Derives the exact SparsePauliOp Hamiltonian H_6q via Fast Walsh-Hadamard Transform,
   quantifying Pauli-Z term count, Hilbert-space density, and 6-local interaction blowup.
3. Hardware-Native 6-Qubit Single-Row Ansatz & Live Multi-Start COBYLA Comparison:
   Benchmarks 12q (5+5 real-edges) vs. 8q (triangle-compressed) vs. 6q (pair-of-triangles)
   side-by-side under identical 10-seed COBYLA settings and exports `ansatz_6q_p1_ablation.qasm`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector
from scipy.optimize import minimize

import bus_stop_ansatz as bsa
import bus_stop_vqe as bsv

# ---------------------------------------------------------------------------
# 1. Exact Information-Theoretic Lower Bound & Counterexample Verification
# ---------------------------------------------------------------------------

def analyze_feasible_vertex_covers() -> dict[str, Any]:
    """Enumerate all feasible and minimal vertex covers of the 12-stop graph."""
    feasible_drops: list[tuple[int, ...]] = []
    for idx in range(1 << bsa.NUM_STOPS):
        x = [(idx >> k) & 1 for k in range(bsa.NUM_STOPS)]
        if all(x[u - 1] == 1 or x[v - 1] == 1 for u, v in bsa.ALL_EDGES):
            dropped = tuple(i + 1 for i, val in enumerate(x) if val == 0)
            feasible_drops.append(dropped)

    # A vertex cover is minimal iff its dropped set is a maximal independent set
    drop_sets = [frozenset(d) for d in feasible_drops]
    minimal_drops = [
        d
        for d, ds in zip(feasible_drops, drop_sets, strict=True)
        if not any(ds < other for other in drop_sets)
    ]
    minimal_4_drop = sorted(d for d in minimal_drops if len(d) == 4)
    minimal_3_drop = sorted(d for d in minimal_drops if len(d) == 3)

    # Verify exactness counterexample: c_3 = c_6 = c_9 = 4, all others = 1 (< P = 5)
    ce_costs = {s: (4 if s in (3, 6, 9) else 1) for s in range(1, bsa.NUM_STOPS + 1)}

    def eval_ce(dropped: Sequence[int]) -> float:
        dset = set(dropped)
        x = [0 if s in dset else 1 for s in range(1, bsa.NUM_STOPS + 1)]
        maint = sum(ce_costs[s] * x[s - 1] for s in range(1, bsa.NUM_STOPS + 1))
        pen = bsa.PENALTY * sum(
            (1 - x[u - 1]) * (1 - x[v - 1]) for u, v in bsa.ALL_EDGES
        )
        return float(maint + pen)

    # True global optimum across all 2^12 states (also captured by 8q encoding!)
    best_true_drop = min(feasible_drops, key=eval_ce)
    best_true_cost = eval_ce(best_true_drop)

    # Best inside 6q subspace (must drop exactly 1 stop from each of the 4 triangles)
    drop_1_per_tri = [
        (d1, d2, d3, d4)
        for d1 in bsa.TRIANGLES[0]
        for d2 in bsa.TRIANGLES[1]
        for d3 in bsa.TRIANGLES[2]
        for d4 in bsa.TRIANGLES[3]
        if (d1, d2) != (1, 4) and (d3, d4) != (9, 12)
    ]
    best_6q_drop = min(drop_1_per_tri, key=eval_ce)
    best_6q_cost = eval_ce(best_6q_drop)

    return {
        "total_feasible": len(feasible_drops),
        "lossless_qubit_lower_bound": math.ceil(math.log2(len(feasible_drops))),
        "minimal_covers_total": len(minimal_drops),
        "minimal_4_drop_count": len(minimal_4_drop),
        "minimal_3_drop": minimal_3_drop,
        "counterexample_costs": ce_costs,
        "counterexample_true_drop": best_true_drop,
        "counterexample_true_cost": best_true_cost,
        "counterexample_6q_drop": best_6q_drop,
        "counterexample_6q_cost": best_6q_cost,
    }


# ---------------------------------------------------------------------------
# 2. 6-Qubit "Pair-of-Triangles" Encoding (3 + 3 = 6 qubits, 64 states)
# ---------------------------------------------------------------------------

# Pair A: (T1, T2) excluding internal bridge (1, 4) -> 3*3 - 1 = 8 states (q0, q1, q2)
PAIR_A_DROPS: tuple[tuple[int, int], ...] = (
    (1, 5),
    (1, 6),
    (2, 4),
    (2, 5),
    (2, 6),
    (3, 4),
    (3, 5),
    (3, 6),
)

# Pair B: (T3, T4) excluding internal bridge (9, 12) -> 3*3 - 1 = 8 states (q3, q4, q5)
PAIR_B_DROPS: tuple[tuple[int, int], ...] = (
    (7, 10),
    (7, 11),
    (7, 12),
    (8, 10),
    (8, 11),
    (8, 12),
    (9, 10),
    (9, 11),
)

ROUNDS_6Q: tuple[tuple[tuple[int, int], ...], ...] = (
    ((0, 1), (2, 3), (4, 5)),  # Round 1: 3 adjacent pairs on 1 row
    ((1, 2), (3, 4)),          # Round 2: 2 shifted pairs (1-col AOD shift)
)


def decode_6q_bitstring(bits_q0_to_q5: str) -> dict[str, Any]:
    """Decode a 6-qubit bitstring in q0..q5 order into 12-stop decisions and exact F(x)."""
    if len(bits_q0_to_q5) != 6 or any(ch not in "01" for ch in bits_q0_to_q5):
        raise ValueError(f"Expected 6-char binary string in q0..q5 order, got {bits_q0_to_q5!r}.")
    idx_a = int(bits_q0_to_q5[0:3], 2)
    idx_b = int(bits_q0_to_q5[3:6], 2)
    dropped = PAIR_A_DROPS[idx_a] + PAIR_B_DROPS[idx_b]
    dset = set(dropped)
    x = tuple(0 if s in dset else 1 for s in range(1, bsa.NUM_STOPS + 1))
    violated_bridges = [e for e in bsa.BRIDGES if x[e[0] - 1] == 0 and x[e[1] - 1] == 0]
    return {
        "bits_q0_to_q5": bits_q0_to_q5,
        "bits_q5_to_q0": bits_q0_to_q5[::-1],
        "x": x,
        "kept_stops": [s for s in range(1, bsa.NUM_STOPS + 1) if s not in dset],
        "dropped_stops": list(dropped),
        "violated_bridges": violated_bridges,
        "is_feasible": len(violated_bridges) == 0,
        "cost": bsa.evaluate_stops_cost(x),
    }


def _build_costs_6q() -> np.ndarray:
    """Precompute exact 64-entry cost vector indexed by Qiskit little-endian state index."""
    costs = np.zeros(1 << 6, dtype=np.float64)
    for idx in range(1 << 6):
        bits = f"{idx:06b}"[::-1]  # q0..q5 order
        costs[idx] = decode_6q_bitstring(bits)["cost"]
    return costs


COSTS_6Q: np.ndarray = _build_costs_6q()
OPTIMAL_IDX_6Q: int = int(np.argmin(COSTS_6Q))


def _fwht_pauli_z(diag_values: np.ndarray, tol: float = 1e-12) -> SparsePauliOp:
    """Derive exact diagonal Pauli-Z SparsePauliOp from a 2^n cost vector via FWHT.

    In Qiskit little-endian convention, state index `z` has qubit `q` at bit `(z >> q) & 1`,
    so Walsh-Hadamard mask `S` corresponds directly to Pauli-Z on `{q : (S >> q) & 1}`.
    """
    n_states = len(diag_values)
    n_qubits = int(round(math.log2(n_states)))
    if (1 << n_qubits) != n_states:
        raise ValueError(f"Expected power-of-2 length, got {n_states}.")

    coeffs = np.array(diag_values, dtype=np.float64, copy=True)
    h = 1
    while h < n_states:
        for i in range(0, n_states, h * 2):
            u = coeffs[i : i + h].copy()
            v = coeffs[i + h : i + 2 * h].copy()
            coeffs[i : i + h] = u + v
            coeffs[i + h : i + 2 * h] = u - v
        h *= 2
    coeffs /= float(n_states)

    pauli_list: list[tuple[str, float]] = []
    for mask, coeff in enumerate(coeffs):
        if abs(coeff) > tol:
            label = "".join("Z" if (mask >> (n_qubits - 1 - pos)) & 1 else "I" for pos in range(n_qubits))
            pauli_list.append((label, float(coeff)))
    return SparsePauliOp.from_list(pauli_list).simplify()


def build_6q_sparse_pauli_hamiltonian() -> SparsePauliOp:
    """Exact 6-qubit SparsePauliOp Hamiltonian H_6q derived via Walsh-Hadamard transform."""
    return _fwht_pauli_z(COSTS_6Q)


def pauli_hamiltonian_stats(op: SparsePauliOp) -> dict[str, Any]:
    """Return term count, density (terms / 2^n), and maximum Pauli locality."""
    n_qubits = op.num_qubits
    labels = op.paulis.to_labels()
    weights = [lbl.count("Z") + lbl.count("X") + lbl.count("Y") for lbl in labels]
    return {
        "num_terms": len(labels),
        "density_pct": 100.0 * len(labels) / float(1 << n_qubits),
        "max_locality": max(weights) if weights else 0,
    }


# ---------------------------------------------------------------------------
# 3. Hardware-Native 6-Qubit Single-Row Ansatz & Multi-Start COBYLA Trainer
# ---------------------------------------------------------------------------

def num_params_6q(layers: int = 1) -> int:
    if layers < 1:
        raise ValueError("layers must be >= 1")
    return 6 * (2 * layers + 1)


def build_6q_ansatz(
    params: Sequence[float],
    layers: int = 1,
    measure: bool = False,
    prune_zeros: bool = True,
    zero_tol: float = 1e-7,
) -> QuantumCircuit:
    """Build hardware-native 6-qubit single-row {rx, rz, cz} ansatz mirroring build_8q_ansatz."""
    expected = num_params_6q(layers)
    p = np.asarray(params, dtype=np.float64)
    if p.shape != (expected,):
        raise ValueError(f"6q ansatz (layers={layers}) expects {expected} params, got {p.shape}.")

    qc = QuantumCircuit(6, 6) if measure else QuantumCircuit(6)
    idx = 0
    for layer in range(layers):
        for k in range(6):
            ang = float(p[idx + k])
            if not prune_zeros or abs(ang) > zero_tol:
                qc.rx(ang, k)
        idx += 6
        for k in range(6):
            ang = float(p[idx + k])
            if not prune_zeros or abs(ang) > zero_tol:
                qc.rz(ang, k)
        idx += 6

        round_order = ROUNDS_6Q if (layer % 2 == 0) else tuple(reversed(ROUNDS_6Q))
        for ent_round in round_order:
            for q_a, q_b in ent_round:
                qc.cz(q_a, q_b)

    for k in range(6):
        ang = float(p[idx + k])
        if not prune_zeros or abs(ang) > zero_tol:
            qc.rx(ang, k)

    if measure:
        qc.measure(range(6), range(6))
    return qc


def train_vqe_6q(
    layers: int = 1,
    n_seeds: int = 10,
    base_seed: int = 42,
    maxiter: int = 2000,
    tol: float = 1e-5,
    success_tol: float = 14.01,
) -> dict[str, Any]:
    """Run multi-start derivative-free COBYLA on the 6q ansatz matching bus_stop_vqe.py."""
    h6_op = build_6q_sparse_pauli_hamiltonian()
    h6_diag = np.asarray(h6_op.to_matrix(sparse=True).diagonal().real, dtype=np.float64)
    n_params = num_params_6q(layers)

    seed_results: list[dict[str, Any]] = []
    best_record: dict[str, Any] | None = None

    for s in range(n_seeds):
        seed = base_seed + s
        rng = np.random.default_rng(seed)
        x0 = rng.uniform(-math.pi, math.pi, size=n_params)

        def objective(params_vec: np.ndarray) -> float:
            qc = build_6q_ansatz(params_vec, layers=layers, measure=False, prune_zeros=False)
            sv = Statevector.from_instruction(qc)
            return float(np.dot(np.abs(sv.data) ** 2, h6_diag))

        res = minimize(objective, x0, method="COBYLA", tol=tol, options={"maxiter": maxiter})
        opt_params = ((res.x + math.pi) % (2.0 * math.pi)) - math.pi
        final_qc = build_6q_ansatz(opt_params, layers=layers, measure=False, prune_zeros=False)
        final_sv = Statevector.from_instruction(final_qc)
        final_energy = float(final_sv.expectation_value(h6_op).real)
        probs = np.abs(final_sv.data) ** 2
        opt_prob = float(probs[OPTIMAL_IDX_6Q])
        top_idx = int(np.argmax(probs))
        top_bits = f"{top_idx:06b}"[::-1]

        rec = {
            "seed": seed,
            "energy": final_energy,
            "optimal_probability": opt_prob,
            "top_bitstring": top_bits,
            "success": final_energy < success_tol,
            "params": opt_params,
        }
        seed_results.append(rec)
        if best_record is None or final_energy < best_record["energy"]:
            best_record = rec

    assert best_record is not None
    succ = sum(1 for r in seed_results if r["success"])
    return {
        "ansatz_type": "6q",
        "layers": layers,
        "num_qubits": 6,
        "num_params": n_params,
        "optimizer": "COBYLA",
        "n_seeds": n_seeds,
        "best_energy": best_record["energy"],
        "best_optimal_probability": best_record["optimal_probability"],
        "success_count": succ,
        "success_rate": succ / n_seeds,
        "best_params": best_record["params"],
        "best_seed": best_record["seed"],
        "top_bitstring": best_record["top_bitstring"],
    }


# ---------------------------------------------------------------------------
# 4. Main Verification, Ablation Benchmark & Summary Table
# ---------------------------------------------------------------------------

def main() -> None:
    root = Path(__file__).resolve().parent

    # --- Part 1: Information-Theoretic Lower Bound & Counterexample ---
    bounds = analyze_feasible_vertex_covers()
    assert bounds["total_feasible"] == 163, bounds
    assert bounds["lossless_qubit_lower_bound"] == 8, bounds
    assert bounds["minimal_covers_total"] == 34, bounds
    assert bounds["minimal_4_drop_count"] == 30, bounds
    expected_3_drop = [(1, 8, 11), (2, 5, 12), (3, 6, 9), (4, 7, 10)]
    assert bounds["minimal_3_drop"] == expected_3_drop, bounds["minimal_3_drop"]
    assert bounds["counterexample_true_drop"] == (3, 6, 9)
    assert bounds["counterexample_true_cost"] == 9.0
    assert bounds["counterexample_6q_cost"] == 11.0

    print("=" * 96)
    print("1. EXACT INFORMATION-THEORETIC LOWER BOUND & LOSSLESSNESS PROOF")
    print("=" * 96)
    print(f"  Feasible vertex covers (12-stop graph) : {bounds['total_feasible']} -> ceil(log2(163)) = {bounds['lossless_qubit_lower_bound']} qubits (exact lower bound)")
    print(f"  Minimal vertex covers                  : {bounds['minimal_covers_total']} = {bounds['minimal_4_drop_count']} (4-drop) + {len(bounds['minimal_3_drop'])} (3-drop)")
    print(f"  3-drop maximal independent sets        : {bounds['minimal_3_drop']}")
    print("  Exactness counterexample (c_3=c_6=c_9=4, others=1 < P=5):")
    print(f"    - True optimum (captured by 8q)      : drop {list(bounds['counterexample_true_drop'])}, cost = {bounds['counterexample_true_cost']:.1f}")
    print(f"    - Best in 6q subspace (lossy!)       : drop {list(bounds['counterexample_6q_drop'])}, cost = {bounds['counterexample_6q_cost']:.1f}  [WRONG OPTIMUM]")

    # --- Part 2: 6q Hamiltonian & Pauli Locality Comparison ---
    assert float(np.min(COSTS_6Q)) == 14.0
    assert int(np.sum(np.isclose(COSTS_6Q, 14.0))) == 1
    opt6_dec = decode_6q_bitstring(f"{OPTIMAL_IDX_6Q:06b}"[::-1])
    assert set(opt6_dec["kept_stops"]) == set(bsa.OPTIMAL_KEPT_STOPS)

    h12_op = bsa.build_sparse_pauli_hamiltonian("12q")
    h8_op = bsa.build_sparse_pauli_hamiltonian("8q")
    h6_op = build_6q_sparse_pauli_hamiltonian()

    h6_diag_err = float(np.max(np.abs(h6_op.to_matrix(sparse=True).diagonal().real - COSTS_6Q)))
    assert h6_diag_err < 1e-12, f"H_6q diagonal error: {h6_diag_err}"

    st12 = pauli_hamiltonian_stats(h12_op)
    st8 = pauli_hamiltonian_stats(h8_op)
    st6 = pauli_hamiltonian_stats(h6_op)
    assert st6["max_locality"] == 6, st6
    assert st8["max_locality"] == 4, st8
    assert st12["max_locality"] == 2, st12

    print("\n" + "=" * 96)
    print("2. HAMILTONIAN PAULI-Z LOCALITY & DENSITY (FWHT VERIFICATION)")
    print("=" * 96)
    print(f"  H_6q max diagonal error vs COSTS_6Q    : {h6_diag_err:.2e} (exact)")
    print(f"  6q unique optimum decoded kept stops   : {opt6_dec['kept_stops']} (bits={opt6_dec['bits_q0_to_q5']}, cost={opt6_dec['cost']:.1f})")
    print(f"  12q H_12q: {st12['num_terms']:2d} terms ({st12['density_pct']:5.2f}% of 4096), max locality = {st12['max_locality']}-local")
    print(f"   8q H_8q : {st8['num_terms']:2d} terms ({st8['density_pct']:5.2f}% of  256), max locality = {st8['max_locality']}-local")
    print(f"   6q H_6q : {st6['num_terms']:2d} terms ({st6['density_pct']:5.2f}% of   64), max locality = {st6['max_locality']}-local")

    # --- Part 3: Live 10-Seed COBYLA Benchmark (12q 5+5 vs 8q vs 6q) ---
    rng = np.random.default_rng(7)
    nm12 = bsa.inspect_native_metrics(bsa.build_12q_ansatz(rng.uniform(-math.pi, math.pi, 36), layers=1, measure=False))
    nm8 = bsa.inspect_native_metrics(bsa.build_8q_ansatz(rng.uniform(-math.pi, math.pi, 24), layers=1, measure=False))
    nm6 = bsa.inspect_native_metrics(build_6q_ansatz(rng.uniform(-math.pi, math.pi, 18), layers=1, measure=False))

    vqe12 = bsv.train_vqe(ansatz_type="12q", layers=1, n_seeds=10, base_seed=42, maxiter=2000, tol=1e-5)
    vqe8 = bsv.train_vqe(ansatz_type="8q", layers=1, n_seeds=10, base_seed=42, maxiter=2000, tol=1e-5)
    vqe6 = train_vqe_6q(layers=1, n_seeds=10, base_seed=42, maxiter=2000, tol=1e-5)

    # Export cleaned best 6q circuit and verify reload
    cleaned_6q = bsa.clean_hardware_params(vqe6["best_params"], decimals=4)
    qc6_export = build_6q_ansatz(cleaned_6q, layers=1, measure=True, prune_zeros=True)
    qasm6_path = bsa.export_qasm(qc6_export, root / "ansatz_6q_p1_ablation.qasm")

    reloaded_6q = QuantumCircuit.from_qasm_str(qasm6_path.read_text(encoding="utf-8"))
    reloaded_nom = reloaded_6q.remove_final_measurements(inplace=False)
    assert set(reloaded_nom.count_ops().keys()) <= {"rx", "rz", "cz"}
    assert reloaded_nom.count_ops().get("cz") == 5
    sv_rel = Statevector.from_instruction(reloaded_nom)
    e_rel = float(sv_rel.expectation_value(h6_op).real)
    p_rel = float((np.abs(sv_rel.data) ** 2)[OPTIMAL_IDX_6Q])
    assert abs(e_rel - 14.0) < 1e-4 and p_rel > 0.9999

    # --- Part 4: Clean Side-by-Side CLI Summary Table ---
    rows = [
        ("12q 5+5 Real-Edges", 12, 4096, "Yes", f"{st12['num_terms']} ({st12['density_pct']:.1f}%)", f"{st12['max_locality']}-local", nm12["native_cz"], nm12["entangling_visits"], nm12["native_depth"], f"{vqe12['best_energy']:.6f}", f"{vqe12['success_count']}/{vqe12['n_seeds']} ({100*vqe12['success_rate']:.0f}%)"),
        ("8q Triangle-Comp. ★", 8, 256, "Yes (exact bound)", f"{st8['num_terms']} ({st8['density_pct']:.1f}%)", f"{st8['max_locality']}-local", nm8["native_cz"], nm8["entangling_visits"], nm8["native_depth"], f"{vqe8['best_energy']:.6f}", f"{vqe8['success_count']}/{vqe8['n_seeds']} ({100*vqe8['success_rate']:.0f}%)"),
        ("6q Pair-of-Tri (Abl.)", 6, 64, "No (misses 3-drop)", f"{st6['num_terms']} ({st6['density_pct']:.1f}%)", f"{st6['max_locality']}-local", nm6["native_cz"], nm6["entangling_visits"], nm6["native_depth"], f"{vqe6['best_energy']:.6f}", f"{vqe6['success_count']}/{vqe6['n_seeds']} ({100*vqe6['success_rate']:.0f}%)"),
    ]

    headers = [
        "Encoding (p=1)",
        "Qubits",
        "Hilbert",
        "Lossless all c_i?",
        "Pauli Terms (Dens)",
        "Max Loc",
        "CZ",
        "Visits",
        "Depth",
        "Best <F>",
        "COBYLA (10 seeds)",
    ]
    col_w = [max(len(h), max(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]
    sep = "+-" + "-+-".join("-" * w for w in col_w) + "-+"
    fmt = "| " + " | ".join(f"{{:<{w}}}" for w in col_w) + " |"

    print("\n" + "=" * 96)
    print("3. GOLDILOCKS ENCODING ABLATION SUMMARY (12q 5+5 vs. 8q vs. 6q, LIVE 10-SEED COBYLA)")
    print("=" * 96)
    print(sep)
    print(fmt.format(*headers))
    print(sep)
    for r in rows:
        print(fmt.format(*(str(x) for x in r)))
    print(sep)
    print(
        f"\nExported & verified {qasm6_path.name}: "
        f"ops={dict(reloaded_nom.count_ops())}, depth={reloaded_nom.depth()}, "
        f"<F>={e_rel:.8f}, P(x*)={p_rel:.8f}"
    )


if __name__ == "__main__":
    main()
