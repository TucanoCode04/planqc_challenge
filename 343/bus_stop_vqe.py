"""Derivative-free multi-start COBYLA VQE engine using Qiskit SparsePauliOp & Statevector.

Integrates cleanly with ``bus_stop_ansatz.run_external_vqe(...)`` and
``bus_stop_bonus_notebook.py`` via the ``train_vqe(...)`` contract.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from qiskit.quantum_info import Statevector
from scipy.optimize import minimize

import bus_stop_ansatz as bsa


def _build_circuit(ansatz_type: str, params: np.ndarray, layers: int):
    """Build unpruned, measurement-free Qiskit circuit for the requested ansatz family."""
    if ansatz_type == "8q":
        return bsa.build_8q_ansatz(params, layers=layers, measure=False, prune_zeros=False)
    if ansatz_type == "12q":
        return bsa.build_12q_ansatz(params, layers=layers, measure=False, prune_zeros=False)
    if ansatz_type == "hea_linear":
        return bsa.build_hea_linear_12q(params, layers=layers, measure=False)
    raise ValueError(
        f"Unsupported ansatz_type {ansatz_type!r}. Choose '8q', '12q', or 'hea_linear'."
    )


def _num_params(ansatz_type: str, layers: int) -> int:
    if ansatz_type == "8q":
        return bsa.num_params_8q(layers)
    if ansatz_type in ("12q", "hea_linear"):
        return bsa.num_params_12q(layers)
    raise ValueError(
        f"Unsupported ansatz_type {ansatz_type!r}. Choose '8q', '12q', or 'hea_linear'."
    )


def train_vqe(
    ansatz_type: str = "8q",
    layers: int = 1,
    n_seeds: int = 10,
    base_seed: int = 42,
    maxiter: int = 2000,
    tol: float = 1e-5,
    success_tol: float = 14.01,
) -> dict[str, Any]:
    """Run multi-start derivative-free COBYLA VQE against Qiskit SparsePauliOp Hamiltonian.

    Parameters
    ----------
    ansatz_type : {"8q", "12q", "hea_linear"}, default="8q"
        Ansatz family defined in ``bus_stop_ansatz``.
    layers : int, default=1
        Number of variational layers.
    n_seeds : int, default=10
        Number of independent random parameter initializations in ``[-pi, pi]``.
    base_seed : int, default=42
        Base RNG seed for reproducibility.
    maxiter : int, default=2000
        Maximum COBYLA function evaluations per seed.
    tol : float, default=1e-5
        COBYLA convergence tolerance.
    success_tol : float, default=14.01
        Energy threshold ``<F> < success_tol`` counting a seed as converged to optimum.

    Returns
    -------
    dict[str, Any]
        Result dictionary conforming to the ``bus_stop_ansatz.run_external_vqe`` contract.
    """
    encoding = "8q" if ansatz_type == "8q" else "12q"
    n_qubits = 8 if encoding == "8q" else 12
    n_params = _num_params(ansatz_type, layers)
    opt_idx = bsa.OPTIMAL_IDX_8Q if encoding == "8q" else bsa.OPTIMAL_IDX_12Q

    # Build exact Qiskit SparsePauliOp for F(x) and cache its real computational-basis diagonal
    hamiltonian_op = bsa.build_sparse_pauli_hamiltonian(encoding)
    h_diag = np.asarray(hamiltonian_op.to_matrix(sparse=True).diagonal().real, dtype=np.float64)

    seed_results: list[dict[str, Any]] = []
    best_record: dict[str, Any] | None = None

    for s in range(n_seeds):
        seed = base_seed + s
        rng = np.random.default_rng(seed)
        x0 = rng.uniform(-math.pi, math.pi, size=n_params)

        history: list[float] = []
        best_seen = math.inf

        def objective(params_vec: np.ndarray) -> float:
            nonlocal best_seen
            qc = _build_circuit(ansatz_type, params_vec, layers)
            sv = Statevector.from_instruction(qc)
            probs = np.abs(sv.data) ** 2
            energy = float(np.dot(probs, h_diag))
            if energy < best_seen:
                best_seen = energy
            history.append(best_seen)
            return energy

        res = minimize(
            objective,
            x0,
            method="COBYLA",
            tol=tol,
            options={"maxiter": maxiter},
        )

        # Wrap angles into (-pi, pi] and verify final energy directly against SparsePauliOp
        opt_params = ((res.x + math.pi) % (2.0 * math.pi)) - math.pi
        final_qc = _build_circuit(ansatz_type, opt_params, layers)
        final_sv = Statevector.from_instruction(final_qc)
        final_energy = float(final_sv.expectation_value(hamiltonian_op).real)
        final_probs = np.abs(final_sv.data) ** 2
        opt_prob = float(final_probs[opt_idx])
        top_idx = int(np.argmax(final_probs))
        top_bits = f"{top_idx:0{n_qubits}b}"[::-1]

        record = {
            "seed": seed,
            "energy": final_energy,
            "optimal_probability": opt_prob,
            "top_bitstring": top_bits,
            "top_probability": float(final_probs[top_idx]),
            "nfev": int(res.nfev),
            "success": final_energy < success_tol,
            "params": opt_params,
            "history": history,
        }
        seed_results.append(record)
        if best_record is None or final_energy < best_record["energy"]:
            best_record = record

    assert best_record is not None
    success_count = sum(1 for r in seed_results if r["success"])
    energies = [r["energy"] for r in seed_results]

    return {
        "ansatz_type": ansatz_type,
        "layers": layers,
        "num_qubits": n_qubits,
        "num_params": n_params,
        "optimizer": "COBYLA",
        "n_seeds": n_seeds,
        "best_energy": best_record["energy"],
        "median_energy": float(np.median(energies)),
        "mean_energy": float(np.mean(energies)),
        "best_optimal_probability": best_record["optimal_probability"],
        "success_count": success_count,
        "success_rate": success_count / n_seeds,
        "best_params": best_record["params"],
        "best_seed": best_record["seed"],
        "best_history": best_record["history"],
        "seed_results": seed_results,
    }


if __name__ == "__main__":
    for atype in ("8q", "12q"):
        out = train_vqe(ansatz_type=atype, layers=1, n_seeds=10, base_seed=42)
        print(
            f"[COBYLA {atype} p=1] best <F>={out['best_energy']:.6f}, "
            f"P(x*)={out['best_optimal_probability']:.6f}, "
            f"success={out['success_count']}/{out['n_seeds']} "
            f"({100 * out['success_rate']:.0f}%)"
        )
