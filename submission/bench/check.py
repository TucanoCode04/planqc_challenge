"""Checks one compiled circuit with the organizers' validator, used as a black box (public functions only)."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# The validator is not part of this repository. Looked up in: $PQ_VALIDATOR_DIR (the folder that CONTAINS
# `pushquantum_validator/`), this project folder, or the sibling local workspace `../Tucano`.
_ROOT = Path(__file__).resolve().parents[1]
for _candidate in (os.environ.get("PQ_VALIDATOR_DIR"), _ROOT, _ROOT.parent / "Tucano"):
    if _candidate and (Path(_candidate) / "pushquantum_validator").is_dir():
        sys.path.insert(0, str(_candidate))
        break

from pushquantum_validator import (  # noqa: E402  (needs the path set up above)
    VerificationError,
    compute_metrics,
    strip_moves,
    verify_equivalence,
    verify_movement,
)
from qiskit import QuantumCircuit  # noqa: E402

from pushquantum.compiler import compile_circuit  # noqa: E402
from pushquantum.compiler.config import DEFAULT, Config  # noqa: E402


def check(reference: QuantumCircuit, config: Config = DEFAULT) -> dict:
    row: dict = {"qubits": reference.num_qubits}
    start = time.perf_counter()
    try:
        compiled, arch = compile_circuit(reference, None, config)
    except Exception as e:  # report and keep going: one crash must not hide the other circuits
        return row | {"valid": False, "error": f"compile: {type(e).__name__}: {e}"}
    row["time_s"] = round(time.perf_counter() - start, 3)
    try:
        movement = verify_movement(compiled, arch.initial_loading)
        gates = strip_moves(compiled)
        verify_equivalence(compiled, reference, gates)
        m = compute_metrics(compiled, movement, gates)
    except VerificationError as e:
        return row | {"valid": False, "error": f"{type(e).__name__}: {str(e)[:300]}"}
    return row | {
        "valid": True,
        "distance": round(m.move_distance, 3),
        "steps": m.move_steps,
        "moves": m.num_moves,
        "depth": m.depth,
        "ops": m.num_operations,
        "move_time": round(m.move_time, 3),
    }
