"""Compiles + validates the public benchmark circuits and a few random ones, prints a table.

uv run python bench/run_bench.py            # local table
uv run python bench/run_bench.py --score    # also the official score (black box, runs in subprocesses)
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pyarrow.parquet as pq
from qiskit import QuantumCircuit
from qiskit.circuit.random import random_circuit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check import check

ROOT = Path(__file__).resolve().parents[1]
COLUMNS = ["valid", "qubits", "distance", "steps", "moves", "depth", "time_s", "error"]


def public_circuits() -> dict[str, QuantumCircuit]:
    table = pq.read_table(ROOT / "pushquantum-2026-metric-hea-0.parquet")
    names, data = table.column("filename").to_pylist(), table.column("filedata").to_pylist()
    return {n: QuantumCircuit.from_qasm_str(d.decode()) for n, d in sorted(zip(names, data, strict=True))}


def random_circuits() -> dict[str, QuantumCircuit]:
    out = {}
    for seed, n in enumerate((4, 6, 8)):
        qc = random_circuit(n, depth=6, max_operands=2, seed=seed)
        qc.measure_all()
        out[f"random_{n}q_s{seed}"] = qc
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score", action="store_true")
    args = parser.parse_args()

    rows = {name: check(qc) for name, qc in (public_circuits() | random_circuits()).items()}
    width = max(map(len, rows))
    lines = [f"{'circuit':<{width}}  " + "  ".join(f"{c:>8}" for c in COLUMNS[:-1]) + "  error"]
    for name, row in rows.items():
        cells = "  ".join(f"{row.get(c, ''):>8}" for c in COLUMNS[:-1])
        lines.append(f"{name:<{width}}  {cells}  {row.get('error', '')}")
    total = sum(r.get("distance", 0) for r in rows.values())
    lines.append(f"all valid: {all(r['valid'] for r in rows.values())}   total distance: {total:.2f}")
    if args.score:
        from pushquantum_validator import score_compiler

        lines.append(f"official score (% better than naive): {score_compiler():.2f}")
    text = "\n".join(lines)
    print(text)
    with open(ROOT / "bench" / "results.md", "a", encoding="utf-8") as f:
        f.write(f"\n## {datetime.now():%Y-%m-%d %H:%M}\n```\n{text}\n```\n")


if __name__ == "__main__":
    main()
