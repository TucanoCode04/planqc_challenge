"""Builds `compiler_notebook.py`: one self-contained marimo notebook with the whole compiler, for an Aqora workspace.

The code is copied verbatim from src/pushquantum (template classes + compiler modules, in dependency order) into the
notebook's `app.setup` block; internal imports are removed and the bundled default spec is inlined.
Run: uv run python aqora/build_notebook.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "pushquantum"
OUT = Path(__file__).resolve().parent / "compiler_notebook.py"

ORDER = [
    "move.py",
    "architecture.py",
    "compiler/config.py",
    "compiler/geometry.py",
    "compiler/ir.py",
    "compiler/cost.py",
    "compiler/preprocess.py",
    "compiler/schedule.py",
    "compiler/placement.py",
    "compiler/routing.py",
    "compiler/emit.py",
    "compiler/__init__.py",
]


def body(path: Path) -> tuple[list[str], str]:
    """External import lines and the remaining code of one module (docstring and internal imports removed)."""
    text = path.read_text(encoding="utf-8")
    text = re.sub(r'^""".*?"""\n', "", text, count=1, flags=re.S)  # module docstring
    imports, code = [], []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith(("from __future__", "import pushquantum", "from pushquantum")):
            if line.rstrip().endswith("("):  # multi-line internal import
                while not lines[i].rstrip().endswith(")"):
                    i += 1
            i += 1
            continue
        if line.startswith(("import ", "from ")) and not line.startswith("from pushquantum"):
            if line.rstrip().endswith("("):
                block = [line]
                while not lines[i].rstrip().endswith(")"):
                    i += 1
                    block.append(lines[i])
                imports.append("\n".join(block))
            else:
                imports.append(line)
            i += 1
            continue
        if line.startswith("if TYPE_CHECKING:"):
            i += 1
            while i < len(lines) and (lines[i].startswith("    ") or not lines[i].strip()):
                i += 1
            continue
        code.append(line)
        i += 1
    return imports, "\n".join(code).strip("\n")


def main() -> None:
    all_imports: list[str] = []
    chunks: list[str] = []
    for rel in ORDER:
        imports, code = body(SRC / rel)
        for imp in imports:
            if imp not in all_imports:
                all_imports.append(imp)
        if rel == "architecture.py":
            spec = json.loads((SRC / "architectures" / "default.json").read_text(encoding="utf-8"))
            code = code.replace(
                'return json.loads(resources.files("pushquantum").joinpath("architectures/default.json").read_text("utf-8"))',
                f"return {spec!r}",
            )
        if rel == "compiler/__init__.py":
            code = code.replace("def compile_circuit(", "def tucano_compile(")
        chunks.append(f"# ---- {rel} " + "-" * max(0, 100 - len(rel)) + "\n" + code)

    all_imports = [i for i in all_imports if "importlib import resources" not in i]
    all_imports.append("from qiskit.circuit import InstructionSet")  # was a type-checking-only import
    all_imports = [i if "TYPE_CHECKING" not in i else i.replace(", TYPE_CHECKING", "").replace("TYPE_CHECKING, ", "")
                   for i in all_imports]  # fmt: skip
    setup = "\n".join(all_imports) + "\n\n" + "\n\n\n".join(chunks)
    setup = "\n".join(("    " + ln) if ln.strip() else "" for ln in setup.splitlines())

    notebook = f'''# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "networkx",
#     "numpy",
#     "qiskit==2.6.2",
#     "scipy",
# ]
# ///
"""Team QubiTO-gether - zoned neutral-atom compiler (planqc x PushQuantum 2026).

Self-contained: the template's Move instruction and Architecture, and our whole compiler, live in the setup block.
`compile_circuit(circuit, architecture_spec)` is the entry point. Source and documentation: submission/ in our repo.
"""

from __future__ import annotations

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")

with app.setup:
{setup}


@app.function
def compile_circuit(circuit: QuantumCircuit, architecture_spec: ArchitectureSpec) -> tuple[QuantumCircuit, Architecture]:
    """Returns the compiled circuit and the `Architecture` (zones of `architecture_spec`, initial loading)."""
    return tucano_compile(circuit, architecture_spec)


@app.cell
def _():
    import marimo as mo

    mo.md(
        """
    # Team QubiTO-gether — zoned neutral-atom compiler

    `compile_circuit(circuit, architecture_spec)` returns the compiled circuit (rx, rz, cz + AOD moves) and the
    initial loading. Every official benchmark circuit is valid; official score 77.7 % better than the naive compiler.

    Pipeline: native gates and ZZ grouping by edge colouring → blocks of commuting CZs → placement (snake / 2D layout,
    shared shifts, interleaved rows, lookahead) → merged AOD moves → best of ~150 variants by an exact copy of the metric.
    """
    )
    return


@app.cell
def _():
    _qc = QuantumCircuit(3)
    _qc.h(0)
    _qc.cx(0, 1)
    _qc.cx(1, 2)
    _qc.measure_all()
    _compiled, _arch = compile_circuit(_qc, None)
    _compiled.count_ops(), _arch.initial_loading
    return


if __name__ == "__main__":
    app.run()
'''
    OUT.write_text(notebook, encoding="utf-8")
    print("written", OUT, f"({len(notebook.splitlines())} lines)")


if __name__ == "__main__":
    main()
