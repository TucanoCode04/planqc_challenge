# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "aqora[pyarrow]==0.39.0",
#     "marimo",
#     "polars==2.1.0",
#     "qiskit==2.6.2",
# ]
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")

with app.setup:
    import itertools
    import json
    from collections.abc import Sequence
    from dataclasses import dataclass
    from numbers import Integral
    from pathlib import Path
    from typing import Any

    import marimo as mo
    import polars as pl
    from aqora.pyarrow import dataset
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit import Instruction, InstructionSet, Qubit

    Site = tuple[int, int]
    ArchitectureSpec = str | Path | dict[str, Any]


@app.cell
def _():
    ghz = QuantumCircuit(3)
    ghz.h(0)
    ghz.cx(0, 1)
    ghz.cx(1, 2)
    ghz.measure_all()

    compiled, architecture = compile_circuit(ghz, default_spec())
    return architecture, compiled, ghz


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Architecture

    The default architecture is returned by `default_spec()` below:

    | Zone       | x     | y       | Allowed gates |
    |------------|-------|---------|---------------|
    | storage    | 0 – 9 | 0 – 9   | `rx`, `rz`    |
    | entangling | 0 – 9 | 10 – 19 | `cz`          |

    `measure`, `barrier` and `move` are allowed anywhere.

    **Geometry.** Row and column indices are shared by all zones, so `cols=(0, 1)` covers columns 0 and 1 in
    storage and in the entangling zone alike. Physical positions come from each zone's geometry:

    - **Storage:** sites are evenly spaced, `spacing` 1.0 apart.
    - **Entangling:** columns form interaction pairs (0, 1), (2, 3), …. The two sites of a pair are 0.3 apart
      and pairs repeat every 2.0, so columns sit at x = 0, 0.3, 2, 2.3, 4, 4.3, …. In the spec this is
      `"pairs": {"axis": "x", "distance": 0.3}`.

    Two consequences follow from the pairs:

    - **`cz` only act within a pair:** both atoms must be on the same row, in columns 2k and 2k+1.

    The initial loading maps each qubit to its starting `(x, y)` site. Every site must be in storage; the
    validator checks this.
    """)
    return


@app.function
def default_spec() -> dict[str, Any]:
    """The default architecture (`architectures/default.json` in the `pushquantum` package)."""
    return {
        "loading_zone": "storage",
        "zones": {
            "storage": {
                "x": [0, 9],
                "y": [0, 9],
                "spacing": 1.0,
                "operations": ["rx", "rz"],
            },
            "entangling": {
                "x": [0, 9],
                "y": [10, 19],
                "spacing": 1.0,
                "pairs": {"axis": "x", "distance": 0.3},
                "operations": ["cz", "rzz"],
            },
        },
    }


@app.class_definition
@dataclass(frozen=True)
class Zone:
    name: str
    x: tuple[int, int]
    y: tuple[int, int]
    operations: frozenset[str]
    spacing: float = 1.0
    pair_axis: str | None = None
    pair_distance: float | None = None


@app.function
def load_spec(spec: ArchitectureSpec | None = None) -> dict[str, Any]:
    """A spec from a path, a dict, or (if `None`) the default."""
    if spec is None:
        return default_spec()
    if isinstance(spec, dict):
        return spec
    return json.loads(Path(spec).read_text(encoding="utf-8"))


@app.function
def parse_zone(name: str, z: dict[str, Any]) -> Zone:
    pairs = z.get("pairs")
    return Zone(
        name=name,
        x=(int(z["x"][0]), int(z["x"][1])),
        y=(int(z["y"][0]), int(z["y"][1])),
        operations=frozenset(z.get("operations", ())),
        spacing=float(z.get("spacing", 1.0)),
        pair_axis=pairs["axis"] if pairs else None,
        pair_distance=float(pairs["distance"]) if pairs else None,
    )


@app.class_definition
class Architecture:
    """The zones of `spec` and `initial_loading` (qubit `0..n-1` -> `(x, y)` site in the loading zone)."""

    def __init__(self, initial_loading: dict[int, Site], spec: ArchitectureSpec | None = None) -> None:
        data = load_spec(spec)
        self.zones: dict[str, Zone] = {name: parse_zone(name, z) for name, z in data["zones"].items()}
        self.loading_zone: str = data.get("loading_zone", "storage")
        self.initial_loading: dict[int, Site] = dict(initial_loading)


@app.cell
def _():
    arch = Architecture({0: (0, 0), 1: (1, 0)})  # or Architecture(loading, spec="my_arch.json")
    {
        "zones": arch.zones,  # zone definitions (bounds, allowed gates, geometry)
        "initial_loading": arch.initial_loading,  # qubit -> starting site
    }
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Moves

    A move uses one AOD pair. It selects `rows` (y) and `cols` (x), and every atom on a rows × cols
    intersection moves to the matching `new_rows` × `new_cols` intersection. You also list the qubits being
    moved. A `move` leaves the quantum state unchanged, so its definition is the identity.
    """)
    return


@app.function
def as_lines(values: int | Sequence[int], name: str) -> tuple[int, ...]:
    items = [values] if isinstance(values, Integral) else list(values)
    if not items:
        msg = f"`{name}` must not be empty."
        raise ValueError(msg)
    if not all(isinstance(v, Integral) for v in items):
        msg = f"`{name}` must contain integers, got {items!r}."
        raise ValueError(msg)
    result = tuple(int(v) for v in items)
    if len(set(result)) != len(result):
        msg = f"`{name}` must not contain duplicates, got {result!r}."
        raise ValueError(msg)
    return result


@app.function
def check_lines(start: tuple[int, ...], end: tuple[int, ...], name: str) -> None:
    if len(end) > len(start):
        msg = f"`new_{name}` cannot have more lines than `{name}`, got {len(end)} and {len(start)}."
        raise ValueError(msg)
    contiguous = tuple(range(start[0], start[0] + len(start)))
    if start != contiguous:
        msg = f"Picked-up {name} {start} must be contiguous and ascending, e.g. {contiguous}."
        raise ValueError(msg)
    if any(b <= a for a, b in itertools.pairwise(end)):
        msg = f"Target {name} {end} must be ascending: the order of {name} is preserved."
        raise ValueError(msg)


@app.class_definition
class Move(Instruction):
    """Params are `(rows, cols, new_rows, new_cols)`, tuples of ints; rows are y, columns are x."""

    def __init__(
        self,
        rows: int | Sequence[int],
        cols: int | Sequence[int],
        new_rows: int | Sequence[int],
        new_cols: int | Sequence[int],
        num_qubits: int,
    ) -> None:
        names = ("rows", "cols", "new_rows", "new_cols")
        lines = [as_lines(v, n) for v, n in zip((rows, cols, new_rows, new_cols), names, strict=True)]
        check_lines(lines[0], lines[2], "rows")
        check_lines(lines[1], lines[3], "cols")
        if num_qubits < 1:
            msg = "A move must act on at least one qubit."
            raise ValueError(msg)
        super().__init__("move", num_qubits, 0, lines)

    rows = property(lambda self: self.params[0])
    cols = property(lambda self: self.params[1])
    new_rows = property(lambda self: self.params[2])
    new_cols = property(lambda self: self.params[3])

    def validate_parameter(self, parameter: object) -> tuple[int, ...]:
        if isinstance(parameter, tuple) and all(isinstance(v, Integral) for v in parameter):
            return parameter
        msg = f"Move params must be tuples of ints, got {parameter!r}."
        raise TypeError(msg)

    def _define(self) -> None:
        self.definition = QuantumCircuit(self.num_qubits, name=self.name)


@app.function
def circuit_method(method):
    """Attaches `method` to `QuantumCircuit` under its own name, e.g. `qc.move(...)`."""
    setattr(QuantumCircuit, method.__name__, method)
    return method


@app.function
@circuit_method
def move(
    self: QuantumCircuit,
    qubits: int | Qubit | Sequence[int | Qubit],
    rows: int | Sequence[int],
    cols: int | Sequence[int],
    new_rows: int | Sequence[int],
    new_cols: int | Sequence[int],
) -> InstructionSet:
    """Appends a `Move` of the atoms at `rows` x `cols`, which must be exactly `qubits`."""
    qubit_list = [qubits] if isinstance(qubits, (int, Qubit)) else list(qubits)
    return self.append(Move(rows, cols, new_rows, new_cols, len(qubit_list)), qubit_list)


@app.cell
def _():
    qc = QuantumCircuit(2)
    qc.rx(0.5, 0)
    qc.move([0, 1], rows=0, cols=(0, 1), new_rows=10, new_cols=(0, 1))  # into the entangling zone
    qc.cz(0, 1)
    qc.move([0, 1], rows=10, cols=(0, 1), new_rows=0, new_cols=(0, 1))  # back to storage
    qc.draw("text")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Rows and columns follow these rules, each axis on its own:

    - **Pickup** (`rows`, `cols`): must be **contiguous** and ascending, e.g. `cols=(0, 1, 2, 3)`.
    - **Drop-off** (`new_rows`, `new_cols`): the exact final coordinates of the moved atoms, ascending.
    - **Proportional spacing** (a warning): every gap between drop-off lines should equal the matching gap
      between the source lines, times one common factor. With all picked-up lines kept, the source is
      contiguous, so the drop-off should be evenly spaced, e.g. `(0, 1, 2, 3) -> (0, 2, 4, 6)`. Moves that
      break this are still allowed; the validator logs a warning, counts it and doubles that move's time.
    - **Dropping empty lines**: the drop-off may list *fewer* lines than the pickup. Selected rows/columns
      with no atom on them are then removed, and the remaining occupied lines must map one-to-one and
      proportionally onto the drop-off lines. With atoms on columns `0, 1, 4` of `range(5)` (gaps 1, 3):
      - `(0, 2, 8)` works (gaps ×2);
      - `(0, 1, 2)` and `(0, 2, 4)` are allowed but logged, because the gaps don't scale evenly.
    """)
    return


@app.cell
def _():
    spread = QuantumCircuit(4)
    # pick up 4 adjacent atoms on row 0 and spread them out on row 10
    spread.move([0, 1, 2, 3], rows=0, cols=(0, 1, 2, 3), new_rows=10, new_cols=(0, 2, 4, 6))
    # reverse it: atoms on columns 0, 2, 4, 6 of row 10 -> columns 0..3 of row 0 (gaps x1/2)
    spread.move([0, 1, 2, 3], rows=10, cols=range(7), new_rows=0, new_cols=(0, 1, 2, 3))
    spread.draw("text")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    A move is rejected when:

    - **the pickup isn't contiguous**, e.g. `cols=(0, 2)`;
    - **the drop-off has more lines than the pickup**, or, after dropping empty lines, the number of occupied
      lines differs from the number of drop-off lines;
    - **the drop-off reorders lines**, e.g. the permutation `(0, 1) -> (2, 0)`;
    - **the atoms at the intersections are not exactly the listed qubits**: an unlisted atom would be picked
      up (a ghost), or a listed qubit isn't at an intersection;
    - **it drops an atom onto an occupied site**, i.e. one held by an atom that isn't part of this move;
    - **a target lies outside every zone**.

    The first three only depend on the lines, so `Move` already rejects them when you build the circuit. The
    others depend on where the atoms are, so only the validator catches them.
    """)
    return


@app.cell
def _():
    _rejected = {
        "the pickup isn't contiguous": dict(rows=0, cols=(0, 2), new_rows=10, new_cols=(0, 2)),
        "the drop-off has more lines than the pickup": dict(rows=0, cols=(0, 1), new_rows=10, new_cols=(0, 1, 2)),
        "the drop-off reorders lines": dict(rows=0, cols=(0, 1), new_rows=10, new_cols=(2, 0)),
    }
    _errors = []
    for _rule, _lines in _rejected.items():
        try:
            Move(**_lines, num_qubits=2)
        except ValueError as _error:
            _errors.append({"rule": _rule, "error": str(_error)})
    mo.ui.table(_errors, selection=None)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Template

    Write your compilation in `compile_circuit(circuit, architecture_spec)` below. It returns the compiled
    circuit and an `Architecture(initial_loading, spec=architecture_spec)` holding your initial loading.
    """)
    return


@app.function
def compile_circuit(
    circuit: QuantumCircuit, architecture_spec: ArchitectureSpec
) -> tuple[QuantumCircuit, Architecture]:
    """Returns the compiled circuit and the `Architecture` (zones of `architecture_spec`, your initial loading)."""
    # TODO: implement your compilation here.
    num_qubits = circuit.num_qubits
    # Transpile to native gates (rx, rz, cz) with no optimization. This is a placeholder for your
    # transpilation logic.
    native = transpile(circuit, basis_gates=["rx", "rz", "cz"], optimization_level=0)

    compiled = native.copy_empty_like()
    # for _instruction in native.data:
    # add movements and gates here
    # pass

    # Determine your initial loading; this naive one puts qubit i on column 0, row i of storage.
    naive_loading = {i: (0, i) for i in range(num_qubits)}
    architecture = Architecture(naive_loading, spec=architecture_spec)
    return compiled, architecture


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Try it

    The placeholder above only transpiles and returns an empty circuit; fill it in and the output below
    updates.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## HEA metric benchmark

    Every circuit of the [`planqc/pushquantum-2026-metric-hea`](https://aqora.io/datasets/planqc/pushquantum-2026-metric-hea)
    dataset (OpenQASM 2.0) goes through `compile_circuit` on the default architecture. The table below
    updates whenever you change `compile_circuit`.
    """)
    return


@app.cell
def _():
    bench_dataset = (
        pl.scan_pyarrow_dataset(dataset("planqc/pushquantum-2026-metric-hea", "v0.1.0"))
        .collect()
        .sort("filename")
    )
    return (bench_dataset,)


@app.cell
def _(bench_dataset):
    bench_results = []
    for _filename, _filedata in bench_dataset.iter_rows():
        _circuit = QuantumCircuit.from_qasm_str(_filedata.decode())
        try:
            _compiled, _architecture = compile_circuit(_circuit, default_spec())
            _error = None
        except Exception as _exception:  # record it so one failing circuit doesn't hide the others
            _compiled, _architecture, _error = None, None, f"{type(_exception).__name__}: {_exception}"
        bench_results.append(
            {
                "filename": _filename,
                "circuit": _circuit,
                "compiled": _compiled,
                "architecture": _architecture,
                "error": _error,
            }
        )
    return (bench_results,)


@app.cell
def _(bench_results):
    _rows = []
    for _result in bench_results:
        _circuit, _compiled = _result["circuit"], _result["compiled"]
        _row = {
            "circuit": _result["filename"],
            "qubits": _circuit.num_qubits,
            "input gates": _circuit.size(),
            "input depth": _circuit.depth(),
            "compiled gates": None,
            "moves": None,
            "cz": None,
            "error": _result["error"],
        }
        if _compiled is not None:
            _ops = _compiled.count_ops()
            _row |= {"compiled gates": _compiled.size(), "moves": _ops.get("move", 0), "cz": _ops.get("cz", 0)}
        _rows.append(_row)
    pl.DataFrame(_rows)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # pushquantum

    A minimal template for compiling circuits for a zoned neutral-atom architecture. It includes:

    - a custom Qiskit `move` instruction for AOD row/column moves;
    - an `Architecture` that loads its zones from a spec, together with the initial positions of the qubits;
    - a template entry point, `compile_circuit(circuit, architecture_spec) -> (circuit, architecture)`.

    Your compiled circuit is checked by a separate validator package (`pushquantum-validator`). It checks that
    every move and gate follows the rules below, and that the circuit is equivalent to the input.

    Open this notebook with `marimo edit --sandbox notebook.py`: the dependencies are declared inline.
    """)
    return


@app.cell
def _(architecture, compiled, ghz):
    mo.vstack(
        [
            mo.md("**Input**"),
            ghz.draw("text"),
            mo.md("**Compiled**"),
            compiled.draw("text"),
            mo.md("**Initial loading** (qubit → `(x, y)`)"),
            architecture.initial_loading,
        ]
    )
    return


if __name__ == "__main__":
    app.run()
