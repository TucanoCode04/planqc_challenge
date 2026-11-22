# /// script
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
    import itertools
    from collections.abc import Sequence
    from numbers import Integral
    from typing import TYPE_CHECKING
    from qiskit.circuit import Instruction, QuantumCircuit, Qubit
    import json
    from dataclasses import dataclass
    from pathlib import Path
    from typing import Any
    from typing import Literal
    import math
    from functools import cache, cached_property
    from dataclasses import dataclass, field
    from qiskit.circuit import ParameterExpression
    from qiskit import QuantumCircuit
    from collections import Counter
    from qiskit import AncillaRegister, QuantumCircuit, transpile
    import numpy as np
    from qiskit.circuit.library import RXGate, RZGate
    from qiskit.synthesis import OneQubitEulerDecomposer
    import random
    from collections import Counter, deque
    import networkx as nx
    from scipy.optimize import linear_sum_assignment
    from dataclasses import replace
    from qiskit.circuit import InstructionSet

    # ---- move.py ---------------------------------------------------------------------------------------------
    Site = tuple[int, int]


    def _as_lines(values: int | Sequence[int], name: str) -> tuple[int, ...]:
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


    def _check_lines(start: tuple[int, ...], end: tuple[int, ...], name: str) -> None:
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
            lines = [_as_lines(v, n) for v, n in zip((rows, cols, new_rows, new_cols), names, strict=True)]
            _check_lines(lines[0], lines[2], "rows")
            _check_lines(lines[1], lines[3], "cols")
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
            self.definition = QuantumCircuit(self.num_qubits, name=self.name)  # identity


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


    QuantumCircuit.move = move  # type: ignore[attr-defined]


    # ---- architecture.py -------------------------------------------------------------------------------------
    ArchitectureSpec = str | Path | dict[str, Any]


    @dataclass(frozen=True)
    class Zone:
        name: str
        x: tuple[int, int]
        y: tuple[int, int]
        operations: frozenset[str]
        spacing: float = 1.0
        pair_axis: str | None = None
        pair_distance: float | None = None


    def load_spec(spec: ArchitectureSpec | None = None) -> dict[str, Any]:
        """A spec from a path, a dict, or (if `None`) the bundled default."""
        if spec is None:
            return {'loading_zone': 'storage', 'zones': {'storage': {'x': [0, 9], 'y': [0, 9], 'spacing': 1.0, 'operations': ['rx', 'rz']}, 'entangling': {'x': [0, 9], 'y': [10, 19], 'spacing': 1.0, 'pairs': {'axis': 'x', 'distance': 0.3}, 'operations': ['cz']}}}
        if isinstance(spec, dict):
            return spec
        return json.loads(Path(spec).read_text(encoding="utf-8"))


    def _parse_zone(name: str, z: dict[str, Any]) -> Zone:
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


    class Architecture:
        """The zones of `spec` and `initial_loading` (qubit `0..n-1` -> `(x, y)` site in the loading zone)."""

        def __init__(self, initial_loading: dict[int, Site], spec: ArchitectureSpec | None = None) -> None:
            data = load_spec(spec)
            self.zones: dict[str, Zone] = {name: _parse_zone(name, z) for name, z in data["zones"].items()}
            self.loading_zone: str = data.get("loading_zone", "storage")
            self.initial_loading: dict[int, Site] = dict(initial_loading)


    # ---- compiler/config.py ----------------------------------------------------------------------------------
    DistModel = Literal["grid", "physical"]
    Colouring = Literal["greedy", "cpsat"]
    HomeMethod = Literal["bfs", "snake", "grid"]
    TargetMethod = Literal["hungarian", "shift", "grid"]
    Rank = Literal["distance_first", "balanced"]


    @dataclass(frozen=True)
    class Config:
        # "physical": the validator measures real positions (entangling cols at 0, 0.3, 2, 2.3, ...), checked by experiment.
        # "grid" (1 per cell) was the first assumption in v0; kept only to compare.
        dist_model: DistModel = "physical"

        # v1a: many atoms per `move`. Scored distance = sum over moves of the longest trip in each move.
        merge_moves: bool = True

        # v0 plan, NOT USED anymore (kept for the record):
        # - row_merge: merge only atoms on the same row. Replaced by merge_moves, which merges any legal rectangle.
        # - parallel_moves: put independent moves in the same parallel step. Experiments showed the validator does group
        #   them into one step, but the scored distance still adds every move, so steps only change time, not the score.
        # row_merge: bool = False
        # parallel_moves: bool = False

        # v1b: after a pulse, atoms that are also in the next pulse stay in the entangling zone (only shift sideways).
        reuse: bool = False

        # v2a: initial loading. "bfs" = v0 order from the busiest qubit; "snake" = Cuthill-McKee order, snake rows.
        home: HomeMethod = "snake"

        # v2b: where each pulse's pairs go. "hungarian" = v0 matching (min sum of trips, ignores merging);
        # "shift" = same column shift for all side-by-side pairs so the whole round lifts in one move.
        targets: TargetMethod = "shift"

        # v2b: after a pulse, an atom parks next to its next CZ partner (if a free storage site is there) instead of
        # going back to its fixed home, so the pair goes up together in one move next time.
        dynamic_return: bool = True

        # v2c (depth): each storage visit writes Rz·Rx instead of Rz·Rx·Rz; the trailing Rz slides to the next visit.
        fuse_rz: bool = True

        # v3a: ZZ rotations (cx rz cx / rzz) commute; regroup each run by edge colour: H - CZ - Rx - CZ - H per colour,
        # so one pair of pulses serves a whole colour instead of a single pair. Busier qubit = control.
        group_zz: bool = True
        # v3b: with group_zz, copy a hub qubit (Max-Cut reference) onto extra atoms and spread its ZZ terms over the copies
        fanout: bool = True

        # v4: also try a 2D layout (simulated annealing: partners become left/right or up/down neighbours),
        # colours chosen by grid position, vertical pairs interleaved onto one entangling row
        grid2d: bool = True

        # v5: Hungarian targets also weigh the distance to each atom's next partner (lookahead weight; 0 = off)
        lookahead: float = 0.0
        # the lookahead weight pick_best tries next to 0
        lookahead_try: float = 0.3

        # compile every combination of reuse x home x targets x dynamic_return, keep the cheapest
        # (distance, then depth), measured by our own copy of the metric (cost.py)
        pick_best: bool = True
        # how pick_best compares variants: distance first (ties: depth), or relative distance + relative depth
        rank: Rank = "balanced"
        # with rank="balanced": weight of relative depth vs relative distance (measured with the official score:
        # distance_first 77.18, 0.3 -> 77.30, 1.0 -> 76.55)
        depth_weight: float = 0.3

        colouring: Colouring = "greedy"
        seed: int = 0


    DEFAULT = Config()


    # ---- compiler/geometry.py --------------------------------------------------------------------------------
    PairSite = tuple[int, int]  # (row, k): the k-th pair of that row


    def _contains(zone: Zone, site: Site) -> bool:
        x, y = site
        return zone.x[0] <= x <= zone.x[1] and zone.y[0] <= y <= zone.y[1]


    @dataclass(frozen=True)
    class Device:
        storage: Zone
        entangling: Zone

        @classmethod
        def from_spec(cls, spec: ArchitectureSpec | None = None) -> Device:
            arch = Architecture({}, spec)
            storage = arch.zones[arch.loading_zone]
            entangling = next(z for z in arch.zones.values() if z.pair_axis is not None)
            if entangling.pair_axis != "x":
                msg = f"Only pairs along x are supported, got {entangling.pair_axis!r}."
                raise NotImplementedError(msg)
            return cls(storage, entangling)

        def zone_of(self, site: Site) -> str | None:
            if _contains(self.storage, site):
                return "storage"
            if _contains(self.entangling, site):
                return "entangling"
            return None

        def in_storage(self, site: Site) -> bool:
            return _contains(self.storage, site)

        def in_entangling(self, site: Site) -> bool:
            return _contains(self.entangling, site)

        @cached_property
        def pairs_per_row(self) -> int:
            return (self.entangling.x[1] - self.entangling.x[0] + 1) // 2

        def _rows_to_entangling(self, row: int) -> int:
            lo, hi = self.entangling.y
            return 0 if lo <= row <= hi else min(abs(row - lo), abs(row - hi))

        @cached_property
        def storage_sites(self) -> list[Site]:
            """All storage sites, closest to the entangling zone first."""
            sites = [
                (x, y)
                for y in range(self.storage.y[0], self.storage.y[1] + 1)
                for x in range(self.storage.x[0], self.storage.x[1] + 1)
            ]
            return sorted(sites, key=lambda s: (self._rows_to_entangling(s[1]), s[0]))

        @cached_property
        def pair_sites(self) -> list[PairSite]:
            """All pair sites, rows closest to storage first."""
            lo, hi = self.entangling.y
            rows = sorted(range(lo, hi + 1), key=lambda r: min(abs(r - y) for y in self.storage.y))
            return [(row, k) for row in rows for k in range(self.pairs_per_row)]

        def pair_slots(self, pair: PairSite) -> tuple[Site, Site]:
            row, k = pair
            left = self.entangling.x[0] + 2 * k
            return (left, row), (left + 1, row)

        def pair_of(self, site: Site) -> PairSite | None:
            if not self.in_entangling(site):
                return None
            k = (site[0] - self.entangling.x[0]) // 2
            return (site[1], k) if k < self.pairs_per_row else None

        def is_pair(self, a: Site, b: Site) -> bool:
            pa = self.pair_of(a)
            return pa is not None and pa == self.pair_of(b) and a != b

        def position(self, site: Site) -> tuple[float, float]:
            """Physical (x, y). Entangling columns: pairs every 2·spacing, the two sites of a pair `pair_distance` apart."""
            x, y = site
            if self.in_entangling(site):
                z = self.entangling
                offset = x - z.x[0]
                px = z.x[0] * z.spacing + (offset // 2) * 2 * z.spacing + (offset % 2) * (z.pair_distance or 0.0)
                return px, y * z.spacing
            return x * self.storage.spacing, y * self.storage.spacing

        def dist(self, a: Site, b: Site, model: DistModel = "physical") -> float:
            if model == "grid":
                return math.hypot(a[0] - b[0], a[1] - b[1])
            (ax, ay), (bx, by) = self.position(a), self.position(b)
            return math.hypot(ax - bx, ay - by)


    @cache
    def default_device() -> Device:
        return Device.from_spec(None)


    # ---- compiler/ir.py --------------------------------------------------------------------------------------
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


    # ---- compiler/cost.py ------------------------------------------------------------------------------------
    def move_distance(compiled: QuantumCircuit, loading: Layout, device: Device) -> float:
        pos = dict(loading)
        total = 0.0
        for inst in compiled.data:
            if inst.operation.name != "move":
                continue
            qubits = [compiled.find_bit(q).index for q in inst.qubits]
            _, _, new_rows, new_cols = inst.operation.params
            rows = sorted({pos[q][1] for q in qubits})
            cols = sorted({pos[q][0] for q in qubits})
            row_map, col_map = dict(zip(rows, new_rows, strict=True)), dict(zip(cols, new_cols, strict=True))
            longest = 0.0
            for q in qubits:
                x, y = pos[q]
                pos[q] = (col_map[x], row_map[y])
                longest = max(longest, device.dist((x, y), pos[q]))
            total += longest
        return total


    # ---- compiler/preprocess.py ------------------------------------------------------------------------------
    NATIVE = ["rx", "rz", "cz"]


    def split_measurements(circuit: QuantumCircuit) -> tuple[QuantumCircuit, list[tuple[int, int]]]:
        """Removes barriers and final measurements; returns the body and the (qubit, clbit) measurements."""
        body = circuit.copy_empty_like()
        measurements: list[tuple[int, int]] = []
        measured: set[int] = set()
        for inst in circuit.data:
            name = inst.operation.name
            qubits = [circuit.find_bit(q).index for q in inst.qubits]
            if name == "barrier":
                continue
            if name == "measure":
                measurements.append((qubits[0], circuit.find_bit(inst.clbits[0]).index))
                measured.add(qubits[0])
                continue
            if measured.intersection(qubits):
                msg = "Only final measurements are supported."
                raise NotImplementedError(msg)
            body.append(inst.operation, inst.qubits, inst.clbits)
        return body, measurements


    def to_native(body: QuantumCircuit) -> QuantumCircuit:
        """Native gates only. Level 1 also fuses runs of single-qubit gates and cancels adjacent inverse gates."""
        return transpile(body, basis_gates=NATIVE, optimization_level=1)


    _DIAGONAL_1Q = {"rz", "p", "u1", "z", "s", "sdg", "t", "tdg"}


    def _colour_edges(edges: list[tuple[int, int]]) -> list[list[tuple[int, int]]]:
        """Greedy edge colouring, busiest edges first: each colour is a set of disjoint pairs (one pair of CZ pulses)."""
        degree = Counter(q for e in edges for q in e)
        colours: list[list[tuple[int, int]]] = []
        used: list[set[int]] = []
        for a, b in sorted(edges, key=lambda e: (-max(degree[e[0]], degree[e[1]]), -min(degree[e[0]], degree[e[1]]))):
            for i, qs in enumerate(used):
                if a not in qs and b not in qs:
                    colours[i].append((a, b))
                    qs.update((a, b))
                    break
            else:
                colours.append([(a, b)])
                used.append({a, b})
        return colours


    Segment = tuple[str, object]  # ("op", (operation, qubits)) or ("zz", (zz terms, diagonal ops))


    def _segments(body: QuantumCircuit) -> list[Segment]:
        """Splits the circuit into plain operations and runs of commuting diagonal ops (ZZ terms, diagonal gates)."""
        data = [(inst.operation, [body.find_bit(q).index for q in inst.qubits]) for inst in body.data]
        segments: list[Segment] = []
        zz: dict[tuple[int, int], float] = {}
        diag: list[tuple[object, list[int]]] = []

        def close() -> None:
            nonlocal zz, diag
            if zz or diag:
                segments.append(("zz", (zz, diag)))
            zz, diag = {}, []

        i = 0
        while i < len(data):
            op, qs = data[i]
            name = op.name
            if (
                name == "cx"
                and i + 2 < len(data)
                and data[i + 1][0].name == "rz"
                and data[i + 1][1] == [qs[1]]
                and data[i + 2][0].name == "cx"
                and data[i + 2][1] == qs
            ):
                key = tuple(sorted(qs))
                zz[key] = zz.get(key, 0.0) + data[i + 1][0].params[0]
                i += 3
                continue
            if name == "rzz":
                key = tuple(sorted(qs))
                zz[key] = zz.get(key, 0.0) + op.params[0]
            elif name in _DIAGONAL_1Q or name == "cz":
                diag.append((op, qs))
            else:
                close()
                segments.append(("op", (op, qs)))
            i += 1
        close()
        return segments


    def _fanout_plan(zz: dict[tuple[int, int], float]) -> tuple[int, int] | None:
        """(hub, number of copies) if one qubit has far more ZZ partners than the others, else None."""
        if not zz:
            return None
        degree = Counter(q for e in zz for q in e)
        hub, d_hub = degree.most_common(1)[0]
        others = Counter(q for e in zz if hub not in e for q in e)
        load = max(2, max(others.values(), default=0))
        if d_hub <= load + 1:
            return None
        return hub, math.ceil(d_hub / load) - 1


    def _copy_rounds(hub: int, copies: list[int]) -> list[list[tuple[int, int]]]:
        """Doubling tree of CX: every qubit that already holds the hub's value copies it to one new qubit per round."""
        holders, todo, rounds = [hub], list(copies), []
        while todo:
            rnd = []
            for src in list(holders):
                if todo:
                    dst = todo.pop(0)
                    rnd.append((src, dst))
                    holders.append(dst)
            rounds.append(rnd)
        return rounds


    def _geo_colours(edges: list[tuple[int, int]], layout: dict[int, tuple[int, int]]) -> list[list[tuple[int, int]]]:
        """v4 (2D): colour pairs by where they sit in the grid layout. Horizontal neighbours with the left one on an even /
        odd column, vertical neighbours with the upper one on an even / odd row: each class is automatically a set of
        disjoint pairs and is lifted with few merged moves. Remaining pairs: greedy colouring."""
        classes: dict[str, list[tuple[int, int]]] = {"h0": [], "h1": [], "v0": [], "v1": []}
        rest: list[tuple[int, int]] = []
        for a, b in edges:
            (xa, ya), (xb, yb) = layout[a], layout[b]
            if ya == yb and abs(xa - xb) == 1:
                classes[f"h{min(xa, xb) % 2}"].append((a, b))
            elif xa == xb and abs(ya - yb) == 1:
                classes[f"v{max(ya, yb) % 2}"].append((a, b))
            else:
                rest.append((a, b))
        return [c for c in classes.values() if c] + _colour_edges(rest)


    def _emit_zz(
        out: QuantumCircuit, zz: dict[tuple[int, int], float], layout: dict[int, tuple[int, int]] | None = None
    ) -> None:
        degree = Counter(q for e in zz for q in e)
        colours = _geo_colours(list(zz), layout) if layout else _colour_edges(list(zz))
        for colour in colours:
            terms = []
            for a, b in colour:
                c, t = (a, b) if degree[a] >= degree[b] else (b, a)
                terms.append((c, t, zz[(a, b)]))
            for _, t, _ in terms:
                out.h(t)
            for c, t, _ in terms:
                out.cz(c, t)
            for _, t, theta in terms:
                out.rx(theta, t)
            for c, t, _ in terms:
                out.cz(c, t)
            for _, t, _ in terms:
                out.h(t)


    def group_zz(
        body: QuantumCircuit,
        fanout: bool = False,
        max_qubits: int = 100,
        layout: dict[int, tuple[int, int]] | None = None,
        collect: set[tuple[int, int]] | None = None,
    ) -> QuantumCircuit:
        """v3a (ZZ grouping): ZZ rotations (`cx(a,b) rz(b) cx(a,b)` or `rzz`) all commute with each other and with diagonal
        single-qubit gates. Each run of them is re-emitted colour by colour: per colour (a set of disjoint pairs)
        H(targets) - CZ - Rx(theta)(targets) - CZ - H(targets), which equals the ZZ rotations exactly (H Rz H = Rx).
        One colour = 2 pulses for ALL its pairs. The busier qubit of each pair is the control.

        v3b (`fanout`): if one hub qubit has far more partners than the rest (Max-Cut reference qubit), copy it onto extra
        qubits in the Z basis with a doubling tree of CX (|x>|0> -> |x>|x>), spread its ZZ terms over hub + copies, then
        undo the copies. ZZ(copy, p) = ZZ(hub, p) while the copy holds the hub's value, and the copies end in |0>.
        `layout` (v4): colour by grid position instead of greedily. `collect`: receives every two-qubit pair used.
        The max degree drops from n-1 to ~3, so far fewer colours (pulses). Extra qubits are allowed (checked: the
        validator accepts correct ancilla circuits and rejects wrong ones)."""
        segments = _segments(body)
        plans = [_fanout_plan(seg[1][0]) if fanout and seg[0] == "zz" else None for seg in segments]
        n_extra = min(max((p[1] for p in plans if p), default=0), max(0, max_qubits - body.num_qubits))
        out = body.copy_empty_like()
        extra: list[int] = []
        if n_extra:
            out.add_register(AncillaRegister(n_extra, "fanout"))
            extra = list(range(body.num_qubits, body.num_qubits + n_extra))

        for (kind, payload), plan in zip(segments, plans, strict=True):
            if kind == "op":
                op, qs = payload
                out.append(op, qs)
                continue
            zz, diag = payload
            for op, qs in diag:
                out.append(op, qs)
            if not plan or not extra:
                if collect is not None:
                    collect.update(zz)
                _emit_zz(out, zz, layout)
                continue
            hub, k = plan
            copies = extra[: min(k, len(extra))]
            rounds = _copy_rounds(hub, copies)
            holders = [hub, *copies]
            spread: dict[tuple[int, int], float] = {}
            i = 0
            for (a, b), theta in zz.items():
                if hub in (a, b):
                    partner = b if a == hub else a
                    key = tuple(sorted((holders[i % len(holders)], partner)))
                    spread[key] = spread.get(key, 0.0) + theta
                    i += 1
                else:
                    spread[(a, b)] = spread.get((a, b), 0.0) + theta
            if collect is not None:
                collect.update(spread)
                collect.update(tuple(sorted(pair)) for rnd in rounds for pair in rnd)
            for rnd in rounds:
                for src, dst in rnd:
                    out.cx(src, dst)
            _emit_zz(out, spread, layout)
            for rnd in reversed(rounds):
                for src, dst in rnd:
                    out.cx(src, dst)
        return out


    # ---- compiler/schedule.py --------------------------------------------------------------------------------
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


    # ---- compiler/placement.py -------------------------------------------------------------------------------
    def interaction_counts(schedule: Schedule) -> Counter[tuple[int, int]]:
        return Counter(pair for step in schedule.steps if isinstance(step, CZRound) for pair in step.pairs)


    def _bfs_order(schedule: Schedule) -> list[int]:
        """v0: BFS from the busiest qubit, neighbours by how many CZs they share."""
        n = schedule.num_qubits
        neighbours: dict[int, Counter[int]] = {q: Counter() for q in range(n)}
        for (a, b), c in interaction_counts(schedule).items():
            neighbours[a][b] += c
            neighbours[b][a] += c
        degree = {q: sum(neighbours[q].values()) for q in range(n)}
        order: list[int] = []
        seen: set[int] = set()
        for start in sorted(range(n), key=lambda q: -degree[q]):
            if start in seen:
                continue
            seen.add(start)
            todo = deque([start])
            while todo:
                q = todo.popleft()
                order.append(q)
                for nb, _ in neighbours[q].most_common():
                    if nb not in seen:
                        seen.add(nb)
                        todo.append(nb)
        return order


    def _cuthill_mckee_order(schedule: Schedule) -> list[int]:
        """v2a: BFS from a qubit on the edge of the interaction graph (lowest degree): connected qubits get nearby
        positions in the order. For a chain (HEA) this is exactly 0, 1, 2, ... from one end."""
        graph = nx.Graph()
        graph.add_nodes_from(range(schedule.num_qubits))
        graph.add_edges_from(interaction_counts(schedule))
        return list(nx.utils.cuthill_mckee_ordering(graph))


    def _snake_sites(device: Device) -> list[Site]:
        """Storage sites row by row (nearest the entangling zone first), alternating direction: row 9 left->right,
        row 8 right->left, ... so consecutive qubits of the order are always neighbours, also across rows."""
        rows: dict[int, list[Site]] = {}
        for site in device.storage_sites:
            rows.setdefault(site[1], []).append(site)
        return [s for i, row in enumerate(rows.values()) for s in (row if i % 2 == 0 else row[::-1])]


    def home_layout(schedule: Schedule, device: Device, method: HomeMethod = "bfs") -> Layout:
        """Where every atom starts (the initial loading) and returns to."""
        n = schedule.num_qubits
        if n > len(device.storage_sites):
            msg = f"{n} qubits do not fit in {len(device.storage_sites)} storage sites."
            raise ValueError(msg)
        if method == "snake":
            order, sites = _cuthill_mckee_order(schedule), _snake_sites(device)
        else:
            order, sites = _bfs_order(schedule), device.storage_sites
        return {q: sites[i] for i, q in enumerate(order)}


    def round_targets(
        rnd: CZRound,
        layout: Layout,
        device: Device,
        occupied: set[Site],
        next_partner: dict[int, int] | None = None,
        lookahead: float = 0.0,
    ) -> Layout:
        """Hungarian matching: each pair -> a free pair site; cost = the longer of the two atoms' trips.
        v5 (`lookahead` > 0): plus lookahead x distance from each slot to the current site of that atom's next
        partner, so pairs are placed towards where they will be needed next (ZAC-style lookahead)."""
        free = [p for p in device.pair_sites if not occupied.intersection(device.pair_slots(p))]
        if len(free) < len(rnd.pairs):
            msg = "Not enough free pair sites for this round."
            raise ValueError(msg)
        cost = np.empty((len(rnd.pairs), len(free)))
        flip = np.zeros_like(cost, dtype=bool)
        for i, (a, b) in enumerate(rnd.pairs):
            for j, p in enumerate(free):
                left, right = device.pair_slots(p)
                straight = max(device.dist(layout[a], left), device.dist(layout[b], right))
                crossed = max(device.dist(layout[b], left), device.dist(layout[a], right))
                if lookahead and next_partner:
                    straight += lookahead * (
                        _ahead(a, left, layout, next_partner, device) + _ahead(b, right, layout, next_partner, device)
                    )
                    crossed += lookahead * (
                        _ahead(b, left, layout, next_partner, device) + _ahead(a, right, layout, next_partner, device)
                    )
                cost[i, j] = min(straight, crossed)
                flip[i, j] = crossed < straight
        rows, cols = linear_sum_assignment(cost)
        targets: Layout = {}
        for i, j in zip(rows, cols, strict=True):
            a, b = rnd.pairs[i]
            left, right = device.pair_slots(free[j])
            if flip[i, j]:
                a, b = b, a
            targets[a], targets[b] = left, right
        return targets


    def _ahead(q: int, slot: Site, layout: Layout, next_partner: dict[int, int], device: Device) -> float:
        """Lookahead term: how far `slot` is from where q's next partner currently is (0 if q has none)."""
        p = next_partner.get(q)
        return device.dist(slot, layout[p]) if p is not None and p in layout else 0.0


    def shift_targets(rnd: CZRound, layout: Layout, device: Device, occupied: set[Site]) -> Layout:
        """v2b: pairs sitting side by side in the same storage row all get the SAME column shift d (chosen so every left
        atom lands on a left slot), and storage rows map in order onto the nearest entangling rows. One shared row map +
        one shared column map = the router can lift them all with a single move. Other pairs: Hungarian on what's left."""
        if not all(device.in_storage(layout[q]) for q in rnd.qubits):
            return round_targets(rnd, layout, device, occupied)
        x0, x1 = device.entangling.x
        side_by_side: list[tuple[int, int]] = []
        for a, b in rnd.pairs:
            (xa, ya), (xb, yb) = layout[a], layout[b]
            if ya == yb and abs(xa - xb) == 1:
                side_by_side.append((a, b) if xa < xb else (b, a))  # (left atom, right atom)

        def fits(pair: tuple[int, int], d: int) -> bool:
            left_x = layout[pair[0]][0] + d
            return (left_x - x0) % 2 == 0 and x0 <= left_x and left_x + 1 <= x1

        shifts = sorted(range(x0 - x1, x1 - x0 + 1), key=abs)
        d = max(shifts, key=lambda s: sum(fits(p, s) for p in side_by_side))  # ties: smallest |d| (max keeps the first)
        placed = [p for p in side_by_side if fits(p, d)]

        storage_rows = sorted({layout[a][1] for a, _ in placed})
        nearest_rows: list[int] = []
        for row, _ in device.pair_sites:
            if row not in nearest_rows:
                nearest_rows.append(row)
        row_map = dict(zip(storage_rows, sorted(nearest_rows[: len(storage_rows)]), strict=True))

        targets: Layout = {}
        for a, b in placed:
            (x, y) = layout[a]
            left, right = (x + d, row_map[y]), (x + d + 1, row_map[y])
            if left in occupied or right in occupied:
                continue  # leave it to the Hungarian fallback
            targets[a], targets[b] = left, right
        rest = [p for p in rnd.pairs if p[0] not in targets]
        if rest:
            taken = occupied | set(targets.values())
            targets |= round_targets(CZRound(rest, rnd.block), layout, device, taken)
        return targets


    def grid_layout(
        edges: set[tuple[int, int]], num_qubits: int, device: Device, seed: int = 0, steps: int = 20000
    ) -> Layout:
        """v4 (2D): simulated annealing over storage positions. Cost = for each pair, 0 if the two atoms are grid
        neighbours (left/right or up/down: both can reach a pair site with merged moves), else their Manhattan distance;
        plus a small penalty per row away from the entangling zone. Move: put a random qubit on a random site of the rows
        nearest the entangling zone (swapping with the qubit there, if any). Starts from the Cuthill-McKee snake."""
        rng = random.Random(seed)
        width = device.storage.x[1] - device.storage.x[0] + 1
        rows_needed = math.ceil(num_qubits / width) + 1
        snake = _snake_sites(device)
        sites = [s for s in snake if device._rows_to_entangling(s[1]) <= rows_needed]
        graph = nx.Graph()
        graph.add_nodes_from(range(num_qubits))
        graph.add_edges_from(edges)
        order = list(nx.utils.cuthill_mckee_ordering(graph))
        pos: dict[int, Site] = {q: snake[i] for i, q in enumerate(order)}
        nbrs = {q: list(graph.neighbors(q)) for q in range(num_qubits)}

        def pair_cost(a: Site, b: Site) -> float:
            d = abs(a[0] - b[0]) + abs(a[1] - b[1])
            return 0.0 if d == 1 else float(d)

        def node_cost(q: int, at: Site) -> float:
            return sum(pair_cost(at, pos[n]) for n in nbrs[q] if n in pos) + 0.05 * device._rows_to_entangling(at[1])

        occupant: dict[Site, int] = {site: q for q, site in pos.items()}
        for step in range(steps):
            temp = 2.0 * (1 - step / steps) + 0.01
            q = rng.randrange(num_qubits)
            target = rng.choice(sites)
            src = pos[q]
            if target == src:
                continue
            other = occupant.get(target)
            old = node_cost(q, src) + (node_cost(other, target) if other is not None else 0.0)
            pos[q] = target
            if other is not None:
                pos[other] = src
            new = node_cost(q, target) + (node_cost(other, src) if other is not None else 0.0)
            if new <= old or rng.random() < math.exp((old - new) / temp):
                occupant.pop(src, None)
                occupant[target] = q
                if other is not None:
                    occupant[src] = other
            else:
                pos[q] = src
                if other is not None:
                    pos[other] = target
        return pos


    def grid_targets(rnd: CZRound, layout: Layout, device: Device, occupied: set[Site]) -> Layout:
        """v4 (2D): vertical neighbours are interleaved onto one entangling row (upper atom -> even column, lower atom ->
        odd column, columns spread x2; storage columns of the right half use a second row), so each storage row moves as
        one block. Horizontal neighbours: shift_targets. Anything else: Hungarian (inside shift_targets)."""
        if not all(device.in_storage(layout[q]) for q in rnd.qubits):
            return round_targets(rnd, layout, device, occupied)
        x0 = device.entangling.x[0]
        half = device.pairs_per_row
        rows_by_distance: list[int] = []
        for row, _ in device.pair_sites:
            if row not in rows_by_distance:
                rows_by_distance.append(row)
        vertical = [
            (a, b)
            for a, b in rnd.pairs
            if layout[a][0] == layout[b][0] and abs(layout[a][1] - layout[b][1]) == 1 and layout[a][0] - x0 < 2 * half
        ]
        row_for: dict[tuple[int, int], int] = {}
        free_rows = iter(rows_by_distance)
        for low in sorted({min(layout[a][1], layout[b][1]) for a, b in vertical}, reverse=True):
            for side in (0, 1):
                row = next(free_rows, None)
                if row is not None:
                    row_for[(low, side)] = row
        targets: Layout = {}
        taken = set(occupied)
        for a, b in vertical:
            upper, lower = (a, b) if layout[a][1] > layout[b][1] else (b, a)
            x, low = layout[upper][0] - x0, layout[lower][1]
            side = 0 if x < half else 1
            if (low, side) not in row_for:
                continue
            col = x0 + 2 * (x - side * half)
            left, right = (col, row_for[(low, side)]), (col + 1, row_for[(low, side)])
            if left in taken or right in taken:
                continue
            targets[upper], targets[lower] = left, right
            taken |= {left, right}
        rest = [pair for pair in rnd.pairs if pair[0] not in targets]
        if rest:
            targets |= shift_targets(CZRound(rest, rnd.block), layout, device, taken)
        return targets


    # ---- compiler/routing.py ---------------------------------------------------------------------------------
    def single_move(q: int, src: Site, dst: Site) -> MoveOp:
        return MoveOp((q,), (src[1],), (src[0],), (dst[1],), (dst[0],))


    def _increasing(values: list[int]) -> bool:
        return all(a < b for a, b in itertools.pairwise(values))


    def build_move(group: list[int], layout: Layout, targets: Layout, device: Device) -> MoveOp | None:
        """One AOD move carrying every atom of `group` to its target, or None if that isn't legal right now."""
        row_map: dict[int, int] = {}
        col_map: dict[int, int] = {}
        for q in group:
            (x, y), (nx, ny) = layout[q], targets[q]
            if row_map.setdefault(y, ny) != ny or col_map.setdefault(x, nx) != nx:
                return None  # one source line would have to split into two destination lines
        rows, cols = sorted(row_map), sorted(col_map)
        new_rows, new_cols = [row_map[r] for r in rows], [col_map[c] for c in cols]
        if not (_increasing(new_rows) and _increasing(new_cols)):
            return None  # lines would cross (AOD order must be preserved)
        if len({device.zone_of(layout[q]) for q in group}) > 1 or len({device.zone_of(targets[q]) for q in group}) > 1:
            return None  # pickup rows in one zone, drop rows in one zone
        members = set(group)
        for q, (x, y) in layout.items():
            if q not in members and rows[0] <= y <= rows[-1] and cols[0] <= x <= cols[-1]:
                return None  # ghost: an outside atom sits inside the pickup rectangle
        taken = {site for q, site in layout.items() if q not in members}
        if any(targets[q] in taken for q in group):
            return None  # would drop onto an atom that isn't part of this move
        return MoveOp(
            tuple(sorted(group)),
            tuple(range(rows[0], rows[-1] + 1)),
            tuple(range(cols[0], cols[-1] + 1)),
            tuple(new_rows),
            tuple(new_cols),
        )


    def _apply(move: MoveOp, layout: Layout, targets: Layout) -> None:
        for q in move.qubits:
            layout[q] = targets[q]


    def route_merged(layout: Layout, targets: Layout, device: Device) -> list[MoveSet]:
        """Greedy: first-fit every atom into an existing group if the group stays a legal move, then try merging groups.
        Groups are executed when legal (drop sites free); anything left falls back to one-atom moves. Updates `layout`."""
        pending = sorted(
            (q for q, dst in targets.items() if layout[q] != dst),
            key=lambda q: (device.zone_of(layout[q]) or "", layout[q][1], layout[q][0]),
        )
        groups: list[list[int]] = []
        for q in pending:
            for g in groups:
                if build_move([*g, q], layout, targets, device) is not None:
                    g.append(q)
                    break
            else:
                groups.append([q])

        merged = True
        while merged:
            merged = False
            for i in range(len(groups)):
                for j in range(i + 1, len(groups)):
                    if build_move(groups[i] + groups[j], layout, targets, device) is not None:
                        groups[i] += groups.pop(j)
                        merged = True
                        break
                if merged:
                    break

        sets: list[MoveSet] = []
        progress = True
        while groups and progress:
            progress = False
            for g in list(groups):
                move = build_move(g, layout, targets, device)
                if move is not None:
                    sets.append([move])
                    _apply(move, layout, targets)
                    groups.remove(g)
                    progress = True
        leftover = {q: targets[q] for g in groups for q in g}
        return sets + route_sequential(layout, leftover, device)


    def route_sequential(layout: Layout, targets: Layout, device: Device) -> list[MoveSet]:
        """Moves every atom in `targets` there, one by one, never dropping onto an occupied site. Updates `layout`."""
        pending = {q: dst for q, dst in targets.items() if layout[q] != dst}
        occupied = set(layout.values())
        sets: list[MoveSet] = []

        def go(q: int, dst: Site) -> None:
            sets.append([single_move(q, layout[q], dst)])
            occupied.discard(layout[q])
            occupied.add(dst)
            layout[q] = dst

        while pending:
            movable = [q for q, dst in pending.items() if dst not in occupied]
            for q in movable:
                go(q, pending.pop(q))
            if not movable:
                # every target is taken by another pending atom (a cycle): park one atom on a free site first
                q = next(iter(pending))
                zone = device.zone_of(pending[q])
                sites = (
                    device.storage_sites
                    if zone == "storage"
                    else [s for p in device.pair_sites for s in device.pair_slots(p)]
                )
                park = min((s for s in sites if s not in occupied), key=lambda s: device.dist(layout[q], s))
                go(q, park)
        return sets


    def route(layout: Layout, targets: Layout, device: Device, merge: bool) -> list[MoveSet]:
        return route_merged(layout, targets, device) if merge else route_sequential(layout, targets, device)


    # ---- compiler/emit.py ------------------------------------------------------------------------------------
    def _write_moves(qc: QuantumCircuit, sets: list[MoveSet]) -> None:
        for move_set in sets:
            for m in move_set:
                qc.move(list(m.qubits), rows=m.rows, cols=m.cols, new_rows=m.new_rows, new_cols=m.new_cols)


    def _in_next_pulse(schedule: Schedule) -> list[set[int]]:
        """For each CZ round i: qubits of round i also in the next CZ round, with no storage gate in between."""
        steps = schedule.steps
        result: list[set[int]] = [set() for _ in steps]
        for i, step in enumerate(steps):
            if not isinstance(step, CZRound):
                continue
            busy: set[int] = set()
            for later in steps[i + 1 :]:
                if isinstance(later, CZRound):
                    result[i] = (step.qubits & later.qubits) - busy
                    break
                busy |= later.qubits
        return result


    def _next_partner(schedule: Schedule) -> list[dict[int, int]]:
        """For CZ round i and each qubit q in it: q's partner in the next CZ round that uses q (if any)."""
        steps = schedule.steps
        result: list[dict[int, int]] = [{} for _ in steps]
        upcoming: dict[int, int] = {}
        for i in range(len(steps) - 1, -1, -1):
            step = steps[i]
            if not isinstance(step, CZRound):
                continue
            result[i] = {q: upcoming[q] for q in step.qubits if q in upcoming}
            for a, b in step.pairs:
                upcoming[a], upcoming[b] = b, a
        return result


    def _return_sites(
        returning: list[int], partner: dict[int, int], home: Layout, layout: Layout, device: Device
    ) -> Layout:
        """v2b dynamic return: a returning atom parks on a free storage site right next to (same row) the home of its next
        CZ partner, so next time the two are side by side and go up in one move. Otherwise it goes back to its home."""
        taken = set(layout.values()) | set(home.values())
        anchored: set[int] = set()
        targets: Layout = {}
        for q in sorted(returning):
            p = partner.get(q)
            if p is not None and q not in anchored:
                px, py = home[p]
                spots = [(px + dx, py) for dx in (-1, 1) if device.in_storage((px + dx, py)) and (px + dx, py) not in taken]
                if spots:
                    site = min(spots, key=lambda s: device.dist(layout[q], s))
                    taken.add(site)
                    home[q] = site
                    anchored.add(p)
            targets[q] = home[q]
        return targets


    def emit(
        original: QuantumCircuit,
        global_phase: float,
        schedule: Schedule,
        device: Device,
        config: Config,
        home_override: Layout | None = None,
    ) -> tuple[QuantumCircuit, Layout]:
        initial = dict(home_override) if home_override else home_layout(schedule, device, config.home)
        home = dict(initial)  # where each atom returns to; changes with dynamic_return
        layout = dict(home)
        qc = original.copy_empty_like()
        qc.global_phase = global_phase
        stay_up = _in_next_pulse(schedule)
        partners = _next_partner(schedule)

        def move_to(targets: Layout) -> None:
            if targets:
                _write_moves(qc, route(layout, targets, device, config.merge_moves))

        for i, step in enumerate(schedule.steps):
            if isinstance(step, StoragePhase):
                move_to({q: home[q] for q in step.qubits if not device.in_storage(layout[q])})
                for op in step.ops:
                    getattr(qc, op.name)(op.angle, op.qubit)
                continue

            assert isinstance(step, CZRound)
            in_round = step.qubits
            # safety net: nobody outside this pulse may be in the entangling zone when it fires
            move_to({q: home[q] for q, site in layout.items() if q not in in_round and device.in_entangling(site)})

            occupied = {site for q, site in layout.items() if q not in in_round}
            if config.targets == "shift":
                move_to(shift_targets(step, layout, device, occupied))
            elif config.targets == "grid":
                move_to(grid_targets(step, layout, device, occupied))
            else:
                move_to(round_targets(step, layout, device, occupied, partners[i], config.lookahead))
            for a, b in step.pairs:
                qc.cz(a, b)
            returning = [q for q in in_round if not (config.reuse and q in stay_up[i])]
            if config.dynamic_return:
                move_to(_return_sites(returning, partners[i], home, layout, device))
            else:
                move_to({q: home[q] for q in returning})

        for q, c in schedule.measurements:
            qc.measure(q, c)
        return qc, initial


    # ---- compiler/__init__.py --------------------------------------------------------------------------------
    def _depth(qc: QuantumCircuit) -> int:
        return qc.depth(filter_function=lambda inst: inst.operation.name not in ("move", "barrier"))


    def tucano_compile(
        circuit: QuantumCircuit, architecture_spec: ArchitectureSpec | None = None, config: Config = DEFAULT
    ) -> tuple[QuantumCircuit, Architecture]:
        device = Device.from_spec(architecture_spec)
        body, measurements = split_measurements(circuit)
        # ZZ modes: (group_zz, fanout) = off / grouped / grouped + hub fan-out on extra atoms
        zz_modes = [(False, False), (True, False), (True, True)] if config.pick_best else [(config.group_zz, config.fanout)]
        fuse_options = (False, True) if config.pick_best else (config.fuse_rz,)
        max_qubits = len(device.storage_sites)
        natives = {m: to_native(group_zz(body, m[1], max_qubits) if m[0] else body) for m in zz_modes}

        # v4 (2D): pair graph -> 2D layout -> circuit coloured by grid position
        grid_modes = [(True, False), (True, True)] if config.pick_best and config.grid2d else []
        grid_natives, grid_layouts = {}, {}
        for m in grid_modes:
            pairs: set[tuple[int, int]] = set()
            first = to_native(group_zz(body, m[1], max_qubits, collect=pairs))
            pairs |= {
                tuple(sorted(first.find_bit(q).index for q in inst.qubits))
                for inst in first.data
                if inst.operation.name == "cz"
            }
            grid_layouts[m] = grid_layout(pairs, first.num_qubits, device, config.seed)
            grid_natives[m] = to_native(group_zz(body, m[1], max_qubits, layout=grid_layouts[m]))
        schedules = {
            (m, f): build_schedule(natives[m], measurements, circuit.num_clbits, f) for m in zz_modes for f in fuse_options
        }

        if config.pick_best:
            variants = [
                replace(config, group_zz=m[0], fanout=m[1], fuse_rz=f, reuse=r, home=h, targets=t, dynamic_return=d)
                for m in zz_modes
                for f in fuse_options
                for r in (False, True)
                for h in ("bfs", "snake")
                for t in ("hungarian", "shift")
                for d in (False, True)
            ]
            # v5: lookahead only changes the Hungarian targets, so only those variants get a lookahead copy
            variants += [replace(v, lookahead=config.lookahead_try) for v in variants if v.targets == "hungarian"]
        else:
            variants = [config]
        for m in grid_modes:
            for f in fuse_options:
                schedules[(("grid", *m), f)] = build_schedule(grid_natives[m], measurements, circuit.num_clbits, f)
            variants += [
                replace(
                    config,
                    group_zz=True,
                    fanout=m[1],
                    fuse_rz=f,
                    reuse=r,
                    home="grid",
                    targets="grid",
                    dynamic_return=False,
                )
                for f in fuse_options
                for r in (False, True)
            ]
        best = None
        ref: tuple[float, int] | None = None
        for v in variants:
            if v.home == "grid":
                m = (v.group_zz, v.fanout)
                native, schedule, given = grid_natives[m], schedules[(("grid", *m), v.fuse_rz)], grid_layouts[m]
            else:
                native, given = natives[(v.group_zz, v.fanout)], None
                schedule = schedules[((v.group_zz, v.fanout), v.fuse_rz)]
            compiled, home = emit(native, native.global_phase, schedule, device, v, given)
            dist, depth = move_distance(compiled, home, device), _depth(compiled)
            ref = ref or (max(dist, 1e-9), max(depth, 1))
            if config.rank == "balanced":
                key: tuple[float, ...] = (round(dist / ref[0] + config.depth_weight * depth / ref[1], 9), dist)
            else:
                key = (round(dist, 9), depth)
            if best is None or key < best[0]:
                best = (key, compiled, home)
        assert best is not None
        _, compiled, home = best
        return compiled, Architecture(home, spec=architecture_spec)


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
