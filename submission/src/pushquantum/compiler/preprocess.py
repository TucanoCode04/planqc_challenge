"""Input circuit -> native circuit (rx, rz, cz only) + the final measurements, kept aside."""

from __future__ import annotations

import math
from collections import Counter

from qiskit import AncillaRegister, QuantumCircuit, transpile

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
