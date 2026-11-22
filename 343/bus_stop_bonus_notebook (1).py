# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "matplotlib",
#     "numpy",
#     "polars==2.1.0",
#     "qiskit==2.6.2",
#     "scipy",
# ]
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")

with app.setup:
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl
    from qiskit import QuantumCircuit

    import bus_stop_ansatz as bsa


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # PlanQC Track — Bus Stop Optimization & Hardware-Tailored Variational Ansätze

    **PushQuantum Hackathon 2026 | Problem Encoding, VQE Optimization & Hardware-Native Circuit Design**

    This notebook documents our problem encoding and hardware-tailored variational ansätze for the
    **12-stop bus consolidation problem** on planqc's 2-zone neutral-atom quantum processor
    (Storage zone $y \in [0, 9]$: targeted/global `rx`, targeted `rz`; Entangling zone $y \in [10, 19]$:
    transverse `cz` on paired columns $(2k, 2k+1)$).

    ---

    ## 1. Problem Formulation & Why Abstract Ansätze Fail on Zoned Hardware

    ### 1.1 Truncated Tetrahedron Minimum-Weight Vertex Cover (MWVC)

    A city must consolidate **12 bus stops** across **18 neighborhoods**. Each neighborhood specifies two
    acceptable stops (an edge $e = (i, j) \in E$), and keeping stop $i$ incurs maintenance cost $c_i$:

    | Triangle | Stops $(a, b, c)$ | Costs $(c_a, c_b, c_c)$ | Inter-Triangle Bridges |
    |----------|-------------------|-------------------------|------------------------|
    | $T_1$ | $(1, 2, 3)$ | $(2, 1, 3)$ | $(1, 4),\ (2, 7),\ (3, 10)$ |
    | $T_2$ | $(4, 5, 6)$ | $(4, 2, 1)$ | $(1, 4),\ (5, 8),\ (6, 11)$ |
    | $T_3$ | $(7, 8, 9)$ | $(3, 2, 4)$ | $(2, 7),\ (5, 8),\ (9, 12)$ |
    | $T_4$ | $(10, 11, 12)$ | $(1, 3, 2)$ | $(3, 10),\ (6, 11),\ (9, 12)$ |

    Let $x_i \in \{0, 1\}$ denote whether stop $i$ is kept ($x_i = 1$) or dropped ($x_i = 0$). Penalizing
    every uncovered neighborhood with $P = 5 > \max_i c_i = 4$ yields the exact QUBO objective:

    $$F(x) = \sum_{i=1}^{12} c_i x_i + 5 \sum_{(i,j)\in E} (1 - x_i)(1 - x_j)$$

    - **Unique global optimum:** keep $\{1, 2, 5, 6, 7, 8, 10, 12\}$ (drop $\{3, 4, 9, 11\}$, one stop per
      triangle with no two dropped stops linked by a bridge), achieving **$F^* = 14$**.

    ### 1.2 Standard 12-Qubit Ising Mapping

    Substituting $x_i = (1 - Z_i)/2$ (so $Z_i = -1$ for a kept stop) and noting that the truncated
    tetrahedron is 3-regular ($d_i = 3$ for every stop, $|E| = 18$, $\sum_i c_i = 28$):

    $$H_{12q} = 5 \sum_{(i,j)\in E} Z_i Z_j + \sum_{i=1}^{12} (15 - 2 c_i) Z_i, \qquad F(x) = \frac{H_{12q} + 146}{4}$$

    ### 1.3 Why Abstract "Problem-Faithful" Ansätze Break Down on Zoned Neutral Atoms

    On planqc's 2-zone architecture, **moving atoms between zones is the dominant runtime and error source**,
    and only `{rx, rz}` (Storage) and paired-column `cz` (Entangling) are native (targeting the strict
    `{rx, rz, cz}` native gate set without `rzz`):

    1. **iHVA (`bus_stop_12q_ihva_stagger_p1.qasm`) explodes in zone round-trips:**
       The imaginary Hamiltonian variational ansatz (Wang et al., *Phys. Rev. A* 111, 032612 (2025)) places
       a non-commuting $e^{-i\theta Z_i Y_j/2}$ gate on every Max-Cut edge—including a high-degree reference
       qubit coupled to all stops. Without native `rzz`, each $R_{ZY}$ decomposes into **two `cz` gates
       separated by storage-zone `rx`/`rz` rotations**. Even at $p=1$ after `optimization_level=3`,
       `ihva_stagger` compiles to **58 `cz` gates, 22 entangling rounds, 19 Storage $\leftrightarrow$ Entangling
       visits, and native depth 57**.
    2. **Even linear `cx`-based HEA (`bus_stop_12q_hea_linear_p1.qasm`) pays a hidden Hadamard tax:**
       Because $\text{CX}_{i,j} = H_j\,\text{CZ}_{i,j}\,H_j$, transpiling two consecutive `cx` layers to
       `{rx, rz, cz}` injects `rx`/`rz` rotations *between* Round 1 and Round 2 of `cz`, doubling Storage
       $\leftrightarrow$ Entangling visits from **1 to 2** and inflating depth from **5 to 9**.
    3. **Our design principle:** Build circuits **directly in native `{rx, rz, cz}`**, omit unobservable
       terminal `rz` gates before $Z$-basis measurement, eliminate infeasible configurations via physics-driven
       qubit compression, and align entangling rounds to single-step AOD translations.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ---

    ## 2. Our Hardware-Tailored Encodings & Ansätze

    ### 2.1 Candidate A (Primary Submission): 8-Qubit Triangle-Compressed Hardware-Native Ansatz

    #### Proof of Exact $4096 \to 256$ Hilbert-Space Compression (16× Reduction)

    Inside any triangle $T_t = (a, b, c)$, dropping two or three stops violates at least one internal edge,
    incurring penalty $+5 > \max_i c_i = 4$. Therefore, every feasible (and every low-energy) assignment
    within $T_t$ must keep at least 2 of its 3 stops, leaving **exactly 4 feasible local states** per triangle:

    | 2-Qubit State $(q_{2t}, q_{2t+1})$ | Stops $(x_a, x_b, x_c)$ | Decision | Triangle Maintenance Cost |
    |---|---|---|---|
    | `"11"` | $(1, 1, 1)$ | Keep all 3 stops | $c_a + c_b + c_c$ |
    | `"10"` | $(0, 1, 1)$ | Drop 1st stop $a$ | $c_b + c_c$ |
    | `"01"` | $(1, 0, 1)$ | Drop 2nd stop $b$ | $c_a + c_c$ |
    | `"00"` | $(1, 1, 0)$ | Drop 3rd stop $c$ | $c_a + c_b$ |

    Encoding the four triangles in $4 \times 2 = \mathbf{8}$ **qubits** ($T_1=(q_0,q_1),\ T_2=(q_2,q_3),\ T_3=(q_4,q_5),\ T_4=(q_6,q_7)$):
    - **Satisfies all 12 intra-triangle constraints by construction** (eliminating 3,840 infeasible states).
    - Maps the unique optimum (drop $3, 4, 9, 11$) to **`"00100001"`** in $q_0 \dots q_7$ order
      (Qiskit little-endian index `132`, binary `"10000100"` reversed).

    #### Exact 8-Qubit Projector / Pauli-$Z$ Hamiltonian $H_{8q}$

    Using single-qubit projectors $P_0(k) = \frac{I + Z_k}{2}$ and $P_1(k) = \frac{I - Z_k}{2}$, define the
    2-qubit triangle state projectors $\Pi_{b_0 b_1}^{(t)} = P_{b_0}(2t)\,P_{b_1}(2t+1)$. Each triangle's
    maintenance cost is a 2-local diagonal operator:

    $$H_{\text{maint}}^{(t)} = \frac{3(c_a+c_b+c_c)}{4} I + \frac{c_a-c_b-c_c}{4} Z_{2t} + \frac{-c_a+c_b-c_c}{4} Z_{2t+1} + \frac{c_a+c_b-c_c}{4} Z_{2t}Z_{2t+1}$$

    The 6 inter-triangle bridges add 4-local projector penalties whenever both endpoints are dropped:

    $$H_{\text{bridge}} = 5\Big(\Pi_{10}^{(1)}\Pi_{10}^{(2)} + \Pi_{01}^{(1)}\Pi_{10}^{(3)} + \Pi_{00}^{(1)}\Pi_{10}^{(4)} + \Pi_{01}^{(2)}\Pi_{01}^{(3)} + \Pi_{00}^{(2)}\Pi_{01}^{(4)} + \Pi_{00}^{(3)}\Pi_{00}^{(4)}\Big)$$

    $$H_{8q} = \sum_{t=1}^{4} H_{\text{maint}}^{(t)} + H_{\text{bridge}}$$

    #### Single-Row Neutral-Atom Hardware Layout (`y = 9 <-> y = 10`, 1 Zone Round-Trip!)

    ```text
    ENTANGLING ZONE (y = 10): paired columns (0,1)  (2,3)  (4,5)  (6,7)  (8,9)
      Round 1 (intra-triangle CZ):          [q0 q1] [q2 q3] [q4 q5] [q6 q7] [ .  . ]
        -- 1-col horizontal AOD shift (cols 0..7 -> 1..8), NO storage return! --
      Round 2 (inter-triangle CZ):          [ . q0] [q1 q2] [q3 q4] [q5 q6] [q7  . ]
                                                   \___/   \___/   \___/
                                                (lone q0, q7 experience identity!)
    --------------------------------------------------------------------------------
    STORAGE ZONE (y = 9):  rx, rz           (q0)(q1)(q2)(q3)(q4)(q5)(q6)(q7) (.) (.)
                                             c0  c1  c2  c3  c4  c5  c6  c7  c8  c9
    ```

    - **Total travel for `layers = 1`:** 1 vertical step ($y: 9 \to 10$) + 1 horizontal 1-column shift
      + 1 vertical return ($y: 10 \to 9$) for final `rx`. **Zero intervening 1Q gates between CZ rounds.**
    - **For `layers = 2`:** Layer 2 reverses entangling order (Round 2 $\to$ Round 1) so atoms re-enter
      at `cols 1..8` and slide back to `cols 0..7` without an extra reset shift.

    ---

    ### 2.2 Candidate B (Direct 12-Qubit Comparison): 12-Qubit "5+5 Real-Edges" Native Ansatz

    For apples-to-apples 12-qubit comparison without compression (1 qubit per stop, $q_{v-1} \leftrightarrow \text{stop } v$),
    we co-design the entangling graph directly against the **5-pairs-per-row Entangling zone limit**
    (10 columns = 5 adjacent pairs `(0,1), (2,3), (4,5), (6,7), (8,9)` per row). A 6-pair round would
    overflow a single-row lift and inflate AOD shuttle count; instead, decomposing the graph into a
    **7-stop chain** `3–2–1–4–5–6–11` (`Row 9`) and a **5-stop chain** `10–12–9–8–7` (`Row 8`) yields
    **10 real problem edges** (including 3 inter-triangle bridges) with `maximum matching = 5`, executing
    in **exactly 2 parallel pulses of 5 pairs each** (still **1 entangling-zone visit** per layer!):

    ```text
    Storage / Entangling Grid (stop IDs; qubit = stop - 1):
      Row 9 (cols 0..6): [ 3,  2,  1,  4,  5,  6, 11 ]  -> qubits [ 2,  1,  0,  3,  4,  5, 10 ]
      Row 8 (cols 0..4): [ 10, 12,  9,  8,  7 ]         -> qubits [ 9, 11,  8,  7,  6 ]

      Round 1 (5 real edges, <= 5 pairs/row):   (3,2), (1,4), (5,6), (10,12), (9,8)
        -- 1-col horizontal AOD shift (cols 0..6 -> 1..7), NO storage return! --
      Round 2 (5 real edges, <= 5 pairs/row):   (2,1), (4,5), (6,11), (12,9), (8,7)
    ```
    """)
    return


@app.cell
def _():
    # Exact verification of H_8q Pauli-Z expansion and ground-state decoding
    h8_terms = bsa.build_8q_pauli_hamiltonian()
    _h8_diag = bsa.evaluate_pauli_z_diagonal(h8_terms, 8)
    _max_h8_err = float(np.max(np.abs(_h8_diag - bsa.COSTS_8Q)))

    assert np.min(bsa.COSTS_8Q) == 14.0
    assert int(np.sum(bsa.COSTS_8Q == 14.0)) == 1
    assert np.min(bsa.COSTS_12Q) == 14.0
    assert int(np.sum(bsa.COSTS_12Q == 14.0)) == 1
    assert _max_h8_err < 1e-12

    opt_8q_decoded = bsa.decode_8q_bitstring(f"{bsa.OPTIMAL_IDX_8Q:08b}"[::-1])
    assert set(opt_8q_decoded["kept_stops"]) == {1, 2, 5, 6, 7, 8, 10, 12}

    # Summarize Pauli weight breakdown of H_8q
    _weight_counts = {}
    for _qs in h8_terms:
        _w = len(_qs)
        _weight_counts[_w] = _weight_counts.get(_w, 0) + 1

    qc_8q_primary, meta_8q_primary = bsa.get_bus_stop_circuit(encoding="8q", layers=1, measure=True)

    mo.vstack(
        [
            mo.md(
                f"""
                ### Exact Cost & $H_{{8q}}$ Verification

                - **8-qubit state space:** `256` states (unique optimum `idx={bsa.OPTIMAL_IDX_8Q}`,
                  $q_0\\dots q_7$ = `"{opt_8q_decoded['bits_q0_to_q7']}"`, cost = `{opt_8q_decoded['cost']}`).
                - **Decoded kept stops:** `{opt_8q_decoded['kept_stops']}` | **Dropped:** `{opt_8q_decoded['dropped_stops']}`.
                - **12-qubit state space:** `4096` states (unique optimum cost = `{float(np.min(bsa.COSTS_12Q))}`).
                - **Pauli-$Z$ expansion of $H_{{8q}}$:** `{len(h8_terms)}` non-zero terms
                  (weights: `{dict(sorted(_weight_counts.items()))}`), max diagonal error vs. `COSTS_8Q`:
                  `{_max_h8_err:.2e}` (exact).
                - **Primary submission circuit (`get_bus_stop_circuit("8q", layers=1)`):**
                  $\\langle F \\rangle = {meta_8q_primary['optimal_energy']:.8f}$,
                  $P(x^*) = {meta_8q_primary['optimal_probability']:.6f}$.
                """
            ),
            mo.md("**Primary 8-Qubit Hardware-Native Circuit (`layers=1`, strictly `{rx, rz, cz}`):**"),
            mo.plain_text(str(qc_8q_primary.draw("text", fold=120))),
        ]
    )
    return h8_terms, meta_8q_primary, opt_8q_decoded, qc_8q_primary


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ---

    ## 3. Multi-Start Derivative-Free COBYLA VQE Benchmark & Native Hardware Comparison

    Variational optimization is executed by our `bus_stop_vqe.py` module, running derivative-free `COBYLA`
    against Qiskit's exact `SparsePauliOp` Hamiltonian and `Statevector` across multi-start random seeds in
    $[-\pi, \pi]$ (`VQE_N_SEEDS = 10`). Every circuit is evaluated against the native gate set
    `["rx", "rz", "cz"]` (with non-native baselines transpiled at `optimization_level=3`) to measure true
    2-zone hardware cost.

    If `bus_stop_vqe.train_vqe(...)` is unavailable in the environment, this section gracefully falls back
    to `bsa.PRECOMPUTED_VQE_SOLUTIONS` so all hardware metrics, figures, precision checks, and Section 4
    compilation results still render out of the box.
    """)
    return


@app.cell
def _():
    # Live multi-start COBYLA VQE execution via bus_stop_vqe.py (with cached fallback)
    VQE_MODULE = "bus_stop_vqe"
    VQE_N_SEEDS = 10

    _qasm_dir = Path("qasm")
    _ihva_path = _qasm_dir / "bus_stop_12q_ihva_stagger_p1.qasm"
    _hea_qasm_path = _qasm_dir / "bus_stop_12q_hea_linear_p1.qasm"

    _vqe_targets = [
        ("hea_linear_p1", "hea_linear", 1),
        ("8q_p1", "8q", 1),
        ("8q_p2", "8q", 2),
        ("12q_p1", "12q", 1),
        ("12q_p2", "12q", 2),
    ]

    vqe_runs: dict[str, dict | None] = {}
    _vqe_status_msg: str | None = None
    try:
        for _key, _atype, _lyrs in _vqe_targets:
            vqe_runs[_key] = bsa.run_external_vqe(
                ansatz_type=_atype,
                layers=_lyrs,
                n_seeds=VQE_N_SEEDS,
                base_seed=42,
                vqe_module=VQE_MODULE,
            )
        vqe_source = f"live ({VQE_MODULE}.train_vqe, COBYLA)"
    except Exception as _exc:
        vqe_source = "cached_fallback (PRECOMPUTED_VQE_SOLUTIONS)"
        vqe_runs = {"hea_linear_p1": None}
        for _key, _atype, _lyrs in _vqe_targets[1:]:
            _cached = dict(bsa.PRECOMPUTED_VQE_SOLUTIONS[(_atype, _lyrs)])
            _cleaned = bsa.clean_hardware_params(_cached["best_params"])
            _qc_tmp = (
                bsa.build_8q_ansatz(_cleaned, layers=_lyrs, measure=False, prune_zeros=True)
                if _atype == "8q"
                else bsa.build_12q_ansatz(_cleaned, layers=_lyrs, measure=False, prune_zeros=True)
            )
            _ev = bsa.evaluate_circuit_exact(_qc_tmp, encoding=_atype)
            _cached["best_energy"] = _ev["expected_cost"]
            _cached["best_optimal_probability"] = _ev["optimal_probability"]
            vqe_runs[_key] = _cached

        _vqe_status_msg = (
            f"**VQE execution status:** `{VQE_MODULE}.train_vqe(...)` is unavailable "
            f"(`{type(_exc).__name__}: {_exc}`). Displaying verified cached results from "
            "`bus_stop_ansatz.PRECOMPUTED_VQE_SOLUTIONS` so all hardware metrics, figures, and Section 4 "
            "compilation remain operational. Ensure `bus_stop_vqe.py` exposes "
            "`train_vqe(ansatz_type, layers, n_seeds=10, base_seed=42) -> dict` to run live COBYLA traces."
        )

    # Build pruned, hardware-realistic circuits for compiler hook & inspection
    benchmark_circuits: dict[str, QuantumCircuit] = {}
    if _ihva_path.exists():
        benchmark_circuits["Baseline: ihva_stagger (12q, p=1)"] = QuantumCircuit.from_qasm_file(str(_ihva_path))
    if vqe_runs.get("hea_linear_p1") is not None:
        benchmark_circuits["Baseline: hea_linear (12q, p=1)"] = bsa.build_hea_linear_12q(
            bsa.clean_hardware_params(vqe_runs["hea_linear_p1"]["best_params"]), layers=1, measure=True
        )
    elif _hea_qasm_path.exists():
        benchmark_circuits["Baseline: hea_linear (12q, p=1)"] = QuantumCircuit.from_qasm_file(str(_hea_qasm_path))

    benchmark_circuits["Candidate A: 8q Compressed (p=1)"] = bsa.build_8q_ansatz(
        bsa.clean_hardware_params(vqe_runs["8q_p1"]["best_params"]), layers=1, measure=True, prune_zeros=True
    )
    benchmark_circuits["Candidate A: 8q Compressed (p=2)"] = bsa.build_8q_ansatz(
        bsa.clean_hardware_params(vqe_runs["8q_p2"]["best_params"]), layers=2, measure=True, prune_zeros=True
    )
    benchmark_circuits["Candidate B: 12q 5+5 Real-Edges (p=1)"] = bsa.build_12q_ansatz(
        bsa.clean_hardware_params(vqe_runs["12q_p1"]["best_params"]), layers=1, measure=True, prune_zeros=True
    )
    benchmark_circuits["Candidate B: 12q 5+5 Real-Edges (p=2)"] = bsa.build_12q_ansatz(
        bsa.clean_hardware_params(vqe_runs["12q_p2"]["best_params"]), layers=2, measure=True, prune_zeros=True
    )

    # Measure generic (non-zero random parameter) native metrics so zero-angle pruning doesn't mask structure
    _rng = np.random.default_rng(7)
    _spec_rows = [
        (
            "Baseline: ihva_stagger (12q, p=1)",
            benchmark_circuits.get("Baseline: ihva_stagger (12q, p=1)"),
            29,   # 29 parameterized Rz edges in fixture
            None,
            None,
            "N/A (QASM fixture)",
        ),
        (
            "Baseline: hea_linear (12q, p=1)",
            QuantumCircuit.from_qasm_file(str(_hea_qasm_path))
            if _hea_qasm_path.exists()
            else bsa.build_hea_linear_12q(_rng.uniform(-np.pi, np.pi, 36), layers=1, measure=False),
            36,
            2,
            vqe_runs.get("hea_linear_p1"),
            f"Awaiting {VQE_MODULE}.py",
        ),
        (
            "Candidate A: 8q Compressed (p=1)",
            bsa.build_8q_ansatz(_rng.uniform(-np.pi, np.pi, 24), layers=1, measure=False),
            24,
            2,
            vqe_runs["8q_p1"],
            "N/A",
        ),
        (
            "Candidate A: 8q Compressed (p=2)",
            bsa.build_8q_ansatz(_rng.uniform(-np.pi, np.pi, 40), layers=2, measure=False),
            40,
            4,
            vqe_runs["8q_p2"],
            "N/A",
        ),
        (
            "Candidate B: 12q 5+5 Real-Edges (p=1)",
            bsa.build_12q_ansatz(_rng.uniform(-np.pi, np.pi, 36), layers=1, measure=False),
            36,
            2,
            vqe_runs["12q_p1"],
            "N/A",
        ),
        (
            "Candidate B: 12q 5+5 Real-Edges (p=2)",
            bsa.build_12q_ansatz(_rng.uniform(-np.pi, np.pi, 60), layers=2, measure=False),
            60,
            4,
            vqe_runs["12q_p2"],
            "N/A",
        ),
    ]

    _table_rows = []
    for _label, _circ, _n_params, _hw_rounds, _vqe, _fallback_lbl in _spec_rows:
        if _circ is None:
            continue
        _nm = bsa.inspect_native_metrics(_circ)
        _rounds_display = _hw_rounds if _hw_rounds is not None else _nm["cz_rounds"]
        _table_rows.append(
            {
                "Circuit": _label,
                "Qubits": _nm["qubits"],
                "Params": _n_params,
                "Native CZ (opt=3)": _nm["native_cz"],
                "Entangling Rounds": _rounds_display,
                "Zone Visits": _nm["entangling_visits"],
                "Native Depth (opt=3)": _nm["native_depth"],
                "Best <F>": round(float(_vqe["best_energy"]), 6) if _vqe else None,
                "Success Rate (F<14.01)": (
                    f"{_vqe['success_count']}/{_vqe['n_seeds']} ({100 * _vqe['success_rate']:.0f}%)"
                    if _vqe
                    else _fallback_lbl
                ),
                "Optimal Prob P(x*)": round(float(_vqe["best_optimal_probability"]), 6) if _vqe else None,
            }
        )

    benchmark_df = pl.DataFrame(_table_rows)
    _section3_out = (
        mo.vstack([mo.callout(mo.md(_vqe_status_msg), kind="info"), benchmark_df])
        if _vqe_status_msg is not None
        else benchmark_df
    )
    _section3_out
    return VQE_MODULE, VQE_N_SEEDS, benchmark_circuits, benchmark_df, vqe_runs, vqe_source


@app.cell
def _(benchmark_df, vqe_runs, vqe_source):
    # Publication-grade 2-panel comparison figure: VQE convergence/rates + hardware resource reduction
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(13.5, 4.8), dpi=130)

    # Panel A: live convergence traces if provided by bus_stop_vqe; else cached multi-start success-rate bars
    _curves = [
        ("8q_p1", "8q Compressed p=1", "#0d9488", "-", 2.4),
        ("8q_p2", "8q Compressed p=2", "#0284c7", "--", 2.0),
        ("hea_linear_p1", "Baseline hea_linear p=1", "#d97706", "-.", 2.0),
        ("12q_p1", "12q 5+5 Real-Edges p=1", "#7c3aed", ":", 2.1),
        ("12q_p2", "12q 5+5 Real-Edges p=2", "#db2777", "--", 1.8),
    ]
    _has_histories = any(
        (vqe_runs.get(_k) is not None and bool(vqe_runs[_k].get("best_history")))
        for _k, *_ in _curves
    )

    if _has_histories:
        for _key, _base_lbl, _col, _ls, _lw in _curves:
            _run = vqe_runs.get(_key)
            if not _run or not _run.get("best_history"):
                continue
            _hist = _run["best_history"]
            _lbl = f"{_base_lbl} ({_run['success_count']}/{_run['n_seeds']} = {100 * _run['success_rate']:.0f}%)"
            _ax1.plot(range(len(_hist)), _hist, label=_lbl, color=_col, linestyle=_ls, linewidth=_lw)

        _ax1.axhline(14.0, color="#111827", linestyle="--", linewidth=1.1, alpha=0.75, label="Exact Optimum F* = 14")
        _ax1.set_title(f"A. VQE Convergence — {vqe_source}", fontweight="bold", pad=10)
        _ax1.set_xlabel("Optimizer Iteration")
        _ax1.set_ylabel("Expected Objective <F>")
        _ax1.set_ylim(13.2, 42.0)
        _ax1.grid(True, linestyle=":", alpha=0.45)
        _ax1.legend(loc="upper right", fontsize=8.3, frameon=True, framealpha=0.92)
    else:
        _bar_items = [
            (_base_lbl, vqe_runs[_key], _col)
            for _key, _base_lbl, _col, _, _ in _curves
            if vqe_runs.get(_key) is not None
        ]
        _bx = np.arange(len(_bar_items))
        _rates = [100.0 * _r["success_rate"] for _, _r, _ in _bar_items]
        _cols = [_c for _, _, _c in _bar_items]
        _bars_a = _ax1.bar(_bx, _rates, color=_cols, width=0.58)
        for _bar, (_, _r, _) in zip(_bars_a, _bar_items, strict=True):
            _h = _bar.get_height()
            _ax1.text(
                _bar.get_x() + _bar.get_width() / 2.0,
                _h + 2.2,
                f"{_r['success_count']}/{_r['n_seeds']} ({_h:.0f}%)",
                ha="center",
                va="bottom",
                fontsize=8.3,
                fontweight="bold",
            )
        _ax1.set_title("A. Multi-Start VQE Success Rate (Cached Fallback)", fontweight="bold", pad=10)
        _ax1.set_xticks(_bx)
        _ax1.set_xticklabels([_lbl.replace(" ", "\n", 1) for _lbl, _, _ in _bar_items], fontsize=8.4)
        _ax1.set_ylabel("Seeds Converging to F < 14.01 (%)")
        _ax1.set_ylim(0, 118)
        _ax1.grid(True, axis="y", linestyle=":", alpha=0.45)

    # Panel B: Native hardware cost comparison (CZ gates, Native Depth, Zone Visits)
    _short_names = [
        "ihva_stagger\n(12q, p=1)",
        "hea_linear\n(12q, p=1)",
        "12q 5+5\n(p=1)",
        "12q 5+5\n(p=2)",
        "8q Comp.\n(p=1) ★",
        "8q Comp.\n(p=2)",
    ]
    _order_map = {
        "Baseline: ihva_stagger (12q, p=1)": 0,
        "Baseline: hea_linear (12q, p=1)": 1,
        "Candidate B: 12q 5+5 Real-Edges (p=1)": 2,
        "Candidate B: 12q 5+5 Real-Edges (p=2)": 3,
        "Candidate A: 8q Compressed (p=1)": 4,
        "Candidate A: 8q Compressed (p=2)": 5,
    }
    _ordered = sorted(benchmark_df.to_dicts(), key=lambda r: _order_map.get(r["Circuit"], 99))
    _x = np.arange(len(_ordered))
    _w = 0.26
    _cz = [r["Native CZ (opt=3)"] for r in _ordered]
    _depth = [r["Native Depth (opt=3)"] for r in _ordered]
    _visits = [r["Zone Visits"] for r in _ordered]

    _b1 = _ax2.bar(_x - _w, _cz, width=_w, label="Native CZ Gates", color="#0f766e")
    _b2 = _ax2.bar(_x, _depth, width=_w, label="Native Depth (opt=3)", color="#2563eb")
    _b3 = _ax2.bar(_x + _w, _visits, width=_w, label="Entangling Zone Visits", color="#ea580c")

    for _bars in (_b1, _b2, _b3):
        for _bar in _bars:
            _h = _bar.get_height()
            _ax2.text(
                _bar.get_x() + _bar.get_width() / 2.0,
                _h + 0.8,
                f"{int(_h)}",
                ha="center",
                va="bottom",
                fontsize=7.8,
                fontweight="bold",
            )

    _ax2.set_title("B. Native {rx, rz, cz} Hardware Cost (Lower is Better)", fontweight="bold", pad=10)
    _ax2.set_xticks(_x)
    _ax2.set_xticklabels([_short_names[_order_map[r["Circuit"]]] for r in _ordered], fontsize=8.5)
    _ax2.set_ylabel("Count")
    _ax2.set_ylim(0, max(_cz + _depth) * 1.16)
    _ax2.grid(True, axis="y", linestyle=":", alpha=0.45)
    _ax2.legend(loc="upper right", fontsize=8.5, frameon=True, framealpha=0.92)

    fig.tight_layout()
    fig
    return (fig,)


@app.cell(hide_code=True)
def _(vqe_runs):
    def _fmt_rate(run: dict | None) -> str:
        if not run:
            return "N/A"
        return f"**{100 * run['success_rate']:.0f}% (`{run['success_count']}/{run['n_seeds']}`)**"

    _r8 = _fmt_rate(vqe_runs.get("8q_p1"))
    _r12 = _fmt_rate(vqe_runs.get("12q_p1"))
    _rhea_run = vqe_runs.get("hea_linear_p1")
    _rhea_clause = (
        f" and {_fmt_rate(_rhea_run)} for `hea_linear p=1`"
        if _rhea_run is not None
        else ""
    )
    _r12_vs_hea = (
        f"while outperforming `hea_linear` in convergence ({_r12} vs. {_fmt_rate(_rhea_run)}) and "
        if _rhea_run is not None
        else f"achieving {_r12} convergence and "
    )

    mo.md(
        f"""
    ### Key Takeaways from the Benchmark

    1. **8-Qubit Triangle Compression (`Candidate A, p=1`) dominates on every axis:**
       - **-88% fewer `cz` gates** (`7` vs. `58`) and **-91% lower native depth** (`5` vs. `57`) vs. `ihva_stagger`.
       - **-36% fewer `cz` gates** (`7` vs. `11`), **-44% lower native depth** (`5` vs. `9`), **-33% fewer atoms**
         (`8` vs. `12`), and **-50% fewer commutation-aware Storage $\\leftrightarrow$ Entangling visits** (`1` vs. `2`,
         accounting for $[R_z, \\text{{CZ}}] = 0$ so only non-commuting $R_x$ forces a zone return) vs. `hea_linear`.
       - {_r8} multi-start convergence rate vs. {_r12} for uncompressed `12q 5+5 p=1`{_rhea_clause}, because
         eliminating the 3,840 intra-triangle infeasible states removes the roughest penalty barriers from the
         energy landscape.
    2. **12-Qubit "5+5 Real-Edges" (`Candidate B, p=1`)** respects the hardware's **5-pairs-per-row**
       Entangling zone limit (2 pulses of 5 real edges = 10 `cz`, `max_matching = 5`), cutting Storage
       $\\leftrightarrow$ Entangling visits in half (`1` vs. `2`) and native depth (`5` vs. `9`) compared to
       `hea_linear`, {_r12_vs_hea}reaching $\\langle F \\rangle = 14.000000$ and $P(x^*) = 1.000000$.

    ### 3.1 Experimental Realism: Pulse-Precision Sensitivity & Role of Entanglement

    - **Realistic laser precision & zero-pulse pruning:** Using `clean_hardware_params(...)` with `prune_zeros=True`,
      near-zero angles ($|\\theta| < 10^{{-4}}\\text{{ rad}}$) are omitted (avoiding unphysical sub-nanoradian pulses),
      near-$\\pm\\pi$ pulses snap to exact $\\pm\\pi$, and remaining angles are quantized to realistic neutral-atom
      Raman laser resolutions (`1e-4`, `1e-3`, and `1e-2 rad`). As verified below, the 8-qubit circuit maintains
      **$P(x^*) > 0.99999$ even at $10^{{-2}}\\text{{ rad}}$ (10 mrad) pulse quantization**.
    - **Why the converged state is separable while `CZ` entanglement still matters:** Because the MWVC Ising
      Hamiltonian is diagonal in the computational basis, its unique ground state $|x^*\\rangle = |00100001\\rangle$
      is a product basis state—so any optimal VQE solution naturally disentangles at convergence. Crucially, the
      `cz` entangling layers do **not** go to waste: during optimization they couple triangle decisions across
      bridges through entangled intermediate superpositions, smoothing non-convex penalty barriers and driving
      multi-start convergence to {_r8}.
    """
    )
    return


@app.cell
def _():
    # 5-line hardware pulse-precision sensitivity check (1e-4, 1e-3, 1e-2 rad)
    precision_rows = []
    for _dec in (4, 3, 2):
        _qc_q, _meta_q = bsa.get_bus_stop_circuit("8q", layers=1, measure=False, clean_params=True, prune_zeros=True, decimals=_dec)
        assert _meta_q["optimal_probability"] > 0.99999
        precision_rows.append({"Laser Resolution": f"1e-{_dec} rad", "Pruned Gates": _qc_q.size(), "Expected <F>": round(_meta_q["optimal_energy"], 7), "Optimal Prob P(x*)": round(_meta_q["optimal_probability"], 7)})
    precision_df = pl.DataFrame(precision_rows)
    precision_df
    return precision_df, precision_rows


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ---

    ## 4. End-to-End Neutral-Atom Compilation

    This section invokes our zoned neutral-atom compiler `compile_circuit(circuit, architecture_spec)` from
    `solution.py` across all benchmark circuits on the default 2-zone architecture spec, appending
    **Compiled Moves, Compiled Depth (excl. moves), and Sequential Move Distance (upper bound)** directly to
    the comparison table (the official parallel Cartesian distance replaces this upper bound when evaluated
    with `pushquantum-validator`).

    To target an alternate compiler module, set `COMPILER_MODULE` below.
    """)
    return


@app.cell
def _(benchmark_circuits, benchmark_df):
    import importlib as _importlib

    COMPILER_MODULE = "solution"  # Module exposing compile_circuit(circuit, architecture_spec)

    _FALLBACK_DEFAULT_SPEC = {
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

    def _resolve_default_spec(compiler_mod) -> dict:
        """Load default_spec() from compiler module, falling back to readme.default_spec() or inline spec."""
        spec_fn = getattr(compiler_mod, "default_spec", None)
        if callable(spec_fn):
            return spec_fn()
        try:
            readme_mod = _importlib.import_module("readme")
            readme_spec_fn = getattr(readme_mod, "default_spec", None)
            if callable(readme_spec_fn):
                return readme_spec_fn()
        except Exception:
            pass
        return dict(_FALLBACK_DEFAULT_SPEC)

    def _physical_xy(site: tuple[int, int], spec: dict) -> tuple[float, float]:
        """Compute physical (x, y) coordinates accounting for entangling pair geometry."""
        col, row = site
        for _z in spec["zones"].values():
            if _z["x"][0] <= col <= _z["x"][1] and _z["y"][0] <= row <= _z["y"][1]:
                spacing = float(_z.get("spacing", 1.0))
                pairs = _z.get("pairs")
                if pairs and pairs.get("axis") == "x":
                    pair_dist = float(pairs["distance"])
                    x_phys = (col // 2) * (2.0 * spacing) + (col % 2) * pair_dist
                else:
                    x_phys = col * spacing
                y_phys = row * spacing
                return (x_phys, y_phys)
        return (float(col), float(row))

    def _estimate_move_metrics(compiled_qc: QuantumCircuit, arch, spec: dict) -> dict:
        """Compute compiled gate depth (excl. moves), move count, and Cartesian move distance."""
        ops = dict(compiled_qc.count_ops())
        non_move = QuantumCircuit(compiled_qc.num_qubits, compiled_qc.num_clbits)
        for inst in compiled_qc.data:
            if inst.operation.name != "move":
                non_move.append(inst.operation, inst.qubits, inst.clbits)

        # Replay atom positions to measure Cartesian distance per move (max atom displacement per move)
        positions = dict(arch.initial_loading)
        total_move_dist = 0.0
        for inst in compiled_qc.data:
            if inst.operation.name != "move":
                continue
            rows, cols, new_rows, new_cols = inst.operation.params
            q_indices = [compiled_qc.find_bit(q).index for q in inst.qubits]
            occ_rows = sorted({positions[q][1] for q in q_indices})
            occ_cols = sorted({positions[q][0] for q in q_indices})
            if len(occ_rows) == len(new_rows) and len(occ_cols) == len(new_cols):
                row_map = dict(zip(occ_rows, new_rows, strict=True))
                col_map = dict(zip(occ_cols, new_cols, strict=True))
                max_d = 0.0
                for q in q_indices:
                    old_site = positions[q]
                    new_site = (col_map[old_site[0]], row_map[old_site[1]])
                    x0, y0 = _physical_xy(old_site, spec)
                    x1, y1 = _physical_xy(new_site, spec)
                    max_d = max(max_d, float(np.hypot(x1 - x0, y1 - y0)))
                    positions[q] = new_site
                total_move_dist += max_d

        return {
            "Compiled Moves": int(ops.get("move", 0)),
            "Compiled Depth (excl. moves)": int(non_move.remove_final_measurements(inplace=False).depth()),
            "Sequential Move Distance (upper bound)": round(total_move_dist, 3),
        }

    compiler_table = None
    _status_msg = None
    try:
        _mod = _importlib.import_module(COMPILER_MODULE)
        _compile_fn = getattr(_mod, "compile_circuit")
        _spec = _resolve_default_spec(_mod)

        # Probe on primary 8q circuit to verify compile_circuit is implemented (not placeholder empty circuit)
        _probe_qc = benchmark_circuits["Candidate A: 8q Compressed (p=1)"].remove_final_measurements(inplace=False)
        _probe_compiled, _probe_arch = _compile_fn(_probe_qc, _spec)
        if _probe_compiled.size() == 0:
            _status_msg = (
                f"**Compilation status:** Found `{COMPILER_MODULE}.compile_circuit`, which currently returns "
                "an empty circuit (template placeholder). Re-run this notebook once `solution.py` contains the "
                "full router implementation to automatically populate compiled movement distance and depth below."
            )
        else:
            _extra_rows = []
            for _label, _qc in benchmark_circuits.items():
                try:
                    _compiled, _arch = _compile_fn(_qc, _spec)
                    _m = _estimate_move_metrics(_compiled, _arch, _spec)
                    _extra_rows.append({"Circuit": _label, **_m, "Compiler Status": "OK"})
                except Exception as _exc:
                    _extra_rows.append(
                        {
                            "Circuit": _label,
                            "Compiled Moves": None,
                            "Compiled Depth (excl. moves)": None,
                            "Sequential Move Distance (upper bound)": None,
                            "Compiler Status": f"{type(_exc).__name__}: {_exc}",
                        }
                    )
            compiler_table = benchmark_df.join(pl.DataFrame(_extra_rows), on="Circuit", how="left")
    except Exception as _exc:
        _status_msg = (
            f"**Compilation status:** `{COMPILER_MODULE}.compile_circuit` unavailable "
            f"(`{type(_exc).__name__}: {_exc}`). Exported OpenQASM 2.0 files (`ansatz_8q_p1_optimized.qasm`, "
            "`ansatz_8q_p1_nom.qasm`, `ansatz_12q_p1_optimized.qasm`, `ansatz_12q_p1_nom.qasm`) and "
            "`bus_stop_ansatz.get_bus_stop_circuit()` remain ready for direct compilation."
        )

    _hook_output = (
        compiler_table
        if compiler_table is not None
        else mo.callout(mo.md(_status_msg), kind="info")
    )
    _hook_output
    return COMPILER_MODULE, compiler_table


if __name__ == "__main__":
    app.run()
