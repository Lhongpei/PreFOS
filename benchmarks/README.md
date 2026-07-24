<!--
Copyright 2026 Hongpei Li
SPDX-License-Identifier: Apache-2.0
-->

# Benchmarks

PreFOS ships benchmark drivers so performance and reduction claims can be
reproduced without committing machine-specific output or third-party datasets.

The drivers are located in `tests/`:

- `benchmark_cbf_prefos.py` measures PreFOS on CBF/SOCP models parsed through
  PDHCG-II.
- `benchmark_exp_power_cbf.py` measures native exponential- and power-cone CBF
  models.
- `benchmark_sdpa_prefos.py` measures PSD structure and propagation on sparse
  SDPA models.
- `benchmark_clarabel_gurobi.py` compares external presolve pipelines with
  Clarabel as the common downstream solver.
- `benchmark_lp_pslp.py` compares CPU-only PreFOS and upstream PSLP on the
  same LP matrix and bounds. It reports constructor and presolve time
  separately and records the reduced row, column, and nonzero counts.

Store downloaded instances under `benchmarks/data/` and generated JSONL,
tables, logs, and dated reports under `benchmarks/results/`. These directories
are intentionally ignored. A result intended for publication should record the
hardware, compiler and build flags, software versions, thread counts, timing
scope, solver tolerances, run count, and input provenance.

Third-party solvers and datasets remain subject to their own licenses. In
particular, confirm that the applicable solver license permits publication of
comparative benchmark results before committing a report.
