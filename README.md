<!--
Copyright 2026 Hongpei Li
SPDX-License-Identifier: Apache-2.0
-->

# PreFOS: A Presolver for First-Order Solvers

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Build and Test Status](https://img.shields.io/github/actions/workflow/status/Lhongpei/PreFOS/cmake.yml?branch=main)](https://github.com/Lhongpei/PreFOS/actions)
[![Language: C](https://img.shields.io/badge/Language-C-blue.svg)](https://en.wikipedia.org/wiki/C_(programming_language))


**PreFOS** is a dependency-free C presolver designed for first-order
optimization solvers. It treats linear programming as a special case of the
following box-and-cone quadratic model:

```math
\begin{aligned}
\min_x \quad & \frac{1}{2}x^\top (Q + R^\top D R)x + c^\top x \\
\text{s.t.} \quad & \ell_c \le Ax \le u_c, \\
                   & \ell_v \le x_I \le u_v, \\
                   & x_{J_k} \in \mathcal K_k, \quad k=1,\ldots,N_K, \\
                   & G_r x + h_r \in \widehat{\mathcal K}_r,
                     \quad r=1,\ldots,N_A.
\end{aligned}
```

The box set `I` and cone blocks `J_k` are pairwise disjoint and partition the
direct variable domains. Affine cone rows may involve variables from any direct
domain. A direct block `x_J in K` remains a projection-friendly primitive; it
is the selector-matrix special case of an affine cone, but is not expanded
unless a reduction benefits from doing so. PreFOS provides a CPU implementation
and an experimental optional CUDA backend for bulk linear propagation. Both
backends use the same bound acceptance and postsolve semantics.

---
## Build

PreFOS is built with CMake and has no runtime dependencies:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DPREFOS_BUILD_TESTING=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

CUDA propagation is opt-in and does not change the default dependency-free
build. For example, an NVIDIA Hopper build uses:

```bash
cmake -S . -B build-gpu -DCMAKE_BUILD_TYPE=Release \
  -DPREFOS_BUILD_TESTING=ON -DPREFOS_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build build-gpu -j
```

Set `linear_propagation_gpu = 1` to request it. A non-CUDA build or CUDA runtime
failure falls back to the CPU bulk engine. Sparse event-driven propagation
always remains on CPU. CUDA 11.2 or newer is required for the asynchronous
allocation pool. `prefos_gpu_warmup()` can initialize the primary context
before a timed solve, while `prefos_gpu_release_cache()` trims cached device
allocations when no presolve call is active.

The public C API is defined in `include/PreFOS/PreFOS.h`. Installed CMake
packages expose the `PreFOS::PreFOS` target:

```cmake
find_package(PreFOS CONFIG REQUIRED)
target_link_libraries(YOUR_PROJECT PRIVATE PreFOS::PreFOS)
```

---
## APIs

PreFOS follows an opaque-presolver lifecycle:

1. `prefos_create_presolver()` validates and deep-copies the input model.
2. `prefos_run_presolve()` executes reductions and exposes the reduced model.
3. `prefos_postsolve_primal()` or `prefos_postsolve_primal_dual()` maps a reduced
   solution back to the original model.
4. Verification APIs independently audit primal equivalence and KKT recovery.

---
## Lineage and Citation

PreFOS continues the work of PSLP, the lightweight LP presolver by Daniel
Cederberg and Stephen Boyd. LP is represented directly as a special case of
the unified PreFOS model; PreFOS does not ship a separate legacy LP API. A
PreFOS paper is under preparation. Until a paper citation is available, please
cite the software directly:

```bibtex
@software{Li2026PreFOS,
  author = {Li, Hongpei},
  title = {{PreFOS}: A Presolver for First-Order Solvers},
  year = {2026},
  version = {0.1.0},
  url = {https://github.com/Lhongpei/PreFOS},
  license = {Apache-2.0}
}
```

PreFOS is distributed under the Apache License 2.0. See [LICENSE](LICENSE)
for the license terms and [NOTICE](NOTICE) for PSLP lineage and upstream
attribution.

---
## Presolve Capabilities

PreFOS provides a lightweight, solver-oriented presolve pipeline:

- Linear reductions include validation, fixed-variable elimination, singleton
  tightening, activity propagation, redundant-row removal, parallel-row
  reduction, and bounded equality aggregation.
- Cone reductions cover nonnegative, SOC, rotated SOC, PSD, exponential, and
  power cones, including conservative envelope propagation, exact exposed-face
  reductions, and affine-cone structure analysis.
- Objective handling preserves sparse `Q + R^T D R` structure through
  elimination and compaction.
- Ordered transformation records support primal, primal-dual, and extended-dual
  postsolve together with independent equivalence and KKT verification.
- Linear propagation uses a hybrid event-driven and bulk CPU engine, with an
  optional CUDA bulk backend.

The full reduction contracts, settings, GPU design, correctness harnesses, and
benchmark entry points are documented in
[docs/PRESOLVE.md](docs/PRESOLVE.md).
