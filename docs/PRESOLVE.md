<!--
Copyright 2026 Hongpei Li
SPDX-License-Identifier: Apache-2.0
-->

# PreFOS Presolve Reference

This document describes the current presolve pipeline, reduction contracts,
postsolve semantics, correctness checks, and benchmark entry points.

[Back to README](../README.md)

## Capabilities

PreFOS provides input validation,
nonnegative-cone canonicalization, fixed-variable elimination, singleton-row
bound tightening, iterative linear activity propagation, cone envelope
propagation, bounded free-column equality aggregation,
activity-based redundant-row removal, parallel-row intersection,
empty-row detection/removal,
conservative zero-cone collapse, factored `Q + R^T D R` objective updates,
sparse compaction, statistics, and primal/primal-dual postsolve recovery.
Affine PSD images additionally receive exact sparsity analysis and block-diagonal
component decomposition.
Linear propagation commits bounds to box variables, including input
nonnegative-cone variables canonicalized to `[0,+inf]`; coupled cone variables
contribute safe scalar envelopes such as `t >= 0` for SOC blocks. Exponential
and three-dimensional power cones are preserved through compaction and
postsolve, with independent primal- and dual-cone verification. The public
API is defined in `include/PreFOS/PreFOS.h` and builds as the `PreFOS` target.

## Architecture

`src/prefos/PREFOS_Presolver.c`
only coordinates reductions and sparse compaction, `src/prefos/explorers/`
contains structural and propagation engines, and `src/prefos/core/` owns model
validation/copying plus postsolve and verification. Cross-module contracts are
private to `src/prefos` and are not part of the installed C API.

## Linear Presolve

PreFOS constructs a model-independent linear row/domain view in
`include/common/LinearPropagationKernel.h`. The core does not dispatch on an LP
or Conic QP problem type: an LP is the case with no quadratic terms, no cone
blocks, and every variable eligible for scalar bound tightening. Activity
construction, row classification, implied-bound propagation, and dirty-row
scheduling are shared. Bound and coefficient updates feed the same
current/next-round state machine for every model. The core is internal and
header-only so Release builds can inline its hot row loops.

Propagation postsolve payloads also use the shared typed transformation log in
`include/common/TransformationLog.h`. API-specific postsolve code may apply its
own dual sign convention, but it consumes the same source-row, bound, direction,
and variable records. Bound, row, cone-collapse, and facial-reduction records
append atomically to one ordered event stream. PreFOS postsolve replays that
stream in reverse, so cone dual repair, propagated-bound transfer,
and parallel-row multiplier recovery follow the actual opposite order of
presolve. Consecutive propagated-bound events are replayed to a local fixed
point because one bound transfer can activate another inferred bound in the
same block. Face events preserve the existing extended-dual certificate
semantics and do not pretend that a finite standard cone multiplier exists.

Parallel-row detection uses the stride-aware sparse row
view in `include/common/ParallelRowDetection.h`. PreFOS CSR rows pass through
normalized hashing, radix sort,
and an exact support/proportionality check. The reduction then uses
`include/common/ParallelRowReduction.h` to select the retained row, intersect
scaled intervals, handle negative ratios, and identify the source row for each
tightened side. Numerically ambiguous interval
crossings within tolerance are left unreduced instead of being declared
infeasible.

Input nonnegative cones are componentwise equivalent to lower-bounded boxes,
so the constructor moves them into the box partition of its internal copy.
This enables partial propagation and elimination without a separate cone-block
reduction path; the caller's input arrays are unchanged, and surviving
components appear as boxes in the reduced model. Subsequent tightening and
fixing of these components is included in the box-related statistics.

Redundant rows are removed when an outward-rounded activity interval from the
current exposed box bounds and intrinsic cone envelopes lies completely inside
the row bounds. This check runs before row-derived cone envelopes are extracted,
so a row cannot use a temporary envelope that it created to prove itself
redundant. Removed redundant rows recover a zero multiplier during postsolve.

Linear propagation uses a hybrid CPU engine. Sparse bound-change fan-out
uses cached row activities, box-column adjacency, and a dirty-row queue; dense
box-column structure uses bulk CSR scans instead. The engine also falls back
from queued updates to bulk scans when incremental activity work exceeds its
configured budget. Finite bound changes must pass configurable absolute and
relative improvement thresholds, while `prefos_strict_settings()` sets both
thresholds to zero for equivalence tests.

The default propagation policy also applies a deterministic sparse-work budget.
`linear_propagation_max_work_ratio` bounds activity construction, row scans,
and incremental activity updates relative to `nnz(A)`; a one-million-work-unit
floor lets small propagation chains reach a fixed point. Consecutive rounds
below `linear_propagation_min_changes_per_million` stop after
`linear_propagation_max_stale_rounds`. A zero work ratio or zero stale-round
limit disables that adaptive guard, and `prefos_strict_settings()` disables both.
Statistics expose event/full rounds, work, fallback count, and budget/stale
stop counts.

## Bound Exposure Policies

`PreFOSSettings.propagated_bound_policy` controls how accepted linear-propagation
bounds are exposed to the reduced solver. The default
`PREFOS_PROPAGATED_BOUND_POLICY_FIRST_ORDER` materializes every accepted bound,
which is appropriate when box projection is native and cheap. The
`PREFOS_PROPAGATED_BOUND_POLICY_INTERIOR_POINT` policy still runs the complete
propagation chain and uses it for infeasibility detection, but suppresses a
new finite side when that side is still unbounded in the exposed box. Finite-to-finite
tightening and bounds that fix a variable remain eligible for materialization.
Redundant-row proofs use only materialized box bounds under this policy, so an
unexposed propagation result cannot justify deleting its source constraints.
The two outcomes are counted separately in
`materialized_propagated_box_bounds` and
`suppressed_propagated_box_bounds`.

## Affine Cones

The public model and reduced-model structs support additional affine cone
blocks through `affine_cone_matrix`, `affine_cone_offset`, and
`PreFOSAffineConeBlock`. Rows are contiguous by block and represent
`G*x + h in K`. Direct `PreFOSConeBlock` domains remain available alongside them.

`affine_cone_coordinate_aggregation` is disabled by default. When enabled, a
direct cone block is converted only if every coordinate appears in exactly one
equality, is absent from `Q`, `R`, `c`, and existing affine cone rows, and its
equality contains only a bounded-width box expression. PreFOS removes those
coordinate variables and equalities and emits one affine cone block. Constant
coordinates become entries of `h`; other coordinates become sparse rows of
`G`. Primal postsolve and primal equivalence verification support affine cones.
Cone propagation also forms interval envelopes for input and generated affine
coordinates, applies the regular SOC, RSOC, PSD, exponential, and power rules,
and propagates the result back to variable envelopes. When an intrinsically
nonnegative cone coordinate is an exact singleton affine expression, PreFOS can
also expose the implied one-sided box bound. A compact certificate transfers its
box normal back to the input affine normal, or through the generated
cone-coordinate aggregation map. General nonlinear affine-cone envelopes remain
inference-only until equally explicit dual certificates are available. The
materialized and suppressed singleton outcomes are reported by
`materialized_affine_cone_box_bounds` and
`suppressed_affine_cone_box_bounds`; the first-order/interior-point bound policy
applies to this rule as well.

After variable compaction, PreFOS also removes affine cone coordinates that are
exactly the zero expression. It reduces an RSOC face when one axis and every
tail coordinate are identically zero, removes zero PSD matrix rows and columns,
reduces `x = y = 0` exponential faces to `z >= 0`, and turns a power cone with
`z = 0` into nonnegative axis rows. Entirely zero blocks and zero coordinates
of affine nonnegative blocks are discarded. These deliberately structural
rules require exact zero CSR rows and offsets, so they do not infer hidden
affine equalities. Their duals embed in the original dual cone and are mapped
through both input and aggregation-generated affine blocks by full postsolve.
The existing per-cone face settings control the corresponding affine rules;
statistics report reduced affine faces, coordinates, and blocks separately.

After affine facial reduction, PreFOS analyzes each remaining PSD image in its
row-major `svec` representation. Matrix indices are connected exactly when a
nonzero affine off-diagonal coordinate couples them. Multiple connected
components prove that the image is block diagonal for every `x`, so one PSD
constraint is replaced by the product of its principal PSD blocks and the
identically zero cross-block coordinates are removed. This is an exact sparsity
decomposition, not a chordal relaxation, and introduces no overlap variables.
Scalar PSD components are coalesced into one affine nonnegative block so a
diagonal PSD image remains a single cheap projection rather than many order-one
PSD calls.
Full dual postsolve embeds component normals into the original `svec` vector
with zero cross-block entries. `psd_structure_analysis` and
`psd_block_decomposition` control the analysis and transformation; both are
enabled by default. `prefos_get_psd_structure_analyses()` exposes per-block active
coordinate, coefficient-column, component, and largest-component statistics.

Before general free-column aggregation, PreFOS can also exploit an exposed
affine RSOC face. If either RSOC axis is already the exact zero expression, the
cone implies that every tail affine expression is zero. For a short tail row,
the presolver can fix or substitute a low-degree free box variable that does
not occur in `Q` or `R`. The normal of each derived equality is recorded against
its source cone coordinate and recovered in reverse transformation order.
Because that tail normal is a normal to the exposed face but need not belong to
the unreduced RSOC dual, ordinary full dual recovery reports unavailable;
`prefos_postsolve_full_extended_dual()` and the corresponding extended KKT audit
recover and validate it. `derived_affine_face_equalities`,
`fixed_affine_face_variables`, `substituted_affine_face_variables`, and
`affine_face_substitution_milliseconds` report this pass. Models without affine
RSOC blocks take a fast exit before matrix-column storage is built.

`prefos_postsolve_full_primal_dual()` and
`prefos_verify_postsolve_full_kkt()` accept a separate affine-coordinate normal.
Their convention is `grad(f) + A^T*y + z + G^T*affine_z = 0`, with
`-affine_z` in the dual cone. Aggregated blocks recover the original direct-cone
normal and equality duals. `prefos_postsolve_primal_dual()` has no affine-dual
argument and returns `PREFOS_STATUS_DUAL_RECOVERY_UNAVAILABLE` when affine
blocks are present.

## CUDA Linear Propagation

The experimental CUDA bulk engine uploads the immutable CSR data once and
keeps it resident across propagation rounds; only bounds and reduced candidates
move each round. Rows shorter than 256 entries use one thread per row. Longer
rows are bucketed at setup and use one block per row with CUB block reductions.
The first pass atomically reduces row implications to the strongest candidate
per variable; a second lock-free pass identifies a valid source row for
postsolve. The CPU then applies candidates through the regular tolerance and
transformation-log path. Per-row floating-point error estimates relax device
candidates outward, and suspected infeasibility is rechecked by the CPU. CUDA's
primary context and asynchronous allocation pool persist at process scope; GPU
setup, transfer, kernel, total, long-row, round, and fallback statistics are
exposed for backend comparisons.

## Direct Cone Reductions

For a rotated SOC `(u, v, z)`, an exact singleton proof of `u <= 0` or
`v <= 0` exposes the face `(0, v, 0)` or `(u, 0, 0)`. PreFOS fixes the zero axis
and tail, removes the RSOC block, and emits the surviving axis as a nonnegative
box. Facial certificates record the source row, zero axis, surviving axis, and
original cone indices. Near-zero bounds accepted by the default feasibility
tolerance are not sufficient for this transformation.

For a PSD block \(X \succeq 0\), an exact singleton proof of
\(X_{ii} \le 0\) combines with \(X_{ii} \ge 0\) to prove \(X_{ii}=0\).
Positive semidefiniteness then forces the entire matrix row and column `i` to
zero. PreFOS fixes every affected `svec`
coordinate and replaces the block by the PSD cone on the retained principal
submatrix. Multiple zero diagonals are reduced together. PSD facial
certificates record the removed matrix indices and an aligned source-row list;
near-zero upper bounds do not trigger this rule.

Generic facial reduction need not preserve the existence of finite standard
dual multipliers for the unreduced cone. Therefore
`prefos_postsolve_primal_dual()` and `prefos_verify_postsolve_kkt()` return
`PREFOS_STATUS_DUAL_RECOVERY_UNAVAILABLE` after an RSOC or PSD face reduction. Use
`prefos_postsolve_extended_dual()` and `prefos_verify_postsolve_extended_kkt()` to
recover and audit normals to the exposed face; structural proofs are available
through `prefos_get_facial_reductions()`.

Integrations that require only standard original-cone multipliers can disable
these transformations with `rsoc_face_reduction = 0` and
`psd_face_reduction = 0`. Exponential and power coordinate reductions are
controlled independently by `exponential_face_reduction` and
`power_face_reduction`.

## Equality Aggregation

Bounded equality aggregation eliminates a fully free box column from a short
equality when the column does not enter `Q` or `R`. By default, source
equalities contain at most three nonzeros, candidate columns have active degree
at most eight, and a substitution adds at most eight net `A` nonzeros. The pass
applies up to four independent batches, materializing the transformed sparse
matrix between batches so that newly exposed equalities and substitution
chains can be processed without unbounded fill-in.

Each eliminated column is recorded as
`x_k = beta + sum_t gamma_t x_t`. Matrix compaction and objective reduction
expand these acyclic expressions, primal postsolve replays them in reverse,
and dual postsolve uses the objective and column coefficients from the batch in
which each elimination occurred. Linear objective incidence is supported. Set
`free_column_substitution = 0` to disable this pass. Statistics count all such
reductions in `substituted_free_variables`; the two-target subset is also
reported as `ternary_substituted_free_variables`.

`max_aggregation_terms`, `max_aggregation_column_degree`,
`max_aggregation_fill`, `max_aggregation_rounds`, and `max_aggregation_scale`
control the structural and numerical budget. Up to four retained terms (a
five-nonzero source equality) are supported. The wider setting is intentionally
not the default: it reduces some benchmark dimensions further but can increase
factorization fill for direct solvers.

## Cone Propagation

Cone propagation extracts scalar envelopes from retained singleton rows and
applies cheap geometry rules. Row activity also recognizes a finite support
value when a row's coefficient slice is conservatively certified to lie in a
cone's dual. This captures signs involving coupled coordinates, allowing rows
such as interior SOC-dual inequalities to be removed even when coordinatewise
interval activity is unbounded. It can also detect cone-supported row
infeasibility.

`cone_propagation` remains the master propagation switch. The dual-support activity pass,
nonlinear exponential rules, nonlinear power rules, and higher-order PSD rules
can be ablated independently with `cone_aware_row_activity`,
`exponential_propagation`, `power_propagation`, and
`psd_higher_order_propagation`. Disabling a nonlinear rule does not disable
intrinsic cone signs or fixed-point cone validation.

SOC and rotated SOC propagation use norm and product envelopes. PSD propagation
uses every 2-by-2 principal minor, fixed 3-by-3 determinants, Schur-complement
diagonal lower bounds, and fixed contiguous principal-submatrix eigenvalue
checks up to order eight. Order-one PSD cones are normalized directly to
nonnegative boxes. Coordinate zero-diagonal facial reduction remains available
for larger PSD blocks. Exact block-diagonal decomposition is supported for
affine PSD images; general nullspace faces and overlapping chordal
decomposition would require non-coordinate variable maps or newly introduced
variables and are not part of the lightweight standard-form pass.

Sparse affine PSD blocks also use a coordinate-work budget. The default
`affine_psd_propagation_max_work_ratio = 16` skips nonlinear envelope scans when
the packed `svec` dimension is too large relative to the nonzeros in `G` and
`h`. Singleton diagonal bounds and exact structural face reduction still run.
Set the ratio to zero, as `prefos_strict_settings()` does, to disable this guard.
Statistics report skipped blocks and packed coordinates.

Exponential cones use `(x,y,z)` with `y * exp(x / y) <= z`. Besides the
intrinsic envelopes `y >= 0` and `z >= 0`, propagation minimizes
`y * exp(lower_x / y)` over the current `y` interval to tighten `z`, and
maximizes `y * log(upper_z / y)` to tighten `x`. Exact singleton proofs of
`y = 0` or `z = 0` replace the nonlinear block by its corresponding directed
box face.

Power cones use `x^alpha * y^(1-alpha) >= |z|`, with `alpha` stored in
`PreFOSConeBlock.power_alpha`. Propagation tightens the radial capacity and derives
axis lower bounds from a nonzero minimum `|z|`. Exact proofs of `x = 0`,
`y = 0`, or `z = 0` reduce the block to one or two nonnegative boxes. Because
exponential and power cones are not self-dual, primal and dual membership and
extended-dual face verification use separate formulas.

PSD blocks use row-major `svec` coordinates with off-diagonal entries scaled by
`sqrt(2)`. This keeps the PSD cone self-dual in the vector Euclidean inner
product and makes the cone KKT convention unambiguous.

## Correctness and Benchmarks

`prefos_strict_settings()` disables tolerance-based bound fixing and is intended
for equivalence regression tests. After solving a reduced model,
`prefos_verify_postsolve_primal()` independently recomputes the original and
reduced objectives, linear-row violations, box violations, and cone violations
(including a minimum-eigenvalue check for PSD blocks and closure-aware checks
for exponential and power cones).

For primal-dual solutions, `prefos_postsolve_primal_dual()` replays bound
provenance in reverse and `prefos_verify_postsolve_kkt()` checks stationarity,
row/domain dual feasibility, complementarity, primal feasibility, and objective
identity in both models. The dual convention is
`grad(f) + A^T y + z = 0`, where `y` and `z` are normals to the ranged rows and
variable domains, respectively.

If a solver returns separate nonnegative lower/upper multipliers, convert them
as `y = y_upper - y_lower` and `z = z_upper - z_lower` before calling the KKT
API. Use `prefos_strict_settings()` for exact-equivalence/KKT regression tests;
the default settings intentionally allow tolerance-based model changes.

The regular C tests include deterministic round-trip and feasible-grid checks,
plus affine PSD component decomposition, primal equivalence, and full-dual
`svec` embedding checks.
Two optional randomized differential harnesses solve both the original and
reduced problems, then audit primal postsolve, objective identity, and KKT
postsolve:

```code
cmake --build build --target prefos_tests
ctest --test-dir build --output-on-failure
python3 tests/differential_prefos.py --seeds 100
python3 tests/differential_conic_prefos.py --seeds 100
```

`differential_prefos.py` uses OSQP for box QPs. The Clarabel-based conic harness
puts nonnegative, SOC, rotated SOC, order-three PSD, exponential, and power
blocks in every generated model. It cycles through whole-block zero eliminations, both
rotated-SOC single-axis faces, and PSD faces with one or two zero diagonals. Its
strongly convex objectives make the direct and postsolved primal solutions
independently comparable. Use `--start-seed` to replay or split a failing seed
range.

The differential scripts require Python packages `numpy` and `scipy`, plus
`osqp` or `clarabel` for the corresponding harness. The presolver library
itself remains dependency-free.

For large CBF/SOCP timing and reduction statistics, use
`tests/benchmark_cbf_prefos.py`. It dynamically reuses the PDHCG-II CBF parser and
keeps parse, adapter, constructor, and presolve timings separate.

Use `tests/benchmark_clarabel_gurobi.py` to compare the PreFOS and Gurobi
presolvers with Clarabel as the common downstream solver. Clarabel's own
presolver is disabled on both compared paths.

Pass `--scalar-cbf` to run the same comparison on scalar CBF exponential and
three-dimensional power cones. Gurobi has no native cone-domain input for these
blocks, so the adapter records its EXP/Power helper variables separately and
includes both lifted-model construction and `Model.presolve()` in the external
presolve pipeline time. Presolved Gurobi general constraints are exported back
to native Clarabel cones; solved-pair objective agreement is reported directly.

For real exponential and power cone models, use the continuous CBF v2/v3
adapter in `tests/cbf_scalar.py` and the benchmark driver
`tests/benchmark_exp_power_cbf.py`. The public
[CBLIB](https://cblib.zib.de/) collection contains `LogExpCR-*` exponential
cone risk models and `HMCR-*` power cone risk models. The driver compares
Clarabel without presolve, PreFOS with linear propagation disabled, PreFOS with
cone propagation disabled, and full PreFOS; Clarabel's presolver is disabled in
every mode. See `benchmarks/README.md` for benchmark dependencies, output
handling, and publication guidance.

`PreFOSStats` reports redundant-row activity, propagation, and cone-collapse time
separately.
EXP, power, and PSD counters distinguish rule attempts from actual bound hits;
one hit is one committed lower- or upper-bound improvement. The benchmark
drivers expose the independent switches and print these counters directly.
