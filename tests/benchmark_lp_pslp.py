#!/usr/bin/env python3
# Copyright 2026 Hongpei Li
# SPDX-License-Identifier: Apache-2.0

"""Compare CPU PreFOS and upstream PSLP on identical parsed LP data."""

import argparse
import contextlib
import ctypes as ct
import gzip
import json
import shutil
import statistics
import tempfile
import time
from pathlib import Path

import highspy
import numpy as np
import scipy.sparse as sp

from benchmark_cbf_prefos import PreFOSStats as PreFOSLPStats
from differential_prefos import (
    AffineConeBlock,
    ConeBlock,
    CsrMatrix,
    ProblemData,
    Settings as PreFOSSettings,
    configure_library as configure_prefos,
    ptr,
)


class PSLPSettings(ct.Structure):
    _fields_ = [
        ("ston_cols", ct.c_bool),
        ("dton_eq", ct.c_bool),
        ("parallel_rows", ct.c_bool),
        ("parallel_cols", ct.c_bool),
        ("primal_propagation", ct.c_bool),
        ("finite_bound_tightening", ct.c_bool),
        ("dual_fix", ct.c_bool),
        ("relax_bounds", ct.c_bool),
        ("max_shift", ct.c_int),
        ("max_time", ct.c_double),
        ("verbose", ct.c_bool),
    ]


class PSLPPresolvedProblem(ct.Structure):
    _fields_ = [
        ("values", ct.POINTER(ct.c_double)),
        ("column_indices", ct.POINTER(ct.c_int)),
        ("row_pointers", ct.POINTER(ct.c_int)),
        ("rows", ct.c_size_t),
        ("columns", ct.c_size_t),
        ("nnz", ct.c_size_t),
        ("row_lower", ct.POINTER(ct.c_double)),
        ("row_upper", ct.POINTER(ct.c_double)),
        ("objective", ct.POINTER(ct.c_double)),
        ("column_lower", ct.POINTER(ct.c_double)),
        ("column_upper", ct.POINTER(ct.c_double)),
        ("objective_offset", ct.c_double),
    ]


class PSLPStats(ct.Structure):
    _fields_ = [
        ("rows_original", ct.c_size_t),
        ("columns_original", ct.c_size_t),
        ("nnz_original", ct.c_size_t),
        ("rows_reduced", ct.c_size_t),
        ("columns_reduced", ct.c_size_t),
        ("nnz_reduced", ct.c_size_t),
        ("nnz_removed_trivial", ct.c_size_t),
        ("nnz_removed_fast", ct.c_size_t),
        ("nnz_removed_primal_propagation", ct.c_size_t),
        ("nnz_removed_parallel_rows", ct.c_size_t),
        ("nnz_removed_parallel_columns", ct.c_size_t),
        ("initialization_seconds", ct.c_double),
        ("fast_reduction_seconds", ct.c_double),
        ("medium_reduction_seconds", ct.c_double),
        ("singleton_column_seconds", ct.c_double),
        ("doubleton_row_seconds", ct.c_double),
        ("primal_propagation_seconds", ct.c_double),
        ("parallel_row_seconds", ct.c_double),
        ("parallel_column_seconds", ct.c_double),
        ("presolve_seconds", ct.c_double),
        ("postsolve_seconds", ct.c_double),
    ]


class PSLPPresolver(ct.Structure):
    _fields_ = [
        ("stats", ct.POINTER(PSLPStats)),
        ("settings", ct.POINTER(PSLPSettings)),
        ("problem", ct.c_void_p),
        ("reduced_problem", ct.POINTER(PSLPPresolvedProblem)),
        ("solution", ct.c_void_p),
    ]


def configure_pslp(path):
    library = ct.CDLL(str(path))
    library.default_settings.restype = ct.POINTER(PSLPSettings)
    library.free_settings.argtypes = [ct.POINTER(PSLPSettings)]
    library.new_presolver.argtypes = [
        ct.POINTER(ct.c_double),
        ct.POINTER(ct.c_int),
        ct.POINTER(ct.c_int),
        ct.c_size_t,
        ct.c_size_t,
        ct.c_size_t,
        ct.POINTER(ct.c_double),
        ct.POINTER(ct.c_double),
        ct.POINTER(ct.c_double),
        ct.POINTER(ct.c_double),
        ct.POINTER(ct.c_double),
        ct.POINTER(PSLPSettings),
    ]
    library.new_presolver.restype = ct.POINTER(PSLPPresolver)
    library.run_presolver.argtypes = [ct.POINTER(PSLPPresolver)]
    library.run_presolver.restype = ct.c_uint8
    library.free_presolver.argtypes = [ct.POINTER(PSLPPresolver)]
    return library


def readable_mps(path, stack):
    if path.suffix != ".gz":
        return path
    temporary = stack.enter_context(
        tempfile.NamedTemporaryFile(suffix=".mps", delete=True)
    )
    with gzip.open(path, "rb") as source:
        shutil.copyfileobj(source, temporary)
    temporary.flush()
    return Path(temporary.name)


def parse_lp(path):
    parse_start = time.perf_counter()
    with contextlib.ExitStack() as stack:
        readable = readable_mps(path, stack)
        highs = highspy.Highs()
        highs.setOptionValue("output_flag", False)
        if highs.readModel(str(readable)) != highspy.HighsStatus.kOk:
            raise RuntimeError("HiGHS could not parse the model")
        lp = highs.getLp()
        matrix = lp.a_matrix_
        starts = np.asarray(matrix.start_, dtype=np.int64)
        indices = np.asarray(matrix.index_, dtype=np.int32)
        values = np.asarray(matrix.value_, dtype=np.float64)
        if matrix.format_ == highspy.MatrixFormat.kColwise:
            sparse = sp.csc_matrix(
                (values, indices, starts),
                shape=(lp.num_row_, lp.num_col_),
            ).tocsr()
        else:
            sparse = sp.csr_matrix(
                (values, indices, starts),
                shape=(lp.num_row_, lp.num_col_),
            )
        sparse.sum_duplicates()
        sparse.eliminate_zeros()
        sparse.sort_indices()
        if sparse.nnz > np.iinfo(np.int32).max:
            raise RuntimeError("matrix has more entries than the C APIs support")
        objective = np.asarray(lp.col_cost_, dtype=np.float64)
        if lp.sense_ == highspy.ObjSense.kMaximize:
            objective = -objective
        result = {
            "A_values": np.ascontiguousarray(sparse.data, dtype=np.float64),
            "A_indices": np.ascontiguousarray(sparse.indices, dtype=np.int32),
            "A_indptr": np.ascontiguousarray(sparse.indptr, dtype=np.int32),
            "row_lower": np.ascontiguousarray(lp.row_lower_, dtype=np.float64),
            "row_upper": np.ascontiguousarray(lp.row_upper_, dtype=np.float64),
            "column_lower": np.ascontiguousarray(
                lp.col_lower_, dtype=np.float64
            ),
            "column_upper": np.ascontiguousarray(
                lp.col_upper_, dtype=np.float64
            ),
            "objective": np.ascontiguousarray(objective, dtype=np.float64),
            "rows": int(lp.num_row_),
            "columns": int(lp.num_col_),
            "nnz": int(sparse.nnz),
            "integer_columns": int(
                sum(
                    variable_type != highspy.HighsVarType.kContinuous
                    for variable_type in lp.integrality_
                )
            ),
        }
    result["parse_seconds"] = time.perf_counter() - parse_start
    return result


def prefos_problem(data):
    rows = data["rows"]
    columns = data["columns"]
    box_indices = np.arange(columns, dtype=np.int32)
    q_indptr = np.zeros(columns + 1, dtype=np.int32)
    r_indptr = np.zeros(1, dtype=np.int32)
    matrix = CsrMatrix(
        rows,
        columns,
        data["nnz"],
        ptr(data["A_values"], ct.c_double),
        ptr(data["A_indices"], ct.c_int),
        ptr(data["A_indptr"], ct.c_int),
    )
    empty_q = CsrMatrix(
        columns,
        columns,
        0,
        ct.POINTER(ct.c_double)(),
        ct.POINTER(ct.c_int)(),
        ptr(q_indptr, ct.c_int),
    )
    empty_r = CsrMatrix(
        0,
        columns,
        0,
        ct.POINTER(ct.c_double)(),
        ct.POINTER(ct.c_int)(),
        ptr(r_indptr, ct.c_int),
    )
    problem = ProblemData(
        columns,
        matrix,
        ptr(data["row_lower"], ct.c_double),
        ptr(data["row_upper"], ct.c_double),
        empty_q,
        0,
        empty_r,
        ct.POINTER(ct.c_double)(),
        ptr(data["objective"], ct.c_double),
        0.0,
        columns,
        ptr(box_indices, ct.c_int),
        ptr(data["column_lower"], ct.c_double),
        ptr(data["column_upper"], ct.c_double),
        0,
        ct.POINTER(ConeBlock)(),
        CsrMatrix(),
        ct.POINTER(ct.c_double)(),
        0,
        ct.POINTER(AffineConeBlock)(),
    )
    return problem, (box_indices, q_indptr, r_indptr)


def run_prefos(library, data, args):
    problem, keepalive = prefos_problem(data)
    settings = (
        library.prefos_strict_settings()
        if args.prefos_strict
        else library.prefos_default_settings()
    )
    if args.prefos_bounded_doubletons:
        settings.bounded_doubleton_substitution = 1
    if args.prefos_exhaustive_bounds:
        settings.finite_bound_improvement_absolute = 0.0
        settings.finite_bound_improvement_relative = 0.0
        settings.linear_propagation_max_work_ratio = 0.0
        settings.linear_propagation_min_changes_per_million = 0.0
        settings.linear_propagation_max_stale_rounds = 0
    if args.prefos_disable_free_columns:
        settings.free_column_substitution = 0
    if args.prefos_disable_linear_propagation:
        settings.linear_propagation = 0
    if args.prefos_disable_cone_propagation:
        settings.cone_propagation = 0
    if args.prefos_disable_redundant_rows:
        settings.remove_redundant_rows = 0
    if args.prefos_skip_parallel_rows:
        settings.parallel_row_max_average_nnz = 1e-300
    if args.prefos_skip_redundant_row_activity:
        settings.redundant_row_max_average_nnz = 1e-300
    if args.prefos_disable_singleton_columns:
        settings.singleton_column_reduction = 0
    if args.prefos_disable_parallel_columns:
        settings.parallel_column_reduction = 0
    if args.prefos_disable_dual_fixing:
        settings.dual_fixing = 0
    if args.prefos_event_max_average_degree is not None:
        settings.event_queue_max_average_column_degree = (
            args.prefos_event_max_average_degree
        )
    if args.prefos_event_update_ratio is not None:
        settings.event_queue_activity_update_ratio = (
            args.prefos_event_update_ratio
        )
    presolver = ct.c_void_p()
    start = time.perf_counter()
    create_status = library.prefos_create_presolver(
        ct.byref(problem), ct.byref(settings), ct.byref(presolver)
    )
    initialized = time.perf_counter()
    if create_status != 0:
        return {
            "status": int(create_status),
            "create_seconds": initialized - start,
            "presolve_seconds": 0.0,
            "total_seconds": initialized - start,
        }
    try:
        status = library.prefos_run_presolve(presolver)
        finished = time.perf_counter()
        result = {
            "status": int(status),
            "create_seconds": initialized - start,
            "presolve_seconds": finished - initialized,
            "total_seconds": finished - start,
        }
        if status in (0, 1):
            reduced = library.prefos_get_reduced_problem(presolver).contents
            stats = library.prefos_get_stats(presolver).contents
            result.update(
                rows=int(reduced.A.rows),
                columns=int(reduced.n),
                nnz=int(reduced.A.nnz),
                propagated_bounds=int(stats.propagated_box_bounds),
                propagation_rounds=int(stats.linear_propagation_rounds),
                propagation_event_rounds=int(stats.linear_event_rounds),
                propagation_full_scan_rounds=int(
                    stats.linear_full_scan_rounds
                ),
                propagation_rows=int(stats.linear_rows_processed),
                propagation_nnz=int(stats.linear_nnz_processed),
                propagation_activity_nnz=int(
                    stats.linear_activity_nnz_computed
                ),
                propagation_activity_updates=int(
                    stats.linear_activity_updates
                ),
                propagation_budget_stops=int(stats.linear_budget_stops),
                propagation_stale_stops=int(stats.linear_stale_stops),
                propagation_milliseconds=float(
                    stats.linear_propagation_milliseconds
                ),
                redundant_activity_milliseconds=float(
                    stats.redundant_row_activity_milliseconds
                ),
                parallel_row_milliseconds=float(
                    stats.parallel_row_detection_milliseconds
                ),
                structural_milliseconds=float(
                    stats.structural_reduction_milliseconds
                ),
                initialization_milliseconds=float(
                    stats.initialization_milliseconds
                ),
                affine_aggregation_milliseconds=float(
                    stats.affine_aggregation_milliseconds
                ),
                fast_fixed_point_milliseconds=float(
                    stats.fast_fixed_point_milliseconds
                ),
                free_column_substitution_milliseconds=float(
                    stats.free_column_substitution_milliseconds
                ),
                trivial_row_reduction_milliseconds=float(
                    stats.trivial_row_reduction_milliseconds
                ),
                medium_fixed_point_milliseconds=float(
                    stats.medium_fixed_point_milliseconds
                ),
                parallel_column_reduction_milliseconds=float(
                    stats.parallel_column_reduction_milliseconds
                ),
                matrix_compaction_milliseconds=float(
                    stats.matrix_compaction_milliseconds
                ),
                quadratic_compaction_milliseconds=float(
                    stats.quadratic_compaction_milliseconds
                ),
                factor_compaction_milliseconds=float(
                    stats.factor_compaction_milliseconds
                ),
                domain_compaction_milliseconds=float(
                    stats.domain_compaction_milliseconds
                ),
                objective_compaction_milliseconds=float(
                    stats.objective_compaction_milliseconds
                ),
                presolve_total_milliseconds=float(
                    stats.presolve_total_milliseconds
                ),
                fast_fixed_point_passes=int(
                    stats.fast_fixed_point_passes
                ),
                fast_fixed_point_rounds=int(
                    stats.fast_fixed_point_rounds
                ),
                medium_fixed_point_rounds=int(
                    stats.medium_fixed_point_rounds
                ),
                residual_row_substitutions=int(
                    stats.residual_row_substitutions
                ),
                substituted_free_variables=int(
                    stats.substituted_free_variables
                ),
                removed_redundant_rows=int(stats.removed_redundant_rows),
                removed_singleton_rows=int(stats.removed_singleton_rows),
                removed_empty_rows=int(stats.removed_empty_rows),
                removed_empty_columns=int(stats.removed_empty_columns),
                removed_singleton_columns=int(
                    stats.removed_singleton_columns
                ),
                tightened_singleton_rows=int(stats.tightened_singleton_rows),
                dual_fixed_columns=int(stats.dual_fixed_columns),
                merged_parallel_columns=int(stats.merged_parallel_columns),
            )
        return result
    finally:
        library.prefos_free_presolver(presolver)
        del keepalive


def run_pslp(
    library, data, max_time, disable_singletons, disable_doubletons,
    disable_propagation, disable_dual_fixing
):
    settings = library.default_settings()
    if not settings:
        raise MemoryError("PSLP settings allocation failed")
    settings.contents.verbose = False
    settings.contents.max_time = max_time
    settings.contents.ston_cols = not disable_singletons
    settings.contents.dton_eq = not disable_doubletons
    settings.contents.primal_propagation = not disable_propagation
    settings.contents.dual_fix = not disable_dual_fixing
    start = time.perf_counter()
    presolver = library.new_presolver(
        ptr(data["A_values"], ct.c_double),
        ptr(data["A_indices"], ct.c_int),
        ptr(data["A_indptr"], ct.c_int),
        data["rows"],
        data["columns"],
        data["nnz"],
        ptr(data["row_lower"], ct.c_double),
        ptr(data["row_upper"], ct.c_double),
        ptr(data["column_lower"], ct.c_double),
        ptr(data["column_upper"], ct.c_double),
        ptr(data["objective"], ct.c_double),
        settings,
    )
    initialized = time.perf_counter()
    if not presolver:
        library.free_settings(settings)
        raise MemoryError("PSLP presolver allocation failed")
    try:
        status = library.run_presolver(presolver)
        finished = time.perf_counter()
        result = {
            "status": int(status),
            "create_seconds": initialized - start,
            "presolve_seconds": finished - initialized,
            "total_seconds": finished - start,
        }
        if status in (0, 1):
            reduced = presolver.contents.reduced_problem.contents
            stats = presolver.contents.stats.contents
            result.update(
                rows=int(reduced.rows),
                columns=int(reduced.columns),
                nnz=int(reduced.nnz),
                nnz_removed_trivial=int(stats.nnz_removed_trivial),
                nnz_removed_fast=int(stats.nnz_removed_fast),
                nnz_removed_propagation=int(
                    stats.nnz_removed_primal_propagation
                ),
                nnz_removed_parallel_rows=int(
                    stats.nnz_removed_parallel_rows
                ),
                nnz_removed_parallel_columns=int(
                    stats.nnz_removed_parallel_columns
                ),
            )
        return result
    finally:
        library.free_presolver(presolver)
        library.free_settings(settings)


def median_run(runs):
    valid = [run for run in runs if "error" not in run]
    if not valid:
        return runs[0]
    target = statistics.median(run["total_seconds"] for run in valid)
    representative = min(
        valid, key=lambda run: abs(run["total_seconds"] - target)
    ).copy()
    for field in ("create_seconds", "presolve_seconds", "total_seconds"):
        representative[field] = statistics.median(
            run[field] for run in valid
        )
    representative["samples"] = len(valid)
    return representative


def reduction_text(original, result):
    if not all(field in result for field in ("rows", "columns", "nnz")):
        return f"status={result['status']}"
    return (
        f"({original['rows']:,},{original['columns']:,},{original['nnz']:,})"
        f" -> ({result['rows']:,},{result['columns']:,},{result['nnz']:,})"
    )


def pslp_breakdown(result):
    if "nnz_removed_fast" not in result:
        return ""
    return (
        " [trivial={nnz_removed_trivial:,},fast={nnz_removed_fast:,},"
        "prop={nnz_removed_propagation:,},"
        "prow={nnz_removed_parallel_rows:,},"
        "pcol={nnz_removed_parallel_columns:,}]"
    ).format(**result)


def benchmark_file(prefos, pslp, path, repeats, args):
    data = parse_lp(path)
    samples = {"prefos": [], "pslp": []}
    for repetition in range(repeats):
        order = ("prefos", "pslp") if repetition % 2 == 0 else ("pslp", "prefos")
        for solver in order:
            try:
                if solver == "prefos":
                    sample = run_prefos(prefos, data, args)
                else:
                    sample = run_pslp(
                        pslp,
                        data,
                        args.pslp_max_time,
                        args.pslp_disable_singletons,
                        args.pslp_disable_doubletons,
                        args.pslp_disable_propagation,
                        args.pslp_disable_dual_fixing,
                    )
            except Exception as error:
                sample = {"error": str(error)}
            samples[solver].append(sample)
    result = {
        "file": str(path),
        "parse_seconds": data["parse_seconds"],
        "rows_original": data["rows"],
        "columns_original": data["columns"],
        "nnz_original": data["nnz"],
        "integer_columns_relaxed": data["integer_columns"],
        "prefos": median_run(samples["prefos"]),
        "pslp": median_run(samples["pslp"]),
    }
    print(
        f"{path.name}: parse={data['parse_seconds']:.3f}s "
        f"PreFOS={result['prefos'].get('total_seconds', float('nan')):.4f}s "
        f"{reduction_text(data, result['prefos'])} | "
        f"PSLP={result['pslp'].get('total_seconds', float('nan')):.4f}s "
        f"{reduction_text(data, result['pslp'])}"
        f"{pslp_breakdown(result['pslp'])}",
        flush=True,
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--prefos-library", required=True, type=Path)
    parser.add_argument("--pslp-library", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--pslp-max-time", type=float, default=600.0)
    parser.add_argument("--prefos-strict", action="store_true")
    parser.add_argument("--prefos-bounded-doubletons", action="store_true")
    parser.add_argument("--prefos-exhaustive-bounds", action="store_true")
    parser.add_argument("--prefos-disable-free-columns", action="store_true")
    parser.add_argument(
        "--prefos-disable-linear-propagation", action="store_true"
    )
    parser.add_argument(
        "--prefos-disable-cone-propagation", action="store_true"
    )
    parser.add_argument(
        "--prefos-disable-redundant-rows", action="store_true"
    )
    parser.add_argument("--prefos-skip-parallel-rows", action="store_true")
    parser.add_argument(
        "--prefos-skip-redundant-row-activity", action="store_true"
    )
    parser.add_argument(
        "--prefos-disable-singleton-columns", action="store_true"
    )
    parser.add_argument(
        "--prefos-disable-parallel-columns", action="store_true"
    )
    parser.add_argument("--prefos-disable-dual-fixing", action="store_true")
    parser.add_argument("--prefos-event-max-average-degree", type=float)
    parser.add_argument("--prefos-event-update-ratio", type=float)
    parser.add_argument("--pslp-disable-singletons", action="store_true")
    parser.add_argument("--pslp-disable-doubletons", action="store_true")
    parser.add_argument("--pslp-disable-propagation", action="store_true")
    parser.add_argument("--pslp-disable-dual-fixing", action="store_true")
    parser.add_argument("--jsonl", type=Path)
    args = parser.parse_args()
    if args.repeats <= 0:
        parser.error("--repeats must be positive")

    prefos = configure_prefos(args.prefos_library)
    prefos.prefos_default_settings.restype = PreFOSSettings
    prefos.prefos_get_stats.argtypes = [ct.c_void_p]
    prefos.prefos_get_stats.restype = ct.POINTER(PreFOSLPStats)
    pslp = configure_pslp(args.pslp_library)
    output = args.jsonl.open("w", encoding="utf-8") if args.jsonl else None
    try:
        for path in args.inputs:
            try:
                result = benchmark_file(
                    prefos, pslp, path, args.repeats, args
                )
            except Exception as error:
                result = {"file": str(path), "error": str(error)}
                print(f"{path.name}: ERROR {error}", flush=True)
            if output:
                output.write(json.dumps(result, sort_keys=True) + "\n")
                output.flush()
    finally:
        if output:
            output.close()


if __name__ == "__main__":
    main()
