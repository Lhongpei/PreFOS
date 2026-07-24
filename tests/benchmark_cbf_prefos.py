#!/usr/bin/env python3
# Copyright 2026 Hongpei Li
# SPDX-License-Identifier: Apache-2.0

"""Benchmark the PreFOS presolver on CBF files parsed by PDHCG-II."""

import argparse
import ctypes as ct
import json
import resource
import time
from pathlib import Path

import numpy as np

from differential_prefos import (
    ConeBlock,
    CsrMatrix,
    ProblemData,
    Settings,
    configure_library,
    ptr,
)


PDHCG_RSOC = 0
PDHCG_SOC = 1
PREFOS_SOC = 1
PREFOS_RSOC = 2
PREFOS_Q_FULL = 2


class PdhcgCsr(ct.Structure):
    _fields_ = [
        ("row_ptr", ct.POINTER(ct.c_int)),
        ("col_ind", ct.POINTER(ct.c_int)),
        ("val", ct.POINTER(ct.c_double)),
    ]


class PdhcgConeBlocks(ct.Structure):
    _fields_ = [
        ("num_cones", ct.c_int),
        ("start_idx", ct.POINTER(ct.c_int)),
        ("v_dim", ct.POINTER(ct.c_int)),
        ("type", ct.POINTER(ct.c_int)),
        ("power_alpha", ct.POINTER(ct.c_double)),
        ("is_fixed", ct.POINTER(ct.c_char)),
    ]


class PdhcgProblem(ct.Structure):
    _fields_ = [
        ("num_variables", ct.c_int),
        ("num_constraints", ct.c_int),
        ("num_rank_lowrank_obj", ct.c_int),
        ("variable_lower_bound", ct.POINTER(ct.c_double)),
        ("variable_upper_bound", ct.POINTER(ct.c_double)),
        ("objective_vector", ct.POINTER(ct.c_double)),
        ("objective_constant", ct.c_double),
        ("constraint_matrix", ct.POINTER(PdhcgCsr)),
        ("constraint_matrix_num_nonzeros", ct.c_int),
        ("objective_sparse_matrix", ct.POINTER(PdhcgCsr)),
        ("objective_sparse_matrix_num_nonzeros", ct.c_int),
        ("objective_lowrank_matrix", ct.POINTER(PdhcgCsr)),
        ("objective_lowrank_matrix_num_nonzeros", ct.c_int),
        ("objective_lowrank_middle_matrix", ct.POINTER(PdhcgCsr)),
        ("objective_lowrank_middle_matrix_num_nonzeros", ct.c_int),
        ("constraint_lower_bound", ct.POINTER(ct.c_double)),
        ("constraint_upper_bound", ct.POINTER(ct.c_double)),
        ("num_quadratic_constraints", ct.c_int),
        ("quadratic_constraint_row_indices", ct.POINTER(ct.c_int)),
        ("quadratic_constraint_matrices", ct.POINTER(ct.POINTER(PdhcgCsr))),
        ("quadratic_constraint_matrix_num_nonzeros", ct.POINTER(ct.c_int)),
        ("cones", PdhcgConeBlocks),
    ]


class PreFOSStats(ct.Structure):
    _fields_ = [
        ("rows_original", ct.c_size_t),
        ("rows_reduced", ct.c_size_t),
        ("variables_original", ct.c_size_t),
        ("variables_reduced", ct.c_size_t),
        ("nnz_A_original", ct.c_size_t),
        ("nnz_A_reduced", ct.c_size_t),
        ("nnz_Q_original", ct.c_size_t),
        ("nnz_Q_reduced", ct.c_size_t),
        ("nnz_R_original", ct.c_size_t),
        ("nnz_R_reduced", ct.c_size_t),
        ("normalized_nonnegative_variables", ct.c_size_t),
        ("normalized_nonnegative_cones", ct.c_size_t),
        ("fixed_box_variables", ct.c_size_t),
        ("substituted_free_variables", ct.c_size_t),
        ("ternary_substituted_free_variables", ct.c_size_t),
        ("tightened_box_bounds", ct.c_size_t),
        ("propagated_box_bounds", ct.c_size_t),
        ("linear_propagation_rounds", ct.c_size_t),
        ("linear_event_rounds", ct.c_size_t),
        ("linear_full_scan_rounds", ct.c_size_t),
        ("linear_full_scan_fallbacks", ct.c_size_t),
        ("linear_activity_nnz_computed", ct.c_size_t),
        ("linear_rows_processed", ct.c_size_t),
        ("linear_nnz_processed", ct.c_size_t),
        ("linear_activity_updates", ct.c_size_t),
        ("linear_budget_stops", ct.c_size_t),
        ("linear_stale_stops", ct.c_size_t),
        ("linear_propagation_milliseconds", ct.c_double),
        ("linear_gpu_rounds", ct.c_size_t),
        ("linear_gpu_fallbacks", ct.c_size_t),
        ("linear_gpu_setup_milliseconds", ct.c_double),
        ("linear_gpu_transfer_milliseconds", ct.c_double),
        ("linear_gpu_kernel_milliseconds", ct.c_double),
        ("linear_gpu_total_milliseconds", ct.c_double),
        ("linear_gpu_long_rows", ct.c_size_t),
        ("tightened_cone_envelopes", ct.c_size_t),
        ("cone_propagation_rounds", ct.c_size_t),
        ("cone_propagation_milliseconds", ct.c_double),
        ("cone_activity_rows", ct.c_size_t),
        ("cone_activity_blocks", ct.c_size_t),
        ("cone_activity_lower_support_hits", ct.c_size_t),
        ("cone_activity_upper_support_hits", ct.c_size_t),
        ("cone_activity_strengthened_rows", ct.c_size_t),
        ("cone_activity_rows_removed", ct.c_size_t),
        ("cone_activity_infeasible_rows", ct.c_size_t),
        ("redundant_row_activity_budget_skips", ct.c_size_t),
        ("redundant_row_activity_milliseconds", ct.c_double),
        ("exponential_cones_processed", ct.c_size_t),
        ("exponential_z_lower_attempts", ct.c_size_t),
        ("exponential_z_lower_hits", ct.c_size_t),
        ("exponential_x_upper_attempts", ct.c_size_t),
        ("exponential_x_upper_hits", ct.c_size_t),
        ("exponential_propagation_milliseconds", ct.c_double),
        ("power_cones_processed", ct.c_size_t),
        ("power_capacity_attempts", ct.c_size_t),
        ("power_unbounded_capacity_attempts", ct.c_size_t),
        ("power_zero_minimum_abs_z_attempts", ct.c_size_t),
        ("power_z_bound_hits", ct.c_size_t),
        ("power_axis_attempts", ct.c_size_t),
        ("power_axis_hits", ct.c_size_t),
        ("power_propagation_milliseconds", ct.c_double),
        ("psd_cones_processed", ct.c_size_t),
        ("psd_two_by_two_attempts", ct.c_size_t),
        ("psd_two_by_two_bound_hits", ct.c_size_t),
        ("psd_three_by_three_attempts", ct.c_size_t),
        ("psd_schur_attempts", ct.c_size_t),
        ("psd_schur_bound_hits", ct.c_size_t),
        ("psd_fixed_window_checks", ct.c_size_t),
        ("psd_propagation_milliseconds", ct.c_double),
        ("psd_higher_order_milliseconds", ct.c_double),
        ("cone_collapse_milliseconds", ct.c_double),
        ("fixed_cone_variables", ct.c_size_t),
        ("collapsed_cones", ct.c_size_t),
        ("reduced_rsoc_faces", ct.c_size_t),
        ("reduced_psd_faces", ct.c_size_t),
        ("reduced_exponential_faces", ct.c_size_t),
        ("reduced_power_faces", ct.c_size_t),
        ("removed_redundant_rows", ct.c_size_t),
        ("removed_singleton_rows", ct.c_size_t),
        ("removed_empty_rows", ct.c_size_t),
        ("parallel_row_budget_skips", ct.c_size_t),
        ("parallel_row_detection_milliseconds", ct.c_double),
        ("materialized_propagated_box_bounds", ct.c_size_t),
        ("suppressed_propagated_box_bounds", ct.c_size_t),
        ("aggregated_affine_cone_coordinates", ct.c_size_t),
        ("generated_affine_cone_blocks", ct.c_size_t),
        ("affine_cones_processed", ct.c_size_t),
        ("affine_cone_propagation_rounds", ct.c_size_t),
        ("affine_psd_budget_skips", ct.c_size_t),
        ("affine_psd_coordinates_skipped", ct.c_size_t),
        ("tightened_affine_cone_envelopes", ct.c_size_t),
        ("tightened_affine_variable_envelopes", ct.c_size_t),
        ("materialized_affine_cone_box_bounds", ct.c_size_t),
        ("suppressed_affine_cone_box_bounds", ct.c_size_t),
        ("reduced_affine_rsoc_faces", ct.c_size_t),
        ("reduced_affine_psd_faces", ct.c_size_t),
        ("reduced_affine_exponential_faces", ct.c_size_t),
        ("reduced_affine_power_faces", ct.c_size_t),
        ("removed_affine_cone_coordinates", ct.c_size_t),
        ("removed_affine_cone_blocks", ct.c_size_t),
        ("affine_face_reduction_milliseconds", ct.c_double),
        ("affine_psd_blocks_analyzed", ct.c_size_t),
        ("affine_psd_active_diagonal_coordinates", ct.c_size_t),
        ("affine_psd_active_offdiagonal_coordinates", ct.c_size_t),
        ("affine_psd_coefficient_columns", ct.c_size_t),
        ("affine_psd_diagonal_coefficient_columns", ct.c_size_t),
        ("affine_psd_single_diagonal_coefficient_columns", ct.c_size_t),
        ("affine_psd_connected_components", ct.c_size_t),
        ("affine_psd_largest_component_order", ct.c_size_t),
        ("affine_psd_splittable_blocks", ct.c_size_t),
        ("decomposed_affine_psd_blocks", ct.c_size_t),
        ("affine_psd_scalar_components", ct.c_size_t),
        ("affine_psd_component_blocks", ct.c_size_t),
        ("removed_affine_psd_cross_coordinates", ct.c_size_t),
        ("affine_psd_structure_milliseconds", ct.c_double),
        ("derived_affine_face_equalities", ct.c_size_t),
        ("fixed_affine_face_variables", ct.c_size_t),
        ("substituted_affine_face_variables", ct.c_size_t),
        ("affine_face_substitution_milliseconds", ct.c_double),
        ("removed_empty_columns", ct.c_size_t),
        ("removed_singleton_columns", ct.c_size_t),
        ("tightened_singleton_rows", ct.c_size_t),
        ("substituted_bounded_doubletons", ct.c_size_t),
        ("dual_fixed_columns", ct.c_size_t),
        ("merged_parallel_columns", ct.c_size_t),
        ("removed_redundant_row_lower_sides", ct.c_size_t),
        ("removed_redundant_row_upper_sides", ct.c_size_t),
        ("removed_redundant_box_lower_bounds", ct.c_size_t),
        ("removed_redundant_box_upper_bounds", ct.c_size_t),
        ("structural_gpu_passes", ct.c_size_t),
        ("structural_gpu_fallbacks", ct.c_size_t),
        ("structural_reduction_milliseconds", ct.c_double),
        ("column_csc_gpu_builds", ct.c_size_t),
        ("column_csc_gpu_fallbacks", ct.c_size_t),
        ("column_csc_gpu_milliseconds", ct.c_double),
        ("singleton_column_gpu_passes", ct.c_size_t),
        ("singleton_column_gpu_fallbacks", ct.c_size_t),
        ("singleton_column_gpu_candidates", ct.c_size_t),
        ("singleton_column_gpu_milliseconds", ct.c_double),
        ("parallel_column_gpu_passes", ct.c_size_t),
        ("parallel_column_gpu_fallbacks", ct.c_size_t),
        ("parallel_column_gpu_milliseconds", ct.c_double),
        ("parallel_row_gpu_passes", ct.c_size_t),
        ("parallel_row_gpu_fallbacks", ct.c_size_t),
        ("cone_activity_gpu_passes", ct.c_size_t),
        ("cone_activity_gpu_fallbacks", ct.c_size_t),
        ("cone_activity_gpu_candidates", ct.c_size_t),
        ("cone_gpu_rounds", ct.c_size_t),
        ("cone_gpu_linear_rounds", ct.c_size_t),
        ("cone_gpu_fallbacks", ct.c_size_t),
        ("cuda_workspace_setup_milliseconds", ct.c_double),
        ("parallel_row_gpu_milliseconds", ct.c_double),
        ("cone_activity_gpu_milliseconds", ct.c_double),
        ("cone_gpu_milliseconds", ct.c_double),
        ("cone_gpu_linear_transfer_milliseconds", ct.c_double),
        ("cone_gpu_linear_kernel_milliseconds", ct.c_double),
        ("affine_cone_gpu_rounds", ct.c_size_t),
        ("affine_cone_gpu_fallbacks", ct.c_size_t),
        ("affine_cone_gpu_milliseconds", ct.c_double),
        ("matrix_compaction_gpu_passes", ct.c_size_t),
        ("matrix_compaction_gpu_fallbacks", ct.c_size_t),
        ("matrix_compaction_gpu_milliseconds", ct.c_double),
        ("initialization_milliseconds", ct.c_double),
        ("affine_aggregation_milliseconds", ct.c_double),
        ("fast_fixed_point_milliseconds", ct.c_double),
        ("free_column_substitution_milliseconds", ct.c_double),
        ("trivial_row_reduction_milliseconds", ct.c_double),
        ("medium_fixed_point_milliseconds", ct.c_double),
        ("parallel_column_reduction_milliseconds", ct.c_double),
        ("matrix_compaction_milliseconds", ct.c_double),
        ("quadratic_compaction_milliseconds", ct.c_double),
        ("factor_compaction_milliseconds", ct.c_double),
        ("domain_compaction_milliseconds", ct.c_double),
        ("objective_compaction_milliseconds", ct.c_double),
        ("presolve_total_milliseconds", ct.c_double),
        ("fast_fixed_point_passes", ct.c_size_t),
        ("fast_fixed_point_rounds", ct.c_size_t),
        ("medium_fixed_point_rounds", ct.c_size_t),
        ("residual_row_substitutions", ct.c_size_t),
    ]


def configure_parser(path):
    library = ct.CDLL(str(path))
    library.read_cbf_file.argtypes = [ct.c_char_p]
    library.read_cbf_file.restype = ct.POINTER(PdhcgProblem)
    library.qp_problem_free.argtypes = [ct.POINTER(PdhcgProblem)]
    return library


def configure_prefos(path):
    library = configure_library(path)
    library.prefos_default_settings.restype = Settings
    library.prefos_gpu_warmup.restype = ct.c_int
    library.prefos_gpu_warmup_async.restype = ct.c_int
    library.prefos_gpu_warmup_ready.restype = ct.c_int
    library.prefos_gpu_warmup_wait.restype = ct.c_int
    library.prefos_gpu_release_cache.argtypes = []
    library.prefos_get_stats.argtypes = [ct.c_void_p]
    library.prefos_get_stats.restype = ct.POINTER(PreFOSStats)
    return library


def current_rss_mib():
    with open("/proc/self/status", encoding="ascii") as status_file:
        for line in status_file:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    return float("nan")


def peak_rss_mib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def pdhcg_csr_view(component, rows, cols, nnz):
    if not component:
        return CsrMatrix(rows, cols, 0, None, None, None)
    matrix = component.contents
    return CsrMatrix(
        rows,
        cols,
        nnz,
        matrix.val,
        matrix.col_ind,
        matrix.row_ptr,
    )


def make_prefos_view(problem_pointer):
    source = problem_pointer.contents
    n = source.num_variables
    m = source.num_constraints
    if n < 0 or m < 0:
        raise ValueError("PDHCG parser returned negative dimensions")
    if source.num_rank_lowrank_obj != 0 or source.objective_lowrank_matrix:
        raise ValueError("low-rank objectives are not supported by this adapter")
    if source.num_quadratic_constraints != 0:
        raise ValueError("quadratic constraints are not supported by PreFOS")

    cone_count = source.cones.num_cones
    if cone_count < 0:
        raise ValueError("PDHCG parser returned a negative cone count")
    dimensions = np.empty(cone_count, dtype=np.int64)
    for k in range(cone_count):
        cone_type = source.cones.type[k]
        if cone_type not in (PDHCG_SOC, PDHCG_RSOC):
            raise ValueError(f"unsupported PDHCG cone type {cone_type}")
        dimensions[k] = source.cones.v_dim[k] + 2
    total_cone_dimension = int(dimensions.sum())
    all_cone_indices = np.empty(total_cone_dimension, dtype=np.int32)
    cone_array = (ConeBlock * cone_count)()
    box_mask = np.ones(n, dtype=np.bool_)
    cone_offset = 0
    soc_count = 0
    rsoc_count = 0
    for k in range(cone_count):
        start = source.cones.start_idx[k]
        v_dim = source.cones.v_dim[k]
        dimension = int(dimensions[k])
        if start < 0 or dimension < 2 or start + dimension > n:
            raise ValueError(f"invalid cone {k}: start={start}, dim={dimension}")
        indices = all_cone_indices[cone_offset : cone_offset + dimension]
        if source.cones.type[k] == PDHCG_SOC:
            indices[0] = start + v_dim + 1
            indices[1:] = np.arange(start, start + v_dim + 1, dtype=np.int32)
            prefos_type = PREFOS_SOC
            soc_count += 1
        else:
            if dimension < 3:
                raise ValueError(f"rotated SOC {k} has dimension {dimension}")
            indices[0] = start + v_dim
            indices[1] = start + v_dim + 1
            indices[2:] = np.arange(start, start + v_dim, dtype=np.int32)
            prefos_type = PREFOS_RSOC
            rsoc_count += 1
        box_mask[start : start + dimension] = False
        cone_array[k] = ConeBlock(
            prefos_type,
            dimension,
            0,
            ptr(indices, ct.c_int),
            0.0,
        )
        cone_offset += dimension

    lower = np.ctypeslib.as_array(source.variable_lower_bound, shape=(n,))
    upper = np.ctypeslib.as_array(source.variable_upper_bound, shape=(n,))
    if total_cone_dimension > 0:
        cone_columns = all_cone_indices
        if np.any(np.isfinite(lower[cone_columns])) or np.any(
            np.isfinite(upper[cone_columns])
        ):
            raise ValueError("a cone variable also has a finite box bound")
    box_indices = np.flatnonzero(box_mask).astype(np.int32, copy=False)
    box_lower = np.ascontiguousarray(lower[box_indices], dtype=np.float64)
    box_upper = np.ascontiguousarray(upper[box_indices], dtype=np.float64)

    A = pdhcg_csr_view(
        source.constraint_matrix,
        m,
        n,
        source.constraint_matrix_num_nonzeros,
    )
    Q = pdhcg_csr_view(
        source.objective_sparse_matrix,
        n,
        n,
        source.objective_sparse_matrix_num_nonzeros,
    )
    R = CsrMatrix(0, n, 0, None, None, None)
    problem = ProblemData(
        n,
        A,
        source.constraint_lower_bound,
        source.constraint_upper_bound,
        Q,
        PREFOS_Q_FULL,
        R,
        None,
        source.objective_vector,
        source.objective_constant,
        len(box_indices),
        ptr(box_indices, ct.c_int),
        ptr(box_lower, ct.c_double),
        ptr(box_upper, ct.c_double),
        cone_count,
        ct.cast(cone_array, ct.POINTER(ConeBlock)),
    )
    keepalive = (all_cone_indices, cone_array, box_indices, box_lower, box_upper)
    metadata = {
        "soc_cones": soc_count,
        "rsoc_cones": rsoc_count,
        "cone_variables": total_cone_dimension,
        "box_variables": len(box_indices),
    }
    return problem, keepalive, metadata


def stats_dict(stats):
    return {
        name: (
            float(getattr(stats, name))
            if field_type is ct.c_double
            else int(getattr(stats, name))
        )
        for name, field_type in PreFOSStats._fields_
    }


def benchmark_file(prefos, parser, filename, setting_overrides):
    result = {
        "file": str(filename),
        "compressed_mib": filename.stat().st_size / (1024.0 * 1024.0),
        "rss_baseline_mib": current_rss_mib(),
    }
    parser_problem = None
    presolver = ct.c_void_p()
    try:
        start = time.perf_counter()
        parser_problem = parser.read_cbf_file(str(filename).encode())
        result["parse_seconds"] = time.perf_counter() - start
        result["rss_after_parse_mib"] = current_rss_mib()
        if not parser_problem:
            raise RuntimeError("PDHCG CBF parser failed")

        start = time.perf_counter()
        problem, keepalive, metadata = make_prefos_view(parser_problem)
        result["adapter_seconds"] = time.perf_counter() - start
        result.update(metadata)

        settings = prefos.prefos_default_settings()
        for name, value in setting_overrides.items():
            setattr(settings, name, value)
        result["settings"] = setting_overrides
        start = time.perf_counter()
        create_status = prefos.prefos_create_presolver(
            ct.byref(problem), ct.byref(settings), ct.byref(presolver)
        )
        result["create_seconds"] = time.perf_counter() - start
        result["create_status"] = int(create_status)
        result["rss_after_create_mib"] = current_rss_mib()
        del keepalive
        parser.qp_problem_free(parser_problem)
        parser_problem = None
        if create_status != 0:
            raise RuntimeError(f"prefos_create_presolver returned {create_status}")

        start = time.perf_counter()
        presolve_status = prefos.prefos_run_presolve(presolver)
        result["presolve_seconds"] = time.perf_counter() - start
        result["presolve_status"] = int(presolve_status)
        result["rss_after_presolve_mib"] = current_rss_mib()
        result["rss_delta_after_presolve_mib"] = (
            result["rss_after_presolve_mib"] - result["rss_baseline_mib"]
        )
        result["peak_rss_mib"] = peak_rss_mib()
        if presolve_status not in (0, 1, 2):
            raise RuntimeError(f"prefos_run_presolve returned {presolve_status}")
        result.update(stats_dict(prefos.prefos_get_stats(presolver).contents))
        result["variable_reduction_percent"] = (
            100.0
            * (result["variables_original"] - result["variables_reduced"])
            / max(1, result["variables_original"])
        )
        result["row_reduction_percent"] = (
            100.0
            * (result["rows_original"] - result["rows_reduced"])
            / max(1, result["rows_original"])
        )
        result["nnz_A_reduction_percent"] = (
            100.0
            * (result["nnz_A_original"] - result["nnz_A_reduced"])
            / max(1, result["nnz_A_original"])
        )
        return result
    finally:
        if parser_problem:
            parser.qp_problem_free(parser_problem)
        if presolver.value:
            prefos.prefos_free_presolver(presolver)


def expand_inputs(inputs):
    files = []
    for input_name in inputs:
        path = Path(input_name).resolve()
        if path.is_dir():
            files.extend(sorted(path.glob("*.cbf")))
            files.extend(sorted(path.glob("*.cbf.gz")))
        else:
            files.append(path)
    return files


def print_summary(result):
    direct_face_count = sum(
        result[name]
        for name in (
            "reduced_rsoc_faces",
            "reduced_psd_faces",
            "reduced_exponential_faces",
            "reduced_power_faces",
        )
    )
    affine_face_count = sum(
        result[name]
        for name in (
            "reduced_affine_rsoc_faces",
            "reduced_affine_psd_faces",
            "reduced_affine_exponential_faces",
            "reduced_affine_power_faces",
        )
    )
    print(
        f"{Path(result['file']).name}: "
        f"parse={result['parse_seconds']:.3f}s "
        f"adapt={result['adapter_seconds']:.3f}s "
        f"create={result['create_seconds']:.3f}s "
        f"presolve={result['presolve_seconds']:.3f}s | "
        f"n={result['variables_original']:,}->{result['variables_reduced']:,} "
        f"m={result['rows_original']:,}->{result['rows_reduced']:,} "
        f"nnzA={result['nnz_A_original']:,}->{result['nnz_A_reduced']:,} | "
        f"fixed={result['fixed_box_variables'] + result['fixed_cone_variables']:,} "
        f"substituted={result['substituted_free_variables']:,} "
        f"ternary={result['ternary_substituted_free_variables']:,} "
        f"collapsed={result['collapsed_cones']:,} "
        f"faces=direct:{direct_face_count:,},affine:{affine_face_count:,} "
        f"affine_coords_removed={result['removed_affine_cone_coordinates']:,} "
        f"affine_face={result['affine_face_reduction_milliseconds']:.1f}ms "
        f"affine_eq={result['derived_affine_face_equalities']:,} "
        f"affine_eq_sub={result['affine_face_substitution_milliseconds']:.1f}ms "
        f"affine_agg=blocks:{result['generated_affine_cone_blocks']:,},"
        f"coords:{result['aggregated_affine_cone_coordinates']:,} "
        f"linear_rounds=event:{result['linear_event_rounds']},"
        f"full:{result['linear_full_scan_rounds']} "
        f"stops=budget:{result['linear_budget_stops']},"
        f"stale:{result['linear_stale_stops']} "
        f"linear:{result['linear_propagation_milliseconds']:.1f}ms "
        f"parallel-row:{result['parallel_row_detection_milliseconds']:.1f}ms "
        f"parallel-skips:{result['parallel_row_budget_skips']} "
        f"row-activity:{result['redundant_row_activity_milliseconds']:.1f}ms "
        f"row-activity-skips:{result['redundant_row_activity_budget_skips']} "
        f"cone:{result['cone_propagation_milliseconds']:.1f}ms "
        f"gpu=rounds:{result['linear_gpu_rounds']},"
        f"fallbacks:{result['linear_gpu_fallbacks']},"
        f"setup:{result['linear_gpu_setup_milliseconds']:.1f}ms,"
        f"transfer:{result['linear_gpu_transfer_milliseconds']:.1f}ms,"
        f"kernel:{result['linear_gpu_kernel_milliseconds']:.1f}ms "
        f"total:{result['linear_gpu_total_milliseconds']:.1f}ms "
        f"long_rows:{result['linear_gpu_long_rows']} "
        f"gpu-struct=workspace:"
        f"{result['cuda_workspace_setup_milliseconds']:.1f}ms,"
        f"csc:{result['column_csc_gpu_builds']}/"
        f"{result['column_csc_gpu_milliseconds']:.1f}ms,"
        f"singleton-col:{result['singleton_column_gpu_passes']}/"
        f"{result['singleton_column_gpu_milliseconds']:.1f}ms,"
        f"parallel-col:{result['parallel_column_gpu_passes']}/"
        f"{result['parallel_column_gpu_milliseconds']:.1f}ms,"
        f"parallel:{result['parallel_row_gpu_passes']}/"
        f"{result['parallel_row_gpu_milliseconds']:.1f}ms,"
        f"activity:{result['cone_activity_gpu_passes']}/"
        f"{result['cone_activity_gpu_milliseconds']:.1f}ms,"
        f"cone:{result['cone_gpu_rounds']}/"
        f"{result['cone_gpu_milliseconds']:.1f}ms,"
        f"cone-linear:{result['cone_gpu_linear_rounds']}/"
        f"{result['cone_gpu_linear_kernel_milliseconds']:.1f}ms,"
        f"affine:{result['affine_cone_gpu_rounds']}/"
        f"{result['affine_cone_gpu_milliseconds']:.1f}ms,"
        f"compact:{result['matrix_compaction_gpu_passes']}/"
        f"{result['matrix_compaction_gpu_milliseconds']:.1f}ms "
        f"rss={result['rss_after_presolve_mib']:.1f}MiB"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--library", default="build/libPreFOS.so")
    parser.add_argument(
        "--pdhcg-library",
        required=True,
        help="path to the PDHCG-II shared library containing the CBF parser",
    )
    parser.add_argument("--jsonl")
    parser.add_argument("--max-linear-rounds", type=int)
    parser.add_argument("--max-cone-rounds", type=int)
    parser.add_argument("--disable-linear-propagation", action="store_true")
    parser.add_argument("--gpu-linear-propagation", action="store_true")
    parser.add_argument("--warmup-gpu", action="store_true")
    parser.add_argument("--async-warmup-gpu", action="store_true")
    parser.add_argument("--disable-cone-propagation", action="store_true")
    parser.add_argument("--disable-cone-aware-activity", action="store_true")
    parser.add_argument("--disable-exponential-propagation", action="store_true")
    parser.add_argument("--disable-power-propagation", action="store_true")
    parser.add_argument(
        "--disable-psd-higher-order-propagation", action="store_true"
    )
    parser.add_argument("--disable-redundant-rows", action="store_true")
    parser.add_argument("--disable-free-column-substitution", action="store_true")
    parser.add_argument("--max-aggregation-terms", type=int)
    parser.add_argument("--max-aggregation-degree", type=int)
    parser.add_argument("--max-aggregation-fill", type=int)
    parser.add_argument("--max-aggregation-rounds", type=int)
    parser.add_argument("--max-aggregation-scale", type=float)
    parser.add_argument("--finite-bound-absolute", type=float)
    parser.add_argument("--finite-bound-relative", type=float)
    parser.add_argument("--event-max-average-degree", type=float)
    parser.add_argument("--event-update-ratio", type=float)
    parser.add_argument("--linear-work-ratio", type=float)
    parser.add_argument("--linear-min-changes-per-million", type=float)
    parser.add_argument("--linear-max-stale-rounds", type=int)
    parser.add_argument("--parallel-row-max-average-nnz", type=float)
    parser.add_argument("--redundant-row-max-average-nnz", type=float)
    parser.add_argument(
        "--propagated-bound-policy",
        choices=("first-order", "interior-point"),
        default="first-order",
    )
    parser.add_argument("--affine-cone-aggregation", action="store_true")
    args = parser.parse_args()
    if args.warmup_gpu and args.async_warmup_gpu:
        parser.error("--warmup-gpu and --async-warmup-gpu are mutually exclusive")

    setting_overrides = {}
    if args.propagated_bound_policy == "interior-point":
        setting_overrides["propagated_bound_policy"] = 1
    if args.affine_cone_aggregation:
        setting_overrides["affine_cone_coordinate_aggregation"] = 1
    if args.max_linear_rounds is not None:
        setting_overrides["max_linear_propagation_rounds"] = args.max_linear_rounds
    if args.max_cone_rounds is not None:
        setting_overrides["max_cone_propagation_rounds"] = args.max_cone_rounds
    if args.disable_linear_propagation:
        setting_overrides["linear_propagation"] = 0
    if args.gpu_linear_propagation:
        setting_overrides["linear_propagation_gpu"] = 1
        setting_overrides["structural_reductions_gpu"] = 1
    if args.disable_cone_propagation:
        setting_overrides["cone_propagation"] = 0
    if args.disable_cone_aware_activity:
        setting_overrides["cone_aware_row_activity"] = 0
    if args.disable_exponential_propagation:
        setting_overrides["exponential_propagation"] = 0
    if args.disable_power_propagation:
        setting_overrides["power_propagation"] = 0
    if args.disable_psd_higher_order_propagation:
        setting_overrides["psd_higher_order_propagation"] = 0
    if args.disable_redundant_rows:
        setting_overrides["remove_redundant_rows"] = 0
    if args.disable_free_column_substitution:
        setting_overrides["free_column_substitution"] = 0
    if args.max_aggregation_terms is not None:
        setting_overrides["max_aggregation_terms"] = args.max_aggregation_terms
    if args.max_aggregation_degree is not None:
        setting_overrides["max_aggregation_column_degree"] = (
            args.max_aggregation_degree
        )
    if args.max_aggregation_fill is not None:
        setting_overrides["max_aggregation_fill"] = args.max_aggregation_fill
    if args.max_aggregation_rounds is not None:
        setting_overrides["max_aggregation_rounds"] = args.max_aggregation_rounds
    if args.max_aggregation_scale is not None:
        setting_overrides["max_aggregation_scale"] = args.max_aggregation_scale
    if args.finite_bound_absolute is not None:
        setting_overrides["finite_bound_improvement_absolute"] = (
            args.finite_bound_absolute
        )
    if args.finite_bound_relative is not None:
        setting_overrides["finite_bound_improvement_relative"] = (
            args.finite_bound_relative
        )
    if args.event_max_average_degree is not None:
        setting_overrides["event_queue_max_average_column_degree"] = (
            args.event_max_average_degree
        )
    if args.event_update_ratio is not None:
        setting_overrides["event_queue_activity_update_ratio"] = (
            args.event_update_ratio
        )
    if args.linear_work_ratio is not None:
        setting_overrides["linear_propagation_max_work_ratio"] = (
            args.linear_work_ratio
        )
    if args.linear_min_changes_per_million is not None:
        setting_overrides["linear_propagation_min_changes_per_million"] = (
            args.linear_min_changes_per_million
        )
    if args.linear_max_stale_rounds is not None:
        setting_overrides["linear_propagation_max_stale_rounds"] = (
            args.linear_max_stale_rounds
        )
    if args.parallel_row_max_average_nnz is not None:
        setting_overrides["parallel_row_max_average_nnz"] = (
            args.parallel_row_max_average_nnz
        )
    if args.redundant_row_max_average_nnz is not None:
        setting_overrides["redundant_row_max_average_nnz"] = (
            args.redundant_row_max_average_nnz
        )

    prefos = configure_prefos(Path(args.library).resolve())
    if args.warmup_gpu and not prefos.prefos_gpu_warmup():
        raise SystemExit("CUDA warmup failed or this is a CPU-only build")
    if args.async_warmup_gpu and not prefos.prefos_gpu_warmup_async():
        raise SystemExit("CUDA asynchronous warmup could not be started")
    cbf_parser = configure_parser(Path(args.pdhcg_library).resolve())
    files = expand_inputs(args.inputs)
    if not files:
        raise SystemExit("no CBF files found")

    output = open(args.jsonl, "a", encoding="utf-8") if args.jsonl else None
    failures = 0
    try:
        for filename in files:
            try:
                result = benchmark_file(
                    prefos, cbf_parser, filename, setting_overrides
                )
                print_summary(result)
                if output:
                    output.write(json.dumps(result, sort_keys=True) + "\n")
                    output.flush()
            except Exception as error:
                failures += 1
                print(f"{filename.name}: FAILED: {error}")
                if output:
                    output.write(
                        json.dumps({"file": str(filename), "error": str(error)})
                        + "\n"
                    )
                    output.flush()
    finally:
        if output:
            output.close()
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
