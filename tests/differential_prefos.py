#!/usr/bin/env python3
# Copyright 2026 Hongpei Li
# SPDX-License-Identifier: Apache-2.0

"""Randomized original-vs-presolved differential check for box QPs."""

import argparse
import ctypes as ct
from pathlib import Path

import numpy as np
import scipy.sparse as sp


class CsrMatrix(ct.Structure):
    _fields_ = [
        ("rows", ct.c_size_t),
        ("cols", ct.c_size_t),
        ("nnz", ct.c_size_t),
        ("values", ct.POINTER(ct.c_double)),
        ("column_indices", ct.POINTER(ct.c_int)),
        ("row_pointers", ct.POINTER(ct.c_int)),
    ]


class ConeBlock(ct.Structure):
    _fields_ = [
        ("type", ct.c_int),
        ("dimension", ct.c_size_t),
        ("matrix_order", ct.c_size_t),
        ("indices", ct.POINTER(ct.c_int)),
        ("power_alpha", ct.c_double),
    ]


class AffineConeBlock(ct.Structure):
    _fields_ = [
        ("type", ct.c_int),
        ("dimension", ct.c_size_t),
        ("matrix_order", ct.c_size_t),
        ("power_alpha", ct.c_double),
    ]


class ProblemData(ct.Structure):
    _fields_ = [
        ("n", ct.c_size_t),
        ("A", CsrMatrix),
        ("constraint_lower", ct.POINTER(ct.c_double)),
        ("constraint_upper", ct.POINTER(ct.c_double)),
        ("Q", CsrMatrix),
        ("q_storage", ct.c_int),
        ("R", CsrMatrix),
        ("D", ct.POINTER(ct.c_double)),
        ("c", ct.POINTER(ct.c_double)),
        ("objective_offset", ct.c_double),
        ("n_box", ct.c_size_t),
        ("box_indices", ct.POINTER(ct.c_int)),
        ("box_lower", ct.POINTER(ct.c_double)),
        ("box_upper", ct.POINTER(ct.c_double)),
        ("n_cones", ct.c_size_t),
        ("cones", ct.POINTER(ConeBlock)),
        ("affine_cone_matrix", CsrMatrix),
        ("affine_cone_offset", ct.POINTER(ct.c_double)),
        ("n_affine_cones", ct.c_size_t),
        ("affine_cones", ct.POINTER(AffineConeBlock)),
    ]


class Settings(ct.Structure):
    _fields_ = [
        ("feasibility_tolerance", ct.c_double),
        ("fixed_variable_tolerance", ct.c_double),
        ("fix_close_box_bounds", ct.c_int),
        ("remove_empty_rows", ct.c_int),
        ("remove_redundant_rows", ct.c_int),
        ("free_column_substitution", ct.c_int),
        ("max_aggregation_terms", ct.c_int),
        ("max_aggregation_column_degree", ct.c_int),
        ("max_aggregation_fill", ct.c_int),
        ("max_aggregation_rounds", ct.c_int),
        ("max_aggregation_scale", ct.c_double),
        ("linear_propagation", ct.c_int),
        ("max_linear_propagation_rounds", ct.c_int),
        ("linear_propagation_gpu", ct.c_int),
        ("linear_propagation_max_work_ratio", ct.c_double),
        ("linear_propagation_min_changes_per_million", ct.c_double),
        ("linear_propagation_max_stale_rounds", ct.c_int),
        ("cone_propagation", ct.c_int),
        ("max_cone_propagation_rounds", ct.c_int),
        ("cone_aware_row_activity", ct.c_int),
        ("exponential_propagation", ct.c_int),
        ("power_propagation", ct.c_int),
        ("psd_higher_order_propagation", ct.c_int),
        ("affine_psd_propagation_max_work_ratio", ct.c_double),
        ("rsoc_face_reduction", ct.c_int),
        ("psd_face_reduction", ct.c_int),
        ("exponential_face_reduction", ct.c_int),
        ("power_face_reduction", ct.c_int),
        ("finite_bound_improvement_absolute", ct.c_double),
        ("finite_bound_improvement_relative", ct.c_double),
        ("event_queue_max_average_column_degree", ct.c_double),
        ("event_queue_activity_update_ratio", ct.c_double),
        ("propagated_bound_policy", ct.c_int),
        ("affine_cone_coordinate_aggregation", ct.c_int),
        ("psd_structure_analysis", ct.c_int),
        ("psd_block_decomposition", ct.c_int),
    ]


class PresolvedProblem(ct.Structure):
    _fields_ = ProblemData._fields_


class PSDStructureAnalysis(ct.Structure):
    _fields_ = [
        ("affine_cone_index", ct.c_size_t),
        ("row_start", ct.c_size_t),
        ("matrix_order", ct.c_size_t),
        ("dimension", ct.c_size_t),
        ("affine_nnz", ct.c_size_t),
        ("active_diagonal_coordinates", ct.c_size_t),
        ("active_offdiagonal_coordinates", ct.c_size_t),
        ("coefficient_columns", ct.c_size_t),
        ("diagonal_coefficient_columns", ct.c_size_t),
        ("single_diagonal_coefficient_columns", ct.c_size_t),
        ("connected_components", ct.c_size_t),
        ("scalar_components", ct.c_size_t),
        ("emitted_cone_blocks", ct.c_size_t),
        ("largest_component_order", ct.c_size_t),
        ("decomposed_dimension", ct.c_size_t),
        ("exactly_block_diagonal", ct.c_int),
    ]


class Verification(ct.Structure):
    _fields_ = [
        ("original_objective", ct.c_double),
        ("reduced_objective", ct.c_double),
        ("objective_absolute_error", ct.c_double),
        ("original_row_violation", ct.c_double),
        ("original_box_violation", ct.c_double),
        ("original_cone_violation", ct.c_double),
        ("reduced_row_violation", ct.c_double),
        ("reduced_box_violation", ct.c_double),
        ("reduced_cone_violation", ct.c_double),
        ("passed", ct.c_int),
    ]


class KktResiduals(ct.Structure):
    _fields_ = [
        ("row_primal_violation", ct.c_double),
        ("box_primal_violation", ct.c_double),
        ("cone_primal_violation", ct.c_double),
        ("stationarity_violation", ct.c_double),
        ("row_dual_violation", ct.c_double),
        ("domain_dual_violation", ct.c_double),
        ("complementarity_violation", ct.c_double),
        ("passed", ct.c_int),
    ]


class PostsolveKktVerification(ct.Structure):
    _fields_ = [
        ("reduced", KktResiduals),
        ("original", KktResiduals),
        ("objective_absolute_error", ct.c_double),
        ("passed", ct.c_int),
    ]


def ptr(array, c_type):
    if array.size == 0:
        return ct.POINTER(c_type)()
    return array.ctypes.data_as(ct.POINTER(c_type))


def as_csr(matrix):
    matrix = sp.csr_matrix(matrix)
    matrix.sort_indices()
    values = np.ascontiguousarray(matrix.data, dtype=np.float64)
    indices = np.ascontiguousarray(matrix.indices, dtype=np.int32)
    indptr = np.ascontiguousarray(matrix.indptr, dtype=np.int32)
    view = CsrMatrix(
        matrix.shape[0],
        matrix.shape[1],
        matrix.nnz,
        ptr(values, ct.c_double),
        ptr(indices, ct.c_int),
        ptr(indptr, ct.c_int),
    )
    return view, (values, indices, indptr)


def numpy_csr(view):
    if view.nnz == 0:
        return sp.csr_matrix((view.rows, view.cols), dtype=np.float64)
    values = np.ctypeslib.as_array(view.values, shape=(view.nnz,)).copy()
    indices = np.ctypeslib.as_array(
        view.column_indices, shape=(view.nnz,)
    ).copy()
    indptr = np.ctypeslib.as_array(
        view.row_pointers, shape=(view.rows + 1,)
    ).copy()
    return sp.csr_matrix((values, indices, indptr), shape=(view.rows, view.cols))


def numpy_vector(pointer, length, dtype=np.float64):
    if length == 0:
        return np.empty(0, dtype=dtype)
    return np.ctypeslib.as_array(pointer, shape=(length,)).copy().astype(dtype)


def symmetric_q(view, storage):
    triangular = numpy_csr(view)
    if storage == 2:
        return triangular
    diagonal = sp.diags(triangular.diagonal())
    return triangular + triangular.T - diagonal


def solve_box_qp(A, lower, upper, Q, R, D, c, box_indices, box_lower,
                 box_upper, offset):
    import osqp

    n = Q.shape[0]
    hessian = Q + R.T @ sp.diags(D) @ R
    selector = sp.csr_matrix(
        (
            np.ones(len(box_indices)),
            (np.arange(len(box_indices)), box_indices),
        ),
        shape=(len(box_indices), n),
    )
    operator = sp.vstack((A, selector), format="csc")
    all_lower = np.concatenate((lower, box_lower))
    all_upper = np.concatenate((upper, box_upper))
    solver = osqp.OSQP()
    solver.setup(
        P=sp.triu(hessian, format="csc"),
        q=c,
        A=operator,
        l=all_lower,
        u=all_upper,
        eps_abs=1e-9,
        eps_rel=1e-9,
        max_iter=100000,
        polishing=False,
        verbose=False,
    )
    result = solver.solve()
    if result.info.status_val not in (1, 2):
        raise AssertionError(f"OSQP failed: {result.info.status}")
    row_dual = result.y[: A.shape[0]]
    box_dual = result.y[A.shape[0] :]
    return result.x, result.info.obj_val + offset, row_dual, box_dual


def configure_library(path):
    library = ct.CDLL(str(path))
    library.prefos_strict_settings.restype = Settings
    library.prefos_create_presolver.argtypes = [
        ct.POINTER(ProblemData),
        ct.POINTER(Settings),
        ct.POINTER(ct.c_void_p),
    ]
    library.prefos_create_presolver.restype = ct.c_int
    library.prefos_run_presolve.argtypes = [ct.c_void_p]
    library.prefos_run_presolve.restype = ct.c_int
    library.prefos_get_reduced_problem.argtypes = [ct.c_void_p]
    library.prefos_get_reduced_problem.restype = ct.POINTER(PresolvedProblem)
    library.prefos_get_psd_structure_analyses.argtypes = [
        ct.c_void_p,
        ct.POINTER(ct.c_size_t),
    ]
    library.prefos_get_psd_structure_analyses.restype = ct.POINTER(
        PSDStructureAnalysis
    )
    library.prefos_verify_postsolve_primal.argtypes = [
        ct.c_void_p,
        ct.POINTER(ct.c_double),
        ct.c_double,
        ct.POINTER(Verification),
    ]
    library.prefos_verify_postsolve_primal.restype = ct.c_int
    library.prefos_verify_postsolve_kkt.argtypes = [
        ct.c_void_p,
        ct.POINTER(ct.c_double),
        ct.POINTER(ct.c_double),
        ct.POINTER(ct.c_double),
        ct.c_double,
        ct.POINTER(PostsolveKktVerification),
    ]
    library.prefos_verify_postsolve_kkt.restype = ct.c_int
    library.prefos_free_presolver.argtypes = [ct.c_void_p]
    return library


def random_problem(rng):
    n = int(rng.integers(3, 8))
    m = int(rng.integers(2, 9))
    rank = int(rng.integers(1, 4))
    planted = rng.uniform(-1.0, 1.0, n)
    A = rng.normal(size=(m, n))
    activity = A @ planted
    slack = rng.uniform(0.2, 2.0, m)
    lower = activity - slack
    upper = activity + slack
    lower[::3] = -np.inf
    upper[1::3] = np.inf

    box_lower = np.full(n, -2.0)
    box_upper = np.full(n, 2.0)
    box_lower[0] = planted[0]
    box_upper[0] = planted[0]

    factor = rng.normal(scale=0.2, size=(n, n))
    Q = factor.T @ factor + 0.05 * np.eye(n)
    R = rng.normal(scale=0.3, size=(rank, n))
    D = rng.uniform(0.1, 1.5, rank)
    c = rng.normal(scale=0.5, size=n)
    offset = float(rng.normal(scale=0.2))
    return A, lower, upper, Q, R, D, c, box_lower, box_upper, offset


def run_seed(library, seed, use_gpu=False, propagated_bound_policy=0):
    rng = np.random.default_rng(seed)
    A, lower, upper, Q, R, D, c, box_lower, box_upper, offset = (
        random_problem(rng)
    )
    n = Q.shape[0]
    box_indices = np.arange(n, dtype=np.int32)
    A_view, A_keepalive = as_csr(A)
    Q_view, Q_keepalive = as_csr(sp.triu(Q, format="csr"))
    R_view, R_keepalive = as_csr(R)
    lower = np.ascontiguousarray(lower, dtype=np.float64)
    upper = np.ascontiguousarray(upper, dtype=np.float64)
    D = np.ascontiguousarray(D, dtype=np.float64)
    c = np.ascontiguousarray(c, dtype=np.float64)
    box_lower = np.ascontiguousarray(box_lower, dtype=np.float64)
    box_upper = np.ascontiguousarray(box_upper, dtype=np.float64)

    problem = ProblemData(
        n,
        A_view,
        ptr(lower, ct.c_double),
        ptr(upper, ct.c_double),
        Q_view,
        0,
        R_view,
        ptr(D, ct.c_double),
        ptr(c, ct.c_double),
        offset,
        n,
        ptr(box_indices, ct.c_int),
        ptr(box_lower, ct.c_double),
        ptr(box_upper, ct.c_double),
        0,
        ct.POINTER(ConeBlock)(),
    )
    original_x, original_objective, original_y, original_z = solve_box_qp(
        sp.csr_matrix(A), lower, upper, sp.csr_matrix(Q), sp.csr_matrix(R),
        D, c, box_indices, box_lower, box_upper, offset
    )
    del original_x, original_y, original_z

    settings = library.prefos_strict_settings()
    settings.propagated_bound_policy = propagated_bound_policy
    if use_gpu:
        settings.linear_propagation_gpu = 1
        settings.event_queue_max_average_column_degree = 0.0
    presolver = ct.c_void_p()
    status = library.prefos_create_presolver(
        ct.byref(problem), ct.byref(settings), ct.byref(presolver)
    )
    if status != 0:
        raise AssertionError(f"create failed for seed {seed}: status {status}")
    try:
        status = library.prefos_run_presolve(presolver)
        if status not in (0, 1):
            raise AssertionError(f"presolve failed for seed {seed}: status {status}")
        reduced = library.prefos_get_reduced_problem(presolver).contents
        reduced_A = numpy_csr(reduced.A)
        reduced_Q = symmetric_q(reduced.Q, reduced.q_storage)
        reduced_R = numpy_csr(reduced.R)
        reduced_lower = numpy_vector(
            reduced.constraint_lower, reduced.A.rows
        )
        reduced_upper = numpy_vector(
            reduced.constraint_upper, reduced.A.rows
        )
        reduced_D = numpy_vector(reduced.D, reduced.R.rows)
        reduced_c = numpy_vector(reduced.c, reduced.n)
        reduced_box_indices = numpy_vector(
            reduced.box_indices, reduced.n_box, np.int32
        )
        reduced_box_lower = numpy_vector(reduced.box_lower, reduced.n_box)
        reduced_box_upper = numpy_vector(reduced.box_upper, reduced.n_box)
        reduced_x, reduced_objective, reduced_y, reduced_box_dual = solve_box_qp(
            reduced_A,
            reduced_lower,
            reduced_upper,
            reduced_Q,
            reduced_R,
            reduced_D,
            reduced_c,
            reduced_box_indices,
            reduced_box_lower,
            reduced_box_upper,
            reduced.objective_offset,
        )

        verification = Verification()
        reduced_x = np.ascontiguousarray(reduced_x, dtype=np.float64)
        status = library.prefos_verify_postsolve_primal(
            presolver,
            ptr(reduced_x, ct.c_double),
            2e-6,
            ct.byref(verification),
        )
        if status != 0 or not verification.passed:
            raise AssertionError(
                f"round-trip failed for seed {seed}: status={status}, "
                f"row={verification.original_row_violation:.3e}, "
                f"box={verification.original_box_violation:.3e}, "
                f"objerr={verification.objective_absolute_error:.3e}"
            )
        scale = max(1.0, abs(original_objective), abs(reduced_objective))
        if abs(original_objective - reduced_objective) > 2e-5 * scale:
            raise AssertionError(
                f"optimal values differ for seed {seed}: "
                f"{original_objective} vs {reduced_objective}"
            )

        reduced_z = np.zeros(reduced.n, dtype=np.float64)
        reduced_z[reduced_box_indices] = reduced_box_dual
        reduced_y = np.ascontiguousarray(reduced_y, dtype=np.float64)
        kkt_verification = PostsolveKktVerification()
        status = library.prefos_verify_postsolve_kkt(
            presolver,
            ptr(reduced_x, ct.c_double),
            ptr(reduced_y, ct.c_double),
            ptr(reduced_z, ct.c_double),
            5e-5,
            ct.byref(kkt_verification),
        )
        if status != 0 or not kkt_verification.passed:
            raise AssertionError(
                f"KKT postsolve failed for seed {seed}: status={status}, "
                f"reduced_stationarity="
                f"{kkt_verification.reduced.stationarity_violation:.3e}, "
                f"original_stationarity="
                f"{kkt_verification.original.stationarity_violation:.3e}, "
                f"original_row_dual="
                f"{kkt_verification.original.row_dual_violation:.3e}, "
                f"original_domain_dual="
                f"{kkt_verification.original.domain_dual_violation:.3e}, "
                f"original_complementarity="
                f"{kkt_verification.original.complementarity_violation:.3e}, "
                f"original_primal="
                f"{max(kkt_verification.original.row_primal_violation, kkt_verification.original.box_primal_violation):.3e}"
            )
    finally:
        library.prefos_free_presolver(presolver)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", default="build/libPreFOS.so")
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--gpu-linear-propagation", action="store_true")
    parser.add_argument(
        "--propagated-bound-policy",
        choices=("first-order", "interior-point"),
        default="first-order",
    )
    args = parser.parse_args()
    library_path = Path(args.library).resolve()
    if not library_path.exists():
        raise SystemExit(f"library not found: {library_path}")
    library = configure_library(library_path)
    propagated_bound_policy = int(
        args.propagated_bound_policy == "interior-point"
    )
    for seed in range(args.start_seed, args.start_seed + args.seeds):
        run_seed(
            library, seed, args.gpu_linear_propagation, propagated_bound_policy
        )
    print(
        f"Differential Conic QP check passed for {args.seeds} seeds "
        f"starting at {args.start_seed}."
    )


if __name__ == "__main__":
    main()
