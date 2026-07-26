#!/usr/bin/env python3
# Copyright 2026 Hongpei Li
# SPDX-License-Identifier: Apache-2.0

"""Randomized postsolve checks for Farkas certificates and recession rays."""

import argparse
import ctypes as ct
from pathlib import Path

import numpy as np
import scipy.optimize as opt
import scipy.sparse as sp

from differential_prefos import (
    AffineConeBlock,
    ConeBlock,
    ProblemData,
    as_csr,
    configure_library,
    numpy_csr,
    numpy_vector,
    ptr,
)


class InfeasibilityResiduals(ct.Structure):
    _fields_ = [
        ("stationarity_violation", ct.c_double),
        ("row_dual_violation", ct.c_double),
        ("domain_dual_violation", ct.c_double),
        ("certificate_value", ct.c_double),
        ("passed", ct.c_int),
    ]


class CertificateVerification(ct.Structure):
    _fields_ = [
        ("reduced", InfeasibilityResiduals),
        ("original", InfeasibilityResiduals),
        ("certificate_value_absolute_error", ct.c_double),
        ("passed", ct.c_int),
    ]


class RayResiduals(ct.Structure):
    _fields_ = [
        ("row_recession_violation", ct.c_double),
        ("domain_recession_violation", ct.c_double),
        ("quadratic_null_violation", ct.c_double),
        ("objective_direction", ct.c_double),
        ("passed", ct.c_int),
    ]


class RayVerification(ct.Structure):
    _fields_ = [
        ("reduced", RayResiduals),
        ("original", RayResiduals),
        ("objective_direction_absolute_error", ct.c_double),
        ("passed", ct.c_int),
    ]


def configure_certificate_api(library):
    certificate_arguments = [
        ct.c_void_p,
        ct.POINTER(ct.c_double),
        ct.POINTER(ct.c_double),
        ct.POINTER(ct.c_double),
        ct.c_double,
        ct.POINTER(CertificateVerification),
    ]
    library.prefos_verify_postsolve_infeasibility_certificate.argtypes = (
        certificate_arguments
    )
    library.prefos_verify_postsolve_infeasibility_certificate.restype = ct.c_int
    library.prefos_verify_postsolve_unbounded_ray.argtypes = [
        ct.c_void_p,
        ct.POINTER(ct.c_double),
        ct.c_double,
        ct.POINTER(RayVerification),
    ]
    library.prefos_verify_postsolve_unbounded_ray.restype = ct.c_int


def make_problem(A, lower, upper, c, box_lower, box_upper, G=None, h=None):
    n = c.size
    A_view, A_keepalive = as_csr(A)
    Q_view, Q_keepalive = as_csr(sp.csr_matrix((n, n)))
    R_view, R_keepalive = as_csr(sp.csr_matrix((0, n)))
    G = sp.csr_matrix((0, n)) if G is None else sp.csr_matrix(G)
    G_view, G_keepalive = as_csr(G)

    lower = np.ascontiguousarray(lower, dtype=np.float64)
    upper = np.ascontiguousarray(upper, dtype=np.float64)
    c = np.ascontiguousarray(c, dtype=np.float64)
    box_lower = np.ascontiguousarray(box_lower, dtype=np.float64)
    box_upper = np.ascontiguousarray(box_upper, dtype=np.float64)
    box_indices = np.arange(n, dtype=np.int32)
    h = np.empty(0, dtype=np.float64) if h is None else np.ascontiguousarray(
        h, dtype=np.float64
    )
    affine_blocks = None
    if G.shape[0] > 0:
        affine_blocks = (AffineConeBlock * 1)(
            AffineConeBlock(0, G.shape[0], 0, 0.0)
        )

    problem = ProblemData()
    problem.n = n
    problem.A = A_view
    problem.constraint_lower = ptr(lower, ct.c_double)
    problem.constraint_upper = ptr(upper, ct.c_double)
    problem.Q = Q_view
    problem.q_storage = 0
    problem.R = R_view
    problem.D = ct.POINTER(ct.c_double)()
    problem.c = ptr(c, ct.c_double)
    problem.objective_offset = 0.0
    problem.n_box = n
    problem.box_indices = ptr(box_indices, ct.c_int)
    problem.box_lower = ptr(box_lower, ct.c_double)
    problem.box_upper = ptr(box_upper, ct.c_double)
    problem.n_cones = 0
    problem.cones = ct.POINTER(ConeBlock)()
    problem.affine_cone_matrix = G_view
    problem.affine_cone_offset = ptr(h, ct.c_double)
    problem.n_affine_cones = 1 if G.shape[0] else 0
    problem.affine_cones = (
        ct.cast(affine_blocks, ct.POINTER(AffineConeBlock))
        if affine_blocks is not None
        else ct.POINTER(AffineConeBlock)()
    )
    keepalive = (
        A_keepalive,
        Q_keepalive,
        R_keepalive,
        G_keepalive,
        lower,
        upper,
        c,
        box_indices,
        box_lower,
        box_upper,
        h,
        affine_blocks,
    )
    return problem, keepalive


def reduced_arrays(reduced):
    return {
        "A": numpy_csr(reduced.A),
        "lower": numpy_vector(reduced.constraint_lower, reduced.A.rows),
        "upper": numpy_vector(reduced.constraint_upper, reduced.A.rows),
        "c": numpy_vector(reduced.c, reduced.n),
        "box_indices": numpy_vector(
            reduced.box_indices, reduced.n_box, np.int32
        ),
        "box_lower": numpy_vector(reduced.box_lower, reduced.n_box),
        "box_upper": numpy_vector(reduced.box_upper, reduced.n_box),
        "G": numpy_csr(reduced.affine_cone_matrix),
        "h": numpy_vector(
            reduced.affine_cone_offset, reduced.affine_cone_matrix.rows
        ),
    }


def interval_ray_bounds(lower, upper):
    if np.isfinite(lower) and np.isfinite(upper):
        return 0.0, 0.0
    if np.isfinite(lower):
        return 0.0, 1.0
    if np.isfinite(upper):
        return -1.0, 0.0
    return -1.0, 1.0


def solve_reduced_ray(data):
    A = data["A"]
    n = A.shape[1]
    A_equalities = []
    A_inequalities = []
    for row in range(A.shape[0]):
        values = A.getrow(row).toarray().ravel()
        finite_lower = np.isfinite(data["lower"][row])
        finite_upper = np.isfinite(data["upper"][row])
        if finite_lower and finite_upper:
            A_equalities.append(values)
        elif finite_lower:
            A_inequalities.append(-values)
        elif finite_upper:
            A_inequalities.append(values)
    direction_bounds = [(-1.0, 1.0)] * n
    for position, column in enumerate(data["box_indices"]):
        direction_bounds[int(column)] = interval_ray_bounds(
            data["box_lower"][position], data["box_upper"][position]
        )
    result = opt.linprog(
        data["c"],
        A_ub=np.asarray(A_inequalities) if A_inequalities else None,
        b_ub=np.zeros(len(A_inequalities)) if A_inequalities else None,
        A_eq=np.asarray(A_equalities) if A_equalities else None,
        b_eq=np.zeros(len(A_equalities)) if A_equalities else None,
        bounds=direction_bounds,
        method="highs",
    )
    if not result.success or result.fun >= -1e-7:
        raise AssertionError(
            f"failed to find reduced ray: {result.message}; objective={result.fun}"
        )
    return np.ascontiguousarray(result.x, dtype=np.float64)


def normal_component_bounds(has_side, positive):
    if not has_side:
        return 0.0, 0.0
    return (0.0, None) if positive else (None, 0.0)


def solve_reduced_certificate(data):
    A = data["A"]
    G = data["G"]
    m, n = A.shape
    p = G.shape[0]
    if p == 0:
        raise AssertionError("certificate stress model lost its affine cone")

    bounds = []
    support = np.zeros(2 * m + 2 * n + p)
    for row in range(m):
        finite_lower = np.isfinite(data["lower"][row])
        finite_upper = np.isfinite(data["upper"][row])
        bounds.append(normal_component_bounds(finite_lower, False))
        support[row] = data["lower"][row] if finite_lower else 0.0
    for row in range(m):
        finite_upper = np.isfinite(data["upper"][row])
        bounds.append(normal_component_bounds(finite_upper, True))
        support[m + row] = data["upper"][row] if finite_upper else 0.0

    box_positions = {
        int(column): position
        for position, column in enumerate(data["box_indices"])
    }
    for column in range(n):
        position = box_positions[column]
        finite_lower = np.isfinite(data["box_lower"][position])
        bounds.append(normal_component_bounds(finite_lower, False))
        support[2 * m + column] = (
            data["box_lower"][position] if finite_lower else 0.0
        )
    for column in range(n):
        position = box_positions[column]
        finite_upper = np.isfinite(data["box_upper"][position])
        bounds.append(normal_component_bounds(finite_upper, True))
        support[2 * m + n + column] = (
            data["box_upper"][position] if finite_upper else 0.0
        )
    for affine_row in range(p):
        bounds.append((None, 0.0))
        support[2 * m + 2 * n + affine_row] = -data["h"][affine_row]

    stationarity = sp.hstack(
        (A.T, A.T, sp.eye(n), sp.eye(n), G.T), format="csr"
    )
    result = opt.linprog(
        np.zeros(stationarity.shape[1]),
        A_ub=support.reshape(1, -1),
        b_ub=np.array([-1.0]),
        A_eq=stationarity,
        b_eq=np.zeros(n),
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        raise AssertionError(f"failed to find reduced certificate: {result.message}")
    y = result.x[:m] + result.x[m : 2 * m]
    z = (
        result.x[2 * m : 2 * m + n]
        + result.x[2 * m + n : 2 * m + 2 * n]
    )
    affine_z = result.x[2 * m + 2 * n :]
    return (
        np.ascontiguousarray(y, dtype=np.float64),
        np.ascontiguousarray(z, dtype=np.float64),
        np.ascontiguousarray(affine_z, dtype=np.float64),
    )


def stress_settings(library, gpu):
    settings = library.prefos_strict_settings()
    settings.remove_empty_columns = 0
    settings.singleton_column_reduction = 0
    settings.dual_fixing = 0
    settings.linear_propagation = 0
    settings.cone_propagation = 0
    settings.max_aggregation_terms = 2
    settings.max_aggregation_rounds = 8
    if gpu:
        settings.structural_reductions_gpu = 1
        settings.linear_propagation_gpu = 1
        settings.event_queue_max_average_column_degree = 0.0
    return settings


def run_ray_seed(library, seed, gpu, bounded):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(4, 13))
    A = np.zeros((n - 1, n))
    if bounded:
        scales = np.ones(n - 1)
    else:
        scales = rng.uniform(0.5, 1.75, n - 1)
    for row, scale in enumerate(scales):
        A[row, row] = 1.0
        A[row, row + 1] = -scale
    direction = np.ones(n)
    for row in range(n - 2, -1, -1):
        direction[row] = scales[row] * direction[row + 1]
    c = -direction / max(1.0, float(direction @ direction))
    lower = np.zeros(n - 1)
    upper = np.zeros(n - 1)
    if bounded:
        box_lower = np.zeros(n)
        box_upper = np.full(n, np.inf)
    else:
        box_lower = np.full(n, -np.inf)
        box_upper = np.full(n, np.inf)
    problem, keepalive = make_problem(
        A, lower, upper, c, box_lower, box_upper
    )
    settings = stress_settings(library, gpu)
    settings.free_column_substitution = int(not bounded)
    settings.bounded_doubleton_substitution = int(bounded)
    settings.parallel_column_reduction = 1
    presolver = ct.c_void_p()
    status = library.prefos_create_presolver(
        ct.byref(problem), ct.byref(settings), ct.byref(presolver)
    )
    if status != 0:
        raise AssertionError(f"ray create failed at seed {seed}: {status}")
    try:
        status = library.prefos_run_presolve(presolver)
        if status not in (0, 1):
            raise AssertionError(f"ray presolve failed at seed {seed}: {status}")
        reduced = library.prefos_get_reduced_problem(presolver).contents
        ray = solve_reduced_ray(reduced_arrays(reduced))
        verification = RayVerification()
        status = library.prefos_verify_postsolve_unbounded_ray(
            presolver, ptr(ray, ct.c_double), 1e-8, ct.byref(verification)
        )
        if status != 0 or not verification.passed:
            raise AssertionError(
                f"ray postsolve failed at seed {seed}: status={status}, "
                f"reduced=({verification.reduced.row_recession_violation:.2e},"
                f"{verification.reduced.domain_recession_violation:.2e}), "
                f"original=({verification.original.row_recession_violation:.2e},"
                f"{verification.original.domain_recession_violation:.2e}), "
                f"objective_error="
                f"{verification.objective_direction_absolute_error:.2e}"
            )
    finally:
        library.prefos_free_presolver(presolver)


def run_certificate_seed(library, seed, gpu):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(4, 13))
    scales = rng.uniform(0.5, 1.75, n - 1)
    A = np.zeros((n, n))
    for row, scale in enumerate(scales):
        A[row, row] = 1.0
        A[row, row + 1] = -scale
    A[-1, 0] = 1.0
    lower = np.concatenate((np.zeros(n - 1), [-np.inf]))
    upper = np.zeros(n)
    G = np.zeros((1, n))
    G[0, -1] = 1.0
    h = np.array([-1.0])
    problem, keepalive = make_problem(
        A,
        lower,
        upper,
        np.zeros(n),
        np.full(n, -np.inf),
        np.full(n, np.inf),
        G,
        h,
    )
    settings = stress_settings(library, gpu)
    settings.free_column_substitution = 1
    settings.bounded_doubleton_substitution = 0
    settings.parallel_column_reduction = 1
    presolver = ct.c_void_p()
    status = library.prefos_create_presolver(
        ct.byref(problem), ct.byref(settings), ct.byref(presolver)
    )
    if status != 0:
        raise AssertionError(f"certificate create failed at seed {seed}: {status}")
    try:
        status = library.prefos_run_presolve(presolver)
        if status not in (0, 1):
            raise AssertionError(
                f"certificate presolve failed at seed {seed}: {status}"
            )
        reduced = library.prefos_get_reduced_problem(presolver).contents
        y, z, affine_z = solve_reduced_certificate(reduced_arrays(reduced))
        verification = CertificateVerification()
        status = library.prefos_verify_postsolve_infeasibility_certificate(
            presolver,
            ptr(y, ct.c_double),
            ptr(z, ct.c_double),
            ptr(affine_z, ct.c_double),
            1e-8,
            ct.byref(verification),
        )
        if status != 0 or not verification.passed:
            raise AssertionError(
                f"certificate postsolve failed at seed {seed}: status={status}, "
                f"reduced_stationarity="
                f"{verification.reduced.stationarity_violation:.2e}, "
                f"original_stationarity="
                f"{verification.original.stationarity_violation:.2e}, "
                f"original_domain="
                f"{verification.original.domain_dual_violation:.2e}, "
                f"value_error="
                f"{verification.certificate_value_absolute_error:.2e}"
            )
    finally:
        library.prefos_free_presolver(presolver)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", default="build/libPreFOS.so")
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()
    library_path = Path(args.library).resolve()
    if not library_path.exists():
        raise SystemExit(f"library not found: {library_path}")
    library = configure_library(library_path)
    configure_certificate_api(library)
    for seed in range(args.start_seed, args.start_seed + args.seeds):
        run_ray_seed(library, seed, args.gpu, bounded=False)
        run_ray_seed(library, seed, args.gpu, bounded=True)
        run_certificate_seed(library, seed, args.gpu)
    print(
        f"certificate/ray differential passed for {args.seeds} seeds "
        f"(gpu={args.gpu})"
    )


if __name__ == "__main__":
    main()
