#!/usr/bin/env python3
# Copyright 2026 Hongpei Li
# SPDX-License-Identifier: Apache-2.0

"""Randomized original-vs-presolved differential checks for Conic QPs."""

import argparse
import ctypes as ct
from dataclasses import dataclass, field
from pathlib import Path

import clarabel
import numpy as np
import scipy.sparse as sp

from differential_prefos import (
    AffineConeBlock,
    ConeBlock,
    PostsolveKktVerification,
    ProblemData,
    Verification,
    as_csr,
    configure_library,
    numpy_csr,
    numpy_vector,
    ptr,
    symmetric_q,
)


NONNEGATIVE = 0
SOC = 1
RSOC = 2
PSD = 3
EXPONENTIAL = 4
POWER = 5


@dataclass
class ConeSpec:
    cone_type: int
    indices: np.ndarray
    matrix_order: int = 0
    power_alpha: float = 0.0


@dataclass
class AffineConeSpec:
    cone_type: int
    dimension: int
    matrix_order: int = 0
    power_alpha: float = 0.0


@dataclass
class Model:
    n: int
    A: sp.csr_matrix
    lower: np.ndarray
    upper: np.ndarray
    Q: sp.csr_matrix
    R: sp.csr_matrix
    D: np.ndarray
    c: np.ndarray
    offset: float
    box_indices: np.ndarray
    box_lower: np.ndarray
    box_upper: np.ndarray
    cones: list[ConeSpec]
    affine_G: sp.csr_matrix | None = None
    affine_h: np.ndarray | None = None
    affine_cones: list[AffineConeSpec] = field(default_factory=list)


def configure_conic_library(path):
    library = configure_library(path)
    dual_postsolve_arguments = [
        ct.c_void_p,
        ct.POINTER(ct.c_double),
        ct.POINTER(ct.c_double),
        ct.POINTER(ct.c_double),
        ct.c_double,
        ct.POINTER(ct.c_double),
        ct.POINTER(ct.c_double),
        ct.POINTER(ct.c_double),
    ]
    library.prefos_postsolve_primal_dual.argtypes = dual_postsolve_arguments
    library.prefos_postsolve_primal_dual.restype = ct.c_int
    library.prefos_postsolve_extended_dual.argtypes = dual_postsolve_arguments
    library.prefos_postsolve_extended_dual.restype = ct.c_int
    full_dual_postsolve_arguments = [
        ct.c_void_p,
        ct.POINTER(ct.c_double),
        ct.POINTER(ct.c_double),
        ct.POINTER(ct.c_double),
        ct.POINTER(ct.c_double),
        ct.c_double,
        ct.POINTER(ct.c_double),
        ct.POINTER(ct.c_double),
        ct.POINTER(ct.c_double),
        ct.POINTER(ct.c_double),
    ]
    library.prefos_postsolve_full_primal_dual.argtypes = (
        full_dual_postsolve_arguments
    )
    library.prefos_postsolve_full_primal_dual.restype = ct.c_int
    library.prefos_postsolve_full_extended_dual.argtypes = (
        full_dual_postsolve_arguments
    )
    library.prefos_postsolve_full_extended_dual.restype = ct.c_int
    library.prefos_verify_postsolve_extended_kkt.argtypes = (
        library.prefos_verify_postsolve_kkt.argtypes
    )
    library.prefos_verify_postsolve_extended_kkt.restype = ct.c_int
    full_kkt_arguments = [
        ct.c_void_p,
        ct.POINTER(ct.c_double),
        ct.POINTER(ct.c_double),
        ct.POINTER(ct.c_double),
        ct.POINTER(ct.c_double),
        ct.c_double,
        ct.POINTER(PostsolveKktVerification),
    ]
    library.prefos_verify_postsolve_full_kkt.argtypes = full_kkt_arguments
    library.prefos_verify_postsolve_full_kkt.restype = ct.c_int
    library.prefos_verify_postsolve_full_extended_kkt.argtypes = full_kkt_arguments
    library.prefos_verify_postsolve_full_extended_kkt.restype = ct.c_int
    return library


def make_c_problem(model):
    A_view, A_keepalive = as_csr(model.A)
    Q_view, Q_keepalive = as_csr(sp.triu(model.Q, format="csr"))
    R_view, R_keepalive = as_csr(model.R)
    lower = np.ascontiguousarray(model.lower, dtype=np.float64)
    upper = np.ascontiguousarray(model.upper, dtype=np.float64)
    D = np.ascontiguousarray(model.D, dtype=np.float64)
    c = np.ascontiguousarray(model.c, dtype=np.float64)
    box_indices = np.ascontiguousarray(model.box_indices, dtype=np.int32)
    box_lower = np.ascontiguousarray(model.box_lower, dtype=np.float64)
    box_upper = np.ascontiguousarray(model.box_upper, dtype=np.float64)

    cone_indices = [
        np.ascontiguousarray(cone.indices, dtype=np.int32) for cone in model.cones
    ]
    cone_array = (ConeBlock * len(model.cones))()
    for i, cone in enumerate(model.cones):
        cone_array[i] = ConeBlock(
            cone.cone_type,
            len(cone.indices),
            cone.matrix_order,
            ptr(cone_indices[i], ct.c_int),
            cone.power_alpha,
        )
    cone_pointer = (
        ct.cast(cone_array, ct.POINTER(ConeBlock))
        if model.cones
        else ct.POINTER(ConeBlock)()
    )
    affine_G = (
        sp.csr_matrix((0, model.n), dtype=np.float64)
        if model.affine_G is None
        else sp.csr_matrix(model.affine_G)
    )
    affine_G_view, affine_G_keepalive = as_csr(affine_G)
    affine_h = np.ascontiguousarray(
        np.empty(0) if model.affine_h is None else model.affine_h,
        dtype=np.float64,
    )
    affine_cone_array = (AffineConeBlock * len(model.affine_cones))()
    for i, cone in enumerate(model.affine_cones):
        affine_cone_array[i] = AffineConeBlock(
            cone.cone_type,
            cone.dimension,
            cone.matrix_order,
            cone.power_alpha,
        )
    affine_cone_pointer = (
        ct.cast(affine_cone_array, ct.POINTER(AffineConeBlock))
        if model.affine_cones
        else ct.POINTER(AffineConeBlock)()
    )

    problem = ProblemData(
        model.n,
        A_view,
        ptr(lower, ct.c_double),
        ptr(upper, ct.c_double),
        Q_view,
        0,
        R_view,
        ptr(D, ct.c_double),
        ptr(c, ct.c_double),
        model.offset,
        len(box_indices),
        ptr(box_indices, ct.c_int),
        ptr(box_lower, ct.c_double),
        ptr(box_upper, ct.c_double),
        len(model.cones),
        cone_pointer,
        affine_G_view,
        ptr(affine_h, ct.c_double),
        len(model.affine_cones),
        affine_cone_pointer,
    )
    keepalive = (
        A_keepalive,
        Q_keepalive,
        R_keepalive,
        lower,
        upper,
        D,
        c,
        box_indices,
        box_lower,
        box_upper,
        cone_indices,
        cone_array,
        affine_G_keepalive,
        affine_h,
        affine_cone_array,
    )
    return problem, keepalive


def read_reduced_model(reduced):
    cones = []
    for i in range(reduced.n_cones):
        block = reduced.cones[i]
        indices = numpy_vector(block.indices, block.dimension, np.int32)
        cones.append(
            ConeSpec(
                int(block.type), indices, block.matrix_order, block.power_alpha
            )
        )
    affine_cones = []
    for i in range(reduced.n_affine_cones):
        block = reduced.affine_cones[i]
        affine_cones.append(
            AffineConeSpec(
                int(block.type),
                int(block.dimension),
                int(block.matrix_order),
                float(block.power_alpha),
            )
        )
    return Model(
        reduced.n,
        numpy_csr(reduced.A),
        numpy_vector(reduced.constraint_lower, reduced.A.rows),
        numpy_vector(reduced.constraint_upper, reduced.A.rows),
        symmetric_q(reduced.Q, reduced.q_storage).tocsr(),
        numpy_csr(reduced.R),
        numpy_vector(reduced.D, reduced.R.rows),
        numpy_vector(reduced.c, reduced.n),
        reduced.objective_offset,
        numpy_vector(reduced.box_indices, reduced.n_box, np.int32),
        numpy_vector(reduced.box_lower, reduced.n_box),
        numpy_vector(reduced.box_upper, reduced.n_box),
        cones,
        numpy_csr(reduced.affine_cone_matrix),
        numpy_vector(
            reduced.affine_cone_offset, reduced.affine_cone_matrix.rows
        ),
        affine_cones,
    )


def selector(indices, n):
    indices = np.asarray(indices, dtype=np.int32)
    return sp.csc_matrix(
        (np.ones(len(indices)), (np.arange(len(indices)), indices)),
        shape=(len(indices), n),
    )


def rsoc_to_soc(dimension):
    transform = np.zeros((dimension, dimension))
    transform[0, 0] = 1.0
    transform[0, 1] = 1.0
    transform[1, 0] = 1.0
    transform[1, 1] = -1.0
    transform[2:, 2:] = np.sqrt(2.0) * np.eye(dimension - 2)
    return transform


def solve_conic_qp(model):
    blocks = []
    right_hand_sides = []
    solver_cones = []
    recoveries = []

    def append_block(matrix, rhs, solver_cone, recovery):
        matrix = sp.csc_matrix(matrix)
        blocks.append(matrix)
        right_hand_sides.append(np.asarray(rhs, dtype=np.float64))
        solver_cones.append(solver_cone)
        recoveries.append((matrix.shape[0], recovery))

    for row in range(model.A.shape[0]):
        row_matrix = model.A.getrow(row)
        lower = model.lower[row]
        upper = model.upper[row]
        if np.isfinite(lower) and lower == upper:
            append_block(
                row_matrix,
                [upper],
                clarabel.ZeroConeT(1),
                ("row", row, 1.0),
            )
        else:
            if np.isfinite(upper):
                append_block(
                    row_matrix,
                    [upper],
                    clarabel.NonnegativeConeT(1),
                    ("row", row, 1.0),
                )
            if np.isfinite(lower):
                append_block(
                    -row_matrix,
                    [-lower],
                    clarabel.NonnegativeConeT(1),
                    ("row", row, -1.0),
                )

    for position, column in enumerate(model.box_indices):
        unit = selector([column], model.n)
        lower = model.box_lower[position]
        upper = model.box_upper[position]
        if np.isfinite(lower) and lower == upper:
            append_block(
                unit,
                [upper],
                clarabel.ZeroConeT(1),
                ("box", int(column), 1.0),
            )
        else:
            if np.isfinite(upper):
                append_block(
                    unit,
                    [upper],
                    clarabel.NonnegativeConeT(1),
                    ("box", int(column), 1.0),
                )
            if np.isfinite(lower):
                append_block(
                    -unit,
                    [-lower],
                    clarabel.NonnegativeConeT(1),
                    ("box", int(column), -1.0),
                )

    for cone in model.cones:
        cone_selector = selector(cone.indices, model.n)
        dimension = len(cone.indices)
        transform = np.eye(dimension)
        if cone.cone_type == NONNEGATIVE:
            solver_cone = clarabel.NonnegativeConeT(dimension)
        elif cone.cone_type == SOC:
            solver_cone = clarabel.SecondOrderConeT(dimension)
        elif cone.cone_type == RSOC:
            transform = rsoc_to_soc(dimension)
            solver_cone = clarabel.SecondOrderConeT(dimension)
        elif cone.cone_type == PSD:
            solver_cone = clarabel.PSDTriangleConeT(cone.matrix_order)
        elif cone.cone_type == EXPONENTIAL:
            solver_cone = clarabel.ExponentialConeT()
        elif cone.cone_type == POWER:
            solver_cone = clarabel.PowerConeT(cone.power_alpha)
        else:
            raise AssertionError(f"unsupported cone type {cone.cone_type}")
        append_block(
            -sp.csc_matrix(transform) @ cone_selector,
            np.zeros(dimension),
            solver_cone,
            ("cone", cone.indices, transform),
        )

    affine_row = 0
    affine_G = (
        sp.csr_matrix((0, model.n), dtype=np.float64)
        if model.affine_G is None
        else sp.csr_matrix(model.affine_G)
    )
    affine_h = (
        np.empty(0, dtype=np.float64)
        if model.affine_h is None
        else np.asarray(model.affine_h, dtype=np.float64)
    )
    for cone in model.affine_cones:
        dimension = cone.dimension
        transform = np.eye(dimension)
        if cone.cone_type == NONNEGATIVE:
            solver_cone = clarabel.NonnegativeConeT(dimension)
        elif cone.cone_type == SOC:
            solver_cone = clarabel.SecondOrderConeT(dimension)
        elif cone.cone_type == RSOC:
            transform = rsoc_to_soc(dimension)
            solver_cone = clarabel.SecondOrderConeT(dimension)
        elif cone.cone_type == PSD:
            solver_cone = clarabel.PSDTriangleConeT(cone.matrix_order)
        elif cone.cone_type == EXPONENTIAL:
            solver_cone = clarabel.ExponentialConeT()
        elif cone.cone_type == POWER:
            solver_cone = clarabel.PowerConeT(cone.power_alpha)
        else:
            raise AssertionError(f"unsupported affine cone type {cone.cone_type}")
        row_slice = slice(affine_row, affine_row + dimension)
        block_G = affine_G[row_slice]
        append_block(
            -sp.csc_matrix(transform) @ block_G,
            transform @ affine_h[row_slice],
            solver_cone,
            ("affine_cone", transform, affine_row),
        )
        affine_row += dimension
    if affine_row != affine_G.shape[0] or affine_row != len(affine_h):
        raise AssertionError("affine cone blocks do not cover G and h")

    operator = (
        sp.vstack(blocks, format="csc") if blocks else sp.csc_matrix((0, model.n))
    )
    right_hand_side = (
        np.concatenate(right_hand_sides)
        if right_hand_sides
        else np.empty(0, dtype=np.float64)
    )
    hessian = model.Q + model.R.T @ sp.diags(model.D) @ model.R
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    settings.max_iter = 500
    settings.tol_gap_abs = 1e-9
    settings.tol_gap_rel = 1e-9
    settings.tol_feas = 1e-9
    settings.presolve_enable = False
    settings.chordal_decomposition_enable = False
    solution = clarabel.DefaultSolver(
        sp.triu(hessian, format="csc"),
        model.c,
        operator,
        right_hand_side,
        solver_cones,
        settings,
    ).solve()
    if str(solution.status) not in ("Solved", "AlmostSolved"):
        raise AssertionError(f"Clarabel failed: {solution.status}")

    row_dual = np.zeros(model.A.shape[0])
    domain_dual = np.zeros(model.n)
    affine_dual = np.zeros(affine_G.shape[0])
    solver_dual = np.asarray(solution.z, dtype=np.float64)
    cursor = 0
    for dimension, recovery in recoveries:
        dual = solver_dual[cursor : cursor + dimension]
        cursor += dimension
        if recovery[0] == "row":
            row_dual[recovery[1]] += recovery[2] * dual[0]
        elif recovery[0] == "box":
            domain_dual[recovery[1]] += recovery[2] * dual[0]
        elif recovery[0] == "affine_cone":
            transform, row_start = recovery[1], recovery[2]
            affine_dual[row_start : row_start + dimension] = -transform.T @ dual
        else:
            indices, transform = recovery[1], recovery[2]
            domain_dual[indices] -= transform.T @ dual
    if cursor != len(solver_dual):
        raise AssertionError("Clarabel dual recovery consumed the wrong dimension")
    return (
        np.asarray(solution.x, dtype=np.float64),
        float(solution.obj_val + model.offset),
        row_dual,
        domain_dual,
        affine_dual,
    )


def matrix_svec(matrix):
    values = []
    for row in range(matrix.shape[0]):
        for column in range(row + 1):
            value = matrix[row, column]
            if row != column:
                value *= np.sqrt(2.0)
            values.append(value)
    return np.asarray(values)


def interior_cone_point(rng, cone_type, matrix_order=0, power_alpha=0.0):
    if cone_type == NONNEGATIVE:
        return rng.uniform(0.4, 1.4, 2)
    if cone_type == SOC:
        tail = rng.normal(scale=0.35, size=2)
        return np.concatenate(([np.linalg.norm(tail) + rng.uniform(0.4, 1.0)], tail))
    if cone_type == RSOC:
        u, v = rng.uniform(0.7, 1.4, 2)
        direction = rng.normal(size=2)
        direction /= np.linalg.norm(direction)
        radius = rng.uniform(0.1, 0.55) * np.sqrt(2.0 * u * v)
        return np.concatenate(([u, v], radius * direction))
    if cone_type == EXPONENTIAL:
        y = rng.uniform(0.6, 1.4)
        x = rng.uniform(-0.5, 0.5)
        z = y * np.exp(x / y) + rng.uniform(0.4, 1.0)
        return np.array([x, y, z])
    if cone_type == POWER:
        x, y = rng.uniform(0.6, 1.4, 2)
        radial = x**power_alpha * y ** (1.0 - power_alpha)
        z = rng.uniform(-0.6, 0.6) * radial
        return np.array([x, y, z])
    factor = rng.normal(scale=0.35, size=(matrix_order, matrix_order))
    return matrix_svec(factor @ factor.T + 0.5 * np.eye(matrix_order))


def psd_face_point(rng, matrix_order, removed_indices):
    retained = [i for i in range(matrix_order) if i not in removed_indices]
    factor = rng.normal(scale=0.35, size=(len(retained), len(retained)))
    principal = factor @ factor.T + 0.5 * np.eye(len(retained))
    matrix = np.zeros((matrix_order, matrix_order))
    matrix[np.ix_(retained, retained)] = principal
    return matrix_svec(matrix)


def random_problem(rng, seed):
    cone_layout = [
        (NONNEGATIVE, 2, 0, 0.0),
        (SOC, 3, 0, 0.0),
        (RSOC, 4, 0, 0.0),
        (PSD, 6, 3, 0.0),
        (EXPONENTIAL, 3, 0, 0.0),
        (POWER, 3, 0, 0.35),
    ]
    reduction_case = seed % 14
    collapse_type = (
        None if reduction_case in (0, 5, 6, 7, 8, 9, 10, 11, 12, 13)
        else reduction_case - 1
    )
    rsoc_face = "u" if reduction_case == 5 else "v" if reduction_case == 6 else None
    psd_face = (1,) if reduction_case == 7 else (0, 2) if reduction_case == 8 else ()
    exp_face = "y" if reduction_case == 9 else "z" if reduction_case == 10 else None
    power_face = (
        "x"
        if reduction_case == 11
        else "y"
        if reduction_case == 12
        else "z"
        if reduction_case == 13
        else None
    )
    n = 2 + sum(dimension for _, dimension, _, _ in cone_layout)
    permutation = rng.permutation(n).astype(np.int32)
    box_indices = permutation[:2]
    planted = np.zeros(n)
    planted[box_indices] = rng.uniform(-0.6, 0.6, 2)
    cursor = 2
    cones = []
    for cone_type, dimension, order, power_alpha in cone_layout:
        indices = permutation[cursor : cursor + dimension]
        cursor += dimension
        cones.append(ConeSpec(cone_type, indices.copy(), order, power_alpha))
        if cone_type == RSOC and rsoc_face == "u":
            planted[indices[1]] = rng.uniform(0.7, 1.4)
        elif cone_type == RSOC and rsoc_face == "v":
            planted[indices[0]] = rng.uniform(0.7, 1.4)
        elif cone_type == PSD and psd_face:
            planted[indices] = psd_face_point(rng, order, psd_face)
        elif cone_type == EXPONENTIAL and exp_face:
            planted[indices[0]] = -rng.uniform(0.4, 1.0)
            planted[indices[2]] = 0.0 if exp_face == "z" else rng.uniform(0.4, 1.0)
        elif cone_type == POWER and power_face:
            planted[indices[0]] = (
                0.0 if power_face == "x" else rng.uniform(0.6, 1.4)
            )
            planted[indices[1]] = (
                0.0 if power_face == "y" else rng.uniform(0.6, 1.4)
            )
        elif cone_type != collapse_type:
            planted[indices] = interior_cone_point(
                rng, cone_type, order, power_alpha
            )

    box_lower = np.array([planted[box_indices[0]], -2.0])
    box_upper = np.array([planted[box_indices[0]], 2.0])
    row_count = int(rng.integers(5, 9))
    A = rng.normal(scale=0.45, size=(row_count, n))
    activity = A @ planted
    lower = np.empty(row_count)
    upper = np.empty(row_count)
    for row in range(row_count):
        slack = rng.uniform(0.4, 1.4)
        mode = row % 4
        lower[row] = activity[row] - slack if mode in (0, 1) else -np.inf
        upper[row] = activity[row] + slack if mode in (0, 2) else np.inf
        if mode == 3:
            lower[row] = activity[row]
            upper[row] = activity[row]

    if collapse_type is not None:
        cone = next(block for block in cones if block.cone_type == collapse_type)
        if collapse_type == NONNEGATIVE:
            pivots = cone.indices
        elif collapse_type == SOC:
            pivots = cone.indices[:1]
        elif collapse_type == RSOC:
            pivots = cone.indices[:2]
        else:
            diagonal_positions = [
                i * (i + 1) // 2 + i for i in range(cone.matrix_order)
            ]
            pivots = cone.indices[diagonal_positions]
        collapse_rows = np.zeros((len(pivots), n))
        collapse_rows[np.arange(len(pivots)), pivots] = 1.0
        A = np.vstack((A, collapse_rows))
        lower = np.concatenate((lower, np.full(len(pivots), -np.inf)))
        upper = np.concatenate((upper, np.zeros(len(pivots))))
    elif rsoc_face is not None:
        cone = next(block for block in cones if block.cone_type == RSOC)
        pivot = cone.indices[0 if rsoc_face == "u" else 1]
        face_row = np.zeros((1, n))
        face_row[0, pivot] = 1.0
        A = np.vstack((A, face_row))
        lower = np.concatenate((lower, [-np.inf]))
        upper = np.concatenate((upper, [0.0]))
    elif psd_face:
        cone = next(block for block in cones if block.cone_type == PSD)
        diagonal_positions = [i * (i + 1) // 2 + i for i in psd_face]
        pivots = cone.indices[diagonal_positions]
        face_rows = np.zeros((len(pivots), n))
        face_rows[np.arange(len(pivots)), pivots] = 1.0
        A = np.vstack((A, face_rows))
        lower = np.concatenate((lower, np.full(len(pivots), -np.inf)))
        upper = np.concatenate((upper, np.zeros(len(pivots))))
    elif exp_face:
        cone = next(block for block in cones if block.cone_type == EXPONENTIAL)
        pivot = cone.indices[1 if exp_face == "y" else 2]
        face_row = np.zeros((1, n))
        face_row[0, pivot] = 1.0
        A = np.vstack((A, face_row))
        lower = np.concatenate((lower, [-np.inf]))
        upper = np.concatenate((upper, [0.0]))
    elif power_face:
        cone = next(block for block in cones if block.cone_type == POWER)
        pivot = cone.indices[{"x": 0, "y": 1, "z": 2}[power_face]]
        face_row = np.zeros((1, n))
        face_row[0, pivot] = 1.0
        A = np.vstack((A, face_row))
        if power_face == "z":
            lower = np.concatenate((lower, [0.0]))
        else:
            lower = np.concatenate((lower, [-np.inf]))
        upper = np.concatenate((upper, [0.0]))

    factor = rng.normal(scale=0.16, size=(n, n))
    Q = factor.T @ factor + 0.35 * np.eye(n)
    rank = 3
    R = rng.normal(scale=0.2, size=(rank, n))
    D = rng.uniform(0.3, 1.4, rank)
    c = rng.normal(scale=0.45, size=n)
    offset = float(rng.normal(scale=0.2))
    names = {
        None: "none",
        NONNEGATIVE: "nonnegative",
        SOC: "soc",
        RSOC: "rsoc",
        PSD: "psd",
    }
    if rsoc_face:
        reduction_name = f"rsoc_{rsoc_face}_face"
    elif psd_face:
        reduction_name = f"psd_{len(psd_face)}_face"
    elif exp_face:
        reduction_name = f"exp_{exp_face}_face"
    elif power_face:
        reduction_name = f"power_{power_face}_face"
    else:
        reduction_name = names[collapse_type]
    return (
        Model(
            n,
            sp.csr_matrix(A),
            np.ascontiguousarray(lower),
            np.ascontiguousarray(upper),
            sp.csr_matrix(Q),
            sp.csr_matrix(R),
            np.ascontiguousarray(D),
            np.ascontiguousarray(c),
            offset,
            np.ascontiguousarray(box_indices),
            np.ascontiguousarray(box_lower),
            np.ascontiguousarray(box_upper),
            cones,
        ),
        reduction_name,
    )


def run_seed(
    library,
    seed,
    use_gpu=False,
    propagated_bound_policy=0,
    affine_cone_aggregation=False,
):
    rng = np.random.default_rng(seed)
    model, collapse_name = random_problem(rng, seed)
    uses_extended_dual = collapse_name.endswith("_face")
    nonlinear_boundary_face = collapse_name.startswith(("exp_", "power_"))
    if nonlinear_boundary_face:
        direct_x = None
        direct_objective = None
    else:
        try:
            direct_x, direct_objective, _, _, _ = solve_conic_qp(model)
        except AssertionError as error:
            if "Clarabel failed:" not in str(error):
                raise
            direct_x = None
            direct_objective = None
    has_direct_solution = direct_x is not None
    problem, keepalive = make_c_problem(model)
    settings = library.prefos_strict_settings()
    settings.propagated_bound_policy = propagated_bound_policy
    settings.affine_cone_coordinate_aggregation = int(affine_cone_aggregation)
    if use_gpu:
        settings.linear_propagation_gpu = 1
        settings.event_queue_max_average_column_degree = 0.0
    presolver = ct.c_void_p()
    status = library.prefos_create_presolver(
        ct.byref(problem), ct.byref(settings), ct.byref(presolver)
    )
    del keepalive
    if status != 0:
        raise AssertionError(f"create failed: status {status}")
    try:
        status = library.prefos_run_presolve(presolver)
        if status not in (0, 1):
            raise AssertionError(f"presolve failed: status {status}")
        reduced_view = library.prefos_get_reduced_problem(presolver).contents
        reduced = read_reduced_model(reduced_view)
        reduced_x, reduced_objective, reduced_y, reduced_z, reduced_affine_z = (
            solve_conic_qp(reduced)
        )
        reduced_x = np.ascontiguousarray(reduced_x)
        reduced_y = np.ascontiguousarray(reduced_y)
        reduced_z = np.ascontiguousarray(reduced_z)
        reduced_affine_z = np.ascontiguousarray(reduced_affine_z)

        verification = Verification()
        status = library.prefos_verify_postsolve_primal(
            presolver,
            ptr(reduced_x, ct.c_double),
            2e-5,
            ct.byref(verification),
        )
        if status != 0 or not verification.passed:
            raise AssertionError(
                f"primal round-trip failed: status={status}, "
                f"row={verification.original_row_violation:.3e}, "
                f"box={verification.original_box_violation:.3e}, "
                f"cone={verification.original_cone_violation:.3e}, "
                f"objerr={verification.objective_absolute_error:.3e}"
            )

        if has_direct_solution:
            objective_scale = max(
                1.0, abs(direct_objective), abs(reduced_objective)
            )
            if abs(direct_objective - reduced_objective) > 3e-5 * objective_scale:
                raise AssertionError(
                    f"optimal values differ: {direct_objective} vs {reduced_objective}"
                )

        postsolved_x = np.empty(model.n)
        postsolved_y = np.empty(model.A.shape[0])
        postsolved_z = np.empty(model.n)
        postsolved_affine_z = np.empty(
            0 if model.affine_G is None else model.affine_G.shape[0]
        )
        uses_full_dual = reduced.affine_G is not None and reduced.affine_G.shape[0] > 0
        if uses_full_dual:
            postsolve_function = (
                library.prefos_postsolve_full_extended_dual
                if uses_extended_dual
                else library.prefos_postsolve_full_primal_dual
            )
            status = postsolve_function(
                presolver,
                ptr(reduced_x, ct.c_double),
                ptr(reduced_y, ct.c_double),
                ptr(reduced_z, ct.c_double),
                ptr(reduced_affine_z, ct.c_double),
                2e-5,
                ptr(postsolved_x, ct.c_double),
                ptr(postsolved_y, ct.c_double),
                ptr(postsolved_z, ct.c_double),
                ptr(postsolved_affine_z, ct.c_double),
            )
        else:
            postsolve_function = (
                library.prefos_postsolve_extended_dual
                if uses_extended_dual
                else library.prefos_postsolve_primal_dual
            )
            status = postsolve_function(
                presolver,
                ptr(reduced_x, ct.c_double),
                ptr(reduced_y, ct.c_double),
                ptr(reduced_z, ct.c_double),
                2e-5,
                ptr(postsolved_x, ct.c_double),
                ptr(postsolved_y, ct.c_double),
                ptr(postsolved_z, ct.c_double),
            )
        if status != 0:
            raise AssertionError(f"postsolve failed: status {status}")
        if has_direct_solution:
            primal_scale = max(1.0, np.linalg.norm(direct_x, ord=np.inf))
            primal_difference = np.linalg.norm(postsolved_x - direct_x, ord=np.inf)
            if primal_difference > 4e-4 * primal_scale:
                raise AssertionError(
                    f"unique primal solutions differ by {primal_difference:.3e}"
                )

        kkt = PostsolveKktVerification()
        if uses_full_dual:
            verify_function = (
                library.prefos_verify_postsolve_full_extended_kkt
                if uses_extended_dual
                else library.prefos_verify_postsolve_full_kkt
            )
            status = verify_function(
                presolver,
                ptr(reduced_x, ct.c_double),
                ptr(reduced_y, ct.c_double),
                ptr(reduced_z, ct.c_double),
                ptr(reduced_affine_z, ct.c_double),
                5e-5,
                ct.byref(kkt),
            )
        else:
            verify_function = (
                library.prefos_verify_postsolve_extended_kkt
                if uses_extended_dual
                else library.prefos_verify_postsolve_kkt
            )
            status = verify_function(
                presolver,
                ptr(reduced_x, ct.c_double),
                ptr(reduced_y, ct.c_double),
                ptr(reduced_z, ct.c_double),
                5e-5,
                ct.byref(kkt),
            )
        if status != 0 or not kkt.passed:
            raise AssertionError(
                f"KKT postsolve failed: status={status}, "
                f"reduced_stationarity={kkt.reduced.stationarity_violation:.3e}, "
                f"original_stationarity={kkt.original.stationarity_violation:.3e}, "
                f"original_dual={kkt.original.domain_dual_violation:.3e}, "
                f"complementarity={kkt.original.complementarity_violation:.3e}"
            )
    except Exception as error:
        raise AssertionError(
            f"seed {seed} (collapse={collapse_name}): {error}"
        ) from error
    finally:
        library.prefos_free_presolver(presolver)
    return collapse_name


def check_affine_aggregation_dual(library):
    model = Model(
        4,
        sp.csr_matrix(
            (
                np.array([1.0, -1.0, 1.0, -1.0]),
                np.array([0, 2, 1, 3]),
                np.array([0, 2, 4]),
            ),
            shape=(2, 4),
        ),
        np.zeros(2),
        np.zeros(2),
        sp.diags([0.0, 0.0, 1.0, 0.0], format="csr"),
        sp.csr_matrix((0, 4)),
        np.empty(0),
        np.array([0.0, 0.0, 0.0, -2.0]),
        0.0,
        np.array([2, 3], dtype=np.int32),
        np.array([-np.inf, -np.inf]),
        np.array([1.0, np.inf]),
        [ConeSpec(SOC, np.array([0, 1], dtype=np.int32))],
    )
    problem, keepalive = make_c_problem(model)
    settings = library.prefos_strict_settings()
    settings.affine_cone_coordinate_aggregation = 1
    presolver = ct.c_void_p()
    status = library.prefos_create_presolver(
        ct.byref(problem), ct.byref(settings), ct.byref(presolver)
    )
    del keepalive
    if status != 0:
        raise AssertionError(f"affine aggregation create failed: {status}")
    try:
        status = library.prefos_run_presolve(presolver)
        if status != 1:
            raise AssertionError(f"affine aggregation did not reduce: {status}")
        reduced = read_reduced_model(
            library.prefos_get_reduced_problem(presolver).contents
        )
        if reduced.n != 2 or len(reduced.affine_cones) != 1:
            raise AssertionError("unexpected affine aggregation dimensions")
        reduced_x, _, reduced_y, reduced_z, reduced_affine_z = solve_conic_qp(
            reduced
        )
        arrays = [
            np.ascontiguousarray(value)
            for value in (reduced_x, reduced_y, reduced_z, reduced_affine_z)
        ]
        kkt = PostsolveKktVerification()
        status = library.prefos_verify_postsolve_full_kkt(
            presolver,
            ptr(arrays[0], ct.c_double),
            ptr(arrays[1], ct.c_double),
            ptr(arrays[2], ct.c_double),
            ptr(arrays[3], ct.c_double),
            5e-5,
            ct.byref(kkt),
        )
        if status != 0 or not kkt.passed:
            raise AssertionError(
                f"affine full KKT failed: status={status}, "
                f"reduced={kkt.reduced.stationarity_violation:.3e}, "
                f"original={kkt.original.stationarity_violation:.3e}"
            )
    finally:
        library.prefos_free_presolver(presolver)


def check_affine_psd_block_decomposition(library):
    affine_G = sp.csr_matrix(
        (
            np.ones(4),
            np.array([0, 1, 2, 3]),
            np.array([0, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4]),
        ),
        shape=(10, 4),
    )
    affine_h = np.zeros(10)
    affine_h[[1, 8]] = 0.25 * np.sqrt(2.0)
    model = Model(
        4,
        sp.csr_matrix((0, 4)),
        np.empty(0),
        np.empty(0),
        sp.eye(4, format="csr"),
        sp.csr_matrix((0, 4)),
        np.empty(0),
        np.ones(4),
        0.0,
        np.arange(4, dtype=np.int32),
        np.full(4, -np.inf),
        np.full(4, np.inf),
        [],
        affine_G,
        affine_h,
        [AffineConeSpec(PSD, 10, 4)],
    )
    direct_x, direct_objective, _, _, _ = solve_conic_qp(model)
    problem, keepalive = make_c_problem(model)
    settings = library.prefos_strict_settings()
    settings.cone_propagation = 0
    presolver = ct.c_void_p()
    status = library.prefos_create_presolver(
        ct.byref(problem), ct.byref(settings), ct.byref(presolver)
    )
    del keepalive
    if status != 0:
        raise AssertionError(f"affine PSD decomposition create failed: {status}")
    try:
        status = library.prefos_run_presolve(presolver)
        if status != 1:
            raise AssertionError(f"affine PSD block was not decomposed: {status}")
        reduced = read_reduced_model(
            library.prefos_get_reduced_problem(presolver).contents
        )
        if len(reduced.affine_cones) != 2 or reduced.affine_G.shape[0] != 6:
            raise AssertionError("unexpected affine PSD decomposition shape")
        reduced_solution = solve_conic_qp(reduced)
        reduced_x, reduced_objective, reduced_y, reduced_z, reduced_affine_z = [
            np.ascontiguousarray(value) if isinstance(value, np.ndarray) else value
            for value in reduced_solution
        ]
        scale = max(1.0, abs(direct_objective), abs(reduced_objective))
        if abs(direct_objective - reduced_objective) > 2e-7 * scale:
            raise AssertionError("affine PSD decomposition changed the optimum")
        if np.linalg.norm(direct_x - reduced_x, ord=np.inf) > 2e-6:
            raise AssertionError("affine PSD decomposition changed the minimizer")
        kkt = PostsolveKktVerification()
        status = library.prefos_verify_postsolve_full_kkt(
            presolver,
            ptr(reduced_x, ct.c_double),
            ptr(reduced_y, ct.c_double),
            ptr(reduced_z, ct.c_double),
            ptr(reduced_affine_z, ct.c_double),
            2e-6,
            ct.byref(kkt),
        )
        if status != 0 or not kkt.passed:
            raise AssertionError(
                f"affine PSD decomposition KKT failed: status={status}, "
                f"reduced={kkt.reduced.stationarity_violation:.3e}, "
                f"original={kkt.original.stationarity_violation:.3e}"
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
    parser.add_argument("--affine-cone-aggregation", action="store_true")
    args = parser.parse_args()
    if args.seeds <= 0 or args.start_seed < 0:
        raise SystemExit("--seeds must be positive and --start-seed nonnegative")
    library_path = Path(args.library).resolve()
    if not library_path.exists():
        raise SystemExit(f"library not found: {library_path}")
    library = configure_conic_library(library_path)
    check_affine_psd_block_decomposition(library)
    if args.affine_cone_aggregation:
        check_affine_aggregation_dual(library)
    propagated_bound_policy = int(
        args.propagated_bound_policy == "interior-point"
    )
    coverage = {
        name: 0
        for name in (
            "none",
            "nonnegative",
            "soc",
            "rsoc",
            "psd",
            "rsoc_u_face",
            "rsoc_v_face",
            "psd_1_face",
            "psd_2_face",
            "exp_y_face",
            "exp_z_face",
            "power_x_face",
            "power_y_face",
            "power_z_face",
        )
    }
    for seed in range(args.start_seed, args.start_seed + args.seeds):
        coverage[
            run_seed(
                library,
                seed,
                args.gpu_linear_propagation,
                propagated_bound_policy,
                args.affine_cone_aggregation,
            )
        ] += 1
    summary = ", ".join(f"{name}={count}" for name, count in coverage.items())
    print(f"Differential Conic QP check passed for {args.seeds} seeds ({summary}).")


if __name__ == "__main__":
    main()
