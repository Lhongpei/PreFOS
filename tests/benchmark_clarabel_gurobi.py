#!/usr/bin/env python3
# Copyright 2026 Hongpei Li
# SPDX-License-Identifier: Apache-2.0

"""Compare PreFOS and Gurobi external presolve with Clarabel downstream."""

import argparse
import ctypes as ct
import gc
import json
import time
from pathlib import Path

import clarabel
import numpy as np
import scipy.sparse as sp

try:
    import gurobipy as gp
except ImportError:
    gp = None

from benchmark_cbf_prefos import (
    configure_prefos,
    configure_parser,
    make_prefos_view,
    stats_dict,
)
from cbf_scalar import load_scalar_cbf
from differential_prefos import Verification, numpy_csr, numpy_vector, ptr, symmetric_q
from differential_conic_prefos import (
    AffineConeSpec,
    ConeSpec,
    Model,
    make_c_problem,
    read_reduced_model,
)


NONNEGATIVE = 0
SOC = 1
RSOC = 2
PSD = 3
EXPONENTIAL = 4
POWER = 5


def copy_csr(view):
    if view.nnz == 0:
        return sp.csr_matrix((view.rows, view.cols), dtype=np.float64)
    return numpy_csr(view)


def copy_symmetric_q(view, storage):
    if view.nnz == 0:
        return sp.csr_matrix((view.rows, view.cols), dtype=np.float64)
    return symmetric_q(view, storage).tocsr()


def copy_source_model(problem):
    cones = []
    for index in range(problem.n_cones):
        block = problem.cones[index]
        cones.append(
            ConeSpec(
                int(block.type),
                numpy_vector(block.indices, block.dimension, np.int32),
                int(block.matrix_order),
                float(block.power_alpha),
            )
        )
    affine_cones = []
    for index in range(problem.n_affine_cones):
        block = problem.affine_cones[index]
        affine_cones.append(
            AffineConeSpec(
                int(block.type),
                int(block.dimension),
                int(block.matrix_order),
                float(block.power_alpha),
            )
        )
    return Model(
        int(problem.n),
        copy_csr(problem.A),
        numpy_vector(problem.constraint_lower, problem.A.rows),
        numpy_vector(problem.constraint_upper, problem.A.rows),
        copy_symmetric_q(problem.Q, problem.q_storage),
        copy_csr(problem.R),
        numpy_vector(problem.D, problem.R.rows),
        numpy_vector(problem.c, problem.n),
        float(problem.objective_offset),
        numpy_vector(problem.box_indices, problem.n_box, np.int32),
        numpy_vector(problem.box_lower, problem.n_box),
        numpy_vector(problem.box_upper, problem.n_box),
        cones,
        copy_csr(problem.affine_cone_matrix),
        numpy_vector(
            problem.affine_cone_offset, problem.affine_cone_matrix.rows
        ),
        affine_cones,
    )


def selector(indices, n, values=None):
    indices = np.asarray(indices, dtype=np.int32)
    if values is None:
        values = np.ones(len(indices), dtype=np.float64)
    return sp.csc_matrix(
        (values, (np.arange(len(indices)), indices)), shape=(len(indices), n)
    )


def clarabel_data(model):
    blocks = []
    rhs = []
    cones = []

    finite_lower = np.isfinite(model.lower)
    finite_upper = np.isfinite(model.upper)
    equality = finite_lower & finite_upper & (model.lower == model.upper)
    upper = finite_upper & ~equality
    lower = finite_lower & ~equality
    if np.any(equality):
        blocks.append(model.A[equality])
        rhs.append(model.upper[equality])
        cones.append(clarabel.ZeroConeT(int(np.count_nonzero(equality))))
    inequality_count = int(np.count_nonzero(upper) + np.count_nonzero(lower))
    if inequality_count:
        inequality_blocks = []
        inequality_rhs = []
        if np.any(upper):
            inequality_blocks.append(model.A[upper])
            inequality_rhs.append(model.upper[upper])
        if np.any(lower):
            inequality_blocks.append(-model.A[lower])
            inequality_rhs.append(-model.lower[lower])
        blocks.append(sp.vstack(inequality_blocks, format="csc"))
        rhs.append(np.concatenate(inequality_rhs))
        cones.append(clarabel.NonnegativeConeT(inequality_count))

    box_fixed = np.isfinite(model.box_lower) & np.isfinite(model.box_upper)
    box_fixed &= model.box_lower == model.box_upper
    box_upper = np.isfinite(model.box_upper) & ~box_fixed
    box_lower = np.isfinite(model.box_lower) & ~box_fixed
    if np.any(box_fixed):
        blocks.append(selector(model.box_indices[box_fixed], model.n))
        rhs.append(model.box_upper[box_fixed])
        cones.append(clarabel.ZeroConeT(int(np.count_nonzero(box_fixed))))
    box_inequality_count = int(
        np.count_nonzero(box_upper) + np.count_nonzero(box_lower)
    )
    if box_inequality_count:
        box_blocks = []
        box_rhs = []
        if np.any(box_upper):
            box_blocks.append(selector(model.box_indices[box_upper], model.n))
            box_rhs.append(model.box_upper[box_upper])
        if np.any(box_lower):
            box_blocks.append(-selector(model.box_indices[box_lower], model.n))
            box_rhs.append(-model.box_lower[box_lower])
        blocks.append(sp.vstack(box_blocks, format="csc"))
        rhs.append(np.concatenate(box_rhs))
        cones.append(clarabel.NonnegativeConeT(box_inequality_count))

    cone_rows = []
    cone_columns = []
    cone_values = []
    cone_dimension = 0
    for cone in model.cones:
        dimension = len(cone.indices)
        if cone.cone_type == NONNEGATIVE:
            for row, column in enumerate(cone.indices):
                cone_rows.append(cone_dimension + row)
                cone_columns.append(column)
                cone_values.append(-1.0)
            cones.append(clarabel.NonnegativeConeT(dimension))
        elif cone.cone_type == SOC:
            for row, column in enumerate(cone.indices):
                cone_rows.append(cone_dimension + row)
                cone_columns.append(column)
                cone_values.append(-1.0)
            cones.append(clarabel.SecondOrderConeT(dimension))
        elif cone.cone_type == RSOC:
            u, v = cone.indices[:2]
            cone_rows.extend(
                [cone_dimension, cone_dimension, cone_dimension + 1, cone_dimension + 1]
            )
            cone_columns.extend([u, v, u, v])
            cone_values.extend([-1.0, -1.0, -1.0, 1.0])
            for row, column in enumerate(cone.indices[2:], start=2):
                cone_rows.append(cone_dimension + row)
                cone_columns.append(column)
                cone_values.append(-np.sqrt(2.0))
            cones.append(clarabel.SecondOrderConeT(dimension))
        elif cone.cone_type == PSD:
            for row, column in enumerate(cone.indices):
                cone_rows.append(cone_dimension + row)
                cone_columns.append(column)
                cone_values.append(-1.0)
            cones.append(clarabel.PSDTriangleConeT(cone.matrix_order))
        elif cone.cone_type == EXPONENTIAL:
            for row, column in enumerate(cone.indices):
                cone_rows.append(cone_dimension + row)
                cone_columns.append(column)
                cone_values.append(-1.0)
            cones.append(clarabel.ExponentialConeT())
        elif cone.cone_type == POWER:
            for row, column in enumerate(cone.indices):
                cone_rows.append(cone_dimension + row)
                cone_columns.append(column)
                cone_values.append(-1.0)
            cones.append(clarabel.PowerConeT(cone.power_alpha))
        else:
            raise ValueError(f"unsupported cone type {cone.cone_type}")
        cone_dimension += dimension
    if cone_dimension:
        blocks.append(
            sp.csc_matrix(
                (cone_values, (cone_rows, cone_columns)),
                shape=(cone_dimension, model.n),
            )
        )
        rhs.append(np.zeros(cone_dimension, dtype=np.float64))

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
    affine_row = 0
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
            raise ValueError(f"unsupported affine cone type {cone.cone_type}")
        row_slice = slice(affine_row, affine_row + dimension)
        blocks.append(-sp.csc_matrix(transform) @ affine_G[row_slice])
        rhs.append(transform @ affine_h[row_slice])
        cones.append(solver_cone)
        affine_row += dimension
    if affine_row != affine_G.shape[0] or affine_row != len(affine_h):
        raise ValueError("affine cone blocks do not cover G and h")

    operator = (
        sp.vstack(blocks, format="csc")
        if blocks
        else sp.csc_matrix((0, model.n), dtype=np.float64)
    )
    right_hand_side = np.concatenate(rhs) if rhs else np.empty(0)
    hessian = model.Q + model.R.T @ sp.diags(model.D) @ model.R
    return (
        sp.triu(hessian, format="csc"),
        np.ascontiguousarray(model.c),
        operator,
        np.ascontiguousarray(right_hand_side),
        cones,
    )


def solve_clarabel_data(data, offset, presolve, time_limit, canonicalize_seconds):
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    settings.max_iter = 500
    settings.time_limit = time_limit
    settings.presolve_enable = presolve
    settings.chordal_decomposition_enable = False
    start = time.perf_counter()
    solver = clarabel.DefaultSolver(*data, settings)
    setup_seconds = time.perf_counter() - start
    start = time.perf_counter()
    solution = solver.solve()
    solve_seconds = time.perf_counter() - start
    result = {
        "canonicalize_seconds": canonicalize_seconds,
        "setup_seconds": setup_seconds,
        "solve_seconds": solve_seconds,
        "total_seconds": canonicalize_seconds + setup_seconds + solve_seconds,
        "status": str(solution.status),
        "iterations": int(solution.iterations),
        "objective": float(solution.obj_val + offset),
        "x": np.asarray(solution.x, dtype=np.float64),
    }
    del solver, data
    gc.collect()
    return result


def solve_clarabel(model, presolve, time_limit):
    start = time.perf_counter()
    data = clarabel_data(model)
    canonicalize_seconds = time.perf_counter() - start
    return solve_clarabel_data(
        data, model.offset, presolve, time_limit, canonicalize_seconds
    )


def fixed_coordinate_values(data, tolerance=1e-10):
    values = {}

    def record(column, value):
        if column in values and abs(value - values[column]) > tolerance:
            raise ValueError(f"inconsistent fixed values for column {column}")
        values[column] = value

    for column, lower, upper in zip(
        data.box_indices, data.box_lower, data.box_upper
    ):
        if np.isfinite(lower) and lower == upper:
            record(int(column), float(lower))
    matrix = data.A.tocsr()
    for row in range(matrix.shape[0]):
        start = matrix.indptr[row]
        stop = matrix.indptr[row + 1]
        if stop - start != 1 or data.lower[row] != data.upper[row]:
            continue
        coefficient = matrix.data[start]
        if coefficient != 0.0 and np.isfinite(data.lower[row]):
            record(
                int(matrix.indices[start]),
                float(data.lower[row] / coefficient),
            )
    return values


def create_gurobi_model(data, time_limit):
    model = gp.Model()
    model.Params.OutputFlag = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = time_limit
    model.Params.FuncNonlinear = 1
    fixed_values = fixed_coordinate_values(data)
    lower = np.full(data.n, -gp.GRB.INFINITY)
    upper = np.full(data.n, gp.GRB.INFINITY)
    lower[data.box_indices] = data.box_lower
    upper[data.box_indices] = data.box_upper
    for cone in data.cones:
        if cone.cone_type == SOC:
            lower[cone.indices[0]] = max(lower[cone.indices[0]], 0.0)
        elif cone.cone_type == RSOC:
            lower[cone.indices[:2]] = np.maximum(lower[cone.indices[:2]], 0.0)
        elif cone.cone_type == EXPONENTIAL:
            lower[cone.indices[1:]] = np.maximum(lower[cone.indices[1:]], 0.0)
        elif cone.cone_type == POWER:
            lower[cone.indices[:2]] = np.maximum(lower[cone.indices[:2]], 0.0)
        elif cone.cone_type not in (NONNEGATIVE,):
            raise ValueError("Gurobi adapter does not support PSD cones")
        if cone.cone_type == NONNEGATIVE:
            lower[cone.indices] = np.maximum(lower[cone.indices], 0.0)
    x = model.addMVar(data.n, lb=lower, ub=upper)
    finite_lower = np.isfinite(data.lower)
    finite_upper = np.isfinite(data.upper)
    equality = finite_lower & finite_upper & (data.lower == data.upper)
    upper_rows = finite_upper & ~equality
    lower_rows = finite_lower & ~equality
    if np.any(equality):
        model.addMConstr(data.A[equality], x, gp.GRB.EQUAL, data.upper[equality])
    if np.any(upper_rows):
        model.addMConstr(
            data.A[upper_rows], x, gp.GRB.LESS_EQUAL, data.upper[upper_rows]
        )
    if np.any(lower_rows):
        model.addMConstr(
            data.A[lower_rows], x, gp.GRB.GREATER_EQUAL, data.lower[lower_rows]
        )
    for cone_index, cone in enumerate(data.cones):
        if cone.cone_type == NONNEGATIVE:
            continue
        if cone.cone_type == SOC:
            head = x[int(cone.indices[0])].item()
            tail = x[cone.indices[1:]]
            model.addConstr(tail @ tail <= head * head)
        elif cone.cone_type == RSOC:
            u = x[int(cone.indices[0])].item()
            v = x[int(cone.indices[1])].item()
            tail = x[cone.indices[2:]]
            model.addConstr(tail @ tail <= 2.0 * u * v)
        elif cone.cone_type == EXPONENTIAL:
            x_column, y_column, z_column = map(int, cone.indices)
            fixed_y = fixed_values.get(y_column)
            if fixed_y is None or abs(fixed_y - 1.0) > 1e-10:
                raise ValueError(
                    "Gurobi EXP comparison requires the perspective axis fixed to 1"
                )
            result = model.addVar(lb=0.0, name=f"__prefos_exp_result_{cone_index}")
            model.addGenConstrExp(
                x[x_column].item(), result, name=f"__prefos_exp_{cone_index}"
            )
            model.addConstr(
                result <= x[z_column].item(),
                name=f"__prefos_exp_epigraph_{cone_index}",
            )
        elif cone.cone_type == POWER:
            u_column, v_column, z_column = map(int, cone.indices)
            u = x[u_column].item()
            v = x[v_column].item()
            z = x[z_column].item()
            result = model.addVar(lb=0.0, name=f"__prefos_power_result_{cone_index}")
            expression = (u ** cone.power_alpha) * (
                v ** (1.0 - cone.power_alpha)
            )
            model.addGenConstrNL(
                result, expression, name=f"__prefos_power_{cone_index}"
            )
            model.addConstr(
                z <= result, name=f"__prefos_power_upper_{cone_index}"
            )
            model.addConstr(
                -z <= result, name=f"__prefos_power_lower_{cone_index}"
            )
    hessian = data.Q + data.R.T @ sp.diags(data.D) @ data.R
    model.setMObjective(
        0.5 * hessian,
        data.c,
        data.offset,
        xQ_L=x,
        xQ_R=x,
        xc=x,
        sense=gp.GRB.MINIMIZE,
    )
    model.update()
    return model


def gurobi_objective_data(model):
    expression = model.getObjective()
    linear = expression.getLinExpr() if isinstance(expression, gp.QuadExpr) else expression
    q = np.zeros(model.NumVars, dtype=np.float64)
    for index in range(linear.size()):
        q[linear.getVar(index).index] += linear.getCoeff(index)
    rows = []
    columns = []
    values = []
    if isinstance(expression, gp.QuadExpr):
        for index in range(expression.size()):
            left = expression.getVar1(index).index
            right = expression.getVar2(index).index
            coefficient = expression.getCoeff(index)
            if left == right:
                rows.append(left)
                columns.append(right)
                values.append(2.0 * coefficient)
            else:
                rows.append(min(left, right))
                columns.append(max(left, right))
                values.append(coefficient)
    hessian = sp.csc_matrix(
        (values, (rows, columns)), shape=(model.NumVars, model.NumVars)
    )
    return hessian, q, float(model.ObjCon)


def gurobi_quadratic_cone(model, constraint, variable_lower, tolerance=1e-10):
    expression = model.getQCRow(constraint)
    linear = expression.getLinExpr()
    if linear.size() != 0 or constraint.QCSense != gp.GRB.LESS_EQUAL:
        raise ValueError("unsupported affine or non-<= presolved Q constraint")
    diagonal = {}
    cross = {}
    for index in range(expression.size()):
        left = expression.getVar1(index).index
        right = expression.getVar2(index).index
        coefficient = float(expression.getCoeff(index))
        if abs(coefficient) <= tolerance:
            continue
        if left == right:
            diagonal[left] = diagonal.get(left, 0.0) + coefficient
        else:
            pair = (min(left, right), max(left, right))
            cross[pair] = cross.get(pair, 0.0) + coefficient
    diagonal = {key: value for key, value in diagonal.items() if abs(value) > tolerance}
    cross = {key: value for key, value in cross.items() if abs(value) > tolerance}
    positive = [(key, value) for key, value in diagonal.items() if value > 0.0]
    negative = [(key, value) for key, value in diagonal.items() if value < 0.0]
    rhs = float(constraint.QCRHS)
    if len(negative) == 1 and not cross and abs(rhs) <= tolerance:
        head, head_value = negative[0]
        if variable_lower[head] < -tolerance:
            raise ValueError("SOC head is not nonnegative after Gurobi presolve")
        cone_rows = [0]
        cone_columns = [head]
        cone_values = [np.sqrt(-head_value)]
        for row, (column, value) in enumerate(positive, start=1):
            cone_rows.append(row)
            cone_columns.append(column)
            cone_values.append(np.sqrt(value))
        dimension = 1 + len(positive)
        return (
            cone_rows,
            cone_columns,
            [-value for value in cone_values],
            np.zeros(dimension),
            clarabel.SecondOrderConeT(dimension),
        )

    if not negative and len(cross) == 1 and abs(rhs) <= tolerance:
        (u, v), cross_value = next(iter(cross.items()))
        if cross_value >= 0.0:
            raise ValueError("rotated cone cross term has the wrong sign")
        if variable_lower[u] < -tolerance or variable_lower[v] < -tolerance:
            raise ValueError("rotated cone axes are not nonnegative")
        rows = [0, 0, 1, 1]
        columns = [u, v, u, v]
        values = [1.0, 1.0, 1.0, -1.0]
        scale = -4.0 / cross_value
        for row, (column, value) in enumerate(positive, start=2):
            rows.append(row)
            columns.append(column)
            values.append(np.sqrt(scale * value))
        dimension = 2 + len(positive)
        return (
            rows,
            columns,
            [-value for value in values],
            np.zeros(dimension),
            clarabel.SecondOrderConeT(dimension),
        )

    if not negative and not cross and rhs > tolerance:
        rows = []
        columns = []
        values = []
        for row, (column, value) in enumerate(positive, start=2):
            rows.append(row)
            columns.append(column)
            values.append(-2.0 * np.sqrt(value))
        constant = np.concatenate(([rhs + 1.0, rhs - 1.0], np.zeros(len(positive))))
        return rows, columns, values, constant, clarabel.SecondOrderConeT(
            2 + len(positive)
        )

    raise ValueError(
        f"unsupported presolved Q constraint: diag+={len(positive)} "
        f"diag-={len(negative)} cross={len(cross)} rhs={rhs}"
    )


def gurobi_nonlinear_power_cone(model, constraint, tolerance=1e-10):
    resultant, opcodes, data, parents = model.getGenConstrNLAdv(constraint)
    children = [[] for _ in opcodes]
    roots = []
    for node, parent in enumerate(parents):
        if parent == -1:
            roots.append(node)
        else:
            children[parent].append(node)
    if len(roots) != 1 or opcodes[roots[0]] != gp.GRB.OPCODE_MULTIPLY:
        raise ValueError("unsupported nonlinear constraint root")
    terms = []
    for power_node in children[roots[0]]:
        if opcodes[power_node] != gp.GRB.OPCODE_POW:
            raise ValueError("nonlinear power cone term is not a power")
        variable = None
        exponent = None
        for child in children[power_node]:
            if opcodes[child] == gp.GRB.OPCODE_VARIABLE:
                variable = data[child]
            elif opcodes[child] == gp.GRB.OPCODE_CONSTANT:
                exponent = float(data[child])
        if variable is None or exponent is None:
            raise ValueError("malformed nonlinear power term")
        terms.append((variable, exponent))
    if (
        len(terms) != 2
        or min(terms[0][1], terms[1][1]) <= 0.0
        or abs(terms[0][1] + terms[1][1] - 1.0) > tolerance
    ):
        raise ValueError("nonlinear expression is not a 3D power-cone capacity")
    return resultant, terms


def clarabel_data_from_gurobi(model):
    blocks = []
    right_hand_sides = []
    cones = []
    matrix = model.getA().tocsc()
    senses = np.asarray(model.getAttr(gp.GRB.Attr.Sense))
    rhs = np.asarray(model.getAttr(gp.GRB.Attr.RHS), dtype=np.float64)
    equality = senses == gp.GRB.EQUAL
    upper = senses == gp.GRB.LESS_EQUAL
    lower = senses == gp.GRB.GREATER_EQUAL
    if np.any(equality):
        blocks.append(matrix[equality])
        right_hand_sides.append(rhs[equality])
        cones.append(clarabel.ZeroConeT(int(np.count_nonzero(equality))))
    inequality_count = int(np.count_nonzero(upper) + np.count_nonzero(lower))
    if inequality_count:
        pieces = []
        bounds = []
        if np.any(upper):
            pieces.append(matrix[upper])
            bounds.append(rhs[upper])
        if np.any(lower):
            pieces.append(-matrix[lower])
            bounds.append(-rhs[lower])
        blocks.append(sp.vstack(pieces, format="csc"))
        right_hand_sides.append(np.concatenate(bounds))
        cones.append(clarabel.NonnegativeConeT(inequality_count))

    variables = model.getVars()
    variable_lower = np.asarray([variable.LB for variable in variables])
    variable_upper = np.asarray([variable.UB for variable in variables])
    finite_lower_bound = variable_lower > -0.5 * gp.GRB.INFINITY
    finite_upper_bound = variable_upper < 0.5 * gp.GRB.INFINITY
    fixed = finite_lower_bound & finite_upper_bound
    fixed &= variable_lower == variable_upper
    finite_upper = finite_upper_bound & ~fixed
    finite_lower = finite_lower_bound & ~fixed
    if np.any(fixed):
        blocks.append(selector(np.flatnonzero(fixed), model.NumVars))
        right_hand_sides.append(variable_upper[fixed])
        cones.append(clarabel.ZeroConeT(int(np.count_nonzero(fixed))))
    bound_count = int(np.count_nonzero(finite_upper) + np.count_nonzero(finite_lower))
    if bound_count:
        pieces = []
        bounds = []
        if np.any(finite_upper):
            pieces.append(selector(np.flatnonzero(finite_upper), model.NumVars))
            bounds.append(variable_upper[finite_upper])
        if np.any(finite_lower):
            pieces.append(-selector(np.flatnonzero(finite_lower), model.NumVars))
            bounds.append(-variable_lower[finite_lower])
        blocks.append(sp.vstack(pieces, format="csc"))
        right_hand_sides.append(np.concatenate(bounds))
        cones.append(clarabel.NonnegativeConeT(bound_count))

    quadratic_rows = []
    quadratic_columns = []
    quadratic_values = []
    quadratic_rhs = []
    row_offset = 0
    for constraint in model.getQConstrs():
        rows, columns, values, bound, cone = gurobi_quadratic_cone(
            model, constraint, variable_lower
        )
        quadratic_rows.extend(row_offset + row for row in rows)
        quadratic_columns.extend(columns)
        quadratic_values.extend(values)
        quadratic_rhs.append(bound)
        cones.append(cone)
        row_offset += len(bound)
    if row_offset:
        blocks.append(
            sp.csc_matrix(
                (quadratic_values, (quadratic_rows, quadratic_columns)),
                shape=(row_offset, model.NumVars),
            )
        )
        right_hand_sides.append(np.concatenate(quadratic_rhs))

    general_rows = []
    general_columns = []
    general_values = []
    general_rhs = []
    general_row_offset = 0
    for constraint in model.getGenConstrs():
        constraint_type = constraint.GenConstrType
        if constraint_type == gp.GRB.GENCONSTR_EXP:
            argument, resultant = model.getGenConstrExp(constraint)
            general_rows.extend(
                [general_row_offset, general_row_offset + 2]
            )
            general_columns.extend([argument.index, resultant.index])
            general_values.extend([-1.0, -1.0])
            general_rhs.append(np.array([0.0, 1.0, 0.0]))
            cones.append(clarabel.ExponentialConeT())
            general_row_offset += 3
        elif constraint_type == gp.GRB.GENCONSTR_NL:
            resultant, terms = gurobi_nonlinear_power_cone(model, constraint)
            for row, (variable, _) in enumerate(terms):
                general_rows.append(general_row_offset + row)
                general_columns.append(variable.index)
                general_values.append(-1.0)
            general_rows.append(general_row_offset + 2)
            general_columns.append(resultant.index)
            general_values.append(-1.0)
            general_rhs.append(np.zeros(3))
            cones.append(clarabel.PowerConeT(terms[0][1]))
            general_row_offset += 3
        else:
            raise ValueError(
                f"unsupported presolved general constraint type {constraint_type}"
            )
    if general_row_offset:
        blocks.append(
            sp.csc_matrix(
                (general_values, (general_rows, general_columns)),
                shape=(general_row_offset, model.NumVars),
            )
        )
        right_hand_sides.append(np.concatenate(general_rhs))

    operator = (
        sp.vstack(blocks, format="csc")
        if blocks
        else sp.csc_matrix((0, model.NumVars), dtype=np.float64)
    )
    all_rhs = np.concatenate(right_hand_sides) if right_hand_sides else np.empty(0)
    hessian, objective, offset = gurobi_objective_data(model)
    return (hessian, objective, operator, all_rhs, cones), offset


def run_gurobi_presolve_clarabel(data, time_limit):
    start = time.perf_counter()
    model = create_gurobi_model(data, time_limit)
    build_seconds = time.perf_counter() - start
    original_size = {
        "source_variables_original": int(data.n),
        "variables_original": int(model.NumVars),
        "reformulation_variables": int(model.NumVars - data.n),
        "linear_constraints_original": int(model.NumConstrs),
        "quadratic_constraints_original": int(model.NumQConstrs),
        "general_constraints_original": int(model.NumGenConstrs),
    }
    start = time.perf_counter()
    reduced = model.presolve()
    presolve_seconds = time.perf_counter() - start
    model.dispose()
    del model
    gc.collect()
    reduced.Params.OutputFlag = 0
    reduced.update()
    reduced_size = {
        "variables_reduced": int(reduced.NumVars),
        "linear_constraints_reduced": int(reduced.NumConstrs),
        "quadratic_constraints_reduced": int(reduced.NumQConstrs),
        "general_constraints_reduced": int(reduced.NumGenConstrs),
    }
    start = time.perf_counter()
    clarabel_data_reduced, objective_offset = clarabel_data_from_gurobi(reduced)
    canonicalize_seconds = time.perf_counter() - start
    clarabel_result = solve_clarabel_data(
        clarabel_data_reduced,
        objective_offset,
        False,
        time_limit,
        canonicalize_seconds,
    )
    del clarabel_result["x"]
    result = {
        "build_seconds": build_seconds,
        "presolve_seconds": presolve_seconds,
        "external_presolve_seconds": build_seconds + presolve_seconds,
        "total_seconds": build_seconds
        + presolve_seconds
        + clarabel_result["total_seconds"],
        "clarabel": clarabel_result,
        **original_size,
        **reduced_size,
    }
    reduced.dispose()
    gc.collect()
    return result


def run_prefos(prefos, source, time_limit, use_gpu=False, setting_overrides=None):
    problem, keepalive = make_c_problem(source)
    settings = prefos.prefos_default_settings()
    if use_gpu:
        settings.linear_propagation_gpu = 1
    if setting_overrides:
        for name, value in setting_overrides.items():
            setattr(settings, name, value)
    presolver = ct.c_void_p()
    start = time.perf_counter()
    status = prefos.prefos_create_presolver(
        ct.byref(problem), ct.byref(settings), ct.byref(presolver)
    )
    create_seconds = time.perf_counter() - start
    del keepalive
    if status != 0:
        raise RuntimeError(f"prefos_create_presolver returned {status}")
    try:
        start = time.perf_counter()
        status = prefos.prefos_run_presolve(presolver)
        presolve_seconds = time.perf_counter() - start
        if status not in (0, 1, 2):
            raise RuntimeError(f"prefos_run_presolve returned {status}")
        reduced = read_reduced_model(prefos.prefos_get_reduced_problem(presolver).contents)
        statistics = stats_dict(prefos.prefos_get_stats(presolver).contents)
        clarabel_result = solve_clarabel(reduced, False, time_limit)
        original_x = np.empty(source.n, dtype=np.float64)
        start = time.perf_counter()
        postsolve_status = prefos.prefos_postsolve_primal(
            presolver,
            ptr(clarabel_result["x"], ct.c_double),
            ptr(original_x, ct.c_double),
        )
        postsolve_seconds = time.perf_counter() - start
        if postsolve_status != 0:
            raise RuntimeError(f"prefos_postsolve_primal returned {postsolve_status}")
        verification = Verification()
        verification_status = prefos.prefos_verify_postsolve_primal(
            presolver,
            ptr(clarabel_result["x"], ct.c_double),
            1e-5,
            ct.byref(verification),
        )
        if verification_status != 0:
            raise RuntimeError(
                f"prefos_verify_postsolve_primal returned {verification_status}"
            )
        del clarabel_result["x"]
        return {
            "create_seconds": create_seconds,
            "presolve_seconds": presolve_seconds,
            "postsolve_seconds": postsolve_seconds,
            "total_seconds": create_seconds
            + presolve_seconds
            + clarabel_result["total_seconds"]
            + postsolve_seconds,
            "presolve_status": int(status),
            "postsolve_verification_passed": bool(verification.passed),
            "postsolve_objective_absolute_error": float(
                verification.objective_absolute_error
            ),
            "postsolve_original_row_violation": float(
                verification.original_row_violation
            ),
            "postsolve_original_cone_violation": float(
                verification.original_cone_violation
            ),
            "clarabel": clarabel_result,
            **statistics,
        }
    finally:
        prefos.prefos_free_presolver(presolver)


def load_cbf(parser, filename):
    pointer = parser.read_cbf_file(str(filename).encode())
    if not pointer:
        raise RuntimeError("CBF parser failed")
    try:
        problem, keepalive, metadata = make_prefos_view(pointer)
        model = copy_source_model(problem)
        del keepalive
        return model, metadata
    finally:
        parser.qp_problem_free(pointer)


def clean_result(result):
    if isinstance(result, dict):
        return {key: clean_result(value) for key, value in result.items()}
    if isinstance(result, np.generic):
        return result.item()
    return result


def print_result(result):
    ours = result["prefos_clarabel"]
    gurobi = result["gurobi_clarabel"]
    ours_label = (
        "PreFOS(linear-off)+Clarabel"
        if result.get("prefos_settings", {}).get("linear_propagation") == 0
        else "PreFOS+Clarabel"
    )
    baseline = result.get("clarabel_original")
    baseline_text = (
        f" | Clarabel={baseline['total_seconds']:.3f}s" if baseline else ""
    )
    print(
        f"{Path(result['file']).name}: "
        f"{ours_label}={ours['total_seconds']:.3f}s "
        f"(pre={ours['create_seconds'] + ours['presolve_seconds']:.3f}, "
        f"solve={ours['clarabel']['setup_seconds'] + ours['clarabel']['solve_seconds']:.3f})"
        f"{baseline_text} | "
        f"Gurobi+Clarabel={gurobi['total_seconds']:.3f}s "
        f"(build={gurobi['build_seconds']:.3f}, "
        f"pre={gurobi['presolve_seconds']:.3f}, "
        f"solve={gurobi['clarabel']['setup_seconds'] + gurobi['clarabel']['solve_seconds']:.3f})"
    )
    print(
        f"  PreFOS n={ours['variables_original']:,}->{ours['variables_reduced']:,} "
        f"m={ours['rows_original']:,}->{ours['rows_reduced']:,}; "
        f"Gurobi n={gurobi['source_variables_original']:,}"
        f"+{gurobi['reformulation_variables']:,} lifted"
        f"->{gurobi['variables_reduced']:,} "
        f"lin={gurobi['linear_constraints_original']:,}->"
        f"{gurobi['linear_constraints_reduced']:,} "
        f"gen={gurobi['general_constraints_original']:,}->"
        f"{gurobi['general_constraints_reduced']:,}"
    )
    objective_gap = abs(
        ours["clarabel"]["objective"] - gurobi["clarabel"]["objective"]
    )
    solved_statuses = {"Solved", "AlmostSolved"}
    comparable = (
        ours["clarabel"]["status"] in solved_statuses
        and gurobi["clarabel"]["status"] in solved_statuses
    )
    objective_text = (
        f"{objective_gap:.3e}" if comparable else f"n/a (raw={objective_gap:.3e})"
    )
    print(
        f"  objective gap={objective_text}; "
        f"statuses={ours['clarabel']['status']}/{gurobi['clarabel']['status']}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--library", default="build/libPreFOS.so")
    parser.add_argument(
        "--pdhcg-library",
        help="path to the PDHCG-II shared library; required unless --scalar-cbf is used",
    )
    parser.add_argument("--time-limit", type=float, default=300.0)
    parser.add_argument("--jsonl")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--scalar-cbf", action="store_true")
    parser.add_argument("--gpu-linear-propagation", action="store_true")
    parser.add_argument("--disable-prefos-linear-propagation", action="store_true")
    parser.add_argument(
        "--prefos-propagated-bound-policy",
        choices=("first-order", "interior-point"),
        default="first-order",
    )
    parser.add_argument("--prefos-affine-cone-aggregation", action="store_true")
    parser.add_argument("--async-warmup-gpu", action="store_true")
    args = parser.parse_args()

    if not args.scalar_cbf and args.pdhcg_library is None:
        parser.error("--pdhcg-library is required unless --scalar-cbf is used")

    if gp is None:
        raise SystemExit("gurobipy is required for this comparison benchmark")

    prefos = configure_prefos(Path(args.library).resolve())
    prefos.prefos_postsolve_primal.argtypes = [
        ct.c_void_p,
        ct.POINTER(ct.c_double),
        ct.POINTER(ct.c_double),
    ]
    prefos.prefos_postsolve_primal.restype = ct.c_int
    if args.async_warmup_gpu and not prefos.prefos_gpu_warmup_async():
        raise SystemExit("CUDA asynchronous warmup could not be started")
    cbf_parser = (
        None
        if args.scalar_cbf
        else configure_parser(Path(args.pdhcg_library).resolve())
    )
    output = open(args.jsonl, "a", encoding="utf-8") if args.jsonl else None
    try:
        for name in args.inputs:
            filename = Path(name).resolve()
            start = time.perf_counter()
            if args.scalar_cbf:
                model, metadata = load_scalar_cbf(filename)
            else:
                model, metadata = load_cbf(cbf_parser, filename)
            load_seconds = time.perf_counter() - start
            baseline = None
            if not args.skip_baseline:
                baseline = solve_clarabel(model, False, args.time_limit)
                del baseline["x"]
            prefos_setting_overrides = {}
            if args.disable_prefos_linear_propagation:
                prefos_setting_overrides["linear_propagation"] = 0
            if args.prefos_propagated_bound_policy == "interior-point":
                prefos_setting_overrides["propagated_bound_policy"] = 1
            if args.prefos_affine_cone_aggregation:
                prefos_setting_overrides["affine_cone_coordinate_aggregation"] = 1
            ours = run_prefos(
                prefos,
                model,
                args.time_limit,
                args.gpu_linear_propagation,
                prefos_setting_overrides,
            )
            gurobi = run_gurobi_presolve_clarabel(model, args.time_limit)
            result_data = {
                "file": str(filename),
                "load_seconds": load_seconds,
                "clarabel_presolve": False,
                **metadata,
                "prefos_settings": prefos_setting_overrides,
                "prefos_clarabel": ours,
                "gurobi_clarabel": gurobi,
            }
            if baseline:
                result_data["clarabel_original"] = baseline
            result = clean_result(result_data)
            print_result(result)
            if output:
                output.write(json.dumps(result, sort_keys=True) + "\n")
                output.flush()
    finally:
        if output:
            output.close()


if __name__ == "__main__":
    main()
