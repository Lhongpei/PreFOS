#!/usr/bin/env python3
# Copyright 2026 Hongpei Li
# SPDX-License-Identifier: Apache-2.0

"""Check original, PreFOS, and PSLP LPs with the same reference solver."""

import argparse
import ctypes as ct
import json
import time
from pathlib import Path

import gurobipy as gp
import numpy as np
import scipy.sparse as sp

from benchmark_lp_pslp import (
    PreFOSSettings,
    configure_prefos,
    configure_pslp,
    parse_lp,
    prefos_problem,
    ptr,
)
from differential_prefos import numpy_csr, numpy_vector


PREFOS_ABLATIONS = {
    "default": (),
    "no-linear": (("linear_propagation", 0),),
    "no-singleton": (("singleton_column_reduction", 0),),
    "no-doubleton": (("bounded_doubleton_substitution", 0),),
    "no-dual": (("dual_fixing", 0),),
    "no-parallel-columns": (("parallel_column_reduction", 0),),
    "no-parallel-rows": (("parallel_row_max_average_nnz", 1e-300),),
    "no-redundant-activity": (("remove_redundant_rows", 0),),
    "no-free-columns": (("free_column_substitution", 0),),
    "no-column-reductions": (
        ("singleton_column_reduction", 0),
        ("bounded_doubleton_substitution", 0),
        ("dual_fixing", 0),
        ("parallel_column_reduction", 0),
        ("free_column_substitution", 0),
    ),
    "no-row-reductions": (
        ("linear_propagation", 0),
        ("parallel_row_max_average_nnz", 1e-300),
        ("remove_redundant_rows", 0),
    ),
    "minimal": (
        ("linear_propagation", 0),
        ("singleton_column_reduction", 0),
        ("bounded_doubleton_substitution", 0),
        ("dual_fixing", 0),
        ("parallel_column_reduction", 0),
        ("parallel_row_max_average_nnz", 1e-300),
        ("remove_redundant_rows", 0),
        ("free_column_substitution", 0),
    ),
}


def solve_lp(data, time_limit):
    environment = gp.Env(empty=True)
    environment.setParam("OutputFlag", 0)
    environment.start()
    model = gp.Model(env=environment)
    try:
        model.Params.OutputFlag = 0
        model.Params.TimeLimit = time_limit
        x = model.addMVar(
            data["columns"],
            lb=data["column_lower"],
            ub=data["column_upper"],
            obj=data["objective"],
        )
        matrix = sp.csr_matrix(
            (
                data["A_values"],
                data["A_indices"],
                data["A_indptr"],
            ),
            shape=(data["rows"], data["columns"]),
        )
        lower = data["row_lower"]
        upper = data["row_upper"]
        equal = np.isfinite(lower) & np.isfinite(upper) & (lower == upper)
        finite_lower = np.isfinite(lower) & ~equal
        finite_upper = np.isfinite(upper) & ~equal
        if np.any(equal):
            model.addMConstr(matrix[equal], x, "=", lower[equal])
        if np.any(finite_lower):
            model.addMConstr(
                matrix[finite_lower], x, ">", lower[finite_lower]
            )
        if np.any(finite_upper):
            model.addMConstr(
                matrix[finite_upper], x, "<", upper[finite_upper]
            )
        model.ObjCon = float(data.get("objective_offset", 0.0))
        start = time.perf_counter()
        model.optimize()
        elapsed = time.perf_counter() - start
        return {
            "status": int(model.Status),
            "objective": float(model.ObjVal) if model.SolCount else None,
            "solve_seconds": elapsed,
        }
    finally:
        model.dispose()
        environment.dispose()


def csr_arrays(matrix):
    return {
        "A_values": np.ascontiguousarray(matrix.data, dtype=np.float64),
        "A_indices": np.ascontiguousarray(matrix.indices, dtype=np.int32),
        "A_indptr": np.ascontiguousarray(matrix.indptr, dtype=np.int32),
    }


def extract_prefos(library, data, ablation, feasibility_tolerance):
    problem, keepalive = prefos_problem(data)
    settings = library.prefos_default_settings()
    settings.feasibility_tolerance = feasibility_tolerance
    for field, value in PREFOS_ABLATIONS[ablation]:
        setattr(settings, field, value)
    presolver = ct.c_void_p()
    start = time.perf_counter()
    create_status = library.prefos_create_presolver(
        ct.byref(problem), ct.byref(settings), ct.byref(presolver)
    )
    if create_status != 0:
        raise RuntimeError(f"PreFOS create status {create_status}")
    try:
        status = library.prefos_run_presolve(presolver)
        elapsed = time.perf_counter() - start
        if status not in (0, 1):
            raise RuntimeError(f"PreFOS status {status}")
        reduced = library.prefos_get_reduced_problem(presolver).contents
        matrix = numpy_csr(reduced.A)
        box_indices = numpy_vector(
            reduced.box_indices, reduced.n_box, np.int32
        )
        column_lower = np.full(reduced.n, -np.inf)
        column_upper = np.full(reduced.n, np.inf)
        column_lower[box_indices] = numpy_vector(
            reduced.box_lower, reduced.n_box
        )
        column_upper[box_indices] = numpy_vector(
            reduced.box_upper, reduced.n_box
        )
        result = {
            **csr_arrays(matrix),
            "rows": int(reduced.A.rows),
            "columns": int(reduced.n),
            "nnz": int(reduced.A.nnz),
            "row_lower": numpy_vector(
                reduced.constraint_lower, reduced.A.rows
            ),
            "row_upper": numpy_vector(
                reduced.constraint_upper, reduced.A.rows
            ),
            "column_lower": column_lower,
            "column_upper": column_upper,
            "objective": numpy_vector(reduced.c, reduced.n),
            "objective_offset": float(reduced.objective_offset),
        }
        return status, elapsed, result
    finally:
        library.prefos_free_presolver(presolver)
        del keepalive


def extract_pslp(library, data, max_time):
    settings = library.default_settings()
    if not settings:
        raise MemoryError("PSLP settings allocation failed")
    settings.contents.verbose = False
    settings.contents.max_time = max_time
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
    if not presolver:
        library.free_settings(settings)
        raise MemoryError("PSLP presolver allocation failed")
    try:
        status = library.run_presolver(presolver)
        elapsed = time.perf_counter() - start
        if status not in (0, 1):
            raise RuntimeError(f"PSLP status {status}")
        reduced = presolver.contents.reduced_problem.contents
        result = {
            "A_values": np.ctypeslib.as_array(
                reduced.values, shape=(reduced.nnz,)
            ).copy(),
            "A_indices": np.ctypeslib.as_array(
                reduced.column_indices, shape=(reduced.nnz,)
            ).copy().astype(np.int32),
            "A_indptr": np.ctypeslib.as_array(
                reduced.row_pointers, shape=(reduced.rows + 1,)
            ).copy().astype(np.int32),
            "rows": int(reduced.rows),
            "columns": int(reduced.columns),
            "nnz": int(reduced.nnz),
            "row_lower": np.ctypeslib.as_array(
                reduced.row_lower, shape=(reduced.rows,)
            ).copy(),
            "row_upper": np.ctypeslib.as_array(
                reduced.row_upper, shape=(reduced.rows,)
            ).copy(),
            "column_lower": np.ctypeslib.as_array(
                reduced.column_lower, shape=(reduced.columns,)
            ).copy(),
            "column_upper": np.ctypeslib.as_array(
                reduced.column_upper, shape=(reduced.columns,)
            ).copy(),
            "objective": np.ctypeslib.as_array(
                reduced.objective, shape=(reduced.columns,)
            ).copy(),
            "objective_offset": (
                float(reduced.objective_offset)
                + float(data.get("objective_offset", 0.0))
            ),
        }
        return status, elapsed, result
    finally:
        library.free_presolver(presolver)
        library.free_settings(settings)


def summarize(name, presolve_status, presolve_seconds, reduced, solved,
              reference):
    result = {
        "presolve_status": int(presolve_status),
        "presolve_seconds": presolve_seconds,
        "rows": reduced["rows"],
        "columns": reduced["columns"],
        "nnz": reduced["nnz"],
        **solved,
    }
    if solved["objective"] is not None and reference is not None:
        result["objective_error"] = solved["objective"] - reference
    objective = solved["objective"]
    objective_text = "n/a" if objective is None else f"{objective:.12g}"
    error = result.get("objective_error")
    error_text = "n/a" if error is None else f"{error:+.3e}"
    print(
        f"  {name:22s} ({reduced['rows']},{reduced['columns']},"
        f"{reduced['nnz']}) pre={presolve_seconds:.4f}s "
        f"solve_status={solved['status']} obj={objective_text} "
        f"error={error_text}",
        flush=True,
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--prefos-library", required=True, type=Path)
    parser.add_argument("--pslp-library", type=Path)
    parser.add_argument(
        "--parser", choices=("highs", "gurobi"), default="gurobi"
    )
    parser.add_argument("--solve-time-limit", type=float, default=300.0)
    parser.add_argument("--pslp-max-time", type=float, default=600.0)
    parser.add_argument("--feasibility-tolerance", type=float, default=1e-6)
    parser.add_argument("--ablate", action="store_true")
    parser.add_argument(
        "--prefos-variants",
        nargs="+",
        choices=PREFOS_ABLATIONS,
        help="run only these PreFOS configurations",
    )
    parser.add_argument("--jsonl", type=Path)
    args = parser.parse_args()

    prefos = configure_prefos(args.prefos_library.resolve())
    prefos.prefos_default_settings.restype = PreFOSSettings
    pslp = (
        configure_pslp(args.pslp_library.resolve())
        if args.pslp_library
        else None
    )
    output = args.jsonl.open("w", encoding="utf-8") if args.jsonl else None
    try:
        for path in args.inputs:
            print(path.name, flush=True)
            data = parse_lp(path, args.parser)
            original = solve_lp(data, args.solve_time_limit)
            reference = original["objective"]
            result = {
                "file": str(path.resolve()),
                "original": {
                    "rows": data["rows"],
                    "columns": data["columns"],
                    "nnz": data["nnz"],
                    **original,
                },
                "prefos": {},
            }
            print(
                f"  {'original':22s} ({data['rows']},{data['columns']},"
                f"{data['nnz']}) solve_status={original['status']} "
                f"obj={reference}",
                flush=True,
            )
            variants = (
                args.prefos_variants
                if args.prefos_variants
                else PREFOS_ABLATIONS
                if args.ablate
                else ("default",)
            )
            for name in variants:
                status, elapsed, reduced = extract_prefos(
                    prefos,
                    data,
                    name,
                    args.feasibility_tolerance,
                )
                solved = solve_lp(reduced, args.solve_time_limit)
                result["prefos"][name] = summarize(
                    f"PreFOS/{name}",
                    status,
                    elapsed,
                    reduced,
                    solved,
                    reference,
                )
            if pslp is not None:
                status, elapsed, reduced = extract_pslp(
                    pslp, data, args.pslp_max_time
                )
                solved = solve_lp(reduced, args.solve_time_limit)
                result["pslp"] = summarize(
                    "PSLP",
                    status,
                    elapsed,
                    reduced,
                    solved,
                    reference,
                )
            if output:
                output.write(json.dumps(result, sort_keys=True) + "\n")
                output.flush()
    finally:
        if output:
            output.close()


if __name__ == "__main__":
    main()
