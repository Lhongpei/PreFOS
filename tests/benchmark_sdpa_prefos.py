#!/usr/bin/env python3
# Copyright 2026 Hongpei Li
# SPDX-License-Identifier: Apache-2.0

"""Benchmark PreFOS on sparse SDPA ``.dat-s`` semidefinite programs."""

import argparse
import ctypes as ct
import json
import math
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from benchmark_cbf_prefos import (
    current_rss_mib,
    peak_rss_mib,
    stats_dict,
    configure_prefos,
)
from differential_prefos import AffineConeBlock, CsrMatrix, ProblemData, ptr


PREFOS_NONNEGATIVE = 0
PREFOS_PSD = 3
PREFOS_Q_UPPER_TRIANGULAR = 0


def sdpa_tokens(path):
    translation = str.maketrans({"{": " ", "}": " ", ",": " "})
    with path.open("r", encoding="ascii") as stream:
        for line in stream:
            stripped = line.lstrip()
            if not stripped or stripped.startswith(("*", '"')):
                continue
            yield from stripped.translate(translation).split()


def parse_float(token):
    return float(token.replace("D", "E").replace("d", "e"))


def take(tokens, count, converter, label):
    values = []
    for _ in range(count):
        try:
            values.append(converter(next(tokens)))
        except StopIteration as error:
            raise ValueError(f"unexpected end of file while reading {label}") from error
    return values


def packed_svec_index(row, column):
    row -= 1
    column -= 1
    if row < column:
        row, column = column, row
    return row * (row + 1) // 2 + column


def parse_sdpa_sparse(path):
    tokens = iter(sdpa_tokens(path))
    try:
        n_variables = int(next(tokens))
        n_blocks = int(next(tokens))
    except StopIteration as error:
        raise ValueError("incomplete SDPA header") from error
    if n_variables < 0 or n_blocks < 0:
        raise ValueError("negative SDPA dimensions")
    block_sizes = take(tokens, n_blocks, int, "block structure")
    # SDPA stores max b^T x with slack sum_i x_i F_i - F_0 in K.
    objective = -np.asarray(
        take(tokens, n_variables, parse_float, "objective"), dtype=np.float64
    )

    block_offsets = []
    blocks = []
    total_rows = 0
    psd_orders = []
    nonnegative_rows = 0
    for size in block_sizes:
        if size == 0:
            raise ValueError("zero-sized SDPA block")
        block_offsets.append(total_rows)
        if size > 0:
            dimension = size * (size + 1) // 2
            blocks.append((PREFOS_PSD, dimension, size))
            psd_orders.append(size)
        else:
            dimension = -size
            blocks.append((PREFOS_NONNEGATIVE, dimension, 0))
            nonnegative_rows += dimension
        total_rows += dimension

    affine_offset = np.zeros(total_rows, dtype=np.float64)
    row_indices = []
    column_indices = []
    coefficients = []
    explicit_zeros = 0
    entry_count = 0
    while True:
        try:
            matrix_number = int(next(tokens))
        except StopIteration:
            break
        fields = take(tokens, 4, str, "sparse matrix entry")
        block_number = int(fields[0])
        matrix_row = int(fields[1])
        matrix_column = int(fields[2])
        value = parse_float(fields[3])
        entry_count += 1
        if matrix_number < 0 or matrix_number > n_variables:
            raise ValueError(f"invalid matrix number {matrix_number}")
        if block_number < 1 or block_number > n_blocks:
            raise ValueError(f"invalid block number {block_number}")
        size = block_sizes[block_number - 1]
        order = abs(size)
        if not (1 <= matrix_row <= order and 1 <= matrix_column <= order):
            raise ValueError("matrix entry lies outside its SDPA block")
        if size < 0:
            if matrix_row != matrix_column:
                raise ValueError("off-diagonal entry in a diagonal SDPA block")
            affine_row = block_offsets[block_number - 1] + matrix_row - 1
            scale = 1.0
        else:
            affine_row = block_offsets[block_number - 1] + packed_svec_index(
                matrix_row, matrix_column
            )
            scale = 1.0 if matrix_row == matrix_column else math.sqrt(2.0)
        scaled_value = scale * value
        if scaled_value == 0.0:
            explicit_zeros += 1
            continue
        if matrix_number == 0:
            affine_offset[affine_row] -= scaled_value
        else:
            row_indices.append(affine_row)
            column_indices.append(matrix_number - 1)
            coefficients.append(scaled_value)

    matrix = sp.coo_matrix(
        (coefficients, (row_indices, column_indices)),
        shape=(total_rows, n_variables),
        dtype=np.float64,
    ).tocsr()
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()
    metadata = {
        "variables": n_variables,
        "affine_rows": total_rows,
        "affine_nnz": int(matrix.nnz),
        "affine_blocks": n_blocks,
        "psd_blocks": len(psd_orders),
        "max_psd_order": max(psd_orders, default=0),
        "nonnegative_rows": nonnegative_rows,
        "source_entries": entry_count,
        "explicit_zeros": explicit_zeros,
    }
    return objective, block_sizes, blocks, affine_offset, matrix, metadata


def make_problem(objective, blocks, affine_offset, matrix):
    n = len(objective)
    objective = np.ascontiguousarray(objective, dtype=np.float64)
    affine_offset = np.ascontiguousarray(affine_offset, dtype=np.float64)
    values = np.ascontiguousarray(matrix.data, dtype=np.float64)
    columns = np.ascontiguousarray(matrix.indices, dtype=np.int32)
    rows = np.ascontiguousarray(matrix.indptr, dtype=np.int32)
    empty_a_rows = np.zeros(1, dtype=np.int32)
    empty_q_rows = np.zeros(n + 1, dtype=np.int32)
    empty_r_rows = np.zeros(1, dtype=np.int32)
    box_indices = np.arange(n, dtype=np.int32)
    box_lower = np.full(n, -np.inf, dtype=np.float64)
    box_upper = np.full(n, np.inf, dtype=np.float64)
    affine_blocks = (AffineConeBlock * len(blocks))()
    for index, (cone_type, dimension, matrix_order) in enumerate(blocks):
        affine_blocks[index] = AffineConeBlock(
            cone_type, dimension, matrix_order, 0.0
        )
    problem = ProblemData(
        n,
        CsrMatrix(0, n, 0, None, None, ptr(empty_a_rows, ct.c_int)),
        None,
        None,
        CsrMatrix(n, n, 0, None, None, ptr(empty_q_rows, ct.c_int)),
        PREFOS_Q_UPPER_TRIANGULAR,
        CsrMatrix(0, n, 0, None, None, ptr(empty_r_rows, ct.c_int)),
        None,
        ptr(objective, ct.c_double),
        0.0,
        n,
        ptr(box_indices, ct.c_int),
        ptr(box_lower, ct.c_double),
        ptr(box_upper, ct.c_double),
        0,
        None,
        CsrMatrix(
            matrix.shape[0],
            matrix.shape[1],
            matrix.nnz,
            ptr(values, ct.c_double),
            ptr(columns, ct.c_int),
            ptr(rows, ct.c_int),
        ),
        ptr(affine_offset, ct.c_double),
        len(blocks),
        ct.cast(affine_blocks, ct.POINTER(AffineConeBlock)),
    )
    keepalive = (
        values,
        columns,
        rows,
        empty_a_rows,
        empty_q_rows,
        empty_r_rows,
        objective,
        affine_offset,
        box_indices,
        box_lower,
        box_upper,
        affine_blocks,
    )
    return problem, keepalive


def benchmark_file(prefos, path, overrides):
    result = {"file": str(path), "rss_baseline_mib": current_rss_mib()}
    presolver = ct.c_void_p()
    try:
        start = time.perf_counter()
        objective, _, blocks, offset, matrix, metadata = parse_sdpa_sparse(path)
        result["parse_seconds"] = time.perf_counter() - start
        result.update(metadata)
        result["rss_after_parse_mib"] = current_rss_mib()

        problem, keepalive = make_problem(objective, blocks, offset, matrix)
        settings = prefos.prefos_default_settings()
        for name, value in overrides.items():
            setattr(settings, name, value)
        start = time.perf_counter()
        status = prefos.prefos_create_presolver(
            ct.byref(problem), ct.byref(settings), ct.byref(presolver)
        )
        result["create_seconds"] = time.perf_counter() - start
        result["create_status"] = int(status)
        if status != 0:
            raise RuntimeError(f"prefos_create_presolver returned {status}")

        start = time.perf_counter()
        status = prefos.prefos_run_presolve(presolver)
        result["presolve_seconds"] = time.perf_counter() - start
        result["presolve_status"] = int(status)
        if status not in (0, 1, 2):
            raise RuntimeError(f"prefos_run_presolve returned {status}")
        result.update(stats_dict(prefos.prefos_get_stats(presolver).contents))
        analysis_count = ct.c_size_t()
        analyses = prefos.prefos_get_psd_structure_analyses(
            presolver, ct.byref(analysis_count)
        )
        result["splittable_psd_blocks"] = [
            {
                "affine_cone_index": int(analyses[index].affine_cone_index),
                "matrix_order": int(analyses[index].matrix_order),
                "components": int(analyses[index].connected_components),
                "scalar_components": int(analyses[index].scalar_components),
                "emitted_cone_blocks": int(
                    analyses[index].emitted_cone_blocks
                ),
                "largest_component_order": int(
                    analyses[index].largest_component_order
                ),
                "removed_cross_coordinates": int(
                    analyses[index].dimension
                    - analyses[index].decomposed_dimension
                ),
            }
            for index in range(analysis_count.value)
            if analyses[index].exactly_block_diagonal
        ]
        if status == 2:
            result["reduced_affine_rows"] = 0
            result["reduced_affine_nnz"] = 0
            result["reduced_affine_blocks"] = 0
        else:
            reduced = prefos.prefos_get_reduced_problem(presolver).contents
            result["reduced_affine_rows"] = int(reduced.affine_cone_matrix.rows)
            result["reduced_affine_nnz"] = int(reduced.affine_cone_matrix.nnz)
            result["reduced_affine_blocks"] = int(reduced.n_affine_cones)
        result["rss_after_presolve_mib"] = current_rss_mib()
        result["peak_rss_mib"] = peak_rss_mib()
        return result
    finally:
        if presolver.value:
            prefos.prefos_free_presolver(presolver)


def print_summary(result):
    affine_faces = sum(
        result[name]
        for name in (
            "reduced_affine_rsoc_faces",
            "reduced_affine_psd_faces",
            "reduced_affine_exponential_faces",
            "reduced_affine_power_faces",
        )
    )
    split_shapes = ",".join(
        f"{entry['matrix_order']}->{entry['emitted_cone_blocks']}cones"
        f"({entry['components']}comp,{entry['scalar_components']}scalar,"
        f"max={entry['largest_component_order']})"
        for entry in result["splittable_psd_blocks"]
    )
    if not split_shapes:
        split_shapes = "none"
    print(
        f"{Path(result['file']).name}: "
        f"status={result['presolve_status']} "
        f"parse={result['parse_seconds']:.3f}s "
        f"create={result['create_seconds']:.3f}s "
        f"presolve={result['presolve_seconds']:.3f}s | "
        f"x={result['variables_original']:,}->{result['variables_reduced']:,} "
        f"affine_rows={result['affine_rows']:,}->{result['reduced_affine_rows']:,} "
        f"nnzG={result['affine_nnz']:,}->{result['reduced_affine_nnz']:,} | "
        f"blocks={result['affine_blocks']} psd_order={result['max_psd_order']} "
        f"largest_component={result['affine_psd_largest_component_order']} "
        f"fixed={result['fixed_box_variables']:,} "
        f"faces={affine_faces:,} "
        f"coords_removed={result['removed_affine_cone_coordinates']:,} "
        f"psd_components={result['affine_psd_connected_components']:,} "
        f"psd_split={result['decomposed_affine_psd_blocks']:,} "
        f"split_shapes={split_shapes} "
        f"cross_removed={result['removed_affine_psd_cross_coordinates']:,} "
        f"psd_skips={result['affine_psd_budget_skips']:,} "
        f"cone={result['cone_propagation_milliseconds']:.1f}ms "
        f"affine_face={result['affine_face_reduction_milliseconds']:.1f}ms "
        f"psd_structure={result['affine_psd_structure_milliseconds']:.1f}ms "
        f"rss={result['rss_after_presolve_mib']:.1f}MiB"
    )


def expand_inputs(inputs):
    files = []
    for name in inputs:
        path = Path(name).resolve()
        if path.is_dir():
            files.extend(sorted(path.glob("*.dat-s")))
        else:
            files.append(path)
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+")
    parser.add_argument(
        "--library", default="build/libPreFOS.so", type=Path
    )
    parser.add_argument("--jsonl", type=Path)
    parser.add_argument("--disable-cone-propagation", action="store_true")
    parser.add_argument("--enable-psd-higher-order", action="store_true")
    parser.add_argument("--max-cone-rounds", type=int, default=2)
    parser.add_argument("--affine-psd-work-ratio", type=float)
    parser.add_argument("--disable-psd-block-decomposition", action="store_true")
    args = parser.parse_args()

    prefos = configure_prefos(args.library.resolve())
    overrides = {
        "cone_propagation": int(not args.disable_cone_propagation),
        "max_cone_propagation_rounds": args.max_cone_rounds,
        "psd_higher_order_propagation": int(args.enable_psd_higher_order),
        "psd_block_decomposition": int(
            not args.disable_psd_block_decomposition
        ),
    }
    if args.affine_psd_work_ratio is not None:
        overrides["affine_psd_propagation_max_work_ratio"] = (
            args.affine_psd_work_ratio
        )
    output = args.jsonl.open("a", encoding="utf-8") if args.jsonl else None
    try:
        for path in expand_inputs(args.inputs):
            try:
                result = benchmark_file(prefos, path, overrides)
                print_summary(result)
                if output:
                    output.write(json.dumps(result, sort_keys=True) + "\n")
                    output.flush()
            except Exception as error:
                print(f"{path.name}: ERROR: {error}")
    finally:
        if output:
            output.close()


if __name__ == "__main__":
    main()
