#!/usr/bin/env python3
# Copyright 2026 Hongpei Li
# SPDX-License-Identifier: Apache-2.0

"""Run a resumable, process-isolated PreFOS/PSLP LP benchmark suite."""

import argparse
import ctypes as ct
import json
import multiprocessing as mp
import os
import time
from pathlib import Path
from types import SimpleNamespace

from benchmark_cbf_prefos import PreFOSStats
from benchmark_lp_pslp import (
    PreFOSSettings,
    configure_prefos,
    configure_pslp,
    median_run,
    parse_lp,
    parse_lp_cached,
    reduction_text,
    run_prefos,
    run_pslp,
)


def prefos_arguments(feasibility_tolerance=None):
    return SimpleNamespace(
        prefos_strict=False,
        prefos_feasibility_tolerance=feasibility_tolerance,
        prefos_fixed_variable_tolerance=None,
        prefos_bound_policy=None,
        prefos_bounded_doubletons=False,
        prefos_disable_bounded_doubletons=False,
        prefos_max_bounded_doubleton_column_degree=None,
        prefos_max_aggregation_column_degree=None,
        prefos_max_aggregation_fill=None,
        prefos_max_aggregation_rounds=None,
        prefos_exhaustive_bounds=False,
        prefos_disable_free_columns=False,
        prefos_disable_linear_propagation=False,
        prefos_linear_rounds=None,
        prefos_linear_max_work_ratio=None,
        prefos_linear_min_changes_per_million=None,
        prefos_linear_max_stale_rounds=None,
        prefos_event_queue_max_average_column_degree=None,
        prefos_event_queue_activity_update_ratio=None,
        prefos_disable_cone_propagation=False,
        prefos_disable_redundant_rows=False,
        prefos_skip_parallel_rows=False,
        prefos_parallel_row_max_average_nnz=None,
        prefos_skip_redundant_row_activity=False,
        prefos_redundant_row_max_average_nnz=None,
        prefos_disable_singleton_columns=False,
        prefos_disable_parallel_columns=False,
        prefos_disable_dual_fixing=False,
        prefos_event_max_average_degree=None,
        prefos_event_update_ratio=None,
    )


def run_worker(
    connection,
    solver,
    library_path,
    data,
    repeats,
    pslp_max_time,
    prefos_feasibility_tolerance,
):
    try:
        samples = []
        if solver == "prefos":
            library = configure_prefos(library_path)
            library.prefos_default_settings.restype = PreFOSSettings
            library.prefos_get_stats.argtypes = [ct.c_void_p]
            library.prefos_get_stats.restype = ct.POINTER(PreFOSStats)
            arguments = prefos_arguments(prefos_feasibility_tolerance)
            for _ in range(repeats):
                samples.append(run_prefos(library, data, arguments))
        else:
            library = configure_pslp(library_path)
            for _ in range(repeats):
                samples.append(
                    run_pslp(
                        library,
                        data,
                        pslp_max_time,
                        False,
                        False,
                        False,
                        False,
                    )
                )
        connection.send(median_run(samples))
    except BaseException as error:
        connection.send(
            {"error": f"{type(error).__name__}: {error}"}
        )
    finally:
        connection.close()


def run_isolated(
    context,
    solver,
    library_path,
    data,
    repeats,
    timeout,
    pslp_max_time,
    prefos_feasibility_tolerance,
):
    parent, child = context.Pipe(duplex=False)
    process = context.Process(
        target=run_worker,
        args=(
            child,
            solver,
            library_path,
            data,
            repeats,
            pslp_max_time,
            prefos_feasibility_tolerance,
        ),
    )
    start = time.perf_counter()
    process.start()
    child.close()
    result = None
    received_result = False
    try:
        if parent.poll(timeout):
            result = parent.recv()
            received_result = True
        else:
            result = {
                "timeout": timeout,
                "wall_seconds": time.perf_counter() - start,
            }
    except EOFError:
        result = {
            "error": "worker exited without returning a result",
        }
    finally:
        parent.close()
        process.join(0.5)
        if process.is_alive():
            process.terminate()
            process.join(1.0)
        if process.is_alive():
            process.kill()
            process.join()
    if (
        not received_result
        and "error" not in result
        and "timeout" not in result
        and process.exitcode not in (0, None)
    ):
        result = {
            "error": f"worker exited with code {process.exitcode}",
        }
    return result


def completed_files(output_path):
    completed = set()
    if not output_path.exists():
        return completed
    with output_path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                completed.add(json.loads(line)["file"])
            except (KeyError, json.JSONDecodeError):
                continue
    return completed


def model_paths(inputs, input_dir, pattern):
    paths = list(inputs)
    if input_dir is not None:
        paths.extend(input_dir.glob(pattern))
    return sorted({path.resolve() for path in paths})


def print_solver(name, original, result):
    if "timeout" in result:
        return f"{name}=TIMEOUT({result['timeout']:.0f}s)"
    if "error" in result:
        return f"{name}=ERROR({result['error']})"
    return (
        f"{name}={result['total_seconds']:.4f}s "
        f"{reduction_text(original, result)}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="*", type=Path)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--pattern", default="*.mps.gz")
    parser.add_argument("--prefos-library", required=True, type=Path)
    parser.add_argument("--pslp-library", required=True, type=Path)
    parser.add_argument("--parser", choices=("highs", "gurobi"), default="gurobi")
    parser.add_argument("--parsed-cache-directory", type=Path)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--pslp-max-time", type=float)
    parser.add_argument(
        "--prefos-feasibility-tolerance",
        type=float,
        default=1e-6,
        help="default matches PSLP's fixed feasibility tolerance",
    )
    parser.add_argument("--jsonl", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--cpus",
        help="comma-separated Linux CPU IDs inherited by both workers",
    )
    args = parser.parse_args()
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    if args.timeout <= 0.0:
        parser.error("--timeout must be positive")
    if not args.inputs and args.input_dir is None:
        parser.error("provide inputs or --input-dir")

    args.prefos_library = args.prefos_library.resolve()
    args.pslp_library = args.pslp_library.resolve()
    if args.cpus:
        cpus = {int(value) for value in args.cpus.split(",")}
        os.sched_setaffinity(0, cpus)

    paths = model_paths(args.inputs, args.input_dir, args.pattern)
    done = completed_files(args.jsonl) if args.resume else set()
    mode = "a" if args.resume else "w"
    context = mp.get_context("fork")
    pslp_max_time = (
        args.pslp_max_time
        if args.pslp_max_time is not None
        else args.timeout
    )

    args.jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.jsonl.open(mode, encoding="utf-8") as output:
        for index, path in enumerate(paths, start=1):
            if str(path) in done:
                continue
            try:
                data = (
                    parse_lp_cached(
                        path, args.parser,
                        args.parsed_cache_directory,
                    )
                    if args.parsed_cache_directory is not None
                    else parse_lp(path, args.parser)
                )
                order = (
                    ("prefos", "pslp")
                    if index % 2
                    else ("pslp", "prefos")
                )
                samples = {}
                for solver in order:
                    library_path = (
                        args.prefos_library
                        if solver == "prefos"
                        else args.pslp_library
                    )
                    samples[solver] = run_isolated(
                        context,
                        solver,
                        library_path,
                        data,
                        args.repeats,
                        args.timeout,
                        pslp_max_time,
                        args.prefos_feasibility_tolerance,
                    )
                result = {
                    "file": str(path),
                    "parser": args.parser,
                    "parse_seconds": data["parse_seconds"],
                    "rows_original": data["rows"],
                    "columns_original": data["columns"],
                    "nnz_original": data["nnz"],
                    "integer_columns_relaxed": data["integer_columns"],
                    "dropped_general_constraints": data.get(
                        "dropped_general_constraints", 0
                    ),
                    "dropped_sos_constraints": data.get(
                        "dropped_sos_constraints", 0
                    ),
                    "prefos_feasibility_tolerance":
                        args.prefos_feasibility_tolerance,
                    "prefos": samples["prefos"],
                    "pslp": samples["pslp"],
                }
                message = " | ".join(
                    (
                        f"[{index}/{len(paths)}] {path.name}",
                        print_solver("PreFOS", data, samples["prefos"]),
                        print_solver("PSLP", data, samples["pslp"]),
                    )
                )
            except BaseException as error:
                result = {
                    "file": str(path),
                    "parser": args.parser,
                    "error": f"{type(error).__name__}: {error}",
                }
                message = (
                    f"[{index}/{len(paths)}] {path.name} | "
                    f"PARSE ERROR({result['error']})"
                )
            output.write(json.dumps(result, sort_keys=True) + "\n")
            output.flush()
            print(message, flush=True)


if __name__ == "__main__":
    main()
