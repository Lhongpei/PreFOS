#!/usr/bin/env python3
# Copyright 2026 Hongpei Li
# SPDX-License-Identifier: Apache-2.0

"""Benchmark PreFOS on continuous CBF exponential and power cone models."""

import argparse
import ctypes as ct
import json
import time
from pathlib import Path

from benchmark_cbf_prefos import configure_prefos
from benchmark_clarabel_gurobi import clean_result, run_prefos, solve_clarabel
from cbf_scalar import load_scalar_cbf


def expand_inputs(inputs):
    files = []
    for name in inputs:
        path = Path(name).resolve()
        if path.is_dir():
            files.extend(sorted(path.glob("*.cbf")))
            files.extend(sorted(path.glob("*.cbf.gz")))
        else:
            files.append(path)
    return files


def solve_without_presolve(model, time_limit):
    result = solve_clarabel(model, False, time_limit)
    del result["x"]
    return result


def run_file(prefos, filename, time_limit, skip_baseline, skip_ablation):
    start = time.perf_counter()
    model, metadata = load_scalar_cbf(filename)
    load_seconds = time.perf_counter() - start
    result = {
        "file": str(filename),
        "load_seconds": load_seconds,
        "clarabel_presolve": False,
        **metadata,
    }
    if not skip_baseline:
        result["clarabel_original"] = solve_without_presolve(model, time_limit)
    if not skip_ablation:
        result["prefos_without_cone_propagation"] = run_prefos(
            prefos,
            model,
            time_limit,
            setting_overrides={"cone_propagation": 0},
        )
        result["prefos_without_linear_propagation"] = run_prefos(
            prefos,
            model,
            time_limit,
            setting_overrides={"linear_propagation": 0},
        )
        result["prefos_without_cone_aware_activity"] = run_prefos(
            prefos,
            model,
            time_limit,
            setting_overrides={"cone_aware_row_activity": 0},
        )
        if metadata["exponential_cones"]:
            result["prefos_without_exponential_propagation"] = run_prefos(
                prefos,
                model,
                time_limit,
                setting_overrides={"exponential_propagation": 0},
            )
        if metadata["power_cones"]:
            result["prefos_without_power_propagation"] = run_prefos(
                prefos,
                model,
                time_limit,
                setting_overrides={"power_propagation": 0},
            )
    result["prefos_full"] = run_prefos(prefos, model, time_limit)
    return clean_result(result)


def pipeline_text(label, result):
    presolve = result["create_seconds"] + result["presolve_seconds"]
    solve = result["clarabel"]["setup_seconds"] + result["clarabel"]["solve_seconds"]
    return (
        f"{label}={result['total_seconds']:.4f}s "
        f"(pre={presolve:.4f}, solve={solve:.4f}, "
        f"status={result['clarabel']['status']}, it={result['clarabel']['iterations']})"
    )


def print_result(result):
    name = Path(result["file"]).name
    cone_text = f"EXP={result['exponential_cones']:,} POW={result['power_cones']:,}"
    print(
        f"{name}: load={result['load_seconds']:.4f}s "
        f"n={result['source_variables']:,} m={result['source_constraints']:,} "
        f"{cone_text}"
    )
    baseline = result.get("clarabel_original")
    if baseline:
        print(
            f"  no-presolve={baseline['total_seconds']:.4f}s "
            f"(status={baseline['status']}, it={baseline['iterations']})"
        )
    ablation = result.get("prefos_without_cone_propagation")
    if ablation:
        print("  " + pipeline_text("cone-off", ablation))
    linear_ablation = result.get("prefos_without_linear_propagation")
    if linear_ablation:
        print("  " + pipeline_text("linear-off", linear_ablation))
    activity_ablation = result.get("prefos_without_cone_aware_activity")
    if activity_ablation:
        print("  " + pipeline_text("activity-off", activity_ablation))
    exponential_ablation = result.get("prefos_without_exponential_propagation")
    if exponential_ablation:
        print("  " + pipeline_text("exp-off", exponential_ablation))
    power_ablation = result.get("prefos_without_power_propagation")
    if power_ablation:
        print("  " + pipeline_text("power-off", power_ablation))
    full = result["prefos_full"]
    print("  " + pipeline_text("full", full))
    print(
        f"  reduced n={full['variables_original']:,}->{full['variables_reduced']:,} "
        f"m={full['rows_original']:,}->{full['rows_reduced']:,}; "
        f"bounds={full['tightened_cone_envelopes']:,} "
        f"faces={full['reduced_exponential_faces'] + full['reduced_power_faces']:,}; "
        f"postsolve={'pass' if full['postsolve_verification_passed'] else 'fail'}"
    )
    dual_support_hits = (
        full["cone_activity_lower_support_hits"]
        + full["cone_activity_upper_support_hits"]
    )
    print(
        "  rules "
        f"activity={full['cone_activity_rows']:,} rows/"
        f"{full['cone_activity_blocks']:,} blocks/"
        f"{full['cone_activity_strengthened_rows']:,} strengthened/"
        f"{full['cone_activity_rows_removed']:,} removed "
        f"(dual-support={dual_support_hits:,}); "
        f"EXP z={full['exponential_z_lower_hits']:,}/"
        f"{full['exponential_z_lower_attempts']:,}, "
        f"x={full['exponential_x_upper_hits']:,}/"
        f"{full['exponential_x_upper_attempts']:,}; "
        f"POW z={full['power_z_bound_hits']:,}/"
        f"{full['power_capacity_attempts']:,}, "
        f"axis={full['power_axis_hits']:,}/"
        f"{full['power_axis_attempts']:,} "
        f"(unbounded-capacity="
        f"{full['power_unbounded_capacity_attempts']:,}, "
        f"zero-|z|-lower="
        f"{full['power_zero_minimum_abs_z_attempts']:,})"
    )
    print(
        "  cone time "
        f"row-activity={full['redundant_row_activity_milliseconds']:.3f}ms, "
        f"prop={full['cone_propagation_milliseconds']:.3f}ms "
        f"(EXP={full['exponential_propagation_milliseconds']:.3f}ms, "
        f"POW={full['power_propagation_milliseconds']:.3f}ms), "
        f"collapse={full['cone_collapse_milliseconds']:.3f}ms"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--library", default="build/libPreFOS.so")
    parser.add_argument("--time-limit", type=float, default=60.0)
    parser.add_argument("--jsonl")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--skip-ablation", action="store_true")
    args = parser.parse_args()

    prefos = configure_prefos(Path(args.library).resolve())
    prefos.prefos_postsolve_primal.argtypes = [
        ct.c_void_p,
        ct.POINTER(ct.c_double),
        ct.POINTER(ct.c_double),
    ]
    prefos.prefos_postsolve_primal.restype = ct.c_int
    files = expand_inputs(args.inputs)
    if not files:
        raise SystemExit("no CBF files found")

    output = open(args.jsonl, "a", encoding="utf-8") if args.jsonl else None
    failures = 0
    try:
        for filename in files:
            try:
                result = run_file(
                    prefos,
                    filename,
                    args.time_limit,
                    args.skip_baseline,
                    args.skip_ablation,
                )
                print_result(result)
                if output:
                    output.write(json.dumps(result, sort_keys=True) + "\n")
                    output.flush()
            except Exception as error:
                failures += 1
                print(f"{filename.name}: FAILED: {error}")
                if output:
                    output.write(
                        json.dumps({"file": str(filename), "error": str(error)}) + "\n"
                    )
                    output.flush()
    finally:
        if output:
            output.close()
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
