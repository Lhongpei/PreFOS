# Copyright 2026 Hongpei Li
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for exporting Gurobi-presolved nonlinear cones."""

import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp


sys.path.insert(0, str(Path(__file__).resolve().parent))

gp = pytest.importorskip("gurobipy")

from benchmark_clarabel_gurobi import (  # noqa: E402
    EXPONENTIAL,
    POWER,
    run_gurobi_presolve_clarabel,
    solve_clarabel,
)
from differential_conic_prefos import ConeSpec, Model  # noqa: E402


def empty_model_parts(n):
    return (
        sp.csr_matrix((n, n)),
        sp.csr_matrix((0, n)),
        np.empty(0),
        np.empty(0, dtype=np.int32),
        np.empty(0),
        np.empty(0),
    )


def require_gurobi_license():
    try:
        model = gp.Model()
    except gp.GurobiError as error:
        pytest.skip(f"Gurobi license unavailable: {error}")
    model.dispose()


@pytest.mark.parametrize("cone_type", [EXPONENTIAL, POWER])
def test_gurobi_presolved_cone_export_matches_direct_clarabel(cone_type):
    require_gurobi_license()
    Q, R, D, box_indices, box_lower, box_upper = empty_model_parts(3)
    if cone_type == EXPONENTIAL:
        A = sp.csr_matrix([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
        lower = np.array([1.0, 0.0])
        upper = np.array([1.0, np.inf])
        objective = np.array([0.1, 0.0, 1.0])
        cone = ConeSpec(EXPONENTIAL, np.array([0, 1, 2], dtype=np.int32))
    else:
        A = sp.csr_matrix([[1.0, 1.0, 0.0]])
        lower = upper = np.array([2.0])
        objective = np.array([0.0, 0.0, -1.0])
        cone = ConeSpec(
            POWER,
            np.array([0, 1, 2], dtype=np.int32),
            power_alpha=0.3,
        )
    source = Model(
        3,
        A,
        lower,
        upper,
        Q,
        R,
        D,
        objective,
        0.0,
        box_indices,
        box_lower,
        box_upper,
        [cone],
    )

    direct = solve_clarabel(source, False, 10.0)
    exported = run_gurobi_presolve_clarabel(source, 10.0)["clarabel"]

    assert direct["status"] == "Solved"
    assert exported["status"] == "Solved"
    assert abs(direct["objective"] - exported["objective"]) <= 1e-6
