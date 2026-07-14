#!/usr/bin/env python3
# Copyright 2026 Hongpei Li
# SPDX-License-Identifier: Apache-2.0

"""Read the scalar-cone subset of CBF v2/v3 into the PreFOS test model."""

import gzip
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from differential_conic_prefos import (
    EXPONENTIAL,
    POWER,
    RSOC,
    SOC,
    ConeSpec,
    Model,
)


@dataclass
class ScalarCbf:
    version: int = 0
    objective_sense: str = ""
    variable_count: int = 0
    variable_stacks: list[tuple[str, int]] = field(default_factory=list)
    constraint_count: int = 0
    constraint_stacks: list[tuple[str, int]] = field(default_factory=list)
    power_parameters: list[np.ndarray] = field(default_factory=list)
    dual_power_parameters: list[np.ndarray] = field(default_factory=list)
    objective_columns: list[int] = field(default_factory=list)
    objective_values: list[float] = field(default_factory=list)
    objective_offset: float = 0.0
    row_indices: list[int] = field(default_factory=list)
    column_indices: list[int] = field(default_factory=list)
    matrix_values: list[float] = field(default_factory=list)
    rhs_indices: list[int] = field(default_factory=list)
    rhs_values: list[float] = field(default_factory=list)


def _data_lines(filename):
    path = Path(filename)
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for raw_line in stream:
            line = raw_line.split("#", 1)[0].strip()
            if line:
                yield line


def _read_counted(lines, width, name):
    count = int(next(lines))
    records = []
    for _ in range(count):
        fields = next(lines).split()
        if len(fields) != width:
            raise ValueError(
                f"{name} record has {len(fields)} fields, expected {width}"
            )
        records.append(fields)
    return records


def _read_power_parameters(lines, name):
    header = next(lines).split()
    if len(header) != 2:
        raise ValueError(f"{name} header must contain two integers")
    count, total_dimension = map(int, header)
    parameters = []
    observed_dimension = 0
    for _ in range(count):
        dimension = int(next(lines))
        parameter = np.array([float(next(lines)) for _ in range(dimension)])
        if dimension == 0 or np.any(~np.isfinite(parameter)) or np.any(parameter <= 0):
            raise ValueError(f"{name} contains invalid parameters")
        parameters.append(parameter)
        observed_dimension += dimension
    if observed_dimension != total_dimension:
        raise ValueError(f"{name} parameter dimensions do not match its header")
    return parameters


def read_scalar_cbf(filename):
    """Parse continuous scalar variables and scalar affine maps from CBF."""
    data = ScalarCbf()
    lines = iter(_data_lines(filename))
    while True:
        try:
            keyword = next(lines)
        except StopIteration:
            break
        if keyword == "VER":
            data.version = int(next(lines))
        elif keyword == "POWCONES":
            data.power_parameters = _read_power_parameters(lines, keyword)
        elif keyword == "POW*CONES":
            data.dual_power_parameters = _read_power_parameters(lines, keyword)
        elif keyword == "OBJSENSE":
            data.objective_sense = next(lines)
        elif keyword in ("VAR", "CON"):
            header = next(lines).split()
            if len(header) != 2:
                raise ValueError(f"{keyword} header must contain two integers")
            dimension, stack_count = map(int, header)
            stacks = []
            for _ in range(stack_count):
                domain, stack_dimension = next(lines).split()
                stacks.append((domain, int(stack_dimension)))
            if sum(stack_dimension for _, stack_dimension in stacks) != dimension:
                raise ValueError(f"{keyword} stack dimensions do not match its header")
            if keyword == "VAR":
                data.variable_count = dimension
                data.variable_stacks = stacks
            else:
                data.constraint_count = dimension
                data.constraint_stacks = stacks
        elif keyword == "INT":
            integer_variables = _read_counted(lines, 1, keyword)
            if integer_variables:
                raise ValueError("integer CBF variables are outside the PreFOS API")
        elif keyword in ("PSDVAR", "PSDCON"):
            dimensions = _read_counted(lines, 1, keyword)
            if dimensions:
                raise ValueError(
                    f"{keyword} is not supported by the scalar CBF adapter"
                )
        elif keyword == "OBJACOORD":
            for column, value in _read_counted(lines, 2, keyword):
                data.objective_columns.append(int(column))
                data.objective_values.append(float(value))
        elif keyword == "OBJBCOORD":
            data.objective_offset = float(next(lines))
        elif keyword == "ACOORD":
            for row, column, value in _read_counted(lines, 3, keyword):
                data.row_indices.append(int(row))
                data.column_indices.append(int(column))
                data.matrix_values.append(float(value))
        elif keyword == "BCOORD":
            for row, value in _read_counted(lines, 2, keyword):
                data.rhs_indices.append(int(row))
                data.rhs_values.append(float(value))
        elif keyword in ("OBJFCOORD", "FCOORD", "HCOORD", "DCOORD"):
            widths = {"OBJFCOORD": 4, "FCOORD": 5, "HCOORD": 5, "DCOORD": 4}
            records = _read_counted(lines, widths[keyword], keyword)
            if records:
                raise ValueError(f"{keyword} requires matrix-valued CBF support")
        elif keyword == "CHANGE":
            raise ValueError("CBF hotstart sequences are not supported")
        else:
            raise ValueError(f"unsupported CBF keyword {keyword}")

    if data.version not in (2, 3):
        raise ValueError(f"unsupported CBF version {data.version}")
    if data.objective_sense not in ("MIN", "MAX"):
        raise ValueError("CBF objective sense is missing or invalid")
    if not data.variable_stacks and data.variable_count:
        raise ValueError("CBF variable structure is missing")
    if not data.constraint_stacks and data.constraint_count:
        raise ValueError("CBF constraint structure is missing")
    return data


def _power_alpha(domain, dimension, parameters, dual_parameters):
    if not (domain.startswith("@") and ":" in domain):
        return None, False
    reference, family = domain[1:].split(":", 1)
    if family not in ("POW", "POW*"):
        return None, False
    table = parameters if family == "POW" else dual_parameters
    index = int(reference)
    if index < 0 or index >= len(table):
        raise ValueError(f"power cone reference {domain} is undefined")
    parameter = table[index]
    if dimension != 3 or len(parameter) != 2:
        raise ValueError("PreFOS currently accepts only three-dimensional power cones")
    alpha = float(parameter[0] / np.sum(parameter))
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"power cone {domain} has an invalid normalized exponent")
    return alpha, family == "POW*"


def _append_domain(domain, indices, data, boxes, cones, counts):
    dimension = len(indices)
    alpha, dual_power = _power_alpha(
        domain, dimension, data.power_parameters, data.dual_power_parameters
    )
    if domain == "F":
        boxes.extend((int(index), -np.inf, np.inf) for index in indices)
    elif domain == "L+":
        boxes.extend((int(index), 0.0, np.inf) for index in indices)
    elif domain == "L-":
        boxes.extend((int(index), -np.inf, 0.0) for index in indices)
    elif domain == "L=":
        boxes.extend((int(index), 0.0, 0.0) for index in indices)
    elif domain == "Q":
        if dimension == 1:
            boxes.append((int(indices[0]), 0.0, np.inf))
        else:
            cones.append(ConeSpec(SOC, np.asarray(indices, dtype=np.int32)))
            counts["soc_cones"] += 1
    elif domain == "QR":
        if dimension == 2:
            boxes.extend((int(index), 0.0, np.inf) for index in indices)
        elif dimension >= 3:
            cones.append(ConeSpec(RSOC, np.asarray(indices, dtype=np.int32)))
            counts["rsoc_cones"] += 1
        else:
            raise ValueError(
                "a rotated quadratic cone must have dimension at least two"
            )
    elif domain == "EXP":
        if dimension != 3:
            raise ValueError("an exponential cone must have dimension three")
        # CBF stores (t, s, r), while PreFOS stores (r, s, t).
        cones.append(ConeSpec(EXPONENTIAL, np.asarray(indices[::-1], dtype=np.int32)))
        counts["exponential_cones"] += 1
    elif domain == "EXP*":
        raise ValueError("dual exponential domains need an explicit linear cone map")
    elif alpha is not None and not dual_power:
        cones.append(
            ConeSpec(POWER, np.asarray(indices, dtype=np.int32), power_alpha=alpha)
        )
        counts["power_cones"] += 1
    elif alpha is not None:
        raise ValueError("dual power domains need an explicit linear cone map")
    else:
        raise ValueError(f"unsupported scalar CBF cone {domain}")


def scalar_cbf_to_model(data):
    """Convert mixed primal-form CBF domains to the PreFOS box-and-cone form."""
    source_A = sp.coo_matrix(
        (data.matrix_values, (data.row_indices, data.column_indices)),
        shape=(data.constraint_count, data.variable_count),
        dtype=np.float64,
    ).tocsr()
    source_b = np.zeros(data.constraint_count, dtype=np.float64)
    if data.rhs_indices:
        source_b[np.asarray(data.rhs_indices, dtype=np.int64)] = data.rhs_values

    counts = {
        "soc_cones": 0,
        "rsoc_cones": 0,
        "exponential_cones": 0,
        "power_cones": 0,
    }
    boxes = []
    cones = []
    offset = 0
    for domain, dimension in data.variable_stacks:
        indices = np.arange(offset, offset + dimension, dtype=np.int32)
        _append_domain(domain, indices, data, boxes, cones, counts)
        offset += dimension

    linear_rows = []
    linear_lower = []
    linear_upper = []
    nonlinear_stacks = []
    row_offset = 0
    slack_count = 0
    for domain, dimension in data.constraint_stacks:
        rows = np.arange(row_offset, row_offset + dimension, dtype=np.int32)
        if domain == "F":
            pass
        elif domain in ("L+", "L-", "L="):
            linear_rows.extend(rows)
            lower = (
                -source_b[rows]
                if domain in ("L+", "L=")
                else np.full(dimension, -np.inf)
            )
            upper = (
                -source_b[rows]
                if domain in ("L-", "L=")
                else np.full(dimension, np.inf)
            )
            linear_lower.extend(lower)
            linear_upper.extend(upper)
        else:
            nonlinear_stacks.append((domain, rows, slack_count))
            slack_count += dimension
        row_offset += dimension

    total_variables = data.variable_count + slack_count
    linear_rows = np.asarray(linear_rows, dtype=np.int32)
    if len(linear_rows):
        linear_A = sp.hstack(
            [source_A[linear_rows], sp.csr_matrix((len(linear_rows), slack_count))],
            format="csr",
        )
    else:
        linear_A = sp.csr_matrix((0, total_variables), dtype=np.float64)

    equality_blocks = []
    equality_rhs = []
    for domain, rows, slack_offset in nonlinear_stacks:
        dimension = len(rows)
        slack_columns = data.variable_count + slack_offset + np.arange(dimension)
        slack_selector = sp.coo_matrix(
            (
                np.ones(dimension),
                (np.arange(dimension), slack_offset + np.arange(dimension)),
            ),
            shape=(dimension, slack_count),
        ).tocsr()
        equality_blocks.append(
            sp.hstack([-source_A[rows], slack_selector], format="csr")
        )
        equality_rhs.append(source_b[rows])
        _append_domain(domain, slack_columns, data, boxes, cones, counts)

    if equality_blocks:
        A = sp.vstack([linear_A, *equality_blocks], format="csr")
        lower = np.concatenate([np.asarray(linear_lower), *equality_rhs])
        upper = np.concatenate([np.asarray(linear_upper), *equality_rhs])
    else:
        A = linear_A
        lower = np.asarray(linear_lower, dtype=np.float64)
        upper = np.asarray(linear_upper, dtype=np.float64)

    c = np.zeros(total_variables, dtype=np.float64)
    if data.objective_columns:
        c[np.asarray(data.objective_columns, dtype=np.int64)] = data.objective_values
    objective_offset = data.objective_offset
    if data.objective_sense == "MAX":
        c = -c
        objective_offset = -objective_offset

    if boxes:
        box_indices, box_lower, box_upper = map(np.asarray, zip(*boxes))
    else:
        box_indices = np.empty(0)
        box_lower = np.empty(0)
        box_upper = np.empty(0)
    model = Model(
        total_variables,
        A,
        np.asarray(lower, dtype=np.float64),
        np.asarray(upper, dtype=np.float64),
        sp.csr_matrix((total_variables, total_variables), dtype=np.float64),
        sp.csr_matrix((0, total_variables), dtype=np.float64),
        np.empty(0, dtype=np.float64),
        c,
        objective_offset,
        np.asarray(box_indices, dtype=np.int32),
        np.asarray(box_lower, dtype=np.float64),
        np.asarray(box_upper, dtype=np.float64),
        cones,
    )
    metadata = {
        "cbf_version": data.version,
        "source_variables": data.variable_count,
        "source_constraints": data.constraint_count,
        "introduced_slack_variables": slack_count,
        "box_variables": len(boxes),
        "cone_variables": total_variables - len(boxes),
        **counts,
    }
    return model, metadata


def load_scalar_cbf(filename):
    return scalar_cbf_to_model(read_scalar_cbf(filename))
