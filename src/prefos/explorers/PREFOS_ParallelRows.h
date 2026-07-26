/*
 * Copyright 2026 Hongpei Li
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef PREFOS_PARALLEL_ROWS_H
#define PREFOS_PARALLEL_ROWS_H

#include "PREFOS_Internal.h"

struct PreFOSColumnWorkspace;

enum
{
    PREFOS_MATERIALIZED_SUPPORT_TRIVIAL = 1,
    PREFOS_MATERIALIZED_SUPPORT_PARALLEL = 2,
    PREFOS_MATERIALIZED_SUPPORT_BOUNDED_DOUBLETON = 4
};

PREFOS_INTERNAL PreFOSStatus
prefos_internal_remove_parallel_rows(
    PreFOSPresolver *presolver,
    struct PreFOSColumnWorkspace *column_workspace);

PREFOS_INTERNAL PreFOSStatus
prefos_internal_remove_parallel_rows_in_working_matrix(
    PreFOSPresolver *presolver, const PreFOSCsrMatrix *matrix,
    double *lower, double *upper,
    struct PreFOSColumnWorkspace *column_workspace);

PREFOS_INTERNAL int
prefos_internal_filtered_parallel_support_is_promising(
    const PreFOSPresolver *presolver);

PREFOS_INTERNAL int
prefos_internal_materialized_support_opportunities(
    const PreFOSPresolver *presolver,
    const struct PreFOSColumnWorkspace *workspace);

#endif
