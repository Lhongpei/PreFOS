/*
 * Copyright 2026 Hongpei Li
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef PREFOS_MATERIALIZED_COLUMN_CLOSURE_H
#define PREFOS_MATERIALIZED_COLUMN_CLOSURE_H

#include "PREFOS_Internal.h"

typedef struct PreFOSColumnWorkspace PreFOSColumnWorkspace;

PREFOS_INTERNAL PreFOSStatus
prefos_internal_run_materialized_column_closure(
    PreFOSPresolver *presolver, size_t max_rounds,
    int remove_parallel_rows, int propagate_linear_bounds,
    size_t *pending_parallel_rows_removed,
    int defer_inserted_parallel_columns);

PREFOS_INTERNAL PreFOSStatus
prefos_internal_run_materialized_column_closure_from_workspace(
    PreFOSPresolver *presolver, size_t max_rounds,
    int remove_parallel_rows, int propagate_linear_bounds,
    size_t *pending_parallel_rows_removed,
    int defer_inserted_parallel_columns,
    PreFOSColumnWorkspace *initial_workspace);

PREFOS_INTERNAL PreFOSStatus
prefos_internal_run_materialized_trivial_closure(
    PreFOSPresolver *presolver, size_t max_rounds);

#endif
