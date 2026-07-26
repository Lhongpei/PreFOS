/*
 * Copyright 2026 Hongpei Li
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef PREFOS_LINEAR_PROPAGATION_H
#define PREFOS_LINEAR_PROPAGATION_H

#include "PREFOS_ColumnReductionInternal.h"

PREFOS_INTERNAL long double
prefos_internal_propagation_margin(const PreFOSPresolver *presolver,
                                long double reference);
PREFOS_INTERNAL int
prefos_internal_is_significant_improvement(const PreFOSPresolver *presolver,
                                        double current, double candidate,
                                        int is_lower);
PREFOS_INTERNAL PreFOSStatus
prefos_internal_propagate_linear_bounds(
    PreFOSPresolver *presolver,
    PreFOSColumnWorkspace *column_workspace);
PREFOS_INTERNAL PreFOSStatus
prefos_internal_probe_linear_rows(
    PreFOSPresolver *presolver, const unsigned char *probe_rows,
    int *bound_changed);
PREFOS_INTERNAL void prefos_internal_free_linear_propagation_cache(
    PreFOSPresolver *presolver);
PREFOS_INTERNAL PreFOSStatus
prefos_internal_remove_redundant_rows_by_activity(
    PreFOSPresolver *presolver,
    PreFOSColumnWorkspace *column_workspace);
PREFOS_INTERNAL PreFOSStatus prefos_internal_verify_linear_row_with_bounds(
    const PreFOSPresolver *presolver, size_t row,
    const double *lower_bounds, const double *upper_bounds);

#endif
