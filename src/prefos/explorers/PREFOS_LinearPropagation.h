/*
 * Copyright 2026 Hongpei Li
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef PREFOS_LINEAR_PROPAGATION_H
#define PREFOS_LINEAR_PROPAGATION_H

#include "PREFOS_Internal.h"

PREFOS_INTERNAL long double
prefos_internal_propagation_margin(const PreFOSPresolver *presolver,
                                long double reference);
PREFOS_INTERNAL int
prefos_internal_is_significant_improvement(const PreFOSPresolver *presolver,
                                        double current, double candidate,
                                        int is_lower);
PREFOS_INTERNAL PreFOSStatus
prefos_internal_propagate_linear_bounds(PreFOSPresolver *presolver);
PREFOS_INTERNAL PreFOSStatus
prefos_internal_remove_redundant_rows_by_activity(PreFOSPresolver *presolver);
PREFOS_INTERNAL PreFOSStatus prefos_internal_verify_linear_row_with_bounds(
    const PreFOSPresolver *presolver, size_t row,
    const double *lower_bounds, const double *upper_bounds);

#endif
