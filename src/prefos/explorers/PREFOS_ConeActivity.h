/*
 * Copyright 2026 Hongpei Li
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef PREFOS_CONE_ACTIVITY_H
#define PREFOS_CONE_ACTIVITY_H

#include "PREFOS_Internal.h"
#include "LinearPropagationKernel.h"

typedef struct
{
    int *column_to_cone;
    double *coefficients;
    int *touched_cones;
    unsigned char *cone_touched;
    const double *lower_bounds;
    const double *upper_bounds;
    int row_support_strengthened;
} PreFOSConeActivityWorkspace;

PREFOS_INTERNAL PreFOSStatus prefos_internal_cone_activity_workspace_init(
    const PreFOSPresolver *presolver, PreFOSConeActivityWorkspace *workspace);
PREFOS_INTERNAL void
prefos_internal_cone_activity_workspace_free(PreFOSConeActivityWorkspace *workspace);
PREFOS_INTERNAL PreFOSStatus prefos_internal_compute_cone_aware_row_activity(
    PreFOSPresolver *presolver, size_t row, int outward,
    int count_statistics, PreFOSConeActivityWorkspace *workspace,
    PresolveLinearActivity *activity);

#endif
