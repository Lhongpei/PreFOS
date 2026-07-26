/*
 * Copyright 2026 Hongpei Li
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef PREFOS_MATERIALIZED_DOUBLETONS_H
#define PREFOS_MATERIALIZED_DOUBLETONS_H

#include "PREFOS_ColumnReductionInternal.h"

PREFOS_INTERNAL int prefos_internal_can_update_materialized_doubleton(
    const PreFOSPresolver *presolver,
    const PreFOSColumnWorkspace *workspace, int source_row,
    int pivot_column, int target_column, double alpha, double beta);

PREFOS_INTERNAL PreFOSStatus
prefos_internal_update_materialized_doubleton(
    PreFOSPresolver *presolver, PreFOSColumnWorkspace *workspace,
    int source_row, int pivot_column, int target_column,
    double alpha, double beta, size_t event_count_before);

#endif
