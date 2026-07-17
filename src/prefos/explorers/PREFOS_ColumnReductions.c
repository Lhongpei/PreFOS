/*
 * Copyright 2026 Hongpei Li
 * SPDX-License-Identifier: Apache-2.0
 */

#include "PREFOS_ColumnReductions.h"
#include "PREFOS_ColumnReductionInternal.h"
#include "core/PREFOS_Timer.h"

PreFOSStatus prefos_internal_reduce_linear_columns(PreFOSPresolver *presolver)
{
    PreFOSColumnWorkspace workspace;
    PreFOSTimestamp start, stop;
    PreFOSStatus status;
    prefos_internal_timer_now(&start);
    status = prefos_internal_build_column_workspace(presolver, &workspace);
    if (status != PREFOS_STATUS_OK) return status;
    status = prefos_internal_populate_gpu_column_stats(presolver, &workspace);
    if (status == PREFOS_STATUS_OK)
        status = prefos_internal_reduce_empty_and_dual_fixed_columns(
            presolver, &workspace);
    if (status == PREFOS_STATUS_OK)
        status = prefos_internal_reduce_singleton_columns(
            presolver, &workspace, 0);
    if (status == PREFOS_STATUS_OK)
        status = prefos_internal_reduce_bounded_doubletons(
            presolver, &workspace);
    prefos_internal_free_column_workspace(&workspace);
    prefos_internal_timer_now(&stop);
    presolver->stats.structural_reduction_milliseconds +=
        prefos_internal_timer_elapsed_milliseconds(&start, &stop);
    return status;
}

PreFOSStatus prefos_internal_reduce_parallel_columns(PreFOSPresolver *presolver)
{
    PreFOSColumnWorkspace workspace;
    PreFOSTimestamp start, stop;
    PreFOSStatus status;
    prefos_internal_timer_now(&start);
    status = prefos_internal_build_column_workspace(presolver, &workspace);
    if (status == PREFOS_STATUS_OK)
    {
        status = prefos_internal_reduce_singleton_columns(
            presolver, &workspace, 1);
        prefos_internal_free_column_workspace(&workspace);
    }
    if (status == PREFOS_STATUS_OK)
        status = prefos_internal_reduce_parallel_column_groups(presolver);
    prefos_internal_timer_now(&stop);
    presolver->stats.structural_reduction_milliseconds +=
        prefos_internal_timer_elapsed_milliseconds(&start, &stop);
    return status;
}
