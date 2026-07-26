/*
 * Copyright 2026 Hongpei Li
 * SPDX-License-Identifier: Apache-2.0
 */

#include "PREFOS_ColumnReductions.h"
#include "PREFOS_ColumnReductionInternal.h"
#include "core/PREFOS_Timer.h"

#include <stdlib.h>

#define PREFOS_INLINE_PARALLEL_CLOSURE_MAX_COLUMNS 1048576U

static int should_retry_parallel_columns_after_singletons(
    const PreFOSPresolver *presolver, size_t removed_rows_before)
{
    const char *override =
        getenv("PREFOS_INLINE_PARALLEL_COLUMN_CLOSURE");
    size_t removed_rows;
    if (presolver->n_removed_rows <= removed_rows_before)
        return 0;
    if (override && *override)
        return *override != '0';
    removed_rows =
        presolver->n_removed_rows - removed_rows_before;
    return presolver->original.n <=
               PREFOS_INLINE_PARALLEL_CLOSURE_MAX_COLUMNS &&
           removed_rows >=
               (presolver->original.A.rows + 1) / 2;
}

static void seed_one_sided_singleton_columns(
    const PreFOSPresolver *presolver,
    PreFOSColumnWorkspace *workspace)
{
    size_t column;
    if (workspace->one_sided_singletons_seeded)
        return;
    for (column = 0; column < presolver->original.n; ++column)
        if (workspace->live_degrees[column] == 1)
            prefos_internal_queue_singleton_column(
                presolver, workspace, (int) column);
    workspace->one_sided_singletons_seeded = 1;
}

PreFOSStatus prefos_internal_reduce_linear_columns_in_workspace(
    PreFOSPresolver *presolver, PreFOSColumnWorkspace *workspace,
    int allow_one_sided_singletons)
{
    PreFOSTimestamp start, stop;
    PreFOSStatus status;
    prefos_internal_timer_now(&start);
    status = prefos_internal_reduce_empty_and_dual_fixed_columns(
        presolver, workspace);
    prefos_internal_timer_now(&stop);
    presolver->stats.column_fixing_milliseconds +=
        prefos_internal_timer_elapsed_milliseconds(&start, &stop);
    if (status == PREFOS_STATUS_OK)
        status = prefos_internal_sync_column_objective(
            presolver, workspace);
    if (status == PREFOS_STATUS_OK)
    {
        prefos_internal_timer_now(&start);
        status = prefos_internal_reduce_singleton_columns(
            presolver, workspace, allow_one_sided_singletons);
        prefos_internal_timer_now(&stop);
        presolver->stats.singleton_column_milliseconds +=
            prefos_internal_timer_elapsed_milliseconds(&start, &stop);
    }
    if (status == PREFOS_STATUS_OK)
        prefos_internal_update_column_live_degrees(
            presolver, workspace);
    return status;
}

PreFOSStatus prefos_internal_reduce_linear_columns(
    PreFOSPresolver *presolver, int allow_one_sided_singletons)
{
    PreFOSColumnWorkspace workspace;
    PreFOSTimestamp start, stop;
    PreFOSStatus status;
    if (presolver->original.n_box == 0) return PREFOS_STATUS_OK;
    prefos_internal_timer_now(&start);
    status = prefos_internal_build_column_workspace(presolver, &workspace);
    if (status != PREFOS_STATUS_OK) return status;
    status = prefos_internal_populate_gpu_column_stats(presolver, &workspace);
    if (status == PREFOS_STATUS_OK)
        status = prefos_internal_reduce_linear_columns_in_workspace(
            presolver, &workspace, allow_one_sided_singletons);
    prefos_internal_free_column_workspace(&workspace);
    prefos_internal_timer_now(&stop);
    presolver->stats.structural_reduction_milliseconds +=
        prefos_internal_timer_elapsed_milliseconds(&start, &stop);
    return status;
}

PreFOSStatus prefos_internal_reduce_parallel_columns_in_workspace(
    PreFOSPresolver *presolver, PreFOSColumnWorkspace *workspace)
{
    PreFOSStatus status;
    size_t removed_rows_before_singletons;
    if (presolver->original.n_box == 0 ||
        (!presolver->settings.parallel_column_reduction &&
         !presolver->settings.singleton_column_reduction))
        return PREFOS_STATUS_OK;
    status = prefos_internal_refresh_column_workspace_incremental(
        presolver, workspace);
    if (status == PREFOS_STATUS_OK &&
        presolver->settings.parallel_column_reduction &&
        presolver->original.n >= 2)
        status = prefos_internal_reduce_parallel_column_groups(
            presolver, workspace);
    if (status == PREFOS_STATUS_OK)
        status = prefos_internal_sync_column_objective(
            presolver, workspace);
    if (status == PREFOS_STATUS_OK)
    {
        removed_rows_before_singletons =
            presolver->n_removed_rows;
        seed_one_sided_singleton_columns(
            presolver, workspace);
        status = prefos_internal_reduce_singleton_columns(
            presolver, workspace, 1);
    }
    if (status == PREFOS_STATUS_OK &&
        presolver->settings.parallel_column_reduction &&
        presolver->original.n >= 2 &&
        should_retry_parallel_columns_after_singletons(
            presolver, removed_rows_before_singletons))
    {
        status = prefos_internal_refresh_column_workspace_incremental(
            presolver, workspace);
        if (status == PREFOS_STATUS_OK)
            status = prefos_internal_reduce_parallel_column_groups(
                presolver, workspace);
    }
    return status;
}

PreFOSStatus prefos_internal_reduce_parallel_columns(PreFOSPresolver *presolver)
{
    PreFOSColumnWorkspace workspace;
    PreFOSTimestamp start, stop;
    PreFOSStatus status;
    size_t removed_rows_before_singletons;
    if (presolver->original.n_box == 0 ||
        (!presolver->settings.parallel_column_reduction &&
         !presolver->settings.singleton_column_reduction))
        return PREFOS_STATUS_OK;
    prefos_internal_timer_now(&start);
    status = prefos_internal_build_column_workspace(presolver, &workspace);
    if (status == PREFOS_STATUS_OK &&
        presolver->settings.parallel_column_reduction &&
        presolver->original.n >= 2)
        status = prefos_internal_reduce_parallel_column_groups(
            presolver, &workspace);
    if (status == PREFOS_STATUS_OK)
        status = prefos_internal_sync_column_objective(
            presolver, &workspace);
    if (status == PREFOS_STATUS_OK)
    {
        removed_rows_before_singletons =
            presolver->n_removed_rows;
        status = prefos_internal_reduce_singleton_columns(
            presolver, &workspace, 1);
    }
    if (status == PREFOS_STATUS_OK &&
        presolver->settings.parallel_column_reduction &&
        presolver->original.n >= 2 &&
        should_retry_parallel_columns_after_singletons(
            presolver, removed_rows_before_singletons))
    {
        status = prefos_internal_refresh_column_workspace_incremental(
            presolver, &workspace);
        if (status == PREFOS_STATUS_OK)
            status = prefos_internal_reduce_parallel_column_groups(
                presolver, &workspace);
    }
    prefos_internal_free_column_workspace(&workspace);
    prefos_internal_timer_now(&stop);
    presolver->stats.structural_reduction_milliseconds +=
        prefos_internal_timer_elapsed_milliseconds(&start, &stop);
    return status;
}
