/*
 * Copyright 2026 Hongpei Li
 * SPDX-License-Identifier: Apache-2.0
 */

#include "PREFOS_PassManager.h"

#include "PREFOS_Timer.h"
#include "explorers/PREFOS_ColumnReductionInternal.h"
#include "explorers/PREFOS_ColumnReductions.h"
#include "explorers/PREFOS_TrivialReductions.h"

#define PREFOS_MAX_FAST_FIXED_POINT_ROUNDS 16

typedef struct
{
    size_t removed_rows;
    size_t eliminated_columns;
    size_t tightened_rows;
    size_t transformation_events;
} PreFOSFastProgress;

static PreFOSFastProgress capture_progress(
    const PreFOSPresolver *presolver)
{
    const PreFOSStats *stats = &presolver->stats;
    PreFOSFastProgress progress;
    progress.removed_rows =
        stats->removed_redundant_rows +
        stats->removed_singleton_rows +
        stats->removed_empty_rows;
    progress.eliminated_columns =
        stats->removed_empty_columns +
        stats->dual_fixed_columns +
        stats->substituted_free_variables;
    progress.tightened_rows = stats->tightened_singleton_rows;
    progress.transformation_events = presolver->transformations.n_events;
    return progress;
}

static int progress_changed(PreFOSFastProgress before,
                            PreFOSFastProgress after)
{
    return after.removed_rows != before.removed_rows ||
           after.eliminated_columns != before.eliminated_columns ||
           after.tightened_rows != before.tightened_rows ||
           after.transformation_events != before.transformation_events;
}

static PreFOSStatus collect_newly_fixed_rows(
    const PreFOSPresolver *presolver,
    const PreFOSColumnWorkspace *workspace,
    size_t *fixed_cursor, unsigned char *row_queued,
    int *candidate_rows, size_t *candidate_count)
{
    *candidate_count = 0;
    while (*fixed_cursor < presolver->n_fixed_columns)
    {
        int column = presolver->fixed_column_log[(*fixed_cursor)++];
        int position;
        for (position = workspace->starts[column];
             position < workspace->ends[column]; ++position)
        {
            int row = workspace->rows[position];
            if (presolver->remove_rows[row] || row_queued[row])
                continue;
            if (*candidate_count >= presolver->original.A.rows)
                return PREFOS_STATUS_NUMERICAL_ERROR;
            row_queued[row] = 1;
            candidate_rows[(*candidate_count)++] = row;
        }
    }
    return PREFOS_STATUS_OK;
}

static PreFOSStatus collect_initial_trivial_rows(
    const PreFOSPresolver *presolver,
    unsigned char *row_queued, int *candidate_rows,
    size_t *candidate_count)
{
    const PreFOSCsrMatrix *matrix = &presolver->original.A;
    size_t row;
    for (row = 0; row < presolver->original.A.rows; ++row)
    {
        size_t live = 0;
        int position;
        if (presolver->remove_rows[row] || row_queued[row])
            continue;
        for (position = matrix->row_pointers[row];
             position < matrix->row_pointers[row + 1] && live <= 1;
             ++position)
        {
            int column = matrix->column_indices[position];
            if (matrix->values[position] == 0.0 ||
                presolver->is_fixed[column] ||
                presolver->is_parallel_removed[column])
                continue;
            if (presolver->is_substituted[column] &&
                !prefos_internal_term_is_active_in_row(
                    presolver, row, column))
                continue;
            ++live;
        }
        if (live > 1) continue;
        if (*candidate_count >= presolver->original.A.rows)
            return PREFOS_STATUS_NUMERICAL_ERROR;
        row_queued[row] = 1;
        candidate_rows[(*candidate_count)++] = (int) row;
    }
    return PREFOS_STATUS_OK;
}

PreFOSStatus prefos_internal_run_fast_fixed_point(
    PreFOSPresolver *presolver, int allow_one_sided_singletons,
    int full_trivial_scan, PreFOSColumnWorkspace *shared_workspace)
{
    PreFOSColumnWorkspace local_workspace;
    PreFOSColumnWorkspace *workspace =
        shared_workspace ? shared_workspace : &local_workspace;
    unsigned char *row_queued = NULL;
    int *candidate_rows = NULL;
    size_t fixed_cursor = presolver->n_fixed_columns;
    int reuse_workspace =
        shared_workspace != NULL ||
        (!presolver->settings.structural_reductions_gpu &&
         presolver->original.n_box > 0);
    int owns_workspace = reuse_workspace && !shared_workspace;
    int round;
    ++presolver->stats.fast_fixed_point_passes;
    memset(&local_workspace, 0, sizeof(local_workspace));
    if (reuse_workspace)
    {
        PreFOSTimestamp start, stop;
        PreFOSStatus status;
        prefos_internal_timer_now(&start);
        status = shared_workspace
                     ? prefos_internal_prepare_column_workspace(
                           presolver, workspace)
                     : prefos_internal_build_column_workspace(
                           presolver, workspace);
        prefos_internal_timer_now(&stop);
        presolver->stats.structural_reduction_milliseconds +=
            prefos_internal_timer_elapsed_milliseconds(&start, &stop);
        if (status != PREFOS_STATUS_OK) return status;
        row_queued = (unsigned char *) calloc(
            presolver->original.A.rows, sizeof(unsigned char));
        candidate_rows = (int *) prefos_internal_alloc_array(
            presolver->original.A.rows, sizeof(int));
        if (presolver->original.A.rows > 0 &&
            (!row_queued || !candidate_rows))
        {
            free(row_queued);
            free(candidate_rows);
            if (owns_workspace)
                prefos_internal_free_column_workspace(workspace);
            return PREFOS_STATUS_OUT_OF_MEMORY;
        }
    }
    for (round = 0; round < PREFOS_MAX_FAST_FIXED_POINT_ROUNDS; ++round)
    {
        PreFOSFastProgress before = capture_progress(presolver);
        PreFOSFastProgress after;
        size_t fixed_before_columns = 0, fixed_after_rows = 0;
        PreFOSStatus status =
            prefos_internal_find_fixed_box_variables(
                presolver, &fixed_before_columns);
        ++presolver->stats.fast_fixed_point_rounds;
        if (status == PREFOS_STATUS_OK)
        {
            if (reuse_workspace)
            {
                PreFOSTimestamp start, stop;
                prefos_internal_timer_now(&start);
                if (round > 0)
                    prefos_internal_update_column_live_degrees(
                        presolver, workspace);
                status =
                    prefos_internal_reduce_linear_columns_in_workspace(
                        presolver, workspace,
                        allow_one_sided_singletons);
                prefos_internal_timer_now(&stop);
                presolver->stats.structural_reduction_milliseconds +=
                    prefos_internal_timer_elapsed_milliseconds(
                        &start, &stop);
            }
            else
                status = prefos_internal_reduce_linear_columns(
                    presolver, allow_one_sided_singletons);
        }
        if (status == PREFOS_STATUS_OK)
        {
            if (!reuse_workspace)
            {
                status = prefos_internal_reduce_trivial_rows(presolver);
            }
            else
            {
                size_t candidate_count = 0, candidate;
                status = collect_newly_fixed_rows(
                    presolver, workspace, &fixed_cursor, row_queued,
                    candidate_rows, &candidate_count);
                if (status == PREFOS_STATUS_OK && round == 0 &&
                    full_trivial_scan)
                    status = collect_initial_trivial_rows(
                        presolver, row_queued,
                        candidate_rows, &candidate_count);
                if (status == PREFOS_STATUS_OK)
                    status = prefos_internal_reduce_trivial_row_candidates(
                        presolver, candidate_rows, candidate_count);
                for (candidate = 0; candidate < candidate_count; ++candidate)
                    row_queued[candidate_rows[candidate]] = 0;
            }
        }
        if (status == PREFOS_STATUS_OK)
            status = prefos_internal_find_fixed_box_variables(
                presolver, &fixed_after_rows);
        if (status != PREFOS_STATUS_OK)
        {
            free(row_queued);
            free(candidate_rows);
            if (owns_workspace)
                prefos_internal_free_column_workspace(workspace);
            return status;
        }
        after = capture_progress(presolver);
        if (!progress_changed(before, after) &&
            fixed_before_columns == 0 && fixed_after_rows == 0)
            break;
    }
    free(row_queued);
    free(candidate_rows);
    if (owns_workspace)
        prefos_internal_free_column_workspace(workspace);
    return PREFOS_STATUS_OK;
}
