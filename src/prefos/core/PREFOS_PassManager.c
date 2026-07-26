/*
 * Copyright 2026 Hongpei Li
 * SPDX-License-Identifier: Apache-2.0
 */

#include "PREFOS_PassManager.h"

#include "PREFOS_Timer.h"
#include "explorers/PREFOS_ColumnReductionInternal.h"
#include "explorers/PREFOS_ColumnReductions.h"
#include "explorers/PREFOS_TrivialReductions.h"

#include <stdio.h>

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
    PreFOSColumnWorkspace *workspace, size_t *fixed_cursor)
{
    while (*fixed_cursor < presolver->n_fixed_columns)
    {
        int column = presolver->fixed_column_log[(*fixed_cursor)++];
        int position;
        for (position = workspace->starts[column];
             position < workspace->ends[column]; ++position)
        {
            int row = workspace->rows[position];
            if (presolver->remove_rows[row])
                continue;
            if (workspace->dirty_row[row]) continue;
            if (workspace->row_degrees[row] > 0)
                --workspace->row_degrees[row];
            if (workspace->row_degrees[row] <= 1)
                prefos_internal_queue_trivial_row(
                    presolver, workspace, row);
        }
    }
    return PREFOS_STATUS_OK;
}

static PreFOSStatus collect_initial_trivial_rows(
    const PreFOSPresolver *presolver,
    PreFOSColumnWorkspace *workspace)
{
    const PreFOSCsrMatrix *matrix = &presolver->original.A;
    size_t row;
    for (row = 0; row < presolver->original.A.rows; ++row)
    {
        size_t live = 0;
        int position;
        if (presolver->remove_rows[row] ||
            workspace->trivial_row_queued[row])
            continue;
        if (!workspace->dirty_row[row])
        {
            if (workspace->row_degrees[row] <= 1)
                prefos_internal_queue_trivial_row(
                    presolver, workspace, (int) row);
            continue;
        }
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
        prefos_internal_queue_trivial_row(
            presolver, workspace, (int) row);
    }
    return PREFOS_STATUS_OK;
}

static void seed_one_sided_singleton_candidates(
    const PreFOSPresolver *presolver,
    PreFOSColumnWorkspace *workspace)
{
    size_t column;
    if (workspace->one_sided_singletons_seeded) return;
    for (column = 0; column < presolver->original.n; ++column)
        if (workspace->live_degrees[column] == 1)
            prefos_internal_queue_singleton_column(
                presolver, workspace, (int) column);
    workspace->one_sided_singletons_seeded = 1;
}

PreFOSStatus prefos_internal_run_fast_fixed_point(
    PreFOSPresolver *presolver, int allow_one_sided_singletons,
    int full_trivial_scan, PreFOSColumnWorkspace *shared_workspace)
{
    PreFOSColumnWorkspace local_workspace;
    PreFOSColumnWorkspace *workspace =
        shared_workspace ? shared_workspace : &local_workspace;
    size_t fixed_cursor =
        shared_workspace
            ? shared_workspace->fixed_column_cursor
            : presolver->n_fixed_columns;
    int reuse_workspace =
        shared_workspace != NULL ||
        (!presolver->settings.structural_reductions_gpu &&
         presolver->original.n_box > 0);
    int owns_workspace = reuse_workspace && !shared_workspace;
    int trace = getenv("PREFOS_TRACE_FAST_FIXED_POINT") != NULL;
    int round;
    ++presolver->stats.fast_fixed_point_passes;
    memset(&local_workspace, 0, sizeof(local_workspace));
    if (reuse_workspace)
    {
        PreFOSTimestamp start, stop;
        PreFOSStatus status;
        prefos_internal_timer_now(&start);
        if (shared_workspace)
        {
            if (getenv("PREFOS_FORCE_FULL_COLUMN_RESEED"))
                status = prefos_internal_prepare_column_workspace(
                    presolver, workspace);
            else
            {
                int objective_dirty;
                prefos_internal_update_column_live_degrees(
                    presolver, workspace);
                objective_dirty =
                    prefos_internal_queue_transformation_events(
                        presolver, workspace);
                if (workspace->fast_pass_started)
                    prefos_internal_queue_bound_changed_singletons(
                        presolver, workspace);
                else
                    workspace->fast_pass_started = 1;
                status = objective_dirty
                             ? prefos_internal_rebuild_column_objective(
                                   presolver, workspace)
                             : prefos_internal_sync_column_objective(
                                   presolver, workspace);
            }
        }
        else
            status = prefos_internal_build_column_workspace(
                presolver, workspace);
        prefos_internal_timer_now(&stop);
        presolver->stats.structural_reduction_milliseconds +=
            prefos_internal_timer_elapsed_milliseconds(&start, &stop);
        if (status != PREFOS_STATUS_OK) return status;
    }
    if (shared_workspace && allow_one_sided_singletons)
        seed_one_sided_singleton_candidates(
            presolver, workspace);
    if (trace)
        fprintf(
            stderr,
            "PreFOS fast pass=%zu allow_one_sided=%d "
            "singleton_queue=%zu dual_queue=%zu fixed=%zu "
            "substituted=%zu removed_rows=%zu removed_cursor=%zu\n",
            presolver->stats.fast_fixed_point_passes,
            allow_one_sided_singletons,
            reuse_workspace
                ? workspace->n_singleton_candidate_columns
                : 0,
            reuse_workspace
                ? workspace->n_dual_candidate_columns
                : 0,
            presolver->n_fixed_columns,
            presolver->stats.substituted_free_variables,
            presolver->n_removed_rows,
            reuse_workspace
                ? workspace->removed_row_cursor
                : 0);
    for (round = 0; round < PREFOS_MAX_FAST_FIXED_POINT_ROUNDS; ++round)
    {
        PreFOSFastProgress before = capture_progress(presolver);
        PreFOSFastProgress after;
        size_t fixed_before_columns = 0, fixed_after_rows = 0;
        size_t trivial_candidate_count = 0;
        double trivial_candidate_milliseconds = 0.0;
        PreFOSTimestamp reduction_start, reduction_stop;
        PreFOSStatus status;
        prefos_internal_timer_now(&reduction_start);
        status = prefos_internal_find_fixed_box_variables(
            presolver, reuse_workspace ? workspace : NULL,
            &fixed_before_columns);
        prefos_internal_timer_now(&reduction_stop);
        presolver->stats.fixed_box_scan_milliseconds +=
            prefos_internal_timer_elapsed_milliseconds(
                &reduction_start, &reduction_stop);
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
                size_t candidate_count, candidate;
                status = collect_newly_fixed_rows(
                    presolver, workspace, &fixed_cursor);
                workspace->fixed_column_cursor = fixed_cursor;
                if (status == PREFOS_STATUS_OK && round == 0 &&
                    full_trivial_scan)
                    status = collect_initial_trivial_rows(
                        presolver, workspace);
                candidate_count =
                    workspace->n_trivial_candidate_rows;
                trivial_candidate_count = candidate_count;
                if (status == PREFOS_STATUS_OK)
                {
                    prefos_internal_timer_now(&reduction_start);
                    status = prefos_internal_reduce_trivial_row_candidates(
                        presolver, workspace->trivial_candidate_rows,
                        candidate_count);
                    prefos_internal_timer_now(&reduction_stop);
                    trivial_candidate_milliseconds =
                        prefos_internal_timer_elapsed_milliseconds(
                            &reduction_start, &reduction_stop);
                    presolver->stats.trivial_candidate_milliseconds +=
                        trivial_candidate_milliseconds;
                }
                for (candidate = 0; candidate < candidate_count; ++candidate)
                    workspace->trivial_row_queued[
                        workspace->trivial_candidate_rows[candidate]] = 0;
                workspace->n_trivial_candidate_rows = 0;
            }
        }
        if (status == PREFOS_STATUS_OK)
        {
            prefos_internal_timer_now(&reduction_start);
            status = prefos_internal_find_fixed_box_variables(
                presolver, reuse_workspace ? workspace : NULL,
                &fixed_after_rows);
            prefos_internal_timer_now(&reduction_stop);
            presolver->stats.fixed_box_scan_milliseconds +=
                prefos_internal_timer_elapsed_milliseconds(
                    &reduction_start, &reduction_stop);
        }
        if (status != PREFOS_STATUS_OK)
        {
            if (owns_workspace)
                prefos_internal_free_column_workspace(workspace);
            return status;
        }
        if (reuse_workspace)
            (void) prefos_internal_queue_transformation_events(
                presolver, workspace);
        after = capture_progress(presolver);
        if (trace)
            fprintf(
                stderr,
                "PreFOS fast pass=%zu round=%d fixed=%zu "
                "substituted=%zu removed_rows=%zu "
                "trivial_candidates=%zu trivial_ms=%.3f "
                "singleton_queue=%zu "
                "dual_queue=%zu changed=%d "
                "singleton_examined=%zu singleton_terms=%zu\n",
                presolver->stats.fast_fixed_point_passes, round,
                presolver->n_fixed_columns,
                presolver->stats.substituted_free_variables,
                presolver->n_removed_rows,
                trivial_candidate_count,
                trivial_candidate_milliseconds,
                reuse_workspace
                    ? workspace->n_singleton_candidate_columns
                    : 0,
                reuse_workspace
                    ? workspace->n_dual_candidate_columns
                    : 0,
                progress_changed(before, after) ||
                    fixed_before_columns != 0 ||
                    fixed_after_rows != 0,
                presolver->stats.singleton_candidates_examined,
                presolver->stats.singleton_row_terms_scanned);
        if (!progress_changed(before, after) &&
            fixed_before_columns == 0 && fixed_after_rows == 0)
            break;
    }
    if (reuse_workspace)
        workspace->fixed_column_cursor = fixed_cursor;
    if (owns_workspace)
        prefos_internal_free_column_workspace(workspace);
    return PREFOS_STATUS_OK;
}
