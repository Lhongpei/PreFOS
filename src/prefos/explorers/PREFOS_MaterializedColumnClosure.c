/*
 * Copyright 2026 Hongpei Li
 * SPDX-License-Identifier: Apache-2.0
 */

#include "PREFOS_MaterializedColumnClosure.h"

#include "PREFOS_ColumnReductionInternal.h"
#include "PREFOS_ColumnReductions.h"
#include "PREFOS_CudaBackend.h"
#include "PREFOS_LinearPropagation.h"
#include "PREFOS_ParallelRows.h"
#include "PREFOS_TrivialReductions.h"
#include "core/PREFOS_PassManager.h"
#include "core/PREFOS_Timer.h"
#include "core/PREFOS_WorkingMatrix.h"

#include <stdio.h>
#include <stdlib.h>

static void trace_materialized_workspace_column(
    const char *stage, const PreFOSColumnWorkspace *workspace)
{
    const char *value =
        getenv("PREFOS_TRACE_MATERIALIZED_COLUMN");
    int column;
    if (!value || !*value) return;
    column = atoi(value);
    if (column < 0) return;
    fprintf(
        stderr,
        "PreFOS materialized column stage=%s column=%d degree=%d "
        "down=%d up=%d objective=%.17g\n",
        stage, column, workspace->live_degrees[column],
        workspace->down_locks[column],
        workspace->up_locks[column],
        workspace->objective[column]);
}

static void trace_materialized_row(
    const char *stage, const PreFOSPresolver *presolver,
    const PreFOSCsrMatrix *matrix, const double *lower,
    const double *upper)
{
    const char *value = getenv("PREFOS_TRACE_MATERIALIZED_ROW");
    char *end;
    long selected;
    int position;
    if (!value || !*value) return;
    selected = strtol(value, &end, 10);
    if (end == value || selected < 0 ||
        (size_t) selected >= matrix->rows)
        return;
    fprintf(
        stderr,
        "PreFOS materialized row stage=%s row=%ld removed=%d "
        "bounds=[%.17g,%.17g] terms=",
        stage, selected, presolver->remove_rows[selected],
        lower[selected], upper[selected]);
    for (position = matrix->row_pointers[selected];
         position < matrix->row_pointers[selected + 1]; ++position)
        fprintf(
            stderr, "%s%.17g*x%d",
            position == matrix->row_pointers[selected] ? "" : ",",
            matrix->values[position], matrix->column_indices[position]);
    fputc('\n', stderr);
}

static void trace_materialized_singletons(
    const char *stage, const PreFOSPresolver *presolver,
    const PreFOSColumnWorkspace *workspace)
{
    size_t column;
    size_t degree_one = 0;
    size_t queued = 0;
    size_t no_box = 0;
    size_t fixed = 0;
    size_t substituted = 0;
    size_t parallel_removed = 0;
    size_t fill_target = 0;
    size_t affine_protected = 0;
    size_t quadratic = 0;
    size_t factor = 0;
    size_t protected_target = 0;
    size_t depth_limited = 0;
    size_t dirty = 0;
    size_t eligible = 0;
    const char *trace = getenv("PREFOS_TRACE_MATERIALIZED_STAGES");
    if (!trace || !*trace || *trace == '0') return;
    for (column = 0; column < presolver->original.n; ++column)
    {
        int p;
        int row = -1;
        if (workspace->live_degrees[column] != 1) continue;
        ++degree_one;
        if (workspace->singleton_column_queued[column]) ++queued;
        if (presolver->variable_to_box[column] < 0)
        {
            ++no_box;
            continue;
        }
        if (presolver->is_fixed[column])
        {
            ++fixed;
            continue;
        }
        if (presolver->is_substituted[column])
        {
            ++substituted;
            continue;
        }
        if (presolver->is_parallel_removed[column])
        {
            ++parallel_removed;
            continue;
        }
        if (presolver->substitution_fill_in_targets[column])
        {
            ++fill_target;
            continue;
        }
        if (presolver->affine_protected_columns[column])
        {
            ++affine_protected;
            continue;
        }
        if (workspace->quadratic[column])
        {
            ++quadratic;
            continue;
        }
        if (workspace->factor[column])
        {
            ++factor;
            continue;
        }
        if (workspace->protected_target[column])
        {
            ++protected_target;
            continue;
        }
        if (presolver->substitution_incoming_depth[column] >=
            PREFOS_MAX_SUBSTITUTION_DEPTH)
        {
            ++depth_limited;
            continue;
        }
        for (p = workspace->starts[column];
             p < workspace->ends[column]; ++p)
            if (!presolver->remove_rows[workspace->rows[p]])
            {
                row = workspace->rows[p];
                break;
            }
        if (row < 0 || workspace->dirty_row[row])
        {
            ++dirty;
            continue;
        }
        ++eligible;
    }
    fprintf(
        stderr,
        "PreFOS closure singleton stage=%s degree_one=%zu queued=%zu "
        "eligible=%zu no_box=%zu fixed=%zu substituted=%zu "
        "parallel_removed=%zu fill_target=%zu affine=%zu quadratic=%zu "
        "factor=%zu protected=%zu depth=%zu dirty=%zu "
        "queue=[%zu,%zu)\n",
        stage, degree_one, queued, eligible, no_box, fixed, substituted,
        parallel_removed, fill_target, affine_protected, quadratic, factor,
        protected_target, depth_limited, dirty,
        workspace->singleton_candidate_position,
        workspace->n_singleton_candidate_columns);
}

static void trace_materialized_closure_failure(
    size_t round, const char *stage, PreFOSStatus status)
{
    const char *trace = getenv("PREFOS_TRACE_MATERIALIZED_CLOSURE");
    if (!trace || !*trace || *trace == '0' ||
        status == PREFOS_STATUS_OK)
        return;
    fprintf(
        stderr,
        "PreFOS materialized closure failure round=%zu stage=%s status=%d\n",
        round, stage, (int) status);
}

static void trace_materialized_closure_round(
    size_t round, size_t nnz, size_t substitutions,
    size_t parallel_columns, size_t fixed_columns, size_t removed_rows,
    double workspace_milliseconds, double reduction_milliseconds)
{
    const char *trace = getenv("PREFOS_TRACE_MATERIALIZED_CLOSURE");
    if (!trace || !*trace || *trace == '0') return;
    fprintf(
        stderr,
        "PreFOS materialized closure round=%zu nnz=%zu substitutions=%zu "
        "parallel_columns=%zu fixed_columns=%zu removed_rows=%zu "
        "workspace_ms=%.3f reduction_ms=%.3f\n",
        round, nnz, substitutions, parallel_columns, fixed_columns,
        removed_rows, workspace_milliseconds, reduction_milliseconds);
}

static void trace_materialized_closure_materialization(
    size_t round, size_t nnz, double milliseconds)
{
    const char *trace = getenv("PREFOS_TRACE_MATERIALIZED_CLOSURE");
    if (!trace || !*trace || *trace == '0') return;
    fprintf(
        stderr,
        "PreFOS materialized closure materialize_after=%zu nnz=%zu ms=%.3f\n",
        round, nnz, milliseconds);
}

static void compact_exact_zeros(PreFOSCsrMatrix *matrix)
{
    size_t row, write = 0;
    int read = 0;

    if (!matrix || !matrix->row_pointers) return;
    for (row = 0; row < matrix->rows; ++row)
    {
        int end = matrix->row_pointers[row + 1];
        matrix->row_pointers[row] = (int) write;
        while (read < end)
        {
            if (matrix->values[read] != 0.0)
            {
                matrix->values[write] = matrix->values[read];
                matrix->column_indices[write] =
                    matrix->column_indices[read];
                ++write;
            }
            ++read;
        }
    }
    matrix->row_pointers[matrix->rows] = (int) write;
    matrix->nnz = write;
}

static int materialized_closure_round_is_stale(
    size_t nnz, size_t substitutions, size_t parallel_columns,
    size_t fixed_columns, size_t removed_rows)
{
    size_t changes, minimum_yield;
    changes = substitutions;
    if (parallel_columns > SIZE_MAX - changes) return 0;
    changes += parallel_columns;
    if (fixed_columns > SIZE_MAX - changes) return 0;
    changes += fixed_columns;
    if (removed_rows > SIZE_MAX - changes) return 0;
    changes += removed_rows;
    /*
     * Rebuilding a multi-million-entry materialized matrix and its CSC is
     * expensive. Require a denser first-wave payoff before buying another
     * round; small models retain the more exhaustive threshold.
     */
    minimum_yield =
        nnz >= 1000000 ? (nnz + 511) / 512
                       : (nnz + 2047) / 2048;
    return changes < minimum_yield;
}

static int materialized_parallel_recheck_is_due(
    size_t nnz, size_t last_checked_nnz)
{
    size_t difference =
        nnz > last_checked_nnz ? nnz - last_checked_nnz
                               : last_checked_nnz - nnz;
    size_t minimum_change =
        last_checked_nnz / 128 +
        (last_checked_nnz % 128 != 0);
    if (minimum_change < 256) minimum_change = 256;
    return difference >= minimum_change;
}

static int materialized_bounded_candidates_need_csc_rebuild(
    const PreFOSPresolver *presolver,
    const PreFOSColumnWorkspace *workspace)
{
    const PreFOSCsrMatrix *matrix = &presolver->original.A;
    size_t row;

    if (!presolver->settings.bounded_doubleton_substitution ||
        !workspace->csc_has_insertions)
        return 0;
    for (row = 0; row < matrix->rows; ++row)
    {
        int position;
        int active_columns = 0;
        int dirty_column = 0;
        if (presolver->remove_rows[row] ||
            workspace->row_degrees[row] != 2 ||
            !isfinite(presolver->working_constraint_lower[row]) ||
            presolver->working_constraint_lower[row] !=
                presolver->working_constraint_upper[row])
            continue;
        for (position = matrix->row_pointers[row];
             position < matrix->row_pointers[row + 1]; ++position)
        {
            int column = matrix->column_indices[position];
            if (matrix->values[position] == 0.0 ||
                presolver->is_fixed[column] ||
                presolver->is_substituted[column] ||
                presolver->is_parallel_removed[column])
                continue;
            ++active_columns;
            dirty_column |=
                workspace->csc_column_dirty[column] != 0;
        }
        if (active_columns == 2 && dirty_column)
            return 1;
    }
    return 0;
}

static PreFOSStatus reduce_queued_trivial_rows(
    PreFOSPresolver *presolver, PreFOSColumnWorkspace *workspace)
{
    size_t candidate;
    size_t candidate_count = workspace->n_trivial_candidate_rows;
    PreFOSStatus status =
        prefos_internal_reduce_trivial_row_candidates(
            presolver, workspace->trivial_candidate_rows,
            candidate_count);
    for (candidate = 0; candidate < candidate_count; ++candidate)
        workspace->trivial_row_queued[
            workspace->trivial_candidate_rows[candidate]] = 0;
    workspace->n_trivial_candidate_rows = 0;
    return status;
}

PreFOSStatus prefos_internal_run_materialized_trivial_closure(
    PreFOSPresolver *presolver, size_t max_rounds)
{
    PreFOSCsrMatrix original_matrix;
    const PreFOSCsrMatrix *source_matrix;
    const double *source_lower;
    const double *source_upper;
    double *original_lower;
    double *original_upper;
    size_t *original_source_records;
    size_t original_source_record_count;
    int original_materialized;
    PreFOSWorkingMatrix current;
    int *candidate_rows = NULL;
    unsigned char *saved_fill_in_targets = NULL;
    PreFOSStatus status;
    size_t round;

    if (!presolver || max_rounds == 0)
        return presolver
                   ? PREFOS_STATUS_OK
                   : PREFOS_STATUS_INVALID_ARGUMENT;
    original_matrix = presolver->original.A;
    original_lower = presolver->working_constraint_lower;
    original_upper = presolver->working_constraint_upper;
    original_source_records =
        presolver->materialized_bound_source_records;
    original_source_record_count =
        presolver->n_materialized_bound_source_records;
    original_materialized =
        presolver->working_matrix_is_materialized;
    prefos_internal_get_working_matrix_source(
        presolver, &source_matrix, &source_lower, &source_upper);
    memset(&current, 0, sizeof(current));
    prefos_internal_free_linear_propagation_cache(presolver);
    prefos_internal_cuda_workspace_release(presolver);
    status = prefos_internal_materialize_working_matrix(
        presolver, source_matrix, source_lower, source_upper, &current);
    if (status != PREFOS_STATUS_OK) return status;
    candidate_rows = (int *) prefos_internal_alloc_array(
        original_matrix.rows, sizeof(int));
    saved_fill_in_targets = (unsigned char *)
        prefos_internal_alloc_array(
            presolver->original.n, sizeof(unsigned char));
    if ((original_matrix.rows > 0 && !candidate_rows) ||
        (presolver->original.n > 0 && !saved_fill_in_targets))
    {
        prefos_internal_free_working_matrix(&current);
        free(candidate_rows);
        free(saved_fill_in_targets);
        return PREFOS_STATUS_OUT_OF_MEMORY;
    }
    if (presolver->original.n > 0)
        memcpy(
            saved_fill_in_targets,
            presolver->substitution_fill_in_targets,
            presolver->original.n * sizeof(unsigned char));

    for (round = 0; round < max_rounds; ++round)
    {
        size_t *source_records = (size_t *) calloc(
            current.matrix.rows, sizeof(size_t));
        size_t fixed_before = presolver->n_fixed_columns;
        size_t candidate_count = 0;
        size_t row;
        if (current.matrix.rows > 0 && !source_records)
        {
            status = PREFOS_STATUS_OUT_OF_MEMORY;
            break;
        }
        presolver->original.A = current.matrix;
        presolver->working_constraint_lower = current.lower;
        presolver->working_constraint_upper = current.upper;
        presolver->working_matrix_is_materialized = 1;
        presolver->materialized_bound_source_records =
            source_records;
        presolver->n_materialized_bound_source_records =
            current.matrix.rows;
        for (row = 0; row < current.matrix.rows; ++row)
            if (!presolver->remove_rows[row] &&
                current.matrix.row_pointers[row + 1] -
                        current.matrix.row_pointers[row] <=
                    1)
                candidate_rows[candidate_count++] = (int) row;
        status = prefos_internal_reduce_trivial_row_candidates(
            presolver, candidate_rows, candidate_count);
        if (status == PREFOS_STATUS_OK && round == 0)
        {
            PreFOSColumnWorkspace workspace;
            memset(&workspace, 0, sizeof(workspace));
            if (presolver->original.n > 0)
                memset(
                    presolver->substitution_fill_in_targets, 0,
                    presolver->original.n *
                        sizeof(unsigned char));
            status =
                prefos_internal_build_structural_column_workspace_cpu(
                presolver, &workspace);
            if (status == PREFOS_STATUS_OK)
                trace_materialized_workspace_column(
                    "trivial-closure-before-dual", &workspace);
            if (status == PREFOS_STATUS_OK)
                status =
                    prefos_internal_reduce_empty_and_dual_fixed_columns(
                        presolver, &workspace);
            if (getenv("PREFOS_TRACE_MATERIALIZED_STAGES"))
                fprintf(
                    stderr,
                    "PreFOS trivial closure round=%zu candidates=%zu "
                    "after_dual_fixed=%zu\n",
                    round, candidate_count,
                    presolver->n_fixed_columns - fixed_before);
            prefos_internal_free_column_workspace(&workspace);
            if (presolver->original.n > 0)
                memcpy(
                    presolver->substitution_fill_in_targets,
                    saved_fill_in_targets,
                    presolver->original.n *
                        sizeof(unsigned char));
        }
        presolver->original.A = original_matrix;
        presolver->working_constraint_lower = original_lower;
        presolver->working_constraint_upper = original_upper;
        presolver->working_matrix_is_materialized =
            original_materialized;
        presolver->materialized_bound_source_records =
            original_source_records;
        presolver->n_materialized_bound_source_records =
            original_source_record_count;
        free(source_records);
        if (status != PREFOS_STATUS_OK ||
            presolver->n_fixed_columns == fixed_before)
            break;
        if (round + 1 < max_rounds)
        {
            PreFOSWorkingMatrix next;
            memset(&next, 0, sizeof(next));
            status = prefos_internal_materialize_working_matrix(
                presolver, &current.matrix, current.lower,
                current.upper, &next);
            if (status != PREFOS_STATUS_OK) break;
            prefos_internal_free_working_matrix(&current);
            current = next;
        }
    }
    if (status == PREFOS_STATUS_OK)
        prefos_internal_store_working_matrix_cache(presolver, &current);
    else
        prefos_internal_free_working_matrix(&current);
    free(candidate_rows);
    free(saved_fill_in_targets);
    return status;
}

PreFOSStatus
prefos_internal_run_materialized_column_closure_from_workspace(
    PreFOSPresolver *presolver, size_t max_rounds,
    int remove_parallel_rows, int propagate_linear_bounds,
    size_t *pending_parallel_rows_removed,
    int defer_inserted_parallel_columns,
    PreFOSColumnWorkspace *initial_workspace)
{
    PreFOSCsrMatrix original_matrix;
    const PreFOSCsrMatrix *source_matrix;
    const double *source_lower;
    const double *source_upper;
    double *original_lower;
    double *original_upper;
    size_t *original_source_records;
    size_t original_source_record_count;
    int original_materialized;
    PreFOSWorkingMatrix current;
    unsigned char *persistent_fill_in_targets = NULL;
    PreFOSStatus status;
    size_t round;
    int current_is_fresh = 1;
    int parallel_rows_checked = 0;
    int defer_parallel_recheck = 0;
    size_t parallel_last_checked_nnz = 0;
    int defer_initial_parallel_rows = 0;
    int reused_current_cache = 0;
    PreFOSColumnWorkspace workspace;
    int workspace_valid = 0;
    PreFOSTimestamp materialize_start, materialize_stop;

    if (!presolver || max_rounds == 0)
        return presolver ? PREFOS_STATUS_OK : PREFOS_STATUS_INVALID_ARGUMENT;
    if (pending_parallel_rows_removed)
        *pending_parallel_rows_removed = 0;
    original_matrix = presolver->original.A;
    original_lower = presolver->working_constraint_lower;
    original_upper = presolver->working_constraint_upper;
    original_source_records =
        presolver->materialized_bound_source_records;
    original_source_record_count =
        presolver->n_materialized_bound_source_records;
    original_materialized =
        presolver->working_matrix_is_materialized;
    if (propagate_linear_bounds)
    {
        prefos_internal_free_linear_propagation_cache(presolver);
        prefos_internal_cuda_workspace_release(presolver);
    }
    if (presolver->original.n > 0)
    {
        persistent_fill_in_targets = (unsigned char *)
            prefos_internal_alloc_array(
                presolver->original.n, sizeof(unsigned char));
        if (!persistent_fill_in_targets)
            return PREFOS_STATUS_OUT_OF_MEMORY;
        memcpy(
            persistent_fill_in_targets,
            presolver->substitution_fill_in_targets,
            presolver->original.n * sizeof(unsigned char));
    }
    memset(&current, 0, sizeof(current));
    memset(&workspace, 0, sizeof(workspace));
    if (initial_workspace)
    {
        workspace = *initial_workspace;
        memset(initial_workspace, 0, sizeof(*initial_workspace));
        workspace_valid = 1;
    }
    if (getenv("PREFOS_TRACE_WORKING_CACHE"))
        fprintf(
            stderr,
            "PreFOS working cache valid=%d fixed=%zu/%zu "
            "columns=%zu/%zu parallel_checked=%d\n",
            presolver->cached_working_matrix_valid,
            presolver->cached_working_fixed_column_epoch,
            presolver->fixed_column_epoch,
            presolver->cached_working_column_transformations,
            presolver->transformations.n_column_transformations,
            presolver->cached_working_parallel_rows_checked);
    reused_current_cache =
        prefos_internal_take_current_working_matrix_cache(
            presolver, &current, &parallel_rows_checked);
    if (reused_current_cache)
    {
        status = PREFOS_STATUS_OK;
        prefos_internal_timer_now(&materialize_start);
        materialize_stop = materialize_start;
    }
    else
    {
        prefos_internal_get_working_matrix_source(
            presolver, &source_matrix, &source_lower, &source_upper);
        prefos_internal_timer_now(&materialize_start);
        status = prefos_internal_materialize_working_matrix(
            presolver, source_matrix, source_lower, source_upper, &current);
        prefos_internal_timer_now(&materialize_stop);
    }
    trace_materialized_closure_materialization(
        SIZE_MAX, current.matrix.nnz,
        prefos_internal_timer_elapsed_milliseconds(
            &materialize_start, &materialize_stop));
    if (status != PREFOS_STATUS_OK)
    {
        free(persistent_fill_in_targets);
        return status;
    }
    if (remove_parallel_rows && !parallel_rows_checked &&
        current.matrix.nnz > 1000000 &&
        presolver->materialization_source_nnz <
            (current.matrix.nnz + 63) / 64)
        defer_initial_parallel_rows = 1;

    for (round = 0; round < max_rounds; ++round)
    {
        size_t substitutions_before =
            presolver->stats.substituted_free_variables;
        size_t parallel_before =
            presolver->stats.merged_parallel_columns;
        size_t fixed_before = presolver->n_fixed_columns;
        size_t removed_before = presolver->n_removed_rows;
        size_t parallel_removed = 0;
        size_t eager_substitutions = 0;
        size_t *source_records = NULL;
        int matrix_changed;
        int fixed_matrix_changed;
        PreFOSTimestamp workspace_start, workspace_stop, reduction_stop;

        trace_materialized_row(
            "before-parallel", presolver, &current.matrix,
            current.lower, current.upper);
        if (remove_parallel_rows &&
            presolver->settings.remove_redundant_rows &&
            !parallel_rows_checked &&
            !(round == 0 && defer_initial_parallel_rows) &&
            (!defer_parallel_recheck ||
             materialized_parallel_recheck_is_due(
                 current.matrix.nnz,
                 parallel_last_checked_nnz)))
        {
            size_t parallel_removed_before =
                presolver->n_removed_rows;
            status = prefos_internal_remove_parallel_rows_in_working_matrix(
                presolver, &current.matrix, current.lower, current.upper,
                workspace_valid ? &workspace : NULL);
            if (status != PREFOS_STATUS_OK)
            {
                if (workspace_valid)
                    prefos_internal_free_column_workspace(&workspace);
                prefos_internal_free_working_matrix(&current);
                free(persistent_fill_in_targets);
                return status;
            }
            parallel_rows_checked = 1;
            parallel_removed =
                presolver->n_removed_rows - parallel_removed_before;
            if (parallel_removed > 0)
                defer_parallel_recheck = 0;
            else
            {
                defer_parallel_recheck = 1;
                parallel_last_checked_nnz = current.matrix.nnz;
            }
        }
        trace_materialized_row(
            "after-parallel", presolver, &current.matrix,
            current.lower, current.upper);
        source_records = (size_t *) calloc(
            current.matrix.rows, sizeof(size_t));
        if (current.matrix.rows > 0 && !source_records)
        {
            if (workspace_valid)
                prefos_internal_free_column_workspace(&workspace);
            prefos_internal_free_working_matrix(&current);
            free(persistent_fill_in_targets);
            return PREFOS_STATUS_OUT_OF_MEMORY;
        }
        if (presolver->original.n > 0)
            memset(presolver->substitution_fill_in_targets, 0,
                   presolver->original.n * sizeof(unsigned char));
        presolver->original.A = current.matrix;
        presolver->working_constraint_lower = current.lower;
        presolver->working_constraint_upper = current.upper;
        presolver->working_matrix_is_materialized = 1;
        presolver->materialized_bound_source_records = source_records;
        presolver->n_materialized_bound_source_records =
            current.matrix.rows;
        if (workspace_valid && workspace.csc_has_insertions &&
            (defer_inserted_parallel_columns ||
             getenv("PREFOS_TRACE_MATERIALIZED_CLOSURE")))
        {
            prefos_internal_update_column_live_degrees(
                presolver, &workspace);
            if (getenv("PREFOS_TRACE_MATERIALIZED_CLOSURE"))
                fprintf(
                    stderr,
                    "PreFOS materialized closure pending-before-csc-rebuild "
                    "round=%zu singleton=%zu dual=%zu trivial=%zu "
                    "parallel_dirty=%zu\n",
                    round,
                    workspace.n_singleton_candidate_columns -
                        workspace.singleton_candidate_position,
                    workspace.n_dual_candidate_columns -
                        workspace.dual_candidate_position,
                    workspace.n_trivial_candidate_rows,
                    workspace.n_parallel_dirty_columns);
            if (defer_inserted_parallel_columns &&
                !propagate_linear_bounds && current_is_fresh &&
                parallel_rows_checked &&
                workspace.singleton_candidate_position ==
                    workspace.n_singleton_candidate_columns &&
                workspace.dual_candidate_position ==
                    workspace.n_dual_candidate_columns &&
                workspace.n_trivial_candidate_rows == 0 &&
                !materialized_bounded_candidates_need_csc_rebuild(
                    presolver, &workspace))
            {
                if (persistent_fill_in_targets)
                    memcpy(
                        presolver->substitution_fill_in_targets,
                        persistent_fill_in_targets,
                        presolver->original.n *
                            sizeof(unsigned char));
                presolver->original.A = original_matrix;
                presolver->working_constraint_lower = original_lower;
                presolver->working_constraint_upper = original_upper;
                presolver->working_matrix_is_materialized =
                    original_materialized;
                presolver->materialized_bound_source_records =
                    original_source_records;
                presolver->n_materialized_bound_source_records =
                    original_source_record_count;
                free(source_records);
                status = PREFOS_STATUS_OK;
                if (getenv("PREFOS_TRACE_MATERIALIZED_CLOSURE"))
                    fprintf(
                        stderr,
                        "PreFOS materialized closure deferred inserted "
                        "parallel columns at round=%zu\n",
                        round);
                break;
            }
        }
        prefos_internal_timer_now(&workspace_start);
        if (workspace_valid)
            status =
                prefos_internal_refresh_column_workspace_incremental(
                    presolver, &workspace);
        else if (current.column_counts && parallel_removed == 0)
        {
            status =
                prefos_internal_build_column_workspace_cpu_with_counts(
                    presolver, &workspace, current.column_counts,
                    propagate_linear_bounds);
            workspace_valid = status == PREFOS_STATUS_OK;
        }
        else
        {
            status =
                propagate_linear_bounds
                    ? prefos_internal_build_column_workspace_cpu(
                          presolver, &workspace)
                    : prefos_internal_build_structural_column_workspace_cpu(
                          presolver, &workspace);
            workspace_valid = status == PREFOS_STATUS_OK;
        }
        prefos_internal_timer_now(&workspace_stop);
        trace_materialized_closure_failure(
            round, "build-workspace", status);
        if (status == PREFOS_STATUS_OK)
        {
            size_t column;
            for (column = 0;
                 persistent_fill_in_targets &&
                 column < presolver->original.n;
                 ++column)
                    workspace.bounded_doubleton_chain_target[column] =
                        persistent_fill_in_targets[column];
            {
                size_t row;
                for (row = 0;
                     row < presolver->original.A.rows; ++row)
                    if (!presolver->remove_rows[row] &&
                        workspace.row_degrees[row] <= 1)
                        prefos_internal_queue_trivial_row(
                            presolver, &workspace, (int) row);
                status = reduce_queued_trivial_rows(
                    presolver, &workspace);
                trace_materialized_closure_failure(
                    round, "trivial-rows", status);
            }
        }
        if (getenv("PREFOS_TRACE_MATERIALIZED_STAGES"))
            fprintf(
                stderr, "PreFOS closure stage=trivial fixed=%zu rows=%zu\n",
                presolver->n_fixed_columns - fixed_before,
                presolver->n_removed_rows - removed_before);
        if (status == PREFOS_STATUS_OK)
        {
            prefos_internal_update_column_live_degrees(
                presolver, &workspace);
            trace_materialized_singletons(
                "before-dual", presolver, &workspace);
            trace_materialized_workspace_column(
                "general-before-dual", &workspace);
            status =
                prefos_internal_reduce_empty_and_dual_fixed_columns(
                    presolver, &workspace);
            trace_materialized_closure_failure(
                round, "dual-fixing", status);
        }
        if (getenv("PREFOS_TRACE_MATERIALIZED_STAGES"))
        {
            size_t fixed;
            fprintf(
                stderr, "PreFOS closure stage=dual fixed=%zu rows=%zu\n",
                presolver->n_fixed_columns - fixed_before,
                presolver->n_removed_rows - removed_before);
            for (fixed = fixed_before;
                 fixed < presolver->n_fixed_columns; ++fixed)
                fprintf(
                    stderr, "PreFOS closure dual-fixed column=%d\n",
                    presolver->fixed_column_log[fixed]);
        }
        if (status == PREFOS_STATUS_OK)
        {
            trace_materialized_singletons(
                "before-parallel", presolver, &workspace);
            status =
                prefos_internal_reduce_parallel_columns_in_workspace(
                    presolver, &workspace);
            trace_materialized_closure_failure(
                round, "parallel-columns", status);
        }
        if (getenv("PREFOS_TRACE_MATERIALIZED_STAGES"))
            fprintf(
                stderr, "PreFOS closure stage=parallel fixed=%zu rows=%zu\n",
                presolver->n_fixed_columns - fixed_before,
                presolver->n_removed_rows - removed_before);
        if (status == PREFOS_STATUS_OK)
        {
            trace_materialized_singletons(
                "after-parallel", presolver, &workspace);
            status = prefos_internal_reduce_singleton_columns(
                presolver, &workspace, 1);
            trace_materialized_closure_failure(
                round, "singleton-columns", status);
        }
        if (getenv("PREFOS_TRACE_MATERIALIZED_STAGES"))
            fprintf(
                stderr, "PreFOS closure stage=singleton fixed=%zu rows=%zu\n",
                presolver->n_fixed_columns - fixed_before,
                presolver->n_removed_rows - removed_before);
        if (status == PREFOS_STATUS_OK)
        {
            status = prefos_internal_reduce_bounded_doubletons(
                presolver, &workspace);
            trace_materialized_closure_failure(
                round, "bounded-doubletons", status);
        }
        if (status == PREFOS_STATUS_OK &&
            workspace.n_trivial_candidate_rows > 0)
        {
            status = reduce_queued_trivial_rows(
                presolver, &workspace);
            trace_materialized_closure_failure(
                round, "post-bounded-trivial-rows", status);
        }
        if (getenv("PREFOS_TRACE_MATERIALIZED_STAGES"))
            fprintf(
                stderr, "PreFOS closure stage=bounded fixed=%zu rows=%zu\n",
                presolver->n_fixed_columns - fixed_before,
                presolver->n_removed_rows - removed_before);
        if (status == PREFOS_STATUS_OK && propagate_linear_bounds &&
            !workspace.csc_has_insertions)
        {
            size_t fixed = 0;
            PreFOSTimestamp propagation_start, propagation_stop;
            prefos_internal_timer_now(&propagation_start);
            status = prefos_internal_propagate_linear_bounds(
                presolver, &workspace);
            if (status == PREFOS_STATUS_OK &&
                presolver->linear_propagation_complete)
                presolver->linear_propagation_bound_cursor =
                    presolver->transformations.n_bound_changes;
            if (status == PREFOS_STATUS_OK)
                status = prefos_internal_find_fixed_box_variables(
                    presolver, &workspace, &fixed);
            prefos_internal_timer_now(&propagation_stop);
            presolver->stats.linear_propagation_milliseconds +=
                prefos_internal_timer_elapsed_milliseconds(
                    &propagation_start, &propagation_stop);
            trace_materialized_closure_failure(
                round, "linear-propagation", status);
            prefos_internal_free_linear_propagation_cache(presolver);
            prefos_internal_cuda_workspace_release(presolver);
        }
        if (status == PREFOS_STATUS_OK &&
            !workspace.csc_has_insertions)
            status = prefos_internal_run_fast_fixed_point(
                presolver, 1, 0, &workspace);
        prefos_internal_timer_now(&reduction_stop);

        if (workspace_valid)
        {
            eager_substitutions =
                workspace.materialized_eager_substitutions;
            workspace.materialized_eager_substitutions = 0;
        }
        if (eager_substitutions > 0)
        {
            compact_exact_zeros(&current.matrix);
            free(current.column_counts);
            current.column_counts = NULL;
        }
        if (persistent_fill_in_targets)
        {
            size_t column;
            for (column = 0; column < presolver->original.n; ++column)
                persistent_fill_in_targets[column] |=
                    presolver->substitution_fill_in_targets[column];
        }
        presolver->original.A = original_matrix;
        presolver->working_constraint_lower = original_lower;
        presolver->working_constraint_upper = original_upper;
        presolver->working_matrix_is_materialized =
            original_materialized;
        presolver->materialized_bound_source_records =
            original_source_records;
        presolver->n_materialized_bound_source_records =
            original_source_record_count;
        free(source_records);
        if (persistent_fill_in_targets)
            memcpy(
                presolver->substitution_fill_in_targets,
                persistent_fill_in_targets,
                presolver->original.n * sizeof(unsigned char));
        trace_materialized_closure_round(
            round, current.matrix.nnz,
            presolver->stats.substituted_free_variables -
                substitutions_before,
            presolver->stats.merged_parallel_columns - parallel_before,
            presolver->n_fixed_columns - fixed_before,
            presolver->n_removed_rows - removed_before,
            prefos_internal_timer_elapsed_milliseconds(
                &workspace_start, &workspace_stop),
            prefos_internal_timer_elapsed_milliseconds(
                &workspace_stop, &reduction_stop));
        if (status != PREFOS_STATUS_OK)
        {
            if (workspace_valid)
                prefos_internal_free_column_workspace(&workspace);
            prefos_internal_free_working_matrix(&current);
            free(persistent_fill_in_targets);
            return status;
        }
        if (presolver->stats.substituted_free_variables ==
                substitutions_before &&
            presolver->stats.merged_parallel_columns ==
                parallel_before &&
            presolver->n_fixed_columns == fixed_before &&
            presolver->n_removed_rows == removed_before)
            break;
        matrix_changed =
            presolver->stats.substituted_free_variables !=
                substitutions_before ||
            presolver->stats.merged_parallel_columns !=
                parallel_before ||
            presolver->n_fixed_columns != fixed_before;
        fixed_matrix_changed =
            presolver->n_fixed_columns != fixed_before;
        current_is_fresh =
            presolver->stats.merged_parallel_columns ==
                parallel_before &&
            presolver->stats.substituted_free_variables -
                    substitutions_before ==
                eager_substitutions &&
            !fixed_matrix_changed;
        if (!matrix_changed) break;
        if (matrix_changed)
            parallel_rows_checked = 0;
        if (presolver->stats.substituted_free_variables ==
                substitutions_before &&
            presolver->stats.merged_parallel_columns ==
                parallel_before &&
            (propagate_linear_bounds ||
             presolver->n_fixed_columns == fixed_before) &&
            presolver->n_removed_rows == removed_before)
            break;
        if (materialized_closure_round_is_stale(
                current.matrix.nnz,
                presolver->stats.substituted_free_variables -
                    substitutions_before,
                presolver->stats.merged_parallel_columns -
                    parallel_before,
                presolver->n_fixed_columns - fixed_before,
                presolver->n_removed_rows - removed_before))
            break;
        if (round + 1 < max_rounds && !current_is_fresh)
        {
            PreFOSWorkingMatrix next;
            if (workspace_valid)
            {
                prefos_internal_free_column_workspace(&workspace);
                workspace_valid = 0;
            }
            memset(&next, 0, sizeof(next));
            prefos_internal_timer_now(&materialize_start);
            status = prefos_internal_materialize_working_matrix(
                presolver, &current.matrix, current.lower, current.upper,
                &next);
            prefos_internal_timer_now(&materialize_stop);
            trace_materialized_closure_materialization(
                round, next.matrix.nnz,
                prefos_internal_timer_elapsed_milliseconds(
                    &materialize_start, &materialize_stop));
            trace_materialized_closure_failure(
                round, "rematerialize", status);
            if (status != PREFOS_STATUS_OK)
            {
                prefos_internal_free_working_matrix(&current);
                free(persistent_fill_in_targets);
                return status;
            }
            prefos_internal_free_working_matrix(&current);
            current = next;
            current_is_fresh = 1;
        }
    }
    if (!current_is_fresh)
    {
        PreFOSWorkingMatrix next;
        if (workspace_valid)
        {
            prefos_internal_free_column_workspace(&workspace);
            workspace_valid = 0;
        }
        memset(&next, 0, sizeof(next));
        status = prefos_internal_materialize_working_matrix(
            presolver, &current.matrix, current.lower, current.upper,
            &next);
        if (status != PREFOS_STATUS_OK)
        {
            prefos_internal_free_working_matrix(&current);
            free(persistent_fill_in_targets);
            return status;
        }
        prefos_internal_free_working_matrix(&current);
        current = next;
        current_is_fresh = 1;
    }
    if (remove_parallel_rows &&
        presolver->settings.remove_redundant_rows &&
        !parallel_rows_checked)
    {
        size_t removed_before = presolver->n_removed_rows;
        status = prefos_internal_remove_parallel_rows_in_working_matrix(
            presolver, &current.matrix, current.lower, current.upper,
            workspace_valid ? &workspace : NULL);
        if (status != PREFOS_STATUS_OK)
        {
            prefos_internal_free_working_matrix(&current);
            free(persistent_fill_in_targets);
            return status;
        }
        parallel_rows_checked = 1;
        if (presolver->n_removed_rows > removed_before)
        {
            if (pending_parallel_rows_removed)
                *pending_parallel_rows_removed =
                    presolver->n_removed_rows - removed_before;
        }
    }
    if (workspace_valid && initial_workspace)
    {
        *initial_workspace = workspace;
        memset(&workspace, 0, sizeof(workspace));
        workspace_valid = 0;
    }
    else if (workspace_valid)
        prefos_internal_free_column_workspace(&workspace);
    prefos_internal_store_working_matrix_cache(presolver, &current);
    presolver->cached_working_parallel_rows_checked =
        parallel_rows_checked && current_is_fresh;
    free(persistent_fill_in_targets);
    return PREFOS_STATUS_OK;
}

PreFOSStatus prefos_internal_run_materialized_column_closure(
    PreFOSPresolver *presolver, size_t max_rounds,
    int remove_parallel_rows, int propagate_linear_bounds,
    size_t *pending_parallel_rows_removed,
    int defer_inserted_parallel_columns)
{
    return prefos_internal_run_materialized_column_closure_from_workspace(
        presolver, max_rounds, remove_parallel_rows,
        propagate_linear_bounds, pending_parallel_rows_removed,
        defer_inserted_parallel_columns, NULL);
}
