/*
 * Copyright 2026 Hongpei Li
 * SPDX-License-Identifier: Apache-2.0
 */

#include "PREFOS_MaterializedDoubletons.h"

#include "PREFOS_LinearPropagationCache.h"

static int find_unique_row_column(
    const PreFOSCsrMatrix *matrix, int row, int column, int *found)
{
    int lower = matrix->row_pointers[row];
    int upper = matrix->row_pointers[row + 1];
    *found = -1;
    while (lower < upper)
    {
        int middle = lower + (upper - lower) / 2;
        if (matrix->column_indices[middle] < column)
            lower = middle + 1;
        else
            upper = middle;
    }
    if (lower >= matrix->row_pointers[row + 1] ||
        matrix->column_indices[lower] != column)
        return 1;
    if (lower + 1 < matrix->row_pointers[row + 1] &&
        matrix->column_indices[lower + 1] == column)
        return 0;
    *found = lower;
    return 1;
}

static int find_unique_column_row(
    const PreFOSColumnWorkspace *workspace, int column, int row,
    int *found)
{
    int lower = workspace->starts[column];
    int upper = workspace->ends[column];
    *found = -1;
    while (lower < upper)
    {
        int middle = lower + (upper - lower) / 2;
        int candidate = workspace->rows[middle];
        if (candidate < row)
            lower = middle + 1;
        else
            upper = middle;
    }
    if (lower >= workspace->ends[column] ||
        workspace->rows[lower] != row)
        return 1;
    if (lower + 1 < workspace->ends[column] &&
        workspace->rows[lower + 1] == row)
        return 0;
    *found = lower;
    return 1;
}

static void replace_row_column(
    PreFOSCsrMatrix *matrix, int row, int position,
    int column, double value)
{
    int begin = matrix->row_pointers[row];
    int end = matrix->row_pointers[row + 1];
    matrix->column_indices[position] = column;
    matrix->values[position] = value;
    while (position > begin &&
           matrix->column_indices[position - 1] > column)
    {
        int previous_column =
            matrix->column_indices[position - 1];
        double previous_value = matrix->values[position - 1];
        matrix->column_indices[position] = previous_column;
        matrix->values[position] = previous_value;
        --position;
        matrix->column_indices[position] = column;
        matrix->values[position] = value;
    }
    while (position + 1 < end &&
           matrix->column_indices[position + 1] < column)
    {
        int next_column = matrix->column_indices[position + 1];
        double next_value = matrix->values[position + 1];
        matrix->column_indices[position] = next_column;
        matrix->values[position] = next_value;
        ++position;
        matrix->column_indices[position] = column;
        matrix->values[position] = value;
    }
}

static int transformed_row_values(
    const PreFOSPresolver *presolver,
    const PreFOSColumnWorkspace *workspace, int row,
    int pivot_column, int target_column, double alpha, double beta,
    int *pivot_position, int *target_position,
    int *target_workspace_position, double *new_target,
    double *old_target, double *new_lower, double *new_upper)
{
    const PreFOSCsrMatrix *matrix = &presolver->original.A;
    double pivot_coefficient;

    if (!find_unique_row_column(
            matrix, row, pivot_column, pivot_position) ||
        !find_unique_row_column(
            matrix, row, target_column, target_position) ||
        !find_unique_column_row(
            workspace, target_column, row,
            target_workspace_position) ||
        *pivot_position < 0)
        return 0;
    pivot_coefficient = matrix->values[*pivot_position];
    if (pivot_coefficient == 0.0)
        return 0;
    if (*target_position < 0)
    {
        if (*target_workspace_position >= 0) return 0;
        *old_target = 0.0;
    }
    else
    {
        *old_target = matrix->values[*target_position];
        if (*target_workspace_position >= 0)
        {
            if (workspace->values[*target_workspace_position] !=
                *old_target)
                return 0;
        }
        else if (*old_target != 0.0 &&
                 !workspace->csc_column_dirty[target_column])
            return 0;
    }
    *new_target = *old_target;
    if (!prefos_internal_safe_add_product(
            new_target, pivot_coefficient, alpha))
        return 0;
    *new_lower = presolver->working_constraint_lower[row];
    *new_upper = presolver->working_constraint_upper[row];
    if (isfinite(*new_lower) &&
        !prefos_internal_safe_add_product(
            new_lower, -pivot_coefficient, beta))
        return 0;
    if (isfinite(*new_upper) &&
        !prefos_internal_safe_add_product(
            new_upper, -pivot_coefficient, beta))
        return 0;
    return !isnan(*new_lower) && !isnan(*new_upper);
}

int prefos_internal_can_update_materialized_doubleton(
    const PreFOSPresolver *presolver,
    const PreFOSColumnWorkspace *workspace, int source_row,
    int pivot_column, int target_column, double alpha, double beta)
{
    int position;
    if (!presolver || !workspace ||
        !presolver->working_matrix_is_materialized ||
        source_row < 0 || pivot_column < 0 || target_column < 0 ||
        workspace->csc_column_dirty[pivot_column])
        return 0;
    for (position = workspace->starts[pivot_column];
         position < workspace->ends[pivot_column]; ++position)
    {
        int row = workspace->rows[position];
        int pivot_position, target_position, target_workspace_position;
        double old_target, new_target, new_lower, new_upper;
        if (row == source_row || presolver->remove_rows[row] ||
            workspace->values[position] == 0.0)
            continue;
        if (workspace->dirty_row[row]) return 0;
        if ((position > workspace->starts[pivot_column] &&
             workspace->rows[position - 1] == row) ||
            (position + 1 < workspace->ends[pivot_column] &&
             workspace->rows[position + 1] == row))
            return 0;
        if (!transformed_row_values(
                presolver, workspace, row, pivot_column,
                target_column, alpha, beta, &pivot_position,
                &target_position, &target_workspace_position,
                &new_target, &old_target, &new_lower, &new_upper))
            return 0;
    }
    return 1;
}

PreFOSStatus prefos_internal_update_materialized_doubleton(
    PreFOSPresolver *presolver, PreFOSColumnWorkspace *workspace,
    int source_row, int pivot_column, int target_column,
    double alpha, double beta, size_t event_count_before)
{
    PreFOSCsrMatrix *matrix = &presolver->original.A;
    int position;

    for (position = workspace->starts[pivot_column];
         position < workspace->ends[pivot_column]; ++position)
    {
        int row = workspace->rows[position];
        int pivot_position, target_position, target_workspace_position;
        double old_pivot = workspace->values[position];
        double old_target, new_target, new_lower, new_upper;
        PreFOSStatus status;
        if (row == source_row || presolver->remove_rows[row] ||
            old_pivot == 0.0)
            continue;
        if (!transformed_row_values(
                presolver, workspace, row, pivot_column,
                target_column, alpha, beta, &pivot_position,
                &target_position, &target_workspace_position,
                &new_target, &old_target, &new_lower, &new_upper))
            return PREFOS_STATUS_NUMERICAL_ERROR;

        workspace->values[position] = 0.0;
        workspace->csc_has_zero_entries = 1;
        if (target_position >= 0)
        {
            matrix->values[pivot_position] = 0.0;
            matrix->values[target_position] = new_target;
        }
        else if (new_target != 0.0)
        {
            replace_row_column(
                matrix, row, pivot_position,
                target_column, new_target);
            workspace->csc_column_dirty[target_column] = 1;
            workspace->csc_has_insertions = 1;
        }
        else
            matrix->values[pivot_position] = 0.0;
        if (target_workspace_position >= 0)
            workspace->values[target_workspace_position] =
                new_target;
        else if (new_target != 0.0)
        {
            workspace->csc_column_dirty[target_column] = 1;
            workspace->csc_has_insertions = 1;
        }
        presolver->working_constraint_lower[row] = new_lower;
        presolver->working_constraint_upper[row] = new_upper;
        status = prefos_internal_update_workspace_row_coefficients(
            presolver, workspace, (size_t) row,
            pivot_column, old_pivot, 0.0,
            target_column, old_target, new_target);
        if (status != PREFOS_STATUS_OK) return status;
        if (workspace->row_degrees[row] > 0)
            --workspace->row_degrees[row];
        if (old_target != 0.0 && new_target == 0.0)
        {
            if (workspace->row_degrees[row] > 0)
                --workspace->row_degrees[row];
            if (workspace->live_degrees[target_column] > 0)
                --workspace->live_degrees[target_column];
        }
        else if (old_target == 0.0 && new_target != 0.0)
        {
            ++workspace->row_degrees[row];
            ++workspace->live_degrees[target_column];
        }
        prefos_internal_invalidate_singleton_row_activity(
            workspace, (size_t) row);
        if (workspace->initial_activity_valid)
            workspace->initial_activity_valid[row] = 0;
        prefos_internal_linear_cache_mark_row_dirty(
            presolver, (size_t) row, 1);
        if (workspace->row_degrees[row] <= 1)
            prefos_internal_queue_trivial_row(
                presolver, workspace, row);
    }
    workspace->live_degrees[pivot_column] = 0;
    workspace->gpu_stats_valid = 0;
    workspace->gpu_csc_valid = 0;
    workspace->gpu_singleton_candidates_valid = 0;
    prefos_internal_queue_dual_column(
        presolver, workspace, pivot_column);
    if (workspace->live_degrees[target_column] == 1)
        prefos_internal_queue_singleton_column(
            presolver, workspace, target_column);
    prefos_internal_queue_dual_column(
        presolver, workspace, target_column);
    prefos_internal_mark_parallel_column_dirty(
        workspace, pivot_column);
    prefos_internal_mark_parallel_column_dirty(
        workspace, target_column);
    if (workspace->transformation_event_cursor == event_count_before)
        workspace->transformation_event_cursor =
            presolver->transformations.n_events;
    ++workspace->materialized_eager_substitutions;
    return PREFOS_STATUS_OK;
}
