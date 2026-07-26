/*
 * Copyright 2026 Hongpei Li
 * SPDX-License-Identifier: Apache-2.0
 */

#include "PREFOS_ColumnReductionInternal.h"

#include <stdio.h>
#include <stdlib.h>

static int traced_dual_column(void)
{
    static int initialized = 0;
    static int selected = -1;
    if (!initialized)
    {
        const char *value = getenv("PREFOS_TRACE_DUAL_COLUMN");
        if (value && *value)
            selected = atoi(value);
        initialized = 1;
    }
    return selected;
}

static void trace_dual_column_locks(
    const PreFOSPresolver *presolver,
    const PreFOSColumnWorkspace *workspace, int column)
{
    int position;
    if (column != traced_dual_column()) return;
    fprintf(
        stderr,
        "PreFOS dual-column col=%d degree=%d down=%d up=%d "
        "fill_target=%d materialized=%d\n",
        column, workspace->live_degrees[column],
        workspace->down_locks[column], workspace->up_locks[column],
        presolver->substitution_fill_in_targets[column],
        presolver->working_matrix_is_materialized);
    for (position = workspace->starts[column];
         position < workspace->ends[column]; ++position)
    {
        int row = workspace->rows[position];
        fprintf(
            stderr,
            "PreFOS dual-column row=%d a=%.17g removed=%d dirty=%d "
            "lower=%.17g upper=%.17g\n",
            row, workspace->values[position],
            presolver->remove_rows[row], workspace->dirty_row[row],
            presolver->working_constraint_lower[row],
            presolver->working_constraint_upper[row]);
    }
}

static double choose_zero_or_bound(double lower, double upper)
{
    if (lower <= 0.0 && upper >= 0.0) return 0.0;
    if (isfinite(lower) && lower > 0.0) return lower;
    if (isfinite(upper) && upper < 0.0) return upper;
    if (isfinite(lower)) return lower;
    if (isfinite(upper)) return upper;
    return 0.0;
}

static PreFOSStatus append_dual_deleted_row(
    PreFOSPresolver *presolver, int row)
{
    PresolveRowTransformationRecord record = {
        .type = PRESOLVE_ROW_DELETED,
        .row = row,
        .source_row = -1,
        .dual_value = 0.0};
    return presolve_transformation_log_append_row_transformation(
               &presolver->transformations, &record, NULL)
               ? PREFOS_STATUS_OK
               : PREFOS_STATUS_OUT_OF_MEMORY;
}

static double infinite_fix_row_side(
    const PreFOSPresolver *presolver, int row,
    double coefficient, int direction)
{
    if (direction > 0)
        return coefficient > 0.0
                   ? presolver->working_constraint_lower[row]
                   : presolver->working_constraint_upper[row];
    return coefficient > 0.0
               ? presolver->working_constraint_upper[row]
               : presolver->working_constraint_lower[row];
}

static PreFOSStatus fix_zero_objective_column_to_infinity(
    PreFOSPresolver *presolver, PreFOSColumnWorkspace *workspace,
    int column, int direction, int *applied)
{
    const PreFOSCsrMatrix *matrix = &presolver->original.A;
    PresolveColumnTransformationRecord record;
    int *deleted_rows = NULL;
    int *row_starts = NULL;
    int *indices = NULL;
    double *coefficients = NULL;
    double *row_sides = NULL;
    size_t degree, n_deleted = 0, n_recovery_rows = 0;
    size_t length = 0, position, row_position;
    int box_position;
    double placeholder;
    PreFOSStatus status = PREFOS_STATUS_OK;

    *applied = 0;
    if (!presolver->settings.remove_redundant_rows ||
        (direction != -1 && direction != 1))
        return PREFOS_STATUS_OK;
    degree = (size_t) (workspace->ends[column] -
                       workspace->starts[column]);
    deleted_rows =
        (int *) prefos_internal_alloc_array(degree, sizeof(int));
    if (degree > 0 && !deleted_rows)
        return PREFOS_STATUS_OUT_OF_MEMORY;

    for (position = (size_t) workspace->starts[column];
         position < (size_t) workspace->ends[column]; ++position)
    {
        int row = workspace->rows[position];
        double side;
        size_t row_length;
        if (presolver->remove_rows[row] ||
            (n_deleted > 0 &&
             deleted_rows[n_deleted - 1] == row))
            continue;
        deleted_rows[n_deleted++] = row;
        side = infinite_fix_row_side(
            presolver, row, workspace->values[position],
            direction);
        if (!isfinite(side)) continue;
        row_length = (size_t)
            (matrix->row_pointers[row + 1] -
             matrix->row_pointers[row]);
        if (row_length > (size_t) INT_MAX ||
            length > (size_t) INT_MAX - row_length)
        {
            free(deleted_rows);
            return PREFOS_STATUS_OK;
        }
        length += row_length;
        ++n_recovery_rows;
    }
    if (n_deleted == 0)
    {
        free(deleted_rows);
        return PREFOS_STATUS_OK;
    }

    row_starts = (int *) prefos_internal_alloc_array(
        n_recovery_rows + 1, sizeof(int));
    row_sides = (double *) prefos_internal_alloc_array(
        n_recovery_rows, sizeof(double));
    indices =
        (int *) prefos_internal_alloc_array(length, sizeof(int));
    coefficients =
        (double *) prefos_internal_alloc_array(length, sizeof(double));
    if (!row_starts ||
        (n_recovery_rows > 0 && !row_sides) ||
        (length > 0 && (!indices || !coefficients)))
    {
        status = PREFOS_STATUS_OUT_OF_MEMORY;
        goto cleanup;
    }

    row_position = 0;
    length = 0;
    row_starts[0] = 0;
    for (position = 0; position < n_deleted; ++position)
    {
        int row = deleted_rows[position];
        double column_coefficient = 0.0;
        double side;
        int entry;
        for (entry = matrix->row_pointers[row];
             entry < matrix->row_pointers[row + 1]; ++entry)
        {
            if (matrix->column_indices[entry] == column)
                column_coefficient += matrix->values[entry];
        }
        if (column_coefficient == 0.0)
        {
            status = PREFOS_STATUS_NUMERICAL_ERROR;
            goto cleanup;
        }
        side = infinite_fix_row_side(
            presolver, row, column_coefficient, direction);
        if (!isfinite(side)) continue;
        row_sides[row_position] = side;
        for (entry = matrix->row_pointers[row];
             entry < matrix->row_pointers[row + 1]; ++entry)
        {
            indices[length] = matrix->column_indices[entry];
            coefficients[length] = matrix->values[entry];
            ++length;
        }
        row_starts[++row_position] = (int) length;
    }
    if (row_position != n_recovery_rows)
    {
        status = PREFOS_STATUS_NUMERICAL_ERROR;
        goto cleanup;
    }

    box_position = presolver->variable_to_box[column];
    placeholder =
        direction > 0
            ? presolver->working_box_lower[box_position]
            : presolver->working_box_upper[box_position];
    if (!isfinite(placeholder)) placeholder = 0.0;
    memset(&record, 0, sizeof(record));
    record.type = PRESOLVE_COLUMN_FIXED_INFINITE;
    record.column = column;
    record.secondary_column = -1;
    record.source_row = -1;
    record.direction = direction;
    record.value = placeholder;
    record.lower = presolver->working_box_lower[box_position];
    record.upper = presolver->working_box_upper[box_position];
    record.indices = indices;
    record.coefficients = coefficients;
    record.length = length;
    record.row_starts = row_starts;
    record.row_sides = row_sides;
    record.n_rows = n_recovery_rows;
    if (!presolve_transformation_log_append_column_transformation(
            &presolver->transformations, &record, NULL))
    {
        status = PREFOS_STATUS_OUT_OF_MEMORY;
        goto cleanup;
    }
    for (position = 0; position < n_deleted; ++position)
    {
        status = append_dual_deleted_row(
            presolver, deleted_rows[position]);
        if (status != PREFOS_STATUS_OK) goto cleanup;
    }

    prefos_internal_mark_fixed_column(
        presolver, column, placeholder);
    for (position = 0; position < n_deleted; ++position)
        if (prefos_internal_mark_removed_row(
                presolver, (size_t) deleted_rows[position]))
            ++presolver->stats.removed_redundant_rows;
    *applied = 1;

cleanup:
    free(deleted_rows);
    free(row_starts);
    free(row_sides);
    free(indices);
    free(coefficients);
    return status;
}

PreFOSStatus prefos_internal_reduce_empty_and_dual_fixed_columns(
    PreFOSPresolver *presolver, PreFOSColumnWorkspace *workspace)
{
    const int trace = getenv("PREFOS_TRACE_DUAL_FIX") != NULL;
    size_t examined = 0, eligible = 0, empty = 0;
    size_t down_free = 0, up_free = 0, fixed = 0;

    while (workspace->dual_candidate_position <
           workspace->n_dual_candidate_columns)
    {
        int candidate = workspace->dual_candidate_columns[
            workspace->dual_candidate_position++];
        size_t column = (size_t) candidate;
        int box_position;
        double lower, upper, objective, value;
        int down_locked = 0, up_locked = 0;
        ++examined;
        workspace->dual_column_queued[column] = 0;
        trace_dual_column_locks(
            presolver, workspace, (int) column);
        if (!prefos_internal_column_is_linear_box(
                presolver, workspace, (int) column))
            continue;
        ++eligible;
        box_position = presolver->variable_to_box[column];
        lower = presolver->working_box_lower[box_position];
        upper = presolver->working_box_upper[box_position];
        objective = workspace->objective[column];

        if ((workspace->gpu_stats_valid &&
             workspace->gpu_degrees[column] == 0) ||
            (!workspace->gpu_stats_valid &&
             workspace->live_degrees[column] == 0))
        {
            ++empty;
            if (!presolver->settings.remove_empty_columns) continue;
            if (objective > 0.0)
            {
                if (!isfinite(lower)) return PREFOS_STATUS_PRIMAL_UNBOUNDED;
                value = lower;
            }
            else if (objective < 0.0)
            {
                if (!isfinite(upper)) return PREFOS_STATUS_PRIMAL_UNBOUNDED;
                value = upper;
            }
            else
                value = choose_zero_or_bound(lower, upper);
            prefos_internal_mark_fixed_column(presolver, (int) column, value);
            ++fixed;
            ++presolver->stats.removed_empty_columns;
            continue;
        }
        if (!presolver->settings.dual_fixing) continue;

        if (workspace->gpu_stats_valid)
        {
            down_locked = workspace->gpu_down_locked[column] != 0;
            up_locked = workspace->gpu_up_locked[column] != 0;
        }
        else
        {
            down_locked = workspace->down_locks[column] > 0;
            up_locked = workspace->up_locks[column] > 0;
        }
        down_free += !down_locked;
        up_free += !up_locked;
        if (objective > 0.0 && !down_locked)
        {
            if (!isfinite(lower)) return PREFOS_STATUS_PRIMAL_UNBOUNDED;
            prefos_internal_mark_fixed_column(presolver, (int) column, lower);
            ++fixed;
            ++presolver->stats.dual_fixed_columns;
        }
        else if (objective < 0.0 && !up_locked)
        {
            if (!isfinite(upper)) return PREFOS_STATUS_PRIMAL_UNBOUNDED;
            prefos_internal_mark_fixed_column(presolver, (int) column, upper);
            ++fixed;
            ++presolver->stats.dual_fixed_columns;
        }
        else if (objective == 0.0)
        {
            if (!down_locked && isfinite(lower))
            {
                prefos_internal_mark_fixed_column(
                    presolver, (int) column, lower);
                ++fixed;
                ++presolver->stats.dual_fixed_columns;
            }
            else if (!up_locked && isfinite(upper))
            {
                prefos_internal_mark_fixed_column(
                    presolver, (int) column, upper);
                ++fixed;
                ++presolver->stats.dual_fixed_columns;
            }
            else if (!down_locked && !isfinite(lower))
            {
                int applied = 0;
                PreFOSStatus status =
                    fix_zero_objective_column_to_infinity(
                        presolver, workspace, (int) column,
                        -1, &applied);
                if (status != PREFOS_STATUS_OK) return status;
                if (applied)
                {
                    ++fixed;
                    ++presolver->stats.dual_fixed_columns;
                }
            }
            else if (!up_locked && !isfinite(upper))
            {
                int applied = 0;
                PreFOSStatus status =
                    fix_zero_objective_column_to_infinity(
                        presolver, workspace, (int) column,
                        1, &applied);
                if (status != PREFOS_STATUS_OK) return status;
                if (applied)
                {
                    ++fixed;
                    ++presolver->stats.dual_fixed_columns;
                }
            }
        }
        if (trace && presolver->is_fixed[column])
            fprintf(
                stderr,
                "PreFOS dual-fix col=%zu value=%.17g objective=%.17g "
                "down=%d up=%d\n",
                column, presolver->fixed_values[column], objective,
                down_locked, up_locked);
    }
    workspace->dual_candidate_position = 0;
    workspace->n_dual_candidate_columns = 0;
    if (trace)
        fprintf(
            stderr,
            "PreFOS dual-fix examined=%zu eligible=%zu empty=%zu "
            "down_free=%zu up_free=%zu fixed=%zu\n",
            examined, eligible, empty, down_free, up_free, fixed);
    return PREFOS_STATUS_OK;
}
