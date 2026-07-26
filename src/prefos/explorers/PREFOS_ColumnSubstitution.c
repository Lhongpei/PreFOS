/*
 * Copyright 2026 Hongpei Li
 * SPDX-License-Identifier: Apache-2.0
 */

#include "PREFOS_ColumnReductionInternal.h"

static int add_objective_product(
    double *accumulator, double left, double right)
{
    double product = left * right;
    double result = *accumulator + product;
    if (isfinite(product) && isfinite(result))
    {
        *accumulator = result;
        return 1;
    }
    return prefos_internal_safe_add_product(
        accumulator, left, right);
}

static int residual_singleton_is_box_redundant(
    const PreFOSPresolver *presolver, int row, int column)
{
    const PreFOSCsrMatrix *matrix = &presolver->original.A;
    int box = presolver->variable_to_box[column];
    double coefficient = 0.0, lower, upper;
    double implied_lower, implied_upper;
    double tolerance = presolver->settings.feasibility_tolerance;
    int position;

    if (box < 0 ||
        prefos_internal_effective_row_bounds(
            presolver, (size_t) row, &lower, &upper) != PREFOS_STATUS_OK)
        return 0;
    for (position = matrix->row_pointers[row];
         position < matrix->row_pointers[row + 1]; ++position)
        if (matrix->column_indices[position] == column)
        {
            coefficient = matrix->values[position];
            break;
        }
    if (coefficient == 0.0) return 0;
    if (coefficient > 0.0)
    {
        implied_lower = lower / coefficient;
        implied_upper = upper / coefficient;
    }
    else
    {
        implied_lower = upper / coefficient;
        implied_upper = lower / coefficient;
    }
    return implied_lower <=
               presolver->working_box_lower[box] + tolerance &&
           implied_upper >=
               presolver->working_box_upper[box] - tolerance;
}

static PreFOSStatus reserve_substitution_terms(PreFOSPresolver *presolver,
                                               size_t additional)
{
    size_t required, capacity;
    int *targets;
    double *scales;
    if (additional > SIZE_MAX - presolver->n_substitution_terms)
        return PREFOS_STATUS_OUT_OF_MEMORY;
    required = presolver->n_substitution_terms + additional;
    if (required <= presolver->substitution_term_capacity) return PREFOS_STATUS_OK;
    capacity = presolver->substitution_term_capacity == 0
                   ? 1024
                   : presolver->substitution_term_capacity;
    while (capacity < required)
    {
        if (capacity > SIZE_MAX / 2)
        {
            capacity = required;
            break;
        }
        capacity *= 2;
    }
    if (capacity > SIZE_MAX / sizeof(int) ||
        capacity > SIZE_MAX / sizeof(double))
        return PREFOS_STATUS_OUT_OF_MEMORY;
    targets = (int *) realloc(
        presolver->substitution_targets, capacity * sizeof(int));
    if (!targets) return PREFOS_STATUS_OUT_OF_MEMORY;
    presolver->substitution_targets = targets;
    scales = (double *) realloc(
        presolver->substitution_scales, capacity * sizeof(double));
    if (!scales) return PREFOS_STATUS_OUT_OF_MEMORY;
    presolver->substitution_scales = scales;
    presolver->substitution_term_capacity = capacity;
    return PREFOS_STATUS_OK;
}

static PreFOSStatus reserve_active_column_scratch(
    PreFOSColumnWorkspace *workspace, size_t capacity)
{
    int *rows;
    double *coefficients;
    if (capacity <= workspace->substitution_active_capacity)
        return PREFOS_STATUS_OK;
    rows = (int *) prefos_internal_alloc_array(capacity, sizeof(int));
    coefficients = (double *)
        prefos_internal_alloc_array(capacity, sizeof(double));
    if (!rows || !coefficients)
    {
        free(rows);
        free(coefficients);
        return PREFOS_STATUS_OUT_OF_MEMORY;
    }
    free(workspace->substitution_active_rows);
    free(workspace->substitution_active_coefficients);
    workspace->substitution_active_rows = rows;
    workspace->substitution_active_coefficients = coefficients;
    workspace->substitution_active_capacity = capacity;
    return PREFOS_STATUS_OK;
}

PreFOSStatus prefos_internal_append_column_substitution(
    PreFOSPresolver *presolver, int column, const int *targets,
    const double *scales, size_t term_count, int source_row, double constant,
    double pivot, PreFOSColumnWorkspace *workspace,
    PreFOSSubstitutionMode mode, int eager_materialized,
    int source_is_only_active_row)
{
    PresolveColumnTransformationRecord record;
    int *active_rows;
    double *active_coefficients;
    double source_objective;
    double updated_objective_offset;
    size_t active_degree = 0, start, term;
    int creates_fill_in = 0;
    int residual_support_unchanged =
        source_is_only_active_row &&
        mode == PREFOS_SUBSTITUTION_RESIDUAL_ROW;
    int box_position = presolver->variable_to_box[column];
    PreFOSStatus status;
    if (term_count == 0 || box_position < 0)
        return PREFOS_STATUS_NUMERICAL_ERROR;
    status = reserve_active_column_scratch(
        workspace,
        source_is_only_active_row
            ? 1U
            : (size_t)
                  (workspace->ends[column] -
                   workspace->starts[column]));
    if (status != PREFOS_STATUS_OK) return status;
    active_rows = workspace->substitution_active_rows;
    active_coefficients = workspace->substitution_active_coefficients;
    source_objective = workspace->objective[column];
    updated_objective_offset = workspace->objective_offset;
    if (source_objective != 0.0 &&
        !prefos_internal_safe_add_product(
            &updated_objective_offset, source_objective, constant))
        return PREFOS_STATUS_NUMERICAL_ERROR;
    status = reserve_substitution_terms(presolver, term_count);
    if (status != PREFOS_STATUS_OK) return status;
    if (source_is_only_active_row)
    {
        if (source_row < 0 ||
            (size_t) source_row >= presolver->original.A.rows ||
            presolver->remove_rows[source_row] || pivot == 0.0 ||
            workspace->substitution_active_capacity == 0)
            return PREFOS_STATUS_NUMERICAL_ERROR;
        active_rows[0] = source_row;
        active_coefficients[0] = pivot;
        active_degree = 1;
    }
    else
        for (int position = workspace->starts[column];
             position < workspace->ends[column]; ++position)
        {
            int row = workspace->rows[position];
            if (presolver->remove_rows[row] ||
                workspace->values[position] == 0.0)
                continue;
            if (active_degree >=
                workspace->substitution_active_capacity)
                return PREFOS_STATUS_NUMERICAL_ERROR;
            active_rows[active_degree] = row;
            active_coefficients[active_degree] =
                workspace->values[position];
            if (row != source_row) creates_fill_in = 1;
            ++active_degree;
        }

    memset(&record, 0, sizeof(record));
    record.type = PRESOLVE_COLUMN_SUBSTITUTED;
    record.column = column;
    record.secondary_column = targets[0];
    record.source_row = source_row;
    record.direction = mode;
    record.value = constant;
    record.objective_coefficient = source_objective;
    record.rhs = pivot;
    record.ratio = scales[0];
    record.lower = presolver->working_box_lower[box_position];
    record.upper = presolver->working_box_upper[box_position];
    record.indices = active_rows;
    record.coefficients = active_coefficients;
    record.length = active_degree;
    if (mode == PREFOS_SUBSTITUTION_BOUNDED_DOUBLETON)
    {
        int target_box = presolver->variable_to_box[targets[0]];
        if (target_box < 0)
            return PREFOS_STATUS_NUMERICAL_ERROR;
        record.secondary_lower = presolver->working_box_lower[target_box];
        record.secondary_upper = presolver->working_box_upper[target_box];
    }
    if (!presolve_transformation_log_append_column_transformation(
            &presolver->transformations, &record, NULL))
        return PREFOS_STATUS_OUT_OF_MEMORY;

    start = presolver->n_substitution_terms;
    presolver->is_substituted[column] = 1;
    presolver->substitution_term_count[column] = term_count;
    presolver->substitution_term_start[column] = start;
    presolver->substitution_source_row[column] = source_row;
    presolver->substitution_keeps_source_row[column] =
        (unsigned char) (mode == PREFOS_SUBSTITUTION_RESIDUAL_ROW);
    presolver->substitution_constant[column] = constant;
    workspace->objective[column] = 0.0;
    workspace->objective_offset = updated_objective_offset;
    if (source_objective != 0.0)
        ++workspace->objective_change_epoch;
    for (term = 0; term < term_count; ++term)
    {
        int target = targets[term];
        uint16_t next_depth =
            (uint16_t)
                (presolver->substitution_incoming_depth[column] + 1);
        presolver->substitution_targets[start + term] = target;
        presolver->substitution_scales[start + term] = scales[term];
        if (presolver->substitution_incoming_depth[target] <
            next_depth)
            presolver->substitution_incoming_depth[target] =
                next_depth;
        if (creates_fill_in)
            presolver->substitution_fill_in_targets[target] = 1;
        if (source_objective != 0.0)
            prefos_internal_queue_dual_column(
                presolver, workspace, target);
        if (source_objective != 0.0 &&
            !add_objective_product(
                &workspace->objective[target],
                source_objective, scales[term]))
            return PREFOS_STATUS_NUMERICAL_ERROR;
        if (!residual_support_unchanged)
            prefos_internal_mark_parallel_column_dirty(
                workspace, target);
    }
    presolver->n_substitution_terms += term_count;
    for (term = 0; term < active_degree; ++term)
        if (active_rows[term] != source_row)
            prefos_internal_mark_row_requires_materialization(
                presolver, (size_t) active_rows[term]);
    if (!eager_materialized && !source_is_only_active_row)
        for (int position = workspace->starts[column];
             position < workspace->ends[column]; ++position)
        {
            int row = workspace->rows[position];
            if (row != source_row)
                prefos_internal_mark_workspace_row_dirty(
                    presolver, workspace, (size_t) row);
        }
    prefos_internal_mark_parallel_column_dirty(
        workspace, column);
    presolver->variable_to_box[column] = -1;
    if (mode != PREFOS_SUBSTITUTION_RESIDUAL_ROW)
        prefos_internal_mark_removed_row(
            presolver, (size_t) source_row);
    else
    {
        int *excluded =
            &presolver->residual_source_column[source_row];
        *excluded = *excluded == -1 ? column : -2;
        if (workspace->row_degrees[source_row] > 0)
            --workspace->row_degrees[source_row];
        if (term_count != 1 ||
            workspace->live_degrees[targets[0]] > 1 ||
            !prefos_internal_column_is_linear_box(
                presolver, workspace, targets[0]) ||
            residual_singleton_is_box_redundant(
                presolver, source_row, targets[0]))
            prefos_internal_queue_trivial_row(
                presolver, workspace, source_row);
        prefos_internal_linear_cache_mark_row_dirty(
            presolver, (size_t) source_row, 1);
        ++presolver->n_residual_row_substitutions;
        ++presolver->stats.residual_row_substitutions;
    }
    ++presolver->stats.substituted_free_variables;
    if (term_count == 2) ++presolver->stats.ternary_substituted_free_variables;
    if (mode == PREFOS_SUBSTITUTION_BOUNDED_DOUBLETON)
        ++presolver->stats.substituted_bounded_doubletons;
    else
        ++presolver->stats.removed_singleton_columns;
    return PREFOS_STATUS_OK;
}
