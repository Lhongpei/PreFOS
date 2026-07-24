/*
 * Copyright 2026 Hongpei Li
 * SPDX-License-Identifier: Apache-2.0
 */

#include "PREFOS_ColumnReductionInternal.h"

#define PREFOS_MAX_RECORDED_COLUMN_DEGREE 8

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
    targets = (int *) prefos_internal_alloc_array(capacity, sizeof(int));
    scales = (double *) prefos_internal_alloc_array(capacity, sizeof(double));
    if (!targets || !scales)
    {
        free(targets);
        free(scales);
        return PREFOS_STATUS_OUT_OF_MEMORY;
    }
    if (presolver->n_substitution_terms > 0)
    {
        memcpy(targets, presolver->substitution_targets,
               presolver->n_substitution_terms * sizeof(int));
        memcpy(scales, presolver->substitution_scales,
               presolver->n_substitution_terms * sizeof(double));
    }
    free(presolver->substitution_targets);
    free(presolver->substitution_scales);
    presolver->substitution_targets = targets;
    presolver->substitution_scales = scales;
    presolver->substitution_term_capacity = capacity;
    return PREFOS_STATUS_OK;
}

PreFOSStatus prefos_internal_append_column_substitution(
    PreFOSPresolver *presolver, int column, const int *targets,
    const double *scales, size_t term_count, int source_row, double constant,
    double pivot, PreFOSColumnWorkspace *workspace,
    PreFOSSubstitutionMode mode)
{
    PresolveColumnTransformationRecord record;
    int active_rows[PREFOS_MAX_RECORDED_COLUMN_DEGREE];
    double active_coefficients[PREFOS_MAX_RECORDED_COLUMN_DEGREE];
    double *updated_objective = NULL;
    double source_objective;
    size_t active_degree = 0, start, term;
    int box_position = presolver->variable_to_box[column];
    PreFOSStatus status;
    if (term_count == 0 || box_position < 0)
        return PREFOS_STATUS_NUMERICAL_ERROR;
    source_objective = workspace->objective[column];
    if (source_objective != 0.0)
    {
        updated_objective = (double *)
            prefos_internal_alloc_array(term_count, sizeof(double));
        if (!updated_objective) return PREFOS_STATUS_OUT_OF_MEMORY;
    }
    status = reserve_substitution_terms(presolver, term_count);
    if (status != PREFOS_STATUS_OK)
    {
        free(updated_objective);
        return status;
    }
    for (term = 0; updated_objective && term < term_count; ++term)
    {
        updated_objective[term] = workspace->objective[targets[term]];
        if (!prefos_internal_safe_add_product(
                &updated_objective[term], source_objective,
                scales[term]))
        {
            free(updated_objective);
            return PREFOS_STATUS_NUMERICAL_ERROR;
        }
    }
    for (int position = workspace->starts[column];
         position < workspace->ends[column]; ++position)
    {
        int row = workspace->rows[position];
        if (presolver->remove_rows[row]) continue;
        if (active_degree >= PREFOS_MAX_RECORDED_COLUMN_DEGREE)
        {
            free(updated_objective);
            return PREFOS_STATUS_NUMERICAL_ERROR;
        }
        active_rows[active_degree] = row;
        active_coefficients[active_degree] = workspace->values[position];
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
        {
            free(updated_objective);
            return PREFOS_STATUS_NUMERICAL_ERROR;
        }
        record.secondary_lower = presolver->working_box_lower[target_box];
        record.secondary_upper = presolver->working_box_upper[target_box];
    }
    if (!presolve_transformation_log_append_column_transformation(
            &presolver->transformations, &record, NULL))
    {
        free(updated_objective);
        return PREFOS_STATUS_OUT_OF_MEMORY;
    }

    start = presolver->n_substitution_terms;
    presolver->is_substituted[column] = 1;
    presolver->substitution_term_count[column] = term_count;
    presolver->substitution_term_start[column] = start;
    presolver->substitution_source_row[column] = source_row;
    presolver->substitution_keeps_source_row[column] =
        (unsigned char) (mode == PREFOS_SUBSTITUTION_RESIDUAL_ROW);
    presolver->substitution_constant[column] = constant;
    for (term = 0; term < term_count; ++term)
    {
        unsigned char next_depth =
            (unsigned char)
                (presolver->substitution_incoming_depth[column] + 1);
        presolver->substitution_targets[start + term] = targets[term];
        presolver->substitution_scales[start + term] = scales[term];
        if (presolver->substitution_incoming_depth[targets[term]] <
            next_depth)
            presolver->substitution_incoming_depth[targets[term]] =
                next_depth;
    }
    presolver->n_substitution_terms += term_count;
    for (int position = workspace->starts[column];
         position < workspace->ends[column]; ++position)
    {
        int row = workspace->rows[position];
        if (mode != PREFOS_SUBSTITUTION_RESIDUAL_ROW ||
            row != source_row)
            workspace->dirty_row[row] = 1;
    }
    workspace->objective[column] = 0.0;
    for (term = 0; updated_objective && term < term_count; ++term)
        workspace->objective[targets[term]] = updated_objective[term];
    free(updated_objective);
    presolver->variable_to_box[column] = -1;
    if (mode != PREFOS_SUBSTITUTION_RESIDUAL_ROW)
        prefos_internal_mark_removed_row(
            presolver, (size_t) source_row);
    else
    {
        int *excluded =
            &presolver->residual_source_column[source_row];
        *excluded = *excluded == -1 ? column : -2;
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
