/*
 * Copyright 2026 Hongpei Li
 * SPDX-License-Identifier: Apache-2.0
 */

#include "PREFOS_ColumnReductionInternal.h"

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
    double updated_objective[PREFOS_MAX_AGGREGATION_TERMS];
    size_t start, term;
    int box_position = presolver->variable_to_box[column];
    PreFOSStatus status = reserve_substitution_terms(presolver, term_count);
    if (status != PREFOS_STATUS_OK) return status;
    if (term_count == 0 || term_count > PREFOS_MAX_AGGREGATION_TERMS ||
        box_position < 0)
        return PREFOS_STATUS_NUMERICAL_ERROR;
    for (term = 0; term < term_count; ++term)
    {
        updated_objective[term] = workspace->objective[targets[term]];
        if (!prefos_internal_safe_add_product(
                &updated_objective[term], workspace->objective[column],
                scales[term]))
            return PREFOS_STATUS_NUMERICAL_ERROR;
    }

    memset(&record, 0, sizeof(record));
    record.type = PRESOLVE_COLUMN_SUBSTITUTED;
    record.column = column;
    record.secondary_column = targets[0];
    record.source_row = source_row;
    record.direction = mode;
    record.value = constant;
    record.objective_coefficient = workspace->objective[column];
    record.rhs = pivot;
    record.ratio = scales[0];
    record.lower = presolver->working_box_lower[box_position];
    record.upper = presolver->working_box_upper[box_position];
    record.indices = workspace->rows + workspace->starts[column];
    record.coefficients = workspace->values + workspace->starts[column];
    record.length =
        (size_t) (workspace->starts[column + 1] - workspace->starts[column]);
    if (mode == PREFOS_SUBSTITUTION_BOUNDED_DOUBLETON)
    {
        int target_box = presolver->variable_to_box[targets[0]];
        if (target_box < 0) return PREFOS_STATUS_NUMERICAL_ERROR;
        record.secondary_lower = presolver->working_box_lower[target_box];
        record.secondary_upper = presolver->working_box_upper[target_box];
    }
    if (!presolve_transformation_log_append_column_transformation(
            &presolver->transformations, &record, NULL))
        return PREFOS_STATUS_OUT_OF_MEMORY;

    start = presolver->n_substitution_terms;
    presolver->is_substituted[column] = 1;
    presolver->substitution_term_count[column] = (unsigned char) term_count;
    presolver->substitution_term_start[column] = start;
    presolver->substitution_source_row[column] = source_row;
    presolver->substitution_keeps_source_row[column] =
        (unsigned char) (mode == PREFOS_SUBSTITUTION_RESIDUAL_ROW);
    presolver->substitution_constant[column] = constant;
    for (term = 0; term < term_count; ++term)
    {
        presolver->substitution_targets[start + term] = targets[term];
        presolver->substitution_scales[start + term] = scales[term];
    }
    presolver->n_substitution_terms += term_count;
    workspace->objective[column] = 0.0;
    for (term = 0; term < term_count; ++term)
        workspace->objective[targets[term]] = updated_objective[term];
    presolver->variable_to_box[column] = -1;
    if (mode != PREFOS_SUBSTITUTION_RESIDUAL_ROW)
        presolver->remove_rows[source_row] = 1;
    ++presolver->stats.substituted_free_variables;
    if (term_count == 2) ++presolver->stats.ternary_substituted_free_variables;
    if (mode == PREFOS_SUBSTITUTION_BOUNDED_DOUBLETON)
        ++presolver->stats.substituted_bounded_doubletons;
    else
        ++presolver->stats.removed_singleton_columns;
    return PREFOS_STATUS_OK;
}
