/*
 * Copyright 2026 Hongpei Li
 * SPDX-License-Identifier: Apache-2.0
 */

#include "PREFOS_AffineFaceSubstitutionInternal.h"

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

int prefos_affine_face_choose_pivot(
    const PreFOSPresolver *presolver, const PreFOSColumnStorage *storage,
    const unsigned char *quadratic_column, const unsigned char *factor_column,
    const PreFOSExpandedAffineRow *row)
{
    const PreFOSProblemData *problem = &presolver->original;
    int best = -1;
    size_t best_degree = SIZE_MAX, position;
    long double best_pivot = 0.0L;
    size_t nonzeros = 0;
    for (position = 0; position < row->count; ++position)
        if (row->values[row->touched[position]] != 0.0) ++nonzeros;
    if (nonzeros == 0 ||
        nonzeros > (size_t) presolver->settings.max_aggregation_terms + 1)
        return -1;
    for (position = 0; position < row->count; ++position)
    {
        int column = row->touched[position];
        int box = presolver->variable_to_box[column];
        double coefficient = row->values[column];
        size_t degree, target_position;
        int valid = 1;
        if (coefficient == 0.0 || box < 0 || presolver->is_fixed[column] ||
            presolver->is_substituted[column] ||
            presolver->affine_face_substitution_targets[column] ||
            quadratic_column[column] || factor_column[column] ||
            problem->box_lower[box] != -INFINITY ||
            problem->box_upper[box] != INFINITY)
            continue;
        degree = prefos_affine_face_active_column_degree(presolver, storage, column);
        if (degree >
            (size_t) presolver->settings.max_aggregation_column_degree)
            continue;
        for (target_position = 0; target_position < row->count; ++target_position)
        {
            int target = row->touched[target_position];
            long double scale;
            if (target == column || row->values[target] == 0.0) continue;
            scale = -(long double) row->values[target] /
                    (long double) coefficient;
            if (!isfinite(scale) ||
                fabsl(scale) >
                    (long double) presolver->settings.max_aggregation_scale)
            {
                valid = 0;
                break;
            }
        }
        if (!valid) continue;
        if (degree < best_degree ||
            (degree == best_degree &&
             fabsl((long double) coefficient) > best_pivot) ||
            (degree == best_degree &&
             fabsl((long double) coefficient) == best_pivot && column < best))
        {
            best = column;
            best_degree = degree;
            best_pivot = fabsl((long double) coefficient);
        }
    }
    return best;
}

PreFOSStatus prefos_affine_face_append_transformation(
    PreFOSPresolver *presolver, const PreFOSColumnStorage *storage,
    const PreFOSColumnStorage *affine_storage,
    const PreFOSExpandedAffineRow *row, size_t affine_row, int pivot)
{
    PresolveColumnTransformationRecord record;
    int targets[PREFOS_MAX_AGGREGATION_TERMS];
    double scales[PREFOS_MAX_AGGREGATION_TERMS];
    int *active_rows = NULL, *affine_rows = NULL;
    double *active_coefficients = NULL, *affine_coefficients = NULL;
    double pivot_coefficient = row->values[pivot];
    long double constant;
    size_t term_count = 0, active_degree = 0, position;
    size_t affine_degree;
    PreFOSStatus status;

    if (affine_row >= (size_t) INT_MAX || pivot_coefficient == 0.0)
        return PREFOS_STATUS_NUMERICAL_ERROR;
    constant = -(long double) row->constant / (long double) pivot_coefficient;
    if (!isfinite(constant) || fabsl(constant) > (long double) DBL_MAX)
        return PREFOS_STATUS_NUMERICAL_ERROR;
    for (position = 0; position < row->count; ++position)
    {
        int target = row->touched[position];
        long double scale;
        if (target == pivot || row->values[target] == 0.0) continue;
        if (term_count >= PREFOS_MAX_AGGREGATION_TERMS)
            return PREFOS_STATUS_NUMERICAL_ERROR;
        scale = -(long double) row->values[target] /
                (long double) pivot_coefficient;
        if (!isfinite(scale) || fabsl(scale) > (long double) DBL_MAX)
            return PREFOS_STATUS_NUMERICAL_ERROR;
        targets[term_count] = target;
        scales[term_count++] = (double) scale;
    }
    for (position = (size_t) storage->starts[pivot];
         position < (size_t) storage->starts[pivot + 1]; ++position)
        if (!presolver->remove_rows[storage->rows[position]]) ++active_degree;
    active_rows =
        (int *) prefos_internal_alloc_array(active_degree, sizeof(int));
    active_coefficients =
        (double *) prefos_internal_alloc_array(active_degree, sizeof(double));
    if (active_degree > 0 && (!active_rows || !active_coefficients))
    {
        free(active_rows);
        free(active_coefficients);
        return PREFOS_STATUS_OUT_OF_MEMORY;
    }
    active_degree = 0;
    for (position = (size_t) storage->starts[pivot];
         position < (size_t) storage->starts[pivot + 1]; ++position)
    {
        int source_row = storage->rows[position];
        if (presolver->remove_rows[source_row]) continue;
        active_rows[active_degree] = source_row;
        active_coefficients[active_degree++] = storage->coefficients[position];
    }
    affine_degree = (size_t) (affine_storage->starts[pivot + 1] -
                              affine_storage->starts[pivot]);
    affine_rows =
        (int *) prefos_internal_alloc_array(affine_degree, sizeof(int));
    affine_coefficients =
        (double *) prefos_internal_alloc_array(affine_degree, sizeof(double));
    if (affine_degree > 0 && (!affine_rows || !affine_coefficients))
    {
        free(active_rows);
        free(active_coefficients);
        free(affine_rows);
        free(affine_coefficients);
        return PREFOS_STATUS_OUT_OF_MEMORY;
    }
    for (position = 0; position < affine_degree; ++position)
    {
        size_t source = (size_t) affine_storage->starts[pivot] + position;
        affine_rows[position] = affine_storage->rows[source];
        affine_coefficients[position] = affine_storage->coefficients[source];
    }

    if (term_count > 0)
    {
        status = reserve_substitution_terms(presolver, term_count);
        if (status != PREFOS_STATUS_OK)
        {
            free(active_rows);
            free(active_coefficients);
            free(affine_rows);
            free(affine_coefficients);
            return status;
        }
    }
    memset(&record, 0, sizeof(record));
    record.type = term_count == 0 ? PRESOLVE_COLUMN_FIXED
                                  : PRESOLVE_COLUMN_SUBSTITUTED;
    record.column = pivot;
    record.secondary_column = term_count == 0 ? -1 : targets[0];
    record.source_row = -1;
    record.column_tag = -(int) (affine_row + 1);
    record.value = (double) constant;
    record.objective_coefficient = presolver->original.c[pivot];
    record.rhs = pivot_coefficient;
    record.ratio = term_count == 0 ? 0.0 : scales[0];
    record.indices = active_rows;
    record.coefficients = active_coefficients;
    record.length = active_degree;
    record.affine_indices = affine_rows;
    record.affine_coefficients = affine_coefficients;
    record.affine_length = affine_degree;
    if (!presolve_transformation_log_append_column_transformation(
            &presolver->transformations, &record, NULL))
    {
        free(active_rows);
        free(active_coefficients);
        free(affine_rows);
        free(affine_coefficients);
        return PREFOS_STATUS_OUT_OF_MEMORY;
    }
    free(active_rows);
    free(active_coefficients);
    free(affine_rows);
    free(affine_coefficients);

    if (term_count == 0)
    {
        presolver->is_fixed[pivot] = 1;
        presolver->fixed_values[pivot] = (double) constant;
        ++presolver->stats.fixed_affine_face_variables;
    }
    else
    {
        size_t start = presolver->n_substitution_terms;
        unsigned char next_depth =
            (unsigned char) (presolver->substitution_incoming_depth[pivot] + 1);
        presolver->is_substituted[pivot] = 1;
        presolver->substitution_term_count[pivot] = (unsigned char) term_count;
        presolver->substitution_term_start[pivot] = start;
        presolver->substitution_constant[pivot] = (double) constant;
        for (position = 0; position < term_count; ++position)
        {
            int target = targets[position];
            presolver->substitution_targets[start + position] = target;
            presolver->substitution_scales[start + position] = scales[position];
            presolver->affine_face_substitution_targets[target] = 1;
            presolver->affine_protected_columns[target] = 1;
            if (presolver->substitution_incoming_depth[target] < next_depth)
                presolver->substitution_incoming_depth[target] = next_depth;
        }
        presolver->n_substitution_terms += term_count;
        presolver->variable_to_box[pivot] = -1;
        ++presolver->stats.substituted_free_variables;
        ++presolver->stats.substituted_affine_face_variables;
        if (term_count == 2)
            ++presolver->stats.ternary_substituted_free_variables;
    }
    presolver->affine_face_eliminated_columns[pivot] = 1;
    ++presolver->stats.derived_affine_face_equalities;
    ++presolver->n_affine_face_substitutions;
    return PREFOS_STATUS_OK;
}
