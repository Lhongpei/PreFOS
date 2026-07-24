/*
 * Copyright 2026 Hongpei Li
 * SPDX-License-Identifier: Apache-2.0
 */

#include "PREFOS_AffineConeAggregation.h"

typedef struct
{
    int source_row;
    double pivot;
    double constant;
    size_t term_count;
    int targets[PREFOS_MAX_AGGREGATION_TERMS];
    double scales[PREFOS_MAX_AGGREGATION_TERMS];
} PreFOSAffineCoordinateCandidate;

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

static int analyze_coordinate(const PreFOSPresolver *presolver, int column,
                              const int *column_degree, const int *column_row,
                              const unsigned char *quadratic_column,
                              const unsigned char *factor_column,
                              const unsigned char *affine_column,
                              PreFOSAffineCoordinateCandidate *candidate)
{
    const PreFOSProblemData *problem = &presolver->original;
    int row, p;
    long double scale;

    memset(candidate, 0, sizeof(*candidate));
    if (column < 0 || (size_t) column >= problem->n || column_degree[column] != 1 ||
        problem->c[column] != 0.0 || quadratic_column[column] ||
        factor_column[column] || affine_column[column])
        return 0;
    row = column_row[column];
    if (row < 0 || presolver->remove_rows[row] ||
        !isfinite(presolver->working_constraint_lower[row]) ||
        presolver->working_constraint_lower[row] !=
            presolver->working_constraint_upper[row])
        return 0;

    candidate->source_row = row;
    for (p = problem->A.row_pointers[row]; p < problem->A.row_pointers[row + 1]; ++p)
    {
        int target = problem->A.column_indices[p];
        double coefficient = problem->A.values[p];
        if (coefficient == 0.0) continue;
        if (target == column)
        {
            candidate->pivot = coefficient;
            continue;
        }
        if (candidate->term_count >=
                (size_t) presolver->settings.max_aggregation_terms ||
            presolver->variable_to_box[target] < 0 || presolver->is_fixed[target] ||
            presolver->is_substituted[target])
            return 0;
        candidate->targets[candidate->term_count] = target;
        candidate->scales[candidate->term_count] = coefficient;
        ++candidate->term_count;
    }
    if (candidate->pivot == 0.0) return 0;
    for (p = 0; p < (int) candidate->term_count; ++p)
    {
        scale = -(long double) candidate->scales[p] / (long double) candidate->pivot;
        if (!isfinite(scale) ||
            fabsl(scale) > (long double) presolver->settings.max_aggregation_scale)
            return 0;
        candidate->scales[p] = (double) scale;
    }
    {
        long double constant =
            (long double) presolver->working_constraint_lower[row] /
            (long double) candidate->pivot;
        if (!isfinite(constant) || fabsl(constant) > (long double) DBL_MAX) return 0;
        candidate->constant = (double) constant;
    }
    return 1;
}

static PreFOSStatus commit_coordinate(PreFOSPresolver *presolver, int column,
                                   const PreFOSAffineCoordinateCandidate *candidate)
{
    PresolveColumnTransformationRecord record;
    size_t start, term;
    PreFOSStatus status;
    presolver->affine_aggregation_source_rows[column] = candidate->source_row;
    presolver->affine_aggregation_pivots[column] = candidate->pivot;
    if (candidate->term_count == 0)
    {
        prefos_internal_mark_fixed_column(
            presolver, column, candidate->constant);
        prefos_internal_mark_removed_row(
            presolver, (size_t) candidate->source_row);
        ++presolver->stats.fixed_cone_variables;
        ++presolver->stats.aggregated_affine_cone_coordinates;
        return PREFOS_STATUS_OK;
    }
    status = reserve_substitution_terms(presolver, candidate->term_count);
    if (status != PREFOS_STATUS_OK) return status;

    memset(&record, 0, sizeof(record));
    record.type = PRESOLVE_COLUMN_SUBSTITUTED;
    record.column = column;
    record.secondary_column = candidate->targets[0];
    record.source_row = candidate->source_row;
    record.value = candidate->constant;
    record.rhs = candidate->pivot;
    record.ratio = candidate->scales[0];
    record.indices = (int *) &candidate->source_row;
    record.coefficients = (double *) &candidate->pivot;
    record.length = 1;
    if (!presolve_transformation_log_append_column_transformation(
            &presolver->transformations, &record, NULL))
        return PREFOS_STATUS_OUT_OF_MEMORY;

    start = presolver->n_substitution_terms;
    presolver->is_substituted[column] = 1;
    presolver->substitution_term_count[column] = candidate->term_count;
    presolver->substitution_term_start[column] = start;
    presolver->substitution_constant[column] = candidate->constant;
    for (term = 0; term < candidate->term_count; ++term)
    {
        int target = candidate->targets[term];
        presolver->substitution_targets[start + term] = target;
        presolver->substitution_scales[start + term] = candidate->scales[term];
        presolver->affine_protected_columns[target] = 1;
    }
    presolver->n_substitution_terms += candidate->term_count;
    prefos_internal_mark_removed_row(
        presolver, (size_t) candidate->source_row);
    ++presolver->stats.aggregated_affine_cone_coordinates;
    return PREFOS_STATUS_OK;
}

PreFOSStatus prefos_internal_aggregate_affine_cone_coordinates(PreFOSPresolver *presolver)
{
    const PreFOSProblemData *problem = &presolver->original;
    int *column_degree = NULL, *column_row = NULL;
    unsigned char *quadratic_column = NULL, *factor_column = NULL;
    unsigned char *affine_column = NULL;
    size_t row, k;
    PreFOSStatus status = PREFOS_STATUS_OK;

    if (!presolver->settings.affine_cone_coordinate_aggregation ||
        problem->n_cones == 0 || problem->A.rows == 0)
        return PREFOS_STATUS_OK;

    column_degree = (int *) calloc(problem->n, sizeof(int));
    column_row = (int *) prefos_internal_alloc_array(problem->n, sizeof(int));
    quadratic_column = (unsigned char *) calloc(problem->n, sizeof(unsigned char));
    factor_column = (unsigned char *) calloc(problem->n, sizeof(unsigned char));
    affine_column = (unsigned char *) calloc(problem->n, sizeof(unsigned char));
    if (problem->n > 0 && (!column_degree || !column_row || !quadratic_column ||
                           !factor_column || !affine_column))
    {
        status = PREFOS_STATUS_OUT_OF_MEMORY;
        goto cleanup;
    }
    for (row = 0; row < problem->n; ++row) column_row[row] = -1;
    for (row = 0; row < problem->A.rows; ++row)
    {
        int p;
        for (p = problem->A.row_pointers[row]; p < problem->A.row_pointers[row + 1];
             ++p)
        {
            int column = problem->A.column_indices[p];
            if (problem->A.values[p] == 0.0) continue;
            ++column_degree[column];
            column_row[column] = (int) row;
        }
    }
    for (row = 0; row < problem->Q.rows; ++row)
    {
        int p;
        for (p = problem->Q.row_pointers[row]; p < problem->Q.row_pointers[row + 1];
             ++p)
        {
            if (problem->Q.values[p] == 0.0) continue;
            quadratic_column[row] = 1;
            quadratic_column[problem->Q.column_indices[p]] = 1;
        }
    }
    for (row = 0; row < problem->R.rows; ++row)
    {
        int p;
        for (p = problem->R.row_pointers[row]; p < problem->R.row_pointers[row + 1];
             ++p)
            if (problem->R.values[p] != 0.0)
                factor_column[problem->R.column_indices[p]] = 1;
    }
    for (row = 0; row < problem->affine_cone_matrix.nnz; ++row)
        affine_column[problem->affine_cone_matrix.column_indices[row]] = 1;

    for (k = 0; k < problem->n_cones; ++k)
    {
        const PreFOSConeBlock *cone = &problem->cones[k];
        size_t coordinate;
        int eligible = 1;
        for (coordinate = 0; coordinate < cone->dimension; ++coordinate)
        {
            PreFOSAffineCoordinateCandidate candidate;
            if (!analyze_coordinate(presolver, cone->indices[coordinate],
                                    column_degree, column_row, quadratic_column,
                                    factor_column, affine_column, &candidate))
            {
                eligible = 0;
                break;
            }
        }
        if (!eligible) continue;
        for (coordinate = 0; coordinate < cone->dimension; ++coordinate)
        {
            PreFOSAffineCoordinateCandidate candidate;
            if (!analyze_coordinate(presolver, cone->indices[coordinate],
                                    column_degree, column_row, quadratic_column,
                                    factor_column, affine_column, &candidate))
            {
                status = PREFOS_STATUS_NUMERICAL_ERROR;
                goto cleanup;
            }
            status =
                commit_coordinate(presolver, cone->indices[coordinate], &candidate);
            if (status != PREFOS_STATUS_OK) goto cleanup;
        }
        presolver->converted_affine_cones[k] = 1;
        ++presolver->stats.generated_affine_cone_blocks;
    }

cleanup:
    free(column_degree);
    free(column_row);
    free(quadratic_column);
    free(factor_column);
    free(affine_column);
    return status;
}
