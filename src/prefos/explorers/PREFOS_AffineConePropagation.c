/*
 * Copyright 2026 Hongpei Li
 * SPDX-License-Identifier: Apache-2.0
 */

#include "PREFOS_AffineConePropagation.h"

#include "PREFOS_ConePropagation.h"
#include "PREFOS_LinearPropagation.h"

typedef struct
{
    const int *columns;
    const double *values;
    size_t count;
    double offset;
} PreFOSAffineEnvelopeRow;

typedef struct
{
    long double finite_min;
    long double finite_max;
    size_t infinite_min;
    size_t infinite_max;
} PreFOSAffineRowActivity;

static int affine_psd_exceeds_work_budget(const PreFOSPresolver *presolver,
                                          const PreFOSAffineConeBlock *block,
                                          size_t first_row)
{
    const PreFOSProblemData *problem = &presolver->original;
    const PreFOSCsrMatrix *matrix = &problem->affine_cone_matrix;
    double ratio = presolver->settings.affine_psd_propagation_max_work_ratio;
    size_t active_entries = 0, row;

    if (block->type != PREFOS_CONE_POSITIVE_SEMIDEFINITE || ratio == 0.0) return 0;
    if (matrix->row_pointers)
    {
        int first = matrix->row_pointers[first_row];
        int last = matrix->row_pointers[first_row + block->dimension];
        active_entries = (size_t) (last - first);
    }
    for (row = first_row; row < first_row + block->dimension; ++row)
        if (problem->affine_cone_offset[row] != 0.0) ++active_entries;
    if (active_entries == 0) active_entries = 1;
    return (long double) block->dimension >
           (long double) ratio * (long double) active_entries;
}

static void add_outward(long double *accumulator, long double value, int lower)
{
    long double result = *accumulator + value;
    *accumulator = nextafterl(result, lower ? -INFINITY : INFINITY);
}

static void compute_row_activity(const PreFOSPresolver *presolver,
                                 const PreFOSAffineEnvelopeRow *row,
                                 PreFOSAffineRowActivity *activity)
{
    size_t position;
    activity->finite_min = (long double) row->offset;
    activity->finite_max = (long double) row->offset;
    activity->infinite_min = 0;
    activity->infinite_max = 0;
    for (position = 0; position < row->count; ++position)
    {
        int column = row->columns[position];
        double coefficient = row->values[position];
        double lower, upper;
        long double minimum, maximum;
        if (coefficient == 0.0) continue;
        lower = presolver->propagation_lower[column];
        upper = presolver->propagation_upper[column];
        minimum = (long double) coefficient *
                  (long double) (coefficient > 0.0 ? lower : upper);
        maximum = (long double) coefficient *
                  (long double) (coefficient > 0.0 ? upper : lower);
        if (isfinite(minimum))
            add_outward(&activity->finite_min, minimum, 1);
        else
            ++activity->infinite_min;
        if (isfinite(maximum))
            add_outward(&activity->finite_max, maximum, 0);
        else
            ++activity->infinite_max;
    }
}

static void row_interval(const PreFOSAffineRowActivity *activity, double *lower,
                         double *upper)
{
    *lower = activity->infinite_min > 0
                 ? -INFINITY
                 : prefos_internal_outward_bound_cast(activity->finite_min, 1);
    *upper = activity->infinite_max > 0
                 ? INFINITY
                 : prefos_internal_outward_bound_cast(activity->finite_max, 0);
}

static PreFOSStatus update_variable_envelope(PreFOSPresolver *presolver, int column,
                                          double candidate, int is_lower,
                                          int *changed)
{
    double current = is_lower ? presolver->propagation_lower[column]
                              : presolver->propagation_upper[column];
    double opposite = is_lower ? presolver->propagation_upper[column]
                               : presolver->propagation_lower[column];
    long double margin;
    if ((is_lower && candidate == -INFINITY) || (!is_lower && candidate == INFINITY))
        return PREFOS_STATUS_OK;
    if ((is_lower && candidate == INFINITY) ||
        (!is_lower && candidate == -INFINITY) || isnan(candidate))
        return PREFOS_STATUS_PRIMAL_INFEASIBLE;
    margin = prefos_internal_propagation_margin(presolver, (long double) opposite);
    if (isfinite(opposite) &&
        ((is_lower && (long double) candidate > (long double) opposite + margin) ||
         (!is_lower && (long double) candidate < (long double) opposite - margin)))
        return PREFOS_STATUS_PRIMAL_INFEASIBLE;
    if (!prefos_internal_is_significant_improvement(presolver, current, candidate,
                                                 is_lower))
        return PREFOS_STATUS_OK;
    if (is_lower)
    {
        if (candidate > opposite) candidate = opposite;
        presolver->propagation_lower[column] = candidate;
    }
    else
    {
        if (candidate < opposite) candidate = opposite;
        presolver->propagation_upper[column] = candidate;
    }
    ++presolver->stats.tightened_affine_variable_envelopes;
    *changed = 1;
    return PREFOS_STATUS_OK;
}

static int activity_without_term(const PreFOSAffineRowActivity *activity,
                                 long double term, int lower, long double *result)
{
    size_t infinite = lower ? activity->infinite_min : activity->infinite_max;
    long double finite = lower ? activity->finite_min : activity->finite_max;
    if (isfinite(term))
    {
        if (infinite > 0) return 0;
        *result = finite - term;
    }
    else
    {
        if (infinite != 1) return 0;
        *result = finite;
    }
    if (!isfinite(*result)) return 0;
    *result = nextafterl(*result, lower ? -INFINITY : INFINITY);
    return 1;
}

static PreFOSStatus propagate_row_to_variables(PreFOSPresolver *presolver,
                                            const PreFOSAffineEnvelopeRow *row,
                                            const PreFOSAffineRowActivity *activity,
                                            double coordinate_lower,
                                            double coordinate_upper, int *changed)
{
    size_t position;
    for (position = 0; position < row->count; ++position)
    {
        int column = row->columns[position];
        double coefficient = row->values[position];
        double variable_lower = presolver->propagation_lower[column];
        double variable_upper = presolver->propagation_upper[column];
        long double term_min, term_max, other_min, other_max, implied;
        PreFOSStatus status;
        if (coefficient == 0.0) continue;
        term_min =
            (long double) coefficient *
            (long double) (coefficient > 0.0 ? variable_lower : variable_upper);
        term_max =
            (long double) coefficient *
            (long double) (coefficient > 0.0 ? variable_upper : variable_lower);
        if (isfinite(coordinate_lower) &&
            activity_without_term(activity, term_max, 0, &other_max))
        {
            implied = ((long double) coordinate_lower - other_max) /
                      (long double) coefficient;
            status = update_variable_envelope(
                presolver, column,
                prefos_internal_outward_bound_cast(implied, coefficient > 0.0),
                coefficient > 0.0, changed);
            if (status != PREFOS_STATUS_OK) return status;
        }
        if (isfinite(coordinate_upper) &&
            activity_without_term(activity, term_min, 1, &other_min))
        {
            implied = ((long double) coordinate_upper - other_min) /
                      (long double) coefficient;
            status = update_variable_envelope(
                presolver, column,
                prefos_internal_outward_bound_cast(implied, coefficient < 0.0),
                coefficient < 0.0, changed);
            if (status != PREFOS_STATUS_OK) return status;
        }
    }
    return PREFOS_STATUS_OK;
}

static PreFOSStatus propagate_block(PreFOSPresolver *presolver,
                                 const PreFOSAffineConeBlock *block,
                                 const PreFOSAffineEnvelopeRow *rows, double *lower,
                                 double *upper, int *indices, int *changed)
{
    PreFOSConeBlock cone;
    size_t coordinate;
    size_t tightened_before = presolver->stats.tightened_cone_envelopes;
    PreFOSStatus status;
    int block_changed = 0;
    for (coordinate = 0; coordinate < block->dimension; ++coordinate)
    {
        PreFOSAffineRowActivity activity;
        compute_row_activity(presolver, &rows[coordinate], &activity);
        row_interval(&activity, &lower[coordinate], &upper[coordinate]);
        indices[coordinate] = (int) coordinate;
    }
    cone = (PreFOSConeBlock){block->type, block->dimension, block->matrix_order,
                          indices, block->power_alpha};
    status = prefos_internal_propagate_cone_block_envelopes(presolver, &cone, lower,
                                                         upper, &block_changed);
    if (status != PREFOS_STATUS_OK) return status;
    presolver->stats.tightened_affine_cone_envelopes +=
        presolver->stats.tightened_cone_envelopes - tightened_before;
    for (coordinate = 0; coordinate < block->dimension; ++coordinate)
    {
        PreFOSAffineRowActivity activity;
        compute_row_activity(presolver, &rows[coordinate], &activity);
        status = propagate_row_to_variables(presolver, &rows[coordinate], &activity,
                                            lower[coordinate], upper[coordinate],
                                            changed);
        if (status != PREFOS_STATUS_OK) return status;
    }
    ++presolver->stats.affine_cones_processed;
    return PREFOS_STATUS_OK;
}

PreFOSStatus prefos_internal_propagate_affine_cones(PreFOSPresolver *presolver, int *changed)
{
    const PreFOSProblemData *problem = &presolver->original;
    PreFOSAffineEnvelopeRow *rows = NULL;
    double *lower = NULL, *upper = NULL;
    int *indices = NULL;
    size_t maximum_dimension = 0, affine_row = 0, k;
    int skipped_affine_psd = 0;
    PreFOSStatus status = PREFOS_STATUS_OK;

    for (k = 0; k < problem->n_affine_cones; ++k)
    {
        const PreFOSAffineConeBlock *block = &problem->affine_cones[k];
        if (affine_psd_exceeds_work_budget(presolver, block, affine_row))
        {
            skipped_affine_psd = 1;
            ++presolver->stats.affine_psd_budget_skips;
            presolver->stats.affine_psd_coordinates_skipped += block->dimension;
        }
        else if (block->dimension > maximum_dimension)
            maximum_dimension = block->dimension;
        affine_row += block->dimension;
    }
    for (k = 0; k < problem->n_cones; ++k)
        if (presolver->converted_affine_cones[k] &&
            problem->cones[k].dimension > maximum_dimension)
            maximum_dimension = problem->cones[k].dimension;
    if (maximum_dimension == 0)
    {
        if (skipped_affine_psd)
            ++presolver->stats.affine_cone_propagation_rounds;
        return PREFOS_STATUS_OK;
    }

    rows = (PreFOSAffineEnvelopeRow *) prefos_internal_alloc_array(
        maximum_dimension, sizeof(PreFOSAffineEnvelopeRow));
    lower = (double *) prefos_internal_alloc_array(maximum_dimension, sizeof(double));
    upper = (double *) prefos_internal_alloc_array(maximum_dimension, sizeof(double));
    indices = (int *) prefos_internal_alloc_array(maximum_dimension, sizeof(int));
    if (!rows || !lower || !upper || !indices)
    {
        status = PREFOS_STATUS_OUT_OF_MEMORY;
        goto cleanup;
    }

    affine_row = 0;
    for (k = 0; k < problem->n_affine_cones; ++k)
    {
        const PreFOSAffineConeBlock *block = &problem->affine_cones[k];
        size_t coordinate;
        if (affine_psd_exceeds_work_budget(presolver, block, affine_row))
        {
            affine_row += block->dimension;
            continue;
        }
        for (coordinate = 0; coordinate < block->dimension; ++coordinate)
        {
            size_t row = affine_row + coordinate;
            int start = problem->affine_cone_matrix.row_pointers[row];
            int end = problem->affine_cone_matrix.row_pointers[row + 1];
            rows[coordinate] = (PreFOSAffineEnvelopeRow){
                problem->affine_cone_matrix.column_indices + start,
                problem->affine_cone_matrix.values + start, (size_t) (end - start),
                problem->affine_cone_offset[row]};
        }
        status =
            propagate_block(presolver, block, rows, lower, upper, indices, changed);
        if (status != PREFOS_STATUS_OK) goto cleanup;
        affine_row += block->dimension;
    }
    if (affine_row != problem->affine_cone_matrix.rows)
    {
        status = PREFOS_STATUS_NUMERICAL_ERROR;
        goto cleanup;
    }

    for (k = 0; k < problem->n_cones; ++k)
    {
        const PreFOSConeBlock *direct = &problem->cones[k];
        PreFOSAffineConeBlock block;
        size_t coordinate;
        if (!presolver->converted_affine_cones[k]) continue;
        for (coordinate = 0; coordinate < direct->dimension; ++coordinate)
        {
            int column = direct->indices[coordinate];
            if (presolver->is_fixed[column])
                rows[coordinate] = (PreFOSAffineEnvelopeRow){
                    NULL, NULL, 0, presolver->fixed_values[column]};
            else if (presolver->is_substituted[column])
            {
                size_t start = presolver->substitution_term_start[column];
                rows[coordinate] = (PreFOSAffineEnvelopeRow){
                    presolver->substitution_targets + start,
                    presolver->substitution_scales + start,
                    presolver->substitution_term_count[column],
                    presolver->substitution_constant[column]};
            }
            else
            {
                status = PREFOS_STATUS_NUMERICAL_ERROR;
                goto cleanup;
            }
        }
        block = (PreFOSAffineConeBlock){direct->type, direct->dimension,
                                     direct->matrix_order, direct->power_alpha};
        status =
            propagate_block(presolver, &block, rows, lower, upper, indices, changed);
        if (status != PREFOS_STATUS_OK) goto cleanup;
    }
    ++presolver->stats.affine_cone_propagation_rounds;

cleanup:
    free(rows);
    free(lower);
    free(upper);
    free(indices);
    return status;
}
