/*
 * Copyright 2026 Hongpei Li
 * SPDX-License-Identifier: Apache-2.0
 */

#include "PREFOS_TrivialReductions.h"
#include "PREFOS_ColumnReductionInternal.h"

static int close_bounds(double lower, double upper, double tolerance)
{
    long double difference, scale;
    double quick_difference, quick_scale, quick_threshold;
    if (!isfinite(lower) || !isfinite(upper)) return 0;
    quick_difference = upper - lower;
    quick_scale = fmax(1.0, fmax(fabs(lower), fabs(upper)));
    quick_threshold = tolerance * quick_scale;
    if (isfinite(quick_difference) &&
        quick_difference > 2.0 * quick_threshold)
        return 0;
    difference = (long double) upper - (long double) lower;
    scale =
        fmaxl(1.0L, fmaxl(fabsl((long double) lower), fabsl((long double) upper)));
    return difference <= (long double) tolerance * scale;
}

static PreFOSStatus append_deleted_row(PreFOSPresolver *presolver, int row)
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

PreFOSStatus prefos_internal_find_fixed_box_variables(
    PreFOSPresolver *presolver, size_t *n_fixed)
{
    const PreFOSProblemData *problem = &presolver->original;
    size_t box;

    if (!n_fixed) return PREFOS_STATUS_INVALID_ARGUMENT;
    *n_fixed = 0;
    if (!presolver->settings.fix_close_box_bounds) return PREFOS_STATUS_OK;
    for (box = 0; box < problem->n_box; ++box)
    {
        int column = problem->box_indices[box];
        if (presolver->is_fixed[column] ||
            presolver->is_substituted[column] ||
            presolver->is_parallel_removed[column] ||
            !close_bounds(presolver->working_box_lower[box],
                          presolver->working_box_upper[box],
                          presolver->settings.fixed_variable_tolerance))
            continue;
        prefos_internal_mark_fixed_column(
            presolver, column,
            prefos_internal_safe_midpoint(
                presolver->working_box_lower[box],
                presolver->working_box_upper[box]));
        ++(*n_fixed);
    }
    return PREFOS_STATUS_OK;
}

static PreFOSStatus reduce_trivial_row(
    PreFOSPresolver *presolver, size_t row)
{
    const PreFOSProblemData *problem = &presolver->original;
    const PreFOSCsrMatrix *matrix = &problem->A;
    const double tolerance = presolver->settings.feasibility_tolerance;
    long double fixed_shift = 0.0L;
    size_t live = 0;
    int live_column = -1;
    double live_coefficient = 0.0;
    int unsupported_elimination = 0;
    double lower, upper;
    int p;

    if (row >= matrix->rows) return PREFOS_STATUS_INVALID_ARGUMENT;
    if (presolver->remove_rows[row]) return PREFOS_STATUS_OK;
    for (p = matrix->row_pointers[row];
         p < matrix->row_pointers[row + 1]; ++p)
    {
        int column = matrix->column_indices[p];
        double coefficient = matrix->values[p];
        if (coefficient == 0.0) continue;
        if (presolver->is_fixed[column])
        {
            long double term =
                (long double) coefficient *
                (long double) presolver->fixed_values[column];
            if (!isfinite(term) || !isfinite(fixed_shift + term))
                return PREFOS_STATUS_NUMERICAL_ERROR;
            fixed_shift += term;
            continue;
        }
        if (presolver->is_substituted[column])
        {
            if (!prefos_internal_term_is_active_in_row(
                    presolver, row, column))
                continue;
            unsupported_elimination = 1;
            continue;
        }
        if (presolver->is_parallel_removed[column])
        {
            unsupported_elimination = 1;
            continue;
        }
        ++live;
        if (live == 1)
        {
            live_column = column;
            live_coefficient = coefficient;
        }
    }
    if (unsupported_elimination) return PREFOS_STATUS_OK;
    lower = (double)
        ((long double) presolver->working_constraint_lower[row] -
         fixed_shift);
    upper = (double)
        ((long double) presolver->working_constraint_upper[row] -
         fixed_shift);
    if (isnan(lower) || isnan(upper))
        return PREFOS_STATUS_NUMERICAL_ERROR;

    if (live == 0)
    {
        PreFOSStatus status;
        if (lower > tolerance || upper < -tolerance)
            return PREFOS_STATUS_PRIMAL_INFEASIBLE;
        if (!presolver->settings.remove_empty_rows) return PREFOS_STATUS_OK;
        status = append_deleted_row(presolver, (int) row);
        if (status != PREFOS_STATUS_OK) return status;
        prefos_internal_mark_removed_row(presolver, row);
        ++presolver->stats.removed_empty_rows;
        return PREFOS_STATUS_OK;
    }
    if (live != 1) return PREFOS_STATUS_OK;
    if (!isfinite(lower) && !isfinite(upper))
    {
        PreFOSStatus status;
        if (!presolver->settings.remove_redundant_rows)
            return PREFOS_STATUS_OK;
        status = append_deleted_row(presolver, (int) row);
        if (status != PREFOS_STATUS_OK) return status;
        prefos_internal_mark_removed_row(presolver, row);
        ++presolver->stats.removed_redundant_rows;
        return PREFOS_STATUS_OK;
    }
    {
        int box_position = presolver->variable_to_box[live_column];
        double implied_lower, implied_upper;
        double old_lower, old_upper, new_lower, new_upper;
        PreFOSStatus status;

        if (box_position < 0 || live_coefficient == 0.0)
            return PREFOS_STATUS_OK;
        if (live_coefficient > 0.0)
        {
            implied_lower = prefos_internal_outward_bound_cast(
                (long double) lower / live_coefficient, 1);
            implied_upper = prefos_internal_outward_bound_cast(
                (long double) upper / live_coefficient, 0);
        }
        else
        {
            implied_lower = prefos_internal_outward_bound_cast(
                (long double) upper / live_coefficient, 1);
            implied_upper = prefos_internal_outward_bound_cast(
                (long double) lower / live_coefficient, 0);
        }
        if (implied_lower == INFINITY || implied_upper == -INFINITY)
            return PREFOS_STATUS_PRIMAL_INFEASIBLE;
        old_lower = presolver->working_box_lower[box_position];
        old_upper = presolver->working_box_upper[box_position];
        new_lower = fmax(old_lower, implied_lower);
        new_upper = fmin(old_upper, implied_upper);
        if (new_lower > new_upper + tolerance)
            return PREFOS_STATUS_PRIMAL_INFEASIBLE;
        if (new_lower > new_upper)
        {
            double midpoint =
                prefos_internal_safe_midpoint(new_lower, new_upper);
            new_lower = midpoint;
            new_upper = midpoint;
        }
        status = prefos_internal_append_bound_record(
            presolver, (int) row, live_column, old_lower, new_lower, 1);
        if (status != PREFOS_STATUS_OK) return status;
        status = prefos_internal_append_bound_record(
            presolver, (int) row, live_column, old_upper, new_upper, 0);
        if (status != PREFOS_STATUS_OK) return status;
        if (new_lower != old_lower)
            ++presolver->stats.tightened_box_bounds;
        if (new_upper != old_upper)
            ++presolver->stats.tightened_box_bounds;
        presolver->working_box_lower[box_position] = new_lower;
        presolver->working_box_upper[box_position] = new_upper;
        presolver->propagation_lower[live_column] = new_lower;
        presolver->propagation_upper[live_column] = new_upper;
        prefos_internal_mark_removed_row(presolver, row);
        ++presolver->stats.removed_singleton_rows;
    }
    return PREFOS_STATUS_OK;
}

PreFOSStatus prefos_internal_reduce_trivial_rows(
    PreFOSPresolver *presolver)
{
    size_t row;
    for (row = 0; row < presolver->original.A.rows; ++row)
    {
        PreFOSStatus status = reduce_trivial_row(presolver, row);
        if (status != PREFOS_STATUS_OK) return status;
    }
    return PREFOS_STATUS_OK;
}

PreFOSStatus prefos_internal_reduce_trivial_row_candidates(
    PreFOSPresolver *presolver, const int *rows, size_t count)
{
    size_t position;
    if (count > 0 && !rows) return PREFOS_STATUS_INVALID_ARGUMENT;
    for (position = 0; position < count; ++position)
    {
        int row = rows[position];
        PreFOSStatus status;
        if (row < 0 || (size_t) row >= presolver->original.A.rows)
            return PREFOS_STATUS_INVALID_ARGUMENT;
        status = reduce_trivial_row(presolver, (size_t) row);
        if (status != PREFOS_STATUS_OK) return status;
    }
    return PREFOS_STATUS_OK;
}
