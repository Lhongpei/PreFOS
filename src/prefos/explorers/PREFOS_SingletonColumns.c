/*
 * Copyright 2026 Hongpei Li
 * SPDX-License-Identifier: Apache-2.0
 */

#include "PREFOS_ColumnReductionInternal.h"

static int singleton_bounds_are_implied(
    const PreFOSPresolver *presolver, size_t row, int pivot_column,
    double pivot, double row_lower, double row_upper, int *free_below,
    int *free_above)
{
    const PreFOSCsrMatrix *A = &presolver->original.A;
    int box_position = presolver->variable_to_box[pivot_column];
    long double rest_min = 0.0L, rest_max = 0.0L;
    int p;
    if (box_position < 0) return 0;
    for (p = A->row_pointers[row]; p < A->row_pointers[row + 1]; ++p)
    {
        int column = A->column_indices[p];
        double coefficient = A->values[p];
        double lower, upper;
        if (column == pivot_column || coefficient == 0.0 ||
            presolver->is_fixed[column])
            continue;
        if (presolver->is_substituted[column] ||
            presolver->is_parallel_removed[column])
            return 0;
        lower = presolver->propagation_lower[column];
        upper = presolver->propagation_upper[column];
        if (coefficient > 0.0)
        {
            rest_min += (long double) coefficient * (long double) lower;
            rest_max += (long double) coefficient * (long double) upper;
        }
        else
        {
            rest_min += (long double) coefficient * (long double) upper;
            rest_max += (long double) coefficient * (long double) lower;
        }
        if (!isfinite(rest_min) || !isfinite(rest_max)) return 0;
    }
    {
        long double x1, x2, implied_lower, implied_upper;
        double lower = presolver->working_box_lower[box_position];
        double upper = presolver->working_box_upper[box_position];
        double tolerance = presolver->settings.feasibility_tolerance;
        if (isfinite(row_lower) && isfinite(row_upper) &&
            fabs(row_upper - row_lower) <= tolerance)
        {
            x1 = ((long double) row_upper - rest_min) / (long double) pivot;
            x2 = ((long double) row_upper - rest_max) / (long double) pivot;
            implied_lower = fminl(x1, x2);
            implied_upper = fmaxl(x1, x2);
        }
        else
        {
            implied_lower = -INFINITY;
            implied_upper = INFINITY;
            if (pivot > 0.0)
            {
                if (isfinite(row_lower))
                    implied_lower =
                        ((long double) row_lower - rest_max) / pivot;
                if (isfinite(row_upper))
                    implied_upper =
                        ((long double) row_upper - rest_min) / pivot;
            }
            else
            {
                if (isfinite(row_upper))
                    implied_lower =
                        ((long double) row_upper - rest_min) / pivot;
                if (isfinite(row_lower))
                    implied_upper =
                        ((long double) row_lower - rest_max) / pivot;
            }
        }
        *free_below = !isfinite(lower) || implied_lower >= lower - tolerance;
        *free_above = !isfinite(upper) || implied_upper <= upper + tolerance;
    }
    return 1;
}

static PreFOSStatus append_equality_relaxed_record(
    PreFOSPresolver *presolver, int row, double side, double normal_sign)
{
    const PreFOSCsrMatrix *matrix = &presolver->original.A;
    int start = matrix->row_pointers[row];
    PresolveRowTransformationRecord record = {
        .type = PRESOLVE_ROW_EQUALITY_RELAXED,
        .row = row,
        .source_row = row,
        .ratio = normal_sign,
        .new_side = side,
        .indices = matrix->column_indices + start,
        .coefficients = matrix->values + start,
        .length = (size_t) (matrix->row_pointers[row + 1] - start)};
    return presolve_transformation_log_append_row_transformation(
               &presolver->transformations, &record, NULL)
               ? PREFOS_STATUS_OK
               : PREFOS_STATUS_OUT_OF_MEMORY;
}

PreFOSStatus prefos_internal_reduce_singleton_columns(
    PreFOSPresolver *presolver, PreFOSColumnWorkspace *workspace,
    int allow_one_sided)
{
    const PreFOSProblemData *problem = &presolver->original;
    size_t column;
    if (!presolver->settings.singleton_column_reduction)
        return PREFOS_STATUS_OK;
    for (column = 0; column < problem->n; ++column)
    {
        int row, p, free_below = 0, free_above = 0;
        int targets[PREFOS_MAX_AGGREGATION_TERMS];
        double scales[PREFOS_MAX_AGGREGATION_TERMS];
        double pivot = 0.0, lower, upper, side;
        int equality;
        PreFOSSubstitutionMode substitution_mode = PREFOS_SUBSTITUTION_STANDARD;
        size_t target_count = 0;
        if (!prefos_internal_column_is_linear_box(
                presolver, workspace, (int) column) ||
            workspace->protected_target[column] ||
            workspace->starts[column + 1] - workspace->starts[column] != 1)
            continue;
        row = workspace->rows[workspace->starts[column]];
        if (presolver->remove_rows[row] || workspace->dirty_row[row]) continue;
        if (prefos_internal_effective_row_bounds(
                presolver, (size_t) row, &lower, &upper) != PREFOS_STATUS_OK)
            return PREFOS_STATUS_NUMERICAL_ERROR;
        for (p = problem->A.row_pointers[row];
             p < problem->A.row_pointers[row + 1]; ++p)
        {
            int target = problem->A.column_indices[p];
            double coefficient = problem->A.values[p];
            if (coefficient == 0.0 || presolver->is_fixed[target]) continue;
            if (target == (int) column)
            {
                pivot = coefficient;
                continue;
            }
            if (presolver->is_substituted[target] ||
                presolver->is_parallel_removed[target] ||
                target_count >= PREFOS_MAX_AGGREGATION_TERMS)
            {
                target_count = PREFOS_MAX_AGGREGATION_TERMS + 1;
                break;
            }
            targets[target_count] = target;
            scales[target_count] = coefficient;
            ++target_count;
        }
        if (pivot == 0.0 || target_count == 0 ||
            target_count >
                (size_t) presolver->settings.max_aggregation_terms ||
            !singleton_bounds_are_implied(presolver, (size_t) row, (int) column,
                                          pivot, lower, upper, &free_below,
                                          &free_above))
            continue;
        equality = isfinite(lower) && isfinite(upper) &&
                   fabs(lower - upper) <=
                       presolver->settings.feasibility_tolerance;
        if ((!free_below || !free_above) && !allow_one_sided) continue;
        if (!free_below || !free_above)
        {
            double objective = workspace->objective[column];
            if (!equality)
            {
                int tighten_lower =
                    ((objective < -presolver->settings.feasibility_tolerance &&
                      pivot > 0.0 && free_above) ||
                     (objective > presolver->settings.feasibility_tolerance &&
                      pivot < 0.0 && free_below)) &&
                    isfinite(upper);
                int tighten_upper =
                    ((objective > presolver->settings.feasibility_tolerance &&
                      pivot > 0.0 && free_below) ||
                     (objective < -presolver->settings.feasibility_tolerance &&
                      pivot < 0.0 && free_above)) &&
                    isfinite(lower);
                PreFOSStatus status;
                if (tighten_lower)
                {
                    status = append_equality_relaxed_record(
                        presolver, row, upper, 1.0);
                    if (status != PREFOS_STATUS_OK) return status;
                    presolver->working_constraint_lower[row] = upper;
                }
                else if (tighten_upper)
                {
                    status = append_equality_relaxed_record(
                        presolver, row, lower, -1.0);
                    if (status != PREFOS_STATUS_OK) return status;
                    presolver->working_constraint_upper[row] = lower;
                }
                else
                    continue;
                workspace->dirty_row[row] = 1;
                ++presolver->stats.tightened_singleton_rows;
                continue;
            }
            else
            {
                int box = presolver->variable_to_box[column];
                double retained_bound = free_above
                                            ? presolver->working_box_lower[box]
                                            : presolver->working_box_upper[box];
                double residual_side = upper;
                int keep_lower = free_above ? pivot < 0.0 : pivot > 0.0;
                if (!isfinite(retained_bound) ||
                    !prefos_internal_safe_add_product(
                        &residual_side, -pivot, retained_bound))
                    continue;
                if (keep_lower)
                {
                    presolver->working_constraint_lower[row] = residual_side;
                    presolver->working_constraint_upper[row] = INFINITY;
                }
                else
                {
                    presolver->working_constraint_lower[row] = -INFINITY;
                    presolver->working_constraint_upper[row] = residual_side;
                }
                substitution_mode = PREFOS_SUBSTITUTION_RESIDUAL_ROW;
            }
        }
        if (workspace->objective[column] > 0.0)
            side = pivot > 0.0 ? lower : upper;
        else if (workspace->objective[column] < 0.0)
            side = pivot > 0.0 ? upper : lower;
        else
            side = isfinite(lower) ? lower : upper;
        if (!isfinite(side)) continue;
        for (p = 0; p < (int) target_count; ++p)
        {
            scales[p] = -scales[p] / pivot;
            workspace->protected_target[targets[p]] = 1;
        }
        {
            PreFOSStatus status = prefos_internal_append_column_substitution(
                presolver, (int) column, targets, scales, target_count, row,
                side / pivot, pivot, workspace, substitution_mode);
            if (status != PREFOS_STATUS_OK) return status;
        }
        workspace->dirty_row[row] = 1;
    }
    return PREFOS_STATUS_OK;
}
