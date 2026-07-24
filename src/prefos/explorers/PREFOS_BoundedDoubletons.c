/*
 * Copyright 2026 Hongpei Li
 * SPDX-License-Identifier: Apache-2.0
 */

#include "PREFOS_ColumnReductionInternal.h"

static size_t doubleton_fill(const PreFOSPresolver *presolver,
                             const PreFOSColumnWorkspace *workspace,
                             int pivot_column, int target_column,
                             size_t source_row)
{
    const PreFOSCsrMatrix *A = &presolver->original.A;
    size_t fill = 0;
    int position;
    for (position = workspace->starts[pivot_column];
         position < workspace->ends[pivot_column]; ++position)
    {
        int row = workspace->rows[position];
        int p, found = 0;
        if ((size_t) row == source_row || presolver->remove_rows[row]) continue;
        for (p = A->row_pointers[row]; p < A->row_pointers[row + 1]; ++p)
        {
            if (A->column_indices[p] == target_column && A->values[p] != 0.0 &&
                !presolver->is_fixed[target_column] &&
                !presolver->is_substituted[target_column])
            {
                found = 1;
                break;
            }
        }
        if (!found) ++fill;
    }
    return fill;
}

PreFOSStatus prefos_internal_reduce_bounded_doubletons(
    PreFOSPresolver *presolver, PreFOSColumnWorkspace *workspace)
{
    const PreFOSProblemData *problem = &presolver->original;
    size_t row;
    if (!presolver->settings.bounded_doubleton_substitution)
        return PREFOS_STATUS_OK;
    for (row = 0; row < problem->A.rows; ++row)
    {
        int columns[2], pivot_column = -1, target_column = -1;
        double coefficients[2], lower, upper, pivot, target_coefficient;
        double pivot_lower, pivot_upper, target_lower, target_upper;
        double alpha, beta, mapped_lower = -INFINITY, mapped_upper = INFINITY;
        double fixed_shift = 0.0;
        double scales[1];
        int targets[1];
        size_t live;
        int candidate, position;
        if (presolver->remove_rows[row] || workspace->dirty_row[row]) continue;
        lower = presolver->working_constraint_lower[row];
        upper = presolver->working_constraint_upper[row];
        if (!isfinite(lower) || lower != upper) continue;
        live = 0;
        for (position = problem->A.row_pointers[row];
             position < problem->A.row_pointers[row + 1]; ++position)
        {
            int column = problem->A.column_indices[position];
            double coefficient = problem->A.values[position];
            if (coefficient == 0.0) continue;
            if (presolver->is_fixed[column])
            {
                if (!prefos_internal_safe_add_product(
                        &fixed_shift, coefficient,
                        presolver->fixed_values[column]))
                    return PREFOS_STATUS_NUMERICAL_ERROR;
                continue;
            }
            if (presolver->is_substituted[column] ||
                presolver->is_parallel_removed[column])
                continue;
            if (live < 2)
            {
                columns[live] = column;
                coefficients[live] = coefficient;
            }
            ++live;
        }
        lower -= fixed_shift;
        upper -= fixed_shift;
        if (isnan(lower) || isnan(upper))
            return PREFOS_STATUS_NUMERICAL_ERROR;
        if (live != 2) continue;
        for (candidate = 0; candidate < 2; ++candidate)
        {
            int box = presolver->variable_to_box[columns[candidate]];
            if (box >= 0 &&
                !isfinite(presolver->working_box_lower[box]) &&
                !isfinite(presolver->working_box_upper[box]))
                break;
        }
        if (candidate < 2) continue;
        for (candidate = 0; candidate < 2; ++candidate)
        {
            int possible_pivot = columns[candidate];
            int possible_target = columns[1 - candidate];
            int pivot_box, target_box;
            if (!prefos_internal_column_is_linear_box(
                    presolver, workspace, possible_pivot) ||
                workspace->protected_target[possible_pivot] ||
                !prefos_internal_column_is_linear_box(
                    presolver, workspace, possible_target) ||
                workspace->protected_target[possible_target])
                continue;
            pivot_box = presolver->variable_to_box[possible_pivot];
            target_box = presolver->variable_to_box[possible_target];
            if (!isfinite(presolver->working_box_lower[pivot_box]) &&
                !isfinite(presolver->working_box_upper[pivot_box]))
                continue;
            if (workspace->live_degrees[possible_pivot] >
                presolver->settings.max_aggregation_column_degree)
                continue;
            if (doubleton_fill(presolver, workspace, possible_pivot,
                               possible_target, row) >
                (size_t) presolver->settings.max_aggregation_fill)
                continue;
            (void) target_box;
            pivot_column = possible_pivot;
            target_column = possible_target;
            pivot = coefficients[candidate];
            target_coefficient = coefficients[1 - candidate];
            break;
        }
        if (pivot_column < 0 || pivot == 0.0) continue;
        pivot_lower = presolver->working_box_lower
            [presolver->variable_to_box[pivot_column]];
        pivot_upper = presolver->working_box_upper
            [presolver->variable_to_box[pivot_column]];
        target_lower = presolver->working_box_lower
            [presolver->variable_to_box[target_column]];
        target_upper = presolver->working_box_upper
            [presolver->variable_to_box[target_column]];
        alpha = -target_coefficient / pivot;
        beta = lower / pivot;
        if (alpha == 0.0 || !isfinite(alpha) || !isfinite(beta)) continue;
        if (isfinite(pivot_lower))
        {
            double mapped = (pivot_lower - beta) / alpha;
            if (alpha > 0.0)
                mapped_lower = prefos_internal_outward_bound_cast(mapped, 1);
            else
                mapped_upper = prefos_internal_outward_bound_cast(mapped, 0);
        }
        if (isfinite(pivot_upper))
        {
            double mapped = (pivot_upper - beta) / alpha;
            if (alpha > 0.0)
                mapped_upper = prefos_internal_outward_bound_cast(mapped, 0);
            else
                mapped_lower = prefos_internal_outward_bound_cast(mapped, 1);
        }
        target_lower = fmax(target_lower, mapped_lower);
        target_upper = fmin(target_upper, mapped_upper);
        if (target_lower > target_upper +
                               presolver->settings.feasibility_tolerance)
            return PREFOS_STATUS_PRIMAL_INFEASIBLE;
        if (target_lower > target_upper)
        {
            double midpoint =
                prefos_internal_safe_midpoint(target_lower, target_upper);
            target_lower = midpoint;
            target_upper = midpoint;
        }
        targets[0] = target_column;
        scales[0] = alpha;
        workspace->protected_target[target_column] = 1;
        {
            PreFOSStatus status = prefos_internal_append_column_substitution(
                presolver, pivot_column, targets, scales, 1, (int) row, beta,
                pivot, workspace, PREFOS_SUBSTITUTION_BOUNDED_DOUBLETON);
            if (status != PREFOS_STATUS_OK) return status;
        }
        {
            int target_box = presolver->variable_to_box[target_column];
            presolver->working_box_lower[target_box] = target_lower;
            presolver->working_box_upper[target_box] = target_upper;
            presolver->propagation_lower[target_column] = target_lower;
            presolver->propagation_upper[target_column] = target_upper;
        }
    }
    return PREFOS_STATUS_OK;
}
