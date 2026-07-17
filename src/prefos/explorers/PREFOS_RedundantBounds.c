/*
 * Copyright 2026 Hongpei Li
 * SPDX-License-Identifier: Apache-2.0
 */

#include "PREFOS_ColumnReductionInternal.h"
#include "PREFOS_ColumnReductions.h"

static int row_implies_box_side(const PreFOSPresolver *presolver, size_t row,
                                int column, double bound, int lower_side)
{
    const PreFOSCsrMatrix *A = &presolver->original.A;
    long double rest_min = 0.0L, rest_max = 0.0L;
    double coefficient = 0.0;
    int p;
    for (p = A->row_pointers[row]; p < A->row_pointers[row + 1]; ++p)
    {
        int current = A->column_indices[p];
        double value = A->values[p];
        double current_lower, current_upper;
        int box;
        if (current == column)
        {
            coefficient = value;
            continue;
        }
        if (presolver->is_substituted[current] ||
            presolver->is_parallel_removed[current])
            return 0;
        if (presolver->is_fixed[current])
            current_lower = current_upper = presolver->fixed_values[current];
        else if ((box = presolver->variable_to_box[current]) >= 0)
        {
            current_lower = presolver->working_box_lower[box];
            current_upper = presolver->working_box_upper[box];
        }
        else
        {
            current_lower = presolver->propagation_lower[current];
            current_upper = presolver->propagation_upper[current];
        }
        if (value > 0.0)
        {
            rest_min += (long double) value * (long double) current_lower;
            rest_max += (long double) value * (long double) current_upper;
        }
        else if (value < 0.0)
        {
            rest_min += (long double) value * (long double) current_upper;
            rest_max += (long double) value * (long double) current_lower;
        }
        if (isnan(rest_min) || isnan(rest_max)) return 0;
    }
    if (coefficient == 0.0) return 0;
    if (lower_side)
    {
        long double implied = -INFINITY;
        if (coefficient > 0.0 &&
            isfinite(presolver->working_constraint_lower[row]) &&
            isfinite(rest_max))
            implied = ((long double) presolver->working_constraint_lower[row] -
                       rest_max) /
                      coefficient;
        else if (coefficient < 0.0 &&
                 isfinite(presolver->working_constraint_upper[row]) &&
                 isfinite(rest_min))
            implied = ((long double) presolver->working_constraint_upper[row] -
                       rest_min) /
                      coefficient;
        return isfinite(implied) &&
               implied >= (long double) bound -
                              presolver->settings.feasibility_tolerance;
    }
    else
    {
        long double implied = INFINITY;
        if (coefficient > 0.0 &&
            isfinite(presolver->working_constraint_upper[row]) &&
            isfinite(rest_min))
            implied = ((long double) presolver->working_constraint_upper[row] -
                       rest_min) /
                      coefficient;
        else if (coefficient < 0.0 &&
                 isfinite(presolver->working_constraint_lower[row]) &&
                 isfinite(rest_max))
            implied = ((long double) presolver->working_constraint_lower[row] -
                       rest_max) /
                      coefficient;
        return isfinite(implied) &&
               implied <= (long double) bound +
                              presolver->settings.feasibility_tolerance;
    }
}

PreFOSStatus
prefos_internal_remove_redundant_box_bounds(PreFOSPresolver *presolver)
{
    PreFOSColumnWorkspace workspace;
    PreFOSStatus status;
    size_t box;
    int changed;
    if (!presolver->settings.remove_redundant_bounds ||
        presolver->settings.propagated_bound_policy !=
            PREFOS_PROPAGATED_BOUND_POLICY_INTERIOR_POINT)
        return PREFOS_STATUS_OK;
    status = prefos_internal_build_column_workspace(presolver, &workspace);
    if (status != PREFOS_STATUS_OK) return status;
    do
    {
        changed = 0;
        for (box = 0; box < presolver->original.n_box; ++box)
        {
            int column = presolver->original.box_indices[box];
            double lower = presolver->working_box_lower[box];
            double upper = presolver->working_box_upper[box];
            int p;
            if (presolver->is_fixed[column] || presolver->is_substituted[column] ||
                presolver->is_parallel_removed[column])
                continue;
            if (isfinite(lower) && !isfinite(upper))
            {
                for (p = workspace.starts[column];
                     p < workspace.starts[column + 1]; ++p)
                {
                    int row = workspace.rows[p];
                    if (!presolver->remove_rows[row] &&
                        row_implies_box_side(presolver, (size_t) row, column,
                                             lower, 1))
                    {
                        presolver->working_box_lower[box] = -INFINITY;
                        ++presolver->stats.removed_redundant_box_lower_bounds;
                        changed = 1;
                        break;
                    }
                }
            }
            else if (!isfinite(lower) && isfinite(upper))
            {
                for (p = workspace.starts[column];
                     p < workspace.starts[column + 1]; ++p)
                {
                    int row = workspace.rows[p];
                    if (!presolver->remove_rows[row] &&
                        row_implies_box_side(presolver, (size_t) row, column,
                                             upper, 0))
                    {
                        presolver->working_box_upper[box] = INFINITY;
                        ++presolver->stats.removed_redundant_box_upper_bounds;
                        changed = 1;
                        break;
                    }
                }
            }
        }
    } while (changed);
    prefos_internal_free_column_workspace(&workspace);
    return PREFOS_STATUS_OK;
}
