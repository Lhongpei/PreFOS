/*
 * Copyright 2026 Hongpei Li
 * SPDX-License-Identifier: Apache-2.0
 */

#include "PREFOS_ColumnReductionInternal.h"

static double choose_zero_or_bound(double lower, double upper)
{
    if (lower <= 0.0 && upper >= 0.0) return 0.0;
    if (isfinite(lower) && lower > 0.0) return lower;
    if (isfinite(upper) && upper < 0.0) return upper;
    if (isfinite(lower)) return lower;
    if (isfinite(upper)) return upper;
    return 0.0;
}

PreFOSStatus prefos_internal_reduce_empty_and_dual_fixed_columns(
    PreFOSPresolver *presolver, const PreFOSColumnWorkspace *workspace)
{
    const PreFOSProblemData *problem = &presolver->original;
    const double zero_tolerance = presolver->settings.feasibility_tolerance;
    size_t column;

    for (column = 0; column < problem->n; ++column)
    {
        int box_position, p;
        double lower, upper, objective, value;
        int down_locked = 0, up_locked = 0;
        if (!prefos_internal_column_is_linear_box(
                presolver, workspace, (int) column))
            continue;
        box_position = presolver->variable_to_box[column];
        lower = presolver->working_box_lower[box_position];
        upper = presolver->working_box_upper[box_position];
        objective = workspace->objective[column];

        if ((workspace->gpu_stats_valid &&
             workspace->gpu_degrees[column] == 0) ||
            (!workspace->gpu_stats_valid &&
             workspace->live_degrees[column] == 0))
        {
            if (!presolver->settings.remove_empty_columns) continue;
            if (objective > zero_tolerance)
            {
                if (!isfinite(lower)) return PREFOS_STATUS_PRIMAL_UNBOUNDED;
                value = lower;
            }
            else if (objective < -zero_tolerance)
            {
                if (!isfinite(upper)) return PREFOS_STATUS_PRIMAL_UNBOUNDED;
                value = upper;
            }
            else
                value = choose_zero_or_bound(lower, upper);
            prefos_internal_mark_fixed_column(presolver, (int) column, value);
            ++presolver->stats.removed_empty_columns;
            continue;
        }
        if (!presolver->settings.dual_fixing) continue;

        if (workspace->gpu_stats_valid)
        {
            down_locked = workspace->gpu_down_locked[column] != 0;
            up_locked = workspace->gpu_up_locked[column] != 0;
        }
        for (p = workspace->starts[column];
             !workspace->gpu_stats_valid &&
             p < workspace->ends[column]; ++p)
        {
            int row = workspace->rows[p];
            double coefficient = workspace->values[p];
            if (presolver->remove_rows[row]) continue;
            if (workspace->dirty_row[row])
            {
                down_locked = 1;
                up_locked = 1;
                break;
            }
            if (coefficient > 0.0)
            {
                down_locked |=
                    isfinite(presolver->working_constraint_lower[row]);
                up_locked |= isfinite(presolver->working_constraint_upper[row]);
            }
            else
            {
                down_locked |=
                    isfinite(presolver->working_constraint_upper[row]);
                up_locked |= isfinite(presolver->working_constraint_lower[row]);
            }
            if (down_locked && up_locked) break;
        }
        if (objective > zero_tolerance && !down_locked)
        {
            if (!isfinite(lower)) return PREFOS_STATUS_PRIMAL_UNBOUNDED;
            prefos_internal_mark_fixed_column(presolver, (int) column, lower);
            ++presolver->stats.dual_fixed_columns;
        }
        else if (objective < -zero_tolerance && !up_locked)
        {
            if (!isfinite(upper)) return PREFOS_STATUS_PRIMAL_UNBOUNDED;
            prefos_internal_mark_fixed_column(presolver, (int) column, upper);
            ++presolver->stats.dual_fixed_columns;
        }
        else if (fabs(objective) <= zero_tolerance)
        {
            if (!down_locked && isfinite(lower))
            {
                prefos_internal_mark_fixed_column(
                    presolver, (int) column, lower);
                ++presolver->stats.dual_fixed_columns;
            }
            else if (!up_locked && isfinite(upper))
            {
                prefos_internal_mark_fixed_column(
                    presolver, (int) column, upper);
                ++presolver->stats.dual_fixed_columns;
            }
        }
    }
    return PREFOS_STATUS_OK;
}
