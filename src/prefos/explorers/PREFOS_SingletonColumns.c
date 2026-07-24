/*
 * Copyright 2026 Hongpei Li
 * SPDX-License-Identifier: Apache-2.0
 */

#include "PREFOS_ColumnReductionInternal.h"
#include "PREFOS_CudaBackend.h"
#include "PREFOS_CudaLinearPropagation.h"

static PreFOSStatus populate_gpu_singleton_candidates(
    PreFOSPresolver *presolver, PreFOSColumnWorkspace *workspace)
{
    const PreFOSProblemData *problem = &presolver->original;
    unsigned char *eligible = NULL;
    PreFOSCudaPropagationStatus cuda_status =
        PREFOS_CUDA_PROPAGATION_UNAVAILABLE;
    PreFOSCudaWorkspace *cuda_workspace = NULL;
    double milliseconds = 0.0;
    size_t column;

    workspace->gpu_singleton_candidates_valid = 0;
    workspace->n_gpu_singleton_candidates = 0;
    if (!presolver->settings.structural_reductions_gpu ||
        !workspace->gpu_csc_valid)
        return PREFOS_STATUS_OK;
    workspace->gpu_singleton_candidates = (int *)
        prefos_internal_alloc_array(problem->n, sizeof(int));
    eligible = (unsigned char *)
        prefos_internal_alloc_array(problem->n, sizeof(unsigned char));
    if (problem->n > 0 &&
        (!workspace->gpu_singleton_candidates || !eligible))
    {
        free(eligible);
        return PREFOS_STATUS_OUT_OF_MEMORY;
    }
    for (column = 0; column < problem->n; ++column)
        eligible[column] = (unsigned char)
            (prefos_internal_column_is_linear_box(
                 presolver, workspace, (int) column) &&
             !workspace->protected_target[column]);
    if (workspace->gpu_csc_valid)
    {
        cuda_workspace =
            prefos_internal_cuda_workspace_get(presolver, &cuda_status);
        if (cuda_workspace && cuda_status == PREFOS_CUDA_PROPAGATION_OK)
            cuda_status = prefos_cuda_singleton_column_candidates(
                cuda_workspace, eligible, workspace->dirty_row,
                workspace->gpu_singleton_candidates,
                &workspace->n_gpu_singleton_candidates, &milliseconds);
    }
    free(eligible);
    presolver->stats.singleton_column_gpu_milliseconds += milliseconds;
    if (cuda_status == PREFOS_CUDA_PROPAGATION_OK)
    {
        workspace->gpu_singleton_candidates_valid = 1;
        ++presolver->stats.singleton_column_gpu_passes;
        presolver->stats.singleton_column_gpu_candidates +=
            workspace->n_gpu_singleton_candidates;
    }
    else
    {
        free(workspace->gpu_singleton_candidates);
        workspace->gpu_singleton_candidates = NULL;
        ++presolver->stats.singleton_column_gpu_fallbacks;
    }
    return PREFOS_STATUS_OK;
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
    int *targets = NULL;
    double *scales = NULL;
    int *live_degrees = NULL;
    int *queue = NULL;
    unsigned char *queued = NULL;
    size_t queue_position = 0, queue_count = 0;
    size_t row_capacity = 0, row_index, column;
    PreFOSStatus status;
    if (!presolver->settings.singleton_column_reduction)
        return PREFOS_STATUS_OK;
    status = populate_gpu_singleton_candidates(presolver, workspace);
    if (status != PREFOS_STATUS_OK) return status;
    live_degrees =
        (int *) prefos_internal_alloc_array(problem->n, sizeof(int));
    queue = (int *) prefos_internal_alloc_array(problem->n, sizeof(int));
    queued = (unsigned char *) calloc(problem->n, sizeof(unsigned char));
    if (problem->n > 0 && (!live_degrees || !queue || !queued))
    {
        free(live_degrees);
        free(queue);
        free(queued);
        return PREFOS_STATUS_OUT_OF_MEMORY;
    }
    for (column = 0; column < problem->n; ++column)
    {
        int degree = workspace->gpu_stats_valid
                         ? workspace->gpu_degrees[column]
                         : workspace->live_degrees[column];
        live_degrees[column] = degree;
        if (degree == 1 &&
            prefos_internal_column_is_linear_box(
                presolver, workspace, (int) column) &&
            !workspace->protected_target[column] &&
            presolver->substitution_incoming_depth[column] <
                PREFOS_MAX_SUBSTITUTION_DEPTH)
        {
            queue[queue_count++] = (int) column;
            queued[column] = 1;
        }
    }
    if (queue_count == 0)
    {
        status = PREFOS_STATUS_OK;
        goto cleanup;
    }
    row_capacity = workspace->max_row_nnz;
    if (row_capacity == 0)
    {
        for (row_index = 0; row_index < problem->A.rows; ++row_index)
        {
            size_t length = (size_t)
                (problem->A.row_pointers[row_index + 1] -
                 problem->A.row_pointers[row_index]);
            if (length > row_capacity) row_capacity = length;
        }
    }
    targets = (int *) prefos_internal_alloc_array(row_capacity, sizeof(int));
    scales = (double *)
        prefos_internal_alloc_array(row_capacity, sizeof(double));
    if (row_capacity > 0 && (!targets || !scales))
    {
        status = PREFOS_STATUS_OUT_OF_MEMORY;
        goto cleanup;
    }
    while (queue_position < queue_count)
    {
        column = (size_t) queue[queue_position++];
        int row, p, free_below = 0, free_above = 0;
        int row_was_removed;
        double fixed_shift = 0.0;
        double pivot = 0.0, lower, upper, side;
        long double rest_min = 0.0L, rest_max = 0.0L;
        int finite_rest_min = 1, finite_rest_max = 1;
        int equality;
        PreFOSSubstitutionMode substitution_mode = PREFOS_SUBSTITUTION_STANDARD;
        size_t target_count = 0;
        if (!prefos_internal_column_is_linear_box(
                presolver, workspace, (int) column) ||
            workspace->protected_target[column] ||
            presolver->substitution_incoming_depth[column] >=
                PREFOS_MAX_SUBSTITUTION_DEPTH ||
            live_degrees[column] != 1)
            continue;
        row = -1;
        for (p = workspace->starts[column];
             p < workspace->ends[column]; ++p)
        {
            int candidate_row = workspace->rows[p];
            if (!presolver->remove_rows[candidate_row])
            {
                row = candidate_row;
                break;
            }
        }
        if (row < 0 || workspace->dirty_row[row]) continue;
        for (p = problem->A.row_pointers[row];
             p < problem->A.row_pointers[row + 1]; ++p)
        {
            int target = problem->A.column_indices[p];
            double coefficient = problem->A.values[p];
            double target_lower, target_upper;
            if (coefficient == 0.0) continue;
            if (target == (int) column)
            {
                pivot = coefficient;
                continue;
            }
            if (presolver->is_fixed[target])
            {
                if (!prefos_internal_safe_add_product(
                        &fixed_shift, coefficient,
                        presolver->fixed_values[target]))
                {
                    status = PREFOS_STATUS_NUMERICAL_ERROR;
                    goto cleanup;
                }
                continue;
            }
            if (presolver->is_substituted[target])
            {
                if (!prefos_internal_term_is_active_in_row(
                        presolver, (size_t) row, target))
                    continue;
                target_count = row_capacity + 1;
                break;
            }
            if (presolver->is_parallel_removed[target])
            {
                target_count = row_capacity + 1;
                break;
            }
            if (target_count >= row_capacity)
            {
                target_count = row_capacity + 1;
                break;
            }
            targets[target_count] = target;
            scales[target_count] = coefficient;
            ++target_count;
            target_lower = presolver->propagation_lower[target];
            target_upper = presolver->propagation_upper[target];
            if (coefficient > 0.0)
            {
                if (finite_rest_min)
                {
                    long double term =
                        (long double) coefficient * target_lower;
                    if (isfinite(term) &&
                        isfinite(rest_min + term))
                        rest_min += term;
                    else
                        finite_rest_min = 0;
                }
                if (finite_rest_max)
                {
                    long double term =
                        (long double) coefficient * target_upper;
                    if (isfinite(term) &&
                        isfinite(rest_max + term))
                        rest_max += term;
                    else
                        finite_rest_max = 0;
                }
            }
            else
            {
                if (finite_rest_min)
                {
                    long double term =
                        (long double) coefficient * target_upper;
                    if (isfinite(term) &&
                        isfinite(rest_min + term))
                        rest_min += term;
                    else
                        finite_rest_min = 0;
                }
                if (finite_rest_max)
                {
                    long double term =
                        (long double) coefficient * target_lower;
                    if (isfinite(term) &&
                        isfinite(rest_max + term))
                        rest_max += term;
                    else
                        finite_rest_max = 0;
                }
            }
        }
        lower = presolver->working_constraint_lower[row] -
                fixed_shift;
        upper = presolver->working_constraint_upper[row] -
                fixed_shift;
        if (isnan(lower) || isnan(upper))
        {
            status = PREFOS_STATUS_NUMERICAL_ERROR;
            goto cleanup;
        }
        if (pivot == 0.0 || target_count == 0 ||
            target_count > row_capacity)
            continue;
        {
            int box_position =
                presolver->variable_to_box[column];
            long double implied_lower = -INFINITY;
            long double implied_upper = INFINITY;
            double box_lower, box_upper;
            double tolerance =
                presolver->settings.feasibility_tolerance;
            if (box_position < 0) continue;
            box_lower =
                presolver->working_box_lower[box_position];
            box_upper =
                presolver->working_box_upper[box_position];
            if (pivot > 0.0)
            {
                if (isfinite(lower) && finite_rest_max)
                    implied_lower =
                        ((long double) lower - rest_max) / pivot;
                if (isfinite(upper) && finite_rest_min)
                    implied_upper =
                        ((long double) upper - rest_min) / pivot;
            }
            else
            {
                if (isfinite(upper) && finite_rest_min)
                    implied_lower =
                        ((long double) upper - rest_min) / pivot;
                if (isfinite(lower) && finite_rest_max)
                    implied_upper =
                        ((long double) lower - rest_max) / pivot;
            }
            free_below =
                !isfinite(box_lower) ||
                (isfinite(implied_lower) &&
                 implied_lower >=
                     (long double) box_lower - tolerance);
            free_above =
                !isfinite(box_upper) ||
                (isfinite(implied_upper) &&
                 implied_upper <=
                     (long double) box_upper + tolerance);
        }
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
                if (tighten_lower)
                {
                    status = append_equality_relaxed_record(
                        presolver, row, upper, 1.0);
                    if (status != PREFOS_STATUS_OK) goto cleanup;
                    presolver->working_constraint_lower[row] = upper;
                }
                else if (tighten_upper)
                {
                    status = append_equality_relaxed_record(
                        presolver, row, lower, -1.0);
                    if (status != PREFOS_STATUS_OK) goto cleanup;
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
        for (row_index = 0; row_index < target_count; ++row_index)
            scales[row_index] = -scales[row_index] / pivot;
        row_was_removed = presolver->remove_rows[row];
        status = prefos_internal_append_column_substitution(
            presolver, (int) column, targets, scales, target_count, row,
            side / pivot, pivot, workspace, substitution_mode);
        if (status != PREFOS_STATUS_OK) goto cleanup;
        workspace->dirty_row[row] =
            (unsigned char)
                (substitution_mode !=
                 PREFOS_SUBSTITUTION_RESIDUAL_ROW);
        if (!row_was_removed && presolver->remove_rows[row])
        {
            for (p = problem->A.row_pointers[row];
                 p < problem->A.row_pointers[row + 1]; ++p)
            {
                int adjacent_column = problem->A.column_indices[p];
                if (problem->A.values[p] == 0.0 ||
                    live_degrees[adjacent_column] <= 0)
                    continue;
                --live_degrees[adjacent_column];
                if (live_degrees[adjacent_column] == 1 &&
                    !queued[adjacent_column])
                {
                    queue[queue_count++] = adjacent_column;
                    queued[adjacent_column] = 1;
                }
            }
        }
    }
    status = PREFOS_STATUS_OK;
cleanup:
    free(targets);
    free(scales);
    free(live_degrees);
    free(queue);
    free(queued);
    return status;
}
