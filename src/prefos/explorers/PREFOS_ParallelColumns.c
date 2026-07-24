/*
 * Copyright 2026 Hongpei Li
 * SPDX-License-Identifier: Apache-2.0
 */

#include "PREFOS_ColumnReductionInternal.h"
#include "PREFOS_CudaBackend.h"
#include "PREFOS_CudaLinearPropagation.h"
#include "ParallelRowDetection.h"

typedef struct
{
    const PreFOSPresolver *presolver;
    const PreFOSColumnWorkspace *workspace;
} PreFOSParallelColumnContext;

static int parallel_column_is_active(const void *context, size_t column)
{
    const PreFOSParallelColumnContext *parallel_context =
        (const PreFOSParallelColumnContext *) context;
    const PreFOSPresolver *presolver = parallel_context->presolver;
    const PreFOSColumnWorkspace *workspace = parallel_context->workspace;
    int p;
    if (!prefos_internal_column_is_linear_box(
            presolver, workspace, (int) column) ||
        workspace->protected_target[column])
        return 0;
    for (p = workspace->starts[column]; p < workspace->starts[column + 1]; ++p)
        if (workspace->dirty_row[workspace->rows[p]]) return 0;
    return workspace->starts[column] < workspace->starts[column + 1];
}

static double merged_lower(double target_lower, double source_lower,
                           double source_upper, double ratio)
{
    double source = ratio > 0.0 ? source_lower : source_upper;
    if (!isfinite(target_lower) || !isfinite(source)) return -INFINITY;
    return target_lower + ratio * source;
}

static double merged_upper(double target_upper, double source_lower,
                           double source_upper, double ratio)
{
    double source = ratio > 0.0 ? source_upper : source_lower;
    if (!isfinite(target_upper) || !isfinite(source)) return INFINITY;
    return target_upper + ratio * source;
}

static PreFOSStatus append_parallel_record(PreFOSPresolver *presolver,
                                           int source, int target, double ratio,
                                           double source_lower,
                                           double source_upper,
                                           double target_lower,
                                           double target_upper)
{
    PresolveColumnTransformationRecord record;
    memset(&record, 0, sizeof(record));
    record.type = PRESOLVE_COLUMNS_PARALLEL;
    record.column = source;
    record.secondary_column = target;
    record.ratio = ratio;
    record.lower = source_lower;
    record.upper = source_upper;
    record.secondary_lower = target_lower;
    record.secondary_upper = target_upper;
    return presolve_transformation_log_append_column_transformation(
               &presolver->transformations, &record, NULL)
               ? PREFOS_STATUS_OK
               : PREFOS_STATUS_OUT_OF_MEMORY;
}

PreFOSStatus
prefos_internal_reduce_parallel_column_groups(
    PreFOSPresolver *presolver, PreFOSColumnWorkspace *workspace)
{
    PreFOSParallelColumnContext context;
    PresolveSparseRowView view;
    int *parallel = NULL, *support_hashes = NULL, *coefficient_hashes = NULL;
    int *sort_auxiliary = NULL, *group_starts = NULL;
    unsigned char *gpu_eligible = NULL;
    size_t n_groups = 0, group;
    PreFOSStatus status = PREFOS_STATUS_OK;
    int detected = 0;
    if (!presolver->settings.parallel_column_reduction ||
        presolver->original.n < 2)
        return PREFOS_STATUS_OK;
    {
        size_t row;
        for (row = 0; row < presolver->original.A.rows; ++row)
        {
            int p;
            if (presolver->remove_rows[row]) continue;
            for (p = presolver->original.A.row_pointers[row];
                 p < presolver->original.A.row_pointers[row + 1]; ++p)
            {
                int column = presolver->original.A.column_indices[p];
                if (presolver->is_substituted[column] ||
                    presolver->is_parallel_removed[column])
                {
                    workspace->dirty_row[row] = 1;
                    break;
                }
            }
        }
    }
    parallel = (int *) prefos_internal_alloc_array(presolver->original.n,
                                                    sizeof(int));
    support_hashes = (int *) prefos_internal_alloc_array(
        presolver->original.n, sizeof(int));
    coefficient_hashes = (int *) prefos_internal_alloc_array(
        presolver->original.n, sizeof(int));
    sort_auxiliary = (int *) prefos_internal_alloc_array(
        presolver->original.n, sizeof(int));
    group_starts = (int *) prefos_internal_alloc_array(
        presolver->original.n + 1, sizeof(int));
    if (!parallel || !support_hashes || !coefficient_hashes ||
        !sort_auxiliary || !group_starts)
    {
        status = PREFOS_STATUS_OUT_OF_MEMORY;
        goto cleanup;
    }
    context.presolver = presolver;
    context.workspace = workspace;
    view = (PresolveSparseRowView){presolver->original.n,
                                  workspace->values,
                                  workspace->rows,
                                  workspace->starts,
                                  workspace->starts + 1,
                                  1};
    if (presolver->settings.structural_reductions_gpu)
    {
        PreFOSCudaPropagationStatus cuda_status =
            PREFOS_CUDA_PROPAGATION_UNAVAILABLE;
        PreFOSCudaWorkspace *cuda_workspace = NULL;
        size_t active_columns = 0, column;
        double gpu_milliseconds = 0.0;
        gpu_eligible = (unsigned char *) prefos_internal_alloc_array(
            presolver->original.n, sizeof(unsigned char));
        if (presolver->original.n > 0 && !gpu_eligible)
        {
            status = PREFOS_STATUS_OUT_OF_MEMORY;
            goto cleanup;
        }
        for (column = 0; column < presolver->original.n; ++column)
            gpu_eligible[column] = (unsigned char)
                (prefos_internal_column_is_linear_box(
                     presolver, workspace, (int) column) &&
                 !workspace->protected_target[column]);
        if (workspace->gpu_csc_valid)
        {
            cuda_workspace =
                prefos_internal_cuda_workspace_get(presolver, &cuda_status);
            if (cuda_workspace &&
                cuda_status == PREFOS_CUDA_PROPAGATION_OK)
                cuda_status = prefos_cuda_parallel_column_hash_sort(
                    cuda_workspace, gpu_eligible, workspace->dirty_row,
                    parallel, support_hashes, coefficient_hashes,
                    &active_columns, &gpu_milliseconds);
        }
        presolver->stats.parallel_column_gpu_milliseconds +=
            gpu_milliseconds;
        if (cuda_status == PREFOS_CUDA_PROPAGATION_OK)
        {
            detected = presolve_collect_parallel_row_groups(
                &view, presolver->settings.feasibility_tolerance,
                parallel, active_columns, support_hashes,
                coefficient_hashes, group_starts,
                presolver->original.n + 1, &n_groups);
            if (detected)
                ++presolver->stats.parallel_column_gpu_passes;
        }
        if (!detected)
            ++presolver->stats.parallel_column_gpu_fallbacks;
    }
    if (!detected)
    {
        detected = presolve_find_parallel_rows(
            &view, parallel_column_is_active, &context,
            presolver->settings.feasibility_tolerance,
            presolve_sort_rows_by_hash, parallel, support_hashes,
            coefficient_hashes, sort_auxiliary, group_starts,
            presolver->original.n + 1, &n_groups);
    }
    if (!detected)
    {
        status = PREFOS_STATUS_NUMERICAL_ERROR;
        goto cleanup;
    }
    for (group = 0; group < n_groups; ++group)
    {
        int start = group_starts[group], end = group_starts[group + 1];
        int target = parallel[end - 1];
        int position;
        if (!parallel_column_is_active(&context, (size_t) target)) continue;
        for (position = start; position < end - 1; ++position)
        {
            int source = parallel[position];
            int target_box, source_box;
            double ratio, objective_gap;
            double source_lower, source_upper, target_lower, target_upper;
            double new_lower, new_upper;
            if (!parallel_column_is_active(&context, (size_t) source)) continue;
            ratio = workspace->values[workspace->starts[source]] /
                    workspace->values[workspace->starts[target]];
            if (!isfinite(ratio) || ratio == 0.0) continue;
            objective_gap = workspace->objective[source] -
                            ratio * workspace->objective[target];
            source_box = presolver->variable_to_box[source];
            target_box = presolver->variable_to_box[target];
            source_lower = presolver->working_box_lower[source_box];
            source_upper = presolver->working_box_upper[source_box];
            target_lower = presolver->working_box_lower[target_box];
            target_upper = presolver->working_box_upper[target_box];
            if (fabs(objective_gap) >
                presolver->settings.feasibility_tolerance)
            {
                int fixed_column = -1;
                double fixed_value = 0.0;
                if (objective_gap > 0.0)
                {
                    if ((ratio > 0.0 && !isfinite(target_upper)) ||
                        (ratio < 0.0 && !isfinite(target_lower)))
                    {
                        fixed_column = source;
                        fixed_value = source_lower;
                    }
                    else if (ratio > 0.0 && !isfinite(source_lower))
                    {
                        fixed_column = target;
                        fixed_value = target_upper;
                    }
                    else if (ratio < 0.0 && !isfinite(source_lower))
                    {
                        fixed_column = target;
                        fixed_value = target_lower;
                    }
                }
                else
                {
                    if ((ratio > 0.0 && !isfinite(target_lower)) ||
                        (ratio < 0.0 && !isfinite(target_upper)))
                    {
                        fixed_column = source;
                        fixed_value = source_upper;
                    }
                    else if (ratio > 0.0 && !isfinite(source_upper))
                    {
                        fixed_column = target;
                        fixed_value = target_lower;
                    }
                    else if (ratio < 0.0 && !isfinite(source_upper))
                    {
                        fixed_column = target;
                        fixed_value = target_upper;
                    }
                }
                if (fixed_column < 0) continue;
                if (!isfinite(fixed_value))
                {
                    status = PREFOS_STATUS_PRIMAL_UNBOUNDED;
                    goto cleanup;
                }
                prefos_internal_mark_fixed_column(
                    presolver, fixed_column, fixed_value);
                ++presolver->stats.dual_fixed_columns;
                if (fixed_column == target) break;
                continue;
            }
            new_lower =
                merged_lower(target_lower, source_lower, source_upper, ratio);
            new_upper =
                merged_upper(target_upper, source_lower, source_upper, ratio);
            status = append_parallel_record(
                presolver, source, target, ratio, source_lower, source_upper,
                target_lower, target_upper);
            if (status != PREFOS_STATUS_OK) goto cleanup;
            presolver->working_box_lower[target_box] = new_lower;
            presolver->working_box_upper[target_box] = new_upper;
            presolver->propagation_lower[target] = new_lower;
            presolver->propagation_upper[target] = new_upper;
            presolver->is_parallel_removed[source] = 1;
            presolver->variable_to_box[source] = -1;
            workspace->protected_target[source] = 1;
            workspace->protected_target[target] = 1;
            ++presolver->stats.merged_parallel_columns;
            ++presolver->n_parallel_column_reductions;
        }
    }

cleanup:
    free(parallel);
    free(support_hashes);
    free(coefficient_hashes);
    free(sort_auxiliary);
    free(group_starts);
    free(gpu_eligible);
    return status;
}
