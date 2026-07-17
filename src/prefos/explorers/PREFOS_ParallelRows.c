/*
 * Copyright 2026 Hongpei Li
 * SPDX-License-Identifier: Apache-2.0
 */

#include "PREFOS_ParallelRows.h"

#include "ParallelRowDetection.h"
#include "ParallelRowReduction.h"
#include "core/PREFOS_Timer.h"
#include "PREFOS_CudaBackend.h"

static int prefos_row_is_active(const void *context, size_t row)
{
    const PreFOSPresolver *presolver = (const PreFOSPresolver *) context;
    return !presolver->remove_rows[row];
}

static int prefos_parallel_values_close(const void *context, double left,
                                     double right, double tolerance)
{
    (void) context;
    return prefos_internal_values_close(left, right, tolerance);
}

static PreFOSStatus append_row_record(
    PreFOSPresolver *presolver, PresolveRowTransformationType type, int row,
    int source_row, double ratio, double new_side)
{
    const PreFOSCsrMatrix *matrix = &presolver->original.A;
    int start = matrix->row_pointers[row];
    PresolveRowTransformationRecord record = {
        .type = type,
        .row = row,
        .source_row = source_row,
        .ratio = ratio,
        .new_side = new_side,
        .indices = matrix->column_indices + start,
        .coefficients = matrix->values + start,
        .length = (size_t) (matrix->row_pointers[row + 1] - start)};
    return presolve_transformation_log_append_row_transformation(
               &presolver->transformations, &record, NULL)
               ? PREFOS_STATUS_OK
               : PREFOS_STATUS_OUT_OF_MEMORY;
}

static PreFOSStatus append_deleted_row_record(PreFOSPresolver *presolver, int row)
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

static PreFOSStatus process_parallel_group(PreFOSPresolver *presolver,
                                        const int *rows, size_t count)
{
    const PreFOSCsrMatrix *matrix = &presolver->original.A;
    double *lower = presolver->working_constraint_lower;
    double *upper = presolver->working_constraint_upper;
    double tolerance = presolver->settings.feasibility_tolerance;
    PresolveSparseRowView view = {
        matrix->rows, matrix->values, matrix->column_indices,
        matrix->row_pointers, matrix->row_pointers + 1, 1};
    PresolveParallelRowReduction reduction;
    PresolveParallelReductionStatus reduction_status;
    size_t position;

    reduction_status = presolve_analyze_parallel_row_group(
        &view, rows, count, lower, upper, tolerance,
        prefos_parallel_values_close, NULL, &reduction);
    if (reduction_status == PRESOLVE_PARALLEL_REDUCTION_INFEASIBLE)
        return PREFOS_STATUS_PRIMAL_INFEASIBLE;
    if (reduction_status == PRESOLVE_PARALLEL_REDUCTION_UNCERTAIN)
        return PREFOS_STATUS_OK;
    if (reduction_status != PRESOLVE_PARALLEL_REDUCTION_OK)
        return PREFOS_STATUS_NUMERICAL_ERROR;

    if (reduction.lower_source_row >= 0)
    {
        PreFOSStatus status = append_row_record(
            presolver, PRESOLVE_ROW_LOWER_CHANGED, reduction.kept_row,
            reduction.lower_source_row, reduction.lower_source_ratio,
            reduction.lower);
        if (status != PREFOS_STATUS_OK) return status;
    }
    if (reduction.upper_source_row >= 0)
    {
        PreFOSStatus status = append_row_record(
            presolver, PRESOLVE_ROW_UPPER_CHANGED, reduction.kept_row,
            reduction.upper_source_row, reduction.upper_source_ratio,
            reduction.upper);
        if (status != PREFOS_STATUS_OK) return status;
    }
    lower[reduction.kept_row] = reduction.lower;
    upper[reduction.kept_row] = reduction.upper;

    for (position = 0; position < count; ++position)
    {
        int row = rows[position];
        PreFOSStatus status;
        if (row == reduction.kept_row) continue;
        status = append_deleted_row_record(presolver, row);
        if (status != PREFOS_STATUS_OK) return status;
        presolver->remove_rows[row] = 1;
        ++presolver->stats.removed_redundant_rows;
    }
    return PREFOS_STATUS_OK;
}

PreFOSStatus prefos_internal_remove_parallel_rows(PreFOSPresolver *presolver)
{
    const PreFOSCsrMatrix *matrix = &presolver->original.A;
    PresolveSparseRowView view;
    int *workspace, *parallel_rows, *support_hashes, *coefficient_hashes;
    int *sort_auxiliary, *group_starts;
    size_t n_groups, group;
    PreFOSTimestamp start, stop;
    PreFOSStatus status = PREFOS_STATUS_OK;
    int detected = 0;
    int over_cpu_budget;

    if (!presolver->settings.remove_redundant_rows || matrix->rows < 2)
        return PREFOS_STATUS_OK;
    over_cpu_budget =
        presolver->settings.parallel_row_max_average_nnz > 0.0 &&
        (long double) matrix->nnz >
            (long double) matrix->rows *
                (long double)
                    presolver->settings.parallel_row_max_average_nnz;
    if (over_cpu_budget && !presolver->settings.structural_reductions_gpu)
    {
        ++presolver->stats.parallel_row_budget_skips;
        return PREFOS_STATUS_OK;
    }
    prefos_internal_timer_now(&start);
    workspace = (int *) prefos_internal_alloc_array(5 * matrix->rows, sizeof(int));
    if (!workspace)
    {
        status = PREFOS_STATUS_OUT_OF_MEMORY;
        goto finish;
    }
    parallel_rows = workspace;
    support_hashes = workspace + matrix->rows;
    coefficient_hashes = workspace + 2 * matrix->rows;
    sort_auxiliary = workspace + 3 * matrix->rows;
    group_starts = workspace + 4 * matrix->rows;
    view = (PresolveSparseRowView){
        matrix->rows, matrix->values, matrix->column_indices,
        matrix->row_pointers, matrix->row_pointers + 1, 1};

    if (presolver->settings.structural_reductions_gpu)
    {
        PreFOSCudaPropagationStatus cuda_status;
        PreFOSCudaWorkspace *cuda_workspace =
            prefos_internal_cuda_workspace_get(presolver, &cuda_status);
        size_t active_rows = 0;
        double gpu_milliseconds = 0.0;
        if (cuda_workspace && cuda_status == PREFOS_CUDA_PROPAGATION_OK)
        {
            cuda_status = prefos_cuda_parallel_row_hash_sort(
                cuda_workspace, presolver->remove_rows, parallel_rows,
                support_hashes, coefficient_hashes, &active_rows,
                &gpu_milliseconds);
        }
        presolver->stats.parallel_row_gpu_milliseconds += gpu_milliseconds;
        if (cuda_status == PREFOS_CUDA_PROPAGATION_OK)
        {
            detected = presolve_collect_parallel_row_groups(
                &view, presolver->settings.feasibility_tolerance,
                parallel_rows, active_rows, support_hashes,
                coefficient_hashes, group_starts, matrix->rows, &n_groups);
            if (detected)
                ++presolver->stats.parallel_row_gpu_passes;
        }
        if (!detected)
        {
            ++presolver->stats.parallel_row_gpu_fallbacks;
        }
    }
    if (!detected)
    {
        if (over_cpu_budget)
        {
            ++presolver->stats.parallel_row_budget_skips;
            goto cleanup;
        }
        detected = presolve_find_parallel_rows(
            &view, prefos_row_is_active, presolver,
            presolver->settings.feasibility_tolerance, NULL, parallel_rows,
            support_hashes, coefficient_hashes, sort_auxiliary, group_starts,
            matrix->rows, &n_groups);
    }
    if (!detected)
    {
        status = PREFOS_STATUS_NUMERICAL_ERROR;
        goto cleanup;
    }
    for (group = 0; group < n_groups; ++group)
    {
        size_t start = (size_t) group_starts[group];
        size_t end = (size_t) group_starts[group + 1];
        status = process_parallel_group(
            presolver, parallel_rows + start, end - start);
        if (status != PREFOS_STATUS_OK) goto cleanup;
    }

cleanup:
    free(workspace);
finish:
    prefos_internal_timer_now(&stop);
    presolver->stats.parallel_row_detection_milliseconds +=
        prefos_internal_timer_elapsed_milliseconds(&start, &stop);
    return status;
}
