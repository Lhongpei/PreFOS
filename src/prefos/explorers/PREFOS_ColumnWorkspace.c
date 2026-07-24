/*
 * Copyright 2026 Hongpei Li
 * SPDX-License-Identifier: Apache-2.0
 */

#include "PREFOS_ColumnReductionInternal.h"
#include "PREFOS_CudaBackend.h"
#include "PREFOS_CudaLinearPropagation.h"

void prefos_internal_free_column_workspace(PreFOSColumnWorkspace *workspace)
{
    if (!workspace) return;
    free(workspace->starts);
    free(workspace->rows);
    free(workspace->values);
    free(workspace->quadratic);
    free(workspace->factor);
    free(workspace->protected_target);
    free(workspace->dirty_row);
    free(workspace->gpu_degrees);
    free(workspace->gpu_down_locked);
    free(workspace->gpu_up_locked);
    free(workspace->gpu_singleton_candidates);
    free(workspace->objective);
    memset(workspace, 0, sizeof(*workspace));
}

static PreFOSStatus build_column_csc_cpu(
    const PreFOSPresolver *presolver, PreFOSColumnWorkspace *workspace)
{
    const PreFOSProblemData *problem = &presolver->original;
    int *cursor;
    size_t row, position;

    cursor = (int *) prefos_internal_alloc_array(problem->n, sizeof(int));
    if (problem->n > 0 && !cursor) return PREFOS_STATUS_OUT_OF_MEMORY;
    memset(workspace->starts, 0, (problem->n + 1) * sizeof(int));
    for (row = 0; row < problem->A.rows; ++row)
    {
        int p;
        if (presolver->remove_rows[row]) continue;
        for (p = problem->A.row_pointers[row];
             p < problem->A.row_pointers[row + 1]; ++p)
        {
            int column = problem->A.column_indices[p];
            if (problem->A.values[p] != 0.0)
                ++workspace->starts[column + 1];
        }
    }
    for (position = 0; position < problem->n; ++position)
        workspace->starts[position + 1] += workspace->starts[position];
    workspace->nnz = (size_t) workspace->starts[problem->n];
    workspace->rows =
        (int *) prefos_internal_alloc_array(workspace->nnz, sizeof(int));
    workspace->values =
        (double *) prefos_internal_alloc_array(workspace->nnz, sizeof(double));
    if (workspace->nnz > 0 && (!workspace->rows || !workspace->values))
    {
        free(cursor);
        free(workspace->rows);
        free(workspace->values);
        workspace->rows = NULL;
        workspace->values = NULL;
        return PREFOS_STATUS_OUT_OF_MEMORY;
    }
    if (problem->n > 0)
        memcpy(cursor, workspace->starts, problem->n * sizeof(int));
    for (row = 0; row < problem->A.rows; ++row)
    {
        int p;
        if (presolver->remove_rows[row]) continue;
        for (p = problem->A.row_pointers[row];
             p < problem->A.row_pointers[row + 1]; ++p)
        {
            int column, write;
            if (problem->A.values[p] == 0.0) continue;
            column = problem->A.column_indices[p];
            write = cursor[column]++;
            workspace->rows[write] = (int) row;
            workspace->values[write] = problem->A.values[p];
        }
    }
    free(cursor);
    return PREFOS_STATUS_OK;
}

static PreFOSStatus build_column_csc_gpu(
    PreFOSPresolver *presolver, PreFOSColumnWorkspace *workspace,
    int *used_gpu)
{
    const PreFOSProblemData *problem = &presolver->original;
    PreFOSCudaPropagationStatus cuda_status;
    PreFOSCudaWorkspace *cuda_workspace;
    double milliseconds = 0.0, copy_milliseconds = 0.0;
    size_t active_nnz = 0;

    *used_gpu = 0;
    if (!presolver->settings.structural_reductions_gpu)
        return PREFOS_STATUS_OK;
    cuda_workspace =
        prefos_internal_cuda_workspace_get(presolver, &cuda_status);
    if (cuda_workspace && cuda_status == PREFOS_CUDA_PROPAGATION_OK)
        cuda_status = prefos_cuda_workspace_build_csc(
            cuda_workspace, presolver->remove_rows, workspace->starts,
            &active_nnz, &milliseconds);
    presolver->stats.column_csc_gpu_milliseconds += milliseconds;
    if (!cuda_workspace || cuda_status != PREFOS_CUDA_PROPAGATION_OK)
    {
        ++presolver->stats.column_csc_gpu_fallbacks;
        return PREFOS_STATUS_OK;
    }
    workspace->rows =
        (int *) prefos_internal_alloc_array(active_nnz, sizeof(int));
    workspace->values =
        (double *) prefos_internal_alloc_array(active_nnz, sizeof(double));
    if (active_nnz > 0 && (!workspace->rows || !workspace->values))
    {
        free(workspace->rows);
        free(workspace->values);
        workspace->rows = NULL;
        workspace->values = NULL;
        return PREFOS_STATUS_OUT_OF_MEMORY;
    }
    cuda_status = prefos_cuda_workspace_copy_csc(
        cuda_workspace, workspace->rows, workspace->values,
        &copy_milliseconds);
    presolver->stats.column_csc_gpu_milliseconds += copy_milliseconds;
    if (cuda_status != PREFOS_CUDA_PROPAGATION_OK)
    {
        free(workspace->rows);
        free(workspace->values);
        workspace->rows = NULL;
        workspace->values = NULL;
        memset(workspace->starts, 0, (problem->n + 1) * sizeof(int));
        ++presolver->stats.column_csc_gpu_fallbacks;
        return PREFOS_STATUS_OK;
    }
    workspace->nnz = active_nnz;
    workspace->gpu_csc_valid = 1;
    ++presolver->stats.column_csc_gpu_builds;
    *used_gpu = 1;
    return PREFOS_STATUS_OK;
}

PreFOSStatus prefos_internal_build_column_workspace(
    PreFOSPresolver *presolver, PreFOSColumnWorkspace *workspace)
{
    const PreFOSProblemData *problem = &presolver->original;
    size_t row;
    int used_gpu = 0;
    PreFOSStatus status;

    memset(workspace, 0, sizeof(*workspace));
    workspace->starts = (int *) calloc(problem->n + 1, sizeof(int));
    workspace->quadratic =
        (unsigned char *) calloc(problem->n, sizeof(unsigned char));
    workspace->factor =
        (unsigned char *) calloc(problem->n, sizeof(unsigned char));
    workspace->protected_target =
        (unsigned char *) calloc(problem->n, sizeof(unsigned char));
    workspace->dirty_row =
        (unsigned char *) calloc(problem->A.rows, sizeof(unsigned char));
    workspace->objective = (double *) calloc(problem->n, sizeof(double));
    if (!workspace->starts ||
        (problem->n > 0 &&
         (!workspace->quadratic || !workspace->factor ||
          !workspace->protected_target || !workspace->objective)) ||
        (problem->A.rows > 0 && !workspace->dirty_row))
    {
        prefos_internal_free_column_workspace(workspace);
        return PREFOS_STATUS_OUT_OF_MEMORY;
    }
    status = build_column_csc_gpu(presolver, workspace, &used_gpu);
    if (status != PREFOS_STATUS_OK)
    {
        prefos_internal_free_column_workspace(workspace);
        return status;
    }
    if (!used_gpu)
    {
        status = build_column_csc_cpu(presolver, workspace);
        if (status != PREFOS_STATUS_OK)
        {
            prefos_internal_free_column_workspace(workspace);
            return status;
        }
    }

    for (row = 0; row < problem->Q.rows; ++row)
    {
        int p;
        for (p = problem->Q.row_pointers[row];
             p < problem->Q.row_pointers[row + 1]; ++p)
        {
            if (problem->Q.values[p] == 0.0) continue;
            workspace->quadratic[row] = 1;
            workspace->quadratic[problem->Q.column_indices[p]] = 1;
        }
    }
    for (row = 0; row < problem->R.rows; ++row)
    {
        int p;
        for (p = problem->R.row_pointers[row];
             p < problem->R.row_pointers[row + 1]; ++p)
            if (problem->R.values[p] != 0.0)
                workspace->factor[problem->R.column_indices[p]] = 1;
    }
    {
        double ignored_offset;
        PreFOSStatus status = prefos_internal_expand_linear_objective(
            presolver, workspace->objective, &ignored_offset);
        if (status != PREFOS_STATUS_OK)
        {
            prefos_internal_free_column_workspace(workspace);
            return status;
        }
    }
    return PREFOS_STATUS_OK;
}

PreFOSStatus prefos_internal_populate_gpu_column_stats(
    PreFOSPresolver *presolver, PreFOSColumnWorkspace *workspace)
{
    const PreFOSProblemData *problem = &presolver->original;
    PreFOSCudaPropagationStatus cuda_status;
    PreFOSCudaWorkspace *cuda_workspace;
    double milliseconds = 0.0;
    if (!presolver->settings.structural_reductions_gpu) return PREFOS_STATUS_OK;
    workspace->gpu_degrees =
        (int *) prefos_internal_alloc_array(problem->n, sizeof(int));
    workspace->gpu_down_locked =
        (unsigned char *) prefos_internal_alloc_array(problem->n, sizeof(unsigned char));
    workspace->gpu_up_locked =
        (unsigned char *) prefos_internal_alloc_array(problem->n, sizeof(unsigned char));
    if (problem->n > 0 &&
        (!workspace->gpu_degrees || !workspace->gpu_down_locked ||
         !workspace->gpu_up_locked))
        return PREFOS_STATUS_OUT_OF_MEMORY;
    cuda_workspace =
        prefos_internal_cuda_workspace_get(presolver, &cuda_status);
    if (cuda_workspace && cuda_status == PREFOS_CUDA_PROPAGATION_OK)
        cuda_status = prefos_cuda_workspace_column_stats(
            cuda_workspace, presolver->working_constraint_lower,
            presolver->working_constraint_upper, presolver->remove_rows,
            workspace->gpu_degrees, workspace->gpu_down_locked,
            workspace->gpu_up_locked, &milliseconds);
    if (cuda_status == PREFOS_CUDA_PROPAGATION_OK)
    {
        workspace->gpu_stats_valid = 1;
        ++presolver->stats.structural_gpu_passes;
    }
    else
    {
        free(workspace->gpu_degrees);
        free(workspace->gpu_down_locked);
        free(workspace->gpu_up_locked);
        workspace->gpu_degrees = NULL;
        workspace->gpu_down_locked = NULL;
        workspace->gpu_up_locked = NULL;
        ++presolver->stats.structural_gpu_fallbacks;
    }
    (void) milliseconds;
    return PREFOS_STATUS_OK;
}

int prefos_internal_column_is_linear_box(
    const PreFOSPresolver *presolver, const PreFOSColumnWorkspace *workspace,
    int column)
{
    return column >= 0 && (size_t) column < presolver->original.n &&
           presolver->variable_to_box[column] >= 0 &&
           !presolver->is_fixed[column] && !presolver->is_substituted[column] &&
           !presolver->is_parallel_removed[column] &&
           !presolver->affine_protected_columns[column] &&
           !workspace->quadratic[column] && !workspace->factor[column];
}

void prefos_internal_mark_fixed_column(PreFOSPresolver *presolver, int column,
                                       double value)
{
    presolver->is_fixed[column] = 1;
    presolver->fixed_values[column] = value;
    presolver->propagation_lower[column] = value;
    presolver->propagation_upper[column] = value;
}

PreFOSStatus prefos_internal_effective_row_bounds(
    const PreFOSPresolver *presolver, size_t row, double *lower, double *upper)
{
    const PreFOSCsrMatrix *A = &presolver->original.A;
    double shift = 0.0;
    int p;
    for (p = A->row_pointers[row]; p < A->row_pointers[row + 1]; ++p)
    {
        int column = A->column_indices[p];
        if (presolver->is_fixed[column] &&
            !prefos_internal_safe_add_product(
                &shift, A->values[p], presolver->fixed_values[column]))
            return PREFOS_STATUS_NUMERICAL_ERROR;
    }
    *lower = presolver->working_constraint_lower[row] - shift;
    *upper = presolver->working_constraint_upper[row] - shift;
    return isnan(*lower) || isnan(*upper) ? PREFOS_STATUS_NUMERICAL_ERROR
                                          : PREFOS_STATUS_OK;
}

size_t prefos_internal_collect_live_row(const PreFOSPresolver *presolver,
                                        size_t row, int *columns,
                                        double *coefficients, size_t capacity)
{
    const PreFOSCsrMatrix *A = &presolver->original.A;
    size_t count = 0;
    int p;
    for (p = A->row_pointers[row]; p < A->row_pointers[row + 1]; ++p)
    {
        int column = A->column_indices[p];
        if (A->values[p] == 0.0 || presolver->is_fixed[column] ||
            presolver->is_substituted[column] ||
            presolver->is_parallel_removed[column])
            continue;
        if (count < capacity)
        {
            columns[count] = column;
            coefficients[count] = A->values[p];
        }
        ++count;
    }
    return count;
}
