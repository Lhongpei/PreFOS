/*
 * Copyright 2026 Hongpei Li
 * SPDX-License-Identifier: Apache-2.0
 */

#include "PREFOS_CudaBackend.h"

PreFOSCudaWorkspace *
prefos_internal_cuda_workspace_get(PreFOSPresolver *presolver,
                                   PreFOSCudaPropagationStatus *status)
{
    const PreFOSProblemData *problem;
    int *cone_types = NULL, *cone_starts = NULL, *cone_indices = NULL;
    int *cone_matrix_orders = NULL, *column_to_cone = NULL;
    int *column_to_cone_position = NULL;
    double *cone_power_alphas = NULL;
    size_t total_indices = 0, k, position;
    PreFOSCudaPropagationStatus result;

    if (status) *status = PREFOS_CUDA_PROPAGATION_ERROR;
    if (!presolver) return NULL;
    if (presolver->cuda_workspace)
    {
        if (status) *status = PREFOS_CUDA_PROPAGATION_OK;
        return presolver->cuda_workspace;
    }
    problem = &presolver->original;
    if (problem->n_cones > (size_t) UINT_MAX ||
        problem->A.rows > (size_t) UINT_MAX ||
        problem->A.nnz > (size_t) INT_MAX)
        return NULL;
    for (k = 0; k < problem->n_cones; ++k)
    {
        if (problem->cones[k].dimension > (size_t) INT_MAX ||
            total_indices > (size_t) INT_MAX - problem->cones[k].dimension)
        {
            if (status) *status = PREFOS_CUDA_PROPAGATION_ERROR;
            return NULL;
        }
        total_indices += problem->cones[k].dimension;
    }

    cone_types = (int *) prefos_internal_alloc_array(problem->n_cones, sizeof(int));
    cone_starts =
        (int *) prefos_internal_alloc_array(problem->n_cones + 1, sizeof(int));
    cone_indices =
        (int *) prefos_internal_alloc_array(total_indices, sizeof(int));
    cone_matrix_orders =
        (int *) prefos_internal_alloc_array(problem->n_cones, sizeof(int));
    cone_power_alphas =
        (double *) prefos_internal_alloc_array(problem->n_cones, sizeof(double));
    column_to_cone =
        (int *) prefos_internal_alloc_array(problem->n, sizeof(int));
    column_to_cone_position =
        (int *) prefos_internal_alloc_array(problem->n, sizeof(int));
    if ((problem->n_cones > 0 &&
         (!cone_types || !cone_starts || !cone_matrix_orders ||
          !cone_power_alphas)) ||
        (total_indices > 0 && !cone_indices) ||
        (problem->n > 0 && (!column_to_cone || !column_to_cone_position)))
    {
        result = PREFOS_CUDA_PROPAGATION_OUT_OF_MEMORY;
        goto cleanup;
    }

    for (position = 0; position < problem->n; ++position)
    {
        column_to_cone[position] = -1;
        column_to_cone_position[position] = -1;
    }
    position = 0;
    for (k = 0; k < problem->n_cones; ++k)
    {
        const PreFOSConeBlock *cone = &problem->cones[k];
        size_t local;
        cone_starts[k] = (int) position;
        cone_types[k] =
            presolver->converted_affine_cones &&
                    presolver->converted_affine_cones[k]
                ? -1
                : (int) cone->type;
        cone_matrix_orders[k] = (int) cone->matrix_order;
        cone_power_alphas[k] = cone->power_alpha;
        for (local = 0; local < cone->dimension; ++local)
        {
            int column = cone->indices[local];
            cone_indices[position++] = column;
            if (cone_types[k] >= 0)
            {
                column_to_cone[column] = (int) k;
                column_to_cone_position[column] = (int) local;
            }
        }
    }
    cone_starts[problem->n_cones] = (int) position;

    result = prefos_cuda_workspace_create(
        problem->A.rows, problem->n, problem->A.nnz,
        problem->A.row_pointers, problem->A.column_indices, problem->A.values,
        presolver->working_constraint_lower,
        presolver->working_constraint_upper, presolver->variable_to_box,
        presolver->remove_rows, problem->n_cones, cone_types, cone_starts,
        cone_indices, cone_matrix_orders, cone_power_alphas, column_to_cone,
        column_to_cone_position, &presolver->cuda_workspace,
        &presolver->stats.cuda_workspace_setup_milliseconds,
        &presolver->stats.linear_gpu_long_rows);

cleanup:
    free(cone_types);
    free(cone_starts);
    free(cone_indices);
    free(cone_matrix_orders);
    free(cone_power_alphas);
    free(column_to_cone);
    free(column_to_cone_position);
    if (status) *status = result;
    return result == PREFOS_CUDA_PROPAGATION_OK ? presolver->cuda_workspace : NULL;
}

void prefos_internal_cuda_workspace_release(PreFOSPresolver *presolver)
{
    if (!presolver || !presolver->cuda_workspace) return;
    prefos_cuda_workspace_free(presolver->cuda_workspace);
    presolver->cuda_workspace = NULL;
}
