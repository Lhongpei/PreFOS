/*
 * Copyright 2026 Hongpei Li
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef PREFOS_CUDA_LINEAR_PROPAGATION_H
#define PREFOS_CUDA_LINEAR_PROPAGATION_H

#include <stddef.h>

#ifdef __cplusplus
extern "C"
{
#endif

    typedef struct PreFOSCudaLinearPropagationContext PreFOSCudaLinearPropagationContext;

    typedef enum
    {
        PREFOS_CUDA_PROPAGATION_OK = 0,
        PREFOS_CUDA_PROPAGATION_UNAVAILABLE,
        PREFOS_CUDA_PROPAGATION_OUT_OF_MEMORY,
        PREFOS_CUDA_PROPAGATION_ERROR
    } PreFOSCudaPropagationStatus;

    PreFOSCudaPropagationStatus prefos_cuda_linear_propagation_warmup(void);
    int prefos_cuda_linear_propagation_warmup_async(void);
    int prefos_cuda_linear_propagation_warmup_ready(void);
    PreFOSCudaPropagationStatus prefos_cuda_linear_propagation_warmup_wait(void);
    void prefos_cuda_linear_propagation_release_cache(void);

    PreFOSCudaPropagationStatus prefos_cuda_linear_propagation_create(
        size_t rows, size_t columns, size_t nnz, const int *row_pointers,
        const int *column_indices, const double *values,
        const double *constraint_lower, const double *constraint_upper,
        const int *candidate_map, const unsigned char *remove_rows,
        PreFOSCudaLinearPropagationContext **context, double *setup_milliseconds,
        size_t *long_rows);

    PreFOSCudaPropagationStatus prefos_cuda_linear_propagation_round(
        PreFOSCudaLinearPropagationContext *context, const double *lower_bounds,
        const double *upper_bounds, double feasibility_tolerance,
        double maximum_inferred_bound_magnitude, double *lower_candidates,
        double *upper_candidates, int *lower_source_rows, int *upper_source_rows,
        int *suspected_infeasible_row, double *transfer_milliseconds,
        double *kernel_milliseconds);

    void prefos_cuda_linear_propagation_free(PreFOSCudaLinearPropagationContext *context);

#ifdef __cplusplus
}
#endif

#endif
