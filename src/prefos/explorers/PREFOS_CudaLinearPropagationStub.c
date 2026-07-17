/*
 * Copyright 2026 Hongpei Li
 * SPDX-License-Identifier: Apache-2.0
 */

#include "PREFOS_CudaLinearPropagation.h"

PreFOSCudaPropagationStatus prefos_cuda_linear_propagation_warmup(void)
{
    return PREFOS_CUDA_PROPAGATION_UNAVAILABLE;
}

int prefos_cuda_linear_propagation_warmup_async(void)
{
    return 0;
}

int prefos_cuda_linear_propagation_warmup_ready(void)
{
    return 0;
}

PreFOSCudaPropagationStatus prefos_cuda_linear_propagation_warmup_wait(void)
{
    return PREFOS_CUDA_PROPAGATION_UNAVAILABLE;
}

void prefos_cuda_linear_propagation_release_cache(void) {}

PreFOSCudaPropagationStatus prefos_cuda_linear_column_stats(
    size_t rows, size_t columns, size_t nnz, const int *row_pointers,
    const int *column_indices, const double *values,
    const double *constraint_lower, const double *constraint_upper,
    const unsigned char *remove_rows, int *column_degrees,
    unsigned char *down_locked, unsigned char *up_locked, double *milliseconds)
{
    (void) rows;
    (void) columns;
    (void) nnz;
    (void) row_pointers;
    (void) column_indices;
    (void) values;
    (void) constraint_lower;
    (void) constraint_upper;
    (void) remove_rows;
    (void) column_degrees;
    (void) down_locked;
    (void) up_locked;
    if (milliseconds) *milliseconds = 0.0;
    return PREFOS_CUDA_PROPAGATION_UNAVAILABLE;
}

PreFOSCudaPropagationStatus prefos_cuda_linear_propagation_create(
    size_t rows, size_t columns, size_t nnz, const int *row_pointers,
    const int *column_indices, const double *values, const double *constraint_lower,
    const double *constraint_upper, const int *candidate_map,
    const unsigned char *remove_rows, PreFOSCudaLinearPropagationContext **context,
    double *setup_milliseconds, size_t *long_rows)
{
    (void) rows;
    (void) columns;
    (void) nnz;
    (void) row_pointers;
    (void) column_indices;
    (void) values;
    (void) constraint_lower;
    (void) constraint_upper;
    (void) candidate_map;
    (void) remove_rows;
    if (context) *context = NULL;
    if (setup_milliseconds) *setup_milliseconds = 0.0;
    if (long_rows) *long_rows = 0;
    return PREFOS_CUDA_PROPAGATION_UNAVAILABLE;
}

PreFOSCudaPropagationStatus prefos_cuda_linear_propagation_round(
    PreFOSCudaLinearPropagationContext *context, const double *lower_bounds,
    const double *upper_bounds, double feasibility_tolerance,
    double maximum_inferred_bound_magnitude, double *lower_candidates,
    double *upper_candidates, int *lower_source_rows, int *upper_source_rows,
    int *suspected_infeasible_row, double *transfer_milliseconds,
    double *kernel_milliseconds)
{
    (void) context;
    (void) lower_bounds;
    (void) upper_bounds;
    (void) feasibility_tolerance;
    (void) maximum_inferred_bound_magnitude;
    (void) lower_candidates;
    (void) upper_candidates;
    (void) lower_source_rows;
    (void) upper_source_rows;
    if (suspected_infeasible_row) *suspected_infeasible_row = -1;
    if (transfer_milliseconds) *transfer_milliseconds = 0.0;
    if (kernel_milliseconds) *kernel_milliseconds = 0.0;
    return PREFOS_CUDA_PROPAGATION_UNAVAILABLE;
}

void prefos_cuda_linear_propagation_free(PreFOSCudaLinearPropagationContext *context)
{
    (void) context;
}
