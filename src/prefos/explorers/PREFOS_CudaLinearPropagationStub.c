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

PreFOSCudaPropagationStatus prefos_cuda_workspace_create(
    size_t rows, size_t columns, size_t nnz, const int *row_pointers,
    const int *column_indices, const double *values,
    const double *constraint_lower, const double *constraint_upper,
    const int *candidate_map, const unsigned char *remove_rows,
    size_t n_cones, const int *cone_types, const int *cone_starts,
    const int *cone_indices, const int *cone_matrix_orders,
    const double *cone_power_alphas, const int *column_to_cone,
    const int *column_to_cone_position, PreFOSCudaWorkspace **context,
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
    (void) n_cones;
    (void) cone_types;
    (void) cone_starts;
    (void) cone_indices;
    (void) cone_matrix_orders;
    (void) cone_power_alphas;
    (void) column_to_cone;
    (void) column_to_cone_position;
    if (context) *context = NULL;
    if (setup_milliseconds) *setup_milliseconds = 0.0;
    if (long_rows) *long_rows = 0;
    return PREFOS_CUDA_PROPAGATION_UNAVAILABLE;
}

void prefos_cuda_workspace_free(PreFOSCudaWorkspace *context)
{
    (void) context;
}

PreFOSCudaPropagationStatus prefos_cuda_workspace_column_stats(
    PreFOSCudaWorkspace *context, const double *constraint_lower,
    const double *constraint_upper, const unsigned char *remove_rows,
    int *column_degrees, unsigned char *down_locked,
    unsigned char *up_locked, double *milliseconds)
{
    (void) context;
    (void) constraint_lower;
    (void) constraint_upper;
    (void) remove_rows;
    (void) column_degrees;
    (void) down_locked;
    (void) up_locked;
    if (milliseconds) *milliseconds = 0.0;
    return PREFOS_CUDA_PROPAGATION_UNAVAILABLE;
}

PreFOSCudaPropagationStatus prefos_cuda_parallel_row_hash_sort(
    PreFOSCudaWorkspace *context, const unsigned char *remove_rows,
    int *sorted_rows, int *support_hashes, int *coefficient_hashes,
    size_t *active_rows, double *milliseconds)
{
    (void) context;
    (void) remove_rows;
    (void) sorted_rows;
    (void) support_hashes;
    (void) coefficient_hashes;
    if (active_rows) *active_rows = 0;
    if (milliseconds) *milliseconds = 0.0;
    return PREFOS_CUDA_PROPAGATION_UNAVAILABLE;
}

PreFOSCudaPropagationStatus prefos_cuda_cone_activity_candidates(
    PreFOSCudaWorkspace *context, const double *lower_bounds,
    const double *upper_bounds, const double *constraint_lower,
    const double *constraint_upper, const unsigned char *remove_rows,
    double feasibility_tolerance, unsigned char *row_flags,
    double *milliseconds)
{
    (void) context;
    (void) lower_bounds;
    (void) upper_bounds;
    (void) constraint_lower;
    (void) constraint_upper;
    (void) remove_rows;
    (void) feasibility_tolerance;
    (void) row_flags;
    if (milliseconds) *milliseconds = 0.0;
    return PREFOS_CUDA_PROPAGATION_UNAVAILABLE;
}

PreFOSCudaPropagationStatus prefos_cuda_cone_envelope_round(
    PreFOSCudaWorkspace *context, const double *lower_bounds,
    const double *upper_bounds, double feasibility_tolerance,
    double *lower_candidates, double *upper_candidates,
    unsigned char *cone_flags, double *milliseconds)
{
    (void) context;
    (void) lower_bounds;
    (void) upper_bounds;
    (void) feasibility_tolerance;
    (void) lower_candidates;
    (void) upper_candidates;
    (void) cone_flags;
    if (milliseconds) *milliseconds = 0.0;
    return PREFOS_CUDA_PROPAGATION_UNAVAILABLE;
}

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
    const double *upper_bounds, const double *constraint_lower,
    const double *constraint_upper, const unsigned char *remove_rows,
    const int *candidate_map, double feasibility_tolerance,
    double maximum_inferred_bound_magnitude, double *lower_candidates,
    double *upper_candidates, int *lower_source_rows, int *upper_source_rows,
    int *suspected_infeasible_row, double *transfer_milliseconds,
    double *kernel_milliseconds)
{
    (void) context;
    (void) lower_bounds;
    (void) upper_bounds;
    (void) constraint_lower;
    (void) constraint_upper;
    (void) remove_rows;
    (void) candidate_map;
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
