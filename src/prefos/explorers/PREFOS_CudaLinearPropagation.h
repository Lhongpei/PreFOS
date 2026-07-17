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

    typedef struct PreFOSCudaWorkspace PreFOSCudaWorkspace;
    typedef PreFOSCudaWorkspace PreFOSCudaLinearPropagationContext;

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

    PreFOSCudaPropagationStatus prefos_cuda_workspace_create(
        size_t rows, size_t columns, size_t nnz, const int *row_pointers,
        const int *column_indices, const double *values,
        const double *constraint_lower, const double *constraint_upper,
        const int *candidate_map, const unsigned char *remove_rows,
        size_t n_cones, const int *cone_types, const int *cone_starts,
        const int *cone_indices, const int *cone_matrix_orders,
        const double *cone_power_alphas, const int *column_to_cone,
        const int *column_to_cone_position, PreFOSCudaWorkspace **context,
        double *setup_milliseconds, size_t *long_rows);

    void prefos_cuda_workspace_free(PreFOSCudaWorkspace *context);

    PreFOSCudaPropagationStatus prefos_cuda_workspace_column_stats(
        PreFOSCudaWorkspace *context, const double *constraint_lower,
        const double *constraint_upper, const unsigned char *remove_rows,
        int *column_degrees, unsigned char *down_locked,
        unsigned char *up_locked, double *milliseconds);

    PreFOSCudaPropagationStatus prefos_cuda_parallel_row_hash_sort(
        PreFOSCudaWorkspace *context, const unsigned char *remove_rows,
        int *sorted_rows, int *support_hashes, int *coefficient_hashes,
        size_t *active_rows, double *milliseconds);

    enum
    {
        PREFOS_CUDA_ROW_LOWER_REDUNDANT = 1 << 0,
        PREFOS_CUDA_ROW_UPPER_REDUNDANT = 1 << 1,
        PREFOS_CUDA_ROW_INFEASIBLE = 1 << 2,
        PREFOS_CUDA_ROW_NEEDS_CPU = 1 << 3,
        PREFOS_CUDA_ROW_CONE_STRENGTHENED = 1 << 4
    };

    PreFOSCudaPropagationStatus prefos_cuda_cone_activity_candidates(
        PreFOSCudaWorkspace *context, const double *lower_bounds,
        const double *upper_bounds, const double *constraint_lower,
        const double *constraint_upper, const unsigned char *remove_rows,
        double feasibility_tolerance, unsigned char *row_flags,
        double *milliseconds);

    enum
    {
        PREFOS_CUDA_CONE_PROCESSED = 1 << 0,
        PREFOS_CUDA_CONE_INFEASIBLE = 1 << 1,
        PREFOS_CUDA_CONE_NEEDS_CPU = 1 << 2
    };

    PreFOSCudaPropagationStatus prefos_cuda_cone_envelope_round(
        PreFOSCudaWorkspace *context, const double *lower_bounds,
        const double *upper_bounds, double feasibility_tolerance,
        double *lower_candidates, double *upper_candidates,
        unsigned char *cone_flags, double *milliseconds);

    PreFOSCudaPropagationStatus prefos_cuda_linear_column_stats(
        size_t rows, size_t columns, size_t nnz, const int *row_pointers,
        const int *column_indices, const double *values,
        const double *constraint_lower, const double *constraint_upper,
        const unsigned char *remove_rows, int *column_degrees,
        unsigned char *down_locked, unsigned char *up_locked,
        double *milliseconds);

    PreFOSCudaPropagationStatus prefos_cuda_linear_propagation_create(
        size_t rows, size_t columns, size_t nnz, const int *row_pointers,
        const int *column_indices, const double *values,
        const double *constraint_lower, const double *constraint_upper,
        const int *candidate_map, const unsigned char *remove_rows,
        PreFOSCudaLinearPropagationContext **context, double *setup_milliseconds,
        size_t *long_rows);

    PreFOSCudaPropagationStatus prefos_cuda_linear_propagation_round(
        PreFOSCudaLinearPropagationContext *context, const double *lower_bounds,
        const double *upper_bounds, const double *constraint_lower,
        const double *constraint_upper, const unsigned char *remove_rows,
        const int *candidate_map, double feasibility_tolerance,
        double maximum_inferred_bound_magnitude, double *lower_candidates,
        double *upper_candidates, int *lower_source_rows, int *upper_source_rows,
        int *suspected_infeasible_row, double *transfer_milliseconds,
        double *kernel_milliseconds);

    void prefos_cuda_linear_propagation_free(PreFOSCudaLinearPropagationContext *context);

#ifdef __cplusplus
}
#endif

#endif
