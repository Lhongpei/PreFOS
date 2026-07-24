/*
 * Copyright 2026 Hongpei Li
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef PREFOS_CUDA_WORKSPACE_INTERNAL_CUH
#define PREFOS_CUDA_WORKSPACE_INTERNAL_CUH

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

struct PreFOSCudaWorkspace
{
    cudaStream_t stream;
    size_t rows;
    size_t columns;
    size_t nnz;
    size_t n_cones;
    size_t n_cone_indices;
    size_t n_long_rows;
    int *row_pointers;
    int *column_indices;
    double *values;
    double *constraint_lower;
    double *constraint_upper;
    int *candidate_map;
    unsigned char *remove_rows;
    double *lower_bounds;
    double *upper_bounds;
    double *lower_candidates;
    double *upper_candidates;
    int *lower_source_rows;
    int *upper_source_rows;
    int *long_rows;
    int *suspected_infeasible_row;
    int *numerical_error;
    int *cone_types;
    int *cone_starts;
    int *cone_indices;
    int *cone_matrix_orders;
    double *cone_power_alphas;
    int *column_to_cone;
    int *column_to_cone_position;

    int csc_ready;
    size_t csc_nnz;
    int *csc_column_pointers;
    int *csc_row_indices;
    double *csc_values;

    size_t affine_rows;
    size_t affine_nnz;
    size_t n_affine_cones;
    int *affine_row_pointers;
    int *affine_column_indices;
    double *affine_values;
    double *affine_offsets;
    int *affine_cone_types;
    int *affine_cone_starts;
    int *affine_cone_indices;
    int *affine_cone_matrix_orders;
    double *affine_cone_power_alphas;
    double *affine_lower_bounds;
    double *affine_upper_bounds;
    double *affine_lower_candidates;
    double *affine_upper_candidates;

    int cone_activity_groups_ready;
    size_t n_cone_activity_groups;
    uint64_t *cone_activity_group_keys;
    int *cone_activity_group_offsets;
    int *cone_activity_sorted_positions;
};

#endif
