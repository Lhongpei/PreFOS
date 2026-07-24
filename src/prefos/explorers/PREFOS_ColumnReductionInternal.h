/*
 * Copyright 2026 Hongpei Li
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef PREFOS_COLUMN_REDUCTION_INTERNAL_H
#define PREFOS_COLUMN_REDUCTION_INTERNAL_H

#include "PREFOS_Internal.h"

typedef struct PreFOSColumnWorkspace
{
    int *starts;
    int *ends;
    int *rows;
    double *values;
    unsigned char *quadratic;
    unsigned char *factor;
    unsigned char *protected_target;
    unsigned char *dirty_row;
    int *row_degrees;
    int *live_degrees;
    double *column_max_abs_coefficient;
    int *gpu_degrees;
    unsigned char *gpu_down_locked;
    unsigned char *gpu_up_locked;
    int *gpu_singleton_candidates;
    double *objective;
    int gpu_stats_valid;
    int gpu_csc_valid;
    int gpu_singleton_candidates_valid;
    size_t nnz;
    size_t max_row_nnz;
    size_t n_gpu_singleton_candidates;
    size_t removed_row_cursor;
} PreFOSColumnWorkspace;

PREFOS_INTERNAL void
prefos_internal_free_column_workspace(PreFOSColumnWorkspace *workspace);

PREFOS_INTERNAL PreFOSStatus prefos_internal_build_column_workspace(
    PreFOSPresolver *presolver, PreFOSColumnWorkspace *workspace);

PREFOS_INTERNAL PreFOSStatus prefos_internal_build_column_workspace_cpu(
    PreFOSPresolver *presolver, PreFOSColumnWorkspace *workspace);

PREFOS_INTERNAL void prefos_internal_refresh_column_workspace(
    const PreFOSPresolver *presolver, PreFOSColumnWorkspace *workspace);

PREFOS_INTERNAL void prefos_internal_update_column_live_degrees(
    const PreFOSPresolver *presolver, PreFOSColumnWorkspace *workspace);

PREFOS_INTERNAL PreFOSStatus prefos_internal_prepare_column_workspace(
    PreFOSPresolver *presolver, PreFOSColumnWorkspace *workspace);

PREFOS_INTERNAL PreFOSStatus prefos_internal_synchronize_column_workspace(
    PreFOSPresolver *presolver, PreFOSColumnWorkspace *workspace);

PREFOS_INTERNAL PreFOSStatus prefos_internal_populate_gpu_column_stats(
    PreFOSPresolver *presolver, PreFOSColumnWorkspace *workspace);

PREFOS_INTERNAL int prefos_internal_column_is_linear_box(
    const PreFOSPresolver *presolver, const PreFOSColumnWorkspace *workspace,
    int column);

PREFOS_INTERNAL PreFOSStatus prefos_internal_effective_row_bounds(
    const PreFOSPresolver *presolver, size_t row, double *lower, double *upper);

PREFOS_INTERNAL size_t prefos_internal_collect_live_row(
    const PreFOSPresolver *presolver, size_t row, int *columns,
    double *coefficients, size_t capacity);

PREFOS_INTERNAL PreFOSStatus prefos_internal_append_column_substitution(
    PreFOSPresolver *presolver, int column, const int *targets,
    const double *scales, size_t term_count, int source_row, double constant,
    double pivot, PreFOSColumnWorkspace *workspace,
    PreFOSSubstitutionMode mode);

PREFOS_INTERNAL PreFOSStatus
prefos_internal_reduce_empty_and_dual_fixed_columns(
    PreFOSPresolver *presolver, const PreFOSColumnWorkspace *workspace);

PREFOS_INTERNAL PreFOSStatus prefos_internal_reduce_singleton_columns(
    PreFOSPresolver *presolver, PreFOSColumnWorkspace *workspace,
    int allow_one_sided);

PREFOS_INTERNAL PreFOSStatus prefos_internal_reduce_bounded_doubletons(
    PreFOSPresolver *presolver, PreFOSColumnWorkspace *workspace);

PREFOS_INTERNAL PreFOSStatus
prefos_internal_reduce_parallel_column_groups(
    PreFOSPresolver *presolver, PreFOSColumnWorkspace *workspace);

PREFOS_INTERNAL PreFOSStatus
prefos_internal_reduce_linear_columns_in_workspace(
    PreFOSPresolver *presolver, PreFOSColumnWorkspace *workspace,
    int allow_one_sided_singletons);

#endif
