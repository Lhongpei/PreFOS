/*
 * Copyright 2025-2026 Daniel Cederberg
 * Copyright 2026 Hongpei Li
 *
 * Modified for PreFOS in 2026.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef LINEAR_PROPAGATION_KERNEL_H
#define LINEAR_PROPAGATION_KERNEL_H

#include <stddef.h>
#include <stdint.h>

#define PRESOLVE_DEFAULT_MAX_INFERRED_BOUND_MAGNITUDE 1e7

typedef struct
{
    double lower;
    double upper;
    int lower_is_infinite;
    int upper_is_infinite;
    int can_tighten;
} PresolveScalarDomain;

typedef struct
{
    double finite_min;
    double finite_max;
    size_t n_infinite_min;
    size_t n_infinite_max;
    size_t n_nonzeros;
} PresolveLinearActivity;

typedef enum
{
    PRESOLVE_ROW_FEASIBLE = 0,
    PRESOLVE_ROW_INFEASIBLE = 1 << 0,
    PRESOLVE_ROW_LOWER_REDUNDANT = 1 << 1,
    PRESOLVE_ROW_UPPER_REDUNDANT = 1 << 2
} PresolveLinearRowState;

typedef struct
{
    const double *values;
    const int *columns;
    int length;
    double lower;
    double upper;
    int lower_is_infinite;
    int upper_is_infinite;
    double finite_min_activity;
    double finite_max_activity;
    size_t n_infinite_min;
    size_t n_infinite_max;
} PresolveLinearPropagationRow;

typedef enum
{
    PRESOLVE_KERNEL_UNCHANGED = 0,
    PRESOLVE_KERNEL_CHANGED = 1,
    PRESOLVE_KERNEL_STOP = 2
} PresolveKernelUpdate;

typedef PresolveKernelUpdate (*PresolveTightenScalarBound)(void *context, int column,
                                                           long double candidate,
                                                           int is_lower);
typedef void (*PresolveRefreshLinearActivity)(void *context,
                                              PresolveLinearPropagationRow *row);

typedef struct
{
    void *context;
    const void *lower_bounds;
    const void *upper_bounds;
    size_t bound_stride;
    const uint8_t *column_flags;
    uint8_t inactive_mask;
    const int *candidate_map;
    const int *row_excluded_columns;
    const uint8_t *row_exclusion_flags;
    const int *row_exclusion_sources;
    int row_index;
    double maximum_inferred_bound_magnitude;
    PresolveTightenScalarBound tighten_bound;
    PresolveRefreshLinearActivity refresh_activity;
} PresolveLinearPropagationOps;

static inline PresolveLinearRowState presolve_internal_classify_linear_row(
    const PresolveLinearActivity *activity, double lower, double upper,
    double feasibility_tolerance, double redundancy_tolerance);
static inline int presolve_internal_compute_linear_activity(
    const double *values, const int *columns, int length,
    const PresolveLinearPropagationOps *ops, int outward,
    PresolveLinearActivity *activity);

#include "LinearPropagationKernelImpl.h"

#endif
