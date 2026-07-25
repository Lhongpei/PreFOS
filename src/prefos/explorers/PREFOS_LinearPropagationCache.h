/*
 * Copyright 2026 Hongpei Li
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef PREFOS_LINEAR_PROPAGATION_CACHE_H
#define PREFOS_LINEAR_PROPAGATION_CACHE_H

#include "PREFOS_ColumnReductionInternal.h"
#include "DirtyRows.h"
#include "LinearPropagationKernel.h"

typedef PresolveLinearActivity PreFOSRowActivity;

typedef struct
{
    int column;
    double old_lower;
    double old_upper;
} PreFOSExternalBoundEvent;

struct PreFOSLinearPropagationCache
{
    const PreFOSColumnWorkspace *column_workspace;
    const double *lower_bounds;
    const double *upper_bounds;
    PreFOSRowActivity *activities;
    PreFOSRowActivity *redundancy_activities;
    int *box_column_pointers;
    int *adjacent_rows;
    int *adjacent_positions;
    unsigned char *redundancy_activity_stale;
    double *box_max_abs_coefficient;
    PreFOSExternalBoundEvent *external_bound_events;
    int *external_rows;
    unsigned char *external_bound_dirty;
    unsigned char *external_row_dirty;
    unsigned char *external_row_activity_dirty;
    size_t n_external_bound_events;
    size_t external_bound_capacity;
    size_t n_external_rows;
    size_t external_row_capacity;
#ifndef NDEBUG
    double *debug_cached_lower;
    double *debug_cached_upper;
    double *debug_cached_constraint_lower;
    double *debug_cached_constraint_upper;
    int *debug_cached_residual_source_column;
#endif
    PresolveDirtyRows dirty_rows;
    size_t activity_update_budget;
    size_t activity_updates_used;
    size_t total_work_limit;
    int integrate_redundancy;
    int fallback_requested;
    int external_event_overflow;
    int current_row;
};

typedef PreFOSLinearPropagationCache PreFOSLinearPropagationState;

PREFOS_INTERNAL void prefos_internal_free_linear_propagation_state(
    PreFOSLinearPropagationState *state);

#endif
