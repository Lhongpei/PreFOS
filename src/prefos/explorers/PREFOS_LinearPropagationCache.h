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
    PreFOSColumnWorkspace *column_workspace;
    const double *lower_bounds;
    const double *upper_bounds;
    PreFOSRowActivity *activities;
    PreFOSRowActivity *redundancy_activities;
    long double *finite_min_accumulators;
    long double *finite_max_accumulators;
    int *box_column_pointers;
    int *adjacent_rows;
    int *adjacent_positions;
    unsigned char *redundancy_activity_stale;
    unsigned char *activity_recompute_required;
    unsigned char *finite_min_accumulator_stale;
    unsigned char *finite_max_accumulator_stale;
    int *unique_infinite_min_positions;
    int *unique_infinite_max_positions;
    long double *max_box_range_contribution;
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
    int lazy_activity_recompute;
    size_t total_work_limit;
    size_t work_budget_quantum;
    size_t round_changes_before;
    size_t round_work_before;
    size_t round_new_finite_bounds;
    size_t profile_residual_rescans;
    size_t profile_residual_rescan_terms;
    size_t profile_exact_zero_checks;
    size_t profile_exact_zero_terms;
    size_t profile_outward_activity_scans;
    size_t profile_outward_activity_terms;
    size_t profile_activity_recomputes;
    size_t profile_activity_recompute_terms;
    size_t profile_single_infinite_candidate_rows;
    size_t profile_bound_candidates;
    size_t profile_bound_changes;
    double profile_confirmation_milliseconds;
    double profile_kernel_milliseconds;
    int integrate_redundancy;
    int defer_activity_updates;
    int retain_deferred_frontier;
    int fallback_requested;
    int external_event_overflow;
    int current_row;
};

typedef PreFOSLinearPropagationCache PreFOSLinearPropagationState;

PREFOS_INTERNAL void prefos_internal_free_linear_propagation_state(
    PreFOSLinearPropagationState *state);

#endif
