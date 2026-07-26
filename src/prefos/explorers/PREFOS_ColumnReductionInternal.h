/*
 * Copyright 2026 Hongpei Li
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef PREFOS_COLUMN_REDUCTION_INTERNAL_H
#define PREFOS_COLUMN_REDUCTION_INTERNAL_H

#include "PREFOS_Internal.h"
#include "LinearPropagationKernel.h"

typedef struct
{
    double fixed_shift;
    long double finite_min;
    long double finite_max;
    long double max_box_range_contribution;
    size_t n_infinite_min;
    size_t n_infinite_max;
    size_t n_active_terms;
    int has_box_column;
    int unsupported;
} PreFOSSingletonRowActivity;

typedef struct PreFOSColumnWorkspace
{
    int *starts;
    int *ends;
    int *rows;
    double *values;
    unsigned char *quadratic;
    unsigned char *factor;
    unsigned char *protected_target;
    unsigned char *bounded_doubleton_chain_target;
    unsigned char *csc_column_dirty;
    unsigned char *dirty_row;
    int *column_dirty_row_counts;
    unsigned char *row_lock_state;
    unsigned char *trivial_row_queued;
    unsigned char *bound_dirty_row_queued;
    unsigned char *singleton_column_queued;
    unsigned char *dual_column_queued;
    double *singleton_activity_lower;
    double *singleton_activity_upper;
    int *row_degrees;
    int *live_degrees;
    int *down_locks;
    int *up_locks;
    int *trivial_candidate_rows;
    int *bound_dirty_rows;
    int *singleton_candidate_columns;
    int *dual_candidate_columns;
    double *column_max_abs_coefficient;
    PresolveLinearActivity *initial_row_activities;
    long double *initial_finite_min_accumulators;
    long double *initial_finite_max_accumulators;
    int *initial_unique_infinite_min_positions;
    int *initial_unique_infinite_max_positions;
    long double *initial_max_box_range_contribution;
    unsigned char *initial_activity_has_box_column;
    unsigned char *initial_activity_valid;
    unsigned char *initial_activity_singleton_reusable;
    int *gpu_degrees;
    unsigned char *gpu_down_locked;
    unsigned char *gpu_up_locked;
    int *gpu_singleton_candidates;
    double *objective;
    double objective_offset;
    int *singleton_targets;
    double *singleton_scales;
    int *singleton_deferred_columns;
    int *singleton_deferred_rows;
    int *singleton_row_activity_map;
    unsigned int *singleton_row_activity_epoch;
    unsigned int *singleton_processed_epoch;
    PreFOSSingletonRowActivity *singleton_row_activities;
    int *parallel_columns;
    int *parallel_support_hashes;
    int *parallel_coefficient_hashes;
    int *parallel_sort_auxiliary;
    int *parallel_group_starts;
    unsigned char *parallel_gpu_eligible;
    unsigned char *parallel_column_dirty;
    int *parallel_dirty_columns;
    int *substitution_active_rows;
    double *substitution_active_coefficients;
    int gpu_stats_valid;
    int gpu_csc_valid;
    int gpu_singleton_candidates_valid;
    size_t nnz;
    size_t max_row_nnz;
    size_t n_gpu_singleton_candidates;
    size_t n_trivial_candidate_rows;
    size_t n_bound_dirty_rows;
    size_t n_singleton_candidate_columns;
    size_t n_dual_candidate_columns;
    size_t n_parallel_dirty_columns;
    size_t singleton_candidate_position;
    size_t dual_candidate_position;
    size_t removed_row_cursor;
    size_t compacted_removed_row_cursor;
    size_t parallel_cache_removed_row_cursor;
    size_t parallel_cached_active_columns;
    size_t fixed_column_cursor;
    size_t singleton_fixed_column_cursor;
    size_t objective_fixed_column_cursor;
    size_t objective_change_epoch;
    size_t n_singleton_row_activities;
    size_t singleton_row_activity_capacity;
    size_t substitution_active_capacity;
    size_t transformation_event_cursor;
    size_t initial_activity_fixed_column_cursor;
    size_t initial_activity_event_cursor;
    size_t materialized_eager_substitutions;
    unsigned int singleton_epoch;
    int csc_has_insertions;
    int csc_has_zero_entries;
    int parallel_no_group_cache_valid;
    int fast_pass_started;
    int one_sided_singletons_seeded;
    int objective_synchronized;
} PreFOSColumnWorkspace;

PREFOS_INTERNAL void
prefos_internal_free_column_workspace(PreFOSColumnWorkspace *workspace);

PREFOS_INTERNAL PreFOSStatus prefos_internal_build_column_workspace(
    PreFOSPresolver *presolver, PreFOSColumnWorkspace *workspace);

PREFOS_INTERNAL PreFOSStatus prefos_internal_build_column_workspace_cpu(
    PreFOSPresolver *presolver, PreFOSColumnWorkspace *workspace);

PREFOS_INTERNAL PreFOSStatus
prefos_internal_build_structural_column_workspace_cpu(
    PreFOSPresolver *presolver, PreFOSColumnWorkspace *workspace);

PREFOS_INTERNAL PreFOSStatus
prefos_internal_build_column_workspace_cpu_with_counts(
    PreFOSPresolver *presolver, PreFOSColumnWorkspace *workspace,
    const int *column_counts, int cache_initial_activity);

PREFOS_INTERNAL void prefos_internal_refresh_column_workspace(
    const PreFOSPresolver *presolver, PreFOSColumnWorkspace *workspace);

PREFOS_INTERNAL void prefos_internal_update_column_live_degrees(
    const PreFOSPresolver *presolver, PreFOSColumnWorkspace *workspace);

PREFOS_INTERNAL void prefos_internal_queue_trivial_row(
    const PreFOSPresolver *presolver, PreFOSColumnWorkspace *workspace,
    int row);

PREFOS_INTERNAL void prefos_internal_queue_singleton_column(
    const PreFOSPresolver *presolver, PreFOSColumnWorkspace *workspace,
    int column);

PREFOS_INTERNAL void prefos_internal_queue_dual_column(
    const PreFOSPresolver *presolver, PreFOSColumnWorkspace *workspace,
    int column);

PREFOS_INTERNAL void prefos_internal_queue_row_side_change(
    PreFOSPresolver *presolver, PreFOSColumnWorkspace *workspace,
    size_t row);

PREFOS_INTERNAL void prefos_internal_notify_row_side_change(
    PreFOSPresolver *presolver, PreFOSColumnWorkspace *workspace,
    size_t row);

PREFOS_INTERNAL void prefos_internal_mark_workspace_row_dirty(
    const PreFOSPresolver *presolver, PreFOSColumnWorkspace *workspace,
    size_t row);

PREFOS_INTERNAL void
prefos_internal_begin_workspace_row_coefficient_update(
    const PreFOSPresolver *presolver, PreFOSColumnWorkspace *workspace,
    size_t row);

PREFOS_INTERNAL void prefos_internal_mark_workspace_row_exact(
    const PreFOSPresolver *presolver, PreFOSColumnWorkspace *workspace,
    size_t row);

PREFOS_INTERNAL PreFOSStatus
prefos_internal_update_workspace_row_coefficients(
    const PreFOSPresolver *presolver, PreFOSColumnWorkspace *workspace,
    size_t row, int first_column, double first_old,
    double first_new, int second_column, double second_old,
    double second_new);

PREFOS_INTERNAL void prefos_internal_invalidate_singleton_row_activity(
    PreFOSColumnWorkspace *workspace, size_t row);

PREFOS_INTERNAL void
prefos_internal_update_cached_singleton_column_bounds(
    const PreFOSPresolver *presolver, PreFOSColumnWorkspace *workspace,
    int column, double old_lower, double old_upper,
    double new_lower, double new_upper);

PREFOS_INTERNAL void
prefos_internal_update_cached_singleton_column_bound(
    const PreFOSPresolver *presolver, PreFOSColumnWorkspace *workspace,
    int column, double old_bound, double new_bound, int is_lower);

PREFOS_INTERNAL void
prefos_internal_remove_cached_singleton_column(
    const PreFOSPresolver *presolver, PreFOSColumnWorkspace *workspace,
    int column, double old_lower, double old_upper);

PREFOS_INTERNAL void prefos_internal_sync_singleton_activity_events(
    const PreFOSPresolver *presolver, PreFOSColumnWorkspace *workspace);

PREFOS_INTERNAL void prefos_internal_sync_initial_row_activity_events(
    const PreFOSPresolver *presolver, PreFOSColumnWorkspace *workspace);

PREFOS_INTERNAL void prefos_internal_queue_bound_changed_singletons(
    const PreFOSPresolver *presolver, PreFOSColumnWorkspace *workspace);

PREFOS_INTERNAL int prefos_internal_queue_transformation_events(
    const PreFOSPresolver *presolver, PreFOSColumnWorkspace *workspace);

PREFOS_INTERNAL PreFOSStatus prefos_internal_prepare_column_workspace(
    PreFOSPresolver *presolver, PreFOSColumnWorkspace *workspace);

PREFOS_INTERNAL PreFOSStatus prefos_internal_synchronize_column_workspace(
    PreFOSPresolver *presolver, PreFOSColumnWorkspace *workspace);

PREFOS_INTERNAL PreFOSStatus
prefos_internal_refresh_column_workspace_incremental(
    PreFOSPresolver *presolver, PreFOSColumnWorkspace *workspace);

PREFOS_INTERNAL void prefos_internal_mark_parallel_column_dirty(
    PreFOSColumnWorkspace *workspace, int column);

PREFOS_INTERNAL void prefos_internal_mark_parallel_column_bound_dirty(
    PreFOSColumnWorkspace *workspace, int column);

PREFOS_INTERNAL PreFOSStatus prefos_internal_rebuild_column_objective(
    const PreFOSPresolver *presolver, PreFOSColumnWorkspace *workspace);

PREFOS_INTERNAL PreFOSStatus prefos_internal_sync_column_objective(
    const PreFOSPresolver *presolver, PreFOSColumnWorkspace *workspace);

PREFOS_INTERNAL PreFOSStatus prefos_internal_populate_gpu_column_stats(
    PreFOSPresolver *presolver, PreFOSColumnWorkspace *workspace);

PREFOS_INTERNAL int prefos_internal_column_is_linear_box(
    const PreFOSPresolver *presolver, const PreFOSColumnWorkspace *workspace,
    int column);

PREFOS_INTERNAL double prefos_internal_column_max_abs_coefficient(
    const PreFOSPresolver *presolver,
    const PreFOSColumnWorkspace *workspace, int column);

PREFOS_INTERNAL PreFOSStatus prefos_internal_effective_row_bounds(
    const PreFOSPresolver *presolver, size_t row, double *lower, double *upper);

PREFOS_INTERNAL size_t prefos_internal_collect_live_row(
    const PreFOSPresolver *presolver, size_t row, int *columns,
    double *coefficients, size_t capacity);

PREFOS_INTERNAL PreFOSStatus prefos_internal_append_column_substitution(
    PreFOSPresolver *presolver, int column, const int *targets,
    const double *scales, size_t term_count, int source_row, double constant,
    double pivot, PreFOSColumnWorkspace *workspace,
    PreFOSSubstitutionMode mode, int eager_materialized,
    int source_is_only_active_row);

PREFOS_INTERNAL PreFOSStatus
prefos_internal_reduce_empty_and_dual_fixed_columns(
    PreFOSPresolver *presolver, PreFOSColumnWorkspace *workspace);

PREFOS_INTERNAL PreFOSStatus prefos_internal_reduce_singleton_columns(
    PreFOSPresolver *presolver, PreFOSColumnWorkspace *workspace,
    int allow_one_sided);

PREFOS_INTERNAL PreFOSStatus prefos_internal_reduce_bounded_doubletons(
    PreFOSPresolver *presolver, PreFOSColumnWorkspace *workspace);

PREFOS_INTERNAL PreFOSStatus
prefos_internal_reduce_parallel_column_groups(
    PreFOSPresolver *presolver, PreFOSColumnWorkspace *workspace);

PREFOS_INTERNAL PreFOSStatus
prefos_internal_reduce_transformed_parallel_columns(
    PreFOSPresolver *presolver, PreFOSColumnWorkspace *workspace,
    int *ran);

PREFOS_INTERNAL PreFOSStatus
prefos_internal_reduce_linear_columns_in_workspace(
    PreFOSPresolver *presolver, PreFOSColumnWorkspace *workspace,
    int allow_one_sided_singletons);

#endif
