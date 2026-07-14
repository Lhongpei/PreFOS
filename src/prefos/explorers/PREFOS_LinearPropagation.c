/*
 * Copyright 2026 Hongpei Li
 * SPDX-License-Identifier: Apache-2.0
 */

#include "PREFOS_LinearPropagation.h"
#include "PREFOS_ConeActivity.h"
#include "PREFOS_CudaLinearPropagation.h"
#include "core/PREFOS_Timer.h"
#include "DirtyRows.h"
#include "LinearPropagationKernel.h"

#define PREFOS_MIN_LINEAR_PROPAGATION_WORK_BUDGET 1000000U

typedef PresolveLinearActivity PreFOSRowActivity;

typedef struct
{
    PreFOSRowActivity *activities;
    int *box_column_pointers;
    int *adjacent_rows;
    int *adjacent_positions;
    PresolveDirtyRows dirty_rows;
    size_t activity_update_budget;
    size_t activity_updates_used;
    size_t total_work_limit;
    int fallback_requested;
} PreFOSLinearPropagationState;

static size_t saturated_work_add(size_t left, size_t right)
{
    return right > SIZE_MAX - left ? SIZE_MAX : left + right;
}

static size_t linear_propagation_work_used(const PreFOSPresolver *presolver)
{
    size_t work = saturated_work_add(presolver->stats.linear_activity_nnz_computed,
                                     presolver->stats.linear_nnz_processed);
    return saturated_work_add(work, presolver->stats.linear_activity_updates);
}

static size_t linear_propagation_work_limit(const PreFOSPresolver *presolver)
{
    long double limit;
    if (presolver->settings.linear_propagation_max_work_ratio == 0.0)
        return SIZE_MAX;
    limit = (long double) presolver->original.A.nnz *
            (long double) presolver->settings.linear_propagation_max_work_ratio;
    if (limit < (long double) PREFOS_MIN_LINEAR_PROPAGATION_WORK_BUDGET)
        limit = (long double) PREFOS_MIN_LINEAR_PROPAGATION_WORK_BUDGET;
    return limit >= (long double) SIZE_MAX ? SIZE_MAX : (size_t) limit;
}

static int propagation_round_is_stale(const PreFOSPresolver *presolver, size_t changes,
                                      size_t work)
{
    long double changes_per_million;
    if (work == 0 || presolver->settings.linear_propagation_max_stale_rounds == 0 ||
        presolver->settings.linear_propagation_min_changes_per_million == 0.0)
        return 0;
    changes_per_million = (long double) changes * 1000000.0L / (long double) work;
    return changes_per_million <
           (long double)
               presolver->settings.linear_propagation_min_changes_per_million;
}

static void free_linear_propagation_state(PreFOSLinearPropagationState *state)
{
    if (!state) return;
    free(state->activities);
    free(state->box_column_pointers);
    free(state->adjacent_rows);
    free(state->adjacent_positions);
    presolve_dirty_rows_free(&state->dirty_rows);
    memset(state, 0, sizeof(*state));
}

static PreFOSStatus compute_row_activity_with_bounds(const PreFOSPresolver *presolver,
                                                  size_t row, int outward,
                                                  const double *lower_bounds,
                                                  const double *upper_bounds,
                                                  PreFOSRowActivity *activity)
{
    const PreFOSCsrMatrix *A = &presolver->original.A;
    PresolveLinearPropagationOps operations = {.lower_bounds = lower_bounds,
                                               .upper_bounds = upper_bounds,
                                               .bound_stride = sizeof(double),
                                               .candidate_map =
                                                   presolver->variable_to_box};
    int start = A->row_pointers[row];
    return presolve_internal_compute_linear_activity(
               A->values + start, A->column_indices + start,
               A->row_pointers[row + 1] - start, &operations, outward, activity)
               ? PREFOS_STATUS_OK
               : PREFOS_STATUS_NUMERICAL_ERROR;
}

static PreFOSStatus compute_row_activity(const PreFOSPresolver *presolver, size_t row,
                                      int outward, PreFOSRowActivity *activity)
{
    return compute_row_activity_with_bounds(presolver, row, outward,
                                            presolver->propagation_lower,
                                            presolver->propagation_upper, activity);
}

long double prefos_internal_propagation_margin(const PreFOSPresolver *presolver,
                                            long double reference)
{
    return (long double) presolver->settings.feasibility_tolerance *
           fmaxl(1.0L, fabsl(reference));
}

static PreFOSStatus check_row_activity(const PreFOSPresolver *presolver, size_t row,
                                    const PreFOSRowActivity *activity)
{
    PresolveLinearRowState state = presolve_internal_classify_linear_row(
        activity, presolver->working_constraint_lower[row],
        presolver->working_constraint_upper[row],
        presolver->settings.feasibility_tolerance, 0.0);
    return (state & PRESOLVE_ROW_INFEASIBLE) != 0 ? PREFOS_STATUS_PRIMAL_INFEASIBLE
                                                  : PREFOS_STATUS_OK;
}

static int row_is_active_for_linear_propagation(const PreFOSPresolver *presolver,
                                                size_t row)
{
    return !presolver->remove_rows[row] &&
           (isfinite(presolver->working_constraint_lower[row]) ||
            isfinite(presolver->working_constraint_upper[row]));
}

static int row_can_propagate(const PreFOSPresolver *presolver, size_t row,
                             const PreFOSRowActivity *activity)
{
    return (isfinite(presolver->working_constraint_upper[row]) &&
            activity->n_infinite_min <= 1) ||
           (isfinite(presolver->working_constraint_lower[row]) &&
            activity->n_infinite_max <= 1);
}

static PreFOSStatus
initialize_linear_propagation_state(PreFOSPresolver *presolver,
                                    PreFOSLinearPropagationState *state)
{
    const PreFOSProblemData *problem = &presolver->original;
    size_t row, box_position, active_nnz = 0;
    int *cursor = NULL;

    memset(state, 0, sizeof(*state));
    state->activities =
        (PreFOSRowActivity *) calloc(problem->A.rows, sizeof(PreFOSRowActivity));
    state->box_column_pointers = (int *) calloc(problem->n_box + 1, sizeof(int));
    if (!presolve_dirty_rows_init(&state->dirty_rows, problem->A.rows))
        return PREFOS_STATUS_OUT_OF_MEMORY;
    if ((problem->A.rows > 0 && !state->activities) || !state->box_column_pointers)
        return PREFOS_STATUS_OUT_OF_MEMORY;

    for (row = 0; row < problem->A.rows; ++row)
    {
        int p, has_box_column = 0;
        PreFOSStatus status;
        if (!row_is_active_for_linear_propagation(presolver, row)) continue;
        status = compute_row_activity(presolver, row, 0, &state->activities[row]);
        if (status != PREFOS_STATUS_OK) return status;
        status = check_row_activity(presolver, row, &state->activities[row]);
        if (status != PREFOS_STATUS_OK) return status;
        presolver->stats.linear_activity_nnz_computed +=
            state->activities[row].n_nonzeros;
        active_nnz += state->activities[row].n_nonzeros;
        for (p = problem->A.row_pointers[row]; p < problem->A.row_pointers[row + 1];
             ++p)
        {
            int column = problem->A.column_indices[p];
            int position = presolver->variable_to_box[column];
            if (problem->A.values[p] == 0.0 || position < 0) continue;
            if (state->box_column_pointers[position + 1] == INT_MAX)
                return PREFOS_STATUS_OUT_OF_MEMORY;
            ++state->box_column_pointers[position + 1];
            has_box_column = 1;
        }
        if (has_box_column &&
            row_can_propagate(presolver, row, &state->activities[row]))
        {
            if (!presolve_dirty_rows_schedule(&state->dirty_rows, (int) row))
                return PREFOS_STATUS_OUT_OF_MEMORY;
        }
    }

    for (box_position = 0; box_position < problem->n_box; ++box_position)
    {
        if (state->box_column_pointers[box_position] >
            INT_MAX - state->box_column_pointers[box_position + 1])
            return PREFOS_STATUS_OUT_OF_MEMORY;
        state->box_column_pointers[box_position + 1] +=
            state->box_column_pointers[box_position];
    }
    state->adjacent_rows = (int *) prefos_internal_alloc_array(
        (size_t) state->box_column_pointers[problem->n_box], sizeof(int));
    state->adjacent_positions = (int *) prefos_internal_alloc_array(
        (size_t) state->box_column_pointers[problem->n_box], sizeof(int));
    cursor = (int *) prefos_internal_alloc_array(problem->n_box, sizeof(int));
    if ((state->box_column_pointers[problem->n_box] > 0 &&
         (!state->adjacent_rows || !state->adjacent_positions)) ||
        (problem->n_box > 0 && !cursor))
    {
        free(cursor);
        return PREFOS_STATUS_OUT_OF_MEMORY;
    }
    if (problem->n_box > 0)
        memcpy(cursor, state->box_column_pointers, problem->n_box * sizeof(int));
    for (row = 0; row < problem->A.rows; ++row)
    {
        int p;
        if (!row_is_active_for_linear_propagation(presolver, row)) continue;
        for (p = problem->A.row_pointers[row]; p < problem->A.row_pointers[row + 1];
             ++p)
        {
            int column = problem->A.column_indices[p];
            int position = presolver->variable_to_box[column];
            int write;
            if (problem->A.values[p] == 0.0 || position < 0) continue;
            write = cursor[position]++;
            state->adjacent_rows[write] = (int) row;
            state->adjacent_positions[write] = p;
        }
    }
    free(cursor);
    {
        long double budget =
            (long double) active_nnz *
            (long double) presolver->settings.event_queue_activity_update_ratio;
        state->activity_update_budget =
            budget >= (long double) SIZE_MAX ? SIZE_MAX : (size_t) budget;
    }
    return PREFOS_STATUS_OK;
}

static PreFOSStatus update_cached_activity_term(long double *finite_sum,
                                             size_t *n_infinite, double coefficient,
                                             double old_bound, double new_bound)
{
    long double update;
    if (isinf(old_bound))
    {
        if (!isfinite(new_bound) || *n_infinite == 0)
            return PREFOS_STATUS_NUMERICAL_ERROR;
        --(*n_infinite);
        update = (long double) coefficient * (long double) new_bound;
    }
    else
    {
        if (!isfinite(new_bound)) return PREFOS_STATUS_NUMERICAL_ERROR;
        update = (long double) coefficient *
                 ((long double) new_bound - (long double) old_bound);
    }
    *finite_sum += update;
    return isfinite(*finite_sum) ? PREFOS_STATUS_OK : PREFOS_STATUS_NUMERICAL_ERROR;
}

static PreFOSStatus update_cached_activities_for_bound_change(
    PreFOSPresolver *presolver, PreFOSLinearPropagationState *state, int box_position,
    double old_bound, double new_bound, int is_lower)
{
    const PreFOSCsrMatrix *A = &presolver->original.A;
    size_t degree = (size_t) (state->box_column_pointers[box_position + 1] -
                              state->box_column_pointers[box_position]);
    int adjacency;
    if (state->activity_updates_used > state->activity_update_budget ||
        degree > state->activity_update_budget - state->activity_updates_used)
    {
        state->fallback_requested = 1;
        return PREFOS_STATUS_OK;
    }
    state->activity_updates_used += degree;
    if (state->total_work_limit != SIZE_MAX)
    {
        size_t used = linear_propagation_work_used(presolver);
        if (used > state->total_work_limit ||
            degree > state->total_work_limit - used)
        {
            state->fallback_requested = 1;
            return PREFOS_STATUS_OK;
        }
    }
    for (adjacency = state->box_column_pointers[box_position];
         adjacency < state->box_column_pointers[box_position + 1]; ++adjacency)
    {
        int row = state->adjacent_rows[adjacency];
        int position = state->adjacent_positions[adjacency];
        double coefficient = A->values[position];
        PreFOSRowActivity *activity = &state->activities[row];
        PreFOSStatus status;
        if ((is_lower && coefficient > 0.0) || (!is_lower && coefficient < 0.0))
            status = update_cached_activity_term(&activity->finite_min,
                                                 &activity->n_infinite_min,
                                                 coefficient, old_bound, new_bound);
        else
            status = update_cached_activity_term(&activity->finite_max,
                                                 &activity->n_infinite_max,
                                                 coefficient, old_bound, new_bound);
        if (status != PREFOS_STATUS_OK) return status;
        ++presolver->stats.linear_activity_updates;

        if (!presolve_dirty_rows_schedule(&state->dirty_rows, row))
            return PREFOS_STATUS_OUT_OF_MEMORY;
    }
    return PREFOS_STATUS_OK;
}

static double relaxed_implied_bound(const PreFOSPresolver *presolver,
                                    long double implied, int is_lower)
{
    long double relaxed;
    if (!isfinite(implied)) return implied > 0.0L ? INFINITY : -INFINITY;
    relaxed =
        implied + (is_lower ? -prefos_internal_propagation_margin(presolver, implied)
                            : prefos_internal_propagation_margin(presolver, implied));
    return prefos_internal_outward_bound_cast(relaxed, is_lower);
}

int prefos_internal_is_significant_improvement(const PreFOSPresolver *presolver,
                                            double current, double candidate,
                                            int is_lower)
{
    long double improvement, required;
    if (!isfinite(current)) return isfinite(candidate);
    improvement = is_lower ? (long double) candidate - (long double) current
                           : (long double) current - (long double) candidate;
    required =
        fmaxl((long double) presolver->settings.finite_bound_improvement_absolute,
              (long double) presolver->settings.finite_bound_improvement_relative *
                  fabsl((long double) current));
    return improvement > required;
}

static int propagated_bound_should_be_materialized(const PreFOSPresolver *presolver,
                                                   int box_position,
                                                   double candidate, int is_lower)
{
    double current, opposite;
    long double difference, scale;
    if (presolver->settings.propagated_bound_policy ==
        PREFOS_PROPAGATED_BOUND_POLICY_FIRST_ORDER)
        return 1;

    current = is_lower ? presolver->working_box_lower[box_position]
                       : presolver->working_box_upper[box_position];
    if (isfinite(current)) return 1;

    opposite = is_lower ? presolver->working_box_upper[box_position]
                        : presolver->working_box_lower[box_position];
    if (!isfinite(opposite)) return 0;
    difference = fabsl((long double) candidate - (long double) opposite);
    scale = fmaxl(
        1.0L, fmaxl(fabsl((long double) candidate), fabsl((long double) opposite)));
    return difference <=
           (long double) presolver->settings.fixed_variable_tolerance * scale;
}

static PreFOSStatus update_propagated_lower(PreFOSPresolver *presolver, int row,
                                         int column, double candidate, int *changed,
                                         PreFOSLinearPropagationState *state)
{
    int box_position = presolver->variable_to_box[column];
    double current, upper;
    long double upper_margin;
    PreFOSStatus status;
    if (box_position < 0 || candidate == -INFINITY) return PREFOS_STATUS_OK;
    if (candidate == INFINITY) return PREFOS_STATUS_PRIMAL_INFEASIBLE;

    current = presolver->propagation_lower[column];
    upper = presolver->propagation_upper[column];
    upper_margin = prefos_internal_propagation_margin(presolver, (long double) upper);
    if (isfinite(upper) &&
        (long double) candidate > (long double) upper + upper_margin)
        return PREFOS_STATUS_PRIMAL_INFEASIBLE;
    if (candidate <= current) return PREFOS_STATUS_OK;
    if (candidate < upper &&
        !prefos_internal_is_significant_improvement(presolver, current, candidate, 1))
        return PREFOS_STATUS_OK;

    if (candidate > upper) candidate = upper;
    if (propagated_bound_should_be_materialized(presolver, box_position, candidate,
                                                1))
    {
        double working_current = presolver->working_box_lower[box_position];
        status = prefos_internal_append_bound_record(presolver, row, column,
                                                  working_current, candidate, 1);
        if (status != PREFOS_STATUS_OK) return status;
        presolver->working_box_lower[box_position] = candidate;
        ++presolver->stats.materialized_propagated_box_bounds;
    }
    else
        ++presolver->stats.suppressed_propagated_box_bounds;
    presolver->propagation_lower[column] = candidate;
    if (state)
    {
        status = update_cached_activities_for_bound_change(
            presolver, state, box_position, current, candidate, 1);
        if (status != PREFOS_STATUS_OK) return status;
    }
    ++presolver->stats.propagated_box_bounds;
    *changed = 1;
    return PREFOS_STATUS_OK;
}

static PreFOSStatus update_propagated_upper(PreFOSPresolver *presolver, int row,
                                         int column, double candidate, int *changed,
                                         PreFOSLinearPropagationState *state)
{
    int box_position = presolver->variable_to_box[column];
    double lower, current;
    long double lower_margin;
    PreFOSStatus status;
    if (box_position < 0 || candidate == INFINITY) return PREFOS_STATUS_OK;
    if (candidate == -INFINITY) return PREFOS_STATUS_PRIMAL_INFEASIBLE;

    lower = presolver->propagation_lower[column];
    current = presolver->propagation_upper[column];
    lower_margin = prefos_internal_propagation_margin(presolver, (long double) lower);
    if (isfinite(lower) &&
        (long double) candidate < (long double) lower - lower_margin)
        return PREFOS_STATUS_PRIMAL_INFEASIBLE;
    if (candidate >= current) return PREFOS_STATUS_OK;
    if (candidate > lower &&
        !prefos_internal_is_significant_improvement(presolver, current, candidate, 0))
        return PREFOS_STATUS_OK;

    if (candidate < lower) candidate = lower;
    if (propagated_bound_should_be_materialized(presolver, box_position, candidate,
                                                0))
    {
        double working_current = presolver->working_box_upper[box_position];
        status = prefos_internal_append_bound_record(presolver, row, column,
                                                  working_current, candidate, 0);
        if (status != PREFOS_STATUS_OK) return status;
        presolver->working_box_upper[box_position] = candidate;
        ++presolver->stats.materialized_propagated_box_bounds;
    }
    else
        ++presolver->stats.suppressed_propagated_box_bounds;
    presolver->propagation_upper[column] = candidate;
    if (state)
    {
        status = update_cached_activities_for_bound_change(
            presolver, state, box_position, current, candidate, 0);
        if (status != PREFOS_STATUS_OK) return status;
    }
    ++presolver->stats.propagated_box_bounds;
    *changed = 1;
    return PREFOS_STATUS_OK;
}

typedef struct
{
    PreFOSPresolver *presolver;
    PreFOSLinearPropagationState *state;
    int row;
    int *changed;
    PreFOSStatus status;
} PreFOSLinearKernelContext;

static PresolveKernelUpdate tighten_prefos_scalar_bound(void *context_pointer,
                                                     int column,
                                                     long double candidate,
                                                     int is_lower)
{
    PreFOSLinearKernelContext *context = (PreFOSLinearKernelContext *) context_pointer;
    size_t changes_before = context->presolver->stats.propagated_box_bounds;
    double relaxed = relaxed_implied_bound(context->presolver, candidate, is_lower);

    context->status =
        is_lower
            ? update_propagated_lower(context->presolver, context->row, column,
                                      relaxed, context->changed, context->state)
            : update_propagated_upper(context->presolver, context->row, column,
                                      relaxed, context->changed, context->state);
    if (context->status != PREFOS_STATUS_OK ||
        (context->state && context->state->fallback_requested))
        return PRESOLVE_KERNEL_STOP;
    return context->presolver->stats.propagated_box_bounds > changes_before
               ? PRESOLVE_KERNEL_CHANGED
               : PRESOLVE_KERNEL_UNCHANGED;
}

static void refresh_prefos_linear_activity(void *context_pointer,
                                        PresolveLinearPropagationRow *row)
{
    PreFOSLinearKernelContext *context = (PreFOSLinearKernelContext *) context_pointer;
    const PreFOSRowActivity *activity;
    if (!context->state) return;
    activity = &context->state->activities[context->row];
    row->finite_min_activity = activity->finite_min;
    row->finite_max_activity = activity->finite_max;
    row->n_infinite_min = activity->n_infinite_min;
    row->n_infinite_max = activity->n_infinite_max;
}

static PreFOSStatus propagate_single_row(PreFOSPresolver *presolver, size_t row,
                                      PreFOSRowActivity *activity, int *changed,
                                      PreFOSLinearPropagationState *state)
{
    const PreFOSProblemData *problem = &presolver->original;
    const PreFOSCsrMatrix *A = &problem->A;
    PreFOSLinearKernelContext context;
    PresolveLinearPropagationRow kernel_row;
    PresolveLinearPropagationOps operations;
    PreFOSStatus status = check_row_activity(presolver, row, activity);
    int stopped;

    if (status != PREFOS_STATUS_OK) return status;
    if (!row_can_propagate(presolver, row, activity)) return PREFOS_STATUS_OK;

    context = (PreFOSLinearKernelContext){presolver, state, (int) row, changed,
                                       PREFOS_STATUS_OK};
    kernel_row = (PresolveLinearPropagationRow){
        A->values + A->row_pointers[row],
        A->column_indices + A->row_pointers[row],
        A->row_pointers[row + 1] - A->row_pointers[row],
        presolver->working_constraint_lower[row],
        presolver->working_constraint_upper[row],
        !isfinite(presolver->working_constraint_lower[row]),
        !isfinite(presolver->working_constraint_upper[row]),
        activity->finite_min,
        activity->finite_max,
        activity->n_infinite_min,
        activity->n_infinite_max};
    operations = (PresolveLinearPropagationOps){
        .context = &context,
        .lower_bounds = presolver->propagation_lower,
        .upper_bounds = presolver->propagation_upper,
        .bound_stride = sizeof(double),
        .candidate_map = presolver->variable_to_box,
        .maximum_inferred_bound_magnitude =
            PRESOLVE_DEFAULT_MAX_INFERRED_BOUND_MAGNITUDE,
        .tighten_bound = tighten_prefos_scalar_bound,
        .refresh_activity = refresh_prefos_linear_activity};
    (void) presolve_internal_propagate_linear_row(&kernel_row, &operations,
                                                  &stopped);
    if (stopped) return context.status;
    return PREFOS_STATUS_OK;
}

static size_t active_linear_nonzeros(const PreFOSPresolver *presolver)
{
    const PreFOSCsrMatrix *A = &presolver->original.A;
    size_t row, nonzeros = 0;
    for (row = 0; row < A->rows; ++row)
    {
        size_t row_nonzeros;
        if (!row_is_active_for_linear_propagation(presolver, row)) continue;
        row_nonzeros = (size_t) (A->row_pointers[row + 1] - A->row_pointers[row]);
        nonzeros = saturated_work_add(nonzeros, row_nonzeros);
    }
    return nonzeros;
}

static PreFOSStatus propagate_linear_bounds_full_scan(PreFOSPresolver *presolver,
                                                   int max_rounds, size_t work_limit)
{
    int round;
    int stale_rounds = 0;
    size_t active_nnz = active_linear_nonzeros(presolver);
    size_t estimated_round_work = saturated_work_add(active_nnz, active_nnz);
    for (round = 1; round <= max_rounds; ++round)
    {
        size_t row;
        size_t changes_before = presolver->stats.propagated_box_bounds;
        size_t work_before = linear_propagation_work_used(presolver);
        size_t used = work_before;
        int changed = 0;
        if (work_limit != SIZE_MAX &&
            (used > work_limit || estimated_round_work > work_limit - used))
        {
            ++presolver->stats.linear_budget_stops;
            break;
        }
        ++presolver->stats.linear_propagation_rounds;
        ++presolver->stats.linear_full_scan_rounds;
        for (row = 0; row < presolver->original.A.rows; ++row)
        {
            PreFOSRowActivity activity;
            PreFOSStatus status;
            if (!row_is_active_for_linear_propagation(presolver, row)) continue;
            status = compute_row_activity(presolver, row, 0, &activity);
            if (status != PREFOS_STATUS_OK) return status;
            presolver->stats.linear_activity_nnz_computed += activity.n_nonzeros;
            ++presolver->stats.linear_rows_processed;
            presolver->stats.linear_nnz_processed += activity.n_nonzeros;
            status = propagate_single_row(presolver, row, &activity, &changed, NULL);
            if (status != PREFOS_STATUS_OK) return status;
        }
        if (!changed) break;
        {
            size_t changes = presolver->stats.propagated_box_bounds - changes_before;
            size_t work = linear_propagation_work_used(presolver) - work_before;
            if (propagation_round_is_stale(presolver, changes, work))
                ++stale_rounds;
            else
                stale_rounds = 0;
            if (presolver->settings.linear_propagation_max_stale_rounds > 0 &&
                stale_rounds >=
                    presolver->settings.linear_propagation_max_stale_rounds)
            {
                ++presolver->stats.linear_stale_stops;
                break;
            }
        }
    }
    return PREFOS_STATUS_OK;
}

static PreFOSStatus propagate_linear_bounds_gpu(PreFOSPresolver *presolver, int max_rounds,
                                             size_t work_limit)
{
    const PreFOSProblemData *problem = &presolver->original;
    PreFOSCudaLinearPropagationContext *context = NULL;
    PreFOSCudaPropagationStatus cuda_status;
    double *lower_candidates = NULL, *upper_candidates = NULL;
    int *lower_sources = NULL, *upper_sources = NULL;
    size_t active_nnz = active_linear_nonzeros(presolver);
    size_t active_rows = 0;
    size_t estimated_round_work = saturated_work_add(active_nnz, active_nnz);
    int stale_rounds = 0;
    int completed_rounds = 0;
    int round;
    PreFOSTimestamp total_start, total_stop;

    prefos_internal_timer_now(&total_start);
    for (size_t row = 0; row < problem->A.rows; ++row)
        if (row_is_active_for_linear_propagation(presolver, row)) ++active_rows;

    cuda_status = prefos_cuda_linear_propagation_create(
        problem->A.rows, problem->n, problem->A.nnz, problem->A.row_pointers,
        problem->A.column_indices, problem->A.values,
        presolver->working_constraint_lower, presolver->working_constraint_upper,
        presolver->variable_to_box, presolver->remove_rows, &context,
        &presolver->stats.linear_gpu_setup_milliseconds,
        &presolver->stats.linear_gpu_long_rows);
    if (cuda_status != PREFOS_CUDA_PROPAGATION_OK)
    {
        ++presolver->stats.linear_gpu_fallbacks;
        return propagate_linear_bounds_full_scan(presolver, max_rounds, work_limit);
    }

    lower_candidates =
        (double *) prefos_internal_alloc_array(problem->n, sizeof(double));
    upper_candidates =
        (double *) prefos_internal_alloc_array(problem->n, sizeof(double));
    lower_sources = (int *) prefos_internal_alloc_array(problem->n, sizeof(int));
    upper_sources = (int *) prefos_internal_alloc_array(problem->n, sizeof(int));
    if (problem->n > 0 &&
        (!lower_candidates || !upper_candidates || !lower_sources || !upper_sources))
    {
        prefos_cuda_linear_propagation_free(context);
        free(lower_candidates);
        free(upper_candidates);
        free(lower_sources);
        free(upper_sources);
        return PREFOS_STATUS_OUT_OF_MEMORY;
    }

    for (round = 1; round <= max_rounds; ++round)
    {
        size_t column;
        size_t changes_before = presolver->stats.propagated_box_bounds;
        size_t work_before = linear_propagation_work_used(presolver);
        size_t used = work_before;
        double transfer_milliseconds = 0.0, kernel_milliseconds = 0.0;
        int suspected_infeasible_row = -1;
        int changed = 0;
        PreFOSStatus status = PREFOS_STATUS_OK;

        if (work_limit != SIZE_MAX &&
            (used > work_limit || estimated_round_work > work_limit - used))
        {
            ++presolver->stats.linear_budget_stops;
            break;
        }
        cuda_status = prefos_cuda_linear_propagation_round(
            context, presolver->propagation_lower, presolver->propagation_upper,
            presolver->settings.feasibility_tolerance,
            PRESOLVE_DEFAULT_MAX_INFERRED_BOUND_MAGNITUDE, lower_candidates,
            upper_candidates, lower_sources, upper_sources,
            &suspected_infeasible_row, &transfer_milliseconds, &kernel_milliseconds);
        presolver->stats.linear_gpu_transfer_milliseconds += transfer_milliseconds;
        presolver->stats.linear_gpu_kernel_milliseconds += kernel_milliseconds;
        if (cuda_status != PREFOS_CUDA_PROPAGATION_OK)
        {
            int remaining_rounds = max_rounds - completed_rounds;
            ++presolver->stats.linear_gpu_fallbacks;
            prefos_cuda_linear_propagation_free(context);
            free(lower_candidates);
            free(upper_candidates);
            free(lower_sources);
            free(upper_sources);
            return propagate_linear_bounds_full_scan(presolver, remaining_rounds,
                                                     work_limit);
        }

        ++completed_rounds;
        ++presolver->stats.linear_propagation_rounds;
        ++presolver->stats.linear_full_scan_rounds;
        ++presolver->stats.linear_gpu_rounds;
        presolver->stats.linear_activity_nnz_computed += active_nnz;
        presolver->stats.linear_rows_processed += active_rows;
        presolver->stats.linear_nnz_processed += active_nnz;

        if (suspected_infeasible_row >= 0)
        {
            PreFOSRowActivity activity;
            status = compute_row_activity(
                presolver, (size_t) suspected_infeasible_row, 0, &activity);
            if (status == PREFOS_STATUS_OK)
                status = check_row_activity(
                    presolver, (size_t) suspected_infeasible_row, &activity);
        }
        for (column = 0; status == PREFOS_STATUS_OK && column < problem->n; ++column)
        {
            if (lower_sources[column] >= 0)
                status = update_propagated_lower(
                    presolver, lower_sources[column], (int) column,
                    relaxed_implied_bound(presolver, lower_candidates[column], 1),
                    &changed, NULL);
            if (status == PREFOS_STATUS_OK && upper_sources[column] >= 0)
                status = update_propagated_upper(
                    presolver, upper_sources[column], (int) column,
                    relaxed_implied_bound(presolver, upper_candidates[column], 0),
                    &changed, NULL);
        }
        if (status != PREFOS_STATUS_OK)
        {
            prefos_cuda_linear_propagation_free(context);
            free(lower_candidates);
            free(upper_candidates);
            free(lower_sources);
            free(upper_sources);
            return status;
        }
        if (!changed) break;
        {
            size_t changes = presolver->stats.propagated_box_bounds - changes_before;
            size_t work = linear_propagation_work_used(presolver) - work_before;
            if (propagation_round_is_stale(presolver, changes, work))
                ++stale_rounds;
            else
                stale_rounds = 0;
            if (presolver->settings.linear_propagation_max_stale_rounds > 0 &&
                stale_rounds >=
                    presolver->settings.linear_propagation_max_stale_rounds)
            {
                ++presolver->stats.linear_stale_stops;
                break;
            }
        }
    }
    prefos_cuda_linear_propagation_free(context);
    free(lower_candidates);
    free(upper_candidates);
    free(lower_sources);
    free(upper_sources);
    prefos_internal_timer_now(&total_stop);
    presolver->stats.linear_gpu_total_milliseconds +=
        prefos_internal_timer_elapsed_milliseconds(&total_start, &total_stop);
    return PREFOS_STATUS_OK;
}

static int prefer_full_scan_linear_propagation(const PreFOSPresolver *presolver)
{
    const PreFOSProblemData *problem = &presolver->original;
    long double average_degree;
    if (problem->n_box == 0) return 0;
    average_degree = (long double) problem->A.nnz / (long double) problem->n_box;
    return average_degree >
           (long double) presolver->settings.event_queue_max_average_column_degree;
}

PreFOSStatus prefos_internal_propagate_linear_bounds(PreFOSPresolver *presolver)
{
    PreFOSLinearPropagationState state;
    PreFOSStatus status;
    int round;
    int stale_rounds = 0;
    size_t work_limit;
    if (!presolver->settings.linear_propagation ||
        presolver->settings.max_linear_propagation_rounds == 0 ||
        presolver->original.A.rows == 0)
        return PREFOS_STATUS_OK;

    work_limit = linear_propagation_work_limit(presolver);
    if (prefer_full_scan_linear_propagation(presolver))
    {
        if (presolver->settings.linear_propagation_gpu)
            return propagate_linear_bounds_gpu(
                presolver, presolver->settings.max_linear_propagation_rounds,
                work_limit);
        return propagate_linear_bounds_full_scan(
            presolver, presolver->settings.max_linear_propagation_rounds,
            work_limit);
    }

    if (work_limit != SIZE_MAX && presolver->original.A.nnz > work_limit)
    {
        ++presolver->stats.linear_budget_stops;
        return PREFOS_STATUS_OK;
    }

    status = initialize_linear_propagation_state(presolver, &state);
    if (status != PREFOS_STATUS_OK)
    {
        free_linear_propagation_state(&state);
        return status;
    }
    state.total_work_limit = work_limit;
    for (round = 1; round <= presolver->settings.max_linear_propagation_rounds &&
                    state.dirty_rows.current_count > 0;
         ++round)
    {
        int row;
        size_t changes_before = presolver->stats.propagated_box_bounds;
        size_t work_before = linear_propagation_work_used(presolver);
        ++presolver->stats.linear_propagation_rounds;
        ++presolver->stats.linear_event_rounds;
        while (presolve_dirty_rows_pop(&state.dirty_rows, &row))
        {
            int changed = 0;
            size_t row_work = state.activities[row].n_nonzeros;
            size_t used = linear_propagation_work_used(presolver);
            if (work_limit != SIZE_MAX &&
                (used > work_limit || row_work > work_limit - used))
            {
                ++presolver->stats.linear_budget_stops;
                free_linear_propagation_state(&state);
                return PREFOS_STATUS_OK;
            }
            ++presolver->stats.linear_rows_processed;
            presolver->stats.linear_nnz_processed +=
                state.activities[row].n_nonzeros;
            status = propagate_single_row(presolver, (size_t) row,
                                          &state.activities[row], &changed, &state);
            if (status != PREFOS_STATUS_OK)
            {
                free_linear_propagation_state(&state);
                return status;
            }
            if (state.fallback_requested)
            {
                int remaining_rounds =
                    presolver->settings.max_linear_propagation_rounds - round + 1;
                ++presolver->stats.linear_full_scan_fallbacks;
                free_linear_propagation_state(&state);
                return propagate_linear_bounds_full_scan(presolver, remaining_rounds,
                                                         work_limit);
            }
        }
        presolve_dirty_rows_finish_round(&state.dirty_rows);
        {
            size_t changes = presolver->stats.propagated_box_bounds - changes_before;
            size_t work = linear_propagation_work_used(presolver) - work_before;
            if (propagation_round_is_stale(presolver, changes, work))
                ++stale_rounds;
            else
                stale_rounds = 0;
            if (presolver->settings.linear_propagation_max_stale_rounds > 0 &&
                stale_rounds >=
                    presolver->settings.linear_propagation_max_stale_rounds)
            {
                ++presolver->stats.linear_stale_stops;
                break;
            }
        }
    }
    free_linear_propagation_state(&state);
    return PREFOS_STATUS_OK;
}

PreFOSStatus prefos_internal_remove_redundant_rows_by_activity(PreFOSPresolver *presolver)
{
    const PreFOSProblemData *problem = &presolver->original;
    PreFOSConeActivityWorkspace workspace;
    double *retained_lower = NULL, *retained_upper = NULL;
    const double *activity_lower = presolver->propagation_lower;
    const double *activity_upper = presolver->propagation_upper;
    PreFOSStatus result;
    size_t row, box_position;
    if (!presolver->settings.remove_redundant_rows) return PREFOS_STATUS_OK;
    memset(&workspace, 0, sizeof(workspace));
    result = PREFOS_STATUS_OK;
    if (presolver->settings.propagated_bound_policy ==
            PREFOS_PROPAGATED_BOUND_POLICY_INTERIOR_POINT &&
        presolver->stats.suppressed_propagated_box_bounds > 0 && problem->n > 0)
    {
        retained_lower =
            (double *) prefos_internal_alloc_array(problem->n, sizeof(double));
        retained_upper =
            (double *) prefos_internal_alloc_array(problem->n, sizeof(double));
        if (!retained_lower || !retained_upper)
        {
            free(retained_lower);
            free(retained_upper);
            return PREFOS_STATUS_OUT_OF_MEMORY;
        }
        memcpy(retained_lower, presolver->propagation_lower,
               problem->n * sizeof(double));
        memcpy(retained_upper, presolver->propagation_upper,
               problem->n * sizeof(double));
        for (box_position = 0; box_position < problem->n_box; ++box_position)
        {
            int column = problem->box_indices[box_position];
            retained_lower[column] = presolver->working_box_lower[box_position];
            retained_upper[column] = presolver->working_box_upper[box_position];
        }
        activity_lower = retained_lower;
        activity_upper = retained_upper;
    }
    if (problem->n_cones > 0 && presolver->settings.cone_aware_row_activity)
    {
        result = prefos_internal_cone_activity_workspace_init(presolver, &workspace);
        if (result != PREFOS_STATUS_OK)
        {
            free(retained_lower);
            free(retained_upper);
            return result;
        }
        workspace.lower_bounds = activity_lower;
        workspace.upper_bounds = activity_upper;
    }

    for (row = 0; row < problem->A.rows; ++row)
    {
        PreFOSRowActivity activity;
        int lower_implied, upper_implied;
        int cone_support_strengthened = 0;
        PresolveLinearRowState row_state;
        PreFOSStatus status;
        if (presolver->remove_rows[row]) continue;
        if (!isfinite(presolver->working_constraint_lower[row]) &&
            !isfinite(presolver->working_constraint_upper[row]))
        {
            int p;
            int has_nonzero = 0;
            for (p = problem->A.row_pointers[row];
                 p < problem->A.row_pointers[row + 1]; ++p)
            {
                if (problem->A.values[p] != 0.0)
                {
                    has_nonzero = 1;
                    break;
                }
            }
            if (has_nonzero)
            {
                presolver->remove_rows[row] = 1;
                ++presolver->stats.removed_redundant_rows;
            }
            continue;
        }
        status =
            problem->n_cones > 0 && presolver->settings.cone_aware_row_activity
                ? prefos_internal_compute_cone_aware_row_activity(presolver, row, 1,
                                                               &workspace, &activity)
                : compute_row_activity_with_bounds(presolver, row, 1, activity_lower,
                                                   activity_upper, &activity);
        if (problem->n_cones > 0 && presolver->settings.cone_aware_row_activity)
            cone_support_strengthened = workspace.row_support_strengthened;
        if (status != PREFOS_STATUS_OK)
        {
            if (cone_support_strengthened && status == PREFOS_STATUS_PRIMAL_INFEASIBLE)
                ++presolver->stats.cone_activity_infeasible_rows;
            result = status;
            break;
        }
        status = check_row_activity(presolver, row, &activity);
        if (status != PREFOS_STATUS_OK)
        {
            result = status;
            break;
        }
        if (activity.n_nonzeros == 0) continue;

        row_state = presolve_internal_classify_linear_row(
            &activity, presolver->working_constraint_lower[row],
            presolver->working_constraint_upper[row],
            presolver->settings.feasibility_tolerance, 0.0);
        lower_implied = (row_state & PRESOLVE_ROW_LOWER_REDUNDANT) != 0;
        upper_implied = (row_state & PRESOLVE_ROW_UPPER_REDUNDANT) != 0;
        if (lower_implied && upper_implied)
        {
            presolver->remove_rows[row] = 1;
            ++presolver->stats.removed_redundant_rows;
            if (cone_support_strengthened)
                ++presolver->stats.cone_activity_rows_removed;
        }
    }
    prefos_internal_cone_activity_workspace_free(&workspace);
    free(retained_lower);
    free(retained_upper);
    return result;
}
