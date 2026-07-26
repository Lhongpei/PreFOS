/*
 * Copyright 2026 Hongpei Li
 * SPDX-License-Identifier: Apache-2.0
 */

#include "PREFOS_LinearPropagation.h"
#include "PREFOS_ConeActivity.h"
#include "PREFOS_CudaBackend.h"
#include "PREFOS_CudaLinearPropagation.h"
#include "PREFOS_LinearPropagationCache.h"
#include "core/PREFOS_Timer.h"

#include <stdio.h>

#define PREFOS_MIN_LINEAR_PROPAGATION_WORK_BUDGET 1000000U
#define PREFOS_UNIQUE_INFINITE_POSITION_UNKNOWN (-2)
#define PREFOS_DEFERRED_ACTIVITY_INCREMENTAL_MIN_TERMS 16U
#define PREFOS_BROAD_FRONTIER_CACHE_MIN_NNZ 2097152U

static int trace_linear_conflicts(void)
{
    static int initialized = 0;
    static int enabled = 0;
    if (!initialized)
    {
        enabled = getenv("PREFOS_TRACE_LINEAR_CONFLICT") != NULL;
        initialized = 1;
    }
    return enabled;
}

static int trace_linear_cache(void)
{
    static int initialized = 0;
    static int enabled = 0;
    if (!initialized)
    {
        enabled = getenv("PREFOS_TRACE_LINEAR_CACHE") != NULL;
        initialized = 1;
    }
    return enabled;
}

static int trace_linear_profile(void)
{
    static int initialized = 0;
    static int enabled = 0;
    if (!initialized)
    {
        enabled = getenv("PREFOS_TRACE_LINEAR_PROFILE") != NULL;
        initialized = 1;
    }
    return enabled;
}

static int deferred_activity_incremental_enabled(void)
{
    static int initialized = 0;
    static int enabled = 1;
    if (!initialized)
    {
        enabled =
            getenv("PREFOS_DISABLE_DEFERRED_ACTIVITY_INCREMENTAL") == NULL;
        initialized = 1;
    }
    return enabled;
}

static void trace_linear_failure(
    const PreFOSPresolver *presolver,
    const PreFOSLinearPropagationState *state,
    const char *stage, int row, PreFOSStatus status)
{
    const PreFOSRowActivity *activity;
    if (!trace_linear_cache()) return;
    if (!state || row < 0 ||
        (size_t) row >= presolver->original.A.rows)
    {
        fprintf(
            stderr,
            "PreFOS linear failure stage=%s row=%d status=%d\n",
            stage, row, (int) status);
        return;
    }
    activity = &state->activities[row];
    fprintf(
        stderr,
        "PreFOS linear failure stage=%s row=%d status=%d "
        "bounds=[%.17g,%.17g] activity=[%.17g,%.17g] "
        "infinite=[%zu,%zu] nnz=%zu recompute=%d lazy=%d\n",
        stage, row, (int) status,
        presolver->working_constraint_lower[row],
        presolver->working_constraint_upper[row],
        activity->finite_min, activity->finite_max,
        activity->n_infinite_min, activity->n_infinite_max,
        activity->n_nonzeros,
        state->activity_recompute_required[row],
        state->lazy_activity_recompute);
}

static int trace_linear_column(int column)
{
    static int initialized = 0;
    static const char *selection = NULL;
    const char *cursor;
    if (!initialized)
    {
        selection = getenv("PREFOS_TRACE_LINEAR_COLUMN");
        initialized = 1;
    }
    cursor = selection;
    while (cursor && *cursor)
    {
        char *end;
        long value = strtol(cursor, &end, 10);
        if (end == cursor) break;
        if (value == column) return 1;
        cursor = *end == ',' ? end + 1 : end;
    }
    return 0;
}

static int traced_linear_row(void)
{
    static int initialized = 0;
    static int selected = -1;
    if (!initialized)
    {
        const char *value = getenv("PREFOS_TRACE_LINEAR_ROW");
        if (value && *value)
            selected = atoi(value);
        initialized = 1;
    }
    return selected;
}

static int trace_linear_updates(void)
{
    static int initialized = 0;
    static int enabled = 0;
    if (!initialized)
    {
        enabled = getenv("PREFOS_TRACE_LINEAR_UPDATES") != NULL;
        initialized = 1;
    }
    return enabled;
}

static int trace_linear_removals(void)
{
    static int initialized = 0;
    static int enabled = 0;
    if (!initialized)
    {
        enabled = getenv("PREFOS_TRACE_LINEAR_REMOVALS") != NULL;
        initialized = 1;
    }
    return enabled;
}

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
    size_t used;
    long double limit;
    if (presolver->settings.linear_propagation_max_work_ratio == 0.0)
        return SIZE_MAX;
    limit = (long double) presolver->original.A.nnz *
            (long double) presolver->settings.linear_propagation_max_work_ratio;
    if (limit < (long double) PREFOS_MIN_LINEAR_PROPAGATION_WORK_BUDGET)
        limit = (long double) PREFOS_MIN_LINEAR_PROPAGATION_WORK_BUDGET;
    if (limit >= (long double) SIZE_MAX) return SIZE_MAX;
    used = linear_propagation_work_used(presolver);
    return saturated_work_add(used, (size_t) limit);
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

static size_t renew_productive_round_work_budget(
    const PreFOSPresolver *presolver, size_t work_limit,
    size_t next_round_work, size_t changes, size_t work)
{
    size_t used;
    if (work_limit == SIZE_MAX || changes == 0 ||
        presolver->linear_propagation_hard_work_budget ||
        presolver->settings.linear_propagation_max_stale_rounds == 0 ||
        presolver->settings.linear_propagation_min_changes_per_million == 0.0 ||
        propagation_round_is_stale(presolver, changes, work))
        return work_limit;
    used = linear_propagation_work_used(presolver);
    if (used <= work_limit && next_round_work <= work_limit - used)
        return work_limit;
    return saturated_work_add(work_limit, next_round_work);
}

static int event_work_fits_or_renews(
    PreFOSPresolver *presolver, PreFOSLinearPropagationState *state,
    size_t additional_work)
{
    size_t used, changes, work, quantum;
    if (state->total_work_limit == SIZE_MAX) return 1;
    used = linear_propagation_work_used(presolver);
    if (used <= state->total_work_limit &&
        additional_work <= state->total_work_limit - used)
        return 1;
    changes =
        presolver->stats.propagated_box_bounds -
        state->round_changes_before;
    work = used - state->round_work_before;
    quantum = state->work_budget_quantum;
    if (quantum < additional_work) quantum = additional_work;
    state->total_work_limit = renew_productive_round_work_budget(
        presolver, state->total_work_limit, quantum, changes, work);
    used = linear_propagation_work_used(presolver);
    if (used <= state->total_work_limit &&
        additional_work <= state->total_work_limit - used)
        return 1;
    if (trace_linear_cache())
        fprintf(
            stderr,
            "PreFOS linear cache work limit: used=%zu additional=%zu "
            "limit=%zu changes=%zu round_work=%zu\n",
            used, additional_work, state->total_work_limit,
            changes, work);
    return 0;
}

static PreFOSStatus compute_row_activity_with_bounds(const PreFOSPresolver *presolver,
                                                  size_t row, int outward,
                                                  const double *lower_bounds,
                                                  const double *upper_bounds,
                                                  PreFOSRowActivity *activity)
{
    const PreFOSCsrMatrix *A = &presolver->original.A;
    int start = A->row_pointers[row];
    int length = A->row_pointers[row + 1] - start;
    if (!outward)
    {
        int position;
        memset(activity, 0, sizeof(*activity));
        for (position = start; position < start + length; ++position)
        {
            int column = A->column_indices[position];
            double coefficient = A->values[position];
            double minimum_bound, maximum_bound;
            double product;
            if (coefficient == 0.0 ||
                !prefos_internal_term_is_active_in_row(
                    presolver, row, column))
                continue;
            ++activity->n_nonzeros;
            minimum_bound = coefficient > 0.0
                                ? lower_bounds[column]
                                : upper_bounds[column];
            maximum_bound = coefficient > 0.0
                                ? upper_bounds[column]
                                : lower_bounds[column];
            if (isfinite(minimum_bound))
            {
                product = coefficient * minimum_bound;
                if (!isfinite(product) ||
                    !isfinite(activity->finite_min + product))
                    return PREFOS_STATUS_NUMERICAL_ERROR;
                activity->finite_min += product;
            }
            else
                ++activity->n_infinite_min;
            if (isfinite(maximum_bound))
            {
                product = coefficient * maximum_bound;
                if (!isfinite(product) ||
                    !isfinite(activity->finite_max + product))
                    return PREFOS_STATUS_NUMERICAL_ERROR;
                activity->finite_max += product;
            }
            else
                ++activity->n_infinite_max;
        }
        return PREFOS_STATUS_OK;
    }
    {
        PresolveLinearPropagationOps operations = {
            .lower_bounds = lower_bounds,
            .upper_bounds = upper_bounds,
            .bound_stride = sizeof(double),
            .candidate_map = presolver->variable_to_box,
            .row_excluded_columns =
                presolver->residual_source_column,
            .row_exclusion_flags =
                presolver->n_residual_row_substitutions > 0
                    ? presolver->substitution_keeps_source_row
                    : NULL,
            .row_exclusion_sources =
                presolver->substitution_source_row,
            .row_index = (int) row};
    return presolve_internal_compute_linear_activity(
               A->values + start, A->column_indices + start,
               length, &operations, outward, activity)
               ? PREFOS_STATUS_OK
               : PREFOS_STATUS_NUMERICAL_ERROR;
    }
}

static PreFOSStatus compute_row_activity(const PreFOSPresolver *presolver, size_t row,
                                      int outward, PreFOSRowActivity *activity)
{
    return compute_row_activity_with_bounds(presolver, row, outward,
                                            presolver->propagation_lower,
                                            presolver->propagation_upper, activity);
}

static int activity_needs_rigorous_confirmation(
    const PreFOSPresolver *presolver, size_t row,
    const PreFOSRowActivity *activity)
{
    double lower = presolver->working_constraint_lower[row];
    double upper = presolver->working_constraint_upper[row];
    PresolveLinearRowState row_state =
        presolve_internal_classify_linear_row(
            activity, lower, upper,
            presolver->settings.feasibility_tolerance,
            presolver->settings.feasibility_tolerance);
    return (row_state & PRESOLVE_ROW_INFEASIBLE) != 0 ||
           (isfinite(lower) &&
            (row_state & PRESOLVE_ROW_LOWER_REDUNDANT) != 0) ||
           (isfinite(upper) &&
            (row_state & PRESOLVE_ROW_UPPER_REDUNDANT) != 0);
}

static PreFOSStatus refresh_rigorous_activity_if_needed(
    const PreFOSPresolver *presolver, size_t row,
    const PreFOSRowActivity *activity, PreFOSRowActivity *outward,
    int *recomputed)
{
    *recomputed = 0;
    if (!activity_needs_rigorous_confirmation(
            presolver, row, activity))
    {
        *outward = *activity;
        return PREFOS_STATUS_OK;
    }
    *recomputed = 1;
    return compute_row_activity(presolver, row, 1, outward);
}

static PreFOSStatus compute_row_activities_with_bounds(
    const PreFOSPresolver *presolver, size_t row,
    const double *lower_bounds, const double *upper_bounds,
    PreFOSRowActivity *activity, PreFOSRowActivity *outward)
{
    PreFOSStatus status = compute_row_activity_with_bounds(
        presolver, row, 0, lower_bounds, upper_bounds, activity);
    if (status != PREFOS_STATUS_OK) return status;
    if (activity_needs_rigorous_confirmation(
            presolver, row, activity))
        return compute_row_activity_with_bounds(
            presolver, row, 1, lower_bounds, upper_bounds, outward);
    *outward = *activity;
    return PREFOS_STATUS_OK;
}

static PreFOSStatus compute_row_activities(
    const PreFOSPresolver *presolver, size_t row,
    PreFOSRowActivity *activity, PreFOSRowActivity *outward)
{
    return compute_row_activities_with_bounds(
        presolver, row, presolver->propagation_lower,
        presolver->propagation_upper, activity, outward);
}

static PreFOSStatus compute_initial_row_activity(
    const PreFOSPresolver *presolver, PreFOSLinearPropagationState *state,
    size_t row, int *has_box_column)
{
    const PreFOSCsrMatrix *A = &presolver->original.A;
    PreFOSRowActivity *activity = &state->activities[row];
    double finite_min = 0.0;
    double finite_max = 0.0;
    int unique_infinite_min_position = -1;
    int unique_infinite_max_position = -1;
    int start = A->row_pointers[row];
    int p;

    memset(activity, 0, sizeof(*activity));
    *has_box_column = 0;
    for (p = start; p < A->row_pointers[row + 1]; ++p)
    {
        int column = A->column_indices[p];
        int box_position = presolver->variable_to_box[column];
        double coefficient = A->values[p];
        double minimum_bound, maximum_bound;

        if (coefficient == 0.0 ||
            !prefos_internal_term_is_active_in_row(
                presolver, row, column))
            continue;
        ++activity->n_nonzeros;
        minimum_bound = coefficient > 0.0
                            ? presolver->propagation_lower[column]
                            : presolver->propagation_upper[column];
        maximum_bound = coefficient > 0.0
                            ? presolver->propagation_upper[column]
                            : presolver->propagation_lower[column];
        if (isfinite(minimum_bound))
        {
            double product = coefficient * minimum_bound;
            if (!isfinite(product) ||
                !isfinite(finite_min + product))
                return PREFOS_STATUS_NUMERICAL_ERROR;
            finite_min += product;
        }
        else
        {
            ++activity->n_infinite_min;
            unique_infinite_min_position =
                activity->n_infinite_min == 1 ? p - start : -1;
        }
        if (isfinite(maximum_bound))
        {
            double product = coefficient * maximum_bound;
            if (!isfinite(product) ||
                !isfinite(finite_max + product))
                return PREFOS_STATUS_NUMERICAL_ERROR;
            finite_max += product;
        }
        else
        {
            ++activity->n_infinite_max;
            unique_infinite_max_position =
                activity->n_infinite_max == 1 ? p - start : -1;
        }

        if (!state->column_workspace)
        {
            if (state->box_column_pointers[column + 1] == INT_MAX)
                return PREFOS_STATUS_OUT_OF_MEMORY;
            ++state->box_column_pointers[column + 1];
        }
        if (box_position < 0) continue;
        {
            double lower = presolver->propagation_lower[column];
            double upper = presolver->propagation_upper[column];
            long double contribution =
                isfinite(lower) && isfinite(upper)
                    ? (long double) (
                          fabs(coefficient) * (upper - lower))
                    : INFINITY;
            if (contribution >
                state->max_box_range_contribution[row])
                state->max_box_range_contribution[row] = contribution;
        }
        if (!state->column_workspace)
        {
            state->box_max_abs_coefficient[box_position] =
                fmax(state->box_max_abs_coefficient[box_position],
                     fabs(coefficient));
        }
        *has_box_column = 1;
    }
    state->finite_min_accumulators[row] = (long double) finite_min;
    state->finite_max_accumulators[row] = (long double) finite_max;
    activity->finite_min = finite_min;
    activity->finite_max = finite_max;
    state->unique_infinite_min_positions[row] =
        unique_infinite_min_position;
    state->unique_infinite_max_positions[row] =
        unique_infinite_max_position;
    return PREFOS_STATUS_OK;
}

static PreFOSStatus defer_initial_row_activity(
    const PreFOSPresolver *presolver,
    PreFOSLinearPropagationState *state, size_t row)
{
    const PreFOSCsrMatrix *A = &presolver->original.A;
    PreFOSRowActivity *activity = &state->activities[row];
    int p;

    memset(activity, 0, sizeof(*activity));
    state->unique_infinite_min_positions[row] =
        PREFOS_UNIQUE_INFINITE_POSITION_UNKNOWN;
    state->unique_infinite_max_positions[row] =
        PREFOS_UNIQUE_INFINITE_POSITION_UNKNOWN;
    for (p = A->row_pointers[row];
         p < A->row_pointers[row + 1]; ++p)
    {
        int column = A->column_indices[p];
        int box_position = presolver->variable_to_box[column];
        double coefficient = A->values[p];
        if (coefficient == 0.0 ||
            !prefos_internal_term_is_active_in_row(
                presolver, row, column))
            continue;
        ++activity->n_nonzeros;
        if (!state->column_workspace)
        {
            if (state->box_column_pointers[column + 1] == INT_MAX)
                return PREFOS_STATUS_OUT_OF_MEMORY;
            ++state->box_column_pointers[column + 1];
        }
        if (box_position >= 0)
        {
            double lower = presolver->propagation_lower[column];
            double upper = presolver->propagation_upper[column];
            long double contribution =
                isfinite(lower) && isfinite(upper)
                    ? (long double) (
                          fabs(coefficient) * (upper - lower))
                    : INFINITY;
            if (contribution >
                state->max_box_range_contribution[row])
                state->max_box_range_contribution[row] =
                    contribution;
            if (!state->column_workspace)
                state->box_max_abs_coefficient[box_position] =
                    fmax(
                        state->box_max_abs_coefficient[box_position],
                        fabs(coefficient));
        }
    }
    state->activity_recompute_required[row] = 1;
    state->redundancy_activity_stale[row] = 1;
    return PREFOS_STATUS_OK;
}

static PreFOSStatus reuse_singleton_row_activity(
    const PreFOSColumnWorkspace *workspace,
    PreFOSLinearPropagationState *state, size_t row,
    int *has_box_column, int *reused)
{
    const PreFOSSingletonRowActivity *source;
    PreFOSRowActivity *target;
    long double finite_min, finite_max;
    int index;

    *reused = 0;
    if (!workspace->singleton_row_activity_map ||
        !workspace->singleton_row_activity_epoch ||
        workspace->singleton_row_activity_epoch[row] == 0U)
        return PREFOS_STATUS_OK;
    index = workspace->singleton_row_activity_map[row];
    if (index < 0 ||
        (size_t) index >= workspace->n_singleton_row_activities)
        return PREFOS_STATUS_NUMERICAL_ERROR;
    source = &workspace->singleton_row_activities[index];
    if (source->unsupported) return PREFOS_STATUS_OK;

    finite_min =
        source->finite_min + (long double) source->fixed_shift;
    finite_max =
        source->finite_max + (long double) source->fixed_shift;
    if (!isfinite(finite_min) || !isfinite(finite_max) ||
        !isfinite((double) finite_min) ||
        !isfinite((double) finite_max))
        return PREFOS_STATUS_NUMERICAL_ERROR;
    target = &state->activities[row];
    target->finite_min = (double) finite_min;
    target->finite_max = (double) finite_max;
    target->n_infinite_min = source->n_infinite_min;
    target->n_infinite_max = source->n_infinite_max;
    target->n_nonzeros = source->n_active_terms;
    state->finite_min_accumulators[row] = finite_min;
    state->finite_max_accumulators[row] = finite_max;
    state->max_box_range_contribution[row] =
        source->max_box_range_contribution;
    *has_box_column = source->has_box_column;
    *reused = 1;
    return PREFOS_STATUS_OK;
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
    if ((state & PRESOLVE_ROW_INFEASIBLE) != 0 &&
        trace_linear_conflicts())
        fprintf(
            stderr,
            "PreFOS linear conflict row=%zu lower=%.17g upper=%.17g "
            "min=%.17g max=%.17g inf_min=%zu inf_max=%zu nnz=%zu\n",
            row, presolver->working_constraint_lower[row],
            presolver->working_constraint_upper[row],
            activity->finite_min, activity->finite_max,
            activity->n_infinite_min, activity->n_infinite_max,
            activity->n_nonzeros);
    return (state & PRESOLVE_ROW_INFEASIBLE) != 0 ? PREFOS_STATUS_PRIMAL_INFEASIBLE
                                                  : PREFOS_STATUS_OK;
}

PreFOSStatus prefos_internal_verify_linear_row_with_bounds(
    const PreFOSPresolver *presolver, size_t row,
    const double *lower_bounds, const double *upper_bounds)
{
    PreFOSRowActivity activity;
    PreFOSStatus status;
    if (!presolver || row >= presolver->original.A.rows || !lower_bounds ||
        !upper_bounds)
        return PREFOS_STATUS_INVALID_ARGUMENT;
    status = compute_row_activity_with_bounds(
        presolver, row, 1, lower_bounds, upper_bounds, &activity);
    return status == PREFOS_STATUS_OK
               ? check_row_activity(presolver, row, &activity)
               : status;
}

static int row_is_active_for_linear_propagation(const PreFOSPresolver *presolver,
                                                size_t row)
{
    return !presolver->remove_rows[row] &&
           prefos_internal_row_has_exact_linear_form(presolver, row) &&
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

static int activity_side_can_trigger(
    const PreFOSPresolver *presolver, size_t row,
    const PreFOSRowActivity *activity, int minimum_side)
{
    int lower_is_finite =
        isfinite(presolver->working_constraint_lower[row]);
    int upper_is_finite =
        isfinite(presolver->working_constraint_upper[row]);
    if (minimum_side)
    {
        if (upper_is_finite && activity->n_infinite_min <= 1)
            return 1;
        if (lower_is_finite && activity->n_infinite_min == 0)
            return 1;
        return !upper_is_finite &&
               activity->n_infinite_min == 1 &&
               activity->n_infinite_max == 0;
    }
    if (lower_is_finite && activity->n_infinite_max <= 1)
        return 1;
    if (upper_is_finite && activity->n_infinite_max == 0)
        return 1;
    return !lower_is_finite &&
           activity->n_infinite_max == 1 &&
           activity->n_infinite_min == 0;
}

static int row_has_potential_bound_improvement(
    const PreFOSPresolver *presolver,
    const PreFOSLinearPropagationState *state, size_t row,
    const PreFOSRowActivity *activity)
{
    long double maximum =
        state->max_box_range_contribution[row];
    if (isfinite(presolver->working_constraint_upper[row]))
    {
        if (activity->n_infinite_min == 1)
            return 1;
        if (activity->n_infinite_min == 0 &&
            (long double) presolver->working_constraint_upper[row] -
                    (long double) activity->finite_min <
                maximum)
            return 1;
    }
    if (isfinite(presolver->working_constraint_lower[row]))
    {
        if (activity->n_infinite_max == 1)
            return 1;
        if (activity->n_infinite_max == 0 &&
            (long double) activity->finite_max -
                    (long double) presolver->working_constraint_lower[row] <
                maximum)
            return 1;
    }
    return 0;
}

static void refresh_finite_row_max_box_range(
    const PreFOSPresolver *presolver, PreFOSLinearPropagationState *state,
    size_t row)
{
    const PreFOSCsrMatrix *matrix = &presolver->original.A;
    long double maximum = 0.0L;
    int position;

    for (position = matrix->row_pointers[row];
         position < matrix->row_pointers[row + 1]; ++position)
    {
        int column = matrix->column_indices[position];
        double coefficient = matrix->values[position];
        double lower, upper;
        long double contribution;
        if (coefficient == 0.0 ||
            presolver->variable_to_box[column] < 0 ||
            !prefos_internal_term_is_active_in_row(
                presolver, row, column))
            continue;
        lower = presolver->propagation_lower[column];
        upper = presolver->propagation_upper[column];
        if (!isfinite(lower) || !isfinite(upper))
        {
            state->max_box_range_contribution[row] = INFINITY;
            return;
        }
        contribution =
            fabsl((long double) coefficient) *
            ((long double) upper - (long double) lower);
        if (contribution > maximum) maximum = contribution;
    }
    state->max_box_range_contribution[row] = maximum;
}

static size_t prune_nonpropagating_round_rows(
    const PreFOSPresolver *presolver, PreFOSLinearPropagationState *state)
{
    PresolveDirtyRows *queue = &state->dirty_rows;
    size_t read, write = 0;

    for (read = 0; read < queue->current_count; ++read)
    {
        int row = queue->current[read];
        PreFOSRowActivity *activity = &state->activities[row];
        if (!state->activity_recompute_required[row] &&
            isinf(state->max_box_range_contribution[row]) &&
            activity->n_infinite_min == 0 &&
            activity->n_infinite_max == 0)
            refresh_finite_row_max_box_range(
                presolver, state, (size_t) row);
        int keep =
            row_is_active_for_linear_propagation(
                presolver, (size_t) row) &&
            (state->activity_recompute_required[row] ||
             row_has_potential_bound_improvement(
                 presolver, state, (size_t) row,
                 activity));
        if (keep)
            queue->current[write++] = row;
        else
            queue->states[row] = PRESOLVE_ROW_IDLE;
    }
    queue->current_count = write;
    queue->current_position = 0;
    return read - write;
}

static int scalar_redundancy_can_share_activity(
    const PreFOSPresolver *presolver)
{
    const PreFOSProblemData *problem = &presolver->original;
    int over_cpu_budget;
    if (!presolver->settings.remove_redundant_rows ||
        presolver->scalar_redundancy_completed ||
        presolver->settings.propagated_bound_policy !=
            PREFOS_PROPAGATED_BOUND_POLICY_FIRST_ORDER)
        return 0;
    over_cpu_budget =
        presolver->settings.redundant_row_max_average_nnz > 0.0 &&
        (long double) problem->A.nnz >
            (long double) problem->A.rows *
                (long double)
                    presolver->settings.redundant_row_max_average_nnz;
    return !over_cpu_budget || presolver->settings.structural_reductions_gpu;
}

static PreFOSStatus apply_scalar_row_classification(
    PreFOSPresolver *presolver, size_t row,
    const PreFOSRowActivity *activity, int *removed,
    PreFOSColumnWorkspace *column_workspace)
{
    PresolveLinearRowState row_state;
    int lower_implied, upper_implied;
    PreFOSStatus status;

    *removed = 0;
    if (presolver->nonmaterialized_bound_source_rows &&
        presolver->nonmaterialized_bound_source_rows[row])
        return PREFOS_STATUS_OK;
    status = check_row_activity(presolver, row, activity);
    if (status != PREFOS_STATUS_OK || activity->n_nonzeros == 0)
        return status;
    row_state = presolve_internal_classify_linear_row(
        activity, presolver->working_constraint_lower[row],
        presolver->working_constraint_upper[row],
        presolver->settings.feasibility_tolerance,
        presolver->settings.feasibility_tolerance);
    {
        if (traced_linear_row() == (int) row)
        {
            const PreFOSCsrMatrix *matrix = &presolver->original.A;
            int position;
            fprintf(
                stderr,
                "PreFOS linear-row row=%zu lower=%.17g upper=%.17g "
                "min=%.17g max=%.17g inf_min=%zu inf_max=%zu "
                "nnz=%zu state=%d\n",
                row, presolver->working_constraint_lower[row],
                presolver->working_constraint_upper[row],
                activity->finite_min, activity->finite_max,
                activity->n_infinite_min, activity->n_infinite_max,
                activity->n_nonzeros, (int) row_state);
            for (position = matrix->row_pointers[row];
                 position < matrix->row_pointers[row + 1]; ++position)
            {
                int column = matrix->column_indices[position];
                if (matrix->values[position] == 0.0 ||
                    !prefos_internal_term_is_active_in_row(
                        presolver, row, column))
                    continue;
                fprintf(
                    stderr,
                    "PreFOS linear-row term col=%d a=%.17g "
                    "lower=%.17g upper=%.17g fixed=%d\n",
                    column, matrix->values[position],
                    presolver->propagation_lower[column],
                    presolver->propagation_upper[column],
                    presolver->is_fixed[column]);
            }
        }
    }
    lower_implied = (row_state & PRESOLVE_ROW_LOWER_REDUNDANT) != 0;
    upper_implied = (row_state & PRESOLVE_ROW_UPPER_REDUNDANT) != 0;
    if (lower_implied && upper_implied)
    {
        if (trace_linear_removals())
            fprintf(
                stderr,
                "PreFOS linear redundant row=%zu nnz=%zu "
                "bounds=[%.17g,%.17g] activity=[%.17g,%.17g] "
                "infinite=[%zu,%zu]\n",
                row, activity->n_nonzeros,
                presolver->working_constraint_lower[row],
                presolver->working_constraint_upper[row],
                activity->finite_min, activity->finite_max,
                activity->n_infinite_min,
                activity->n_infinite_max);
        prefos_internal_mark_removed_row(presolver, row);
        ++presolver->stats.removed_redundant_rows;
        *removed = 1;
    }
    else if (lower_implied &&
             isfinite(presolver->working_constraint_lower[row]))
    {
        prefos_internal_note_constraint_bound_change(presolver);
        presolver->working_constraint_lower[row] = -INFINITY;
        prefos_internal_queue_row_side_change(
            presolver, column_workspace, row);
        ++presolver->stats.removed_redundant_row_lower_sides;
    }
    else if (upper_implied &&
             isfinite(presolver->working_constraint_upper[row]))
    {
        prefos_internal_note_constraint_bound_change(presolver);
        presolver->working_constraint_upper[row] = INFINITY;
        prefos_internal_queue_row_side_change(
            presolver, column_workspace, row);
        ++presolver->stats.removed_redundant_row_upper_sides;
    }
    return PREFOS_STATUS_OK;
}

static PreFOSStatus
initialize_linear_propagation_state(PreFOSPresolver *presolver,
                                    PreFOSLinearPropagationState *state,
                                    PreFOSColumnWorkspace *column_workspace)
{
    const PreFOSProblemData *problem = &presolver->original;
    PreFOSTimestamp profile_start, profile_rows_start;
    PreFOSTimestamp profile_rows_done, profile_stop;
    size_t row, box_position, active_nnz = 0;
    const unsigned char *seed_rows =
        presolver->linear_propagation_seed_rows;
    int reuse_initial_workspace_activities = 0;
    int *cursor = NULL;

    if (trace_linear_profile())
        prefos_internal_timer_now(&profile_start);
    memset(state, 0, sizeof(*state));
    state->current_row = -1;
    state->column_workspace = column_workspace;
    state->lower_bounds = presolver->propagation_lower;
    state->upper_bounds = presolver->propagation_upper;
    state->integrate_redundancy =
        scalar_redundancy_can_share_activity(presolver);
    if (column_workspace)
    {
        prefos_internal_sync_singleton_activity_events(
            presolver, column_workspace);
        prefos_internal_sync_initial_row_activity_events(
            presolver, column_workspace);
        reuse_initial_workspace_activities =
            column_workspace->initial_row_activities &&
            column_workspace->initial_finite_min_accumulators &&
            column_workspace->initial_finite_max_accumulators &&
            column_workspace
                ->initial_unique_infinite_min_positions &&
            column_workspace
                ->initial_unique_infinite_max_positions &&
            column_workspace->initial_max_box_range_contribution &&
            column_workspace->initial_activity_has_box_column &&
            column_workspace->initial_activity_valid;
    }
    if (reuse_initial_workspace_activities)
    {
        state->activities =
            column_workspace->initial_row_activities;
        state->finite_min_accumulators =
            column_workspace->initial_finite_min_accumulators;
        state->finite_max_accumulators =
            column_workspace->initial_finite_max_accumulators;
        state->unique_infinite_min_positions =
            column_workspace
                ->initial_unique_infinite_min_positions;
        state->unique_infinite_max_positions =
            column_workspace
                ->initial_unique_infinite_max_positions;
        state->max_box_range_contribution =
            column_workspace
                ->initial_max_box_range_contribution;
        column_workspace->initial_row_activities = NULL;
        column_workspace->initial_finite_min_accumulators = NULL;
        column_workspace->initial_finite_max_accumulators = NULL;
        column_workspace
            ->initial_unique_infinite_min_positions = NULL;
        column_workspace
            ->initial_unique_infinite_max_positions = NULL;
        column_workspace->initial_max_box_range_contribution = NULL;
    }
    else
    {
        state->activities =
            (PreFOSRowActivity *) calloc(
                problem->A.rows, sizeof(PreFOSRowActivity));
        state->finite_min_accumulators =
            (long double *) calloc(
                problem->A.rows, sizeof(long double));
        state->finite_max_accumulators =
            (long double *) calloc(
                problem->A.rows, sizeof(long double));
        state->unique_infinite_min_positions =
            (int *) prefos_internal_alloc_array(
                problem->A.rows, sizeof(int));
        state->unique_infinite_max_positions =
            (int *) prefos_internal_alloc_array(
                problem->A.rows, sizeof(int));
        state->max_box_range_contribution =
            (long double *) calloc(
                problem->A.rows, sizeof(long double));
    }
    if (state->integrate_redundancy)
        state->redundancy_activities = (PreFOSRowActivity *)
            calloc(problem->A.rows, sizeof(PreFOSRowActivity));
    state->box_column_pointers =
        (int *) calloc(problem->n + 1, sizeof(int));
    state->redundancy_activity_stale =
        (unsigned char *) calloc(problem->A.rows, sizeof(unsigned char));
    state->activity_recompute_required =
        (unsigned char *) calloc(problem->A.rows, sizeof(unsigned char));
    state->finite_min_accumulator_stale =
        (unsigned char *) calloc(problem->A.rows, sizeof(unsigned char));
    state->finite_max_accumulator_stale =
        (unsigned char *) calloc(problem->A.rows, sizeof(unsigned char));
    if (!column_workspace)
        state->box_max_abs_coefficient =
            (double *) calloc(problem->n_box, sizeof(double));
#ifndef NDEBUG
    state->debug_cached_lower =
        (double *) prefos_internal_alloc_array(problem->n, sizeof(double));
    state->debug_cached_upper =
        (double *) prefos_internal_alloc_array(problem->n, sizeof(double));
    state->debug_cached_constraint_lower =
        (double *) prefos_internal_alloc_array(
            problem->A.rows, sizeof(double));
    state->debug_cached_constraint_upper =
        (double *) prefos_internal_alloc_array(
            problem->A.rows, sizeof(double));
    state->debug_cached_residual_source_column =
        (int *) prefos_internal_alloc_array(
            problem->A.rows, sizeof(int));
#endif
    if (!presolve_dirty_rows_init(&state->dirty_rows, problem->A.rows))
        return PREFOS_STATUS_OUT_OF_MEMORY;
    if ((problem->A.rows > 0 &&
         (!state->activities ||
          !state->finite_min_accumulators ||
          !state->finite_max_accumulators ||
          (state->integrate_redundancy &&
           !state->redundancy_activities) ||
          !state->redundancy_activity_stale ||
          !state->activity_recompute_required ||
          !state->finite_min_accumulator_stale ||
          !state->finite_max_accumulator_stale ||
          !state->unique_infinite_min_positions ||
          !state->unique_infinite_max_positions ||
          !state->max_box_range_contribution)) ||
        !state->box_column_pointers ||
        (!column_workspace && problem->n_box > 0 &&
         !state->box_max_abs_coefficient))
        return PREFOS_STATUS_OUT_OF_MEMORY;
#ifndef NDEBUG
    if ((problem->A.rows > 0 &&
         (!state->debug_cached_constraint_lower ||
          !state->debug_cached_constraint_upper ||
          !state->debug_cached_residual_source_column)) ||
        (problem->n > 0 &&
         (!state->debug_cached_lower ||
          !state->debug_cached_upper)))
        return PREFOS_STATUS_OUT_OF_MEMORY;
#endif

    if (!reuse_initial_workspace_activities)
        for (row = 0; row < problem->A.rows; ++row)
        {
            state->unique_infinite_min_positions[row] =
                PREFOS_UNIQUE_INFINITE_POSITION_UNKNOWN;
            state->unique_infinite_max_positions[row] =
                PREFOS_UNIQUE_INFINITE_POSITION_UNKNOWN;
        }
    if (column_workspace)
    {
        for (box_position = 0;
             box_position < problem->n; ++box_position)
        {
            int degree =
                column_workspace->live_degrees[box_position];
            if (degree < 0)
                return PREFOS_STATUS_NUMERICAL_ERROR;
            state->box_column_pointers[box_position + 1] =
                degree;
        }
    }

    if (trace_linear_profile())
        prefos_internal_timer_now(&profile_rows_start);
    for (row = 0; row < problem->A.rows; ++row)
    {
        int has_box_column = 0;
        int reused_singleton_activity = 0;
        int removed = 0;
        int rigorous_recomputed = 0;
        int propagation_candidate = 0;
        int integrated_redundancy_candidate = 0;
        PreFOSStatus status;
        if (!presolver->remove_rows[row] &&
            !isfinite(presolver->working_constraint_lower[row]) &&
            !isfinite(presolver->working_constraint_upper[row]) &&
            state->integrate_redundancy)
        {
            prefos_internal_mark_removed_row(presolver, row);
            ++presolver->stats.removed_redundant_rows;
            continue;
        }
        if (!row_is_active_for_linear_propagation(presolver, row)) continue;
        if (seed_rows && !seed_rows[row])
        {
            status = defer_initial_row_activity(
                presolver, state, row);
            if (status != PREFOS_STATUS_OK) return status;
            active_nnz += state->activities[row].n_nonzeros;
            continue;
        }
        if (column_workspace)
            status = reuse_singleton_row_activity(
                column_workspace, state, row, &has_box_column,
                &reused_singleton_activity);
        else
            status = PREFOS_STATUS_OK;
        if (status == PREFOS_STATUS_OK &&
            !reused_singleton_activity)
        {
            if (reuse_initial_workspace_activities &&
                column_workspace->initial_activity_valid[row])
            {
                has_box_column =
                    column_workspace
                        ->initial_activity_has_box_column[row];
            }
            else
                status = compute_initial_row_activity(
                    presolver, state, row, &has_box_column);
        }
        else if (status == PREFOS_STATUS_OK)
        {
            state->unique_infinite_min_positions[row] =
                PREFOS_UNIQUE_INFINITE_POSITION_UNKNOWN;
            state->unique_infinite_max_positions[row] =
                PREFOS_UNIQUE_INFINITE_POSITION_UNKNOWN;
        }
        if (status != PREFOS_STATUS_OK)
        {
            trace_linear_failure(
                presolver, state, "initial-row-activity",
                (int) row, status);
            return status;
        }
        propagation_candidate =
            has_box_column &&
            row_has_potential_bound_improvement(
                presolver, state, row, &state->activities[row]);
        integrated_redundancy_candidate =
            state->integrate_redundancy;
        if (state->integrate_redundancy)
        {
            status = refresh_rigorous_activity_if_needed(
                presolver, row, &state->activities[row],
                &state->redundancy_activities[row],
                &rigorous_recomputed);
        }
        else
        {
            status = check_row_activity(
                presolver, row, &state->activities[row]);
            if (status == PREFOS_STATUS_PRIMAL_INFEASIBLE)
            {
                PreFOSRowActivity outward;
                status = compute_row_activity(
                    presolver, row, 1, &outward);
                if (status == PREFOS_STATUS_OK)
                    status = check_row_activity(
                        presolver, row, &outward);
            }
        }
        if (status == PREFOS_STATUS_OK && state->integrate_redundancy)
        {
            if (integrated_redundancy_candidate)
                status = apply_scalar_row_classification(
                    presolver, row,
                    &state->redundancy_activities[row],
                    &removed, column_workspace);
            else
                status = check_row_activity(
                    presolver, row,
                    &state->redundancy_activities[row]);
        }
        if (status != PREFOS_STATUS_OK)
        {
            trace_linear_failure(
                presolver, state, "initial-row-classification",
                (int) row, status);
            return status;
        }
        presolver->stats.linear_activity_nnz_computed +=
            state->activities[row].n_nonzeros;
        if (rigorous_recomputed)
            presolver->stats.linear_activity_nnz_computed +=
                state->redundancy_activities[row].n_nonzeros;
        if (removed)
        {
            int p;
            for (p = problem->A.row_pointers[row];
                 p < problem->A.row_pointers[row + 1]; ++p)
            {
                int column = problem->A.column_indices[p];
                if (problem->A.values[p] == 0.0 ||
                    !prefos_internal_term_is_active_in_row(
                        presolver, row, column))
                    continue;
                --state->box_column_pointers[column + 1];
            }
            continue;
        }
        active_nnz += state->activities[row].n_nonzeros;
        if (propagation_candidate)
        {
            if (!presolve_dirty_rows_schedule(&state->dirty_rows, (int) row))
                return PREFOS_STATUS_OUT_OF_MEMORY;
        }
    }

    if (trace_linear_profile())
        prefos_internal_timer_now(&profile_rows_done);
    if (reuse_initial_workspace_activities)
    {
        free(column_workspace->initial_activity_has_box_column);
        free(column_workspace->initial_activity_valid);
        free(column_workspace->initial_activity_singleton_reusable);
        column_workspace->initial_activity_has_box_column = NULL;
        column_workspace->initial_activity_valid = NULL;
        column_workspace->initial_activity_singleton_reusable = NULL;
    }

    for (box_position = 0; box_position < problem->n; ++box_position)
    {
        if (state->box_column_pointers[box_position] >
            INT_MAX - state->box_column_pointers[box_position + 1])
            return PREFOS_STATUS_OUT_OF_MEMORY;
        state->box_column_pointers[box_position + 1] +=
            state->box_column_pointers[box_position];
    }
    if (!column_workspace)
    {
        state->adjacent_rows = (int *) prefos_internal_alloc_array(
            (size_t) state->box_column_pointers[problem->n],
            sizeof(int));
        state->adjacent_positions = (int *) prefos_internal_alloc_array(
            (size_t) state->box_column_pointers[problem->n],
            sizeof(int));
        cursor = (int *) prefos_internal_alloc_array(
            problem->n, sizeof(int));
        if ((state->box_column_pointers[problem->n] > 0 &&
             (!state->adjacent_rows || !state->adjacent_positions)) ||
            (problem->n > 0 && !cursor))
        {
            free(cursor);
            return PREFOS_STATUS_OUT_OF_MEMORY;
        }
        if (problem->n > 0)
            memcpy(cursor, state->box_column_pointers,
                   problem->n * sizeof(int));
        for (row = 0; row < problem->A.rows; ++row)
        {
            int p;
            if (!row_is_active_for_linear_propagation(
                    presolver, row))
                continue;
            for (p = problem->A.row_pointers[row];
                 p < problem->A.row_pointers[row + 1]; ++p)
            {
                int column = problem->A.column_indices[p];
                int write;
                if (problem->A.values[p] == 0.0 ||
                    !prefos_internal_term_is_active_in_row(
                        presolver, row, column))
                    continue;
                write = cursor[column]++;
                state->adjacent_rows[write] = (int) row;
                state->adjacent_positions[write] = p;
            }
        }
        free(cursor);
    }
    {
        long double budget =
            (long double) active_nnz *
            (long double) presolver->settings.event_queue_activity_update_ratio;
        state->activity_update_budget =
            budget >= (long double) SIZE_MAX ? SIZE_MAX : (size_t) budget;
    }
    state->work_budget_quantum = active_nnz;
    if (seed_rows)
        state->lazy_activity_recompute = 1;
#ifndef NDEBUG
    if (problem->n > 0)
    {
        memcpy(state->debug_cached_lower,
               presolver->propagation_lower,
               problem->n * sizeof(double));
        memcpy(state->debug_cached_upper,
               presolver->propagation_upper,
               problem->n * sizeof(double));
    }
    if (problem->A.rows > 0)
    {
        memcpy(state->debug_cached_constraint_lower,
               presolver->working_constraint_lower,
               problem->A.rows * sizeof(double));
        memcpy(state->debug_cached_constraint_upper,
               presolver->working_constraint_upper,
               problem->A.rows * sizeof(double));
        memcpy(state->debug_cached_residual_source_column,
               presolver->residual_source_column,
               problem->A.rows * sizeof(int));
    }
#endif
    ++presolver->stats.linear_cache_builds;
    if (trace_linear_profile())
    {
        prefos_internal_timer_now(&profile_stop);
        fprintf(
            stderr,
            "PreFOS linear-profile init setup=%.6f rows=%.6f "
            "finalize=%.6f active_nnz=%zu dirty=%zu redundancy=%d\n",
            prefos_internal_timer_elapsed_milliseconds(
                &profile_start, &profile_rows_start),
            prefos_internal_timer_elapsed_milliseconds(
                &profile_rows_start, &profile_rows_done),
            prefos_internal_timer_elapsed_milliseconds(
                &profile_rows_done, &profile_stop),
            active_nnz, state->dirty_rows.current_count,
            state->integrate_redundancy);
    }
    return PREFOS_STATUS_OK;
}

static PreFOSStatus update_cached_activity_term(
    double *finite_sum, long double *accumulator,
    size_t *n_infinite, unsigned char *accumulator_stale,
    double coefficient, double old_bound, double new_bound,
    int *sum_updated, int *recompute_required)
{
    long double bound_delta;
    long double update;
    *sum_updated = 0;
    *recompute_required = 0;
    if (isinf(old_bound))
    {
        if (!isfinite(new_bound) || *n_infinite == 0)
            return PREFOS_STATUS_NUMERICAL_ERROR;
        --(*n_infinite);
        if (*accumulator_stale)
        {
            if (*n_infinite <= 1) *recompute_required = 1;
            return PREFOS_STATUS_OK;
        }
        if (new_bound == 0.0) return PREFOS_STATUS_OK;
        if (*n_infinite > 1)
        {
            *accumulator_stale = 1;
            return PREFOS_STATUS_OK;
        }
        bound_delta = (long double) new_bound;
    }
    else
    {
        if (!isfinite(new_bound)) return PREFOS_STATUS_NUMERICAL_ERROR;
        if (new_bound == old_bound) return PREFOS_STATUS_OK;
        if (*accumulator_stale)
        {
            if (*n_infinite <= 1) *recompute_required = 1;
            return PREFOS_STATUS_OK;
        }
        if (*n_infinite > 1)
        {
            *accumulator_stale = 1;
            return PREFOS_STATUS_OK;
        }
        bound_delta =
            (long double) new_bound - (long double) old_bound;
    }
    if (coefficient == 1.0)
        update = bound_delta;
    else if (coefficient == -1.0)
        update = -bound_delta;
    else
        update = (long double) coefficient * bound_delta;
    *accumulator += update;
    *finite_sum = (double) *accumulator;
    *sum_updated = 1;
    return isfinite(*accumulator) && isfinite(*finite_sum)
               ? PREFOS_STATUS_OK
               : PREFOS_STATUS_NUMERICAL_ERROR;
}

static int schedule_cached_row(PreFOSPresolver *presolver,
                               PreFOSLinearPropagationState *state,
                               int row);

static PreFOSStatus update_cached_activities_for_bound_change(
    PreFOSPresolver *presolver, PreFOSLinearPropagationState *state, int column,
    double old_bound, double new_bound, int is_lower)
{
    const PreFOSCsrMatrix *A = &presolver->original.A;
    int adjacency_start, adjacency_end;
    int active_start, active_end;
    int can_exclude_source_term =
        presolver->is_substituted[column] &&
        presolver->substitution_keeps_source_row[column];
    size_t degree;
    int adjacency;
    active_start = state->box_column_pointers[column];
    active_end = state->box_column_pointers[column + 1];
    degree = (size_t) (active_end - active_start);
    if (state->column_workspace)
    {
        adjacency_start = state->column_workspace->starts[column];
        adjacency_end = state->column_workspace->ends[column];
    }
    else
    {
        adjacency_start = active_start;
        adjacency_end = active_end;
    }
    if (!state->lazy_activity_recompute &&
        (state->activity_updates_used >
             state->activity_update_budget ||
         degree > state->activity_update_budget -
                      state->activity_updates_used))
    {
        if (state->activity_update_budget == 0)
        {
            state->fallback_requested = 1;
            return PREFOS_STATUS_OK;
        }
        state->lazy_activity_recompute = 1;
    }
    state->activity_updates_used =
        saturated_work_add(
            state->activity_updates_used, degree);
    if (!event_work_fits_or_renews(presolver, state, degree))
    {
        state->fallback_requested = 1;
        return PREFOS_STATUS_OK;
    }
    for (adjacency = adjacency_start;
         adjacency < adjacency_end; ++adjacency)
    {
        int row;
        double coefficient;
        if (state->column_workspace)
        {
            row = state->column_workspace->rows[adjacency];
            coefficient =
                state->column_workspace->values[adjacency];
            if (coefficient == 0.0 ||
                (can_exclude_source_term &&
                 !prefos_internal_term_is_active_in_row(
                     presolver, (size_t) row, column)))
                continue;
        }
        else
        {
            row = state->adjacent_rows[adjacency];
            coefficient =
                A->values[state->adjacent_positions[adjacency]];
        }
        PreFOSRowActivity *activity = &state->activities[row];
        size_t *updated_infinite_count = NULL;
        long double *updated_accumulator = NULL;
        unsigned char *updated_accumulator_stale = NULL;
        int *unique_infinite_position = NULL;
        size_t infinite_count_before;
        long double accumulator_before = 0.0L;
        PreFOSStatus status = PREFOS_STATUS_OK;
        int sum_updated = 0;
        int recompute_required = 0;
        int minimum_side;
        if (presolver->remove_rows[row]) continue;
        if (can_exclude_source_term &&
            presolver->substitution_source_row[column] == row)
            continue;
        if (state->lazy_activity_recompute)
        {
            state->activity_recompute_required[row] = 1;
            state->redundancy_activity_stale[row] = 1;
            ++presolver->stats.linear_activity_updates;
            if (!schedule_cached_row(presolver, state, row))
                return PREFOS_STATUS_OUT_OF_MEMORY;
            continue;
        }
        minimum_side =
            (is_lower && coefficient > 0.0) ||
            (!is_lower && coefficient < 0.0);
        if (minimum_side)
        {
            updated_infinite_count =
                &activity->n_infinite_min;
            updated_accumulator =
                &state->finite_min_accumulators[row];
            updated_accumulator_stale =
                &state->finite_min_accumulator_stale[row];
            unique_infinite_position =
                &state->unique_infinite_min_positions[row];
            infinite_count_before = *updated_infinite_count;
            accumulator_before = *updated_accumulator;
            status = update_cached_activity_term(
                &activity->finite_min,
                updated_accumulator,
                &activity->n_infinite_min,
                updated_accumulator_stale, coefficient,
                old_bound, new_bound, &sum_updated,
                &recompute_required);
        }
        else
        {
            updated_infinite_count =
                &activity->n_infinite_max;
            updated_accumulator =
                &state->finite_max_accumulators[row];
            updated_accumulator_stale =
                &state->finite_max_accumulator_stale[row];
            unique_infinite_position =
                &state->unique_infinite_max_positions[row];
            infinite_count_before = *updated_infinite_count;
            accumulator_before = *updated_accumulator;
            status = update_cached_activity_term(
                &activity->finite_max,
                updated_accumulator,
                &activity->n_infinite_max,
                updated_accumulator_stale, coefficient,
                old_bound, new_bound, &sum_updated,
                &recompute_required);
        }
        if (status != PREFOS_STATUS_OK)
        {
            trace_linear_failure(
                presolver, state, "cached-activity-term",
                row, status);
            return status;
        }
        if (recompute_required)
            state->activity_recompute_required[row] = 1;
        if (*updated_infinite_count != infinite_count_before)
            *unique_infinite_position =
                PREFOS_UNIQUE_INFINITE_POSITION_UNKNOWN;
        ++presolver->stats.linear_activity_updates;
        state->redundancy_activity_stale[row] = 1;
        if (sum_updated && isinf(old_bound) &&
            *updated_infinite_count == 0)
        {
            long double new_term =
                (long double) coefficient *
                (long double) new_bound;
            long double cancellation_scale =
                fabsl(accumulator_before) + fabsl(new_term);
            if (fabsl(*updated_accumulator) <=
                64.0L * (long double) DBL_EPSILON *
                    fmaxl(1.0L, cancellation_scale))
                state->activity_recompute_required[row] = 1;
        }
        if ((state->integrate_redundancy ||
             activity_side_can_trigger(
                 presolver, (size_t) row, activity,
                 minimum_side)) &&
            !schedule_cached_row(presolver, state, row))
            return PREFOS_STATUS_OUT_OF_MEMORY;
    }
#ifndef NDEBUG
    if (is_lower)
        state->debug_cached_lower[column] = new_bound;
    else
        state->debug_cached_upper[column] = new_bound;
#endif
    return PREFOS_STATUS_OK;
}

static int schedule_cached_row(
    PreFOSPresolver *presolver,
    PreFOSLinearPropagationState *state, int row);

static PreFOSStatus
update_deferred_activity_rows_for_bound_change(
    PreFOSPresolver *presolver,
    PreFOSLinearPropagationState *state, int column,
    double old_bound, double new_bound, int is_lower)
{
    const PreFOSCsrMatrix *matrix = &presolver->original.A;
    int start, end;
    int adjacency;
    int can_exclude_source_term =
        presolver->is_substituted[column] &&
        presolver->substitution_keeps_source_row[column];

    if (state->column_workspace)
    {
        start = state->column_workspace->starts[column];
        end = state->column_workspace->ends[column];
    }
    else
    {
        start = state->box_column_pointers[column];
        end = state->box_column_pointers[column + 1];
    }
    for (adjacency = start; adjacency < end; ++adjacency)
    {
        int row =
            state->column_workspace
                ? state->column_workspace->rows[adjacency]
                : state->adjacent_rows[adjacency];
        double coefficient =
            state->column_workspace
                ? state->column_workspace->values[adjacency]
                : matrix->values[state->adjacent_positions[adjacency]];
        PresolveDirtyRowState queue_state;
        if (presolver->remove_rows[row] ||
            (can_exclude_source_term &&
             presolver->substitution_source_row[column] == row))
            continue;
        if (coefficient == 0.0 ||
            (can_exclude_source_term &&
             !prefos_internal_term_is_active_in_row(
                 presolver, (size_t) row, column)))
            continue;
        queue_state =
            (PresolveDirtyRowState) state->dirty_rows.states[row];
        /*
         * Broad-frontier mode stops after the current wave, but the cache
         * survives into later structural passes.  A row already visited in
         * this wave therefore belongs to the next wave and can recompute its
         * activity lazily when that wave is eventually consumed.
         */
        if (queue_state == PRESOLVE_ROW_PROCESSED ||
            queue_state == PRESOLVE_ROW_NEXT)
        {
            if (!state->retain_deferred_frontier)
                continue;
            state->activity_recompute_required[row] = 1;
            state->redundancy_activity_stale[row] = 1;
            if (queue_state == PRESOLVE_ROW_PROCESSED &&
                !schedule_cached_row(presolver, state, row))
                return PREFOS_STATUS_OUT_OF_MEMORY;
            continue;
        }
        if (!state->activity_recompute_required[row] &&
            (!deferred_activity_incremental_enabled() ||
             state->activities[row].n_nonzeros <
                 PREFOS_DEFERRED_ACTIVITY_INCREMENTAL_MIN_TERMS))
            state->activity_recompute_required[row] = 1;
        else if (!state->activity_recompute_required[row])
        {
            PreFOSRowActivity *activity = &state->activities[row];
            double *finite_sum;
            long double *accumulator;
            size_t *n_infinite;
            unsigned char *accumulator_stale;
            int *unique_infinite_position;
            size_t infinite_count_before;
            long double accumulator_before;
            int minimum_side =
                (is_lower && coefficient > 0.0) ||
                (!is_lower && coefficient < 0.0);
            int sum_updated = 0;
            int recompute_required = 0;
            PreFOSStatus status;

            if (minimum_side)
            {
                finite_sum = &activity->finite_min;
                accumulator =
                    &state->finite_min_accumulators[row];
                n_infinite = &activity->n_infinite_min;
                accumulator_stale =
                    &state->finite_min_accumulator_stale[row];
                unique_infinite_position =
                    &state->unique_infinite_min_positions[row];
            }
            else
            {
                finite_sum = &activity->finite_max;
                accumulator =
                    &state->finite_max_accumulators[row];
                n_infinite = &activity->n_infinite_max;
                accumulator_stale =
                    &state->finite_max_accumulator_stale[row];
                unique_infinite_position =
                    &state->unique_infinite_max_positions[row];
            }
            infinite_count_before = *n_infinite;
            accumulator_before = *accumulator;
            status = update_cached_activity_term(
                finite_sum, accumulator, n_infinite,
                accumulator_stale, coefficient, old_bound,
                new_bound, &sum_updated, &recompute_required);
            if (status != PREFOS_STATUS_OK)
                return status;
            if (recompute_required)
                state->activity_recompute_required[row] = 1;
            if (*n_infinite != infinite_count_before)
                *unique_infinite_position =
                    PREFOS_UNIQUE_INFINITE_POSITION_UNKNOWN;
            if (sum_updated && isinf(old_bound) &&
                *n_infinite == 0)
            {
                long double new_term =
                    (long double) coefficient *
                    (long double) new_bound;
                long double cancellation_scale =
                    fabsl(accumulator_before) + fabsl(new_term);
                if (fabsl(*accumulator) <=
                    64.0L * (long double) DBL_EPSILON *
                        fmaxl(1.0L, cancellation_scale))
                    state->activity_recompute_required[row] = 1;
            }
            ++presolver->stats.linear_activity_updates;
        }
        state->redundancy_activity_stale[row] = 1;
        if (queue_state == PRESOLVE_ROW_IDLE &&
            !schedule_cached_row(presolver, state, row))
            return PREFOS_STATUS_OUT_OF_MEMORY;
    }
#ifndef NDEBUG
    if (is_lower)
        state->debug_cached_lower[column] = new_bound;
    else
        state->debug_cached_upper[column] = new_bound;
#endif
    return PREFOS_STATUS_OK;
}

static int linear_propagation_cache_matches(
    const PreFOSPresolver *presolver,
    const PreFOSLinearPropagationState *state,
    const PreFOSColumnWorkspace *column_workspace)
{
    return state->column_workspace == column_workspace &&
           state->lower_bounds == presolver->propagation_lower &&
           state->upper_bounds == presolver->propagation_upper;
}

static int schedule_cached_row(PreFOSPresolver *presolver,
                               PreFOSLinearPropagationState *state,
                               int row)
{
    int was_idle;
    if (row < 0 || (size_t) row >= state->dirty_rows.capacity)
        return 0;
    was_idle =
        state->dirty_rows.states[row] == PRESOLVE_ROW_IDLE;
    if (!presolve_dirty_rows_schedule(&state->dirty_rows, row))
        return 0;
    if (was_idle)
        ++presolver->stats.linear_cache_rows_scheduled;
    return 1;
}

#ifndef NDEBUG
static int linear_cache_notifications_are_complete(
    const PreFOSPresolver *presolver,
    const PreFOSLinearPropagationState *state)
{
    size_t column, row;
    for (column = 0; column < presolver->original.n; ++column)
    {
        if (state->debug_cached_lower[column] !=
                presolver->propagation_lower[column] ||
            state->debug_cached_upper[column] !=
                presolver->propagation_upper[column])
        {
            if (!state->external_bound_dirty ||
                !state->external_bound_dirty[column])
            {
                if (trace_linear_cache())
                    fprintf(
                        stderr,
                        "PreFOS linear cache missed bound notification "
                        "column=%zu cached=[%.17g,%.17g] "
                        "current=[%.17g,%.17g]\n",
                        column, state->debug_cached_lower[column],
                        state->debug_cached_upper[column],
                        presolver->propagation_lower[column],
                        presolver->propagation_upper[column]);
                return 0;
            }
        }
    }
    for (row = 0; row < presolver->original.A.rows; ++row)
    {
        if (state->debug_cached_constraint_lower[row] !=
                presolver->working_constraint_lower[row] ||
            state->debug_cached_constraint_upper[row] !=
                presolver->working_constraint_upper[row] ||
            state->debug_cached_residual_source_column[row] !=
                presolver->residual_source_column[row])
        {
            if (!state->external_row_dirty ||
                !state->external_row_dirty[row])
            {
                if (trace_linear_cache())
                    fprintf(
                        stderr,
                        "PreFOS linear cache missed row notification "
                        "row=%zu bounds=[%.17g,%.17g]->[%.17g,%.17g] "
                        "residual=%d->%d\n",
                        row,
                        state->debug_cached_constraint_lower[row],
                        state->debug_cached_constraint_upper[row],
                        presolver->working_constraint_lower[row],
                        presolver->working_constraint_upper[row],
                        state->debug_cached_residual_source_column[row],
                        presolver->residual_source_column[row]);
                return 0;
            }
        }
    }
    return 1;
}
#endif

static PreFOSStatus synchronize_linear_propagation_cache(
    PreFOSPresolver *presolver, PreFOSLinearPropagationState *state)
{
    size_t event;

    state->activity_updates_used = 0;
    state->lazy_activity_recompute = 0;
    state->fallback_requested = 0;
    state->current_row = -1;
    if (state->external_event_overflow)
    {
        if (trace_linear_cache())
            fprintf(stderr, "PreFOS linear cache rebuild: event overflow\n");
        state->fallback_requested = 1;
        return PREFOS_STATUS_OK;
    }
#ifndef NDEBUG
    if (!linear_cache_notifications_are_complete(
            presolver, state))
        return PREFOS_STATUS_NUMERICAL_ERROR;
#endif
    state->integrate_redundancy =
        scalar_redundancy_can_share_activity(presolver);
    if (state->integrate_redundancy &&
        !state->redundancy_activities)
    {
        if (trace_linear_cache())
            fprintf(
                stderr,
                "PreFOS linear cache rebuild: redundancy state missing\n");
        state->fallback_requested = 1;
        return PREFOS_STATUS_OK;
    }

    for (event = 0; event < state->n_external_bound_events; ++event)
    {
        const PreFOSExternalBoundEvent *bound_event =
            &state->external_bound_events[event];
        int column = bound_event->column;
        double old_lower = bound_event->old_lower;
        double old_upper = bound_event->old_upper;
        double new_lower = presolver->propagation_lower[column];
        double new_upper = presolver->propagation_upper[column];
        PreFOSStatus status;

        state->external_bound_dirty[column] = 0;
        if (new_lower < old_lower || new_upper > old_upper)
        {
            if (trace_linear_cache())
                fprintf(
                    stderr,
                    "PreFOS linear cache rebuild: relaxed bound column=%d "
                    "old=[%.17g,%.17g] new=[%.17g,%.17g]\n",
                    column, old_lower, old_upper, new_lower, new_upper);
            state->fallback_requested = 1;
            return PREFOS_STATUS_OK;
        }
        if (new_lower != old_lower)
        {
            ++presolver->stats.linear_cache_bound_changes;
            status = update_cached_activities_for_bound_change(
                presolver, state, (int) column, old_lower, new_lower, 1);
            if (status != PREFOS_STATUS_OK || state->fallback_requested)
            {
                if (status != PREFOS_STATUS_OK)
                    trace_linear_failure(
                        presolver, state,
                        "cache-lower-update", -1, status);
                if (state->fallback_requested && trace_linear_cache())
                    fprintf(
                        stderr,
                        "PreFOS linear cache rebuild: lower update budget "
                        "column=%d used=%zu limit=%zu degree=%zu\n",
                        column, state->activity_updates_used,
                        state->activity_update_budget,
                        (size_t) (
                            state->box_column_pointers[column + 1] -
                            state->box_column_pointers[column]));
                return status;
            }
        }
        if (new_upper != old_upper)
        {
            ++presolver->stats.linear_cache_bound_changes;
            status = update_cached_activities_for_bound_change(
                presolver, state, (int) column, old_upper, new_upper, 0);
            if (status != PREFOS_STATUS_OK || state->fallback_requested)
            {
                if (status != PREFOS_STATUS_OK)
                    trace_linear_failure(
                        presolver, state,
                        "cache-upper-update", -1, status);
                if (state->fallback_requested && trace_linear_cache())
                    fprintf(
                        stderr,
                        "PreFOS linear cache rebuild: upper update budget "
                        "column=%d used=%zu limit=%zu degree=%zu\n",
                        column, state->activity_updates_used,
                        state->activity_update_budget,
                        (size_t) (
                            state->box_column_pointers[column + 1] -
                            state->box_column_pointers[column]));
                return status;
            }
        }
    }
    state->n_external_bound_events = 0;
    for (event = 0; event < state->n_external_rows; ++event)
    {
        size_t row = (size_t) state->external_rows[event];
        int activity_changed =
            state->external_row_activity_dirty[row] != 0;

        state->external_row_dirty[row] = 0;
        state->external_row_activity_dirty[row] = 0;
        if (activity_changed && !presolver->remove_rows[row])
        {
            int recomputed;
            PreFOSStatus status;
            status = compute_row_activity(
                presolver, row, 0, &state->activities[row]);
            if (status != PREFOS_STATUS_OK) return status;
            state->finite_min_accumulators[row] =
                (long double) state->activities[row].finite_min;
            state->finite_max_accumulators[row] =
                (long double) state->activities[row].finite_max;
            state->finite_min_accumulator_stale[row] = 0;
            state->finite_max_accumulator_stale[row] = 0;
            state->max_box_range_contribution[row] = INFINITY;
            recomputed = 0;
            if (state->redundancy_activities)
            {
                status = refresh_rigorous_activity_if_needed(
                    presolver, row, &state->activities[row],
                    &state->redundancy_activities[row],
                    &recomputed);
                if (status != PREFOS_STATUS_OK) return status;
            }
            presolver->stats.linear_activity_nnz_computed +=
                state->activities[row].n_nonzeros;
            if (recomputed)
                presolver->stats.linear_activity_nnz_computed +=
                    state->activities[row].n_nonzeros;
            state->redundancy_activity_stale[row] = 0;
            state->activity_recompute_required[row] = 0;
            state->unique_infinite_min_positions[row] =
                PREFOS_UNIQUE_INFINITE_POSITION_UNKNOWN;
            state->unique_infinite_max_positions[row] =
                PREFOS_UNIQUE_INFINITE_POSITION_UNKNOWN;
        }
        else if (!activity_changed)
            state->redundancy_activity_stale[row] = 1;
        if (!presolver->remove_rows[row] &&
            !schedule_cached_row(presolver, state, (int) row))
            return PREFOS_STATUS_OUT_OF_MEMORY;
#ifndef NDEBUG
        state->debug_cached_constraint_lower[row] =
            presolver->working_constraint_lower[row];
        state->debug_cached_constraint_upper[row] =
            presolver->working_constraint_upper[row];
        state->debug_cached_residual_source_column[row] =
            presolver->residual_source_column[row];
#endif
    }
    state->n_external_rows = 0;
#ifndef NDEBUG
    if (!linear_cache_notifications_are_complete(
            presolver, state))
        return PREFOS_STATUS_NUMERICAL_ERROR;
#endif
    return PREFOS_STATUS_OK;
}

static double relaxed_implied_bound(const PreFOSPresolver *presolver,
                                    long double implied, int is_lower)
{
    (void) presolver;
    if (!isfinite(implied)) return implied > 0.0L ? INFINITY : -INFINITY;
    return prefos_internal_outward_bound_cast(implied, is_lower);
}

int prefos_internal_is_significant_improvement(const PreFOSPresolver *presolver,
                                            double current, double candidate,
                                            int is_lower)
{
    double improvement, required;
    if (!isfinite(current)) return isfinite(candidate);
    improvement = is_lower ? candidate - current : current - candidate;
    required =
        fmax(presolver->settings.finite_bound_improvement_absolute,
             presolver->settings.finite_bound_improvement_relative *
                 fabs(current));
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

static int has_pending_fixed_box_candidate(
    const PreFOSPresolver *presolver,
    const PreFOSColumnWorkspace *column_workspace)
{
    size_t position;
    for (position = 0; position < presolver->n_fixed_box_dirty; ++position)
    {
        int box = presolver->fixed_box_dirty_queue[position];
        int column = presolver->original.box_indices[box];
        double lower = presolver->working_box_lower[box];
        double upper = presolver->working_box_upper[box];
        long double difference;
        long double scale;
        if (presolver->is_fixed[column] ||
            presolver->is_substituted[column] ||
            presolver->is_parallel_removed[column])
            continue;
        if (!isfinite(lower) || !isfinite(upper)) continue;
        difference = (long double) upper - (long double) lower;
        scale = fmaxl(
            1.0L,
            fmaxl(fabsl((long double) lower),
                  fabsl((long double) upper)));
        if (difference <=
            (long double) presolver->settings.fixed_variable_tolerance * scale)
            return 1;
        if (column_workspace &&
            presolver->settings.feasibility_tolerance > 0.0 &&
            difference >= 0.0L &&
            prefos_internal_column_is_linear_box(
                presolver, column_workspace, column))
        {
            double column_scale =
                prefos_internal_column_max_abs_coefficient(
                    presolver, column_workspace, column);
            if (column_scale > 0.0 &&
                difference * (long double) column_scale <=
                    (long double)
                        presolver->settings.feasibility_tolerance)
                return 1;
        }
    }
    return 0;
}

static void fix_close_materialized_full_scan_column(
    PreFOSPresolver *presolver, PreFOSLinearPropagationState *state,
    int column, int box_position)
{
    double lower, upper;
    long double difference, scale;

    if (state || !presolver->working_matrix_is_materialized ||
        !presolver->settings.fix_close_box_bounds ||
        presolver->is_fixed[column])
        return;
    lower = presolver->working_box_lower[box_position];
    upper = presolver->working_box_upper[box_position];
    if (!isfinite(lower) || !isfinite(upper)) return;
    difference = (long double) upper - (long double) lower;
    scale = fmaxl(
        1.0L,
        fmaxl(fabsl((long double) lower), fabsl((long double) upper)));
    if (difference < 0.0L ||
        difference >
            (long double) presolver->settings.fixed_variable_tolerance *
                scale)
        return;
    prefos_internal_mark_fixed_column(
        presolver, column,
        prefos_internal_safe_midpoint(lower, upper));
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
    {
        if (trace_linear_conflicts())
        {
            fprintf(
                stderr,
                "PreFOS lower-bound conflict source_row=%d column=%d "
                "candidate=%.17g current_lower=%.17g upper=%.17g\n",
                row, column, candidate, current, upper);
        }
        return PREFOS_STATUS_PRIMAL_INFEASIBLE;
    }
    if (candidate <= current) return PREFOS_STATUS_OK;
    if (candidate < upper &&
        !prefos_internal_is_significant_improvement(presolver, current, candidate, 1))
        return PREFOS_STATUS_OK;

    if (candidate > upper) candidate = upper;
    if (trace_linear_updates())
        fprintf(
            stderr,
            "PreFOS prop-update row=%d col=%d side=lower old=%.17g "
            "candidate=%.17g\n",
            row, column, current, candidate);
    if (propagated_bound_should_be_materialized(presolver, box_position, candidate,
                                                1))
    {
        double working_current = presolver->working_box_lower[box_position];
        status = prefos_internal_append_bound_record(
            presolver, row, column, working_current, candidate, 1);
        if (status != PREFOS_STATUS_OK) return status;
        prefos_internal_mark_fixed_box_dirty(
            presolver, column);
        presolver->working_box_lower[box_position] = candidate;
        ++presolver->stats.materialized_propagated_box_bounds;
    }
    else
        ++presolver->stats.suppressed_propagated_box_bounds;
    if (state && state->defer_activity_updates)
    {
        status = update_deferred_activity_rows_for_bound_change(
            presolver, state, column, current, candidate, 1);
        if (status != PREFOS_STATUS_OK) return status;
    }
    presolver->propagation_lower[column] = candidate;
    if (state && state->column_workspace)
        prefos_internal_mark_parallel_column_bound_dirty(
            state->column_workspace, column);
    if (state && !isfinite(current))
        ++state->round_new_finite_bounds;
    if (trace_linear_column(column))
        fprintf(
            stderr,
            "PreFOS propagated lower source_row=%d column=%d old=%.17g "
            "new=%.17g upper=%.17g\n",
            row, column, current, candidate, upper);
    if (state && !state->defer_activity_updates)
    {
        status = update_cached_activities_for_bound_change(
            presolver, state, column, current, candidate, 1);
        if (status != PREFOS_STATUS_OK) return status;
    }
    fix_close_materialized_full_scan_column(
        presolver, state, column, box_position);
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
    {
        if (trace_linear_conflicts())
        {
            fprintf(
                stderr,
                "PreFOS upper-bound conflict source_row=%d column=%d "
                "candidate=%.17g lower=%.17g current_upper=%.17g\n",
                row, column, candidate, lower, current);
        }
        return PREFOS_STATUS_PRIMAL_INFEASIBLE;
    }
    if (candidate >= current) return PREFOS_STATUS_OK;
    if (candidate > lower &&
        !prefos_internal_is_significant_improvement(presolver, current, candidate, 0))
        return PREFOS_STATUS_OK;

    if (candidate < lower) candidate = lower;
    if (trace_linear_updates())
        fprintf(
            stderr,
            "PreFOS prop-update row=%d col=%d side=upper old=%.17g "
            "candidate=%.17g\n",
            row, column, current, candidate);
    if (propagated_bound_should_be_materialized(presolver, box_position, candidate,
                                                0))
    {
        double working_current = presolver->working_box_upper[box_position];
        status = prefos_internal_append_bound_record(
            presolver, row, column, working_current, candidate, 0);
        if (status != PREFOS_STATUS_OK) return status;
        prefos_internal_mark_fixed_box_dirty(
            presolver, column);
        presolver->working_box_upper[box_position] = candidate;
        ++presolver->stats.materialized_propagated_box_bounds;
    }
    else
        ++presolver->stats.suppressed_propagated_box_bounds;
    if (state && state->defer_activity_updates)
    {
        status = update_deferred_activity_rows_for_bound_change(
            presolver, state, column, current, candidate, 0);
        if (status != PREFOS_STATUS_OK) return status;
    }
    presolver->propagation_upper[column] = candidate;
    if (state && state->column_workspace)
        prefos_internal_mark_parallel_column_bound_dirty(
            state->column_workspace, column);
    if (state && !isfinite(current))
        ++state->round_new_finite_bounds;
    if (trace_linear_column(column))
        fprintf(
            stderr,
            "PreFOS propagated upper source_row=%d column=%d lower=%.17g "
            "old=%.17g new=%.17g\n",
            row, column, lower, current, candidate);
    if (state && !state->defer_activity_updates)
    {
        status = update_cached_activities_for_bound_change(
            presolver, state, column, current, candidate, 0);
        if (status != PREFOS_STATUS_OK) return status;
    }
    fix_close_materialized_full_scan_column(
        presolver, state, column, box_position);
    ++presolver->stats.propagated_box_bounds;
    *changed = 1;
    return PREFOS_STATUS_OK;
}

typedef struct
{
    PreFOSPresolver *presolver;
    PreFOSLinearPropagationState *state;
    const PreFOSColumnWorkspace *column_workspace;
    int row;
    int *changed;
    PreFOSStatus status;
    int minimum_activity_is_exact_zero;
    int maximum_activity_is_exact_zero;
    PreFOSRowActivity outward_row_activity;
    int outward_minimum_valid;
    int outward_maximum_valid;
    double changed_coefficient;
    double changed_old_bound;
    double changed_bound;
    int changed_is_lower;
    int changed_bound_valid;
} PreFOSLinearKernelContext;

static double finalize_implied_bound(
    const PreFOSPresolver *presolver,
    const PreFOSLinearPropagationState *state,
    const PreFOSColumnWorkspace *column_workspace, int column,
    long double candidate, int is_lower)
{
    int box_position = presolver->variable_to_box[column];
    if (box_position >= 0)
    {
        double opposite =
            is_lower ? presolver->propagation_upper[column]
                     : presolver->propagation_lower[column];
        double column_scale = 0.0;
        if (state && state->column_workspace)
            column_scale = prefos_internal_column_max_abs_coefficient(
                presolver, state->column_workspace, column);
        else if (column_workspace)
            column_scale = prefos_internal_column_max_abs_coefficient(
                presolver, column_workspace, column);
        else if (state)
            column_scale =
                state->box_max_abs_coefficient[box_position];
        if (isfinite(opposite) && column_scale > 0.0 &&
            (presolver->settings.propagated_bound_policy ==
                 PREFOS_PROPAGATED_BOUND_POLICY_FIRST_ORDER ||
             opposite ==
                 (is_lower
                      ? presolver->original.box_upper[box_position]
                      : presolver->original.box_lower[box_position])))
        {
            long double gap =
                is_lower ? (long double) opposite - candidate
                         : candidate - (long double) opposite;
            long double scaled_gap =
                gap * (long double) column_scale;
            if (gap >= 0.0L &&
                scaled_gap <=
                    (long double)
                        presolver->settings.feasibility_tolerance)
                return opposite;
        }
    }
    return relaxed_implied_bound(presolver, candidate, is_lower);
}

static PresolveKernelUpdate tighten_prefos_scalar_bound(
    void *context_pointer, int column, double coefficient,
    long double candidate, int is_lower)
{
    PreFOSLinearKernelContext *context = (PreFOSLinearKernelContext *) context_pointer;
    size_t changes_before = context->presolver->stats.propagated_box_bounds;
    PreFOSPresolver *presolver = context->presolver;
    double old_bound;
    double relaxed;

    if (context->status != PREFOS_STATUS_OK)
        return PRESOLVE_KERNEL_STOP;
    old_bound =
        is_lower ? presolver->propagation_lower[column]
                 : presolver->propagation_upper[column];
    if (context->state && trace_linear_profile())
        ++context->state->profile_bound_candidates;
    relaxed = finalize_implied_bound(
        presolver, context->state, context->column_workspace, column,
        candidate, is_lower);
    context->status =
        is_lower
            ? update_propagated_lower(presolver, context->row, column,
                                      relaxed, context->changed,
                                      context->state)
            : update_propagated_upper(presolver, context->row, column,
                                      relaxed, context->changed,
                                      context->state);
    if (presolver->stats.propagated_box_bounds > changes_before)
    {
        if (context->state && trace_linear_profile())
            ++context->state->profile_bound_changes;
        context->changed_coefficient = coefficient;
        context->changed_old_bound = old_bound;
        context->changed_bound =
            is_lower ? presolver->propagation_lower[column]
                     : presolver->propagation_upper[column];
        context->changed_is_lower = is_lower;
        context->changed_bound_valid = 1;
    }
    if (context->status != PREFOS_STATUS_OK ||
        (context->state && context->state->fallback_requested))
        return PRESOLVE_KERNEL_STOP;
    return presolver->stats.propagated_box_bounds > changes_before
               ? PRESOLVE_KERNEL_CHANGED
               : PRESOLVE_KERNEL_UNCHANGED;
}

static void refresh_prefos_linear_activity(
    void *context_pointer, PresolveLinearPropagationRow *row)
{
    PreFOSLinearKernelContext *context =
        (PreFOSLinearKernelContext *) context_pointer;
    const PreFOSRowActivity *activity;
    if (context->changed_bound_valid && context->state &&
        context->state->defer_activity_updates)
    {
        PreFOSRowActivity *cached =
            &context->state->activities[context->row];
        int minimum_side =
            (context->changed_is_lower &&
             context->changed_coefficient > 0.0) ||
            (!context->changed_is_lower &&
             context->changed_coefficient < 0.0);
        size_t *infinite_count =
            minimum_side ? &cached->n_infinite_min
                         : &cached->n_infinite_max;
        long double *accumulator =
            minimum_side
                ? &context->state
                       ->finite_min_accumulators[context->row]
                : &context->state
                       ->finite_max_accumulators[context->row];
        unsigned char *accumulator_stale =
            minimum_side
                ? &context->state
                       ->finite_min_accumulator_stale[context->row]
                : &context->state
                       ->finite_max_accumulator_stale[context->row];
        int *unique_position =
            minimum_side
                ? &context->state
                       ->unique_infinite_min_positions[context->row]
                : &context->state
                       ->unique_infinite_max_positions[context->row];
        double *finite_sum =
            minimum_side ? &cached->finite_min
                         : &cached->finite_max;
        size_t infinite_count_before = *infinite_count;
        int sum_updated = 0;
        int recompute_required = 0;

        context->status = update_cached_activity_term(
            finite_sum, accumulator, infinite_count,
            accumulator_stale, context->changed_coefficient,
            context->changed_old_bound, context->changed_bound,
            &sum_updated, &recompute_required);
        if (context->status != PREFOS_STATUS_OK) return;
        if (*infinite_count != infinite_count_before)
            *unique_position =
                PREFOS_UNIQUE_INFINITE_POSITION_UNKNOWN;
        if (recompute_required)
        {
            context->status = compute_row_activity(
                context->presolver, (size_t) context->row, 0,
                cached);
            if (context->status != PREFOS_STATUS_OK) return;
            context->state->finite_min_accumulators[context->row] =
                (long double) cached->finite_min;
            context->state->finite_max_accumulators[context->row] =
                (long double) cached->finite_max;
            context->state
                ->finite_min_accumulator_stale[context->row] = 0;
            context->state
                ->finite_max_accumulator_stale[context->row] = 0;
            context->state
                ->unique_infinite_min_positions[context->row] =
                PREFOS_UNIQUE_INFINITE_POSITION_UNKNOWN;
            context->state
                ->unique_infinite_max_positions[context->row] =
                PREFOS_UNIQUE_INFINITE_POSITION_UNKNOWN;
        }
        context->state->redundancy_activity_stale[context->row] = 1;
    }
    if (context->changed_bound_valid)
    {
        int changes_minimum =
            (context->changed_is_lower &&
             context->changed_coefficient > 0.0) ||
            (!context->changed_is_lower &&
             context->changed_coefficient < 0.0);
        int *cached =
            changes_minimum
                ? &context->minimum_activity_is_exact_zero
                : &context->maximum_activity_is_exact_zero;
        int *outward_valid =
            changes_minimum
                ? &context->outward_minimum_valid
                : &context->outward_maximum_valid;
        if (*cached > 0 && context->changed_bound != 0.0)
            *cached = 0;
        *outward_valid = 0;
        context->changed_bound_valid = 0;
    }
    else
    {
        context->minimum_activity_is_exact_zero = -1;
        context->maximum_activity_is_exact_zero = -1;
        context->outward_minimum_valid = 0;
        context->outward_maximum_valid = 0;
    }
    if (!context->state) return;
    activity = &context->state->activities[context->row];
    row->finite_min_activity = activity->finite_min;
    row->finite_max_activity = activity->finite_max;
    row->n_infinite_min = activity->n_infinite_min;
    row->n_infinite_max = activity->n_infinite_max;
}

static int prefos_linear_activity_is_exact_zero(
    void *context_pointer, int use_minimum)
{
    PreFOSLinearKernelContext *context =
        (PreFOSLinearKernelContext *) context_pointer;
    const PreFOSCsrMatrix *matrix =
        &context->presolver->original.A;
    int *cached = use_minimum
                      ? &context->minimum_activity_is_exact_zero
                      : &context->maximum_activity_is_exact_zero;
    int position;

    if (*cached >= 0) return *cached;
    if (context->state && trace_linear_profile())
        ++context->state->profile_exact_zero_checks;
    *cached = 1;
    for (position = matrix->row_pointers[context->row];
         position < matrix->row_pointers[context->row + 1];
         ++position)
    {
        int column = matrix->column_indices[position];
        double coefficient = matrix->values[position];
        double bound;
        if (context->state && trace_linear_profile())
            ++context->state->profile_exact_zero_terms;
        if (coefficient == 0.0 ||
            !prefos_internal_term_is_active_in_row(
                context->presolver, (size_t) context->row, column))
            continue;
        if (use_minimum)
            bound = coefficient > 0.0
                        ? context->presolver->propagation_lower[column]
                        : context->presolver->propagation_upper[column];
        else
            bound = coefficient > 0.0
                        ? context->presolver->propagation_upper[column]
                        : context->presolver->propagation_lower[column];
        if (bound != 0.0)
        {
            *cached = 0;
            break;
        }
    }
    return *cached;
}

static int prefos_linear_outward_activity(
    void *context_pointer, int use_minimum, long double *value)
{
    PreFOSLinearKernelContext *context =
        (PreFOSLinearKernelContext *) context_pointer;
    int *valid = use_minimum
                     ? &context->outward_minimum_valid
                     : &context->outward_maximum_valid;
    size_t n_infinite;

    if (!*valid)
    {
        context->status = compute_row_activity(
            context->presolver, (size_t) context->row, 1,
            &context->outward_row_activity);
        if (context->status != PREFOS_STATUS_OK) return 0;
        context->outward_minimum_valid = 1;
        context->outward_maximum_valid = 1;
        if (context->state && trace_linear_profile())
        {
            const PreFOSCsrMatrix *matrix =
                &context->presolver->original.A;
            ++context->state->profile_outward_activity_scans;
            context->state->profile_outward_activity_terms +=
                (size_t) (matrix->row_pointers[context->row + 1] -
                          matrix->row_pointers[context->row]);
        }
    }
    n_infinite =
        use_minimum
            ? context->outward_row_activity.n_infinite_min
            : context->outward_row_activity.n_infinite_max;
    if (n_infinite != 0) return 0;
    *value =
        (long double)
            (use_minimum
                 ? context->outward_row_activity.finite_min
                 : context->outward_row_activity.finite_max);
    return 1;
}

static PreFOSStatus refresh_unique_infinite_positions(
    const PreFOSPresolver *presolver,
    PreFOSLinearPropagationState *state, size_t row,
    PreFOSRowActivity *activity)
{
    const PreFOSCsrMatrix *matrix = &presolver->original.A;
    int start = matrix->row_pointers[row];
    int end = matrix->row_pointers[row + 1];
    int minimum_position = -1;
    int maximum_position = -1;
    size_t n_infinite_min = 0;
    size_t n_infinite_max = 0;
    int position;

    for (position = start; position < end; ++position)
    {
        int column = matrix->column_indices[position];
        double coefficient = matrix->values[position];
        double minimum_bound;
        double maximum_bound;
        int relative_position = position - start;

        if (coefficient == 0.0 ||
            !prefos_internal_term_is_active_in_row(
                presolver, row, column))
            continue;
        minimum_bound = coefficient > 0.0
                            ? presolver->propagation_lower[column]
                            : presolver->propagation_upper[column];
        maximum_bound = coefficient > 0.0
                            ? presolver->propagation_upper[column]
                            : presolver->propagation_lower[column];
        if (!isfinite(minimum_bound))
        {
            ++n_infinite_min;
            minimum_position =
                n_infinite_min == 1 ? relative_position : -1;
        }
        if (!isfinite(maximum_bound))
        {
            ++n_infinite_max;
            maximum_position =
                n_infinite_max == 1 ? relative_position : -1;
        }
    }
    if (n_infinite_min != activity->n_infinite_min ||
        n_infinite_max != activity->n_infinite_max)
    {
        PreFOSRowActivity refreshed;
        PreFOSStatus status;
        if (trace_linear_cache())
            fprintf(
                stderr,
                "PreFOS unique-infinite mismatch row=%zu "
                "cached=(%zu,%zu) actual=(%zu,%zu)\n",
                row, activity->n_infinite_min,
                activity->n_infinite_max, n_infinite_min,
                n_infinite_max);
        status = compute_row_activity(
            presolver, row, 0, &refreshed);
        if (status != PREFOS_STATUS_OK) return status;
        *activity = refreshed;
        state->finite_min_accumulators[row] =
            (long double) refreshed.finite_min;
        state->finite_max_accumulators[row] =
            (long double) refreshed.finite_max;
        state->finite_min_accumulator_stale[row] = 0;
        state->finite_max_accumulator_stale[row] = 0;
        n_infinite_min = refreshed.n_infinite_min;
        n_infinite_max = refreshed.n_infinite_max;
        minimum_position = -1;
        maximum_position = -1;
        for (position = start; position < end; ++position)
        {
            int column = matrix->column_indices[position];
            double coefficient = matrix->values[position];
            double minimum_bound;
            double maximum_bound;
            int relative_position = position - start;
            if (coefficient == 0.0 ||
                !prefos_internal_term_is_active_in_row(
                    presolver, row, column))
                continue;
            minimum_bound =
                coefficient > 0.0
                    ? presolver->propagation_lower[column]
                    : presolver->propagation_upper[column];
            maximum_bound =
                coefficient > 0.0
                    ? presolver->propagation_upper[column]
                    : presolver->propagation_lower[column];
            if (!isfinite(minimum_bound))
                minimum_position =
                    n_infinite_min == 1 ? relative_position : -1;
            if (!isfinite(maximum_bound))
                maximum_position =
                    n_infinite_max == 1 ? relative_position : -1;
        }
    }
    state->unique_infinite_min_positions[row] =
        minimum_position;
    state->unique_infinite_max_positions[row] =
        maximum_position;
    return PREFOS_STATUS_OK;
}

static PreFOSStatus propagate_single_row(PreFOSPresolver *presolver, size_t row,
                                      PreFOSRowActivity *activity, int *changed,
                                      PreFOSLinearPropagationState *state,
                                      const PreFOSColumnWorkspace *column_workspace,
                                      int activity_is_verified)
{
    const PreFOSProblemData *problem = &presolver->original;
    const PreFOSCsrMatrix *A = &problem->A;
    PreFOSTimestamp profile_start, profile_stop;
    PreFOSLinearKernelContext context;
    PresolveLinearPropagationRow kernel_row;
    PresolveLinearPropagationOps operations;
    PreFOSStatus status;
    int propagate_minimum;
    int propagate_maximum;
    int use_single_infinite_candidates = 0;
    int stopped;

    if (!activity_is_verified &&
        activity_needs_rigorous_confirmation(
            presolver, row, activity))
    {
        PreFOSRowActivity exact, outward;
        if (state && trace_linear_profile())
            prefos_internal_timer_now(&profile_start);
        status = compute_row_activities(
            presolver, row, &exact, &outward);
        if (status != PREFOS_STATUS_OK) return status;
        status = check_row_activity(presolver, row, &outward);
        if (status != PREFOS_STATUS_OK) return status;
        *activity = exact;
        if (state)
        {
            state->finite_min_accumulators[row] =
                (long double) exact.finite_min;
            state->finite_max_accumulators[row] =
                (long double) exact.finite_max;
            state->finite_min_accumulator_stale[row] = 0;
            state->finite_max_accumulator_stale[row] = 0;
        }
        if (state && trace_linear_profile())
        {
            prefos_internal_timer_now(&profile_stop);
            state->profile_confirmation_milliseconds +=
                prefos_internal_timer_elapsed_milliseconds(
                    &profile_start, &profile_stop);
        }
    }
    else if (!activity_is_verified)
        status = check_row_activity(presolver, row, activity);
    else
        status = PREFOS_STATUS_OK;
    if (status != PREFOS_STATUS_OK) return status;
    if (!row_can_propagate(presolver, row, activity)) return PREFOS_STATUS_OK;
    if (state &&
        !row_has_potential_bound_improvement(
            presolver, state, row, activity))
        return PREFOS_STATUS_OK;
    if (state &&
        ((activity->n_infinite_min == 1 &&
          state->unique_infinite_min_positions[row] ==
              PREFOS_UNIQUE_INFINITE_POSITION_UNKNOWN) ||
         (activity->n_infinite_max == 1 &&
          state->unique_infinite_max_positions[row] ==
              PREFOS_UNIQUE_INFINITE_POSITION_UNKNOWN)))
    {
        status = refresh_unique_infinite_positions(
            presolver, state, row, activity);
        if (status != PREFOS_STATUS_OK) return status;
    }

    context = (PreFOSLinearKernelContext){
        .presolver = presolver,
        .state = state,
        .column_workspace = column_workspace,
        .row = (int) row,
        .changed = changed,
        .status = PREFOS_STATUS_OK,
        .minimum_activity_is_exact_zero = -1,
        .maximum_activity_is_exact_zero = -1};
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
        .column_flags = presolver->is_fixed,
        .inactive_mask = 1,
        .row_excluded_columns =
            presolver->residual_source_column,
        .row_exclusion_flags =
            presolver->n_residual_row_substitutions > 0
                ? presolver->substitution_keeps_source_row
                : NULL,
        .row_exclusion_sources = presolver->substitution_source_row,
        .row_index = (int) row,
        .activity_is_outward = 0,
        .maximum_inferred_bound_magnitude =
            PRESOLVE_DEFAULT_MAX_INFERRED_BOUND_MAGNITUDE,
        .tighten_bound = tighten_prefos_scalar_bound,
        .refresh_activity = refresh_prefos_linear_activity,
        .activity_is_exact_zero =
            prefos_linear_activity_is_exact_zero,
        .outward_activity =
            prefos_linear_outward_activity,
        .residual_rescans =
            state && trace_linear_profile()
                ? &state->profile_residual_rescans
                : NULL,
        .residual_rescan_terms =
            state && trace_linear_profile()
                ? &state->profile_residual_rescan_terms
                : NULL};
    propagate_minimum =
        (!kernel_row.upper_is_infinite &&
         kernel_row.n_infinite_min <= 1) ||
        (kernel_row.upper_is_infinite &&
         kernel_row.n_infinite_min == 1 &&
         kernel_row.n_infinite_max == 0);
    propagate_maximum =
        (!kernel_row.lower_is_infinite &&
         kernel_row.n_infinite_max <= 1) ||
        (kernel_row.lower_is_infinite &&
         kernel_row.n_infinite_max == 1 &&
         kernel_row.n_infinite_min == 0);
    if (state && (propagate_minimum || propagate_maximum) &&
        (!propagate_minimum ||
         (kernel_row.n_infinite_min == 1 &&
          state->unique_infinite_min_positions[row] >= 0)) &&
        (!propagate_maximum ||
         (kernel_row.n_infinite_max == 1 &&
          state->unique_infinite_max_positions[row] >= 0)))
        use_single_infinite_candidates = 1;
    if (state && trace_linear_profile())
        prefos_internal_timer_now(&profile_start);
    if (use_single_infinite_candidates)
    {
        if (trace_linear_profile())
            ++state->profile_single_infinite_candidate_rows;
        (void) presolve_internal_propagate_single_infinite_candidates(
            &kernel_row, &operations,
            state->unique_infinite_min_positions[row],
            state->unique_infinite_max_positions[row], &stopped);
    }
    else
        (void) presolve_internal_propagate_linear_row(
            &kernel_row, &operations, &stopped);
    if (state && trace_linear_profile())
    {
        prefos_internal_timer_now(&profile_stop);
        state->profile_kernel_milliseconds +=
            prefos_internal_timer_elapsed_milliseconds(
                &profile_start, &profile_stop);
    }
    if (stopped || context.status != PREFOS_STATUS_OK)
        return context.status;
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

PreFOSStatus prefos_internal_probe_linear_rows(
    PreFOSPresolver *presolver, const unsigned char *probe_rows,
    int *bound_changed)
{
    size_t row;
    int integrate_redundancy;

    if (!presolver || !probe_rows || !bound_changed)
        return PREFOS_STATUS_INVALID_ARGUMENT;
    *bound_changed = 0;
    integrate_redundancy =
        scalar_redundancy_can_share_activity(presolver);

    for (row = 0; row < presolver->original.A.rows; ++row)
    {
        PreFOSRowActivity activity, outward_activity;
        PreFOSStatus status;
        int changed = 0;
        int removed = 0;

        if (!probe_rows[row] ||
            !row_is_active_for_linear_propagation(presolver, row))
            continue;
        status = compute_row_activities(
            presolver, row, &activity, &outward_activity);
        if (status != PREFOS_STATUS_OK) return status;
        presolver->stats.linear_activity_nnz_computed +=
            activity.n_nonzeros;
        ++presolver->stats.linear_rows_processed;
        presolver->stats.linear_nnz_processed +=
            activity.n_nonzeros;

        if (integrate_redundancy)
        {
            status = apply_scalar_row_classification(
                presolver, row, &outward_activity, &removed, NULL);
            if (status != PREFOS_STATUS_OK) return status;
            if (removed) continue;
        }
        status = propagate_single_row(
            presolver, row, &activity, &changed, NULL, NULL,
            integrate_redundancy);
        if (status != PREFOS_STATUS_OK) return status;
        if (changed) *bound_changed = 1;
    }
    return PREFOS_STATUS_OK;
}

static PreFOSStatus propagate_linear_bounds_full_scan(PreFOSPresolver *presolver,
                                                   int max_rounds, size_t work_limit,
                                                   PreFOSColumnWorkspace
                                                       *column_workspace)
{
    int integrate_redundancy =
        scalar_redundancy_can_share_activity(presolver);
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
            PreFOSRowActivity activity, outward_activity;
            PreFOSStatus status;
            int removed = 0;
            if (!presolver->remove_rows[row] &&
                !isfinite(presolver->working_constraint_lower[row]) &&
                !isfinite(presolver->working_constraint_upper[row]) &&
                integrate_redundancy)
            {
                prefos_internal_mark_removed_row(presolver, row);
                ++presolver->stats.removed_redundant_rows;
                continue;
            }
            if (!row_is_active_for_linear_propagation(presolver, row)) continue;
            status = compute_row_activities(
                presolver, row, &activity, &outward_activity);
            if (status != PREFOS_STATUS_OK) return status;
            presolver->stats.linear_activity_nnz_computed += activity.n_nonzeros;
            ++presolver->stats.linear_rows_processed;
            presolver->stats.linear_nnz_processed += activity.n_nonzeros;
            if (integrate_redundancy)
            {
                status = apply_scalar_row_classification(
                    presolver, row, &outward_activity, &removed,
                    column_workspace);
                if (status != PREFOS_STATUS_OK) return status;
                if (removed) continue;
            }
            status = propagate_single_row(
                presolver, row, &activity, &changed, NULL,
                column_workspace, integrate_redundancy);
            if (status != PREFOS_STATUS_OK) return status;
        }
        if (!changed)
        {
            presolver->linear_propagation_complete = 1;
            if (integrate_redundancy)
                presolver->scalar_redundancy_completed = 1;
            break;
        }
        {
            size_t changes = presolver->stats.propagated_box_bounds - changes_before;
            size_t work = linear_propagation_work_used(presolver) - work_before;
            int stale = propagation_round_is_stale(presolver, changes, work);
            if (has_pending_fixed_box_candidate(
                    presolver, column_workspace))
                break;
            if (stale)
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
            work_limit = renew_productive_round_work_budget(
                presolver, work_limit, estimated_round_work, changes, work);
        }
    }
    return PREFOS_STATUS_OK;
}

static PreFOSStatus propagate_linear_bounds_gpu(PreFOSPresolver *presolver, int max_rounds,
                                             size_t work_limit,
                                             PreFOSColumnWorkspace
                                                 *column_workspace)
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

    context = prefos_internal_cuda_workspace_get(presolver, &cuda_status);
    presolver->stats.linear_gpu_setup_milliseconds =
        presolver->stats.cuda_workspace_setup_milliseconds;
    if (!context || cuda_status != PREFOS_CUDA_PROPAGATION_OK)
    {
        ++presolver->stats.linear_gpu_fallbacks;
        return propagate_linear_bounds_full_scan(
            presolver, max_rounds, work_limit, column_workspace);
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
            presolver->working_constraint_lower,
            presolver->working_constraint_upper, presolver->remove_rows,
            NULL, presolver->settings.feasibility_tolerance,
            PRESOLVE_DEFAULT_MAX_INFERRED_BOUND_MAGNITUDE, lower_candidates,
            upper_candidates, lower_sources, upper_sources,
            &suspected_infeasible_row, &transfer_milliseconds, &kernel_milliseconds);
        presolver->stats.linear_gpu_transfer_milliseconds += transfer_milliseconds;
        presolver->stats.linear_gpu_kernel_milliseconds += kernel_milliseconds;
        if (cuda_status != PREFOS_CUDA_PROPAGATION_OK)
        {
            int remaining_rounds = max_rounds - completed_rounds;
            ++presolver->stats.linear_gpu_fallbacks;
            free(lower_candidates);
            free(upper_candidates);
            free(lower_sources);
            free(upper_sources);
            return propagate_linear_bounds_full_scan(
                presolver, remaining_rounds, work_limit,
                column_workspace);
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
                presolver, (size_t) suspected_infeasible_row, 1, &activity);
            if (status == PREFOS_STATUS_OK)
                status = check_row_activity(
                    presolver, (size_t) suspected_infeasible_row, &activity);
        }
        for (column = 0; status == PREFOS_STATUS_OK && column < problem->n; ++column)
        {
            if (lower_sources[column] >= 0)
                status = update_propagated_lower(
                    presolver, lower_sources[column], (int) column,
                    finalize_implied_bound(
                        presolver, NULL, column_workspace, (int) column,
                        (long double) lower_candidates[column], 1),
                    &changed, NULL);
            if (status == PREFOS_STATUS_OK && upper_sources[column] >= 0)
                status = update_propagated_upper(
                    presolver, upper_sources[column], (int) column,
                    finalize_implied_bound(
                        presolver, NULL, column_workspace, (int) column,
                        (long double) upper_candidates[column], 0),
                    &changed, NULL);
        }
        if (status != PREFOS_STATUS_OK)
        {
            free(lower_candidates);
            free(upper_candidates);
            free(lower_sources);
            free(upper_sources);
            return status;
        }
        if (!changed)
        {
            presolver->linear_propagation_complete = 1;
            break;
        }
        if (has_pending_fixed_box_candidate(
                presolver, column_workspace))
            break;
        {
            size_t changes = presolver->stats.propagated_box_bounds - changes_before;
            size_t work = linear_propagation_work_used(presolver) - work_before;
            int stale = propagation_round_is_stale(presolver, changes, work);
            if (stale)
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
            work_limit = renew_productive_round_work_budget(
                presolver, work_limit, estimated_round_work, changes, work);
        }
    }
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

static PreFOSStatus rebuild_linear_propagation_cache(
    PreFOSPresolver *presolver,
    PreFOSColumnWorkspace *column_workspace,
    PreFOSLinearPropagationState **state)
{
    PreFOSStatus status;
    prefos_internal_free_linear_propagation_cache(presolver);
    *state = (PreFOSLinearPropagationState *) calloc(1, sizeof(**state));
    if (!*state) return PREFOS_STATUS_OUT_OF_MEMORY;
    status = initialize_linear_propagation_state(
        presolver, *state, column_workspace);
    if (status != PREFOS_STATUS_OK)
    {
        prefos_internal_free_linear_propagation_state(*state);
        free(*state);
        *state = NULL;
        return status;
    }
    presolver->linear_propagation_cache = *state;
    return PREFOS_STATUS_OK;
}

PreFOSStatus prefos_internal_propagate_linear_bounds(
    PreFOSPresolver *presolver,
    PreFOSColumnWorkspace *column_workspace)
{
    PreFOSLinearPropagationState *state;
    PreFOSTimestamp profile_start, profile_ready, profile_stop;
    PreFOSTimestamp profile_row_start, profile_row_stop;
    PreFOSStatus status;
    int round;
    int stale_rounds = 0;
    size_t work_limit;
    double profile_row_milliseconds = 0.0;
    size_t profile_row_calls = 0;
    size_t profile_rows_pruned = 0;
    presolver->linear_propagation_complete = 0;
    if (!presolver->settings.linear_propagation ||
        presolver->settings.max_linear_propagation_rounds == 0 ||
        presolver->original.A.rows == 0)
    {
        if (presolver->original.A.rows == 0)
            presolver->linear_propagation_complete = 1;
        return PREFOS_STATUS_OK;
    }
    if (trace_linear_profile())
        prefos_internal_timer_now(&profile_start);
    work_limit = linear_propagation_work_limit(presolver);
    if (prefer_full_scan_linear_propagation(presolver))
    {
        prefos_internal_free_linear_propagation_cache(presolver);
        if (presolver->settings.linear_propagation_gpu &&
            presolver->n_residual_row_substitutions == 0)
            return propagate_linear_bounds_gpu(
                presolver, presolver->settings.max_linear_propagation_rounds,
                work_limit, column_workspace);
        return propagate_linear_bounds_full_scan(
            presolver, presolver->settings.max_linear_propagation_rounds,
            work_limit, column_workspace);
    }

    if (work_limit != SIZE_MAX && presolver->original.A.nnz > work_limit)
    {
        ++presolver->stats.linear_budget_stops;
        return PREFOS_STATUS_OK;
    }

    state = presolver->linear_propagation_cache;
    if (!state ||
        !linear_propagation_cache_matches(
            presolver, state, column_workspace))
    {
        status = rebuild_linear_propagation_cache(
            presolver, column_workspace, &state);
        if (status != PREFOS_STATUS_OK) return status;
    }
    else
    {
        state->total_work_limit = work_limit;
        status = synchronize_linear_propagation_cache(
            presolver, state);
        if (status != PREFOS_STATUS_OK)
        {
            trace_linear_failure(
                presolver, state, "cache-synchronize",
                state->current_row, status);
            prefos_internal_free_linear_propagation_cache(presolver);
            return status;
        }
        if (state->fallback_requested)
        {
            status = rebuild_linear_propagation_cache(
                presolver, column_workspace, &state);
            if (status != PREFOS_STATUS_OK) return status;
        }
        else
            ++presolver->stats.linear_cache_reuses;
    }
    if (trace_linear_profile())
        prefos_internal_timer_now(&profile_ready);
    state->total_work_limit = work_limit;
    if (state->integrate_redundancy &&
        state->total_work_limit != SIZE_MAX)
        state->total_work_limit = saturated_work_add(
            state->total_work_limit, state->work_budget_quantum);
    if (state->integrate_redundancy &&
        state->dirty_rows.current_count == 0)
        presolver->scalar_redundancy_completed = 1;
    if (state->dirty_rows.current_count == 0 &&
        state->dirty_rows.next_count == 0)
        presolver->linear_propagation_complete = 1;
    state->retain_deferred_frontier =
        getenv("PREFOS_DISABLE_BROAD_FRONTIER_CACHE_PERSISTENCE") == NULL &&
        (getenv("PREFOS_FORCE_BROAD_FRONTIER_STOP") ||
         state->work_budget_quantum >=
             PREFOS_BROAD_FRONTIER_CACHE_MIN_NNZ);
    state->defer_activity_updates =
        presolver->settings.linear_propagation_max_stale_rounds > 0 &&
        !getenv("PREFOS_DISABLE_BROAD_FRONTIER_STOP") &&
        (getenv("PREFOS_FORCE_BROAD_FRONTIER_STOP") ||
         (state->work_budget_quantum >= 1000000 &&
          state->dirty_rows.current_count >= 65536 &&
          state->dirty_rows.current_count >=
              (presolver->original.A.rows + 4) / 5));
    for (round = 1; round <= presolver->settings.max_linear_propagation_rounds &&
                    state->dirty_rows.current_count > 0;
         ++round)
    {
        int row;
        size_t changes_before = presolver->stats.propagated_box_bounds;
        size_t work_before = linear_propagation_work_used(presolver);
        state->round_changes_before = changes_before;
        state->round_work_before = work_before;
        state->round_new_finite_bounds = 0;
        ++presolver->stats.linear_propagation_rounds;
        ++presolver->stats.linear_event_rounds;
        while (presolve_dirty_rows_pop(&state->dirty_rows, &row))
        {
            int changed = 0;
            int removed = 0;
            int redundancy_classification_pending = 0;
            size_t row_work = state->activities[row].n_nonzeros;
            if (!row_is_active_for_linear_propagation(
                    presolver, (size_t) row))
                continue;
            if (!event_work_fits_or_renews(
                    presolver, state, row_work))
            {
                ++presolver->stats.linear_budget_stops;
                if (!schedule_cached_row(presolver, state, row))
                {
                    prefos_internal_free_linear_propagation_cache(
                        presolver);
                    return PREFOS_STATUS_OUT_OF_MEMORY;
                }
                return PREFOS_STATUS_OK;
            }
            ++presolver->stats.linear_rows_processed;
            presolver->stats.linear_nnz_processed +=
                state->activities[row].n_nonzeros;
            if (state->activity_recompute_required[row])
            {
                if (trace_linear_profile())
                {
                    ++state->profile_activity_recomputes;
                    state->profile_activity_recompute_terms +=
                        state->activities[row].n_nonzeros;
                }
                status = compute_row_activity(
                    presolver, (size_t) row, 0,
                    &state->activities[row]);
                if (status != PREFOS_STATUS_OK)
                {
                    trace_linear_failure(
                        presolver, state, "activity-recompute",
                        row, status);
                    prefos_internal_free_linear_propagation_cache(
                        presolver);
                    return status;
                }
                state->finite_min_accumulators[row] =
                    (long double) state->activities[row].finite_min;
                state->finite_max_accumulators[row] =
                    (long double) state->activities[row].finite_max;
                state->finite_min_accumulator_stale[row] = 0;
                state->finite_max_accumulator_stale[row] = 0;
                state->unique_infinite_min_positions[row] =
                    PREFOS_UNIQUE_INFINITE_POSITION_UNKNOWN;
                state->unique_infinite_max_positions[row] =
                    PREFOS_UNIQUE_INFINITE_POSITION_UNKNOWN;
                presolver->stats.linear_activity_nnz_computed +=
                    state->activities[row].n_nonzeros;
                state->activity_recompute_required[row] = 0;
                state->redundancy_activity_stale[row] = 1;
            }
            if (state->redundancy_activity_stale[row])
            {
                if (state->integrate_redundancy)
                {
                    int recomputed;
                    status = refresh_rigorous_activity_if_needed(
                        presolver, (size_t) row,
                        &state->activities[row],
                        &state->redundancy_activities[row],
                        &recomputed);
                    if (status != PREFOS_STATUS_OK)
                    {
                        trace_linear_failure(
                            presolver, state,
                            "rigorous-activity", row, status);
                        prefos_internal_free_linear_propagation_cache(
                            presolver);
                        return status;
                    }
                    if (recomputed)
                        presolver->stats.linear_activity_nnz_computed +=
                            state->activities[row].n_nonzeros;
                    redundancy_classification_pending = 1;
                }
                state->redundancy_activity_stale[row] = 0;
            }
            if (state->integrate_redundancy &&
                redundancy_classification_pending)
            {
                status = apply_scalar_row_classification(
                    presolver, (size_t) row,
                    &state->redundancy_activities[row],
                    &removed, column_workspace);
                if (status != PREFOS_STATUS_OK)
                {
                    trace_linear_failure(
                        presolver, state,
                        "row-classification", row, status);
                    prefos_internal_free_linear_propagation_cache(
                        presolver);
                    return status;
                }
#ifndef NDEBUG
                state->debug_cached_constraint_lower[row] =
                    presolver->working_constraint_lower[row];
                state->debug_cached_constraint_upper[row] =
                    presolver->working_constraint_upper[row];
#endif
                if (removed) continue;
            }
            state->current_row = row;
            if (trace_linear_profile())
                prefos_internal_timer_now(&profile_row_start);
            status = propagate_single_row(
                presolver, (size_t) row, &state->activities[row],
                &changed, state, column_workspace,
                state->integrate_redundancy);
            if (trace_linear_profile())
            {
                prefos_internal_timer_now(&profile_row_stop);
                profile_row_milliseconds +=
                    prefos_internal_timer_elapsed_milliseconds(
                        &profile_row_start, &profile_row_stop);
                ++profile_row_calls;
            }
            state->current_row = -1;
            if (status != PREFOS_STATUS_OK)
            {
                trace_linear_failure(
                    presolver, state, "row-propagation",
                    row, status);
                prefos_internal_free_linear_propagation_cache(presolver);
                return status;
            }
            if (state->fallback_requested)
            {
                int remaining_rounds =
                    presolver->settings.max_linear_propagation_rounds - round + 1;
                ++presolver->stats.linear_full_scan_fallbacks;
                work_limit = state->total_work_limit;
                prefos_internal_free_linear_propagation_cache(presolver);
                return propagate_linear_bounds_full_scan(
                    presolver, remaining_rounds, work_limit,
                    column_workspace);
            }
        }
        presolve_dirty_rows_finish_round(&state->dirty_rows);
        if (state->defer_activity_updates)
        {
            ++presolver->stats.linear_stale_stops;
            if (trace_linear_profile())
            {
                prefos_internal_timer_now(&profile_stop);
                fprintf(
                    stderr,
                    "PreFOS linear-profile broad-frontier prepare=%.6f "
                    "process=%.6f rows=%.6f confirm=%.6f kernel=%.6f "
                    "row_calls=%zu dirty=%zu activity_recomputes=%zu "
                    "activity_recompute_terms=%zu candidates=%zu "
                    "changes=%zu single_infinite=%zu rescans=%zu "
                    "rescan_terms=%zu outward_scans=%zu "
                    "outward_terms=%zu\n",
                    prefos_internal_timer_elapsed_milliseconds(
                        &profile_start, &profile_ready),
                    prefos_internal_timer_elapsed_milliseconds(
                        &profile_ready, &profile_stop),
                    profile_row_milliseconds,
                    state->profile_confirmation_milliseconds,
                    state->profile_kernel_milliseconds,
                    profile_row_calls,
                    state->dirty_rows.current_count,
                    state->profile_activity_recomputes,
                    state->profile_activity_recompute_terms,
                    state->profile_bound_candidates,
                    state->profile_bound_changes,
                    state->profile_single_infinite_candidate_rows,
                    state->profile_residual_rescans,
                    state->profile_residual_rescan_terms,
                    state->profile_outward_activity_scans,
                    state->profile_outward_activity_terms);
            }
            presolver->scalar_redundancy_completed = 1;
            presolver->linear_propagation_complete = 1;
            presolver->linear_propagation_broad_frontier_stop = 1;
            presolver->linear_propagation_broad_frontier_fixed_epoch =
                presolver->fixed_column_epoch;
            presolver
                ->linear_propagation_broad_frontier_column_transformations =
                presolver->transformations.n_column_transformations;
            presolver->linear_propagation_bound_cursor =
                presolver->transformations.n_bound_changes;
            if (!state->retain_deferred_frontier)
                prefos_internal_free_linear_propagation_cache(
                    presolver);
            return PREFOS_STATUS_OK;
        }
        if (presolver->settings.linear_propagation_max_stale_rounds > 0 &&
            state->dirty_rows.current_count > 0 &&
            !getenv("PREFOS_DISABLE_LINEAR_ROW_PRUNING"))
            profile_rows_pruned +=
                prune_nonpropagating_round_rows(presolver, state);
        if (has_pending_fixed_box_candidate(
                presolver, column_workspace))
            break;
        /*
         * A finite-to-finite-only wave cannot make another row's activity
         * finite. In the default budgeted mode, defer those revisits to a
         * later structural pass instead of chasing marginal improvements to
         * a full fixed point. Strict mode disables stale-round budgeting and
         * therefore retains the exhaustive behavior.
         */
        if (presolver->settings.linear_propagation_max_stale_rounds > 0 &&
            state->round_new_finite_bounds == 0 &&
            !getenv("PREFOS_ALLOW_PRODUCTIVE_FINITE_WAVES"))
        {
            ++presolver->stats.linear_stale_stops;
            break;
        }
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
            state->total_work_limit = renew_productive_round_work_budget(
                presolver, state->total_work_limit,
                state->work_budget_quantum, changes, work);
            work_limit = state->total_work_limit;
        }
    }
    if (state->integrate_redundancy &&
        state->dirty_rows.current_count == 0)
        presolver->scalar_redundancy_completed = 1;
    if (state->dirty_rows.current_count == 0 &&
        state->dirty_rows.next_count == 0)
        presolver->linear_propagation_complete = 1;
    if (trace_linear_profile())
    {
        prefos_internal_timer_now(&profile_stop);
        fprintf(
            stderr,
            "PreFOS linear-profile prepare=%.6f process=%.6f "
            "rows=%.6f confirm=%.6f kernel=%.6f row_calls=%zu "
            "rescans=%zu rescan_terms=%zu zero_checks=%zu "
            "zero_terms=%zu outward_scans=%zu outward_terms=%zu "
            "activity_recomputes=%zu activity_recompute_terms=%zu "
            "single_infinite_rows=%zu candidates=%zu changes=%zu "
            "pruned=%zu round=%d "
            "remaining=%zu next=%zu\n",
            prefos_internal_timer_elapsed_milliseconds(
                &profile_start, &profile_ready),
            prefos_internal_timer_elapsed_milliseconds(
                &profile_ready, &profile_stop),
            profile_row_milliseconds,
            state->profile_confirmation_milliseconds,
            state->profile_kernel_milliseconds, profile_row_calls,
            state->profile_residual_rescans,
            state->profile_residual_rescan_terms,
            state->profile_exact_zero_checks,
            state->profile_exact_zero_terms,
            state->profile_outward_activity_scans,
            state->profile_outward_activity_terms,
            state->profile_activity_recomputes,
            state->profile_activity_recompute_terms,
            state->profile_single_infinite_candidate_rows,
            state->profile_bound_candidates,
            state->profile_bound_changes,
            profile_rows_pruned, round,
            state->dirty_rows.current_count,
            state->dirty_rows.next_count);
    }
    return PREFOS_STATUS_OK;
}

static PreFOSStatus mark_cone_activity_candidate_rows(
    const PreFOSPresolver *presolver,
    const PreFOSColumnWorkspace *column_workspace,
    const int *column_to_cone,
    unsigned char *candidate_rows)
{
    const PreFOSProblemData *problem = &presolver->original;
    size_t cone_index;
    if (column_workspace)
    {
        for (cone_index = 0; cone_index < problem->n_cones; ++cone_index)
        {
            const PreFOSConeBlock *cone = &problem->cones[cone_index];
            size_t coordinate;
            if (presolver->converted_affine_cones[cone_index]) continue;
            for (coordinate = 0; coordinate < cone->dimension; ++coordinate)
            {
                int column = cone->indices[coordinate];
                int position;
                for (position = column_workspace->starts[column];
                     position < column_workspace->ends[column]; ++position)
                {
                    int row = column_workspace->rows[position];
                    if (!presolver->remove_rows[row] &&
                        column_workspace->values[position] != 0.0 &&
                        prefos_internal_term_is_active_in_row(
                            presolver, (size_t) row, column))
                        candidate_rows[row] = 1;
                }
            }
        }
        return PREFOS_STATUS_OK;
    }
    for (size_t row = 0; row < problem->A.rows; ++row)
    {
        int position;
        if (presolver->remove_rows[row]) continue;
        for (position = problem->A.row_pointers[row];
             position < problem->A.row_pointers[row + 1]; ++position)
        {
            int column = problem->A.column_indices[position];
            if (problem->A.values[position] != 0.0 &&
                column_to_cone[column] >= 0 &&
                prefos_internal_term_is_active_in_row(
                    presolver, row, column))
            {
                candidate_rows[row] = 1;
                break;
            }
        }
    }
    return PREFOS_STATUS_OK;
}

PreFOSStatus prefos_internal_remove_redundant_rows_by_activity(
    PreFOSPresolver *presolver,
    PreFOSColumnWorkspace *column_workspace)
{
    const PreFOSProblemData *problem = &presolver->original;
    PreFOSConeActivityWorkspace workspace;
    double *retained_lower = NULL, *retained_upper = NULL;
    const double *activity_lower = presolver->propagation_lower;
    const double *activity_upper = presolver->propagation_upper;
    unsigned char *gpu_row_flags = NULL;
    unsigned char *cone_candidate_rows = NULL;
    PreFOSStatus result;
    size_t row, box_position;
    int gpu_activity_valid = 0;
    int over_cpu_budget;
    if (!presolver->settings.remove_redundant_rows) return PREFOS_STATUS_OK;
    if (presolver->scalar_redundancy_completed &&
        !(problem->n_cones > 0 &&
          presolver->settings.cone_aware_row_activity))
        return PREFOS_STATUS_OK;
    over_cpu_budget =
        presolver->settings.redundant_row_max_average_nnz > 0.0 &&
        (long double) problem->A.nnz >
            (long double) problem->A.rows *
                (long double)
                    presolver->settings.redundant_row_max_average_nnz;
    if (over_cpu_budget && !presolver->settings.structural_reductions_gpu)
    {
        ++presolver->stats.redundant_row_activity_budget_skips;
        return PREFOS_STATUS_OK;
    }
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
        if (column_workspace ||
            presolver->scalar_redundancy_completed)
        {
            cone_candidate_rows = (unsigned char *) calloc(
                problem->A.rows, sizeof(unsigned char));
            if (problem->A.rows > 0 && !cone_candidate_rows)
            {
                result = PREFOS_STATUS_OUT_OF_MEMORY;
                goto cleanup;
            }
            result = mark_cone_activity_candidate_rows(
                presolver, column_workspace, workspace.column_to_cone,
                cone_candidate_rows);
            if (result != PREFOS_STATUS_OK) goto cleanup;
        }
    }
    if (presolver->settings.structural_reductions_gpu)
    {
        PreFOSCudaPropagationStatus cuda_status;
        PreFOSCudaWorkspace *cuda_workspace;
        double gpu_milliseconds = 0.0;
        gpu_row_flags = (unsigned char *) prefos_internal_alloc_array(
            problem->A.rows, sizeof(unsigned char));
        if (problem->A.rows > 0 && !gpu_row_flags)
        {
            result = PREFOS_STATUS_OUT_OF_MEMORY;
            goto cleanup;
        }
        cuda_workspace =
            prefos_internal_cuda_workspace_get(presolver, &cuda_status);
        if (cuda_workspace && cuda_status == PREFOS_CUDA_PROPAGATION_OK)
        {
            cuda_status = prefos_cuda_cone_activity_candidates(
                cuda_workspace, activity_lower, activity_upper,
                presolver->working_constraint_lower,
                presolver->working_constraint_upper, presolver->remove_rows,
                presolver->settings.feasibility_tolerance, gpu_row_flags,
                &gpu_milliseconds);
        }
        presolver->stats.cone_activity_gpu_milliseconds += gpu_milliseconds;
        if (cuda_status == PREFOS_CUDA_PROPAGATION_OK)
        {
            gpu_activity_valid = 1;
            ++presolver->stats.cone_activity_gpu_passes;
            for (row = 0; row < problem->A.rows; ++row)
                if (gpu_row_flags[row] != 0)
                    ++presolver->stats.cone_activity_gpu_candidates;
        }
        else
        {
            ++presolver->stats.cone_activity_gpu_fallbacks;
            if (over_cpu_budget)
            {
                ++presolver->stats.redundant_row_activity_budget_skips;
                goto cleanup;
            }
        }
    }

    for (row = 0; row < problem->A.rows; ++row)
    {
        PreFOSRowActivity activity, exact_activity;
        int lower_implied, upper_implied;
        int cone_support_strengthened = 0;
        int use_cone_activity;
        PresolveLinearRowState row_state;
        PreFOSStatus status;
        if (presolver->remove_rows[row]) continue;
        if (presolver->nonmaterialized_bound_source_rows &&
            presolver->nonmaterialized_bound_source_rows[row])
            continue;
        if (!prefos_internal_row_has_exact_linear_form(
                presolver, row))
            continue;
        if (presolver->scalar_redundancy_completed &&
            cone_candidate_rows && !cone_candidate_rows[row])
            continue;
        if (gpu_activity_valid && gpu_row_flags[row] == 0) continue;
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
                prefos_internal_mark_removed_row(presolver, row);
                ++presolver->stats.removed_redundant_rows;
            }
            continue;
        }
        use_cone_activity =
            problem->n_cones > 0 &&
            presolver->settings.cone_aware_row_activity &&
            (!cone_candidate_rows || cone_candidate_rows[row]);
        if (use_cone_activity)
        {
            status = prefos_internal_compute_cone_aware_row_activity(
                presolver, row, 0, 1, &workspace, &activity);
            if (status == PREFOS_STATUS_OK &&
                activity_needs_rigorous_confirmation(
                    presolver, row, &activity))
                status = prefos_internal_compute_cone_aware_row_activity(
                    presolver, row, 1, 0, &workspace, &activity);
        }
        else
            status = compute_row_activities_with_bounds(
                presolver, row, activity_lower, activity_upper,
                &exact_activity, &activity);
        if (use_cone_activity)
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
        {
            if (traced_linear_row() == (int) row)
                fprintf(
                    stderr,
                    "PreFOS activity-row row=%zu lower=%.17g upper=%.17g "
                    "min=%.17g max=%.17g inf_min=%zu inf_max=%zu nnz=%zu "
                    "cone=%d strengthened=%d\n",
                    row, presolver->working_constraint_lower[row],
                    presolver->working_constraint_upper[row],
                    activity.finite_min, activity.finite_max,
                    activity.n_infinite_min, activity.n_infinite_max,
                    activity.n_nonzeros, use_cone_activity,
                    cone_support_strengthened);
        }

        row_state = presolve_internal_classify_linear_row(
            &activity, presolver->working_constraint_lower[row],
            presolver->working_constraint_upper[row],
            presolver->settings.feasibility_tolerance,
            presolver->settings.feasibility_tolerance);
        lower_implied = (row_state & PRESOLVE_ROW_LOWER_REDUNDANT) != 0;
        upper_implied = (row_state & PRESOLVE_ROW_UPPER_REDUNDANT) != 0;
        if (lower_implied && upper_implied)
        {
            prefos_internal_mark_removed_row(presolver, row);
            ++presolver->stats.removed_redundant_rows;
            if (cone_support_strengthened)
                ++presolver->stats.cone_activity_rows_removed;
        }
        else if (lower_implied &&
                 isfinite(presolver->working_constraint_lower[row]))
        {
            prefos_internal_note_constraint_bound_change(presolver);
            presolver->working_constraint_lower[row] = -INFINITY;
            prefos_internal_queue_row_side_change(
                presolver, column_workspace, row);
            ++presolver->stats.removed_redundant_row_lower_sides;
        }
        else if (upper_implied &&
                 isfinite(presolver->working_constraint_upper[row]))
        {
            prefos_internal_note_constraint_bound_change(presolver);
            presolver->working_constraint_upper[row] = INFINITY;
            prefos_internal_queue_row_side_change(
                presolver, column_workspace, row);
            ++presolver->stats.removed_redundant_row_upper_sides;
        }
    }
cleanup:
    prefos_internal_cone_activity_workspace_free(&workspace);
    free(gpu_row_flags);
    free(cone_candidate_rows);
    free(retained_lower);
    free(retained_upper);
    return result;
}
