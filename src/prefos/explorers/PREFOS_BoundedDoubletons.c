/*
 * Copyright 2026 Hongpei Li
 * SPDX-License-Identifier: Apache-2.0
 */

#include "PREFOS_ColumnReductionInternal.h"
#include "PREFOS_MaterializedDoubletons.h"
#include "core/PREFOS_Timer.h"

#include <stdio.h>
#include <stdlib.h>

typedef struct
{
    int *rows;
    unsigned char *queued;
    size_t capacity;
    size_t head;
    size_t tail;
    size_t count;
} PreFOSBoundedDoubletonQueue;

static int bounded_doubleton_queue_init(
    PreFOSBoundedDoubletonQueue *queue, size_t rows)
{
    memset(queue, 0, sizeof(*queue));
    if (rows == 0) return 1;
    queue->rows = (int *) prefos_internal_alloc_array(
        rows, sizeof(int));
    queue->queued = (unsigned char *) calloc(
        rows, sizeof(unsigned char));
    if (!queue->rows || !queue->queued)
    {
        free(queue->rows);
        free(queue->queued);
        memset(queue, 0, sizeof(*queue));
        return 0;
    }
    queue->capacity = rows;
    return 1;
}

static void bounded_doubleton_queue_free(
    PreFOSBoundedDoubletonQueue *queue)
{
    free(queue->rows);
    free(queue->queued);
    memset(queue, 0, sizeof(*queue));
}

static int bounded_doubleton_queue_schedule(
    PreFOSBoundedDoubletonQueue *queue, int row)
{
    if (row < 0 || (size_t) row >= queue->capacity ||
        queue->queued[row])
        return 0;
    queue->rows[queue->tail] = row;
    queue->tail = (queue->tail + 1) % queue->capacity;
    ++queue->count;
    queue->queued[row] = 1;
    return 1;
}

static int bounded_doubleton_queue_pop(
    PreFOSBoundedDoubletonQueue *queue, int *row)
{
    if (queue->count == 0) return 0;
    *row = queue->rows[queue->head];
    queue->head = (queue->head + 1) % queue->capacity;
    --queue->count;
    queue->queued[*row] = 0;
    return 1;
}

static int exact_integral_ratio(double numerator, double denominator)
{
    double ratio;
    if (denominator == 0.0) return 0;
    ratio = fabs(numerator / denominator);
    return isfinite(ratio) && ratio <= (double) INT_MAX &&
           ratio == (double) (int) ratio;
}

static int bounded_doubleton_column_is_eligible(
    const PreFOSPresolver *presolver,
    const PreFOSColumnWorkspace *workspace, int column)
{
    if (prefos_internal_column_is_linear_box(
            presolver, workspace, column))
        return 1;
    return column >= 0 &&
           (size_t) column < presolver->original.n &&
           (workspace->bounded_doubleton_chain_target[column] ||
            presolver->substitution_fill_in_targets[column]) &&
           presolver->variable_to_box[column] >= 0 &&
           !presolver->is_fixed[column] &&
           !presolver->is_substituted[column] &&
           !presolver->is_parallel_removed[column] &&
           !presolver->affine_protected_columns[column] &&
           !workspace->quadratic[column] &&
           !workspace->factor[column];
}

static void trace_bounded_doubleton_failure(
    size_t row, int pivot_column, int target_column,
    const char *stage, PreFOSStatus status)
{
    const char *trace = getenv("PREFOS_TRACE_BOUNDED_DOUBLETON");
    if (!trace || !*trace || *trace == '0') return;
    fprintf(
        stderr,
        "PreFOS bounded-doubleton failure row=%zu pivot=%d target=%d "
        "stage=%s status=%d\n",
        row, pivot_column, target_column, stage, (int) status);
}

static int row_is_bounded_doubleton_candidate(
    const PreFOSPresolver *presolver,
    const PreFOSColumnWorkspace *workspace, size_t row)
{
    double lower, upper;
    if (presolver->remove_rows[row] || workspace->dirty_row[row] ||
        workspace->row_degrees[row] != 2)
        return 0;
    lower = presolver->working_constraint_lower[row];
    upper = presolver->working_constraint_upper[row];
    return isfinite(lower) && lower == upper;
}

PreFOSStatus prefos_internal_reduce_bounded_doubletons(
    PreFOSPresolver *presolver, PreFOSColumnWorkspace *workspace)
{
    const PreFOSProblemData *problem = &presolver->original;
    PreFOSBoundedDoubletonQueue queue;
    PreFOSStatus result = PREFOS_STATUS_OK;
    size_t equality_rows = 0, doubleton_rows = 0, dirty_rows = 0;
    size_t ineligible_candidates = 0, protected_candidates = 0;
    size_t depth_candidates = 0, degree_candidates = 0;
    size_t stale_csc_candidates = 0, accepted = 0;
    size_t candidate_visits = 0, requeued = 0, row;
    size_t profiled_pivot_degree = 0, profiled_affected_row_terms = 0;
    double profiled_append_milliseconds = 0.0;
    double profiled_update_milliseconds = 0.0;
    double profiled_degree_milliseconds = 0.0;
    int profile =
        getenv("PREFOS_TRACE_BOUNDED_PROFILE") != NULL;
    int queued_row;
    if (!presolver->settings.bounded_doubleton_substitution)
        return PREFOS_STATUS_OK;
    if (!bounded_doubleton_queue_init(
            &queue, problem->A.rows))
        return PREFOS_STATUS_OUT_OF_MEMORY;
    (void) prefos_internal_queue_transformation_events(
        presolver, workspace);
    for (row = 0; row < problem->A.rows; ++row)
    {
        double lower, upper;
        if (presolver->remove_rows[row]) continue;
        lower = presolver->working_constraint_lower[row];
        upper = presolver->working_constraint_upper[row];
        if (!isfinite(lower) || lower != upper) continue;
        ++equality_rows;
        if (workspace->dirty_row[row])
        {
            ++dirty_rows;
            continue;
        }
        if (workspace->row_degrees[row] != 2) continue;
        ++doubleton_rows;
        (void) bounded_doubleton_queue_schedule(
            &queue, (int) row);
    }
    while (bounded_doubleton_queue_pop(&queue, &queued_row))
    {
        row = (size_t) queued_row;
        ++candidate_visits;
        {
            int columns[2], pivot_column = -1, target_column = -1;
            double coefficients[2], lower, upper, pivot, target_coefficient;
            double pivot_lower, pivot_upper, target_lower, target_upper;
            double alpha, beta;
            double mapped_lower = -INFINITY, mapped_upper = INFINITY;
            double fixed_shift = 0.0;
            double scales[1];
            int targets[1];
            size_t live;
            int candidate, candidate_order[2], position;
            if (!row_is_bounded_doubleton_candidate(
                    presolver, workspace, row))
                continue;
            lower = presolver->working_constraint_lower[row];
            upper = presolver->working_constraint_upper[row];
            live = 0;
            for (position = problem->A.row_pointers[row];
                 position < problem->A.row_pointers[row + 1]; ++position)
            {
                int column = problem->A.column_indices[position];
                double coefficient = problem->A.values[position];
                if (coefficient == 0.0) continue;
                if (presolver->is_fixed[column])
                {
                    if (!prefos_internal_safe_add_product(
                            &fixed_shift, coefficient,
                            presolver->fixed_values[column]))
                    {
                        trace_bounded_doubleton_failure(
                            row, -1, -1, "fixed-shift",
                            PREFOS_STATUS_NUMERICAL_ERROR);
                        result = PREFOS_STATUS_NUMERICAL_ERROR;
                        goto cleanup;
                    }
                    continue;
                }
                if (presolver->is_substituted[column] ||
                    presolver->is_parallel_removed[column])
                    continue;
                if (live < 2)
                {
                    columns[live] = column;
                    coefficients[live] = coefficient;
                }
                ++live;
            }
            lower -= fixed_shift;
            upper -= fixed_shift;
            if (isnan(lower) || isnan(upper))
            {
                trace_bounded_doubleton_failure(
                    row, -1, -1, "shifted-side",
                    PREFOS_STATUS_NUMERICAL_ERROR);
                result = PREFOS_STATUS_NUMERICAL_ERROR;
                goto cleanup;
            }
            if (live != 2) continue;
            candidate_order[0] = 0;
            candidate_order[1] = 1;
            {
                int first_degree = workspace->live_degrees[columns[0]];
                int second_degree = workspace->live_degrees[columns[1]];
                int first_integral = exact_integral_ratio(
                    coefficients[1], coefficients[0]);
                int second_integral = exact_integral_ratio(
                    coefficients[0], coefficients[1]);
                if (first_degree != 1 && second_degree == 1)
                {
                    candidate_order[0] = 1;
                    candidate_order[1] = 0;
                }
                else if ((first_degree == 1) == (second_degree == 1) &&
                         !first_integral && second_integral)
                {
                    candidate_order[0] = 1;
                    candidate_order[1] = 0;
                }
                else if ((first_degree == 1) == (second_degree == 1) &&
                         first_integral == second_integral &&
                         second_degree <= first_degree)
                {
                    candidate_order[0] = 1;
                    candidate_order[1] = 0;
                }
            }
            for (position = 0; position < 2; ++position)
            {
                int possible_pivot, possible_target;
                candidate = candidate_order[position];
                possible_pivot = columns[candidate];
                possible_target = columns[1 - candidate];
                if (presolver->working_matrix_is_materialized &&
                    workspace->csc_column_dirty[possible_pivot])
                {
                    ++stale_csc_candidates;
                    continue;
                }
                if (!bounded_doubleton_column_is_eligible(
                        presolver, workspace, possible_pivot) ||
                    !bounded_doubleton_column_is_eligible(
                        presolver, workspace, possible_target))
                {
                    ++ineligible_candidates;
                    continue;
                }
                if (workspace->protected_target[possible_pivot] ||
                    workspace->protected_target[possible_target])
                {
                    ++protected_candidates;
                    continue;
                }
                if (presolver->substitution_incoming_depth[possible_pivot] >=
                    PREFOS_MAX_SUBSTITUTION_DEPTH)
                {
                    ++depth_candidates;
                    continue;
                }
                if (workspace->live_degrees[possible_pivot] >
                    presolver->settings
                        .max_bounded_doubleton_column_degree)
                {
                    ++degree_candidates;
                    continue;
                }
                pivot_column = possible_pivot;
                target_column = possible_target;
                pivot = coefficients[candidate];
                target_coefficient = coefficients[1 - candidate];
                break;
            }
            if (pivot_column < 0 || pivot == 0.0) continue;
            pivot_lower = presolver->working_box_lower
                [presolver->variable_to_box[pivot_column]];
            pivot_upper = presolver->working_box_upper
                [presolver->variable_to_box[pivot_column]];
            target_lower = presolver->working_box_lower
                [presolver->variable_to_box[target_column]];
            target_upper = presolver->working_box_upper
                [presolver->variable_to_box[target_column]];
            alpha = -target_coefficient / pivot;
            beta = lower / pivot;
            if (alpha == 0.0 || !isfinite(alpha) || !isfinite(beta))
                continue;
            if (isfinite(pivot_lower))
            {
                double mapped = (pivot_lower - beta) / alpha;
                if (alpha > 0.0)
                    mapped_lower =
                        prefos_internal_outward_bound_cast(mapped, 1);
                else
                    mapped_upper =
                        prefos_internal_outward_bound_cast(mapped, 0);
            }
            if (isfinite(pivot_upper))
            {
                double mapped = (pivot_upper - beta) / alpha;
                if (alpha > 0.0)
                    mapped_upper =
                        prefos_internal_outward_bound_cast(mapped, 0);
                else
                    mapped_lower =
                        prefos_internal_outward_bound_cast(mapped, 1);
            }
            target_lower = fmax(target_lower, mapped_lower);
            target_upper = fmin(target_upper, mapped_upper);
            if (target_lower > target_upper +
                                   presolver->settings.feasibility_tolerance)
            {
                if (getenv("PREFOS_TRACE_BOUNDED_DOUBLETON"))
                    fprintf(
                        stderr,
                        "PreFOS bounded-doubleton infeasible row=%zu "
                        "pivot=%d target=%d "
                        "pivot_bounds=[%.17g,%.17g] "
                        "old_target=[%.17g,%.17g] "
                        "mapped=[%.17g,%.17g] "
                        "new_target=[%.17g,%.17g] "
                        "alpha=%.17g beta=%.17g\n",
                        row, pivot_column, target_column, pivot_lower,
                        pivot_upper,
                        presolver->working_box_lower[
                            presolver->variable_to_box[target_column]],
                        presolver->working_box_upper[
                            presolver->variable_to_box[target_column]],
                        mapped_lower, mapped_upper, target_lower,
                        target_upper, alpha, beta);
                if (getenv("PREFOS_TRACE_BOUNDED_DOUBLETON"))
                {
                    int p;
                    fprintf(
                        stderr,
                        "PreFOS bounded-doubleton infeasible row_state "
                        "excluded=%d terms:",
                        presolver->residual_source_column[row]);
                    for (p = problem->A.row_pointers[row];
                         p < problem->A.row_pointers[row + 1]; ++p)
                        fprintf(
                            stderr,
                            " (%d,%.17g,sub=%d,src=%d,const=%.17g)",
                            problem->A.column_indices[p],
                            problem->A.values[p],
                            presolver->is_substituted[
                                problem->A.column_indices[p]],
                            presolver->substitution_source_row[
                                problem->A.column_indices[p]],
                            presolver->substitution_constant[
                                problem->A.column_indices[p]]);
                    fprintf(stderr, "\n");
                }
                result = PREFOS_STATUS_PRIMAL_INFEASIBLE;
                goto cleanup;
            }
            if (target_lower > target_upper)
            {
                double midpoint =
                    prefos_internal_safe_midpoint(
                        target_lower, target_upper);
                target_lower = midpoint;
                target_upper = midpoint;
            }
            targets[0] = target_column;
            scales[0] = alpha;
            {
                int eager_materialized =
                    prefos_internal_can_update_materialized_doubleton(
                        presolver, workspace, (int) row,
                        pivot_column, target_column, alpha, beta);
                size_t event_count_before =
                    presolver->transformations.n_events;
                PreFOSTimestamp profile_start, profile_stop;
                if (profile)
                {
                    int degree =
                        workspace->live_degrees[pivot_column];
                    if (degree > 0)
                        profiled_pivot_degree += (size_t) degree;
                    prefos_internal_timer_now(&profile_start);
                }
                PreFOSStatus status =
                    prefos_internal_append_column_substitution(
                        presolver, pivot_column, targets, scales, 1,
                        (int) row, beta, pivot, workspace,
                        PREFOS_SUBSTITUTION_BOUNDED_DOUBLETON,
                        eager_materialized, 0);
                if (profile)
                {
                    prefos_internal_timer_now(&profile_stop);
                    profiled_append_milliseconds +=
                        prefos_internal_timer_elapsed_milliseconds(
                            &profile_start, &profile_stop);
                }
                if (status != PREFOS_STATUS_OK)
                {
                    trace_bounded_doubleton_failure(
                        row, pivot_column, target_column,
                        "append-substitution", status);
                    result = status;
                    goto cleanup;
                }
                if (eager_materialized)
                {
                    int csc_position;
                    if (profile)
                        prefos_internal_timer_now(&profile_start);
                    status =
                        prefos_internal_update_materialized_doubleton(
                            presolver, workspace, (int) row,
                            pivot_column, target_column, alpha, beta,
                            event_count_before);
                    if (status != PREFOS_STATUS_OK)
                    {
                        trace_bounded_doubleton_failure(
                            row, pivot_column, target_column,
                            "materialized-update", status);
                        result = status;
                        goto cleanup;
                    }
                    if (profile)
                    {
                        prefos_internal_timer_now(&profile_stop);
                        profiled_update_milliseconds +=
                            prefos_internal_timer_elapsed_milliseconds(
                                &profile_start, &profile_stop);
                    }
                    for (csc_position = workspace->starts[pivot_column];
                         csc_position < workspace->ends[pivot_column];
                         ++csc_position)
                    {
                        int affected_row =
                            workspace->rows[csc_position];
                        if (profile && affected_row != (int) row &&
                            !presolver->remove_rows[affected_row])
                            profiled_affected_row_terms +=
                                (size_t)
                                    (problem->A.row_pointers[
                                         affected_row + 1] -
                                     problem->A.row_pointers[
                                         affected_row]);
                        if (row_is_bounded_doubleton_candidate(
                                presolver, workspace,
                                (size_t) affected_row) &&
                            bounded_doubleton_queue_schedule(
                                &queue, affected_row))
                            ++requeued;
                    }
                }
                else
                    (void) prefos_internal_queue_transformation_events(
                        presolver, workspace);
                workspace->bounded_doubleton_chain_target[
                    target_column] = 1;
                if (profile)
                    prefos_internal_timer_now(&profile_start);
                prefos_internal_update_column_live_degrees(
                    presolver, workspace);
                if (profile)
                {
                    prefos_internal_timer_now(&profile_stop);
                    profiled_degree_milliseconds +=
                        prefos_internal_timer_elapsed_milliseconds(
                            &profile_start, &profile_stop);
                }
                ++accepted;
                if (getenv("PREFOS_TRACE_BOUNDED_DOUBLETON"))
                    fprintf(
                        stderr,
                        "PreFOS bounded-doubleton accept row=%zu "
                        "pivot=%d target=%d beta=%.17g alpha=%.17g "
                        "eager=%d visit=%zu\n",
                        row, pivot_column, target_column, beta, alpha,
                        eager_materialized, candidate_visits);
            }
            {
                int target_box =
                    presolver->variable_to_box[target_column];
                prefos_internal_linear_cache_mark_bound_dirty(
                    presolver, target_column);
                prefos_internal_mark_fixed_box_dirty(
                    presolver, target_column);
                prefos_internal_clear_box_bound_provenance(
                    presolver, target_column, 1);
                prefos_internal_clear_box_bound_provenance(
                    presolver, target_column, 0);
                presolver->working_box_lower[target_box] = target_lower;
                presolver->working_box_upper[target_box] = target_upper;
                presolver->propagation_lower[target_column] =
                    target_lower;
                presolver->propagation_upper[target_column] =
                    target_upper;
            }
        }
    }
cleanup:
    if (getenv("PREFOS_TRACE_BOUNDED_DOUBLETON"))
        fprintf(
            stderr,
            "PreFOS bounded-doubleton materialized=%d equalities=%zu "
            "doubletons=%zu dirty=%zu ineligible=%zu protected=%zu "
            "depth=%zu degree=%zu stale_csc=%zu accepted=%zu "
            "visits=%zu requeued=%zu\n",
            presolver->working_matrix_is_materialized, equality_rows,
            doubleton_rows, dirty_rows, ineligible_candidates,
            protected_candidates, depth_candidates, degree_candidates,
            stale_csc_candidates, accepted, candidate_visits,
            requeued);
    bounded_doubleton_queue_free(&queue);
    if (profile)
        fprintf(
            stderr,
            "PreFOS bounded-profile accepted=%zu pivot_degree=%zu "
            "affected_row_terms=%zu append_ms=%.6f update_ms=%.6f "
            "degree_ms=%.6f\n",
            accepted, profiled_pivot_degree,
            profiled_affected_row_terms,
            profiled_append_milliseconds,
            profiled_update_milliseconds,
            profiled_degree_milliseconds);
    return result;
}
