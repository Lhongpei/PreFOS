/*
 * Copyright 2026 Hongpei Li
 * SPDX-License-Identifier: Apache-2.0
 */

#include "PREFOS_TrivialReductions.h"
#include "PREFOS_ColumnReductionInternal.h"
#include "common/PreFOSThread.h"

#include <stdio.h>
#include <stdlib.h>

#define PREFOS_TRIVIAL_CLASSIFICATION_THREADS 4
#define PREFOS_TRIVIAL_CLASSIFICATION_THRESHOLD 262144U
#define PREFOS_TRIVIAL_CLASSIFICATION_SAMPLE 2048U

static void trace_trivial_row_failure(
    size_t row, int column, const char *stage, PreFOSStatus status)
{
    const char *trace = getenv("PREFOS_TRACE_TRIVIAL_ROW");
    if (!trace || !*trace || *trace == '0') return;
    fprintf(
        stderr,
        "PreFOS trivial-row failure row=%zu column=%d stage=%s status=%d\n",
        row, column, stage, (int) status);
}

static int close_bounds(double lower, double upper, double tolerance)
{
    long double difference, scale;
    double quick_difference, quick_scale, quick_threshold;
    if (!isfinite(lower) || !isfinite(upper)) return 0;
    quick_difference = upper - lower;
    quick_scale = fmax(1.0, fmax(fabs(lower), fabs(upper)));
    quick_threshold = tolerance * quick_scale;
    if (isfinite(quick_difference) &&
        quick_difference > 2.0 * quick_threshold)
        return 0;
    difference = (long double) upper - (long double) lower;
    scale =
        fmaxl(1.0L, fmaxl(fabsl((long double) lower), fabsl((long double) upper)));
    return difference <= (long double) tolerance * scale;
}

static int close_linear_box_bounds(
    const PreFOSPresolver *presolver,
    const PreFOSColumnWorkspace *workspace, int column,
    double lower, double upper)
{
    long double difference;
    double column_scale;
    if (close_bounds(
            lower, upper,
            presolver->settings.fixed_variable_tolerance))
        return 1;
    if (!workspace ||
        presolver->settings.feasibility_tolerance <= 0.0 ||
        !prefos_internal_column_is_linear_box(
            presolver, workspace, column) ||
        !isfinite(lower) || !isfinite(upper))
        return 0;
    difference = (long double) upper - (long double) lower;
    if (difference < 0.0L) return 0;
    column_scale = prefos_internal_column_max_abs_coefficient(
        presolver, workspace, column);
    return column_scale > 0.0 &&
           difference * (long double) column_scale <=
               (long double)
                   presolver->settings.feasibility_tolerance;
}

static int singleton_row_is_implied_by_box(
    double row_lower, double row_upper, double coefficient,
    double box_lower, double box_upper)
{
    double minimum_bound =
        coefficient > 0.0 ? box_lower : box_upper;
    double maximum_bound =
        coefficient > 0.0 ? box_upper : box_lower;
    long double minimum_activity =
        (long double) coefficient * (long double) minimum_bound;
    long double maximum_activity =
        (long double) coefficient * (long double) maximum_bound;

    if (isnan(minimum_activity) || isnan(maximum_activity))
        return 0;
    return minimum_activity >= (long double) row_lower &&
           maximum_activity <= (long double) row_upper;
}

static int trivial_row_is_strictly_implied_by_box(
    const PreFOSPresolver *presolver, size_t row)
{
    const PreFOSCsrMatrix *matrix = &presolver->original.A;
    long double fixed_shift = 0.0L;
    size_t live = 0;
    int live_column = -1;
    double live_coefficient = 0.0;
    int position;

    if (row >= matrix->rows || presolver->remove_rows[row])
        return 0;
    for (position = matrix->row_pointers[row];
         position < matrix->row_pointers[row + 1]; ++position)
    {
        int column = matrix->column_indices[position];
        double coefficient = matrix->values[position];
        if (coefficient == 0.0) continue;
        if (presolver->is_fixed[column])
        {
            long double term =
                (long double) coefficient *
                (long double) presolver->fixed_values[column];
            if (!isfinite(term) ||
                !isfinite(fixed_shift + term))
                return 0;
            fixed_shift += term;
            continue;
        }
        if (presolver->is_substituted[column])
        {
            if (!prefos_internal_term_is_active_in_row(
                    presolver, row, column))
                continue;
            return 0;
        }
        if (presolver->is_parallel_removed[column])
            continue;
        if (++live > 1) return 0;
        live_column = column;
        live_coefficient = coefficient;
    }
    if (live != 1 || live_coefficient == 0.0)
        return 0;
    {
        int box_position =
            presolver->variable_to_box[live_column];
        double lower, upper;
        if (box_position < 0) return 0;
        lower = (double)
            ((long double)
                 presolver->working_constraint_lower[row] -
             fixed_shift);
        upper = (double)
            ((long double)
                 presolver->working_constraint_upper[row] -
             fixed_shift);
        if (isnan(lower) || isnan(upper)) return 0;
        return singleton_row_is_implied_by_box(
            lower, upper, live_coefficient,
            presolver->working_box_lower[box_position],
            presolver->working_box_upper[box_position]);
    }
}

typedef struct
{
    const PreFOSPresolver *presolver;
    const int *rows;
    unsigned char *implied;
    size_t begin;
    size_t end;
} PreFOSTrivialClassificationTask;

static void *classify_trivial_rows(void *argument)
{
    PreFOSTrivialClassificationTask *task =
        (PreFOSTrivialClassificationTask *) argument;
    size_t position;
    for (position = task->begin; position < task->end;
         ++position)
        task->implied[position] = (unsigned char)
            trivial_row_is_strictly_implied_by_box(
                task->presolver,
                (size_t) task->rows[position]);
    return NULL;
}

static int parallel_trivial_classification_is_promising(
    const PreFOSPresolver *presolver, const int *rows,
    size_t count)
{
    size_t sample_count =
        count < PREFOS_TRIVIAL_CLASSIFICATION_SAMPLE
            ? count
            : PREFOS_TRIVIAL_CLASSIFICATION_SAMPLE;
    size_t implied = 0, sample;
    if (sample_count == 0) return 0;
    for (sample = 0; sample < sample_count; ++sample)
    {
        size_t position =
            sample * (count / sample_count) +
            sample * (count % sample_count) / sample_count;
        implied += (size_t)
            trivial_row_is_strictly_implied_by_box(
                presolver, (size_t) rows[position]);
    }
    return implied * 4U >= sample_count * 3U;
}

static unsigned char *classify_trivial_rows_in_parallel(
    const PreFOSPresolver *presolver, const int *rows,
    size_t count)
{
    PreFOSTrivialClassificationTask
        tasks[PREFOS_TRIVIAL_CLASSIFICATION_THREADS];
    PreFOSThread
        threads[PREFOS_TRIVIAL_CLASSIFICATION_THREADS - 1];
    unsigned char
        started[PREFOS_TRIVIAL_CLASSIFICATION_THREADS - 1] = {0};
    unsigned char *implied;
    int n_threads = prefos_cpu_thread_limit(
        PREFOS_TRIVIAL_CLASSIFICATION_THREADS);
    int thread;

    if (n_threads <= 1 ||
        count < PREFOS_TRIVIAL_CLASSIFICATION_THRESHOLD ||
        !parallel_trivial_classification_is_promising(
            presolver, rows, count))
        return NULL;
    implied = (unsigned char *)
        prefos_internal_alloc_array(
            count, sizeof(unsigned char));
    if (!implied) return NULL;
    for (thread = 0; thread < n_threads; ++thread)
    {
        tasks[thread] = (PreFOSTrivialClassificationTask){
            presolver, rows, implied,
            count * (size_t) thread / (size_t) n_threads,
            count * (size_t) (thread + 1) /
                (size_t) n_threads};
    }
    for (thread = 1; thread < n_threads; ++thread)
        if (prefos_thread_create(
                &threads[thread - 1],
                classify_trivial_rows,
                &tasks[thread]) == 0)
            started[thread - 1] = 1;
        else
            classify_trivial_rows(&tasks[thread]);
    classify_trivial_rows(&tasks[0]);
    for (thread = 1; thread < n_threads; ++thread)
        if (started[thread - 1])
            (void) prefos_thread_join(
                &threads[thread - 1]);
    return implied;
}

PreFOSStatus prefos_internal_find_fixed_box_variables(
    PreFOSPresolver *presolver,
    const PreFOSColumnWorkspace *workspace, size_t *n_fixed)
{
    const PreFOSProblemData *problem = &presolver->original;
    size_t candidate_count, position;

    if (!n_fixed) return PREFOS_STATUS_INVALID_ARGUMENT;
    *n_fixed = 0;
    if (!presolver->settings.fix_close_box_bounds) return PREFOS_STATUS_OK;
    candidate_count = presolver->n_fixed_box_dirty;
    presolver->n_fixed_box_dirty = 0;
    for (position = 0; position < candidate_count; ++position)
    {
        int box = presolver->fixed_box_dirty_queue[position];
        int column = problem->box_indices[box];
        presolver->fixed_box_dirty[box] = 0;
        if (presolver->is_fixed[column] ||
            presolver->is_substituted[column] ||
            presolver->is_parallel_removed[column] ||
            !close_linear_box_bounds(
                presolver, workspace, column,
                presolver->working_box_lower[box],
                presolver->working_box_upper[box]))
            continue;
        prefos_internal_mark_fixed_column(
            presolver, column,
            prefos_internal_safe_midpoint(
                presolver->working_box_lower[box],
                presolver->working_box_upper[box]));
        ++(*n_fixed);
    }
    return PREFOS_STATUS_OK;
}

static PreFOSStatus reduce_trivial_row_state(
    PreFOSPresolver *presolver, size_t row, long double fixed_shift,
    size_t live, int live_column, double live_coefficient)
{
    const PreFOSProblemData *problem = &presolver->original;
    const PreFOSCsrMatrix *matrix = &problem->A;
    const double tolerance = presolver->settings.feasibility_tolerance;
    double lower, upper;

    if (row >= matrix->rows) return PREFOS_STATUS_INVALID_ARGUMENT;
    if (presolver->remove_rows[row]) return PREFOS_STATUS_OK;
    lower = (double)
        ((long double) presolver->working_constraint_lower[row] -
         fixed_shift);
    upper = (double)
        ((long double) presolver->working_constraint_upper[row] -
         fixed_shift);
    if (isnan(lower) || isnan(upper))
    {
        trace_trivial_row_failure(
            row, live_column, "shifted-side",
            PREFOS_STATUS_NUMERICAL_ERROR);
        return PREFOS_STATUS_NUMERICAL_ERROR;
    }

    if (live == 0)
    {
        if (lower > tolerance || upper < -tolerance)
        {
            trace_trivial_row_failure(
                row, -1, "empty-row-infeasible",
                PREFOS_STATUS_PRIMAL_INFEASIBLE);
            {
                const char *trace = getenv("PREFOS_TRACE_TRIVIAL_ROW");
                if (trace && *trace && *trace != '0')
                    fprintf(
                        stderr,
                        "PreFOS empty-row residual row=%zu lower=%.17g "
                        "upper=%.17g tolerance=%.17g\n",
                        row, lower, upper, tolerance);
            }
            return PREFOS_STATUS_PRIMAL_INFEASIBLE;
        }
        if (!presolver->settings.remove_empty_rows) return PREFOS_STATUS_OK;
        prefos_internal_mark_removed_row(presolver, row);
        ++presolver->stats.removed_empty_rows;
        return PREFOS_STATUS_OK;
    }
    if (live != 1) return PREFOS_STATUS_OK;
    if (!isfinite(lower) && !isfinite(upper))
    {
        if (!presolver->settings.remove_redundant_rows)
            return PREFOS_STATUS_OK;
        prefos_internal_mark_removed_row(presolver, row);
        ++presolver->stats.removed_redundant_rows;
        return PREFOS_STATUS_OK;
    }
    {
        int box_position = presolver->variable_to_box[live_column];
        double implied_lower, implied_upper;
        double old_lower, old_upper, new_lower, new_upper;
        PreFOSStatus status;

        if (box_position < 0 || live_coefficient == 0.0)
            return PREFOS_STATUS_OK;
        old_lower = presolver->working_box_lower[box_position];
        old_upper = presolver->working_box_upper[box_position];
        if (singleton_row_is_implied_by_box(
                lower, upper, live_coefficient,
                old_lower, old_upper))
        {
            prefos_internal_mark_removed_row(presolver, row);
            ++presolver->stats.removed_singleton_rows;
            return PREFOS_STATUS_OK;
        }
        if (live_coefficient > 0.0)
        {
            implied_lower = prefos_internal_outward_bound_cast(
                (long double) lower / live_coefficient, 1);
            implied_upper = prefos_internal_outward_bound_cast(
                (long double) upper / live_coefficient, 0);
        }
        else
        {
            implied_lower = prefos_internal_outward_bound_cast(
                (long double) upper / live_coefficient, 1);
            implied_upper = prefos_internal_outward_bound_cast(
                (long double) lower / live_coefficient, 0);
        }
        if (implied_lower == INFINITY || implied_upper == -INFINITY)
        {
            trace_trivial_row_failure(
                row, live_column, "infinite-implied-bound",
                PREFOS_STATUS_PRIMAL_INFEASIBLE);
            return PREFOS_STATUS_PRIMAL_INFEASIBLE;
        }
        new_lower = fmax(old_lower, implied_lower);
        new_upper = fmin(old_upper, implied_upper);
        if (new_lower > new_upper + tolerance)
        {
            trace_trivial_row_failure(
                row, live_column, "inconsistent-implied-bound",
                PREFOS_STATUS_PRIMAL_INFEASIBLE);
            return PREFOS_STATUS_PRIMAL_INFEASIBLE;
        }
        if (new_lower > new_upper)
        {
            double midpoint =
                prefos_internal_safe_midpoint(new_lower, new_upper);
            new_lower = midpoint;
            new_upper = midpoint;
        }
        status = prefos_internal_append_bound_record(
            presolver, (int) row, live_column, old_lower, new_lower, 1);
        if (status != PREFOS_STATUS_OK)
        {
            trace_trivial_row_failure(
                row, live_column, "lower-bound-record", status);
            return status;
        }
        status = prefos_internal_append_bound_record(
            presolver, (int) row, live_column, old_upper, new_upper, 0);
        if (status != PREFOS_STATUS_OK)
        {
            trace_trivial_row_failure(
                row, live_column, "upper-bound-record", status);
            return status;
        }
        if (new_lower != old_lower)
            ++presolver->stats.tightened_box_bounds;
        if (new_upper != old_upper)
            ++presolver->stats.tightened_box_bounds;
        prefos_internal_linear_cache_mark_bound_dirty(
            presolver, live_column);
        prefos_internal_mark_fixed_box_dirty(
            presolver, live_column);
        presolver->working_box_lower[box_position] = new_lower;
        presolver->working_box_upper[box_position] = new_upper;
        presolver->propagation_lower[live_column] = new_lower;
        presolver->propagation_upper[live_column] = new_upper;
        prefos_internal_mark_removed_row(presolver, row);
        ++presolver->stats.removed_singleton_rows;
    }
    return PREFOS_STATUS_OK;
}

static PreFOSStatus reduce_trivial_row(
    PreFOSPresolver *presolver, size_t row)
{
    const PreFOSCsrMatrix *matrix = &presolver->original.A;
    long double fixed_shift = 0.0L;
    size_t live = 0;
    int live_column = -1;
    double live_coefficient = 0.0;
    int p;

    if (row >= matrix->rows) return PREFOS_STATUS_INVALID_ARGUMENT;
    if (presolver->remove_rows[row]) return PREFOS_STATUS_OK;
    for (p = matrix->row_pointers[row];
         p < matrix->row_pointers[row + 1]; ++p)
    {
        int column = matrix->column_indices[p];
        double coefficient = matrix->values[p];
        if (coefficient == 0.0) continue;
        if (presolver->is_fixed[column])
        {
            long double term =
                (long double) coefficient *
                (long double) presolver->fixed_values[column];
            if (!isfinite(term) || !isfinite(fixed_shift + term))
            {
                trace_trivial_row_failure(
                    row, column, "fixed-shift",
                    PREFOS_STATUS_NUMERICAL_ERROR);
                return PREFOS_STATUS_NUMERICAL_ERROR;
            }
            fixed_shift += term;
            continue;
        }
        if (presolver->is_substituted[column])
        {
            if (!prefos_internal_term_is_active_in_row(
                    presolver, row, column))
                continue;
            return PREFOS_STATUS_OK;
        }
        if (presolver->is_parallel_removed[column])
            continue;
        ++live;
        if (live == 1)
        {
            live_column = column;
            live_coefficient = coefficient;
        }
    }
    return reduce_trivial_row_state(
        presolver, row, fixed_shift, live, live_column,
        live_coefficient);
}

PreFOSStatus prefos_internal_reduce_trivial_rows(
    PreFOSPresolver *presolver)
{
    size_t row;
    for (row = 0; row < presolver->original.A.rows; ++row)
    {
        PreFOSStatus status = reduce_trivial_row(presolver, row);
        if (status != PREFOS_STATUS_OK) return status;
    }
    return PREFOS_STATUS_OK;
}

PreFOSStatus prefos_internal_reduce_trivial_row_candidates(
    PreFOSPresolver *presolver, int *rows, size_t count)
{
    unsigned char *box_implied;
    size_t left = 0, right = count;
    if (count > 0 && !rows) return PREFOS_STATUS_INVALID_ARGUMENT;
    while (left < right)
    {
        int row = rows[left];
        int is_equality;
        if (row < 0 ||
            (size_t) row >= presolver->original.A.rows)
            return PREFOS_STATUS_INVALID_ARGUMENT;
        is_equality =
            isfinite(
                presolver->working_constraint_lower[row]) &&
            presolver->working_constraint_lower[row] ==
                presolver->working_constraint_upper[row];
        if (is_equality)
        {
            ++left;
            continue;
        }
        do
        {
            int tail_row;
            --right;
            if (right == left) break;
            tail_row = rows[right];
            if (tail_row < 0 ||
                (size_t) tail_row >= presolver->original.A.rows)
                return PREFOS_STATUS_INVALID_ARGUMENT;
            is_equality =
                isfinite(
                    presolver->working_constraint_lower[tail_row]) &&
                presolver->working_constraint_lower[tail_row] ==
                    presolver->working_constraint_upper[tail_row];
        } while (!is_equality);
        if (right == left) break;
        {
            int temporary = rows[left];
            rows[left] = rows[right];
            rows[right] = temporary;
        }
        ++left;
    }
    box_implied = classify_trivial_rows_in_parallel(
        presolver, rows, count);
    for (left = 0; left < count; ++left)
    {
        PreFOSStatus status;
        if (box_implied && box_implied[left] &&
            !presolver->remove_rows[rows[left]])
        {
            prefos_internal_mark_removed_row(
                presolver, (size_t) rows[left]);
            ++presolver->stats.removed_singleton_rows;
            continue;
        }
        status = reduce_trivial_row(
            presolver, (size_t) rows[left]);
        if (status != PREFOS_STATUS_OK)
        {
            free(box_implied);
            return status;
        }
    }
    free(box_implied);
    return PREFOS_STATUS_OK;
}
