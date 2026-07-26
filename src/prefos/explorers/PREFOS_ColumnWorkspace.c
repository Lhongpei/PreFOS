/*
 * Copyright 2026 Hongpei Li
 * SPDX-License-Identifier: Apache-2.0
 */

#include "PREFOS_ColumnReductionInternal.h"
#include "PREFOS_CudaBackend.h"
#include "PREFOS_CudaLinearPropagation.h"
#include "common/PreFOSThread.h"
#include "core/PREFOS_Timer.h"

#include <limits.h>
#include <stdio.h>

#ifndef PREFOS_PARALLEL_COLUMN_WORKSPACE_MIN_NNZ
#define PREFOS_PARALLEL_COLUMN_WORKSPACE_MIN_NNZ 262144U
#endif

#ifndef PREFOS_PARALLEL_CSC_PARTITION_MIN_NNZ
#define PREFOS_PARALLEL_CSC_PARTITION_MIN_NNZ 4194304U
#endif

#ifndef PREFOS_INITIAL_ROW_ACTIVITY_CACHE
#define PREFOS_INITIAL_ROW_ACTIVITY_CACHE 1
#endif

#define PREFOS_CSC_CPU_THREADS 4
#define PREFOS_PARALLEL_CSC_MAX_OFFSET_BYTES (128U * 1024U * 1024U)

static int column_workspace_cpu_threading_enabled(void)
{
    return prefos_cpu_thread_limit(PREFOS_CSC_CPU_THREADS) > 1;
}

void prefos_internal_free_column_workspace(PreFOSColumnWorkspace *workspace)
{
    if (!workspace) return;
    free(workspace->starts);
    free(workspace->ends);
    free(workspace->rows);
    free(workspace->values);
    free(workspace->quadratic);
    free(workspace->factor);
    free(workspace->protected_target);
    free(workspace->bounded_doubleton_chain_target);
    free(workspace->csc_column_dirty);
    free(workspace->dirty_row);
    free(workspace->column_dirty_row_counts);
    free(workspace->row_lock_state);
    free(workspace->trivial_row_queued);
    free(workspace->bound_dirty_row_queued);
    free(workspace->singleton_column_queued);
    free(workspace->dual_column_queued);
    free(workspace->singleton_activity_lower);
    free(workspace->singleton_activity_upper);
    free(workspace->row_degrees);
    free(workspace->live_degrees);
    free(workspace->down_locks);
    free(workspace->up_locks);
    free(workspace->trivial_candidate_rows);
    free(workspace->bound_dirty_rows);
    free(workspace->singleton_candidate_columns);
    free(workspace->dual_candidate_columns);
    free(workspace->column_max_abs_coefficient);
    free(workspace->initial_row_activities);
    free(workspace->initial_finite_min_accumulators);
    free(workspace->initial_finite_max_accumulators);
    free(workspace->initial_unique_infinite_min_positions);
    free(workspace->initial_unique_infinite_max_positions);
    free(workspace->initial_max_box_range_contribution);
    free(workspace->initial_activity_has_box_column);
    free(workspace->initial_activity_valid);
    free(workspace->initial_activity_singleton_reusable);
    free(workspace->gpu_degrees);
    free(workspace->gpu_down_locked);
    free(workspace->gpu_up_locked);
    free(workspace->gpu_singleton_candidates);
    free(workspace->objective);
    free(workspace->singleton_targets);
    free(workspace->singleton_scales);
    free(workspace->singleton_deferred_columns);
    free(workspace->singleton_deferred_rows);
    free(workspace->singleton_row_activity_map);
    free(workspace->singleton_row_activity_epoch);
    free(workspace->singleton_processed_epoch);
    free(workspace->singleton_row_activities);
    free(workspace->parallel_columns);
    free(workspace->parallel_support_hashes);
    free(workspace->parallel_coefficient_hashes);
    free(workspace->parallel_sort_auxiliary);
    free(workspace->parallel_group_starts);
    free(workspace->parallel_gpu_eligible);
    free(workspace->parallel_column_dirty);
    free(workspace->parallel_dirty_columns);
    free(workspace->substitution_active_rows);
    free(workspace->substitution_active_coefficients);
    memset(workspace, 0, sizeof(*workspace));
}

static PreFOSStatus build_column_csc_cpu_sequential(
    const PreFOSPresolver *presolver, PreFOSColumnWorkspace *workspace,
    const int *column_counts)
{
    const PreFOSProblemData *problem = &presolver->original;
    int *cursor;
    size_t row, position;

    cursor = (int *) prefos_internal_alloc_array(problem->n, sizeof(int));
    if (problem->n > 0 && !cursor) return PREFOS_STATUS_OUT_OF_MEMORY;
    memset(workspace->starts, 0, (problem->n + 1) * sizeof(int));
    if (column_counts)
    {
        for (position = 0; position < problem->n; ++position)
        {
            if (column_counts[position] < 0)
            {
                free(cursor);
                return PREFOS_STATUS_NUMERICAL_ERROR;
            }
            workspace->starts[position + 1] =
                column_counts[position];
        }
    }
    else
    {
        for (row = 0; row < problem->A.rows; ++row)
        {
            int p;
            if (presolver->remove_rows[row]) continue;
            for (p = problem->A.row_pointers[row];
                 p < problem->A.row_pointers[row + 1]; ++p)
            {
                int column = problem->A.column_indices[p];
                if (problem->A.values[p] != 0.0)
                    ++workspace->starts[column + 1];
            }
        }
    }
    for (position = 0; position < problem->n; ++position)
        workspace->starts[position + 1] += workspace->starts[position];
    for (position = 0; position < problem->n; ++position)
        workspace->ends[position] = workspace->starts[position + 1];
    workspace->nnz = (size_t) workspace->starts[problem->n];
    workspace->rows =
        (int *) prefos_internal_alloc_array(workspace->nnz, sizeof(int));
    workspace->values =
        (double *) prefos_internal_alloc_array(workspace->nnz, sizeof(double));
    if (workspace->nnz > 0 && (!workspace->rows || !workspace->values))
    {
        free(cursor);
        free(workspace->rows);
        free(workspace->values);
        workspace->rows = NULL;
        workspace->values = NULL;
        return PREFOS_STATUS_OUT_OF_MEMORY;
    }
    if (problem->n > 0)
        memcpy(cursor, workspace->starts, problem->n * sizeof(int));
    for (row = 0; row < problem->A.rows; ++row)
    {
        int p;
        if (presolver->remove_rows[row]) continue;
        for (p = problem->A.row_pointers[row];
             p < problem->A.row_pointers[row + 1]; ++p)
        {
            int column, write;
            if (problem->A.values[p] == 0.0) continue;
            column = problem->A.column_indices[p];
            write = cursor[column]++;
            workspace->rows[write] = (int) row;
            workspace->values[write] = problem->A.values[p];
        }
    }
    free(cursor);
    return PREFOS_STATUS_OK;
}

typedef struct
{
    const PreFOSPresolver *presolver;
    size_t row_begin;
    size_t row_end;
    int *column_offsets;
    int *rows;
    double *values;
    int fill;
} PreFOSCscCpuTask;

static void *build_column_csc_partition(void *argument)
{
    PreFOSCscCpuTask *task = (PreFOSCscCpuTask *) argument;
    const PreFOSPresolver *presolver = task->presolver;
    const PreFOSCsrMatrix *matrix = &presolver->original.A;
    size_t row;
    for (row = task->row_begin; row < task->row_end; ++row)
    {
        int position;
        if (presolver->remove_rows[row]) continue;
        for (position = matrix->row_pointers[row];
             position < matrix->row_pointers[row + 1]; ++position)
        {
            int column, write;
            double value = matrix->values[position];
            if (value == 0.0) continue;
            column = matrix->column_indices[position];
            if (!task->fill)
            {
                ++task->column_offsets[column];
                continue;
            }
            write = task->column_offsets[column]++;
            task->rows[write] = (int) row;
            task->values[write] = value;
        }
    }
    return NULL;
}

static void run_csc_cpu_tasks(
    PreFOSCscCpuTask *tasks, int n_threads)
{
    PreFOSThread threads[PREFOS_CSC_CPU_THREADS - 1];
    unsigned char started[PREFOS_CSC_CPU_THREADS - 1] = {0};
    int thread;
    if (n_threads < 1) n_threads = 1;
    if (n_threads > PREFOS_CSC_CPU_THREADS)
        n_threads = PREFOS_CSC_CPU_THREADS;
    for (thread = 1; thread < n_threads; ++thread)
        if (prefos_thread_create(
                &threads[thread - 1], build_column_csc_partition,
                &tasks[thread]) == 0)
            started[thread - 1] = 1;
        else
            build_column_csc_partition(&tasks[thread]);
    build_column_csc_partition(&tasks[0]);
    for (thread = 1; thread < n_threads; ++thread)
        if (started[thread - 1])
            (void) prefos_thread_join(&threads[thread - 1]);
}

static int parallel_csc_cpu_thread_count(
    const PreFOSProblemData *problem, int maximum_threads)
{
    size_t offset_bytes;
    int n_threads =
        prefos_cpu_thread_limit(maximum_threads);
    if (n_threads <= 1 ||
        problem->A.nnz < PREFOS_PARALLEL_CSC_PARTITION_MIN_NNZ ||
        problem->A.rows < (size_t) n_threads ||
        problem->n == 0 ||
        (long double) problem->A.nnz <
            4.0L * (long double) problem->n ||
        problem->n >
            SIZE_MAX /
                ((size_t) n_threads * sizeof(int)))
        return 1;
    offset_bytes =
        problem->n * (size_t) n_threads * sizeof(int);
    return offset_bytes <=
                   (size_t) PREFOS_PARALLEL_CSC_MAX_OFFSET_BYTES
               ? n_threads
               : 1;
}

static size_t csc_partition_end(
    const PreFOSCsrMatrix *matrix, size_t target_nnz,
    size_t minimum_row, size_t maximum_row)
{
    size_t lower = minimum_row;
    size_t upper = maximum_row;
    while (lower < upper)
    {
        size_t middle = lower + (upper - lower) / 2;
        if ((size_t) matrix->row_pointers[middle] < target_nnz)
            lower = middle + 1;
        else
            upper = middle;
    }
    return lower;
}

static PreFOSStatus build_column_csc_cpu_parallel(
    const PreFOSPresolver *presolver, PreFOSColumnWorkspace *workspace,
    const int *column_counts, int n_threads)
{
    const PreFOSProblemData *problem = &presolver->original;
    PreFOSCscCpuTask tasks[PREFOS_CSC_CPU_THREADS];
    int *offsets;
    size_t base_nnz, extra_nnz, row_begin = 0;
    size_t column, total = 0;
    int thread;

    offsets = (int *) calloc(
        problem->n * (size_t) n_threads, sizeof(int));
    if (!offsets) return PREFOS_STATUS_OUT_OF_MEMORY;
    base_nnz = problem->A.nnz / (size_t) n_threads;
    extra_nnz = problem->A.nnz % (size_t) n_threads;
    for (thread = 0; thread < n_threads; ++thread)
    {
        size_t row_end;
        if (thread + 1 == n_threads)
            row_end = problem->A.rows;
        else
        {
            size_t completed_parts = (size_t) thread + 1U;
            size_t target_nnz =
                base_nnz * completed_parts +
                (completed_parts < extra_nnz
                     ? completed_parts
                     : extra_nnz);
            size_t maximum_row =
                problem->A.rows -
                (size_t) (n_threads - thread - 1);
            row_end = csc_partition_end(
                &problem->A, target_nnz,
                row_begin + 1U, maximum_row);
        }
        tasks[thread] = (PreFOSCscCpuTask){
            presolver, row_begin, row_end,
            offsets + (size_t) thread * problem->n,
            NULL, NULL, 0};
        row_begin = row_end;
    }
    run_csc_cpu_tasks(tasks, n_threads);

    for (column = 0; column < problem->n; ++column)
    {
        size_t column_total = 0;
        size_t write;
        for (thread = 0; thread < n_threads; ++thread)
        {
            int count =
                offsets[(size_t) thread * problem->n + column];
            if (count < 0 ||
                (size_t) count > SIZE_MAX - column_total)
            {
                free(offsets);
                return PREFOS_STATUS_NUMERICAL_ERROR;
            }
            column_total += (size_t) count;
        }
        if ((column_counts &&
             (column_counts[column] < 0 ||
              (size_t) column_counts[column] != column_total)) ||
            column_total > (size_t) INT_MAX ||
            total > (size_t) INT_MAX ||
            column_total > (size_t) INT_MAX - total)
        {
            free(offsets);
            return PREFOS_STATUS_NUMERICAL_ERROR;
        }
        workspace->starts[column] = (int) total;
        write = total;
        for (thread = 0; thread < n_threads; ++thread)
        {
            size_t index =
                (size_t) thread * problem->n + column;
            int count = offsets[index];
            offsets[index] = (int) write;
            write += (size_t) count;
        }
        total += column_total;
        workspace->ends[column] = (int) total;
    }
    workspace->starts[problem->n] = (int) total;
    workspace->nnz = total;
    workspace->rows =
        (int *) prefos_internal_alloc_array(total, sizeof(int));
    workspace->values =
        (double *) prefos_internal_alloc_array(total, sizeof(double));
    if (total > 0 && (!workspace->rows || !workspace->values))
    {
        free(offsets);
        free(workspace->rows);
        free(workspace->values);
        workspace->rows = NULL;
        workspace->values = NULL;
        return PREFOS_STATUS_OUT_OF_MEMORY;
    }
    for (thread = 0; thread < n_threads; ++thread)
    {
        tasks[thread].rows = workspace->rows;
        tasks[thread].values = workspace->values;
        tasks[thread].fill = 1;
    }
    run_csc_cpu_tasks(tasks, n_threads);
    free(offsets);
    return PREFOS_STATUS_OK;
}

static PreFOSStatus build_column_csc_cpu_with_limit(
    const PreFOSPresolver *presolver, PreFOSColumnWorkspace *workspace,
    const int *column_counts, int maximum_threads)
{
    int n_threads = parallel_csc_cpu_thread_count(
        &presolver->original, maximum_threads);
    if (n_threads > 1)
        return build_column_csc_cpu_parallel(
            presolver, workspace, column_counts, n_threads);
    return build_column_csc_cpu_sequential(
        presolver, workspace, column_counts);
}

static PreFOSStatus build_column_csc_cpu(
    const PreFOSPresolver *presolver, PreFOSColumnWorkspace *workspace,
    const int *column_counts)
{
    return build_column_csc_cpu_with_limit(
        presolver, workspace, column_counts,
        PREFOS_CSC_CPU_THREADS);
}

static PreFOSStatus build_column_csc_gpu(
    PreFOSPresolver *presolver, PreFOSColumnWorkspace *workspace,
    int *used_gpu)
{
    const PreFOSProblemData *problem = &presolver->original;
    PreFOSCudaPropagationStatus cuda_status;
    PreFOSCudaWorkspace *cuda_workspace;
    double milliseconds = 0.0, copy_milliseconds = 0.0;
    size_t active_nnz = 0;

    *used_gpu = 0;
    if (!presolver->settings.structural_reductions_gpu)
        return PREFOS_STATUS_OK;
    cuda_workspace =
        prefos_internal_cuda_workspace_get(presolver, &cuda_status);
    if (cuda_workspace && cuda_status == PREFOS_CUDA_PROPAGATION_OK)
        cuda_status = prefos_cuda_workspace_build_csc(
            cuda_workspace, presolver->remove_rows, workspace->starts,
            &active_nnz, &milliseconds);
    presolver->stats.column_csc_gpu_milliseconds += milliseconds;
    if (!cuda_workspace || cuda_status != PREFOS_CUDA_PROPAGATION_OK)
    {
        ++presolver->stats.column_csc_gpu_fallbacks;
        return PREFOS_STATUS_OK;
    }
    workspace->rows =
        (int *) prefos_internal_alloc_array(active_nnz, sizeof(int));
    workspace->values =
        (double *) prefos_internal_alloc_array(active_nnz, sizeof(double));
    if (active_nnz > 0 && (!workspace->rows || !workspace->values))
    {
        free(workspace->rows);
        free(workspace->values);
        workspace->rows = NULL;
        workspace->values = NULL;
        return PREFOS_STATUS_OUT_OF_MEMORY;
    }
    cuda_status = prefos_cuda_workspace_copy_csc(
        cuda_workspace, workspace->rows, workspace->values,
        &copy_milliseconds);
    presolver->stats.column_csc_gpu_milliseconds += copy_milliseconds;
    if (cuda_status != PREFOS_CUDA_PROPAGATION_OK)
    {
        free(workspace->rows);
        free(workspace->values);
        workspace->rows = NULL;
        workspace->values = NULL;
        memset(workspace->starts, 0, (problem->n + 1) * sizeof(int));
        ++presolver->stats.column_csc_gpu_fallbacks;
        return PREFOS_STATUS_OK;
    }
    workspace->nnz = active_nnz;
    for (size_t column = 0; column < problem->n; ++column)
        workspace->ends[column] = workspace->starts[column + 1];
    workspace->gpu_csc_valid = 1;
    ++presolver->stats.column_csc_gpu_builds;
    *used_gpu = 1;
    return PREFOS_STATUS_OK;
}

#define PREFOS_ROW_LOCK_LOWER 1U
#define PREFOS_ROW_LOCK_UPPER 2U
#define PREFOS_ROW_LOCK_DIRTY 4U

static void update_column_dirty_row_counts(
    const PreFOSPresolver *presolver,
    PreFOSColumnWorkspace *workspace, size_t row, int delta)
{
    const PreFOSCsrMatrix *matrix = &presolver->original.A;
    int position;
    if (!workspace->column_dirty_row_counts ||
        row >= matrix->rows || delta == 0)
        return;
    for (position = matrix->row_pointers[row];
         position < matrix->row_pointers[row + 1]; ++position)
    {
        int column = matrix->column_indices[position];
        int *count;
        if (matrix->values[position] == 0.0) continue;
        count = &workspace->column_dirty_row_counts[column];
        if (delta > 0)
        {
            if (*count < INT_MAX) ++(*count);
        }
        else if (*count > 0)
            --(*count);
    }
}

static void rebuild_column_dirty_row_counts(
    const PreFOSPresolver *presolver,
    PreFOSColumnWorkspace *workspace)
{
    size_t row;
    if (!workspace->column_dirty_row_counts) return;
    memset(
        workspace->column_dirty_row_counts, 0,
        presolver->original.n * sizeof(int));
    for (row = 0; row < presolver->original.A.rows; ++row)
        if (workspace->dirty_row[row] &&
            !presolver->remove_rows[row])
            update_column_dirty_row_counts(
                presolver, workspace, row, 1);
}

static unsigned char current_row_lock_state(
    const PreFOSPresolver *presolver,
    const PreFOSColumnWorkspace *workspace, size_t row)
{
    unsigned char state = 0;
    if (presolver->remove_rows[row]) return 0;
    if (isfinite(presolver->working_constraint_lower[row]))
        state |= PREFOS_ROW_LOCK_LOWER;
    if (isfinite(presolver->working_constraint_upper[row]))
        state |= PREFOS_ROW_LOCK_UPPER;
    if (workspace->dirty_row[row])
        state |= PREFOS_ROW_LOCK_DIRTY;
    return state;
}

static int row_state_down_locks(
    unsigned char state, double coefficient)
{
    if (state & PREFOS_ROW_LOCK_DIRTY) return 1;
    return coefficient > 0.0
               ? (state & PREFOS_ROW_LOCK_LOWER) != 0
               : (state & PREFOS_ROW_LOCK_UPPER) != 0;
}

static int row_state_up_locks(
    unsigned char state, double coefficient)
{
    if (state & PREFOS_ROW_LOCK_DIRTY) return 1;
    return coefficient > 0.0
               ? (state & PREFOS_ROW_LOCK_UPPER) != 0
               : (state & PREFOS_ROW_LOCK_LOWER) != 0;
}

static void initialize_column_locks(
    const PreFOSPresolver *presolver, PreFOSColumnWorkspace *workspace)
{
    const PreFOSCsrMatrix *matrix = &presolver->original.A;
    size_t row;
    if (presolver->original.n > 0)
    {
        memset(
            workspace->down_locks, 0,
            presolver->original.n * sizeof(int));
        memset(
            workspace->up_locks, 0,
            presolver->original.n * sizeof(int));
    }
    for (row = 0; row < matrix->rows; ++row)
    {
        unsigned char state =
            current_row_lock_state(presolver, workspace, row);
        int position;
        workspace->row_lock_state[row] = state;
        if (state == 0) continue;
        for (position = matrix->row_pointers[row];
             position < matrix->row_pointers[row + 1]; ++position)
        {
            int column = matrix->column_indices[position];
            double coefficient = matrix->values[position];
            if (coefficient == 0.0) continue;
            workspace->down_locks[column] +=
                row_state_down_locks(state, coefficient);
            workspace->up_locks[column] +=
                row_state_up_locks(state, coefficient);
        }
    }
}

static void initialize_row_degrees_and_locks(
    const PreFOSPresolver *presolver, PreFOSColumnWorkspace *workspace)
{
    const PreFOSCsrMatrix *matrix = &presolver->original.A;
    int build_activity = workspace->initial_row_activities != NULL;
    size_t row;

    if (matrix->rows > 0)
        memset(workspace->row_degrees, 0, matrix->rows * sizeof(int));
    if (presolver->original.n > 0)
    {
        memset(
            workspace->down_locks, 0,
            presolver->original.n * sizeof(int));
        memset(
            workspace->up_locks, 0,
            presolver->original.n * sizeof(int));
    }
    workspace->max_row_nnz = 0;
    for (row = 0; row < matrix->rows; ++row)
    {
        PresolveLinearActivity *activity =
            build_activity
                ? &workspace->initial_row_activities[row]
                : NULL;
        double finite_min = 0.0, finite_max = 0.0;
        long double maximum_box_range = 0.0L;
        unsigned char state =
            current_row_lock_state(presolver, workspace, row);
        int activity_valid = build_activity;
        int singleton_reusable = build_activity;
        int has_box_column = 0;
        int unique_infinite_min_position = -1;
        int unique_infinite_max_position = -1;
        int position;
        workspace->row_lock_state[row] = state;
        if (activity)
            memset(activity, 0, sizeof(*activity));
        if (presolver->remove_rows[row])
        {
            if (build_activity)
                workspace->initial_activity_valid[row] = 0;
            continue;
        }
        for (position = matrix->row_pointers[row];
             position < matrix->row_pointers[row + 1]; ++position)
        {
            int column = matrix->column_indices[position];
            double coefficient = matrix->values[position];
            if (coefficient == 0.0) continue;
            if (presolver->is_fixed[column] ||
                presolver->is_substituted[column] ||
                presolver->is_parallel_removed[column])
                singleton_reusable = 0;
            if (!presolver->is_fixed[column] &&
                !presolver->is_substituted[column] &&
                !presolver->is_parallel_removed[column])
                ++workspace->row_degrees[row];
            if (state != 0)
            {
                workspace->down_locks[column] +=
                    row_state_down_locks(state, coefficient);
                workspace->up_locks[column] +=
                    row_state_up_locks(state, coefficient);
            }
            if (build_activity &&
                prefos_internal_term_is_active_in_row(
                    presolver, row, column))
            {
                int box_position =
                    presolver->variable_to_box[column];
                double minimum_bound =
                    coefficient > 0.0
                        ? presolver->propagation_lower[column]
                        : presolver->propagation_upper[column];
                double maximum_bound =
                    coefficient > 0.0
                        ? presolver->propagation_upper[column]
                        : presolver->propagation_lower[column];
                ++activity->n_nonzeros;
                if (isfinite(minimum_bound))
                {
                    double product = coefficient * minimum_bound;
                    if (!isfinite(product) ||
                        !isfinite(finite_min + product))
                        activity_valid = 0;
                    else
                        finite_min += product;
                }
                else
                {
                    ++activity->n_infinite_min;
                    unique_infinite_min_position =
                        activity->n_infinite_min == 1
                            ? position -
                                  matrix->row_pointers[row]
                            : -1;
                }
                if (isfinite(maximum_bound))
                {
                    double product = coefficient * maximum_bound;
                    if (!isfinite(product) ||
                        !isfinite(finite_max + product))
                        activity_valid = 0;
                    else
                        finite_max += product;
                }
                else
                {
                    ++activity->n_infinite_max;
                    unique_infinite_max_position =
                        activity->n_infinite_max == 1
                            ? position -
                                  matrix->row_pointers[row]
                            : -1;
                }
                if (box_position >= 0)
                {
                    double lower =
                        presolver->propagation_lower[column];
                    double upper =
                        presolver->propagation_upper[column];
                    long double contribution =
                        isfinite(lower) && isfinite(upper)
                            ? fabsl((long double) coefficient) *
                                  ((long double) upper -
                                   (long double) lower)
                            : INFINITY;
                    if (contribution > maximum_box_range)
                        maximum_box_range = contribution;
                    has_box_column = 1;
                }
            }
        }
        if (build_activity)
        {
            activity->finite_min = (double) finite_min;
            activity->finite_max = (double) finite_max;
            if (!isfinite(activity->finite_min) ||
                !isfinite(activity->finite_max))
                activity_valid = 0;
            workspace->initial_finite_min_accumulators[row] =
                (long double) finite_min;
            workspace->initial_finite_max_accumulators[row] =
                (long double) finite_max;
            workspace->initial_unique_infinite_min_positions[row] =
                unique_infinite_min_position;
            workspace->initial_unique_infinite_max_positions[row] =
                unique_infinite_max_position;
            workspace->initial_max_box_range_contribution[row] =
                maximum_box_range;
            workspace->initial_activity_has_box_column[row] =
                (unsigned char) has_box_column;
            workspace->initial_activity_valid[row] =
                (unsigned char) activity_valid;
            workspace->initial_activity_singleton_reusable[row] =
                (unsigned char)
                    (activity_valid && singleton_reusable);
        }
        if ((size_t) workspace->row_degrees[row] >
            workspace->max_row_nnz)
            workspace->max_row_nnz =
                (size_t) workspace->row_degrees[row];
    }
}

typedef struct
{
    const PreFOSPresolver *presolver;
    PreFOSColumnWorkspace *workspace;
    const int *column_counts;
    int maximum_threads;
    PreFOSStatus status;
} PreFOSColumnCscBuildTask;

static void *build_column_csc_cpu_worker(void *argument)
{
    PreFOSColumnCscBuildTask *task =
        (PreFOSColumnCscBuildTask *) argument;
    task->status =
        build_column_csc_cpu_with_limit(
            task->presolver, task->workspace,
            task->column_counts, task->maximum_threads);
    return NULL;
}

static int traced_dual_column(void)
{
    static int initialized = 0;
    static int selected = -1;
    if (!initialized)
    {
        const char *value = getenv("PREFOS_TRACE_DUAL_COLUMN");
        if (value && *value)
            selected = atoi(value);
        initialized = 1;
    }
    return selected;
}

static int trace_removed_rows(void)
{
    static int initialized = 0;
    static int enabled = 0;
    if (!initialized)
    {
        enabled = getenv("PREFOS_TRACE_REMOVED_ROWS") != NULL;
        initialized = 1;
    }
    return enabled;
}

static void synchronize_row_locks(
    const PreFOSPresolver *presolver, PreFOSColumnWorkspace *workspace,
    size_t row)
{
    const PreFOSCsrMatrix *matrix = &presolver->original.A;
    unsigned char old_state, new_state;
    int position;
    if (!workspace || row >= matrix->rows) return;
    old_state = workspace->row_lock_state[row];
    new_state = current_row_lock_state(presolver, workspace, row);
    if (old_state == new_state) return;
    workspace->gpu_stats_valid = 0;
    workspace->row_lock_state[row] = new_state;
    for (position = matrix->row_pointers[row];
         position < matrix->row_pointers[row + 1]; ++position)
    {
        int column = matrix->column_indices[position];
        double coefficient = matrix->values[position];
        int down_before, up_before, down_change, up_change;
        if (coefficient == 0.0) continue;
        down_before = workspace->down_locks[column];
        up_before = workspace->up_locks[column];
        down_change =
            row_state_down_locks(new_state, coefficient) -
            row_state_down_locks(old_state, coefficient);
        up_change =
            row_state_up_locks(new_state, coefficient) -
            row_state_up_locks(old_state, coefficient);
        workspace->down_locks[column] += down_change;
        workspace->up_locks[column] += up_change;
        if (workspace->down_locks[column] < 0)
            workspace->down_locks[column] = 0;
        if (workspace->up_locks[column] < 0)
            workspace->up_locks[column] = 0;
        /*
         * Dual fixing depends only on whether each lock count is zero.  A
         * count change that stays positive cannot expose a reduction.  This
         * matters when a dense equality is substituted: deleting its row may
         * decrement nearly every column while the replacement rows still
         * keep those columns locked.
         */
        if ((down_before > 0 &&
             workspace->down_locks[column] == 0) ||
            (up_before > 0 &&
             workspace->up_locks[column] == 0))
            prefos_internal_queue_dual_column(
                presolver, workspace, column);
    }
}

static PreFOSStatus build_column_workspace(
    PreFOSPresolver *presolver, PreFOSColumnWorkspace *workspace,
    int allow_gpu, int cache_initial_activity,
    const int *column_counts)
{
    const PreFOSProblemData *problem = &presolver->original;
    size_t row;
    int used_gpu = 0;
    int row_state_initialized = 0;
    int build_initial_activity =
        cache_initial_activity && PREFOS_INITIAL_ROW_ACTIVITY_CACHE &&
        presolver->settings.linear_propagation &&
        problem->A.rows > 0 && problem->n_box > 0 &&
        (long double) problem->A.nnz <=
            (long double) problem->n_box *
                (long double)
                    presolver->settings
                        .event_queue_max_average_column_degree;
    int trace_workspace =
        getenv("PREFOS_TRACE_COLUMN_WORKSPACE") != NULL;
    PreFOSTimestamp trace_total_start, trace_start, trace_stop;
    PreFOSStatus status;

    if (trace_workspace)
    {
        prefos_internal_timer_now(&trace_total_start);
        trace_start = trace_total_start;
    }
    memset(workspace, 0, sizeof(*workspace));
    workspace->starts = (int *) calloc(problem->n + 1, sizeof(int));
    workspace->ends = (int *) calloc(problem->n, sizeof(int));
    workspace->quadratic =
        (unsigned char *) calloc(problem->n, sizeof(unsigned char));
    workspace->factor =
        (unsigned char *) calloc(problem->n, sizeof(unsigned char));
    workspace->protected_target =
        (unsigned char *) calloc(problem->n, sizeof(unsigned char));
    workspace->bounded_doubleton_chain_target =
        (unsigned char *) calloc(problem->n, sizeof(unsigned char));
    workspace->csc_column_dirty =
        (unsigned char *) calloc(problem->n, sizeof(unsigned char));
    workspace->dirty_row =
        (unsigned char *) calloc(problem->A.rows, sizeof(unsigned char));
    if (presolver->settings.parallel_column_reduction)
        workspace->column_dirty_row_counts =
            (int *) calloc(problem->n, sizeof(int));
    workspace->row_lock_state =
        (unsigned char *) calloc(problem->A.rows, sizeof(unsigned char));
    workspace->trivial_row_queued =
        (unsigned char *) calloc(problem->A.rows, sizeof(unsigned char));
    workspace->bound_dirty_row_queued =
        (unsigned char *) calloc(problem->A.rows, sizeof(unsigned char));
    workspace->singleton_column_queued =
        (unsigned char *) calloc(problem->n, sizeof(unsigned char));
    workspace->dual_column_queued =
        (unsigned char *) calloc(problem->n, sizeof(unsigned char));
    workspace->singleton_activity_lower =
        (double *) prefos_internal_alloc_array(problem->n, sizeof(double));
    workspace->singleton_activity_upper =
        (double *) prefos_internal_alloc_array(problem->n, sizeof(double));
    workspace->row_degrees =
        (int *) calloc(problem->A.rows, sizeof(int));
    workspace->live_degrees =
        (int *) calloc(problem->n, sizeof(int));
    workspace->down_locks =
        (int *) calloc(problem->n, sizeof(int));
    workspace->up_locks =
        (int *) calloc(problem->n, sizeof(int));
    workspace->trivial_candidate_rows =
        (int *) prefos_internal_alloc_array(problem->A.rows, sizeof(int));
    workspace->bound_dirty_rows =
        (int *) prefos_internal_alloc_array(problem->A.rows, sizeof(int));
    workspace->singleton_candidate_columns =
        (int *) prefos_internal_alloc_array(problem->n, sizeof(int));
    workspace->dual_candidate_columns =
        (int *) prefos_internal_alloc_array(problem->n, sizeof(int));
    workspace->column_max_abs_coefficient =
        (double *) prefos_internal_alloc_array(problem->n, sizeof(double));
    workspace->objective = (double *) calloc(problem->n, sizeof(double));
    workspace->parallel_column_dirty =
        (unsigned char *) calloc(problem->n, sizeof(unsigned char));
    workspace->parallel_dirty_columns =
        (int *) prefos_internal_alloc_array(problem->n, sizeof(int));
    if (build_initial_activity)
    {
        workspace->initial_row_activities =
            (PresolveLinearActivity *) calloc(
                problem->A.rows, sizeof(PresolveLinearActivity));
        workspace->initial_finite_min_accumulators =
            (long double *) calloc(
                problem->A.rows, sizeof(long double));
        workspace->initial_finite_max_accumulators =
            (long double *) calloc(
                problem->A.rows, sizeof(long double));
        workspace->initial_unique_infinite_min_positions =
            (int *) prefos_internal_alloc_array(
                problem->A.rows, sizeof(int));
        workspace->initial_unique_infinite_max_positions =
            (int *) prefos_internal_alloc_array(
                problem->A.rows, sizeof(int));
        workspace->initial_max_box_range_contribution =
            (long double *) calloc(
                problem->A.rows, sizeof(long double));
        workspace->initial_activity_has_box_column =
            (unsigned char *) calloc(
                problem->A.rows, sizeof(unsigned char));
        workspace->initial_activity_valid =
            (unsigned char *) calloc(
                problem->A.rows, sizeof(unsigned char));
        workspace->initial_activity_singleton_reusable =
            (unsigned char *) calloc(
                problem->A.rows, sizeof(unsigned char));
    }
    if (!workspace->starts ||
        (problem->n > 0 &&
         (!workspace->ends || !workspace->quadratic || !workspace->factor ||
          !workspace->protected_target ||
          !workspace->bounded_doubleton_chain_target ||
          !workspace->csc_column_dirty ||
          (presolver->settings.parallel_column_reduction &&
           !workspace->column_dirty_row_counts) ||
          !workspace->singleton_column_queued ||
          !workspace->dual_column_queued ||
          !workspace->singleton_activity_lower ||
          !workspace->singleton_activity_upper ||
          !workspace->live_degrees ||
          !workspace->down_locks || !workspace->up_locks ||
          !workspace->singleton_candidate_columns ||
          !workspace->dual_candidate_columns ||
          !workspace->column_max_abs_coefficient ||
          !workspace->parallel_column_dirty ||
          !workspace->parallel_dirty_columns ||
          !workspace->objective)) ||
        (problem->A.rows > 0 &&
         (!workspace->dirty_row || !workspace->row_lock_state ||
          !workspace->trivial_row_queued ||
          !workspace->bound_dirty_row_queued ||
          !workspace->row_degrees || !workspace->trivial_candidate_rows ||
          !workspace->bound_dirty_rows ||
          (build_initial_activity &&
           (!workspace->initial_row_activities ||
            !workspace->initial_finite_min_accumulators ||
            !workspace->initial_finite_max_accumulators ||
            !workspace->initial_unique_infinite_min_positions ||
            !workspace->initial_unique_infinite_max_positions ||
            !workspace->initial_max_box_range_contribution ||
            !workspace->initial_activity_has_box_column ||
            !workspace->initial_activity_valid ||
            !workspace->initial_activity_singleton_reusable)))))
    {
        prefos_internal_free_column_workspace(workspace);
        return PREFOS_STATUS_OUT_OF_MEMORY;
    }
    if (trace_workspace)
    {
        prefos_internal_timer_now(&trace_stop);
        fprintf(
            stderr,
            "PreFOS column-workspace allocate_ms=%.3f rows=%zu "
            "columns=%zu nnz=%zu\n",
            prefos_internal_timer_elapsed_milliseconds(
                &trace_start, &trace_stop),
            problem->A.rows, problem->n, problem->A.nnz);
        trace_start = trace_stop;
    }
    status = allow_gpu
                 ? build_column_csc_gpu(
                       presolver, workspace, &used_gpu)
                 : PREFOS_STATUS_OK;
    if (status != PREFOS_STATUS_OK)
    {
        prefos_internal_free_column_workspace(workspace);
        return status;
    }
    if (!used_gpu)
    {
        if (column_workspace_cpu_threading_enabled() &&
            problem->A.nnz >=
                PREFOS_PARALLEL_COLUMN_WORKSPACE_MIN_NNZ)
        {
            int available_threads =
                prefos_cpu_thread_limit(PREFOS_CSC_CPU_THREADS);
            PreFOSColumnCscBuildTask task = {
                .presolver = presolver,
                .workspace = workspace,
                .column_counts = column_counts,
                .maximum_threads = available_threads - 1,
                .status = PREFOS_STATUS_OK};
            PreFOSThread thread;
            if (prefos_thread_create(
                    &thread, build_column_csc_cpu_worker,
                    &task) == 0)
            {
                initialize_row_degrees_and_locks(
                    presolver, workspace);
                (void) prefos_thread_join(&thread);
                status = task.status;
            }
            else
            {
                status = build_column_csc_cpu(
                    presolver, workspace, column_counts);
                if (status == PREFOS_STATUS_OK)
                    initialize_row_degrees_and_locks(
                        presolver, workspace);
            }
        }
        else
        {
            status = build_column_csc_cpu(
                presolver, workspace, column_counts);
            if (status == PREFOS_STATUS_OK)
                initialize_row_degrees_and_locks(
                    presolver, workspace);
        }
        if (status != PREFOS_STATUS_OK)
        {
            prefos_internal_free_column_workspace(workspace);
            return status;
        }
        row_state_initialized = 1;
    }
    else
    {
        initialize_row_degrees_and_locks(presolver, workspace);
        row_state_initialized = 1;
    }
    if (trace_workspace)
    {
        prefos_internal_timer_now(&trace_stop);
        fprintf(
            stderr,
            "PreFOS column-workspace csc_ms=%.3f gpu=%d\n",
            prefos_internal_timer_elapsed_milliseconds(
                &trace_start, &trace_stop),
            used_gpu);
        trace_start = trace_stop;
    }
    for (row = 0; row < problem->n; ++row)
    {
        workspace->singleton_activity_lower[row] =
            presolver->propagation_lower[row];
        workspace->singleton_activity_upper[row] =
            presolver->propagation_upper[row];
        workspace->column_max_abs_coefficient[row] = -1.0;
        workspace->protected_target[row] = (unsigned char)
            presolver->affine_face_substitution_targets[row];
        workspace->live_degrees[row] =
            workspace->ends[row] - workspace->starts[row];
        if (workspace->live_degrees[row] == 1)
            prefos_internal_queue_singleton_column(
                presolver, workspace, (int) row);
    }
    workspace->removed_row_cursor = presolver->n_removed_rows;
    workspace->compacted_removed_row_cursor =
        presolver->n_removed_rows;
    workspace->fixed_column_cursor = presolver->n_fixed_columns;
    workspace->singleton_fixed_column_cursor =
        presolver->n_fixed_columns;
    if (presolver->n_substitution_terms > 0)
        for (row = 0; row < problem->A.rows; ++row)
        {
            int p;
            if (presolver->remove_rows[row]) continue;
            for (p = problem->A.row_pointers[row];
                 p < problem->A.row_pointers[row + 1]; ++p)
            {
                int column = problem->A.column_indices[p];
                if (presolver->is_substituted[column])
                {
                    if (!prefos_internal_term_is_active_in_row(
                            presolver, row, column))
                        continue;
                    workspace->dirty_row[row] = 1;
                    break;
                }
            }
        }
    rebuild_column_dirty_row_counts(presolver, workspace);
    if (!row_state_initialized ||
        presolver->n_substitution_terms > 0)
        initialize_column_locks(presolver, workspace);

    for (row = 0; row < problem->Q.rows; ++row)
    {
        int p;
        for (p = problem->Q.row_pointers[row];
             p < problem->Q.row_pointers[row + 1]; ++p)
        {
            if (problem->Q.values[p] == 0.0) continue;
            workspace->quadratic[row] = 1;
            workspace->quadratic[problem->Q.column_indices[p]] = 1;
        }
    }
    for (row = 0; row < problem->R.rows; ++row)
    {
        int p;
        for (p = problem->R.row_pointers[row];
             p < problem->R.row_pointers[row + 1]; ++p)
            if (problem->R.values[p] != 0.0)
                workspace->factor[problem->R.column_indices[p]] = 1;
    }
    if (trace_workspace)
    {
        prefos_internal_timer_now(&trace_stop);
        fprintf(
            stderr,
            "PreFOS column-workspace state_ms=%.3f\n",
            prefos_internal_timer_elapsed_milliseconds(
                &trace_start, &trace_stop));
        trace_start = trace_stop;
    }
    status = prefos_internal_rebuild_column_objective(
        presolver, workspace);
    if (status != PREFOS_STATUS_OK)
    {
        prefos_internal_free_column_workspace(workspace);
        return status;
    }
    if (trace_workspace)
    {
        prefos_internal_timer_now(&trace_stop);
        fprintf(
            stderr,
            "PreFOS column-workspace objective_ms=%.3f\n",
            prefos_internal_timer_elapsed_milliseconds(
                &trace_start, &trace_stop));
        trace_start = trace_stop;
    }
    for (row = 0; row < problem->n; ++row)
        prefos_internal_queue_dual_column(
            presolver, workspace, (int) row);
    workspace->transformation_event_cursor =
        presolver->transformations.n_events;
    workspace->initial_activity_fixed_column_cursor =
        presolver->n_fixed_columns;
    workspace->initial_activity_event_cursor =
        presolver->transformations.n_events;
    if (trace_workspace)
    {
        prefos_internal_timer_now(&trace_stop);
        fprintf(
            stderr,
            "PreFOS column-workspace candidates_ms=%.3f total_ms=%.3f\n",
            prefos_internal_timer_elapsed_milliseconds(
                &trace_start, &trace_stop),
            prefos_internal_timer_elapsed_milliseconds(
                &trace_total_start, &trace_stop));
    }
    return PREFOS_STATUS_OK;
}

PreFOSStatus prefos_internal_build_column_workspace(
    PreFOSPresolver *presolver, PreFOSColumnWorkspace *workspace)
{
    return build_column_workspace(
        presolver, workspace, 1, 1, NULL);
}

PreFOSStatus prefos_internal_build_column_workspace_cpu(
    PreFOSPresolver *presolver, PreFOSColumnWorkspace *workspace)
{
    return build_column_workspace(
        presolver, workspace, 0, 1, NULL);
}

PreFOSStatus prefos_internal_build_structural_column_workspace_cpu(
    PreFOSPresolver *presolver, PreFOSColumnWorkspace *workspace)
{
    return build_column_workspace(
        presolver, workspace, 0, 0, NULL);
}

PreFOSStatus prefos_internal_build_column_workspace_cpu_with_counts(
    PreFOSPresolver *presolver, PreFOSColumnWorkspace *workspace,
    const int *column_counts, int cache_initial_activity)
{
    return build_column_workspace(
        presolver, workspace, 0, cache_initial_activity,
        column_counts);
}

void prefos_internal_refresh_column_workspace(
    const PreFOSPresolver *presolver, PreFOSColumnWorkspace *workspace)
{
    size_t column;
    prefos_internal_update_column_live_degrees(presolver, workspace);
    if (workspace->compacted_removed_row_cursor ==
            presolver->n_removed_rows &&
        !workspace->csc_has_zero_entries)
        return;
    for (column = 0; column < presolver->original.n; ++column)
    {
        int read, write = workspace->starts[column];
        for (read = workspace->starts[column];
             read < workspace->ends[column]; ++read)
        {
            int row = workspace->rows[read];
            if (presolver->remove_rows[row] ||
                workspace->values[read] == 0.0)
                continue;
            if (write != read)
            {
                workspace->rows[write] = row;
                workspace->values[write] = workspace->values[read];
            }
            ++write;
        }
        workspace->ends[column] = write;
        workspace->live_degrees[column] =
            write - workspace->starts[column];
    }
    workspace->compacted_removed_row_cursor =
        presolver->n_removed_rows;
    workspace->csc_has_zero_entries = 0;
}

void prefos_internal_update_column_live_degrees(
    const PreFOSPresolver *presolver, PreFOSColumnWorkspace *workspace)
{
    const PreFOSCsrMatrix *matrix = &presolver->original.A;
    if (workspace->removed_row_cursor < presolver->n_removed_rows)
        workspace->gpu_stats_valid = 0;
    while (workspace->removed_row_cursor < presolver->n_removed_rows)
    {
        int row =
            presolver->removed_row_log[workspace->removed_row_cursor++];
        int was_dirty = workspace->dirty_row[row] != 0;
        int position;
        synchronize_row_locks(
            presolver, workspace, (size_t) row);
        workspace->dirty_row[row] = 0;
        for (position = matrix->row_pointers[row];
             position < matrix->row_pointers[row + 1]; ++position)
        {
            int column = matrix->column_indices[position];
            if (matrix->values[position] == 0.0) continue;
            if (was_dirty &&
                workspace->column_dirty_row_counts &&
                workspace->column_dirty_row_counts[column] > 0)
                --workspace->column_dirty_row_counts[column];
            if (workspace->live_degrees[column] > 0)
            {
                workspace->column_max_abs_coefficient[column] = -1.0;
                --workspace->live_degrees[column];
                prefos_internal_queue_dual_column(
                    presolver, workspace, column);
                if (workspace->live_degrees[column] == 1)
                    prefos_internal_queue_singleton_column(
                        presolver, workspace, column);
            }
        }
    }
}

void prefos_internal_queue_trivial_row(
    const PreFOSPresolver *presolver, PreFOSColumnWorkspace *workspace,
    int row)
{
    if (row < 0 || (size_t) row >= presolver->original.A.rows ||
        presolver->remove_rows[row] || workspace->trivial_row_queued[row])
        return;
    workspace->trivial_row_queued[row] = 1;
    workspace->trivial_candidate_rows[
        workspace->n_trivial_candidate_rows++] = row;
}

void prefos_internal_queue_singleton_column(
    const PreFOSPresolver *presolver, PreFOSColumnWorkspace *workspace,
    int column)
{
    size_t remaining;
    if (column < 0 || (size_t) column >= presolver->original.n ||
        workspace->singleton_column_queued[column])
        return;
    if (presolver->variable_to_box[column] < 0 ||
        presolver->is_fixed[column] ||
        presolver->is_substituted[column] ||
        presolver->is_parallel_removed[column] ||
        presolver->substitution_fill_in_targets[column] ||
        presolver->affine_protected_columns[column])
        return;
    if (workspace->n_singleton_candidate_columns >= presolver->original.n)
    {
        remaining = workspace->n_singleton_candidate_columns -
                    workspace->singleton_candidate_position;
        if (workspace->singleton_candidate_position > 0 && remaining > 0)
            memmove(workspace->singleton_candidate_columns,
                    workspace->singleton_candidate_columns +
                        workspace->singleton_candidate_position,
                    remaining * sizeof(int));
        workspace->n_singleton_candidate_columns = remaining;
        workspace->singleton_candidate_position = 0;
    }
    if (workspace->n_singleton_candidate_columns >= presolver->original.n)
        return;
    workspace->singleton_column_queued[column] = 1;
    workspace->singleton_candidate_columns[
        workspace->n_singleton_candidate_columns++] = column;
}

void prefos_internal_queue_dual_column(
    const PreFOSPresolver *presolver, PreFOSColumnWorkspace *workspace,
    int column)
{
    size_t remaining;
    int trace_column = traced_dual_column();
    if (trace_column >= 0 && column >= 0 &&
        (size_t) column < presolver->original.n &&
        column == trace_column)
    {
        int position;
        fprintf(
            stderr,
            "PreFOS dual-queue col=%d degree=%d down=%d up=%d "
            "box=%d fixed=%d substituted=%d parallel=%d fill=%d "
            "affine=%d materialized=%d\n",
            column, workspace->live_degrees[column],
            workspace->down_locks[column],
            workspace->up_locks[column],
            presolver->variable_to_box[column],
            presolver->is_fixed[column],
            presolver->is_substituted[column],
            presolver->is_parallel_removed[column],
            presolver->substitution_fill_in_targets[column],
            presolver->affine_protected_columns[column],
            presolver->working_matrix_is_materialized);
        if (presolver->working_matrix_is_materialized &&
            workspace->live_degrees[column] <= 2)
            for (position = workspace->starts[column];
                 position < workspace->ends[column]; ++position)
            {
                int row = workspace->rows[position];
                fprintf(
                    stderr,
                    "PreFOS dual-queue-row col=%d row=%d a=%.17g "
                    "removed=%d dirty=%d lock_state=%u "
                    "lower=%.17g upper=%.17g\n",
                    column, row, workspace->values[position],
                    presolver->remove_rows[row],
                    workspace->dirty_row[row],
                    (unsigned int) workspace->row_lock_state[row],
                    presolver->working_constraint_lower[row],
                    presolver->working_constraint_upper[row]);
            }
    }
    if (column < 0 || (size_t) column >= presolver->original.n ||
        workspace->dual_column_queued[column])
        return;
    if (presolver->variable_to_box[column] < 0 ||
        presolver->is_fixed[column] ||
        presolver->is_substituted[column] ||
        presolver->is_parallel_removed[column] ||
        presolver->substitution_fill_in_targets[column] ||
        presolver->affine_protected_columns[column])
        return;
    /*
     * A nonempty column with locks in both directions cannot be dual fixed,
     * independently of its objective or bounds.  Keep the queue sparse and
     * let lock/live-degree transitions enqueue the column when this predicate
     * can actually change.
     */
    if (workspace->live_degrees[column] > 0 &&
        workspace->down_locks[column] > 0 &&
        workspace->up_locks[column] > 0)
        return;
    if (workspace->objective_synchronized &&
        workspace->live_degrees[column] > 0)
    {
        double objective = workspace->objective[column];
        if ((objective > 0.0 &&
             workspace->down_locks[column] > 0) ||
            (objective < 0.0 &&
             workspace->up_locks[column] > 0))
            return;
    }
    if (workspace->n_dual_candidate_columns >= presolver->original.n)
    {
        remaining = workspace->n_dual_candidate_columns -
                    workspace->dual_candidate_position;
        if (workspace->dual_candidate_position > 0 && remaining > 0)
            memmove(workspace->dual_candidate_columns,
                    workspace->dual_candidate_columns +
                        workspace->dual_candidate_position,
                    remaining * sizeof(int));
        workspace->n_dual_candidate_columns = remaining;
        workspace->dual_candidate_position = 0;
    }
    if (workspace->n_dual_candidate_columns >= presolver->original.n)
        return;
    workspace->dual_column_queued[column] = 1;
    workspace->dual_candidate_columns[
        workspace->n_dual_candidate_columns++] = column;
}

static void queue_changed_column(
    const PreFOSPresolver *presolver, PreFOSColumnWorkspace *workspace,
    int column)
{
    prefos_internal_queue_dual_column(
        presolver, workspace, column);
    if (column >= 0 &&
        (size_t) column < presolver->original.n &&
        workspace->live_degrees[column] == 1)
        prefos_internal_queue_singleton_column(
            presolver, workspace, column);
}

void prefos_internal_notify_row_side_change(
    PreFOSPresolver *presolver, PreFOSColumnWorkspace *workspace,
    size_t row)
{
    if (row >= presolver->original.A.rows ||
        presolver->remove_rows[row])
        return;
    prefos_internal_linear_cache_mark_row_dirty(
        presolver, row, 0);
    if (!workspace) return;
    synchronize_row_locks(presolver, workspace, row);
}

void prefos_internal_queue_row_side_change(
    PreFOSPresolver *presolver, PreFOSColumnWorkspace *workspace,
    size_t row)
{
    const PreFOSCsrMatrix *matrix = &presolver->original.A;
    int position;
    prefos_internal_notify_row_side_change(
        presolver, workspace, row);
    if (!workspace || row >= matrix->rows ||
        presolver->remove_rows[row])
        return;
    for (position = matrix->row_pointers[row];
         position < matrix->row_pointers[row + 1]; ++position)
    {
        int column = matrix->column_indices[position];
        if (matrix->values[position] == 0.0) continue;
        if (workspace->live_degrees[column] == 1)
            prefos_internal_queue_singleton_column(
                presolver, workspace, column);
    }
}

void prefos_internal_mark_workspace_row_dirty(
    const PreFOSPresolver *presolver, PreFOSColumnWorkspace *workspace,
    size_t row)
{
    const PreFOSCsrMatrix *matrix;
    int position;
    if (!workspace || row >= presolver->original.A.rows ||
        presolver->remove_rows[row] || workspace->dirty_row[row])
        return;
    matrix = &presolver->original.A;
    prefos_internal_invalidate_singleton_row_activity(workspace, row);
    if (workspace->initial_activity_valid)
        workspace->initial_activity_valid[row] = 0;
    workspace->dirty_row[row] = 1;
    if (presolver->settings.parallel_column_reduction)
        for (position = matrix->row_pointers[row];
             position < matrix->row_pointers[row + 1]; ++position)
            if (matrix->values[position] != 0.0)
            {
                int column = matrix->column_indices[position];
                int *count =
                    &workspace->column_dirty_row_counts[column];
                if (*count < INT_MAX) ++(*count);
                prefos_internal_mark_parallel_column_dirty(
                    workspace, column);
            }
    synchronize_row_locks(presolver, workspace, row);
}

void prefos_internal_begin_workspace_row_coefficient_update(
    const PreFOSPresolver *presolver, PreFOSColumnWorkspace *workspace,
    size_t row)
{
    const PreFOSCsrMatrix *matrix = &presolver->original.A;
    unsigned char state;
    int was_dirty;
    int position;
    if (!workspace || row >= matrix->rows) return;
    state = workspace->row_lock_state[row];
    if (state == 0) return;
    was_dirty = workspace->dirty_row[row] != 0;
    workspace->gpu_stats_valid = 0;
    workspace->gpu_singleton_candidates_valid = 0;
    workspace->row_lock_state[row] = 0;
    workspace->dirty_row[row] = 1;
    for (position = matrix->row_pointers[row];
         position < matrix->row_pointers[row + 1]; ++position)
    {
        int column = matrix->column_indices[position];
        double coefficient = matrix->values[position];
        if (coefficient == 0.0) continue;
        if (presolver->settings.parallel_column_reduction)
        {
            if (!was_dirty &&
                workspace->column_dirty_row_counts[column] < INT_MAX)
                ++workspace->column_dirty_row_counts[column];
            prefos_internal_mark_parallel_column_dirty(
                workspace, column);
        }
        workspace->column_max_abs_coefficient[column] = -1.0;
        workspace->down_locks[column] -=
            row_state_down_locks(state, coefficient);
        workspace->up_locks[column] -=
            row_state_up_locks(state, coefficient);
        if (workspace->down_locks[column] < 0)
            workspace->down_locks[column] = 0;
        if (workspace->up_locks[column] < 0)
            workspace->up_locks[column] = 0;
        prefos_internal_queue_dual_column(
            presolver, workspace, column);
    }
}

void prefos_internal_mark_workspace_row_exact(
    const PreFOSPresolver *presolver, PreFOSColumnWorkspace *workspace,
    size_t row)
{
    const PreFOSCsrMatrix *matrix;
    int position;
    if (!workspace || row >= presolver->original.A.rows ||
        !workspace->dirty_row[row])
        return;
    matrix = &presolver->original.A;
    workspace->dirty_row[row] = 0;
    if (presolver->settings.parallel_column_reduction)
        for (position = matrix->row_pointers[row];
             position < matrix->row_pointers[row + 1]; ++position)
            if (matrix->values[position] != 0.0)
            {
                int column = matrix->column_indices[position];
                if (workspace->column_dirty_row_counts[column] > 0)
                    --workspace->column_dirty_row_counts[column];
                prefos_internal_mark_parallel_column_dirty(
                    workspace, column);
            }
    synchronize_row_locks(presolver, workspace, row);
}

static int coefficient_down_locks(
    unsigned char state, double coefficient)
{
    return coefficient == 0.0
               ? 0
               : row_state_down_locks(state, coefficient);
}

static int coefficient_up_locks(
    unsigned char state, double coefficient)
{
    return coefficient == 0.0
               ? 0
               : row_state_up_locks(state, coefficient);
}

PreFOSStatus prefos_internal_update_workspace_row_coefficients(
    const PreFOSPresolver *presolver, PreFOSColumnWorkspace *workspace,
    size_t row, int first_column, double first_old,
    double first_new, int second_column, double second_old,
    double second_new)
{
    unsigned char state;
    int first_down, first_up, second_down, second_up;
    if (!presolver || !workspace ||
        row >= presolver->original.A.rows ||
        first_column < 0 || second_column < 0 ||
        (size_t) first_column >= presolver->original.n ||
        (size_t) second_column >= presolver->original.n ||
        first_column == second_column || workspace->dirty_row[row])
        return PREFOS_STATUS_INVALID_ARGUMENT;
    state = workspace->row_lock_state[row];
    if (state != current_row_lock_state(
                     presolver, workspace, row))
        return PREFOS_STATUS_NUMERICAL_ERROR;

    first_down =
        coefficient_down_locks(state, first_new) -
        coefficient_down_locks(state, first_old);
    first_up =
        coefficient_up_locks(state, first_new) -
        coefficient_up_locks(state, first_old);
    second_down =
        coefficient_down_locks(state, second_new) -
        coefficient_down_locks(state, second_old);
    second_up =
        coefficient_up_locks(state, second_new) -
        coefficient_up_locks(state, second_old);
    if (workspace->down_locks[first_column] < -first_down ||
        workspace->up_locks[first_column] < -first_up ||
        workspace->down_locks[second_column] < -second_down ||
        workspace->up_locks[second_column] < -second_up)
        return PREFOS_STATUS_NUMERICAL_ERROR;
    workspace->down_locks[first_column] += first_down;
    workspace->up_locks[first_column] += first_up;
    workspace->down_locks[second_column] += second_down;
    workspace->up_locks[second_column] += second_up;
    workspace->column_max_abs_coefficient[first_column] = -1.0;
    workspace->column_max_abs_coefficient[second_column] = -1.0;
    workspace->gpu_stats_valid = 0;
    workspace->gpu_csc_valid = 0;
    workspace->gpu_singleton_candidates_valid = 0;
    return PREFOS_STATUS_OK;
}

void prefos_internal_invalidate_singleton_row_activity(
    PreFOSColumnWorkspace *workspace, size_t row)
{
    if (!workspace || !workspace->singleton_row_activity_epoch)
        return;
    workspace->singleton_row_activity_epoch[row] = 0U;
}

static int remove_cached_singleton_bound_contribution(
    PreFOSSingletonRowActivity *activity, double coefficient,
    double lower, double upper)
{
    double minimum_bound = coefficient > 0.0 ? lower : upper;
    double maximum_bound = coefficient > 0.0 ? upper : lower;
    long double term;
    if (activity->n_active_terms == 0) return 0;
    if (isfinite(minimum_bound))
    {
        term = (long double) coefficient *
               (long double) minimum_bound;
        activity->finite_min -= term;
        if (!isfinite(activity->finite_min)) return 0;
    }
    else
    {
        if (activity->n_infinite_min == 0) return 0;
        --activity->n_infinite_min;
    }
    if (isfinite(maximum_bound))
    {
        term = (long double) coefficient *
               (long double) maximum_bound;
        activity->finite_max -= term;
        if (!isfinite(activity->finite_max)) return 0;
    }
    else
    {
        if (activity->n_infinite_max == 0) return 0;
        --activity->n_infinite_max;
    }
    --activity->n_active_terms;
    return 1;
}

static int add_cached_singleton_bound_contribution(
    PreFOSSingletonRowActivity *activity, double coefficient,
    double lower, double upper)
{
    double minimum_bound = coefficient > 0.0 ? lower : upper;
    double maximum_bound = coefficient > 0.0 ? upper : lower;
    long double term;
    if (isfinite(minimum_bound))
    {
        term = (long double) coefficient *
               (long double) minimum_bound;
        activity->finite_min += term;
        if (!isfinite(activity->finite_min)) return 0;
    }
    else
        ++activity->n_infinite_min;
    if (isfinite(maximum_bound))
    {
        term = (long double) coefficient *
               (long double) maximum_bound;
        activity->finite_max += term;
        if (!isfinite(activity->finite_max)) return 0;
    }
    else
        ++activity->n_infinite_max;
    ++activity->n_active_terms;
    return 1;
}

static PreFOSSingletonRowActivity *
cached_singleton_row_activity(
    PreFOSColumnWorkspace *workspace, int row)
{
    int index;
    if (!workspace->singleton_row_activity_map ||
        !workspace->singleton_row_activity_epoch ||
        row < 0 ||
        workspace->singleton_row_activity_epoch[row] == 0U)
        return NULL;
    index = workspace->singleton_row_activity_map[row];
    if (index < 0 ||
        (size_t) index >= workspace->n_singleton_row_activities)
    {
        workspace->singleton_row_activity_epoch[row] = 0U;
        return NULL;
    }
    return &workspace->singleton_row_activities[index];
}

static void invalidate_cached_singleton_column_rows(
    const PreFOSPresolver *presolver,
    PreFOSColumnWorkspace *workspace, int column)
{
    int position;
    if (!workspace || column < 0 ||
        (size_t) column >= presolver->original.n)
        return;
    for (position = workspace->starts[column];
         position < workspace->ends[column]; ++position)
    {
        int row = workspace->rows[position];
        if (row >= 0)
            prefos_internal_invalidate_singleton_row_activity(
                workspace, (size_t) row);
    }
}

void prefos_internal_update_cached_singleton_column_bounds(
    const PreFOSPresolver *presolver, PreFOSColumnWorkspace *workspace,
    int column, double old_lower, double old_upper,
    double new_lower, double new_upper)
{
    int position;
    if (!workspace || column < 0 ||
        (size_t) column >= presolver->original.n ||
        (old_lower == new_lower && old_upper == new_upper))
        return;
    if (workspace->singleton_activity_lower[column] != old_lower ||
        workspace->singleton_activity_upper[column] != old_upper)
    {
        invalidate_cached_singleton_column_rows(
            presolver, workspace, column);
        workspace->singleton_activity_lower[column] = new_lower;
        workspace->singleton_activity_upper[column] = new_upper;
        return;
    }
    for (position = workspace->starts[column];
         position < workspace->ends[column]; ++position)
    {
        int row = workspace->rows[position];
        double coefficient = workspace->values[position];
        PreFOSSingletonRowActivity *activity;
        if (coefficient == 0.0 || presolver->remove_rows[row])
            continue;
        activity = cached_singleton_row_activity(workspace, row);
        if (!activity) continue;
        if (workspace->dirty_row[row] || activity->unsupported ||
            !remove_cached_singleton_bound_contribution(
                activity, coefficient, old_lower, old_upper) ||
            !add_cached_singleton_bound_contribution(
                activity, coefficient, new_lower, new_upper))
        {
            prefos_internal_invalidate_singleton_row_activity(
                workspace, (size_t) row);
            continue;
        }
        activity->max_box_range_contribution = INFINITY;
    }
    workspace->singleton_activity_lower[column] = new_lower;
    workspace->singleton_activity_upper[column] = new_upper;
}

static int update_cached_singleton_bound_contribution(
    PreFOSSingletonRowActivity *activity, double coefficient,
    double old_bound, double new_bound, int updates_minimum)
{
    long double *finite_sum =
        updates_minimum ? &activity->finite_min : &activity->finite_max;
    size_t *n_infinite =
        updates_minimum ? &activity->n_infinite_min
                        : &activity->n_infinite_max;
    long double term;
    if (isfinite(old_bound))
    {
        term = (long double) coefficient *
               (long double) old_bound;
        *finite_sum -= term;
        if (!isfinite(*finite_sum)) return 0;
    }
    else
    {
        if (*n_infinite == 0) return 0;
        --(*n_infinite);
    }
    if (isfinite(new_bound))
    {
        term = (long double) coefficient *
               (long double) new_bound;
        *finite_sum += term;
        if (!isfinite(*finite_sum)) return 0;
    }
    else
        ++(*n_infinite);
    return 1;
}

void prefos_internal_update_cached_singleton_column_bound(
    const PreFOSPresolver *presolver, PreFOSColumnWorkspace *workspace,
    int column, double old_bound, double new_bound, int is_lower)
{
    double *cached_bound;
    int position;
    if (!workspace || column < 0 ||
        (size_t) column >= presolver->original.n ||
        old_bound == new_bound)
        return;
    cached_bound = is_lower
                       ? &workspace->singleton_activity_lower[column]
                       : &workspace->singleton_activity_upper[column];
    if (*cached_bound != old_bound)
    {
        invalidate_cached_singleton_column_rows(
            presolver, workspace, column);
        *cached_bound = new_bound;
        return;
    }
    for (position = workspace->starts[column];
         position < workspace->ends[column]; ++position)
    {
        int row = workspace->rows[position];
        double coefficient = workspace->values[position];
        int updates_minimum =
            (is_lower && coefficient > 0.0) ||
            (!is_lower && coefficient < 0.0);
        PreFOSSingletonRowActivity *activity;
        if (coefficient == 0.0 || presolver->remove_rows[row])
            continue;
        activity = cached_singleton_row_activity(workspace, row);
        if (!activity) continue;
        if (workspace->dirty_row[row] || activity->unsupported ||
            !update_cached_singleton_bound_contribution(
                activity, coefficient, old_bound, new_bound,
                updates_minimum))
        {
            prefos_internal_invalidate_singleton_row_activity(
                workspace, (size_t) row);
            continue;
        }
        activity->max_box_range_contribution = INFINITY;
    }
    *cached_bound = new_bound;
}

void prefos_internal_remove_cached_singleton_column(
    const PreFOSPresolver *presolver, PreFOSColumnWorkspace *workspace,
    int column, double old_lower, double old_upper)
{
    int position;
    if (!workspace || column < 0 ||
        (size_t) column >= presolver->original.n)
        return;
    for (position = workspace->starts[column];
         position < workspace->ends[column]; ++position)
    {
        int row = workspace->rows[position];
        double coefficient = workspace->values[position];
        PreFOSSingletonRowActivity *activity;
        if (coefficient == 0.0 || presolver->remove_rows[row])
            continue;
        activity = cached_singleton_row_activity(workspace, row);
        if (!activity) continue;
        if (workspace->dirty_row[row] || activity->unsupported ||
            !remove_cached_singleton_bound_contribution(
                activity, coefficient, old_lower, old_upper))
        {
            prefos_internal_invalidate_singleton_row_activity(
                workspace, (size_t) row);
            continue;
        }
        activity->max_box_range_contribution = INFINITY;
    }
}

void prefos_internal_sync_singleton_activity_events(
    const PreFOSPresolver *presolver, PreFOSColumnWorkspace *workspace)
{
    while (workspace->singleton_fixed_column_cursor <
           presolver->n_fixed_columns)
    {
        int column = presolver->fixed_column_log[
            workspace->singleton_fixed_column_cursor++];
        int position;
        if (column < 0 ||
            (size_t) column >= presolver->original.n)
            continue;
        for (position = workspace->starts[column];
             position < workspace->ends[column]; ++position)
        {
            int row = workspace->rows[position];
            if (row >= 0)
                prefos_internal_invalidate_singleton_row_activity(
                    workspace, (size_t) row);
        }
    }
}

static void invalidate_initial_row_activity(
    PreFOSColumnWorkspace *workspace, int row)
{
    if (!workspace->initial_activity_valid || row < 0)
        return;
    workspace->initial_activity_valid[row] = 0;
}

static void invalidate_initial_column_rows(
    const PreFOSPresolver *presolver,
    PreFOSColumnWorkspace *workspace, int column)
{
    int position;
    if (!workspace->initial_activity_valid || column < 0 ||
        (size_t) column >= presolver->original.n)
        return;
    for (position = workspace->starts[column];
         position < workspace->ends[column]; ++position)
        invalidate_initial_row_activity(
            workspace, workspace->rows[position]);
}

void prefos_internal_sync_initial_row_activity_events(
    const PreFOSPresolver *presolver, PreFOSColumnWorkspace *workspace)
{
    const PresolveTransformationLog *log;
    size_t end;

    if (!workspace || !workspace->initial_activity_valid)
        return;
    while (workspace->initial_activity_fixed_column_cursor <
           presolver->n_fixed_columns)
    {
        int column = presolver->fixed_column_log[
            workspace->initial_activity_fixed_column_cursor++];
        invalidate_initial_column_rows(
            presolver, workspace, column);
    }

    log = &presolver->transformations;
    end = log->n_events;
    if (workspace->initial_activity_event_cursor > end)
        workspace->initial_activity_event_cursor = end;
    while (workspace->initial_activity_event_cursor < end)
    {
        const PresolveTransformationEvent *event =
            &log->events[
                workspace->initial_activity_event_cursor++];
        if (event->type == PRESOLVE_TRANSFORMATION_BOUND_CHANGE)
        {
            if (event->record_index < log->n_bound_changes)
                invalidate_initial_column_rows(
                    presolver, workspace,
                    log->bound_changes[event->record_index].column);
        }
        else if (event->type == PRESOLVE_TRANSFORMATION_ROW)
        {
            if (event->record_index < log->n_row_transformations)
            {
                const PresolveRowTransformationRecord *record =
                    &log->row_transformations[event->record_index];
                invalidate_initial_row_activity(
                    workspace, record->row);
                invalidate_initial_row_activity(
                    workspace, record->source_row);
            }
        }
        else if (event->type == PRESOLVE_TRANSFORMATION_COLUMN)
        {
            const PresolveColumnTransformationRecord *record;
            size_t position;
            if (event->record_index >=
                log->n_column_transformations)
                continue;
            record =
                &log->column_transformations[event->record_index];
            invalidate_initial_column_rows(
                presolver, workspace, record->column);
            invalidate_initial_column_rows(
                presolver, workspace, record->secondary_column);
            invalidate_initial_row_activity(
                workspace, record->source_row);
            if (record->type != PRESOLVE_COLUMN_SUBSTITUTED)
                continue;
            for (position = 0; position < record->length; ++position)
                invalidate_initial_row_activity(
                    workspace, record->indices[position]);
        }
    }
}

static void queue_bound_dirty_row(
    const PreFOSPresolver *presolver, PreFOSColumnWorkspace *workspace,
    int row, int invalidate_singleton)
{
    if (row < 0 || (size_t) row >= presolver->original.A.rows ||
        presolver->remove_rows[row])
        return;
    if (invalidate_singleton)
        prefos_internal_invalidate_singleton_row_activity(
            workspace, (size_t) row);
    invalidate_initial_row_activity(workspace, row);
    if (workspace->bound_dirty_row_queued[row]) return;
    workspace->bound_dirty_row_queued[row] = 1;
    workspace->bound_dirty_rows[
        workspace->n_bound_dirty_rows++] = row;
}

static void queue_bound_dirty_column_rows(
    const PreFOSPresolver *presolver, PreFOSColumnWorkspace *workspace,
    int column, int synchronize_cache)
{
    int invalidate_singleton = 0;
    int position;
    if (column < 0 || (size_t) column >= presolver->original.n)
        return;
    if (synchronize_cache)
    {
        invalidate_singleton =
            workspace->singleton_activity_lower[column] !=
                presolver->propagation_lower[column] ||
            workspace->singleton_activity_upper[column] !=
                presolver->propagation_upper[column];
        if (!invalidate_singleton) return;
    }
    for (position = workspace->starts[column];
         position < workspace->ends[column]; ++position)
        queue_bound_dirty_row(
            presolver, workspace, workspace->rows[position],
            invalidate_singleton);
    if (synchronize_cache)
    {
        workspace->singleton_activity_lower[column] =
            presolver->propagation_lower[column];
        workspace->singleton_activity_upper[column] =
            presolver->propagation_upper[column];
    }
}

static void flush_bound_dirty_rows(
    const PreFOSPresolver *presolver, PreFOSColumnWorkspace *workspace)
{
    const PreFOSCsrMatrix *matrix = &presolver->original.A;
    size_t dirty, position;
    for (dirty = 0; dirty < workspace->n_bound_dirty_rows; ++dirty)
    {
        int row = workspace->bound_dirty_rows[dirty];
        workspace->bound_dirty_row_queued[row] = 0;
        if (presolver->remove_rows[row]) continue;
        for (position = (size_t) matrix->row_pointers[row];
             position < (size_t) matrix->row_pointers[row + 1];
             ++position)
        {
            int column = matrix->column_indices[position];
            if (matrix->values[position] != 0.0 &&
                workspace->live_degrees[column] == 1)
                prefos_internal_queue_singleton_column(
                    presolver, workspace, column);
        }
    }
    workspace->n_bound_dirty_rows = 0;
}

void prefos_internal_queue_bound_changed_singletons(
    const PreFOSPresolver *presolver, PreFOSColumnWorkspace *workspace)
{
    size_t dirty;
    for (dirty = 0; dirty < presolver->n_fixed_box_dirty; ++dirty)
    {
        int box = presolver->fixed_box_dirty_queue[dirty];
        if (box < 0 || (size_t) box >= presolver->original.n_box)
            continue;
        queue_bound_dirty_column_rows(
            presolver, workspace,
            presolver->original.box_indices[box], 1);
    }
    flush_bound_dirty_rows(presolver, workspace);
}

static void queue_row_side_candidates(
    const PreFOSPresolver *presolver, PreFOSColumnWorkspace *workspace,
    const PresolveRowTransformationRecord *record)
{
    size_t position;
    if (record->row >= 0)
        synchronize_row_locks(
            presolver, workspace, (size_t) record->row);
    for (position = 0; position < record->length; ++position)
    {
        int column = record->indices[position];
        if (record->coefficients[position] == 0.0 ||
            column < 0 ||
            (size_t) column >= presolver->original.n)
            continue;
        queue_changed_column(
            presolver, workspace, column);
    }
}

static void queue_column_transformation(
    const PreFOSPresolver *presolver, PreFOSColumnWorkspace *workspace,
    const PresolveColumnTransformationRecord *record)
{
    size_t position;
    int column = record->column;

    prefos_internal_mark_parallel_column_dirty(
        workspace, column);
    prefos_internal_mark_parallel_column_dirty(
        workspace, record->secondary_column);
    queue_changed_column(
        presolver, workspace, column);
    queue_changed_column(
        presolver, workspace, record->secondary_column);
    if (column >= 0 &&
        (size_t) column < presolver->original.n &&
        presolver->is_substituted[column])
    {
        size_t start = presolver->substitution_term_start[column];
        size_t count = presolver->substitution_term_count[column];
        for (position = 0; position < count; ++position)
        {
            int target = presolver->substitution_targets[start + position];
            prefos_internal_mark_parallel_column_dirty(
                workspace, target);
            queue_changed_column(
                presolver, workspace, target);
            if (target >= 0 &&
                (size_t) target < presolver->original.n)
                workspace->protected_target[target] |=
                    presolver->affine_face_substitution_targets[target];
        }
    }
    if (record->type != PRESOLVE_COLUMN_SUBSTITUTED)
        return;
    for (position = 0; position < record->length; ++position)
    {
        int row = record->indices[position];
        if (row < 0 ||
            (size_t) row >= presolver->original.A.rows ||
            presolver->remove_rows[row])
            continue;
        if (record->direction == PREFOS_SUBSTITUTION_RESIDUAL_ROW &&
            row == record->source_row)
            continue;
        prefos_internal_invalidate_singleton_row_activity(
            workspace, (size_t) row);
        prefos_internal_mark_workspace_row_dirty(
            presolver, workspace, (size_t) row);
    }
}

int prefos_internal_queue_transformation_events(
    const PreFOSPresolver *presolver, PreFOSColumnWorkspace *workspace)
{
    const PresolveTransformationLog *log = &presolver->transformations;
    size_t end = log->n_events;
    int column_transformation_seen = 0;

    if (workspace->transformation_event_cursor > end)
        workspace->transformation_event_cursor = end;
    while (workspace->transformation_event_cursor < end)
    {
        const PresolveTransformationEvent *event =
            &log->events[workspace->transformation_event_cursor++];
        if (event->type == PRESOLVE_TRANSFORMATION_BOUND_CHANGE)
        {
            const PresolveBoundChangeRecord *record;
            if (event->record_index >= log->n_bound_changes)
                continue;
            record = &log->bound_changes[event->record_index];
            prefos_internal_mark_parallel_column_bound_dirty(
                workspace, record->column);
            prefos_internal_queue_dual_column(
                presolver, workspace, record->column);
            queue_bound_dirty_column_rows(
                presolver, workspace, record->column, 1);
        }
        else if (event->type == PRESOLVE_TRANSFORMATION_ROW)
        {
            const PresolveRowTransformationRecord *record;
            if (event->record_index >= log->n_row_transformations)
                continue;
            record = &log->row_transformations[event->record_index];
            if (record->type == PRESOLVE_ROW_LOWER_CHANGED ||
                record->type == PRESOLVE_ROW_UPPER_CHANGED ||
                record->type == PRESOLVE_ROW_EQUALITY_RELAXED)
                queue_row_side_candidates(
                    presolver, workspace, record);
        }
        else if (event->type == PRESOLVE_TRANSFORMATION_COLUMN)
        {
            if (event->record_index >= log->n_column_transformations)
                continue;
            workspace->objective_synchronized = 0;
            queue_column_transformation(
                presolver, workspace,
                &log->column_transformations[event->record_index]);
            column_transformation_seen = 1;
        }
    }
    flush_bound_dirty_rows(presolver, workspace);
    return column_transformation_seen;
}

PreFOSStatus prefos_internal_rebuild_column_objective(
    const PreFOSPresolver *presolver, PreFOSColumnWorkspace *workspace)
{
    PreFOSStatus status = prefos_internal_expand_linear_objective(
        presolver, workspace->objective, &workspace->objective_offset);
    if (status == PREFOS_STATUS_OK)
    {
        workspace->objective_fixed_column_cursor =
            presolver->n_fixed_columns;
        workspace->objective_synchronized = 1;
    }
    return status;
}

PreFOSStatus prefos_internal_sync_column_objective(
    const PreFOSPresolver *presolver, PreFOSColumnWorkspace *workspace)
{
    while (workspace->objective_fixed_column_cursor <
           presolver->n_fixed_columns)
    {
        int column = presolver->fixed_column_log[
            workspace->objective_fixed_column_cursor];
        double coefficient;
        if (column < 0 ||
            (size_t) column >= presolver->original.n)
            return PREFOS_STATUS_NUMERICAL_ERROR;
        coefficient = workspace->objective[column];
        if (coefficient != 0.0 &&
            !prefos_internal_safe_add_product(
                &workspace->objective_offset, coefficient,
                presolver->fixed_values[column]))
            return PREFOS_STATUS_NUMERICAL_ERROR;
        workspace->objective[column] = 0.0;
        ++workspace->objective_fixed_column_cursor;
    }
    return PREFOS_STATUS_OK;
}

static void seed_column_candidates(
    const PreFOSPresolver *presolver, PreFOSColumnWorkspace *workspace)
{
    size_t column;
    workspace->n_singleton_candidate_columns = 0;
    workspace->n_dual_candidate_columns = 0;
    workspace->singleton_candidate_position = 0;
    workspace->dual_candidate_position = 0;
    if (presolver->original.n > 0)
    {
        memset(workspace->singleton_column_queued, 0,
               presolver->original.n * sizeof(unsigned char));
        memset(workspace->dual_column_queued, 0,
               presolver->original.n * sizeof(unsigned char));
    }
    for (column = 0; column < presolver->original.n; ++column)
    {
        prefos_internal_queue_dual_column(
            presolver, workspace, (int) column);
        if (workspace->live_degrees[column] == 1)
            prefos_internal_queue_singleton_column(
                presolver, workspace, (int) column);
    }
}

PreFOSStatus prefos_internal_prepare_column_workspace(
    PreFOSPresolver *presolver, PreFOSColumnWorkspace *workspace)
{
    PreFOSStatus status;
    size_t column;
    prefos_internal_update_column_live_degrees(presolver, workspace);
    for (column = 0; column < presolver->original.n; ++column)
    {
        int position;
        workspace->protected_target[column] =
            presolver->affine_face_substitution_targets[column];
        if (!presolver->is_substituted[column])
            continue;
        for (position = workspace->starts[column];
             position < workspace->ends[column]; ++position)
        {
            int row = workspace->rows[position];
            if (presolver->remove_rows[row]) continue;
            if (prefos_internal_term_is_active_in_row(
                    presolver, (size_t) row, (int) column))
                workspace->dirty_row[row] = 1;
        }
    }
    rebuild_column_dirty_row_counts(presolver, workspace);
    initialize_column_locks(presolver, workspace);
    workspace->objective_synchronized = 0;
    status = prefos_internal_rebuild_column_objective(
        presolver, workspace);
    if (status == PREFOS_STATUS_OK)
    {
        seed_column_candidates(presolver, workspace);
        workspace->fixed_column_cursor =
            presolver->n_fixed_columns;
        workspace->singleton_fixed_column_cursor =
            presolver->n_fixed_columns;
        workspace->transformation_event_cursor =
            presolver->transformations.n_events;
        workspace->initial_activity_fixed_column_cursor =
            presolver->n_fixed_columns;
        workspace->initial_activity_event_cursor =
            presolver->transformations.n_events;
        if (workspace->initial_activity_valid &&
            presolver->original.A.rows > 0)
            memset(
                workspace->initial_activity_valid, 0,
                presolver->original.A.rows *
                    sizeof(unsigned char));
        if (workspace->singleton_row_activity_epoch &&
            presolver->original.A.rows > 0)
            memset(
                workspace->singleton_row_activity_epoch, 0,
                presolver->original.A.rows * sizeof(unsigned int));
    }
    return status;
}

void prefos_internal_mark_parallel_column_dirty(
    PreFOSColumnWorkspace *workspace, int column)
{
    if (!workspace || !workspace->parallel_column_dirty ||
        !workspace->parallel_dirty_columns || column < 0 ||
        workspace->parallel_column_dirty[column])
        return;
    workspace->parallel_column_dirty[column] = 1;
    workspace->parallel_dirty_columns[
        workspace->n_parallel_dirty_columns++] = column;
}

void prefos_internal_mark_parallel_column_bound_dirty(
    PreFOSColumnWorkspace *workspace, int column)
{
    if (!workspace || workspace->parallel_no_group_cache_valid)
        return;
    prefos_internal_mark_parallel_column_dirty(workspace, column);
}

PreFOSStatus prefos_internal_synchronize_column_workspace(
    PreFOSPresolver *presolver, PreFOSColumnWorkspace *workspace)
{
    prefos_internal_refresh_column_workspace(presolver, workspace);
    if (presolver->original.A.rows > 0)
        memset(workspace->dirty_row, 0,
               presolver->original.A.rows * sizeof(unsigned char));
    return prefos_internal_prepare_column_workspace(
        presolver, workspace);
}

static PreFOSStatus rebuild_inserted_column_workspace(
    PreFOSPresolver *presolver, PreFOSColumnWorkspace *workspace)
{
    const PreFOSProblemData *problem = &presolver->original;
    PreFOSStatus status;
    size_t column;

    free(workspace->rows);
    free(workspace->values);
    workspace->rows = NULL;
    workspace->values = NULL;
    workspace->nnz = 0;
    status = build_column_csc_cpu(presolver, workspace, NULL);
    if (status != PREFOS_STATUS_OK) return status;

    if (problem->A.rows > 0)
    {
        memset(
            workspace->dirty_row, 0,
            problem->A.rows * sizeof(unsigned char));
        memset(
            workspace->trivial_row_queued, 0,
            problem->A.rows * sizeof(unsigned char));
        memset(
            workspace->bound_dirty_row_queued, 0,
            problem->A.rows * sizeof(unsigned char));
    }
    if (problem->n > 0)
    {
        memset(
            workspace->csc_column_dirty, 0,
            problem->n * sizeof(unsigned char));
        if (workspace->column_dirty_row_counts)
            memset(
                workspace->column_dirty_row_counts, 0,
                problem->n * sizeof(int));
    }
    workspace->n_trivial_candidate_rows = 0;
    workspace->n_bound_dirty_rows = 0;
    workspace->csc_has_insertions = 0;
    workspace->csc_has_zero_entries = 0;
    workspace->gpu_stats_valid = 0;
    workspace->gpu_csc_valid = 0;
    workspace->gpu_singleton_candidates_valid = 0;
    workspace->one_sided_singletons_seeded = 0;
    initialize_row_degrees_and_locks(presolver, workspace);

    for (column = 0; column < problem->n; ++column)
    {
        workspace->live_degrees[column] =
            workspace->ends[column] - workspace->starts[column];
        workspace->column_max_abs_coefficient[column] = -1.0;
        workspace->singleton_activity_lower[column] =
            presolver->propagation_lower[column];
        workspace->singleton_activity_upper[column] =
            presolver->propagation_upper[column];
    }
    seed_column_candidates(presolver, workspace);
    workspace->removed_row_cursor = presolver->n_removed_rows;
    workspace->compacted_removed_row_cursor =
        presolver->n_removed_rows;
    workspace->fixed_column_cursor = presolver->n_fixed_columns;
    workspace->singleton_fixed_column_cursor =
        presolver->n_fixed_columns;
    workspace->transformation_event_cursor =
        presolver->transformations.n_events;
    workspace->initial_activity_fixed_column_cursor =
        presolver->n_fixed_columns;
    workspace->initial_activity_event_cursor =
        presolver->transformations.n_events;
    workspace->fast_pass_started = 1;
    workspace->n_singleton_row_activities = 0;
    if (workspace->singleton_row_activity_epoch &&
        problem->A.rows > 0)
        memset(
            workspace->singleton_row_activity_epoch, 0,
            problem->A.rows * sizeof(unsigned int));
    return PREFOS_STATUS_OK;
}

PreFOSStatus prefos_internal_refresh_column_workspace_incremental(
    PreFOSPresolver *presolver, PreFOSColumnWorkspace *workspace)
{
    int objective_dirty;
    PreFOSStatus status;
    if (workspace->csc_has_insertions)
    {
        status = rebuild_inserted_column_workspace(
            presolver, workspace);
        if (status != PREFOS_STATUS_OK) return status;
    }
    prefos_internal_refresh_column_workspace(presolver, workspace);
    objective_dirty = prefos_internal_queue_transformation_events(
        presolver, workspace);
    prefos_internal_queue_bound_changed_singletons(
        presolver, workspace);
    return objective_dirty
               ? prefos_internal_rebuild_column_objective(
                     presolver, workspace)
               : prefos_internal_sync_column_objective(
                     presolver, workspace);
}

PreFOSStatus prefos_internal_populate_gpu_column_stats(
    PreFOSPresolver *presolver, PreFOSColumnWorkspace *workspace)
{
    const PreFOSProblemData *problem = &presolver->original;
    PreFOSCudaPropagationStatus cuda_status;
    PreFOSCudaWorkspace *cuda_workspace;
    double milliseconds = 0.0;
    if (!presolver->settings.structural_reductions_gpu) return PREFOS_STATUS_OK;
    workspace->gpu_degrees =
        (int *) prefos_internal_alloc_array(problem->n, sizeof(int));
    workspace->gpu_down_locked =
        (unsigned char *) prefos_internal_alloc_array(problem->n, sizeof(unsigned char));
    workspace->gpu_up_locked =
        (unsigned char *) prefos_internal_alloc_array(problem->n, sizeof(unsigned char));
    if (problem->n > 0 &&
        (!workspace->gpu_degrees || !workspace->gpu_down_locked ||
         !workspace->gpu_up_locked))
        return PREFOS_STATUS_OUT_OF_MEMORY;
    cuda_workspace =
        prefos_internal_cuda_workspace_get(presolver, &cuda_status);
    if (cuda_workspace && cuda_status == PREFOS_CUDA_PROPAGATION_OK)
        cuda_status = prefos_cuda_workspace_column_stats(
            cuda_workspace, presolver->working_constraint_lower,
            presolver->working_constraint_upper, presolver->remove_rows,
            workspace->gpu_degrees, workspace->gpu_down_locked,
            workspace->gpu_up_locked, &milliseconds);
    if (cuda_status == PREFOS_CUDA_PROPAGATION_OK)
    {
        workspace->gpu_stats_valid = 1;
        ++presolver->stats.structural_gpu_passes;
    }
    else
    {
        free(workspace->gpu_degrees);
        free(workspace->gpu_down_locked);
        free(workspace->gpu_up_locked);
        workspace->gpu_degrees = NULL;
        workspace->gpu_down_locked = NULL;
        workspace->gpu_up_locked = NULL;
        ++presolver->stats.structural_gpu_fallbacks;
    }
    (void) milliseconds;
    return PREFOS_STATUS_OK;
}

int prefos_internal_column_is_linear_box(
    const PreFOSPresolver *presolver, const PreFOSColumnWorkspace *workspace,
    int column)
{
    return column >= 0 && (size_t) column < presolver->original.n &&
           presolver->variable_to_box[column] >= 0 &&
           !presolver->is_fixed[column] && !presolver->is_substituted[column] &&
           !presolver->is_parallel_removed[column] &&
           !presolver->substitution_fill_in_targets[column] &&
           !presolver->affine_protected_columns[column] &&
           !workspace->quadratic[column] && !workspace->factor[column];
}

double prefos_internal_column_max_abs_coefficient(
    const PreFOSPresolver *presolver,
    const PreFOSColumnWorkspace *workspace, int column)
{
    double maximum = workspace->column_max_abs_coefficient[column];
    int position;
    if (maximum >= 0.0) return maximum;
    maximum = 0.0;
    for (position = workspace->starts[column];
         position < workspace->ends[column]; ++position)
    {
        int row = workspace->rows[position];
        double coefficient = workspace->values[position];
        if (coefficient != 0.0 && !presolver->remove_rows[row])
            maximum = fmax(maximum, fabs(coefficient));
    }
    workspace->column_max_abs_coefficient[column] = maximum;
    return maximum;
}

int prefos_internal_mark_fixed_column(PreFOSPresolver *presolver, int column,
                                      double value)
{
    int newly_fixed = !presolver->is_fixed[column];
    if (!presolver->is_fixed[column])
    {
        ++presolver->fixed_column_epoch;
        if (presolver->n_fixed_columns < presolver->original.n)
            presolver->fixed_column_log[presolver->n_fixed_columns++] =
                column;
    }
    prefos_internal_linear_cache_mark_bound_dirty(
        presolver, column);
    presolver->is_fixed[column] = 1;
    presolver->fixed_values[column] = value;
    presolver->propagation_lower[column] = value;
    presolver->propagation_upper[column] = value;
    return newly_fixed;
}

void prefos_internal_mark_fixed_box_dirty(
    PreFOSPresolver *presolver, int column)
{
    int box_position;
    if (!presolver || !presolver->fixed_box_dirty ||
        !presolver->fixed_box_dirty_queue || column < 0 ||
        (size_t) column >= presolver->original.n)
        return;
    box_position = presolver->variable_to_box[column];
    if (box_position < 0 ||
        presolver->fixed_box_dirty[box_position])
        return;
    presolver->fixed_box_dirty[box_position] = 1;
    if (presolver->n_fixed_box_dirty < presolver->original.n_box)
        presolver->fixed_box_dirty_queue[presolver->n_fixed_box_dirty++] =
            box_position;
}

int prefos_internal_mark_removed_row(PreFOSPresolver *presolver, size_t row)
{
    int newly_removed;
    if (!presolver || row >= presolver->original.A.rows) return 0;
    newly_removed = !presolver->remove_rows[row];
    if (newly_removed)
    {
        presolver->remove_rows[row] = 1;
        if (presolver->n_removed_rows < presolver->original.A.rows)
            presolver->removed_row_log[presolver->n_removed_rows++] = (int) row;
        if (trace_removed_rows())
            fprintf(
                stderr, "PreFOS removed-row row=%zu fixed=%zu "
                "substituted=%zu parallel=%zu\n",
                row, presolver->n_fixed_columns,
                presolver->stats.substituted_free_variables,
                presolver->n_parallel_column_reductions);
    }
    return newly_removed;
}

PreFOSStatus prefos_internal_effective_row_bounds(
    const PreFOSPresolver *presolver, size_t row, double *lower, double *upper)
{
    const PreFOSCsrMatrix *A = &presolver->original.A;
    double shift = 0.0;
    int p;
    for (p = A->row_pointers[row]; p < A->row_pointers[row + 1]; ++p)
    {
        int column = A->column_indices[p];
        if (presolver->is_fixed[column] &&
            !prefos_internal_safe_add_product(
                &shift, A->values[p], presolver->fixed_values[column]))
            return PREFOS_STATUS_NUMERICAL_ERROR;
    }
    *lower = presolver->working_constraint_lower[row] - shift;
    *upper = presolver->working_constraint_upper[row] - shift;
    return isnan(*lower) || isnan(*upper) ? PREFOS_STATUS_NUMERICAL_ERROR
                                          : PREFOS_STATUS_OK;
}

size_t prefos_internal_collect_live_row(const PreFOSPresolver *presolver,
                                        size_t row, int *columns,
                                        double *coefficients, size_t capacity)
{
    const PreFOSCsrMatrix *A = &presolver->original.A;
    size_t count = 0;
    int p;
    for (p = A->row_pointers[row]; p < A->row_pointers[row + 1]; ++p)
    {
        int column = A->column_indices[p];
        if (A->values[p] == 0.0 || presolver->is_fixed[column] ||
            presolver->is_substituted[column] ||
            presolver->is_parallel_removed[column])
            continue;
        if (count < capacity)
        {
            columns[count] = column;
            coefficients[count] = A->values[p];
        }
        ++count;
    }
    return count;
}
