/*
 * Copyright 2026 Hongpei Li
 * SPDX-License-Identifier: Apache-2.0
 */

#include "PREFOS_MatrixCompaction.h"

#include "common/LinearPropagationKernel.h"
#include "common/LinearPropagationKernelImpl.h"
#include "PREFOS_WorkingMatrix.h"
#include "common/PreFOSThread.h"
#include "explorers/PREFOS_CudaBackend.h"
#include "explorers/PREFOS_CudaLinearPropagation.h"

#include <stdio.h>
#include <stdlib.h>

#define PREFOS_MATRIX_COMPACTION_CPU_THREADS 4
#define PREFOS_PARALLEL_MATRIX_COMPACTION_MIN_NNZ 1048576U

static PreFOSStatus compact_a_rows_only(
    PreFOSPresolver *presolver, int *used)
{
    const PreFOSProblemData *source = &presolver->original;
    PreFOSPresolvedProblem *target = &presolver->reduced;
    size_t row, column, kept_rows = 0, nnz = 0;
    size_t removed_empty_rows = 0, write = 0;

    *used = 0;
    if (target->n != source->n) return PREFOS_STATUS_OK;
    for (column = 0; column < source->n; ++column)
        if (presolver->original_to_reduced[column] != (int) column)
            return PREFOS_STATUS_OK;
    for (column = 0; column < source->A.nnz; ++column)
        if (source->A.values[column] == 0.0)
            return PREFOS_STATUS_OK;

    for (row = 0; row < source->A.rows; ++row)
    {
        size_t length;
        double lower, upper;
        if (presolver->remove_rows[row]) continue;
        length = (size_t) (
            source->A.row_pointers[row + 1] -
            source->A.row_pointers[row]);
        lower = presolver->working_constraint_lower[row];
        upper = presolver->working_constraint_upper[row];
        if (isnan(lower) || isnan(upper))
            return PREFOS_STATUS_NUMERICAL_ERROR;
        if (length == 0 && presolver->settings.remove_empty_rows)
        {
            if (lower > presolver->settings.feasibility_tolerance ||
                upper < -presolver->settings.feasibility_tolerance)
                return PREFOS_STATUS_PRIMAL_INFEASIBLE;
            ++removed_empty_rows;
            continue;
        }
        if (length > (size_t) INT_MAX - nnz)
            return PREFOS_STATUS_OUT_OF_MEMORY;
        nnz += length;
        ++kept_rows;
    }

    target->A.rows = kept_rows;
    target->A.cols = target->n;
    target->A.nnz = nnz;
    target->A.row_pointers =
        (int *) prefos_internal_alloc_array(kept_rows + 1, sizeof(int));
    target->A.values =
        (double *) prefos_internal_alloc_array(nnz, sizeof(double));
    target->A.column_indices =
        (int *) prefos_internal_alloc_array(nnz, sizeof(int));
    target->constraint_lower =
        (double *) prefos_internal_alloc_array(kept_rows, sizeof(double));
    target->constraint_upper =
        (double *) prefos_internal_alloc_array(kept_rows, sizeof(double));
    if (!target->A.row_pointers ||
        (nnz > 0 && (!target->A.values ||
                     !target->A.column_indices)) ||
        (kept_rows > 0 && (!target->constraint_lower ||
                           !target->constraint_upper)))
    {
        prefos_internal_free_csr(&target->A);
        free(target->constraint_lower);
        free(target->constraint_upper);
        target->constraint_lower = NULL;
        target->constraint_upper = NULL;
        return PREFOS_STATUS_OUT_OF_MEMORY;
    }

    kept_rows = 0;
    target->A.row_pointers[0] = 0;
    for (row = 0; row < source->A.rows; ++row)
    {
        int begin, end;
        size_t length;
        if (presolver->remove_rows[row]) continue;
        begin = source->A.row_pointers[row];
        end = source->A.row_pointers[row + 1];
        length = (size_t) (end - begin);
        if (length == 0 && presolver->settings.remove_empty_rows)
            continue;
        presolver->original_to_reduced_rows[row] = (int) kept_rows;
        target->constraint_lower[kept_rows] =
            presolver->working_constraint_lower[row];
        target->constraint_upper[kept_rows] =
            presolver->working_constraint_upper[row];
        if (length > 0)
        {
            memcpy(target->A.values + write,
                   source->A.values + begin,
                   length * sizeof(double));
            memcpy(target->A.column_indices + write,
                   source->A.column_indices + begin,
                   length * sizeof(int));
            write += length;
        }
        target->A.row_pointers[++kept_rows] = (int) write;
    }
    if (write != nnz)
    {
        prefos_internal_free_csr(&target->A);
        free(target->constraint_lower);
        free(target->constraint_upper);
        target->constraint_lower = NULL;
        target->constraint_upper = NULL;
        return PREFOS_STATUS_NUMERICAL_ERROR;
    }
    presolver->stats.removed_empty_rows += removed_empty_rows;
    *used = 1;
    return PREFOS_STATUS_OK;
}

static PreFOSStatus compact_a_without_substitutions_gpu(
    PreFOSPresolver *presolver, int *used_gpu, int *attempted_gpu)
{
    const PreFOSProblemData *source = &presolver->original;
    PreFOSPresolvedProblem *target = &presolver->reduced;
    PreFOSCudaWorkspace *workspace;
    PreFOSCudaPropagationStatus cuda_status;
    int *row_nnz = NULL, *row_map = NULL;
    double *shifts = NULL;
    unsigned char *exact_shift = NULL;
    size_t row, kept_rows = 0, nnz = 0, removed_empty_rows = 0;
    double milliseconds = 0.0;

    size_t active_rows = 0;
    *used_gpu = 0;
    *attempted_gpu = 0;
    if (!presolver->settings.structural_reductions_gpu ||
        source->A.rows == 0 || source->n == 0)
        return PREFOS_STATUS_OK;
    for (row = 0; row < source->A.rows; ++row)
        if (!presolver->remove_rows[row]) ++active_rows;
    if ((long double) active_rows >
            0.95L * (long double) source->A.rows &&
        (long double) target->n > 0.95L * (long double) source->n)
        return PREFOS_STATUS_OK;
    *attempted_gpu = 1;
    workspace = prefos_internal_cuda_workspace_get(presolver, &cuda_status);
    if (!workspace || cuda_status != PREFOS_CUDA_PROPAGATION_OK)
        return PREFOS_STATUS_OK;

    row_nnz = (int *) prefos_internal_alloc_array(source->A.rows, sizeof(int));
    row_map = (int *) prefos_internal_alloc_array(source->A.rows, sizeof(int));
    shifts =
        (double *) prefos_internal_alloc_array(source->A.rows, sizeof(double));
    exact_shift = (unsigned char *) prefos_internal_alloc_array(
        source->A.rows, sizeof(unsigned char));
    if (!row_nnz || !row_map || !shifts || !exact_shift)
    {
        free(row_nnz);
        free(row_map);
        free(shifts);
        free(exact_shift);
        return PREFOS_STATUS_OUT_OF_MEMORY;
    }
    cuda_status = prefos_cuda_compact_a_analyze(
        workspace, presolver->remove_rows, presolver->is_fixed,
        presolver->fixed_values, presolver->original_to_reduced,
        row_nnz, shifts, exact_shift, &milliseconds);
    presolver->stats.matrix_compaction_gpu_milliseconds += milliseconds;
    if (cuda_status != PREFOS_CUDA_PROPAGATION_OK) goto fallback;

    for (row = 0; row < source->A.rows; ++row)
    {
        double lower, upper;
        row_map[row] = -1;
        if (presolver->remove_rows[row]) continue;
        if (row_nnz[row] < 0)
        {
            cuda_status = PREFOS_CUDA_PROPAGATION_ERROR;
            goto fallback;
        }
        if (exact_shift[row])
        {
            int position;
            shifts[row] = 0.0;
            for (position = source->A.row_pointers[row];
                 position < source->A.row_pointers[row + 1]; ++position)
            {
                int column = source->A.column_indices[position];
                if (presolver->is_fixed[column] &&
                    !prefos_internal_safe_add_product(
                        &shifts[row], source->A.values[position],
                        presolver->fixed_values[column]))
                {
                    free(row_nnz);
                    free(row_map);
                    free(shifts);
                    free(exact_shift);
                    return PREFOS_STATUS_NUMERICAL_ERROR;
                }
            }
        }
        lower = presolver->working_constraint_lower[row] - shifts[row];
        upper = presolver->working_constraint_upper[row] - shifts[row];
        if (isnan(lower) || isnan(upper))
        {
            free(row_nnz);
            free(row_map);
            free(shifts);
            free(exact_shift);
            return PREFOS_STATUS_NUMERICAL_ERROR;
        }
        if (row_nnz[row] == 0 && presolver->settings.remove_empty_rows)
        {
            if (lower > presolver->settings.feasibility_tolerance ||
                upper < -presolver->settings.feasibility_tolerance)
            {
                if (getenv("PREFOS_TRACE_MATRIX_COMPACTION"))
                    fprintf(
                        stderr,
                        "PreFOS compaction empty-row conflict row=%zu "
                        "lower=%.17g upper=%.17g shift=%.17g\n",
                        row, lower, upper, shifts[row]);
                free(row_nnz);
                free(row_map);
                free(shifts);
                free(exact_shift);
                return PREFOS_STATUS_PRIMAL_INFEASIBLE;
            }
            ++removed_empty_rows;
            continue;
        }
        if ((size_t) row_nnz[row] > (size_t) INT_MAX - nnz)
        {
            free(row_nnz);
            free(row_map);
            free(shifts);
            free(exact_shift);
            return PREFOS_STATUS_OUT_OF_MEMORY;
        }
        row_map[row] = (int) kept_rows++;
        nnz += (size_t) row_nnz[row];
    }

    target->A.rows = kept_rows;
    target->A.cols = target->n;
    target->A.nnz = nnz;
    target->A.row_pointers = (int *) calloc(kept_rows + 1, sizeof(int));
    target->A.values =
        (double *) prefos_internal_alloc_array(nnz, sizeof(double));
    target->A.column_indices =
        (int *) prefos_internal_alloc_array(nnz, sizeof(int));
    target->constraint_lower =
        (double *) prefos_internal_alloc_array(kept_rows, sizeof(double));
    target->constraint_upper =
        (double *) prefos_internal_alloc_array(kept_rows, sizeof(double));
    if (!target->A.row_pointers ||
        (nnz > 0 && (!target->A.values || !target->A.column_indices)) ||
        (kept_rows > 0 &&
         (!target->constraint_lower || !target->constraint_upper)))
    {
        free(row_nnz);
        free(row_map);
        free(shifts);
        free(exact_shift);
        prefos_internal_free_csr(&target->A);
        free(target->constraint_lower);
        free(target->constraint_upper);
        target->constraint_lower = NULL;
        target->constraint_upper = NULL;
        return PREFOS_STATUS_OUT_OF_MEMORY;
    }
    kept_rows = 0;
    {
        int output_write = 0;
        for (row = 0; row < source->A.rows; ++row)
        {
            if (row_map[row] < 0) continue;
            target->A.row_pointers[kept_rows] = output_write;
            target->constraint_lower[kept_rows] =
                presolver->working_constraint_lower[row] - shifts[row];
            target->constraint_upper[kept_rows] =
                presolver->working_constraint_upper[row] - shifts[row];
            output_write += row_nnz[row];
            ++kept_rows;
        }
        target->A.row_pointers[kept_rows] = output_write;
        if ((size_t) output_write != nnz)
        {
            prefos_internal_free_csr(&target->A);
            free(target->constraint_lower);
            free(target->constraint_upper);
            target->constraint_lower = NULL;
            target->constraint_upper = NULL;
            cuda_status = PREFOS_CUDA_PROPAGATION_ERROR;
            goto fallback;
        }
    }

    milliseconds = 0.0;
    cuda_status = prefos_cuda_compact_a_write(
        workspace, presolver->original_to_reduced, row_map,
        target->A.row_pointers, kept_rows, nnz, target->A.column_indices,
        target->A.values, &milliseconds);
    presolver->stats.matrix_compaction_gpu_milliseconds += milliseconds;
    if (cuda_status != PREFOS_CUDA_PROPAGATION_OK)
    {
        prefos_internal_free_csr(&target->A);
        free(target->constraint_lower);
        free(target->constraint_upper);
        target->constraint_lower = NULL;
        target->constraint_upper = NULL;
        goto fallback;
    }
    memcpy(presolver->original_to_reduced_rows, row_map,
           source->A.rows * sizeof(int));
    presolver->stats.removed_empty_rows += removed_empty_rows;
    ++presolver->stats.matrix_compaction_gpu_passes;
    *used_gpu = 1;
    free(row_nnz);
    free(row_map);
    free(shifts);
    free(exact_shift);
    return PREFOS_STATUS_OK;

fallback:
    free(row_nnz);
    free(row_map);
    free(shifts);
    free(exact_shift);
    return PREFOS_STATUS_OK;
}

typedef struct
{
    const PreFOSPresolver *presolver;
    PreFOSCsrMatrix *target;
    const int *row_map;
    int *row_nnz;
    double *shifts;
    size_t row_begin;
    size_t row_end;
    int write_entries;
    PreFOSStatus status;
} PreFOSMatrixCompactionCpuTask;

static void *run_matrix_compaction_cpu_task(void *argument)
{
    PreFOSMatrixCompactionCpuTask *task =
        (PreFOSMatrixCompactionCpuTask *) argument;
    const PreFOSPresolver *presolver = task->presolver;
    const PreFOSCsrMatrix *source = &presolver->original.A;
    size_t row;

    task->status = PREFOS_STATUS_OK;
    for (row = task->row_begin; row < task->row_end; ++row)
    {
        int position;
        if (task->write_entries)
        {
            int reduced_row = task->row_map[row];
            int write, end;
            if (reduced_row < 0) continue;
            write = task->target->row_pointers[reduced_row];
            end = task->target->row_pointers[reduced_row + 1];
            for (position = source->row_pointers[row];
                 position < source->row_pointers[row + 1]; ++position)
            {
                int mapped = presolver->original_to_reduced[
                    source->column_indices[position]];
                double value = source->values[position];
                if (mapped < 0 || value == 0.0) continue;
                if (write >= end)
                {
                    task->status = PREFOS_STATUS_NUMERICAL_ERROR;
                    return NULL;
                }
                task->target->column_indices[write] = mapped;
                task->target->values[write++] = value;
            }
            if (write != end)
            {
                task->status = PREFOS_STATUS_NUMERICAL_ERROR;
                return NULL;
            }
            continue;
        }

        task->row_nnz[row] = -1;
        task->shifts[row] = 0.0;
        if (presolver->remove_rows[row]) continue;
        {
            size_t remaining = 0;
            for (position = source->row_pointers[row];
                 position < source->row_pointers[row + 1]; ++position)
            {
                int column = source->column_indices[position];
                double value = source->values[position];
                if (presolver->is_fixed[column])
                {
                    if (!prefos_internal_safe_add_product(
                            &task->shifts[row], value,
                            presolver->fixed_values[column]))
                    {
                        task->status =
                            PREFOS_STATUS_NUMERICAL_ERROR;
                        return NULL;
                    }
                }
                else if (
                    presolver->original_to_reduced[column] >= 0 &&
                    value != 0.0)
                    ++remaining;
            }
            if (remaining > (size_t) INT_MAX)
            {
                task->status = PREFOS_STATUS_OUT_OF_MEMORY;
                return NULL;
            }
            task->row_nnz[row] = (int) remaining;
        }
    }
    return NULL;
}

static PreFOSStatus run_matrix_compaction_cpu_tasks(
    PreFOSMatrixCompactionCpuTask *tasks, int n_threads)
{
    PreFOSThread threads[PREFOS_MATRIX_COMPACTION_CPU_THREADS - 1];
    unsigned char started[
        PREFOS_MATRIX_COMPACTION_CPU_THREADS - 1] = {0};
    int thread;

    if (n_threads < 1) n_threads = 1;
    if (n_threads > PREFOS_MATRIX_COMPACTION_CPU_THREADS)
        n_threads = PREFOS_MATRIX_COMPACTION_CPU_THREADS;
    for (thread = 1; thread < n_threads; ++thread)
        if (prefos_thread_create(
                &threads[thread - 1],
                run_matrix_compaction_cpu_task,
                &tasks[thread]) == 0)
            started[thread - 1] = 1;
        else
            run_matrix_compaction_cpu_task(&tasks[thread]);
    run_matrix_compaction_cpu_task(&tasks[0]);
    for (thread = 1; thread < n_threads; ++thread)
        if (started[thread - 1])
            (void) prefos_thread_join(&threads[thread - 1]);
    for (thread = 0; thread < n_threads; ++thread)
        if (tasks[thread].status != PREFOS_STATUS_OK)
            return tasks[thread].status;
    return PREFOS_STATUS_OK;
}

static size_t matrix_compaction_partition_end(
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

static PreFOSStatus compact_a_without_substitutions_cpu_parallel(
    PreFOSPresolver *presolver, int *used)
{
    const PreFOSProblemData *source = &presolver->original;
    PreFOSPresolvedProblem *target = &presolver->reduced;
    PreFOSMatrixCompactionCpuTask
        tasks[PREFOS_MATRIX_COMPACTION_CPU_THREADS];
    int *row_nnz = NULL;
    int *row_map = NULL;
    double *shifts = NULL;
    size_t row, kept_rows = 0, nnz = 0;
    size_t removed_empty_rows = 0;
    size_t base_nnz, extra_nnz, row_begin = 0;
    int n_threads = prefos_cpu_thread_limit(
        PREFOS_MATRIX_COMPACTION_CPU_THREADS);
    int thread;
    PreFOSStatus status;

    *used = 0;
    if (n_threads <= 1 ||
        source->A.nnz <
            PREFOS_PARALLEL_MATRIX_COMPACTION_MIN_NNZ ||
        source->A.rows < 2)
        return PREFOS_STATUS_OK;
    if ((size_t) n_threads > source->A.rows)
        n_threads = (int) source->A.rows;

    row_nnz = (int *) prefos_internal_alloc_array(
        source->A.rows, sizeof(int));
    row_map = (int *) prefos_internal_alloc_array(
        source->A.rows, sizeof(int));
    shifts = (double *) prefos_internal_alloc_array(
        source->A.rows, sizeof(double));
    if (!row_nnz || !row_map || !shifts)
    {
        free(row_nnz);
        free(row_map);
        free(shifts);
        return PREFOS_STATUS_OUT_OF_MEMORY;
    }

    base_nnz = source->A.nnz / (size_t) n_threads;
    extra_nnz = source->A.nnz % (size_t) n_threads;
    for (thread = 0; thread < n_threads; ++thread)
    {
        size_t row_end;
        if (thread + 1 == n_threads)
            row_end = source->A.rows;
        else
        {
            size_t completed_parts = (size_t) thread + 1U;
            size_t target_nnz =
                base_nnz * completed_parts +
                (completed_parts < extra_nnz
                     ? completed_parts
                     : extra_nnz);
            size_t maximum_row =
                source->A.rows -
                (size_t) (n_threads - thread - 1);
            row_end = matrix_compaction_partition_end(
                &source->A, target_nnz,
                row_begin + 1U, maximum_row);
        }
        tasks[thread] = (PreFOSMatrixCompactionCpuTask){
            presolver, &target->A, row_map, row_nnz, shifts,
            row_begin, row_end, 0, PREFOS_STATUS_OK};
        row_begin = row_end;
    }
    status = run_matrix_compaction_cpu_tasks(tasks, n_threads);
    if (status != PREFOS_STATUS_OK) goto cleanup;

    for (row = 0; row < source->A.rows; ++row)
    {
        double lower, upper;
        row_map[row] = -1;
        presolver->original_to_reduced_rows[row] = -1;
        if (row_nnz[row] < 0) continue;
        lower =
            presolver->working_constraint_lower[row] - shifts[row];
        upper =
            presolver->working_constraint_upper[row] - shifts[row];
        if (isnan(lower) || isnan(upper))
        {
            status = PREFOS_STATUS_NUMERICAL_ERROR;
            goto cleanup;
        }
        if (row_nnz[row] == 0 &&
            presolver->settings.remove_empty_rows)
        {
            if (lower > presolver->settings.feasibility_tolerance ||
                upper < -presolver->settings.feasibility_tolerance)
            {
                status = PREFOS_STATUS_PRIMAL_INFEASIBLE;
                goto cleanup;
            }
            ++removed_empty_rows;
            continue;
        }
        if ((size_t) row_nnz[row] > (size_t) INT_MAX - nnz)
        {
            status = PREFOS_STATUS_OUT_OF_MEMORY;
            goto cleanup;
        }
        row_map[row] = (int) kept_rows++;
        nnz += (size_t) row_nnz[row];
    }

    target->A.rows = kept_rows;
    target->A.cols = target->n;
    target->A.nnz = nnz;
    target->A.row_pointers = (int *)
        prefos_internal_alloc_array(kept_rows + 1, sizeof(int));
    target->A.values =
        (double *) prefos_internal_alloc_array(nnz, sizeof(double));
    target->A.column_indices =
        (int *) prefos_internal_alloc_array(nnz, sizeof(int));
    target->constraint_lower =
        (double *) prefos_internal_alloc_array(
            kept_rows, sizeof(double));
    target->constraint_upper =
        (double *) prefos_internal_alloc_array(
            kept_rows, sizeof(double));
    if (!target->A.row_pointers ||
        (nnz > 0 &&
         (!target->A.values || !target->A.column_indices)) ||
        (kept_rows > 0 &&
         (!target->constraint_lower ||
          !target->constraint_upper)))
    {
        status = PREFOS_STATUS_OUT_OF_MEMORY;
        goto cleanup_target;
    }

    kept_rows = 0;
    nnz = 0;
    target->A.row_pointers[0] = 0;
    for (row = 0; row < source->A.rows; ++row)
    {
        double lower, upper;
        if (row_map[row] < 0) continue;
        lower =
            presolver->working_constraint_lower[row] - shifts[row];
        upper =
            presolver->working_constraint_upper[row] - shifts[row];
        presolver->original_to_reduced_rows[row] =
            (int) kept_rows;
        target->constraint_lower[kept_rows] = lower;
        target->constraint_upper[kept_rows] = upper;
        nnz += (size_t) row_nnz[row];
        target->A.row_pointers[++kept_rows] = (int) nnz;
    }

    for (thread = 0; thread < n_threads; ++thread)
        tasks[thread].write_entries = 1;
    status = run_matrix_compaction_cpu_tasks(tasks, n_threads);
    if (status != PREFOS_STATUS_OK) goto cleanup_target;

    presolver->stats.removed_empty_rows += removed_empty_rows;
    *used = 1;
    status = PREFOS_STATUS_OK;
    goto cleanup;

cleanup_target:
    prefos_internal_free_csr(&target->A);
    free(target->constraint_lower);
    free(target->constraint_upper);
    target->constraint_lower = NULL;
    target->constraint_upper = NULL;
cleanup:
    free(row_nnz);
    free(row_map);
    free(shifts);
    return status;
}

static PreFOSStatus compact_a_without_substitutions(PreFOSPresolver *presolver)
{
    const PreFOSProblemData *source = &presolver->original;
    PreFOSPresolvedProblem *target = &presolver->reduced;
    unsigned char *keep_row;
    double *shifts;
    size_t row, kept_rows = 0, nnz = 0;
    int write = 0;
    int used_gpu = 0;
    int attempted_gpu = 0;
    int used_cpu_parallel = 0;
    PreFOSStatus gpu_status;
    int used_rows_only = 0;
    gpu_status = compact_a_rows_only(presolver, &used_rows_only);
    if (gpu_status != PREFOS_STATUS_OK) return gpu_status;
    if (used_rows_only) return PREFOS_STATUS_OK;
    gpu_status =
        compact_a_without_substitutions_gpu(presolver, &used_gpu,
                                            &attempted_gpu);
    if (gpu_status != PREFOS_STATUS_OK) return gpu_status;
    if (used_gpu) return PREFOS_STATUS_OK;
    if (attempted_gpu)
        ++presolver->stats.matrix_compaction_gpu_fallbacks;
    gpu_status = compact_a_without_substitutions_cpu_parallel(
        presolver, &used_cpu_parallel);
    if (gpu_status != PREFOS_STATUS_OK) return gpu_status;
    if (used_cpu_parallel) return PREFOS_STATUS_OK;

    keep_row = (unsigned char *) calloc(source->A.rows, sizeof(unsigned char));
    shifts = (double *) calloc(source->A.rows, sizeof(double));
    if (source->A.rows > 0 && (!keep_row || !shifts))
    {
        free(keep_row);
        free(shifts);
        return PREFOS_STATUS_OUT_OF_MEMORY;
    }
    for (row = 0; row < source->A.rows; ++row)
    {
        int p;
        size_t remaining = 0;
        double lower, upper;
        if (presolver->remove_rows[row]) continue;
        for (p = source->A.row_pointers[row]; p < source->A.row_pointers[row + 1];
             ++p)
        {
            int column = source->A.column_indices[p];
            if (presolver->is_fixed[column])
            {
                if (!prefos_internal_safe_add_product(&shifts[row], source->A.values[p],
                                                   presolver->fixed_values[column]))
                {
                    free(keep_row);
                    free(shifts);
                    return PREFOS_STATUS_NUMERICAL_ERROR;
                }
            }
            else if (presolver->original_to_reduced[column] >= 0 &&
                     source->A.values[p] != 0.0)
                ++remaining;
        }
        lower = presolver->working_constraint_lower[row] - shifts[row];
        upper = presolver->working_constraint_upper[row] - shifts[row];
        if (isnan(lower) || isnan(upper))
        {
            free(keep_row);
            free(shifts);
            return PREFOS_STATUS_NUMERICAL_ERROR;
        }
        if (remaining == 0 && presolver->settings.remove_empty_rows)
        {
            if (lower > presolver->settings.feasibility_tolerance ||
                upper < -presolver->settings.feasibility_tolerance)
            {
                if (getenv("PREFOS_TRACE_MATRIX_COMPACTION"))
                    fprintf(
                        stderr,
                        "PreFOS compaction empty-row conflict row=%zu "
                        "lower=%.17g upper=%.17g shift=%.17g\n",
                        row, lower, upper, shifts[row]);
                free(keep_row);
                free(shifts);
                return PREFOS_STATUS_PRIMAL_INFEASIBLE;
            }
            ++presolver->stats.removed_empty_rows;
        }
        else
        {
            keep_row[row] = 1;
            ++kept_rows;
            nnz += remaining;
        }
    }

    target->A.rows = kept_rows;
    target->A.cols = target->n;
    target->A.nnz = nnz;
    target->A.row_pointers = (int *) calloc(kept_rows + 1, sizeof(int));
    target->A.values = (double *) prefos_internal_alloc_array(nnz, sizeof(double));
    target->A.column_indices = (int *) prefos_internal_alloc_array(nnz, sizeof(int));
    target->constraint_lower =
        (double *) prefos_internal_alloc_array(kept_rows, sizeof(double));
    target->constraint_upper =
        (double *) prefos_internal_alloc_array(kept_rows, sizeof(double));
    if (!target->A.row_pointers ||
        (nnz > 0 && (!target->A.values || !target->A.column_indices)) ||
        (kept_rows > 0 && (!target->constraint_lower || !target->constraint_upper)))
    {
        free(keep_row);
        free(shifts);
        return PREFOS_STATUS_OUT_OF_MEMORY;
    }

    kept_rows = 0;
    for (row = 0; row < source->A.rows; ++row)
    {
        int p;
        if (!keep_row[row]) continue;
        presolver->original_to_reduced_rows[row] = (int) kept_rows;
        target->A.row_pointers[kept_rows] = write;
        target->constraint_lower[kept_rows] =
            presolver->working_constraint_lower[row] - shifts[row];
        target->constraint_upper[kept_rows] =
            presolver->working_constraint_upper[row] - shifts[row];
        for (p = source->A.row_pointers[row]; p < source->A.row_pointers[row + 1];
             ++p)
        {
            int mapped = presolver->original_to_reduced[source->A.column_indices[p]];
            if (mapped >= 0 && source->A.values[p] != 0.0)
            {
                target->A.values[write] = source->A.values[p];
                target->A.column_indices[write] = mapped;
                ++write;
            }
        }
        ++kept_rows;
    }
    target->A.row_pointers[kept_rows] = write;
    free(keep_row);
    free(shifts);
    return PREFOS_STATUS_OK;
}

static PreFOSStatus accumulate_transformed_column(const PreFOSPresolver *presolver,
                                               int column, double coefficient,
                                               size_t depth, double *row_values,
                                               int *row_marks, int *touched_columns,
                                               size_t *n_touched, double *shift)
{
    int mapped;
    size_t term;

    if (column < 0 || (size_t) column >= presolver->original.n ||
        depth > PREFOS_MAX_SUBSTITUTION_DEPTH)
        return PREFOS_STATUS_NUMERICAL_ERROR;
    if (presolver->is_fixed[column])
        return prefos_internal_safe_add_product(shift, coefficient,
                                             presolver->fixed_values[column])
                   ? PREFOS_STATUS_OK
                   : PREFOS_STATUS_NUMERICAL_ERROR;
    if (presolver->is_parallel_removed[column]) return PREFOS_STATUS_OK;
    if (presolver->is_substituted[column])
    {
        size_t start = presolver->substitution_term_start[column];
        size_t count = presolver->substitution_term_count[column];
        if (count == 0 || start > presolver->n_substitution_terms ||
            count > presolver->n_substitution_terms - start ||
            !prefos_internal_safe_add_product(shift, coefficient,
                                           presolver->substitution_constant[column]))
            return PREFOS_STATUS_NUMERICAL_ERROR;
        for (term = 0; term < count; ++term)
        {
            double propagated;
            PreFOSStatus status;
            if (!prefos_internal_safe_product(
                    coefficient, presolver->substitution_scales[start + term],
                    &propagated))
                return PREFOS_STATUS_NUMERICAL_ERROR;
            status = accumulate_transformed_column(
                presolver, presolver->substitution_targets[start + term], propagated,
                depth + 1, row_values, row_marks, touched_columns, n_touched, shift);
            if (status != PREFOS_STATUS_OK) return status;
        }
        return PREFOS_STATUS_OK;
    }
    mapped = presolver->original_to_reduced[column];
    if (mapped < 0) return PREFOS_STATUS_NUMERICAL_ERROR;
    if (row_marks[mapped] < 0)
    {
        row_marks[mapped] = 1;
        row_values[mapped] = coefficient;
        touched_columns[(*n_touched)++] = mapped;
    }
    else if (!prefos_internal_safe_add_product(&row_values[mapped], 1.0, coefficient))
        return PREFOS_STATUS_NUMERICAL_ERROR;
    return PREFOS_STATUS_OK;
}

static PreFOSStatus accumulate_transformed_a_row(
    const PreFOSPresolver *presolver, const PreFOSCsrMatrix *matrix,
    size_t row, double *row_values, int *row_marks, int *touched_columns,
    size_t *n_touched, double *shift)
{
    int p;

    *n_touched = 0;
    *shift = 0.0;
    for (p = matrix->row_pointers[row];
         p < matrix->row_pointers[row + 1]; ++p)
    {
        int column = matrix->column_indices[p];
        double coefficient = matrix->values[p];
        PreFOSStatus status;
        if (coefficient == 0.0) continue;
        if (!prefos_internal_term_is_active_in_row(
                presolver, row, column))
            continue;
        status = accumulate_transformed_column(presolver, column, coefficient, 0,
                                               row_values, row_marks,
                                               touched_columns, n_touched, shift);
        if (status != PREFOS_STATUS_OK) return status;
    }
    return PREFOS_STATUS_OK;
}

static void clear_transformed_a_row(double *row_values, int *row_marks,
                                    const int *touched_columns, size_t n_touched)
{
    size_t position;
    for (position = 0; position < n_touched; ++position)
    {
        int column = touched_columns[position];
        row_values[column] = 0.0;
        row_marks[column] = -1;
    }
}

static void build_reduced_activity_bounds(
    const PreFOSPresolver *presolver, double *lower, double *upper)
{
    size_t column;
    for (column = 0; column < presolver->reduced.n; ++column)
    {
        lower[column] = -INFINITY;
        upper[column] = INFINITY;
    }
    for (column = 0; column < presolver->original.n; ++column)
    {
        int mapped = presolver->original_to_reduced[column];
        int box;
        if (mapped < 0) continue;
        lower[mapped] = presolver->propagation_lower[column];
        upper[mapped] = presolver->propagation_upper[column];
        box = presolver->variable_to_box[column];
        if (box >= 0)
        {
            lower[mapped] = presolver->working_box_lower[box];
            upper[mapped] = presolver->working_box_upper[box];
        }
    }
}

static PreFOSStatus classify_transformed_row(
    const double *row_values, const int *touched_columns,
    size_t n_touched, const double *lower_bounds,
    const double *upper_bounds, double lower, double upper,
    PresolveLinearRowState *state)
{
    PresolveLinearActivity activity;
    long double minimum = 0.0L, maximum = 0.0L;
    size_t position;

    memset(&activity, 0, sizeof(activity));
    for (position = 0; position < n_touched; ++position)
    {
        int column = touched_columns[position];
        double coefficient = row_values[column];
        double minimum_bound, maximum_bound;
        if (coefficient == 0.0) continue;
        minimum_bound =
            coefficient > 0.0 ? lower_bounds[column]
                              : upper_bounds[column];
        maximum_bound =
            coefficient > 0.0 ? upper_bounds[column]
                              : lower_bounds[column];
        ++activity.n_nonzeros;
        if (isfinite(minimum_bound))
        {
            minimum = presolve_internal_directed_add(
                minimum,
                presolve_internal_directed_product(
                    coefficient, minimum_bound, 1),
                1);
            if (!isfinite(minimum))
                return PREFOS_STATUS_OK;
        }
        else
            ++activity.n_infinite_min;
        if (isfinite(maximum_bound))
        {
            maximum = presolve_internal_directed_add(
                maximum,
                presolve_internal_directed_product(
                    coefficient, maximum_bound, 0),
                0);
            if (!isfinite(maximum))
                return PREFOS_STATUS_OK;
        }
        else
            ++activity.n_infinite_max;
    }
    activity.finite_min =
        presolve_internal_directed_double_cast(minimum, 1);
    activity.finite_max =
        presolve_internal_directed_double_cast(maximum, 0);
    if (!isfinite(activity.finite_min) ||
        !isfinite(activity.finite_max))
        return PREFOS_STATUS_OK;
    *state = presolve_internal_classify_linear_row(
        &activity, lower, upper,
        0.0, 0.0);
    return (*state & PRESOLVE_ROW_INFEASIBLE) != 0
               ? PREFOS_STATUS_PRIMAL_INFEASIBLE
               : PREFOS_STATUS_OK;
}

static PreFOSStatus reserve_compacted_a_entries(PreFOSCsrMatrix *matrix,
                                                size_t *capacity,
                                                size_t required)
{
    size_t new_capacity;
    double *new_values;
    int *new_columns;

    if (required <= *capacity) return PREFOS_STATUS_OK;
    if (required > (size_t) INT_MAX) return PREFOS_STATUS_OUT_OF_MEMORY;
    new_capacity = *capacity == 0 ? 1024 : *capacity;
    while (new_capacity < required)
    {
        size_t grown = new_capacity + new_capacity / 2 + 1;
        if (grown > (size_t) INT_MAX || grown <= new_capacity)
        {
            new_capacity = (size_t) INT_MAX;
            break;
        }
        new_capacity = grown;
    }
    if (new_capacity < required) return PREFOS_STATUS_OUT_OF_MEMORY;

    new_values =
        (double *) realloc(matrix->values, new_capacity * sizeof(double));
    if (!new_values) return PREFOS_STATUS_OUT_OF_MEMORY;
    matrix->values = new_values;
    new_columns =
        (int *) realloc(matrix->column_indices, new_capacity * sizeof(int));
    if (!new_columns) return PREFOS_STATUS_OUT_OF_MEMORY;
    matrix->column_indices = new_columns;
    *capacity = new_capacity;
    return PREFOS_STATUS_OK;
}

static void materialized_column_bounds(
    const PreFOSPresolver *presolver, int column,
    double *lower, double *upper)
{
    int box = presolver->variable_to_box[column];
    *lower = presolver->propagation_lower[column];
    *upper = presolver->propagation_upper[column];
    if (box >= 0)
    {
        *lower = presolver->working_box_lower[box];
        *upper = presolver->working_box_upper[box];
    }
}

static PreFOSStatus compact_current_working_matrix(
    PreFOSPresolver *presolver, PreFOSWorkingMatrix *working,
    int *used)
{
    PreFOSPresolvedProblem *target = &presolver->reduced;
    PreFOSCsrMatrix *matrix = &working->matrix;
    int *row_pointers = NULL;
    size_t row, kept_rows = 0, write = 0;
    size_t removed_empty_rows = 0;
    int classify_changed_rows =
        presolver->settings.remove_redundant_rows &&
        !presolver->scalar_redundancy_completed;
    int exhaustive_row_classification =
        presolver->settings.linear_propagation_max_stale_rounds == 0;

    *used = 0;
    if (matrix->rows != presolver->original.A.rows ||
        matrix->cols != presolver->original.n ||
        !working->lower || !working->upper)
        return PREFOS_STATUS_OK;

    row_pointers = (int *) prefos_internal_alloc_array(
        matrix->rows + 1, sizeof(int));
    if (!row_pointers) return PREFOS_STATUS_OUT_OF_MEMORY;
    row_pointers[0] = 0;

    for (row = 0; row < matrix->rows; ++row)
    {
        PresolveLinearActivity activity;
        PresolveLinearRowState row_state = PRESOLVE_ROW_FEASIBLE;
        size_t row_write = write;
        int position;
        double lower, upper;
        long double minimum = 0.0L, maximum = 0.0L;
        int activity_valid = 1;
        int classify_row =
            classify_changed_rows &&
            (exhaustive_row_classification ||
             (presolver->rows_require_materialization &&
              presolver->rows_require_materialization[row]));

        presolver->original_to_reduced_rows[row] = -1;
        if (presolver->remove_rows[row]) continue;
        memset(&activity, 0, sizeof(activity));
        lower = working->lower[row];
        upper = working->upper[row];

        for (position = matrix->row_pointers[row];
             position < matrix->row_pointers[row + 1]; ++position)
        {
            int column = matrix->column_indices[position];
            int mapped;
            double coefficient = matrix->values[position];
            double column_lower, column_upper;
            double minimum_bound, maximum_bound;
            if (coefficient == 0.0) continue;
            if (column < 0 ||
                (size_t) column >= presolver->original.n)
            {
                free(row_pointers);
                return PREFOS_STATUS_NUMERICAL_ERROR;
            }
            mapped = presolver->original_to_reduced[column];
            if (mapped < 0 || (size_t) mapped >= target->n)
            {
                free(row_pointers);
                return PREFOS_STATUS_NUMERICAL_ERROR;
            }

            matrix->values[write] = coefficient;
            matrix->column_indices[write] = mapped;
            ++write;
            if (!classify_row)
                continue;

            materialized_column_bounds(
                presolver, column, &column_lower, &column_upper);
            minimum_bound =
                coefficient > 0.0 ? column_lower : column_upper;
            maximum_bound =
                coefficient > 0.0 ? column_upper : column_lower;
            ++activity.n_nonzeros;
            if (isfinite(minimum_bound))
            {
                minimum = presolve_internal_directed_add(
                    minimum,
                    presolve_internal_directed_product(
                        coefficient, minimum_bound, 1),
                    1);
                if (!isfinite(minimum)) activity_valid = 0;
            }
            else
                ++activity.n_infinite_min;
            if (isfinite(maximum_bound))
            {
                maximum = presolve_internal_directed_add(
                    maximum,
                    presolve_internal_directed_product(
                        coefficient, maximum_bound, 0),
                    0);
                if (!isfinite(maximum)) activity_valid = 0;
            }
            else
                ++activity.n_infinite_max;
        }

        if (classify_row &&
            write > row_write && activity_valid)
        {
            activity.finite_min =
                presolve_internal_directed_double_cast(minimum, 1);
            activity.finite_max =
                presolve_internal_directed_double_cast(maximum, 0);
            if (isfinite(activity.finite_min) &&
                isfinite(activity.finite_max))
            {
                row_state = presolve_internal_classify_linear_row(
                    &activity, lower, upper, 0.0, 0.0);
                if ((row_state & PRESOLVE_ROW_INFEASIBLE) != 0)
                {
                    free(row_pointers);
                    return PREFOS_STATUS_PRIMAL_INFEASIBLE;
                }
            }
        }
        if ((row_state & PRESOLVE_ROW_LOWER_REDUNDANT) != 0 &&
            (row_state & PRESOLVE_ROW_UPPER_REDUNDANT) != 0)
        {
            write = row_write;
            prefos_internal_mark_removed_row(presolver, row);
            ++presolver->stats.removed_redundant_rows;
            continue;
        }
        if ((row_state & PRESOLVE_ROW_LOWER_REDUNDANT) != 0 &&
            isfinite(lower))
        {
            lower = -INFINITY;
            presolver->working_constraint_lower[row] = -INFINITY;
            ++presolver->stats.removed_redundant_row_lower_sides;
        }
        if ((row_state & PRESOLVE_ROW_UPPER_REDUNDANT) != 0 &&
            isfinite(upper))
        {
            upper = INFINITY;
            presolver->working_constraint_upper[row] = INFINITY;
            ++presolver->stats.removed_redundant_row_upper_sides;
        }
        if (write == row_write &&
            presolver->settings.remove_empty_rows)
        {
            if (lower > presolver->settings.feasibility_tolerance ||
                upper < -presolver->settings.feasibility_tolerance)
            {
                free(row_pointers);
                return PREFOS_STATUS_PRIMAL_INFEASIBLE;
            }
            ++removed_empty_rows;
            continue;
        }

        presolver->original_to_reduced_rows[row] = (int) kept_rows;
        working->lower[kept_rows] = lower;
        working->upper[kept_rows] = upper;
        row_pointers[++kept_rows] = (int) write;
    }

    free(matrix->row_pointers);
    matrix->row_pointers = row_pointers;
    target->A = *matrix;
    target->A.rows = kept_rows;
    target->A.cols = target->n;
    target->A.nnz = write;
    target->constraint_lower = working->lower;
    target->constraint_upper = working->upper;
    presolver->stats.removed_empty_rows += removed_empty_rows;

    memset(matrix, 0, sizeof(*matrix));
    working->lower = NULL;
    working->upper = NULL;
    free(working->column_counts);
    working->column_counts = NULL;
    *used = 1;
    return PREFOS_STATUS_OK;
}

PreFOSStatus prefos_internal_compact_a(PreFOSPresolver *presolver)
{
    const PreFOSCsrMatrix *source_matrix;
    const double *source_lower;
    const double *source_upper;
    PreFOSPresolvedProblem *target = &presolver->reduced;
    double *row_values;
    double *activity_lower;
    double *activity_upper;
    int *row_marks;
    int *touched_columns;
    size_t row, kept_rows = 0, write = 0, entry_capacity;
    size_t removed_empty_rows = 0;
    int exhaustive_row_classification =
        presolver->settings.linear_propagation_max_stale_rounds == 0;
    PreFOSStatus status = PREFOS_STATUS_OK;
    PreFOSWorkingMatrix current;
    int used_current = 0;

    memset(&current, 0, sizeof(current));
    if (getenv("PREFOS_TRACE_MATRIX_COMPACTION"))
        fprintf(
            stderr,
            "PreFOS compaction cache valid=%d fixed=%zu/%zu "
            "columns=%zu/%zu scalar_complete=%d\n",
            presolver->cached_working_matrix_valid,
            presolver->cached_working_fixed_column_epoch,
            presolver->fixed_column_epoch,
            presolver->cached_working_column_transformations,
            presolver->transformations.n_column_transformations,
            presolver->scalar_redundancy_completed);
    if (!presolver->settings.structural_reductions_gpu &&
        prefos_internal_take_current_working_matrix_cache(
            presolver, &current, NULL))
    {
        status = compact_current_working_matrix(
            presolver, &current, &used_current);
        if (getenv("PREFOS_TRACE_MATRIX_COMPACTION"))
            fprintf(
                stderr,
                "PreFOS compaction current used=%d status=%d\n",
                used_current, (int) status);
        if (status != PREFOS_STATUS_OK || used_current)
        {
            prefos_internal_free_working_matrix(&current);
            return status;
        }
        prefos_internal_store_working_matrix_cache(
            presolver, &current);
    }

    if (!presolver->materialized_row_updates_require_cache &&
        (presolver->stats.substituted_free_variables == 0 ||
         presolver->n_rows_require_materialization == 0))
    {
        status = compact_a_without_substitutions(presolver);
        prefos_internal_clear_working_matrix_cache(presolver);
        return status;
    }
    prefos_internal_get_working_matrix_source(
        presolver, &source_matrix, &source_lower, &source_upper);
    if (source_matrix->rows != presolver->original.A.rows)
    {
        prefos_internal_clear_working_matrix_cache(presolver);
        return PREFOS_STATUS_NUMERICAL_ERROR;
    }

    row_values = (double *) calloc(target->n, sizeof(double));
    activity_lower =
        (double *) prefos_internal_alloc_array(target->n, sizeof(double));
    activity_upper =
        (double *) prefos_internal_alloc_array(target->n, sizeof(double));
    row_marks = (int *) prefos_internal_alloc_array(target->n, sizeof(int));
    touched_columns = (int *) prefos_internal_alloc_array(target->n, sizeof(int));
    target->A.row_pointers = (int *) prefos_internal_alloc_array(
        source_matrix->rows + 1, sizeof(int));
    target->constraint_lower = (double *) prefos_internal_alloc_array(
        source_matrix->rows, sizeof(double));
    target->constraint_upper = (double *) prefos_internal_alloc_array(
        source_matrix->rows, sizeof(double));
    entry_capacity = source_matrix->nnz;
    target->A.values =
        (double *) prefos_internal_alloc_array(entry_capacity, sizeof(double));
    target->A.column_indices =
        (int *) prefos_internal_alloc_array(entry_capacity, sizeof(int));
    if ((target->n > 0 &&
         (!row_values || !activity_lower || !activity_upper ||
          !row_marks || !touched_columns)) ||
        !target->A.row_pointers ||
        (source_matrix->rows > 0 &&
         (!target->constraint_lower || !target->constraint_upper)) ||
        (entry_capacity > 0 &&
         (!target->A.values || !target->A.column_indices)))
    {
        status = PREFOS_STATUS_OUT_OF_MEMORY;
        goto compact_failure;
    }
    for (row = 0; row < target->n; ++row) row_marks[row] = -1;
    build_reduced_activity_bounds(
        presolver, activity_lower, activity_upper);

    for (row = 0; row < source_matrix->rows; ++row)
    {
        size_t remaining = 0, n_touched = 0, position;
        double shift, lower, upper;
        presolver->original_to_reduced_rows[row] = -1;
        if (presolver->remove_rows[row]) continue;
        if (!exhaustive_row_classification &&
            (!presolver->rows_require_materialization ||
             !presolver->rows_require_materialization[row]))
        {
            int begin = source_matrix->row_pointers[row];
            int end = source_matrix->row_pointers[row + 1];
            size_t row_write = write;
            int entry;
            status = reserve_compacted_a_entries(
                &target->A, &entry_capacity,
                write + (size_t) (end - begin));
            if (status != PREFOS_STATUS_OK)
                goto compact_failure;
            shift = 0.0;
            for (entry = begin; entry < end; ++entry)
            {
                int column = source_matrix->column_indices[entry];
                int mapped;
                double coefficient = source_matrix->values[entry];
                if (coefficient == 0.0 ||
                    !prefos_internal_term_is_active_in_row(
                        presolver, row, column))
                    continue;
                if (presolver->is_fixed[column])
                {
                    if (!prefos_internal_safe_add_product(
                            &shift, coefficient,
                            presolver->fixed_values[column]))
                    {
                        status = PREFOS_STATUS_NUMERICAL_ERROR;
                        goto compact_failure;
                    }
                    continue;
                }
                mapped = presolver->original_to_reduced[column];
                if (mapped < 0) continue;
                target->A.values[write] = coefficient;
                target->A.column_indices[write] = mapped;
                ++write;
            }
            lower = source_lower[row] - shift;
            upper = source_upper[row] - shift;
            if (isnan(lower) || isnan(upper))
            {
                status = PREFOS_STATUS_NUMERICAL_ERROR;
                goto compact_failure;
            }
            remaining = write - row_write;
            if (remaining == 0 &&
                presolver->settings.remove_empty_rows)
            {
                write = row_write;
                if (lower >
                        presolver->settings.feasibility_tolerance ||
                    upper <
                        -presolver->settings.feasibility_tolerance)
                {
                    status = PREFOS_STATUS_PRIMAL_INFEASIBLE;
                    goto compact_failure;
                }
                ++removed_empty_rows;
                continue;
            }
            presolver->original_to_reduced_rows[row] =
                (int) kept_rows;
            target->A.row_pointers[kept_rows] =
                (int) row_write;
            target->constraint_lower[kept_rows] = lower;
            target->constraint_upper[kept_rows] = upper;
            ++kept_rows;
            continue;
        }
        status = accumulate_transformed_a_row(
            presolver, source_matrix, row, row_values, row_marks,
            touched_columns, &n_touched, &shift);
        if (status != PREFOS_STATUS_OK) goto compact_failure;
        for (position = 0; position < n_touched; ++position)
            if (row_values[touched_columns[position]] != 0.0) ++remaining;
        lower = source_lower[row] - shift;
        upper = source_upper[row] - shift;
        if (isnan(lower) || isnan(upper))
        {
            clear_transformed_a_row(row_values, row_marks, touched_columns,
                                    n_touched);
            status = PREFOS_STATUS_NUMERICAL_ERROR;
            goto compact_failure;
        }
        /*
         * Exact-form rows were already classified by the scalar activity
         * passes. Rechecking every reduced nonzero here is particularly
         * costly because classification uses directed arithmetic. In the
         * default budgeted mode, only rows whose algebra changed through a
         * substitution need another check; exhaustive settings retain the
         * full final verification.
         */
        if (presolver->settings.remove_redundant_rows &&
            remaining > 0 &&
            (exhaustive_row_classification ||
             !presolver->rows_require_materialization ||
             presolver->rows_require_materialization[row]))
        {
            PresolveLinearRowState row_state =
                PRESOLVE_ROW_FEASIBLE;
            int lower_redundant, upper_redundant;
            status = classify_transformed_row(
                row_values, touched_columns, n_touched,
                activity_lower, activity_upper, lower, upper,
                &row_state);
            if (status != PREFOS_STATUS_OK)
            {
                if (getenv("PREFOS_TRACE_MATRIX_COMPACTION"))
                {
                    fprintf(
                        stderr,
                        "PreFOS transformed compaction conflict row=%zu "
                        "lower=%.17g upper=%.17g shift=%.17g "
                        "remaining=%zu\n",
                        row, lower, upper, shift, remaining);
                    for (position = 0; position < n_touched; ++position)
                    {
                        int mapped = touched_columns[position];
                        double coefficient = row_values[mapped];
                        if (coefficient == 0.0) continue;
                        fprintf(
                            stderr,
                            "  mapped_column=%d coefficient=%.17g "
                            "bounds=[%.17g,%.17g]\n",
                            mapped, coefficient,
                            activity_lower[mapped],
                            activity_upper[mapped]);
                    }
                }
                clear_transformed_a_row(
                    row_values, row_marks, touched_columns,
                    n_touched);
                goto compact_failure;
            }
            lower_redundant =
                (row_state & PRESOLVE_ROW_LOWER_REDUNDANT) != 0;
            upper_redundant =
                (row_state & PRESOLVE_ROW_UPPER_REDUNDANT) != 0;
            if (lower_redundant && upper_redundant)
            {
                prefos_internal_mark_removed_row(presolver, row);
                ++presolver->stats.removed_redundant_rows;
                clear_transformed_a_row(
                    row_values, row_marks, touched_columns,
                    n_touched);
                continue;
            }
            if (lower_redundant && isfinite(lower))
            {
                lower = -INFINITY;
                presolver->working_constraint_lower[row] =
                    -INFINITY;
                ++presolver->stats.removed_redundant_row_lower_sides;
            }
            if (upper_redundant && isfinite(upper))
            {
                upper = INFINITY;
                presolver->working_constraint_upper[row] =
                    INFINITY;
                ++presolver->stats.removed_redundant_row_upper_sides;
            }
        }
        if (remaining == 0 && presolver->settings.remove_empty_rows)
        {
            if (lower > presolver->settings.feasibility_tolerance ||
                upper < -presolver->settings.feasibility_tolerance)
            {
                if (getenv("PREFOS_TRACE_MATRIX_COMPACTION"))
                    fprintf(
                        stderr,
                        "PreFOS transformed compaction empty-row conflict "
                        "row=%zu lower=%.17g upper=%.17g shift=%.17g\n",
                        row, lower, upper, shift);
                clear_transformed_a_row(row_values, row_marks, touched_columns,
                                        n_touched);
                status = PREFOS_STATUS_PRIMAL_INFEASIBLE;
                goto compact_failure;
            }
            clear_transformed_a_row(row_values, row_marks, touched_columns,
                                    n_touched);
            ++removed_empty_rows;
            continue;
        }

        status = reserve_compacted_a_entries(&target->A, &entry_capacity,
                                             write + remaining);
        if (status != PREFOS_STATUS_OK)
        {
            clear_transformed_a_row(row_values, row_marks, touched_columns,
                                    n_touched);
            goto compact_failure;
        }
        presolver->original_to_reduced_rows[row] = (int) kept_rows;
        target->A.row_pointers[kept_rows] = (int) write;
        target->constraint_lower[kept_rows] = lower;
        target->constraint_upper[kept_rows] = upper;
        for (position = 0; position < n_touched; ++position)
        {
            int mapped = touched_columns[position];
            if (row_values[mapped] != 0.0)
            {
                target->A.values[write] = row_values[mapped];
                target->A.column_indices[write] = mapped;
                ++write;
            }
        }
        clear_transformed_a_row(row_values, row_marks, touched_columns, n_touched);
        ++kept_rows;
    }
    target->A.row_pointers[kept_rows] = (int) write;
    target->A.rows = kept_rows;
    target->A.cols = target->n;
    target->A.nnz = write;
    presolver->stats.removed_empty_rows += removed_empty_rows;
    free(row_values);
    free(activity_lower);
    free(activity_upper);
    free(row_marks);
    free(touched_columns);
    prefos_internal_clear_working_matrix_cache(presolver);
    return PREFOS_STATUS_OK;

compact_failure:
    prefos_internal_free_csr(&target->A);
    free(target->constraint_lower);
    free(target->constraint_upper);
    target->constraint_lower = NULL;
    target->constraint_upper = NULL;
    free(row_values);
    free(activity_lower);
    free(activity_upper);
    free(row_marks);
    free(touched_columns);
    prefos_internal_clear_working_matrix_cache(presolver);
    return status;
}
