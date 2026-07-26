/*
 * Copyright 2026 Hongpei Li
 * SPDX-License-Identifier: Apache-2.0
 */

#include "PREFOS_WorkingMatrix.h"
#include "common/PreFOSThread.h"

#include <stdio.h>

#ifndef PREFOS_WORKING_MATRIX_THREADS
#define PREFOS_WORKING_MATRIX_THREADS 2
#endif
#ifndef PREFOS_WORKING_MATRIX_THREAD_THRESHOLD
#define PREFOS_WORKING_MATRIX_THREAD_THRESHOLD 1000000
#endif
#ifndef PREFOS_SUBSTITUTION_MATRIX_THREAD_THRESHOLD
#define PREFOS_SUBSTITUTION_MATRIX_THREAD_THRESHOLD 8000000
#endif
#ifndef PREFOS_WORKING_MATRIX_COLUMN_COUNT_THRESHOLD
#define PREFOS_WORKING_MATRIX_COLUMN_COUNT_THRESHOLD 100000
#endif
#ifndef PREFOS_WORKING_MATRIX_COLUMN_COUNT_MAX_COLUMNS
#define PREFOS_WORKING_MATRIX_COLUMN_COUNT_MAX_COLUMNS 65536
#endif

typedef struct
{
    const PreFOSPresolver *presolver;
    const PreFOSCsrMatrix *source;
    const double *source_lower;
    const double *source_upper;
    PreFOSWorkingMatrix *target;
    size_t begin;
    size_t end;
    int write_values;
    PreFOSStatus status;
} PreFOSWorkingMatrixChunk;

static int compare_column_indices(const void *left, const void *right)
{
    int left_column = *(const int *) left;
    int right_column = *(const int *) right;
    return (left_column > right_column) - (left_column < right_column);
}

static int should_cache_column_counts(const PreFOSCsrMatrix *source)
{
    return source->nnz < PREFOS_WORKING_MATRIX_COLUMN_COUNT_THRESHOLD ||
           source->cols <=
               PREFOS_WORKING_MATRIX_COLUMN_COUNT_MAX_COLUMNS;
}

static size_t working_matrix_partition_end(
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

static void sort_touched_columns(int *columns, size_t count)
{
    size_t position;
    int sorted = 1;
    for (position = 1; position < count; ++position)
        if (columns[position - 1] > columns[position])
        {
            sorted = 0;
            break;
        }
    if (sorted) return;
    if (count <= 64)
    {
        for (position = 1; position < count; ++position)
        {
            int column = columns[position];
            size_t insertion = position;
            while (insertion > 0 &&
                   columns[insertion - 1] > column)
            {
                columns[insertion] = columns[insertion - 1];
                --insertion;
            }
            columns[insertion] = column;
        }
        return;
    }
    qsort(columns, count, sizeof(*columns), compare_column_indices);
}

static void *materialize_fixed_only_chunk(void *argument)
{
    PreFOSWorkingMatrixChunk *chunk =
        (PreFOSWorkingMatrixChunk *) argument;
    const PreFOSPresolver *presolver = chunk->presolver;
    const PreFOSCsrMatrix *source = chunk->source;
    size_t row;
    chunk->status = PREFOS_STATUS_OK;
    for (row = chunk->begin; row < chunk->end; ++row)
    {
        int p;
        if (chunk->write_values)
        {
            int write = chunk->target->matrix.row_pointers[row];
            if (presolver->remove_rows[row]) continue;
            for (p = source->row_pointers[row];
                 p < source->row_pointers[row + 1]; ++p)
            {
                int column = source->column_indices[p];
                if (source->values[p] == 0.0 ||
                    presolver->is_fixed[column] ||
                    presolver->is_parallel_removed[column])
                    continue;
                if (presolver->is_substituted[column])
                {
                    if (!prefos_internal_term_is_active_in_row(
                            presolver, row, column))
                        continue;
                    chunk->status = PREFOS_STATUS_NUMERICAL_ERROR;
                    return NULL;
                }
                chunk->target->matrix.values[write] = source->values[p];
                chunk->target->matrix.column_indices[write++] = column;
            }
            if (write != chunk->target->matrix.row_pointers[row + 1])
            {
                chunk->status = PREFOS_STATUS_NUMERICAL_ERROR;
                return NULL;
            }
        }
        else
        {
            size_t count = 0;
            double shift = 0.0;
            chunk->target->lower[row] = chunk->source_lower[row];
            chunk->target->upper[row] = chunk->source_upper[row];
            if (presolver->remove_rows[row])
            {
                chunk->target->matrix.row_pointers[row + 1] = 0;
                continue;
            }
            for (p = source->row_pointers[row];
                 p < source->row_pointers[row + 1]; ++p)
            {
                int column = source->column_indices[p];
                if (source->values[p] == 0.0) continue;
                if (presolver->is_fixed[column])
                {
                    if (!prefos_internal_safe_add_product(
                            &shift, source->values[p],
                            presolver->fixed_values[column]))
                    {
                        chunk->status = PREFOS_STATUS_NUMERICAL_ERROR;
                        return NULL;
                    }
                }
                else if (presolver->is_substituted[column])
                {
                    if (prefos_internal_term_is_active_in_row(
                            presolver, row, column))
                    {
                        chunk->status = PREFOS_STATUS_NUMERICAL_ERROR;
                        return NULL;
                    }
                }
                else if (!presolver->is_parallel_removed[column])
                    ++count;
            }
            chunk->target->lower[row] = chunk->source_lower[row] - shift;
            chunk->target->upper[row] = chunk->source_upper[row] - shift;
            if (isnan(chunk->target->lower[row]) ||
                isnan(chunk->target->upper[row]) ||
                count > (size_t) INT_MAX)
            {
                chunk->status = PREFOS_STATUS_NUMERICAL_ERROR;
                return NULL;
            }
            chunk->target->matrix.row_pointers[row + 1] = (int) count;
        }
    }
    return NULL;
}

static PreFOSStatus run_fixed_only_chunks(
    const PreFOSPresolver *presolver, const PreFOSCsrMatrix *source,
    const double *source_lower, const double *source_upper,
    PreFOSWorkingMatrix *target, int write_values)
{
    PreFOSWorkingMatrixChunk chunks[PREFOS_WORKING_MATRIX_THREADS];
    PreFOSThread threads[PREFOS_WORKING_MATRIX_THREADS - 1];
    unsigned char started[PREFOS_WORKING_MATRIX_THREADS - 1] = {0};
    size_t base_nnz, extra_nnz;
    size_t begin = 0;
    int available_threads =
        prefos_cpu_thread_limit(PREFOS_WORKING_MATRIX_THREADS);
    int n_chunks =
        source->nnz >= PREFOS_WORKING_MATRIX_THREAD_THRESHOLD &&
                source->rows >= PREFOS_WORKING_MATRIX_THREADS
            ? available_threads
            : 1;
    int chunk;
    if (n_chunks == 1)
    {
        chunks[0] = (PreFOSWorkingMatrixChunk){
            presolver, source, source_lower, source_upper, target,
            0, source->rows, write_values, PREFOS_STATUS_OK};
        materialize_fixed_only_chunk(&chunks[0]);
        return chunks[0].status;
    }
    base_nnz = source->nnz / (size_t) n_chunks;
    extra_nnz = source->nnz % (size_t) n_chunks;
    for (chunk = 0; chunk < n_chunks; ++chunk)
    {
        size_t end;
        if (chunk + 1 == n_chunks)
            end = source->rows;
        else
        {
            size_t completed_parts = (size_t) chunk + 1U;
            size_t target_nnz =
                base_nnz * completed_parts +
                (completed_parts < extra_nnz
                     ? completed_parts
                     : extra_nnz);
            size_t maximum_row =
                source->rows -
                (size_t) (n_chunks - chunk - 1);
            end = working_matrix_partition_end(
                source, target_nnz, begin + 1U, maximum_row);
        }
        chunks[chunk] = (PreFOSWorkingMatrixChunk){
            presolver, source, source_lower, source_upper, target,
            begin, end, write_values, PREFOS_STATUS_OK};
        begin = end;
    }
    for (chunk = 1; chunk < n_chunks; ++chunk)
        if (prefos_thread_create(
                &threads[chunk - 1], materialize_fixed_only_chunk,
                &chunks[chunk]) == 0)
            started[chunk - 1] = 1;
        else
            materialize_fixed_only_chunk(&chunks[chunk]);
    materialize_fixed_only_chunk(&chunks[0]);
    for (chunk = 1; chunk < n_chunks; ++chunk)
        if (started[chunk - 1])
            (void) prefos_thread_join(&threads[chunk - 1]);
    for (chunk = 0; chunk < n_chunks; ++chunk)
        if (chunks[chunk].status != PREFOS_STATUS_OK)
            return chunks[chunk].status;
    return PREFOS_STATUS_OK;
}

static PreFOSStatus materialize_fixed_only_matrix_serial(
    const PreFOSPresolver *presolver, const PreFOSCsrMatrix *source,
    const double *source_lower, const double *source_upper,
    PreFOSWorkingMatrix *target)
{
    size_t row, write = 0;

    target->matrix.row_pointers =
        (int *) calloc(source->rows + 1, sizeof(int));
    target->lower =
        (double *) prefos_internal_alloc_array(source->rows, sizeof(double));
    target->upper =
        (double *) prefos_internal_alloc_array(source->rows, sizeof(double));
    target->matrix.values =
        (double *) prefos_internal_alloc_array(source->nnz, sizeof(double));
    target->matrix.column_indices =
        (int *) prefos_internal_alloc_array(source->nnz, sizeof(int));
    if (should_cache_column_counts(source))
        target->column_counts =
            (int *) calloc(source->cols, sizeof(int));
    if (!target->matrix.row_pointers ||
        (source->rows > 0 && (!target->lower || !target->upper)) ||
        (source->cols > 0 &&
         should_cache_column_counts(source) &&
         !target->column_counts) ||
        (source->nnz > 0 &&
         (!target->matrix.values ||
          !target->matrix.column_indices)))
        return PREFOS_STATUS_OUT_OF_MEMORY;

    for (row = 0; row < source->rows; ++row)
    {
        double shift = 0.0;
        int p;
        target->matrix.row_pointers[row] = (int) write;
        target->lower[row] = source_lower[row];
        target->upper[row] = source_upper[row];
        if (presolver->remove_rows[row])
        {
            target->matrix.row_pointers[row + 1] = (int) write;
            continue;
        }
        for (p = source->row_pointers[row];
             p < source->row_pointers[row + 1]; ++p)
        {
            int column = source->column_indices[p];
            double coefficient = source->values[p];
            if (coefficient == 0.0) continue;
            if (presolver->is_fixed[column])
            {
                if (!prefos_internal_safe_add_product(
                        &shift, coefficient,
                        presolver->fixed_values[column]))
                    return PREFOS_STATUS_NUMERICAL_ERROR;
                continue;
            }
            if (presolver->is_substituted[column])
            {
                if (prefos_internal_term_is_active_in_row(
                        presolver, row, column))
                    return PREFOS_STATUS_NUMERICAL_ERROR;
                continue;
            }
            if (presolver->is_parallel_removed[column]) continue;
            target->matrix.values[write] = coefficient;
            target->matrix.column_indices[write++] = column;
            if (target->column_counts)
                ++target->column_counts[column];
        }
        target->lower[row] = source_lower[row] - shift;
        target->upper[row] = source_upper[row] - shift;
        if (isnan(target->lower[row]) || isnan(target->upper[row]) ||
            write > (size_t) INT_MAX)
            return PREFOS_STATUS_NUMERICAL_ERROR;
        target->matrix.row_pointers[row + 1] = (int) write;
    }
    target->matrix.rows = source->rows;
    target->matrix.cols = source->cols;
    target->matrix.nnz = write;
    return PREFOS_STATUS_OK;
}

static PreFOSStatus materialize_fixed_only_matrix(
    const PreFOSPresolver *presolver, const PreFOSCsrMatrix *source,
    const double *source_lower, const double *source_upper,
    PreFOSWorkingMatrix *target)
{
    size_t row, nnz = 0;
    PreFOSStatus status;
    if (source->nnz < PREFOS_WORKING_MATRIX_THREAD_THRESHOLD)
        return materialize_fixed_only_matrix_serial(
            presolver, source, source_lower, source_upper, target);
    target->matrix.row_pointers =
        (int *) calloc(source->rows + 1, sizeof(int));
    target->lower =
        (double *) prefos_internal_alloc_array(source->rows, sizeof(double));
    target->upper =
        (double *) prefos_internal_alloc_array(source->rows, sizeof(double));
    if (!target->matrix.row_pointers ||
        (source->rows > 0 && (!target->lower || !target->upper)))
        return PREFOS_STATUS_OUT_OF_MEMORY;
    status = run_fixed_only_chunks(
        presolver, source, source_lower, source_upper, target, 0);
    if (status != PREFOS_STATUS_OK) return status;
    for (row = 0; row < source->rows; ++row)
    {
        size_t count =
            (size_t) target->matrix.row_pointers[row + 1];
        target->matrix.row_pointers[row] = (int) nnz;
        if (count > (size_t) INT_MAX - nnz)
            return PREFOS_STATUS_OUT_OF_MEMORY;
        nnz += count;
    }
    target->matrix.row_pointers[source->rows] = (int) nnz;
    target->matrix.values =
        (double *) prefos_internal_alloc_array(nnz, sizeof(double));
    target->matrix.column_indices =
        (int *) prefos_internal_alloc_array(nnz, sizeof(int));
    if (nnz > 0 &&
        (!target->matrix.values || !target->matrix.column_indices))
        return PREFOS_STATUS_OUT_OF_MEMORY;
    target->matrix.rows = source->rows;
    target->matrix.cols = source->cols;
    target->matrix.nnz = nnz;
    return run_fixed_only_chunks(
        presolver, source, source_lower, source_upper, target, 1);
}

static PreFOSStatus accumulate_expanded_column(
    const PreFOSPresolver *presolver, int column, double coefficient,
    size_t depth, double *row_values, int *row_marks, int *touched_columns,
    size_t *n_touched, double *shift)
{
    size_t term;
    if (column < 0 || (size_t) column >= presolver->original.n ||
        depth > PREFOS_MAX_SUBSTITUTION_DEPTH)
        return PREFOS_STATUS_NUMERICAL_ERROR;
    if (presolver->is_fixed[column])
        return prefos_internal_safe_add_product(
                   shift, coefficient, presolver->fixed_values[column])
                   ? PREFOS_STATUS_OK
                   : PREFOS_STATUS_NUMERICAL_ERROR;
    if (presolver->is_parallel_removed[column]) return PREFOS_STATUS_OK;
    if (presolver->is_substituted[column])
    {
        size_t start = presolver->substitution_term_start[column];
        size_t count = presolver->substitution_term_count[column];
        if (count == 0 || start > presolver->n_substitution_terms ||
            count > presolver->n_substitution_terms - start ||
            !prefos_internal_safe_add_product(
                shift, coefficient,
                presolver->substitution_constant[column]))
            return PREFOS_STATUS_NUMERICAL_ERROR;
        for (term = 0; term < count; ++term)
        {
            double propagated;
            PreFOSStatus status;
            if (!prefos_internal_safe_product(
                    coefficient,
                    presolver->substitution_scales[start + term],
                    &propagated))
                return PREFOS_STATUS_NUMERICAL_ERROR;
            status = accumulate_expanded_column(
                presolver,
                presolver->substitution_targets[start + term],
                propagated, depth + 1, row_values, row_marks,
                touched_columns, n_touched, shift);
            if (status != PREFOS_STATUS_OK) return status;
        }
        return PREFOS_STATUS_OK;
    }
    if (row_marks[column] < 0)
    {
        row_marks[column] = 1;
        row_values[column] = coefficient;
        touched_columns[(*n_touched)++] = column;
    }
    else if (!prefos_internal_safe_add_product(
                 &row_values[column], 1.0, coefficient))
        return PREFOS_STATUS_NUMERICAL_ERROR;
    return PREFOS_STATUS_OK;
}

void prefos_internal_clear_expanded_working_row(
    double *row_values, int *row_marks, const int *touched_columns,
    size_t n_touched)
{
    size_t position;
    for (position = 0; position < n_touched; ++position)
    {
        int column = touched_columns[position];
        row_values[column] = 0.0;
        row_marks[column] = -1;
    }
}

PreFOSStatus prefos_internal_expand_working_row(
    const PreFOSPresolver *presolver, const PreFOSCsrMatrix *source,
    size_t row, double *row_values, int *row_marks,
    int *touched_columns, size_t *n_touched, double *shift)
{
    int source_position;
    PreFOSStatus status;
    if (!presolver || !source || row >= source->rows ||
        !row_values || !row_marks || !touched_columns ||
        !n_touched || !shift)
        return PREFOS_STATUS_INVALID_ARGUMENT;
    *n_touched = 0;
    *shift = 0.0;
    for (source_position = source->row_pointers[row];
         source_position < source->row_pointers[row + 1];
         ++source_position)
    {
        int column = source->column_indices[source_position];
        if (source->values[source_position] == 0.0 ||
            !prefos_internal_term_is_active_in_row(
                presolver, row, column))
            continue;
        status = accumulate_expanded_column(
            presolver, column, source->values[source_position], 0,
            row_values, row_marks, touched_columns, n_touched, shift);
        if (status != PREFOS_STATUS_OK) return status;
    }
    sort_touched_columns(touched_columns, *n_touched);
    return PREFOS_STATUS_OK;
}

static int reserve_working_matrix_entries(
    PreFOSWorkingMatrix *target, size_t required, size_t *capacity)
{
    double *values;
    int *columns;
    size_t next;
    if (required <= *capacity) return 1;
    next = *capacity < 1024 ? 1024 : *capacity;
    while (next < required)
    {
        size_t growth = next / 2 + 1;
        if (growth > (size_t) INT_MAX - next)
        {
            next = required;
            break;
        }
        next += growth;
    }
    if (next > (size_t) INT_MAX ||
        next > SIZE_MAX / sizeof(double) ||
        next > SIZE_MAX / sizeof(int))
        return 0;
    values = (double *) realloc(
        target->matrix.values, next * sizeof(double));
    if (!values) return 0;
    target->matrix.values = values;
    columns = (int *) realloc(
        target->matrix.column_indices, next * sizeof(int));
    if (!columns) return 0;
    target->matrix.column_indices = columns;
    *capacity = next;
    return 1;
}

static PreFOSStatus materialize_substitution_matrix_serial(
    const PreFOSPresolver *presolver, const PreFOSCsrMatrix *source,
    const double *source_lower, const double *source_upper,
    PreFOSWorkingMatrix *target)
{
    double *row_values = NULL;
    int *row_marks = NULL;
    int *touched_columns = NULL;
    size_t capacity = source->nnz;
    size_t write = 0, row;
    PreFOSStatus status = PREFOS_STATUS_OK;

    target->matrix.row_pointers =
        (int *) calloc(source->rows + 1, sizeof(int));
    target->lower =
        (double *) prefos_internal_alloc_array(source->rows, sizeof(double));
    target->upper =
        (double *) prefos_internal_alloc_array(source->rows, sizeof(double));
    target->matrix.values =
        (double *) prefos_internal_alloc_array(capacity, sizeof(double));
    target->matrix.column_indices =
        (int *) prefos_internal_alloc_array(capacity, sizeof(int));
    if (should_cache_column_counts(source))
        target->column_counts =
            (int *) calloc(source->cols, sizeof(int));
    row_values = (double *) calloc(source->cols, sizeof(double));
    row_marks =
        (int *) prefos_internal_alloc_array(source->cols, sizeof(int));
    touched_columns =
        (int *) prefos_internal_alloc_array(source->cols, sizeof(int));
    if (!target->matrix.row_pointers ||
        (source->rows > 0 && (!target->lower || !target->upper)) ||
        (capacity > 0 &&
         (!target->matrix.values ||
          !target->matrix.column_indices)) ||
        (source->cols > 0 &&
         ((should_cache_column_counts(source) &&
           !target->column_counts) ||
          !row_values || !row_marks || !touched_columns)))
    {
        status = PREFOS_STATUS_OUT_OF_MEMORY;
        goto cleanup;
    }
    for (row = 0; row < source->cols; ++row) row_marks[row] = -1;

    for (row = 0; row < source->rows; ++row)
    {
        size_t n_touched = 0, position;
        double shift = 0.0;
        int source_position;

        target->matrix.row_pointers[row] = (int) write;
        target->lower[row] = source_lower[row];
        target->upper[row] = source_upper[row];
        if (presolver->remove_rows[row])
        {
            target->matrix.row_pointers[row + 1] = (int) write;
            continue;
        }
        for (source_position = source->row_pointers[row];
             source_position < source->row_pointers[row + 1];
             ++source_position)
        {
            int source_column =
                source->column_indices[source_position];
            if (source->values[source_position] == 0.0 ||
                !prefos_internal_term_is_active_in_row(
                    presolver, row, source_column))
                continue;
            status = accumulate_expanded_column(
                presolver, source_column,
                source->values[source_position], 0, row_values,
                row_marks, touched_columns, &n_touched, &shift);
            if (status != PREFOS_STATUS_OK) goto cleanup;
        }
        sort_touched_columns(touched_columns, n_touched);
        if (n_touched > (size_t) INT_MAX - write ||
            !reserve_working_matrix_entries(
                target, write + n_touched, &capacity))
        {
            status = PREFOS_STATUS_OUT_OF_MEMORY;
            goto cleanup;
        }
        for (position = 0; position < n_touched; ++position)
        {
            int column = touched_columns[position];
            if (row_values[column] == 0.0) continue;
            target->matrix.values[write] = row_values[column];
            target->matrix.column_indices[write++] = column;
            if (target->column_counts)
                ++target->column_counts[column];
        }
        target->lower[row] = source_lower[row] - shift;
        target->upper[row] = source_upper[row] - shift;
        if (isnan(target->lower[row]) ||
            isnan(target->upper[row]))
        {
            status = PREFOS_STATUS_NUMERICAL_ERROR;
            goto cleanup;
        }
        target->matrix.row_pointers[row + 1] = (int) write;
        prefos_internal_clear_expanded_working_row(
            row_values, row_marks, touched_columns, n_touched);
    }
    target->matrix.rows = source->rows;
    target->matrix.cols = source->cols;
    target->matrix.nnz = write;

cleanup:
    free(row_values);
    free(row_marks);
    free(touched_columns);
    return status;
}

typedef struct
{
    const PreFOSPresolver *presolver;
    const PreFOSCsrMatrix *source;
    const double *source_lower;
    const double *source_upper;
    PreFOSWorkingMatrix *target;
    size_t begin;
    size_t end;
    int write_values;
    PreFOSStatus status;
} PreFOSSubstitutionMatrixChunk;

static void *materialize_substitution_chunk(void *argument)
{
    PreFOSSubstitutionMatrixChunk *chunk =
        (PreFOSSubstitutionMatrixChunk *) argument;
    const PreFOSPresolver *presolver = chunk->presolver;
    const PreFOSCsrMatrix *source = chunk->source;
    double *row_values =
        (double *) calloc(source->cols, sizeof(double));
    int *row_marks =
        (int *) prefos_internal_alloc_array(source->cols, sizeof(int));
    int *touched_columns =
        (int *) prefos_internal_alloc_array(source->cols, sizeof(int));
    size_t row;

    chunk->status = PREFOS_STATUS_OK;
    if (source->cols > 0 &&
        (!row_values || !row_marks || !touched_columns))
    {
        chunk->status = PREFOS_STATUS_OUT_OF_MEMORY;
        goto cleanup;
    }
    for (row = 0; row < source->cols; ++row) row_marks[row] = -1;
    for (row = chunk->begin; row < chunk->end; ++row)
    {
        size_t n_touched = 0, position;
        double shift = 0.0;
        int source_position;

        if (!chunk->write_values)
        {
            chunk->target->matrix.row_pointers[row + 1] = 0;
            chunk->target->lower[row] = chunk->source_lower[row];
            chunk->target->upper[row] = chunk->source_upper[row];
        }
        if (presolver->remove_rows[row]) continue;
        for (source_position = source->row_pointers[row];
             source_position < source->row_pointers[row + 1];
             ++source_position)
        {
            int source_column =
                source->column_indices[source_position];
            if (source->values[source_position] == 0.0 ||
                !prefos_internal_term_is_active_in_row(
                    presolver, row, source_column))
                continue;
            chunk->status = accumulate_expanded_column(
                presolver, source_column,
                source->values[source_position], 0, row_values, row_marks,
                touched_columns, &n_touched, &shift);
            if (chunk->status != PREFOS_STATUS_OK)
                goto cleanup;
        }
        if (chunk->write_values)
        {
            int write = chunk->target->matrix.row_pointers[row];
            sort_touched_columns(touched_columns, n_touched);
            for (position = 0; position < n_touched; ++position)
            {
                int column = touched_columns[position];
                if (row_values[column] == 0.0) continue;
                chunk->target->matrix.values[write] =
                    row_values[column];
                chunk->target->matrix.column_indices[write++] =
                    column;
            }
            if (write != chunk->target->matrix.row_pointers[row + 1])
            {
                chunk->status = PREFOS_STATUS_NUMERICAL_ERROR;
                goto cleanup;
            }
        }
        else
        {
            size_t count = 0;
            for (position = 0; position < n_touched; ++position)
                if (row_values[touched_columns[position]] != 0.0)
                    ++count;
            if (count > (size_t) INT_MAX)
            {
                chunk->status = PREFOS_STATUS_OUT_OF_MEMORY;
                goto cleanup;
            }
            chunk->target->matrix.row_pointers[row + 1] =
                (int) count;
            chunk->target->lower[row] =
                chunk->source_lower[row] - shift;
            chunk->target->upper[row] =
                chunk->source_upper[row] - shift;
            if (isnan(chunk->target->lower[row]) ||
                isnan(chunk->target->upper[row]))
            {
                chunk->status = PREFOS_STATUS_NUMERICAL_ERROR;
                goto cleanup;
            }
        }
        prefos_internal_clear_expanded_working_row(
            row_values, row_marks, touched_columns, n_touched);
    }

cleanup:
    free(row_values);
    free(row_marks);
    free(touched_columns);
    return NULL;
}

static PreFOSStatus run_substitution_chunks(
    const PreFOSPresolver *presolver, const PreFOSCsrMatrix *source,
    const double *source_lower, const double *source_upper,
    PreFOSWorkingMatrix *target, int write_values)
{
    PreFOSSubstitutionMatrixChunk
        chunks[PREFOS_WORKING_MATRIX_THREADS];
    PreFOSThread threads[PREFOS_WORKING_MATRIX_THREADS - 1];
    unsigned char started[PREFOS_WORKING_MATRIX_THREADS - 1] = {0};
    size_t base_nnz, extra_nnz;
    size_t begin = 0;
    int available_threads =
        prefos_cpu_thread_limit(PREFOS_WORKING_MATRIX_THREADS);
    int n_chunks =
        source->nnz >= PREFOS_SUBSTITUTION_MATRIX_THREAD_THRESHOLD &&
                source->rows >= PREFOS_WORKING_MATRIX_THREADS
            ? available_threads
            : 1;
    int chunk;
    if (n_chunks == 1)
    {
        chunks[0] = (PreFOSSubstitutionMatrixChunk){
            presolver, source, source_lower, source_upper, target,
            0, source->rows, write_values, PREFOS_STATUS_OK};
        materialize_substitution_chunk(&chunks[0]);
        return chunks[0].status;
    }
    base_nnz = source->nnz / (size_t) n_chunks;
    extra_nnz = source->nnz % (size_t) n_chunks;
    for (chunk = 0; chunk < n_chunks; ++chunk)
    {
        size_t end;
        if (chunk + 1 == n_chunks)
            end = source->rows;
        else
        {
            size_t completed_parts = (size_t) chunk + 1U;
            size_t target_nnz =
                base_nnz * completed_parts +
                (completed_parts < extra_nnz
                     ? completed_parts
                     : extra_nnz);
            size_t maximum_row =
                source->rows -
                (size_t) (n_chunks - chunk - 1);
            end = working_matrix_partition_end(
                source, target_nnz, begin + 1U, maximum_row);
        }
        chunks[chunk] = (PreFOSSubstitutionMatrixChunk){
            presolver, source, source_lower, source_upper, target,
            begin, end, write_values, PREFOS_STATUS_OK};
        begin = end;
    }
    for (chunk = 1; chunk < n_chunks; ++chunk)
        if (prefos_thread_create(
                &threads[chunk - 1], materialize_substitution_chunk,
                &chunks[chunk]) == 0)
            started[chunk - 1] = 1;
        else
            materialize_substitution_chunk(&chunks[chunk]);
    materialize_substitution_chunk(&chunks[0]);
    for (chunk = 1; chunk < n_chunks; ++chunk)
        if (started[chunk - 1])
            (void) prefos_thread_join(&threads[chunk - 1]);
    for (chunk = 0; chunk < n_chunks; ++chunk)
        if (chunks[chunk].status != PREFOS_STATUS_OK)
            return chunks[chunk].status;
    return PREFOS_STATUS_OK;
}

void prefos_internal_free_working_matrix(PreFOSWorkingMatrix *working)
{
    if (!working) return;
    prefos_internal_free_csr(&working->matrix);
    free(working->lower);
    free(working->upper);
    free(working->column_counts);
    memset(working, 0, sizeof(*working));
}

void prefos_internal_clear_working_matrix_cache(
    PreFOSPresolver *presolver)
{
    if (!presolver) return;
    prefos_internal_free_csr(&presolver->cached_working_matrix);
    free(presolver->cached_working_lower);
    free(presolver->cached_working_upper);
    presolver->cached_working_lower = NULL;
    presolver->cached_working_upper = NULL;
    presolver->cached_working_matrix_valid = 0;
    presolver->cached_working_fixed_column_epoch = 0;
    presolver->cached_working_column_transformations = 0;
    presolver->cached_working_parallel_rows_checked = 0;
}

void prefos_internal_store_working_matrix_cache(
    PreFOSPresolver *presolver, PreFOSWorkingMatrix *working)
{
    if (!presolver || !working) return;
    prefos_internal_clear_working_matrix_cache(presolver);
    presolver->cached_working_matrix = working->matrix;
    presolver->cached_working_lower = working->lower;
    presolver->cached_working_upper = working->upper;
    presolver->cached_working_matrix_valid = 1;
    presolver->cached_working_fixed_column_epoch =
        presolver->fixed_column_epoch;
    presolver->cached_working_column_transformations =
        presolver->transformations.n_column_transformations;
    presolver->cached_working_parallel_rows_checked = 0;
    free(working->column_counts);
    memset(working, 0, sizeof(*working));
}

int prefos_internal_take_current_working_matrix_cache(
    PreFOSPresolver *presolver, PreFOSWorkingMatrix *working,
    int *parallel_rows_checked)
{
    if (!presolver || !working ||
        !presolver->cached_working_matrix_valid ||
        presolver->cached_working_fixed_column_epoch !=
            presolver->fixed_column_epoch ||
        presolver->cached_working_column_transformations !=
            presolver->transformations.n_column_transformations)
        return 0;
    memset(working, 0, sizeof(*working));
    working->matrix = presolver->cached_working_matrix;
    working->lower = presolver->cached_working_lower;
    working->upper = presolver->cached_working_upper;
    if (parallel_rows_checked)
        *parallel_rows_checked =
            presolver->cached_working_parallel_rows_checked;
    memset(
        &presolver->cached_working_matrix, 0,
        sizeof(presolver->cached_working_matrix));
    presolver->cached_working_lower = NULL;
    presolver->cached_working_upper = NULL;
    presolver->cached_working_matrix_valid = 0;
    presolver->cached_working_fixed_column_epoch = 0;
    presolver->cached_working_column_transformations = 0;
    presolver->cached_working_parallel_rows_checked = 0;
    return 1;
}

void prefos_internal_get_working_matrix_source(
    const PreFOSPresolver *presolver, const PreFOSCsrMatrix **matrix,
    const double **lower, const double **upper)
{
    if (presolver->cached_working_matrix_valid)
    {
        *matrix = &presolver->cached_working_matrix;
        *lower = presolver->cached_working_lower;
        *upper = presolver->cached_working_upper;
        return;
    }
    *matrix = &presolver->original.A;
    *lower = presolver->working_constraint_lower;
    *upper = presolver->working_constraint_upper;
}

static PreFOSStatus compress_single_target_substitutions(
    PreFOSPresolver *presolver)
{
    const PresolveTransformationLog *log = &presolver->transformations;
    size_t position;
    size_t compressed = 0;

    /*
     * A target can only be substituted after its sources. Walking the
     * transformation log backwards therefore makes each target canonical
     * before it is folded into its predecessors.
     */
    for (position = log->n_column_transformations; position > 0; --position)
    {
        const PresolveColumnTransformationRecord *record =
            &log->column_transformations[position - 1];
        int column, target, resolved_target;
        size_t start, target_start;
        double scale, constant, resolved_scale, resolved_constant;

        if (record->type != PRESOLVE_COLUMN_SUBSTITUTED)
            continue;
        column = record->column;
        if (column < 0 || (size_t) column >= presolver->original.n ||
            !presolver->is_substituted[column] ||
            presolver->substitution_term_count[column] != 1)
            continue;
        start = presolver->substitution_term_start[column];
        if (start >= presolver->n_substitution_terms)
            return PREFOS_STATUS_NUMERICAL_ERROR;
        target = presolver->substitution_targets[start];
        if (target < 0 || (size_t) target >= presolver->original.n ||
            !presolver->is_substituted[target] ||
            presolver->substitution_term_count[target] != 1)
            continue;
        target_start = presolver->substitution_term_start[target];
        if (target_start >= presolver->n_substitution_terms)
            return PREFOS_STATUS_NUMERICAL_ERROR;
        resolved_target =
            presolver->substitution_targets[target_start];
        if (resolved_target == column)
            return PREFOS_STATUS_NUMERICAL_ERROR;

        scale = presolver->substitution_scales[start];
        constant = presolver->substitution_constant[column];
        resolved_scale =
            presolver->substitution_scales[target_start];
        resolved_constant =
            presolver->substitution_constant[target];
        if (!prefos_internal_safe_add_product(
                &constant, scale, resolved_constant) ||
            !prefos_internal_safe_product(
                scale, resolved_scale, &scale))
            return PREFOS_STATUS_NUMERICAL_ERROR;
        presolver->substitution_targets[start] = resolved_target;
        presolver->substitution_scales[start] = scale;
        presolver->substitution_constant[column] = constant;
        ++compressed;
    }
    if (compressed > 0 &&
        getenv("PREFOS_TRACE_SUBSTITUTION_COMPRESSION"))
        fprintf(
            stderr,
            "PreFOS substitution-compression paths=%zu transformations=%zu\n",
            compressed, log->n_column_transformations);
    return PREFOS_STATUS_OK;
}

PreFOSStatus prefos_internal_materialize_working_matrix(
    PreFOSPresolver *presolver, const PreFOSCsrMatrix *source,
    const double *source_lower, const double *source_upper,
    PreFOSWorkingMatrix *target)
{
    size_t row, nnz = 0;
    PreFOSStatus status = PREFOS_STATUS_OK;

    if (!presolver || !source || !source_lower || !source_upper || !target)
        return PREFOS_STATUS_INVALID_ARGUMENT;
    memset(target, 0, sizeof(*target));
    status = compress_single_target_substitutions(presolver);
    if (status != PREFOS_STATUS_OK) return status;
    if (presolver->n_substitution_terms == 0 ||
        presolver->n_rows_require_materialization == 0)
    {
        status = materialize_fixed_only_matrix(
            presolver, source, source_lower, source_upper, target);
        if (status != PREFOS_STATUS_OK)
            prefos_internal_free_working_matrix(target);
        return status;
    }
    if (source->nnz < PREFOS_SUBSTITUTION_MATRIX_THREAD_THRESHOLD)
    {
        status = materialize_substitution_matrix_serial(
            presolver, source, source_lower, source_upper, target);
        if (status != PREFOS_STATUS_OK)
            prefos_internal_free_working_matrix(target);
        return status;
    }
    target->matrix.row_pointers =
        (int *) calloc(source->rows + 1, sizeof(int));
    target->lower =
        (double *) prefos_internal_alloc_array(source->rows, sizeof(double));
    target->upper =
        (double *) prefos_internal_alloc_array(source->rows, sizeof(double));
    if (!target->matrix.row_pointers ||
        (source->rows > 0 && (!target->lower || !target->upper)))
    {
        status = PREFOS_STATUS_OUT_OF_MEMORY;
        goto cleanup;
    }
    status = run_substitution_chunks(
        presolver, source, source_lower, source_upper, target, 0);
    if (status != PREFOS_STATUS_OK) goto cleanup;
    for (row = 0; row < source->rows; ++row)
    {
        size_t count =
            (size_t) target->matrix.row_pointers[row + 1];
        target->matrix.row_pointers[row] = (int) nnz;
        if (count > (size_t) INT_MAX - nnz)
        {
            status = PREFOS_STATUS_OUT_OF_MEMORY;
            goto cleanup;
        }
        nnz += count;
    }
    target->matrix.row_pointers[source->rows] = (int) nnz;
    target->matrix.values =
        (double *) prefos_internal_alloc_array(nnz, sizeof(double));
    target->matrix.column_indices =
        (int *) prefos_internal_alloc_array(nnz, sizeof(int));
    if (nnz > 0 &&
        (!target->matrix.values || !target->matrix.column_indices))
    {
        status = PREFOS_STATUS_OUT_OF_MEMORY;
        goto cleanup;
    }
    status = run_substitution_chunks(
        presolver, source, source_lower, source_upper, target, 1);
    if (status != PREFOS_STATUS_OK) goto cleanup;
    target->matrix.rows = source->rows;
    target->matrix.cols = source->cols;
    target->matrix.nnz = nnz;

cleanup:
    if (status != PREFOS_STATUS_OK)
        prefos_internal_free_working_matrix(target);
    return status;
}
