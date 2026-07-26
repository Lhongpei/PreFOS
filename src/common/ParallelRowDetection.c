/*
 * Copyright 2025-2026 Daniel Cederberg
 * Copyright 2026 Hongpei Li
 *
 * Modified for PreFOS in 2026.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ParallelRowDetection.h"
#include "PreFOSThread.h"

#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define PARALLEL_HASH_INV_PRECISION 1e6
#define PARALLEL_CPU_THREADS 4
#define PARALLEL_CPU_THRESHOLD 100000

static void insertion_sort_rows(int *rows, size_t count,
                                const int *support_hashes,
                                const int *coefficient_hashes)
{
    size_t i;
    for (i = 1; i < count; ++i)
    {
        int key = rows[i];
        int key_support = support_hashes[key];
        int key_coefficient = coefficient_hashes[key];
        size_t position = i;
        while (position > 0)
        {
            int previous = rows[position - 1];
            if (support_hashes[previous] < key_support ||
                (support_hashes[previous] == key_support &&
                 coefficient_hashes[previous] < key_coefficient))
                break;
            rows[position] = previous;
            --position;
        }
        rows[position] = key;
    }
}

static void radix_sort_rows_by_hash(int *rows, size_t count,
                                    const int *support_hashes,
                                    const int *coefficient_hashes,
                                    int *auxiliary)
{
    size_t counts[256];
    int *source = rows, *destination = auxiliary;
    int phase, pass;

    if (count < 256)
    {
        insertion_sort_rows(rows, count, support_hashes, coefficient_hashes);
        return;
    }
    for (phase = 0; phase < 2; ++phase)
    {
        const int *keys = phase == 0 ? coefficient_hashes : support_hashes;
        for (pass = 0; pass < 4; ++pass)
        {
            int shift = pass * 8;
            size_t i, total = 0;
            int bucket, skip = 0;
            memset(counts, 0, sizeof(counts));
            for (i = 0; i < count; ++i)
            {
                unsigned byte =
                    ((uint32_t) keys[source[i]] >> shift) & 0xFFU;
                ++counts[byte];
            }
            if (!(phase == 0 && pass == 0))
            {
                for (bucket = 0; bucket < 256; ++bucket)
                    if (counts[bucket] == count)
                    {
                        skip = 1;
                        break;
                    }
                if (skip) continue;
            }
            for (bucket = 0; bucket < 256; ++bucket)
            {
                size_t bucket_count = counts[bucket];
                counts[bucket] = total;
                total += bucket_count;
            }
            if (phase == 0 && pass == 0)
            {
                for (i = count; i > 0; --i)
                {
                    unsigned byte =
                        ((uint32_t) keys[source[i - 1]] >> shift) & 0xFFU;
                    destination[counts[byte]++] = source[i - 1];
                }
            }
            else
            {
                for (i = 0; i < count; ++i)
                {
                    unsigned byte =
                        ((uint32_t) keys[source[i]] >> shift) & 0xFFU;
                    destination[counts[byte]++] = source[i];
                }
            }
            {
                int *temporary = source;
                source = destination;
                destination = temporary;
            }
        }
    }
    if (source != rows) memcpy(rows, source, count * sizeof(int));
}

typedef struct
{
    int *rows;
    size_t count;
    const int *support_hashes;
    const int *coefficient_hashes;
    int *auxiliary;
} ParallelSortChunk;

static void *sort_parallel_chunk(void *argument)
{
    ParallelSortChunk *chunk = (ParallelSortChunk *) argument;
    radix_sort_rows_by_hash(
        chunk->rows, chunk->count, chunk->support_hashes,
        chunk->coefficient_hashes, chunk->auxiliary);
    return NULL;
}

static int compare_hash_keys(int left, int right,
                             const int *support_hashes,
                             const int *coefficient_hashes)
{
    uint32_t left_support = (uint32_t) support_hashes[left];
    uint32_t right_support = (uint32_t) support_hashes[right];
    uint32_t left_coefficient, right_coefficient;
    if (left_support != right_support)
        return left_support < right_support ? -1 : 1;
    left_coefficient = (uint32_t) coefficient_hashes[left];
    right_coefficient = (uint32_t) coefficient_hashes[right];
    if (left_coefficient == right_coefficient) return 0;
    return left_coefficient < right_coefficient ? -1 : 1;
}

void presolve_sort_rows_by_hash(int *rows, size_t count,
                                const int *support_hashes,
                                const int *coefficient_hashes, int *auxiliary)
{
    ParallelSortChunk chunks[PARALLEL_CPU_THREADS];
    PreFOSThread threads[PARALLEL_CPU_THREADS - 1];
    unsigned char started[PARALLEL_CPU_THREADS - 1] = {0};
    size_t positions[PARALLEL_CPU_THREADS] = {0};
    size_t base, extra, offset = 0, output;
    int n_chunks =
        prefos_cpu_thread_limit(PARALLEL_CPU_THREADS);
    int chunk;

    if (count < PARALLEL_CPU_THRESHOLD ||
        n_chunks == 1)
    {
        radix_sort_rows_by_hash(
            rows, count, support_hashes, coefficient_hashes, auxiliary);
        return;
    }
    base = count / (size_t) n_chunks;
    extra = count % (size_t) n_chunks;
    for (chunk = 0; chunk < n_chunks; ++chunk)
    {
        size_t chunk_count =
            base + ((size_t) chunk < extra ? 1U : 0U);
        chunks[chunk] = (ParallelSortChunk){
            rows + offset, chunk_count, support_hashes,
            coefficient_hashes, auxiliary + offset};
        offset += chunk_count;
    }
    for (chunk = 1; chunk < n_chunks; ++chunk)
        if (prefos_thread_create(
                &threads[chunk - 1], sort_parallel_chunk,
                &chunks[chunk]) == 0)
            started[chunk - 1] = 1;
        else
            sort_parallel_chunk(&chunks[chunk]);
    sort_parallel_chunk(&chunks[0]);
    for (chunk = 1; chunk < n_chunks; ++chunk)
        if (started[chunk - 1])
            (void) prefos_thread_join(&threads[chunk - 1]);

    for (output = 0; output < count; ++output)
    {
        int best = -1;
        for (chunk = 0; chunk < n_chunks; ++chunk)
        {
            if (positions[chunk] >= chunks[chunk].count) continue;
            if (best < 0 ||
                compare_hash_keys(
                    chunks[chunk].rows[positions[chunk]],
                    chunks[best].rows[positions[best]],
                    support_hashes, coefficient_hashes) < 0)
                best = chunk;
        }
        auxiliary[output] =
            chunks[best].rows[positions[best]++];
    }
    memcpy(rows, auxiliary, count * sizeof(int));
}

static int row_start(const PresolveSparseRowView *matrix, size_t row)
{
    return matrix->row_starts[row * matrix->row_range_stride];
}

static int row_end(const PresolveSparseRowView *matrix, size_t row)
{
    return matrix->row_ends[row * matrix->row_range_stride];
}

static uint32_t hash_int_array(const int *values, int length)
{
    uint32_t hash = 5381U;
    int i;
    for (i = 0; i < length; ++i)
        hash = ((hash << 5U) + hash) + (uint32_t) values[i];
    return hash;
}

static uint32_t hash_scaled_double_array(const double *values, int length)
{
    double maximum = fabs(values[0]);
    double scale;
    uint32_t hash = 5381U;
    int i;

    for (i = 1; i < length; ++i)
    {
        double magnitude = fabs(values[i]);
        if (magnitude > maximum) maximum = magnitude;
    }
    scale = values[0] > 0.0 ? 1.0 / maximum : -1.0 / maximum;
    for (i = 0; i < length; ++i)
    {
        uint32_t normalized = (uint32_t) round(
            values[i] * scale * PARALLEL_HASH_INV_PRECISION);
        hash = ((hash << 5U) + hash) + normalized;
    }
    return hash;
}

typedef struct
{
    const PresolveSparseRowView *matrix;
    PresolveRowIsActive row_is_active;
    const void *active_context;
    int *support_hashes;
    int *coefficient_hashes;
    size_t begin;
    size_t end;
} ParallelHashChunk;

static void *compute_parallel_hash_chunk(void *argument)
{
    ParallelHashChunk *chunk = (ParallelHashChunk *) argument;
    size_t row;
    for (row = chunk->begin; row < chunk->end; ++row)
    {
        int start, length;
        if (!chunk->row_is_active(chunk->active_context, row))
        {
            chunk->support_hashes[row] = INT_MAX;
            chunk->coefficient_hashes[row] = INT_MAX;
            continue;
        }
        start = row_start(chunk->matrix, row);
        length = row_end(chunk->matrix, row) - start;
        if (start < 0 || length <= 0 || !chunk->matrix->values ||
            !chunk->matrix->columns ||
            chunk->matrix->values[start] == 0.0)
        {
            chunk->support_hashes[row] = INT_MAX;
            chunk->coefficient_hashes[row] = INT_MAX;
            continue;
        }
        chunk->support_hashes[row] =
            (int) hash_int_array(
                chunk->matrix->columns + start, length);
        chunk->coefficient_hashes[row] =
            (int) hash_scaled_double_array(
                chunk->matrix->values + start, length);
    }
    return NULL;
}

int presolve_compute_parallel_row_hashes(
    const PresolveSparseRowView *matrix, PresolveRowIsActive row_is_active,
    const void *active_context, int *support_hashes, int *coefficient_hashes)
{
    ParallelHashChunk chunks[PARALLEL_CPU_THREADS];
    PreFOSThread threads[PARALLEL_CPU_THREADS - 1];
    unsigned char started[PARALLEL_CPU_THREADS - 1] = {0};
    size_t base, extra, begin = 0;
    int n_chunks =
        prefos_cpu_thread_limit(PARALLEL_CPU_THREADS);
    int chunk;
    if (!matrix || !row_is_active || !support_hashes || !coefficient_hashes ||
        matrix->row_range_stride == 0 ||
        (matrix->n_rows > 0 && (!matrix->row_starts || !matrix->row_ends)))
        return 0;

    if (matrix->n_rows < PARALLEL_CPU_THRESHOLD ||
        n_chunks == 1)
    {
        ParallelHashChunk sequential = {
            matrix, row_is_active, active_context, support_hashes,
            coefficient_hashes, 0, matrix->n_rows};
        compute_parallel_hash_chunk(&sequential);
        return 1;
    }
    base = matrix->n_rows / (size_t) n_chunks;
    extra = matrix->n_rows % (size_t) n_chunks;
    for (chunk = 0; chunk < n_chunks; ++chunk)
    {
        size_t count =
            base + ((size_t) chunk < extra ? 1U : 0U);
        chunks[chunk] = (ParallelHashChunk){
            matrix, row_is_active, active_context, support_hashes,
            coefficient_hashes, begin, begin + count};
        begin += count;
    }
    for (chunk = 1; chunk < n_chunks; ++chunk)
        if (prefos_thread_create(
                &threads[chunk - 1], compute_parallel_hash_chunk,
                &chunks[chunk]) == 0)
            started[chunk - 1] = 1;
        else
            compute_parallel_hash_chunk(&chunks[chunk]);
    compute_parallel_hash_chunk(&chunks[0]);
    for (chunk = 1; chunk < n_chunks; ++chunk)
        if (started[chunk - 1])
            (void) prefos_thread_join(&threads[chunk - 1]);
    return 1;
}

typedef struct
{
    const PresolveSparseRowView *matrix;
    const int *rows;
    int *support_hashes;
    int *coefficient_hashes;
    size_t begin;
    size_t end;
} ParallelActiveHashChunk;

typedef struct
{
    const PresolveSparseRowView *matrix;
    const int *rows;
    int *support_hashes;
    int *coefficient_hashes;
    size_t begin;
    size_t end;
} ParallelCompactHashChunk;

static void *compute_parallel_active_hash_chunk(void *argument)
{
    ParallelActiveHashChunk *chunk =
        (ParallelActiveHashChunk *) argument;
    size_t position;
    for (position = chunk->begin; position < chunk->end; ++position)
    {
        int row = chunk->rows[position];
        int start, length;
        if (row < 0 ||
            (size_t) row >= chunk->matrix->n_rows)
            continue;
        start = row_start(chunk->matrix, (size_t) row);
        length = row_end(chunk->matrix, (size_t) row) - start;
        if (start < 0 || length <= 0 || !chunk->matrix->values ||
            !chunk->matrix->columns ||
            chunk->matrix->values[start] == 0.0)
        {
            chunk->support_hashes[row] = INT_MAX;
            chunk->coefficient_hashes[row] = INT_MAX;
            continue;
        }
        chunk->support_hashes[row] =
            (int) hash_int_array(
                chunk->matrix->columns + start, length);
        chunk->coefficient_hashes[row] =
            (int) hash_scaled_double_array(
                chunk->matrix->values + start, length);
    }
    return NULL;
}

static void *compute_parallel_compact_hash_chunk(void *argument)
{
    ParallelCompactHashChunk *chunk =
        (ParallelCompactHashChunk *) argument;
    size_t position;
    for (position = chunk->begin; position < chunk->end; ++position)
    {
        int row = chunk->rows[position];
        int start, length;
        if (row < 0 ||
            (size_t) row >= chunk->matrix->n_rows)
        {
            chunk->support_hashes[position] = INT_MAX;
            chunk->coefficient_hashes[position] = INT_MAX;
            continue;
        }
        start = row_start(chunk->matrix, (size_t) row);
        length = row_end(chunk->matrix, (size_t) row) - start;
        if (start < 0 || length <= 0 || !chunk->matrix->values ||
            !chunk->matrix->columns ||
            chunk->matrix->values[start] == 0.0)
        {
            chunk->support_hashes[position] = INT_MAX;
            chunk->coefficient_hashes[position] = INT_MAX;
            continue;
        }
        chunk->support_hashes[position] =
            (int) hash_int_array(
                chunk->matrix->columns + start, length);
        chunk->coefficient_hashes[position] =
            (int) hash_scaled_double_array(
                chunk->matrix->values + start, length);
    }
    return NULL;
}

int presolve_compute_parallel_row_hash_keys(
    const PresolveSparseRowView *matrix, const int *rows, size_t count,
    int *support_hashes, int *coefficient_hashes)
{
    ParallelCompactHashChunk chunks[PARALLEL_CPU_THREADS];
    PreFOSThread threads[PARALLEL_CPU_THREADS - 1];
    unsigned char started[PARALLEL_CPU_THREADS - 1] = {0};
    size_t base, extra, begin = 0;
    int n_chunks =
        prefos_cpu_thread_limit(PARALLEL_CPU_THREADS);
    int chunk;

    if (!matrix || (count > 0 && !rows) ||
        (count > 0 && (!support_hashes || !coefficient_hashes)) ||
        matrix->row_range_stride == 0 ||
        (matrix->n_rows > 0 &&
         (!matrix->row_starts || !matrix->row_ends)))
        return 0;
    if (count < PARALLEL_CPU_THRESHOLD || n_chunks == 1)
    {
        ParallelCompactHashChunk sequential = {
            matrix, rows, support_hashes, coefficient_hashes, 0, count};
        compute_parallel_compact_hash_chunk(&sequential);
        return 1;
    }
    base = count / (size_t) n_chunks;
    extra = count % (size_t) n_chunks;
    for (chunk = 0; chunk < n_chunks; ++chunk)
    {
        size_t chunk_count =
            base + ((size_t) chunk < extra ? 1U : 0U);
        chunks[chunk] = (ParallelCompactHashChunk){
            matrix, rows, support_hashes, coefficient_hashes,
            begin, begin + chunk_count};
        begin += chunk_count;
    }
    for (chunk = 1; chunk < n_chunks; ++chunk)
        if (prefos_thread_create(
                &threads[chunk - 1],
                compute_parallel_compact_hash_chunk,
                &chunks[chunk]) == 0)
            started[chunk - 1] = 1;
        else
            compute_parallel_compact_hash_chunk(&chunks[chunk]);
    compute_parallel_compact_hash_chunk(&chunks[0]);
    for (chunk = 1; chunk < n_chunks; ++chunk)
        if (started[chunk - 1])
            (void) prefos_thread_join(&threads[chunk - 1]);
    return 1;
}

static int compute_parallel_row_hashes_in_set(
    const PresolveSparseRowView *matrix, const int *rows,
    size_t count, int *support_hashes, int *coefficient_hashes)
{
    ParallelActiveHashChunk chunks[PARALLEL_CPU_THREADS];
    PreFOSThread threads[PARALLEL_CPU_THREADS - 1];
    unsigned char started[PARALLEL_CPU_THREADS - 1] = {0};
    size_t base, extra, begin = 0;
    int n_chunks =
        prefos_cpu_thread_limit(PARALLEL_CPU_THREADS);
    int chunk;

    if (!matrix || (count > 0 && !rows) || !support_hashes ||
        !coefficient_hashes || matrix->row_range_stride == 0 ||
        (matrix->n_rows > 0 &&
         (!matrix->row_starts || !matrix->row_ends)))
        return 0;
    if (count < PARALLEL_CPU_THRESHOLD || n_chunks == 1)
    {
        ParallelActiveHashChunk sequential = {
            matrix, rows, support_hashes, coefficient_hashes, 0, count};
        compute_parallel_active_hash_chunk(&sequential);
        return 1;
    }
    base = count / (size_t) n_chunks;
    extra = count % (size_t) n_chunks;
    for (chunk = 0; chunk < n_chunks; ++chunk)
    {
        size_t chunk_count =
            base + ((size_t) chunk < extra ? 1U : 0U);
        chunks[chunk] = (ParallelActiveHashChunk){
            matrix, rows, support_hashes, coefficient_hashes,
            begin, begin + chunk_count};
        begin += chunk_count;
    }
    for (chunk = 1; chunk < n_chunks; ++chunk)
        if (prefos_thread_create(
                &threads[chunk - 1],
                compute_parallel_active_hash_chunk,
                &chunks[chunk]) == 0)
            started[chunk - 1] = 1;
        else
            compute_parallel_active_hash_chunk(&chunks[chunk]);
    compute_parallel_active_hash_chunk(&chunks[0]);
    for (chunk = 1; chunk < n_chunks; ++chunk)
        if (started[chunk - 1])
            (void) prefos_thread_join(&threads[chunk - 1]);
    return 1;
}

static int rows_are_parallel(const PresolveSparseRowView *matrix, int first,
                             int second, double tolerance)
{
    int first_start = row_start(matrix, (size_t) first);
    int second_start = row_start(matrix, (size_t) second);
    int first_length = row_end(matrix, (size_t) first) - first_start;
    int second_length = row_end(matrix, (size_t) second) - second_start;
    double ratio;
    int position;

    if (first_length != second_length || first_length <= 0) return 0;
    ratio = matrix->values[first_start] / matrix->values[second_start];
    for (position = 0; position < first_length; ++position)
    {
        double difference = matrix->values[first_start + position] -
                            ratio * matrix->values[second_start + position];
        if (fabs(difference) > tolerance ||
            matrix->columns[first_start + position] !=
                matrix->columns[second_start + position])
            return 0;
    }
    return 1;
}

int presolve_find_parallel_rows(
    const PresolveSparseRowView *matrix, PresolveRowIsActive row_is_active,
    const void *active_context, double tolerance, int support_only,
    PresolveSortRows sort_rows, int *parallel_rows, int *support_hashes,
    int *coefficient_hashes, int *sort_auxiliary, int *group_starts,
    size_t group_starts_capacity, size_t *n_groups)
{
    size_t row, active_count = 0;
    if (!matrix || !parallel_rows || !sort_auxiliary ||
        !group_starts || !n_groups || !isfinite(tolerance) || tolerance < 0.0 ||
        group_starts_capacity == 0 || matrix->n_rows > (size_t) INT_MAX)
        return 0;
    if (!presolve_compute_parallel_row_hashes(
            matrix, row_is_active, active_context, support_hashes,
            coefficient_hashes))
        return 0;

    for (row = 0; row < matrix->n_rows; ++row)
        if (support_hashes[row] != INT_MAX ||
            coefficient_hashes[row] != INT_MAX)
        {
            if (support_only)
                coefficient_hashes[row] = 0;
            parallel_rows[active_count++] = (int) row;
        }
    if (!sort_rows) sort_rows = presolve_sort_rows_by_hash;
    sort_rows(parallel_rows, active_count, support_hashes, coefficient_hashes,
              sort_auxiliary);

    return presolve_collect_parallel_row_groups(
        matrix, tolerance, parallel_rows, active_count, support_hashes,
        coefficient_hashes, group_starts, group_starts_capacity, n_groups);
}

static int find_parallel_rows_in_set(
    const PresolveSparseRowView *matrix, double tolerance, int support_only,
    PresolveSortRows sort_rows, int *parallel_rows, size_t active_count,
    int *support_hashes, int *coefficient_hashes, int *sort_auxiliary,
    int *sorted_active_rows, size_t *sorted_active_count,
    int *group_starts, size_t group_starts_capacity, size_t *n_groups)
{
    size_t position, write = 0;
    if (!matrix || (active_count > 0 && !parallel_rows) ||
        !sort_auxiliary || !group_starts || !n_groups ||
        !isfinite(tolerance) || tolerance < 0.0 ||
        group_starts_capacity == 0)
        return 0;
    if (!compute_parallel_row_hashes_in_set(
            matrix, parallel_rows, active_count,
            support_hashes, coefficient_hashes))
        return 0;
    for (position = 0; position < active_count; ++position)
    {
        int row = parallel_rows[position];
        if (row < 0 || (size_t) row >= matrix->n_rows)
            continue;
        if (support_hashes[row] == INT_MAX &&
            coefficient_hashes[row] == INT_MAX)
            continue;
        if (support_only)
            coefficient_hashes[row] = 0;
        parallel_rows[write++] = row;
    }
    if (!sort_rows) sort_rows = presolve_sort_rows_by_hash;
    sort_rows(parallel_rows, write, support_hashes, coefficient_hashes,
              sort_auxiliary);
    if (sorted_active_rows && write > 0)
        memcpy(sorted_active_rows, parallel_rows, write * sizeof(int));
    if (sorted_active_count) *sorted_active_count = write;
    return presolve_collect_parallel_row_groups(
        matrix, tolerance, parallel_rows, write, support_hashes,
        coefficient_hashes, group_starts, group_starts_capacity, n_groups);
}

int presolve_find_parallel_rows_in_set(
    const PresolveSparseRowView *matrix, double tolerance, int support_only,
    PresolveSortRows sort_rows, int *parallel_rows, size_t active_count,
    int *support_hashes, int *coefficient_hashes, int *sort_auxiliary,
    int *group_starts, size_t group_starts_capacity, size_t *n_groups)
{
    return find_parallel_rows_in_set(
        matrix, tolerance, support_only, sort_rows, parallel_rows,
        active_count, support_hashes, coefficient_hashes, sort_auxiliary,
        NULL, NULL, group_starts, group_starts_capacity, n_groups);
}

int presolve_find_parallel_rows_in_set_with_sorted_copy(
    const PresolveSparseRowView *matrix, double tolerance, int support_only,
    PresolveSortRows sort_rows, int *parallel_rows, size_t active_count,
    int *support_hashes, int *coefficient_hashes, int *sort_auxiliary,
    int *sorted_active_rows, size_t *sorted_active_count,
    int *group_starts, size_t group_starts_capacity, size_t *n_groups)
{
    if (!sorted_active_rows || !sorted_active_count) return 0;
    return find_parallel_rows_in_set(
        matrix, tolerance, support_only, sort_rows, parallel_rows,
        active_count, support_hashes, coefficient_hashes, sort_auxiliary,
        sorted_active_rows, sorted_active_count, group_starts,
        group_starts_capacity, n_groups);
}

int presolve_collect_parallel_row_groups(
    const PresolveSparseRowView *matrix, double tolerance,
    int *parallel_rows, size_t active_count, const int *support_hashes,
    const int *coefficient_hashes, int *group_starts,
    size_t group_starts_capacity, size_t *n_groups)
{
    size_t row, output_count = 0, group_count = 0;
    if (!matrix || !parallel_rows || !support_hashes || !coefficient_hashes ||
        !group_starts || !n_groups || !isfinite(tolerance) || tolerance < 0.0 ||
        group_starts_capacity == 0)
        return 0;
    group_starts[0] = 0;
    for (row = 0; row < active_count;)
    {
        size_t bin_end = row + 1;
        int seed = parallel_rows[row];
        size_t subgroup_start;
        while (bin_end < active_count &&
               support_hashes[parallel_rows[bin_end]] == support_hashes[seed] &&
               coefficient_hashes[parallel_rows[bin_end]] ==
                   coefficient_hashes[seed])
            ++bin_end;

        if (bin_end - row > 1)
        {
            subgroup_start = row;
            while (subgroup_start < bin_end)
            {
                size_t group_end = subgroup_start + 1;
                size_t candidate;
                seed = parallel_rows[subgroup_start];
                for (candidate = group_end; candidate < bin_end; ++candidate)
                {
                    if (rows_are_parallel(
                            matrix, seed, parallel_rows[candidate],
                            tolerance))
                    {
                        int temporary = parallel_rows[group_end];
                        parallel_rows[group_end] = parallel_rows[candidate];
                        parallel_rows[candidate] = temporary;
                        ++group_end;
                    }
                }
                if (group_end - subgroup_start > 1)
                {
                    size_t group_size = group_end - subgroup_start;
                    int subgroup_seed = parallel_rows[subgroup_start];
                    if (group_count + 1 >= group_starts_capacity)
                        return 0;
                    memmove(parallel_rows + subgroup_start,
                            parallel_rows + subgroup_start + 1,
                            (group_size - 1) * sizeof(int));
                    parallel_rows[group_end - 1] = subgroup_seed;
                    memmove(parallel_rows + output_count,
                            parallel_rows + subgroup_start,
                            group_size * sizeof(int));
                    output_count += group_size;
                    group_starts[++group_count] = (int) output_count;
                }
                subgroup_start = group_end;
            }
        }
        row = bin_end;
    }

    *n_groups = group_count;
    return 1;
}
