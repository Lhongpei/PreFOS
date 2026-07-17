/*
 * Copyright 2025-2026 Daniel Cederberg
 * Copyright 2026 Hongpei Li
 *
 * Modified for PreFOS in 2026.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ParallelRowDetection.h"

#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#define PARALLEL_HASH_INV_PRECISION 1e6

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

void presolve_sort_rows_by_hash(int *rows, size_t count,
                                const int *support_hashes,
                                const int *coefficient_hashes, int *auxiliary)
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
        maximum = fmax(maximum, fabs(values[i]));
    scale = values[0] > 0.0 ? 1.0 / maximum : -1.0 / maximum;
    for (i = 0; i < length; ++i)
    {
        uint32_t normalized = (uint32_t) round(
            values[i] * scale * PARALLEL_HASH_INV_PRECISION);
        hash = ((hash << 5U) + hash) + normalized;
    }
    return hash;
}

int presolve_compute_parallel_row_hashes(
    const PresolveSparseRowView *matrix, PresolveRowIsActive row_is_active,
    const void *active_context, int *support_hashes, int *coefficient_hashes)
{
    size_t row;
    if (!matrix || !row_is_active || !support_hashes || !coefficient_hashes ||
        matrix->row_range_stride == 0 ||
        (matrix->n_rows > 0 && (!matrix->row_starts || !matrix->row_ends)))
        return 0;

    for (row = 0; row < matrix->n_rows; ++row)
    {
        int start, length;
        if (!row_is_active(active_context, row))
        {
            support_hashes[row] = INT_MAX;
            coefficient_hashes[row] = INT_MAX;
            continue;
        }
        start = row_start(matrix, row);
        length = row_end(matrix, row) - start;
        if (start < 0 || length <= 0 || !matrix->values || !matrix->columns ||
            matrix->values[start] == 0.0)
        {
            support_hashes[row] = INT_MAX;
            coefficient_hashes[row] = INT_MAX;
            continue;
        }
        support_hashes[row] =
            (int) hash_int_array(matrix->columns + start, length);
        coefficient_hashes[row] =
            (int) hash_scaled_double_array(matrix->values + start, length);
    }
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
    const void *active_context, double tolerance, PresolveSortRows sort_rows,
    int *parallel_rows, int *support_hashes, int *coefficient_hashes,
    int *sort_auxiliary, int *group_starts, size_t group_starts_capacity,
    size_t *n_groups)
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
            parallel_rows[active_count++] = (int) row;
    if (!sort_rows) sort_rows = presolve_sort_rows_by_hash;
    sort_rows(parallel_rows, active_count, support_hashes, coefficient_hashes,
              sort_auxiliary);

    return presolve_collect_parallel_row_groups(
        matrix, tolerance, parallel_rows, active_count, support_hashes,
        coefficient_hashes, group_starts, group_starts_capacity, n_groups);
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
        size_t candidate, group_size = 0;
        while (bin_end < active_count &&
               support_hashes[parallel_rows[bin_end]] == support_hashes[seed] &&
               coefficient_hashes[parallel_rows[bin_end]] ==
                   coefficient_hashes[seed])
            ++bin_end;

        if (bin_end - row > 1)
        {
            for (candidate = row + 1; candidate < bin_end; ++candidate)
            {
                if (rows_are_parallel(matrix, seed, parallel_rows[candidate],
                                      tolerance))
                    parallel_rows[output_count + group_size++] =
                        parallel_rows[candidate];
            }
            if (group_size > 0)
            {
                parallel_rows[output_count + group_size++] = seed;
                output_count += group_size;
                if (group_count + 1 >= group_starts_capacity) return 0;
                group_starts[++group_count] = (int) output_count;
            }
        }
        row = bin_end;
    }

    *n_groups = group_count;
    return 1;
}
