/*
 * Copyright 2025-2026 Daniel Cederberg
 * Copyright 2026 Hongpei Li
 *
 * Modified for PreFOS in 2026.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef PRESOLVE_PARALLEL_ROW_DETECTION_H
#define PRESOLVE_PARALLEL_ROW_DETECTION_H

#include <stddef.h>

typedef struct
{
    size_t n_rows;
    const double *values;
    const int *columns;
    const int *row_starts;
    const int *row_ends;
    size_t row_range_stride;
} PresolveSparseRowView;

typedef int (*PresolveRowIsActive)(const void *context, size_t row);
typedef void (*PresolveSortRows)(int *rows, size_t count,
                                 const int *support_hashes,
                                 const int *coefficient_hashes, int *auxiliary);

void presolve_sort_rows_by_hash(int *rows, size_t count,
                                const int *support_hashes,
                                const int *coefficient_hashes, int *auxiliary);

int presolve_compute_parallel_row_hashes(
    const PresolveSparseRowView *matrix, PresolveRowIsActive row_is_active,
    const void *active_context, int *support_hashes, int *coefficient_hashes);

int presolve_compute_parallel_row_hash_keys(
    const PresolveSparseRowView *matrix, const int *rows, size_t count,
    int *support_hashes, int *coefficient_hashes);

int presolve_find_parallel_rows(
    const PresolveSparseRowView *matrix, PresolveRowIsActive row_is_active,
    const void *active_context, double tolerance, int support_only,
    PresolveSortRows sort_rows, int *parallel_rows, int *support_hashes,
    int *coefficient_hashes, int *sort_auxiliary, int *group_starts,
    size_t group_starts_capacity, size_t *n_groups);

int presolve_find_parallel_rows_in_set(
    const PresolveSparseRowView *matrix, double tolerance, int support_only,
    PresolveSortRows sort_rows, int *parallel_rows, size_t active_count,
    int *support_hashes, int *coefficient_hashes, int *sort_auxiliary,
    int *group_starts, size_t group_starts_capacity, size_t *n_groups);

int presolve_find_parallel_rows_in_set_with_sorted_copy(
    const PresolveSparseRowView *matrix, double tolerance, int support_only,
    PresolveSortRows sort_rows, int *parallel_rows, size_t active_count,
    int *support_hashes, int *coefficient_hashes, int *sort_auxiliary,
    int *sorted_active_rows, size_t *sorted_active_count,
    int *group_starts, size_t group_starts_capacity, size_t *n_groups);

int presolve_collect_parallel_row_groups(
    const PresolveSparseRowView *matrix, double tolerance,
    int *parallel_rows, size_t active_count, const int *support_hashes,
    const int *coefficient_hashes, int *group_starts,
    size_t group_starts_capacity, size_t *n_groups);

#endif
