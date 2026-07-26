/*
 * Copyright 2026 Hongpei Li
 * SPDX-License-Identifier: Apache-2.0
 */

#include "PREFOS_TransformedColumnView.h"

#include "PREFOS_ColumnReductionInternal.h"
#include "core/PREFOS_Timer.h"
#include "core/PREFOS_WorkingMatrix.h"

#include <limits.h>
#include <stdint.h>
#include <stdio.h>

#define PREFOS_TRANSFORMED_VIEW_MIN_DIRTY_NNZ_LIMIT 262144U
#define PREFOS_TRANSFORMED_VIEW_MAX_DIRTY_NNZ_LIMIT 1048576U
#define PREFOS_TRANSFORMED_VIEW_DIRTY_NNZ_DIVISOR 32U
#define PREFOS_TRANSFORMED_VIEW_MAX_AFFECTED_NNZ 1048576U
#define PREFOS_TRANSFORMED_HASH_INV_PRECISION 1e6

typedef struct
{
    int row;
    int column;
    double value;
} PreFOSExpandedEntry;

static int transformed_row_is_dirty(
    const PreFOSPresolver *presolver,
    const PreFOSColumnWorkspace *workspace, size_t row)
{
    return workspace->dirty_row[row] ||
           (presolver->rows_require_materialization &&
            presolver->rows_require_materialization[row]);
}

static int transformed_column_is_eligible(
    const PreFOSPresolver *presolver,
    const PreFOSColumnWorkspace *workspace, size_t column)
{
    return column < presolver->original.n &&
           presolver->variable_to_box[column] >= 0 &&
           !presolver->is_fixed[column] &&
           !presolver->is_substituted[column] &&
           !presolver->is_parallel_removed[column] &&
           !presolver->affine_protected_columns[column] &&
           !workspace->quadratic[column] &&
           !workspace->factor[column] &&
           !workspace->protected_target[column];
}

static int reserve_expanded_entries(
    PreFOSExpandedEntry **entries, size_t *capacity, size_t required)
{
    PreFOSExpandedEntry *next_entries;
    size_t next;
    if (required <= *capacity) return 1;
    next = *capacity < 1024 ? 1024 : *capacity;
    while (next < required)
    {
        size_t growth = next / 2 + 1;
        if (growth > SIZE_MAX - next)
        {
            next = required;
            break;
        }
        next += growth;
    }
    if (next > SIZE_MAX / sizeof(**entries)) return 0;
    next_entries = (PreFOSExpandedEntry *) realloc(
        *entries, next * sizeof(**entries));
    if (!next_entries) return 0;
    *entries = next_entries;
    *capacity = next;
    return 1;
}

static int compare_hash_keys(const void *left, const void *right)
{
    uint64_t left_key = *(const uint64_t *) left;
    uint64_t right_key = *(const uint64_t *) right;
    return (left_key > right_key) - (left_key < right_key);
}

static int hash_key_present(
    const uint64_t *keys, size_t count, uint64_t key)
{
    size_t begin = 0, end = count;
    while (begin < end)
    {
        size_t middle = begin + (end - begin) / 2;
        if (keys[middle] < key)
            begin = middle + 1;
        else
            end = middle;
    }
    return begin < count && keys[begin] == key;
}

static uint64_t cached_column_hash(
    const PreFOSColumnWorkspace *workspace, int column)
{
    return ((uint64_t) (uint32_t)
                workspace->parallel_support_hashes[column]
            << 32U) |
           (uint32_t)
               workspace->parallel_coefficient_hashes[column];
}

static int transformed_column_hash(
    const int *starts, const int *ends, const int *rows,
    const double *values, size_t column, uint64_t *key)
{
    int begin = starts[column];
    int end = ends[column];
    uint32_t support = 5381U;
    uint32_t coefficients = 5381U;
    double maximum, scale;
    int position;
    if (begin >= end || values[begin] == 0.0) return 0;
    maximum = fabs(values[begin]);
    for (position = begin; position < end; ++position)
    {
        support =
            ((support << 5U) + support) +
            (uint32_t) rows[position];
        maximum = fmax(maximum, fabs(values[position]));
    }
    if (maximum == 0.0) return 0;
    scale = values[begin] > 0.0
                ? 1.0 / maximum
                : -1.0 / maximum;
    for (position = begin; position < end; ++position)
    {
        uint32_t normalized = (uint32_t) round(
            values[position] * scale *
            PREFOS_TRANSFORMED_HASH_INV_PRECISION);
        coefficients =
            ((coefficients << 5U) + coefficients) + normalized;
    }
    *key = ((uint64_t) support << 32U) | coefficients;
    return 1;
}

void prefos_internal_free_transformed_column_view(
    PreFOSTransformedColumnView *view)
{
    if (!view) return;
    free(view->starts);
    free(view->ends);
    free(view->rows);
    free(view->values);
    free(view->live_degrees);
    free(view->eligible);
    memset(view, 0, sizeof(*view));
}

PreFOSStatus prefos_internal_build_transformed_column_view(
    const PreFOSPresolver *presolver,
    const PreFOSColumnWorkspace *workspace,
    PreFOSTransformedColumnView *view, int *built)
{
    const PreFOSCsrMatrix *matrix;
    PreFOSExpandedEntry *entries = NULL;
    unsigned char *affected = NULL;
    double *row_values = NULL;
    int *row_marks = NULL, *touched_columns = NULL;
    int *expanded_starts = NULL, *expanded_rows = NULL;
    double *expanded_values = NULL;
    int *cursor = NULL;
    int *exact_starts = NULL, *exact_ends = NULL;
    int *exact_rows = NULL, *exact_live_degrees = NULL;
    double *exact_values = NULL;
    unsigned char *eligible = NULL;
    uint64_t *affected_hashes = NULL;
    size_t entry_count = 0, entry_capacity = 0;
    size_t dirty_limit, row, column, position;
    size_t expanded_nnz = 0, exact_nnz = 0;
    size_t affected_columns = 0, affected_candidates = 0;
    size_t candidate_columns = 0, clean_candidate_nnz = 0;
    size_t affected_list_count = 0, clean_candidate_count = 0;
    size_t affected_hash_count = 0;
    double dirty_scan_ms = 0.0, expand_ms = 0.0;
    double expanded_csc_ms = 0.0, exact_csc_ms = 0.0;
    double match_ms = 0.0, final_copy_ms = 0.0;
    int trace =
        getenv("PREFOS_TRACE_PARALLEL_COLUMNS") != NULL;
    PreFOSTimestamp trace_start, trace_stop;
    PreFOSStatus status = PREFOS_STATUS_OK;

    if (!presolver || !workspace || !view || !built)
        return PREFOS_STATUS_INVALID_ARGUMENT;
    memset(view, 0, sizeof(*view));
    *built = 0;
    if (presolver->working_matrix_is_materialized ||
        presolver->n_rows_require_materialization == 0 ||
        presolver->original.n < 2 ||
        getenv("PREFOS_DISABLE_TRANSFORMED_COLUMN_VIEW"))
        return PREFOS_STATUS_OK;
    if (trace) prefos_internal_timer_now(&trace_start);
    matrix = &presolver->original.A;
    for (row = 0; row < matrix->rows; ++row)
        if (!presolver->remove_rows[row] &&
            transformed_row_is_dirty(presolver, workspace, row))
        {
            ++view->dirty_rows;
            view->dirty_source_nnz +=
                (size_t) (matrix->row_pointers[row + 1] -
                          matrix->row_pointers[row]);
        }
    if (view->dirty_rows == 0) return PREFOS_STATUS_OK;
    if (trace)
    {
        prefos_internal_timer_now(&trace_stop);
        dirty_scan_ms = prefos_internal_timer_elapsed_milliseconds(
            &trace_start, &trace_stop);
        trace_start = trace_stop;
    }
    dirty_limit =
        matrix->nnz / PREFOS_TRANSFORMED_VIEW_DIRTY_NNZ_DIVISOR;
    if (dirty_limit < PREFOS_TRANSFORMED_VIEW_MIN_DIRTY_NNZ_LIMIT)
        dirty_limit = PREFOS_TRANSFORMED_VIEW_MIN_DIRTY_NNZ_LIMIT;
    if (dirty_limit > PREFOS_TRANSFORMED_VIEW_MAX_DIRTY_NNZ_LIMIT)
        dirty_limit = PREFOS_TRANSFORMED_VIEW_MAX_DIRTY_NNZ_LIMIT;
    if (view->dirty_source_nnz > dirty_limit)
        return PREFOS_STATUS_OK;

    affected = (unsigned char *) calloc(
        presolver->original.n, sizeof(unsigned char));
    row_values = (double *) calloc(
        presolver->original.n, sizeof(double));
    row_marks = (int *) prefos_internal_alloc_array(
        presolver->original.n, sizeof(int));
    touched_columns = (int *) prefos_internal_alloc_array(
        presolver->original.n, sizeof(int));
    if (presolver->original.n > 0 &&
        (!affected || !row_values || !row_marks ||
         !touched_columns))
        goto optional_oom;
    for (column = 0; column < presolver->original.n; ++column)
        row_marks[column] = -1;
    for (position = workspace->parallel_cache_removed_row_cursor;
         position < presolver->n_removed_rows; ++position)
    {
        int removed_row = presolver->removed_row_log[position];
        int source_position;
        if (removed_row < 0 ||
            (size_t) removed_row >= matrix->rows)
            continue;
        for (source_position = matrix->row_pointers[removed_row];
             source_position <
                 matrix->row_pointers[removed_row + 1];
             ++source_position)
            if (matrix->values[source_position] != 0.0)
                affected[matrix->column_indices[source_position]] = 1;
    }

    for (row = 0; row < matrix->rows; ++row)
    {
        size_t n_touched = 0;
        double shift = 0.0;
        int source_position;
        if (presolver->remove_rows[row] ||
            !transformed_row_is_dirty(presolver, workspace, row))
            continue;
        for (source_position = matrix->row_pointers[row];
             source_position < matrix->row_pointers[row + 1];
             ++source_position)
            if (matrix->values[source_position] != 0.0)
                affected[matrix->column_indices[source_position]] = 1;
        status = prefos_internal_expand_working_row(
            presolver, matrix, row, row_values, row_marks,
            touched_columns, &n_touched, &shift);
        if (status != PREFOS_STATUS_OK)
        {
            prefos_internal_clear_expanded_working_row(
                row_values, row_marks, touched_columns, n_touched);
            goto cleanup;
        }
        if (!reserve_expanded_entries(
                &entries, &entry_capacity,
                entry_count + n_touched))
        {
            prefos_internal_clear_expanded_working_row(
                row_values, row_marks, touched_columns, n_touched);
            goto optional_oom;
        }
        for (position = 0; position < n_touched; ++position)
        {
            int touched = touched_columns[position];
            double value = row_values[touched];
            if (value == 0.0) continue;
            affected[touched] = 1;
            entries[entry_count++] = (PreFOSExpandedEntry){
                (int) row, touched, value};
        }
        prefos_internal_clear_expanded_working_row(
            row_values, row_marks, touched_columns, n_touched);
    }
    free(row_values);
    row_values = NULL;
    for (column = 0; column < presolver->original.n; ++column)
        if (affected[column])
            row_marks[affected_list_count++] = (int) column;
    if (trace)
    {
        prefos_internal_timer_now(&trace_stop);
        expand_ms = prefos_internal_timer_elapsed_milliseconds(
            &trace_start, &trace_stop);
        trace_start = trace_stop;
    }

    expanded_starts = (int *) calloc(
        presolver->original.n + 1, sizeof(int));
    if (!expanded_starts) goto optional_oom;
    for (position = 0; position < entry_count; ++position)
    {
        int entry_column = entries[position].column;
        if (expanded_starts[entry_column + 1] == INT_MAX)
        {
            status = PREFOS_STATUS_NUMERICAL_ERROR;
            goto cleanup;
        }
        ++expanded_starts[entry_column + 1];
    }
    for (column = 0; column < presolver->original.n; ++column)
    {
        if ((size_t) expanded_starts[column + 1] >
            (size_t) INT_MAX - expanded_nnz)
        {
            status = PREFOS_STATUS_NUMERICAL_ERROR;
            goto cleanup;
        }
        expanded_nnz += (size_t) expanded_starts[column + 1];
        expanded_starts[column + 1] = (int) expanded_nnz;
    }
    expanded_rows = (int *) prefos_internal_alloc_array(
        expanded_nnz, sizeof(int));
    expanded_values = (double *) prefos_internal_alloc_array(
        expanded_nnz, sizeof(double));
    cursor = (int *) prefos_internal_alloc_array(
        presolver->original.n, sizeof(int));
    if ((expanded_nnz > 0 &&
         (!expanded_rows || !expanded_values)) ||
        (presolver->original.n > 0 && !cursor))
        goto optional_oom;
    if (presolver->original.n > 0)
        memcpy(
            cursor, expanded_starts,
            presolver->original.n * sizeof(int));
    for (position = 0; position < entry_count; ++position)
    {
        int entry_column = entries[position].column;
        int write = cursor[entry_column]++;
        expanded_rows[write] = entries[position].row;
        expanded_values[write] = entries[position].value;
    }
    free(entries);
    entries = NULL;
    if (trace)
    {
        prefos_internal_timer_now(&trace_stop);
        expanded_csc_ms =
            prefos_internal_timer_elapsed_milliseconds(
                &trace_start, &trace_stop);
        trace_start = trace_stop;
    }

    exact_starts = (int *) calloc(
        presolver->original.n + 1, sizeof(int));
    exact_ends = (int *) calloc(
        presolver->original.n, sizeof(int));
    exact_live_degrees = (int *) calloc(
        presolver->original.n, sizeof(int));
    eligible = (unsigned char *) calloc(
        presolver->original.n, sizeof(unsigned char));
    if (!exact_starts ||
        (presolver->original.n > 0 &&
         (!exact_ends || !exact_live_degrees || !eligible)))
        goto optional_oom;
    for (position = 0; position < affected_list_count; ++position)
    {
        column = (size_t) row_marks[position];
        if (transformed_column_is_eligible(
                presolver, workspace, column))
        {
            size_t count =
                (size_t) (expanded_starts[column + 1] -
                          expanded_starts[column]);
            int workspace_position;
            ++affected_columns;
            for (workspace_position = workspace->starts[column];
                 workspace_position < workspace->ends[column];
                 ++workspace_position)
            {
                int workspace_row =
                    workspace->rows[workspace_position];
                if (!presolver->remove_rows[workspace_row] &&
                    !transformed_row_is_dirty(
                        presolver, workspace,
                        (size_t) workspace_row))
                    ++count;
            }
            if (count > (size_t) INT_MAX)
            {
                status = PREFOS_STATUS_NUMERICAL_ERROR;
                goto cleanup;
            }
            if (count > (size_t) INT_MAX - exact_nnz)
            {
                status = PREFOS_STATUS_NUMERICAL_ERROR;
                goto cleanup;
            }
            exact_starts[column] = (int) exact_nnz;
            exact_live_degrees[column] = (int) count;
            exact_nnz += count;
        }
    }
    exact_starts[presolver->original.n] = (int) exact_nnz;
    if (exact_nnz > PREFOS_TRANSFORMED_VIEW_MAX_AFFECTED_NNZ &&
        exact_nnz - PREFOS_TRANSFORMED_VIEW_MAX_AFFECTED_NNZ >
            workspace->nnz / 2)
        goto cleanup;
    exact_rows = (int *) prefos_internal_alloc_array(
        exact_nnz, sizeof(int));
    exact_values = (double *) prefos_internal_alloc_array(
        exact_nnz, sizeof(double));
    if (exact_nnz > 0 && (!exact_rows || !exact_values))
        goto optional_oom;

    for (position = 0; position < affected_list_count; ++position)
    {
        int expected;
        column = (size_t) row_marks[position];
        int clean = workspace->starts[column];
        int clean_end = workspace->ends[column];
        int expanded = expanded_starts[column];
        int expanded_end = expanded_starts[column + 1];
        int write = exact_starts[column];
        if (exact_live_degrees[column] == 0) continue;
        expected = write + exact_live_degrees[column];
        for (;;)
        {
            int clean_row = INT_MAX;
            int expanded_row = INT_MAX;
            while (clean < clean_end)
            {
                clean_row = workspace->rows[clean];
                if (!presolver->remove_rows[clean_row] &&
                    !transformed_row_is_dirty(
                        presolver, workspace,
                        (size_t) clean_row))
                    break;
                ++clean;
                clean_row = INT_MAX;
            }
            if (expanded < expanded_end)
                expanded_row = expanded_rows[expanded];
            if (clean_row == INT_MAX &&
                expanded_row == INT_MAX)
                break;
            if (clean_row < expanded_row)
            {
                exact_rows[write] = clean_row;
                exact_values[write++] = workspace->values[clean++];
            }
            else
            {
                exact_rows[write] = expanded_row;
                exact_values[write++] = expanded_values[expanded++];
            }
        }
        if (write != expected)
        {
            status = PREFOS_STATUS_NUMERICAL_ERROR;
            goto cleanup;
        }
        exact_ends[column] = write;
        if (exact_live_degrees[column] > 0)
        {
            eligible[column] = 1;
            ++affected_candidates;
        }
    }
    if (affected_candidates == 0) goto cleanup;
    if (trace)
    {
        prefos_internal_timer_now(&trace_stop);
        exact_csc_ms = prefos_internal_timer_elapsed_milliseconds(
            &trace_start, &trace_stop);
        trace_start = trace_stop;
    }

    affected_hashes = (uint64_t *) prefos_internal_alloc_array(
        affected_candidates, sizeof(uint64_t));
    if (!affected_hashes) goto optional_oom;
    for (position = 0; position < affected_list_count; ++position)
    {
        column = (size_t) row_marks[position];
        if (eligible[column] &&
            transformed_column_hash(
                exact_starts, exact_ends, exact_rows,
                exact_values, column,
                &affected_hashes[affected_hash_count]))
            ++affected_hash_count;
    }
    if (affected_hash_count == 0) goto cleanup;
    qsort(
        affected_hashes, affected_hash_count, sizeof(uint64_t),
        compare_hash_keys);
    {
        size_t unique = 0;
        for (position = 0;
             position < affected_hash_count; ++position)
            if (unique == 0 ||
                affected_hashes[position] !=
                    affected_hashes[unique - 1])
                affected_hashes[unique++] =
                    affected_hashes[position];
        affected_hash_count = unique;
    }

    candidate_columns = affected_candidates;
    if (workspace->parallel_no_group_cache_valid &&
        workspace->parallel_columns &&
        workspace->parallel_support_hashes &&
        workspace->parallel_coefficient_hashes &&
        workspace->parallel_column_dirty)
    {
        size_t cached_position = 0, key_position = 0;
        while (cached_position <
                   workspace->parallel_cached_active_columns &&
               key_position < affected_hash_count)
        {
            int cached_column =
                workspace->parallel_columns[cached_position];
            uint64_t cached_key =
                cached_column_hash(workspace, cached_column);
            uint64_t affected_key =
                affected_hashes[key_position];
            if (cached_key < affected_key)
            {
                ++cached_position;
                continue;
            }
            if (cached_key > affected_key)
            {
                ++key_position;
                continue;
            }
            do
            {
                cached_column =
                    workspace->parallel_columns[cached_position];
                if (!affected[cached_column] &&
                    !workspace->parallel_column_dirty[cached_column] &&
                    transformed_column_is_eligible(
                        presolver, workspace,
                        (size_t) cached_column) &&
                    workspace->starts[cached_column] <
                        workspace->ends[cached_column])
                {
                    eligible[cached_column] = 1;
                    touched_columns[clean_candidate_count++] =
                        cached_column;
                    ++candidate_columns;
                    clean_candidate_nnz +=
                        (size_t)
                            (workspace->ends[cached_column] -
                             workspace->starts[cached_column]);
                }
                ++cached_position;
            } while (
                cached_position <
                    workspace->parallel_cached_active_columns &&
                cached_column_hash(
                    workspace,
                    workspace->parallel_columns[cached_position]) ==
                    affected_key);
            ++key_position;
        }
        for (position = 0;
             position < workspace->n_parallel_dirty_columns;
             ++position)
        {
            int dirty_column =
                workspace->parallel_dirty_columns[position];
            uint64_t key;
            if (dirty_column < 0 ||
                (size_t) dirty_column >=
                    presolver->original.n ||
                affected[dirty_column] ||
                eligible[dirty_column] ||
                !transformed_column_is_eligible(
                    presolver, workspace,
                    (size_t) dirty_column) ||
                workspace->starts[dirty_column] >=
                    workspace->ends[dirty_column] ||
                !transformed_column_hash(
                    workspace->starts, workspace->ends,
                    workspace->rows, workspace->values,
                    (size_t) dirty_column, &key) ||
                !hash_key_present(
                    affected_hashes, affected_hash_count, key))
                continue;
            eligible[dirty_column] = 1;
            touched_columns[clean_candidate_count++] =
                dirty_column;
            ++candidate_columns;
            clean_candidate_nnz +=
                (size_t) (workspace->ends[dirty_column] -
                          workspace->starts[dirty_column]);
        }
    }
    else
        for (column = 0; column < presolver->original.n; ++column)
            if (!affected[column] &&
                transformed_column_is_eligible(
                    presolver, workspace, column) &&
                workspace->starts[column] < workspace->ends[column])
            {
                uint64_t key;
                if (transformed_column_hash(
                        workspace->starts, workspace->ends,
                        workspace->rows, workspace->values,
                        column, &key) &&
                    hash_key_present(
                        affected_hashes, affected_hash_count, key))
                {
                    eligible[column] = 1;
                    touched_columns[clean_candidate_count++] =
                        (int) column;
                    ++candidate_columns;
                    clean_candidate_nnz +=
                        (size_t) (workspace->ends[column] -
                                  workspace->starts[column]);
                }
            }
    if (candidate_columns < 2) goto cleanup;
    if (trace)
    {
        prefos_internal_timer_now(&trace_stop);
        match_ms = prefos_internal_timer_elapsed_milliseconds(
            &trace_start, &trace_stop);
        trace_start = trace_stop;
    }

    if (clean_candidate_nnz > 0)
    {
        int *grown_rows;
        double *grown_values;
        size_t required_nnz, write;
        if (clean_candidate_nnz > (size_t) INT_MAX - exact_nnz)
        {
            status = PREFOS_STATUS_NUMERICAL_ERROR;
            goto cleanup;
        }
        required_nnz = exact_nnz + clean_candidate_nnz;
        grown_rows = (int *) realloc(
            exact_rows, required_nnz * sizeof(int));
        if (!grown_rows) goto optional_oom;
        exact_rows = grown_rows;
        grown_values = (double *) realloc(
            exact_values, required_nnz * sizeof(double));
        if (!grown_values) goto optional_oom;
        exact_values = grown_values;
        write = exact_nnz;
        for (position = 0; position < clean_candidate_count; ++position)
        {
            int source_start, source_end;
            size_t count;
            column = (size_t) touched_columns[position];
            source_start = workspace->starts[column];
            source_end = workspace->ends[column];
            count = (size_t) (source_end - source_start);
            exact_starts[column] = (int) write;
            memcpy(
                exact_rows + write,
                workspace->rows + source_start,
                count * sizeof(int));
            memcpy(
                exact_values + write,
                workspace->values + source_start,
                count * sizeof(double));
            write += count;
            exact_ends[column] = (int) write;
            exact_live_degrees[column] = (int) count;
        }
        if (write != required_nnz)
        {
            status = PREFOS_STATUS_NUMERICAL_ERROR;
            goto cleanup;
        }
        exact_starts[presolver->original.n] =
            (int) required_nnz;
        exact_nnz = required_nnz;
    }
    if (trace)
    {
        prefos_internal_timer_now(&trace_stop);
        final_copy_ms =
            prefos_internal_timer_elapsed_milliseconds(
                &trace_start, &trace_stop);
    }

    view->starts = exact_starts;
    view->ends = exact_ends;
    view->rows = exact_rows;
    view->values = exact_values;
    view->live_degrees = exact_live_degrees;
    view->eligible = eligible;
    view->nnz = exact_nnz;
    view->affected_columns = affected_columns;
    view->candidate_columns = candidate_columns;
    exact_starts = NULL;
    exact_ends = NULL;
    exact_rows = NULL;
    exact_values = NULL;
    exact_live_degrees = NULL;
    eligible = NULL;
    *built = 1;
    if (trace)
        fprintf(
            stderr,
            "PreFOS transformed-column view dirty_rows=%zu "
            "dirty_source_nnz=%zu affected=%zu candidates=%zu nnz=%zu "
            "scan_ms=%.3f expand_ms=%.3f expanded_csc_ms=%.3f "
            "exact_csc_ms=%.3f match_ms=%.3f final_copy_ms=%.3f\n",
            view->dirty_rows, view->dirty_source_nnz,
            view->affected_columns, view->candidate_columns,
            view->nnz, dirty_scan_ms, expand_ms, expanded_csc_ms,
            exact_csc_ms, match_ms, final_copy_ms);
    goto cleanup;

optional_oom:
    status = PREFOS_STATUS_OK;

cleanup:
    free(entries);
    free(affected);
    free(row_values);
    free(row_marks);
    free(touched_columns);
    free(expanded_starts);
    free(expanded_rows);
    free(expanded_values);
    free(cursor);
    free(exact_starts);
    free(exact_ends);
    free(exact_rows);
    free(exact_values);
    free(exact_live_degrees);
    free(eligible);
    free(affected_hashes);
    return status;
}
