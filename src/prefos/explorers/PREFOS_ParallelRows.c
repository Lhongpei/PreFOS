/*
 * Copyright 2026 Hongpei Li
 * SPDX-License-Identifier: Apache-2.0
 */

#include "PREFOS_ParallelRows.h"

#include "PREFOS_ColumnReductionInternal.h"
#include "ParallelRowDetection.h"
#include "ParallelRowReduction.h"
#include "core/PREFOS_Timer.h"
#include "PREFOS_CudaBackend.h"

#include <stdint.h>
#include <stdio.h>

#define PREFOS_PARALLEL_ROW_PREFILTER_AVERAGE_NNZ 256.0
#define PREFOS_PARALLEL_ROW_HASH_INV_PRECISION 1e6
#define PREFOS_SPARSE_MATERIALIZED_SUPPORT_MIN_NNZ 262144U
#define PREFOS_INCREMENTAL_PARALLEL_ROW_MIN_ROWS 32768U
#define PREFOS_INCREMENTAL_PARALLEL_ROW_MAX_DIRTY 1048576U

static int trace_parallel_rows(void)
{
    const char *value = getenv("PREFOS_TRACE_PARALLEL_ROWS");
    return value && *value && *value != '0';
}

typedef struct
{
    uint64_t support_hash;
    uint32_t coefficient_hash;
    int length;
} PreFOSParallelSupportKey;

static size_t saturated_size_add(size_t left, size_t right)
{
    return right > SIZE_MAX - left ? SIZE_MAX : left + right;
}

static size_t saturated_size_multiply(size_t left, size_t right)
{
    return left != 0 && right > SIZE_MAX / left
               ? SIZE_MAX
               : left * right;
}

static int compare_parallel_support_keys(
    const void *left_pointer, const void *right_pointer)
{
    const PreFOSParallelSupportKey *left =
        (const PreFOSParallelSupportKey *) left_pointer;
    const PreFOSParallelSupportKey *right =
        (const PreFOSParallelSupportKey *) right_pointer;
    if (left->length != right->length)
        return (left->length > right->length) -
               (left->length < right->length);
    if (left->support_hash != right->support_hash)
        return (left->support_hash > right->support_hash) -
               (left->support_hash < right->support_hash);
    return (left->coefficient_hash > right->coefficient_hash) -
           (left->coefficient_hash < right->coefficient_hash);
}

static int compare_support_columns(
    const void *left_pointer, const void *right_pointer)
{
    int left = *(const int *) left_pointer;
    int right = *(const int *) right_pointer;
    return (left > right) - (left < right);
}

static void sort_support_columns(int *columns, size_t count)
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
                columns[insertion] =
                    columns[insertion - 1];
                --insertion;
            }
            columns[insertion] = column;
        }
        return;
    }
    qsort(
        columns, count, sizeof(*columns),
        compare_support_columns);
}

static size_t lower_bound_parallel_support_prefix(
    const PreFOSParallelSupportKey *keys, size_t count,
    int length, uint64_t support_hash)
{
    size_t lower = 0;
    size_t upper = count;
    while (lower < upper)
    {
        size_t middle = lower + (upper - lower) / 2;
        if (keys[middle].length < length ||
            (keys[middle].length == length &&
             keys[middle].support_hash < support_hash))
            lower = middle + 1;
        else
            upper = middle;
    }
    return lower;
}

static int exact_rows_match_transformed_support(
    const PreFOSPresolver *presolver, const PreFOSCsrMatrix *matrix,
    const PreFOSParallelSupportKey *keys, size_t key_count)
{
    size_t row;
    if (key_count == 0) return 0;
    for (row = 0; row < matrix->rows; ++row)
    {
        uint64_t support_hash = UINT64_C(1469598103934665603);
        uint32_t coefficient_hash = 5381U;
        double coefficient_maximum = 0.0;
        double first_coefficient = 0.0;
        size_t candidate;
        int begin, end, position, length = 0;

        if (presolver->remove_rows[row] ||
            !prefos_internal_row_has_exact_linear_form(
                presolver, row))
            continue;
        begin = matrix->row_pointers[row];
        end = matrix->row_pointers[row + 1];
        for (position = begin; position < end; ++position)
        {
            int column = matrix->column_indices[position];
            double coefficient = matrix->values[position];
            if (coefficient == 0.0 ||
                presolver->is_fixed[column] ||
                presolver->is_substituted[column] ||
                presolver->is_parallel_removed[column] ||
                !prefos_internal_term_is_active_in_row(
                    presolver, row, column))
                continue;
            if (length == 0) first_coefficient = coefficient;
            coefficient_maximum =
                fmax(coefficient_maximum, fabs(coefficient));
            support_hash ^= (uint32_t) column;
            support_hash *= UINT64_C(1099511628211);
            ++length;
        }
        if (length <= 0) continue;
        support_hash ^= (uint32_t) length;
        candidate = lower_bound_parallel_support_prefix(
            keys, key_count, length, support_hash);
        if (candidate == key_count ||
            keys[candidate].length != length ||
            keys[candidate].support_hash != support_hash)
            continue;

        {
            double scale =
                first_coefficient > 0.0
                    ? 1.0 / coefficient_maximum
                    : -1.0 / coefficient_maximum;
            for (position = begin; position < end; ++position)
            {
                int column = matrix->column_indices[position];
                double coefficient = matrix->values[position];
                uint32_t normalized;
                if (coefficient == 0.0 ||
                    presolver->is_fixed[column] ||
                    presolver->is_substituted[column] ||
                    presolver->is_parallel_removed[column] ||
                    !prefos_internal_term_is_active_in_row(
                        presolver, row, column))
                    continue;
                normalized = (uint32_t) round(
                    coefficient * scale *
                    PREFOS_PARALLEL_ROW_HASH_INV_PRECISION);
                coefficient_hash =
                    ((coefficient_hash << 5U) +
                     coefficient_hash) +
                    normalized;
            }
        }
        while (candidate < key_count &&
               keys[candidate].length == length &&
               keys[candidate].support_hash == support_hash)
        {
            if (keys[candidate].coefficient_hash ==
                coefficient_hash)
                return 1;
            ++candidate;
        }
    }
    return 0;
}

static int accumulate_expanded_support(
    const PreFOSPresolver *presolver, int column, double coefficient,
    size_t depth, double *values, uint32_t *marks, uint32_t epoch,
    int *touched_columns, size_t *n_touched)
{
    size_t term;
    if (column < 0 ||
        (size_t) column >= presolver->original.n ||
        depth > PREFOS_MAX_SUBSTITUTION_DEPTH)
        return 0;
    if (presolver->is_fixed[column] ||
        presolver->is_parallel_removed[column])
        return 1;
    if (presolver->is_substituted[column])
    {
        size_t start =
            presolver->substitution_term_start[column];
        size_t count =
            presolver->substitution_term_count[column];
        if (count == 0 ||
            start > presolver->n_substitution_terms ||
            count > presolver->n_substitution_terms - start)
            return 0;
        for (term = 0; term < count; ++term)
        {
            double propagated;
            if (!prefos_internal_safe_product(
                    coefficient,
                    presolver->substitution_scales[start + term],
                    &propagated) ||
                !accumulate_expanded_support(
                    presolver,
                    presolver->substitution_targets[start + term],
                    propagated, depth + 1, values, marks, epoch,
                    touched_columns, n_touched))
                return 0;
        }
        return 1;
    }
    if (marks[column] != epoch)
    {
        marks[column] = epoch;
        values[column] = coefficient;
        touched_columns[(*n_touched)++] = column;
        return 1;
    }
    return prefos_internal_safe_add_product(
        &values[column], 1.0, coefficient);
}

static int parallel_support_is_promising(
    const PreFOSPresolver *presolver, const PreFOSCsrMatrix *matrix,
    int filter_transformed_columns,
    int *has_transformed_trivial_support,
    int *has_transformed_bounded_doubleton,
    int transformed_rows_only,
    const PreFOSColumnWorkspace *workspace)
{
    PreFOSParallelSupportKey *keys;
    double *expanded_values = NULL;
    uint32_t *expanded_marks = NULL;
    int *expanded_columns = NULL;
    uint32_t expanded_epoch = 0;
    size_t active_count = 0, potential_nnz = 0;
    size_t scan, scan_count, start;
    int use_materialization_log =
        transformed_rows_only &&
        presolver->materialization_row_log_complete &&
        presolver->n_materialization_row_log ==
            presolver->n_rows_require_materialization;

    if (has_transformed_trivial_support)
        *has_transformed_trivial_support = 0;
    if (has_transformed_bounded_doubleton)
        *has_transformed_bounded_doubleton = 0;
    scan_count =
        use_materialization_log
            ? presolver->n_rows_require_materialization
            : matrix->rows;
    if (scan_count == 0) return 0;
    keys = (PreFOSParallelSupportKey *)
        prefos_internal_alloc_array(scan_count, sizeof(*keys));
    if (!keys) return 1;
    if (filter_transformed_columns &&
        presolver->n_rows_require_materialization > 0 &&
        matrix->cols <= 2000000)
    {
        expanded_values = (double *)
            prefos_internal_alloc_array(
                matrix->cols, sizeof(double));
        expanded_marks = (uint32_t *)
            calloc(matrix->cols, sizeof(uint32_t));
        expanded_columns = (int *)
            prefos_internal_alloc_array(
                matrix->cols, sizeof(int));
        if ((matrix->cols > 0) &&
            (!expanded_values || !expanded_marks ||
             !expanded_columns))
        {
            free(expanded_values);
            free(expanded_marks);
            free(expanded_columns);
            free(keys);
            return 1;
        }
    }
    for (scan = 0; scan < scan_count; ++scan)
    {
        size_t row =
            use_materialization_log
                ? (size_t) presolver->materialization_row_log[scan]
                : scan;
        size_t n_touched = 0;
        int begin, end, position, length;
        int exact_linear_form;
        uint64_t hash = UINT64_C(1469598103934665603);
        uint32_t coefficient_hash = 5381U;
        double coefficient_maximum = 0.0;
        double first_coefficient = 0.0;
        if (presolver->remove_rows[row])
            continue;
        exact_linear_form =
            prefos_internal_row_has_exact_linear_form(
                presolver, row);
        if (transformed_rows_only && exact_linear_form)
            continue;
        if (!filter_transformed_columns && !exact_linear_form)
            continue;
        begin = matrix->row_pointers[row];
        end = matrix->row_pointers[row + 1];
        length = 0;
        if (!exact_linear_form && expanded_values)
        {
            size_t touched;
            if (++expanded_epoch == 0)
            {
                memset(
                    expanded_marks, 0,
                    matrix->cols * sizeof(uint32_t));
                expanded_epoch = 1;
            }
            for (position = begin; position < end; ++position)
            {
                int column = matrix->column_indices[position];
                if (matrix->values[position] == 0.0 ||
                    !prefos_internal_term_is_active_in_row(
                        presolver, row, column))
                    continue;
                if (!accumulate_expanded_support(
                        presolver, column, matrix->values[position],
                        0, expanded_values, expanded_marks,
                        expanded_epoch, expanded_columns,
                        &n_touched))
                {
                    free(expanded_values);
                    free(expanded_marks);
                    free(expanded_columns);
                    free(keys);
                    return 1;
                }
            }
            sort_support_columns(
                expanded_columns, n_touched);
            for (touched = 0; touched < n_touched; ++touched)
            {
                int column = expanded_columns[touched];
                double coefficient = expanded_values[column];
                if (coefficient == 0.0) continue;
                if (length == 0) first_coefficient = coefficient;
                coefficient_maximum =
                    fmax(coefficient_maximum, fabs(coefficient));
                hash ^= (uint32_t) column;
                hash *= UINT64_C(1099511628211);
                ++length;
            }
            if (length > 0)
            {
                double scale =
                    first_coefficient > 0.0
                        ? 1.0 / coefficient_maximum
                        : -1.0 / coefficient_maximum;
                for (touched = 0; touched < n_touched; ++touched)
                {
                    int column = expanded_columns[touched];
                    double coefficient = expanded_values[column];
                    uint32_t normalized;
                    if (coefficient == 0.0) continue;
                    normalized = (uint32_t) round(
                        coefficient * scale *
                        PREFOS_PARALLEL_ROW_HASH_INV_PRECISION);
                    coefficient_hash =
                        ((coefficient_hash << 5U) +
                         coefficient_hash) +
                        normalized;
                }
            }
        }
        else
        {
            for (position = begin; position < end; ++position)
            {
                int column = matrix->column_indices[position];
                double coefficient = matrix->values[position];
                if (coefficient == 0.0)
                    continue;
                if (filter_transformed_columns &&
                    (presolver->is_fixed[column] ||
                     presolver->is_substituted[column] ||
                     presolver->is_parallel_removed[column] ||
                     !prefos_internal_term_is_active_in_row(
                         presolver, row, column)))
                    continue;
                if (length == 0) first_coefficient = coefficient;
                coefficient_maximum =
                    fmax(coefficient_maximum, fabs(coefficient));
                hash ^= (uint32_t) column;
                hash *= UINT64_C(1099511628211);
                ++length;
            }
            if (length > 0)
            {
                double scale =
                    first_coefficient > 0.0
                        ? 1.0 / coefficient_maximum
                        : -1.0 / coefficient_maximum;
                for (position = begin; position < end; ++position)
                {
                    int column = matrix->column_indices[position];
                    double coefficient = matrix->values[position];
                    uint32_t normalized;
                    if (coefficient == 0.0)
                        continue;
                    if (filter_transformed_columns &&
                        (presolver->is_fixed[column] ||
                         presolver->is_substituted[column] ||
                         presolver->is_parallel_removed[column] ||
                         !prefos_internal_term_is_active_in_row(
                             presolver, row, column)))
                        continue;
                    normalized = (uint32_t) round(
                        coefficient * scale *
                        PREFOS_PARALLEL_ROW_HASH_INV_PRECISION);
                    coefficient_hash =
                        ((coefficient_hash << 5U) +
                         coefficient_hash) +
                        normalized;
                }
            }
        }
        if (!exact_linear_form && expanded_values &&
            length <= 1 &&
            has_transformed_trivial_support)
            *has_transformed_trivial_support = 1;
        if (!exact_linear_form && expanded_values &&
            length == 2 && has_transformed_bounded_doubleton &&
            workspace &&
            isfinite(presolver->working_constraint_lower[row]) &&
            presolver->working_constraint_lower[row] ==
                presolver->working_constraint_upper[row])
        {
            size_t eligible = 0;
            size_t touched;
            for (touched = 0; touched < n_touched; ++touched)
            {
                int column = expanded_columns[touched];
                if (expanded_values[column] == 0.0)
                    continue;
                if (column < 0 ||
                    (size_t) column >= presolver->original.n ||
                    presolver->variable_to_box[column] < 0 ||
                    presolver->is_fixed[column] ||
                    presolver->is_substituted[column] ||
                    presolver->is_parallel_removed[column] ||
                    presolver->affine_protected_columns[column] ||
                    workspace->quadratic[column] ||
                    workspace->factor[column] ||
                    workspace->protected_target[column])
                {
                    eligible = 0;
                    break;
                }
                ++eligible;
            }
            if (eligible == 2)
            {
                *has_transformed_bounded_doubleton = 1;
                free(expanded_values);
                free(expanded_marks);
                free(expanded_columns);
                free(keys);
                return 0;
            }
        }
        if (length <= 0)
            continue;
        hash ^= (uint32_t) length;
        keys[active_count++] =
            (PreFOSParallelSupportKey){
                hash, coefficient_hash, length};
    }
    free(expanded_values);
    free(expanded_marks);
    free(expanded_columns);
    qsort(
        keys, active_count, sizeof(*keys),
        compare_parallel_support_keys);
    for (start = 0; start < active_count;)
    {
        size_t end = start + 1;
        while (end < active_count &&
               keys[end].length == keys[start].length &&
               keys[end].support_hash == keys[start].support_hash &&
               keys[end].coefficient_hash ==
                   keys[start].coefficient_hash)
            ++end;
        if (end - start > 1)
        {
            size_t duplicate_nnz =
                saturated_size_multiply(
                    end - start - 1,
                    (size_t) keys[start].length);
            potential_nnz = saturated_size_add(
                potential_nnz, duplicate_nnz);
        }
        start = end;
    }
    if (potential_nnz == 0 && transformed_rows_only &&
        exact_rows_match_transformed_support(
            presolver, matrix, keys, active_count))
        potential_nnz = 1;
    free(keys);
    return potential_nnz > 0;
}

static int dense_parallel_support_is_promising(
    const PreFOSPresolver *presolver, const PreFOSCsrMatrix *matrix)
{
    return parallel_support_is_promising(
        presolver, matrix, 0, NULL, NULL, 0, NULL);
}

int prefos_internal_filtered_parallel_support_is_promising(
    const PreFOSPresolver *presolver)
{
    if (!presolver) return 0;
    return parallel_support_is_promising(
        presolver, &presolver->original.A, 1, NULL, NULL, 0, NULL);
}

int prefos_internal_materialized_support_opportunities(
    const PreFOSPresolver *presolver,
    const PreFOSColumnWorkspace *workspace)
{
    int has_trivial = 0;
    int has_bounded_doubleton = 0;
    int has_parallel;
    if (!presolver) return 0;
    has_parallel = parallel_support_is_promising(
        presolver, &presolver->original.A, 1,
        &has_trivial, &has_bounded_doubleton,
        presolver->original.A.nnz >=
                PREFOS_SPARSE_MATERIALIZED_SUPPORT_MIN_NNZ &&
            getenv("PREFOS_EXHAUSTIVE_MATERIALIZED_SUPPORT") == NULL,
        workspace);
    return (has_trivial
                ? PREFOS_MATERIALIZED_SUPPORT_TRIVIAL
                : 0) |
           (has_parallel
                ? PREFOS_MATERIALIZED_SUPPORT_PARALLEL
                : 0) |
           (has_bounded_doubleton
                ? PREFOS_MATERIALIZED_SUPPORT_BOUNDED_DOUBLETON
                : 0);
}

static int prefos_row_is_active(const void *context, size_t row)
{
    const PreFOSPresolver *presolver = (const PreFOSPresolver *) context;
    return !presolver->remove_rows[row] &&
           (presolver->working_matrix_is_materialized ||
            prefos_internal_row_has_exact_linear_form(presolver, row));
}

static int has_rows_requiring_materialization(
    const PreFOSPresolver *presolver)
{
    size_t position;
    if (presolver->working_matrix_is_materialized ||
        !presolver->rows_require_materialization)
        return 0;
    if (presolver->materialization_row_log_complete &&
        presolver->n_materialization_row_log ==
            presolver->n_rows_require_materialization)
    {
        for (position = 0;
             position < presolver->n_materialization_row_log;
             ++position)
        {
            int row = presolver->materialization_row_log[position];
            if (!presolver->remove_rows[row] &&
                presolver->rows_require_materialization[row])
                return 1;
        }
        return 0;
    }
    for (position = 0;
         position < presolver->original.A.rows;
         ++position)
        if (!presolver->remove_rows[position] &&
            presolver->rows_require_materialization[position])
            return 1;
    return 0;
}

static void mark_incremental_parallel_row(
    const PreFOSPresolver *presolver, size_t rows, int row,
    int *marks, int *dirty_rows, size_t *dirty_count)
{
    if (row < 0 || (size_t) row >= rows ||
        presolver->remove_rows[row] || marks[row])
        return;
    marks[row] = 1;
    dirty_rows[(*dirty_count)++] = row;
}

static void mark_incremental_parallel_column_rows(
    const PreFOSPresolver *presolver,
    const PreFOSColumnWorkspace *workspace, size_t rows, int column,
    int *marks, int *dirty_rows, size_t *dirty_count)
{
    int position;
    if (column < 0 ||
        (size_t) column >= presolver->original.n)
        return;
    for (position = workspace->starts[column];
         position < workspace->ends[column]; ++position)
        if (workspace->values[position] != 0.0)
            mark_incremental_parallel_row(
                presolver, rows, workspace->rows[position],
                marks, dirty_rows, dirty_count);
}

static size_t collect_incremental_parallel_rows(
    const PreFOSPresolver *presolver,
    const PreFOSColumnWorkspace *workspace, size_t rows,
    int *marks, int *dirty_rows)
{
    const PresolveTransformationLog *log =
        &presolver->transformations;
    size_t position, dirty_count = 0;
    size_t fixed_cursor =
        presolver->materialized_parallel_rows_fixed_column_epoch;
    size_t transformation_cursor =
        presolver->materialized_parallel_rows_column_transformations;

    if (fixed_cursor > presolver->n_fixed_columns ||
        transformation_cursor > log->n_column_transformations)
        return SIZE_MAX;
    if (!workspace && fixed_cursor < presolver->n_fixed_columns)
        return SIZE_MAX;
    memset(marks, 0, rows * sizeof(*marks));
    for (position = fixed_cursor;
         position < presolver->n_fixed_columns; ++position)
        mark_incremental_parallel_column_rows(
            presolver, workspace, rows,
            presolver->fixed_column_log[position],
            marks, dirty_rows, &dirty_count);
    for (position = transformation_cursor;
         position < log->n_column_transformations; ++position)
    {
        const PresolveColumnTransformationRecord *record =
            &log->column_transformations[position];
        size_t entry;
        if (record->type == PRESOLVE_COLUMN_SUBSTITUTED)
        {
            for (entry = 0; entry < record->length; ++entry)
                mark_incremental_parallel_row(
                    presolver, rows, record->indices[entry],
                    marks, dirty_rows, &dirty_count);
        }
        else if (record->type == PRESOLVE_COLUMNS_PARALLEL)
        {
            if (!workspace) return SIZE_MAX;
            mark_incremental_parallel_column_rows(
                presolver, workspace, rows, record->column,
                marks, dirty_rows, &dirty_count);
        }
    }
    return dirty_count;
}

static int compare_parallel_hash_key(
    int left_support, int left_coefficient,
    int right_support, int right_coefficient)
{
    uint32_t left_support_key = (uint32_t) left_support;
    uint32_t right_support_key = (uint32_t) right_support;
    uint32_t left_coefficient_key;
    uint32_t right_coefficient_key;
    if (left_support_key != right_support_key)
        return left_support_key < right_support_key ? -1 : 1;
    left_coefficient_key = (uint32_t) left_coefficient;
    right_coefficient_key = (uint32_t) right_coefficient;
    if (left_coefficient_key == right_coefficient_key) return 0;
    return left_coefficient_key < right_coefficient_key ? -1 : 1;
}

static size_t lower_bound_parallel_hash_snapshot(
    const int *sorted_rows, size_t count,
    const int *support_hashes, const int *coefficient_hashes,
    int support_hash, int coefficient_hash)
{
    size_t lower = 0, upper = count;
    while (lower < upper)
    {
        size_t middle = lower + (upper - lower) / 2;
        int row = sorted_rows[middle];
        if (compare_parallel_hash_key(
                support_hashes[row], coefficient_hashes[row],
                support_hash, coefficient_hash) < 0)
            lower = middle + 1;
        else
            upper = middle;
    }
    return lower;
}

static int find_incremental_materialized_parallel_rows(
    PreFOSPresolver *presolver, const PresolveSparseRowView *view,
    const PreFOSColumnWorkspace *workspace, int *parallel_rows,
    int *support_hashes, int *coefficient_hashes,
    int *sorted_snapshot, int *row_marks, int *group_starts,
    size_t *n_groups, size_t *dirty_count_out,
    size_t *candidate_count_out)
{
    size_t rows = view->n_rows;
    size_t dirty_count, candidate_count, position;
    size_t sorted_count =
        presolver->materialized_parallel_rows_sorted_count;
    int *new_support = NULL;
    int *new_coefficient = NULL;
    int *candidate_scratch = NULL;
    int *candidate_groups = NULL;
    int detected;

    *dirty_count_out = 0;
    *candidate_count_out = 0;
    if (!presolver->materialized_parallel_rows_signature_valid ||
        (rows < PREFOS_INCREMENTAL_PARALLEL_ROW_MIN_ROWS &&
         !getenv("PREFOS_FORCE_INCREMENTAL_PARALLEL_ROWS")) ||
        sorted_count == 0 || sorted_count > rows)
        return 0;
    dirty_count = collect_incremental_parallel_rows(
        presolver, workspace, rows, row_marks, parallel_rows);
    if (dirty_count == SIZE_MAX ||
        dirty_count > PREFOS_INCREMENTAL_PARALLEL_ROW_MAX_DIRTY ||
        dirty_count > rows / 4)
        return 0;
    *dirty_count_out = dirty_count;
    if (dirty_count == 0)
    {
        *n_groups = 0;
        return 1;
    }
    new_support = (int *)
        prefos_internal_alloc_array(dirty_count, sizeof(int));
    new_coefficient = (int *)
        prefos_internal_alloc_array(dirty_count, sizeof(int));
    if (!new_support || !new_coefficient)
        goto fallback;
    if (!presolve_compute_parallel_row_hash_keys(
            view, parallel_rows, dirty_count,
            new_support, new_coefficient))
        goto fallback;

    candidate_count = dirty_count;
    for (position = 0; position < dirty_count; ++position)
    {
        size_t match;
        int support = new_support[position];
        int coefficient = new_coefficient[position];
        if (support == INT_MAX && coefficient == INT_MAX)
            continue;
        match = lower_bound_parallel_hash_snapshot(
            sorted_snapshot, sorted_count,
            support_hashes, coefficient_hashes,
            support, coefficient);
        while (match < sorted_count)
        {
            int row = sorted_snapshot[match++];
            if (compare_parallel_hash_key(
                    support_hashes[row], coefficient_hashes[row],
                    support, coefficient) != 0)
                break;
            if (!presolver->remove_rows[row] && !row_marks[row])
            {
                row_marks[row] = 1;
                parallel_rows[candidate_count++] = row;
            }
        }
    }
    *candidate_count_out = candidate_count;
    if (candidate_count > rows / 2 ||
        candidate_count > (SIZE_MAX - 1U) / 2U)
        goto fallback;
    candidate_scratch = (int *) prefos_internal_alloc_array(
        2 * candidate_count + 1U, sizeof(int));
    if (!candidate_scratch) goto fallback;
    candidate_groups = candidate_scratch + candidate_count;
    detected = presolve_find_parallel_rows_in_set(
        view, presolver->settings.feasibility_tolerance, 0, NULL,
        parallel_rows, candidate_count, support_hashes,
        coefficient_hashes, candidate_scratch,
        candidate_groups, candidate_count + 1U, n_groups);
    if (!detected) goto fallback;
    memcpy(
        group_starts, candidate_groups,
        (*n_groups + 1U) * sizeof(*group_starts));
    free(new_support);
    free(new_coefficient);
    free(candidate_scratch);
    return 1;

fallback:
    free(new_support);
    free(new_coefficient);
    free(candidate_scratch);
    return 0;
}

static size_t rebuild_materialized_parallel_row_snapshot(
    const PreFOSPresolver *presolver, size_t rows,
    int *parallel_rows, const int *support_hashes,
    const int *coefficient_hashes, int *sort_auxiliary,
    int *sorted_snapshot)
{
    size_t row, active_count = 0;
    for (row = 0; row < rows; ++row)
        if (!presolver->remove_rows[row] &&
            (support_hashes[row] != INT_MAX ||
             coefficient_hashes[row] != INT_MAX))
            parallel_rows[active_count++] = (int) row;
    presolve_sort_rows_by_hash(
        parallel_rows, active_count, support_hashes,
        coefficient_hashes, sort_auxiliary);
    if (active_count > 0)
        memcpy(
            sorted_snapshot, parallel_rows,
            active_count * sizeof(*sorted_snapshot));
    return active_count;
}

static int prefos_parallel_values_close(const void *context, double left,
                                     double right, double tolerance)
{
    (void) context;
    return prefos_internal_values_close(left, right, tolerance);
}

static PreFOSStatus append_row_record(
    PreFOSPresolver *presolver, const PreFOSCsrMatrix *matrix,
    PresolveRowTransformationType type, int row, int source_row,
    double ratio, double new_side)
{
    int start = matrix->row_pointers[row];
    PresolveRowTransformationRecord record = {
        .type = type,
        .row = row,
        .source_row = source_row,
        .ratio = ratio,
        .new_side = new_side,
        .indices = matrix->column_indices + start,
        .coefficients = matrix->values + start,
        .length = (size_t) (matrix->row_pointers[row + 1] - start)};
    return presolve_transformation_log_append_row_transformation(
               &presolver->transformations, &record, NULL)
               ? PREFOS_STATUS_OK
               : PREFOS_STATUS_OUT_OF_MEMORY;
}

static PreFOSStatus append_deleted_row_record(PreFOSPresolver *presolver, int row)
{
    PresolveRowTransformationRecord record = {
        .type = PRESOLVE_ROW_DELETED,
        .row = row,
        .source_row = -1,
        .dual_value = 0.0};
    return presolve_transformation_log_append_row_transformation(
               &presolver->transformations, &record, NULL)
               ? PREFOS_STATUS_OK
               : PREFOS_STATUS_OUT_OF_MEMORY;
}

static PreFOSStatus process_parallel_group(PreFOSPresolver *presolver,
                                        const PreFOSCsrMatrix *matrix,
                                        double *lower, double *upper,
                                        const int *rows, size_t count,
                                        PreFOSColumnWorkspace *column_workspace)
{
    double tolerance = presolver->settings.feasibility_tolerance;
    PresolveSparseRowView view = {
        matrix->rows, matrix->values, matrix->column_indices,
        matrix->row_pointers, matrix->row_pointers + 1, 1};
    PresolveParallelRowReduction reduction;
    PresolveParallelReductionStatus reduction_status;
    size_t position;

    reduction_status = presolve_analyze_parallel_row_group(
        &view, rows, count, lower, upper, tolerance,
        prefos_parallel_values_close, NULL, &reduction);
    if (reduction_status == PRESOLVE_PARALLEL_REDUCTION_INFEASIBLE)
        return PREFOS_STATUS_PRIMAL_INFEASIBLE;
    if (reduction_status == PRESOLVE_PARALLEL_REDUCTION_UNCERTAIN)
        return PREFOS_STATUS_OK;
    if (reduction_status != PRESOLVE_PARALLEL_REDUCTION_OK)
        return PREFOS_STATUS_NUMERICAL_ERROR;
    prefos_internal_note_constraint_bound_change(presolver);

    if (reduction.lower_source_row >= 0)
    {
        PreFOSStatus status = append_row_record(
            presolver, matrix, PRESOLVE_ROW_LOWER_CHANGED,
            reduction.kept_row, reduction.lower_source_row,
            reduction.lower_source_ratio, reduction.lower);
        if (status != PREFOS_STATUS_OK) return status;
    }
    if (reduction.upper_source_row >= 0)
    {
        PreFOSStatus status = append_row_record(
            presolver, matrix, PRESOLVE_ROW_UPPER_CHANGED,
            reduction.kept_row, reduction.upper_source_row,
            reduction.upper_source_ratio, reduction.upper);
        if (status != PREFOS_STATUS_OK) return status;
    }
    if (lower != presolver->working_constraint_lower)
    {
        int kept = reduction.kept_row;
        double shift = 0.0;
        if (isfinite(lower[kept]) &&
            isfinite(presolver->working_constraint_lower[kept]))
            shift =
                presolver->working_constraint_lower[kept] - lower[kept];
        else if (isfinite(upper[kept]) &&
                 isfinite(presolver->working_constraint_upper[kept]))
            shift =
                presolver->working_constraint_upper[kept] - upper[kept];
        presolver->working_constraint_lower[kept] =
            isfinite(reduction.lower) ? reduction.lower + shift
                                      : reduction.lower;
        presolver->working_constraint_upper[kept] =
            isfinite(reduction.upper) ? reduction.upper + shift
                                      : reduction.upper;
        if (isnan(presolver->working_constraint_lower[kept]) ||
            isnan(presolver->working_constraint_upper[kept]))
            return PREFOS_STATUS_NUMERICAL_ERROR;
    }
    lower[reduction.kept_row] = reduction.lower;
    upper[reduction.kept_row] = reduction.upper;
    prefos_internal_queue_row_side_change(
        presolver, column_workspace, (size_t) reduction.kept_row);

    for (position = 0; position < count; ++position)
    {
        int row = rows[position];
        PreFOSStatus status;
        if (row == reduction.kept_row) continue;
        status = append_deleted_row_record(presolver, row);
        if (status != PREFOS_STATUS_OK) return status;
        prefos_internal_mark_removed_row(presolver, (size_t) row);
        ++presolver->stats.removed_redundant_rows;
    }
    return PREFOS_STATUS_OK;
}

static PreFOSStatus remove_parallel_rows_in_matrix(
    PreFOSPresolver *presolver, const PreFOSCsrMatrix *matrix,
    double *lower, double *upper, int allow_gpu,
    PreFOSColumnWorkspace *column_workspace)
{
    PresolveSparseRowView view;
    int *workspace, *parallel_rows, *support_hashes, *coefficient_hashes;
    int *sort_auxiliary, *group_starts;
    size_t n_groups, group;
    size_t removed_before = presolver->n_removed_rows;
    size_t incremental_dirty = 0;
    size_t incremental_candidates = 0;
    PreFOSTimestamp start, detection_stop, stop;
    PreFOSStatus status = PREFOS_STATUS_OK;
    int detected = 0;
    int incremental_scan = 0;
    int over_cpu_budget;
    int use_gpu =
        allow_gpu && presolver->settings.structural_reductions_gpu;
    long double average_nnz;

    if (!presolver->settings.remove_redundant_rows || matrix->rows < 2)
        return PREFOS_STATUS_OK;
    if (!getenv("PREFOS_FORCE_PARALLEL_ROW_RESCAN") &&
        !presolver->working_matrix_is_materialized &&
        presolver->source_parallel_rows_closed)
    {
        if (trace_parallel_rows())
            fprintf(
                stderr,
                "PreFOS parallel rows skip rows=%zu nnz=%zu "
                "materialized=0 reason=closed\n",
                matrix->rows, matrix->nnz);
        return PREFOS_STATUS_OK;
    }
    if (!getenv("PREFOS_FORCE_PARALLEL_ROW_RESCAN") &&
        presolver->working_matrix_is_materialized &&
        presolver->materialized_parallel_rows_signature_valid &&
        presolver->materialized_parallel_rows_fixed_column_epoch ==
            presolver->fixed_column_epoch &&
        presolver->materialized_parallel_rows_column_transformations ==
            presolver->transformations.n_column_transformations)
    {
        if (trace_parallel_rows())
            fprintf(
                stderr,
                "PreFOS parallel rows skip rows=%zu nnz=%zu "
                "materialized=1 reason=closed\n",
                matrix->rows, matrix->nnz);
        return PREFOS_STATUS_OK;
    }
    if (use_gpu && has_rows_requiring_materialization(presolver))
        use_gpu = 0;
    average_nnz =
        (long double) matrix->nnz / (long double) matrix->rows;
    over_cpu_budget =
        presolver->settings.parallel_row_max_average_nnz > 0.0 &&
        average_nnz >
            (long double)
                presolver->settings.parallel_row_max_average_nnz;
    if (over_cpu_budget && !use_gpu)
    {
        ++presolver->stats.parallel_row_budget_skips;
        if (trace_parallel_rows())
            fprintf(
                stderr,
                "PreFOS parallel rows skip rows=%zu nnz=%zu "
                "materialized=%d reason=budget\n",
                matrix->rows, matrix->nnz,
                presolver->working_matrix_is_materialized);
        return PREFOS_STATUS_OK;
    }
    if (!use_gpu &&
        presolver->settings.parallel_row_max_average_nnz >
            PREFOS_PARALLEL_ROW_PREFILTER_AVERAGE_NNZ &&
        average_nnz >
            (long double) PREFOS_PARALLEL_ROW_PREFILTER_AVERAGE_NNZ &&
        !dense_parallel_support_is_promising(presolver, matrix))
    {
        ++presolver->stats.parallel_row_budget_skips;
        if (trace_parallel_rows())
            fprintf(
                stderr,
                "PreFOS parallel rows skip rows=%zu nnz=%zu "
                "materialized=%d reason=prefilter\n",
                matrix->rows, matrix->nnz,
                presolver->working_matrix_is_materialized);
        return PREFOS_STATUS_OK;
    }
    prefos_internal_timer_now(&start);
    if (matrix->rows > SIZE_MAX / 5 ||
        5 * matrix->rows > SIZE_MAX / sizeof(int))
    {
        status = PREFOS_STATUS_OUT_OF_MEMORY;
        goto finish;
    }
    if (presolver->parallel_row_workspace_capacity <
        5 * matrix->rows)
    {
        int *resized = (int *) realloc(
            presolver->parallel_row_workspace,
            5 * matrix->rows * sizeof(int));
        if (!resized)
        {
            status = PREFOS_STATUS_OUT_OF_MEMORY;
            goto finish;
        }
        presolver->parallel_row_workspace = resized;
        presolver->parallel_row_workspace_capacity =
            5 * matrix->rows;
        presolver->materialized_parallel_rows_signature_valid = 0;
        presolver->materialized_parallel_rows_sorted_count = 0;
        presolver->materialized_parallel_rows_snapshot_sorted = 0;
    }
    workspace = presolver->parallel_row_workspace;
    parallel_rows = workspace;
    support_hashes = workspace + matrix->rows;
    coefficient_hashes = workspace + 2 * matrix->rows;
    sort_auxiliary = workspace + 3 * matrix->rows;
    group_starts = workspace + 4 * matrix->rows;
    view = (PresolveSparseRowView){
        matrix->rows, matrix->values, matrix->column_indices,
        matrix->row_pointers, matrix->row_pointers + 1, 1};

    if (use_gpu)
    {
        PreFOSCudaPropagationStatus cuda_status;
        PreFOSCudaWorkspace *cuda_workspace =
            prefos_internal_cuda_workspace_get(presolver, &cuda_status);
        size_t active_rows = 0;
        double gpu_milliseconds = 0.0;
        if (cuda_workspace && cuda_status == PREFOS_CUDA_PROPAGATION_OK)
        {
            cuda_status = prefos_cuda_parallel_row_hash_sort(
                cuda_workspace, presolver->remove_rows, parallel_rows,
                support_hashes, coefficient_hashes, &active_rows,
                &gpu_milliseconds);
        }
        presolver->stats.parallel_row_gpu_milliseconds += gpu_milliseconds;
        if (cuda_status == PREFOS_CUDA_PROPAGATION_OK)
        {
            if (presolver->working_matrix_is_materialized &&
                active_rows > 0)
                memcpy(
                    sort_auxiliary, parallel_rows,
                    active_rows * sizeof(*sort_auxiliary));
            if (presolver->working_matrix_is_materialized)
            {
                presolver->materialized_parallel_rows_sorted_count =
                    active_rows;
                presolver->materialized_parallel_rows_snapshot_sorted = 1;
            }
            detected = presolve_collect_parallel_row_groups(
                &view, presolver->settings.feasibility_tolerance,
                parallel_rows, active_rows, support_hashes,
                coefficient_hashes, group_starts, matrix->rows, &n_groups);
            if (detected)
                ++presolver->stats.parallel_row_gpu_passes;
        }
        if (!detected)
        {
            ++presolver->stats.parallel_row_gpu_fallbacks;
        }
    }
    if (!detected)
    {
        size_t cpu_active_rows = 0, row;
        int support_only =
            presolver->working_matrix_is_materialized &&
            matrix->rows <= 32768 &&
            average_nnz <= 64.0L &&
            !getenv("PREFOS_FORCE_INCREMENTAL_PARALLEL_ROWS");
        if (over_cpu_budget)
        {
            ++presolver->stats.parallel_row_budget_skips;
            goto cleanup;
        }
        if (!support_only &&
            !getenv("PREFOS_FORCE_PARALLEL_ROW_RESCAN") &&
            presolver->working_matrix_is_materialized)
        {
            if (presolver->materialized_parallel_rows_signature_valid &&
                !presolver->materialized_parallel_rows_snapshot_sorted)
            {
                presolver->materialized_parallel_rows_sorted_count =
                    rebuild_materialized_parallel_row_snapshot(
                        presolver, matrix->rows, parallel_rows,
                        support_hashes, coefficient_hashes,
                        group_starts, sort_auxiliary);
                presolver->materialized_parallel_rows_snapshot_sorted = 1;
            }
            incremental_scan =
                find_incremental_materialized_parallel_rows(
                    presolver, &view, column_workspace,
                    parallel_rows, support_hashes,
                    coefficient_hashes, sort_auxiliary,
                    group_starts, group_starts, &n_groups,
                    &incremental_dirty,
                    &incremental_candidates);
            detected = incremental_scan;
        }
        if (!detected)
        {
            for (row = 0; row < matrix->rows; ++row)
                if (prefos_row_is_active(presolver, row))
                    parallel_rows[cpu_active_rows++] = (int) row;
            if (presolver->working_matrix_is_materialized &&
                !support_only)
            {
                size_t sorted_count = 0;
                detected =
                    presolve_find_parallel_rows_in_set_with_sorted_copy(
                        &view,
                        presolver->settings.feasibility_tolerance,
                        0, NULL, parallel_rows, cpu_active_rows,
                        support_hashes, coefficient_hashes,
                        sort_auxiliary, sort_auxiliary,
                        &sorted_count, group_starts,
                        matrix->rows, &n_groups);
                if (detected)
                {
                    presolver
                        ->materialized_parallel_rows_sorted_count =
                        sorted_count;
                    presolver
                        ->materialized_parallel_rows_snapshot_sorted = 1;
                }
            }
            else
            {
                detected = presolve_find_parallel_rows_in_set(
                    &view,
                    presolver->settings.feasibility_tolerance,
                    support_only, NULL, parallel_rows,
                    cpu_active_rows, support_hashes,
                    coefficient_hashes, sort_auxiliary,
                    group_starts, matrix->rows, &n_groups);
                if (presolver->working_matrix_is_materialized)
                {
                    presolver
                        ->materialized_parallel_rows_sorted_count = 0;
                    presolver
                        ->materialized_parallel_rows_snapshot_sorted = 0;
                }
            }
        }
    }
    if (trace_parallel_rows())
        prefos_internal_timer_now(&detection_stop);
    if (!detected)
    {
        status = PREFOS_STATUS_NUMERICAL_ERROR;
        goto cleanup;
    }
    for (group = 0; group < n_groups; ++group)
    {
        size_t start = (size_t) group_starts[group];
        size_t end = (size_t) group_starts[group + 1];
        if (getenv("PREFOS_TRACE_PARALLEL_ROW_MEMBERS"))
        {
            size_t position;
            fprintf(stderr, "PreFOS parallel-row group=%zu rows=", group);
            for (position = start; position < end; ++position)
                fprintf(
                    stderr, "%s%d", position == start ? "" : ",",
                    parallel_rows[position]);
            fputc('\n', stderr);
        }
        status = process_parallel_group(
            presolver, matrix, lower, upper, parallel_rows + start,
            end - start, column_workspace);
        if (status != PREFOS_STATUS_OK) goto cleanup;
    }
    if (incremental_scan && incremental_dirty > 0)
        presolver->materialized_parallel_rows_snapshot_sorted = 0;
    if (presolver->working_matrix_is_materialized)
    {
        presolver->materialized_parallel_rows_signature_valid = 1;
        presolver->materialized_parallel_rows_fixed_column_epoch =
            presolver->fixed_column_epoch;
        presolver->materialized_parallel_rows_column_transformations =
            presolver->transformations.n_column_transformations;
    }
    else
    {
        presolver->source_parallel_rows_closed = 1;
        presolver->materialized_parallel_rows_signature_valid = 0;
        presolver->materialized_parallel_rows_sorted_count = 0;
        presolver->materialized_parallel_rows_snapshot_sorted = 0;
    }

cleanup:
    if (trace_parallel_rows())
    {
        prefos_internal_timer_now(&stop);
        fprintf(
            stderr,
            "PreFOS parallel rows rows=%zu nnz=%zu materialized=%d "
            "groups=%zu removed=%zu incremental=%d dirty=%zu "
            "candidates=%zu detect_ms=%.3f reduce_ms=%.3f\n",
            matrix->rows, matrix->nnz,
            presolver->working_matrix_is_materialized,
            detected ? n_groups : 0,
            presolver->n_removed_rows - removed_before,
            incremental_scan, incremental_dirty,
            incremental_candidates,
            prefos_internal_timer_elapsed_milliseconds(
                &start, &detection_stop),
            prefos_internal_timer_elapsed_milliseconds(
                &detection_stop, &stop));
    }
finish:
    prefos_internal_timer_now(&stop);
    presolver->stats.parallel_row_detection_milliseconds +=
        prefos_internal_timer_elapsed_milliseconds(&start, &stop);
    return status;
}

PreFOSStatus prefos_internal_remove_parallel_rows(
    PreFOSPresolver *presolver,
    PreFOSColumnWorkspace *column_workspace)
{
    return remove_parallel_rows_in_matrix(
        presolver, &presolver->original.A,
        presolver->working_constraint_lower,
        presolver->working_constraint_upper, 1, column_workspace);
}

PreFOSStatus prefos_internal_remove_parallel_rows_in_working_matrix(
    PreFOSPresolver *presolver, const PreFOSCsrMatrix *matrix,
    double *lower, double *upper,
    PreFOSColumnWorkspace *column_workspace)
{
    PreFOSStatus status;
    int was_materialized;
    if (!matrix || !lower || !upper)
        return PREFOS_STATUS_INVALID_ARGUMENT;
    was_materialized = presolver->working_matrix_is_materialized;
    presolver->working_matrix_is_materialized = 1;
    status = remove_parallel_rows_in_matrix(
        presolver, matrix, lower, upper, 0, column_workspace);
    presolver->working_matrix_is_materialized = was_materialized;
    return status;
}
