/*
 * Copyright 2026 Hongpei Li
 * SPDX-License-Identifier: Apache-2.0
 */

#include "PREFOS_ColumnReductionInternal.h"
#include "PREFOS_CudaBackend.h"
#include "PREFOS_CudaLinearPropagation.h"
#include "PREFOS_TransformedColumnView.h"
#include "core/PREFOS_Timer.h"
#include "ParallelRowDetection.h"

#include <limits.h>
#include <stdint.h>
#include <stdio.h>

#define PREFOS_PARALLEL_HASH_INV_PRECISION 1e6
#define PREFOS_INCREMENTAL_PARALLEL_COLUMN_DIVISOR 2U
#define PREFOS_INCREMENTAL_PARALLEL_COLLISION_CHECKS 4096U

typedef struct
{
    const PreFOSPresolver *presolver;
    const PreFOSColumnWorkspace *workspace;
    const unsigned char *eligible;
    int exact_support;
} PreFOSParallelColumnContext;

static int exact_parallel_column_is_linear_box(
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
           !workspace->factor[column];
}

static int parallel_column_is_active(const void *context, size_t column)
{
    const PreFOSParallelColumnContext *parallel_context =
        (const PreFOSParallelColumnContext *) context;
    const PreFOSPresolver *presolver = parallel_context->presolver;
    const PreFOSColumnWorkspace *workspace = parallel_context->workspace;
    int p;
    if (parallel_context->exact_support)
        return (!parallel_context->eligible ||
                parallel_context->eligible[column]) &&
               exact_parallel_column_is_linear_box(
                   presolver, workspace, column) &&
               !workspace->protected_target[column] &&
               workspace->starts[column] <
                   workspace->ends[column];
    if (!prefos_internal_column_is_linear_box(
            presolver, workspace, (int) column) ||
        workspace->protected_target[column])
        return 0;
    if (workspace->column_dirty_row_counts)
        return workspace->column_dirty_row_counts[column] == 0 &&
               workspace->starts[column] < workspace->ends[column];
    for (p = workspace->starts[column]; p < workspace->ends[column]; ++p)
        if (workspace->dirty_row[workspace->rows[p]]) return 0;
    return workspace->starts[column] < workspace->ends[column];
}

static int parallel_column_remains_active(
    const PreFOSParallelColumnContext *context, int column)
{
    const PreFOSPresolver *presolver = context->presolver;
    const PreFOSColumnWorkspace *workspace = context->workspace;
    return column >= 0 &&
           (size_t) column < presolver->original.n &&
           presolver->variable_to_box[column] >= 0 &&
           !presolver->is_fixed[column] &&
           !presolver->is_substituted[column] &&
           !presolver->is_parallel_removed[column] &&
           !workspace->protected_target[column];
}

static void trace_parallel_column_eligibility(
    const PreFOSParallelColumnContext *context)
{
    const PreFOSPresolver *presolver = context->presolver;
    const PreFOSColumnWorkspace *workspace = context->workspace;
    size_t eligible = 0, no_box = 0, fixed = 0, substituted = 0;
    size_t parallel_removed = 0, affine = 0, nonlinear = 0;
    size_t protected_target = 0, dirty = 0, empty = 0;
    size_t column;

    if (!getenv("PREFOS_TRACE_PARALLEL_COLUMNS")) return;
    for (column = 0; column < presolver->original.n; ++column)
    {
        if (presolver->variable_to_box[column] < 0)
        {
            if (presolver->is_fixed[column])
                ++fixed;
            else if (presolver->is_substituted[column])
                ++substituted;
            else if (presolver->is_parallel_removed[column])
                ++parallel_removed;
            else
                ++no_box;
        }
        else if (presolver->affine_protected_columns[column])
            ++affine;
        else if (workspace->quadratic[column] ||
                 workspace->factor[column])
            ++nonlinear;
        else if (workspace->protected_target[column])
            ++protected_target;
        else if (context->exact_support &&
                 context->eligible &&
                 !context->eligible[column])
            ++dirty;
        else if (!context->exact_support &&
                 workspace->column_dirty_row_counts &&
                 workspace->column_dirty_row_counts[column] != 0)
            ++dirty;
        else if (workspace->starts[column] >= workspace->ends[column])
            ++empty;
        else
            ++eligible;
    }
    fprintf(
        stderr,
        "PreFOS parallel-column eligibility active=%zu no_box=%zu "
        "fixed=%zu substituted=%zu parallel=%zu affine=%zu nonlinear=%zu "
        "protected=%zu dirty=%zu empty=%zu exact=%d\n",
        eligible, no_box, fixed, substituted, parallel_removed, affine,
        nonlinear, protected_target, dirty, empty, context->exact_support);
}

static double normalized_objective_key(
    const PreFOSColumnWorkspace *workspace, int column)
{
    return workspace->objective[column] /
           workspace->values[workspace->starts[column]];
}

static int normalized_objective_column_less(
    const PreFOSColumnWorkspace *workspace, int left, int right)
{
    double left_key =
        normalized_objective_key(workspace, left);
    double right_key =
        normalized_objective_key(workspace, right);
    return left_key < right_key ||
           (left_key == right_key && left < right);
}

static void sift_normalized_objective_columns(
    int *columns, size_t root, size_t count,
    const PreFOSColumnWorkspace *workspace)
{
    for (;;)
    {
        size_t child = root * 2 + 1;
        size_t selected = root;
        int temporary;
        if (child >= count) return;
        if (normalized_objective_column_less(
                workspace, columns[selected], columns[child]))
            selected = child;
        if (child + 1 < count &&
            normalized_objective_column_less(
                workspace, columns[selected], columns[child + 1]))
            selected = child + 1;
        if (selected == root) return;
        temporary = columns[root];
        columns[root] = columns[selected];
        columns[selected] = temporary;
        root = selected;
    }
}

static void sort_parallel_group_by_normalized_objective(
    int *columns, size_t count,
    const PreFOSColumnWorkspace *workspace)
{
    size_t position;
    if (count < 2) return;
    for (position = count / 2; position > 0; --position)
        sift_normalized_objective_columns(
            columns, position - 1, count, workspace);
    for (position = count; position > 1; --position)
    {
        int temporary = columns[0];
        columns[0] = columns[position - 1];
        columns[position - 1] = temporary;
        sift_normalized_objective_columns(
            columns, 0, position - 1, workspace);
    }
}

static void compute_parallel_column_hash(
    const PreFOSParallelColumnContext *context, int column,
    int *support_hash, int *coefficient_hash)
{
    const PreFOSColumnWorkspace *workspace = context->workspace;
    int begin = workspace->starts[column];
    int end = workspace->ends[column];
    uint32_t support = 5381U;
    uint32_t coefficients = 5381U;
    double maximum, scale;
    int position;

    if (!parallel_column_is_active(context, (size_t) column) ||
        begin >= end || workspace->values[begin] == 0.0)
    {
        *support_hash = INT_MAX;
        *coefficient_hash = INT_MAX;
        return;
    }
    maximum = fabs(workspace->values[begin]);
    for (position = begin; position < end; ++position)
    {
        support =
            ((support << 5U) + support) +
            (uint32_t) workspace->rows[position];
        maximum = fmax(maximum, fabs(workspace->values[position]));
    }
    if (maximum == 0.0)
    {
        *support_hash = INT_MAX;
        *coefficient_hash = INT_MAX;
        return;
    }
    scale =
        workspace->values[begin] > 0.0
            ? 1.0 / maximum
            : -1.0 / maximum;
    for (position = begin; position < end; ++position)
    {
        uint32_t normalized = (uint32_t) round(
            workspace->values[position] * scale *
            PREFOS_PARALLEL_HASH_INV_PRECISION);
        coefficients =
            ((coefficients << 5U) + coefficients) + normalized;
    }
    *support_hash = (int) support;
    *coefficient_hash = (int) coefficients;
}

static void clear_parallel_column_dirty_cache(
    PreFOSColumnWorkspace *workspace)
{
    size_t position;
    for (position = 0;
         position < workspace->n_parallel_dirty_columns; ++position)
        workspace->parallel_column_dirty[
            workspace->parallel_dirty_columns[position]] = 0;
    workspace->n_parallel_dirty_columns = 0;
}

static int cache_closed_parallel_column_groups(
    const PreFOSParallelColumnContext *context,
    PreFOSColumnWorkspace *workspace, const int *sorted_active_columns,
    size_t active_count, const int *group_columns,
    const int *group_starts, size_t n_groups)
{
    size_t group, position, output = 0;

    if (!sorted_active_columns || !group_columns || !group_starts)
        return 0;
    if (n_groups == 0)
    {
        if (active_count > 0 &&
            sorted_active_columns != workspace->parallel_columns)
            memcpy(
                workspace->parallel_columns, sorted_active_columns,
                active_count * sizeof(int));
        workspace->parallel_cached_active_columns = active_count;
        workspace->parallel_cache_removed_row_cursor =
            context->presolver->n_removed_rows;
        clear_parallel_column_dirty_cache(workspace);
        return 1;
    }
    for (group = 0; group < n_groups; ++group)
    {
        size_t remaining = 0;
        int start = group_starts[group];
        int end = group_starts[group + 1];
        int group_position;
        for (group_position = start;
             group_position < end; ++group_position)
            if (parallel_column_remains_active(
                    context, group_columns[group_position]) &&
                ++remaining > 1)
                return 0;
    }
    for (position = 0; position < active_count; ++position)
    {
        int column = sorted_active_columns[position];
        if (parallel_column_is_active(context, (size_t) column))
            workspace->parallel_columns[output++] = column;
    }
    workspace->parallel_cached_active_columns = output;
    workspace->parallel_cache_removed_row_cursor =
        context->presolver->n_removed_rows;
    clear_parallel_column_dirty_cache(workspace);
    return 1;
}

static int parallel_hash_key_compare(
    const PreFOSColumnWorkspace *workspace, int left, int right)
{
    uint32_t left_support =
        (uint32_t) workspace->parallel_support_hashes[left];
    uint32_t right_support =
        (uint32_t) workspace->parallel_support_hashes[right];
    uint32_t left_coefficients, right_coefficients;
    if (left_support != right_support)
        return left_support < right_support ? -1 : 1;
    left_coefficients =
        (uint32_t) workspace->parallel_coefficient_hashes[left];
    right_coefficients =
        (uint32_t) workspace->parallel_coefficient_hashes[right];
    if (left_coefficients == right_coefficients) return 0;
    return left_coefficients < right_coefficients ? -1 : 1;
}

static int parallel_columns_are_parallel(
    const PreFOSParallelColumnContext *context, int first, int second)
{
    const PreFOSColumnWorkspace *workspace = context->workspace;
    int first_start = workspace->starts[first];
    int second_start = workspace->starts[second];
    int first_length = workspace->ends[first] - first_start;
    int second_length = workspace->ends[second] - second_start;
    double ratio;
    int position;
    if (first_length != second_length || first_length <= 0)
        return 0;
    ratio = workspace->values[first_start] /
            workspace->values[second_start];
    for (position = 0; position < first_length; ++position)
        if (workspace->rows[first_start + position] !=
                workspace->rows[second_start + position] ||
            fabs(
                workspace->values[first_start + position] -
                ratio *
                    workspace->values[second_start + position]) >
                context->presolver->settings.feasibility_tolerance)
            return 0;
    return 1;
}

static int incremental_cache_proves_no_parallel_columns(
    const PreFOSParallelColumnContext *context)
{
    const PreFOSPresolver *presolver = context->presolver;
    PreFOSColumnWorkspace *workspace =
        (PreFOSColumnWorkspace *) context->workspace;
    const PreFOSCsrMatrix *matrix = &presolver->original.A;
    size_t position, dirty_count, dirty_active_count = 0;
    size_t unchanged_count = 0, left = 0, right = 0, output = 0;
    size_t collision_checks = 0;
    int trace = getenv("PREFOS_TRACE_PARALLEL_COLUMNS") != NULL;

    if (!workspace->parallel_no_group_cache_valid)
    {
        if (trace)
            fprintf(
                stderr,
                "PreFOS parallel-column cache miss valid=%d dirty=%zu "
                "removed=%zu\n",
                workspace->parallel_no_group_cache_valid,
                workspace->n_parallel_dirty_columns,
                presolver->n_removed_rows -
                    workspace->parallel_cache_removed_row_cursor);
        return 0;
    }

    for (position = workspace->parallel_cache_removed_row_cursor;
         position < presolver->n_removed_rows; ++position)
    {
        int row = presolver->removed_row_log[position];
        int entry;
        if (row < 0 || (size_t) row >= matrix->rows)
            return 0;
        for (entry = matrix->row_pointers[row];
             entry < matrix->row_pointers[row + 1]; ++entry)
        {
            int column = matrix->column_indices[entry];
            if (matrix->values[entry] == 0.0) continue;
            prefos_internal_mark_parallel_column_dirty(
                workspace, column);
        }
    }
    dirty_count = workspace->n_parallel_dirty_columns;
    if (dirty_count == 0)
    {
        workspace->parallel_cache_removed_row_cursor =
            presolver->n_removed_rows;
        if (trace)
            fprintf(stderr, "PreFOS parallel-column cache hit dirty=0\n");
        return 1;
    }
    if (workspace->parallel_cached_active_columns >
            presolver->original.n ||
        (dirty_count > 4096U &&
         dirty_count >
             presolver->original.n /
                 PREFOS_INCREMENTAL_PARALLEL_COLUMN_DIVISOR))
    {
        if (trace)
            fprintf(
                stderr,
                "PreFOS parallel-column cache miss dirty=%zu active=%zu\n",
                dirty_count,
                workspace->parallel_cached_active_columns);
        return 0;
    }

    for (position = 0;
         position < workspace->parallel_cached_active_columns; ++position)
    {
        int column = workspace->parallel_columns[position];
        if (!workspace->parallel_column_dirty[column])
            workspace->parallel_columns[unchanged_count++] = column;
    }
    for (position = 0; position < dirty_count; ++position)
    {
        int column = workspace->parallel_dirty_columns[position];
        compute_parallel_column_hash(
            context, column,
            &workspace->parallel_support_hashes[column],
            &workspace->parallel_coefficient_hashes[column]);
    }
    for (position = 0; position < dirty_count; ++position)
    {
        int column = workspace->parallel_dirty_columns[position];
        workspace->parallel_column_dirty[column] = 0;
        if (workspace->parallel_support_hashes[column] != INT_MAX)
            workspace->parallel_dirty_columns[dirty_active_count++] =
                column;
    }
    workspace->n_parallel_dirty_columns = 0;

    presolve_sort_rows_by_hash(
        workspace->parallel_dirty_columns, dirty_active_count,
        workspace->parallel_support_hashes,
        workspace->parallel_coefficient_hashes,
        workspace->parallel_sort_auxiliary);
    for (position = 0; position < dirty_active_count;)
    {
        size_t end = position + 1;
        size_t first, second;
        while (end < dirty_active_count &&
               parallel_hash_key_compare(
                   workspace,
                   workspace->parallel_dirty_columns[position],
                   workspace->parallel_dirty_columns[end]) == 0)
            ++end;
        for (first = position; first < end; ++first)
            for (second = first + 1; second < end; ++second)
            {
                if (collision_checks++ >=
                    PREFOS_INCREMENTAL_PARALLEL_COLLISION_CHECKS)
                    return 0;
                if (!parallel_columns_are_parallel(
                        context,
                        workspace->parallel_dirty_columns[first],
                        workspace->parallel_dirty_columns[second]))
                    continue;
                if (trace)
                    fprintf(
                        stderr,
                        "PreFOS parallel-column cache miss dirty pair "
                        "(%d,%d)\n",
                        workspace->parallel_dirty_columns[first],
                        workspace->parallel_dirty_columns[second]);
                return 0;
            }
        position = end;
    }

    while (left < unchanged_count && right < dirty_active_count)
    {
        int unchanged = workspace->parallel_columns[left];
        int dirty = workspace->parallel_dirty_columns[right];
        int comparison =
            parallel_hash_key_compare(workspace, unchanged, dirty);
        if (comparison == 0)
        {
            size_t left_end = left + 1;
            size_t right_end = right + 1;
            size_t left_position, right_position;
            while (left_end < unchanged_count &&
                   parallel_hash_key_compare(
                       workspace, unchanged,
                       workspace->parallel_columns[left_end]) == 0)
                ++left_end;
            while (right_end < dirty_active_count &&
                   parallel_hash_key_compare(
                       workspace, dirty,
                       workspace->parallel_dirty_columns[right_end]) == 0)
                ++right_end;
            for (right_position = right;
                 right_position < right_end; ++right_position)
                for (left_position = left;
                     left_position < left_end; ++left_position)
                {
                    if (collision_checks++ >=
                        PREFOS_INCREMENTAL_PARALLEL_COLLISION_CHECKS)
                        return 0;
                    if (!parallel_columns_are_parallel(
                            context,
                            workspace->parallel_columns[left_position],
                            workspace->parallel_dirty_columns[
                                right_position]))
                        continue;
                    if (trace)
                        fprintf(
                            stderr,
                            "PreFOS parallel-column cache miss pair "
                            "(%d,%d)\n",
                            workspace->parallel_columns[left_position],
                            workspace->parallel_dirty_columns[
                                right_position]);
                    return 0;
                }
            while (left < left_end)
                workspace->parallel_sort_auxiliary[output++] =
                    workspace->parallel_columns[left++];
            while (right < right_end)
                workspace->parallel_sort_auxiliary[output++] =
                    workspace->parallel_dirty_columns[right++];
            continue;
        }
        workspace->parallel_sort_auxiliary[output++] =
            comparison < 0
                ? workspace->parallel_columns[left++]
                : workspace->parallel_dirty_columns[right++];
    }
    while (left < unchanged_count)
        workspace->parallel_sort_auxiliary[output++] =
            workspace->parallel_columns[left++];
    while (right < dirty_active_count)
        workspace->parallel_sort_auxiliary[output++] =
            workspace->parallel_dirty_columns[right++];
    if (output > 0)
        memcpy(
            workspace->parallel_columns,
            workspace->parallel_sort_auxiliary,
            output * sizeof(int));
    workspace->parallel_cached_active_columns = output;
    workspace->parallel_cache_removed_row_cursor =
        presolver->n_removed_rows;
    if (trace)
        fprintf(
            stderr,
            "PreFOS parallel-column cache hit dirty=%zu "
            "dirty_active=%zu active=%zu\n",
            dirty_count, dirty_active_count, output);
    return 1;
}

static double merged_lower(double target_lower, double source_lower,
                           double source_upper, double ratio)
{
    double source = ratio > 0.0 ? source_lower : source_upper;
    if (!isfinite(target_lower) || !isfinite(source)) return -INFINITY;
    return target_lower + ratio * source;
}

static double merged_upper(double target_upper, double source_lower,
                           double source_upper, double ratio)
{
    double source = ratio > 0.0 ? source_upper : source_lower;
    if (!isfinite(target_upper) || !isfinite(source)) return INFINITY;
    return target_upper + ratio * source;
}

static PreFOSStatus append_parallel_record(PreFOSPresolver *presolver,
                                           int source, int target, double ratio,
                                           double source_lower,
                                           double source_upper,
                                           double target_lower,
                                           double target_upper)
{
    PresolveColumnTransformationRecord record;
    memset(&record, 0, sizeof(record));
    record.type = PRESOLVE_COLUMNS_PARALLEL;
    record.column = source;
    record.secondary_column = target;
    record.ratio = ratio;
    record.lower = source_lower;
    record.upper = source_upper;
    record.secondary_lower = target_lower;
    record.secondary_upper = target_upper;
    return presolve_transformation_log_append_column_transformation(
               &presolver->transformations, &record, NULL)
               ? PREFOS_STATUS_OK
               : PREFOS_STATUS_OUT_OF_MEMORY;
}

static PreFOSStatus ensure_parallel_column_scratch(
    const PreFOSPresolver *presolver, PreFOSColumnWorkspace *workspace)
{
    size_t columns = presolver->original.n;
    if (!workspace->parallel_columns)
        workspace->parallel_columns = (int *)
            prefos_internal_alloc_array(columns, sizeof(int));
    if (!workspace->parallel_support_hashes)
        workspace->parallel_support_hashes = (int *)
            prefos_internal_alloc_array(columns, sizeof(int));
    if (!workspace->parallel_coefficient_hashes)
        workspace->parallel_coefficient_hashes = (int *)
            prefos_internal_alloc_array(columns, sizeof(int));
    if (!workspace->parallel_sort_auxiliary)
        workspace->parallel_sort_auxiliary = (int *)
            prefos_internal_alloc_array(columns, sizeof(int));
    if (!workspace->parallel_group_starts)
        workspace->parallel_group_starts = (int *)
            prefos_internal_alloc_array(columns + 1, sizeof(int));
    if (!workspace->parallel_columns ||
        !workspace->parallel_support_hashes ||
        !workspace->parallel_coefficient_hashes ||
        !workspace->parallel_sort_auxiliary ||
        !workspace->parallel_group_starts)
        return PREFOS_STATUS_OUT_OF_MEMORY;
    return PREFOS_STATUS_OK;
}

static PreFOSStatus reduce_parallel_column_groups_with_support(
    PreFOSPresolver *presolver, PreFOSColumnWorkspace *workspace,
    const PreFOSColumnWorkspace *support_workspace,
    const unsigned char *eligible, int exact_support)
{
    PreFOSParallelColumnContext context;
    PresolveSparseRowView view;
    int *parallel = NULL, *support_hashes = NULL, *coefficient_hashes = NULL;
    int *sort_auxiliary = NULL, *group_starts = NULL;
    unsigned char *gpu_eligible = NULL;
    size_t n_groups = 0, group;
    size_t max_group_size = 0, source_checks = 0, active_pairs = 0;
    size_t source_fixes = 0, target_fixes = 0, merges = 0;
    double minimum_rejected_gap = INFINITY;
    double minimum_rejected_cross_gap = INFINITY;
    double minimum_gap_ratio = NAN;
    double minimum_gap_source_objective = NAN;
    double minimum_gap_target_objective = NAN;
    double minimum_gap_source_lower = NAN;
    double minimum_gap_source_upper = NAN;
    double minimum_gap_target_lower = NAN;
    double minimum_gap_target_upper = NAN;
    int minimum_gap_source = -1, minimum_gap_target = -1;
    PreFOSStatus status = PREFOS_STATUS_OK;
    PreFOSTimestamp detection_start, detection_stop;
    PreFOSTimestamp reduction_start, reduction_stop;
    int trace = getenv("PREFOS_TRACE_PARALLEL_COLUMNS") != NULL;
    int detected = 0;
    int cache_hit = 0;
    size_t detected_active_columns = 0;
    if (!presolver->settings.parallel_column_reduction ||
        presolver->original.n < 2)
        return PREFOS_STATUS_OK;
    context.presolver = presolver;
    context.workspace = support_workspace;
    context.eligible = eligible;
    context.exact_support = exact_support;
    trace_parallel_column_eligibility(&context);
    if (trace && !presolver->working_matrix_is_materialized)
    {
        const PreFOSCsrMatrix *matrix = &presolver->original.A;
        size_t dirty_rows = 0, dirty_source_nnz = 0, row;
        for (row = 0; row < matrix->rows; ++row)
            if (!presolver->remove_rows[row] &&
                workspace->dirty_row[row])
            {
                ++dirty_rows;
                dirty_source_nnz +=
                    (size_t) (matrix->row_pointers[row + 1] -
                              matrix->row_pointers[row]);
            }
        fprintf(
            stderr,
            "PreFOS parallel-column dirty rows=%zu source_nnz=%zu "
            "materialized_rows=%zu\n",
            dirty_rows, dirty_source_nnz,
            presolver->n_rows_require_materialization);
    }
    status = ensure_parallel_column_scratch(presolver, workspace);
    if (status != PREFOS_STATUS_OK) goto cleanup;
    parallel = workspace->parallel_columns;
    support_hashes = workspace->parallel_support_hashes;
    coefficient_hashes = workspace->parallel_coefficient_hashes;
    sort_auxiliary = workspace->parallel_sort_auxiliary;
    group_starts = workspace->parallel_group_starts;
    view = (PresolveSparseRowView){presolver->original.n,
                                  support_workspace->values,
                                  support_workspace->rows,
                                  support_workspace->starts,
                                  support_workspace->ends,
                                  1};
    if (trace) prefos_internal_timer_now(&detection_start);
    if (!exact_support &&
        support_workspace == workspace &&
        incremental_cache_proves_no_parallel_columns(
            &context))
    {
        detected = 1;
        cache_hit = 1;
        detected_active_columns =
            workspace->parallel_cached_active_columns;
    }
    if (!detected &&
        presolver->settings.structural_reductions_gpu &&
        workspace->gpu_csc_valid && !exact_support &&
        support_workspace == workspace)
    {
        PreFOSCudaPropagationStatus cuda_status =
            PREFOS_CUDA_PROPAGATION_UNAVAILABLE;
        PreFOSCudaWorkspace *cuda_workspace = NULL;
        size_t active_columns = 0, column;
        double gpu_milliseconds = 0.0;
        if (!workspace->parallel_gpu_eligible)
            workspace->parallel_gpu_eligible = (unsigned char *)
                prefos_internal_alloc_array(
                    presolver->original.n, sizeof(unsigned char));
        gpu_eligible = workspace->parallel_gpu_eligible;
        if (presolver->original.n > 0 && !gpu_eligible)
        {
            status = PREFOS_STATUS_OUT_OF_MEMORY;
            goto cleanup;
        }
        for (column = 0; column < presolver->original.n; ++column)
            gpu_eligible[column] = (unsigned char)
                (prefos_internal_column_is_linear_box(
                     presolver, workspace, (int) column) &&
                 !workspace->protected_target[column]);
        if (workspace->gpu_csc_valid)
        {
            cuda_workspace =
                prefos_internal_cuda_workspace_get(presolver, &cuda_status);
            if (cuda_workspace &&
                cuda_status == PREFOS_CUDA_PROPAGATION_OK)
                cuda_status = prefos_cuda_parallel_column_hash_sort(
                    cuda_workspace, gpu_eligible, workspace->dirty_row,
                    parallel, support_hashes, coefficient_hashes,
                    &active_columns, &gpu_milliseconds);
        }
        presolver->stats.parallel_column_gpu_milliseconds +=
            gpu_milliseconds;
        if (cuda_status == PREFOS_CUDA_PROPAGATION_OK)
        {
            if (active_columns > 0)
                memcpy(
                    sort_auxiliary, parallel,
                    active_columns * sizeof(int));
            detected = presolve_collect_parallel_row_groups(
                &view, presolver->settings.feasibility_tolerance,
                parallel, active_columns, support_hashes,
                coefficient_hashes, group_starts,
                presolver->original.n + 1, &n_groups);
            if (detected)
            {
                detected_active_columns = active_columns;
                ++presolver->stats.parallel_column_gpu_passes;
            }
        }
        if (!detected)
            ++presolver->stats.parallel_column_gpu_fallbacks;
    }
    if (!detected)
    {
        size_t active_columns = 0, column;
        for (column = 0;
             column < presolver->original.n; ++column)
            if (parallel_column_is_active(&context, column))
                parallel[active_columns++] = (int) column;
        detected = presolve_find_parallel_rows_in_set_with_sorted_copy(
            &view, presolver->settings.feasibility_tolerance, 0,
            presolve_sort_rows_by_hash, parallel, active_columns,
            support_hashes, coefficient_hashes, sort_auxiliary,
            sort_auxiliary, &detected_active_columns,
            group_starts, presolver->original.n + 1, &n_groups);
    }
    if (!detected)
    {
        status = PREFOS_STATUS_NUMERICAL_ERROR;
        goto cleanup;
    }
    workspace->parallel_cache_removed_row_cursor =
        presolver->n_removed_rows;
    clear_parallel_column_dirty_cache(workspace);
    if (trace)
    {
        prefos_internal_timer_now(&detection_stop);
        prefos_internal_timer_now(&reduction_start);
    }
    for (group = 0; group < n_groups; ++group)
    {
        int start = group_starts[group], end = group_starts[group + 1];
        int target_position;
        size_t group_size = (size_t) (end - start);
        int finite_bounded_group = group_size > 1;
        double minimum_first_coefficient_abs = INFINITY;
        if (group_size > max_group_size) max_group_size = group_size;
        if (finite_bounded_group)
        {
            int position;
            for (position = start;
                 finite_bounded_group && position < end; ++position)
            {
                int column = parallel[position];
                int box = presolver->variable_to_box[column];
                double first_coefficient =
                    support_workspace->values[
                        support_workspace->starts[column]];
                double key =
                    normalized_objective_key(
                        support_workspace, column);
                finite_bounded_group =
                    box >= 0 &&
                    isfinite(first_coefficient) &&
                    first_coefficient != 0.0 &&
                    isfinite(key) &&
                    isfinite(workspace->objective[column]) &&
                    isfinite(presolver->working_box_lower[box]) &&
                    isfinite(presolver->working_box_upper[box]);
                minimum_first_coefficient_abs = fmin(
                    minimum_first_coefficient_abs,
                    fabs(first_coefficient));
            }
            if (finite_bounded_group)
                sort_parallel_group_by_normalized_objective(
                    parallel + start, group_size,
                    support_workspace);
        }
        for (target_position = start;
             target_position < end - 1; ++target_position)
        {
            int target = parallel[target_position];
            int source_position;
            if (!parallel_column_remains_active(
                    &context, target))
                continue;
            for (source_position = target_position + 1;
                 source_position < end; ++source_position)
            {
                int source = parallel[source_position];
                int target_box, source_box;
                double ratio, objective_gap;
                double target_coefficient, source_coefficient;
                double objective_cross_gap;
                double source_lower, source_upper;
                double target_lower, target_upper;
                double new_lower, new_upper;
                if (finite_bounded_group &&
                    ((long double)
                         normalized_objective_key(
                             support_workspace, source) -
                     (long double)
                         normalized_objective_key(
                             support_workspace, target)) *
                            fabsl((long double)
                                      support_workspace->values[
                                          support_workspace
                                              ->starts[target]]) *
                            (long double)
                                minimum_first_coefficient_abs >
                        (long double)
                            presolver->settings.feasibility_tolerance)
                    break;
                ++source_checks;
                if (!parallel_column_remains_active(
                        &context, source))
                    continue;
                ++active_pairs;
                ratio =
                    support_workspace->values[
                        support_workspace->starts[source]] /
                    support_workspace->values[
                        support_workspace->starts[target]];
                if (!isfinite(ratio) || ratio == 0.0) continue;
                target_coefficient =
                    support_workspace->values[
                        support_workspace->starts[target]];
                source_coefficient =
                    support_workspace->values[
                        support_workspace->starts[source]];
                objective_cross_gap =
                    workspace->objective[source] *
                        target_coefficient -
                    workspace->objective[target] *
                        source_coefficient;
                source_box = presolver->variable_to_box[source];
                target_box = presolver->variable_to_box[target];
                source_lower =
                    presolver->working_box_lower[source_box];
                source_upper =
                    presolver->working_box_upper[source_box];
                target_lower =
                    presolver->working_box_lower[target_box];
                target_upper =
                    presolver->working_box_upper[target_box];
                if (fabs(objective_cross_gap) >
                    presolver->settings.feasibility_tolerance)
                {
                    if (isfinite(source_lower) &&
                        isfinite(source_upper) &&
                        isfinite(target_lower) &&
                        isfinite(target_upper))
                        continue;
                    objective_gap =
                        workspace->objective[source] -
                        ratio * workspace->objective[target];
                    if (trace &&
                        fabs(objective_gap) <
                            minimum_rejected_gap)
                    {
                        minimum_rejected_gap =
                            fabs(objective_gap);
                        minimum_rejected_cross_gap =
                            fabs(objective_cross_gap);
                        minimum_gap_ratio = ratio;
                        minimum_gap_source_objective =
                            workspace->objective[source];
                        minimum_gap_target_objective =
                            workspace->objective[target];
                        minimum_gap_source_lower = source_lower;
                        minimum_gap_source_upper = source_upper;
                        minimum_gap_target_lower = target_lower;
                        minimum_gap_target_upper = target_upper;
                        minimum_gap_source = source;
                        minimum_gap_target = target;
                    }
                    int fixed_column = -1;
                    double fixed_value = 0.0;
                    if (objective_gap > 0.0)
                    {
                        if ((ratio > 0.0 &&
                             !isfinite(target_upper)) ||
                            (ratio < 0.0 &&
                             !isfinite(target_lower)))
                        {
                            fixed_column = source;
                            fixed_value = source_lower;
                        }
                        else if (ratio > 0.0 &&
                                 !isfinite(source_lower))
                        {
                            fixed_column = target;
                            fixed_value = target_upper;
                        }
                        else if (ratio < 0.0 &&
                                 !isfinite(source_lower))
                        {
                            fixed_column = target;
                            fixed_value = target_lower;
                        }
                    }
                    else
                    {
                        if ((ratio > 0.0 &&
                             !isfinite(target_lower)) ||
                            (ratio < 0.0 &&
                             !isfinite(target_upper)))
                        {
                            fixed_column = source;
                            fixed_value = source_upper;
                        }
                        else if (ratio > 0.0 &&
                                 !isfinite(source_upper))
                        {
                            fixed_column = target;
                            fixed_value = target_lower;
                        }
                        else if (ratio < 0.0 &&
                                 !isfinite(source_upper))
                        {
                            fixed_column = target;
                            fixed_value = target_upper;
                        }
                    }
                    if (fixed_column < 0) continue;
                    if (!isfinite(fixed_value))
                    {
                        status = PREFOS_STATUS_PRIMAL_UNBOUNDED;
                        goto cleanup;
                    }
                    prefos_internal_mark_fixed_column(
                        presolver, fixed_column, fixed_value);
                    ++presolver->stats.dual_fixed_columns;
                    if (fixed_column == target)
                    {
                        ++target_fixes;
                        break;
                    }
                    ++source_fixes;
                    continue;
                }
                new_lower = merged_lower(
                    target_lower, source_lower, source_upper, ratio);
                new_upper = merged_upper(
                    target_upper, source_lower, source_upper, ratio);
                status = append_parallel_record(
                    presolver, source, target, ratio,
                    source_lower, source_upper,
                    target_lower, target_upper);
                if (status != PREFOS_STATUS_OK) goto cleanup;
                prefos_internal_linear_cache_mark_bound_dirty(
                    presolver, target);
                prefos_internal_linear_cache_mark_bound_dirty(
                    presolver, source);
                prefos_internal_mark_fixed_box_dirty(
                    presolver, target);
                prefos_internal_update_cached_singleton_column_bounds(
                    presolver, workspace, target,
                    target_lower, target_upper, new_lower, new_upper);
                prefos_internal_remove_cached_singleton_column(
                    presolver, workspace, source,
                    source_lower, source_upper);
                prefos_internal_clear_box_bound_provenance(
                    presolver, target, 1);
                prefos_internal_clear_box_bound_provenance(
                    presolver, target, 0);
                presolver->working_box_lower[target_box] = new_lower;
                presolver->working_box_upper[target_box] = new_upper;
                presolver->propagation_lower[target] = new_lower;
                presolver->propagation_upper[target] = new_upper;
                presolver->propagation_lower[source] = 0.0;
                presolver->propagation_upper[source] = 0.0;
                presolver->is_parallel_removed[source] = 1;
                presolver->variable_to_box[source] = -1;
                workspace->objective[source] = 0.0;
                workspace->protected_target[source] = 1;
                for (int source_position =
                         support_workspace->starts[source];
                     source_position <
                         support_workspace->ends[source];
                     ++source_position)
                {
                    int source_row =
                        support_workspace->rows[source_position];
                    if (presolver->remove_rows[source_row] ||
                        workspace->dirty_row[source_row])
                        continue;
                    if (workspace->row_degrees[source_row] > 0)
                        --workspace->row_degrees[source_row];
                    if (workspace->row_degrees[source_row] <= 1)
                        prefos_internal_queue_trivial_row(
                            presolver, workspace, source_row);
                }
                ++presolver->stats.merged_parallel_columns;
                ++presolver->n_parallel_column_reductions;
                ++merges;
            }
        }
    }

cleanup:
    workspace->parallel_no_group_cache_valid =
        !exact_support && status == PREFOS_STATUS_OK &&
        detected &&
        cache_closed_parallel_column_groups(
            &context, workspace,
            cache_hit ? parallel : sort_auxiliary,
            detected_active_columns, parallel, group_starts,
            n_groups);
    if (!workspace->parallel_no_group_cache_valid)
        workspace->parallel_cached_active_columns = 0;
    if (merges > 0)
        presolver->scalar_redundancy_completed = 0;
    if (merges > 0 && workspace->protected_target)
        for (size_t column = 0;
             column < presolver->original.n; ++column)
            workspace->protected_target[column] =
                presolver->affine_face_substitution_targets[column];
    if (trace)
    {
        prefos_internal_timer_now(&reduction_stop);
        fprintf(
            stderr,
            "PreFOS parallel columns groups=%zu max_group=%zu "
            "materialized=%d "
            "source_checks=%zu active_pairs=%zu source_fixes=%zu "
            "target_fixes=%zu merges=%zu min_rejected_gap=%.17g "
            "min_cross_gap=%.17g min_pair=(%d,%d) "
            "min_ratio=%.17g min_obj=(%.17g,%.17g) "
            "min_bounds=([%.17g,%.17g],[%.17g,%.17g]) "
            "detect_ms=%.3f reduce_ms=%.3f\n",
            n_groups, max_group_size,
            presolver->working_matrix_is_materialized ||
                exact_support,
            source_checks, active_pairs,
            source_fixes, target_fixes, merges,
            minimum_rejected_gap, minimum_rejected_cross_gap,
            minimum_gap_source, minimum_gap_target,
            minimum_gap_ratio,
            minimum_gap_source_objective,
            minimum_gap_target_objective,
            minimum_gap_source_lower, minimum_gap_source_upper,
            minimum_gap_target_lower, minimum_gap_target_upper,
            detected
                ? prefos_internal_timer_elapsed_milliseconds(
                      &detection_start, &detection_stop)
                : 0.0,
            detected
                ? prefos_internal_timer_elapsed_milliseconds(
                      &reduction_start, &reduction_stop)
                : 0.0);
    }
    return status;
}

PreFOSStatus prefos_internal_reduce_transformed_parallel_columns(
    PreFOSPresolver *presolver, PreFOSColumnWorkspace *workspace,
    int *ran)
{
    PreFOSTransformedColumnView transformed;
    PreFOSColumnWorkspace support;
    PreFOSStatus status;
    int built = 0;

    if (!ran) return PREFOS_STATUS_INVALID_ARGUMENT;
    *ran = 0;
    memset(&transformed, 0, sizeof(transformed));
    status = prefos_internal_build_transformed_column_view(
        presolver, workspace, &transformed, &built);
    if (status != PREFOS_STATUS_OK) return status;
    if (!built)
        return PREFOS_STATUS_OK;

    *ran = 1;
    support = *workspace;
    support.starts = transformed.starts;
    support.ends = transformed.ends;
    support.rows = transformed.rows;
    support.values = transformed.values;
    support.live_degrees = transformed.live_degrees;
    support.nnz = transformed.nnz;
    status = reduce_parallel_column_groups_with_support(
        presolver, workspace, &support,
        transformed.eligible, 1);
    prefos_internal_free_transformed_column_view(&transformed);
    return status;
}

PreFOSStatus
prefos_internal_reduce_parallel_column_groups(
    PreFOSPresolver *presolver, PreFOSColumnWorkspace *workspace)
{
    PreFOSStatus status;
    int ran = 0;
    status = prefos_internal_reduce_transformed_parallel_columns(
        presolver, workspace, &ran);
    if (status != PREFOS_STATUS_OK || ran) return status;
    return reduce_parallel_column_groups_with_support(
        presolver, workspace, workspace, NULL, 0);
}
