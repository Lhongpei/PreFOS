/*
 * Copyright 2026 Hongpei Li
 * SPDX-License-Identifier: Apache-2.0
 */

#include "PREFOS_MatrixCompaction.h"

#include "explorers/PREFOS_CudaBackend.h"
#include "explorers/PREFOS_CudaLinearPropagation.h"

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
    PreFOSStatus gpu_status =
        compact_a_without_substitutions_gpu(presolver, &used_gpu,
                                            &attempted_gpu);
    if (gpu_status != PREFOS_STATUS_OK) return gpu_status;
    if (used_gpu) return PREFOS_STATUS_OK;
    if (attempted_gpu)
        ++presolver->stats.matrix_compaction_gpu_fallbacks;

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
            else if (!presolver->is_parallel_removed[column] &&
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

static PreFOSStatus accumulate_transformed_a_row(const PreFOSPresolver *presolver,
                                              size_t row, double *row_values,
                                              int *row_marks, int *touched_columns,
                                              size_t *n_touched, double *shift)
{
    const PreFOSCsrMatrix *A = &presolver->original.A;
    int p;

    *n_touched = 0;
    *shift = 0.0;
    for (p = A->row_pointers[row]; p < A->row_pointers[row + 1]; ++p)
    {
        int column = A->column_indices[p];
        double coefficient = A->values[p];
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

PreFOSStatus prefos_internal_compact_a(PreFOSPresolver *presolver)
{
    const PreFOSProblemData *source = &presolver->original;
    PreFOSPresolvedProblem *target = &presolver->reduced;
    double *row_values;
    int *row_marks;
    int *touched_columns;
    size_t row, kept_rows = 0, write = 0, entry_capacity;
    size_t removed_empty_rows = 0;
    PreFOSStatus status = PREFOS_STATUS_OK;

    if (presolver->stats.substituted_free_variables == 0)
        return compact_a_without_substitutions(presolver);

    row_values = (double *) calloc(target->n, sizeof(double));
    row_marks = (int *) prefos_internal_alloc_array(target->n, sizeof(int));
    touched_columns = (int *) prefos_internal_alloc_array(target->n, sizeof(int));
    target->A.row_pointers = (int *) prefos_internal_alloc_array(
        source->A.rows + 1, sizeof(int));
    target->constraint_lower = (double *) prefos_internal_alloc_array(
        source->A.rows, sizeof(double));
    target->constraint_upper = (double *) prefos_internal_alloc_array(
        source->A.rows, sizeof(double));
    entry_capacity = source->A.nnz;
    target->A.values =
        (double *) prefos_internal_alloc_array(entry_capacity, sizeof(double));
    target->A.column_indices =
        (int *) prefos_internal_alloc_array(entry_capacity, sizeof(int));
    if ((target->n > 0 && (!row_values || !row_marks || !touched_columns)) ||
        !target->A.row_pointers ||
        (source->A.rows > 0 &&
         (!target->constraint_lower || !target->constraint_upper)) ||
        (entry_capacity > 0 &&
         (!target->A.values || !target->A.column_indices)))
    {
        status = PREFOS_STATUS_OUT_OF_MEMORY;
        goto compact_failure;
    }
    for (row = 0; row < target->n; ++row) row_marks[row] = -1;

    for (row = 0; row < source->A.rows; ++row)
    {
        size_t remaining = 0, n_touched = 0, position;
        double shift, lower, upper;
        presolver->original_to_reduced_rows[row] = -1;
        if (presolver->remove_rows[row]) continue;
        status =
            accumulate_transformed_a_row(presolver, row, row_values, row_marks,
                                         touched_columns, &n_touched, &shift);
        if (status != PREFOS_STATUS_OK) goto compact_failure;
        for (position = 0; position < n_touched; ++position)
            if (row_values[touched_columns[position]] != 0.0) ++remaining;
        lower = presolver->working_constraint_lower[row] - shift;
        upper = presolver->working_constraint_upper[row] - shift;
        if (isnan(lower) || isnan(upper))
        {
            clear_transformed_a_row(row_values, row_marks, touched_columns,
                                    n_touched);
            status = PREFOS_STATUS_NUMERICAL_ERROR;
            goto compact_failure;
        }
        if (remaining == 0 && presolver->settings.remove_empty_rows)
        {
            if (lower > presolver->settings.feasibility_tolerance ||
                upper < -presolver->settings.feasibility_tolerance)
            {
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
    free(row_marks);
    free(touched_columns);
    return PREFOS_STATUS_OK;

compact_failure:
    prefos_internal_free_csr(&target->A);
    free(target->constraint_lower);
    free(target->constraint_upper);
    target->constraint_lower = NULL;
    target->constraint_upper = NULL;
    free(row_values);
    free(row_marks);
    free(touched_columns);
    return status;
}
