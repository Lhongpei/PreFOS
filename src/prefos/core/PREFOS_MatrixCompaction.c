/*
 * Copyright 2026 Hongpei Li
 * SPDX-License-Identifier: Apache-2.0
 */

#include "PREFOS_MatrixCompaction.h"

static PreFOSStatus compact_a_without_substitutions(PreFOSPresolver *presolver)
{
    const PreFOSProblemData *source = &presolver->original;
    PreFOSPresolvedProblem *target = &presolver->reduced;
    unsigned char *keep_row;
    double *shifts;
    size_t row, kept_rows = 0, nnz = 0;
    int write = 0;

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
            else if (source->A.values[p] != 0.0)
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
    if (presolver->is_substituted[column])
    {
        size_t start = presolver->substitution_term_start[column];
        unsigned char count = presolver->substitution_term_count[column];
        if (count == 0 || count > PREFOS_MAX_AGGREGATION_TERMS ||
            start > presolver->n_substitution_terms ||
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

PreFOSStatus prefos_internal_compact_a(PreFOSPresolver *presolver)
{
    const PreFOSProblemData *source = &presolver->original;
    PreFOSPresolvedProblem *target = &presolver->reduced;
    unsigned char *keep_row;
    double *shifts;
    double *row_values;
    int *row_marks;
    int *touched_columns;
    size_t row, kept_rows = 0, nnz = 0;
    int write = 0;
    PreFOSStatus status = PREFOS_STATUS_OK;

    if (presolver->stats.substituted_free_variables == 0)
        return compact_a_without_substitutions(presolver);

    keep_row = (unsigned char *) calloc(source->A.rows, sizeof(unsigned char));
    shifts = (double *) calloc(source->A.rows, sizeof(double));
    row_values = (double *) calloc(target->n, sizeof(double));
    row_marks = (int *) prefos_internal_alloc_array(target->n, sizeof(int));
    touched_columns = (int *) prefos_internal_alloc_array(target->n, sizeof(int));
    if ((source->A.rows > 0 && (!keep_row || !shifts)) ||
        (target->n > 0 && (!row_values || !row_marks || !touched_columns)))
    {
        free(keep_row);
        free(shifts);
        free(row_values);
        free(row_marks);
        free(touched_columns);
        return PREFOS_STATUS_OUT_OF_MEMORY;
    }
    for (row = 0; row < target->n; ++row) row_marks[row] = -1;

    for (row = 0; row < source->A.rows; ++row)
    {
        size_t remaining = 0, n_touched = 0, position;
        double lower, upper;
        if (presolver->remove_rows[row]) continue;
        status =
            accumulate_transformed_a_row(presolver, row, row_values, row_marks,
                                         touched_columns, &n_touched, &shifts[row]);
        if (status != PREFOS_STATUS_OK) goto compact_failure;
        for (position = 0; position < n_touched; ++position)
            if (row_values[touched_columns[position]] != 0.0) ++remaining;
        clear_transformed_a_row(row_values, row_marks, touched_columns, n_touched);
        lower = presolver->working_constraint_lower[row] - shifts[row];
        upper = presolver->working_constraint_upper[row] - shifts[row];
        if (isnan(lower) || isnan(upper))
        {
            status = PREFOS_STATUS_NUMERICAL_ERROR;
            goto compact_failure;
        }
        if (remaining == 0 && presolver->settings.remove_empty_rows)
        {
            if (lower > presolver->settings.feasibility_tolerance ||
                upper < -presolver->settings.feasibility_tolerance)
            {
                status = PREFOS_STATUS_PRIMAL_INFEASIBLE;
                goto compact_failure;
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
        status = PREFOS_STATUS_OUT_OF_MEMORY;
        goto compact_failure;
    }

    kept_rows = 0;
    for (row = 0; row < source->A.rows; ++row)
    {
        size_t n_touched = 0, position;
        double ignored_shift;
        if (!keep_row[row]) continue;
        presolver->original_to_reduced_rows[row] = (int) kept_rows;
        target->A.row_pointers[kept_rows] = write;
        target->constraint_lower[kept_rows] =
            presolver->working_constraint_lower[row] - shifts[row];
        target->constraint_upper[kept_rows] =
            presolver->working_constraint_upper[row] - shifts[row];
        status = accumulate_transformed_a_row(presolver, row, row_values, row_marks,
                                              touched_columns, &n_touched,
                                              &ignored_shift);
        if (status != PREFOS_STATUS_OK) goto compact_failure;
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
    target->A.row_pointers[kept_rows] = write;
    free(keep_row);
    free(shifts);
    free(row_values);
    free(row_marks);
    free(touched_columns);
    return PREFOS_STATUS_OK;

compact_failure:
    free(keep_row);
    free(shifts);
    free(row_values);
    free(row_marks);
    free(touched_columns);
    return status;
}
