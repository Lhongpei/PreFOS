/*
 * Copyright 2026 Hongpei Li
 * SPDX-License-Identifier: Apache-2.0
 */

#include "PREFOS_AffineConeCompaction.h"

static PreFOSStatus accumulate_term(const PreFOSPresolver *presolver, int column,
                                 double coefficient, size_t depth,
                                 double *row_values, int *row_marks,
                                 int *touched_columns, size_t *n_touched,
                                 double *offset)
{
    size_t term;
    int mapped;
    if (column < 0 || (size_t) column >= presolver->original.n ||
        depth > PREFOS_MAX_SUBSTITUTION_DEPTH)
        return PREFOS_STATUS_NUMERICAL_ERROR;
    if (presolver->is_fixed[column])
        return prefos_internal_safe_add_product(offset, coefficient,
                                             presolver->fixed_values[column])
                   ? PREFOS_STATUS_OK
                   : PREFOS_STATUS_NUMERICAL_ERROR;
    if (presolver->is_substituted[column])
    {
        size_t start = presolver->substitution_term_start[column];
        size_t count = presolver->substitution_term_count[column];
        if (count == 0 || start > presolver->n_substitution_terms ||
            count > presolver->n_substitution_terms - start ||
            !prefos_internal_safe_add_product(offset, coefficient,
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
            status = accumulate_term(presolver,
                                     presolver->substitution_targets[start + term],
                                     propagated, depth + 1, row_values, row_marks,
                                     touched_columns, n_touched, offset);
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

static void clear_row(double *row_values, int *row_marks, const int *touched_columns,
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

static PreFOSStatus accumulate_source_row(const PreFOSPresolver *presolver,
                                       size_t affine_row, int direct_column,
                                       double *row_values, int *row_marks,
                                       int *touched_columns, size_t *n_touched,
                                       double *offset)
{
    const PreFOSProblemData *source = &presolver->original;
    PreFOSStatus status;
    *n_touched = 0;
    *offset = 0.0;
    if (direct_column >= 0)
        return accumulate_term(presolver, direct_column, 1.0, 0, row_values,
                               row_marks, touched_columns, n_touched, offset);
    if (affine_row >= source->affine_cone_matrix.rows)
        return PREFOS_STATUS_NUMERICAL_ERROR;
    *offset = source->affine_cone_offset[affine_row];
    {
        int p;
        for (p = source->affine_cone_matrix.row_pointers[affine_row];
             p < source->affine_cone_matrix.row_pointers[affine_row + 1]; ++p)
        {
            status = accumulate_term(
                presolver, source->affine_cone_matrix.column_indices[p],
                source->affine_cone_matrix.values[p], 0, row_values, row_marks,
                touched_columns, n_touched, offset);
            if (status != PREFOS_STATUS_OK) return status;
        }
    }
    return PREFOS_STATUS_OK;
}

PreFOSStatus prefos_internal_build_reduced_affine_cones(PreFOSPresolver *presolver)
{
    const PreFOSProblemData *source = &presolver->original;
    PreFOSPresolvedProblem *target = &presolver->reduced;
    double *row_values = NULL;
    int *row_marks = NULL, *touched_columns = NULL;
    size_t k, row, total_rows = source->affine_cone_matrix.rows;
    size_t block_write = 0, row_write = 0, nnz = 0;
    int matrix_write = 0;
    PreFOSStatus status = PREFOS_STATUS_OK;

    target->n_affine_cones = source->n_affine_cones;
    for (k = 0; k < source->n_cones; ++k)
    {
        if (!presolver->converted_affine_cones[k]) continue;
        ++target->n_affine_cones;
        if (source->cones[k].dimension > SIZE_MAX - total_rows)
            return PREFOS_STATUS_OUT_OF_MEMORY;
        total_rows += source->cones[k].dimension;
    }
    target->affine_cones = (PreFOSAffineConeBlock *) prefos_internal_alloc_array(
        target->n_affine_cones, sizeof(PreFOSAffineConeBlock));
    target->affine_cone_offset =
        (double *) prefos_internal_alloc_array(total_rows, sizeof(double));
    target->affine_cone_matrix.rows = total_rows;
    target->affine_cone_matrix.cols = target->n;
    target->affine_cone_matrix.row_pointers =
        (int *) calloc(total_rows + 1, sizeof(int));
    if ((target->n_affine_cones > 0 && !target->affine_cones) ||
        (total_rows > 0 && !target->affine_cone_offset) ||
        !target->affine_cone_matrix.row_pointers)
        return PREFOS_STATUS_OUT_OF_MEMORY;

    for (k = 0; k < source->n_affine_cones; ++k)
        target->affine_cones[block_write++] = source->affine_cones[k];
    for (k = 0; k < source->n_cones; ++k)
    {
        const PreFOSConeBlock *cone;
        if (!presolver->converted_affine_cones[k]) continue;
        cone = &source->cones[k];
        target->affine_cones[block_write++] = (PreFOSAffineConeBlock){
            cone->type, cone->dimension, cone->matrix_order, cone->power_alpha};
    }
    if (block_write != target->n_affine_cones) return PREFOS_STATUS_NUMERICAL_ERROR;

    row_values = (double *) calloc(target->n, sizeof(double));
    row_marks = (int *) prefos_internal_alloc_array(target->n, sizeof(int));
    touched_columns = (int *) prefos_internal_alloc_array(target->n, sizeof(int));
    if (target->n > 0 && (!row_values || !row_marks || !touched_columns))
    {
        status = PREFOS_STATUS_OUT_OF_MEMORY;
        goto cleanup;
    }
    for (row = 0; row < target->n; ++row) row_marks[row] = -1;

#define PREFOS_COUNT_AFFINE_ROW(source_row, direct_column)                             \
    do                                                                              \
    {                                                                               \
        size_t n_touched = 0, position;                                             \
        double offset = 0.0;                                                        \
        status = accumulate_source_row(presolver, (source_row), (direct_column),    \
                                       row_values, row_marks, touched_columns,      \
                                       &n_touched, &offset);                        \
        if (status != PREFOS_STATUS_OK) goto cleanup;                                  \
        target->affine_cone_matrix.row_pointers[row_write] = (int) nnz;             \
        target->affine_cone_offset[row_write++] = offset;                           \
        for (position = 0; position < n_touched; ++position)                        \
            if (row_values[touched_columns[position]] != 0.0) ++nnz;                \
        clear_row(row_values, row_marks, touched_columns, n_touched);               \
        if (nnz > (size_t) INT_MAX)                                                 \
        {                                                                           \
            status = PREFOS_STATUS_OUT_OF_MEMORY;                                      \
            goto cleanup;                                                           \
        }                                                                           \
    } while (0)

    for (row = 0; row < source->affine_cone_matrix.rows; ++row)
        PREFOS_COUNT_AFFINE_ROW(row, -1);
    for (k = 0; k < source->n_cones; ++k)
    {
        size_t coordinate;
        if (!presolver->converted_affine_cones[k]) continue;
        for (coordinate = 0; coordinate < source->cones[k].dimension; ++coordinate)
            PREFOS_COUNT_AFFINE_ROW(0, source->cones[k].indices[coordinate]);
    }
    if (row_write != total_rows)
    {
        status = PREFOS_STATUS_NUMERICAL_ERROR;
        goto cleanup;
    }
    target->affine_cone_matrix.row_pointers[total_rows] = (int) nnz;
    target->affine_cone_matrix.nnz = nnz;
    target->affine_cone_matrix.values =
        (double *) prefos_internal_alloc_array(nnz, sizeof(double));
    target->affine_cone_matrix.column_indices =
        (int *) prefos_internal_alloc_array(nnz, sizeof(int));
    if (nnz > 0 && (!target->affine_cone_matrix.values ||
                    !target->affine_cone_matrix.column_indices))
    {
        status = PREFOS_STATUS_OUT_OF_MEMORY;
        goto cleanup;
    }

    row_write = 0;
#define PREFOS_WRITE_AFFINE_ROW(source_row, direct_column)                             \
    do                                                                              \
    {                                                                               \
        size_t n_touched = 0, position;                                             \
        double offset = 0.0;                                                        \
        status = accumulate_source_row(presolver, (source_row), (direct_column),    \
                                       row_values, row_marks, touched_columns,      \
                                       &n_touched, &offset);                        \
        if (status != PREFOS_STATUS_OK) goto cleanup;                                  \
        target->affine_cone_matrix.row_pointers[row_write++] = matrix_write;        \
        for (position = 0; position < n_touched; ++position)                        \
        {                                                                           \
            int column = touched_columns[position];                                 \
            if (row_values[column] == 0.0) continue;                                \
            target->affine_cone_matrix.values[matrix_write] = row_values[column];   \
            target->affine_cone_matrix.column_indices[matrix_write++] = column;     \
        }                                                                           \
        clear_row(row_values, row_marks, touched_columns, n_touched);               \
    } while (0)

    for (row = 0; row < source->affine_cone_matrix.rows; ++row)
        PREFOS_WRITE_AFFINE_ROW(row, -1);
    for (k = 0; k < source->n_cones; ++k)
    {
        size_t coordinate;
        if (!presolver->converted_affine_cones[k]) continue;
        for (coordinate = 0; coordinate < source->cones[k].dimension; ++coordinate)
            PREFOS_WRITE_AFFINE_ROW(0, source->cones[k].indices[coordinate]);
    }
    target->affine_cone_matrix.row_pointers[total_rows] = matrix_write;
    if (row_write != total_rows || (size_t) matrix_write != nnz)
        status = PREFOS_STATUS_NUMERICAL_ERROR;

#undef PREFOS_COUNT_AFFINE_ROW
#undef PREFOS_WRITE_AFFINE_ROW

cleanup:
    free(row_values);
    free(row_marks);
    free(touched_columns);
    return status;
}
