/*
 * Copyright 2026 Hongpei Li
 * SPDX-License-Identifier: Apache-2.0
 */

#include "PREFOS_AffineFaceSubstitutionInternal.h"

void prefos_affine_face_free_column_storage(PreFOSColumnStorage *storage)
{
    free(storage->starts);
    free(storage->rows);
    free(storage->coefficients);
    memset(storage, 0, sizeof(*storage));
}

PreFOSStatus prefos_affine_face_build_column_storage(const PreFOSCsrMatrix *matrix,
                                               PreFOSColumnStorage *storage)
{
    int *cursor = NULL;
    size_t row, position;
    memset(storage, 0, sizeof(*storage));
    storage->starts = (int *) calloc(matrix->cols + 1, sizeof(int));
    cursor = (int *) prefos_internal_alloc_array(matrix->cols, sizeof(int));
    storage->rows = (int *) prefos_internal_alloc_array(matrix->nnz, sizeof(int));
    storage->coefficients =
        (double *) prefos_internal_alloc_array(matrix->nnz, sizeof(double));
    if (!storage->starts || (matrix->cols > 0 && !cursor) ||
        (matrix->nnz > 0 && (!storage->rows || !storage->coefficients)))
    {
        free(cursor);
        prefos_affine_face_free_column_storage(storage);
        return PREFOS_STATUS_OUT_OF_MEMORY;
    }
    for (position = 0; position < matrix->nnz; ++position)
        if (matrix->values[position] != 0.0)
            ++storage->starts[matrix->column_indices[position] + 1];
    for (position = 0; position < matrix->cols; ++position)
        storage->starts[position + 1] += storage->starts[position];
    if (matrix->cols > 0)
        memcpy(cursor, storage->starts, matrix->cols * sizeof(int));
    for (row = 0; row < matrix->rows; ++row)
    {
        int p;
        for (p = matrix->row_pointers[row]; p < matrix->row_pointers[row + 1]; ++p)
        {
            int column, write;
            if (matrix->values[p] == 0.0) continue;
            column = matrix->column_indices[p];
            write = cursor[column]++;
            storage->rows[write] = (int) row;
            storage->coefficients[write] = matrix->values[p];
        }
    }
    free(cursor);
    return PREFOS_STATUS_OK;
}

PreFOSStatus prefos_affine_face_build_affine_column_storage(
    const PreFOSPresolver *presolver, PreFOSColumnStorage *storage)
{
    const PreFOSProblemData *problem = &presolver->original;
    int *cursor = NULL;
    size_t cone_index, row, generated_row, nnz;
    memset(storage, 0, sizeof(*storage));
    nnz = problem->affine_cone_matrix.nnz;
    for (cone_index = 0; cone_index < problem->n_cones; ++cone_index)
    {
        const PreFOSConeBlock *cone = &problem->cones[cone_index];
        size_t coordinate;
        if (!presolver->converted_affine_cones[cone_index]) continue;
        for (coordinate = 0; coordinate < cone->dimension; ++coordinate)
        {
            int column = cone->indices[coordinate];
            if (presolver->is_fixed[column]) continue;
            if (!presolver->is_substituted[column] ||
                presolver->affine_aggregation_source_rows[column] < 0)
                return PREFOS_STATUS_NUMERICAL_ERROR;
            if (presolver->substitution_term_count[column] > SIZE_MAX - nnz)
                return PREFOS_STATUS_OUT_OF_MEMORY;
            nnz += presolver->substitution_term_count[column];
        }
    }
    storage->starts = (int *) calloc(problem->n + 1, sizeof(int));
    cursor = (int *) prefos_internal_alloc_array(problem->n, sizeof(int));
    storage->rows = (int *) prefos_internal_alloc_array(nnz, sizeof(int));
    storage->coefficients =
        (double *) prefos_internal_alloc_array(nnz, sizeof(double));
    if (!storage->starts || (problem->n > 0 && !cursor) ||
        (nnz > 0 && (!storage->rows || !storage->coefficients)))
    {
        free(cursor);
        prefos_affine_face_free_column_storage(storage);
        return PREFOS_STATUS_OUT_OF_MEMORY;
    }
    for (row = 0; row < problem->affine_cone_matrix.nnz; ++row)
        if (problem->affine_cone_matrix.values[row] != 0.0)
            ++storage->starts
                [problem->affine_cone_matrix.column_indices[row] + 1];
    for (cone_index = 0; cone_index < problem->n_cones; ++cone_index)
    {
        const PreFOSConeBlock *cone = &problem->cones[cone_index];
        size_t coordinate;
        if (!presolver->converted_affine_cones[cone_index]) continue;
        for (coordinate = 0; coordinate < cone->dimension; ++coordinate)
        {
            int column = cone->indices[coordinate];
            size_t start, term;
            if (presolver->is_fixed[column]) continue;
            start = presolver->substitution_term_start[column];
            for (term = 0; term < presolver->substitution_term_count[column]; ++term)
                if (presolver->substitution_scales[start + term] != 0.0)
                    ++storage->starts
                        [presolver->substitution_targets[start + term] + 1];
        }
    }
    for (row = 0; row < problem->n; ++row)
        storage->starts[row + 1] += storage->starts[row];
    if (problem->n > 0)
        memcpy(cursor, storage->starts, problem->n * sizeof(int));
    for (row = 0; row < problem->affine_cone_matrix.rows; ++row)
    {
        int p;
        for (p = problem->affine_cone_matrix.row_pointers[row];
             p < problem->affine_cone_matrix.row_pointers[row + 1]; ++p)
        {
            int column, write;
            if (problem->affine_cone_matrix.values[p] == 0.0) continue;
            column = problem->affine_cone_matrix.column_indices[p];
            write = cursor[column]++;
            storage->rows[write] = (int) row;
            storage->coefficients[write] =
                problem->affine_cone_matrix.values[p];
        }
    }
    generated_row = problem->affine_cone_matrix.rows;
    for (cone_index = 0; cone_index < problem->n_cones; ++cone_index)
    {
        const PreFOSConeBlock *cone = &problem->cones[cone_index];
        size_t coordinate;
        if (!presolver->converted_affine_cones[cone_index]) continue;
        for (coordinate = 0; coordinate < cone->dimension;
             ++coordinate, ++generated_row)
        {
            int source = cone->indices[coordinate];
            size_t start, term;
            if (presolver->is_fixed[source]) continue;
            start = presolver->substitution_term_start[source];
            for (term = 0; term < presolver->substitution_term_count[source]; ++term)
            {
                int column, write;
                double coefficient =
                    presolver->substitution_scales[start + term];
                if (coefficient == 0.0) continue;
                column = presolver->substitution_targets[start + term];
                write = cursor[column]++;
                storage->rows[write] = (int) generated_row;
                storage->coefficients[write] = coefficient;
            }
        }
    }
    free(cursor);
    return PREFOS_STATUS_OK;
}

static int exact_box_value(const PreFOSPresolver *presolver, int column,
                           double *value)
{
    int box = presolver->variable_to_box[column];
    if (presolver->is_fixed[column])
    {
        *value = presolver->fixed_values[column];
        return 1;
    }
    if (box >= 0 && isfinite(presolver->working_box_lower[box]) &&
        presolver->working_box_lower[box] == presolver->working_box_upper[box])
    {
        *value = presolver->working_box_lower[box];
        return 1;
    }
    return 0;
}

static PreFOSStatus expand_column(const PreFOSPresolver *presolver, int column,
                               double coefficient, size_t depth,
                               PreFOSExpandedAffineRow *row)
{
    double fixed;
    size_t term;
    if (column < 0 || (size_t) column >= presolver->original.n ||
        depth > PREFOS_MAX_SUBSTITUTION_DEPTH)
        return PREFOS_STATUS_NUMERICAL_ERROR;
    if (presolver->is_substituted[column])
    {
        size_t start = presolver->substitution_term_start[column];
        unsigned char count = presolver->substitution_term_count[column];
        if (count == 0 || count > PREFOS_MAX_AGGREGATION_TERMS ||
            start > presolver->n_substitution_terms ||
            count > presolver->n_substitution_terms - start ||
            !prefos_internal_safe_add_product(
                &row->constant, coefficient,
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
            status = expand_column(
                presolver, presolver->substitution_targets[start + term],
                propagated, depth + 1, row);
            if (status != PREFOS_STATUS_OK) return status;
        }
        return PREFOS_STATUS_OK;
    }
    if (exact_box_value(presolver, column, &fixed))
        return prefos_internal_safe_add_product(&row->constant, coefficient, fixed)
                   ? PREFOS_STATUS_OK
                   : PREFOS_STATUS_NUMERICAL_ERROR;
    if (row->marks[column] < 0)
    {
        row->marks[column] = 1;
        row->values[column] = coefficient;
        row->touched[row->count++] = column;
    }
    else if (!prefos_internal_safe_add_product(&row->values[column], 1.0,
                                            coefficient))
        return PREFOS_STATUS_NUMERICAL_ERROR;
    return PREFOS_STATUS_OK;
}

void prefos_affine_face_clear_expanded_row(PreFOSExpandedAffineRow *row)
{
    size_t position;
    for (position = 0; position < row->count; ++position)
    {
        int column = row->touched[position];
        row->values[column] = 0.0;
        row->marks[column] = -1;
    }
    row->count = 0;
    row->constant = 0.0;
}

PreFOSStatus prefos_affine_face_expand_input_row(const PreFOSPresolver *presolver,
                                           size_t affine_row,
                                           PreFOSExpandedAffineRow *expanded)
{
    const PreFOSProblemData *problem = &presolver->original;
    int p;
    prefos_affine_face_clear_expanded_row(expanded);
    expanded->constant = problem->affine_cone_offset[affine_row];
    for (p = problem->affine_cone_matrix.row_pointers[affine_row];
         p < problem->affine_cone_matrix.row_pointers[affine_row + 1]; ++p)
    {
        PreFOSStatus status;
        if (problem->affine_cone_matrix.values[p] == 0.0) continue;
        status = expand_column(
            presolver, problem->affine_cone_matrix.column_indices[p],
            problem->affine_cone_matrix.values[p], 0, expanded);
        if (status != PREFOS_STATUS_OK) return status;
    }
    return PREFOS_STATUS_OK;
}

PreFOSStatus prefos_affine_face_expand_generated_row(const PreFOSPresolver *presolver,
                                               int coordinate_column,
                                               PreFOSExpandedAffineRow *expanded)
{
    prefos_affine_face_clear_expanded_row(expanded);
    return expand_column(presolver, coordinate_column, 1.0, 0, expanded);
}

size_t prefos_affine_face_active_column_degree(const PreFOSPresolver *presolver,
                                            const PreFOSColumnStorage *storage,
                                            int column)
{
    size_t degree = 0;
    int position;
    for (position = storage->starts[column];
         position < storage->starts[column + 1]; ++position)
        if (!presolver->remove_rows[storage->rows[position]]) ++degree;
    return degree;
}
