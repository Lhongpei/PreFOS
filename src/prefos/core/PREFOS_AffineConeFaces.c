/*
 * Copyright 2026 Hongpei Li
 * SPDX-License-Identifier: Apache-2.0
 */

#include "PREFOS_AffineConeFaces.h"

typedef struct
{
    size_t rsoc_faces;
    size_t psd_faces;
    size_t exponential_faces;
    size_t power_faces;
    size_t removed_coordinates;
    size_t removed_blocks;
} PreFOSAffineFaceCounts;

static int row_is_zero(const PreFOSPresolvedProblem *problem, size_t row)
{
    return problem->affine_cone_matrix.row_pointers[row] ==
               problem->affine_cone_matrix.row_pointers[row + 1] &&
           problem->affine_cone_offset[row] == 0.0;
}

static size_t packed_index(size_t row, size_t column)
{
    if (row < column)
    {
        size_t temporary = row;
        row = column;
        column = temporary;
    }
    return row * (row + 1) / 2 + column;
}

static int block_is_zero(const PreFOSPresolvedProblem *problem, size_t first_row,
                         size_t dimension)
{
    size_t coordinate;
    for (coordinate = 0; coordinate < dimension; ++coordinate)
        if (!row_is_zero(problem, first_row + coordinate)) return 0;
    return 1;
}

static void remove_block(unsigned char *keep, size_t first_row, size_t dimension,
                         unsigned char *emit, PreFOSAffineFaceCounts *counts)
{
    size_t coordinate;
    for (coordinate = 0; coordinate < dimension; ++coordinate)
        keep[first_row + coordinate] = 0;
    *emit = 0;
    counts->removed_coordinates += dimension;
    ++counts->removed_blocks;
}

static void reduce_nonnegative_block(const PreFOSPresolvedProblem *problem,
                                     size_t first_row,
                                     const PreFOSAffineConeBlock *source,
                                     PreFOSAffineConeBlock *target,
                                     unsigned char *keep, unsigned char *emit,
                                     PreFOSAffineFaceCounts *counts)
{
    size_t coordinate, retained = 0;
    for (coordinate = 0; coordinate < source->dimension; ++coordinate)
    {
        if (row_is_zero(problem, first_row + coordinate))
            keep[first_row + coordinate] = 0;
        else
            ++retained;
    }
    if (retained == source->dimension) return;
    counts->removed_coordinates += source->dimension - retained;
    if (retained == 0)
    {
        *emit = 0;
        ++counts->removed_blocks;
        return;
    }
    target->dimension = retained;
}

static void reduce_rsoc_block(const PreFOSPresolvedProblem *problem, size_t first_row,
                              const PreFOSAffineConeBlock *source,
                              PreFOSAffineConeBlock *target, unsigned char *keep,
                              PreFOSAffineFaceCounts *counts)
{
    size_t coordinate, survivor;
    int tails_zero = 1;
    for (coordinate = 2; coordinate < source->dimension; ++coordinate)
        if (!row_is_zero(problem, first_row + coordinate))
        {
            tails_zero = 0;
            break;
        }
    if (!tails_zero) return;
    if (row_is_zero(problem, first_row))
        survivor = 1;
    else if (row_is_zero(problem, first_row + 1))
        survivor = 0;
    else
        return;
    for (coordinate = 0; coordinate < source->dimension; ++coordinate)
        if (coordinate != survivor) keep[first_row + coordinate] = 0;
    *target = (PreFOSAffineConeBlock){PREFOS_CONE_NONNEGATIVE, 1, 0, 0.0};
    counts->removed_coordinates += source->dimension - 1;
    ++counts->rsoc_faces;
}

static PreFOSStatus reduce_psd_block(
    const PreFOSPresolvedProblem *problem, size_t first_row,
    const PreFOSAffineConeBlock *source, PreFOSAffineConeBlock *target,
    unsigned char *keep, unsigned char *emit, PreFOSAffineFaceCounts *counts)
{
    unsigned char *remove_index = NULL;
    size_t i, j, removed = 0, retained_order;
    remove_index = (unsigned char *) calloc(source->matrix_order,
                                            sizeof(unsigned char));
    if (!remove_index) return PREFOS_STATUS_OUT_OF_MEMORY;
    for (i = 0; i < source->matrix_order; ++i)
    {
        int removable = row_is_zero(problem, first_row + packed_index(i, i));
        for (j = 0; removable && j < source->matrix_order; ++j)
            removable = row_is_zero(problem, first_row + packed_index(i, j));
        if (removable)
        {
            remove_index[i] = 1;
            ++removed;
        }
    }
    if (removed == 0)
    {
        free(remove_index);
        return PREFOS_STATUS_OK;
    }
    retained_order = source->matrix_order - removed;
    for (i = 0; i < source->matrix_order; ++i)
        for (j = 0; j <= i; ++j)
            if (remove_index[i] || remove_index[j])
                keep[first_row + packed_index(i, j)] = 0;
    if (retained_order == 0)
    {
        *emit = 0;
        ++counts->removed_blocks;
    }
    else
    {
        target->dimension = retained_order * (retained_order + 1) / 2;
        target->matrix_order = retained_order;
    }
    counts->removed_coordinates +=
        source->dimension - retained_order * (retained_order + 1) / 2;
    ++counts->psd_faces;
    free(remove_index);
    return PREFOS_STATUS_OK;
}

static void reduce_exponential_block(const PreFOSPresolvedProblem *problem,
                                     size_t first_row,
                                     PreFOSAffineConeBlock *target,
                                     unsigned char *keep,
                                     PreFOSAffineFaceCounts *counts)
{
    if (!row_is_zero(problem, first_row) ||
        !row_is_zero(problem, first_row + 1))
        return;
    keep[first_row] = 0;
    keep[first_row + 1] = 0;
    *target = (PreFOSAffineConeBlock){PREFOS_CONE_NONNEGATIVE, 1, 0, 0.0};
    counts->removed_coordinates += 2;
    ++counts->exponential_faces;
}

static void reduce_power_block(const PreFOSPresolvedProblem *problem, size_t first_row,
                               PreFOSAffineConeBlock *target, unsigned char *keep,
                               PreFOSAffineFaceCounts *counts)
{
    size_t retained = 0;
    if (!row_is_zero(problem, first_row + 2)) return;
    keep[first_row + 2] = 0;
    if (row_is_zero(problem, first_row))
        keep[first_row] = 0;
    else
        ++retained;
    if (row_is_zero(problem, first_row + 1))
        keep[first_row + 1] = 0;
    else
        ++retained;
    *target = (PreFOSAffineConeBlock){PREFOS_CONE_NONNEGATIVE, retained, 0, 0.0};
    counts->removed_coordinates += 3 - retained;
    ++counts->power_faces;
}

static PreFOSStatus rebuild_affine_model(PreFOSPresolver *presolver,
                                      const unsigned char *keep,
                                      const unsigned char *emit,
                                      const PreFOSAffineConeBlock *planned,
                                      size_t new_rows, size_t new_blocks,
                                      const PreFOSAffineFaceCounts *counts)
{
    PreFOSPresolvedProblem *problem = &presolver->reduced;
    PreFOSCsrMatrix matrix;
    PreFOSAffineConeBlock *blocks = NULL;
    double *offset = NULL;
    size_t pre_row, row_write = 0, block, block_write = 0, planned_rows = 0;
    size_t nnz = 0;
    int matrix_write = 0;
    memset(&matrix, 0, sizeof(matrix));
    for (pre_row = 0; pre_row < problem->affine_cone_matrix.rows; ++pre_row)
        if (keep[pre_row])
        {
            int start = problem->affine_cone_matrix.row_pointers[pre_row];
            int end = problem->affine_cone_matrix.row_pointers[pre_row + 1];
            nnz += (size_t) (end - start);
        }
    matrix.rows = new_rows;
    matrix.cols = problem->n;
    matrix.nnz = nnz;
    matrix.row_pointers = (int *) calloc(new_rows + 1, sizeof(int));
    matrix.values = (double *) prefos_internal_alloc_array(nnz, sizeof(double));
    matrix.column_indices = (int *) prefos_internal_alloc_array(nnz, sizeof(int));
    offset = (double *) prefos_internal_alloc_array(new_rows, sizeof(double));
    blocks = (PreFOSAffineConeBlock *) prefos_internal_alloc_array(
        new_blocks, sizeof(PreFOSAffineConeBlock));
    if (!matrix.row_pointers ||
        (nnz > 0 && (!matrix.values || !matrix.column_indices)) ||
        (new_rows > 0 && !offset) || (new_blocks > 0 && !blocks))
    {
        prefos_internal_free_csr(&matrix);
        free(offset);
        free(blocks);
        return PREFOS_STATUS_OUT_OF_MEMORY;
    }

    for (block = 0; block < problem->n_affine_cones; ++block)
        if (emit[block])
        {
            blocks[block_write++] = planned[block];
            planned_rows += planned[block].dimension;
        }
    for (pre_row = 0; pre_row < problem->affine_cone_matrix.rows; ++pre_row)
    {
        int p;
        if (!keep[pre_row]) continue;
        presolver->affine_pre_to_reduced_rows[pre_row] = (int) row_write;
        matrix.row_pointers[row_write] = matrix_write;
        offset[row_write] = problem->affine_cone_offset[pre_row];
        for (p = problem->affine_cone_matrix.row_pointers[pre_row];
             p < problem->affine_cone_matrix.row_pointers[pre_row + 1]; ++p)
        {
            matrix.values[matrix_write] = problem->affine_cone_matrix.values[p];
            matrix.column_indices[matrix_write] =
                problem->affine_cone_matrix.column_indices[p];
            ++matrix_write;
        }
        ++row_write;
    }
    matrix.row_pointers[new_rows] = matrix_write;
    if (row_write != new_rows || planned_rows != new_rows ||
        block_write != new_blocks ||
        (size_t) matrix_write != nnz)
    {
        prefos_internal_free_csr(&matrix);
        free(offset);
        free(blocks);
        return PREFOS_STATUS_NUMERICAL_ERROR;
    }

    prefos_internal_free_csr(&problem->affine_cone_matrix);
    free(problem->affine_cone_offset);
    free(problem->affine_cones);
    problem->affine_cone_matrix = matrix;
    problem->affine_cone_offset = offset;
    problem->affine_cones = blocks;
    problem->n_affine_cones = new_blocks;
    presolver->stats.reduced_affine_rsoc_faces += counts->rsoc_faces;
    presolver->stats.reduced_affine_psd_faces += counts->psd_faces;
    presolver->stats.reduced_affine_exponential_faces += counts->exponential_faces;
    presolver->stats.reduced_affine_power_faces += counts->power_faces;
    presolver->stats.removed_affine_cone_coordinates += counts->removed_coordinates;
    presolver->stats.removed_affine_cone_blocks += counts->removed_blocks;
    return PREFOS_STATUS_OK;
}

PreFOSStatus prefos_internal_reduce_affine_cone_faces(PreFOSPresolver *presolver)
{
    PreFOSPresolvedProblem *problem = &presolver->reduced;
    unsigned char *keep = NULL, *emit = NULL;
    PreFOSAffineConeBlock *planned = NULL;
    PreFOSAffineFaceCounts counts;
    size_t block, first_row = 0, new_rows = 0, new_blocks = 0;
    PreFOSStatus status = PREFOS_STATUS_OK;

    presolver->n_pre_face_affine_rows = problem->affine_cone_matrix.rows;
    presolver->affine_pre_to_reduced_rows = (int *) prefos_internal_alloc_array(
        presolver->n_pre_face_affine_rows, sizeof(int));
    if (presolver->n_pre_face_affine_rows > 0 &&
        !presolver->affine_pre_to_reduced_rows)
        return PREFOS_STATUS_OUT_OF_MEMORY;
    for (first_row = 0; first_row < presolver->n_pre_face_affine_rows; ++first_row)
        presolver->affine_pre_to_reduced_rows[first_row] = (int) first_row;
    if (problem->n_affine_cones == 0 || !presolver->settings.cone_propagation)
        return PREFOS_STATUS_OK;

    keep = (unsigned char *) prefos_internal_alloc_array(
        problem->affine_cone_matrix.rows, sizeof(unsigned char));
    emit = (unsigned char *) calloc(problem->n_affine_cones, sizeof(unsigned char));
    planned = (PreFOSAffineConeBlock *) prefos_internal_alloc_array(
        problem->n_affine_cones, sizeof(PreFOSAffineConeBlock));
    if (!keep || !emit || !planned)
    {
        status = PREFOS_STATUS_OUT_OF_MEMORY;
        goto cleanup;
    }
    memset(keep, 1, problem->affine_cone_matrix.rows * sizeof(unsigned char));
    memset(&counts, 0, sizeof(counts));
    first_row = 0;
    for (block = 0; block < problem->n_affine_cones; ++block)
    {
        const PreFOSAffineConeBlock *source = &problem->affine_cones[block];
        PreFOSAffineConeBlock *target = &planned[block];
        if (first_row > problem->affine_cone_matrix.rows ||
            source->dimension > problem->affine_cone_matrix.rows - first_row)
        {
            status = PREFOS_STATUS_NUMERICAL_ERROR;
            goto cleanup;
        }
        *target = *source;
        emit[block] = 1;
        if (block_is_zero(problem, first_row, source->dimension))
            remove_block(keep, first_row, source->dimension, &emit[block], &counts);
        else if (source->type == PREFOS_CONE_NONNEGATIVE)
            reduce_nonnegative_block(problem, first_row, source, target, keep,
                                     &emit[block], &counts);
        else if (source->type == PREFOS_CONE_ROTATED_SECOND_ORDER &&
                 presolver->settings.rsoc_face_reduction)
            reduce_rsoc_block(problem, first_row, source, target, keep, &counts);
        else if (source->type == PREFOS_CONE_POSITIVE_SEMIDEFINITE &&
                 presolver->settings.psd_face_reduction)
        {
            status = reduce_psd_block(problem, first_row, source, target, keep,
                                      &emit[block], &counts);
            if (status != PREFOS_STATUS_OK) goto cleanup;
        }
        else if (source->type == PREFOS_CONE_EXPONENTIAL &&
                 presolver->settings.exponential_face_reduction)
            reduce_exponential_block(problem, first_row, target, keep, &counts);
        else if (source->type == PREFOS_CONE_POWER &&
                 presolver->settings.power_face_reduction)
            reduce_power_block(problem, first_row, target, keep, &counts);
        first_row += source->dimension;
    }
    if (first_row != problem->affine_cone_matrix.rows)
    {
        status = PREFOS_STATUS_NUMERICAL_ERROR;
        goto cleanup;
    }
    if (counts.removed_coordinates == 0) goto cleanup;
    for (first_row = 0; first_row < problem->affine_cone_matrix.rows; ++first_row)
        if (keep[first_row]) ++new_rows;
    for (block = 0; block < problem->n_affine_cones; ++block)
        if (emit[block]) ++new_blocks;
    for (first_row = 0; first_row < presolver->n_pre_face_affine_rows; ++first_row)
        presolver->affine_pre_to_reduced_rows[first_row] = -1;
    status = rebuild_affine_model(presolver, keep, emit, planned, new_rows,
                                  new_blocks, &counts);

cleanup:
    free(keep);
    free(emit);
    free(planned);
    return status;
}
