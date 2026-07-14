/*
 * Copyright 2026 Hongpei Li
 * SPDX-License-Identifier: Apache-2.0
 */

#include "PREFOS_SDPStructureAnalysis.h"

static size_t find_root(size_t *parent, size_t node)
{
    size_t root = node;
    while (parent[root] != root) root = parent[root];
    while (parent[node] != node)
    {
        size_t next = parent[node];
        parent[node] = root;
        node = next;
    }
    return root;
}

static void merge_components(size_t *parent, size_t *component_size,
                             size_t left, size_t right)
{
    size_t left_root = find_root(parent, left);
    size_t right_root = find_root(parent, right);
    if (left_root == right_root) return;
    if (component_size[left_root] < component_size[right_root])
    {
        size_t temporary = left_root;
        left_root = right_root;
        right_root = temporary;
    }
    parent[right_root] = left_root;
    component_size[left_root] += component_size[right_root];
}

static int affine_row_is_active(const PreFOSPresolvedProblem *problem, size_t row)
{
    int p;
    if (problem->affine_cone_offset[row] != 0.0) return 1;
    for (p = problem->affine_cone_matrix.row_pointers[row];
         p < problem->affine_cone_matrix.row_pointers[row + 1]; ++p)
        if (problem->affine_cone_matrix.values[p] != 0.0) return 1;
    return 0;
}

static void initialize_components(size_t *parent, size_t *component_size,
                                  size_t order)
{
    size_t i;
    for (i = 0; i < order; ++i)
    {
        parent[i] = i;
        component_size[i] = 1;
    }
}

static void build_components(const PreFOSPresolvedProblem *problem, size_t row_start,
                             size_t order, size_t *parent,
                             size_t *component_size)
{
    size_t matrix_row, matrix_column, coordinate = 0;
    initialize_components(parent, component_size, order);
    for (matrix_row = 0; matrix_row < order; ++matrix_row)
        for (matrix_column = 0; matrix_column <= matrix_row;
             ++matrix_column, ++coordinate)
            if (matrix_row != matrix_column &&
                affine_row_is_active(problem, row_start + coordinate))
                merge_components(parent, component_size, matrix_row,
                                 matrix_column);
}

static PreFOSStatus analyze_psd_block(
    const PreFOSPresolvedProblem *problem, size_t block_index, size_t row_start,
    const PreFOSAffineConeBlock *block, size_t *parent, size_t *component_size,
    size_t *column_count, unsigned char *column_has_offdiagonal,
    int *touched_columns, PreFOSPSDStructureAnalysis *analysis)
{
    size_t matrix_row, matrix_column, coordinate = 0, n_touched = 0;
    size_t component_count = 0, scalar_components = 0;
    size_t largest_component = 0, decomposed_dimension = 0;
    size_t position;

    memset(analysis, 0, sizeof(*analysis));
    analysis->affine_cone_index = block_index;
    analysis->row_start = row_start;
    analysis->matrix_order = block->matrix_order;
    analysis->dimension = block->dimension;
    analysis->affine_nnz =
        (size_t) (problem->affine_cone_matrix
                      .row_pointers[row_start + block->dimension] -
                  problem->affine_cone_matrix.row_pointers[row_start]);

    initialize_components(parent, component_size, block->matrix_order);
    for (matrix_row = 0; matrix_row < block->matrix_order; ++matrix_row)
    {
        for (matrix_column = 0; matrix_column <= matrix_row;
             ++matrix_column, ++coordinate)
        {
            size_t affine_row = row_start + coordinate;
            int active = problem->affine_cone_offset[affine_row] != 0.0;
            int p;
            for (p = problem->affine_cone_matrix.row_pointers[affine_row];
                 p < problem->affine_cone_matrix.row_pointers[affine_row + 1];
                 ++p)
            {
                int column = problem->affine_cone_matrix.column_indices[p];
                if (problem->affine_cone_matrix.values[p] == 0.0) continue;
                active = 1;
                if (column_count[column] == 0)
                    touched_columns[n_touched++] = column;
                ++column_count[column];
                if (matrix_row != matrix_column)
                    column_has_offdiagonal[column] = 1;
            }
            if (!active) continue;
            if (matrix_row == matrix_column)
                ++analysis->active_diagonal_coordinates;
            else
            {
                ++analysis->active_offdiagonal_coordinates;
                merge_components(parent, component_size, matrix_row,
                                 matrix_column);
            }
        }
    }

    for (matrix_row = 0; matrix_row < block->matrix_order; ++matrix_row)
    {
        size_t root = find_root(parent, matrix_row);
        if (root != matrix_row) continue;
        ++component_count;
        if (component_size[root] == 1) ++scalar_components;
        decomposed_dimension +=
            component_size[root] * (component_size[root] + 1) / 2;
        if (component_size[root] > largest_component)
            largest_component = component_size[root];
    }
    analysis->connected_components = component_count;
    analysis->scalar_components = scalar_components;
    analysis->emitted_cone_blocks =
        component_count - scalar_components + (scalar_components > 0 ? 1 : 0);
    analysis->largest_component_order = largest_component;
    analysis->decomposed_dimension = decomposed_dimension;
    analysis->exactly_block_diagonal = component_count > 1;
    analysis->coefficient_columns = n_touched;
    for (position = 0; position < n_touched; ++position)
    {
        int column = touched_columns[position];
        if (!column_has_offdiagonal[column])
        {
            ++analysis->diagonal_coefficient_columns;
            if (column_count[column] == 1)
                ++analysis->single_diagonal_coefficient_columns;
        }
        column_count[column] = 0;
        column_has_offdiagonal[column] = 0;
    }
    return PREFOS_STATUS_OK;
}

static PreFOSStatus copy_affine_row(const PreFOSPresolvedProblem *source,
                                 size_t source_row, PreFOSCsrMatrix *target,
                                 double *target_offset, size_t *target_row,
                                 int *target_nnz, int *old_to_new)
{
    int p;
    if (*target_row >= target->rows || *target_nnz < 0)
        return PREFOS_STATUS_NUMERICAL_ERROR;
    if (*target_row > (size_t) INT_MAX) return PREFOS_STATUS_OUT_OF_MEMORY;
    target->row_pointers[*target_row] = *target_nnz;
    target_offset[*target_row] = source->affine_cone_offset[source_row];
    old_to_new[source_row] = (int) *target_row;
    for (p = source->affine_cone_matrix.row_pointers[source_row];
         p < source->affine_cone_matrix.row_pointers[source_row + 1]; ++p)
    {
        target->values[*target_nnz] = source->affine_cone_matrix.values[p];
        target->column_indices[*target_nnz] =
            source->affine_cone_matrix.column_indices[p];
        ++(*target_nnz);
    }
    ++(*target_row);
    return PREFOS_STATUS_OK;
}

static PreFOSStatus rebuild_decomposed_model(PreFOSPresolver *presolver,
                                          size_t maximum_order)
{
    PreFOSPresolvedProblem *problem = &presolver->reduced;
    PreFOSCsrMatrix matrix;
    PreFOSAffineConeBlock *blocks = NULL;
    double *offset = NULL;
    int *old_to_new = NULL;
    size_t *parent = NULL, *component_size = NULL;
    unsigned char *component_emitted = NULL;
    size_t block_index, analysis_index = 0, row_start = 0;
    size_t new_rows = problem->affine_cone_matrix.rows;
    size_t new_blocks = problem->n_affine_cones;
    size_t row_write = 0, block_write = 0;
    int nnz_write = 0;
    PreFOSStatus status = PREFOS_STATUS_OK;

    memset(&matrix, 0, sizeof(matrix));
    for (analysis_index = 0;
         analysis_index < presolver->n_psd_structure_analyses; ++analysis_index)
    {
        const PreFOSPSDStructureAnalysis *analysis =
            &presolver->psd_structure_analyses[analysis_index];
        if (!analysis->exactly_block_diagonal) continue;
        if (analysis->decomposed_dimension > analysis->dimension ||
            analysis->connected_components == 0)
            return PREFOS_STATUS_NUMERICAL_ERROR;
        new_rows -= analysis->dimension - analysis->decomposed_dimension;
        if (analysis->emitted_cone_blocks == 0 ||
            analysis->emitted_cone_blocks - 1 > SIZE_MAX - new_blocks)
            return PREFOS_STATUS_OUT_OF_MEMORY;
        new_blocks += analysis->emitted_cone_blocks - 1;
    }

    matrix.rows = new_rows;
    matrix.cols = problem->n;
    matrix.row_pointers = (int *) calloc(new_rows + 1, sizeof(int));
    matrix.values = (double *) prefos_internal_alloc_array(
        problem->affine_cone_matrix.nnz, sizeof(double));
    matrix.column_indices = (int *) prefos_internal_alloc_array(
        problem->affine_cone_matrix.nnz, sizeof(int));
    offset = (double *) prefos_internal_alloc_array(new_rows, sizeof(double));
    blocks = (PreFOSAffineConeBlock *) prefos_internal_alloc_array(
        new_blocks, sizeof(PreFOSAffineConeBlock));
    old_to_new = (int *) prefos_internal_alloc_array(
        problem->affine_cone_matrix.rows, sizeof(int));
    parent = (size_t *) prefos_internal_alloc_array(maximum_order, sizeof(size_t));
    component_size =
        (size_t *) prefos_internal_alloc_array(maximum_order, sizeof(size_t));
    component_emitted =
        (unsigned char *) calloc(maximum_order, sizeof(unsigned char));
    if (!matrix.row_pointers ||
        (problem->affine_cone_matrix.nnz > 0 &&
         (!matrix.values || !matrix.column_indices)) ||
        (new_rows > 0 && !offset) || (new_blocks > 0 && !blocks) ||
        (problem->affine_cone_matrix.rows > 0 && !old_to_new) ||
        (maximum_order > 0 &&
         (!parent || !component_size || !component_emitted)))
    {
        status = PREFOS_STATUS_OUT_OF_MEMORY;
        goto cleanup;
    }
    for (row_start = 0; row_start < problem->affine_cone_matrix.rows; ++row_start)
        old_to_new[row_start] = -1;

    row_start = 0;
    analysis_index = 0;
    for (block_index = 0; block_index < problem->n_affine_cones; ++block_index)
    {
        const PreFOSAffineConeBlock *source_block =
            &problem->affine_cones[block_index];
        const PreFOSPSDStructureAnalysis *analysis = NULL;
        size_t coordinate;
        if (source_block->type == PREFOS_CONE_POSITIVE_SEMIDEFINITE)
        {
            if (analysis_index >= presolver->n_psd_structure_analyses)
            {
                status = PREFOS_STATUS_NUMERICAL_ERROR;
                goto cleanup;
            }
            analysis = &presolver->psd_structure_analyses[analysis_index++];
        }
        if (!analysis || !analysis->exactly_block_diagonal)
        {
            blocks[block_write++] = *source_block;
            for (coordinate = 0; coordinate < source_block->dimension;
                 ++coordinate)
            {
                status = copy_affine_row(problem, row_start + coordinate, &matrix,
                                         offset, &row_write, &nnz_write,
                                         old_to_new);
                if (status != PREFOS_STATUS_OK) goto cleanup;
            }
        }
        else
        {
            size_t seed, matrix_row, matrix_column;
            build_components(problem, row_start, source_block->matrix_order,
                             parent, component_size);
            memset(component_emitted, 0,
                   source_block->matrix_order * sizeof(unsigned char));
            for (seed = 0; seed < source_block->matrix_order; ++seed)
            {
                size_t root = find_root(parent, seed);
                size_t component_order = component_size[root];
                if (component_emitted[root]) continue;
                component_emitted[root] = 1;
                if (component_order == 1) continue;
                blocks[block_write++] = (PreFOSAffineConeBlock){
                    PREFOS_CONE_POSITIVE_SEMIDEFINITE,
                    component_order * (component_order + 1) / 2,
                    component_order, 0.0};
                for (matrix_row = 0; matrix_row < source_block->matrix_order;
                     ++matrix_row)
                {
                    if (find_root(parent, matrix_row) != root) continue;
                    for (matrix_column = 0; matrix_column <= matrix_row;
                         ++matrix_column)
                    {
                        size_t source_coordinate =
                            matrix_row * (matrix_row + 1) / 2 + matrix_column;
                        if (find_root(parent, matrix_column) != root) continue;
                        status = copy_affine_row(
                            problem, row_start + source_coordinate, &matrix,
                            offset, &row_write, &nnz_write, old_to_new);
                        if (status != PREFOS_STATUS_OK) goto cleanup;
                    }
                }
            }
            if (analysis->scalar_components > 0)
            {
                blocks[block_write++] = (PreFOSAffineConeBlock){
                    PREFOS_CONE_NONNEGATIVE, analysis->scalar_components, 0, 0.0};
                for (seed = 0; seed < source_block->matrix_order; ++seed)
                {
                    size_t source_coordinate;
                    if (component_size[find_root(parent, seed)] != 1) continue;
                    source_coordinate = seed * (seed + 1) / 2 + seed;
                    status = copy_affine_row(
                        problem, row_start + source_coordinate, &matrix, offset,
                        &row_write, &nnz_write, old_to_new);
                    if (status != PREFOS_STATUS_OK) goto cleanup;
                }
            }
        }
        row_start += source_block->dimension;
    }
    matrix.row_pointers[new_rows] = nnz_write;
    matrix.nnz = (size_t) nnz_write;
    if (row_start != problem->affine_cone_matrix.rows || row_write != new_rows ||
        block_write != new_blocks ||
        analysis_index != presolver->n_psd_structure_analyses)
    {
        status = PREFOS_STATUS_NUMERICAL_ERROR;
        goto cleanup;
    }
    for (row_start = 0; row_start < problem->affine_cone_matrix.rows;
         ++row_start)
        if (old_to_new[row_start] < 0 && affine_row_is_active(problem, row_start))
        {
            status = PREFOS_STATUS_NUMERICAL_ERROR;
            goto cleanup;
        }
    for (row_start = 0; row_start < presolver->n_pre_face_affine_rows;
         ++row_start)
    {
        int current = presolver->affine_pre_to_reduced_rows[row_start];
        if (current < 0) continue;
        if ((size_t) current >= problem->affine_cone_matrix.rows)
        {
            status = PREFOS_STATUS_NUMERICAL_ERROR;
            goto cleanup;
        }
        presolver->affine_pre_to_reduced_rows[row_start] = old_to_new[current];
    }

    prefos_internal_free_csr(&problem->affine_cone_matrix);
    free(problem->affine_cone_offset);
    free(problem->affine_cones);
    problem->affine_cone_matrix = matrix;
    problem->affine_cone_offset = offset;
    problem->affine_cones = blocks;
    problem->n_affine_cones = new_blocks;
    matrix.values = NULL;
    matrix.column_indices = NULL;
    matrix.row_pointers = NULL;
    offset = NULL;
    blocks = NULL;

cleanup:
    prefos_internal_free_csr(&matrix);
    free(offset);
    free(blocks);
    free(old_to_new);
    free(parent);
    free(component_size);
    free(component_emitted);
    return status;
}

PreFOSStatus prefos_internal_analyze_and_decompose_affine_psd(PreFOSPresolver *presolver)
{
    const PreFOSPresolvedProblem *problem = &presolver->reduced;
    size_t block_index, row_start = 0, psd_count = 0, maximum_order = 0;
    size_t analysis_index = 0;
    size_t *parent = NULL, *component_size = NULL, *column_count = NULL;
    unsigned char *column_has_offdiagonal = NULL;
    int *touched_columns = NULL;
    PreFOSStatus status = PREFOS_STATUS_OK;

    if (!presolver->settings.psd_structure_analysis) return PREFOS_STATUS_OK;
    for (block_index = 0; block_index < problem->n_affine_cones; ++block_index)
    {
        const PreFOSAffineConeBlock *block = &problem->affine_cones[block_index];
        if (block->type != PREFOS_CONE_POSITIVE_SEMIDEFINITE) continue;
        ++psd_count;
        if (block->matrix_order > maximum_order)
            maximum_order = block->matrix_order;
    }
    presolver->psd_structure_analyses =
        (PreFOSPSDStructureAnalysis *) prefos_internal_alloc_array(
            psd_count, sizeof(PreFOSPSDStructureAnalysis));
    presolver->n_psd_structure_analyses = psd_count;
    parent = (size_t *) prefos_internal_alloc_array(maximum_order, sizeof(size_t));
    component_size =
        (size_t *) prefos_internal_alloc_array(maximum_order, sizeof(size_t));
    column_count = (size_t *) calloc(problem->n, sizeof(size_t));
    column_has_offdiagonal =
        (unsigned char *) calloc(problem->n, sizeof(unsigned char));
    touched_columns =
        (int *) prefos_internal_alloc_array(problem->n, sizeof(int));
    if ((psd_count > 0 && !presolver->psd_structure_analyses) ||
        (maximum_order > 0 && (!parent || !component_size)) ||
        (problem->n > 0 &&
         (!column_count || !column_has_offdiagonal || !touched_columns)))
    {
        status = PREFOS_STATUS_OUT_OF_MEMORY;
        goto cleanup;
    }

    for (block_index = 0; block_index < problem->n_affine_cones; ++block_index)
    {
        const PreFOSAffineConeBlock *block = &problem->affine_cones[block_index];
        PreFOSPSDStructureAnalysis *analysis;
        if (block->type != PREFOS_CONE_POSITIVE_SEMIDEFINITE)
        {
            row_start += block->dimension;
            continue;
        }
        analysis = &presolver->psd_structure_analyses[analysis_index++];
        status = analyze_psd_block(
            problem, block_index, row_start, block, parent, component_size,
            column_count, column_has_offdiagonal, touched_columns, analysis);
        if (status != PREFOS_STATUS_OK) goto cleanup;
        ++presolver->stats.affine_psd_blocks_analyzed;
        presolver->stats.affine_psd_active_diagonal_coordinates +=
            analysis->active_diagonal_coordinates;
        presolver->stats.affine_psd_active_offdiagonal_coordinates +=
            analysis->active_offdiagonal_coordinates;
        presolver->stats.affine_psd_coefficient_columns +=
            analysis->coefficient_columns;
        presolver->stats.affine_psd_diagonal_coefficient_columns +=
            analysis->diagonal_coefficient_columns;
        presolver->stats.affine_psd_single_diagonal_coefficient_columns +=
            analysis->single_diagonal_coefficient_columns;
        presolver->stats.affine_psd_connected_components +=
            analysis->connected_components;
        if (analysis->largest_component_order >
            presolver->stats.affine_psd_largest_component_order)
            presolver->stats.affine_psd_largest_component_order =
                analysis->largest_component_order;
        if (analysis->exactly_block_diagonal)
        {
            ++presolver->stats.affine_psd_splittable_blocks;
            if (presolver->settings.psd_block_decomposition)
            {
                ++presolver->stats.decomposed_affine_psd_blocks;
                presolver->stats.affine_psd_scalar_components +=
                    analysis->scalar_components;
                presolver->stats.affine_psd_component_blocks +=
                    analysis->emitted_cone_blocks;
                presolver->stats.removed_affine_psd_cross_coordinates +=
                    analysis->dimension - analysis->decomposed_dimension;
            }
        }
        row_start += block->dimension;
    }
    if (row_start != problem->affine_cone_matrix.rows ||
        analysis_index != psd_count)
    {
        status = PREFOS_STATUS_NUMERICAL_ERROR;
        goto cleanup;
    }
    if (presolver->settings.psd_block_decomposition &&
        presolver->stats.decomposed_affine_psd_blocks > 0)
    {
        status = rebuild_decomposed_model(presolver, maximum_order);
        if (status == PREFOS_STATUS_OK)
            presolver->stats.removed_affine_cone_coordinates +=
                presolver->stats.removed_affine_psd_cross_coordinates;
    }

cleanup:
    free(parent);
    free(component_size);
    free(column_count);
    free(column_has_offdiagonal);
    free(touched_columns);
    return status;
}
