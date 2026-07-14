/*
 * Copyright 2026 Hongpei Li
 * SPDX-License-Identifier: Apache-2.0
 */

#include "PREFOS_AffineFaceSubstitutionInternal.h"

static void mark_input_face(PreFOSPresolver *presolver, size_t block,
                            size_t zero_axis)
{
    presolver->input_affine_rsoc_zero_axis[block] =
        (unsigned char) (zero_axis + 1);
}

static PreFOSStatus mark_generated_face(PreFOSPresolver *presolver, size_t cone_index,
                                     size_t zero_axis)
{
    const PreFOSConeBlock *cone = &presolver->original.cones[cone_index];
    PreFOSFacialReductionCertificate *certificate;
    size_t coordinate;
    int survivor = zero_axis == 0 ? 1 : 0;
    if (presolver->generated_affine_rsoc_zero_axis[cone_index])
        return PREFOS_STATUS_OK;
    if (presolver->n_facial_reductions >= presolver->original.n_cones)
        return PREFOS_STATUS_NUMERICAL_ERROR;
    presolver->generated_affine_rsoc_zero_axis[cone_index] =
        (unsigned char) (zero_axis + 1);
    presolver->cone_face_survivors[cone_index] = cone->indices[survivor];
    for (coordinate = 0; coordinate < cone->dimension; ++coordinate)
    {
        int column = cone->indices[coordinate];
        presolver->cone_face_box[column] = 1;
        if ((int) coordinate == survivor)
        {
            presolver->cone_face_box_lower[column] = 0.0;
            presolver->cone_face_box_upper[column] = INFINITY;
        }
        else
        {
            presolver->cone_face_box_lower[column] = 0.0;
            presolver->cone_face_box_upper[column] = 0.0;
        }
    }
    certificate =
        &presolver->facial_reductions[presolver->n_facial_reductions++];
    memset(certificate, 0, sizeof(*certificate));
    certificate->type = zero_axis == 0 ? PREFOS_FACE_RSOC_U_ZERO
                                       : PREFOS_FACE_RSOC_V_ZERO;
    certificate->source_row = presolver->affine_aggregation_source_rows
        [cone->indices[zero_axis]];
    certificate->zero_axis_column = cone->indices[zero_axis];
    certificate->surviving_axis_column = cone->indices[survivor];
    certificate->cone_dimension = cone->dimension;
    certificate->cone_indices = cone->indices;
    return PREFOS_STATUS_OK;
}

static PreFOSStatus process_input_blocks(
    PreFOSPresolver *presolver, const PreFOSColumnStorage *storage,
    const PreFOSColumnStorage *affine_storage,
    const unsigned char *quadratic_column, const unsigned char *factor_column,
    PreFOSExpandedAffineRow *expanded)
{
    const PreFOSProblemData *problem = &presolver->original;
    size_t block, first_row = 0;
    for (block = 0; block < problem->n_affine_cones; ++block)
    {
        const PreFOSAffineConeBlock *cone = &problem->affine_cones[block];
        size_t zero_axis = SIZE_MAX, coordinate;
        size_t accepted_before = presolver->n_affine_face_substitutions;
        PreFOSStatus status;
        if (cone->type != PREFOS_CONE_ROTATED_SECOND_ORDER)
        {
            first_row += cone->dimension;
            continue;
        }
        for (coordinate = 0; coordinate < 2; ++coordinate)
        {
            status = prefos_affine_face_expand_input_row(
                presolver, first_row + coordinate, expanded);
            if (status != PREFOS_STATUS_OK) return status;
            if (expanded->count == 0 && expanded->constant == 0.0)
            {
                zero_axis = coordinate;
                break;
            }
        }
        if (zero_axis == SIZE_MAX)
        {
            first_row += cone->dimension;
            continue;
        }
        for (coordinate = 2; coordinate < cone->dimension; ++coordinate)
        {
            int pivot;
            status = prefos_affine_face_expand_input_row(
                presolver, first_row + coordinate, expanded);
            if (status != PREFOS_STATUS_OK) return status;
            if (expanded->count == 0)
            {
                if (expanded->constant != 0.0)
                    return PREFOS_STATUS_PRIMAL_INFEASIBLE;
                continue;
            }
            pivot = prefos_affine_face_choose_pivot(
                presolver, storage, quadratic_column, factor_column, expanded);
            if (pivot < 0) continue;
            status = prefos_affine_face_append_transformation(
                presolver, storage, affine_storage, expanded,
                first_row + coordinate, pivot);
            if (status != PREFOS_STATUS_OK) return status;
        }
        if (presolver->n_affine_face_substitutions > accepted_before)
            mark_input_face(presolver, block, zero_axis);
        first_row += cone->dimension;
    }
    return first_row == problem->affine_cone_matrix.rows
               ? PREFOS_STATUS_OK
               : PREFOS_STATUS_NUMERICAL_ERROR;
}

static PreFOSStatus process_generated_blocks(
    PreFOSPresolver *presolver, const PreFOSColumnStorage *storage,
    const PreFOSColumnStorage *affine_storage,
    const unsigned char *quadratic_column, const unsigned char *factor_column,
    PreFOSExpandedAffineRow *expanded)
{
    const PreFOSProblemData *problem = &presolver->original;
    size_t cone_index;
    size_t affine_row = problem->affine_cone_matrix.rows;
    for (cone_index = 0; cone_index < problem->n_cones; ++cone_index)
    {
        const PreFOSConeBlock *cone = &problem->cones[cone_index];
        size_t zero_axis = SIZE_MAX, coordinate;
        size_t accepted_before = presolver->n_affine_face_substitutions;
        PreFOSStatus status;
        if (!presolver->converted_affine_cones[cone_index]) continue;
        if (cone->type != PREFOS_CONE_ROTATED_SECOND_ORDER)
        {
            affine_row += cone->dimension;
            continue;
        }
        for (coordinate = 0; coordinate < 2; ++coordinate)
        {
            status = prefos_affine_face_expand_generated_row(
                presolver, cone->indices[coordinate], expanded);
            if (status != PREFOS_STATUS_OK) return status;
            if (expanded->count == 0 && expanded->constant == 0.0)
            {
                zero_axis = coordinate;
                break;
            }
        }
        if (zero_axis == SIZE_MAX)
        {
            affine_row += cone->dimension;
            continue;
        }
        for (coordinate = 2; coordinate < cone->dimension; ++coordinate)
        {
            int pivot;
            status = prefos_affine_face_expand_generated_row(
                presolver, cone->indices[coordinate], expanded);
            if (status != PREFOS_STATUS_OK) return status;
            if (expanded->count == 0)
            {
                if (expanded->constant != 0.0)
                    return PREFOS_STATUS_PRIMAL_INFEASIBLE;
                continue;
            }
            pivot = prefos_affine_face_choose_pivot(
                presolver, storage, quadratic_column, factor_column, expanded);
            if (pivot < 0) continue;
            status = prefos_affine_face_append_transformation(
                presolver, storage, affine_storage, expanded,
                affine_row + coordinate, pivot);
            if (status != PREFOS_STATUS_OK) return status;
        }
        if (presolver->n_affine_face_substitutions > accepted_before)
        {
            status = mark_generated_face(presolver, cone_index, zero_axis);
            if (status != PREFOS_STATUS_OK) return status;
        }
        affine_row += cone->dimension;
    }
    return PREFOS_STATUS_OK;
}

PreFOSStatus prefos_internal_substitute_affine_face_equalities(PreFOSPresolver *presolver)
{
    const PreFOSProblemData *problem = &presolver->original;
    PreFOSColumnStorage storage, affine_storage;
    PreFOSExpandedAffineRow expanded;
    unsigned char *quadratic_column = NULL, *factor_column = NULL;
    size_t row;
    int has_rsoc = 0;
    PreFOSStatus status = PREFOS_STATUS_OK;
    memset(&storage, 0, sizeof(storage));
    memset(&affine_storage, 0, sizeof(affine_storage));
    memset(&expanded, 0, sizeof(expanded));
    if (!presolver->settings.free_column_substitution ||
        !presolver->settings.rsoc_face_reduction ||
        (problem->n_affine_cones == 0 &&
         presolver->stats.generated_affine_cone_blocks == 0))
        return PREFOS_STATUS_OK;
    for (row = 0; row < problem->n_affine_cones; ++row)
        if (problem->affine_cones[row].type ==
            PREFOS_CONE_ROTATED_SECOND_ORDER)
        {
            has_rsoc = 1;
            break;
        }
    if (!has_rsoc)
        for (row = 0; row < problem->n_cones; ++row)
            if (presolver->converted_affine_cones[row] &&
                problem->cones[row].type == PREFOS_CONE_ROTATED_SECOND_ORDER)
            {
                has_rsoc = 1;
                break;
            }
    if (!has_rsoc) return PREFOS_STATUS_OK;

    status = prefos_affine_face_build_column_storage(&problem->A, &storage);
    if (status != PREFOS_STATUS_OK) goto cleanup;
    status =
        prefos_affine_face_build_affine_column_storage(presolver, &affine_storage);
    if (status != PREFOS_STATUS_OK) goto cleanup;
    expanded.values = (double *) calloc(problem->n, sizeof(double));
    expanded.marks = (int *) prefos_internal_alloc_array(problem->n, sizeof(int));
    expanded.touched = (int *) prefos_internal_alloc_array(problem->n, sizeof(int));
    quadratic_column = (unsigned char *) calloc(problem->n, sizeof(unsigned char));
    factor_column = (unsigned char *) calloc(problem->n, sizeof(unsigned char));
    if (problem->n > 0 &&
        (!expanded.values || !expanded.marks || !expanded.touched ||
         !quadratic_column || !factor_column))
    {
        status = PREFOS_STATUS_OUT_OF_MEMORY;
        goto cleanup;
    }
    for (row = 0; row < problem->n; ++row) expanded.marks[row] = -1;
    for (row = 0; row < problem->Q.rows; ++row)
    {
        int p;
        for (p = problem->Q.row_pointers[row]; p < problem->Q.row_pointers[row + 1];
             ++p)
        {
            if (problem->Q.values[p] == 0.0) continue;
            quadratic_column[row] = 1;
            quadratic_column[problem->Q.column_indices[p]] = 1;
        }
    }
    for (row = 0; row < problem->R.rows; ++row)
    {
        int p;
        for (p = problem->R.row_pointers[row]; p < problem->R.row_pointers[row + 1];
             ++p)
            if (problem->R.values[p] != 0.0)
                factor_column[problem->R.column_indices[p]] = 1;
    }
    status = process_input_blocks(presolver, &storage, &affine_storage,
                                  quadratic_column, factor_column, &expanded);
    if (status != PREFOS_STATUS_OK) goto cleanup;
    status = process_generated_blocks(presolver, &storage, &affine_storage,
                                      quadratic_column, factor_column, &expanded);

cleanup:
    prefos_affine_face_free_column_storage(&storage);
    prefos_affine_face_free_column_storage(&affine_storage);
    free(expanded.values);
    free(expanded.marks);
    free(expanded.touched);
    free(quadratic_column);
    free(factor_column);
    return status;
}
