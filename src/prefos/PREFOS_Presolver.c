/*
 * Copyright 2026 Hongpei Li
 * SPDX-License-Identifier: Apache-2.0
 */

#include "PREFOS_Internal.h"
#include "cones/PREFOS_ExponentialCone.h"
#include "cones/PREFOS_PowerCone.h"
#include "core/PREFOS_AffineConeCompaction.h"
#include "core/PREFOS_AffineConeFaces.h"
#include "core/PREFOS_MatrixCompaction.h"
#include "core/PREFOS_PassManager.h"
#include "core/PREFOS_Timer.h"
#include "explorers/PREFOS_AffineConeAggregation.h"
#include "explorers/PREFOS_AffineFaceSubstitution.h"
#include "explorers/PREFOS_ConePropagation.h"
#include "explorers/PREFOS_ColumnReductionInternal.h"
#include "explorers/PREFOS_ColumnReductions.h"
#include "explorers/PREFOS_CudaBackend.h"
#include "explorers/PREFOS_CudaLinearPropagation.h"
#include "explorers/PREFOS_FreeColumnSubstitution.h"
#include "explorers/PREFOS_LinearPropagation.h"
#include "explorers/PREFOS_ParallelRows.h"
#include "explorers/PREFOS_SDPStructureAnalysis.h"
#include "explorers/PREFOS_TrivialReductions.h"

int prefos_gpu_warmup(void)
{
    return prefos_cuda_linear_propagation_warmup() == PREFOS_CUDA_PROPAGATION_OK;
}

int prefos_gpu_warmup_async(void)
{
    return prefos_cuda_linear_propagation_warmup_async();
}

int prefos_gpu_warmup_ready(void)
{
    return prefos_cuda_linear_propagation_warmup_ready();
}

int prefos_gpu_warmup_wait(void)
{
    return prefos_cuda_linear_propagation_warmup_wait() == PREFOS_CUDA_PROPAGATION_OK;
}

void prefos_gpu_release_cache(void)
{
    prefos_cuda_linear_propagation_release_cache();
}

static PreFOSStatus initialize_working_state(PreFOSPresolver *presolver)
{
    size_t i;
    const PreFOSProblemData *problem = &presolver->original;
    presolver->original_to_reduced =
        (int *) prefos_internal_alloc_array(problem->n, sizeof(int));
    presolver->original_to_reduced_rows =
        (int *) prefos_internal_alloc_array(problem->A.rows, sizeof(int));
    presolver->fixed_values = (double *) calloc(problem->n, sizeof(double));
    presolver->is_fixed =
        (unsigned char *) calloc(problem->n, sizeof(unsigned char));
    presolver->fixed_column_log =
        (int *) prefos_internal_alloc_array(problem->n, sizeof(int));
    presolver->is_substituted =
        (unsigned char *) calloc(problem->n, sizeof(unsigned char));
    presolver->is_parallel_removed =
        (unsigned char *) calloc(problem->n, sizeof(unsigned char));
    presolver->substitution_term_count =
        (size_t *) calloc(problem->n, sizeof(size_t));
    presolver->substitution_incoming_depth =
        (unsigned char *) calloc(problem->n, sizeof(unsigned char));
    presolver->substitution_keeps_source_row =
        (unsigned char *) calloc(problem->n, sizeof(unsigned char));
    presolver->substitution_term_start =
        (size_t *) calloc(problem->n, sizeof(size_t));
    presolver->substitution_source_row =
        (int *) prefos_internal_alloc_array(problem->n, sizeof(int));
    presolver->residual_source_column =
        (int *) prefos_internal_alloc_array(problem->A.rows, sizeof(int));
    presolver->substitution_constant = (double *) calloc(problem->n, sizeof(double));
    presolver->variable_to_box =
        (int *) prefos_internal_alloc_array(problem->n, sizeof(int));
    presolver->working_box_lower =
        (double *) prefos_internal_alloc_array(problem->n_box, sizeof(double));
    presolver->working_box_upper =
        (double *) prefos_internal_alloc_array(problem->n_box, sizeof(double));
    presolver->working_constraint_lower =
        (double *) prefos_internal_alloc_array(problem->A.rows, sizeof(double));
    presolver->working_constraint_upper =
        (double *) prefos_internal_alloc_array(problem->A.rows, sizeof(double));
    presolver->propagation_lower =
        (double *) prefos_internal_alloc_array(problem->n, sizeof(double));
    presolver->propagation_upper =
        (double *) prefos_internal_alloc_array(problem->n, sizeof(double));
    presolver->converted_affine_cones =
        (unsigned char *) calloc(problem->n_cones, sizeof(unsigned char));
    presolver->affine_protected_columns =
        (unsigned char *) calloc(problem->n, sizeof(unsigned char));
    presolver->affine_aggregation_source_rows =
        (int *) prefos_internal_alloc_array(problem->n, sizeof(int));
    presolver->affine_aggregation_pivots =
        (double *) calloc(problem->n, sizeof(double));
    presolver->input_affine_rsoc_zero_axis =
        (unsigned char *) calloc(problem->n_affine_cones, sizeof(unsigned char));
    presolver->generated_affine_rsoc_zero_axis =
        (unsigned char *) calloc(problem->n_cones, sizeof(unsigned char));
    presolver->affine_face_substitution_targets =
        (unsigned char *) calloc(problem->n, sizeof(unsigned char));
    presolver->affine_face_eliminated_columns =
        (unsigned char *) calloc(problem->n, sizeof(unsigned char));
    presolver->remove_rows =
        (unsigned char *) calloc(problem->A.rows, sizeof(unsigned char));
    presolver->removed_row_log =
        (int *) prefos_internal_alloc_array(problem->A.rows, sizeof(int));
    presolver->remove_cones =
        (unsigned char *) calloc(problem->n_cones, sizeof(unsigned char));
    presolver->cone_face_survivors =
        (int *) prefos_internal_alloc_array(problem->n_cones, sizeof(int));
    presolver->cone_face_box =
        (unsigned char *) calloc(problem->n, sizeof(unsigned char));
    presolver->cone_face_box_lower =
        (double *) prefos_internal_alloc_array(problem->n, sizeof(double));
    presolver->cone_face_box_upper =
        (double *) prefos_internal_alloc_array(problem->n, sizeof(double));
    presolver->cone_collapse_source_rows =
        (int *) prefos_internal_alloc_array(problem->n, sizeof(int));
    presolver->psd_face_reductions = (PreFOSPSDFaceReduction *) calloc(
        problem->n_cones, sizeof(PreFOSPSDFaceReduction));
    presolver->facial_reductions = (PreFOSFacialReductionCertificate *) calloc(
        problem->n_cones, sizeof(PreFOSFacialReductionCertificate));
    if (problem->n > 0 &&
        (!presolver->original_to_reduced || !presolver->fixed_values ||
         !presolver->is_fixed || !presolver->fixed_column_log ||
         !presolver->is_substituted ||
         !presolver->is_parallel_removed ||
         !presolver->substitution_term_count ||
         !presolver->substitution_incoming_depth ||
         !presolver->substitution_keeps_source_row ||
         !presolver->substitution_term_start ||
         !presolver->substitution_source_row ||
         !presolver->substitution_constant ||
         !presolver->variable_to_box || !presolver->propagation_lower ||
         !presolver->propagation_upper || !presolver->affine_protected_columns ||
         !presolver->affine_aggregation_source_rows ||
         !presolver->affine_aggregation_pivots ||
         !presolver->affine_face_substitution_targets ||
         !presolver->affine_face_eliminated_columns ||
         !presolver->cone_face_box ||
         !presolver->cone_face_box_lower || !presolver->cone_face_box_upper ||
         !presolver->cone_collapse_source_rows))
    {
        return PREFOS_STATUS_OUT_OF_MEMORY;
    }
    if ((problem->A.rows > 0 && (!presolver->original_to_reduced_rows ||
                                 !presolver->working_constraint_lower ||
                                 !presolver->working_constraint_upper ||
                                 !presolver->residual_source_column)) ||
        (problem->n_box > 0 &&
         (!presolver->working_box_lower || !presolver->working_box_upper)) ||
        (problem->A.rows > 0 &&
         (!presolver->remove_rows || !presolver->removed_row_log)) ||
        (problem->n_cones > 0 &&
         (!presolver->converted_affine_cones || !presolver->remove_cones ||
          !presolver->generated_affine_rsoc_zero_axis ||
          !presolver->cone_face_survivors ||
          !presolver->psd_face_reductions || !presolver->facial_reductions)))
    {
        return PREFOS_STATUS_OUT_OF_MEMORY;
    }
    if (problem->n_affine_cones > 0 &&
        !presolver->input_affine_rsoc_zero_axis)
        return PREFOS_STATUS_OUT_OF_MEMORY;

    for (i = 0; i < problem->n; ++i)
    {
        presolver->variable_to_box[i] = -1;
        presolver->substitution_source_row[i] = -1;
        presolver->propagation_lower[i] = -INFINITY;
        presolver->propagation_upper[i] = INFINITY;
        presolver->cone_face_box_lower[i] = -INFINITY;
        presolver->cone_face_box_upper[i] = INFINITY;
        presolver->cone_collapse_source_rows[i] = -1;
        presolver->affine_aggregation_source_rows[i] = -1;
    }
    for (i = 0; i < problem->A.rows; ++i)
    {
        presolver->original_to_reduced_rows[i] = -1;
        presolver->residual_source_column[i] = -1;
        presolver->working_constraint_lower[i] = problem->constraint_lower[i];
        presolver->working_constraint_upper[i] = problem->constraint_upper[i];
    }
    for (i = 0; i < problem->n_cones; ++i) presolver->cone_face_survivors[i] = -1;
    for (i = 0; i < problem->affine_cone_matrix.nnz; ++i)
        presolver->affine_protected_columns
            [problem->affine_cone_matrix.column_indices[i]] = 1;
    for (i = 0; i < problem->n_box; ++i)
    {
        int index = problem->box_indices[i];
        presolver->variable_to_box[index] = (int) i;
        presolver->working_box_lower[i] = problem->box_lower[i];
        presolver->working_box_upper[i] = problem->box_upper[i];
        presolver->propagation_lower[index] = problem->box_lower[i];
        presolver->propagation_upper[index] = problem->box_upper[i];
    }
    for (i = 0; i < problem->n_cones; ++i)
    {
        const PreFOSConeBlock *cone = &problem->cones[i];
        size_t j;
        if (cone->type == PREFOS_CONE_NONNEGATIVE)
        {
            for (j = 0; j < cone->dimension; ++j)
                presolver->propagation_lower[cone->indices[j]] = 0.0;
        }
        else if (cone->type == PREFOS_CONE_SECOND_ORDER)
        {
            presolver->propagation_lower[cone->indices[0]] = 0.0;
        }
        else if (cone->type == PREFOS_CONE_ROTATED_SECOND_ORDER)
        {
            presolver->propagation_lower[cone->indices[0]] = 0.0;
            presolver->propagation_lower[cone->indices[1]] = 0.0;
        }
        else if (cone->type == PREFOS_CONE_POSITIVE_SEMIDEFINITE)
        {
            for (j = 0; j < cone->matrix_order; ++j)
            {
                size_t packed_diagonal = j * (j + 1) / 2 + j;
                presolver->propagation_lower[cone->indices[packed_diagonal]] = 0.0;
            }
        }
        else if (cone->type == PREFOS_CONE_EXPONENTIAL)
        {
            for (j = 0; j < cone->dimension; ++j)
                if (prefos_internal_exponential_coordinate_is_nonnegative(j))
                    presolver->propagation_lower[cone->indices[j]] = 0.0;
        }
        else if (cone->type == PREFOS_CONE_POWER)
        {
            for (j = 0; j < cone->dimension; ++j)
                if (prefos_internal_power_coordinate_is_nonnegative(j))
                    presolver->propagation_lower[cone->indices[j]] = 0.0;
        }
    }
    return PREFOS_STATUS_OK;
}

PreFOSStatus prefos_internal_append_bound_record(PreFOSPresolver *presolver, int row,
                                           int column, double old_bound,
                                           double new_bound, int is_lower)
{
    PresolveBoundChangeRecord record;
    if (old_bound == new_bound) return PREFOS_STATUS_OK;
    record = (PresolveBoundChangeRecord){
        row, column, old_bound, new_bound, 0.0, (uint8_t) is_lower, 1, 0};
    return presolve_transformation_log_append_bound_change(
               &presolver->transformations, &record, NULL)
               ? PREFOS_STATUS_OK
               : PREFOS_STATUS_OUT_OF_MEMORY;
}

static void build_variable_map(PreFOSPresolver *presolver)
{
    size_t i;
    int next = 0;
    for (i = 0; i < presolver->original.n; ++i)
    {
        if (presolver->is_fixed[i] || presolver->is_substituted[i] ||
            presolver->is_parallel_removed[i])
            presolver->original_to_reduced[i] = -1;
        else
            presolver->original_to_reduced[i] = next++;
    }
}

typedef struct
{
    size_t transformation_events;
    size_t removed_rows;
    size_t changed_bounds;
    size_t cone_reductions;
} PreFOSFastTriggerSignature;

static PreFOSFastTriggerSignature capture_fast_trigger_signature(
    const PreFOSPresolver *presolver)
{
    const PreFOSStats *stats = &presolver->stats;
    PreFOSFastTriggerSignature signature;
    signature.transformation_events = presolver->transformations.n_events;
    signature.removed_rows =
        stats->removed_redundant_rows +
        stats->removed_singleton_rows +
        stats->removed_empty_rows +
        stats->removed_affine_cone_coordinates +
        stats->removed_affine_cone_blocks;
    signature.changed_bounds =
        stats->tightened_box_bounds +
        stats->propagated_box_bounds +
        stats->tightened_cone_envelopes +
        stats->tightened_affine_cone_envelopes +
        stats->tightened_affine_variable_envelopes +
        stats->materialized_affine_cone_box_bounds +
        stats->fixed_affine_face_variables +
        stats->removed_redundant_row_lower_sides +
        stats->removed_redundant_row_upper_sides +
        stats->removed_redundant_box_lower_bounds +
        stats->removed_redundant_box_upper_bounds;
    signature.cone_reductions =
        stats->fixed_cone_variables +
        stats->collapsed_cones +
        stats->reduced_rsoc_faces +
        stats->reduced_psd_faces +
        stats->reduced_exponential_faces +
        stats->reduced_power_faces +
        stats->reduced_affine_rsoc_faces +
        stats->reduced_affine_psd_faces +
        stats->reduced_affine_exponential_faces +
        stats->reduced_affine_power_faces;
    return signature;
}

static int fast_trigger_signature_changed(
    PreFOSFastTriggerSignature before,
    PreFOSFastTriggerSignature after)
{
    return before.transformation_events != after.transformation_events ||
           before.removed_rows != after.removed_rows ||
           before.changed_bounds != after.changed_bounds ||
           before.cone_reductions != after.cone_reductions;
}

static PreFOSStatus multiply_fixed_quadratic(const PreFOSPresolver *presolver,
                                          double *product)
{
    const PreFOSProblemData *problem = &presolver->original;
    const PreFOSCsrMatrix *Q = &problem->Q;
    const PreFOSCsrMatrix *R = &problem->R;
    size_t row;
    double *r_alpha;

    if (presolver->stats.fixed_box_variables == 0 &&
        presolver->stats.fixed_cone_variables == 0)
        return PREFOS_STATUS_OK;
    r_alpha = (double *) calloc(R->rows, sizeof(double));

    if (Q->row_pointers)
    {
        for (row = 0; row < Q->rows; ++row)
        {
            int p;
            for (p = Q->row_pointers[row]; p < Q->row_pointers[row + 1]; ++p)
            {
                int column = Q->column_indices[p];
                double value = Q->values[p];
                if (!prefos_internal_safe_add_product(&product[row], value,
                                                   presolver->fixed_values[column]))
                {
                    free(r_alpha);
                    return PREFOS_STATUS_NUMERICAL_ERROR;
                }
                if (problem->q_storage != PREFOS_Q_FULL && column != (int) row)
                {
                    if (!prefos_internal_safe_add_product(&product[column], value,
                                                       presolver->fixed_values[row]))
                    {
                        free(r_alpha);
                        return PREFOS_STATUS_NUMERICAL_ERROR;
                    }
                }
            }
        }
    }

    if (R->rows > 0 && !r_alpha) return PREFOS_STATUS_OUT_OF_MEMORY;
    for (row = 0; row < R->rows; ++row)
    {
        int p;
        for (p = R->row_pointers[row]; p < R->row_pointers[row + 1]; ++p)
        {
            if (!prefos_internal_safe_add_product(
                    &r_alpha[row], R->values[p],
                    presolver->fixed_values[R->column_indices[p]]))
            {
                free(r_alpha);
                return PREFOS_STATUS_NUMERICAL_ERROR;
            }
        }
        if (!prefos_internal_safe_product(r_alpha[row], problem->D[row], &r_alpha[row]))
        {
            free(r_alpha);
            return PREFOS_STATUS_NUMERICAL_ERROR;
        }
    }
    for (row = 0; row < R->rows; ++row)
    {
        int p;
        for (p = R->row_pointers[row]; p < R->row_pointers[row + 1]; ++p)
        {
            if (!prefos_internal_safe_add_product(&product[R->column_indices[p]],
                                               R->values[p], r_alpha[row]))
            {
                free(r_alpha);
                return PREFOS_STATUS_NUMERICAL_ERROR;
            }
        }
    }
    free(r_alpha);
    return PREFOS_STATUS_OK;
}

static PreFOSStatus compact_general_matrix(const PreFOSCsrMatrix *source,
                                        const int *column_map,
                                        size_t reduced_columns, PreFOSCsrMatrix *target)
{
    size_t row, nnz = 0;
    int write = 0;
    for (row = 0; row < source->nnz; ++row)
        if (column_map[source->column_indices[row]] >= 0 &&
            source->values[row] != 0.0)
            ++nnz;

    memset(target, 0, sizeof(*target));
    target->rows = source->rows;
    target->cols = reduced_columns;
    target->nnz = nnz;
    target->row_pointers = (int *) calloc(source->rows + 1, sizeof(int));
    target->values = (double *) prefos_internal_alloc_array(nnz, sizeof(double));
    target->column_indices = (int *) prefos_internal_alloc_array(nnz, sizeof(int));
    if (!target->row_pointers ||
        (nnz > 0 && (!target->values || !target->column_indices)))
    {
        prefos_internal_free_csr(target);
        return PREFOS_STATUS_OUT_OF_MEMORY;
    }

    for (row = 0; row < source->rows; ++row)
    {
        int p;
        target->row_pointers[row] = write;
        for (p = source->row_pointers[row]; p < source->row_pointers[row + 1]; ++p)
        {
            int mapped = column_map[source->column_indices[p]];
            if (mapped >= 0 && source->values[p] != 0.0)
            {
                target->values[write] = source->values[p];
                target->column_indices[write] = mapped;
                ++write;
            }
        }
    }
    target->row_pointers[source->rows] = write;
    return PREFOS_STATUS_OK;
}

static PreFOSStatus compact_q(const PreFOSPresolver *presolver, PreFOSCsrMatrix *target)
{
    const PreFOSCsrMatrix *source = &presolver->original.Q;
    size_t row, nnz = 0;
    int write = 0;
    size_t reduced_n = presolver->reduced.n;

    for (row = 0; row < source->rows; ++row)
    {
        int p;
        if (presolver->original_to_reduced[row] < 0) continue;
        for (p = source->row_pointers[row]; p < source->row_pointers[row + 1]; ++p)
        {
            if (presolver->original_to_reduced[source->column_indices[p]] >= 0 &&
                source->values[p] != 0.0)
                ++nnz;
        }
    }

    memset(target, 0, sizeof(*target));
    target->rows = reduced_n;
    target->cols = reduced_n;
    target->nnz = nnz;
    target->row_pointers = (int *) calloc(reduced_n + 1, sizeof(int));
    target->values = (double *) prefos_internal_alloc_array(nnz, sizeof(double));
    target->column_indices = (int *) prefos_internal_alloc_array(nnz, sizeof(int));
    if (!target->row_pointers ||
        (nnz > 0 && (!target->values || !target->column_indices)))
    {
        prefos_internal_free_csr(target);
        return PREFOS_STATUS_OUT_OF_MEMORY;
    }

    for (row = 0; row < source->rows; ++row)
    {
        int p;
        int mapped_row = presolver->original_to_reduced[row];
        if (mapped_row < 0) continue;
        target->row_pointers[mapped_row] = write;
        for (p = source->row_pointers[row]; p < source->row_pointers[row + 1]; ++p)
        {
            int mapped_column =
                presolver->original_to_reduced[source->column_indices[p]];
            if (mapped_column >= 0 && source->values[p] != 0.0)
            {
                target->values[write] = source->values[p];
                target->column_indices[write] = mapped_column;
                ++write;
            }
        }
    }
    target->row_pointers[reduced_n] = write;
    return PREFOS_STATUS_OK;
}

static PreFOSStatus build_reduced_domains(PreFOSPresolver *presolver)
{
    const PreFOSProblemData *source = &presolver->original;
    PreFOSPresolvedProblem *target = &presolver->reduced;
    size_t i, k, box_write = 0, cone_write = 0;

    target->n_box = 0;
    for (i = 0; i < source->n_box; ++i)
        if (presolver->original_to_reduced[source->box_indices[i]] >= 0)
            ++target->n_box;
    for (i = 0; i < source->n; ++i)
        if (presolver->cone_face_box[i] && presolver->original_to_reduced[i] >= 0)
            ++target->n_box;
    target->box_indices =
        (int *) prefos_internal_alloc_array(target->n_box, sizeof(int));
    target->box_lower =
        (double *) prefos_internal_alloc_array(target->n_box, sizeof(double));
    target->box_upper =
        (double *) prefos_internal_alloc_array(target->n_box, sizeof(double));
    if (target->n_box > 0 &&
        (!target->box_indices || !target->box_lower || !target->box_upper))
        return PREFOS_STATUS_OUT_OF_MEMORY;

    for (i = 0; i < source->n_box; ++i)
    {
        int mapped = presolver->original_to_reduced[source->box_indices[i]];
        if (mapped < 0) continue;
        target->box_indices[box_write] = mapped;
        target->box_lower[box_write] = presolver->working_box_lower[i];
        target->box_upper[box_write] = presolver->working_box_upper[i];
        ++box_write;
    }
    for (i = 0; i < source->n; ++i)
    {
        int mapped;
        if (!presolver->cone_face_box[i]) continue;
        mapped = presolver->original_to_reduced[i];
        if (mapped < 0)
        {
            if (presolver->affine_aggregation_source_rows[i] >= 0) continue;
            return PREFOS_STATUS_NUMERICAL_ERROR;
        }
        target->box_indices[box_write] = mapped;
        target->box_lower[box_write] = presolver->cone_face_box_lower[i];
        target->box_upper[box_write] = presolver->cone_face_box_upper[i];
        ++box_write;
    }
    if (box_write != target->n_box) return PREFOS_STATUS_NUMERICAL_ERROR;

    target->n_cones = 0;
    for (k = 0; k < source->n_cones; ++k)
        if (!presolver->converted_affine_cones[k] &&
            (!presolver->remove_cones[k] ||
             presolver->psd_face_reductions[k].n_removed > 0))
            ++target->n_cones;
    target->cones = (PreFOSConeBlock *) calloc(target->n_cones, sizeof(PreFOSConeBlock));
    if (target->n_cones > 0 && !target->cones) return PREFOS_STATUS_OUT_OF_MEMORY;
    for (k = 0; k < source->n_cones; ++k)
    {
        const PreFOSPSDFaceReduction *face = &presolver->psd_face_reductions[k];
        if (presolver->converted_affine_cones[k]) continue;
        if (face->n_removed > 0)
        {
            const PreFOSConeBlock *original_cone = &source->cones[k];
            PreFOSConeBlock *reduced_cone = &target->cones[cone_write];
            size_t reduced_order = original_cone->matrix_order - face->n_removed;
            size_t reduced_dimension = reduced_order * (reduced_order + 1) / 2;
            size_t row, column, index_write = 0;
            if (reduced_order == 0) return PREFOS_STATUS_NUMERICAL_ERROR;
            reduced_cone->type = PREFOS_CONE_POSITIVE_SEMIDEFINITE;
            reduced_cone->dimension = reduced_dimension;
            reduced_cone->matrix_order = reduced_order;
            reduced_cone->indices =
                (int *) prefos_internal_alloc_array(reduced_dimension, sizeof(int));
            if (!reduced_cone->indices) return PREFOS_STATUS_OUT_OF_MEMORY;
            for (row = 0; row < original_cone->matrix_order; ++row)
            {
                if (prefos_internal_psd_matrix_index_is_removed(face, row)) continue;
                for (column = 0; column <= row; ++column)
                {
                    size_t packed;
                    int mapped;
                    if (prefos_internal_psd_matrix_index_is_removed(face, column))
                        continue;
                    packed = row * (row + 1) / 2 + column;
                    mapped =
                        presolver
                            ->original_to_reduced[original_cone->indices[packed]];
                    if (mapped < 0) return PREFOS_STATUS_NUMERICAL_ERROR;
                    reduced_cone->indices[index_write++] = mapped;
                }
            }
            if (index_write != reduced_dimension) return PREFOS_STATUS_NUMERICAL_ERROR;
            ++cone_write;
            continue;
        }
        if (presolver->remove_cones[k]) continue;
        target->cones[cone_write] = source->cones[k];
        target->cones[cone_write].indices = (int *) prefos_internal_alloc_array(
            source->cones[k].dimension, sizeof(int));
        if (!target->cones[cone_write].indices) return PREFOS_STATUS_OUT_OF_MEMORY;
        for (i = 0; i < source->cones[k].dimension; ++i)
        {
            int original_index = source->cones[k].indices[i];
            target->cones[cone_write].indices[i] =
                presolver->original_to_reduced[original_index];
        }
        ++cone_write;
    }
    return PREFOS_STATUS_OK;
}

static PreFOSStatus
accumulate_substituted_linear_objective(const PreFOSPresolver *presolver, int column,
                                        double coefficient, size_t depth,
                                        double *target_c, double *target_offset)
{
    size_t term;

    if (coefficient == 0.0) return PREFOS_STATUS_OK;
    if (column < 0 || (size_t) column >= presolver->original.n ||
        depth > PREFOS_MAX_SUBSTITUTION_DEPTH)
        return PREFOS_STATUS_NUMERICAL_ERROR;
    if (presolver->is_fixed[column])
        return prefos_internal_safe_add_product(target_offset, coefficient,
                                             presolver->fixed_values[column])
                   ? PREFOS_STATUS_OK
                   : PREFOS_STATUS_NUMERICAL_ERROR;
    if (presolver->is_substituted[column])
    {
        size_t start = presolver->substitution_term_start[column];
        size_t count = presolver->substitution_term_count[column];
        if (count == 0 || start > presolver->n_substitution_terms ||
            count > presolver->n_substitution_terms - start ||
            !prefos_internal_safe_add_product(target_offset, coefficient,
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
            status = accumulate_substituted_linear_objective(
                presolver, presolver->substitution_targets[start + term], propagated,
                depth + 1, target_c, target_offset);
            if (status != PREFOS_STATUS_OK) return status;
        }
        return PREFOS_STATUS_OK;
    }
    if (!prefos_internal_safe_add_product(&target_c[column], 1.0, coefficient))
        return PREFOS_STATUS_NUMERICAL_ERROR;
    return PREFOS_STATUS_OK;
}

PreFOSStatus prefos_internal_expand_linear_objective(
    const PreFOSPresolver *presolver, double *objective, double *offset)
{
    size_t column;
    if (!presolver || !offset ||
        (presolver->original.n > 0 && !objective))
        return PREFOS_STATUS_INVALID_ARGUMENT;
    if (presolver->original.n > 0)
        memset(objective, 0, presolver->original.n * sizeof(double));
    *offset = presolver->original.objective_offset;
    for (column = 0; column < presolver->original.n; ++column)
    {
        if (presolver->original.c[column] == 0.0) continue;
        PreFOSStatus status = accumulate_substituted_linear_objective(
            presolver, (int) column, presolver->original.c[column], 0,
            objective, offset);
        if (status != PREFOS_STATUS_OK) return status;
    }
    return PREFOS_STATUS_OK;
}

static PreFOSStatus build_reduced_objective(PreFOSPresolver *presolver)
{
    const PreFOSProblemData *source = &presolver->original;
    PreFOSPresolvedProblem *target = &presolver->reduced;
    double *quadratic_product;
    double *expanded_objective;
    size_t i;

    quadratic_product = (double *) calloc(source->n, sizeof(double));
    expanded_objective = (double *) calloc(source->n, sizeof(double));
    target->c = (double *) calloc(target->n, sizeof(double));
    if ((source->n > 0 && (!quadratic_product || !expanded_objective)) ||
        (target->n > 0 && !target->c))
    {
        free(quadratic_product);
        free(expanded_objective);
        return PREFOS_STATUS_OUT_OF_MEMORY;
    }
    {
        PreFOSStatus status = multiply_fixed_quadratic(presolver, quadratic_product);
        if (status != PREFOS_STATUS_OK)
        {
            free(quadratic_product);
            free(expanded_objective);
            return status;
        }
    }
    {
        PreFOSStatus status = prefos_internal_expand_linear_objective(
            presolver, expanded_objective, &target->objective_offset);
        if (status != PREFOS_STATUS_OK)
        {
            free(quadratic_product);
            free(expanded_objective);
            return status;
        }
    }
    for (i = 0; i < source->n; ++i)
    {
        int mapped = presolver->original_to_reduced[i];
        if (mapped >= 0)
        {
            if (!prefos_internal_safe_add_product(&target->c[mapped], 1.0,
                                               expanded_objective[i]) ||
                !prefos_internal_safe_add_product(&target->c[mapped], 1.0,
                                               quadratic_product[i]))
            {
                free(quadratic_product);
                free(expanded_objective);
                return PREFOS_STATUS_NUMERICAL_ERROR;
            }
        }
        else if (presolver->is_fixed[i])
        {
            if (!prefos_internal_safe_add_product(&target->objective_offset,
                                               0.5 * presolver->fixed_values[i],
                                               quadratic_product[i]))
            {
                free(quadratic_product);
                free(expanded_objective);
                return PREFOS_STATUS_NUMERICAL_ERROR;
            }
        }
        else if (presolver->is_substituted[i])
        {
            if (quadratic_product[i] != 0.0 || expanded_objective[i] != 0.0)
            {
                free(quadratic_product);
                free(expanded_objective);
                return PREFOS_STATUS_NUMERICAL_ERROR;
            }
        }
        else if (presolver->is_parallel_removed[i])
        {
            if (quadratic_product[i] != 0.0)
            {
                free(quadratic_product);
                free(expanded_objective);
                return PREFOS_STATUS_NUMERICAL_ERROR;
            }
        }
        else
        {
            free(quadratic_product);
            free(expanded_objective);
            return PREFOS_STATUS_NUMERICAL_ERROR;
        }
    }
    free(quadratic_product);
    free(expanded_objective);
    return PREFOS_STATUS_OK;
}

#define PREFOS_MAX_MEDIUM_FIXED_POINT_ROUNDS 3

static PreFOSStatus run_medium_reduction_pass(
    PreFOSPresolver *presolver, int include_parallel_rows,
    int include_row_redundancy,
    int *changed_after_linear,
    PreFOSColumnWorkspace *shared_workspace)
{
    PreFOSFastTriggerSignature after_linear;
    PreFOSTimestamp start, stop;
    PreFOSStatus status;

    *changed_after_linear = 0;
    if (include_parallel_rows)
    {
        status = prefos_internal_remove_parallel_rows(presolver);
        if (status != PREFOS_STATUS_OK) return status;
    }
    if (shared_workspace)
        prefos_internal_update_column_live_degrees(
            presolver, shared_workspace);
    prefos_internal_timer_now(&start);
    status = prefos_internal_propagate_linear_bounds(
        presolver, shared_workspace);
    prefos_internal_timer_now(&stop);
    presolver->stats.linear_propagation_milliseconds +=
        prefos_internal_timer_elapsed_milliseconds(&start, &stop);
    if (status != PREFOS_STATUS_OK) return status;
    after_linear = capture_fast_trigger_signature(presolver);

    if (include_row_redundancy)
    {
        prefos_internal_timer_now(&start);
        status = prefos_internal_remove_redundant_rows_by_activity(
            presolver, shared_workspace);
        prefos_internal_timer_now(&stop);
        presolver->stats.redundant_row_activity_milliseconds +=
            prefos_internal_timer_elapsed_milliseconds(&start, &stop);
        if (status != PREFOS_STATUS_OK) return status;
        presolver->scalar_redundancy_completed = 1;
    }

    prefos_internal_timer_now(&start);
    status = prefos_internal_propagate_cone_envelopes(presolver);
    prefos_internal_timer_now(&stop);
    presolver->stats.cone_propagation_milliseconds +=
        prefos_internal_timer_elapsed_milliseconds(&start, &stop);
    if (status != PREFOS_STATUS_OK) return status;

    prefos_internal_timer_now(&start);
    status = prefos_internal_detect_zero_cone_collapses(presolver);
    prefos_internal_timer_now(&stop);
    presolver->stats.cone_collapse_milliseconds +=
        prefos_internal_timer_elapsed_milliseconds(&start, &stop);
    if (status != PREFOS_STATUS_OK) return status;

    status = prefos_internal_remove_redundant_box_bounds(presolver);
    if (status != PREFOS_STATUS_OK) return status;
    *changed_after_linear = fast_trigger_signature_changed(
        after_linear, capture_fast_trigger_signature(presolver));
    return PREFOS_STATUS_OK;
}

static PreFOSStatus run_fast_fixed_point_timed(
    PreFOSPresolver *presolver, int allow_one_sided_singletons,
    int full_trivial_scan, PreFOSColumnWorkspace *shared_workspace)
{
    PreFOSTimestamp start, stop;
    PreFOSStatus status;
    prefos_internal_timer_now(&start);
    status = prefos_internal_run_fast_fixed_point(
        presolver, allow_one_sided_singletons, full_trivial_scan,
        shared_workspace);
    prefos_internal_timer_now(&stop);
    presolver->stats.fast_fixed_point_milliseconds +=
        prefos_internal_timer_elapsed_milliseconds(&start, &stop);
    return status;
}

PreFOSStatus prefos_run_presolve(PreFOSPresolver *presolver)
{
    PreFOSStatus status;
    size_t n_fixed_total = 0, n_substituted_total = 0;
    size_t n_parallel_removed = 0, i;
    size_t fixed_affine_before_substitution;
    const PreFOSProblemData *source;
    PreFOSPresolvedProblem *target;
    PreFOSColumnWorkspace shared_column_workspace;
    int shared_column_workspace_valid = 0;
    PreFOSTimestamp presolve_start, phase_start, phase_stop;

    if (!presolver) return PREFOS_STATUS_INVALID_ARGUMENT;
    memset(&shared_column_workspace, 0, sizeof(shared_column_workspace));
    prefos_internal_cuda_workspace_release(presolver);
    prefos_internal_free_reduced_problem(&presolver->reduced);
    free(presolver->original_to_reduced);
    free(presolver->original_to_reduced_rows);
    free(presolver->fixed_values);
    free(presolver->is_fixed);
    free(presolver->fixed_column_log);
    free(presolver->is_substituted);
    free(presolver->is_parallel_removed);
    free(presolver->substitution_term_count);
    free(presolver->substitution_incoming_depth);
    free(presolver->substitution_keeps_source_row);
    free(presolver->substitution_term_start);
    free(presolver->substitution_source_row);
    free(presolver->residual_source_column);
    free(presolver->substitution_constant);
    free(presolver->substitution_targets);
    free(presolver->substitution_scales);
    free(presolver->variable_to_box);
    free(presolver->working_box_lower);
    free(presolver->working_box_upper);
    free(presolver->working_constraint_lower);
    free(presolver->working_constraint_upper);
    free(presolver->propagation_lower);
    free(presolver->propagation_upper);
    free(presolver->converted_affine_cones);
    free(presolver->affine_protected_columns);
    free(presolver->affine_aggregation_source_rows);
    free(presolver->affine_aggregation_pivots);
    free(presolver->affine_bound_certificates);
    free(presolver->affine_pre_to_reduced_rows);
    free(presolver->psd_structure_analyses);
    free(presolver->input_affine_rsoc_zero_axis);
    free(presolver->generated_affine_rsoc_zero_axis);
    free(presolver->affine_face_substitution_targets);
    free(presolver->affine_face_eliminated_columns);
    free(presolver->remove_rows);
    free(presolver->removed_row_log);
    free(presolver->remove_cones);
    free(presolver->cone_face_survivors);
    free(presolver->cone_face_box);
    free(presolver->cone_face_box_lower);
    free(presolver->cone_face_box_upper);
    free(presolver->cone_collapse_source_rows);
    prefos_internal_free_psd_face_reductions(presolver->psd_face_reductions,
                                          presolver->original.n_cones);
    free(presolver->facial_reductions);
    presolve_transformation_log_free(&presolver->transformations);
    presolver->original_to_reduced = NULL;
    presolver->original_to_reduced_rows = NULL;
    presolver->fixed_values = NULL;
    presolver->is_fixed = NULL;
    presolver->fixed_column_log = NULL;
    presolver->n_fixed_columns = 0;
    presolver->is_substituted = NULL;
    presolver->is_parallel_removed = NULL;
    presolver->substitution_term_count = NULL;
    presolver->substitution_incoming_depth = NULL;
    presolver->substitution_keeps_source_row = NULL;
    presolver->substitution_term_start = NULL;
    presolver->substitution_source_row = NULL;
    presolver->residual_source_column = NULL;
    presolver->substitution_constant = NULL;
    presolver->substitution_targets = NULL;
    presolver->substitution_scales = NULL;
    presolver->n_substitution_terms = 0;
    presolver->substitution_term_capacity = 0;
    presolver->n_residual_row_substitutions = 0;
    presolver->n_parallel_column_reductions = 0;
    presolver->variable_to_box = NULL;
    presolver->working_box_lower = NULL;
    presolver->working_box_upper = NULL;
    presolver->working_constraint_lower = NULL;
    presolver->working_constraint_upper = NULL;
    presolver->propagation_lower = NULL;
    presolver->propagation_upper = NULL;
    presolver->converted_affine_cones = NULL;
    presolver->affine_protected_columns = NULL;
    presolver->affine_aggregation_source_rows = NULL;
    presolver->affine_aggregation_pivots = NULL;
    presolver->affine_bound_certificates = NULL;
    presolver->n_affine_bound_certificates = 0;
    presolver->affine_bound_certificate_capacity = 0;
    presolver->n_pre_face_affine_rows = 0;
    presolver->affine_pre_to_reduced_rows = NULL;
    presolver->psd_structure_analyses = NULL;
    presolver->n_psd_structure_analyses = 0;
    presolver->input_affine_rsoc_zero_axis = NULL;
    presolver->generated_affine_rsoc_zero_axis = NULL;
    presolver->affine_face_substitution_targets = NULL;
    presolver->affine_face_eliminated_columns = NULL;
    presolver->n_affine_face_substitutions = 0;
    presolver->remove_rows = NULL;
    presolver->removed_row_log = NULL;
    presolver->n_removed_rows = 0;
    presolver->remove_cones = NULL;
    presolver->cone_face_survivors = NULL;
    presolver->cone_face_box = NULL;
    presolver->cone_face_box_lower = NULL;
    presolver->cone_face_box_upper = NULL;
    presolver->cone_collapse_source_rows = NULL;
    presolver->psd_face_reductions = NULL;
    presolver->facial_reductions = NULL;
    presolver->n_facial_reductions = 0;
    presolve_transformation_log_init(&presolver->transformations);
    memset(&presolver->stats, 0, sizeof(presolver->stats));
    presolver->scalar_redundancy_completed = 0;
    presolver->fixed_column_epoch = 0;
    presolver->has_run = 0;
    prefos_internal_timer_now(&presolve_start);

    source = &presolver->original;
    target = &presolver->reduced;
    presolver->stats.rows_original = source->A.rows;
    presolver->stats.variables_original = source->n;
    presolver->stats.nnz_A_original = source->A.nnz;
    presolver->stats.nnz_Q_original = source->Q.nnz;
    presolver->stats.nnz_R_original = source->R.nnz;
    presolver->stats.normalized_nonnegative_variables =
        presolver->normalized_nonnegative_variables;
    presolver->stats.normalized_nonnegative_cones =
        presolver->normalized_nonnegative_cones;

    prefos_internal_timer_now(&phase_start);
    status = initialize_working_state(presolver);
    prefos_internal_timer_now(&phase_stop);
    presolver->stats.initialization_milliseconds =
        prefos_internal_timer_elapsed_milliseconds(
            &phase_start, &phase_stop);
    if (status != PREFOS_STATUS_OK) goto failure;
    {
        size_t aggregations_before =
            presolver->stats.aggregated_affine_cone_coordinates;
        prefos_internal_timer_now(&phase_start);
        status =
            prefos_internal_aggregate_affine_cone_coordinates(presolver);
        prefos_internal_timer_now(&phase_stop);
        presolver->stats.affine_aggregation_milliseconds =
            prefos_internal_timer_elapsed_milliseconds(
                &phase_start, &phase_stop);
        if (presolver->stats.aggregated_affine_cone_coordinates >
            aggregations_before)
            prefos_internal_cuda_workspace_release(presolver);
    }
    if (status != PREFOS_STATUS_OK) goto failure;
    {
        size_t ignored_fixed;
        status = prefos_internal_find_fixed_box_variables(
            presolver, &ignored_fixed);
    }
    if (status != PREFOS_STATUS_OK) goto failure;
    if (source->n_box > 0)
    {
        prefos_internal_timer_now(&phase_start);
        status = prefos_internal_build_column_workspace_cpu(
            presolver, &shared_column_workspace);
        prefos_internal_timer_now(&phase_stop);
        presolver->stats.structural_reduction_milliseconds +=
            prefos_internal_timer_elapsed_milliseconds(
                &phase_start, &phase_stop);
        if (status != PREFOS_STATUS_OK) goto failure;
        shared_column_workspace_valid = 1;
    }
    if (source->A.rows > 0 && source->n >= 65536 &&
        (long double) source->n >
            8.0L * (long double) source->A.rows)
    {
        status = shared_column_workspace_valid
                     ? prefos_internal_reduce_parallel_columns_in_workspace(
                           presolver, &shared_column_workspace)
                     : prefos_internal_reduce_parallel_columns(presolver);
        if (status != PREFOS_STATUS_OK) goto failure;
    }
    status = run_fast_fixed_point_timed(
        presolver, 0, 1,
        shared_column_workspace_valid ? &shared_column_workspace : NULL);
    if (status != PREFOS_STATUS_OK) goto failure;
    fixed_affine_before_substitution =
        presolver->stats.fixed_affine_face_variables;
    {
        PreFOSTimestamp start, stop;
        prefos_internal_timer_now(&start);
        status = prefos_internal_substitute_affine_face_equalities(presolver);
        prefos_internal_timer_now(&stop);
        presolver->stats.affine_face_substitution_milliseconds =
            prefos_internal_timer_elapsed_milliseconds(&start, &stop);
    }
    if (status != PREFOS_STATUS_OK) goto failure;
    prefos_internal_timer_now(&phase_start);
    status = prefos_internal_substitute_free_columns(presolver);
    prefos_internal_timer_now(&phase_stop);
    presolver->stats.free_column_substitution_milliseconds =
        prefos_internal_timer_elapsed_milliseconds(
            &phase_start, &phase_stop);
    if (status != PREFOS_STATUS_OK) goto failure;
    if (presolver->stats.fixed_affine_face_variables >
        fixed_affine_before_substitution)
    {
        prefos_internal_timer_now(&phase_start);
        status = prefos_internal_reduce_trivial_rows(presolver);
        prefos_internal_timer_now(&phase_stop);
        presolver->stats.trivial_row_reduction_milliseconds =
            prefos_internal_timer_elapsed_milliseconds(
                &phase_start, &phase_stop);
        if (status != PREFOS_STATUS_OK) goto failure;
    }
    {
        int round;
        prefos_internal_timer_now(&phase_start);
        for (round = 0; round < PREFOS_MAX_MEDIUM_FIXED_POINT_ROUNDS;
             ++round)
        {
            PreFOSFastTriggerSignature before_medium =
                capture_fast_trigger_signature(presolver);
            PreFOSFastTriggerSignature before_fast;
            int changed_after_linear = 0;
            int fast_changed = 0;
            int fast_fixed_columns = 0;
            size_t fixed_epoch_before_fast;
            ++presolver->stats.medium_fixed_point_rounds;

            status = run_medium_reduction_pass(
                presolver, round == 0, round == 0,
                &changed_after_linear,
                shared_column_workspace_valid
                    ? &shared_column_workspace
                    : NULL);
            if (status != PREFOS_STATUS_OK) goto failure;
            if (source->A.rows == 0 || source->n_box == 0)
                changed_after_linear = 0;
            before_fast = capture_fast_trigger_signature(presolver);
            fixed_epoch_before_fast = presolver->fixed_column_epoch;
            if (fast_trigger_signature_changed(
                    before_medium, before_fast))
            {
                status = run_fast_fixed_point_timed(
                    presolver, 1, 0,
                    shared_column_workspace_valid
                        ? &shared_column_workspace
                        : NULL);
                if (status != PREFOS_STATUS_OK) goto failure;
                fast_changed = fast_trigger_signature_changed(
                    before_fast,
                    capture_fast_trigger_signature(presolver));
                fast_fixed_columns =
                    presolver->fixed_column_epoch !=
                    fixed_epoch_before_fast;
            }
            if (source->n_cones == 0 &&
                source->n_affine_cones == 0)
            {
                if (fast_fixed_columns)
                    presolver->scalar_redundancy_completed = 0;
                else
                    break;
            }
            if (!changed_after_linear && !fast_changed) break;
        }
        prefos_internal_timer_now(&phase_stop);
        presolver->stats.medium_fixed_point_milliseconds =
            prefos_internal_timer_elapsed_milliseconds(
                &phase_start, &phase_stop);
    }
    {
        PreFOSFastTriggerSignature before_parallel_columns =
            capture_fast_trigger_signature(presolver);
        prefos_internal_timer_now(&phase_start);
        status = shared_column_workspace_valid
                     ? prefos_internal_reduce_parallel_columns_in_workspace(
                           presolver, &shared_column_workspace)
                     : prefos_internal_reduce_parallel_columns(presolver);
        prefos_internal_timer_now(&phase_stop);
        presolver->stats.parallel_column_reduction_milliseconds =
            prefos_internal_timer_elapsed_milliseconds(
                &phase_start, &phase_stop);
        if (status != PREFOS_STATUS_OK) goto failure;
        if (fast_trigger_signature_changed(
                before_parallel_columns,
                capture_fast_trigger_signature(presolver)))
        {
            status = run_fast_fixed_point_timed(
                presolver, 1, 0,
                shared_column_workspace_valid
                    ? &shared_column_workspace
                    : NULL);
            if (status != PREFOS_STATUS_OK) goto failure;
        }
    }
    if (shared_column_workspace_valid)
    {
        prefos_internal_free_column_workspace(
            &shared_column_workspace);
        shared_column_workspace_valid = 0;
    }
    for (i = 0; i < source->n; ++i)
    {
        if (presolver->is_fixed[i]) ++n_fixed_total;
        if (presolver->is_substituted[i]) ++n_substituted_total;
        if (presolver->is_parallel_removed[i]) ++n_parallel_removed;
    }
    presolver->stats.fixed_box_variables = 0;
    for (i = 0; i < source->n_box; ++i)
        if (presolver->is_fixed[source->box_indices[i]])
            ++presolver->stats.fixed_box_variables;
    target->n =
        source->n - n_fixed_total - n_substituted_total - n_parallel_removed;
    target->q_storage = source->q_storage;
    build_variable_map(presolver);

    prefos_internal_timer_now(&phase_start);
    status = prefos_internal_compact_a(presolver);
    prefos_internal_timer_now(&phase_stop);
    presolver->stats.matrix_compaction_milliseconds =
        prefos_internal_timer_elapsed_milliseconds(
            &phase_start, &phase_stop);
    if (status != PREFOS_STATUS_OK) goto failure;
    prefos_internal_timer_now(&phase_start);
    status = compact_q(presolver, &target->Q);
    prefos_internal_timer_now(&phase_stop);
    presolver->stats.quadratic_compaction_milliseconds =
        prefos_internal_timer_elapsed_milliseconds(
            &phase_start, &phase_stop);
    if (status != PREFOS_STATUS_OK) goto failure;
    prefos_internal_timer_now(&phase_start);
    status = compact_general_matrix(&source->R, presolver->original_to_reduced,
                                    target->n, &target->R);
    prefos_internal_timer_now(&phase_stop);
    presolver->stats.factor_compaction_milliseconds =
        prefos_internal_timer_elapsed_milliseconds(
            &phase_start, &phase_stop);
    if (status != PREFOS_STATUS_OK) goto failure;
    status = prefos_internal_copy_vector(source->D, source->R.rows, sizeof(double),
                                      (void **) &target->D);
    if (status != PREFOS_STATUS_OK) goto failure;
    prefos_internal_timer_now(&phase_start);
    status = build_reduced_domains(presolver);
    prefos_internal_timer_now(&phase_stop);
    presolver->stats.domain_compaction_milliseconds =
        prefos_internal_timer_elapsed_milliseconds(
            &phase_start, &phase_stop);
    if (status != PREFOS_STATUS_OK) goto failure;
    status = prefos_internal_build_reduced_affine_cones(presolver);
    if (status != PREFOS_STATUS_OK) goto failure;
    {
        PreFOSTimestamp start, stop;
        prefos_internal_timer_now(&start);
        status = prefos_internal_reduce_affine_cone_faces(presolver);
        prefos_internal_timer_now(&stop);
        presolver->stats.affine_face_reduction_milliseconds =
            prefos_internal_timer_elapsed_milliseconds(&start, &stop);
    }
    if (status != PREFOS_STATUS_OK) goto failure;
    {
        PreFOSTimestamp start, stop;
        prefos_internal_timer_now(&start);
        status = prefos_internal_analyze_and_decompose_affine_psd(presolver);
        prefos_internal_timer_now(&stop);
        presolver->stats.affine_psd_structure_milliseconds =
            prefos_internal_timer_elapsed_milliseconds(&start, &stop);
    }
    if (status != PREFOS_STATUS_OK) goto failure;
    prefos_internal_timer_now(&phase_start);
    status = build_reduced_objective(presolver);
    prefos_internal_timer_now(&phase_stop);
    presolver->stats.objective_compaction_milliseconds =
        prefos_internal_timer_elapsed_milliseconds(
            &phase_start, &phase_stop);
    if (status != PREFOS_STATUS_OK) goto failure;

    presolver->stats.rows_reduced = target->A.rows;
    presolver->stats.variables_reduced = target->n;
    presolver->stats.nnz_A_reduced = target->A.nnz;
    presolver->stats.nnz_Q_reduced = target->Q.nnz;
    presolver->stats.nnz_R_reduced = target->R.nnz;
    presolver->has_run = 1;
    prefos_internal_timer_now(&phase_stop);
    presolver->stats.presolve_total_milliseconds =
        prefos_internal_timer_elapsed_milliseconds(
            &presolve_start, &phase_stop);

    if (n_fixed_total > 0 || presolver->stats.substituted_free_variables > 0 ||
        presolver->stats.normalized_nonnegative_cones > 0 ||
        presolver->stats.aggregated_affine_cone_coordinates > 0 ||
        presolver->stats.materialized_affine_cone_box_bounds > 0 ||
        presolver->stats.derived_affine_face_equalities > 0 ||
        presolver->stats.removed_affine_cone_coordinates > 0 ||
        presolver->stats.decomposed_affine_psd_blocks > 0 ||
        presolver->stats.materialized_propagated_box_bounds > 0 ||
        presolver->stats.collapsed_cones > 0 ||
        presolver->stats.reduced_rsoc_faces > 0 ||
        presolver->stats.reduced_psd_faces > 0 ||
        presolver->stats.reduced_exponential_faces > 0 ||
        presolver->stats.reduced_power_faces > 0 ||
        presolver->stats.removed_redundant_rows > 0 ||
        presolver->stats.removed_redundant_row_lower_sides > 0 ||
        presolver->stats.removed_redundant_row_upper_sides > 0 ||
        presolver->stats.removed_redundant_box_lower_bounds > 0 ||
        presolver->stats.removed_redundant_box_upper_bounds > 0 ||
        presolver->stats.removed_empty_columns > 0 ||
        presolver->stats.removed_singleton_columns > 0 ||
        presolver->stats.tightened_singleton_rows > 0 ||
        presolver->stats.substituted_bounded_doubletons > 0 ||
        presolver->stats.dual_fixed_columns > 0 ||
        presolver->stats.merged_parallel_columns > 0 ||
        presolver->stats.removed_singleton_rows > 0 ||
        presolver->stats.removed_empty_rows > 0 || target->A.nnz != source->A.nnz ||
        target->Q.nnz != source->Q.nnz || target->R.nnz != source->R.nnz)
    {
        return PREFOS_STATUS_REDUCED;
    }
    return PREFOS_STATUS_OK;

failure:
    if (shared_column_workspace_valid)
        prefos_internal_free_column_workspace(
            &shared_column_workspace);
    prefos_internal_timer_now(&phase_stop);
    presolver->stats.presolve_total_milliseconds =
        prefos_internal_timer_elapsed_milliseconds(
            &presolve_start, &phase_stop);
    prefos_internal_free_reduced_problem(target);
    return status;
}

const PreFOSPresolvedProblem *prefos_get_reduced_problem(const PreFOSPresolver *presolver)
{
    if (!presolver || !presolver->has_run) return NULL;
    return &presolver->reduced;
}

const PreFOSStats *prefos_get_stats(const PreFOSPresolver *presolver)
{
    if (!presolver) return NULL;
    return &presolver->stats;
}

const PreFOSPSDStructureAnalysis *
prefos_get_psd_structure_analyses(const PreFOSPresolver *presolver, size_t *count)
{
    if (!count) return NULL;
    *count = 0;
    if (!presolver || !presolver->has_run) return NULL;
    *count = presolver->n_psd_structure_analyses;
    return presolver->n_psd_structure_analyses > 0
               ? presolver->psd_structure_analyses
               : NULL;
}

const PreFOSFacialReductionCertificate *
prefos_get_facial_reductions(const PreFOSPresolver *presolver, size_t *count)
{
    if (!count) return NULL;
    *count = 0;
    if (!presolver || !presolver->has_run) return NULL;
    *count = presolver->n_facial_reductions;
    return presolver->n_facial_reductions > 0 ? presolver->facial_reductions : NULL;
}
