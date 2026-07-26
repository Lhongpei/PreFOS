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
#include "core/PREFOS_WorkingMatrix.h"
#include "explorers/PREFOS_AffineConeAggregation.h"
#include "explorers/PREFOS_AffineFaceSubstitution.h"
#include "explorers/PREFOS_ConePropagation.h"
#include "explorers/PREFOS_ColumnReductionInternal.h"
#include "explorers/PREFOS_ColumnReductions.h"
#include "explorers/PREFOS_CudaBackend.h"
#include "explorers/PREFOS_CudaLinearPropagation.h"
#include "explorers/PREFOS_FreeColumnSubstitution.h"
#include "explorers/PREFOS_LinearPropagation.h"
#include "explorers/PREFOS_LinearPropagationCache.h"
#include "explorers/PREFOS_MaterializedColumnClosure.h"
#include "explorers/PREFOS_ParallelRows.h"
#include "explorers/PREFOS_SDPStructureAnalysis.h"
#include "explorers/PREFOS_TrivialReductions.h"

#include <stdio.h>
#include <stdlib.h>

#define PREFOS_HARD_TAIL_PROPAGATION_MIN_NNZ 262144U
#define PREFOS_MATERIALIZED_FIRST_BOUNDED_MIN_NNZ 2097152U

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
        (uint16_t *) calloc(problem->n, sizeof(uint16_t));
    presolver->substitution_fill_in_targets =
        (unsigned char *) calloc(problem->n, sizeof(unsigned char));
    presolver->substitution_keeps_source_row =
        (unsigned char *) calloc(problem->n, sizeof(unsigned char));
    presolver->substitution_term_start =
        (size_t *) calloc(problem->n, sizeof(size_t));
    presolver->substitution_source_row =
        (int *) prefos_internal_alloc_array(problem->n, sizeof(int));
    presolver->residual_source_column =
        (int *) prefos_internal_alloc_array(problem->A.rows, sizeof(int));
    presolver->rows_require_materialization =
        (unsigned char *) calloc(problem->A.rows, sizeof(unsigned char));
    presolver->materialization_row_log_complete = 1;
    presolver->substitution_constant = (double *) calloc(problem->n, sizeof(double));
    presolver->variable_to_box =
        (int *) prefos_internal_alloc_array(problem->n, sizeof(int));
    presolver->working_box_lower =
        (double *) prefos_internal_alloc_array(problem->n_box, sizeof(double));
    presolver->working_box_upper =
        (double *) prefos_internal_alloc_array(problem->n_box, sizeof(double));
    presolver->latest_lower_bound_change =
        (size_t *) prefos_internal_alloc_array(problem->n, sizeof(size_t));
    presolver->latest_upper_bound_change =
        (size_t *) prefos_internal_alloc_array(problem->n, sizeof(size_t));
    presolver->fixed_box_dirty =
        (unsigned char *) calloc(problem->n_box, sizeof(unsigned char));
    presolver->fixed_box_dirty_queue =
        (int *) prefos_internal_alloc_array(problem->n_box, sizeof(int));
    presolver->working_constraint_lower =
        (double *) prefos_internal_alloc_array(problem->A.rows, sizeof(double));
    presolver->working_constraint_upper =
        (double *) prefos_internal_alloc_array(problem->A.rows, sizeof(double));
    presolver->propagation_lower =
        (double *) prefos_internal_alloc_array(problem->n, sizeof(double));
    presolver->propagation_upper =
        (double *) prefos_internal_alloc_array(problem->n, sizeof(double));
    presolver->nonmaterialized_bound_source_rows =
        (unsigned char *) calloc(problem->A.rows, sizeof(unsigned char));
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
         !presolver->substitution_fill_in_targets ||
         !presolver->substitution_keeps_source_row ||
         !presolver->substitution_term_start ||
         !presolver->substitution_source_row ||
         !presolver->substitution_constant ||
         !presolver->variable_to_box || !presolver->propagation_lower ||
         !presolver->propagation_upper ||
         !presolver->latest_lower_bound_change ||
         !presolver->latest_upper_bound_change ||
         !presolver->affine_protected_columns ||
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
                                 !presolver->residual_source_column ||
                                 !presolver->rows_require_materialization ||
                                 !presolver->nonmaterialized_bound_source_rows)) ||
        (problem->n_box > 0 &&
         (!presolver->working_box_lower || !presolver->working_box_upper ||
          !presolver->fixed_box_dirty ||
          !presolver->fixed_box_dirty_queue)) ||
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
        presolver->latest_lower_bound_change[i] = SIZE_MAX;
        presolver->latest_upper_bound_change[i] = SIZE_MAX;
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
        presolver->fixed_box_dirty[i] = 1;
        presolver->fixed_box_dirty_queue[i] = (int) i;
        presolver->propagation_lower[index] = problem->box_lower[i];
        presolver->propagation_upper[index] = problem->box_upper[i];
    }
    presolver->n_fixed_box_dirty = problem->n_box;
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
    size_t record_index;
    size_t *latest = NULL;
    if (old_bound == new_bound) return PREFOS_STATUS_OK;
    memset(&record, 0, sizeof(record));
    record.row = row;
    record.column = column;
    record.previous_bound = old_bound;
    record.implied_bound = new_bound;
    record.is_lower = (uint8_t) is_lower;
    record.has_previous_bound = 1;
    if (!presolver->working_matrix_is_materialized &&
        column >= 0 && (size_t) column < presolver->original.n)
    {
        latest = is_lower ? presolver->latest_lower_bound_change
                          : presolver->latest_upper_bound_change;
        if (latest && latest[column] != SIZE_MAX)
        {
            record.previous_same_side_record_index = latest[column];
            record.has_previous_same_side_record = 1;
        }
    }
    if (presolver->working_matrix_is_materialized)
    {
        const PreFOSCsrMatrix *matrix = &presolver->original.A;
        PresolveRowTransformationRecord source;
        size_t source_index;
        int begin, end;
        if (row < 0 || (size_t) row >= matrix->rows ||
            !presolver->materialized_bound_source_records ||
            (size_t) row >=
                presolver->n_materialized_bound_source_records)
            return PREFOS_STATUS_NUMERICAL_ERROR;
        source_index =
            presolver->materialized_bound_source_records[row];
        if (source_index == 0)
        {
            begin = matrix->row_pointers[row];
            end = matrix->row_pointers[row + 1];
            memset(&source, 0, sizeof(source));
            source.type = PRESOLVE_ROW_BOUND_CHANGE_SOURCE;
            source.row = row;
            source.indices = matrix->column_indices + begin;
            source.coefficients = matrix->values + begin;
            source.length = (size_t) (end - begin);
            if (!presolve_transformation_log_store_row_source(
                    &presolver->transformations, &source,
                    &source_index))
                return PREFOS_STATUS_OUT_OF_MEMORY;
            if (source_index == SIZE_MAX)
                return PREFOS_STATUS_OUT_OF_MEMORY;
            presolver->materialized_bound_source_records[row] =
                source_index + 1;
        }
        else
            --source_index;
        record.source_row_record_index = source_index;
        record.has_source_row_record = 1;
    }
    if (!presolve_transformation_log_append_bound_change(
            &presolver->transformations, &record, &record_index))
        return PREFOS_STATUS_OUT_OF_MEMORY;
    if (latest) latest[column] = record_index;
    return PREFOS_STATUS_OK;
}

void prefos_internal_clear_box_bound_provenance(
    PreFOSPresolver *presolver, int column, int is_lower)
{
    size_t *latest;
    if (!presolver || column < 0 ||
        (size_t) column >= presolver->original.n)
        return;
    latest = is_lower ? presolver->latest_lower_bound_change
                      : presolver->latest_upper_bound_change;
    if (latest) latest[column] = SIZE_MAX;
}

double prefos_internal_box_bound_without_row(
    const PreFOSPresolver *presolver, int column, int row, int is_lower)
{
    const PresolveTransformationLog *log;
    const size_t *latest;
    size_t record_index, visited = 0;
    int box_position;
    double bound;

    if (!presolver || column < 0 ||
        (size_t) column >= presolver->original.n)
        return is_lower ? -INFINITY : INFINITY;
    box_position = presolver->variable_to_box[column];
    if (box_position < 0)
        return is_lower ? -INFINITY : INFINITY;
    bound = is_lower ? presolver->working_box_lower[box_position]
                     : presolver->working_box_upper[box_position];
    latest = is_lower ? presolver->latest_lower_bound_change
                      : presolver->latest_upper_bound_change;
    if (!latest) return bound;
    log = &presolver->transformations;
    record_index = latest[column];
    while (record_index != SIZE_MAX &&
           record_index < log->n_bound_changes &&
           visited++ < log->n_bound_changes)
    {
        const PresolveBoundChangeRecord *record =
            &log->bound_changes[record_index];
        if (record->column != column ||
            (int) record->is_lower != !!is_lower ||
            record->row != row || !record->has_previous_bound ||
            record->implied_bound != bound)
            break;
        bound = record->previous_bound;
        if (!record->has_previous_same_side_record)
            break;
        record_index = record->previous_same_side_record_index;
    }
    return bound;
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

PreFOSStatus prefos_internal_expand_linear_objective(
    const PreFOSPresolver *presolver, double *objective, double *offset)
{
    const PresolveTransformationLog *log;
    size_t position;
    if (!presolver || !offset ||
        (presolver->original.n > 0 && !objective))
        return PREFOS_STATUS_INVALID_ARGUMENT;
    if (presolver->original.n > 0)
        memcpy(
            objective, presolver->original.c,
            presolver->original.n * sizeof(double));
    *offset = presolver->original.objective_offset;

    /*
     * A substitution can only target a column that is still active.  The
     * transformation log is therefore a topological order for the
     * substitution graph. Replaying it moves every objective coefficient
     * exactly once, avoiding a recursive walk for every original nonzero.
     */
    log = &presolver->transformations;
    for (position = 0;
         position < log->n_column_transformations; ++position)
    {
        const PresolveColumnTransformationRecord *record =
            &log->column_transformations[position];
        int column = record->column;
        double coefficient;
        if (column < 0 ||
            (size_t) column >= presolver->original.n)
            return PREFOS_STATUS_NUMERICAL_ERROR;
        if (record->type == PRESOLVE_COLUMNS_PARALLEL)
        {
            objective[column] = 0.0;
            continue;
        }
        if (record->type != PRESOLVE_COLUMN_SUBSTITUTED)
            continue;
        if (!presolver->is_substituted[column])
            return PREFOS_STATUS_NUMERICAL_ERROR;
        coefficient = objective[column];
        objective[column] = 0.0;
        if (coefficient != 0.0)
        {
            size_t start =
                presolver->substitution_term_start[column];
            size_t count =
                presolver->substitution_term_count[column];
            size_t term;
            if (count == 0 ||
                start > presolver->n_substitution_terms ||
                count > presolver->n_substitution_terms - start ||
                !prefos_internal_safe_add_product(
                    offset, coefficient,
                    presolver->substitution_constant[column]))
                return PREFOS_STATUS_NUMERICAL_ERROR;
            for (term = 0; term < count; ++term)
            {
                int target =
                    presolver->substitution_targets[start + term];
                if (target < 0 ||
                    (size_t) target >= presolver->original.n ||
                    !prefos_internal_safe_add_product(
                        &objective[target], coefficient,
                        presolver->substitution_scales[start + term]))
                    return PREFOS_STATUS_NUMERICAL_ERROR;
            }
        }
    }
    for (position = 0;
         position < presolver->n_fixed_columns; ++position)
    {
        int column = presolver->fixed_column_log[position];
        double coefficient;
        if (column < 0 ||
            (size_t) column >= presolver->original.n)
            return PREFOS_STATUS_NUMERICAL_ERROR;
        coefficient = objective[column];
        objective[column] = 0.0;
        if (coefficient != 0.0 &&
            !prefos_internal_safe_add_product(
                offset, coefficient,
                presolver->fixed_values[column]))
            return PREFOS_STATUS_NUMERICAL_ERROR;
    }
    return PREFOS_STATUS_OK;
}

static PreFOSStatus build_reduced_objective(
    PreFOSPresolver *presolver, double *expanded_objective,
    double expanded_objective_offset, int has_expanded_objective)
{
    const PreFOSProblemData *source = &presolver->original;
    PreFOSPresolvedProblem *target = &presolver->reduced;
    double *quadratic_product;
    size_t i;

    quadratic_product = (double *) calloc(source->n, sizeof(double));
    if (!expanded_objective)
        expanded_objective =
            (double *) calloc(source->n, sizeof(double));
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
    if (has_expanded_objective)
        target->objective_offset = expanded_objective_offset;
    else
    {
        PreFOSStatus status =
            prefos_internal_expand_linear_objective(
                presolver, expanded_objective,
                &target->objective_offset);
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

#define PREFOS_MAX_MEDIUM_FIXED_POINT_ROUNDS 16
#define PREFOS_LAZY_MATERIALIZED_PROPAGATION_MIN_NNZ 262144U

static size_t saturated_change_sum(size_t first, size_t second)
{
    return second > SIZE_MAX - first ? SIZE_MAX : first + second;
}

static void trace_medium_stage(
    const PreFOSPresolver *presolver, const char *stage)
{
    if (!getenv("PREFOS_TRACE_MEDIUM_STAGES")) return;
    fprintf(
        stderr,
        "PreFOS medium-stage stage=%s fixed=%zu substituted=%zu "
        "parallel=%zu removed=%zu cache_valid=%d cache_nnz=%zu "
        "fixed_epoch=%zu column_events=%zu\n",
        stage, presolver->n_fixed_columns,
        presolver->stats.substituted_free_variables,
        presolver->n_parallel_column_reductions,
        presolver->n_removed_rows,
        presolver->cached_working_matrix_valid,
        presolver->cached_working_matrix_valid
            ? presolver->cached_working_matrix.nnz
            : 0,
        presolver->fixed_column_epoch,
        presolver->transformations.n_column_transformations);
}

static void trace_presolve_stage(
    const PreFOSPresolver *presolver, const char *stage,
    const PreFOSTimestamp *presolve_start)
{
    PreFOSTimestamp now;
    double elapsed_milliseconds = 0.0;
    if (!getenv("PREFOS_TRACE_PRESOLVE_STAGES")) return;
    if (presolve_start)
    {
        prefos_internal_timer_now(&now);
        elapsed_milliseconds =
            prefos_internal_timer_elapsed_milliseconds(
                presolve_start, &now);
    }
    fprintf(
        stderr,
        "PreFOS stage=%s fixed=%zu substituted=%zu parallel=%zu "
        "removed_rows=%zu events=%zu dirty_rows=%zu dirty_nnz=%zu "
        "scalar_complete=%d elapsed_ms=%.3f\n",
        stage, presolver->n_fixed_columns,
        presolver->stats.substituted_free_variables,
        presolver->n_parallel_column_reductions,
        presolver->n_removed_rows,
        presolver->transformations.n_events,
        presolver->n_rows_require_materialization,
        presolver->materialization_source_nnz,
        presolver->scalar_redundancy_completed,
        elapsed_milliseconds);
}

static int medium_fixed_point_round_is_stale(
    const PreFOSPresolver *presolver,
    PreFOSFastTriggerSignature before,
    PreFOSFastTriggerSignature after,
    size_t fixed_before, size_t substituted_before,
    size_t parallel_before)
{
    const PreFOSLinearPropagationState *state =
        presolver->linear_propagation_cache;
    size_t changes = 0;
    size_t active_nnz;
    size_t minimum_yield;

    if (presolver->settings.linear_propagation_max_stale_rounds == 0)
        return 0;
    if (state)
        active_nnz = state->work_budget_quantum;
    else if (presolver->cached_working_matrix_valid)
        active_nnz = presolver->cached_working_matrix.nnz;
    else
        active_nnz = presolver->original.A.nnz;

    changes = saturated_change_sum(
        changes, presolver->n_fixed_columns - fixed_before);
    changes = saturated_change_sum(
        changes,
        presolver->stats.substituted_free_variables -
            substituted_before);
    changes = saturated_change_sum(
        changes,
        presolver->n_parallel_column_reductions - parallel_before);
    changes = saturated_change_sum(
        changes, after.removed_rows - before.removed_rows);
    if (presolver->original.n_cones > 0 ||
        presolver->original.n_affine_cones > 0)
    {
        changes = saturated_change_sum(
            changes, after.changed_bounds - before.changed_bounds);
        changes = saturated_change_sum(
            changes, after.cone_reductions - before.cone_reductions);
    }

    minimum_yield = (active_nnz + 4095) / 4096;
    return changes < minimum_yield;
}

static PreFOSStatus run_medium_parallel_rows(
    PreFOSPresolver *presolver,
    PreFOSColumnWorkspace *shared_workspace)
{
    const PreFOSProblemData *problem = &presolver->original;
    const PreFOSCsrMatrix *source_matrix;
    const double *source_lower;
    const double *source_upper;
    PreFOSWorkingMatrix working;
    PreFOSStatus status;
    int parallel_rows_checked = 0;
    int reused_current_cache;
    int has_transformed_rows =
        presolver->n_rows_require_materialization > 0 ||
        presolver->n_residual_row_substitutions > 0 ||
        presolver->n_parallel_column_reductions > 0;
    int force_materialized_rows =
        getenv("PREFOS_FORCE_MEDIUM_MATERIALIZED_ROWS") != NULL;

    if (!has_transformed_rows ||
        (!force_materialized_rows &&
         problem->A.rows > 4096 &&
         problem->A.nnz > 1000000) ||
        problem->A.nnz > 20000000)
        return prefos_internal_remove_parallel_rows(
            presolver, shared_workspace);

    memset(&working, 0, sizeof(working));
    reused_current_cache =
        prefos_internal_take_current_working_matrix_cache(
            presolver, &working, &parallel_rows_checked);
    if (reused_current_cache)
        status = PREFOS_STATUS_OK;
    else
    {
        prefos_internal_get_working_matrix_source(
            presolver, &source_matrix, &source_lower, &source_upper);
        status = prefos_internal_materialize_working_matrix(
            presolver, source_matrix, source_lower, source_upper,
            &working);
    }
    if (status == PREFOS_STATUS_OK && !parallel_rows_checked)
        status =
            prefos_internal_remove_parallel_rows_in_working_matrix(
                presolver, &working.matrix,
                working.lower, working.upper, shared_workspace);
    if (status == PREFOS_STATUS_OK)
    {
        prefos_internal_store_working_matrix_cache(
            presolver, &working);
        presolver->cached_working_parallel_rows_checked = 1;
    }
    else
        prefos_internal_free_working_matrix(&working);
    return status;
}

static unsigned char *build_materialized_propagation_probe_rows(
    const PreFOSPresolver *presolver,
    const PreFOSColumnWorkspace *workspace)
{
    const PreFOSProblemData *problem = &presolver->original;
    const PresolveTransformationLog *log =
        &presolver->transformations;
    unsigned char *probe_rows;
    size_t column, record_index;

    probe_rows = (unsigned char *) calloc(
        problem->A.rows, sizeof(unsigned char));
    if (problem->A.rows > 0 && !probe_rows) return NULL;
    if (problem->A.rows > 0 &&
        presolver->rows_require_materialization)
        memcpy(
            probe_rows, presolver->rows_require_materialization,
            problem->A.rows * sizeof(unsigned char));

    if (!workspace)
    {
        if (presolver->linear_propagation_bound_cursor <
            log->n_bound_changes)
            memset(
                probe_rows, 1,
                problem->A.rows * sizeof(unsigned char));
        return probe_rows;
    }
    for (record_index =
             presolver->linear_propagation_bound_cursor;
         record_index < log->n_bound_changes; ++record_index)
    {
        int changed_column =
            log->bound_changes[record_index].column;
        int position;
        if (changed_column < 0 ||
            (size_t) changed_column >= problem->n)
            continue;
        for (position = workspace->starts[changed_column];
             position < workspace->ends[changed_column];
             ++position)
        {
            int row = workspace->rows[position];
            if (row >= 0 && (size_t) row < problem->A.rows)
                probe_rows[row] = 1;
        }
    }
    for (column = 0; column < problem->n; ++column)
    {
        int position;
        if (!presolver->is_fixed[column] &&
            !presolver->is_parallel_removed[column])
            continue;
        for (position = workspace->starts[column];
             position < workspace->ends[column]; ++position)
        {
            int row = workspace->rows[position];
            if (row >= 0 && (size_t) row < problem->A.rows)
                probe_rows[row] = 1;
        }
    }
    return probe_rows;
}

static size_t build_materialized_followup_probe_rows(
    const PreFOSPresolver *presolver,
    size_t first_bound_change, unsigned char *probe_rows,
    unsigned char *changed_columns)
{
    const PreFOSProblemData *problem = &presolver->original;
    const PresolveTransformationLog *log =
        &presolver->transformations;
    size_t count = 0;
    size_t row, record_index;

    if (!probe_rows || !changed_columns) return 0;
    if (problem->A.rows > 0)
        memset(
            probe_rows, 0,
            problem->A.rows * sizeof(unsigned char));
    if (problem->n > 0)
        memset(
            changed_columns, 0,
            problem->n * sizeof(unsigned char));
    if (presolver->rows_require_materialization)
        for (row = 0; row < problem->A.rows; ++row)
            if (presolver->rows_require_materialization[row] &&
                !presolver->remove_rows[row])
            {
                probe_rows[row] = 1;
                ++count;
            }

    for (record_index = first_bound_change;
         record_index < log->n_bound_changes; ++record_index)
    {
        int column = log->bound_changes[record_index].column;
        if (column < 0 || (size_t) column >= problem->n)
            continue;
        changed_columns[column] = 1;
    }
    for (row = 0; row < problem->A.rows; ++row)
    {
        int position;
        if (presolver->remove_rows[row] || probe_rows[row])
            continue;
        for (position = problem->A.row_pointers[row];
             position < problem->A.row_pointers[row + 1];
             ++position)
        {
            int column = problem->A.column_indices[position];
            if (problem->A.values[position] == 0.0 ||
                !changed_columns[column])
                continue;
            probe_rows[row] = 1;
            ++count;
            break;
        }
    }
    return count;
}

static int current_working_matrix_cache_available(
    const PreFOSPresolver *presolver);

static PreFOSStatus run_materialized_linear_propagation(
    PreFOSPresolver *presolver,
    PreFOSColumnWorkspace *column_workspace)
{
    PreFOSCsrMatrix original_matrix = presolver->original.A;
    const PreFOSCsrMatrix *source_matrix;
    const double *source_lower;
    const double *source_upper;
    double *original_lower = presolver->working_constraint_lower;
    double *original_upper = presolver->working_constraint_upper;
    PreFOSWorkingMatrix materialized;
    int original_remove_redundant_rows =
        presolver->settings.remove_redundant_rows;
    double original_event_queue_max_average_column_degree =
        presolver->settings.event_queue_max_average_column_degree;
    int classify_materialized_redundancy;
    int original_materialized =
        presolver->working_matrix_is_materialized;
    int original_hard_work_budget =
        presolver->linear_propagation_hard_work_budget;
    size_t *original_source_records =
        presolver->materialized_bound_source_records;
    size_t original_source_record_count =
        presolver->n_materialized_bound_source_records;
    const unsigned char *original_seed_rows =
        presolver->linear_propagation_seed_rows;
    size_t materialized_fixed_column_epoch;
    size_t materialized_column_transformations;
    unsigned char *probe_rows = NULL;
    unsigned char *probe_columns = NULL;
    int probe_before_full =
        presolver->stats.propagated_box_bounds == 0 ||
        presolver->linear_propagation_broad_frontier_stop ||
        current_working_matrix_cache_available(presolver);
    int reused_current_cache;
    int propagation_closed = 0;
    int trace_final_propagation =
        getenv("PREFOS_TRACE_FINAL_PROPAGATION") != NULL;
    PreFOSTimestamp trace_start, trace_stop;
    PreFOSStatus status;

    if (probe_before_full)
    {
        probe_rows =
            build_materialized_propagation_probe_rows(
                presolver, column_workspace);
        if (presolver->original.A.rows > 0 && !probe_rows)
            return PREFOS_STATUS_OUT_OF_MEMORY;
        if (column_workspace && presolver->original.n > 0)
            probe_columns = (unsigned char *) calloc(
                presolver->original.n, sizeof(unsigned char));
    }
    memset(&materialized, 0, sizeof(materialized));
    if (getenv("PREFOS_TRACE_WORKING_CACHE"))
        fprintf(
            stderr,
            "PreFOS propagation cache valid=%d fixed=%zu/%zu "
            "columns=%zu/%zu\n",
            presolver->cached_working_matrix_valid,
            presolver->cached_working_fixed_column_epoch,
            presolver->fixed_column_epoch,
            presolver->cached_working_column_transformations,
            presolver->transformations.n_column_transformations);
    reused_current_cache =
        prefos_internal_take_current_working_matrix_cache(
            presolver, &materialized, NULL);
    if (trace_final_propagation)
        prefos_internal_timer_now(&trace_start);
    if (reused_current_cache)
        status = PREFOS_STATUS_OK;
    else
    {
        prefos_internal_get_working_matrix_source(
            presolver, &source_matrix, &source_lower, &source_upper);
        status = prefos_internal_materialize_working_matrix(
            presolver, source_matrix, source_lower, source_upper,
            &materialized);
    }
    if (trace_final_propagation)
    {
        prefos_internal_timer_now(&trace_stop);
        fprintf(
            stderr,
            "PreFOS final-propagation materialize_ms=%.3f "
            "cache=%d rows=%zu nnz=%zu status=%d\n",
            prefos_internal_timer_elapsed_milliseconds(
                &trace_start, &trace_stop),
            reused_current_cache, materialized.matrix.rows,
            materialized.matrix.nnz, (int) status);
    }
    materialized_fixed_column_epoch = presolver->fixed_column_epoch;
    materialized_column_transformations =
        presolver->transformations.n_column_transformations;
    classify_materialized_redundancy =
        original_remove_redundant_rows;
    if (status == PREFOS_STATUS_OK)
    {
        int round;
        int probe_bound_changed = 1;
        size_t probe_bound_change_start =
            presolver->transformations.n_bound_changes;
        size_t initial_followup_rows = SIZE_MAX;
        size_t *source_records = (size_t *) calloc(
            materialized.matrix.rows, sizeof(size_t));
        if (materialized.matrix.rows > 0 && !source_records)
        {
            prefos_internal_free_working_matrix(&materialized);
            free(probe_rows);
            free(probe_columns);
            return PREFOS_STATUS_OUT_OF_MEMORY;
        }
        prefos_internal_free_linear_propagation_cache(presolver);
        prefos_internal_cuda_workspace_release(presolver);
        presolver->original.A = materialized.matrix;
        presolver->working_constraint_lower = materialized.lower;
        presolver->working_constraint_upper = materialized.upper;
        presolver->working_matrix_is_materialized = 1;
        presolver->linear_propagation_hard_work_budget =
            original_hard_work_budget ||
            (!getenv("PREFOS_DISABLE_HARD_TAIL_BUDGET") &&
             materialized.matrix.nnz >=
                 PREFOS_HARD_TAIL_PROPAGATION_MIN_NNZ &&
             presolver->stats.propagated_box_bounds > 0 &&
             presolver->settings
                     .linear_propagation_max_stale_rounds > 0);
        presolver->materialized_bound_source_records = source_records;
        presolver->n_materialized_bound_source_records =
            materialized.matrix.rows;
        presolver->settings.remove_redundant_rows =
            classify_materialized_redundancy;
        /*
         * Keep the normal degree-based backend selection here.  The
         * materialized matrix can remain active for several follow-up
         * waves, so forcing a full scan solely because it is large repeats
         * the most expensive traversal and discards the event cache.
         */
        if (probe_before_full)
        {
            if (trace_final_propagation)
                prefos_internal_timer_now(&trace_start);
            status = prefos_internal_probe_linear_rows(
                presolver, probe_rows, &probe_bound_changed);
            if (trace_final_propagation)
            {
                prefos_internal_timer_now(&trace_stop);
                fprintf(
                    stderr,
                    "PreFOS final-propagation probe_ms=%.3f "
                    "changed=%d status=%d\n",
                    prefos_internal_timer_elapsed_milliseconds(
                        &trace_start, &trace_stop),
                    probe_bound_changed, (int) status);
            }
            if (status == PREFOS_STATUS_OK && !probe_bound_changed)
                propagation_closed = 1;
            else if (
                status == PREFOS_STATUS_OK && probe_columns &&
                materialized.matrix.nnz >=
                    PREFOS_LAZY_MATERIALIZED_PROPAGATION_MIN_NNZ &&
                !getenv(
                    "PREFOS_FORCE_GLOBAL_MATERIALIZED_PROPAGATION") &&
                presolver->settings
                        .linear_propagation_max_stale_rounds > 0)
            {
                initial_followup_rows =
                    build_materialized_followup_probe_rows(
                        presolver, probe_bound_change_start,
                        probe_rows, probe_columns);
                if (trace_final_propagation)
                    fprintf(
                        stderr,
                        "PreFOS final-propagation seed_rows=%zu "
                        "of=%zu\n",
                        initial_followup_rows,
                        materialized.matrix.rows);
                if (initial_followup_rows == 0)
                {
                    probe_bound_changed = 0;
                    propagation_closed = 1;
                }
            }
        }
        if (status == PREFOS_STATUS_OK && probe_bound_changed &&
            probe_rows && probe_columns &&
            initial_followup_rows != SIZE_MAX &&
            initial_followup_rows > 0 &&
            initial_followup_rows <=
                (materialized.matrix.rows + 1) / 2 &&
            presolver->settings
                    .linear_propagation_max_stale_rounds > 0)
        {
            int probe_round;
            int max_probe_rounds =
                presolver->settings.max_linear_propagation_rounds;
            for (probe_round = 1;
                 probe_round < max_probe_rounds;
                 ++probe_round)
            {
                size_t first_followup_bound_change =
                    presolver->transformations.n_bound_changes;
                if (trace_final_propagation)
                    prefos_internal_timer_now(&trace_start);
                status = prefos_internal_probe_linear_rows(
                    presolver, probe_rows, &probe_bound_changed);
                if (trace_final_propagation)
                {
                    prefos_internal_timer_now(&trace_stop);
                    fprintf(
                        stderr,
                        "PreFOS final-propagation initial-followup=%d "
                        "rows=%zu probe_ms=%.3f changed=%d status=%d\n",
                        probe_round, initial_followup_rows,
                        prefos_internal_timer_elapsed_milliseconds(
                            &trace_start, &trace_stop),
                        probe_bound_changed, (int) status);
                }
                if (status != PREFOS_STATUS_OK)
                    break;
                if (!probe_bound_changed)
                {
                    propagation_closed = 1;
                    break;
                }
                initial_followup_rows =
                    build_materialized_followup_probe_rows(
                        presolver, first_followup_bound_change,
                        probe_rows, probe_columns);
                if (initial_followup_rows == 0)
                {
                    probe_bound_changed = 0;
                    propagation_closed = 1;
                    break;
                }
                if (initial_followup_rows >
                    (materialized.matrix.rows + 1) / 2)
                {
                    initial_followup_rows = SIZE_MAX;
                    break;
                }
            }
            if (status == PREFOS_STATUS_OK && propagation_closed)
            {
                size_t fixed = 0;
                status = prefos_internal_find_fixed_box_variables(
                    presolver, NULL, &fixed);
                if (fixed > 0)
                {
                    propagation_closed = 0;
                    probe_bound_changed = 1;
                    initial_followup_rows = SIZE_MAX;
                }
            }
        }
        for (round = 0;
             status == PREFOS_STATUS_OK && probe_bound_changed &&
             round < PREFOS_MAX_MEDIUM_FIXED_POINT_ROUNDS;
             ++round)
        {
            size_t fixed = 0;
            size_t propagated_before =
                presolver->stats.propagated_box_bounds;
            size_t first_followup_bound_change =
                presolver->transformations.n_bound_changes;
            int original_max_rounds =
                presolver->settings.max_linear_propagation_rounds;
            int adaptive_followup =
                probe_rows && probe_columns &&
                original_max_rounds > 1 &&
                presolver->settings
                        .linear_propagation_max_stale_rounds > 0;
            int needs_global_retry = 0;
            int followup_changed = 0;
            int wave_closed;
            if (adaptive_followup)
                presolver->settings.max_linear_propagation_rounds = 1;
            if (round == 0 && probe_rows &&
                initial_followup_rows != SIZE_MAX &&
                initial_followup_rows > 0 &&
                initial_followup_rows <=
                    (materialized.matrix.rows + 1) / 2)
                presolver->linear_propagation_seed_rows =
                    probe_rows;
            if (trace_final_propagation)
                prefos_internal_timer_now(&trace_start);
            status = prefos_internal_propagate_linear_bounds(
                presolver, NULL);
            presolver->linear_propagation_seed_rows =
                original_seed_rows;
            wave_closed = presolver->linear_propagation_complete;
            presolver->settings.max_linear_propagation_rounds =
                original_max_rounds;
            if (trace_final_propagation)
            {
                prefos_internal_timer_now(&trace_stop);
                fprintf(
                    stderr,
                    "PreFOS final-propagation round=%d "
                    "propagate_ms=%.3f delta_bounds=%zu status=%d\n",
                    round,
                    prefos_internal_timer_elapsed_milliseconds(
                        &trace_start, &trace_stop),
                    presolver->stats.propagated_box_bounds -
                        propagated_before,
                    (int) status);
            }
            if (status != PREFOS_STATUS_OK) break;
            if (adaptive_followup &&
                presolver->transformations.n_bound_changes >
                    first_followup_bound_change)
            {
                int probe_round;
                followup_changed = 1;
                for (probe_round = 1;
                     probe_round < original_max_rounds;
                     ++probe_round)
                {
                    size_t candidate_rows =
                        build_materialized_followup_probe_rows(
                            presolver,
                            first_followup_bound_change,
                            probe_rows, probe_columns);
                    if (candidate_rows == 0)
                    {
                        followup_changed = 0;
                        break;
                    }
                    if (candidate_rows >
                        (materialized.matrix.rows + 1) / 2)
                    {
                        needs_global_retry = 1;
                        break;
                    }
                    first_followup_bound_change =
                        presolver->transformations.n_bound_changes;
                    if (trace_final_propagation)
                        prefos_internal_timer_now(&trace_start);
                    status = prefos_internal_probe_linear_rows(
                        presolver, probe_rows,
                        &followup_changed);
                    if (trace_final_propagation)
                    {
                        prefos_internal_timer_now(&trace_stop);
                        fprintf(
                            stderr,
                            "PreFOS final-propagation followup=%d "
                            "rows=%zu probe_ms=%.3f changed=%d status=%d\n",
                            probe_round, candidate_rows,
                            prefos_internal_timer_elapsed_milliseconds(
                                &trace_start, &trace_stop),
                            followup_changed, (int) status);
                    }
                    if (status != PREFOS_STATUS_OK ||
                        !followup_changed)
                        break;
                }
                if (status == PREFOS_STATUS_OK &&
                    followup_changed)
                    needs_global_retry = 1;
                wave_closed =
                    status == PREFOS_STATUS_OK &&
                    !followup_changed &&
                    !needs_global_retry;
            }
            if (status != PREFOS_STATUS_OK) break;
            if (trace_final_propagation)
                prefos_internal_timer_now(&trace_start);
            status = prefos_internal_find_fixed_box_variables(
                presolver, NULL, &fixed);
            if (trace_final_propagation)
            {
                prefos_internal_timer_now(&trace_stop);
                fprintf(
                    stderr,
                    "PreFOS final-propagation round=%d "
                    "fixing_ms=%.3f fixed=%zu status=%d\n",
                    round,
                    prefos_internal_timer_elapsed_milliseconds(
                        &trace_start, &trace_stop),
                    fixed, (int) status);
            }
            if (status != PREFOS_STATUS_OK)
                break;
            if (fixed == 0 && needs_global_retry &&
                presolver->linear_propagation_hard_work_budget)
            {
                ++presolver->stats.linear_budget_stops;
                break;
            }
            if (fixed == 0 && !needs_global_retry)
            {
                if (wave_closed) propagation_closed = 1;
                break;
            }
        }
        prefos_internal_free_linear_propagation_cache(presolver);
        prefos_internal_cuda_workspace_release(presolver);
        presolver->settings.remove_redundant_rows =
            original_remove_redundant_rows;
        presolver->settings.event_queue_max_average_column_degree =
            original_event_queue_max_average_column_degree;
        presolver->working_matrix_is_materialized =
            original_materialized;
        presolver->linear_propagation_hard_work_budget =
            original_hard_work_budget;
        presolver->materialized_bound_source_records =
            original_source_records;
        presolver->n_materialized_bound_source_records =
            original_source_record_count;
        presolver->linear_propagation_seed_rows =
            original_seed_rows;
        presolver->original.A = original_matrix;
        presolver->working_constraint_lower = original_lower;
        presolver->working_constraint_upper = original_upper;
        presolver->linear_propagation_complete =
            status == PREFOS_STATUS_OK && propagation_closed;
        if (presolver->linear_propagation_complete)
            presolver->linear_propagation_bound_cursor =
                presolver->transformations.n_bound_changes;
        free(source_records);
    }
    if (status == PREFOS_STATUS_OK &&
        materialized_fixed_column_epoch ==
            presolver->fixed_column_epoch &&
        materialized_column_transformations ==
            presolver->transformations.n_column_transformations)
        prefos_internal_store_working_matrix_cache(
            presolver, &materialized);
    else
        prefos_internal_free_working_matrix(&materialized);
    free(probe_rows);
    free(probe_columns);
    return status;
}

static PreFOSStatus run_medium_reduction_pass(
    PreFOSPresolver *presolver, int include_parallel_rows,
    int include_row_redundancy,
    int *changed_after_linear,
    int *parallel_rows_reduced,
    PreFOSColumnWorkspace *shared_workspace)
{
    PreFOSFastTriggerSignature before_linear;
    PreFOSFastTriggerSignature after_linear;
    PreFOSFastTriggerSignature after_cone;
    PreFOSTimestamp start, stop;
    PreFOSStatus status;
    int defer_integrated_redundancy =
        include_row_redundancy &&
        presolver->settings.remove_redundant_rows;
    int redundancy_was_completed =
        presolver->scalar_redundancy_completed;

    *changed_after_linear = 0;
    if (parallel_rows_reduced) *parallel_rows_reduced = 0;
    trace_medium_stage(presolver, "before-linear");
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
        redundancy_was_completed =
            presolver->scalar_redundancy_completed;
        trace_medium_stage(presolver, "after-row-activity");
    }
    if (shared_workspace)
        prefos_internal_update_column_live_degrees(
            presolver, shared_workspace);
    if (defer_integrated_redundancy)
        presolver->scalar_redundancy_completed = 1;
    before_linear = capture_fast_trigger_signature(presolver);
    prefos_internal_timer_now(&start);
    status = prefos_internal_propagate_linear_bounds(
        presolver, shared_workspace);
    prefos_internal_timer_now(&stop);
    if (defer_integrated_redundancy)
        presolver->scalar_redundancy_completed =
            redundancy_was_completed;
    presolver->stats.linear_propagation_milliseconds +=
        prefos_internal_timer_elapsed_milliseconds(&start, &stop);
    if (status != PREFOS_STATUS_OK) return status;
    if (presolver->linear_propagation_complete)
        presolver->linear_propagation_bound_cursor =
            presolver->transformations.n_bound_changes;
    trace_medium_stage(presolver, "after-linear");
    after_linear = capture_fast_trigger_signature(presolver);
    *changed_after_linear =
        fast_trigger_signature_changed(before_linear, after_linear);
    if (after_linear.changed_bounds != before_linear.changed_bounds)
        presolver->scalar_redundancy_completed = 0;

    /*
     * Propagation and activity reductions can expose parallel rows.  Run
     * detection afterwards so the same medium wave can consume them and feed
     * the resulting singleton columns into the following fast fixed point.
     */
    if (include_parallel_rows)
    {
        size_t removed_before_parallel =
            presolver->n_removed_rows;
        status = run_medium_parallel_rows(
            presolver, shared_workspace);
        if (status != PREFOS_STATUS_OK) return status;
        if (parallel_rows_reduced)
            *parallel_rows_reduced =
                presolver->n_removed_rows >
                removed_before_parallel;
        trace_medium_stage(presolver, "after-parallel-rows");
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
    after_cone = capture_fast_trigger_signature(presolver);
    *changed_after_linear |=
        fast_trigger_signature_changed(after_linear, after_cone);
    if (after_cone.changed_bounds != after_linear.changed_bounds)
        presolver->scalar_redundancy_completed = 0;
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

static size_t parallel_row_retry_threshold(
    const PreFOSPresolver *presolver)
{
    size_t eliminated = presolver->n_fixed_columns;
    size_t active;
    size_t threshold;
    eliminated = saturated_change_sum(
        eliminated,
        presolver->stats.substituted_free_variables);
    eliminated = saturated_change_sum(
        eliminated,
        presolver->n_parallel_column_reductions);
    active = eliminated < presolver->original.n
                 ? presolver->original.n - eliminated
                 : 0;
    threshold = (active + 9) / 10;
    if (threshold == 0) threshold = 1;
    if (threshold > 4096) threshold = 4096;
    return threshold;
}

static PreFOSStatus run_scalar_tail_fixed_point(
    PreFOSPresolver *presolver,
    PreFOSColumnWorkspace *shared_workspace,
    int include_parallel_rows)
{
    size_t round;
    for (round = 0; round < 4; ++round)
    {
        PreFOSFastTriggerSignature before =
            capture_fast_trigger_signature(presolver);
        PreFOSFastTriggerSignature after_medium;
        PreFOSFastTriggerSignature after;
        size_t fixed_before = presolver->n_fixed_columns;
        size_t substituted_before =
            presolver->stats.substituted_free_variables;
        size_t parallel_before =
            presolver->n_parallel_column_reductions;
        int changed_after_linear = 0;
        int parallel_rows_reduced = 0;
        PreFOSStatus status;

        ++presolver->stats.medium_fixed_point_rounds;
        status = run_medium_reduction_pass(
            presolver, include_parallel_rows, 1,
            &changed_after_linear, &parallel_rows_reduced,
            shared_workspace);
        if (status != PREFOS_STATUS_OK) return status;
        after_medium = capture_fast_trigger_signature(presolver);
        if (fast_trigger_signature_changed(before, after_medium))
        {
            status = run_fast_fixed_point_timed(
                presolver, 1, 0, shared_workspace);
            if (status != PREFOS_STATUS_OK) return status;
        }
        after = capture_fast_trigger_signature(presolver);
        include_parallel_rows =
            parallel_rows_reduced &&
            (presolver->n_fixed_columns != fixed_before ||
             presolver->stats.substituted_free_variables !=
                 substituted_before ||
             presolver->n_parallel_column_reductions != parallel_before);
        if (!fast_trigger_signature_changed(before, after))
            break;
        if (medium_fixed_point_round_is_stale(
                presolver, before, after, fixed_before,
                substituted_before, parallel_before))
            break;
    }
    return PREFOS_STATUS_OK;
}

static size_t transformed_support_incident_nnz(
    const PreFOSPresolver *presolver,
    const PreFOSColumnWorkspace *workspace)
{
    size_t incident_nnz = 0;
    size_t column;
    for (column = 0; column < presolver->original.n; ++column)
    {
        if (!presolver->is_fixed[column] &&
            !presolver->is_substituted[column] &&
            !presolver->is_parallel_removed[column])
            continue;
        size_t degree =
            (size_t) (workspace->ends[column] - workspace->starts[column]);
        if (degree > SIZE_MAX - incident_nnz) return SIZE_MAX;
        incident_nnz += degree;
    }
    return incident_nnz;
}

static int should_run_materialized_parallel_rows(
    const PreFOSPresolver *presolver,
    const PreFOSColumnWorkspace *workspace)
{
    const PreFOSProblemData *problem = &presolver->original;
    size_t incident_nnz, minimum_yield, estimated_remaining;
    int manageable_dense_cpu_model;
    long double density_limit;
    if (!presolver->settings.remove_redundant_rows ||
        !workspace || problem->A.rows < 2)
        return 0;
    incident_nnz = transformed_support_incident_nnz(
        presolver, workspace);
    manageable_dense_cpu_model =
        problem->A.rows <= 4096 &&
        problem->A.nnz <= 50000000;
    if (manageable_dense_cpu_model)
        return incident_nnz > 0;
    minimum_yield =
        problem->A.nnz <= 65536
            ? 1
            : (problem->A.nnz + 63) / 64;
    if (incident_nnz < minimum_yield)
        return incident_nnz > 0 &&
               (problem->A.nnz <= 1000000 ||
                incident_nnz >= (minimum_yield + 7) / 8) &&
               problem->A.nnz <= 10000000 &&
               (long double) problem->A.nnz <=
                   64.0L * (long double) problem->A.rows &&
               prefos_internal_filtered_parallel_support_is_promising(
                   presolver);
    if (presolver->settings.parallel_row_max_average_nnz == 0.0)
        return 1;
    estimated_remaining =
        incident_nnz < problem->A.nnz
            ? problem->A.nnz - incident_nnz
            : 0;
    density_limit =
        (long double) problem->A.rows *
        (long double)
            presolver->settings.parallel_row_max_average_nnz;
    return (long double) estimated_remaining <= density_limit;
}

static size_t active_original_row_degree(
    const PreFOSPresolver *presolver, size_t row)
{
    const PreFOSCsrMatrix *matrix = &presolver->original.A;
    size_t degree = 0;
    int position;
    for (position = matrix->row_pointers[row];
         position < matrix->row_pointers[row + 1]; ++position)
    {
        int column = matrix->column_indices[position];
        if (matrix->values[position] != 0.0 &&
            prefos_internal_term_is_active_in_row(
                presolver, row, column))
            ++degree;
    }
    return degree;
}

static int bounded_doubleton_chain_is_promising(
    const PreFOSPresolver *presolver)
{
    int use_log =
        presolver->materialization_row_log_complete &&
        presolver->n_materialization_row_log ==
            presolver->n_rows_require_materialization;
    size_t position;
    size_t count =
        use_log ? presolver->n_materialization_row_log
                : presolver->original.A.rows;
    for (position = 0; position < count; ++position)
    {
        int row =
            use_log ? presolver->materialization_row_log[position]
                    : (int) position;
        if (row < 0 ||
            (size_t) row >= presolver->original.A.rows ||
            !presolver->rows_require_materialization[row] ||
            presolver->remove_rows[row])
            continue;
        if (isfinite(
                presolver->working_constraint_lower[row]) &&
            presolver->working_constraint_lower[row] ==
                presolver->working_constraint_upper[row] &&
            active_original_row_degree(
                presolver, (size_t) row) <= 2)
            return 1;
    }
    return 0;
}

static int should_run_materialized_first_bounded(
    const PreFOSPresolver *presolver,
    const PreFOSColumnWorkspace *workspace,
    size_t *candidate_count, size_t *candidate_threshold)
{
    const PreFOSProblemData *problem = &presolver->original;
    const char *force =
        getenv("PREFOS_FORCE_MATERIALIZED_FIRST_BOUNDED");
    const char *disable =
        getenv("PREFOS_DISABLE_MATERIALIZED_FIRST_BOUNDED");
    int trace =
        getenv("PREFOS_TRACE_BOUNDED_SCHEDULER") != NULL;
    size_t threshold;
    size_t candidates = 0;
    size_t row;

    if (candidate_count) *candidate_count = 0;
    if (candidate_threshold) *candidate_threshold = 0;
    if (force && *force && *force != '0')
        return workspace != NULL;
    if ((disable && *disable && *disable != '0') ||
        !workspace ||
        presolver->settings.linear_propagation_max_stale_rounds == 0 ||
        problem->A.nnz < PREFOS_MATERIALIZED_FIRST_BOUNDED_MIN_NNZ)
        return 0;

    threshold = (problem->A.nnz + 127) / 128;
    if (threshold < 256) threshold = 256;
    if (problem->A.rows < problem->n)
    {
        size_t broad_frontier = (problem->n + 3) / 4;
        if (broad_frontier < 262144)
            broad_frontier = 262144;
        if (threshold < broad_frontier)
            threshold = broad_frontier;
    }
    if (candidate_threshold) *candidate_threshold = threshold;

    for (row = 0; row < problem->A.rows; ++row)
    {
        double lower;
        if (presolver->remove_rows[row] ||
            workspace->dirty_row[row] ||
            workspace->row_degrees[row] != 2)
            continue;
        lower = presolver->working_constraint_lower[row];
        if (!isfinite(lower) ||
            lower != presolver->working_constraint_upper[row])
            continue;
        ++candidates;
        if (candidates >= threshold && !trace)
            break;
    }
    if (candidate_count) *candidate_count = candidates;
    return candidates >= threshold;
}

static int current_working_matrix_cache_available(
    const PreFOSPresolver *presolver)
{
    return presolver->cached_working_matrix_valid &&
           presolver->cached_working_fixed_column_epoch ==
               presolver->fixed_column_epoch &&
           presolver->cached_working_column_transformations ==
               presolver->transformations.n_column_transformations;
}

static int should_run_post_substitution_propagation(
    const PreFOSPresolver *presolver)
{
    const PreFOSProblemData *problem = &presolver->original;
    const char *override =
        getenv("PREFOS_FINAL_MATERIALIZED_PROPAGATION");
    size_t structural_replacements, eliminated, remaining, minimum_yield;
    size_t post_frontier_transformations;
    size_t minimum_dirty_rows, minimum_dirty_nnz;

    if (presolver->stats.substituted_free_variables >
        SIZE_MAX - presolver->n_parallel_column_reductions)
        structural_replacements = SIZE_MAX;
    else
        structural_replacements =
            presolver->stats.substituted_free_variables +
            presolver->n_parallel_column_reductions;
    if (!presolver->settings.linear_propagation ||
        structural_replacements == 0 ||
        (presolver->n_rows_require_materialization == 0 &&
         presolver->n_parallel_column_reductions == 0))
        return 0;
    if (override && *override)
        return *override != '0';
    minimum_yield = (problem->A.nnz + 1023) / 1024;
    if (presolver->linear_propagation_broad_frontier_stop &&
        presolver->linear_propagation_broad_frontier_fixed_epoch ==
            presolver->fixed_column_epoch)
    {
        size_t frontier_transformations =
            presolver
                ->linear_propagation_broad_frontier_column_transformations;
        size_t current_transformations =
            presolver->transformations.n_column_transformations;
        post_frontier_transformations =
            current_transformations > frontier_transformations
                ? current_transformations - frontier_transformations
                : 0;
        if (post_frontier_transformations < minimum_yield)
            return 0;
    }
    if (problem->A.nnz <= 131072) return 1;

    if (structural_replacements < minimum_yield)
        return 0;
    minimum_dirty_rows = (problem->A.rows + 2047) / 2048;
    minimum_dirty_nnz = (problem->A.nnz + 2047) / 2048;
    if (presolver->n_parallel_column_reductions == 0 &&
        presolver->n_rows_require_materialization <
            minimum_dirty_rows &&
        presolver->materialization_source_nnz <
            minimum_dirty_nnz)
        return 0;
    if (current_working_matrix_cache_available(presolver))
        return 1;

    eliminated = presolver->n_fixed_columns;
    if (presolver->stats.substituted_free_variables >
        SIZE_MAX - eliminated)
        return 0;
    eliminated += presolver->stats.substituted_free_variables;
    if (presolver->n_parallel_column_reductions >
        SIZE_MAX - eliminated)
        return 0;
    eliminated += presolver->n_parallel_column_reductions;
    remaining =
        eliminated < problem->n ? problem->n - eliminated : 0;
    return remaining >= (problem->n + 3) / 4;
}

static PreFOSStatus run_materialized_parallel_row_closure(
    PreFOSPresolver *presolver, PreFOSColumnWorkspace *workspace,
    int force)
{
    const PreFOSCsrMatrix *source_matrix;
    const double *source_lower;
    const double *source_upper;
    PreFOSWorkingMatrix working;
    PreFOSTimestamp start, stop;
    PreFOSStatus status;
    size_t removed_before = presolver->n_removed_rows;
    int parallel_rows_checked = 0;
    int reused_current_cache;
    if (!force &&
        !should_run_materialized_parallel_rows(
            presolver, workspace))
        return PREFOS_STATUS_OK;
    memset(&working, 0, sizeof(working));
    prefos_internal_timer_now(&start);
    reused_current_cache =
        prefos_internal_take_current_working_matrix_cache(
            presolver, &working, &parallel_rows_checked);
    if (reused_current_cache)
        status = PREFOS_STATUS_OK;
    else
    {
        prefos_internal_get_working_matrix_source(
            presolver, &source_matrix, &source_lower, &source_upper);
        status = prefos_internal_materialize_working_matrix(
            presolver, source_matrix, source_lower, source_upper,
            &working);
    }
    prefos_internal_timer_now(&stop);
    presolver->stats.structural_reduction_milliseconds +=
        prefos_internal_timer_elapsed_milliseconds(&start, &stop);
    if (status != PREFOS_STATUS_OK) return status;
    if (parallel_rows_checked)
    {
        prefos_internal_store_working_matrix_cache(
            presolver, &working);
        presolver->cached_working_parallel_rows_checked = 1;
        return PREFOS_STATUS_OK;
    }
    status = prefos_internal_remove_parallel_rows_in_working_matrix(
        presolver, &working.matrix, working.lower, working.upper,
        workspace);
    if (status == PREFOS_STATUS_OK)
    {
        prefos_internal_store_working_matrix_cache(presolver, &working);
        presolver->cached_working_parallel_rows_checked = 1;
    }
    else
        prefos_internal_free_working_matrix(&working);
    if (status != PREFOS_STATUS_OK) return status;
    if (presolver->n_removed_rows > removed_before)
    {
        size_t pending_parallel_rows_removed = 0;
        presolver->scalar_redundancy_completed = 0;
        status = run_fast_fixed_point_timed(
            presolver, 1, 0, workspace);
        if (status == PREFOS_STATUS_OK &&
            getenv("PREFOS_FORCE_MATERIALIZED_PARALLEL_CASCADE"))
        {
            status = prefos_internal_run_materialized_column_closure(
                presolver, 2, 0, 0,
                &pending_parallel_rows_removed, 0);
            if (status == PREFOS_STATUS_OK)
            {
                presolver->scalar_redundancy_completed = 0;
                status = run_materialized_linear_propagation(
                    presolver, workspace);
            }
            if (status == PREFOS_STATUS_OK)
                status = run_fast_fixed_point_timed(
                    presolver, 1, 0, workspace);
        }
    }
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
    double *cached_linear_objective = NULL;
    double cached_linear_objective_offset = 0.0;
    int shared_column_workspace_valid = 0;
    int cached_linear_objective_valid = 0;
    int early_parallel_signature_valid = 0;
    size_t early_parallel_materialized_rows = 0;
    size_t early_parallel_removed_rows = 0;
    size_t early_parallel_objective_epoch = 0;
    const char *failure_stage = "initialization";
    PreFOSFastTriggerSignature early_parallel_signature = {0};
    PreFOSTimestamp presolve_start, phase_start, phase_stop;

    if (!presolver) return PREFOS_STATUS_INVALID_ARGUMENT;
    memset(&shared_column_workspace, 0, sizeof(shared_column_workspace));
    prefos_internal_free_linear_propagation_cache(presolver);
    prefos_internal_cuda_workspace_release(presolver);
    prefos_internal_clear_working_matrix_cache(presolver);
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
    free(presolver->substitution_fill_in_targets);
    free(presolver->substitution_keeps_source_row);
    free(presolver->substitution_term_start);
    free(presolver->substitution_source_row);
    free(presolver->residual_source_column);
    free(presolver->rows_require_materialization);
    free(presolver->materialization_row_log);
    free(presolver->materialized_bound_source_records);
    free(presolver->substitution_constant);
    free(presolver->substitution_targets);
    free(presolver->substitution_scales);
    free(presolver->variable_to_box);
    free(presolver->working_box_lower);
    free(presolver->working_box_upper);
    free(presolver->latest_lower_bound_change);
    free(presolver->latest_upper_bound_change);
    free(presolver->fixed_box_dirty);
    free(presolver->fixed_box_dirty_queue);
    free(presolver->working_constraint_lower);
    free(presolver->working_constraint_upper);
    free(presolver->propagation_lower);
    free(presolver->propagation_upper);
    free(presolver->nonmaterialized_bound_source_rows);
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
    presolver->substitution_fill_in_targets = NULL;
    presolver->substitution_keeps_source_row = NULL;
    presolver->substitution_term_start = NULL;
    presolver->substitution_source_row = NULL;
    presolver->residual_source_column = NULL;
    presolver->rows_require_materialization = NULL;
    presolver->materialization_row_log = NULL;
    presolver->n_materialization_row_log = 0;
    presolver->materialization_row_log_capacity = 0;
    presolver->materialization_row_log_complete = 0;
    presolver->n_rows_require_materialization = 0;
    presolver->materialization_source_nnz = 0;
    presolver->working_matrix_is_materialized = 0;
    presolver->materialized_row_updates_require_cache = 0;
    presolver->materialized_bound_source_records = NULL;
    presolver->n_materialized_bound_source_records = 0;
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
    presolver->latest_lower_bound_change = NULL;
    presolver->latest_upper_bound_change = NULL;
    presolver->fixed_box_dirty = NULL;
    presolver->fixed_box_dirty_queue = NULL;
    presolver->n_fixed_box_dirty = 0;
    presolver->working_constraint_lower = NULL;
    presolver->working_constraint_upper = NULL;
    presolver->propagation_lower = NULL;
    presolver->propagation_upper = NULL;
    presolver->linear_propagation_seed_rows = NULL;
    presolver->nonmaterialized_bound_source_rows = NULL;
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
    presolver->source_parallel_rows_closed = 0;
    presolver->materialized_parallel_rows_signature_valid = 0;
    presolver->materialized_parallel_rows_snapshot_sorted = 0;
    presolver->materialized_parallel_rows_fixed_column_epoch = 0;
    presolver->materialized_parallel_rows_column_transformations = 0;
    presolver->materialized_parallel_rows_sorted_count = 0;
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
            presolver, NULL, &ignored_fixed);
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
    status = run_fast_fixed_point_timed(
        presolver,
        presolver->settings.linear_propagation != 0 &&
            !presolver->settings.parallel_column_reduction,
        1,
        shared_column_workspace_valid ? &shared_column_workspace : NULL);
    if (status != PREFOS_STATUS_OK) goto failure;
    /*
     * Keep finite-bounded singleton columns until the first parallel-column
     * pass.  Several singleton columns in the same row are parallel; eagerly
     * converting each one to a residual row hides that aggregate structure.
     * The column pass seeds the deferred one-sided singleton candidates after
     * consuming the groups, so the following fast closure retains the former
     * behavior without another global candidate scan.
     */
    if (source->A.rows > 0 && source->n >= 2 &&
        shared_column_workspace_valid &&
        presolver->settings.parallel_column_reduction &&
        !getenv("PREFOS_DISABLE_EARLY_PARALLEL_COLUMNS"))
    {
        PreFOSFastTriggerSignature before_early_parallel =
            capture_fast_trigger_signature(presolver);
        prefos_internal_timer_now(&phase_start);
        status = shared_column_workspace_valid
                     ? prefos_internal_reduce_parallel_columns_in_workspace(
                           presolver, &shared_column_workspace)
                     : prefos_internal_reduce_parallel_columns(presolver);
        prefos_internal_timer_now(&phase_stop);
        presolver->stats.parallel_column_reduction_milliseconds +=
            prefos_internal_timer_elapsed_milliseconds(
                &phase_start, &phase_stop);
        if (status != PREFOS_STATUS_OK) goto failure;
        if (fast_trigger_signature_changed(
                before_early_parallel,
                capture_fast_trigger_signature(presolver)))
        {
            status = run_fast_fixed_point_timed(
                presolver, 0, 0,
                shared_column_workspace_valid
                    ? &shared_column_workspace
                    : NULL);
            if (status != PREFOS_STATUS_OK) goto failure;
        }
        early_parallel_signature =
            capture_fast_trigger_signature(presolver);
        early_parallel_materialized_rows =
            presolver->n_rows_require_materialization;
        early_parallel_removed_rows =
            presolver->n_removed_rows;
        early_parallel_objective_epoch =
            shared_column_workspace.objective_change_epoch;
        early_parallel_signature_valid = 1;
    }
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
        int include_parallel_rows = 1;
        int last_parallel_row_pass_reduced = 0;
        size_t structural_changes_since_parallel_rows = 0;
        int defer_scalar_activity_rescans =
            source->n_cones == 0 &&
            source->n_affine_cones == 0 &&
            presolver->settings.remove_redundant_rows &&
            presolver->settings.linear_propagation &&
            presolver->settings.propagated_bound_policy ==
                PREFOS_PROPAGATED_BOUND_POLICY_FIRST_ORDER;
        PreFOSFastTriggerSignature before_scalar_waves =
            capture_fast_trigger_signature(presolver);
        size_t fixed_before_scalar_waves =
            presolver->n_fixed_columns;
        size_t substituted_before_scalar_waves =
            presolver->stats.substituted_free_variables;
        size_t parallel_before_scalar_waves =
            presolver->n_parallel_column_reductions;
        prefos_internal_timer_now(&phase_start);
        for (round = 0; round < PREFOS_MAX_MEDIUM_FIXED_POINT_ROUNDS;
             ++round)
        {
            PreFOSFastTriggerSignature before_medium =
                capture_fast_trigger_signature(presolver);
            PreFOSFastTriggerSignature before_fast;
            PreFOSFastTriggerSignature after_round;
            int changed_after_linear = 0;
            int parallel_rows_reduced = 0;
            int parallel_rows_requested =
                include_parallel_rows;
            int fast_changed = 0;
            int fast_fixed_columns = 0;
            int fast_structural_changed = 0;
            int stale_round;
            size_t fixed_before_medium =
                presolver->n_fixed_columns;
            size_t substituted_before_medium =
                presolver->stats.substituted_free_variables;
            size_t parallel_before_medium =
                presolver->n_parallel_column_reductions;
            size_t substituted_before_fast;
            size_t parallel_columns_before_fast;
            size_t fixed_epoch_before_fast;
            ++presolver->stats.medium_fixed_point_rounds;

            status = run_medium_reduction_pass(
                presolver, include_parallel_rows,
                !defer_scalar_activity_rescans,
                &changed_after_linear,
                &parallel_rows_reduced,
                shared_column_workspace_valid
                    ? &shared_column_workspace
                    : NULL);
            if (status != PREFOS_STATUS_OK) goto failure;
            if (parallel_rows_requested)
            {
                structural_changes_since_parallel_rows = 0;
                last_parallel_row_pass_reduced =
                    parallel_rows_reduced;
            }
            if (getenv("PREFOS_FORCE_MEDIUM_PARALLEL_COLUMNS"))
            {
                PreFOSTimestamp parallel_start, parallel_stop;
                size_t pending_parallel_rows_removed = 0;
                prefos_internal_timer_now(&parallel_start);
                if (getenv(
                        "PREFOS_FORCE_MEDIUM_MATERIALIZED_COLUMNS"))
                    status =
                        prefos_internal_run_materialized_column_closure(
                            presolver, 3, 1, 0,
                            &pending_parallel_rows_removed, 0);
                else
                    status =
                        shared_column_workspace_valid
                            ? prefos_internal_reduce_parallel_columns_in_workspace(
                                  presolver,
                                  &shared_column_workspace)
                            : prefos_internal_reduce_parallel_columns(
                                  presolver);
                prefos_internal_timer_now(&parallel_stop);
                presolver->stats.parallel_column_reduction_milliseconds +=
                    prefos_internal_timer_elapsed_milliseconds(
                        &parallel_start, &parallel_stop);
                if (status != PREFOS_STATUS_OK) goto failure;
            }
            if (source->A.rows == 0 || source->n_box == 0)
                changed_after_linear = 0;
            before_fast = capture_fast_trigger_signature(presolver);
            fixed_epoch_before_fast = presolver->fixed_column_epoch;
            substituted_before_fast =
                presolver->stats.substituted_free_variables;
            parallel_columns_before_fast =
                presolver->n_parallel_column_reductions;
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
                fast_structural_changed =
                    presolver->stats.substituted_free_variables !=
                        substituted_before_fast ||
                    presolver->n_parallel_column_reductions !=
                        parallel_columns_before_fast ||
                    fast_fixed_columns;
                if (fast_structural_changed)
                    presolver->linear_propagation_broad_frontier_stop = 0;
            }
            structural_changes_since_parallel_rows =
                saturated_change_sum(
                    structural_changes_since_parallel_rows,
                    presolver->n_fixed_columns -
                        fixed_before_medium);
            structural_changes_since_parallel_rows =
                saturated_change_sum(
                    structural_changes_since_parallel_rows,
                    presolver->stats.substituted_free_variables -
                        substituted_before_medium);
            structural_changes_since_parallel_rows =
                saturated_change_sum(
                    structural_changes_since_parallel_rows,
                    presolver->n_parallel_column_reductions -
                        parallel_before_medium);
            /*
             * A productive parallel-row pass is allowed to cascade
             * immediately.  After a miss, wait until enough column support
             * has disappeared to justify another global signature scan.
             */
            include_parallel_rows =
                fast_structural_changed &&
                (last_parallel_row_pass_reduced ||
                 structural_changes_since_parallel_rows >=
                     parallel_row_retry_threshold(presolver));
            after_round = capture_fast_trigger_signature(presolver);
            stale_round =
                medium_fixed_point_round_is_stale(
                    presolver, before_medium, after_round,
                    fixed_before_medium, substituted_before_medium,
                    parallel_before_medium);
            if (getenv("PREFOS_TRACE_MEDIUM_FIXED_POINT"))
                fprintf(
                    stderr,
                    "PreFOS medium round=%d events=%zu rows=%zu "
                    "bounds=%zu cones=%zu fixed=%zu substituted=%zu "
                    "parallel=%zu stale=%d\n",
                    round,
                    after_round.transformation_events -
                        before_medium.transformation_events,
                    after_round.removed_rows -
                        before_medium.removed_rows,
                    after_round.changed_bounds -
                        before_medium.changed_bounds,
                    after_round.cone_reductions -
                        before_medium.cone_reductions,
                    presolver->n_fixed_columns -
                        fixed_before_medium,
                    presolver->stats.substituted_free_variables -
                        substituted_before_medium,
                    presolver->n_parallel_column_reductions -
                        parallel_before_medium,
                    stale_round);
            if (source->n_cones == 0 &&
                source->n_affine_cones == 0)
            {
                if (defer_scalar_activity_rescans)
                    presolver->scalar_redundancy_completed =
                        !fast_structural_changed &&
                        after_round.changed_bounds ==
                            before_medium.changed_bounds;
                else if (fast_fixed_columns)
                    presolver->scalar_redundancy_completed = 0;
                if (!fast_structural_changed)
                    break;
            }
            if (!changed_after_linear && !fast_changed) break;
            if (round > 0 && stale_round) break;
        }
        if (status == PREFOS_STATUS_OK &&
            defer_scalar_activity_rescans)
        {
            PreFOSFastTriggerSignature after_scalar_waves =
                capture_fast_trigger_signature(presolver);
            int activity_inputs_changed =
                after_scalar_waves.changed_bounds !=
                    before_scalar_waves.changed_bounds ||
                presolver->n_fixed_columns !=
                    fixed_before_scalar_waves ||
                presolver->stats.substituted_free_variables !=
                    substituted_before_scalar_waves ||
                presolver->n_parallel_column_reductions !=
                    parallel_before_scalar_waves;
            if (activity_inputs_changed &&
                !presolver->scalar_redundancy_completed)
            {
                PreFOSFastTriggerSignature before_final_activity =
                    after_scalar_waves;
                PreFOSTimestamp activity_start, activity_stop;
                presolver->scalar_redundancy_completed = 0;
                prefos_internal_timer_now(&activity_start);
                status =
                    prefos_internal_remove_redundant_rows_by_activity(
                        presolver,
                        shared_column_workspace_valid
                            ? &shared_column_workspace
                            : NULL);
                prefos_internal_timer_now(&activity_stop);
                presolver->stats.redundant_row_activity_milliseconds +=
                    prefos_internal_timer_elapsed_milliseconds(
                        &activity_start, &activity_stop);
                if (status == PREFOS_STATUS_OK)
                {
                    size_t fixed_before_tail =
                        presolver->n_fixed_columns;
                    size_t substituted_before_tail =
                        presolver->stats.substituted_free_variables;
                    size_t parallel_before_tail =
                        presolver->n_parallel_column_reductions;
                    presolver->scalar_redundancy_completed = 1;
                    if (fast_trigger_signature_changed(
                            before_final_activity,
                            capture_fast_trigger_signature(presolver)))
                    {
                        status = run_fast_fixed_point_timed(
                            presolver, 1, 0,
                            shared_column_workspace_valid
                                ? &shared_column_workspace
                                : NULL);
                        if (status == PREFOS_STATUS_OK &&
                            (presolver->n_fixed_columns !=
                                 fixed_before_tail ||
                             presolver->stats.substituted_free_variables !=
                                 substituted_before_tail ||
                             presolver->n_parallel_column_reductions !=
                                 parallel_before_tail))
                        {
                            presolver->scalar_redundancy_completed = 0;
                            status = run_scalar_tail_fixed_point(
                                presolver,
                                shared_column_workspace_valid
                                    ? &shared_column_workspace
                                    : NULL,
                                include_parallel_rows);
                        }
                    }
                }
            }
        }
        prefos_internal_timer_now(&phase_stop);
        presolver->stats.medium_fixed_point_milliseconds =
            prefos_internal_timer_elapsed_milliseconds(
                &phase_start, &phase_stop);
        if (status != PREFOS_STATUS_OK) goto failure;
    }
    prefos_internal_free_linear_propagation_cache(presolver);
    {
        PreFOSFastTriggerSignature before_parallel_columns =
            capture_fast_trigger_signature(presolver);
        size_t minimum_bound_changes =
            source->n <= 65536
                ? 1
                : (source->n + 255) / 256;
        size_t bound_changes_since_early =
            early_parallel_signature_valid &&
                    before_parallel_columns.changed_bounds >=
                        early_parallel_signature.changed_bounds
                ? before_parallel_columns.changed_bounds -
                      early_parallel_signature.changed_bounds
                : SIZE_MAX;
        int rerun_parallel_columns =
            !early_parallel_signature_valid ||
            presolver->n_removed_rows !=
                early_parallel_removed_rows ||
            before_parallel_columns.removed_rows !=
                early_parallel_signature.removed_rows ||
            before_parallel_columns.cone_reductions !=
                early_parallel_signature.cone_reductions ||
            presolver->n_rows_require_materialization !=
                early_parallel_materialized_rows ||
            shared_column_workspace.objective_change_epoch -
                    early_parallel_objective_epoch >=
                minimum_bound_changes ||
            bound_changes_since_early >= minimum_bound_changes;
        if (getenv("PREFOS_TRACE_PARALLEL_COLUMN_SCHEDULER"))
            fprintf(
                stderr,
                "PreFOS parallel-column scheduler rerun=%d "
                "removed=%zu/%zu signature_rows=%zu/%zu "
                "materialized=%zu/%zu objective=%zu/%zu "
                "bounds=%zu threshold=%zu\n",
                rerun_parallel_columns,
                presolver->n_removed_rows,
                early_parallel_removed_rows,
                before_parallel_columns.removed_rows,
                early_parallel_signature.removed_rows,
                presolver->n_rows_require_materialization,
                early_parallel_materialized_rows,
                shared_column_workspace.objective_change_epoch,
                early_parallel_objective_epoch,
                bound_changes_since_early,
                minimum_bound_changes);
        prefos_internal_timer_now(&phase_start);
        if (rerun_parallel_columns)
            status =
                shared_column_workspace_valid
                    ? prefos_internal_reduce_parallel_columns_in_workspace(
                          presolver, &shared_column_workspace)
                    : prefos_internal_reduce_parallel_columns(presolver);
        else
            status = PREFOS_STATUS_OK;
        prefos_internal_timer_now(&phase_stop);
        presolver->stats.parallel_column_reduction_milliseconds +=
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
        if (shared_column_workspace_valid)
        {
            status = run_materialized_parallel_row_closure(
                presolver, &shared_column_workspace, 0);
            if (status != PREFOS_STATUS_OK) goto failure;
        }
    }
    {
        size_t structural_eliminations =
            presolver->stats.substituted_free_variables;
        structural_eliminations = saturated_change_sum(
            structural_eliminations,
            presolver->stats.merged_parallel_columns);
        size_t remaining_columns =
            structural_eliminations < source->n
                ? source->n - structural_eliminations
                : 0;
        int high_structural_yield =
            structural_eliminations >= (source->n + 7) / 8;
        int force_early_materialized_closure =
            getenv("PREFOS_FORCE_EARLY_MATERIALIZED_CLOSURE") != NULL;
        if (source->n >= 1024 && source->A.rows > 0 &&
            (force_early_materialized_closure ||
             (high_structural_yield &&
              (source->A.nnz <= 1000000 ||
               remaining_columns <= 4096))))
        {
            PreFOSTimestamp start, stop;
            size_t pending_parallel_rows_removed = 0;
            size_t closure_rounds =
                2 * (size_t)
                    presolver->settings.max_aggregation_rounds;
            if (closure_rounds < 4) closure_rounds = 4;
            if (closure_rounds > 8) closure_rounds = 8;
            prefos_internal_timer_now(&start);
            status = prefos_internal_run_materialized_column_closure(
                presolver, closure_rounds, 1, 0,
                &pending_parallel_rows_removed, 0);
            prefos_internal_timer_now(&stop);
            presolver->stats.structural_reduction_milliseconds +=
                prefos_internal_timer_elapsed_milliseconds(&start, &stop);
            if (status != PREFOS_STATUS_OK) goto failure;
            if (pending_parallel_rows_removed > 0)
            {
                status = run_fast_fixed_point_timed(
                    presolver, 1, 0,
                    shared_column_workspace_valid
                        ? &shared_column_workspace
                        : NULL);
                if (status != PREFOS_STATUS_OK) goto failure;
            }
        }
    }
    if (presolver->settings.bounded_doubleton_substitution &&
        source->n_box > 0 && source->A.rows > 0)
    {
        PreFOSTimestamp start, stop;
        size_t bounded_before =
            presolver->stats.substituted_bounded_doubletons;
        size_t materialized_first_candidates = 0;
        size_t materialized_first_threshold = 0;
        int seeded_materialized_closure = 0;
        int run_materialized_first =
            shared_column_workspace_valid &&
            presolver->n_rows_require_materialization == 0 &&
            should_run_materialized_first_bounded(
                presolver, &shared_column_workspace,
                &materialized_first_candidates,
                &materialized_first_threshold);
        trace_presolve_stage(
            presolver, "before-bounded-doubletons", &presolve_start);
        prefos_internal_timer_now(&start);
        failure_stage = "bounded-doubleton";
        if (getenv("PREFOS_TRACE_BOUNDED_SCHEDULER"))
            fprintf(
                stderr,
                "PreFOS bounded-materialized-first run=%d "
                "candidates=%zu threshold=%zu nnz=%zu\n",
                run_materialized_first,
                materialized_first_candidates,
                materialized_first_threshold, source->A.nnz);
        if (run_materialized_first)
        {
            size_t pending_parallel_rows_removed = 0;
            size_t closure_rounds =
                2 * (size_t)
                    presolver->settings.max_aggregation_rounds;
            if (closure_rounds > 8) closure_rounds = 8;
            status =
                prefos_internal_run_materialized_column_closure_from_workspace(
                    presolver, closure_rounds, 1, 0,
                    &pending_parallel_rows_removed, 1,
                    &shared_column_workspace);
            shared_column_workspace_valid =
                shared_column_workspace.starts != NULL;
            if (status == PREFOS_STATUS_OK &&
                pending_parallel_rows_removed > 0)
            {
                failure_stage =
                    "bounded-doubleton-seeded-fast-closure";
                status = run_fast_fixed_point_timed(
                    presolver, 1, 0,
                    shared_column_workspace_valid
                        ? &shared_column_workspace
                        : NULL);
            }
            seeded_materialized_closure =
                status == PREFOS_STATUS_OK;
        }
        else if (shared_column_workspace_valid)
        {
            status = prefos_internal_refresh_column_workspace_incremental(
                presolver, &shared_column_workspace);
            if (status == PREFOS_STATUS_OK)
                status = prefos_internal_reduce_bounded_doubletons(
                    presolver, &shared_column_workspace);
        }
        else
        {
            PreFOSColumnWorkspace workspace;
            status =
                prefos_internal_build_structural_column_workspace_cpu(
                    presolver, &workspace);
            if (status == PREFOS_STATUS_OK)
            {
                status = prefos_internal_reduce_bounded_doubletons(
                    presolver, &workspace);
                prefos_internal_free_column_workspace(&workspace);
            }
        }
        prefos_internal_timer_now(&stop);
        presolver->stats.structural_reduction_milliseconds +=
            prefos_internal_timer_elapsed_milliseconds(&start, &stop);
        if (status != PREFOS_STATUS_OK) goto failure;
        {
            size_t accepted =
                presolver->stats.substituted_bounded_doubletons -
                bounded_before;
            if (seeded_materialized_closure)
                accepted = 0;
            int high_bounded_yield =
                accepted >= 64 &&
                accepted >= (source->n + 7) / 8;
            int moderate_bounded_yield =
                !high_bounded_yield &&
                accepted >= 64 &&
                accepted >= (source->n + 9) / 10;
            int chain_is_promising =
                accepted > 0 &&
                bounded_doubleton_chain_is_promising(presolver);
            int run_full_closure = 0;
            if (getenv("PREFOS_FORCE_GENERAL_TRIVIAL_CLOSURE"))
                run_full_closure = 1;
            if (accepted > 0 && source->A.nnz <= 65536)
                run_full_closure = 1;
            int materialized_opportunities = 0;
            double opportunity_milliseconds = 0.0;
            if (!run_full_closure && !chain_is_promising &&
                accepted > 0)
            {
                PreFOSTimestamp opportunity_start;
                PreFOSTimestamp opportunity_stop;
                prefos_internal_timer_now(&opportunity_start);
                materialized_opportunities =
                    prefos_internal_materialized_support_opportunities(
                        presolver, &shared_column_workspace);
                prefos_internal_timer_now(&opportunity_stop);
                opportunity_milliseconds =
                    prefos_internal_timer_elapsed_milliseconds(
                        &opportunity_start, &opportunity_stop);
            }
            if (materialized_opportunities &
                PREFOS_MATERIALIZED_SUPPORT_BOUNDED_DOUBLETON)
                run_full_closure = 1;
            if ((materialized_opportunities &
                 PREFOS_MATERIALIZED_SUPPORT_TRIVIAL) &&
                (materialized_opportunities &
                 PREFOS_MATERIALIZED_SUPPORT_PARALLEL))
                run_full_closure = 1;
            if (high_bounded_yield &&
                (materialized_opportunities &
                 PREFOS_MATERIALIZED_SUPPORT_TRIVIAL))
                run_full_closure = 1;
            if (high_bounded_yield && !run_full_closure &&
                materialized_opportunities == 0)
                materialized_opportunities =
                    PREFOS_MATERIALIZED_SUPPORT_PARALLEL;
            if (moderate_bounded_yield &&
                materialized_opportunities == 0)
                run_full_closure = 1;
            if (getenv("PREFOS_TRACE_BOUNDED_SCHEDULER"))
                fprintf(
                    stderr,
                    "PreFOS bounded-scheduler accepted=%zu high_yield=%d "
                    "moderate_yield=%d "
                    "chain=%d "
                    "full=%d opportunities=%d opportunity_ms=%.3f\n",
                    accepted, high_bounded_yield,
                    moderate_bounded_yield,
                    chain_is_promising, run_full_closure,
                    materialized_opportunities,
                    opportunity_milliseconds);
            if (run_full_closure || chain_is_promising)
            {
                size_t pending_parallel_rows_removed = 0;
                int interleave_propagation =
                    chain_is_promising &&
                    (presolver->settings
                         .linear_propagation_max_stale_rounds == 0 ||
                     getenv(
                         "PREFOS_FORCE_INTERLEAVED_CLOSURE_PROPAGATION"));
                size_t closure_rounds =
                    2 * (size_t)
                        presolver->settings.max_aggregation_rounds;
                if (closure_rounds > 8) closure_rounds = 8;
                if (high_bounded_yield ||
                    moderate_bounded_yield)
                    closure_rounds = 1;
                if (source->n >= 1048576 &&
                    source->A.nnz >= 4194304)
                    closure_rounds = 1;
                {
                    const char *deep =
                        getenv("PREFOS_FORCE_DEEP_BOUNDED_CLOSURE");
                    if (deep && atoi(deep) > 0)
                        closure_rounds = (size_t) atoi(deep);
                }
                prefos_internal_timer_now(&start);
                failure_stage = "bounded-doubleton-closure";
                status = prefos_internal_run_materialized_column_closure(
                    presolver, closure_rounds, 1,
                    interleave_propagation,
                    &pending_parallel_rows_removed,
                    shared_column_workspace_valid);
                prefos_internal_timer_now(&stop);
                presolver->stats.structural_reduction_milliseconds +=
                    prefos_internal_timer_elapsed_milliseconds(
                        &start, &stop);
                if (status != PREFOS_STATUS_OK) goto failure;
                if (pending_parallel_rows_removed > 0)
                {
                    failure_stage = "bounded-doubleton-fast-closure";
                    status = run_fast_fixed_point_timed(
                        presolver, 1, 0, NULL);
                    if (status != PREFOS_STATUS_OK) goto failure;
                }
            }
            else if (
                materialized_opportunities &
                PREFOS_MATERIALIZED_SUPPORT_TRIVIAL)
            {
                PreFOSFastTriggerSignature closure_before =
                    capture_fast_trigger_signature(presolver);
                size_t closure_rounds =
                    2 * (size_t)
                        presolver->settings.max_aggregation_rounds;
                if (closure_rounds > 8) closure_rounds = 8;
                prefos_internal_timer_now(&start);
                failure_stage =
                    "bounded-doubleton-trivial-closure";
                status =
                    prefos_internal_run_materialized_trivial_closure(
                        presolver, closure_rounds);
                prefos_internal_timer_now(&stop);
                presolver->stats.structural_reduction_milliseconds +=
                    prefos_internal_timer_elapsed_milliseconds(
                        &start, &stop);
                if (status != PREFOS_STATUS_OK) goto failure;
                if (fast_trigger_signature_changed(
                        closure_before,
                        capture_fast_trigger_signature(presolver)))
                {
                    presolver->scalar_redundancy_completed = 0;
                    failure_stage =
                        "bounded-doubleton-fast-trivial-closure";
                    status = run_fast_fixed_point_timed(
                        presolver, 1, 0, NULL);
                    if (status != PREFOS_STATUS_OK) goto failure;
                }
            }
            else if (
                materialized_opportunities &
                PREFOS_MATERIALIZED_SUPPORT_PARALLEL)
            {
                failure_stage =
                    "bounded-doubleton-parallel-row-closure";
                status = run_materialized_parallel_row_closure(
                    presolver,
                    shared_column_workspace_valid
                        ? &shared_column_workspace
                        : NULL,
                    1);
                if (status != PREFOS_STATUS_OK) goto failure;
            }
            if (accepted > 0 && shared_column_workspace_valid &&
                presolver->settings.parallel_column_reduction)
            {
                size_t parallel_before =
                    presolver->n_parallel_column_reductions;
                int ran = 0;
                prefos_internal_timer_now(&start);
                failure_stage =
                    "bounded-doubleton-transformed-columns";
                status =
                    prefos_internal_refresh_column_workspace_incremental(
                        presolver, &shared_column_workspace);
                if (status == PREFOS_STATUS_OK)
                    status =
                        prefos_internal_reduce_transformed_parallel_columns(
                            presolver, &shared_column_workspace, &ran);
                prefos_internal_timer_now(&stop);
                presolver->stats.parallel_column_reduction_milliseconds +=
                    prefos_internal_timer_elapsed_milliseconds(
                        &start, &stop);
                if (status != PREFOS_STATUS_OK) goto failure;
                if (ran &&
                    presolver->n_parallel_column_reductions >
                        parallel_before)
                {
                    presolver->scalar_redundancy_completed = 0;
                    status = run_fast_fixed_point_timed(
                        presolver, 1, 0,
                        &shared_column_workspace);
                    if (status != PREFOS_STATUS_OK) goto failure;
                }
            }
        }
        trace_presolve_stage(
            presolver, "after-bounded-doubletons", &presolve_start);
    }
    {
        PreFOSFastTriggerSignature before =
            capture_fast_trigger_signature(presolver);
        size_t pending_parallel_rows_removed = 0;
        trace_presolve_stage(
            presolver, "before-free-columns", &presolve_start);
        prefos_internal_timer_now(&phase_start);
        status = prefos_internal_substitute_free_columns(presolver);
        prefos_internal_timer_now(&phase_stop);
        presolver->stats.free_column_substitution_milliseconds =
            prefos_internal_timer_elapsed_milliseconds(
                &phase_start, &phase_stop);
        if (status != PREFOS_STATUS_OK) goto failure;
        if (fast_trigger_signature_changed(
                before, capture_fast_trigger_signature(presolver)))
        {
            status = run_fast_fixed_point_timed(
                presolver, 1, 0,
                shared_column_workspace_valid
                    ? &shared_column_workspace
                    : NULL);
            if (status != PREFOS_STATUS_OK) goto failure;
            status = prefos_internal_run_materialized_column_closure(
                presolver,
                (size_t) presolver->settings.max_aggregation_rounds,
                1, 0,
                &pending_parallel_rows_removed, 0);
            if (status != PREFOS_STATUS_OK) goto failure;
            if (pending_parallel_rows_removed > 0)
            {
                status = run_fast_fixed_point_timed(
                    presolver, 1, 0,
                    shared_column_workspace_valid
                        ? &shared_column_workspace
                        : NULL);
                if (status != PREFOS_STATUS_OK) goto failure;
            }
        }
        trace_presolve_stage(
            presolver, "after-free-columns", &presolve_start);
    }
    trace_presolve_stage(
        presolver, "before-final-propagation", &presolve_start);
    if (should_run_post_substitution_propagation(presolver))
    {
        int final_round;
        int trace_final_propagation =
            getenv("PREFOS_TRACE_FINAL_PROPAGATION") != NULL;
        for (final_round = 0; final_round < 4; ++final_round)
        {
            PreFOSFastTriggerSignature before =
                capture_fast_trigger_signature(presolver);
            PreFOSFastTriggerSignature after_propagation;
            PreFOSFastTriggerSignature after_fast;
            size_t fixed_before_fast;
            size_t substituted_before_fast;
            size_t parallel_before_fast;
            size_t fixed_before_round =
                presolver->n_fixed_columns;
            size_t substituted_before_round =
                presolver->stats.substituted_free_variables;
            size_t parallel_before_round =
                presolver->n_parallel_column_reductions;
            size_t budget_stops_before =
                presolver->stats.linear_budget_stops;
            int fast_structural_changed;
            int tail_budget_stopped;
            int scalar_redundancy_was_completed =
                presolver->scalar_redundancy_completed;

            presolver->scalar_redundancy_completed = 0;
            prefos_internal_timer_now(&phase_start);
            failure_stage = "final-materialized-propagation";
            status = run_materialized_linear_propagation(
                presolver,
                shared_column_workspace_valid
                    ? &shared_column_workspace
                    : NULL);
            prefos_internal_timer_now(&phase_stop);
            presolver->stats.linear_propagation_milliseconds +=
                prefos_internal_timer_elapsed_milliseconds(
                    &phase_start, &phase_stop);
            if (status != PREFOS_STATUS_OK) goto failure;
            if (scalar_redundancy_was_completed &&
                presolver->linear_propagation_complete)
                presolver->scalar_redundancy_completed = 1;
            tail_budget_stopped =
                presolver->stats.linear_budget_stops >
                budget_stops_before;
            after_propagation =
                capture_fast_trigger_signature(presolver);
            if (trace_final_propagation)
                fprintf(
                    stderr,
                    "PreFOS final-propagation outer=%d delta_events=%zu "
                    "delta_rows=%zu delta_bounds=%zu delta_cones=%zu\n",
                    final_round,
                    after_propagation.transformation_events -
                        before.transformation_events,
                    after_propagation.removed_rows -
                        before.removed_rows,
                    after_propagation.changed_bounds -
                        before.changed_bounds,
                    after_propagation.cone_reductions -
                        before.cone_reductions);
            if (!fast_trigger_signature_changed(
                    before, after_propagation))
                break;

            fixed_before_fast = presolver->n_fixed_columns;
            substituted_before_fast =
                presolver->stats.substituted_free_variables;
            parallel_before_fast =
                presolver->n_parallel_column_reductions;
            failure_stage = "final-materialized-propagation-fast";
            if (trace_final_propagation)
                prefos_internal_timer_now(&phase_start);
            status = run_fast_fixed_point_timed(
                presolver, 1, 0,
                shared_column_workspace_valid
                    ? &shared_column_workspace
                    : NULL);
            after_fast =
                capture_fast_trigger_signature(presolver);
            if (trace_final_propagation)
            {
                prefos_internal_timer_now(&phase_stop);
                fprintf(
                    stderr,
                    "PreFOS final-propagation outer=%d fast_ms=%.3f "
                    "delta_events=%zu delta_rows=%zu delta_bounds=%zu "
                    "delta_cones=%zu\n",
                    final_round,
                    prefos_internal_timer_elapsed_milliseconds(
                        &phase_start, &phase_stop),
                    after_fast.transformation_events -
                        after_propagation.transformation_events,
                    after_fast.removed_rows -
                        after_propagation.removed_rows,
                    after_fast.changed_bounds -
                        after_propagation.changed_bounds,
                    after_fast.cone_reductions -
                        after_propagation.cone_reductions);
            }
            if (status != PREFOS_STATUS_OK) goto failure;
            fast_structural_changed =
                presolver->n_fixed_columns != fixed_before_fast ||
                presolver->stats.substituted_free_variables !=
                    substituted_before_fast ||
                presolver->n_parallel_column_reductions !=
                    parallel_before_fast;

            if (tail_budget_stopped)
            {
                if (trace_final_propagation)
                    fprintf(
                        stderr,
                        "PreFOS final-propagation outer=%d "
                        "hard-budget-stop\n",
                        final_round);
                break;
            }

            if ((source->A.nnz <= 1000000 ||
                 presolver->settings
                         .linear_propagation_max_stale_rounds == 0 ||
                 (presolver->n_fixed_columns +
                          presolver->stats.substituted_free_variables +
                          presolver->n_parallel_column_reductions >=
                      (source->n + 1) / 2 &&
                  saturated_change_sum(
                      saturated_change_sum(
                          presolver->n_fixed_columns -
                              fixed_before_round,
                          presolver->stats.substituted_free_variables -
                              substituted_before_round),
                      saturated_change_sum(
                          presolver->n_parallel_column_reductions -
                              parallel_before_round,
                          after_fast.removed_rows -
                              before.removed_rows)) >=
                      (source->A.nnz + 1023) / 1024)) &&
                (fast_structural_changed ||
                 after_fast.removed_rows != before.removed_rows ||
                 after_fast.changed_bounds != before.changed_bounds ||
                 presolver->settings
                         .linear_propagation_max_stale_rounds == 0))
            {
                PreFOSFastTriggerSignature closure_before =
                    capture_fast_trigger_signature(presolver);
                size_t pending_parallel_rows_removed = 0;
                failure_stage =
                    "final-materialized-propagation-closure";
                if (trace_final_propagation)
                    prefos_internal_timer_now(&phase_start);
                status =
                    prefos_internal_run_materialized_column_closure(
                        presolver, 2, 1, 0,
                        &pending_parallel_rows_removed, 0);
                if (trace_final_propagation)
                {
                    prefos_internal_timer_now(&phase_stop);
                    fprintf(
                        stderr,
                        "PreFOS final-propagation outer=%d "
                        "closure_ms=%.3f pending_parallel=%zu\n",
                        final_round,
                        prefos_internal_timer_elapsed_milliseconds(
                            &phase_start, &phase_stop),
                        pending_parallel_rows_removed);
                }
                if (status != PREFOS_STATUS_OK) goto failure;
                if (pending_parallel_rows_removed > 0 ||
                    fast_trigger_signature_changed(
                        closure_before,
                        capture_fast_trigger_signature(presolver)))
                {
                    failure_stage =
                        "final-materialized-propagation-closure-fast";
                    status = run_fast_fixed_point_timed(
                        presolver, 1, 0,
                        shared_column_workspace_valid
                            ? &shared_column_workspace
                            : NULL);
                    if (status != PREFOS_STATUS_OK) goto failure;
                }
            }

            if (medium_fixed_point_round_is_stale(
                    presolver, after_propagation,
                    capture_fast_trigger_signature(presolver),
                    fixed_before_fast, substituted_before_fast,
                    parallel_before_fast))
                break;
            if (!presolver->materialized_row_updates_require_cache)
                prefos_internal_clear_working_matrix_cache(presolver);
        }
    }
    trace_presolve_stage(
        presolver, "after-final-propagation", &presolve_start);
    if (shared_column_workspace_valid)
    {
        int objective_dirty =
            prefos_internal_queue_transformation_events(
                presolver, &shared_column_workspace);
        failure_stage = "objective-cache-finalize";
        status = objective_dirty
                     ? prefos_internal_rebuild_column_objective(
                           presolver, &shared_column_workspace)
                     : prefos_internal_sync_column_objective(
                           presolver, &shared_column_workspace);
        if (status != PREFOS_STATUS_OK) goto failure;
        cached_linear_objective =
            shared_column_workspace.objective;
        cached_linear_objective_offset =
            shared_column_workspace.objective_offset;
        cached_linear_objective_valid = 1;
        shared_column_workspace.objective = NULL;
        prefos_internal_free_column_workspace(
            &shared_column_workspace);
        shared_column_workspace_valid = 0;
    }
    trace_presolve_stage(
        presolver, "after-workspace-finalize", &presolve_start);
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
    failure_stage = "matrix-compaction";
    status = prefos_internal_compact_a(presolver);
    prefos_internal_timer_now(&phase_stop);
    presolver->stats.matrix_compaction_milliseconds =
        prefos_internal_timer_elapsed_milliseconds(
            &phase_start, &phase_stop);
    if (status != PREFOS_STATUS_OK) goto failure;
    trace_presolve_stage(
        presolver, "after-matrix-compaction", &presolve_start);
    prefos_internal_timer_now(&phase_start);
    failure_stage = "quadratic-compaction";
    status = compact_q(presolver, &target->Q);
    prefos_internal_timer_now(&phase_stop);
    presolver->stats.quadratic_compaction_milliseconds =
        prefos_internal_timer_elapsed_milliseconds(
            &phase_start, &phase_stop);
    if (status != PREFOS_STATUS_OK) goto failure;
    prefos_internal_timer_now(&phase_start);
    failure_stage = "factor-compaction";
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
    failure_stage = "domain-compaction";
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
    failure_stage = "objective-compaction";
    status = build_reduced_objective(
        presolver, cached_linear_objective,
        cached_linear_objective_offset,
        cached_linear_objective_valid);
    cached_linear_objective = NULL;
    prefos_internal_timer_now(&phase_stop);
    presolver->stats.objective_compaction_milliseconds =
        prefos_internal_timer_elapsed_milliseconds(
            &phase_start, &phase_stop);
    if (status != PREFOS_STATUS_OK) goto failure;
    trace_presolve_stage(
        presolver, "after-output-compaction", &presolve_start);

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
    {
        const char *trace = getenv("PREFOS_TRACE_FAILURE_STAGE");
        if (trace && *trace && *trace != '0')
            fprintf(
                stderr, "PreFOS failure stage=%s status=%d\n",
                failure_stage, (int) status);
    }
    prefos_internal_free_linear_propagation_cache(presolver);
    free(cached_linear_objective);
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
