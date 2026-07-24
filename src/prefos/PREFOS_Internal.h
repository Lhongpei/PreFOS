/*
 * Copyright 2026 Hongpei Li
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef PREFOS_INTERNAL_H
#define PREFOS_INTERNAL_H

#include <PreFOS/PreFOS.h>
#include "TransformationLog.h"

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifndef PREFOS_MAX_AGGREGATION_TERMS
#define PREFOS_MAX_AGGREGATION_TERMS 4
#endif
#ifndef PREFOS_MAX_SUBSTITUTION_DEPTH
#define PREFOS_MAX_SUBSTITUTION_DEPTH 32
#endif

#if defined(__GNUC__) || defined(__clang__)
#define PREFOS_INTERNAL __attribute__((visibility("hidden")))
#else
#define PREFOS_INTERNAL
#endif

typedef struct
{
    size_t n_removed;
    int *removed_matrix_indices;
    int *source_rows;
} PreFOSPSDFaceReduction;

typedef struct
{
    size_t affine_row;
    int column;
    int generated_column;
    double coefficient;
    double offset;
    double implied_bound;
    unsigned char is_lower;
} PreFOSAffineBoundCertificate;

typedef enum
{
    PREFOS_SUBSTITUTION_STANDARD = 0,
    PREFOS_SUBSTITUTION_BOUNDED_DOUBLETON,
    PREFOS_SUBSTITUTION_RESIDUAL_ROW
} PreFOSSubstitutionMode;

typedef struct PreFOSCudaWorkspace PreFOSCudaWorkspace;

struct PreFOSPresolver
{
    PreFOSProblemData original;
    PreFOSSettings settings;
    PreFOSPresolvedProblem reduced;
    PreFOSStats stats;
    int *original_to_reduced;
    int *original_to_reduced_rows;
    double *fixed_values;
    unsigned char *is_fixed;
    int *fixed_column_log;
    size_t n_fixed_columns;
    unsigned char *is_substituted;
    unsigned char *is_parallel_removed;
    size_t *substitution_term_count;
    unsigned char *substitution_incoming_depth;
    unsigned char *substitution_keeps_source_row;
    size_t *substitution_term_start;
    int *substitution_source_row;
    int *residual_source_column;
    double *substitution_constant;
    int *substitution_targets;
    double *substitution_scales;
    size_t n_substitution_terms;
    size_t substitution_term_capacity;
    size_t n_residual_row_substitutions;
    int *variable_to_box;
    double *working_box_lower;
    double *working_box_upper;
    double *working_constraint_lower;
    double *working_constraint_upper;
    double *propagation_lower;
    double *propagation_upper;
    int scalar_redundancy_completed;
    size_t fixed_column_epoch;
    unsigned char *converted_affine_cones;
    unsigned char *affine_protected_columns;
    int *affine_aggregation_source_rows;
    double *affine_aggregation_pivots;
    PreFOSAffineBoundCertificate *affine_bound_certificates;
    size_t n_affine_bound_certificates;
    size_t affine_bound_certificate_capacity;
    size_t n_pre_face_affine_rows;
    int *affine_pre_to_reduced_rows;
    PreFOSPSDStructureAnalysis *psd_structure_analyses;
    size_t n_psd_structure_analyses;
    unsigned char *input_affine_rsoc_zero_axis;
    unsigned char *generated_affine_rsoc_zero_axis;
    unsigned char *affine_face_substitution_targets;
    unsigned char *affine_face_eliminated_columns;
    size_t n_affine_face_substitutions;
    unsigned char *remove_rows;
    int *removed_row_log;
    size_t n_removed_rows;
    unsigned char *remove_cones;
    int *cone_face_survivors;
    unsigned char *cone_face_box;
    double *cone_face_box_lower;
    double *cone_face_box_upper;
    int *cone_collapse_source_rows;
    PreFOSPSDFaceReduction *psd_face_reductions;
    PreFOSFacialReductionCertificate *facial_reductions;
    size_t n_facial_reductions;
    PresolveTransformationLog transformations;
    size_t normalized_nonnegative_variables;
    size_t normalized_nonnegative_cones;
    size_t n_parallel_column_reductions;
    PreFOSCudaWorkspace *cuda_workspace;
    int has_run;
};

static inline int prefos_internal_term_is_active_in_row(
    const PreFOSPresolver *presolver, size_t row, int column)
{
    int excluded = presolver->residual_source_column
                       ? presolver->residual_source_column[row]
                       : -2;
    if (excluded >= 0) return column != excluded;
    if (excluded == -1) return 1;
    return !(presolver->is_substituted[column] &&
             presolver->substitution_keeps_source_row[column] &&
             presolver->substitution_source_row[column] == (int) row);
}

PREFOS_INTERNAL void *prefos_internal_alloc_array(size_t count, size_t element_size);
PREFOS_INTERNAL void prefos_internal_free_csr(PreFOSCsrMatrix *matrix);
PREFOS_INTERNAL double prefos_internal_safe_midpoint(double lower, double upper);
PREFOS_INTERNAL int prefos_internal_safe_add_product(double *accumulator, double left,
                                               double right);
PREFOS_INTERNAL int prefos_internal_safe_product(double left, double right,
                                           double *product);
PREFOS_INTERNAL double prefos_internal_outward_bound_cast(long double value, int is_lower);
PREFOS_INTERNAL int prefos_internal_mark_fixed_column(
    PreFOSPresolver *presolver, int column, double value);
PREFOS_INTERNAL int prefos_internal_mark_removed_row(
    PreFOSPresolver *presolver, size_t row);
PREFOS_INTERNAL PreFOSStatus prefos_internal_copy_vector(const void *source, size_t count,
                                                size_t element_size, void **target);
PREFOS_INTERNAL void prefos_internal_free_reduced_problem(PreFOSPresolvedProblem *problem);
PREFOS_INTERNAL void prefos_internal_free_psd_face_reductions(PreFOSPSDFaceReduction *faces,
                                                        size_t n_cones);

PREFOS_INTERNAL PreFOSStatus prefos_internal_append_bound_record(PreFOSPresolver *presolver,
                                                        int row, int column,
                                                        double old_bound,
                                                        double new_bound,
                                                        int is_lower);
PREFOS_INTERNAL int prefos_internal_values_close(double left, double right,
                                               double tolerance);
PREFOS_INTERNAL PreFOSStatus prefos_internal_expand_linear_objective(
    const PreFOSPresolver *presolver, double *objective, double *offset);

#endif
