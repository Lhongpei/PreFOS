/*
 * Copyright 2026 Hongpei Li
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef PREFOS_AFFINE_FACE_SUBSTITUTION_INTERNAL_H
#define PREFOS_AFFINE_FACE_SUBSTITUTION_INTERNAL_H

#include "PREFOS_AffineFaceSubstitution.h"

typedef struct
{
    int *starts;
    int *rows;
    double *coefficients;
} PreFOSColumnStorage;

typedef struct
{
    double *values;
    int *marks;
    int *touched;
    size_t count;
    double constant;
} PreFOSExpandedAffineRow;

PREFOS_INTERNAL void prefos_affine_face_free_column_storage(
    PreFOSColumnStorage *storage);
PREFOS_INTERNAL PreFOSStatus prefos_affine_face_build_column_storage(
    const PreFOSCsrMatrix *matrix, PreFOSColumnStorage *storage);
PREFOS_INTERNAL PreFOSStatus prefos_affine_face_build_affine_column_storage(
    const PreFOSPresolver *presolver, PreFOSColumnStorage *storage);
PREFOS_INTERNAL void prefos_affine_face_clear_expanded_row(
    PreFOSExpandedAffineRow *row);
PREFOS_INTERNAL PreFOSStatus prefos_affine_face_expand_input_row(
    const PreFOSPresolver *presolver, size_t affine_row,
    PreFOSExpandedAffineRow *expanded);
PREFOS_INTERNAL PreFOSStatus prefos_affine_face_expand_generated_row(
    const PreFOSPresolver *presolver, int coordinate_column,
    PreFOSExpandedAffineRow *expanded);
PREFOS_INTERNAL size_t prefos_affine_face_active_column_degree(
    const PreFOSPresolver *presolver, const PreFOSColumnStorage *storage, int column);
PREFOS_INTERNAL int prefos_affine_face_choose_pivot(
    const PreFOSPresolver *presolver, const PreFOSColumnStorage *storage,
    const unsigned char *quadratic_column, const unsigned char *factor_column,
    const PreFOSExpandedAffineRow *row);
PREFOS_INTERNAL PreFOSStatus prefos_affine_face_append_transformation(
    PreFOSPresolver *presolver, const PreFOSColumnStorage *storage,
    const PreFOSColumnStorage *affine_storage,
    const PreFOSExpandedAffineRow *row, size_t affine_row, int pivot);

#endif
