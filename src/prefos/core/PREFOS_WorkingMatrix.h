/*
 * Copyright 2026 Hongpei Li
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef PREFOS_WORKING_MATRIX_H
#define PREFOS_WORKING_MATRIX_H

#include "PREFOS_Internal.h"

typedef struct
{
    PreFOSCsrMatrix matrix;
    double *lower;
    double *upper;
    int *column_counts;
} PreFOSWorkingMatrix;

PREFOS_INTERNAL void prefos_internal_free_working_matrix(
    PreFOSWorkingMatrix *working);

PREFOS_INTERNAL void prefos_internal_clear_working_matrix_cache(
    PreFOSPresolver *presolver);

PREFOS_INTERNAL void prefos_internal_store_working_matrix_cache(
    PreFOSPresolver *presolver, PreFOSWorkingMatrix *working);

PREFOS_INTERNAL int prefos_internal_take_current_working_matrix_cache(
    PreFOSPresolver *presolver, PreFOSWorkingMatrix *working,
    int *parallel_rows_checked);

PREFOS_INTERNAL void prefos_internal_get_working_matrix_source(
    const PreFOSPresolver *presolver, const PreFOSCsrMatrix **matrix,
    const double **lower, const double **upper);

PREFOS_INTERNAL PreFOSStatus prefos_internal_materialize_working_matrix(
    PreFOSPresolver *presolver, const PreFOSCsrMatrix *source,
    const double *source_lower, const double *source_upper,
    PreFOSWorkingMatrix *target);

PREFOS_INTERNAL PreFOSStatus prefos_internal_expand_working_row(
    const PreFOSPresolver *presolver, const PreFOSCsrMatrix *source,
    size_t row, double *row_values, int *row_marks,
    int *touched_columns, size_t *n_touched, double *shift);

PREFOS_INTERNAL void prefos_internal_clear_expanded_working_row(
    double *row_values, int *row_marks, const int *touched_columns,
    size_t n_touched);

#endif
