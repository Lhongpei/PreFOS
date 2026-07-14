/*
 * Copyright 2025-2026 Daniel Cederberg
 * Copyright 2026 Hongpei Li
 *
 * Modified for PreFOS in 2026.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef PRESOLVE_PARALLEL_ROW_REDUCTION_H
#define PRESOLVE_PARALLEL_ROW_REDUCTION_H

#include "ParallelRowDetection.h"

typedef enum
{
    PRESOLVE_PARALLEL_REDUCTION_OK = 0,
    PRESOLVE_PARALLEL_REDUCTION_UNCERTAIN,
    PRESOLVE_PARALLEL_REDUCTION_INFEASIBLE,
    PRESOLVE_PARALLEL_REDUCTION_NUMERICAL_ERROR
} PresolveParallelReductionStatus;

typedef int (*PresolveValuesClose)(const void *context, double left,
                                   double right, double tolerance);

typedef struct
{
    int kept_row;
    double lower;
    double upper;
    int lower_source_row;
    int upper_source_row;
    double lower_source_ratio;
    double upper_source_ratio;
} PresolveParallelRowReduction;

PresolveParallelReductionStatus presolve_analyze_parallel_row_group(
    const PresolveSparseRowView *matrix, const int *rows, size_t count,
    const double *lower, const double *upper, double tolerance,
    PresolveValuesClose values_close, const void *close_context,
    PresolveParallelRowReduction *reduction);

#endif
