/*
 * Copyright 2025-2026 Daniel Cederberg
 * Copyright 2026 Hongpei Li
 *
 * Modified for PreFOS in 2026.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ParallelRowReduction.h"

#include <math.h>

static int row_start(const PresolveSparseRowView *matrix, int row)
{
    return matrix->row_starts[(size_t) row * matrix->row_range_stride];
}

PresolveParallelReductionStatus presolve_analyze_parallel_row_group(
    const PresolveSparseRowView *matrix, const int *rows, size_t count,
    const double *lower, const double *upper, double tolerance,
    PresolveValuesClose values_close, const void *close_context,
    PresolveParallelRowReduction *reduction)
{
    double kept_coefficient;
    size_t position;

    if (!matrix || !rows || count < 2 || !lower || !upper || !values_close ||
        !reduction || !isfinite(tolerance) || tolerance < 0.0)
        return PRESOLVE_PARALLEL_REDUCTION_NUMERICAL_ERROR;

    reduction->kept_row = rows[0];
    for (position = 0; position < count; ++position)
    {
        int row = rows[position];
        if (isfinite(lower[row]) && isfinite(upper[row]) &&
            values_close(close_context, lower[row], upper[row], tolerance))
        {
            reduction->kept_row = row;
            break;
        }
    }
    reduction->lower = lower[reduction->kept_row];
    reduction->upper = upper[reduction->kept_row];
    reduction->lower_source_row = -1;
    reduction->upper_source_row = -1;
    reduction->lower_source_ratio = 1.0;
    reduction->upper_source_ratio = 1.0;
    kept_coefficient =
        matrix->values[row_start(matrix, reduction->kept_row)];

    for (position = 0; position < count; ++position)
    {
        int row = rows[position];
        double ratio, scaled_lower, scaled_upper;
        if (row == reduction->kept_row) continue;
        ratio = kept_coefficient / matrix->values[row_start(matrix, row)];
        if (!isfinite(ratio) || ratio == 0.0)
            return PRESOLVE_PARALLEL_REDUCTION_NUMERICAL_ERROR;
        if (ratio > 0.0)
        {
            scaled_lower = ratio * lower[row];
            scaled_upper = ratio * upper[row];
        }
        else
        {
            scaled_lower = ratio * upper[row];
            scaled_upper = ratio * lower[row];
        }
        if (isnan(scaled_lower) || isnan(scaled_upper))
            return PRESOLVE_PARALLEL_REDUCTION_NUMERICAL_ERROR;
        if (scaled_lower > reduction->lower)
        {
            reduction->lower = scaled_lower;
            reduction->lower_source_row = row;
            reduction->lower_source_ratio = ratio;
        }
        if (scaled_upper < reduction->upper)
        {
            reduction->upper = scaled_upper;
            reduction->upper_source_row = row;
            reduction->upper_source_ratio = ratio;
        }
    }

    if (reduction->lower > reduction->upper)
    {
        if (values_close(close_context, reduction->lower, reduction->upper,
                         tolerance))
            return PRESOLVE_PARALLEL_REDUCTION_UNCERTAIN;
        return PRESOLVE_PARALLEL_REDUCTION_INFEASIBLE;
    }
    return PRESOLVE_PARALLEL_REDUCTION_OK;
}
