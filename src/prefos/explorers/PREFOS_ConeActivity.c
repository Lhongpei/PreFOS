/*
 * Copyright 2026 Hongpei Li
 * SPDX-License-Identifier: Apache-2.0
 */

#include "PREFOS_ConeActivity.h"

#include "cones/PREFOS_ExponentialCone.h"
#include "cones/PREFOS_PowerCone.h"

static int add_bound(double *sum, long double value, int lower, int outward)
{
    long double old_sum, result;
    double converted;
    if (!isfinite(value)) return 0;
    if (!outward)
    {
        converted = (double) value;
        if (!isfinite(converted) || !isfinite(*sum + converted))
            return 0;
        *sum += converted;
        return 1;
    }
    old_sum = (long double) *sum;
    result = old_sum + value;
    if (!isfinite(result)) return 0;
    if (outward)
    {
        long double recovered = result - old_sum;
        long double error = (old_sum - (result - recovered)) + (value - recovered);
        if ((lower && error < 0.0L) || (!lower && error > 0.0L))
            result = nextafterl(result, lower ? -INFINITY : INFINITY);
    }
    converted = (double) result;
    if (!isfinite(converted)) return 0;
    if (outward && lower && (long double) converted > result)
        converted = nextafter(converted, -INFINITY);
    if (outward && !lower && (long double) converted < result)
        converted = nextafter(converted, INFINITY);
    *sum = converted;
    return 1;
}

static int add_scalar_term(double *sum, double coefficient, double bound,
                           int lower, int outward)
{
    if (!outward)
    {
        double product = coefficient * bound;
        if (!isfinite(product) || !isfinite(*sum + product))
            return 0;
        *sum += product;
        return 1;
    }
    long double left = (long double) coefficient;
    long double right = (long double) bound;
    long double product = left * right;
    if (!isfinite(product)) return 0;
    if (outward)
    {
        long double error = fmal(left, right, -product);
        if ((lower && error < 0.0L) || (!lower && error > 0.0L))
            product = nextafterl(product, lower ? -INFINITY : INFINITY);
    }
    return add_bound(sum, product, lower, outward);
}

static int soc_contains(const PreFOSConeBlock *cone, const double *point)
{
    long double norm = 0.0L;
    long double t = (long double) point[cone->indices[0]];
    long double margin;
    size_t i;
    if (!isfinite(t) || t < 0.0L) return 0;
    for (i = 1; i < cone->dimension; ++i)
    {
        long double value = (long double) point[cone->indices[i]];
        if (!isfinite(value)) return 0;
        norm = hypotl(norm, value);
    }
    if (norm == 0.0L) return 1;
    margin = 64.0L * LDBL_EPSILON * fmaxl(1.0L, fmaxl(t, norm));
    return t >= norm + margin;
}

static int rsoc_contains(const PreFOSConeBlock *cone, const double *point)
{
    long double u = (long double) point[cone->indices[0]];
    long double v = (long double) point[cone->indices[1]];
    long double norm_squared = 0.0L;
    long double product, margin;
    size_t i;
    if (!isfinite(u) || !isfinite(v) || u < 0.0L || v < 0.0L) return 0;
    for (i = 2; i < cone->dimension; ++i)
    {
        long double value = (long double) point[cone->indices[i]];
        if (!isfinite(value)) return 0;
        norm_squared += value * value;
        if (!isfinite(norm_squared)) return 0;
    }
    if (norm_squared == 0.0L) return 1;
    product = 2.0L * u * v;
    if (!isfinite(product)) return 0;
    margin = 128.0L * LDBL_EPSILON * fmaxl(1.0L, fmaxl(product, norm_squared));
    return product >= norm_squared + margin;
}

static long double psd_entry(const PreFOSConeBlock *cone, const double *point,
                             size_t row, size_t column)
{
    size_t packed;
    if (column > row)
    {
        size_t swap = row;
        row = column;
        column = swap;
    }
    packed = row * (row + 1) / 2 + column;
    return (long double) point[cone->indices[packed]] /
           (row == column ? 1.0L : sqrtl(2.0L));
}

static int psd_contains(const PreFOSConeBlock *cone, const double *point)
{
    size_t order = cone->matrix_order;
    long double *factor;
    size_t i, j, k;
    int diagonal = 1, diagonally_dominant = 1;
    if (order > 32) return 0;
    for (i = 0; i < order; ++i)
    {
        long double diagonal_value = psd_entry(cone, point, i, i);
        long double off_diagonal_sum = 0.0L;
        if (!isfinite(diagonal_value) || diagonal_value < 0.0L) return 0;
        for (j = 0; j < order; ++j)
        {
            long double value;
            if (j == i) continue;
            value = psd_entry(cone, point, i, j);
            if (!isfinite(value)) return 0;
            if (value != 0.0L) diagonal = 0;
            off_diagonal_sum += fabsl(value);
        }
        if (off_diagonal_sum > 0.0L &&
            diagonal_value <
                off_diagonal_sum +
                    128.0L * LDBL_EPSILON *
                        fmaxl(1.0L, fmaxl(diagonal_value, off_diagonal_sum)))
            diagonally_dominant = 0;
    }
    if (diagonal || diagonally_dominant) return 1;

    if (order > 16) return 0;
    if (order > SIZE_MAX / order) return 0;
    factor = (long double *) calloc(order * order, sizeof(long double));
    if (!factor) return 0;
    for (i = 0; i < order; ++i)
    {
        for (j = 0; j <= i; ++j)
        {
            long double value = psd_entry(cone, point, i, j);
            long double scale;
            for (k = 0; k < j; ++k)
                value -= factor[i * order + k] * factor[j * order + k];
            scale = fmaxl(1.0L, fabsl(psd_entry(cone, point, i, i)));
            if (i == j)
            {
                if (value <= 128.0L * LDBL_EPSILON * scale)
                {
                    free(factor);
                    return 0;
                }
                factor[i * order + j] = sqrtl(value);
            }
            else
                factor[i * order + j] = value / factor[j * order + j];
        }
    }
    free(factor);
    return 1;
}

static int dual_contains(const PreFOSConeBlock *cone, const double *point)
{
    size_t i;
    if (cone->type == PREFOS_CONE_NONNEGATIVE)
    {
        for (i = 0; i < cone->dimension; ++i)
            if (point[cone->indices[i]] < 0.0) return 0;
        return 1;
    }
    if (cone->type == PREFOS_CONE_SECOND_ORDER) return soc_contains(cone, point);
    if (cone->type == PREFOS_CONE_ROTATED_SECOND_ORDER)
        return rsoc_contains(cone, point);
    if (cone->type == PREFOS_CONE_POSITIVE_SEMIDEFINITE)
        return psd_contains(cone, point);
    if (cone->type == PREFOS_CONE_EXPONENTIAL)
        return prefos_internal_exponential_cone_contains(cone, point, 1);
    return prefos_internal_power_cone_contains(cone, point, 1);
}

PreFOSStatus
prefos_internal_cone_activity_workspace_init(const PreFOSPresolver *presolver,
                                          PreFOSConeActivityWorkspace *workspace)
{
    size_t i, k;
    memset(workspace, 0, sizeof(*workspace));
    workspace->lower_bounds = presolver->propagation_lower;
    workspace->upper_bounds = presolver->propagation_upper;
    workspace->column_to_cone =
        (int *) prefos_internal_alloc_array(presolver->original.n, sizeof(int));
    workspace->coefficients =
        (double *) calloc(presolver->original.n, sizeof(double));
    workspace->touched_cones =
        (int *) prefos_internal_alloc_array(presolver->original.n_cones, sizeof(int));
    workspace->cone_touched =
        (unsigned char *) calloc(presolver->original.n_cones, sizeof(unsigned char));
    if ((presolver->original.n > 0 &&
         (!workspace->column_to_cone || !workspace->coefficients)) ||
        (presolver->original.n_cones > 0 &&
         (!workspace->touched_cones || !workspace->cone_touched)))
    {
        prefos_internal_cone_activity_workspace_free(workspace);
        return PREFOS_STATUS_OUT_OF_MEMORY;
    }
    for (i = 0; i < presolver->original.n; ++i) workspace->column_to_cone[i] = -1;
    for (k = 0; k < presolver->original.n_cones; ++k)
    {
        const PreFOSConeBlock *cone = &presolver->original.cones[k];
        if (presolver->converted_affine_cones[k]) continue;
        for (i = 0; i < cone->dimension; ++i)
            workspace->column_to_cone[cone->indices[i]] = (int) k;
    }
    return PREFOS_STATUS_OK;
}

void prefos_internal_cone_activity_workspace_free(PreFOSConeActivityWorkspace *workspace)
{
    if (!workspace) return;
    free(workspace->column_to_cone);
    free(workspace->coefficients);
    free(workspace->touched_cones);
    free(workspace->cone_touched);
    memset(workspace, 0, sizeof(*workspace));
}

static int standard_dual_summary_contains(
    const PreFOSConeBlock *cone, int sign, int sign_compatible,
    long double first, long double second, long double norm,
    long double norm_squared)
{
    first *= (long double) sign;
    second *= (long double) sign;
    if (cone->type == PREFOS_CONE_NONNEGATIVE) return sign_compatible;
    if (cone->type == PREFOS_CONE_SECOND_ORDER)
    {
        long double margin;
        if (first < 0.0L) return 0;
        if (norm == 0.0L) return 1;
        margin =
            64.0L * LDBL_EPSILON * fmaxl(1.0L, fmaxl(first, norm));
        return first >= norm + margin;
    }
    if (first < 0.0L || second < 0.0L) return 0;
    if (norm_squared == 0.0L) return 1;
    {
        long double product = 2.0L * first * second;
        long double margin;
        if (!isfinite(product)) return 0;
        margin = 128.0L * LDBL_EPSILON *
                 fmaxl(1.0L, fmaxl(product, norm_squared));
        return product >= norm_squared + margin;
    }
}

static int add_cone_group(PreFOSPresolver *presolver, int cone_index,
                          size_t row, int outward,
                          int count_statistics,
                          PreFOSConeActivityWorkspace *workspace,
                          PresolveLinearActivity *activity)
{
    const PreFOSCsrMatrix *A = &presolver->original.A;
    const PreFOSConeBlock *cone =
        &presolver->original.cones[cone_index];
    double scalar_min = 0.0, scalar_max = 0.0;
    size_t infinite_min = 0, infinite_max = 0;
    long double first = 0.0L, second = 0.0L;
    long double norm = 0.0L, norm_squared = 0.0L;
    int position;
    int nonnegative_lower = 1, nonnegative_upper = 1;
    int lower_supported, upper_supported;
    if (count_statistics)
        ++presolver->stats.cone_activity_blocks;
    for (position = A->row_pointers[row];
         position < A->row_pointers[row + 1]; ++position)
    {
        int column = A->column_indices[position];
        double coefficient = workspace->coefficients[column];
        double min_bound, max_bound;
        if (coefficient == 0.0 ||
            workspace->column_to_cone[column] != cone_index)
            continue;
        if (cone->type == PREFOS_CONE_NONNEGATIVE)
        {
            if (coefficient < 0.0) nonnegative_lower = 0;
            if (coefficient > 0.0) nonnegative_upper = 0;
        }
        else if (cone->type == PREFOS_CONE_SECOND_ORDER)
        {
            if (column == cone->indices[0])
                first = (long double) coefficient;
            else
                norm = hypotl(norm, (long double) coefficient);
        }
        else if (cone->type == PREFOS_CONE_ROTATED_SECOND_ORDER)
        {
            if (column == cone->indices[0])
                first = (long double) coefficient;
            else if (column == cone->indices[1])
                second = (long double) coefficient;
            else
            {
                long double value = (long double) coefficient;
                norm_squared += value * value;
            }
        }
        min_bound = coefficient > 0.0 ? workspace->lower_bounds[column]
                                      : workspace->upper_bounds[column];
        max_bound = coefficient > 0.0 ? workspace->upper_bounds[column]
                                      : workspace->lower_bounds[column];
        if (isfinite(min_bound))
        {
            if (!add_scalar_term(&scalar_min, coefficient, min_bound, 1, outward))
                return 0;
        }
        else
            ++infinite_min;
        if (isfinite(max_bound))
        {
            if (!add_scalar_term(&scalar_max, coefficient, max_bound, 0, outward))
                return 0;
        }
        else
            ++infinite_max;
    }

    if (cone->type == PREFOS_CONE_NONNEGATIVE ||
        cone->type == PREFOS_CONE_SECOND_ORDER ||
        cone->type == PREFOS_CONE_ROTATED_SECOND_ORDER)
    {
        lower_supported = standard_dual_summary_contains(
            cone, 1, nonnegative_lower, first, second, norm,
            norm_squared);
        upper_supported = standard_dual_summary_contains(
            cone, -1, nonnegative_upper, first, second, norm,
            norm_squared);
    }
    else
    {
        lower_supported =
            cone->dimension <= 4096 &&
            dual_contains(cone, workspace->coefficients);
        if (cone->dimension <= 4096)
        {
            for (position = A->row_pointers[row];
                 position < A->row_pointers[row + 1]; ++position)
            {
                int column = A->column_indices[position];
                if (workspace->column_to_cone[column] == cone_index)
                    workspace->coefficients[column] =
                        -workspace->coefficients[column];
            }
        }
        upper_supported =
            cone->dimension <= 4096 &&
            dual_contains(cone, workspace->coefficients);
        if (cone->dimension <= 4096)
        {
            for (position = A->row_pointers[row];
                 position < A->row_pointers[row + 1]; ++position)
            {
                int column = A->column_indices[position];
                if (workspace->column_to_cone[column] == cone_index)
                    workspace->coefficients[column] =
                        -workspace->coefficients[column];
            }
        }
    }
    if (count_statistics && lower_supported)
        ++presolver->stats.cone_activity_lower_support_hits;
    if (count_statistics && upper_supported)
        ++presolver->stats.cone_activity_upper_support_hits;

    if (lower_supported)
    {
        long double lower = infinite_min == 0 ? fmaxl(0.0L, scalar_min) : 0.0L;
        if (infinite_min > 0 || scalar_min < 0.0L)
            workspace->row_support_strengthened = 1;
        if (!add_bound(&activity->finite_min, lower, 1, outward)) return 0;
    }
    else if (infinite_min > 0)
        ++activity->n_infinite_min;
    else if (!add_bound(&activity->finite_min, scalar_min, 1, outward))
        return 0;

    if (upper_supported)
    {
        long double upper = infinite_max == 0 ? fminl(0.0L, scalar_max) : 0.0L;
        if (infinite_max > 0 || scalar_max > 0.0L)
            workspace->row_support_strengthened = 1;
        if (!add_bound(&activity->finite_max, upper, 0, outward)) return 0;
    }
    else if (infinite_max > 0)
        ++activity->n_infinite_max;
    else if (!add_bound(&activity->finite_max, scalar_max, 0, outward))
        return 0;
    return 1;
}

PreFOSStatus prefos_internal_compute_cone_aware_row_activity(
    PreFOSPresolver *presolver, size_t row, int outward,
    int count_statistics, PreFOSConeActivityWorkspace *workspace,
    PresolveLinearActivity *activity)
{
    const PreFOSCsrMatrix *A = &presolver->original.A;
    size_t n_touched = 0, position;
    int p;
    if (count_statistics)
        ++presolver->stats.cone_activity_rows;
    workspace->row_support_strengthened = 0;
    memset(activity, 0, sizeof(*activity));
    for (p = A->row_pointers[row]; p < A->row_pointers[row + 1]; ++p)
    {
        int column = A->column_indices[p];
        double coefficient = A->values[p];
        int cone_index;
        double min_bound, max_bound;
        if (coefficient == 0.0 ||
            !prefos_internal_term_is_active_in_row(
                presolver, row, column))
            continue;
        ++activity->n_nonzeros;
        cone_index = workspace->column_to_cone[column];
        if (cone_index >= 0)
        {
            workspace->coefficients[column] = coefficient;
            if (!workspace->cone_touched[cone_index])
            {
                workspace->cone_touched[cone_index] = 1;
                workspace->touched_cones[n_touched++] = cone_index;
            }
            continue;
        }
        min_bound = coefficient > 0.0 ? workspace->lower_bounds[column]
                                      : workspace->upper_bounds[column];
        max_bound = coefficient > 0.0 ? workspace->upper_bounds[column]
                                      : workspace->lower_bounds[column];
        if (isfinite(min_bound))
        {
            if (!add_scalar_term(&activity->finite_min, coefficient, min_bound, 1,
                                 outward))
                return PREFOS_STATUS_NUMERICAL_ERROR;
        }
        else
            ++activity->n_infinite_min;
        if (isfinite(max_bound))
        {
            if (!add_scalar_term(&activity->finite_max, coefficient, max_bound, 0,
                                 outward))
                return PREFOS_STATUS_NUMERICAL_ERROR;
        }
        else
            ++activity->n_infinite_max;
    }
    if (n_touched == 0) return PREFOS_STATUS_OK;
    for (position = 0; position < n_touched; ++position)
    {
        int cone_index = workspace->touched_cones[position];
        if (!add_cone_group(
                presolver, cone_index, row, outward,
                count_statistics, workspace, activity))
            return PREFOS_STATUS_NUMERICAL_ERROR;
        workspace->cone_touched[cone_index] = 0;
    }
    for (p = A->row_pointers[row]; p < A->row_pointers[row + 1]; ++p)
        workspace->coefficients[A->column_indices[p]] = 0.0;
    if (count_statistics && workspace->row_support_strengthened)
        ++presolver->stats.cone_activity_strengthened_rows;
    return PREFOS_STATUS_OK;
}
