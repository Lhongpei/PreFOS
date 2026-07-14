/*
 * Copyright 2026 Hongpei Li
 * SPDX-License-Identifier: Apache-2.0
 */

#include "PREFOS_PowerCone.h"

#include <float.h>
#include <math.h>

static void update_violation(long double candidate, long double *violation)
{
    if (isnan(candidate))
        *violation = INFINITY;
    else if (candidate > *violation)
        *violation = candidate;
}

static PreFOSStatus evaluate_primal(long double x, long double y, long double z,
                                 long double alpha, long double *violation)
{
    long double radial = 0.0L;
    long double result = 0.0L;
    if (!isfinite(x) || !isfinite(y) || !isfinite(z) || !isfinite(alpha) ||
        !(alpha > 0.0L && alpha < 1.0L) || !violation)
        return PREFOS_STATUS_INVALID_ARGUMENT;

    update_violation(-x, &result);
    update_violation(-y, &result);
    if (x > 0.0L && y > 0.0L)
    {
        long double log_value = alpha * logl(x) + (1.0L - alpha) * logl(y);
        if (log_value > logl(LDBL_MAX))
            radial = INFINITY;
        else if (log_value >= logl(LDBL_MIN))
            radial = expl(log_value);
    }
    update_violation(fabsl(z) - radial, &result);
    *violation = result;
    return PREFOS_STATUS_OK;
}

PreFOSStatus prefos_internal_power_cone_violation(const PreFOSConeBlock *cone,
                                            const double *point, int dual,
                                            long double *violation)
{
    long double x, y, z, alpha;
    if (!cone || !point || !violation || cone->dimension != 3)
        return PREFOS_STATUS_INVALID_ARGUMENT;
    alpha = (long double) cone->power_alpha;
    x = (long double) point[cone->indices[0]];
    y = (long double) point[cone->indices[1]];
    z = (long double) point[cone->indices[2]];
    if (dual)
    {
        x /= alpha;
        y /= 1.0L - alpha;
    }
    return evaluate_primal(x, y, z, alpha, violation);
}

int prefos_internal_power_coordinate_is_nonnegative(size_t position)
{
    return position < 2;
}

static int primal_contains(long double x, long double y, long double z,
                           long double alpha)
{
    long double left, right, margin;
    if (!isfinite(x) || !isfinite(y) || !isfinite(z) || x < 0.0L || y < 0.0L)
        return 0;
    if (z == 0.0L) return 1;
    if (x == 0.0L || y == 0.0L) return 0;
    left = alpha * logl(x) + (1.0L - alpha) * logl(y);
    right = logl(fabsl(z));
    margin = 64.0L * LDBL_EPSILON * fmaxl(1.0L, fmaxl(fabsl(left), fabsl(right)));
    return left >= right + margin;
}

int prefos_internal_power_cone_contains(const PreFOSConeBlock *cone, const double *point,
                                     int dual)
{
    long double x, y, z, alpha;
    if (!cone || !point || cone->dimension != 3 ||
        !(cone->power_alpha > 0.0 && cone->power_alpha < 1.0))
        return 0;
    alpha = (long double) cone->power_alpha;
    x = (long double) point[cone->indices[0]];
    y = (long double) point[cone->indices[1]];
    z = (long double) point[cone->indices[2]];
    if (dual)
    {
        x /= alpha;
        y /= 1.0L - alpha;
    }
    return primal_contains(x, y, z, alpha);
}

PreFOSStatus prefos_internal_power_capacity(double alpha, double upper_x, double upper_y,
                                      long double *capacity)
{
    long double log_capacity;
    if (!capacity || !(alpha > 0.0 && alpha < 1.0) || isnan(upper_x) ||
        isnan(upper_y) || upper_x < 0.0 || upper_y < 0.0)
        return PREFOS_STATUS_INVALID_ARGUMENT;
    if (upper_x == 0.0 || upper_y == 0.0)
    {
        *capacity = 0.0L;
        return PREFOS_STATUS_OK;
    }
    if (isinf(upper_x) || isinf(upper_y))
    {
        *capacity = INFINITY;
        return PREFOS_STATUS_OK;
    }
    log_capacity = (long double) alpha * logl((long double) upper_x) +
                   (1.0L - (long double) alpha) * logl((long double) upper_y);
    *capacity = log_capacity > logl(LDBL_MAX) ? INFINITY : expl(log_capacity);
    if (isfinite(*capacity))
        *capacity += 128.0L * LDBL_EPSILON * fmaxl(1.0L, fabsl(*capacity));
    return PREFOS_STATUS_OK;
}

PreFOSStatus prefos_internal_power_implied_axis_lower(double alpha,
                                                long double minimum_abs_z,
                                                double other_upper, int infer_x,
                                                long double *lower)
{
    long double exponent =
        infer_x ? (long double) alpha : 1.0L - (long double) alpha;
    long double other_exponent = 1.0L - exponent;
    long double log_lower;
    if (!lower || !(alpha > 0.0 && alpha < 1.0) || minimum_abs_z < 0.0L ||
        isnan(minimum_abs_z) || isnan(other_upper) || other_upper < 0.0)
        return PREFOS_STATUS_INVALID_ARGUMENT;
    if (minimum_abs_z == 0.0L || isinf(other_upper))
    {
        *lower = 0.0L;
        return PREFOS_STATUS_OK;
    }
    if (other_upper == 0.0)
    {
        *lower = INFINITY;
        return PREFOS_STATUS_OK;
    }
    log_lower =
        (logl(minimum_abs_z) - other_exponent * logl((long double) other_upper)) /
        exponent;
    *lower = log_lower > logl(LDBL_MAX) ? INFINITY : expl(log_lower);
    if (isfinite(*lower))
        *lower =
            fmaxl(0.0L, *lower - 128.0L * LDBL_EPSILON * fmaxl(1.0L, fabsl(*lower)));
    return PREFOS_STATUS_OK;
}
