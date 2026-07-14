/*
 * Copyright 2026 Hongpei Li
 * SPDX-License-Identifier: Apache-2.0
 */

#include "PREFOS_ExponentialCone.h"

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
                                 long double *violation)
{
    long double result = 0.0L;
    if (!isfinite(x) || !isfinite(y) || !isfinite(z) || !violation)
        return PREFOS_STATUS_INVALID_ARGUMENT;

    update_violation(-y, &result);
    update_violation(-z, &result);
    if (y > 0.0L)
    {
        long double log_value = logl(y) + x / y;
        long double exponential_value;
        if (log_value > logl(LDBL_MAX))
            exponential_value = INFINITY;
        else if (log_value < logl(LDBL_MIN))
            exponential_value = 0.0L;
        else
            exponential_value = expl(log_value);
        update_violation(exponential_value - z, &result);
    }
    else
        update_violation(x, &result);
    *violation = result;
    return PREFOS_STATUS_OK;
}

PreFOSStatus prefos_internal_exponential_cone_violation(const PreFOSConeBlock *cone,
                                                  const double *point, int dual,
                                                  long double *violation)
{
    long double x, y, z;
    if (!cone || !point || !violation || cone->dimension != 3)
        return PREFOS_STATUS_INVALID_ARGUMENT;
    x = (long double) point[cone->indices[0]];
    y = (long double) point[cone->indices[1]];
    z = (long double) point[cone->indices[2]];
    if (dual) return evaluate_primal(-y, -x, expl(1.0L) * z, violation);
    return evaluate_primal(x, y, z, violation);
}

int prefos_internal_exponential_coordinate_is_nonnegative(size_t position)
{
    return position == 1 || position == 2;
}

static int primal_contains(long double x, long double y, long double z)
{
    long double left, right, margin;
    if (!isfinite(x) || !isfinite(y) || !isfinite(z) || y < 0.0L || z < 0.0L)
        return 0;
    if (y == 0.0L) return x <= 0.0L;
    if (z == 0.0L) return 0;
    left = logl(y) + x / y;
    right = logl(z);
    margin = 64.0L * LDBL_EPSILON * fmaxl(1.0L, fmaxl(fabsl(left), fabsl(right)));
    return left <= right - margin;
}

int prefos_internal_exponential_cone_contains(const PreFOSConeBlock *cone,
                                           const double *point, int dual)
{
    long double x, y, z;
    if (!cone || !point || cone->dimension != 3) return 0;
    x = (long double) point[cone->indices[0]];
    y = (long double) point[cone->indices[1]];
    z = (long double) point[cone->indices[2]];
    return dual ? primal_contains(-y, -x, expl(1.0L) * z) : primal_contains(x, y, z);
}

static long double exp_from_log(long double value)
{
    if (value > logl(LDBL_MAX)) return INFINITY;
    if (value < logl(LDBL_MIN)) return 0.0L;
    return expl(value);
}

PreFOSStatus prefos_internal_exponential_minimum_z(double lower_x, double lower_y,
                                             double upper_y, long double *minimum_z)
{
    long double x = (long double) lower_x;
    long double lower = fmaxl(0.0L, (long double) lower_y);
    long double upper = (long double) upper_y;
    long double y;
    if (!minimum_z || isnan(x) || isnan(lower) || isnan(upper) || lower > upper)
        return PREFOS_STATUS_INVALID_ARGUMENT;
    if (x == -INFINITY || (lower == 0.0L && x <= 0.0L))
    {
        *minimum_z = 0.0L;
        return PREFOS_STATUS_OK;
    }
    if (x == INFINITY || upper <= 0.0L)
    {
        *minimum_z = INFINITY;
        return PREFOS_STATUS_OK;
    }
    if (x > 0.0L)
        y = fmaxl(lower, fminl(upper, x));
    else
        y = lower;
    if (y <= 0.0L)
    {
        *minimum_z = 0.0L;
        return PREFOS_STATUS_OK;
    }
    *minimum_z = exp_from_log(logl(y) + x / y);
    if (isfinite(*minimum_z))
        *minimum_z = fmaxl(0.0L, *minimum_z - 128.0L * LDBL_EPSILON *
                                                  fmaxl(1.0L, *minimum_z));
    return PREFOS_STATUS_OK;
}

PreFOSStatus prefos_internal_exponential_maximum_x(double upper_z, double lower_y,
                                             double upper_y, long double *maximum_x)
{
    long double z = (long double) upper_z;
    long double lower = fmaxl(0.0L, (long double) lower_y);
    long double upper = (long double) upper_y;
    long double y, candidate;
    if (!maximum_x || isnan(z) || isnan(lower) || isnan(upper) || lower > upper)
        return PREFOS_STATUS_INVALID_ARGUMENT;
    if (z == INFINITY)
    {
        *maximum_x = INFINITY;
        return PREFOS_STATUS_OK;
    }
    if (z < 0.0L || (z == 0.0L && lower > 0.0L))
    {
        *maximum_x = -INFINITY;
        return PREFOS_STATUS_OK;
    }
    if (z == 0.0L || upper == 0.0L)
    {
        *maximum_x = 0.0L;
        return PREFOS_STATUS_OK;
    }
    y = fmaxl(lower, fminl(upper, z / expl(1.0L)));
    candidate = y > 0.0L ? y * logl(z / y) : 0.0L;
    *maximum_x = lower == 0.0L ? fmaxl(0.0L, candidate) : candidate;
    *maximum_x += 128.0L * LDBL_EPSILON * fmaxl(1.0L, fabsl(*maximum_x));
    return PREFOS_STATUS_OK;
}
