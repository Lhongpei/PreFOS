/*
 * Copyright 2026 Hongpei Li
 * SPDX-License-Identifier: Apache-2.0
 */

#include "PREFOS_PositiveSemidefiniteCone.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>

PreFOSStatus prefos_internal_psd_minimum_eigenvalue(const PreFOSConeBlock *cone,
                                              const double *point, double tolerance,
                                              long double *minimum_eigenvalue)
{
    size_t order, entries, i, j, iteration, max_iterations;
    long double *matrix;
    if (!cone || !point || !minimum_eigenvalue ||
        cone->type != PREFOS_CONE_POSITIVE_SEMIDEFINITE || !isfinite(tolerance) ||
        tolerance < 0.0)
        return PREFOS_STATUS_INVALID_ARGUMENT;
    order = cone->matrix_order;
    if (order == 0 || order > SIZE_MAX / order) return PREFOS_STATUS_OUT_OF_MEMORY;
    entries = order * order;
    matrix = (long double *) calloc(entries, sizeof(long double));
    if (!matrix) return PREFOS_STATUS_OUT_OF_MEMORY;

    for (i = 0; i < order; ++i)
    {
        for (j = 0; j <= i; ++j)
        {
            size_t packed = i * (i + 1) / 2 + j;
            long double value = (long double) point[cone->indices[packed]];
            if (!isfinite(value))
            {
                free(matrix);
                return PREFOS_STATUS_INVALID_ARGUMENT;
            }
            if (i != j) value /= sqrtl(2.0L);
            matrix[i * order + j] = value;
            matrix[j * order + i] = value;
        }
    }

    max_iterations = entries > (SIZE_MAX - 1) / 50 ? SIZE_MAX : 50 * entries + 1;
    for (iteration = 0; iteration < max_iterations; ++iteration)
    {
        size_t p = 0, q = 0, k;
        long double largest_off_diagonal = 0.0L;
        long double scale = 1.0L;
        for (i = 0; i < order; ++i)
        {
            scale = fmaxl(scale, fabsl(matrix[i * order + i]));
            for (j = i + 1; j < order; ++j)
            {
                long double candidate = fabsl(matrix[i * order + j]);
                if (candidate > largest_off_diagonal)
                {
                    largest_off_diagonal = candidate;
                    p = i;
                    q = j;
                }
            }
        }
        if (largest_off_diagonal <= (long double) tolerance * scale ||
            largest_off_diagonal == 0.0L)
            break;
        {
            long double app = matrix[p * order + p];
            long double aqq = matrix[q * order + q];
            long double apq = matrix[p * order + q];
            long double tau = (aqq - app) / (2.0L * apq);
            long double tangent =
                copysignl(1.0L, tau) / (fabsl(tau) + hypotl(1.0L, tau));
            long double cosine = 1.0L / sqrtl(1.0L + tangent * tangent);
            long double sine = tangent * cosine;
            for (k = 0; k < order; ++k)
            {
                long double akp, akq;
                if (k == p || k == q) continue;
                akp = matrix[k * order + p];
                akq = matrix[k * order + q];
                matrix[k * order + p] = cosine * akp - sine * akq;
                matrix[p * order + k] = matrix[k * order + p];
                matrix[k * order + q] = sine * akp + cosine * akq;
                matrix[q * order + k] = matrix[k * order + q];
            }
            matrix[p * order + p] = app - tangent * apq;
            matrix[q * order + q] = aqq + tangent * apq;
            matrix[p * order + q] = 0.0L;
            matrix[q * order + p] = 0.0L;
        }
    }
    *minimum_eigenvalue = matrix[0];
    for (i = 1; i < order; ++i)
        *minimum_eigenvalue = fminl(*minimum_eigenvalue, matrix[i * order + i]);
    free(matrix);
    return PREFOS_STATUS_OK;
}

PreFOSStatus prefos_internal_evaluate_psd_violation(const PreFOSConeBlock *cone,
                                              const double *point, double tolerance,
                                              long double *violation)
{
    long double minimum_eigenvalue;
    PreFOSStatus status;
    if (!violation) return PREFOS_STATUS_INVALID_ARGUMENT;
    status = prefos_internal_psd_minimum_eigenvalue(cone, point, tolerance,
                                                 &minimum_eigenvalue);
    if (status != PREFOS_STATUS_OK) return status;
    if (-minimum_eigenvalue > *violation) *violation = -minimum_eigenvalue;
    return PREFOS_STATUS_OK;
}
