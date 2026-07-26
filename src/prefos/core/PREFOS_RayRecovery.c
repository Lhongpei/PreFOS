/*
 * Copyright 2026 Hongpei Li
 * SPDX-License-Identifier: Apache-2.0
 */

#include "PREFOS_CertificateInternal.h"

static void recession_interval(double lower, double upper,
                               double *direction_lower,
                               double *direction_upper)
{
    if (isfinite(lower) && isfinite(upper))
        *direction_lower = *direction_upper = 0.0;
    else if (isfinite(lower))
    {
        *direction_lower = 0.0;
        *direction_upper = INFINITY;
    }
    else if (isfinite(upper))
    {
        *direction_lower = -INFINITY;
        *direction_upper = 0.0;
    }
    else
    {
        *direction_lower = -INFINITY;
        *direction_upper = INFINITY;
    }
}

static PreFOSStatus replay_parallel_column_ray(
    const PresolveColumnTransformationRecord *record, size_t columns,
    double tolerance, double *ray)
{
    double source_lower, source_upper, target_lower, target_upper;
    double feasible_lower, feasible_upper;
    double aggregate, source_direction, target_direction;
    double ratio = record->ratio;

    if (record->column < 0 || record->secondary_column < 0 ||
        (size_t) record->column >= columns ||
        (size_t) record->secondary_column >= columns ||
        ratio == 0.0 || !isfinite(ratio))
        return PREFOS_STATUS_NUMERICAL_ERROR;
    aggregate = ray[record->secondary_column];
    recession_interval(
        record->lower, record->upper, &source_lower, &source_upper);
    recession_interval(
        record->secondary_lower, record->secondary_upper,
        &target_lower, &target_upper);
    feasible_lower = source_lower;
    feasible_upper = source_upper;
    if (ratio > 0.0)
    {
        if (isfinite(target_upper))
            feasible_lower =
                fmax(feasible_lower,
                     (aggregate - target_upper) / ratio);
        if (isfinite(target_lower))
            feasible_upper =
                fmin(feasible_upper,
                     (aggregate - target_lower) / ratio);
    }
    else
    {
        if (isfinite(target_lower))
            feasible_lower =
                fmax(feasible_lower,
                     (aggregate - target_lower) / ratio);
        if (isfinite(target_upper))
            feasible_upper =
                fmin(feasible_upper,
                     (aggregate - target_upper) / ratio);
    }
    if (feasible_lower > feasible_upper)
    {
        double scale =
            fmax(1.0, fmax(fabs(feasible_lower), fabs(feasible_upper)));
        if (!isfinite(scale) ||
            feasible_lower - feasible_upper > tolerance * scale)
            return PREFOS_STATUS_NUMERICAL_ERROR;
        feasible_lower = feasible_upper =
            prefos_internal_safe_midpoint(
                feasible_lower, feasible_upper);
    }
    if (feasible_lower <= 0.0 && feasible_upper >= 0.0)
        source_direction = 0.0;
    else if (feasible_lower > 0.0)
        source_direction = feasible_lower;
    else
        source_direction = feasible_upper;
    target_direction = aggregate - ratio * source_direction;
    if (!isfinite(source_direction) || !isfinite(target_direction))
        return PREFOS_STATUS_NUMERICAL_ERROR;
    ray[record->column] = source_direction;
    ray[record->secondary_column] = target_direction;
    return PREFOS_STATUS_OK;
}

static PreFOSStatus replay_infinite_fixed_column_ray(
    const PresolveColumnTransformationRecord *record, size_t columns,
    double *ray)
{
    long double extreme = 0.0L;
    size_t row;
    if (record->column < 0 ||
        (size_t) record->column >= columns ||
        (record->direction != -1 && record->direction != 1) ||
        (record->n_rows > 0 &&
         (!record->row_starts || !record->row_sides)) ||
        (record->length > 0 &&
         (!record->indices || !record->coefficients)))
        return PREFOS_STATUS_NUMERICAL_ERROR;
    for (row = 0; row < record->n_rows; ++row)
    {
        int start = record->row_starts[row];
        int end = record->row_starts[row + 1];
        long double residual = 0.0L;
        long double pivot = 0.0L;
        int position;
        if (start < 0 || end < start ||
            (size_t) end > record->length)
            return PREFOS_STATUS_NUMERICAL_ERROR;
        for (position = start; position < end; ++position)
        {
            int column = record->indices[position];
            double coefficient = record->coefficients[position];
            if (column < 0 || (size_t) column >= columns ||
                !isfinite(coefficient))
                return PREFOS_STATUS_NUMERICAL_ERROR;
            if (column == record->column)
                pivot += (long double) coefficient;
            else
                residual -=
                    (long double) coefficient *
                    (long double) ray[column];
        }
        if (pivot == 0.0L || !isfinite(pivot) ||
            !isfinite(residual))
            return PREFOS_STATUS_NUMERICAL_ERROR;
        residual /= pivot;
        if (record->direction > 0)
            extreme = fmaxl(extreme, residual);
        else
            extreme = fminl(extreme, residual);
    }
    if (!isfinite(extreme) ||
        fabsl(extreme) > (long double) DBL_MAX)
        return PREFOS_STATUS_NUMERICAL_ERROR;
    ray[record->column] = (double) extreme;
    return PREFOS_STATUS_OK;
}

static PreFOSStatus replay_column_ray(
    const PreFOSPresolver *presolver,
    const PresolveColumnTransformationRecord *record, double tolerance,
    double *ray)
{
    size_t term_count, start, term;
    long double direction = 0.0L;
    if (record->type == PRESOLVE_COLUMN_FIXED_INFINITE)
        return replay_infinite_fixed_column_ray(
            record, presolver->original.n, ray);
    if (record->type == PRESOLVE_COLUMNS_PARALLEL)
        return replay_parallel_column_ray(
            record, presolver->original.n, tolerance, ray);
    if (record->type == PRESOLVE_COLUMN_FIXED)
    {
        if (record->column < 0 ||
            (size_t) record->column >= presolver->original.n)
            return PREFOS_STATUS_NUMERICAL_ERROR;
        ray[record->column] = 0.0;
        return PREFOS_STATUS_OK;
    }
    if (record->type != PRESOLVE_COLUMN_SUBSTITUTED ||
        record->column < 0 ||
        (size_t) record->column >= presolver->original.n)
        return PREFOS_STATUS_NUMERICAL_ERROR;
    term_count =
        presolver->substitution_term_count[record->column];
    start = presolver->substitution_term_start[record->column];
    if (term_count == 0 ||
        start > presolver->n_substitution_terms ||
        term_count > presolver->n_substitution_terms - start)
        return PREFOS_STATUS_NUMERICAL_ERROR;
    for (term = 0; term < term_count; ++term)
    {
        int target = presolver->substitution_targets[start + term];
        double scale = presolver->substitution_scales[start + term];
        if (target < 0 ||
            (size_t) target >= presolver->original.n ||
            !isfinite(scale))
            return PREFOS_STATUS_NUMERICAL_ERROR;
        direction +=
            (long double) scale * (long double) ray[target];
    }
    if (!isfinite(direction) ||
        fabsl(direction) > (long double) DBL_MAX)
        return PREFOS_STATUS_NUMERICAL_ERROR;
    ray[record->column] = (double) direction;
    return PREFOS_STATUS_OK;
}

PreFOSStatus prefos_postsolve_unbounded_ray(
    const PreFOSPresolver *presolver, const double *reduced_ray,
    double tolerance, double *original_ray)
{
    size_t column, event_index;
    if (!presolver || !presolver->has_run || !isfinite(tolerance) ||
        tolerance < 0.0 ||
        (presolver->reduced.n > 0 && !reduced_ray) ||
        (presolver->original.n > 0 && !original_ray) ||
        !prefos_internal_certificate_vector_is_finite(
            reduced_ray, presolver->reduced.n))
        return PREFOS_STATUS_INVALID_ARGUMENT;
    for (column = 0; column < presolver->original.n; ++column)
    {
        int mapped = presolver->original_to_reduced[column];
        original_ray[column] =
            mapped < 0 ? 0.0 : reduced_ray[mapped];
    }
    for (event_index = presolver->transformations.n_events;
         event_index > 0; --event_index)
    {
        const PresolveTransformationEvent *event =
            &presolver->transformations.events[event_index - 1];
        PreFOSStatus status;
        if (event->type != PRESOLVE_TRANSFORMATION_COLUMN) continue;
        status = replay_column_ray(
            presolver,
            &presolver->transformations
                 .column_transformations[event->record_index],
            tolerance, original_ray);
        if (status != PREFOS_STATUS_OK) return status;
    }
    return prefos_internal_certificate_vector_is_finite(
               original_ray, presolver->original.n)
               ? PREFOS_STATUS_OK
               : PREFOS_STATUS_NUMERICAL_ERROR;
}
