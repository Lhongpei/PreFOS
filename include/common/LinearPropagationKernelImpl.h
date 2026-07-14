/*
 * Copyright 2025-2026 Daniel Cederberg
 * Copyright 2026 Hongpei Li
 *
 * Modified for PreFOS in 2026.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef LINEAR_PROPAGATION_KERNEL_IMPL_H
#define LINEAR_PROPAGATION_KERNEL_IMPL_H

#include <math.h>
#include <string.h>

static inline long double scaled_tolerance(double tolerance, double reference)
{
    return (long double) tolerance * fmaxl(1.0L, fabsl((long double) reference));
}

static inline PresolveLinearRowState presolve_internal_classify_linear_row(
    const PresolveLinearActivity *activity, double lower, double upper,
    double feasibility_tolerance, double redundancy_tolerance)
{
    PresolveLinearRowState state = PRESOLVE_ROW_FEASIBLE;

    if (isfinite(upper))
    {
        long double side = (long double) upper;
        if (activity->n_infinite_min == 0 &&
            activity->finite_min >
                side + scaled_tolerance(feasibility_tolerance, upper))
            return PRESOLVE_ROW_INFEASIBLE;
        if (activity->n_infinite_max == 0 &&
            activity->finite_max <=
                side + scaled_tolerance(redundancy_tolerance, upper))
            state = (PresolveLinearRowState) (state | PRESOLVE_ROW_UPPER_REDUNDANT);
    }
    else
        state = (PresolveLinearRowState) (state | PRESOLVE_ROW_UPPER_REDUNDANT);

    if (isfinite(lower))
    {
        long double side = (long double) lower;
        if (activity->n_infinite_max == 0 &&
            activity->finite_max <
                side - scaled_tolerance(feasibility_tolerance, lower))
            return PRESOLVE_ROW_INFEASIBLE;
        if (activity->n_infinite_min == 0 &&
            activity->finite_min >=
                side - scaled_tolerance(redundancy_tolerance, lower))
            state = (PresolveLinearRowState) (state | PRESOLVE_ROW_LOWER_REDUNDANT);
    }
    else
        state = (PresolveLinearRowState) (state | PRESOLVE_ROW_LOWER_REDUNDANT);

    return state;
}

static inline double load_bound(const void *bounds, size_t stride, int column)
{
    double value;
    const unsigned char *address =
        (const unsigned char *) bounds + (size_t) column * stride;
    memcpy(&value, address, sizeof(value));
    return value;
}

static inline void load_domain(const PresolveLinearPropagationOps *ops, int column,
                               PresolveScalarDomain *domain)
{
    uint8_t flags = ops->column_flags ? ops->column_flags[column] : 0;
    domain->lower = load_bound(ops->lower_bounds, ops->bound_stride, column);
    domain->upper = load_bound(ops->upper_bounds, ops->bound_stride, column);
    domain->lower_is_infinite = !isfinite(domain->lower);
    domain->upper_is_infinite = !isfinite(domain->upper);
    domain->can_tighten = (flags & ops->inactive_mask) == 0 &&
                          (!ops->candidate_map || ops->candidate_map[column] >= 0);
}

static inline int add_activity_term(long double *sum, double coefficient,
                                    double bound, int is_lower, int outward)
{
    long double left = (long double) coefficient;
    long double right = (long double) bound;
    long double product = left * right;
    long double old_sum, result;

    if (!isfinite(product)) return 0;
    if (outward)
    {
        long double product_error = fmal(left, right, -product);
        if ((is_lower && product_error < 0.0L) ||
            (!is_lower && product_error > 0.0L))
            product = nextafterl(product, is_lower ? -INFINITY : INFINITY);
    }

    old_sum = *sum;
    result = old_sum + product;
    if (!isfinite(result)) return 0;
    if (outward)
    {
        long double recovered = result - old_sum;
        long double error = (old_sum - (result - recovered)) + (product - recovered);
        if ((is_lower && error < 0.0L) || (!is_lower && error > 0.0L))
            result = nextafterl(result, is_lower ? -INFINITY : INFINITY);
    }
    if (!isfinite(result)) return 0;
    *sum = result;
    return 1;
}

static inline int presolve_internal_compute_linear_activity(
    const double *values, const int *columns, int length,
    const PresolveLinearPropagationOps *ops, int outward,
    PresolveLinearActivity *activity)
{
    int position;
    memset(activity, 0, sizeof(*activity));
    for (position = 0; position < length; ++position)
    {
        PresolveScalarDomain domain;
        double coefficient = values[position];
        double min_bound, max_bound;
        if (coefficient == 0.0) continue;
        ++activity->n_nonzeros;
        load_domain(ops, columns[position], &domain);
        min_bound = coefficient > 0.0 ? domain.lower : domain.upper;
        max_bound = coefficient > 0.0 ? domain.upper : domain.lower;

        if (isfinite(min_bound))
        {
            if (!add_activity_term(&activity->finite_min, coefficient, min_bound, 1,
                                   outward))
                return 0;
        }
        else
            ++activity->n_infinite_min;

        if (isfinite(max_bound))
        {
            if (!add_activity_term(&activity->finite_max, coefficient, max_bound, 0,
                                   outward))
                return 0;
        }
        else
            ++activity->n_infinite_max;
    }
    return 1;
}

static inline double selected_bound(const PresolveScalarDomain *domain,
                                    double coefficient, int use_minimum)
{
    if (use_minimum) return coefficient > 0.0 ? domain->lower : domain->upper;
    return coefficient > 0.0 ? domain->upper : domain->lower;
}

static inline int selected_bound_is_infinite(const PresolveScalarDomain *domain,
                                             double coefficient, int use_minimum)
{
    if (use_minimum)
        return coefficient > 0.0 ? domain->lower_is_infinite
                                 : domain->upper_is_infinite;
    return coefficient > 0.0 ? domain->upper_is_infinite : domain->lower_is_infinite;
}

static inline int compute_residual(const PresolveLinearPropagationRow *row,
                                   const PresolveLinearPropagationOps *ops,
                                   int target_position, int use_minimum,
                                   const PresolveScalarDomain *target_domain,
                                   long double *residual)
{
    double target_coefficient = row->values[target_position];
    double target_bound =
        selected_bound(target_domain, target_coefficient, use_minimum);
    int target_is_infinite =
        selected_bound_is_infinite(target_domain, target_coefficient, use_minimum);
    size_t n_infinite = use_minimum ? row->n_infinite_min : row->n_infinite_max;
    int position;

    if (n_infinite < (size_t) target_is_infinite ||
        n_infinite - (size_t) target_is_infinite != 0)
        return 0;

    if (n_infinite == 0)
    {
        long double activity =
            use_minimum ? row->finite_min_activity : row->finite_max_activity;
        *residual =
            activity - (long double) target_coefficient * (long double) target_bound;
        return 1;
    }

    *residual = 0.0L;
    for (position = 0; position < row->length; ++position)
    {
        PresolveScalarDomain domain;
        double coefficient, bound;
        if (position == target_position) continue;
        coefficient = row->values[position];
        if (coefficient == 0.0) continue;
        load_domain(ops, row->columns[position], &domain);
        if (selected_bound_is_infinite(&domain, coefficient, use_minimum)) return 0;
        bound = selected_bound(&domain, coefficient, use_minimum);
        *residual += (long double) coefficient * (long double) bound;
    }
    return 1;
}

static inline long double implied_bound(double side, long double residual,
                                        double coefficient)
{
    return ((long double) side - residual) / (long double) coefficient;
}

static inline PresolveKernelUpdate
propagate_candidate(PresolveLinearPropagationRow *row,
                    const PresolveLinearPropagationOps *ops, int position,
                    const PresolveScalarDomain *domain, int use_minimum, double side,
                    int side_is_infinite, size_t *n_changed)
{
    size_t n_infinite = use_minimum ? row->n_infinite_min : row->n_infinite_max;
    size_t opposite_infinite =
        use_minimum ? row->n_infinite_max : row->n_infinite_min;
    PresolveKernelUpdate update;
    double coefficient = row->values[position];
    long double residual, candidate;
    int is_lower;

    if (!compute_residual(row, ops, position, use_minimum, domain, &residual))
        return PRESOLVE_KERNEL_UNCHANGED;

    if (side_is_infinite)
    {
        if (n_infinite != 1 || opposite_infinite != 0)
            return PRESOLVE_KERNEL_UNCHANGED;
        side = (double) (use_minimum ? row->finite_max_activity
                                     : row->finite_min_activity);
    }

    candidate = implied_bound(side, residual, coefficient);
    if (!isfinite(candidate)) return PRESOLVE_KERNEL_UNCHANGED;
    if (ops->maximum_inferred_bound_magnitude > 0.0 &&
        fabsl(candidate) >= (long double) ops->maximum_inferred_bound_magnitude)
        return PRESOLVE_KERNEL_UNCHANGED;
    is_lower = use_minimum ? coefficient < 0.0 : coefficient > 0.0;
    update = ops->tighten_bound(ops->context, row->columns[position], candidate,
                                is_lower);
    if (update == PRESOLVE_KERNEL_CHANGED)
    {
        ++*n_changed;
        if (ops->refresh_activity) ops->refresh_activity(ops->context, row);
    }
    return update;
}

static inline size_t
presolve_internal_propagate_linear_row(PresolveLinearPropagationRow *row,
                                       const PresolveLinearPropagationOps *ops,
                                       int *stopped)
{
    size_t n_changed = 0;
    int position;
    *stopped = 0;
    for (position = 0; position < row->length; ++position)
    {
        PresolveScalarDomain domain;
        PresolveKernelUpdate update;
        if (row->values[position] == 0.0) continue;
        load_domain(ops, row->columns[position], &domain);
        if (!domain.can_tighten) continue;
        update = propagate_candidate(row, ops, position, &domain, 1, row->upper,
                                     row->upper_is_infinite, &n_changed);
        if (update == PRESOLVE_KERNEL_STOP)
        {
            *stopped = 1;
            break;
        }
        load_domain(ops, row->columns[position], &domain);
        update = propagate_candidate(row, ops, position, &domain, 0, row->lower,
                                     row->lower_is_infinite, &n_changed);
        if (update == PRESOLVE_KERNEL_STOP)
        {
            *stopped = 1;
            break;
        }
    }
    return n_changed;
}

#endif
