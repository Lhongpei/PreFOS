/*
 * Copyright 2025-2026 Daniel Cederberg
 * Copyright 2026 Hongpei Li
 *
 * Modified for PreFOS in 2026.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef LINEAR_PROPAGATION_KERNEL_IMPL_H
#define LINEAR_PROPAGATION_KERNEL_IMPL_H

#include <float.h>
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
                side + (long double) redundancy_tolerance)
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
                side - (long double) redundancy_tolerance)
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

static inline void load_tightenable_domain(
    const PresolveLinearPropagationOps *ops, int column,
    PresolveScalarDomain *domain)
{
    domain->lower =
        load_bound(ops->lower_bounds, ops->bound_stride, column);
    domain->upper =
        load_bound(ops->upper_bounds, ops->bound_stride, column);
    domain->lower_is_infinite = !isfinite(domain->lower);
    domain->upper_is_infinite = !isfinite(domain->upper);
    domain->can_tighten = 1;
}

static inline int linear_column_can_tighten(
    const PresolveLinearPropagationOps *ops, int column)
{
    return (!ops->candidate_map || ops->candidate_map[column] >= 0) &&
           (!ops->column_flags ||
            (ops->column_flags[column] & ops->inactive_mask) == 0);
}

static inline int linear_term_is_active(
    const PresolveLinearPropagationOps *ops, int column)
{
    if (ops->row_excluded_columns)
    {
        int excluded = ops->row_excluded_columns[ops->row_index];
        if (excluded >= 0) return column != excluded;
        if (excluded == -1) return 1;
    }
    return !ops->row_exclusion_flags ||
           !ops->row_exclusion_flags[column] ||
           ops->row_exclusion_sources[column] != ops->row_index;
}

static inline int add_activity_term(double *sum, double coefficient,
                                    double bound, int is_lower, int outward)
{
    if (!outward)
    {
        double product = coefficient * bound;
        double result = *sum + product;
        if (!isfinite(product) || !isfinite(result)) return 0;
        *sum = result;
        return 1;
    }
    {
        long double left = (long double) coefficient;
        long double right = (long double) bound;
        long double product = left * right;
        long double old_sum, result;
        double converted;
        long double product_error;
        if (!isfinite(product)) return 0;
        product_error = fmal(left, right, -product);
        if ((is_lower && product_error < 0.0L) ||
            (!is_lower && product_error > 0.0L))
            product = nextafterl(
                product, is_lower ? -INFINITY : INFINITY);
        old_sum = (long double) *sum;
        result = old_sum + product;
        if (!isfinite(result)) return 0;
        {
        long double recovered = result - old_sum;
        long double error = (old_sum - (result - recovered)) + (product - recovered);
        if ((is_lower && error < 0.0L) || (!is_lower && error > 0.0L))
            result = nextafterl(result, is_lower ? -INFINITY : INFINITY);
        }
        converted = (double) result;
        if (!isfinite(converted)) return 0;
        if (is_lower && (long double) converted > result)
            converted = nextafter(converted, -INFINITY);
        if (!is_lower && (long double) converted < result)
            converted = nextafter(converted, INFINITY);
        *sum = converted;
        return 1;
    }
}

static inline long double presolve_internal_directed_add(
    long double left, long double right, int toward_lower)
{
    long double result = left + right;
    long double recovered, error;
    if (!isfinite(result)) return result;
    recovered = result - left;
    error = (left - (result - recovered)) + (right - recovered);
    if ((toward_lower && error < 0.0L) ||
        (!toward_lower && error > 0.0L))
        result = nextafterl(
            result, toward_lower ? -INFINITY : INFINITY);
    return result;
}

static inline long double presolve_internal_directed_product(
    double left_value, double right_value, int toward_lower)
{
    if (left_value == 1.0)
        return (long double) right_value;
    if (left_value == -1.0)
        return -(long double) right_value;
    double rounded = left_value * right_value;
    if (isfinite(rounded) &&
        (fabs(rounded) >= DBL_MIN ||
         left_value == 0.0 || right_value == 0.0))
    {
        double residual = fma(left_value, right_value, -rounded);
        return presolve_internal_directed_add(
            (long double) rounded, (long double) residual,
            toward_lower);
    }
    else
    {
        long double left = (long double) left_value;
        long double right = (long double) right_value;
        long double product = left * right;
        long double error;
        if (!isfinite(product)) return product;
        error = fmal(left, right, -product);
        if ((toward_lower && error < 0.0L) ||
            (!toward_lower && error > 0.0L))
            product = nextafterl(
                product, toward_lower ? -INFINITY : INFINITY);
        return product;
    }
}

static inline double presolve_internal_directed_double_cast(
    long double value, int toward_lower)
{
    double converted = (double) value;
    if (!isfinite(value)) return converted;
    if ((toward_lower && (long double) converted > value) ||
        (!toward_lower && (long double) converted < value))
        converted = nextafter(
            converted, toward_lower ? -INFINITY : INFINITY);
    return converted;
}

static inline int presolve_internal_compute_outward_linear_activity(
    const double *values, const int *columns, int length,
    const PresolveLinearPropagationOps *ops,
    PresolveLinearActivity *activity)
{
    long double minimum = 0.0L, maximum = 0.0L;
    int position;

    memset(activity, 0, sizeof(*activity));
    for (position = 0; position < length; ++position)
    {
        PresolveScalarDomain domain;
        double coefficient = values[position];
        double min_bound, max_bound;
        if (coefficient == 0.0 ||
            !linear_term_is_active(ops, columns[position]))
            continue;
        ++activity->n_nonzeros;
        load_domain(ops, columns[position], &domain);
        min_bound = coefficient > 0.0 ? domain.lower : domain.upper;
        max_bound = coefficient > 0.0 ? domain.upper : domain.lower;

        if (isfinite(min_bound))
        {
            minimum = presolve_internal_directed_add(
                minimum,
                presolve_internal_directed_product(
                    coefficient, min_bound, 1),
                1);
            if (!isfinite(minimum)) return 0;
        }
        else
            ++activity->n_infinite_min;

        if (isfinite(max_bound))
        {
            maximum = presolve_internal_directed_add(
                maximum,
                presolve_internal_directed_product(
                    coefficient, max_bound, 0),
                0);
            if (!isfinite(maximum)) return 0;
        }
        else
            ++activity->n_infinite_max;
    }

    activity->finite_min =
        presolve_internal_directed_double_cast(minimum, 1);
    activity->finite_max =
        presolve_internal_directed_double_cast(maximum, 0);
    return isfinite(activity->finite_min) &&
           isfinite(activity->finite_max);
}

static inline int presolve_internal_compute_linear_activity(
    const double *values, const int *columns, int length,
    const PresolveLinearPropagationOps *ops, int outward,
    PresolveLinearActivity *activity)
{
    int position;
    if (outward &&
        presolve_internal_compute_outward_linear_activity(
            values, columns, length, ops, activity))
        return 1;
    memset(activity, 0, sizeof(*activity));
    for (position = 0; position < length; ++position)
    {
        PresolveScalarDomain domain;
        double coefficient = values[position];
        double min_bound, max_bound;
        if (coefficient == 0.0 ||
            !linear_term_is_active(ops, columns[position]))
            continue;
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

static inline int compute_residual_by_scan(
    const PresolveLinearPropagationRow *row,
    const PresolveLinearPropagationOps *ops, int target_position,
    int use_minimum, long double *residual)
{
    long double sum = 0.0L;
    int position;

    if (row->length <= 2)
    {
        PresolveScalarDomain domain;
        double coefficient, bound;
        int other_position;
        if (row->length <= 1)
        {
            *residual = 0.0L;
            return 1;
        }
        other_position = target_position == 0 ? 1 : 0;
        coefficient = row->values[other_position];
        if (coefficient == 0.0 ||
            !linear_term_is_active(
                ops, row->columns[other_position]))
        {
            *residual = 0.0L;
            return 1;
        }
        load_domain(ops, row->columns[other_position], &domain);
        bound = selected_bound(
            &domain, coefficient, use_minimum);
        if (!isfinite(bound)) return 0;
        *residual = presolve_internal_directed_product(
            coefficient, bound, use_minimum);
        return isfinite(*residual);
    }

    for (position = 0; position < row->length; ++position)
    {
        PresolveScalarDomain domain;
        double coefficient, bound;
        long double product;
        if (position == target_position ||
            !linear_term_is_active(ops, row->columns[position]))
            continue;
        coefficient = row->values[position];
        if (coefficient == 0.0) continue;
        load_domain(ops, row->columns[position], &domain);
        bound = selected_bound(&domain, coefficient, use_minimum);
        if (!isfinite(bound)) return 0;
        product = presolve_internal_directed_product(
            coefficient, bound, use_minimum);
        sum = presolve_internal_directed_add(
            sum, product, use_minimum);
        if (!isfinite(sum)) return 0;
    }
    *residual = sum;
    return 1;
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

    if (n_infinite < (size_t) target_is_infinite ||
        n_infinite - (size_t) target_is_infinite != 0)
        return 0;

    if (n_infinite == 0 && row->length <= 2)
        return compute_residual_by_scan(
            row, ops, target_position, use_minimum, residual);

    if (n_infinite == 0)
    {
        long double activity = (long double)
            (use_minimum ? row->finite_min_activity
                         : row->finite_max_activity);
        long double target_product =
            (long double) target_coefficient *
            (long double) target_bound;
        long double cancellation_scale =
            fabsl(activity) + fabsl(target_product);
        if (ops->activity_is_outward)
        {
            target_product = presolve_internal_directed_product(
                target_coefficient, target_bound, !use_minimum);
            *residual = presolve_internal_directed_add(
                activity, -target_product, use_minimum);
        }
        else
            *residual = activity - target_product;
        /*
         * The cached activity is a double.  If the target contribution nearly
         * cancels it, subtracting the target can erase the complete residual
         * and produce an invalid implied bound.  Re-scan only these
         * ill-conditioned cases; ordinary rows keep the O(1) residual path.
         */
        if (fabsl(*residual) <=
            64.0L * (long double) DBL_EPSILON *
                fmaxl(1.0L, cancellation_scale))
        {
            long double outward_activity;
            if (*residual == 0.0L && target_product == 0.0L &&
                ops->activity_is_exact_zero &&
                ops->activity_is_exact_zero(
                    ops->context, use_minimum))
                return 1;
            /*
             * On a short row, summing the residual directly is cheaper than
             * rebuilding both outward activity sides and then subtracting the
             * target again.  The directed scan provides the same enclosure.
             */
            if (row->length <= 4)
            {
                if (ops->residual_rescans)
                    ++*ops->residual_rescans;
                if (ops->residual_rescan_terms)
                    *ops->residual_rescan_terms +=
                        (size_t) row->length;
                return compute_residual_by_scan(
                    row, ops, target_position, use_minimum,
                    residual);
            }
            if (ops->outward_activity &&
                ops->outward_activity(
                    ops->context, use_minimum,
                    &outward_activity))
            {
                target_product =
                    presolve_internal_directed_product(
                        target_coefficient, target_bound,
                        !use_minimum);
                *residual = presolve_internal_directed_add(
                    outward_activity, -target_product,
                    use_minimum);
                return isfinite(*residual);
            }
            if (ops->residual_rescans)
                ++*ops->residual_rescans;
            if (ops->residual_rescan_terms)
                *ops->residual_rescan_terms += (size_t) row->length;
            return compute_residual_by_scan(
                row, ops, target_position, use_minimum, residual);
        }
        return 1;
    }

    /*
     * The target supplies the row's only infinite contribution, so the
     * cached finite activity is already the residual with that target
     * removed. Re-scanning the row here is both redundant and particularly
     * expensive for the common free-to-finite propagation case.
     */
    *residual = (long double)
        (use_minimum ? row->finite_min_activity
                     : row->finite_max_activity);
    return 1;
}

static inline long double implied_bound(double side, long double residual,
                                        double coefficient)
{
    if (coefficient == 1.0)
        return (long double) side - residual;
    if (coefficient == -1.0)
        return residual - (long double) side;
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

    if (!side_is_infinite && n_infinite == 0 &&
        isfinite(domain->lower) && isfinite(domain->upper))
    {
        long double slack =
            use_minimum
                ? (long double) side -
                      (long double) row->finite_min_activity
                : (long double) row->finite_max_activity -
                      (long double) side;
        long double range =
            fabsl((long double) coefficient) *
            ((long double) domain->upper -
             (long double) domain->lower);
        if (slack >= range)
            return PRESOLVE_KERNEL_UNCHANGED;
    }
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
    update = ops->tighten_bound(
        ops->context, row->columns[position], coefficient,
        candidate, is_lower);
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
    int propagate_minimum =
        (!row->upper_is_infinite && row->n_infinite_min <= 1) ||
        (row->upper_is_infinite && row->n_infinite_min == 1 &&
         row->n_infinite_max == 0);
    int propagate_maximum =
        (!row->lower_is_infinite && row->n_infinite_max <= 1) ||
        (row->lower_is_infinite && row->n_infinite_max == 1 &&
         row->n_infinite_min == 0);
    *stopped = 0;
    if (!propagate_minimum && !propagate_maximum) return 0;
    for (position = 0; position < row->length; ++position)
    {
        PresolveScalarDomain domain;
        PresolveKernelUpdate update;
        if (row->values[position] == 0.0 ||
            !linear_term_is_active(ops, row->columns[position]) ||
            !linear_column_can_tighten(
                ops, row->columns[position]))
            continue;
        load_tightenable_domain(
            ops, row->columns[position], &domain);
        if (propagate_minimum)
        {
            update = propagate_candidate(
                row, ops, position, &domain, 1, row->upper,
                row->upper_is_infinite, &n_changed);
            if (update == PRESOLVE_KERNEL_STOP)
            {
                *stopped = 1;
                break;
            }
        }
        if (propagate_maximum)
        {
            if (propagate_minimum &&
                update == PRESOLVE_KERNEL_CHANGED)
                load_domain(ops, row->columns[position], &domain);
            update = propagate_candidate(
                row, ops, position, &domain, 0, row->lower,
                row->lower_is_infinite, &n_changed);
            if (update == PRESOLVE_KERNEL_STOP)
            {
                *stopped = 1;
                break;
            }
        }
    }
    return n_changed;
}

static inline size_t
presolve_internal_propagate_single_infinite_candidates(
    PresolveLinearPropagationRow *row,
    const PresolveLinearPropagationOps *ops,
    int minimum_position, int maximum_position, int *stopped)
{
    size_t n_changed = 0;
    int propagate_minimum =
        (!row->upper_is_infinite && row->n_infinite_min == 1) ||
        (row->upper_is_infinite && row->n_infinite_min == 1 &&
         row->n_infinite_max == 0);
    int propagate_maximum =
        (!row->lower_is_infinite && row->n_infinite_max == 1) ||
        (row->lower_is_infinite && row->n_infinite_max == 1 &&
         row->n_infinite_min == 0);
    PresolveKernelUpdate update = PRESOLVE_KERNEL_UNCHANGED;

    *stopped = 0;
    if (propagate_minimum && minimum_position >= 0 &&
        minimum_position < row->length &&
        row->values[minimum_position] != 0.0 &&
        linear_term_is_active(
            ops, row->columns[minimum_position]) &&
        linear_column_can_tighten(
            ops, row->columns[minimum_position]))
    {
        PresolveScalarDomain domain;
        load_tightenable_domain(
            ops, row->columns[minimum_position], &domain);
        update = propagate_candidate(
            row, ops, minimum_position, &domain, 1, row->upper,
            row->upper_is_infinite, &n_changed);
        if (update == PRESOLVE_KERNEL_STOP)
        {
            *stopped = 1;
            return n_changed;
        }
    }
    if (propagate_maximum && maximum_position >= 0 &&
        maximum_position < row->length &&
        row->values[maximum_position] != 0.0 &&
        linear_term_is_active(
            ops, row->columns[maximum_position]) &&
        linear_column_can_tighten(
            ops, row->columns[maximum_position]))
    {
        PresolveScalarDomain domain;
        load_tightenable_domain(
            ops, row->columns[maximum_position], &domain);
        update = propagate_candidate(
            row, ops, maximum_position, &domain, 0, row->lower,
            row->lower_is_infinite, &n_changed);
        if (update == PRESOLVE_KERNEL_STOP)
            *stopped = 1;
    }
    return n_changed;
}

#endif
