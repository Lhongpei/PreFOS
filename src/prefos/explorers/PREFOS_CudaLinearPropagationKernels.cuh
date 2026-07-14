/*
 * Copyright 2026 Hongpei Li
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef PREFOS_CUDA_LINEAR_PROPAGATION_KERNELS_CUH
#define PREFOS_CUDA_LINEAR_PROPAGATION_KERNELS_CUH

__global__ void initialize_round_kernel(size_t columns, double *lower_candidates,
                                        double *upper_candidates,
                                        int *lower_source_rows,
                                        int *upper_source_rows,
                                        int *suspected_infeasible_row,
                                        int *numerical_error)
{
    size_t column = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (column < columns)
    {
        lower_candidates[column] = -INFINITY;
        upper_candidates[column] = INFINITY;
        lower_source_rows[column] = -1;
        upper_source_rows[column] = -1;
    }
    if (column == 0)
    {
        *suspected_infeasible_row = INT_MAX;
        *numerical_error = 0;
    }
}

__device__ double selected_bound(double coefficient, double lower, double upper,
                                 bool use_minimum)
{
    if (use_minimum) return coefficient > 0.0 ? lower : upper;
    return coefficient > 0.0 ? upper : lower;
}

__device__ void atomic_reduce_candidate(double candidate, double *target,
                                        bool is_lower)
{
    unsigned long long *address = reinterpret_cast<unsigned long long *>(target);
    unsigned long long old = *address;
    while (true)
    {
        double current = __longlong_as_double(static_cast<long long>(old));
        unsigned long long assumed;
        if ((is_lower && candidate <= current) ||
            (!is_lower && candidate >= current))
            return;
        assumed = old;
        old = atomicCAS(address, assumed, __double_as_longlong(candidate));
        if (old == assumed) return;
    }
}

__device__ void propagate_side(int row, double coefficient, double domain_lower,
                               double domain_upper, bool use_minimum, double side,
                               double finite_activity, double opposite_activity,
                               int n_infinite, int opposite_infinite,
                               double activity_error, double opposite_error,
                               double maximum_magnitude, int column,
                               double *lower_candidates, double *upper_candidates,
                               int *lower_source_rows, int *upper_source_rows,
                               bool collect_sources)
{
    double target_bound =
        selected_bound(coefficient, domain_lower, domain_upper, use_minimum);
    bool target_is_infinite = !isfinite(target_bound);
    double residual;
    double residual_error = activity_error;
    double candidate;
    double arithmetic_error;
    bool is_lower;

    if (n_infinite - static_cast<int>(target_is_infinite) != 0) return;
    if (n_infinite == 0)
    {
        double product = coefficient * target_bound;
        residual = finite_activity - product;
        residual_error += 8.0 * DBL_EPSILON *
                          (fabs(finite_activity) + fabs(product) + fabs(residual));
    }
    else
        residual = finite_activity;

    if (!isfinite(side))
    {
        if (n_infinite != 1 || opposite_infinite != 0) return;
        side = opposite_activity;
        residual_error += opposite_error;
    }
    candidate = (side - residual) / coefficient;
    if (!isfinite(candidate) ||
        (maximum_magnitude > 0.0 && fabs(candidate) >= maximum_magnitude))
        return;

    arithmetic_error =
        residual_error / fabs(coefficient) +
        8.0 * DBL_EPSILON *
            (fabs(side) + fabs(residual) + fabs(candidate * coefficient)) /
            fabs(coefficient);
    is_lower = use_minimum ? coefficient < 0.0 : coefficient > 0.0;
    candidate += is_lower ? -arithmetic_error : arithmetic_error;
    if (!isfinite(candidate)) return;
    {
        double *target =
            is_lower ? lower_candidates + column : upper_candidates + column;
        int *source =
            is_lower ? lower_source_rows + column : upper_source_rows + column;
        if (collect_sources)
        {
            if (candidate == *target) (void) atomicCAS(source, -1, row);
        }
        else
            atomic_reduce_candidate(candidate, target, is_lower);
    }
}

__global__ void
propagation_kernel(size_t rows, const int *row_pointers, const int *column_indices,
                   const double *values, const double *constraint_lower,
                   const double *constraint_upper, const int *candidate_map,
                   const unsigned char *remove_rows, const double *lower_bounds,
                   const double *upper_bounds, double feasibility_tolerance,
                   double maximum_magnitude, double *lower_candidates,
                   double *upper_candidates, int *lower_source_rows,
                   int *upper_source_rows, int *suspected_infeasible_row,
                   int *numerical_error, bool collect_sources)
{
    size_t row_index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int start, end, position;
    double finite_min = 0.0, finite_max = 0.0;
    double absolute_min = 0.0, absolute_max = 0.0;
    double min_error, max_error;
    int n_infinite_min = 0, n_infinite_max = 0;
    int n_nonzeros = 0;
    double row_lower, row_upper;

    if (row_index >= rows || remove_rows[row_index]) return;
    row_lower = constraint_lower[row_index];
    row_upper = constraint_upper[row_index];
    if (!isfinite(row_lower) && !isfinite(row_upper)) return;
    start = row_pointers[row_index];
    end = row_pointers[row_index + 1];
    if (end - start >= kLongRowThreshold) return;

    for (position = start; position < end; ++position)
    {
        int column = column_indices[position];
        double coefficient = values[position];
        double min_bound, max_bound, product;
        if (coefficient == 0.0) continue;
        ++n_nonzeros;
        min_bound = selected_bound(coefficient, lower_bounds[column],
                                   upper_bounds[column], true);
        max_bound = selected_bound(coefficient, lower_bounds[column],
                                   upper_bounds[column], false);
        if (isfinite(min_bound))
        {
            product = coefficient * min_bound;
            finite_min += product;
            absolute_min += fabs(product);
        }
        else
            ++n_infinite_min;
        if (isfinite(max_bound))
        {
            product = coefficient * max_bound;
            finite_max += product;
            absolute_max += fabs(product);
        }
        else
            ++n_infinite_max;
    }
    if (!isfinite(finite_min) || !isfinite(finite_max) || !isfinite(absolute_min) ||
        !isfinite(absolute_max))
    {
        atomicExch(numerical_error, 1);
        return;
    }
    {
        double error_factor =
            (4.0 * static_cast<double>(n_nonzeros) + 16.0) * DBL_EPSILON;
        if (error_factor >= 0.5)
        {
            atomicExch(numerical_error, 1);
            return;
        }
        error_factor /= 1.0 - error_factor;
        min_error =
            error_factor * absolute_min + static_cast<double>(n_nonzeros) * DBL_MIN;
        max_error =
            error_factor * absolute_max + static_cast<double>(n_nonzeros) * DBL_MIN;
    }

    if (!collect_sources && isfinite(row_upper) && n_infinite_min == 0)
    {
        double tolerance = feasibility_tolerance * fmax(1.0, fabs(row_upper));
        if (finite_min - min_error > row_upper + tolerance)
            atomicMin(suspected_infeasible_row, static_cast<int>(row_index));
    }
    if (!collect_sources && isfinite(row_lower) && n_infinite_max == 0)
    {
        double tolerance = feasibility_tolerance * fmax(1.0, fabs(row_lower));
        if (finite_max + max_error < row_lower - tolerance)
            atomicMin(suspected_infeasible_row, static_cast<int>(row_index));
    }

    for (position = start; position < end; ++position)
    {
        int column = column_indices[position];
        double coefficient = values[position];
        if (coefficient == 0.0 || candidate_map[column] < 0) continue;
        propagate_side(static_cast<int>(row_index), coefficient,
                       lower_bounds[column], upper_bounds[column], true, row_upper,
                       finite_min, finite_max, n_infinite_min, n_infinite_max,
                       min_error, max_error, maximum_magnitude, column,
                       lower_candidates, upper_candidates, lower_source_rows,
                       upper_source_rows, collect_sources);
        propagate_side(static_cast<int>(row_index), coefficient,
                       lower_bounds[column], upper_bounds[column], false, row_lower,
                       finite_max, finite_min, n_infinite_max, n_infinite_min,
                       max_error, min_error, maximum_magnitude, column,
                       lower_candidates, upper_candidates, lower_source_rows,
                       upper_source_rows, collect_sources);
    }
}

struct BlockActivity
{
    double finite_min;
    double finite_max;
    double absolute_min;
    double absolute_max;
    int n_infinite_min;
    int n_infinite_max;
    int n_nonzeros;
    int numerical_error;
};

struct AddBlockActivity
{
    __device__ BlockActivity operator()(const BlockActivity &left,
                                        const BlockActivity &right) const
    {
        return BlockActivity{left.finite_min + right.finite_min,
                             left.finite_max + right.finite_max,
                             left.absolute_min + right.absolute_min,
                             left.absolute_max + right.absolute_max,
                             left.n_infinite_min + right.n_infinite_min,
                             left.n_infinite_max + right.n_infinite_max,
                             left.n_nonzeros + right.n_nonzeros,
                             left.numerical_error | right.numerical_error};
    }
};

__global__ void long_row_propagation_kernel(
    size_t n_long_rows, const int *long_rows, const int *row_pointers,
    const int *column_indices, const double *values, const double *constraint_lower,
    const double *constraint_upper, const int *candidate_map,
    const unsigned char *remove_rows, const double *lower_bounds,
    const double *upper_bounds, double feasibility_tolerance,
    double maximum_magnitude, double *lower_candidates, double *upper_candidates,
    int *lower_source_rows, int *upper_source_rows, int *suspected_infeasible_row,
    int *numerical_error, bool collect_sources)
{
    using BlockReduce = cub::BlockReduce<BlockActivity, kThreads>;
    __shared__ typename BlockReduce::TempStorage reduction_storage;
    __shared__ BlockActivity activity;
    __shared__ double min_error;
    __shared__ double max_error;
    __shared__ int activity_valid;
    size_t long_position = blockIdx.x;
    int row, start, end;
    double row_lower, row_upper;
    BlockActivity local{};

    if (long_position >= n_long_rows) return;
    row = long_rows[long_position];
    if (remove_rows[row]) return;
    row_lower = constraint_lower[row];
    row_upper = constraint_upper[row];
    if (!isfinite(row_lower) && !isfinite(row_upper)) return;
    start = row_pointers[row];
    end = row_pointers[row + 1];

    for (int position = start + threadIdx.x; position < end; position += blockDim.x)
    {
        int column = column_indices[position];
        double coefficient = values[position];
        double min_bound, max_bound, product;
        if (coefficient == 0.0) continue;
        ++local.n_nonzeros;
        min_bound = selected_bound(coefficient, lower_bounds[column],
                                   upper_bounds[column], true);
        max_bound = selected_bound(coefficient, lower_bounds[column],
                                   upper_bounds[column], false);
        if (isfinite(min_bound))
        {
            product = coefficient * min_bound;
            local.finite_min += product;
            local.absolute_min += fabs(product);
            if (!isfinite(local.finite_min) || !isfinite(local.absolute_min))
                local.numerical_error = 1;
        }
        else
            ++local.n_infinite_min;
        if (isfinite(max_bound))
        {
            product = coefficient * max_bound;
            local.finite_max += product;
            local.absolute_max += fabs(product);
            if (!isfinite(local.finite_max) || !isfinite(local.absolute_max))
                local.numerical_error = 1;
        }
        else
            ++local.n_infinite_max;
    }

    BlockActivity aggregate =
        BlockReduce(reduction_storage).Reduce(local, AddBlockActivity{});
    if (threadIdx.x == 0)
    {
        double error_factor =
            (4.0 * static_cast<double>(aggregate.n_nonzeros) + 16.0) * DBL_EPSILON;
        activity = aggregate;
        activity_valid =
            !aggregate.numerical_error && isfinite(aggregate.finite_min) &&
            isfinite(aggregate.finite_max) && isfinite(aggregate.absolute_min) &&
            isfinite(aggregate.absolute_max) && error_factor < 0.5;
        if (!activity_valid)
            atomicExch(numerical_error, 1);
        else
        {
            error_factor /= 1.0 - error_factor;
            min_error = error_factor * aggregate.absolute_min +
                        static_cast<double>(aggregate.n_nonzeros) * DBL_MIN;
            max_error = error_factor * aggregate.absolute_max +
                        static_cast<double>(aggregate.n_nonzeros) * DBL_MIN;
            if (!collect_sources && isfinite(row_upper) &&
                aggregate.n_infinite_min == 0)
            {
                double tolerance =
                    feasibility_tolerance * fmax(1.0, fabs(row_upper));
                if (aggregate.finite_min - min_error > row_upper + tolerance)
                    atomicMin(suspected_infeasible_row, row);
            }
            if (!collect_sources && isfinite(row_lower) &&
                aggregate.n_infinite_max == 0)
            {
                double tolerance =
                    feasibility_tolerance * fmax(1.0, fabs(row_lower));
                if (aggregate.finite_max + max_error < row_lower - tolerance)
                    atomicMin(suspected_infeasible_row, row);
            }
        }
    }
    __syncthreads();
    if (!activity_valid) return;

    for (int position = start + threadIdx.x; position < end; position += blockDim.x)
    {
        int column = column_indices[position];
        double coefficient = values[position];
        if (coefficient == 0.0 || candidate_map[column] < 0) continue;
        propagate_side(row, coefficient, lower_bounds[column], upper_bounds[column],
                       true, row_upper, activity.finite_min, activity.finite_max,
                       activity.n_infinite_min, activity.n_infinite_max, min_error,
                       max_error, maximum_magnitude, column, lower_candidates,
                       upper_candidates, lower_source_rows, upper_source_rows,
                       collect_sources);
        propagate_side(row, coefficient, lower_bounds[column], upper_bounds[column],
                       false, row_lower, activity.finite_max, activity.finite_min,
                       activity.n_infinite_max, activity.n_infinite_min, max_error,
                       min_error, maximum_magnitude, column, lower_candidates,
                       upper_candidates, lower_source_rows, upper_source_rows,
                       collect_sources);
    }
}

#endif
