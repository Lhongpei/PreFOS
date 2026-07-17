/*
 * Copyright 2026 Hongpei Li
 * SPDX-License-Identifier: Apache-2.0
 */

#include "PREFOS_CudaLinearPropagation.h"
#include "PREFOS_CudaWorkspaceInternal.cuh"

#include <cuda_runtime.h>

#include <cfloat>
#include <chrono>
#include <cmath>

namespace
{

constexpr int kThreads = 256;
constexpr int kConeNonnegative = 0;
constexpr int kConeSoc = 1;
constexpr int kConeRsoc = 2;
constexpr int kConePsd = 3;
constexpr int kConeExponential = 4;
constexpr int kConePower = 5;

template <typename T>
cudaError_t allocate_device(T **pointer, size_t count, cudaStream_t stream)
{
    *pointer = nullptr;
    return count == 0 ? cudaSuccess
                      : cudaMallocAsync(reinterpret_cast<void **>(pointer),
                                        count * sizeof(T), stream);
}

template <typename T>
cudaError_t copy_to_device(T *target, const T *source, size_t count,
                           cudaStream_t stream)
{
    return count == 0 ? cudaSuccess
                      : cudaMemcpyAsync(target, source, count * sizeof(T),
                                        cudaMemcpyHostToDevice, stream);
}

template <typename T>
cudaError_t copy_to_host(T *target, const T *source, size_t count,
                         cudaStream_t stream)
{
    return count == 0 ? cudaSuccess
                      : cudaMemcpyAsync(target, source, count * sizeof(T),
                                        cudaMemcpyDeviceToHost, stream);
}

PreFOSCudaPropagationStatus status_from_cuda(cudaError_t status)
{
    if (status == cudaSuccess) return PREFOS_CUDA_PROPAGATION_OK;
    if (status == cudaErrorMemoryAllocation)
        return PREFOS_CUDA_PROPAGATION_OUT_OF_MEMORY;
    return PREFOS_CUDA_PROPAGATION_ERROR;
}

__device__ double minimum_absolute_value(double lower, double upper)
{
    if (lower > 0.0) return lower;
    if (upper < 0.0) return -upper;
    return 0.0;
}

__device__ double relax_lower(double value, double work)
{
    if (!isfinite(value)) return value;
    double error = 65536.0 * DBL_EPSILON *
                   fmax(1.0, fmax(fabs(value), fabs(work)));
    return value - error;
}

__device__ double relax_upper(double value, double work)
{
    if (!isfinite(value)) return value;
    double error = 65536.0 * DBL_EPSILON *
                   fmax(1.0, fmax(fabs(value), fabs(work)));
    return value + error;
}

__device__ void tighten_lower(double *candidates, int column, double value)
{
    if (value > candidates[column]) candidates[column] = value;
}

__device__ void tighten_upper(double *candidates, int column, double value)
{
    if (value < candidates[column]) candidates[column] = value;
}

__device__ double clamp_value(double value, double lower, double upper)
{
    return fmax(lower, fmin(upper, value));
}

__global__ void initialize_cone_candidates_kernel(
    size_t columns, size_t cones, double *lower_candidates,
    double *upper_candidates, unsigned char *cone_flags)
{
    size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < columns)
    {
        lower_candidates[index] = -INFINITY;
        upper_candidates[index] = INFINITY;
    }
    if (index < cones) cone_flags[index] = 0;
}

__global__ void cone_envelope_kernel(
    size_t cones, const int *cone_types, const int *cone_starts,
    const int *cone_indices, const int *cone_matrix_orders,
    const double *cone_power_alphas,
    const double *lower_bounds, const double *upper_bounds,
    double feasibility_tolerance, double *lower_candidates,
    double *upper_candidates, unsigned char *cone_flags)
{
    size_t cone = static_cast<size_t>(blockIdx.x);
    if (cone >= cones || threadIdx.x != 0) return;
    int type = cone_types[cone];
    if (type < 0) return;
    int start = cone_starts[cone];
    int end = cone_starts[cone + 1];
    int dimension = end - start;
    unsigned char flags = PREFOS_CUDA_CONE_PROCESSED;

    if (type == kConeNonnegative)
    {
        for (int local = 0; local < dimension; ++local)
            tighten_lower(lower_candidates, cone_indices[start + local], 0.0);
    }
    else if (type == kConeSoc && dimension >= 2)
    {
        int axis = cone_indices[start];
        double minimum_norm_squared = 0.0;
        for (int local = 1; local < dimension; ++local)
        {
            int column = cone_indices[start + local];
            double minimum =
                minimum_absolute_value(lower_bounds[column],
                                       upper_bounds[column]);
            minimum_norm_squared += minimum * minimum;
        }
        if (!isfinite(minimum_norm_squared) || dimension > 4096)
        {
            flags |= PREFOS_CUDA_CONE_NEEDS_CPU;
            cone_flags[cone] = flags;
            return;
        }
        double minimum_norm = sqrt(minimum_norm_squared);
        tighten_lower(lower_candidates, axis, 0.0);
        tighten_lower(lower_candidates, axis,
                      relax_lower(minimum_norm, minimum_norm_squared));
        double axis_upper = upper_bounds[axis];
        if (isfinite(axis_upper))
        {
            double tolerance =
                feasibility_tolerance * fmax(1.0, fabs(axis_upper));
            double numerical = 1024.0 * DBL_EPSILON *
                               fmax(1.0, fmax(fabs(axis_upper), minimum_norm));
            if (minimum_norm > axis_upper + tolerance + numerical)
                flags |= PREFOS_CUDA_CONE_INFEASIBLE;
            double capacity = axis_upper * axis_upper;
            for (int local = 1; local < dimension; ++local)
            {
                int column = cone_indices[start + local];
                double minimum =
                    minimum_absolute_value(lower_bounds[column],
                                           upper_bounds[column]);
                double remaining = fmax(
                    0.0, capacity -
                             (minimum_norm_squared - minimum * minimum));
                double limit = sqrt(remaining);
                tighten_lower(lower_candidates, column,
                              relax_lower(-limit, capacity));
                tighten_upper(upper_candidates, column,
                              relax_upper(limit, capacity));
            }
        }
    }
    else if (type == kConeRsoc && dimension >= 3)
    {
        int u_column = cone_indices[start];
        int v_column = cone_indices[start + 1];
        double minimum_norm_squared = 0.0;
        for (int local = 2; local < dimension; ++local)
        {
            int column = cone_indices[start + local];
            double minimum =
                minimum_absolute_value(lower_bounds[column],
                                       upper_bounds[column]);
            minimum_norm_squared += minimum * minimum;
        }
        if (!isfinite(minimum_norm_squared) || dimension > 4096)
        {
            flags |= PREFOS_CUDA_CONE_NEEDS_CPU;
            cone_flags[cone] = flags;
            return;
        }
        tighten_lower(lower_candidates, u_column, 0.0);
        tighten_lower(lower_candidates, v_column, 0.0);
        double u_upper = upper_bounds[u_column];
        double v_upper = upper_bounds[v_column];
        if (isfinite(v_upper) && v_upper > 0.0)
            tighten_lower(
                lower_candidates, u_column,
                relax_lower(minimum_norm_squared / (2.0 * v_upper),
                            minimum_norm_squared));
        if (isfinite(u_upper) && u_upper > 0.0)
            tighten_lower(
                lower_candidates, v_column,
                relax_lower(minimum_norm_squared / (2.0 * u_upper),
                            minimum_norm_squared));
        if ((u_upper == 0.0 || v_upper == 0.0) &&
            minimum_norm_squared > 0.0)
            flags |= PREFOS_CUDA_CONE_INFEASIBLE;
        if (u_upper == 0.0 || v_upper == 0.0)
        {
            for (int local = 2; local < dimension; ++local)
            {
                int column = cone_indices[start + local];
                tighten_lower(lower_candidates, column, 0.0);
                tighten_upper(upper_candidates, column, 0.0);
            }
        }
        else if (isfinite(u_upper) && isfinite(v_upper))
        {
            double capacity = 2.0 * u_upper * v_upper;
            double tolerance =
                feasibility_tolerance * fmax(1.0, fabs(capacity));
            double numerical = 1024.0 * DBL_EPSILON *
                               fmax(1.0, fmax(fabs(capacity),
                                              minimum_norm_squared));
            if (minimum_norm_squared > capacity + tolerance + numerical)
                flags |= PREFOS_CUDA_CONE_INFEASIBLE;
            for (int local = 2; local < dimension; ++local)
            {
                int column = cone_indices[start + local];
                double minimum =
                    minimum_absolute_value(lower_bounds[column],
                                           upper_bounds[column]);
                double remaining = fmax(
                    0.0, capacity -
                             (minimum_norm_squared - minimum * minimum));
                double limit = sqrt(remaining);
                tighten_lower(lower_candidates, column,
                              relax_lower(-limit, capacity));
                tighten_upper(upper_candidates, column,
                              relax_upper(limit, capacity));
            }
        }
    }
    else if (type == kConePsd)
    {
        int order = cone_matrix_orders[cone];
        for (int row = 0; row < order; ++row)
        {
            int diagonal = row * (row + 1) / 2 + row;
            if (diagonal < dimension)
                tighten_lower(lower_candidates,
                              cone_indices[start + diagonal], 0.0);
        }
        const double sqrt_two = sqrt(2.0);
        for (int row = 1; row < order; ++row)
        {
            int diagonal_row = row * (row + 1) / 2 + row;
            int diagonal_row_column =
                cone_indices[start + diagonal_row];
            for (int column_index = 0; column_index < row; ++column_index)
            {
                int diagonal_column =
                    column_index * (column_index + 1) / 2 + column_index;
                int off_diagonal = row * (row + 1) / 2 + column_index;
                int diagonal_column_variable =
                    cone_indices[start + diagonal_column];
                int off_diagonal_variable =
                    cone_indices[start + off_diagonal];
                double upper_row = upper_bounds[diagonal_row_column];
                double upper_column =
                    upper_bounds[diagonal_column_variable];
                double minimum_svec = minimum_absolute_value(
                    lower_bounds[off_diagonal_variable],
                    upper_bounds[off_diagonal_variable]);
                double minimum_entry = minimum_svec / sqrt_two;
                if ((upper_row == 0.0 || upper_column == 0.0) &&
                    minimum_entry > 0.0)
                    flags |= PREFOS_CUDA_CONE_INFEASIBLE;
                if (upper_row == 0.0 || upper_column == 0.0)
                {
                    tighten_lower(lower_candidates,
                                  off_diagonal_variable, 0.0);
                    tighten_upper(upper_candidates,
                                  off_diagonal_variable, 0.0);
                }
                else if (isfinite(upper_row) && isfinite(upper_column))
                {
                    double capacity = upper_row * upper_column;
                    double minimum_squared = minimum_entry * minimum_entry;
                    double tolerance = feasibility_tolerance *
                                       fmax(1.0, fabs(capacity));
                    double numerical =
                        1024.0 * DBL_EPSILON *
                        fmax(1.0, fmax(fabs(capacity), minimum_squared));
                    if (minimum_squared >
                        capacity + tolerance + numerical)
                        flags |= PREFOS_CUDA_CONE_INFEASIBLE;
                    double limit =
                        sqrt_two * sqrt(fmax(0.0, capacity));
                    tighten_lower(lower_candidates,
                                  off_diagonal_variable,
                                  relax_lower(-limit, capacity));
                    tighten_upper(upper_candidates,
                                  off_diagonal_variable,
                                  relax_upper(limit, capacity));
                }
                if (isfinite(upper_column) && upper_column > 0.0)
                    tighten_lower(
                        lower_candidates, diagonal_row_column,
                        relax_lower(minimum_entry * minimum_entry /
                                        upper_column,
                                    minimum_entry * minimum_entry));
                if (isfinite(upper_row) && upper_row > 0.0)
                    tighten_lower(
                        lower_candidates, diagonal_column_variable,
                        relax_lower(minimum_entry * minimum_entry / upper_row,
                                    minimum_entry * minimum_entry));
            }
        }
        flags |= PREFOS_CUDA_CONE_NEEDS_CPU;
    }
    else if (type == kConeExponential && dimension == 3)
    {
        int x_column = cone_indices[start];
        int y_column = cone_indices[start + 1];
        int z_column = cone_indices[start + 2];
        tighten_lower(lower_candidates, y_column, 0.0);
        tighten_lower(lower_candidates, z_column, 0.0);
        double x = lower_bounds[x_column];
        double lower_y = fmax(0.0, lower_bounds[y_column]);
        double upper_y = upper_bounds[y_column];
        double minimum_z = 0.0;
        if (x == -INFINITY || (lower_y == 0.0 && x <= 0.0))
            minimum_z = 0.0;
        else if (x == INFINITY || upper_y <= 0.0)
            flags |= PREFOS_CUDA_CONE_NEEDS_CPU;
        else
        {
            double y = x > 0.0 ? clamp_value(x, lower_y, upper_y)
                               : lower_y;
            if (y > 0.0)
            {
                double log_value = log(y) + x / y;
                minimum_z =
                    log_value > log(DBL_MAX) ? INFINITY : exp(log_value);
            }
            if (isfinite(minimum_z))
                tighten_lower(lower_candidates, z_column,
                              relax_lower(fmax(0.0, minimum_z), minimum_z));
        }
        double z = upper_bounds[z_column];
        double maximum_x = INFINITY;
        if (z < 0.0 || (z == 0.0 && lower_y > 0.0))
            flags |= PREFOS_CUDA_CONE_INFEASIBLE;
        else if (z == 0.0 || upper_y == 0.0)
            maximum_x = 0.0;
        else if (isfinite(z))
        {
            double y =
                clamp_value(z / exp(1.0), lower_y, upper_y);
            double candidate = y > 0.0 ? y * log(z / y) : 0.0;
            maximum_x = lower_y == 0.0 ? fmax(0.0, candidate)
                                       : candidate;
        }
        if (isfinite(maximum_x))
            tighten_upper(upper_candidates, x_column,
                          relax_upper(maximum_x, maximum_x));
        flags |= PREFOS_CUDA_CONE_NEEDS_CPU;
    }
    else if (type == kConePower && dimension == 3)
    {
        int x_column = cone_indices[start];
        int y_column = cone_indices[start + 1];
        int z_column = cone_indices[start + 2];
        double alpha = cone_power_alphas[cone];
        tighten_lower(lower_candidates, x_column, 0.0);
        tighten_lower(lower_candidates, y_column, 0.0);
        double upper_x = upper_bounds[x_column];
        double upper_y = upper_bounds[y_column];
        double minimum_abs_z = minimum_absolute_value(
            lower_bounds[z_column], upper_bounds[z_column]);
        double capacity = INFINITY;
        if (upper_x == 0.0 || upper_y == 0.0)
            capacity = 0.0;
        else if (isfinite(upper_x) && isfinite(upper_y) &&
                 upper_x > 0.0 && upper_y > 0.0)
        {
            double log_capacity =
                alpha * log(upper_x) + (1.0 - alpha) * log(upper_y);
            capacity = log_capacity > log(DBL_MAX)
                           ? INFINITY
                           : exp(log_capacity);
        }
        if (isfinite(capacity))
        {
            double safe_capacity = relax_upper(capacity, capacity);
            tighten_lower(lower_candidates, z_column, -safe_capacity);
            tighten_upper(upper_candidates, z_column, safe_capacity);
            double tolerance =
                feasibility_tolerance * fmax(1.0, fabs(capacity));
            if (minimum_abs_z > safe_capacity + tolerance)
                flags |= PREFOS_CUDA_CONE_INFEASIBLE;
        }
        if (minimum_abs_z > 0.0)
        {
            if (upper_y == 0.0 || upper_x == 0.0)
                flags |= PREFOS_CUDA_CONE_INFEASIBLE;
            if (isfinite(upper_y) && upper_y > 0.0)
            {
                double log_lower =
                    (log(minimum_abs_z) -
                     (1.0 - alpha) * log(upper_y)) /
                    alpha;
                double candidate =
                    log_lower > log(DBL_MAX) ? INFINITY : exp(log_lower);
                if (isfinite(candidate))
                    tighten_lower(lower_candidates, x_column,
                                  relax_lower(fmax(0.0, candidate),
                                              candidate));
            }
            if (isfinite(upper_x) && upper_x > 0.0)
            {
                double log_lower =
                    (log(minimum_abs_z) - alpha * log(upper_x)) /
                    (1.0 - alpha);
                double candidate =
                    log_lower > log(DBL_MAX) ? INFINITY : exp(log_lower);
                if (isfinite(candidate))
                    tighten_lower(lower_candidates, y_column,
                                  relax_lower(fmax(0.0, candidate),
                                              candidate));
            }
        }
        flags |= PREFOS_CUDA_CONE_NEEDS_CPU;
    }
    else
        flags |= PREFOS_CUDA_CONE_NEEDS_CPU;
    cone_flags[cone] = flags;
}

} // namespace

extern "C" PreFOSCudaPropagationStatus prefos_cuda_cone_envelope_round(
    PreFOSCudaWorkspace *context, const double *lower_bounds,
    const double *upper_bounds, double feasibility_tolerance,
    double *lower_candidates, double *upper_candidates,
    unsigned char *cone_flags, double *milliseconds)
{
    using Clock = std::chrono::steady_clock;
    auto start = Clock::now();
    unsigned char *device_flags = nullptr;
    cudaError_t cuda_status = cudaSuccess;
    PreFOSCudaPropagationStatus result = PREFOS_CUDA_PROPAGATION_OK;

    if (milliseconds) *milliseconds = 0.0;
    if (!context || !lower_bounds || !upper_bounds || !lower_candidates ||
        !upper_candidates || !cone_flags ||
        !isfinite(feasibility_tolerance) || feasibility_tolerance < 0.0)
        return PREFOS_CUDA_PROPAGATION_ERROR;

#define PREFOS_CUDA_CONE_CHECK(call)                                      \
    do                                                                     \
    {                                                                      \
        cuda_status = (call);                                              \
        if (cuda_status != cudaSuccess)                                    \
        {                                                                  \
            result = status_from_cuda(cuda_status);                        \
            goto cleanup;                                                  \
        }                                                                  \
    } while (0)

    PREFOS_CUDA_CONE_CHECK(copy_to_device(
        context->lower_bounds, lower_bounds, context->columns,
        context->stream));
    PREFOS_CUDA_CONE_CHECK(copy_to_device(
        context->upper_bounds, upper_bounds, context->columns,
        context->stream));
    PREFOS_CUDA_CONE_CHECK(
        allocate_device(&device_flags, context->n_cones, context->stream));
    {
        size_t count =
            context->columns > context->n_cones ? context->columns
                                                : context->n_cones;
        unsigned int blocks =
            static_cast<unsigned int>((count + kThreads - 1) / kThreads);
        if (blocks > 0)
        {
            initialize_cone_candidates_kernel<<<blocks, kThreads, 0,
                                                context->stream>>>(
                context->columns, context->n_cones,
                context->lower_candidates, context->upper_candidates,
                device_flags);
            PREFOS_CUDA_CONE_CHECK(cudaGetLastError());
        }
    }
    if (context->n_cones > 0)
    {
        cone_envelope_kernel<<<static_cast<unsigned int>(context->n_cones),
                               32, 0, context->stream>>>(
            context->n_cones, context->cone_types, context->cone_starts,
            context->cone_indices, context->cone_matrix_orders,
            context->cone_power_alphas,
            context->lower_bounds, context->upper_bounds,
            feasibility_tolerance, context->lower_candidates,
            context->upper_candidates, device_flags);
        PREFOS_CUDA_CONE_CHECK(cudaGetLastError());
    }
    PREFOS_CUDA_CONE_CHECK(copy_to_host(
        lower_candidates, context->lower_candidates, context->columns,
        context->stream));
    PREFOS_CUDA_CONE_CHECK(copy_to_host(
        upper_candidates, context->upper_candidates, context->columns,
        context->stream));
    PREFOS_CUDA_CONE_CHECK(copy_to_host(
        cone_flags, device_flags, context->n_cones, context->stream));
    PREFOS_CUDA_CONE_CHECK(cudaStreamSynchronize(context->stream));

cleanup:
    cudaFreeAsync(device_flags, context->stream);
    (void) cudaStreamSynchronize(context->stream);
    if (milliseconds)
        *milliseconds =
            std::chrono::duration<double, std::milli>(Clock::now() - start)
                .count();
    return result;
#undef PREFOS_CUDA_CONE_CHECK
}
