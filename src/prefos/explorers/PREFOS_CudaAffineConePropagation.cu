/*
 * Copyright 2026 Hongpei Li
 * SPDX-License-Identifier: Apache-2.0
 */

#include "PREFOS_CudaLinearPropagation.h"
#include "PREFOS_CudaWorkspaceInternal.cuh"

#include <cuda_runtime.h>

#include <cfloat>
#include <chrono>
#include <climits>
#include <cmath>

namespace
{

constexpr int kThreads = 256;
constexpr int kConeNonnegative = 0;
constexpr int kConeSoc = 1;
constexpr int kConeRsoc = 2;

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

__device__ double minimum_absolute_value(double lower, double upper)
{
    if (lower > 0.0) return lower;
    if (upper < 0.0) return -upper;
    return 0.0;
}

__device__ void atomic_reduce_candidate(double candidate, double *target,
                                        bool is_lower)
{
    unsigned long long *address =
        reinterpret_cast<unsigned long long *>(target);
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

__global__ void initialize_affine_indices_kernel(size_t rows, int *indices)
{
    size_t row =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row < rows) indices[row] = static_cast<int>(row);
}

__global__ void affine_activity_kernel(
    size_t rows, const int *row_pointers, const int *column_indices,
    const double *values, const double *offsets, const double *lower_bounds,
    const double *upper_bounds, double *coordinate_lower,
    double *coordinate_upper)
{
    size_t row =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    double minimum = offsets[row];
    double maximum = offsets[row];
    double work = fabs(offsets[row]);
    int infinite_minimum = 0;
    int infinite_maximum = 0;
    for (int position = row_pointers[row];
         position < row_pointers[row + 1]; ++position)
    {
        int column = column_indices[position];
        double coefficient = values[position];
        if (coefficient == 0.0) continue;
        double term_minimum =
            coefficient *
            (coefficient > 0.0 ? lower_bounds[column] : upper_bounds[column]);
        double term_maximum =
            coefficient *
            (coefficient > 0.0 ? upper_bounds[column] : lower_bounds[column]);
        if (isfinite(term_minimum))
        {
            minimum += term_minimum;
            work += fabs(term_minimum);
        }
        else
            ++infinite_minimum;
        if (isfinite(term_maximum))
        {
            maximum += term_maximum;
            work += fabs(term_maximum);
        }
        else
            ++infinite_maximum;
    }
    coordinate_lower[row] =
        infinite_minimum > 0 ? -INFINITY : relax_lower(minimum, work);
    coordinate_upper[row] =
        infinite_maximum > 0 ? INFINITY : relax_upper(maximum, work);
}

__global__ void initialize_affine_cone_candidates_kernel(
    size_t rows, size_t cones, double *lower_candidates,
    double *upper_candidates, unsigned char *flags)
{
    size_t index =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < rows)
    {
        lower_candidates[index] = -INFINITY;
        upper_candidates[index] = INFINITY;
    }
    if (index < cones) flags[index] = 0;
}

__device__ void tighten_lower(double *candidate, int coordinate, double value)
{
    if (value > candidate[coordinate]) candidate[coordinate] = value;
}

__device__ void tighten_upper(double *candidate, int coordinate, double value)
{
    if (value < candidate[coordinate]) candidate[coordinate] = value;
}

__global__ void affine_cone_envelope_kernel(
    size_t cones, const int *cone_types, const int *cone_starts,
    const int *cone_indices, const double *lower_bounds,
    const double *upper_bounds, double feasibility_tolerance,
    double *lower_candidates, double *upper_candidates,
    unsigned char *cone_flags)
{
    size_t cone = static_cast<size_t>(blockIdx.x);
    if (cone >= cones || threadIdx.x != 0) return;
    int type = cone_types[cone];
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
            int coordinate = cone_indices[start + local];
            double minimum = minimum_absolute_value(
                lower_bounds[coordinate], upper_bounds[coordinate]);
            minimum_norm_squared += minimum * minimum;
        }
        if (!isfinite(minimum_norm_squared) || dimension > 4096)
        {
            cone_flags[cone] =
                static_cast<unsigned char>(flags |
                                           PREFOS_CUDA_CONE_NEEDS_CPU);
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
            if (minimum_norm >
                axis_upper + tolerance +
                    1024.0 * DBL_EPSILON *
                        fmax(1.0, fmax(fabs(axis_upper), minimum_norm)))
                flags |= PREFOS_CUDA_CONE_INFEASIBLE;
            double capacity = axis_upper * axis_upper;
            for (int local = 1; local < dimension; ++local)
            {
                int coordinate = cone_indices[start + local];
                double minimum = minimum_absolute_value(
                    lower_bounds[coordinate], upper_bounds[coordinate]);
                double remaining = fmax(
                    0.0, capacity -
                             (minimum_norm_squared - minimum * minimum));
                double limit = sqrt(remaining);
                tighten_lower(lower_candidates, coordinate,
                              relax_lower(-limit, capacity));
                tighten_upper(upper_candidates, coordinate,
                              relax_upper(limit, capacity));
            }
        }
    }
    else if (type == kConeRsoc && dimension >= 3)
    {
        int u_coordinate = cone_indices[start];
        int v_coordinate = cone_indices[start + 1];
        double minimum_norm_squared = 0.0;
        for (int local = 2; local < dimension; ++local)
        {
            int coordinate = cone_indices[start + local];
            double minimum = minimum_absolute_value(
                lower_bounds[coordinate], upper_bounds[coordinate]);
            minimum_norm_squared += minimum * minimum;
        }
        if (!isfinite(minimum_norm_squared) || dimension > 4096)
        {
            cone_flags[cone] =
                static_cast<unsigned char>(flags |
                                           PREFOS_CUDA_CONE_NEEDS_CPU);
            return;
        }
        tighten_lower(lower_candidates, u_coordinate, 0.0);
        tighten_lower(lower_candidates, v_coordinate, 0.0);
        double u_upper = upper_bounds[u_coordinate];
        double v_upper = upper_bounds[v_coordinate];
        if (isfinite(v_upper) && v_upper > 0.0)
            tighten_lower(
                lower_candidates, u_coordinate,
                relax_lower(minimum_norm_squared / (2.0 * v_upper),
                            minimum_norm_squared));
        if (isfinite(u_upper) && u_upper > 0.0)
            tighten_lower(
                lower_candidates, v_coordinate,
                relax_lower(minimum_norm_squared / (2.0 * u_upper),
                            minimum_norm_squared));
        if ((u_upper == 0.0 || v_upper == 0.0) &&
            minimum_norm_squared > 0.0)
            flags |= PREFOS_CUDA_CONE_INFEASIBLE;
        if (u_upper == 0.0 || v_upper == 0.0)
        {
            for (int local = 2; local < dimension; ++local)
            {
                int coordinate = cone_indices[start + local];
                tighten_lower(lower_candidates, coordinate, 0.0);
                tighten_upper(upper_candidates, coordinate, 0.0);
            }
        }
        else if (isfinite(u_upper) && isfinite(v_upper))
        {
            double capacity = 2.0 * u_upper * v_upper;
            double tolerance =
                feasibility_tolerance * fmax(1.0, fabs(capacity));
            if (minimum_norm_squared >
                capacity + tolerance +
                    1024.0 * DBL_EPSILON *
                        fmax(1.0, fmax(fabs(capacity),
                                       minimum_norm_squared)))
                flags |= PREFOS_CUDA_CONE_INFEASIBLE;
            for (int local = 2; local < dimension; ++local)
            {
                int coordinate = cone_indices[start + local];
                double minimum = minimum_absolute_value(
                    lower_bounds[coordinate], upper_bounds[coordinate]);
                double remaining = fmax(
                    0.0, capacity -
                             (minimum_norm_squared - minimum * minimum));
                double limit = sqrt(remaining);
                tighten_lower(lower_candidates, coordinate,
                              relax_lower(-limit, capacity));
                tighten_upper(upper_candidates, coordinate,
                              relax_upper(limit, capacity));
            }
        }
    }
    else
        flags |= PREFOS_CUDA_CONE_NEEDS_CPU;
    cone_flags[cone] = flags;
}

__global__ void merge_affine_cone_candidates_kernel(
    size_t rows, const double *activity_lower,
    const double *activity_upper, double *lower_candidates,
    double *upper_candidates)
{
    size_t row =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    lower_candidates[row] =
        fmax(activity_lower[row], lower_candidates[row]);
    upper_candidates[row] =
        fmin(activity_upper[row], upper_candidates[row]);
}

__global__ void initialize_variable_candidates_kernel(
    size_t columns, double *lower_candidates, double *upper_candidates)
{
    size_t column =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (column >= columns) return;
    lower_candidates[column] = -INFINITY;
    upper_candidates[column] = INFINITY;
}

__device__ bool activity_without_term(double finite, int infinite,
                                      double term, double *result)
{
    if (isfinite(term))
    {
        if (infinite != 0) return false;
        *result = finite - term;
    }
    else
    {
        if (infinite != 1) return false;
        *result = finite;
    }
    return isfinite(*result);
}

__global__ void affine_row_propagation_kernel(
    size_t rows, const int *row_pointers, const int *column_indices,
    const double *values, const double *offsets,
    const double *variable_lower, const double *variable_upper,
    const double *coordinate_lower, const double *coordinate_upper,
    double maximum_magnitude, double *lower_candidates,
    double *upper_candidates)
{
    size_t row =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    double finite_minimum = offsets[row];
    double finite_maximum = offsets[row];
    double work = fabs(offsets[row]);
    int infinite_minimum = 0;
    int infinite_maximum = 0;
    for (int position = row_pointers[row];
         position < row_pointers[row + 1]; ++position)
    {
        int column = column_indices[position];
        double coefficient = values[position];
        if (coefficient == 0.0) continue;
        double term_minimum =
            coefficient *
            (coefficient > 0.0 ? variable_lower[column]
                               : variable_upper[column]);
        double term_maximum =
            coefficient *
            (coefficient > 0.0 ? variable_upper[column]
                               : variable_lower[column]);
        if (isfinite(term_minimum))
        {
            finite_minimum += term_minimum;
            work += fabs(term_minimum);
        }
        else
            ++infinite_minimum;
        if (isfinite(term_maximum))
        {
            finite_maximum += term_maximum;
            work += fabs(term_maximum);
        }
        else
            ++infinite_maximum;
    }
    for (int position = row_pointers[row];
         position < row_pointers[row + 1]; ++position)
    {
        int column = column_indices[position];
        double coefficient = values[position];
        if (coefficient == 0.0) continue;
        double term_minimum =
            coefficient *
            (coefficient > 0.0 ? variable_lower[column]
                               : variable_upper[column]);
        double term_maximum =
            coefficient *
            (coefficient > 0.0 ? variable_upper[column]
                               : variable_lower[column]);
        double other;
        if (isfinite(coordinate_lower[row]) &&
            activity_without_term(finite_maximum, infinite_maximum,
                                  term_maximum, &other))
        {
            double candidate =
                (coordinate_lower[row] - other) / coefficient;
            double error =
                65536.0 * DBL_EPSILON *
                fmax(1.0, work + fabs(coordinate_lower[row]) +
                              fabs(candidate * coefficient)) /
                fabs(coefficient);
            bool is_lower = coefficient > 0.0;
            candidate += is_lower ? -error : error;
            if (isfinite(candidate) &&
                (maximum_magnitude <= 0.0 ||
                 fabs(candidate) < maximum_magnitude))
                atomic_reduce_candidate(
                    candidate,
                    is_lower ? lower_candidates + column
                             : upper_candidates + column,
                    is_lower);
        }
        if (isfinite(coordinate_upper[row]) &&
            activity_without_term(finite_minimum, infinite_minimum,
                                  term_minimum, &other))
        {
            double candidate =
                (coordinate_upper[row] - other) / coefficient;
            double error =
                65536.0 * DBL_EPSILON *
                fmax(1.0, work + fabs(coordinate_upper[row]) +
                              fabs(candidate * coefficient)) /
                fabs(coefficient);
            bool is_lower = coefficient < 0.0;
            candidate += is_lower ? -error : error;
            if (isfinite(candidate) &&
                (maximum_magnitude <= 0.0 ||
                 fabs(candidate) < maximum_magnitude))
                atomic_reduce_candidate(
                    candidate,
                    is_lower ? lower_candidates + column
                             : upper_candidates + column,
                    is_lower);
        }
    }
}

void clear_affine_workspace(PreFOSCudaWorkspace *context)
{
    cudaFreeAsync(context->affine_row_pointers, context->stream);
    cudaFreeAsync(context->affine_column_indices, context->stream);
    cudaFreeAsync(context->affine_values, context->stream);
    cudaFreeAsync(context->affine_offsets, context->stream);
    cudaFreeAsync(context->affine_cone_types, context->stream);
    cudaFreeAsync(context->affine_cone_starts, context->stream);
    cudaFreeAsync(context->affine_cone_indices, context->stream);
    cudaFreeAsync(context->affine_cone_matrix_orders, context->stream);
    cudaFreeAsync(context->affine_cone_power_alphas, context->stream);
    cudaFreeAsync(context->affine_lower_bounds, context->stream);
    cudaFreeAsync(context->affine_upper_bounds, context->stream);
    cudaFreeAsync(context->affine_lower_candidates, context->stream);
    cudaFreeAsync(context->affine_upper_candidates, context->stream);
    context->affine_row_pointers = nullptr;
    context->affine_column_indices = nullptr;
    context->affine_values = nullptr;
    context->affine_offsets = nullptr;
    context->affine_cone_types = nullptr;
    context->affine_cone_starts = nullptr;
    context->affine_cone_indices = nullptr;
    context->affine_cone_matrix_orders = nullptr;
    context->affine_cone_power_alphas = nullptr;
    context->affine_lower_bounds = nullptr;
    context->affine_upper_bounds = nullptr;
    context->affine_lower_candidates = nullptr;
    context->affine_upper_candidates = nullptr;
    context->affine_rows = 0;
    context->affine_nnz = 0;
    context->n_affine_cones = 0;
}

} // namespace

extern "C" PreFOSCudaPropagationStatus prefos_cuda_workspace_attach_affine(
    PreFOSCudaWorkspace *context, size_t rows, size_t nnz,
    const int *row_pointers, const int *column_indices,
    const double *values, const double *offsets, size_t n_cones,
    const int *cone_types, const int *cone_starts,
    const int *cone_matrix_orders, const double *cone_power_alphas)
{
    cudaError_t cuda_status = cudaSuccess;
    if (!context || context->affine_rows != 0 ||
        rows > static_cast<size_t>(INT_MAX) ||
        n_cones > static_cast<size_t>(UINT_MAX) ||
        (rows > 0 && (!row_pointers || !offsets)) ||
        (nnz > 0 && (!column_indices || !values)) ||
        (n_cones > 0 &&
         (!cone_types || !cone_starts || !cone_matrix_orders ||
          !cone_power_alphas)))
        return PREFOS_CUDA_PROPAGATION_ERROR;

#define PREFOS_CUDA_AFFINE_ATTACH_CHECK(call)                              \
    do                                                                     \
    {                                                                      \
        cuda_status = (call);                                              \
        if (cuda_status != cudaSuccess) goto failure;                      \
    } while (0)

    PREFOS_CUDA_AFFINE_ATTACH_CHECK(
        allocate_device(&context->affine_row_pointers, rows + 1,
                        context->stream));
    PREFOS_CUDA_AFFINE_ATTACH_CHECK(
        allocate_device(&context->affine_column_indices, nnz,
                        context->stream));
    PREFOS_CUDA_AFFINE_ATTACH_CHECK(
        allocate_device(&context->affine_values, nnz, context->stream));
    PREFOS_CUDA_AFFINE_ATTACH_CHECK(
        allocate_device(&context->affine_offsets, rows, context->stream));
    PREFOS_CUDA_AFFINE_ATTACH_CHECK(
        allocate_device(&context->affine_cone_types, n_cones,
                        context->stream));
    PREFOS_CUDA_AFFINE_ATTACH_CHECK(
        allocate_device(&context->affine_cone_starts,
                        n_cones > 0 ? n_cones + 1 : 0, context->stream));
    PREFOS_CUDA_AFFINE_ATTACH_CHECK(
        allocate_device(&context->affine_cone_indices, rows,
                        context->stream));
    PREFOS_CUDA_AFFINE_ATTACH_CHECK(
        allocate_device(&context->affine_cone_matrix_orders, n_cones,
                        context->stream));
    PREFOS_CUDA_AFFINE_ATTACH_CHECK(
        allocate_device(&context->affine_cone_power_alphas, n_cones,
                        context->stream));
    PREFOS_CUDA_AFFINE_ATTACH_CHECK(
        allocate_device(&context->affine_lower_bounds, rows,
                        context->stream));
    PREFOS_CUDA_AFFINE_ATTACH_CHECK(
        allocate_device(&context->affine_upper_bounds, rows,
                        context->stream));
    PREFOS_CUDA_AFFINE_ATTACH_CHECK(
        allocate_device(&context->affine_lower_candidates, rows,
                        context->stream));
    PREFOS_CUDA_AFFINE_ATTACH_CHECK(
        allocate_device(&context->affine_upper_candidates, rows,
                        context->stream));
    PREFOS_CUDA_AFFINE_ATTACH_CHECK(copy_to_device(
        context->affine_row_pointers, row_pointers, rows + 1,
        context->stream));
    PREFOS_CUDA_AFFINE_ATTACH_CHECK(copy_to_device(
        context->affine_column_indices, column_indices, nnz,
        context->stream));
    PREFOS_CUDA_AFFINE_ATTACH_CHECK(copy_to_device(
        context->affine_values, values, nnz, context->stream));
    PREFOS_CUDA_AFFINE_ATTACH_CHECK(copy_to_device(
        context->affine_offsets, offsets, rows, context->stream));
    PREFOS_CUDA_AFFINE_ATTACH_CHECK(copy_to_device(
        context->affine_cone_types, cone_types, n_cones,
        context->stream));
    PREFOS_CUDA_AFFINE_ATTACH_CHECK(copy_to_device(
        context->affine_cone_starts, cone_starts,
        n_cones > 0 ? n_cones + 1 : 0, context->stream));
    PREFOS_CUDA_AFFINE_ATTACH_CHECK(copy_to_device(
        context->affine_cone_matrix_orders, cone_matrix_orders, n_cones,
        context->stream));
    PREFOS_CUDA_AFFINE_ATTACH_CHECK(copy_to_device(
        context->affine_cone_power_alphas, cone_power_alphas, n_cones,
        context->stream));
    if (rows > 0)
    {
        unsigned int blocks =
            static_cast<unsigned int>((rows + kThreads - 1) / kThreads);
        initialize_affine_indices_kernel<<<blocks, kThreads, 0,
                                           context->stream>>>(
            rows, context->affine_cone_indices);
        PREFOS_CUDA_AFFINE_ATTACH_CHECK(cudaGetLastError());
    }
    PREFOS_CUDA_AFFINE_ATTACH_CHECK(
        cudaStreamSynchronize(context->stream));
    context->affine_rows = rows;
    context->affine_nnz = nnz;
    context->n_affine_cones = n_cones;
    return PREFOS_CUDA_PROPAGATION_OK;

failure:
    clear_affine_workspace(context);
    (void) cudaStreamSynchronize(context->stream);
    return status_from_cuda(cuda_status);
#undef PREFOS_CUDA_AFFINE_ATTACH_CHECK
}

extern "C" PreFOSCudaPropagationStatus
prefos_cuda_affine_coordinate_activity(
    PreFOSCudaWorkspace *context, const double *lower_bounds,
    const double *upper_bounds, double *coordinate_lower,
    double *coordinate_upper, double *milliseconds)
{
    using Clock = std::chrono::steady_clock;
    auto start = Clock::now();
    cudaError_t cuda_status = cudaSuccess;
    if (milliseconds) *milliseconds = 0.0;
    if (!context || context->affine_rows == 0 || !lower_bounds ||
        !upper_bounds || !coordinate_lower || !coordinate_upper)
        return PREFOS_CUDA_PROPAGATION_UNAVAILABLE;

#define PREFOS_CUDA_AFFINE_ACTIVITY_CHECK(call)                            \
    do                                                                     \
    {                                                                      \
        cuda_status = (call);                                              \
        if (cuda_status != cudaSuccess)                                    \
            return status_from_cuda(cuda_status);                          \
    } while (0)

    PREFOS_CUDA_AFFINE_ACTIVITY_CHECK(copy_to_device(
        context->lower_bounds, lower_bounds, context->columns,
        context->stream));
    PREFOS_CUDA_AFFINE_ACTIVITY_CHECK(copy_to_device(
        context->upper_bounds, upper_bounds, context->columns,
        context->stream));
    {
        unsigned int blocks = static_cast<unsigned int>(
            (context->affine_rows + kThreads - 1) / kThreads);
        affine_activity_kernel<<<blocks, kThreads, 0, context->stream>>>(
            context->affine_rows, context->affine_row_pointers,
            context->affine_column_indices, context->affine_values,
            context->affine_offsets, context->lower_bounds,
            context->upper_bounds, context->affine_lower_bounds,
            context->affine_upper_bounds);
        PREFOS_CUDA_AFFINE_ACTIVITY_CHECK(cudaGetLastError());
    }
    PREFOS_CUDA_AFFINE_ACTIVITY_CHECK(copy_to_host(
        coordinate_lower, context->affine_lower_bounds,
        context->affine_rows, context->stream));
    PREFOS_CUDA_AFFINE_ACTIVITY_CHECK(copy_to_host(
        coordinate_upper, context->affine_upper_bounds,
        context->affine_rows, context->stream));
    PREFOS_CUDA_AFFINE_ACTIVITY_CHECK(
        cudaStreamSynchronize(context->stream));
    if (milliseconds)
        *milliseconds =
            std::chrono::duration<double, std::milli>(Clock::now() - start)
                .count();
    return PREFOS_CUDA_PROPAGATION_OK;
#undef PREFOS_CUDA_AFFINE_ACTIVITY_CHECK
}

extern "C" PreFOSCudaPropagationStatus
prefos_cuda_affine_cone_envelope_round(
    PreFOSCudaWorkspace *context, const double *coordinate_lower,
    const double *coordinate_upper, double feasibility_tolerance,
    double *lower_candidates, double *upper_candidates,
    unsigned char *cone_flags, double *milliseconds)
{
    using Clock = std::chrono::steady_clock;
    auto start = Clock::now();
    unsigned char *device_flags = nullptr;
    cudaError_t cuda_status = cudaSuccess;
    PreFOSCudaPropagationStatus result = PREFOS_CUDA_PROPAGATION_OK;
    if (milliseconds) *milliseconds = 0.0;
    if (!context || context->affine_rows == 0 || !coordinate_lower ||
        !coordinate_upper || !lower_candidates || !upper_candidates ||
        !cone_flags || !isfinite(feasibility_tolerance) ||
        feasibility_tolerance < 0.0)
        return PREFOS_CUDA_PROPAGATION_UNAVAILABLE;

#define PREFOS_CUDA_AFFINE_CONE_CHECK(call)                                \
    do                                                                     \
    {                                                                      \
        cuda_status = (call);                                              \
        if (cuda_status != cudaSuccess)                                    \
        {                                                                  \
            result = status_from_cuda(cuda_status);                        \
            goto cleanup;                                                  \
        }                                                                  \
    } while (0)

    PREFOS_CUDA_AFFINE_CONE_CHECK(copy_to_device(
        context->affine_lower_bounds, coordinate_lower,
        context->affine_rows, context->stream));
    PREFOS_CUDA_AFFINE_CONE_CHECK(copy_to_device(
        context->affine_upper_bounds, coordinate_upper,
        context->affine_rows, context->stream));
    PREFOS_CUDA_AFFINE_CONE_CHECK(allocate_device(
        &device_flags, context->n_affine_cones, context->stream));
    {
        size_t count = context->affine_rows > context->n_affine_cones
                           ? context->affine_rows
                           : context->n_affine_cones;
        unsigned int blocks =
            static_cast<unsigned int>((count + kThreads - 1) / kThreads);
        initialize_affine_cone_candidates_kernel<<<blocks, kThreads, 0,
                                                    context->stream>>>(
            context->affine_rows, context->n_affine_cones,
            context->affine_lower_candidates,
            context->affine_upper_candidates, device_flags);
        PREFOS_CUDA_AFFINE_CONE_CHECK(cudaGetLastError());
    }
    affine_cone_envelope_kernel<<<
        static_cast<unsigned int>(context->n_affine_cones), 32, 0,
        context->stream>>>(
        context->n_affine_cones, context->affine_cone_types,
        context->affine_cone_starts, context->affine_cone_indices,
        context->affine_lower_bounds, context->affine_upper_bounds,
        feasibility_tolerance, context->affine_lower_candidates,
        context->affine_upper_candidates, device_flags);
    PREFOS_CUDA_AFFINE_CONE_CHECK(cudaGetLastError());
    {
        unsigned int blocks = static_cast<unsigned int>(
            (context->affine_rows + kThreads - 1) / kThreads);
        merge_affine_cone_candidates_kernel<<<blocks, kThreads, 0,
                                               context->stream>>>(
            context->affine_rows, context->affine_lower_bounds,
            context->affine_upper_bounds,
            context->affine_lower_candidates,
            context->affine_upper_candidates);
        PREFOS_CUDA_AFFINE_CONE_CHECK(cudaGetLastError());
    }
    PREFOS_CUDA_AFFINE_CONE_CHECK(copy_to_host(
        lower_candidates, context->affine_lower_candidates,
        context->affine_rows, context->stream));
    PREFOS_CUDA_AFFINE_CONE_CHECK(copy_to_host(
        upper_candidates, context->affine_upper_candidates,
        context->affine_rows, context->stream));
    PREFOS_CUDA_AFFINE_CONE_CHECK(copy_to_host(
        cone_flags, device_flags, context->n_affine_cones,
        context->stream));
    PREFOS_CUDA_AFFINE_CONE_CHECK(
        cudaStreamSynchronize(context->stream));

cleanup:
    cudaFreeAsync(device_flags, context->stream);
    (void) cudaStreamSynchronize(context->stream);
    if (milliseconds)
        *milliseconds =
            std::chrono::duration<double, std::milli>(Clock::now() - start)
                .count();
    return result;
#undef PREFOS_CUDA_AFFINE_CONE_CHECK
}

extern "C" PreFOSCudaPropagationStatus
prefos_cuda_affine_row_propagation(
    PreFOSCudaWorkspace *context, const double *lower_bounds,
    const double *upper_bounds, const double *coordinate_lower,
    const double *coordinate_upper, double maximum_inferred_bound_magnitude,
    double *lower_candidates, double *upper_candidates,
    double *milliseconds)
{
    using Clock = std::chrono::steady_clock;
    auto start = Clock::now();
    cudaError_t cuda_status = cudaSuccess;
    if (milliseconds) *milliseconds = 0.0;
    if (!context || context->affine_rows == 0 || !lower_bounds ||
        !upper_bounds || !coordinate_lower || !coordinate_upper ||
        !lower_candidates || !upper_candidates)
        return PREFOS_CUDA_PROPAGATION_UNAVAILABLE;

#define PREFOS_CUDA_AFFINE_ROW_CHECK(call)                                 \
    do                                                                     \
    {                                                                      \
        cuda_status = (call);                                              \
        if (cuda_status != cudaSuccess)                                    \
            return status_from_cuda(cuda_status);                          \
    } while (0)

    PREFOS_CUDA_AFFINE_ROW_CHECK(copy_to_device(
        context->lower_bounds, lower_bounds, context->columns,
        context->stream));
    PREFOS_CUDA_AFFINE_ROW_CHECK(copy_to_device(
        context->upper_bounds, upper_bounds, context->columns,
        context->stream));
    PREFOS_CUDA_AFFINE_ROW_CHECK(copy_to_device(
        context->affine_lower_bounds, coordinate_lower,
        context->affine_rows, context->stream));
    PREFOS_CUDA_AFFINE_ROW_CHECK(copy_to_device(
        context->affine_upper_bounds, coordinate_upper,
        context->affine_rows, context->stream));
    {
        unsigned int blocks = static_cast<unsigned int>(
            (context->columns + kThreads - 1) / kThreads);
        if (blocks > 0)
            initialize_variable_candidates_kernel<<<blocks, kThreads, 0,
                                                     context->stream>>>(
                context->columns, context->lower_candidates,
                context->upper_candidates);
        PREFOS_CUDA_AFFINE_ROW_CHECK(cudaGetLastError());
    }
    {
        unsigned int blocks = static_cast<unsigned int>(
            (context->affine_rows + kThreads - 1) / kThreads);
        affine_row_propagation_kernel<<<blocks, kThreads, 0,
                                        context->stream>>>(
            context->affine_rows, context->affine_row_pointers,
            context->affine_column_indices, context->affine_values,
            context->affine_offsets, context->lower_bounds,
            context->upper_bounds, context->affine_lower_bounds,
            context->affine_upper_bounds,
            maximum_inferred_bound_magnitude,
            context->lower_candidates, context->upper_candidates);
        PREFOS_CUDA_AFFINE_ROW_CHECK(cudaGetLastError());
    }
    PREFOS_CUDA_AFFINE_ROW_CHECK(copy_to_host(
        lower_candidates, context->lower_candidates, context->columns,
        context->stream));
    PREFOS_CUDA_AFFINE_ROW_CHECK(copy_to_host(
        upper_candidates, context->upper_candidates, context->columns,
        context->stream));
    PREFOS_CUDA_AFFINE_ROW_CHECK(
        cudaStreamSynchronize(context->stream));
    if (milliseconds)
        *milliseconds =
            std::chrono::duration<double, std::milli>(Clock::now() - start)
                .count();
    return PREFOS_CUDA_PROPAGATION_OK;
#undef PREFOS_CUDA_AFFINE_ROW_CHECK
}
