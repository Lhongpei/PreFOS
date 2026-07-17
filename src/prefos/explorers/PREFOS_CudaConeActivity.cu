/*
 * Copyright 2026 Hongpei Li
 * SPDX-License-Identifier: Apache-2.0
 */

#include "PREFOS_CudaLinearPropagation.h"
#include "PREFOS_CudaWorkspaceInternal.cuh"

#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_run_length_encode.cuh>
#include <cub/device/device_scan.cuh>
#include <cuda_runtime.h>

#include <cfloat>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <new>

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

__global__ void build_activity_keys_kernel(
    size_t rows, const int *row_pointers, const int *column_indices,
    const int *column_to_cone, int has_cones, uint64_t *keys, int *positions)
{
    size_t row = static_cast<size_t>(blockIdx.x);
    if (row >= rows) return;
    int start = row_pointers[row];
    int end = row_pointers[row + 1];
    for (int position = start + static_cast<int>(threadIdx.x); position < end;
         position += static_cast<int>(blockDim.x))
    {
        int column = column_indices[position];
        int cone = has_cones ? column_to_cone[column] : -1;
        uint32_t group = cone >= 0 ? static_cast<uint32_t>(cone) + 1U : 0U;
        keys[position] = (static_cast<uint64_t>(row) << 32U) |
                         static_cast<uint64_t>(group);
        positions[position] = position;
    }
}

PreFOSCudaPropagationStatus build_cone_activity_groups(
    PreFOSCudaWorkspace *context)
{
    uint64_t *keys_in = nullptr, *keys_out = nullptr;
    int *positions_in = nullptr;
    int *group_counts = nullptr, *device_groups = nullptr;
    void *sort_storage = nullptr, *rle_storage = nullptr, *scan_storage = nullptr;
    size_t sort_bytes = 0, rle_bytes = 0, scan_bytes = 0;
    int host_groups = 0;
    int host_nnz;
    cudaError_t cuda_status = cudaSuccess;
    PreFOSCudaPropagationStatus result = PREFOS_CUDA_PROPAGATION_OK;

    if (context->cone_activity_groups_ready)
        return PREFOS_CUDA_PROPAGATION_OK;
    if (context->nnz > static_cast<size_t>(INT_MAX) ||
        context->rows > static_cast<size_t>(UINT_MAX))
        return PREFOS_CUDA_PROPAGATION_ERROR;
    if (context->nnz == 0)
    {
        context->cone_activity_groups_ready = 1;
        return PREFOS_CUDA_PROPAGATION_OK;
    }
    host_nnz = static_cast<int>(context->nnz);

#define PREFOS_CUDA_GROUP_CHECK(call)                                      \
    do                                                                      \
    {                                                                       \
        cuda_status = (call);                                               \
        if (cuda_status != cudaSuccess)                                     \
        {                                                                   \
            result = status_from_cuda(cuda_status);                         \
            goto cleanup;                                                   \
        }                                                                   \
    } while (0)

    PREFOS_CUDA_GROUP_CHECK(
        allocate_device(&keys_in, context->nnz, context->stream));
    PREFOS_CUDA_GROUP_CHECK(
        allocate_device(&keys_out, context->nnz, context->stream));
    PREFOS_CUDA_GROUP_CHECK(
        allocate_device(&positions_in, context->nnz, context->stream));
    PREFOS_CUDA_GROUP_CHECK(allocate_device(
        &context->cone_activity_sorted_positions, context->nnz,
        context->stream));
    PREFOS_CUDA_GROUP_CHECK(allocate_device(
        &context->cone_activity_group_keys, context->nnz, context->stream));
    PREFOS_CUDA_GROUP_CHECK(allocate_device(
        &context->cone_activity_group_offsets, context->nnz + 1,
        context->stream));
    PREFOS_CUDA_GROUP_CHECK(
        allocate_device(&group_counts, context->nnz, context->stream));
    PREFOS_CUDA_GROUP_CHECK(
        allocate_device(&device_groups, 1, context->stream));

    build_activity_keys_kernel<<<static_cast<unsigned int>(context->rows),
                                 kThreads, 0, context->stream>>>(
        context->rows, context->row_pointers, context->column_indices,
        context->column_to_cone, context->n_cones > 0, keys_in,
        positions_in);
    PREFOS_CUDA_GROUP_CHECK(cudaGetLastError());
    PREFOS_CUDA_GROUP_CHECK(cub::DeviceRadixSort::SortPairs(
        nullptr, sort_bytes, keys_in, keys_out, positions_in,
        context->cone_activity_sorted_positions, host_nnz, 0, 64,
        context->stream));
    PREFOS_CUDA_GROUP_CHECK(
        cudaMallocAsync(&sort_storage, sort_bytes, context->stream));
    PREFOS_CUDA_GROUP_CHECK(cub::DeviceRadixSort::SortPairs(
        sort_storage, sort_bytes, keys_in, keys_out, positions_in,
        context->cone_activity_sorted_positions, host_nnz, 0, 64,
        context->stream));
    PREFOS_CUDA_GROUP_CHECK(cub::DeviceRunLengthEncode::Encode(
        nullptr, rle_bytes, keys_out, context->cone_activity_group_keys,
        group_counts, device_groups, host_nnz, context->stream));
    PREFOS_CUDA_GROUP_CHECK(
        cudaMallocAsync(&rle_storage, rle_bytes, context->stream));
    PREFOS_CUDA_GROUP_CHECK(cub::DeviceRunLengthEncode::Encode(
        rle_storage, rle_bytes, keys_out,
        context->cone_activity_group_keys, group_counts, device_groups,
        host_nnz, context->stream));
    PREFOS_CUDA_GROUP_CHECK(
        copy_to_host(&host_groups, device_groups, 1, context->stream));
    PREFOS_CUDA_GROUP_CHECK(cudaStreamSynchronize(context->stream));
    if (host_groups <= 0 || host_groups > host_nnz)
    {
        result = PREFOS_CUDA_PROPAGATION_ERROR;
        goto cleanup;
    }
    PREFOS_CUDA_GROUP_CHECK(cub::DeviceScan::ExclusiveSum(
        nullptr, scan_bytes, group_counts,
        context->cone_activity_group_offsets, host_groups,
        context->stream));
    PREFOS_CUDA_GROUP_CHECK(
        cudaMallocAsync(&scan_storage, scan_bytes, context->stream));
    PREFOS_CUDA_GROUP_CHECK(cub::DeviceScan::ExclusiveSum(
        scan_storage, scan_bytes, group_counts,
        context->cone_activity_group_offsets, host_groups,
        context->stream));
    PREFOS_CUDA_GROUP_CHECK(cudaMemcpyAsync(
        context->cone_activity_group_offsets + host_groups, &host_nnz,
        sizeof(int), cudaMemcpyHostToDevice, context->stream));
    PREFOS_CUDA_GROUP_CHECK(cudaStreamSynchronize(context->stream));
    context->n_cone_activity_groups = static_cast<size_t>(host_groups);
    context->cone_activity_groups_ready = 1;

cleanup:
    cudaFreeAsync(keys_in, context->stream);
    cudaFreeAsync(keys_out, context->stream);
    cudaFreeAsync(positions_in, context->stream);
    cudaFreeAsync(group_counts, context->stream);
    cudaFreeAsync(device_groups, context->stream);
    cudaFreeAsync(sort_storage, context->stream);
    cudaFreeAsync(rle_storage, context->stream);
    cudaFreeAsync(scan_storage, context->stream);
    (void) cudaStreamSynchronize(context->stream);
    if (result != PREFOS_CUDA_PROPAGATION_OK)
    {
        cudaFreeAsync(context->cone_activity_sorted_positions,
                      context->stream);
        cudaFreeAsync(context->cone_activity_group_keys, context->stream);
        cudaFreeAsync(context->cone_activity_group_offsets,
                      context->stream);
        (void) cudaStreamSynchronize(context->stream);
        context->cone_activity_sorted_positions = nullptr;
        context->cone_activity_group_keys = nullptr;
        context->cone_activity_group_offsets = nullptr;
        context->n_cone_activity_groups = 0;
    }
    return result;
#undef PREFOS_CUDA_GROUP_CHECK
}

__device__ double selected_bound(double coefficient, double lower,
                                 double upper, bool minimum)
{
    return (coefficient > 0.0) == minimum ? lower : upper;
}

__device__ bool exp_primal_contains(double x, double y, double z)
{
    if (!isfinite(x) || !isfinite(y) || !isfinite(z) || y < 0.0 ||
        z < 0.0)
        return false;
    if (y == 0.0) return x <= 0.0;
    if (z == 0.0) return false;
    double left = log(y) + x / y;
    double right = log(z);
    double margin =
        256.0 * DBL_EPSILON * fmax(1.0, fmax(fabs(left), fabs(right)));
    return left <= right - margin;
}

__device__ bool exp_dual_contains(double x, double y, double z)
{
    return exp_primal_contains(-y, -x, exp(1.0) * z);
}

__device__ bool power_dual_contains(double x, double y, double z,
                                    double alpha)
{
    x /= alpha;
    y /= 1.0 - alpha;
    if (!isfinite(x) || !isfinite(y) || !isfinite(z) || x < 0.0 ||
        y < 0.0)
        return false;
    if (z == 0.0) return true;
    if (x == 0.0 || y == 0.0) return false;
    double left = alpha * log(x) + (1.0 - alpha) * log(y);
    double right = log(fabs(z));
    double margin =
        256.0 * DBL_EPSILON * fmax(1.0, fmax(fabs(left), fabs(right)));
    return left >= right + margin;
}

__device__ void atomic_add_double(double *address, double value)
{
    unsigned long long *integer_address =
        reinterpret_cast<unsigned long long *>(address);
    unsigned long long old = *integer_address;
    unsigned long long assumed;
    do
    {
        assumed = old;
        old = atomicCAS(
            integer_address, assumed,
            __double_as_longlong(value + __longlong_as_double(assumed)));
    } while (assumed != old);
}

__device__ void atomic_add_activity(
    size_t row, double finite_min, double finite_max, double absolute_min,
    double absolute_max, int infinite_min, int infinite_max, int nonzeros,
    double *row_finite_min, double *row_finite_max, double *row_absolute_min,
    double *row_absolute_max, int *row_infinite_min, int *row_infinite_max,
    int *row_nonzeros)
{
    atomic_add_double(row_finite_min + row, finite_min);
    atomic_add_double(row_finite_max + row, finite_max);
    atomic_add_double(row_absolute_min + row, absolute_min);
    atomic_add_double(row_absolute_max + row, absolute_max);
    atomicAdd(row_infinite_min + row, infinite_min);
    atomicAdd(row_infinite_max + row, infinite_max);
    atomicAdd(row_nonzeros + row, nonzeros);
}

__global__ void cone_activity_group_kernel(
    size_t groups, const uint64_t *group_keys, const int *group_offsets,
    const int *sorted_positions, const int *column_indices,
    const double *values, const int *column_to_cone_position,
    const int *cone_types, const int *cone_matrix_orders,
    const double *cone_power_alphas, const double *lower_bounds,
    const double *upper_bounds, const unsigned char *remove_rows,
    double *row_finite_min, double *row_finite_max,
    double *row_absolute_min, double *row_absolute_max,
    int *row_infinite_min, int *row_infinite_max, int *row_nonzeros,
    int *row_needs_cpu, int *row_strengthened)
{
    size_t group = static_cast<size_t>(blockIdx.x);
    if (group >= groups || threadIdx.x != 0) return;
    uint64_t key = group_keys[group];
    size_t row = static_cast<size_t>(key >> 32U);
    uint32_t token = static_cast<uint32_t>(key);
    if (remove_rows[row]) return;
    int start = group_offsets[group];
    int end = group_offsets[group + 1];
    double finite_min = 0.0, finite_max = 0.0;
    double absolute_min = 0.0, absolute_max = 0.0;
    int infinite_min = 0, infinite_max = 0, nonzeros = 0;

    for (int item = start; item < end; ++item)
    {
        int position = sorted_positions[item];
        int column = column_indices[position];
        double coefficient = values[position];
        if (coefficient == 0.0) continue;
        ++nonzeros;
        double minimum = selected_bound(
            coefficient, lower_bounds[column], upper_bounds[column], true);
        double maximum = selected_bound(
            coefficient, lower_bounds[column], upper_bounds[column], false);
        if (isfinite(minimum))
        {
            double product = coefficient * minimum;
            finite_min += product;
            absolute_min += fabs(product);
        }
        else
            ++infinite_min;
        if (isfinite(maximum))
        {
            double product = coefficient * maximum;
            finite_max += product;
            absolute_max += fabs(product);
        }
        else
            ++infinite_max;
    }
    if (token == 0U)
    {
        atomic_add_activity(
            row, finite_min, finite_max, absolute_min, absolute_max,
            infinite_min, infinite_max, nonzeros, row_finite_min,
            row_finite_max, row_absolute_min, row_absolute_max,
            row_infinite_min, row_infinite_max, row_nonzeros);
        return;
    }

    int cone = static_cast<int>(token - 1U);
    int type = cone_types[cone];
    bool lower_supported = false, upper_supported = false;
    bool needs_cpu = false;
    if (type == kConeNonnegative)
    {
        lower_supported = true;
        upper_supported = true;
        for (int item = start; item < end; ++item)
        {
            double coefficient = values[sorted_positions[item]];
            lower_supported = lower_supported && coefficient >= 0.0;
            upper_supported = upper_supported && coefficient <= 0.0;
        }
    }
    else if (type == kConeSoc)
    {
        double axis = 0.0, norm_squared = 0.0;
        for (int item = start; item < end; ++item)
        {
            int position = sorted_positions[item];
            int local =
                column_to_cone_position[column_indices[position]];
            double coefficient = values[position];
            if (local == 0)
                axis = coefficient;
            else
                norm_squared += coefficient * coefficient;
        }
        double norm = sqrt(norm_squared);
        double margin =
            256.0 * DBL_EPSILON * fmax(1.0, fmax(fabs(axis), norm));
        lower_supported = axis >= norm + margin || norm == 0.0 && axis >= 0.0;
        upper_supported = -axis >= norm + margin || norm == 0.0 && axis <= 0.0;
    }
    else if (type == kConeRsoc)
    {
        double u = 0.0, v = 0.0, norm_squared = 0.0;
        for (int item = start; item < end; ++item)
        {
            int position = sorted_positions[item];
            int local =
                column_to_cone_position[column_indices[position]];
            double coefficient = values[position];
            if (local == 0)
                u = coefficient;
            else if (local == 1)
                v = coefficient;
            else
                norm_squared += coefficient * coefficient;
        }
        double product = 2.0 * u * v;
        double margin = 512.0 * DBL_EPSILON *
                        fmax(1.0, fmax(fabs(product), norm_squared));
        lower_supported =
            u >= 0.0 && v >= 0.0 &&
            (norm_squared == 0.0 || product >= norm_squared + margin);
        upper_supported =
            u <= 0.0 && v <= 0.0 &&
            (norm_squared == 0.0 || product >= norm_squared + margin);
    }
    else if (type == kConePsd)
    {
        int order = cone_matrix_orders[cone];
        if (order <= 0 || order > 32)
            needs_cpu = true;
        else
        {
            double diagonal[32] = {};
            double off_diagonal_sum[32] = {};
            for (int item = start; item < end; ++item)
            {
                int position = sorted_positions[item];
                int packed =
                    column_to_cone_position[column_indices[position]];
                int matrix_row = 0;
                while ((matrix_row + 1) * (matrix_row + 2) / 2 <= packed)
                    ++matrix_row;
                int matrix_column =
                    packed - matrix_row * (matrix_row + 1) / 2;
                double coefficient = values[position];
                if (matrix_row == matrix_column)
                    diagonal[matrix_row] = coefficient;
                else
                {
                    double entry = fabs(coefficient) / sqrt(2.0);
                    off_diagonal_sum[matrix_row] += entry;
                    off_diagonal_sum[matrix_column] += entry;
                }
            }
            lower_supported = true;
            upper_supported = true;
            for (int i = 0; i < order; ++i)
            {
                double margin = 512.0 * DBL_EPSILON *
                                fmax(1.0, fmax(fabs(diagonal[i]),
                                               off_diagonal_sum[i]));
                lower_supported =
                    lower_supported &&
                    (off_diagonal_sum[i] == 0.0
                         ? diagonal[i] >= 0.0
                         : diagonal[i] >= off_diagonal_sum[i] + margin);
                upper_supported =
                    upper_supported &&
                    (off_diagonal_sum[i] == 0.0
                         ? diagonal[i] <= 0.0
                         : -diagonal[i] >= off_diagonal_sum[i] + margin);
            }
            if (!lower_supported || !upper_supported) needs_cpu = true;
        }
    }
    else if (type == kConeExponential || type == kConePower)
    {
        double coordinate[3] = {};
        for (int item = start; item < end; ++item)
        {
            int position = sorted_positions[item];
            int local =
                column_to_cone_position[column_indices[position]];
            if (local >= 0 && local < 3) coordinate[local] = values[position];
        }
        if (type == kConeExponential)
        {
            lower_supported =
                exp_dual_contains(coordinate[0], coordinate[1], coordinate[2]);
            upper_supported =
                exp_dual_contains(-coordinate[0], -coordinate[1],
                                  -coordinate[2]);
        }
        else
        {
            double alpha = cone_power_alphas[cone];
            lower_supported = power_dual_contains(
                coordinate[0], coordinate[1], coordinate[2], alpha);
            upper_supported = power_dual_contains(
                -coordinate[0], -coordinate[1], -coordinate[2], alpha);
        }
    }
    else
        needs_cpu = true;

    if (needs_cpu) atomicExch(row_needs_cpu + row, 1);
    if (lower_supported)
    {
        if (infinite_min > 0 || finite_min < 0.0)
            atomicExch(row_strengthened + row, 1);
        finite_min = infinite_min == 0 ? fmax(0.0, finite_min) : 0.0;
        absolute_min = fmax(absolute_min, fabs(finite_min));
        infinite_min = 0;
    }
    if (upper_supported)
    {
        if (infinite_max > 0 || finite_max > 0.0)
            atomicExch(row_strengthened + row, 1);
        finite_max = infinite_max == 0 ? fmin(0.0, finite_max) : 0.0;
        absolute_max = fmax(absolute_max, fabs(finite_max));
        infinite_max = 0;
    }
    atomic_add_activity(
        row, finite_min, finite_max, absolute_min, absolute_max, infinite_min,
        infinite_max, nonzeros, row_finite_min, row_finite_max,
        row_absolute_min, row_absolute_max, row_infinite_min,
        row_infinite_max, row_nonzeros);
}

__global__ void classify_cone_activity_kernel(
    size_t rows, const double *finite_min, const double *finite_max,
    const double *absolute_min, const double *absolute_max,
    const int *infinite_min, const int *infinite_max, const int *nonzeros,
    const int *needs_cpu, const int *strengthened,
    const double *constraint_lower, const double *constraint_upper,
    const unsigned char *remove_rows, double feasibility_tolerance,
    unsigned char *flags)
{
    size_t row = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= rows || remove_rows[row])
    {
        if (row < rows) flags[row] = 0;
        return;
    }
    unsigned char state = 0;
    if (needs_cpu[row]) state |= PREFOS_CUDA_ROW_NEEDS_CPU;
    if (strengthened[row]) state |= PREFOS_CUDA_ROW_CONE_STRENGTHENED;
    if (nonzeros[row] == 0)
    {
        flags[row] = state | PREFOS_CUDA_ROW_NEEDS_CPU;
        return;
    }
    double factor =
        (4.0 * static_cast<double>(nonzeros[row]) + 32.0) * DBL_EPSILON;
    if (factor >= 0.5 || !isfinite(finite_min[row]) ||
        !isfinite(finite_max[row]) || !isfinite(absolute_min[row]) ||
        !isfinite(absolute_max[row]))
    {
        flags[row] = state | PREFOS_CUDA_ROW_NEEDS_CPU;
        return;
    }
    factor /= 1.0 - factor;
    double min_error =
        factor * absolute_min[row] + nonzeros[row] * DBL_MIN;
    double max_error =
        factor * absolute_max[row] + nonzeros[row] * DBL_MIN;
    double lower = constraint_lower[row];
    double upper = constraint_upper[row];
    if (isfinite(upper))
    {
        double tolerance = feasibility_tolerance * fmax(1.0, fabs(upper));
        if (infinite_min[row] == 0 &&
            finite_min[row] - min_error > upper + tolerance)
            state |= PREFOS_CUDA_ROW_INFEASIBLE;
        if (infinite_max[row] == 0 &&
            finite_max[row] + max_error <= upper)
            state |= PREFOS_CUDA_ROW_UPPER_REDUNDANT;
    }
    else
        state |= PREFOS_CUDA_ROW_UPPER_REDUNDANT;
    if (isfinite(lower))
    {
        double tolerance = feasibility_tolerance * fmax(1.0, fabs(lower));
        if (infinite_max[row] == 0 &&
            finite_max[row] + max_error < lower - tolerance)
            state |= PREFOS_CUDA_ROW_INFEASIBLE;
        if (infinite_min[row] == 0 &&
            finite_min[row] - min_error >= lower)
            state |= PREFOS_CUDA_ROW_LOWER_REDUNDANT;
    }
    else
        state |= PREFOS_CUDA_ROW_LOWER_REDUNDANT;
    flags[row] = state;
}

} // namespace

extern "C" PreFOSCudaPropagationStatus prefos_cuda_cone_activity_candidates(
    PreFOSCudaWorkspace *context, const double *lower_bounds,
    const double *upper_bounds, const double *constraint_lower,
    const double *constraint_upper, const unsigned char *remove_rows,
    double feasibility_tolerance, unsigned char *row_flags,
    double *milliseconds)
{
    using Clock = std::chrono::steady_clock;
    auto start = Clock::now();
    double *finite_min = nullptr, *finite_max = nullptr;
    double *absolute_min = nullptr, *absolute_max = nullptr;
    int *infinite_min = nullptr, *infinite_max = nullptr, *nonzeros = nullptr;
    int *needs_cpu = nullptr, *strengthened = nullptr;
    unsigned char *device_flags = nullptr;
    unsigned int row_blocks;
    cudaError_t cuda_status = cudaSuccess;
    PreFOSCudaPropagationStatus result;

    if (milliseconds) *milliseconds = 0.0;
    if (!context || !lower_bounds || !upper_bounds || !constraint_lower ||
        !constraint_upper || !remove_rows || !row_flags ||
        !isfinite(feasibility_tolerance) || feasibility_tolerance < 0.0)
        return PREFOS_CUDA_PROPAGATION_ERROR;
    result = build_cone_activity_groups(context);
    if (result != PREFOS_CUDA_PROPAGATION_OK) return result;

#define PREFOS_CUDA_ACTIVITY_CHECK(call)                                    \
    do                                                                       \
    {                                                                        \
        cuda_status = (call);                                                \
        if (cuda_status != cudaSuccess)                                      \
        {                                                                    \
            result = status_from_cuda(cuda_status);                          \
            goto cleanup;                                                    \
        }                                                                    \
    } while (0)

    PREFOS_CUDA_ACTIVITY_CHECK(copy_to_device(
        context->lower_bounds, lower_bounds, context->columns,
        context->stream));
    PREFOS_CUDA_ACTIVITY_CHECK(copy_to_device(
        context->upper_bounds, upper_bounds, context->columns,
        context->stream));
    PREFOS_CUDA_ACTIVITY_CHECK(copy_to_device(
        context->constraint_lower, constraint_lower, context->rows,
        context->stream));
    PREFOS_CUDA_ACTIVITY_CHECK(copy_to_device(
        context->constraint_upper, constraint_upper, context->rows,
        context->stream));
    PREFOS_CUDA_ACTIVITY_CHECK(copy_to_device(
        context->remove_rows, remove_rows, context->rows, context->stream));
    PREFOS_CUDA_ACTIVITY_CHECK(
        allocate_device(&finite_min, context->rows, context->stream));
    PREFOS_CUDA_ACTIVITY_CHECK(
        allocate_device(&finite_max, context->rows, context->stream));
    PREFOS_CUDA_ACTIVITY_CHECK(
        allocate_device(&absolute_min, context->rows, context->stream));
    PREFOS_CUDA_ACTIVITY_CHECK(
        allocate_device(&absolute_max, context->rows, context->stream));
    PREFOS_CUDA_ACTIVITY_CHECK(
        allocate_device(&infinite_min, context->rows, context->stream));
    PREFOS_CUDA_ACTIVITY_CHECK(
        allocate_device(&infinite_max, context->rows, context->stream));
    PREFOS_CUDA_ACTIVITY_CHECK(
        allocate_device(&nonzeros, context->rows, context->stream));
    PREFOS_CUDA_ACTIVITY_CHECK(
        allocate_device(&needs_cpu, context->rows, context->stream));
    PREFOS_CUDA_ACTIVITY_CHECK(
        allocate_device(&strengthened, context->rows, context->stream));
    PREFOS_CUDA_ACTIVITY_CHECK(
        allocate_device(&device_flags, context->rows, context->stream));
    PREFOS_CUDA_ACTIVITY_CHECK(
        cudaMemsetAsync(finite_min, 0, context->rows * sizeof(double),
                        context->stream));
    PREFOS_CUDA_ACTIVITY_CHECK(
        cudaMemsetAsync(finite_max, 0, context->rows * sizeof(double),
                        context->stream));
    PREFOS_CUDA_ACTIVITY_CHECK(
        cudaMemsetAsync(absolute_min, 0, context->rows * sizeof(double),
                        context->stream));
    PREFOS_CUDA_ACTIVITY_CHECK(
        cudaMemsetAsync(absolute_max, 0, context->rows * sizeof(double),
                        context->stream));
    PREFOS_CUDA_ACTIVITY_CHECK(
        cudaMemsetAsync(infinite_min, 0, context->rows * sizeof(int),
                        context->stream));
    PREFOS_CUDA_ACTIVITY_CHECK(
        cudaMemsetAsync(infinite_max, 0, context->rows * sizeof(int),
                        context->stream));
    PREFOS_CUDA_ACTIVITY_CHECK(
        cudaMemsetAsync(nonzeros, 0, context->rows * sizeof(int),
                        context->stream));
    PREFOS_CUDA_ACTIVITY_CHECK(
        cudaMemsetAsync(needs_cpu, 0, context->rows * sizeof(int),
                        context->stream));
    PREFOS_CUDA_ACTIVITY_CHECK(
        cudaMemsetAsync(strengthened, 0, context->rows * sizeof(int),
                        context->stream));
    if (context->n_cone_activity_groups > 0)
    {
        cone_activity_group_kernel<<<static_cast<unsigned int>(
                                         context->n_cone_activity_groups),
                                     32, 0, context->stream>>>(
            context->n_cone_activity_groups,
            context->cone_activity_group_keys,
            context->cone_activity_group_offsets,
            context->cone_activity_sorted_positions, context->column_indices,
            context->values, context->column_to_cone_position,
            context->cone_types, context->cone_matrix_orders,
            context->cone_power_alphas, context->lower_bounds,
            context->upper_bounds, context->remove_rows, finite_min, finite_max,
            absolute_min, absolute_max, infinite_min, infinite_max, nonzeros,
            needs_cpu, strengthened);
        PREFOS_CUDA_ACTIVITY_CHECK(cudaGetLastError());
    }
    row_blocks = static_cast<unsigned int>(
        (context->rows + kThreads - 1) / kThreads);
    if (row_blocks > 0)
    {
        classify_cone_activity_kernel<<<row_blocks, kThreads, 0,
                                        context->stream>>>(
            context->rows, finite_min, finite_max, absolute_min, absolute_max,
            infinite_min, infinite_max, nonzeros, needs_cpu, strengthened,
            context->constraint_lower, context->constraint_upper,
            context->remove_rows, feasibility_tolerance, device_flags);
        PREFOS_CUDA_ACTIVITY_CHECK(cudaGetLastError());
    }
    PREFOS_CUDA_ACTIVITY_CHECK(copy_to_host(
        row_flags, device_flags, context->rows, context->stream));
    PREFOS_CUDA_ACTIVITY_CHECK(cudaStreamSynchronize(context->stream));

cleanup:
    cudaFreeAsync(finite_min, context->stream);
    cudaFreeAsync(finite_max, context->stream);
    cudaFreeAsync(absolute_min, context->stream);
    cudaFreeAsync(absolute_max, context->stream);
    cudaFreeAsync(infinite_min, context->stream);
    cudaFreeAsync(infinite_max, context->stream);
    cudaFreeAsync(nonzeros, context->stream);
    cudaFreeAsync(needs_cpu, context->stream);
    cudaFreeAsync(strengthened, context->stream);
    cudaFreeAsync(device_flags, context->stream);
    (void) cudaStreamSynchronize(context->stream);
    if (milliseconds)
        *milliseconds =
            std::chrono::duration<double, std::milli>(Clock::now() - start)
                .count();
    return result;
#undef PREFOS_CUDA_ACTIVITY_CHECK
}
