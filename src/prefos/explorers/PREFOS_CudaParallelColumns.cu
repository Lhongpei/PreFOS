/*
 * Copyright 2026 Hongpei Li
 * SPDX-License-Identifier: Apache-2.0
 */

#include "PREFOS_CudaLinearPropagation.h"
#include "PREFOS_CudaWorkspaceInternal.cuh"

#include <cub/device/device_radix_sort.cuh>
#include <cuda_runtime.h>

#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>

namespace
{

constexpr int kThreads = 256;

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

__global__ void parallel_column_hash_kernel(
    size_t columns, const int *column_pointers, const int *row_indices,
    const double *values, const unsigned char *eligible_columns,
    const unsigned char *dirty_rows, int *support_hashes,
    int *coefficient_hashes, uint64_t *keys, int *column_ids,
    int *active_columns)
{
    size_t column =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (column >= columns) return;
    column_ids[column] = static_cast<int>(column);
    int start = column_pointers[column];
    int end = column_pointers[column + 1];
    if (!eligible_columns[column] || end <= start ||
        values[start] == 0.0)
    {
        support_hashes[column] = INT_MAX;
        coefficient_hashes[column] = INT_MAX;
        keys[column] = UINT64_MAX;
        return;
    }
    for (int position = start; position < end; ++position)
    {
        if (dirty_rows[row_indices[position]])
        {
            support_hashes[column] = INT_MAX;
            coefficient_hashes[column] = INT_MAX;
            keys[column] = UINT64_MAX;
            return;
        }
    }
    uint32_t support = 5381U;
    uint32_t coefficient = 5381U;
    double maximum = fabs(values[start]);
    for (int position = start; position < end; ++position)
    {
        support = ((support << 5U) + support) +
                  static_cast<uint32_t>(row_indices[position]);
        maximum = fmax(maximum, fabs(values[position]));
    }
    double scale = values[start] > 0.0 ? 1.0 / maximum : -1.0 / maximum;
    for (int position = start; position < end; ++position)
    {
        uint32_t normalized = static_cast<uint32_t>(
            round(values[position] * scale * 1e6));
        coefficient = ((coefficient << 5U) + coefficient) + normalized;
    }
    support_hashes[column] = static_cast<int>(support);
    coefficient_hashes[column] = static_cast<int>(coefficient);
    uint64_t ordered_support =
        static_cast<uint64_t>(support ^ UINT32_C(0x80000000));
    uint64_t ordered_coefficient =
        static_cast<uint64_t>(coefficient ^ UINT32_C(0x80000000));
    keys[column] = (ordered_support << 32U) | ordered_coefficient;
    if (keys[column] == UINT64_MAX) --keys[column];
    atomicAdd(active_columns, 1);
}

} // namespace

extern "C" PreFOSCudaPropagationStatus
prefos_cuda_parallel_column_hash_sort(
    PreFOSCudaWorkspace *context,
    const unsigned char *eligible_columns,
    const unsigned char *dirty_rows, int *sorted_columns,
    int *support_hashes, int *coefficient_hashes,
    size_t *active_columns, double *milliseconds)
{
    using Clock = std::chrono::steady_clock;
    auto start = Clock::now();
    unsigned char *device_eligible = nullptr, *device_dirty = nullptr;
    uint64_t *keys_in = nullptr, *keys_out = nullptr;
    int *columns_in = nullptr, *columns_out = nullptr;
    int *device_support = nullptr, *device_coefficient = nullptr;
    int *device_active = nullptr;
    void *sort_storage = nullptr;
    size_t sort_storage_bytes = 0;
    int host_active = 0;
    cudaError_t cuda_status = cudaSuccess;
    PreFOSCudaPropagationStatus result = PREFOS_CUDA_PROPAGATION_OK;

    if (active_columns) *active_columns = 0;
    if (milliseconds) *milliseconds = 0.0;
    if (!context || !context->csc_ready || !active_columns ||
        (context->columns > 0 &&
         (!eligible_columns || !sorted_columns || !support_hashes ||
          !coefficient_hashes)) ||
        (context->rows > 0 && !dirty_rows) ||
        context->columns > static_cast<size_t>(INT_MAX))
        return PREFOS_CUDA_PROPAGATION_ERROR;

#define PREFOS_CUDA_PARALLEL_COLUMN_CHECK(call)                            \
    do                                                                     \
    {                                                                      \
        cuda_status = (call);                                              \
        if (cuda_status != cudaSuccess)                                    \
        {                                                                  \
            result = status_from_cuda(cuda_status);                        \
            goto cleanup;                                                  \
        }                                                                  \
    } while (0)

    PREFOS_CUDA_PARALLEL_COLUMN_CHECK(allocate_device(
        &device_eligible, context->columns, context->stream));
    PREFOS_CUDA_PARALLEL_COLUMN_CHECK(allocate_device(
        &device_dirty, context->rows, context->stream));
    PREFOS_CUDA_PARALLEL_COLUMN_CHECK(
        allocate_device(&keys_in, context->columns, context->stream));
    PREFOS_CUDA_PARALLEL_COLUMN_CHECK(
        allocate_device(&keys_out, context->columns, context->stream));
    PREFOS_CUDA_PARALLEL_COLUMN_CHECK(
        allocate_device(&columns_in, context->columns, context->stream));
    PREFOS_CUDA_PARALLEL_COLUMN_CHECK(
        allocate_device(&columns_out, context->columns, context->stream));
    PREFOS_CUDA_PARALLEL_COLUMN_CHECK(allocate_device(
        &device_support, context->columns, context->stream));
    PREFOS_CUDA_PARALLEL_COLUMN_CHECK(allocate_device(
        &device_coefficient, context->columns, context->stream));
    PREFOS_CUDA_PARALLEL_COLUMN_CHECK(
        allocate_device(&device_active, 1, context->stream));
    PREFOS_CUDA_PARALLEL_COLUMN_CHECK(copy_to_device(
        device_eligible, eligible_columns, context->columns,
        context->stream));
    PREFOS_CUDA_PARALLEL_COLUMN_CHECK(copy_to_device(
        device_dirty, dirty_rows, context->rows, context->stream));
    PREFOS_CUDA_PARALLEL_COLUMN_CHECK(cudaMemsetAsync(
        device_active, 0, sizeof(int), context->stream));
    if (context->columns > 0)
    {
        unsigned int blocks = static_cast<unsigned int>(
            (context->columns + kThreads - 1) / kThreads);
        parallel_column_hash_kernel<<<blocks, kThreads, 0,
                                      context->stream>>>(
            context->columns, context->csc_column_pointers,
            context->csc_row_indices, context->csc_values,
            device_eligible, device_dirty, device_support,
            device_coefficient, keys_in, columns_in, device_active);
        PREFOS_CUDA_PARALLEL_COLUMN_CHECK(cudaGetLastError());
        PREFOS_CUDA_PARALLEL_COLUMN_CHECK(
            cub::DeviceRadixSort::SortPairs(
                nullptr, sort_storage_bytes, keys_in, keys_out, columns_in,
                columns_out, static_cast<int>(context->columns), 0, 64,
                context->stream));
        PREFOS_CUDA_PARALLEL_COLUMN_CHECK(
            cudaMallocAsync(&sort_storage, sort_storage_bytes,
                            context->stream));
        PREFOS_CUDA_PARALLEL_COLUMN_CHECK(
            cub::DeviceRadixSort::SortPairs(
                sort_storage, sort_storage_bytes, keys_in, keys_out,
                columns_in, columns_out,
                static_cast<int>(context->columns), 0, 64,
                context->stream));
    }
    PREFOS_CUDA_PARALLEL_COLUMN_CHECK(copy_to_host(
        &host_active, device_active, 1, context->stream));
    PREFOS_CUDA_PARALLEL_COLUMN_CHECK(copy_to_host(
        support_hashes, device_support, context->columns,
        context->stream));
    PREFOS_CUDA_PARALLEL_COLUMN_CHECK(copy_to_host(
        coefficient_hashes, device_coefficient, context->columns,
        context->stream));
    PREFOS_CUDA_PARALLEL_COLUMN_CHECK(
        cudaStreamSynchronize(context->stream));
    if (host_active < 0 ||
        static_cast<size_t>(host_active) > context->columns)
    {
        result = PREFOS_CUDA_PROPAGATION_ERROR;
        goto cleanup;
    }
    PREFOS_CUDA_PARALLEL_COLUMN_CHECK(copy_to_host(
        sorted_columns, columns_out, static_cast<size_t>(host_active),
        context->stream));
    PREFOS_CUDA_PARALLEL_COLUMN_CHECK(
        cudaStreamSynchronize(context->stream));
    *active_columns = static_cast<size_t>(host_active);

cleanup:
    cudaFreeAsync(device_eligible, context->stream);
    cudaFreeAsync(device_dirty, context->stream);
    cudaFreeAsync(keys_in, context->stream);
    cudaFreeAsync(keys_out, context->stream);
    cudaFreeAsync(columns_in, context->stream);
    cudaFreeAsync(columns_out, context->stream);
    cudaFreeAsync(device_support, context->stream);
    cudaFreeAsync(device_coefficient, context->stream);
    cudaFreeAsync(device_active, context->stream);
    cudaFreeAsync(sort_storage, context->stream);
    (void) cudaStreamSynchronize(context->stream);
    if (milliseconds)
        *milliseconds =
            std::chrono::duration<double, std::milli>(Clock::now() - start)
                .count();
    return result;
#undef PREFOS_CUDA_PARALLEL_COLUMN_CHECK
}
