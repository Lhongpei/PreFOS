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

__global__ void parallel_row_hash_kernel(
    size_t rows, const int *row_pointers, const int *column_indices,
    const double *values, const unsigned char *remove_rows,
    int *support_hashes, int *coefficient_hashes, uint64_t *keys,
    int *row_ids, int *active_rows)
{
    size_t row = static_cast<size_t>(blockIdx.x);
    if (row >= rows || threadIdx.x != 0) return;
    row_ids[row] = static_cast<int>(row);
    if (remove_rows[row])
    {
        support_hashes[row] = INT_MAX;
        coefficient_hashes[row] = INT_MAX;
        keys[row] = UINT64_MAX;
        return;
    }
    int start = row_pointers[row];
    int end = row_pointers[row + 1];
    if (end <= start || values[start] == 0.0)
    {
        support_hashes[row] = INT_MAX;
        coefficient_hashes[row] = INT_MAX;
        keys[row] = UINT64_MAX;
        return;
    }
    uint32_t support = 5381U;
    uint32_t coefficient = 5381U;
    double maximum = fabs(values[start]);
    for (int position = start; position < end; ++position)
    {
        support = ((support << 5U) + support) +
                  static_cast<uint32_t>(column_indices[position]);
        maximum = fmax(maximum, fabs(values[position]));
    }
    double scale = values[start] > 0.0 ? 1.0 / maximum : -1.0 / maximum;
    for (int position = start; position < end; ++position)
    {
        uint32_t normalized = static_cast<uint32_t>(
            round(values[position] * scale * 1e6));
        coefficient = ((coefficient << 5U) + coefficient) + normalized;
    }
    support_hashes[row] = static_cast<int>(support);
    coefficient_hashes[row] = static_cast<int>(coefficient);
    uint64_t ordered_support =
        static_cast<uint64_t>(support ^ UINT32_C(0x80000000));
    uint64_t ordered_coefficient =
        static_cast<uint64_t>(coefficient ^ UINT32_C(0x80000000));
    keys[row] = (ordered_support << 32U) | ordered_coefficient;
    atomicAdd(active_rows, 1);
}

} // namespace

extern "C" PreFOSCudaPropagationStatus prefos_cuda_parallel_row_hash_sort(
    PreFOSCudaWorkspace *context, const unsigned char *remove_rows,
    int *sorted_rows, int *support_hashes, int *coefficient_hashes,
    size_t *active_rows, double *milliseconds)
{
    using Clock = std::chrono::steady_clock;
    auto start = Clock::now();
    uint64_t *keys_in = nullptr, *keys_out = nullptr;
    int *rows_in = nullptr, *rows_out = nullptr;
    int *device_support = nullptr, *device_coefficient = nullptr;
    int *device_active = nullptr;
    void *sort_storage = nullptr;
    size_t sort_storage_bytes = 0;
    int host_active = 0;
    cudaError_t cuda_status = cudaSuccess;
    PreFOSCudaPropagationStatus result = PREFOS_CUDA_PROPAGATION_OK;

    if (active_rows) *active_rows = 0;
    if (milliseconds) *milliseconds = 0.0;
    if (!context || !remove_rows || !sorted_rows || !support_hashes ||
        !coefficient_hashes || !active_rows ||
        context->rows > static_cast<size_t>(INT_MAX) ||
        context->rows > static_cast<size_t>(UINT_MAX))
        return PREFOS_CUDA_PROPAGATION_ERROR;

#define PREFOS_CUDA_PARALLEL_CHECK(call)                                   \
    do                                                                      \
    {                                                                       \
        cuda_status = (call);                                               \
        if (cuda_status != cudaSuccess)                                     \
        {                                                                   \
            result = status_from_cuda(cuda_status);                         \
            goto cleanup;                                                   \
        }                                                                   \
    } while (0)

    PREFOS_CUDA_PARALLEL_CHECK(copy_to_device(
        context->remove_rows, remove_rows, context->rows, context->stream));
    PREFOS_CUDA_PARALLEL_CHECK(
        allocate_device(&keys_in, context->rows, context->stream));
    PREFOS_CUDA_PARALLEL_CHECK(
        allocate_device(&keys_out, context->rows, context->stream));
    PREFOS_CUDA_PARALLEL_CHECK(
        allocate_device(&rows_in, context->rows, context->stream));
    PREFOS_CUDA_PARALLEL_CHECK(
        allocate_device(&rows_out, context->rows, context->stream));
    PREFOS_CUDA_PARALLEL_CHECK(
        allocate_device(&device_support, context->rows, context->stream));
    PREFOS_CUDA_PARALLEL_CHECK(
        allocate_device(&device_coefficient, context->rows, context->stream));
    PREFOS_CUDA_PARALLEL_CHECK(
        allocate_device(&device_active, 1, context->stream));
    PREFOS_CUDA_PARALLEL_CHECK(
        cudaMemsetAsync(device_active, 0, sizeof(int), context->stream));
    if (context->rows > 0)
    {
        parallel_row_hash_kernel<<<static_cast<unsigned int>(context->rows),
                                   kThreads, 0, context->stream>>>(
            context->rows, context->row_pointers, context->column_indices,
            context->values, context->remove_rows, device_support,
            device_coefficient, keys_in, rows_in, device_active);
        PREFOS_CUDA_PARALLEL_CHECK(cudaGetLastError());
        PREFOS_CUDA_PARALLEL_CHECK(cub::DeviceRadixSort::SortPairs(
            nullptr, sort_storage_bytes, keys_in, keys_out, rows_in, rows_out,
            static_cast<int>(context->rows), 0, 64, context->stream));
        PREFOS_CUDA_PARALLEL_CHECK(
            cudaMallocAsync(&sort_storage, sort_storage_bytes,
                            context->stream));
        PREFOS_CUDA_PARALLEL_CHECK(cub::DeviceRadixSort::SortPairs(
            sort_storage, sort_storage_bytes, keys_in, keys_out, rows_in,
            rows_out, static_cast<int>(context->rows), 0, 64,
            context->stream));
    }
    PREFOS_CUDA_PARALLEL_CHECK(
        copy_to_host(&host_active, device_active, 1, context->stream));
    PREFOS_CUDA_PARALLEL_CHECK(copy_to_host(
        support_hashes, device_support, context->rows, context->stream));
    PREFOS_CUDA_PARALLEL_CHECK(copy_to_host(
        coefficient_hashes, device_coefficient, context->rows,
        context->stream));
    PREFOS_CUDA_PARALLEL_CHECK(cudaStreamSynchronize(context->stream));
    if (host_active < 0 || static_cast<size_t>(host_active) > context->rows)
    {
        result = PREFOS_CUDA_PROPAGATION_ERROR;
        goto cleanup;
    }
    PREFOS_CUDA_PARALLEL_CHECK(copy_to_host(
        sorted_rows, rows_out, static_cast<size_t>(host_active),
        context->stream));
    PREFOS_CUDA_PARALLEL_CHECK(cudaStreamSynchronize(context->stream));
    *active_rows = static_cast<size_t>(host_active);

cleanup:
    cudaFreeAsync(keys_in, context->stream);
    cudaFreeAsync(keys_out, context->stream);
    cudaFreeAsync(rows_in, context->stream);
    cudaFreeAsync(rows_out, context->stream);
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
#undef PREFOS_CUDA_PARALLEL_CHECK
}
