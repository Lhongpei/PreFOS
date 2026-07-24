/*
 * Copyright 2026 Hongpei Li
 * SPDX-License-Identifier: Apache-2.0
 */

#include "PREFOS_CudaLinearPropagation.h"
#include "PREFOS_CudaWorkspaceInternal.cuh"

#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_select.cuh>
#include <cuda_runtime.h>

#include <chrono>
#include <climits>
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

void clear_csc(PreFOSCudaWorkspace *context)
{
    cudaFreeAsync(context->csc_column_pointers, context->stream);
    cudaFreeAsync(context->csc_row_indices, context->stream);
    cudaFreeAsync(context->csc_values, context->stream);
    context->csc_column_pointers = nullptr;
    context->csc_row_indices = nullptr;
    context->csc_values = nullptr;
    context->csc_nnz = 0;
    context->csc_ready = 0;
}

__global__ void initialize_csc_sort_inputs_kernel(
    size_t rows, const int *row_pointers, const int *column_indices,
    const double *values, const unsigned char *remove_rows,
    uint64_t *keys, double *sort_values, int *column_counts)
{
    size_t row = static_cast<size_t>(blockIdx.x);
    if (row >= rows) return;
    for (int position = row_pointers[row] + threadIdx.x;
         position < row_pointers[row + 1]; position += blockDim.x)
    {
        int column = column_indices[position];
        double value = values[position];
        if (!remove_rows[row] && value != 0.0)
        {
            keys[position] =
                (static_cast<uint64_t>(static_cast<uint32_t>(column)) << 32U) |
                static_cast<uint32_t>(row);
            sort_values[position] = value;
            atomicAdd(column_counts + column + 1, 1);
        }
        else
        {
            keys[position] = UINT64_MAX;
            sort_values[position] = 0.0;
        }
    }
}

__global__ void materialize_csc_rows_kernel(
    size_t input_nnz, size_t columns, const uint64_t *sorted_keys,
    const int *column_pointers, int *row_indices)
{
    size_t position =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (position >= input_nnz) return;
    int active_nnz = column_pointers[columns];
    if (position < static_cast<size_t>(active_nnz))
        row_indices[position] =
            static_cast<int>(sorted_keys[position] & UINT32_MAX);
}

__global__ void singleton_candidate_flags_kernel(
    size_t columns, const int *column_pointers, const int *row_indices,
    const unsigned char *eligible_columns,
    const unsigned char *dirty_rows, int *column_ids,
    unsigned char *candidate_flags)
{
    size_t column =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (column >= columns) return;
    int start = column_pointers[column];
    int end = column_pointers[column + 1];
    column_ids[column] = static_cast<int>(column);
    candidate_flags[column] =
        static_cast<unsigned char>(
            eligible_columns[column] && end - start == 1 &&
            !dirty_rows[row_indices[start]]);
}

} // namespace

extern "C" PreFOSCudaPropagationStatus prefos_cuda_workspace_build_csc(
    PreFOSCudaWorkspace *context, const unsigned char *remove_rows,
    int *column_pointers, size_t *active_nnz, double *milliseconds)
{
    using Clock = std::chrono::steady_clock;
    auto start = Clock::now();
    uint64_t *keys_in = nullptr, *keys_out = nullptr;
    double *values_in = nullptr;
    void *sort_storage = nullptr, *scan_storage = nullptr;
    size_t sort_storage_bytes = 0, scan_storage_bytes = 0;
    cudaError_t cuda_status = cudaSuccess;
    PreFOSCudaPropagationStatus result = PREFOS_CUDA_PROPAGATION_OK;

    if (active_nnz) *active_nnz = 0;
    if (milliseconds) *milliseconds = 0.0;
    if (!context || !active_nnz ||
        (context->rows > 0 && !remove_rows) ||
        (context->columns > 0 && !column_pointers) ||
        context->rows > static_cast<size_t>(INT_MAX) ||
        context->columns > static_cast<size_t>(INT_MAX) ||
        context->nnz > static_cast<size_t>(INT_MAX))
        return PREFOS_CUDA_PROPAGATION_ERROR;

#define PREFOS_CUDA_CSC_CHECK(call)                                        \
    do                                                                     \
    {                                                                      \
        cuda_status = (call);                                              \
        if (cuda_status != cudaSuccess)                                    \
        {                                                                  \
            result = status_from_cuda(cuda_status);                        \
            goto cleanup;                                                  \
        }                                                                  \
    } while (0)

    clear_csc(context);
    PREFOS_CUDA_CSC_CHECK(copy_to_device(
        context->remove_rows, remove_rows, context->rows,
        context->stream));
    PREFOS_CUDA_CSC_CHECK(allocate_device(
        &context->csc_column_pointers, context->columns + 1,
        context->stream));
    PREFOS_CUDA_CSC_CHECK(allocate_device(
        &context->csc_row_indices, context->nnz, context->stream));
    PREFOS_CUDA_CSC_CHECK(allocate_device(
        &context->csc_values, context->nnz, context->stream));
    PREFOS_CUDA_CSC_CHECK(
        allocate_device(&keys_in, context->nnz, context->stream));
    PREFOS_CUDA_CSC_CHECK(
        allocate_device(&keys_out, context->nnz, context->stream));
    PREFOS_CUDA_CSC_CHECK(
        allocate_device(&values_in, context->nnz, context->stream));
    PREFOS_CUDA_CSC_CHECK(cudaMemsetAsync(
        context->csc_column_pointers, 0,
        (context->columns + 1) * sizeof(int), context->stream));
    if (context->rows > 0)
    {
        initialize_csc_sort_inputs_kernel<<<
            static_cast<unsigned int>(context->rows), kThreads, 0,
            context->stream>>>(
            context->rows, context->row_pointers,
            context->column_indices, context->values,
            context->remove_rows, keys_in, values_in,
            context->csc_column_pointers);
        PREFOS_CUDA_CSC_CHECK(cudaGetLastError());
    }
    PREFOS_CUDA_CSC_CHECK(cub::DeviceScan::InclusiveSum(
        nullptr, scan_storage_bytes, context->csc_column_pointers,
        context->csc_column_pointers,
        static_cast<int>(context->columns + 1), context->stream));
    PREFOS_CUDA_CSC_CHECK(
        cudaMallocAsync(&scan_storage, scan_storage_bytes,
                        context->stream));
    PREFOS_CUDA_CSC_CHECK(cub::DeviceScan::InclusiveSum(
        scan_storage, scan_storage_bytes, context->csc_column_pointers,
        context->csc_column_pointers,
        static_cast<int>(context->columns + 1), context->stream));
    if (context->nnz > 0)
    {
        PREFOS_CUDA_CSC_CHECK(cub::DeviceRadixSort::SortPairs(
            nullptr, sort_storage_bytes, keys_in, keys_out, values_in,
            context->csc_values, static_cast<int>(context->nnz), 0, 64,
            context->stream));
        PREFOS_CUDA_CSC_CHECK(
            cudaMallocAsync(&sort_storage, sort_storage_bytes,
                            context->stream));
        PREFOS_CUDA_CSC_CHECK(cub::DeviceRadixSort::SortPairs(
            sort_storage, sort_storage_bytes, keys_in, keys_out, values_in,
            context->csc_values, static_cast<int>(context->nnz), 0, 64,
            context->stream));
        {
            unsigned int blocks = static_cast<unsigned int>(
                (context->nnz + kThreads - 1) / kThreads);
            materialize_csc_rows_kernel<<<blocks, kThreads, 0,
                                          context->stream>>>(
                context->nnz, context->columns, keys_out,
                context->csc_column_pointers,
                context->csc_row_indices);
            PREFOS_CUDA_CSC_CHECK(cudaGetLastError());
        }
    }
    PREFOS_CUDA_CSC_CHECK(copy_to_host(
        column_pointers, context->csc_column_pointers,
        context->columns + 1, context->stream));
    PREFOS_CUDA_CSC_CHECK(cudaStreamSynchronize(context->stream));
    if (column_pointers[context->columns] < 0 ||
        static_cast<size_t>(column_pointers[context->columns]) >
            context->nnz)
    {
        result = PREFOS_CUDA_PROPAGATION_ERROR;
        goto cleanup;
    }
    context->csc_nnz =
        static_cast<size_t>(column_pointers[context->columns]);
    context->csc_ready = 1;
    *active_nnz = context->csc_nnz;

cleanup:
    cudaFreeAsync(keys_in, context->stream);
    cudaFreeAsync(keys_out, context->stream);
    cudaFreeAsync(values_in, context->stream);
    cudaFreeAsync(sort_storage, context->stream);
    cudaFreeAsync(scan_storage, context->stream);
    if (result != PREFOS_CUDA_PROPAGATION_OK) clear_csc(context);
    (void) cudaStreamSynchronize(context->stream);
    if (milliseconds)
        *milliseconds =
            std::chrono::duration<double, std::milli>(Clock::now() - start)
                .count();
    return result;
#undef PREFOS_CUDA_CSC_CHECK
}

extern "C" PreFOSCudaPropagationStatus prefos_cuda_workspace_copy_csc(
    PreFOSCudaWorkspace *context, int *row_indices, double *values,
    double *milliseconds)
{
    using Clock = std::chrono::steady_clock;
    auto start = Clock::now();
    cudaError_t cuda_status;
    if (milliseconds) *milliseconds = 0.0;
    if (!context || !context->csc_ready ||
        (context->csc_nnz > 0 && (!row_indices || !values)))
        return PREFOS_CUDA_PROPAGATION_ERROR;
    cuda_status = copy_to_host(
        row_indices, context->csc_row_indices, context->csc_nnz,
        context->stream);
    if (cuda_status == cudaSuccess)
        cuda_status = copy_to_host(
            values, context->csc_values, context->csc_nnz,
            context->stream);
    if (cuda_status == cudaSuccess)
        cuda_status = cudaStreamSynchronize(context->stream);
    if (milliseconds)
        *milliseconds =
            std::chrono::duration<double, std::milli>(Clock::now() - start)
                .count();
    return status_from_cuda(cuda_status);
}

extern "C" PreFOSCudaPropagationStatus
prefos_cuda_singleton_column_candidates(
    PreFOSCudaWorkspace *context,
    const unsigned char *eligible_columns,
    const unsigned char *dirty_rows, int *candidate_columns,
    size_t *candidate_count, double *milliseconds)
{
    using Clock = std::chrono::steady_clock;
    auto start = Clock::now();
    unsigned char *device_eligible = nullptr, *device_dirty = nullptr;
    unsigned char *device_flags = nullptr;
    int *device_ids = nullptr, *device_candidates = nullptr;
    int *device_count = nullptr;
    void *select_storage = nullptr;
    size_t select_storage_bytes = 0;
    int host_count = 0;
    cudaError_t cuda_status = cudaSuccess;
    PreFOSCudaPropagationStatus result = PREFOS_CUDA_PROPAGATION_OK;

    if (candidate_count) *candidate_count = 0;
    if (milliseconds) *milliseconds = 0.0;
    if (!context || !context->csc_ready || !candidate_count ||
        (context->columns > 0 &&
         (!eligible_columns || !candidate_columns)) ||
        (context->rows > 0 && !dirty_rows) ||
        context->columns > static_cast<size_t>(INT_MAX))
        return PREFOS_CUDA_PROPAGATION_ERROR;

#define PREFOS_CUDA_SINGLETON_CHECK(call)                                  \
    do                                                                     \
    {                                                                      \
        cuda_status = (call);                                              \
        if (cuda_status != cudaSuccess)                                    \
        {                                                                  \
            result = status_from_cuda(cuda_status);                        \
            goto cleanup;                                                  \
        }                                                                  \
    } while (0)

    PREFOS_CUDA_SINGLETON_CHECK(allocate_device(
        &device_eligible, context->columns, context->stream));
    PREFOS_CUDA_SINGLETON_CHECK(allocate_device(
        &device_dirty, context->rows, context->stream));
    PREFOS_CUDA_SINGLETON_CHECK(allocate_device(
        &device_flags, context->columns, context->stream));
    PREFOS_CUDA_SINGLETON_CHECK(
        allocate_device(&device_ids, context->columns, context->stream));
    PREFOS_CUDA_SINGLETON_CHECK(allocate_device(
        &device_candidates, context->columns, context->stream));
    PREFOS_CUDA_SINGLETON_CHECK(
        allocate_device(&device_count, 1, context->stream));
    PREFOS_CUDA_SINGLETON_CHECK(copy_to_device(
        device_eligible, eligible_columns, context->columns,
        context->stream));
    PREFOS_CUDA_SINGLETON_CHECK(copy_to_device(
        device_dirty, dirty_rows, context->rows, context->stream));
    if (context->columns > 0)
    {
        unsigned int blocks = static_cast<unsigned int>(
            (context->columns + kThreads - 1) / kThreads);
        singleton_candidate_flags_kernel<<<blocks, kThreads, 0,
                                            context->stream>>>(
            context->columns, context->csc_column_pointers,
            context->csc_row_indices, device_eligible, device_dirty,
            device_ids, device_flags);
        PREFOS_CUDA_SINGLETON_CHECK(cudaGetLastError());
        PREFOS_CUDA_SINGLETON_CHECK(cub::DeviceSelect::Flagged(
            nullptr, select_storage_bytes, device_ids, device_flags,
            device_candidates, device_count,
            static_cast<int>(context->columns), context->stream));
        PREFOS_CUDA_SINGLETON_CHECK(
            cudaMallocAsync(&select_storage, select_storage_bytes,
                            context->stream));
        PREFOS_CUDA_SINGLETON_CHECK(cub::DeviceSelect::Flagged(
            select_storage, select_storage_bytes, device_ids, device_flags,
            device_candidates, device_count,
            static_cast<int>(context->columns), context->stream));
    }
    else
        PREFOS_CUDA_SINGLETON_CHECK(
            cudaMemsetAsync(device_count, 0, sizeof(int),
                            context->stream));
    PREFOS_CUDA_SINGLETON_CHECK(copy_to_host(
        &host_count, device_count, 1, context->stream));
    PREFOS_CUDA_SINGLETON_CHECK(cudaStreamSynchronize(context->stream));
    if (host_count < 0 ||
        static_cast<size_t>(host_count) > context->columns)
    {
        result = PREFOS_CUDA_PROPAGATION_ERROR;
        goto cleanup;
    }
    PREFOS_CUDA_SINGLETON_CHECK(copy_to_host(
        candidate_columns, device_candidates,
        static_cast<size_t>(host_count), context->stream));
    PREFOS_CUDA_SINGLETON_CHECK(cudaStreamSynchronize(context->stream));
    *candidate_count = static_cast<size_t>(host_count);

cleanup:
    cudaFreeAsync(device_eligible, context->stream);
    cudaFreeAsync(device_dirty, context->stream);
    cudaFreeAsync(device_flags, context->stream);
    cudaFreeAsync(device_ids, context->stream);
    cudaFreeAsync(device_candidates, context->stream);
    cudaFreeAsync(device_count, context->stream);
    cudaFreeAsync(select_storage, context->stream);
    (void) cudaStreamSynchronize(context->stream);
    if (milliseconds)
        *milliseconds =
            std::chrono::duration<double, std::milli>(Clock::now() - start)
                .count();
    return result;
#undef PREFOS_CUDA_SINGLETON_CHECK
}
