/*
 * Copyright 2026 Hongpei Li
 * SPDX-License-Identifier: Apache-2.0
 */

#include "PREFOS_CudaLinearPropagation.h"
#include "PREFOS_CudaWorkspaceInternal.cuh"

#include <cuda_runtime.h>

#include <chrono>
#include <climits>
#include <cmath>

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

__global__ void compact_a_analyze_kernel(
    size_t rows, const int *row_pointers, const int *column_indices,
    const double *values, const unsigned char *remove_rows,
    const unsigned char *is_fixed, const double *fixed_values,
    const int *column_map, int *row_nnz, double *row_shifts,
    unsigned char *row_needs_exact_shift)
{
    size_t row =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    int count = 0;
    double shift = 0.0;
    unsigned char exact_shift = 0;
    if (!remove_rows[row])
    {
        for (int position = row_pointers[row];
             position < row_pointers[row + 1]; ++position)
        {
            int column = column_indices[position];
            double value = values[position];
            if (is_fixed[column])
            {
                shift += value * fixed_values[column];
                exact_shift = 1;
            }
            else if (column_map[column] >= 0 && value != 0.0)
                ++count;
        }
    }
    row_nnz[row] = count;
    row_shifts[row] = shift;
    row_needs_exact_shift[row] = exact_shift;
}

__global__ void compact_a_write_kernel(
    size_t rows, const int *source_row_pointers,
    const int *source_columns, const double *source_values,
    const int *column_map, const int *row_map,
    const int *output_row_pointers, int *output_columns,
    double *output_values, int *error)
{
    size_t row =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    int mapped_row = row_map[row];
    if (mapped_row < 0) return;
    int write = output_row_pointers[mapped_row];
    int expected_end = output_row_pointers[mapped_row + 1];
    for (int position = source_row_pointers[row];
         position < source_row_pointers[row + 1]; ++position)
    {
        int mapped_column = column_map[source_columns[position]];
        if (mapped_column < 0 || source_values[position] == 0.0) continue;
        if (write >= expected_end)
        {
            atomicExch(error, 1);
            return;
        }
        output_columns[write] = mapped_column;
        output_values[write] = source_values[position];
        ++write;
    }
    if (write != expected_end) atomicExch(error, 1);
}

} // namespace

extern "C" PreFOSCudaPropagationStatus prefos_cuda_compact_a_analyze(
    PreFOSCudaWorkspace *context, const unsigned char *remove_rows,
    const unsigned char *is_fixed, const double *fixed_values,
    const int *column_map, int *row_nnz, double *row_shifts,
    unsigned char *row_needs_exact_shift, double *milliseconds)
{
    using Clock = std::chrono::steady_clock;
    auto start = Clock::now();
    unsigned char *device_is_fixed = nullptr;
    double *device_fixed_values = nullptr;
    int *device_column_map = nullptr, *device_row_nnz = nullptr;
    double *device_row_shifts = nullptr;
    unsigned char *device_exact_shift = nullptr;
    cudaError_t cuda_status = cudaSuccess;
    PreFOSCudaPropagationStatus result = PREFOS_CUDA_PROPAGATION_OK;

    if (milliseconds) *milliseconds = 0.0;
    if (!context || !remove_rows || !is_fixed || !fixed_values ||
        !column_map || !row_nnz || !row_shifts ||
        !row_needs_exact_shift ||
        context->rows > static_cast<size_t>(UINT_MAX))
        return PREFOS_CUDA_PROPAGATION_ERROR;

#define PREFOS_CUDA_COMPACT_ANALYZE_CHECK(call)                            \
    do                                                                     \
    {                                                                      \
        cuda_status = (call);                                              \
        if (cuda_status != cudaSuccess)                                    \
        {                                                                  \
            result = status_from_cuda(cuda_status);                        \
            goto cleanup;                                                  \
        }                                                                  \
    } while (0)

    PREFOS_CUDA_COMPACT_ANALYZE_CHECK(copy_to_device(
        context->remove_rows, remove_rows, context->rows,
        context->stream));
    PREFOS_CUDA_COMPACT_ANALYZE_CHECK(allocate_device(
        &device_is_fixed, context->columns, context->stream));
    PREFOS_CUDA_COMPACT_ANALYZE_CHECK(allocate_device(
        &device_fixed_values, context->columns, context->stream));
    PREFOS_CUDA_COMPACT_ANALYZE_CHECK(allocate_device(
        &device_column_map, context->columns, context->stream));
    PREFOS_CUDA_COMPACT_ANALYZE_CHECK(
        allocate_device(&device_row_nnz, context->rows, context->stream));
    PREFOS_CUDA_COMPACT_ANALYZE_CHECK(allocate_device(
        &device_row_shifts, context->rows, context->stream));
    PREFOS_CUDA_COMPACT_ANALYZE_CHECK(allocate_device(
        &device_exact_shift, context->rows, context->stream));
    PREFOS_CUDA_COMPACT_ANALYZE_CHECK(copy_to_device(
        device_is_fixed, is_fixed, context->columns, context->stream));
    PREFOS_CUDA_COMPACT_ANALYZE_CHECK(copy_to_device(
        device_fixed_values, fixed_values, context->columns,
        context->stream));
    PREFOS_CUDA_COMPACT_ANALYZE_CHECK(copy_to_device(
        device_column_map, column_map, context->columns,
        context->stream));
    if (context->rows > 0)
    {
        unsigned int blocks = static_cast<unsigned int>(
            (context->rows + kThreads - 1) / kThreads);
        compact_a_analyze_kernel<<<blocks, kThreads, 0, context->stream>>>(
            context->rows, context->row_pointers,
            context->column_indices, context->values,
            context->remove_rows, device_is_fixed, device_fixed_values,
            device_column_map, device_row_nnz, device_row_shifts,
            device_exact_shift);
        PREFOS_CUDA_COMPACT_ANALYZE_CHECK(cudaGetLastError());
    }
    PREFOS_CUDA_COMPACT_ANALYZE_CHECK(copy_to_host(
        row_nnz, device_row_nnz, context->rows, context->stream));
    PREFOS_CUDA_COMPACT_ANALYZE_CHECK(copy_to_host(
        row_shifts, device_row_shifts, context->rows, context->stream));
    PREFOS_CUDA_COMPACT_ANALYZE_CHECK(copy_to_host(
        row_needs_exact_shift, device_exact_shift, context->rows,
        context->stream));
    PREFOS_CUDA_COMPACT_ANALYZE_CHECK(
        cudaStreamSynchronize(context->stream));

cleanup:
    cudaFreeAsync(device_is_fixed, context->stream);
    cudaFreeAsync(device_fixed_values, context->stream);
    cudaFreeAsync(device_column_map, context->stream);
    cudaFreeAsync(device_row_nnz, context->stream);
    cudaFreeAsync(device_row_shifts, context->stream);
    cudaFreeAsync(device_exact_shift, context->stream);
    (void) cudaStreamSynchronize(context->stream);
    if (milliseconds)
        *milliseconds =
            std::chrono::duration<double, std::milli>(Clock::now() - start)
                .count();
    return result;
#undef PREFOS_CUDA_COMPACT_ANALYZE_CHECK
}

extern "C" PreFOSCudaPropagationStatus prefos_cuda_compact_a_write(
    PreFOSCudaWorkspace *context, const int *column_map,
    const int *row_map, const int *output_row_pointers,
    size_t output_rows, size_t output_nnz, int *output_columns,
    double *output_values, double *milliseconds)
{
    using Clock = std::chrono::steady_clock;
    auto start = Clock::now();
    int *device_column_map = nullptr, *device_row_map = nullptr;
    int *device_output_rows = nullptr, *device_output_columns = nullptr;
    double *device_output_values = nullptr;
    int *device_error = nullptr;
    int host_error = 0;
    cudaError_t cuda_status = cudaSuccess;
    PreFOSCudaPropagationStatus result = PREFOS_CUDA_PROPAGATION_OK;

    if (milliseconds) *milliseconds = 0.0;
    if (!context || !column_map || !row_map || !output_row_pointers ||
        (output_nnz > 0 && (!output_columns || !output_values)) ||
        output_rows > static_cast<size_t>(INT_MAX) ||
        output_nnz > static_cast<size_t>(INT_MAX))
        return PREFOS_CUDA_PROPAGATION_ERROR;

#define PREFOS_CUDA_COMPACT_WRITE_CHECK(call)                              \
    do                                                                     \
    {                                                                      \
        cuda_status = (call);                                              \
        if (cuda_status != cudaSuccess)                                    \
        {                                                                  \
            result = status_from_cuda(cuda_status);                        \
            goto cleanup;                                                  \
        }                                                                  \
    } while (0)

    PREFOS_CUDA_COMPACT_WRITE_CHECK(allocate_device(
        &device_column_map, context->columns, context->stream));
    PREFOS_CUDA_COMPACT_WRITE_CHECK(
        allocate_device(&device_row_map, context->rows, context->stream));
    PREFOS_CUDA_COMPACT_WRITE_CHECK(allocate_device(
        &device_output_rows, output_rows + 1, context->stream));
    PREFOS_CUDA_COMPACT_WRITE_CHECK(allocate_device(
        &device_output_columns, output_nnz, context->stream));
    PREFOS_CUDA_COMPACT_WRITE_CHECK(allocate_device(
        &device_output_values, output_nnz, context->stream));
    PREFOS_CUDA_COMPACT_WRITE_CHECK(
        allocate_device(&device_error, 1, context->stream));
    PREFOS_CUDA_COMPACT_WRITE_CHECK(copy_to_device(
        device_column_map, column_map, context->columns,
        context->stream));
    PREFOS_CUDA_COMPACT_WRITE_CHECK(copy_to_device(
        device_row_map, row_map, context->rows, context->stream));
    PREFOS_CUDA_COMPACT_WRITE_CHECK(copy_to_device(
        device_output_rows, output_row_pointers, output_rows + 1,
        context->stream));
    PREFOS_CUDA_COMPACT_WRITE_CHECK(
        cudaMemsetAsync(device_error, 0, sizeof(int), context->stream));
    if (context->rows > 0)
    {
        unsigned int blocks = static_cast<unsigned int>(
            (context->rows + kThreads - 1) / kThreads);
        compact_a_write_kernel<<<blocks, kThreads, 0, context->stream>>>(
            context->rows, context->row_pointers,
            context->column_indices, context->values, device_column_map,
            device_row_map, device_output_rows, device_output_columns,
            device_output_values, device_error);
        PREFOS_CUDA_COMPACT_WRITE_CHECK(cudaGetLastError());
    }
    PREFOS_CUDA_COMPACT_WRITE_CHECK(copy_to_host(
        output_columns, device_output_columns, output_nnz,
        context->stream));
    PREFOS_CUDA_COMPACT_WRITE_CHECK(copy_to_host(
        output_values, device_output_values, output_nnz,
        context->stream));
    PREFOS_CUDA_COMPACT_WRITE_CHECK(copy_to_host(
        &host_error, device_error, 1, context->stream));
    PREFOS_CUDA_COMPACT_WRITE_CHECK(
        cudaStreamSynchronize(context->stream));
    if (host_error) result = PREFOS_CUDA_PROPAGATION_ERROR;

cleanup:
    cudaFreeAsync(device_column_map, context->stream);
    cudaFreeAsync(device_row_map, context->stream);
    cudaFreeAsync(device_output_rows, context->stream);
    cudaFreeAsync(device_output_columns, context->stream);
    cudaFreeAsync(device_output_values, context->stream);
    cudaFreeAsync(device_error, context->stream);
    (void) cudaStreamSynchronize(context->stream);
    if (milliseconds)
        *milliseconds =
            std::chrono::duration<double, std::milli>(Clock::now() - start)
                .count();
    return result;
#undef PREFOS_CUDA_COMPACT_WRITE_CHECK
}
