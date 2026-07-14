/*
 * Copyright 2026 Hongpei Li
 * SPDX-License-Identifier: Apache-2.0
 */

#include "PREFOS_CudaLinearPropagation.h"

#include <cub/block/block_reduce.cuh>
#include <cuda_runtime.h>

#include <cfloat>
#include <chrono>
#include <climits>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <new>
#include <thread>
#include <utility>

struct PreFOSCudaLinearPropagationContext
{
    cudaStream_t stream;
    size_t rows;
    size_t columns;
    size_t n_long_rows;
    int *row_pointers;
    int *column_indices;
    double *values;
    double *constraint_lower;
    double *constraint_upper;
    int *candidate_map;
    unsigned char *remove_rows;
    double *lower_bounds;
    double *upper_bounds;
    double *lower_candidates;
    double *upper_candidates;
    int *lower_source_rows;
    int *upper_source_rows;
    int *long_rows;
    int *suspected_infeasible_row;
    int *numerical_error;
};

namespace
{

constexpr int kThreads = 256;
constexpr int kLongRowThreshold = 256;

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

#include "PREFOS_CudaLinearPropagationKernels.cuh"

enum class WarmupPhase
{
    idle,
    running,
    ready,
    failed
};

struct WarmupState
{
    std::mutex mutex;
    std::condition_variable condition;
    std::thread worker;
    WarmupPhase phase = WarmupPhase::idle;
    PreFOSCudaPropagationStatus status = PREFOS_CUDA_PROPAGATION_UNAVAILABLE;

    ~WarmupState()
    {
        if (worker.joinable()) worker.join();
    }
};

WarmupState &warmup_state()
{
    static WarmupState state;
    return state;
}

PreFOSCudaPropagationStatus perform_warmup()
{
    int device = 0;
    int device_count = 0;
    cudaMemPool_t pool;
    uint64_t release_threshold = UINT64_MAX;
    cudaError_t status = cudaGetDeviceCount(&device_count);
    if (status != cudaSuccess || device_count == 0)
        return PREFOS_CUDA_PROPAGATION_UNAVAILABLE;
    status = cudaFree(nullptr);
    if (status != cudaSuccess) return status_from_cuda(status);
    status = cudaGetDevice(&device);
    if (status != cudaSuccess) return status_from_cuda(status);
    status = cudaDeviceGetDefaultMemPool(&pool, device);
    if (status != cudaSuccess) return status_from_cuda(status);
    status = cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold,
                                     &release_threshold);
    return status_from_cuda(status);
}

void free_context(PreFOSCudaLinearPropagationContext *context)
{
    if (!context) return;
    if (context->stream)
    {
        cudaFreeAsync(context->row_pointers, context->stream);
        cudaFreeAsync(context->column_indices, context->stream);
        cudaFreeAsync(context->values, context->stream);
        cudaFreeAsync(context->constraint_lower, context->stream);
        cudaFreeAsync(context->constraint_upper, context->stream);
        cudaFreeAsync(context->candidate_map, context->stream);
        cudaFreeAsync(context->remove_rows, context->stream);
        cudaFreeAsync(context->lower_bounds, context->stream);
        cudaFreeAsync(context->upper_bounds, context->stream);
        cudaFreeAsync(context->lower_candidates, context->stream);
        cudaFreeAsync(context->upper_candidates, context->stream);
        cudaFreeAsync(context->lower_source_rows, context->stream);
        cudaFreeAsync(context->upper_source_rows, context->stream);
        cudaFreeAsync(context->long_rows, context->stream);
        cudaFreeAsync(context->suspected_infeasible_row, context->stream);
        cudaFreeAsync(context->numerical_error, context->stream);
        cudaStreamSynchronize(context->stream);
        cudaStreamDestroy(context->stream);
    }
    delete context;
}

} // namespace

extern "C" PreFOSCudaPropagationStatus prefos_cuda_linear_propagation_warmup(void)
{
    (void) prefos_cuda_linear_propagation_warmup_async();
    return prefos_cuda_linear_propagation_warmup_wait();
}

extern "C" int prefos_cuda_linear_propagation_warmup_async(void)
{
    WarmupState &state = warmup_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (state.phase == WarmupPhase::ready || state.phase == WarmupPhase::running)
        return 1;
    if (state.phase == WarmupPhase::failed) return 0;
    state.phase = WarmupPhase::running;
    try
    {
        state.worker = std::thread(
            [&state]()
            {
                PreFOSCudaPropagationStatus status = perform_warmup();
                {
                    std::lock_guard<std::mutex> worker_lock(state.mutex);
                    state.status = status;
                    state.phase = status == PREFOS_CUDA_PROPAGATION_OK
                                      ? WarmupPhase::ready
                                      : WarmupPhase::failed;
                }
                state.condition.notify_all();
            });
    }
    catch (...)
    {
        state.status = PREFOS_CUDA_PROPAGATION_ERROR;
        state.phase = WarmupPhase::failed;
        return 0;
    }
    return 1;
}

extern "C" int prefos_cuda_linear_propagation_warmup_ready(void)
{
    WarmupState &state = warmup_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    return state.phase == WarmupPhase::ready;
}

extern "C" PreFOSCudaPropagationStatus prefos_cuda_linear_propagation_warmup_wait(void)
{
    WarmupState &state = warmup_state();
    std::thread completed_worker;
    PreFOSCudaPropagationStatus status;
    {
        std::unique_lock<std::mutex> lock(state.mutex);
        if (state.phase == WarmupPhase::idle)
        {
            lock.unlock();
            if (!prefos_cuda_linear_propagation_warmup_async())
            {
                lock.lock();
                return state.status;
            }
            lock.lock();
        }
        state.condition.wait(lock, [&state]()
                             { return state.phase != WarmupPhase::running; });
        status = state.status;
        if (state.worker.joinable()) completed_worker = std::move(state.worker);
    }
    if (completed_worker.joinable()) completed_worker.join();
    return status;
}

extern "C" void prefos_cuda_linear_propagation_release_cache(void)
{
    int device = 0;
    cudaMemPool_t pool;
    uint64_t release_threshold = 0;
    uint64_t retained_threshold = UINT64_MAX;
    if (prefos_cuda_linear_propagation_warmup_wait() != PREFOS_CUDA_PROPAGATION_OK) return;
    if (cudaGetDevice(&device) != cudaSuccess) return;
    if (cudaDeviceGetDefaultMemPool(&pool, device) != cudaSuccess) return;
    (void) cudaDeviceSynchronize();
    (void) cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold,
                                   &release_threshold);
    (void) cudaMemPoolTrimTo(pool, 0);
    (void) cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold,
                                   &retained_threshold);
}

extern "C" PreFOSCudaPropagationStatus prefos_cuda_linear_propagation_create(
    size_t rows, size_t columns, size_t nnz, const int *row_pointers,
    const int *column_indices, const double *values, const double *constraint_lower,
    const double *constraint_upper, const int *candidate_map,
    const unsigned char *remove_rows, PreFOSCudaLinearPropagationContext **output,
    double *setup_milliseconds, size_t *long_rows)
{
    using Clock = std::chrono::steady_clock;
    auto start = Clock::now();
    PreFOSCudaLinearPropagationContext *context;
    int *host_long_rows = nullptr;
    cudaError_t status;
    PreFOSCudaPropagationStatus warmup_status;

    if (!output) return PREFOS_CUDA_PROPAGATION_ERROR;
    *output = nullptr;
    if (setup_milliseconds) *setup_milliseconds = 0.0;
    if (long_rows) *long_rows = 0;
    warmup_status = prefos_cuda_linear_propagation_warmup();
    if (warmup_status != PREFOS_CUDA_PROPAGATION_OK) return warmup_status;
    context = new (std::nothrow) PreFOSCudaLinearPropagationContext{};
    if (!context) return PREFOS_CUDA_PROPAGATION_OUT_OF_MEMORY;
    context->rows = rows;
    context->columns = columns;

#define PREFOS_CUDA_CHECK(call)                                                        \
    do                                                                              \
    {                                                                               \
        status = (call);                                                            \
        if (status != cudaSuccess)                                                  \
        {                                                                           \
            PreFOSCudaPropagationStatus result = status_from_cuda(status);             \
            delete[] host_long_rows;                                                \
            free_context(context);                                                  \
            return result;                                                          \
        }                                                                           \
    } while (0)

    PREFOS_CUDA_CHECK(
        cudaStreamCreateWithFlags(&context->stream, cudaStreamNonBlocking));
    host_long_rows = new (std::nothrow) int[rows];
    if (rows > 0 && !host_long_rows)
    {
        free_context(context);
        return PREFOS_CUDA_PROPAGATION_OUT_OF_MEMORY;
    }
    for (size_t row = 0; row < rows; ++row)
        if (row_pointers[row + 1] - row_pointers[row] >= kLongRowThreshold)
            host_long_rows[context->n_long_rows++] = static_cast<int>(row);

    PREFOS_CUDA_CHECK(
        allocate_device(&context->row_pointers, rows + 1, context->stream));
    PREFOS_CUDA_CHECK(allocate_device(&context->column_indices, nnz, context->stream));
    PREFOS_CUDA_CHECK(allocate_device(&context->values, nnz, context->stream));
    PREFOS_CUDA_CHECK(
        allocate_device(&context->constraint_lower, rows, context->stream));
    PREFOS_CUDA_CHECK(
        allocate_device(&context->constraint_upper, rows, context->stream));
    PREFOS_CUDA_CHECK(
        allocate_device(&context->candidate_map, columns, context->stream));
    PREFOS_CUDA_CHECK(allocate_device(&context->remove_rows, rows, context->stream));
    PREFOS_CUDA_CHECK(
        allocate_device(&context->lower_bounds, columns, context->stream));
    PREFOS_CUDA_CHECK(
        allocate_device(&context->upper_bounds, columns, context->stream));
    PREFOS_CUDA_CHECK(
        allocate_device(&context->lower_candidates, columns, context->stream));
    PREFOS_CUDA_CHECK(
        allocate_device(&context->upper_candidates, columns, context->stream));
    PREFOS_CUDA_CHECK(
        allocate_device(&context->lower_source_rows, columns, context->stream));
    PREFOS_CUDA_CHECK(
        allocate_device(&context->upper_source_rows, columns, context->stream));
    PREFOS_CUDA_CHECK(
        allocate_device(&context->long_rows, context->n_long_rows, context->stream));
    PREFOS_CUDA_CHECK(
        allocate_device(&context->suspected_infeasible_row, 1, context->stream));
    PREFOS_CUDA_CHECK(allocate_device(&context->numerical_error, 1, context->stream));

    PREFOS_CUDA_CHECK(copy_to_device(context->row_pointers, row_pointers, rows + 1,
                                  context->stream));
    PREFOS_CUDA_CHECK(copy_to_device(context->column_indices, column_indices, nnz,
                                  context->stream));
    PREFOS_CUDA_CHECK(copy_to_device(context->values, values, nnz, context->stream));
    PREFOS_CUDA_CHECK(copy_to_device(context->constraint_lower, constraint_lower, rows,
                                  context->stream));
    PREFOS_CUDA_CHECK(copy_to_device(context->constraint_upper, constraint_upper, rows,
                                  context->stream));
    PREFOS_CUDA_CHECK(copy_to_device(context->candidate_map, candidate_map, columns,
                                  context->stream));
    PREFOS_CUDA_CHECK(
        copy_to_device(context->remove_rows, remove_rows, rows, context->stream));
    PREFOS_CUDA_CHECK(copy_to_device(context->long_rows, host_long_rows,
                                  context->n_long_rows, context->stream));
    PREFOS_CUDA_CHECK(cudaStreamSynchronize(context->stream));
    delete[] host_long_rows;
    host_long_rows = nullptr;
    *output = context;
    if (long_rows) *long_rows = context->n_long_rows;
    if (setup_milliseconds)
        *setup_milliseconds =
            std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    return PREFOS_CUDA_PROPAGATION_OK;
#undef PREFOS_CUDA_CHECK
}

extern "C" PreFOSCudaPropagationStatus prefos_cuda_linear_propagation_round(
    PreFOSCudaLinearPropagationContext *context, const double *lower_bounds,
    const double *upper_bounds, double feasibility_tolerance,
    double maximum_inferred_bound_magnitude, double *lower_candidates,
    double *upper_candidates, int *lower_source_rows, int *upper_source_rows,
    int *suspected_infeasible_row, double *transfer_milliseconds,
    double *kernel_milliseconds)
{
    using Clock = std::chrono::steady_clock;
    auto total_start = Clock::now();
    cudaEvent_t kernel_start = nullptr, kernel_stop = nullptr;
    cudaError_t status;
    int numerical_error = 0;
    float elapsed = 0.0F;
    if (!context) return PREFOS_CUDA_PROPAGATION_ERROR;
    unsigned int column_blocks =
        static_cast<unsigned int>((context->columns + kThreads - 1) / kThreads);
    unsigned int row_blocks =
        static_cast<unsigned int>((context->rows + kThreads - 1) / kThreads);
    if (transfer_milliseconds) *transfer_milliseconds = 0.0;
    if (kernel_milliseconds) *kernel_milliseconds = 0.0;

#define PREFOS_CUDA_ROUND_CHECK(call)                                                  \
    do                                                                              \
    {                                                                               \
        status = (call);                                                            \
        if (status != cudaSuccess)                                                  \
        {                                                                           \
            if (kernel_start) cudaEventDestroy(kernel_start);                       \
            if (kernel_stop) cudaEventDestroy(kernel_stop);                         \
            return status_from_cuda(status);                                        \
        }                                                                           \
    } while (0)

    PREFOS_CUDA_ROUND_CHECK(copy_to_device(context->lower_bounds, lower_bounds,
                                        context->columns, context->stream));
    PREFOS_CUDA_ROUND_CHECK(copy_to_device(context->upper_bounds, upper_bounds,
                                        context->columns, context->stream));
    PREFOS_CUDA_ROUND_CHECK(cudaEventCreate(&kernel_start));
    PREFOS_CUDA_ROUND_CHECK(cudaEventCreate(&kernel_stop));
    PREFOS_CUDA_ROUND_CHECK(cudaEventRecord(kernel_start, context->stream));
    initialize_round_kernel<<<column_blocks, kThreads, 0, context->stream>>>(
        context->columns, context->lower_candidates, context->upper_candidates,
        context->lower_source_rows, context->upper_source_rows,
        context->suspected_infeasible_row, context->numerical_error);
    PREFOS_CUDA_ROUND_CHECK(cudaGetLastError());
    propagation_kernel<<<row_blocks, kThreads, 0, context->stream>>>(
        context->rows, context->row_pointers, context->column_indices,
        context->values, context->constraint_lower, context->constraint_upper,
        context->candidate_map, context->remove_rows, context->lower_bounds,
        context->upper_bounds, feasibility_tolerance,
        maximum_inferred_bound_magnitude, context->lower_candidates,
        context->upper_candidates, context->lower_source_rows,
        context->upper_source_rows, context->suspected_infeasible_row,
        context->numerical_error, false);
    PREFOS_CUDA_ROUND_CHECK(cudaGetLastError());
    if (context->n_long_rows > 0)
    {
        long_row_propagation_kernel<<<static_cast<unsigned int>(
                                          context->n_long_rows),
                                      kThreads, 0, context->stream>>>(
            context->n_long_rows, context->long_rows, context->row_pointers,
            context->column_indices, context->values, context->constraint_lower,
            context->constraint_upper, context->candidate_map, context->remove_rows,
            context->lower_bounds, context->upper_bounds, feasibility_tolerance,
            maximum_inferred_bound_magnitude, context->lower_candidates,
            context->upper_candidates, context->lower_source_rows,
            context->upper_source_rows, context->suspected_infeasible_row,
            context->numerical_error, false);
        PREFOS_CUDA_ROUND_CHECK(cudaGetLastError());
    }
    propagation_kernel<<<row_blocks, kThreads, 0, context->stream>>>(
        context->rows, context->row_pointers, context->column_indices,
        context->values, context->constraint_lower, context->constraint_upper,
        context->candidate_map, context->remove_rows, context->lower_bounds,
        context->upper_bounds, feasibility_tolerance,
        maximum_inferred_bound_magnitude, context->lower_candidates,
        context->upper_candidates, context->lower_source_rows,
        context->upper_source_rows, context->suspected_infeasible_row,
        context->numerical_error, true);
    PREFOS_CUDA_ROUND_CHECK(cudaGetLastError());
    if (context->n_long_rows > 0)
    {
        long_row_propagation_kernel<<<static_cast<unsigned int>(
                                          context->n_long_rows),
                                      kThreads, 0, context->stream>>>(
            context->n_long_rows, context->long_rows, context->row_pointers,
            context->column_indices, context->values, context->constraint_lower,
            context->constraint_upper, context->candidate_map, context->remove_rows,
            context->lower_bounds, context->upper_bounds, feasibility_tolerance,
            maximum_inferred_bound_magnitude, context->lower_candidates,
            context->upper_candidates, context->lower_source_rows,
            context->upper_source_rows, context->suspected_infeasible_row,
            context->numerical_error, true);
        PREFOS_CUDA_ROUND_CHECK(cudaGetLastError());
    }
    PREFOS_CUDA_ROUND_CHECK(cudaEventRecord(kernel_stop, context->stream));
    PREFOS_CUDA_ROUND_CHECK(cudaEventSynchronize(kernel_stop));
    PREFOS_CUDA_ROUND_CHECK(cudaEventElapsedTime(&elapsed, kernel_start, kernel_stop));
    PREFOS_CUDA_ROUND_CHECK(copy_to_host(lower_candidates, context->lower_candidates,
                                      context->columns, context->stream));
    PREFOS_CUDA_ROUND_CHECK(copy_to_host(upper_candidates, context->upper_candidates,
                                      context->columns, context->stream));
    PREFOS_CUDA_ROUND_CHECK(copy_to_host(lower_source_rows, context->lower_source_rows,
                                      context->columns, context->stream));
    PREFOS_CUDA_ROUND_CHECK(copy_to_host(upper_source_rows, context->upper_source_rows,
                                      context->columns, context->stream));
    PREFOS_CUDA_ROUND_CHECK(copy_to_host(suspected_infeasible_row,
                                      context->suspected_infeasible_row, 1,
                                      context->stream));
    PREFOS_CUDA_ROUND_CHECK(copy_to_host(&numerical_error, context->numerical_error, 1,
                                      context->stream));
    PREFOS_CUDA_ROUND_CHECK(cudaStreamSynchronize(context->stream));
    cudaEventDestroy(kernel_start);
    cudaEventDestroy(kernel_stop);
    if (*suspected_infeasible_row == INT_MAX) *suspected_infeasible_row = -1;
    for (size_t column = 0; column < context->columns; ++column)
    {
        if ((isfinite(lower_candidates[column]) && lower_source_rows[column] < 0) ||
            (isfinite(upper_candidates[column]) && upper_source_rows[column] < 0))
            return PREFOS_CUDA_PROPAGATION_ERROR;
    }
    if (kernel_milliseconds) *kernel_milliseconds = static_cast<double>(elapsed);
    if (transfer_milliseconds)
    {
        double total =
            std::chrono::duration<double, std::milli>(Clock::now() - total_start)
                .count();
        *transfer_milliseconds = fmax(0.0, total - static_cast<double>(elapsed));
    }
    return numerical_error ? PREFOS_CUDA_PROPAGATION_ERROR : PREFOS_CUDA_PROPAGATION_OK;
#undef PREFOS_CUDA_ROUND_CHECK
}

extern "C" void
prefos_cuda_linear_propagation_free(PreFOSCudaLinearPropagationContext *context)
{
    free_context(context);
}
