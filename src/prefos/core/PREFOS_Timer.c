/*
 * Copyright 2026 Hongpei Li
 * SPDX-License-Identifier: Apache-2.0
 */

#include "PREFOS_Timer.h"

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

void prefos_internal_timer_now(PreFOSTimestamp *timestamp)
{
    LARGE_INTEGER counter = {0};
    (void) QueryPerformanceCounter(&counter);
    timestamp->ticks = (int64_t) counter.QuadPart;
}

double prefos_internal_timer_elapsed_milliseconds(const PreFOSTimestamp *start,
                                                  const PreFOSTimestamp *stop)
{
    LARGE_INTEGER frequency = {0};
    if (!QueryPerformanceFrequency(&frequency) || frequency.QuadPart <= 0)
        return 0.0;
    return 1000.0 * (double) (stop->ticks - start->ticks) /
           (double) frequency.QuadPart;
}

#else

#include <time.h>

void prefos_internal_timer_now(PreFOSTimestamp *timestamp)
{
    struct timespec value = {0, 0};
    (void) clock_gettime(CLOCK_MONOTONIC, &value);
    timestamp->ticks =
        (int64_t) value.tv_sec * INT64_C(1000000000) + (int64_t) value.tv_nsec;
}

double prefos_internal_timer_elapsed_milliseconds(const PreFOSTimestamp *start,
                                                  const PreFOSTimestamp *stop)
{
    return 1e-6 * (double) (stop->ticks - start->ticks);
}

#endif
