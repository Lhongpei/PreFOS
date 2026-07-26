/*
 * Copyright 2026 Hongpei Li
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef PREFOS_THREAD_H
#define PREFOS_THREAD_H

#include <stdlib.h>

static inline int prefos_cpu_thread_limit(int maximum)
{
    const char *value;
    char *end;
    long requested;

    if (maximum <= 1) return 1;
#if defined(_WIN32) || defined(_WIN64)
    return 1;
#else
    value = getenv("PREFOS_CPU_THREADS");
    if (!value || !*value) return maximum;
    requested = strtol(value, &end, 10);
    if (end == value || *end != '\0' || requested <= 1)
        return 1;
    return requested < (long) maximum ? (int) requested
                                      : maximum;
#endif
}

#if defined(_WIN32) || defined(_WIN64)

typedef struct
{
    int unused;
} PreFOSThread;

static inline int prefos_thread_create(
    PreFOSThread *thread, void *(*function)(void *), void *argument)
{
    (void) thread;
    function(argument);
    return 0;
}

static inline int prefos_thread_join(PreFOSThread *thread)
{
    (void) thread;
    return 0;
}

#else

#include <pthread.h>

typedef pthread_t PreFOSThread;

static inline int prefos_thread_create(
    PreFOSThread *thread, void *(*function)(void *), void *argument)
{
    return pthread_create(thread, NULL, function, argument);
}

static inline int prefos_thread_join(PreFOSThread *thread)
{
    return pthread_join(*thread, NULL);
}

#endif

#endif
