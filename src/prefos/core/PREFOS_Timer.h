/*
 * Copyright 2026 Hongpei Li
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef PREFOS_TIMER_H
#define PREFOS_TIMER_H

#include <stdint.h>

typedef struct
{
    int64_t ticks;
} PreFOSTimestamp;

void prefos_internal_timer_now(PreFOSTimestamp *timestamp);
double prefos_internal_timer_elapsed_milliseconds(const PreFOSTimestamp *start,
                                                  const PreFOSTimestamp *stop);

#endif /* PREFOS_TIMER_H */
