/*
 * Copyright 2025-2026 Daniel Cederberg
 * Copyright 2026 Hongpei Li
 *
 * Modified for PreFOS in 2026.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef PRESOLVE_DIRTY_ROWS_H
#define PRESOLVE_DIRTY_ROWS_H

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef enum
{
    PRESOLVE_ROW_IDLE = 0,
    PRESOLVE_ROW_CURRENT = 1,
    PRESOLVE_ROW_PROCESSED = 2,
    PRESOLVE_ROW_NEXT = 3
} PresolveDirtyRowState;

typedef struct
{
    int *current;
    int *next;
    uint8_t *states;
    size_t capacity;
    size_t current_count;
    size_t current_position;
    size_t next_count;
} PresolveDirtyRows;

static inline int presolve_dirty_rows_init(PresolveDirtyRows *queue,
                                           size_t capacity)
{
    memset(queue, 0, sizeof(*queue));
    if (capacity == 0) return 1;
    if (capacity > SIZE_MAX / sizeof(int)) return 0;
    queue->current = (int *) malloc(capacity * sizeof(int));
    queue->next = (int *) malloc(capacity * sizeof(int));
    queue->states = (uint8_t *) calloc(capacity, sizeof(uint8_t));
    if (!queue->current || !queue->next || !queue->states)
    {
        free(queue->current);
        free(queue->next);
        free(queue->states);
        memset(queue, 0, sizeof(*queue));
        return 0;
    }
    queue->capacity = capacity;
    return 1;
}

static inline void presolve_dirty_rows_free(PresolveDirtyRows *queue)
{
    if (!queue) return;
    free(queue->current);
    free(queue->next);
    free(queue->states);
    memset(queue, 0, sizeof(*queue));
}

static inline int presolve_dirty_rows_schedule(PresolveDirtyRows *queue, int row)
{
    PresolveDirtyRowState state;
    if (row < 0 || (size_t) row >= queue->capacity) return 0;
    state = (PresolveDirtyRowState) queue->states[row];
    if (state == PRESOLVE_ROW_IDLE)
    {
        if (queue->current_count >= queue->capacity) return 0;
        queue->states[row] = PRESOLVE_ROW_CURRENT;
        queue->current[queue->current_count++] = row;
    }
    else if (state == PRESOLVE_ROW_PROCESSED)
    {
        if (queue->next_count >= queue->capacity) return 0;
        queue->states[row] = PRESOLVE_ROW_NEXT;
        queue->next[queue->next_count++] = row;
    }
    return 1;
}

static inline int presolve_dirty_rows_pop(PresolveDirtyRows *queue, int *row)
{
    while (queue->current_position < queue->current_count)
    {
        int candidate = queue->current[queue->current_position++];
        if (queue->states[candidate] != PRESOLVE_ROW_CURRENT) continue;
        queue->states[candidate] = PRESOLVE_ROW_PROCESSED;
        *row = candidate;
        return 1;
    }
    return 0;
}

static inline void presolve_dirty_rows_finish_round(PresolveDirtyRows *queue)
{
    size_t position;
    int *swap;
    for (position = 0; position < queue->current_count; ++position)
    {
        int row = queue->current[position];
        if (queue->states[row] == PRESOLVE_ROW_PROCESSED)
            queue->states[row] = PRESOLVE_ROW_IDLE;
    }
    swap = queue->current;
    queue->current = queue->next;
    queue->next = swap;
    queue->current_count = queue->next_count;
    queue->current_position = 0;
    queue->next_count = 0;
    for (position = 0; position < queue->current_count; ++position)
        queue->states[queue->current[position]] = PRESOLVE_ROW_CURRENT;
}

static inline void presolve_dirty_rows_clear(PresolveDirtyRows *queue)
{
    if (queue->capacity > 0)
        memset(queue->states, 0, queue->capacity * sizeof(uint8_t));
    queue->current_count = 0;
    queue->current_position = 0;
    queue->next_count = 0;
}

static inline void presolve_dirty_rows_remap(PresolveDirtyRows *queue,
                                             const int *row_map)
{
    size_t read, write = 0;
    for (read = queue->current_position; read < queue->current_count; ++read)
    {
        int mapped = row_map[queue->current[read]];
        if (mapped >= 0) queue->current[write++] = mapped;
    }
    if (queue->capacity > 0)
        memset(queue->states, 0, queue->capacity * sizeof(uint8_t));
    queue->current_count = write;
    queue->current_position = 0;
    queue->next_count = 0;
    for (read = 0; read < write; ++read)
        queue->states[queue->current[read]] = PRESOLVE_ROW_CURRENT;
}

static inline int presolve_dirty_rows_is_valid(const PresolveDirtyRows *queue)
{
    size_t position;
    if (queue->current_count > queue->capacity ||
        queue->current_position > queue->current_count ||
        queue->next_count > queue->capacity)
        return 0;
    for (position = queue->current_position; position < queue->current_count;
         ++position)
    {
        int row = queue->current[position];
        if (row < 0 || (size_t) row >= queue->capacity ||
            queue->states[row] != PRESOLVE_ROW_CURRENT)
            return 0;
    }
    for (position = 0; position < queue->next_count; ++position)
    {
        int row = queue->next[position];
        if (row < 0 || (size_t) row >= queue->capacity ||
            queue->states[row] != PRESOLVE_ROW_NEXT)
            return 0;
    }
    return 1;
}

#endif
