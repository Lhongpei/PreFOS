/*
 * Copyright 2026 Hongpei Li
 * SPDX-License-Identifier: Apache-2.0
 */

#include "PREFOS_LinearPropagationCache.h"
#include "PREFOS_LinearPropagation.h"

void prefos_internal_free_linear_propagation_state(
    PreFOSLinearPropagationState *state)
{
    if (!state) return;
    free(state->activities);
    free(state->redundancy_activities);
    free(state->box_column_pointers);
    free(state->adjacent_rows);
    free(state->adjacent_positions);
    free(state->redundancy_activity_stale);
    free(state->box_max_abs_coefficient);
    free(state->external_bound_events);
    free(state->external_rows);
    free(state->external_bound_dirty);
    free(state->external_row_dirty);
    free(state->external_row_activity_dirty);
#ifndef NDEBUG
    free(state->debug_cached_lower);
    free(state->debug_cached_upper);
    free(state->debug_cached_constraint_lower);
    free(state->debug_cached_constraint_upper);
    free(state->debug_cached_residual_source_column);
#endif
    presolve_dirty_rows_free(&state->dirty_rows);
    memset(state, 0, sizeof(*state));
}

void prefos_internal_free_linear_propagation_cache(
    PreFOSPresolver *presolver)
{
    PreFOSLinearPropagationState *state;
    if (!presolver || !presolver->linear_propagation_cache) return;
    state = presolver->linear_propagation_cache;
    presolver->linear_propagation_cache = NULL;
    prefos_internal_free_linear_propagation_state(state);
    free(state);
}

static int reserve_external_bound_event(
    PreFOSLinearPropagationState *state, size_t limit)
{
    PreFOSExternalBoundEvent *events;
    size_t capacity;
    if (state->n_external_bound_events <
        state->external_bound_capacity)
        return 1;
    capacity = state->external_bound_capacity == 0
                   ? (limit < 64 ? limit : 64)
                   : state->external_bound_capacity * 2;
    if (capacity < state->external_bound_capacity ||
        capacity > limit ||
        capacity > SIZE_MAX / sizeof(*events))
        capacity = limit;
    if (capacity <= state->external_bound_capacity ||
        capacity > SIZE_MAX / sizeof(*events))
        return 0;
    events = (PreFOSExternalBoundEvent *) realloc(
        state->external_bound_events, capacity * sizeof(*events));
    if (!events) return 0;
    state->external_bound_events = events;
    state->external_bound_capacity = capacity;
    return 1;
}

static int reserve_external_row(
    PreFOSLinearPropagationState *state, size_t limit)
{
    int *rows;
    size_t capacity;
    if (state->n_external_rows < state->external_row_capacity)
        return 1;
    capacity = state->external_row_capacity == 0
                   ? (limit < 64 ? limit : 64)
                   : state->external_row_capacity * 2;
    if (capacity < state->external_row_capacity ||
        capacity > limit ||
        capacity > SIZE_MAX / sizeof(*rows))
        capacity = limit;
    if (capacity <= state->external_row_capacity ||
        capacity > SIZE_MAX / sizeof(*rows))
        return 0;
    rows = (int *) realloc(
        state->external_rows, capacity * sizeof(*rows));
    if (!rows) return 0;
    state->external_rows = rows;
    state->external_row_capacity = capacity;
    return 1;
}

void prefos_internal_linear_cache_mark_bound_dirty(
    PreFOSPresolver *presolver, int column)
{
    PreFOSLinearPropagationState *state;
    PreFOSExternalBoundEvent *event;
    if (!presolver || column < 0 ||
        (size_t) column >= presolver->original.n)
        return;
    state = presolver->linear_propagation_cache;
    if (!state ||
        state->lower_bounds != presolver->propagation_lower ||
        state->upper_bounds != presolver->propagation_upper)
        return;
    if (!state->external_bound_dirty)
    {
        state->external_bound_dirty = (unsigned char *) calloc(
            presolver->original.n, sizeof(unsigned char));
        if (!state->external_bound_dirty)
        {
            state->external_event_overflow = 1;
            return;
        }
    }
    if (state->external_bound_dirty[column]) return;
    if (!reserve_external_bound_event(
            state, presolver->original.n))
    {
        state->external_event_overflow = 1;
        return;
    }
    event =
        &state->external_bound_events[state->n_external_bound_events++];
    event->column = column;
    event->old_lower = presolver->propagation_lower[column];
    event->old_upper = presolver->propagation_upper[column];
    state->external_bound_dirty[column] = 1;
}

void prefos_internal_linear_cache_mark_row_dirty(
    PreFOSPresolver *presolver, size_t row, int activity_changed)
{
    PreFOSLinearPropagationState *state;
    if (!presolver || row >= presolver->original.A.rows) return;
    state = presolver->linear_propagation_cache;
    if (!state) return;
    if (!state->external_row_dirty)
    {
        state->external_row_dirty = (unsigned char *) calloc(
            presolver->original.A.rows, sizeof(unsigned char));
        state->external_row_activity_dirty =
            (unsigned char *) calloc(
                presolver->original.A.rows, sizeof(unsigned char));
        if (!state->external_row_dirty ||
            !state->external_row_activity_dirty)
        {
            free(state->external_row_dirty);
            free(state->external_row_activity_dirty);
            state->external_row_dirty = NULL;
            state->external_row_activity_dirty = NULL;
            state->external_event_overflow = 1;
            return;
        }
    }
    if (state->external_row_dirty[row])
    {
        if (activity_changed)
            state->external_row_activity_dirty[row] = 1;
        return;
    }
    if (!reserve_external_row(
            state, presolver->original.A.rows))
    {
        state->external_event_overflow = 1;
        return;
    }
    state->external_row_dirty[row] = 1;
    state->external_row_activity_dirty[row] =
        (unsigned char) (activity_changed != 0);
    state->external_rows[state->n_external_rows++] = (int) row;
}
