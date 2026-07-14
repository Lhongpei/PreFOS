/*
 * Copyright 2025-2026 Daniel Cederberg
 * Copyright 2026 Hongpei Li
 *
 * Modified for PreFOS in 2026.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef PRESOLVE_TRANSFORMATION_LOG_H
#define PRESOLVE_TRANSFORMATION_LOG_H

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct
{
    int row;
    int column;
    double previous_bound;
    double implied_bound;
    double opposite_bound;
    uint8_t is_lower;
    uint8_t has_previous_bound;
    uint8_t has_opposite_bound;
} PresolveBoundChangeRecord;

typedef enum
{
    PRESOLVE_ROW_DELETED = 0,
    PRESOLVE_ROW_ADDED,
    PRESOLVE_ROWS_ADDED,
    PRESOLVE_ROW_LOWER_CHANGED,
    PRESOLVE_ROW_UPPER_CHANGED,
    PRESOLVE_ROW_BOUND_CHANGE_SOURCE,
    PRESOLVE_ROW_EQUALITY_RELAXED
} PresolveRowTransformationType;

typedef struct
{
    PresolveRowTransformationType type;
    int row;
    int source_row;
    double dual_value;
    double ratio;
    double new_side;
    double pivot;
    int *indices;
    double *coefficients;
    size_t length;
    size_t related_count;
} PresolveRowTransformationRecord;

typedef enum
{
    PRESOLVE_COLUMN_FIXED = 0,
    PRESOLVE_COLUMN_FIXED_INFINITE,
    PRESOLVE_COLUMN_SUBSTITUTED,
    PRESOLVE_COLUMNS_PARALLEL
} PresolveColumnTransformationType;

typedef struct
{
    PresolveColumnTransformationType type;
    int column;
    int secondary_column;
    int source_row;
    int direction;
    int column_tag;
    int secondary_column_tag;
    double value;
    double objective_coefficient;
    double rhs;
    double ratio;
    double lower;
    double upper;
    double secondary_lower;
    double secondary_upper;
    int *indices;
    double *coefficients;
    size_t length;
    int *row_starts;
    double *row_sides;
    size_t n_rows;
    int *affine_indices;
    double *affine_coefficients;
    size_t affine_length;
} PresolveColumnTransformationRecord;

typedef enum
{
    PRESOLVE_CONE_COLLAPSED = 0,
    PRESOLVE_CONE_FACE_REDUCED
} PresolveConeTransformationType;

typedef struct
{
    PresolveConeTransformationType type;
    size_t cone_index;
} PresolveConeTransformationRecord;

typedef enum
{
    PRESOLVE_TRANSFORMATION_BOUND_CHANGE = 0,
    PRESOLVE_TRANSFORMATION_ROW,
    PRESOLVE_TRANSFORMATION_COLUMN,
    PRESOLVE_TRANSFORMATION_CONE
} PresolveTransformationEventType;

typedef struct
{
    PresolveTransformationEventType type;
    size_t record_index;
} PresolveTransformationEvent;

typedef struct
{
    PresolveBoundChangeRecord *bound_changes;
    size_t n_bound_changes;
    size_t bound_change_capacity;
    PresolveRowTransformationRecord *row_transformations;
    size_t n_row_transformations;
    size_t row_transformation_capacity;
    PresolveColumnTransformationRecord *column_transformations;
    size_t n_column_transformations;
    size_t column_transformation_capacity;
    PresolveConeTransformationRecord *cone_transformations;
    size_t n_cone_transformations;
    size_t cone_transformation_capacity;
    PresolveTransformationEvent *events;
    size_t n_events;
    size_t event_capacity;
} PresolveTransformationLog;

static inline void presolve_transformation_log_init(PresolveTransformationLog *log)
{
    memset(log, 0, sizeof(*log));
}

static inline void presolve_transformation_log_free(PresolveTransformationLog *log)
{
    size_t i;
    if (!log) return;
    free(log->bound_changes);
    for (i = 0; i < log->n_row_transformations; ++i)
    {
        free(log->row_transformations[i].indices);
        free(log->row_transformations[i].coefficients);
    }
    free(log->row_transformations);
    for (i = 0; i < log->n_column_transformations; ++i)
    {
        free(log->column_transformations[i].indices);
        free(log->column_transformations[i].coefficients);
        free(log->column_transformations[i].row_starts);
        free(log->column_transformations[i].row_sides);
        free(log->column_transformations[i].affine_indices);
        free(log->column_transformations[i].affine_coefficients);
    }
    free(log->column_transformations);
    free(log->cone_transformations);
    free(log->events);
    memset(log, 0, sizeof(*log));
}

static inline int
presolve_transformation_log_reserve_events(PresolveTransformationLog *log)
{
    PresolveTransformationEvent *events;
    size_t capacity;
    if (log->n_events < log->event_capacity) return 1;
    capacity = log->event_capacity == 0 ? 16 : 2 * log->event_capacity;
    if (capacity < log->event_capacity ||
        capacity > SIZE_MAX / sizeof(PresolveTransformationEvent))
        return 0;
    events = (PresolveTransformationEvent *) realloc(
        log->events, capacity * sizeof(PresolveTransformationEvent));
    if (!events) return 0;
    log->events = events;
    log->event_capacity = capacity;
    return 1;
}

static inline int
presolve_transformation_log_reserve_bound_changes(PresolveTransformationLog *log)
{
    PresolveBoundChangeRecord *records;
    size_t capacity;
    if (log->n_bound_changes < log->bound_change_capacity) return 1;
    capacity = log->bound_change_capacity == 0 ? 16 : 2 * log->bound_change_capacity;
    if (capacity < log->bound_change_capacity ||
        capacity > SIZE_MAX / sizeof(PresolveBoundChangeRecord))
        return 0;
    records = (PresolveBoundChangeRecord *) realloc(
        log->bound_changes, capacity * sizeof(PresolveBoundChangeRecord));
    if (!records) return 0;
    log->bound_changes = records;
    log->bound_change_capacity = capacity;
    return 1;
}

static inline int presolve_transformation_log_reserve_row_transformations(
    PresolveTransformationLog *log)
{
    PresolveRowTransformationRecord *records;
    size_t capacity;
    if (log->n_row_transformations < log->row_transformation_capacity) return 1;
    capacity = log->row_transformation_capacity == 0
                   ? 16
                   : 2 * log->row_transformation_capacity;
    if (capacity < log->row_transformation_capacity ||
        capacity > SIZE_MAX / sizeof(PresolveRowTransformationRecord))
        return 0;
    records = (PresolveRowTransformationRecord *) realloc(
        log->row_transformations,
        capacity * sizeof(PresolveRowTransformationRecord));
    if (!records) return 0;
    log->row_transformations = records;
    log->row_transformation_capacity = capacity;
    return 1;
}

static inline int presolve_transformation_log_reserve_cone_transformations(
    PresolveTransformationLog *log)
{
    PresolveConeTransformationRecord *records;
    size_t capacity;
    if (log->n_cone_transformations < log->cone_transformation_capacity) return 1;
    capacity = log->cone_transformation_capacity == 0
                   ? 16
                   : 2 * log->cone_transformation_capacity;
    if (capacity < log->cone_transformation_capacity ||
        capacity > SIZE_MAX / sizeof(PresolveConeTransformationRecord))
        return 0;
    records = (PresolveConeTransformationRecord *) realloc(
        log->cone_transformations,
        capacity * sizeof(PresolveConeTransformationRecord));
    if (!records) return 0;
    log->cone_transformations = records;
    log->cone_transformation_capacity = capacity;
    return 1;
}

static inline int presolve_transformation_log_reserve_column_transformations(
    PresolveTransformationLog *log)
{
    PresolveColumnTransformationRecord *records;
    size_t capacity;
    if (log->n_column_transformations < log->column_transformation_capacity)
        return 1;
    capacity = log->column_transformation_capacity == 0
                   ? 16
                   : 2 * log->column_transformation_capacity;
    if (capacity < log->column_transformation_capacity ||
        capacity > SIZE_MAX / sizeof(PresolveColumnTransformationRecord))
        return 0;
    records = (PresolveColumnTransformationRecord *) realloc(
        log->column_transformations,
        capacity * sizeof(PresolveColumnTransformationRecord));
    if (!records) return 0;
    log->column_transformations = records;
    log->column_transformation_capacity = capacity;
    return 1;
}

static inline int presolve_transformation_log_append_bound_change(
    PresolveTransformationLog *log, const PresolveBoundChangeRecord *record,
    size_t *record_index)
{
    size_t index;
    if (!presolve_transformation_log_reserve_bound_changes(log) ||
        !presolve_transformation_log_reserve_events(log))
        return 0;
    index = log->n_bound_changes;
    if (record_index) *record_index = log->n_bound_changes;
    log->bound_changes[log->n_bound_changes++] = *record;
    log->events[log->n_events++] =
        (PresolveTransformationEvent) {PRESOLVE_TRANSFORMATION_BOUND_CHANGE, index};
    return 1;
}

static inline int presolve_transformation_log_assign_recent_bound_changes(
    PresolveTransformationLog *log, size_t count, int row)
{
    size_t first, position;
    if (count > log->n_bound_changes) return 0;
    first = log->n_bound_changes - count;
    for (position = first; position < log->n_bound_changes; ++position)
        log->bound_changes[position].row = row;
    return 1;
}

static inline int presolve_transformation_log_append_row_transformation(
    PresolveTransformationLog *log, const PresolveRowTransformationRecord *record,
    size_t *record_index)
{
    PresolveRowTransformationRecord copy = *record;
    size_t bytes, index;

    copy.indices = NULL;
    copy.coefficients = NULL;
    if (record->length > 0)
    {
        if (!record->indices || !record->coefficients ||
            record->length > SIZE_MAX / sizeof(int) ||
            record->length > SIZE_MAX / sizeof(double))
            return 0;
        bytes = record->length * sizeof(int);
        copy.indices = (int *) malloc(bytes);
        if (!copy.indices) return 0;
        memcpy(copy.indices, record->indices, bytes);

        bytes = record->length * sizeof(double);
        copy.coefficients = (double *) malloc(bytes);
        if (!copy.coefficients)
        {
            free(copy.indices);
            return 0;
        }
        memcpy(copy.coefficients, record->coefficients, bytes);
    }

    if (!presolve_transformation_log_reserve_row_transformations(log) ||
        !presolve_transformation_log_reserve_events(log))
    {
        free(copy.indices);
        free(copy.coefficients);
        return 0;
    }

    index = log->n_row_transformations;
    if (record_index) *record_index = log->n_row_transformations;
    log->row_transformations[log->n_row_transformations++] = copy;
    log->events[log->n_events++] =
        (PresolveTransformationEvent) {PRESOLVE_TRANSFORMATION_ROW, index};
    return 1;
}

static inline int presolve_transformation_log_append_cone_transformation(
    PresolveTransformationLog *log, const PresolveConeTransformationRecord *record,
    size_t *record_index)
{
    size_t index;
    if (!presolve_transformation_log_reserve_cone_transformations(log) ||
        !presolve_transformation_log_reserve_events(log))
        return 0;
    index = log->n_cone_transformations;
    if (record_index) *record_index = index;
    log->cone_transformations[log->n_cone_transformations++] = *record;
    log->events[log->n_events++] =
        (PresolveTransformationEvent) {PRESOLVE_TRANSFORMATION_CONE, index};
    return 1;
}

static inline int presolve_transformation_log_append_column_transformation(
    PresolveTransformationLog *log, const PresolveColumnTransformationRecord *record,
    size_t *record_index)
{
    PresolveColumnTransformationRecord copy = *record;
    size_t index, position;

    copy.indices = NULL;
    copy.coefficients = NULL;
    copy.row_starts = NULL;
    copy.row_sides = NULL;
    copy.affine_indices = NULL;
    copy.affine_coefficients = NULL;
    if (record->length > 0)
    {
        if (!record->indices || !record->coefficients ||
            record->length > SIZE_MAX / sizeof(int) ||
            record->length > SIZE_MAX / sizeof(double))
            return 0;
        copy.indices = (int *) malloc(record->length * sizeof(int));
        copy.coefficients = (double *) malloc(record->length * sizeof(double));
        if (!copy.indices || !copy.coefficients)
        {
            free(copy.indices);
            free(copy.coefficients);
            return 0;
        }
        memcpy(copy.indices, record->indices, record->length * sizeof(int));
        memcpy(copy.coefficients, record->coefficients,
               record->length * sizeof(double));
    }
    if (record->n_rows > 0)
    {
        if (!record->row_starts || !record->row_sides ||
            record->n_rows == SIZE_MAX ||
            record->n_rows + 1 > SIZE_MAX / sizeof(int) ||
            record->n_rows > SIZE_MAX / sizeof(double) ||
            record->row_starts[0] != 0 ||
            (size_t) record->row_starts[record->n_rows] != record->length)
        {
            free(copy.indices);
            free(copy.coefficients);
            return 0;
        }
        for (position = 0; position < record->n_rows; ++position)
            if (record->row_starts[position] < 0 ||
                record->row_starts[position] > record->row_starts[position + 1])
            {
                free(copy.indices);
                free(copy.coefficients);
                return 0;
            }
        copy.row_starts = (int *) malloc((record->n_rows + 1) * sizeof(int));
        copy.row_sides = (double *) malloc(record->n_rows * sizeof(double));
        if (!copy.row_starts || !copy.row_sides)
        {
            free(copy.indices);
            free(copy.coefficients);
            free(copy.row_starts);
            free(copy.row_sides);
            return 0;
        }
        memcpy(copy.row_starts, record->row_starts,
               (record->n_rows + 1) * sizeof(int));
        memcpy(copy.row_sides, record->row_sides, record->n_rows * sizeof(double));
    }
    if (record->affine_length > 0)
    {
        if (!record->affine_indices || !record->affine_coefficients ||
            record->affine_length > SIZE_MAX / sizeof(int) ||
            record->affine_length > SIZE_MAX / sizeof(double))
        {
            free(copy.indices);
            free(copy.coefficients);
            free(copy.row_starts);
            free(copy.row_sides);
            return 0;
        }
        copy.affine_indices =
            (int *) malloc(record->affine_length * sizeof(int));
        copy.affine_coefficients =
            (double *) malloc(record->affine_length * sizeof(double));
        if (!copy.affine_indices || !copy.affine_coefficients)
        {
            free(copy.indices);
            free(copy.coefficients);
            free(copy.row_starts);
            free(copy.row_sides);
            free(copy.affine_indices);
            free(copy.affine_coefficients);
            return 0;
        }
        memcpy(copy.affine_indices, record->affine_indices,
               record->affine_length * sizeof(int));
        memcpy(copy.affine_coefficients, record->affine_coefficients,
               record->affine_length * sizeof(double));
    }
    if (!presolve_transformation_log_reserve_column_transformations(log) ||
        !presolve_transformation_log_reserve_events(log))
    {
        free(copy.indices);
        free(copy.coefficients);
        free(copy.row_starts);
        free(copy.row_sides);
        free(copy.affine_indices);
        free(copy.affine_coefficients);
        return 0;
    }
    index = log->n_column_transformations;
    if (record_index) *record_index = index;
    log->column_transformations[log->n_column_transformations++] = copy;
    log->events[log->n_events++] =
        (PresolveTransformationEvent) {PRESOLVE_TRANSFORMATION_COLUMN, index};
    return 1;
}

#endif
