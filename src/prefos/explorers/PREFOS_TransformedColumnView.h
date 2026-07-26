/*
 * Copyright 2026 Hongpei Li
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef PREFOS_TRANSFORMED_COLUMN_VIEW_H
#define PREFOS_TRANSFORMED_COLUMN_VIEW_H

#include "PREFOS_Internal.h"

struct PreFOSColumnWorkspace;

typedef struct
{
    int *starts;
    int *ends;
    int *rows;
    double *values;
    int *live_degrees;
    unsigned char *eligible;
    size_t nnz;
    size_t affected_columns;
    size_t candidate_columns;
    size_t dirty_rows;
    size_t dirty_source_nnz;
} PreFOSTransformedColumnView;

PREFOS_INTERNAL PreFOSStatus
prefos_internal_build_transformed_column_view(
    const PreFOSPresolver *presolver,
    const struct PreFOSColumnWorkspace *workspace,
    PreFOSTransformedColumnView *view, int *built);

PREFOS_INTERNAL void prefos_internal_free_transformed_column_view(
    PreFOSTransformedColumnView *view);

#endif
