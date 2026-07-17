/*
 * Copyright 2026 Hongpei Li
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef PREFOS_COLUMN_REDUCTIONS_H
#define PREFOS_COLUMN_REDUCTIONS_H

#include "PREFOS_Internal.h"

PREFOS_INTERNAL PreFOSStatus
prefos_internal_reduce_linear_columns(PreFOSPresolver *presolver);

PREFOS_INTERNAL PreFOSStatus
prefos_internal_reduce_parallel_columns(PreFOSPresolver *presolver);

PREFOS_INTERNAL PreFOSStatus
prefos_internal_remove_redundant_box_bounds(PreFOSPresolver *presolver);

#endif
