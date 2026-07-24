/*
 * Copyright 2026 Hongpei Li
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef PREFOS_TRIVIAL_REDUCTIONS_H
#define PREFOS_TRIVIAL_REDUCTIONS_H

#include "PREFOS_Internal.h"

PREFOS_INTERNAL PreFOSStatus prefos_internal_find_fixed_box_variables(
    PreFOSPresolver *presolver, size_t *n_fixed);

PREFOS_INTERNAL PreFOSStatus prefos_internal_reduce_trivial_rows(
    PreFOSPresolver *presolver);

PREFOS_INTERNAL PreFOSStatus prefos_internal_reduce_trivial_row_candidates(
    PreFOSPresolver *presolver, const int *rows, size_t count);

#endif
