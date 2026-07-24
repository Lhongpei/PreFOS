/*
 * Copyright 2026 Hongpei Li
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef PREFOS_PASS_MANAGER_H
#define PREFOS_PASS_MANAGER_H

#include "explorers/PREFOS_ColumnReductionInternal.h"

PREFOS_INTERNAL PreFOSStatus prefos_internal_run_fast_fixed_point(
    PreFOSPresolver *presolver, int allow_one_sided_singletons,
    int full_trivial_scan, PreFOSColumnWorkspace *shared_workspace);

#endif
