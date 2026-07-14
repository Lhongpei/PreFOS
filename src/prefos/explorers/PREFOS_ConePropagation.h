/*
 * Copyright 2026 Hongpei Li
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef PREFOS_CONE_PROPAGATION_H
#define PREFOS_CONE_PROPAGATION_H

#include "PREFOS_Internal.h"

PREFOS_INTERNAL PreFOSStatus
prefos_internal_propagate_cone_envelopes(PreFOSPresolver *presolver);
PREFOS_INTERNAL PreFOSStatus prefos_internal_propagate_cone_block_envelopes(
    PreFOSPresolver *presolver, const PreFOSConeBlock *cone, double *lower, double *upper,
    int *changed);
PREFOS_INTERNAL PreFOSStatus
prefos_internal_detect_zero_cone_collapses(PreFOSPresolver *presolver);
PREFOS_INTERNAL int
prefos_internal_psd_matrix_index_is_removed(const PreFOSPSDFaceReduction *face,
                                         size_t matrix_index);
#endif
