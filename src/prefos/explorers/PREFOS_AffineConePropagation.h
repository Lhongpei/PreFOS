/*
 * Copyright 2026 Hongpei Li
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef PREFOS_AFFINE_CONE_PROPAGATION_H
#define PREFOS_AFFINE_CONE_PROPAGATION_H

#include "PREFOS_Internal.h"

PREFOS_INTERNAL PreFOSStatus prefos_internal_propagate_affine_cones(PreFOSPresolver *presolver,
                                                           int *changed);

#endif
