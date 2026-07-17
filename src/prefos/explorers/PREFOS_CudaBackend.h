/*
 * Copyright 2026 Hongpei Li
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef PREFOS_CUDA_BACKEND_H
#define PREFOS_CUDA_BACKEND_H

#include "PREFOS_Internal.h"
#include "PREFOS_CudaLinearPropagation.h"

PREFOS_INTERNAL PreFOSCudaWorkspace *
prefos_internal_cuda_workspace_get(PreFOSPresolver *presolver,
                                   PreFOSCudaPropagationStatus *status);

PREFOS_INTERNAL void
prefos_internal_cuda_workspace_release(PreFOSPresolver *presolver);

#endif
