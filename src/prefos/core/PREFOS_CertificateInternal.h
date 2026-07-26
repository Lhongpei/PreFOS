/*
 * Copyright 2026 Hongpei Li
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef PREFOS_CERTIFICATE_INTERNAL_H
#define PREFOS_CERTIFICATE_INTERNAL_H

#include "PREFOS_Internal.h"

typedef struct
{
    size_t n;
    const PreFOSCsrMatrix *A;
    const double *constraint_lower;
    const double *constraint_upper;
    const PreFOSCsrMatrix *Q;
    PreFOSQStorage q_storage;
    const PreFOSCsrMatrix *R;
    const double *D;
    const double *c;
    size_t n_box;
    const int *box_indices;
    const double *box_lower;
    const double *box_upper;
    size_t n_cones;
    const PreFOSConeBlock *cones;
    const PreFOSCsrMatrix *affine_cone_matrix;
    const double *affine_cone_offset;
    size_t n_affine_cones;
    const PreFOSAffineConeBlock *affine_cones;
} PreFOSCertificateModel;

PREFOS_INTERNAL PreFOSCertificateModel
prefos_internal_original_certificate_model(
    const PreFOSPresolver *presolver);

PREFOS_INTERNAL PreFOSCertificateModel
prefos_internal_reduced_certificate_model(
    const PreFOSPresolver *presolver);

PREFOS_INTERNAL int prefos_internal_certificate_vector_is_finite(
    const double *values, size_t count);

PREFOS_INTERNAL PreFOSStatus
prefos_internal_evaluate_infeasibility_certificate(
    const PreFOSCertificateModel *model, const double *y, const double *z,
    const double *affine_z, double tolerance,
    const PreFOSPresolver *facial_context,
    PreFOSInfeasibilityCertificateResiduals *residuals);

PREFOS_INTERNAL PreFOSStatus prefos_internal_evaluate_unbounded_ray(
    const PreFOSCertificateModel *model, const double *ray,
    double tolerance, PreFOSUnboundedRayResiduals *residuals);

#endif
