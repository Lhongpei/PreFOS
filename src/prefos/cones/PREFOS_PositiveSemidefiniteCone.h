/*
 * Copyright 2026 Hongpei Li
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef PREFOS_POSITIVE_SEMIDEFINITE_CONE_H
#define PREFOS_POSITIVE_SEMIDEFINITE_CONE_H

#include <PreFOS/PreFOS.h>

PreFOSStatus prefos_internal_psd_minimum_eigenvalue(const PreFOSConeBlock *cone,
                                              const double *point, double tolerance,
                                              long double *minimum_eigenvalue);
PreFOSStatus prefos_internal_evaluate_psd_violation(const PreFOSConeBlock *cone,
                                              const double *point, double tolerance,
                                              long double *violation);

#endif
