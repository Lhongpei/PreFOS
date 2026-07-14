/*
 * Copyright 2026 Hongpei Li
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef PREFOS_EXPONENTIAL_CONE_H
#define PREFOS_EXPONENTIAL_CONE_H

#include <PreFOS/PreFOS.h>

PreFOSStatus prefos_internal_exponential_cone_violation(const PreFOSConeBlock *cone,
                                                  const double *point, int dual,
                                                  long double *violation);
int prefos_internal_exponential_coordinate_is_nonnegative(size_t position);
int prefos_internal_exponential_cone_contains(const PreFOSConeBlock *cone,
                                           const double *point, int dual);
PreFOSStatus prefos_internal_exponential_minimum_z(double lower_x, double lower_y,
                                             double upper_y, long double *minimum_z);
PreFOSStatus prefos_internal_exponential_maximum_x(double upper_z, double lower_y,
                                             double upper_y, long double *maximum_x);

#endif
