/*
 * Copyright 2026 Hongpei Li
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef PREFOS_POWER_CONE_H
#define PREFOS_POWER_CONE_H

#include <PreFOS/PreFOS.h>

PreFOSStatus prefos_internal_power_cone_violation(const PreFOSConeBlock *cone,
                                            const double *point, int dual,
                                            long double *violation);
int prefos_internal_power_coordinate_is_nonnegative(size_t position);
int prefos_internal_power_cone_contains(const PreFOSConeBlock *cone, const double *point,
                                     int dual);
PreFOSStatus prefos_internal_power_capacity(double alpha, double upper_x, double upper_y,
                                      long double *capacity);
PreFOSStatus prefos_internal_power_implied_axis_lower(double alpha,
                                                long double minimum_abs_z,
                                                double other_upper, int infer_x,
                                                long double *lower);

#endif
