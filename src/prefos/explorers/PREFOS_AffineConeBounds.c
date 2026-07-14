/*
 * Copyright 2026 Hongpei Li
 * SPDX-License-Identifier: Apache-2.0
 */

#include "PREFOS_AffineConeBounds.h"

#include "PREFOS_LinearPropagation.h"
#include "cones/PREFOS_ExponentialCone.h"
#include "cones/PREFOS_PowerCone.h"

static size_t intrinsic_candidate_count(const PreFOSAffineConeBlock *block)
{
    if (block->type == PREFOS_CONE_SECOND_ORDER) return 1;
    if (block->type == PREFOS_CONE_ROTATED_SECOND_ORDER) return 2;
    if (block->type == PREFOS_CONE_POSITIVE_SEMIDEFINITE) return block->matrix_order;
    return block->dimension;
}

static int intrinsic_candidate(const PreFOSAffineConeBlock *block, size_t candidate,
                               size_t *coordinate)
{
    *coordinate = candidate;
    if (block->type == PREFOS_CONE_NONNEGATIVE) return 1;
    if (block->type == PREFOS_CONE_SECOND_ORDER ||
        block->type == PREFOS_CONE_ROTATED_SECOND_ORDER)
        return 1;
    if (block->type == PREFOS_CONE_POSITIVE_SEMIDEFINITE)
    {
        *coordinate = candidate * (candidate + 1) / 2 + candidate;
        return 1;
    }
    if (block->type == PREFOS_CONE_EXPONENTIAL)
        return prefos_internal_exponential_coordinate_is_nonnegative(candidate);
    if (block->type == PREFOS_CONE_POWER)
        return prefos_internal_power_coordinate_is_nonnegative(candidate);
    return 0;
}

static int bound_policy_accepts(const PreFOSPresolver *presolver, int box_position,
                                double candidate, int is_lower)
{
    double current, opposite;
    long double difference, scale;
    if (presolver->settings.propagated_bound_policy ==
        PREFOS_PROPAGATED_BOUND_POLICY_FIRST_ORDER)
        return 1;
    current = is_lower ? presolver->working_box_lower[box_position]
                       : presolver->working_box_upper[box_position];
    if (isfinite(current)) return 1;
    opposite = is_lower ? presolver->working_box_upper[box_position]
                        : presolver->working_box_lower[box_position];
    if (!isfinite(opposite)) return 0;
    difference = fabsl((long double) candidate - (long double) opposite);
    scale = fmaxl(
        1.0L, fmaxl(fabsl((long double) candidate), fabsl((long double) opposite)));
    return difference <=
           (long double) presolver->settings.fixed_variable_tolerance * scale;
}

static PreFOSStatus append_certificate(PreFOSPresolver *presolver, size_t affine_row,
                                    int column, int generated_column,
                                    double coefficient, double offset,
                                    double implied_bound, int is_lower)
{
    PreFOSAffineBoundCertificate *certificates;
    size_t capacity;
    if (presolver->n_affine_bound_certificates ==
        presolver->affine_bound_certificate_capacity)
    {
        capacity = presolver->affine_bound_certificate_capacity == 0
                       ? 16
                       : 2 * presolver->affine_bound_certificate_capacity;
        if (capacity < presolver->affine_bound_certificate_capacity ||
            capacity > SIZE_MAX / sizeof(PreFOSAffineBoundCertificate))
            return PREFOS_STATUS_OUT_OF_MEMORY;
        certificates = (PreFOSAffineBoundCertificate *) realloc(
            presolver->affine_bound_certificates,
            capacity * sizeof(PreFOSAffineBoundCertificate));
        if (!certificates) return PREFOS_STATUS_OUT_OF_MEMORY;
        presolver->affine_bound_certificates = certificates;
        presolver->affine_bound_certificate_capacity = capacity;
    }
    presolver->affine_bound_certificates[presolver->n_affine_bound_certificates++] =
        (PreFOSAffineBoundCertificate){
            affine_row, column,        generated_column,        coefficient,
            offset,     implied_bound, (unsigned char) is_lower};
    return PREFOS_STATUS_OK;
}

static PreFOSStatus materialize_singleton(PreFOSPresolver *presolver, size_t affine_row,
                                       int column, int generated_column,
                                       double coefficient, double offset)
{
    int box_position, is_lower;
    double candidate, opposite, current;
    long double exact, margin;
    PreFOSStatus status;
    if (column < 0 || (size_t) column >= presolver->original.n ||
        coefficient == 0.0 || !isfinite(coefficient) || !isfinite(offset) ||
        presolver->is_fixed[column] || presolver->is_substituted[column])
        return PREFOS_STATUS_OK;
    box_position = presolver->variable_to_box[column];
    if (box_position < 0) return PREFOS_STATUS_OK;

    is_lower = coefficient > 0.0;
    current = is_lower ? presolver->working_box_lower[box_position]
                       : presolver->working_box_upper[box_position];
    exact = -(long double) offset / (long double) coefficient;
    candidate = prefos_internal_outward_bound_cast(exact, is_lower);
    if (!isfinite(candidate)) return PREFOS_STATUS_OK;
    opposite = is_lower ? presolver->working_box_upper[box_position]
                        : presolver->working_box_lower[box_position];
    margin = prefos_internal_propagation_margin(presolver, (long double) opposite);
    if (isfinite(opposite) &&
        ((is_lower && (long double) candidate > (long double) opposite + margin) ||
         (!is_lower && (long double) candidate < (long double) opposite - margin)))
        return PREFOS_STATUS_PRIMAL_INFEASIBLE;
    if (isfinite(opposite) &&
        ((is_lower && candidate > opposite) || (!is_lower && candidate < opposite)))
        return PREFOS_STATUS_OK;
    if ((is_lower && candidate <= current) || (!is_lower && candidate >= current))
        return PREFOS_STATUS_OK;
    if ((!isfinite(opposite) || candidate != opposite) &&
        !prefos_internal_is_significant_improvement(presolver, current, candidate,
                                                 is_lower))
        return PREFOS_STATUS_OK;
    if (!bound_policy_accepts(presolver, box_position, candidate, is_lower))
    {
        ++presolver->stats.suppressed_affine_cone_box_bounds;
        return PREFOS_STATUS_OK;
    }

    status = append_certificate(presolver, affine_row, column, generated_column,
                                coefficient, offset, candidate, is_lower);
    if (status != PREFOS_STATUS_OK) return status;
    if (is_lower)
    {
        presolver->working_box_lower[box_position] = candidate;
        if (candidate > presolver->propagation_lower[column])
            presolver->propagation_lower[column] = candidate;
    }
    else
    {
        presolver->working_box_upper[box_position] = candidate;
        if (candidate < presolver->propagation_upper[column])
            presolver->propagation_upper[column] = candidate;
    }
    ++presolver->stats.materialized_affine_cone_box_bounds;
    return PREFOS_STATUS_OK;
}

static PreFOSStatus materialize_input_block(PreFOSPresolver *presolver,
                                         const PreFOSAffineConeBlock *block,
                                         size_t first_row)
{
    const PreFOSProblemData *problem = &presolver->original;
    size_t candidate;
    for (candidate = 0; candidate < intrinsic_candidate_count(block); ++candidate)
    {
        size_t coordinate;
        size_t row;
        int p, column = -1;
        double coefficient = 0.0;
        size_t nonzeros = 0;
        PreFOSStatus status;
        if (!intrinsic_candidate(block, candidate, &coordinate)) continue;
        row = first_row + coordinate;
        for (p = problem->affine_cone_matrix.row_pointers[row];
             p < problem->affine_cone_matrix.row_pointers[row + 1]; ++p)
        {
            if (problem->affine_cone_matrix.values[p] == 0.0) continue;
            column = problem->affine_cone_matrix.column_indices[p];
            coefficient = problem->affine_cone_matrix.values[p];
            if (++nonzeros > 1) break;
        }
        if (nonzeros != 1) continue;
        status = materialize_singleton(presolver, row, column, -1, coefficient,
                                       problem->affine_cone_offset[row]);
        if (status != PREFOS_STATUS_OK) return status;
    }
    return PREFOS_STATUS_OK;
}

static PreFOSStatus materialize_generated_block(PreFOSPresolver *presolver,
                                             const PreFOSConeBlock *direct,
                                             size_t first_row)
{
    PreFOSAffineConeBlock block = {direct->type, direct->dimension,
                                direct->matrix_order, direct->power_alpha};
    size_t candidate;
    for (candidate = 0; candidate < intrinsic_candidate_count(&block); ++candidate)
    {
        size_t coordinate;
        int generated_column;
        size_t start;
        int column;
        double coefficient;
        PreFOSStatus status;
        if (!intrinsic_candidate(&block, candidate, &coordinate)) continue;
        generated_column = direct->indices[coordinate];
        if (!presolver->is_substituted[generated_column] ||
            presolver->substitution_term_count[generated_column] != 1)
            continue;
        start = presolver->substitution_term_start[generated_column];
        if (start >= presolver->n_substitution_terms)
            return PREFOS_STATUS_NUMERICAL_ERROR;
        column = presolver->substitution_targets[start];
        coefficient = presolver->substitution_scales[start];
        status = materialize_singleton(
            presolver, first_row + coordinate, column, generated_column, coefficient,
            presolver->substitution_constant[generated_column]);
        if (status != PREFOS_STATUS_OK) return status;
    }
    return PREFOS_STATUS_OK;
}

PreFOSStatus prefos_internal_materialize_affine_cone_bounds(PreFOSPresolver *presolver)
{
    const PreFOSProblemData *problem = &presolver->original;
    size_t affine_row = 0, k;
    PreFOSStatus status;

    for (k = 0; k < problem->n_affine_cones; ++k)
    {
        const PreFOSAffineConeBlock *block = &problem->affine_cones[k];
        if (affine_row > problem->affine_cone_matrix.rows ||
            block->dimension > problem->affine_cone_matrix.rows - affine_row)
            return PREFOS_STATUS_NUMERICAL_ERROR;
        status = materialize_input_block(presolver, block, affine_row);
        if (status != PREFOS_STATUS_OK) return status;
        affine_row += block->dimension;
    }
    if (affine_row != problem->affine_cone_matrix.rows)
        return PREFOS_STATUS_NUMERICAL_ERROR;

    for (k = 0; k < problem->n_cones; ++k)
    {
        if (!presolver->converted_affine_cones[k]) continue;
        status =
            materialize_generated_block(presolver, &problem->cones[k], affine_row);
        if (status != PREFOS_STATUS_OK) return status;
        affine_row += problem->cones[k].dimension;
    }
    return PREFOS_STATUS_OK;
}
