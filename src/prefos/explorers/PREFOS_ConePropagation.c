/*
 * Copyright 2026 Hongpei Li
 * SPDX-License-Identifier: Apache-2.0
 */

#include "PREFOS_ConePropagation.h"

#include "PREFOS_AffineConeBounds.h"
#include "PREFOS_AffineConePropagation.h"
#include "PREFOS_LinearPropagation.h"
#include "core/PREFOS_Timer.h"
#include "cones/PREFOS_ExponentialCone.h"
#include "cones/PREFOS_PositiveSemidefiniteCone.h"
#include "cones/PREFOS_PowerCone.h"

static PreFOSStatus update_cone_envelope_lower(PreFOSPresolver *presolver, int column,
                                            double candidate, int *changed)
{
    double current = presolver->propagation_lower[column];
    double upper = presolver->propagation_upper[column];
    long double margin;
    if (candidate == -INFINITY) return PREFOS_STATUS_OK;
    if (candidate == INFINITY) return PREFOS_STATUS_PRIMAL_INFEASIBLE;
    margin = prefos_internal_propagation_margin(presolver, (long double) upper);
    if (isfinite(upper) && (long double) candidate > (long double) upper + margin)
        return PREFOS_STATUS_PRIMAL_INFEASIBLE;
    if (!prefos_internal_is_significant_improvement(presolver, current, candidate, 1))
        return PREFOS_STATUS_OK;
    if (candidate > upper) candidate = upper;
    presolver->propagation_lower[column] = candidate;
    ++presolver->stats.tightened_cone_envelopes;
    *changed = 1;
    return PREFOS_STATUS_OK;
}

static PreFOSStatus update_cone_envelope_upper(PreFOSPresolver *presolver, int column,
                                            double candidate, int *changed)
{
    double lower = presolver->propagation_lower[column];
    double current = presolver->propagation_upper[column];
    long double margin;
    if (candidate == INFINITY) return PREFOS_STATUS_OK;
    if (candidate == -INFINITY) return PREFOS_STATUS_PRIMAL_INFEASIBLE;
    margin = prefos_internal_propagation_margin(presolver, (long double) lower);
    if (isfinite(lower) && (long double) candidate < (long double) lower - margin)
        return PREFOS_STATUS_PRIMAL_INFEASIBLE;
    if (!prefos_internal_is_significant_improvement(presolver, current, candidate, 0))
        return PREFOS_STATUS_OK;
    if (candidate < lower) candidate = lower;
    presolver->propagation_upper[column] = candidate;
    ++presolver->stats.tightened_cone_envelopes;
    *changed = 1;
    return PREFOS_STATUS_OK;
}

static PreFOSStatus extract_singleton_cone_envelopes(PreFOSPresolver *presolver)
{
    const PreFOSProblemData *problem = &presolver->original;
    size_t row;
    for (row = 0; row < problem->A.rows; ++row)
    {
        int p, column = -1, changed = 0;
        double coefficient = 0.0;
        size_t nonzeros = 0;
        double implied_lower, implied_upper;
        PreFOSStatus status;
        for (p = problem->A.row_pointers[row]; p < problem->A.row_pointers[row + 1];
             ++p)
        {
            if (problem->A.values[p] == 0.0) continue;
            column = problem->A.column_indices[p];
            coefficient = problem->A.values[p];
            if (++nonzeros > 1) break;
        }
        if (nonzeros != 1 || presolver->variable_to_box[column] >= 0 ||
            presolver->is_substituted[column])
            continue;

        if (coefficient > 0.0)
        {
            implied_lower = prefos_internal_outward_bound_cast(
                (long double) presolver->working_constraint_lower[row] / coefficient,
                1);
            implied_upper = prefos_internal_outward_bound_cast(
                (long double) presolver->working_constraint_upper[row] / coefficient,
                0);
        }
        else
        {
            implied_lower = prefos_internal_outward_bound_cast(
                (long double) presolver->working_constraint_upper[row] / coefficient,
                1);
            implied_upper = prefos_internal_outward_bound_cast(
                (long double) presolver->working_constraint_lower[row] / coefficient,
                0);
        }
        status =
            update_cone_envelope_lower(presolver, column, implied_lower, &changed);
        if (status != PREFOS_STATUS_OK) return status;
        status =
            update_cone_envelope_upper(presolver, column, implied_upper, &changed);
        if (status != PREFOS_STATUS_OK) return status;
    }
    return PREFOS_STATUS_OK;
}

static long double minimum_absolute_value(double lower, double upper)
{
    if (lower > 0.0) return (long double) lower;
    if (upper < 0.0) return -(long double) upper;
    return 0.0L;
}

static PreFOSStatus propagate_nonnegative_cone(PreFOSPresolver *presolver,
                                            const PreFOSConeBlock *cone, int *changed)
{
    size_t i;
    for (i = 0; i < cone->dimension; ++i)
    {
        PreFOSStatus status =
            update_cone_envelope_lower(presolver, cone->indices[i], 0.0, changed);
        if (status != PREFOS_STATUS_OK) return status;
    }
    return PREFOS_STATUS_OK;
}

static PreFOSStatus propagate_exponential_cone(PreFOSPresolver *presolver,
                                            const PreFOSConeBlock *cone, int *changed)
{
    int x_column = cone->indices[0];
    int y_column = cone->indices[1];
    int z_column = cone->indices[2];
    long double candidate;
    size_t i;
    ++presolver->stats.exponential_cones_processed;
    for (i = 0; i < cone->dimension; ++i)
    {
        PreFOSStatus status;
        if (!prefos_internal_exponential_coordinate_is_nonnegative(i)) continue;
        status =
            update_cone_envelope_lower(presolver, cone->indices[i], 0.0, changed);
        if (status != PREFOS_STATUS_OK) return status;
    }
    if (!presolver->settings.exponential_propagation) return PREFOS_STATUS_OK;
    ++presolver->stats.exponential_z_lower_attempts;
    if (prefos_internal_exponential_minimum_z(presolver->propagation_lower[x_column],
                                           presolver->propagation_lower[y_column],
                                           presolver->propagation_upper[y_column],
                                           &candidate) != PREFOS_STATUS_OK)
        return PREFOS_STATUS_NUMERICAL_ERROR;
    {
        size_t hits_before = presolver->stats.tightened_cone_envelopes;
        PreFOSStatus status = update_cone_envelope_lower(
            presolver, z_column, prefos_internal_outward_bound_cast(candidate, 1),
            changed);
        if (status != PREFOS_STATUS_OK) return status;
        presolver->stats.exponential_z_lower_hits +=
            presolver->stats.tightened_cone_envelopes - hits_before;
    }
    ++presolver->stats.exponential_x_upper_attempts;
    if (prefos_internal_exponential_maximum_x(presolver->propagation_upper[z_column],
                                           presolver->propagation_lower[y_column],
                                           presolver->propagation_upper[y_column],
                                           &candidate) != PREFOS_STATUS_OK)
        return PREFOS_STATUS_NUMERICAL_ERROR;
    {
        size_t hits_before = presolver->stats.tightened_cone_envelopes;
        PreFOSStatus status = update_cone_envelope_upper(
            presolver, x_column, prefos_internal_outward_bound_cast(candidate, 0),
            changed);
        if (status != PREFOS_STATUS_OK) return status;
        presolver->stats.exponential_x_upper_hits +=
            presolver->stats.tightened_cone_envelopes - hits_before;
    }
    return PREFOS_STATUS_OK;
}

static PreFOSStatus propagate_power_cone(PreFOSPresolver *presolver,
                                      const PreFOSConeBlock *cone, int *changed)
{
    int x_column = cone->indices[0];
    int y_column = cone->indices[1];
    int z_column = cone->indices[2];
    long double minimum_abs_z, capacity, candidate;
    size_t i;
    ++presolver->stats.power_cones_processed;
    for (i = 0; i < cone->dimension; ++i)
    {
        PreFOSStatus status;
        if (!prefos_internal_power_coordinate_is_nonnegative(i)) continue;
        status =
            update_cone_envelope_lower(presolver, cone->indices[i], 0.0, changed);
        if (status != PREFOS_STATUS_OK) return status;
    }
    if (!presolver->settings.power_propagation) return PREFOS_STATUS_OK;
    minimum_abs_z = minimum_absolute_value(presolver->propagation_lower[z_column],
                                           presolver->propagation_upper[z_column]);
    if (minimum_abs_z == 0.0L) ++presolver->stats.power_zero_minimum_abs_z_attempts;
    ++presolver->stats.power_capacity_attempts;
    if (prefos_internal_power_capacity(
            cone->power_alpha, presolver->propagation_upper[x_column],
            presolver->propagation_upper[y_column], &capacity) != PREFOS_STATUS_OK)
        return PREFOS_STATUS_NUMERICAL_ERROR;
    if (!isfinite(capacity)) ++presolver->stats.power_unbounded_capacity_attempts;
    if (minimum_abs_z >
        capacity + prefos_internal_propagation_margin(presolver, capacity))
        return PREFOS_STATUS_PRIMAL_INFEASIBLE;
    if (isfinite(capacity))
    {
        size_t hits_before = presolver->stats.tightened_cone_envelopes;
        PreFOSStatus status = update_cone_envelope_lower(
            presolver, z_column, prefos_internal_outward_bound_cast(-capacity, 1),
            changed);
        if (status != PREFOS_STATUS_OK) return status;
        status = update_cone_envelope_upper(
            presolver, z_column, prefos_internal_outward_bound_cast(capacity, 0),
            changed);
        if (status != PREFOS_STATUS_OK) return status;
        presolver->stats.power_z_bound_hits +=
            presolver->stats.tightened_cone_envelopes - hits_before;
    }
    if (minimum_abs_z == 0.0L) return PREFOS_STATUS_OK;
    ++presolver->stats.power_axis_attempts;
    if (prefos_internal_power_implied_axis_lower(cone->power_alpha, minimum_abs_z,
                                              presolver->propagation_upper[y_column],
                                              1, &candidate) != PREFOS_STATUS_OK)
        return PREFOS_STATUS_NUMERICAL_ERROR;
    {
        size_t hits_before = presolver->stats.tightened_cone_envelopes;
        PreFOSStatus status = update_cone_envelope_lower(
            presolver, x_column, prefos_internal_outward_bound_cast(candidate, 1),
            changed);
        if (status != PREFOS_STATUS_OK) return status;
        presolver->stats.power_axis_hits +=
            presolver->stats.tightened_cone_envelopes - hits_before;
    }
    ++presolver->stats.power_axis_attempts;
    if (prefos_internal_power_implied_axis_lower(cone->power_alpha, minimum_abs_z,
                                              presolver->propagation_upper[x_column],
                                              0, &candidate) != PREFOS_STATUS_OK)
        return PREFOS_STATUS_NUMERICAL_ERROR;
    {
        size_t hits_before = presolver->stats.tightened_cone_envelopes;
        PreFOSStatus status = update_cone_envelope_lower(
            presolver, y_column, prefos_internal_outward_bound_cast(candidate, 1),
            changed);
        if (status != PREFOS_STATUS_OK) return status;
        presolver->stats.power_axis_hits +=
            presolver->stats.tightened_cone_envelopes - hits_before;
    }
    return PREFOS_STATUS_OK;
}

static PreFOSStatus propagate_soc_cone(PreFOSPresolver *presolver,
                                    const PreFOSConeBlock *cone, int *changed)
{
    int t_column = cone->indices[0];
    double t_upper;
    long double minimum_norm_squared = 0.0L;
    size_t i;
    PreFOSStatus status = update_cone_envelope_lower(presolver, t_column, 0.0, changed);
    if (status != PREFOS_STATUS_OK) return status;

    for (i = 1; i < cone->dimension; ++i)
    {
        int column = cone->indices[i];
        long double minimum =
            minimum_absolute_value(presolver->propagation_lower[column],
                                   presolver->propagation_upper[column]);
        minimum_norm_squared += minimum * minimum;
        if (!isfinite(minimum_norm_squared)) return PREFOS_STATUS_NUMERICAL_ERROR;
    }
    status = update_cone_envelope_lower(
        presolver, t_column,
        prefos_internal_outward_bound_cast(sqrtl(minimum_norm_squared), 1), changed);
    if (status != PREFOS_STATUS_OK) return status;

    t_upper = presolver->propagation_upper[t_column];
    if (isfinite(t_upper))
    {
        long double capacity = (long double) t_upper * t_upper;
        long double minimum_norm = sqrtl(minimum_norm_squared);
        if (minimum_norm > (long double) t_upper +
                               prefos_internal_propagation_margin(presolver, t_upper))
            return PREFOS_STATUS_PRIMAL_INFEASIBLE;
        for (i = 1; i < cone->dimension; ++i)
        {
            int column = cone->indices[i];
            long double minimum =
                minimum_absolute_value(presolver->propagation_lower[column],
                                       presolver->propagation_upper[column]);
            long double remaining =
                fmaxl(0.0L, capacity - (minimum_norm_squared - minimum * minimum));
            long double limit = sqrtl(remaining);
            status = update_cone_envelope_lower(
                presolver, column, prefos_internal_outward_bound_cast(-limit, 1),
                changed);
            if (status != PREFOS_STATUS_OK) return status;
            status = update_cone_envelope_upper(
                presolver, column, prefos_internal_outward_bound_cast(limit, 0),
                changed);
            if (status != PREFOS_STATUS_OK) return status;
        }
    }
    return PREFOS_STATUS_OK;
}

static PreFOSStatus propagate_rsoc_cone(PreFOSPresolver *presolver,
                                     const PreFOSConeBlock *cone, int *changed)
{
    int u_column = cone->indices[0];
    int v_column = cone->indices[1];
    double u_upper, v_upper;
    long double minimum_norm_squared = 0.0L;
    size_t i;
    PreFOSStatus status = update_cone_envelope_lower(presolver, u_column, 0.0, changed);
    if (status != PREFOS_STATUS_OK) return status;
    status = update_cone_envelope_lower(presolver, v_column, 0.0, changed);
    if (status != PREFOS_STATUS_OK) return status;

    for (i = 2; i < cone->dimension; ++i)
    {
        int column = cone->indices[i];
        long double minimum =
            minimum_absolute_value(presolver->propagation_lower[column],
                                   presolver->propagation_upper[column]);
        minimum_norm_squared += minimum * minimum;
        if (!isfinite(minimum_norm_squared)) return PREFOS_STATUS_NUMERICAL_ERROR;
    }
    u_upper = presolver->propagation_upper[u_column];
    v_upper = presolver->propagation_upper[v_column];

    if (isfinite(v_upper) && v_upper > 0.0)
    {
        long double implied = minimum_norm_squared / (2.0L * v_upper);
        status = update_cone_envelope_lower(
            presolver, u_column, prefos_internal_outward_bound_cast(implied, 1),
            changed);
        if (status != PREFOS_STATUS_OK) return status;
    }
    if (isfinite(u_upper) && u_upper > 0.0)
    {
        long double implied = minimum_norm_squared / (2.0L * u_upper);
        status = update_cone_envelope_lower(
            presolver, v_column, prefos_internal_outward_bound_cast(implied, 1),
            changed);
        if (status != PREFOS_STATUS_OK) return status;
    }

    if ((u_upper == 0.0 || v_upper == 0.0) && minimum_norm_squared > 0.0L)
        return PREFOS_STATUS_PRIMAL_INFEASIBLE;
    if (u_upper == 0.0 || v_upper == 0.0)
    {
        for (i = 2; i < cone->dimension; ++i)
        {
            int column = cone->indices[i];
            status = update_cone_envelope_lower(presolver, column, 0.0, changed);
            if (status != PREFOS_STATUS_OK) return status;
            status = update_cone_envelope_upper(presolver, column, 0.0, changed);
            if (status != PREFOS_STATUS_OK) return status;
        }
    }
    else if (isfinite(u_upper) && isfinite(v_upper))
    {
        long double capacity = 2.0L * u_upper * v_upper;
        if (minimum_norm_squared >
            capacity + prefos_internal_propagation_margin(presolver, capacity))
            return PREFOS_STATUS_PRIMAL_INFEASIBLE;
        for (i = 2; i < cone->dimension; ++i)
        {
            int column = cone->indices[i];
            long double minimum =
                minimum_absolute_value(presolver->propagation_lower[column],
                                       presolver->propagation_upper[column]);
            long double remaining =
                fmaxl(0.0L, capacity - (minimum_norm_squared - minimum * minimum));
            long double limit = sqrtl(remaining);
            status = update_cone_envelope_lower(
                presolver, column, prefos_internal_outward_bound_cast(-limit, 1),
                changed);
            if (status != PREFOS_STATUS_OK) return status;
            status = update_cone_envelope_upper(
                presolver, column, prefos_internal_outward_bound_cast(limit, 0),
                changed);
            if (status != PREFOS_STATUS_OK) return status;
        }
    }
    return PREFOS_STATUS_OK;
}

static size_t psd_packed_position(size_t row, size_t column)
{
    if (column > row)
    {
        size_t swap = row;
        row = column;
        column = swap;
    }
    return row * (row + 1) / 2 + column;
}

static int fixed_psd_entry(const PreFOSPresolver *presolver, const PreFOSConeBlock *cone,
                           size_t row, size_t column, long double *value)
{
    size_t packed = psd_packed_position(row, column);
    int variable = cone->indices[packed];
    double lower = presolver->propagation_lower[variable];
    double upper = presolver->propagation_upper[variable];
    if (!isfinite(lower) || lower != upper) return 0;
    *value = (long double) lower;
    if (row != column) *value /= sqrtl(2.0L);
    return 1;
}

static PreFOSStatus propagate_psd_three_by_three(PreFOSPresolver *presolver,
                                              const PreFOSConeBlock *cone, int *changed)
{
    size_t i, j, k;
    if (cone->matrix_order > 32) return PREFOS_STATUS_OK;
    for (i = 0; i < cone->matrix_order; ++i)
    {
        for (j = i + 1; j < cone->matrix_order; ++j)
        {
            for (k = j + 1; k < cone->matrix_order; ++k)
            {
                size_t indices[3] = {i, j, k};
                long double matrix[3][3] = {{0.0L}};
                int fixed[3][3] = {{0}};
                size_t row, column, target;
                ++presolver->stats.psd_three_by_three_attempts;
                for (row = 0; row < 3; ++row)
                {
                    for (column = 0; column <= row; ++column)
                    {
                        fixed[row][column] = fixed[column][row] =
                            fixed_psd_entry(presolver, cone, indices[row],
                                            indices[column], &matrix[row][column]);
                        matrix[column][row] = matrix[row][column];
                    }
                }
                if (fixed[0][0] && fixed[1][1] && fixed[2][2] && fixed[0][1] &&
                    fixed[0][2] && fixed[1][2])
                {
                    long double determinant =
                        matrix[0][0] * matrix[1][1] * matrix[2][2] +
                        2.0L * matrix[0][1] * matrix[0][2] * matrix[1][2] -
                        matrix[0][0] * matrix[1][2] * matrix[1][2] -
                        matrix[1][1] * matrix[0][2] * matrix[0][2] -
                        matrix[2][2] * matrix[0][1] * matrix[0][1];
                    long double scale = 1.0L;
                    for (row = 0; row < 3; ++row)
                        for (column = 0; column < 3; ++column)
                            scale = fmaxl(scale, fabsl(matrix[row][column]));
                    if (determinant <
                        -((long double) presolver->settings.feasibility_tolerance +
                          512.0L * LDBL_EPSILON) *
                            scale * scale * scale)
                        return PREFOS_STATUS_PRIMAL_INFEASIBLE;
                }
                for (target = 0; target < 3; ++target)
                {
                    size_t first = (target + 1) % 3;
                    size_t second = (target + 2) % 3;
                    long double denominator, numerator, scale, candidate;
                    int diagonal_column;
                    PreFOSStatus status;
                    if (!fixed[first][first] || !fixed[second][second] ||
                        !fixed[first][second] || !fixed[target][first] ||
                        !fixed[target][second])
                        continue;
                    denominator = matrix[first][first] * matrix[second][second] -
                                  matrix[first][second] * matrix[first][second];
                    scale = fmaxl(
                        1.0L,
                        fmaxl(fabsl(matrix[first][first] * matrix[second][second]),
                              matrix[first][second] * matrix[first][second]));
                    if (denominator <=
                        prefos_internal_propagation_margin(presolver, scale))
                        continue;
                    ++presolver->stats.psd_schur_attempts;
                    numerator = matrix[second][second] * matrix[target][first] *
                                    matrix[target][first] +
                                matrix[first][first] * matrix[target][second] *
                                    matrix[target][second] -
                                2.0L * matrix[first][second] *
                                    matrix[target][first] * matrix[target][second];
                    candidate = numerator / denominator;
                    candidate -=
                        256.0L * LDBL_EPSILON * fmaxl(1.0L, fabsl(candidate));
                    candidate = fmaxl(0.0L, candidate);
                    diagonal_column = cone->indices[psd_packed_position(
                        indices[target], indices[target])];
                    {
                        size_t hits_before =
                            presolver->stats.tightened_cone_envelopes;
                        status = update_cone_envelope_lower(
                            presolver, diagonal_column,
                            prefos_internal_outward_bound_cast(candidate, 1), changed);
                        if (status != PREFOS_STATUS_OK) return status;
                        presolver->stats.psd_schur_bound_hits +=
                            presolver->stats.tightened_cone_envelopes - hits_before;
                    }
                }
            }
        }
    }
    return PREFOS_STATUS_OK;
}

static PreFOSStatus check_fixed_psd_windows(PreFOSPresolver *presolver,
                                         const PreFOSConeBlock *cone)
{
    size_t window, start;
    size_t maximum_window = cone->matrix_order < 8 ? cone->matrix_order : 8;
    double *point = (double *) calloc(36, sizeof(double));
    int *indices = (int *) prefos_internal_alloc_array(36, sizeof(int));
    if (!point || !indices)
    {
        free(point);
        free(indices);
        return PREFOS_STATUS_OUT_OF_MEMORY;
    }
    for (window = 0; window < 36; ++window) indices[window] = (int) window;
    for (window = 4; window <= maximum_window; ++window)
    {
        for (start = 0; start + window <= cone->matrix_order; ++start)
        {
            PreFOSConeBlock principal = {PREFOS_CONE_POSITIVE_SEMIDEFINITE,
                                      window * (window + 1) / 2, window, indices,
                                      0.0};
            size_t row, column, write = 0;
            int all_fixed = 1;
            long double minimum_eigenvalue, scale = 1.0L;
            PreFOSStatus status;
            for (row = 0; row < window && all_fixed; ++row)
            {
                for (column = 0; column <= row; ++column)
                {
                    size_t packed = psd_packed_position(start + row, start + column);
                    int variable = cone->indices[packed];
                    double lower = presolver->propagation_lower[variable];
                    double upper = presolver->propagation_upper[variable];
                    if (!isfinite(lower) || lower != upper)
                    {
                        all_fixed = 0;
                        break;
                    }
                    point[write++] = lower;
                    scale = fmaxl(scale, fabsl((long double) lower));
                }
            }
            if (!all_fixed) continue;
            ++presolver->stats.psd_fixed_window_checks;
            status = prefos_internal_psd_minimum_eigenvalue(
                &principal, point, presolver->settings.feasibility_tolerance,
                &minimum_eigenvalue);
            if (status != PREFOS_STATUS_OK)
            {
                free(point);
                free(indices);
                return status;
            }
            if (minimum_eigenvalue <
                -((long double) presolver->settings.feasibility_tolerance +
                  1024.0L * LDBL_EPSILON * (long double) window *
                      (long double) window) *
                    scale)
            {
                free(point);
                free(indices);
                return PREFOS_STATUS_PRIMAL_INFEASIBLE;
            }
        }
    }
    free(point);
    free(indices);
    return PREFOS_STATUS_OK;
}

static PreFOSStatus propagate_psd_cone(PreFOSPresolver *presolver,
                                    const PreFOSConeBlock *cone, int *changed)
{
    const long double sqrt_two = sqrtl(2.0L);
    size_t i, j;
    int all_fixed = 1;
    PreFOSStatus status;
    ++presolver->stats.psd_cones_processed;
    for (i = 0; i < cone->matrix_order; ++i)
    {
        size_t diagonal = i * (i + 1) / 2 + i;
        status = update_cone_envelope_lower(presolver, cone->indices[diagonal], 0.0,
                                            changed);
        if (status != PREFOS_STATUS_OK) return status;
    }

    for (i = 1; i < cone->matrix_order; ++i)
    {
        size_t diagonal_i = i * (i + 1) / 2 + i;
        int diagonal_i_column = cone->indices[diagonal_i];
        for (j = 0; j < i; ++j)
        {
            size_t diagonal_j = j * (j + 1) / 2 + j;
            size_t off_diagonal = i * (i + 1) / 2 + j;
            int diagonal_j_column = cone->indices[diagonal_j];
            int off_diagonal_column = cone->indices[off_diagonal];
            double upper_i = presolver->propagation_upper[diagonal_i_column];
            double upper_j = presolver->propagation_upper[diagonal_j_column];
            long double minimum_svec = minimum_absolute_value(
                presolver->propagation_lower[off_diagonal_column],
                presolver->propagation_upper[off_diagonal_column]);
            long double minimum_entry = minimum_svec / sqrt_two;
            size_t hits_before = presolver->stats.tightened_cone_envelopes;
            ++presolver->stats.psd_two_by_two_attempts;

            if ((upper_i == 0.0 || upper_j == 0.0) && minimum_entry > 0.0L)
                return PREFOS_STATUS_PRIMAL_INFEASIBLE;
            if (upper_i == 0.0 || upper_j == 0.0)
            {
                status = update_cone_envelope_lower(presolver, off_diagonal_column,
                                                    0.0, changed);
                if (status != PREFOS_STATUS_OK) return status;
                status = update_cone_envelope_upper(presolver, off_diagonal_column,
                                                    0.0, changed);
                if (status != PREFOS_STATUS_OK) return status;
            }
            else if (isfinite(upper_i) && isfinite(upper_j))
            {
                long double capacity = (long double) upper_i * upper_j;
                long double minimum_squared = minimum_entry * minimum_entry;
                long double limit = sqrt_two * sqrtl(fmaxl(0.0L, capacity));
                if (minimum_squared >
                    capacity + prefos_internal_propagation_margin(presolver, capacity))
                    return PREFOS_STATUS_PRIMAL_INFEASIBLE;
                status = update_cone_envelope_lower(
                    presolver, off_diagonal_column,
                    prefos_internal_outward_bound_cast(-limit, 1), changed);
                if (status != PREFOS_STATUS_OK) return status;
                status = update_cone_envelope_upper(
                    presolver, off_diagonal_column,
                    prefos_internal_outward_bound_cast(limit, 0), changed);
                if (status != PREFOS_STATUS_OK) return status;
            }
            if (isfinite(upper_j) && upper_j > 0.0)
            {
                long double implied = minimum_entry * minimum_entry / upper_j;
                status = update_cone_envelope_lower(
                    presolver, diagonal_i_column,
                    prefos_internal_outward_bound_cast(implied, 1), changed);
                if (status != PREFOS_STATUS_OK) return status;
            }
            if (isfinite(upper_i) && upper_i > 0.0)
            {
                long double implied = minimum_entry * minimum_entry / upper_i;
                status = update_cone_envelope_lower(
                    presolver, diagonal_j_column,
                    prefos_internal_outward_bound_cast(implied, 1), changed);
                if (status != PREFOS_STATUS_OK) return status;
            }
            presolver->stats.psd_two_by_two_bound_hits +=
                presolver->stats.tightened_cone_envelopes - hits_before;
        }
    }

    if (presolver->settings.psd_higher_order_propagation)
    {
        PreFOSTimestamp start, stop;
        prefos_internal_timer_now(&start);
        status = propagate_psd_three_by_three(presolver, cone, changed);
        if (status == PREFOS_STATUS_OK)
            status = check_fixed_psd_windows(presolver, cone);
        prefos_internal_timer_now(&stop);
        presolver->stats.psd_higher_order_milliseconds +=
            prefos_internal_timer_elapsed_milliseconds(&start, &stop);
        if (status != PREFOS_STATUS_OK) return status;
    }

    for (i = 0; i < cone->dimension; ++i)
    {
        int column = cone->indices[i];
        if (!prefos_internal_values_close(presolver->propagation_lower[column],
                                       presolver->propagation_upper[column],
                                       presolver->settings.feasibility_tolerance))
        {
            all_fixed = 0;
            break;
        }
    }
    if (all_fixed)
    {
        size_t point_size = 0;
        double *point;
        long double violation = 0.0L;
        long double scale = 1.0L;
        for (i = 0; i < cone->dimension; ++i)
            if ((size_t) cone->indices[i] >= point_size)
                point_size = (size_t) cone->indices[i] + 1;
        point = (double *) calloc(point_size, sizeof(double));
        if (point_size > 0 && !point) return PREFOS_STATUS_OUT_OF_MEMORY;
        for (i = 0; i < cone->dimension; ++i)
        {
            int column = cone->indices[i];
            point[column] =
                prefos_internal_safe_midpoint(presolver->propagation_lower[column],
                                           presolver->propagation_upper[column]);
            scale = fmaxl(scale, fabsl((long double) point[column]));
        }
        status = prefos_internal_evaluate_psd_violation(
            cone, point, presolver->settings.feasibility_tolerance, &violation);
        free(point);
        if (status != PREFOS_STATUS_OK) return status;
        if (violation > ((long double) presolver->settings.feasibility_tolerance +
                         1024.0L * LDBL_EPSILON * (long double) cone->matrix_order *
                             (long double) cone->matrix_order) *
                            scale)
            return PREFOS_STATUS_PRIMAL_INFEASIBLE;
    }
    return PREFOS_STATUS_OK;
}

PreFOSStatus prefos_internal_propagate_cone_block_envelopes(
    PreFOSPresolver *presolver, const PreFOSConeBlock *cone, double *lower,
    double *upper, int *changed)
{
    double *saved_lower = presolver->propagation_lower;
    double *saved_upper = presolver->propagation_upper;
    PreFOSStatus status;
    presolver->propagation_lower = lower;
    presolver->propagation_upper = upper;
    if (cone->type == PREFOS_CONE_NONNEGATIVE)
        status = propagate_nonnegative_cone(presolver, cone, changed);
    else if (cone->type == PREFOS_CONE_SECOND_ORDER)
        status = propagate_soc_cone(presolver, cone, changed);
    else if (cone->type == PREFOS_CONE_ROTATED_SECOND_ORDER)
        status = propagate_rsoc_cone(presolver, cone, changed);
    else if (cone->type == PREFOS_CONE_POSITIVE_SEMIDEFINITE)
    {
        PreFOSTimestamp start, stop;
        prefos_internal_timer_now(&start);
        status = propagate_psd_cone(presolver, cone, changed);
        prefos_internal_timer_now(&stop);
        presolver->stats.psd_propagation_milliseconds +=
            prefos_internal_timer_elapsed_milliseconds(&start, &stop);
    }
    else if (cone->type == PREFOS_CONE_EXPONENTIAL)
    {
        PreFOSTimestamp start, stop;
        prefos_internal_timer_now(&start);
        status = propagate_exponential_cone(presolver, cone, changed);
        prefos_internal_timer_now(&stop);
        presolver->stats.exponential_propagation_milliseconds +=
            prefos_internal_timer_elapsed_milliseconds(&start, &stop);
    }
    else
    {
        PreFOSTimestamp start, stop;
        prefos_internal_timer_now(&start);
        status = propagate_power_cone(presolver, cone, changed);
        prefos_internal_timer_now(&stop);
        presolver->stats.power_propagation_milliseconds +=
            prefos_internal_timer_elapsed_milliseconds(&start, &stop);
    }
    presolver->propagation_lower = saved_lower;
    presolver->propagation_upper = saved_upper;
    return status;
}

PreFOSStatus prefos_internal_propagate_cone_envelopes(PreFOSPresolver *presolver)
{
    int round;
    PreFOSStatus status;
    if (!presolver->settings.cone_propagation ||
        presolver->settings.max_cone_propagation_rounds == 0 ||
        (presolver->original.n_cones == 0 &&
         presolver->original.n_affine_cones == 0))
        return PREFOS_STATUS_OK;

    status = extract_singleton_cone_envelopes(presolver);
    if (status != PREFOS_STATUS_OK) return status;
    status = prefos_internal_materialize_affine_cone_bounds(presolver);
    if (status != PREFOS_STATUS_OK) return status;
    for (round = 0; round < presolver->settings.max_cone_propagation_rounds; ++round)
    {
        size_t k;
        int changed = 0;
        ++presolver->stats.cone_propagation_rounds;
        for (k = 0; k < presolver->original.n_cones; ++k)
        {
            const PreFOSConeBlock *cone = &presolver->original.cones[k];
            if (presolver->converted_affine_cones[k]) continue;
            status = prefos_internal_propagate_cone_block_envelopes(
                presolver, cone, presolver->propagation_lower,
                presolver->propagation_upper, &changed);
            if (status != PREFOS_STATUS_OK) return status;
        }
        status = prefos_internal_propagate_affine_cones(presolver, &changed);
        if (status != PREFOS_STATUS_OK) return status;
        if (!changed) break;
    }
    return PREFOS_STATUS_OK;
}

static int find_zero_upper_singleton_source(const PreFOSPresolver *presolver,
                                            int target_column, int require_exact)
{
    const PreFOSProblemData *problem = &presolver->original;
    size_t row;
    for (row = 0; row < problem->A.rows; ++row)
    {
        int p, column = -1;
        double coefficient = 0.0;
        size_t nonzeros = 0;
        long double implied_upper;
        for (p = problem->A.row_pointers[row]; p < problem->A.row_pointers[row + 1];
             ++p)
        {
            if (problem->A.values[p] == 0.0) continue;
            column = problem->A.column_indices[p];
            coefficient = problem->A.values[p];
            if (++nonzeros > 1) break;
        }
        if (nonzeros != 1 || column != target_column) continue;
        implied_upper =
            coefficient > 0.0
                ? (long double) presolver->working_constraint_upper[row] /
                      coefficient
                : (long double) presolver->working_constraint_lower[row] /
                      coefficient;
        if (isfinite(implied_upper) &&
            ((require_exact && implied_upper == 0.0L) ||
             (!require_exact &&
              fabsl(implied_upper) <=
                  prefos_internal_propagation_margin(presolver, implied_upper))))
            return (int) row;
    }
    return -1;
}

static int find_exact_zero_singleton_source(const PreFOSPresolver *presolver,
                                            int target_column)
{
    const PreFOSProblemData *problem = &presolver->original;
    size_t row;
    for (row = 0; row < problem->A.rows; ++row)
    {
        int p, column = -1;
        double coefficient = 0.0;
        size_t nonzeros = 0;
        long double implied_lower, implied_upper;
        for (p = problem->A.row_pointers[row]; p < problem->A.row_pointers[row + 1];
             ++p)
        {
            if (problem->A.values[p] == 0.0) continue;
            column = problem->A.column_indices[p];
            coefficient = problem->A.values[p];
            if (++nonzeros > 1) break;
        }
        if (nonzeros != 1 || column != target_column) continue;
        if (coefficient > 0.0)
        {
            implied_lower =
                (long double) presolver->working_constraint_lower[row] / coefficient;
            implied_upper =
                (long double) presolver->working_constraint_upper[row] / coefficient;
        }
        else
        {
            implied_lower =
                (long double) presolver->working_constraint_upper[row] / coefficient;
            implied_upper =
                (long double) presolver->working_constraint_lower[row] / coefficient;
        }
        if (implied_lower == 0.0L && implied_upper == 0.0L) return (int) row;
    }
    return -1;
}

static PreFOSStatus append_cone_transformation(PreFOSPresolver *presolver,
                                            size_t cone_index,
                                            PresolveConeTransformationType type)
{
    PresolveConeTransformationRecord record = {type, cone_index};
    return presolve_transformation_log_append_cone_transformation(
               &presolver->transformations, &record, NULL)
               ? PREFOS_STATUS_OK
               : PREFOS_STATUS_OUT_OF_MEMORY;
}

static PreFOSStatus mark_zero_cone_collapse(PreFOSPresolver *presolver, size_t cone_index)
{
    const PreFOSConeBlock *cone = &presolver->original.cones[cone_index];
    PreFOSStatus status =
        append_cone_transformation(presolver, cone_index, PRESOLVE_CONE_COLLAPSED);
    size_t i;
    if (status != PREFOS_STATUS_OK) return status;
    presolver->remove_cones[cone_index] = 1;
    ++presolver->stats.collapsed_cones;
    for (i = 0; i < cone->dimension; ++i)
    {
        int column = cone->indices[i];
        if (!presolver->is_fixed[column])
        {
            presolver->is_fixed[column] = 1;
            presolver->fixed_values[column] = 0.0;
            ++presolver->stats.fixed_cone_variables;
        }
    }
    return PREFOS_STATUS_OK;
}

static PreFOSStatus mark_rsoc_face_reduction(PreFOSPresolver *presolver, size_t cone_index,
                                          size_t zero_axis_position, int source_row)
{
    const PreFOSConeBlock *cone = &presolver->original.cones[cone_index];
    size_t survivor_position = zero_axis_position == 0 ? 1 : 0;
    PreFOSFacialReductionCertificate *certificate;
    PreFOSStatus status = append_cone_transformation(presolver, cone_index,
                                                  PRESOLVE_CONE_FACE_REDUCED);
    size_t i;
    if (status != PREFOS_STATUS_OK) return status;
    certificate = &presolver->facial_reductions[presolver->n_facial_reductions++];

    presolver->remove_cones[cone_index] = 1;
    presolver->cone_face_survivors[cone_index] = cone->indices[survivor_position];
    presolver->cone_face_box[cone->indices[survivor_position]] = 1;
    presolver->cone_face_box_lower[cone->indices[survivor_position]] = 0.0;
    presolver->cone_face_box_upper[cone->indices[survivor_position]] = INFINITY;
    presolver->cone_collapse_source_rows[cone->indices[zero_axis_position]] =
        source_row;
    ++presolver->stats.reduced_rsoc_faces;

    certificate->type =
        zero_axis_position == 0 ? PREFOS_FACE_RSOC_U_ZERO : PREFOS_FACE_RSOC_V_ZERO;
    certificate->source_row = source_row;
    certificate->zero_axis_column = cone->indices[zero_axis_position];
    certificate->surviving_axis_column = cone->indices[survivor_position];
    certificate->cone_dimension = cone->dimension;
    certificate->cone_indices = cone->indices;

    for (i = 0; i < cone->dimension; ++i)
    {
        int column;
        if (i == survivor_position) continue;
        column = cone->indices[i];
        presolver->is_fixed[column] = 1;
        presolver->fixed_values[column] = 0.0;
        ++presolver->stats.fixed_cone_variables;
    }
    return PREFOS_STATUS_OK;
}

static void expose_face_box(PreFOSPresolver *presolver, int column, double lower,
                            double upper)
{
    presolver->cone_face_box[column] = 1;
    presolver->cone_face_box_lower[column] = lower;
    presolver->cone_face_box_upper[column] = upper;
}

static void fix_face_column(PreFOSPresolver *presolver, int column)
{
    if (presolver->is_fixed[column]) return;
    presolver->is_fixed[column] = 1;
    presolver->fixed_values[column] = 0.0;
    ++presolver->stats.fixed_cone_variables;
}

static PreFOSStatus mark_exponential_face_reduction(PreFOSPresolver *presolver,
                                                 size_t cone_index, int zero_z,
                                                 int source_row)
{
    const PreFOSConeBlock *cone = &presolver->original.cones[cone_index];
    PreFOSFacialReductionCertificate *certificate;
    PreFOSStatus status = append_cone_transformation(presolver, cone_index,
                                                  PRESOLVE_CONE_FACE_REDUCED);
    if (status != PREFOS_STATUS_OK) return status;
    presolver->remove_cones[cone_index] = 1;
    expose_face_box(presolver, cone->indices[0], -INFINITY, 0.0);
    if (!zero_z) expose_face_box(presolver, cone->indices[2], 0.0, INFINITY);
    fix_face_column(presolver, cone->indices[1]);
    if (zero_z) fix_face_column(presolver, cone->indices[2]);
    presolver->cone_collapse_source_rows[cone->indices[zero_z ? 2 : 1]] = source_row;
    ++presolver->stats.reduced_exponential_faces;

    certificate = &presolver->facial_reductions[presolver->n_facial_reductions++];
    certificate->type =
        zero_z ? PREFOS_FACE_EXPONENTIAL_Z_ZERO : PREFOS_FACE_EXPONENTIAL_Y_ZERO;
    certificate->source_row = source_row;
    certificate->zero_axis_column = cone->indices[zero_z ? 2 : 1];
    certificate->surviving_axis_column = -1;
    certificate->cone_dimension = cone->dimension;
    certificate->cone_indices = cone->indices;
    return PREFOS_STATUS_OK;
}

static PreFOSStatus mark_power_face_reduction(PreFOSPresolver *presolver,
                                           size_t cone_index, size_t zero_position,
                                           int source_row)
{
    const PreFOSConeBlock *cone = &presolver->original.cones[cone_index];
    PreFOSFacialReductionCertificate *certificate;
    PreFOSFacialReductionType type;
    PreFOSStatus status = append_cone_transformation(presolver, cone_index,
                                                  PRESOLVE_CONE_FACE_REDUCED);
    if (status != PREFOS_STATUS_OK) return status;
    presolver->remove_cones[cone_index] = 1;
    if (zero_position == 0)
    {
        type = PREFOS_FACE_POWER_X_ZERO;
        fix_face_column(presolver, cone->indices[0]);
        fix_face_column(presolver, cone->indices[2]);
        expose_face_box(presolver, cone->indices[1], 0.0, INFINITY);
    }
    else if (zero_position == 1)
    {
        type = PREFOS_FACE_POWER_Y_ZERO;
        fix_face_column(presolver, cone->indices[1]);
        fix_face_column(presolver, cone->indices[2]);
        expose_face_box(presolver, cone->indices[0], 0.0, INFINITY);
    }
    else
    {
        type = PREFOS_FACE_POWER_Z_ZERO;
        fix_face_column(presolver, cone->indices[2]);
        expose_face_box(presolver, cone->indices[0], 0.0, INFINITY);
        expose_face_box(presolver, cone->indices[1], 0.0, INFINITY);
    }
    presolver->cone_collapse_source_rows[cone->indices[zero_position]] = source_row;
    ++presolver->stats.reduced_power_faces;

    certificate = &presolver->facial_reductions[presolver->n_facial_reductions++];
    certificate->type = type;
    certificate->source_row = source_row;
    certificate->zero_axis_column = cone->indices[zero_position];
    certificate->surviving_axis_column = -1;
    certificate->cone_dimension = cone->dimension;
    certificate->cone_indices = cone->indices;
    return PREFOS_STATUS_OK;
}

int prefos_internal_psd_matrix_index_is_removed(const PreFOSPSDFaceReduction *face,
                                             size_t matrix_index)
{
    size_t i;
    for (i = 0; i < face->n_removed; ++i)
        if (face->removed_matrix_indices[i] == (int) matrix_index) return 1;
    return 0;
}

static PreFOSStatus mark_psd_face_reduction(PreFOSPresolver *presolver, size_t cone_index)
{
    const PreFOSConeBlock *cone = &presolver->original.cones[cone_index];
    PreFOSPSDFaceReduction *face = &presolver->psd_face_reductions[cone_index];
    PreFOSFacialReductionCertificate *certificate;
    size_t i, j, removed_count = 0, write = 0;

    for (i = 0; i < cone->matrix_order; ++i)
    {
        size_t diagonal = i * (i + 1) / 2 + i;
        int column = cone->indices[diagonal];
        if (presolver->propagation_upper[column] == 0.0 &&
            find_zero_upper_singleton_source(presolver, column, 1) >= 0)
            ++removed_count;
    }
    if (removed_count == 0) return PREFOS_STATUS_OK;

    face->removed_matrix_indices =
        (int *) prefos_internal_alloc_array(removed_count, sizeof(int));
    face->source_rows = (int *) prefos_internal_alloc_array(removed_count, sizeof(int));
    if (!face->removed_matrix_indices || !face->source_rows)
    {
        free(face->removed_matrix_indices);
        free(face->source_rows);
        memset(face, 0, sizeof(*face));
        return PREFOS_STATUS_OUT_OF_MEMORY;
    }
    face->n_removed = removed_count;
    for (i = 0; i < cone->matrix_order; ++i)
    {
        size_t diagonal = i * (i + 1) / 2 + i;
        int column = cone->indices[diagonal];
        int source = presolver->propagation_upper[column] == 0.0
                         ? find_zero_upper_singleton_source(presolver, column, 1)
                         : -1;
        if (source < 0) continue;
        face->removed_matrix_indices[write] = (int) i;
        face->source_rows[write] = source;
        presolver->cone_collapse_source_rows[column] = source;
        ++write;
    }
    if (write != removed_count)
    {
        free(face->removed_matrix_indices);
        free(face->source_rows);
        memset(face, 0, sizeof(*face));
        return PREFOS_STATUS_NUMERICAL_ERROR;
    }
    {
        PreFOSStatus status = append_cone_transformation(presolver, cone_index,
                                                      PRESOLVE_CONE_FACE_REDUCED);
        if (status != PREFOS_STATUS_OK)
        {
            free(face->removed_matrix_indices);
            free(face->source_rows);
            memset(face, 0, sizeof(*face));
            return status;
        }
    }

    presolver->remove_cones[cone_index] = 1;
    ++presolver->stats.reduced_psd_faces;
    certificate = &presolver->facial_reductions[presolver->n_facial_reductions++];
    certificate->type = PREFOS_FACE_PSD_ZERO_DIAGONALS;
    certificate->source_row = -1;
    certificate->zero_axis_column = -1;
    certificate->surviving_axis_column = -1;
    certificate->cone_dimension = cone->dimension;
    certificate->cone_indices = cone->indices;
    certificate->matrix_order = cone->matrix_order;
    certificate->n_removed_matrix_indices = face->n_removed;
    certificate->removed_matrix_indices = face->removed_matrix_indices;
    certificate->source_rows = face->source_rows;

    for (i = 0; i < cone->matrix_order; ++i)
    {
        int remove_i = prefos_internal_psd_matrix_index_is_removed(face, i);
        for (j = 0; j <= i; ++j)
        {
            size_t packed;
            int column;
            if (!remove_i && !prefos_internal_psd_matrix_index_is_removed(face, j))
                continue;
            packed = i * (i + 1) / 2 + j;
            column = cone->indices[packed];
            if (presolver->is_fixed[column]) continue;
            presolver->is_fixed[column] = 1;
            presolver->fixed_values[column] = 0.0;
            ++presolver->stats.fixed_cone_variables;
        }
    }
    return PREFOS_STATUS_OK;
}

PreFOSStatus prefos_internal_detect_zero_cone_collapses(PreFOSPresolver *presolver)
{
    size_t k;
    for (k = 0; k < presolver->original.n_cones; ++k)
    {
        const PreFOSConeBlock *cone = &presolver->original.cones[k];
        size_t i;
        int can_collapse = 1;
        PreFOSStatus transformation_status;
        if (presolver->converted_affine_cones[k]) continue;
        if (cone->type == PREFOS_CONE_NONNEGATIVE)
        {
            for (i = 0; i < cone->dimension; ++i)
            {
                int column = cone->indices[i];
                int source;
                if (presolver->propagation_upper[column] != 0.0)
                {
                    can_collapse = 0;
                    break;
                }
                source = find_zero_upper_singleton_source(presolver, column, 0);
                if (source < 0)
                {
                    can_collapse = 0;
                    break;
                }
                presolver->cone_collapse_source_rows[column] = source;
            }
        }
        else if (cone->type == PREFOS_CONE_SECOND_ORDER)
        {
            int column = cone->indices[0];
            int source;
            can_collapse = presolver->propagation_upper[column] == 0.0;
            source = can_collapse
                         ? find_zero_upper_singleton_source(presolver, column, 0)
                         : -1;
            if (source < 0)
                can_collapse = 0;
            else
                presolver->cone_collapse_source_rows[column] = source;
        }
        else if (cone->type == PREFOS_CONE_ROTATED_SECOND_ORDER)
        {
            int u = cone->indices[0];
            int v = cone->indices[1];
            int source_u, source_v, exact_source_u = -1, exact_source_v = -1;
            source_u = presolver->propagation_upper[u] == 0.0
                           ? find_zero_upper_singleton_source(presolver, u, 0)
                           : -1;
            source_v = presolver->propagation_upper[v] == 0.0
                           ? find_zero_upper_singleton_source(presolver, v, 0)
                           : -1;
            if (source_u >= 0 && source_v >= 0)
            {
                presolver->cone_collapse_source_rows[u] = source_u;
                presolver->cone_collapse_source_rows[v] = source_v;
                transformation_status = mark_zero_cone_collapse(presolver, k);
                if (transformation_status != PREFOS_STATUS_OK)
                    return transformation_status;
            }
            else if (presolver->settings.rsoc_face_reduction)
            {
                exact_source_u =
                    presolver->propagation_upper[u] == 0.0
                        ? find_zero_upper_singleton_source(presolver, u, 1)
                        : -1;
                exact_source_v =
                    presolver->propagation_upper[v] == 0.0
                        ? find_zero_upper_singleton_source(presolver, v, 1)
                        : -1;
                if (exact_source_u >= 0)
                {
                    transformation_status =
                        mark_rsoc_face_reduction(presolver, k, 0, exact_source_u);
                    if (transformation_status != PREFOS_STATUS_OK)
                        return transformation_status;
                }
                else if (exact_source_v >= 0)
                {
                    transformation_status =
                        mark_rsoc_face_reduction(presolver, k, 1, exact_source_v);
                    if (transformation_status != PREFOS_STATUS_OK)
                        return transformation_status;
                }
            }
            continue;
        }
        else if (cone->type == PREFOS_CONE_POSITIVE_SEMIDEFINITE)
        {
            PreFOSStatus status;
            for (i = 0; i < cone->matrix_order; ++i)
            {
                size_t diagonal = i * (i + 1) / 2 + i;
                int column = cone->indices[diagonal];
                int source;
                if (presolver->propagation_upper[column] != 0.0)
                {
                    can_collapse = 0;
                    break;
                }
                source = find_zero_upper_singleton_source(presolver, column, 0);
                if (source < 0)
                {
                    can_collapse = 0;
                    break;
                }
                presolver->cone_collapse_source_rows[column] = source;
            }
            if (can_collapse)
            {
                transformation_status = mark_zero_cone_collapse(presolver, k);
                if (transformation_status != PREFOS_STATUS_OK)
                    return transformation_status;
            }
            else if (presolver->settings.psd_face_reduction)
            {
                status = mark_psd_face_reduction(presolver, k);
                if (status != PREFOS_STATUS_OK) return status;
            }
            continue;
        }
        else if (cone->type == PREFOS_CONE_EXPONENTIAL)
        {
            int y_source = -1, z_source = -1;
            if (!presolver->settings.exponential_face_reduction) continue;
            if (presolver->propagation_upper[cone->indices[2]] == 0.0)
                z_source =
                    find_zero_upper_singleton_source(presolver, cone->indices[2], 1);
            if (z_source >= 0)
                transformation_status =
                    mark_exponential_face_reduction(presolver, k, 1, z_source);
            else
            {
                if (presolver->propagation_upper[cone->indices[1]] == 0.0)
                    y_source = find_zero_upper_singleton_source(presolver,
                                                                cone->indices[1], 1);
                if (y_source < 0) continue;
                transformation_status =
                    mark_exponential_face_reduction(presolver, k, 0, y_source);
            }
            if (transformation_status != PREFOS_STATUS_OK) return transformation_status;
            continue;
        }
        else if (cone->type == PREFOS_CONE_POWER)
        {
            int x_source = -1, y_source = -1, z_source = -1;
            if (!presolver->settings.power_face_reduction) continue;
            if (presolver->propagation_upper[cone->indices[0]] == 0.0)
                x_source =
                    find_zero_upper_singleton_source(presolver, cone->indices[0], 1);
            if (presolver->propagation_upper[cone->indices[1]] == 0.0)
                y_source =
                    find_zero_upper_singleton_source(presolver, cone->indices[1], 1);
            if (x_source >= 0)
                transformation_status =
                    mark_power_face_reduction(presolver, k, 0, x_source);
            else if (y_source >= 0)
                transformation_status =
                    mark_power_face_reduction(presolver, k, 1, y_source);
            else
            {
                if (presolver->propagation_lower[cone->indices[2]] == 0.0 &&
                    presolver->propagation_upper[cone->indices[2]] == 0.0)
                    z_source = find_exact_zero_singleton_source(presolver,
                                                                cone->indices[2]);
                if (z_source < 0) continue;
                transformation_status =
                    mark_power_face_reduction(presolver, k, 2, z_source);
            }
            if (transformation_status != PREFOS_STATUS_OK) return transformation_status;
            continue;
        }
        else
            continue;
        if (can_collapse)
        {
            transformation_status = mark_zero_cone_collapse(presolver, k);
            if (transformation_status != PREFOS_STATUS_OK) return transformation_status;
        }
    }
    return PREFOS_STATUS_OK;
}
