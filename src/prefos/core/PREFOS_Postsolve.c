/*
 * Copyright 2026 Hongpei Li
 * SPDX-License-Identifier: Apache-2.0
 */

#include "PREFOS_Internal.h"
#include "cones/PREFOS_ExponentialCone.h"
#include "cones/PREFOS_PositiveSemidefiniteCone.h"
#include "cones/PREFOS_PowerCone.h"
#include "explorers/PREFOS_ConePropagation.h"

static int vector_is_finite(const double *values, size_t count)
{
    size_t i;
    for (i = 0; i < count; ++i)
        if (!isfinite(values[i])) return 0;
    return 1;
}

PreFOSStatus prefos_postsolve_primal(const PreFOSPresolver *presolver,
                               const double *reduced_x, double *original_x)
{
    size_t i;
    if (!presolver || !presolver->has_run ||
        (presolver->original.n > 0 && !original_x) ||
        (presolver->reduced.n > 0 && !reduced_x) ||
        !vector_is_finite(reduced_x, presolver->reduced.n))
    {
        return PREFOS_STATUS_INVALID_ARGUMENT;
    }
    for (i = 0; i < presolver->original.n; ++i)
    {
        int mapped = presolver->original_to_reduced[i];
        original_x[i] = mapped < 0 ? presolver->fixed_values[i] : reduced_x[mapped];
    }
    for (i = presolver->transformations.n_events; i > 0; --i)
    {
        const PresolveTransformationEvent *event =
            &presolver->transformations.events[i - 1];
        const PresolveColumnTransformationRecord *record;
        double recovered_value;
        unsigned char term_count;
        size_t term, start;
        if (event->type != PRESOLVE_TRANSFORMATION_COLUMN) continue;
        record =
            &presolver->transformations.column_transformations[event->record_index];
        if (record->type == PRESOLVE_COLUMN_FIXED && record->source_row == -1 &&
            record->column_tag < 0)
            continue;
        if (record->type != PRESOLVE_COLUMN_SUBSTITUTED || record->column < 0 ||
            (size_t) record->column >= presolver->original.n)
            return PREFOS_STATUS_NUMERICAL_ERROR;
        term_count = presolver->substitution_term_count[record->column];
        start = presolver->substitution_term_start[record->column];
        if (term_count == 0 || term_count > PREFOS_MAX_AGGREGATION_TERMS ||
            start > presolver->n_substitution_terms ||
            term_count > presolver->n_substitution_terms - start)
            return PREFOS_STATUS_NUMERICAL_ERROR;
        recovered_value = presolver->substitution_constant[record->column];
        for (term = 0; term < term_count; ++term)
        {
            int target = presolver->substitution_targets[start + term];
            double scale = presolver->substitution_scales[start + term];
            if (target < 0 || (size_t) target >= presolver->original.n ||
                !prefos_internal_safe_add_product(&recovered_value, scale,
                                               original_x[target]))
                return PREFOS_STATUS_NUMERICAL_ERROR;
        }
        original_x[record->column] = recovered_value;
    }
    return PREFOS_STATUS_OK;
}

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
    double objective_offset;
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
} PreFOSModelView;

static double violation_as_double(long double violation)
{
    if (violation <= 0.0L) return 0.0;
    if (violation > (long double) DBL_MAX) return INFINITY;
    return (double) violation;
}

static void update_max_violation(long double candidate, long double *maximum)
{
    if (isnan(candidate))
        *maximum = INFINITY;
    else if (candidate > *maximum)
        *maximum = candidate;
}

static PreFOSStatus evaluate_objective(const PreFOSModelView *model, const double *x,
                                    double *objective)
{
    long double value = (long double) model->objective_offset;
    size_t row;

    for (row = 0; row < model->n; ++row)
        value += (long double) model->c[row] * (long double) x[row];

    for (row = 0; row < model->Q->rows; ++row)
    {
        int p;
        for (p = model->Q->row_pointers[row]; p < model->Q->row_pointers[row + 1];
             ++p)
        {
            int column = model->Q->column_indices[p];
            long double term = (long double) model->Q->values[p] *
                               (long double) x[row] * (long double) x[column];
            if (model->q_storage == PREFOS_Q_FULL || column == (int) row)
                value += 0.5L * term;
            else
                value += term;
        }
    }

    for (row = 0; row < model->R->rows; ++row)
    {
        int p;
        long double product = 0.0L;
        for (p = model->R->row_pointers[row]; p < model->R->row_pointers[row + 1];
             ++p)
        {
            product += (long double) model->R->values[p] *
                       (long double) x[model->R->column_indices[p]];
        }
        value += 0.5L * (long double) model->D[row] * product * product;
    }

    if (!isfinite(value) || fabsl(value) > (long double) DBL_MAX)
        return PREFOS_STATUS_NUMERICAL_ERROR;
    *objective = (double) value;
    return PREFOS_STATUS_OK;
}

static PreFOSStatus evaluate_row_violation(const PreFOSModelView *model, const double *x,
                                        double *violation)
{
    long double maximum = 0.0L;
    size_t row;
    for (row = 0; row < model->A->rows; ++row)
    {
        int p;
        long double activity = 0.0L;
        for (p = model->A->row_pointers[row]; p < model->A->row_pointers[row + 1];
             ++p)
        {
            activity += (long double) model->A->values[p] *
                        (long double) x[model->A->column_indices[p]];
        }
        if (!isfinite(activity)) return PREFOS_STATUS_NUMERICAL_ERROR;
        if (isfinite(model->constraint_lower[row]))
            update_max_violation(
                (long double) model->constraint_lower[row] - activity, &maximum);
        if (isfinite(model->constraint_upper[row]))
            update_max_violation(
                activity - (long double) model->constraint_upper[row], &maximum);
    }
    *violation = violation_as_double(maximum);
    return PREFOS_STATUS_OK;
}

static void evaluate_box_violation(const PreFOSModelView *model, const double *x,
                                   double *violation)
{
    long double maximum = 0.0L;
    size_t i;
    for (i = 0; i < model->n_box; ++i)
    {
        int index = model->box_indices[i];
        if (isfinite(model->box_lower[i]))
            update_max_violation((long double) model->box_lower[i] -
                                     (long double) x[index],
                                 &maximum);
        if (isfinite(model->box_upper[i]))
            update_max_violation((long double) x[index] -
                                     (long double) model->box_upper[i],
                                 &maximum);
    }
    *violation = violation_as_double(maximum);
}

static PreFOSStatus evaluate_psd_face_dual_violation(const PreFOSConeBlock *cone,
                                                  const PreFOSPSDFaceReduction *face,
                                                  const double *negative_z,
                                                  double tolerance,
                                                  long double *violation)
{
    PreFOSConeBlock reduced_face;
    size_t reduced_order = cone->matrix_order - face->n_removed;
    size_t reduced_dimension = reduced_order * (reduced_order + 1) / 2;
    size_t row, column, write = 0;
    PreFOSStatus status;

    memset(&reduced_face, 0, sizeof(reduced_face));
    if (reduced_order == 0) return PREFOS_STATUS_NUMERICAL_ERROR;
    reduced_face.type = PREFOS_CONE_POSITIVE_SEMIDEFINITE;
    reduced_face.dimension = reduced_dimension;
    reduced_face.matrix_order = reduced_order;
    reduced_face.indices =
        (int *) prefos_internal_alloc_array(reduced_dimension, sizeof(int));
    if (!reduced_face.indices) return PREFOS_STATUS_OUT_OF_MEMORY;
    for (row = 0; row < cone->matrix_order; ++row)
    {
        if (prefos_internal_psd_matrix_index_is_removed(face, row)) continue;
        for (column = 0; column <= row; ++column)
        {
            size_t packed;
            if (prefos_internal_psd_matrix_index_is_removed(face, column)) continue;
            packed = row * (row + 1) / 2 + column;
            reduced_face.indices[write++] = cone->indices[packed];
        }
    }
    if (write != reduced_dimension)
    {
        free(reduced_face.indices);
        return PREFOS_STATUS_NUMERICAL_ERROR;
    }
    status = prefos_internal_evaluate_psd_violation(&reduced_face, negative_z,
                                                 tolerance, violation);
    free(reduced_face.indices);
    return status;
}

static PreFOSStatus evaluate_cone_violation(const PreFOSModelView *model, const double *x,
                                         double tolerance, int dual,
                                         double *violation)
{
    long double maximum = 0.0L;
    size_t k;
    for (k = 0; k < model->n_cones; ++k)
    {
        const PreFOSConeBlock *cone = &model->cones[k];
        size_t j;
        if (cone->type == PREFOS_CONE_NONNEGATIVE)
        {
            for (j = 0; j < cone->dimension; ++j)
                update_max_violation(-(long double) x[cone->indices[j]], &maximum);
        }
        else if (cone->type == PREFOS_CONE_SECOND_ORDER)
        {
            long double norm_squared = 0.0L;
            for (j = 1; j < cone->dimension; ++j)
            {
                long double value = (long double) x[cone->indices[j]];
                norm_squared += value * value;
            }
            update_max_violation(
                sqrtl(norm_squared) - (long double) x[cone->indices[0]], &maximum);
        }
        else if (cone->type == PREFOS_CONE_ROTATED_SECOND_ORDER)
        {
            long double u = (long double) x[cone->indices[0]];
            long double v = (long double) x[cone->indices[1]];
            long double norm_squared = 0.0L;
            long double radial_limit = 0.0L;
            for (j = 2; j < cone->dimension; ++j)
            {
                long double value = (long double) x[cone->indices[j]];
                norm_squared += value * value;
            }
            update_max_violation(-u, &maximum);
            update_max_violation(-v, &maximum);
            if (u >= 0.0L && v >= 0.0L) radial_limit = sqrtl(2.0L * u * v);
            update_max_violation(sqrtl(norm_squared) - radial_limit, &maximum);
        }
        else if (cone->type == PREFOS_CONE_POSITIVE_SEMIDEFINITE)
        {
            PreFOSStatus status =
                prefos_internal_evaluate_psd_violation(cone, x, tolerance, &maximum);
            if (status != PREFOS_STATUS_OK) return status;
        }
        else
        {
            long double cone_violation = 0.0L;
            PreFOSStatus status = cone->type == PREFOS_CONE_EXPONENTIAL
                                   ? prefos_internal_exponential_cone_violation(
                                         cone, x, dual, &cone_violation)
                                   : prefos_internal_power_cone_violation(
                                         cone, x, dual, &cone_violation);
            if (status != PREFOS_STATUS_OK) return status;
            update_max_violation(cone_violation, &maximum);
        }
    }
    if (model->n_affine_cones > 0)
    {
        double *point = NULL;
        int *indices = NULL;
        size_t affine_row = 0;
        if (!model->affine_cone_matrix || !model->affine_cone_offset ||
            !model->affine_cones)
            return PREFOS_STATUS_NUMERICAL_ERROR;
        point = (double *) prefos_internal_alloc_array(
            model->affine_cone_matrix->rows, sizeof(double));
        indices = (int *) prefos_internal_alloc_array(
            model->affine_cone_matrix->rows, sizeof(int));
        if (model->affine_cone_matrix->rows > 0 && (!point || !indices))
        {
            free(point);
            free(indices);
            return PREFOS_STATUS_OUT_OF_MEMORY;
        }
        for (k = 0; k < model->n_affine_cones; ++k)
        {
            const PreFOSAffineConeBlock *affine = &model->affine_cones[k];
            PreFOSConeBlock cone;
            PreFOSModelView point_model;
            double block_violation = 0.0;
            size_t j;
            PreFOSStatus status;
            if (affine->dimension > model->affine_cone_matrix->rows - affine_row)
            {
                free(point);
                free(indices);
                return PREFOS_STATUS_NUMERICAL_ERROR;
            }
            for (j = 0; j < affine->dimension; ++j)
            {
                size_t source_row = affine_row + j;
                long double value =
                    (long double) model->affine_cone_offset[source_row];
                int p;
                for (p = model->affine_cone_matrix->row_pointers[source_row];
                     p < model->affine_cone_matrix->row_pointers[source_row + 1];
                     ++p)
                    value +=
                        (long double) model->affine_cone_matrix->values[p] *
                        (long double)
                            x[model->affine_cone_matrix->column_indices[p]];
                if (!isfinite(value) || fabsl(value) > (long double) DBL_MAX)
                {
                    free(point);
                    free(indices);
                    return PREFOS_STATUS_NUMERICAL_ERROR;
                }
                point[j] = (double) value;
                indices[j] = (int) j;
            }
            cone = (PreFOSConeBlock){affine->type, affine->dimension,
                                  affine->matrix_order, indices,
                                  affine->power_alpha};
            memset(&point_model, 0, sizeof(point_model));
            point_model.n = affine->dimension;
            point_model.n_cones = 1;
            point_model.cones = &cone;
            status = evaluate_cone_violation(&point_model, point, tolerance, dual,
                                             &block_violation);
            if (status != PREFOS_STATUS_OK)
            {
                free(point);
                free(indices);
                return status;
            }
            update_max_violation((long double) block_violation, &maximum);
            affine_row += affine->dimension;
        }
        free(point);
        free(indices);
        if (affine_row != model->affine_cone_matrix->rows)
            return PREFOS_STATUS_NUMERICAL_ERROR;
    }
    *violation = violation_as_double(maximum);
    return PREFOS_STATUS_OK;
}

static PreFOSStatus evaluate_model(const PreFOSModelView *model, const double *x,
                                double tolerance, double *objective,
                                double *row_violation, double *box_violation,
                                double *cone_violation)
{
    size_t i;
    PreFOSStatus status;
    for (i = 0; i < model->n; ++i)
        if (!isfinite(x[i])) return PREFOS_STATUS_INVALID_ARGUMENT;

    status = evaluate_objective(model, x, objective);
    if (status != PREFOS_STATUS_OK) return status;
    status = evaluate_row_violation(model, x, row_violation);
    if (status != PREFOS_STATUS_OK) return status;
    evaluate_box_violation(model, x, box_violation);
    return evaluate_cone_violation(model, x, tolerance, 0, cone_violation);
}

static PreFOSStatus evaluate_gradient(const PreFOSModelView *model, const double *x,
                                   double *gradient)
{
    double *r_product;
    size_t row;
    for (row = 0; row < model->n; ++row) gradient[row] = model->c[row];

    for (row = 0; row < model->Q->rows; ++row)
    {
        int p;
        for (p = model->Q->row_pointers[row]; p < model->Q->row_pointers[row + 1];
             ++p)
        {
            int column = model->Q->column_indices[p];
            double value = model->Q->values[p];
            if (!prefos_internal_safe_add_product(&gradient[row], value, x[column]))
                return PREFOS_STATUS_NUMERICAL_ERROR;
            if (model->q_storage != PREFOS_Q_FULL && column != (int) row)
            {
                if (!prefos_internal_safe_add_product(&gradient[column], value, x[row]))
                    return PREFOS_STATUS_NUMERICAL_ERROR;
            }
        }
    }

    r_product = (double *) calloc(model->R->rows, sizeof(double));
    if (model->R->rows > 0 && !r_product) return PREFOS_STATUS_OUT_OF_MEMORY;
    for (row = 0; row < model->R->rows; ++row)
    {
        int p;
        for (p = model->R->row_pointers[row]; p < model->R->row_pointers[row + 1];
             ++p)
        {
            if (!prefos_internal_safe_add_product(&r_product[row], model->R->values[p],
                                               x[model->R->column_indices[p]]))
            {
                free(r_product);
                return PREFOS_STATUS_NUMERICAL_ERROR;
            }
        }
        if (!prefos_internal_safe_product(r_product[row], model->D[row],
                                       &r_product[row]))
        {
            free(r_product);
            return PREFOS_STATUS_NUMERICAL_ERROR;
        }
    }
    for (row = 0; row < model->R->rows; ++row)
    {
        int p;
        for (p = model->R->row_pointers[row]; p < model->R->row_pointers[row + 1];
             ++p)
        {
            int column = model->R->column_indices[p];
            if (!prefos_internal_safe_add_product(&gradient[column],
                                               model->R->values[p], r_product[row]))
            {
                free(r_product);
                return PREFOS_STATUS_NUMERICAL_ERROR;
            }
        }
    }
    free(r_product);
    return PREFOS_STATUS_OK;
}

static PreFOSStatus transfer_bound_dual(const PreFOSPresolver *presolver,
                                     const PresolveBoundChangeRecord *record,
                                     double tolerance, const double *x, double *y,
                                     double *z, int *changed)
{
    const PreFOSCsrMatrix *A = &presolver->original.A;
    int p, target_position = -1;
    double multiplier = z[record->column];
    double coefficient, delta, multiplier_tolerance = tolerance;
    int box_position = presolver->variable_to_box[record->column];

    if (box_position >= 0)
    {
        double original_bound = record->is_lower
                                    ? presolver->original.box_lower[box_position]
                                    : presolver->original.box_upper[box_position];
        if (isfinite(original_bound))
        {
            double slack = fabs(x[record->column] - original_bound);
            multiplier_tolerance /= fmax(1.0, slack);
        }
    }

    if (record->has_previous_bound &&
        prefos_internal_values_close(x[record->column], record->previous_bound,
                                  tolerance))
    {
        if ((record->is_lower && multiplier <= tolerance) ||
            (!record->is_lower && multiplier >= -tolerance))
            return PREFOS_STATUS_OK;
    }
    if (!prefos_internal_values_close(x[record->column], record->implied_bound,
                                   tolerance))
        return PREFOS_STATUS_OK;
    if ((record->is_lower && multiplier >= -multiplier_tolerance) ||
        (!record->is_lower && multiplier <= multiplier_tolerance))
        return PREFOS_STATUS_OK;

    for (p = A->row_pointers[record->row]; p < A->row_pointers[record->row + 1]; ++p)
    {
        if (A->column_indices[p] == record->column)
        {
            target_position = p;
            break;
        }
    }
    if (target_position < 0) return PREFOS_STATUS_NUMERICAL_ERROR;
    coefficient = A->values[target_position];
    if (coefficient == 0.0 ||
        !prefos_internal_safe_product(1.0 / coefficient, multiplier, &delta))
        return PREFOS_STATUS_NUMERICAL_ERROR;

    if (!prefos_internal_safe_add_product(&y[record->row], 1.0, delta))
        return PREFOS_STATUS_NUMERICAL_ERROR;
    for (p = A->row_pointers[record->row]; p < A->row_pointers[record->row + 1]; ++p)
    {
        int column = A->column_indices[p];
        if (column == record->column) continue;
        if (!prefos_internal_safe_add_product(&z[column], -A->values[p], delta))
            return PREFOS_STATUS_NUMERICAL_ERROR;
    }
    z[record->column] = 0.0;
    if (changed) *changed = 1;
    return PREFOS_STATUS_OK;
}

static PreFOSStatus replay_bound_change_block(const PreFOSPresolver *presolver,
                                           size_t first_event,
                                           size_t past_last_event, double tolerance,
                                           const double *original_x,
                                           double *original_y, double *original_z)
{
    size_t sweep;
    const size_t maximum_sweeps = 32;

    /* A transfer can activate another inferred bound in the same event block. */
    for (sweep = 0; sweep < maximum_sweeps; ++sweep)
    {
        size_t position;
        int changed = 0;
        for (position = past_last_event; position > first_event; --position)
        {
            const PresolveTransformationEvent *event =
                &presolver->transformations.events[position - 1];
            int event_changed = 0;
            PreFOSStatus status;
            if (event->type != PRESOLVE_TRANSFORMATION_BOUND_CHANGE)
                return PREFOS_STATUS_NUMERICAL_ERROR;
            status = transfer_bound_dual(
                presolver,
                &presolver->transformations.bound_changes[event->record_index],
                tolerance, original_x, original_y, original_z, &event_changed);
            if (status != PREFOS_STATUS_OK) return status;
            changed |= event_changed;
        }
        if (!changed) return PREFOS_STATUS_OK;
    }
    return PREFOS_STATUS_DUAL_RECOVERY_UNAVAILABLE;
}

static PreFOSStatus shift_collapsed_cone_pivot(const PreFOSPresolver *presolver,
                                            int column, long double decrease,
                                            double *y, double *z)
{
    const PreFOSCsrMatrix *A = &presolver->original.A;
    int row = presolver->cone_collapse_source_rows[column];
    int p;
    double coefficient = 0.0;
    long double delta;
    if (decrease <= 0.0L) return PREFOS_STATUS_OK;
    if (decrease > (long double) DBL_MAX) return PREFOS_STATUS_NUMERICAL_ERROR;
    if (row < 0) return PREFOS_STATUS_NUMERICAL_ERROR;
    for (p = A->row_pointers[row]; p < A->row_pointers[row + 1]; ++p)
    {
        if (A->column_indices[p] == column)
        {
            coefficient = A->values[p];
            break;
        }
    }
    if (coefficient == 0.0) return PREFOS_STATUS_NUMERICAL_ERROR;
    delta = decrease / (long double) coefficient;
    if (!isfinite(delta) || fabsl(delta) > (long double) DBL_MAX)
        return PREFOS_STATUS_NUMERICAL_ERROR;
    if (!prefos_internal_safe_add_product(&y[row], 1.0, (double) delta))
        return PREFOS_STATUS_NUMERICAL_ERROR;
    if (!prefos_internal_safe_add_product(&z[column], -1.0, (double) decrease))
        return PREFOS_STATUS_NUMERICAL_ERROR;
    return PREFOS_STATUS_OK;
}

static PreFOSStatus repair_collapsed_cone_dual(const PreFOSPresolver *presolver,
                                            size_t cone_index, double tolerance,
                                            double *y, double *z)
{
    const PreFOSConeBlock *cone;
    PreFOSStatus status;
    size_t i;
    if (cone_index >= presolver->original.n_cones ||
        !presolver->remove_cones[cone_index] ||
        presolver->cone_face_survivors[cone_index] >= 0 ||
        presolver->psd_face_reductions[cone_index].n_removed > 0)
        return PREFOS_STATUS_NUMERICAL_ERROR;
    cone = &presolver->original.cones[cone_index];
    if (cone->type == PREFOS_CONE_NONNEGATIVE)
    {
        for (i = 0; i < cone->dimension; ++i)
        {
            int column = cone->indices[i];
            status = shift_collapsed_cone_pivot(
                presolver, column, fmaxl(0.0L, (long double) z[column]), y, z);
            if (status != PREFOS_STATUS_OK) return status;
        }
    }
    else if (cone->type == PREFOS_CONE_SECOND_ORDER)
    {
        int t_column = cone->indices[0];
        long double norm_squared = 0.0L;
        long double required_t;
        for (i = 1; i < cone->dimension; ++i)
        {
            long double value = (long double) z[cone->indices[i]];
            norm_squared += value * value;
        }
        required_t = -sqrtl(norm_squared);
        status = shift_collapsed_cone_pivot(
            presolver, t_column, fmaxl(0.0L, (long double) z[t_column] - required_t),
            y, z);
        if (status != PREFOS_STATUS_OK) return status;
    }
    else if (cone->type == PREFOS_CONE_ROTATED_SECOND_ORDER)
    {
        int u_column = cone->indices[0];
        int v_column = cone->indices[1];
        long double norm_squared = 0.0L;
        long double minimum_dual_diagonal;
        long double current_u = -(long double) z[u_column];
        long double current_v = -(long double) z[v_column];
        for (i = 2; i < cone->dimension; ++i)
        {
            long double value = (long double) z[cone->indices[i]];
            norm_squared += value * value;
        }
        minimum_dual_diagonal = sqrtl(norm_squared / 2.0L);
        status = shift_collapsed_cone_pivot(
            presolver, u_column, fmaxl(0.0L, minimum_dual_diagonal - current_u), y,
            z);
        if (status != PREFOS_STATUS_OK) return status;
        status = shift_collapsed_cone_pivot(
            presolver, v_column, fmaxl(0.0L, minimum_dual_diagonal - current_v), y,
            z);
        if (status != PREFOS_STATUS_OK) return status;
    }
    else if (cone->type == PREFOS_CONE_POSITIVE_SEMIDEFINITE)
    {
        double *negative_z =
            (double *) calloc(presolver->original.n, sizeof(double));
        long double violation = 0.0L;
        if (presolver->original.n > 0 && !negative_z)
            return PREFOS_STATUS_OUT_OF_MEMORY;
        for (i = 0; i < cone->dimension; ++i)
        {
            int column = cone->indices[i];
            negative_z[column] = -z[column];
        }
        status = prefos_internal_evaluate_psd_violation(cone, negative_z, tolerance,
                                                     &violation);
        free(negative_z);
        if (status != PREFOS_STATUS_OK) return status;
        for (i = 0; i < cone->matrix_order; ++i)
        {
            size_t diagonal = i * (i + 1) / 2 + i;
            status = shift_collapsed_cone_pivot(presolver, cone->indices[diagonal],
                                                violation, y, z);
            if (status != PREFOS_STATUS_OK) return status;
        }
    }
    else
        return PREFOS_STATUS_NUMERICAL_ERROR;
    return PREFOS_STATUS_OK;
}

static PreFOSStatus
replay_row_transformation(const PresolveRowTransformationRecord *record,
                          double tolerance, const double *original_x,
                          double *original_y)
{
    long double activity = 0.0L;
    size_t position;
    double row_dual;
    int correct_sign;

    if (record->type == PRESOLVE_ROW_DELETED)
    {
        original_y[record->row] = record->dual_value;
        return PREFOS_STATUS_OK;
    }
    if (record->type != PRESOLVE_ROW_LOWER_CHANGED &&
        record->type != PRESOLVE_ROW_UPPER_CHANGED)
        return PREFOS_STATUS_NUMERICAL_ERROR;

    row_dual = original_y[record->row];
    correct_sign =
        record->type == PRESOLVE_ROW_LOWER_CHANGED ? row_dual < 0.0 : row_dual > 0.0;
    if (!correct_sign) return PREFOS_STATUS_OK;
    for (position = 0; position < record->length; ++position)
        activity += (long double) record->coefficients[position] *
                    (long double) original_x[record->indices[position]];
    if (!isfinite(activity) ||
        !prefos_internal_values_close((double) activity, record->new_side, tolerance))
        return PREFOS_STATUS_OK;
    if (!prefos_internal_safe_add_product(&original_y[record->source_row],
                                       record->ratio, row_dual))
        return PREFOS_STATUS_NUMERICAL_ERROR;
    original_y[record->row] = 0.0;
    return PREFOS_STATUS_OK;
}

static PreFOSStatus
replay_cone_transformation(const PreFOSPresolver *presolver,
                           const PresolveConeTransformationRecord *record,
                           double tolerance, double *original_y, double *original_z)
{
    if (record->type == PRESOLVE_CONE_FACE_REDUCED) return PREFOS_STATUS_OK;
    if (record->type != PRESOLVE_CONE_COLLAPSED) return PREFOS_STATUS_NUMERICAL_ERROR;
    return repair_collapsed_cone_dual(presolver, record->cone_index, tolerance,
                                      original_y, original_z);
}

static PreFOSStatus
replay_affine_face_column(const PreFOSPresolver *presolver,
                          const PresolveColumnTransformationRecord *record,
                          double *pre_face_affine_z, double *original_y,
                          double *original_z)
{
    long double stationarity;
    long double multiplier;
    size_t affine_row, position;
    if ((record->type != PRESOLVE_COLUMN_FIXED &&
         record->type != PRESOLVE_COLUMN_SUBSTITUTED) ||
        record->source_row != -1 || record->column_tag >= 0 ||
        record->column < 0 ||
        (size_t) record->column >= presolver->original.n || record->rhs == 0.0 ||
        !pre_face_affine_z)
        return PREFOS_STATUS_NUMERICAL_ERROR;
    affine_row = (size_t) (-(long long) record->column_tag - 1);
    if (affine_row >= presolver->n_pre_face_affine_rows)
        return PREFOS_STATUS_NUMERICAL_ERROR;
    stationarity = (long double) record->objective_coefficient;
    for (position = 0; position < record->length; ++position)
    {
        int row = record->indices[position];
        if (row < 0 || (size_t) row >= presolver->original.A.rows)
            return PREFOS_STATUS_NUMERICAL_ERROR;
        stationarity += (long double) record->coefficients[position] *
                        (long double) original_y[row];
    }
    for (position = 0; position < record->affine_length; ++position)
    {
        int row = record->affine_indices[position];
        if (row < 0 || (size_t) row >= presolver->n_pre_face_affine_rows)
            return PREFOS_STATUS_NUMERICAL_ERROR;
        stationarity += (long double) record->affine_coefficients[position] *
                        (long double) pre_face_affine_z[row];
    }
    multiplier = -stationarity / (long double) record->rhs;
    if (!isfinite(multiplier) || fabsl(multiplier) > (long double) DBL_MAX ||
        !prefos_internal_safe_add_product(&pre_face_affine_z[affine_row], 1.0,
                                       (double) multiplier))
        return PREFOS_STATUS_NUMERICAL_ERROR;
    original_z[record->column] = 0.0;
    return PREFOS_STATUS_OK;
}

static PreFOSStatus
replay_column_transformation(const PreFOSPresolver *presolver,
                             const PresolveColumnTransformationRecord *record,
                             double *pre_face_affine_z, double *original_y,
                             double *original_z)
{
    long double stationarity;
    double source_coefficient = 0.0;
    size_t position;

    if (record->source_row == -1 && record->column_tag < 0)
        return replay_affine_face_column(presolver, record, pre_face_affine_z,
                                         original_y, original_z);
    if (record->type != PRESOLVE_COLUMN_SUBSTITUTED || record->column < 0 ||
        record->source_row < 0 || (size_t) record->column >= presolver->original.n ||
        (size_t) record->source_row >= presolver->original.A.rows)
        return PREFOS_STATUS_NUMERICAL_ERROR;
    if (presolver->affine_aggregation_source_rows[record->column] >= 0)
        return PREFOS_STATUS_OK;
    stationarity = (long double) record->objective_coefficient;
    for (position = 0; position < record->length; ++position)
    {
        int row = record->indices[position];
        double coefficient = record->coefficients[position];
        if (row < 0 || (size_t) row >= presolver->original.A.rows)
            return PREFOS_STATUS_NUMERICAL_ERROR;
        if (row == record->source_row)
            source_coefficient = coefficient;
        else
            stationarity +=
                (long double) coefficient * (long double) original_y[row];
    }
    if (source_coefficient == 0.0 || !isfinite(stationarity) ||
        fabsl(stationarity / (long double) source_coefficient) >
            (long double) DBL_MAX)
        return PREFOS_STATUS_NUMERICAL_ERROR;
    original_y[record->source_row] =
        (double) (-stationarity / (long double) source_coefficient);
    original_z[record->column] = 0.0;
    return PREFOS_STATUS_OK;
}

static PreFOSStatus add_affine_stationarity(const PreFOSModelView *model,
                                         const double *affine_z,
                                         double *stationarity)
{
    size_t row;
    if (model->affine_cone_matrix->rows == 0) return PREFOS_STATUS_OK;
    if (!affine_z) return PREFOS_STATUS_INVALID_ARGUMENT;
    for (row = 0; row < model->affine_cone_matrix->rows; ++row)
    {
        int p;
        for (p = model->affine_cone_matrix->row_pointers[row];
             p < model->affine_cone_matrix->row_pointers[row + 1]; ++p)
            if (!prefos_internal_safe_add_product(
                    &stationarity[model->affine_cone_matrix->column_indices[p]],
                    model->affine_cone_matrix->values[p], affine_z[row]))
                return PREFOS_STATUS_NUMERICAL_ERROR;
    }
    return PREFOS_STATUS_OK;
}

static PreFOSStatus recover_fixed_box_normals(const PreFOSPresolver *presolver,
                                           const PreFOSModelView *model,
                                           const double *x, const double *y,
                                           const double *affine_z, double *z)
{
    double *stationarity =
        (double *) prefos_internal_alloc_array(model->n, sizeof(double));
    size_t row, column;
    PreFOSStatus status;
    if (model->n > 0 && !stationarity) return PREFOS_STATUS_OUT_OF_MEMORY;
    status = evaluate_gradient(model, x, stationarity);
    if (status != PREFOS_STATUS_OK) goto cleanup;
    for (row = 0; row < model->A->rows; ++row)
    {
        int p;
        for (p = model->A->row_pointers[row];
             p < model->A->row_pointers[row + 1]; ++p)
            if (!prefos_internal_safe_add_product(
                    &stationarity[model->A->column_indices[p]],
                    model->A->values[p], y[row]))
            {
                status = PREFOS_STATUS_NUMERICAL_ERROR;
                goto cleanup;
            }
    }
    status = add_affine_stationarity(model, affine_z, stationarity);
    if (status != PREFOS_STATUS_OK) goto cleanup;
    for (column = 0; column < model->n; ++column)
        if (presolver->is_fixed[column] &&
            !presolver->affine_face_eliminated_columns[column] &&
            presolver->variable_to_box[column] >= 0)
            z[column] = -stationarity[column];

cleanup:
    free(stationarity);
    return status;
}

static PreFOSStatus transfer_affine_bound_duals(
    const PreFOSPresolver *presolver, double tolerance, const double *original_x,
    double *original_y, double *original_z, double *original_affine_z)
{
    size_t i;
    for (i = 0; i < presolver->n_affine_bound_certificates; ++i)
    {
        const PreFOSAffineBoundCertificate *certificate =
            &presolver->affine_bound_certificates[i];
        double multiplier, delta;
        long double coordinate_value, coordinate_scale, coordinate_tolerance;
        if (certificate->column < 0 ||
            (size_t) certificate->column >= presolver->original.n ||
            certificate->affine_row >=
                presolver->n_pre_face_affine_rows ||
            certificate->coefficient == 0.0 ||
            !isfinite(certificate->coefficient) ||
            !isfinite(certificate->offset) ||
            !isfinite(certificate->implied_bound))
            return PREFOS_STATUS_NUMERICAL_ERROR;
        if (!prefos_internal_values_close(original_x[certificate->column],
                                       certificate->implied_bound, tolerance))
            continue;
        coordinate_value =
            (long double) certificate->coefficient *
                (long double) original_x[certificate->column] +
            (long double) certificate->offset;
        coordinate_scale =
            fmaxl(1.0L,
                  fmaxl(fabsl((long double) certificate->coefficient *
                              (long double) original_x[certificate->column]),
                        fabsl((long double) certificate->offset)));
        coordinate_tolerance =
            ((long double) tolerance + 16.0L * LDBL_EPSILON) * coordinate_scale;
        if (!isfinite(coordinate_value) ||
            fabsl(coordinate_value) > coordinate_tolerance)
            continue;
        multiplier = original_z[certificate->column];
        if ((certificate->is_lower && multiplier >= -tolerance) ||
            (!certificate->is_lower && multiplier <= tolerance))
            continue;
        if (!prefos_internal_safe_product(1.0 / certificate->coefficient,
                                       multiplier, &delta) ||
            delta > tolerance)
            return PREFOS_STATUS_NUMERICAL_ERROR;

        if (certificate->generated_column < 0)
        {
            if (certificate->affine_row >=
                    presolver->original.affine_cone_matrix.rows ||
                !original_affine_z ||
                !prefos_internal_safe_add_product(
                    &original_affine_z[certificate->affine_row], 1.0, delta))
                return PREFOS_STATUS_NUMERICAL_ERROR;
        }
        else
        {
            int generated_column = certificate->generated_column;
            int source_row;
            double pivot, row_delta;
            if (generated_column < 0 ||
                (size_t) generated_column >= presolver->original.n ||
                certificate->affine_row <
                    presolver->original.affine_cone_matrix.rows)
                return PREFOS_STATUS_NUMERICAL_ERROR;
            source_row =
                presolver->affine_aggregation_source_rows[generated_column];
            pivot = presolver->affine_aggregation_pivots[generated_column];
            if (source_row < 0 ||
                (size_t) source_row >= presolver->original.A.rows ||
                pivot == 0.0 ||
                !prefos_internal_safe_product(-1.0 / pivot, delta, &row_delta) ||
                !prefos_internal_safe_add_product(&original_z[generated_column], 1.0,
                                               delta) ||
                !prefos_internal_safe_add_product(&original_y[source_row], 1.0,
                                               row_delta))
                return PREFOS_STATUS_NUMERICAL_ERROR;
        }
        if (!prefos_internal_safe_add_product(&original_z[certificate->column], -1.0,
                                           multiplier))
            return PREFOS_STATUS_NUMERICAL_ERROR;
    }
    return PREFOS_STATUS_OK;
}

static PreFOSStatus postsolve_primal_dual_internal(
    const PreFOSPresolver *presolver, const double *reduced_x, const double *reduced_y,
    const double *reduced_z, const double *reduced_affine_z, double tolerance,
    double *original_x, double *original_y, double *original_z,
    double *original_affine_z, int affine_enabled)
{
    PreFOSModelView original;
    double *gradient, *pre_face_affine_z = NULL;
    size_t i, row, affine_row;
    PreFOSStatus status;

    if (!presolver || !presolver->has_run || !isfinite(tolerance) ||
        tolerance < 0.0 ||
        (presolver->original.n > 0 && (!original_x || !original_z)) ||
        (presolver->original.A.rows > 0 && !original_y) ||
        (presolver->reduced.n > 0 && (!reduced_x || !reduced_z)) ||
        (presolver->reduced.A.rows > 0 && !reduced_y) ||
        !vector_is_finite(reduced_x, presolver->reduced.n) ||
        !vector_is_finite(reduced_y, presolver->reduced.A.rows) ||
        !vector_is_finite(reduced_z, presolver->reduced.n))
        return PREFOS_STATUS_INVALID_ARGUMENT;
    if (!affine_enabled && (presolver->original.n_affine_cones > 0 ||
                            presolver->reduced.n_affine_cones > 0))
        return PREFOS_STATUS_DUAL_RECOVERY_UNAVAILABLE;
    if ((presolver->reduced.affine_cone_matrix.rows > 0 &&
         (!reduced_affine_z ||
          !vector_is_finite(reduced_affine_z,
                            presolver->reduced.affine_cone_matrix.rows))) ||
        (presolver->original.affine_cone_matrix.rows > 0 && !original_affine_z))
        return PREFOS_STATUS_INVALID_ARGUMENT;

    status = prefos_postsolve_primal(presolver, reduced_x, original_x);
    if (status != PREFOS_STATUS_OK) return status;
    for (row = 0; row < presolver->original.A.rows; ++row)
    {
        int mapped = presolver->original_to_reduced_rows[row];
        original_y[row] = mapped < 0 ? 0.0 : reduced_y[mapped];
    }
    for (i = 0; i < presolver->original.n; ++i)
    {
        int mapped = presolver->original_to_reduced[i];
        original_z[i] = mapped < 0 ? 0.0 : reduced_z[mapped];
    }
    if (presolver->n_pre_face_affine_rows > 0 &&
        !presolver->affine_pre_to_reduced_rows)
        return PREFOS_STATUS_NUMERICAL_ERROR;
    for (affine_row = 0;
         affine_row < presolver->original.affine_cone_matrix.rows; ++affine_row)
    {
        int mapped;
        if (affine_row >= presolver->n_pre_face_affine_rows)
            return PREFOS_STATUS_NUMERICAL_ERROR;
        mapped = presolver->affine_pre_to_reduced_rows[affine_row];
        if (mapped < -1 ||
            (mapped >= 0 &&
             (size_t) mapped >= presolver->reduced.affine_cone_matrix.rows))
            return PREFOS_STATUS_NUMERICAL_ERROR;
        original_affine_z[affine_row] =
            mapped < 0 ? 0.0 : reduced_affine_z[mapped];
    }

    original = (PreFOSModelView){presolver->original.n,
                              &presolver->original.A,
                              presolver->original.constraint_lower,
                              presolver->original.constraint_upper,
                              &presolver->original.Q,
                              presolver->original.q_storage,
                              &presolver->original.R,
                              presolver->original.D,
                              presolver->original.c,
                              presolver->original.objective_offset,
                              presolver->original.n_box,
                              presolver->original.box_indices,
                              presolver->original.box_lower,
                              presolver->original.box_upper,
                              presolver->original.n_cones,
                              presolver->original.cones,
                              &presolver->original.affine_cone_matrix,
                              presolver->original.affine_cone_offset,
                              presolver->original.n_affine_cones,
                              presolver->original.affine_cones};
    gradient = (double *) prefos_internal_alloc_array(original.n, sizeof(double));
    if (original.n > 0 && !gradient) return PREFOS_STATUS_OUT_OF_MEMORY;
    status = evaluate_gradient(&original, original_x, gradient);
    if (status != PREFOS_STATUS_OK)
    {
        free(gradient);
        return status;
    }
    for (row = 0; row < original.A->rows; ++row)
    {
        int p;
        for (p = original.A->row_pointers[row];
             p < original.A->row_pointers[row + 1]; ++p)
        {
            if (!prefos_internal_safe_add_product(
                    &gradient[original.A->column_indices[p]], original.A->values[p],
                    original_y[row]))
            {
                free(gradient);
                return PREFOS_STATUS_NUMERICAL_ERROR;
            }
        }
    }
    status = add_affine_stationarity(&original, original_affine_z, gradient);
    if (status != PREFOS_STATUS_OK)
    {
        free(gradient);
        return status;
    }
    for (i = 0; i < original.n; ++i)
        if (presolver->is_fixed[i] &&
            !presolver->affine_face_eliminated_columns[i] &&
            presolver->affine_aggregation_source_rows[i] < 0)
            original_z[i] = -gradient[i];
    free(gradient);

    pre_face_affine_z = (double *) prefos_internal_alloc_array(
        presolver->n_pre_face_affine_rows, sizeof(double));
    if (presolver->n_pre_face_affine_rows > 0 && !pre_face_affine_z)
        return PREFOS_STATUS_OUT_OF_MEMORY;
    for (affine_row = 0; affine_row < presolver->n_pre_face_affine_rows;
         ++affine_row)
    {
        int mapped = presolver->affine_pre_to_reduced_rows[affine_row];
        if (mapped < -1 ||
            (mapped >= 0 &&
             (size_t) mapped >= presolver->reduced.affine_cone_matrix.rows))
        {
            status = PREFOS_STATUS_NUMERICAL_ERROR;
            goto affine_cleanup;
        }
        pre_face_affine_z[affine_row] =
            mapped < 0 ? 0.0 : reduced_affine_z[mapped];
    }

    i = presolver->transformations.n_events;
    while (i > 0)
    {
        const PresolveTransformationEvent *event =
            &presolver->transformations.events[i - 1];
        if (event->type == PRESOLVE_TRANSFORMATION_BOUND_CHANGE)
        {
            size_t first = i - 1;
            while (first > 0 && presolver->transformations.events[first - 1].type ==
                                    PRESOLVE_TRANSFORMATION_BOUND_CHANGE)
                --first;
            status = replay_bound_change_block(presolver, first, i, tolerance,
                                               original_x, original_y, original_z);
            i = first;
        }
        else if (event->type == PRESOLVE_TRANSFORMATION_ROW)
        {
            status = replay_row_transformation(
                &presolver->transformations.row_transformations[event->record_index],
                tolerance, original_x, original_y);
            --i;
        }
        else if (event->type == PRESOLVE_TRANSFORMATION_CONE)
        {
            status = replay_cone_transformation(
                presolver,
                &presolver->transformations
                     .cone_transformations[event->record_index],
                tolerance, original_y, original_z);
            --i;
        }
        else if (event->type == PRESOLVE_TRANSFORMATION_COLUMN)
        {
            status = replay_column_transformation(
                presolver,
                &presolver->transformations
                     .column_transformations[event->record_index],
                pre_face_affine_z, original_y, original_z);
            --i;
        }
        else
        {
            status = PREFOS_STATUS_NUMERICAL_ERROR;
            --i;
        }
        if (status != PREFOS_STATUS_OK) goto affine_cleanup;
    }
    for (affine_row = 0;
         affine_row < presolver->original.affine_cone_matrix.rows; ++affine_row)
        original_affine_z[affine_row] = pre_face_affine_z[affine_row];
    affine_row = presolver->original.affine_cone_matrix.rows;
    for (i = 0; i < presolver->original.n_cones; ++i)
    {
        const PreFOSConeBlock *cone = &presolver->original.cones[i];
        size_t coordinate;
        if (!presolver->converted_affine_cones[i]) continue;
        if (affine_row > presolver->n_pre_face_affine_rows ||
            cone->dimension > presolver->n_pre_face_affine_rows - affine_row)
        {
            status = PREFOS_STATUS_NUMERICAL_ERROR;
            goto affine_cleanup;
        }
        for (coordinate = 0; coordinate < cone->dimension; ++coordinate)
        {
            int column = cone->indices[coordinate];
            int source_row = presolver->affine_aggregation_source_rows[column];
            double pivot = presolver->affine_aggregation_pivots[column];
            double cone_normal;
            if (affine_row >= presolver->n_pre_face_affine_rows)
            {
                status = PREFOS_STATUS_NUMERICAL_ERROR;
                goto affine_cleanup;
            }
            cone_normal = pre_face_affine_z[affine_row++];
            if (source_row < 0 ||
                (size_t) source_row >= presolver->original.A.rows || pivot == 0.0)
            {
                status = PREFOS_STATUS_NUMERICAL_ERROR;
                goto affine_cleanup;
            }
            original_z[column] = cone_normal;
            original_y[source_row] = -cone_normal / pivot;
            if (!isfinite(original_y[source_row]))
            {
                status = PREFOS_STATUS_NUMERICAL_ERROR;
                goto affine_cleanup;
            }
        }
    }
    if (affine_row != presolver->n_pre_face_affine_rows)
    {
        status = PREFOS_STATUS_NUMERICAL_ERROR;
        goto affine_cleanup;
    }
    status = recover_fixed_box_normals(presolver, &original, original_x, original_y,
                                       original_affine_z, original_z);
    if (status == PREFOS_STATUS_OK)
        status = transfer_affine_bound_duals(
            presolver, tolerance, original_x, original_y, original_z,
            original_affine_z);

affine_cleanup:
    free(pre_face_affine_z);
    return status;
}

PreFOSStatus prefos_postsolve_primal_dual(const PreFOSPresolver *presolver,
                                    const double *reduced_x, const double *reduced_y,
                                    const double *reduced_z, double tolerance,
                                    double *original_x, double *original_y,
                                    double *original_z)
{
    if (presolver && presolver->has_run &&
        (presolver->n_facial_reductions > 0 ||
         presolver->n_affine_face_substitutions > 0))
        return PREFOS_STATUS_DUAL_RECOVERY_UNAVAILABLE;
    return postsolve_primal_dual_internal(
        presolver, reduced_x, reduced_y, reduced_z, NULL, tolerance, original_x,
        original_y, original_z, NULL, 0);
}

PreFOSStatus prefos_postsolve_extended_dual(const PreFOSPresolver *presolver,
                                      const double *reduced_x,
                                      const double *reduced_y,
                                      const double *reduced_z, double tolerance,
                                      double *original_x, double *original_y,
                                      double *original_z)
{
    return postsolve_primal_dual_internal(
        presolver, reduced_x, reduced_y, reduced_z, NULL, tolerance, original_x,
        original_y, original_z, NULL, 0);
}

PreFOSStatus prefos_postsolve_full_primal_dual(
    const PreFOSPresolver *presolver, const double *reduced_x, const double *reduced_y,
    const double *reduced_z, const double *reduced_affine_z, double tolerance,
    double *original_x, double *original_y, double *original_z,
    double *original_affine_z)
{
    if (presolver && presolver->has_run &&
        (presolver->n_facial_reductions > 0 ||
         presolver->n_affine_face_substitutions > 0))
        return PREFOS_STATUS_DUAL_RECOVERY_UNAVAILABLE;
    return postsolve_primal_dual_internal(
        presolver, reduced_x, reduced_y, reduced_z, reduced_affine_z, tolerance,
        original_x, original_y, original_z, original_affine_z, 1);
}

PreFOSStatus prefos_postsolve_full_extended_dual(
    const PreFOSPresolver *presolver, const double *reduced_x, const double *reduced_y,
    const double *reduced_z, const double *reduced_affine_z, double tolerance,
    double *original_x, double *original_y, double *original_z,
    double *original_affine_z)
{
    return postsolve_primal_dual_internal(
        presolver, reduced_x, reduced_y, reduced_z, reduced_affine_z, tolerance,
        original_x, original_y, original_z, original_affine_z, 1);
}

static void evaluate_interval_normal(double value, double lower, double upper,
                                     double dual, double tolerance,
                                     long double *dual_violation,
                                     long double *complementarity_violation)
{
    int lower_active =
        isfinite(lower) && prefos_internal_values_close(value, lower, tolerance);
    int upper_active =
        isfinite(upper) && prefos_internal_values_close(value, upper, tolerance);
    long double complementarity = 0.0L;

    if (lower_active && !upper_active)
        update_max_violation((long double) dual, dual_violation);
    else if (upper_active && !lower_active)
        update_max_violation(-(long double) dual, dual_violation);
    else if (!lower_active && !upper_active)
        update_max_violation(fabsl((long double) dual), dual_violation);

    if (dual < 0.0 && isfinite(lower))
        complementarity = fabsl((long double) dual * ((long double) value - lower));
    else if (dual > 0.0 && isfinite(upper))
        complementarity = fabsl((long double) dual * ((long double) upper - value));
    update_max_violation(complementarity, complementarity_violation);
}

static int cone_has_face_boxes(const PreFOSPresolver *presolver,
                               const PreFOSConeBlock *cone)
{
    size_t i;
    for (i = 0; i < cone->dimension; ++i)
        if (presolver->cone_face_box[cone->indices[i]]) return 1;
    return 0;
}

static PreFOSStatus evaluate_affine_dual_residual(
    const PreFOSModelView *model, const double *x, const double *affine_z,
    double tolerance, const PreFOSPresolver *facial_context,
    long double *dual_violation,
    long double *complementarity_violation)
{
    size_t total_rows = model->affine_cone_matrix->rows;
    double *point = NULL, *negative_dual = NULL;
    int *indices = NULL;
    size_t row, k, block_start = 0;
    PreFOSStatus status = PREFOS_STATUS_OK;
    if (total_rows == 0) return PREFOS_STATUS_OK;
    point = (double *) prefos_internal_alloc_array(total_rows, sizeof(double));
    negative_dual =
        (double *) prefos_internal_alloc_array(total_rows, sizeof(double));
    indices = (int *) prefos_internal_alloc_array(total_rows, sizeof(int));
    if (!point || !negative_dual || !indices)
    {
        status = PREFOS_STATUS_OUT_OF_MEMORY;
        goto cleanup;
    }
    for (row = 0; row < total_rows; ++row)
    {
        long double value = (long double) model->affine_cone_offset[row];
        int p;
        for (p = model->affine_cone_matrix->row_pointers[row];
             p < model->affine_cone_matrix->row_pointers[row + 1]; ++p)
            value += (long double) model->affine_cone_matrix->values[p] *
                     (long double)
                         x[model->affine_cone_matrix->column_indices[p]];
        if (!isfinite(value) || fabsl(value) > (long double) DBL_MAX)
        {
            status = PREFOS_STATUS_NUMERICAL_ERROR;
            goto cleanup;
        }
        point[row] = (double) value;
        negative_dual[row] = -affine_z[row];
        indices[row] = (int) row;
    }
    for (k = 0; k < model->n_affine_cones; ++k)
    {
        const PreFOSAffineConeBlock *affine = &model->affine_cones[k];
        PreFOSConeBlock cone;
        PreFOSModelView cone_model;
        double block_dual_violation = 0.0;
        long double inner_product = 0.0L;
        size_t coordinate;
        if (block_start > total_rows ||
            affine->dimension > total_rows - block_start)
        {
            status = PREFOS_STATUS_NUMERICAL_ERROR;
            goto cleanup;
        }
        if (facial_context && k < facial_context->original.n_affine_cones &&
            facial_context->input_affine_rsoc_zero_axis[k])
        {
            unsigned char zero_axis =
                facial_context->input_affine_rsoc_zero_axis[k] - 1;
            size_t survivor = zero_axis == 0 ? 1 : 0;
            if (affine->type != PREFOS_CONE_ROTATED_SECOND_ORDER ||
                affine->dimension < 3 || zero_axis > 1)
            {
                status = PREFOS_STATUS_NUMERICAL_ERROR;
                goto cleanup;
            }
            evaluate_interval_normal(
                point[block_start + survivor], 0.0, INFINITY,
                affine_z[block_start + survivor], tolerance, dual_violation,
                complementarity_violation);
            block_start += affine->dimension;
            continue;
        }
        cone = (PreFOSConeBlock){affine->type, affine->dimension,
                              affine->matrix_order, indices + block_start,
                              affine->power_alpha};
        memset(&cone_model, 0, sizeof(cone_model));
        cone_model.n = total_rows;
        cone_model.n_cones = 1;
        cone_model.cones = &cone;
        status = evaluate_cone_violation(&cone_model, negative_dual, tolerance, 1,
                                         &block_dual_violation);
        if (status != PREFOS_STATUS_OK) goto cleanup;
        update_max_violation((long double) block_dual_violation, dual_violation);
        for (coordinate = 0; coordinate < affine->dimension; ++coordinate)
        {
            size_t current = block_start + coordinate;
            inner_product +=
                (long double) point[current] * (long double) affine_z[current];
        }
        update_max_violation(fabsl(inner_product), complementarity_violation);
        block_start += affine->dimension;
    }
    if (block_start != total_rows) status = PREFOS_STATUS_NUMERICAL_ERROR;

cleanup:
    free(point);
    free(negative_dual);
    free(indices);
    return status;
}

static PreFOSStatus evaluate_kkt(const PreFOSModelView *model, const double *x,
                              const double *y, const double *z,
                              const double *affine_z, double tolerance,
                              const PreFOSPresolver *facial_context,
                              PreFOSKKTResiduals *residuals)
{
    double objective;
    double *gradient, *negative_z;
    long double stationarity = 0.0L, row_dual = 0.0L, domain_dual = 0.0L,
                complementarity = 0.0L;
    size_t row, i, k;
    PreFOSStatus status;

    memset(residuals, 0, sizeof(*residuals));
    if (!vector_is_finite(x, model->n) || !vector_is_finite(y, model->A->rows) ||
        !vector_is_finite(z, model->n) ||
        (model->affine_cone_matrix->rows > 0 &&
         (!affine_z ||
          !vector_is_finite(affine_z, model->affine_cone_matrix->rows))))
        return PREFOS_STATUS_INVALID_ARGUMENT;
    status = evaluate_model(
        model, x, tolerance, &objective, &residuals->row_primal_violation,
        &residuals->box_primal_violation, &residuals->cone_primal_violation);
    if (status != PREFOS_STATUS_OK) return status;
    gradient = (double *) prefos_internal_alloc_array(model->n, sizeof(double));
    negative_z = (double *) prefos_internal_alloc_array(model->n, sizeof(double));
    if (model->n > 0 && (!gradient || !negative_z))
    {
        free(gradient);
        free(negative_z);
        return PREFOS_STATUS_OUT_OF_MEMORY;
    }
    status = evaluate_gradient(model, x, gradient);
    if (status != PREFOS_STATUS_OK)
    {
        free(gradient);
        free(negative_z);
        return status;
    }
    for (row = 0; row < model->A->rows; ++row)
    {
        int p;
        long double activity = 0.0L;
        for (p = model->A->row_pointers[row]; p < model->A->row_pointers[row + 1];
             ++p)
        {
            int column = model->A->column_indices[p];
            activity += (long double) model->A->values[p] * (long double) x[column];
            if (!prefos_internal_safe_add_product(&gradient[column],
                                               model->A->values[p], y[row]))
            {
                free(gradient);
                free(negative_z);
                return PREFOS_STATUS_NUMERICAL_ERROR;
            }
        }
        evaluate_interval_normal((double) activity, model->constraint_lower[row],
                                 model->constraint_upper[row], y[row], tolerance,
                                 &row_dual, &complementarity);
    }
    status = add_affine_stationarity(model, affine_z, gradient);
    if (status != PREFOS_STATUS_OK)
    {
        free(gradient);
        free(negative_z);
        return status;
    }
    status = evaluate_affine_dual_residual(model, x, affine_z, tolerance,
                                           facial_context,
                                           &domain_dual, &complementarity);
    if (status != PREFOS_STATUS_OK)
    {
        free(gradient);
        free(negative_z);
        return status;
    }
    for (i = 0; i < model->n; ++i)
    {
        update_max_violation(fabsl((long double) gradient[i] + z[i]), &stationarity);
        negative_z[i] = -z[i];
    }
    for (i = 0; i < model->n_box; ++i)
    {
        int column = model->box_indices[i];
        evaluate_interval_normal(x[column], model->box_lower[i], model->box_upper[i],
                                 z[column], tolerance, &domain_dual,
                                 &complementarity);
    }
    for (k = 0; k < model->n_cones; ++k)
    {
        const PreFOSConeBlock *cone = &model->cones[k];
        long double inner_product = 0.0L;
        if (facial_context && cone_has_face_boxes(facial_context, cone))
        {
            for (i = 0; i < cone->dimension; ++i)
            {
                int column = cone->indices[i];
                if (!facial_context->cone_face_box[column]) continue;
                evaluate_interval_normal(
                    x[column], facial_context->cone_face_box_lower[column],
                    facial_context->cone_face_box_upper[column], z[column],
                    tolerance, &domain_dual, &complementarity);
            }
        }
        else if (facial_context &&
                 facial_context->psd_face_reductions[k].n_removed > 0)
        {
            status = evaluate_psd_face_dual_violation(
                cone, &facial_context->psd_face_reductions[k], negative_z, tolerance,
                &domain_dual);
            if (status != PREFOS_STATUS_OK)
            {
                free(gradient);
                free(negative_z);
                return status;
            }
        }
        else
        {
            PreFOSModelView single_cone_model = *model;
            double cone_dual_violation;
            single_cone_model.n_cones = 1;
            single_cone_model.cones = cone;
            single_cone_model.n_affine_cones = 0;
            single_cone_model.affine_cones = NULL;
            status = evaluate_cone_violation(&single_cone_model, negative_z,
                                             tolerance, 1, &cone_dual_violation);
            if (status != PREFOS_STATUS_OK)
            {
                free(gradient);
                free(negative_z);
                return status;
            }
            update_max_violation((long double) cone_dual_violation, &domain_dual);
        }
        for (i = 0; i < cone->dimension; ++i)
        {
            int column = cone->indices[i];
            inner_product += (long double) x[column] * (long double) z[column];
        }
        update_max_violation(fabsl(inner_product), &complementarity);
    }
    free(gradient);
    free(negative_z);

    residuals->stationarity_violation = violation_as_double(stationarity);
    residuals->row_dual_violation = violation_as_double(row_dual);
    residuals->domain_dual_violation = violation_as_double(domain_dual);
    residuals->complementarity_violation = violation_as_double(complementarity);
    residuals->passed = residuals->row_primal_violation <= tolerance &&
                        residuals->box_primal_violation <= tolerance &&
                        residuals->cone_primal_violation <= tolerance &&
                        residuals->stationarity_violation <= tolerance &&
                        residuals->row_dual_violation <= tolerance &&
                        residuals->domain_dual_violation <= tolerance &&
                        residuals->complementarity_violation <= tolerance;
    return PREFOS_STATUS_OK;
}

PreFOSStatus prefos_verify_postsolve_primal(const PreFOSPresolver *presolver,
                                      const double *reduced_x, double tolerance,
                                      PreFOSPrimalVerification *verification)
{
    PreFOSModelView original, reduced;
    double *original_x;
    double objective_scale, maximum_original_violation, maximum_reduced_violation;
    PreFOSStatus status;

    if (!presolver || !presolver->has_run || !verification || !isfinite(tolerance) ||
        tolerance < 0.0 || (presolver->reduced.n > 0 && !reduced_x))
        return PREFOS_STATUS_INVALID_ARGUMENT;
    memset(verification, 0, sizeof(*verification));

    original_x =
        (double *) prefos_internal_alloc_array(presolver->original.n, sizeof(double));
    if (presolver->original.n > 0 && !original_x) return PREFOS_STATUS_OUT_OF_MEMORY;
    status = prefos_postsolve_primal(presolver, reduced_x, original_x);
    if (status != PREFOS_STATUS_OK)
    {
        free(original_x);
        return status;
    }

    original = (PreFOSModelView){presolver->original.n,
                              &presolver->original.A,
                              presolver->original.constraint_lower,
                              presolver->original.constraint_upper,
                              &presolver->original.Q,
                              presolver->original.q_storage,
                              &presolver->original.R,
                              presolver->original.D,
                              presolver->original.c,
                              presolver->original.objective_offset,
                              presolver->original.n_box,
                              presolver->original.box_indices,
                              presolver->original.box_lower,
                              presolver->original.box_upper,
                              presolver->original.n_cones,
                              presolver->original.cones,
                              &presolver->original.affine_cone_matrix,
                              presolver->original.affine_cone_offset,
                              presolver->original.n_affine_cones,
                              presolver->original.affine_cones};
    reduced = (PreFOSModelView){presolver->reduced.n,
                             &presolver->reduced.A,
                             presolver->reduced.constraint_lower,
                             presolver->reduced.constraint_upper,
                             &presolver->reduced.Q,
                             presolver->reduced.q_storage,
                             &presolver->reduced.R,
                             presolver->reduced.D,
                             presolver->reduced.c,
                             presolver->reduced.objective_offset,
                             presolver->reduced.n_box,
                             presolver->reduced.box_indices,
                             presolver->reduced.box_lower,
                             presolver->reduced.box_upper,
                             presolver->reduced.n_cones,
                             presolver->reduced.cones,
                             &presolver->reduced.affine_cone_matrix,
                             presolver->reduced.affine_cone_offset,
                             presolver->reduced.n_affine_cones,
                             presolver->reduced.affine_cones};

    status = evaluate_model(
        &original, original_x, tolerance, &verification->original_objective,
        &verification->original_row_violation, &verification->original_box_violation,
        &verification->original_cone_violation);
    free(original_x);
    if (status != PREFOS_STATUS_OK) return status;
    status = evaluate_model(
        &reduced, reduced_x, tolerance, &verification->reduced_objective,
        &verification->reduced_row_violation, &verification->reduced_box_violation,
        &verification->reduced_cone_violation);
    if (status != PREFOS_STATUS_OK) return status;

    verification->objective_absolute_error =
        fabs(verification->original_objective - verification->reduced_objective);
    objective_scale = fmax(1.0, fmax(fabs(verification->original_objective),
                                     fabs(verification->reduced_objective)));
    maximum_original_violation = fmax(verification->original_row_violation,
                                      fmax(verification->original_box_violation,
                                           verification->original_cone_violation));
    maximum_reduced_violation = fmax(verification->reduced_row_violation,
                                     fmax(verification->reduced_box_violation,
                                          verification->reduced_cone_violation));
    verification->passed =
        verification->objective_absolute_error <= tolerance * objective_scale &&
        maximum_original_violation <= tolerance &&
        maximum_reduced_violation <= tolerance;
    return PREFOS_STATUS_OK;
}

static PreFOSStatus
verify_postsolve_kkt_internal(const PreFOSPresolver *presolver, const double *reduced_x,
                              const double *reduced_y, const double *reduced_z,
                              const double *reduced_affine_z, double tolerance,
                              int extended, int full,
                              PreFOSPostsolveKKTVerification *verification)
{
    PreFOSModelView original, reduced;
    double *original_x, *original_y, *original_z, *original_affine_z;
    double original_objective = 0.0;
    double reduced_objective = 0.0;
    double objective_scale;
    size_t n, m;
    PreFOSStatus status;

    if (!presolver || !presolver->has_run || !verification || !isfinite(tolerance) ||
        tolerance < 0.0 ||
        (presolver->reduced.n > 0 && (!reduced_x || !reduced_z)) ||
        (presolver->reduced.A.rows > 0 && !reduced_y) ||
        (full && presolver->reduced.affine_cone_matrix.rows > 0 &&
         !reduced_affine_z))
        return PREFOS_STATUS_INVALID_ARGUMENT;
    memset(verification, 0, sizeof(*verification));
    n = presolver->original.n;
    m = presolver->original.A.rows;
    original_x = (double *) prefos_internal_alloc_array(n, sizeof(double));
    original_y = (double *) prefos_internal_alloc_array(m, sizeof(double));
    original_z = (double *) prefos_internal_alloc_array(n, sizeof(double));
    original_affine_z = (double *) prefos_internal_alloc_array(
        presolver->original.affine_cone_matrix.rows, sizeof(double));
    if ((n > 0 && (!original_x || !original_z)) || (m > 0 && !original_y))
    {
        free(original_x);
        free(original_y);
        free(original_z);
        free(original_affine_z);
        return PREFOS_STATUS_OUT_OF_MEMORY;
    }
    if (presolver->original.affine_cone_matrix.rows > 0 && !original_affine_z)
    {
        free(original_x);
        free(original_y);
        free(original_z);
        return PREFOS_STATUS_OUT_OF_MEMORY;
    }
    if (full)
        status = extended ? prefos_postsolve_full_extended_dual(
                                presolver, reduced_x, reduced_y, reduced_z,
                                reduced_affine_z, tolerance, original_x, original_y,
                                original_z, original_affine_z)
                          : prefos_postsolve_full_primal_dual(
                                presolver, reduced_x, reduced_y, reduced_z,
                                reduced_affine_z, tolerance, original_x, original_y,
                                original_z, original_affine_z);
    else
        status = extended ? prefos_postsolve_extended_dual(
                                presolver, reduced_x, reduced_y, reduced_z, tolerance,
                                original_x, original_y, original_z)
                          : prefos_postsolve_primal_dual(
                                presolver, reduced_x, reduced_y, reduced_z, tolerance,
                                original_x, original_y, original_z);
    if (status != PREFOS_STATUS_OK)
    {
        free(original_x);
        free(original_y);
        free(original_z);
        free(original_affine_z);
        return status;
    }

    original = (PreFOSModelView){presolver->original.n,
                              &presolver->original.A,
                              presolver->original.constraint_lower,
                              presolver->original.constraint_upper,
                              &presolver->original.Q,
                              presolver->original.q_storage,
                              &presolver->original.R,
                              presolver->original.D,
                              presolver->original.c,
                              presolver->original.objective_offset,
                              presolver->original.n_box,
                              presolver->original.box_indices,
                              presolver->original.box_lower,
                              presolver->original.box_upper,
                              presolver->original.n_cones,
                              presolver->original.cones,
                              &presolver->original.affine_cone_matrix,
                              presolver->original.affine_cone_offset,
                              presolver->original.n_affine_cones,
                              presolver->original.affine_cones};
    reduced = (PreFOSModelView){presolver->reduced.n,
                             &presolver->reduced.A,
                             presolver->reduced.constraint_lower,
                             presolver->reduced.constraint_upper,
                             &presolver->reduced.Q,
                             presolver->reduced.q_storage,
                             &presolver->reduced.R,
                             presolver->reduced.D,
                             presolver->reduced.c,
                             presolver->reduced.objective_offset,
                             presolver->reduced.n_box,
                             presolver->reduced.box_indices,
                             presolver->reduced.box_lower,
                             presolver->reduced.box_upper,
                             presolver->reduced.n_cones,
                             presolver->reduced.cones,
                             &presolver->reduced.affine_cone_matrix,
                             presolver->reduced.affine_cone_offset,
                             presolver->reduced.n_affine_cones,
                             presolver->reduced.affine_cones};

    status = evaluate_kkt(&reduced, reduced_x, reduced_y, reduced_z,
                          reduced_affine_z, tolerance, NULL,
                          &verification->reduced);
    if (status == PREFOS_STATUS_OK)
        status = evaluate_kkt(&original, original_x, original_y, original_z,
                              original_affine_z, tolerance,
                              extended ? presolver : NULL, &verification->original);
    if (status == PREFOS_STATUS_OK)
        status = evaluate_objective(&original, original_x, &original_objective);
    if (status == PREFOS_STATUS_OK)
        status = evaluate_objective(&reduced, reduced_x, &reduced_objective);
    free(original_x);
    free(original_y);
    free(original_z);
    free(original_affine_z);
    if (status != PREFOS_STATUS_OK) return status;

    verification->objective_absolute_error =
        fabs(original_objective - reduced_objective);
    objective_scale =
        fmax(1.0, fmax(fabs(original_objective), fabs(reduced_objective)));
    verification->passed =
        verification->reduced.passed && verification->original.passed &&
        verification->objective_absolute_error <= tolerance * objective_scale;
    return PREFOS_STATUS_OK;
}

PreFOSStatus prefos_verify_postsolve_kkt(const PreFOSPresolver *presolver,
                                   const double *reduced_x, const double *reduced_y,
                                   const double *reduced_z, double tolerance,
                                   PreFOSPostsolveKKTVerification *verification)
{
    return verify_postsolve_kkt_internal(presolver, reduced_x, reduced_y, reduced_z,
                                         NULL, tolerance, 0, 0, verification);
}

PreFOSStatus
prefos_verify_postsolve_extended_kkt(const PreFOSPresolver *presolver,
                                  const double *reduced_x, const double *reduced_y,
                                  const double *reduced_z, double tolerance,
                                  PreFOSPostsolveKKTVerification *verification)
{
    return verify_postsolve_kkt_internal(presolver, reduced_x, reduced_y, reduced_z,
                                         NULL, tolerance, 1, 0, verification);
}

PreFOSStatus prefos_verify_postsolve_full_kkt(
    const PreFOSPresolver *presolver, const double *reduced_x, const double *reduced_y,
    const double *reduced_z, const double *reduced_affine_z, double tolerance,
    PreFOSPostsolveKKTVerification *verification)
{
    return verify_postsolve_kkt_internal(
        presolver, reduced_x, reduced_y, reduced_z, reduced_affine_z, tolerance, 0,
        1, verification);
}

PreFOSStatus prefos_verify_postsolve_full_extended_kkt(
    const PreFOSPresolver *presolver, const double *reduced_x, const double *reduced_y,
    const double *reduced_z, const double *reduced_affine_z, double tolerance,
    PreFOSPostsolveKKTVerification *verification)
{
    return verify_postsolve_kkt_internal(
        presolver, reduced_x, reduced_y, reduced_z, reduced_affine_z, tolerance, 1,
        1, verification);
}

const char *prefos_status_string(PreFOSStatus status)
{
    switch (status)
    {
        case PREFOS_STATUS_OK:
            return "ok";
        case PREFOS_STATUS_REDUCED:
            return "reduced";
        case PREFOS_STATUS_PRIMAL_INFEASIBLE:
            return "primal infeasible";
        case PREFOS_STATUS_INVALID_ARGUMENT:
            return "invalid argument";
        case PREFOS_STATUS_NUMERICAL_ERROR:
            return "numerical error";
        case PREFOS_STATUS_OUT_OF_MEMORY:
            return "out of memory";
        case PREFOS_STATUS_DUAL_RECOVERY_UNAVAILABLE:
            return "standard dual recovery unavailable after facial reduction";
        default:
            return "unknown status";
    }
}
