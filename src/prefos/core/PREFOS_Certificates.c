/*
 * Copyright 2026 Hongpei Li
 * SPDX-License-Identifier: Apache-2.0
 */

#include "PREFOS_CertificateInternal.h"
#include "cones/PREFOS_ExponentialCone.h"
#include "cones/PREFOS_PositiveSemidefiniteCone.h"
#include "cones/PREFOS_PowerCone.h"
#include "explorers/PREFOS_ConePropagation.h"

PreFOSCertificateModel prefos_internal_original_certificate_model(
    const PreFOSPresolver *presolver)
{
    PreFOSCertificateModel model = {
        presolver->original.n,
        &presolver->original.A,
        presolver->original.constraint_lower,
        presolver->original.constraint_upper,
        &presolver->original.Q,
        presolver->original.q_storage,
        &presolver->original.R,
        presolver->original.D,
        presolver->original.c,
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
    return model;
}

PreFOSCertificateModel prefos_internal_reduced_certificate_model(
    const PreFOSPresolver *presolver)
{
    PreFOSCertificateModel model = {
        presolver->reduced.n,
        &presolver->reduced.A,
        presolver->reduced.constraint_lower,
        presolver->reduced.constraint_upper,
        &presolver->reduced.Q,
        presolver->reduced.q_storage,
        &presolver->reduced.R,
        presolver->reduced.D,
        presolver->reduced.c,
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
    return model;
}

int prefos_internal_certificate_vector_is_finite(
    const double *values, size_t count)
{
    size_t i;
    for (i = 0; i < count; ++i)
        if (!isfinite(values[i])) return 0;
    return 1;
}

static void update_maximum(long double candidate, long double *maximum)
{
    if (isnan(candidate))
        *maximum = INFINITY;
    else if (candidate > *maximum)
        *maximum = candidate;
}

static double finite_residual(long double value)
{
    if (value <= 0.0L) return 0.0;
    if (!isfinite(value) || value > (long double) DBL_MAX) return INFINITY;
    return (double) value;
}

static double finite_value(long double value)
{
    if (!isfinite(value) || fabsl(value) > (long double) DBL_MAX)
        return value < 0.0L ? -INFINITY : INFINITY;
    return (double) value;
}

static void interval_support(double normal, double lower, double upper,
                             long double *support, long double *violation,
                             long double *data_scale)
{
    if (isfinite(lower))
        update_maximum(fabsl((long double) lower), data_scale);
    if (isfinite(upper))
        update_maximum(fabsl((long double) upper), data_scale);
    if (normal < 0.0)
    {
        if (isfinite(lower))
            *support += (long double) normal * (long double) lower;
        else
            update_maximum(-(long double) normal, violation);
    }
    else if (normal > 0.0)
    {
        if (isfinite(upper))
            *support += (long double) normal * (long double) upper;
        else
            update_maximum((long double) normal, violation);
    }
}

static void interval_recession_violation(double direction, double lower,
                                          double upper,
                                          long double *violation)
{
    if (isfinite(lower) && isfinite(upper))
        update_maximum(fabsl((long double) direction), violation);
    else if (isfinite(lower))
        update_maximum(-(long double) direction, violation);
    else if (isfinite(upper))
        update_maximum((long double) direction, violation);
}

static void interval_polar_violation(double normal, double lower,
                                     double upper,
                                     long double *violation)
{
    if (normal < 0.0 && !isfinite(lower))
        update_maximum(-(long double) normal, violation);
    else if (normal > 0.0 && !isfinite(upper))
        update_maximum((long double) normal, violation);
}

static PreFOSStatus evaluate_one_cone(const PreFOSConeBlock *cone,
                                     const double *point, double tolerance,
                                     int dual, long double *violation)
{
    size_t coordinate;
    if (cone->type == PREFOS_CONE_NONNEGATIVE)
    {
        for (coordinate = 0; coordinate < cone->dimension; ++coordinate)
            update_maximum(-(long double) point[cone->indices[coordinate]],
                           violation);
        return PREFOS_STATUS_OK;
    }
    if (cone->type == PREFOS_CONE_SECOND_ORDER)
    {
        long double norm_squared = 0.0L;
        for (coordinate = 1; coordinate < cone->dimension; ++coordinate)
        {
            long double value =
                (long double) point[cone->indices[coordinate]];
            norm_squared += value * value;
        }
        update_maximum(
            sqrtl(norm_squared) -
                (long double) point[cone->indices[0]],
            violation);
        return PREFOS_STATUS_OK;
    }
    if (cone->type == PREFOS_CONE_ROTATED_SECOND_ORDER)
    {
        long double u = (long double) point[cone->indices[0]];
        long double v = (long double) point[cone->indices[1]];
        long double norm_squared = 0.0L;
        long double radial_limit = 0.0L;
        for (coordinate = 2; coordinate < cone->dimension; ++coordinate)
        {
            long double value =
                (long double) point[cone->indices[coordinate]];
            norm_squared += value * value;
        }
        update_maximum(-u, violation);
        update_maximum(-v, violation);
        if (u >= 0.0L && v >= 0.0L)
            radial_limit = sqrtl(2.0L * u * v);
        update_maximum(sqrtl(norm_squared) - radial_limit, violation);
        return PREFOS_STATUS_OK;
    }
    if (cone->type == PREFOS_CONE_POSITIVE_SEMIDEFINITE)
        return prefos_internal_evaluate_psd_violation(
            cone, point, tolerance, violation);
    if (cone->type == PREFOS_CONE_EXPONENTIAL)
        return prefos_internal_exponential_cone_violation(
            cone, point, dual, violation);
    if (cone->type == PREFOS_CONE_POWER)
        return prefos_internal_power_cone_violation(
            cone, point, dual, violation);
    return PREFOS_STATUS_NUMERICAL_ERROR;
}

static PreFOSStatus evaluate_psd_face_polar(
    const PreFOSConeBlock *cone, const PreFOSPSDFaceReduction *face,
    const double *negative_normal, double tolerance, long double *violation)
{
    PreFOSConeBlock reduced_face;
    size_t reduced_order = cone->matrix_order - face->n_removed;
    size_t reduced_dimension =
        reduced_order * (reduced_order + 1U) / 2U;
    size_t row, column, write = 0;
    PreFOSStatus status;

    if (reduced_order == 0) return PREFOS_STATUS_OK;
    memset(&reduced_face, 0, sizeof(reduced_face));
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
            if (prefos_internal_psd_matrix_index_is_removed(face, column))
                continue;
            packed = row * (row + 1U) / 2U + column;
            reduced_face.indices[write++] = cone->indices[packed];
        }
    }
    if (write != reduced_dimension)
    {
        free(reduced_face.indices);
        return PREFOS_STATUS_NUMERICAL_ERROR;
    }
    status = prefos_internal_evaluate_psd_violation(
        &reduced_face, negative_normal, tolerance, violation);
    free(reduced_face.indices);
    return status;
}

static int cone_uses_face_boxes(const PreFOSPresolver *presolver,
                                const PreFOSConeBlock *cone)
{
    size_t coordinate;
    for (coordinate = 0; coordinate < cone->dimension; ++coordinate)
        if (presolver->cone_face_box[cone->indices[coordinate]]) return 1;
    return 0;
}

static PreFOSStatus evaluate_direct_domain_polar(
    const PreFOSCertificateModel *model, const double *normal,
    double tolerance, const PreFOSPresolver *facial_context,
    long double *violation)
{
    double *negative_normal;
    size_t cone_index, coordinate;
    PreFOSStatus status = PREFOS_STATUS_OK;

    negative_normal =
        (double *) prefos_internal_alloc_array(model->n, sizeof(double));
    if (model->n > 0 && !negative_normal)
        return PREFOS_STATUS_OUT_OF_MEMORY;
    for (coordinate = 0; coordinate < model->n; ++coordinate)
        negative_normal[coordinate] = -normal[coordinate];

    for (cone_index = 0; cone_index < model->n_cones; ++cone_index)
    {
        const PreFOSConeBlock *cone = &model->cones[cone_index];
        if (facial_context && cone_uses_face_boxes(facial_context, cone))
        {
            for (coordinate = 0; coordinate < cone->dimension; ++coordinate)
            {
                int column = cone->indices[coordinate];
                if (!facial_context->cone_face_box[column]) continue;
                interval_polar_violation(
                    normal[column],
                    facial_context->cone_face_box_lower[column],
                    facial_context->cone_face_box_upper[column],
                    violation);
            }
        }
        else if (facial_context &&
                 facial_context->psd_face_reductions[cone_index].n_removed > 0)
            status = evaluate_psd_face_polar(
                cone, &facial_context->psd_face_reductions[cone_index],
                negative_normal, tolerance, violation);
        else
            status = evaluate_one_cone(
                cone, negative_normal, tolerance, 1, violation);
        if (status != PREFOS_STATUS_OK) break;
    }
    free(negative_normal);
    return status;
}

static PreFOSStatus evaluate_affine_domain_polar(
    const PreFOSCertificateModel *model, const double *affine_normal,
    double tolerance, const PreFOSPresolver *facial_context,
    long double *violation)
{
    double *negative_normal = NULL;
    int *indices = NULL;
    size_t row, block, block_start = 0;
    PreFOSStatus status = PREFOS_STATUS_OK;

    if (model->affine_cone_matrix->rows == 0) return PREFOS_STATUS_OK;
    negative_normal = (double *) prefos_internal_alloc_array(
        model->affine_cone_matrix->rows, sizeof(double));
    indices = (int *) prefos_internal_alloc_array(
        model->affine_cone_matrix->rows, sizeof(int));
    if (!negative_normal || !indices)
    {
        status = PREFOS_STATUS_OUT_OF_MEMORY;
        goto cleanup;
    }
    for (row = 0; row < model->affine_cone_matrix->rows; ++row)
    {
        negative_normal[row] = -affine_normal[row];
        indices[row] = (int) row;
    }
    for (block = 0; block < model->n_affine_cones; ++block)
    {
        const PreFOSAffineConeBlock *affine = &model->affine_cones[block];
        PreFOSConeBlock cone;
        if (block_start > model->affine_cone_matrix->rows ||
            affine->dimension >
                model->affine_cone_matrix->rows - block_start)
        {
            status = PREFOS_STATUS_NUMERICAL_ERROR;
            goto cleanup;
        }
        if (facial_context &&
            block < facial_context->original.n_affine_cones &&
            facial_context->input_affine_rsoc_zero_axis[block])
        {
            unsigned char zero_axis = (unsigned char)
                (facial_context->input_affine_rsoc_zero_axis[block] - 1U);
            size_t survivor = zero_axis == 0U ? 1U : 0U;
            if (affine->type != PREFOS_CONE_ROTATED_SECOND_ORDER ||
                affine->dimension < 3U || zero_axis > 1U)
            {
                status = PREFOS_STATUS_NUMERICAL_ERROR;
                goto cleanup;
            }
            update_maximum(
                (long double) affine_normal[block_start + survivor],
                violation);
        }
        else
        {
            cone = (PreFOSConeBlock){
                affine->type, affine->dimension, affine->matrix_order,
                indices + block_start, affine->power_alpha};
            status = evaluate_one_cone(
                &cone, negative_normal, tolerance, 1, violation);
            if (status != PREFOS_STATUS_OK) goto cleanup;
        }
        block_start += affine->dimension;
    }
    if (block_start != model->affine_cone_matrix->rows)
        status = PREFOS_STATUS_NUMERICAL_ERROR;

cleanup:
    free(negative_normal);
    free(indices);
    return status;
}

PreFOSStatus prefos_internal_evaluate_infeasibility_certificate(
    const PreFOSCertificateModel *model, const double *y, const double *z,
    const double *affine_z, double tolerance,
    const PreFOSPresolver *facial_context,
    PreFOSInfeasibilityCertificateResiduals *residuals)
{
    long double *stationarity = NULL;
    long double *column_scale = NULL;
    long double support = 0.0L;
    long double row_dual = 0.0L, domain_dual = 0.0L;
    long double stationarity_violation = 0.0L;
    long double normal_scale = 0.0L, data_scale = 1.0L;
    long double coefficient_scale = 1.0L;
    size_t row, column, i;
    PreFOSStatus status;

    memset(residuals, 0, sizeof(*residuals));
    if (!prefos_internal_certificate_vector_is_finite(
            y, model->A->rows) ||
        !prefos_internal_certificate_vector_is_finite(z, model->n) ||
        (model->affine_cone_matrix->rows > 0 &&
         (!affine_z ||
          !prefos_internal_certificate_vector_is_finite(
              affine_z, model->affine_cone_matrix->rows))))
        return PREFOS_STATUS_INVALID_ARGUMENT;
    stationarity =
        (long double *) calloc(model->n, sizeof(long double));
    column_scale =
        (long double *) calloc(model->n, sizeof(long double));
    if (model->n > 0 && (!stationarity || !column_scale))
    {
        free(stationarity);
        free(column_scale);
        return PREFOS_STATUS_OUT_OF_MEMORY;
    }

    for (row = 0; row < model->A->rows; ++row)
    {
        int p;
        update_maximum(fabsl((long double) y[row]), &normal_scale);
        interval_support(y[row], model->constraint_lower[row],
                         model->constraint_upper[row], &support, &row_dual,
                         &data_scale);
        for (p = model->A->row_pointers[row];
             p < model->A->row_pointers[row + 1]; ++p)
        {
            column = (size_t) model->A->column_indices[p];
            stationarity[column] +=
                (long double) model->A->values[p] *
                (long double) y[row];
            column_scale[column] +=
                fabsl((long double) model->A->values[p]);
        }
    }
    for (column = 0; column < model->n; ++column)
    {
        stationarity[column] += (long double) z[column];
        column_scale[column] += 1.0L;
        update_maximum(fabsl((long double) z[column]), &normal_scale);
    }
    for (row = 0; row < model->affine_cone_matrix->rows; ++row)
    {
        int p;
        update_maximum(fabsl((long double) affine_z[row]), &normal_scale);
        update_maximum(
            fabsl((long double) model->affine_cone_offset[row]),
            &data_scale);
        support -= (long double) affine_z[row] *
                   (long double) model->affine_cone_offset[row];
        for (p = model->affine_cone_matrix->row_pointers[row];
             p < model->affine_cone_matrix->row_pointers[row + 1]; ++p)
        {
            column =
                (size_t) model->affine_cone_matrix->column_indices[p];
            stationarity[column] +=
                (long double) model->affine_cone_matrix->values[p] *
                (long double) affine_z[row];
            column_scale[column] +=
                fabsl((long double)
                          model->affine_cone_matrix->values[p]);
        }
    }
    for (i = 0; i < model->n_box; ++i)
    {
        int box_column = model->box_indices[i];
        interval_support(z[box_column], model->box_lower[i],
                         model->box_upper[i], &support, &domain_dual,
                         &data_scale);
    }
    status = evaluate_direct_domain_polar(
        model, z, tolerance, facial_context, &domain_dual);
    if (status != PREFOS_STATUS_OK) goto cleanup;
    status = evaluate_affine_domain_polar(
        model, affine_z, tolerance, facial_context, &domain_dual);
    if (status != PREFOS_STATUS_OK) goto cleanup;
    for (column = 0; column < model->n; ++column)
    {
        update_maximum(fabsl(stationarity[column]),
                       &stationarity_violation);
        update_maximum(column_scale[column], &coefficient_scale);
    }

    residuals->stationarity_violation =
        finite_residual(stationarity_violation);
    residuals->row_dual_violation = finite_residual(row_dual);
    residuals->domain_dual_violation = finite_residual(domain_dual);
    residuals->certificate_value = finite_value(support);
    if (normal_scale > 0.0L && isfinite(normal_scale) &&
        isfinite(support))
    {
        long double stationarity_limit =
            (long double) tolerance * normal_scale * coefficient_scale;
        long double dual_limit =
            (long double) tolerance * normal_scale;
        long double value_limit =
            (long double) tolerance * normal_scale * data_scale;
        residuals->passed =
            stationarity_violation <= stationarity_limit &&
            row_dual <= dual_limit && domain_dual <= dual_limit &&
            support < -value_limit;
    }

cleanup:
    free(stationarity);
    free(column_scale);
    return status;
}

static PreFOSStatus evaluate_affine_recession(
    const PreFOSCertificateModel *model, const double *ray,
    double tolerance, long double *violation)
{
    double *point = NULL;
    int *indices = NULL;
    size_t row, block, block_start = 0;
    PreFOSStatus status = PREFOS_STATUS_OK;
    if (model->affine_cone_matrix->rows == 0) return PREFOS_STATUS_OK;
    point = (double *) prefos_internal_alloc_array(
        model->affine_cone_matrix->rows, sizeof(double));
    indices = (int *) prefos_internal_alloc_array(
        model->affine_cone_matrix->rows, sizeof(int));
    if (!point || !indices)
    {
        status = PREFOS_STATUS_OUT_OF_MEMORY;
        goto cleanup;
    }
    for (row = 0; row < model->affine_cone_matrix->rows; ++row)
    {
        long double value = 0.0L;
        int p;
        for (p = model->affine_cone_matrix->row_pointers[row];
             p < model->affine_cone_matrix->row_pointers[row + 1]; ++p)
            value +=
                (long double) model->affine_cone_matrix->values[p] *
                (long double)
                    ray[model->affine_cone_matrix->column_indices[p]];
        if (!isfinite(value) ||
            fabsl(value) > (long double) DBL_MAX)
        {
            status = PREFOS_STATUS_NUMERICAL_ERROR;
            goto cleanup;
        }
        point[row] = (double) value;
        indices[row] = (int) row;
    }
    for (block = 0; block < model->n_affine_cones; ++block)
    {
        const PreFOSAffineConeBlock *affine = &model->affine_cones[block];
        PreFOSConeBlock cone;
        if (block_start > model->affine_cone_matrix->rows ||
            affine->dimension >
                model->affine_cone_matrix->rows - block_start)
        {
            status = PREFOS_STATUS_NUMERICAL_ERROR;
            goto cleanup;
        }
        cone = (PreFOSConeBlock){
            affine->type, affine->dimension, affine->matrix_order,
            indices + block_start, affine->power_alpha};
        status = evaluate_one_cone(
            &cone, point, tolerance, 0, violation);
        if (status != PREFOS_STATUS_OK) goto cleanup;
        block_start += affine->dimension;
    }
    if (block_start != model->affine_cone_matrix->rows)
        status = PREFOS_STATUS_NUMERICAL_ERROR;

cleanup:
    free(point);
    free(indices);
    return status;
}

static PreFOSStatus evaluate_quadratic_ray(
    const PreFOSCertificateModel *model, const double *ray,
    long double *violation, long double *coefficient_scale)
{
    long double *product = NULL;
    long double *factor_product = NULL;
    size_t row;
    if (model->Q->rows != model->n || model->Q->cols != model->n ||
        model->R->cols != model->n)
        return PREFOS_STATUS_NUMERICAL_ERROR;
    if (model->n > 0)
    {
        product =
            (long double *) calloc(model->n, sizeof(long double));
        if (!product) return PREFOS_STATUS_OUT_OF_MEMORY;
    }
    if (model->R->rows > 0)
    {
        factor_product = (long double *) calloc(
            model->R->rows, sizeof(long double));
        if (!factor_product)
        {
            free(product);
            return PREFOS_STATUS_OUT_OF_MEMORY;
        }
    }
    for (row = 0; row < model->Q->rows; ++row)
    {
        int p;
        for (p = model->Q->row_pointers[row];
             p < model->Q->row_pointers[row + 1]; ++p)
        {
            int column = model->Q->column_indices[p];
            long double value =
                (long double) model->Q->values[p];
            product[row] += value * (long double) ray[column];
            update_maximum(fabsl(value), coefficient_scale);
            if (model->q_storage != PREFOS_Q_FULL &&
                column != (int) row)
                product[column] += value * (long double) ray[row];
        }
    }
    for (row = 0; row < model->R->rows; ++row)
    {
        int p;
        for (p = model->R->row_pointers[row];
             p < model->R->row_pointers[row + 1]; ++p)
            factor_product[row] +=
                (long double) model->R->values[p] *
                (long double) ray[model->R->column_indices[p]];
        factor_product[row] *= (long double) model->D[row];
        update_maximum(
            fabsl((long double) model->D[row]), coefficient_scale);
    }
    for (row = 0; row < model->R->rows; ++row)
    {
        int p;
        for (p = model->R->row_pointers[row];
             p < model->R->row_pointers[row + 1]; ++p)
        {
            int column = model->R->column_indices[p];
            long double value =
                (long double) model->R->values[p];
            product[column] += value * factor_product[row];
            update_maximum(fabsl(value), coefficient_scale);
        }
    }
    for (row = 0; row < model->n; ++row)
        update_maximum(fabsl(product[row]), violation);
    free(product);
    free(factor_product);
    return PREFOS_STATUS_OK;
}

PreFOSStatus prefos_internal_evaluate_unbounded_ray(
    const PreFOSCertificateModel *model, const double *ray,
    double tolerance, PreFOSUnboundedRayResiduals *residuals)
{
    long double row_violation = 0.0L, domain_violation = 0.0L;
    long double quadratic_violation = 0.0L;
    long double direction_scale = 0.0L;
    long double linear_scale = 1.0L, quadratic_scale = 1.0L;
    long double objective = 0.0L;
    size_t row, column, i, cone_index;
    PreFOSStatus status;

    memset(residuals, 0, sizeof(*residuals));
    if (!prefos_internal_certificate_vector_is_finite(ray, model->n))
        return PREFOS_STATUS_INVALID_ARGUMENT;
    for (column = 0; column < model->n; ++column)
    {
        update_maximum(fabsl((long double) ray[column]),
                       &direction_scale);
        update_maximum(
            fabsl((long double) model->c[column]), &linear_scale);
        objective +=
            (long double) model->c[column] *
            (long double) ray[column];
    }
    for (row = 0; row < model->A->rows; ++row)
    {
        long double activity = 0.0L;
        int p;
        for (p = model->A->row_pointers[row];
             p < model->A->row_pointers[row + 1]; ++p)
            activity +=
                (long double) model->A->values[p] *
                (long double) ray[model->A->column_indices[p]];
        if (!isfinite(activity))
            return PREFOS_STATUS_NUMERICAL_ERROR;
        interval_recession_violation(
            (double) activity, model->constraint_lower[row],
            model->constraint_upper[row], &row_violation);
    }
    for (i = 0; i < model->n_box; ++i)
        interval_recession_violation(
            ray[model->box_indices[i]], model->box_lower[i],
            model->box_upper[i], &domain_violation);
    for (cone_index = 0; cone_index < model->n_cones; ++cone_index)
    {
        status = evaluate_one_cone(
            &model->cones[cone_index], ray, tolerance, 0,
            &domain_violation);
        if (status != PREFOS_STATUS_OK) return status;
    }
    status = evaluate_affine_recession(
        model, ray, tolerance, &domain_violation);
    if (status != PREFOS_STATUS_OK) return status;
    status = evaluate_quadratic_ray(
        model, ray, &quadratic_violation, &quadratic_scale);
    if (status != PREFOS_STATUS_OK) return status;

    residuals->row_recession_violation =
        finite_residual(row_violation);
    residuals->domain_recession_violation =
        finite_residual(domain_violation);
    residuals->quadratic_null_violation =
        finite_residual(quadratic_violation);
    residuals->objective_direction = finite_value(objective);
    if (direction_scale > 0.0L && isfinite(direction_scale) &&
        isfinite(objective))
    {
        long double homogeneous_limit =
            (long double) tolerance * direction_scale;
        residuals->passed =
            row_violation <= homogeneous_limit &&
            domain_violation <= homogeneous_limit &&
            quadratic_violation <=
                homogeneous_limit * quadratic_scale &&
            objective <
                -homogeneous_limit * linear_scale;
    }
    return PREFOS_STATUS_OK;
}

static PreFOSStatus verify_infeasibility_certificate_internal(
    const PreFOSPresolver *presolver, const double *reduced_y,
    const double *reduced_z, const double *reduced_affine_z,
    double tolerance, int extended,
    PreFOSPostsolveInfeasibilityCertificateVerification *verification)
{
    PreFOSCertificateModel original, reduced;
    double *original_y = NULL, *original_z = NULL;
    double *original_affine_z = NULL;
    double value_scale;
    PreFOSStatus status;

    if (!presolver || !presolver->has_run || !verification ||
        !isfinite(tolerance) || tolerance < 0.0 ||
        (presolver->reduced.A.rows > 0 && !reduced_y) ||
        (presolver->reduced.n > 0 && !reduced_z) ||
        (presolver->reduced.affine_cone_matrix.rows > 0 &&
         !reduced_affine_z))
        return PREFOS_STATUS_INVALID_ARGUMENT;
    memset(verification, 0, sizeof(*verification));
    original_y = (double *) prefos_internal_alloc_array(
        presolver->original.A.rows, sizeof(double));
    original_z = (double *) prefos_internal_alloc_array(
        presolver->original.n, sizeof(double));
    original_affine_z = (double *) prefos_internal_alloc_array(
        presolver->original.affine_cone_matrix.rows, sizeof(double));
    if ((presolver->original.A.rows > 0 && !original_y) ||
        (presolver->original.n > 0 && !original_z) ||
        (presolver->original.affine_cone_matrix.rows > 0 &&
         !original_affine_z))
    {
        status = PREFOS_STATUS_OUT_OF_MEMORY;
        goto cleanup;
    }
    status = extended
                 ? prefos_postsolve_extended_infeasibility_certificate(
                       presolver, reduced_y, reduced_z, reduced_affine_z,
                       tolerance, original_y, original_z,
                       original_affine_z)
                 : prefos_postsolve_infeasibility_certificate(
                       presolver, reduced_y, reduced_z, reduced_affine_z,
                       tolerance, original_y, original_z,
                       original_affine_z);
    if (status != PREFOS_STATUS_OK) goto cleanup;
    original = prefos_internal_original_certificate_model(presolver);
    reduced = prefos_internal_reduced_certificate_model(presolver);
    status = prefos_internal_evaluate_infeasibility_certificate(
        &reduced, reduced_y, reduced_z, reduced_affine_z, tolerance,
        NULL, &verification->reduced);
    if (status != PREFOS_STATUS_OK) goto cleanup;
    status = prefos_internal_evaluate_infeasibility_certificate(
        &original, original_y, original_z, original_affine_z,
        tolerance, extended ? presolver : NULL,
        &verification->original);
    if (status != PREFOS_STATUS_OK) goto cleanup;
    verification->certificate_value_absolute_error =
        fabs(verification->original.certificate_value -
             verification->reduced.certificate_value);
    value_scale =
        fmax(1.0,
             fmax(fabs(verification->original.certificate_value),
                  fabs(verification->reduced.certificate_value)));
    verification->passed =
        verification->reduced.passed &&
        verification->original.passed &&
        verification->certificate_value_absolute_error <=
            tolerance * value_scale;

cleanup:
    free(original_y);
    free(original_z);
    free(original_affine_z);
    return status;
}

PreFOSStatus prefos_verify_postsolve_infeasibility_certificate(
    const PreFOSPresolver *presolver, const double *reduced_y,
    const double *reduced_z, const double *reduced_affine_z,
    double tolerance,
    PreFOSPostsolveInfeasibilityCertificateVerification *verification)
{
    return verify_infeasibility_certificate_internal(
        presolver, reduced_y, reduced_z, reduced_affine_z, tolerance,
        0, verification);
}

PreFOSStatus prefos_verify_postsolve_extended_infeasibility_certificate(
    const PreFOSPresolver *presolver, const double *reduced_y,
    const double *reduced_z, const double *reduced_affine_z,
    double tolerance,
    PreFOSPostsolveInfeasibilityCertificateVerification *verification)
{
    return verify_infeasibility_certificate_internal(
        presolver, reduced_y, reduced_z, reduced_affine_z, tolerance,
        1, verification);
}

PreFOSStatus prefos_verify_postsolve_unbounded_ray(
    const PreFOSPresolver *presolver, const double *reduced_ray,
    double tolerance, PreFOSPostsolveUnboundedRayVerification *verification)
{
    PreFOSCertificateModel original, reduced;
    double *original_ray;
    double objective_scale;
    PreFOSStatus status;
    if (!presolver || !presolver->has_run || !verification ||
        !isfinite(tolerance) || tolerance < 0.0 ||
        (presolver->reduced.n > 0 && !reduced_ray))
        return PREFOS_STATUS_INVALID_ARGUMENT;
    memset(verification, 0, sizeof(*verification));
    original_ray = (double *) prefos_internal_alloc_array(
        presolver->original.n, sizeof(double));
    if (presolver->original.n > 0 && !original_ray)
        return PREFOS_STATUS_OUT_OF_MEMORY;
    status = prefos_postsolve_unbounded_ray(
        presolver, reduced_ray, tolerance, original_ray);
    if (status != PREFOS_STATUS_OK)
    {
        free(original_ray);
        return status;
    }
    original = prefos_internal_original_certificate_model(presolver);
    reduced = prefos_internal_reduced_certificate_model(presolver);
    status = prefos_internal_evaluate_unbounded_ray(
        &reduced, reduced_ray, tolerance, &verification->reduced);
    if (status == PREFOS_STATUS_OK)
        status = prefos_internal_evaluate_unbounded_ray(
            &original, original_ray, tolerance, &verification->original);
    free(original_ray);
    if (status != PREFOS_STATUS_OK) return status;
    verification->objective_direction_absolute_error =
        fabs(verification->original.objective_direction -
             verification->reduced.objective_direction);
    objective_scale =
        fmax(1.0,
             fmax(fabs(verification->original.objective_direction),
                  fabs(verification->reduced.objective_direction)));
    verification->passed =
        verification->reduced.passed &&
        verification->original.passed &&
        verification->objective_direction_absolute_error <=
            tolerance * objective_scale;
    return PREFOS_STATUS_OK;
}
