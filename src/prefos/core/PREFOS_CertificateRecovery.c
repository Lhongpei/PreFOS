/*
 * Copyright 2026 Hongpei Li
 * SPDX-License-Identifier: Apache-2.0
 */

#include "PREFOS_CertificateInternal.h"
#include "cones/PREFOS_PositiveSemidefiniteCone.h"

static PreFOSStatus certificate_bound_row(
    const PreFOSPresolver *presolver,
    const PresolveBoundChangeRecord *record, const int **indices,
    const double **coefficients, size_t *length)
{
    if (record->row < 0 ||
        (size_t) record->row >= presolver->original.A.rows)
        return PREFOS_STATUS_NUMERICAL_ERROR;
    if (record->has_source_row_record)
    {
        const PresolveRowTransformationRecord *source;
        if (record->source_row_record_index >=
            presolver->transformations.n_row_transformations)
            return PREFOS_STATUS_NUMERICAL_ERROR;
        source = &presolver->transformations.row_transformations
                      [record->source_row_record_index];
        if (source->type != PRESOLVE_ROW_BOUND_CHANGE_SOURCE ||
            source->row != record->row ||
            (source->length > 0 &&
             (!source->indices || !source->coefficients)))
            return PREFOS_STATUS_NUMERICAL_ERROR;
        *indices = source->indices;
        *coefficients = source->coefficients;
        *length = source->length;
    }
    else
    {
        int begin = presolver->original.A.row_pointers[record->row];
        int end = presolver->original.A.row_pointers[record->row + 1];
        *indices = presolver->original.A.column_indices + begin;
        *coefficients = presolver->original.A.values + begin;
        *length = (size_t) (end - begin);
    }
    return PREFOS_STATUS_OK;
}

static PreFOSStatus transfer_certificate_bound(
    const PreFOSPresolver *presolver,
    const PresolveBoundChangeRecord *record, double *y, double *z,
    int *changed)
{
    const int *indices = NULL;
    const double *coefficients = NULL;
    size_t length = 0, position;
    double coefficient = 0.0, delta;
    double multiplier;
    PreFOSStatus status;

    if (record->column < 0 ||
        (size_t) record->column >= presolver->original.n)
        return PREFOS_STATUS_NUMERICAL_ERROR;
    multiplier = z[record->column];
    if ((record->is_lower && multiplier >= 0.0) ||
        (!record->is_lower && multiplier <= 0.0))
        return PREFOS_STATUS_OK;
    status = certificate_bound_row(
        presolver, record, &indices, &coefficients, &length);
    if (status != PREFOS_STATUS_OK) return status;
    for (position = 0; position < length; ++position)
        if (indices[position] == record->column)
        {
            coefficient = coefficients[position];
            break;
        }
    if (coefficient == 0.0 ||
        !prefos_internal_safe_product(
            1.0 / coefficient, multiplier, &delta) ||
        !prefos_internal_safe_add_product(
            &y[record->row], 1.0, delta))
        return PREFOS_STATUS_NUMERICAL_ERROR;
    for (position = 0; position < length; ++position)
    {
        int column = indices[position];
        if (column == record->column) continue;
        if (column < 0 || (size_t) column >= presolver->original.n ||
            !prefos_internal_safe_add_product(
                &z[column], -coefficients[position], delta))
            return PREFOS_STATUS_NUMERICAL_ERROR;
    }
    z[record->column] = 0.0;
    *changed = 1;
    return PREFOS_STATUS_OK;
}

static PreFOSStatus replay_certificate_bound_block(
    const PreFOSPresolver *presolver, size_t first_event,
    size_t past_last_event, double *y, double *z)
{
    size_t maximum_sweeps;
    size_t sweep;
    if (past_last_event < first_event)
        return PREFOS_STATUS_NUMERICAL_ERROR;
    maximum_sweeps = past_last_event - first_event + 1;
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
            status = transfer_certificate_bound(
                presolver,
                &presolver->transformations
                     .bound_changes[event->record_index],
                y, z, &event_changed);
            if (status != PREFOS_STATUS_OK) return status;
            changed |= event_changed;
        }
        if (!changed) return PREFOS_STATUS_OK;
    }
    return PREFOS_STATUS_DUAL_RECOVERY_UNAVAILABLE;
}

static PreFOSStatus replay_certificate_row(
    const PresolveRowTransformationRecord *record, double *y)
{
    double multiplier;
    if (record->row < 0) return PREFOS_STATUS_NUMERICAL_ERROR;
    if (record->type == PRESOLVE_ROW_DELETED)
    {
        y[record->row] = 0.0;
        return PREFOS_STATUS_OK;
    }
    if (record->type == PRESOLVE_ROW_EQUALITY_RELAXED)
        return PREFOS_STATUS_OK;
    if (record->type != PRESOLVE_ROW_LOWER_CHANGED &&
        record->type != PRESOLVE_ROW_UPPER_CHANGED)
        return PREFOS_STATUS_NUMERICAL_ERROR;
    multiplier = y[record->row];
    if ((record->type == PRESOLVE_ROW_LOWER_CHANGED &&
         multiplier >= 0.0) ||
        (record->type == PRESOLVE_ROW_UPPER_CHANGED &&
         multiplier <= 0.0))
        return PREFOS_STATUS_OK;
    if (record->source_row < 0 ||
        !prefos_internal_safe_add_product(
            &y[record->source_row], record->ratio, multiplier))
        return PREFOS_STATUS_NUMERICAL_ERROR;
    y[record->row] = 0.0;
    return PREFOS_STATUS_OK;
}

static PreFOSStatus shift_certificate_cone_pivot(
    const PreFOSPresolver *presolver, int column, long double decrease,
    double *y, double *z)
{
    int row;
    int p;
    double coefficient = 0.0;
    long double delta;
    if (decrease <= 0.0L) return PREFOS_STATUS_OK;
    if (column < 0 || (size_t) column >= presolver->original.n ||
        decrease > (long double) DBL_MAX)
        return PREFOS_STATUS_NUMERICAL_ERROR;
    row = presolver->cone_collapse_source_rows[column];
    if (row < 0 || (size_t) row >= presolver->original.A.rows)
        return PREFOS_STATUS_NUMERICAL_ERROR;
    for (p = presolver->original.A.row_pointers[row];
         p < presolver->original.A.row_pointers[row + 1]; ++p)
        if (presolver->original.A.column_indices[p] == column)
        {
            coefficient = presolver->original.A.values[p];
            break;
        }
    if (coefficient == 0.0) return PREFOS_STATUS_NUMERICAL_ERROR;
    delta = decrease / (long double) coefficient;
    if (!isfinite(delta) || fabsl(delta) > (long double) DBL_MAX ||
        !prefos_internal_safe_add_product(
            &y[row], 1.0, (double) delta) ||
        !prefos_internal_safe_add_product(
            &z[column], -1.0, (double) decrease))
        return PREFOS_STATUS_NUMERICAL_ERROR;
    return PREFOS_STATUS_OK;
}

static PreFOSStatus repair_certificate_collapsed_cone(
    const PreFOSPresolver *presolver, size_t cone_index, double tolerance,
    double *y, double *z)
{
    const PreFOSConeBlock *cone;
    size_t coordinate;
    PreFOSStatus status;
    if (cone_index >= presolver->original.n_cones ||
        !presolver->remove_cones[cone_index] ||
        presolver->cone_face_survivors[cone_index] >= 0 ||
        presolver->psd_face_reductions[cone_index].n_removed > 0)
        return PREFOS_STATUS_NUMERICAL_ERROR;
    cone = &presolver->original.cones[cone_index];
    if (cone->type == PREFOS_CONE_NONNEGATIVE)
    {
        for (coordinate = 0; coordinate < cone->dimension; ++coordinate)
        {
            int column = cone->indices[coordinate];
            status = shift_certificate_cone_pivot(
                presolver, column,
                fmaxl(0.0L, (long double) z[column]), y, z);
            if (status != PREFOS_STATUS_OK) return status;
        }
        return PREFOS_STATUS_OK;
    }
    if (cone->type == PREFOS_CONE_SECOND_ORDER)
    {
        int t_column = cone->indices[0];
        long double norm_squared = 0.0L;
        for (coordinate = 1; coordinate < cone->dimension; ++coordinate)
        {
            long double value =
                (long double) z[cone->indices[coordinate]];
            norm_squared += value * value;
        }
        return shift_certificate_cone_pivot(
            presolver, t_column,
            fmaxl(0.0L,
                  (long double) z[t_column] + sqrtl(norm_squared)),
            y, z);
    }
    if (cone->type == PREFOS_CONE_ROTATED_SECOND_ORDER)
    {
        int u_column = cone->indices[0];
        int v_column = cone->indices[1];
        long double norm_squared = 0.0L;
        long double minimum_diagonal;
        for (coordinate = 2; coordinate < cone->dimension; ++coordinate)
        {
            long double value =
                (long double) z[cone->indices[coordinate]];
            norm_squared += value * value;
        }
        minimum_diagonal = sqrtl(norm_squared / 2.0L);
        status = shift_certificate_cone_pivot(
            presolver, u_column,
            fmaxl(0.0L,
                  minimum_diagonal + (long double) z[u_column]),
            y, z);
        if (status != PREFOS_STATUS_OK) return status;
        return shift_certificate_cone_pivot(
            presolver, v_column,
            fmaxl(0.0L,
                  minimum_diagonal + (long double) z[v_column]),
            y, z);
    }
    if (cone->type == PREFOS_CONE_POSITIVE_SEMIDEFINITE)
    {
        double *negative_z =
            (double *) calloc(presolver->original.n, sizeof(double));
        long double violation = 0.0L;
        if (presolver->original.n > 0 && !negative_z)
            return PREFOS_STATUS_OUT_OF_MEMORY;
        for (coordinate = 0; coordinate < cone->dimension; ++coordinate)
        {
            int column = cone->indices[coordinate];
            negative_z[column] = -z[column];
        }
        status = prefos_internal_evaluate_psd_violation(
            cone, negative_z, tolerance, &violation);
        free(negative_z);
        if (status != PREFOS_STATUS_OK) return status;
        for (coordinate = 0; coordinate < cone->matrix_order; ++coordinate)
        {
            size_t diagonal =
                coordinate * (coordinate + 1U) / 2U + coordinate;
            status = shift_certificate_cone_pivot(
                presolver, cone->indices[diagonal], violation, y, z);
            if (status != PREFOS_STATUS_OK) return status;
        }
        return PREFOS_STATUS_OK;
    }
    return PREFOS_STATUS_DUAL_RECOVERY_UNAVAILABLE;
}

static int values_close(double left, double right, double tolerance)
{
    double scale = fmax(1.0, fmax(fabs(left), fabs(right)));
    return fabs(left - right) <= tolerance * scale;
}

static PreFOSStatus split_certificate_bounded_normal(
    const PresolveColumnTransformationRecord *record, double tolerance,
    double *z)
{
    long double mapped_lower, mapped_upper;
    long double reduced_lower, reduced_upper;
    double aggregate, source_normal = 0.0, target_normal = 0.0;
    double alpha = record->ratio;
    int target_active, source_active;

    if (record->column < 0 || record->secondary_column < 0 ||
        alpha == 0.0 || !isfinite(alpha) ||
        !isfinite(record->value))
        return PREFOS_STATUS_NUMERICAL_ERROR;
    aggregate = z[record->secondary_column];
    if (alpha > 0.0)
    {
        mapped_lower =
            ((long double) record->lower -
             (long double) record->value) /
            (long double) alpha;
        mapped_upper =
            ((long double) record->upper -
             (long double) record->value) /
            (long double) alpha;
    }
    else
    {
        mapped_lower =
            ((long double) record->upper -
             (long double) record->value) /
            (long double) alpha;
        mapped_upper =
            ((long double) record->lower -
             (long double) record->value) /
            (long double) alpha;
    }
    reduced_lower =
        fmaxl((long double) record->secondary_lower, mapped_lower);
    reduced_upper =
        fminl((long double) record->secondary_upper, mapped_upper);

    if (aggregate > 0.0)
    {
        if (!isfinite(reduced_upper))
            return PREFOS_STATUS_DUAL_RECOVERY_UNAVAILABLE;
        target_active =
            isfinite(record->secondary_upper) &&
            values_close(record->secondary_upper,
                         (double) reduced_upper, tolerance);
        source_active =
            isfinite(mapped_upper) &&
            values_close((double) mapped_upper,
                         (double) reduced_upper, tolerance);
    }
    else if (aggregate < 0.0)
    {
        if (!isfinite(reduced_lower))
            return PREFOS_STATUS_DUAL_RECOVERY_UNAVAILABLE;
        target_active =
            isfinite(record->secondary_lower) &&
            values_close(record->secondary_lower,
                         (double) reduced_lower, tolerance);
        source_active =
            isfinite(mapped_lower) &&
            values_close((double) mapped_lower,
                         (double) reduced_lower, tolerance);
    }
    else
    {
        z[record->column] = 0.0;
        return PREFOS_STATUS_OK;
    }
    if (target_active)
        target_normal = aggregate;
    else if (source_active)
        source_normal = aggregate / alpha;
    else
        return PREFOS_STATUS_DUAL_RECOVERY_UNAVAILABLE;
    if (!isfinite(source_normal) || !isfinite(target_normal))
        return PREFOS_STATUS_NUMERICAL_ERROR;
    z[record->column] = source_normal;
    z[record->secondary_column] = target_normal;
    return PREFOS_STATUS_OK;
}

static PreFOSStatus certificate_source_stationarity(
    const PreFOSPresolver *presolver,
    const PresolveColumnTransformationRecord *record, const double *y,
    long double *stationarity, double *source_coefficient)
{
    size_t position;
    *stationarity = 0.0L;
    *source_coefficient = 0.0;
    for (position = 0; position < record->length; ++position)
    {
        int row = record->indices[position];
        double coefficient = record->coefficients[position];
        if (row < 0 || (size_t) row >= presolver->original.A.rows)
            return PREFOS_STATUS_NUMERICAL_ERROR;
        if (row == record->source_row)
            *source_coefficient = coefficient;
        else
            *stationarity +=
                (long double) coefficient * (long double) y[row];
    }
    if (*source_coefficient == 0.0 || !isfinite(*stationarity))
        return PREFOS_STATUS_NUMERICAL_ERROR;
    return PREFOS_STATUS_OK;
}

static PreFOSStatus replay_certificate_affine_face_column(
    const PreFOSPresolver *presolver,
    const PresolveColumnTransformationRecord *record,
    double *pre_face_affine_z, const double *y, double *z)
{
    long double stationarity = 0.0L;
    long double multiplier;
    size_t affine_row, position;
    if ((record->type != PRESOLVE_COLUMN_FIXED &&
         record->type != PRESOLVE_COLUMN_SUBSTITUTED) ||
        record->source_row != -1 || record->column_tag >= 0 ||
        record->column < 0 ||
        (size_t) record->column >= presolver->original.n ||
        record->rhs == 0.0 || !pre_face_affine_z)
        return PREFOS_STATUS_NUMERICAL_ERROR;
    affine_row = (size_t) (-(long long) record->column_tag - 1LL);
    if (affine_row >= presolver->n_pre_face_affine_rows)
        return PREFOS_STATUS_NUMERICAL_ERROR;
    for (position = 0; position < record->length; ++position)
    {
        int row = record->indices[position];
        if (row < 0 || (size_t) row >= presolver->original.A.rows)
            return PREFOS_STATUS_NUMERICAL_ERROR;
        stationarity +=
            (long double) record->coefficients[position] *
            (long double) y[row];
    }
    for (position = 0; position < record->affine_length; ++position)
    {
        int row = record->affine_indices[position];
        if (row < 0 ||
            (size_t) row >= presolver->n_pre_face_affine_rows)
            return PREFOS_STATUS_NUMERICAL_ERROR;
        stationarity +=
            (long double) record->affine_coefficients[position] *
            (long double) pre_face_affine_z[row];
    }
    multiplier = -stationarity / (long double) record->rhs;
    if (!isfinite(multiplier) ||
        fabsl(multiplier) > (long double) DBL_MAX ||
        !prefos_internal_safe_add_product(
            &pre_face_affine_z[affine_row], 1.0,
            (double) multiplier))
        return PREFOS_STATUS_NUMERICAL_ERROR;
    z[record->column] = 0.0;
    return PREFOS_STATUS_OK;
}

static PreFOSStatus replay_certificate_column(
    const PreFOSPresolver *presolver,
    const PresolveColumnTransformationRecord *record, double tolerance,
    double *pre_face_affine_z, double *y, double *z)
{
    long double stationarity;
    double source_coefficient;
    PreFOSStatus status;

    if (record->type == PRESOLVE_COLUMN_FIXED_INFINITE)
    {
        if (record->column < 0 ||
            (size_t) record->column >= presolver->original.n)
            return PREFOS_STATUS_NUMERICAL_ERROR;
        z[record->column] = 0.0;
        return PREFOS_STATUS_OK;
    }
    if (record->type == PRESOLVE_COLUMNS_PARALLEL)
    {
        if (record->column < 0 || record->secondary_column < 0 ||
            (size_t) record->column >= presolver->original.n ||
            (size_t) record->secondary_column >= presolver->original.n ||
            !prefos_internal_safe_product(
                record->ratio, z[record->secondary_column],
                &z[record->column]))
            return PREFOS_STATUS_NUMERICAL_ERROR;
        return PREFOS_STATUS_OK;
    }
    if (record->source_row == -1 && record->column_tag < 0)
        return replay_certificate_affine_face_column(
            presolver, record, pre_face_affine_z, y, z);
    if (record->type != PRESOLVE_COLUMN_SUBSTITUTED ||
        record->column < 0 || record->source_row < 0 ||
        (size_t) record->column >= presolver->original.n ||
        (size_t) record->source_row >= presolver->original.A.rows)
        return PREFOS_STATUS_NUMERICAL_ERROR;
    if (presolver->affine_aggregation_source_rows[record->column] >= 0)
        return PREFOS_STATUS_OK;
    status = certificate_source_stationarity(
        presolver, record, y, &stationarity, &source_coefficient);
    if (status != PREFOS_STATUS_OK) return status;
    if (record->direction == PREFOS_SUBSTITUTION_BOUNDED_DOUBLETON)
    {
        status = split_certificate_bounded_normal(record, tolerance, z);
        if (status != PREFOS_STATUS_OK) return status;
    }
    else if (record->direction == PREFOS_SUBSTITUTION_RESIDUAL_ROW)
    {
        double residual_dual = y[record->source_row];
        if (!prefos_internal_safe_product(
                -source_coefficient, residual_dual,
                &z[record->column]))
            return PREFOS_STATUS_NUMERICAL_ERROR;
    }
    else if (record->direction == PREFOS_SUBSTITUTION_STANDARD)
        z[record->column] = 0.0;
    else
        return PREFOS_STATUS_NUMERICAL_ERROR;

    stationarity += (long double) z[record->column];
    if (fabsl(stationarity / (long double) source_coefficient) >
        (long double) DBL_MAX)
        return PREFOS_STATUS_NUMERICAL_ERROR;
    y[record->source_row] =
        (double) (-stationarity / (long double) source_coefficient);
    return PREFOS_STATUS_OK;
}

static PreFOSStatus add_zero_objective_stationarity(
    const PreFOSCertificateModel *model, const double *y,
    const double *affine_z, double *stationarity)
{
    size_t row;
    for (row = 0; row < model->n; ++row) stationarity[row] = 0.0;
    for (row = 0; row < model->A->rows; ++row)
    {
        int p;
        for (p = model->A->row_pointers[row];
             p < model->A->row_pointers[row + 1]; ++p)
        {
            int column = model->A->column_indices[p];
            if (!prefos_internal_safe_add_product(
                    &stationarity[column], model->A->values[p], y[row]))
                return PREFOS_STATUS_NUMERICAL_ERROR;
        }
    }
    for (row = 0; row < model->affine_cone_matrix->rows; ++row)
    {
        int p;
        for (p = model->affine_cone_matrix->row_pointers[row];
             p < model->affine_cone_matrix->row_pointers[row + 1]; ++p)
        {
            int column =
                model->affine_cone_matrix->column_indices[p];
            if (!prefos_internal_safe_add_product(
                    &stationarity[column],
                    model->affine_cone_matrix->values[p],
                    affine_z[row]))
                return PREFOS_STATUS_NUMERICAL_ERROR;
        }
    }
    return PREFOS_STATUS_OK;
}

static PreFOSStatus initialize_fixed_certificate_normals(
    const PreFOSPresolver *presolver,
    const PreFOSCertificateModel *model, const double *y,
    const double *affine_z, double *z)
{
    double *stationarity =
        (double *) prefos_internal_alloc_array(model->n, sizeof(double));
    size_t column;
    PreFOSStatus status;
    if (model->n > 0 && !stationarity)
        return PREFOS_STATUS_OUT_OF_MEMORY;
    status = add_zero_objective_stationarity(
        model, y, affine_z, stationarity);
    if (status == PREFOS_STATUS_OK)
        for (column = 0; column < model->n; ++column)
            if (presolver->is_fixed[column] &&
                !presolver->affine_face_eliminated_columns[column] &&
                presolver->affine_aggregation_source_rows[column] < 0)
                z[column] = -stationarity[column];
    free(stationarity);
    return status;
}

static PreFOSStatus recover_fixed_certificate_box_normals(
    const PreFOSPresolver *presolver,
    const PreFOSCertificateModel *model, const double *y,
    const double *affine_z, double *z)
{
    double *stationarity =
        (double *) prefos_internal_alloc_array(model->n, sizeof(double));
    size_t column;
    PreFOSStatus status;
    if (model->n > 0 && !stationarity)
        return PREFOS_STATUS_OUT_OF_MEMORY;
    status = add_zero_objective_stationarity(
        model, y, affine_z, stationarity);
    if (status == PREFOS_STATUS_OK)
        for (column = 0; column < model->n; ++column)
            if (presolver->is_fixed[column] &&
                !presolver->affine_face_eliminated_columns[column] &&
                presolver->variable_to_box[column] >= 0)
                z[column] = -stationarity[column];
    free(stationarity);
    return status;
}

static PreFOSStatus transfer_certificate_affine_bounds(
    const PreFOSPresolver *presolver, double tolerance, double *y,
    double *z, double *affine_z)
{
    size_t i;
    for (i = 0; i < presolver->n_affine_bound_certificates; ++i)
    {
        const PreFOSAffineBoundCertificate *certificate =
            &presolver->affine_bound_certificates[i];
        double multiplier, delta;
        if (certificate->column < 0 ||
            (size_t) certificate->column >= presolver->original.n ||
            certificate->affine_row >=
                presolver->n_pre_face_affine_rows ||
            certificate->coefficient == 0.0 ||
            !isfinite(certificate->coefficient))
            return PREFOS_STATUS_NUMERICAL_ERROR;
        multiplier = z[certificate->column];
        if ((certificate->is_lower && multiplier >= 0.0) ||
            (!certificate->is_lower && multiplier <= 0.0))
            continue;
        if (!prefos_internal_safe_product(
                1.0 / certificate->coefficient, multiplier, &delta))
            return PREFOS_STATUS_NUMERICAL_ERROR;
        if (delta >
            tolerance * fmax(1.0, fabs(multiplier)))
            return PREFOS_STATUS_DUAL_RECOVERY_UNAVAILABLE;
        if (certificate->generated_column < 0)
        {
            if (certificate->affine_row >=
                    presolver->original.affine_cone_matrix.rows ||
                !affine_z ||
                !prefos_internal_safe_add_product(
                    &affine_z[certificate->affine_row], 1.0, delta))
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
            pivot =
                presolver->affine_aggregation_pivots[generated_column];
            if (source_row < 0 ||
                (size_t) source_row >= presolver->original.A.rows ||
                pivot == 0.0 ||
                !prefos_internal_safe_product(
                    -1.0 / pivot, delta, &row_delta) ||
                !prefos_internal_safe_add_product(
                    &z[generated_column], 1.0, delta) ||
                !prefos_internal_safe_add_product(
                    &y[source_row], 1.0, row_delta))
                return PREFOS_STATUS_NUMERICAL_ERROR;
        }
        z[certificate->column] = 0.0;
    }
    return PREFOS_STATUS_OK;
}

static PreFOSStatus postsolve_infeasibility_certificate_internal(
    const PreFOSPresolver *presolver, const double *reduced_y,
    const double *reduced_z, const double *reduced_affine_z,
    double tolerance, double *original_y, double *original_z,
    double *original_affine_z, int extended)
{
    PreFOSCertificateModel original;
    double *pre_face_affine_z = NULL;
    size_t row, column, affine_row, event_index;
    PreFOSStatus status;

    if (!presolver || !presolver->has_run || !isfinite(tolerance) ||
        tolerance < 0.0 ||
        (presolver->original.A.rows > 0 && !original_y) ||
        (presolver->original.n > 0 && !original_z) ||
        (presolver->original.affine_cone_matrix.rows > 0 &&
         !original_affine_z) ||
        (presolver->reduced.A.rows > 0 && !reduced_y) ||
        (presolver->reduced.n > 0 && !reduced_z) ||
        (presolver->reduced.affine_cone_matrix.rows > 0 &&
         !reduced_affine_z) ||
        !prefos_internal_certificate_vector_is_finite(
            reduced_y, presolver->reduced.A.rows) ||
        !prefos_internal_certificate_vector_is_finite(
            reduced_z, presolver->reduced.n) ||
        !prefos_internal_certificate_vector_is_finite(
            reduced_affine_z,
            presolver->reduced.affine_cone_matrix.rows))
        return PREFOS_STATUS_INVALID_ARGUMENT;
    if (!extended &&
        (presolver->n_facial_reductions > 0 ||
         presolver->n_affine_face_substitutions > 0))
        return PREFOS_STATUS_DUAL_RECOVERY_UNAVAILABLE;
    if (presolver->n_pre_face_affine_rows > 0 &&
        !presolver->affine_pre_to_reduced_rows)
        return PREFOS_STATUS_NUMERICAL_ERROR;

    for (row = 0; row < presolver->original.A.rows; ++row)
    {
        int mapped = presolver->original_to_reduced_rows[row];
        original_y[row] = mapped < 0 ? 0.0 : reduced_y[mapped];
    }
    for (column = 0; column < presolver->original.n; ++column)
    {
        int mapped = presolver->original_to_reduced[column];
        original_z[column] = mapped < 0 ? 0.0 : reduced_z[mapped];
    }
    for (affine_row = 0;
         affine_row < presolver->original.affine_cone_matrix.rows;
         ++affine_row)
    {
        int mapped;
        if (affine_row >= presolver->n_pre_face_affine_rows)
            return PREFOS_STATUS_NUMERICAL_ERROR;
        mapped = presolver->affine_pre_to_reduced_rows[affine_row];
        if (mapped < -1 ||
            (mapped >= 0 &&
             (size_t) mapped >=
                 presolver->reduced.affine_cone_matrix.rows))
            return PREFOS_STATUS_NUMERICAL_ERROR;
        original_affine_z[affine_row] =
            mapped < 0 ? 0.0 : reduced_affine_z[mapped];
    }

    original = prefos_internal_original_certificate_model(presolver);
    status = initialize_fixed_certificate_normals(
        presolver, &original, original_y, original_affine_z, original_z);
    if (status != PREFOS_STATUS_OK) return status;

    pre_face_affine_z = (double *) prefos_internal_alloc_array(
        presolver->n_pre_face_affine_rows, sizeof(double));
    if (presolver->n_pre_face_affine_rows > 0 && !pre_face_affine_z)
        return PREFOS_STATUS_OUT_OF_MEMORY;
    for (affine_row = 0;
         affine_row < presolver->n_pre_face_affine_rows; ++affine_row)
    {
        int mapped = presolver->affine_pre_to_reduced_rows[affine_row];
        if (mapped < -1 ||
            (mapped >= 0 &&
             (size_t) mapped >=
                 presolver->reduced.affine_cone_matrix.rows))
        {
            status = PREFOS_STATUS_NUMERICAL_ERROR;
            goto cleanup;
        }
        pre_face_affine_z[affine_row] =
            mapped < 0 ? 0.0 : reduced_affine_z[mapped];
    }

    event_index = presolver->transformations.n_events;
    while (event_index > 0)
    {
        const PresolveTransformationEvent *event =
            &presolver->transformations.events[event_index - 1];
        if (event->type == PRESOLVE_TRANSFORMATION_BOUND_CHANGE)
        {
            size_t first = event_index - 1;
            while (first > 0 &&
                   presolver->transformations.events[first - 1].type ==
                       PRESOLVE_TRANSFORMATION_BOUND_CHANGE)
                --first;
            status = replay_certificate_bound_block(
                presolver, first, event_index, original_y, original_z);
            event_index = first;
        }
        else if (event->type == PRESOLVE_TRANSFORMATION_ROW)
        {
            const PresolveRowTransformationRecord *record =
                &presolver->transformations
                     .row_transformations[event->record_index];
            if (record->row < 0 ||
                (size_t) record->row >= presolver->original.A.rows ||
                (record->source_row >= 0 &&
                 (size_t) record->source_row >=
                     presolver->original.A.rows))
                status = PREFOS_STATUS_NUMERICAL_ERROR;
            else
                status = replay_certificate_row(record, original_y);
            --event_index;
        }
        else if (event->type == PRESOLVE_TRANSFORMATION_CONE)
        {
            const PresolveConeTransformationRecord *record =
                &presolver->transformations
                     .cone_transformations[event->record_index];
            if (record->type == PRESOLVE_CONE_FACE_REDUCED)
                status = extended ? PREFOS_STATUS_OK
                                  : PREFOS_STATUS_DUAL_RECOVERY_UNAVAILABLE;
            else if (record->type == PRESOLVE_CONE_COLLAPSED)
                status = repair_certificate_collapsed_cone(
                    presolver, record->cone_index, tolerance,
                    original_y, original_z);
            else
                status = PREFOS_STATUS_NUMERICAL_ERROR;
            --event_index;
        }
        else if (event->type == PRESOLVE_TRANSFORMATION_COLUMN)
        {
            status = replay_certificate_column(
                presolver,
                &presolver->transformations
                     .column_transformations[event->record_index],
                tolerance, pre_face_affine_z, original_y, original_z);
            --event_index;
        }
        else
        {
            status = PREFOS_STATUS_NUMERICAL_ERROR;
            --event_index;
        }
        if (status != PREFOS_STATUS_OK) goto cleanup;
    }

    for (affine_row = 0;
         affine_row < presolver->original.affine_cone_matrix.rows;
         ++affine_row)
        original_affine_z[affine_row] =
            pre_face_affine_z[affine_row];
    affine_row = presolver->original.affine_cone_matrix.rows;
    for (column = 0; column < presolver->original.n_cones; ++column)
    {
        const PreFOSConeBlock *cone =
            &presolver->original.cones[column];
        size_t coordinate;
        if (!presolver->converted_affine_cones[column]) continue;
        if (affine_row > presolver->n_pre_face_affine_rows ||
            cone->dimension >
                presolver->n_pre_face_affine_rows - affine_row)
        {
            status = PREFOS_STATUS_NUMERICAL_ERROR;
            goto cleanup;
        }
        for (coordinate = 0; coordinate < cone->dimension; ++coordinate)
        {
            int cone_column = cone->indices[coordinate];
            int source_row =
                presolver->affine_aggregation_source_rows[cone_column];
            double pivot =
                presolver->affine_aggregation_pivots[cone_column];
            double cone_normal = pre_face_affine_z[affine_row++];
            if (source_row < 0 ||
                (size_t) source_row >= presolver->original.A.rows ||
                pivot == 0.0)
            {
                status = PREFOS_STATUS_NUMERICAL_ERROR;
                goto cleanup;
            }
            original_z[cone_column] = cone_normal;
            original_y[source_row] = -cone_normal / pivot;
            if (!isfinite(original_y[source_row]))
            {
                status = PREFOS_STATUS_NUMERICAL_ERROR;
                goto cleanup;
            }
        }
    }
    if (affine_row != presolver->n_pre_face_affine_rows)
    {
        status = PREFOS_STATUS_NUMERICAL_ERROR;
        goto cleanup;
    }
    status = recover_fixed_certificate_box_normals(
        presolver, &original, original_y, original_affine_z, original_z);
    if (status == PREFOS_STATUS_OK)
        status = transfer_certificate_affine_bounds(
            presolver, tolerance, original_y, original_z,
            original_affine_z);

cleanup:
    free(pre_face_affine_z);
    return status;
}

PreFOSStatus prefos_postsolve_infeasibility_certificate(
    const PreFOSPresolver *presolver, const double *reduced_y,
    const double *reduced_z, const double *reduced_affine_z,
    double tolerance, double *original_y, double *original_z,
    double *original_affine_z)
{
    return postsolve_infeasibility_certificate_internal(
        presolver, reduced_y, reduced_z, reduced_affine_z, tolerance,
        original_y, original_z, original_affine_z, 0);
}

PreFOSStatus prefos_postsolve_extended_infeasibility_certificate(
    const PreFOSPresolver *presolver, const double *reduced_y,
    const double *reduced_z, const double *reduced_affine_z,
    double tolerance, double *original_y, double *original_z,
    double *original_affine_z)
{
    return postsolve_infeasibility_certificate_internal(
        presolver, reduced_y, reduced_z, reduced_affine_z, tolerance,
        original_y, original_z, original_affine_z, 1);
}
