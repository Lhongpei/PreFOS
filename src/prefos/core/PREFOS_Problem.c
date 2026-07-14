/*
 * Copyright 2026 Hongpei Li
 * SPDX-License-Identifier: Apache-2.0
 */

#include "PREFOS_Internal.h"

void prefos_internal_free_csr(PreFOSCsrMatrix *matrix)
{
    if (!matrix) return;
    free(matrix->values);
    free(matrix->column_indices);
    free(matrix->row_pointers);
    memset(matrix, 0, sizeof(*matrix));
}

static void free_cones(PreFOSConeBlock *cones, size_t n_cones)
{
    size_t k;
    if (!cones) return;
    for (k = 0; k < n_cones; ++k) free(cones[k].indices);
    free(cones);
}

void prefos_internal_free_psd_face_reductions(PreFOSPSDFaceReduction *faces,
                                           size_t n_cones)
{
    size_t k;
    if (!faces) return;
    for (k = 0; k < n_cones; ++k)
    {
        free(faces[k].removed_matrix_indices);
        free(faces[k].source_rows);
    }
    free(faces);
}

static void free_problem_data(PreFOSProblemData *problem)
{
    if (!problem) return;
    prefos_internal_free_csr(&problem->A);
    free(problem->constraint_lower);
    free(problem->constraint_upper);
    prefos_internal_free_csr(&problem->Q);
    prefos_internal_free_csr(&problem->R);
    free(problem->D);
    free(problem->c);
    free(problem->box_indices);
    free(problem->box_lower);
    free(problem->box_upper);
    free_cones(problem->cones, problem->n_cones);
    prefos_internal_free_csr(&problem->affine_cone_matrix);
    free(problem->affine_cone_offset);
    free(problem->affine_cones);
    memset(problem, 0, sizeof(*problem));
}

void prefos_internal_free_reduced_problem(PreFOSPresolvedProblem *problem)
{
    if (!problem) return;
    prefos_internal_free_csr(&problem->A);
    free(problem->constraint_lower);
    free(problem->constraint_upper);
    prefos_internal_free_csr(&problem->Q);
    prefos_internal_free_csr(&problem->R);
    free(problem->D);
    free(problem->c);
    free(problem->box_indices);
    free(problem->box_lower);
    free(problem->box_upper);
    free_cones(problem->cones, problem->n_cones);
    prefos_internal_free_csr(&problem->affine_cone_matrix);
    free(problem->affine_cone_offset);
    free(problem->affine_cones);
    memset(problem, 0, sizeof(*problem));
}

void *prefos_internal_alloc_array(size_t count, size_t element_size)
{
    if (count == 0) return NULL;
    if (element_size != 0 && count > SIZE_MAX / element_size) return NULL;
    return malloc(count * element_size);
}

double prefos_internal_safe_midpoint(double lower, double upper)
{
    return 0.5 * lower + 0.5 * upper;
}

int prefos_internal_values_close(double left, double right, double tolerance)
{
    double scale;
    if (!isfinite(left) || !isfinite(right)) return left == right;
    scale = fmax(1.0, fmax(fabs(left), fabs(right)));
    return fabs(left - right) <= tolerance * scale;
}

int prefos_internal_safe_add_product(double *accumulator, double left, double right)
{
    long double result =
        (long double) *accumulator + (long double) left * (long double) right;
    if (!isfinite(result) || fabsl(result) > (long double) DBL_MAX) return 0;
    *accumulator = (double) result;
    return 1;
}

int prefos_internal_safe_product(double left, double right, double *product)
{
    long double result = (long double) left * (long double) right;
    if (!isfinite(result) || fabsl(result) > (long double) DBL_MAX) return 0;
    *product = (double) result;
    return 1;
}

double prefos_internal_outward_bound_cast(long double value, int is_lower)
{
    double converted;
    if (!isfinite(value)) return value > 0.0L ? INFINITY : -INFINITY;
    if (value > (long double) DBL_MAX) return INFINITY;
    if (value < (long double) -DBL_MAX) return -INFINITY;

    converted = (double) value;
    if (is_lower && (long double) converted > value)
        converted = nextafter(converted, -INFINITY);
    if (!is_lower && (long double) converted < value)
        converted = nextafter(converted, INFINITY);
    return converted;
}

static PreFOSStatus validate_csr(const PreFOSCsrMatrix *matrix)
{
    size_t row;
    int previous;

    if (!matrix) return PREFOS_STATUS_INVALID_ARGUMENT;
    if (matrix->rows > (size_t) INT_MAX || matrix->cols > (size_t) INT_MAX)
        return PREFOS_STATUS_INVALID_ARGUMENT;
    if (matrix->nnz > (size_t) INT_MAX) return PREFOS_STATUS_INVALID_ARGUMENT;
    if (matrix->nnz > 0 &&
        (!matrix->values || !matrix->column_indices || !matrix->row_pointers))
    {
        return PREFOS_STATUS_INVALID_ARGUMENT;
    }
    if (matrix->rows > 0 && !matrix->row_pointers && matrix->nnz > 0)
        return PREFOS_STATUS_INVALID_ARGUMENT;
    if (!matrix->row_pointers) return PREFOS_STATUS_OK;
    if (matrix->row_pointers[0] != 0) return PREFOS_STATUS_INVALID_ARGUMENT;

    previous = 0;
    for (row = 0; row <= matrix->rows; ++row)
    {
        int current = matrix->row_pointers[row];
        if (current < previous || current < 0 || (size_t) current > matrix->nnz)
            return PREFOS_STATUS_INVALID_ARGUMENT;
        previous = current;
    }
    if ((size_t) previous != matrix->nnz) return PREFOS_STATUS_INVALID_ARGUMENT;

    for (row = 0; row < matrix->nnz; ++row)
    {
        int column = matrix->column_indices[row];
        if (column < 0 || (size_t) column >= matrix->cols ||
            !isfinite(matrix->values[row]))
        {
            return PREFOS_STATUS_INVALID_ARGUMENT;
        }
    }
    return PREFOS_STATUS_OK;
}

static PreFOSStatus validate_unique_columns_per_row(const PreFOSCsrMatrix *matrix)
{
    int *last_seen;
    size_t column, row;
    if (matrix->cols == 0 || matrix->nnz == 0) return PREFOS_STATUS_OK;

    last_seen = (int *) prefos_internal_alloc_array(matrix->cols, sizeof(int));
    if (!last_seen) return PREFOS_STATUS_OUT_OF_MEMORY;
    for (column = 0; column < matrix->cols; ++column) last_seen[column] = -1;

    for (row = 0; row < matrix->rows; ++row)
    {
        int p;
        for (p = matrix->row_pointers[row]; p < matrix->row_pointers[row + 1]; ++p)
        {
            int current_column = matrix->column_indices[p];
            if (last_seen[current_column] == (int) row)
            {
                free(last_seen);
                return PREFOS_STATUS_INVALID_ARGUMENT;
            }
            last_seen[current_column] = (int) row;
        }
    }
    free(last_seen);
    return PREFOS_STATUS_OK;
}

typedef struct
{
    int row;
    int column;
    double value;
} PreFOSMatrixEntry;

static int compare_matrix_entries(const void *left_pointer,
                                  const void *right_pointer)
{
    const PreFOSMatrixEntry *left = (const PreFOSMatrixEntry *) left_pointer;
    const PreFOSMatrixEntry *right = (const PreFOSMatrixEntry *) right_pointer;
    if (left->row != right->row) return left->row < right->row ? -1 : 1;
    if (left->column != right->column) return left->column < right->column ? -1 : 1;
    return 0;
}

static const PreFOSMatrixEntry *find_matrix_entry(const PreFOSMatrixEntry *entries,
                                               size_t count, int row, int column)
{
    size_t lower = 0, upper = count;
    while (lower < upper)
    {
        size_t middle = lower + (upper - lower) / 2;
        const PreFOSMatrixEntry *entry = &entries[middle];
        if (entry->row < row || (entry->row == row && entry->column < column))
            lower = middle + 1;
        else
            upper = middle;
    }
    if (lower < count && entries[lower].row == row &&
        entries[lower].column == column)
        return &entries[lower];
    return NULL;
}

static PreFOSStatus validate_full_q_symmetry(const PreFOSCsrMatrix *Q)
{
    PreFOSMatrixEntry *entries;
    size_t row, count = 0, i;
    entries =
        (PreFOSMatrixEntry *) prefos_internal_alloc_array(Q->nnz, sizeof(PreFOSMatrixEntry));
    if (Q->nnz > 0 && !entries) return PREFOS_STATUS_OUT_OF_MEMORY;

    for (row = 0; row < Q->rows; ++row)
    {
        int p;
        for (p = Q->row_pointers[row]; p < Q->row_pointers[row + 1]; ++p)
        {
            if (Q->values[p] == 0.0) continue;
            entries[count++] =
                (PreFOSMatrixEntry){(int) row, Q->column_indices[p], Q->values[p]};
        }
    }
    if (count > 1)
        qsort(entries, count, sizeof(PreFOSMatrixEntry), compare_matrix_entries);
    for (i = 0; i < count; ++i)
    {
        const PreFOSMatrixEntry *transpose;
        if (i > 0 && entries[i - 1].row == entries[i].row &&
            entries[i - 1].column == entries[i].column)
        {
            free(entries);
            return PREFOS_STATUS_INVALID_ARGUMENT;
        }
        transpose =
            find_matrix_entry(entries, count, entries[i].column, entries[i].row);
        if (!transpose || transpose->value != entries[i].value)
        {
            free(entries);
            return PREFOS_STATUS_INVALID_ARGUMENT;
        }
    }
    free(entries);
    return PREFOS_STATUS_OK;
}

static PreFOSStatus validate_q(const PreFOSProblemData *problem)
{
    size_t row;
    const PreFOSCsrMatrix *Q = &problem->Q;
    PreFOSStatus status = validate_csr(Q);
    if (status != PREFOS_STATUS_OK) return status;
    if (Q->rows != problem->n || Q->cols != problem->n)
        return PREFOS_STATUS_INVALID_ARGUMENT;
    if (problem->q_storage < PREFOS_Q_UPPER_TRIANGULAR ||
        problem->q_storage > PREFOS_Q_FULL)
    {
        return PREFOS_STATUS_INVALID_ARGUMENT;
    }
    if (!Q->row_pointers) return PREFOS_STATUS_OK;

    for (row = 0; row < Q->rows; ++row)
    {
        int p;
        for (p = Q->row_pointers[row]; p < Q->row_pointers[row + 1]; ++p)
        {
            int column = Q->column_indices[p];
            if (problem->q_storage == PREFOS_Q_UPPER_TRIANGULAR && column < (int) row)
            {
                return PREFOS_STATUS_INVALID_ARGUMENT;
            }
            if (problem->q_storage == PREFOS_Q_LOWER_TRIANGULAR && column > (int) row)
            {
                return PREFOS_STATUS_INVALID_ARGUMENT;
            }
        }
    }
    if (problem->q_storage == PREFOS_Q_FULL) return validate_full_q_symmetry(Q);
    return PREFOS_STATUS_OK;
}

static int cone_shape_is_valid(PreFOSConeType type, size_t dimension,
                               size_t matrix_order, double power_alpha,
                               size_t maximum_dimension)
{
    return dimension > 0 && dimension <= maximum_dimension &&
           type >= PREFOS_CONE_NONNEGATIVE && type <= PREFOS_CONE_POWER &&
           (type != PREFOS_CONE_SECOND_ORDER || dimension >= 2) &&
           (type != PREFOS_CONE_ROTATED_SECOND_ORDER || dimension >= 3) &&
           (type != PREFOS_CONE_EXPONENTIAL || dimension == 3) &&
           (type != PREFOS_CONE_POWER || (dimension == 3 && isfinite(power_alpha) &&
                                       power_alpha > 0.0 && power_alpha < 1.0)) &&
           (type != PREFOS_CONE_POSITIVE_SEMIDEFINITE ||
            (matrix_order > 0 && matrix_order <= maximum_dimension &&
             dimension == matrix_order * (matrix_order + 1) / 2));
}

static PreFOSStatus validate_problem(const PreFOSProblemData *problem,
                                  const PreFOSSettings *settings)
{
    size_t i, k;
    unsigned char *domain_owner;
    PreFOSStatus status;

    if (!problem || !settings || problem->n > (size_t) INT_MAX)
        return PREFOS_STATUS_INVALID_ARGUMENT;
    if (!isfinite(settings->feasibility_tolerance) ||
        settings->feasibility_tolerance < 0.0 ||
        !isfinite(settings->fixed_variable_tolerance) ||
        settings->fixed_variable_tolerance < 0.0 ||
        !isfinite(settings->finite_bound_improvement_absolute) ||
        settings->finite_bound_improvement_absolute < 0.0 ||
        !isfinite(settings->finite_bound_improvement_relative) ||
        settings->finite_bound_improvement_relative < 0.0 ||
        !isfinite(settings->event_queue_max_average_column_degree) ||
        settings->event_queue_max_average_column_degree < 0.0 ||
        !isfinite(settings->event_queue_activity_update_ratio) ||
        settings->event_queue_activity_update_ratio < 0.0 ||
        settings->max_aggregation_terms < 1 ||
        settings->max_aggregation_terms > PREFOS_MAX_AGGREGATION_TERMS ||
        settings->max_aggregation_column_degree < 1 ||
        settings->max_aggregation_column_degree > 8 ||
        settings->max_aggregation_fill < 0 || settings->max_aggregation_fill > 8 ||
        settings->max_aggregation_rounds < 1 ||
        settings->max_aggregation_rounds > 8 ||
        !isfinite(settings->max_aggregation_scale) ||
        settings->max_aggregation_scale < 1.0 ||
        settings->max_linear_propagation_rounds < 0 ||
        (settings->linear_propagation_gpu != 0 &&
         settings->linear_propagation_gpu != 1) ||
        !isfinite(settings->linear_propagation_max_work_ratio) ||
        settings->linear_propagation_max_work_ratio < 0.0 ||
        !isfinite(settings->linear_propagation_min_changes_per_million) ||
        settings->linear_propagation_min_changes_per_million < 0.0 ||
        settings->linear_propagation_max_stale_rounds < 0 ||
        settings->max_cone_propagation_rounds < 0 ||
        !isfinite(settings->affine_psd_propagation_max_work_ratio) ||
        settings->affine_psd_propagation_max_work_ratio < 0.0 ||
        settings->propagated_bound_policy <
            PREFOS_PROPAGATED_BOUND_POLICY_FIRST_ORDER ||
        settings->propagated_bound_policy >
            PREFOS_PROPAGATED_BOUND_POLICY_INTERIOR_POINT ||
        (settings->affine_cone_coordinate_aggregation != 0 &&
         settings->affine_cone_coordinate_aggregation != 1) ||
        (settings->psd_structure_analysis != 0 &&
         settings->psd_structure_analysis != 1) ||
        (settings->psd_block_decomposition != 0 &&
         settings->psd_block_decomposition != 1))
    {
        return PREFOS_STATUS_INVALID_ARGUMENT;
    }
    if (problem->A.cols != problem->n || problem->R.cols != problem->n ||
        ((problem->affine_cone_matrix.rows > 0 ||
          problem->affine_cone_matrix.nnz > 0 ||
          problem->affine_cone_matrix.cols > 0) &&
         problem->affine_cone_matrix.cols != problem->n))
        return PREFOS_STATUS_INVALID_ARGUMENT;
    if ((problem->A.rows > 0 &&
         (!problem->constraint_lower || !problem->constraint_upper)) ||
        (problem->n > 0 && !problem->c) ||
        (problem->n_box > 0 &&
         (!problem->box_indices || !problem->box_lower || !problem->box_upper)) ||
        (problem->n_cones > 0 && !problem->cones) ||
        (problem->affine_cone_matrix.rows > 0 && !problem->affine_cone_offset) ||
        (problem->n_affine_cones > 0 && !problem->affine_cones) ||
        (problem->R.rows > 0 && !problem->D))
    {
        return PREFOS_STATUS_INVALID_ARGUMENT;
    }
    if (!isfinite(problem->objective_offset)) return PREFOS_STATUS_INVALID_ARGUMENT;

    status = validate_csr(&problem->A);
    if (status != PREFOS_STATUS_OK) return status;
    status = validate_unique_columns_per_row(&problem->A);
    if (status != PREFOS_STATUS_OK) return status;
    status = validate_q(problem);
    if (status != PREFOS_STATUS_OK) return status;
    status = validate_csr(&problem->R);
    if (status != PREFOS_STATUS_OK) return status;
    status = validate_csr(&problem->affine_cone_matrix);
    if (status != PREFOS_STATUS_OK) return status;
    status = validate_unique_columns_per_row(&problem->affine_cone_matrix);
    if (status != PREFOS_STATUS_OK) return status;

    for (i = 0; i < problem->A.rows; ++i)
    {
        double lower = problem->constraint_lower[i];
        double upper = problem->constraint_upper[i];
        if (isnan(lower) || isnan(upper)) return PREFOS_STATUS_INVALID_ARGUMENT;
        if (lower == INFINITY || upper == -INFINITY)
            return PREFOS_STATUS_PRIMAL_INFEASIBLE;
        if (lower > upper + settings->feasibility_tolerance)
            return PREFOS_STATUS_PRIMAL_INFEASIBLE;
    }
    for (i = 0; i < problem->n; ++i)
        if (!isfinite(problem->c[i])) return PREFOS_STATUS_INVALID_ARGUMENT;
    for (i = 0; i < problem->R.rows; ++i)
    {
        if (!isfinite(problem->D[i])) return PREFOS_STATUS_INVALID_ARGUMENT;
        if (problem->D[i] < -settings->feasibility_tolerance)
            return PREFOS_STATUS_INVALID_ARGUMENT;
    }

    domain_owner = (unsigned char *) calloc(problem->n, sizeof(unsigned char));
    if (problem->n > 0 && !domain_owner) return PREFOS_STATUS_OUT_OF_MEMORY;
    for (i = 0; i < problem->n_box; ++i)
    {
        int index = problem->box_indices[i];
        double lower = problem->box_lower[i];
        double upper = problem->box_upper[i];
        if (index < 0 || (size_t) index >= problem->n || domain_owner[index] ||
            isnan(lower) || isnan(upper))
        {
            free(domain_owner);
            return PREFOS_STATUS_INVALID_ARGUMENT;
        }
        if (lower == INFINITY || upper == -INFINITY)
        {
            free(domain_owner);
            return PREFOS_STATUS_PRIMAL_INFEASIBLE;
        }
        if (lower > upper + settings->feasibility_tolerance)
        {
            free(domain_owner);
            return PREFOS_STATUS_PRIMAL_INFEASIBLE;
        }
        domain_owner[index] = 1;
    }

    for (k = 0; k < problem->n_cones; ++k)
    {
        const PreFOSConeBlock *cone = &problem->cones[k];
        if (!cone->indices ||
            !cone_shape_is_valid(cone->type, cone->dimension, cone->matrix_order,
                                 cone->power_alpha, problem->n))
        {
            free(domain_owner);
            return PREFOS_STATUS_INVALID_ARGUMENT;
        }
        for (i = 0; i < cone->dimension; ++i)
        {
            int index = cone->indices[i];
            if (index < 0 || (size_t) index >= problem->n || domain_owner[index])
            {
                free(domain_owner);
                return PREFOS_STATUS_INVALID_ARGUMENT;
            }
            domain_owner[index] = 1;
        }
    }

    {
        size_t affine_rows = 0;
        for (k = 0; k < problem->n_affine_cones; ++k)
        {
            const PreFOSAffineConeBlock *cone = &problem->affine_cones[k];
            if (!cone_shape_is_valid(cone->type, cone->dimension, cone->matrix_order,
                                     cone->power_alpha,
                                     problem->affine_cone_matrix.rows) ||
                cone->dimension > problem->affine_cone_matrix.rows - affine_rows)
            {
                free(domain_owner);
                return PREFOS_STATUS_INVALID_ARGUMENT;
            }
            affine_rows += cone->dimension;
        }
        if (affine_rows != problem->affine_cone_matrix.rows)
        {
            free(domain_owner);
            return PREFOS_STATUS_INVALID_ARGUMENT;
        }
        for (i = 0; i < problem->affine_cone_matrix.rows; ++i)
        {
            if (!isfinite(problem->affine_cone_offset[i]))
            {
                free(domain_owner);
                return PREFOS_STATUS_INVALID_ARGUMENT;
            }
        }
    }

    for (i = 0; i < problem->n; ++i)
    {
        if (!domain_owner[i])
        {
            free(domain_owner);
            return PREFOS_STATUS_INVALID_ARGUMENT;
        }
    }
    free(domain_owner);
    return PREFOS_STATUS_OK;
}

static PreFOSStatus copy_csr(const PreFOSCsrMatrix *source, PreFOSCsrMatrix *target)
{
    size_t row_pointer_count = source->rows + 1;
    memset(target, 0, sizeof(*target));
    target->rows = source->rows;
    target->cols = source->cols;
    target->nnz = source->nnz;
    target->row_pointers = (int *) calloc(row_pointer_count, sizeof(int));
    target->values =
        (double *) prefos_internal_alloc_array(source->nnz, sizeof(double));
    target->column_indices =
        (int *) prefos_internal_alloc_array(source->nnz, sizeof(int));
    if (!target->row_pointers ||
        (source->nnz > 0 && (!target->values || !target->column_indices)))
    {
        prefos_internal_free_csr(target);
        return PREFOS_STATUS_OUT_OF_MEMORY;
    }
    if (source->row_pointers)
        memcpy(target->row_pointers, source->row_pointers,
               row_pointer_count * sizeof(int));
    if (source->nnz > 0)
    {
        memcpy(target->values, source->values, source->nnz * sizeof(double));
        memcpy(target->column_indices, source->column_indices,
               source->nnz * sizeof(int));
    }
    return PREFOS_STATUS_OK;
}

PreFOSStatus prefos_internal_copy_vector(const void *source, size_t count,
                                   size_t element_size, void **target)
{
    *target = prefos_internal_alloc_array(count, element_size);
    if (count > 0 && !*target) return PREFOS_STATUS_OUT_OF_MEMORY;
    if (count > 0) memcpy(*target, source, count * element_size);
    return PREFOS_STATUS_OK;
}

static PreFOSStatus copy_cones(const PreFOSConeBlock *source, size_t n_cones,
                            PreFOSConeBlock **target)
{
    size_t k;
    *target = (PreFOSConeBlock *) calloc(n_cones, sizeof(PreFOSConeBlock));
    if (n_cones > 0 && !*target) return PREFOS_STATUS_OUT_OF_MEMORY;
    for (k = 0; k < n_cones; ++k)
    {
        (*target)[k] = source[k];
        (*target)[k].indices = NULL;
        if (prefos_internal_copy_vector(source[k].indices, source[k].dimension,
                                     sizeof(int), (void **) &(*target)[k].indices) !=
            PREFOS_STATUS_OK)
        {
            free_cones(*target, n_cones);
            *target = NULL;
            return PREFOS_STATUS_OUT_OF_MEMORY;
        }
    }
    return PREFOS_STATUS_OK;
}

static PreFOSStatus copy_problem(const PreFOSProblemData *source, PreFOSProblemData *target)
{
    PreFOSStatus status;
    memset(target, 0, sizeof(*target));
    target->n = source->n;
    target->q_storage = source->q_storage;
    target->objective_offset = source->objective_offset;
    target->n_box = source->n_box;
    target->n_cones = source->n_cones;
    target->n_affine_cones = source->n_affine_cones;

#define PREFOS_COPY_OR_FAIL(expression)                                                \
    do                                                                              \
    {                                                                               \
        status = (expression);                                                      \
        if (status != PREFOS_STATUS_OK) goto failure;                                  \
    } while (0)

    PREFOS_COPY_OR_FAIL(copy_csr(&source->A, &target->A));
    PREFOS_COPY_OR_FAIL(prefos_internal_copy_vector(source->constraint_lower,
                                              source->A.rows, sizeof(double),
                                              (void **) &target->constraint_lower));
    PREFOS_COPY_OR_FAIL(prefos_internal_copy_vector(source->constraint_upper,
                                              source->A.rows, sizeof(double),
                                              (void **) &target->constraint_upper));
    PREFOS_COPY_OR_FAIL(copy_csr(&source->Q, &target->Q));
    PREFOS_COPY_OR_FAIL(copy_csr(&source->R, &target->R));
    PREFOS_COPY_OR_FAIL(prefos_internal_copy_vector(source->D, source->R.rows,
                                              sizeof(double), (void **) &target->D));
    PREFOS_COPY_OR_FAIL(prefos_internal_copy_vector(source->c, source->n, sizeof(double),
                                              (void **) &target->c));
    PREFOS_COPY_OR_FAIL(prefos_internal_copy_vector(source->box_indices, source->n_box,
                                              sizeof(int),
                                              (void **) &target->box_indices));
    PREFOS_COPY_OR_FAIL(prefos_internal_copy_vector(source->box_lower, source->n_box,
                                              sizeof(double),
                                              (void **) &target->box_lower));
    PREFOS_COPY_OR_FAIL(prefos_internal_copy_vector(source->box_upper, source->n_box,
                                              sizeof(double),
                                              (void **) &target->box_upper));
    PREFOS_COPY_OR_FAIL(copy_cones(source->cones, source->n_cones, &target->cones));
    PREFOS_COPY_OR_FAIL(
        copy_csr(&source->affine_cone_matrix, &target->affine_cone_matrix));
    if (target->affine_cone_matrix.rows == 0 && target->affine_cone_matrix.cols == 0)
        target->affine_cone_matrix.cols = source->n;
    PREFOS_COPY_OR_FAIL(prefos_internal_copy_vector(
        source->affine_cone_offset, source->affine_cone_matrix.rows, sizeof(double),
        (void **) &target->affine_cone_offset));
    PREFOS_COPY_OR_FAIL(prefos_internal_copy_vector(
        source->affine_cones, source->n_affine_cones, sizeof(PreFOSAffineConeBlock),
        (void **) &target->affine_cones));
#undef PREFOS_COPY_OR_FAIL
    return PREFOS_STATUS_OK;

failure:
    free_problem_data(target);
    return status;
}

static void canonicalize_copied_problem(PreFOSProblemData *problem)
{
    size_t i;
    for (i = 0; i < problem->A.rows; ++i)
    {
        if (problem->constraint_lower[i] > problem->constraint_upper[i])
        {
            double midpoint = prefos_internal_safe_midpoint(
                problem->constraint_lower[i], problem->constraint_upper[i]);
            problem->constraint_lower[i] = midpoint;
            problem->constraint_upper[i] = midpoint;
        }
    }
    for (i = 0; i < problem->n_box; ++i)
    {
        if (problem->box_lower[i] > problem->box_upper[i])
        {
            double midpoint = prefos_internal_safe_midpoint(problem->box_lower[i],
                                                         problem->box_upper[i]);
            problem->box_lower[i] = midpoint;
            problem->box_upper[i] = midpoint;
        }
    }
    for (i = 0; i < problem->R.rows; ++i)
        if (problem->D[i] < 0.0) problem->D[i] = 0.0;
}

static PreFOSStatus normalize_nonnegative_cones(PreFOSPresolver *presolver)
{
    PreFOSProblemData *problem = &presolver->original;
    int *box_indices;
    double *box_lower, *box_upper;
    size_t k, i, box_write, cone_write = 0, added_variables = 0, removed_cones = 0;

    for (k = 0; k < problem->n_cones; ++k)
    {
        if (problem->cones[k].type != PREFOS_CONE_NONNEGATIVE &&
            !(problem->cones[k].type == PREFOS_CONE_POSITIVE_SEMIDEFINITE &&
              problem->cones[k].matrix_order == 1))
            continue;
        if (added_variables > SIZE_MAX - problem->cones[k].dimension)
            return PREFOS_STATUS_OUT_OF_MEMORY;
        added_variables += problem->cones[k].dimension;
        ++removed_cones;
    }
    if (removed_cones == 0) return PREFOS_STATUS_OK;
    if (problem->n_box > SIZE_MAX - added_variables) return PREFOS_STATUS_OUT_OF_MEMORY;

    box_indices = (int *) prefos_internal_alloc_array(problem->n_box + added_variables,
                                                   sizeof(int));
    box_lower = (double *) prefos_internal_alloc_array(problem->n_box + added_variables,
                                                    sizeof(double));
    box_upper = (double *) prefos_internal_alloc_array(problem->n_box + added_variables,
                                                    sizeof(double));
    if (!box_indices || !box_lower || !box_upper)
    {
        free(box_indices);
        free(box_lower);
        free(box_upper);
        return PREFOS_STATUS_OUT_OF_MEMORY;
    }
    if (problem->n_box > 0)
    {
        memcpy(box_indices, problem->box_indices, problem->n_box * sizeof(int));
        memcpy(box_lower, problem->box_lower, problem->n_box * sizeof(double));
        memcpy(box_upper, problem->box_upper, problem->n_box * sizeof(double));
    }

    box_write = problem->n_box;
    for (k = 0; k < problem->n_cones; ++k)
    {
        PreFOSConeBlock cone = problem->cones[k];
        if (cone.type == PREFOS_CONE_NONNEGATIVE ||
            (cone.type == PREFOS_CONE_POSITIVE_SEMIDEFINITE && cone.matrix_order == 1))
        {
            for (i = 0; i < cone.dimension; ++i)
            {
                box_indices[box_write] = cone.indices[i];
                box_lower[box_write] = 0.0;
                box_upper[box_write] = INFINITY;
                ++box_write;
            }
            free(cone.indices);
            problem->cones[k].indices = NULL;
            continue;
        }
        if (cone_write != k)
        {
            problem->cones[cone_write] = cone;
            problem->cones[k].indices = NULL;
        }
        ++cone_write;
    }

    free(problem->box_indices);
    free(problem->box_lower);
    free(problem->box_upper);
    problem->box_indices = box_indices;
    problem->box_lower = box_lower;
    problem->box_upper = box_upper;
    problem->n_box += added_variables;
    problem->n_cones = cone_write;
    presolver->normalized_nonnegative_variables = added_variables;
    presolver->normalized_nonnegative_cones = removed_cones;
    return PREFOS_STATUS_OK;
}

PreFOSSettings prefos_default_settings(void)
{
    PreFOSSettings settings;
    settings.feasibility_tolerance = 1e-7;
    settings.fixed_variable_tolerance = 1e-10;
    settings.fix_close_box_bounds = 1;
    settings.remove_empty_rows = 1;
    settings.remove_redundant_rows = 1;
    settings.free_column_substitution = 1;
    settings.max_aggregation_terms = 2;
    settings.max_aggregation_column_degree = 8;
    settings.max_aggregation_fill = 8;
    settings.max_aggregation_rounds = 4;
    settings.max_aggregation_scale = 1e3;
    settings.linear_propagation = 1;
    settings.max_linear_propagation_rounds = 8;
    settings.linear_propagation_gpu = 0;
    settings.linear_propagation_max_work_ratio = 2.0;
    settings.linear_propagation_min_changes_per_million = 100.0;
    settings.linear_propagation_max_stale_rounds = 1;
    settings.cone_propagation = 1;
    settings.max_cone_propagation_rounds = 4;
    settings.cone_aware_row_activity = 1;
    settings.exponential_propagation = 1;
    settings.power_propagation = 1;
    settings.psd_higher_order_propagation = 1;
    settings.affine_psd_propagation_max_work_ratio = 16.0;
    settings.rsoc_face_reduction = 1;
    settings.psd_face_reduction = 1;
    settings.exponential_face_reduction = 1;
    settings.power_face_reduction = 1;
    settings.finite_bound_improvement_absolute = 1e-3;
    settings.finite_bound_improvement_relative = 1e-2;
    settings.event_queue_max_average_column_degree = 64.0;
    settings.event_queue_activity_update_ratio = 4.0;
    settings.propagated_bound_policy = PREFOS_PROPAGATED_BOUND_POLICY_FIRST_ORDER;
    settings.affine_cone_coordinate_aggregation = 0;
    settings.psd_structure_analysis = 1;
    settings.psd_block_decomposition = 1;
    return settings;
}

PreFOSSettings prefos_strict_settings(void)
{
    PreFOSSettings settings = prefos_default_settings();
    settings.feasibility_tolerance = 0.0;
    settings.fixed_variable_tolerance = 0.0;
    settings.finite_bound_improvement_absolute = 0.0;
    settings.finite_bound_improvement_relative = 0.0;
    settings.linear_propagation_max_work_ratio = 0.0;
    settings.linear_propagation_min_changes_per_million = 0.0;
    settings.linear_propagation_max_stale_rounds = 0;
    settings.affine_psd_propagation_max_work_ratio = 0.0;
    return settings;
}

PreFOSStatus prefos_create_presolver(const PreFOSProblemData *problem,
                               const PreFOSSettings *settings, PreFOSPresolver **presolver)
{
    PreFOSPresolver *result;
    PreFOSSettings actual_settings;
    PreFOSStatus status;

    if (!presolver) return PREFOS_STATUS_INVALID_ARGUMENT;
    *presolver = NULL;
    actual_settings = settings ? *settings : prefos_default_settings();
    status = validate_problem(problem, &actual_settings);
    if (status != PREFOS_STATUS_OK) return status;

    result = (PreFOSPresolver *) calloc(1, sizeof(PreFOSPresolver));
    if (!result) return PREFOS_STATUS_OUT_OF_MEMORY;
    result->settings = actual_settings;
    status = copy_problem(problem, &result->original);
    if (status != PREFOS_STATUS_OK)
    {
        prefos_free_presolver(result);
        return status;
    }
    canonicalize_copied_problem(&result->original);
    status = normalize_nonnegative_cones(result);
    if (status != PREFOS_STATUS_OK)
    {
        prefos_free_presolver(result);
        return status;
    }
    if (result->settings.linear_propagation_gpu && result->original.n_box > 0 &&
        (long double) result->original.A.nnz / (long double) result->original.n_box >
            (long double) result->settings.event_queue_max_average_column_degree)
        (void) prefos_gpu_warmup_async();
    *presolver = result;
    return PREFOS_STATUS_OK;
}

void prefos_free_presolver(PreFOSPresolver *presolver)
{
    if (!presolver) return;
    prefos_internal_free_psd_face_reductions(presolver->psd_face_reductions,
                                          presolver->original.n_cones);
    free_problem_data(&presolver->original);
    prefos_internal_free_reduced_problem(&presolver->reduced);
    free(presolver->original_to_reduced);
    free(presolver->original_to_reduced_rows);
    free(presolver->fixed_values);
    free(presolver->is_fixed);
    free(presolver->is_substituted);
    free(presolver->substitution_term_count);
    free(presolver->substitution_incoming_depth);
    free(presolver->substitution_term_start);
    free(presolver->substitution_constant);
    free(presolver->substitution_targets);
    free(presolver->substitution_scales);
    free(presolver->variable_to_box);
    free(presolver->working_box_lower);
    free(presolver->working_box_upper);
    free(presolver->working_constraint_lower);
    free(presolver->working_constraint_upper);
    free(presolver->propagation_lower);
    free(presolver->propagation_upper);
    free(presolver->converted_affine_cones);
    free(presolver->affine_protected_columns);
    free(presolver->affine_aggregation_source_rows);
    free(presolver->affine_aggregation_pivots);
    free(presolver->affine_bound_certificates);
    free(presolver->affine_pre_to_reduced_rows);
    free(presolver->psd_structure_analyses);
    free(presolver->input_affine_rsoc_zero_axis);
    free(presolver->generated_affine_rsoc_zero_axis);
    free(presolver->affine_face_substitution_targets);
    free(presolver->affine_face_eliminated_columns);
    free(presolver->remove_rows);
    free(presolver->remove_cones);
    free(presolver->cone_face_survivors);
    free(presolver->cone_face_box);
    free(presolver->cone_face_box_lower);
    free(presolver->cone_face_box_upper);
    free(presolver->cone_collapse_source_rows);
    free(presolver->facial_reductions);
    presolve_transformation_log_free(&presolver->transformations);
    free(presolver);
}
