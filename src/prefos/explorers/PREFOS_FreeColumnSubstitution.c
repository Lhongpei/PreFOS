/*
 * Copyright 2026 Hongpei Li
 * SPDX-License-Identifier: Apache-2.0
 */

#include "PREFOS_FreeColumnSubstitution.h"

#define PREFOS_MAX_AGGREGATION_ROW_NNZ (PREFOS_MAX_AGGREGATION_TERMS + 1)
#ifndef PREFOS_MAX_AGGREGATION_COLUMN_DEGREE
#define PREFOS_MAX_AGGREGATION_COLUMN_DEGREE 8
#endif

typedef struct
{
    PreFOSCsrMatrix matrix;
    double *lower;
    double *upper;
} PreFOSWorkingAggregationMatrix;

static int row_contains_column(const PreFOSCsrMatrix *matrix, int row, int column)
{
    int position;
    for (position = matrix->row_pointers[row];
         position < matrix->row_pointers[row + 1]; ++position)
    {
        if (matrix->values[position] != 0.0 &&
            matrix->column_indices[position] == column)
            return 1;
    }
    return 0;
}

static PreFOSStatus
append_substitution_record(PreFOSPresolver *presolver, int column, const int *targets,
                           const double *scales, size_t term_count, int source_row,
                           double constant, double pivot,
                           double objective_coefficient, const int *column_rows,
                           const double *column_coefficients, size_t degree)
{
    PresolveColumnTransformationRecord record = {
        .type = PRESOLVE_COLUMN_SUBSTITUTED,
        .column = column,
        .secondary_column = targets[0],
        .source_row = source_row,
        .value = constant,
        .objective_coefficient = objective_coefficient,
        .rhs = pivot,
        .ratio = scales[0],
        .indices = (int *) column_rows,
        .coefficients = (double *) column_coefficients,
        .length = degree};
    if (term_count == 0 || term_count > PREFOS_MAX_AGGREGATION_TERMS)
        return PREFOS_STATUS_NUMERICAL_ERROR;
    return presolve_transformation_log_append_column_transformation(
               &presolver->transformations, &record, NULL)
               ? PREFOS_STATUS_OK
               : PREFOS_STATUS_OUT_OF_MEMORY;
}

static PreFOSStatus reserve_substitution_terms(PreFOSPresolver *presolver,
                                            size_t additional)
{
    size_t required, capacity;
    int *targets;
    double *scales;

    if (additional > SIZE_MAX - presolver->n_substitution_terms)
        return PREFOS_STATUS_OUT_OF_MEMORY;
    required = presolver->n_substitution_terms + additional;
    if (required <= presolver->substitution_term_capacity) return PREFOS_STATUS_OK;
    capacity = presolver->substitution_term_capacity == 0
                   ? 1024
                   : presolver->substitution_term_capacity;
    while (capacity < required)
    {
        if (capacity > SIZE_MAX / 2)
        {
            capacity = required;
            break;
        }
        capacity *= 2;
    }
    targets = (int *) prefos_internal_alloc_array(capacity, sizeof(int));
    scales = (double *) prefos_internal_alloc_array(capacity, sizeof(double));
    if (!targets || !scales)
    {
        free(targets);
        free(scales);
        return PREFOS_STATUS_OUT_OF_MEMORY;
    }
    if (presolver->n_substitution_terms > 0)
    {
        memcpy(targets, presolver->substitution_targets,
               presolver->n_substitution_terms * sizeof(int));
        memcpy(scales, presolver->substitution_scales,
               presolver->n_substitution_terms * sizeof(double));
    }
    free(presolver->substitution_targets);
    free(presolver->substitution_scales);
    presolver->substitution_targets = targets;
    presolver->substitution_scales = scales;
    presolver->substitution_term_capacity = capacity;
    return PREFOS_STATUS_OK;
}

static void mark_substitution(PreFOSPresolver *presolver, int column,
                              const int *targets, const double *scales,
                              size_t term_count, double constant, int source_row,
                              unsigned char *protected_target)
{
    size_t start = presolver->n_substitution_terms;
    size_t term;
    unsigned char next_depth =
        (unsigned char) (presolver->substitution_incoming_depth[column] + 1);

    presolver->is_substituted[column] = 1;
    presolver->substitution_term_count[column] = (unsigned char) term_count;
    presolver->substitution_term_start[column] = start;
    presolver->substitution_constant[column] = constant;
    for (term = 0; term < term_count; ++term)
    {
        int target = targets[term];
        presolver->substitution_targets[start + term] = target;
        presolver->substitution_scales[start + term] = scales[term];
        protected_target[target] = 1;
        if (presolver->substitution_incoming_depth[target] < next_depth)
            presolver->substitution_incoming_depth[target] = next_depth;
    }
    presolver->n_substitution_terms += term_count;
    presolver->variable_to_box[column] = -1;
    presolver->remove_rows[source_row] = 1;
    ++presolver->stats.substituted_free_variables;
    if (term_count == 2) ++presolver->stats.ternary_substituted_free_variables;
}

static int column_can_be_eliminated(const PreFOSPresolver *presolver, int column,
                                    const unsigned char *quadratic_column,
                                    const unsigned char *factor_column,
                                    const unsigned char *protected_target)
{
    int box_position = presolver->variable_to_box[column];
    return box_position >= 0 && !presolver->is_fixed[column] &&
           !presolver->is_substituted[column] && !protected_target[column] &&
           !presolver->affine_protected_columns[column] &&
           presolver->substitution_incoming_depth[column] <
               PREFOS_MAX_SUBSTITUTION_DEPTH &&
           presolver->working_box_lower[box_position] == -INFINITY &&
           presolver->working_box_upper[box_position] == INFINITY &&
           !quadratic_column[column] && !factor_column[column];
}

static size_t active_column_degree(const PreFOSPresolver *presolver,
                                   const int *column_starts, const int *column_rows,
                                   int column)
{
    size_t degree = 0;
    int position;
    for (position = column_starts[column]; position < column_starts[column + 1];
         ++position)
        if (!presolver->remove_rows[column_rows[position]]) ++degree;
    return degree;
}

static size_t aggregation_fill(const PreFOSPresolver *presolver,
                               const PreFOSCsrMatrix *matrix, const int *column_starts,
                               const int *column_rows, int eliminated,
                               int source_row, const int *targets, size_t term_count)
{
    size_t fill = 0;
    int position;

    for (position = column_starts[eliminated];
         position < column_starts[eliminated + 1]; ++position)
    {
        int adjacent_row = column_rows[position];
        size_t absent = 0, term;
        if (adjacent_row == source_row || presolver->remove_rows[adjacent_row])
            continue;
        for (term = 0; term < term_count; ++term)
            if (!row_contains_column(matrix, adjacent_row, targets[term])) ++absent;
        if (absent > 1)
        {
            fill += absent - 1;
            if (fill > (size_t) presolver->settings.max_aggregation_fill)
                return fill;
        }
    }
    return fill;
}

static PreFOSStatus accumulate_expanded_column(const PreFOSPresolver *presolver,
                                            int column, double coefficient,
                                            size_t depth, double *row_values,
                                            int *row_marks, int *touched_columns,
                                            size_t *n_touched, double *shift)
{
    size_t term;
    if (column < 0 || (size_t) column >= presolver->original.n ||
        depth > PREFOS_MAX_SUBSTITUTION_DEPTH)
        return PREFOS_STATUS_NUMERICAL_ERROR;
    if (presolver->is_fixed[column])
        return prefos_internal_safe_add_product(
                   shift, coefficient, presolver->fixed_values[column])
                   ? PREFOS_STATUS_OK
                   : PREFOS_STATUS_NUMERICAL_ERROR;
    if (presolver->is_parallel_removed[column]) return PREFOS_STATUS_OK;
    if (presolver->is_substituted[column])
    {
        size_t start = presolver->substitution_term_start[column];
        unsigned char count = presolver->substitution_term_count[column];
        if (count == 0 || count > PREFOS_MAX_AGGREGATION_TERMS ||
            start > presolver->n_substitution_terms ||
            count > presolver->n_substitution_terms - start ||
            !prefos_internal_safe_add_product(shift, coefficient,
                                           presolver->substitution_constant[column]))
            return PREFOS_STATUS_NUMERICAL_ERROR;
        for (term = 0; term < count; ++term)
        {
            double propagated;
            PreFOSStatus status;
            if (!prefos_internal_safe_product(
                    coefficient, presolver->substitution_scales[start + term],
                    &propagated))
                return PREFOS_STATUS_NUMERICAL_ERROR;
            status = accumulate_expanded_column(
                presolver, presolver->substitution_targets[start + term], propagated,
                depth + 1, row_values, row_marks, touched_columns, n_touched, shift);
            if (status != PREFOS_STATUS_OK) return status;
        }
        return PREFOS_STATUS_OK;
    }
    if (row_marks[column] < 0)
    {
        row_marks[column] = 1;
        row_values[column] = coefficient;
        touched_columns[(*n_touched)++] = column;
    }
    else if (!prefos_internal_safe_add_product(&row_values[column], 1.0, coefficient))
        return PREFOS_STATUS_NUMERICAL_ERROR;
    return PREFOS_STATUS_OK;
}

static void clear_expanded_row(double *row_values, int *row_marks,
                               const int *touched_columns, size_t n_touched)
{
    size_t position;
    for (position = 0; position < n_touched; ++position)
    {
        int column = touched_columns[position];
        row_values[column] = 0.0;
        row_marks[column] = -1;
    }
}

static void free_working_matrix(PreFOSWorkingAggregationMatrix *working)
{
    prefos_internal_free_csr(&working->matrix);
    free(working->lower);
    free(working->upper);
    memset(working, 0, sizeof(*working));
}

static PreFOSStatus materialize_aggregation_matrix(const PreFOSPresolver *presolver,
                                                const PreFOSCsrMatrix *source,
                                                const double *source_lower,
                                                const double *source_upper,
                                                PreFOSWorkingAggregationMatrix *target)
{
    double *row_values = NULL;
    int *row_marks = NULL;
    int *touched_columns = NULL;
    size_t row, nnz = 0;
    int write = 0;
    PreFOSStatus status = PREFOS_STATUS_OK;

    memset(target, 0, sizeof(*target));
    row_values = (double *) calloc(source->cols, sizeof(double));
    row_marks = (int *) prefos_internal_alloc_array(source->cols, sizeof(int));
    touched_columns = (int *) prefos_internal_alloc_array(source->cols, sizeof(int));
    target->matrix.row_pointers = (int *) calloc(source->rows + 1, sizeof(int));
    target->lower =
        (double *) prefos_internal_alloc_array(source->rows, sizeof(double));
    target->upper =
        (double *) prefos_internal_alloc_array(source->rows, sizeof(double));
    if ((source->cols > 0 && (!row_values || !row_marks || !touched_columns)) ||
        !target->matrix.row_pointers ||
        (source->rows > 0 && (!target->lower || !target->upper)))
    {
        status = PREFOS_STATUS_OUT_OF_MEMORY;
        goto cleanup;
    }
    for (row = 0; row < source->cols; ++row) row_marks[row] = -1;

    for (row = 0; row < source->rows; ++row)
    {
        size_t n_touched = 0, position;
        double shift = 0.0;
        int p;
        target->matrix.row_pointers[row] = (int) nnz;
        target->lower[row] = source_lower[row];
        target->upper[row] = source_upper[row];
        if (presolver->remove_rows[row]) continue;
        for (p = source->row_pointers[row]; p < source->row_pointers[row + 1]; ++p)
        {
            if (source->values[p] == 0.0) continue;
            status = accumulate_expanded_column(
                presolver, source->column_indices[p], source->values[p], 0,
                row_values, row_marks, touched_columns, &n_touched, &shift);
            if (status != PREFOS_STATUS_OK) goto cleanup;
        }
        for (position = 0; position < n_touched; ++position)
            if (row_values[touched_columns[position]] != 0.0) ++nnz;
        clear_expanded_row(row_values, row_marks, touched_columns, n_touched);
        target->lower[row] = source_lower[row] - shift;
        target->upper[row] = source_upper[row] - shift;
        if (isnan(target->lower[row]) || isnan(target->upper[row]) ||
            nnz > (size_t) INT_MAX)
        {
            status = PREFOS_STATUS_NUMERICAL_ERROR;
            goto cleanup;
        }
    }
    target->matrix.row_pointers[source->rows] = (int) nnz;
    target->matrix.values = (double *) prefos_internal_alloc_array(nnz, sizeof(double));
    target->matrix.column_indices =
        (int *) prefos_internal_alloc_array(nnz, sizeof(int));
    if (nnz > 0 && (!target->matrix.values || !target->matrix.column_indices))
    {
        status = PREFOS_STATUS_OUT_OF_MEMORY;
        goto cleanup;
    }

    for (row = 0; row < source->rows; ++row)
    {
        size_t n_touched = 0, position;
        double shift = 0.0;
        int p;
        target->matrix.row_pointers[row] = write;
        if (presolver->remove_rows[row]) continue;
        for (p = source->row_pointers[row]; p < source->row_pointers[row + 1]; ++p)
        {
            if (source->values[p] == 0.0) continue;
            status = accumulate_expanded_column(
                presolver, source->column_indices[p], source->values[p], 0,
                row_values, row_marks, touched_columns, &n_touched, &shift);
            if (status != PREFOS_STATUS_OK) goto cleanup;
        }
        for (position = 0; position < n_touched; ++position)
        {
            int column = touched_columns[position];
            if (row_values[column] == 0.0) continue;
            target->matrix.values[write] = row_values[column];
            target->matrix.column_indices[write++] = column;
        }
        clear_expanded_row(row_values, row_marks, touched_columns, n_touched);
    }
    target->matrix.row_pointers[source->rows] = write;
    target->matrix.rows = source->rows;
    target->matrix.cols = source->cols;
    target->matrix.nnz = (size_t) write;

cleanup:
    free(row_values);
    free(row_marks);
    free(touched_columns);
    if (status != PREFOS_STATUS_OK) free_working_matrix(target);
    return status;
}

static PreFOSStatus build_column_storage(const PreFOSCsrMatrix *matrix,
                                      int **column_starts_out, int **column_rows_out,
                                      double **column_coefficients_out)
{
    int *column_starts = NULL, *column_rows = NULL, *cursor = NULL;
    double *column_coefficients = NULL;
    size_t row, position;

    column_starts = (int *) calloc(matrix->cols + 1, sizeof(int));
    cursor = (int *) prefos_internal_alloc_array(matrix->cols, sizeof(int));
    column_rows = (int *) prefos_internal_alloc_array(matrix->nnz, sizeof(int));
    column_coefficients =
        (double *) prefos_internal_alloc_array(matrix->nnz, sizeof(double));
    if (!column_starts || (matrix->cols > 0 && !cursor) ||
        (matrix->nnz > 0 && (!column_rows || !column_coefficients)))
    {
        free(column_starts);
        free(cursor);
        free(column_rows);
        free(column_coefficients);
        return PREFOS_STATUS_OUT_OF_MEMORY;
    }
    for (position = 0; position < matrix->nnz; ++position)
        if (matrix->values[position] != 0.0)
            ++column_starts[matrix->column_indices[position] + 1];
    for (position = 0; position < matrix->cols; ++position)
        column_starts[position + 1] += column_starts[position];
    if (matrix->cols > 0) memcpy(cursor, column_starts, matrix->cols * sizeof(int));
    for (row = 0; row < matrix->rows; ++row)
    {
        int p;
        for (p = matrix->row_pointers[row]; p < matrix->row_pointers[row + 1]; ++p)
        {
            int column, write;
            if (matrix->values[p] == 0.0) continue;
            column = matrix->column_indices[p];
            write = cursor[column]++;
            column_rows[write] = (int) row;
            column_coefficients[write] = matrix->values[p];
        }
    }
    free(cursor);
    *column_starts_out = column_starts;
    *column_rows_out = column_rows;
    *column_coefficients_out = column_coefficients;
    return PREFOS_STATUS_OK;
}

PreFOSStatus prefos_internal_substitute_free_columns(PreFOSPresolver *presolver)
{
    const PreFOSProblemData *problem = &presolver->original;
    const PreFOSCsrMatrix *current_matrix = &problem->A;
    const double *current_lower = presolver->working_constraint_lower;
    const double *current_upper = presolver->working_constraint_upper;
    PreFOSWorkingAggregationMatrix owned_current;
    unsigned char *quadratic_column = NULL;
    unsigned char *factor_column = NULL;
    unsigned char *protected_target = NULL;
    double *working_objective = NULL;
    double working_objective_offset = 0.0;
    size_t row, round;
    PreFOSStatus status = PREFOS_STATUS_OK;

    memset(&owned_current, 0, sizeof(owned_current));
    if (!presolver->settings.free_column_substitution || problem->n_box == 0 ||
        problem->A.rows == 0)
        return PREFOS_STATUS_OK;

    quadratic_column = (unsigned char *) calloc(problem->n, sizeof(unsigned char));
    factor_column = (unsigned char *) calloc(problem->n, sizeof(unsigned char));
    protected_target = (unsigned char *) calloc(problem->n, sizeof(unsigned char));
    working_objective =
        (double *) prefos_internal_alloc_array(problem->n, sizeof(double));
    if (problem->n > 0 && (!quadratic_column || !factor_column ||
                           !protected_target || !working_objective))
    {
        status = PREFOS_STATUS_OUT_OF_MEMORY;
        goto cleanup;
    }
    if (problem->n > 0)
    {
        status = prefos_internal_expand_linear_objective(
            presolver, working_objective, &working_objective_offset);
        if (status != PREFOS_STATUS_OK) goto cleanup;
    }

    for (row = 0; row < problem->Q.rows; ++row)
    {
        int p;
        for (p = problem->Q.row_pointers[row]; p < problem->Q.row_pointers[row + 1];
             ++p)
        {
            if (problem->Q.values[p] == 0.0) continue;
            quadratic_column[row] = 1;
            quadratic_column[problem->Q.column_indices[p]] = 1;
        }
    }
    for (row = 0; row < problem->R.rows; ++row)
    {
        int p;
        for (p = problem->R.row_pointers[row]; p < problem->R.row_pointers[row + 1];
             ++p)
            if (problem->R.values[p] != 0.0)
                factor_column[problem->R.column_indices[p]] = 1;
    }

    if (presolver->stats.substituted_free_variables > 0 ||
        presolver->stats.removed_empty_columns > 0 ||
        presolver->stats.dual_fixed_columns > 0 ||
        presolver->stats.merged_parallel_columns > 0)
    {
        status = materialize_aggregation_matrix(
            presolver, current_matrix, current_lower, current_upper,
            &owned_current);
        if (status != PREFOS_STATUS_OK) goto cleanup;
        current_matrix = &owned_current.matrix;
        current_lower = owned_current.lower;
        current_upper = owned_current.upper;
    }

    for (round = 0;
         round < (size_t) presolver->settings.max_aggregation_rounds; ++round)
    {
        int *column_starts = NULL, *column_rows = NULL;
        double *column_coefficients = NULL;
        size_t accepted = 0;

        memset(protected_target, 0, problem->n * sizeof(unsigned char));
        status = build_column_storage(current_matrix, &column_starts, &column_rows,
                                      &column_coefficients);
        if (status != PREFOS_STATUS_OK) goto cleanup;

        for (row = 0; row < current_matrix->rows; ++row)
        {
            int columns[PREFOS_MAX_AGGREGATION_ROW_NNZ];
            double coefficients[PREFOS_MAX_AGGREGATION_ROW_NNZ];
            int best_eliminated = -1;
            int best_targets[PREFOS_MAX_AGGREGATION_TERMS];
            double best_scales[PREFOS_MAX_AGGREGATION_TERMS];
            double best_constant = 0.0, best_pivot = 0.0;
            long double best_max_scale = LDBL_MAX;
            size_t nonzeros = 0, best_degree = SIZE_MAX;
            size_t best_fill = SIZE_MAX, best_term_count = 0;
            int p, candidate_position;

            if (presolver->remove_rows[row] || !isfinite(current_lower[row]) ||
                current_lower[row] != current_upper[row])
                continue;
            for (p = current_matrix->row_pointers[row];
                 p < current_matrix->row_pointers[row + 1]; ++p)
            {
                if (current_matrix->values[p] == 0.0) continue;
                if (nonzeros < PREFOS_MAX_AGGREGATION_ROW_NNZ)
                {
                    columns[nonzeros] = current_matrix->column_indices[p];
                    coefficients[nonzeros] = current_matrix->values[p];
                }
                ++nonzeros;
            }
            if (nonzeros < 2 ||
                nonzeros >
                    (size_t) presolver->settings.max_aggregation_terms + 1)
                continue;

            for (candidate_position = 0; candidate_position < (int) nonzeros;
                 ++candidate_position)
            {
                int eliminated = columns[candidate_position];
                int targets[PREFOS_MAX_AGGREGATION_TERMS];
                double scales[PREFOS_MAX_AGGREGATION_TERMS];
                double constant;
                long double value, max_scale = 0.0L;
                size_t active_degree, fill, term_count = 0;
                int target_position, valid = 1;

                if (!column_can_be_eliminated(presolver, eliminated,
                                              quadratic_column, factor_column,
                                              protected_target))
                    continue;
                active_degree = active_column_degree(presolver, column_starts,
                                                     column_rows, eliminated);
                if (active_degree == 0 ||
                    active_degree >
                        (size_t) presolver->settings.max_aggregation_column_degree)
                    continue;
                for (target_position = 0; target_position < (int) nonzeros;
                     ++target_position)
                {
                    int target;
                    if (target_position == candidate_position) continue;
                    target = columns[target_position];
                    if (presolver->is_fixed[target] ||
                        presolver->is_substituted[target])
                    {
                        valid = 0;
                        break;
                    }
                    targets[term_count] = target;
                    value = -(long double) coefficients[target_position] /
                            (long double) coefficients[candidate_position];
                    if (!isfinite(value) ||
                        fabsl(value) >
                            (long double) presolver->settings.max_aggregation_scale)
                    {
                        valid = 0;
                        break;
                    }
                    max_scale = fmaxl(max_scale, fabsl(value));
                    scales[term_count++] = (double) value;
                }
                if (!valid) continue;
                value = (long double) current_lower[row] /
                        (long double) coefficients[candidate_position];
                if (!isfinite(value) || fabsl(value) > (long double) DBL_MAX)
                    continue;
                constant = (double) value;
                fill = aggregation_fill(presolver, current_matrix, column_starts,
                                        column_rows, eliminated, (int) row, targets,
                                        term_count);
                if (fill > (size_t) presolver->settings.max_aggregation_fill)
                    continue;
                if (fill < best_fill ||
                    (fill == best_fill && max_scale < best_max_scale) ||
                    (fill == best_fill && max_scale == best_max_scale &&
                     active_degree < best_degree) ||
                    (fill == best_fill && max_scale == best_max_scale &&
                     active_degree == best_degree &&
                     eliminated < best_eliminated))
                {
                    size_t term;
                    best_eliminated = eliminated;
                    best_constant = constant;
                    best_pivot = coefficients[candidate_position];
                    best_fill = fill;
                    best_max_scale = max_scale;
                    best_degree = active_degree;
                    best_term_count = term_count;
                    for (term = 0; term < term_count; ++term)
                    {
                        best_targets[term] = targets[term];
                        best_scales[term] = scales[term];
                    }
                }
            }
            if (best_eliminated >= 0)
            {
                int record_rows[PREFOS_MAX_AGGREGATION_COLUMN_DEGREE];
                double record_coefficients[PREFOS_MAX_AGGREGATION_COLUMN_DEGREE];
                double updated_objective[PREFOS_MAX_AGGREGATION_TERMS];
                size_t record_degree = 0, term;
                int adjacent;

                for (adjacent = column_starts[best_eliminated];
                     adjacent < column_starts[best_eliminated + 1]; ++adjacent)
                {
                    int adjacent_row = column_rows[adjacent];
                    if (presolver->remove_rows[adjacent_row]) continue;
                    if (record_degree >= PREFOS_MAX_AGGREGATION_COLUMN_DEGREE)
                    {
                        status = PREFOS_STATUS_NUMERICAL_ERROR;
                        break;
                    }
                    record_rows[record_degree] = adjacent_row;
                    record_coefficients[record_degree++] =
                        column_coefficients[adjacent];
                }
                if (status != PREFOS_STATUS_OK) break;
                for (term = 0; term < best_term_count; ++term)
                {
                    updated_objective[term] = working_objective[best_targets[term]];
                    if (!prefos_internal_safe_add_product(
                            &updated_objective[term],
                            working_objective[best_eliminated], best_scales[term]))
                    {
                        status = PREFOS_STATUS_NUMERICAL_ERROR;
                        break;
                    }
                }
                if (status != PREFOS_STATUS_OK) break;
                status = reserve_substitution_terms(presolver, best_term_count);
                if (status == PREFOS_STATUS_OK)
                    status = append_substitution_record(
                        presolver, best_eliminated, best_targets, best_scales,
                        best_term_count, (int) row, best_constant, best_pivot,
                        working_objective[best_eliminated], record_rows,
                        record_coefficients, record_degree);
                if (status != PREFOS_STATUS_OK) break;
                mark_substitution(presolver, best_eliminated, best_targets,
                                  best_scales, best_term_count, best_constant,
                                  (int) row, protected_target);
                working_objective[best_eliminated] = 0.0;
                for (term = 0; term < best_term_count; ++term)
                    working_objective[best_targets[term]] = updated_objective[term];
                ++accepted;
            }
        }
        free(column_starts);
        free(column_rows);
        free(column_coefficients);
        if (status != PREFOS_STATUS_OK) goto cleanup;
        if (accepted == 0 ||
            round + 1 ==
                (size_t) presolver->settings.max_aggregation_rounds)
            break;
        {
            PreFOSWorkingAggregationMatrix next;
            status = materialize_aggregation_matrix(
                presolver, current_matrix, current_lower, current_upper, &next);
            if (status != PREFOS_STATUS_OK) goto cleanup;
            free_working_matrix(&owned_current);
            owned_current = next;
            current_matrix = &owned_current.matrix;
            current_lower = owned_current.lower;
            current_upper = owned_current.upper;
        }
    }

cleanup:
    free_working_matrix(&owned_current);
    free(quadratic_column);
    free(factor_column);
    free(protected_target);
    free(working_objective);
    return status;
}
