/*
 * Copyright 2026 Hongpei Li
 * SPDX-License-Identifier: Apache-2.0
 */

#include "PREFOS_FreeColumnSubstitution.h"
#include "core/PREFOS_WorkingMatrix.h"

#include <stdio.h>

#define PREFOS_MAX_AGGREGATION_ROW_NNZ (PREFOS_MAX_AGGREGATION_TERMS + 1)
#define PREFOS_FREE_AGGREGATION_PREFILTER_NNZ 100000U
#define PREFOS_FREE_AGGREGATION_NNZ_PER_CANDIDATE 512U

static int column_contains_row(const int *column_starts,
                               const int *column_rows,
                               int column, int row)
{
    int lower = column_starts[column];
    int upper = column_starts[column + 1];
    while (lower < upper)
    {
        int middle = lower + (upper - lower) / 2;
        int candidate = column_rows[middle];
        if (candidate < row)
            lower = middle + 1;
        else
            upper = middle;
    }
    return lower < column_starts[column + 1] &&
           column_rows[lower] == row;
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
                              unsigned char *protected_target,
                              const int *active_rows, size_t active_degree)
{
    size_t start = presolver->n_substitution_terms;
    size_t term;
    int creates_fill_in = 0;
    uint16_t next_depth =
        (uint16_t) (presolver->substitution_incoming_depth[column] + 1);

    for (term = 0; term < active_degree; ++term)
        if (active_rows[term] != source_row)
        {
            creates_fill_in = 1;
            break;
        }
    presolver->is_substituted[column] = 1;
    presolver->substitution_term_count[column] = term_count;
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
        if (creates_fill_in)
            presolver->substitution_fill_in_targets[target] = 1;
    }
    presolver->n_substitution_terms += term_count;
    for (term = 0; term < active_degree; ++term)
        if (active_rows[term] != source_row)
            prefos_internal_mark_row_requires_materialization(
                presolver, (size_t) active_rows[term]);
    presolver->variable_to_box[column] = -1;
    prefos_internal_mark_removed_row(presolver, (size_t) source_row);
    ++presolver->stats.substituted_free_variables;
    if (term_count == 2) ++presolver->stats.ternary_substituted_free_variables;
}

static int column_has_free_input_domain(
    const PreFOSPresolver *presolver, int box_position)
{
    return box_position >= 0 &&
           presolver->original.box_lower[box_position] == -INFINITY &&
           presolver->original.box_upper[box_position] == INFINITY;
}

static int working_bound_can_be_dropped(
    const PreFOSPresolver *presolver, int column, int box_position,
    int source_row, int is_lower)
{
    const PresolveTransformationLog *log = &presolver->transformations;
    const size_t *latest =
        is_lower ? presolver->latest_lower_bound_change
                 : presolver->latest_upper_bound_change;
    double bound =
        is_lower ? presolver->working_box_lower[box_position]
                 : presolver->working_box_upper[box_position];
    size_t record_index;
    const PresolveBoundChangeRecord *record;

    /*
     * A bound certified by a live row remains implied after substitution.
     * A bound whose source row was already removed is the surviving form of
     * that row and must not disappear with the column.
     */
    if (!isfinite(bound)) return 1;
    if (source_row < 0 || !latest) return 0;
    record_index = latest[column];
    if (record_index == SIZE_MAX || record_index >= log->n_bound_changes)
        return 0;
    record = &log->bound_changes[record_index];
    if (record->column != column ||
        (int) record->is_lower != !!is_lower ||
        record->implied_bound != bound ||
        record->row < 0 ||
        (size_t) record->row >= presolver->original.A.rows)
        return 0;
    return record->row == source_row ||
           !presolver->remove_rows[record->row];
}

static int column_has_droppable_working_domain(
    const PreFOSPresolver *presolver, int column, int box_position,
    int source_row)
{
    return column_has_free_input_domain(presolver, box_position) &&
           (source_row < 0 ||
            (working_bound_can_be_dropped(
                 presolver, column, box_position, source_row, 1) &&
             working_bound_can_be_dropped(
                 presolver, column, box_position, source_row, 0)));
}

static int column_can_be_eliminated(const PreFOSPresolver *presolver, int column,
                                    const unsigned char *quadratic_column,
                                    const unsigned char *factor_column,
                                    const unsigned char *protected_target,
                                    int source_row)
{
    int box_position = presolver->variable_to_box[column];
    return box_position >= 0 && !presolver->is_fixed[column] &&
           !presolver->is_substituted[column] && !protected_target[column] &&
           !presolver->affine_protected_columns[column] &&
           presolver->substitution_incoming_depth[column] <
               PREFOS_MAX_SUBSTITUTION_DEPTH &&
           column_has_droppable_working_domain(
               presolver, column, box_position, source_row) &&
           !quadratic_column[column] && !factor_column[column];
}

static int has_basic_free_column_candidate(const PreFOSPresolver *presolver)
{
    size_t column;

    for (column = 0; column < presolver->original.n; ++column)
    {
        int box_position = presolver->variable_to_box[column];
        if (box_position >= 0 && !presolver->is_fixed[column] &&
            !presolver->is_substituted[column] &&
            !presolver->affine_protected_columns[column] &&
            presolver->substitution_incoming_depth[column] <
                PREFOS_MAX_SUBSTITUTION_DEPTH &&
            column_has_free_input_domain(
                presolver, box_position))
            return 1;
    }
    return 0;
}

static size_t aggregation_fill(const PreFOSPresolver *presolver,
                               const int *column_starts,
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
            if (!column_contains_row(
                    column_starts, column_rows,
                    targets[term], adjacent_row))
                ++absent;
        if (absent > 1)
        {
            fill += absent - 1;
            if (fill > (size_t) presolver->settings.max_aggregation_fill)
                return fill;
        }
    }
    return fill;
}

static PreFOSStatus build_column_storage(const PreFOSCsrMatrix *matrix,
                                      const unsigned char *remove_rows,
                                      int **column_starts_out, int **column_rows_out,
                                      double **column_coefficients_out,
                                      int **active_degrees_out)
{
    int *column_starts = NULL, *column_rows = NULL, *cursor = NULL;
    int *active_degrees = NULL;
    double *column_coefficients = NULL;
    size_t row, position;

    column_starts = (int *) calloc(matrix->cols + 1, sizeof(int));
    cursor = (int *) prefos_internal_alloc_array(matrix->cols, sizeof(int));
    column_rows = (int *) prefos_internal_alloc_array(matrix->nnz, sizeof(int));
    column_coefficients =
        (double *) prefos_internal_alloc_array(matrix->nnz, sizeof(double));
    active_degrees =
        (int *) calloc(matrix->cols, sizeof(int));
    if (!column_starts || (matrix->cols > 0 && !cursor) ||
        (matrix->nnz > 0 && (!column_rows || !column_coefficients)) ||
        (matrix->cols > 0 && !active_degrees))
    {
        free(column_starts);
        free(cursor);
        free(column_rows);
        free(column_coefficients);
        free(active_degrees);
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
            if (!remove_rows[row]) ++active_degrees[column];
        }
    }
    free(cursor);
    *column_starts_out = column_starts;
    *column_rows_out = column_rows;
    *column_coefficients_out = column_coefficients;
    *active_degrees_out = active_degrees;
    return PREFOS_STATUS_OK;
}

static int trace_free_aggregation(void)
{
    static int initialized = 0;
    static int enabled = 0;
    if (!initialized)
    {
        enabled = getenv("PREFOS_TRACE_FREE_AGGREGATION") != NULL;
        initialized = 1;
    }
    return enabled;
}

static int has_enough_free_aggregation_candidates(
    const PreFOSPresolver *presolver,
    const unsigned char *quadratic_column,
    const unsigned char *factor_column,
    const unsigned char *protected_target)
{
    const PreFOSProblemData *problem = &presolver->original;
    size_t required, candidates = 0, row;
    if (problem->A.nnz < PREFOS_FREE_AGGREGATION_PREFILTER_NNZ)
        return 1;
    required =
        (problem->A.nnz +
         PREFOS_FREE_AGGREGATION_NNZ_PER_CANDIDATE - 1) /
        PREFOS_FREE_AGGREGATION_NNZ_PER_CANDIDATE;
    if (required < 256) required = 256;
    for (row = 0; row < problem->A.rows && candidates < required; ++row)
    {
        size_t live = 0;
        int has_candidate = 0;
        int p;
        if (presolver->remove_rows[row] ||
            !isfinite(presolver->working_constraint_lower[row]) ||
            presolver->working_constraint_lower[row] !=
                presolver->working_constraint_upper[row])
            continue;
        for (p = problem->A.row_pointers[row];
             p < problem->A.row_pointers[row + 1]; ++p)
        {
            int column = problem->A.column_indices[p];
            if (problem->A.values[p] == 0.0 ||
                !prefos_internal_term_is_active_in_row(
                    presolver, row, column) ||
                presolver->is_fixed[column] ||
                presolver->is_parallel_removed[column])
                continue;
            if (presolver->is_substituted[column])
            {
                live = (size_t)
                    presolver->settings.max_aggregation_terms + 2;
                break;
            }
            ++live;
            if (live >
                (size_t) presolver->settings.max_aggregation_terms + 1)
                break;
            if (column_can_be_eliminated(
                    presolver, column, quadratic_column,
                    factor_column, protected_target, -1))
                has_candidate = 1;
        }
        if (live >= 2 &&
            live <=
                (size_t) presolver->settings.max_aggregation_terms + 1 &&
            has_candidate)
            ++candidates;
    }
    if (trace_free_aggregation())
        fprintf(
            stderr,
            "PreFOS free aggregation prefilter candidates=%zu "
            "required=%zu nnz=%zu decision=%s\n",
            candidates, required, problem->A.nnz,
            candidates >= required ? "run" : "skip");
    return candidates >= required;
}

PreFOSStatus prefos_internal_substitute_free_columns(PreFOSPresolver *presolver)
{
    const PreFOSProblemData *problem = &presolver->original;
    const PreFOSCsrMatrix *current_matrix = &problem->A;
    const double *current_lower = presolver->working_constraint_lower;
    const double *current_upper = presolver->working_constraint_upper;
    PreFOSWorkingMatrix owned_current;
    unsigned char *quadratic_column = NULL;
    unsigned char *factor_column = NULL;
    unsigned char *protected_target = NULL;
    double *working_objective = NULL;
    int *record_rows = NULL;
    double *record_coefficients = NULL;
    double working_objective_offset = 0.0;
    size_t row, round;
    PreFOSStatus status = PREFOS_STATUS_OK;

    memset(&owned_current, 0, sizeof(owned_current));
    if (!presolver->settings.free_column_substitution || problem->n_box == 0 ||
        problem->A.rows == 0)
        return PREFOS_STATUS_OK;
    if (!has_basic_free_column_candidate(presolver))
        return PREFOS_STATUS_OK;

    quadratic_column = (unsigned char *) calloc(problem->n, sizeof(unsigned char));
    factor_column = (unsigned char *) calloc(problem->n, sizeof(unsigned char));
    protected_target = (unsigned char *) calloc(problem->n, sizeof(unsigned char));
    if (problem->n > 0 &&
        (!quadratic_column || !factor_column || !protected_target))
    {
        status = PREFOS_STATUS_OUT_OF_MEMORY;
        goto cleanup;
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
    {
        int has_candidate = 0;
        for (row = 0; row < problem->n; ++row)
        {
            if (column_can_be_eliminated(
                    presolver, (int) row, quadratic_column, factor_column,
                    protected_target, -1))
            {
                has_candidate = 1;
                break;
            }
        }
        if (!has_candidate) goto cleanup;
    }
    if (!has_enough_free_aggregation_candidates(
            presolver, quadratic_column, factor_column,
            protected_target))
        goto cleanup;

    working_objective =
        (double *) prefos_internal_alloc_array(problem->n, sizeof(double));
    record_rows = (int *) prefos_internal_alloc_array(
        (size_t) presolver->settings.max_aggregation_column_degree,
        sizeof(int));
    record_coefficients = (double *) prefos_internal_alloc_array(
        (size_t) presolver->settings.max_aggregation_column_degree,
        sizeof(double));
    if ((problem->n > 0 && !working_objective) ||
        !record_rows || !record_coefficients)
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

    if (presolver->n_fixed_columns > 0 ||
        presolver->n_residual_row_substitutions > 0 ||
        presolver->stats.substituted_bounded_doubletons > 0 ||
        presolver->n_parallel_column_reductions > 0 ||
        presolver->n_affine_face_substitutions > 0)
    {
        status = prefos_internal_materialize_working_matrix(
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
        int *active_degrees = NULL;
        double *column_coefficients = NULL;
        size_t accepted = 0;

        memset(protected_target, 0, problem->n * sizeof(unsigned char));
        status = build_column_storage(
            current_matrix, presolver->remove_rows,
            &column_starts, &column_rows,
            &column_coefficients, &active_degrees);
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
                                              protected_target, (int) row))
                    continue;
                active_degree = (size_t) active_degrees[eliminated];
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
                fill = aggregation_fill(
                    presolver, column_starts, column_rows,
                    eliminated, (int) row, targets, term_count);
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
                double updated_objective[PREFOS_MAX_AGGREGATION_TERMS];
                size_t record_degree = 0, term;
                int adjacent;

                for (adjacent = column_starts[best_eliminated];
                     adjacent < column_starts[best_eliminated + 1]; ++adjacent)
                {
                    int adjacent_row = column_rows[adjacent];
                    if (presolver->remove_rows[adjacent_row]) continue;
                    if (record_degree >=
                        (size_t) presolver->settings
                            .max_aggregation_column_degree)
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
                                  (int) row, protected_target, record_rows,
                                  record_degree);
                for (p = current_matrix->row_pointers[row];
                     p < current_matrix->row_pointers[row + 1]; ++p)
                {
                    int column = current_matrix->column_indices[p];
                    if (current_matrix->values[p] != 0.0 &&
                        active_degrees[column] > 0)
                        --active_degrees[column];
                }
                working_objective[best_eliminated] = 0.0;
                for (term = 0; term < best_term_count; ++term)
                    working_objective[best_targets[term]] = updated_objective[term];
                ++accepted;
            }
        }
        free(column_starts);
        free(column_rows);
        free(column_coefficients);
        free(active_degrees);
        if (status != PREFOS_STATUS_OK) goto cleanup;
        if (accepted == 0 ||
            round + 1 ==
                (size_t) presolver->settings.max_aggregation_rounds)
            break;
        {
            PreFOSWorkingMatrix next;
            status = prefos_internal_materialize_working_matrix(
                presolver, current_matrix, current_lower, current_upper, &next);
            if (status != PREFOS_STATUS_OK) goto cleanup;
            prefos_internal_free_working_matrix(&owned_current);
            owned_current = next;
            current_matrix = &owned_current.matrix;
            current_lower = owned_current.lower;
            current_upper = owned_current.upper;
        }
    }

cleanup:
    prefos_internal_free_working_matrix(&owned_current);
    free(quadratic_column);
    free(factor_column);
    free(protected_target);
    free(working_objective);
    free(record_rows);
    free(record_coefficients);
    return status;
}
