/*
 * Copyright 2026 Hongpei Li
 * SPDX-License-Identifier: Apache-2.0
 */

#include "PREFOS_ColumnReductionInternal.h"
#include "PREFOS_CudaBackend.h"
#include "PREFOS_CudaLinearPropagation.h"
#include "core/PREFOS_Timer.h"

#include <stdio.h>

#define PREFOS_MAX_SINGLETON_RESIDUAL_WAVES 16

static int trace_singleton_row(int row)
{
    static int initialized = 0;
    static int selected = -1;
    if (!initialized)
    {
        const char *value = getenv("PREFOS_TRACE_SINGLETON_ROW");
        char *end;
        if (value && *value)
        {
            long parsed = strtol(value, &end, 10);
            if (end != value)
                selected = (int) parsed;
        }
        initialized = 1;
    }
    return selected == row;
}

static int trace_singleton_columns(void)
{
    static int initialized = 0;
    static int enabled = 0;
    if (!initialized)
    {
        const char *value = getenv("PREFOS_TRACE_SINGLETON_COLUMNS");
        enabled = value && *value && *value != '0';
        initialized = 1;
    }
    return enabled;
}

static PreFOSStatus populate_gpu_singleton_candidates(
    PreFOSPresolver *presolver, PreFOSColumnWorkspace *workspace)
{
    const PreFOSProblemData *problem = &presolver->original;
    unsigned char *eligible = NULL;
    PreFOSCudaPropagationStatus cuda_status =
        PREFOS_CUDA_PROPAGATION_UNAVAILABLE;
    PreFOSCudaWorkspace *cuda_workspace = NULL;
    double milliseconds = 0.0;
    size_t column;

    workspace->gpu_singleton_candidates_valid = 0;
    workspace->n_gpu_singleton_candidates = 0;
    if (!presolver->settings.structural_reductions_gpu ||
        !workspace->gpu_csc_valid)
        return PREFOS_STATUS_OK;
    workspace->gpu_singleton_candidates = (int *)
        prefos_internal_alloc_array(problem->n, sizeof(int));
    eligible = (unsigned char *)
        prefos_internal_alloc_array(problem->n, sizeof(unsigned char));
    if (problem->n > 0 &&
        (!workspace->gpu_singleton_candidates || !eligible))
    {
        free(eligible);
        return PREFOS_STATUS_OUT_OF_MEMORY;
    }
    for (column = 0; column < problem->n; ++column)
        eligible[column] = (unsigned char)
            (prefos_internal_column_is_linear_box(
                 presolver, workspace, (int) column) &&
             !workspace->protected_target[column]);
    if (workspace->gpu_csc_valid)
    {
        cuda_workspace =
            prefos_internal_cuda_workspace_get(presolver, &cuda_status);
        if (cuda_workspace && cuda_status == PREFOS_CUDA_PROPAGATION_OK)
            cuda_status = prefos_cuda_singleton_column_candidates(
                cuda_workspace, eligible, workspace->dirty_row,
                workspace->gpu_singleton_candidates,
                &workspace->n_gpu_singleton_candidates, &milliseconds);
    }
    free(eligible);
    presolver->stats.singleton_column_gpu_milliseconds += milliseconds;
    if (cuda_status == PREFOS_CUDA_PROPAGATION_OK)
    {
        workspace->gpu_singleton_candidates_valid = 1;
        ++presolver->stats.singleton_column_gpu_passes;
        presolver->stats.singleton_column_gpu_candidates +=
            workspace->n_gpu_singleton_candidates;
    }
    else
    {
        free(workspace->gpu_singleton_candidates);
        workspace->gpu_singleton_candidates = NULL;
        ++presolver->stats.singleton_column_gpu_fallbacks;
    }
    return PREFOS_STATUS_OK;
}

static PreFOSStatus append_equality_relaxed_record(
    PreFOSPresolver *presolver, int row, double side, double normal_sign)
{
    const PreFOSCsrMatrix *matrix = &presolver->original.A;
    int start = matrix->row_pointers[row];
    PresolveRowTransformationRecord record = {
        .type = PRESOLVE_ROW_EQUALITY_RELAXED,
        .row = row,
        .source_row = row,
        .ratio = normal_sign,
        .new_side = side,
        .indices = matrix->column_indices + start,
        .coefficients = matrix->values + start,
        .length = (size_t) (matrix->row_pointers[row + 1] - start)};
    return presolve_transformation_log_append_row_transformation(
               &presolver->transformations, &record, NULL)
               ? PREFOS_STATUS_OK
               : PREFOS_STATUS_OUT_OF_MEMORY;
}

static PreFOSStatus compute_singleton_row_activity(
    const PreFOSPresolver *presolver, int row, int excluded_column,
    int *targets, double *scales, size_t target_capacity,
    size_t *target_count, PreFOSSingletonRowActivity *activity)
{
    const PreFOSCsrMatrix *matrix = &presolver->original.A;
    int position;
    memset(activity, 0, sizeof(*activity));
    *target_count = 0;
    for (position = matrix->row_pointers[row];
         position < matrix->row_pointers[row + 1]; ++position)
    {
        int column = matrix->column_indices[position];
        int box_position = presolver->variable_to_box[column];
        double coefficient = matrix->values[position];
        double minimum_bound, maximum_bound;
        long double term;
        if (coefficient == 0.0) continue;
        if (presolver->is_fixed[column])
        {
            if (!prefos_internal_safe_add_product(
                    &activity->fixed_shift, coefficient,
                    presolver->fixed_values[column]))
                return PREFOS_STATUS_NUMERICAL_ERROR;
            continue;
        }
        if (presolver->is_substituted[column])
        {
            if (!prefos_internal_term_is_active_in_row(
                    presolver, (size_t) row, column))
                continue;
            activity->unsupported = 1;
            continue;
        }
        if (presolver->is_parallel_removed[column]) continue;
        ++activity->n_active_terms;
        if (column != excluded_column)
        {
            if (*target_count >= target_capacity)
                return PREFOS_STATUS_NUMERICAL_ERROR;
            targets[*target_count] = column;
            scales[*target_count] = coefficient;
            ++(*target_count);
        }
        minimum_bound =
            coefficient > 0.0
                ? presolver->propagation_lower[column]
                : presolver->propagation_upper[column];
        maximum_bound =
            coefficient > 0.0
                ? presolver->propagation_upper[column]
                : presolver->propagation_lower[column];
        if (box_position >= 0)
        {
            double lower =
                presolver->propagation_lower[column];
            double upper =
                presolver->propagation_upper[column];
            long double contribution =
                isfinite(lower) && isfinite(upper)
                    ? fabsl((long double) coefficient) *
                          ((long double) upper -
                           (long double) lower)
                    : INFINITY;
            if (contribution >
                activity->max_box_range_contribution)
                activity->max_box_range_contribution =
                    contribution;
            activity->has_box_column = 1;
        }
        if (isfinite(minimum_bound))
        {
            term = (long double) coefficient *
                   (long double) minimum_bound;
            if (!isfinite(term) ||
                !isfinite(activity->finite_min + term))
                return PREFOS_STATUS_NUMERICAL_ERROR;
            activity->finite_min += term;
        }
        else
            ++activity->n_infinite_min;
        if (isfinite(maximum_bound))
        {
            term = (long double) coefficient *
                   (long double) maximum_bound;
            if (!isfinite(term) ||
                !isfinite(activity->finite_max + term))
                return PREFOS_STATUS_NUMERICAL_ERROR;
            activity->finite_max += term;
        }
        else
            ++activity->n_infinite_max;
    }
    return PREFOS_STATUS_OK;
}

static PreFOSStatus get_singleton_row_activity(
    const PreFOSPresolver *presolver, PreFOSColumnWorkspace *workspace,
    int row, int *row_activity_map, unsigned int *row_activity_epochs,
    PreFOSSingletonRowActivity **activities, size_t *n_activities,
    size_t *activity_capacity, int excluded_column, int *targets,
    double *scales, size_t target_capacity, size_t *target_count,
    PreFOSSingletonRowActivity **activity, int *computed)
{
    int index = row_activity_map[row];
    *computed = 0;
    if (index < 0)
    {
        PreFOSSingletonRowActivity *expanded;
        size_t capacity;
        if (*n_activities == *activity_capacity)
        {
            capacity =
                *activity_capacity == 0 ? 64 : 2 * *activity_capacity;
            if (capacity < *activity_capacity ||
                capacity > presolver->original.A.rows)
                capacity = presolver->original.A.rows;
            if (capacity <= *activity_capacity ||
                capacity >
                    SIZE_MAX / sizeof(PreFOSSingletonRowActivity))
                return PREFOS_STATUS_OUT_OF_MEMORY;
            expanded = (PreFOSSingletonRowActivity *) realloc(
                *activities,
                capacity * sizeof(PreFOSSingletonRowActivity));
            if (!expanded) return PREFOS_STATUS_OUT_OF_MEMORY;
            *activities = expanded;
            *activity_capacity = capacity;
        }
        index = (int) (*n_activities)++;
        row_activity_map[row] = index;
    }
    if (row_activity_epochs[row] == 0U)
    {
        if (workspace->initial_row_activities &&
            workspace->initial_finite_min_accumulators &&
            workspace->initial_finite_max_accumulators &&
            workspace->initial_max_box_range_contribution &&
            workspace->initial_activity_has_box_column &&
            workspace->initial_activity_valid &&
            workspace->initial_activity_singleton_reusable &&
            workspace->initial_activity_valid[row] &&
            workspace->initial_activity_singleton_reusable[row])
        {
            const PresolveLinearActivity *source =
                &workspace->initial_row_activities[row];
            PreFOSSingletonRowActivity *target =
                *activities + index;
            memset(target, 0, sizeof(*target));
            target->finite_min =
                workspace->initial_finite_min_accumulators[row];
            target->finite_max =
                workspace->initial_finite_max_accumulators[row];
            target->max_box_range_contribution =
                workspace->initial_max_box_range_contribution[row];
            target->n_infinite_min = source->n_infinite_min;
            target->n_infinite_max = source->n_infinite_max;
            target->n_active_terms = source->n_nonzeros;
            target->has_box_column =
                workspace->initial_activity_has_box_column[row];
            *target_count = 0;
        }
        else
        {
            PreFOSStatus status = compute_singleton_row_activity(
                presolver, row, excluded_column, targets, scales,
                target_capacity, target_count, *activities + index);
            if (status != PREFOS_STATUS_OK) return status;
            *computed = 1;
        }
        row_activity_epochs[row] = 1U;
    }
    else
        *target_count = 0;
    *activity = *activities + index;
    return PREFOS_STATUS_OK;
}

static PreFOSStatus ensure_singleton_scratch(
    const PreFOSPresolver *presolver, PreFOSColumnWorkspace *workspace)
{
    size_t rows = presolver->original.A.rows;
    size_t row_capacity = workspace->max_row_nnz;
    if (!workspace->singleton_targets && row_capacity > 0)
        workspace->singleton_targets = (int *)
            prefos_internal_alloc_array(row_capacity, sizeof(int));
    if (!workspace->singleton_scales && row_capacity > 0)
        workspace->singleton_scales = (double *)
            prefos_internal_alloc_array(row_capacity, sizeof(double));
    if (!workspace->singleton_deferred_columns && rows > 0)
    {
        size_t row;
        workspace->singleton_deferred_columns = (int *)
            prefos_internal_alloc_array(rows, sizeof(int));
        if (workspace->singleton_deferred_columns)
            for (row = 0; row < rows; ++row)
                workspace->singleton_deferred_columns[row] = -1;
    }
    if (!workspace->singleton_deferred_rows && rows > 0)
        workspace->singleton_deferred_rows = (int *)
            prefos_internal_alloc_array(rows, sizeof(int));
    if (!workspace->singleton_row_activity_map && rows > 0)
    {
        size_t row;
        workspace->singleton_row_activity_map = (int *)
            prefos_internal_alloc_array(rows, sizeof(int));
        if (workspace->singleton_row_activity_map)
            for (row = 0; row < rows; ++row)
                workspace->singleton_row_activity_map[row] = -1;
    }
    if (!workspace->singleton_row_activity_epoch && rows > 0)
        workspace->singleton_row_activity_epoch = (unsigned int *)
            calloc(rows, sizeof(unsigned int));
    if (!workspace->singleton_processed_epoch && rows > 0)
        workspace->singleton_processed_epoch = (unsigned int *)
            calloc(rows, sizeof(unsigned int));
    if ((row_capacity > 0 &&
         (!workspace->singleton_targets ||
          !workspace->singleton_scales)) ||
        (rows > 0 &&
         (!workspace->singleton_deferred_columns ||
          !workspace->singleton_deferred_rows ||
          !workspace->singleton_row_activity_map ||
          !workspace->singleton_row_activity_epoch ||
          !workspace->singleton_processed_epoch)))
        return PREFOS_STATUS_OUT_OF_MEMORY;
    return PREFOS_STATUS_OK;
}

static unsigned int begin_singleton_epoch(
    const PreFOSPresolver *presolver, PreFOSColumnWorkspace *workspace)
{
    unsigned int epoch = workspace->singleton_epoch + 1U;
    if (epoch == 0U)
    {
        size_t rows = presolver->original.A.rows;
        memset(workspace->singleton_processed_epoch, 0,
               rows * sizeof(unsigned int));
        epoch = 1U;
    }
    workspace->singleton_epoch = epoch;
    return epoch;
}

static PreFOSStatus remove_cached_singleton_term(
    const PreFOSPresolver *presolver,
    PreFOSSingletonRowActivity *activity, int column, double coefficient)
{
    double minimum_bound =
        coefficient > 0.0 ? presolver->propagation_lower[column]
                          : presolver->propagation_upper[column];
    double maximum_bound =
        coefficient > 0.0 ? presolver->propagation_upper[column]
                          : presolver->propagation_lower[column];
    long double term;

    if (activity->n_active_terms == 0)
        return PREFOS_STATUS_NUMERICAL_ERROR;
    if (isfinite(minimum_bound))
    {
        term = (long double) coefficient *
               (long double) minimum_bound;
        activity->finite_min -= term;
        if (!isfinite(activity->finite_min))
            return PREFOS_STATUS_NUMERICAL_ERROR;
    }
    else
    {
        if (activity->n_infinite_min == 0)
            return PREFOS_STATUS_NUMERICAL_ERROR;
        --activity->n_infinite_min;
    }
    if (isfinite(maximum_bound))
    {
        term = (long double) coefficient *
               (long double) maximum_bound;
        activity->finite_max -= term;
        if (!isfinite(activity->finite_max))
            return PREFOS_STATUS_NUMERICAL_ERROR;
    }
    else
    {
        if (activity->n_infinite_max == 0)
            return PREFOS_STATUS_NUMERICAL_ERROR;
        --activity->n_infinite_max;
    }
    --activity->n_active_terms;
    activity->max_box_range_contribution = INFINITY;
    return PREFOS_STATUS_OK;
}

static void exclude_singleton_from_activity(
    const PreFOSPresolver *presolver,
    const PreFOSSingletonRowActivity *activity, int column,
    double pivot, long double *rest_min, long double *rest_max,
    int *finite_rest_min, int *finite_rest_max)
{
    double minimum_bound =
        pivot > 0.0 ? presolver->propagation_lower[column]
                    : presolver->propagation_upper[column];
    double maximum_bound =
        pivot > 0.0 ? presolver->propagation_upper[column]
                    : presolver->propagation_lower[column];
    *rest_min = activity->finite_min;
    *rest_max = activity->finite_max;
    if (isfinite(minimum_bound))
    {
        *finite_rest_min = activity->n_infinite_min == 0;
        *rest_min -=
            (long double) pivot * (long double) minimum_bound;
    }
    else
        *finite_rest_min = activity->n_infinite_min == 1;
    if (isfinite(maximum_bound))
    {
        *finite_rest_max = activity->n_infinite_max == 0;
        *rest_max -=
            (long double) pivot * (long double) maximum_bound;
    }
    else
        *finite_rest_max = activity->n_infinite_max == 1;
}

static PreFOSStatus analyze_singleton_candidate(
    const PreFOSPresolver *presolver,
    const PreFOSColumnWorkspace *workspace, int row, int column,
    double pivot, double fixed_shift, long double rest_min,
    long double rest_max, int finite_rest_min, int finite_rest_max,
    double box_lower, double box_upper, int allow_one_sided,
    double *lower, double *upper,
    int *free_below, int *free_above, int *equality,
    int *row_tightening, int *reducible)
{
    int box_position = presolver->variable_to_box[column];
    long double implied_lower = -INFINITY;
    long double implied_upper = INFINITY;
    double objective = workspace->objective[column];
    *reducible = 0;
    *row_tightening = 0;
    if (box_position < 0 || pivot == 0.0) return PREFOS_STATUS_OK;
    *lower = presolver->working_constraint_lower[row] - fixed_shift;
    *upper = presolver->working_constraint_upper[row] - fixed_shift;
    if (isnan(*lower) || isnan(*upper))
        return PREFOS_STATUS_NUMERICAL_ERROR;
    if (pivot > 0.0)
    {
        if (isfinite(*lower) && finite_rest_max)
            implied_lower =
                ((long double) *lower - rest_max) / pivot;
        if (isfinite(*upper) && finite_rest_min)
            implied_upper =
                ((long double) *upper - rest_min) / pivot;
    }
    else
    {
        if (isfinite(*upper) && finite_rest_min)
            implied_lower =
                ((long double) *upper - rest_min) / pivot;
        if (isfinite(*lower) && finite_rest_max)
            implied_upper =
                ((long double) *lower - rest_max) / pivot;
    }
    *free_below =
        !isfinite(box_lower) ||
        (isfinite(implied_lower) &&
         implied_lower >= (long double) box_lower);
    *free_above =
        !isfinite(box_upper) ||
        (isfinite(implied_upper) &&
         implied_upper <= (long double) box_upper);
    *equality =
        isfinite(*lower) && isfinite(*upper) &&
        *lower == *upper;
    if (!*free_below && !*free_above)
    {
        if (*equality && allow_one_sided)
            *reducible = 1;
        return PREFOS_STATUS_OK;
    }
    if ((!*free_below || !*free_above) && !allow_one_sided)
        return PREFOS_STATUS_OK;
    if ((!*free_below || !*free_above) && !*equality)
    {
        int tighten_lower =
            ((objective < 0.0 && pivot > 0.0 && *free_above) ||
             (objective > 0.0 && pivot < 0.0 && *free_below)) &&
            isfinite(*upper);
        int tighten_upper =
            ((objective > 0.0 && pivot > 0.0 && *free_below) ||
             (objective < 0.0 && pivot < 0.0 && *free_above)) &&
            isfinite(*lower);
        if (tighten_lower)
            *row_tightening = 1;
        else if (tighten_upper)
            *row_tightening = -1;
        else
            return PREFOS_STATUS_OK;
    }
    *reducible = 1;
    return PREFOS_STATUS_OK;
}

static PreFOSStatus retain_singleton_box_as_row_bounds(
    PreFOSPresolver *presolver, PreFOSColumnWorkspace *workspace,
    int row, int column, double pivot,
    double box_lower, double box_upper,
    int free_below, int free_above)
{
    double equality_side =
        presolver->working_constraint_upper[row];
    double residual_lower = -INFINITY;
    double residual_upper = INFINITY;

    if (!free_below)
    {
        double side = equality_side;
        if (!isfinite(box_lower) ||
            !prefos_internal_safe_add_product(
                &side, -pivot, box_lower))
            return PREFOS_STATUS_NUMERICAL_ERROR;
        if (pivot > 0.0)
            residual_upper = side;
        else
            residual_lower = side;
    }
    if (!free_above)
    {
        double side = equality_side;
        if (!isfinite(box_upper) ||
            !prefos_internal_safe_add_product(
                &side, -pivot, box_upper))
            return PREFOS_STATUS_NUMERICAL_ERROR;
        if (pivot > 0.0)
            residual_lower = side;
        else
            residual_upper = side;
    }
    if (residual_lower >
        residual_upper +
            presolver->settings.feasibility_tolerance)
        return PREFOS_STATUS_PRIMAL_INFEASIBLE;
    prefos_internal_note_constraint_bound_change(presolver);
    presolver->working_constraint_lower[row] = residual_lower;
    presolver->working_constraint_upper[row] = residual_upper;
    if (presolver->working_matrix_is_materialized)
    {
        presolver->materialized_row_updates_require_cache = 1;
        prefos_internal_mark_row_requires_materialization(
            presolver, (size_t) row);
    }
    if (getenv("PREFOS_SKIP_RESIDUAL_SINGLETON_REQUEUE"))
        prefos_internal_notify_row_side_change(
            presolver, workspace, (size_t) row);
    else
        prefos_internal_queue_row_side_change(
            presolver, workspace, (size_t) row);
    if (trace_singleton_row(row))
        fprintf(
            stderr,
            "PreFOS singleton residual row=%d column=%d "
            "new_row=[%.17g,%.17g]\n",
            row, column, residual_lower, residual_upper);
    return PREFOS_STATUS_OK;
}

PreFOSStatus prefos_internal_reduce_singleton_columns(
    PreFOSPresolver *presolver, PreFOSColumnWorkspace *workspace,
    int allow_one_sided)
{
    const PreFOSProblemData *problem = &presolver->original;
    int *targets = NULL;
    int *deferred_columns = NULL;
    int *deferred_rows = NULL;
    int *row_activity_map = NULL;
    unsigned int *row_activity_epochs = NULL;
    unsigned int *processed_epochs = NULL;
    double *scales = NULL;
    size_t row_capacity = 0, row_index, column;
    size_t n_deferred_rows = 0;
    size_t residual_wave = 0;
    unsigned int epoch;
    int trace_timing =
        getenv("PREFOS_TRACE_SINGLETON_TIMING") != NULL;
    PreFOSTimestamp pass_start, pass_stop;
    double setup_milliseconds = 0.0;
    double activity_prep_milliseconds = 0.0;
    double activity_milliseconds = 0.0;
    double analysis_milliseconds = 0.0;
    double target_milliseconds = 0.0;
    double residual_milliseconds = 0.0;
    double scaling_milliseconds = 0.0;
    double substitution_milliseconds = 0.0;
    double degree_update_milliseconds = 0.0;
    PreFOSStatus status;
    if (trace_timing)
        prefos_internal_timer_now(&pass_start);
    if (!presolver->settings.singleton_column_reduction)
        return PREFOS_STATUS_OK;
    if (presolver->settings.structural_reductions_gpu)
    {
        status = populate_gpu_singleton_candidates(presolver, workspace);
        if (status != PREFOS_STATUS_OK) return status;
    }
    if (workspace->n_singleton_candidate_columns == 0)
    {
        status = PREFOS_STATUS_OK;
        goto cleanup;
    }
    row_capacity = workspace->max_row_nnz;
    if (row_capacity == 0)
    {
        for (row_index = 0; row_index < problem->A.rows; ++row_index)
        {
            size_t length = (size_t)
                (problem->A.row_pointers[row_index + 1] -
                 problem->A.row_pointers[row_index]);
            if (length > row_capacity) row_capacity = length;
        }
        workspace->max_row_nnz = row_capacity;
    }
    status = ensure_singleton_scratch(presolver, workspace);
    if (status != PREFOS_STATUS_OK) goto cleanup;
    targets = workspace->singleton_targets;
    scales = workspace->singleton_scales;
    deferred_columns = workspace->singleton_deferred_columns;
    deferred_rows = workspace->singleton_deferred_rows;
    row_activity_map = workspace->singleton_row_activity_map;
    row_activity_epochs = workspace->singleton_row_activity_epoch;
    processed_epochs = workspace->singleton_processed_epoch;
    epoch = begin_singleton_epoch(presolver, workspace);
    prefos_internal_sync_singleton_activity_events(
        presolver, workspace);
    prefos_internal_sync_initial_row_activity_events(
        presolver, workspace);
process_candidates:
    while (workspace->singleton_candidate_position <
           workspace->n_singleton_candidate_columns)
    {
        PreFOSTimestamp operation_start, operation_stop, candidate_start;
        if (trace_timing)
            prefos_internal_timer_now(&candidate_start);
        column = (size_t) workspace->singleton_candidate_columns[
            workspace->singleton_candidate_position++];
        int row, p, free_below = 0, free_above = 0;
        int row_was_removed;
        double fixed_shift = 0.0;
        double pivot = 0.0, lower, upper, side;
        double box_lower_without_row, box_upper_without_row;
        long double rest_min = 0.0L, rest_max = 0.0L;
        int finite_rest_min = 1, finite_rest_max = 1;
        int equality, computed_activity, reducible;
        int row_tightening = 0;
        PreFOSSingletonRowActivity *row_activity;
        PreFOSSubstitutionMode substitution_mode = PREFOS_SUBSTITUTION_STANDARD;
        size_t target_count = 0;
        ++presolver->stats.singleton_candidates_examined;
        workspace->singleton_column_queued[column] = 0;
        if (!prefos_internal_column_is_linear_box(
                presolver, workspace, (int) column) ||
            workspace->protected_target[column] ||
            presolver->substitution_incoming_depth[column] >=
                PREFOS_MAX_SUBSTITUTION_DEPTH ||
            workspace->live_degrees[column] != 1)
            continue;
        row = -1;
        for (p = workspace->starts[column];
             p < workspace->ends[column]; ++p)
        {
            int candidate_row = workspace->rows[p];
            if (!presolver->remove_rows[candidate_row])
            {
                row = candidate_row;
                pivot = workspace->values[p];
                break;
            }
        }
        if (row < 0 || workspace->dirty_row[row]) continue;
        if (processed_epochs[row] == epoch)
        {
            if (deferred_columns[row] < 0)
            {
                deferred_columns[row] = (int) column;
                deferred_rows[n_deferred_rows++] = row;
            }
            continue;
        }
        if (trace_timing)
        {
            prefos_internal_timer_now(&operation_stop);
            setup_milliseconds +=
                prefos_internal_timer_elapsed_milliseconds(
                    &candidate_start, &operation_stop);
        }
        if (trace_timing)
            prefos_internal_timer_now(&operation_start);
        status = get_singleton_row_activity(
            presolver, workspace, row, row_activity_map,
            row_activity_epochs,
            &workspace->singleton_row_activities,
            &workspace->n_singleton_row_activities,
            &workspace->singleton_row_activity_capacity, (int) column,
            targets, scales, row_capacity, &target_count, &row_activity,
            &computed_activity);
        if (trace_timing)
            prefos_internal_timer_now(&operation_stop);
        if (trace_timing)
            activity_milliseconds +=
                prefos_internal_timer_elapsed_milliseconds(
                    &operation_start, &operation_stop);
        if (status != PREFOS_STATUS_OK) goto cleanup;
        if (computed_activity)
        {
            presolver->stats.singleton_row_terms_scanned +=
                (size_t) (problem->A.row_pointers[row + 1] -
                          problem->A.row_pointers[row]);
            if (trace_timing)
                presolver->stats.singleton_row_scan_milliseconds +=
                    prefos_internal_timer_elapsed_milliseconds(
                        &operation_start, &operation_stop);
        }
        if (row_activity->unsupported ||
            row_activity->n_active_terms <= 1)
            continue;
        fixed_shift = row_activity->fixed_shift;
        if (!computed_activity)
            target_count = row_activity->n_active_terms - 1;
        if (trace_timing)
            prefos_internal_timer_now(&operation_start);
        exclude_singleton_from_activity(
            presolver, row_activity, (int) column, pivot,
            &rest_min, &rest_max, &finite_rest_min, &finite_rest_max);
        if (trace_timing)
        {
            prefos_internal_timer_now(&operation_stop);
            activity_prep_milliseconds +=
                prefos_internal_timer_elapsed_milliseconds(
                    &operation_start, &operation_stop);
        }
        if (pivot == 0.0 || target_count == 0 ||
            target_count > row_capacity)
            continue;
        box_lower_without_row =
            prefos_internal_box_bound_without_row(
                presolver, (int) column, row, 1);
        box_upper_without_row =
            prefos_internal_box_bound_without_row(
                presolver, (int) column, row, 0);
        if (trace_timing)
            prefos_internal_timer_now(&operation_start);
        status = analyze_singleton_candidate(
            presolver, workspace, row, (int) column, pivot,
            fixed_shift, rest_min, rest_max, finite_rest_min,
            finite_rest_max, box_lower_without_row,
            box_upper_without_row, allow_one_sided, &lower, &upper,
            &free_below, &free_above, &equality, &row_tightening,
            &reducible);
        if (trace_timing)
        {
            prefos_internal_timer_now(&operation_stop);
            analysis_milliseconds +=
                prefos_internal_timer_elapsed_milliseconds(
                    &operation_start, &operation_stop);
        }
        if (status != PREFOS_STATUS_OK) goto cleanup;
        /*
         * A live-degree-one substitution changes no other active row.  The
         * generic event replay would nevertheless walk the eliminated and
         * target columns on the next pass.  Invalidate the only affected
         * cached source row here and acknowledge the events produced by this
         * reduction.
         */
        if (workspace->initial_activity_valid)
            workspace->initial_activity_valid[row] = 0;
        workspace->initial_activity_event_cursor =
            presolver->transformations.n_events;
        if (trace_singleton_columns())
            fprintf(
                stderr,
                "PreFOS singleton candidate column=%zu row=%d pivot=%.17g "
                "objective=%.17g row=[%.17g,%.17g] "
                "box=[%.17g,%.17g] free_below=%d free_above=%d "
                "equality=%d tightening=%d reducible=%d targets=%zu "
                "rest=[%.21Lg,%.21Lg] finite=[%d,%d]\n",
                column, row, pivot, workspace->objective[column],
                presolver->working_constraint_lower[row],
                presolver->working_constraint_upper[row],
                presolver->working_box_lower[
                    presolver->variable_to_box[column]],
                presolver->working_box_upper[
                    presolver->variable_to_box[column]],
                free_below, free_above, equality, row_tightening,
                reducible, target_count, rest_min, rest_max,
                finite_rest_min, finite_rest_max);
        if (!reducible) continue;

        if (!computed_activity)
        {
            size_t expected_targets =
                row_activity->n_active_terms - 1;
            int cache_mismatch = 0;
            target_count = 0;
            if (trace_timing)
                prefos_internal_timer_now(&operation_start);
            presolver->stats.singleton_row_terms_scanned +=
                (size_t) (problem->A.row_pointers[row + 1] -
                          problem->A.row_pointers[row]);
            for (p = problem->A.row_pointers[row];
                 p < problem->A.row_pointers[row + 1]; ++p)
            {
                int target = problem->A.column_indices[p];
                double coefficient = problem->A.values[p];
                if (coefficient == 0.0 ||
                    target == (int) column)
                    continue;
                if (presolver->is_fixed[target]) continue;
                if (presolver->is_substituted[target])
                {
                    if (!prefos_internal_term_is_active_in_row(
                            presolver, (size_t) row, target))
                        continue;
                    cache_mismatch = 1;
                    break;
                }
                if (presolver->is_parallel_removed[target]) continue;
                if (target_count >= row_capacity)
                {
                    cache_mismatch = 1;
                    break;
                }
                targets[target_count] = target;
                scales[target_count] = coefficient;
                ++target_count;
            }
            if (trace_timing)
                prefos_internal_timer_now(&operation_stop);
            if (trace_timing)
            {
                target_milliseconds +=
                    prefos_internal_timer_elapsed_milliseconds(
                        &operation_start, &operation_stop);
                presolver->stats.singleton_row_scan_milliseconds +=
                    prefos_internal_timer_elapsed_milliseconds(
                        &operation_start, &operation_stop);
            }
            if (cache_mismatch ||
                target_count != expected_targets)
            {
                prefos_internal_invalidate_singleton_row_activity(
                    workspace, (size_t) row);
                prefos_internal_queue_singleton_column(
                    presolver, workspace, (int) column);
                continue;
            }
            if (target_count == 0) continue;
        }

        if (!free_below || !free_above)
        {
            if (trace_singleton_row(row))
                fprintf(
                    stderr,
                    "PreFOS singleton residual candidate row=%d column=%zu "
                    "pivot=%.17g row=[%.17g,%.17g] box=[%.17g,%.17g] "
                    "free_below=%d free_above=%d equality=%d "
                    "row_tightening=%d\n",
                    row, column, pivot,
                    presolver->working_constraint_lower[row],
                    presolver->working_constraint_upper[row],
                    presolver->working_box_lower[
                        presolver->variable_to_box[column]],
                    presolver->working_box_upper[
                        presolver->variable_to_box[column]],
                    free_below, free_above, equality, row_tightening);
            if (!equality)
            {
                double active_side;
                if (row_tightening > 0)
                {
                    active_side =
                        presolver->working_constraint_upper[row];
                    status = append_equality_relaxed_record(
                        presolver, row, active_side, 1.0);
                    if (status != PREFOS_STATUS_OK) goto cleanup;
                    prefos_internal_note_constraint_bound_change(presolver);
                    presolver->working_constraint_lower[row] = active_side;
                    if (presolver->working_matrix_is_materialized)
                    {
                        presolver->materialized_row_updates_require_cache = 1;
                        prefos_internal_mark_row_requires_materialization(
                            presolver, (size_t) row);
                    }
                    lower = upper;
                }
                else
                {
                    active_side =
                        presolver->working_constraint_lower[row];
                    status = append_equality_relaxed_record(
                        presolver, row, active_side, -1.0);
                    if (status != PREFOS_STATUS_OK) goto cleanup;
                    prefos_internal_note_constraint_bound_change(presolver);
                    presolver->working_constraint_upper[row] = active_side;
                    if (presolver->working_matrix_is_materialized)
                    {
                        presolver->materialized_row_updates_require_cache = 1;
                        prefos_internal_mark_row_requires_materialization(
                            presolver, (size_t) row);
                    }
                    upper = lower;
                }
                ++presolver->stats.tightened_singleton_rows;
                prefos_internal_queue_row_side_change(
                    presolver, workspace, (size_t) row);
                row_tightening = 0;
                status = analyze_singleton_candidate(
                    presolver, workspace, row, (int) column, pivot,
                    fixed_shift, rest_min, rest_max, finite_rest_min,
                    finite_rest_max, box_lower_without_row,
                    box_upper_without_row, allow_one_sided,
                    &lower, &upper,
                    &free_below, &free_above, &equality, &row_tightening,
                    &reducible);
                if (status != PREFOS_STATUS_OK) goto cleanup;
                if (!reducible) continue;
            }
            if (!free_below || !free_above)
            {
                if (trace_timing)
                    prefos_internal_timer_now(&operation_start);
                status = retain_singleton_box_as_row_bounds(
                    presolver, workspace, row, (int) column,
                    pivot, box_lower_without_row,
                    box_upper_without_row, free_below, free_above);
                if (trace_timing)
                {
                    prefos_internal_timer_now(&operation_stop);
                    residual_milliseconds +=
                        prefos_internal_timer_elapsed_milliseconds(
                            &operation_start, &operation_stop);
                }
                if (status != PREFOS_STATUS_OK) goto cleanup;
                substitution_mode = PREFOS_SUBSTITUTION_RESIDUAL_ROW;
            }
        }
        if (workspace->objective[column] > 0.0)
            side = pivot > 0.0 ? lower : upper;
        else if (workspace->objective[column] < 0.0)
            side = pivot > 0.0 ? upper : lower;
        else
            side = isfinite(lower) ? lower : upper;
        if (!isfinite(side)) continue;
        if (trace_timing)
            prefos_internal_timer_now(&operation_start);
        for (row_index = 0; row_index < target_count; ++row_index)
            scales[row_index] = -scales[row_index] / pivot;
        if (trace_timing)
        {
            prefos_internal_timer_now(&operation_stop);
            scaling_milliseconds +=
                prefos_internal_timer_elapsed_milliseconds(
                    &operation_start, &operation_stop);
        }
        row_was_removed = presolver->remove_rows[row];
        if (trace_timing)
            prefos_internal_timer_now(&operation_start);
        status = prefos_internal_append_column_substitution(
            presolver, (int) column, targets, scales, target_count, row,
            side / pivot, pivot, workspace, substitution_mode, 0, 1);
        if (trace_timing)
        {
            prefos_internal_timer_now(&operation_stop);
            presolver->stats.singleton_substitution_milliseconds +=
                prefos_internal_timer_elapsed_milliseconds(
                    &operation_start, &operation_stop);
            substitution_milliseconds +=
                prefos_internal_timer_elapsed_milliseconds(
                    &operation_start, &operation_stop);
        }
        if (status != PREFOS_STATUS_OK) goto cleanup;
        if (trace_singleton_columns())
            fprintf(
                stderr,
                "PreFOS singleton applied column=%zu row=%d mode=%d "
                "constant=%.17g targets=%zu\n",
                column, row, (int) substitution_mode, side / pivot,
                target_count);
        if (substitution_mode == PREFOS_SUBSTITUTION_RESIDUAL_ROW)
        {
            status = remove_cached_singleton_term(
                presolver, row_activity, (int) column, pivot);
            if (status != PREFOS_STATUS_OK) goto cleanup;
            processed_epochs[row] = epoch;
            deferred_columns[row] = -1;
        }
        workspace->dirty_row[row] =
            (unsigned char)
                (substitution_mode !=
                 PREFOS_SUBSTITUTION_RESIDUAL_ROW);
        if (!row_was_removed && presolver->remove_rows[row])
        {
            if (trace_timing)
                prefos_internal_timer_now(&operation_start);
            prefos_internal_update_column_live_degrees(
                presolver, workspace);
            if (trace_timing)
            {
                prefos_internal_timer_now(&operation_stop);
                degree_update_milliseconds +=
                    prefos_internal_timer_elapsed_milliseconds(
                        &operation_start, &operation_stop);
            }
        }
    }
    status = PREFOS_STATUS_OK;
cleanup:
    workspace->singleton_candidate_position = 0;
    workspace->n_singleton_candidate_columns = 0;
    if (status == PREFOS_STATUS_OK)
        for (row_index = 0; row_index < n_deferred_rows; ++row_index)
        {
            int row = deferred_rows[row_index];
            int deferred_column = deferred_columns[row];
            deferred_columns[row] = -1;
            prefos_internal_queue_singleton_column(
                presolver, workspace, deferred_column);
        }
    if (status == PREFOS_STATUS_OK && n_deferred_rows > 0 &&
        ++residual_wave < PREFOS_MAX_SINGLETON_RESIDUAL_WAVES)
    {
        n_deferred_rows = 0;
        epoch = begin_singleton_epoch(presolver, workspace);
        goto process_candidates;
    }
    if (trace_timing)
    {
        prefos_internal_timer_now(&pass_stop);
        fprintf(
            stderr,
            "PreFOS singleton timing total_ms=%.3f setup_ms=%.3f "
            "activity_ms=%.3f prep_ms=%.3f analysis_ms=%.3f "
            "targets_ms=%.3f residual_ms=%.3f scaling_ms=%.3f "
            "substitution_ms=%.3f degree_update_ms=%.3f "
            "examined=%zu substituted=%zu\n",
            prefos_internal_timer_elapsed_milliseconds(
                &pass_start, &pass_stop),
            setup_milliseconds, activity_milliseconds,
            activity_prep_milliseconds, analysis_milliseconds,
            target_milliseconds, residual_milliseconds,
            scaling_milliseconds, substitution_milliseconds,
            degree_update_milliseconds,
            presolver->stats.singleton_candidates_examined,
            presolver->stats.substituted_free_variables);
    }
    return status;
}
