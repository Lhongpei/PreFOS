/*
 * Copyright 2026 Hongpei Li
 * SPDX-License-Identifier: Apache-2.0
 */

#include <PreFOS/PreFOS.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(condition)                                                        \
    do                                                                          \
    {                                                                           \
        if (!(condition))                                                       \
        {                                                                       \
            fprintf(stderr, "check failed at %s:%d: %s\n", __FILE__, __LINE__,  \
                    #condition);                                                \
            return 1;                                                           \
        }                                                                       \
    } while (0)

static PreFOSSettings gpu_base_settings(void)
{
    PreFOSSettings settings = prefos_strict_settings();
    settings.remove_empty_rows = 1;
    settings.remove_redundant_rows = 0;
    settings.free_column_substitution = 0;
    settings.linear_propagation = 0;
    settings.linear_propagation_gpu = 0;
    settings.cone_propagation = 0;
    settings.cone_aware_row_activity = 0;
    settings.exponential_propagation = 0;
    settings.power_propagation = 0;
    settings.psd_higher_order_propagation = 0;
    settings.affine_cone_coordinate_aggregation = 0;
    settings.psd_structure_analysis = 0;
    settings.psd_block_decomposition = 0;
    settings.remove_empty_columns = 0;
    settings.singleton_column_reduction = 0;
    settings.bounded_doubleton_substitution = 0;
    settings.dual_fixing = 0;
    settings.parallel_column_reduction = 0;
    settings.remove_redundant_bounds = 0;
    settings.structural_reductions_gpu = 0;
    return settings;
}

static int values_close(double left, double right)
{
    double scale = fmax(1.0, fmax(fabs(left), fabs(right)));
    return fabs(left - right) <= 1e-11 * scale;
}

static int compare_csr(const PreFOSCsrMatrix *left,
                       const PreFOSCsrMatrix *right)
{
    size_t row, position;
    if (left->rows != right->rows || left->cols != right->cols ||
        left->nnz != right->nnz)
        return 0;
    for (row = 0; row <= left->rows; ++row)
        if (left->row_pointers[row] != right->row_pointers[row]) return 0;
    for (position = 0; position < left->nnz; ++position)
        if (left->column_indices[position] !=
                right->column_indices[position] ||
            !values_close(left->values[position], right->values[position]))
            return 0;
    return 1;
}

static int compare_reduced(const PreFOSPresolvedProblem *left,
                           const PreFOSPresolvedProblem *right)
{
    size_t row;
    if (!left || !right || left->n != right->n ||
        left->n_box != right->n_box ||
        left->n_cones != right->n_cones ||
        left->n_affine_cones != right->n_affine_cones ||
        !compare_csr(&left->A, &right->A) ||
        !compare_csr(&left->Q, &right->Q) ||
        !compare_csr(&left->R, &right->R) ||
        !compare_csr(&left->affine_cone_matrix,
                     &right->affine_cone_matrix))
        return 0;
    for (row = 0; row < left->A.rows; ++row)
        if (!values_close(left->constraint_lower[row],
                          right->constraint_lower[row]) ||
            !values_close(left->constraint_upper[row],
                          right->constraint_upper[row]))
            return 0;
    for (row = 0; row < left->n; ++row)
        if (!values_close(left->c[row], right->c[row])) return 0;
    for (row = 0; row < left->n_box; ++row)
        if (left->box_indices[row] != right->box_indices[row] ||
            !values_close(left->box_lower[row], right->box_lower[row]) ||
            !values_close(left->box_upper[row], right->box_upper[row]))
            return 0;
    return values_close(left->objective_offset,
                        right->objective_offset);
}

static int test_bulk_linear_and_structural_paths(void)
{
    const size_t n = 128;
    const size_t m = 64;
    const size_t nnz = n * m;
    double *a_values = NULL, *lower = NULL, *upper = NULL;
    double *c = NULL, *box_lower = NULL, *box_upper = NULL;
    int *a_columns = NULL, *a_rows = NULL, *q_rows = NULL;
    int *box_indices = NULL;
    int r_rows[] = {0};
    PreFOSProblemData problem;
    PreFOSSettings gpu_settings = gpu_base_settings();
    PreFOSSettings cpu_settings;
    PreFOSPresolver *gpu = NULL, *cpu = NULL;
    const PreFOSStats *stats;
    size_t row, column, position = 0;
    int result = 1;

    a_values = (double *) malloc(nnz * sizeof(double));
    a_columns = (int *) malloc(nnz * sizeof(int));
    a_rows = (int *) malloc((m + 1) * sizeof(int));
    lower = (double *) malloc(m * sizeof(double));
    upper = (double *) malloc(m * sizeof(double));
    q_rows = (int *) calloc(n + 1, sizeof(int));
    c = (double *) calloc(n, sizeof(double));
    box_indices = (int *) malloc(n * sizeof(int));
    box_lower = (double *) malloc(n * sizeof(double));
    box_upper = (double *) malloc(n * sizeof(double));
    if (!a_values || !a_columns || !a_rows || !lower || !upper ||
        !q_rows || !c || !box_indices || !box_lower || !box_upper)
        goto cleanup;
    for (row = 0; row < m; ++row)
    {
        a_rows[row] = (int) position;
        lower[row] = -10000.0;
        upper[row] = 10000.0;
        for (column = 0; column < n; ++column)
        {
            int residue =
                (int) ((17U * row + 13U * column) % 23U) - 11;
            a_values[position] =
                residue == 0 ? 0.25 : (double) residue / 7.0;
            a_columns[position] = (int) column;
            ++position;
        }
    }
    a_rows[m] = (int) position;
    for (column = 0; column < n; ++column)
    {
        box_indices[column] = (int) column;
        box_lower[column] = column < 16 ? 0.0 : -1.0;
        box_upper[column] = column < 16 ? 0.0 : 1.0;
    }
    memset(&problem, 0, sizeof(problem));
    problem.n = n;
    problem.A = (PreFOSCsrMatrix){
        m, n, nnz, a_values, a_columns, a_rows};
    problem.constraint_lower = lower;
    problem.constraint_upper = upper;
    problem.Q =
        (PreFOSCsrMatrix){n, n, 0, NULL, NULL, q_rows};
    problem.q_storage = PREFOS_Q_UPPER_TRIANGULAR;
    problem.R =
        (PreFOSCsrMatrix){0, n, 0, NULL, NULL, r_rows};
    problem.c = c;
    problem.n_box = n;
    problem.box_indices = box_indices;
    problem.box_lower = box_lower;
    problem.box_upper = box_upper;

    gpu_settings.fix_close_box_bounds = 1;
    gpu_settings.linear_propagation = 1;
    gpu_settings.max_linear_propagation_rounds = 1;
    gpu_settings.linear_propagation_gpu = 1;
    gpu_settings.event_queue_max_average_column_degree = 0.1;
    gpu_settings.singleton_column_reduction = 1;
    gpu_settings.parallel_column_reduction = 1;
    gpu_settings.structural_reductions_gpu = 1;
    cpu_settings = gpu_settings;
    cpu_settings.linear_propagation_gpu = 0;
    cpu_settings.structural_reductions_gpu = 0;

    if (prefos_create_presolver(&problem, &gpu_settings, &gpu) !=
            PREFOS_STATUS_OK ||
        prefos_create_presolver(&problem, &cpu_settings, &cpu) !=
            PREFOS_STATUS_OK)
        goto cleanup;
    if (prefos_run_presolve(gpu) != PREFOS_STATUS_REDUCED ||
        prefos_run_presolve(cpu) != PREFOS_STATUS_REDUCED)
        goto cleanup;
    stats = prefos_get_stats(gpu);
    if (!stats ||
        stats->column_csc_gpu_builds == 0 ||
        stats->column_csc_gpu_fallbacks != 0 ||
        stats->singleton_column_gpu_passes == 0 ||
        stats->singleton_column_gpu_fallbacks != 0 ||
        stats->parallel_column_gpu_passes == 0 ||
        stats->parallel_column_gpu_fallbacks != 0 ||
        stats->linear_gpu_rounds == 0 ||
        stats->linear_gpu_fallbacks != 0 ||
        stats->matrix_compaction_gpu_passes == 0 ||
        stats->matrix_compaction_gpu_fallbacks != 0 ||
        !compare_reduced(prefos_get_reduced_problem(gpu),
                         prefos_get_reduced_problem(cpu)))
        goto cleanup;
    result = 0;

cleanup:
    prefos_free_presolver(gpu);
    prefos_free_presolver(cpu);
    free(a_values);
    free(a_columns);
    free(a_rows);
    free(lower);
    free(upper);
    free(q_rows);
    free(c);
    free(box_indices);
    free(box_lower);
    free(box_upper);
    return result;
}

static int test_parallel_row_and_cone_activity_paths(void)
{
    const size_t n = 512;
    const size_t m = 256;
    const size_t nnz = 2 * m;
    double *a_values = NULL, *lower = NULL, *upper = NULL;
    double *c = NULL, *box_lower = NULL, *box_upper = NULL;
    int *a_columns = NULL, *a_rows = NULL, *q_rows = NULL;
    int *box_indices = NULL, *cone_indices = NULL;
    int r_rows[] = {0};
    PreFOSConeBlock cone;
    PreFOSProblemData problem;
    PreFOSSettings settings = gpu_base_settings();
    PreFOSPresolver *presolver = NULL;
    const PreFOSStats *stats;
    size_t row, coordinate;
    int result = 1;

    a_values = (double *) malloc(nnz * sizeof(double));
    a_columns = (int *) malloc(nnz * sizeof(int));
    a_rows = (int *) malloc((m + 1) * sizeof(int));
    lower = (double *) malloc(m * sizeof(double));
    upper = (double *) malloc(m * sizeof(double));
    q_rows = (int *) calloc(n + 1, sizeof(int));
    c = (double *) calloc(n, sizeof(double));
    box_indices = (int *) malloc((n - 64) * sizeof(int));
    box_lower = (double *) malloc((n - 64) * sizeof(double));
    box_upper = (double *) malloc((n - 64) * sizeof(double));
    cone_indices = (int *) malloc(64 * sizeof(int));
    if (!a_values || !a_columns || !a_rows || !lower || !upper ||
        !q_rows || !c || !box_indices || !box_lower || !box_upper ||
        !cone_indices)
        goto cleanup;
    for (coordinate = 0; coordinate < 64; ++coordinate)
        cone_indices[coordinate] = (int) coordinate;
    for (coordinate = 64; coordinate < n; ++coordinate)
    {
        size_t box = coordinate - 64;
        box_indices[box] = (int) coordinate;
        box_lower[box] = -1.0;
        box_upper[box] = 1.0;
    }
    for (row = 0; row < m; ++row)
    {
        size_t pair = row / 2;
        a_rows[row] = (int) (2 * row);
        a_values[2 * row] = 1.0;
        a_columns[2 * row] = 0;
        a_values[2 * row + 1] =
            1.0 + 0.125 * (double) (pair % 7);
        a_columns[2 * row + 1] =
            (int) (64 + pair % (n - 64));
        lower[row] = -100.0;
        upper[row] = 100.0;
    }
    a_rows[m] = (int) nnz;
    cone = (PreFOSConeBlock){
        PREFOS_CONE_SECOND_ORDER, 64, 0, cone_indices, 0.0};
    memset(&problem, 0, sizeof(problem));
    problem.n = n;
    problem.A = (PreFOSCsrMatrix){
        m, n, nnz, a_values, a_columns, a_rows};
    problem.constraint_lower = lower;
    problem.constraint_upper = upper;
    problem.Q =
        (PreFOSCsrMatrix){n, n, 0, NULL, NULL, q_rows};
    problem.q_storage = PREFOS_Q_UPPER_TRIANGULAR;
    problem.R =
        (PreFOSCsrMatrix){0, n, 0, NULL, NULL, r_rows};
    problem.c = c;
    problem.n_box = n - 64;
    problem.box_indices = box_indices;
    problem.box_lower = box_lower;
    problem.box_upper = box_upper;
    problem.n_cones = 1;
    problem.cones = &cone;

    settings.remove_redundant_rows = 1;
    settings.cone_aware_row_activity = 1;
    settings.structural_reductions_gpu = 1;
    if (prefos_create_presolver(&problem, &settings, &presolver) !=
        PREFOS_STATUS_OK)
        goto cleanup;
    {
        PreFOSStatus status = prefos_run_presolve(presolver);
        if (status != PREFOS_STATUS_OK &&
            status != PREFOS_STATUS_REDUCED)
            goto cleanup;
    }
    stats = prefos_get_stats(presolver);
    if (!stats || stats->parallel_row_gpu_passes == 0 ||
        stats->parallel_row_gpu_fallbacks != 0 ||
        stats->cone_activity_gpu_passes == 0 ||
        stats->cone_activity_gpu_fallbacks != 0)
        goto cleanup;
    result = 0;

cleanup:
    prefos_free_presolver(presolver);
    free(a_values);
    free(a_columns);
    free(a_rows);
    free(lower);
    free(upper);
    free(q_rows);
    free(c);
    free(box_indices);
    free(box_lower);
    free(box_upper);
    free(cone_indices);
    return result;
}

static int test_direct_cone_gpu_path(void)
{
    const size_t n = 64;
    int a_rows[] = {0};
    int *q_rows = NULL, *indices = NULL;
    int r_rows[] = {0};
    double *c = NULL;
    PreFOSConeBlock cone;
    PreFOSProblemData problem;
    PreFOSSettings settings = gpu_base_settings();
    PreFOSPresolver *presolver = NULL;
    const PreFOSStats *stats;
    size_t i;
    int result = 1;

    q_rows = (int *) calloc(n + 1, sizeof(int));
    indices = (int *) malloc(n * sizeof(int));
    c = (double *) calloc(n, sizeof(double));
    if (!q_rows || !indices || !c) goto cleanup;
    for (i = 0; i < n; ++i) indices[i] = (int) i;
    cone = (PreFOSConeBlock){
        PREFOS_CONE_SECOND_ORDER, n, 0, indices, 0.0};
    memset(&problem, 0, sizeof(problem));
    problem.n = n;
    problem.A =
        (PreFOSCsrMatrix){0, n, 0, NULL, NULL, a_rows};
    problem.Q =
        (PreFOSCsrMatrix){n, n, 0, NULL, NULL, q_rows};
    problem.q_storage = PREFOS_Q_UPPER_TRIANGULAR;
    problem.R =
        (PreFOSCsrMatrix){0, n, 0, NULL, NULL, r_rows};
    problem.c = c;
    problem.n_cones = 1;
    problem.cones = &cone;
    settings.cone_propagation = 1;
    settings.max_cone_propagation_rounds = 1;
    settings.linear_propagation_gpu = 1;
    if (prefos_create_presolver(&problem, &settings, &presolver) !=
        PREFOS_STATUS_OK)
        goto cleanup;
    {
        PreFOSStatus status = prefos_run_presolve(presolver);
        if (status != PREFOS_STATUS_OK &&
            status != PREFOS_STATUS_REDUCED)
            goto cleanup;
    }
    stats = prefos_get_stats(presolver);
    if (!stats || stats->cone_gpu_rounds == 0 ||
        stats->cone_gpu_fallbacks != 0)
        goto cleanup;
    result = 0;

cleanup:
    prefos_free_presolver(presolver);
    free(q_rows);
    free(indices);
    free(c);
    return result;
}

static int test_affine_cone_gpu_path(void)
{
    const size_t n = 64;
    int a_rows[] = {0};
    int *q_rows = NULL, *g_rows = NULL, *g_columns = NULL;
    int *box_indices = NULL;
    int r_rows[] = {0};
    double *c = NULL, *g_values = NULL, *h = NULL;
    double *box_lower = NULL, *box_upper = NULL;
    PreFOSAffineConeBlock cone = {
        PREFOS_CONE_SECOND_ORDER, 64, 0, 0.0};
    PreFOSProblemData problem;
    PreFOSSettings settings = gpu_base_settings();
    PreFOSPresolver *presolver = NULL;
    const PreFOSStats *stats;
    size_t i;
    int result = 1;

    q_rows = (int *) calloc(n + 1, sizeof(int));
    g_rows = (int *) malloc((n + 1) * sizeof(int));
    g_columns = (int *) malloc(n * sizeof(int));
    g_values = (double *) malloc(n * sizeof(double));
    h = (double *) calloc(n, sizeof(double));
    c = (double *) calloc(n, sizeof(double));
    box_indices = (int *) malloc(n * sizeof(int));
    box_lower = (double *) malloc(n * sizeof(double));
    box_upper = (double *) malloc(n * sizeof(double));
    if (!q_rows || !g_rows || !g_columns || !g_values || !h || !c ||
        !box_indices || !box_lower || !box_upper)
        goto cleanup;
    for (i = 0; i < n; ++i)
    {
        g_rows[i] = (int) i;
        g_columns[i] = (int) i;
        g_values[i] = 1.0;
        box_indices[i] = (int) i;
        box_lower[i] = -1.0;
        box_upper[i] = 1.0;
    }
    g_rows[n] = (int) n;
    h[0] = 100.0;
    memset(&problem, 0, sizeof(problem));
    problem.n = n;
    problem.A =
        (PreFOSCsrMatrix){0, n, 0, NULL, NULL, a_rows};
    problem.Q =
        (PreFOSCsrMatrix){n, n, 0, NULL, NULL, q_rows};
    problem.q_storage = PREFOS_Q_UPPER_TRIANGULAR;
    problem.R =
        (PreFOSCsrMatrix){0, n, 0, NULL, NULL, r_rows};
    problem.c = c;
    problem.n_box = n;
    problem.box_indices = box_indices;
    problem.box_lower = box_lower;
    problem.box_upper = box_upper;
    problem.affine_cone_matrix = (PreFOSCsrMatrix){
        n, n, n, g_values, g_columns, g_rows};
    problem.affine_cone_offset = h;
    problem.n_affine_cones = 1;
    problem.affine_cones = &cone;
    settings.cone_propagation = 1;
    settings.max_cone_propagation_rounds = 1;
    settings.linear_propagation_gpu = 1;
    if (prefos_create_presolver(&problem, &settings, &presolver) !=
        PREFOS_STATUS_OK)
        goto cleanup;
    {
        PreFOSStatus status = prefos_run_presolve(presolver);
        if (status != PREFOS_STATUS_OK &&
            status != PREFOS_STATUS_REDUCED)
            goto cleanup;
    }
    stats = prefos_get_stats(presolver);
    if (!stats || stats->affine_cone_gpu_rounds == 0 ||
        stats->affine_cone_gpu_fallbacks != 0)
        goto cleanup;
    result = 0;

cleanup:
    prefos_free_presolver(presolver);
    free(q_rows);
    free(g_rows);
    free(g_columns);
    free(g_values);
    free(h);
    free(c);
    free(box_indices);
    free(box_lower);
    free(box_upper);
    return result;
}

int main(void)
{
    if (!prefos_gpu_warmup())
    {
        fprintf(stderr, "CUDA device unavailable; skipping path tests\n");
        return 77;
    }
    CHECK(test_bulk_linear_and_structural_paths() == 0);
    CHECK(test_parallel_row_and_cone_activity_paths() == 0);
    CHECK(test_direct_cone_gpu_path() == 0);
    CHECK(test_affine_cone_gpu_path() == 0);
    prefos_gpu_release_cache();
    printf("All PreFOS CUDA path tests passed!\n");
    return 0;
}
