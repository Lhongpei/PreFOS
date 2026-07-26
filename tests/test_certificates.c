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

static int close_to(double left, double right)
{
    return fabs(left - right) <= 1e-10;
}

static PreFOSSettings certificate_settings(void)
{
    PreFOSSettings settings = prefos_strict_settings();
    settings.fix_close_box_bounds = 0;
    settings.remove_empty_rows = 0;
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

static void initialize_empty_quadratic(PreFOSProblemData *problem,
                                       size_t n, int *q_rows,
                                       int *r_rows)
{
    size_t i;
    for (i = 0; i <= n; ++i) q_rows[i] = 0;
    r_rows[0] = 0;
    problem->Q =
        (PreFOSCsrMatrix){n, n, 0, NULL, NULL, q_rows};
    problem->q_storage = PREFOS_Q_UPPER_TRIANGULAR;
    problem->R =
        (PreFOSCsrMatrix){0, n, 0, NULL, NULL, r_rows};
}

static int test_linear_infeasibility_certificate(void)
{
    double a_values[] = {1.0, 1.0, 1.0, 1.0};
    int a_columns[] = {0, 1, 0, 1};
    int a_rows[] = {0, 2, 4};
    double lower[] = {1.0, -INFINITY};
    double upper[] = {INFINITY, 0.0};
    int q_rows[3], r_rows[1];
    double c[] = {0.0, 0.0};
    int box_indices[] = {0, 1};
    double box_lower[] = {-INFINITY, -INFINITY};
    double box_upper[] = {INFINITY, INFINITY};
    double reduced_y[] = {-1.0, 1.0};
    double reduced_z[] = {0.0, 0.0};
    double original_y[2], original_z[2];
    PreFOSProblemData problem;
    PreFOSSettings settings = certificate_settings();
    PreFOSPresolver *presolver = NULL;
    PreFOSPostsolveInfeasibilityCertificateVerification verification;

    memset(&problem, 0, sizeof(problem));
    problem.n = 2;
    problem.A =
        (PreFOSCsrMatrix){2, 2, 4, a_values, a_columns, a_rows};
    problem.constraint_lower = lower;
    problem.constraint_upper = upper;
    initialize_empty_quadratic(&problem, 2, q_rows, r_rows);
    problem.c = c;
    problem.n_box = 2;
    problem.box_indices = box_indices;
    problem.box_lower = box_lower;
    problem.box_upper = box_upper;

    CHECK(prefos_create_presolver(&problem, &settings, &presolver) ==
          PREFOS_STATUS_OK);
    CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_OK);
    CHECK(prefos_postsolve_infeasibility_certificate(
              presolver, reduced_y, reduced_z, NULL, 1e-10,
              original_y, original_z, NULL) == PREFOS_STATUS_OK);
    CHECK(close_to(original_y[0], -1.0));
    CHECK(close_to(original_y[1], 1.0));
    CHECK(close_to(original_z[0], 0.0));
    CHECK(close_to(original_z[1], 0.0));
    CHECK(prefos_verify_postsolve_infeasibility_certificate(
              presolver, reduced_y, reduced_z, NULL, 1e-10,
              &verification) == PREFOS_STATUS_OK);
    CHECK(verification.passed);
    CHECK(close_to(verification.reduced.certificate_value, -1.0));
    CHECK(close_to(verification.original.certificate_value, -1.0));

    reduced_y[0] = 1.0;
    reduced_y[1] = -1.0;
    CHECK(prefos_verify_postsolve_infeasibility_certificate(
              presolver, reduced_y, reduced_z, NULL, 1e-10,
              &verification) == PREFOS_STATUS_OK);
    CHECK(!verification.passed);
    prefos_free_presolver(presolver);
    return 0;
}

static int test_affine_infeasibility_certificate(void)
{
    int a_rows[] = {0};
    int q_rows[2], r_rows[1];
    double c[] = {0.0};
    int box_indices[] = {0};
    double box_lower[] = {-INFINITY};
    double box_upper[] = {0.0};
    double g_values[] = {1.0};
    int g_columns[] = {0};
    int g_rows[] = {0, 1};
    double h[] = {-1.0};
    PreFOSAffineConeBlock affine = {
        PREFOS_CONE_NONNEGATIVE, 1, 0, 0.0};
    double reduced_z[] = {1.0};
    double reduced_affine_z[] = {-1.0};
    double original_z[1], original_affine_z[1];
    PreFOSProblemData problem;
    PreFOSSettings settings = certificate_settings();
    PreFOSPresolver *presolver = NULL;
    PreFOSPostsolveInfeasibilityCertificateVerification verification;

    memset(&problem, 0, sizeof(problem));
    problem.n = 1;
    problem.A =
        (PreFOSCsrMatrix){0, 1, 0, NULL, NULL, a_rows};
    initialize_empty_quadratic(&problem, 1, q_rows, r_rows);
    problem.c = c;
    problem.n_box = 1;
    problem.box_indices = box_indices;
    problem.box_lower = box_lower;
    problem.box_upper = box_upper;
    problem.affine_cone_matrix =
        (PreFOSCsrMatrix){1, 1, 1, g_values, g_columns, g_rows};
    problem.affine_cone_offset = h;
    problem.n_affine_cones = 1;
    problem.affine_cones = &affine;

    CHECK(prefos_create_presolver(&problem, &settings, &presolver) ==
          PREFOS_STATUS_OK);
    CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_OK);
    CHECK(prefos_postsolve_infeasibility_certificate(
              presolver, NULL, reduced_z, reduced_affine_z, 1e-10,
              NULL, original_z, original_affine_z) == PREFOS_STATUS_OK);
    CHECK(close_to(original_z[0], 1.0));
    CHECK(close_to(original_affine_z[0], -1.0));
    CHECK(prefos_verify_postsolve_infeasibility_certificate(
              presolver, NULL, reduced_z, reduced_affine_z, 1e-10,
              &verification) == PREFOS_STATUS_OK);
    CHECK(verification.passed);
    CHECK(close_to(verification.original.certificate_value, -1.0));
    prefos_free_presolver(presolver);
    return 0;
}

static int test_working_bound_blocks_free_substitution(void)
{
    double a_values[] = {1.0, -1.0, 1.0, -2.0, 1.0};
    int a_columns[] = {0, 1, 1, 2, 0};
    int a_rows[] = {0, 2, 4, 5};
    double lower[] = {0.0, 0.0, -INFINITY};
    double upper[] = {0.0, 0.0, 0.0};
    int q_rows[4], r_rows[1];
    double c[] = {0.0, 0.0, 0.0};
    int box_indices[] = {0, 1, 2};
    double box_lower[] = {-INFINITY, -INFINITY, -INFINITY};
    double box_upper[] = {INFINITY, INFINITY, INFINITY};
    double g_values[] = {1.0};
    int g_columns[] = {2};
    int g_rows[] = {0, 1};
    double h[] = {-1.0};
    PreFOSAffineConeBlock affine = {
        PREFOS_CONE_NONNEGATIVE, 1, 0, 0.0};
    double reduced_y[] = {-0.5};
    double reduced_z[] = {0.5, 0.0};
    double reduced_affine_z[] = {-1.0};
    PreFOSProblemData problem;
    PreFOSSettings settings = prefos_strict_settings();
    PreFOSPresolver *presolver = NULL;
    const PreFOSPresolvedProblem *reduced;
    PreFOSPostsolveInfeasibilityCertificateVerification verification;

    settings.remove_empty_columns = 0;
    settings.singleton_column_reduction = 0;
    settings.bounded_doubleton_substitution = 0;
    settings.dual_fixing = 0;
    settings.parallel_column_reduction = 0;
    settings.linear_propagation = 0;
    settings.cone_propagation = 0;
    settings.affine_cone_coordinate_aggregation = 0;
    memset(&problem, 0, sizeof(problem));
    problem.n = 3;
    problem.A =
        (PreFOSCsrMatrix){3, 3, 5, a_values, a_columns, a_rows};
    problem.constraint_lower = lower;
    problem.constraint_upper = upper;
    initialize_empty_quadratic(&problem, 3, q_rows, r_rows);
    problem.c = c;
    problem.n_box = 3;
    problem.box_indices = box_indices;
    problem.box_lower = box_lower;
    problem.box_upper = box_upper;
    problem.affine_cone_matrix =
        (PreFOSCsrMatrix){1, 3, 1, g_values, g_columns, g_rows};
    problem.affine_cone_offset = h;
    problem.n_affine_cones = 1;
    problem.affine_cones = &affine;

    CHECK(prefos_create_presolver(&problem, &settings, &presolver) ==
          PREFOS_STATUS_OK);
    CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_REDUCED);
    reduced = prefos_get_reduced_problem(presolver);
    CHECK(reduced && reduced->n == 2 && reduced->A.rows == 1);
    CHECK(reduced->n_box == 2 && close_to(reduced->box_upper[0], 0.0));
    CHECK(prefos_verify_postsolve_infeasibility_certificate(
              presolver, reduced_y, reduced_z, reduced_affine_z, 1e-10,
              &verification) == PREFOS_STATUS_OK);
    CHECK(verification.passed);
    prefos_free_presolver(presolver);
    return 0;
}

static int test_direct_cone_certificate(PreFOSConeType type,
                                        size_t dimension,
                                        size_t matrix_order,
                                        double power_alpha,
                                        size_t separated_coordinate)
{
    double a_value[] = {1.0};
    int a_column[] = {(int) separated_coordinate};
    int a_rows[] = {0, 1};
    double lower[] = {-INFINITY};
    double upper[] = {-1.0};
    int *q_rows = NULL;
    int r_rows[] = {0};
    double *c = NULL;
    int *indices = NULL;
    double *reduced_z = NULL;
    double reduced_y[] = {1.0};
    PreFOSConeBlock cone;
    PreFOSProblemData problem;
    PreFOSSettings settings = certificate_settings();
    PreFOSPresolver *presolver = NULL;
    PreFOSPostsolveInfeasibilityCertificateVerification verification;
    size_t i;
    int result = 1;
    PreFOSStatus status;

    q_rows = (int *) calloc(dimension + 1, sizeof(int));
    c = (double *) calloc(dimension, sizeof(double));
    indices = (int *) malloc(dimension * sizeof(int));
    reduced_z = (double *) calloc(dimension, sizeof(double));
    if (!q_rows || !c || !indices || !reduced_z) goto cleanup;
    for (i = 0; i < dimension; ++i) indices[i] = (int) i;
    reduced_z[separated_coordinate] = -1.0;
    cone = (PreFOSConeBlock){
        type, dimension, matrix_order, indices, power_alpha};
    memset(&problem, 0, sizeof(problem));
    problem.n = dimension;
    problem.A = (PreFOSCsrMatrix){
        1, dimension, 1, a_value, a_column, a_rows};
    problem.constraint_lower = lower;
    problem.constraint_upper = upper;
    initialize_empty_quadratic(&problem, dimension, q_rows, r_rows);
    problem.c = c;
    problem.n_cones = 1;
    problem.cones = &cone;
    if (prefos_create_presolver(&problem, &settings, &presolver) !=
        PREFOS_STATUS_OK)
        goto cleanup;
    status = prefos_run_presolve(presolver);
    if (status != PREFOS_STATUS_OK && status != PREFOS_STATUS_REDUCED)
        goto cleanup;
    if (prefos_verify_postsolve_infeasibility_certificate(
            presolver, reduced_y, reduced_z, NULL, 1e-9,
            &verification) != PREFOS_STATUS_OK ||
        !verification.passed)
        goto cleanup;
    result = 0;

cleanup:
    prefos_free_presolver(presolver);
    free(q_rows);
    free(c);
    free(indices);
    free(reduced_z);
    return result;
}

static int test_all_direct_cone_certificates(void)
{
    CHECK(test_direct_cone_certificate(
              PREFOS_CONE_SECOND_ORDER, 3, 0, 0.0, 0) == 0);
    CHECK(test_direct_cone_certificate(
              PREFOS_CONE_ROTATED_SECOND_ORDER, 3, 0, 0.0, 0) == 0);
    CHECK(test_direct_cone_certificate(
              PREFOS_CONE_POSITIVE_SEMIDEFINITE, 3, 2, 0.0, 0) == 0);
    CHECK(test_direct_cone_certificate(
              PREFOS_CONE_EXPONENTIAL, 3, 0, 0.0, 2) == 0);
    CHECK(test_direct_cone_certificate(
              PREFOS_CONE_POWER, 3, 0, 0.4, 0) == 0);
    return 0;
}

static int test_affine_cone_certificate(
    PreFOSConeType type, size_t dimension, size_t matrix_order,
    double power_alpha, size_t separated_coordinate)
{
    int a_rows[] = {0};
    int q_rows[2], r_rows[1];
    int *g_rows = NULL;
    double c[] = {0.0};
    int box_indices[] = {0};
    double box_lower[] = {-INFINITY};
    double box_upper[] = {INFINITY};
    double *h = NULL, *reduced_affine_z = NULL;
    double reduced_z[] = {0.0};
    PreFOSAffineConeBlock affine = {
        type, dimension, matrix_order, power_alpha};
    PreFOSProblemData problem;
    PreFOSSettings settings = certificate_settings();
    PreFOSPresolver *presolver = NULL;
    PreFOSPostsolveInfeasibilityCertificateVerification verification;
    int result = 1;
    PreFOSStatus status;

    g_rows = (int *) calloc(dimension + 1, sizeof(int));
    h = (double *) calloc(dimension, sizeof(double));
    reduced_affine_z =
        (double *) calloc(dimension, sizeof(double));
    if (!g_rows || !h || !reduced_affine_z) goto cleanup;
    h[separated_coordinate] = -1.0;
    reduced_affine_z[separated_coordinate] = -1.0;

    memset(&problem, 0, sizeof(problem));
    problem.n = 1;
    problem.A =
        (PreFOSCsrMatrix){0, 1, 0, NULL, NULL, a_rows};
    initialize_empty_quadratic(&problem, 1, q_rows, r_rows);
    problem.c = c;
    problem.n_box = 1;
    problem.box_indices = box_indices;
    problem.box_lower = box_lower;
    problem.box_upper = box_upper;
    problem.affine_cone_matrix = (PreFOSCsrMatrix){
        dimension, 1, 0, NULL, NULL, g_rows};
    problem.affine_cone_offset = h;
    problem.n_affine_cones = 1;
    problem.affine_cones = &affine;
    if (prefos_create_presolver(&problem, &settings, &presolver) !=
        PREFOS_STATUS_OK)
        goto cleanup;
    status = prefos_run_presolve(presolver);
    if (status != PREFOS_STATUS_OK && status != PREFOS_STATUS_REDUCED)
        goto cleanup;
    if (prefos_verify_postsolve_infeasibility_certificate(
            presolver, NULL, reduced_z, reduced_affine_z, 1e-9,
            &verification) != PREFOS_STATUS_OK ||
        !verification.passed)
        goto cleanup;
    result = 0;

cleanup:
    prefos_free_presolver(presolver);
    free(g_rows);
    free(h);
    free(reduced_affine_z);
    return result;
}

static int test_all_affine_cone_certificates(void)
{
    CHECK(test_affine_cone_certificate(
              PREFOS_CONE_NONNEGATIVE, 1, 0, 0.0, 0) == 0);
    CHECK(test_affine_cone_certificate(
              PREFOS_CONE_SECOND_ORDER, 3, 0, 0.0, 0) == 0);
    CHECK(test_affine_cone_certificate(
              PREFOS_CONE_ROTATED_SECOND_ORDER, 3, 0, 0.0, 0) == 0);
    CHECK(test_affine_cone_certificate(
              PREFOS_CONE_POSITIVE_SEMIDEFINITE, 3, 2, 0.0, 0) == 0);
    CHECK(test_affine_cone_certificate(
              PREFOS_CONE_EXPONENTIAL, 3, 0, 0.0, 2) == 0);
    CHECK(test_affine_cone_certificate(
              PREFOS_CONE_POWER, 3, 0, 0.4, 0) == 0);
    return 0;
}

static int test_propagated_bound_certificate(void)
{
    double a_values[] = {1.0, 1.0};
    int a_columns[] = {0, 1};
    int a_rows[] = {0, 2};
    double lower[] = {-INFINITY};
    double upper[] = {0.0};
    int q_rows[3], r_rows[1];
    double c[] = {0.0, 0.0};
    int box_indices[] = {0, 1};
    double box_lower[] = {-INFINITY, 0.0};
    double box_upper[] = {INFINITY, INFINITY};
    double g_values[] = {1.0};
    int g_columns[] = {0};
    int g_rows[] = {0, 1};
    double h[] = {-1.0};
    PreFOSAffineConeBlock affine = {
        PREFOS_CONE_NONNEGATIVE, 1, 0, 0.0};
    double reduced_z[] = {1.0, 0.0};
    double reduced_affine_z[] = {-1.0};
    double original_y[1], original_z[2], original_affine_z[1];
    PreFOSProblemData problem;
    PreFOSSettings settings = certificate_settings();
    PreFOSPresolver *presolver = NULL;
    const PreFOSPresolvedProblem *reduced;
    PreFOSPostsolveInfeasibilityCertificateVerification verification;

    settings.linear_propagation = 1;
    settings.max_linear_propagation_rounds = 1;
    settings.propagated_bound_policy =
        PREFOS_PROPAGATED_BOUND_POLICY_FIRST_ORDER;
    memset(&problem, 0, sizeof(problem));
    problem.n = 2;
    problem.A =
        (PreFOSCsrMatrix){1, 2, 2, a_values, a_columns, a_rows};
    problem.constraint_lower = lower;
    problem.constraint_upper = upper;
    initialize_empty_quadratic(&problem, 2, q_rows, r_rows);
    problem.c = c;
    problem.n_box = 2;
    problem.box_indices = box_indices;
    problem.box_lower = box_lower;
    problem.box_upper = box_upper;
    problem.affine_cone_matrix =
        (PreFOSCsrMatrix){1, 2, 1, g_values, g_columns, g_rows};
    problem.affine_cone_offset = h;
    problem.n_affine_cones = 1;
    problem.affine_cones = &affine;

    CHECK(prefos_create_presolver(&problem, &settings, &presolver) ==
          PREFOS_STATUS_OK);
    CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_REDUCED);
    reduced = prefos_get_reduced_problem(presolver);
    CHECK(reduced != NULL && reduced->n == 2);
    CHECK(reduced->box_upper[0] == 0.0);
    CHECK(prefos_postsolve_infeasibility_certificate(
              presolver, reduced->A.rows > 0 ? (double[]){0.0} : NULL,
              reduced_z, reduced_affine_z, 1e-10, original_y,
              original_z, original_affine_z) == PREFOS_STATUS_OK);
    CHECK(close_to(original_y[0], 1.0));
    CHECK(close_to(original_z[0], 0.0));
    CHECK(close_to(original_z[1], -1.0));
    CHECK(close_to(original_affine_z[0], -1.0));
    CHECK(prefos_verify_postsolve_infeasibility_certificate(
              presolver, reduced->A.rows > 0 ? (double[]){0.0} : NULL,
              reduced_z, reduced_affine_z, 1e-10,
              &verification) == PREFOS_STATUS_OK);
    CHECK(verification.passed);
    CHECK(verification.certificate_value_absolute_error <= 1e-10);
    prefos_free_presolver(presolver);
    return 0;
}

static int test_long_propagated_bound_certificate(void)
{
    const size_t n = 97;
    const size_t rows = n - 1;
    double *a_values = NULL, *lower = NULL, *upper = NULL;
    double *c = NULL, *box_lower = NULL, *box_upper = NULL;
    double *reduced_y = NULL, *reduced_z = NULL;
    int *a_columns = NULL, *a_rows = NULL, *q_rows = NULL;
    int *box_indices = NULL;
    int r_rows[] = {0};
    double g_values[] = {-1.0};
    int g_columns[] = {(int) (n - 1)};
    int g_rows[] = {0, 1};
    double h[] = {0.0};
    double reduced_affine_z[] = {-1.0};
    PreFOSAffineConeBlock affine = {
        PREFOS_CONE_NONNEGATIVE, 1, 0, 0.0};
    PreFOSProblemData problem;
    PreFOSSettings settings = certificate_settings();
    PreFOSPresolver *presolver = NULL;
    PreFOSPostsolveInfeasibilityCertificateVerification verification;
    size_t row, column;
    int result = 1;

    a_values = (double *) malloc(2 * rows * sizeof(double));
    a_columns = (int *) malloc(2 * rows * sizeof(int));
    a_rows = (int *) malloc((rows + 1) * sizeof(int));
    lower = (double *) malloc(rows * sizeof(double));
    upper = (double *) malloc(rows * sizeof(double));
    q_rows = (int *) calloc(n + 1, sizeof(int));
    c = (double *) calloc(n, sizeof(double));
    box_indices = (int *) malloc(n * sizeof(int));
    box_lower = (double *) malloc(n * sizeof(double));
    box_upper = (double *) malloc(n * sizeof(double));
    reduced_y = (double *) calloc(rows, sizeof(double));
    reduced_z = (double *) calloc(n, sizeof(double));
    if (!a_values || !a_columns || !a_rows || !lower || !upper ||
        !q_rows || !c || !box_indices || !box_lower || !box_upper ||
        !reduced_y || !reduced_z)
        goto cleanup;
    for (row = 0; row < rows; ++row)
    {
        a_rows[row] = (int) (2 * row);
        a_values[2 * row] = 1.0;
        a_values[2 * row + 1] = -1.0;
        a_columns[2 * row] = (int) row;
        a_columns[2 * row + 1] = (int) (row + 1);
        lower[row] = -INFINITY;
        upper[row] = 0.0;
    }
    a_rows[rows] = (int) (2 * rows);
    for (column = 0; column < n; ++column)
    {
        box_indices[column] = (int) column;
        box_lower[column] = column == 0 ? 1.0 : -INFINITY;
        box_upper[column] = INFINITY;
    }
    reduced_z[n - 1] = -1.0;

    settings.linear_propagation = 1;
    settings.max_linear_propagation_rounds = 8;
    memset(&problem, 0, sizeof(problem));
    problem.n = n;
    problem.A = (PreFOSCsrMatrix){
        rows, n, 2 * rows, a_values, a_columns, a_rows};
    problem.constraint_lower = lower;
    problem.constraint_upper = upper;
    initialize_empty_quadratic(&problem, n, q_rows, r_rows);
    problem.c = c;
    problem.n_box = n;
    problem.box_indices = box_indices;
    problem.box_lower = box_lower;
    problem.box_upper = box_upper;
    problem.affine_cone_matrix =
        (PreFOSCsrMatrix){1, n, 1, g_values, g_columns, g_rows};
    problem.affine_cone_offset = h;
    problem.n_affine_cones = 1;
    problem.affine_cones = &affine;
    if (prefos_create_presolver(&problem, &settings, &presolver) !=
        PREFOS_STATUS_OK)
        goto cleanup;
    if (prefos_run_presolve(presolver) != PREFOS_STATUS_REDUCED)
        goto cleanup;
    if (prefos_get_reduced_problem(presolver)->n != n ||
        prefos_get_reduced_problem(presolver)->A.rows != rows ||
        prefos_get_stats(presolver)->propagated_box_bounds < rows)
        goto cleanup;
    if (prefos_verify_postsolve_infeasibility_certificate(
            presolver, reduced_y, reduced_z, reduced_affine_z, 1e-10,
            &verification) != PREFOS_STATUS_OK ||
        !verification.passed)
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
    free(reduced_y);
    free(reduced_z);
    return result;
}

static int test_input_affine_bound_certificate(void)
{
    double a_values[] = {1.0, 1.0};
    int a_columns[] = {0, 1};
    int a_rows[] = {0, 2};
    double lower[] = {-INFINITY};
    double upper[] = {-1.0};
    int q_rows[3], r_rows[1];
    double c[] = {0.0, 0.0};
    int box_indices[] = {0, 1};
    double box_lower[] = {-INFINITY, 0.0};
    double box_upper[] = {INFINITY, 1.0};
    double g_values[] = {1.0};
    int g_columns[] = {0};
    int g_rows[] = {0, 1, 1};
    double h[] = {0.0, 0.0};
    PreFOSAffineConeBlock affine = {
        PREFOS_CONE_SECOND_ORDER, 2, 0, 0.0};
    double reduced_y[] = {1.0};
    double reduced_z[] = {-1.0, -1.0};
    double reduced_affine_z[] = {0.0, 0.0};
    double original_y[1], original_z[2], original_affine_z[2];
    PreFOSProblemData problem;
    PreFOSSettings settings = certificate_settings();
    PreFOSPresolver *presolver = NULL;
    const PreFOSPresolvedProblem *reduced;
    PreFOSPostsolveInfeasibilityCertificateVerification verification;

    settings.cone_propagation = 1;
    settings.max_cone_propagation_rounds = 2;
    memset(&problem, 0, sizeof(problem));
    problem.n = 2;
    problem.A =
        (PreFOSCsrMatrix){1, 2, 2, a_values, a_columns, a_rows};
    problem.constraint_lower = lower;
    problem.constraint_upper = upper;
    initialize_empty_quadratic(&problem, 2, q_rows, r_rows);
    problem.c = c;
    problem.n_box = 2;
    problem.box_indices = box_indices;
    problem.box_lower = box_lower;
    problem.box_upper = box_upper;
    problem.affine_cone_matrix =
        (PreFOSCsrMatrix){2, 2, 1, g_values, g_columns, g_rows};
    problem.affine_cone_offset = h;
    problem.n_affine_cones = 1;
    problem.affine_cones = &affine;

    CHECK(prefos_create_presolver(&problem, &settings, &presolver) ==
          PREFOS_STATUS_OK);
    CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_REDUCED);
    reduced = prefos_get_reduced_problem(presolver);
    CHECK(reduced != NULL && reduced->n == 2 && reduced->A.rows == 1);
    CHECK(close_to(reduced->box_lower[0], 0.0));
    CHECK(prefos_get_stats(presolver)->materialized_affine_cone_box_bounds == 1);
    CHECK(prefos_postsolve_infeasibility_certificate(
              presolver, reduced_y, reduced_z, reduced_affine_z, 1e-10,
              original_y, original_z, original_affine_z) ==
          PREFOS_STATUS_OK);
    CHECK(close_to(original_y[0], 1.0));
    CHECK(close_to(original_z[0], 0.0));
    CHECK(close_to(original_z[1], -1.0));
    CHECK(close_to(original_affine_z[0], -1.0));
    CHECK(close_to(original_affine_z[1], 0.0));
    CHECK(prefos_verify_postsolve_infeasibility_certificate(
              presolver, reduced_y, reduced_z, reduced_affine_z, 1e-10,
              &verification) == PREFOS_STATUS_OK);
    CHECK(verification.passed);
    CHECK(close_to(verification.original.certificate_value, -1.0));
    prefos_free_presolver(presolver);
    return 0;
}

static int test_generated_affine_bound_certificate(void)
{
    double a_values[] = {1.0, -1.0, 1.0, 1.0, 1.0};
    int a_columns[] = {0, 2, 1, 2, 3};
    int a_rows[] = {0, 2, 3, 5};
    double lower[] = {0.0, 0.0, -INFINITY};
    double upper[] = {0.0, 0.0, -1.0};
    int q_rows[5], r_rows[1];
    double c[] = {0.0, 0.0, 0.0, 0.0};
    int box_indices[] = {2, 3};
    double box_lower[] = {-INFINITY, 0.0};
    double box_upper[] = {INFINITY, 1.0};
    int cone_indices[] = {0, 1};
    PreFOSConeBlock cone = {
        PREFOS_CONE_SECOND_ORDER, 2, 0, cone_indices, 0.0};
    double reduced_y[] = {1.0};
    double reduced_z[] = {-1.0, -1.0};
    double reduced_affine_z[] = {0.0, 0.0};
    double original_y[3], original_z[4];
    PreFOSProblemData problem;
    PreFOSSettings settings = certificate_settings();
    PreFOSPresolver *presolver = NULL;
    const PreFOSPresolvedProblem *reduced;
    PreFOSPostsolveInfeasibilityCertificateVerification verification;

    settings.cone_propagation = 1;
    settings.max_cone_propagation_rounds = 2;
    settings.affine_cone_coordinate_aggregation = 1;
    memset(&problem, 0, sizeof(problem));
    problem.n = 4;
    problem.A =
        (PreFOSCsrMatrix){3, 4, 5, a_values, a_columns, a_rows};
    problem.constraint_lower = lower;
    problem.constraint_upper = upper;
    initialize_empty_quadratic(&problem, 4, q_rows, r_rows);
    problem.c = c;
    problem.n_box = 2;
    problem.box_indices = box_indices;
    problem.box_lower = box_lower;
    problem.box_upper = box_upper;
    problem.n_cones = 1;
    problem.cones = &cone;

    CHECK(prefos_create_presolver(&problem, &settings, &presolver) ==
          PREFOS_STATUS_OK);
    CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_REDUCED);
    reduced = prefos_get_reduced_problem(presolver);
    CHECK(reduced != NULL && reduced->n == 2 && reduced->A.rows == 1);
    CHECK(reduced->n_cones == 0 && reduced->n_affine_cones == 1);
    CHECK(close_to(reduced->box_lower[0], 0.0));
    CHECK(prefos_get_stats(presolver)->generated_affine_cone_blocks == 1);
    CHECK(prefos_get_stats(presolver)->materialized_affine_cone_box_bounds == 1);
    CHECK(prefos_postsolve_infeasibility_certificate(
              presolver, reduced_y, reduced_z, reduced_affine_z, 1e-10,
              original_y, original_z, NULL) == PREFOS_STATUS_OK);
    CHECK(close_to(original_y[0], 1.0));
    CHECK(close_to(original_y[1], 0.0));
    CHECK(close_to(original_y[2], 1.0));
    CHECK(close_to(original_z[0], -1.0));
    CHECK(close_to(original_z[1], 0.0));
    CHECK(close_to(original_z[2], 0.0));
    CHECK(close_to(original_z[3], -1.0));
    CHECK(prefos_verify_postsolve_infeasibility_certificate(
              presolver, reduced_y, reduced_z, reduced_affine_z, 1e-10,
              &verification) == PREFOS_STATUS_OK);
    CHECK(verification.passed);
    CHECK(close_to(verification.original.certificate_value, -1.0));
    prefos_free_presolver(presolver);
    return 0;
}

static int test_affine_aggregation_certificate(void)
{
    double a_values[] = {1.0, -1.0, 1.0, -1.0, 1.0};
    int a_columns[] = {0, 2, 1, 3, 2};
    int a_rows[] = {0, 2, 4, 5};
    double lower[] = {0.0, 0.0, -INFINITY};
    double upper[] = {0.0, 0.0, -1.0};
    int q_rows[5], r_rows[1];
    double c[] = {0.0, 0.0, 0.0, 0.0};
    int box_indices[] = {2, 3};
    double box_lower[] = {-INFINITY, -INFINITY};
    double box_upper[] = {INFINITY, INFINITY};
    int cone_indices[] = {0, 1};
    PreFOSConeBlock cone = {
        PREFOS_CONE_SECOND_ORDER, 2, 0, cone_indices, 0.0};
    double reduced_z[] = {1.0, 0.0};
    double reduced_affine_z[] = {-1.0, 0.0};
    double original_y[3], original_z[4];
    PreFOSProblemData problem;
    PreFOSSettings settings = certificate_settings();
    PreFOSPresolver *presolver = NULL;
    const PreFOSPresolvedProblem *reduced;
    PreFOSPostsolveInfeasibilityCertificateVerification verification;

    settings.affine_cone_coordinate_aggregation = 1;
    memset(&problem, 0, sizeof(problem));
    problem.n = 4;
    problem.A =
        (PreFOSCsrMatrix){3, 4, 5, a_values, a_columns, a_rows};
    problem.constraint_lower = lower;
    problem.constraint_upper = upper;
    initialize_empty_quadratic(&problem, 4, q_rows, r_rows);
    problem.c = c;
    problem.n_box = 2;
    problem.box_indices = box_indices;
    problem.box_lower = box_lower;
    problem.box_upper = box_upper;
    problem.n_cones = 1;
    problem.cones = &cone;
    CHECK(prefos_create_presolver(&problem, &settings, &presolver) ==
          PREFOS_STATUS_OK);
    CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_REDUCED);
    reduced = prefos_get_reduced_problem(presolver);
    CHECK(reduced && reduced->n == 2 && reduced->A.rows == 0);
    CHECK(reduced->n_cones == 0 && reduced->n_affine_cones == 1);
    CHECK(prefos_postsolve_infeasibility_certificate(
              presolver, NULL, reduced_z, reduced_affine_z, 1e-10,
              original_y, original_z, NULL) == PREFOS_STATUS_OK);
    CHECK(close_to(original_y[0], 1.0));
    CHECK(close_to(original_y[2], 1.0));
    CHECK(close_to(original_z[0], -1.0));
    CHECK(close_to(original_z[2], 0.0));
    CHECK(prefos_verify_postsolve_infeasibility_certificate(
              presolver, NULL, reduced_z, reduced_affine_z, 1e-10,
              &verification) == PREFOS_STATUS_OK);
    CHECK(verification.passed);
    prefos_free_presolver(presolver);
    return 0;
}

static int test_substitution_certificate(void)
{
    double a_values[] = {1.0, -1.0, 1.0, 1.0, 1.0, 1.0};
    int a_columns[] = {0, 1, 0, 2, 1, 2};
    int a_rows[] = {0, 2, 4, 6};
    double lower[] = {0.0, -INFINITY, 1.0};
    double upper[] = {0.0, 0.0, INFINITY};
    int q_rows[4], r_rows[1];
    double c[] = {0.0, 0.0, 0.0};
    int box_indices[] = {0, 1, 2};
    double box_lower[] = {-INFINITY, -INFINITY, -INFINITY};
    double box_upper[] = {INFINITY, INFINITY, INFINITY};
    PreFOSProblemData problem;
    PreFOSSettings settings = certificate_settings();
    PreFOSPresolver *presolver = NULL;
    const PreFOSPresolvedProblem *reduced;
    const PreFOSStats *stats;
    double *reduced_y = NULL, *reduced_z = NULL;
    PreFOSPostsolveInfeasibilityCertificateVerification verification;
    size_t row;

    settings.free_column_substitution = 1;
    settings.max_aggregation_terms = 2;
    settings.max_aggregation_rounds = 1;
    memset(&problem, 0, sizeof(problem));
    problem.n = 3;
    problem.A =
        (PreFOSCsrMatrix){3, 3, 6, a_values, a_columns, a_rows};
    problem.constraint_lower = lower;
    problem.constraint_upper = upper;
    initialize_empty_quadratic(&problem, 3, q_rows, r_rows);
    problem.c = c;
    problem.n_box = 3;
    problem.box_indices = box_indices;
    problem.box_lower = box_lower;
    problem.box_upper = box_upper;
    CHECK(prefos_create_presolver(&problem, &settings, &presolver) ==
          PREFOS_STATUS_OK);
    CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_REDUCED);
    reduced = prefos_get_reduced_problem(presolver);
    stats = prefos_get_stats(presolver);
    CHECK(reduced != NULL && stats != NULL);
    CHECK(stats->substituted_free_variables > 0);
    CHECK(reduced->A.rows == 2);
    reduced_y =
        (double *) calloc(reduced->A.rows, sizeof(double));
    reduced_z = (double *) calloc(reduced->n, sizeof(double));
    CHECK(reduced_y != NULL && reduced_z != NULL);
    for (row = 0; row < reduced->A.rows; ++row)
    {
        if (isfinite(reduced->constraint_upper[row]) &&
            !isfinite(reduced->constraint_lower[row]))
            reduced_y[row] = 1.0;
        else if (isfinite(reduced->constraint_lower[row]) &&
                 !isfinite(reduced->constraint_upper[row]))
            reduced_y[row] = -1.0;
        else
            CHECK(0);
    }
    CHECK(prefos_verify_postsolve_infeasibility_certificate(
              presolver, reduced_y, reduced_z, NULL, 1e-10,
              &verification) == PREFOS_STATUS_OK);
    CHECK(verification.passed);
    CHECK(close_to(verification.original.certificate_value, -1.0));
    free(reduced_y);
    free(reduced_z);
    prefos_free_presolver(presolver);
    return 0;
}

static int test_residual_row_certificate(void)
{
    double a_values[] = {1.0, 1.0, 1.0, 1.0};
    int a_columns[] = {0, 1, 1, 2};
    int a_rows[] = {0, 2, 4};
    double lower[] = {0.0, 2.0};
    double upper[] = {0.0, INFINITY};
    int q_rows[4], r_rows[1];
    double c[] = {0.0, 0.0, 0.0};
    int box_indices[] = {0, 1, 2};
    double box_lower[] = {0.0, -INFINITY, 0.0};
    double box_upper[] = {1.0, INFINITY, 1.0};
    double reduced_y[] = {-1.0};
    double reduced_z[] = {1.0, 1.0};
    double original_y[2], original_z[3];
    PreFOSProblemData problem;
    PreFOSSettings settings = certificate_settings();
    PreFOSPresolver *presolver = NULL;
    const PreFOSPresolvedProblem *reduced;
    PreFOSPostsolveInfeasibilityCertificateVerification verification;

    settings.singleton_column_reduction = 1;
    memset(&problem, 0, sizeof(problem));
    problem.n = 3;
    problem.A =
        (PreFOSCsrMatrix){2, 3, 4, a_values, a_columns, a_rows};
    problem.constraint_lower = lower;
    problem.constraint_upper = upper;
    initialize_empty_quadratic(&problem, 3, q_rows, r_rows);
    problem.c = c;
    problem.n_box = 3;
    problem.box_indices = box_indices;
    problem.box_lower = box_lower;
    problem.box_upper = box_upper;

    CHECK(prefos_create_presolver(&problem, &settings, &presolver) ==
          PREFOS_STATUS_OK);
    CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_REDUCED);
    reduced = prefos_get_reduced_problem(presolver);
    CHECK(reduced != NULL && reduced->n == 2 && reduced->A.rows == 1);
    CHECK(prefos_get_stats(presolver)->residual_row_substitutions == 1);
    CHECK(prefos_postsolve_infeasibility_certificate(
              presolver, reduced_y, reduced_z, NULL, 1e-10,
              original_y, original_z, NULL) == PREFOS_STATUS_OK);
    CHECK(close_to(original_y[0], 1.0));
    CHECK(close_to(original_y[1], -1.0));
    CHECK(close_to(original_z[0], -1.0));
    CHECK(close_to(original_z[1], 0.0));
    CHECK(close_to(original_z[2], 1.0));
    CHECK(prefos_verify_postsolve_infeasibility_certificate(
              presolver, reduced_y, reduced_z, NULL, 1e-10,
              &verification) == PREFOS_STATUS_OK);
    CHECK(verification.passed);
    CHECK(close_to(verification.original.certificate_value, -1.0));
    prefos_free_presolver(presolver);
    return 0;
}

static int test_fixed_column_certificate(void)
{
    double a_values[] = {1.0, 1.0};
    int a_columns[] = {0, 1};
    int a_rows[] = {0, 2};
    double lower[] = {-INFINITY};
    double upper[] = {0.0};
    int q_rows[3], r_rows[1];
    double c[] = {0.0, 0.0};
    int box_indices[] = {0, 1};
    double box_lower[] = {2.0, -INFINITY};
    double box_upper[] = {2.0, INFINITY};
    double g_values[] = {1.0};
    int g_columns[] = {1};
    int g_rows[] = {0, 1};
    double h[] = {-1.0};
    PreFOSAffineConeBlock affine = {
        PREFOS_CONE_NONNEGATIVE, 1, 0, 0.0};
    double reduced_z[] = {1.0};
    double reduced_affine_z[] = {-1.0};
    double original_y[1], original_z[2], original_affine_z[1];
    PreFOSProblemData problem;
    PreFOSSettings settings = certificate_settings();
    PreFOSPresolver *presolver = NULL;
    PreFOSPostsolveInfeasibilityCertificateVerification verification;

    settings.fix_close_box_bounds = 1;
    memset(&problem, 0, sizeof(problem));
    problem.n = 2;
    problem.A =
        (PreFOSCsrMatrix){1, 2, 2, a_values, a_columns, a_rows};
    problem.constraint_lower = lower;
    problem.constraint_upper = upper;
    initialize_empty_quadratic(&problem, 2, q_rows, r_rows);
    problem.c = c;
    problem.n_box = 2;
    problem.box_indices = box_indices;
    problem.box_lower = box_lower;
    problem.box_upper = box_upper;
    problem.affine_cone_matrix =
        (PreFOSCsrMatrix){1, 2, 1, g_values, g_columns, g_rows};
    problem.affine_cone_offset = h;
    problem.n_affine_cones = 1;
    problem.affine_cones = &affine;
    CHECK(prefos_create_presolver(&problem, &settings, &presolver) ==
          PREFOS_STATUS_OK);
    CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_REDUCED);
    CHECK(prefos_get_reduced_problem(presolver)->n == 1);
    CHECK(prefos_get_reduced_problem(presolver)->A.rows == 0);
    CHECK(prefos_postsolve_infeasibility_certificate(
              presolver, NULL, reduced_z, reduced_affine_z, 1e-10,
              original_y, original_z, original_affine_z) ==
          PREFOS_STATUS_OK);
    CHECK(close_to(original_y[0], 1.0));
    CHECK(close_to(original_z[0], -1.0));
    CHECK(close_to(original_z[1], 0.0));
    CHECK(prefos_verify_postsolve_infeasibility_certificate(
              presolver, NULL, reduced_z, reduced_affine_z, 1e-10,
              &verification) == PREFOS_STATUS_OK);
    CHECK(verification.passed);
    CHECK(close_to(verification.original.certificate_value, -3.0));
    prefos_free_presolver(presolver);
    return 0;
}

static int test_parallel_row_certificate(void)
{
    double a_values[] = {1.0, 1.0, 2.0, 2.0};
    int a_columns[] = {0, 1, 0, 1};
    int a_rows[] = {0, 2, 4};
    double lower[] = {-INFINITY, -INFINITY};
    double upper[] = {4.0, 6.0};
    int q_rows[3], r_rows[1];
    double c[] = {0.0, 0.0};
    int box_indices[] = {0, 1};
    double box_lower[] = {-INFINITY, -INFINITY};
    double box_upper[] = {INFINITY, INFINITY};
    double g_values[] = {1.0, 1.0};
    int g_columns[] = {0, 1};
    int g_rows[] = {0, 2};
    double h[] = {-4.0};
    PreFOSAffineConeBlock affine = {
        PREFOS_CONE_NONNEGATIVE, 1, 0, 0.0};
    double reduced_y[] = {1.0};
    double reduced_z[] = {0.0, 0.0};
    double reduced_affine_z[] = {-1.0};
    double original_y[2], original_z[2], original_affine_z[1];
    PreFOSProblemData problem;
    PreFOSSettings settings = certificate_settings();
    PreFOSPresolver *presolver = NULL;
    const PreFOSPresolvedProblem *reduced;
    PreFOSPostsolveInfeasibilityCertificateVerification verification;

    settings.remove_redundant_rows = 1;
    memset(&problem, 0, sizeof(problem));
    problem.n = 2;
    problem.A =
        (PreFOSCsrMatrix){2, 2, 4, a_values, a_columns, a_rows};
    problem.constraint_lower = lower;
    problem.constraint_upper = upper;
    initialize_empty_quadratic(&problem, 2, q_rows, r_rows);
    problem.c = c;
    problem.n_box = 2;
    problem.box_indices = box_indices;
    problem.box_lower = box_lower;
    problem.box_upper = box_upper;
    problem.affine_cone_matrix =
        (PreFOSCsrMatrix){1, 2, 2, g_values, g_columns, g_rows};
    problem.affine_cone_offset = h;
    problem.n_affine_cones = 1;
    problem.affine_cones = &affine;
    CHECK(prefos_create_presolver(&problem, &settings, &presolver) ==
          PREFOS_STATUS_OK);
    CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_REDUCED);
    reduced = prefos_get_reduced_problem(presolver);
    CHECK(reduced != NULL && reduced->A.rows == 1);
    CHECK(close_to(reduced->constraint_upper[0], 3.0));
    CHECK(prefos_postsolve_infeasibility_certificate(
              presolver, reduced_y, reduced_z, reduced_affine_z, 1e-10,
              original_y, original_z, original_affine_z) ==
          PREFOS_STATUS_OK);
    CHECK(close_to(original_y[0], 0.0));
    CHECK(close_to(original_y[1], 0.5));
    CHECK(prefos_verify_postsolve_infeasibility_certificate(
              presolver, reduced_y, reduced_z, reduced_affine_z, 1e-10,
              &verification) == PREFOS_STATUS_OK);
    CHECK(verification.passed);
    CHECK(close_to(verification.original.certificate_value, -1.0));
    prefos_free_presolver(presolver);
    return 0;
}

static int test_parallel_column_certificate(void)
{
    double a_values[] = {1.0, 1.0, 1.0};
    int a_columns[] = {0, 1, 2};
    int a_rows[] = {0, 3};
    double lower[] = {3.0};
    double upper[] = {INFINITY};
    int q_rows[4], r_rows[1];
    double c[] = {0.0, 0.0, 0.0};
    int box_indices[] = {0, 1, 2};
    double box_lower[] = {0.0, 0.0, 0.0};
    double box_upper[] = {2.0, 2.0, 1.0};
    double g_values[] = {1.0};
    int g_columns[] = {2};
    int g_rows[] = {0, 1};
    double h[] = {10.0};
    PreFOSAffineConeBlock affine = {
        PREFOS_CONE_NONNEGATIVE, 1, 0, 0.0};
    double reduced_y[] = {-1.0};
    double reduced_z[] = {1.0, 1.0};
    double reduced_affine_z[] = {0.0};
    double original_y[1], original_z[3], original_affine_z[1];
    PreFOSProblemData problem;
    PreFOSSettings settings = certificate_settings();
    PreFOSPresolver *presolver = NULL;
    const PreFOSPresolvedProblem *reduced;
    PreFOSPostsolveInfeasibilityCertificateVerification verification;

    settings.parallel_column_reduction = 1;
    memset(&problem, 0, sizeof(problem));
    problem.n = 3;
    problem.A =
        (PreFOSCsrMatrix){1, 3, 3, a_values, a_columns, a_rows};
    problem.constraint_lower = lower;
    problem.constraint_upper = upper;
    initialize_empty_quadratic(&problem, 3, q_rows, r_rows);
    problem.c = c;
    problem.n_box = 3;
    problem.box_indices = box_indices;
    problem.box_lower = box_lower;
    problem.box_upper = box_upper;
    problem.affine_cone_matrix =
        (PreFOSCsrMatrix){1, 3, 1, g_values, g_columns, g_rows};
    problem.affine_cone_offset = h;
    problem.n_affine_cones = 1;
    problem.affine_cones = &affine;
    CHECK(prefos_create_presolver(&problem, &settings, &presolver) ==
          PREFOS_STATUS_OK);
    CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_REDUCED);
    reduced = prefos_get_reduced_problem(presolver);
    CHECK(reduced != NULL && reduced->n == 2 && reduced->A.rows == 1);
    CHECK(prefos_get_stats(presolver)->merged_parallel_columns == 1);
    CHECK(close_to(reduced->box_lower[0], 0.0));
    CHECK(close_to(reduced->box_upper[0], 4.0));
    CHECK(prefos_postsolve_infeasibility_certificate(
              presolver, reduced_y, reduced_z, reduced_affine_z, 1e-10,
              original_y, original_z, original_affine_z) ==
          PREFOS_STATUS_OK);
    CHECK(close_to(original_z[0], 1.0));
    CHECK(close_to(original_z[1], 1.0));
    CHECK(close_to(original_z[2], 1.0));
    CHECK(prefos_verify_postsolve_infeasibility_certificate(
              presolver, reduced_y, reduced_z, reduced_affine_z, 1e-10,
              &verification) == PREFOS_STATUS_OK);
    CHECK(!verification.passed);
    CHECK(verification.reduced.stationarity_violation <= 1e-10);
    CHECK(verification.original.stationarity_violation <= 1e-10);
    CHECK(verification.reduced.domain_dual_violation <= 1e-10);
    CHECK(verification.original.domain_dual_violation <= 1e-10);
    CHECK(close_to(verification.original.certificate_value, 2.0));
    CHECK(verification.certificate_value_absolute_error <= 1e-10);
    prefos_free_presolver(presolver);
    return 0;
}

static int test_bounded_doubleton_certificate(void)
{
    double a_values[] = {1.0, 1.0, 1.0, 1.0};
    int a_columns[] = {0, 1, 0, 2};
    int a_rows[] = {0, 2, 4};
    double lower[] = {5.0, 4.0};
    double upper[] = {5.0, INFINITY};
    int q_rows[4], r_rows[1];
    double c[] = {0.0, 0.0, 0.0};
    int box_indices[] = {0, 1, 2};
    double box_lower[] = {0.0, 0.0, 0.0};
    double box_upper[] = {10.0, 10.0, 1.0};
    double g_values[] = {1.0};
    int g_columns[] = {2};
    int g_rows[] = {0, 1};
    double h[] = {10.0};
    PreFOSAffineConeBlock affine = {
        PREFOS_CONE_NONNEGATIVE, 1, 0, 0.0};
    PreFOSProblemData problem;
    PreFOSSettings settings = certificate_settings();
    PreFOSPresolver *presolver = NULL;
    const PreFOSPresolvedProblem *reduced;
    double reduced_y[] = {-1.0};
    double reduced_z[] = {1.0, 1.0};
    double reduced_affine_z[] = {0.0};
    PreFOSPostsolveInfeasibilityCertificateVerification verification;

    settings.bounded_doubleton_substitution = 1;
    memset(&problem, 0, sizeof(problem));
    problem.n = 3;
    problem.A =
        (PreFOSCsrMatrix){2, 3, 4, a_values, a_columns, a_rows};
    problem.constraint_lower = lower;
    problem.constraint_upper = upper;
    initialize_empty_quadratic(&problem, 3, q_rows, r_rows);
    problem.c = c;
    problem.n_box = 3;
    problem.box_indices = box_indices;
    problem.box_lower = box_lower;
    problem.box_upper = box_upper;
    problem.affine_cone_matrix =
        (PreFOSCsrMatrix){1, 3, 1, g_values, g_columns, g_rows};
    problem.affine_cone_offset = h;
    problem.n_affine_cones = 1;
    problem.affine_cones = &affine;
    CHECK(prefos_create_presolver(&problem, &settings, &presolver) ==
          PREFOS_STATUS_OK);
    CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_REDUCED);
    reduced = prefos_get_reduced_problem(presolver);
    CHECK(reduced != NULL && reduced->n == 2 && reduced->A.rows == 1);
    CHECK(prefos_get_stats(presolver)->substituted_bounded_doubletons == 1);
    CHECK(prefos_verify_postsolve_infeasibility_certificate(
              presolver, reduced_y, reduced_z, reduced_affine_z, 1e-10,
              &verification) == PREFOS_STATUS_OK);
    CHECK(!verification.passed);
    CHECK(verification.reduced.stationarity_violation <= 1e-10);
    CHECK(verification.original.stationarity_violation <= 1e-10);
    CHECK(verification.reduced.domain_dual_violation <= 1e-10);
    CHECK(verification.original.domain_dual_violation <= 1e-10);
    CHECK(close_to(verification.original.certificate_value, 2.0));
    CHECK(verification.certificate_value_absolute_error <= 1e-10);
    prefos_free_presolver(presolver);
    return 0;
}

static int test_soc_collapse_certificate(void)
{
    double a_values[] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    int a_columns[] = {0, 1, 2, 3, 2, 3};
    int a_rows[] = {0, 1, 4, 6};
    double lower[] = {-INFINITY, -INFINITY, 1.0};
    double upper[] = {0.0, 0.0, INFINITY};
    int q_rows[5], r_rows[1];
    double c[] = {0.0, 0.0, 0.0, 0.0};
    int box_indices[] = {2, 3};
    double box_lower[] = {-INFINITY, -INFINITY};
    double box_upper[] = {INFINITY, INFINITY};
    int cone_indices[] = {0, 1};
    PreFOSConeBlock cone = {
        PREFOS_CONE_SECOND_ORDER, 2, 0, cone_indices, 0.0};
    double reduced_y[] = {0.0, 1.0, -1.0};
    double reduced_z[] = {0.0, 0.0};
    double original_y[3], original_z[4];
    PreFOSProblemData problem;
    PreFOSSettings settings = certificate_settings();
    PreFOSPresolver *presolver = NULL;
    const PreFOSPresolvedProblem *reduced;
    PreFOSPostsolveInfeasibilityCertificateVerification verification;

    settings.cone_propagation = 1;
    settings.max_cone_propagation_rounds = 2;
    memset(&problem, 0, sizeof(problem));
    problem.n = 4;
    problem.A =
        (PreFOSCsrMatrix){3, 4, 6, a_values, a_columns, a_rows};
    problem.constraint_lower = lower;
    problem.constraint_upper = upper;
    initialize_empty_quadratic(&problem, 4, q_rows, r_rows);
    problem.c = c;
    problem.n_box = 2;
    problem.box_indices = box_indices;
    problem.box_lower = box_lower;
    problem.box_upper = box_upper;
    problem.n_cones = 1;
    problem.cones = &cone;
    CHECK(prefos_create_presolver(&problem, &settings, &presolver) ==
          PREFOS_STATUS_OK);
    CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_REDUCED);
    reduced = prefos_get_reduced_problem(presolver);
    CHECK(reduced != NULL && reduced->n == 2 && reduced->A.rows == 3);
    CHECK(prefos_get_stats(presolver)->collapsed_cones == 1);
    CHECK(prefos_postsolve_infeasibility_certificate(
              presolver, reduced_y, reduced_z, NULL, 1e-10,
              original_y, original_z, NULL) == PREFOS_STATUS_OK);
    CHECK(close_to(original_y[0], 1.0));
    CHECK(close_to(original_z[0], -1.0));
    CHECK(close_to(original_z[1], -1.0));
    CHECK(prefos_verify_postsolve_infeasibility_certificate(
              presolver, reduced_y, reduced_z, NULL, 1e-10,
              &verification) == PREFOS_STATUS_OK);
    CHECK(verification.passed);
    CHECK(close_to(verification.original.certificate_value, -1.0));
    prefos_free_presolver(presolver);
    return 0;
}

static int test_rsoc_face_extended_certificate(void)
{
    double a_values[] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    int a_columns[] = {0, 2, 3, 4, 3, 4};
    int a_rows[] = {0, 1, 4, 6};
    double lower[] = {-INFINITY, -INFINITY, 1.0};
    double upper[] = {0.0, 0.0, INFINITY};
    int q_rows[6], r_rows[1];
    double c[] = {0.0, 0.0, 0.0, 0.0, 0.0};
    int box_indices[] = {3, 4};
    double box_lower[] = {-INFINITY, -INFINITY};
    double box_upper[] = {INFINITY, INFINITY};
    int cone_indices[] = {0, 1, 2};
    PreFOSConeBlock cone = {
        PREFOS_CONE_ROTATED_SECOND_ORDER, 3, 0, cone_indices, 0.0};
    double reduced_y[] = {0.0, 1.0, -1.0};
    double reduced_z[] = {0.0, 0.0, 0.0};
    PreFOSProblemData problem;
    PreFOSSettings settings = certificate_settings();
    PreFOSPresolver *presolver = NULL;
    const PreFOSPresolvedProblem *reduced;
    PreFOSPostsolveInfeasibilityCertificateVerification verification;

    settings.cone_propagation = 1;
    settings.max_cone_propagation_rounds = 2;
    settings.rsoc_face_reduction = 1;
    memset(&problem, 0, sizeof(problem));
    problem.n = 5;
    problem.A =
        (PreFOSCsrMatrix){3, 5, 6, a_values, a_columns, a_rows};
    problem.constraint_lower = lower;
    problem.constraint_upper = upper;
    initialize_empty_quadratic(&problem, 5, q_rows, r_rows);
    problem.c = c;
    problem.n_box = 2;
    problem.box_indices = box_indices;
    problem.box_lower = box_lower;
    problem.box_upper = box_upper;
    problem.n_cones = 1;
    problem.cones = &cone;
    CHECK(prefos_create_presolver(&problem, &settings, &presolver) ==
          PREFOS_STATUS_OK);
    CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_REDUCED);
    reduced = prefos_get_reduced_problem(presolver);
    CHECK(reduced != NULL && reduced->A.rows == 3);
    CHECK(prefos_get_stats(presolver)->reduced_rsoc_faces == 1);
    CHECK(prefos_verify_postsolve_infeasibility_certificate(
              presolver, reduced_y, reduced_z, NULL, 1e-10,
              &verification) ==
          PREFOS_STATUS_DUAL_RECOVERY_UNAVAILABLE);
    CHECK(prefos_verify_postsolve_extended_infeasibility_certificate(
              presolver, reduced_y, reduced_z, NULL, 1e-10,
              &verification) == PREFOS_STATUS_OK);
    CHECK(verification.passed);
    CHECK(close_to(verification.original.certificate_value, -1.0));
    prefos_free_presolver(presolver);
    return 0;
}

static int test_affine_rsoc_face_extended_certificate(void)
{
    double a_values[] = {1.0, 1.0};
    int a_columns[] = {0, 2};
    int a_rows[] = {0, 2};
    double lower[] = {-INFINITY};
    double upper[] = {-4.0};
    int q_rows[4], r_rows[1];
    double c[] = {0.0, 0.0, 0.0};
    int box_indices[] = {0, 1, 2};
    double box_lower[] = {-INFINITY, 0.0, 0.0};
    double box_upper[] = {INFINITY, 1.0, 1.0};
    double g_values[] = {1.0, 1.0};
    int g_columns[] = {0, 1};
    int g_rows[] = {0, 0, 0, 2};
    double h[] = {0.0, 1.0, 2.0};
    PreFOSAffineConeBlock affine = {
        PREFOS_CONE_ROTATED_SECOND_ORDER, 3, 0, 0.0};
    double reduced_y[] = {1.0};
    double reduced_z[] = {1.0, -1.0};
    double reduced_affine_z[] = {0.0, 0.0, 0.0};
    double original_y[1], original_z[3], original_affine_z[3];
    PreFOSProblemData problem;
    PreFOSSettings settings = certificate_settings();
    PreFOSPresolver *presolver = NULL;
    const PreFOSPresolvedProblem *reduced;
    PreFOSPostsolveInfeasibilityCertificateVerification verification;

    settings.free_column_substitution = 1;
    settings.rsoc_face_reduction = 1;
    memset(&problem, 0, sizeof(problem));
    problem.n = 3;
    problem.A =
        (PreFOSCsrMatrix){1, 3, 2, a_values, a_columns, a_rows};
    problem.constraint_lower = lower;
    problem.constraint_upper = upper;
    initialize_empty_quadratic(&problem, 3, q_rows, r_rows);
    problem.c = c;
    problem.n_box = 3;
    problem.box_indices = box_indices;
    problem.box_lower = box_lower;
    problem.box_upper = box_upper;
    problem.affine_cone_matrix =
        (PreFOSCsrMatrix){3, 3, 2, g_values, g_columns, g_rows};
    problem.affine_cone_offset = h;
    problem.n_affine_cones = 1;
    problem.affine_cones = &affine;

    CHECK(prefos_create_presolver(&problem, &settings, &presolver) ==
          PREFOS_STATUS_OK);
    CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_REDUCED);
    reduced = prefos_get_reduced_problem(presolver);
    CHECK(reduced != NULL && reduced->n == 2 && reduced->A.rows == 1);
    CHECK(reduced->affine_cone_matrix.rows == 3);
    CHECK(prefos_get_stats(presolver)->derived_affine_face_equalities == 1);
    CHECK(prefos_postsolve_infeasibility_certificate(
              presolver, reduced_y, reduced_z, reduced_affine_z, 1e-10,
              original_y, original_z, original_affine_z) ==
          PREFOS_STATUS_DUAL_RECOVERY_UNAVAILABLE);
    CHECK(prefos_postsolve_extended_infeasibility_certificate(
              presolver, reduced_y, reduced_z, reduced_affine_z, 1e-10,
              original_y, original_z, original_affine_z) ==
          PREFOS_STATUS_OK);
    CHECK(close_to(original_y[0], 1.0));
    CHECK(close_to(original_z[0], 0.0));
    CHECK(close_to(original_z[1], 1.0));
    CHECK(close_to(original_z[2], -1.0));
    CHECK(close_to(original_affine_z[0], 0.0));
    CHECK(close_to(original_affine_z[1], 0.0));
    CHECK(close_to(original_affine_z[2], -1.0));
    CHECK(prefos_verify_postsolve_extended_infeasibility_certificate(
              presolver, reduced_y, reduced_z, reduced_affine_z, 1e-10,
              &verification) == PREFOS_STATUS_OK);
    CHECK(verification.passed);
    CHECK(close_to(verification.reduced.certificate_value, -1.0));
    CHECK(close_to(verification.original.certificate_value, -1.0));
    prefos_free_presolver(presolver);
    return 0;
}

static int test_generated_affine_rsoc_face_extended_certificate(void)
{
    double a_values[] = {1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0};
    int a_columns[] = {0, 1, 2, 3, 3, 4, 5};
    int a_rows[] = {0, 1, 2, 4, 7};
    double lower[] = {0.0, 1.0, 0.0, -INFINITY};
    double upper[] = {0.0, 1.0, 0.0, -3.0};
    int q_rows[7], r_rows[1];
    double c[] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    int box_indices[] = {3, 4, 5};
    double box_lower[] = {-INFINITY, 0.0, 0.0};
    double box_upper[] = {INFINITY, 1.0, 1.0};
    int cone_indices[] = {0, 1, 2};
    PreFOSConeBlock cone = {
        PREFOS_CONE_ROTATED_SECOND_ORDER, 3, 0, cone_indices, 0.0};
    double reduced_y[] = {1.0};
    double reduced_z[] = {-1.0, -1.0};
    double reduced_affine_z[] = {0.0, 0.0, 0.0};
    double original_y[4], original_z[6];
    PreFOSProblemData problem;
    PreFOSSettings settings = certificate_settings();
    PreFOSPresolver *presolver = NULL;
    const PreFOSPresolvedProblem *reduced;
    PreFOSPostsolveInfeasibilityCertificateVerification verification;

    settings.free_column_substitution = 1;
    settings.rsoc_face_reduction = 1;
    settings.affine_cone_coordinate_aggregation = 1;
    memset(&problem, 0, sizeof(problem));
    problem.n = 6;
    problem.A =
        (PreFOSCsrMatrix){4, 6, 7, a_values, a_columns, a_rows};
    problem.constraint_lower = lower;
    problem.constraint_upper = upper;
    initialize_empty_quadratic(&problem, 6, q_rows, r_rows);
    problem.c = c;
    problem.n_box = 3;
    problem.box_indices = box_indices;
    problem.box_lower = box_lower;
    problem.box_upper = box_upper;
    problem.n_cones = 1;
    problem.cones = &cone;

    CHECK(prefos_create_presolver(&problem, &settings, &presolver) ==
          PREFOS_STATUS_OK);
    CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_REDUCED);
    reduced = prefos_get_reduced_problem(presolver);
    CHECK(reduced != NULL && reduced->n == 2 && reduced->A.rows == 1);
    CHECK(reduced->n_cones == 0 && reduced->n_affine_cones == 1);
    CHECK(reduced->affine_cone_matrix.rows == 3);
    CHECK(prefos_get_stats(presolver)->generated_affine_cone_blocks == 1);
    CHECK(prefos_get_stats(presolver)->fixed_affine_face_variables == 1);
    CHECK(prefos_postsolve_infeasibility_certificate(
              presolver, reduced_y, reduced_z, reduced_affine_z, 1e-10,
              original_y, original_z, NULL) ==
          PREFOS_STATUS_DUAL_RECOVERY_UNAVAILABLE);
    CHECK(prefos_postsolve_extended_infeasibility_certificate(
              presolver, reduced_y, reduced_z, reduced_affine_z, 1e-10,
              original_y, original_z, NULL) == PREFOS_STATUS_OK);
    CHECK(close_to(original_y[0], 0.0));
    CHECK(close_to(original_y[1], 0.0));
    CHECK(close_to(original_y[2], 1.0));
    CHECK(close_to(original_y[3], 1.0));
    CHECK(close_to(original_z[0], 0.0));
    CHECK(close_to(original_z[1], 0.0));
    CHECK(close_to(original_z[2], -1.0));
    CHECK(close_to(original_z[3], 0.0));
    CHECK(close_to(original_z[4], -1.0));
    CHECK(close_to(original_z[5], -1.0));
    CHECK(prefos_verify_postsolve_extended_infeasibility_certificate(
              presolver, reduced_y, reduced_z, reduced_affine_z, 1e-10,
              &verification) == PREFOS_STATUS_OK);
    CHECK(verification.passed);
    CHECK(close_to(verification.reduced.certificate_value, -3.0));
    CHECK(close_to(verification.original.certificate_value, -3.0));
    prefos_free_presolver(presolver);
    return 0;
}

static int test_unbounded_ray_identity(void)
{
    int a_rows[] = {0};
    double q_values[] = {2.0};
    int q_columns[] = {0};
    int q_rows[] = {0, 1, 1};
    double r_values[] = {1.0};
    int r_columns[] = {0};
    int r_rows[] = {0, 1};
    double d[] = {3.0};
    double c[] = {0.0, -1.0};
    int box_indices[] = {0, 1};
    double box_lower[] = {-INFINITY, -INFINITY};
    double box_upper[] = {INFINITY, INFINITY};
    double ray[] = {0.0, 1.0};
    PreFOSProblemData problem;
    PreFOSSettings settings = certificate_settings();
    PreFOSPresolver *presolver = NULL;
    PreFOSPostsolveUnboundedRayVerification verification;

    memset(&problem, 0, sizeof(problem));
    problem.n = 2;
    problem.A =
        (PreFOSCsrMatrix){0, 2, 0, NULL, NULL, a_rows};
    problem.Q =
        (PreFOSCsrMatrix){2, 2, 1, q_values, q_columns, q_rows};
    problem.q_storage = PREFOS_Q_UPPER_TRIANGULAR;
    problem.R =
        (PreFOSCsrMatrix){1, 2, 1, r_values, r_columns, r_rows};
    problem.D = d;
    problem.c = c;
    problem.n_box = 2;
    problem.box_indices = box_indices;
    problem.box_lower = box_lower;
    problem.box_upper = box_upper;
    CHECK(prefos_create_presolver(&problem, &settings, &presolver) ==
          PREFOS_STATUS_OK);
    CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_OK);
    CHECK(prefos_verify_postsolve_unbounded_ray(
              presolver, ray, 1e-10, &verification) == PREFOS_STATUS_OK);
    CHECK(verification.passed);
    CHECK(close_to(verification.original.objective_direction, -1.0));
    ray[0] = 1.0;
    CHECK(prefos_verify_postsolve_unbounded_ray(
              presolver, ray, 1e-10, &verification) == PREFOS_STATUS_OK);
    CHECK(!verification.passed);
    prefos_free_presolver(presolver);
    return 0;
}

static int test_substitution_unbounded_ray(void)
{
    double a_values[] = {1.0, -1.0};
    int a_columns[] = {0, 1};
    int a_rows[] = {0, 2};
    double side[] = {0.0};
    int q_rows[3], r_rows[1];
    double c[] = {-1.0, 0.0};
    int box_indices[] = {0, 1};
    double box_lower[] = {-INFINITY, -INFINITY};
    double box_upper[] = {INFINITY, INFINITY};
    PreFOSProblemData problem;
    PreFOSSettings settings = certificate_settings();
    PreFOSPresolver *presolver = NULL;
    const PreFOSPresolvedProblem *reduced;
    double reduced_ray[1], original_ray[2];
    PreFOSPostsolveUnboundedRayVerification verification;

    settings.free_column_substitution = 1;
    settings.max_aggregation_terms = 1;
    settings.max_aggregation_rounds = 1;
    memset(&problem, 0, sizeof(problem));
    problem.n = 2;
    problem.A =
        (PreFOSCsrMatrix){1, 2, 2, a_values, a_columns, a_rows};
    problem.constraint_lower = side;
    problem.constraint_upper = side;
    initialize_empty_quadratic(&problem, 2, q_rows, r_rows);
    problem.c = c;
    problem.n_box = 2;
    problem.box_indices = box_indices;
    problem.box_lower = box_lower;
    problem.box_upper = box_upper;
    CHECK(prefos_create_presolver(&problem, &settings, &presolver) ==
          PREFOS_STATUS_OK);
    CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_REDUCED);
    reduced = prefos_get_reduced_problem(presolver);
    CHECK(reduced != NULL && reduced->n == 1 && reduced->A.rows == 0);
    reduced_ray[0] = reduced->c[0] < 0.0 ? 1.0 : -1.0;
    CHECK(prefos_postsolve_unbounded_ray(
              presolver, reduced_ray, 1e-10, original_ray) ==
          PREFOS_STATUS_OK);
    CHECK(close_to(original_ray[0], original_ray[1]));
    CHECK(prefos_verify_postsolve_unbounded_ray(
              presolver, reduced_ray, 1e-10, &verification) ==
          PREFOS_STATUS_OK);
    CHECK(verification.passed);
    CHECK(verification.objective_direction_absolute_error <= 1e-10);
    prefos_free_presolver(presolver);
    return 0;
}

static int test_parallel_column_unbounded_ray(void)
{
    double a_values[] = {1.0, 1.0};
    int a_columns[] = {0, 1};
    int a_rows[] = {0, 2};
    double lower[] = {0.0};
    double upper[] = {INFINITY};
    int q_rows[3], r_rows[1];
    double c[] = {-1.0, -1.0};
    int box_indices[] = {0, 1};
    double box_lower[] = {0.0, 0.0};
    double box_upper[] = {INFINITY, INFINITY};
    double reduced_ray[] = {1.0};
    double original_ray[2];
    PreFOSProblemData problem;
    PreFOSSettings settings = certificate_settings();
    PreFOSPresolver *presolver = NULL;
    PreFOSPostsolveUnboundedRayVerification verification;

    settings.parallel_column_reduction = 1;
    memset(&problem, 0, sizeof(problem));
    problem.n = 2;
    problem.A =
        (PreFOSCsrMatrix){1, 2, 2, a_values, a_columns, a_rows};
    problem.constraint_lower = lower;
    problem.constraint_upper = upper;
    initialize_empty_quadratic(&problem, 2, q_rows, r_rows);
    problem.c = c;
    problem.n_box = 2;
    problem.box_indices = box_indices;
    problem.box_lower = box_lower;
    problem.box_upper = box_upper;
    CHECK(prefos_create_presolver(&problem, &settings, &presolver) ==
          PREFOS_STATUS_OK);
    CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_REDUCED);
    CHECK(prefos_get_reduced_problem(presolver)->n == 1);
    CHECK(prefos_postsolve_unbounded_ray(
              presolver, reduced_ray, 1e-10, original_ray) ==
          PREFOS_STATUS_OK);
    CHECK(original_ray[0] >= 0.0 && original_ray[1] >= 0.0);
    CHECK(close_to(original_ray[0] + original_ray[1], 1.0));
    CHECK(prefos_verify_postsolve_unbounded_ray(
              presolver, reduced_ray, 1e-10, &verification) ==
          PREFOS_STATUS_OK);
    CHECK(verification.passed);
    CHECK(close_to(verification.original.objective_direction, -1.0));
    prefos_free_presolver(presolver);
    return 0;
}

static int test_infinite_fixed_column_unbounded_ray(void)
{
    double a_values[] = {1.0, -1.0};
    int a_columns[] = {0, 1};
    int a_rows[] = {0, 2};
    double lower[] = {0.0};
    double upper[] = {INFINITY};
    int q_rows[3], r_rows[1];
    double c[] = {0.0, -1.0};
    int box_indices[] = {0, 1};
    double box_lower[] = {-INFINITY, 0.0};
    double box_upper[] = {INFINITY, INFINITY};
    double reduced_ray[] = {1.0};
    double original_ray[2];
    PreFOSProblemData problem;
    PreFOSSettings settings = certificate_settings();
    PreFOSPresolver *presolver = NULL;
    PreFOSPostsolveUnboundedRayVerification verification;

    settings.remove_redundant_rows = 1;
    settings.dual_fixing = 1;
    memset(&problem, 0, sizeof(problem));
    problem.n = 2;
    problem.A =
        (PreFOSCsrMatrix){1, 2, 2, a_values, a_columns, a_rows};
    problem.constraint_lower = lower;
    problem.constraint_upper = upper;
    initialize_empty_quadratic(&problem, 2, q_rows, r_rows);
    problem.c = c;
    problem.n_box = 2;
    problem.box_indices = box_indices;
    problem.box_lower = box_lower;
    problem.box_upper = box_upper;
    CHECK(prefos_create_presolver(&problem, &settings, &presolver) ==
          PREFOS_STATUS_OK);
    CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_REDUCED);
    CHECK(prefos_get_reduced_problem(presolver)->n == 1);
    CHECK(prefos_get_reduced_problem(presolver)->A.rows == 0);
    CHECK(prefos_get_stats(presolver)->dual_fixed_columns == 1);
    CHECK(prefos_postsolve_unbounded_ray(
              presolver, reduced_ray, 1e-10, original_ray) ==
          PREFOS_STATUS_OK);
    CHECK(original_ray[0] >= original_ray[1]);
    CHECK(close_to(original_ray[1], 1.0));
    CHECK(prefos_verify_postsolve_unbounded_ray(
              presolver, reduced_ray, 1e-10, &verification) ==
          PREFOS_STATUS_OK);
    CHECK(verification.passed);
    prefos_free_presolver(presolver);
    return 0;
}

static int test_affine_aggregation_unbounded_ray(void)
{
    double a_values[] = {1.0, -1.0, 1.0, -1.0};
    int a_columns[] = {0, 2, 1, 3};
    int a_rows[] = {0, 2, 4};
    double sides[] = {0.0, 0.0};
    int q_rows[5], r_rows[1];
    double c[] = {0.0, 0.0, -1.0, 0.0};
    int box_indices[] = {2, 3};
    double box_lower[] = {-INFINITY, -INFINITY};
    double box_upper[] = {INFINITY, INFINITY};
    int cone_indices[] = {0, 1};
    PreFOSConeBlock cone = {
        PREFOS_CONE_SECOND_ORDER, 2, 0, cone_indices, 0.0};
    double reduced_ray[] = {1.0, 0.0};
    double original_ray[4];
    PreFOSProblemData problem;
    PreFOSSettings settings = certificate_settings();
    PreFOSPresolver *presolver = NULL;
    const PreFOSPresolvedProblem *reduced;
    PreFOSPostsolveUnboundedRayVerification verification;

    settings.affine_cone_coordinate_aggregation = 1;
    memset(&problem, 0, sizeof(problem));
    problem.n = 4;
    problem.A =
        (PreFOSCsrMatrix){2, 4, 4, a_values, a_columns, a_rows};
    problem.constraint_lower = sides;
    problem.constraint_upper = sides;
    initialize_empty_quadratic(&problem, 4, q_rows, r_rows);
    problem.c = c;
    problem.n_box = 2;
    problem.box_indices = box_indices;
    problem.box_lower = box_lower;
    problem.box_upper = box_upper;
    problem.n_cones = 1;
    problem.cones = &cone;
    CHECK(prefos_create_presolver(&problem, &settings, &presolver) ==
          PREFOS_STATUS_OK);
    CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_REDUCED);
    reduced = prefos_get_reduced_problem(presolver);
    CHECK(reduced && reduced->n == 2 && reduced->A.rows == 0);
    CHECK(reduced->n_cones == 0 && reduced->n_affine_cones == 1);
    CHECK(prefos_postsolve_unbounded_ray(
              presolver, reduced_ray, 1e-10, original_ray) ==
          PREFOS_STATUS_OK);
    CHECK(close_to(original_ray[0], 1.0));
    CHECK(close_to(original_ray[1], 0.0));
    CHECK(close_to(original_ray[2], 1.0));
    CHECK(close_to(original_ray[3], 0.0));
    CHECK(prefos_verify_postsolve_unbounded_ray(
              presolver, reduced_ray, 1e-10, &verification) ==
          PREFOS_STATUS_OK);
    CHECK(verification.passed);
    prefos_free_presolver(presolver);
    return 0;
}

static int test_direct_cone_unbounded_ray(
    PreFOSConeType type, size_t dimension, size_t matrix_order,
    double power_alpha, size_t ray_coordinate)
{
    int a_rows[] = {0};
    int *q_rows = NULL, *cone_indices = NULL;
    int r_rows[] = {0};
    double *c = NULL, *ray = NULL;
    PreFOSConeBlock cone;
    PreFOSProblemData problem;
    PreFOSSettings settings = certificate_settings();
    PreFOSPresolver *presolver = NULL;
    PreFOSPostsolveUnboundedRayVerification verification;
    size_t i;
    int result = 1;
    PreFOSStatus status;

    q_rows = (int *) calloc(dimension + 1, sizeof(int));
    cone_indices = (int *) malloc(dimension * sizeof(int));
    c = (double *) calloc(dimension, sizeof(double));
    ray = (double *) calloc(dimension, sizeof(double));
    if (!q_rows || !cone_indices || !c || !ray) goto cleanup;
    for (i = 0; i < dimension; ++i) cone_indices[i] = (int) i;
    c[ray_coordinate] = -1.0;
    ray[ray_coordinate] = 1.0;
    cone = (PreFOSConeBlock){
        type, dimension, matrix_order, cone_indices, power_alpha};

    memset(&problem, 0, sizeof(problem));
    problem.n = dimension;
    problem.A =
        (PreFOSCsrMatrix){0, dimension, 0, NULL, NULL, a_rows};
    initialize_empty_quadratic(&problem, dimension, q_rows, r_rows);
    problem.c = c;
    problem.n_cones = 1;
    problem.cones = &cone;
    if (prefos_create_presolver(&problem, &settings, &presolver) !=
        PREFOS_STATUS_OK)
        goto cleanup;
    status = prefos_run_presolve(presolver);
    if (status != PREFOS_STATUS_OK && status != PREFOS_STATUS_REDUCED)
        goto cleanup;
    if (prefos_verify_postsolve_unbounded_ray(
            presolver, ray, 1e-9, &verification) != PREFOS_STATUS_OK ||
        !verification.passed)
        goto cleanup;
    result = 0;

cleanup:
    prefos_free_presolver(presolver);
    free(q_rows);
    free(cone_indices);
    free(c);
    free(ray);
    return result;
}

static int test_affine_cone_unbounded_ray(
    PreFOSConeType type, size_t dimension, size_t matrix_order,
    double power_alpha, size_t ray_coordinate)
{
    int a_rows[] = {0};
    int *q_rows = NULL, *g_rows = NULL, *g_columns = NULL;
    int *box_indices = NULL;
    int r_rows[] = {0};
    double *c = NULL, *ray = NULL, *g_values = NULL;
    double *h = NULL, *box_lower = NULL, *box_upper = NULL;
    PreFOSAffineConeBlock affine = {
        type, dimension, matrix_order, power_alpha};
    PreFOSProblemData problem;
    PreFOSSettings settings = certificate_settings();
    PreFOSPresolver *presolver = NULL;
    PreFOSPostsolveUnboundedRayVerification verification;
    size_t i;
    int result = 1;
    PreFOSStatus status;

    q_rows = (int *) calloc(dimension + 1, sizeof(int));
    g_rows = (int *) malloc((dimension + 1) * sizeof(int));
    g_columns = (int *) malloc(dimension * sizeof(int));
    box_indices = (int *) malloc(dimension * sizeof(int));
    c = (double *) calloc(dimension, sizeof(double));
    ray = (double *) calloc(dimension, sizeof(double));
    g_values = (double *) malloc(dimension * sizeof(double));
    h = (double *) calloc(dimension, sizeof(double));
    box_lower = (double *) malloc(dimension * sizeof(double));
    box_upper = (double *) malloc(dimension * sizeof(double));
    if (!q_rows || !g_rows || !g_columns || !box_indices || !c || !ray ||
        !g_values || !h || !box_lower || !box_upper)
        goto cleanup;
    for (i = 0; i < dimension; ++i)
    {
        g_rows[i] = (int) i;
        g_columns[i] = (int) i;
        g_values[i] = 1.0;
        box_indices[i] = (int) i;
        box_lower[i] = -INFINITY;
        box_upper[i] = INFINITY;
    }
    g_rows[dimension] = (int) dimension;
    c[ray_coordinate] = -1.0;
    ray[ray_coordinate] = 1.0;

    memset(&problem, 0, sizeof(problem));
    problem.n = dimension;
    problem.A =
        (PreFOSCsrMatrix){0, dimension, 0, NULL, NULL, a_rows};
    initialize_empty_quadratic(&problem, dimension, q_rows, r_rows);
    problem.c = c;
    problem.n_box = dimension;
    problem.box_indices = box_indices;
    problem.box_lower = box_lower;
    problem.box_upper = box_upper;
    problem.affine_cone_matrix = (PreFOSCsrMatrix){
        dimension, dimension, dimension, g_values, g_columns, g_rows};
    problem.affine_cone_offset = h;
    problem.n_affine_cones = 1;
    problem.affine_cones = &affine;
    if (prefos_create_presolver(&problem, &settings, &presolver) !=
        PREFOS_STATUS_OK)
        goto cleanup;
    status = prefos_run_presolve(presolver);
    if (status != PREFOS_STATUS_OK && status != PREFOS_STATUS_REDUCED)
        goto cleanup;
    if (prefos_verify_postsolve_unbounded_ray(
            presolver, ray, 1e-9, &verification) != PREFOS_STATUS_OK ||
        !verification.passed)
        goto cleanup;
    result = 0;

cleanup:
    prefos_free_presolver(presolver);
    free(q_rows);
    free(g_rows);
    free(g_columns);
    free(box_indices);
    free(c);
    free(ray);
    free(g_values);
    free(h);
    free(box_lower);
    free(box_upper);
    return result;
}

static int test_all_cone_unbounded_rays(void)
{
#define CHECK_CONE_RAY(type, dimension, order, alpha, coordinate)               \
    do                                                                          \
    {                                                                           \
        CHECK(test_direct_cone_unbounded_ray(                                   \
                  type, dimension, order, alpha, coordinate) == 0);             \
        CHECK(test_affine_cone_unbounded_ray(                                   \
                  type, dimension, order, alpha, coordinate) == 0);             \
    } while (0)

    CHECK_CONE_RAY(PREFOS_CONE_NONNEGATIVE, 1, 0, 0.0, 0);
    CHECK_CONE_RAY(PREFOS_CONE_SECOND_ORDER, 3, 0, 0.0, 0);
    CHECK_CONE_RAY(PREFOS_CONE_ROTATED_SECOND_ORDER, 3, 0, 0.0, 0);
    CHECK_CONE_RAY(PREFOS_CONE_POSITIVE_SEMIDEFINITE, 3, 2, 0.0, 0);
    CHECK_CONE_RAY(PREFOS_CONE_EXPONENTIAL, 3, 0, 0.0, 2);
    CHECK_CONE_RAY(PREFOS_CONE_POWER, 3, 0, 0.4, 0);
#undef CHECK_CONE_RAY
    return 0;
}

int main(void)
{
    if (test_linear_infeasibility_certificate()) return 1;
    if (test_affine_infeasibility_certificate()) return 1;
    if (test_working_bound_blocks_free_substitution()) return 1;
    if (test_all_direct_cone_certificates()) return 1;
    if (test_all_affine_cone_certificates()) return 1;
    if (test_propagated_bound_certificate()) return 1;
    if (test_long_propagated_bound_certificate()) return 1;
    if (test_input_affine_bound_certificate()) return 1;
    if (test_generated_affine_bound_certificate()) return 1;
    if (test_affine_aggregation_certificate()) return 1;
    if (test_substitution_certificate()) return 1;
    if (test_residual_row_certificate()) return 1;
    if (test_fixed_column_certificate()) return 1;
    if (test_parallel_row_certificate()) return 1;
    if (test_parallel_column_certificate()) return 1;
    if (test_bounded_doubleton_certificate()) return 1;
    if (test_soc_collapse_certificate()) return 1;
    if (test_rsoc_face_extended_certificate()) return 1;
    if (test_affine_rsoc_face_extended_certificate()) return 1;
    if (test_generated_affine_rsoc_face_extended_certificate()) return 1;
    if (test_unbounded_ray_identity()) return 1;
    if (test_substitution_unbounded_ray()) return 1;
    if (test_parallel_column_unbounded_ray()) return 1;
    if (test_infinite_fixed_column_unbounded_ray()) return 1;
    if (test_affine_aggregation_unbounded_ray()) return 1;
    if (test_all_cone_unbounded_rays()) return 1;
    printf("All PreFOS certificate tests passed!\n");
    return 0;
}
