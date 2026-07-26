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
            fprintf(stderr, "check failed at %s:%d: %s\n", __FILE__, __LINE__, \
                    #condition);                                                \
            return 1;                                                           \
        }                                                                       \
    } while (0)

static int close_to(double left, double right)
{
    return fabs(left - right) <= 1e-10;
}

static void init_linear_problem(
    PreFOSProblemData *problem, size_t n, size_t rows, size_t nnz,
    double *A_values, int *A_columns, int *A_rows, double *row_lower,
    double *row_upper, int *Q_rows, int *R_rows, double *c,
    int *box_indices, double *box_lower, double *box_upper)
{
    memset(problem, 0, sizeof(*problem));
    problem->n = n;
    problem->A = (PreFOSCsrMatrix){rows, n, nnz, A_values, A_columns, A_rows};
    problem->constraint_lower = row_lower;
    problem->constraint_upper = row_upper;
    problem->Q = (PreFOSCsrMatrix){n, n, 0, NULL, NULL, Q_rows};
    problem->q_storage = PREFOS_Q_UPPER_TRIANGULAR;
    problem->R = (PreFOSCsrMatrix){0, n, 0, NULL, NULL, R_rows};
    problem->c = c;
    problem->n_box = n;
    problem->box_indices = box_indices;
    problem->box_lower = box_lower;
    problem->box_upper = box_upper;
}

static int test_empty_columns(void)
{
    int A_rows[] = {0};
    int Q_rows[] = {0, 0};
    int R_rows[] = {0};
    double c[] = {2.0};
    int box_indices[] = {0};
    double box_lower[] = {-2.0};
    double box_upper[] = {3.0};
    PreFOSProblemData problem;
    PreFOSPresolver *presolver = NULL;
    const PreFOSPresolvedProblem *reduced;
    double original_x[1];

    init_linear_problem(&problem, 1, 0, 0, NULL, NULL, A_rows, NULL, NULL,
                        Q_rows, R_rows, c, box_indices, box_lower, box_upper);
    CHECK(prefos_create_presolver(&problem, NULL, &presolver) == PREFOS_STATUS_OK);
    CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_REDUCED);
    reduced = prefos_get_reduced_problem(presolver);
    CHECK(reduced != NULL && reduced->n == 0);
    CHECK(close_to(reduced->objective_offset, -4.0));
    CHECK(prefos_get_stats(presolver)->removed_empty_columns == 1);
    CHECK(prefos_postsolve_primal(presolver, NULL, original_x) == PREFOS_STATUS_OK);
    CHECK(close_to(original_x[0], -2.0));
    prefos_free_presolver(presolver);

    box_lower[0] = -INFINITY;
    CHECK(prefos_create_presolver(&problem, NULL, &presolver) == PREFOS_STATUS_OK);
    CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_PRIMAL_UNBOUNDED);
    CHECK(prefos_get_reduced_problem(presolver) == NULL);
    CHECK(strcmp(prefos_status_string(PREFOS_STATUS_PRIMAL_UNBOUNDED),
                 "primal unbounded") == 0);
    prefos_free_presolver(presolver);
    return 0;
}

static int test_quadratic_empty_column_is_protected(void)
{
    int A_rows[] = {0};
    double Q_values[] = {1.0};
    int Q_columns[] = {0};
    int Q_rows[] = {0, 1};
    int R_rows[] = {0};
    double c[] = {1.0};
    int box_indices[] = {0};
    double box_lower[] = {-INFINITY};
    double box_upper[] = {INFINITY};
    PreFOSProblemData problem;
    PreFOSPresolver *presolver = NULL;

    init_linear_problem(&problem, 1, 0, 0, NULL, NULL, A_rows, NULL, NULL,
                        Q_rows, R_rows, c, box_indices, box_lower, box_upper);
    problem.Q.values = Q_values;
    problem.Q.column_indices = Q_columns;
    problem.Q.nnz = 1;
    CHECK(prefos_create_presolver(&problem, NULL, &presolver) == PREFOS_STATUS_OK);
    CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_OK);
    CHECK(prefos_get_reduced_problem(presolver)->n == 1);
    CHECK(prefos_get_stats(presolver)->removed_empty_columns == 0);
    prefos_free_presolver(presolver);
    return 0;
}

static int test_dual_fixing_with_structural_gpu_enabled(void)
{
    double A_values[] = {1.0};
    int A_columns[] = {0};
    int A_rows[] = {0, 1};
    double row_lower[] = {-INFINITY};
    double row_upper[] = {5.0};
    int Q_rows[] = {0, 0};
    int R_rows[] = {0};
    double c[] = {1.0};
    int box_indices[] = {0};
    double box_lower[] = {0.0};
    double box_upper[] = {INFINITY};
    PreFOSProblemData problem;
    PreFOSSettings settings = prefos_strict_settings();
    PreFOSPresolver *presolver = NULL;
    const PreFOSStats *stats;
    double original_x[1], original_y[1], original_z[1];
    PreFOSPostsolveKKTVerification verification;
    settings.structural_reductions_gpu = 1;
    init_linear_problem(&problem, 1, 1, 1, A_values, A_columns, A_rows,
                        row_lower, row_upper, Q_rows, R_rows, c, box_indices,
                        box_lower, box_upper);
    CHECK(prefos_create_presolver(&problem, &settings, &presolver) ==
          PREFOS_STATUS_OK);
    CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_REDUCED);
    stats = prefos_get_stats(presolver);
    CHECK(stats->dual_fixed_columns == 1);
    CHECK(stats->structural_gpu_fallbacks == 0);
    CHECK(prefos_postsolve_primal(presolver, NULL, original_x) == PREFOS_STATUS_OK);
    CHECK(close_to(original_x[0], 0.0));
    prefos_free_presolver(presolver);

    c[0] = 0.0;
    CHECK(prefos_create_presolver(&problem, &settings, &presolver) ==
          PREFOS_STATUS_OK);
    CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_REDUCED);
    CHECK(prefos_get_stats(presolver)->dual_fixed_columns == 1);
    CHECK(prefos_postsolve_primal_dual(
              presolver, NULL, NULL, NULL, 1e-10,
              original_x, original_y, original_z) == PREFOS_STATUS_OK);
    CHECK(close_to(original_x[0], 0.0));
    CHECK(close_to(original_y[0], 0.0));
    CHECK(close_to(original_z[0], 0.0));
    CHECK(prefos_verify_postsolve_kkt(
              presolver, NULL, NULL, NULL, 1e-10,
              &verification) == PREFOS_STATUS_OK);
    CHECK(verification.passed);
    prefos_free_presolver(presolver);

    c[0] = 1.0;
    box_lower[0] = -INFINITY;
    CHECK(prefos_create_presolver(&problem, &settings, &presolver) ==
          PREFOS_STATUS_OK);
    CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_PRIMAL_UNBOUNDED);
    prefos_free_presolver(presolver);
    prefos_gpu_release_cache();
    return 0;
}

static int test_zero_objective_infinite_dual_fixing(void)
{
    double A_values[] = {1.0};
    int A_columns[] = {0};
    int A_rows[] = {0, 1};
    double row_lower[] = {3.0};
    double row_upper[] = {INFINITY};
    int Q_rows[] = {0, 0};
    int R_rows[] = {0};
    double c[] = {0.0};
    int box_indices[] = {0};
    double box_lower[] = {0.0};
    double box_upper[] = {INFINITY};
    PreFOSProblemData problem;
    PreFOSSettings settings = prefos_strict_settings();
    PreFOSPresolver *presolver = NULL;
    PreFOSPostsolveKKTVerification verification;
    double original_x[1], original_y[1], original_z[1];

    init_linear_problem(&problem, 1, 1, 1, A_values, A_columns, A_rows,
                        row_lower, row_upper, Q_rows, R_rows, c,
                        box_indices, box_lower, box_upper);
    CHECK(prefos_create_presolver(&problem, &settings, &presolver) ==
          PREFOS_STATUS_OK);
    CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_REDUCED);
    CHECK(prefos_get_reduced_problem(presolver)->n == 0);
    CHECK(prefos_get_reduced_problem(presolver)->A.rows == 0);
    CHECK(prefos_get_stats(presolver)->dual_fixed_columns == 1);
    CHECK(prefos_get_stats(presolver)->removed_redundant_rows == 1);
    CHECK(prefos_postsolve_primal_dual(
              presolver, NULL, NULL, NULL, 1e-10,
              original_x, original_y, original_z) == PREFOS_STATUS_OK);
    CHECK(close_to(original_x[0], 3.0));
    CHECK(close_to(original_y[0], 0.0));
    CHECK(close_to(original_z[0], 0.0));
    CHECK(prefos_verify_postsolve_kkt(
              presolver, NULL, NULL, NULL, 1e-10,
              &verification) == PREFOS_STATUS_OK);
    CHECK(verification.passed);
    prefos_free_presolver(presolver);

    row_lower[0] = -INFINITY;
    row_upper[0] = -2.0;
    box_lower[0] = -INFINITY;
    box_upper[0] = 0.0;
    CHECK(prefos_create_presolver(&problem, &settings, &presolver) ==
          PREFOS_STATUS_OK);
    CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_REDUCED);
    CHECK(prefos_postsolve_primal_dual(
              presolver, NULL, NULL, NULL, 1e-10,
              original_x, original_y, original_z) == PREFOS_STATUS_OK);
    CHECK(close_to(original_x[0], -2.0));
    CHECK(close_to(original_y[0], 0.0));
    CHECK(close_to(original_z[0], 0.0));
    CHECK(prefos_verify_postsolve_kkt(
              presolver, NULL, NULL, NULL, 1e-10,
              &verification) == PREFOS_STATUS_OK);
    CHECK(verification.passed);
    prefos_free_presolver(presolver);
    return 0;
}

static int test_singleton_column_substitution(void)
{
    double A_values[] = {1.0, 1.0};
    int A_columns[] = {0, 1};
    int A_rows[] = {0, 2};
    double row_side[] = {2.0};
    int Q_rows[] = {0, 0, 0};
    int R_rows[] = {0};
    double c[] = {1.0, 0.0};
    int box_indices[] = {0, 1};
    double box_lower[] = {-INFINITY, 0.0};
    double box_upper[] = {INFINITY, 2.0};
    double original_x[2];
    PreFOSProblemData problem;
    PreFOSSettings settings = prefos_strict_settings();
    PreFOSPresolver *presolver = NULL;
    const PreFOSPresolvedProblem *reduced;
    const PreFOSStats *stats;
    PreFOSPrimalVerification verification;
    settings.structural_reductions_gpu = 1;
    init_linear_problem(&problem, 2, 1, 2, A_values, A_columns, A_rows,
                        row_side, row_side, Q_rows, R_rows, c, box_indices,
                        box_lower, box_upper);
    CHECK(prefos_create_presolver(&problem, &settings, &presolver) ==
          PREFOS_STATUS_OK);
    CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_REDUCED);
    reduced = prefos_get_reduced_problem(presolver);
    stats = prefos_get_stats(presolver);
    CHECK(reduced->n == 0 && reduced->A.rows == 0);
    CHECK(close_to(reduced->objective_offset, 0.0));
    CHECK(stats->removed_singleton_columns == 1);
    CHECK(stats->removed_empty_columns == 1);
    if (stats->column_csc_gpu_builds > 0)
    {
        CHECK(stats->column_csc_gpu_fallbacks == 0);
        CHECK(stats->singleton_column_gpu_passes > 0);
        CHECK(stats->singleton_column_gpu_fallbacks == 0);
    }
    else
        CHECK(stats->singleton_column_gpu_passes == 0);
    CHECK(prefos_postsolve_primal(presolver, NULL, original_x) ==
          PREFOS_STATUS_OK);
    CHECK(close_to(original_x[0], 0.0) && close_to(original_x[1], 2.0));
    CHECK(prefos_verify_postsolve_primal(presolver, NULL, 1e-10,
                                         &verification) == PREFOS_STATUS_OK);
    CHECK(verification.passed);
    prefos_free_presolver(presolver);
    prefos_gpu_release_cache();
    return 0;
}

static int test_singleton_residual_row_preserves_fixed_shift(void)
{
    double A_values[] = {1.0, -1.0, -1.0, -1.0, -1.0, 1.0};
    int A_columns[] = {0, 1, 2, 3, 4, 5};
    int A_rows[] = {0, 6};
    double row_side[] = {0.0};
    double Q_values[] = {1.0, 1.0, 1.0};
    int Q_columns[] = {0, 1, 2};
    int Q_rows[] = {0, 1, 2, 3, 3, 3, 3};
    int R_rows[] = {0};
    double c[] = {1.0, 1.0, 1.0, 0.0, 0.0, 0.0};
    int box_indices[] = {0, 1, 2, 3, 4, 5};
    double box_lower[] = {31.0, 30.0, 17.0, 30.0, 16.0, 18.0};
    double box_upper[] = {38.0, 37.0, 24.0, 30.0, 16.0, 71.0};
    double reduced_x[] = {31.0, 30.0, 17.0};
    double original_x[6];
    PreFOSProblemData problem;
    PreFOSPresolver *presolver = NULL;
    const PreFOSPresolvedProblem *reduced;
    PreFOSPrimalVerification verification;

    init_linear_problem(
        &problem, 6, 1, 6, A_values, A_columns, A_rows, row_side, row_side,
        Q_rows, R_rows, c, box_indices, box_lower, box_upper);
    problem.Q.values = Q_values;
    problem.Q.column_indices = Q_columns;
    problem.Q.nnz = 3;
    CHECK(prefos_create_presolver(&problem, NULL, &presolver) ==
          PREFOS_STATUS_OK);
    CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_REDUCED);
    CHECK(prefos_get_stats(presolver)->residual_row_substitutions == 1);
    reduced = prefos_get_reduced_problem(presolver);
    CHECK(reduced->n == 3);
    CHECK(reduced->A.rows == 1 && reduced->A.nnz == 3);
    CHECK(close_to(reduced->constraint_lower[0], -25.0));
    CHECK(isinf(reduced->constraint_upper[0]) &&
          reduced->constraint_upper[0] > 0.0);
    CHECK(prefos_postsolve_primal(presolver, reduced_x, original_x) ==
          PREFOS_STATUS_OK);
    CHECK(close_to(original_x[0], 31.0));
    CHECK(close_to(original_x[1], 30.0));
    CHECK(close_to(original_x[2], 17.0));
    CHECK(close_to(original_x[3], 30.0));
    CHECK(close_to(original_x[4], 16.0));
    CHECK(close_to(original_x[5], 62.0));
    CHECK(prefos_verify_postsolve_primal(
              presolver, reduced_x, 1e-10, &verification) ==
          PREFOS_STATUS_OK);
    CHECK(verification.passed);
    prefos_free_presolver(presolver);
    return 0;
}

static int test_singleton_equality_retains_two_sided_box(void)
{
    double A_values[] = {1.0, 1.0};
    int A_columns[] = {0, 1};
    int A_rows[] = {0, 2};
    double row_side[] = {0.0};
    double Q_values[] = {1.0};
    int Q_columns[] = {1};
    int Q_rows[] = {0, 0, 1};
    int R_rows[] = {0};
    double c[] = {0.0, 0.0};
    int box_indices[] = {0, 1};
    double box_lower[] = {0.0, -10.0};
    double box_upper[] = {1.0, 10.0};
    double reduced_x[] = {-0.5};
    double original_x[2];
    PreFOSProblemData problem;
    PreFOSSettings settings = prefos_strict_settings();
    PreFOSPresolver *presolver = NULL;
    const PreFOSPresolvedProblem *reduced;
    PreFOSPrimalVerification verification;

    settings.dual_fixing = 0;
    settings.linear_propagation = 0;
    settings.remove_redundant_rows = 0;
    settings.remove_redundant_bounds = 0;
    settings.bounded_doubleton_substitution = 0;
    settings.parallel_column_reduction = 0;
    init_linear_problem(
        &problem, 2, 1, 2, A_values, A_columns, A_rows, row_side, row_side,
        Q_rows, R_rows, c, box_indices, box_lower, box_upper);
    problem.Q.values = Q_values;
    problem.Q.column_indices = Q_columns;
    problem.Q.nnz = 1;

    CHECK(prefos_create_presolver(&problem, &settings, &presolver) ==
          PREFOS_STATUS_OK);
    CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_REDUCED);
    reduced = prefos_get_reduced_problem(presolver);
    CHECK(reduced != NULL);
    CHECK(reduced->n == 1 && reduced->A.rows == 0 && reduced->A.nnz == 0);
    CHECK(reduced->n_box == 1 && reduced->box_indices[0] == 0);
    CHECK(close_to(reduced->box_lower[0], -1.0));
    CHECK(close_to(reduced->box_upper[0], 0.0));
    CHECK(prefos_get_stats(presolver)->removed_singleton_columns == 1);
    CHECK(prefos_get_stats(presolver)->residual_row_substitutions == 1);
    CHECK(prefos_postsolve_primal(presolver, reduced_x, original_x) ==
          PREFOS_STATUS_OK);
    CHECK(close_to(original_x[0], 0.5));
    CHECK(close_to(original_x[1], -0.5));
    CHECK(prefos_verify_postsolve_primal(
              presolver, reduced_x, 1e-10, &verification) ==
          PREFOS_STATUS_OK);
    CHECK(verification.passed);
    prefos_free_presolver(presolver);
    return 0;
}

static int test_singleton_implied_free_avoids_large_bound_cancellation(void)
{
    double A_values[] = {
        1.0, 1.0,
        1.0, -1.0,
        1.0, -1.0,
        -1.0, 1.0};
    int A_columns[] = {
        0, 1,
        1, 2,
        2, 3,
        1, 3};
    int A_rows[] = {0, 2, 4, 6, 8};
    double row_lower[] = {40.0, 0.0, 0.0, 0.0};
    double row_upper[] = {INFINITY, 0.0, 0.0, 0.0};
    int Q_rows[] = {0, 0, 0, 0, 0};
    int R_rows[] = {0};
    double c[] = {1.0, -1.0, 0.0, 0.0};
    int box_indices[] = {0, 1, 2, 3};
    double box_lower[] = {0.0, 0.0, 0.0, 0.0};
    double box_upper[] = {1e20, 1640.0, 1640.0, 1640.0};
    PreFOSProblemData problem;
    PreFOSSettings settings = prefos_strict_settings();
    PreFOSPresolver *presolver = NULL;
    const PreFOSPresolvedProblem *reduced;

    settings.linear_propagation = 0;
    settings.remove_redundant_rows = 0;
    settings.remove_redundant_bounds = 0;
    settings.free_column_substitution = 0;
    settings.bounded_doubleton_substitution = 0;
    settings.parallel_column_reduction = 0;
    settings.dual_fixing = 0;
    init_linear_problem(
        &problem, 4, 4, 8, A_values, A_columns, A_rows,
        row_lower, row_upper, Q_rows, R_rows, c, box_indices,
        box_lower, box_upper);

    CHECK(prefos_create_presolver(&problem, &settings, &presolver) ==
          PREFOS_STATUS_OK);
    CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_OK);
    reduced = prefos_get_reduced_problem(presolver);
    CHECK(reduced != NULL);
    CHECK(reduced->n == 4 && reduced->A.rows == 4 &&
          reduced->A.nnz == 8);
    CHECK(prefos_get_stats(presolver)->removed_singleton_columns == 0);
    prefos_free_presolver(presolver);
    return 0;
}

static int test_one_sided_singleton_reductions(void)
{
    {
        double A_values[] = {1.0, -10.0};
        int A_columns[] = {0, 1};
        int A_rows[] = {0, 2};
        double row_lower[] = {-INFINITY};
        double row_upper[] = {0.0};
        double Q_values[] = {1.0};
        int Q_columns[] = {0};
        int Q_rows[] = {0, 1, 1};
        int R_rows[] = {0};
        double c[] = {0.0, 1.0};
        int box_indices[] = {0, 1};
        double box_lower[] = {0.0, 0.0};
        double box_upper[] = {20.0, 1.0};
        double reduced_x[] = {5.0};
        double original_x[2];
        PreFOSProblemData problem;
        PreFOSSettings settings = prefos_strict_settings();
        PreFOSPresolver *presolver = NULL;
        const PreFOSPresolvedProblem *reduced;
        PreFOSPrimalVerification verification;
        PreFOSStatus status;

        settings.dual_fixing = 0;
        settings.linear_propagation = 0;
        settings.remove_redundant_rows = 0;
        settings.remove_redundant_bounds = 0;
        settings.bounded_doubleton_substitution = 0;
        settings.parallel_column_reduction = 0;
        init_linear_problem(&problem, 2, 1, 2, A_values, A_columns, A_rows,
                            row_lower, row_upper, Q_rows, R_rows, c, box_indices,
                            box_lower, box_upper);
        problem.Q.values = Q_values;
        problem.Q.column_indices = Q_columns;
        problem.Q.nnz = 1;
        CHECK(prefos_create_presolver(&problem, &settings, &presolver) ==
              PREFOS_STATUS_OK);
        status = prefos_run_presolve(presolver);
        CHECK(status == PREFOS_STATUS_REDUCED);
        reduced = prefos_get_reduced_problem(presolver);
        CHECK(reduced->n == 1 && reduced->A.rows == 0 && reduced->A.nnz == 0);
        CHECK(close_to(reduced->box_upper[0], 10.0));
        CHECK(close_to(reduced->c[0], 0.1));
        CHECK(prefos_get_stats(presolver)->tightened_singleton_rows == 1);
        CHECK(prefos_get_stats(presolver)->removed_singleton_columns == 1);
        CHECK(prefos_get_stats(presolver)->residual_row_substitutions == 1);
        CHECK(prefos_postsolve_primal(presolver, reduced_x, original_x) ==
              PREFOS_STATUS_OK);
        CHECK(close_to(original_x[0], 5.0));
        CHECK(close_to(original_x[1], 0.5));
        CHECK(prefos_verify_postsolve_primal(
                  presolver, reduced_x, 1e-10, &verification) ==
              PREFOS_STATUS_OK);
        CHECK(verification.passed);
        prefos_free_presolver(presolver);
    }
    {
        double A_values[] = {1.0, 1.0, 1.0};
        int A_columns[] = {0, 1, 2};
        int A_rows[] = {0, 3};
        double row_side[] = {2.0};
        int Q_rows[] = {0, 0, 0, 0};
        int R_rows[] = {0};
        double c[] = {-1.0, 2.0, 2.0};
        int box_indices[] = {0, 1, 2};
        double box_lower[] = {-INFINITY, 0.0, 0.0};
        double box_upper[] = {1.0, 10.0, 10.0};
        double reduced_x[] = {1.0, 0.0};
        double original_x[3];
        double reduced_y[] = {-3.0};
        double reduced_z[] = {0.0, 0.0};
        double original_y[1], original_z[3];
        PreFOSProblemData problem;
        PreFOSSettings settings = prefos_strict_settings();
        PreFOSPresolver *presolver = NULL;
        const PreFOSPresolvedProblem *reduced;
        PreFOSPostsolveKKTVerification verification;

        init_linear_problem(&problem, 3, 1, 3, A_values, A_columns, A_rows,
                            row_side, row_side, Q_rows, R_rows, c, box_indices,
                            box_lower, box_upper);
        settings.parallel_column_reduction = 0;
        CHECK(prefos_create_presolver(&problem, &settings, &presolver) ==
              PREFOS_STATUS_OK);
        CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_REDUCED);
        reduced = prefos_get_reduced_problem(presolver);
        CHECK(reduced->n == 2 && reduced->A.rows == 1 && reduced->A.nnz == 2);
        CHECK(close_to(reduced->constraint_lower[0], 1.0));
        CHECK(isinf(reduced->constraint_upper[0]) &&
              reduced->constraint_upper[0] > 0.0);
        CHECK(prefos_get_stats(presolver)->removed_singleton_columns == 1);
        CHECK(prefos_postsolve_primal(presolver, reduced_x, original_x) ==
              PREFOS_STATUS_OK);
        CHECK(close_to(original_x[0], 1.0));
        CHECK(prefos_postsolve_primal_dual(
                  presolver, reduced_x, reduced_y, reduced_z, 1e-10,
                  original_x, original_y, original_z) ==
              PREFOS_STATUS_OK);
        CHECK(close_to(original_y[0], -2.0));
        CHECK(close_to(original_z[0], 3.0));
        CHECK(close_to(original_z[1], 0.0));
        CHECK(close_to(original_z[2], 0.0));
        CHECK(prefos_verify_postsolve_kkt(
                  presolver, reduced_x, reduced_y, reduced_z, 1e-10,
                  &verification) == PREFOS_STATUS_OK);
        CHECK(verification.passed);
        prefos_free_presolver(presolver);
    }
    {
        double A_values[] = {1.0, 1.0};
        int A_columns[] = {0, 1};
        int A_rows[] = {0, 2};
        double row_lower[] = {4.0};
        double row_upper[] = {INFINITY};
        int Q_rows[] = {0, 0, 0};
        int R_rows[] = {0};
        double c[] = {1.0, 0.0};
        int box_indices[] = {0, 1};
        double box_lower[] = {0.0, 0.0};
        double box_upper[] = {10.0, 1.0};
        PreFOSProblemData problem;
        PreFOSSettings settings = prefos_default_settings();
        PreFOSPresolver *presolver = NULL;
        const PreFOSPresolvedProblem *reduced;
        double original_x[2], original_y[1], original_z[2];
        PreFOSPostsolveKKTVerification verification;

        init_linear_problem(&problem, 2, 1, 2, A_values, A_columns, A_rows,
                            row_lower, row_upper, Q_rows, R_rows, c, box_indices,
                            box_lower, box_upper);
        settings.dual_fixing = 0;
        CHECK(prefos_create_presolver(&problem, &settings, &presolver) ==
              PREFOS_STATUS_OK);
        CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_REDUCED);
        reduced = prefos_get_reduced_problem(presolver);
        CHECK(reduced->n == 0 && reduced->A.rows == 0);
        CHECK(close_to(reduced->objective_offset, 3.0));
        CHECK(prefos_get_stats(presolver)->tightened_singleton_rows == 1);
        CHECK(prefos_postsolve_primal_dual(
                  presolver, NULL, NULL, NULL, 1e-10,
                  original_x, original_y, original_z) == PREFOS_STATUS_OK);
        CHECK(close_to(original_y[0], -1.0));
        CHECK(close_to(original_z[0], 0.0));
        CHECK(close_to(original_z[1], 1.0));
        CHECK(prefos_verify_postsolve_kkt(
                  presolver, NULL, NULL, NULL, 1e-10,
                  &verification) == PREFOS_STATUS_OK);
        CHECK(verification.passed);
        prefos_free_presolver(presolver);
    }
    {
        double A_values[] = {1.0, 1.0};
        int A_columns[] = {0, 1};
        int A_rows[] = {0, 2};
        double row_lower[] = {-INFINITY};
        double row_upper[] = {4.0};
        int Q_rows[] = {0, 0, 0};
        int R_rows[] = {0};
        double c[] = {-1.0, 0.0};
        int box_indices[] = {0, 1};
        double box_lower[] = {0.0, 0.0};
        double box_upper[] = {10.0, 1.0};
        double original_x[2], original_y[1], original_z[2];
        PreFOSProblemData problem;
        PreFOSPresolver *presolver = NULL;
        PreFOSPostsolveKKTVerification verification;

        init_linear_problem(&problem, 2, 1, 2, A_values, A_columns, A_rows,
                            row_lower, row_upper, Q_rows, R_rows, c, box_indices,
                            box_lower, box_upper);
        CHECK(prefos_create_presolver(&problem, NULL, &presolver) ==
              PREFOS_STATUS_OK);
        CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_REDUCED);
        CHECK(prefos_get_reduced_problem(presolver)->n == 0);
        CHECK(prefos_get_reduced_problem(presolver)->A.rows == 0);
        CHECK(close_to(prefos_get_reduced_problem(presolver)->objective_offset,
                       -4.0));
        CHECK(prefos_postsolve_primal_dual(
                  presolver, NULL, NULL, NULL, 1e-10,
                  original_x, original_y, original_z) == PREFOS_STATUS_OK);
        CHECK(close_to(original_y[0], 1.0));
        CHECK(close_to(original_z[0], 0.0));
        CHECK(close_to(original_z[1], -1.0));
        CHECK(prefos_verify_postsolve_kkt(
                  presolver, NULL, NULL, NULL, 1e-10,
                  &verification) == PREFOS_STATUS_OK);
        CHECK(verification.passed);
        prefos_free_presolver(presolver);
    }
    {
        double A_values[] = {-1.0, 1.0};
        int A_columns[] = {0, 1};
        int A_rows[] = {0, 2};
        double row_side[] = {0.0};
        int Q_rows[] = {0, 0, 0};
        int R_rows[] = {0};
        double c[] = {1.0, 2.0};
        int box_indices[] = {0, 1};
        double box_lower[] = {0.0, -10.0};
        double box_upper[] = {INFINITY, 10.0};
        double reduced_x[] = {0.0};
        double reduced_y[] = {-3.0};
        double reduced_z[] = {0.0};
        double original_x[2], original_y[1], original_z[2];
        PreFOSProblemData problem;
        PreFOSSettings settings = prefos_strict_settings();
        PreFOSPresolver *presolver = NULL;
        PreFOSPostsolveKKTVerification verification;

        settings.linear_propagation = 0;
        settings.bounded_doubleton_substitution = 0;
        init_linear_problem(&problem, 2, 1, 2, A_values, A_columns, A_rows,
                            row_side, row_side, Q_rows, R_rows, c, box_indices,
                            box_lower, box_upper);
        CHECK(prefos_create_presolver(&problem, &settings, &presolver) ==
              PREFOS_STATUS_OK);
        CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_REDUCED);
        CHECK(prefos_get_reduced_problem(presolver)->n == 1);
        CHECK(close_to(prefos_get_reduced_problem(presolver)
                           ->constraint_lower[0],
                       0.0));
        CHECK(prefos_postsolve_primal_dual(
                  presolver, reduced_x, reduced_y, reduced_z, 1e-10,
                  original_x, original_y, original_z) == PREFOS_STATUS_OK);
        CHECK(close_to(original_x[0], 0.0));
        CHECK(close_to(original_y[0], -2.0));
        CHECK(close_to(original_z[0], -3.0));
        CHECK(close_to(original_z[1], 0.0));
        CHECK(prefos_verify_postsolve_kkt(
                  presolver, reduced_x, reduced_y, reduced_z, 1e-10,
                  &verification) == PREFOS_STATUS_OK);
        CHECK(verification.passed);
        prefos_free_presolver(presolver);
    }
    return 0;
}

static int test_bounded_doubleton_substitution(void)
{
    double A_values[] = {1.0, 1.0};
    int A_columns[] = {0, 1};
    int A_rows[] = {0, 2};
    double row_side[] = {5.0};
    int Q_rows[] = {0, 0, 0};
    int R_rows[] = {0};
    double c[] = {1.0, -2.0};
    int box_indices[] = {0, 1};
    double box_lower[] = {0.0, 0.0};
    double box_upper[] = {10.0, 10.0};
    double reduced_x[] = {0.0};
    double original_x[2];
    double reduced_z[] = {-3.0};
    double original_y[1];
    double original_z[2];
    PreFOSProblemData problem;
    PreFOSSettings settings = prefos_strict_settings();
    PreFOSPresolver *presolver = NULL;
    const PreFOSPresolvedProblem *reduced;
    PreFOSPostsolveKKTVerification verification;

    settings.singleton_column_reduction = 0;
    settings.bounded_doubleton_substitution = 1;
    settings.remove_empty_columns = 0;
    init_linear_problem(&problem, 2, 1, 2, A_values, A_columns, A_rows,
                        row_side, row_side, Q_rows, R_rows, c, box_indices,
                        box_lower, box_upper);
    CHECK(prefos_create_presolver(&problem, &settings, &presolver) ==
          PREFOS_STATUS_OK);
    CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_REDUCED);
    reduced = prefos_get_reduced_problem(presolver);
    CHECK(reduced->n == 1 && reduced->A.rows == 0);
    CHECK(close_to(reduced->box_lower[0], 0.0));
    CHECK(close_to(reduced->box_upper[0], 5.0));
    CHECK(close_to(reduced->objective_offset, -10.0));
    CHECK(close_to(reduced->c[0], 3.0));
    CHECK(prefos_get_stats(presolver)->substituted_bounded_doubletons == 1);
    CHECK(prefos_postsolve_primal(presolver, reduced_x, original_x) ==
          PREFOS_STATUS_OK);
    CHECK(close_to(original_x[0], 0.0) && close_to(original_x[1], 5.0));
    CHECK(prefos_postsolve_primal_dual(presolver, reduced_x, NULL, reduced_z,
                                       1e-10, original_x, original_y, original_z) ==
          PREFOS_STATUS_OK);
    CHECK(close_to(original_y[0], 2.0));
    CHECK(close_to(original_z[0], -3.0));
    CHECK(close_to(original_z[1], 0.0));
    CHECK(prefos_verify_postsolve_kkt(presolver, reduced_x, NULL, reduced_z,
                                      1e-10, &verification) == PREFOS_STATUS_OK);
    CHECK(verification.passed);
    prefos_free_presolver(presolver);

    {
        double positive_A_values[] = {1.0, -1.0};
        int positive_A_columns[] = {0, 1};
        int positive_A_rows[] = {0, 2};
        double positive_side[] = {1.0};
        int positive_Q_rows[] = {0, 0, 0};
        int positive_R_rows[] = {0};
        double positive_c[] = {-1.0, -2.0};
        int positive_box_indices[] = {0, 1};
        double positive_box_lower[] = {0.0, 0.0};
        double positive_box_upper[] = {2.0, 10.0};
        double positive_reduced_x[] = {2.0};
        double positive_reduced_z[] = {3.0};
        double positive_original_x[2], positive_original_y[1];
        double positive_original_z[2];
        PreFOSProblemData positive_problem;
        PreFOSSettings positive_settings = prefos_strict_settings();
        PreFOSPresolver *positive_presolver = NULL;
        PreFOSPostsolveKKTVerification positive_verification;

        positive_settings.singleton_column_reduction = 0;
        positive_settings.bounded_doubleton_substitution = 1;
        positive_settings.remove_empty_columns = 0;
        init_linear_problem(
            &positive_problem, 2, 1, 2, positive_A_values,
            positive_A_columns, positive_A_rows, positive_side, positive_side,
            positive_Q_rows, positive_R_rows, positive_c,
            positive_box_indices, positive_box_lower, positive_box_upper);
        CHECK(prefos_create_presolver(&positive_problem, &positive_settings,
                                      &positive_presolver) == PREFOS_STATUS_OK);
        CHECK(prefos_run_presolve(positive_presolver) == PREFOS_STATUS_REDUCED);
        CHECK(prefos_postsolve_primal_dual(
                  positive_presolver, positive_reduced_x, NULL,
                  positive_reduced_z, 1e-10, positive_original_x,
                  positive_original_y, positive_original_z) == PREFOS_STATUS_OK);
        CHECK(close_to(positive_original_x[0], 2.0));
        CHECK(close_to(positive_original_y[0], -2.0));
        CHECK(close_to(positive_original_z[0], 3.0));
        CHECK(close_to(positive_original_z[1], 0.0));
        CHECK(prefos_verify_postsolve_kkt(
                  positive_presolver, positive_reduced_x, NULL,
                  positive_reduced_z, 1e-10, &positive_verification) ==
              PREFOS_STATUS_OK);
        CHECK(positive_verification.passed);
        prefos_free_presolver(positive_presolver);
    }
    {
        double target_A_values[] = {1.0, 1.0};
        int target_A_columns[] = {0, 1};
        int target_A_rows[] = {0, 2};
        double target_side[] = {5.0};
        int target_Q_rows[] = {0, 0, 0};
        int target_R_rows[] = {0};
        double target_c[] = {2.0, 3.0};
        int target_box_indices[] = {0, 1};
        double target_box_lower[] = {0.0, 0.0};
        double target_box_upper[] = {10.0, 10.0};
        double target_reduced_x[] = {5.0};
        double target_reduced_z[] = {1.0};
        double target_original_x[2], target_original_y[1], target_original_z[2];
        PreFOSProblemData target_problem;
        PreFOSSettings target_settings = prefos_strict_settings();
        PreFOSPresolver *target_presolver = NULL;
        PreFOSPostsolveKKTVerification target_verification;

        target_settings.singleton_column_reduction = 0;
        target_settings.bounded_doubleton_substitution = 1;
        target_settings.remove_empty_columns = 0;
        init_linear_problem(
            &target_problem, 2, 1, 2, target_A_values, target_A_columns,
            target_A_rows, target_side, target_side, target_Q_rows,
            target_R_rows, target_c, target_box_indices, target_box_lower,
            target_box_upper);
        CHECK(prefos_create_presolver(&target_problem, &target_settings,
                                      &target_presolver) == PREFOS_STATUS_OK);
        CHECK(prefos_run_presolve(target_presolver) == PREFOS_STATUS_REDUCED);
        CHECK(prefos_postsolve_primal_dual(
                  target_presolver, target_reduced_x, NULL, target_reduced_z,
                  1e-10, target_original_x, target_original_y,
                  target_original_z) == PREFOS_STATUS_OK);
        CHECK(close_to(target_original_x[0], 5.0));
        CHECK(close_to(target_original_y[0], -2.0));
        CHECK(close_to(target_original_z[0], 0.0));
        CHECK(close_to(target_original_z[1], -1.0));
        CHECK(prefos_verify_postsolve_kkt(
                  target_presolver, target_reduced_x, NULL, target_reduced_z,
                  1e-10, &target_verification) == PREFOS_STATUS_OK);
        CHECK(target_verification.passed);
        prefos_free_presolver(target_presolver);
    }
    return 0;
}

static int test_materialized_propagation_dual_source(void)
{
    double A_values[] = {1.0, 1.0, 1.0, 1.0, 1.0};
    int A_columns[] = {0, 1, 0, 1, 2};
    int A_rows[] = {0, 2, 5};
    double row_lower[] = {5.0, 7.0};
    double row_upper[] = {5.0, INFINITY};
    int Q_rows[] = {0, 0, 0, 0};
    int R_rows[] = {0};
    double c[] = {0.0, 0.0, 1.0};
    int box_indices[] = {0, 1, 2};
    double box_lower[] = {0.0, 0.0, 0.0};
    double box_upper[] = {10.0, 10.0, 10.0};
    double reduced_x[] = {0.0, 2.0};
    double reduced_z[] = {0.0, -1.0};
    double original_x[3], original_y[2], original_z[3];
    PreFOSProblemData problem;
    PreFOSSettings settings = prefos_strict_settings();
    PreFOSPresolver *presolver = NULL;
    const PreFOSPresolvedProblem *reduced;
    PreFOSPostsolveKKTVerification verification;

    settings.remove_redundant_rows = 0;
    settings.free_column_substitution = 0;
    settings.remove_empty_columns = 0;
    settings.singleton_column_reduction = 0;
    settings.parallel_column_reduction = 0;
    settings.dual_fixing = 0;
    settings.remove_redundant_bounds = 0;
    settings.bounded_doubleton_substitution = 1;
    init_linear_problem(
        &problem, 3, 2, 5, A_values, A_columns, A_rows,
        row_lower, row_upper, Q_rows, R_rows, c, box_indices,
        box_lower, box_upper);
    CHECK(prefos_create_presolver(&problem, &settings, &presolver) ==
          PREFOS_STATUS_OK);
    CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_REDUCED);
    reduced = prefos_get_reduced_problem(presolver);
    CHECK(reduced->n == 2 && reduced->A.rows == 0 &&
          reduced->A.nnz == 0);
    CHECK(close_to(reduced->box_lower[1], 2.0));
    CHECK(prefos_get_stats(
              presolver)->substituted_bounded_doubletons == 1);
    CHECK(prefos_postsolve_primal_dual(
              presolver, reduced_x, NULL, reduced_z, 1e-10,
              original_x, original_y, original_z) ==
          PREFOS_STATUS_OK);
    CHECK(close_to(original_x[0], 0.0));
    CHECK(close_to(original_x[1], 5.0));
    CHECK(close_to(original_x[2], 2.0));
    CHECK(close_to(original_y[0], 1.0));
    CHECK(close_to(original_y[1], -1.0));
    CHECK(close_to(original_z[0], 0.0));
    CHECK(close_to(original_z[1], 0.0));
    CHECK(close_to(original_z[2], 0.0));
    CHECK(prefos_verify_postsolve_kkt(
              presolver, reduced_x, NULL, reduced_z, 1e-10,
              &verification) == PREFOS_STATUS_OK);
    CHECK(verification.passed);
    prefos_free_presolver(presolver);
    return 0;
}

static int test_budgeted_materialized_propagation_closure(void)
{
    double A_values[] = {
        1.0, 1.0,
        1.0, 1.0,
        1.0, 1.0, 1.0};
    int A_columns[] = {0, 1, 2, 3, 0, 1, 2};
    int A_rows[] = {0, 2, 4, 7};
    double row_lower[] = {5.0, -INFINITY, 7.0};
    double row_upper[] = {5.0, 3.0, INFINITY};
    int Q_rows[] = {0, 0, 0, 0, 0};
    int R_rows[] = {0};
    double c[] = {0.0, 0.0, 0.0, 0.0};
    int box_indices[] = {0, 1, 2, 3};
    double box_lower[] = {0.0, 0.0, 0.0, 0.0};
    double box_upper[] = {10.0, 10.0, 10.0, 10.0};
    PreFOSProblemData problem;
    PreFOSSettings settings = prefos_default_settings();
    PreFOSPresolver *presolver = NULL;
    const PreFOSPresolvedProblem *reduced;

    settings.remove_redundant_rows = 0;
    settings.free_column_substitution = 0;
    settings.remove_empty_columns = 0;
    settings.singleton_column_reduction = 0;
    settings.parallel_column_reduction = 0;
    settings.dual_fixing = 0;
    settings.remove_redundant_bounds = 0;
    settings.bounded_doubleton_substitution = 1;
    init_linear_problem(
        &problem, 4, 3, 7, A_values, A_columns, A_rows,
        row_lower, row_upper, Q_rows, R_rows, c, box_indices,
        box_lower, box_upper);
    CHECK(prefos_create_presolver(&problem, &settings, &presolver) ==
          PREFOS_STATUS_OK);
    CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_REDUCED);
    reduced = prefos_get_reduced_problem(presolver);
    CHECK(reduced != NULL && reduced->n == 3);
    CHECK(close_to(reduced->box_lower[1], 2.0));
    CHECK(close_to(reduced->box_upper[1], 3.0));
    CHECK(close_to(reduced->box_upper[2], 1.0));
    CHECK(prefos_get_stats(
              presolver)->substituted_bounded_doubletons == 1);
    prefos_free_presolver(presolver);
    return 0;
}

static int test_bounded_doubleton_skips_dirty_rows(void)
{
    double A_values[] = {1.0, 1.0, -1.0, 1.0, 1.0};
    int A_columns[] = {0, 1, 0, 2, 3};
    int A_rows[] = {0, 2, 5};
    double row_side[] = {1.0, -1.0};
    int Q_rows[] = {0, 0, 0, 0, 0};
    int R_rows[] = {0};
    double c[] = {0.0, 0.0, 0.0, 0.0};
    int box_indices[] = {0, 1, 2, 3};
    double box_lower[] = {0.0, 0.0, 0.0, 0.0};
    double box_upper[] = {INFINITY, INFINITY, INFINITY, INFINITY};
    PreFOSProblemData problem;
    PreFOSSettings settings = prefos_strict_settings();
    PreFOSPresolver *presolver = NULL;
    PreFOSStatus status;

    settings.singleton_column_reduction = 0;
    settings.bounded_doubleton_substitution = 1;
    settings.linear_propagation = 0;
    settings.remove_redundant_rows = 0;
    settings.parallel_column_reduction = 0;
    settings.dual_fixing = 0;
    init_linear_problem(&problem, 4, 2, 5, A_values, A_columns, A_rows,
                        row_side, row_side, Q_rows, R_rows, c, box_indices,
                        box_lower, box_upper);
    CHECK(prefos_create_presolver(&problem, &settings, &presolver) ==
          PREFOS_STATUS_OK);
    status = prefos_run_presolve(presolver);
    CHECK(status == PREFOS_STATUS_REDUCED || status == PREFOS_STATUS_OK);
    CHECK(prefos_get_stats(presolver)->substituted_bounded_doubletons == 1);
    prefos_free_presolver(presolver);
    return 0;
}

static int test_bounded_doubleton_has_zero_net_fill(void)
{
    double A_values[] = {
        1.0, -1.0,
        1.0, 1.0,
        1.0, 1.0,
        1.0, 1.0,
        1.0, 1.0,
        1.0, 1.0,
        1.0, 1.0,
        1.0, 1.0,
        1.0, 1.0,
        1.0, 1.0,
        1.0, 1.0};
    int A_columns[] = {
        0, 1,
        0, 2,
        0, 3,
        0, 4,
        0, 5,
        0, 6,
        1, 7,
        1, 8,
        1, 9,
        1, 10,
        1, 11};
    int A_rows[] = {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22};
    double row_lower[] = {0.0, -1.0, -1.0, -1.0, -1.0, -1.0,
                          -1.0, -1.0, -1.0, -1.0, -1.0};
    double row_upper[] = {0.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                          1.0, 1.0, 1.0, 1.0, 1.0};
    int Q_rows[13] = {0};
    int R_rows[] = {0};
    double c[12] = {0.0};
    int box_indices[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    double box_lower[] = {
        -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
        -1.0, -1.0, -1.0, -1.0, -1.0, -1.0};
    double box_upper[] = {
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    double reduced_x[11] = {0.0};
    PreFOSProblemData problem;
    PreFOSSettings settings = prefos_strict_settings();
    PreFOSPresolver *presolver = NULL;
    PreFOSPrimalVerification verification;

    settings.remove_redundant_rows = 0;
    settings.free_column_substitution = 0;
    settings.remove_empty_columns = 0;
    settings.singleton_column_reduction = 0;
    settings.bounded_doubleton_substitution = 1;
    settings.max_aggregation_column_degree = 2;
    settings.max_bounded_doubleton_column_degree = 16;
    settings.max_aggregation_fill = 0;
    settings.linear_propagation = 0;
    settings.parallel_column_reduction = 0;
    settings.dual_fixing = 0;
    settings.remove_redundant_bounds = 0;
    init_linear_problem(
        &problem, 12, 11, 22, A_values, A_columns, A_rows,
        row_lower, row_upper, Q_rows, R_rows, c, box_indices,
        box_lower, box_upper);
    CHECK(prefos_create_presolver(&problem, &settings, &presolver) ==
          PREFOS_STATUS_OK);
    CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_REDUCED);
    CHECK(prefos_get_stats(
              presolver)->substituted_bounded_doubletons == 1);
    CHECK(prefos_get_reduced_problem(presolver)->n == 11);
    CHECK(prefos_verify_postsolve_primal(
              presolver, reduced_x, 1e-10, &verification) ==
          PREFOS_STATUS_OK);
    CHECK(verification.passed);
    prefos_free_presolver(presolver);
    return 0;
}

static int test_bounded_doubleton_exposes_parallel_columns(void)
{
    double A_values[] = {
        1.0, 1.0,
        1.0, -1.0,
        1.0, -1.0,
        1.0, 1.0,
        1.0, 1.0,
        1.0, 1.0};
    int A_columns[] = {
        0, 1,
        0, 2,
        0, 2,
        1, 2,
        1, 2,
        1, 2};
    int A_rows[] = {0, 2, 4, 6, 8, 10, 12};
    double row_lower[] = {0.0, -100.0, -100.0,
                          -100.0, -100.0, -100.0};
    double row_upper[] = {0.0, 100.0, 100.0,
                          100.0, 100.0, 100.0};
    int Q_rows[] = {0, 0, 0, 0};
    int R_rows[] = {0};
    double c[] = {0.0, 0.0, 0.0};
    int box_indices[] = {0, 1, 2};
    double box_lower[] = {-1.0, -1.0, -1.0};
    double box_upper[] = {1.0, 1.0, 1.0};
    double reduced_x[] = {0.0};
    double original_x[3];
    PreFOSProblemData problem;
    PreFOSSettings settings = prefos_strict_settings();
    PreFOSPresolver *presolver = NULL;
    PreFOSPrimalVerification verification;

    settings.remove_redundant_rows = 0;
    settings.free_column_substitution = 0;
    settings.remove_empty_columns = 0;
    settings.singleton_column_reduction = 0;
    settings.bounded_doubleton_substitution = 1;
    settings.max_bounded_doubleton_column_degree = 16;
    settings.linear_propagation = 0;
    settings.parallel_column_reduction = 1;
    settings.dual_fixing = 0;
    settings.remove_redundant_bounds = 0;
    init_linear_problem(
        &problem, 3, 6, 12, A_values, A_columns, A_rows,
        row_lower, row_upper, Q_rows, R_rows, c, box_indices,
        box_lower, box_upper);
    CHECK(prefos_create_presolver(&problem, &settings, &presolver) ==
          PREFOS_STATUS_OK);
    CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_REDUCED);
    CHECK(prefos_get_stats(
              presolver)->substituted_bounded_doubletons == 1);
    CHECK(prefos_get_stats(presolver)->merged_parallel_columns == 1);
    CHECK(prefos_get_reduced_problem(presolver)->n == 1);
    CHECK(prefos_postsolve_primal(
              presolver, reduced_x, original_x) == PREFOS_STATUS_OK);
    CHECK(prefos_verify_postsolve_primal(
              presolver, reduced_x, 1e-10, &verification) ==
          PREFOS_STATUS_OK);
    CHECK(verification.passed);
    prefos_free_presolver(presolver);
    return 0;
}

static int test_materialized_doubleton_in_place_fill(void)
{
    double A_values[] = {
        1.0, 1.0,
        1.0, 1.0,
        1.0, 1.0,
        1.0, 1.0};
    int A_columns[] = {
        0, 1,
        0, 2,
        1, 3,
        2, 3};
    int A_rows[] = {0, 2, 4, 6, 8};
    double row_lower[] = {1.0, 2.0, -INFINITY, -INFINITY};
    double row_upper[] = {1.0, 2.0, 10.0, 10.0};
    int Q_rows[] = {0, 0, 0, 0, 0};
    int R_rows[] = {0};
    double c[] = {0.0, 0.0, 0.0, 0.0};
    int box_indices[] = {0, 1, 2, 3};
    double box_lower[] = {-100.0, -100.0, -100.0, -100.0};
    double box_upper[] = {100.0, 100.0, 100.0, 100.0};
    double reduced_x[] = {2.0, 0.0};
    double original_x[4];
    PreFOSProblemData problem;
    PreFOSSettings settings = prefos_strict_settings();
    PreFOSPresolver *presolver = NULL;
    const PreFOSPresolvedProblem *reduced;
    PreFOSPrimalVerification verification;

    settings.remove_redundant_rows = 0;
    settings.free_column_substitution = 0;
    settings.remove_empty_columns = 0;
    settings.singleton_column_reduction = 0;
    settings.bounded_doubleton_substitution = 1;
    settings.linear_propagation = 0;
    settings.parallel_column_reduction = 0;
    settings.dual_fixing = 0;
    settings.remove_redundant_bounds = 0;
    init_linear_problem(
        &problem, 4, 4, 8, A_values, A_columns, A_rows,
        row_lower, row_upper, Q_rows, R_rows, c, box_indices,
        box_lower, box_upper);
    CHECK(prefos_create_presolver(&problem, &settings, &presolver) ==
          PREFOS_STATUS_OK);
    CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_REDUCED);
    reduced = prefos_get_reduced_problem(presolver);
    CHECK(reduced != NULL);
    CHECK(reduced->n == 2 && reduced->A.rows == 2 &&
          reduced->A.nnz == 4);
    CHECK(prefos_get_stats(
              presolver)->substituted_bounded_doubletons == 2);
    CHECK(prefos_postsolve_primal(
              presolver, reduced_x, original_x) == PREFOS_STATUS_OK);
    CHECK(close_to(original_x[0], 0.0));
    CHECK(close_to(original_x[1], 1.0));
    CHECK(close_to(original_x[2], 2.0));
    CHECK(close_to(original_x[3], 0.0));
    CHECK(prefos_verify_postsolve_primal(
              presolver, reduced_x, 1e-10, &verification) ==
          PREFOS_STATUS_OK);
    CHECK(verification.passed);
    prefos_free_presolver(presolver);
    return 0;
}

static int test_materialized_parallel_rows_preserve_fill_targets(void)
{
    double A_values[] = {
        1.0, 1.0, 1.0,
        1.0, 1.0, 1.0,
        1.0, -1.0};
    int A_columns[] = {0, 1, 2, 1, 2, 3, 0, 3};
    int A_rows[] = {0, 3, 6, 8};
    double row_lower[] = {1.0, 0.8, 0.0};
    double row_upper[] = {1.0, 1.2, 0.0};
    int Q_rows[] = {0, 0, 0, 0, 0};
    int R_rows[] = {0};
    double c[] = {0.0, 0.0, 0.0, 0.0};
    int box_indices[] = {0, 1, 2, 3};
    double box_lower[] = {0.0, 0.0, 0.0, 0.0};
    double box_upper[] = {1.0, 0.2, 0.2, 1.0};
    double reduced_x[] = {0.6, 0.2, 0.2};
    PreFOSProblemData problem;
    PreFOSSettings settings = prefos_strict_settings();
    PreFOSPresolver *presolver = NULL;
    const PreFOSPresolvedProblem *reduced;
    PreFOSPrimalVerification verification;

    settings.free_column_substitution = 0;
    settings.singleton_column_reduction = 0;
    settings.bounded_doubleton_substitution = 1;
    settings.linear_propagation = 0;
    settings.parallel_column_reduction = 0;
    settings.dual_fixing = 0;
    settings.remove_redundant_bounds = 0;
    init_linear_problem(
        &problem, 4, 3, 8, A_values, A_columns, A_rows,
        row_lower, row_upper, Q_rows, R_rows, c, box_indices,
        box_lower, box_upper);
    CHECK(prefos_create_presolver(&problem, &settings, &presolver) ==
          PREFOS_STATUS_OK);
    CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_REDUCED);
    reduced = prefos_get_reduced_problem(presolver);
    CHECK(reduced->n == 3 && reduced->A.rows == 1 &&
          reduced->A.nnz == 3);
    CHECK(prefos_get_stats(
              presolver)->substituted_bounded_doubletons == 1);
    CHECK(prefos_get_stats(presolver)->removed_redundant_rows == 1);
    CHECK(prefos_get_stats(presolver)->removed_empty_columns == 0);
    CHECK(prefos_verify_postsolve_primal(
              presolver, reduced_x, 1e-10, &verification) ==
          PREFOS_STATUS_OK);
    CHECK(verification.passed);
    prefos_free_presolver(presolver);
    return 0;
}

static int test_parallel_row_roundoff_gap_is_snapped(void)
{
    double side = 37.644970399062103;
    double A_values[] = {
        6.1541178073836198, -1.0,
        -6.1541178073836198, 1.0};
    int A_columns[] = {0, 1, 0, 1};
    int A_rows[] = {0, 2, 4};
    double row_lower[] = {-INFINITY, -INFINITY};
    double row_upper[] = {side, nextafter(-side, -INFINITY)};
    int Q_rows[] = {0, 0, 0};
    int R_rows[] = {0};
    double c[] = {0.0, 0.0};
    int box_indices[] = {0, 1};
    double box_lower[] = {-INFINITY, -INFINITY};
    double box_upper[] = {INFINITY, INFINITY};
    double reduced_x[] = {0.0, -side};
    PreFOSProblemData problem;
    PreFOSSettings settings = prefos_strict_settings();
    PreFOSPresolver *presolver = NULL;
    const PreFOSPresolvedProblem *reduced;
    PreFOSPrimalVerification verification;

    settings.feasibility_tolerance = 1e-6;
    settings.free_column_substitution = 0;
    settings.remove_empty_columns = 0;
    settings.singleton_column_reduction = 0;
    settings.bounded_doubleton_substitution = 0;
    settings.linear_propagation = 0;
    settings.parallel_column_reduction = 0;
    settings.dual_fixing = 0;
    settings.remove_redundant_bounds = 0;
    init_linear_problem(
        &problem, 2, 2, 4, A_values, A_columns, A_rows,
        row_lower, row_upper, Q_rows, R_rows, c, box_indices,
        box_lower, box_upper);
    CHECK(prefos_create_presolver(&problem, &settings, &presolver) ==
          PREFOS_STATUS_OK);
    CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_REDUCED);
    reduced = prefos_get_reduced_problem(presolver);
    CHECK(reduced != NULL && reduced->A.rows == 1 && reduced->A.nnz == 2);
    CHECK(close_to(reduced->constraint_lower[0], side));
    CHECK(close_to(reduced->constraint_upper[0], side));
    CHECK(prefos_get_stats(presolver)->removed_redundant_rows == 1);
    CHECK(prefos_verify_postsolve_primal(
              presolver, reduced_x, 1e-12, &verification) ==
          PREFOS_STATUS_OK);
    CHECK(verification.passed);
    prefos_free_presolver(presolver);

    row_upper[1] = -side - 1e-8;
    CHECK(prefos_create_presolver(&problem, &settings, &presolver) ==
          PREFOS_STATUS_OK);
    CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_OK);
    reduced = prefos_get_reduced_problem(presolver);
    CHECK(reduced != NULL && reduced->A.rows == 2 && reduced->A.nnz == 4);
    CHECK(prefos_get_stats(presolver)->removed_redundant_rows == 0);
    prefos_free_presolver(presolver);
    return 0;
}

static int test_materialized_closure_preserves_residual_source_exclusion(void)
{
    double A_values[] = {
        1.0, 0.25, -1.0, -0.25,
        1.0, 1.0,
        1.0, 1.0,
        1.0, 1.0};
    int A_columns[] = {
        0, 1, 2, 3,
        4, 5,
        4, 6,
        5, 7};
    int A_rows[] = {0, 4, 6, 8, 10};
    double row_side[] = {0.0, 1.0, 0.5, 0.5};
    double Q_values[] = {1.0, 1.0, 1.0, 1.0, 1.0};
    int Q_columns[] = {0, 1, 3, 6, 7};
    int Q_rows[] = {0, 1, 2, 2, 3, 3, 3, 4, 5};
    int R_rows[] = {0};
    double c[8] = {0.0};
    int box_indices[] = {0, 1, 2, 3, 4, 5, 6, 7};
    double box_lower[] = {4.5, 0.0, 4.5, 0.0, 0.0, 0.0, -1.0, -1.0};
    double box_upper[] = {45.0, 1.0, 46.0, 100.0, 1.0, 1.0, 1.0, 1.0};
    double reduced_x[] = {4.5, 0.0, 0.0, 0.5, 0.0, 0.0};
    double original_x[8];
    PreFOSProblemData problem;
    PreFOSSettings settings = prefos_strict_settings();
    PreFOSPresolver *presolver = NULL;
    PreFOSPrimalVerification verification;

    settings.free_column_substitution = 0;
    settings.bounded_doubleton_substitution = 1;
    settings.linear_propagation = 0;
    settings.parallel_column_reduction = 0;
    settings.dual_fixing = 0;
    settings.remove_redundant_bounds = 0;
    init_linear_problem(
        &problem, 8, 4, 10, A_values, A_columns, A_rows,
        row_side, row_side, Q_rows, R_rows, c, box_indices,
        box_lower, box_upper);
    problem.Q.values = Q_values;
    problem.Q.column_indices = Q_columns;
    problem.Q.nnz = 5;
    CHECK(prefos_create_presolver(&problem, &settings, &presolver) ==
          PREFOS_STATUS_OK);
    CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_REDUCED);
    CHECK(prefos_get_stats(presolver)->residual_row_substitutions == 1);
    CHECK(prefos_get_stats(
              presolver)->substituted_bounded_doubletons == 1);
    CHECK(prefos_get_reduced_problem(presolver)->n == 6);
    CHECK(prefos_verify_postsolve_primal(
              presolver, reduced_x, 1e-10, &verification) ==
          PREFOS_STATUS_OK);
    CHECK(verification.passed);
    CHECK(prefos_postsolve_primal(
              presolver, reduced_x, original_x) == PREFOS_STATUS_OK);
    CHECK(close_to(original_x[2], 4.5));
    CHECK(close_to(original_x[4], 0.5));
    prefos_free_presolver(presolver);
    return 0;
}

static int test_materialized_singleton_row_update_survives_compaction(void)
{
    double A_values[] = {
        1.0, 1.0,
        1.0, 1.0, 1.0,
        1.0, 1.0,
        1.0, 1.0,
        1.0, 1.0};
    int A_columns[] = {
        0, 1,
        0, 1, 4,
        0, 2,
        2, 3,
        4, 5};
    int A_rows[] = {0, 2, 5, 7, 9, 11};
    double row_lower[] = {0.0, 0.0, 0.0, 1.0, 1.0};
    double row_upper[] = {0.0, INFINITY, INFINITY, INFINITY, INFINITY};
    int Q_rows[] = {0, 0, 0, 0, 0, 0, 0};
    int R_rows[] = {0};
    double c[] = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    int box_indices[] = {0, 1, 2, 3, 4, 5};
    double box_lower[] = {-10.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double box_upper[] = {10.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    double reduced_x[] = {1.0, 0.0, 1.0, 0.0};
    double original_x[6];
    PreFOSProblemData problem;
    PreFOSSettings settings = prefos_strict_settings();
    PreFOSPresolver *presolver = NULL;
    const PreFOSPresolvedProblem *reduced;
    PreFOSPrimalVerification verification;

    settings.linear_propagation = 0;
    settings.remove_redundant_rows = 0;
    settings.remove_redundant_bounds = 0;
    settings.free_column_substitution = 0;
    settings.parallel_column_reduction = 0;
    settings.dual_fixing = 0;
    init_linear_problem(
        &problem, 6, 5, 11, A_values, A_columns, A_rows,
        row_lower, row_upper, Q_rows, R_rows, c, box_indices,
        box_lower, box_upper);

    CHECK(prefos_create_presolver(&problem, &settings, &presolver) ==
          PREFOS_STATUS_OK);
    CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_REDUCED);
    reduced = prefos_get_reduced_problem(presolver);
    CHECK(reduced != NULL);
    CHECK(prefos_get_stats(
              presolver)->substituted_bounded_doubletons == 1);
    CHECK(prefos_get_stats(presolver)->tightened_singleton_rows == 1);
    CHECK(reduced->n == 4 && reduced->A.rows == 2 &&
          reduced->A.nnz == 4);
    CHECK(prefos_postsolve_primal(presolver, reduced_x, original_x) ==
          PREFOS_STATUS_OK);
    CHECK(close_to(original_x[0], -1.0));
    CHECK(close_to(original_x[1], 1.0));
    CHECK(close_to(original_x[2], 1.0));
    CHECK(close_to(original_x[3], 0.0));
    CHECK(close_to(original_x[4], 1.0));
    CHECK(close_to(original_x[5], 0.0));
    CHECK(prefos_verify_postsolve_primal(
              presolver, reduced_x, 1e-10, &verification) ==
          PREFOS_STATUS_OK);
    CHECK(verification.passed);
    prefos_free_presolver(presolver);
    return 0;
}

static int test_substitution_targets_survive_column_reductions(void)
{
    double A_values[] = {
        1.0, 1.0, 1.0,
        1.0, -1.0,
        1.0, -1.0,
        1.0, -1.0};
    int A_columns[] = {0, 1, 2, 0, 3, 1, 4, 2, 5};
    int A_rows[] = {0, 3, 5, 7, 9};
    double row_side[] = {1.0, 0.0, 0.0, 0.0};
    int Q_rows[] = {0, 0, 0, 0, 0, 0, 0};
    int R_rows[] = {0};
    double c[] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    int box_indices[] = {0, 1, 2, 3, 4, 5};
    double box_lower[] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double box_upper[] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    double reduced_x[] = {1.0, 0.0, 0.0};
    double original_x[6];
    PreFOSProblemData problem;
    PreFOSSettings settings = prefos_strict_settings();
    PreFOSPresolver *presolver = NULL;
    const PreFOSPresolvedProblem *reduced;
    PreFOSPrimalVerification verification;

    settings.remove_redundant_rows = 0;
    settings.free_column_substitution = 0;
    settings.singleton_column_reduction = 0;
    settings.bounded_doubleton_substitution = 1;
    settings.linear_propagation = 0;
    settings.parallel_column_reduction = 0;
    settings.remove_redundant_bounds = 0;
    init_linear_problem(
        &problem, 6, 4, 9, A_values, A_columns, A_rows, row_side, row_side,
        Q_rows, R_rows, c, box_indices, box_lower, box_upper);
    CHECK(prefos_create_presolver(&problem, &settings, &presolver) ==
          PREFOS_STATUS_OK);
    CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_REDUCED);
    reduced = prefos_get_reduced_problem(presolver);
    CHECK(reduced != NULL);
    CHECK(reduced->n == 3 && reduced->A.rows == 1 && reduced->A.nnz == 3);
    CHECK(prefos_get_stats(presolver)->substituted_bounded_doubletons == 3);
    CHECK(prefos_get_stats(presolver)->removed_empty_columns == 0);
    CHECK(prefos_postsolve_primal(presolver, reduced_x, original_x) ==
          PREFOS_STATUS_OK);
    CHECK(close_to(original_x[0], 1.0));
    CHECK(close_to(original_x[1], 0.0));
    CHECK(close_to(original_x[2], 0.0));
    CHECK(close_to(original_x[3], 1.0));
    CHECK(close_to(original_x[4], 0.0));
    CHECK(close_to(original_x[5], 0.0));
    CHECK(prefos_verify_postsolve_primal(
              presolver, reduced_x, 1e-10, &verification) ==
          PREFOS_STATUS_OK);
    CHECK(verification.passed);
    prefos_free_presolver(presolver);
    return 0;
}

static PreFOSSettings parallel_only_settings(void)
{
    PreFOSSettings settings = prefos_strict_settings();
    settings.remove_redundant_rows = 0;
    settings.free_column_substitution = 0;
    settings.remove_empty_columns = 0;
    settings.singleton_column_reduction = 0;
    settings.bounded_doubleton_substitution = 0;
    settings.linear_propagation = 0;
    settings.remove_redundant_bounds = 0;
    return settings;
}

static int test_parallel_column_reductions(void)
{
    {
        double A_values[] = {1, -1, 1, 1, 1, 2, 2, -1};
        int A_columns[] = {0, 1, 1, 2, 3, 1, 2, 3};
        int A_rows[] = {0, 2, 5, 8};
        double row_lower[] = {0.0, -100.0, -100.0};
        double row_upper[] = {0.0, 100.0, 100.0};
        int Q_rows[] = {0, 0, 0, 0, 0};
        int R_rows[] = {0};
        double c[] = {1.0, 0.0, 1.0, 0.0};
        int box_indices[] = {0, 1, 2, 3};
        double box_lower[] = {-INFINITY, -1.0, -1.0, -10.0};
        double box_upper[] = {INFINITY, 1.0, 1.0, 10.0};
        double reduced_x[] = {0.0, 0.0};
        double original_x[4];
        PreFOSProblemData problem;
        PreFOSSettings settings = parallel_only_settings();
        PreFOSPresolver *presolver = NULL;
        const PreFOSPresolvedProblem *reduced;

        settings.free_column_substitution = 1;
        init_linear_problem(&problem, 4, 3, 8, A_values, A_columns, A_rows,
                            row_lower, row_upper, Q_rows, R_rows, c, box_indices,
                            box_lower, box_upper);
        CHECK(prefos_create_presolver(&problem, &settings, &presolver) ==
              PREFOS_STATUS_OK);
        CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_REDUCED);
        reduced = prefos_get_reduced_problem(presolver);
        CHECK(reduced->n == 2);
        CHECK(prefos_get_stats(presolver)->substituted_free_variables == 1);
        CHECK(prefos_get_stats(presolver)->merged_parallel_columns == 1);
        CHECK(close_to(reduced->c[0], 1.0));
        CHECK(close_to(reduced->box_lower[0], -2.0));
        CHECK(close_to(reduced->box_upper[0], 2.0));
        CHECK(prefos_postsolve_primal(presolver, reduced_x, original_x) ==
              PREFOS_STATUS_OK);
        CHECK(close_to(original_x[0], 0.0));
        prefos_free_presolver(presolver);
    }
    {
        double A_values[] = {1, 1, 1, 1, 2, 2, 2, -1};
        int A_columns[] = {0, 1, 2, 3, 0, 1, 2, 3};
        int A_rows[] = {0, 4, 8};
        double row_lower[] = {-100.0, -100.0};
        double row_upper[] = {100.0, 100.0};
        int Q_rows[] = {0, 0, 0, 0, 0};
        int R_rows[] = {0};
        double c[] = {1.0, 1.0, 1.0, 0.0};
        int box_indices[] = {0, 1, 2, 3};
        double box_lower[] = {-1.0, -1.0, -1.0, -10.0};
        double box_upper[] = {1.0, 1.0, 1.0, 10.0};
        double reduced_x[] = {0.0, 0.0};
        double original_x[4];
        PreFOSProblemData problem;
        PreFOSSettings settings = parallel_only_settings();
        PreFOSPresolver *presolver = NULL;
        const PreFOSPresolvedProblem *reduced;
        PreFOSPrimalVerification verification;

        settings.structural_reductions_gpu = 1;
        init_linear_problem(&problem, 4, 2, 8, A_values, A_columns, A_rows,
                            row_lower, row_upper, Q_rows, R_rows, c, box_indices,
                            box_lower, box_upper);
        CHECK(prefos_create_presolver(&problem, &settings, &presolver) ==
              PREFOS_STATUS_OK);
        CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_REDUCED);
        reduced = prefos_get_reduced_problem(presolver);
        CHECK(reduced->n == 2 && reduced->A.nnz == 4);
        CHECK(prefos_get_stats(presolver)->merged_parallel_columns == 2);
        if (prefos_get_stats(presolver)->column_csc_gpu_builds > 0)
        {
            CHECK(prefos_get_stats(presolver)->parallel_column_gpu_passes > 0);
            CHECK(prefos_get_stats(presolver)->parallel_column_gpu_fallbacks == 0);
        }
        else
            CHECK(prefos_get_stats(presolver)->parallel_column_gpu_passes == 0);
        CHECK(close_to(reduced->box_lower[0], -3.0));
        CHECK(close_to(reduced->box_upper[0], 3.0));
        CHECK(prefos_postsolve_primal(presolver, reduced_x, original_x) ==
              PREFOS_STATUS_OK);
        CHECK(prefos_verify_postsolve_primal(presolver, reduced_x, 1e-10,
                                             &verification) == PREFOS_STATUS_OK);
        CHECK(verification.passed);
        prefos_free_presolver(presolver);
    }
    {
        double A_values[] = {1, 1, 1, 2, 2, 2};
        int A_columns[] = {0, 1, 2, 0, 1, 2};
        int A_rows[] = {0, 3, 6};
        double row_lower[] = {-100.0, -100.0};
        double row_upper[] = {100.0, 100.0};
        int Q_rows[] = {0, 0, 0, 0};
        int R_rows[] = {0};
        double c[] = {0.0, 1.0, 1.0};
        int box_indices[] = {0, 1, 2};
        double box_lower[] = {-1.0, -1.0, -1.0};
        double box_upper[] = {1.0, 1.0, 1.0};
        double reduced_x[] = {0.0, 0.0};
        double original_x[3];
        PreFOSProblemData problem;
        PreFOSSettings settings = parallel_only_settings();
        PreFOSPresolver *presolver = NULL;
        PreFOSPrimalVerification verification;

        init_linear_problem(
            &problem, 3, 2, 6, A_values, A_columns, A_rows,
            row_lower, row_upper, Q_rows, R_rows, c,
            box_indices, box_lower, box_upper);
        CHECK(prefos_create_presolver(
                  &problem, &settings, &presolver) ==
              PREFOS_STATUS_OK);
        CHECK(prefos_run_presolve(presolver) ==
              PREFOS_STATUS_REDUCED);
        CHECK(prefos_get_reduced_problem(presolver)->n == 2);
        CHECK(prefos_get_stats(
                  presolver)->merged_parallel_columns == 1);
        CHECK(prefos_postsolve_primal(
                  presolver, reduced_x, original_x) ==
              PREFOS_STATUS_OK);
        CHECK(prefos_verify_postsolve_primal(
                  presolver, reduced_x, 1e-10, &verification) ==
              PREFOS_STATUS_OK);
        CHECK(verification.passed);
        prefos_free_presolver(presolver);
    }
    {
        double A_values[] = {1, 1, 1, 2, 2, -1};
        int A_columns[] = {0, 1, 2, 0, 1, 2};
        int A_rows[] = {0, 3, 6};
        double row_side[] = {0.0, 0.0};
        int Q_rows[] = {0, 0, 0, 0};
        int R_rows[] = {0};
        double c[] = {0.0, 1.0, 0.0};
        int box_indices[] = {0, 1, 2};
        double box_lower[] = {-INFINITY, -INFINITY, -10.0};
        double box_upper[] = {2.0, INFINITY, 10.0};
        double reduced_x[] = {-2.0, 0.0};
        double original_x[3];
        PreFOSProblemData problem;
        PreFOSSettings settings = parallel_only_settings();
        PreFOSPresolver *presolver = NULL;

        init_linear_problem(&problem, 3, 2, 6, A_values, A_columns, A_rows,
                            row_side, row_side, Q_rows, R_rows, c, box_indices,
                            box_lower, box_upper);
        CHECK(prefos_create_presolver(&problem, &settings, &presolver) ==
              PREFOS_STATUS_OK);
        CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_REDUCED);
        CHECK(prefos_get_reduced_problem(presolver)->n == 2);
        CHECK(prefos_get_stats(presolver)->dual_fixed_columns == 1);
        CHECK(prefos_get_stats(presolver)->merged_parallel_columns == 0);
        CHECK(prefos_postsolve_primal(presolver, reduced_x, original_x) ==
              PREFOS_STATUS_OK);
        CHECK(close_to(original_x[0], 2.0));
        CHECK(close_to(original_x[1], -2.0));
        prefos_free_presolver(presolver);
    }
    {
        double A_values[] = {1.0, 1.0};
        int A_columns[] = {0, 1};
        int A_rows[] = {0, 2};
        double row_side[] = {0.0};
        int Q_rows[] = {0, 0, 0};
        int R_rows[] = {0};
        double c[] = {0.0, 1.0};
        int box_indices[] = {0, 1};
        double box_lower[] = {-INFINITY, -INFINITY};
        double box_upper[] = {INFINITY, INFINITY};
        PreFOSProblemData problem;
        PreFOSSettings settings = parallel_only_settings();
        PreFOSPresolver *presolver = NULL;

        init_linear_problem(&problem, 2, 1, 2, A_values, A_columns, A_rows,
                            row_side, row_side, Q_rows, R_rows, c, box_indices,
                            box_lower, box_upper);
        CHECK(prefos_create_presolver(&problem, &settings, &presolver) ==
              PREFOS_STATUS_OK);
        CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_PRIMAL_UNBOUNDED);
        prefos_free_presolver(presolver);
    }
    prefos_gpu_release_cache();
    return 0;
}

static int test_parallel_column_cache_after_row_removal(void)
{
    const size_t n = 65536;
    double A_values[] = {1.0, 1.0, 1.0, 1.0};
    int A_columns[] = {0, 1, 0, 2};
    int A_rows[] = {0, 2, 4};
    double row_lower[] = {0.0, -10.0};
    double row_upper[] = {0.0, 10.0};
    int R_rows[] = {0};
    int *Q_rows = (int *) calloc(n + 1, sizeof(int));
    int *box_indices = (int *) malloc(n * sizeof(int));
    double *box_lower = (double *) malloc(n * sizeof(double));
    double *box_upper = (double *) malloc(n * sizeof(double));
    double *c = (double *) calloc(n, sizeof(double));
    PreFOSProblemData problem;
    PreFOSSettings settings = parallel_only_settings();
    PreFOSPresolver *presolver = NULL;
    size_t column;

    CHECK(Q_rows != NULL && box_indices != NULL &&
          box_lower != NULL && box_upper != NULL && c != NULL);
    for (column = 0; column < n; ++column)
    {
        box_indices[column] = (int) column;
        box_lower[column] = -1.0;
        box_upper[column] = 1.0;
    }
    memset(&problem, 0, sizeof(problem));
    problem.n = n;
    problem.A = (PreFOSCsrMatrix){
        2, n, 4, A_values, A_columns, A_rows};
    problem.constraint_lower = row_lower;
    problem.constraint_upper = row_upper;
    problem.Q = (PreFOSCsrMatrix){
        n, n, 0, NULL, NULL, Q_rows};
    problem.q_storage = PREFOS_Q_UPPER_TRIANGULAR;
    problem.R = (PreFOSCsrMatrix){
        0, n, 0, NULL, NULL, R_rows};
    problem.c = c;
    problem.n_box = n;
    problem.box_indices = box_indices;
    problem.box_lower = box_lower;
    problem.box_upper = box_upper;
    settings.remove_redundant_rows = 1;

    CHECK(prefos_create_presolver(
              &problem, &settings, &presolver) ==
          PREFOS_STATUS_OK);
    CHECK(prefos_run_presolve(presolver) ==
          PREFOS_STATUS_REDUCED);
    CHECK(prefos_get_stats(presolver)->removed_redundant_rows >= 1);
    CHECK(prefos_get_stats(presolver)->merged_parallel_columns == 1);
    prefos_free_presolver(presolver);
    free(Q_rows);
    free(box_indices);
    free(box_lower);
    free(box_upper);
    free(c);
    return 0;
}

static int test_singleton_rows_expose_parallel_columns(void)
{
    double A_values[] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    int A_columns[] = {0, 2, 1, 3, 2, 3};
    int A_rows[] = {0, 2, 4, 6};
    double row_lower[] = {-INFINITY, -INFINITY, -10.0};
    double row_upper[] = {0.0, 0.0, 10.0};
    int Q_rows[] = {0, 0, 0, 0, 0};
    int R_rows[] = {0};
    double c[] = {-1.0, -1.0, 0.0, 0.0};
    int box_indices[] = {0, 1, 2, 3};
    double box_lower[] = {-1.0, -1.0, -1.0, -1.0};
    double box_upper[] = {1.0, 1.0, 1.0, 1.0};
    PreFOSProblemData problem;
    PreFOSSettings settings = parallel_only_settings();
    PreFOSPresolver *presolver = NULL;

    settings.singleton_column_reduction = 1;
    init_linear_problem(
        &problem, 4, 3, 6, A_values, A_columns, A_rows,
        row_lower, row_upper, Q_rows, R_rows, c,
        box_indices, box_lower, box_upper);
    CHECK(prefos_create_presolver(
              &problem, &settings, &presolver) ==
          PREFOS_STATUS_OK);
    CHECK(prefos_run_presolve(presolver) ==
          PREFOS_STATUS_REDUCED);
    CHECK(prefos_get_stats(
              presolver)->removed_singleton_columns >= 2);
    CHECK(prefos_get_stats(
              presolver)->merged_parallel_columns >= 1);
    prefos_free_presolver(presolver);
    return 0;
}

static int test_deferred_activity_wave_consistency(void)
{
    const size_t rows = 65536;
    const size_t columns = 16384;
    const size_t active_columns = 16;
    const size_t nnz = rows * active_columns;
    double *A_values = (double *) malloc(nnz * sizeof(double));
    int *A_columns = (int *) malloc(nnz * sizeof(int));
    int *A_rows = (int *) malloc((rows + 1) * sizeof(int));
    double *row_lower = (double *) malloc(rows * sizeof(double));
    double *row_upper = (double *) malloc(rows * sizeof(double));
    int *Q_rows = (int *) calloc(columns + 1, sizeof(int));
    double *c = (double *) calloc(columns, sizeof(double));
    int *box_indices = (int *) malloc(columns * sizeof(int));
    double *box_lower = (double *) malloc(columns * sizeof(double));
    double *box_upper = (double *) malloc(columns * sizeof(double));
    int R_rows[] = {0};
    PreFOSProblemData problem;
    PreFOSSettings settings = prefos_default_settings();
    PreFOSPresolver *presolver = NULL;
    PreFOSStatus status;
    size_t row, column;

    CHECK(A_values != NULL && A_columns != NULL && A_rows != NULL &&
          row_lower != NULL && row_upper != NULL && Q_rows != NULL &&
          c != NULL && box_indices != NULL && box_lower != NULL &&
          box_upper != NULL);
    for (row = 0; row < rows; ++row)
    {
        A_rows[row] = (int) (row * active_columns);
        row_lower[row] = -INFINITY;
        row_upper[row] = 0.0;
        for (column = 0; column < active_columns; ++column)
        {
            size_t position = row * active_columns + column;
            A_values[position] = 1.0;
            A_columns[position] = (int) column;
        }
    }
    A_rows[rows] = (int) nnz;
    for (column = 0; column < columns; ++column)
    {
        box_indices[column] = (int) column;
        box_lower[column] = 0.0;
        box_upper[column] = INFINITY;
    }
    init_linear_problem(
        &problem, columns, rows, nnz, A_values, A_columns, A_rows,
        row_lower, row_upper, Q_rows, R_rows, c, box_indices,
        box_lower, box_upper);
    settings.remove_redundant_rows = 0;
    settings.free_column_substitution = 0;
    settings.cone_propagation = 0;
    settings.remove_empty_columns = 0;
    settings.singleton_column_reduction = 0;
    settings.bounded_doubleton_substitution = 0;
    settings.dual_fixing = 0;
    settings.parallel_column_reduction = 0;

    CHECK(prefos_create_presolver(
              &problem, &settings, &presolver) ==
          PREFOS_STATUS_OK);
    status = prefos_run_presolve(presolver);
    CHECK(status == PREFOS_STATUS_REDUCED);
    CHECK(prefos_get_stats(presolver)->propagated_box_bounds >=
          active_columns);
    CHECK(prefos_get_stats(presolver)->linear_event_rounds >= 1);
    CHECK(prefos_get_stats(presolver)->linear_full_scan_rounds == 0);
    CHECK(prefos_get_stats(presolver)->linear_stale_stops >= 1);
    prefos_free_presolver(presolver);
    free(A_values);
    free(A_columns);
    free(A_rows);
    free(row_lower);
    free(row_upper);
    free(Q_rows);
    free(c);
    free(box_indices);
    free(box_lower);
    free(box_upper);
    return 0;
}

static int test_redundant_sides_and_bounds(void)
{
    {
        double A_values[] = {1.0, 1.0};
        int A_columns[] = {0, 1};
        int A_rows[] = {0, 2};
        double row_lower[] = {-1.0};
        double row_upper[] = {0.5};
        int Q_rows[] = {0, 0, 0};
        int R_rows[] = {0};
        double c[] = {0.0, 0.0};
        int box_indices[] = {0, 1};
        double box_lower[] = {0.0, 0.0};
        double box_upper[] = {1.0, 1.0};
        PreFOSProblemData problem;
        PreFOSSettings settings = prefos_strict_settings();
        PreFOSPresolver *presolver = NULL;
        const PreFOSPresolvedProblem *reduced;

        settings.linear_propagation = 0;
        settings.parallel_column_reduction = 0;
        settings.dual_fixing = 0;
        init_linear_problem(&problem, 2, 1, 2, A_values, A_columns, A_rows,
                            row_lower, row_upper, Q_rows, R_rows, c, box_indices,
                            box_lower, box_upper);
        CHECK(prefos_create_presolver(&problem, &settings, &presolver) ==
              PREFOS_STATUS_OK);
        CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_REDUCED);
        reduced = prefos_get_reduced_problem(presolver);
        CHECK(isinf(reduced->constraint_lower[0]) &&
              reduced->constraint_lower[0] < 0.0);
        CHECK(close_to(reduced->constraint_upper[0], 0.5));
        CHECK(prefos_get_stats(presolver)->removed_redundant_row_lower_sides == 1);
        prefos_free_presolver(presolver);
    }
    {
        double A_values[] = {-1.0, 1.0};
        int A_columns[] = {0, 1};
        int A_rows[] = {0, 2};
        double row_lower[] = {-INFINITY};
        double row_upper[] = {0.0};
        int Q_rows[] = {0, 0, 0};
        int R_rows[] = {0};
        double c[] = {0.0, 0.0};
        int box_indices[] = {0, 1};
        double box_lower[] = {0.0, 0.0};
        double box_upper[] = {INFINITY, INFINITY};
        PreFOSProblemData problem;
        PreFOSSettings settings = prefos_strict_settings();
        PreFOSPresolver *presolver = NULL;
        const PreFOSPresolvedProblem *reduced;

        settings.propagated_bound_policy =
            PREFOS_PROPAGATED_BOUND_POLICY_INTERIOR_POINT;
        settings.singleton_column_reduction = 0;
        settings.parallel_column_reduction = 0;
        settings.dual_fixing = 0;
        init_linear_problem(&problem, 2, 1, 2, A_values, A_columns, A_rows,
                            row_lower, row_upper, Q_rows, R_rows, c, box_indices,
                            box_lower, box_upper);
        CHECK(prefos_create_presolver(&problem, &settings, &presolver) ==
              PREFOS_STATUS_OK);
        CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_REDUCED);
        reduced = prefos_get_reduced_problem(presolver);
        CHECK(isinf(reduced->box_lower[0]) && reduced->box_lower[0] < 0.0);
        CHECK(close_to(reduced->box_lower[1], 0.0));
        CHECK(prefos_get_stats(presolver)->removed_redundant_box_lower_bounds == 1);
        prefos_free_presolver(presolver);

        settings.propagated_bound_policy =
            PREFOS_PROPAGATED_BOUND_POLICY_FIRST_ORDER;
        CHECK(prefos_create_presolver(&problem, &settings, &presolver) ==
              PREFOS_STATUS_OK);
        CHECK(prefos_run_presolve(presolver) != PREFOS_STATUS_INVALID_ARGUMENT);
        CHECK(close_to(prefos_get_reduced_problem(presolver)->box_lower[0], 0.0));
        prefos_free_presolver(presolver);
    }
    return 0;
}

int main(void)
{
    if (test_empty_columns()) return 1;
    if (test_quadratic_empty_column_is_protected()) return 1;
    if (test_dual_fixing_with_structural_gpu_enabled()) return 1;
    if (test_zero_objective_infinite_dual_fixing()) return 1;
    if (test_singleton_column_substitution()) return 1;
    if (test_singleton_residual_row_preserves_fixed_shift()) return 1;
    if (test_singleton_equality_retains_two_sided_box()) return 1;
    if (test_singleton_implied_free_avoids_large_bound_cancellation()) return 1;
    if (test_one_sided_singleton_reductions()) return 1;
    if (test_bounded_doubleton_substitution()) return 1;
    if (test_materialized_propagation_dual_source()) return 1;
    if (test_budgeted_materialized_propagation_closure()) return 1;
    if (test_bounded_doubleton_skips_dirty_rows()) return 1;
    if (test_bounded_doubleton_has_zero_net_fill()) return 1;
    if (test_bounded_doubleton_exposes_parallel_columns()) return 1;
    if (test_materialized_doubleton_in_place_fill()) return 1;
    if (test_materialized_parallel_rows_preserve_fill_targets()) return 1;
    if (test_parallel_row_roundoff_gap_is_snapped()) return 1;
    if (test_materialized_closure_preserves_residual_source_exclusion()) return 1;
    if (test_materialized_singleton_row_update_survives_compaction()) return 1;
    if (test_substitution_targets_survive_column_reductions()) return 1;
    if (test_parallel_column_reductions()) return 1;
    if (test_parallel_column_cache_after_row_removal()) return 1;
    if (test_singleton_rows_expose_parallel_columns()) return 1;
    if (test_deferred_activity_wave_consistency()) return 1;
    if (test_redundant_sides_and_bounds()) return 1;
    printf("All PreFOS column reduction tests passed!\n");
    return 0;
}
