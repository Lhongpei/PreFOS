/*
 * Copyright 2026 Hongpei Li
 * SPDX-License-Identifier: Apache-2.0
 */

#include <PreFOS/PreFOS.h>

#include <math.h>
#include <stdio.h>
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

static int test_dual_fixing_and_gpu_fallback(void)
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
    double original_x[1];
    int gpu_available = prefos_gpu_warmup();

    settings.structural_reductions_gpu = 1;
    init_linear_problem(&problem, 1, 1, 1, A_values, A_columns, A_rows,
                        row_lower, row_upper, Q_rows, R_rows, c, box_indices,
                        box_lower, box_upper);
    CHECK(prefos_create_presolver(&problem, &settings, &presolver) ==
          PREFOS_STATUS_OK);
    CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_REDUCED);
    stats = prefos_get_stats(presolver);
    CHECK(stats->dual_fixed_columns == 1);
    CHECK(stats->structural_gpu_passes == (size_t) gpu_available);
    CHECK(stats->structural_gpu_fallbacks == (size_t) !gpu_available);
    CHECK(prefos_postsolve_primal(presolver, NULL, original_x) == PREFOS_STATUS_OK);
    CHECK(close_to(original_x[0], 0.0));
    prefos_free_presolver(presolver);

    box_lower[0] = -INFINITY;
    CHECK(prefos_create_presolver(&problem, &settings, &presolver) ==
          PREFOS_STATUS_OK);
    CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_PRIMAL_UNBOUNDED);
    prefos_free_presolver(presolver);
    prefos_gpu_release_cache();
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
    double reduced_x[] = {2.0};
    double original_x[2];
    PreFOSProblemData problem;
    PreFOSPresolver *presolver = NULL;
    const PreFOSPresolvedProblem *reduced;
    PreFOSPrimalVerification verification;

    init_linear_problem(&problem, 2, 1, 2, A_values, A_columns, A_rows,
                        row_side, row_side, Q_rows, R_rows, c, box_indices,
                        box_lower, box_upper);
    CHECK(prefos_create_presolver(&problem, NULL, &presolver) == PREFOS_STATUS_OK);
    CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_REDUCED);
    reduced = prefos_get_reduced_problem(presolver);
    CHECK(reduced->n == 1 && reduced->A.rows == 0);
    CHECK(close_to(reduced->objective_offset, 2.0));
    CHECK(close_to(reduced->c[0], -1.0));
    CHECK(prefos_get_stats(presolver)->removed_singleton_columns == 1);
    CHECK(prefos_postsolve_primal(presolver, reduced_x, original_x) ==
          PREFOS_STATUS_OK);
    CHECK(close_to(original_x[0], 0.0) && close_to(original_x[1], 2.0));
    CHECK(prefos_verify_postsolve_primal(presolver, reduced_x, 1e-10,
                                         &verification) == PREFOS_STATUS_OK);
    CHECK(verification.passed);
    prefos_free_presolver(presolver);
    return 0;
}

static int test_one_sided_singleton_reductions(void)
{
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
        PreFOSPresolver *presolver = NULL;
        const PreFOSPresolvedProblem *reduced;
        PreFOSPostsolveKKTVerification verification;

        init_linear_problem(&problem, 3, 1, 3, A_values, A_columns, A_rows,
                            row_side, row_side, Q_rows, R_rows, c, box_indices,
                            box_lower, box_upper);
        CHECK(prefos_create_presolver(&problem, NULL, &presolver) ==
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
        PreFOSPresolver *presolver = NULL;
        const PreFOSPresolvedProblem *reduced;
        double reduced_x[] = {3.0, 1.0};
        double reduced_y[] = {-1.0};
        double reduced_z[] = {0.0, 1.0};
        double original_x[2], original_y[1], original_z[2];
        PreFOSPostsolveKKTVerification verification;

        init_linear_problem(&problem, 2, 1, 2, A_values, A_columns, A_rows,
                            row_lower, row_upper, Q_rows, R_rows, c, box_indices,
                            box_lower, box_upper);
        CHECK(prefos_create_presolver(&problem, NULL, &presolver) ==
              PREFOS_STATUS_OK);
        CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_REDUCED);
        reduced = prefos_get_reduced_problem(presolver);
        CHECK(reduced->A.rows == 1);
        CHECK(close_to(reduced->constraint_lower[0], 4.0));
        CHECK(close_to(reduced->constraint_upper[0], 4.0));
        CHECK(prefos_get_stats(presolver)->tightened_singleton_rows == 1);
        CHECK(prefos_postsolve_primal_dual(
                  presolver, reduced_x, reduced_y, reduced_z, 1e-10,
                  original_x, original_y, original_z) == PREFOS_STATUS_OK);
        CHECK(close_to(original_y[0], -1.0));
        CHECK(close_to(original_z[0], 0.0));
        CHECK(close_to(original_z[1], 1.0));
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
        double row_lower[] = {-INFINITY};
        double row_upper[] = {4.0};
        int Q_rows[] = {0, 0, 0};
        int R_rows[] = {0};
        double c[] = {-1.0, 0.0};
        int box_indices[] = {0, 1};
        double box_lower[] = {0.0, 0.0};
        double box_upper[] = {10.0, 1.0};
        double reduced_x[] = {4.0, 0.0};
        double reduced_y[] = {1.0};
        double reduced_z[] = {0.0, -1.0};
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
        CHECK(close_to(prefos_get_reduced_problem(presolver)
                           ->constraint_lower[0],
                       4.0));
        CHECK(close_to(prefos_get_reduced_problem(presolver)
                           ->constraint_upper[0],
                       4.0));
        CHECK(prefos_postsolve_primal_dual(
                  presolver, reduced_x, reduced_y, reduced_z, 1e-10,
                  original_x, original_y, original_z) == PREFOS_STATUS_OK);
        CHECK(close_to(original_y[0], 1.0));
        CHECK(close_to(original_z[0], 0.0));
        CHECK(close_to(original_z[1], -1.0));
        CHECK(prefos_verify_postsolve_kkt(
                  presolver, reduced_x, reduced_y, reduced_z, 1e-10,
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
    double reduced_x[] = {5.0};
    double original_x[2];
    double reduced_z[] = {3.0};
    double original_y[1];
    double original_z[2];
    PreFOSProblemData problem;
    PreFOSSettings settings = prefos_strict_settings();
    PreFOSPresolver *presolver = NULL;
    const PreFOSPresolvedProblem *reduced;
    PreFOSPostsolveKKTVerification verification;

    settings.singleton_column_reduction = 0;
    settings.bounded_doubleton_substitution = 1;
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
    CHECK(close_to(reduced->objective_offset, 5.0));
    CHECK(close_to(reduced->c[0], -3.0));
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
        double positive_reduced_x[] = {1.0};
        double positive_reduced_z[] = {3.0};
        double positive_original_x[2], positive_original_y[1];
        double positive_original_z[2];
        PreFOSProblemData positive_problem;
        PreFOSSettings positive_settings = prefos_strict_settings();
        PreFOSPresolver *positive_presolver = NULL;
        PreFOSPostsolveKKTVerification positive_verification;

        positive_settings.singleton_column_reduction = 0;
        positive_settings.bounded_doubleton_substitution = 1;
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
        double target_reduced_x[] = {0.0};
        double target_reduced_z[] = {-1.0};
        double target_original_x[2], target_original_y[1], target_original_z[2];
        PreFOSProblemData target_problem;
        PreFOSSettings target_settings = prefos_strict_settings();
        PreFOSPresolver *target_presolver = NULL;
        PreFOSPostsolveKKTVerification target_verification;

        target_settings.singleton_column_reduction = 0;
        target_settings.bounded_doubleton_substitution = 1;
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

static PreFOSSettings parallel_only_settings(void)
{
    PreFOSSettings settings = prefos_strict_settings();
    settings.remove_redundant_rows = 0;
    settings.free_column_substitution = 0;
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

        init_linear_problem(&problem, 4, 2, 8, A_values, A_columns, A_rows,
                            row_lower, row_upper, Q_rows, R_rows, c, box_indices,
                            box_lower, box_upper);
        CHECK(prefos_create_presolver(&problem, &settings, &presolver) ==
              PREFOS_STATUS_OK);
        CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_REDUCED);
        reduced = prefos_get_reduced_problem(presolver);
        CHECK(reduced->n == 2 && reduced->A.nnz == 4);
        CHECK(prefos_get_stats(presolver)->merged_parallel_columns == 2);
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
    if (test_dual_fixing_and_gpu_fallback()) return 1;
    if (test_singleton_column_substitution()) return 1;
    if (test_one_sided_singleton_reductions()) return 1;
    if (test_bounded_doubleton_substitution()) return 1;
    if (test_parallel_column_reductions()) return 1;
    if (test_redundant_sides_and_bounds()) return 1;
    printf("All PreFOS column reduction tests passed!\n");
    return 0;
}
