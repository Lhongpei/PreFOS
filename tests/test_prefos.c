/*
 * Copyright 2026 Hongpei Li
 * SPDX-License-Identifier: Apache-2.0
 */

#include <PreFOS/PreFOS.h>

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#define CHECK(condition)                                                            \
    do                                                                              \
    {                                                                               \
        if (!(condition))                                                           \
        {                                                                           \
            fprintf(stderr, "check failed at %s:%d: %s\n", __FILE__, __LINE__,      \
                    #condition);                                                    \
            return 1;                                                               \
        }                                                                           \
    } while (0)

static int close_to(double left, double right)
{
    return fabs(left - right) <= 1e-12;
}

static int test_fixed_variable_and_objective_update(void)
{
    double A_values[] = {2.0, 1.0};
    int A_columns[] = {0, 1};
    int A_rows[] = {0, 2, 2};
    double constraint_lower[] = {5.0, -1.0};
    double constraint_upper[] = {7.0, 1.0};

    double Q_values[] = {4.0, 1.0, 2.0, 3.0, 5.0};
    int Q_columns[] = {0, 1, 2, 1, 2};
    int Q_rows[] = {0, 3, 4, 5};

    double R_values[] = {1.0, 2.0, -1.0, 3.0};
    int R_columns[] = {0, 1, 1, 2};
    int R_rows[] = {0, 2, 4};
    double D[] = {2.0, 4.0};
    double c[] = {5.0, 6.0, 7.0};

    int box_indices[] = {0};
    double box_lower[] = {2.0};
    double box_upper[] = {2.0};
    int cone_indices[] = {1, 2};
    PreFOSConeBlock cone = {PREFOS_CONE_SECOND_ORDER, 2, 0, cone_indices};

    PreFOSProblemData problem;
    PreFOSPresolver *presolver = NULL;
    const PreFOSPresolvedProblem *reduced;
    const PreFOSStats *stats;
    double reduced_x[] = {2.5, 1.5};
    double infeasible_reduced_x[] = {1.5, 2.5};
    double original_x[3];
    PreFOSPrimalVerification verification;
    PreFOSStatus status;

    memset(&problem, 0, sizeof(problem));
    problem.n = 3;
    problem.A = (PreFOSCsrMatrix){2, 3, 2, A_values, A_columns, A_rows};
    problem.constraint_lower = constraint_lower;
    problem.constraint_upper = constraint_upper;
    problem.Q = (PreFOSCsrMatrix){3, 3, 5, Q_values, Q_columns, Q_rows};
    problem.q_storage = PREFOS_Q_UPPER_TRIANGULAR;
    problem.R = (PreFOSCsrMatrix){2, 3, 4, R_values, R_columns, R_rows};
    problem.D = D;
    problem.c = c;
    problem.n_box = 1;
    problem.box_indices = box_indices;
    problem.box_lower = box_lower;
    problem.box_upper = box_upper;
    problem.n_cones = 1;
    problem.cones = &cone;

    status = prefos_create_presolver(&problem, NULL, &presolver);
    CHECK(status == PREFOS_STATUS_OK);
    CHECK(presolver != NULL);
    status = prefos_run_presolve(presolver);
    CHECK(status == PREFOS_STATUS_REDUCED);

    reduced = prefos_get_reduced_problem(presolver);
    stats = prefos_get_stats(presolver);
    CHECK(reduced != NULL && stats != NULL);
    CHECK(reduced->n == 2);
    CHECK(reduced->A.rows == 1 && reduced->A.nnz == 1);
    CHECK(reduced->A.column_indices[0] == 0);
    CHECK(close_to(reduced->constraint_lower[0], 1.0));
    CHECK(close_to(reduced->constraint_upper[0], 3.0));
    CHECK(reduced->Q.rows == 2 && reduced->Q.nnz == 2);
    CHECK(close_to(reduced->Q.values[0], 3.0));
    CHECK(close_to(reduced->Q.values[1], 5.0));
    CHECK(reduced->R.rows == 2 && reduced->R.nnz == 3);
    CHECK(close_to(reduced->c[0], 16.0));
    CHECK(close_to(reduced->c[1], 11.0));
    CHECK(close_to(reduced->objective_offset, 22.0));
    CHECK(reduced->n_box == 0);
    CHECK(reduced->n_cones == 1);
    CHECK(reduced->cones[0].indices[0] == 0);
    CHECK(reduced->cones[0].indices[1] == 1);
    CHECK(stats->fixed_box_variables == 1);
    CHECK(stats->removed_empty_rows == 1);

    status = prefos_postsolve_primal(presolver, reduced_x, original_x);
    CHECK(status == PREFOS_STATUS_OK);
    CHECK(close_to(original_x[0], 2.0));
    CHECK(close_to(original_x[1], 2.5));
    CHECK(close_to(original_x[2], 1.5));

    status = prefos_verify_postsolve_primal(presolver, reduced_x, 1e-10, &verification);
    CHECK(status == PREFOS_STATUS_OK);
    CHECK(verification.passed);
    CHECK(verification.objective_absolute_error <= 1e-10);
    CHECK(verification.original_row_violation == 0.0);
    CHECK(verification.original_cone_violation == 0.0);

    status = prefos_verify_postsolve_primal(presolver, infeasible_reduced_x, 1e-10,
                                         &verification);
    CHECK(status == PREFOS_STATUS_OK);
    CHECK(!verification.passed);
    CHECK(verification.reduced_cone_violation > 0.9);

    prefos_free_presolver(presolver);
    return 0;
}

static int test_infeasible_empty_row(void)
{
    int A_rows[] = {0, 0};
    int Q_rows[] = {0, 0};
    int R_rows[] = {0};
    double lower[] = {1.0};
    double upper[] = {INFINITY};
    double c[] = {0.0};
    int box_indices[] = {0};
    double box_lower[] = {-INFINITY};
    double box_upper[] = {INFINITY};
    PreFOSProblemData problem;
    PreFOSPresolver *presolver = NULL;
    PreFOSStatus status;

    memset(&problem, 0, sizeof(problem));
    problem.n = 1;
    problem.A = (PreFOSCsrMatrix){1, 1, 0, NULL, NULL, A_rows};
    problem.constraint_lower = lower;
    problem.constraint_upper = upper;
    problem.Q = (PreFOSCsrMatrix){1, 1, 0, NULL, NULL, Q_rows};
    problem.q_storage = PREFOS_Q_UPPER_TRIANGULAR;
    problem.R = (PreFOSCsrMatrix){0, 1, 0, NULL, NULL, R_rows};
    problem.c = c;
    problem.n_box = 1;
    problem.box_indices = box_indices;
    problem.box_lower = box_lower;
    problem.box_upper = box_upper;

    status = prefos_create_presolver(&problem, NULL, &presolver);
    CHECK(status == PREFOS_STATUS_OK);
    CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_PRIMAL_INFEASIBLE);
    CHECK(prefos_get_reduced_problem(presolver) == NULL);
    prefos_free_presolver(presolver);
    return 0;
}

static int test_domain_overlap_is_rejected(void)
{
    int A_rows[] = {0};
    int Q_rows[] = {0, 0, 0};
    int R_rows[] = {0};
    double c[] = {0.0, 0.0};
    int box_indices[] = {0};
    double box_lower[] = {-INFINITY};
    double box_upper[] = {INFINITY};
    int cone_indices[] = {0, 1};
    PreFOSConeBlock cone = {PREFOS_CONE_SECOND_ORDER, 2, 0, cone_indices};
    PreFOSProblemData problem;
    PreFOSPresolver *presolver = NULL;

    memset(&problem, 0, sizeof(problem));
    problem.n = 2;
    problem.A = (PreFOSCsrMatrix){0, 2, 0, NULL, NULL, A_rows};
    problem.Q = (PreFOSCsrMatrix){2, 2, 0, NULL, NULL, Q_rows};
    problem.q_storage = PREFOS_Q_UPPER_TRIANGULAR;
    problem.R = (PreFOSCsrMatrix){0, 2, 0, NULL, NULL, R_rows};
    problem.c = c;
    problem.n_box = 1;
    problem.box_indices = box_indices;
    problem.box_lower = box_lower;
    problem.box_upper = box_upper;
    problem.n_cones = 1;
    problem.cones = &cone;

    CHECK(prefos_create_presolver(&problem, NULL, &presolver) ==
          PREFOS_STATUS_INVALID_ARGUMENT);
    CHECK(presolver == NULL);
    return 0;
}

static int test_singleton_box_row_tightening(void)
{
    double A_values[] = {2.0, 1.0};
    int A_columns[] = {0, 1};
    int A_rows[] = {0, 1, 2};
    double constraint_lower[] = {2.0, -INFINITY};
    double constraint_upper[] = {6.0, 4.0};
    int Q_rows[] = {0, 0, 0, 0};
    int R_rows[] = {0};
    double c[] = {0.0, 0.0, 0.0};
    int box_indices[] = {0};
    double box_lower[] = {-INFINITY};
    double box_upper[] = {INFINITY};
    int cone_indices[] = {1, 2};
    PreFOSConeBlock cone = {PREFOS_CONE_SECOND_ORDER, 2, 0, cone_indices};
    PreFOSProblemData problem;
    PreFOSPresolver *presolver = NULL;
    const PreFOSPresolvedProblem *reduced;
    const PreFOSStats *stats;

    memset(&problem, 0, sizeof(problem));
    problem.n = 3;
    problem.A = (PreFOSCsrMatrix){2, 3, 2, A_values, A_columns, A_rows};
    problem.constraint_lower = constraint_lower;
    problem.constraint_upper = constraint_upper;
    problem.Q = (PreFOSCsrMatrix){3, 3, 0, NULL, NULL, Q_rows};
    problem.q_storage = PREFOS_Q_UPPER_TRIANGULAR;
    problem.R = (PreFOSCsrMatrix){0, 3, 0, NULL, NULL, R_rows};
    problem.c = c;
    problem.n_box = 1;
    problem.box_indices = box_indices;
    problem.box_lower = box_lower;
    problem.box_upper = box_upper;
    problem.n_cones = 1;
    problem.cones = &cone;

    CHECK(prefos_create_presolver(&problem, NULL, &presolver) == PREFOS_STATUS_OK);
    CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_REDUCED);
    reduced = prefos_get_reduced_problem(presolver);
    stats = prefos_get_stats(presolver);
    CHECK(reduced != NULL && stats != NULL);
    CHECK(reduced->n == 3);
    CHECK(reduced->A.rows == 1 && reduced->A.nnz == 1);
    CHECK(reduced->A.column_indices[0] == 1);
    CHECK(reduced->n_box == 1);
    CHECK(close_to(reduced->box_lower[0], 1.0));
    CHECK(close_to(reduced->box_upper[0], 3.0));
    CHECK(stats->tightened_box_bounds == 2);
    CHECK(stats->removed_singleton_rows == 1);

    prefos_free_presolver(presolver);
    return 0;
}

static int test_objective_overflow_is_reported(void)
{
    int A_rows[] = {0};
    double Q_values[] = {DBL_MAX};
    int Q_columns[] = {0};
    int Q_rows[] = {0, 1};
    int R_rows[] = {0};
    double c[] = {0.0};
    int box_indices[] = {0};
    double box_lower[] = {DBL_MAX};
    double box_upper[] = {DBL_MAX};
    PreFOSProblemData problem;
    PreFOSPresolver *presolver = NULL;

    memset(&problem, 0, sizeof(problem));
    problem.n = 1;
    problem.A = (PreFOSCsrMatrix){0, 1, 0, NULL, NULL, A_rows};
    problem.Q = (PreFOSCsrMatrix){1, 1, 1, Q_values, Q_columns, Q_rows};
    problem.q_storage = PREFOS_Q_UPPER_TRIANGULAR;
    problem.R = (PreFOSCsrMatrix){0, 1, 0, NULL, NULL, R_rows};
    problem.c = c;
    problem.n_box = 1;
    problem.box_indices = box_indices;
    problem.box_lower = box_lower;
    problem.box_upper = box_upper;

    CHECK(prefos_create_presolver(&problem, NULL, &presolver) == PREFOS_STATUS_OK);
    CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_NUMERICAL_ERROR);
    CHECK(prefos_get_reduced_problem(presolver) == NULL);
    prefos_free_presolver(presolver);
    return 0;
}

static int test_iterative_linear_propagation_with_soc_envelope(void)
{
    double A_values[] = {1.0, 1.0, 1.0, -1.0, 1.0, 1.0};
    int A_columns[] = {0, 1, 0, 1, 0, 2};
    int A_rows[] = {0, 2, 4, 6};
    double constraint_lower[] = {-INFINITY, 4.0, -INFINITY};
    double constraint_upper[] = {10.0, INFINITY, 7.0};
    int Q_rows[] = {0, 0, 0, 0, 0};
    int R_rows[] = {0};
    double c[] = {0.0, 0.0, 0.0, 0.0};
    int box_indices[] = {0, 1};
    double box_lower[] = {0.0, 0.0};
    double box_upper[] = {INFINITY, INFINITY};
    int cone_indices[] = {2, 3};
    PreFOSConeBlock cone = {PREFOS_CONE_SECOND_ORDER, 2, 0, cone_indices};
    PreFOSProblemData problem;
    PreFOSSettings settings = prefos_default_settings();
    PreFOSPresolver *presolver = NULL;
    const PreFOSPresolvedProblem *reduced;
    const PreFOSStats *stats;

    memset(&problem, 0, sizeof(problem));
    problem.n = 4;
    problem.A = (PreFOSCsrMatrix){3, 4, 6, A_values, A_columns, A_rows};
    problem.constraint_lower = constraint_lower;
    problem.constraint_upper = constraint_upper;
    problem.Q = (PreFOSCsrMatrix){4, 4, 0, NULL, NULL, Q_rows};
    problem.q_storage = PREFOS_Q_UPPER_TRIANGULAR;
    problem.R = (PreFOSCsrMatrix){0, 4, 0, NULL, NULL, R_rows};
    problem.c = c;
    problem.n_box = 2;
    problem.box_indices = box_indices;
    problem.box_lower = box_lower;
    problem.box_upper = box_upper;
    problem.n_cones = 1;
    problem.cones = &cone;
    settings.feasibility_tolerance = 0.0;

    CHECK(prefos_create_presolver(&problem, &settings, &presolver) == PREFOS_STATUS_OK);
    CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_REDUCED);
    reduced = prefos_get_reduced_problem(presolver);
    stats = prefos_get_stats(presolver);
    CHECK(reduced != NULL && stats != NULL);
    CHECK(reduced->n == 4 && reduced->A.rows == 2);
    CHECK(close_to(reduced->box_lower[0], 4.0));
    CHECK(close_to(reduced->box_upper[0], 7.0));
    CHECK(close_to(reduced->box_lower[1], 0.0));
    CHECK(close_to(reduced->box_upper[1], 3.0));
    CHECK(stats->propagated_box_bounds == 6);
    CHECK(stats->linear_propagation_rounds == 3);
    CHECK(stats->removed_redundant_rows == 1);

    {
        int x0, x1, t, z;
        size_t feasible_points = 0;
        for (x0 = 0; x0 <= 10; ++x0)
        {
            for (x1 = 0; x1 <= 10; ++x1)
            {
                for (t = 0; t <= 7; ++t)
                {
                    for (z = -t; z <= t; ++z)
                    {
                        int original_feasible =
                            x0 + x1 <= 10 && x0 - x1 >= 4 && x0 + t <= 7;
                        int reduced_feasible = original_feasible &&
                                               x0 >= reduced->box_lower[0] &&
                                               x0 <= reduced->box_upper[0] &&
                                               x1 >= reduced->box_lower[1] &&
                                               x1 <= reduced->box_upper[1];
                        CHECK(original_feasible == reduced_feasible);
                        if (reduced_feasible)
                        {
                            double reduced_x[] = {(double) x0, (double) x1,
                                                  (double) t, (double) z};
                            PreFOSPrimalVerification verification;
                            CHECK(prefos_verify_postsolve_primal(
                                      presolver, reduced_x, 1e-12, &verification) ==
                                  PREFOS_STATUS_OK);
                            CHECK(verification.passed);
                            ++feasible_points;
                        }
                    }
                }
            }
        }
        CHECK(feasible_points == 50);
    }

    prefos_free_presolver(presolver);
    return 0;
}

static int test_event_driven_linear_propagation(void)
{
    {
        double A_values[] = {-1.0, 1.0, -1.0, 1.0};
        int A_columns[] = {1, 2, 0, 1};
        int A_rows[] = {0, 2, 4};
        double lower[] = {0.0, 0.0};
        double upper[] = {INFINITY, INFINITY};
        int Q_rows[] = {0, 0, 0, 0};
        int R_rows[] = {0};
        double c[] = {0.0, 0.0, 0.0};
        int box_indices[] = {0, 1, 2};
        double box_lower[] = {5.0, 0.0, 0.0};
        double box_upper[] = {10.0, 10.0, 10.0};
        PreFOSProblemData problem;
        PreFOSSettings settings = prefos_strict_settings();
        PreFOSPresolver *presolver = NULL;
        const PreFOSPresolvedProblem *reduced;
        const PreFOSStats *stats;

        memset(&problem, 0, sizeof(problem));
        problem.n = 3;
        problem.A = (PreFOSCsrMatrix){2, 3, 4, A_values, A_columns, A_rows};
        problem.constraint_lower = lower;
        problem.constraint_upper = upper;
        problem.Q = (PreFOSCsrMatrix){3, 3, 0, NULL, NULL, Q_rows};
        problem.q_storage = PREFOS_Q_UPPER_TRIANGULAR;
        problem.R = (PreFOSCsrMatrix){0, 3, 0, NULL, NULL, R_rows};
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
        CHECK(close_to(reduced->box_lower[0], 5.0));
        CHECK(close_to(reduced->box_lower[1], 5.0));
        CHECK(close_to(reduced->box_lower[2], 5.0));
        CHECK(stats->linear_activity_nnz_computed == 4);
        CHECK(stats->linear_rows_processed > 0);
        CHECK(stats->linear_nnz_processed == 2 * stats->linear_rows_processed);
        CHECK(stats->linear_rows_processed < 2 * stats->linear_propagation_rounds);
        CHECK(stats->linear_activity_updates > 0);
        prefos_free_presolver(presolver);

        settings.linear_propagation_min_changes_per_million = 1e9;
        settings.linear_propagation_max_stale_rounds = 1;
        CHECK(prefos_create_presolver(&problem, &settings, &presolver) ==
              PREFOS_STATUS_OK);
        CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_REDUCED);
        reduced = prefos_get_reduced_problem(presolver);
        stats = prefos_get_stats(presolver);
        CHECK(close_to(reduced->box_lower[1], 5.0));
        CHECK(close_to(reduced->box_lower[2], 0.0));
        CHECK(stats->linear_event_rounds == 1);
        CHECK(stats->linear_stale_stops == 1);
        prefos_free_presolver(presolver);

        settings.linear_propagation_min_changes_per_million = 0.0;
        settings.linear_propagation_max_stale_rounds = 0;
        settings.event_queue_activity_update_ratio = 0.0;
        CHECK(prefos_create_presolver(&problem, &settings, &presolver) ==
              PREFOS_STATUS_OK);
        CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_REDUCED);
        reduced = prefos_get_reduced_problem(presolver);
        stats = prefos_get_stats(presolver);
        CHECK(close_to(reduced->box_lower[1], 5.0));
        CHECK(close_to(reduced->box_lower[2], 5.0));
        CHECK(stats->linear_full_scan_fallbacks == 1);
        CHECK(stats->linear_event_rounds > 0);
        CHECK(stats->linear_full_scan_rounds > 0);
        prefos_free_presolver(presolver);

        settings.linear_propagation_gpu = 1;
        settings.event_queue_max_average_column_degree = 0.0;
        CHECK(prefos_create_presolver(&problem, &settings, &presolver) ==
              PREFOS_STATUS_OK);
        CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_REDUCED);
        reduced = prefos_get_reduced_problem(presolver);
        stats = prefos_get_stats(presolver);
        CHECK(close_to(reduced->box_lower[1], 5.0));
        CHECK(close_to(reduced->box_lower[2], 5.0));
        CHECK(stats->linear_gpu_rounds > 0 || stats->linear_gpu_fallbacks == 1);
        prefos_free_presolver(presolver);
    }
    {
        double A_values[] = {1.0, 1.0};
        int A_columns[] = {0, 1};
        int A_rows[] = {0, 2};
        double lower[] = {-INFINITY};
        double upper[] = {9.95};
        int Q_rows[] = {0, 0, 0};
        int R_rows[] = {0};
        double c[] = {0.0, 0.0};
        int box_indices[] = {0, 1};
        double box_lower[] = {0.0, 0.0};
        double box_upper[] = {10.0, 10.0};
        PreFOSProblemData problem;
        PreFOSSettings settings = prefos_default_settings();
        PreFOSPresolver *presolver = NULL;
        const PreFOSPresolvedProblem *reduced;

        memset(&problem, 0, sizeof(problem));
        problem.n = 2;
        problem.A = (PreFOSCsrMatrix){1, 2, 2, A_values, A_columns, A_rows};
        problem.constraint_lower = lower;
        problem.constraint_upper = upper;
        problem.Q = (PreFOSCsrMatrix){2, 2, 0, NULL, NULL, Q_rows};
        problem.q_storage = PREFOS_Q_UPPER_TRIANGULAR;
        problem.R = (PreFOSCsrMatrix){0, 2, 0, NULL, NULL, R_rows};
        problem.c = c;
        problem.n_box = 2;
        problem.box_indices = box_indices;
        problem.box_lower = box_lower;
        problem.box_upper = box_upper;

        CHECK(prefos_create_presolver(&problem, &settings, &presolver) ==
              PREFOS_STATUS_OK);
        CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_OK);
        reduced = prefos_get_reduced_problem(presolver);
        CHECK(close_to(reduced->box_upper[0], 10.0));
        CHECK(close_to(reduced->box_upper[1], 10.0));
        CHECK(prefos_get_stats(presolver)->propagated_box_bounds == 0);
        prefos_free_presolver(presolver);

        settings = prefos_strict_settings();
        settings.event_queue_max_average_column_degree = 0.0;
        CHECK(prefos_create_presolver(&problem, &settings, &presolver) ==
              PREFOS_STATUS_OK);
        CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_REDUCED);
        reduced = prefos_get_reduced_problem(presolver);
        CHECK(close_to(reduced->box_upper[0], 9.95));
        CHECK(close_to(reduced->box_upper[1], 9.95));
        CHECK(prefos_get_stats(presolver)->linear_event_rounds == 0);
        CHECK(prefos_get_stats(presolver)->linear_full_scan_rounds > 0);
        prefos_free_presolver(presolver);
    }
    {
        enum
        {
            N = 300
        };
        double A_values[N];
        int A_columns[N];
        int A_rows[] = {0, N};
        double lower[] = {-INFINITY};
        double upper[] = {100.0};
        int Q_rows[N + 1];
        int R_rows[] = {0};
        double c[N];
        int box_indices[N];
        double box_lower[N];
        double box_upper[N];
        PreFOSProblemData problem;
        PreFOSSettings settings = prefos_strict_settings();
        PreFOSPresolver *presolver = NULL;
        const PreFOSPresolvedProblem *reduced;
        const PreFOSStats *stats;
        int i;

        memset(&problem, 0, sizeof(problem));
        memset(Q_rows, 0, sizeof(Q_rows));
        memset(c, 0, sizeof(c));
        for (i = 0; i < N; ++i)
        {
            A_values[i] = 1.0;
            A_columns[i] = i;
            box_indices[i] = i;
            box_lower[i] = 0.0;
            box_upper[i] = INFINITY;
        }
        problem.n = N;
        problem.A = (PreFOSCsrMatrix){1, N, N, A_values, A_columns, A_rows};
        problem.constraint_lower = lower;
        problem.constraint_upper = upper;
        problem.Q = (PreFOSCsrMatrix){N, N, 0, NULL, NULL, Q_rows};
        problem.q_storage = PREFOS_Q_UPPER_TRIANGULAR;
        problem.R = (PreFOSCsrMatrix){0, N, 0, NULL, NULL, R_rows};
        problem.c = c;
        problem.n_box = N;
        problem.box_indices = box_indices;
        problem.box_lower = box_lower;
        problem.box_upper = box_upper;
        settings.linear_propagation_gpu = 1;
        settings.event_queue_max_average_column_degree = 0.0;

        CHECK(prefos_create_presolver(&problem, &settings, &presolver) ==
              PREFOS_STATUS_OK);
        CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_REDUCED);
        reduced = prefos_get_reduced_problem(presolver);
        stats = prefos_get_stats(presolver);
        for (i = 0; i < N; ++i) CHECK(close_to(reduced->box_upper[i], 100.0));
        CHECK(stats->linear_gpu_rounds == 0 || stats->linear_gpu_long_rows == 1);
        prefos_free_presolver(presolver);
    }
    return 0;
}

static int test_huge_implied_bound_is_skipped(void)
{
    double A_values[] = {1e-8, 1.0};
    int A_columns[] = {0, 1};
    int A_rows[] = {0, 2};
    double constraint_lower[] = {-INFINITY};
    double constraint_upper[] = {1.0};
    int Q_rows[] = {0, 0, 0};
    int R_rows[] = {0};
    double c[] = {0.0, 0.0};
    int box_indices[] = {0, 1};
    double box_lower[] = {0.0, 0.0};
    double box_upper[] = {INFINITY, 1.0};
    PreFOSProblemData problem;
    PreFOSSettings settings = prefos_strict_settings();
    PreFOSPresolver *presolver = NULL;
    const PreFOSPresolvedProblem *reduced;

    memset(&problem, 0, sizeof(problem));
    problem.n = 2;
    problem.A = (PreFOSCsrMatrix){1, 2, 2, A_values, A_columns, A_rows};
    problem.constraint_lower = constraint_lower;
    problem.constraint_upper = constraint_upper;
    problem.Q = (PreFOSCsrMatrix){2, 2, 0, NULL, NULL, Q_rows};
    problem.q_storage = PREFOS_Q_UPPER_TRIANGULAR;
    problem.R = (PreFOSCsrMatrix){0, 2, 0, NULL, NULL, R_rows};
    problem.c = c;
    problem.n_box = 2;
    problem.box_indices = box_indices;
    problem.box_lower = box_lower;
    problem.box_upper = box_upper;
    settings.remove_redundant_rows = 0;

    CHECK(prefos_create_presolver(&problem, &settings, &presolver) == PREFOS_STATUS_OK);
    CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_OK);
    reduced = prefos_get_reduced_problem(presolver);
    CHECK(isinf(reduced->box_upper[0]));
    CHECK(prefos_get_stats(presolver)->propagated_box_bounds == 0);
    prefos_free_presolver(presolver);

    A_values[0] = 2e-7;
    CHECK(prefos_create_presolver(&problem, &settings, &presolver) == PREFOS_STATUS_OK);
    CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_REDUCED);
    reduced = prefos_get_reduced_problem(presolver);
    CHECK(reduced->box_upper[0] < 5.01e6 && reduced->box_upper[0] > 4.99e6);
    CHECK(prefos_get_stats(presolver)->propagated_box_bounds == 1);
    prefos_free_presolver(presolver);
    return 0;
}

static int test_linear_propagation_detects_infeasibility(void)
{
    double A_values[] = {1.0, 1.0};
    int A_columns[] = {0, 1};
    int A_rows[] = {0, 2};
    double constraint_lower[] = {3.0};
    double constraint_upper[] = {INFINITY};
    int Q_rows[] = {0, 0, 0};
    int R_rows[] = {0};
    double c[] = {0.0, 0.0};
    int box_indices[] = {0, 1};
    double box_lower[] = {0.0, 0.0};
    double box_upper[] = {1.0, 1.0};
    PreFOSProblemData problem;
    PreFOSPresolver *presolver = NULL;

    memset(&problem, 0, sizeof(problem));
    problem.n = 2;
    problem.A = (PreFOSCsrMatrix){1, 2, 2, A_values, A_columns, A_rows};
    problem.constraint_lower = constraint_lower;
    problem.constraint_upper = constraint_upper;
    problem.Q = (PreFOSCsrMatrix){2, 2, 0, NULL, NULL, Q_rows};
    problem.q_storage = PREFOS_Q_UPPER_TRIANGULAR;
    problem.R = (PreFOSCsrMatrix){0, 2, 0, NULL, NULL, R_rows};
    problem.c = c;
    problem.n_box = 2;
    problem.box_indices = box_indices;
    problem.box_lower = box_lower;
    problem.box_upper = box_upper;

    CHECK(prefos_create_presolver(&problem, NULL, &presolver) == PREFOS_STATUS_OK);
    CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_PRIMAL_INFEASIBLE);
    CHECK(prefos_get_reduced_problem(presolver) == NULL);
    prefos_free_presolver(presolver);
    return 0;
}

static int test_propagated_bound_policy(void)
{
    double A_values[] = {1.0, 1.0, -1.0, 1.0};
    int A_columns[] = {0, 1, 0, 2};
    int A_rows[] = {0, 2, 4};
    double constraint_lower[] = {-INFINITY, -INFINITY};
    double constraint_upper[] = {1.0, 0.0};
    int Q_rows[] = {0, 0, 0, 0};
    int R_rows[] = {0};
    double c[] = {0.0, 0.0, 0.0};
    int box_indices[] = {0, 1, 2};
    double box_lower[] = {0.0, 0.0, 0.0};
    double box_upper[] = {INFINITY, INFINITY, INFINITY};
    PreFOSProblemData problem;
    PreFOSSettings settings = prefos_strict_settings();
    PreFOSPresolver *presolver = NULL;
    const PreFOSPresolvedProblem *reduced;
    const PreFOSStats *stats;
    size_t i;

    memset(&problem, 0, sizeof(problem));
    problem.n = 3;
    problem.A = (PreFOSCsrMatrix){2, 3, 4, A_values, A_columns, A_rows};
    problem.constraint_lower = constraint_lower;
    problem.constraint_upper = constraint_upper;
    problem.Q = (PreFOSCsrMatrix){3, 3, 0, NULL, NULL, Q_rows};
    problem.q_storage = PREFOS_Q_UPPER_TRIANGULAR;
    problem.R = (PreFOSCsrMatrix){0, 3, 0, NULL, NULL, R_rows};
    problem.c = c;
    problem.n_box = 3;
    problem.box_indices = box_indices;
    problem.box_lower = box_lower;
    problem.box_upper = box_upper;

    CHECK(settings.propagated_bound_policy ==
          PREFOS_PROPAGATED_BOUND_POLICY_FIRST_ORDER);
    CHECK(prefos_create_presolver(&problem, &settings, &presolver) == PREFOS_STATUS_OK);
    CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_REDUCED);
    reduced = prefos_get_reduced_problem(presolver);
    stats = prefos_get_stats(presolver);
    for (i = 0; i < 3; ++i) CHECK(close_to(reduced->box_upper[i], 1.0));
    CHECK(stats->propagated_box_bounds == 3);
    CHECK(stats->materialized_propagated_box_bounds == 3);
    CHECK(stats->suppressed_propagated_box_bounds == 0);
    prefos_free_presolver(presolver);

    settings.propagated_bound_policy = PREFOS_PROPAGATED_BOUND_POLICY_INTERIOR_POINT;
    CHECK(prefos_create_presolver(&problem, &settings, &presolver) == PREFOS_STATUS_OK);
    CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_OK);
    reduced = prefos_get_reduced_problem(presolver);
    stats = prefos_get_stats(presolver);
    CHECK(reduced->A.rows == 2);
    for (i = 0; i < 3; ++i) CHECK(isinf(reduced->box_upper[i]));
    CHECK(stats->propagated_box_bounds == 3);
    CHECK(stats->materialized_propagated_box_bounds == 0);
    CHECK(stats->suppressed_propagated_box_bounds == 3);
    CHECK(stats->removed_redundant_rows == 0);
    prefos_free_presolver(presolver);

    {
        double fixing_A_values[] = {1.0, 1.0};
        int fixing_A_columns[] = {0, 1};
        int fixing_A_rows[] = {0, 2};
        double fixing_lower[] = {1.0};
        double fixing_upper[] = {INFINITY};
        int fixing_Q_rows[] = {0, 0, 0};
        double fixing_c[] = {0.0, 0.0};
        int fixing_box_indices[] = {0, 1};
        double fixing_box_lower[] = {-INFINITY, 0.0};
        double fixing_box_upper[] = {1.0, 0.0};

        memset(&problem, 0, sizeof(problem));
        problem.n = 2;
        problem.A = (PreFOSCsrMatrix){
            1, 2, 2, fixing_A_values, fixing_A_columns, fixing_A_rows};
        problem.constraint_lower = fixing_lower;
        problem.constraint_upper = fixing_upper;
        problem.Q = (PreFOSCsrMatrix){2, 2, 0, NULL, NULL, fixing_Q_rows};
        problem.q_storage = PREFOS_Q_UPPER_TRIANGULAR;
        problem.R = (PreFOSCsrMatrix){0, 2, 0, NULL, NULL, R_rows};
        problem.c = fixing_c;
        problem.n_box = 2;
        problem.box_indices = fixing_box_indices;
        problem.box_lower = fixing_box_lower;
        problem.box_upper = fixing_box_upper;

        CHECK(prefos_create_presolver(&problem, &settings, &presolver) ==
              PREFOS_STATUS_OK);
        CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_REDUCED);
        reduced = prefos_get_reduced_problem(presolver);
        stats = prefos_get_stats(presolver);
        CHECK(reduced->n == 0);
        CHECK(stats->fixed_box_variables == 2);
        CHECK(stats->materialized_propagated_box_bounds == 1);
        CHECK(stats->suppressed_propagated_box_bounds == 0);
        prefos_free_presolver(presolver);
    }

    settings.propagated_bound_policy = (PreFOSPropagatedBoundPolicy) 2;
    CHECK(prefos_create_presolver(&problem, &settings, &presolver) ==
          PREFOS_STATUS_INVALID_ARGUMENT);
    CHECK(presolver == NULL);
    return 0;
}

static int test_affine_cone_coordinate_aggregation(void)
{
    double A_values[] = {1.0, -1.0, 1.0, -1.0};
    int A_columns[] = {0, 2, 1, 3};
    int A_rows[] = {0, 2, 4};
    double lower[] = {0.0, 0.0};
    double upper[] = {0.0, 0.0};
    int Q_rows[] = {0, 0, 0, 0, 0};
    int R_rows[] = {0};
    double c[] = {0.0, 0.0, 0.0, 0.0};
    int box_indices[] = {2, 3};
    double box_lower[] = {-INFINITY, -INFINITY};
    double box_upper[] = {INFINITY, INFINITY};
    int cone_indices[] = {0, 1};
    PreFOSConeBlock cone = {PREFOS_CONE_SECOND_ORDER, 2, 0, cone_indices, 0.0};
    PreFOSProblemData problem;
    PreFOSSettings settings = prefos_strict_settings();
    PreFOSPresolver *presolver = NULL;
    const PreFOSPresolvedProblem *reduced;
    const PreFOSStats *stats;
    double reduced_x[] = {1.0, 0.5};
    double reduced_z[] = {0.0, 0.0};
    double original_x[4];
    double original_y[2], original_z[4];
    PreFOSPrimalVerification verification;

    memset(&problem, 0, sizeof(problem));
    problem.n = 4;
    problem.A = (PreFOSCsrMatrix){2, 4, 4, A_values, A_columns, A_rows};
    problem.constraint_lower = lower;
    problem.constraint_upper = upper;
    problem.Q = (PreFOSCsrMatrix){4, 4, 0, NULL, NULL, Q_rows};
    problem.q_storage = PREFOS_Q_UPPER_TRIANGULAR;
    problem.R = (PreFOSCsrMatrix){0, 4, 0, NULL, NULL, R_rows};
    problem.c = c;
    problem.n_box = 2;
    problem.box_indices = box_indices;
    problem.box_lower = box_lower;
    problem.box_upper = box_upper;
    problem.n_cones = 1;
    problem.cones = &cone;
    settings.affine_cone_coordinate_aggregation = 1;

    CHECK(prefos_create_presolver(&problem, &settings, &presolver) == PREFOS_STATUS_OK);
    CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_REDUCED);
    reduced = prefos_get_reduced_problem(presolver);
    stats = prefos_get_stats(presolver);
    CHECK(reduced->n == 2);
    CHECK(reduced->A.rows == 0);
    CHECK(reduced->n_cones == 0);
    CHECK(reduced->n_affine_cones == 1);
    CHECK(reduced->affine_cone_matrix.rows == 2);
    CHECK(reduced->affine_cone_matrix.cols == 2);
    CHECK(reduced->affine_cone_matrix.nnz == 2);
    CHECK(reduced->affine_cones[0].type == PREFOS_CONE_SECOND_ORDER);
    CHECK(reduced->affine_cones[0].dimension == 2);
    CHECK(close_to(reduced->affine_cone_matrix.values[0], 1.0));
    CHECK(close_to(reduced->affine_cone_matrix.values[1], 1.0));
    CHECK(reduced->affine_cone_offset[0] == 0.0);
    CHECK(reduced->affine_cone_offset[1] == 0.0);
    CHECK(stats->aggregated_affine_cone_coordinates == 2);
    CHECK(stats->generated_affine_cone_blocks == 1);
    CHECK(prefos_postsolve_primal(presolver, reduced_x, original_x) == PREFOS_STATUS_OK);
    CHECK(close_to(original_x[0], 1.0));
    CHECK(close_to(original_x[1], 0.5));
    CHECK(close_to(original_x[2], 1.0));
    CHECK(close_to(original_x[3], 0.5));
    CHECK(prefos_verify_postsolve_primal(presolver, reduced_x, 1e-12, &verification) ==
          PREFOS_STATUS_OK);
    CHECK(verification.passed);
    reduced_x[0] = 0.25;
    CHECK(prefos_verify_postsolve_primal(presolver, reduced_x, 1e-12, &verification) ==
          PREFOS_STATUS_OK);
    CHECK(!verification.passed);
    CHECK(verification.reduced_cone_violation > 0.2);
    CHECK(prefos_postsolve_primal_dual(presolver, reduced_x, NULL, reduced_z, 1e-12,
                                    original_x, original_y, original_z) ==
          PREFOS_STATUS_DUAL_RECOVERY_UNAVAILABLE);
    CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_REDUCED);
    reduced = prefos_get_reduced_problem(presolver);
    CHECK(reduced->n == 2 && reduced->A.rows == 0);
    CHECK(reduced->n_affine_cones == 1);
    prefos_free_presolver(presolver);
    return 0;
}

static int test_affine_cone_input_passthrough(void)
{
    int A_rows[] = {0};
    int Q_rows[] = {0, 0};
    int R_rows[] = {0};
    double c[] = {-1.0};
    int box_indices[] = {0};
    double box_lower[] = {-INFINITY};
    double box_upper[] = {INFINITY};
    double G_values[] = {1.0};
    int G_columns[] = {0};
    int G_rows[] = {0, 0, 1};
    double h[] = {1.0, 0.0};
    PreFOSAffineConeBlock affine_cone = {PREFOS_CONE_SECOND_ORDER, 2, 0, 0.0};
    PreFOSProblemData problem;
    PreFOSPresolver *presolver = NULL;
    const PreFOSPresolvedProblem *reduced;
    PreFOSPrimalVerification verification;
    double x[] = {0.5};
    double reduced_z[] = {0.0};
    double reduced_affine_z[] = {-1.0, 1.0};
    double original_x[1], original_z[1], original_affine_z[2];
    PreFOSPostsolveKKTVerification kkt;

    memset(&problem, 0, sizeof(problem));
    problem.n = 1;
    problem.A = (PreFOSCsrMatrix){0, 1, 0, NULL, NULL, A_rows};
    problem.Q = (PreFOSCsrMatrix){1, 1, 0, NULL, NULL, Q_rows};
    problem.q_storage = PREFOS_Q_UPPER_TRIANGULAR;
    problem.R = (PreFOSCsrMatrix){0, 1, 0, NULL, NULL, R_rows};
    problem.c = c;
    problem.n_box = 1;
    problem.box_indices = box_indices;
    problem.box_lower = box_lower;
    problem.box_upper = box_upper;
    problem.affine_cone_matrix =
        (PreFOSCsrMatrix){2, 1, 1, G_values, G_columns, G_rows};
    problem.affine_cone_offset = h;
    problem.n_affine_cones = 1;
    problem.affine_cones = &affine_cone;

    CHECK(prefos_create_presolver(&problem, NULL, &presolver) == PREFOS_STATUS_OK);
    CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_OK);
    reduced = prefos_get_reduced_problem(presolver);
    CHECK(reduced->n_affine_cones == 1);
    CHECK(reduced->affine_cone_matrix.rows == 2);
    CHECK(reduced->affine_cone_matrix.nnz == 1);
    CHECK(prefos_verify_postsolve_primal(presolver, x, 1e-12, &verification) ==
          PREFOS_STATUS_OK);
    CHECK(verification.passed);
    x[0] = 2.0;
    CHECK(prefos_verify_postsolve_primal(presolver, x, 1e-12, &verification) ==
          PREFOS_STATUS_OK);
    CHECK(!verification.passed);
    CHECK(verification.original_cone_violation > 0.9);
    x[0] = 1.0;
    CHECK(prefos_postsolve_full_primal_dual(
              presolver, x, NULL, reduced_z, reduced_affine_z, 1e-12, original_x,
              NULL, original_z, original_affine_z) == PREFOS_STATUS_OK);
    CHECK(close_to(original_x[0], 1.0));
    CHECK(close_to(original_z[0], 0.0));
    CHECK(close_to(original_affine_z[0], -1.0));
    CHECK(close_to(original_affine_z[1], 1.0));
    CHECK(prefos_verify_postsolve_full_kkt(presolver, x, NULL, reduced_z,
                                        reduced_affine_z, 1e-12,
                                        &kkt) == PREFOS_STATUS_OK);
    CHECK(kkt.passed);
    prefos_free_presolver(presolver);

    problem.n_affine_cones = 0;
    problem.affine_cones = NULL;
    CHECK(prefos_create_presolver(&problem, NULL, &presolver) ==
          PREFOS_STATUS_INVALID_ARGUMENT);
    CHECK(presolver == NULL);
    return 0;
}

static int test_affine_aggregation_full_dual(void)
{
    double A_values[] = {1.0, -1.0, 1.0, -1.0};
    int A_columns[] = {0, 2, 1, 3};
    int A_rows[] = {0, 2, 4};
    double sides[] = {0.0, 0.0};
    int Q_rows[] = {0, 0, 0, 0, 0};
    int R_rows[] = {0};
    double c[] = {0.0, 0.0, 0.0, 0.0};
    int box_indices[] = {2, 3};
    double box_values[] = {1.0, 1.0};
    int cone_indices[] = {0, 1};
    PreFOSConeBlock cone = {PREFOS_CONE_SECOND_ORDER, 2, 0, cone_indices, 0.0};
    PreFOSProblemData problem;
    PreFOSSettings settings = prefos_strict_settings();
    PreFOSPresolver *presolver = NULL;
    const PreFOSPresolvedProblem *reduced;
    double reduced_affine_z[] = {-1.0, 1.0};
    double original_x[4], original_y[2], original_z[4];
    PreFOSPostsolveKKTVerification kkt;

    memset(&problem, 0, sizeof(problem));
    problem.n = 4;
    problem.A = (PreFOSCsrMatrix){2, 4, 4, A_values, A_columns, A_rows};
    problem.constraint_lower = sides;
    problem.constraint_upper = sides;
    problem.Q = (PreFOSCsrMatrix){4, 4, 0, NULL, NULL, Q_rows};
    problem.q_storage = PREFOS_Q_UPPER_TRIANGULAR;
    problem.R = (PreFOSCsrMatrix){0, 4, 0, NULL, NULL, R_rows};
    problem.c = c;
    problem.n_box = 2;
    problem.box_indices = box_indices;
    problem.box_lower = box_values;
    problem.box_upper = box_values;
    problem.n_cones = 1;
    problem.cones = &cone;
    settings.affine_cone_coordinate_aggregation = 1;

    CHECK(prefos_create_presolver(&problem, &settings, &presolver) == PREFOS_STATUS_OK);
    CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_REDUCED);
    reduced = prefos_get_reduced_problem(presolver);
    CHECK(reduced->n == 0);
    CHECK(reduced->A.rows == 0);
    CHECK(reduced->affine_cone_matrix.rows == 2);
    CHECK(reduced->affine_cone_matrix.nnz == 0);
    CHECK(prefos_postsolve_full_primal_dual(
              presolver, NULL, NULL, NULL, reduced_affine_z, 1e-12, original_x,
              original_y, original_z, NULL) == PREFOS_STATUS_OK);
    CHECK(close_to(original_y[0], 1.0));
    CHECK(close_to(original_y[1], -1.0));
    CHECK(close_to(original_z[0], -1.0));
    CHECK(close_to(original_z[1], 1.0));
    CHECK(close_to(original_z[2], 1.0));
    CHECK(close_to(original_z[3], -1.0));
    CHECK(prefos_verify_postsolve_full_kkt(presolver, NULL, NULL, NULL,
                                        reduced_affine_z, 1e-12,
                                        &kkt) == PREFOS_STATUS_OK);
    CHECK(kkt.passed);
    prefos_free_presolver(presolver);
    return 0;
}

static int test_affine_cone_propagation(void)
{
    int A_rows[] = {0};
    int Q_rows[] = {0, 0};
    int R_rows[] = {0};
    double c[] = {0.0};
    int box_indices[] = {0};
    double box_lower[] = {-INFINITY};
    double box_upper[] = {INFINITY};
    double G_values[] = {1.0};
    int G_columns[] = {0};
    int G_rows[] = {0, 0, 1};
    double h[] = {1.0, 0.0};
    PreFOSAffineConeBlock affine_cone = {PREFOS_CONE_SECOND_ORDER, 2, 0, 0.0};
    PreFOSProblemData problem;
    PreFOSPresolver *presolver = NULL;
    const PreFOSPresolvedProblem *reduced;
    const PreFOSStats *stats;

    memset(&problem, 0, sizeof(problem));
    problem.n = 1;
    problem.A = (PreFOSCsrMatrix){0, 1, 0, NULL, NULL, A_rows};
    problem.Q = (PreFOSCsrMatrix){1, 1, 0, NULL, NULL, Q_rows};
    problem.q_storage = PREFOS_Q_UPPER_TRIANGULAR;
    problem.R = (PreFOSCsrMatrix){0, 1, 0, NULL, NULL, R_rows};
    problem.c = c;
    problem.n_box = 1;
    problem.box_indices = box_indices;
    problem.box_lower = box_lower;
    problem.box_upper = box_upper;
    problem.affine_cone_matrix =
        (PreFOSCsrMatrix){2, 1, 1, G_values, G_columns, G_rows};
    problem.affine_cone_offset = h;
    problem.n_affine_cones = 1;
    problem.affine_cones = &affine_cone;

    CHECK(prefos_create_presolver(&problem, NULL, &presolver) == PREFOS_STATUS_OK);
    CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_OK);
    reduced = prefos_get_reduced_problem(presolver);
    stats = prefos_get_stats(presolver);
    CHECK(isinf(reduced->box_lower[0]) && reduced->box_lower[0] < 0.0);
    CHECK(isinf(reduced->box_upper[0]) && reduced->box_upper[0] > 0.0);
    CHECK(stats->tightened_affine_cone_envelopes >= 2);
    CHECK(stats->tightened_affine_variable_envelopes == 2);
    CHECK(stats->affine_cone_propagation_rounds == 2);
    prefos_free_presolver(presolver);

    box_lower[0] = 2.0;
    CHECK(prefos_create_presolver(&problem, NULL, &presolver) == PREFOS_STATUS_OK);
    CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_PRIMAL_INFEASIBLE);
    prefos_free_presolver(presolver);
    return 0;
}

static int test_affine_singleton_bound_certificate(void)
{
    int A_rows[] = {0};
    int Q_rows[] = {0, 0};
    int R_rows[] = {0};
    double c[] = {1.0};
    int box_indices[] = {0};
    double box_lower[] = {-INFINITY};
    double box_upper[] = {INFINITY};
    double G_values[] = {1.0};
    int G_columns[] = {0};
    int G_rows[] = {0, 1, 1};
    double h[] = {0.0, 0.0};
    PreFOSAffineConeBlock affine_cone = {PREFOS_CONE_SECOND_ORDER, 2, 0, 0.0};
    PreFOSProblemData problem;
    PreFOSSettings settings = prefos_default_settings();
    PreFOSPresolver *presolver = NULL;
    const PreFOSPresolvedProblem *reduced;
    const PreFOSStats *stats;
    double reduced_x[] = {0.0};
    double reduced_z[] = {-1.0};
    double reduced_affine_z[] = {0.0, 0.0};
    double original_x[1], original_z[1], original_affine_z[2];
    PreFOSPostsolveKKTVerification kkt;

    memset(&problem, 0, sizeof(problem));
    problem.n = 1;
    problem.A = (PreFOSCsrMatrix){0, 1, 0, NULL, NULL, A_rows};
    problem.Q = (PreFOSCsrMatrix){1, 1, 0, NULL, NULL, Q_rows};
    problem.q_storage = PREFOS_Q_UPPER_TRIANGULAR;
    problem.R = (PreFOSCsrMatrix){0, 1, 0, NULL, NULL, R_rows};
    problem.c = c;
    problem.n_box = 1;
    problem.box_indices = box_indices;
    problem.box_lower = box_lower;
    problem.box_upper = box_upper;
    problem.affine_cone_matrix =
        (PreFOSCsrMatrix){2, 1, 1, G_values, G_columns, G_rows};
    problem.affine_cone_offset = h;
    problem.n_affine_cones = 1;
    problem.affine_cones = &affine_cone;

    CHECK(prefos_create_presolver(&problem, &settings, &presolver) == PREFOS_STATUS_OK);
    CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_REDUCED);
    reduced = prefos_get_reduced_problem(presolver);
    stats = prefos_get_stats(presolver);
    CHECK(close_to(reduced->box_lower[0], 0.0));
    CHECK(isinf(reduced->box_upper[0]) && reduced->box_upper[0] > 0.0);
    CHECK(stats->materialized_affine_cone_box_bounds == 1);
    CHECK(stats->suppressed_affine_cone_box_bounds == 0);
    CHECK(prefos_postsolve_full_primal_dual(
              presolver, reduced_x, NULL, reduced_z, reduced_affine_z, 1e-12,
              original_x, NULL, original_z, original_affine_z) == PREFOS_STATUS_OK);
    CHECK(close_to(original_z[0], 0.0));
    CHECK(close_to(original_affine_z[0], -1.0));
    CHECK(close_to(original_affine_z[1], 0.0));
    CHECK(prefos_verify_postsolve_full_kkt(presolver, reduced_x, NULL, reduced_z,
                                        reduced_affine_z, 1e-12,
                                        &kkt) == PREFOS_STATUS_OK);
    CHECK(kkt.passed);

    CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_REDUCED);
    CHECK(prefos_get_stats(presolver)->materialized_affine_cone_box_bounds == 1);
    CHECK(prefos_postsolve_full_primal_dual(
              presolver, reduced_x, NULL, reduced_z, reduced_affine_z, 1e-12,
              original_x, NULL, original_z, original_affine_z) == PREFOS_STATUS_OK);
    CHECK(close_to(original_z[0], 0.0));
    CHECK(close_to(original_affine_z[0], -1.0));
    prefos_free_presolver(presolver);

    settings.propagated_bound_policy = PREFOS_PROPAGATED_BOUND_POLICY_INTERIOR_POINT;
    CHECK(prefos_create_presolver(&problem, &settings, &presolver) == PREFOS_STATUS_OK);
    CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_OK);
    reduced = prefos_get_reduced_problem(presolver);
    stats = prefos_get_stats(presolver);
    CHECK(isinf(reduced->box_lower[0]) && reduced->box_lower[0] < 0.0);
    CHECK(stats->materialized_affine_cone_box_bounds == 0);
    CHECK(stats->suppressed_affine_cone_box_bounds == 1);
    prefos_free_presolver(presolver);

    box_lower[0] = 0.0;
    h[0] = -1.0;
    CHECK(prefos_create_presolver(&problem, &settings, &presolver) == PREFOS_STATUS_OK);
    CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_REDUCED);
    reduced = prefos_get_reduced_problem(presolver);
    CHECK(close_to(reduced->box_lower[0], 1.0));
    reduced_x[0] = 1.0;
    CHECK(prefos_postsolve_full_primal_dual(
              presolver, reduced_x, NULL, reduced_z, reduced_affine_z, 1e-12,
              original_x, NULL, original_z, original_affine_z) == PREFOS_STATUS_OK);
    CHECK(close_to(original_z[0], 0.0));
    CHECK(close_to(original_affine_z[0], -1.0));
    CHECK(prefos_verify_postsolve_full_kkt(presolver, reduced_x, NULL, reduced_z,
                                        reduced_affine_z, 1e-12,
                                        &kkt) == PREFOS_STATUS_OK);
    CHECK(kkt.passed);
    prefos_free_presolver(presolver);
    box_lower[0] = -INFINITY;
    h[0] = 0.0;

    box_upper[0] = 0.0;
    CHECK(prefos_create_presolver(&problem, &settings, &presolver) == PREFOS_STATUS_OK);
    CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_REDUCED);
    reduced = prefos_get_reduced_problem(presolver);
    CHECK(reduced->n == 0);
    CHECK(prefos_get_stats(presolver)->materialized_affine_cone_box_bounds == 1);
    CHECK(prefos_postsolve_full_primal_dual(
              presolver, NULL, NULL, NULL, reduced_affine_z, 1e-12, original_x, NULL,
              original_z, original_affine_z) == PREFOS_STATUS_OK);
    CHECK(close_to(original_z[0], 0.0));
    CHECK(close_to(original_affine_z[0], -1.0));
    CHECK(prefos_verify_postsolve_full_kkt(presolver, NULL, NULL, NULL,
                                        reduced_affine_z, 1e-12,
                                        &kkt) == PREFOS_STATUS_OK);
    CHECK(kkt.passed);
    prefos_free_presolver(presolver);
    box_upper[0] = INFINITY;

    settings.propagated_bound_policy = PREFOS_PROPAGATED_BOUND_POLICY_FIRST_ORDER;
    G_values[0] = -1.0;
    c[0] = -1.0;
    reduced_x[0] = 0.0;
    reduced_z[0] = 1.0;
    CHECK(prefos_create_presolver(&problem, &settings, &presolver) == PREFOS_STATUS_OK);
    CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_REDUCED);
    reduced = prefos_get_reduced_problem(presolver);
    CHECK(isinf(reduced->box_lower[0]) && reduced->box_lower[0] < 0.0);
    CHECK(close_to(reduced->box_upper[0], 0.0));
    CHECK(prefos_postsolve_full_primal_dual(
              presolver, reduced_x, NULL, reduced_z, reduced_affine_z, 1e-12,
              original_x, NULL, original_z, original_affine_z) == PREFOS_STATUS_OK);
    CHECK(close_to(original_z[0], 0.0));
    CHECK(close_to(original_affine_z[0], -1.0));
    CHECK(prefos_verify_postsolve_full_kkt(presolver, reduced_x, NULL, reduced_z,
                                        reduced_affine_z, 1e-12,
                                        &kkt) == PREFOS_STATUS_OK);
    CHECK(kkt.passed);
    prefos_free_presolver(presolver);
    return 0;
}

static int test_zero_affine_block_removal(void)
{
    int empty_rows[] = {0};
    int G_rows[] = {0, 0, 0};
    double h[] = {0.0, 0.0};
    PreFOSAffineConeBlock block = {PREFOS_CONE_SECOND_ORDER, 2, 0, 0.0};
    PreFOSProblemData problem;
    PreFOSPresolver *presolver = NULL;
    const PreFOSPresolvedProblem *reduced;
    double original_affine_z[2];
    PreFOSPostsolveKKTVerification kkt;

    memset(&problem, 0, sizeof(problem));
    problem.A = (PreFOSCsrMatrix){0, 0, 0, NULL, NULL, empty_rows};
    problem.Q = (PreFOSCsrMatrix){0, 0, 0, NULL, NULL, empty_rows};
    problem.q_storage = PREFOS_Q_UPPER_TRIANGULAR;
    problem.R = (PreFOSCsrMatrix){0, 0, 0, NULL, NULL, empty_rows};
    problem.affine_cone_matrix = (PreFOSCsrMatrix){2, 0, 0, NULL, NULL, G_rows};
    problem.affine_cone_offset = h;
    problem.n_affine_cones = 1;
    problem.affine_cones = &block;

    CHECK(prefos_create_presolver(&problem, NULL, &presolver) == PREFOS_STATUS_OK);
    CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_REDUCED);
    reduced = prefos_get_reduced_problem(presolver);
    CHECK(reduced->n_affine_cones == 0);
    CHECK(reduced->affine_cone_matrix.rows == 0);
    CHECK(prefos_get_stats(presolver)->removed_affine_cone_blocks == 1);
    CHECK(prefos_get_stats(presolver)->removed_affine_cone_coordinates == 2);
    CHECK(prefos_postsolve_full_primal_dual(presolver, NULL, NULL, NULL, NULL, 1e-12,
                                         NULL, NULL, NULL,
                                         original_affine_z) == PREFOS_STATUS_OK);
    CHECK(close_to(original_affine_z[0], 0.0));
    CHECK(close_to(original_affine_z[1], 0.0));
    CHECK(prefos_verify_postsolve_full_kkt(presolver, NULL, NULL, NULL, NULL, 1e-12,
                                        &kkt) == PREFOS_STATUS_OK);
    CHECK(kkt.passed);
    prefos_free_presolver(presolver);
    return 0;
}

static int test_generated_affine_singleton_bound_certificate(void)
{
    double A_values[] = {1.0, -1.0, 1.0};
    int A_columns[] = {0, 2, 1};
    int A_rows[] = {0, 2, 3};
    double sides[] = {0.0, 0.0};
    int Q_rows[] = {0, 0, 0, 0};
    int R_rows[] = {0};
    double c[] = {0.0, 0.0, 1.0};
    int box_indices[] = {2};
    double box_lower[] = {-INFINITY};
    double box_upper[] = {INFINITY};
    int cone_indices[] = {0, 1};
    PreFOSConeBlock cone = {PREFOS_CONE_SECOND_ORDER, 2, 0, cone_indices, 0.0};
    PreFOSProblemData problem;
    PreFOSSettings settings = prefos_default_settings();
    PreFOSPresolver *presolver = NULL;
    const PreFOSPresolvedProblem *reduced;
    double reduced_x[] = {0.0};
    double reduced_z[] = {-1.0};
    double reduced_affine_z[] = {0.0, 0.0};
    double original_x[3], original_y[2], original_z[3];
    PreFOSPostsolveKKTVerification kkt;

    memset(&problem, 0, sizeof(problem));
    problem.n = 3;
    problem.A = (PreFOSCsrMatrix){2, 3, 3, A_values, A_columns, A_rows};
    problem.constraint_lower = sides;
    problem.constraint_upper = sides;
    problem.Q = (PreFOSCsrMatrix){3, 3, 0, NULL, NULL, Q_rows};
    problem.q_storage = PREFOS_Q_UPPER_TRIANGULAR;
    problem.R = (PreFOSCsrMatrix){0, 3, 0, NULL, NULL, R_rows};
    problem.c = c;
    problem.n_box = 1;
    problem.box_indices = box_indices;
    problem.box_lower = box_lower;
    problem.box_upper = box_upper;
    problem.n_cones = 1;
    problem.cones = &cone;
    settings.affine_cone_coordinate_aggregation = 1;

    CHECK(prefos_create_presolver(&problem, &settings, &presolver) == PREFOS_STATUS_OK);
    CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_REDUCED);
    reduced = prefos_get_reduced_problem(presolver);
    CHECK(reduced->n == 1);
    CHECK(reduced->A.rows == 0);
    CHECK(reduced->n_cones == 0);
    CHECK(reduced->n_affine_cones == 1);
    CHECK(close_to(reduced->box_lower[0], 0.0));
    CHECK(prefos_get_stats(presolver)->materialized_affine_cone_box_bounds == 1);
    CHECK(prefos_postsolve_full_primal_dual(
              presolver, reduced_x, NULL, reduced_z, reduced_affine_z, 1e-12,
              original_x, original_y, original_z, NULL) == PREFOS_STATUS_OK);
    CHECK(close_to(original_y[0], 1.0));
    CHECK(close_to(original_y[1], 0.0));
    CHECK(close_to(original_z[0], -1.0));
    CHECK(close_to(original_z[1], 0.0));
    CHECK(close_to(original_z[2], 0.0));
    CHECK(prefos_verify_postsolve_full_kkt(presolver, reduced_x, NULL, reduced_z,
                                        reduced_affine_z, 1e-12,
                                        &kkt) == PREFOS_STATUS_OK);
    CHECK(kkt.passed);
    prefos_free_presolver(presolver);
    return 0;
}

static int test_input_affine_structural_faces(void)
{
    {
        int A_rows[] = {0};
        int Q_rows[] = {0, 0};
        int R_rows[] = {0};
        double c[] = {1.0};
        int box_indices[] = {0};
        double box_lower[] = {-INFINITY};
        double box_upper[] = {INFINITY};
        double G_values[] = {1.0};
        int G_columns[] = {0};
        int G_rows[] = {0, 0, 1, 1};
        double h[] = {0.0, 0.0, 0.0};
        PreFOSAffineConeBlock block = {PREFOS_CONE_ROTATED_SECOND_ORDER, 3, 0, 0.0};
        PreFOSProblemData problem;
        PreFOSSettings settings = prefos_default_settings();
        PreFOSPresolver *presolver = NULL;
        const PreFOSPresolvedProblem *reduced;
        double reduced_x[] = {0.0};
        double reduced_z[] = {0.0};
        double reduced_affine_z[] = {-1.0};
        double original_x[1], original_z[1], original_affine_z[3];
        PreFOSPostsolveKKTVerification kkt;

        memset(&problem, 0, sizeof(problem));
        problem.n = 1;
        problem.A = (PreFOSCsrMatrix){0, 1, 0, NULL, NULL, A_rows};
        problem.Q = (PreFOSCsrMatrix){1, 1, 0, NULL, NULL, Q_rows};
        problem.q_storage = PREFOS_Q_UPPER_TRIANGULAR;
        problem.R = (PreFOSCsrMatrix){0, 1, 0, NULL, NULL, R_rows};
        problem.c = c;
        problem.n_box = 1;
        problem.box_indices = box_indices;
        problem.box_lower = box_lower;
        problem.box_upper = box_upper;
        problem.affine_cone_matrix =
            (PreFOSCsrMatrix){3, 1, 1, G_values, G_columns, G_rows};
        problem.affine_cone_offset = h;
        problem.n_affine_cones = 1;
        problem.affine_cones = &block;

        CHECK(prefos_create_presolver(&problem, &settings, &presolver) ==
              PREFOS_STATUS_OK);
        CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_REDUCED);
        reduced = prefos_get_reduced_problem(presolver);
        CHECK(reduced->n_affine_cones == 1);
        CHECK(reduced->affine_cone_matrix.rows == 1);
        CHECK(reduced->affine_cones[0].type == PREFOS_CONE_NONNEGATIVE);
        CHECK(reduced->affine_cones[0].dimension == 1);
        CHECK(prefos_get_stats(presolver)->reduced_affine_rsoc_faces == 1);
        CHECK(prefos_get_stats(presolver)->removed_affine_cone_coordinates == 2);
        CHECK(prefos_postsolve_full_primal_dual(
                  presolver, reduced_x, NULL, reduced_z, reduced_affine_z, 1e-12,
                  original_x, NULL, original_z, original_affine_z) == PREFOS_STATUS_OK);
        CHECK(close_to(original_affine_z[0], 0.0));
        CHECK(close_to(original_affine_z[1], -1.0));
        CHECK(close_to(original_affine_z[2], 0.0));
        CHECK(prefos_verify_postsolve_full_kkt(presolver, reduced_x, NULL, reduced_z,
                                            reduced_affine_z, 1e-12,
                                            &kkt) == PREFOS_STATUS_OK);
        CHECK(kkt.passed);
        prefos_free_presolver(presolver);

        settings.rsoc_face_reduction = 0;
        CHECK(prefos_create_presolver(&problem, &settings, &presolver) ==
              PREFOS_STATUS_OK);
        CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_REDUCED);
        reduced = prefos_get_reduced_problem(presolver);
        CHECK(reduced->affine_cones[0].type == PREFOS_CONE_ROTATED_SECOND_ORDER);
        CHECK(reduced->affine_cones[0].dimension == 3);
        CHECK(prefos_get_stats(presolver)->reduced_affine_rsoc_faces == 0);
        prefos_free_presolver(presolver);
    }
    {
        int A_rows[] = {0};
        int Q_rows[] = {0, 0};
        int R_rows[] = {0};
        double c[] = {1.0};
        int box_indices[] = {0};
        double box_lower[] = {-INFINITY};
        double box_upper[] = {INFINITY};
        double G_values[] = {1.0};
        int G_columns[] = {0};
        int G_rows[] = {0, 0, 0, 1};
        double h[] = {0.0, 0.0, 0.0};
        PreFOSAffineConeBlock block = {PREFOS_CONE_POSITIVE_SEMIDEFINITE, 3, 2, 0.0};
        PreFOSProblemData problem;
        PreFOSPresolver *presolver = NULL;
        const PreFOSPresolvedProblem *reduced;
        double reduced_x[] = {0.0};
        double reduced_z[] = {0.0};
        double reduced_affine_z[] = {-1.0};
        double original_x[1], original_z[1], original_affine_z[3];
        PreFOSPostsolveKKTVerification kkt;

        memset(&problem, 0, sizeof(problem));
        problem.n = 1;
        problem.A = (PreFOSCsrMatrix){0, 1, 0, NULL, NULL, A_rows};
        problem.Q = (PreFOSCsrMatrix){1, 1, 0, NULL, NULL, Q_rows};
        problem.q_storage = PREFOS_Q_UPPER_TRIANGULAR;
        problem.R = (PreFOSCsrMatrix){0, 1, 0, NULL, NULL, R_rows};
        problem.c = c;
        problem.n_box = 1;
        problem.box_indices = box_indices;
        problem.box_lower = box_lower;
        problem.box_upper = box_upper;
        problem.affine_cone_matrix =
            (PreFOSCsrMatrix){3, 1, 1, G_values, G_columns, G_rows};
        problem.affine_cone_offset = h;
        problem.n_affine_cones = 1;
        problem.affine_cones = &block;

        CHECK(prefos_create_presolver(&problem, NULL, &presolver) == PREFOS_STATUS_OK);
        CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_REDUCED);
        reduced = prefos_get_reduced_problem(presolver);
        CHECK(reduced->affine_cones[0].type == PREFOS_CONE_POSITIVE_SEMIDEFINITE);
        CHECK(reduced->affine_cones[0].matrix_order == 1);
        CHECK(reduced->affine_cones[0].dimension == 1);
        CHECK(prefos_get_stats(presolver)->reduced_affine_psd_faces == 1);
        CHECK(prefos_postsolve_full_primal_dual(
                  presolver, reduced_x, NULL, reduced_z, reduced_affine_z, 1e-12,
                  original_x, NULL, original_z, original_affine_z) == PREFOS_STATUS_OK);
        CHECK(close_to(original_affine_z[0], 0.0));
        CHECK(close_to(original_affine_z[1], 0.0));
        CHECK(close_to(original_affine_z[2], -1.0));
        CHECK(prefos_verify_postsolve_full_kkt(presolver, reduced_x, NULL, reduced_z,
                                            reduced_affine_z, 1e-12,
                                            &kkt) == PREFOS_STATUS_OK);
        CHECK(kkt.passed);
        prefos_free_presolver(presolver);
    }
    {
        int A_rows[] = {0};
        int Q_rows[] = {0, 0};
        int R_rows[] = {0};
        double c[] = {1.0};
        int box_indices[] = {0};
        double box_lower[] = {-INFINITY};
        double box_upper[] = {INFINITY};
        double G_values[] = {1.0};
        int G_columns[] = {0};
        int G_rows[] = {0, 0, 0, 1};
        double h[] = {0.0, 0.0, 0.0};
        PreFOSAffineConeBlock block = {PREFOS_CONE_EXPONENTIAL, 3, 0, 0.0};
        PreFOSProblemData problem;
        PreFOSPresolver *presolver = NULL;
        const PreFOSPresolvedProblem *reduced;
        double reduced_x[] = {0.0};
        double reduced_z[] = {0.0};
        double reduced_affine_z[] = {-1.0};
        double original_x[1], original_z[1], original_affine_z[3];
        PreFOSPostsolveKKTVerification kkt;

        memset(&problem, 0, sizeof(problem));
        problem.n = 1;
        problem.A = (PreFOSCsrMatrix){0, 1, 0, NULL, NULL, A_rows};
        problem.Q = (PreFOSCsrMatrix){1, 1, 0, NULL, NULL, Q_rows};
        problem.q_storage = PREFOS_Q_UPPER_TRIANGULAR;
        problem.R = (PreFOSCsrMatrix){0, 1, 0, NULL, NULL, R_rows};
        problem.c = c;
        problem.n_box = 1;
        problem.box_indices = box_indices;
        problem.box_lower = box_lower;
        problem.box_upper = box_upper;
        problem.affine_cone_matrix =
            (PreFOSCsrMatrix){3, 1, 1, G_values, G_columns, G_rows};
        problem.affine_cone_offset = h;
        problem.n_affine_cones = 1;
        problem.affine_cones = &block;

        CHECK(prefos_create_presolver(&problem, NULL, &presolver) == PREFOS_STATUS_OK);
        CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_REDUCED);
        reduced = prefos_get_reduced_problem(presolver);
        CHECK(reduced->affine_cones[0].type == PREFOS_CONE_NONNEGATIVE);
        CHECK(reduced->affine_cones[0].dimension == 1);
        CHECK(prefos_get_stats(presolver)->reduced_affine_exponential_faces == 1);
        CHECK(prefos_postsolve_full_primal_dual(
                  presolver, reduced_x, NULL, reduced_z, reduced_affine_z, 1e-12,
                  original_x, NULL, original_z, original_affine_z) == PREFOS_STATUS_OK);
        CHECK(close_to(original_affine_z[0], 0.0));
        CHECK(close_to(original_affine_z[1], 0.0));
        CHECK(close_to(original_affine_z[2], -1.0));
        CHECK(prefos_verify_postsolve_full_kkt(presolver, reduced_x, NULL, reduced_z,
                                            reduced_affine_z, 1e-12,
                                            &kkt) == PREFOS_STATUS_OK);
        CHECK(kkt.passed);
        prefos_free_presolver(presolver);
    }
    {
        int A_rows[] = {0};
        int Q_rows[] = {0, 0, 0};
        int R_rows[] = {0};
        double c[] = {1.0, 1.0};
        int box_indices[] = {0, 1};
        double box_lower[] = {-INFINITY, -INFINITY};
        double box_upper[] = {INFINITY, INFINITY};
        double G_values[] = {1.0, 1.0};
        int G_columns[] = {0, 1};
        int G_rows[] = {0, 1, 2, 2};
        double h[] = {0.0, 0.0, 0.0};
        PreFOSAffineConeBlock block = {PREFOS_CONE_POWER, 3, 0, 0.3};
        PreFOSProblemData problem;
        PreFOSPresolver *presolver = NULL;
        const PreFOSPresolvedProblem *reduced;
        double reduced_x[] = {0.0, 0.0};
        double reduced_z[] = {0.0, 0.0};
        double reduced_affine_z[] = {-1.0, -1.0};
        double original_x[2], original_z[2], original_affine_z[3];
        PreFOSPostsolveKKTVerification kkt;

        memset(&problem, 0, sizeof(problem));
        problem.n = 2;
        problem.A = (PreFOSCsrMatrix){0, 2, 0, NULL, NULL, A_rows};
        problem.Q = (PreFOSCsrMatrix){2, 2, 0, NULL, NULL, Q_rows};
        problem.q_storage = PREFOS_Q_UPPER_TRIANGULAR;
        problem.R = (PreFOSCsrMatrix){0, 2, 0, NULL, NULL, R_rows};
        problem.c = c;
        problem.n_box = 2;
        problem.box_indices = box_indices;
        problem.box_lower = box_lower;
        problem.box_upper = box_upper;
        problem.affine_cone_matrix =
            (PreFOSCsrMatrix){3, 2, 2, G_values, G_columns, G_rows};
        problem.affine_cone_offset = h;
        problem.n_affine_cones = 1;
        problem.affine_cones = &block;

        CHECK(prefos_create_presolver(&problem, NULL, &presolver) == PREFOS_STATUS_OK);
        CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_REDUCED);
        reduced = prefos_get_reduced_problem(presolver);
        CHECK(reduced->affine_cones[0].type == PREFOS_CONE_NONNEGATIVE);
        CHECK(reduced->affine_cones[0].dimension == 2);
        CHECK(prefos_get_stats(presolver)->reduced_affine_power_faces == 1);
        CHECK(prefos_postsolve_full_primal_dual(
                  presolver, reduced_x, NULL, reduced_z, reduced_affine_z, 1e-12,
                  original_x, NULL, original_z, original_affine_z) == PREFOS_STATUS_OK);
        CHECK(close_to(original_affine_z[0], -1.0));
        CHECK(close_to(original_affine_z[1], -1.0));
        CHECK(close_to(original_affine_z[2], 0.0));
        CHECK(prefos_verify_postsolve_full_kkt(presolver, reduced_x, NULL, reduced_z,
                                            reduced_affine_z, 1e-12,
                                            &kkt) == PREFOS_STATUS_OK);
        CHECK(kkt.passed);
        prefos_free_presolver(presolver);
    }
    return 0;
}

static int test_generated_affine_structural_face(void)
{
    double A_values[] = {1.0, 1.0, -1.0, 1.0};
    int A_columns[] = {0, 1, 3, 2};
    int A_rows[] = {0, 1, 3, 4};
    double sides[] = {0.0, 0.0, 0.0};
    int Q_rows[] = {0, 0, 0, 0, 0};
    int R_rows[] = {0};
    double c[] = {0.0, 0.0, 0.0, 1.0};
    int box_indices[] = {3};
    double box_lower[] = {-INFINITY};
    double box_upper[] = {INFINITY};
    int cone_indices[] = {0, 1, 2};
    PreFOSConeBlock cone = {PREFOS_CONE_ROTATED_SECOND_ORDER, 3, 0, cone_indices, 0.0};
    PreFOSProblemData problem;
    PreFOSSettings settings = prefos_default_settings();
    PreFOSPresolver *presolver = NULL;
    const PreFOSPresolvedProblem *reduced;
    double reduced_x[] = {0.0};
    double reduced_z[] = {0.0};
    double reduced_affine_z[] = {-1.0};
    double original_x[4], original_y[3], original_z[4];
    PreFOSPostsolveKKTVerification kkt;

    memset(&problem, 0, sizeof(problem));
    problem.n = 4;
    problem.A = (PreFOSCsrMatrix){3, 4, 4, A_values, A_columns, A_rows};
    problem.constraint_lower = sides;
    problem.constraint_upper = sides;
    problem.Q = (PreFOSCsrMatrix){4, 4, 0, NULL, NULL, Q_rows};
    problem.q_storage = PREFOS_Q_UPPER_TRIANGULAR;
    problem.R = (PreFOSCsrMatrix){0, 4, 0, NULL, NULL, R_rows};
    problem.c = c;
    problem.n_box = 1;
    problem.box_indices = box_indices;
    problem.box_lower = box_lower;
    problem.box_upper = box_upper;
    problem.n_cones = 1;
    problem.cones = &cone;
    settings.affine_cone_coordinate_aggregation = 1;

    CHECK(prefos_create_presolver(&problem, &settings, &presolver) == PREFOS_STATUS_OK);
    CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_REDUCED);
    reduced = prefos_get_reduced_problem(presolver);
    CHECK(reduced->n == 1 && reduced->A.rows == 0);
    CHECK(reduced->n_cones == 0 && reduced->n_affine_cones == 1);
    CHECK(reduced->affine_cones[0].type == PREFOS_CONE_NONNEGATIVE);
    CHECK(reduced->affine_cones[0].dimension == 1);
    CHECK(prefos_get_stats(presolver)->reduced_affine_rsoc_faces == 1);
    CHECK(prefos_postsolve_full_primal_dual(
              presolver, reduced_x, NULL, reduced_z, reduced_affine_z, 1e-12,
              original_x, original_y, original_z, NULL) == PREFOS_STATUS_OK);
    CHECK(close_to(original_y[0], 0.0));
    CHECK(close_to(original_y[1], 1.0));
    CHECK(close_to(original_y[2], 0.0));
    CHECK(close_to(original_z[0], 0.0));
    CHECK(close_to(original_z[1], -1.0));
    CHECK(close_to(original_z[2], 0.0));
    CHECK(close_to(original_z[3], 0.0));
    CHECK(prefos_verify_postsolve_full_kkt(presolver, reduced_x, NULL, reduced_z,
                                        reduced_affine_z, 1e-12,
                                        &kkt) == PREFOS_STATUS_OK);
    CHECK(kkt.passed);
    CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_REDUCED);
    CHECK(prefos_get_reduced_problem(presolver)->affine_cone_matrix.rows == 1);
    CHECK(prefos_verify_postsolve_full_kkt(presolver, reduced_x, NULL, reduced_z,
                                        reduced_affine_z, 1e-12,
                                        &kkt) == PREFOS_STATUS_OK);
    CHECK(kkt.passed);
    prefos_free_presolver(presolver);
    return 0;
}

static int test_input_affine_rsoc_face_substitution(void)
{
    int A_rows[] = {0};
    int Q_rows[] = {0, 0, 0};
    int R_rows[] = {0};
    double c[] = {1.0, 0.0};
    int box_indices[] = {0, 1};
    double box_lower[] = {-INFINITY, 0.0};
    double box_upper[] = {INFINITY, 1.0};
    double G_values[] = {1.0, 1.0};
    int G_columns[] = {0, 1};
    int G_rows[] = {0, 0, 0, 2};
    double h[] = {0.0, 1.0, 0.0};
    PreFOSAffineConeBlock block = {PREFOS_CONE_ROTATED_SECOND_ORDER, 3, 0, 0.0};
    PreFOSProblemData problem;
    PreFOSPresolver *presolver = NULL;
    const PreFOSPresolvedProblem *reduced;
    double reduced_x[] = {1.0};
    double reduced_z[] = {1.0};
    double reduced_affine_z[] = {0.0};
    double original_x[2], original_z[2], original_affine_z[3];
    PreFOSPostsolveKKTVerification kkt;

    memset(&problem, 0, sizeof(problem));
    problem.n = 2;
    problem.A = (PreFOSCsrMatrix){0, 2, 0, NULL, NULL, A_rows};
    problem.Q = (PreFOSCsrMatrix){2, 2, 0, NULL, NULL, Q_rows};
    problem.q_storage = PREFOS_Q_UPPER_TRIANGULAR;
    problem.R = (PreFOSCsrMatrix){0, 2, 0, NULL, NULL, R_rows};
    problem.c = c;
    problem.n_box = 2;
    problem.box_indices = box_indices;
    problem.box_lower = box_lower;
    problem.box_upper = box_upper;
    problem.affine_cone_matrix =
        (PreFOSCsrMatrix){3, 2, 2, G_values, G_columns, G_rows};
    problem.affine_cone_offset = h;
    problem.n_affine_cones = 1;
    problem.affine_cones = &block;

    CHECK(prefos_create_presolver(&problem, NULL, &presolver) == PREFOS_STATUS_OK);
    CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_REDUCED);
    reduced = prefos_get_reduced_problem(presolver);
    CHECK(reduced->n == 1 && reduced->n_affine_cones == 1);
    CHECK(reduced->affine_cones[0].type == PREFOS_CONE_NONNEGATIVE);
    CHECK(reduced->affine_cones[0].dimension == 1);
    CHECK(prefos_get_stats(presolver)->derived_affine_face_equalities == 1);
    CHECK(prefos_get_stats(presolver)->substituted_affine_face_variables == 1);
    CHECK(prefos_postsolve_full_primal_dual(presolver, reduced_x, NULL, reduced_z,
                                         reduced_affine_z, 1e-12, original_x, NULL,
                                         original_z, original_affine_z) ==
          PREFOS_STATUS_DUAL_RECOVERY_UNAVAILABLE);
    CHECK(prefos_postsolve_full_extended_dual(
              presolver, reduced_x, NULL, reduced_z, reduced_affine_z, 1e-12,
              original_x, NULL, original_z, original_affine_z) == PREFOS_STATUS_OK);
    CHECK(close_to(original_x[0], -1.0));
    CHECK(close_to(original_x[1], 1.0));
    CHECK(close_to(original_z[0], 0.0));
    CHECK(close_to(original_z[1], 1.0));
    CHECK(close_to(original_affine_z[0], 0.0));
    CHECK(close_to(original_affine_z[1], 0.0));
    CHECK(close_to(original_affine_z[2], -1.0));
    CHECK(prefos_verify_postsolve_full_extended_kkt(presolver, reduced_x, NULL,
                                                 reduced_z, reduced_affine_z, 1e-12,
                                                 &kkt) == PREFOS_STATUS_OK);
    CHECK(kkt.passed);
    prefos_free_presolver(presolver);
    return 0;
}

static int test_generated_affine_rsoc_face_fixing(void)
{
    double A_values[] = {1.0, 1.0, 1.0, -1.0};
    int A_columns[] = {0, 1, 2, 3};
    int A_rows[] = {0, 1, 2, 4};
    double sides[] = {0.0, 1.0, 0.0};
    int Q_rows[] = {0, 0, 0, 0, 0};
    int R_rows[] = {0};
    double c[] = {0.0, 0.0, 0.0, 1.0};
    int box_indices[] = {3};
    double box_lower[] = {-INFINITY};
    double box_upper[] = {INFINITY};
    int cone_indices[] = {0, 1, 2};
    PreFOSConeBlock cone = {PREFOS_CONE_ROTATED_SECOND_ORDER, 3, 0, cone_indices, 0.0};
    PreFOSProblemData problem;
    PreFOSSettings settings = prefos_default_settings();
    PreFOSPresolver *presolver = NULL;
    const PreFOSPresolvedProblem *reduced;
    double reduced_affine_z[] = {0.0};
    double original_x[4], original_y[3], original_z[4];
    PreFOSPostsolveKKTVerification kkt;

    memset(&problem, 0, sizeof(problem));
    problem.n = 4;
    problem.A = (PreFOSCsrMatrix){3, 4, 4, A_values, A_columns, A_rows};
    problem.constraint_lower = sides;
    problem.constraint_upper = sides;
    problem.Q = (PreFOSCsrMatrix){4, 4, 0, NULL, NULL, Q_rows};
    problem.q_storage = PREFOS_Q_UPPER_TRIANGULAR;
    problem.R = (PreFOSCsrMatrix){0, 4, 0, NULL, NULL, R_rows};
    problem.c = c;
    problem.n_box = 1;
    problem.box_indices = box_indices;
    problem.box_lower = box_lower;
    problem.box_upper = box_upper;
    problem.n_cones = 1;
    problem.cones = &cone;
    settings.affine_cone_coordinate_aggregation = 1;

    CHECK(prefos_create_presolver(&problem, &settings, &presolver) == PREFOS_STATUS_OK);
    CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_REDUCED);
    reduced = prefos_get_reduced_problem(presolver);
    CHECK(reduced->n == 0 && reduced->A.rows == 0);
    CHECK(reduced->n_cones == 0 && reduced->n_affine_cones == 1);
    CHECK(reduced->affine_cones[0].type == PREFOS_CONE_NONNEGATIVE);
    CHECK(prefos_get_stats(presolver)->derived_affine_face_equalities == 1);
    CHECK(prefos_get_stats(presolver)->fixed_affine_face_variables == 1);
    CHECK(prefos_postsolve_full_primal_dual(
              presolver, NULL, NULL, NULL, reduced_affine_z, 1e-12, original_x,
              original_y, original_z, NULL) == PREFOS_STATUS_DUAL_RECOVERY_UNAVAILABLE);
    CHECK(prefos_postsolve_full_extended_dual(
              presolver, NULL, NULL, NULL, reduced_affine_z, 1e-12, original_x,
              original_y, original_z, NULL) == PREFOS_STATUS_OK);
    CHECK(close_to(original_x[0], 0.0));
    CHECK(close_to(original_x[1], 1.0));
    CHECK(close_to(original_x[2], 0.0));
    CHECK(close_to(original_x[3], 0.0));
    CHECK(close_to(original_y[0], 0.0));
    CHECK(close_to(original_y[1], 0.0));
    CHECK(close_to(original_y[2], 1.0));
    CHECK(close_to(original_z[0], 0.0));
    CHECK(close_to(original_z[1], 0.0));
    CHECK(close_to(original_z[2], -1.0));
    CHECK(close_to(original_z[3], 0.0));
    CHECK(prefos_verify_postsolve_full_extended_kkt(presolver, NULL, NULL, NULL,
                                                 reduced_affine_z, 1e-12,
                                                 &kkt) == PREFOS_STATUS_OK);
    CHECK(kkt.passed);
    CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_REDUCED);
    CHECK(prefos_get_stats(presolver)->derived_affine_face_equalities == 1);
    CHECK(prefos_verify_postsolve_full_extended_kkt(presolver, NULL, NULL, NULL,
                                                 reduced_affine_z, 1e-12,
                                                 &kkt) == PREFOS_STATUS_OK);
    CHECK(kkt.passed);
    prefos_free_presolver(presolver);
    return 0;
}

static int test_duplicate_a_columns_are_rejected(void)
{
    double A_values[] = {1.0, 1.0};
    int A_columns[] = {0, 0};
    int A_rows[] = {0, 2};
    double constraint_lower[] = {-INFINITY};
    double constraint_upper[] = {1.0};
    int Q_rows[] = {0, 0, 0};
    int R_rows[] = {0};
    double c[] = {0.0, 0.0};
    int box_indices[] = {0, 1};
    double box_lower[] = {-INFINITY, -INFINITY};
    double box_upper[] = {INFINITY, INFINITY};
    PreFOSProblemData problem;
    PreFOSPresolver *presolver = NULL;

    memset(&problem, 0, sizeof(problem));
    problem.n = 2;
    problem.A = (PreFOSCsrMatrix){1, 2, 2, A_values, A_columns, A_rows};
    problem.constraint_lower = constraint_lower;
    problem.constraint_upper = constraint_upper;
    problem.Q = (PreFOSCsrMatrix){2, 2, 0, NULL, NULL, Q_rows};
    problem.q_storage = PREFOS_Q_UPPER_TRIANGULAR;
    problem.R = (PreFOSCsrMatrix){0, 2, 0, NULL, NULL, R_rows};
    problem.c = c;
    problem.n_box = 2;
    problem.box_indices = box_indices;
    problem.box_lower = box_lower;
    problem.box_upper = box_upper;

    CHECK(prefos_create_presolver(&problem, NULL, &presolver) ==
          PREFOS_STATUS_INVALID_ARGUMENT);
    CHECK(presolver == NULL);
    return 0;
}

static int test_psd_verification(void)
{
    int A_rows[] = {0};
    int Q_rows[] = {0, 0, 0, 0};
    int R_rows[] = {0};
    double c[] = {0.0, 0.0, 0.0};
    int cone_indices[] = {0, 1, 2};
    PreFOSConeBlock cone = {PREFOS_CONE_POSITIVE_SEMIDEFINITE, 3, 2, cone_indices};
    PreFOSProblemData problem;
    PreFOSPresolver *presolver = NULL;
    PreFOSPrimalVerification verification;
    double psd_x[] = {1.0, 0.5, 1.0};
    double indefinite_x[] = {1.0, 3.0, 1.0};

    memset(&problem, 0, sizeof(problem));
    problem.n = 3;
    problem.A = (PreFOSCsrMatrix){0, 3, 0, NULL, NULL, A_rows};
    problem.Q = (PreFOSCsrMatrix){3, 3, 0, NULL, NULL, Q_rows};
    problem.q_storage = PREFOS_Q_UPPER_TRIANGULAR;
    problem.R = (PreFOSCsrMatrix){0, 3, 0, NULL, NULL, R_rows};
    problem.c = c;
    problem.n_cones = 1;
    problem.cones = &cone;

    CHECK(prefos_create_presolver(&problem, NULL, &presolver) == PREFOS_STATUS_OK);
    CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_OK);
    CHECK(prefos_verify_postsolve_primal(presolver, psd_x, 1e-10, &verification) ==
          PREFOS_STATUS_OK);
    CHECK(verification.passed);
    CHECK(verification.original_cone_violation == 0.0);

    CHECK(prefos_verify_postsolve_primal(presolver, indefinite_x, 1e-10,
                                      &verification) == PREFOS_STATUS_OK);
    CHECK(!verification.passed);
    CHECK(verification.original_cone_violation > 0.9);

    prefos_free_presolver(presolver);
    return 0;
}

static int test_strict_settings(void)
{
    PreFOSSettings settings = prefos_strict_settings();
    CHECK(settings.feasibility_tolerance == 0.0);
    CHECK(settings.fixed_variable_tolerance == 0.0);
    CHECK(settings.finite_bound_improvement_absolute == 0.0);
    CHECK(settings.finite_bound_improvement_relative == 0.0);
    CHECK(settings.fix_close_box_bounds);
    CHECK(settings.linear_propagation);
    CHECK(settings.cone_propagation);
    CHECK(settings.cone_aware_row_activity);
    CHECK(settings.exponential_propagation);
    CHECK(settings.power_propagation);
    CHECK(settings.psd_higher_order_propagation);
    CHECK(settings.affine_psd_propagation_max_work_ratio == 0.0);
    CHECK(settings.psd_structure_analysis);
    CHECK(settings.psd_block_decomposition);
    return 0;
}

static int test_affine_psd_structure_and_block_decomposition(void)
{
    int A_rows[] = {0};
    int Q_rows[] = {0, 0, 0, 0, 0};
    int R_rows[] = {0};
    double c[] = {0.0, 0.0, 0.0, 0.0};
    int box_indices[] = {0, 1, 2, 3};
    double box_lower[] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};
    double box_upper[] = {INFINITY, INFINITY, INFINITY, INFINITY};
    double G_values[] = {1.0, 1.0, 1.0, 1.0};
    int G_columns[] = {0, 1, 2, 3};
    int G_rows[] = {0, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4};
    double h[] = {0.0, 0.25 * 1.4142135623730951, 0.0, 0.0, 0.0,
                  0.0, 0.0, 0.0, 0.25 * 1.4142135623730951, 0.0};
    PreFOSAffineConeBlock block = {PREFOS_CONE_POSITIVE_SEMIDEFINITE, 10, 4, 0.0};
    PreFOSProblemData problem;
    PreFOSSettings settings = prefos_strict_settings();
    PreFOSPresolver *presolver = NULL;
    const PreFOSPresolvedProblem *reduced;
    const PreFOSPSDStructureAnalysis *analysis;
    const PreFOSStats *stats;
    size_t analysis_count = 0;
    double reduced_x[] = {1.0, 1.0, 1.0, 1.0};
    double reduced_z[] = {0.0, 0.0, 0.0, 0.0};
    double reduced_affine_z[] = {-1.0, -2.0, -3.0, -4.0, -5.0, -6.0};
    double original_x[4], original_z[4], original_affine_z[10];
    PreFOSPrimalVerification verification;

    memset(&problem, 0, sizeof(problem));
    problem.n = 4;
    problem.A = (PreFOSCsrMatrix){0, 4, 0, NULL, NULL, A_rows};
    problem.Q = (PreFOSCsrMatrix){4, 4, 0, NULL, NULL, Q_rows};
    problem.q_storage = PREFOS_Q_UPPER_TRIANGULAR;
    problem.R = (PreFOSCsrMatrix){0, 4, 0, NULL, NULL, R_rows};
    problem.c = c;
    problem.n_box = 4;
    problem.box_indices = box_indices;
    problem.box_lower = box_lower;
    problem.box_upper = box_upper;
    problem.affine_cone_matrix =
        (PreFOSCsrMatrix){10, 4, 4, G_values, G_columns, G_rows};
    problem.affine_cone_offset = h;
    problem.n_affine_cones = 1;
    problem.affine_cones = &block;
    settings.cone_propagation = 0;

    CHECK(prefos_create_presolver(&problem, &settings, &presolver) == PREFOS_STATUS_OK);
    CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_REDUCED);
    reduced = prefos_get_reduced_problem(presolver);
    stats = prefos_get_stats(presolver);
    analysis = prefos_get_psd_structure_analyses(presolver, &analysis_count);
    CHECK(reduced != NULL && stats != NULL && analysis != NULL);
    CHECK(analysis_count == 1);
    CHECK(analysis[0].affine_cone_index == 0 && analysis[0].row_start == 0);
    CHECK(analysis[0].matrix_order == 4 && analysis[0].dimension == 10);
    CHECK(analysis[0].affine_nnz == 4);
    CHECK(analysis[0].active_diagonal_coordinates == 4);
    CHECK(analysis[0].active_offdiagonal_coordinates == 2);
    CHECK(analysis[0].coefficient_columns == 4);
    CHECK(analysis[0].diagonal_coefficient_columns == 4);
    CHECK(analysis[0].single_diagonal_coefficient_columns == 4);
    CHECK(analysis[0].connected_components == 2);
    CHECK(analysis[0].scalar_components == 0);
    CHECK(analysis[0].emitted_cone_blocks == 2);
    CHECK(analysis[0].largest_component_order == 2);
    CHECK(analysis[0].decomposed_dimension == 6);
    CHECK(analysis[0].exactly_block_diagonal);
    CHECK(reduced->n_affine_cones == 2);
    CHECK(reduced->affine_cone_matrix.rows == 6);
    CHECK(reduced->affine_cone_matrix.nnz == 4);
    CHECK(reduced->affine_cones[0].type == PREFOS_CONE_POSITIVE_SEMIDEFINITE);
    CHECK(reduced->affine_cones[0].matrix_order == 2);
    CHECK(reduced->affine_cones[0].dimension == 3);
    CHECK(reduced->affine_cones[1].type == PREFOS_CONE_POSITIVE_SEMIDEFINITE);
    CHECK(reduced->affine_cones[1].matrix_order == 2);
    CHECK(reduced->affine_cones[1].dimension == 3);
    CHECK(stats->affine_psd_blocks_analyzed == 1);
    CHECK(stats->affine_psd_splittable_blocks == 1);
    CHECK(stats->decomposed_affine_psd_blocks == 1);
    CHECK(stats->affine_psd_component_blocks == 2);
    CHECK(stats->removed_affine_psd_cross_coordinates == 4);
    CHECK(prefos_verify_postsolve_primal(presolver, reduced_x, 1e-12,
                                      &verification) == PREFOS_STATUS_OK);
    CHECK(verification.passed);
    CHECK(prefos_postsolve_full_primal_dual(
              presolver, reduced_x, NULL, reduced_z, reduced_affine_z, 1e-12,
              original_x, NULL, original_z,
              original_affine_z) == PREFOS_STATUS_OK);
    CHECK(close_to(original_affine_z[0], -1.0));
    CHECK(close_to(original_affine_z[1], -2.0));
    CHECK(close_to(original_affine_z[2], -3.0));
    CHECK(close_to(original_affine_z[3], 0.0));
    CHECK(close_to(original_affine_z[4], 0.0));
    CHECK(close_to(original_affine_z[5], -4.0));
    CHECK(close_to(original_affine_z[6], 0.0));
    CHECK(close_to(original_affine_z[7], 0.0));
    CHECK(close_to(original_affine_z[8], -5.0));
    CHECK(close_to(original_affine_z[9], -6.0));
    reduced_x[1] = -1.0;
    CHECK(prefos_verify_postsolve_primal(presolver, reduced_x, 1e-12,
                                      &verification) == PREFOS_STATUS_OK);
    CHECK(!verification.passed);
    CHECK(close_to(verification.original_cone_violation,
                   verification.reduced_cone_violation));
    reduced_x[1] = 1.0;
    prefos_free_presolver(presolver);

    settings.psd_block_decomposition = 0;
    CHECK(prefos_create_presolver(&problem, &settings, &presolver) == PREFOS_STATUS_OK);
    CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_OK);
    reduced = prefos_get_reduced_problem(presolver);
    stats = prefos_get_stats(presolver);
    analysis = prefos_get_psd_structure_analyses(presolver, &analysis_count);
    CHECK(reduced != NULL && stats != NULL && analysis != NULL);
    CHECK(analysis_count == 1 && analysis[0].exactly_block_diagonal);
    CHECK(reduced->n_affine_cones == 1);
    CHECK(reduced->affine_cone_matrix.rows == 10);
    CHECK(stats->affine_psd_splittable_blocks == 1);
    CHECK(stats->decomposed_affine_psd_blocks == 0);
    prefos_free_presolver(presolver);

    settings.psd_block_decomposition = 1;
    h[1] = 0.0;
    h[8] = 0.0;
    CHECK(prefos_create_presolver(&problem, &settings, &presolver) == PREFOS_STATUS_OK);
    CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_REDUCED);
    reduced = prefos_get_reduced_problem(presolver);
    analysis = prefos_get_psd_structure_analyses(presolver, &analysis_count);
    CHECK(reduced != NULL && analysis != NULL && analysis_count == 1);
    CHECK(analysis[0].connected_components == 4);
    CHECK(analysis[0].scalar_components == 4);
    CHECK(analysis[0].emitted_cone_blocks == 1);
    CHECK(analysis[0].largest_component_order == 1);
    CHECK(analysis[0].decomposed_dimension == 4);
    CHECK(reduced->n_affine_cones == 1);
    CHECK(reduced->affine_cones[0].type == PREFOS_CONE_NONNEGATIVE);
    CHECK(reduced->affine_cones[0].dimension == 4);
    CHECK(reduced->affine_cone_matrix.rows == 4);
    CHECK(reduced->affine_cone_matrix.nnz == 4);
    CHECK(prefos_get_stats(presolver)->affine_psd_scalar_components == 4);
    CHECK(prefos_get_stats(presolver)->affine_psd_component_blocks == 1);
    CHECK(prefos_get_stats(presolver)->removed_affine_psd_cross_coordinates == 6);
    CHECK(prefos_verify_postsolve_primal(presolver, reduced_x, 1e-12,
                                      &verification) == PREFOS_STATUS_OK);
    CHECK(verification.passed);
    CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_REDUCED);
    reduced = prefos_get_reduced_problem(presolver);
    analysis = prefos_get_psd_structure_analyses(presolver, &analysis_count);
    CHECK(reduced != NULL && reduced->affine_cone_matrix.rows == 4);
    CHECK(analysis != NULL && analysis_count == 1);
    CHECK(analysis[0].scalar_components == 4);
    prefos_free_presolver(presolver);
    return 0;
}

static int test_affine_psd_propagation_budget(void)
{
    int A_rows[] = {0};
    int Q_rows[] = {0, 0};
    int R_rows[] = {0};
    double c[] = {0.0};
    int box_indices[] = {0};
    double box_lower[] = {-INFINITY};
    double box_upper[] = {INFINITY};
    double G_values[] = {1.0};
    int G_columns[] = {0};
    int G_rows[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1};
    double h[10] = {0.0};
    PreFOSAffineConeBlock block = {PREFOS_CONE_POSITIVE_SEMIDEFINITE, 10, 4, 0.0};
    PreFOSProblemData problem;
    PreFOSSettings settings = prefos_strict_settings();
    PreFOSPresolver *presolver = NULL;

    memset(&problem, 0, sizeof(problem));
    problem.n = 1;
    problem.A = (PreFOSCsrMatrix){0, 1, 0, NULL, NULL, A_rows};
    problem.Q = (PreFOSCsrMatrix){1, 1, 0, NULL, NULL, Q_rows};
    problem.q_storage = PREFOS_Q_UPPER_TRIANGULAR;
    problem.R = (PreFOSCsrMatrix){0, 1, 0, NULL, NULL, R_rows};
    problem.c = c;
    problem.n_box = 1;
    problem.box_indices = box_indices;
    problem.box_lower = box_lower;
    problem.box_upper = box_upper;
    problem.affine_cone_matrix =
        (PreFOSCsrMatrix){10, 1, 1, G_values, G_columns, G_rows};
    problem.affine_cone_offset = h;
    problem.n_affine_cones = 1;
    problem.affine_cones = &block;
    settings.affine_psd_propagation_max_work_ratio = 1.0;

    CHECK(prefos_create_presolver(&problem, &settings, &presolver) == PREFOS_STATUS_OK);
    CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_REDUCED);
    CHECK(prefos_get_stats(presolver)->affine_psd_budget_skips == 1);
    CHECK(prefos_get_stats(presolver)->affine_psd_coordinates_skipped == 10);
    CHECK(prefos_get_stats(presolver)->reduced_affine_psd_faces == 1);
    prefos_free_presolver(presolver);
    return 0;
}

static int test_fixed_variable_kkt_postsolve(void)
{
    int A_rows[] = {0};
    double Q_values[] = {1.0, 1.0};
    int Q_columns[] = {0, 1};
    int Q_rows[] = {0, 1, 2};
    int R_rows[] = {0};
    double c[] = {0.0, -3.0};
    int box_indices[] = {0, 1};
    double box_lower[] = {2.0, -INFINITY};
    double box_upper[] = {2.0, INFINITY};
    double reduced_x[] = {3.0};
    double reduced_z[] = {0.0};
    double original_x[2], original_z[2];
    PreFOSProblemData problem;
    PreFOSSettings settings = prefos_strict_settings();
    PreFOSPresolver *presolver = NULL;
    PreFOSPostsolveKKTVerification verification;

    memset(&problem, 0, sizeof(problem));
    problem.n = 2;
    problem.A = (PreFOSCsrMatrix){0, 2, 0, NULL, NULL, A_rows};
    problem.Q = (PreFOSCsrMatrix){2, 2, 2, Q_values, Q_columns, Q_rows};
    problem.q_storage = PREFOS_Q_UPPER_TRIANGULAR;
    problem.R = (PreFOSCsrMatrix){0, 2, 0, NULL, NULL, R_rows};
    problem.c = c;
    problem.n_box = 2;
    problem.box_indices = box_indices;
    problem.box_lower = box_lower;
    problem.box_upper = box_upper;

    CHECK(prefos_create_presolver(&problem, &settings, &presolver) == PREFOS_STATUS_OK);
    CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_REDUCED);
    CHECK(prefos_postsolve_primal_dual(presolver, reduced_x, NULL, reduced_z, 1e-10,
                                    original_x, NULL, original_z) == PREFOS_STATUS_OK);
    CHECK(close_to(original_x[0], 2.0));
    CHECK(close_to(original_x[1], 3.0));
    CHECK(close_to(original_z[0], -2.0));
    CHECK(close_to(original_z[1], 0.0));
    CHECK(prefos_verify_postsolve_kkt(presolver, reduced_x, NULL, reduced_z, 1e-10,
                                   &verification) == PREFOS_STATUS_OK);
    CHECK(verification.reduced.passed);
    CHECK(verification.original.passed);
    CHECK(verification.passed);

    prefos_free_presolver(presolver);
    return 0;
}

static int test_propagation_dual_transfer(void)
{
    double A_values[] = {1.0, 1.0};
    int A_columns[] = {0, 1};
    int A_rows[] = {0, 2};
    double constraint_lower[] = {4.0};
    double constraint_upper[] = {INFINITY};
    int Q_rows[] = {0, 0, 0};
    int R_rows[] = {0};
    double c[] = {1.0, 0.0};
    int box_indices[] = {0, 1};
    double box_lower[] = {0.0, 0.0};
    double box_upper[] = {10.0, 1.0};
    double reduced_x[] = {3.0, 1.0};
    double reduced_y[] = {0.0};
    double reduced_z[] = {-1.0, 0.0};
    double original_x[2], original_y[1], original_z[2];
    PreFOSProblemData problem;
    PreFOSSettings settings = prefos_strict_settings();
    PreFOSPresolver *presolver = NULL;
    PreFOSPostsolveKKTVerification verification;

    memset(&problem, 0, sizeof(problem));
    problem.n = 2;
    problem.A = (PreFOSCsrMatrix){1, 2, 2, A_values, A_columns, A_rows};
    problem.constraint_lower = constraint_lower;
    problem.constraint_upper = constraint_upper;
    problem.Q = (PreFOSCsrMatrix){2, 2, 0, NULL, NULL, Q_rows};
    problem.q_storage = PREFOS_Q_UPPER_TRIANGULAR;
    problem.R = (PreFOSCsrMatrix){0, 2, 0, NULL, NULL, R_rows};
    problem.c = c;
    problem.n_box = 2;
    problem.box_indices = box_indices;
    problem.box_lower = box_lower;
    problem.box_upper = box_upper;

    CHECK(prefos_create_presolver(&problem, &settings, &presolver) == PREFOS_STATUS_OK);
    CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_REDUCED);
    CHECK(prefos_postsolve_primal_dual(presolver, reduced_x, reduced_y, reduced_z,
                                    1e-10, original_x, original_y,
                                    original_z) == PREFOS_STATUS_OK);
    CHECK(close_to(original_y[0], -1.0));
    CHECK(close_to(original_z[0], 0.0));
    CHECK(close_to(original_z[1], 1.0));
    CHECK(prefos_verify_postsolve_kkt(presolver, reduced_x, reduced_y, reduced_z, 1e-10,
                                   &verification) == PREFOS_STATUS_OK);
    CHECK(verification.reduced.passed);
    CHECK(verification.original.passed);
    CHECK(verification.passed);

    prefos_free_presolver(presolver);
    return 0;
}

static int test_cyclic_propagation_dual_transfer(void)
{
    /* Seed 291 has two simultaneously active, mutually dependent bounds. */
    double A_values[] = {
        0.8805258751206381,   1.531353397833695,   0.778432489358002,
        -0.6717191732894124,  0.9876155150184114,  -1.1770751691937285,
        -0.04709566854999819, 0.04257203686731929, 1.6038129723913135};
    int A_columns[] = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    int A_rows[] = {0, 3, 6, 9};
    double constraint_lower[] = {-INFINITY, -2.0312864777022854,
                                 -0.7421171104235635};
    double constraint_upper[] = {-0.652548717994005, INFINITY, 0.9193834932198093};
    double Q_values[] = {0.12884368469073437, -0.00150669778666853,
                         -0.0224778221216516, 0.2755350079725075,
                         0.06411302271872042, 0.18917331412960764};
    int Q_columns[] = {0, 1, 2, 1, 2, 2};
    int Q_rows[] = {0, 3, 5, 6};
    double R_values[] = {-0.0094006785174104, -0.2671566641462189,
                         0.06886980835917704};
    int R_columns[] = {0, 1, 2};
    int R_rows[] = {0, 3};
    double D[] = {0.36714411126718594};
    double c[] = {-0.04307161059341403, -0.10506143223177274, 1.2121065354885103};
    int box_indices[] = {0, 1, 2};
    double box_lower[] = {0.48861340185795465, -2.0, -2.0};
    double box_upper[] = {0.48861340185795465, 2.0, 2.0};
    double reduced_x[] = {-0.48571020782096624, -0.43547961934820184};
    double reduced_y[] = {0.04002119792127062, 0.0, -0.3476939478938626};
    double reduced_z[] = {0.2303984877016704, -0.5635263741232248};
    double original_x[3], original_y[3], original_z[3];
    PreFOSProblemData problem;
    PreFOSSettings settings = prefos_strict_settings();
    PreFOSPresolver *presolver = NULL;
    PreFOSPostsolveKKTVerification verification;

    memset(&problem, 0, sizeof(problem));
    problem.n = 3;
    problem.A = (PreFOSCsrMatrix){3, 3, 9, A_values, A_columns, A_rows};
    problem.constraint_lower = constraint_lower;
    problem.constraint_upper = constraint_upper;
    problem.Q = (PreFOSCsrMatrix){3, 3, 6, Q_values, Q_columns, Q_rows};
    problem.q_storage = PREFOS_Q_UPPER_TRIANGULAR;
    problem.R = (PreFOSCsrMatrix){1, 3, 3, R_values, R_columns, R_rows};
    problem.D = D;
    problem.c = c;
    problem.objective_offset = -0.3805334550323097;
    problem.n_box = 3;
    problem.box_indices = box_indices;
    problem.box_lower = box_lower;
    problem.box_upper = box_upper;

    CHECK(prefos_create_presolver(&problem, &settings, &presolver) == PREFOS_STATUS_OK);
    CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_REDUCED);
    CHECK(prefos_get_reduced_problem(presolver)->n == 2);
    CHECK(prefos_postsolve_primal_dual(presolver, reduced_x, reduced_y, reduced_z, 5e-5,
                                    original_x, original_y,
                                    original_z) == PREFOS_STATUS_OK);
    CHECK(fabs(original_z[1]) <= 5e-5);
    CHECK(fabs(original_z[2]) <= 5e-5);
    CHECK(prefos_verify_postsolve_kkt(presolver, reduced_x, reduced_y, reduced_z, 5e-5,
                                   &verification) == PREFOS_STATUS_OK);
    CHECK(verification.passed);

    prefos_free_presolver(presolver);
    return 0;
}

static int test_deleted_singleton_row_dual_transfer(void)
{
    double A_values[] = {1.0};
    int A_columns[] = {0};
    int A_rows[] = {0, 1};
    double constraint_lower[] = {4.0};
    double constraint_upper[] = {INFINITY};
    int Q_rows[] = {0, 0};
    int R_rows[] = {0};
    double c[] = {1.0};
    int box_indices[] = {0};
    double box_lower[] = {0.0};
    double box_upper[] = {10.0};
    double reduced_x[] = {4.0};
    double reduced_z[] = {-1.0};
    double original_x[1], original_y[1], original_z[1];
    PreFOSProblemData problem;
    PreFOSSettings settings = prefos_strict_settings();
    PreFOSPresolver *presolver = NULL;
    PreFOSPostsolveKKTVerification verification;

    memset(&problem, 0, sizeof(problem));
    problem.n = 1;
    problem.A = (PreFOSCsrMatrix){1, 1, 1, A_values, A_columns, A_rows};
    problem.constraint_lower = constraint_lower;
    problem.constraint_upper = constraint_upper;
    problem.Q = (PreFOSCsrMatrix){1, 1, 0, NULL, NULL, Q_rows};
    problem.q_storage = PREFOS_Q_UPPER_TRIANGULAR;
    problem.R = (PreFOSCsrMatrix){0, 1, 0, NULL, NULL, R_rows};
    problem.c = c;
    problem.n_box = 1;
    problem.box_indices = box_indices;
    problem.box_lower = box_lower;
    problem.box_upper = box_upper;

    CHECK(prefos_create_presolver(&problem, &settings, &presolver) == PREFOS_STATUS_OK);
    CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_REDUCED);
    CHECK(prefos_postsolve_primal_dual(presolver, reduced_x, NULL, reduced_z, 1e-10,
                                    original_x, original_y,
                                    original_z) == PREFOS_STATUS_OK);
    CHECK(close_to(original_y[0], -1.0));
    CHECK(close_to(original_z[0], 0.0));
    CHECK(prefos_verify_postsolve_kkt(presolver, reduced_x, NULL, reduced_z, 1e-10,
                                   &verification) == PREFOS_STATUS_OK);
    CHECK(verification.passed);

    prefos_free_presolver(presolver);
    return 0;
}

static int test_soc_kkt_convention(void)
{
    int A_rows[] = {0};
    int Q_rows[] = {0, 0, 0};
    int R_rows[] = {0};
    double c[] = {1.0, 0.0};
    int cone_indices[] = {0, 1};
    PreFOSConeBlock cone = {PREFOS_CONE_SECOND_ORDER, 2, 0, cone_indices};
    double reduced_x[] = {0.0, 0.0};
    double reduced_z[] = {-1.0, 0.0};
    PreFOSProblemData problem;
    PreFOSPresolver *presolver = NULL;
    PreFOSPostsolveKKTVerification verification;

    memset(&problem, 0, sizeof(problem));
    problem.n = 2;
    problem.A = (PreFOSCsrMatrix){0, 2, 0, NULL, NULL, A_rows};
    problem.Q = (PreFOSCsrMatrix){2, 2, 0, NULL, NULL, Q_rows};
    problem.q_storage = PREFOS_Q_UPPER_TRIANGULAR;
    problem.R = (PreFOSCsrMatrix){0, 2, 0, NULL, NULL, R_rows};
    problem.c = c;
    problem.n_cones = 1;
    problem.cones = &cone;

    CHECK(prefos_create_presolver(&problem, NULL, &presolver) == PREFOS_STATUS_OK);
    CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_OK);
    CHECK(prefos_verify_postsolve_kkt(presolver, reduced_x, NULL, reduced_z, 1e-10,
                                   &verification) == PREFOS_STATUS_OK);
    CHECK(verification.reduced.passed);
    CHECK(verification.original.passed);
    CHECK(verification.passed);

    prefos_free_presolver(presolver);
    return 0;
}

static int test_soc_envelope_propagation(void)
{
    double A_values[] = {1.0, 1.0};
    int A_columns[] = {0, 1};
    int A_rows[] = {0, 1, 2};
    double constraint_lower[] = {-INFINITY, 1.0};
    double constraint_upper[] = {2.0, INFINITY};
    int Q_rows[] = {0, 0, 0};
    int R_rows[] = {0};
    double c[] = {0.0, 0.0};
    int cone_indices[] = {0, 1};
    PreFOSConeBlock cone = {PREFOS_CONE_SECOND_ORDER, 2, 0, cone_indices};
    PreFOSProblemData problem;
    PreFOSSettings settings = prefos_strict_settings();
    PreFOSPresolver *presolver = NULL;
    const PreFOSStats *stats;

    memset(&problem, 0, sizeof(problem));
    problem.n = 2;
    problem.A = (PreFOSCsrMatrix){2, 2, 2, A_values, A_columns, A_rows};
    problem.constraint_lower = constraint_lower;
    problem.constraint_upper = constraint_upper;
    problem.Q = (PreFOSCsrMatrix){2, 2, 0, NULL, NULL, Q_rows};
    problem.q_storage = PREFOS_Q_UPPER_TRIANGULAR;
    problem.R = (PreFOSCsrMatrix){0, 2, 0, NULL, NULL, R_rows};
    problem.c = c;
    problem.n_cones = 1;
    problem.cones = &cone;

    CHECK(prefos_create_presolver(&problem, &settings, &presolver) == PREFOS_STATUS_OK);
    CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_OK);
    stats = prefos_get_stats(presolver);
    CHECK(stats != NULL);
    CHECK(stats->tightened_cone_envelopes >= 4);
    CHECK(stats->cone_propagation_rounds == 2);
    prefos_free_presolver(presolver);
    return 0;
}

static int test_soc_and_rsoc_infeasibility(void)
{
    {
        double A_values[] = {1.0, 1.0};
        int A_columns[] = {0, 1};
        int A_rows[] = {0, 1, 2};
        double lower[] = {-INFINITY, 2.0};
        double upper[] = {1.0, INFINITY};
        int Q_rows[] = {0, 0, 0};
        int R_rows[] = {0};
        double c[] = {0.0, 0.0};
        int cone_indices[] = {0, 1};
        PreFOSConeBlock cone = {PREFOS_CONE_SECOND_ORDER, 2, 0, cone_indices};
        PreFOSProblemData problem;
        PreFOSPresolver *presolver = NULL;
        memset(&problem, 0, sizeof(problem));
        problem.n = 2;
        problem.A = (PreFOSCsrMatrix){2, 2, 2, A_values, A_columns, A_rows};
        problem.constraint_lower = lower;
        problem.constraint_upper = upper;
        problem.Q = (PreFOSCsrMatrix){2, 2, 0, NULL, NULL, Q_rows};
        problem.q_storage = PREFOS_Q_UPPER_TRIANGULAR;
        problem.R = (PreFOSCsrMatrix){0, 2, 0, NULL, NULL, R_rows};
        problem.c = c;
        problem.n_cones = 1;
        problem.cones = &cone;
        CHECK(prefos_create_presolver(&problem, NULL, &presolver) == PREFOS_STATUS_OK);
        CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_PRIMAL_INFEASIBLE);
        prefos_free_presolver(presolver);
    }
    {
        double A_values[] = {1.0, 1.0, 1.0};
        int A_columns[] = {0, 1, 2};
        int A_rows[] = {0, 1, 2, 3};
        double lower[] = {-INFINITY, -INFINITY, 2.0};
        double upper[] = {1.0, 1.0, INFINITY};
        int Q_rows[] = {0, 0, 0, 0};
        int R_rows[] = {0};
        double c[] = {0.0, 0.0, 0.0};
        int cone_indices[] = {0, 1, 2};
        PreFOSConeBlock cone = {PREFOS_CONE_ROTATED_SECOND_ORDER, 3, 0, cone_indices};
        PreFOSProblemData problem;
        PreFOSPresolver *presolver = NULL;
        memset(&problem, 0, sizeof(problem));
        problem.n = 3;
        problem.A = (PreFOSCsrMatrix){3, 3, 3, A_values, A_columns, A_rows};
        problem.constraint_lower = lower;
        problem.constraint_upper = upper;
        problem.Q = (PreFOSCsrMatrix){3, 3, 0, NULL, NULL, Q_rows};
        problem.q_storage = PREFOS_Q_UPPER_TRIANGULAR;
        problem.R = (PreFOSCsrMatrix){0, 3, 0, NULL, NULL, R_rows};
        problem.c = c;
        problem.n_cones = 1;
        problem.cones = &cone;
        CHECK(prefos_create_presolver(&problem, NULL, &presolver) == PREFOS_STATUS_OK);
        CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_PRIMAL_INFEASIBLE);
        prefos_free_presolver(presolver);
    }
    return 0;
}

static int test_psd_pairwise_and_full_infeasibility(void)
{
    {
        double A_values[] = {1.0, 1.0, 1.0};
        int A_columns[] = {0, 1, 2};
        int A_rows[] = {0, 1, 2, 3};
        double lower[] = {-INFINITY, 2.0, -INFINITY};
        double upper[] = {1.0, INFINITY, 1.0};
        int Q_rows[] = {0, 0, 0, 0};
        int R_rows[] = {0};
        double c[] = {0.0, 0.0, 0.0};
        int cone_indices[] = {0, 1, 2};
        PreFOSConeBlock cone = {PREFOS_CONE_POSITIVE_SEMIDEFINITE, 3, 2, cone_indices};
        PreFOSProblemData problem;
        PreFOSPresolver *presolver = NULL;
        memset(&problem, 0, sizeof(problem));
        problem.n = 3;
        problem.A = (PreFOSCsrMatrix){3, 3, 3, A_values, A_columns, A_rows};
        problem.constraint_lower = lower;
        problem.constraint_upper = upper;
        problem.Q = (PreFOSCsrMatrix){3, 3, 0, NULL, NULL, Q_rows};
        problem.q_storage = PREFOS_Q_UPPER_TRIANGULAR;
        problem.R = (PreFOSCsrMatrix){0, 3, 0, NULL, NULL, R_rows};
        problem.c = c;
        problem.n_cones = 1;
        problem.cones = &cone;
        CHECK(prefos_create_presolver(&problem, NULL, &presolver) == PREFOS_STATUS_OK);
        CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_PRIMAL_INFEASIBLE);
        prefos_free_presolver(presolver);
    }
    {
        const double sqrt_two = sqrt(2.0);
        double fixed_matrix[] = {1.0, sqrt_two, 1.0, sqrt_two, -sqrt_two, 1.0};
        double A_values[] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
        int A_columns[] = {0, 1, 2, 3, 4, 5};
        int A_rows[] = {0, 1, 2, 3, 4, 5, 6};
        int Q_rows[] = {0, 0, 0, 0, 0, 0, 0};
        int R_rows[] = {0};
        double c[] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        int cone_indices[] = {0, 1, 2, 3, 4, 5};
        PreFOSConeBlock cone = {PREFOS_CONE_POSITIVE_SEMIDEFINITE, 6, 3, cone_indices};
        PreFOSProblemData problem;
        PreFOSPresolver *presolver = NULL;
        memset(&problem, 0, sizeof(problem));
        problem.n = 6;
        problem.A = (PreFOSCsrMatrix){6, 6, 6, A_values, A_columns, A_rows};
        problem.constraint_lower = fixed_matrix;
        problem.constraint_upper = fixed_matrix;
        problem.Q = (PreFOSCsrMatrix){6, 6, 0, NULL, NULL, Q_rows};
        problem.q_storage = PREFOS_Q_UPPER_TRIANGULAR;
        problem.R = (PreFOSCsrMatrix){0, 6, 0, NULL, NULL, R_rows};
        problem.c = c;
        problem.n_cones = 1;
        problem.cones = &cone;
        CHECK(prefos_create_presolver(&problem, NULL, &presolver) == PREFOS_STATUS_OK);
        CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_PRIMAL_INFEASIBLE);
        prefos_free_presolver(presolver);
    }
    return 0;
}

static int test_nonnegative_box_normalization(void)
{
    {
        double A_values[] = {1.0, 1.0};
        int A_columns[] = {0, 1};
        int A_rows[] = {0, 1, 2};
        double lower[] = {-INFINITY, -INFINITY};
        double upper[] = {0.0, 0.0};
        int Q_rows[] = {0, 0, 0};
        int R_rows[] = {0};
        double c[] = {-1.0, 2.0};
        int cone_indices[] = {0, 1};
        PreFOSConeBlock cone = {PREFOS_CONE_NONNEGATIVE, 2, 0, cone_indices};
        PreFOSProblemData problem;
        PreFOSPresolver *presolver = NULL;
        const PreFOSStats *stats;
        PreFOSPostsolveKKTVerification verification;

        memset(&problem, 0, sizeof(problem));
        problem.n = 2;
        problem.A = (PreFOSCsrMatrix){2, 2, 2, A_values, A_columns, A_rows};
        problem.constraint_lower = lower;
        problem.constraint_upper = upper;
        problem.Q = (PreFOSCsrMatrix){2, 2, 0, NULL, NULL, Q_rows};
        problem.q_storage = PREFOS_Q_UPPER_TRIANGULAR;
        problem.R = (PreFOSCsrMatrix){0, 2, 0, NULL, NULL, R_rows};
        problem.c = c;
        problem.n_cones = 1;
        problem.cones = &cone;

        CHECK(prefos_create_presolver(&problem, NULL, &presolver) == PREFOS_STATUS_OK);
        CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_REDUCED);
        CHECK(prefos_get_reduced_problem(presolver)->n == 0);
        stats = prefos_get_stats(presolver);
        CHECK(stats->normalized_nonnegative_cones == 1);
        CHECK(stats->normalized_nonnegative_variables == 2);
        CHECK(stats->fixed_box_variables == 2);
        CHECK(stats->fixed_cone_variables == 0 && stats->collapsed_cones == 0);
        CHECK(prefos_verify_postsolve_kkt(presolver, NULL, NULL, NULL, 1e-10,
                                       &verification) == PREFOS_STATUS_OK);
        CHECK(verification.passed);
        prefos_free_presolver(presolver);
    }
    {
        double A_values[] = {1.0};
        int A_columns[] = {0};
        int A_rows[] = {0, 1};
        double lower[] = {-INFINITY};
        double upper[] = {0.0};
        int Q_rows[] = {0, 0, 0};
        int R_rows[] = {0};
        double c[] = {0.0, 0.0};
        int cone_indices[] = {0, 1};
        PreFOSConeBlock cone = {PREFOS_CONE_NONNEGATIVE, 2, 0, cone_indices};
        PreFOSProblemData problem;
        PreFOSPresolver *presolver = NULL;
        const PreFOSPresolvedProblem *reduced;
        const PreFOSStats *stats;
        double reduced_x[] = {1.0};
        double reduced_z[] = {0.0};
        PreFOSPostsolveKKTVerification verification;

        memset(&problem, 0, sizeof(problem));
        problem.n = 2;
        problem.A = (PreFOSCsrMatrix){1, 2, 1, A_values, A_columns, A_rows};
        problem.constraint_lower = lower;
        problem.constraint_upper = upper;
        problem.Q = (PreFOSCsrMatrix){2, 2, 0, NULL, NULL, Q_rows};
        problem.q_storage = PREFOS_Q_UPPER_TRIANGULAR;
        problem.R = (PreFOSCsrMatrix){0, 2, 0, NULL, NULL, R_rows};
        problem.c = c;
        problem.n_cones = 1;
        problem.cones = &cone;

        CHECK(prefos_create_presolver(&problem, NULL, &presolver) == PREFOS_STATUS_OK);
        CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_REDUCED);
        reduced = prefos_get_reduced_problem(presolver);
        stats = prefos_get_stats(presolver);
        CHECK(reduced != NULL && stats != NULL);
        CHECK(reduced->n == 1 && reduced->n_box == 1 && reduced->n_cones == 0);
        CHECK(reduced->box_indices[0] == 0);
        CHECK(close_to(reduced->box_lower[0], 0.0));
        CHECK(reduced->box_upper[0] == INFINITY);
        CHECK(stats->normalized_nonnegative_cones == 1);
        CHECK(stats->normalized_nonnegative_variables == 2);
        CHECK(stats->fixed_box_variables == 1);
        CHECK(stats->fixed_cone_variables == 0 && stats->collapsed_cones == 0);
        CHECK(prefos_verify_postsolve_kkt(presolver, reduced_x, NULL, reduced_z, 1e-10,
                                       &verification) == PREFOS_STATUS_OK);
        CHECK(verification.passed);
        prefos_free_presolver(presolver);
    }
    {
        int A_rows[] = {0};
        int Q_rows[10] = {0};
        int R_rows[] = {0};
        double c[9] = {0.0};
        int box_indices[] = {8};
        double box_lower[] = {-1.0};
        double box_upper[] = {1.0};
        int soc_indices[] = {0, 1};
        int first_nonnegative_indices[] = {2, 3};
        int rsoc_indices[] = {4, 5, 6};
        int second_nonnegative_indices[] = {7};
        PreFOSConeBlock cones[] = {
            {PREFOS_CONE_SECOND_ORDER, 2, 0, soc_indices},
            {PREFOS_CONE_NONNEGATIVE, 2, 0, first_nonnegative_indices},
            {PREFOS_CONE_ROTATED_SECOND_ORDER, 3, 0, rsoc_indices},
            {PREFOS_CONE_NONNEGATIVE, 1, 0, second_nonnegative_indices}};
        PreFOSProblemData problem;
        PreFOSPresolver *presolver = NULL;
        const PreFOSPresolvedProblem *reduced;
        const PreFOSStats *stats;
        double reduced_x[] = {1.0, 0.0, 1.0, 2.0, 1.0, 1.0, 0.0, 3.0, 0.0};
        double reduced_z[9] = {0.0};
        PreFOSPostsolveKKTVerification verification;

        memset(&problem, 0, sizeof(problem));
        problem.n = 9;
        problem.A = (PreFOSCsrMatrix){0, 9, 0, NULL, NULL, A_rows};
        problem.Q = (PreFOSCsrMatrix){9, 9, 0, NULL, NULL, Q_rows};
        problem.q_storage = PREFOS_Q_UPPER_TRIANGULAR;
        problem.R = (PreFOSCsrMatrix){0, 9, 0, NULL, NULL, R_rows};
        problem.c = c;
        problem.n_box = 1;
        problem.box_indices = box_indices;
        problem.box_lower = box_lower;
        problem.box_upper = box_upper;
        problem.n_cones = 4;
        problem.cones = cones;

        CHECK(prefos_create_presolver(&problem, NULL, &presolver) == PREFOS_STATUS_OK);
        CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_REDUCED);
        reduced = prefos_get_reduced_problem(presolver);
        stats = prefos_get_stats(presolver);
        CHECK(reduced != NULL && stats != NULL);
        CHECK(reduced->n == 9 && reduced->n_box == 4 && reduced->n_cones == 2);
        CHECK(reduced->box_indices[0] == 8 && reduced->box_indices[1] == 2);
        CHECK(reduced->box_indices[2] == 3 && reduced->box_indices[3] == 7);
        CHECK(reduced->cones[0].type == PREFOS_CONE_SECOND_ORDER);
        CHECK(reduced->cones[1].type == PREFOS_CONE_ROTATED_SECOND_ORDER);
        CHECK(stats->normalized_nonnegative_cones == 2);
        CHECK(stats->normalized_nonnegative_variables == 3);
        CHECK(prefos_verify_postsolve_kkt(presolver, reduced_x, NULL, reduced_z, 1e-10,
                                       &verification) == PREFOS_STATUS_OK);
        CHECK(verification.passed);
        prefos_free_presolver(presolver);
    }
    return 0;
}

static int test_activity_redundant_rows(void)
{
    {
        double A_values[] = {1.0, 1.0, 1.0, -1.0, 1.0, 1.0};
        int A_columns[] = {0, 1, 0, 1, 0, 1};
        int A_rows[] = {0, 2, 4, 6};
        double lower[] = {-INFINITY, -1.0, -INFINITY};
        double upper[] = {3.0, 1.0, 1.5};
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
        double reduced_x[] = {0.5, 0.5};
        double reduced_y[] = {0.0};
        double reduced_z[] = {0.0, 0.0};
        PreFOSPostsolveKKTVerification verification;

        memset(&problem, 0, sizeof(problem));
        problem.n = 2;
        problem.A = (PreFOSCsrMatrix){3, 2, 6, A_values, A_columns, A_rows};
        problem.constraint_lower = lower;
        problem.constraint_upper = upper;
        problem.Q = (PreFOSCsrMatrix){2, 2, 0, NULL, NULL, Q_rows};
        problem.q_storage = PREFOS_Q_UPPER_TRIANGULAR;
        problem.R = (PreFOSCsrMatrix){0, 2, 0, NULL, NULL, R_rows};
        problem.c = c;
        problem.n_box = 2;
        problem.box_indices = box_indices;
        problem.box_lower = box_lower;
        problem.box_upper = box_upper;

        CHECK(prefos_create_presolver(&problem, &settings, &presolver) ==
              PREFOS_STATUS_OK);
        CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_REDUCED);
        reduced = prefos_get_reduced_problem(presolver);
        CHECK(reduced != NULL && reduced->A.rows == 1 && reduced->A.nnz == 2);
        CHECK(prefos_get_stats(presolver)->removed_redundant_rows == 2);
        CHECK(prefos_verify_postsolve_kkt(presolver, reduced_x, reduced_y, reduced_z,
                                       1e-12, &verification) == PREFOS_STATUS_OK);
        CHECK(verification.passed);
        prefos_free_presolver(presolver);
    }
    {
        double A_values[] = {1.0};
        int A_columns[] = {0};
        int A_rows[] = {0, 1};
        double lower[] = {0.0};
        double upper[] = {INFINITY};
        int Q_rows[] = {0, 0, 0};
        int R_rows[] = {0};
        double c[] = {0.0, 0.0};
        int cone_indices[] = {0, 1};
        PreFOSConeBlock cone = {PREFOS_CONE_SECOND_ORDER, 2, 0, cone_indices};
        PreFOSProblemData problem;
        PreFOSSettings settings = prefos_strict_settings();
        PreFOSPresolver *presolver = NULL;
        double reduced_x[] = {1.0, 0.0};
        double reduced_z[] = {0.0, 0.0};
        PreFOSPostsolveKKTVerification verification;

        memset(&problem, 0, sizeof(problem));
        problem.n = 2;
        problem.A = (PreFOSCsrMatrix){1, 2, 1, A_values, A_columns, A_rows};
        problem.constraint_lower = lower;
        problem.constraint_upper = upper;
        problem.Q = (PreFOSCsrMatrix){2, 2, 0, NULL, NULL, Q_rows};
        problem.q_storage = PREFOS_Q_UPPER_TRIANGULAR;
        problem.R = (PreFOSCsrMatrix){0, 2, 0, NULL, NULL, R_rows};
        problem.c = c;
        problem.n_cones = 1;
        problem.cones = &cone;

        CHECK(prefos_create_presolver(&problem, &settings, &presolver) ==
              PREFOS_STATUS_OK);
        CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_REDUCED);
        CHECK(prefos_get_reduced_problem(presolver)->A.rows == 0);
        CHECK(prefos_get_stats(presolver)->removed_redundant_rows == 1);
        CHECK(prefos_verify_postsolve_kkt(presolver, reduced_x, NULL, reduced_z, 1e-12,
                                       &verification) == PREFOS_STATUS_OK);
        CHECK(verification.passed);
        prefos_free_presolver(presolver);
    }
    {
        double A_values[] = {1.0, 1.0};
        int A_columns[] = {0, 1};
        int A_rows[] = {0, 2};
        double lower[] = {-INFINITY};
        double upper[] = {1.0};
        int Q_rows[] = {0, 0, 0};
        int R_rows[] = {0};
        double c[] = {0.0, 0.0};
        int box_indices[] = {0, 1};
        double box_lower[] = {-INFINITY, -INFINITY};
        double box_upper[] = {INFINITY, INFINITY};
        PreFOSProblemData problem;
        PreFOSSettings settings = prefos_strict_settings();
        PreFOSPresolver *presolver = NULL;

        memset(&problem, 0, sizeof(problem));
        problem.n = 2;
        problem.A = (PreFOSCsrMatrix){1, 2, 2, A_values, A_columns, A_rows};
        problem.constraint_lower = lower;
        problem.constraint_upper = upper;
        problem.Q = (PreFOSCsrMatrix){2, 2, 0, NULL, NULL, Q_rows};
        problem.q_storage = PREFOS_Q_UPPER_TRIANGULAR;
        problem.R = (PreFOSCsrMatrix){0, 2, 0, NULL, NULL, R_rows};
        problem.c = c;
        problem.n_box = 2;
        problem.box_indices = box_indices;
        problem.box_lower = box_lower;
        problem.box_upper = box_upper;

        CHECK(prefos_create_presolver(&problem, &settings, &presolver) ==
              PREFOS_STATUS_OK);
        CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_OK);
        CHECK(prefos_get_reduced_problem(presolver)->A.rows == 1);
        CHECK(prefos_get_stats(presolver)->removed_redundant_rows == 0);
        prefos_free_presolver(presolver);

        settings.remove_redundant_rows = 0;
        box_lower[0] = 0.0;
        box_lower[1] = 0.0;
        box_upper[0] = 1.0;
        box_upper[1] = 1.0;
        upper[0] = 3.0;
        CHECK(prefos_create_presolver(&problem, &settings, &presolver) ==
              PREFOS_STATUS_OK);
        CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_OK);
        CHECK(prefos_get_reduced_problem(presolver)->A.rows == 1);
        CHECK(prefos_get_stats(presolver)->removed_redundant_rows == 0);
        prefos_free_presolver(presolver);
    }
    return 0;
}

static int test_soc_zero_collapse_kkt(void)
{
    double A_values[] = {1.0};
    int A_columns[] = {0};
    int A_rows[] = {0, 1};
    double lower[] = {-INFINITY};
    double upper[] = {0.0};
    int Q_rows[] = {0, 0, 0};
    int R_rows[] = {0};
    double c[] = {-1.0, 2.0};
    int cone_indices[] = {0, 1};
    PreFOSConeBlock cone = {PREFOS_CONE_SECOND_ORDER, 2, 0, cone_indices};
    PreFOSProblemData problem;
    PreFOSSettings settings = prefos_strict_settings();
    PreFOSPresolver *presolver = NULL;
    const PreFOSPresolvedProblem *reduced;
    const PreFOSStats *stats;
    double original_x[2], original_y[1], original_z[2];
    PreFOSPostsolveKKTVerification verification;

    memset(&problem, 0, sizeof(problem));
    problem.n = 2;
    problem.A = (PreFOSCsrMatrix){1, 2, 1, A_values, A_columns, A_rows};
    problem.constraint_lower = lower;
    problem.constraint_upper = upper;
    problem.Q = (PreFOSCsrMatrix){2, 2, 0, NULL, NULL, Q_rows};
    problem.q_storage = PREFOS_Q_UPPER_TRIANGULAR;
    problem.R = (PreFOSCsrMatrix){0, 2, 0, NULL, NULL, R_rows};
    problem.c = c;
    problem.n_cones = 1;
    problem.cones = &cone;

    CHECK(prefos_create_presolver(&problem, &settings, &presolver) == PREFOS_STATUS_OK);
    CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_REDUCED);
    reduced = prefos_get_reduced_problem(presolver);
    stats = prefos_get_stats(presolver);
    CHECK(reduced != NULL && stats != NULL);
    CHECK(reduced->n == 0 && reduced->A.rows == 0 && reduced->n_cones == 0);
    CHECK(stats->collapsed_cones == 1 && stats->fixed_cone_variables == 2);
    CHECK(prefos_postsolve_primal_dual(presolver, NULL, NULL, NULL, 1e-10, original_x,
                                    original_y, original_z) == PREFOS_STATUS_OK);
    CHECK(close_to(original_y[0], 3.0));
    CHECK(close_to(original_z[0], -2.0));
    CHECK(close_to(original_z[1], -2.0));
    CHECK(prefos_verify_postsolve_kkt(presolver, NULL, NULL, NULL, 1e-10,
                                   &verification) == PREFOS_STATUS_OK);
    CHECK(verification.passed);

    prefos_free_presolver(presolver);
    return 0;
}

static int test_zero_cone_collapse_compacts_objective(void)
{
    double A_values[] = {1.0};
    int A_columns[] = {1};
    int A_rows[] = {0, 1};
    double lower[] = {-INFINITY};
    double upper[] = {0.0};
    double Q_values[] = {4.0, 1.0, 2.0, 3.0, 5.0, 6.0};
    int Q_columns[] = {0, 1, 2, 1, 2, 2};
    int Q_rows[] = {0, 3, 5, 6};
    double R_values[] = {2.0, 3.0, 4.0};
    int R_columns[] = {0, 1, 2};
    int R_rows[] = {0, 3};
    double D[] = {2.0};
    double c[] = {1.0, -1.0, 2.0};
    int box_indices[] = {0};
    double box_lower[] = {-2.0};
    double box_upper[] = {2.0};
    int cone_indices[] = {1, 2};
    PreFOSConeBlock cone = {PREFOS_CONE_SECOND_ORDER, 2, 0, cone_indices};
    PreFOSProblemData problem;
    PreFOSSettings settings = prefos_strict_settings();
    PreFOSPresolver *presolver = NULL;
    const PreFOSPresolvedProblem *reduced;
    double reduced_x[] = {-1.0 / 12.0};
    double reduced_z[] = {0.0};
    double original_x[3], original_y[1], original_z[3];
    PreFOSPrimalVerification primal_verification;
    PreFOSPostsolveKKTVerification kkt_verification;

    memset(&problem, 0, sizeof(problem));
    problem.n = 3;
    problem.A = (PreFOSCsrMatrix){1, 3, 1, A_values, A_columns, A_rows};
    problem.constraint_lower = lower;
    problem.constraint_upper = upper;
    problem.Q = (PreFOSCsrMatrix){3, 3, 6, Q_values, Q_columns, Q_rows};
    problem.q_storage = PREFOS_Q_UPPER_TRIANGULAR;
    problem.R = (PreFOSCsrMatrix){1, 3, 3, R_values, R_columns, R_rows};
    problem.D = D;
    problem.c = c;
    problem.objective_offset = 7.0;
    problem.n_box = 1;
    problem.box_indices = box_indices;
    problem.box_lower = box_lower;
    problem.box_upper = box_upper;
    problem.n_cones = 1;
    problem.cones = &cone;

    CHECK(prefos_create_presolver(&problem, &settings, &presolver) == PREFOS_STATUS_OK);
    CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_REDUCED);
    reduced = prefos_get_reduced_problem(presolver);
    CHECK(reduced != NULL);
    CHECK(reduced->n == 1 && reduced->A.rows == 0 && reduced->n_cones == 0);
    CHECK(reduced->n_box == 1 && reduced->box_indices[0] == 0);
    CHECK(reduced->Q.rows == 1 && reduced->Q.nnz == 1);
    CHECK(reduced->Q.column_indices[0] == 0 && close_to(reduced->Q.values[0], 4.0));
    CHECK(reduced->R.rows == 1 && reduced->R.nnz == 1);
    CHECK(reduced->R.column_indices[0] == 0 && close_to(reduced->R.values[0], 2.0));
    CHECK(close_to(reduced->D[0], 2.0));
    CHECK(close_to(reduced->c[0], 1.0));
    CHECK(close_to(reduced->objective_offset, 7.0));

    CHECK(prefos_verify_postsolve_primal(presolver, reduced_x, 1e-12,
                                      &primal_verification) == PREFOS_STATUS_OK);
    CHECK(primal_verification.passed);
    CHECK(prefos_postsolve_primal_dual(presolver, reduced_x, NULL, reduced_z, 1e-12,
                                    original_x, original_y,
                                    original_z) == PREFOS_STATUS_OK);
    CHECK(close_to(original_x[0], -1.0 / 12.0));
    CHECK(close_to(original_x[1], 0.0) && close_to(original_x[2], 0.0));
    CHECK(close_to(original_y[0], 31.0 / 12.0));
    CHECK(close_to(original_z[0], 0.0));
    CHECK(close_to(original_z[1], -0.5));
    CHECK(close_to(original_z[2], -0.5));
    CHECK(prefos_verify_postsolve_kkt(presolver, reduced_x, NULL, reduced_z, 1e-10,
                                   &kkt_verification) == PREFOS_STATUS_OK);
    CHECK(kkt_verification.passed);

    prefos_free_presolver(presolver);
    return 0;
}

static int test_rsoc_zero_collapse_kkt(void)
{
    double A_values[] = {1.0, 1.0};
    int A_columns[] = {0, 1};
    int A_rows[] = {0, 1, 2};
    double lower[] = {-INFINITY, -INFINITY};
    double upper[] = {0.0, 0.0};
    int Q_rows[] = {0, 0, 0, 0};
    int R_rows[] = {0};
    double c[] = {-1.0, -1.0, 2.0};
    int cone_indices[] = {0, 1, 2};
    PreFOSConeBlock cone = {PREFOS_CONE_ROTATED_SECOND_ORDER, 3, 0, cone_indices};
    PreFOSProblemData problem;
    PreFOSPresolver *presolver = NULL;
    PreFOSPostsolveKKTVerification verification;

    memset(&problem, 0, sizeof(problem));
    problem.n = 3;
    problem.A = (PreFOSCsrMatrix){2, 3, 2, A_values, A_columns, A_rows};
    problem.constraint_lower = lower;
    problem.constraint_upper = upper;
    problem.Q = (PreFOSCsrMatrix){3, 3, 0, NULL, NULL, Q_rows};
    problem.q_storage = PREFOS_Q_UPPER_TRIANGULAR;
    problem.R = (PreFOSCsrMatrix){0, 3, 0, NULL, NULL, R_rows};
    problem.c = c;
    problem.n_cones = 1;
    problem.cones = &cone;

    CHECK(prefos_create_presolver(&problem, NULL, &presolver) == PREFOS_STATUS_OK);
    CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_REDUCED);
    CHECK(prefos_get_reduced_problem(presolver)->n == 0);
    CHECK(prefos_verify_postsolve_kkt(presolver, NULL, NULL, NULL, 1e-9,
                                   &verification) == PREFOS_STATUS_OK);
    CHECK(verification.passed);

    prefos_free_presolver(presolver);
    return 0;
}

static int test_rsoc_facial_reduction_extended_dual(void)
{
    {
        double A_values[] = {1.0};
        int A_columns[] = {0};
        int A_rows[] = {0, 1};
        double lower[] = {-INFINITY};
        double upper[] = {0.0};
        double Q_values[] = {1.0};
        int Q_columns[] = {1};
        int Q_rows[] = {0, 0, 1, 1};
        int R_rows[] = {0};
        double c[] = {0.0, -1.0, 1.0};
        int cone_indices[] = {0, 1, 2};
        PreFOSConeBlock cone = {PREFOS_CONE_ROTATED_SECOND_ORDER, 3, 0, cone_indices};
        PreFOSProblemData problem;
        PreFOSSettings settings = prefos_strict_settings();
        PreFOSPresolver *presolver = NULL;
        const PreFOSPresolvedProblem *reduced;
        const PreFOSFacialReductionCertificate *certificates;
        size_t certificate_count = 0;
        double reduced_x[] = {1.0};
        double reduced_z[] = {0.0};
        double original_x[3], original_y[1], original_z[3];
        PreFOSPrimalVerification primal_verification;
        PreFOSPostsolveKKTVerification kkt_verification;

        memset(&problem, 0, sizeof(problem));
        problem.n = 3;
        problem.A = (PreFOSCsrMatrix){1, 3, 1, A_values, A_columns, A_rows};
        problem.constraint_lower = lower;
        problem.constraint_upper = upper;
        problem.Q = (PreFOSCsrMatrix){3, 3, 1, Q_values, Q_columns, Q_rows};
        problem.q_storage = PREFOS_Q_UPPER_TRIANGULAR;
        problem.R = (PreFOSCsrMatrix){0, 3, 0, NULL, NULL, R_rows};
        problem.c = c;
        problem.n_cones = 1;
        problem.cones = &cone;

        CHECK(prefos_create_presolver(&problem, &settings, &presolver) ==
              PREFOS_STATUS_OK);
        CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_REDUCED);
        reduced = prefos_get_reduced_problem(presolver);
        CHECK(reduced != NULL && reduced->n == 1 && reduced->A.rows == 0);
        CHECK(reduced->n_box == 1 && reduced->n_cones == 0);
        CHECK(reduced->box_indices[0] == 0);
        CHECK(close_to(reduced->box_lower[0], 0.0));
        CHECK(reduced->box_upper[0] == INFINITY);
        CHECK(close_to(reduced->Q.values[0], 1.0));
        CHECK(close_to(reduced->c[0], -1.0));
        CHECK(prefos_get_stats(presolver)->reduced_rsoc_faces == 1);
        CHECK(prefos_get_stats(presolver)->fixed_cone_variables == 2);

        certificates = prefos_get_facial_reductions(presolver, &certificate_count);
        CHECK(certificates != NULL && certificate_count == 1);
        CHECK(certificates[0].type == PREFOS_FACE_RSOC_U_ZERO);
        CHECK(certificates[0].source_row == 0);
        CHECK(certificates[0].zero_axis_column == 0);
        CHECK(certificates[0].surviving_axis_column == 1);
        CHECK(certificates[0].cone_dimension == 3);
        CHECK(certificates[0].cone_indices[2] == 2);

        CHECK(prefos_verify_postsolve_primal(presolver, reduced_x, 1e-12,
                                          &primal_verification) == PREFOS_STATUS_OK);
        CHECK(primal_verification.passed);
        CHECK(prefos_postsolve_primal_dual(presolver, reduced_x, NULL, reduced_z, 1e-12,
                                        original_x, original_y, original_z) ==
              PREFOS_STATUS_DUAL_RECOVERY_UNAVAILABLE);
        CHECK(prefos_verify_postsolve_kkt(presolver, reduced_x, NULL, reduced_z, 1e-12,
                                       &kkt_verification) ==
              PREFOS_STATUS_DUAL_RECOVERY_UNAVAILABLE);
        CHECK(prefos_postsolve_extended_dual(presolver, reduced_x, NULL, reduced_z,
                                          1e-12, original_x, original_y,
                                          original_z) == PREFOS_STATUS_OK);
        CHECK(close_to(original_x[0], 0.0));
        CHECK(close_to(original_x[1], 1.0));
        CHECK(close_to(original_x[2], 0.0));
        CHECK(close_to(original_y[0], 0.0));
        CHECK(close_to(original_z[0], 0.0));
        CHECK(close_to(original_z[1], 0.0));
        CHECK(close_to(original_z[2], -1.0));
        CHECK(prefos_verify_postsolve_extended_kkt(presolver, reduced_x, NULL,
                                                reduced_z, 1e-12,
                                                &kkt_verification) == PREFOS_STATUS_OK);
        CHECK(kkt_verification.passed);
        prefos_free_presolver(presolver);

        settings.rsoc_face_reduction = 0;
        CHECK(prefos_create_presolver(&problem, &settings, &presolver) ==
              PREFOS_STATUS_OK);
        CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_OK);
        CHECK(prefos_get_reduced_problem(presolver)->n == 3);
        CHECK(prefos_get_reduced_problem(presolver)->n_cones == 1);
        CHECK(prefos_get_stats(presolver)->reduced_rsoc_faces == 0);
        prefos_free_presolver(presolver);

        settings = prefos_default_settings();
        upper[0] = 5e-8;
        CHECK(prefos_create_presolver(&problem, &settings, &presolver) ==
              PREFOS_STATUS_OK);
        CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_OK);
        CHECK(prefos_get_reduced_problem(presolver)->n_cones == 1);
        CHECK(prefos_get_stats(presolver)->reduced_rsoc_faces == 0);
        prefos_free_presolver(presolver);
    }
    {
        double A_values[] = {1.0};
        int A_columns[] = {1};
        int A_rows[] = {0, 1};
        double lower[] = {-INFINITY};
        double upper[] = {0.0};
        double Q_values[] = {1.0};
        int Q_columns[] = {0};
        int Q_rows[] = {0, 1, 1, 1};
        int R_rows[] = {0};
        double c[] = {-2.0, 0.0, 1.0};
        int cone_indices[] = {0, 1, 2};
        PreFOSConeBlock cone = {PREFOS_CONE_ROTATED_SECOND_ORDER, 3, 0, cone_indices};
        PreFOSProblemData problem;
        PreFOSSettings settings = prefos_strict_settings();
        PreFOSPresolver *presolver = NULL;
        const PreFOSFacialReductionCertificate *certificates;
        size_t certificate_count = 0;
        double reduced_x[] = {2.0};
        double reduced_z[] = {0.0};
        PreFOSPostsolveKKTVerification verification;

        memset(&problem, 0, sizeof(problem));
        problem.n = 3;
        problem.A = (PreFOSCsrMatrix){1, 3, 1, A_values, A_columns, A_rows};
        problem.constraint_lower = lower;
        problem.constraint_upper = upper;
        problem.Q = (PreFOSCsrMatrix){3, 3, 1, Q_values, Q_columns, Q_rows};
        problem.q_storage = PREFOS_Q_UPPER_TRIANGULAR;
        problem.R = (PreFOSCsrMatrix){0, 3, 0, NULL, NULL, R_rows};
        problem.c = c;
        problem.n_cones = 1;
        problem.cones = &cone;

        CHECK(prefos_create_presolver(&problem, &settings, &presolver) ==
              PREFOS_STATUS_OK);
        CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_REDUCED);
        certificates = prefos_get_facial_reductions(presolver, &certificate_count);
        CHECK(certificates != NULL && certificate_count == 1);
        CHECK(certificates[0].type == PREFOS_FACE_RSOC_V_ZERO);
        CHECK(certificates[0].zero_axis_column == 1);
        CHECK(certificates[0].surviving_axis_column == 0);
        CHECK(prefos_verify_postsolve_extended_kkt(presolver, reduced_x, NULL,
                                                reduced_z, 1e-12,
                                                &verification) == PREFOS_STATUS_OK);
        CHECK(verification.passed);
        prefos_free_presolver(presolver);
    }
    return 0;
}

static int test_psd_facial_reduction_extended_dual(void)
{
    {
        double A_values[] = {1.0};
        int A_columns[] = {2};
        int A_rows[] = {0, 1};
        double lower[] = {-INFINITY};
        double upper[] = {0.0};
        double Q_values[] = {1.0, 1.0, 1.0};
        int Q_columns[] = {0, 3, 5};
        int Q_rows[] = {0, 1, 1, 1, 2, 2, 3};
        int R_rows[] = {0};
        double c[] = {-1.0, 1.0, 3.0, 0.0, -2.0, -1.0};
        int cone_indices[] = {0, 1, 2, 3, 4, 5};
        PreFOSConeBlock cone = {PREFOS_CONE_POSITIVE_SEMIDEFINITE, 6, 3, cone_indices};
        PreFOSProblemData problem;
        PreFOSSettings settings = prefos_strict_settings();
        PreFOSPresolver *presolver = NULL;
        const PreFOSPresolvedProblem *reduced;
        const PreFOSFacialReductionCertificate *certificates;
        size_t certificate_count = 0;
        double reduced_x[] = {1.0, 0.0, 1.0};
        double reduced_z[] = {0.0, 0.0, 0.0};
        double original_x[6], original_y[1], original_z[6];
        PreFOSPrimalVerification primal_verification;
        PreFOSPostsolveKKTVerification kkt_verification;

        memset(&problem, 0, sizeof(problem));
        problem.n = 6;
        problem.A = (PreFOSCsrMatrix){1, 6, 1, A_values, A_columns, A_rows};
        problem.constraint_lower = lower;
        problem.constraint_upper = upper;
        problem.Q = (PreFOSCsrMatrix){6, 6, 3, Q_values, Q_columns, Q_rows};
        problem.q_storage = PREFOS_Q_UPPER_TRIANGULAR;
        problem.R = (PreFOSCsrMatrix){0, 6, 0, NULL, NULL, R_rows};
        problem.c = c;
        problem.n_cones = 1;
        problem.cones = &cone;

        CHECK(prefos_create_presolver(&problem, &settings, &presolver) ==
              PREFOS_STATUS_OK);
        CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_REDUCED);
        reduced = prefos_get_reduced_problem(presolver);
        CHECK(reduced != NULL && reduced->n == 3 && reduced->A.rows == 0);
        CHECK(reduced->n_box == 0 && reduced->n_cones == 1);
        CHECK(reduced->cones[0].type == PREFOS_CONE_POSITIVE_SEMIDEFINITE);
        CHECK(reduced->cones[0].matrix_order == 2);
        CHECK(reduced->cones[0].dimension == 3);
        CHECK(reduced->cones[0].indices[0] == 0);
        CHECK(reduced->cones[0].indices[1] == 1);
        CHECK(reduced->cones[0].indices[2] == 2);
        CHECK(prefos_get_stats(presolver)->reduced_psd_faces == 1);
        CHECK(prefos_get_stats(presolver)->fixed_cone_variables == 3);

        certificates = prefos_get_facial_reductions(presolver, &certificate_count);
        CHECK(certificates != NULL && certificate_count == 1);
        CHECK(certificates[0].type == PREFOS_FACE_PSD_ZERO_DIAGONALS);
        CHECK(certificates[0].source_row == -1);
        CHECK(certificates[0].zero_axis_column == -1);
        CHECK(certificates[0].surviving_axis_column == -1);
        CHECK(certificates[0].matrix_order == 3);
        CHECK(certificates[0].n_removed_matrix_indices == 1);
        CHECK(certificates[0].removed_matrix_indices[0] == 1);
        CHECK(certificates[0].source_rows[0] == 0);
        CHECK(certificates[0].cone_indices[4] == 4);

        CHECK(prefos_verify_postsolve_primal(presolver, reduced_x, 1e-12,
                                          &primal_verification) == PREFOS_STATUS_OK);
        CHECK(primal_verification.passed);
        CHECK(prefos_postsolve_primal_dual(presolver, reduced_x, NULL, reduced_z, 1e-12,
                                        original_x, original_y, original_z) ==
              PREFOS_STATUS_DUAL_RECOVERY_UNAVAILABLE);
        CHECK(prefos_verify_postsolve_kkt(presolver, reduced_x, NULL, reduced_z, 1e-12,
                                       &kkt_verification) ==
              PREFOS_STATUS_DUAL_RECOVERY_UNAVAILABLE);
        CHECK(prefos_postsolve_extended_dual(presolver, reduced_x, NULL, reduced_z,
                                          1e-12, original_x, original_y,
                                          original_z) == PREFOS_STATUS_OK);
        CHECK(close_to(original_x[0], 1.0));
        CHECK(close_to(original_x[1], 0.0));
        CHECK(close_to(original_x[2], 0.0));
        CHECK(close_to(original_x[3], 0.0));
        CHECK(close_to(original_x[4], 0.0));
        CHECK(close_to(original_x[5], 1.0));
        CHECK(close_to(original_z[0], 0.0));
        CHECK(close_to(original_z[1], -1.0));
        CHECK(close_to(original_z[2], -3.0));
        CHECK(close_to(original_z[3], 0.0));
        CHECK(close_to(original_z[4], 2.0));
        CHECK(close_to(original_z[5], 0.0));
        CHECK(prefos_verify_postsolve_extended_kkt(presolver, reduced_x, NULL,
                                                reduced_z, 1e-12,
                                                &kkt_verification) == PREFOS_STATUS_OK);
        CHECK(kkt_verification.passed);
        CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_REDUCED);
        certificates = prefos_get_facial_reductions(presolver, &certificate_count);
        CHECK(certificates != NULL && certificate_count == 1);
        CHECK(certificates[0].removed_matrix_indices[0] == 1);
        CHECK(prefos_get_stats(presolver)->reduced_psd_faces == 1);
        prefos_free_presolver(presolver);

        settings.psd_face_reduction = 0;
        CHECK(prefos_create_presolver(&problem, &settings, &presolver) ==
              PREFOS_STATUS_OK);
        CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_OK);
        CHECK(prefos_get_reduced_problem(presolver)->n == 6);
        CHECK(prefos_get_reduced_problem(presolver)->cones[0].matrix_order == 3);
        CHECK(prefos_get_stats(presolver)->reduced_psd_faces == 0);
        prefos_free_presolver(presolver);

        settings = prefos_default_settings();
        upper[0] = 5e-8;
        CHECK(prefos_create_presolver(&problem, &settings, &presolver) ==
              PREFOS_STATUS_OK);
        CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_OK);
        CHECK(prefos_get_reduced_problem(presolver)->n == 6);
        CHECK(prefos_get_reduced_problem(presolver)->cones[0].matrix_order == 3);
        CHECK(prefos_get_stats(presolver)->reduced_psd_faces == 0);
        prefos_free_presolver(presolver);
    }
    {
        double A_values[] = {1.0, 1.0};
        int A_columns[] = {0, 5};
        int A_rows[] = {0, 1, 2};
        double lower[] = {-INFINITY, -INFINITY};
        double upper[] = {0.0, 0.0};
        int Q_rows[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        int R_rows[] = {0};
        double c[] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        int cone_indices[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        PreFOSConeBlock cone = {PREFOS_CONE_POSITIVE_SEMIDEFINITE, 10, 4, cone_indices};
        PreFOSProblemData problem;
        PreFOSSettings settings = prefos_strict_settings();
        PreFOSPresolver *presolver = NULL;
        const PreFOSPresolvedProblem *reduced;
        const PreFOSFacialReductionCertificate *certificates;
        size_t certificate_count = 0;
        double reduced_x[] = {1.0, 0.0, 1.0};
        double reduced_z[] = {0.0, 0.0, 0.0};
        PreFOSPostsolveKKTVerification verification;

        memset(&problem, 0, sizeof(problem));
        problem.n = 10;
        problem.A = (PreFOSCsrMatrix){2, 10, 2, A_values, A_columns, A_rows};
        problem.constraint_lower = lower;
        problem.constraint_upper = upper;
        problem.Q = (PreFOSCsrMatrix){10, 10, 0, NULL, NULL, Q_rows};
        problem.q_storage = PREFOS_Q_UPPER_TRIANGULAR;
        problem.R = (PreFOSCsrMatrix){0, 10, 0, NULL, NULL, R_rows};
        problem.c = c;
        problem.n_cones = 1;
        problem.cones = &cone;

        CHECK(prefos_create_presolver(&problem, &settings, &presolver) ==
              PREFOS_STATUS_OK);
        CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_REDUCED);
        reduced = prefos_get_reduced_problem(presolver);
        CHECK(reduced != NULL && reduced->n == 3 && reduced->n_cones == 1);
        CHECK(reduced->cones[0].matrix_order == 2);
        CHECK(reduced->cones[0].indices[0] == 0);
        CHECK(reduced->cones[0].indices[1] == 1);
        CHECK(reduced->cones[0].indices[2] == 2);
        CHECK(prefos_get_stats(presolver)->fixed_cone_variables == 7);
        certificates = prefos_get_facial_reductions(presolver, &certificate_count);
        CHECK(certificates != NULL && certificate_count == 1);
        CHECK(certificates[0].n_removed_matrix_indices == 2);
        CHECK(certificates[0].removed_matrix_indices[0] == 0);
        CHECK(certificates[0].removed_matrix_indices[1] == 2);
        CHECK(certificates[0].source_rows[0] == 0);
        CHECK(certificates[0].source_rows[1] == 1);
        CHECK(prefos_verify_postsolve_extended_kkt(presolver, reduced_x, NULL,
                                                reduced_z, 1e-12,
                                                &verification) == PREFOS_STATUS_OK);
        CHECK(verification.passed);
        prefos_free_presolver(presolver);
    }
    return 0;
}

static int test_psd_zero_collapse_kkt(void)
{
    double A_values[] = {1.0, 1.0};
    int A_columns[] = {0, 2};
    int A_rows[] = {0, 1, 2};
    double lower[] = {-INFINITY, -INFINITY};
    double upper[] = {0.0, 0.0};
    int Q_rows[] = {0, 0, 0, 0};
    int R_rows[] = {0};
    double c[] = {-1.0, 2.0, -1.0};
    int cone_indices[] = {0, 1, 2};
    PreFOSConeBlock cone = {PREFOS_CONE_POSITIVE_SEMIDEFINITE, 3, 2, cone_indices};
    PreFOSProblemData problem;
    PreFOSPresolver *presolver = NULL;
    PreFOSPostsolveKKTVerification verification;

    memset(&problem, 0, sizeof(problem));
    problem.n = 3;
    problem.A = (PreFOSCsrMatrix){2, 3, 2, A_values, A_columns, A_rows};
    problem.constraint_lower = lower;
    problem.constraint_upper = upper;
    problem.Q = (PreFOSCsrMatrix){3, 3, 0, NULL, NULL, Q_rows};
    problem.q_storage = PREFOS_Q_UPPER_TRIANGULAR;
    problem.R = (PreFOSCsrMatrix){0, 3, 0, NULL, NULL, R_rows};
    problem.c = c;
    problem.n_cones = 1;
    problem.cones = &cone;

    CHECK(prefos_create_presolver(&problem, NULL, &presolver) == PREFOS_STATUS_OK);
    CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_REDUCED);
    CHECK(prefos_get_reduced_problem(presolver)->n == 0);
    CHECK(prefos_verify_postsolve_kkt(presolver, NULL, NULL, NULL, 1e-8,
                                   &verification) == PREFOS_STATUS_OK);
    CHECK(verification.passed);

    prefos_free_presolver(presolver);
    return 0;
}

static int test_nonfinite_postsolve_inputs_are_rejected(void)
{
    int A_rows[] = {0};
    double Q_values[] = {1.0};
    int Q_columns[] = {0};
    int Q_rows[] = {0, 1};
    int R_rows[] = {0};
    double c[] = {0.0};
    int box_indices[] = {0};
    double box_lower[] = {-1.0};
    double box_upper[] = {1.0};
    PreFOSProblemData problem;
    PreFOSPresolver *presolver = NULL;
    double reduced_x[] = {0.0};
    double reduced_z[] = {NAN};
    double original_x[1];
    PreFOSPostsolveKKTVerification verification;

    memset(&problem, 0, sizeof(problem));
    problem.n = 1;
    problem.A = (PreFOSCsrMatrix){0, 1, 0, NULL, NULL, A_rows};
    problem.Q = (PreFOSCsrMatrix){1, 1, 1, Q_values, Q_columns, Q_rows};
    problem.q_storage = PREFOS_Q_UPPER_TRIANGULAR;
    problem.R = (PreFOSCsrMatrix){0, 1, 0, NULL, NULL, R_rows};
    problem.c = c;
    problem.n_box = 1;
    problem.box_indices = box_indices;
    problem.box_lower = box_lower;
    problem.box_upper = box_upper;

    CHECK(prefos_create_presolver(&problem, NULL, &presolver) == PREFOS_STATUS_OK);
    CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_OK);
    CHECK(prefos_verify_postsolve_kkt(presolver, reduced_x, NULL, reduced_z, 1e-8,
                                   &verification) == PREFOS_STATUS_INVALID_ARGUMENT);
    reduced_x[0] = NAN;
    CHECK(prefos_postsolve_primal(presolver, reduced_x, original_x) ==
          PREFOS_STATUS_INVALID_ARGUMENT);
    prefos_free_presolver(presolver);
    return 0;
}

static int test_full_q_symmetry_validation(void)
{
    int A_rows[] = {0};
    double asymmetric_values[] = {1.0};
    int asymmetric_columns[] = {1};
    int asymmetric_rows[] = {0, 1, 1};
    double symmetric_values[] = {1.0, 1.0};
    int symmetric_columns[] = {1, 0};
    int symmetric_rows[] = {0, 1, 2};
    int R_rows[] = {0};
    double c[] = {0.0, 0.0};
    int box_indices[] = {0, 1};
    double box_lower[] = {1.0, -2.0};
    double box_upper[] = {1.0, 2.0};
    PreFOSProblemData problem;
    PreFOSSettings settings = prefos_strict_settings();
    PreFOSPresolver *presolver = NULL;
    const PreFOSPresolvedProblem *reduced;
    double reduced_x[] = {0.5};
    PreFOSPrimalVerification verification;

    memset(&problem, 0, sizeof(problem));
    problem.n = 2;
    problem.A = (PreFOSCsrMatrix){0, 2, 0, NULL, NULL, A_rows};
    problem.Q = (PreFOSCsrMatrix){
        2, 2, 1, asymmetric_values, asymmetric_columns, asymmetric_rows};
    problem.q_storage = PREFOS_Q_FULL;
    problem.R = (PreFOSCsrMatrix){0, 2, 0, NULL, NULL, R_rows};
    problem.c = c;
    problem.n_box = 2;
    problem.box_indices = box_indices;
    problem.box_lower = box_lower;
    problem.box_upper = box_upper;

    CHECK(prefos_create_presolver(&problem, &settings, &presolver) ==
          PREFOS_STATUS_INVALID_ARGUMENT);
    CHECK(presolver == NULL);

    problem.Q =
        (PreFOSCsrMatrix){2, 2, 2, symmetric_values, symmetric_columns, symmetric_rows};
    CHECK(prefos_create_presolver(&problem, &settings, &presolver) == PREFOS_STATUS_OK);
    CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_REDUCED);
    reduced = prefos_get_reduced_problem(presolver);
    CHECK(reduced != NULL && reduced->n == 1);
    CHECK(close_to(reduced->c[0], 1.0));
    CHECK(prefos_verify_postsolve_primal(presolver, reduced_x, 1e-12, &verification) ==
          PREFOS_STATUS_OK);
    CHECK(verification.passed);
    prefos_free_presolver(presolver);
    return 0;
}

static int test_empty_model_round_trip(void)
{
    int row_pointers[] = {0};
    PreFOSProblemData problem;
    PreFOSSettings settings = prefos_strict_settings();
    PreFOSPresolver *presolver = NULL;
    PreFOSPrimalVerification primal_verification;
    PreFOSPostsolveKKTVerification kkt_verification;

    memset(&problem, 0, sizeof(problem));
    problem.A = (PreFOSCsrMatrix){0, 0, 0, NULL, NULL, row_pointers};
    problem.Q = (PreFOSCsrMatrix){0, 0, 0, NULL, NULL, row_pointers};
    problem.q_storage = PREFOS_Q_UPPER_TRIANGULAR;
    problem.R = (PreFOSCsrMatrix){0, 0, 0, NULL, NULL, row_pointers};

    CHECK(prefos_create_presolver(&problem, &settings, &presolver) == PREFOS_STATUS_OK);
    CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_OK);
    CHECK(prefos_postsolve_primal(presolver, NULL, NULL) == PREFOS_STATUS_OK);
    CHECK(prefos_verify_postsolve_primal(presolver, NULL, 1e-12,
                                      &primal_verification) == PREFOS_STATUS_OK);
    CHECK(primal_verification.passed);
    CHECK(prefos_verify_postsolve_kkt(presolver, NULL, NULL, NULL, 1e-12,
                                   &kkt_verification) == PREFOS_STATUS_OK);
    CHECK(kkt_verification.passed);
    prefos_free_presolver(presolver);
    return 0;
}

static int test_parallel_row_reduction_and_postsolve(void)
{
    {
        double A_values[] = {1.0, 1.0, 2.0, 2.0};
        int A_columns[] = {0, 1, 0, 1};
        int A_rows[] = {0, 2, 4};
        double lower[] = {-INFINITY, -INFINITY};
        double upper[] = {4.0, 6.0};
        int Q_rows[] = {0, 0, 0};
        int R_rows[] = {0};
        double c[] = {-1.0, -1.0};
        int box_indices[] = {0, 1};
        double box_lower[] = {-INFINITY, -INFINITY};
        double box_upper[] = {INFINITY, INFINITY};
        double reduced_x[] = {1.0, 2.0};
        double reduced_y[] = {1.0};
        double reduced_z[] = {0.0, 0.0};
        double original_x[2], original_y[2], original_z[2];
        PreFOSProblemData problem;
        PreFOSSettings settings = prefos_strict_settings();
        PreFOSPresolver *presolver = NULL;
        const PreFOSPresolvedProblem *reduced;
        PreFOSPostsolveKKTVerification verification;

        memset(&problem, 0, sizeof(problem));
        problem.n = 2;
        problem.A = (PreFOSCsrMatrix){2, 2, 4, A_values, A_columns, A_rows};
        problem.constraint_lower = lower;
        problem.constraint_upper = upper;
        problem.Q = (PreFOSCsrMatrix){2, 2, 0, NULL, NULL, Q_rows};
        problem.q_storage = PREFOS_Q_UPPER_TRIANGULAR;
        problem.R = (PreFOSCsrMatrix){0, 2, 0, NULL, NULL, R_rows};
        problem.c = c;
        problem.n_box = 2;
        problem.box_indices = box_indices;
        problem.box_lower = box_lower;
        problem.box_upper = box_upper;

        CHECK(prefos_create_presolver(&problem, &settings, &presolver) ==
              PREFOS_STATUS_OK);
        CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_REDUCED);
        reduced = prefos_get_reduced_problem(presolver);
        CHECK(reduced->A.rows == 1 && reduced->A.nnz == 2);
        CHECK(close_to(reduced->A.values[0], 1.0));
        CHECK(close_to(reduced->constraint_upper[0], 3.0));
        CHECK(prefos_postsolve_primal_dual(presolver, reduced_x, reduced_y, reduced_z,
                                        1e-12, original_x, original_y,
                                        original_z) == PREFOS_STATUS_OK);
        CHECK(close_to(original_y[0], 0.0));
        CHECK(close_to(original_y[1], 0.5));
        CHECK(prefos_verify_postsolve_kkt(presolver, reduced_x, reduced_y, reduced_z,
                                       1e-12, &verification) == PREFOS_STATUS_OK);
        CHECK(verification.passed);
        prefos_free_presolver(presolver);
    }
    {
        double A_values[] = {1.0, 1.0, -2.0, -2.0};
        int A_columns[] = {0, 1, 0, 1};
        int A_rows[] = {0, 2, 4};
        double lower[] = {-INFINITY, -8.0};
        double upper[] = {5.0, INFINITY};
        int Q_rows[] = {0, 0, 0};
        int R_rows[] = {0};
        double c[] = {-1.0, -1.0};
        int box_indices[] = {0, 1};
        double box_lower[] = {-INFINITY, -INFINITY};
        double box_upper[] = {INFINITY, INFINITY};
        double reduced_x[] = {2.0, 2.0};
        double reduced_y[] = {1.0};
        double reduced_z[] = {0.0, 0.0};
        double original_x[2], original_y[2], original_z[2];
        PreFOSProblemData problem;
        PreFOSSettings settings = prefos_strict_settings();
        PreFOSPresolver *presolver = NULL;
        PreFOSPostsolveKKTVerification verification;

        memset(&problem, 0, sizeof(problem));
        problem.n = 2;
        problem.A = (PreFOSCsrMatrix){2, 2, 4, A_values, A_columns, A_rows};
        problem.constraint_lower = lower;
        problem.constraint_upper = upper;
        problem.Q = (PreFOSCsrMatrix){2, 2, 0, NULL, NULL, Q_rows};
        problem.q_storage = PREFOS_Q_UPPER_TRIANGULAR;
        problem.R = (PreFOSCsrMatrix){0, 2, 0, NULL, NULL, R_rows};
        problem.c = c;
        problem.n_box = 2;
        problem.box_indices = box_indices;
        problem.box_lower = box_lower;
        problem.box_upper = box_upper;

        CHECK(prefos_create_presolver(&problem, &settings, &presolver) ==
              PREFOS_STATUS_OK);
        CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_REDUCED);
        CHECK(
            close_to(prefos_get_reduced_problem(presolver)->constraint_upper[0], 4.0));
        CHECK(prefos_postsolve_primal_dual(presolver, reduced_x, reduced_y, reduced_z,
                                        1e-12, original_x, original_y,
                                        original_z) == PREFOS_STATUS_OK);
        CHECK(close_to(original_y[0], 0.0));
        CHECK(close_to(original_y[1], -0.5));
        CHECK(prefos_verify_postsolve_kkt(presolver, reduced_x, reduced_y, reduced_z,
                                       1e-12, &verification) == PREFOS_STATUS_OK);
        CHECK(verification.passed);
        prefos_free_presolver(presolver);
    }
    return 0;
}

static int test_infeasible_parallel_rows(void)
{
    double A_values[] = {1.0, 1.0, -1.0, -1.0};
    int A_columns[] = {0, 1, 0, 1};
    int A_rows[] = {0, 2, 4};
    double lower[] = {-INFINITY, -INFINITY};
    double upper[] = {2.0, -3.0};
    int Q_rows[] = {0, 0, 0};
    int R_rows[] = {0};
    double c[] = {0.0, 0.0};
    int box_indices[] = {0, 1};
    double box_lower[] = {-INFINITY, -INFINITY};
    double box_upper[] = {INFINITY, INFINITY};
    PreFOSProblemData problem;
    PreFOSSettings settings = prefos_strict_settings();
    PreFOSPresolver *presolver = NULL;

    memset(&problem, 0, sizeof(problem));
    problem.n = 2;
    problem.A = (PreFOSCsrMatrix){2, 2, 4, A_values, A_columns, A_rows};
    problem.constraint_lower = lower;
    problem.constraint_upper = upper;
    problem.Q = (PreFOSCsrMatrix){2, 2, 0, NULL, NULL, Q_rows};
    problem.q_storage = PREFOS_Q_UPPER_TRIANGULAR;
    problem.R = (PreFOSCsrMatrix){0, 2, 0, NULL, NULL, R_rows};
    problem.c = c;
    problem.n_box = 2;
    problem.box_indices = box_indices;
    problem.box_lower = box_lower;
    problem.box_upper = box_upper;

    CHECK(prefos_create_presolver(&problem, &settings, &presolver) == PREFOS_STATUS_OK);
    CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_PRIMAL_INFEASIBLE);
    prefos_free_presolver(presolver);

    return 0;
}

static int test_interleaved_row_and_bound_postsolve(void)
{
    double A_values[] = {1.0, 1.0, 2.0, 2.0};
    int A_columns[] = {0, 1, 0, 1};
    int A_rows[] = {0, 2, 4};
    double lower[] = {-INFINITY, -INFINITY};
    double upper[] = {4.0, 6.0};
    int Q_rows[] = {0, 0, 0};
    int R_rows[] = {0};
    double c[] = {-1.0, 0.0};
    int box_indices[] = {0, 1};
    double box_lower[] = {0.0, 0.0};
    double box_upper[] = {10.0, 0.0};
    double reduced_x[] = {3.0};
    double reduced_z[] = {1.0};
    double original_x[2], original_y[2], original_z[2];
    PreFOSProblemData problem;
    PreFOSSettings settings = prefos_strict_settings();
    PreFOSPresolver *presolver = NULL;
    const PreFOSPresolvedProblem *reduced;
    PreFOSPostsolveKKTVerification verification;

    memset(&problem, 0, sizeof(problem));
    problem.n = 2;
    problem.A = (PreFOSCsrMatrix){2, 2, 4, A_values, A_columns, A_rows};
    problem.constraint_lower = lower;
    problem.constraint_upper = upper;
    problem.Q = (PreFOSCsrMatrix){2, 2, 0, NULL, NULL, Q_rows};
    problem.q_storage = PREFOS_Q_UPPER_TRIANGULAR;
    problem.R = (PreFOSCsrMatrix){0, 2, 0, NULL, NULL, R_rows};
    problem.c = c;
    problem.n_box = 2;
    problem.box_indices = box_indices;
    problem.box_lower = box_lower;
    problem.box_upper = box_upper;

    CHECK(prefos_create_presolver(&problem, &settings, &presolver) == PREFOS_STATUS_OK);
    CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_REDUCED);
    reduced = prefos_get_reduced_problem(presolver);
    CHECK(reduced->n == 1 && reduced->A.rows == 0);
    CHECK(prefos_postsolve_primal_dual(presolver, reduced_x, NULL, reduced_z, 1e-12,
                                    original_x, original_y,
                                    original_z) == PREFOS_STATUS_OK);
    CHECK(close_to(original_y[0], 0.0));
    CHECK(close_to(original_y[1], 0.5));
    CHECK(close_to(original_z[0], 0.0));
    CHECK(close_to(original_z[1], -1.0));
    CHECK(prefos_verify_postsolve_kkt(presolver, reduced_x, NULL, reduced_z, 1e-12,
                                   &verification) == PREFOS_STATUS_OK);
    CHECK(verification.passed);
    prefos_free_presolver(presolver);
    return 0;
}

static int test_cone_linked_free_column_substitution(void)
{
    double A_values[] = {1.0, -1.0, 1.0, 1.0, 1.0};
    int A_columns[] = {0, 1, 0, 0, 2};
    int A_rows[] = {0, 2, 3, 5};
    double lower[] = {2.0, 4.0, 5.0};
    double upper[] = {2.0, 4.0, 5.0};
    int Q_rows[] = {0, 0, 0, 0, 0};
    int R_rows[] = {0};
    double c[] = {3.0, -3.0, 0.0, 0.0};
    int box_indices[] = {0};
    double box_lower[] = {-INFINITY};
    double box_upper[] = {INFINITY};
    int cone_indices[] = {1, 2, 3};
    PreFOSConeBlock cone = {PREFOS_CONE_ROTATED_SECOND_ORDER, 3, 0, cone_indices};
    double reduced_x[] = {2.0, 1.0, 0.0};
    double reduced_y[] = {0.0, 0.0};
    double reduced_z[] = {0.0, 0.0, 0.0};
    double original_x[4], original_y[3], original_z[4];
    PreFOSProblemData problem;
    PreFOSSettings settings = prefos_strict_settings();
    PreFOSPresolver *presolver = NULL;
    const PreFOSPresolvedProblem *reduced;
    const PreFOSStats *stats;
    PreFOSPostsolveKKTVerification verification;

    memset(&problem, 0, sizeof(problem));
    problem.n = 4;
    problem.A = (PreFOSCsrMatrix){3, 4, 5, A_values, A_columns, A_rows};
    problem.constraint_lower = lower;
    problem.constraint_upper = upper;
    problem.Q = (PreFOSCsrMatrix){4, 4, 0, NULL, NULL, Q_rows};
    problem.q_storage = PREFOS_Q_UPPER_TRIANGULAR;
    problem.R = (PreFOSCsrMatrix){0, 4, 0, NULL, NULL, R_rows};
    problem.c = c;
    problem.n_box = 1;
    problem.box_indices = box_indices;
    problem.box_lower = box_lower;
    problem.box_upper = box_upper;
    problem.n_cones = 1;
    problem.cones = &cone;

    CHECK(prefos_create_presolver(&problem, &settings, &presolver) == PREFOS_STATUS_OK);
    CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_REDUCED);
    reduced = prefos_get_reduced_problem(presolver);
    stats = prefos_get_stats(presolver);
    CHECK(reduced != NULL && stats != NULL);
    CHECK(stats->substituted_free_variables == 1);
    CHECK(reduced->n == 3 && reduced->A.rows == 2 && reduced->A.nnz == 3);
    CHECK(reduced->A.column_indices[0] == 0);
    CHECK(close_to(reduced->A.values[0], 1.0));
    CHECK(close_to(reduced->constraint_lower[0], 2.0));
    CHECK(close_to(reduced->constraint_upper[0], 2.0));
    CHECK(reduced->A.column_indices[1] == 0);
    CHECK(reduced->A.column_indices[2] == 1);
    CHECK(close_to(reduced->constraint_lower[1], 3.0));
    CHECK(close_to(reduced->constraint_upper[1], 3.0));
    CHECK(reduced->n_box == 0 && reduced->n_cones == 1);
    CHECK(close_to(reduced->c[0], 0.0));
    CHECK(close_to(reduced->objective_offset, 6.0));

    CHECK(prefos_postsolve_primal_dual(presolver, reduced_x, reduced_y, reduced_z,
                                    1e-12, original_x, original_y,
                                    original_z) == PREFOS_STATUS_OK);
    CHECK(close_to(original_x[0], 4.0));
    CHECK(close_to(original_x[1], 2.0));
    CHECK(close_to(original_y[0], -3.0));
    CHECK(close_to(original_y[1], 0.0));
    CHECK(close_to(original_y[2], 0.0));
    CHECK(close_to(original_z[0], 0.0));
    CHECK(prefos_verify_postsolve_kkt(presolver, reduced_x, reduced_y, reduced_z, 1e-12,
                                   &verification) == PREFOS_STATUS_OK);
    CHECK(verification.passed);

    prefos_free_presolver(presolver);
    return 0;
}

static int test_ternary_free_column_substitution(void)
{
    double A_values[] = {1.0, -1.0, -2.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0};
    int A_columns[] = {0, 1, 2, 0, 4, 5, 0, 4, 5};
    int A_rows[] = {0, 3, 6, 9};
    double sides[] = {1.0, 5.0, 5.0};
    int Q_rows[] = {0, 0, 0, 0, 0, 0, 0};
    int R_rows[] = {0};
    double c[] = {3.0, -3.0, -6.0, 0.0, 0.0, 0.0};
    int box_indices[] = {0, 4, 5};
    double box_lower[] = {-INFINITY, -10.0, -10.0};
    double box_upper[] = {INFINITY, 10.0, 10.0};
    int cone_indices[] = {1, 2, 3};
    PreFOSConeBlock cone = {PREFOS_CONE_ROTATED_SECOND_ORDER, 3, 0, cone_indices};
    double reduced_x[] = {2.0, 1.0, 0.0, 0.0, 0.0};
    double reduced_y[] = {0.0, 0.0};
    double reduced_z[] = {0.0, 0.0, 0.0, 0.0, 0.0};
    double original_x[6], original_y[3], original_z[6];
    PreFOSProblemData problem;
    PreFOSSettings settings = prefos_strict_settings();
    PreFOSPresolver *presolver = NULL;
    const PreFOSPresolvedProblem *reduced;
    PreFOSPostsolveKKTVerification verification;

    memset(&problem, 0, sizeof(problem));
    problem.n = 6;
    problem.A = (PreFOSCsrMatrix){3, 6, 9, A_values, A_columns, A_rows};
    problem.constraint_lower = sides;
    problem.constraint_upper = sides;
    problem.Q = (PreFOSCsrMatrix){6, 6, 0, NULL, NULL, Q_rows};
    problem.q_storage = PREFOS_Q_UPPER_TRIANGULAR;
    problem.R = (PreFOSCsrMatrix){0, 6, 0, NULL, NULL, R_rows};
    problem.c = c;
    problem.n_box = 3;
    problem.box_indices = box_indices;
    problem.box_lower = box_lower;
    problem.box_upper = box_upper;
    problem.n_cones = 1;
    problem.cones = &cone;

    CHECK(prefos_create_presolver(&problem, &settings, &presolver) == PREFOS_STATUS_OK);
    CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_REDUCED);
    reduced = prefos_get_reduced_problem(presolver);
    CHECK(prefos_get_stats(presolver)->substituted_free_variables == 1);
    CHECK(prefos_get_stats(presolver)->ternary_substituted_free_variables == 1);
    CHECK(reduced->n == 5 && reduced->A.rows == 2 && reduced->A.nnz == 8);
    CHECK(close_to(reduced->constraint_lower[0], 4.0));
    CHECK(close_to(reduced->constraint_upper[0], 4.0));
    CHECK(close_to(reduced->constraint_lower[1], 4.0));
    CHECK(close_to(reduced->constraint_upper[1], 4.0));
    CHECK(close_to(reduced->c[0], 0.0));
    CHECK(close_to(reduced->c[1], 0.0));
    CHECK(reduced->n_box == 2);
    CHECK(close_to(reduced->objective_offset, 3.0));

    CHECK(prefos_postsolve_primal_dual(presolver, reduced_x, reduced_y, reduced_z,
                                    1e-12, original_x, original_y,
                                    original_z) == PREFOS_STATUS_OK);
    CHECK(close_to(original_x[0], 5.0));
    CHECK(close_to(original_y[0], -3.0));
    CHECK(close_to(original_y[1], 0.0));
    CHECK(close_to(original_y[2], 0.0));
    CHECK(close_to(original_z[0], 0.0));
    CHECK(prefos_verify_postsolve_kkt(presolver, reduced_x, reduced_y, reduced_z, 1e-12,
                                   &verification) == PREFOS_STATUS_OK);
    CHECK(verification.passed);

    prefos_free_presolver(presolver);
    return 0;
}

static int test_lp_ternary_free_column_substitution(void)
{
    double A_values[] = {1.0, -1.0, -2.0};
    int A_columns[] = {0, 1, 2};
    int A_rows[] = {0, 3};
    double side[] = {1.0};
    int Q_rows[] = {0, 0, 0, 0};
    int R_rows[] = {0};
    double c[] = {3.0, -3.0, -6.0};
    int box_indices[] = {0, 1, 2};
    double box_lower[] = {-INFINITY, -10.0, -10.0};
    double box_upper[] = {INFINITY, 10.0, 10.0};
    double reduced_x[] = {2.0, 1.0};
    double reduced_z[] = {0.0, 0.0};
    double original_x[3], original_y[1], original_z[3];
    PreFOSProblemData problem;
    PreFOSSettings settings = prefos_strict_settings();
    PreFOSPresolver *presolver = NULL;
    const PreFOSPresolvedProblem *reduced;
    PreFOSPostsolveKKTVerification verification;

    memset(&problem, 0, sizeof(problem));
    problem.n = 3;
    problem.A = (PreFOSCsrMatrix){1, 3, 3, A_values, A_columns, A_rows};
    problem.constraint_lower = side;
    problem.constraint_upper = side;
    problem.Q = (PreFOSCsrMatrix){3, 3, 0, NULL, NULL, Q_rows};
    problem.q_storage = PREFOS_Q_UPPER_TRIANGULAR;
    problem.R = (PreFOSCsrMatrix){0, 3, 0, NULL, NULL, R_rows};
    problem.c = c;
    problem.n_box = 3;
    problem.box_indices = box_indices;
    problem.box_lower = box_lower;
    problem.box_upper = box_upper;

    CHECK(prefos_create_presolver(&problem, &settings, &presolver) == PREFOS_STATUS_OK);
    CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_REDUCED);
    reduced = prefos_get_reduced_problem(presolver);
    CHECK(prefos_get_stats(presolver)->substituted_free_variables == 1);
    CHECK(prefos_get_stats(presolver)->ternary_substituted_free_variables == 1);
    CHECK(reduced->n == 2 && reduced->A.rows == 0 && reduced->n_box == 2);
    CHECK(close_to(reduced->c[0], 0.0));
    CHECK(close_to(reduced->c[1], 0.0));
    CHECK(close_to(reduced->objective_offset, 3.0));
    CHECK(prefos_postsolve_primal_dual(presolver, reduced_x, NULL, reduced_z, 1e-12,
                                    original_x, original_y,
                                    original_z) == PREFOS_STATUS_OK);
    CHECK(close_to(original_x[0], 5.0));
    CHECK(close_to(original_y[0], -3.0));
    CHECK(prefos_verify_postsolve_kkt(presolver, reduced_x, NULL, reduced_z, 1e-12,
                                   &verification) == PREFOS_STATUS_OK);
    CHECK(verification.passed);

    prefos_free_presolver(presolver);
    return 0;
}

static int test_chained_free_column_substitution(void)
{
    double A_values[] = {1.0, -1.0, 1.0, -1.0};
    int A_columns[] = {0, 1, 1, 2};
    int A_rows[] = {0, 2, 4};
    double sides[] = {1.0, 2.0};
    int Q_rows[] = {0, 0, 0, 0};
    int R_rows[] = {0};
    double c[] = {3.0, 4.0, -7.0};
    int box_indices[] = {0, 1, 2};
    double box_lower[] = {-INFINITY, -INFINITY, -10.0};
    double box_upper[] = {INFINITY, INFINITY, 10.0};
    double reduced_x[] = {0.0};
    double reduced_z[] = {0.0};
    double original_x[3], original_y[2], original_z[3];
    PreFOSProblemData problem;
    PreFOSSettings settings = prefos_strict_settings();
    PreFOSPresolver *presolver = NULL;
    const PreFOSPresolvedProblem *reduced;
    PreFOSPostsolveKKTVerification verification;

    memset(&problem, 0, sizeof(problem));
    problem.n = 3;
    problem.A = (PreFOSCsrMatrix){2, 3, 4, A_values, A_columns, A_rows};
    problem.constraint_lower = sides;
    problem.constraint_upper = sides;
    problem.Q = (PreFOSCsrMatrix){3, 3, 0, NULL, NULL, Q_rows};
    problem.q_storage = PREFOS_Q_UPPER_TRIANGULAR;
    problem.R = (PreFOSCsrMatrix){0, 3, 0, NULL, NULL, R_rows};
    problem.c = c;
    problem.n_box = 3;
    problem.box_indices = box_indices;
    problem.box_lower = box_lower;
    problem.box_upper = box_upper;

    CHECK(prefos_create_presolver(&problem, &settings, &presolver) == PREFOS_STATUS_OK);
    CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_REDUCED);
    reduced = prefos_get_reduced_problem(presolver);
    CHECK(prefos_get_stats(presolver)->substituted_free_variables == 2);
    CHECK(reduced->n == 1 && reduced->A.rows == 0 && reduced->n_box == 1);
    CHECK(close_to(reduced->c[0], 0.0));
    CHECK(close_to(reduced->objective_offset, 17.0));
    CHECK(prefos_postsolve_primal_dual(presolver, reduced_x, NULL, reduced_z, 1e-12,
                                    original_x, original_y,
                                    original_z) == PREFOS_STATUS_OK);
    CHECK(close_to(original_x[0], 3.0));
    CHECK(close_to(original_x[1], 2.0));
    CHECK(close_to(original_x[2], 0.0));
    CHECK(close_to(original_y[0], -3.0));
    CHECK(close_to(original_y[1], -7.0));
    CHECK(prefos_verify_postsolve_kkt(presolver, reduced_x, NULL, reduced_z, 1e-12,
                                   &verification) == PREFOS_STATUS_OK);
    CHECK(verification.passed);

    prefos_free_presolver(presolver);
    return 0;
}

static int test_five_term_free_column_aggregation(void)
{
    double A_values[] = {2.0, 1.0, -2.0, 3.0, -4.0};
    int A_columns[] = {0, 1, 2, 3, 4};
    int A_rows[] = {0, 5};
    double side[] = {10.0};
    int Q_rows[] = {0, 0, 0, 0, 0, 0};
    int R_rows[] = {0};
    double c[] = {2.0, 1.0, -2.0, 3.0, -4.0};
    int box_indices[] = {0, 1, 2, 3, 4};
    double box_lower[] = {-INFINITY, -10.0, -10.0, -10.0, -10.0};
    double box_upper[] = {INFINITY, 10.0, 10.0, 10.0, 10.0};
    double reduced_x[] = {0.0, 0.0, 0.0, 0.0};
    double reduced_z[] = {0.0, 0.0, 0.0, 0.0};
    double original_x[5], original_y[1], original_z[5];
    PreFOSProblemData problem;
    PreFOSSettings settings = prefos_strict_settings();
    PreFOSPresolver *presolver = NULL;
    const PreFOSPresolvedProblem *reduced;
    PreFOSPostsolveKKTVerification verification;
    size_t i;

    memset(&problem, 0, sizeof(problem));
    settings.max_aggregation_terms = 4;
    problem.n = 5;
    problem.A = (PreFOSCsrMatrix){1, 5, 5, A_values, A_columns, A_rows};
    problem.constraint_lower = side;
    problem.constraint_upper = side;
    problem.Q = (PreFOSCsrMatrix){5, 5, 0, NULL, NULL, Q_rows};
    problem.q_storage = PREFOS_Q_UPPER_TRIANGULAR;
    problem.R = (PreFOSCsrMatrix){0, 5, 0, NULL, NULL, R_rows};
    problem.c = c;
    problem.n_box = 5;
    problem.box_indices = box_indices;
    problem.box_lower = box_lower;
    problem.box_upper = box_upper;

    CHECK(prefos_create_presolver(&problem, &settings, &presolver) == PREFOS_STATUS_OK);
    CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_REDUCED);
    reduced = prefos_get_reduced_problem(presolver);
    CHECK(prefos_get_stats(presolver)->substituted_free_variables == 1);
    CHECK(prefos_get_stats(presolver)->ternary_substituted_free_variables == 0);
    CHECK(reduced->n == 4 && reduced->A.rows == 0 && reduced->n_box == 4);
    for (i = 0; i < reduced->n; ++i) CHECK(close_to(reduced->c[i], 0.0));
    CHECK(close_to(reduced->objective_offset, 10.0));
    CHECK(prefos_postsolve_primal_dual(presolver, reduced_x, NULL, reduced_z, 1e-12,
                                    original_x, original_y,
                                    original_z) == PREFOS_STATUS_OK);
    CHECK(close_to(original_x[0], 5.0));
    CHECK(close_to(original_y[0], -1.0));
    CHECK(prefos_verify_postsolve_kkt(presolver, reduced_x, NULL, reduced_z, 1e-12,
                                   &verification) == PREFOS_STATUS_OK);
    CHECK(verification.passed);

    prefos_free_presolver(presolver);
    return 0;
}

static int test_free_column_substitution_rejects_quadratic_incidence(void)
{
    double A_values[] = {1.0, -1.0};
    int A_columns[] = {0, 1};
    int A_rows[] = {0, 2};
    double sides[] = {0.0};
    double quadratic_value[] = {1.0};
    int quadratic_column[] = {0};
    int quadratic_rows[] = {0, 1, 1, 1, 1};
    int empty_quadratic_rows[] = {0, 0, 0, 0, 0};
    double factor_value[] = {1.0};
    int factor_column[] = {0};
    int factor_rows[] = {0, 1};
    int empty_factor_rows[] = {0};
    double D[] = {1.0};
    double c[] = {0.0, 0.0, 0.0, 0.0};
    int box_indices[] = {0};
    double box_lower[] = {-INFINITY};
    double box_upper[] = {INFINITY};
    int cone_indices[] = {1, 2, 3};
    PreFOSConeBlock cone = {PREFOS_CONE_ROTATED_SECOND_ORDER, 3, 0, cone_indices};
    int objective_kind;

    for (objective_kind = 0; objective_kind < 2; ++objective_kind)
    {
        PreFOSProblemData problem;
        PreFOSSettings settings = prefos_strict_settings();
        PreFOSPresolver *presolver = NULL;
        const PreFOSPresolvedProblem *reduced;
        const PreFOSStats *stats;

        memset(&problem, 0, sizeof(problem));
        problem.n = 4;
        problem.A = (PreFOSCsrMatrix){1, 4, 2, A_values, A_columns, A_rows};
        problem.constraint_lower = sides;
        problem.constraint_upper = sides;
        if (objective_kind == 0)
        {
            problem.Q = (PreFOSCsrMatrix){
                4, 4, 1, quadratic_value, quadratic_column, quadratic_rows};
            problem.R = (PreFOSCsrMatrix){0, 4, 0, NULL, NULL, empty_factor_rows};
        }
        else
        {
            problem.Q = (PreFOSCsrMatrix){4, 4, 0, NULL, NULL, empty_quadratic_rows};
            problem.R =
                (PreFOSCsrMatrix){1, 4, 1, factor_value, factor_column, factor_rows};
            problem.D = D;
        }
        problem.q_storage = PREFOS_Q_UPPER_TRIANGULAR;
        problem.c = c;
        problem.n_box = 1;
        problem.box_indices = box_indices;
        problem.box_lower = box_lower;
        problem.box_upper = box_upper;
        problem.n_cones = 1;
        problem.cones = &cone;
        settings.linear_propagation = 0;
        settings.cone_propagation = 0;
        settings.remove_redundant_rows = 0;

        CHECK(prefos_create_presolver(&problem, &settings, &presolver) ==
              PREFOS_STATUS_OK);
        CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_OK);
        reduced = prefos_get_reduced_problem(presolver);
        stats = prefos_get_stats(presolver);
        CHECK(reduced != NULL && stats != NULL);
        CHECK(stats->substituted_free_variables == 0);
        CHECK(reduced->n == 4 && reduced->A.rows == 1);
        prefos_free_presolver(presolver);
    }
    return 0;
}

static int test_gpu_warmup_api(void)
{
    int started = prefos_gpu_warmup_async();
    if (started)
    {
        int available = prefos_gpu_warmup_wait();
        CHECK(prefos_gpu_warmup_ready() == available);
        CHECK(prefos_gpu_warmup() == available);
    }
    else
    {
        CHECK(!prefos_gpu_warmup_ready());
        CHECK(!prefos_gpu_warmup_wait());
    }
    prefos_gpu_release_cache();
    return 0;
}

static int test_exponential_and_power_cone_passthrough(void)
{
    int A_rows[] = {0};
    int Q_rows[] = {0, 0, 0, 0};
    int R_rows[] = {0};
    int cone_indices[] = {0, 1, 2};
    double exp_c[] = {-1.0, -2.0, 3.0};
    double exp_z[] = {1.0, 2.0, -3.0};
    double exp_feasible[] = {0.0, 1.0, 1.0};
    double exp_infeasible[] = {1.0, 1.0, 1.0};
    double power_c[] = {0.3, 0.7, 1.0};
    double power_z[] = {-0.3, -0.7, -1.0};
    double power_feasible[] = {1.0, 1.0, 1.0};
    double power_infeasible[] = {1.0, 1.0, 2.0};
    PreFOSProblemData problem;
    PreFOSSettings settings = prefos_strict_settings();
    PreFOSPresolver *presolver = NULL;
    PreFOSPrimalVerification primal_verification;
    PreFOSPostsolveKKTVerification kkt_verification;
    PreFOSConeBlock cone = {PREFOS_CONE_EXPONENTIAL, 3, 0, cone_indices, 0.0};

    memset(&problem, 0, sizeof(problem));
    problem.n = 3;
    problem.A = (PreFOSCsrMatrix){0, 3, 0, NULL, NULL, A_rows};
    problem.Q = (PreFOSCsrMatrix){3, 3, 0, NULL, NULL, Q_rows};
    problem.q_storage = PREFOS_Q_UPPER_TRIANGULAR;
    problem.R = (PreFOSCsrMatrix){0, 3, 0, NULL, NULL, R_rows};
    problem.c = exp_c;
    problem.n_cones = 1;
    problem.cones = &cone;

    CHECK(prefos_create_presolver(&problem, &settings, &presolver) == PREFOS_STATUS_OK);
    CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_OK);
    CHECK(prefos_get_reduced_problem(presolver)->n_cones == 1);
    CHECK(prefos_get_reduced_problem(presolver)->cones[0].type == PREFOS_CONE_EXPONENTIAL);
    CHECK(prefos_verify_postsolve_primal(presolver, exp_feasible, 1e-10,
                                      &primal_verification) == PREFOS_STATUS_OK);
    CHECK(primal_verification.passed);
    CHECK(prefos_verify_postsolve_primal(presolver, exp_infeasible, 1e-10,
                                      &primal_verification) == PREFOS_STATUS_OK);
    CHECK(!primal_verification.passed);
    CHECK(prefos_verify_postsolve_kkt(presolver, (double[]){0.0, 0.0, 0.0}, NULL, exp_z,
                                   1e-10, &kkt_verification) == PREFOS_STATUS_OK);
    CHECK(kkt_verification.passed);
    prefos_free_presolver(presolver);

    cone.type = PREFOS_CONE_POWER;
    cone.power_alpha = 0.3;
    problem.c = power_c;
    CHECK(prefos_create_presolver(&problem, &settings, &presolver) == PREFOS_STATUS_OK);
    CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_OK);
    CHECK(prefos_get_reduced_problem(presolver)->cones[0].type == PREFOS_CONE_POWER);
    CHECK(close_to(prefos_get_reduced_problem(presolver)->cones[0].power_alpha, 0.3));
    CHECK(prefos_get_stats(presolver)->power_capacity_attempts > 0);
    CHECK(prefos_get_stats(presolver)->power_zero_minimum_abs_z_attempts > 0);
    CHECK(prefos_get_stats(presolver)->power_axis_attempts == 0);
    CHECK(prefos_verify_postsolve_primal(presolver, power_feasible, 1e-10,
                                      &primal_verification) == PREFOS_STATUS_OK);
    CHECK(primal_verification.passed);
    CHECK(prefos_verify_postsolve_primal(presolver, power_infeasible, 1e-10,
                                      &primal_verification) == PREFOS_STATUS_OK);
    CHECK(!primal_verification.passed);
    CHECK(prefos_verify_postsolve_kkt(presolver, (double[]){0.0, 0.0, 0.0}, NULL,
                                   power_z, 1e-10,
                                   &kkt_verification) == PREFOS_STATUS_OK);
    CHECK(kkt_verification.passed);
    prefos_free_presolver(presolver);

    cone.power_alpha = 0.0;
    CHECK(prefos_create_presolver(&problem, &settings, &presolver) ==
          PREFOS_STATUS_INVALID_ARGUMENT);
    cone.power_alpha = 1.0;
    CHECK(prefos_create_presolver(&problem, &settings, &presolver) ==
          PREFOS_STATUS_INVALID_ARGUMENT);
    cone.power_alpha = NAN;
    CHECK(prefos_create_presolver(&problem, &settings, &presolver) ==
          PREFOS_STATUS_INVALID_ARGUMENT);

    {
        double A_value[] = {1.0};
        int A_column[] = {1};
        int envelope_rows[] = {0, 1};
        double lower[] = {0.0};
        double upper[] = {INFINITY};
        int cone_type;
        problem.A = (PreFOSCsrMatrix){1, 3, 1, A_value, A_column, envelope_rows};
        problem.constraint_lower = lower;
        problem.constraint_upper = upper;
        problem.c = exp_c;
        for (cone_type = PREFOS_CONE_EXPONENTIAL; cone_type <= PREFOS_CONE_POWER;
             ++cone_type)
        {
            cone.type = (PreFOSConeType) cone_type;
            cone.power_alpha = cone_type == PREFOS_CONE_POWER ? 0.3 : 0.0;
            A_column[0] = cone_type == PREFOS_CONE_POWER ? 0 : 1;
            CHECK(prefos_create_presolver(&problem, &settings, &presolver) ==
                  PREFOS_STATUS_OK);
            CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_REDUCED);
            CHECK(prefos_get_reduced_problem(presolver)->A.rows == 0);
            CHECK(prefos_get_stats(presolver)->removed_redundant_rows == 1);
            prefos_free_presolver(presolver);
        }
    }
    return 0;
}

static int test_exponential_and_power_nonlinear_propagation(void)
{
    double A_values[] = {1.0, 1.0, 1.0};
    int A_columns[] = {0, 1, 2};
    int A_rows[] = {0, 1, 2, 3};
    int Q_rows[] = {0, 0, 0, 0};
    int R_rows[] = {0};
    int cone_indices[] = {0, 1, 2};
    double c[] = {0.0, 0.0, 0.0};
    double lower[] = {1.0, 1.0, -INFINITY};
    double upper[] = {INFINITY, 1.0, 2.0};
    PreFOSConeBlock cone = {PREFOS_CONE_EXPONENTIAL, 3, 0, cone_indices, 0.0};
    PreFOSProblemData problem;
    PreFOSSettings settings = prefos_strict_settings();
    PreFOSPresolver *presolver = NULL;

    memset(&problem, 0, sizeof(problem));
    problem.n = 3;
    problem.A = (PreFOSCsrMatrix){3, 3, 3, A_values, A_columns, A_rows};
    problem.constraint_lower = lower;
    problem.constraint_upper = upper;
    problem.Q = (PreFOSCsrMatrix){3, 3, 0, NULL, NULL, Q_rows};
    problem.q_storage = PREFOS_Q_UPPER_TRIANGULAR;
    problem.R = (PreFOSCsrMatrix){0, 3, 0, NULL, NULL, R_rows};
    problem.c = c;
    problem.n_cones = 1;
    problem.cones = &cone;

    CHECK(prefos_create_presolver(&problem, &settings, &presolver) == PREFOS_STATUS_OK);
    CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_PRIMAL_INFEASIBLE);
    CHECK(prefos_get_stats(presolver)->exponential_cones_processed > 0);
    CHECK(prefos_get_stats(presolver)->exponential_z_lower_attempts > 0);
    prefos_free_presolver(presolver);

    settings.exponential_propagation = 0;
    CHECK(prefos_create_presolver(&problem, &settings, &presolver) == PREFOS_STATUS_OK);
    CHECK(prefos_run_presolve(presolver) != PREFOS_STATUS_PRIMAL_INFEASIBLE);
    CHECK(prefos_get_stats(presolver)->exponential_cones_processed > 0);
    CHECK(prefos_get_stats(presolver)->exponential_z_lower_attempts == 0);
    CHECK(prefos_get_stats(presolver)->exponential_x_upper_attempts == 0);
    prefos_free_presolver(presolver);
    settings.exponential_propagation = 1;

    cone.type = PREFOS_CONE_POWER;
    cone.power_alpha = 0.5;
    lower[0] = -INFINITY;
    lower[1] = -INFINITY;
    lower[2] = 7.0;
    upper[0] = 4.0;
    upper[1] = 9.0;
    upper[2] = 7.0;
    CHECK(prefos_create_presolver(&problem, &settings, &presolver) == PREFOS_STATUS_OK);
    CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_PRIMAL_INFEASIBLE);
    CHECK(prefos_get_stats(presolver)->power_cones_processed > 0);
    CHECK(prefos_get_stats(presolver)->power_capacity_attempts > 0);
    prefos_free_presolver(presolver);

    settings.power_propagation = 0;
    CHECK(prefos_create_presolver(&problem, &settings, &presolver) == PREFOS_STATUS_OK);
    CHECK(prefos_run_presolve(presolver) != PREFOS_STATUS_PRIMAL_INFEASIBLE);
    CHECK(prefos_get_stats(presolver)->power_cones_processed > 0);
    CHECK(prefos_get_stats(presolver)->power_capacity_attempts == 0);
    CHECK(prefos_get_stats(presolver)->power_axis_attempts == 0);
    prefos_free_presolver(presolver);
    return 0;
}

static int test_exponential_and_power_coordinate_reductions(void)
{
    double A_value[] = {1.0};
    int A_column[] = {1};
    int A_rows[] = {0, 1};
    int Q_rows[] = {0, 0, 0, 0};
    int R_rows[] = {0};
    int cone_indices[] = {0, 1, 2};
    double c[] = {0.0, 0.0, 0.0};
    double lower[] = {-INFINITY};
    double upper[] = {0.0};
    PreFOSConeBlock cone = {PREFOS_CONE_EXPONENTIAL, 3, 0, cone_indices, 0.0};
    PreFOSProblemData problem;
    PreFOSSettings settings = prefos_strict_settings();
    PreFOSPresolver *presolver = NULL;
    const PreFOSPresolvedProblem *reduced;
    const PreFOSFacialReductionCertificate *certificate;
    size_t certificate_count;
    double original_x[3];

    memset(&problem, 0, sizeof(problem));
    problem.n = 3;
    problem.A = (PreFOSCsrMatrix){1, 3, 1, A_value, A_column, A_rows};
    problem.constraint_lower = lower;
    problem.constraint_upper = upper;
    problem.Q = (PreFOSCsrMatrix){3, 3, 0, NULL, NULL, Q_rows};
    problem.q_storage = PREFOS_Q_UPPER_TRIANGULAR;
    problem.R = (PreFOSCsrMatrix){0, 3, 0, NULL, NULL, R_rows};
    problem.c = c;
    problem.n_cones = 1;
    problem.cones = &cone;

    CHECK(prefos_create_presolver(&problem, &settings, &presolver) == PREFOS_STATUS_OK);
    CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_REDUCED);
    reduced = prefos_get_reduced_problem(presolver);
    CHECK(reduced->n == 2 && reduced->n_cones == 0 && reduced->n_box == 2);
    CHECK(reduced->box_lower[0] == -INFINITY && reduced->box_upper[0] == 0.0);
    CHECK(reduced->box_lower[1] == 0.0 && reduced->box_upper[1] == INFINITY);
    CHECK(prefos_postsolve_primal(presolver, (double[]){-1.0, 2.0}, original_x) ==
          PREFOS_STATUS_OK);
    CHECK(close_to(original_x[0], -1.0) && close_to(original_x[1], 0.0) &&
          close_to(original_x[2], 2.0));
    CHECK(prefos_postsolve_primal_dual(presolver, (double[]){-1.0, 2.0}, NULL,
                                    (double[]){0.0, 0.0}, 1e-10, original_x, NULL,
                                    (double[]){0.0, 0.0, 0.0}) ==
          PREFOS_STATUS_DUAL_RECOVERY_UNAVAILABLE);
    certificate = prefos_get_facial_reductions(presolver, &certificate_count);
    CHECK(certificate && certificate_count == 1);
    CHECK(certificate[0].type == PREFOS_FACE_EXPONENTIAL_Y_ZERO);
    CHECK(prefos_get_stats(presolver)->reduced_exponential_faces == 1);
    prefos_free_presolver(presolver);

    settings.exponential_face_reduction = 0;
    CHECK(prefos_create_presolver(&problem, &settings, &presolver) == PREFOS_STATUS_OK);
    CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_OK);
    CHECK(prefos_get_reduced_problem(presolver)->n_cones == 1);
    prefos_free_presolver(presolver);
    settings.exponential_face_reduction = 1;

    A_column[0] = 2;
    lower[0] = 0.0;
    upper[0] = 0.0;
    CHECK(prefos_create_presolver(&problem, &settings, &presolver) == PREFOS_STATUS_OK);
    CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_REDUCED);
    reduced = prefos_get_reduced_problem(presolver);
    CHECK(reduced->n == 1 && reduced->n_cones == 0 && reduced->n_box == 1);
    CHECK(reduced->box_lower[0] == -INFINITY && reduced->box_upper[0] == 0.0);
    certificate = prefos_get_facial_reductions(presolver, &certificate_count);
    CHECK(certificate && certificate[0].type == PREFOS_FACE_EXPONENTIAL_Z_ZERO);
    prefos_free_presolver(presolver);

    cone.type = PREFOS_CONE_POWER;
    cone.power_alpha = 0.3;
    A_column[0] = 0;
    lower[0] = -INFINITY;
    upper[0] = 0.0;
    CHECK(prefos_create_presolver(&problem, &settings, &presolver) == PREFOS_STATUS_OK);
    CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_REDUCED);
    reduced = prefos_get_reduced_problem(presolver);
    CHECK(reduced->n == 1 && reduced->n_cones == 0 && reduced->n_box == 1);
    CHECK(reduced->box_lower[0] == 0.0 && reduced->box_upper[0] == INFINITY);
    certificate = prefos_get_facial_reductions(presolver, &certificate_count);
    CHECK(certificate && certificate[0].type == PREFOS_FACE_POWER_X_ZERO);
    prefos_free_presolver(presolver);

    settings.power_face_reduction = 0;
    CHECK(prefos_create_presolver(&problem, &settings, &presolver) == PREFOS_STATUS_OK);
    CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_OK);
    CHECK(prefos_get_reduced_problem(presolver)->n_cones == 1);
    prefos_free_presolver(presolver);
    settings.power_face_reduction = 1;

    A_column[0] = 2;
    lower[0] = 0.0;
    upper[0] = 0.0;
    CHECK(prefos_create_presolver(&problem, &settings, &presolver) == PREFOS_STATUS_OK);
    CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_REDUCED);
    reduced = prefos_get_reduced_problem(presolver);
    CHECK(reduced->n == 2 && reduced->n_cones == 0 && reduced->n_box == 2);
    CHECK(reduced->box_lower[0] == 0.0 && reduced->box_lower[1] == 0.0);
    certificate = prefos_get_facial_reductions(presolver, &certificate_count);
    CHECK(certificate && certificate[0].type == PREFOS_FACE_POWER_Z_ZERO);
    CHECK(prefos_get_stats(presolver)->reduced_power_faces == 1);
    prefos_free_presolver(presolver);
    return 0;
}

static int test_cone_aware_row_activity(void)
{
    double A_values[] = {2.0, 1.0};
    int A_columns[] = {0, 1};
    int A_rows[] = {0, 2};
    int Q_rows[] = {0, 0, 0};
    int R_rows[] = {0};
    int cone_indices[] = {0, 1};
    double c[] = {0.0, 0.0};
    double lower[] = {0.0};
    double upper[] = {INFINITY};
    PreFOSConeBlock cone = {PREFOS_CONE_SECOND_ORDER, 2, 0, cone_indices, 0.0};
    PreFOSProblemData problem;
    PreFOSSettings settings = prefos_strict_settings();
    PreFOSPresolver *presolver = NULL;

    memset(&problem, 0, sizeof(problem));
    problem.n = 2;
    problem.A = (PreFOSCsrMatrix){1, 2, 2, A_values, A_columns, A_rows};
    problem.constraint_lower = lower;
    problem.constraint_upper = upper;
    problem.Q = (PreFOSCsrMatrix){2, 2, 0, NULL, NULL, Q_rows};
    problem.q_storage = PREFOS_Q_UPPER_TRIANGULAR;
    problem.R = (PreFOSCsrMatrix){0, 2, 0, NULL, NULL, R_rows};
    problem.c = c;
    problem.n_cones = 1;
    problem.cones = &cone;

    CHECK(prefos_create_presolver(&problem, &settings, &presolver) == PREFOS_STATUS_OK);
    CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_REDUCED);
    CHECK(prefos_get_reduced_problem(presolver)->A.rows == 0);
    CHECK(prefos_get_stats(presolver)->removed_redundant_rows == 1);
    CHECK(prefos_get_stats(presolver)->cone_activity_rows == 1);
    CHECK(prefos_get_stats(presolver)->cone_activity_blocks == 1);
    CHECK(prefos_get_stats(presolver)->cone_activity_lower_support_hits == 1);
    CHECK(prefos_get_stats(presolver)->cone_activity_strengthened_rows == 1);
    CHECK(prefos_get_stats(presolver)->cone_activity_rows_removed == 1);
    prefos_free_presolver(presolver);

    settings.cone_aware_row_activity = 0;
    CHECK(prefos_create_presolver(&problem, &settings, &presolver) == PREFOS_STATUS_OK);
    CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_OK);
    CHECK(prefos_get_reduced_problem(presolver)->A.rows == 1);
    CHECK(prefos_get_stats(presolver)->cone_activity_rows == 0);
    prefos_free_presolver(presolver);
    settings.cone_aware_row_activity = 1;

    lower[0] = -INFINITY;
    upper[0] = -1.0;
    CHECK(prefos_create_presolver(&problem, &settings, &presolver) == PREFOS_STATUS_OK);
    CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_PRIMAL_INFEASIBLE);
    prefos_free_presolver(presolver);

    {
        double coupled_values[] = {-1.0, 0.0, 1.0};
        int coupled_columns[] = {0, 1, 2};
        int coupled_rows[] = {0, 3};
        int coupled_q_rows[] = {0, 0, 0, 0};
        int coupled_indices[] = {0, 1, 2};
        double coupled_c[] = {0.0, 0.0, 0.0};
        double coupled_lower[] = {0.0};
        double coupled_upper[] = {INFINITY};
        PreFOSConeBlock coupled_cone = {PREFOS_CONE_EXPONENTIAL, 3, 0, coupled_indices,
                                     0.0};
        int cone_case;
        memset(&problem, 0, sizeof(problem));
        problem.n = 3;
        problem.A =
            (PreFOSCsrMatrix){1, 3, 3, coupled_values, coupled_columns, coupled_rows};
        problem.constraint_lower = coupled_lower;
        problem.constraint_upper = coupled_upper;
        problem.Q = (PreFOSCsrMatrix){3, 3, 0, NULL, NULL, coupled_q_rows};
        problem.q_storage = PREFOS_Q_UPPER_TRIANGULAR;
        problem.R = (PreFOSCsrMatrix){0, 3, 0, NULL, NULL, R_rows};
        problem.c = coupled_c;
        problem.n_cones = 1;
        problem.cones = &coupled_cone;
        for (cone_case = 0; cone_case < 3; ++cone_case)
        {
            if (cone_case == 0)
            {
                coupled_cone.type = PREFOS_CONE_EXPONENTIAL;
                coupled_cone.matrix_order = 0;
                coupled_cone.power_alpha = 0.0;
                coupled_values[0] = -1.0;
                coupled_values[1] = 0.0;
                coupled_values[2] = 1.0;
            }
            else if (cone_case == 1)
            {
                coupled_cone.type = PREFOS_CONE_POWER;
                coupled_cone.matrix_order = 0;
                coupled_cone.power_alpha = 0.3;
                coupled_values[0] = 0.3;
                coupled_values[1] = 0.7;
                coupled_values[2] = 0.5;
            }
            else
            {
                coupled_cone.type = PREFOS_CONE_POSITIVE_SEMIDEFINITE;
                coupled_cone.matrix_order = 2;
                coupled_cone.power_alpha = 0.0;
                coupled_values[0] = 1.0;
                coupled_values[1] = 0.25 * sqrt(2.0);
                coupled_values[2] = 1.0;
            }
            CHECK(prefos_create_presolver(&problem, &settings, &presolver) ==
                  PREFOS_STATUS_OK);
            CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_REDUCED);
            CHECK(prefos_get_reduced_problem(presolver)->A.rows == 0);
            CHECK(prefos_get_stats(presolver)->removed_redundant_rows == 1);
            prefos_free_presolver(presolver);
        }
    }
    return 0;
}

static int test_psd_higher_order_propagation(void)
{
    int R_rows[] = {0};
    PreFOSSettings settings = prefos_strict_settings();
    PreFOSPresolver *presolver = NULL;

    {
        int empty_rows[] = {0};
        int Q_rows[] = {0, 0};
        int cone_indices[] = {0};
        double c[] = {0.0};
        PreFOSConeBlock cone = {PREFOS_CONE_POSITIVE_SEMIDEFINITE, 1, 1, cone_indices,
                             0.0};
        PreFOSProblemData problem;
        memset(&problem, 0, sizeof(problem));
        problem.n = 1;
        problem.A = (PreFOSCsrMatrix){0, 1, 0, NULL, NULL, empty_rows};
        problem.Q = (PreFOSCsrMatrix){1, 1, 0, NULL, NULL, Q_rows};
        problem.q_storage = PREFOS_Q_UPPER_TRIANGULAR;
        problem.R = (PreFOSCsrMatrix){0, 1, 0, NULL, NULL, R_rows};
        problem.c = c;
        problem.n_cones = 1;
        problem.cones = &cone;
        CHECK(prefos_create_presolver(&problem, &settings, &presolver) ==
              PREFOS_STATUS_OK);
        CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_REDUCED);
        CHECK(prefos_get_reduced_problem(presolver)->n_cones == 0);
        CHECK(prefos_get_reduced_problem(presolver)->n_box == 1);
        prefos_free_presolver(presolver);
    }
    {
        double A_values[] = {1, 1, 1, 1, 1, 1};
        int A_columns[] = {0, 1, 2, 3, 4, 5};
        int A_rows[] = {0, 1, 2, 3, 4, 5, 6};
        int Q_rows[] = {0, 0, 0, 0, 0, 0, 0};
        int cone_indices[] = {0, 1, 2, 3, 4, 5};
        double c[] = {0, 0, 0, 0, 0, 0};
        double lower[] = {-INFINITY, sqrt(2.0), 1.0, sqrt(2.0), 0.0, 1.0};
        double upper[] = {1.5, sqrt(2.0), 1.0, sqrt(2.0), 0.0, 1.0};
        PreFOSConeBlock cone = {PREFOS_CONE_POSITIVE_SEMIDEFINITE, 6, 3, cone_indices,
                             0.0};
        PreFOSProblemData problem;
        memset(&problem, 0, sizeof(problem));
        problem.n = 6;
        problem.A = (PreFOSCsrMatrix){6, 6, 6, A_values, A_columns, A_rows};
        problem.constraint_lower = lower;
        problem.constraint_upper = upper;
        problem.Q = (PreFOSCsrMatrix){6, 6, 0, NULL, NULL, Q_rows};
        problem.q_storage = PREFOS_Q_UPPER_TRIANGULAR;
        problem.R = (PreFOSCsrMatrix){0, 6, 0, NULL, NULL, R_rows};
        problem.c = c;
        problem.n_cones = 1;
        problem.cones = &cone;
        CHECK(prefos_create_presolver(&problem, &settings, &presolver) ==
              PREFOS_STATUS_OK);
        CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_PRIMAL_INFEASIBLE);
        CHECK(prefos_get_stats(presolver)->psd_cones_processed > 0);
        CHECK(prefos_get_stats(presolver)->psd_three_by_three_attempts > 0);
        CHECK(prefos_get_stats(presolver)->psd_schur_attempts > 0);
        prefos_free_presolver(presolver);

        settings.psd_higher_order_propagation = 0;
        CHECK(prefos_create_presolver(&problem, &settings, &presolver) ==
              PREFOS_STATUS_OK);
        CHECK(prefos_run_presolve(presolver) != PREFOS_STATUS_PRIMAL_INFEASIBLE);
        CHECK(prefos_get_stats(presolver)->psd_cones_processed > 0);
        CHECK(prefos_get_stats(presolver)->psd_two_by_two_attempts > 0);
        CHECK(prefos_get_stats(presolver)->psd_three_by_three_attempts == 0);
        CHECK(prefos_get_stats(presolver)->psd_schur_attempts == 0);
        prefos_free_presolver(presolver);
        settings.psd_higher_order_propagation = 1;

        lower[0] = upper[0] = 1.0;
        lower[1] = upper[1] = -0.9 * sqrt(2.0);
        lower[2] = upper[2] = 1.0;
        lower[3] = upper[3] = -0.9 * sqrt(2.0);
        lower[4] = upper[4] = -0.9 * sqrt(2.0);
        lower[5] = upper[5] = 1.0;
        CHECK(prefos_create_presolver(&problem, &settings, &presolver) ==
              PREFOS_STATUS_OK);
        CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_PRIMAL_INFEASIBLE);
        prefos_free_presolver(presolver);
    }
    {
        double A_values[10], c[10], lower[10], upper[10];
        int A_columns[10], A_rows[11], Q_rows[11], cone_indices[10];
        PreFOSConeBlock cone = {PREFOS_CONE_POSITIVE_SEMIDEFINITE, 10, 4, cone_indices,
                             0.0};
        PreFOSProblemData problem;
        size_t row, column, position = 0;
        for (row = 0; row < 10; ++row)
        {
            A_values[row] = 1.0;
            A_columns[row] = (int) row;
            A_rows[row] = (int) row;
            Q_rows[row] = 0;
            cone_indices[row] = (int) row;
            c[row] = 0.0;
        }
        A_rows[10] = 10;
        Q_rows[10] = 0;
        for (row = 0; row < 4; ++row)
        {
            for (column = 0; column <= row; ++column)
            {
                double value = row == column ? 1.0 : -0.34 * sqrt(2.0);
                lower[position] = upper[position] = value;
                ++position;
            }
        }
        memset(&problem, 0, sizeof(problem));
        problem.n = 10;
        problem.A = (PreFOSCsrMatrix){10, 10, 10, A_values, A_columns, A_rows};
        problem.constraint_lower = lower;
        problem.constraint_upper = upper;
        problem.Q = (PreFOSCsrMatrix){10, 10, 0, NULL, NULL, Q_rows};
        problem.q_storage = PREFOS_Q_UPPER_TRIANGULAR;
        problem.R = (PreFOSCsrMatrix){0, 10, 0, NULL, NULL, R_rows};
        problem.c = c;
        problem.n_cones = 1;
        problem.cones = &cone;
        CHECK(prefos_create_presolver(&problem, &settings, &presolver) ==
              PREFOS_STATUS_OK);
        CHECK(prefos_run_presolve(presolver) == PREFOS_STATUS_PRIMAL_INFEASIBLE);
        CHECK(prefos_get_stats(presolver)->psd_fixed_window_checks > 0);
        prefos_free_presolver(presolver);
    }
    return 0;
}

int main(void)
{
    if (test_gpu_warmup_api()) return 1;
    if (test_exponential_and_power_cone_passthrough()) return 1;
    if (test_exponential_and_power_nonlinear_propagation()) return 1;
    if (test_exponential_and_power_coordinate_reductions()) return 1;
    if (test_cone_aware_row_activity()) return 1;
    if (test_psd_higher_order_propagation()) return 1;
    if (test_fixed_variable_and_objective_update()) return 1;
    if (test_infeasible_empty_row()) return 1;
    if (test_domain_overlap_is_rejected()) return 1;
    if (test_singleton_box_row_tightening()) return 1;
    if (test_objective_overflow_is_reported()) return 1;
    if (test_iterative_linear_propagation_with_soc_envelope()) return 1;
    if (test_event_driven_linear_propagation()) return 1;
    if (test_huge_implied_bound_is_skipped()) return 1;
    if (test_linear_propagation_detects_infeasibility()) return 1;
    if (test_propagated_bound_policy()) return 1;
    if (test_affine_cone_coordinate_aggregation()) return 1;
    if (test_affine_cone_input_passthrough()) return 1;
    if (test_affine_aggregation_full_dual()) return 1;
    if (test_affine_cone_propagation()) return 1;
    if (test_affine_singleton_bound_certificate()) return 1;
    if (test_generated_affine_singleton_bound_certificate()) return 1;
    if (test_input_affine_structural_faces()) return 1;
    if (test_generated_affine_structural_face()) return 1;
    if (test_input_affine_rsoc_face_substitution()) return 1;
    if (test_generated_affine_rsoc_face_fixing()) return 1;
    if (test_zero_affine_block_removal()) return 1;
    if (test_affine_psd_propagation_budget()) return 1;
    if (test_affine_psd_structure_and_block_decomposition()) return 1;
    if (test_duplicate_a_columns_are_rejected()) return 1;
    if (test_psd_verification()) return 1;
    if (test_strict_settings()) return 1;
    if (test_fixed_variable_kkt_postsolve()) return 1;
    if (test_propagation_dual_transfer()) return 1;
    if (test_cyclic_propagation_dual_transfer()) return 1;
    if (test_deleted_singleton_row_dual_transfer()) return 1;
    if (test_soc_kkt_convention()) return 1;
    if (test_soc_envelope_propagation()) return 1;
    if (test_soc_and_rsoc_infeasibility()) return 1;
    if (test_psd_pairwise_and_full_infeasibility()) return 1;
    if (test_nonnegative_box_normalization()) return 1;
    if (test_activity_redundant_rows()) return 1;
    if (test_soc_zero_collapse_kkt()) return 1;
    if (test_zero_cone_collapse_compacts_objective()) return 1;
    if (test_rsoc_zero_collapse_kkt()) return 1;
    if (test_rsoc_facial_reduction_extended_dual()) return 1;
    if (test_psd_facial_reduction_extended_dual()) return 1;
    if (test_psd_zero_collapse_kkt()) return 1;
    if (test_nonfinite_postsolve_inputs_are_rejected()) return 1;
    if (test_full_q_symmetry_validation()) return 1;
    if (test_empty_model_round_trip()) return 1;
    if (test_parallel_row_reduction_and_postsolve()) return 1;
    if (test_infeasible_parallel_rows()) return 1;
    if (test_interleaved_row_and_bound_postsolve()) return 1;
    if (test_cone_linked_free_column_substitution()) return 1;
    if (test_ternary_free_column_substitution()) return 1;
    if (test_lp_ternary_free_column_substitution()) return 1;
    if (test_chained_free_column_substitution()) return 1;
    if (test_five_term_free_column_aggregation()) return 1;
    if (test_free_column_substitution_rejects_quadratic_incidence()) return 1;
    printf("All PreFOS tests passed!\n");
    return 0;
}
