/*
 * Copyright 2026 Hongpei Li
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef PREFOS_H
#define PREFOS_H

#define PREFOS_VERSION_MAJOR 0
#define PREFOS_VERSION_MINOR 1
#define PREFOS_VERSION_PATCH 0
#define PREFOS_VERSION "0.1.0"

#if defined(_WIN32) && defined(PREFOS_BUILDING_SHARED_LIBRARY)
#define PREFOS_API __declspec(dllexport)
#elif defined(_WIN32) && defined(PREFOS_USING_SHARED_LIBRARY)
#define PREFOS_API __declspec(dllimport)
#elif defined(__GNUC__) || defined(__clang__)
#define PREFOS_API __attribute__((visibility("default")))
#else
#define PREFOS_API
#endif

#ifdef __cplusplus
#include <cstddef>
extern "C"
{
#else
#include <stddef.h>
#endif

    typedef enum
    {
        PREFOS_STATUS_OK = 0,
        PREFOS_STATUS_REDUCED,
        PREFOS_STATUS_PRIMAL_INFEASIBLE,
        PREFOS_STATUS_INVALID_ARGUMENT,
        PREFOS_STATUS_NUMERICAL_ERROR,
        PREFOS_STATUS_OUT_OF_MEMORY,
        PREFOS_STATUS_DUAL_RECOVERY_UNAVAILABLE,
        PREFOS_STATUS_PRIMAL_UNBOUNDED
    } PreFOSStatus;

    typedef enum
    {
        PREFOS_CONE_NONNEGATIVE = 0,
        PREFOS_CONE_SECOND_ORDER,
        PREFOS_CONE_ROTATED_SECOND_ORDER,
        PREFOS_CONE_POSITIVE_SEMIDEFINITE,
        PREFOS_CONE_EXPONENTIAL,
        PREFOS_CONE_POWER
    } PreFOSConeType;

    typedef enum
    {
        PREFOS_Q_UPPER_TRIANGULAR = 0,
        PREFOS_Q_LOWER_TRIANGULAR,
        PREFOS_Q_FULL
    } PreFOSQStorage;

    typedef enum
    {
        /* Keep every accepted propagated box bound in the reduced model. */
        PREFOS_PROPAGATED_BOUND_POLICY_FIRST_ORDER = 0,
        /* Avoid adding a finite side to a box that was unbounded on that side. */
        PREFOS_PROPAGATED_BOUND_POLICY_INTERIOR_POINT
    } PreFOSPropagatedBoundPolicy;

    /* A general sparse matrix in CSR format. A rows must not contain duplicate
     * column indices; column order does not matter. */
    typedef struct
    {
        size_t rows;
        size_t cols;
        size_t nnz;
        double *values;
        int *column_indices;
        int *row_pointers;
    } PreFOSCsrMatrix;

    /*
     * Cone indices are ordered. For SOC they are (t, z), for rotated SOC they
     * are (u, v, z), and PSD variables use row-major svec order:
     * (0,0), sqrt(2)*(1,0), (1,1), sqrt(2)*(2,0), ... . This scaling makes
     * the PSD cone self-dual under the vector Euclidean inner product.
     * Exponential cones use (x,y,z) with y*exp(x/y) <= z. Power cones use
     * (x,y,z) with x^alpha*y^(1-alpha) >= |z|. matrix_order is used only by
     * PSD cones and must satisfy
     * dimension = matrix_order * (matrix_order + 1) / 2.
     */
    typedef struct
    {
        PreFOSConeType type;
        size_t dimension;
        size_t matrix_order;
        int *indices;
        /* Required in (0,1) for power cones; ignored for other cone types. */
        double power_alpha;
    } PreFOSConeBlock;

    /*
     * Rows of affine_cone_matrix are partitioned into contiguous cone blocks.
     * The blocks appear in this array in row order, so row_start is implicit
     * from the sum of preceding dimensions.
     */
    typedef struct
    {
        PreFOSConeType type;
        size_t dimension;
        size_t matrix_order;
        double power_alpha;
    } PreFOSAffineConeBlock;

    /*
     * Input model. The constructor deep-copies every array. The box indices and
     * cone block indices must be disjoint and together cover all n variables.
     * Infinite row and box bounds use IEEE-754 infinities. Q must be symmetric:
     * triangular storage contains each entry once, while full storage contains
     * each nonzero once in both triangles with exactly matching values. D must
     * be elementwise nonnegative. The lightweight input validation does not
     * perform a positive-semidefiniteness test for Q. Input nonnegative-cone
     * variables are canonicalized to [0, +infinity] boxes in the presolver's
     * internal copy and in the reduced model.
     */
    typedef struct
    {
        size_t n;

        PreFOSCsrMatrix A;
        double *constraint_lower;
        double *constraint_upper;

        PreFOSCsrMatrix Q;
        PreFOSQStorage q_storage;
        PreFOSCsrMatrix R;
        double *D;

        double *c;
        double objective_offset;

        size_t n_box;
        int *box_indices;
        double *box_lower;
        double *box_upper;

        size_t n_cones;
        PreFOSConeBlock *cones;

        /* Additional affine-image cone constraints G*x + h in K. */
        PreFOSCsrMatrix affine_cone_matrix;
        double *affine_cone_offset;
        size_t n_affine_cones;
        PreFOSAffineConeBlock *affine_cones;
    } PreFOSProblemData;

    typedef struct
    {
        double feasibility_tolerance;
        double fixed_variable_tolerance;
        int fix_close_box_bounds;
        int remove_empty_rows;
        int remove_redundant_rows;
        /* Bounded equality aggregation for free box columns outside Q and R. */
        int free_column_substitution;
        /* Number of retained terms; source equality width is one larger. */
        int max_aggregation_terms;
        int max_aggregation_column_degree;
        int max_aggregation_fill;
        int max_aggregation_rounds;
        double max_aggregation_scale;
        int linear_propagation;
        int max_linear_propagation_rounds;
        /* CUDA bulk linear and cone propagation; ignored without CUDA. */
        int linear_propagation_gpu;
        /* Zero work ratio or zero stale rounds disables that adaptive limit. */
        double linear_propagation_max_work_ratio;
        double linear_propagation_min_changes_per_million;
        int linear_propagation_max_stale_rounds;
        int cone_propagation;
        int max_cone_propagation_rounds;
        /* Use dual-cone support when checking linear row activity. */
        int cone_aware_row_activity;
        /* Nonlinear coordinate propagation; intrinsic cone signs stay active. */
        int exponential_propagation;
        int power_propagation;
        /* Three-by-three Schur bounds and fixed PSD window checks. */
        int psd_higher_order_propagation;
        /* Zero disables the sparse affine-PSD coordinate work budget. */
        double affine_psd_propagation_max_work_ratio;
        int rsoc_face_reduction;
        int psd_face_reduction;
        int exponential_face_reduction;
        int power_face_reduction;
        /* Finite bound changes must exceed both improvement thresholds. */
        double finite_bound_improvement_absolute;
        double finite_bound_improvement_relative;
        /* Dense columns use bulk scans; queued fan-out is capped by active nnz. */
        double event_queue_max_average_column_degree;
        double event_queue_activity_update_ratio;
        /* Controls which propagated and affine-singleton bounds are exposed. */
        PreFOSPropagatedBoundPolicy propagated_bound_policy;
        /* Convert eligible direct cone coordinates into affine cone rows. */
        int affine_cone_coordinate_aggregation;
        /* Analyze affine PSD sparsity after facial reduction. */
        int psd_structure_analysis;
        /* Split exactly block-diagonal affine PSD images into PSD components. */
        int psd_block_decomposition;
        /* LP-style column reductions; nonlinear and coupled columns are skipped. */
        int remove_empty_columns;
        int singleton_column_reduction;
        int bounded_doubleton_substitution;
        int dual_fixing;
        int parallel_column_reduction;
        /* Remove row-implied box sides in the interior-point bound policy. */
        int remove_redundant_bounds;
        /* Use CUDA for structural passes and profitable A compaction. */
        int structural_reductions_gpu;
        /*
         * Skip parallel-row detection above this average input row width.
         * Zero disables the adaptive work limit.
         */
        double parallel_row_max_average_nnz;
        /*
         * Skip activity-based redundant-row checks above this average width.
         * Zero disables the adaptive work limit.
         */
        double redundant_row_max_average_nnz;
    } PreFOSSettings;

    /*
     * Structural description of one affine PSD block before optional block
     * decomposition. affine_cone_index and row_start refer to the reduced
     * affine model immediately after facial reduction; a decomposed block can
     * therefore produce several blocks in the final reduced model.
     */
    typedef struct
    {
        size_t affine_cone_index;
        size_t row_start;
        size_t matrix_order;
        size_t dimension;
        size_t affine_nnz;
        size_t active_diagonal_coordinates;
        size_t active_offdiagonal_coordinates;
        size_t coefficient_columns;
        size_t diagonal_coefficient_columns;
        size_t single_diagonal_coefficient_columns;
        size_t connected_components;
        size_t scalar_components;
        size_t emitted_cone_blocks;
        size_t largest_component_order;
        size_t decomposed_dimension;
        int exactly_block_diagonal;
    } PreFOSPSDStructureAnalysis;

    typedef struct
    {
        size_t rows_original;
        size_t rows_reduced;
        size_t variables_original;
        size_t variables_reduced;
        size_t nnz_A_original;
        size_t nnz_A_reduced;
        size_t nnz_Q_original;
        size_t nnz_Q_reduced;
        size_t nnz_R_original;
        size_t nnz_R_reduced;
        size_t normalized_nonnegative_variables;
        size_t normalized_nonnegative_cones;
        size_t fixed_box_variables;
        size_t substituted_free_variables;
        size_t ternary_substituted_free_variables;
        size_t tightened_box_bounds;
        size_t propagated_box_bounds;
        size_t linear_propagation_rounds;
        size_t linear_event_rounds;
        size_t linear_full_scan_rounds;
        size_t linear_full_scan_fallbacks;
        size_t linear_activity_nnz_computed;
        size_t linear_rows_processed;
        size_t linear_nnz_processed;
        size_t linear_activity_updates;
        size_t linear_budget_stops;
        size_t linear_stale_stops;
        double linear_propagation_milliseconds;
        size_t linear_gpu_rounds;
        size_t linear_gpu_fallbacks;
        double linear_gpu_setup_milliseconds;
        double linear_gpu_transfer_milliseconds;
        double linear_gpu_kernel_milliseconds;
        double linear_gpu_total_milliseconds;
        size_t linear_gpu_long_rows;
        size_t tightened_cone_envelopes;
        size_t cone_propagation_rounds;
        double cone_propagation_milliseconds;
        size_t cone_activity_rows;
        size_t cone_activity_blocks;
        size_t cone_activity_lower_support_hits;
        size_t cone_activity_upper_support_hits;
        size_t cone_activity_strengthened_rows;
        size_t cone_activity_rows_removed;
        size_t cone_activity_infeasible_rows;
        size_t redundant_row_activity_budget_skips;
        double redundant_row_activity_milliseconds;
        size_t exponential_cones_processed;
        size_t exponential_z_lower_attempts;
        size_t exponential_z_lower_hits;
        size_t exponential_x_upper_attempts;
        size_t exponential_x_upper_hits;
        double exponential_propagation_milliseconds;
        size_t power_cones_processed;
        size_t power_capacity_attempts;
        size_t power_unbounded_capacity_attempts;
        size_t power_zero_minimum_abs_z_attempts;
        size_t power_z_bound_hits;
        size_t power_axis_attempts;
        size_t power_axis_hits;
        double power_propagation_milliseconds;
        size_t psd_cones_processed;
        size_t psd_two_by_two_attempts;
        size_t psd_two_by_two_bound_hits;
        size_t psd_three_by_three_attempts;
        size_t psd_schur_attempts;
        size_t psd_schur_bound_hits;
        size_t psd_fixed_window_checks;
        double psd_propagation_milliseconds;
        double psd_higher_order_milliseconds;
        double cone_collapse_milliseconds;
        size_t fixed_cone_variables;
        size_t collapsed_cones;
        size_t reduced_rsoc_faces;
        size_t reduced_psd_faces;
        size_t reduced_exponential_faces;
        size_t reduced_power_faces;
        size_t removed_redundant_rows;
        size_t removed_singleton_rows;
        size_t removed_empty_rows;
        size_t parallel_row_budget_skips;
        double parallel_row_detection_milliseconds;
        size_t materialized_propagated_box_bounds;
        size_t suppressed_propagated_box_bounds;
        size_t aggregated_affine_cone_coordinates;
        size_t generated_affine_cone_blocks;
        size_t affine_cones_processed;
        size_t affine_cone_propagation_rounds;
        size_t affine_psd_budget_skips;
        size_t affine_psd_coordinates_skipped;
        size_t tightened_affine_cone_envelopes;
        size_t tightened_affine_variable_envelopes;
        size_t materialized_affine_cone_box_bounds;
        size_t suppressed_affine_cone_box_bounds;
        size_t reduced_affine_rsoc_faces;
        size_t reduced_affine_psd_faces;
        size_t reduced_affine_exponential_faces;
        size_t reduced_affine_power_faces;
        size_t removed_affine_cone_coordinates;
        size_t removed_affine_cone_blocks;
        double affine_face_reduction_milliseconds;
        size_t affine_psd_blocks_analyzed;
        size_t affine_psd_active_diagonal_coordinates;
        size_t affine_psd_active_offdiagonal_coordinates;
        size_t affine_psd_coefficient_columns;
        size_t affine_psd_diagonal_coefficient_columns;
        size_t affine_psd_single_diagonal_coefficient_columns;
        size_t affine_psd_connected_components;
        size_t affine_psd_largest_component_order;
        size_t affine_psd_splittable_blocks;
        size_t decomposed_affine_psd_blocks;
        size_t affine_psd_scalar_components;
        size_t affine_psd_component_blocks;
        size_t removed_affine_psd_cross_coordinates;
        double affine_psd_structure_milliseconds;
        size_t derived_affine_face_equalities;
        size_t fixed_affine_face_variables;
        size_t substituted_affine_face_variables;
        double affine_face_substitution_milliseconds;
        size_t removed_empty_columns;
        size_t removed_singleton_columns;
        size_t tightened_singleton_rows;
        size_t substituted_bounded_doubletons;
        size_t dual_fixed_columns;
        size_t merged_parallel_columns;
        size_t removed_redundant_row_lower_sides;
        size_t removed_redundant_row_upper_sides;
        size_t removed_redundant_box_lower_bounds;
        size_t removed_redundant_box_upper_bounds;
        size_t structural_gpu_passes;
        size_t structural_gpu_fallbacks;
        double structural_reduction_milliseconds;
        size_t column_csc_gpu_builds;
        size_t column_csc_gpu_fallbacks;
        double column_csc_gpu_milliseconds;
        size_t singleton_column_gpu_passes;
        size_t singleton_column_gpu_fallbacks;
        size_t singleton_column_gpu_candidates;
        double singleton_column_gpu_milliseconds;
        size_t parallel_column_gpu_passes;
        size_t parallel_column_gpu_fallbacks;
        double parallel_column_gpu_milliseconds;
        size_t parallel_row_gpu_passes;
        size_t parallel_row_gpu_fallbacks;
        size_t cone_activity_gpu_passes;
        size_t cone_activity_gpu_fallbacks;
        size_t cone_activity_gpu_candidates;
        size_t cone_gpu_rounds;
        size_t cone_gpu_linear_rounds;
        size_t cone_gpu_fallbacks;
        double cuda_workspace_setup_milliseconds;
        double parallel_row_gpu_milliseconds;
        double cone_activity_gpu_milliseconds;
        double cone_gpu_milliseconds;
        double cone_gpu_linear_transfer_milliseconds;
        double cone_gpu_linear_kernel_milliseconds;
        size_t affine_cone_gpu_rounds;
        size_t affine_cone_gpu_fallbacks;
        double affine_cone_gpu_milliseconds;
        size_t matrix_compaction_gpu_passes;
        size_t matrix_compaction_gpu_fallbacks;
        double matrix_compaction_gpu_milliseconds;
        double initialization_milliseconds;
        double affine_aggregation_milliseconds;
        double fast_fixed_point_milliseconds;
        double free_column_substitution_milliseconds;
        double trivial_row_reduction_milliseconds;
        double medium_fixed_point_milliseconds;
        double parallel_column_reduction_milliseconds;
        double matrix_compaction_milliseconds;
        double quadratic_compaction_milliseconds;
        double factor_compaction_milliseconds;
        double domain_compaction_milliseconds;
        double objective_compaction_milliseconds;
        double presolve_total_milliseconds;
        size_t fast_fixed_point_passes;
        size_t fast_fixed_point_rounds;
        size_t medium_fixed_point_rounds;
        size_t residual_row_substitutions;
    } PreFOSStats;

    typedef struct
    {
        double original_objective;
        double reduced_objective;
        double objective_absolute_error;

        double original_row_violation;
        double original_box_violation;
        double original_cone_violation;

        double reduced_row_violation;
        double reduced_box_violation;
        double reduced_cone_violation;

        int passed;
    } PreFOSPrimalVerification;

    typedef struct
    {
        double row_primal_violation;
        double box_primal_violation;
        double cone_primal_violation;
        double stationarity_violation;
        double row_dual_violation;
        double domain_dual_violation;
        double complementarity_violation;
        int passed;
    } PreFOSKKTResiduals;

    typedef struct
    {
        PreFOSKKTResiduals reduced;
        PreFOSKKTResiduals original;
        double objective_absolute_error;
        int passed;
    } PreFOSPostsolveKKTVerification;

    typedef enum
    {
        PREFOS_FACE_RSOC_U_ZERO = 0,
        PREFOS_FACE_RSOC_V_ZERO,
        PREFOS_FACE_PSD_ZERO_DIAGONALS,
        PREFOS_FACE_EXPONENTIAL_Y_ZERO,
        PREFOS_FACE_EXPONENTIAL_Z_ZERO,
        PREFOS_FACE_POWER_X_ZERO,
        PREFOS_FACE_POWER_Y_ZERO,
        PREFOS_FACE_POWER_Z_ZERO
    } PreFOSFacialReductionType;

    /*
     * Read-only structural proof that a coupled cone was replaced by a face.
     * RSOC, exponential, and power certificates use the singular source/axis
     * fields. PSD certificates set those fields to -1 and use
     * removed_matrix_indices/source_rows, whose entries are aligned. All
     * pointers remain valid until the next presolve run or until the presolver
     * is freed.
     */
    typedef struct
    {
        PreFOSFacialReductionType type;
        int source_row;
        int zero_axis_column;
        int surviving_axis_column;
        size_t cone_dimension;
        const int *cone_indices;
        size_t matrix_order;
        size_t n_removed_matrix_indices;
        const int *removed_matrix_indices;
        const int *source_rows;
    } PreFOSFacialReductionCertificate;

    /* Arrays in this struct are owned by the presolver. */
    typedef struct
    {
        size_t n;

        PreFOSCsrMatrix A;
        double *constraint_lower;
        double *constraint_upper;

        PreFOSCsrMatrix Q;
        PreFOSQStorage q_storage;
        PreFOSCsrMatrix R;
        double *D;

        double *c;
        double objective_offset;

        size_t n_box;
        int *box_indices;
        double *box_lower;
        double *box_upper;

        size_t n_cones;
        PreFOSConeBlock *cones;

        PreFOSCsrMatrix affine_cone_matrix;
        double *affine_cone_offset;
        size_t n_affine_cones;
        PreFOSAffineConeBlock *affine_cones;
    } PreFOSPresolvedProblem;

    typedef struct PreFOSPresolver PreFOSPresolver;

    PREFOS_API PreFOSSettings prefos_default_settings(void);

    /* Exact-bound settings for transformation equivalence tests. */
    PREFOS_API PreFOSSettings prefos_strict_settings(void);

    /* Initialize the CUDA primary context and allocation pool. Returns one
     * when CUDA is ready and zero in CPU-only or unavailable environments. */
    PREFOS_API int prefos_gpu_warmup(void);
    /* Start warmup on a background host thread. Wait before interpreting a
     * zero ready result as failure. */
    PREFOS_API int prefos_gpu_warmup_async(void);
    PREFOS_API int prefos_gpu_warmup_ready(void);
    PREFOS_API int prefos_gpu_warmup_wait(void);
    /* Release cached device allocations when no presolve call is active. */
    PREFOS_API void prefos_gpu_release_cache(void);

    PREFOS_API PreFOSStatus prefos_create_presolver(const PreFOSProblemData *problem,
                                                    const PreFOSSettings *settings,
                                                    PreFOSPresolver **presolver);
    PREFOS_API void prefos_free_presolver(PreFOSPresolver *presolver);

    PREFOS_API PreFOSStatus prefos_run_presolve(PreFOSPresolver *presolver);

    PREFOS_API const PreFOSPresolvedProblem *
    prefos_get_reduced_problem(const PreFOSPresolver *presolver);
    PREFOS_API const PreFOSStats *prefos_get_stats(const PreFOSPresolver *presolver);
    PREFOS_API const PreFOSPSDStructureAnalysis *
    prefos_get_psd_structure_analyses(const PreFOSPresolver *presolver,
                                      size_t *count);
    PREFOS_API const PreFOSFacialReductionCertificate *
    prefos_get_facial_reductions(const PreFOSPresolver *presolver, size_t *count);

    /* Recover the original primal vector from a reduced primal vector. */
    PREFOS_API PreFOSStatus prefos_postsolve_primal(const PreFOSPresolver *presolver,
                                                    const double *reduced_x,
                                                    double *original_x);

    /*
     * Dual convention: y is the normal to [constraint_lower,
     * constraint_upper], and z is the normal to the variable domain. Thus
     * stationarity is grad(f) + A^T y + z = 0. Cone entries of -z lie in the
     * corresponding dual cone. This API returns
     * PREFOS_STATUS_DUAL_RECOVERY_UNAVAILABLE when the original or reduced model
     * has affine cone blocks, whose row duals require a separate vector, or
     * when an enabled structural reduction has only primal recovery support.
     */
    PREFOS_API PreFOSStatus prefos_postsolve_primal_dual(
        const PreFOSPresolver *presolver, const double *reduced_x,
        const double *reduced_y, const double *reduced_z, double tolerance,
        double *original_x, double *original_y, double *original_z);

    /*
     * Recover an extended dual after facial reduction. On an exposed RSOC,
     * PSD, exponential, or power face, z is a normal to that reduced domain
     * rather than necessarily a normal to the unreduced cone. The structural
     * proofs are returned by prefos_get_facial_reductions().
     */
    PREFOS_API PreFOSStatus prefos_postsolve_extended_dual(
        const PreFOSPresolver *presolver, const double *reduced_x,
        const double *reduced_y, const double *reduced_z, double tolerance,
        double *original_x, double *original_y, double *original_z);

    /*
     * Full conic dual recovery. affine_z is the coordinate-space normal for
     * G*x+h in K, so stationarity is
     * grad(f) + A^T*y + z + G^T*affine_z = 0 and -affine_z lies in K^*.
     * Its length is affine_cone_matrix.rows in the corresponding model.
     */
    PREFOS_API PreFOSStatus prefos_postsolve_full_primal_dual(
        const PreFOSPresolver *presolver, const double *reduced_x,
        const double *reduced_y, const double *reduced_z,
        const double *reduced_affine_z, double tolerance, double *original_x,
        double *original_y, double *original_z, double *original_affine_z);

    PREFOS_API PreFOSStatus prefos_postsolve_full_extended_dual(
        const PreFOSPresolver *presolver, const double *reduced_x,
        const double *reduced_y, const double *reduced_z,
        const double *reduced_affine_z, double tolerance, double *original_x,
        double *original_y, double *original_z, double *original_affine_z);

    /*
     * Audit a reduced primal point after postsolve. This checks objective
     * identity and primal feasibility in both the reduced and original model.
     * It is intended as a debug/correctness gate, not a solver-side hot path.
     */
    PREFOS_API PreFOSStatus prefos_verify_postsolve_primal(
        const PreFOSPresolver *presolver, const double *reduced_x, double tolerance,
        PreFOSPrimalVerification *verification);

    PREFOS_API PreFOSStatus prefos_verify_postsolve_kkt(
        const PreFOSPresolver *presolver, const double *reduced_x,
        const double *reduced_y, const double *reduced_z, double tolerance,
        PreFOSPostsolveKKTVerification *verification);

    PREFOS_API PreFOSStatus prefos_verify_postsolve_extended_kkt(
        const PreFOSPresolver *presolver, const double *reduced_x,
        const double *reduced_y, const double *reduced_z, double tolerance,
        PreFOSPostsolveKKTVerification *verification);

    PREFOS_API PreFOSStatus prefos_verify_postsolve_full_kkt(
        const PreFOSPresolver *presolver, const double *reduced_x,
        const double *reduced_y, const double *reduced_z,
        const double *reduced_affine_z, double tolerance,
        PreFOSPostsolveKKTVerification *verification);

    PREFOS_API PreFOSStatus prefos_verify_postsolve_full_extended_kkt(
        const PreFOSPresolver *presolver, const double *reduced_x,
        const double *reduced_y, const double *reduced_z,
        const double *reduced_affine_z, double tolerance,
        PreFOSPostsolveKKTVerification *verification);

    PREFOS_API const char *prefos_status_string(PreFOSStatus status);

#ifdef __cplusplus
}
#endif

#endif /* PREFOS_H */
