/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <linear_programming/pdlp.cuh>
#include <linear_programming/pdlp_constants.hpp>
#include <linear_programming/solve.cuh>
#include <linear_programming/utils.cuh>
#include <mps_parser.hpp>
#include "utilities/pdlp_test_utilities.cuh"

#include <utilities/base_fixture.hpp>
#include <utilities/common_utils.hpp>

#include <cuopt/linear_programming/constants.h>
#include <cuopt/linear_programming/pdlp/pdlp_hyper_params.cuh>
#include <cuopt/linear_programming/pdlp/solver_settings.hpp>
#include <cuopt/linear_programming/pdlp/solver_solution.hpp>
#include <cuopt/linear_programming/solve.hpp>
#include <mip/problem/problem.cuh>
#include <mps_parser/parser.hpp>

#include <utilities/copy_helpers.hpp>
#include <utilities/error.hpp>

// Papilo includes
#include <papilo/core/Presolve.hpp>
#include <papilo/core/PresolveMethod.hpp>
#include <papilo/io/MpsParser.hpp>

#include <raft/sparse/detail/cusparse_macros.h>
#include <raft/sparse/detail/cusparse_wrappers.h>
#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>

#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/logical.h>

#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <sstream>
#include <vector>

namespace cuopt::linear_programming::test {

constexpr double afiro_primal_objective = -464;

// Accept a 1% error
static bool is_incorrect_objective(double reference, double objective)
{
  if (reference == 0) { return std::abs(objective) > 0.01; }
  if (objective == 0) { return std::abs(reference) > 0.01; }
  return std::abs((reference - objective) / reference) > 0.01;
}

// Function to compare cuopt and papilo models
void compare_models(const cuopt::mps_parser::mps_data_model_t<int, double>& cuopt_model,
                    const papilo::Problem<double>& papilo_model,
                    const std::string& instance_name)
{
  std::cout << "\n=== Comparing models for instance: " << instance_name << " ===" << std::endl;

  // Problem dimensions
  std::cout << "\n--- Problem Dimensions ---" << std::endl;
  std::cout << "CuOpt  - Variables: " << cuopt_model.get_n_variables()
            << ", Constraints: " << cuopt_model.get_n_constraints()
            << ", NNZ: " << cuopt_model.get_nnz() << std::endl;
  std::cout << "Papilo - Variables: " << papilo_model.getNCols()
            << ", Constraints: " << papilo_model.getNRows() << std::endl;

  // Problem name
  std::cout << "\n--- Problem Names ---" << std::endl;
  std::cout << "CuOpt problem name: '" << cuopt_model.get_problem_name() << "'" << std::endl;
  std::cout << "Papilo problem name: '" << papilo_model.getName() << "'" << std::endl;

  // Objective properties
  std::cout << "\n--- Objective Properties ---" << std::endl;
  std::cout << "CuOpt  - Maximize: " << cuopt_model.get_sense()
            << ", Offset: " << cuopt_model.get_objective_offset()
            << ", Scaling: " << cuopt_model.get_objective_scaling_factor() << std::endl;
  std::cout << "Papilo - Offset: " << papilo_model.getObjective().offset << std::endl;

  // Print some vectors using cuopt::print
  std::cout << "\n--- Objective Coefficients (first 10) ---" << std::endl;
  auto cuopt_obj  = cuopt_model.get_objective_coefficients();
  auto papilo_obj = papilo_model.getObjective().coefficients;
  if (!cuopt_obj.empty()) {
    std::vector<double> cuopt_obj_subset(cuopt_obj.begin(),
                                         cuopt_obj.begin() + std::min(10ul, cuopt_obj.size()));
    cuopt::print("CuOpt objective coefficients", cuopt_obj_subset);
  }
  if (!papilo_obj.empty()) {
    std::vector<double> papilo_obj_subset(papilo_obj.begin(),
                                          papilo_obj.begin() + std::min(10ul, papilo_obj.size()));
    cuopt::print("Papilo objective coefficients", papilo_obj_subset);
  }

  // Variable bounds
  std::cout << "\n--- Variable Bounds (first 10) ---" << std::endl;
  auto cuopt_lb  = cuopt_model.get_variable_lower_bounds();
  auto cuopt_ub  = cuopt_model.get_variable_upper_bounds();
  auto papilo_lb = papilo_model.getLowerBounds();
  auto papilo_ub = papilo_model.getUpperBounds();

  if (!cuopt_lb.empty()) {
    std::vector<double> cuopt_lb_subset(cuopt_lb.begin(),
                                        cuopt_lb.begin() + std::min(10ul, cuopt_lb.size()));
    cuopt::print("CuOpt variable lower bounds", cuopt_lb_subset);
  }
  if (!cuopt_ub.empty()) {
    std::vector<double> cuopt_ub_subset(cuopt_ub.begin(),
                                        cuopt_ub.begin() + std::min(10ul, cuopt_ub.size()));
    cuopt::print("CuOpt variable upper bounds", cuopt_ub_subset);
  }
  if (!papilo_lb.empty()) {
    std::vector<double> papilo_lb_subset(papilo_lb.begin(),
                                         papilo_lb.begin() + std::min(10ul, papilo_lb.size()));
    cuopt::print("Papilo variable lower bounds", papilo_lb_subset);
  }
  if (!papilo_ub.empty()) {
    std::vector<double> papilo_ub_subset(papilo_ub.begin(),
                                         papilo_ub.begin() + std::min(10ul, papilo_ub.size()));
    cuopt::print("Papilo variable upper bounds", papilo_ub_subset);
  }

  // Constraint bounds
  std::cout << "\n--- Constraint Bounds (first 10) ---" << std::endl;
  auto cuopt_clb          = cuopt_model.get_constraint_lower_bounds();
  auto cuopt_cub          = cuopt_model.get_constraint_upper_bounds();
  auto& constraint_matrix = papilo_model.getConstraintMatrix();
  auto papilo_clb         = constraint_matrix.getLeftHandSides();
  auto papilo_crb         = constraint_matrix.getRightHandSides();

  if (!cuopt_clb.empty()) {
    std::vector<double> cuopt_clb_subset(cuopt_clb.begin(),
                                         cuopt_clb.begin() + std::min(10ul, cuopt_clb.size()));
    cuopt::print("CuOpt constraint lower bounds", cuopt_clb_subset);
  }
  if (!cuopt_cub.empty()) {
    std::vector<double> cuopt_cub_subset(cuopt_cub.begin(),
                                         cuopt_cub.begin() + std::min(10ul, cuopt_cub.size()));
    cuopt::print("CuOpt constraint upper bounds", cuopt_cub_subset);
  }
  if (!papilo_clb.empty()) {
    std::vector<double> papilo_clb_subset(papilo_clb.begin(),
                                          papilo_clb.begin() + std::min(10ul, papilo_clb.size()));
    cuopt::print("Papilo constraint left hand sides", papilo_clb_subset);
  }
  if (!papilo_crb.empty()) {
    std::vector<double> papilo_crb_subset(papilo_crb.begin(),
                                          papilo_crb.begin() + std::min(10ul, papilo_crb.size()));
    cuopt::print("Papilo constraint right hand sides", papilo_crb_subset);
  }

  // Constraint matrix values
  std::cout << "\n--- Constraint Matrix Values (first 10) ---" << std::endl;
  auto cuopt_A         = cuopt_model.get_constraint_matrix_values();
  auto cuopt_A_indices = cuopt_model.get_constraint_matrix_indices();
  auto cuopt_A_offsets = cuopt_model.get_constraint_matrix_offsets();

  if (!cuopt_A.empty()) {
    std::vector<double> cuopt_A_subset(cuopt_A.begin(),
                                       cuopt_A.begin() + std::min(10ul, cuopt_A.size()));
    cuopt::print("CuOpt matrix values", cuopt_A_subset);
  }
  if (!cuopt_A_indices.empty()) {
    std::vector<int> cuopt_indices_subset(
      cuopt_A_indices.begin(), cuopt_A_indices.begin() + std::min(10ul, cuopt_A_indices.size()));
    cuopt::print("CuOpt matrix indices", cuopt_indices_subset);
  }
  if (!cuopt_A_offsets.empty()) {
    std::vector<int> cuopt_offsets_subset(
      cuopt_A_offsets.begin(), cuopt_A_offsets.begin() + std::min(10ul, cuopt_A_offsets.size()));
    cuopt::print("CuOpt matrix offsets", cuopt_offsets_subset);
  }

  // Variable names
  std::cout << "\n--- Variable Names (first 5) ---" << std::endl;
  auto cuopt_var_names  = cuopt_model.get_variable_names();
  auto papilo_var_names = papilo_model.getVariableNames();

  std::cout << "CuOpt variable names: ";
  for (size_t i = 0; i < std::min(5ul, cuopt_var_names.size()); ++i) {
    std::cout << "'" << cuopt_var_names[i] << "' ";
  }
  std::cout << std::endl;

  std::cout << "Papilo variable names: ";
  for (size_t i = 0; i < std::min(5ul, papilo_var_names.size()); ++i) {
    std::cout << "'" << papilo_var_names[i] << "' ";
  }
  std::cout << std::endl;

  // Row names
  std::cout << "\n--- Row Names (first 5) ---" << std::endl;
  auto cuopt_row_names  = cuopt_model.get_row_names();
  auto papilo_row_names = papilo_model.getConstraintNames();

  std::cout << "CuOpt row names: ";
  for (size_t i = 0; i < std::min(5ul, cuopt_row_names.size()); ++i) {
    std::cout << "'" << cuopt_row_names[i] << "' ";
  }
  std::cout << std::endl;

  std::cout << "Papilo row names: ";
  for (size_t i = 0; i < std::min(5ul, papilo_row_names.size()); ++i) {
    std::cout << "'" << papilo_row_names[i] << "' ";
  }
  std::cout << std::endl;

  std::cout << "\n=== End comparison for " << instance_name << " ===\n" << std::endl;
}

// TEST(pdlp_class, run_double)
// {
//   const raft::handle_t handle_{};

//   auto path = make_path_absolute("linear_programming/afiro_original.mps");
//   cuopt::mps_parser::mps_data_model_t<int, double> op_problem =
//     cuopt::mps_parser::parse_mps<int, double>(path, true);

//   auto solver_settings   = pdlp_solver_settings_t<int, double>{};
//   solver_settings.method = cuopt::linear_programming::method_t::PDLP;

//   optimization_problem_solution_t<int, double> solution =
//     solve_lp(&handle_, op_problem, solver_settings);
//   EXPECT_EQ((int)solution.get_termination_status(), CUOPT_TERIMINATION_STATUS_OPTIMAL);
//   EXPECT_FALSE(is_incorrect_objective(
//     afiro_primal_objective, solution.get_additional_termination_information().primal_objective));
// }

// TEST(pdlp_class, run_double_very_low_accuracy)
// {
//   const raft::handle_t handle_{};

//   auto path = make_path_absolute("linear_programming/afiro_original.mps");
//   cuopt::mps_parser::mps_data_model_t<int, double> op_problem =
//     cuopt::mps_parser::parse_mps<int, double>(path, true);

//   cuopt::linear_programming::pdlp_solver_settings_t<int, double> settings =
//     cuopt::linear_programming::pdlp_solver_settings_t<int, double>{};
//   // With all 0 afiro with return an error
//   // Setting absolute tolerance to the minimal value of 1e-12 will make it work
//   settings.tolerances.absolute_dual_tolerance   = settings.minimal_absolute_tolerance;
//   settings.tolerances.relative_dual_tolerance   = 0.0;
//   settings.tolerances.absolute_primal_tolerance = settings.minimal_absolute_tolerance;
//   settings.tolerances.relative_primal_tolerance = 0.0;
//   settings.tolerances.absolute_gap_tolerance    = settings.minimal_absolute_tolerance;
//   settings.tolerances.relative_gap_tolerance    = 0.0;
//   settings.method                               = cuopt::linear_programming::method_t::PDLP;

//   optimization_problem_solution_t<int, double> solution = solve_lp(&handle_, op_problem,
//   settings); EXPECT_EQ((int)solution.get_termination_status(),
//   CUOPT_TERIMINATION_STATUS_OPTIMAL); EXPECT_FALSE(is_incorrect_objective(
//     afiro_primal_objective, solution.get_additional_termination_information().primal_objective));
// }

// TEST(pdlp_class, run_double_initial_solution)
// {
//   const raft::handle_t handle_{};

//   auto path = make_path_absolute("linear_programming/afiro_original.mps");
//   cuopt::mps_parser::mps_data_model_t<int, double> op_problem =
//     cuopt::mps_parser::parse_mps<int, double>(path, true);

//   std::vector<double> inital_primal_sol(op_problem.get_n_variables());
//   std::fill(inital_primal_sol.begin(), inital_primal_sol.end(), 1.0);
//   op_problem.set_initial_primal_solution(inital_primal_sol.data(), inital_primal_sol.size());

//   auto solver_settings   = pdlp_solver_settings_t<int, double>{};
//   solver_settings.method = cuopt::linear_programming::method_t::PDLP;

//   optimization_problem_solution_t<int, double> solution =
//     solve_lp(&handle_, op_problem, solver_settings);
//   EXPECT_EQ((int)solution.get_termination_status(), CUOPT_TERIMINATION_STATUS_OPTIMAL);
//   EXPECT_FALSE(is_incorrect_objective(
//     afiro_primal_objective, solution.get_additional_termination_information().primal_objective));
// }

// TEST(pdlp_class, run_iteration_limit)
// {
//   const raft::handle_t handle_{};

//   auto path = make_path_absolute("linear_programming/afiro_original.mps");
//   cuopt::mps_parser::mps_data_model_t<int, double> op_problem =
//     cuopt::mps_parser::parse_mps<int, double>(path, true);

//   cuopt::linear_programming::pdlp_solver_settings_t<int, double> settings =
//     cuopt::linear_programming::pdlp_solver_settings_t<int, double>{};

//   settings.iteration_limit = 10;
//   // To make sure it doesn't return before the iteration limit
//   settings.set_optimality_tolerance(0);
//   settings.method = cuopt::linear_programming::method_t::PDLP;

//   optimization_problem_solution_t<int, double> solution = solve_lp(&handle_, op_problem,
//   settings); EXPECT_EQ((int)solution.get_termination_status(),
//   CUOPT_TERIMINATION_STATUS_ITERATION_LIMIT);
//   // By default we would return all 0, we now return what we currently have so not all 0
//   EXPECT_FALSE(thrust::all_of(handle_.get_thrust_policy(),
//                               solution.get_primal_solution().begin(),
//                               solution.get_primal_solution().end(),
//                               thrust::placeholders::_1 == 0.0));
// }

// TEST(pdlp_class, run_time_limit)
// {
//   const raft::handle_t handle_{};
//   auto path = make_path_absolute("linear_programming/savsched1/savsched1.mps");
//   cuopt::mps_parser::mps_data_model_t<int, double> op_problem =
//     cuopt::mps_parser::parse_mps<int, double>(path);

//   cuopt::linear_programming::pdlp_solver_settings_t<int, double> settings =
//     cuopt::linear_programming::pdlp_solver_settings_t<int, double>{};

//   // 200 ms
//   constexpr double time_limit_seconds = 0.2;
//   settings.time_limit                 = time_limit_seconds;
//   // To make sure it doesn't return before the time limit
//   settings.set_optimality_tolerance(0);
//   settings.method = cuopt::linear_programming::method_t::PDLP;

//   optimization_problem_solution_t<int, double> solution = solve_lp(&handle_, op_problem,
//   settings);

//   EXPECT_EQ((int)solution.get_termination_status(), CUOPT_TERIMINATION_STATUS_TIME_LIMIT);
//   // By default we would return all 0, we now return what we currently have so not all 0
//   EXPECT_FALSE(thrust::all_of(handle_.get_thrust_policy(),
//                               solution.get_primal_solution().begin(),
//                               solution.get_primal_solution().end(),
//                               thrust::placeholders::_1 == 0.0));
//   // Check that indeed it didn't run for more than x time
//   EXPECT_TRUE(solution.get_additional_termination_information().solve_time <
//               (time_limit_seconds * 5) * 1000);
// }

TEST(pdlp_class, run_sub_mittleman)
{
  std::vector<std::pair<std::string,  // Instance name
                        double>>      // Expected objective value
    instances{
      // {"graph40-40", -300.0},
      //{"ex10", 100.0003411893773},
      {"datt256_lp", 255.9992298290425},
      // {"woodlands09", 0.0},
      // {"savsched1", 217.4054085795689},
      // {"nug08-3rd", 214.0141488989151},
      // {"qap15", 1040.999546647414},
      // {"scpm1", 413.7787723060584},
      // {"neos3", 27773.54059633068},
      // {"a2864", -282.9962521965164}
    };

  for (const auto& entry : instances) {
    const auto& name                    = entry.first;
    const auto expected_objective_value = entry.second;

    std::cout << "Running " << name << std::endl;
    auto path = make_path_absolute("linear_programming/" + name + "/" + name + ".mps");

    // Parse with CuOpt
    cuopt::mps_parser::mps_data_model_t<int, double> cuopt_op_problem =
      cuopt::mps_parser::parse_mps<int, double>(path);

    // Parse with Papilo
    auto papilo_problem_opt = papilo::MpsParser<double>::loadProblem(path);
    if (!papilo_problem_opt) {
      std::cout << "Failed to parse " << name << " with Papilo parser" << std::endl;
      continue;
    }
    auto papilo_problem = papilo_problem_opt.value();

    // Presolve the Papilo problem using cuOpt's configuration
    std::cout << "Original Papilo problem - Variables: " << papilo_problem.getNCols()
              << ", Constraints: " << papilo_problem.getNRows() << std::endl;

    papilo::Presolve<double> presolve;

    // Add cuOpt's specific presolve methods (same as in third_party_presolve.cu)
    using uptr = std::unique_ptr<papilo::PresolveMethod<double>>;

    // fast presolvers
    presolve.addPresolveMethod(uptr(new papilo::SingletonCols<double>()));
    presolve.addPresolveMethod(uptr(new papilo::CoefficientStrengthening<double>()));
    presolve.addPresolveMethod(uptr(new papilo::ConstraintPropagation<double>()));

    // medium presolvers
    presolve.addPresolveMethod(uptr(new papilo::FixContinuous<double>()));
    presolve.addPresolveMethod(uptr(new papilo::SimpleProbing<double>()));
    presolve.addPresolveMethod(uptr(new papilo::ParallelRowDetection<double>()));
    presolve.addPresolveMethod(uptr(new papilo::ParallelColDetection<double>()));
    // Note: SingletonStuffing excluded due to postsolve issues in cuOpt
    presolve.addPresolveMethod(uptr(new papilo::DualFix<double>()));
    presolve.addPresolveMethod(uptr(new papilo::SimplifyInequalities<double>()));

    // exhaustive presolvers
    presolve.addPresolveMethod(uptr(new papilo::ImplIntDetection<double>()));
    presolve.addPresolveMethod(uptr(new papilo::DominatedCols<double>()));
    presolve.addPresolveMethod(uptr(new papilo::Probing<double>()));

    // Set cuOpt's presolve options for LP problems
    constexpr double absolute_tolerance            = 1e-4;  // typical LP tolerance
    constexpr double time_limit                    = 10.0;  // 10 seconds default
    presolve.getPresolveOptions().tlim             = time_limit;
    presolve.getPresolveOptions().epsilon          = absolute_tolerance;
    presolve.getPresolveOptions().feastol          = absolute_tolerance;
    presolve.getPresolveOptions().componentsmaxint = -1;  // for LP problems
    presolve.getPresolveOptions().detectlindep     = 0;   // for LP problems

    papilo::PresolveResult<double> presolve_result = presolve.apply(papilo_problem);

    std::cout << "Presolve status: ";
    switch (presolve_result.status) {
      case papilo::PresolveStatus::kUnchanged: std::cout << "Unchanged"; break;
      case papilo::PresolveStatus::kReduced: std::cout << "Reduced"; break;
      case papilo::PresolveStatus::kUnbndOrInfeas: std::cout << "Unbounded or Infeasible"; break;
      case papilo::PresolveStatus::kUnbounded: std::cout << "Unbounded"; break;
      case papilo::PresolveStatus::kInfeasible: std::cout << "Infeasible"; break;
    }
    std::cout << std::endl;

    std::cout << "Presolved Papilo problem - Variables: " << papilo_problem.getNCols()
              << ", Constraints: " << papilo_problem.getNRows() << std::endl;

    // Check if presolving detected infeasibility/unboundedness
    if (presolve_result.status == papilo::PresolveStatus::kInfeasible ||
        presolve_result.status == papilo::PresolveStatus::kUnbounded ||
        presolve_result.status == papilo::PresolveStatus::kUnbndOrInfeas) {
      std::cout << "Skipping " << name << " due to presolve status" << std::endl;
      continue;
    }

    // Compare the two models
    compare_models(cuopt_op_problem, papilo_problem, name);

    // Testing for each solver_mode is ok as it's parsing that is the bottleneck here, not
    // solving
    auto solver_mode_list = {
      cuopt::linear_programming::pdlp_solver_mode_t::Stable2,
      cuopt::linear_programming::pdlp_solver_mode_t::Methodical1,
      cuopt::linear_programming::pdlp_solver_mode_t::Fast1,
    };
    for (auto solver_mode : solver_mode_list) {
      auto settings             = pdlp_solver_settings_t<int, double>{};
      settings.pdlp_solver_mode = solver_mode;
      settings.method           = cuopt::linear_programming::method_t::PDLP;
      settings.presolve         = true;
      const raft::handle_t handle_{};
      optimization_problem_solution_t<int, double> solution =
        solve_lp(&handle_, cuopt_op_problem, settings);
      EXPECT_EQ((int)solution.get_termination_status(), CUOPT_TERIMINATION_STATUS_OPTIMAL);
      EXPECT_FALSE(
        is_incorrect_objective(expected_objective_value,
                               solution.get_additional_termination_information().primal_objective));
      test_objective_sanity(cuopt_op_problem,
                            solution.get_primal_solution(),
                            solution.get_additional_termination_information().primal_objective);
      test_constraint_sanity(cuopt_op_problem, solution);
    }
  }
}

constexpr double initial_step_size_afiro     = 1.4893;
constexpr double initial_primal_weight_afiro = 0.0141652;
constexpr double factor_tolerance            = 1e-4f;

// Should be added to google test
#define EXPECT_NOT_NEAR(val1, val2, abs_error) \
  EXPECT_FALSE((std::abs((val1) - (val2)) <= (abs_error)))

TEST(pdlp_class, initial_solution_test)
{
  const raft::handle_t handle_{};

  auto path = make_path_absolute("linear_programming/afiro_original.mps");
  cuopt::mps_parser::mps_data_model_t<int, double> mps_data_model =
    cuopt::mps_parser::parse_mps<int, double>(path);

  auto op_problem = cuopt::linear_programming::mps_data_model_to_optimization_problem<int, double>(
    &handle_, mps_data_model);
  cuopt::linear_programming::detail::problem_t<int, double> problem(op_problem);

  auto solver_settings = pdlp_solver_settings_t<int, double>{};
  // We are just testing initial scaling on initial solution scheme so we don't care about solver
  solver_settings.iteration_limit = 0;
  solver_settings.method          = cuopt::linear_programming::method_t::PDLP;
  // Empty call solve to set the parameters and init the handler since calling pdlp object directly
  // doesn't
  solver_settings.pdlp_solver_mode = cuopt::linear_programming::pdlp_solver_mode_t::Methodical1;
  solve_lp(op_problem, solver_settings);
  EXPECT_EQ(cuopt::linear_programming::pdlp_hyper_params::initial_step_size_scaling, 1);
  EXPECT_EQ(cuopt::linear_programming::pdlp_hyper_params::default_l_inf_ruiz_iterations, 5);
  EXPECT_TRUE(cuopt::linear_programming::pdlp_hyper_params::do_pock_chambolle_scaling);
  EXPECT_TRUE(cuopt::linear_programming::pdlp_hyper_params::do_ruiz_scaling);
  EXPECT_EQ(cuopt::linear_programming::pdlp_hyper_params::default_alpha_pock_chambolle_rescaling,
            1.0);

  EXPECT_FALSE(cuopt::linear_programming::pdlp_hyper_params::update_step_size_on_initial_solution);
  EXPECT_FALSE(
    cuopt::linear_programming::pdlp_hyper_params::update_primal_weight_on_initial_solution);

  {
    cuopt::linear_programming::detail::pdlp_solver_t<int, double> solver(problem, solver_settings);
    auto start_solver = std::chrono::high_resolution_clock::now();
    solver.run_solver(start_solver);
    RAFT_CUDA_TRY(cudaStreamSynchronize(handle_.get_stream()));
    EXPECT_NEAR(initial_step_size_afiro, solver.get_step_size_h(), factor_tolerance);
    EXPECT_NEAR(initial_primal_weight_afiro, solver.get_primal_weight_h(), factor_tolerance);
  }

  // First add an initial primal then dual, then both, which shouldn't influence the values as the
  // scale on initial option is not toggled
  {
    cuopt::linear_programming::detail::pdlp_solver_t<int, double> solver(problem, solver_settings);
    auto start_solver = std::chrono::high_resolution_clock::now();
    std::vector<double> initial_primal(op_problem.get_n_variables(), 1);
    auto d_initial_primal = device_copy(initial_primal, handle_.get_stream());
    solver.set_initial_primal_solution(d_initial_primal);
    solver.run_solver(start_solver);
    RAFT_CUDA_TRY(cudaStreamSynchronize(handle_.get_stream()));
    EXPECT_NEAR(initial_step_size_afiro, solver.get_step_size_h(), factor_tolerance);
    EXPECT_NEAR(initial_primal_weight_afiro, solver.get_primal_weight_h(), factor_tolerance);
  }
  {
    cuopt::linear_programming::detail::pdlp_solver_t<int, double> solver(problem, solver_settings);
    auto start_solver = std::chrono::high_resolution_clock::now();
    std::vector<double> initial_dual(op_problem.get_n_constraints(), 1);
    auto d_initial_dual = device_copy(initial_dual, handle_.get_stream());
    solver.set_initial_dual_solution(d_initial_dual);
    solver.run_solver(start_solver);
    RAFT_CUDA_TRY(cudaStreamSynchronize(handle_.get_stream()));
    EXPECT_NEAR(initial_step_size_afiro, solver.get_step_size_h(), factor_tolerance);
    EXPECT_NEAR(initial_primal_weight_afiro, solver.get_primal_weight_h(), factor_tolerance);
  }
  {
    cuopt::linear_programming::detail::pdlp_solver_t<int, double> solver(problem, solver_settings);
    auto start_solver = std::chrono::high_resolution_clock::now();
    std::vector<double> initial_primal(op_problem.get_n_variables(), 1);
    auto d_initial_primal = device_copy(initial_primal, handle_.get_stream());
    solver.set_initial_primal_solution(d_initial_primal);
    std::vector<double> initial_dual(op_problem.get_n_constraints(), 1);
    auto d_initial_dual = device_copy(initial_dual, handle_.get_stream());
    solver.set_initial_dual_solution(d_initial_dual);
    solver.run_solver(start_solver);
    RAFT_CUDA_TRY(cudaStreamSynchronize(handle_.get_stream()));
    EXPECT_NEAR(initial_step_size_afiro, solver.get_step_size_h(), factor_tolerance);
    EXPECT_NEAR(initial_primal_weight_afiro, solver.get_primal_weight_h(), factor_tolerance);
  }

  // Toggle the scale on initial solution while not providing should yield the same
  {
    cuopt::linear_programming::detail::pdlp_solver_t<int, double> solver(problem, solver_settings);
    auto start_solver = std::chrono::high_resolution_clock::now();
    cuopt::linear_programming::pdlp_hyper_params::update_step_size_on_initial_solution = true;
    solver.run_solver(start_solver);
    RAFT_CUDA_TRY(cudaStreamSynchronize(handle_.get_stream()));
    EXPECT_NEAR(initial_step_size_afiro, solver.get_step_size_h(), factor_tolerance);
    EXPECT_NEAR(initial_primal_weight_afiro, solver.get_primal_weight_h(), factor_tolerance);
    cuopt::linear_programming::pdlp_hyper_params::update_step_size_on_initial_solution = false;
  }
  {
    cuopt::linear_programming::detail::pdlp_solver_t<int, double> solver(problem, solver_settings);
    auto start_solver = std::chrono::high_resolution_clock::now();
    cuopt::linear_programming::pdlp_hyper_params::update_primal_weight_on_initial_solution = true;
    solver.run_solver(start_solver);
    RAFT_CUDA_TRY(cudaStreamSynchronize(handle_.get_stream()));
    EXPECT_NEAR(initial_step_size_afiro, solver.get_step_size_h(), factor_tolerance);
    EXPECT_NEAR(initial_primal_weight_afiro, solver.get_primal_weight_h(), factor_tolerance);
    cuopt::linear_programming::pdlp_hyper_params::update_primal_weight_on_initial_solution = false;
  }
  {
    cuopt::linear_programming::detail::pdlp_solver_t<int, double> solver(problem, solver_settings);
    auto start_solver = std::chrono::high_resolution_clock::now();
    cuopt::linear_programming::pdlp_hyper_params::update_primal_weight_on_initial_solution = true;
    cuopt::linear_programming::pdlp_hyper_params::update_step_size_on_initial_solution     = true;
    solver.run_solver(start_solver);
    RAFT_CUDA_TRY(cudaStreamSynchronize(handle_.get_stream()));
    EXPECT_NEAR(initial_step_size_afiro, solver.get_step_size_h(), factor_tolerance);
    EXPECT_NEAR(initial_primal_weight_afiro, solver.get_primal_weight_h(), factor_tolerance);
    cuopt::linear_programming::pdlp_hyper_params::update_primal_weight_on_initial_solution = false;
    cuopt::linear_programming::pdlp_hyper_params::update_step_size_on_initial_solution     = false;
  }

  // Asking for initial scaling on step size with initial solution being only primal or only dual
  // should not break but not modify the step size
  {
    cuopt::linear_programming::pdlp_hyper_params::update_step_size_on_initial_solution = true;
    cuopt::linear_programming::detail::pdlp_solver_t<int, double> solver(problem, solver_settings);
    auto start_solver = std::chrono::high_resolution_clock::now();
    std::vector<double> initial_primal(op_problem.get_n_variables(), 1);
    auto d_initial_primal = device_copy(initial_primal, handle_.get_stream());
    solver.set_initial_primal_solution(d_initial_primal);
    solver.run_solver(start_solver);
    RAFT_CUDA_TRY(cudaStreamSynchronize(handle_.get_stream()));
    EXPECT_NEAR(initial_step_size_afiro, solver.get_step_size_h(), factor_tolerance);
    EXPECT_NEAR(initial_primal_weight_afiro, solver.get_primal_weight_h(), factor_tolerance);
    cuopt::linear_programming::pdlp_hyper_params::update_step_size_on_initial_solution = false;
  }
  {
    cuopt::linear_programming::pdlp_hyper_params::update_step_size_on_initial_solution = true;
    cuopt::linear_programming::detail::pdlp_solver_t<int, double> solver(problem, solver_settings);
    auto start_solver = std::chrono::high_resolution_clock::now();
    std::vector<double> initial_dual(op_problem.get_n_constraints(), 1);
    auto d_initial_dual = device_copy(initial_dual, handle_.get_stream());
    solver.set_initial_dual_solution(d_initial_dual);
    solver.run_solver(start_solver);
    RAFT_CUDA_TRY(cudaStreamSynchronize(handle_.get_stream()));
    EXPECT_NEAR(initial_step_size_afiro, solver.get_step_size_h(), factor_tolerance);
    EXPECT_NEAR(initial_primal_weight_afiro, solver.get_primal_weight_h(), factor_tolerance);
    cuopt::linear_programming::pdlp_hyper_params::update_step_size_on_initial_solution = false;
  }

  // Asking for initial scaling on primal weight with initial solution being only primal or only
  // dual should *not* break but the primal weight should not change
  {
    cuopt::linear_programming::pdlp_hyper_params::update_primal_weight_on_initial_solution = true;
    cuopt::linear_programming::detail::pdlp_solver_t<int, double> solver(problem, solver_settings);
    auto start_solver = std::chrono::high_resolution_clock::now();
    std::vector<double> initial_primal(op_problem.get_n_variables(), 1);
    auto d_initial_primal = device_copy(initial_primal, handle_.get_stream());
    solver.set_initial_primal_solution(d_initial_primal);
    solver.run_solver(start_solver);
    EXPECT_NEAR(initial_step_size_afiro, solver.get_step_size_h(), factor_tolerance);
    EXPECT_NEAR(initial_primal_weight_afiro, solver.get_primal_weight_h(), factor_tolerance);
    cuopt::linear_programming::pdlp_hyper_params::update_primal_weight_on_initial_solution = false;
  }
  {
    cuopt::linear_programming::pdlp_hyper_params::update_primal_weight_on_initial_solution = true;
    cuopt::linear_programming::detail::pdlp_solver_t<int, double> solver(problem, solver_settings);
    auto start_solver = std::chrono::high_resolution_clock::now();
    std::vector<double> initial_dual(op_problem.get_n_constraints(), 1);
    auto d_initial_dual = device_copy(initial_dual, handle_.get_stream());
    solver.set_initial_dual_solution(d_initial_dual);
    solver.run_solver(start_solver);
    EXPECT_NEAR(initial_step_size_afiro, solver.get_step_size_h(), factor_tolerance);
    EXPECT_NEAR(initial_primal_weight_afiro, solver.get_primal_weight_h(), factor_tolerance);
    cuopt::linear_programming::pdlp_hyper_params::update_primal_weight_on_initial_solution = false;
  }

  // All 0 solution when given an initial primal and dual with scale on the step size should not
  // break but not change primal weight and step size
  {
    cuopt::linear_programming::pdlp_hyper_params::update_step_size_on_initial_solution = true;
    cuopt::linear_programming::detail::pdlp_solver_t<int, double> solver(problem, solver_settings);
    auto start_solver = std::chrono::high_resolution_clock::now();
    std::vector<double> initial_primal(op_problem.get_n_variables(), 0);
    auto d_initial_primal = device_copy(initial_primal, handle_.get_stream());
    solver.set_initial_primal_solution(d_initial_primal);
    std::vector<double> initial_dual(op_problem.get_n_constraints(), 0);
    auto d_initial_dual = device_copy(initial_dual, handle_.get_stream());
    solver.set_initial_dual_solution(d_initial_dual);
    solver.run_solver(start_solver);
    EXPECT_NEAR(initial_step_size_afiro, solver.get_step_size_h(), factor_tolerance);
    EXPECT_NEAR(initial_primal_weight_afiro, solver.get_primal_weight_h(), factor_tolerance);
    cuopt::linear_programming::pdlp_hyper_params::update_step_size_on_initial_solution = false;
  }

  // All 0 solution when given an initial primal and/or dual with scale on the primal weight is
  // *not* an error but should not change primal weight and step size
  {
    cuopt::linear_programming::pdlp_hyper_params::update_primal_weight_on_initial_solution = true;
    cuopt::linear_programming::detail::pdlp_solver_t<int, double> solver(problem, solver_settings);
    auto start_solver = std::chrono::high_resolution_clock::now();
    std::vector<double> initial_primal(op_problem.get_n_variables(), 0);
    auto d_initial_primal = device_copy(initial_primal, handle_.get_stream());
    solver.set_initial_primal_solution(d_initial_primal);
    solver.run_solver(start_solver);
    EXPECT_NEAR(initial_step_size_afiro, solver.get_step_size_h(), factor_tolerance);
    EXPECT_NEAR(initial_primal_weight_afiro, solver.get_primal_weight_h(), factor_tolerance);
    cuopt::linear_programming::pdlp_hyper_params::update_primal_weight_on_initial_solution = false;
  }
  {
    cuopt::linear_programming::pdlp_hyper_params::update_primal_weight_on_initial_solution = true;
    cuopt::linear_programming::detail::pdlp_solver_t<int, double> solver(problem, solver_settings);
    auto start_solver = std::chrono::high_resolution_clock::now();
    std::vector<double> initial_dual(op_problem.get_n_constraints(), 0);
    auto d_initial_dual = device_copy(initial_dual, handle_.get_stream());
    solver.set_initial_dual_solution(d_initial_dual);
    solver.run_solver(start_solver);
    EXPECT_NEAR(initial_step_size_afiro, solver.get_step_size_h(), factor_tolerance);
    EXPECT_NEAR(initial_primal_weight_afiro, solver.get_primal_weight_h(), factor_tolerance);
    cuopt::linear_programming::pdlp_hyper_params::update_primal_weight_on_initial_solution = false;
  }
  {
    cuopt::linear_programming::pdlp_hyper_params::update_primal_weight_on_initial_solution = true;
    cuopt::linear_programming::detail::pdlp_solver_t<int, double> solver(problem, solver_settings);
    auto start_solver = std::chrono::high_resolution_clock::now();
    std::vector<double> initial_primal(op_problem.get_n_variables(), 0);
    auto d_initial_primal = device_copy(initial_primal, handle_.get_stream());
    solver.set_initial_primal_solution(d_initial_primal);
    std::vector<double> initial_dual(op_problem.get_n_constraints(), 0);
    auto d_initial_dual = device_copy(initial_dual, handle_.get_stream());
    solver.set_initial_dual_solution(d_initial_dual);
    solver.run_solver(start_solver);
    EXPECT_NEAR(initial_step_size_afiro, solver.get_step_size_h(), factor_tolerance);
    EXPECT_NEAR(initial_primal_weight_afiro, solver.get_primal_weight_h(), factor_tolerance);
    cuopt::linear_programming::pdlp_hyper_params::update_primal_weight_on_initial_solution = false;
  }

  // A non-all-0 vector for both initial primal and dual set should trigger a modification in primal
  // weight and step size
  {
    cuopt::linear_programming::pdlp_hyper_params::update_primal_weight_on_initial_solution = true;
    cuopt::linear_programming::detail::pdlp_solver_t<int, double> solver(problem, solver_settings);
    auto start_solver = std::chrono::high_resolution_clock::now();
    std::vector<double> initial_primal(op_problem.get_n_variables(), 1);
    auto d_initial_primal = device_copy(initial_primal, handle_.get_stream());
    solver.set_initial_primal_solution(d_initial_primal);
    std::vector<double> initial_dual(op_problem.get_n_constraints(), 1);
    auto d_initial_dual = device_copy(initial_dual, handle_.get_stream());
    solver.set_initial_dual_solution(d_initial_dual);
    solver.run_solver(start_solver);
    EXPECT_NEAR(initial_step_size_afiro, solver.get_step_size_h(), factor_tolerance);
    EXPECT_NOT_NEAR(initial_primal_weight_afiro, solver.get_primal_weight_h(), factor_tolerance);
    cuopt::linear_programming::pdlp_hyper_params::update_primal_weight_on_initial_solution = false;
  }
  {
    cuopt::linear_programming::pdlp_hyper_params::update_step_size_on_initial_solution = true;
    cuopt::linear_programming::detail::pdlp_solver_t<int, double> solver(problem, solver_settings);
    auto start_solver = std::chrono::high_resolution_clock::now();
    std::vector<double> initial_primal(op_problem.get_n_variables(), 1);
    auto d_initial_primal = device_copy(initial_primal, handle_.get_stream());
    solver.set_initial_primal_solution(d_initial_primal);
    std::vector<double> initial_dual(op_problem.get_n_constraints(), 1);
    auto d_initial_dual = device_copy(initial_dual, handle_.get_stream());
    solver.set_initial_dual_solution(d_initial_dual);
    solver.run_solver(start_solver);
    EXPECT_NOT_NEAR(initial_step_size_afiro, solver.get_step_size_h(), factor_tolerance);
    EXPECT_NEAR(initial_primal_weight_afiro, solver.get_primal_weight_h(), factor_tolerance);
    cuopt::linear_programming::pdlp_hyper_params::update_step_size_on_initial_solution = false;
  }
  {
    cuopt::linear_programming::pdlp_hyper_params::update_primal_weight_on_initial_solution = true;
    cuopt::linear_programming::pdlp_hyper_params::update_step_size_on_initial_solution     = true;
    cuopt::linear_programming::detail::pdlp_solver_t<int, double> solver(problem, solver_settings);
    auto start_solver = std::chrono::high_resolution_clock::now();
    std::vector<double> initial_primal(op_problem.get_n_variables(), 1);
    auto d_initial_primal = device_copy(initial_primal, handle_.get_stream());
    solver.set_initial_primal_solution(d_initial_primal);
    std::vector<double> initial_dual(op_problem.get_n_constraints(), 1);
    auto d_initial_dual = device_copy(initial_dual, handle_.get_stream());
    solver.set_initial_dual_solution(d_initial_dual);
    solver.run_solver(start_solver);
    EXPECT_NOT_NEAR(initial_step_size_afiro, solver.get_step_size_h(), factor_tolerance);
    EXPECT_NOT_NEAR(initial_primal_weight_afiro, solver.get_primal_weight_h(), factor_tolerance);
    cuopt::linear_programming::pdlp_hyper_params::update_primal_weight_on_initial_solution = false;
    cuopt::linear_programming::pdlp_hyper_params::update_step_size_on_initial_solution     = false;
  }
}

TEST(pdlp_class, initial_primal_weight_step_size_test)
{
  const raft::handle_t handle_{};

  auto path = make_path_absolute("linear_programming/afiro_original.mps");
  cuopt::mps_parser::mps_data_model_t<int, double> mps_data_model =
    cuopt::mps_parser::parse_mps<int, double>(path);

  auto op_problem = cuopt::linear_programming::mps_data_model_to_optimization_problem<int, double>(
    &handle_, mps_data_model);
  cuopt::linear_programming::detail::problem_t<int, double> problem(op_problem);

  auto solver_settings = pdlp_solver_settings_t<int, double>{};
  // We are just testing initial scaling on initial solution scheme so we don't care about solver
  solver_settings.iteration_limit = 0;
  solver_settings.method          = cuopt::linear_programming::method_t::PDLP;
  // Select the default/legacy solver with no action upon the initial scaling on initial solution
  solver_settings.pdlp_solver_mode = cuopt::linear_programming::pdlp_solver_mode_t::Methodical1;
  EXPECT_FALSE(cuopt::linear_programming::pdlp_hyper_params::update_step_size_on_initial_solution);
  EXPECT_FALSE(
    cuopt::linear_programming::pdlp_hyper_params::update_primal_weight_on_initial_solution);

  // Check setting an initial primal weight and step size
  {
    cuopt::linear_programming::detail::pdlp_solver_t<int, double> solver(problem, solver_settings);
    auto start_solver                           = std::chrono::high_resolution_clock::now();
    constexpr double test_initial_step_size     = 1.0;
    constexpr double test_initial_primal_weight = 2.0;
    solver.set_initial_primal_weight(test_initial_primal_weight);
    solver.set_initial_step_size(test_initial_step_size);
    solver.run_solver(start_solver);
    RAFT_CUDA_TRY(cudaStreamSynchronize(handle_.get_stream()));
    EXPECT_EQ(test_initial_step_size, solver.get_step_size_h());
    EXPECT_EQ(test_initial_primal_weight, solver.get_primal_weight_h());
  }

  // Check that after setting an initial step size and primal weight, the computed one when adding
  // an initial primal / dual is indeed different
  {
    // Launching without an inital step size / primal weight and query the value
    cuopt::linear_programming::pdlp_hyper_params::update_primal_weight_on_initial_solution = true;
    cuopt::linear_programming::pdlp_hyper_params::update_step_size_on_initial_solution     = true;
    cuopt::linear_programming::detail::pdlp_solver_t<int, double> solver(problem, solver_settings);
    auto start_solver = std::chrono::high_resolution_clock::now();
    std::vector<double> initial_primal(op_problem.get_n_variables(), 1);
    auto d_initial_primal = device_copy(initial_primal, handle_.get_stream());
    solver.set_initial_primal_solution(d_initial_primal);
    std::vector<double> initial_dual(op_problem.get_n_constraints(), 1);
    auto d_initial_dual = device_copy(initial_dual, handle_.get_stream());
    solver.set_initial_dual_solution(d_initial_dual);
    solver.run_solver(start_solver);
    const double previous_step_size     = solver.get_step_size_h();
    const double previous_primal_weight = solver.get_primal_weight_h();

    // Start again but with an initial and check the impact
    cuopt::linear_programming::detail::pdlp_solver_t<int, double> solver2(problem, solver_settings);
    start_solver                                = std::chrono::high_resolution_clock::now();
    constexpr double test_initial_step_size     = 1.0;
    constexpr double test_initial_primal_weight = 2.0;
    solver2.set_initial_primal_weight(test_initial_primal_weight);
    solver2.set_initial_step_size(test_initial_step_size);
    solver2.set_initial_primal_solution(d_initial_primal);
    solver2.set_initial_dual_solution(d_initial_dual);
    solver2.run_solver(start_solver);
    RAFT_CUDA_TRY(cudaStreamSynchronize(handle_.get_stream()));
    const double sovler2_step_size     = solver2.get_step_size_h();
    const double sovler2_primal_weight = solver2.get_primal_weight_h();
    EXPECT_NOT_NEAR(previous_step_size, sovler2_step_size, factor_tolerance);
    EXPECT_NOT_NEAR(previous_primal_weight, sovler2_primal_weight, factor_tolerance);

    // Again but with an initial k which should change the step size only, not the primal weight
    cuopt::linear_programming::detail::pdlp_solver_t<int, double> solver3(problem, solver_settings);
    start_solver = std::chrono::high_resolution_clock::now();
    solver3.set_initial_primal_weight(test_initial_primal_weight);
    solver3.set_initial_step_size(test_initial_step_size);
    solver3.set_initial_primal_solution(d_initial_primal);
    solver3.set_initial_k(10000);
    solver3.set_initial_dual_solution(d_initial_dual);
    solver3.set_initial_dual_solution(d_initial_dual);
    solver3.run_solver(start_solver);
    RAFT_CUDA_TRY(cudaStreamSynchronize(handle_.get_stream()));
    EXPECT_NOT_NEAR(sovler2_step_size, solver3.get_step_size_h(), factor_tolerance);
    EXPECT_NEAR(sovler2_primal_weight, solver3.get_primal_weight_h(), factor_tolerance);
  }
}

TEST(pdlp_class, initial_rhs_and_c)
{
  const raft::handle_t handle_{};

  auto path = make_path_absolute("linear_programming/afiro_original.mps");
  cuopt::mps_parser::mps_data_model_t<int, double> mps_data_model =
    cuopt::mps_parser::parse_mps<int, double>(path);

  auto op_problem = cuopt::linear_programming::mps_data_model_to_optimization_problem<int, double>(
    &handle_, mps_data_model);
  cuopt::linear_programming::detail::problem_t<int, double> problem(op_problem);

  cuopt::linear_programming::detail::pdlp_solver_t<int, double> solver(problem);
  constexpr double test_initial_primal_factor = 1.0;
  constexpr double test_initial_dual_factor   = 2.0;
  solver.set_relative_dual_tolerance_factor(test_initial_dual_factor);
  solver.set_relative_primal_tolerance_factor(test_initial_primal_factor);

  EXPECT_EQ(solver.get_relative_dual_tolerance_factor(), test_initial_dual_factor);
  EXPECT_EQ(solver.get_relative_primal_tolerance_factor(), test_initial_primal_factor);
}

TEST(pdlp_class, per_constraint_test)
{
  /*
   * Define the following LP:
   * x1=0.01 <= 0
   * x2=0.01 <= 0
   * x3=0.1  <= 0
   *
   * With a tol of 0.1 per constraint will pass but the L2 version will not as L2 of primal residual
   * will be 0.1009
   */
  raft::handle_t handle;
  auto op_problem = optimization_problem_t<int, double>(&handle);

  std::vector<double> A_host           = {1.0, 1.0, 1.0};
  std::vector<int> indices_host        = {0, 1, 2};
  std::vector<int> offset_host         = {0, 1, 2, 3};
  std::vector<double> b_host           = {0.0, 0.0, 0.0};
  std::vector<double> h_initial_primal = {0.02, 0.03, 0.1};
  rmm::device_uvector<double> d_initial_primal(3, handle.get_stream());
  raft::copy(
    d_initial_primal.data(), h_initial_primal.data(), h_initial_primal.size(), handle.get_stream());

  op_problem.set_csr_constraint_matrix(A_host.data(),
                                       A_host.size(),
                                       indices_host.data(),
                                       indices_host.size(),
                                       offset_host.data(),
                                       offset_host.size());
  op_problem.set_constraint_lower_bounds(b_host.data(), b_host.size());
  op_problem.set_constraint_upper_bounds(b_host.data(), b_host.size());
  op_problem.set_objective_coefficients(b_host.data(), b_host.size());

  auto problem = cuopt::linear_programming::detail::problem_t<int, double>(op_problem);

  pdlp_solver_settings_t<int, double> solver_settings;
  solver_settings.tolerances.relative_primal_tolerance = 0;  // Shouldn't matter
  solver_settings.tolerances.absolute_primal_tolerance = 0.1;
  solver_settings.tolerances.relative_dual_tolerance   = 0;  // Shoudln't matter
  solver_settings.tolerances.absolute_dual_tolerance   = 0.1;
  solver_settings.method                               = cuopt::linear_programming::method_t::PDLP;

  // First solve without the per constraint and it should break
  {
    cuopt::linear_programming::detail::pdlp_solver_t<int, double> solver(problem, solver_settings);

    raft::copy(solver.pdhg_solver_.get_primal_solution().data(),
               d_initial_primal.data(),
               d_initial_primal.size(),
               handle.get_stream());

    auto& current_termination_strategy = solver.get_current_termination_strategy();
    pdlp_termination_status_t termination_average =
      current_termination_strategy.evaluate_termination_criteria(solver.pdhg_solver_,
                                                                 d_initial_primal,
                                                                 d_initial_primal,
                                                                 problem.combined_bounds,
                                                                 problem.objective_coefficients);

    EXPECT_TRUE(termination_average != pdlp_termination_status_t::Optimal);
  }
  {
    solver_settings.per_constraint_residual = true;
    cuopt::linear_programming::detail::pdlp_solver_t<int, double> solver(problem, solver_settings);

    raft::copy(solver.pdhg_solver_.get_primal_solution().data(),
               d_initial_primal.data(),
               d_initial_primal.size(),
               handle.get_stream());

    auto& current_termination_strategy = solver.get_current_termination_strategy();
    pdlp_termination_status_t termination_average =
      current_termination_strategy.evaluate_termination_criteria(solver.pdhg_solver_,
                                                                 d_initial_primal,
                                                                 d_initial_primal,
                                                                 problem.combined_bounds,
                                                                 problem.objective_coefficients);
    EXPECT_EQ(current_termination_strategy.get_convergence_information()
                .get_relative_linf_primal_residual()
                .value(handle.get_stream()),
              0.1);
  }
}

TEST(pdlp_class, best_primal_so_far_iteration)
{
  const raft::handle_t handle1{};
  const raft::handle_t handle2{};

  auto path            = make_path_absolute("linear_programming/ns1687037/ns1687037.mps");
  auto solver_settings = pdlp_solver_settings_t<int, double>{};
  solver_settings.iteration_limit         = 3000;
  solver_settings.per_constraint_residual = true;
  solver_settings.method                  = cuopt::linear_programming::method_t::PDLP;

  cuopt::mps_parser::mps_data_model_t<int, double> op_problem1 =
    cuopt::mps_parser::parse_mps<int, double>(path);
  cuopt::mps_parser::mps_data_model_t<int, double> op_problem2 =
    cuopt::mps_parser::parse_mps<int, double>(path);

  optimization_problem_solution_t<int, double> solution1 =
    solve_lp(&handle1, op_problem1, solver_settings);
  RAFT_CUDA_TRY(cudaDeviceSynchronize());
  solver_settings.save_best_primal_so_far = true;
  optimization_problem_solution_t<int, double> solution2 =
    solve_lp(&handle2, op_problem2, solver_settings);
  RAFT_CUDA_TRY(cudaDeviceSynchronize());

  EXPECT_TRUE(solution2.get_additional_termination_information().l2_primal_residual <
              solution1.get_additional_termination_information().l2_primal_residual);
}

TEST(pdlp_class, best_primal_so_far_time)
{
  const raft::handle_t handle1{};
  const raft::handle_t handle2{};

  auto path                  = make_path_absolute("linear_programming/ns1687037/ns1687037.mps");
  auto solver_settings       = pdlp_solver_settings_t<int, double>{};
  solver_settings.time_limit = 2;
  solver_settings.per_constraint_residual = true;
  solver_settings.pdlp_solver_mode        = cuopt::linear_programming::pdlp_solver_mode_t::Stable1;
  solver_settings.method                  = cuopt::linear_programming::method_t::PDLP;

  cuopt::mps_parser::mps_data_model_t<int, double> op_problem1 =
    cuopt::mps_parser::parse_mps<int, double>(path);
  cuopt::mps_parser::mps_data_model_t<int, double> op_problem2 =
    cuopt::mps_parser::parse_mps<int, double>(path);

  optimization_problem_solution_t<int, double> solution1 =
    solve_lp(&handle1, op_problem1, solver_settings);
  RAFT_CUDA_TRY(cudaDeviceSynchronize());
  solver_settings.save_best_primal_so_far = true;
  optimization_problem_solution_t<int, double> solution2 =
    solve_lp(&handle2, op_problem2, solver_settings);
  RAFT_CUDA_TRY(cudaDeviceSynchronize());

  EXPECT_TRUE(solution2.get_additional_termination_information().l2_primal_residual <
              solution1.get_additional_termination_information().l2_primal_residual);
}

TEST(pdlp_class, first_primal_feasible)
{
  const raft::handle_t handle1{};
  const raft::handle_t handle2{};

  auto path            = make_path_absolute("linear_programming/ns1687037/ns1687037.mps");
  auto solver_settings = pdlp_solver_settings_t<int, double>{};
  solver_settings.iteration_limit         = 1000;
  solver_settings.per_constraint_residual = true;
  solver_settings.set_optimality_tolerance(1e-2);
  solver_settings.method = cuopt::linear_programming::method_t::PDLP;

  cuopt::mps_parser::mps_data_model_t<int, double> op_problem1 =
    cuopt::mps_parser::parse_mps<int, double>(path);
  cuopt::mps_parser::mps_data_model_t<int, double> op_problem2 =
    cuopt::mps_parser::parse_mps<int, double>(path);

  optimization_problem_solution_t<int, double> solution1 =
    solve_lp(&handle1, op_problem1, solver_settings);
  RAFT_CUDA_TRY(cudaDeviceSynchronize());
  solver_settings.first_primal_feasible = true;
  optimization_problem_solution_t<int, double> solution2 =
    solve_lp(&handle2, op_problem2, solver_settings);
  RAFT_CUDA_TRY(cudaDeviceSynchronize());

  EXPECT_EQ(solution1.get_termination_status(), pdlp_termination_status_t::IterationLimit);
  EXPECT_EQ(solution2.get_termination_status(), pdlp_termination_status_t::PrimalFeasible);
}

TEST(pdlp_class, warm_start)
{
  std::vector<std::string> instance_names{"graph40-40",
                                          "ex10",
                                          "datt256_lp",
                                          "woodlands09",
                                          "savsched1",
                                          "nug08-3rd",
                                          "qap15",
                                          "scpm1",
                                          "neos3",
                                          "a2864"};
  for (auto instance_name : instance_names) {
    const raft::handle_t handle{};

    auto path =
      make_path_absolute("linear_programming/" + instance_name + "/" + instance_name + ".mps");
    auto solver_settings             = pdlp_solver_settings_t<int, double>{};
    solver_settings.pdlp_solver_mode = cuopt::linear_programming::pdlp_solver_mode_t::Stable2;
    solver_settings.set_optimality_tolerance(1e-2);
    solver_settings.detect_infeasibility = false;
    solver_settings.method               = cuopt::linear_programming::method_t::PDLP;

    cuopt::mps_parser::mps_data_model_t<int, double> mps_data_model =
      cuopt::mps_parser::parse_mps<int, double>(path);
    auto op_problem1 =
      cuopt::linear_programming::mps_data_model_to_optimization_problem<int, double>(
        &handle, mps_data_model);

    // Solving from scratch until 1e-2
    optimization_problem_solution_t<int, double> solution1 = solve_lp(op_problem1, solver_settings);

    // Solving until 1e-1 to use the result as a warm start
    solver_settings.set_optimality_tolerance(1e-1);
    auto op_problem2 =
      cuopt::linear_programming::mps_data_model_to_optimization_problem<int, double>(
        &handle, mps_data_model);
    optimization_problem_solution_t<int, double> solution2 = solve_lp(op_problem2, solver_settings);

    // Solving until 1e-2 using the previous state as a warm start
    solver_settings.set_optimality_tolerance(1e-2);
    auto op_problem3 =
      cuopt::linear_programming::mps_data_model_to_optimization_problem<int, double>(
        &handle, mps_data_model);
    solver_settings.set_pdlp_warm_start_data(solution2.get_pdlp_warm_start_data());
    optimization_problem_solution_t<int, double> solution3 = solve_lp(op_problem3, solver_settings);

    EXPECT_EQ(solution1.get_additional_termination_information().number_of_steps_taken,
              solution3.get_additional_termination_information().number_of_steps_taken +
                solution2.get_additional_termination_information().number_of_steps_taken);
  }
}

TEST(dual_simplex, afiro)
{
  cuopt::linear_programming::pdlp_solver_settings_t<int, double> settings =
    cuopt::linear_programming::pdlp_solver_settings_t<int, double>{};
  settings.method = cuopt::linear_programming::method_t::DualSimplex;

  const raft::handle_t handle_{};

  auto path = make_path_absolute("linear_programming/afiro_original.mps");
  cuopt::mps_parser::mps_data_model_t<int, double> op_problem =
    cuopt::mps_parser::parse_mps<int, double>(path, true);

  optimization_problem_solution_t<int, double> solution = solve_lp(&handle_, op_problem, settings);
  EXPECT_EQ(solution.get_termination_status(), pdlp_termination_status_t::Optimal);
  EXPECT_FALSE(is_incorrect_objective(
    afiro_primal_objective, solution.get_additional_termination_information().primal_objective));
}

// Should return a numerical error
TEST(pdlp_class, run_empty_matrix_pdlp)
{
  const raft::handle_t handle_{};

  auto path = make_path_absolute("linear_programming/empty_matrix.mps");
  cuopt::mps_parser::mps_data_model_t<int, double> op_problem =
    cuopt::mps_parser::parse_mps<int, double>(path);

  auto solver_settings   = pdlp_solver_settings_t<int, double>{};
  solver_settings.method = cuopt::linear_programming::method_t::PDLP;

  optimization_problem_solution_t<int, double> solution =
    solve_lp(&handle_, op_problem, solver_settings);
  EXPECT_EQ((int)solution.get_termination_status(), CUOPT_TERIMINATION_STATUS_NUMERICAL_ERROR);
}

// Should run thanks to Dual Simplex
TEST(pdlp_class, run_empty_matrix_dual_simplex)
{
  const raft::handle_t handle_{};

  auto path = make_path_absolute("linear_programming/empty_matrix.mps");
  cuopt::mps_parser::mps_data_model_t<int, double> op_problem =
    cuopt::mps_parser::parse_mps<int, double>(path);

  auto solver_settings   = pdlp_solver_settings_t<int, double>{};
  solver_settings.method = cuopt::linear_programming::method_t::Concurrent;

  optimization_problem_solution_t<int, double> solution =
    solve_lp(&handle_, op_problem, solver_settings);
  EXPECT_EQ((int)solution.get_termination_status(), CUOPT_TERIMINATION_STATUS_OPTIMAL);
  EXPECT_FALSE(solution.get_additional_termination_information().solved_by_pdlp);
}

TEST(pdlp_class, test_max)
{
  const raft::handle_t handle_{};

  auto path = make_path_absolute("linear_programming/good-max.mps");
  cuopt::mps_parser::mps_data_model_t<int, double> op_problem =
    cuopt::mps_parser::parse_mps<int, double>(path);

  auto solver_settings   = pdlp_solver_settings_t<int, double>{};
  solver_settings.method = cuopt::linear_programming::method_t::PDLP;

  optimization_problem_solution_t<int, double> solution =
    solve_lp(&handle_, op_problem, solver_settings);
  EXPECT_EQ((int)solution.get_termination_status(), CUOPT_TERIMINATION_STATUS_OPTIMAL);
  EXPECT_NEAR(
    solution.get_additional_termination_information().primal_objective, 17.0, factor_tolerance);
}

TEST(pdlp_class, test_max_with_offset)
{
  const raft::handle_t handle_{};

  auto path = make_path_absolute("linear_programming/max_offset.mps");
  cuopt::mps_parser::mps_data_model_t<int, double> op_problem =
    cuopt::mps_parser::parse_mps<int, double>(path);

  auto solver_settings   = pdlp_solver_settings_t<int, double>{};
  solver_settings.method = cuopt::linear_programming::method_t::PDLP;

  optimization_problem_solution_t<int, double> solution =
    solve_lp(&handle_, op_problem, solver_settings);
  EXPECT_EQ((int)solution.get_termination_status(), CUOPT_TERIMINATION_STATUS_OPTIMAL);
  EXPECT_NEAR(
    solution.get_additional_termination_information().primal_objective, 0.0, factor_tolerance);
}

TEST(pdlp_class, test_lp_no_constraints)
{
  const raft::handle_t handle_{};

  auto path = make_path_absolute("linear_programming/lp-model-no-constraints.mps");
  cuopt::mps_parser::mps_data_model_t<int, double> op_problem =
    cuopt::mps_parser::parse_mps<int, double>(path);

  auto solver_settings = pdlp_solver_settings_t<int, double>{};

  optimization_problem_solution_t<int, double> solution =
    solve_lp(&handle_, op_problem, solver_settings);
  EXPECT_EQ((int)solution.get_termination_status(), CUOPT_TERIMINATION_STATUS_OPTIMAL);
  EXPECT_NEAR(
    solution.get_additional_termination_information().primal_objective, 1.0, factor_tolerance);
}

}  // namespace cuopt::linear_programming::test

CUOPT_TEST_PROGRAM_MAIN()
