/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
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

#include "problem_checking.cuh"

#include <cuopt/error.hpp>
#include <cuopt/linear_programming/optimization_problem.hpp>
#include <mip/mip_constants.hpp>

#include <thrust/functional.h>
#include <thrust/logical.h>
#include <thrust/sort.h>

namespace cuopt::linear_programming {

template <typename i_t, typename f_t>
void problem_checking_t<i_t, f_t>::check_csr_representation(
  const optimization_problem_t<i_t, f_t>& op_problem)
{
  cuopt_expects(op_problem.get_constraint_matrix_indices().size() ==
                  op_problem.get_constraint_matrix_values().size(),
                error_type_t::ValidationError,
                "A_index and A_values must have same sizes.");

  // Check offset values
  const i_t first_value = op_problem.get_constraint_matrix_offsets().front_element(
    op_problem.get_handle_ptr()->get_stream());
  cuopt_expects(
    first_value == 0, error_type_t::ValidationError, "A_offsets first value should be 0.");

  cuopt_expects(thrust::is_sorted(op_problem.get_handle_ptr()->get_thrust_policy(),
                                  op_problem.get_constraint_matrix_offsets().cbegin(),
                                  op_problem.get_constraint_matrix_offsets().cend()),
                error_type_t::ValidationError,
                "A_offsets values must in an increasing order.");

  // Check indices
  cuopt_expects(thrust::all_of(op_problem.get_handle_ptr()->get_thrust_policy(),
                               op_problem.get_constraint_matrix_indices().cbegin(),
                               op_problem.get_constraint_matrix_indices().cend(),
                               [n_variables = op_problem.get_n_variables()] __device__(i_t val) {
                                 return val >= 0 && val < n_variables;
                               }),
                error_type_t::ValidationError,
                "A_indices values must positive lower than the number of variables (c size).");
}

template <typename i_t, typename f_t>
void problem_checking_t<i_t, f_t>::check_initial_primal_representation(
  const rmm::device_uvector<f_t>& objective_coefficients,
  const rmm::device_uvector<f_t>& primal_initial_solution)
{
  // Inital solution check if set
  if (!primal_initial_solution.is_empty()) {
    cuopt_expects(
      (primal_initial_solution.size() == objective_coefficients.size()),
      error_type_t::ValidationError,
      "Sizes for vectors related to the variables are not the same. The initial primal variable "
      "has size %zu, while objective vector has size %zu.",
      primal_initial_solution.size(),
      objective_coefficients.size());
  }
}

template <typename i_t, typename f_t>
void problem_checking_t<i_t, f_t>::check_initial_dual_representation(
  const rmm::device_uvector<f_t>& constraints,
  const rmm::device_uvector<f_t>& dual_initial_solution)
{
  if (!dual_initial_solution.is_empty()) {
    cuopt_expects(
      (dual_initial_solution.size() == constraints.size()),
      error_type_t::ValidationError,
      "Sizes for vectors related to the variables are not the same. The initial dual variable "
      "has size %zu, while constraint vector has size %zu.",
      dual_initial_solution.size(),
      constraints.size());
  }
}

template <typename i_t, typename f_t>
void problem_checking_t<i_t, f_t>::check_initial_solution_representation(
  const optimization_problem_t<i_t, f_t>& op_problem,
  const pdlp_solver_settings_t<i_t, f_t>& settings)
{
  if (settings.initial_primal_solution_.get() != nullptr) {
    check_initial_primal_representation(op_problem.get_objective_coefficients(), settings.get_initial_primal_solution());
  }
  if (settings.initial_dual_solution_.get() != nullptr) {
        const auto& constraints = (op_problem.get_constraint_lower_bounds().is_empty())
                                        ? op_problem.get_constraint_bounds()
                                        : op_problem.get_constraint_lower_bounds();
    check_initial_dual_representation(constraints, settings.get_initial_dual_solution());
  }
}

template <typename i_t, typename f_t>
void problem_checking_t<i_t, f_t>::check_initial_solution_representation(
  const detail::problem_t<i_t, f_t>& problem,
  const pdlp_solver_settings_t<i_t, f_t>& settings)
{
  if (settings.initial_primal_solution_.get() != nullptr) {
    check_initial_primal_representation(problem.objective_coefficients, settings.get_initial_primal_solution());
  }
  if (settings.initial_dual_solution_.get() != nullptr) {
    check_initial_dual_representation(problem.constraint_lower_bounds, settings.get_initial_dual_solution());
  }
}

template <typename i_t, typename f_t>
void problem_checking_t<i_t, f_t>::check_initial_solution_representation(
  const optimization_problem_t<i_t, f_t>& op_problem,
  const mip_solver_settings_t<i_t, f_t>& settings)
{
  if (settings.initial_solution_.get() != nullptr) {
    check_initial_primal_representation(op_problem.get_objective_coefficients(), settings.get_initial_solution());
  }
}

template <typename i_t, typename f_t>
void problem_checking_t<i_t, f_t>::check_problem_representation(
  const optimization_problem_t<i_t, f_t>& op_problem)
{
  bool empty_problem = op_problem.get_constraint_matrix_values().is_empty();

  // Check for empty fields
  cuopt_expects(empty_problem || !op_problem.get_constraint_matrix_values().is_empty(),
                error_type_t::ValidationError,
                "A_values must be set before calling the solver.");
  cuopt_expects(empty_problem || !op_problem.get_constraint_matrix_indices().is_empty(),
                error_type_t::ValidationError,
                "A_indices must be set before calling the solver.");
  // There should always be at least one offset in the constraint matrix offsets, even if the
  // problem is empty
  cuopt_expects(!op_problem.get_constraint_matrix_offsets().is_empty(),
                error_type_t::ValidationError,
                "A_offsets must be set before calling the solver.");
  cuopt_expects(empty_problem || !op_problem.get_objective_coefficients().is_empty(),
                error_type_t::ValidationError,
                "c must be set before calling the solver.");

  // Check CSR validity
  check_csr_representation(op_problem);

  // Check if bounds are set, that they are both set
  cuopt_expects(!(op_problem.get_constraint_lower_bounds().is_empty() &&
                  !op_problem.get_constraint_upper_bounds().is_empty()),
                error_type_t::ValidationError,
                "Constraints lower bounds must be set along with constraints upper bounds.");

  cuopt_expects(!(!op_problem.get_constraint_lower_bounds().is_empty() &&
                  op_problem.get_constraint_upper_bounds().is_empty()),
                error_type_t::ValidationError,
                "Constraints upper bounds must be set along with constraints lower bounds.");

  // Check constraint bounds
  cuopt_expects(
    empty_problem ||
      (!op_problem.get_constraint_lower_bounds().is_empty() &&
       !op_problem.get_constraint_upper_bounds().is_empty()) ||
      (!op_problem.get_row_types().is_empty() && !op_problem.get_constraint_bounds().is_empty()),
    error_type_t::ValidationError,
    "Either constraints lower/upper bounds or row types and constraints bounds needs to be set "
    "before calling the solver.");

  // Check row type if set
  if (!op_problem.get_row_types().is_empty()) {
    cuopt_expects(
      thrust::all_of(op_problem.get_handle_ptr()->get_thrust_policy(),
                     op_problem.get_row_types().cbegin(),
                     op_problem.get_row_types().cend(),
                     [] __device__(char val) { return val == 'E' || val == 'G' || val == 'L'; }),
      error_type_t::ValidationError,
      "row_types values must equal to 'E', 'G' or 'L'.");

    cuopt_expects(
      op_problem.get_row_types().size() == op_problem.get_constraint_bounds().size(),
      error_type_t::ValidationError,
      "Sizes for vectors related to the constraints are not the same. The right hand side "
      "vector has size %zu and the row type has size %zu.",
      op_problem.get_constraint_bounds().size(),
      op_problem.get_row_types().size());

    cuopt_expects(
      (op_problem.get_constraint_matrix_offsets().size() - 1) ==
        op_problem.get_constraint_bounds().size(),
      error_type_t::ValidationError,
      "Sizes for vectors related to the constraints are not the same. The right hand side "
      "vector has size %zu and the number of rows (constraints) in the matrix is %zu.",
      op_problem.get_constraint_bounds().size(),
      op_problem.get_constraint_matrix_offsets().size());
  }

  // Check sizes only if user has set it (we can't compare lower/upper bounds size because one might
  // have not been set by the user)
  if (!op_problem.get_variable_lower_bounds().is_empty()) {
    cuopt_expects(op_problem.get_variable_lower_bounds().size() ==
                    op_problem.get_objective_coefficients().size(),
                  error_type_t::ValidationError,
                  "Sizes for vectors related to the variables are not the same. The objective "
                  "vector has size %zu and the variable lower bounds vector has size %zu.",
                  op_problem.get_objective_coefficients().size(),
                  op_problem.get_variable_lower_bounds().size());
  }
  if (!op_problem.get_variable_upper_bounds().is_empty()) {
    cuopt_expects(op_problem.get_variable_upper_bounds().size() ==
                    op_problem.get_objective_coefficients().size(),
                  error_type_t::ValidationError,
                  "Sizes for vectors related to the variables are not the same. The objective "
                  "vector has size %zu and the variable upper bounds vector has size %zu.",
                  op_problem.get_objective_coefficients().size(),
                  op_problem.get_variable_upper_bounds().size());
  }

  // Check constraints sizes
  cuopt_expects(
    op_problem.get_constraint_lower_bounds().size() ==
      op_problem.get_constraint_upper_bounds().size(),
    error_type_t::ValidationError,
    "Sizes for vectors related to the constraints are not the same. The constraint lower bounds "
    "vector has size %zu and the constraint upper bounds vector has size %zu.",
    op_problem.get_constraint_lower_bounds().size(),
    op_problem.get_constraint_upper_bounds().size());

  // If constraint are set by user, check size
  if (!op_problem.get_constraint_lower_bounds().is_empty()) {
    cuopt_expects(
      (op_problem.get_constraint_matrix_offsets().size() - 1) ==
        op_problem.get_constraint_lower_bounds().size(),
      error_type_t::ValidationError,
      "Sizes for vectors related to the constraints are not the same. The constraint lower bounds "
      "vector has size %zu and the number of rows (constraints) in the matrix is %zu.",
      op_problem.get_constraint_lower_bounds().size(),
      op_problem.get_constraint_matrix_offsets().size());
  }
  if (!op_problem.get_constraint_upper_bounds().is_empty()) {
    cuopt_expects(
      (op_problem.get_constraint_matrix_offsets().size() - 1) ==
        op_problem.get_constraint_upper_bounds().size(),
      error_type_t::ValidationError,
      "Sizes for vectors related to the constraints are not the same. The constraint upper bounds "
      "vector has size %zu and the number of rows (constraints) in the matrix is %zu.",
      op_problem.get_constraint_upper_bounds().size(),
      op_problem.get_constraint_matrix_offsets().size());
  }
}

template <typename i_t, typename f_t>
void problem_checking_t<i_t, f_t>::check_scaled_problem(
  detail::problem_t<i_t, f_t> const& scaled_problem, detail::problem_t<i_t, f_t> const& op_problem)
{
  // original problem to host
  auto& d_variable_upper_bounds = op_problem.variable_upper_bounds;
  auto& d_variable_lower_bounds = op_problem.variable_lower_bounds;
  auto& d_variable_types        = op_problem.variable_types;
  std::vector<f_t> variable_upper_bounds(d_variable_upper_bounds.size());
  std::vector<f_t> variable_lower_bounds(d_variable_lower_bounds.size());
  std::vector<var_t> variable_types(d_variable_types.size());

  raft::copy(variable_upper_bounds.data(),
             d_variable_upper_bounds.data(),
             d_variable_upper_bounds.size(),
             op_problem.handle_ptr->get_stream());
  raft::copy(variable_lower_bounds.data(),
             d_variable_lower_bounds.data(),
             d_variable_lower_bounds.size(),
             op_problem.handle_ptr->get_stream());
  raft::copy(variable_types.data(),
             d_variable_types.data(),
             d_variable_types.size(),
             op_problem.handle_ptr->get_stream());

  // scaled problem to host
  std::vector<f_t> scaled_variable_upper_bounds(scaled_problem.variable_upper_bounds.size());
  std::vector<f_t> scaled_variable_lower_bounds(scaled_problem.variable_lower_bounds.size());
  std::vector<f_t> scaled_variables(scaled_problem.variable_lower_bounds.size());

  raft::copy(scaled_variable_upper_bounds.data(),
             scaled_problem.variable_upper_bounds.data(),
             scaled_problem.variable_upper_bounds.size(),
             op_problem.handle_ptr->get_stream());
  raft::copy(scaled_variable_lower_bounds.data(),
             scaled_problem.variable_lower_bounds.data(),
             scaled_problem.variable_lower_bounds.size(),
             op_problem.handle_ptr->get_stream());
  for (size_t i = 0; i < variable_types.size(); ++i) {
    auto var_type = variable_types[i];
    if (var_type == var_t::INTEGER) {
      // Integers should be untouched
      cuopt_assert(variable_upper_bounds[i] == scaled_variable_upper_bounds[i],
                   "Mismatch upper scaling");
      cuopt_assert(variable_lower_bounds[i] == scaled_variable_lower_bounds[i],
                   "Mismatch lower scaling");
    }
  }
}

template <typename i_t, typename f_t>
void problem_checking_t<i_t, f_t>::check_unscaled_solution(
  detail::problem_t<i_t, f_t>& op_problem, rmm::device_uvector<f_t> const& assignment)
{
  auto& d_variable_upper_bounds = op_problem.variable_upper_bounds;
  auto& d_variable_lower_bounds = op_problem.variable_lower_bounds;
  std::vector<f_t> variable_upper_bounds(d_variable_upper_bounds.size());
  std::vector<f_t> variable_lower_bounds(d_variable_lower_bounds.size());
  std::vector<f_t> h_assignment(assignment.size());

  raft::copy(variable_upper_bounds.data(),
             d_variable_upper_bounds.data(),
             d_variable_upper_bounds.size(),
             op_problem.handle_ptr->get_stream());
  raft::copy(variable_lower_bounds.data(),
             d_variable_lower_bounds.data(),
             d_variable_lower_bounds.size(),
             op_problem.handle_ptr->get_stream());
  raft::copy(
    h_assignment.data(), assignment.data(), assignment.size(), op_problem.handle_ptr->get_stream());
  const f_t int_tol = op_problem.tolerances.integrality_tolerance;
  for (size_t i = 0; i < variable_upper_bounds.size(); ++i) {
    cuopt_assert(h_assignment[i] <= variable_upper_bounds[i] + int_tol, "Excess upper bound");
    cuopt_assert(h_assignment[i] >= variable_lower_bounds[i] - int_tol, "Excess lower bound");
  }
}

#define INSTANTIATE(F_TYPE) template class problem_checking_t<int, F_TYPE>;

#if MIP_INSTANTIATE_FLOAT
INSTANTIATE(float)
#endif

#if MIP_INSTANTIATE_DOUBLE
INSTANTIATE(double)
#endif

#undef INSTANTIATE

}  // namespace cuopt::linear_programming
