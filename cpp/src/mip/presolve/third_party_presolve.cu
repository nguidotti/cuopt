/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

#include <mip/mip_constants.hpp>
#include <mip/presolve/third_party_presolve.cuh>
#include <papilo/core/Presolve.hpp>
#include <papilo/core/ProblemBuilder.hpp>
#include <utilities/copy_helpers.hpp>

namespace cuopt::linear_programming::detail {

template <typename i_t, typename f_t>
papilo::Problem<f_t> build_papilo_problem(const optimization_problem_t<i_t, f_t>& op_problem)
{
  // Build papilo problem from optimization problem
  papilo::ProblemBuilder<f_t> builder;

  // Get problem dimensions
  const i_t num_cols = op_problem.get_n_variables();
  const i_t num_rows = op_problem.get_n_constraints();
  const i_t nnz      = op_problem.get_nnz();

  cuopt_expects(op_problem.get_sense() == false,
                error_type_t::ValidationError,
                "Papilo does not support maximization problems");

  builder.reserve(nnz, num_rows, num_cols);

  // Get problem data from optimization problem
  const auto& coefficients = op_problem.get_constraint_matrix_values();
  const auto& offsets      = op_problem.get_constraint_matrix_offsets();
  const auto& variables    = op_problem.get_constraint_matrix_indices();
  const auto& obj_coeffs   = op_problem.get_objective_coefficients();
  const auto& var_lb       = op_problem.get_variable_lower_bounds();
  const auto& var_ub       = op_problem.get_variable_upper_bounds();
  const auto& bounds       = op_problem.get_constraint_bounds();
  const auto& row_types    = op_problem.get_row_types();
  const auto& constr_lb    = op_problem.get_constraint_lower_bounds();
  const auto& constr_ub    = op_problem.get_constraint_upper_bounds();
  const auto& var_types    = op_problem.get_variable_types();

  // Copy data to host
  std::vector<f_t> h_coefficients = cuopt::host_copy(coefficients);
  std::vector<i_t> h_offsets      = cuopt::host_copy(offsets);
  std::vector<i_t> h_variables    = cuopt::host_copy(variables);
  std::vector<f_t> h_obj_coeffs   = cuopt::host_copy(obj_coeffs);
  std::vector<f_t> h_var_lb       = cuopt::host_copy(var_lb);
  std::vector<f_t> h_var_ub       = cuopt::host_copy(var_ub);
  std::vector<f_t> h_bounds       = cuopt::host_copy(bounds);
  std::vector<char> h_row_types   = cuopt::host_copy(row_types);
  std::vector<f_t> h_constr_lb    = cuopt::host_copy(constr_lb);
  std::vector<f_t> h_constr_ub    = cuopt::host_copy(constr_ub);
  std::vector<var_t> h_var_types  = cuopt::host_copy(var_types);

  auto constr_bounds_empty = h_constr_lb.empty() && h_constr_ub.empty();
  if (constr_bounds_empty) {
    for (size_t i = 0; i < h_row_types.size(); ++i) {
      if (h_row_types[i] == 'L') {
        h_constr_lb.push_back(-std::numeric_limits<f_t>::infinity());
        h_constr_ub.push_back(h_bounds[i]);
      } else if (h_row_types[i] == 'G') {
        h_constr_lb.push_back(h_bounds[i]);
        h_constr_ub.push_back(std::numeric_limits<f_t>::infinity());
      } else if (h_row_types[i] == 'E') {
        h_constr_lb.push_back(h_bounds[i]);
        h_constr_ub.push_back(h_bounds[i]);
      }
    }
  }

  builder.setNumCols(num_cols);
  builder.setNumRows(num_rows);

  builder.setObjAll(h_obj_coeffs);
  builder.setObjOffset(op_problem.get_objective_offset());

  if (!h_var_lb.empty() && !h_var_ub.empty()) {
    builder.setColLbAll(h_var_lb);
    builder.setColUbAll(h_var_ub);
  }

  for (size_t i = 0; i < h_var_types.size(); ++i) {
    builder.setColIntegral(i, h_var_types[i] == var_t::INTEGER);
  }

  if (!h_constr_lb.empty() && !h_constr_ub.empty()) {
    builder.setRowLhsAll(h_constr_lb);
    builder.setRowRhsAll(h_constr_ub);
  }

  // Add constraints row by row
  for (size_t i = 0; i < h_constr_lb.size(); ++i) {
    // Get row entries
    i_t row_start   = h_offsets[i];
    i_t row_end     = h_offsets[i + 1];
    i_t num_entries = row_end - row_start;
    builder.addRowEntries(
      i, num_entries, h_variables.data() + row_start, h_coefficients.data() + row_start);
    builder.setRowLhsInf(i, h_constr_lb[i] == -std::numeric_limits<f_t>::infinity());
    builder.setRowRhsInf(i, h_constr_ub[i] == std::numeric_limits<f_t>::infinity());
    if (h_constr_lb[i] == -std::numeric_limits<f_t>::infinity()) { builder.setRowLhs(i, 0); }
    if (h_constr_ub[i] == std::numeric_limits<f_t>::infinity()) { builder.setRowRhs(i, 0); }
  }

  for (size_t i = 0; i < h_var_lb.size(); ++i) {
    builder.setColLbInf(i, h_var_lb[i] == -std::numeric_limits<f_t>::infinity());
    builder.setColUbInf(i, h_var_ub[i] == std::numeric_limits<f_t>::infinity());
    if (h_var_lb[i] == -std::numeric_limits<f_t>::infinity()) { builder.setColLb(i, 0); }
    if (h_var_ub[i] == std::numeric_limits<f_t>::infinity()) { builder.setColUb(i, 0); }
  }
  return builder.build();
}

template <typename i_t, typename f_t>
optimization_problem_t<i_t, f_t> build_optimization_problem(
  papilo::Problem<f_t> const& papilo_problem, raft::handle_t const* handle_ptr)
{
  optimization_problem_t<i_t, f_t> op_problem(handle_ptr);
  if (papilo_problem.getNRows() == 0 && papilo_problem.getNCols() == 0) { return op_problem; }

  auto obj = papilo_problem.getObjective();
  op_problem.set_objective_coefficients(obj.coefficients.data(), obj.coefficients.size());
  op_problem.set_objective_offset(obj.offset);

  auto& constraint_matrix = papilo_problem.getConstraintMatrix();
  auto row_lower          = constraint_matrix.getLeftHandSides();
  auto row_upper          = constraint_matrix.getRightHandSides();
  auto col_lower          = papilo_problem.getLowerBounds();
  auto col_upper          = papilo_problem.getUpperBounds();

  auto row_flags = constraint_matrix.getRowFlags();
  for (size_t i = 0; i < row_flags.size(); i++) {
    if (row_flags[i].test(papilo::RowFlag::kLhsInf)) {
      row_lower[i] = -std::numeric_limits<f_t>::infinity();
    }
    if (row_flags[i].test(papilo::RowFlag::kRhsInf)) {
      row_upper[i] = std::numeric_limits<f_t>::infinity();
    }
  }

  op_problem.set_constraint_lower_bounds(row_lower.data(), row_lower.size());
  op_problem.set_constraint_upper_bounds(row_upper.data(), row_upper.size());

  auto [index_range, nrows] = constraint_matrix.getRangeInfo();

  std::vector<i_t> offsets(nrows + 1);
  // papilo indices do not start from 0 after presolve
  size_t start = index_range[0].start;
  for (i_t i = 0; i < nrows; i++) {
    offsets[i] = index_range[i].start - start;
  }
  offsets[nrows] = index_range[nrows - 1].end - start;

  i_t nnz = constraint_matrix.getNnz();
  assert(offsets[nrows] == nnz);

  const int* cols   = constraint_matrix.getConstraintMatrix().getColumns();
  const f_t* coeffs = constraint_matrix.getConstraintMatrix().getValues();
  op_problem.set_csr_constraint_matrix(
    &(coeffs[start]), nnz, &(cols[start]), nnz, offsets.data(), nrows + 1);

  auto col_flags = papilo_problem.getColFlags();
  std::vector<var_t> var_types(col_flags.size());
  for (size_t i = 0; i < col_flags.size(); i++) {
    var_types[i] =
      col_flags[i].test(papilo::ColFlag::kIntegral) ? var_t::INTEGER : var_t::CONTINUOUS;
    if (col_flags[i].test(papilo::ColFlag::kLbInf)) {
      col_lower[i] = -std::numeric_limits<f_t>::infinity();
    }
    if (col_flags[i].test(papilo::ColFlag::kUbInf)) {
      col_upper[i] = std::numeric_limits<f_t>::infinity();
    }
  }

  op_problem.set_variable_lower_bounds(col_lower.data(), col_lower.size());
  op_problem.set_variable_upper_bounds(col_upper.data(), col_upper.size());
  op_problem.set_variable_types(var_types.data(), var_types.size());

  return op_problem;
}

void check_presolve_status(const papilo::PresolveStatus& status)
{
  switch (status) {
    case papilo::PresolveStatus::kUnchanged:
      CUOPT_LOG_INFO("Presolve did not result in any changes");
      break;
    case papilo::PresolveStatus::kReduced: CUOPT_LOG_INFO("Presolve reduced the problem"); break;
    case papilo::PresolveStatus::kUnbndOrInfeas:
      CUOPT_LOG_INFO("Presolve found an unbounded or infeasible problem");
      break;
    case papilo::PresolveStatus::kInfeasible:
      CUOPT_LOG_INFO("Presolve found an infeasible problem");
      break;
    case papilo::PresolveStatus::kUnbounded:
      CUOPT_LOG_INFO("Presolve found an unbounded problem");
      break;
  }
}

void check_postsolve_status(const papilo::PostsolveStatus& status)
{
  switch (status) {
    case papilo::PostsolveStatus::kOk:
      CUOPT_LOG_INFO("Post-solve succeeded");
      break;
      // This occurs when the solution is not feasible
    case papilo::PostsolveStatus::kFailed: CUOPT_LOG_INFO("Post-solve failed"); break;
  }
}

template <typename f_t>
void set_presolve_methods(papilo::Presolve<f_t>& presolver, problem_category_t category)
{
  using uptr = std::unique_ptr<papilo::PresolveMethod<f_t>>;

  // fast presolvers
  presolver.addPresolveMethod(uptr(new papilo::SingletonCols<f_t>()));
  presolver.addPresolveMethod(uptr(new papilo::CoefficientStrengthening<f_t>()));
  presolver.addPresolveMethod(uptr(new papilo::ConstraintPropagation<f_t>()));

  // medium presolvers
  presolver.addPresolveMethod(uptr(new papilo::FixContinuous<f_t>()));
  presolver.addPresolveMethod(uptr(new papilo::SimpleProbing<f_t>()));
  presolver.addPresolveMethod(uptr(new papilo::ParallelRowDetection<f_t>()));
  presolver.addPresolveMethod(uptr(new papilo::ParallelColDetection<f_t>()));
  // FIXME: Postsolve fails with this method
  // presolver.addPresolveMethod(uptr(new papilo::SingletonStuffing<f_t>()));
  presolver.addPresolveMethod(uptr(new papilo::DualFix<f_t>()));
  presolver.addPresolveMethod(uptr(new papilo::SimplifyInequalities<f_t>()));

  // exhaustive presolvers
  presolver.addPresolveMethod(uptr(new papilo::ImplIntDetection<f_t>()));
  presolver.addPresolveMethod(uptr(new papilo::DominatedCols<f_t>()));
  presolver.addPresolveMethod(uptr(new papilo::Probing<f_t>()));

  // if (category == problem_category_t::MIP) {
  presolver.addPresolveMethod(uptr(new papilo::DualInfer<f_t>));
  presolver.addPresolveMethod(uptr(new papilo::SimpleSubstitution<f_t>()));
  presolver.addPresolveMethod(uptr(new papilo::Sparsify<f_t>()));
  presolver.addPresolveMethod(uptr(new papilo::Substitution<f_t>()));
  // }
}

template <typename f_t>
void set_presolve_options(papilo::Presolve<f_t>& presolver,
                          problem_category_t category,
                          f_t absolute_tolerance,
                          double time_limit)
{
  presolver.getPresolveOptions().tlim    = time_limit;
  presolver.getPresolveOptions().epsilon = absolute_tolerance;
  presolver.getPresolveOptions().feastol = absolute_tolerance;
  // if (category == problem_category_t::LP) {
  //   presolver.getPresolveOptions().componentsmaxint = -1;
  //   presolver.getPresolveOptions().detectlindep     = 0;
  // }
}

template <typename i_t, typename f_t>
optimization_problem_t<i_t, f_t> third_party_presolve_t<i_t, f_t>::apply(
  optimization_problem_t<i_t, f_t> const& op_problem,
  problem_category_t category,
  f_t absolute_tolerance,
  double time_limit)
{
  papilo::Problem<f_t> papilo_problem = build_papilo_problem(op_problem);

  CUOPT_LOG_INFO("Unpresolved problem:: Num variables: %d Num constraints: %d, NNZ: %d",
                 papilo_problem.getNCols(),
                 papilo_problem.getNRows(),
                 papilo_problem.getConstraintMatrix().getNnz());

  papilo::Presolve<f_t> presolver;
  set_presolve_methods<f_t>(presolver, category);
  set_presolve_options<f_t>(presolver, category, absolute_tolerance, time_limit);

  auto result = presolver.apply(papilo_problem);
  if (result.status == papilo::PresolveStatus::kInfeasible) {
    return optimization_problem_t<i_t, f_t>(op_problem.get_handle_ptr());
  }

  post_solve_storage_ = result.postsolve;

  check_presolve_status(result.status);
  CUOPT_LOG_INFO("Presolved problem:: Num variables: %d Num constraints: %d, NNZ: %d",
                 papilo_problem.getNCols(),
                 papilo_problem.getNRows(),
                 papilo_problem.getConstraintMatrix().getNnz());

  return build_optimization_problem<i_t, f_t>(papilo_problem, op_problem.get_handle_ptr());
}

template <typename i_t, typename f_t>
void third_party_presolve_t<i_t, f_t>::undo(rmm::device_uvector<f_t>& primal_solution,
                                            rmm::device_uvector<f_t>& dual_solution,
                                            rmm::device_uvector<f_t>& reduced_costs,
                                            problem_category_t category,
                                            rmm::cuda_stream_view stream_view)
{
  auto primal_sol_vec_h    = cuopt::host_copy(primal_solution, stream_view);
  auto dual_sol_vec_h      = cuopt::host_copy(dual_solution, stream_view);
  auto reduced_costs_vec_h = cuopt::host_copy(reduced_costs, stream_view);

  papilo::Solution<f_t> reduced_sol(primal_sol_vec_h);
  papilo::Solution<f_t> full_sol;
  // if (category == problem_category_t::LP) {
  //   reduced_sol.type         = papilo::SolutionType::kPrimalDual;
  //   reduced_sol.dual         = dual_sol_vec_h;
  //   reduced_sol.reducedCosts = reduced_costs_vec_h;
  // }

  papilo::Message Msg{};
  Msg.setVerbosityLevel(papilo::VerbosityLevel::kQuiet);
  papilo::Postsolve<f_t> post_solver{Msg, post_solve_storage_.getNum()};

  bool is_optimal = false;
  auto status     = post_solver.undo(reduced_sol, full_sol, post_solve_storage_, is_optimal);
  check_postsolve_status(status);

  primal_solution.resize(full_sol.primal.size(), stream_view);
  dual_solution.resize(full_sol.primal.size(), stream_view);
  reduced_costs.resize(full_sol.primal.size(), stream_view);
  raft::copy(primal_solution.data(), full_sol.primal.data(), full_sol.primal.size(), stream_view);
  thrust::fill(rmm::exec_policy(stream_view),
               dual_solution.data(),
               dual_solution.data() + dual_solution.size(),
               std::numeric_limits<f_t>::signaling_NaN());
  thrust::fill(rmm::exec_policy(stream_view),
               reduced_costs.data(),
               reduced_costs.data() + reduced_costs.size(),
               std::numeric_limits<f_t>::signaling_NaN());
}

#if MIP_INSTANTIATE_FLOAT
template class third_party_presolve_t<int, float>;
#endif

#if MIP_INSTANTIATE_DOUBLE
template class third_party_presolve_t<int, double>;
#endif

}  // namespace cuopt::linear_programming::detail
