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

#include <cuopt/error.hpp>
#include <mip/mip_constants.hpp>
#include <mip/presolve/gf2_presolve.hpp>
#include <mip/presolve/third_party_presolve.hpp>
#include <utilities/logger.hpp>
#include <utilities/timer.hpp>

#include <raft/common/nvtx.hpp>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstringop-overflow"  // ignore boost error for pip wheel build
#include <papilo/core/Presolve.hpp>
#include <papilo/core/ProblemBuilder.hpp>
#pragma GCC diagnostic pop

namespace cuopt::linear_programming::detail {

static papilo::PostsolveStorage<double> post_solve_storage_;
static bool maximize_ = false;

template <typename i_t, typename f_t>
papilo::Problem<f_t> build_papilo_problem(const optimization_problem_t<i_t, f_t>& op_problem,
                                          problem_category_t category)
{
  raft::common::nvtx::range fun_scope("Build papilo problem");
  // Build papilo problem from optimization problem
  papilo::ProblemBuilder<f_t> builder;

  // Get problem dimensions
  const i_t num_cols = op_problem.get_n_variables();
  const i_t num_rows = op_problem.get_n_constraints();
  const i_t nnz      = op_problem.get_nnz();

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
  std::vector<f_t> h_coefficients(coefficients.size());
  auto stream_view = op_problem.get_handle_ptr()->get_stream();
  raft::copy(h_coefficients.data(), coefficients.data(), coefficients.size(), stream_view);
  std::vector<i_t> h_offsets(offsets.size());
  raft::copy(h_offsets.data(), offsets.data(), offsets.size(), stream_view);
  std::vector<i_t> h_variables(variables.size());
  raft::copy(h_variables.data(), variables.data(), variables.size(), stream_view);
  std::vector<f_t> h_obj_coeffs(obj_coeffs.size());
  raft::copy(h_obj_coeffs.data(), obj_coeffs.data(), obj_coeffs.size(), stream_view);
  std::vector<f_t> h_var_lb(var_lb.size());
  raft::copy(h_var_lb.data(), var_lb.data(), var_lb.size(), stream_view);
  std::vector<f_t> h_var_ub(var_ub.size());
  raft::copy(h_var_ub.data(), var_ub.data(), var_ub.size(), stream_view);
  std::vector<f_t> h_bounds(bounds.size());
  raft::copy(h_bounds.data(), bounds.data(), bounds.size(), stream_view);
  std::vector<char> h_row_types(row_types.size());
  raft::copy(h_row_types.data(), row_types.data(), row_types.size(), stream_view);
  std::vector<f_t> h_constr_lb(constr_lb.size());
  raft::copy(h_constr_lb.data(), constr_lb.data(), constr_lb.size(), stream_view);
  std::vector<f_t> h_constr_ub(constr_ub.size());
  raft::copy(h_constr_ub.data(), constr_ub.data(), constr_ub.size(), stream_view);
  std::vector<var_t> h_var_types(var_types.size());
  raft::copy(h_var_types.data(), var_types.data(), var_types.size(), stream_view);

  maximize_ = op_problem.get_sense();
  if (maximize_) {
    for (size_t i = 0; i < h_obj_coeffs.size(); ++i) {
      h_obj_coeffs[i] = -h_obj_coeffs[i];
    }
  }

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
  builder.setObjOffset(maximize_ ? -op_problem.get_objective_offset()
                                 : op_problem.get_objective_offset());

  if (!h_var_lb.empty() && !h_var_ub.empty()) {
    builder.setColLbAll(h_var_lb);
    builder.setColUbAll(h_var_ub);
    if (op_problem.get_variable_names().size() == h_var_lb.size()) {
      builder.setColNameAll(op_problem.get_variable_names());
    }
  }

  for (size_t i = 0; i < h_var_types.size(); ++i) {
    builder.setColIntegral(i, h_var_types[i] == var_t::INTEGER);
  }

  if (!h_constr_lb.empty() && !h_constr_ub.empty()) {
    builder.setRowLhsAll(h_constr_lb);
    builder.setRowRhsAll(h_constr_ub);
  }

  std::vector<papilo::RowFlags> h_row_flags(h_constr_lb.size());
  std::vector<std::tuple<i_t, i_t, f_t>> h_entries;
  // Add constraints row by row
  for (size_t i = 0; i < h_constr_lb.size(); ++i) {
    // Get row entries
    i_t row_start   = h_offsets[i];
    i_t row_end     = h_offsets[i + 1];
    i_t num_entries = row_end - row_start;
    for (size_t j = 0; j < num_entries; ++j) {
      h_entries.push_back(
        std::make_tuple(i, h_variables[row_start + j], h_coefficients[row_start + j]));
    }

    if (h_constr_lb[i] == -std::numeric_limits<f_t>::infinity()) {
      h_row_flags[i].set(papilo::RowFlag::kLhsInf);
    } else {
      h_row_flags[i].unset(papilo::RowFlag::kLhsInf);
    }
    if (h_constr_ub[i] == std::numeric_limits<f_t>::infinity()) {
      h_row_flags[i].set(papilo::RowFlag::kRhsInf);
    } else {
      h_row_flags[i].unset(papilo::RowFlag::kRhsInf);
    }

    if (h_constr_lb[i] == -std::numeric_limits<f_t>::infinity()) { h_constr_lb[i] = 0; }
    if (h_constr_ub[i] == std::numeric_limits<f_t>::infinity()) { h_constr_ub[i] = 0; }
  }

  for (size_t i = 0; i < h_var_lb.size(); ++i) {
    builder.setColLbInf(i, h_var_lb[i] == -std::numeric_limits<f_t>::infinity());
    builder.setColUbInf(i, h_var_ub[i] == std::numeric_limits<f_t>::infinity());
    if (h_var_lb[i] == -std::numeric_limits<f_t>::infinity()) { builder.setColLb(i, 0); }
    if (h_var_ub[i] == std::numeric_limits<f_t>::infinity()) { builder.setColUb(i, 0); }
  }

  auto problem = builder.build();

  if (h_entries.size()) {
    auto constexpr const sorted_entries = true;
    // MIP reductions like clique merging and substituition require more fillin
    const double spare_ratio      = category == problem_category_t::MIP ? 4.0 : 2.0;
    const int min_inter_row_space = category == problem_category_t::MIP ? 30 : 4;
    auto csr_storage              = papilo::SparseStorage<f_t>(
      h_entries, num_rows, num_cols, sorted_entries, spare_ratio, min_inter_row_space);
    problem.setConstraintMatrix(csr_storage, h_constr_lb, h_constr_ub, h_row_flags);

    papilo::ConstraintMatrix<f_t>& matrix = problem.getConstraintMatrix();
    for (int i = 0; i < problem.getNRows(); ++i) {
      papilo::RowFlags rowFlag = matrix.getRowFlags()[i];
      if (!rowFlag.test(papilo::RowFlag::kRhsInf) && !rowFlag.test(papilo::RowFlag::kLhsInf) &&
          matrix.getLeftHandSides()[i] == matrix.getRightHandSides()[i])
        matrix.getRowFlags()[i].set(papilo::RowFlag::kEquation);
    }
  }

  return problem;
}

template <typename i_t, typename f_t>
optimization_problem_t<i_t, f_t> build_optimization_problem(
  papilo::Problem<f_t> const& papilo_problem, raft::handle_t const* handle_ptr)
{
  raft::common::nvtx::range fun_scope("Build optimization problem");
  optimization_problem_t<i_t, f_t> op_problem(handle_ptr);

  auto obj = papilo_problem.getObjective();
  op_problem.set_objective_offset(maximize_ ? -obj.offset : obj.offset);
  op_problem.set_maximize(maximize_);

  if (papilo_problem.getNRows() == 0 && papilo_problem.getNCols() == 0) {
    // FIXME: Shouldn't need to set offsets
    std::vector<i_t> h_offsets{0};
    std::vector<i_t> h_indices{};
    std::vector<f_t> h_values{};
    op_problem.set_csr_constraint_matrix(h_values.data(),
                                         h_values.size(),
                                         h_indices.data(),
                                         h_indices.size(),
                                         h_offsets.data(),
                                         h_offsets.size());

    return op_problem;
  }
  if (maximize_) {
    for (size_t i = 0; i < obj.coefficients.size(); ++i) {
      obj.coefficients[i] = -obj.coefficients[i];
    }
  }
  op_problem.set_objective_coefficients(obj.coefficients.data(), obj.coefficients.size());

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
      CUOPT_LOG_INFO("Presolve status: did not result in any changes");
      break;
    case papilo::PresolveStatus::kReduced:
      CUOPT_LOG_INFO("Presolve status: reduced the problem");
      break;
    case papilo::PresolveStatus::kUnbndOrInfeas:
      CUOPT_LOG_INFO("Presolve status: found an unbounded or infeasible problem");
      break;
    case papilo::PresolveStatus::kInfeasible:
      CUOPT_LOG_INFO("Presolve status: found an infeasible problem");
      break;
    case papilo::PresolveStatus::kUnbounded:
      CUOPT_LOG_INFO("Presolve status: found an unbounded problem");
      break;
  }
}

void check_postsolve_status(const papilo::PostsolveStatus& status)
{
  switch (status) {
    case papilo::PostsolveStatus::kOk: CUOPT_LOG_INFO("Post-solve status: succeeded"); break;
    case papilo::PostsolveStatus::kFailed:
      CUOPT_LOG_INFO(
        "Post-solve status: Post solved solution violates constraints. This is most likely due to "
        "different tolerances.");
      break;
  }
}

template <typename f_t>
void set_presolve_methods(papilo::Presolve<f_t>& presolver,
                          problem_category_t category,
                          bool dual_postsolve)
{
  using uptr = std::unique_ptr<papilo::PresolveMethod<f_t>>;

  if (category == problem_category_t::MIP) {
    // cuOpt custom GF2 presolver
    presolver.addPresolveMethod(uptr(new cuopt::linear_programming::detail::GF2Presolve<f_t>()));
  }
  // fast presolvers
  presolver.addPresolveMethod(uptr(new papilo::SingletonCols<f_t>()));
  presolver.addPresolveMethod(uptr(new papilo::CoefficientStrengthening<f_t>()));
  presolver.addPresolveMethod(uptr(new papilo::ConstraintPropagation<f_t>()));

  // medium presolvers
  presolver.addPresolveMethod(uptr(new papilo::FixContinuous<f_t>()));
  presolver.addPresolveMethod(uptr(new papilo::SimpleProbing<f_t>()));
  presolver.addPresolveMethod(uptr(new papilo::ParallelRowDetection<f_t>()));
  presolver.addPresolveMethod(uptr(new papilo::ParallelColDetection<f_t>()));

  presolver.addPresolveMethod(uptr(new papilo::SingletonStuffing<f_t>()));
  presolver.addPresolveMethod(uptr(new papilo::DualFix<f_t>()));
  presolver.addPresolveMethod(uptr(new papilo::SimplifyInequalities<f_t>()));
  presolver.addPresolveMethod(uptr(new papilo::CliqueMerging<f_t>()));

  // exhaustive presolvers
  presolver.addPresolveMethod(uptr(new papilo::ImplIntDetection<f_t>()));
  presolver.addPresolveMethod(uptr(new papilo::DominatedCols<f_t>()));
  presolver.addPresolveMethod(uptr(new papilo::Probing<f_t>()));

  if (!dual_postsolve) {
    presolver.addPresolveMethod(uptr(new papilo::DualInfer<f_t>()));
    presolver.addPresolveMethod(uptr(new papilo::SimpleSubstitution<f_t>()));
    presolver.addPresolveMethod(uptr(new papilo::Sparsify<f_t>()));
    presolver.addPresolveMethod(uptr(new papilo::Substitution<f_t>()));
  } else {
    CUOPT_LOG_INFO("Disabling the presolver methods that do not support dual postsolve");
  }
}

template <typename i_t, typename f_t>
void set_presolve_options(papilo::Presolve<f_t>& presolver,
                          problem_category_t category,
                          f_t absolute_tolerance,
                          f_t relative_tolerance,
                          double time_limit,
                          i_t num_cpu_threads)
{
  presolver.getPresolveOptions().tlim    = time_limit;
  presolver.getPresolveOptions().threads = num_cpu_threads;  //  user setting or  0 (automatic)
  presolver.getPresolveOptions().feastol = 1e-5;
}

template <typename f_t>
void set_presolve_parameters(papilo::Presolve<f_t>& presolver,
                             problem_category_t category,
                             int nrows,
                             int ncols)
{
  // It looks like a copy. But this copy has the pointers to relevant variables in papilo
  auto params = presolver.getParameters();
  if (category == problem_category_t::MIP) {
    // Papilo has work unit measurements for probing. Because of this when the first batch fails to
    // produce any reductions, the algorithm stops. To avoid stopping the algorithm, we set a
    // minimum badge size to a huge value. The time limit makes sure that we exit if it takes too
    // long
    int min_badgesize = std::max(ncols / 2, 32);
    params.setParameter("probing.minbadgesize", min_badgesize);
    params.setParameter("cliquemerging.enabled", true);
    params.setParameter("cliquemerging.maxcalls", 50);
  }
}

template <typename i_t, typename f_t>
std::optional<third_party_presolve_result_t<i_t, f_t>> third_party_presolve_t<i_t, f_t>::apply(
  optimization_problem_t<i_t, f_t> const& op_problem,
  problem_category_t category,
  bool dual_postsolve,
  f_t absolute_tolerance,
  f_t relative_tolerance,
  double time_limit,
  i_t num_cpu_threads)
{
  papilo::Problem<f_t> papilo_problem = build_papilo_problem(op_problem, category);

  CUOPT_LOG_INFO("Original problem: %d constraints, %d variables, %d nonzeros",
                 papilo_problem.getNRows(),
                 papilo_problem.getNCols(),
                 papilo_problem.getConstraintMatrix().getNnz());

  CUOPT_LOG_INFO("Calling Papilo presolver");
  if (category == problem_category_t::MIP) { dual_postsolve = false; }
  papilo::Presolve<f_t> presolver;
  set_presolve_methods<f_t>(presolver, category, dual_postsolve);
  set_presolve_options<i_t, f_t>(
    presolver, category, absolute_tolerance, relative_tolerance, time_limit, num_cpu_threads);
  set_presolve_parameters<f_t>(
    presolver, category, op_problem.get_n_constraints(), op_problem.get_n_variables());

  // Disable papilo logs
  presolver.setVerbosityLevel(papilo::VerbosityLevel::kQuiet);

  auto result = presolver.apply(papilo_problem);
  check_presolve_status(result.status);
  if (result.status == papilo::PresolveStatus::kInfeasible ||
      result.status == papilo::PresolveStatus::kUnbndOrInfeas) {
    return std::nullopt;
  }
  post_solve_storage_ = result.postsolve;
  CUOPT_LOG_INFO("Presolve removed: %d constraints, %d variables, %d nonzeros",
                 op_problem.get_n_constraints() - papilo_problem.getNRows(),
                 op_problem.get_n_variables() - papilo_problem.getNCols(),
                 op_problem.get_nnz() - papilo_problem.getConstraintMatrix().getNnz());
  CUOPT_LOG_INFO("Presolved problem: %d constraints, %d variables, %d nonzeros",
                 papilo_problem.getNRows(),
                 papilo_problem.getNCols(),
                 papilo_problem.getConstraintMatrix().getNnz());

  auto opt_problem =
    build_optimization_problem<i_t, f_t>(papilo_problem, op_problem.get_handle_ptr());
  auto col_flags = papilo_problem.getColFlags();
  std::vector<i_t> implied_integer_indices;
  for (size_t i = 0; i < col_flags.size(); i++) {
    if (col_flags[i].test(papilo::ColFlag::kImplInt)) implied_integer_indices.push_back(i);
  }

  return std::make_optional(
    third_party_presolve_result_t<i_t, f_t>{opt_problem, implied_integer_indices});
}

template <typename i_t, typename f_t>
void third_party_presolve_t<i_t, f_t>::undo(rmm::device_uvector<f_t>& primal_solution,
                                            rmm::device_uvector<f_t>& dual_solution,
                                            rmm::device_uvector<f_t>& reduced_costs,
                                            problem_category_t category,
                                            bool status_to_skip,
                                            rmm::cuda_stream_view stream_view)
{
  if (status_to_skip) { return; }
  std::vector<f_t> primal_sol_vec_h(primal_solution.size());
  raft::copy(primal_sol_vec_h.data(), primal_solution.data(), primal_solution.size(), stream_view);
  std::vector<f_t> dual_sol_vec_h(dual_solution.size());
  raft::copy(dual_sol_vec_h.data(), dual_solution.data(), dual_solution.size(), stream_view);
  std::vector<f_t> reduced_costs_vec_h(reduced_costs.size());
  raft::copy(reduced_costs_vec_h.data(), reduced_costs.data(), reduced_costs.size(), stream_view);

  papilo::Solution<f_t> reduced_sol(primal_sol_vec_h);
  papilo::Solution<f_t> full_sol;

  papilo::Message Msg{};
  Msg.setVerbosityLevel(papilo::VerbosityLevel::kQuiet);
  papilo::Postsolve<f_t> post_solver{Msg, post_solve_storage_.getNum()};

  bool is_optimal = false;
  auto status     = post_solver.undo(reduced_sol, full_sol, post_solve_storage_, is_optimal);
  check_postsolve_status(status);

  primal_solution.resize(full_sol.primal.size(), stream_view);
  dual_solution.resize(full_sol.dual.size(), stream_view);
  reduced_costs.resize(full_sol.reducedCosts.size(), stream_view);
  raft::copy(primal_solution.data(), full_sol.primal.data(), full_sol.primal.size(), stream_view);
  raft::copy(dual_solution.data(), full_sol.dual.data(), full_sol.dual.size(), stream_view);
  raft::copy(
    reduced_costs.data(), full_sol.reducedCosts.data(), full_sol.reducedCosts.size(), stream_view);
}

#if MIP_INSTANTIATE_FLOAT
template class third_party_presolve_t<int, float>;
#endif

#if MIP_INSTANTIATE_DOUBLE
template class third_party_presolve_t<int, double>;
#endif

}  // namespace cuopt::linear_programming::detail
