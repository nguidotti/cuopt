/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <cstdio>

#include <utilities/common_utils.hpp>

#include <gtest/gtest.h>

#include <dual_simplex/presolve.hpp>
#include <dual_simplex/solve.hpp>
#include <dual_simplex/user_problem.hpp>
#include <math_optimization/tic_toc.hpp>

#include <cuopt/mathematical_optimization/io/parser.hpp>
#include <utilities/logger.hpp>

namespace cuopt::mathematical_optimization::simplex::test {

TEST(dual_simplex, chess_set)
{
  cuopt::init_logger_t log("", true);
  namespace simplex = cuopt::mathematical_optimization::simplex;
  raft::handle_t handle{};
  simplex::user_problem_t<int, double> user_problem(&handle);
  // maximize   5*xs + 20*xl
  // subject to  1*xs +  3*xl <= 200
  //             3*xs +  2*xl <= 160
  constexpr int m  = 2;
  constexpr int n  = 2;
  constexpr int nz = 4;

  user_problem.num_rows = m;
  user_problem.num_cols = n;
  user_problem.objective.resize(n);
  user_problem.objective[0] = -5;
  user_problem.objective[1] = -20;
  user_problem.A.m          = m;
  user_problem.A.n          = n;
  user_problem.A.nz_max     = nz;
  user_problem.A.reallocate(nz);
  user_problem.A.col_start.resize(n + 1);
  user_problem.A.col_start[0] = 0;
  user_problem.A.col_start[1] = 2;
  user_problem.A.col_start[2] = 4;
  user_problem.A.i[0]         = 0;
  user_problem.A.x[0]         = 1.0;
  user_problem.A.i[1]         = 1;
  user_problem.A.x[1]         = 3.0;
  user_problem.A.i[2]         = 0;
  user_problem.A.x[2]         = 3.0;
  user_problem.A.i[3]         = 1;
  user_problem.A.x[3]         = 2.0;
  user_problem.rhs.resize(m);
  user_problem.rhs[0] = 200;
  user_problem.rhs[1] = 160;
  user_problem.row_sense.resize(m);
  user_problem.row_sense[0] = 'L';
  user_problem.row_sense[1] = 'L';
  user_problem.lower.resize(n);
  user_problem.lower[0] = 0;
  user_problem.lower[1] = 0.0;
  user_problem.upper.resize(n);
  user_problem.upper[0]       = inf;
  user_problem.upper[1]       = inf;
  user_problem.num_range_rows = 0;
  user_problem.problem_name   = "chess set";
  user_problem.row_names.resize(m);
  user_problem.row_names[0] = "boxwood";
  user_problem.row_names[1] = "lathe hours";
  user_problem.col_names.resize(n);
  user_problem.col_names[0] = "xs";
  user_problem.col_names[1] = "xl";
  user_problem.obj_constant = 0.0;
  user_problem.var_types.resize(n);
  user_problem.var_types[0] = simplex::variable_type_t::CONTINUOUS;
  user_problem.var_types[1] = simplex::variable_type_t::CONTINUOUS;

  simplex::simplex_solver_settings_t<int, double> settings;
  simplex::lp_solution_t<int, double> solution(user_problem.num_rows, user_problem.num_cols);
  EXPECT_EQ((simplex::solve_linear_program(user_problem, settings, solution)),
            simplex::lp_status_t::OPTIMAL);
  const double objective = -solution.objective;
  EXPECT_NEAR(objective, 1333.33, 1e-2);
  EXPECT_NEAR(solution.x[0], 0.0, 1e-6);
  EXPECT_NEAR(solution.x[1], 66.6667, 1e-3);

  user_problem.var_types[0] = simplex::variable_type_t::INTEGER;
  user_problem.var_types[1] = simplex::variable_type_t::INTEGER;

  EXPECT_EQ((simplex::solve(user_problem, settings, solution.x)), 0);
}

TEST(dual_simplex, burglar)
{
  cuopt::init_logger_t log("", true);
  constexpr int num_items     = 8;
  constexpr double max_weight = 102;

  std::vector<double> value({15, 100, 90, 60, 40, 15, 10, 1});
  std::vector<double> weight({2, 20, 20, 30, 40, 30, 60, 10});

  // maximize  sum_i value[i] * take[i]
  //           sum_i weight[i] * take[i] <= max_weight
  //           take[i] binary for all i

  raft::handle_t handle{};
  cuopt::mathematical_optimization::simplex::user_problem_t<int, double> user_problem(&handle);
  constexpr int m  = 1;
  constexpr int n  = num_items;
  constexpr int nz = num_items;

  user_problem.num_rows = m;
  user_problem.num_cols = n;
  user_problem.objective.resize(n);
  for (int j = 0; j < num_items; ++j) {
    user_problem.objective[j] = -value[j];
  }
  user_problem.A.m      = m;
  user_problem.A.n      = n;
  user_problem.A.nz_max = nz;
  user_problem.A.reallocate(nz);
  user_problem.A.col_start.resize(n + 1);
  for (int j = 0; j < num_items; ++j) {
    user_problem.A.col_start[j] = j;
    user_problem.A.i[j]         = 0;
    user_problem.A.x[j]         = weight[j];
  }
  user_problem.A.col_start[n] = nz;
  user_problem.rhs.resize(m);
  user_problem.rhs[0] = max_weight;
  user_problem.row_sense.resize(m);
  user_problem.row_sense[0] = 'L';
  user_problem.lower.resize(n);
  user_problem.upper.resize(n);
  for (int j = 0; j < num_items; ++j) {
    user_problem.lower[j] = 0.0;
    user_problem.upper[j] = 1.0;
  }
  user_problem.num_range_rows = 0;
  user_problem.problem_name   = "burglar";
  user_problem.row_names.resize(m);
  user_problem.row_names[0] = "weight restriction";
  user_problem.col_names.resize(n);
  for (int j = 0; j < num_items; ++j) {
    user_problem.col_names[j] = "x";
  }
  user_problem.obj_constant = 0.0;
  user_problem.var_types.resize(n);
  for (int j = 0; j < num_items; ++j) {
    user_problem.var_types[j] = cuopt::mathematical_optimization::simplex::variable_type_t::INTEGER;
  }

  cuopt::mathematical_optimization::simplex::simplex_solver_settings_t<int, double> settings;
  std::vector<double> solution(num_items);
  EXPECT_EQ((cuopt::mathematical_optimization::simplex::solve(user_problem, settings, solution)),
            0);
  double objective = 0.0;
  for (int j = 0; j < num_items; ++j) {
    objective += value[j] * solution[j];
  }
  EXPECT_NEAR(objective, 280, 1e-6);
  EXPECT_NEAR(solution[0], 1, 1e-6);
  EXPECT_NEAR(solution[1], 1, 1e-6);
  EXPECT_NEAR(solution[2], 1, 1e-6);
  EXPECT_NEAR(solution[3], 1, 1e-6);
  EXPECT_NEAR(solution[5], 1, 1e-6);
}

TEST(dual_simplex, empty_columns)
{
  cuopt::init_logger_t log("", true);
  // Same as burglar problem above but with an empty column inserted
  constexpr int num_items     = 9;
  constexpr double max_weight = 102;

  std::vector<double> value({15, 100, 90, 0, 60, 40, 15, 10, 1});
  std::vector<double> weight({2, 20, 20, 0, 30, 40, 30, 60, 10});

  // maximize  sum_i value[i] * take[i]
  //           sum_i weight[i] * take[i] <= max_weight
  //           take[i] binary for all i

  raft::handle_t handle{};
  cuopt::mathematical_optimization::simplex::user_problem_t<int, double> user_problem(&handle);
  constexpr int m  = 1;
  constexpr int n  = num_items;
  constexpr int nz = num_items - 1;

  user_problem.num_rows = m;
  user_problem.num_cols = n;
  user_problem.objective.resize(n);
  for (int j = 0; j < num_items; ++j) {
    user_problem.objective[j] = -value[j];
  }
  user_problem.A.m      = m;
  user_problem.A.n      = n;
  user_problem.A.nz_max = nz;
  user_problem.A.reallocate(nz);
  user_problem.A.col_start.resize(n + 1);
  int nnz = 0;
  for (int j = 0; j < num_items; ++j) {
    user_problem.A.col_start[j] = nnz;
    if (weight[j] > 0) {
      user_problem.A.i[nnz] = 0;
      user_problem.A.x[nnz] = weight[j];
      nnz++;
    }
  }
  user_problem.A.col_start[n] = nnz;
  user_problem.rhs.resize(m);
  user_problem.rhs[0] = max_weight;
  user_problem.row_sense.resize(m);
  user_problem.row_sense[0] = 'L';
  user_problem.lower.resize(n);
  user_problem.upper.resize(n);
  for (int j = 0; j < num_items; ++j) {
    user_problem.lower[j] = 0.0;
    user_problem.upper[j] = 1.0;
  }
  user_problem.num_range_rows = 0;
  user_problem.problem_name   = "burglar";
  user_problem.row_names.resize(m);
  user_problem.row_names[0] = "weight restriction";
  user_problem.col_names.resize(n);
  for (int j = 0; j < num_items; ++j) {
    user_problem.col_names[j] = "x";
  }
  user_problem.obj_constant = 0.0;
  user_problem.var_types.resize(n);
  for (int j = 0; j < num_items; ++j) {
    user_problem.var_types[j] =
      cuopt::mathematical_optimization::simplex::variable_type_t::CONTINUOUS;
  }

  cuopt::mathematical_optimization::simplex::simplex_solver_settings_t<int, double> settings;

  cuopt::mathematical_optimization::simplex::lp_solution_t<int, double> solution(
    user_problem.num_rows, user_problem.num_cols);
  EXPECT_EQ((cuopt::mathematical_optimization::simplex::solve_linear_program(
              user_problem, settings, solution)),
            cuopt::mathematical_optimization::simplex::lp_status_t::OPTIMAL);
  double objective = 0.0;
  for (int j = 0; j < num_items; ++j) {
    objective += value[j] * solution.x[j];
  }
  EXPECT_NEAR(objective, 295, 1e-6);
  EXPECT_NEAR(solution.x[0], 1, 1e-6);
  EXPECT_NEAR(solution.x[1], 1, 1e-6);
  EXPECT_NEAR(solution.x[2], 1, 1e-6);
  EXPECT_NEAR(solution.x[3], 0, 1e-6);
  EXPECT_NEAR(solution.x[4], 1, 1e-6);
  EXPECT_NEAR(solution.x[5], 0.75, 1e-6);
  EXPECT_NEAR(solution.x[6], 0, 1e-6);
  EXPECT_NEAR(solution.x[7], 0, 1e-6);
  EXPECT_NEAR(solution.x[8], 0, 1e-6);
}

TEST(dual_simplex, dual_variable_greater_than)
{
  cuopt::init_logger_t log("", true);
  // minimize   3*x0 + 2 * x1
  // subject to  x0 + x1  >= 1
  //             x0 + 2x1 >= 3
  //             x0, x1 >= 0

  raft::handle_t handle{};
  cuopt::mathematical_optimization::simplex::user_problem_t<int, double> user_problem(&handle);
  constexpr int m  = 2;
  constexpr int n  = 2;
  constexpr int nz = 4;

  user_problem.num_rows = m;
  user_problem.num_cols = n;
  user_problem.objective.resize(n);
  user_problem.objective[0] = 3.0;
  user_problem.objective[1] = 2.0;
  user_problem.A.m          = m;
  user_problem.A.n          = n;
  user_problem.A.nz_max     = nz;
  user_problem.A.reallocate(nz);
  user_problem.A.col_start.resize(n + 1);
  user_problem.A.col_start[0] = 0;  // x0 start
  user_problem.A.col_start[1] = 2;
  user_problem.A.col_start[2] = 4;

  int nnz                 = 0;
  user_problem.A.i[nnz]   = 0;
  user_problem.A.x[nnz++] = 1.0;
  user_problem.A.i[nnz]   = 1;
  user_problem.A.x[nnz++] = 1.0;
  user_problem.A.i[nnz]   = 0;
  user_problem.A.x[nnz++] = 1.0;
  user_problem.A.i[nnz]   = 1;
  user_problem.A.x[nnz++] = 2.0;
  user_problem.A.print_matrix();
  EXPECT_EQ(nnz, nz);

  user_problem.rhs.resize(m);
  user_problem.rhs[0] = 1.0;
  user_problem.rhs[1] = 3.0;

  user_problem.row_sense.resize(m);
  user_problem.row_sense[0] = 'G';
  user_problem.row_sense[1] = 'G';

  user_problem.lower.resize(n);
  user_problem.lower[0] = 0.0;
  user_problem.lower[1] = 0.0;

  user_problem.upper.resize(n);
  user_problem.upper[0] = inf;
  user_problem.upper[1] = inf;

  user_problem.num_range_rows = 0;
  user_problem.problem_name   = "dual_variable_greater_than";

  simplex::simplex_solver_settings_t<int, double> settings;
  simplex::lp_solution_t<int, double> solution(user_problem.num_rows, user_problem.num_cols);
  EXPECT_EQ((simplex::solve_linear_program(user_problem, settings, solution)),
            simplex::lp_status_t::OPTIMAL);
  EXPECT_NEAR(solution.objective, 3.0, 1e-6);
  EXPECT_NEAR(solution.x[0], 0.0, 1e-6);
  EXPECT_NEAR(solution.x[1], 1.5, 1e-6);
  EXPECT_NEAR(solution.y[0], 0.0, 1e-6);
  EXPECT_NEAR(solution.y[1], 1.0, 1e-6);
  EXPECT_NEAR(solution.z[0], 2.0, 1e-6);
  EXPECT_NEAR(solution.z[1], 0.0, 1e-6);
}

// Round-trip a MIP through convert_user_problem (range form -> simplex standard
// form, appending one slack/artificial column per row) and
// convert_simplex_problem (the inverse: drop the slacks and recover the row
// bounds). The problem exercises every row type: '<=', '==', '>=', and a range
// row.
//
// '>=' rows are folded into '<=' rows by negating their coefficients/rhs in the
// forward pass, so the inverse recovers them as the equivalent negated '<=' row
// rather than the original '>='. The recovered problem is therefore feasibly
// equivalent but not textually identical. We assert two things:
//   1. the directly-predictable recovered fields (sizes, row_sense, rhs, range,
//      objective, bounds, var_types), and
//   2. the round-trip invariant: re-running the forward conversion on the
//      recovered problem reproduces the original standard-form problem exactly.
TEST(dual_simplex, convert_simplex_problem_mip_round_trip)
{
  cuopt::init_logger_t log("", true);
  raft::handle_t handle{};

  // minimize  x0 + 2 x1 + 3 x2
  // subject to 2 x0 + x1        <= 8   (row 0, 'L')
  //              x0        + x2   = 4   (row 1, 'E')
  //                   x1   + x2  >= 3   (row 2, 'G')
  //            1 <=  x0 + x1     <= 7   (row 3, range row stored as 'E' + range)
  //            x0 integer in [0, 10]
  //            x1 continuous in [0, inf)
  //            x2 integer in [0, 5]
  user_problem_t<int, double> original(&handle);
  constexpr int m  = 4;
  constexpr int n  = 3;
  constexpr int nz = 8;

  original.num_rows = m;
  original.num_cols = n;
  original.objective.assign({1.0, 2.0, 3.0});
  original.A.m      = m;
  original.A.n      = n;
  original.A.nz_max = nz;
  original.A.reallocate(nz);
  // column 0 (x0): rows {0, 1, 3} -> {2, 1, 1}
  // column 1 (x1): rows {0, 2, 3} -> {1, 1, 1}
  // column 2 (x2): rows {1, 2}    -> {1, 1}
  original.A.col_start.assign({0, 3, 6, 8});
  original.A.i = {0, 1, 3, 0, 2, 3, 1, 2};
  original.A.x = {2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
  original.rhs.assign({8.0, 4.0, 3.0, 1.0});
  original.row_sense.assign({'L', 'E', 'G', 'E'});
  original.lower.assign({0.0, 0.0, 0.0});
  original.upper.assign({10.0, std::numeric_limits<double>::infinity(), 5.0});
  // row 3 is the range row 1 <= a^T x <= 7: stored as an 'E' range row, which
  // convert_range_rows reads as [rhs, rhs + range] = [1, 7].
  original.range_rows.assign({3});
  original.range_value.assign({6.0});
  original.num_range_rows = 1;
  original.obj_constant   = 0.0;
  original.var_types      = {
    variable_type_t::INTEGER, variable_type_t::CONTINUOUS, variable_type_t::INTEGER};

  // Forward: range form -> simplex standard form.
  simplex_solver_settings_t<int, double> settings;
  lp_problem_t<int, double> simplex_problem(
    &handle, original.num_rows, original.num_cols, original.A.col_start[original.A.n]);
  std::vector<int> new_slacks;
  dualize_info_t<int, double> dualize_info;
  convert_user_problem(original, settings, simplex_problem, new_slacks, dualize_info);

  // Each row gets exactly one slack/artificial column, appended after the
  // structural columns.
  EXPECT_EQ(new_slacks.size(), static_cast<size_t>(m));
  EXPECT_EQ(simplex_problem.num_cols, n + m);

  // var_types spans the full simplex problem; the appended columns are
  // continuous (mirrors full_variable_types in branch_and_bound).
  std::vector<simplex::variable_type_t> var_types = original.var_types;
  var_types.resize(simplex_problem.num_cols, simplex::variable_type_t::CONTINUOUS);

  // Inverse: simplex standard form -> range form.
  user_problem_t<int, double> recovered(&handle);
  convert_simplex_problem(simplex_problem, var_types, settings, new_slacks, recovered);

  // (1) Directly-predictable recovered fields.
  EXPECT_EQ(recovered.num_rows, m);
  EXPECT_EQ(recovered.num_cols, n);
  // row 0 '<=' -> 'L' rhs 8; row 1 '==' -> 'E' rhs 4;
  // row 2 '>=' -> recovered as the negated 'L' (-x1 - x2 <= -3);
  // row 3 range [1, 7] -> canonical 'E' range row: rhs 1 with range 6.
  EXPECT_EQ(recovered.row_sense, (std::vector<char>{'L', 'E', 'L', 'E'}));
  EXPECT_EQ(recovered.rhs, (std::vector<double>{8.0, 4.0, -3.0, 1.0}));
  EXPECT_EQ(recovered.num_range_rows, 1);
  EXPECT_EQ(recovered.range_rows, (std::vector<int>{3}));
  EXPECT_EQ(recovered.range_value, (std::vector<double>{6.0}));
  // Column data (objective / bounds / types) carries over unchanged.
  EXPECT_EQ(recovered.objective, original.objective);
  EXPECT_EQ(recovered.lower, original.lower);
  EXPECT_EQ(recovered.upper, original.upper);
  EXPECT_EQ(recovered.var_types, original.var_types);

  // (2) Round-trip invariant: converting the recovered problem forward again
  // must reproduce the original standard-form problem exactly.
  lp_problem_t<int, double> simplex_again(
    &handle, recovered.num_rows, recovered.num_cols, recovered.A.col_start[recovered.A.n]);
  std::vector<int> new_slacks_again;
  dualize_info_t<int, double> dualize_info_again;
  convert_user_problem(recovered, settings, simplex_again, new_slacks_again, dualize_info_again);

  EXPECT_EQ(simplex_again.num_rows, simplex_problem.num_rows);
  EXPECT_EQ(simplex_again.num_cols, simplex_problem.num_cols);
  EXPECT_EQ(simplex_again.A.col_start, simplex_problem.A.col_start);
  EXPECT_EQ(simplex_again.A.i, simplex_problem.A.i);
  EXPECT_EQ(simplex_again.A.x, simplex_problem.A.x);
  EXPECT_EQ(simplex_again.rhs, simplex_problem.rhs);
  EXPECT_EQ(simplex_again.lower, simplex_problem.lower);
  EXPECT_EQ(simplex_again.upper, simplex_problem.upper);
  EXPECT_EQ(new_slacks_again, new_slacks);
}

}  // namespace cuopt::mathematical_optimization::simplex::test
