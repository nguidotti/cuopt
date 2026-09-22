/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "activated_capacity.hpp"

#include <mip_heuristics/mip_constants.hpp>
#include <utilities/logger.hpp>
#include <utilities/macros.cuh>

#include <algorithm>
#include <iterator>
#include <utility>
#include <vector>

// A group capacity row caps how many members of a group may be selected, but says nothing about
// whether the group is open. Where every member is gated by the same activation z, the cap is only
// available once z is paid for:
//
//     sum_{i in S} x_i - s <= K,  x_i <= z for all i in S,  s >= 0
//     =>  sum_{i in S} x_i - s <= K z.
//
// At z = 0 the gates force every x_i to 0 and the relaxing terms are non-positive, so the
// strengthened row reads 0 - s <= 0; at z = 1 it is the original row. Valid for integral z only,
// which is why this presolver is registered on the MIP path alone.

namespace cuopt::mathematical_optimization::mip {

template <typename f_t>
papilo::PresolveStatus ActivatedCapacity<f_t>::execute(
  const papilo::Problem<f_t>& problem,
  const papilo::ProblemUpdate<f_t>& problemUpdate,
  const papilo::Num<f_t>& num,
  papilo::Reductions<f_t>& reductions,
  const papilo::Timer& timer,
  int& reason_of_infeasibility)
{
  const auto& constraint_matrix = problem.getConstraintMatrix();
  const auto& lhs_values        = constraint_matrix.getLeftHandSides();
  const auto& rhs_values        = constraint_matrix.getRightHandSides();
  const auto& row_flags         = constraint_matrix.getRowFlags();
  const auto& domains           = problem.getVariableDomains();
  const auto& col_flags         = domains.flags;
  const auto& lower_bounds      = domains.lower_bounds;
  const auto& upper_bounds      = domains.upper_bounds;
  const auto& presolve_options  = problemUpdate.getPresolveOptions();

  const int num_rows = constraint_matrix.getNRows();

  auto is_free_binary = [&](int col) {
    const auto& flags = col_flags[col];
    return flags.test(papilo::ColFlag::kIntegral) && !flags.test(papilo::ColFlag::kLbInf) &&
           !flags.test(papilo::ColFlag::kUbInf) && !flags.test(papilo::ColFlag::kFixed) &&
           num.isZero(lower_bounds[col]) && num.isEq(upper_bounds[col], f_t{1});
  };

  // Orientation of a one-sided row, or 0 when the row is an equation, a range or free.
  // +1 means the stored row reads a.x <= side, -1 means a.x >= side; multiplying by the direction
  // puts it in <= form either way.
  auto orientation = [&](int row) {
    const auto& row_flag = row_flags[row];
    if (row_flag.test(papilo::RowFlag::kRedundant)) return 0;
    const bool lhs_infinite = row_flag.test(papilo::RowFlag::kLhsInf);
    const bool rhs_infinite = row_flag.test(papilo::RowFlag::kRhsInf);
    if (lhs_infinite == rhs_infinite) return 0;
    return lhs_infinite ? 1 : -1;
  };

  // Pass one: every two-term row x - z <= 0 over free binaries, as sorted (x, z) pairs.
  std::vector<std::pair<int, int>> gates;
  for (int row = 0; row < num_rows; ++row) {
    const int direction = orientation(row);
    if (direction == 0) continue;
    auto row_coefficients = constraint_matrix.getRowCoefficients(row);
    if (row_coefficients.getLength() != 2) continue;
    const f_t side = direction == 1 ? rhs_values[row] : lhs_values[row];
    if (!num.isZero(side)) continue;

    const int* indices = row_coefficients.getIndices();
    const f_t* values  = row_coefficients.getValues();
    int gated = -1, activation = -1;
    for (int j = 0; j < 2; ++j) {
      if (!is_free_binary(indices[j])) break;
      const f_t v = direction * values[j];
      if (num.isEq(v, f_t{1}))
        gated = indices[j];
      else if (num.isEq(v, f_t{-1}))
        activation = indices[j];
    }
    if (gated >= 0 && activation >= 0) gates.emplace_back(gated, activation);
  }
  std::sort(gates.begin(), gates.end());
  gates.erase(std::unique(gates.begin(), gates.end()), gates.end());

  auto gates_of = [&](int col) {
    const auto lo =
      std::lower_bound(gates.begin(), gates.end(), col, [](const std::pair<int, int>& g, int c) {
        return g.first < c;
      });
    const auto hi = std::upper_bound(
      lo, gates.end(), col, [](int c, const std::pair<int, int>& g) { return c < g.first; });
    return std::make_pair(lo, hi);
  };

  papilo::PresolveStatus status = papilo::PresolveStatus::kUnchanged;
  int rows_strengthened         = 0;
  std::vector<int> selections;
  std::vector<int> common;
  std::vector<int> candidates;
  std::vector<int> intersection;
  std::vector<int> support;

  // Pass two: the capacity rows themselves.
  for (int row = 0; row < num_rows && !gates.empty(); ++row) {
    if (reductions.size() >= presolve_options.max_reduction_seq) break;
    if (papilo::PresolveMethod<f_t>::is_interrupted(
          timer, presolve_options.tlim, presolve_options.early_exit_callback))
      break;

    const int direction = orientation(row);
    if (direction == 0) continue;
    auto row_coefficients = constraint_matrix.getRowCoefficients(row);
    const int len         = row_coefficients.getLength();
    if (len < 2 || len > ACTIVATED_CAPACITY_MAX_LEN) continue;

    const f_t side     = direction == 1 ? rhs_values[row] : lhs_values[row];
    const f_t capacity = direction * side;
    if (!num.isGT(capacity, f_t{0})) continue;

    const int* indices = row_coefficients.getIndices();
    const f_t* values  = row_coefficients.getValues();
    selections.clear();
    support.assign(indices, indices + len);
    bool usable = true;
    for (int j = 0; j < len && usable; ++j) {
      const int col = indices[j];
      const f_t v   = direction * values[j];
      if (col_flags[col].test(papilo::ColFlag::kIntegral)) {
        // A negative coefficient on an integral column is what this presolver itself writes, so
        // rejecting it here is also what keeps a rewritten row from matching a second time.
        usable = is_free_binary(col) && num.isEq(v, f_t{1});
        if (usable) selections.push_back(col);
      } else {
        // A relaxing term: non-positive over the whole box, so it cannot violate the row at z = 0.
        usable = num.isLT(v, f_t{0}) && !col_flags[col].test(papilo::ColFlag::kLbInf) &&
                 !num.isLT(lower_bounds[col], f_t{0});
      }
    }
    if (!usable || selections.size() < 2) continue;
    // At or above its own support size the cap is implied by the gates already, and so is K z.
    const f_t n_selections = selections.size();
    if (!num.isLT(capacity, n_selections)) continue;

    auto [lo, hi] = gates_of(selections[0]);
    common.clear();
    for (auto it = lo; it != hi; ++it)
      common.push_back(it->second);
    std::sort(common.begin(), common.end());
    for (size_t k = 1; k < selections.size() && !common.empty(); ++k) {
      auto [klo, khi] = gates_of(selections[k]);
      candidates.clear();
      for (auto it = klo; it != khi; ++it)
        candidates.push_back(it->second);
      std::sort(candidates.begin(), candidates.end());
      intersection.clear();
      std::set_intersection(common.begin(),
                            common.end(),
                            candidates.begin(),
                            candidates.end(),
                            std::back_inserter(intersection));
      common.swap(intersection);
    }
    if (common.empty()) continue;

    std::sort(support.begin(), support.end());
    int activation = -1;
    for (int z : common) {
      if (std::binary_search(support.begin(), support.end(), z)) continue;
      activation = z;
      break;
    }
    if (activation < 0) continue;

    cuopt_assert(is_free_binary(activation), "the activation of a gate row is a free binary");

    papilo::TransactionGuard<f_t> guard{reductions};
    reductions.lockRow(row);
    reductions.changeMatrixEntry(row, activation, direction * -capacity);
    if (direction == 1)
      reductions.changeRowRHS(row, f_t{0});
    else
      reductions.changeRowLHS(row, f_t{0});
    ++rows_strengthened;
    status = papilo::PresolveStatus::kReduced;
  }

  // Proposed, not applied: the activation is a nonzero the row does not have yet, and
  // ConstraintMatrix::change_coefficient refuses the insert when the row or the column has no slack
  // space left in its range, which rejects the whole transaction. The presolved nonzero count is
  // the figure to check this against.
  if (rows_strengthened > 0) {
    CUOPT_LOG_INFO("Activated capacity: proposed %d strengthened rows against %zu gates",
                   rows_strengthened,
                   gates.size());
  }

  return status;
}

#define INSTANTIATE(F_TYPE) template class ActivatedCapacity<F_TYPE>;

#if MIP_INSTANTIATE_FLOAT || PDLP_INSTANTIATE_FLOAT
INSTANTIATE(float)
#endif

#if MIP_INSTANTIATE_DOUBLE
INSTANTIATE(double)
#endif

#undef INSTANTIATE

}  // namespace cuopt::mathematical_optimization::mip
