/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <cmath>
#include <cstdio>
#include <dual_simplex/rounding.hpp>
#include <vector>
namespace cuopt::linear_programming::dual_simplex {

template <typename i_t, typename f_t>
bool simple_rounding(const lp_problem_t<i_t, f_t>& lp_problem,
                     lp_solution_t<i_t, f_t>& lp_solution,
                     std::vector<i_t>& fractional)
{
  if (fractional.size() == 0) { return true; }

  bool rounding_success = true;
  std::vector<i_t> new_fractional;
  new_fractional.reserve(fractional.size());

  for (i_t var_idx : fractional) {
    i_t up_lock   = 0;
    i_t down_lock = 0;
    i_t start     = lp_problem.A.col_start[var_idx];
    i_t end       = lp_problem.A.col_start[var_idx + 1];

    for (i_t k = start; k < end; ++k) {
      f_t nz_val = lp_problem.A.x[k];
      i_t nz_row = lp_problem.A.i[k];

      if (std::isfinite(lp_problem.upper[nz_row]) && std::isfinite(lp_problem.lower[nz_row])) {
        down_lock += 1;
        up_lock += 1;
        continue;
      }

      f_t sign = std::isfinite(lp_problem.upper[nz_row]) ? 1 : -1;

      if (nz_val * sign > 0) {
        up_lock += 1;
      } else {
        down_lock += 1;
      }
    }

    f_t curr_val        = lp_solution.x[var_idx];
    bool can_round_up   = up_lock == 0;
    bool can_round_down = down_lock == 0;

    if (can_round_up && can_round_down) {
      if (lp_problem.objective[var_idx] > 0) {
        lp_solution.x[var_idx] = std::floor(curr_val);
      } else {
        lp_solution.x[var_idx] = std::ceil(curr_val);
      }
    } else if (can_round_up) {
      lp_solution.x[var_idx] = std::ceil(curr_val);
    } else if (can_round_down) {
      lp_solution.x[var_idx] = std::floor(curr_val);
    } else {
      rounding_success = false;
      new_fractional.push_back(var_idx);
    }
  }

  fractional = new_fractional;
  return rounding_success;
}

// rounds to the nearest integer within bounds
template <typename f_t>
f_t round_nearest(f_t val, f_t lb, f_t ub, f_t int_tol, pcg_t& rng)
{
  if (val > ub) {
    return floor(ub + int_tol);
  } else if (val < lb) {
    return ceil(lb - int_tol);
  } else {
    f_t w = rng.next_float();
    f_t t = 2 * w * (1 - w);
    if (w > 0.5) { t = 1 - t; }
    return floor(val + t);
  }
}

// template <typename i_t, typename f_t>
// bool nearest_integer_rounding(const lp_problem_t<i_t, f_t>& lp_problem,
//                               const f_t int_tol,
//                               const i_t seed,
//                               const i_t stream,
//                               lp_solution_t<i_t, f_t>& lp_solution,
//                               std::vector<i_t>& fractional)
// {
//   if (fractional.size() == 0) { return true; }

//   pcg_t rng(pcg_t::default_seed + seed, pcg_t::default_stream + stream);

//   for (auto var_idx : fractional) {
//     f_t curr_val           = lp_solution.x[var_idx];
//     f_t lb                 = lp_problem.lower[var_idx];
//     f_t ub                 = lp_problem.upper[var_idx];
//     lp_solution.x[var_idx] = round_nearest(curr_val, lb, ub, int_tol, rng);
//   }

// }

#ifdef DUAL_SIMPLEX_INSTANTIATE_DOUBLE

template bool simple_rounding(const lp_problem_t<int, double>& lp_problem,
                              lp_solution_t<int, double>& lp_solution,
                              std::vector<int>& fractional);

#endif

}  // namespace cuopt::linear_programming::dual_simplex
