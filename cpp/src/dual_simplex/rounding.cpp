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
#include <dual_simplex/rounding.hpp>

namespace cuopt::linear_programming::dual_simplex {

template <typename i_t, typename f_t>
bool simple_rounding(lp_solution_t<i_t, f_t>& lp_solution,
                     const lp_problem_t<i_t, f_t>& lp_problem,
                     const std::vector<i_t>& fractional)
{
  bool rounding_success = true;
  for (i_t var_idx : fractional) {
    i_t up_lock   = 0;
    i_t down_lock = 0;
    i_t start     = lp_problem.A.col_start[var_idx];
    i_t end       = lp_problem.A.col_start[var_idx + 1];

    for (i_t k = start; k < end; ++k) {
      f_t Aij = lp_problem.A.x[k];

      if (std::isfinite(lp_problem.upper[k]) && std::isfinite(lp_problem.lower[k])) {
        down_lock += 1;
        up_lock += 1;
        continue;
      }

      f_t sign = std::isfinite(lp_problem.upper[k]) ? 1 : -1;

      if (Aij * sign > 0) {
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
    }
  }

  return rounding_success;
}

}  // namespace cuopt::linear_programming::dual_simplex
