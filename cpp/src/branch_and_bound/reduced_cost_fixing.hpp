/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <dual_simplex/presolve.hpp>
#include <vector>

namespace cuopt::linear_programming::dual_simplex {

// Applies reduced cost fixing over the lower and bounds. Stores the bounds changes
// for applying bound strengthening later. Returns {num_fixed, num_improved}.
template <typename i_t, typename f_t>
std::pair<i_t, i_t> reduced_cost_fixing(const std::vector<f_t>& reduced_costs,
                                        const std::vector<variable_type_t>& var_types,
                                        const simplex_solver_settings_t<i_t, f_t>& settings,
                                        f_t obj,
                                        f_t upper_bound,
                                        std::vector<f_t>& lower_bounds,
                                        std::vector<f_t>& upper_bounds,
                                        std::vector<bool>& bounds_changed)
{
  const f_t threshold   = 100.0 * settings.integer_tol;
  const f_t weaken      = settings.integer_tol;
  const f_t fixed_tol   = settings.fixed_tol;
  const f_t abs_gap     = upper_bound - obj;
  i_t num_improved      = 0;
  i_t num_fixed         = 0;
  i_t num_cols_to_check = reduced_costs.size();  // Reduced costs will be smaller than the original
                                                 // problem because we have added slacks for cuts

  bounds_changed.assign(lower_bounds.size(), false);

  for (i_t j = 0; j < num_cols_to_check; j++) {
    if (std::isfinite(reduced_costs[j]) && std::abs(reduced_costs[j]) > threshold) {
      const f_t lower_j = lower_bounds[j];
      const f_t upper_j = upper_bounds[j];
      const bool is_integer =
        var_types[j] == variable_type_t::INTEGER || var_types[j] == variable_type_t::BINARY;

      if (lower_j > -inf && reduced_costs[j] > 0) {
        f_t new_upper_bound = lower_j + abs_gap / reduced_costs[j];
        if (is_integer) { new_upper_bound = std::floor(new_upper_bound + weaken); }

        if (new_upper_bound < upper_j) {
          ++num_improved;
          upper_bounds[j]   = new_upper_bound;
          bounds_changed[j] = true;
        }
      }

      if (upper_j < inf && reduced_costs[j] < 0) {
        f_t new_lower_bound = upper_j + abs_gap / reduced_costs[j];
        if (is_integer) { new_lower_bound = std::ceil(new_lower_bound - weaken); }

        if (new_lower_bound > lower_j) {
          ++num_improved;
          lower_bounds[j]   = new_lower_bound;
          bounds_changed[j] = true;
        }
      }

      if (is_integer && upper_bounds[j] <= lower_bounds[j] + fixed_tol) { ++num_fixed; }
    }
  }

  if (num_fixed > 0 || num_improved > 0) {
    settings.log.printf(
      "Reduced costs: Found %d improved bounds and %d fixed variables\n", num_improved, num_fixed);
  }
  return {num_fixed, num_improved};
}
}  // namespace cuopt::linear_programming::dual_simplex
