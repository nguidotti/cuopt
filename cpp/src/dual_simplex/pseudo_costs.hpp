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

#pragma once

#include <dual_simplex/logger.hpp>
#include <dual_simplex/mip_node.hpp>
#include <dual_simplex/simplex_solver_settings.hpp>
#include <dual_simplex/types.hpp>
#include <utilities/omp_helpers.hpp>

#include <omp.h>
#include <vector>

namespace cuopt::linear_programming::dual_simplex {

// Variable selection method (See [1]).
// [1] T. Achterberg, “Constraint Integer Programming,” PhD, Technischen Universität Berlin,
// Berlin, 2007. doi: 10.14279/depositonce-1634.
enum class selection_method_t {
  PSEUDOCOST_BRANCHING = 0,  // Standard pseudocost branching + Martin's child selection criteria
  LINE_SEARCH_DIVING   = 1,  // Line search diving (9.2.4)
  PSEUDOCOST_DIVING    = 2,  // Pseudocost diving (9.2.5)
  GUIDED_DIVING = 3  // Guided diving (9.2.3). If no incumbent is found yet, use pseudocost diving.
};

template <typename i_t>
struct selected_variable_t {
  i_t variable;
  round_dir_t direction;
};

template <typename i_t, typename f_t>
class pseudo_costs_t {
 public:
  explicit pseudo_costs_t(i_t num_variables)
    : pseudo_cost_sum_down(num_variables),
      pseudo_cost_sum_up(num_variables),
      pseudo_cost_num_down(num_variables),
      pseudo_cost_num_up(num_variables)
  {
  }

  void update_pseudo_costs(mip_node_t<i_t, f_t>* node_ptr, f_t leaf_objective);

  void resize(i_t num_variables)
  {
    pseudo_cost_sum_down.resize(num_variables);
    pseudo_cost_sum_up.resize(num_variables);
    pseudo_cost_num_down.resize(num_variables);
    pseudo_cost_num_up.resize(num_variables);
  }

  void initialized(i_t& num_initialized_down,
                   i_t& num_initialized_up,
                   f_t& pseudo_cost_down_avg,
                   f_t& pseudo_cost_up_avg) const;

  // Martin's criteria for the preferred rounding direction (see [1])
  // [1] A. Martin, “Integer Programs with Block Structure,”
  // Technische Universit¨at Berlin, Berlin, 1999. Accessed: Aug. 08, 2025.
  // [Online]. Available: https://opus4.kobv.de/opus4-zib/frontdoor/index/index/docId/391
  round_dir_t martin_criteria(f_t val, f_t ref_val) const;

  // Selects the variable to branch on.
  selected_variable_t<i_t> variable_selection(const std::vector<i_t>& fractional,
                                              const std::vector<f_t>& solution,
                                              const std::vector<f_t>& root_solution,
                                              const std::vector<f_t>& incumbent,
                                              selection_method_t method,
                                              logger_t& log)
  {
    switch (method) {
      case selection_method_t::PSEUDOCOST_BRANCHING:
        return pseudocost_branching(fractional, solution, root_solution, log);

      case selection_method_t::LINE_SEARCH_DIVING:
        return line_search_diving(fractional, solution, root_solution, log);

      case selection_method_t::PSEUDOCOST_DIVING:
        return pseudocost_diving(fractional, solution, root_solution, log);

      case selection_method_t::GUIDED_DIVING:
        if (incumbent.size() != root_solution.size()) {
          return pseudocost_diving(fractional, solution, root_solution, log);
        } else {
          return guided_diving(fractional, solution, incumbent, log);
        }

      default:
        log.debug("Unknown variable selection method: %d\n", method);
        return {-1, round_dir_t::NONE};
    }
  }

  selected_variable_t<i_t> pseudocost_branching(const std::vector<i_t>& fractional,
                                                const std::vector<f_t>& solution,
                                                const std::vector<f_t>& root_solution,
                                                logger_t& log);

  selected_variable_t<i_t> line_search_diving(const std::vector<i_t>& fractional,
                                              const std::vector<f_t>& solution,
                                              const std::vector<f_t>& root_solution,
                                              logger_t& log);

  selected_variable_t<i_t> pseudocost_diving(const std::vector<i_t>& fractional,
                                             const std::vector<f_t>& solution,
                                             const std::vector<f_t>& root_solution,
                                             logger_t& log);

  selected_variable_t<i_t> guided_diving(const std::vector<i_t>& fractional,
                                         const std::vector<f_t>& solution,
                                         const std::vector<f_t>& incumbent,
                                         logger_t& log);

  void update_pseudo_costs_from_strong_branching(const std::vector<i_t>& fractional,
                                                 const std::vector<f_t>& root_soln);
  std::vector<f_t> pseudo_cost_sum_up;
  std::vector<f_t> pseudo_cost_sum_down;
  std::vector<i_t> pseudo_cost_num_up;
  std::vector<i_t> pseudo_cost_num_down;
  std::vector<f_t> strong_branch_down;
  std::vector<f_t> strong_branch_up;

  omp_mutex_t mutex;
  omp_atomic_t<i_t> num_strong_branches_completed = 0;
};

template <typename i_t, typename f_t>
void strong_branching(const lp_problem_t<i_t, f_t>& original_lp,
                      const simplex_solver_settings_t<i_t, f_t>& settings,
                      f_t start_time,
                      const std::vector<variable_type_t>& var_types,
                      const std::vector<f_t> root_soln,
                      const std::vector<i_t>& fractional,
                      f_t root_obj,
                      const std::vector<variable_status_t>& root_vstatus,
                      const std::vector<f_t>& edge_norms,
                      pseudo_costs_t<i_t, f_t>& pc);

inline const char* selection_method_to_string(selection_method_t method)
{
  switch (method) {
    case selection_method_t::PSEUDOCOST_BRANCHING: return "Pseudocost branching";
    case selection_method_t::LINE_SEARCH_DIVING: return "Line search diving";
    case selection_method_t::PSEUDOCOST_DIVING: return "Pseudocost diving";
    case selection_method_t::GUIDED_DIVING: return "Guided diving";
    default: return "Unknown method";
  }
}

}  // namespace cuopt::linear_programming::dual_simplex
