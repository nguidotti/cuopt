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

// Martin's criteria for the preferred rounding direction (see [1])
// [1] A. Martin, “Integer Programs with Block Structure,”
// Technische Universit¨at Berlin, Berlin, 1999. Accessed: Aug. 08, 2025.
// [Online]. Available: https://opus4.kobv.de/opus4-zib/frontdoor/index/index/docId/391
template <typename f_t>
round_dir_t martin_criteria(f_t val, f_t root_val);

template <typename i_t, typename f_t>
i_t pseudocost_branching(pseudo_costs_t<i_t, f_t>& pc,
                         const std::vector<i_t>& fractional,
                         const std::vector<f_t>& solution,
                         logger_t& log);

template <typename i_t, typename f_t>
selected_variable_t<i_t> line_search_diving(const std::vector<i_t>& fractional,
                                            const std::vector<f_t>& solution,
                                            const std::vector<f_t>& root_solution,
                                            logger_t& log);

template <typename i_t, typename f_t>
selected_variable_t<i_t> pseudocost_diving(pseudo_costs_t<i_t, f_t>& pc,
                                           const std::vector<i_t>& fractional,
                                           const std::vector<f_t>& solution,
                                           const std::vector<f_t>& root_solution,
                                           logger_t& log);

template <typename i_t, typename f_t>
selected_variable_t<i_t> guided_diving(pseudo_costs_t<i_t, f_t>& pc,
                                       const std::vector<i_t>& fractional,
                                       const std::vector<f_t>& solution,
                                       const std::vector<f_t>& incumbent,
                                       logger_t& log);

template <typename i_t, typename f_t>
f_t best_pseudocost_estimate(pseudo_costs_t<i_t, f_t>& pc,
                             const std::vector<i_t>& fractional,
                             const std::vector<f_t>& solution,
                             f_t lower_bound,
                             logger_t& log);

}  // namespace cuopt::linear_programming::dual_simplex
