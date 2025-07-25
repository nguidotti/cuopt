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

#pragma once

#include "recombiner.cuh"

#include <dual_simplex/branch_and_bound.hpp>
#include <dual_simplex/simplex_solver_settings.hpp>
#include <dual_simplex/solve.hpp>

namespace cuopt::linear_programming::detail {

template <typename i_t, typename f_t>
class sub_mip_recombiner_t : public recombiner_t<i_t, f_t> {
 public:
  sub_mip_recombiner_t(mip_solver_context_t<i_t, f_t>& context,
                       i_t n_vars,
                       const raft::handle_t* handle_ptr)
    : recombiner_t<i_t, f_t>(context, n_vars, handle_ptr),
      vars_to_fix(n_vars, handle_ptr->get_stream()),
      context(context)
  {
  }

  std::pair<solution_t<i_t, f_t>, bool> recombine(solution_t<i_t, f_t>& a,
                                                  solution_t<i_t, f_t>& b,
                                                  const weight_t<i_t, f_t>& weights)
  {
    raft::common::nvtx::range fun_scope("Sub-MIP recombiner");
    auto& guiding_solution = a.get_feasible() ? a : b;
    auto& other_solution   = a.get_feasible() ? b : a;
    // copy the solution from A
    solution_t<i_t, f_t> offspring(guiding_solution);
    // find same values and populate it to offspring
    i_t n_different_vars =
      this->assign_same_integer_values(guiding_solution, other_solution, offspring);
    CUOPT_LOG_DEBUG("SUB_MIP rec: Number of different variables %d MAX_VARS %d",
                    n_different_vars,
                    sub_mip_recombiner_config_t::max_n_of_vars_from_other);
    i_t n_vars_from_other = n_different_vars;
    if (n_vars_from_other > (i_t)sub_mip_recombiner_config_t::max_n_of_vars_from_other) {
      n_vars_from_other = sub_mip_recombiner_config_t::max_n_of_vars_from_other;
      thrust::default_random_engine g{(unsigned int)cuopt::seed_generator::get_seed()};
      thrust::shuffle(a.handle_ptr->get_thrust_policy(),
                      this->remaining_indices.data(),
                      this->remaining_indices.data() + n_different_vars,
                      g);
    }
    i_t n_vars_from_guiding = a.problem_ptr->n_integer_vars - n_vars_from_other;
    if (n_vars_from_other == 0 || n_vars_from_guiding == 0) {
      CUOPT_LOG_DEBUG("Returning false because all vars are common or different");
      return std::make_pair(offspring, false);
    }
    CUOPT_LOG_DEBUG(
      "n_vars_from_guiding %d n_vars_from_other %d", n_vars_from_guiding, n_vars_from_other);
    this->compute_vars_to_fix(offspring, vars_to_fix, n_vars_from_other, n_vars_from_guiding);
    auto [fixed_problem, fixed_assignment, variable_map] = offspring.fix_variables(vars_to_fix);
    fixed_problem.check_problem_representation(true);
    if (!guiding_solution.get_feasible() && !other_solution.get_feasible()) {
      relaxed_lp_settings_t lp_settings;
      lp_settings.time_limit = sub_mip_recombiner_config_t::infeasibility_detection_time_limit;
      lp_settings.tolerance  = fixed_problem.tolerances.absolute_tolerance;
      lp_settings.return_first_feasible = true;
      lp_settings.save_state            = true;
      lp_settings.check_infeasibility   = true;
      // run lp with infeasibility detection on
      auto lp_response =
        get_relaxed_lp_solution(fixed_problem, fixed_assignment, offspring.lp_state, lp_settings);
      if (lp_response.get_termination_status() == pdlp_termination_status_t::PrimalInfeasible ||
          lp_response.get_termination_status() == pdlp_termination_status_t::DualInfeasible ||
          lp_response.get_termination_status() == pdlp_termination_status_t::TimeLimit) {
        CUOPT_LOG_DEBUG("SUB_MIP recombiner failed because LP found infeasible!");
        return std::make_pair(offspring, false);
      }
    }
    // brute force rounding threshold is 8
    const bool run_sub_mip = fixed_problem.n_integer_vars > 8;
    if (run_sub_mip) {
      // run sub-mip
      namespace dual_simplex = cuopt::linear_programming::dual_simplex;
      dual_simplex::mip_status_t branch_and_bound_status;
      dual_simplex::user_problem_t<i_t, f_t> branch_and_bound_problem;
      dual_simplex::simplex_solver_settings_t<i_t, f_t> branch_and_bound_settings;
      dual_simplex::mip_solution_t<i_t, f_t> branch_and_bound_solution(1);
      fixed_problem.get_host_user_problem(branch_and_bound_problem);
      branch_and_bound_solution.resize(branch_and_bound_problem.num_cols);
      // Fill in the settings for branch and bound
      branch_and_bound_settings.time_limit = sub_mip_recombiner_config_t::sub_mip_time_limit;
      branch_and_bound_settings.print_presolve_stats = false;
      branch_and_bound_settings.absolute_mip_gap_tol = context.settings.tolerances.absolute_mip_gap;
      branch_and_bound_settings.relative_mip_gap_tol = context.settings.tolerances.relative_mip_gap;
      branch_and_bound_settings.integer_tol = context.settings.tolerances.integrality_tolerance;
      dual_simplex::branch_and_bound_t<i_t, f_t> branch_and_bound(branch_and_bound_problem,
                                                                  branch_and_bound_settings);
      branch_and_bound_status = branch_and_bound.solve(branch_and_bound_solution);
      // TODO do partial solutions too
      if (branch_and_bound_status == dual_simplex::mip_status_t::OPTIMAL) {
        cuopt_assert(fixed_assignment.size() == branch_and_bound_solution.x.size(),
                     "Assignment size mismatch");
        raft::copy(fixed_assignment.data(),
                   branch_and_bound_solution.x.data(),
                   fixed_assignment.size(),
                   offspring.handle_ptr->get_stream());
      }
      offspring.handle_ptr->sync_stream();
    }
    // unfix the assignment on given result no matter if it is feasible
    offspring.unfix_variables(fixed_assignment, variable_map);
    if (!run_sub_mip) { offspring.round_nearest(); }
    cuopt_assert(offspring.test_number_all_integer(), "All must be integers after offspring");
    offspring.compute_feasibility();
    bool same_as_parents = this->check_if_offspring_is_same_as_parents(offspring, a, b);
    // adjust the max_n_of_vars_from_other
    if (n_different_vars > (i_t)sub_mip_recombiner_config_t::max_n_of_vars_from_other) {
      if (same_as_parents) {
        sub_mip_recombiner_config_t::increase_max_n_of_vars_from_other();
      } else {
        sub_mip_recombiner_config_t::decrease_max_n_of_vars_from_other();
      }
    }
    bool better_cost_than_parents =
      offspring.get_quality(weights) <
      std::min(other_solution.get_quality(weights), guiding_solution.get_quality(weights));
    bool better_feasibility_than_parents = offspring.get_feasible() &&
                                           !other_solution.get_feasible() &&
                                           !guiding_solution.get_feasible();
    if (better_cost_than_parents || better_feasibility_than_parents) {
      CUOPT_LOG_DEBUG("Offspring is feasible or better than both parents");
      return std::make_pair(offspring, true);
    }
    return std::make_pair(offspring, !same_as_parents);
  }
  rmm::device_uvector<i_t> vars_to_fix;
  mip_solver_context_t<i_t, f_t>& context;
};

}  // namespace cuopt::linear_programming::detail
