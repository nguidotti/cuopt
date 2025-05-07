/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <mip/local_search/rounding/constraint_prop.cuh>
#include <mip/relaxed_lp/relaxed_lp.cuh>
#include <mip/solution/solution.cuh>
#include <utilities/seed_generator.cuh>

namespace cuopt::linear_programming::detail {

template <typename i_t, typename f_t>
class bound_prop_recombiner_t : public recombiner_t<i_t, f_t> {
 public:
  bound_prop_recombiner_t(mip_solver_context_t<i_t, f_t>& context,
                          i_t n_vars,
                          constraint_prop_t<i_t, f_t>& constraint_prop_,
                          const raft::handle_t* handle_ptr)
    : recombiner_t<i_t, f_t>(context, n_vars, handle_ptr),
      constraint_prop(constraint_prop_),
      rng(cuopt::seed_generator::get_seed())
  {
  }

  std::pair<solution_t<i_t, f_t>, bool> recombine(solution_t<i_t, f_t>& a, solution_t<i_t, f_t>& b)
  {
    raft::common::nvtx::range fun_scope("bound_prop_recombiner");
    // copy the solution from A
    solution_t<i_t, f_t> offspring(a);
    // find same values and populate it to offspring
    this->assign_same_integer_values(a, b, offspring);
    i_t remaining_variables = this->n_remaining.value(a.handle_ptr->get_stream());
    // // from the remaining integers, populate randomly.
    CUOPT_LOG_DEBUG("n_vars from A/B %d remaining_variables %d",
                    a.problem_ptr->n_variables - remaining_variables,
                    remaining_variables);
    // if either all integers are from A(meaning all are common) or all integers are from B(meaning
    // all are different), return
    if (a.problem_ptr->n_integer_vars - remaining_variables == 0 || remaining_variables == 0) {
      return std::make_pair(offspring, false);
    }

    cuopt_assert(a.problem_ptr == b.problem_ptr,
                 "The two solutions should not refer to different problems");
    auto a_view         = a.view();
    auto b_view         = b.view();
    auto offspring_view = offspring.view();
    const f_t int_tol   = a.problem_ptr->tolerances.integrality_tolerance;
    rmm::device_uvector<thrust::pair<f_t, f_t>> probing_values(a.problem_ptr->n_variables,
                                                               a.handle_ptr->get_stream());
    // this is to give two possibilities to round in case of conflict
    thrust::for_each(a.handle_ptr->get_thrust_policy(),
                     thrust::make_counting_iterator(0),
                     thrust::make_counting_iterator(a.problem_ptr->n_variables),
                     [a_view, b_view, probing_values = probing_values.data()] __device__(i_t idx) {
                       f_t a_val = a_view.assignment[idx];
                       f_t b_val = b_view.assignment[idx];
                       cuopt_assert(a_view.problem.check_variable_within_bounds(idx, a_val), "");
                       cuopt_assert(b_view.problem.check_variable_within_bounds(idx, b_val), "");
                       probing_values[idx] = thrust::make_pair(a_val, b_val);
                     });
    // populate remaining N integers randomly/average from each solution
    thrust::for_each(a.handle_ptr->get_thrust_policy(),
                     this->remaining_indices.data(),
                     this->remaining_indices.data() + remaining_variables,
                     [a_view,
                      b_view,
                      offspring_view,
                      int_tol,
                      probing_values = probing_values.data(),
                      seed           = cuopt::seed_generator::get_seed()] __device__(i_t idx) {
                       raft::random::PCGenerator rng(seed, idx, 0);
                       f_t a_val = a_view.assignment[idx];
                       f_t b_val = b_view.assignment[idx];
                       cuopt_assert(a_view.problem.check_variable_within_bounds(idx, a_val), "");
                       cuopt_assert(b_view.problem.check_variable_within_bounds(idx, b_val), "");
                       f_t avg_val = (b_val + a_val) / 2;
                       if (a_view.problem.is_integer_var(idx)) {
                         const bool rnd = rng.next_u32() % 2;
                         // if the var is integer, populate probing vals
                         f_t first_val = rnd ? b_val : a_val;
                         cuopt_assert(is_integer<f_t>(first_val, int_tol),
                                      "The value must be integer");
                         // TODO check the rounding direction and var bounds
                         probing_values[idx] = thrust::make_pair(first_val, round(avg_val));
                         // assign some floating value, so that they can be rounded by bounds prop
                         f_t lb = a_view.problem.variable_lower_bounds[idx];
                         f_t ub = a_view.problem.variable_upper_bounds[idx];
                         if (integer_equal<f_t>(lb, ub, int_tol)) {
                           cuopt_assert(false, "The var values must be different in A and B!");
                         } else if (isfinite(lb)) {
                           offspring_view.assignment[idx] = lb + 0.1;
                         } else {
                           offspring_view.assignment[idx] = ub - 0.1;
                         }
                       } else {
                         // if the var is continuous, take the average
                         offspring_view.assignment[idx] = avg_val;
                       }
                     });
    const f_t lp_run_time_after_feasible = 2.;
    timer_t timer(2.);
    auto h_probing_values = host_copy(probing_values);
    constraint_prop.apply_round(offspring, lp_run_time_after_feasible, timer, h_probing_values);
    cuopt_func_call(offspring.test_number_all_integer());
    offspring.compute_feasibility();
    bool same_as_parents = this->check_if_offspring_is_same_as_parents(offspring, a, b);
    return std::make_pair(offspring, !same_as_parents);
  }

  constraint_prop_t<i_t, f_t>& constraint_prop;
  thrust::default_random_engine rng;
};

}  // namespace cuopt::linear_programming::detail
