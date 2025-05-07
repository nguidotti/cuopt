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

#include <mip/solution/solution.cuh>
#include <mip/solver.cuh>
#include <mip/utils.cuh>
#include <utilities/copy_helpers.hpp>
#include <utilities/device_utils.cuh>

#include <thrust/random.h>
#include <thrust/shuffle.h>

namespace cuopt::linear_programming::detail {

// checks whether the values of a variable are equal when we consider them in a diversity
// measurement context
template <typename f_t>
HDI bool diverse_equal(f_t val1, f_t val2, f_t lb, f_t ub, bool is_integer, f_t int_tol)
{
  f_t range;
  if (is_integer) {
    range = int_tol;
  } else {
    range = (ub - lb) / 10;
    range = min(range, 0.25);
  }
  return integer_equal<f_t>(val1, val2, range);
}

template <typename i_t, typename f_t>
__global__ void assign_same_variables_kernel(typename solution_t<i_t, f_t>::view_t a,
                                             typename solution_t<i_t, f_t>::view_t b,
                                             typename solution_t<i_t, f_t>::view_t offspring,
                                             raft::device_span<i_t> remaining_indices,
                                             i_t* n_remaining)
{
  if (TH_ID_X >= a.assignment.size()) return;
  const i_t var_idx   = TH_ID_X;
  bool is_integer_var = a.problem.is_integer_var(var_idx);

  if (diverse_equal<f_t>(a.assignment[var_idx],
                         b.assignment[var_idx],
                         a.problem.variable_lower_bounds[var_idx],
                         a.problem.variable_upper_bounds[var_idx],
                         is_integer_var,
                         a.problem.tolerances.integrality_tolerance)) {
    offspring.assignment[var_idx] = a.assignment[var_idx];
  } else {
    i_t idx                = atomicAdd(n_remaining, 1);
    remaining_indices[idx] = var_idx;
  }
}

template <typename i_t, typename f_t>
class recombiner_t {
 public:
  recombiner_t(mip_solver_context_t<i_t, f_t>& context_,
               i_t n_vars,
               const raft::handle_t* handle_ptr)
    : context(context_),
      remaining_indices(n_vars, handle_ptr->get_stream()),
      n_remaining(handle_ptr->get_stream())
  {
  }

  void reset(i_t n_vars, const raft::handle_t* handle_ptr)
  {
    n_remaining.set_value_to_zero_async(handle_ptr->get_stream());
    remaining_indices.resize(n_vars, handle_ptr->get_stream());
  }

  void assign_same_integer_values(solution_t<i_t, f_t>& a,
                                  solution_t<i_t, f_t>& b,
                                  solution_t<i_t, f_t>& offspring)
  {
    reset(a.problem_ptr->n_variables, a.handle_ptr);
    const i_t TPB = 128;
    i_t n_blocks  = (a.problem_ptr->n_variables + TPB - 1) / TPB;
    assign_same_variables_kernel<i_t, f_t>
      <<<n_blocks, TPB, 0, a.handle_ptr->get_stream()>>>(a.view(),
                                                         b.view(),
                                                         offspring.view(),
                                                         cuopt::make_span(remaining_indices),
                                                         n_remaining.data());
  }

  bool check_if_offspring_is_same_as_parents(solution_t<i_t, f_t>& offspring,
                                             solution_t<i_t, f_t>& a,
                                             solution_t<i_t, f_t>& b)
  {
    bool equals_a = check_integer_equal_on_indices(offspring.problem_ptr->integer_indices,
                                                   offspring.assignment,
                                                   a.assignment,
                                                   a.problem_ptr->tolerances.integrality_tolerance,
                                                   a.handle_ptr);
    if (equals_a) { return true; }
    bool equals_b = check_integer_equal_on_indices(offspring.problem_ptr->integer_indices,
                                                   offspring.assignment,
                                                   b.assignment,
                                                   b.problem_ptr->tolerances.integrality_tolerance,
                                                   b.handle_ptr);
    return equals_b;
  }

  mip_solver_context_t<i_t, f_t>& context;
  rmm::device_uvector<i_t> remaining_indices;
  rmm::device_scalar<i_t> n_remaining;
};

}  // namespace cuopt::linear_programming::detail
