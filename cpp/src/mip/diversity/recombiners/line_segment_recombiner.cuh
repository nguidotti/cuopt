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

#include <mip/local_search/line_segment_search/line_segment_search.cuh>
#include <mip/relaxed_lp/relaxed_lp.cuh>
#include <mip/solution/solution.cuh>
#include <utilities/seed_generator.cuh>

namespace cuopt::linear_programming::detail {

template <typename i_t, typename f_t>
class line_segment_recombiner_t : public recombiner_t<i_t, f_t> {
 public:
  line_segment_recombiner_t(mip_solver_context_t<i_t, f_t> context,
                            i_t n_vars,
                            line_segment_search_t<i_t, f_t>& line_segment_search_,
                            const raft::handle_t* handle_ptr)
    : recombiner_t<i_t, f_t>(context, n_vars, handle_ptr), line_segment_search(line_segment_search_)
  {
  }
  std::pair<solution_t<i_t, f_t>, bool> recombine(solution_t<i_t, f_t>& a,
                                                  solution_t<i_t, f_t>& b,
                                                  const weight_t<i_t, f_t>& weights)
  {
    raft::common::nvtx::range fun_scope("line_segment_recombiner");
    // copy the solution from A
    solution_t<i_t, f_t> offspring(a);
    // TODO test the time limit of two seconds
    timer_t line_segment_timer{2.};
    // TODO after we have the conic combination, detect the lambda change
    // (i.e. the integral variables flip on line segment)
    i_t n_points_to_search  = 20;
    bool is_feasibility_run = false;
    line_segment_search.fj.copy_weights(weights, offspring.handle_ptr);
    // TODO fix common part and run FJ on remaining
    line_segment_search.search_line_segment(offspring,
                                            a.assignment,
                                            b.assignment,
                                            n_points_to_search,
                                            is_feasibility_run,
                                            line_segment_timer);
    bool same_as_parents = this->check_if_offspring_is_same_as_parents(offspring, a, b);
    return std::make_pair(offspring, !same_as_parents);
  }

  line_segment_search_t<i_t, f_t>& line_segment_search;
};

}  // namespace cuopt::linear_programming::detail
