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

#include <cuopt/linear_programming/optimization_problem.hpp>
#include <mip/problem/problem.cuh>
#include <mip/solution/solution.cuh>

#include <papilo/core/Presolve.hpp>

namespace cuopt::linear_programming::detail {

template <typename i_t, typename f_t>
class third_party_presolve_t {
 public:
  third_party_presolve_t() = default;

  optimization_problem_t<i_t, f_t> apply(optimization_problem_t<i_t, f_t> const& op_problem,
                                         problem_category_t category,
                                         f_t absolute_tolerance,
                                         double time_limit);

  void undo(rmm::device_uvector<f_t>& primal_solution,
            rmm::device_uvector<f_t>& dual_solution,
            rmm::device_uvector<f_t>& reduced_costs,
            problem_category_t category,
            rmm::cuda_stream_view stream_view);

 private:
  papilo::PostsolveStorage<f_t> post_solve_storage_;
};

}  // namespace cuopt::linear_programming::detail
