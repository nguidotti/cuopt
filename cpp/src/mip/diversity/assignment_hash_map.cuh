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

namespace cuopt {
namespace linear_programming {
namespace detail {

template <typename i_t, typename f_t>
class assignment_hash_map_t {
 public:
  assignment_hash_map_t(const problem_t<i_t, f_t>& problem);
  void fill_integer_assignment(solution_t<i_t, f_t>& solution);
  size_t hash_solution(solution_t<i_t, f_t>& solution);
  void insert(solution_t<i_t, f_t>& solution);
  bool check_skip_solution(solution_t<i_t, f_t>& solution, i_t max_occurance);

  // keep the hash to encounter count of solution hash
  std::unordered_map<size_t, i_t> solution_hash_count;
  rmm::device_uvector<size_t> reduction_buffer;
  rmm::device_uvector<size_t> integer_assignment;
  rmm::device_scalar<size_t> hash_sum;
  rmm::device_buffer temp_storage;
};

}  // namespace detail
}  // namespace linear_programming
}  // namespace cuopt
