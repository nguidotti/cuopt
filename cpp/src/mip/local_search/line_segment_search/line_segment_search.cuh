/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <mip/feasibility_jump/feasibility_jump.cuh>
#include <utilities/timer.hpp>

namespace cuopt::linear_programming::detail {

template <typename i_t, typename f_t>
class line_segment_search_t {
 public:
  line_segment_search_t() = delete;
  line_segment_search_t(fj_t<i_t, f_t>& fj);
  bool search_line_segment(solution_t<i_t, f_t>& solution,
                           const rmm::device_uvector<f_t>& point_1,
                           const rmm::device_uvector<f_t>& point_2,
                           i_t n_points_to_search,
                           bool is_feasibility_run,
                           cuopt::timer_t& timer);

  fj_t<i_t, f_t>& fj;
};

}  // namespace cuopt::linear_programming::detail
