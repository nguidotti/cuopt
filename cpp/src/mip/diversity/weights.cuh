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

#include <thrust/fill.h>
#include <raft/core/handle.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

namespace cuopt::linear_programming::detail {

template <typename i_t, typename f_t>
struct weight_t {
  weight_t(i_t cstr_size, const raft::handle_t* handle_ptr)
    : cstr_weights(cstr_size, handle_ptr->get_stream()), objective_weight(handle_ptr->get_stream())
  {
    thrust::fill(handle_ptr->get_thrust_policy(), cstr_weights.begin(), cstr_weights.end(), 1.0);
    // objective_weight.set_value_to_zero_async(handle_ptr->get_stream());
    const f_t one = 1.;
    objective_weight.set_value_async(one, handle_ptr->get_stream());
  }

  rmm::device_uvector<f_t> cstr_weights;
  rmm::device_scalar<f_t> objective_weight;
};

}  // namespace cuopt::linear_programming::detail
