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

#include <utilities/event_handler.cuh>

#include <rmm/cuda_stream_pool.hpp>
#include <rmm/device_uvector.hpp>

#include <raft/linalg/detail/cublas_wrappers.hpp>
#include <raft/core/handle.hpp>

#include <vector>

namespace cuopt::linear_programming::detail {

// This class is used to start a batched dot product
// With large problem size (>10K) and small batch size (<100), this is faster than using Segmented Reduce
template <typename i_t, typename f_t>
struct batched_dot_product_handler_t {
  batched_dot_product_handler_t(i_t batch_size, raft::handle_t const* handle_ptr)
    : batch_size_(batch_size), handle_ptr_(handle_ptr), stream_pool_(batch_size), dot_events_(batch_size) {}

  // Empty constructor for when used in non batch mode
  batched_dot_product_handler_t() {}

  void batch_dot_product(const rmm::device_uvector<f_t>& input_vector_1,
                         const rmm::device_uvector<f_t>& input_vector_2,
                         i_t problem_size,
                         rmm::device_uvector<f_t>& result)
  {
        // We need to make sure operations on the main stream are done before capturing the parallel dot products
        capture_event_.record(handle_ptr_->get_stream());
        for (i_t climber = 0; climber < batch_size_; ++climber) {
          capture_event_.stream_wait(stream_pool_.get_stream(climber));
        }
    for (i_t climber = 0; climber < batch_size_; ++climber) {
      RAFT_CUBLAS_TRY(raft::linalg::detail::cublasdot(handle_ptr_->get_cublas_handle(),
      problem_size,
      input_vector_1.data() + climber * problem_size,
      1,
      input_vector_2.data() + climber * problem_size,
      1,
      result.data() + climber,
      stream_pool_.get_stream(climber)));
        dot_events_[climber].record(stream_pool_.get_stream(climber));
    }
        for (i_t climber = 0; climber < batch_size_; ++climber) {
          dot_events_[climber].stream_wait(handle_ptr_->get_stream());
        }
  }

  i_t batch_size_{-1};
  raft::handle_t const* handle_ptr_{nullptr};
  rmm::cuda_stream_pool stream_pool_;
  event_handler_t capture_event_;
  std::vector<event_handler_t> dot_events_;
};

} // namespace cuopt::linear_programming::detail