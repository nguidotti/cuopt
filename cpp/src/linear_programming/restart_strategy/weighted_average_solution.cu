/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <cuopt/error.hpp>

#include <linear_programming/pdlp_constants.hpp>
#include <linear_programming/restart_strategy/weighted_average_solution.hpp>
#include <linear_programming/utils.cuh>
#include <mip/mip_constants.hpp>

#include <raft/linalg/binary_op.cuh>
#include <raft/linalg/divide.cuh>

namespace cuopt::linear_programming::detail {
template <typename i_t, typename f_t>
weighted_average_solution_t<i_t, f_t>::weighted_average_solution_t(raft::handle_t const* handle_ptr,
                                                                   i_t primal_size,
                                                                   i_t dual_size,
                                                                   bool batch_mode)
  : handle_ptr_(handle_ptr),
    stream_view_(handle_ptr_->get_stream()),
    primal_size_h_(primal_size),
    dual_size_h_(dual_size),
    sum_primal_solutions_{(batch_mode ? static_cast<size_t>((0 + 3)/*@@*/) : 1) * primal_size_h_, stream_view_},
    sum_dual_solutions_{(batch_mode ? static_cast<size_t>((0 + 3)/*@@*/) : 1) * dual_size_h_, stream_view_},
    sum_primal_solution_weights_{(batch_mode ? static_cast<size_t>((0 + 3)/*@@*/) : 1), stream_view_},
    sum_dual_solution_weights_{(batch_mode ? static_cast<size_t>((0 + 3)/*@@*/) : 1), stream_view_},
    iterations_since_last_restart_{0},
    graph(stream_view_),
    batch_mode_(batch_mode)
{
  RAFT_CUDA_TRY(
    cudaMemsetAsync(sum_primal_solutions_.data(), 0.0, (batch_mode_ ? static_cast<size_t>((0 + 3)/*@@*/) : 1) * sizeof(f_t) * primal_size_h_, stream_view_));
  RAFT_CUDA_TRY(
    cudaMemsetAsync(sum_dual_solutions_.data(), 0.0, (batch_mode_ ? static_cast<size_t>((0 + 3)/*@@*/) : 1) * sizeof(f_t) * dual_size_h_, stream_view_));
  RAFT_CUDA_TRY(
    cudaMemsetAsync(sum_primal_solution_weights_.data(), 0.0, (batch_mode_ ? static_cast<size_t>((0 + 3)/*@@*/) : 1) * sizeof(f_t), stream_view_));
  RAFT_CUDA_TRY(
    cudaMemsetAsync(sum_dual_solution_weights_.data(), 0.0, (batch_mode_ ? static_cast<size_t>((0 + 3)/*@@*/) : 1) * sizeof(f_t), stream_view_));
}

template <typename i_t, typename f_t>
void weighted_average_solution_t<i_t, f_t>::reset_weighted_average_solution()
{
  RAFT_CUDA_TRY(
    cudaMemsetAsync(sum_primal_solutions_.data(), 0.0, (batch_mode_ ? static_cast<size_t>((0 + 3)/*@@*/) : 1) * sizeof(f_t) * primal_size_h_, stream_view_));
  RAFT_CUDA_TRY(
    cudaMemsetAsync(sum_dual_solutions_.data(), 0.0, (batch_mode_ ? static_cast<size_t>((0 + 3)/*@@*/) : 1) * sizeof(f_t) * dual_size_h_, stream_view_));
  RAFT_CUDA_TRY(
    cudaMemsetAsync(sum_primal_solution_weights_.data(), 0.0, (batch_mode_ ? static_cast<size_t>((0 + 3)/*@@*/) : 1) * sizeof(f_t), stream_view_));
  RAFT_CUDA_TRY(
    cudaMemsetAsync(sum_dual_solution_weights_.data(), 0.0, (batch_mode_ ? static_cast<size_t>((0 + 3)/*@@*/) : 1) * sizeof(f_t), stream_view_));
  iterations_since_last_restart_ = 0;
}

template <typename i_t, typename f_t>
__global__ void add_weight_sums(raft::device_span<const f_t> primal_weight,
                                raft::device_span<const f_t> dual_weight,
                                raft::device_span<f_t> sum_primal_solution_weights,
                                raft::device_span<f_t> sum_dual_solution_weights,
                                i_t batch_size)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= batch_size) return;

  sum_primal_solution_weights[idx] += primal_weight[idx];
  sum_dual_solution_weights[idx] += dual_weight[idx];
}

template <typename i_t, typename f_t>
void weighted_average_solution_t<i_t, f_t>::add_current_solution_to_weighted_average_solution(
  const f_t* primal_solution,
  const f_t* dual_solution,
  const rmm::device_uvector<f_t>& weight,
  i_t total_pdlp_iterations)
{
  // primalavg += primal_sol*weight     -- weight is just set to be step_size for the new solution
  // (same for primal and dual although julia repo makes it seem as though these should/could be
  // different)

  // TODO: handle batch mode

  if (!graph.is_initialized(total_pdlp_iterations)) {
    graph.start_capture(total_pdlp_iterations);

    cub::DeviceTransform::Transform(
      cuda::std::make_tuple(sum_primal_solutions_.data(), primal_solution,
      thrust::make_transform_iterator(
        thrust::make_counting_iterator(0),
        batch_wrapped_iterator<f_t>(weight.data(), primal_size_h_)
      )
    ),
      sum_primal_solutions_.data(),
      primal_size_h_ * (batch_mode_ ? static_cast<size_t>((0 + 3)/*@@*/) : 1),
      batch_a_add_scalar_times_b<f_t>(),
      stream_view_);

    cub::DeviceTransform::Transform(
      cuda::std::make_tuple(sum_dual_solutions_.data(), dual_solution,
      thrust::make_transform_iterator(
        thrust::make_counting_iterator(0),
        batch_wrapped_iterator<f_t>(weight.data(), dual_size_h_)
      )
    ),
      sum_dual_solutions_.data(),
      dual_size_h_ * (batch_mode_ ? static_cast<size_t>((0 + 3)/*@@*/) : 1),
      batch_a_add_scalar_times_b<f_t>(),
      stream_view_);

    // update weight sums and count (add weight and +1 respectively)
    const int block_size = (batch_mode_ ? std::min(256, (0 + 3)/*@@*/) : 1);
    const int grid_size = (batch_mode_ ? cuda::ceil_div((0 + 3)/*@@*/, block_size) : 1);
    add_weight_sums<<<grid_size, block_size, 0, stream_view_>>>(
      raft::device_span<const f_t>(weight.data(), (batch_mode_ ? static_cast<size_t>((0 + 3)/*@@*/) : 1)),
      raft::device_span<const f_t>(weight.data(), (batch_mode_ ? static_cast<size_t>((0 + 3)/*@@*/) : 1)),
      raft::device_span<f_t>(sum_primal_solution_weights_.data(), (batch_mode_ ? static_cast<size_t>((0 + 3)/*@@*/) : 1)),
      raft::device_span<f_t>(sum_dual_solution_weights_.data(), (batch_mode_ ? static_cast<size_t>((0 + 3)/*@@*/) : 1)),
      batch_mode_ ? static_cast<i_t>((0 + 3)/*@@*/) : 1);

    graph.end_capture(total_pdlp_iterations);
  }
  graph.launch(total_pdlp_iterations);

  iterations_since_last_restart_ += 1;
}

template <typename i_t, typename f_t>
void weighted_average_solution_t<i_t, f_t>::compute_averages(rmm::device_uvector<f_t>& avg_primal,
                                                             rmm::device_uvector<f_t>& avg_dual)
{
  // no iterations have added to the sum, so avg is all zero vector
  if (!iterations_since_last_restart_) {
    RAFT_CUDA_TRY(
      cudaMemsetAsync(avg_primal.data(), f_t(0.0), (batch_mode_ ? static_cast<size_t>((0 + 3)/*@@*/) : 1) * sizeof(f_t) * primal_size_h_, stream_view_));
    RAFT_CUDA_TRY(
      cudaMemsetAsync(avg_dual.data(), f_t(0.0), (batch_mode_ ? static_cast<size_t>((0 + 3)/*@@*/) : 1) * sizeof(f_t) * dual_size_h_, stream_view_));
    return;
  }

  // compute sum_primal_solutions/primal_size
  cub::DeviceTransform::Transform(
    cuda::std::make_tuple(sum_primal_solutions_.data(),
                          thrust::make_transform_iterator(
                            thrust::make_counting_iterator(0),
                            batch_wrapped_iterator<f_t>(sum_primal_solution_weights_.data(), primal_size_h_)
                          )
    ),
    avg_primal.data(),
    primal_size_h_ * (batch_mode_ ? static_cast<size_t>((0 + 3)/*@@*/) : 1),
    batch_safe_div<f_t>(),
    stream_view_);

  cub::DeviceTransform::Transform(
    cuda::std::make_tuple(sum_dual_solutions_.data(),
                          thrust::make_transform_iterator(
                            thrust::make_counting_iterator(0),
                            batch_wrapped_iterator<f_t>(sum_dual_solution_weights_.data(), dual_size_h_)
                          )
    ),
    avg_dual.data(),
    dual_size_h_ * (batch_mode_ ? static_cast<size_t>((0 + 3)/*@@*/) : 1),
    batch_safe_div<f_t>(),
    stream_view_);
}

template <typename i_t, typename f_t>
i_t weighted_average_solution_t<i_t, f_t>::get_iterations_since_last_restart() const
{
  return iterations_since_last_restart_;
}

#if MIP_INSTANTIATE_FLOAT
template __global__ void add_weight_sums<int, float>(raft::device_span<const float> primal_weight,
                                                raft::device_span<const float> dual_weight,
                                                raft::device_span<float> sum_primal_solution_weights,
                                                raft::device_span<float> sum_dual_solution_weights,
                                                int batch_size);

template class weighted_average_solution_t<int, float>;
#endif

#if MIP_INSTANTIATE_DOUBLE
template __global__ void add_weight_sums<int, double>(raft::device_span<const double> primal_weight,
                                                 raft::device_span<const double> dual_weight,
                                                 raft::device_span<double> sum_primal_solution_weights,
                                                 raft::device_span<double> sum_dual_solution_weights,
                                                 int batch_size);

template class weighted_average_solution_t<int, double>;
#endif

}  // namespace cuopt::linear_programming::detail
