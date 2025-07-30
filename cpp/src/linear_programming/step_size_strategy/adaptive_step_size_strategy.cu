/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <cuopt/linear_programming/pdlp/pdlp_hyper_params.cuh>
#include <linear_programming/pdlp_constants.hpp>
#include <linear_programming/step_size_strategy/adaptive_step_size_strategy.hpp>
#include <linear_programming/utils.cuh>
#include <mip/mip_constants.hpp>
#include <utilities/copy_helpers.hpp>

#include <raft/sparse/detail/cusparse_macros.h>
#include <raft/sparse/detail/cusparse_wrappers.h>
#include <raft/common/nvtx.hpp>
#include <raft/core/operators.hpp>
#include <raft/linalg/binary_op.cuh>
#include <raft/linalg/detail/cublas_wrappers.hpp>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <limits>

namespace cuopt::linear_programming::detail {

constexpr int parallel_stream_computation = 2;

template <typename i_t, typename f_t>
adaptive_step_size_strategy_t<i_t, f_t>::adaptive_step_size_strategy_t(
  raft::handle_t const* handle_ptr,
  rmm::device_uvector<f_t>* primal_weight,
  rmm::device_uvector<f_t>* step_size,
  bool batch_mode)
  : stream_pool_(parallel_stream_computation),
    dot_delta_X_(cudaEventDisableTiming),
    dot_delta_Y_(cudaEventDisableTiming),
    deltas_are_done_(cudaEventDisableTiming),
    handle_ptr_(handle_ptr),
    stream_view_(handle_ptr_->get_stream()),
    primal_weight_(primal_weight),
    step_size_(step_size),
    // This should just use a "number of problems" parameter (and be one for non batch)
    valid_step_size_((batch_mode ? static_cast<size_t>((0 + 3)/*@@*/) : 1)),
    interaction_{(batch_mode ? static_cast<size_t>((0 + 3)/*@@*/) : 1), stream_view_},
    norm_squared_delta_primal_{(batch_mode ? static_cast<size_t>((0 + 3)/*@@*/) : 1), stream_view_},
    norm_squared_delta_dual_{(batch_mode ? static_cast<size_t>((0 + 3)/*@@*/) : 1), stream_view_},
    reusable_device_scalar_value_1_{f_t(1.0), stream_view_},
    reusable_device_scalar_value_0_{f_t(0.0), stream_view_},
    graph_(stream_view_),
    batch_mode_(batch_mode),
    batched_dot_product_handler_(batch_mode ? batched_transform_reduce_handler_t<i_t, f_t>((0 + 3)/*@@*/, handle_ptr_) : batched_transform_reduce_handler_t<i_t, f_t>())
{
}

void set_adaptive_step_size_hyper_parameters(rmm::cuda_stream_view stream_view)
{
  RAFT_CUDA_TRY(cudaMemcpyToSymbolAsync(pdlp_hyper_params::default_reduction_exponent,
                                        &pdlp_hyper_params::host_default_reduction_exponent,
                                        sizeof(double),
                                        0,
                                        cudaMemcpyHostToDevice,
                                        stream_view));
  RAFT_CUDA_TRY(cudaMemcpyToSymbolAsync(pdlp_hyper_params::default_growth_exponent,
                                        &pdlp_hyper_params::host_default_growth_exponent,
                                        sizeof(double),
                                        0,
                                        cudaMemcpyHostToDevice,
                                        stream_view));
  RAFT_CUDA_TRY(cudaMemcpyToSymbolAsync(pdlp_hyper_params::primal_distance_smoothing,
                                        &pdlp_hyper_params::host_primal_distance_smoothing,
                                        sizeof(double),
                                        0,
                                        cudaMemcpyHostToDevice,
                                        stream_view));
  RAFT_CUDA_TRY(cudaMemcpyToSymbolAsync(pdlp_hyper_params::dual_distance_smoothing,
                                        &pdlp_hyper_params::host_dual_distance_smoothing,
                                        sizeof(double),
                                        0,
                                        cudaMemcpyHostToDevice,
                                        stream_view));
}

template <typename i_t, typename f_t>
__global__ void compute_step_sizes_from_movement_and_interaction(
  typename adaptive_step_size_strategy_t<i_t, f_t>::view_t step_size_strategy_view,
  raft::device_span<f_t> primal_step_size,
  raft::device_span<f_t> dual_step_size,
  i_t* pdhg_iteration,
  int batch_size)
{
  const int id = threadIdx.x + blockIdx.x * blockDim.x;
  if (id >= batch_size) { return; }

  f_t primal_weight_ = step_size_strategy_view.primal_weight[id];

  f_t movement = pdlp_hyper_params::primal_distance_smoothing * primal_weight_ *
                   step_size_strategy_view.norm_squared_delta_primal[id] +
                 (pdlp_hyper_params::dual_distance_smoothing / primal_weight_) *
                   step_size_strategy_view.norm_squared_delta_dual[id];

#ifdef PDLP_DEBUG_MODE
  printf("-compute_step_sizes_from_movement_and_interaction:\n");
#endif
  if (movement <= 0 || movement >= divergent_movement<f_t>) {
    step_size_strategy_view.valid_step_size[id] = -1;
#ifdef PDLP_DEBUG_MODE
    printf("  Movement is %lf. Done or numerical error has happened\n", movement);
#endif
    return;
  }

  f_t interaction_ = raft::abs(step_size_strategy_view.interaction[id]);
  f_t step_size_   = step_size_strategy_view.step_size[id];

  // Increase PDHG iteration
  *pdhg_iteration += 1;

  f_t iteration_coefficient_ = *pdhg_iteration;

  // proof of thm 1 requires movement / step_size >= interaction.
  f_t step_size_limit = interaction_ > 0.0 ? movement / interaction_ : raft::myInf<f_t>();

#ifdef PDLP_DEBUG_MODE
  printf("    interaction_=%lf movement=%lf\n", interaction_, movement);
  printf("    step_size_=%lf step_size_limit=%lf pdhg_iteration=%d iteration_coefficient_=%lf\n",
         step_size_,
         step_size_limit,
         *pdhg_iteration,
         iteration_coefficient_);
#endif

  // TODO: every batch should have a different step size
  if (step_size_ <= step_size_limit && id == 0) {
    step_size_strategy_view.valid_step_size[id] = 1;

#ifdef PDLP_DEBUG_MODE
    printf("    Step size is smaller\n");
#endif
  }

  // The step size was too large and therefore we now compute the next stepsize to test out.
  // We have two candidates of which we take the smaller to retry taking a step
  const f_t potential_new_step_size_1 =
    (f_t(1.0) - raft::pow<f_t>(iteration_coefficient_ + f_t(1.0),
                               -pdlp_hyper_params::default_reduction_exponent)) *
    step_size_limit;
  const f_t potential_new_step_size_2 =
    (f_t(1.0) + raft::pow<f_t>(iteration_coefficient_ + f_t(1.0),
                               -pdlp_hyper_params::default_growth_exponent)) *
    step_size_;

#ifdef PDLP_DEBUG_MODE
  printf(
    "Compute adaptative step size: iteration_coefficient_=%lf "
    "-pdlp_hyper_params::default_reduction_exponent=%lf step_size_limit=%lf\n",
    iteration_coefficient_,
    -pdlp_hyper_params::default_reduction_exponent,
    step_size_limit);
  printf(
    "Compute adaptative step size: iteration_coefficient_=%lf "
    "-pdlp_hyper_params::default_growth_exponent=%lf step_size_=%lf\n",
    iteration_coefficient_,
    -pdlp_hyper_params::default_growth_exponent,
    step_size_);
  printf(
    "Compute adaptative step size: potential_new_step_size_1=%lf potential_new_step_size_2=%lf\n",
    potential_new_step_size_1,
    potential_new_step_size_2);
#endif

  step_size_ = raft::min<f_t>(potential_new_step_size_1, potential_new_step_size_2);

#ifdef PDLP_DEBUG_MODE
  printf("Compute adaptative step size: min_step_size_picked=%lf\n", step_size_);
#endif


  primal_step_size[id] = step_size_ / primal_weight_;
  dual_step_size[id]   = step_size_ * primal_weight_;

  step_size_strategy_view.step_size[id] = step_size_;
  cuopt_assert(!isnan(step_size_), "step size can't be nan");
  cuopt_assert(!isinf(step_size_), "step size can't be inf");
}

template <typename i_t, typename f_t>
void adaptive_step_size_strategy_t<i_t, f_t>::compute_step_sizes(
  pdhg_solver_t<i_t, f_t>& pdhg_solver,
  rmm::device_uvector<f_t>& primal_step_size,
  rmm::device_uvector<f_t>& dual_step_size,
  i_t total_pdlp_iterations)
{
  raft::common::nvtx::range fun_scope("compute_step_sizes");

  if (!graph_.is_initialized(total_pdlp_iterations)) {
    graph_.start_capture(total_pdlp_iterations);

    // compute numerator and deminator of n_lim
    compute_interaction_and_movement(pdhg_solver.get_primal_tmp_resource(),
                                     pdhg_solver.get_cusparse_view(),
                                     pdhg_solver.get_saddle_point_state());
    // Compute n_lim, n_next and decide if step size is valid
    const int block_size = std::min(256, (batch_mode_ ? (0 + 3)/*@@*/ : 1));
    const int num_blocks = (batch_mode_ ? cuda::ceil_div((0 + 3)/*@@*/, block_size) : 1);
    compute_step_sizes_from_movement_and_interaction<i_t, f_t>
      <<<num_blocks, block_size, 0, stream_view_>>>(this->view(),
                                  make_span(primal_step_size),
                                  make_span(dual_step_size),
                                  pdhg_solver.get_d_total_pdhg_iterations().data(),
                                  (batch_mode_ ? (0 + 3)/*@@*/ : 1));
    graph_.end_capture(total_pdlp_iterations);
  }
  graph_.launch(total_pdlp_iterations);
  // Steam sync so that next call can see modification made to host var valid_step_size
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream_view_));
}

template <typename i_t, typename f_t>
void adaptive_step_size_strategy_t<i_t, f_t>::compute_interaction_and_movement(
  rmm::device_uvector<f_t>& tmp_primal, // Conditionnaly is batch or non batch
  cusparse_view_t<i_t, f_t>& cusparse_view,
  saddle_point_state_t<i_t, f_t>& current_saddle_point_state)
{
  cuopt_assert(current_saddle_point_state.get_next_AtY().size() == current_saddle_point_state.get_current_AtY().size(), "next_AtY and current_AtY must have the same size");
  cuopt_assert(current_saddle_point_state.get_next_AtY().size() == tmp_primal.size(), "next_AtY and tmp_primal must have the same size");
  cuopt_assert(current_saddle_point_state.get_next_AtY().size() == current_saddle_point_state.get_primal_solution().size(), "primal_size and next_AtY must have the same size");

  // QP would need this:
  // if iszero(problem.objective_matrix)
  //   primal_objective_interaction = 0.0
  // else
  //   primal_objective_interaction =
  //     0.5 * (delta_primal' * problem.objective_matrix * delta_primal)
  // end
  // would need to add abs(primal_objective_interaction) to interaction as well

  /*
    Here we compute : movement / interaction

    Movement: ||(x' - x), (y' - y)||²
    Interaction: (y' - y)_t . A @ (x' - x)

    Deltas x & y were computed during pdhg step

    We will compute in parallel (parallel cuda graph):
    ||(x' - x)||
    ||(y' - y)||
    (y' - y)_t . A @ (x' - x)

    And finally merge the results
  */

  // We need to make sure both dot products happens after previous operations (next_primal/dual)
  // Thus, we add another node in the main stream before starting the SpMVs

  deltas_are_done_.record(stream_view_);

  // primal_dual_interaction computation => we purposly diverge from the paper (delta_y . (A @ x' -
  // A@x)) to save one SpMV
  // Instead we do: delta_x . (A_t @ y' - A_t @ y)
  // A_t @ y has already been computed during compute next_primal
  // A_t @ y' is computed here each time but, if a valid step is found, A @ y'
  // becomes A @ y for next step (as what was y' becomes y if valid for next step). This saves the
  // first A @ y SpMV in the compute_next_primal of next PDHG step

  // Compute A_t @ (y' - y) = A_t @ y' - 1 * current_AtY

  // First compute Ay' to be reused as Ay in next PDHG iteration (if found step size if valid)
  if (!batch_mode_) {
    RAFT_CUSPARSE_TRY(
      raft::sparse::detail::cusparsespmv(handle_ptr_->get_cusparse_handle(),
                                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                                        reusable_device_scalar_value_1_.data(),  // alpha
                                        cusparse_view.A_T,
                                        cusparse_view.potential_next_dual_solution,
                                        reusable_device_scalar_value_0_.data(),  // beta
                                        cusparse_view.next_AtY,
                                        CUSPARSE_SPMV_CSR_ALG2,
                                        (f_t*)cusparse_view.buffer_transpose.data(),
                                        stream_view_));

    // Compute Ay' - Ay = next_Aty - current_Aty
    cub::DeviceTransform::Transform(
      cuda::std::make_tuple(current_saddle_point_state.get_next_AtY().data(),
                            current_saddle_point_state.get_current_AtY().data()),
      tmp_primal.data(),
      current_saddle_point_state.get_primal_size(),
      sub_op<f_t>(),
      stream_view_);
  } else {
    RAFT_CUSPARSE_TRY(raft::sparse::detail::cusparsespmm(handle_ptr_->get_cusparse_handle(),
                                                       CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                       CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                       reusable_device_scalar_value_1_.data(),
                                                       cusparse_view.A_T,
                                                       cusparse_view.batch_potential_next_dual_solution,
                                                       reusable_device_scalar_value_0_.data(),
                                                       cusparse_view.batch_next_AtYs,
                                                       CUSPARSE_SPMM_CSR_ALG3,
                                                       (f_t*)cusparse_view.buffer_transpose_batch.data(),
                                                       stream_view_));
    // Compute Ay' - Ay = next_Aty - current_Aty
    cub::DeviceTransform::Transform(
      cuda::std::make_tuple(current_saddle_point_state.get_next_AtY().data(),
                            current_saddle_point_state.get_current_AtY().data()),
      tmp_primal.data(),
      tmp_primal.size(),
      sub_op<f_t>(),
      stream_view_);
  }
#ifdef PDLP_DEBUG_MODE
  RAFT_CUDA_TRY(cudaDeviceSynchronize());
#endif

  // compute interaction (x'-x) . (A(y'-y))
  if (!batch_mode_) {
  RAFT_CUBLAS_TRY(
    raft::linalg::detail::cublasdot(handle_ptr_->get_cublas_handle(),
                                    current_saddle_point_state.get_primal_size(),
                                    tmp_primal.data(),
                                    primal_stride,
                                    current_saddle_point_state.get_delta_primal().data(),
                                    primal_stride,
                                    interaction_.data(),
                                    stream_view_));
  } else {
    batched_dot_product_handler_.batch_transform_reduce([&](i_t climber, rmm::cuda_stream_view stream){
      RAFT_CUBLAS_TRY(raft::linalg::detail::cublasdot(handle_ptr_->get_cublas_handle(),
        current_saddle_point_state.get_primal_size(),
        tmp_primal.data() + climber * current_saddle_point_state.get_primal_size(),
        primal_stride,
        current_saddle_point_state.get_delta_primal().data() + climber * current_saddle_point_state.get_primal_size(),
        primal_stride,
        interaction_.data() + climber,
        stream));
    });
  }

  // Compute movement
  //  compute euclidean norm squared which is
  //  same as taking the dot product with itself
  //    movement = 0.5 * solver_state.primal_weight
  //    * norm(delta_primal) ^
  //               2 + (0.5 /
  //               solver_state.primal_weight) *
  //               norm(delta_dual) ^ 2;
  if (!batch_mode_) {
    deltas_are_done_.stream_wait(stream_pool_.get_stream(0));
    RAFT_CUBLAS_TRY(
      raft::linalg::detail::cublasdot(handle_ptr_->get_cublas_handle(),
                                      current_saddle_point_state.get_primal_size(),
                                      current_saddle_point_state.get_delta_primal().data(),
                                      primal_stride,
                                      current_saddle_point_state.get_delta_primal().data(),
                                      primal_stride,
                                      norm_squared_delta_primal_.data(),
                                      stream_pool_.get_stream(0)));
    dot_delta_X_.record(stream_pool_.get_stream(0));

    deltas_are_done_.stream_wait(stream_pool_.get_stream(1));
    RAFT_CUBLAS_TRY(
      raft::linalg::detail::cublasdot(handle_ptr_->get_cublas_handle(),
                                      current_saddle_point_state.get_dual_size(),
                                      current_saddle_point_state.get_delta_dual().data(),
                                      dual_stride,
                                      current_saddle_point_state.get_delta_dual().data(),
                                      dual_stride,
                                      norm_squared_delta_dual_.data(),
                                      stream_pool_.get_stream(1)));
    dot_delta_Y_.record(stream_pool_.get_stream(1));

    // Wait on main stream for both dot to be done before launching the next kernel
    dot_delta_X_.stream_wait(stream_view_);
    dot_delta_Y_.stream_wait(stream_view_);
  } else {
    // In batch mode we don't need to parallelize the dot products since we already have many to launch
    batched_dot_product_handler_.batch_transform_reduce([&](i_t climber, rmm::cuda_stream_view stream){
      RAFT_CUBLAS_TRY(raft::linalg::detail::cublasdot(handle_ptr_->get_cublas_handle(),
        current_saddle_point_state.get_primal_size(),
        current_saddle_point_state.get_delta_primal().data() + climber * current_saddle_point_state.get_primal_size(),
        primal_stride,
        current_saddle_point_state.get_delta_primal().data() + climber * current_saddle_point_state.get_primal_size(),
        primal_stride,
        norm_squared_delta_primal_.data() + climber,
        stream));
    });
    batched_dot_product_handler_.batch_transform_reduce([&](i_t climber, rmm::cuda_stream_view stream){
      RAFT_CUBLAS_TRY(raft::linalg::detail::cublasdot(handle_ptr_->get_cublas_handle(),
        current_saddle_point_state.get_dual_size(),
        current_saddle_point_state.get_delta_dual().data() + climber * current_saddle_point_state.get_dual_size(),
        dual_stride,
        current_saddle_point_state.get_delta_dual().data() + climber * current_saddle_point_state.get_dual_size(),
        dual_stride,
        norm_squared_delta_dual_.data() + climber,
        stream));
    });
  }
}

template <typename i_t, typename f_t>
__global__ void compute_actual_stepsizes(
  const typename adaptive_step_size_strategy_t<i_t, f_t>::view_t step_size_strategy_view,
  raft::device_span<f_t> primal_step_size,
  raft::device_span<f_t> dual_step_size,
  int batch_size)
{
  const int id = threadIdx.x + blockIdx.x * blockDim.x;
  if (id >= batch_size) { return; }
  f_t step_size_     = step_size_strategy_view.step_size[id];
  f_t primal_weight_ = step_size_strategy_view.primal_weight[id];

  primal_step_size[id] = step_size_ / primal_weight_;
  dual_step_size[id]   = step_size_ * primal_weight_;
}

template <typename i_t, typename f_t>
void adaptive_step_size_strategy_t<i_t, f_t>::get_primal_and_dual_stepsizes(
  rmm::device_uvector<f_t>& primal_step_size, rmm::device_uvector<f_t>& dual_step_size)
{
  const int block_size = std::min(256, (batch_mode_ ? (0 + 3)/*@@*/ : 1));
  const int num_blocks = (batch_mode_ ? cuda::ceil_div((0 + 3)/*@@*/, block_size) : 1);
  compute_actual_stepsizes<i_t, f_t>
    <<<num_blocks, block_size, 0, stream_view_>>>(this->view(),
                                                  make_span(primal_step_size),
                                                  make_span(dual_step_size),
                                                  (batch_mode_ ? (0 + 3)/*@@*/ : 1));
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

template <typename i_t, typename f_t>
typename adaptive_step_size_strategy_t<i_t, f_t>::view_t
adaptive_step_size_strategy_t<i_t, f_t>::view()
{
  adaptive_step_size_strategy_t<i_t, f_t>::view_t v{};

  v.primal_weight   = raft::device_span<f_t>(primal_weight_->data(), primal_weight_->size());
  v.step_size       = raft::device_span<f_t>(step_size_->data(), step_size_->size());
  v.valid_step_size = raft::device_span<i_t>(thrust::raw_pointer_cast(valid_step_size_.data()), valid_step_size_.size());

  v.interaction = raft::device_span<f_t>(interaction_.data(), interaction_.size());

  v.norm_squared_delta_primal = raft::device_span<f_t>(norm_squared_delta_primal_.data(), norm_squared_delta_primal_.size());
  v.norm_squared_delta_dual   = raft::device_span<f_t>(norm_squared_delta_dual_.data(), norm_squared_delta_dual_.size());

  return v;
}

template <typename i_t, typename f_t>
bool adaptive_step_size_strategy_t<i_t, f_t>::all_invalid() const
{
  return std::all_of(valid_step_size_.begin(), valid_step_size_.end(), [](i_t v) { return v == -1; });
}

template <typename i_t, typename f_t>
void adaptive_step_size_strategy_t<i_t, f_t>::reset_valid_step_size()
{
  std::fill(valid_step_size_.begin(), valid_step_size_.end(), 0);
}

template <typename i_t, typename f_t>
i_t adaptive_step_size_strategy_t<i_t, f_t>::get_valid_step_size() const
{
  // TODO: batch mode
  return valid_step_size_[0];
}

#define INSTANTIATE(F_TYPE)                                                                    \
  template class adaptive_step_size_strategy_t<int, F_TYPE>;                                   \
  template __global__ void compute_actual_stepsizes<int, F_TYPE>(                              \
    const typename adaptive_step_size_strategy_t<int, F_TYPE>::view_t step_size_strategy_view, \
    raft::device_span<F_TYPE> primal_step_size,                                                                  \
    raft::device_span<F_TYPE> dual_step_size,                                                                     \
    int batch_size);                                                                            \
                                                                                               \
  template __global__ void compute_step_sizes_from_movement_and_interaction<int, F_TYPE>(      \
    typename adaptive_step_size_strategy_t<int, F_TYPE>::view_t step_size_strategy_view,       \
    raft::device_span<F_TYPE> primal_step_size,                                                                 \
    raft::device_span<F_TYPE> dual_step_size,                                                                   \
    int* pdhg_iteration,                                                                         \
    int batch_size);

#if MIP_INSTANTIATE_FLOAT
INSTANTIATE(float)
#endif

#if MIP_INSTANTIATE_DOUBLE
INSTANTIATE(double)
#endif

}  // namespace cuopt::linear_programming::detail
