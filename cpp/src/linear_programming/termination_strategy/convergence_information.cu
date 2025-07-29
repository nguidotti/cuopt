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

#include <linear_programming/pdlp_constants.hpp>
#include <linear_programming/termination_strategy/convergence_information.hpp>
#include <linear_programming/utils.cuh>
#include <mip/mip_constants.hpp>

#include <cuopt/linear_programming/pdlp/solver_settings.hpp>

#include <utilities/copy_helpers.hpp>

#include <raft/sparse/detail/cusparse_wrappers.h>
#include <raft/common/nvtx.hpp>
#include <raft/linalg/binary_op.cuh>
#include <raft/linalg/detail/cublas_wrappers.hpp>
#include <raft/linalg/eltwise.cuh>
#include <raft/linalg/ternary_op.cuh>
#include <raft/util/cuda_utils.cuh>

#include <thrust/device_ptr.h>
#include <thrust/iterator/zip_iterator.h>

#include <cub/cub.cuh>

namespace cuopt::linear_programming::detail {
template <typename i_t, typename f_t>
convergence_information_t<i_t, f_t>::convergence_information_t(
  raft::handle_t const* handle_ptr,
  problem_t<i_t, f_t>& op_problem,
  cusparse_view_t<i_t, f_t>& cusparse_view,
  i_t primal_size,
  i_t dual_size,
  bool batch_mode)
  : handle_ptr_(handle_ptr),
    stream_view_(handle_ptr_->get_stream()),
    primal_size_h_(primal_size),
    dual_size_h_(dual_size),
    problem_ptr(&op_problem),
    op_problem_cusparse_view_(cusparse_view),
    l2_norm_primal_linear_objective_{0.0, stream_view_},
    l2_norm_primal_right_hand_side_{0.0, stream_view_},
    primal_objective_{static_cast<size_t>((batch_mode ? (0 + 3)/*@@*/ : 1)), stream_view_},
    dual_objective_{static_cast<size_t>((batch_mode ? (0 + 3)/*@@*/ : 1)), stream_view_},
    reduced_cost_dual_objective_{static_cast<size_t>((batch_mode ? (0 + 3)/*@@*/ : 1)), stream_view_},
    l2_primal_residual_{static_cast<size_t>((batch_mode ? (0 + 3)/*@@*/ : 1)), stream_view_},
    l2_dual_residual_{static_cast<size_t>((batch_mode ? (0 + 3)/*@@*/ : 1)), stream_view_},
    linf_primal_residual_{0.0, stream_view_},
    linf_dual_residual_{0.0, stream_view_},
    nb_violated_constraints_{0, stream_view_},
    gap_{static_cast<size_t>((batch_mode ? (0 + 3)/*@@*/ : 1)), stream_view_},
    abs_objective_{static_cast<size_t>((batch_mode ? (0 + 3)/*@@*/ : 1)), stream_view_},
    l2_primal_variable_{static_cast<size_t>(batch_mode ? (0 + 3)/*@@*/ : 1), stream_view_},
    l2_dual_variable_{static_cast<size_t>((batch_mode ? (0 + 3)/*@@*/ : 1)), stream_view_},
    primal_residual_{static_cast<size_t>((batch_mode ? (0 + 3)/*@@*/ : 1) * dual_size_h_), stream_view_},
    dual_residual_{static_cast<size_t>((batch_mode ?  (0 + 3)/*@@*/ : 1) * primal_size_h_), stream_view_},
    reduced_cost_{static_cast<size_t>((batch_mode ? (0 + 3)/*@@*/ : 1) * primal_size_h_), stream_view_},
    bound_value_{static_cast<size_t>((batch_mode ? (0 + 3)/*@@*/ : 1) * std::max(primal_size_h_, dual_size_h_)), stream_view_},
    rmm_tmp_buffer_(0, stream_view_),
    reusable_device_scalar_value_1_{1.0, stream_view_},
    reusable_device_scalar_value_0_{0.0, stream_view_},
    reusable_device_scalar_value_neg_1_{-1.0, stream_view_},
    batch_mode_(batch_mode),
    batched_dot_product_handler_(batch_mode ? batched_transform_reduce_handler_t<i_t, f_t>((0 + 3)/*@@*/, handle_ptr_) : batched_transform_reduce_handler_t<i_t, f_t>())
{
  RAFT_CUDA_TRY(cudaMemsetAsync(primal_objective_.data(), 0, sizeof(f_t) * primal_objective_.size(), stream_view_));
  RAFT_CUDA_TRY(cudaMemsetAsync(dual_objective_.data(), 0, sizeof(f_t) * dual_objective_.size(), stream_view_));
  RAFT_CUDA_TRY(cudaMemsetAsync(reduced_cost_dual_objective_.data(), 0, sizeof(f_t) * reduced_cost_dual_objective_.size(), stream_view_));
  RAFT_CUDA_TRY(cudaMemsetAsync(gap_.data(), 0, sizeof(f_t) * gap_.size(), stream_view_));
  RAFT_CUDA_TRY(cudaMemsetAsync(abs_objective_.data(), 0, sizeof(f_t) * abs_objective_.size(), stream_view_));

  RAFT_CUDA_TRY(cudaMemsetAsync(l2_primal_variable_.data(), 0, sizeof(f_t) * l2_primal_variable_.size(), stream_view_));
  RAFT_CUDA_TRY(cudaMemsetAsync(l2_dual_variable_.data(), 0, sizeof(f_t) * l2_dual_variable_.size(), stream_view_));
  RAFT_CUDA_TRY(cudaMemsetAsync(l2_dual_residual_.data(), 0, sizeof(f_t) * l2_dual_residual_.size(), stream_view_));
  RAFT_CUDA_TRY(cudaMemsetAsync(l2_primal_residual_.data(), 0, sizeof(f_t) * l2_primal_residual_.size(), stream_view_));

  // TODO: batch different constraint bounds
  combine_constraint_bounds(
    *problem_ptr,
    primal_residual_,
    batch_mode_);  // primal_residual_ will contain abs max of bounds when
                        // finite, otherwise 0 //just reused allocated mem here

  // TODO: batch different objective coefficients
  // constant throughout solving, so precompute
  my_l2_norm<i_t, f_t>(
    problem_ptr->objective_coefficients, l2_norm_primal_linear_objective_.data(), handle_ptr_);
  // TODO: batch different constraint bounds
  my_l2_norm<i_t, f_t>(primal_residual_, l2_norm_primal_right_hand_side_.data(), handle_ptr_);

  void* d_temp_storage        = NULL;
  size_t temp_storage_bytes_1 = 0;
  cub::DeviceReduce::Sum(d_temp_storage,
                         temp_storage_bytes_1,
                         bound_value_.begin(),
                         dual_objective_.data(),
                         dual_size_h_,
                         stream_view_);

  size_t temp_storage_bytes_2 = 0;
  cub::DeviceReduce::Sum(d_temp_storage,
                         temp_storage_bytes_2,
                         bound_value_.begin(),
                         reduced_cost_dual_objective_.data(),
                         primal_size_h_,
                         stream_view_);

  size_of_buffer_       = std::max({temp_storage_bytes_1, temp_storage_bytes_2});
  rmm_tmp_buffer_.resize((batch_mode_ ? (0 + 3)/*@@*/ : 1) * size_of_buffer_, stream_view_);

  RAFT_CUDA_TRY(cudaMemsetAsync(
    primal_residual_.data(), 0.0, sizeof(f_t) * primal_residual_.size(), stream_view_));
  RAFT_CUDA_TRY(
    cudaMemsetAsync(dual_residual_.data(), 0.0, sizeof(f_t) * dual_residual_.size(), stream_view_));
}

template <typename i_t, typename f_t>
void convergence_information_t<i_t, f_t>::set_relative_dual_tolerance_factor(
  f_t dual_tolerance_factor)
{
  l2_norm_primal_linear_objective_.set_value_async(dual_tolerance_factor, stream_view_);
}

template <typename i_t, typename f_t>
void convergence_information_t<i_t, f_t>::set_relative_primal_tolerance_factor(
  f_t primal_tolerance_factor)
{
  l2_norm_primal_right_hand_side_.set_value_async(primal_tolerance_factor, stream_view_);
}

template <typename i_t, typename f_t>
f_t convergence_information_t<i_t, f_t>::get_relative_dual_tolerance_factor() const
{
  return l2_norm_primal_linear_objective_.value(stream_view_);
}

template <typename i_t, typename f_t>
f_t convergence_information_t<i_t, f_t>::get_relative_primal_tolerance_factor() const
{
  return l2_norm_primal_right_hand_side_.value(stream_view_);
}

template <typename i_t, typename f_t>
__global__ void compute_remaining_stats_kernel(
  typename convergence_information_t<i_t, f_t>::view_t convergence_information_view,
  int batch_size)
{
  const int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= batch_size) { return; }

  convergence_information_view.gap[idx] = raft::abs(convergence_information_view.primal_objective[idx] -
                                                convergence_information_view.dual_objective[idx]);
  convergence_information_view.abs_objective[idx] =
    raft::abs(convergence_information_view.primal_objective[idx]) +
    raft::abs(convergence_information_view.dual_objective[idx]);
}

template <typename i_t, typename f_t>
void convergence_information_t<i_t, f_t>::compute_convergence_information(
  pdhg_solver_t<i_t, f_t>& current_pdhg_solver,
  rmm::device_uvector<f_t>& primal_iterate,
  rmm::device_uvector<f_t>& dual_iterate,
  const rmm::device_uvector<f_t>& combined_bounds,
  const rmm::device_uvector<f_t>& objective_coefficients,
  const pdlp_solver_settings_t<i_t, f_t>& settings)
{
  cuopt_assert(primal_residual_.size() % l2_primal_residual_.size() == 0, "primal_iterate size must be a multiple of l2_primal_residual_ size");
  cuopt_assert(primal_iterate.size() % l2_primal_variable_.size() == 0, "primal_iterate size must be a multiple of l2_primal_variable_ size");
  cuopt_assert(dual_residual_.size() % l2_dual_residual_.size() == 0, "dual_iterate size must be a multiple of l2_dual_residual_ size");
  cuopt_assert(dual_iterate.size() % l2_dual_variable_.size() == 0, "dual_iterate size must be a multiple of l2_dual_variable_ size");
  cuopt_assert(l2_primal_residual_.size() == l2_primal_variable_.size(), "l2_primal_residual_ size must be equal to l2_primal_variable_ size");
  cuopt_assert(l2_primal_residual_.size() == l2_dual_residual_.size(), "l2_primal_residual_ size must be equal to l2_dual_residual_ size");
  cuopt_assert(l2_dual_residual_.size() == l2_dual_variable_.size(), "l2_dual_residual_ size must be equal to l2_dual_variable_ size");

  raft::common::nvtx::range fun_scope("compute_convergence_information");

  compute_primal_residual(op_problem_cusparse_view_, current_pdhg_solver.get_dual_tmp_resource());
  compute_primal_objective(primal_iterate);
  if (!batch_mode_) {
    my_l2_norm<i_t, f_t>(primal_residual_, l2_primal_residual_.data(), handle_ptr_);
  } else {
    batched_dot_product_handler_.batch_transform_reduce([&](i_t climber, rmm::cuda_stream_view stream){
      RAFT_CUBLAS_TRY(raft::linalg::detail::cublasnrm2(handle_ptr_->get_cublas_handle(),
        dual_size_h_,
        primal_residual_.data() + climber * dual_size_h_,
        1,
        l2_primal_residual_.data() + climber,
        stream));
    });
  }

  // If per_constraint_residual is false we still need to perform the l2 since it's used in kkt
  if (settings.per_constraint_residual) {
    // TODO: batch mode
    cuopt_assert(!batch_mode_, "Batch mode not supported for per_constraint_residual");
    // Compute the linf of (residual_i - rel * b_i)
    thrust::device_ptr<f_t> result_ptr(linf_primal_residual_.data());
    const f_t neutral = f_t(0.0);

    if (settings.save_best_primal_so_far) {
      const i_t zero_int = 0;
      nb_violated_constraints_.set_value_async(zero_int, handle_ptr_->get_stream());
      *result_ptr = thrust::transform_reduce(
        handle_ptr_->get_thrust_policy(),
        thrust::make_zip_iterator(primal_residual_.cbegin(), combined_bounds.cbegin()),
        thrust::make_zip_iterator(primal_residual_.cend(), combined_bounds.cend()),
        relative_residual_t<i_t, f_t>{settings.tolerances.relative_primal_tolerance},
        neutral,
        thrust::maximum<f_t>());
    } else {
      *result_ptr = thrust::transform_reduce(
        handle_ptr_->get_thrust_policy(),
        thrust::make_zip_iterator(primal_residual_.cbegin(), combined_bounds.cbegin()),
        thrust::make_zip_iterator(primal_residual_.cend(), combined_bounds.cend()),
        relative_residual_t<i_t, f_t>{settings.tolerances.relative_primal_tolerance},
        neutral,
        thrust::maximum<f_t>());
    }
  }
  if (!batch_mode_) {
    my_l2_norm<i_t, f_t>(primal_iterate, l2_primal_variable_.data(), handle_ptr_);
  } else {
    batched_dot_product_handler_.batch_transform_reduce([&](i_t climber, rmm::cuda_stream_view stream){
      RAFT_CUBLAS_TRY(raft::linalg::detail::cublasnrm2(handle_ptr_->get_cublas_handle(),
        primal_size_h_,
        primal_iterate.data() + climber * primal_size_h_,
        1,
        l2_primal_variable_.data() + climber,
        stream));
    });
  }

  compute_dual_residual(
    op_problem_cusparse_view_, current_pdhg_solver.get_primal_tmp_resource(), primal_iterate);
  compute_dual_objective(dual_iterate);
  if (!batch_mode_) {
    my_l2_norm<i_t, f_t>(dual_residual_, l2_dual_residual_.data(), handle_ptr_);
  } else {
    batched_dot_product_handler_.batch_transform_reduce([&](i_t climber, rmm::cuda_stream_view stream){
      RAFT_CUBLAS_TRY(raft::linalg::detail::cublasnrm2(handle_ptr_->get_cublas_handle(),
        primal_size_h_,
        dual_residual_.data() + climber * primal_size_h_,
        1,
        l2_dual_residual_.data() + climber,
        stream));
    });
  }

  // If per_constraint_residual is false we still need to perform the l2 since it's used in kkt
  if (settings.per_constraint_residual) {
    // TODO: batch mode
    cuopt_assert(!batch_mode_, "Batch mode not supported for per_constraint_residual");
    // Compute the linf of (residual_i - rel * c_i)
    thrust::device_ptr<f_t> result_ptr(linf_dual_residual_.data());
    const f_t neutral = f_t(0.0);

    *result_ptr = thrust::transform_reduce(
      handle_ptr_->get_thrust_policy(),
      thrust::make_zip_iterator(dual_residual_.cbegin(), objective_coefficients.cbegin()),
      thrust::make_zip_iterator(dual_residual_.cend(), objective_coefficients.cend()),
      relative_residual_t<i_t, f_t>{settings.tolerances.relative_dual_tolerance},
      neutral,
      thrust::maximum<f_t>());
  }
  if (!batch_mode_) {
    my_l2_norm<i_t, f_t>(dual_iterate, l2_dual_variable_.data(), handle_ptr_);
  } else {
    batched_dot_product_handler_.batch_transform_reduce([&](i_t climber, rmm::cuda_stream_view stream){
      RAFT_CUBLAS_TRY(raft::linalg::detail::cublasnrm2(handle_ptr_->get_cublas_handle(),
        dual_size_h_,
        dual_iterate.data() + climber * dual_size_h_,
        1,
        l2_dual_variable_.data() + climber,
        stream));
    });
  }

  const int block_size = (batch_mode_ ? std::min(256, (0 + 3)/*@@*/) : 1);
  const int grid_size = (batch_mode_ ? cuda::ceil_div((0 + 3)/*@@*/, block_size) : 1);
  compute_remaining_stats_kernel<i_t, f_t><<<grid_size, block_size, 0, stream_view_>>>(this->view(), (batch_mode_ ? (0 + 3)/*@@*/ : 1));
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  //  cleanup for next termination evaluation
  RAFT_CUDA_TRY(cudaMemsetAsync(
    primal_residual_.data(), 0.0, sizeof(f_t) * primal_residual_.size(), stream_view_));
  RAFT_CUDA_TRY(
    cudaMemsetAsync(dual_residual_.data(), 0.0, sizeof(f_t) * dual_residual_.size(), stream_view_));
}

template <typename i_t, typename f_t>
void convergence_information_t<i_t, f_t>::compute_primal_residual(
  cusparse_view_t<i_t, f_t>& cusparse_view, rmm::device_uvector<f_t>& tmp_dual)
{
  raft::common::nvtx::range fun_scope("compute_primal_residual");

  // primal_product
  if (!batch_mode_) {
  RAFT_CUSPARSE_TRY(
      raft::sparse::detail::cusparsespmv(handle_ptr_->get_cusparse_handle(),
                                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                                        reusable_device_scalar_value_1_.data(),
                                        cusparse_view.A,
                                        cusparse_view.primal_solution,
                                        reusable_device_scalar_value_0_.data(),
                                        cusparse_view.tmp_dual,
                                        CUSPARSE_SPMV_CSR_ALG2,
                                        (f_t*)cusparse_view.buffer_non_transpose.data(),
                                        stream_view_));
      // The constraint bound violations for the first part of the residual
    raft::linalg::ternaryOp<f_t, violation<f_t>>(primal_residual_.data(),
                                                 tmp_dual.data(),
                                                 problem_ptr->constraint_lower_bounds.data(),
                                                 problem_ptr->constraint_upper_bounds.data(),
                                                 dual_size_h_,
                                                 violation<f_t>(),
                                                 stream_view_);
  } else {
    RAFT_CUSPARSE_TRY(raft::sparse::detail::cusparsespmm(handle_ptr_->get_cusparse_handle(),
                                       CUSPARSE_OPERATION_NON_TRANSPOSE,
                                       CUSPARSE_OPERATION_NON_TRANSPOSE,
                                       reusable_device_scalar_value_1_.data(),
                                       cusparse_view.A,
                                       cusparse_view.batch_primal_solutions,
                                       reusable_device_scalar_value_0_.data(),
                                       cusparse_view.batch_tmp_duals,
                                       CUSPARSE_SPMM_CSR_ALG3,
                                       (f_t*)cusparse_view.buffer_non_transpose_batch.data(),
                                       stream_view_));
            cub::DeviceTransform::Transform(
                                        cuda::std::make_tuple(tmp_dual.data(),
                                                              thrust::make_transform_iterator(
                                                                thrust::make_counting_iterator(0),
                                                                problem_wrapped_iterator<f_t>(problem_ptr->constraint_lower_bounds.data(),
                                                                                             dual_size_h_)),
                                                              thrust::make_transform_iterator(
                                                                thrust::make_counting_iterator(0),
                                                                problem_wrapped_iterator<f_t>(problem_ptr->constraint_upper_bounds.data(),
                                                                                             dual_size_h_))
                                                              ),
                                        primal_residual_.data(),
                                        primal_residual_.size(),
                                        violation<f_t>(),
                                        stream_view_);
  }

#ifdef PDLP_DEBUG_MODE
  RAFT_CUDA_TRY(cudaDeviceSynchronize());
#endif
}

template <typename i_t, typename f_t>
__global__ void apply_objective_scaling_and_offset(raft::device_span<f_t> objective,
                                                   f_t objective_scaling_factor,
                                                   f_t objective_offset,
                                                   int batch_size)
{
  const int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= batch_size) { return; }

  objective[idx] = (objective_scaling_factor * objective[idx]) + objective_offset;
}

template <typename i_t, typename f_t>
void convergence_information_t<i_t, f_t>::compute_primal_objective(
  rmm::device_uvector<f_t>& primal_solution)
{
  raft::common::nvtx::range fun_scope("compute_primal_objective");

  if (!batch_mode_) {
  RAFT_CUBLAS_TRY(raft::linalg::detail::cublasdot(handle_ptr_->get_cublas_handle(),
                                                  (int)primal_size_h_,
                                                  primal_solution.data(),
                                                  primal_stride,
                                                  problem_ptr->objective_coefficients.data(),
                                                  primal_stride,
                                                  primal_objective_.data(),
                                                  stream_view_));
  } else {
    // TODO: batch different objective coefficients
    batched_dot_product_handler_.batch_transform_reduce([&](i_t climber, rmm::cuda_stream_view stream){
      RAFT_CUBLAS_TRY(raft::linalg::detail::cublasdot(handle_ptr_->get_cublas_handle(),
        primal_size_h_,
        primal_solution.data() + climber * primal_size_h_,
        1,
        problem_ptr->objective_coefficients.data(),
        1,
        primal_objective_.data() + climber,
        stream));
    });
  }

  // primal_objective = 1 * (primal_objective + 0) = primal_objective
  if (problem_ptr->presolve_data.objective_scaling_factor != 1 ||
      problem_ptr->presolve_data.objective_offset != 0) {
    const int block_size = (batch_mode_ ? std::min(256, (0 + 3)/*@@*/) : 1);
    const int grid_size = (batch_mode_ ? cuda::ceil_div((0 + 3)/*@@*/, block_size) : 1);
    apply_objective_scaling_and_offset<i_t, f_t>
      <<<grid_size, block_size, 0, stream_view_>>>(make_span(primal_objective_),
                                  problem_ptr->presolve_data.objective_scaling_factor,
                                  problem_ptr->presolve_data.objective_offset,
                                  batch_mode_ ? (0 + 3)/*@@*/ : 1);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  }
}

template <typename i_t, typename f_t>
void convergence_information_t<i_t, f_t>::compute_dual_residual(
  cusparse_view_t<i_t, f_t>& cusparse_view,
  rmm::device_uvector<f_t>& tmp_primal,
  rmm::device_uvector<f_t>& primal_solution)
{
  cuopt_assert(tmp_primal.size() == primal_solution.size(), "tmp_primal size must be equal to primal_solution size");
  cuopt_assert(dual_residual_.size() == primal_solution.size(), "dual_residual_ size must be equal to primal_solution size");
  cuopt_assert(reduced_cost_.size() == primal_solution.size(), "reduced_cost_ size must be equal to primal_solution size");

  raft::common::nvtx::range fun_scope("compute_dual_residual");

  // compute objective product (Q*x) if QP

  // gradient is recomputed with the dual solution that has been computed since the gradient was
  // last computed
  //  c-K^Ty -> copy c to gradient first
  if (!batch_mode_) {
    raft::copy(
      tmp_primal.data(), problem_ptr->objective_coefficients.data(), primal_size_h_, stream_view_);

      RAFT_CUSPARSE_TRY(raft::sparse::detail::cusparsespmv(handle_ptr_->get_cusparse_handle(),
                                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                           reusable_device_scalar_value_neg_1_.data(),
                                                           cusparse_view.A_T,
                                                           cusparse_view.dual_solution,
                                                           reusable_device_scalar_value_1_.data(),
                                                           cusparse_view.tmp_primal,
                                                           CUSPARSE_SPMV_CSR_ALG2,
                                                           (f_t*)cusparse_view.buffer_transpose.data(),
                                                           stream_view_));
  } else {
    // TODO: batch different objective coefficients
    thrust::copy_n(
      handle_ptr_->get_thrust_policy(),
      thrust::make_transform_iterator(
        thrust::make_counting_iterator(0),
        problem_wrapped_iterator<f_t>(problem_ptr->objective_coefficients.data(),
                                     primal_size_h_)),
      tmp_primal.size(),
      tmp_primal.data()
    );
    RAFT_CUSPARSE_TRY(raft::sparse::detail::cusparsespmm(handle_ptr_->get_cusparse_handle(),
                                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                           reusable_device_scalar_value_neg_1_.data(),
                                                           cusparse_view.A_T,
                                                           cusparse_view.batch_dual_solutions,
                                                           reusable_device_scalar_value_1_.data(),
                                                           cusparse_view.batch_tmp_primals,
                                                           CUSPARSE_SPMM_CSR_ALG3,
                                                           (f_t*)cusparse_view.buffer_transpose_batch.data(),
                                                           stream_view_));
  }


  compute_reduced_cost_from_primal_gradient(tmp_primal, primal_solution);

  // primal_gradient - reduced_costs
  raft::linalg::eltwiseSub(dual_residual_.data(),
                           tmp_primal.data(),  // primal_gradient
                           reduced_cost_.data(),
                           reduced_cost_.size(),
                           stream_view_);
}

template <typename i_t, typename f_t>
void convergence_information_t<i_t, f_t>::compute_dual_objective(
  rmm::device_uvector<f_t>& dual_solution)
{
  raft::common::nvtx::range fun_scope("compute_dual_objective");

  // for QP would need to add + problem.objective_constant - 0.5 * objective_product' *
  // primal_solution (iteration_stats.jl:186)

  // the value of y term in the objective of the dual problem, see[]
  //  (l^c)^T[y]_+ − (u^c)^T[y]_− in the dual objective

  if (!batch_mode_) {
    raft::linalg::ternaryOp(bound_value_.data(),
                            dual_solution.data(),
                            problem_ptr->constraint_lower_bounds.data(),
                            problem_ptr->constraint_upper_bounds.data(),
                            dual_size_h_,
                            bound_value_reduced_cost_product<f_t>(),
                            stream_view_);

    cub::DeviceReduce::Sum(rmm_tmp_buffer_.data(),
                          size_of_buffer_,
                          bound_value_.begin(),
                          dual_objective_.data(),
                          dual_size_h_,
                          stream_view_);
  } else {
    // TODO: batch mode different constraint bounds
    cub::DeviceTransform::Transform(
      cuda::std::make_tuple(dual_solution.data(),
      thrust::make_transform_iterator(
        thrust::make_counting_iterator(0),
        problem_wrapped_iterator<f_t>(problem_ptr->constraint_lower_bounds.data(),
                                     dual_size_h_)),
      thrust::make_transform_iterator(
        thrust::make_counting_iterator(0),
        problem_wrapped_iterator<f_t>(problem_ptr->constraint_upper_bounds.data(),
                                     dual_size_h_))),
      bound_value_.data(),
      dual_solution.size(),
      bound_value_reduced_cost_product<f_t>(),
      stream_view_);

      batched_dot_product_handler_.batch_transform_reduce([&](i_t climber, rmm::cuda_stream_view stream){
        cub::DeviceReduce::Sum(rmm_tmp_buffer_.data() + climber * size_of_buffer_,
                              size_of_buffer_,
                              bound_value_.begin() + climber * dual_size_h_,
                              dual_objective_.data() + climber,
                              dual_size_h_,
                              stream);
      });
  }

  compute_reduced_costs_dual_objective_contribution();

  raft::linalg::eltwiseAdd(dual_objective_.data(),
                           dual_objective_.data(),
                           reduced_cost_dual_objective_.data(),
                           reduced_cost_dual_objective_.size(),
                           stream_view_);

    // dual_objective = 1 * (dual_objective + 0) = dual_objective
    if (problem_ptr->presolve_data.objective_scaling_factor != 1 ||
    problem_ptr->presolve_data.objective_offset != 0) {
    const int block_size = (batch_mode_ ? std::min(256, (0 + 3)/*@@*/) : 1);
    const int grid_size = (batch_mode_ ? cuda::ceil_div((0 + 3)/*@@*/, block_size) : 1);
    apply_objective_scaling_and_offset<i_t, f_t>
      <<<grid_size, block_size, 0, stream_view_>>>(make_span(dual_objective_),
                                                   problem_ptr->presolve_data.objective_scaling_factor,
                                                   problem_ptr->presolve_data.objective_offset,
                                                   batch_mode_ ? (0 + 3)/*@@*/ : 1);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  }

  #ifdef PDLP_DEBUG_MODE
    RAFT_CUDA_TRY(cudaDeviceSynchronize());
  #endif
}

template <typename i_t, typename f_t>
void convergence_information_t<i_t, f_t>::compute_reduced_cost_from_primal_gradient(
  const rmm::device_uvector<f_t>& primal_gradient, const rmm::device_uvector<f_t>& primal_solution)
{
  cuopt_assert(primal_gradient.size() == primal_solution.size(), "primal_gradient size must be equal to primal_solution size");
  // >= since we reuse it for primal and dual
  cuopt_assert(bound_value_.size() >= primal_gradient.size(), "bound_value_ size must be equal to primal_gradient size");
  cuopt_assert(reduced_cost_.size() == primal_gradient.size(), "reduced_cost_ size must be equal to primal_gradient size");

  raft::common::nvtx::range fun_scope("compute_reduced_cost_from_primal_gradient");

  if (!batch_mode_) {
    raft::linalg::ternaryOp(bound_value_.data(),
    primal_gradient.data(),
    problem_ptr->variable_lower_bounds.data(),
    problem_ptr->variable_upper_bounds.data(),
    primal_size_h_,
    bound_value_gradient<f_t>(),
    stream_view_);
  } else {
    // TODO: batch mode different variable bounds
    cub::DeviceTransform::Transform(
      cuda::std::make_tuple(primal_gradient.data(),
      thrust::make_transform_iterator(
        thrust::make_counting_iterator(0),
        problem_wrapped_iterator<f_t>(problem_ptr->variable_lower_bounds.data(),
                                     primal_size_h_)),
      thrust::make_transform_iterator(
        thrust::make_counting_iterator(0),
        problem_wrapped_iterator<f_t>(problem_ptr->variable_upper_bounds.data(),
                                     primal_size_h_))),
      bound_value_.data(),
      primal_gradient.size(),
      bound_value_gradient<f_t>(),
      stream_view_);
  }

  if (pdlp_hyper_params::handle_some_primal_gradients_on_finite_bounds_as_residuals) {
    raft::linalg::ternaryOp(reduced_cost_.data(),
                            primal_solution.data(),
                            bound_value_.data(),
                            primal_gradient.data(),
                            primal_solution.size(),
                            copy_gradient_if_should_be_reduced_cost<f_t>(),
                            stream_view_);
  } else {
    raft::linalg::binaryOp(reduced_cost_.data(),
                           bound_value_.data(),
                           primal_gradient.data(),
                           primal_solution.size(),
                           copy_gradient_if_finite_bounds<f_t>(),
                           stream_view_);
  }
}

template <typename i_t, typename f_t>
void convergence_information_t<i_t, f_t>::compute_reduced_costs_dual_objective_contribution()
{
  raft::common::nvtx::range fun_scope("compute_reduced_costs_dual_objective_contribution");

  // if reduced cost is positive -> lower bound, negative -> upper bounds, 0 -> 0
  // if bound_val is not finite let element be -inf, otherwise bound_value*reduced_cost
  if (!batch_mode_) {
    raft::linalg::ternaryOp(bound_value_.data(),
                            reduced_cost_.data(),
                            problem_ptr->variable_lower_bounds.data(),
                            problem_ptr->variable_upper_bounds.data(),
                            primal_size_h_,
                            bound_value_reduced_cost_product<f_t>(),
                            stream_view_);

    // sum over bound_value*reduced_cost, but should be -inf if any element is -inf
    cub::DeviceReduce::Sum(rmm_tmp_buffer_.data(),
                          size_of_buffer_,
                          bound_value_.begin(),
                          reduced_cost_dual_objective_.data(),
                          primal_size_h_,
                          stream_view_);
  } else {
    // TODO: batch mode different variable bounds
    cub::DeviceTransform::Transform(
      cuda::std::make_tuple(reduced_cost_.data(),
      thrust::make_transform_iterator(
        thrust::make_counting_iterator(0),
        problem_wrapped_iterator<f_t>(problem_ptr->variable_lower_bounds.data(),
                                     primal_size_h_)),
      thrust::make_transform_iterator(
        thrust::make_counting_iterator(0),
        problem_wrapped_iterator<f_t>(problem_ptr->variable_upper_bounds.data(),
                                     primal_size_h_))),
      bound_value_.data(),
      reduced_cost_.size(),
      bound_value_reduced_cost_product<f_t>(),
      stream_view_);

    batched_dot_product_handler_.batch_transform_reduce([&](i_t climber, rmm::cuda_stream_view stream){
      cub::DeviceReduce::Sum(rmm_tmp_buffer_.data() + climber * size_of_buffer_,
                            size_of_buffer_,
                            bound_value_.begin() + climber * primal_size_h_,
                            reduced_cost_dual_objective_.data() + climber,
                            primal_size_h_,
                            stream);
    });
  }
}

template <typename i_t, typename f_t>
rmm::device_uvector<f_t>& convergence_information_t<i_t, f_t>::get_reduced_cost()
{
  return reduced_cost_;
}

template <typename i_t, typename f_t>
const rmm::device_uvector<f_t>& convergence_information_t<i_t, f_t>::get_l2_primal_residual() const
{
  return l2_primal_residual_;
}

template <typename i_t, typename f_t>
const rmm::device_uvector<f_t>& convergence_information_t<i_t, f_t>::get_primal_objective() const
{
  return primal_objective_;
}

template <typename i_t, typename f_t>
const rmm::device_uvector<f_t>& convergence_information_t<i_t, f_t>::get_dual_objective() const
{
  return dual_objective_;
}

template <typename i_t, typename f_t>
const rmm::device_uvector<f_t>& convergence_information_t<i_t, f_t>::get_l2_dual_residual() const
{
  return l2_dual_residual_;
}

template <typename i_t, typename f_t>
const rmm::device_scalar<f_t>&
convergence_information_t<i_t, f_t>::get_relative_linf_primal_residual() const
{
  return linf_primal_residual_;
}

template <typename i_t, typename f_t>
const rmm::device_scalar<f_t>&
convergence_information_t<i_t, f_t>::get_relative_linf_dual_residual() const
{
  return linf_dual_residual_;
}

template <typename i_t, typename f_t>
const rmm::device_uvector<f_t>& convergence_information_t<i_t, f_t>::get_gap() const
{
  return gap_;
}

template <typename i_t, typename f_t>
f_t convergence_information_t<i_t, f_t>::get_relative_gap_value() const
{
  // TODO: batch mode
  return gap_.element(0, stream_view_) / (f_t(1.0) + abs_objective_.element(0, stream_view_));
}

template <typename i_t, typename f_t>
f_t convergence_information_t<i_t, f_t>::get_relative_l2_primal_residual_value() const
{
  // TODO: batch mode
  return l2_primal_residual_.element(0, stream_view_) /
         (f_t(1.0) + l2_norm_primal_right_hand_side_.value(stream_view_));
}

template <typename i_t, typename f_t>
f_t convergence_information_t<i_t, f_t>::get_relative_l2_dual_residual_value() const
{
  // TODO: batch mode
  return l2_dual_residual_.element(0, stream_view_) /
         (f_t(1.0) + l2_norm_primal_linear_objective_.value(stream_view_));
}

template <typename i_t, typename f_t>
typename convergence_information_t<i_t, f_t>::view_t convergence_information_t<i_t, f_t>::view()
{
  convergence_information_t<i_t, f_t>::view_t v;
  v.primal_size = primal_size_h_;
  v.dual_size   = dual_size_h_;

  v.l2_norm_primal_linear_objective = l2_norm_primal_linear_objective_.data();
  v.l2_norm_primal_right_hand_side  = l2_norm_primal_right_hand_side_.data();

  v.primal_objective               = make_span(primal_objective_);
  v.dual_objective                 = make_span(dual_objective_);
  v.l2_primal_residual             = make_span(l2_primal_residual_);
  v.l2_dual_residual               = make_span(l2_dual_residual_);
  v.relative_l_inf_primal_residual = linf_primal_residual_.data();
  v.relative_l_inf_dual_residual   = linf_dual_residual_.data();

  v.gap           = make_span(gap_);
  v.abs_objective = make_span(abs_objective_);

  v.l2_primal_variable = make_span(l2_primal_variable_);
  v.l2_dual_variable   = make_span(l2_dual_variable_);

  v.primal_residual = make_span(primal_residual_);
  v.dual_residual   = make_span(dual_residual_);
  v.reduced_cost    = make_span(reduced_cost_);
  v.bound_value     = make_span(bound_value_);

  return v;
}

template <typename i_t, typename f_t>
typename convergence_information_t<i_t, f_t>::primal_quality_adapter_t
convergence_information_t<i_t, f_t>::to_primal_quality_adapter(
  bool is_primal_feasible) const noexcept
{
  // TODO: batch mode
  return {is_primal_feasible,
          nb_violated_constraints_.value(stream_view_),
          l2_primal_residual_.element(0, stream_view_),
          primal_objective_.element(0, stream_view_)};
}

#if MIP_INSTANTIATE_FLOAT
template class convergence_information_t<int, float>;

template __global__ void compute_remaining_stats_kernel<int, float>(
  typename convergence_information_t<int, float>::view_t convergence_information_view);
#endif

#if MIP_INSTANTIATE_DOUBLE
template class convergence_information_t<int, double>;

template __global__ void compute_remaining_stats_kernel<int, double>(
  typename convergence_information_t<int, double>::view_t convergence_information_view, int batch_size);
#endif

}  // namespace cuopt::linear_programming::detail
