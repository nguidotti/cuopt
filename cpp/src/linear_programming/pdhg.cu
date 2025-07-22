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
#include <linear_programming/pdhg.hpp>
#include <linear_programming/pdlp_constants.hpp>
#include <linear_programming/utilities/ping_pong_graph.cuh>
#include <linear_programming/utils.cuh>
#include <mip/mip_constants.hpp>

#include <raft/sparse/detail/cusparse_macros.h>
#include <raft/sparse/detail/cusparse_wrappers.h>
#include <raft/common/nvtx.hpp>
#include <raft/linalg/binary_op.cuh>
#include <raft/linalg/eltwise.cuh>
#include <raft/linalg/ternary_op.cuh>

#include <cub/cub.cuh>

#include <cusparse_v2.h>

namespace cuopt::linear_programming::detail {

template <typename i_t, typename f_t>
pdhg_solver_t<i_t, f_t>::pdhg_solver_t(raft::handle_t const* handle_ptr,
                                       problem_t<i_t, f_t>& op_problem_scaled,
                                       bool batch_mode)
  : handle_ptr_(handle_ptr),
    stream_view_(handle_ptr_->get_stream()),
    problem_ptr(&op_problem_scaled),
    primal_size_h_(problem_ptr->n_variables),
    dual_size_h_(problem_ptr->n_constraints),
    current_saddle_point_state_{handle_ptr_, problem_ptr->n_variables, problem_ptr->n_constraints, batch_mode},
    tmp_primal_{static_cast<size_t>(problem_ptr->n_variables), stream_view_},
    batch_tmp_primals_{static_cast<size_t>(problem_ptr->n_variables * (0 + 3)/*@@*/), stream_view_},
    tmp_dual_{static_cast<size_t>(problem_ptr->n_constraints), stream_view_},
    potential_next_primal_solution_{(batch_mode ? static_cast<size_t>(problem_ptr->n_variables * (0 + 3)/*@@*/) : static_cast<size_t>(problem_ptr->n_variables)), stream_view_},
    potential_next_dual_solution_{(batch_mode ? static_cast<size_t>(problem_ptr->n_constraints * (0 + 3)/*@@*/) : static_cast<size_t>(problem_ptr->n_constraints)), stream_view_},
    total_pdhg_iterations_{0},
    cusparse_view_{handle_ptr_,
                   op_problem_scaled,
                   current_saddle_point_state_,
                   tmp_primal_,
                   batch_tmp_primals_,
                   tmp_dual_,
                   potential_next_dual_solution_},
    reusable_device_scalar_value_1_{1.0, stream_view_},
    reusable_device_scalar_value_0_{0.0, stream_view_},
    reusable_device_scalar_value_neg_1_{f_t(-1.0), stream_view_},
    reusable_device_scalar_1_{stream_view_},
    graph_all{stream_view_},
    graph_prim_proj_gradient_dual{stream_view_},
    d_total_pdhg_iterations_{0, stream_view_}
{
  batch_mode_ = batch_mode;
}

template <typename i_t, typename f_t>
rmm::device_scalar<i_t>& pdhg_solver_t<i_t, f_t>::get_d_total_pdhg_iterations()
{
  return d_total_pdhg_iterations_;
}

template <typename i_t, typename f_t>
void pdhg_solver_t<i_t, f_t>::compute_next_dual_solution(rmm::device_uvector<f_t>& dual_step_size)
{
  raft::common::nvtx::range fun_scope("compute_next_dual_solution");
  // proj(y+sigma(b-K(2x'-x)))
  // rewritten as proj(y+sigma(b-K(x'+delta_x)))
  // with the introduction of constraint lower and upper bounds the b
  // term no longer exists, but instead becomes
  // max(min(0, sigma*constraint_upper+primal_product),sigma*constraint_lower+primal_product)
  // where primal_product = y-sigma(K(x'+delta_x))

  // x+delta_x
  // Done in previous function

  // K(x'+delta_x)
  if (!batch_mode_) {
  RAFT_CUSPARSE_TRY(
    raft::sparse::detail::cusparsespmv(handle_ptr_->get_cusparse_handle(),
                                       CUSPARSE_OPERATION_NON_TRANSPOSE,
                                       reusable_device_scalar_value_1_.data(),  // 1
                                       cusparse_view_.A,
                                       cusparse_view_.tmp_primal,
                                       reusable_device_scalar_value_0_.data(),  // 1
                                       cusparse_view_.dual_gradient,
                                       CUSPARSE_SPMV_CSR_ALG2,
                                       (f_t*)cusparse_view_.buffer_non_transpose.data(),
                                       stream_view_));
  // y - (sigma*dual_gradient)
  // max(min(0, sigma*constraint_upper+primal_product), sigma*constraint_lower+primal_product)
  // Each element of y - (sigma*dual_gradient) of the min is the critical point
  // of the respective 1D minimization problem if it's negative.
  // Likewise the argument to the max is the critical point if
  // positive.

  // All is fused in a single call to limit number of read / write in memory
  cub::DeviceTransform::Transform(
    cuda::std::make_tuple(current_saddle_point_state_.get_dual_solution().data(),
                          current_saddle_point_state_.get_dual_gradient().data(),
                          problem_ptr->constraint_lower_bounds.data(),
                          problem_ptr->constraint_upper_bounds.data()),
    thrust::make_zip_iterator(potential_next_dual_solution_.data(),
                              current_saddle_point_state_.get_delta_dual().data()),
    dual_size_h_,
    dual_projection<f_t>(dual_step_size.data()),
    stream_view_);
  } else {
    // TMP: for now just copy in and out dual in the matrix to make sure SpMM is working
    raft::sparse::detail::cusparsespmm(handle_ptr_->get_cusparse_handle(),
               CUSPARSE_OPERATION_NON_TRANSPOSE,
               CUSPARSE_OPERATION_NON_TRANSPOSE,
               reusable_device_scalar_value_1_.data(),
               cusparse_view_.A,
               cusparse_view_.batch_tmp_primals,
               reusable_device_scalar_value_0_.data(),
               cusparse_view_.batch_dual_gradients,
               CUSPARSE_SPMM_CSR_ALG3,
               (f_t*)cusparse_view_.buffer_non_transpose_batch.data(),
               stream_view_);
  // y - (sigma*dual_gradient)
  // max(min(0, sigma*constraint_upper+primal_product), sigma*constraint_lower+primal_product)
  // Each element of y - (sigma*dual_gradient) of the min is the critical point
  // of the respective 1D minimization problem if it's negative.
  // Likewise the argument to the max is the critical point if
  // positive.

  // All is fused in a single call to limit number of read / write in memory
  cub::DeviceTransform::Transform(
    cuda::std::make_tuple(current_saddle_point_state_.get_dual_solution().data(),
                          current_saddle_point_state_.get_dual_gradient().data(),
                          thrust::make_transform_iterator(
                            thrust::make_counting_iterator(0),
                            problem_wrapped_iterator<f_t>(problem_ptr->constraint_lower_bounds.data(),
                                                         dual_size_h_)),
                          thrust::make_transform_iterator(
                            thrust::make_counting_iterator(0),
                            problem_wrapped_iterator<f_t>(problem_ptr->constraint_upper_bounds.data(),
                                                         dual_size_h_)),
                          thrust::make_transform_iterator(
                            thrust::make_counting_iterator(0),
                            batch_wrapped_iterator<f_t>(dual_step_size.data(),
                                                         dual_size_h_))
                          ),
    thrust::make_zip_iterator(potential_next_dual_solution_.data(),
                              current_saddle_point_state_.get_delta_dual().data()),
    dual_size_h_ * (0 + 3)/*@@*/,
    batch_dual_projection<f_t>(),
    stream_view_);
  }
}

template <typename i_t, typename f_t>
void pdhg_solver_t<i_t, f_t>::compute_At_y()
{
  // A_t @ y
  if (!batch_mode_) {
    RAFT_CUSPARSE_TRY(raft::sparse::detail::cusparsespmv(handle_ptr_->get_cusparse_handle(),
                                                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                        reusable_device_scalar_value_1_.data(),
                                                        cusparse_view_.A_T,
                                                        cusparse_view_.dual_solution,
                                                        reusable_device_scalar_value_0_.data(),
                                                        cusparse_view_.current_AtY,
                                                        CUSPARSE_SPMV_CSR_ALG2,
                                                        (f_t*)cusparse_view_.buffer_transpose.data(),
                                                        stream_view_));
  } else {
    // TODO: for batch mode if only a single one has restarted to average most likely faster to recompute the whole thing
      RAFT_CUSPARSE_TRY(raft::sparse::detail::cusparsespmm(handle_ptr_->get_cusparse_handle(),
                                                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                        reusable_device_scalar_value_1_.data(),
                                                        cusparse_view_.A_T,
                                                        cusparse_view_.batch_dual_solutions,
                                                        reusable_device_scalar_value_0_.data(),
                                                        cusparse_view_.batch_current_AtYs,
                                                        CUSPARSE_SPMM_CSR_ALG3,
                                                        (f_t*)cusparse_view_.buffer_transpose_batch.data(),
                                                        stream_view_));
  }
}

template <typename i_t, typename f_t>
void pdhg_solver_t<i_t, f_t>::compute_primal_projection_with_gradient(
  rmm::device_uvector<f_t>& primal_step_size)
{
  // Applying *c -* A_t @ y
  // x-(tau*primal_gradient)
  // project by max(min(x[i], upperbound[i]),lowerbound[i])
  // compute delta_primal x'-x

  // All is fused in a single call to limit number of read / write in memory
  if(!batch_mode_) {
  cub::DeviceTransform::Transform(
    cuda::std::make_tuple(current_saddle_point_state_.get_primal_solution().data(),
                          problem_ptr->objective_coefficients.data(),
                          current_saddle_point_state_.get_current_AtY().data(),
                          problem_ptr->variable_lower_bounds.data(),
                          problem_ptr->variable_upper_bounds.data()),
    thrust::make_zip_iterator(potential_next_primal_solution_.data(),
                              current_saddle_point_state_.get_delta_primal().data(),
                              tmp_primal_.data()),
    primal_size_h_,
    primal_projection<f_t>(primal_step_size.data()),
    stream_view_);
  } else {
    cub::DeviceTransform::Transform(
    cuda::std::make_tuple(current_saddle_point_state_.get_primal_solution().data(),
                          thrust::make_transform_iterator(
                            thrust::make_counting_iterator(0),
                            problem_wrapped_iterator<f_t>(problem_ptr->objective_coefficients.data(),
                                                         primal_size_h_)),
                          current_saddle_point_state_.get_current_AtY().data(),
                          thrust::make_transform_iterator(
                            thrust::make_counting_iterator(0),
                            problem_wrapped_iterator<f_t>(problem_ptr->variable_lower_bounds.data(),
                                                         primal_size_h_)),
                          thrust::make_transform_iterator(
                            thrust::make_counting_iterator(0),
                            problem_wrapped_iterator<f_t>(problem_ptr->variable_upper_bounds.data(),
                                                         primal_size_h_)),
                          thrust::make_transform_iterator(
                            thrust::make_counting_iterator(0),
                            batch_wrapped_iterator<f_t>(primal_step_size.data(),
                                                         primal_size_h_))
                          ),
    thrust::make_zip_iterator(potential_next_primal_solution_.data(),
                              current_saddle_point_state_.get_delta_primal().data(),
                              batch_tmp_primals_.data()),
    primal_size_h_ * (0 + 3)/*@@*/,
    batch_primal_projection<f_t>(),
    stream_view_);
  }
}

template <typename i_t, typename f_t>
void pdhg_solver_t<i_t, f_t>::compute_next_primal_dual_solution(
  rmm::device_uvector<f_t>& primal_step_size,
  i_t iterations_since_last_restart,
  bool last_restart_was_average,
  rmm::device_uvector<f_t>& dual_step_size,
  i_t total_pdlp_iterations)
{
  raft::common::nvtx::range fun_scope("compute_next_primal_solution");
#ifdef PDLP_DEBUG_MODE
  std::cout << "  compute_next_primal_solution:" << std::endl;
#endif

  // proj(x-(tau(c-K^Ty)))
  // K = A, tau = primal_step_size, x = primal_solution, y = dual_solution

  // QP if quadratic program: proj(x-(tau(Q*x + c-K^Ty)))

  // Computation should only take place during very first iteration or after each resart to
  // average (after a restart to average, previous A_t @ y is not valid anymore since it was on
  // current)
  // Indeed, adaptative_step_size has already computed what was next (now current) A_t @ y,
  // so we don't need to recompute it here
  if (total_pdhg_iterations_ == 0 ||
      (iterations_since_last_restart == 0 && last_restart_was_average)) {
#ifdef PDLP_DEBUG_MODE
    std::cout << "    Very first or first iteration since last restart and was average, "
                 "recomputing A_t * Y"
              << std::endl;
#endif

    // Primal and dual steps are captured in a cuda graph since called very often
    //if (!graph_all.is_initialized(total_pdlp_iterations)) {
    //  graph_all.start_capture(total_pdlp_iterations);
      // First compute only A_t @ y, needed later in adaptative step size
      compute_At_y();
      // Compute fused primal gradient with projection
      compute_primal_projection_with_gradient(primal_step_size);
      // Compute next dual solution
      compute_next_dual_solution(dual_step_size);
      //graph_all.end_capture(total_pdlp_iterations);
    //}
    //graph_all.launch(total_pdlp_iterations);
  } else {
#ifdef PDLP_DEBUG_MODE
    std::cout << "    Not computing A_t * Y" << std::endl;
#endif
    // A_t * y was already computed in previous iteration
    //if (!graph_prim_proj_gradient_dual.is_initialized(total_pdlp_iterations)) {
    //  graph_prim_proj_gradient_dual.start_capture(total_pdlp_iterations);
      compute_primal_projection_with_gradient(primal_step_size);
      compute_next_dual_solution(dual_step_size);
    //  graph_prim_proj_gradient_dual.end_capture(total_pdlp_iterations);
    //}
    //graph_prim_proj_gradient_dual.launch(total_pdlp_iterations);
  }
}

template <typename i_t, typename f_t>
void pdhg_solver_t<i_t, f_t>::take_step(rmm::device_uvector<f_t>& primal_step_size,
                                        rmm::device_uvector<f_t>& dual_step_size,
                                        i_t iterations_since_last_restart,
                                        bool last_restart_was_average,
                                        i_t total_pdlp_iterations)
{
#ifdef PDLP_DEBUG_MODE
  std::cout << "Take Step:" << std::endl;
#endif

  compute_next_primal_dual_solution(primal_step_size,
                                    iterations_since_last_restart,
                                    last_restart_was_average,
                                    dual_step_size,
                                    total_pdlp_iterations);
  total_pdhg_iterations_ += 1;
}

template <typename i_t, typename f_t>
void pdhg_solver_t<i_t, f_t>::update_solution(
  cusparse_view_t<i_t, f_t>& current_op_problem_evaluation_cusparse_view_)
{
  raft::common::nvtx::range fun_scope("update_solution");

  // Instead of copying, use a swap (that moves pointers)
  // It's ok because the next will be overwritten next iteration anyways
  // No need to sync, compute_step_sizes has already synced the host

  // Accepted (valid step size) next_Aty will be current Aty next PDHG iteration, saves an SpMV
  std::swap(current_saddle_point_state_.current_AtY_, current_saddle_point_state_.next_AtY_);
  std::swap(current_saddle_point_state_.primal_solution_, potential_next_primal_solution_);
  std::swap(current_saddle_point_state_.dual_solution_, potential_next_dual_solution_);

  // Forced to reinite cusparse views but that's ok, cost is marginal
  // TODO do I need that in batch mode?
  RAFT_CUSPARSE_TRY(
    raft::sparse::detail::cusparsecreatednvec(&cusparse_view_.current_AtY,
                                              current_saddle_point_state_.get_primal_size(),
                                              current_saddle_point_state_.current_AtY_.data()));
  RAFT_CUSPARSE_TRY(
    raft::sparse::detail::cusparsecreatednvec(&cusparse_view_.next_AtY,
                                              current_saddle_point_state_.get_primal_size(),
                                              current_saddle_point_state_.next_AtY_.data()));
  RAFT_CUSPARSE_TRY(
    raft::sparse::detail::cusparsecreatednvec(&cusparse_view_.potential_next_dual_solution,
                                              current_saddle_point_state_.get_dual_size(),
                                              potential_next_dual_solution_.data()));
  RAFT_CUSPARSE_TRY(
    raft::sparse::detail::cusparsecreatednvec(&cusparse_view_.primal_solution,
                                              current_saddle_point_state_.get_primal_size(),
                                              current_saddle_point_state_.primal_solution_.data()));
  RAFT_CUSPARSE_TRY(
    raft::sparse::detail::cusparsecreatednvec(&cusparse_view_.dual_solution,
                                              current_saddle_point_state_.get_dual_size(),
                                              current_saddle_point_state_.dual_solution_.data()));

  if(batch_mode_) {
    RAFT_CUSPARSE_TRY(raft::sparse::detail::cusparsecreatednmat(
      &cusparse_view_.batch_current_AtYs,
      current_saddle_point_state_.get_primal_size(),
      (0 + 3)/*@@*/,
      current_saddle_point_state_.get_primal_size(),
      current_saddle_point_state_.get_current_AtY().data(),
      CUSPARSE_ORDER_COL));
      RAFT_CUSPARSE_TRY(raft::sparse::detail::cusparsecreatednmat(
        &cusparse_view_.batch_next_AtYs,
        current_saddle_point_state_.get_primal_size(),
        (0 + 3)/*@@*/,
        current_saddle_point_state_.get_primal_size(),
        current_saddle_point_state_.get_next_AtY().data(),
        CUSPARSE_ORDER_COL));
    RAFT_CUSPARSE_TRY(raft::sparse::detail::cusparsecreatednmat(
          &cusparse_view_.batch_potential_next_dual_solution,
          current_saddle_point_state_.get_dual_size(),
          (0 + 3)/*@@*/,
          current_saddle_point_state_.get_dual_size(),
          potential_next_dual_solution_.data(),
          CUSPARSE_ORDER_COL));
 }
  RAFT_CUSPARSE_TRY(raft::sparse::detail::cusparsecreatednvec(
    &current_op_problem_evaluation_cusparse_view_.primal_solution,
    current_saddle_point_state_.get_primal_size(),
    current_saddle_point_state_.primal_solution_.data()));
  RAFT_CUSPARSE_TRY(raft::sparse::detail::cusparsecreatednvec(
    &current_op_problem_evaluation_cusparse_view_.dual_solution,
    current_saddle_point_state_.get_dual_size(),
    current_saddle_point_state_.dual_solution_.data()));
}

template <typename i_t, typename f_t>
saddle_point_state_t<i_t, f_t>& pdhg_solver_t<i_t, f_t>::get_saddle_point_state()
{
  return current_saddle_point_state_;
}

template <typename i_t, typename f_t>
cusparse_view_t<i_t, f_t>& pdhg_solver_t<i_t, f_t>::get_cusparse_view()
{
  return cusparse_view_;
}

template <typename i_t, typename f_t>
rmm::device_uvector<f_t>& pdhg_solver_t<i_t, f_t>::get_primal_tmp_resource(bool batch_mode)
{
  if (batch_mode) {
    return batch_tmp_primals_;
  } else {
    return tmp_primal_;
  }
}

template <typename i_t, typename f_t>
rmm::device_uvector<f_t>& pdhg_solver_t<i_t, f_t>::get_dual_tmp_resource()
{
  return tmp_dual_;
}

template <typename i_t, typename f_t>
const rmm::device_uvector<f_t>& pdhg_solver_t<i_t, f_t>::get_potential_next_primal_solution() const
{
  return potential_next_primal_solution_;
}

template <typename i_t, typename f_t>
const rmm::device_uvector<f_t>& pdhg_solver_t<i_t, f_t>::get_potential_next_dual_solution() const
{
  return potential_next_dual_solution_;
}

template <typename i_t, typename f_t>
rmm::device_uvector<f_t>& pdhg_solver_t<i_t, f_t>::get_potential_next_dual_solution()
{
  return potential_next_dual_solution_;
}

template <typename i_t, typename f_t>
i_t pdhg_solver_t<i_t, f_t>::get_total_pdhg_iterations()
{
  return total_pdhg_iterations_;
}

template <typename i_t, typename f_t>
rmm::device_uvector<f_t>& pdhg_solver_t<i_t, f_t>::get_primal_solution()
{
  return current_saddle_point_state_.get_primal_solution();
}

template <typename i_t, typename f_t>
rmm::device_uvector<f_t>& pdhg_solver_t<i_t, f_t>::get_dual_solution()
{
  return current_saddle_point_state_.get_dual_solution();
}

#if MIP_INSTANTIATE_FLOAT
template class pdhg_solver_t<int, float>;
#endif
#if MIP_INSTANTIATE_DOUBLE
template class pdhg_solver_t<int, double>;
#endif

}  // namespace cuopt::linear_programming::detail
