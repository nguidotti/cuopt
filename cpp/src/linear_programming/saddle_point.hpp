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

#pragma once

#include <raft/core/handle.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

namespace cuopt::linear_programming::detail {
/**
 * @brief Structure to hold the current solution to the saddle point problem
 *
 * @tparam f_t  Data type of the variables and their weights in the equations
 * @tparam i_t  Data type of indexes
 *
 */
template <typename i_t, typename f_t>
class saddle_point_state_t {
 public:
  static_assert(std::is_floating_point<f_t>::value,
                "'saddle_point_state_t' accepts only floating point types");
  /**
   * @brief A device-side view of the `saddle_point_state_t` structure with the RAII stuffs
   *        stripped out, to make it easy to work inside kernels
   *
   * @note It is assumed that the pointers are NOT owned by this class, but rather
   *       by the encompassing `saddle_point_state_t` class via RAII abstractions like
   *       `rmm::device_uvector`
   */
  struct view_t {
    /** size of primal problem */
    const i_t primal_size;
    /** size of dual problem */
    const i_t dual_size;

    f_t* primal_solution;
    f_t* dual_solution;
    f_t* delta_primal;
    f_t* delta_dual;
    f_t* primal_gradient;
    f_t* dual_gradient;
  };  // struct view_t

  /**
   * @brief Construct a new saddle point state object.
   * Initializes all needed variables but without any content
   *
   * @param handle_ptr Pointer to library handle (RAFT) containing hardware resources
   * information. A default handle is valid.
   * @param primal_size The size of the primal problem
   * @param dual_size The size of the dual problem
   *
   * @throws cuopt::logic_error if the problem sizes are not larger than 0.
   */
  saddle_point_state_t(raft::handle_t const* handle_ptr, i_t primal_size, i_t dual_size, bool batch_mode);

  /**
   * @brief Copies the values of the solutions in another saddle_point_state_t
   *
   * @tparam i_t
   * @tparam f_t
   * @param other saddle_point_state_t object from which the solution should be copied
   * @param stream cuda stream used for the copying
   *
   * @pre Primal and dual solutions must be of the same size respectively
   *
   * @throws cuopt::logic_error if the solutions are not of the same size
   */
  void copy(saddle_point_state_t<i_t, f_t>& other, rmm::cuda_stream_view stream);

  i_t get_primal_size() const;
  i_t get_dual_size() const;
  rmm::device_uvector<f_t>& get_primal_solution();
  rmm::device_uvector<f_t>& get_dual_solution();
  rmm::device_uvector<f_t>& get_delta_primal();
  rmm::device_uvector<f_t>& get_delta_dual();
  rmm::device_uvector<f_t>& get_primal_gradient();
  rmm::device_uvector<f_t>& get_dual_gradient();
  rmm::device_uvector<f_t>& get_current_AtY();
  rmm::device_uvector<f_t>& get_next_AtY();

  /**
   * @brief Gets the device-side view (with raw pointers), for ease of access
   *        inside cuda kernels
   */

  const i_t primal_size_;
  const i_t dual_size_;

  rmm::device_uvector<f_t> primal_solution_;
  rmm::device_uvector<f_t> dual_solution_;
  rmm::device_uvector<f_t> primal_gradient_;
  rmm::device_uvector<f_t> dual_gradient_;
  rmm::device_uvector<f_t> delta_primal_;
  rmm::device_uvector<f_t> delta_dual_;
  rmm::device_uvector<f_t> current_AtY_;
  rmm::device_uvector<f_t> next_AtY_;

  // TODO comment : eventually should be the same vectors as above but bigger
  rmm::device_uvector<f_t> batch_current_AtYs_;
  rmm::device_uvector<f_t> batch_dual_gradients_;
  rmm::device_uvector<f_t> batch_next_AtYs_;

  bool batch_mode_;
};

}  // namespace cuopt::linear_programming::detail
