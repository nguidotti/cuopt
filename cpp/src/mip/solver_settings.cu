/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <cuopt/linear_programming/mip/solver_settings.hpp>
#include <mip/mip_constants.hpp>
#include <raft/util/cudart_utils.hpp>
#include <utilities/error.hpp>

namespace cuopt::linear_programming {

template <typename i_t, typename f_t>
void mip_solver_settings_t<i_t, f_t>::set_absolute_tolerance(f_t absolute_tolerance)
{
  tolerances.absolute_tolerance = absolute_tolerance;
}

template <typename i_t, typename f_t>
void mip_solver_settings_t<i_t, f_t>::set_relative_tolerance(f_t relative_tolerance)
{
  tolerances.relative_tolerance = relative_tolerance;
}

template <typename i_t, typename f_t>
void mip_solver_settings_t<i_t, f_t>::set_integrality_tolerance(f_t integrality_tolerance)
{
  tolerances.integrality_tolerance = integrality_tolerance;
}

template <typename i_t, typename f_t>
void mip_solver_settings_t<i_t, f_t>::set_absolute_mip_gap(f_t absolute_mip_gap)
{
  tolerances.absolute_mip_gap = absolute_mip_gap;
}

template <typename i_t, typename f_t>
void mip_solver_settings_t<i_t, f_t>::set_relative_mip_gap(f_t relative_mip_gap)
{
  tolerances.relative_mip_gap = relative_mip_gap;
}

template <typename i_t, typename f_t>
void mip_solver_settings_t<i_t, f_t>::set_time_limit(double time_limit) noexcept
{
  time_limit_ = time_limit;
}

template <typename i_t, typename f_t>
void mip_solver_settings_t<i_t, f_t>::set_heuristics_only(bool heuristics_only) noexcept
{
  heuristics_only_ = heuristics_only;
}

template <typename i_t, typename f_t>
void mip_solver_settings_t<i_t, f_t>::set_num_cpu_threads(i_t num_cpu_threads) noexcept
{
  num_cpu_threads_ = num_cpu_threads;
}

template <typename i_t, typename f_t>
void mip_solver_settings_t<i_t, f_t>::set_log_file(std::string log_file) noexcept
{
  log_file_ = log_file;
}

template <typename i_t, typename f_t>
void mip_solver_settings_t<i_t, f_t>::set_log_to_console(bool log_to_console) noexcept
{
  log_to_console_ = log_to_console;
}

template <typename i_t, typename f_t>
void mip_solver_settings_t<i_t, f_t>::set_initial_solution(const f_t* initial_solution,
                                                           i_t size,
                                                           rmm::cuda_stream_view stream)
{
  cuopt_expects(
    initial_solution != nullptr, error_type_t::ValidationError, "initial_solution cannot be null");
  if (!initial_solution_) {
    initial_solution_ = std::make_shared<rmm::device_uvector<f_t>>(size, stream);
  }

  raft::copy(initial_solution_.get()->data(), initial_solution, size, stream);
}

template <typename i_t, typename f_t>
void mip_solver_settings_t<i_t, f_t>::set_incumbent_solution_callback(
  internals::lp_incumbent_sol_callback_t* callback)
{
  incumbent_sol_callback_ = callback;
  if (incumbent_sol_callback_ != nullptr) { incumbent_sol_callback_->template setup<f_t>(); }
}

template <typename i_t, typename f_t>
void mip_solver_settings_t<i_t, f_t>::set_mip_scaling(bool mip_scaling)
{
  mip_scaling_ = mip_scaling;
}

template <typename i_t, typename f_t>
f_t mip_solver_settings_t<i_t, f_t>::get_absolute_tolerance() const noexcept
{
  return tolerances.absolute_tolerance;
}

template <typename i_t, typename f_t>
f_t mip_solver_settings_t<i_t, f_t>::get_relative_tolerance() const noexcept
{
  return tolerances.relative_tolerance;
}

template <typename i_t, typename f_t>
f_t mip_solver_settings_t<i_t, f_t>::get_integrality_tolerance() const noexcept
{
  return tolerances.integrality_tolerance;
}

template <typename i_t, typename f_t>
f_t mip_solver_settings_t<i_t, f_t>::get_absolute_mip_gap() const noexcept
{
  return tolerances.absolute_mip_gap;
}

template <typename i_t, typename f_t>
f_t mip_solver_settings_t<i_t, f_t>::get_relative_mip_gap() const noexcept
{
  return tolerances.relative_mip_gap;
}

template <typename i_t, typename f_t>
typename mip_solver_settings_t<i_t, f_t>::tolerances_t
mip_solver_settings_t<i_t, f_t>::get_tolerances() const noexcept
{
  return tolerances;
}

template <typename i_t, typename f_t>
double mip_solver_settings_t<i_t, f_t>::get_time_limit() const noexcept
{
  return time_limit_;
}

template <typename i_t, typename f_t>
bool mip_solver_settings_t<i_t, f_t>::get_heuristics_only() const noexcept
{
  return heuristics_only_;
}

template <typename i_t, typename f_t>
i_t mip_solver_settings_t<i_t, f_t>::get_num_cpu_threads() const noexcept
{
  return num_cpu_threads_;
}

template <typename i_t, typename f_t>
std::string mip_solver_settings_t<i_t, f_t>::get_log_file() const noexcept
{
  return log_file_;
}

template <typename i_t, typename f_t>
bool mip_solver_settings_t<i_t, f_t>::get_log_to_console() const noexcept
{
  return log_to_console_;
}

template <typename i_t, typename f_t>
rmm::device_uvector<f_t>& mip_solver_settings_t<i_t, f_t>::get_initial_solution() const
{
  if (!initial_solution_) { throw std::runtime_error("Initial solution has not been set"); }
  return *initial_solution_;
}

template <typename i_t, typename f_t>
bool mip_solver_settings_t<i_t, f_t>::has_initial_solution() const
{
  return initial_solution_.get() != nullptr;
}

template <typename i_t, typename f_t>
internals::lp_incumbent_sol_callback_t*
mip_solver_settings_t<i_t, f_t>::get_incumbent_solution_callback() const
{
  return incumbent_sol_callback_;
}

template <typename i_t, typename f_t>
bool mip_solver_settings_t<i_t, f_t>::get_mip_scaling() const noexcept
{
  return mip_scaling_;
}

// Explicit template instantiations for common types
#if MIP_INSTANTIATE_FLOAT
template class mip_solver_settings_t<int, float>;
#endif

#if MIP_INSTANTIATE_DOUBLE
template class mip_solver_settings_t<int, double>;
#endif

}  // namespace cuopt::linear_programming
