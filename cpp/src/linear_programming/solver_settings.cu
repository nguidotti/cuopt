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

#include <cuopt/linear_programming/pdlp/pdlp_warm_start_data.hpp>
#include <cuopt/linear_programming/pdlp/solver_settings.hpp>
#include <cuopt/logger.hpp>
#include <mip/mip_constants.hpp>
#include <mps_parser/utilities/span.hpp>
#include <raft/util/cudart_utils.hpp>
#include <rmm/exec_policy.hpp>
#include <utilities/error.hpp>

namespace cuopt::linear_programming {

template <typename i_t, typename f_t>
pdlp_solver_settings_t<i_t, f_t>::pdlp_solver_settings_t(const pdlp_solver_settings_t& other,
                                                         rmm::cuda_stream_view stream_view)
  : tolerances(other.tolerances),
    detect_infeasibility_(other.detect_infeasibility_),
    strict_infeasibility_(other.strict_infeasibility_),
    iteration_limit_(other.iteration_limit_),
    time_limit_(other.time_limit_),
    solver_mode_(other.solver_mode_),
    log_file_(other.log_file_),
    per_constraint_residual_(other.per_constraint_residual_),
    crossover_(other.crossover_),
    save_best_primal_so_far_(other.save_best_primal_so_far_),
    first_primal_feasible_(other.first_primal_feasible_),
    pdlp_warm_start_data_(other.pdlp_warm_start_data_, stream_view),
    concurrent_halt_(other.concurrent_halt_)
{
}

template <typename i_t, typename f_t>
void pdlp_solver_settings_t<i_t, f_t>::set_optimality_tolerance(f_t eps_optimal)
{
  set_absolute_dual_tolerance(eps_optimal);
  set_relative_dual_tolerance(eps_optimal);
  set_absolute_primal_tolerance(eps_optimal);
  set_relative_primal_tolerance(eps_optimal);
  set_absolute_gap_tolerance(eps_optimal);
  set_relative_gap_tolerance(eps_optimal);
}

template <typename i_t, typename f_t>
void pdlp_solver_settings_t<i_t, f_t>::set_absolute_dual_tolerance(f_t absolute_dual_tolerance)
{
  if (absolute_dual_tolerance < minimal_absolute_tolerance) {
    CUOPT_LOG_INFO(
      "Warning: settings the absolute dual tolerance to the minimum allowed value %.0e",
      minimal_absolute_tolerance);
    tolerances.absolute_dual_tolerance = minimal_absolute_tolerance;
  } else {
    tolerances.absolute_dual_tolerance = absolute_dual_tolerance;
  }
}
template <typename i_t, typename f_t>
void pdlp_solver_settings_t<i_t, f_t>::set_relative_dual_tolerance(f_t relative_dual_tolerance)
{
  tolerances.relative_dual_tolerance = relative_dual_tolerance;
}
template <typename i_t, typename f_t>
void pdlp_solver_settings_t<i_t, f_t>::set_absolute_primal_tolerance(f_t absolute_primal_tolerance)
{
  if (absolute_primal_tolerance < minimal_absolute_tolerance) {
    CUOPT_LOG_INFO(
      "Warning: settings the absolute primal tolerance to the minimum allowed value %.0e",
      minimal_absolute_tolerance);
    tolerances.absolute_primal_tolerance = minimal_absolute_tolerance;
  } else {
    tolerances.absolute_primal_tolerance = absolute_primal_tolerance;
  }
}
template <typename i_t, typename f_t>
void pdlp_solver_settings_t<i_t, f_t>::set_relative_primal_tolerance(f_t relative_primal_tolerance)
{
  tolerances.relative_primal_tolerance = relative_primal_tolerance;
}
template <typename i_t, typename f_t>
void pdlp_solver_settings_t<i_t, f_t>::set_absolute_gap_tolerance(f_t absolute_gap_tolerance)
{
  if (absolute_gap_tolerance < minimal_absolute_tolerance) {
    CUOPT_LOG_INFO("Warning: settings the absolute gap tolerance to the minimum allowed value %.0e",
                   minimal_absolute_tolerance);
    tolerances.absolute_gap_tolerance = minimal_absolute_tolerance;
  } else {
    tolerances.absolute_gap_tolerance = absolute_gap_tolerance;
  }
}
template <typename i_t, typename f_t>
void pdlp_solver_settings_t<i_t, f_t>::set_relative_gap_tolerance(f_t relative_gap_tolerance)
{
  tolerances.relative_gap_tolerance = relative_gap_tolerance;
}

template <typename i_t, typename f_t>
void pdlp_solver_settings_t<i_t, f_t>::set_infeasibility_detection(bool detect)
{
  detect_infeasibility_ = detect;
}

template <typename i_t, typename f_t>
void pdlp_solver_settings_t<i_t, f_t>::set_strict_infeasibility(bool strict_infeasibility)
{
  strict_infeasibility_ = strict_infeasibility;
}

template <typename i_t, typename f_t>
void pdlp_solver_settings_t<i_t, f_t>::set_primal_infeasible_tolerance(
  f_t primal_infeasible_tolerance)
{
  tolerances.primal_infeasible_tolerance = primal_infeasible_tolerance;
}

template <typename i_t, typename f_t>
void pdlp_solver_settings_t<i_t, f_t>::set_dual_infeasible_tolerance(f_t dual_infeasible_tolerance)
{
  tolerances.dual_infeasible_tolerance = dual_infeasible_tolerance;
}

template <typename i_t, typename f_t>
void pdlp_solver_settings_t<i_t, f_t>::set_iteration_limit(i_t iteration_limit)
{
  iteration_limit_ = iteration_limit;
}

template <typename i_t, typename f_t>
void pdlp_solver_settings_t<i_t, f_t>::set_time_limit(double time_limit)
{
  time_limit_ = time_limit;
}

template <typename i_t, typename f_t>
void pdlp_solver_settings_t<i_t, f_t>::set_pdlp_solver_mode(pdlp_solver_mode_t solver_mode)
{
  solver_mode_ = solver_mode;
}

template <typename i_t, typename f_t>
void pdlp_solver_settings_t<i_t, f_t>::set_method(method_t method)
{
  method_ = method;
}

template <typename i_t, typename f_t>
void pdlp_solver_settings_t<i_t, f_t>::set_log_file(std::string log_file)
{
  log_file_ = log_file;
}

template <typename i_t, typename f_t>
void pdlp_solver_settings_t<i_t, f_t>::set_log_to_console(bool log_to_console)
{
  log_to_console_ = log_to_console;
}

template <typename i_t, typename f_t>
void pdlp_solver_settings_t<i_t, f_t>::set_per_constraint_residual(bool per_constraint_residual)
{
  per_constraint_residual_ = per_constraint_residual;
}

template <typename i_t, typename f_t>
void pdlp_solver_settings_t<i_t, f_t>::set_crossover(bool crossover)
{
  crossover_ = crossover;
}

template <typename i_t, typename f_t>
void pdlp_solver_settings_t<i_t, f_t>::set_save_best_primal_so_far(bool save_best_primal_so_far)
{
  save_best_primal_so_far_ = save_best_primal_so_far;
}

template <typename i_t, typename f_t>
void pdlp_solver_settings_t<i_t, f_t>::set_first_primal_feasible_encountered(
  bool first_primal_feasible)
{
  first_primal_feasible_ = first_primal_feasible;
}

template <typename i_t, typename f_t>
void pdlp_solver_settings_t<i_t, f_t>::set_initial_primal_solution(
  const f_t* initial_primal_solution, i_t size, rmm::cuda_stream_view stream)
{
  cuopt_expects(initial_primal_solution != nullptr,
                error_type_t::ValidationError,
                "initial_primal_solution cannot be null");

  initial_primal_solution_ = std::make_shared<rmm::device_uvector<f_t>>(size, stream);
  raft::copy(initial_primal_solution_.get()->data(), initial_primal_solution, size, stream);
}

template <typename i_t, typename f_t>
void pdlp_solver_settings_t<i_t, f_t>::set_initial_dual_solution(const f_t* initial_dual_solution,
                                                                 i_t size,
                                                                 rmm::cuda_stream_view stream)
{
  cuopt_expects(initial_dual_solution != nullptr,
                error_type_t::ValidationError,
                "initial_dual_solution cannot be null");

  initial_dual_solution_ = std::make_shared<rmm::device_uvector<f_t>>(size, stream);
  raft::copy(initial_dual_solution_.get()->data(), initial_dual_solution, size, stream);
}

template <typename i_t, typename f_t>
void pdlp_solver_settings_t<i_t, f_t>::set_pdlp_warm_start_data(
  pdlp_warm_start_data_t<i_t, f_t>& pdlp_warm_start_data_view,
  const rmm::device_uvector<i_t>& var_mapping,
  const rmm::device_uvector<i_t>& constraint_mapping)
{
  pdlp_warm_start_data_ = std::move(pdlp_warm_start_data_view);

  // A var_mapping was given
  if (var_mapping.size() != 0) {
    // If less variables, scatter using the passed argument and reduce the size of all primal
    // related vectors
    if (var_mapping.size() <
        pdlp_warm_start_data_.last_restart_duality_gap_primal_solution_.size()) {
      thrust::scatter(rmm::exec_policy(var_mapping.stream()),
                      pdlp_warm_start_data_.current_primal_solution_.begin(),
                      pdlp_warm_start_data_.current_primal_solution_.end(),
                      var_mapping.begin(),
                      pdlp_warm_start_data_.current_primal_solution_.begin());
      thrust::scatter(rmm::exec_policy(var_mapping.stream()),
                      pdlp_warm_start_data_.initial_primal_average_.begin(),
                      pdlp_warm_start_data_.initial_primal_average_.end(),
                      var_mapping.begin(),
                      pdlp_warm_start_data_.initial_primal_average_.begin());
      thrust::scatter(rmm::exec_policy(var_mapping.stream()),
                      pdlp_warm_start_data_.current_ATY_.begin(),
                      pdlp_warm_start_data_.current_ATY_.end(),
                      var_mapping.begin(),
                      pdlp_warm_start_data_.current_ATY_.begin());
      thrust::scatter(rmm::exec_policy(var_mapping.stream()),
                      pdlp_warm_start_data_.sum_primal_solutions_.begin(),
                      pdlp_warm_start_data_.sum_primal_solutions_.end(),
                      var_mapping.begin(),
                      pdlp_warm_start_data_.sum_primal_solutions_.begin());
      thrust::scatter(rmm::exec_policy(var_mapping.stream()),
                      pdlp_warm_start_data_.last_restart_duality_gap_primal_solution_.begin(),
                      pdlp_warm_start_data_.last_restart_duality_gap_primal_solution_.end(),
                      var_mapping.begin(),
                      pdlp_warm_start_data_.last_restart_duality_gap_primal_solution_.begin());

      pdlp_warm_start_data_.current_primal_solution_.resize(var_mapping.size(),
                                                            var_mapping.stream());
      pdlp_warm_start_data_.initial_primal_average_.resize(var_mapping.size(),
                                                           var_mapping.stream());
      pdlp_warm_start_data_.current_ATY_.resize(var_mapping.size(), var_mapping.stream());
      pdlp_warm_start_data_.sum_primal_solutions_.resize(var_mapping.size(), var_mapping.stream());
      pdlp_warm_start_data_.last_restart_duality_gap_primal_solution_.resize(var_mapping.size(),
                                                                             var_mapping.stream());
    } else if (var_mapping.size() >
               pdlp_warm_start_data_.last_restart_duality_gap_primal_solution_.size()) {
      const auto previous_size =
        pdlp_warm_start_data_.last_restart_duality_gap_primal_solution_.size();

      // If more variables just pad with 0s
      pdlp_warm_start_data_.current_primal_solution_.resize(var_mapping.size(),
                                                            var_mapping.stream());
      pdlp_warm_start_data_.initial_primal_average_.resize(var_mapping.size(),
                                                           var_mapping.stream());
      pdlp_warm_start_data_.current_ATY_.resize(var_mapping.size(), var_mapping.stream());
      pdlp_warm_start_data_.sum_primal_solutions_.resize(var_mapping.size(), var_mapping.stream());
      pdlp_warm_start_data_.last_restart_duality_gap_primal_solution_.resize(var_mapping.size(),
                                                                             var_mapping.stream());

      thrust::fill(rmm::exec_policy(var_mapping.stream()),
                   pdlp_warm_start_data_.current_primal_solution_.begin() + previous_size,
                   pdlp_warm_start_data_.current_primal_solution_.end(),
                   f_t(0));
      thrust::fill(rmm::exec_policy(var_mapping.stream()),
                   pdlp_warm_start_data_.initial_primal_average_.begin() + previous_size,
                   pdlp_warm_start_data_.initial_primal_average_.end(),
                   f_t(0));
      thrust::fill(rmm::exec_policy(var_mapping.stream()),
                   pdlp_warm_start_data_.current_ATY_.begin() + previous_size,
                   pdlp_warm_start_data_.current_ATY_.end(),
                   f_t(0));
      thrust::fill(rmm::exec_policy(var_mapping.stream()),
                   pdlp_warm_start_data_.sum_primal_solutions_.begin() + previous_size,
                   pdlp_warm_start_data_.sum_primal_solutions_.end(),
                   f_t(0));
      thrust::fill(
        rmm::exec_policy(var_mapping.stream()),
        pdlp_warm_start_data_.last_restart_duality_gap_primal_solution_.begin() + previous_size,
        pdlp_warm_start_data_.last_restart_duality_gap_primal_solution_.end(),
        f_t(0));
    }
  }

  // A constraint_mapping was given
  if (constraint_mapping.size() != 0) {
    // If less variables, scatter using the passed argument and reduce the size of all dual related
    // vectors
    if (constraint_mapping.size() <
        pdlp_warm_start_data_.last_restart_duality_gap_dual_solution_.size()) {
      thrust::scatter(rmm::exec_policy(constraint_mapping.stream()),
                      pdlp_warm_start_data_.current_dual_solution_.begin(),
                      pdlp_warm_start_data_.current_dual_solution_.end(),
                      constraint_mapping.begin(),
                      pdlp_warm_start_data_.current_dual_solution_.begin());
      thrust::scatter(rmm::exec_policy(constraint_mapping.stream()),
                      pdlp_warm_start_data_.initial_dual_average_.begin(),
                      pdlp_warm_start_data_.initial_dual_average_.end(),
                      constraint_mapping.begin(),
                      pdlp_warm_start_data_.initial_dual_average_.begin());
      thrust::scatter(rmm::exec_policy(constraint_mapping.stream()),
                      pdlp_warm_start_data_.sum_dual_solutions_.begin(),
                      pdlp_warm_start_data_.sum_dual_solutions_.end(),
                      constraint_mapping.begin(),
                      pdlp_warm_start_data_.sum_dual_solutions_.begin());
      thrust::scatter(rmm::exec_policy(constraint_mapping.stream()),
                      pdlp_warm_start_data_.last_restart_duality_gap_dual_solution_.begin(),
                      pdlp_warm_start_data_.last_restart_duality_gap_dual_solution_.end(),
                      constraint_mapping.begin(),
                      pdlp_warm_start_data_.last_restart_duality_gap_dual_solution_.begin());

      pdlp_warm_start_data_.current_dual_solution_.resize(constraint_mapping.size(),
                                                          constraint_mapping.stream());
      pdlp_warm_start_data_.initial_dual_average_.resize(constraint_mapping.size(),
                                                         constraint_mapping.stream());
      pdlp_warm_start_data_.sum_dual_solutions_.resize(constraint_mapping.size(),
                                                       constraint_mapping.stream());
      pdlp_warm_start_data_.last_restart_duality_gap_dual_solution_.resize(
        constraint_mapping.size(), constraint_mapping.stream());
    } else if (constraint_mapping.size() >
               pdlp_warm_start_data_.last_restart_duality_gap_dual_solution_.size()) {
      const auto previous_size =
        pdlp_warm_start_data_.last_restart_duality_gap_dual_solution_.size();

      // If more variables just pad with 0s
      pdlp_warm_start_data_.current_dual_solution_.resize(constraint_mapping.size(),
                                                          constraint_mapping.stream());
      pdlp_warm_start_data_.initial_dual_average_.resize(constraint_mapping.size(),
                                                         constraint_mapping.stream());
      pdlp_warm_start_data_.sum_dual_solutions_.resize(constraint_mapping.size(),
                                                       constraint_mapping.stream());
      pdlp_warm_start_data_.last_restart_duality_gap_dual_solution_.resize(
        constraint_mapping.size(), constraint_mapping.stream());

      thrust::fill(rmm::exec_policy(constraint_mapping.stream()),
                   pdlp_warm_start_data_.current_dual_solution_.begin() + previous_size,
                   pdlp_warm_start_data_.current_dual_solution_.end(),
                   f_t(0));
      thrust::fill(rmm::exec_policy(constraint_mapping.stream()),
                   pdlp_warm_start_data_.initial_dual_average_.begin() + previous_size,
                   pdlp_warm_start_data_.initial_dual_average_.end(),
                   f_t(0));
      thrust::fill(rmm::exec_policy(constraint_mapping.stream()),
                   pdlp_warm_start_data_.sum_dual_solutions_.begin() + previous_size,
                   pdlp_warm_start_data_.sum_dual_solutions_.end(),
                   f_t(0));
      thrust::fill(
        rmm::exec_policy(constraint_mapping.stream()),
        pdlp_warm_start_data_.last_restart_duality_gap_dual_solution_.begin() + previous_size,
        pdlp_warm_start_data_.last_restart_duality_gap_dual_solution_.end(),
        f_t(0));
    }
  }
}

template <typename i_t, typename f_t>
void pdlp_solver_settings_t<i_t, f_t>::set_pdlp_warm_start_data(
  const f_t* current_primal_solution,
  const f_t* current_dual_solution,
  const f_t* initial_primal_average,
  const f_t* initial_dual_average,
  const f_t* current_ATY,
  const f_t* sum_primal_solutions,
  const f_t* sum_dual_solutions,
  const f_t* last_restart_duality_gap_primal_solution,
  const f_t* last_restart_duality_gap_dual_solution,
  i_t primal_size,
  i_t dual_size,
  f_t initial_primal_weight,
  f_t initial_step_size,
  i_t total_pdlp_iterations,
  i_t total_pdhg_iterations,
  f_t last_candidate_kkt_score,
  f_t last_restart_kkt_score,
  f_t sum_solution_weight,
  i_t iterations_since_last_restart)
{
  cuopt_expects(current_primal_solution != nullptr,
                error_type_t::ValidationError,
                "current_primal_solution cannot be null");
  cuopt_expects(current_dual_solution != nullptr,
                error_type_t::ValidationError,
                "current_dual_solution cannot be null");
  cuopt_expects(initial_primal_average != nullptr,
                error_type_t::ValidationError,
                "initial_primal_average cannot be null");
  cuopt_expects(initial_dual_average != nullptr,
                error_type_t::ValidationError,
                "initial_dual_average cannot be null");
  cuopt_expects(
    current_ATY != nullptr, error_type_t::ValidationError, "current_ATY cannot be null");
  cuopt_expects(sum_primal_solutions != nullptr,
                error_type_t::ValidationError,
                "sum_primal_solutions cannot be null");
  cuopt_expects(sum_dual_solutions != nullptr,
                error_type_t::ValidationError,
                "sum_dual_solutions cannot be null");
  cuopt_expects(last_restart_duality_gap_primal_solution != nullptr,
                error_type_t::ValidationError,
                "last_restart_duality_gap_primal_solution cannot be null");
  cuopt_expects(last_restart_duality_gap_dual_solution != nullptr,
                error_type_t::ValidationError,
                "last_restart_duality_gap_dual_solution cannot be null");

  pdlp_warm_start_data_view_.current_primal_solution_ =
    cuopt::mps_parser::span<f_t const>(current_primal_solution, primal_size);
  pdlp_warm_start_data_view_.current_dual_solution_ =
    cuopt::mps_parser::span<f_t const>(current_dual_solution, dual_size);
  pdlp_warm_start_data_view_.initial_primal_average_ =
    cuopt::mps_parser::span<f_t const>(initial_primal_average, primal_size);
  pdlp_warm_start_data_view_.initial_dual_average_ =
    cuopt::mps_parser::span<f_t const>(initial_dual_average, dual_size);
  pdlp_warm_start_data_view_.current_ATY_ =
    cuopt::mps_parser::span<f_t const>(current_ATY, primal_size);
  pdlp_warm_start_data_view_.sum_primal_solutions_ =
    cuopt::mps_parser::span<f_t const>(sum_primal_solutions, primal_size);
  pdlp_warm_start_data_view_.sum_dual_solutions_ =
    cuopt::mps_parser::span<f_t const>(sum_dual_solutions, dual_size);
  pdlp_warm_start_data_view_.last_restart_duality_gap_primal_solution_ =
    cuopt::mps_parser::span<f_t const>(last_restart_duality_gap_primal_solution, primal_size);
  pdlp_warm_start_data_view_.last_restart_duality_gap_dual_solution_ =
    cuopt::mps_parser::span<f_t const>(last_restart_duality_gap_dual_solution, dual_size);
  pdlp_warm_start_data_view_.initial_primal_weight_         = initial_primal_weight;
  pdlp_warm_start_data_view_.initial_step_size_             = initial_step_size;
  pdlp_warm_start_data_view_.total_pdlp_iterations_         = total_pdlp_iterations;
  pdlp_warm_start_data_view_.total_pdhg_iterations_         = total_pdhg_iterations;
  pdlp_warm_start_data_view_.last_candidate_kkt_score_      = last_candidate_kkt_score;
  pdlp_warm_start_data_view_.last_restart_kkt_score_        = last_restart_kkt_score;
  pdlp_warm_start_data_view_.sum_solution_weight_           = sum_solution_weight;
  pdlp_warm_start_data_view_.iterations_since_last_restart_ = iterations_since_last_restart;
}

template <typename i_t, typename f_t>
void pdlp_solver_settings_t<i_t, f_t>::set_concurrent_halt(
  std::atomic<i_t>* concurrent_halt) noexcept
{
  concurrent_halt_ = concurrent_halt;
}

template <typename i_t, typename f_t>
f_t pdlp_solver_settings_t<i_t, f_t>::get_absolute_dual_tolerance() const noexcept
{
  return tolerances.absolute_dual_tolerance;
}
template <typename i_t, typename f_t>
f_t pdlp_solver_settings_t<i_t, f_t>::get_relative_dual_tolerance() const noexcept
{
  return tolerances.relative_dual_tolerance;
}
template <typename i_t, typename f_t>
f_t pdlp_solver_settings_t<i_t, f_t>::get_absolute_primal_tolerance() const noexcept
{
  return tolerances.absolute_primal_tolerance;
}
template <typename i_t, typename f_t>
f_t pdlp_solver_settings_t<i_t, f_t>::get_relative_primal_tolerance() const noexcept
{
  return tolerances.relative_primal_tolerance;
}
template <typename i_t, typename f_t>
f_t pdlp_solver_settings_t<i_t, f_t>::get_absolute_gap_tolerance() const noexcept
{
  return tolerances.absolute_gap_tolerance;
}
template <typename i_t, typename f_t>
f_t pdlp_solver_settings_t<i_t, f_t>::get_relative_gap_tolerance() const noexcept
{
  return tolerances.relative_gap_tolerance;
}

template <typename i_t, typename f_t>
bool pdlp_solver_settings_t<i_t, f_t>::get_infeasibility_detection() const noexcept
{
  return detect_infeasibility_;
}

template <typename i_t, typename f_t>
bool pdlp_solver_settings_t<i_t, f_t>::get_strict_infeasibility() const noexcept
{
  return strict_infeasibility_;
}

template <typename i_t, typename f_t>
f_t pdlp_solver_settings_t<i_t, f_t>::get_primal_infeasible_tolerance() const noexcept
{
  return tolerances.primal_infeasible_tolerance;
}

template <typename i_t, typename f_t>
f_t pdlp_solver_settings_t<i_t, f_t>::get_dual_infeasible_tolerance() const noexcept
{
  return tolerances.dual_infeasible_tolerance;
}

template <typename i_t, typename f_t>
typename pdlp_solver_settings_t<i_t, f_t>::tolerances_t
pdlp_solver_settings_t<i_t, f_t>::get_tolerances() const noexcept
{
  return tolerances;
}

template <typename i_t, typename f_t>
i_t pdlp_solver_settings_t<i_t, f_t>::get_iteration_limit() const noexcept
{
  return iteration_limit_;
}

template <typename i_t, typename f_t>
std::optional<double> pdlp_solver_settings_t<i_t, f_t>::get_time_limit() const noexcept
{
  return time_limit_;
}

template <typename i_t, typename f_t>
pdlp_solver_mode_t pdlp_solver_settings_t<i_t, f_t>::get_pdlp_solver_mode() const noexcept
{
  return solver_mode_;
}

template <typename i_t, typename f_t>
method_t pdlp_solver_settings_t<i_t, f_t>::get_method() const noexcept
{
  return method_;
}

template <typename i_t, typename f_t>
std::string pdlp_solver_settings_t<i_t, f_t>::get_log_file() const noexcept
{
  return log_file_;
}

template <typename i_t, typename f_t>
bool pdlp_solver_settings_t<i_t, f_t>::get_log_to_console() const noexcept
{
  return log_to_console_;
}

template <typename i_t, typename f_t>
bool pdlp_solver_settings_t<i_t, f_t>::get_crossover() const noexcept
{
  return crossover_;
}

template <typename i_t, typename f_t>
bool pdlp_solver_settings_t<i_t, f_t>::get_per_constraint_residual() const noexcept
{
  return per_constraint_residual_;
}

template <typename i_t, typename f_t>
bool pdlp_solver_settings_t<i_t, f_t>::get_save_best_primal_so_far() const noexcept
{
  return save_best_primal_so_far_;
}

template <typename i_t, typename f_t>
bool pdlp_solver_settings_t<i_t, f_t>::get_first_primal_feasible_encountered() const noexcept
{
  return first_primal_feasible_;
}

template <typename i_t, typename f_t>
const rmm::device_uvector<f_t>& pdlp_solver_settings_t<i_t, f_t>::get_initial_primal_solution()
  const
{
  cuopt_expects(initial_primal_solution_.get() != nullptr,
                error_type_t::ValidationError,
                "Initial primal solution was not set, but accessed!");
  return *initial_primal_solution_.get();
}

template <typename i_t, typename f_t>
const rmm::device_uvector<f_t>& pdlp_solver_settings_t<i_t, f_t>::get_initial_dual_solution() const
{
  cuopt_expects(initial_dual_solution_.get() != nullptr,
                error_type_t::ValidationError,
                "Initial dual solution was not set, but accessed!");
  return *initial_dual_solution_.get();
}

template <typename i_t, typename f_t>
bool pdlp_solver_settings_t<i_t, f_t>::has_initial_primal_solution() const
{
  return initial_primal_solution_.get() != nullptr;
}

template <typename i_t, typename f_t>
bool pdlp_solver_settings_t<i_t, f_t>::has_initial_dual_solution() const
{
  return initial_dual_solution_.get() != nullptr;
}

template <typename i_t, typename f_t>
const pdlp_warm_start_data_t<i_t, f_t>& pdlp_solver_settings_t<i_t, f_t>::get_pdlp_warm_start_data()
  const noexcept
{
  return pdlp_warm_start_data_;
}

template <typename i_t, typename f_t>
pdlp_warm_start_data_t<i_t, f_t>& pdlp_solver_settings_t<i_t, f_t>::get_pdlp_warm_start_data()
{
  return pdlp_warm_start_data_;
}

template <typename i_t, typename f_t>
const pdlp_warm_start_data_view_t<i_t, f_t>&
pdlp_solver_settings_t<i_t, f_t>::get_pdlp_warm_start_data_view() const noexcept
{
  return pdlp_warm_start_data_view_;
}

template <typename i_t, typename f_t>
std::atomic<i_t>* pdlp_solver_settings_t<i_t, f_t>::get_concurrent_halt() const noexcept
{
  return concurrent_halt_;
}

#if MIP_INSTANTIATE_FLOAT
template class pdlp_solver_settings_t<int, float>;
#endif

#if MIP_INSTANTIATE_DOUBLE
template class pdlp_solver_settings_t<int, double>;
#endif

}  // namespace cuopt::linear_programming
