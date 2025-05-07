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

#pragma once

#include <cuopt/linear_programming/pdlp/pdlp_warm_start_data.hpp>

#include <raft/core/device_span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

#include <cuopt/linear_programming/mip/solver_settings.hpp>
#include <cuopt/linear_programming/pdlp/solver_settings.hpp>
#include <cuopt/linear_programming/utilities/internals.hpp>
#include <optional>

namespace cuopt::linear_programming {

template <typename i_t, typename f_t>
class solver_settings_t {
 public:
  solver_settings_t()                                  = default;
  solver_settings_t(const solver_settings_t& settings) = default;

  // PDLP Settings
  void set_optimality_tolerance(f_t eps_optimal);
  void set_absolute_dual_tolerance(f_t absolute_dual_tolerance);
  void set_relative_dual_tolerance(f_t relative_dual_tolerance);
  void set_absolute_primal_tolerance(f_t absolute_primal_tolerance);
  void set_relative_primal_tolerance(f_t relative_primal_tolerance);
  void set_absolute_gap_tolerance(f_t absolute_gap_tolerance);
  void set_relative_gap_tolerance(f_t relative_gap_tolerance);
  void set_infeasibility_detection(bool detect);
  void set_strict_infeasibility(bool strict_infeasibility);
  void set_primal_infeasible_tolerance(f_t primal_infeasible_tolerance);
  void set_dual_infeasible_tolerance(f_t dual_infeasible_tolerance);
  void set_iteration_limit(i_t iteration_limit);
  void set_time_limit(double time_limit);
  void set_pdlp_solver_mode(pdlp_solver_mode_t solver_mode);
  void set_method(method_t method);
  void set_crossover(bool crossover);
  void set_log_file(std::string log_file);
  void set_log_to_console(bool log_to_console);
  void set_initial_pdlp_primal_solution(const f_t* initial_primal_solution,
                                        i_t size,
                                        rmm::cuda_stream_view stream = rmm::cuda_stream_default);
  void set_initial_pdlp_dual_solution(const f_t* initial_dual_solution,
                                      i_t size,
                                      rmm::cuda_stream_view stream = rmm::cuda_stream_default);
  void set_pdlp_warm_start_data(const f_t* current_primal_solution,
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
                                f_t initial_primal_weight_,
                                f_t initial_step_size_,
                                i_t total_pdlp_iterations_,
                                i_t total_pdhg_iterations_,
                                f_t last_candidate_kkt_score_,
                                f_t last_restart_kkt_score_,
                                f_t sum_solution_weight_,
                                i_t iterations_since_last_restart_);

  f_t get_absolute_dual_tolerance() const noexcept;
  f_t get_relative_dual_tolerance() const noexcept;
  f_t get_absolute_primal_tolerance() const noexcept;
  f_t get_relative_primal_tolerance() const noexcept;
  f_t get_absolute_gap_tolerance() const noexcept;
  f_t get_relative_gap_tolerance() const noexcept;
  bool get_infeasibility_detection() const noexcept;
  bool get_strict_infeasibility() const noexcept;
  f_t get_primal_infeasible_tolerance() const noexcept;
  f_t get_dual_infeasible_tolerance() const noexcept;
  i_t get_iteration_limit() const noexcept;
  double get_time_limit() const noexcept;
  pdlp_solver_mode_t get_pdlp_solver_mode() const noexcept;
  method_t get_method() const noexcept;
  bool get_crossover() const noexcept;
  std::string get_log_file() const noexcept;
  bool get_log_to_console() const noexcept;
  const rmm::device_uvector<f_t>& get_initial_pdlp_primal_solution() const;
  const rmm::device_uvector<f_t>& get_initial_pdlp_dual_solution() const;

  // MIP Settings
  void set_absolute_tolerance(f_t absolute_tolerance);
  void set_relative_tolerance(f_t relative_tolerance);
  void set_integrality_tolerance(f_t integrality_tolerance);
  void set_absolute_mip_gap(f_t absolute_mip_gap);
  void set_relative_mip_gap(f_t relative_mip_gap);
  void set_initial_mip_solution(const f_t* initial_solution, i_t size);
  void set_mip_incumbent_solution_callback(
    internals::lp_incumbent_sol_callback_t* callback = nullptr);
  void set_mip_scaling(bool mip_scaling);
  void set_mip_heuristics_only(bool heuristics_only);
  void set_mip_num_cpu_threads(i_t num_cpu_threads);
  bool get_mip_heuristics_only() const noexcept;
  i_t get_mip_num_cpu_threads() const noexcept;

  f_t get_absolute_tolerance() const noexcept;
  f_t get_relative_tolerance() const noexcept;
  f_t get_integrality_tolerance() const noexcept;
  f_t get_absolute_mip_gap() const noexcept;
  f_t get_relative_mip_gap() const noexcept;
  const rmm::device_uvector<f_t>& get_initial_mip_solution() const;
  const pdlp_warm_start_data_view_t<i_t, f_t>& get_pdlp_warm_start_data_view() const noexcept;
  const internals::lp_incumbent_sol_callback_t* get_mip_incumbent_solution_callback() const;

  pdlp_solver_settings_t<i_t, f_t>& get_pdlp_settings();
  mip_solver_settings_t<i_t, f_t>& get_mip_settings();

 private:
  pdlp_solver_settings_t<i_t, f_t> pdlp_settings;
  mip_solver_settings_t<i_t, f_t> mip_settings;
};

}  // namespace cuopt::linear_programming
