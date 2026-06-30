/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <branch_and_bound/worker.hpp>
#include <branch_and_bound/worker_pool.hpp>

namespace cuopt::mathematical_optimization::mip {

template <typename i_t, typename f_t>
class branch_and_bound_t;

struct submip_stats_t {
  omp_atomic_t<int> total_success             = 0;
  omp_atomic_t<double> success_fixrate_sum    = 0;
  omp_atomic_t<int> total_infeasible          = 0;
  omp_atomic_t<double> infeasible_fixrate_sum = 0;
  omp_atomic_t<int> total_calls               = 0;

  void save_success(double fixrate)
  {
    ++total_success;
    success_fixrate_sum += fixrate;
  }

  void save_infeasible(double fixrate)
  {
    ++total_infeasible;
    infeasible_fixrate_sum += fixrate;
  }

  double average_infeasible_fixrate() const { return infeasible_fixrate_sum / total_infeasible; }
  double average_success_fixrate() const { return success_fixrate_sum / total_success; }
};

inline double submip_get_max_fixrate(const submip_stats_t& stats,
                                     const simplex::submip_settings_t& submip_settings,
                                     pcgenerator_t& rng)
{
  // Adaptive fix rate based on previous successes and failures.
  double low  = submip_settings.base_target_fixrate;
  double high = submip_settings.base_target_fixrate;

  if (stats.total_infeasible > 0) {
    double infeasible_avg_fixrate = stats.average_infeasible_fixrate();
    high                          = 0.9 * infeasible_avg_fixrate;
    low                           = std::min(low, high);
  }

  if (stats.total_success > 0) {
    double success_avg_fixrate = stats.average_success_fixrate();
    low                        = std::min(low, 0.9 * success_avg_fixrate);
    high                       = std::max(high, 1.1 * success_avg_fixrate);
  }

  double fixrate = high > low ? rng.uniform(low, high) : low;
  return fixrate;
}

template <typename i_t, typename f_t>
class submip_worker_t : public branch_and_bound_worker_t<i_t, f_t> {
 public:
  using Base = branch_and_bound_worker_t<i_t, f_t>;

  submip_worker_t(i_t worker_id,
                  const simplex::lp_problem_t<i_t, f_t>& original_lp,
                  const csr_matrix_t<i_t, f_t>& Arow,
                  const std::vector<simplex::variable_type_t>& var_type,
                  const simplex::simplex_solver_settings_t<i_t, f_t>& settings,
                  uint64_t rng_offset = 0)
    : Base(worker_id, original_lp, Arow, var_type, settings, rng_offset)
  {
    this->search_strategy = SUBMIP;
  }

  // Set this node inactive
  void set_inactive()
  {
    if (!this->is_active.load()) { return; }
    this->is_active = false;
  }

  f_t get_lower_bound() { return this->lower_bound; }
};

template <typename i_t, typename f_t>
using submip_worker_pool_t = worker_pool_t<submip_worker_t<i_t, f_t>>;

}  // namespace cuopt::mathematical_optimization::mip
