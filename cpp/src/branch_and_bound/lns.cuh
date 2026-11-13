/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <branch_and_bound/diving_heuristics.hpp>

namespace cuopt::mathematical_optimization::mip {

enum submip_status_t { SUCCESS = 0, INFEASIBLE = 1, TRUNCATED = 2, EMPTY = 3, UNSET };

template <typename i_t, typename f_t>
struct submip_manager_t {
  omp_mutex_t mutex_;
  std::vector<i_t> status_count_;
  std::vector<f_t> fixrate_sum_per_status_;

  f_t upper_guide_;
  f_t lower_guide_;
  f_t peak_;

  omp_atomic_t<i_t> num_calls_;

  std::vector<branch_and_bound_t<i_t, f_t>*> child_bnb_;
  omp_mutex_t child_bnb_mutex_;

  submip_manager_t(const simplex::simplex_solver_settings_t<i_t, f_t>& settings)
    : status_count_(4, 0),
      fixrate_sum_per_status_(4, 0),
      upper_guide_(settings.mip_hyper_params.submip_base_upper_guide),
      lower_guide_(settings.mip_hyper_params.submip_base_lower_guide),
      peak_(settings.mip_hyper_params.submip_base_fixrate),
      num_calls_(0)
  {
  }

  f_t get_target_fixrate(const mip_hyper_params_t<i_t, f_t>& settings, pcgenerator_t& rng);
  void update_stats(const simplex::simplex_solver_settings_t<i_t, f_t>& settings,
                    submip_status_t status,
                    f_t fixrate);

  void resize_bnb_slots(i_t num_slots)
  {
    std::lock_guard lock(child_bnb_mutex_);
    child_bnb_.assign(num_slots, nullptr);
  }

  void halt_submip()
  {
    std::lock_guard lock(child_bnb_mutex_);
    for (auto child_bnb : child_bnb_) {
      if (child_bnb) { child_bnb->halt_solver(); }
    }
  }
};

template <typename i_t, typename f_t>
void solve_submip(diving_worker_t<i_t, f_t>* worker,
                  branch_and_bound_stats_t<i_t, f_t>& bnb_stats,
                  const simplex::user_problem_t<i_t, f_t>& original_problem,
                  submip_manager_t<i_t, f_t>& submip_manager,
                  f_t fixrate,
                  int64_t simplex_iter_used,
                  simplex::simplex_solver_settings_t<i_t, f_t> submip_settings);

template <typename i_t, typename f_t>
void mutation(diving_worker_t<i_t, f_t>* worker,
              branch_and_bound_stats_t<i_t, f_t>& bnb_stats,
              const simplex::user_problem_t<i_t, f_t>& original_problem,
              submip_manager_t<i_t, f_t>& submip_manager,
              simplex::simplex_solver_settings_t<i_t, f_t> settings);

template <typename i_t, typename f_t>
void rins(diving_worker_t<i_t, f_t>* worker,
          branch_and_bound_stats_t<i_t, f_t>& bnb_stats,
          const simplex::user_problem_t<i_t, f_t>& original_problem,
          submip_manager_t<i_t, f_t>& submip_manager,
          simplex::simplex_solver_settings_t<i_t, f_t> submip_settings);

template <typename i_t, typename f_t>
void rens(diving_worker_t<i_t, f_t>* worker,
          branch_and_bound_stats_t<i_t, f_t>& bnb_stats,
          const simplex::user_problem_t<i_t, f_t>& original_problem,
          submip_manager_t<i_t, f_t>& submip_manager,
          simplex::simplex_solver_settings_t<i_t, f_t> submip_settings);

}  // namespace cuopt::mathematical_optimization::mip
