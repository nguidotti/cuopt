/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "lns.cuh"

#include <branch_and_bound/branch_and_bound.hpp>
#include <branch_and_bound/fractional.hpp>
#include <branch_and_bound/utils.hpp>
#include <dual_simplex/crossover.hpp>

#include <dual_simplex/presolve.hpp>
#include <dual_simplex/primal.hpp>

#include <mip_heuristics/feasibility_jump/fj_cpu_worker.cuh>

#include <presolve/third_party_presolve.hpp>

#include <utilities/scope_guard.hpp>

#include <raft/core/nvtx.hpp>

#define SUBMIP_VERBOSE true
#if SUBMIP_VERBOSE
#define DEBUG_SUBMIP(...) settings.log.print_format(__VA_ARGS__);
#else
#define DEBUG_SUBMIP(...)
#endif

namespace cuopt::mathematical_optimization::mip {

using simplex::domain_t;
using simplex::lp_problem_t;
using simplex::simplex_solver_settings_t;
using simplex::user_problem_t;
using simplex::variable_type_t;

namespace {

template <typename i_t, typename f_t>
void get_unfixed_integer_variables(const std::vector<f_t>& lower,
                                   const std::vector<f_t>& upper,
                                   const std::vector<variable_type_t>& var_types,
                                   f_t fixed_tol,
                                   std::vector<i_t>& integer_list)
{
  const i_t num_cols = var_types.size();
  for (i_t j = 0; j < num_cols; ++j) {
    if (var_types[j] == variable_type_t::CONTINUOUS) { continue; }
    if (std::abs(lower[j] - upper[j]) <= fixed_tol) { continue; }
    integer_list.push_back(j);
  }

  assert(!integer_list.empty() && "The integer list cannot be empty!");
}

template <typename i_t, typename f_t>
i_t calculate_fixrate(const std::vector<variable_type_t>& var_types,
                      const simplex_solver_settings_t<i_t, f_t>& settings,
                      const simplex::domain_t<i_t, f_t>& domain,
                      const size_t start)
{
  i_t num_fixed = 0;

  for (size_t k = start; k < domain.size(); ++k) {
    const simplex::bound_change_t<i_t, f_t>& change = domain[k];
    if (var_types[change.var] != variable_type_t::CONTINUOUS &&
        change.old_upper - change.old_lower > settings.fixed_tol &&
        change.new_upper - change.new_lower <= settings.fixed_tol) {
      ++num_fixed;
    }
  }

  return num_fixed;
}

template <typename i_t, typename f_t>
i_t extend_variable_fixings(const csr_matrix_t<i_t, f_t>& Arow,
                            const std::vector<variable_type_t>& var_types,
                            const simplex_solver_settings_t<i_t, f_t>& settings,
                            const std::vector<f_t>& obj_coeffs,
                            const std::vector<i_t>& fractional,
                            const std::vector<f_t>& current_sol,
                            const std::vector<f_t>& root_solution,
                            i_t target_num_fixed,
                            f_t start_time,
                            lp_problem_t<i_t, f_t>& lp,
                            domain_t<i_t, f_t>& domain)
{
  std::vector<std::tuple<f_t, i_t, f_t>> candidates;
  for (i_t j : fractional) {
    if (std::abs(lp.lower[j] - lp.upper[j]) <= settings.fixed_tol) { continue; }
    if (toc(start_time) >= settings.time_limit || settings.received_halt_signal()) return -1;

    f_t root_change = current_sol[j] - root_solution[j];
    f_t obj_coeff   = obj_coeffs[j];
    f_t fixed_val   = 0;

    if (root_change >= 0.4) {
      fixed_val = std::ceil(current_sol[j]);
    } else if (root_change <= -0.4) {
      fixed_val = std::floor(current_sol[j]);
    } else if (obj_coeff > settings.zero_tol) {
      fixed_val = std::ceil(current_sol[j]);
    } else if (obj_coeff < -settings.zero_tol) {
      fixed_val = std::floor(current_sol[j]);
    } else {
      fixed_val = std::round(current_sol[j]);
    }

    candidates.push_back(std::make_tuple(std::abs(fixed_val - current_sol[j]), j, fixed_val));
  }

  std::sort(candidates.begin(), candidates.end(), [](auto a, auto b) {
    return std::get<0>(a) < std::get<0>(b);
  });

  i_t num_fixed         = 0;
  f_t accumulate_change = 0;

  for (auto [dist, j, fixed_val] : candidates) {
    if (num_fixed >= target_num_fixed) break;
    if (toc(start_time) >= settings.time_limit || settings.received_halt_signal()) return -1;

    fixed_val = std::clamp(fixed_val, lp.lower[j], lp.upper[j]);
    i_t start = domain.size();
    simplex::bound_change_t<i_t, f_t> bound_change{
      .var       = j,
      .old_upper = lp.upper[j],
      .new_upper = fixed_val,
      .old_lower = lp.lower[j],
      .new_lower = fixed_val,
      .origin    = simplex::bound_change_origin_t::BRANCH};
    if (!domain.apply_and_propagate(Arow, var_types, settings, bound_change, lp)) {
      DEBUG_SUBMIP(
        "Bound propagation detect infeasibility. var={}, lower={:.4g} -> {:.4g}, "
        "upper={:.4g} -> {:.4g}",
        j,
        bound_change.old_lower,
        bound_change.new_lower,
        bound_change.old_upper,
        bound_change.new_upper);
      domain.backtrack_to_parent(lp);
    }
    num_fixed += calculate_fixrate(var_types, settings, domain, start);

    // Limit the amount of fixing to the current LP.
    accumulate_change += dist;
    if (accumulate_change >= 0.5) break;
  }

  return num_fixed;
}

template <typename i_t, typename f_t>
bool solve_lp(const branch_and_bound_stats_t<i_t, f_t>& bnb_stats,
              const simplex_solver_settings_t<i_t, f_t>& settings,
              diving_worker_t<i_t, f_t>* worker,
              branch_and_bound_stats_t<i_t, f_t>& stats,
              std::vector<i_t>& fractional)
{
  branch_and_bound_t<i_t, f_t>* bnb_ptr = worker->branch_and_bound_ptr;
  mip_node_t<i_t, f_t>* node            = &worker->start_node;

  // After fixing the variables, re-solve the LP relaxation. We use the optimal solution
  // in the next iteration to find additional variable fixings.
  // We continue to do this until enough variables were fixed or no variable is left to fix.
  simplex::logger_t log;
  log.log = false;

  int64_t iter_offset =
    settings.inside_submip ? 0 : settings.mip_hyper_params.submip_iteration_limit_offset;
  int64_t simplex_iter       = bnb_stats.total_simplex_iters;
  f_t iter_ratio             = settings.mip_hyper_params.submip_iteration_limit_ratio;
  int64_t simplex_iter_limit = iter_offset + simplex_iter * iter_ratio;
  int64_t max_iter           = simplex_iter_limit - stats.total_simplex_iters;
  if (max_iter <= 0) {
    DEBUG_SUBMIP("Max iteration reached! {}/{}",
                 stats.total_simplex_iters.load(std::memory_order::relaxed),
                 simplex_iter_limit)
    return false;
  }

  f_t leaf_obj;
  f_t upper_bound = bnb_ptr->get_upper_bound();
  simplex::dual_status_t lp_status =
    worker->solve_node_lp(settings, node, stats, log, max_iter, bnb_stats.start_time, upper_bound);

  node_status_t node_status =
    worker->classify_node(node, lp_status, upper_bound, settings, fractional, leaf_obj);

  if (node_status == node_status_t::FATHOMED) {
    DEBUG_SUBMIP("Dual simplex returned {}. leaf_obj={:.4g}, upper_bound={:.4g}",
                 simplex::dual_status_to_string(lp_status),
                 leaf_obj,
                 upper_bound)
    return false;
  }

  // We found a feasible solution when fixing the variables in RINS/RENS.
  if (node_status == node_status_t::INTEGER_FEASIBLE) {
    bnb_ptr->set_solution_from_worker(
      worker->leaf_problem, leaf_obj, worker->leaf_solution.x, -1, worker->search_strategy);
    DEBUG_SUBMIP("Found an integer-feasible solution with obj={:.4g}. upper_bound={:.4g}",
                 leaf_obj,
                 bnb_ptr->get_upper_bound())
    return false;
  }

  worker->recompute_basis = false;
  return true;
}

submip_status_t convert_bnb_status(mip_status_t status, bool improved_incumbent)
{
  if (improved_incumbent) return SUCCESS;

  switch (status) {
    case mip_status_t::INFEASIBLE: return INFEASIBLE;
    case mip_status_t::UNBOUNDED: return INFEASIBLE;
    case mip_status_t::OPTIMAL: return EMPTY;
    case mip_status_t::ITERATION_LIMIT: return TRUNCATED;
    case mip_status_t::TIME_LIMIT: return TRUNCATED;
    case mip_status_t::NODE_LIMIT: return TRUNCATED;
    case mip_status_t::WORK_LIMIT: return TRUNCATED;
    case mip_status_t::HALT: return UNSET;
    case mip_status_t::NUMERICAL: return UNSET;
    case mip_status_t::UNSET: return UNSET;
  }
  return UNSET;
}

}  // namespace

template <typename i_t, typename f_t>
f_t submip_manager_t<i_t, f_t>::get_target_fixrate(const mip_hyper_params_t<i_t, f_t>& settings,
                                                   pcgenerator_t& rng)
{
  mutex_.lock();
  f_t peak    = peak_;
  f_t sigma_r = std::max(settings.submip_min_sigma, (upper_guide_ - peak_) / 2);
  f_t sigma_l = std::max(settings.submip_min_sigma, (peak_ - lower_guide_) / 2);
  mutex_.unlock();

  // Rejection sampling to get a fix rate within the range [L, U].
  f_t fixrate = 0;
  do {
    fixrate = rng.split_normal(peak, sigma_r, sigma_l);
  } while (fixrate < settings.submip_min_fixrate || fixrate > settings.submip_max_fixrate);

  return fixrate;
}

template <typename i_t, typename f_t>
void submip_manager_t<i_t, f_t>::update_stats(const simplex_solver_settings_t<i_t, f_t>& settings,
                                              submip_status_t status,
                                              f_t fixrate)
{
  std::lock_guard lock(mutex_);

  f_t lambda         = settings.mip_hyper_params.submip_observation_decay;
  f_t omega_T        = settings.mip_hyper_params.submip_truncated_weight;
  f_t omega_I        = settings.mip_hyper_params.submip_infeasible_weight;
  f_t omega_S        = settings.mip_hyper_params.submip_success_weight;
  f_t omega_E        = settings.mip_hyper_params.submip_empty_weight;
  f_t target_fixrate = settings.mip_hyper_params.submip_max_fixrate;
  f_t min_fixrate    = settings.mip_hyper_params.submip_min_fixrate;
  f_t base_upper     = settings.mip_hyper_params.submip_base_upper_guide;
  f_t base_lower     = settings.mip_hyper_params.submip_base_lower_guide;
  f_t phi            = std::clamp(fixrate, min_fixrate, target_fixrate);

  switch (status) {
    case SUCCESS:
      upper_guide_ = base_upper + lambda * (upper_guide_ - base_upper);
      lower_guide_ = base_lower + lambda * (lower_guide_ - base_lower);
      peak_ += omega_S * (phi - peak_);
      break;

    case TRUNCATED:
      upper_guide_ = base_upper + lambda * (upper_guide_ - base_upper);
      lower_guide_ += omega_T * (phi - lower_guide_);
      peak_ += omega_T * (0.5 * (std::max(phi, peak_) + upper_guide_) - peak_);
      break;

    case EMPTY:
      lower_guide_ = base_lower + lambda * (lower_guide_ - base_lower);
      upper_guide_ += omega_E * (phi - upper_guide_);
      peak_ += omega_E * (0.5 * (std::min(phi, peak_) + lower_guide_) - peak_);
      break;

    case INFEASIBLE:
      lower_guide_ = base_lower + lambda * (lower_guide_ - base_lower);
      upper_guide_ += omega_I * (phi - upper_guide_);
      peak_ += omega_I * (0.5 * (std::min(phi, peak_) + lower_guide_) - peak_);
      break;

    case UNSET: return;  // This should never happen
  }

  lower_guide_ = std::min(lower_guide_, peak_);
  upper_guide_ = std::max(upper_guide_, peak_);

  ++status_count_[status];
  fixrate_sum_per_status_[status] += fixrate;

  DEBUG_SUBMIP(
    "Updated controller with status={}, fixrate={}, count={}/{}, mean fixrate={}, "
    "lower_guide={:.4g}, peak={:.4g}, upper_guide={:.4g}",
    (int)status,
    fixrate,
    status_count_[status],
    num_calls_.load(std::memory_order_acquire),
    fixrate_sum_per_status_[status] / status_count_[status],
    lower_guide_,
    peak_,
    upper_guide_);
}

template <typename i_t, typename f_t>
void solve_submip(diving_worker_t<i_t, f_t>* worker,
                  branch_and_bound_stats_t<i_t, f_t>& bnb_stats,
                  const user_problem_t<i_t, f_t>& original_problem,
                  submip_manager_t<i_t, f_t>& submip_manager,
                  f_t fixrate,
                  int64_t simplex_iter_used,
                  simplex_solver_settings_t<i_t, f_t> settings)
{
  f_t time_limit = settings.time_limit - toc(bnb_stats.start_time);
  if (time_limit <= 0) { return; }

  double start_time                     = tic();
  branch_and_bound_t<i_t, f_t>* bnb_ptr = worker->branch_and_bound_ptr;
  bool inside_submip                    = settings.inside_submip;
  omp_atomic_t<bool> improved_incumbent = false;

  submip_manager.mutex_.lock();
  i_t past_success_count = submip_manager.status_count_[SUCCESS];
  submip_manager.mutex_.unlock();

  // If not enough variables was fixed (the neighbourhood is too loose) or the sub-MIP already
  // found a solution that improved the incumbent, then do a DFS with a backtrack_limit of 5
  // levels up to try to find a feasible solution quickly from the neighbourhood.
  if (fixrate < settings.mip_hyper_params.submip_min_fixrate_cap ||
      (inside_submip && past_success_count != 0)) {
    worker->start_node.packed_vstatus = simplex::compress_vstatus(worker->leaf_vstatus);

    fj_cpu_worker_t<i_t, f_t> submip_fj_cpu_worker;
    if (settings.mip_hyper_params.submip_enable_cpufj) {
      submip_fj_cpu_worker.improvement_callback =
        [bnb_ptr](f_t obj, const std::vector<f_t>& assignment, double work_units) {
          bnb_ptr->set_solution_from_cpu_fj(obj, assignment, work_units);
        };

      f_t work_limit = 1.0;
      submip_fj_cpu_worker.create_worker(worker->leaf_problem,
                                         worker->var_types,
                                         original_problem.num_cols,
                                         worker->leaf_solution.x,
                                         settings,
                                         std::format("{} [CPU FJ]", settings.log.log_prefix),
                                         worker->rng.next_i64());
      submip_fj_cpu_worker.run_sync(time_limit, work_limit);
    }

    DEBUG_SUBMIP("Running a quick DFS. fixrate={:.4g}", fixrate);

    simplex_solver_settings_t<i_t, f_t> dfs_settings = settings;
    dfs_settings.mip_hyper_params.diving_backtrack_limit =
      dfs_settings.mip_hyper_params.submip_dfs_max_backtrack;
    worker->domain.clear();
    worker->dive(dfs_settings, bnb_stats);
    return;
  }

  bool is_root_heuristic = settings.inside_root_node;
  i_t submip_level       = settings.mip_hyper_params.submip_level + 1;
  f_t upper_bound        = bnb_ptr->get_upper_bound();
  f_t user_lower         = compute_user_objective(worker->leaf_problem, bnb_ptr->get_lower_bound());
  f_t user_obj           = compute_user_objective(worker->leaf_problem, upper_bound);
  f_t rel_gap            = user_relative_gap(user_obj, user_lower);
  i_t explored           = bnb_stats.nodes_explored;

  bool max_recursion = submip_level > settings.mip_hyper_params.submip_max_level;
  settings.mip_hyper_params.submip_rins =
    settings.mip_hyper_params.submip_rins != 0 && !max_recursion;
  settings.mip_hyper_params.submip_rens =
    settings.mip_hyper_params.submip_rens != 0 && !max_recursion;

  settings.node_limit = settings.mip_hyper_params.submip_node_limit_offset + explored / 20;

  // Add offset only on the top call, we want number of simplex iteration to decay
  // as we go down the recursion to avoid spending too much time in the deeper levels.
  int64_t iter_offset = inside_submip ? 0 : settings.mip_hyper_params.submip_iteration_limit_offset;
  int64_t simplex_iter = bnb_stats.total_simplex_iters;
  f_t iter_ratio       = settings.mip_hyper_params.submip_iteration_limit_ratio;

  settings.mip_hyper_params.bnb_simplex_iteration_limit =
    iter_offset + simplex_iter * iter_ratio - simplex_iter_used;
  if (settings.mip_hyper_params.bnb_simplex_iteration_limit <= 0) { return; }

  settings.time_limit = time_limit;

  settings.relative_mip_gap_tol =
    std::min(settings.mip_hyper_params.submip_target_mip_gap, rel_gap);

  settings.print_presolve_stats                     = false;
  settings.num_threads                              = 1;
  settings.reliability_branching                    = 0;
  settings.clique_cuts                              = 0;
  settings.zero_half_cuts                           = 0;
  settings.inside_submip                            = 1;
  settings.strong_branching_simplex_iteration_limit = 50;
  settings.inside_root_node                         = 0;
  settings.mip_hyper_params.submip_level            = submip_level;
  settings.benchmark_info_ptr                       = nullptr;
  settings.log.log                                  = false;
  std::string_view log_prefix                       = settings.log.log_prefix;

  DEBUG_SUBMIP("Sub-MIP: fixrate={:.2f}", fixrate)
  DEBUG_SUBMIP(
    "Sub-MIP solve settings: time_limit={:.2f}, node_limit={}, iter_limit={} (current_iter={}), "
    "tol={:g}",
    settings.time_limit,
    settings.node_limit,
    settings.mip_hyper_params.bnb_simplex_iteration_limit,
    bnb_stats.total_simplex_iters.load(std::memory_order::relaxed),
    settings.relative_mip_gap_tol);

  // The `worker->leaf_problem` is directly converted to an `user_problem_t`, meaning that
  // there is only equality rows (the range row vector is empty) and it contains
  // structural + slacks + cuts constraints/variables.
  user_problem_t<i_t, f_t> submip_problem(original_problem.handle_ptr);
  simplex::convert_lp_to_user_problem(
    worker->leaf_problem, worker->var_types, settings, submip_problem);

  third_party_presolve_t<i_t, f_t> presolver;
  f_t presolve_time_limit = std::min(0.1 * settings.time_limit, 60.0);
  third_party_presolve_status_t presolver_status =
    presolver.apply_to_subproblem(submip_problem, settings, presolve_time_limit, 1);

  if (presolver_status == third_party_presolve_status_t::INFEASIBLE ||
      presolver_status == third_party_presolve_status_t::UNBNDORINFEAS ||
      presolver_status == third_party_presolve_status_t::UNBOUNDED) {
    DEBUG_SUBMIP("Presolve detected infeasibility");
    submip_manager.update_stats(settings, INFEASIBLE, fixrate);
    return;
  }

  // Also handle optimal
  if (submip_problem.num_rows == 0 || submip_problem.num_cols == 0) {
    DEBUG_SUBMIP("Reduced to a trivial {} x {} problem; solving by bound pushing",
                 submip_problem.num_rows,
                 submip_problem.num_cols);

    // No rows left: each variable goes to the bound that minimizes its objective term.
    std::vector<f_t> solution(submip_problem.num_cols, 0.0);
    for (i_t j = 0; j < submip_problem.num_cols; ++j) {
      const f_t obj_coeff = submip_problem.objective[j];
      const f_t lower     = submip_problem.lower[j];
      const f_t upper     = submip_problem.upper[j];

      if (obj_coeff > 0) {
        solution[j] = lower;
      } else if (obj_coeff < 0) {
        solution[j] = upper;
      } else if (std::isfinite(lower)) {
        solution[j] = lower;
      } else if (std::isfinite(upper)) {
        solution[j] = upper;
      }

      if (!std::isfinite(solution[j])) {
        DEBUG_SUBMIP("Trivial problem is unbounded in variable {}", j);
        submip_manager.update_stats(settings, INFEASIBLE, fixrate);
        return;
      }
    }

    bool success = bnb_ptr->set_solution_from_submip(
      worker->leaf_problem, solution, presolver, worker->search_strategy);
    submip_manager.update_stats(settings, success ? SUCCESS : EMPTY, fixrate);
    return;
  }

  double presolve_time = toc(start_time);

  if (settings.received_halt_signal()) { return; }

  settings.heuristic_preemption_callback   = nullptr;
  settings.dual_simplex_objective_callback = nullptr;
  settings.set_simplex_solution_callback   = nullptr;
  settings.solution_callback               = [bnb_ptr, &improved_incumbent, &presolver, worker](
                                 const std::vector<f_t>& solution, f_t obj) {
    improved_incumbent |= bnb_ptr->set_solution_from_submip(
      worker->leaf_problem, solution, presolver, worker->search_strategy);
  };

  DEBUG_SUBMIP("Sub-MIP: {} constraints, {} variables, {} nonzeros\n",
               submip_problem.num_rows,
               submip_problem.num_cols,
               submip_problem.A.nnz());

  probing_implied_bound_t<i_t, f_t> empty_probing(submip_problem.num_cols);
  settings.time_limit -= presolve_time;
  if (settings.time_limit <= 0.0) { return; }

  branch_and_bound_t submip_bnb(submip_problem, settings, tic(), empty_probing);
  simplex::mip_solution_t<i_t, f_t> submip_solution(submip_problem.num_cols);

  std::vector<f_t> presolved_incumbent;

  // We do not have an incumbent yet, so skip the initial guess.
  if (!worker->current_incumbent.empty()) {
    // Crush the incumbent to presolve space. It may not be valid for the sub-MIP since we
    // may fix integer variables that does not match the current incumbent to reach the target
    // fix rate.
    presolver.crush_primal_solution(submip_problem, worker->current_incumbent, presolved_incumbent);
    submip_bnb.set_initial_guess(presolved_incumbent);
  }

  // Even if we do not have a valid incumbent now, the upper bound can still be set by the early
  // heuristics.
  upper_bound = bnb_ptr->get_upper_bound();
  if (std::isfinite(upper_bound)) {
    const f_t user_upper    = compute_user_objective(worker->leaf_problem, upper_bound);
    const f_t submip_cutoff = compute_presolved_objective(submip_bnb.original_lp_, user_upper);
    submip_bnb.set_initial_upper_bound(submip_cutoff);
  }

  if (!is_root_heuristic)
    submip_bnb.set_initial_pseudocost(bnb_ptr->pc_, presolver.get_reduced_to_original_map());

  if (bnb_ptr->submip_halt_callback_) {
    // Copy the halt callback to the deeper level.
    submip_bnb.set_submip_halt_callback(bnb_ptr->submip_halt_callback_);
  } else {
    // This should only be called by the main solver.
    submip_bnb.set_submip_halt_callback([bnb_ptr](f_t, f_t submip_lower_bound) {
      f_t user_upper = compute_user_objective(bnb_ptr->original_lp_, bnb_ptr->get_upper_bound());
      bool is_cutoff = bnb_ptr->original_lp_.obj_scale > 0 ? submip_lower_bound > user_upper
                                                           : user_upper > submip_lower_bound;
      bool is_solver_running = !bnb_ptr->has_solver_stopped();
      return is_cutoff || !is_solver_running || bnb_ptr->received_halt_signal();
    });
  }

  fj_cpu_worker_t<i_t, f_t> submip_fj_cpu_worker;

  if (settings.mip_hyper_params.submip_enable_cpufj) {
    // Since we do not have an incumbent, use the LP solution of the last round of variable fixing
    // in RENS.
    if (worker->search_strategy == search_strategy_t::RENS) {
      presolver.crush_primal_solution(submip_problem, worker->leaf_solution.x, presolved_incumbent);
    }

    // Launch a CPU FJ worker on the presolved sub-MIP with a fixed budget (in terms of work units)
    // to run in parallel with the cut-and-branch algorithm with the goal of finding a quick
    // feasible solution for the sub-MIP problem. The CPU FJ uses the current incumbent (crushed
    // into the presolved space) as initial guess. The worker is automatically stop when we go out
    // of the scope.
    std::vector<f_t> initial_guess;
    crush_primal_solution(submip_problem,
                          submip_bnb.original_lp_,
                          presolved_incumbent,
                          submip_bnb.new_slacks_,
                          initial_guess);

    submip_fj_cpu_worker.improvement_callback =
      [&submip_bnb](f_t obj, const std::vector<f_t>& solution, double work_units) {
        submip_bnb.set_solution_from_cpu_fj(obj, solution, work_units);
      };

    f_t work_limit = 1.0;
    submip_fj_cpu_worker.create_worker(submip_bnb.original_lp_,
                                       submip_bnb.var_types_,
                                       submip_bnb.original_problem_.num_cols,
                                       initial_guess,
                                       submip_bnb.settings_,
                                       std::format("{} [CPU FJ]", log_prefix),
                                       worker->rng.next_i64());
    submip_fj_cpu_worker.run_async(settings.time_limit, work_limit);
  }

  submip_manager.child_bnb_mutex_.lock();
  submip_manager.child_bnb_[worker->worker_id] = &submip_bnb;
  submip_manager.child_bnb_mutex_.unlock();

  scope_guard solver_scope([&]() {
    std::lock_guard lock(submip_manager.child_bnb_mutex_);
    submip_manager.child_bnb_[worker->worker_id] = nullptr;
  });

  if (bnb_ptr->received_halt_signal()) { submip_bnb.halt_solver(); }

  mip_status_t submip_status = submip_bnb.solve(submip_solution);
  f_t submip_time            = toc(start_time);

  DEBUG_SUBMIP(
    "Sub-MIP: status={}, iterations={} (total={}), presolve_time={:.2f}, total_time={:.2f} \n",
    mip_status_to_string(submip_status),
    submip_solution.simplex_iterations,
    bnb_stats.total_simplex_iters.load(std::memory_order::relaxed),
    presolve_time,
    submip_time);

  // Accumulate simplex iterations to determine when to stop exploring the sub-MIP
  if (inside_submip) { bnb_stats.total_simplex_iters += submip_solution.simplex_iterations; }

  if (submip_solution.has_incumbent) {
    improved_incumbent |= bnb_ptr->set_solution_from_submip(
      worker->leaf_problem, submip_solution.x, presolver, worker->search_strategy);
  }

  submip_status_t status = convert_bnb_status(submip_status, improved_incumbent);
  if (status != UNSET) { submip_manager.update_stats(settings, status, fixrate); }
}

template <typename i_t, typename f_t>
void mutation(diving_worker_t<i_t, f_t>* worker,
              branch_and_bound_stats_t<i_t, f_t>& bnb_stats,
              const user_problem_t<i_t, f_t>& original_problem,
              submip_manager_t<i_t, f_t>& submip_manager,
              simplex_solver_settings_t<i_t, f_t> settings)
{
  raft::common::nvtx::range scope("BB::mutation_thread");
  if (worker->orbital_fixing) { worker->orbital_fixing->disable(); }

  i_t submip_level = settings.mip_hyper_params.submip_level + 1;
  settings.log.log_prefix +=
    std::format("[{} {}] ", search_strategy_to_string(worker->search_strategy), submip_level);

  assert((worker->search_strategy == search_strategy_t::MUTATION) &&
         "Sub-MIP worker must be set to MUTATION type");

  ++submip_manager.num_calls_;

  lp_problem_t<i_t, f_t>& lp  = worker->leaf_problem;
  std::vector<f_t>& incumbent = worker->current_incumbent;
  domain_t<i_t, f_t>& domain  = worker->domain;

  std::vector<i_t> integer_list;
  get_unfixed_integer_variables(
    lp.lower, lp.upper, worker->var_types, settings.fixed_tol, integer_list);
  worker->rng.shuffle(integer_list);
  worker->domain.compute_activities(worker->Arow, worker->leaf_problem);

  f_t target_fixrate   = submip_manager.get_target_fixrate(settings.mip_hyper_params, worker->rng);
  i_t target_num_fixed = target_fixrate * integer_list.size();
  i_t num_fixed        = 0;

  for (i_t j : integer_list) {
    if (toc(bnb_stats.start_time) >= settings.time_limit) {
      DEBUG_SUBMIP("Time limit reached.")
      return;
    }

    if (settings.received_halt_signal()) {
      DEBUG_SUBMIP("Halt signal received.")
      return;
    }

    if (num_fixed >= target_num_fixed) break;
    if (std::abs(lp.lower[j] - lp.upper[j]) <= settings.fixed_tol) continue;

    f_t fixed_val = std::clamp(std::round(incumbent[j]), lp.lower[j], lp.upper[j]);
    i_t start     = domain.size();
    simplex::bound_change_t<i_t, f_t> bound_change{
      .var       = j,
      .old_upper = lp.upper[j],
      .new_upper = fixed_val,
      .old_lower = lp.lower[j],
      .new_lower = fixed_val,
      .origin    = simplex::bound_change_origin_t::BRANCH};

    if (!domain.apply_and_propagate(worker->Arow, worker->var_types, settings, bound_change, lp)) {
      DEBUG_SUBMIP(
        "Bound propagation detect infeasibility. var={}, lower={:.4g} -> {:.4g}, "
        "upper={:.4g} -> {:.4g}",
        j,
        bound_change.old_lower,
        bound_change.new_lower,
        bound_change.old_upper,
        bound_change.new_upper);

      domain.backtrack_to_parent(lp);
    }

    num_fixed += calculate_fixrate(worker->var_types, settings, domain, start);
  }

  f_t fixrate = static_cast<f_t>(num_fixed) / integer_list.size();

  DEBUG_SUBMIP("variables fixed = {:.0f} ({:.2f}), target fixrate = {:.0f} ({:.2f})",
               fixrate * integer_list.size(),
               fixrate,
               target_fixrate * integer_list.size(),
               target_fixrate);

  solve_submip(worker, bnb_stats, original_problem, submip_manager, fixrate, 0, settings);
}

template <typename i_t, typename f_t>
void rins(diving_worker_t<i_t, f_t>* worker,
          branch_and_bound_stats_t<i_t, f_t>& bnb_stats,
          const user_problem_t<i_t, f_t>& original_problem,
          submip_manager_t<i_t, f_t>& submip_manager,
          simplex_solver_settings_t<i_t, f_t> settings)
{
  raft::common::nvtx::range scope("BB::submip_thread");
  if (worker->orbital_fixing) { worker->orbital_fixing->disable(); }

  i_t submip_level = settings.mip_hyper_params.submip_level + 1;
  settings.log.log_prefix +=
    std::format("[{} {}] ", search_strategy_to_string(worker->search_strategy), submip_level);
  branch_and_bound_t<i_t, f_t>* bnb_ptr = worker->branch_and_bound_ptr;

  assert(worker->search_strategy == search_strategy_t::RINS &&
         "Sub-MIP worker must be set to RINS type");

  ++submip_manager.num_calls_;

  bool has_submip         = false;
  worker->recompute_basis = true;

  branch_and_bound_stats_t<i_t, f_t> stats;
  lp_problem_t<i_t, f_t>& lp    = worker->leaf_problem;
  std::vector<f_t>& incumbent   = worker->current_incumbent;
  domain_t<i_t, f_t>& domain    = worker->domain;
  std::vector<f_t>& current_sol = worker->leaf_solution.x;
  worker->domain.compute_activities(worker->Arow, worker->leaf_problem);

  std::vector<i_t> fractional;
  fractional_variables(settings, current_sol, worker->var_types, fractional);

  std::vector<i_t> integer_list;
  get_unfixed_integer_variables(
    lp.lower, lp.upper, worker->var_types, settings.fixed_tol, integer_list);

  i_t num_integers   = integer_list.size();
  f_t target_fixrate = submip_manager.get_target_fixrate(settings.mip_hyper_params, worker->rng);
  f_t min_fixrate    = settings.mip_hyper_params.submip_min_fixrate;
  f_t fixrate        = 0;
  i_t num_fixed      = 0;
  bool has_stopped   = false;

  i_t round = 0;

  while (!bnb_ptr->has_solver_stopped() && !settings.received_halt_signal()) {
    f_t prev_stack_size = worker->domain.size();
    f_t prev_fixrate    = fixrate;
    i_t new_fixings     = 0;
    i_t round_num_fixed = (target_fixrate - prev_fixrate) * num_integers;

    if (round_num_fixed == 0) {
      has_submip = fixrate > 0;
      break;
    }

    // Shuffle the fractional and integer list, so every variable has the same chance to the picked
    // (we iterate the list in order).
    worker->rng.shuffle(integer_list);
    worker->rng.shuffle(fractional);

    // RINS neighbourhood: Fix all the integer variables where the current solution matches the
    // incumbent. We are using the `target_fixrate` here to allow RINS to fix all integer variables
    // that it can within our budget.
    for (i_t j : integer_list) {
      if (toc(bnb_stats.start_time) >= settings.time_limit) {
        DEBUG_SUBMIP("Time limit reached.")
        has_stopped = true;
        break;
      }

      if (settings.received_halt_signal()) {
        DEBUG_SUBMIP("Halt signal received.")
        has_stopped = true;
        break;
      }

      if (new_fixings >= round_num_fixed) break;
      if (std::abs(lp.lower[j] - lp.upper[j]) <= settings.fixed_tol) continue;
      if (std::abs(current_sol[j] - incumbent[j]) <= settings.integer_tol) {
        f_t fixed_val = std::clamp(std::round(incumbent[j]), lp.lower[j], lp.upper[j]);
        i_t start     = domain.size();
        simplex::bound_change_t<i_t, f_t> bound_change{
          .var       = j,
          .old_upper = lp.upper[j],
          .new_upper = fixed_val,
          .old_lower = lp.lower[j],
          .new_lower = fixed_val,
          .origin    = simplex::bound_change_origin_t::BRANCH};
        if (!domain.apply_and_propagate(
              worker->Arow, worker->var_types, settings, bound_change, lp)) {
          DEBUG_SUBMIP(
            "Bound propagation detect infeasibility. var={}, lower={:.4g} -> {:.4g}, "
            "upper={:.4g} -> {:.4g}",
            j,
            bound_change.old_lower,
            bound_change.new_lower,
            bound_change.old_upper,
            bound_change.new_upper);

          domain.backtrack_to_parent(lp);
        }

        new_fixings += calculate_fixrate(worker->var_types, settings, domain, start);
      }
    }

    if (has_stopped) { break; }

    num_fixed += new_fixings;
    fixrate = static_cast<f_t>(num_fixed) / integer_list.size();

    // The RINS neighbourhood ran dry. If it is already tight enough, take it rather than
    // diluting it with fixings that do not agree with the incumbent.
    if (prev_fixrate == fixrate && fixrate >= min_fixrate) {
      has_submip = true;
      break;
    }

    // Even considering the entire integer list, we were unable to fix a single variable in this
    // iteration. Iterate over the fractional variables again and fixing those that closest to
    // an integer solution first in order to reach the fixing threshold.
    if (fixrate == prev_fixrate && prev_stack_size == worker->domain.size()) {
      new_fixings = extend_variable_fixings(worker->Arow,
                                            worker->var_types,
                                            settings,
                                            worker->leaf_problem.objective,
                                            fractional,
                                            current_sol,
                                            worker->root_solution,
                                            round_num_fixed,
                                            bnb_stats.start_time,
                                            worker->leaf_problem,
                                            worker->domain);
      if (new_fixings < 0) {
        DEBUG_SUBMIP("Round {}: stopped fixing (infeasible, halted or time limit reached).", round)
        break;
      }

      num_fixed += new_fixings;
      fixrate = static_cast<f_t>(num_fixed) / integer_list.size();

      if (fixrate >= min_fixrate) {
        has_submip = true;
        break;
      }
    }

    DEBUG_SUBMIP(
      "Round {}: fixed {:.0f} ({:.2f}) -> {:.0f} ({:.2f}) variables ({:+}). target fixrate = "
      "{:.4g} ({:.0f})",
      round,
      prev_fixrate * num_integers,
      prev_fixrate,
      fixrate * num_integers,
      fixrate,
      new_fixings,
      target_fixrate,
      target_fixrate * num_integers);

    if (fixrate >= target_fixrate) {
      has_submip = true;
      break;
    }

    if (toc(bnb_stats.start_time) >= settings.time_limit) { break; }

    bool continue_dive = solve_lp(bnb_stats, settings, worker, stats, fractional);
    if (!continue_dive) { break; }
    ++round;
  }

  // Accumulate the iterations for sub-MIP so it stops when it reaches the allocated budget.
  if (settings.inside_submip) { bnb_stats.total_simplex_iters += stats.total_simplex_iters; }

  DEBUG_SUBMIP("variables fixed = {:.0f} ({:.2f}), target fixrate = {:.0f} ({:.2f})",
               fixrate * num_integers,
               fixrate,
               target_fixrate * num_integers,
               target_fixrate);

  if (has_submip) {
    solve_submip(worker,
                 bnb_stats,
                 original_problem,
                 submip_manager,
                 fixrate,
                 stats.total_simplex_iters,
                 settings);
  }
}

template <typename i_t, typename f_t>
void rens(diving_worker_t<i_t, f_t>* worker,
          branch_and_bound_stats_t<i_t, f_t>& bnb_stats,
          const user_problem_t<i_t, f_t>& original_problem,
          submip_manager_t<i_t, f_t>& submip_manager,
          simplex_solver_settings_t<i_t, f_t> settings)
{
  raft::common::nvtx::range scope("BB::submip_thread");
  if (worker->orbital_fixing) { worker->orbital_fixing->disable(); }

  i_t submip_level = settings.mip_hyper_params.submip_level + 1;
  settings.log.log_prefix +=
    std::format("[{} {}] ", search_strategy_to_string(worker->search_strategy), submip_level);
  branch_and_bound_t<i_t, f_t>* bnb_ptr = worker->branch_and_bound_ptr;

  assert(worker->search_strategy == search_strategy_t::RENS &&
         "Sub-MIP worker must be set to RENS type");

  ++submip_manager.num_calls_;

  bool has_submip         = false;
  worker->recompute_basis = true;

  branch_and_bound_stats_t<i_t, f_t> stats;
  lp_problem_t<i_t, f_t>& lp    = worker->leaf_problem;
  domain_t<i_t, f_t>& domain    = worker->domain;
  std::vector<f_t>& current_sol = worker->leaf_solution.x;
  worker->domain.compute_activities(worker->Arow, worker->leaf_problem);

  std::vector<i_t> fractional;
  fractional_variables(settings, current_sol, worker->var_types, fractional);

  std::vector<i_t> integer_list;
  get_unfixed_integer_variables(
    lp.lower, lp.upper, worker->var_types, settings.fixed_tol, integer_list);

  i_t num_integers   = integer_list.size();
  f_t target_fixrate = submip_manager.get_target_fixrate(settings.mip_hyper_params, worker->rng);
  f_t min_fixrate    = settings.mip_hyper_params.submip_min_fixrate;
  f_t fixrate        = 0;
  i_t num_fixed      = 0;
  f_t close_ratio    = settings.mip_hyper_params.submip_round_close_ratio;
  bool has_stopped   = false;

  i_t round = 0;

  while (!bnb_ptr->has_solver_stopped() && !settings.received_halt_signal()) {
    f_t prev_stack_size = worker->domain.size();
    f_t prev_fixrate    = fixrate;
    f_t round_target    = std::min(1.0 - (1.0 - fixrate) * close_ratio, target_fixrate);
    i_t new_fixings     = 0;

    i_t round_num_fixed = (round_target - prev_fixrate) * num_integers;
    if (round_num_fixed == 0) {
      round_target    = target_fixrate;
      round_num_fixed = (target_fixrate - prev_fixrate) * num_integers;
      if (round_num_fixed == 0) {
        has_submip = fixrate > 0;
        break;
      }
    }

    // Shuffle the fractional and integer list, so every variable has the same chance to the picked
    // (we iterate the list in order).
    worker->rng.shuffle(integer_list);
    worker->rng.shuffle(fractional);

    for (i_t j : integer_list) {
      if (toc(bnb_stats.start_time) >= settings.time_limit) {
        DEBUG_SUBMIP("Time limit reached.")
        has_stopped = true;
        break;
      }

      if (settings.received_halt_signal()) {
        DEBUG_SUBMIP("Halt signal received.")
        has_stopped = true;
        break;
      }

      if (new_fixings >= round_num_fixed) break;
      if (std::abs(lp.lower[j] - lp.upper[j]) <= settings.fixed_tol) continue;

      simplex::bound_change_t<i_t, f_t> bound_change{
        .var = j, .origin = simplex::bound_change_origin_t::BRANCH};
      bound_change.old_lower = lp.lower[j];
      bound_change.old_upper = lp.upper[j];
      bound_change.new_lower = std::clamp(std::floor(current_sol[j]), lp.lower[j], lp.upper[j]);
      bound_change.new_upper = std::clamp(std::ceil(current_sol[j]), lp.lower[j], lp.upper[j]);

      if (bound_change.new_lower != lp.lower[j] || bound_change.new_upper != lp.upper[j]) {
        i_t start = domain.size();
        if (!domain.apply_and_propagate(
              worker->Arow, worker->var_types, settings, bound_change, lp)) {
          DEBUG_SUBMIP(
            "Bound propagation detect infeasibility. var={}, lower={:.4g} -> {:.4g}, "
            "upper={:.4g} -> {:.4g}",
            j,
            bound_change.old_lower,
            bound_change.new_lower,
            bound_change.old_upper,
            bound_change.new_upper);
          domain.backtrack_to_parent(lp);
        }

        new_fixings += calculate_fixrate(worker->var_types, settings, domain, start);
      }
    }

    if (has_stopped) { break; }

    num_fixed += new_fixings;
    fixrate = static_cast<f_t>(num_fixed) / integer_list.size();

    // Even considering the entire integer list, we were unable to fix a single variable in this
    // iteration. Iterate over the fractional variables again and fixing those that closest to
    // an integer solution first in order to reach the fixing threshold.
    if (fixrate == prev_fixrate && prev_stack_size == worker->domain.size()) {
      new_fixings = extend_variable_fixings(worker->Arow,
                                            worker->var_types,
                                            settings,
                                            worker->leaf_problem.objective,
                                            fractional,
                                            current_sol,
                                            worker->root_solution,
                                            round_num_fixed,
                                            bnb_stats.start_time,
                                            worker->leaf_problem,
                                            worker->domain);
      if (new_fixings < 0) {
        DEBUG_SUBMIP("Round {}: stopped fixing (infeasible, halted or time limit reached).", round)
        break;
      }

      num_fixed += new_fixings;
      fixrate = static_cast<f_t>(num_fixed) / integer_list.size();

      if (fixrate >= min_fixrate) {
        has_submip = true;
        break;
      }
    }

    DEBUG_SUBMIP(
      "Round {}: fixed {:.0f} ({:.2f}) -> {:.0f} ({:.2f}) variables ({:+}). round target = {:.4g} "
      "({}), target fixrate = {:.4g} ({:.0f})",
      round,
      prev_fixrate * num_integers,
      prev_fixrate,
      fixrate * num_integers,
      fixrate,
      new_fixings,
      round_target,
      round_num_fixed,
      target_fixrate,
      target_fixrate * num_integers);

    if (fixrate >= target_fixrate) {
      has_submip = true;
      break;
    }

    if (toc(bnb_stats.start_time) >= settings.time_limit) { break; }

    bool continue_dive = solve_lp(bnb_stats, settings, worker, stats, fractional);
    if (!continue_dive) { break; }
    ++round;
  }

  // Accumulate the iterations for sub-MIP so it stops when it reaches the allocated budget.
  if (settings.inside_submip) { bnb_stats.total_simplex_iters += stats.total_simplex_iters; }

  DEBUG_SUBMIP("variables fixed = {:.0f} ({:.2f}), target fixrate = {:.0f} ({:.2f})",
               fixrate * num_integers,
               fixrate,
               target_fixrate * num_integers,
               target_fixrate);

  if (has_submip) {
    solve_submip(worker,
                 bnb_stats,
                 original_problem,
                 submip_manager,
                 fixrate,
                 stats.total_simplex_iters,
                 settings);
  }
}

#ifdef DUAL_SIMPLEX_INSTANTIATE_DOUBLE

template void solve_submip(diving_worker_t<int, double>* worker,
                           branch_and_bound_stats_t<int, double>& bnb_stats,
                           const user_problem_t<int, double>& original_problem,
                           submip_manager_t<int, double>& submip_manager,
                           double fixrate,
                           int64_t simplex_iter_used,
                           simplex_solver_settings_t<int, double> submip_settings);

template void mutation(diving_worker_t<int, double>* worker,
                       branch_and_bound_stats_t<int, double>& bnb_stats,
                       const user_problem_t<int, double>& original_problem,
                       submip_manager_t<int, double>& submip_manager,
                       simplex_solver_settings_t<int, double> submip_settings);

template void rins(diving_worker_t<int, double>* worker,
                   branch_and_bound_stats_t<int, double>& bnb_stats,
                   const user_problem_t<int, double>& original_problem,
                   submip_manager_t<int, double>& submip_manager,
                   simplex_solver_settings_t<int, double> settings);

template void rens(diving_worker_t<int, double>* worker,
                   branch_and_bound_stats_t<int, double>& bnb_stats,
                   const user_problem_t<int, double>& original_problem,
                   submip_manager_t<int, double>& submip_manager,
                   simplex_solver_settings_t<int, double> settings);

#endif

}  // namespace cuopt::mathematical_optimization::mip
