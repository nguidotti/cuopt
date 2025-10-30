/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

#include <dual_simplex/phase2.hpp>
#include <dual_simplex/pseudo_costs.hpp>
#include <dual_simplex/simplex_solver_settings.hpp>
#include <dual_simplex/solve.hpp>
#include <dual_simplex/tic_toc.hpp>

#include <omp.h>

namespace cuopt::linear_programming::dual_simplex {

namespace {

template <typename i_t, typename f_t>
void strong_branch_helper(i_t start,
                          i_t end,
                          f_t start_time,
                          const lp_problem_t<i_t, f_t>& original_lp,
                          const simplex_solver_settings_t<i_t, f_t>& settings,
                          const std::vector<variable_type_t>& var_types,
                          const std::vector<i_t>& fractional,
                          f_t root_obj,
                          const std::vector<f_t>& root_soln,
                          const std::vector<variable_status_t>& root_vstatus,
                          const std::vector<f_t>& edge_norms,
                          pseudo_costs_t<i_t, f_t>& pc)
{
  lp_problem_t child_problem = original_lp;

  constexpr bool verbose = false;
  f_t last_log           = tic();
  i_t thread_id          = omp_get_thread_num();
  for (i_t k = start; k < end; ++k) {
    const i_t j = fractional[k];

    for (i_t branch = 0; branch < 2; branch++) {
      // Do the down branch
      if (branch == 0) {
        child_problem.lower[j] = original_lp.lower[j];
        child_problem.upper[j] = std::floor(root_soln[j]);
      } else {
        child_problem.lower[j] = std::ceil(root_soln[j]);
        child_problem.upper[j] = original_lp.upper[j];
      }

      simplex_solver_settings_t<i_t, f_t> child_settings = settings;
      child_settings.set_log(false);
      f_t lp_start_time = tic();
      f_t elapsed_time  = toc(start_time);
      if (elapsed_time > settings.time_limit) { break; }
      child_settings.time_limit      = std::max(0.0, settings.time_limit - elapsed_time);
      child_settings.iteration_limit = 200;
      lp_solution_t<i_t, f_t> solution(original_lp.num_rows, original_lp.num_cols);
      i_t iter                               = 0;
      std::vector<variable_status_t> vstatus = root_vstatus;
      std::vector<f_t> child_edge_norms      = edge_norms;
      dual::status_t status                  = dual_phase2(2,
                                          0,
                                          lp_start_time,
                                          child_problem,
                                          child_settings,
                                          vstatus,
                                          solution,
                                          iter,
                                          child_edge_norms);

      f_t obj = std::numeric_limits<f_t>::infinity();
      if (status == dual::status_t::DUAL_UNBOUNDED) {
        // LP was infeasible
      } else if (status == dual::status_t::OPTIMAL || status == dual::status_t::ITERATION_LIMIT) {
        obj = compute_objective(child_problem, solution.x);
      } else {
        settings.log.debug("Thread id %2d remaining %d variable %d branch %d status %d\n",
                           thread_id,
                           end - 1 - k,
                           j,
                           branch,
                           status);
      }

      if (branch == 0) {
        pc.strong_branch_down[k] = obj - root_obj;
        if (verbose) {
          settings.log.printf("Thread id %2d remaining %d variable %d branch %d obj %e time %.2f\n",
                              thread_id,
                              end - 1 - k,
                              j,
                              branch,
                              obj,
                              toc(start_time));
        }
      } else {
        pc.strong_branch_up[k] = obj - root_obj;
        if (verbose) {
          settings.log.printf(
            "Thread id %2d remaining %d variable %d branch %d obj %e change down %e change up %e "
            "time %.2f\n",
            thread_id,
            end - 1 - k,
            j,
            branch,
            obj,
            pc.strong_branch_down[k],
            pc.strong_branch_up[k],
            toc(start_time));
        }
      }
      if (toc(start_time) > settings.time_limit) { break; }
    }
    if (toc(start_time) > settings.time_limit) { break; }

    const i_t completed = pc.num_strong_branches_completed++;

    if (thread_id == 0 && toc(last_log) > 10) {
      last_log = tic();
      settings.log.printf("%d of %ld strong branches completed in %.1fs\n",
                          completed,
                          fractional.size(),
                          toc(start_time));
    }

    child_problem.lower[j] = original_lp.lower[j];
    child_problem.upper[j] = original_lp.upper[j];

    if (toc(start_time) > settings.time_limit) { break; }
  }
}

}  // namespace

template <typename i_t, typename f_t>
void strong_branching(const lp_problem_t<i_t, f_t>& original_lp,
                      const simplex_solver_settings_t<i_t, f_t>& settings,
                      f_t start_time,
                      const std::vector<variable_type_t>& var_types,
                      const std::vector<f_t> root_soln,
                      const std::vector<i_t>& fractional,
                      f_t root_obj,
                      const std::vector<variable_status_t>& root_vstatus,
                      const std::vector<f_t>& edge_norms,
                      pseudo_costs_t<i_t, f_t>& pc)
{
  pc.resize(original_lp.num_cols);
  pc.strong_branch_down.resize(fractional.size());
  pc.strong_branch_up.resize(fractional.size());
  pc.num_strong_branches_completed = 0;

  settings.log.printf("Strong branching using %d threads and %ld fractional variables\n",
                      settings.num_threads,
                      fractional.size());

#pragma omp parallel num_threads(settings.num_threads)
  {
    i_t n = std::min<i_t>(4 * settings.num_threads, fractional.size());

    // Here we are creating more tasks than the number of threads
    // such that they can be scheduled dynamically to the threads.
#pragma omp for schedule(dynamic, 1)
    for (i_t k = 0; k < n; k++) {
      i_t start = std::floor(k * fractional.size() / n);
      i_t end   = std::floor((k + 1) * fractional.size() / n);

      constexpr bool verbose = false;
      if (verbose) {
        settings.log.printf("Thread id %d task id %d start %d end %d. size %d\n",
                            omp_get_thread_num(),
                            k,
                            start,
                            end,
                            end - start);
      }

      strong_branch_helper(start,
                           end,
                           start_time,
                           original_lp,
                           settings,
                           var_types,
                           fractional,
                           root_obj,
                           root_soln,
                           root_vstatus,
                           edge_norms,
                           pc);
    }
  }

  pc.update_pseudo_costs_from_strong_branching(fractional, root_soln);
}

template <typename i_t, typename f_t>
void pseudo_costs_t<i_t, f_t>::update_pseudo_costs(mip_node_t<i_t, f_t>* node_ptr,
                                                   f_t leaf_objective)
{
  mutex.lock();
  const f_t change_in_obj = leaf_objective - node_ptr->lower_bound;
  const f_t frac          = node_ptr->branch_dir == round_dir_t::DOWN
                              ? node_ptr->fractional_val - std::floor(node_ptr->fractional_val)
                              : std::ceil(node_ptr->fractional_val) - node_ptr->fractional_val;
  if (node_ptr->branch_dir == round_dir_t::DOWN) {
    pseudo_cost_sum_down[node_ptr->branch_var] += change_in_obj / frac;
    pseudo_cost_num_down[node_ptr->branch_var]++;
  } else {
    pseudo_cost_sum_up[node_ptr->branch_var] += change_in_obj / frac;
    pseudo_cost_num_up[node_ptr->branch_var]++;
  }
  mutex.unlock();
}

template <typename i_t, typename f_t>
void pseudo_costs_t<i_t, f_t>::initialized(i_t& num_initialized_down,
                                           i_t& num_initialized_up,
                                           f_t& pseudo_cost_down_avg,
                                           f_t& pseudo_cost_up_avg) const
{
  num_initialized_down = 0;
  num_initialized_up   = 0;
  pseudo_cost_down_avg = 0;
  pseudo_cost_up_avg   = 0;
  const i_t n          = pseudo_cost_sum_down.size();
  for (i_t j = 0; j < n; j++) {
    if (pseudo_cost_num_down[j] > 0) {
      num_initialized_down++;
      pseudo_cost_down_avg += pseudo_cost_sum_down[j] / pseudo_cost_num_down[j];
    }
    if (pseudo_cost_num_up[j] > 0) {
      num_initialized_up++;
      pseudo_cost_up_avg += pseudo_cost_sum_up[j] / pseudo_cost_num_up[j];
    }
  }
  if (num_initialized_down > 0) {
    pseudo_cost_down_avg /= num_initialized_down;
  } else {
    pseudo_cost_down_avg = 1.0;
  }
  if (num_initialized_up > 0) {
    pseudo_cost_up_avg /= num_initialized_up;
  } else {
    pseudo_cost_up_avg = 1.0;
  }
}

template <typename f_t>
round_dir_t martin_criteria(f_t val, f_t root_val)
{
  const f_t down_val  = std::floor(root_val);
  const f_t up_val    = std::ceil(root_val);
  const f_t down_dist = val - down_val;
  const f_t up_dist   = up_val - val;
  constexpr f_t eps   = 1e-6;

  if (down_dist < up_dist + eps) {
    return round_dir_t::DOWN;
  } else {
    return round_dir_t::UP;
  }
}

template <typename i_t, typename f_t>
i_t pseudocost_branching(pseudo_costs_t<i_t, f_t>& pc,
                         const std::vector<i_t>& fractional,
                         const std::vector<f_t>& solution,
                         logger_t& log)
{
  pc.mutex.lock();

  constexpr f_t eps = 1e-6;
  i_t branch_var    = fractional[0];
  f_t max_score     = -1;

  i_t num_initialized_down;
  i_t num_initialized_up;
  f_t pc_down_avg;
  f_t pc_up_avg;
  pc.initialized(num_initialized_down, num_initialized_up, pc_down_avg, pc_up_avg);

  for (auto j : fractional) {
    f_t f_down = solution[j] - std::floor(solution[j]);
    f_t f_up   = std::ceil(solution[j]) - solution[j];

    f_t pc_down = pc.pseudo_cost_num_down[j] != 0
                    ? pc.pseudo_cost_sum_down[j] / pc.pseudo_cost_num_down[j]
                    : pc_down_avg;

    f_t pc_up = pc.pseudo_cost_num_up[j] != 0 ? pc.pseudo_cost_sum_up[j] / pc.pseudo_cost_num_up[j]
                                              : pc_up_avg;

    f_t score = std::max(f_down * pc_down, eps) * std::max(f_up * pc_up, eps);

    if (score > max_score) {
      max_score  = score;
      branch_var = j;
    }
  }

  log.debug("Pseudocost branching: selected %d with val = %e and score = %e\n",
            branch_var,
            solution[branch_var],
            max_score);
  pc.mutex.unlock();

  return branch_var;
}

template <typename i_t, typename f_t>
selected_variable_t<i_t> line_search_diving(const std::vector<i_t>& fractional,
                                            const std::vector<f_t>& solution,
                                            const std::vector<f_t>& root_solution,
                                            logger_t& log)
{
  constexpr f_t eps     = 1e-6;
  i_t branch_var        = fractional[0];
  f_t min_score         = inf;
  round_dir_t round_dir = round_dir_t::DOWN;

  for (auto j : fractional) {
    f_t score       = inf;
    round_dir_t dir = round_dir_t::NONE;

    if (solution[j] < root_solution[j] + eps) {
      f_t f = solution[j] - std::floor(solution[j]);
      f_t d = root_solution[j] - solution[j];
      score = f / d;
      dir   = round_dir_t::DOWN;

    } else if (root_solution[j] < solution[j] + eps) {
      f_t f = std::ceil(solution[j]) - solution[j];
      f_t d = solution[j] - root_solution[j];
      score = f / d;
      dir   = round_dir_t::UP;
    }

    if (min_score > score) {
      min_score  = score;
      branch_var = j;
      round_dir  = dir;
    }
  }

  log.debug("Line search diving: selected %d with val = %e, round dir = %d and score = %e\n",
            branch_var,
            solution[branch_var],
            round_dir,
            min_score);

  return {branch_var, round_dir};
}

template <typename i_t, typename f_t>
selected_variable_t<i_t> pseudocost_diving(pseudo_costs_t<i_t, f_t>& pc,
                                           const std::vector<i_t>& fractional,
                                           const std::vector<f_t>& solution,
                                           const std::vector<f_t>& root_solution,
                                           logger_t& log)
{
  pc.mutex.lock();
  i_t branch_var        = fractional[0];
  f_t max_score         = -1;
  round_dir_t round_dir = round_dir_t::DOWN;
  constexpr f_t eps     = 1e-6;

  i_t num_initialized_down;
  i_t num_initialized_up;
  f_t pseudo_cost_down_avg;
  f_t pseudo_cost_up_avg;
  pc.initialized(
    num_initialized_down, num_initialized_up, pseudo_cost_down_avg, pseudo_cost_up_avg);

  for (auto j : fractional) {
    round_dir_t dir = round_dir_t::NONE;
    f_t f_down      = solution[j] - std::floor(solution[j]);
    f_t f_up        = std::ceil(solution[j]) - solution[j];

    f_t pc_down = pc.pseudo_cost_num_down[j] != 0
                    ? pc.pseudo_cost_sum_down[j] / pc.pseudo_cost_num_down[j]
                    : pseudo_cost_down_avg;

    f_t pc_up = pc.pseudo_cost_num_up[j] != 0 ? pc.pseudo_cost_sum_up[j] / pc.pseudo_cost_num_up[j]
                                              : pseudo_cost_up_avg;

    f_t score_down = std::sqrt(f_up) * (1 + pc_up) / (1 + pc_down);
    f_t score_up   = std::sqrt(f_down) * (1 + pc_down) / (1 + pc_up);
    f_t score      = 0;

    if (solution[j] < root_solution[j] - 0.4) {
      score = score_down;
      dir   = round_dir_t::DOWN;
    } else if (solution[j] > root_solution[j] + 0.4) {
      score = score_up;
      dir   = round_dir_t::UP;
    } else if (f_down < 0.3) {
      score = score_down;
      dir   = round_dir_t::DOWN;
    } else if (f_down > 0.7) {
      score = score_up;
      dir   = round_dir_t::UP;
    } else if (pc_down < pc_up + eps) {
      score = score_down;
      dir   = round_dir_t::DOWN;
    } else {
      score = score_up;
      dir   = round_dir_t::UP;
    }

    if (score > max_score) {
      max_score  = score;
      branch_var = j;
      round_dir  = dir;
    }
  }
  log.debug("Pseudocost diving: selected %d with val = %e, round dir = %d and score = %e\n",
            branch_var,
            solution[branch_var],
            round_dir,
            max_score);

  pc.mutex.unlock();

  return {branch_var, round_dir};
}

template <typename i_t, typename f_t>
selected_variable_t<i_t> guided_diving(pseudo_costs_t<i_t, f_t>& pc,
                                       const std::vector<i_t>& fractional,
                                       const std::vector<f_t>& solution,
                                       const std::vector<f_t>& incumbent,
                                       logger_t& log)
{
  pc.mutex.lock();
  i_t branch_var        = fractional[0];
  f_t max_score         = -1;
  round_dir_t round_dir = round_dir_t::DOWN;
  constexpr f_t eps     = 1e-6;

  i_t num_initialized_down;
  i_t num_initialized_up;
  f_t pseudo_cost_down_avg;
  f_t pseudo_cost_up_avg;
  pc.initialized(
    num_initialized_down, num_initialized_up, pseudo_cost_down_avg, pseudo_cost_up_avg);

  for (auto j : fractional) {
    f_t f_down      = solution[j] - std::floor(solution[j]);
    f_t f_up        = std::ceil(solution[j]) - solution[j];
    f_t down_dist   = std::abs(incumbent[j] - std::floor(solution[j]));
    f_t up_dist     = std::abs(std::ceil(solution[j]) - incumbent[j]);
    round_dir_t dir = down_dist < up_dist + eps ? round_dir_t::DOWN : round_dir_t::UP;

    f_t pc_down = pc.pseudo_cost_num_down[j] != 0
                    ? pc.pseudo_cost_sum_down[j] / pc.pseudo_cost_num_down[j]
                    : pseudo_cost_down_avg;

    f_t pc_up = pc.pseudo_cost_num_up[j] != 0 ? pc.pseudo_cost_sum_up[j] / pc.pseudo_cost_num_up[j]
                                              : pseudo_cost_up_avg;

    f_t score1 = dir == round_dir_t::DOWN ? 5 * pc_down * f_down : 5 * pc_up * f_up;
    f_t score2 = dir == round_dir_t::DOWN ? pc_up * f_up : pc_down * f_down;
    f_t score  = (score1 + score2) / 6;

    if (score > max_score) {
      max_score  = score;
      branch_var = j;
      round_dir  = dir;
    }
  }

  log.debug("Guided diving: selected %d with val = %e, round dir = %d and score = %e\n",
            branch_var,
            solution[branch_var],
            round_dir,
            max_score);

  pc.mutex.unlock();

  return {branch_var, round_dir};
}

template <typename i_t, typename f_t>
std::pair<f_t, f_t> calculate_variable_locks(const lp_problem_t<i_t, f_t>& lp_problem, i_t var_idx)
{
  i_t up_lock   = 0;
  i_t down_lock = 0;
  i_t start     = lp_problem.A.col_start[var_idx];
  i_t end       = lp_problem.A.col_start[var_idx + 1];

  for (i_t k = start; k < end; ++k) {
    f_t nz_val = lp_problem.A.x[k];
    i_t nz_row = lp_problem.A.i[k];

    if (std::isfinite(lp_problem.upper[nz_row]) && std::isfinite(lp_problem.lower[nz_row])) {
      down_lock += 1;
      up_lock += 1;
      continue;
    }

    f_t sign = std::isfinite(lp_problem.upper[nz_row]) ? 1 : -1;

    if (nz_val * sign > 0) {
      up_lock += 1;
    } else {
      down_lock += 1;
    }
  }

  return {up_lock, down_lock};
}

template <typename i_t, typename f_t>
selected_variable_t<i_t> coefficient_diving(const lp_problem_t<i_t, f_t>& lp_problem,
                                            const std::vector<i_t>& fractional,
                                            const std::vector<f_t>& solution,
                                            logger_t& log)
{
  i_t branch_var        = fractional[0];
  f_t min_locks         = INT_MAX;
  round_dir_t round_dir = round_dir_t::DOWN;
  constexpr f_t eps     = 1e-6;

  for (auto j : fractional) {
    f_t f_down                = solution[j] - std::floor(solution[j]);
    f_t f_up                  = std::ceil(solution[j]) - solution[j];
    auto [up_lock, down_lock] = calculate_variable_locks(lp_problem, j);
    f_t locks                 = std::min(up_lock, down_lock);

    if (min_locks > locks) {
      min_locks  = locks;
      branch_var = j;

      if (up_lock < down_lock) {
        round_dir = round_dir_t::UP;
      } else if (up_lock > down_lock) {
        round_dir = round_dir_t::DOWN;
      } else if (f_down < f_up + eps) {
        round_dir = round_dir_t::DOWN;
      } else {
        round_dir = round_dir_t::UP;
      }
    }
  }

  log.debug("Guided diving: selected %d with val = %e, round dir = %d and min locks = %e\n",
            branch_var,
            solution[branch_var],
            round_dir,
            min_locks);

  return {branch_var, round_dir};
}

template <typename i_t, typename f_t>
f_t best_pseudocost_estimate(pseudo_costs_t<i_t, f_t>& pc,
                             const std::vector<i_t>& fractional,
                             const std::vector<f_t>& solution,
                             f_t lower_bound,
                             logger_t& log)
{
  pc.mutex.lock();

  constexpr f_t eps = 1e-6;
  f_t estimate      = lower_bound;

  i_t num_initialized_down;
  i_t num_initialized_up;
  f_t pc_down_avg;
  f_t pc_up_avg;
  pc.initialized(num_initialized_down, num_initialized_up, pc_down_avg, pc_up_avg);

  for (auto j : fractional) {
    f_t f_down = solution[j] - std::floor(solution[j]);
    f_t f_up   = std::ceil(solution[j]) - solution[j];

    f_t pc_down = pc.pseudo_cost_num_down[j] != 0
                    ? pc.pseudo_cost_sum_down[j] / pc.pseudo_cost_num_down[j]
                    : pc_down_avg;

    f_t pc_up = pc.pseudo_cost_num_up[j] != 0 ? pc.pseudo_cost_sum_up[j] / pc.pseudo_cost_num_up[j]
                                              : pc_up_avg;
    estimate += std::min(std::max(pc_down * f_down, eps), std::max(pc_up * f_up, eps));
  }
  pc.mutex.unlock();
  return estimate;
}

template <typename i_t, typename f_t>
void pseudo_costs_t<i_t, f_t>::update_pseudo_costs_from_strong_branching(
  const std::vector<i_t>& fractional, const std::vector<f_t>& root_soln)
{
  for (i_t k = 0; k < fractional.size(); k++) {
    const i_t j = fractional[k];
    for (i_t branch = 0; branch < 2; branch++) {
      const f_t frac = branch == 0 ? root_soln[j] - std::floor(root_soln[j])
                                   : std::ceil(root_soln[j]) - root_soln[j];
      if (branch == 0) {
        f_t change_in_obj = strong_branch_down[k];
        pseudo_cost_sum_down[j] += change_in_obj / frac;
        pseudo_cost_num_down[j]++;
      } else {
        f_t change_in_obj = strong_branch_up[k];
        pseudo_cost_sum_up[j] += change_in_obj / frac;
        pseudo_cost_num_up[j]++;
      }
    }
  }
}

#ifdef DUAL_SIMPLEX_INSTANTIATE_DOUBLE

template class pseudo_costs_t<int, double>;

template void strong_branching<int, double>(const lp_problem_t<int, double>& original_lp,
                                            const simplex_solver_settings_t<int, double>& settings,
                                            double start_time,
                                            const std::vector<variable_type_t>& var_types,
                                            const std::vector<double> root_soln,
                                            const std::vector<int>& fractional,
                                            double root_obj,
                                            const std::vector<variable_status_t>& root_vstatus,
                                            const std::vector<double>& edge_norms,
                                            pseudo_costs_t<int, double>& pc);

template round_dir_t martin_criteria(double val, double root_val);

template int pseudocost_branching(pseudo_costs_t<int, double>& pc,
                                  const std::vector<int>& fractional,
                                  const std::vector<double>& solution,
                                  logger_t& log);

template selected_variable_t<int> line_search_diving(const std::vector<int>& fractional,
                                                     const std::vector<double>& solution,
                                                     const std::vector<double>& root_solution,
                                                     logger_t& log);

template selected_variable_t<int> pseudocost_diving(pseudo_costs_t<int, double>& pc,
                                                    const std::vector<int>& fractional,
                                                    const std::vector<double>& solution,
                                                    const std::vector<double>& root_solution,
                                                    logger_t& log);

template selected_variable_t<int> guided_diving(pseudo_costs_t<int, double>& pc,
                                                const std::vector<int>& fractional,
                                                const std::vector<double>& solution,
                                                const std::vector<double>& incumbent,
                                                logger_t& log);

template selected_variable_t<int> coefficient_diving(const lp_problem_t<int, double>& lp_problem,
                                                     const std::vector<int>& fractional,
                                                     const std::vector<double>& solution,
                                                     logger_t& log);

template double best_pseudocost_estimate(pseudo_costs_t<int, double>& pc,
                                         const std::vector<int>& fractional,
                                         const std::vector<double>& solution,
                                         double lower_bound,
                                         logger_t& log);

#endif

}  // namespace cuopt::linear_programming::dual_simplex
