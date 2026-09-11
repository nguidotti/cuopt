/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once
#include <branch_and_bound/worker.hpp>
#include <linear_algebra/sparse_matrix.hpp>

#include <list>
#include <memory>
#include <vector>

namespace cuopt::mathematical_optimization::mip {

template <typename i_t, typename f_t>
class branch_and_bound_t;

template <typename i_t, typename f_t>
struct core_lns_params_t {
  // Minimum size of the group of variables associated to a control variable
  // for it to be considered a "core" variable
  i_t min_group_size = 2;

  // Maximum number of "core" variables (absolute)
  i_t max_core_size = 5000;

  // Maximum number of "core" variables in term of the total number of binary variables
  f_t max_binary_fraction = 0.10;

  // Minimum set of variables covered by the core to turn on this heuristic.
  f_t min_coverage = 0.25;

  // How far above the incumbent's own value the budget slack is forced. Must be > 0 or the
  // incumbent stays feasible and nothing compels the sub-MIP to cross the cap.
  f_t budget_push = 1.0;

  // Wall-clock cap on each seed sub-MIP, so the sweep cannot eat the solve.
  f_t seed_time_limit = 5.0;

  // How many groups a destroy operator releases, and the bounds the controller keeps it within.
  // The controller steps by `radius_step` in both directions, every round, so the drift is
  // (1 - 2p) * radius_step at success rate p -- neutral at p = 0.5 rather than collapsing onto
  // the floor the way an every-round gain against an every-fifth-round loss does.
  i_t radius_step = 2;
  i_t min_radius  = 30;
  i_t max_radius  = 60;

  // The destroy operators run concurrently, one sub-MIP each, with this many threads per solve.
  i_t threads_per_solve = 8;
  i_t num_workers       = 8;
};

template <typename i_t, typename f_t>
class core_lns_t {
 public:
  // `csr_matrix_t` has no default constructor, so both start as empty matrices and are rebuilt by
  // `recognize()` once the group count and edge count are known.
  core_lns_t(branch_and_bound_t<i_t, f_t>* branch_and_bound)
    : branch_and_bound_ptr(branch_and_bound), variable_groups_(0, 0, 0)
  {
  }

  ~core_lns_t() { stop_and_sync(); }

  bool recognize();

  void run(const simplex::lp_problem_t<i_t, f_t>& lp,
           const csr_matrix_t<i_t, f_t>& Arow,
           const std::vector<simplex::variable_type_t>& var_types,
           const std::vector<f_t>& root_solution,
           const std::vector<f_t>& root_edge_norm,
           pseudo_costs_t<i_t, f_t>& pseudo_costs);

  void stop_and_sync()
  {
    halt.store(true, std::memory_order_release);
    for (auto& worker : workers_) {
      diving_worker_t<i_t, f_t>* worker_ptr = worker.get();
#pragma omp taskwait depend(in : *worker_ptr)
    }
  }

  i_t num_workers() { return params_.num_workers; }

 private:
  enum destroy_operator_t : uint8_t {
    DESTROY_LP_GUIDED = 0,
    DESTROY_INCUMBENT = 1,
    DESTROY_RANDOM    = 2,
    DESTROY_CLOSED    = 3,
    DESTROY_BUDGET    = 4
  };

  // One destroy operator: seeds its share of the level sweep, then destroys and repairs until the
  // solve is halted. Runs as its own task on its own worker.
  void search(diving_worker_t<i_t, f_t>* worker, destroy_operator_t destroy_operator);
  void create_workers(i_t num_workers,
                      const simplex::lp_problem_t<i_t, f_t>& lp,
                      const csr_matrix_t<i_t, f_t>& Arow,
                      const std::vector<simplex::variable_type_t>& var_types,
                      const std::vector<f_t>& root_solution,
                      const std::vector<f_t>& root_edge_norm,
                      pseudo_costs_t<i_t, f_t>& pseudo_costs);

  // Release the groups the relaxation is least certain about, and where it disagrees with the
  // incumbent.
  void destroy_lp_guided(diving_worker_t<i_t, f_t>* worker,
                         const std::vector<uint8_t>& best,
                         const std::vector<f_t>& volatility,
                         i_t num_to_release,
                         std::vector<uint8_t>& released);

  // Release groups the incumbent has active. Those are its committed decisions; the inactive ones
  // are the default. Tops up with random groups when the incumbent holds too few to fill the
  // radius.
  void destroy_incumbent(diving_worker_t<i_t, f_t>* worker,
                         const std::vector<uint8_t>& best,
                         const std::vector<f_t>& volatility,
                         i_t num_to_release,
                         std::vector<uint8_t>& released);

  void destroy_random(diving_worker_t<i_t, f_t>* worker,
                      i_t num_to_release,
                      std::vector<uint8_t>& released);

  // The mirror of destroy_incumbent: release only groups the incumbent has *closed*. Everything
  // already open stays fixed at 1, so the residual solve cannot claw back the cap charge by
  // closing a control elsewhere -- its only move is to open more, or to leave them shut. With the
  // cutoff in force that makes the expensive step across the budget the one thing on offer.
  // The soft budget row's slack: a positively-priced continuous singleton the relaxation is
  // willing to buy, so the row is a cap that may be exceeded for a price rather than a hard limit.
  // -1 when the model carries no such row.
  void find_budget_slack(const std::vector<f_t>& root_solution);
  i_t budget_slack_col_{-1};

  void destroy_closed(diving_worker_t<i_t, f_t>* worker,
                      const std::vector<uint8_t>& best,
                      const std::vector<f_t>& volatility,
                      i_t num_to_release,
                      std::vector<uint8_t>& released);

  // Outcome of one destroy/repair round. INFEASIBLE and NO_IMPROVEMENT both mean "no better
  // solution", but they call for opposite responses from the radius controller, so they are kept
  // apart: releasing more groups fixes fewer of them and is strictly more permissive.
  enum class evaluate_result_t : uint8_t { IMPROVED, NO_IMPROVEMENT, INFEASIBLE };

  // Fix every group outside `released`, propagate, and hand the residual to solve_submip.
  evaluate_result_t evaluate(diving_worker_t<i_t, f_t>* worker,
                             const std::vector<uint8_t>& value,
                             const std::vector<uint8_t>* released,
                             f_t forced_slack_lb = -1);

  // Long-term memory over the core: how often releasing a group led to the incumbent actually
  // giving it a different value. A group the residual solve keeps putting back the same way is
  // settled, and spending the radius on it buys nothing. Shared by every worker, so the three
  // streams pool their evidence.
  void snapshot_volatility(std::vector<f_t>& out);
  void record_release(const std::vector<uint8_t>& released,
                      const std::vector<uint8_t>& before,
                      const std::vector<uint8_t>& after);

  omp_mutex_t mutex_memory_;
  std::vector<i_t> released_count_;
  std::vector<i_t> changed_count_;

  branch_and_bound_t<i_t, f_t>* branch_and_bound_ptr;
  std::atomic<int> halt{false};

  std::vector<std::unique_ptr<diving_worker_t<i_t, f_t>>> workers_;

  submip_stats_t submip_stats_;
  core_lns_params_t<i_t, f_t> params_;

  // The core, one group per row: row k holds the variables of group k, with the group's parity
  // for each of them as the value -- variable `j[p]` takes the group's bit XOR `x[p]`.
  csr_matrix_t<i_t, f_t> variable_groups_;  // m = groups, n = variables

  // Filled once by run() before the operators launch, read-only thereafter.
  std::vector<f_t> lp_value_;
  std::vector<i_t> ranked_;
  f_t lp_mass_{0};
};

}  // namespace cuopt::mathematical_optimization::mip
