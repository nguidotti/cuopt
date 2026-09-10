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

  // How many groups a destroy operator releases, and the bounds the controller keeps it within.
  i_t min_radius = 15;
  i_t max_radius = 60;

  // The destroy operators run concurrently, one sub-MIP each, with this many threads per solve.
  i_t threads_per_solve = 8;
};

template <typename i_t, typename f_t>
class core_lns_t {
 public:
  // `csr_matrix_t` has no default constructor, so both start as empty matrices and are rebuilt by
  // `recognize()` once the group count and edge count are known.
  core_lns_t(branch_and_bound_t<i_t, f_t>* branch_and_bound)
    : branch_and_bound_ptr(branch_and_bound), variable_groups_(0, 0, 0), neighbours_groups_(0, 0, 0)
  {
  }

  ~core_lns_t() { stop_and_sync(); }

  bool recognize();

  diving_worker_t<i_t, f_t>* create_submip_worker(
    const simplex::lp_problem_t<i_t, f_t>& lp,
    const csr_matrix_t<i_t, f_t>& Arow,
    const std::vector<simplex::variable_type_t>& var_types,
    const std::vector<f_t>& root_solution,
    const std::vector<f_t>& root_edge_norm,
    pseudo_costs_t<i_t, f_t>& pseudo_costs)
  {
    const i_t id           = workers_.size();
    auto& worker           = workers_.emplace_back(id,
                                         lp,
                                         Arow,
                                         var_types,
                                         branch_and_bound_ptr->settings_,
                                         pseudo_costs,
                                         root_solution,
                                         root_edge_norm,
                                         5000);
    worker.search_strategy = search_strategy_t::CORE_LNS;
    worker.set_active();
    return &worker;
  }

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
      diving_worker_t<i_t, f_t>* worker_ptr = &worker;
#pragma omp taskwait depend(in : *worker_ptr)
    }
  }

 private:
  enum destroy_operator_t : uint8_t {
    DESTROY_LP_GUIDED     = 0,
    DESTROY_NEIGHBOUR     = 1,
    DESTROY_INCUMBENT     = 2,
    DESTROY_RANDOM        = 3,
    NUM_DESTROY_OPERATORS = 4
  };

  // One destroy operator: seeds its share of the level sweep, then destroys and repairs until the
  // solve is halted. Runs as its own task on its own worker.
  void search(diving_worker_t<i_t, f_t>* worker, destroy_operator_t destroy_operator);

  // Release the groups the relaxation is least certain about, and where it disagrees with the
  // incumbent.
  void destroy_lp_guided(diving_worker_t<i_t, f_t>* worker,
                         const std::vector<uint8_t>& best,
                         i_t num_to_release,
                         std::vector<uint8_t>& released);

  // Flood the adjacency from a random seed so the released groups compete for the same downstream
  // variables. Falls back to `destroy_random` when the core carries no edges.
  void destroy_neighbour(diving_worker_t<i_t, f_t>* worker,
                         i_t num_to_release,
                         std::vector<uint8_t>& released);

  // Release groups the incumbent has active. Those are its committed decisions; the inactive ones
  // are the default. Tops up with random groups when the incumbent holds too few to fill the
  // radius.
  void destroy_incumbent(diving_worker_t<i_t, f_t>* worker,
                         const std::vector<uint8_t>& best,
                         i_t num_to_release,
                         std::vector<uint8_t>& released);

  void destroy_random(diving_worker_t<i_t, f_t>* worker,
                      i_t num_to_release,
                      std::vector<uint8_t>& released);

  // Fix every group outside `released`, propagate, and hand the residual to solve_submip. Returns
  // whether the global upper bound improved.
  bool evaluate(diving_worker_t<i_t, f_t>* worker,
                const std::vector<uint8_t>& value,
                const std::vector<uint8_t>* released);

  branch_and_bound_t<i_t, f_t>* branch_and_bound_ptr;
  std::atomic<int> halt;

  std::list<diving_worker_t<i_t, f_t>> workers_;
  submip_stats_t submip_stats_;
  core_lns_params_t<i_t, f_t> params_;

  // The core, one group per row: row k holds the variables of group k, with the group's parity
  // for each of them as the value -- variable `j[p]` takes the group's bit XOR `x[p]`.
  csr_matrix_t<i_t, f_t> variable_groups_;  // m = groups, n = variables

  // Adjacency over the groups: two groups are neighbours when they dominate a common group, i.e.
  // when they compete for the same downstream variables. Symmetric, so every edge appears in both
  // rows.
  csr_matrix_t<i_t, f_t> neighbours_groups_;  // m = n = groups

  // Filled once by run() before the operators launch, read-only thereafter.
  std::vector<f_t> lp_value_;
  std::vector<i_t> ranked_;
  f_t lp_mass_{0};
};

}  // namespace cuopt::mathematical_optimization::mip
