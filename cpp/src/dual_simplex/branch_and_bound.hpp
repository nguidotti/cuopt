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

#pragma once

#include <dual_simplex/diving_queue.hpp>
#include <dual_simplex/initial_basis.hpp>
#include <dual_simplex/mip_node.hpp>
#include <dual_simplex/phase2.hpp>
#include <dual_simplex/pseudo_costs.hpp>
#include <dual_simplex/simplex_solver_settings.hpp>
#include <dual_simplex/solution.hpp>
#include <dual_simplex/types.hpp>
#include <utilities/omp_helpers.hpp>

#include <omp.h>
#include <queue>
#include <vector>

namespace cuopt::linear_programming::dual_simplex {

enum class mip_status_t {
  OPTIMAL    = 0,  // The optimal integer solution was found
  UNBOUNDED  = 1,  // The problem is unbounded
  INFEASIBLE = 2,  // The problem is infeasible
  TIME_LIMIT = 3,  // The solver reached a time limit
  NODE_LIMIT = 4,  // The maximum number of nodes was reached (not implemented)
  NUMERICAL  = 5,  // The solver encountered a numerical error
  UNSET      = 6,  // The status is not set
};

enum class mip_exploration_status_t {
  UNSET      = 0,  // The status is not set
  TIME_LIMIT = 1,  // The solver reached a time limit
  NODE_LIMIT = 2,  // The maximum number of nodes was reached (not implemented)
  NUMERICAL  = 3,  // The solver encountered a numerical error
  RUNNING    = 4,  // The solver is currently exploring the tree
  COMPLETED  = 5,  // The solver finished exploring the tree
};

// Indicate the search and variable selection algorithms used by the thread (See [1]).
//
// [1] T. Achterberg, “Constraint Integer Programming,” PhD, Technischen Universität Berlin,
// Berlin, 2007. doi: 10.14279/depositonce-1634.
enum class thread_type_t {
  EXPLORATION = 0,  // Best-First + Plunging. Pseudocost branching + Martin's criteria.
  DIVING      = 1,
};

template <typename i_t, typename f_t>
class node_presolver_t;

template <typename i_t, typename f_t>
void upper_bound_callback(f_t upper_bound);

template <typename i_t, typename f_t>
class branch_and_bound_t {
 public:
  template <typename T>
  using mip_node_heap_t = std::priority_queue<T, std::vector<T>, node_compare_t<i_t, f_t>>;

  branch_and_bound_t(const user_problem_t<i_t, f_t>& user_problem,
                     const simplex_solver_settings_t<i_t, f_t>& solver_settings);

  // Set an initial guess based on the user_problem. This should be called before solve.
  void set_initial_guess(const std::vector<f_t>& user_guess) { guess_ = user_guess; }

  // Set a solution based on the user problem during the course of the solve
  void set_new_solution(const std::vector<f_t>& solution);

  // Repair a low-quality solution from the heuristics.
  bool repair_solution(const std::vector<f_t>& leaf_edge_norms,
                       const std::vector<f_t>& potential_solution,
                       f_t& repaired_obj,
                       std::vector<f_t>& repaired_solution) const;

  f_t get_upper_bound();
  f_t get_lower_bound();
  i_t get_heap_size();

  // The main entry routine. Returns the solver status and populates solution with the incumbent.
  mip_status_t solve(mip_solution_t<i_t, f_t>& solution);

 private:
  const user_problem_t<i_t, f_t>& original_problem_;
  const simplex_solver_settings_t<i_t, f_t> settings_;

  // Initial guess.
  std::vector<f_t> guess_;

  // LP relaxation
  lp_problem_t<i_t, f_t> original_lp_;
  std::vector<i_t> new_slacks_;
  std::vector<variable_type_t> var_types_;

  // Local lower bounds for each thread
  std::vector<omp_atomic_t<f_t>> local_lower_bounds_;

  // Mutex for upper bound
  omp_mutex_t mutex_upper_;

  // Global variable for upper bound
  f_t upper_bound_;

  // Global variable for incumbent. The incumbent should be updated with the upper bound
  mip_solution_t<i_t, f_t> incumbent_;

  // Structure with the general info of the solver.
  struct stats_t {
    f_t start_time                        = 0.0;
    omp_atomic_t<f_t> total_lp_solve_time = 0.0;
    omp_atomic_t<i_t> nodes_explored      = 0;
    omp_atomic_t<i_t> nodes_unexplored    = 0;
    omp_atomic_t<f_t> total_lp_iters      = 0;

    // This should only be used by the main thread
    f_t last_log                           = 0.0;
    omp_atomic_t<i_t> nodes_since_last_log = 0;
  } stats_;

  // Mutex for repair
  omp_mutex_t mutex_repair_;
  std::vector<std::vector<f_t>> repair_queue_;

  // Variables for the root node in the search tree.
  std::vector<variable_status_t> root_vstatus_;
  f_t root_objective_;
  lp_solution_t<i_t, f_t> root_relax_soln_;
  std::vector<f_t> edge_norms_;

  // Pseudocosts
  pseudo_costs_t<i_t, f_t> pc_;

  // Heap storing the nodes to be explored.
  omp_mutex_t mutex_heap_;
  mip_node_heap_t<mip_node_t<i_t, f_t>*> heap_;

  // Count the number of subtrees that are currently being explored.
  omp_atomic_t<i_t> active_subtrees_;

  // Queue for storing the promising node for performing dives.
  omp_mutex_t mutex_dive_queue_;
  diving_queue_t<i_t, f_t> dive_queue_;
  i_t min_diving_queue_size_;

  // Global status of the solver.
  omp_atomic_t<mip_exploration_status_t> status_;

  // In case, a best-first thread encounters a numerical issue when solving a node,
  // its blocks the progression of the lower bound.
  omp_atomic_t<f_t> lower_bound_ceiling_;

  // Set the final solution.
  mip_status_t set_final_solution(mip_solution_t<i_t, f_t>& solution, f_t lower_bound);

  // Update the incumbent solution with the new feasible solution
  // found during branch and bound.
  void add_feasible_solution(f_t leaf_objective,
                             const std::vector<f_t>& leaf_solution,
                             i_t leaf_depth,
                             thread_type_t thread_type);

  // Repairs low-quality solutions from the heuristics, if it is applicable.
  void repair_heuristic_solutions();

  // Ramp-up phase of the solver, where we greedily expand the tree until
  // there is enough unexplored nodes. This is done recursively using OpenMP tasks.
  void exploration_ramp_up(mip_node_t<i_t, f_t>* node,
                           search_tree_t<i_t, f_t>* search_tree,
                           i_t initial_heap_size);

  // Explore the search tree using the best-first search with plunging strategy.
  void explore_subtree(i_t task_id,
                       mip_node_t<i_t, f_t>* start_node,
                       search_tree_t<i_t, f_t>& search_tree,
                       lp_problem_t<i_t, f_t>& leaf_problem,
                       node_presolver_t<i_t, f_t>& presolver,
                       basis_update_mpf_t<i_t, f_t>& basis_update,
                       std::vector<i_t>& basic_list,
                       std::vector<i_t>& nonbasic_list);

  // Each "main" thread pops a node from the global heap and then performs a plunge
  // (i.e., a shallow dive) into the subtree determined by the node.
  void best_first_thread(i_t id, search_tree_t<i_t, f_t>& search_tree);

  // Each diving thread pops the first node from the dive queue and then performs
  // a deep dive into the subtree determined by the node.
  void diving_thread();

  // Solve the LP relaxation of a leaf node and update the tree.
  node_status_t solve_node(mip_node_t<i_t, f_t>* node_ptr,
                           search_tree_t<i_t, f_t>& search_tree,
                           lp_problem_t<i_t, f_t>& leaf_problem,
                           basis_update_mpf_t<i_t, f_t>& ft,
                           std::vector<i_t>& basic_list,
                           std::vector<i_t>& nonbasic_list,
                           node_presolver_t<i_t, f_t>& presolver,
                           thread_type_t thread_type,
                           bool recompute,
                           const std::vector<f_t>& root_lower,
                           const std::vector<f_t>& root_upper,
                           logger_t& log);

  // Sort the children based on the Martin's criteria.
  std::pair<mip_node_t<i_t, f_t>*, mip_node_t<i_t, f_t>*> child_selection(
    mip_node_t<i_t, f_t>* node_ptr);
};

}  // namespace cuopt::linear_programming::dual_simplex
