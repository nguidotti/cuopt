/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "lns.hpp"
#include <branch_and_bound/branch_and_bound.hpp>

#include <utilities/logger.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <format>
#include <numeric>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#define SUBMIP_VERBOSE true
#if SUBMIP_VERBOSE
#define DEBUG_SUBMIP(fmt, ...) settings_.log.print_format(fmt, __VA_ARGS__);
#else
#define DEBUG_SUBMIP(fmt, ...)
#endif

namespace cuopt::mathematical_optimization::mip {

namespace {

// Union-find over binary columns carrying a parity: value(a) == value(root(a)) ^ parity(a).
// The signed form is required because complement pairs x_i + x_j = 1 merge two columns of
// opposite polarity, which a plain union-find cannot represent.
template <typename i_t>
struct signed_union_find_t {
  std::vector<i_t> parent;
  std::vector<uint8_t> parity;

  explicit signed_union_find_t(i_t n) : parent(n), parity(n, 0)
  {
    std::iota(parent.begin(), parent.end(), 0);
  }

  std::pair<i_t, int> find(i_t a)
  {
    int flip = 0;
    i_t root = a;
    while (parent[root] != root) {
      flip ^= parity[root];
      root = parent[root];
    }
    int walk = flip;
    while (parent[a] != a) {
      const i_t next    = parent[a];
      const int carried = parity[a];
      parent[a]         = root;
      parity[a]         = walk;
      walk ^= carried;
      a = next;
    }
    return {root, flip};
  }

  // Assert value(a) == value(b) ^ relation.
  void merge(i_t a, i_t b, int relation)
  {
    const auto [ra, fa] = find(a);
    const auto [rb, fb] = find(b);
    if (ra == rb) return;
    parent[ra] = rb;
    parity[ra] = fa ^ fb ^ relation;
  }
};

// For a two-variable row, the value the partner is forced to when the source is set, or -1 when
// both partner values remain feasible.
template <typename f_t>
int forced_partner(f_t source_coeff, int setting, f_t target_coeff, char sense, f_t rhs, f_t tol)
{
  if (sense == 'G') {
    source_coeff = -source_coeff;
    target_coeff = -target_coeff;
    rhs          = -rhs;
  } else if (sense != 'L') {
    return -1;
  }
  if (std::abs(target_coeff) <= tol) return -1;

  const f_t slack    = rhs - source_coeff * setting;
  const bool zero_ok = 0 <= slack + tol;
  const bool one_ok  = target_coeff <= slack + tol;
  if (zero_ok && one_ok) return -1;
  if (zero_ok) return 0;
  if (one_ok) return 1;
  return -1;  // the setting itself is infeasible; propagation will catch it
}

}  // namespace

template <typename i_t, typename f_t>
bool core_lns_t<i_t, f_t>::recognize()
{
  constexpr f_t tol = 1e-9;
  // A target dominated by nearly every group carries no locality, so it contributes no edges.
  constexpr size_t max_shared_owners = 64;

  const auto& problem = branch_and_bound_ptr->original_problem_;
  const i_t num_rows  = problem.num_rows;
  const i_t num_cols  = problem.num_cols;

  std::vector<uint8_t> is_binary(num_cols, 0);
  i_t num_binaries = 0;
  for (i_t j = 0; j < num_cols; ++j) {
    if (problem.var_types[j] == simplex::variable_type_t::BINARY) {
      is_binary[j] = 1;
      ++num_binaries;
    }
  }
  if (num_binaries == 0) return false;

  // Collect the rows holding exactly two nonzeros, both binary. `A` is column major, so this
  // takes one counting pass and one filling pass rather than a transpose.
  std::vector<i_t> row_length(num_rows, 0);
  for (i_t j = 0; j < num_cols; ++j) {
    for (i_t p = problem.A.col_start[j]; p < problem.A.col_start[j + 1]; ++p) {
      ++row_length[problem.A.i[p]];
    }
  }

  std::vector<i_t> col_a(num_rows, -1), col_b(num_rows, -1);
  std::vector<f_t> val_a(num_rows, 0), val_b(num_rows, 0);
  for (i_t j = 0; j < num_cols; ++j) {
    for (i_t p = problem.A.col_start[j]; p < problem.A.col_start[j + 1]; ++p) {
      const i_t row = problem.A.i[p];
      if (row_length[row] != 2) continue;
      if (col_a[row] < 0) {
        col_a[row] = j;
        val_a[row] = problem.A.x[p];
      } else {
        col_b[row] = j;
        val_b[row] = problem.A.x[p];
      }
    }
  }

  // Contract the binaries into groups under the two-variable equalities.
  signed_union_find_t<i_t> groups(num_cols);
  std::vector<i_t> implication_rows;
  for (i_t row = 0; row < num_rows; ++row) {
    const i_t a = col_a[row], b = col_b[row];
    if (a < 0 || b < 0 || a == b) continue;
    if (!is_binary[a] || !is_binary[b]) continue;

    const f_t ca = val_a[row], cb = val_b[row], rhs = problem.rhs[row];
    if (problem.row_sense[row] == 'E') {
      if (std::abs(ca + cb) <= tol && std::abs(rhs) <= tol) {
        groups.merge(a, b, 0);  // alias      x_a - x_b = 0
      } else if (std::abs(ca - cb) <= tol && std::abs(rhs - ca) <= tol && std::abs(ca) > tol) {
        groups.merge(a, b, 1);  // complement x_a + x_b = 1
      }
    } else {
      implication_rows.push_back(row);
    }
  }

  // Literal implication digraph over the groups. A literal is (root, value), keyed as
  // 2 * root + value. Contrapositives are essential and are what the second (source, target)
  // orientation supplies: a control normally dominates *downward* -- switching it off
  // extinguishes its block -- so forward arcs alone score every leaf and no control at all.
  std::unordered_map<int64_t, std::vector<int64_t>> implies;
  for (const i_t row : implication_rows) {
    const i_t a = col_a[row], b = col_b[row];
    const f_t ca = val_a[row], cb = val_b[row];
    const char sense = problem.row_sense[row];
    const f_t rhs    = problem.rhs[row];

    const auto [ra, fa] = groups.find(a);
    const auto [rb, fb] = groups.find(b);
    if (ra == rb) continue;

    for (int orientation = 0; orientation < 2; ++orientation) {
      const f_t source_coeff = orientation == 0 ? ca : cb;
      const f_t target_coeff = orientation == 0 ? cb : ca;
      const i_t source_root  = orientation == 0 ? ra : rb;
      const int source_par   = orientation == 0 ? fa : fb;
      const i_t target_root  = orientation == 0 ? rb : ra;
      const int target_par   = orientation == 0 ? fb : fa;

      for (int setting = 0; setting < 2; ++setting) {
        const int forced = forced_partner(source_coeff, setting, target_coeff, sense, rhs, tol);
        if (forced < 0) continue;
        const int64_t key    = 2 * int64_t{source_root} + (setting ^ source_par);
        const int64_t target = 2 * int64_t{target_root} + (forced ^ target_par);
        implies[key].push_back(target);
      }
    }
  }

  for (auto& [key, targets] : implies) {
    std::sort(targets.begin(), targets.end());
    targets.erase(std::unique(targets.begin(), targets.end()), targets.end());
  }

  // The group size of a control is how many other groups its better direction forces.
  std::unordered_map<i_t, i_t> group_size;
  for (const auto& [key, targets] : implies) {
    const i_t root = key / 2;
    i_t& best      = group_size[root];
    best           = std::max<i_t>(best, targets.size());
  }

  std::vector<i_t> candidates;
  for (const auto& [root, size] : group_size) {
    if (size >= params_.min_group_size) candidates.push_back(root);
  }
  if (candidates.empty()) return false;

  std::sort(candidates.begin(), candidates.end(), [&](i_t a, i_t b) {
    return group_size[a] != group_size[b] ? group_size[a] > group_size[b] : a < b;
  });

  const i_t limit =
    std::min<i_t>(params_.max_core_size, params_.max_binary_fraction * num_binaries);
  if (limit <= 0) return false;
  i_t num_groups = candidates.size();
  if (num_groups > limit) {
    candidates.resize(limit);
    num_groups = limit;
  }

  std::unordered_map<i_t, i_t> root_to_group;
  for (i_t k = 0; k < num_groups; ++k) {
    root_to_group[candidates[k]] = k;
  }

  // Every binary belongs to some group; only those whose group was selected reach the core.
  std::unordered_map<i_t, i_t> members_per_root;
  std::vector<i_t> column_group(num_cols, -1);
  std::vector<uint8_t> column_parity(num_cols, 0);
  for (i_t j = 0; j < num_cols; ++j) {
    if (!is_binary[j]) continue;
    const auto [root, parity] = groups.find(j);
    ++members_per_root[root];
    const auto it = root_to_group.find(root);
    if (it == root_to_group.end()) continue;
    column_group[j]  = it->second;
    column_parity[j] = parity;
  }

  // Coverage: how much of the model the core settles in one propagation step. Without this the
  // detector accepts a handful of controls that settle nothing, and every residual solve is the
  // original problem again at full cost.
  std::unordered_set<i_t> determined(candidates.begin(), candidates.end());
  for (const i_t root : candidates) {
    for (int value = 0; value < 2; ++value) {
      const auto it = implies.find(2 * int64_t{root} + value);
      if (it == implies.end()) continue;
      for (const int64_t target : it->second) {
        determined.insert(target / 2);
      }
    }
  }

  i_t covered = 0;
  for (const i_t root : determined) {
    const auto it = members_per_root.find(root);
    if (it != members_per_root.end()) covered += it->second;
  }

  const f_t binaries = num_binaries;
  if (covered / binaries < params_.min_coverage) return false;

  // ---- variable_groups_: row k holds the columns of group k, the value carrying the parity ----
  std::vector<i_t> counts(num_groups, 0);
  i_t num_members = 0;
  for (i_t j = 0; j < num_cols; ++j) {
    if (column_group[j] < 0) continue;
    ++counts[column_group[j]];
    ++num_members;
  }

  variable_groups_              = csr_matrix_t<i_t, f_t>(num_groups, num_cols, num_members);
  variable_groups_.row_start[0] = 0;
  for (i_t k = 0; k < num_groups; ++k) {
    variable_groups_.row_start[k + 1] = variable_groups_.row_start[k] + counts[k];
  }
  std::vector<i_t> cursor(variable_groups_.row_start.begin(), variable_groups_.row_start.end() - 1);
  // Sweeping the columns in order leaves every row sorted by column index.
  for (i_t j = 0; j < num_cols; ++j) {
    const i_t k = column_group[j];
    if (k < 0) continue;
    const i_t p           = cursor[k]++;
    variable_groups_.j[p] = j;
    variable_groups_.x[p] = column_parity[j];
  }

  // ---- neighbours_groups_: two groups are adjacent when they dominate a common group ----
  std::unordered_map<i_t, std::vector<i_t>> owners;
  for (const i_t root : candidates) {
    for (int value = 0; value < 2; ++value) {
      const auto it = implies.find(2 * int64_t{root} + value);
      if (it == implies.end()) continue;
      for (const int64_t target : it->second) {
        owners[target / 2].push_back(root_to_group[root]);
      }
    }
  }

  std::vector<std::pair<i_t, i_t>> edges;
  for (auto& [target, list] : owners) {
    std::sort(list.begin(), list.end());
    list.erase(std::unique(list.begin(), list.end()), list.end());
    if (list.size() < 2 || list.size() > max_shared_owners) continue;
    for (const i_t a : list) {
      for (const i_t b : list) {
        if (a != b) edges.emplace_back(a, b);
      }
    }
  }
  std::sort(edges.begin(), edges.end());
  edges.erase(std::unique(edges.begin(), edges.end()), edges.end());

  const i_t num_edges = edges.size();
  neighbours_groups_  = csr_matrix_t<i_t, f_t>(num_groups, num_groups, num_edges);
  std::fill(neighbours_groups_.row_start.begin(), neighbours_groups_.row_start.end(), 0);
  for (const auto& [a, b] : edges) {
    ++neighbours_groups_.row_start[a + 1];
  }
  for (i_t k = 0; k < num_groups; ++k) {
    neighbours_groups_.row_start[k + 1] += neighbours_groups_.row_start[k];
  }
  // `edges` is sorted, so appending in order leaves every row sorted by neighbour index.
  i_t position = 0;
  for (const auto& [a, b] : edges) {
    neighbours_groups_.j[position] = b;
    neighbours_groups_.x[position] = 1;
    ++position;
  }

  CUOPT_LOG_INFO("%s",
                 std::format("Decision core: {} groups over {} variables, {} covered ({:.1f}%), "
                             "{} adjacency edges",
                             num_groups,
                             num_members,
                             covered,
                             100.0 * covered / binaries,
                             num_edges)
                   .c_str());
  return true;
}

template <typename i_t, typename f_t>
bool core_lns_t<i_t, f_t>::evaluate(diving_worker_t<i_t, f_t>* worker,
                                    const std::vector<uint8_t>& value,
                                    const std::vector<uint8_t>* released)
{
  simplex::simplex_solver_settings_t<i_t, f_t> settings = branch_and_bound_ptr->settings_;
  settings.print_presolve_stats                         = false;
  settings.num_threads                                  = params_.threads_per_solve;
  settings.concurrent_halt                              = &halt;
  settings.reliability_branching                        = 0;
  settings.clique_cuts                                  = 0;
  settings.zero_half_cuts                               = 0;
  settings.inside_submip                                = 1;
  settings.inside_root_node                             = 1;
  settings.max_cut_passes                               = 5;
  settings.strong_branching_simplex_iteration_limit     = 50;
  settings.submip_settings.level                        = 1;
  settings.benchmark_info_ptr                           = nullptr;
  settings.log.log                                      = SUBMIP_VERBOSE;
  settings.log.log_prefix = std::format("[CORE LNS {}] ", worker->worker_id);

  worker->leaf_problem.lower = worker->start_lower;
  worker->leaf_problem.upper = worker->start_upper;
  std::fill(worker->bounds_changed.begin(), worker->bounds_changed.end(), false);

  i_t num_fixed = 0;
  for (i_t k = 0; k < variable_groups_.m; ++k) {
    if (released != nullptr && (*released)[k]) continue;
    for (i_t p = variable_groups_.row_start[k]; p < variable_groups_.row_start[k + 1]; ++p) {
      const i_t col    = variable_groups_.j[p];
      const int parity = variable_groups_.x[p] > 0.5 ? 1 : 0;
      const f_t fixed  = value[k] ^ parity;
      if (fixed < worker->start_lower[col] - settings.fixed_tol) continue;
      if (fixed > worker->start_upper[col] + settings.fixed_tol) continue;
      worker->leaf_problem.lower[col] = fixed;
      worker->leaf_problem.upper[col] = fixed;
      worker->bounds_changed[col]     = true;
      ++num_fixed;
    }
  }
  if (num_fixed == 0) return false;

  i_t total = 0, num_integer_fixed = 0;
  for (i_t j = 0; j < worker->var_types.size(); ++j) {
    if (worker->var_types[j] == simplex::variable_type_t::CONTINUOUS) continue;
    ++total;
    num_integer_fixed +=
      std::abs(worker->leaf_problem.lower[j] - worker->leaf_problem.upper[j]) <= settings.fixed_tol;
  }
  const f_t denominator = total;
  const f_t fixrate     = total > 0 ? num_integer_fixed / denominator : f_t{0};

  bool is_feasible = worker->node_presolver.bounds_strengthening(
    settings, worker->bounds_changed, worker->leaf_problem.lower, worker->leaf_problem.upper);
  if (!is_feasible) {
    submip_stats_.save_infeasible(fixrate);
    return false;
  }

  const f_t before = branch_and_bound_ptr->upper_bound_.load();
  ++submip_stats_.total_calls;

  branch_and_bound_ptr->solve_submip(worker, submip_stats_, fixrate, 0, settings);
  return branch_and_bound_ptr->upper_bound_.load() < before - settings.zero_tol;
}

template <typename i_t, typename f_t>
void core_lns_t<i_t, f_t>::destroy_lp_guided(diving_worker_t<i_t, f_t>* worker,
                                             const std::vector<uint8_t>& best,
                                             i_t num_to_release,
                                             std::vector<uint8_t>& released)
{
  const i_t num_groups = variable_groups_.m;
  std::vector<std::pair<f_t, i_t>> score(num_groups);
  for (i_t k = 0; k < num_groups; ++k) {
    const f_t fractionality = std::min(lp_value_[k], 1.0 - lp_value_[k]);
    const f_t disagreement  = std::abs(lp_value_[k] - best[k]);
    score[k] = {-(fractionality + disagreement + 1e-3 * worker->rng.next_double()), k};
  }
  std::partial_sort(score.begin(), score.begin() + num_to_release, score.end());
  for (i_t r = 0; r < num_to_release; ++r) {
    released[score[r].second] = 1;
  }
}

template <typename i_t, typename f_t>
void core_lns_t<i_t, f_t>::destroy_neighbour(diving_worker_t<i_t, f_t>* worker,
                                             i_t num_to_release,
                                             std::vector<uint8_t>& released)
{
  if (neighbours_groups_.nz_max == 0) {
    destroy_random(worker, num_to_release, released);
    return;
  }

  const i_t num_groups = variable_groups_.m;
  std::vector<i_t> stack;
  i_t chosen = 0;
  while (chosen < num_to_release) {
    stack.push_back(worker->rng.next_double() * num_groups);
    while (!stack.empty() && chosen < num_to_release) {
      const i_t node = stack.back();
      stack.pop_back();
      if (node >= num_groups || released[node]) continue;
      released[node] = 1;
      ++chosen;
      for (i_t p = neighbours_groups_.row_start[node]; p < neighbours_groups_.row_start[node + 1];
           ++p) {
        const i_t other = neighbours_groups_.j[p];
        if (!released[other]) stack.push_back(other);
      }
    }
  }
}

template <typename i_t, typename f_t>
void core_lns_t<i_t, f_t>::destroy_random(diving_worker_t<i_t, f_t>* worker,
                                          i_t num_to_release,
                                          std::vector<uint8_t>& released)
{
  const i_t num_groups = variable_groups_.m;
  i_t chosen           = 0;
  while (chosen < num_to_release) {
    const i_t k = worker->rng.next_double() * num_groups;
    if (k < num_groups && !released[k]) {
      released[k] = 1;
      ++chosen;
    }
  }
}

template <typename i_t, typename f_t>
void core_lns_t<i_t, f_t>::search(diving_worker_t<i_t, f_t>* worker,
                                  destroy_operator_t destroy_operator)
{
  const i_t num_groups = variable_groups_.m;
  std::vector<uint8_t> best(num_groups, 0);
  std::vector<uint8_t> assignment(num_groups, 0);
  std::vector<uint8_t> released(num_groups, 0);

  // The relaxation ranks the core well but understates how many groups should be active, so the
  // level is swept rather than trusted. The levels are dealt round-robin so the seed is parallel.
  constexpr f_t levels[]   = {0.5, 0.75, 1.0, 1.25, 1.5};
  constexpr i_t num_levels = 5;

  for (i_t idx = worker->worker_id; idx < num_levels; idx += NUM_DESTROY_OPERATORS) {
    if (!branch_and_bound_ptr->is_running() || halt.load(std::memory_order::acquire)) { return; }
    const i_t level = std::clamp<i_t>(std::lround(levels[idx] * lp_mass_), 1, num_groups);
    std::fill(assignment.begin(), assignment.end(), 0);
    for (i_t r = 0; r < level; ++r) {
      assignment[ranked_[r]] = 1;
    }
    evaluate(worker, assignment, nullptr);
  }

  const i_t max_radius = std::min<i_t>(params_.max_radius, num_groups);
  i_t radius           = std::min<i_t>(params_.min_radius, max_radius);
  i_t failures         = 0;

  while (branch_and_bound_ptr->is_running() && !halt.load(std::memory_order::acquire)) {
    // Picks up improvements from the other operators and from every other heuristic.
    branch_and_bound_ptr->mutex_upper_.lock();
    worker->current_incumbent = branch_and_bound_ptr->incumbent_.has_incumbent
                                  ? branch_and_bound_ptr->incumbent_.x
                                  : std::vector<f_t>{};
    branch_and_bound_ptr->mutex_upper_.unlock();

    for (i_t k = 0; k < variable_groups_.m; ++k) {
      const i_t p      = variable_groups_.row_start[k];
      const i_t col    = variable_groups_.j[p];
      const int parity = variable_groups_.x[p] > 0.5 ? 1 : 0;
      if (col < worker->current_incumbent.size()) {
        best[k] = (worker->current_incumbent[col] > 0.5 ? 1 : 0) ^ parity;
      }
    }

    std::fill(released.begin(), released.end(), 0);
    const i_t num_to_release = std::min(radius, num_groups);

    switch (destroy_operator) {
      case DESTROY_LP_GUIDED: destroy_lp_guided(worker, best, num_to_release, released); break;
      case DESTROY_NEIGHBOUR: destroy_neighbour(worker, num_to_release, released); break;
      case DESTROY_RANDOM: destroy_random(worker, num_to_release, released); break;
      case NUM_DESTROY_OPERATORS: break;
    }

    assignment = best;
    if (evaluate(worker, assignment, &released)) {
      failures = 0;
      radius   = std::min(max_radius, radius + 2);
    } else if (++failures % 5 == 0) {
      radius = std::min(max_radius, radius + 5);
    }
  }
}

template <typename i_t, typename f_t>
void core_lns_t<i_t, f_t>::run(const simplex::lp_problem_t<i_t, f_t>& lp,
                               const csr_matrix_t<i_t, f_t>& Arow,
                               const std::vector<simplex::variable_type_t>& var_types,
                               const std::vector<f_t>& root_solution,
                               const std::vector<f_t>& root_edge_norm,
                               pseudo_costs_t<i_t, f_t>& pseudo_costs)
{
  const i_t num_groups = variable_groups_.m;
  if (num_groups == 0) return;

  lp_value_.assign(num_groups, 0);
  for (i_t k = 0; k < num_groups; ++k) {
    const i_t p      = variable_groups_.row_start[k];
    const i_t col    = variable_groups_.j[p];
    const int parity = variable_groups_.x[p] > 0.5 ? 1 : 0;
    const f_t x      = col < root_solution.size() ? root_solution[col] : f_t{0};
    lp_value_[k]     = parity ? 1.0 - x : x;
  }

  ranked_.resize(num_groups);
  std::iota(ranked_.begin(), ranked_.end(), 0);
  std::sort(
    ranked_.begin(), ranked_.end(), [&](i_t a, i_t b) { return lp_value_[a] > lp_value_[b]; });
  lp_mass_ = std::accumulate(lp_value_.begin(), lp_value_.end(), f_t{0});

  CUOPT_LOG_INFO("%s",
                 std::format("Core LNS: {} groups, {} operators x {} threads, LP mass {:.1f}",
                             num_groups,
                             (i_t)NUM_DESTROY_OPERATORS,
                             params_.threads_per_solve,
                             lp_mass_)
                   .c_str());

  for (auto op : {DESTROY_LP_GUIDED, DESTROY_NEIGHBOUR, DESTROY_RANDOM}) {
    diving_worker_t<i_t, f_t>* worker =
      create_submip_worker(lp, Arow, var_types, root_solution, root_edge_norm, pseudo_costs);
#pragma omp task firstprivate(worker, op) depend(out : *worker)
    {
      search(worker, op);
      worker->set_inactive();
    }
  }
}

#ifdef DUAL_SIMPLEX_INSTANTIATE_DOUBLE
template class core_lns_t<int, double>;
#endif

}  // namespace cuopt::mathematical_optimization::mip
