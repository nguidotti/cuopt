/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <dual_simplex/user_problem.hpp>

#include <atomic>
#include <cstdint>
#include <limits>
#include <vector>

namespace cuopt::mathematical_optimization::mip {

// Prefix length into the column order. The detector caps the core column count well below the
// sentinel, so a byte is enough and it halves the cache pressure of the hottest table lookup.
using markshare_prefix_t = uint8_t;
constexpr markshare_prefix_t markshare_unreachable = 0xFF;

// Normalized coefficients, right hand sides and residuals. All search arithmetic is exact
// integer arithmetic in this type; nothing in the proof path touches floating point.
using markshare_coeff_t = int32_t;

/**
 * @brief Subset sum reachability table for a single row.
 *
 * f[u] is the smallest prefix length p such that some subset of coefficients[0, p) sums to
 * exactly u, for every u in [0, capacity]. f[0] is 0 (the empty subset) and unreachable residuals
 * are markshare_unreachable.
 *
 * Requires every coefficient to be non-negative and coefficients.size() < markshare_unreachable.
 */
void markshare_build_row_table(const std::vector<markshare_coeff_t>& coefficients,
                               markshare_coeff_t capacity,
                               std::vector<markshare_prefix_t>& f);

/**
 * @brief Joint subset sum reachability table over two rows.
 *
 * f[u0 * (capacity1 + 1) + u1] is the smallest prefix length p such that a *single* subset of
 * [0, p) sums to u0 in the first row and simultaneously to u1 in the second. This is far stronger
 * than the conjunction of the two single row tables, and it is what makes the search tractable.
 *
 * Both coefficient vectors must be non-negative and of equal length.
 */
void markshare_build_joint_table(const std::vector<markshare_coeff_t>& coefficients0,
                                 const std::vector<markshare_coeff_t>& coefficients1,
                                 markshare_coeff_t capacity0,
                                 markshare_coeff_t capacity1,
                                 std::vector<markshare_prefix_t>& f);

enum class markshare_status_t {
  NOT_APPLICABLE = 0,  // detection rejected the model, nothing was searched
  OPTIMAL        = 1,  // solution found and every lower objective level was exhausted
  FEASIBLE       = 2,  // solution found, but some lower level was left incomplete
  BOUND_ONLY     = 3,  // no solution, but levels below levels_exhausted are ruled out
  ABORTED        = 4   // the budget ran out before anything was proven
};

const char* markshare_status_to_string(markshare_status_t status);

template <typename i_t, typename f_t>
struct markshare_result_t {
  markshare_status_t status{markshare_status_t::NOT_APPLICABLE};

  // Solver space assignment, sized user_problem.num_cols. Empty unless a solution was found.
  std::vector<f_t> solution;
  // Raw c^T x of `solution`, i.e. the same objective space as branch_and_bound_t's internal
  // bounds. Converted to user space only at reporting boundaries.
  f_t objective{std::numeric_limits<f_t>::infinity()};
  // Proven: every feasible point has objective >= this, in solver space. Informational only --
  // cuOpt has no dual bound injection point, so this is logged and never pushed into B&B.
  f_t proven_lower_bound{-std::numeric_limits<f_t>::infinity()};

  i_t levels_exhausted{0};  // first objective level that was not fully ruled out
  int64_t nodes{0};
  int64_t targets{0};
  double search_time{0.0};

  bool has_solution() const { return !solution.empty(); }
  bool proved_optimal() const { return status == markshare_status_t::OPTIMAL; }
};

template <typename i_t, typename f_t>
struct markshare_settings_t {
  // Structural caps. These are the cheap rejects, so keep them tight: they are what makes the
  // recognizer free on every model that is not a market split.
  i_t max_rows{8};
  i_t max_core_cols{64};
  markshare_coeff_t max_normalized_rhs{1 << 20};
  size_t max_table_bytes{size_t{64} << 20};

  // Budget. The search consumes whatever it is given: if detection fires, branch and bound
  // provably cannot solve the instance, so there is nothing to reserve time for.
  double time_limit{std::numeric_limits<double>::infinity()};
  int64_t node_limit{std::numeric_limits<int64_t>::max()};
  i_t max_level{64};

  // Meet-in-the-middle terminal. Instead of descending the last `hash_depth` levels, the search
  // asks one exact m-dimensional question about the columns below it. Sized from the problem:
  // small models finish in milliseconds without it and would only pay the build cost.
  i_t hash_min_cols{40};  // enable the terminal only when the core column count exceeds this
  // Depth balances two costs that move in opposite directions: building the table costs about
  // 2^depth, and the search above it costs about 2^(n - depth). Equalising them puts the optimum
  // near n/2, which is what measurement shows (n=50 wants 26, n=45 wants 24; forcing n=50 up to
  // 29 cut nodes 7.9x but ran 5x slower because the build and its cache misses dominated).
  i_t hash_depth_offset{1};
  i_t hash_max_depth{30};
  // Zero means size the table from the memory actually available at solve time. Set a nonzero
  // value to pin the budget instead.
  size_t max_hash_bytes{0};
  size_t hash_bytes_cap{size_t{8} << 30};
  // Fraction of available memory the table may claim when sizing itself.
  i_t hash_memory_divisor{4};
  // Above this the search cannot finish even with the terminal, so leave the model to branch and
  // bound rather than spending the budget on it. markshare2 has 60 core columns.
  i_t max_search_cols{62};

  f_t integrality_tolerance{1e-6};
  // Round trip check on the row normalization. Deliberately far tighter than the integrality
  // tolerance: exact row scaling passes it with room, an inexact model does not.
  double exactness_tolerance{1e-9};
};

/**
 * @brief Exact special case solver for markshare / market split models.
 *
 * Recognizes `min sum_k s_k  s.t.  sum_j a_kj x_j + s_k = b_k, x binary, s_k >= 0` with small
 * integer data, and solves it by subset sum dynamic programming plus a pruned backward
 * enumeration, ascending through objective levels. Exhausting a level is a proof, so a solution
 * found at level T with every level below it exhausted is optimal.
 *
 * This is a read only analyzer: it never modifies the problem. It reports a solution and, when
 * it has one, whether that solution is proven optimal.
 */
template <typename i_t, typename f_t>
class markshare_solver_t {
 public:
  markshare_solver_t(const simplex::user_problem_t<i_t, f_t>& user_problem,
                     const markshare_settings_t<i_t, f_t>& settings);

  // Structural recognizer. Cheap rejects come first and nothing is allocated until they pass.
  bool detect();

  // Level search. Returns NOT_APPLICABLE unless detect() ran and succeeded.
  markshare_result_t<i_t, f_t> solve();

  // detect() followed by solve(). The only entry point the solver needs.
  static markshare_result_t<i_t, f_t> try_solve(
    const simplex::user_problem_t<i_t, f_t>& user_problem,
    const markshare_settings_t<i_t, f_t>& settings);

 private:
  // The normalized integer model the search runs on. Built by detect(), read only afterwards.
  struct model_t {
    i_t m{0};  // rows
    i_t n{0};  // core binaries kept in the search

    std::vector<i_t> core_col;    // search index -> solver column
    std::vector<i_t> pinned_col;  // columns with no constraint role, written at their lower bound

    std::vector<markshare_coeff_t> a_row;       // m x n, for the table builders
    std::vector<markshare_coeff_t> a_col;       // n x m, the DFS hot layout
    std::vector<markshare_coeff_t> b;           // row -> normalized rhs
    std::vector<markshare_coeff_t> prefix_max;  // m x (n + 1), running column sums
    std::vector<markshare_coeff_t> row_gcd;     // row -> gcd of its coefficients
    f_t weight{0};  // the common w in c_j = -w * sum_k a_kj
  };

  enum class dfs_result_t { FOUND, EXHAUSTED, BUDGET };

  /**
   * @brief Open addressed set of 64 bit fingerprints of partial sum vectors.
   *
   * A collision can only report a residual as reachable when it is not, which costs one wasted
   * verification. It can never report a reachable residual as unreachable, so it cannot prune a
   * subtree that contains a solution -- the optimality proof stays exact.
   */
  struct fingerprint_set_t {
    std::vector<uint64_t> slot;  // zero marks an empty slot
    size_t mask{0};

    void init(size_t capacity);
    void insert(uint64_t fingerprint);
    bool contains(uint64_t fingerprint) const;
    size_t bytes() const { return slot.size() * sizeof(uint64_t); }
  };

  // Per-search scratch. Held separately from the solver so that several enumerations can run
  // concurrently over the same (read only) model and tables.
  struct dfs_context_t {
    std::vector<markshare_coeff_t> residual;  // (n + 1) * m
    std::vector<uint8_t> branch;              // n + 1
    std::vector<uint8_t> value;               // n
    int64_t nodes{0};

    void resize(i_t n, i_t m)
    {
      residual.assign(size_t(n + 1) * m, 0);
      branch.assign(n + 1, 0);
      value.assign(n, 0);
      nodes = 0;
    }
  };

  // One partially fixed subtree: the values of the trailing columns plus the residual they leave.
  struct seed_t {
    std::vector<uint8_t> value;
    std::vector<markshare_coeff_t> residual;
  };

  bool detect_impl();
  void build_tables();
  i_t choose_hash_depth() const;
  void build_hash();
  uint64_t residual_fingerprint(const markshare_coeff_t* residual) const;
  // Reentrant: reads only the model and the tables, so tasks may run it concurrently.
  // `terminal_depth` is the depth at which the hash is consulted instead of descending further;
  // pass 0 to descend all the way, which is also how a hash hit is turned into an assignment.
  dfs_result_t run_dfs_from(dfs_context_t& ctx,
                            i_t start_depth,
                            const markshare_coeff_t* start_residual,
                            const std::atomic<bool>* stop,
                            i_t terminal_depth) const;
  // Enumerates the trailing `depth` columns, keeping the subtrees that survive pruning.
  void collect_seeds(const std::vector<markshare_coeff_t>& target,
                     i_t depth,
                     std::vector<seed_t>& seeds);
  dfs_result_t run_dfs(const std::vector<markshare_coeff_t>& target);
  bool enumerate_level(i_t level, std::vector<markshare_coeff_t>& slack, i_t index, bool& found);
  // Rebuilds the solver space assignment and verifies it against the untouched user problem.
  bool reconstruct(std::vector<f_t>& solution, f_t& objective) const;

  markshare_prefix_t joint_at(markshare_coeff_t u0, markshare_coeff_t u1) const
  {
    const size_t index = size_t(u0) * joint_stride_ + size_t(u1);
    return joint_[index];
  }


  const simplex::user_problem_t<i_t, f_t>& problem_;
  markshare_settings_t<i_t, f_t> settings_;
  model_t model_;
  bool detected_{false};

  std::vector<std::vector<markshare_prefix_t>> row_tables_;
  std::vector<markshare_prefix_t> joint_;
  std::vector<i_t> extra_rows_;  // rows outside the joint pair
  i_t joint_row0_{0};
  i_t joint_row1_{1};
  size_t joint_stride_{0};

  // Serial DFS state, allocated once and reused across every target vector.
  dfs_context_t context_;
  std::vector<uint8_t> value_;  // the winning assignment
  std::vector<markshare_coeff_t> target_;
  i_t num_threads_{1};

  fingerprint_set_t hash_;
  i_t hash_depth_{0};  // zero means the terminal is disabled

  int64_t nodes_{0};
  int64_t targets_{0};
  double deadline_{0.0};
  bool budget_exhausted_{false};
};

}  // namespace cuopt::mathematical_optimization::mip
