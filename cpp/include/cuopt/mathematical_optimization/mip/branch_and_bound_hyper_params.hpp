/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once
#include <cstdint>
#include <limits>

namespace cuopt::mathematical_optimization::mip {

template <typename i_t, typename f_t>
struct branch_and_bound_hyper_params_t {
  // Set the number of workers for best-first search, diving and sub-MIP. This is given in terms
  // of the total number of threads allocated to B&B and do not need to sum to 1 as
  // the OpenMP will automatically switch between them at every sync point. The only exception is
  // that BFS workers are long-lived so setting a high number here may prevent the other
  // workers to execute.
  f_t bfs_workers_ratio    = 0.5;
  f_t diving_workers_ratio = 0.5;
  f_t submip_workers_ratio = 0.125;

  // Maximum difference between the best node of a worker and the global lower bound, as a
  // fraction of the open gap. A larger difference triggers stealing the best nodes of another
  // worker.
  f_t max_lower_bound_degradation = 0.05;

  // Indicate the number of nodes to steal. Takes the best nodes from the victim heap.
  // Keep this value low to avoid thrashing.
  i_t nodes_to_steal = 10;

  // Indicate the number of nodes to steal if the worker has no nodes in the heap
  i_t nodes_to_steal_when_empty = 1000;

  // Number of victims considered when stealing nodes.
  i_t num_victims = 3;

  // Root heuristics: -1 (automatic) / 0 (disable) / 1 (enable)
  i_t root_heuristics = -1;

  // Limit of the total number of simplex iterations in B&B
  std::int64_t simplex_iteration_limit = std::numeric_limits<std::int64_t>::max();
};

}  // namespace cuopt::mathematical_optimization::mip
