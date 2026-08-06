/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

namespace cuopt::mathematical_optimization {

template <typename i_t, typename f_t>
struct mip_restart_hyper_params_t {
  // Minimum number of nodes that needs to be explored before triggering a restart.
  i_t min_nodes = 10000;

  // Minimum number of "huge" tree estimations before triggering a restart.
  i_t min_huge_tree_estimates = 10;

  // Indicates how the threshold (regarding the number of "huge" tree estimations) grows
  // with the number of nodes explored. Make it harder to restart if the tree is large
  // (nodes * restart_threshold_grow_per_node).
  f_t threshold_grow_per_leaf = 0.0015;

  // Indicates how the threshold (regarding the number of "huge" tree estimations) grows
  // with the number of restarts. Each restart make it harder to trigger another restart
  // (restart_count ^ 1.5).
  f_t threshold_grow_per_restart = 1.5;

  // Indicates the multiple of the current number of explored nodes for the tree to be considered
  // "huge".
  i_t tree_size_multiple = 50;

  // The maximum improvement in the absolute gap for the solver to be considered stagnated
  f_t max_gap_improvement = 0.05;

  // The frequency in terms of the nodes for checking if we should restart
  i_t check_freq = 100;

  // Maximum number of restarts allowed.
  i_t max_restarts = 0;
};

}  // namespace cuopt::mathematical_optimization
