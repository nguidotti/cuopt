/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

namespace cuopt::mathematical_optimization::mip {

constexpr int num_search_strategies = 8;

// Indicate the search and variable selection algorithms used by each thread
// in B&B (See [1]).
//
// [1] T. Achterberg, “Constraint Integer Programming,” PhD, Technischen Universität Berlin,
// Berlin, 2007. doi: 10.14279/depositonce-1634.
// [2] J. Witzig and A. Gleixner, “Conflict-Driven Heuristics for Mixed Integer Programming,”
// Feb. 07, 2019, _arXiv_: arXiv:1902.02615. doi:
// [10.48550/arXiv.1902.02615](https://doi.org/10.48550/arXiv.1902.02615).
enum search_strategy_t : int {
  BEST_FIRST           = 0,  // Best-First + Plunging.
  PSEUDOCOST_DIVING    = 1,  // Pseudocost diving (9.2.5)
  LINE_SEARCH_DIVING   = 2,  // Line search diving (9.2.4)
  GUIDED_DIVING        = 3,  // Guided diving (9.2.3).
  COEFFICIENT_DIVING   = 4,  // Coefficient diving (9.2.1)
  FARKAS_DIVING        = 5,  // Farkas Diving (see [2])
  VECTOR_LENGTH_DIVING = 6,  // Vector Length Diving (9.2.6)
  SUBMIP               = 7,  // RINS/RENS (akin to a guided diving, see HiGHS)
};

constexpr search_strategy_t search_strategies[] = {BEST_FIRST,
                                                   PSEUDOCOST_DIVING,
                                                   LINE_SEARCH_DIVING,
                                                   GUIDED_DIVING,
                                                   COEFFICIENT_DIVING,
                                                   FARKAS_DIVING,
                                                   VECTOR_LENGTH_DIVING,
                                                   SUBMIP};

enum class branch_direction_t { NONE = -1, DOWN = 0, UP = 1 };

enum class branch_and_bound_mode_t { PARALLEL = 0, DETERMINISTIC = 1 };

enum class mip_status_t {
  OPTIMAL         = 0,  // The optimal integer solution was found
  UNBOUNDED       = 1,  // The problem is unbounded
  INFEASIBLE      = 2,  // The problem is infeasible
  TIME_LIMIT      = 3,  // The solver reached a time limit
  NODE_LIMIT      = 4,  // The maximum number of nodes was reached
  ITERATION_LIMIT = 5,  // The maximum number of simplex iterations was reached
  NUMERICAL       = 6,  // The solver encountered a numerical error
  UNSET           = 7,  // The status is not set
  WORK_LIMIT      = 8,  // The solver reached a deterministic work limit
  SUBMIP_HALT     = 9   // Halt the search on a suboptimal sub-MIP
};

}  // namespace cuopt::mathematical_optimization::mip
