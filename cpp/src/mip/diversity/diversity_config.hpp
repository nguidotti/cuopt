/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
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

namespace cuopt::linear_programming::detail {

namespace diversity_config_t {
static double time_ratio_on_init_lp              = 0.1;
static double max_time_on_lp                     = 30;
static double time_ratio_of_probing_cache        = 0.10;
static double max_time_on_probing                = 60;
static size_t max_iterations_without_improvement = 15;
static int max_var_diff                          = 256;
static size_t max_solutions                      = 32;
static double initial_infeasibility_weight       = 1000.;
static double default_time_limit                 = 10.;
static int initial_island_size                   = 3;
static int maximum_island_size                   = 8;
static bool use_avg_diversity                    = false;
static double generation_time_limit_ratio        = 0.6;
static double max_island_gen_time                = 600;
static size_t n_sol_for_skip_init_gen            = 3;
static double max_fast_sol_time                  = 10;
static double lp_run_time_if_feasible            = 15.;
static double lp_run_time_if_infeasible          = 1;
static bool halve_population                     = false;
};  // namespace diversity_config_t

}  // namespace cuopt::linear_programming::detail