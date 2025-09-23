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

#include <dual_simplex/presolve.hpp>
#include <dual_simplex/solution.hpp>

namespace cuopt::linear_programming::dual_simplex {

// Applies the simple rounding procedure from [1, Section 9.1.2]
// [1] T. Achterberg, “Constraint Integer Programming,” PhD,
// Technischen Universität Berlin, Berlin, 2007. doi: 10.14279/depositonce-1634.
template <typename i_t, typename f_t>
bool simple_rounding(lp_solution_t<i_t, f_t>& lp_solution,
                     const lp_problem_t<i_t, f_t>& lp_problem,
                     const std::vector<i_t>& fractional);

}  // namespace cuopt::linear_programming::dual_simplex
