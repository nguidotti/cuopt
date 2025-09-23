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

#include <cmath>
#include <cstdint>
#include <dual_simplex/presolve.hpp>
#include <dual_simplex/solution.hpp>

namespace cuopt::linear_programming::dual_simplex {

// PCG random number generator (https://www.pcg-random.org/).
// This is the same as raft::random::PCGenerator, but for the host.
class pcg_t {
 public:
  static constexpr uint64_t default_seed   = 0x853c49e6748fea9bULL;
  static constexpr uint64_t default_stream = 0xda3e39cb94b95bdbULL;

  pcg_t(uint64_t seed = default_seed, uint64_t stream = default_stream)
    : state(seed), stream((stream << 1u) | 1u)
  {
  }

  void set_seed(uint64_t seed) { state = seed; }
  void set_stream(uint64_t stream) { stream = (stream << 1u) | 1u; }

  uint32_t next_u32()
  {
    uint32_t ret;
    uint64_t oldstate   = state;
    state               = oldstate * 6364136223846793005ULL + stream;
    uint32_t xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
    uint32_t rot        = oldstate >> 59u;
    ret                 = (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
    return ret;
  }

  uint64_t next_u64()
  {
    uint64_t hi = uint64_t(next_u32()) << 32;
    uint64_t lo = next_u32();
    return hi | lo;
  }

  int32_t next_i32() { return next_u32() >> 1; }
  int64_t next_i64() { return next_u64() >> 1; }
  float next_float() { return static_cast<float>(next_u32() >> 8) * 0x1.0p-24; }
  double next_double() { return static_cast<double>(next_u64() >> 11) * 0x1.0p-53; }

 private:
  uint64_t state;
  uint64_t stream;
};

// Applies the simple rounding procedure from [1, Section 9.1.2]
// [1] T. Achterberg, “Constraint Integer Programming,” PhD,
// Technischen Universität Berlin, Berlin, 2007. doi: 10.14279/depositonce-1634.
template <typename i_t, typename f_t>
bool simple_rounding(const lp_problem_t<i_t, f_t>& lp_problem,
                     lp_solution_t<i_t, f_t>& lp_solution,
                     std::vector<i_t>& fractional);

// template <typename i_t, typename f_t>
// bool nearest_integer_rounding(const lp_problem_t<i_t, f_t>& lp_problem,
//                               const f_t int_tol,
//                               const i_t seed,
//                               const i_t stream,
//                               lp_solution_t<i_t, f_t>& lp_solution,
//                               std::vector<i_t>& fractional);

// template <typename i_t, typename f_t>
// bool rounding(const lp_problem_t<i_t, f_t>& lp_problem,
//               const f_t int_tol,
//               const i_t seed,
//               const i_t stream,
//               lp_solution_t<i_t, f_t>& lp_solution,
//               std::vector<i_t>& fractional)
// {
// }

}  // namespace cuopt::linear_programming::dual_simplex
