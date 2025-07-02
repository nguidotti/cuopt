/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <random>

namespace cuopt::linear_programming::dual_simplex {

template <typename i_t>
class random_t {
 public:
  random_t(i_t seed) : gen(seed) {}

  template <typename T>
  T random_index(T n)
  {
    std::uniform_int_distribution<> distrib(
      0, n - 1);  // Generate random number in the range [min, max]
    return distrib(gen);
  }
  template <typename T>
  T random_value(T min, T max)
  {
    std::uniform_real_distribution<> distrib(min, max);
    return distrib(gen);
  }
 private:
  std::mt19937 gen;
};

}  // namespace cuopt::linear_programming::dual_simplex
