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

#include <algorithm>
#include <vector>

#include <dual_simplex/mip_node.hpp>
#include <utilities/pcg.hpp>

namespace cuopt::linear_programming::dual_simplex {

template <typename i_t, typename f_t>
struct diving_root_t {
  mip_node_t<i_t, f_t> root_node;
  f_t score;

  diving_root_t(mip_node_t<i_t, f_t>&& root_node) : root_node(std::move(root_node))
  {
    score = root_node.best_pseudocost_estimate;
  }

  friend bool operator>(const diving_root_t<i_t, f_t>& a, const diving_root_t<i_t, f_t>& b)
  {
    return a.score > b.score;
  }
};

// A min-heap for storing the starting nodes for the dives.
// This has a maximum size of INT_MAX, such that the container
// will discard the least promising node if the queue is full.
template <typename i_t, typename f_t>
class diving_queue_t {
 private:
  std::vector<diving_root_t<i_t, f_t>> buffer;
  PCG rng;
  const double epsilon = 0.1;  // Probability to grab a random node

 public:
  diving_queue_t() {}

  void push(diving_root_t<i_t, f_t>&& node)
  {
    buffer.push_back(std::move(node));
    std::push_heap(buffer.begin(), buffer.end(), std::greater<>());
  }

  void emplace(mip_node_t<i_t, f_t>&& root_node)
  {
    buffer.emplace_back(std::move(root_node));
    std::push_heap(buffer.begin(), buffer.end(), std::greater<>());
  }

  diving_root_t<i_t, f_t> pop()
  {
    if (rng.next<f_t>() <= epsilon) {
      i_t idx = rng.uniform<i_t>(0, buffer.size());
      std::swap(buffer[idx], buffer.back());
      diving_root_t<i_t, f_t> node = std::move(buffer.back());
      buffer.pop_back();
      std::make_heap(buffer.begin(), buffer.end(), std::greater<>());
      return node;

    } else {
      std::pop_heap(buffer.begin(), buffer.end(), std::greater<>());
      diving_root_t<i_t, f_t> node = std::move(buffer.back());
      buffer.pop_back();
      return node;
    }
  }

  i_t size() const { return buffer.size(); }
  const diving_root_t<i_t, f_t>& top() const { return buffer.front(); }
  void clear() { buffer.clear(); }
};

}  // namespace cuopt::linear_programming::dual_simplex
