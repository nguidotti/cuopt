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
#include <unordered_set>
#include <vector>

#include <dual_simplex/mip_node.hpp>
#include <utilities/pcg.hpp>

namespace cuopt::linear_programming::dual_simplex {

// Indicate the search and variable selection algorithms used by the thread (See [1]).
//
// [1] T. Achterberg, “Constraint Integer Programming,” PhD, Technischen Universität Berlin,
// Berlin, 2007. doi: 10.14279/depositonce-1634.
enum class thread_type_t {
  EXPLORATION        = 0,  // Best-First + Plunging. Pseudocost branching + Martin's criteria.
  COEFFICIENT_DIVING = 1,  // Coefficient diving (9.2.1)
  LINE_SEARCH_DIVING = 2,  // Line search diving (9.2.4)
  PSEUDOCOST_DIVING  = 3,  // Pseudocost diving (9.2.5)
  GUIDED_DIVING = 4  // Guided diving (9.2.3). If no incumbent is found yet, use pseudocost diving.
};

template <typename i_t, typename f_t>
struct diving_root_t {
  mip_node_t<i_t, f_t> node;
  std::vector<f_t> lower;
  std::vector<f_t> upper;
  std::unordered_set<thread_type_t> tags;
  f_t score;

  diving_root_t(mip_node_t<i_t, f_t>&& new_node, std::vector<f_t>&& lower, std::vector<f_t>&& upper)
    : node(std::move(new_node)), lower(std::move(lower)), upper(std::move(upper))
  {
    score = node.best_pseudocost_estimate;
  }

  diving_root_t(mip_node_t<i_t, f_t>&& new_node,
                const std::vector<f_t>& lower,
                const std::vector<f_t>& upper)
    : node(std::move(new_node)), lower(lower), upper(upper)
  {
    score = node.best_pseudocost_estimate;
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
  static constexpr i_t max_size_ = INT_MAX;
  PCG rng;
  const double epsilon = 0.1;  // Probability to grab a random node

  void update_buffer(i_t idx, thread_type_t type)
  {
    buffer[idx].tags.insert(type);
    if (buffer[idx].tags.size() == 4) {
      std::swap(buffer[idx], buffer.back());
      buffer.pop_back();
      std::make_heap(buffer.begin(), buffer.end(), std::greater<>());
    }
  }

 public:
  diving_queue_t() {}

  void push(diving_root_t<i_t, f_t>&& node)
  {
    buffer.push_back(std::move(node));
    std::push_heap(buffer.begin(), buffer.end(), std::greater<>());
    if (buffer.size() > max_size() - 1) { buffer.pop_back(); }
  }

  void emplace(mip_node_t<i_t, f_t>&& node, std::vector<f_t>&& lower, std::vector<f_t>&& upper)
  {
    buffer.emplace_back(std::move(node), std::move(lower), std::move(upper));
    std::push_heap(buffer.begin(), buffer.end(), std::greater<>());
    if (buffer.size() > max_size() - 1) { buffer.pop_back(); }
  }

  std::optional<diving_root_t<i_t, f_t>> pop(thread_type_t type)
  {
    i_t idx = 0;

    while (idx < buffer.size()) {
      i_t k = idx;

      if (idx + 1 < buffer.size() && rng.next<f_t>() <= epsilon) {
        k = rng.uniform<i_t>(idx + 1, buffer.size());
      }

      if (buffer[k].tags.find(type) == buffer[k].tags.end()) {
        diving_root_t<i_t, f_t> node(
          buffer[k].node.detach_copy(), buffer[k].lower, buffer[k].upper);
        update_buffer(k, type);
        return node;
      }

      ++idx;
    }

    return std::nullopt;
  }

  i_t size() const { return buffer.size(); }
  constexpr i_t max_size() const { return max_size_; }
  const diving_root_t<i_t, f_t>& top() const { return buffer.front(); }
  void clear() { buffer.clear(); }
};

}  // namespace cuopt::linear_programming::dual_simplex
