/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <mip_heuristics/tricks/markshare.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <numeric>
#include <random>
#include <vector>

namespace cuopt::mathematical_optimization::mip {

namespace {

// Brute force reference for the single row table: the smallest prefix length whose subset sums
// to exactly u, over all 2^n subsets.
std::vector<markshare_prefix_t> brute_row_table(const std::vector<markshare_coeff_t>& c,
                                                markshare_coeff_t capacity)
{
  const size_t n = c.size();
  std::vector<markshare_prefix_t> f(size_t(capacity) + 1, markshare_unreachable);
  for (uint64_t mask = 0; mask < (uint64_t(1) << n); ++mask) {
    int64_t sum   = 0;
    size_t prefix = 0;
    for (size_t j = 0; j < n; ++j) {
      if ((mask >> j & 1) != 0) {
        sum += c[j];
        prefix = j + 1;
      }
    }
    if (sum >= 0 && sum <= capacity && prefix < f[size_t(sum)]) {
      f[size_t(sum)] = markshare_prefix_t(prefix);
    }
  }
  return f;
}

std::vector<markshare_prefix_t> brute_joint_table(const std::vector<markshare_coeff_t>& c0,
                                                  const std::vector<markshare_coeff_t>& c1,
                                                  markshare_coeff_t cap0,
                                                  markshare_coeff_t cap1)
{
  const size_t n     = c0.size();
  const size_t bits1 = size_t(cap1) + 1;
  std::vector<markshare_prefix_t> f((size_t(cap0) + 1) * bits1, markshare_unreachable);
  for (uint64_t mask = 0; mask < (uint64_t(1) << n); ++mask) {
    int64_t s0 = 0, s1 = 0;
    size_t prefix = 0;
    for (size_t j = 0; j < n; ++j) {
      if ((mask >> j & 1) != 0) {
        s0 += c0[j];
        s1 += c1[j];
        prefix = j + 1;
      }
    }
    if (s0 >= 0 && s0 <= cap0 && s1 >= 0 && s1 <= cap1) {
      const size_t index = size_t(s0) * bits1 + size_t(s1);
      if (prefix < f[index]) { f[index] = markshare_prefix_t(prefix); }
    }
  }
  return f;
}

}  // namespace

TEST(markshare_tables, row_table_edge_cases)
{
  struct case_t {
    std::vector<markshare_coeff_t> coefficients;
    markshare_coeff_t capacity;
  };
  const std::vector<case_t> cases = {
    {{}, 0},
    {{}, 10},
    {{5}, 4},               // coefficient larger than the capacity
    {{0, 0, 3}, 5},         // zero coefficients
    {{3, 3, 3}, 9},         // duplicates
    {{1, 2, 4, 8, 16}, 31}, // every residual reachable
    {{64, 65, 127, 128}, 200},
  };

  for (const auto& c : cases) {
    std::vector<markshare_prefix_t> got;
    markshare_build_row_table(c.coefficients, c.capacity, got);
    const auto want = brute_row_table(c.coefficients, c.capacity);
    ASSERT_EQ(got.size(), want.size());
    for (size_t u = 0; u < want.size(); ++u) {
      EXPECT_EQ(got[u], want[u]) << "u=" << u << " capacity=" << c.capacity;
    }
  }
}

TEST(markshare_tables, joint_table_aliasing_cases)
{
  struct case_t {
    std::vector<markshare_coeff_t> c0;
    std::vector<markshare_coeff_t> c1;
    markshare_coeff_t cap0;
    markshare_coeff_t cap1;
    const char* what;
  };
  // Each u0 block is padded to whole words so a shift cannot bleed into the next block. These
  // cases are the ones that would expose it if the padding were wrong: a grid whose width is not
  // word aligned, coefficients that push u1 past cap1, and a zero first-row coefficient, which
  // makes the source and destination blocks the same memory.
  const std::vector<case_t> cases = {
    {{1, 2, 3, 4}, {70, 65, 33, 5}, 6, 100, "cap1+1 = 101, not word aligned"},
    {{0, 0, 1}, {7, 13, 5}, 3, 37, "a0 == 0, source aliases destination"},
    {{2, 0, 1}, {0, 5, 0}, 4, 9, "zero coefficients in both rows"},
    {{1, 1}, {64, 64}, 2, 128, "exact word multiple"},
    {{1, 1, 1}, {63, 1, 65}, 3, 129, "straddles a word boundary"},
  };

  for (const auto& c : cases) {
    std::vector<markshare_prefix_t> got;
    markshare_build_joint_table(c.c0, c.c1, c.cap0, c.cap1, got);
    const auto want = brute_joint_table(c.c0, c.c1, c.cap0, c.cap1);
    ASSERT_EQ(got.size(), want.size()) << c.what;
    for (size_t index = 0; index < want.size(); ++index) {
      EXPECT_EQ(got[index], want[index]) << c.what << " index=" << index;
    }
  }
}

TEST(markshare_tables, row_table_matches_brute_force)
{
  std::mt19937 rng(12345);
  for (int trial = 0; trial < 200; ++trial) {
    const size_t n            = 1 + rng() % 14;
    const markshare_coeff_t c = 1 + markshare_coeff_t(rng() % 200);
    std::vector<markshare_coeff_t> coefficients(n);
    for (auto& v : coefficients) { v = markshare_coeff_t(rng() % (c + 30)); }

    std::vector<markshare_prefix_t> got;
    markshare_build_row_table(coefficients, c, got);
    const auto want = brute_row_table(coefficients, c);
    ASSERT_EQ(got, want) << "trial=" << trial;
  }
}

TEST(markshare_tables, joint_table_matches_brute_force)
{
  std::mt19937 rng(6789);
  for (int trial = 0; trial < 200; ++trial) {
    const size_t n               = 1 + rng() % 12;
    const markshare_coeff_t cap0 = 1 + markshare_coeff_t(rng() % 40);
    const markshare_coeff_t cap1 = 1 + markshare_coeff_t(rng() % 150);
    std::vector<markshare_coeff_t> c0(n), c1(n);
    for (size_t j = 0; j < n; ++j) {
      c0[j] = markshare_coeff_t(rng() % (cap0 + 5));
      c1[j] = markshare_coeff_t(rng() % (cap1 + 20));
    }

    std::vector<markshare_prefix_t> got;
    markshare_build_joint_table(c0, c1, cap0, cap1, got);
    const auto want = brute_joint_table(c0, c1, cap0, cap1);
    ASSERT_EQ(got, want) << "trial=" << trial;
  }
}

TEST(markshare_tables, joint_table_dominates_row_table)
{
  // Reaching u0 jointly with some u1 can never need a shorter prefix than reaching u0 alone.
  std::mt19937 rng(2468);
  for (int trial = 0; trial < 100; ++trial) {
    const size_t n               = 1 + rng() % 10;
    const markshare_coeff_t cap0 = 1 + markshare_coeff_t(rng() % 30);
    const markshare_coeff_t cap1 = 1 + markshare_coeff_t(rng() % 60);
    std::vector<markshare_coeff_t> c0(n), c1(n);
    for (size_t j = 0; j < n; ++j) {
      c0[j] = markshare_coeff_t(rng() % (cap0 + 3));
      c1[j] = markshare_coeff_t(rng() % (cap1 + 3));
    }

    std::vector<markshare_prefix_t> row, joint;
    markshare_build_row_table(c0, cap0, row);
    markshare_build_joint_table(c0, c1, cap0, cap1, joint);

    const size_t bits1 = size_t(cap1) + 1;
    for (size_t u0 = 0; u0 <= size_t(cap0); ++u0) {
      markshare_prefix_t best = markshare_unreachable;
      for (size_t u1 = 0; u1 < bits1; ++u1) {
        best = std::min(best, joint[u0 * bits1 + u1]);
      }
      EXPECT_LE(row[u0], best) << "trial=" << trial << " u0=" << u0;
    }
  }
}

}  // namespace cuopt::mathematical_optimization::mip
