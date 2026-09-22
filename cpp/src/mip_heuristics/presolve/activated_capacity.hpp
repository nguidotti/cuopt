/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#if !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstringop-overflow"  // ignore boost error for pip wheel build
#pragma GCC diagnostic ignored "-Wnarrowing"
#endif
#include <papilo/Config.hpp>
#include <papilo/core/PresolveMethod.hpp>
#include <papilo/core/Problem.hpp>
#include <papilo/core/ProblemUpdate.hpp>
#if !defined(__clang__)
#pragma GCC diagnostic pop
#endif

#include <vector>

namespace cuopt::mathematical_optimization::mip {

// A capacity row wider than this is not worth the gate-set intersection.
static constexpr int ACTIVATED_CAPACITY_MAX_LEN = 4096;

template <typename f_t>
class ActivatedCapacity : public papilo::PresolveMethod<f_t> {
 public:
  ActivatedCapacity() : papilo::PresolveMethod<f_t>()
  {
    this->setName("activatedcapacity");
    this->setType(papilo::PresolverType::kIntegralCols);
    this->setTiming(papilo::PresolverTiming::kMedium);
    // The vacuous copies of these capacity rows are dropped by the cheaper presolvers first, which
    // keeps the gate-set intersection off rows that would gain nothing.
    this->setDelayed(true);
  }

  papilo::PresolveStatus execute(const papilo::Problem<f_t>& problem,
                                 const papilo::ProblemUpdate<f_t>& problemUpdate,
                                 const papilo::Num<f_t>& num,
                                 papilo::Reductions<f_t>& reductions,
                                 const papilo::Timer& timer,
                                 int& reason_of_infeasibility) override;
};

}  // namespace cuopt::mathematical_optimization::mip
