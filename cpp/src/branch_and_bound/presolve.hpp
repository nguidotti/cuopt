/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <vector>

#include <dual_simplex/simplex_solver_settings.hpp>
#include <dual_simplex/user_problem.hpp>
#include <mip_heuristics/mip_constants.hpp>
#include <mip_heuristics/presolve/third_party_presolve.hpp>

#include "constants.hpp"

namespace cuopt::mathematical_optimization::mip {

// Thin owner of a PaPILO presolver scoped to a single sub-MIP solve. apply() reduces the
// problem in place; the retained column maps let a reduced-space solution be mapped back to
// the original column space via uncrush().
template <typename i_t, typename f_t>
class presolver_t {
 public:
  // Presolve `problem` in place using PaPILO. Returns the presolve status; on
  // INFEASIBLE/UNBOUNDED the problem is left untouched.
  mip_status_t apply(simplex::user_problem_t<i_t, f_t>& problem,
                     const simplex::simplex_solver_settings_t<i_t, f_t>& settings);

  // Map a reduced-space primal solution back to the original column space.
  void uncrush(const std::vector<f_t>& reduced_primal, std::vector<f_t>& full_primal) const;

  const std::vector<i_t>& reduced_to_original_map() const;
  const std::vector<i_t>& original_to_reduced_map() const;

 private:
  third_party_presolve_t<i_t, f_t> third_party_presolver_;
};

}  // namespace cuopt::mathematical_optimization::mip
