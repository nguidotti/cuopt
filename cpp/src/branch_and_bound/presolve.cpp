/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <branch_and_bound/presolve.hpp>

#include <algorithm>
#include <limits>

namespace cuopt::mathematical_optimization::mip {

mip_status_t presolve_status_to_mip_status(third_party_presolve_status_t status)
{
  switch (status) {
    case third_party_presolve_status_t::OPTIMAL: return mip_status_t::OPTIMAL;
    case third_party_presolve_status_t::INFEASIBLE: return mip_status_t::INFEASIBLE;
    case third_party_presolve_status_t::UNBOUNDED: return mip_status_t::UNBOUNDED;
    case third_party_presolve_status_t::UNBNDORINFEAS: return mip_status_t::INFEASIBLE;
    case third_party_presolve_status_t::REDUCED: return mip_status_t::UNSET;
    case third_party_presolve_status_t::UNCHANGED: return mip_status_t::UNSET;
  }
  return mip_status_t::UNSET;
}

template <typename i_t, typename f_t>
mip_status_t presolver_t<i_t, f_t>::apply(
  simplex::user_problem_t<i_t, f_t>& problem,
  const simplex::simplex_solver_settings_t<i_t, f_t>& settings)
{
  f_t presolve_time_limit = std::min(0.1 * settings.time_limit, 60.0);
  third_party_presolve_status_t status =
    third_party_presolver_.apply(problem, settings, presolve_time_limit, 1);
  return presolve_status_to_mip_status(status);
}

template <typename i_t, typename f_t>
void presolver_t<i_t, f_t>::uncrush(const std::vector<f_t>& reduced_primal,
                                    std::vector<f_t>& full_primal) const
{
  third_party_presolver_.uncrush_primal_solution(reduced_primal, full_primal);
}
template <typename i_t, typename f_t>
void presolver_t<i_t, f_t>::crush(const std::vector<f_t>& full_primal,
                                  std::vector<f_t>& reduced_primal) const
{
  third_party_presolver_.crush_primal_solution(full_primal, reduced_primal);
}

template <typename i_t, typename f_t>
const std::vector<i_t>& presolver_t<i_t, f_t>::reduced_to_original_map() const
{
  return third_party_presolver_.get_reduced_to_original_map();
}

template <typename i_t, typename f_t>
const std::vector<i_t>& presolver_t<i_t, f_t>::original_to_reduced_map() const
{
  return third_party_presolver_.get_original_to_reduced_map();
}

#if MIP_INSTANTIATE_FLOAT
template class presolver_t<int, float>;
#endif

#if MIP_INSTANTIATE_DOUBLE
template class presolver_t<int, double>;
#endif

}  // namespace cuopt::mathematical_optimization::mip
