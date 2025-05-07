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

#include "../utilities/pdlp_test_utilities.cuh"
#include "mip_utils.cuh"

#include <cuopt/linear_programming/solve.hpp>
#include <cuopt/linear_programming/utilities/internals.hpp>
#include <mps_parser/parser.hpp>
#include <utilities/common_utils.hpp>
#include <utilities/error.hpp>

#include <raft/sparse/detail/cusparse_wrappers.h>
#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>

#include <gtest/gtest.h>

#include <thrust/count.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sequence.h>

#include <cstdint>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace cuopt::linear_programming::test {

class test_incumbent_callback_t : public cuopt::internals::lp_incumbent_sol_callback_t {
 public:
  void set_solution(void* data, size_t size, double cost) override
  {
    rmm::cuda_stream_view stream{};
    rmm::device_uvector<double> assignment(size, stream);
    raft::copy(assignment.data(), static_cast<double*>(data), size, stream);
    stream.synchronize();
    solutions.push_back(std::make_pair(cost, std::move(assignment)));
  }
  std::vector<std::pair<double, rmm::device_uvector<double>>> solutions;
};

void check_solutions(const test_incumbent_callback_t& incumbent_callback,
                     const cuopt::mps_parser::mps_data_model_t<int, double>& op_problem,
                     const cuopt::linear_programming::mip_solver_settings_t<int, double>& settings)
{
  for (const auto& solution : incumbent_callback.solutions) {
    EXPECT_EQ(solution.second.size(), op_problem.get_variable_lower_bounds().size());
    test_variable_bounds(op_problem, solution.second, settings);
    const double unscaled_acceptable_tol = 0.1;
    test_constraint_sanity_per_row(
      op_problem,
      solution.second,
      // because of scaling the values are not as accurate, so add more relative tolerance
      unscaled_acceptable_tol,
      settings.get_relative_tolerance());
    test_objective_sanity(op_problem, solution.second, solution.first, 1e-4);
  }
}

void test_incumbent_callback(std::string test_instance)
{
  const raft::handle_t handle_{};
  std::cout << "Running: " << test_instance << std::endl;
  auto path = make_path_absolute(test_instance);
  cuopt::mps_parser::mps_data_model_t<int, double> mps_problem =
    cuopt::mps_parser::parse_mps<int, double>(path, false);
  handle_.sync_stream();
  auto op_problem = mps_data_model_to_optimization_problem(&handle_, mps_problem);

  auto settings = mip_solver_settings_t<int, double>{};
  settings.set_time_limit(30.);
  test_incumbent_callback_t incumbent_callback;
  settings.set_incumbent_solution_callback(&incumbent_callback);
  auto solution = solve_mip(op_problem, settings);
  check_solutions(incumbent_callback, mps_problem, settings);
}

TEST(mip_solve, incumbent_callback_test)
{
  std::vector<std::string> test_instances = {"mip/50v-10.mps", "mip/neos5.mps", "mip/swath1.mps"};
  for (const auto& test_instance : test_instances) {
    test_incumbent_callback(test_instance);
  }
}

}  // namespace cuopt::linear_programming::test
