/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES. All rights
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

#include "routing_test.cuh"

namespace cuopt {
namespace routing {
namespace test {

TEST_P(regression_routing_test_cvrp_t, CVRP) { test_cvrp(); }
TEST_P(regression_routing_test_25_t, CVRPTW_25) { test_cvrptw(); }
TEST_P(regression_routing_test_50_t, CVRPTW_50) { test_cvrptw(); }
TEST_P(regression_routing_test_100_t, CVRPTW_100) { test_cvrptw(); }
TEST_P(float_regression_test_t, CVRPTW) { test_cvrptw(); }
TEST_P(float_pickup_regression_test_t, CVRPTW) { test_cvrptw(); }

INSTANTIATE_TEST_SUITE_P(
  l2_cvrp,
  regression_routing_test_cvrp_t,
  ::testing::ValuesIn(parse_tests(cuopt::test::read_tests("datasets/ref/cvrp.txt"))));
INSTANTIATE_TEST_SUITE_P(
  l2_25,
  regression_routing_test_25_t,
  ::testing::ValuesIn(parse_tests(cuopt::test::read_tests("datasets/ref/solomon_25.txt"))));
INSTANTIATE_TEST_SUITE_P(
  l2_50,
  regression_routing_test_50_t,
  ::testing::ValuesIn(parse_tests(cuopt::test::read_tests("datasets/ref/solomon_50.txt"))));
INSTANTIATE_TEST_SUITE_P(
  l2_100,
  regression_routing_test_100_t,
  ::testing::ValuesIn(parse_tests(cuopt::test::read_tests("datasets/ref/solomon_100.txt"))));
INSTANTIATE_TEST_SUITE_P(
  l2_homberger,
  float_regression_test_t,
  ::testing::ValuesIn(parse_tests(cuopt::test::read_tests("datasets/ref/homberger.txt"))));
INSTANTIATE_TEST_SUITE_P(
  l2_pdptw,
  float_pickup_regression_test_t,
  ::testing::ValuesIn(parse_tests(cuopt::test::read_tests("datasets/ref/l2_pickup.txt"))));
CUOPT_TEST_PROGRAM_MAIN()

}  // namespace test
}  // namespace routing
}  // namespace cuopt
