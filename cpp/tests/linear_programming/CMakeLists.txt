# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

ConfigureTest(LP_UNIT_TEST
    ${CMAKE_CURRENT_SOURCE_DIR}/unit_tests/optimization_problem_test.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/unit_tests/solver_settings_test.cu
)# ##################################################################################################
# - Linear programming PDLP tests ----------------------------------------------------------------------
ConfigureTest(PDLP_TEST
    ${CMAKE_CURRENT_SOURCE_DIR}/pdlp_test.cu
)
# ##################################################################################################
# - MIP tests ----------------------------------------------------------------------
ConfigureTest(MIP_TEST
    ${CMAKE_CURRENT_SOURCE_DIR}/mip_unit_tests/miplib_test.cu
)
ConfigureTest(PROBLEM_TEST
    ${CMAKE_CURRENT_SOURCE_DIR}/mip_unit_tests/problem_test.cu
)
ConfigureTest(ELIM_VAR_REMAP_TEST
    ${CMAKE_CURRENT_SOURCE_DIR}/mip_unit_tests/elim_var_remap_test.cu
)
ConfigureTest(STANDARDIZATION_TEST
    ${CMAKE_CURRENT_SOURCE_DIR}/mip_unit_tests/bounds_standardization_test.cu
)
ConfigureTest(MULTI_PROBE_TEST
    ${CMAKE_CURRENT_SOURCE_DIR}/mip_unit_tests/multi_probe_test.cu
)
ConfigureTest(INCUMBENT_CALLBACK_TEST
    ${CMAKE_CURRENT_SOURCE_DIR}/mip_unit_tests/incumbent_callback_test.cu
)
ConfigureTest(DOC_EXAMPLE_TEST
    ${CMAKE_CURRENT_SOURCE_DIR}/mip_unit_tests/doc_example_test.cu
)
ConfigureTest(UNIT_TEST
    ${CMAKE_CURRENT_SOURCE_DIR}/mip_unit_tests/unit_test.cu
)
ConfigureTest(EMPTY_FIXED_PROBLEMS_TEST
    ${CMAKE_CURRENT_SOURCE_DIR}/mip_unit_tests/empty_fixed_problems_test.cu
)