# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

cmake_minimum_required(VERSION 3.30.4 FATAL_ERROR)

set(cuopt_version 25.08.00)

include(../../cmake/rapids_config.cmake)
include(rapids-cuda)
rapids_cuda_init_architectures(libcuopt-python)

project(
  libcuopt-python
  VERSION "${cuopt_version}"
  LANGUAGES CXX CUDA
)

# Check if cuopt is already available. If so, it is the user's responsibility to ensure that the
# CMake package is also available at build time of the Python cuopt package.
find_package(cuopt "${cuopt_version}")

if(cuopt_FOUND)
  return()
endif()

unset(cuopt_FOUND)


include(FetchContent)
FetchContent_Declare(
  argparse
  GIT_REPOSITORY https://github.com/p-ranav/argparse.git
  GIT_TAG v3.2
)
FetchContent_MakeAvailable(argparse)

# Try to find Boost in the system
find_package(Boost 1.65 QUIET COMPONENTS system filesystem thread)

if(NOT Boost_FOUND)
  message(STATUS "Boost not found in system, trying alternative locations...")
  
  # Try common Boost installation paths
  set(Boost_USE_STATIC_LIBS OFF)
  set(Boost_USE_MULTITHREADED ON)
  set(Boost_USE_STATIC_RUNTIME OFF)
  
  # Try to find Boost again with more options
  find_package(Boost 1.65 COMPONENTS system filesystem thread)
  
  if(NOT Boost_FOUND)
    message(FATAL_ERROR "Boost not found. Please install Boost 1.65 or later with components: system, filesystem, thread")
  endif()
else()
  message(STATUS "Using system Boost installation")
endif()

set(BUILD_TESTS OFF)
set(BUILD_BENCHMARKS OFF)
set(CUOPT_BUILD_TESTUTIL OFF)
set(CUDA_STATIC_RUNTIME ON)

add_subdirectory(../../cpp cuopt-cpp)

target_link_libraries(cuopt PRIVATE argparse)
if(Boost_FOUND)
  target_link_libraries(cuopt PRIVATE Boost::system Boost::filesystem Boost::thread)
endif()
target_link_libraries(cuopt_cli PRIVATE argparse)

set(rpaths
  "$ORIGIN/../lib64"
  "$ORIGIN/../../rapids_logger/lib64"
  "$ORIGIN/../../librmm/lib64"
  "$ORIGIN/../../nvidia/cublas/lib"
  "$ORIGIN/../../nvidia/curand/lib"
  "$ORIGIN/../../nvidia/cusolver/lib"
  "$ORIGIN/../../nvidia/cusparse/lib"
)

set_property(TARGET cuopt PROPERTY INSTALL_RPATH ${rpaths} APPEND)
set_property(TARGET cuopt_cli PROPERTY INSTALL_RPATH ${rpaths} APPEND)
