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


# Set up Boost variables globally before any FetchContent operations
# This ensures any subproject that tries to find_package(Boost) will find our setup
set(Boost_FOUND TRUE CACHE BOOL "Boost found via FetchContent" FORCE)
set(Boost_IOSTREAMS_FOUND TRUE CACHE BOOL "Boost iostreams found" FORCE)
set(Boost_PROGRAM_OPTIONS_FOUND TRUE CACHE BOOL "Boost program_options found" FORCE)
set(Boost_SERIALIZATION_FOUND TRUE CACHE BOOL "Boost serialization found" FORCE)
set(Boost_FILESYSTEM_FOUND TRUE CACHE BOOL "Boost filesystem found" FORCE)
set(Boost_SYSTEM_FOUND TRUE CACHE BOOL "Boost system found" FORCE)
set(Boost_THREAD_FOUND TRUE CACHE BOOL "Boost thread found" FORCE)
set(Boost_VERSION "1.85.0" CACHE STRING "Boost version" FORCE)
set(Boost_VERSION_STRING "1.85.0" CACHE STRING "Boost version string" FORCE)
set(PAPILO_HAVE_BOOST_IOSTREAMS 1 CACHE STRING "Papilo boost iostreams" FORCE)
set(PAPILO_COMMAND_LINE_AVAILABLE 1 CACHE STRING "Papilo command line available" FORCE)
set(PAPILO_SERIALIZATION_AVAILABLE 1 CACHE STRING "Papilo serialization available" FORCE)

include(FetchContent)

# Fetch dependencies
FetchContent_Declare(
  argparse
  GIT_REPOSITORY https://github.com/p-ranav/argparse.git
  GIT_TAG v3.2
)
FetchContent_MakeAvailable(argparse)

#FetchContent_Declare(
#  tbb
#  GIT_REPOSITORY https://github.com/oneapi-src/oneTBB.git
#  GIT_TAG v2022.2.0
#)
#FetchContent_MakeAvailable(tbb)

FetchContent_Declare(
  boost
  GIT_REPOSITORY https://github.com/boostorg/boost.git
  GIT_TAG boost-1.85.0
)
FetchContent_MakeAvailable(boost)

# Path to boost root
set(BOOST_ROOT "${boost_SOURCE_DIR}")

# Only do this the first time Boost is fetched
if(NOT EXISTS "${boost_SOURCE_DIR}/boost/version.hpp")
  message(STATUS "Running Boost bootstrap and b2 headers...")

  execute_process(
    COMMAND ./bootstrap.sh
    WORKING_DIRECTORY ${boost_SOURCE_DIR}
    RESULT_VARIABLE BOOTSTRAP_RESULT
  )
  if(NOT BOOTSTRAP_RESULT EQUAL 0)
    message(FATAL_ERROR "Boost bootstrap.sh failed with code ${BOOTSTRAP_RESULT}")
  endif()

  execute_process(
    COMMAND ./b2 headers
    WORKING_DIRECTORY ${boost_SOURCE_DIR}
    RESULT_VARIABLE B2_RESULT
  )
  if(NOT B2_RESULT EQUAL 0)
    message(FATAL_ERROR "Boost b2 headers failed with code ${B2_RESULT}")
  endif()
 
  execute_process(
    COMMAND ./b2
      link=shared
      variant=release
      --with-regex
      --with-serialization
      --with-program_options
      --with-iostreams
      threading=multi
      cxxflags=-fPIC
    WORKING_DIRECTORY ${boost_SOURCE_DIR}
    RESULT_VARIABLE B2_BUILD_RESULT
  )

  if(NOT B2_BUILD_RESULT EQUAL 0)
    message(FATAL_ERROR "Boost b2 library build failed with code ${B2_BUILD_RESULT}")
  endif()

endif()

# --- Add include path manually ---
include_directories(${boost_SOURCE_DIR})

list(APPEND CMAKE_PREFIX_PATH "${boost_SOURCE_DIR}/stage/lib/cmake")

list(APPEND CMAKE_PREFIX_PATH
  "/repo/python/libcuopt/build/py3-none-linux_x86_64/_deps/boost-src/stage/lib/cmake"
)


# Now include specific components
find_package(Boost 1.85 REQUIRED COMPONENTS
  iostreams
  program_options
  serialization
  regex
)

set(BOOST_STAGE_LIB "${boost_SOURCE_DIR}/stage/lib")

# Manually define Boost paths and targets since boostorg/boost doesn't define them
# Tell CMake exactly where to find boost/version.hpp
#set(Boost_INCLUDE_DIR "${boost_SOURCE_DIR}/libs/config/include")
set(Boost_INCLUDE_DIR "${boost_SOURCE_DIR}")
#set(BOOST_ROOT "${boost_SOURCE_DIR}")
#set(Boost_NO_SYSTEM_PATHS ON)
set(BOOST_ENABLE_CMAKE ON)
set(BOOST_ENABLE_LIBRARIES serialization program_options iostreams regex)


# message(STATUS "TBB_DIR: ${tbb_BINARY_DIR}/")
# set(TBB_DIR "${tbb_BINARY_DIR}/" CACHE PATH "TBB build directory" FORCE)


# Set TBB auto-download ON before Papilo is configured
# set(TBB_DOWNLOAD ON CACHE BOOL "Enable auto-download of TBB")
# Setting TBB to OFF in the cache disables TBB for this project and all submodules that use this cache variable.
set(TBB OFF CACHE BOOL "Disable TBB")

set(BUILD_TESTS OFF)
set(BUILD_BENCHMARKS OFF)
set(CUOPT_BUILD_TESTUTIL OFF)
set(CUDA_STATIC_RUNTIME ON)

add_subdirectory(../../cpp cuopt-cpp)

target_link_libraries(cuopt PRIVATE 
    argparse 
      ${BOOST_STAGE_LIB}/libboost_regex.so

    Boost::regex 
    Boost::program_options 
    Boost::iostreams 
    Boost::serialization
)
target_link_libraries(cuopt_cli PRIVATE 
    argparse Boost::regex Boost::program_options Boost::iostreams Boost::serialization)

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
