#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Fetch LLVM's OpenMP runtime from conda-forge and set LIBOMP_LIBRARY to it. Source this script:
# cuOpt is compiled with GCC, which emits GOMP_* calls that libomp implements, and the build links
# libomp through -DOpenMP_omp_LIBRARY=${LIBOMP_LIBRARY} (see cpp/CMakeLists.txt). The fetched
# build satisfies the same llvm-openmp floor as the conda packages (dependencies.yaml), which the
# distribution's own libomp may not.

LIBOMP_DIR="$(pwd)/llvm_openmp"
python3 -m pip install --quiet zstandard
python3 ci/utils/fetch_llvm_openmp.py "${LIBOMP_DIR}"
LIBOMP_LIBRARY="${LIBOMP_DIR}/libomp.so"
export LIBOMP_LIBRARY

# auditwheel repair does its own dependency resolution separately from the compiler; without
# this it can't see the fetched copy.
export LD_LIBRARY_PATH="${LIBOMP_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
