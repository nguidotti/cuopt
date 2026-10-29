#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Same full C++ configure as every other cuOpt wheel; only the install components
# staged into it differ, set by install.components in this package's pyproject.toml.
# Kept in step with ci/build_wheel_libcuopt.sh -- regenerate all four together.

source rapids-init-pip

package_name="libcuopt_client"
package_dir="python/libcuopt_client"

# Install rockylinux repo
if command -v dnf &> /dev/null; then
    bash ci/utils/update_rockylinux_repo.sh
fi

# Install Boost and TBB
bash ci/utils/install_boost_tbb.sh

# Install libuuid (needed by cuopt_grpc_server)
if command -v dnf &> /dev/null; then
    dnf install -y libuuid-devel
elif command -v apt-get &> /dev/null; then
    apt-get update
    apt-get install -y uuid-dev
fi

source rapids-configure-sccache

# Install Protobuf + gRPC (protoc + grpc_cpp_plugin)
bash ci/utils/install_protobuf_grpc.sh

# Compile with GCC, but bundle LLVM libomp as the OpenMP runtime. CMake also builds
# our own cuDSS threading layer against it (see cpp/CMakeLists.txt,
# cpp/src/barrier/cudss_mtlayer_cuopt.cpp), so cuOpt and cuDSS share one runtime (#1219).
source ci/utils/install_llvm_openmp.sh
export SKBUILD_CMAKE_ARGS="-DOpenMP_omp_LIBRARY:FILEPATH=${LIBOMP_LIBRARY};-DSKIP_MATHOPT_BUILD=ON;-DSKIP_ROUTING_BUILD=ON"

# OpenSSL 3 hints for libcuopt's own find_package(OpenSSL).
#
# install_protobuf_grpc.sh links gRPC against OpenSSL 3 (see that script for
# rationale). libcuopt then re-resolves OpenSSL via find_package because
# gRPC's imported targets propagate it transitively. On Rocky/RHEL 8 the
# EPEL openssl3-devel package installs in non-default paths, so we have to
# point CMake at them; on Rocky/RHEL 9+ and Ubuntu 22.04+ the default
# OpenSSL is already 3.x and no hints are needed.
if [ -f /etc/os-release ]; then
    . /etc/os-release
    if [[ "$ID" == "rocky" || "$ID" == "centos" || "$ID" == "rhel" || "$ID" == "fedora" ]] && \
       [[ "${VERSION_ID%%.*}" == "8" ]]; then
        SKBUILD_CMAKE_ARGS="${SKBUILD_CMAKE_ARGS};-DOPENSSL_INCLUDE_DIR=/usr/include/openssl3;-DOPENSSL_SSL_LIBRARY=/usr/lib64/openssl3/libssl.so;-DOPENSSL_CRYPTO_LIBRARY=/usr/lib64/openssl3/libcrypto.so"
    fi
fi

# For pull requests we are enabling assert mode.
if [ "$RAPIDS_BUILD_TYPE" = "pull-request" ]; then
    echo "Building in assert mode"
    export SKBUILD_CMAKE_ARGS="${SKBUILD_CMAKE_ARGS};-DDEFINE_ASSERT=True"
else
    echo "Building in release mode"
fi

# Install cudss
bash ci/utils/install_cudss.sh

rapids-logger "Generating build requirements"

rapids-dependency-file-generator \
  --output requirements \
  --file-key "py_build_${package_name}" \
  --file-key "py_rapids_build_${package_name}" \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION};cuda_suffixed=true;use_cuda_wheels=true" \
| tee /tmp/requirements-build.txt

rapids-logger "Installing build requirements"
rapids-pip-retry install \
    -v \
    --prefer-binary \
    -r /tmp/requirements-build.txt

# build with '--no-build-isolation', for better sccache hit rate
# 0 really means "add --no-build-isolation" (ref: https://github.com/pypa/pip/issues/5735)
export PIP_NO_BUILD_ISOLATION=0


EXCLUDE_ARGS=(
  --exclude "libraft.so"
  --exclude "libcublas.so.*"
  --exclude "libcublasLt.so.*"
  --exclude "libcuda.so.1"
  --exclude "libcudss.so.*"
  --exclude "libcurand.so.*"
  --exclude "libcusolver.so.*"
  --exclude "libcusparse.so.*"
  --exclude "libnccl.so.*"
  --exclude "libnvJitLink.so*"
  --exclude "librapids_logger.so"
  --exclude "librmm.so"
  # Provided by the sibling libcuopt-* wheels, so they must not be vendored here: doing so
  # would ship several copies of each engine and defeat the split. RPATH resolves them.
  --exclude "libcuopt_client.so"
  --exclude "libcuopt_mathopt.so"
  --exclude "libcuopt_routing.so"
  # OpenSSL 3 is intentionally NOT bundled. Resolving libssl.so.3 / libcrypto.so.3
  # at runtime via the host (or container image) keeps libcrypto and the FIPS
  # provider (system or mounted) byte-version-matched, which is required for
  # the FIPS provider's HMAC integrity check and avoids loading two libcrypto.so.3
  # in the same process. Hosts must provide libssl.so.3 / libcrypto.so.3 (Ubuntu
  # 22.04+, RHEL/Rocky 9+, manylinux_2_28+ with openssl3, Debian 12+).
  --exclude "libssl.so.3"
  --exclude "libcrypto.so.3"
)

ci/build_wheel.sh ${package_name} ${package_dir}

mkdir -p final_dist
python -m auditwheel repair "${EXCLUDE_ARGS[@]}" -w "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}" ${package_dir}/dist/*

ci/validate_wheel.sh ${package_dir} "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}"

# The artifact is cuopt_wheel_cpp_libcuopt_client_{arch}; publish matches on the prefix.
RAPIDS_PACKAGE_NAME="$(rapids-artifact-name wheel_cpp libcuopt_client cuopt)"
export RAPIDS_PACKAGE_NAME
