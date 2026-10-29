#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

source rapids-init-pip

package_name="libcuopt"
package_dir="python/libcuopt"

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"

LIBCUOPT_CLIENT_WHEELHOUSE=$(rapids-download-from-github "$(rapids-artifact-name wheel_cpp libcuopt_client cuopt)")
echo "libcuopt-client @ file://$(echo ${LIBCUOPT_CLIENT_WHEELHOUSE}/libcuopt_client-*.whl)" >> "${PIP_CONSTRAINT}"

LIBCUOPT_MATHOPT_WHEELHOUSE=$(rapids-download-from-github "$(rapids-artifact-name wheel_cpp libcuopt_mathopt cuopt --cuda "$RAPIDS_CUDA_VERSION")")
echo "libcuopt-mathopt-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo ${LIBCUOPT_MATHOPT_WHEELHOUSE}/libcuopt_mathopt_*.whl)" >> "${PIP_CONSTRAINT}"

LIBCUOPT_ROUTING_WHEELHOUSE=$(rapids-download-from-github "$(rapids-artifact-name wheel_cpp libcuopt_routing cuopt --cuda "$RAPIDS_CUDA_VERSION")")
echo "libcuopt-routing-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo ${LIBCUOPT_ROUTING_WHEELHOUSE}/libcuopt_routing_*.whl)" >> "${PIP_CONSTRAINT}"

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

# Compile with GCC, but bundle LLVM libomp as the OpenMP runtime. With libomp, CMake also builds
# our own cuDSS threading layer against it (see cpp/CMakeLists.txt,
# cpp/src/barrier/cudss_mtlayer_cuopt.cpp), so cuOpt and cuDSS share one runtime (#1219).
source ci/utils/install_llvm_openmp.sh
export SKBUILD_CMAKE_ARGS="-DOpenMP_omp_LIBRARY:FILEPATH=${LIBOMP_LIBRARY};-DCUOPT_FIND_CLIENT_EXTERNALLY=ON;-DCUOPT_FIND_MATHOPT_EXTERNALLY=ON;-DCUOPT_FIND_ROUTING_EXTERNALLY=ON"

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
  --exclude "libcuopt_client.so"
  --exclude "libcuopt_mathopt.so"
  --exclude "libcuopt_routing.so"
  --exclude "libnccl.so.*"
  --exclude "libnvJitLink.so*"
  --exclude "librapids_logger.so"
  --exclude "librmm.so"
  # OpenSSL 3 is intentionally NOT bundled. Resolving libssl.so.3 / libcrypto.so.3
  # at runtime via the host (or container image) keeps libcrypto and the FIPS
  # provider (system or mounted) byte-version-matched, which is required for
  # the FIPS provider's HMAC integrity check and avoids loading two libcrypto.so.3
  # in the same process. Hosts must provide libssl.so.3 / libcrypto.so.3 (Ubuntu
  # 22.04+, RHEL/Rocky 9+, manylinux_2_28+ with openssl3, Debian 12+).
  --exclude "libssl.so.3"
  --exclude "libcrypto.so.3"
)

ci/build_wheel.sh libcuopt ${package_dir}

mkdir -p final_dist
python -m auditwheel repair "${EXCLUDE_ARGS[@]}" -w "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}" ${package_dir}/dist/*

ci/validate_wheel.sh ${package_dir} "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}"

RAPIDS_PACKAGE_NAME="$(rapids-artifact-name wheel_cpp libcuopt cuopt --cuda "$RAPIDS_CUDA_VERSION")"
export RAPIDS_PACKAGE_NAME
