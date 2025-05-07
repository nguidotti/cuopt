#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

set -eou pipefail

mkdir -p ./dist
RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"

RAPIDS_PY_WHEEL_NAME="cuopt_mps_parser" rapids-download-wheels-from-s3 ./dist
rapids-pip-retry install --no-deps ./dist/cuopt_mps_parser*.whl

RAPIDS_PY_WHEEL_NAME="libcuopt_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 cpp ./dist
rapids-pip-retry install --no-deps ./dist/libcuopt_*.whl

RAPIDS_PY_WHEEL_NAME="cuopt_server_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 ./dist

# Download the cuopt built in the previous step
RAPIDS_PY_WHEEL_NAME="cuopt_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 ./local-cuopt-dep
rapids-pip-retry install --no-deps ./local-cuopt-dep/cuopt*.whl --extra-index-url=https://pypi.nvidia.com
rapids-pip-retry install "$(echo ./dist/libcuopt_${RAPIDS_PY_CUDA_SUFFIX}*.whl)[test]" --extra-index-url=https://pypi.nvidia.com

# echo to expand wildcard before adding `[extra]` requires for pip
rapids-pip-retry install "$(echo ./dist/cuopt_server*.whl)[test]" --extra-index-url=https://pypi.nvidia.com

./datasets/linear_programming/download_pdlp_test_dataset.sh

RAPIDS_DATASET_ROOT_DIR=./datasets timeout 30m python -m pytest --verbose --capture=no ./python/cuopt_server/cuopt_server/tests/
