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

set -euo pipefail

mkdir -p ./dist
RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"

# Keeping it in different directory since cuopt and cuopt_mps_parser are similar in naming.
RAPIDS_PY_WHEEL_NAME="cuopt_mps_parser" rapids-download-wheels-from-s3 ./local-cuopt-dep

RAPIDS_PY_WHEEL_NAME="libcuopt_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 cpp ./dist

RAPIDS_PY_WHEEL_NAME="cuopt_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 ./dist

rapids-pip-retry install \
    ./local-cuopt-dep/cuopt_mps_parser*.whl \
    ./dist/libcuopt_*.whl \
    ./dist/cuopt*.whl --extra-index-url=https://pypi.nvidia.com

python -c "import cuopt"

# echo to expand wildcard before adding `[extra]` requires for pip
rapids-pip-retry install "$(echo ./dist/cuopt*.whl)[test]" --extra-index-url=https://pypi.nvidia.com

if command -v apt-get &> /dev/null; then
    apt-get -y update
    apt-get -y install file unzip
elif command -v dnf &> /dev/null; then
    dnf -y update
    dnf -y install file unzip
fi

./datasets/linear_programming/download_pdlp_test_dataset.sh
cd ./datasets
./get_test_data.sh --solomon
./get_test_data.sh --tsp
cd -
RAPIDS_DATASET_ROOT_DIR=./datasets timeout 30m python -m pytest --verbose --capture=no ./python/cuopt/cuopt/tests/
