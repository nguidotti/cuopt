#!/bin/bash

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

# Create manifest for dockerhub and nvstaging

docker manifest create --amend nvidia/cuopt:${IMAGE_TAG_PREFIX}-cuda${CUDA_VER}-py${PYTHON_VER} \
    nvidia/cuopt:${IMAGE_TAG_PREFIX}-cuda${CUDA_VER}-py${PYTHON_VER}-amd64 \
    nvidia/cuopt:${IMAGE_TAG_PREFIX}-cuda${CUDA_VER}-py${PYTHON_VER}-arm64

docker manifest annotate nvidia/cuopt:${IMAGE_TAG_PREFIX}-cuda${CUDA_VER}-py${PYTHON_VER} \
    nvidia/cuopt:${IMAGE_TAG_PREFIX}-cuda${CUDA_VER}-py${PYTHON_VER}-arm64 \
    --arch arm64

docker manifest annotate nvidia/cuopt:${IMAGE_TAG_PREFIX}-cuda${CUDA_VER}-py${PYTHON_VER} \
    nvidia/cuopt:${IMAGE_TAG_PREFIX}-cuda${CUDA_VER}-py${PYTHON_VER}-amd64 \
    --arch amd64

docker manifest push nvidia/cuopt:${IMAGE_TAG_PREFIX}-cuda${CUDA_VER}-py${PYTHON_VER}

docker manifest create --amend nvcr.io/nvstaging/nvaie/cuopt:${IMAGE_TAG_PREFIX}-cuda${CUDA_VER}-py${PYTHON_VER} \
    nvcr.io/nvstaging/nvaie/cuopt:${IMAGE_TAG_PREFIX}-cuda${CUDA_VER}-py${PYTHON_VER}-amd64 \
    nvcr.io/nvstaging/nvaie/cuopt:${IMAGE_TAG_PREFIX}-cuda${CUDA_VER}-py${PYTHON_VER}-arm64

docker manifest annotate nvcr.io/nvstaging/nvaie/cuopt:${IMAGE_TAG_PREFIX}-cuda${CUDA_VER}-py${PYTHON_VER} \
    nvcr.io/nvstaging/nvaie/cuopt:${IMAGE_TAG_PREFIX}-cuda${CUDA_VER}-py${PYTHON_VER}-arm64 \
    --arch arm64

docker manifest annotate nvcr.io/nvstaging/nvaie/cuopt:${IMAGE_TAG_PREFIX}-cuda${CUDA_VER}-py${PYTHON_VER} \
    nvcr.io/nvstaging/nvaie/cuopt:${IMAGE_TAG_PREFIX}-cuda${CUDA_VER}-py${PYTHON_VER}-amd64 \
    --arch amd64

docker manifest push nvcr.io/nvstaging/nvaie/cuopt:${IMAGE_TAG_PREFIX}-cuda${CUDA_VER}-py${PYTHON_VER}

# Only create latest manifests for release builds
if [[ "${BUILD_TYPE}" == "release" ]]; then
    docker manifest create --amend nvidia/cuopt:latest-cuda${CUDA_VER}-py${PYTHON_VER} \
        nvidia/cuopt:${IMAGE_TAG_PREFIX}-cuda${CUDA_VER}-py${PYTHON_VER}-amd64 \
        nvidia/cuopt:${IMAGE_TAG_PREFIX}-cuda${CUDA_VER}-py${PYTHON_VER}-arm64

    docker manifest annotate nvidia/cuopt:latest-cuda${CUDA_VER}-py${PYTHON_VER} \
        nvidia/cuopt:${IMAGE_TAG_PREFIX}-cuda${CUDA_VER}-py${PYTHON_VER}-arm64 \
        --arch arm64

    docker manifest annotate nvidia/cuopt:latest-cuda${CUDA_VER}-py${PYTHON_VER} \
        nvidia/cuopt:${IMAGE_TAG_PREFIX}-cuda${CUDA_VER}-py${PYTHON_VER}-amd64 \
        --arch amd64

    docker manifest push nvidia/cuopt:latest-cuda${CUDA_VER}-py${PYTHON_VER}

    docker manifest create --amend nvcr.io/nvstaging/nvaie/cuopt:latest-cuda${CUDA_VER}-py${PYTHON_VER} \
        nvcr.io/nvstaging/nvaie/cuopt:${IMAGE_TAG_PREFIX}-cuda${CUDA_VER}-py${PYTHON_VER}-amd64 \
        nvcr.io/nvstaging/nvaie/cuopt:${IMAGE_TAG_PREFIX}-cuda${CUDA_VER}-py${PYTHON_VER}-arm64

    docker manifest annotate nvcr.io/nvstaging/nvaie/cuopt:latest-cuda${CUDA_VER}-py${PYTHON_VER} \
        nvcr.io/nvstaging/nvaie/cuopt:${IMAGE_TAG_PREFIX}-cuda${CUDA_VER}-py${PYTHON_VER}-arm64 \
        --arch arm64

    docker manifest annotate nvcr.io/nvstaging/nvaie/cuopt:latest-cuda${CUDA_VER}-py${PYTHON_VER} \
        nvcr.io/nvstaging/nvaie/cuopt:${IMAGE_TAG_PREFIX}-cuda${CUDA_VER}-py${PYTHON_VER}-amd64 \
        --arch amd64

    docker manifest push nvcr.io/nvstaging/nvaie/cuopt:latest-cuda${CUDA_VER}-py${PYTHON_VER}
fi