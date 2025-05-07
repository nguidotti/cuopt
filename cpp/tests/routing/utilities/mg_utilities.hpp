/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights
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
#pragma once

#include <raft/core/comms.hpp>
#include <rmm/device_scalar.hpp>

namespace cuopt {
namespace routing {
namespace test {

template <typename input_t>
input_t host_reduce_scalar(raft::comms::comms_t const& comm,
                           input_t local_val,
                           raft::comms::op_t op,
                           int root,
                           cudaStream_t stream)
{
  rmm::device_scalar<input_t> local_scalar(local_val, stream);
  rmm::device_scalar<input_t> aggregate_scalar(0, stream);
  comm.reduce(local_scalar.data(), aggregate_scalar.data(), 1, op, root, stream);
  return aggregate_scalar.value(stream);
}

}  // namespace test
}  // namespace routing
}  // namespace cuopt
