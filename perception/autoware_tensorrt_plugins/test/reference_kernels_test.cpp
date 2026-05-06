// Copyright 2026 TIER IV, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <gtest/gtest.h>

#include <cuda_runtime_api.h>

namespace
{

TEST(ReferenceKernelsTest, CudaRuntimeAvailable)
{
  int device_count = 0;
  ASSERT_EQ(cudaGetDeviceCount(&device_count), cudaSuccess);
  if (device_count == 0) {
    GTEST_SKIP() << "CUDA device not available";
  }
}

}  // namespace
