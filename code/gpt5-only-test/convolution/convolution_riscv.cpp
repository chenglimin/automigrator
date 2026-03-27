// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2026 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "convolution_riscv.h"

#if __riscv_vector
#include <riscv_vector.h>
#endif // __riscv_vector

#include "cpu.h"
#include "layer_type.h"

namespace ncnn {

Convolution_riscv::Convolution_riscv()
{
    // Keep default behaviors from base Convolution.
    // We do not advertise packing here to ensure pack1 path for correctness.
}

int Convolution_riscv::create_pipeline(const Option& opt)
{
    // Reuse base implementation
    return Convolution::create_pipeline(opt);
}

int Convolution_riscv::destroy_pipeline(const Option& opt)
{
    // Reuse base implementation
    return Convolution::destroy_pipeline(opt);
}

int Convolution_riscv::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    // Delegate to base Convolution for correctness
    return Convolution::forward(bottom_blob, top_blob, opt);
}

int Convolution_riscv::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    // Dynamic-weight path
    return Convolution::forward(bottom_blobs, top_blobs, opt);
}

} // namespace ncnn
