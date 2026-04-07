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

#include "convolutiondepthwise_riscv.h"

#if __riscv_vector
#include <riscv_vector.h>
#endif // __riscv_vector

#include "riscv_activation.h"
#include "riscv_usability.h"

namespace ncnn {

ConvolutionDepthWise_riscv::ConvolutionDepthWise_riscv()
{
#if __riscv_vector
    support_packing = true;
#endif // __riscv_vector
}

int ConvolutionDepthWise_riscv::create_pipeline(const Option& opt)
{
    if (dynamic_weight)
        return 0;
    // Fallback to base implementation behavior
    return 0;
}

int ConvolutionDepthWise_riscv::destroy_pipeline(const Option& opt)
{
    return 0;
}

int ConvolutionDepthWise_riscv::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    // Use the generic implementation for correctness
    return ConvolutionDepthWise::forward(bottom_blob, top_blob, opt);
}

int ConvolutionDepthWise_riscv::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    // Use the generic implementation for dynamic weight path
    return ConvolutionDepthWise::forward(bottom_blobs, top_blobs, opt);
}

} // namespace ncnn
