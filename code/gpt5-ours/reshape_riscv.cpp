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

#include "reshape_riscv.h"

#if __riscv_vector
#include <riscv_vector.h>
#endif // __riscv_vector

#include "riscv_usability.h"

namespace ncnn {

Reshape_riscv::Reshape_riscv()
{
#if __riscv_vector
    support_packing = true;
#endif // __riscv_vector
}

int Reshape_riscv::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    Mat& top_blob = top_blobs[0];

    // Always operate on pack1 for correctness, then optionally repack to packn
    std::vector<Mat> bottom_blobs_p1(bottom_blobs.size());
    for (size_t i = 0; i < bottom_blobs.size(); i++)
    {
        if (bottom_blobs[i].elempack != 1)
        {
            convert_packing(bottom_blobs[i], bottom_blobs_p1[i], 1, opt);
            if (bottom_blobs_p1[i].empty())
                return -100;
        }
        else
        {
            bottom_blobs_p1[i] = bottom_blobs[i];
        }
    }

    // Use generic reshape to produce pack1 output first
    std::vector<Mat> _top_blobs(1);
    int ret = Reshape::forward(bottom_blobs_p1, _top_blobs, opt);
    if (ret != 0)
        return ret;
    Mat top_blob_p1 = _top_blobs[0];
    if (top_blob_p1.empty())
        return -100;

    // Decide desired out elempack according to packn rule
    int desired_out_elempack = 1;
    if (opt.use_packing_layout)
    {
        const int packn = ncnn::cpu_riscv_vlenb() / 4;
        if (top_blob_p1.dims == 2)
        {
            if (top_blob_p1.h % packn == 0)
                desired_out_elempack = packn;
        }
        else if (top_blob_p1.dims == 3)
        {
            if (top_blob_p1.c % packn == 0)
                desired_out_elempack = packn;
        }
        else if (top_blob_p1.dims == 4)
        {
            if (top_blob_p1.c % packn == 0)
                desired_out_elempack = packn;
        }
    }

    if (desired_out_elempack == 1)
    {
        top_blob = top_blob_p1;
        return 0;
    }

    // Use ncnn generic packing conversion which is RVV-optimized internally on RISC-V
    convert_packing(top_blob_p1, top_blob, desired_out_elempack, opt);
    if (top_blob.empty())
        return -100;
    return 0;
}

} // namespace ncnn
