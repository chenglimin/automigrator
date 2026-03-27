// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2022 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "deformableconv2d_riscv.h"

#if __riscv_vector
#include <riscv_vector.h>
#endif // __riscv_vector

#include "riscv_activation.h"
#include "riscv_usability.h"

#include "benchmark.h"
#include "cpu.h"
#include "layer_type.h"

namespace ncnn {

DeformableConv2D_riscv::DeformableConv2D_riscv()
{
#if __riscv_vector
    support_packing = true;
#endif // __riscv_vector
}

int DeformableConv2D_riscv::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    const Mat& offset = bottom_blobs[1];
    const bool has_mask = (bottom_blobs.size() == 3);

    // determine target out_elempack following packn spec
    int out_elempack = 1;
#if __riscv_vector
    if (opt.use_packing_layout)
    {
        const int packn = csrr_vlenb() / 4;
        if (num_output % packn == 0)
            out_elempack = packn;
    }
#endif // __riscv_vector

    // If everything is pack1 already, call generic
    if (bottom_blob.elempack == 1 && (!has_mask || bottom_blobs.size() < 3 || bottom_blobs[2].elempack == 1) && offset.elempack == 1)
    {
        // Let base create and fill top
        return DeformableConv2D::forward(bottom_blobs, top_blobs, opt);
    }

    // Fallback for all other cases: unpack all inputs to pack1, run generic, then repack output if desired
    std::vector<Mat> bb = bottom_blobs;

    Mat bottom_unpacked;
    if (bottom_blob.elempack != 1)
        convert_packing(bottom_blob, bottom_unpacked, 1, opt);
    else
        bottom_unpacked = bottom_blob;

    Mat offset_unpacked;
    if (offset.elempack != 1)
        convert_packing(offset, offset_unpacked, 1, opt);
    else
        offset_unpacked = offset;

    Mat mask_unpacked;
    if (has_mask)
    {
        const Mat& mask = bottom_blobs[2];
        if (mask.elempack != 1)
            convert_packing(mask, mask_unpacked, 1, opt);
        else
            mask_unpacked = mask;
    }

    bb[0] = bottom_unpacked;
    bb[1] = offset_unpacked;
    if (has_mask)
    {
        if (bb.size() < 3)
            bb.resize(3);
        bb[2] = mask_unpacked;
    }

    Mat top_unpacked;
    std::vector<Mat> tb(1);
    tb[0] = top_unpacked;

    int ret = DeformableConv2D::forward(bb, tb, opt);
    if (ret != 0)
        return ret;

    // repack result if needed
    if (out_elempack > 1)
    {
        Mat top_packed;
        convert_packing(tb[0], top_packed, out_elempack, opt);
        top_blobs[0] = top_packed;
    }
    else
    {
        top_blobs[0] = tb[0];
    }

    return 0;
}

} // namespace ncnn
