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

#ifndef NCNN_PADDING_PACKN_FP16_RVV_H
#define NCNN_PADDING_PACKN_FP16_RVV_H

#include "riscv_usability.h"

namespace ncnn {

#if __riscv_zvfh
static void padding_constant_packn_rvv_fp16(const Mat& src, Mat& dst, int top, int bottom, int left, int right, const __fp16* v)
{
    const __fp16* ptr = src;
    __fp16* outptr = dst;
    const int packn = csrr_vlenb() / 2;
    size_t vl = __riscv_vsetvl_e16m1(packn);
    vfloat16m1_t _v = __riscv_vle16_v_f16m1(v, vl);

    int top_size = top * dst.w;
    int bottom_size = bottom * dst.w;

    for (int y = 0; y < top_size; y++)
    {
        __riscv_vse16_v_f16m1(outptr, _v, vl);
        outptr += packn;
    }
    for (int y = 0; y < src.h; y++)
    {
        for (int x = 0; x < left; x++)
        {
            __riscv_vse16_v_f16m1(outptr, _v, vl);
            outptr += packn;
        }
        for (int x = 0; x < src.w; x++)
        {
            vfloat16m1_t _p = __riscv_vle16_v_f16m1(ptr, vl);
            __riscv_vse16_v_f16m1(outptr, _p, vl);
            ptr += packn;
            outptr += packn;
        }
        for (int x = 0; x < right; x++)
        {
            __riscv_vse16_v_f16m1(outptr, _v, vl);
            outptr += packn;
        }
    }
    for (int y = 0; y < bottom_size; y++)
    {
        __riscv_vse16_v_f16m1(outptr, _v, vl);
        outptr += packn;
    }
}
#endif // __riscv_zvfh

} // namespace ncnn

#endif // NCNN_PADDING_PACKN_FP16_RVV_H
