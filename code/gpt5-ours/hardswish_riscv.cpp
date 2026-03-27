// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "hardswish_riscv.h"

#if __riscv_vector
#include <riscv_vector.h>
#endif // __riscv_vector

namespace ncnn {

HardSwish_riscv::HardSwish_riscv()
{
#if __riscv_vector
    support_packing = true;
#endif // __riscv_vector
}

int HardSwish_riscv::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int d = bottom_top_blob.d;
    int channels = bottom_top_blob.c;
    int elempack = bottom_top_blob.elempack;
    int size = w * h * d * elempack;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        float* ptr = bottom_top_blob.channel(q);

#if __riscv_vector
        int n = size;
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e32m8(n);
            vfloat32m8_t _v = __riscv_vle32_v_f32m8(ptr, vl);

            // masks for regions
            vbool4_t _lower = __riscv_vmflt_vf_f32m8_b4(_v, lower, vl);
            vbool4_t _higher = __riscv_vmfgt_vf_f32m8_b4(_v, upper, vl);
            vbool4_t _apply = __riscv_vmnor_mm_b4(_lower, _higher, vl); // middle region

            // set lower region to 0, keep others
            _v = __riscv_vfmerge_vfm_f32m8(_v, 0.f, _lower, vl);

            // compute x*(x*alpha + beta) on apply region
            vfloat32m8_t _p0 = __riscv_vfmul_vf_f32m8_mu(_apply, _v, _v, alpha, vl);
            _p0 = __riscv_vfadd_vf_f32m8_m(_apply, _p0, beta, vl);
            _v = __riscv_vfmul_vv_f32m8_mu(_apply, _v, _v, _p0, vl);

            __riscv_vse32_v_f32m8(ptr, _v, vl);
            ptr += vl;
            n -= vl;
        }
#else
        for (int i = 0; i < size; i++)
        {
            if (*ptr < lower)
                *ptr = 0.f;
            else if (*ptr > upper)
                ;
            else
                *ptr = *ptr * (*ptr * alpha + beta);
            ptr++;
        }
#endif // __riscv_vector
    }

    return 0;
}

} // namespace ncnn
