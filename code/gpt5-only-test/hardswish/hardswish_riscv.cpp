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

#include "riscv_usability.h"

namespace ncnn {

HardSwish_riscv::HardSwish_riscv()
{
#if __riscv_vector
    support_packing = true;
#endif // __riscv_vector
}

int HardSwish_riscv::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
#if NCNN_ZFH
    int elembits = bottom_top_blob.elembits();
    if (opt.use_fp16_storage && elembits == 16)
        return HardSwish::forward_inplace(bottom_top_blob, opt); // fp16 route not implemented here
#endif

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

        int i = 0;
#if __riscv_vector
        int n = size;
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e32m8(n);
            vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vl);

            // compute ans = clamp(p*alpha + beta, 0, 1) * p
            vfloat32m8_t _ans = __riscv_vfmul_vf_f32m8(_p, alpha, vl);
            _ans = __riscv_vfadd_vf_f32m8(_ans, beta, vl);
            _ans = __riscv_vfmax_vf_f32m8(_ans, 0.f, vl);
            _ans = __riscv_vfmin_vf_f32m8(_ans, 1.f, vl);
            _ans = __riscv_vfmul_vv_f32m8(_ans, _p, vl);

            __riscv_vse32_v_f32m8(ptr, _ans, vl);
            ptr += vl;
            n -= vl;
        }
#endif // __riscv_vector
        for (; i < size; i++)
        {
            if (*ptr < lower)
                *ptr = 0.f;
            else if (*ptr > upper)
                ;
            else
                *ptr = *ptr * (*ptr * alpha + beta);
            ++ptr;
        }
    }

    return 0;
}

} // namespace ncnn
