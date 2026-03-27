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

#include "bias_riscv.h"

#if __riscv_vector
#include <riscv_vector.h>
#endif // __riscv_vector

#include "riscv_usability.h"

namespace ncnn {

Bias_riscv::Bias_riscv()
{
#if __riscv_vector
    support_packing = true;
#endif // __riscv_vector
}

int Bias_riscv::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int d = bottom_top_blob.d;
    int channels = bottom_top_blob.c;
    int size = w * h * d;

    const float* bias_ptr = bias_data;

    int elempack = bottom_top_blob.elempack;

#if __riscv_vector
    // Fast path for packed layout with dynamic packn
    const int packn = csrr_vlenb() / 4;
    if (elempack == packn)
    {
        const size_t vl = __riscv_vsetvl_e32m1(packn);
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);
            // Broadcast bias vector for this channel pack
            vfloat32m1_t _bias = __riscv_vle32_v_f32m1((const float*)bias_ptr + q * vl, vl);

            for (int i = 0; i < size; i++)
            {
                vfloat32m1_t _p = __riscv_vle32_v_f32m1(ptr + i * vl, vl);
                _p = __riscv_vfadd_vv_f32m1(_p, _bias, vl);
                __riscv_vse32_v_f32m1(ptr + i * vl, _p, vl);
            }
        }
        return 0;
    }
#endif // __riscv_vector

    // Generic path for elempack == 1 or no RVV
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        float* ptr = bottom_top_blob.channel(q);
        float bias = bias_ptr[q];

#if __riscv_vector
        int n = size * elempack;
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e32m8(n);
            vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vl);
            vfloat32m8_t _b = __riscv_vfmv_v_f_f32m8(bias, vl);
            _p = __riscv_vfadd_vv_f32m8(_p, _b, vl);
            __riscv_vse32_v_f32m8(ptr, _p, vl);
            ptr += vl;
            n -= vl;
        }
#else
        for (int i = 0; i < size; i++)
        {
            *ptr = *ptr + bias;
            ptr++;
        }
#endif // __riscv_vector
    }

    return 0;
}

} // namespace ncnn
