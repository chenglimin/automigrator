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

#include "relu_riscv.h"

#if __riscv_vector
#include <riscv_vector.h>
#endif // __riscv_vector

#include "cpu.h"

namespace ncnn {

ReLU_riscv::ReLU_riscv()
{
#if __riscv_vector
    support_packing = true;
#endif // __riscv_vector
}

int ReLU_riscv::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    int elembits = bottom_top_blob.elembits();

    if (elembits == 8)
    {
        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;
        int d = bottom_top_blob.d;
        int channels = bottom_top_blob.c;
        int elempack = bottom_top_blob.elempack;
        int size = w * h * d * elempack;

        if (slope == 0.f)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                signed char* ptr = bottom_top_blob.channel(q);
                for (int i = 0; i < size; i++)
                {
                    if (ptr[i] < 0)
                        ptr[i] = 0;
                }
            }
        }
        else
        {
            // TODO: int8 leaky relu if needed
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                signed char* ptr = bottom_top_blob.channel(q);
                for (int i = 0; i < size; i++)
                {
                    if (ptr[i] < 0)
                        ptr[i] = (signed char)((int)roundf((float)ptr[i] * slope));
                }
            }
        }
        return 0;
    }

    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int d = bottom_top_blob.d;
    int channels = bottom_top_blob.c;
    int elempack = bottom_top_blob.elempack;
    int size = w * h * d * elempack;

    if (slope == 0.f)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);
#if __riscv_vector
            int n = size;
            while (n > 0)
            {
                size_t vl = __riscv_vsetvl_e32m8(n);
                vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vl);
                _p = __riscv_vfmax_vf_f32m8(_p, 0.f, vl);
                __riscv_vse32_v_f32m8(ptr, _p, vl);
                ptr += vl;
                n -= vl;
            }
#else
            for (int i = 0; i < size; i++)
            {
                if (ptr[i] < 0)
                    ptr[i] = 0.f;
            }
#endif
        }
    }
    else
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);
#if __riscv_vector
            int n = size;
            while (n > 0)
            {
                size_t vl = __riscv_vsetvl_e32m8(n);
                vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vl);
                // pos = max(p, 0), neg = min(p, 0)
                vfloat32m8_t _zero = __riscv_vfmv_v_f_f32m8(0.f, vl);
                vfloat32m8_t _pos = __riscv_vfmax_vv_f32m8(_p, _zero, vl);
                vfloat32m8_t _neg = __riscv_vfmin_vv_f32m8(_p, _zero, vl);
                vfloat32m8_t _s = __riscv_vfmv_v_f_f32m8(slope, vl);
                vfloat32m8_t _out = __riscv_vfadd_vv_f32m8(_pos, __riscv_vfmul_vv_f32m8(_s, _neg, vl), vl);
                __riscv_vse32_v_f32m8(ptr, _out, vl);
                ptr += vl;
                n -= vl;
            }
#else
            for (int i = 0; i < size; i++)
            {
                if (ptr[i] < 0)
                    ptr[i] *= slope;
            }
#endif
        }
    }

    return 0;
}

} // namespace ncnn
