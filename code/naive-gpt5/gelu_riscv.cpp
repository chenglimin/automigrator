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

#include "gelu_riscv.h"

#if __riscv_vector
#include <riscv_vector.h>
#include "rvv_mathfun.h"
#endif // __riscv_vector

#include "cpu.h"

namespace ncnn {

GELU_riscv::GELU_riscv()
{
#if __riscv_vector
    support_packing = true;
#endif // __riscv_vector
}

int GELU_riscv::create_pipeline(const Option& /*opt*/)
{
    if (!fast_gelu)
    {
        support_packing = false;
    }
    return 0;
}

int GELU_riscv::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    if (!fast_gelu)
    {
        return GELU::forward_inplace(bottom_top_blob, opt);
    }

    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int d = bottom_top_blob.d;
    int elempack = bottom_top_blob.elempack;
    int channels = bottom_top_blob.c;
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

            vfloat32m8_t _pLoad = __riscv_vle32_v_f32m8(ptr, vl);

            // cube = x^3
            vfloat32m8_t _cube = __riscv_vfmul_vv_f32m8(_pLoad, _pLoad, vl);
            _cube = __riscv_vfmul_vv_f32m8(_pLoad, _cube, vl);

            // blob = tanh( sqrt(2/pi) * (x + 0.044715 * x^3) )
            vfloat32m8_t _blob = __riscv_vfmadd_vf_f32m8(_cube, 0.044715f, _pLoad, vl);
            _blob = __riscv_vfmul_vf_f32m8(_blob, 0.79788452f, vl);
            _blob = tanh_ps(_blob, vl);

            // y = 0.5 * x * (1 + blob)
            _blob = __riscv_vfadd_vf_f32m8(_blob, 1.f, vl);
            _blob = __riscv_vfmul_vv_f32m8(_blob, _pLoad, vl);
            _blob = __riscv_vfmul_vf_f32m8(_blob, 0.5f, vl);

            __riscv_vse32_v_f32m8(ptr, _blob, vl);

            ptr += vl;
            n -= vl;
        }
#else  // __riscv_vector
        for (int i = 0; i < size; i++)
        {
            // y = 0.5x * (1 + tanh(sqrt(2/Pi) * (x + 0.044715x^3)))
            *ptr = 0.5f * *ptr * (1.0f + tanhf(0.79788452f * (*ptr + 0.044715f * *ptr * *ptr * *ptr)));
            ptr++;
        }
#endif // __riscv_vector
    }

    return 0;
}

} // namespace ncnn
