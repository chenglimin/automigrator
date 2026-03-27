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

#include "mish_riscv.h"

#if __riscv_vector
#include <riscv_vector.h>
#include "rvv_mathfun.h"
#include "riscv_activation.h"
#endif // __riscv_vector

namespace ncnn {

Mish_riscv::Mish_riscv()
{
#if __riscv_vector
    support_packing = true;
#endif // __riscv_vector
}

int Mish_riscv::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int d = bottom_top_blob.d;
    int channels = bottom_top_blob.c;
    int elempack = bottom_top_blob.elempack;
    int size = w * h * d * elempack;

    // Fallback for scalar path
#if !__riscv_vector
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        float* ptr = bottom_top_blob.channel(q);
        for (int i = 0; i < size; i++)
        {
            const float MISH_THRESHOLD = 20.f;
            float x = ptr[i];
            float y;
            if (x > MISH_THRESHOLD)
                y = x;
            else if (x < -MISH_THRESHOLD)
                y = expf(x);
            else
                y = logf(expf(x) + 1.f);
            ptr[i] = x * tanhf(y);
        }
    }
    return 0;
#else
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        float* ptr = bottom_top_blob.channel(q);
        int n = size;
        while (n > 0)
        {
            // Prefer m8 to utilize wide vectors; vl adapts to VLEN
            size_t vl = __riscv_vsetvl_e32m8(n);
            vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vl);
            // mish(x) = x * tanh(log(1 + exp(x)))
            vfloat32m8_t _exp = exp_ps(_p, vl);
            vfloat32m8_t _log1p = log_ps(__riscv_vfadd_vf_f32m8(_exp, 1.f, vl), vl);
            vfloat32m8_t _tanh = tanh_ps(_log1p, vl);
            _p = __riscv_vfmul_vv_f32m8(_p, _tanh, vl);
            __riscv_vse32_v_f32m8(ptr, _p, vl);
            ptr += vl;
            n -= vl;
        }
    }
    return 0;
#endif // __riscv_vector
}

} // namespace ncnn
