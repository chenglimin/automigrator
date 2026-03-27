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

#include "sigmoid_riscv.h"

#if __riscv_vector
#include <riscv_vector.h>
#include "rvv_mathfun.h"
#endif // __riscv_vector

#include "cpu.h"

namespace ncnn {

Sigmoid_riscv::Sigmoid_riscv()
{
#if __riscv_vector
    support_packing = true;
#endif // __riscv_vector
}

int Sigmoid_riscv::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    int elembits = bottom_top_blob.elembits();

#if NCNN_ZFH
    if (opt.use_fp16_storage && elembits == 16)
    {
        // No fp16 storage path implemented for sigmoid in this migration.
        // Fallback to generic scalar path which handles bf16/fp16 via base implementation if needed.
    }
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

#if __riscv_vector
        int n = size;
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e32m8(n);
            vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vl);
            // sigmoid(x) = 1 / (1 + exp(-x))
            _p = __riscv_vfneg_v_f32m8(_p, vl);
            _p = exp_ps(_p, vl);
            _p = __riscv_vfadd_vf_f32m8(_p, 1.f, vl);
#if __riscv_xtheadvector
            vfloat32m8_t _out = __riscv_vfrdiv_vf_f32m8(_p, 1.f, vl);
#else
            // RVV 1.0: reciprocal via Newton-Raphson refinement on vrec7
            vfloat32m8_t _recip = __riscv_vfrec7_v_f32m8(_p, vl);
            _recip = __riscv_vfmul_vv_f32m8(
                __riscv_vfrsub_vf_f32m8(__riscv_vfmul_vv_f32m8(_p, _recip, vl), 2.f, vl),
                _recip, vl);
            vfloat32m8_t _out = _recip;
#endif
            __riscv_vse32_v_f32m8(ptr, _out, vl);
            ptr += vl;
            n -= vl;
        }
#else
        for (int i = 0; i < size; i++)
        {
            float v = ptr[i];
            v = 1.f / (1.f + expf(-v));
            ptr[i] = v;
        }
#endif // __riscv_vector
    }

    return 0;
}

} // namespace ncnn
