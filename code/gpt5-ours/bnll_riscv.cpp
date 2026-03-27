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

#include "bnll_riscv.h"

#if __riscv_vector
#include <riscv_vector.h>
#include "rvv_mathfun.h"
#endif // __riscv_vector

#include "cpu.h"

namespace ncnn {

BNLL_riscv::BNLL_riscv()
{
#if __riscv_vector
    support_packing = true;
#endif // __riscv_vector
#if NCNN_ZFH
#if __riscv_vector
    support_fp16_storage = cpu_support_riscv_zvfh();
#else
    support_fp16_storage = cpu_support_riscv_zfh();
#endif
#endif
}

int BNLL_riscv::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
#if NCNN_ZFH
    int elembits = bottom_top_blob.elembits();
    if (opt.use_fp16_storage && elembits == 16)
    {
        // No fp16 specialization for BNLL in this migration. Fallback to base implementation.
        return BNLL::forward_inplace(bottom_top_blob, opt);
    }
#endif

    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int d = bottom_top_blob.d;
    int channels = bottom_top_blob.c;
    int elempack = bottom_top_blob.elempack;
    int size = w * h * d * elempack;

    // Packn behavior: rely on vsetvl to adapt all VLEN (128/256/512 bits). See packn_spec.txt.

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
            // BNLL: x > 0 ? x + log(1 + exp(-x)) : log(1 + exp(x))
            vbool4_t _gt0 = __riscv_vmfgt_vf_f32m8_b4(_p, 0.f, vl);

            vfloat32m8_t _abs_p = __riscv_vfsgnjx_vv_f32m8(_p, _p, vl); // |x| using sign xor
            vfloat32m8_t _neg_abs = __riscv_vfneg_v_f32m8(_abs_p, vl);
            vfloat32m8_t _tmp = log_ps(__riscv_vfadd_vf_f32m8(exp_ps(_neg_abs, vl), 1.f, vl), vl);

            // masked add: when x>0, result = tmp + x ; else = tmp
            vfloat32m8_t _res = _tmp;
            _res = __riscv_vfadd_vv_f32m8_mu(_gt0, _res, _res, _p, vl);

            __riscv_vse32_v_f32m8(ptr, _res, vl);
            ptr += vl;
            n -= vl;
        }
#else  // __riscv_vector
        for (int i = 0; i < size; i++)
        {
            if (*ptr > 0)
                *ptr = *ptr + logf(1.f + expf(-(*ptr)));
            else
                *ptr = logf(1.f + expf(*ptr));
            ptr++;
        }
#endif // __riscv_vector
    }

    return 0;
}

} // namespace ncnn
