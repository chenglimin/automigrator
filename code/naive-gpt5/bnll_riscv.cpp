// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
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
        return forward_inplace_fp16s(bottom_top_blob, opt);
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
            // mask for p > 0
            vbool4_t mask_pos = __riscv_vmfgt_vf_f32m8_b4(_p, 0.f, vl);
            // abs(p)
            vfloat32m8_t _abs_p = __riscv_vfsgnj_vf_f32m8(_p, 1.f, vl);
            // tmp = log(1 + exp(-abs(p)))
            vfloat32m8_t _tmp = log_ps(__riscv_vfadd_vf_f32m8(exp_ps(__riscv_vfrsub_vf_f32m8(_abs_p, 0.f, vl), vl), 1.f, vl), vl);
            // x = mask_pos ? p : 0
            vfloat32m8_t _x = __riscv_vfmv_v_f_f32m8(0.f, vl);
            _x = __riscv_vmerge_vvm_f32m8(_x, _p, mask_pos, vl);
            // p = x + tmp
            _p = __riscv_vfadd_vv_f32m8(_x, _tmp, vl);
            __riscv_vse32_v_f32m8(ptr, _p, vl);

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
