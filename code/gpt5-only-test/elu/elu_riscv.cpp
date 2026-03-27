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

#include "elu_riscv.h"

#if __riscv_vector
#include <riscv_vector.h>
#include "rvv_mathfun.h"
#if __riscv_zvfh
#include "rvv_mathfun_fp16s.h"
#endif
#endif // __riscv_vector

#include "cpu.h"

namespace ncnn {

ELU_riscv::ELU_riscv()
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

int ELU_riscv::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
#if NCNN_ZFH
    int elembits = bottom_top_blob.elembits();

    if (opt.use_fp16_storage && elembits == 16)
    {
        return forward_inplace_fp16s(bottom_top_blob, opt);
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
            // mask for p < 0
            vbool4_t _mask = __riscv_vmflt_vf_f32m8_b4(_p, 0.f, vl);
            vfloat32m8_t _exp = exp_ps(_p, vl);
            _exp = __riscv_vfsub_vf_f32m8(_exp, 1.f, vl);
            _exp = __riscv_vfmul_vf_f32m8(_exp, alpha, vl);
            // select: p<0 ? alpha*(exp(p)-1) : p
            vfloat32m8_t _res = __riscv_vmerge_vvm_f32m8(_p, _exp, _mask, vl);
            __riscv_vse32_v_f32m8(ptr, _res, vl);

            ptr += vl;
            n -= vl;
        }
#else  // __riscv_vector
        for (int i = 0; i < size; i++)
        {
            if (*ptr < 0.f)
                *ptr = alpha * (expf(*ptr) - 1.f);
            ptr++;
        }
#endif // __riscv_vector
    }

    return 0;
}

#if NCNN_ZFH
int ELU_riscv::forward_inplace_fp16s(Mat& bottom_top_blob, const Option& opt) const
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
        __fp16* ptr = bottom_top_blob.channel(q);

#if __riscv_zvfh
        int n = size;
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e16m8(n);

            vfloat16m8_t _p = __riscv_vle16_v_f16m8(ptr, vl);
            vbool2_t _mask = __riscv_vmflt_vf_f16m8_b2(_p, (__fp16)0.f, vl);
            vfloat16m8_t _exp = exp_ps(_p, vl);
            _exp = __riscv_vfsub_vf_f16m8(_exp, (__fp16)1.f, vl);
            _exp = __riscv_vfmul_vf_f16m8(_exp, (__fp16)alpha, vl);
            vfloat16m8_t _res = __riscv_vmerge_vvm_f16m8(_p, _exp, _mask, vl);
            __riscv_vse16_v_f16m8(ptr, _res, vl);

            ptr += vl;
            n -= vl;
        }
#else  // __riscv_zvfh
        for (int i = 0; i < size; i++)
        {
            float v = (float)*ptr;
            if (v < 0.f)
                *ptr = (__fp16)(alpha * (expf(v) - 1.f));
            ptr++;
        }
#endif // __riscv_zvfh
    }

    return 0;
}
#endif // NCNN_ZFH

} // namespace ncnn
