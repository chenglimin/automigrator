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

#include "selu_riscv.h"

#if __riscv_vector
#include <riscv_vector.h>
#include "rvv_mathfun.h"
#endif // __riscv_vector

#include "cpu.h"

namespace ncnn {

SELU_riscv::SELU_riscv()
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

int SELU_riscv::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
#if NCNN_ZFH
    int elembits = bottom_top_blob.elembits();

    if (opt.use_fp16_storage && elembits == 16)
    {
        if (opt.use_fp16_arithmetic)
            return forward_inplace_fp16sa(bottom_top_blob, opt);
        else
            return forward_inplace_fp16s(bottom_top_blob, opt);
    }
#endif

    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int d = bottom_top_blob.d;
    int channels = bottom_top_blob.c;
    int elempack = bottom_top_blob.elempack;
    int size = w * h * d * elempack;

    float alphaxlambda = alpha * lambda;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        float* ptr = bottom_top_blob.channel(q);

#if __riscv_vector
        int n = size;
        while (n > 0)
        {
            // Use dynamic vl to adapt to different VLEN
            size_t vl = __riscv_vsetvl_e32m8(n);

            vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vl);
            // mask for x > 0
            vbool4_t _gtmask = __riscv_vmfgt_vf_f32m8_b4(_p, 0.f, vl);

            // pos path: lambda * x
            vfloat32m8_t _lambda = __riscv_vfmv_v_f_f32m8(lambda, vl);
            vfloat32m8_t _pos = __riscv_vfmul_vv_f32m8(_p, _lambda, vl);

            // neg path: alphaxlambda * (exp(x) - 1)
            vfloat32m8_t _one = __riscv_vfmv_v_f_f32m8(1.f, vl);
            vfloat32m8_t _negblob = exp_ps(_p, vl);
            _negblob = __riscv_vfsub_vv_f32m8(_negblob, _one, vl);
            vfloat32m8_t _axl = __riscv_vfmv_v_f_f32m8(alphaxlambda, vl);
            _negblob = __riscv_vfmul_vv_f32m8(_negblob, _axl, vl);

            // merge by mask: if x > 0 use pos path, else neg path
            vfloat32m8_t _res = __riscv_vmerge_vvm_f32m8(_negblob, _pos, _gtmask, vl);
            __riscv_vse32_v_f32m8(ptr, _res, vl);

            ptr += vl;
            n -= vl;
        }
#else  // __riscv_vector
        for (int i = 0; i < size; i++)
        {
            if (*ptr < 0.f)
                *ptr = (expf(*ptr) - 1.f) * alphaxlambda;
            else
                *ptr *= lambda;
            ptr++;
        }
#endif // __riscv_vector
    }

    return 0;
}

#if NCNN_ZFH
int SELU_riscv::forward_inplace_fp16s(Mat& bottom_top_blob, const Option& opt) const
{
    // Fallback to generic path for fp16 storage without fp16 arithmetic
    return SELU::forward_inplace(bottom_top_blob, opt);
}

int SELU_riscv::forward_inplace_fp16sa(Mat& bottom_top_blob, const Option& opt) const
{
#if __riscv_vector && __riscv_zvfh
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int d = bottom_top_blob.d;
    int channels = bottom_top_blob.c;
    int elempack = bottom_top_blob.elempack;
    int size = w * h * d * elempack;

    __fp16 halpha = (__fp16)alpha;
    __fp16 hlambda = (__fp16)lambda;
    __fp16 halphaxlambda = (__fp16)(alpha * lambda);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        __fp16* ptr = (__fp16*)bottom_top_blob.channel(q);
        int n = size;
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e16m8(n);
            vfloat16m8_t _p = __riscv_vle16_v_f16m8(ptr, vl);
            vbool2_t _lemask = __riscv_vmfle_vf_f16m8_b2(_p, (__fp16)0, vl);
            vfloat16m8_t _lambda = __riscv_vfmv_v_f_f16m8(hlambda, vl);
            vfloat16m8_t _pos = __riscv_vfmul_vv_f16m8(_p, _lambda, vl);
            vfloat16m8_t _one = __riscv_vfmv_v_f_f16m8((__fp16)1, vl);
            vfloat16m8_t _negblob = exp_ps(_p, vl);
            _negblob = __riscv_vfsub_vv_f16m8(_negblob, _one, vl);
            vfloat16m8_t _axl = __riscv_vfmv_v_f_f16m8(halphaxlambda, vl);
            _negblob = __riscv_vfmul_vv_f16m8(_negblob, _axl, vl);
            vfloat16m8_t _res = __riscv_vmerge_vvm_f16m8(_negblob, _pos, _lemask, vl);
            __riscv_vse16_v_f16m8(ptr, _res, vl);
            ptr += vl;
            n -= vl;
        }
    }
    return 0;
#else
    return SELU::forward_inplace(bottom_top_blob, opt);
#endif
}
#endif // NCNN_ZFH

} // namespace ncnn
