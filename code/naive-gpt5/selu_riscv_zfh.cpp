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
#if __riscv_zvfh
#include "rvv_mathfun_fp16s.h"
#endif
#endif // __riscv_vector

namespace ncnn {

#if NCNN_ZFH
int SELU_riscv::forward_inplace_fp16s(Mat& bottom_top_blob, const Option& opt) const
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
            size_t vl = __riscv_vsetvl_e16m4(n);

            vfloat32m8_t _p = __riscv_vfwcvt_f_f_v_f32m8(__riscv_vle16_v_f16m4(ptr, vl), vl);

            vbool4_t _is_pos = __riscv_vmfgt_vf_f32m8_b4(_p, 0.f, vl);
            vfloat32m8_t _pos = __riscv_vmerge_vvm_f32m8(__riscv_vfmv_v_f_f32m8(0.f, vl), _p, _is_pos, vl);
            vfloat32m8_t _neg = __riscv_vmerge_vvm_f32m8(_p, __riscv_vfmv_v_f_f32m8(0.f, vl), _is_pos, vl);

            vfloat32m8_t _blob = exp_ps(_neg, vl);
            _blob = __riscv_vfsub_vf_f32m8(_blob, 1.f, vl);
            _blob = __riscv_vfmul_vf_f32m8(_blob, alpha, vl);
            _blob = __riscv_vfmul_vf_f32m8(__riscv_vfadd_vv_f32m8(_pos, _blob, vl), lambda, vl);

            __riscv_vse16_v_f16m4(ptr, __riscv_vfncvt_f_f_w_f16m4(_blob, vl), vl);

            ptr += vl;
            n -= vl;
        }
#else  // __riscv_zvfh
        for (int i = 0; i < size; i++)
        {
            float v = (float)*ptr;
            if (v < 0)
                *ptr = (__fp16)((expf(v) - 1.f) * alpha * lambda);
            else
                *ptr = (__fp16)(v * lambda);
            ptr++;
        }
#endif // __riscv_zvfh
    }

    return 0;
}

int SELU_riscv::forward_inplace_fp16sa(Mat& bottom_top_blob, const Option& opt) const
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

            vbool2_t _is_pos = __riscv_vmfgt_vf_f16m8_b2(_p, (__fp16)0.f, vl);
            vfloat16m8_t _pos = __riscv_vmerge_vvm_f16m8(__riscv_vfmv_v_f_f16m8((__fp16)0.f, vl), _p, _is_pos, vl);
            vfloat16m8_t _neg = __riscv_vmerge_vvm_f16m8(_p, __riscv_vfmv_v_f_f16m8((__fp16)0.f, vl), _is_pos, vl);

            vfloat16m8_t _blob = exp_ps(_neg, vl);
            _blob = __riscv_vfsub_vf_f16m8(_blob, (__fp16)1.f, vl);
            _blob = __riscv_vfmul_vf_f16m8(_blob, (__fp16)alpha, vl);
            _blob = __riscv_vfmul_vf_f16m8(__riscv_vfadd_vv_f16m8(_pos, _blob, vl), (__fp16)lambda, vl);

            __riscv_vse16_v_f16m8(ptr, _blob, vl);

            ptr += vl;
            n -= vl;
        }
#else  // __riscv_zvfh
        for (int i = 0; i < size; i++)
        {
            float v = (float)*ptr;
            if (v < 0)
                *ptr = (__fp16)((expf(v) - 1.f) * alpha * lambda);
            else
                *ptr = (__fp16)(v * lambda);
            ptr++;
        }
#endif // __riscv_zvfh
    }

    return 0;
}
#endif // NCNN_ZFH

} // namespace ncnn
