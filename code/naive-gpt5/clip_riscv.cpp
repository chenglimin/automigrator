// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2018 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "clip_riscv.h"

#include <float.h>

#if __riscv_vector
#include <riscv_vector.h>
#endif // __riscv_vector

#include "cpu.h"

namespace ncnn {

Clip_riscv::Clip_riscv()
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

int Clip_riscv::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
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
            vfloat32m8_t _minv = __riscv_vfmv_v_f_f32m8(min, vl);
            vfloat32m8_t _maxv = __riscv_vfmv_v_f_f32m8(max, vl);
            _p = __riscv_vfmax_vv_f32m8(_p, _minv, vl);
            _p = __riscv_vfmin_vv_f32m8(_p, _maxv, vl);
            __riscv_vse32_v_f32m8(ptr, _p, vl);

            ptr += vl;
            n -= vl;
        }
#else
        for (int i = 0; i < size; i++)
        {
            if (*ptr < min) *ptr = min;
            if (*ptr > max) *ptr = max;
            ptr++;
        }
#endif // __riscv_vector
    }

    return 0;
}

#if NCNN_ZFH
int Clip_riscv::forward_inplace_fp16s(Mat& bottom_top_blob, const Option& opt) const
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
            vfloat16m8_t _minv = __riscv_vfmv_v_f_f16m8((__fp16)min, vl);
            vfloat16m8_t _maxv = __riscv_vfmv_v_f_f16m8((__fp16)max, vl);
            _p = __riscv_vfmax_vv_f16m8(_p, _minv, vl);
            _p = __riscv_vfmin_vv_f16m8(_p, _maxv, vl);
            __riscv_vse16_v_f16m8(ptr, _p, vl);

            ptr += vl;
            n -= vl;
        }
#else
        for (int i = 0; i < size; i++)
        {
            float v = (float)*ptr;
            if (v < min) v = min;
            if (v > max) v = max;
            *ptr = (__fp16)v;
            ptr++;
        }
#endif // __riscv_zvfh
    }

    return 0;
}
#endif // NCNN_ZFH

} // namespace ncnn
