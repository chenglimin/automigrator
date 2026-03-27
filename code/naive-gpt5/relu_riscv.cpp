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
#include "riscv_activation.h"
#endif // __riscv_vector

#include "cpu.h"

namespace ncnn {

ReLU_riscv::ReLU_riscv()
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

int ReLU_riscv::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
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

    if (slope == 0.f)
    {
        // pure relu
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
                    ptr[i] = 0;
            }
#endif // __riscv_vector
        }
    }
    else
    {
        // leaky relu
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
                vbool4_t _lower = __riscv_vmflt_vf_f32m8_b4(_p, 0.f, vl);
                _p = __riscv_vfmul_vf_f32m8_mu(_lower, _p, _p, slope, vl);
                __riscv_vse32_v_f32m8(ptr, _p, vl);
                ptr += vl;
                n -= vl;
            }
#else
            for (int i = 0; i < size; i++)
            {
                if (ptr[i] < 0)
                    ptr[i] *= slope;
            }
#endif // __riscv_vector
        }
    }

    return 0;
}

#if NCNN_ZFH
int ReLU_riscv::forward_inplace_fp16s(Mat& bottom_top_blob, const Option& opt) const
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
            if (slope == 0.f)
            {
                _p = __riscv_vfmax_vf_f16m8(_p, (__fp16)0.f, vl);
            }
            else
            {
                vbool2_t _lower = __riscv_vmflt_vf_f16m8_b2(_p, (__fp16)0.f, vl);
                _p = __riscv_vfmul_vf_f16m8_mu(_lower, _p, _p, (__fp16)slope, vl);
            }
            __riscv_vse16_v_f16m8(ptr, _p, vl);
            ptr += vl;
            n -= vl;
        }
#else
        for (int i = 0; i < size; i++)
        {
            if (slope == 0.f)
                ptr[i] = ptr[i] < (__fp16)0.f ? (__fp16)0.f : ptr[i];
            else
                if (ptr[i] < (__fp16)0.f) ptr[i] = (__fp16)((float)ptr[i] * slope);
        }
#endif // __riscv_zvfh
    }

    return 0;
}

int ReLU_riscv::forward_inplace_fp16sa(Mat& bottom_top_blob, const Option& opt) const
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
            if (slope == 0.f)
            {
                _p = __riscv_vfmax_vf_f16m8(_p, (__fp16)0.f, vl);
            }
            else
            {
                vbool2_t _lower = __riscv_vmflt_vf_f16m8_b2(_p, (__fp16)0.f, vl);
                _p = __riscv_vfmul_vf_f16m8_mu(_lower, _p, _p, (__fp16)slope, vl);
            }
            __riscv_vse16_v_f16m8(ptr, _p, vl);
            ptr += vl;
            n -= vl;
        }
#else
        for (int i = 0; i < size; i++)
        {
            if (slope == 0.f)
                ptr[i] = ptr[i] < (__fp16)0.f ? (__fp16)0.f : ptr[i];
            else
                if (ptr[i] < (__fp16)0.f) ptr[i] *= (__fp16)slope;
        }
#endif // __riscv_zvfh
    }

    return 0;
}
#endif // NCNN_ZFH

} // namespace ncnn
