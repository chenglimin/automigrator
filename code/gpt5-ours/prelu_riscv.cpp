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

#include "prelu_riscv.h"

#if __riscv_vector
#include <riscv_vector.h>
#endif // __riscv_vector

#include "cpu.h"

namespace ncnn {

PReLU_riscv::PReLU_riscv()
{
#if __riscv_vector
    support_packing = true;
#endif // __riscv_vector
}

int PReLU_riscv::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int d = bottom_top_blob.d;
    int channels = bottom_top_blob.c;
    int elempack = bottom_top_blob.elempack;
    int dims = bottom_top_blob.dims;

#if __riscv_vector
    // Prefer m8 types for performance per RVV 1.0 guidance
#endif

    if (dims == 1)
    {
        const int size = w * elempack;
        float* ptr = bottom_top_blob;

        if (num_slope > 1)
        {
            const float* slope = slope_data;
#if __riscv_vector
            int n = size;
            while (n > 0)
            {
                size_t vl = __riscv_vsetvl_e32m8(n);
                vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vl);
                vfloat32m8_t _s = __riscv_vle32_v_f32m8(slope, vl);
                vbool4_t _mask = __riscv_vmflt_vf_f32m8_b4(_p, 0.f, vl);
                vfloat32m8_t _neg = __riscv_vfmul_vv_f32m8(_p, _s, vl);
                vfloat32m8_t _res = __riscv_vmerge_vvm_f32m8(_p, _neg, _mask, vl);
                __riscv_vse32_v_f32m8(ptr, _res, vl);
                ptr += vl;
                slope += vl;
                n -= vl;
            }
#else
            for (int i = 0; i < size; i++)
            {
                if (ptr[i] < 0)
                    ptr[i] *= slope[i];
            }
#endif // __riscv_vector
        }
        else
        {
            const float slope = slope_data[0];
#if __riscv_vector
            int n = size;
            while (n > 0)
            {
                size_t vl = __riscv_vsetvl_e32m8(n);
                vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vl);
                vbool4_t _mask = __riscv_vmflt_vf_f32m8_b4(_p, 0.f, vl);
                vfloat32m8_t _neg = __riscv_vfmul_vf_f32m8(_p, slope, vl);
                vfloat32m8_t _res = __riscv_vmerge_vvm_f32m8(_p, _neg, _mask, vl);
                __riscv_vse32_v_f32m8(ptr, _res, vl);
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

    if (dims == 2)
    {
        const int size = w * elempack;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            float* ptr = bottom_top_blob.row(i);

            if (num_slope > 1)
            {
                if (elempack == 1)
                {
                    const float slope = slope_data[i];
#if __riscv_vector
                    int n = size;
                    while (n > 0)
                    {
                        size_t vl = __riscv_vsetvl_e32m8(n);
                        vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vl);
                        vbool4_t _mask = __riscv_vmflt_vf_f32m8_b4(_p, 0.f, vl);
                        vfloat32m8_t _neg = __riscv_vfmul_vf_f32m8(_p, slope, vl);
                        vfloat32m8_t _res = __riscv_vmerge_vvm_f32m8(_p, _neg, _mask, vl);
                        __riscv_vse32_v_f32m8(ptr, _res, vl);
                        ptr += vl;
                        n -= vl;
                    }
#else
                    for (int j = 0; j < size; j++)
                    {
                        if (ptr[j] < 0)
                            ptr[j] *= slope;
                    }
#endif // __riscv_vector
                }
                else
                {
                    // elempack > 1, slope per lane
                    const float* slope_vec = (const float*)slope_data + i * elempack;
                    for (int j = 0; j < size; j++)
                    {
                        float s = slope_vec[j % elempack];
                        if (ptr[j] < 0)
                            ptr[j] *= s;
                    }
                }
            }
            else
            {
                const float slope = slope_data[0];
#if __riscv_vector
                int n = size;
                while (n > 0)
                {
                    size_t vl = __riscv_vsetvl_e32m8(n);
                    vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vl);
                    vbool4_t _mask = __riscv_vmflt_vf_f32m8_b4(_p, 0.f, vl);
                    vfloat32m8_t _neg = __riscv_vfmul_vf_f32m8(_p, slope, vl);
                    vfloat32m8_t _res = __riscv_vmerge_vvm_f32m8(_p, _neg, _mask, vl);
                    __riscv_vse32_v_f32m8(ptr, _res, vl);
                    ptr += vl;
                    n -= vl;
                }
#else
                for (int j = 0; j < size; j++)
                {
                    if (ptr[j] < 0)
                        ptr[j] *= slope;
                }
#endif // __riscv_vector
            }
        }
    }

    if (dims == 3)
    {
        const int size = w * h * elempack;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            if (num_slope > 1)
            {
                if (elempack == 1)
                {
                    const float slope = slope_data[q];
#if __riscv_vector
                    int n = size;
                    while (n > 0)
                    {
                        size_t vl = __riscv_vsetvl_e32m8(n);
                        vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vl);
                        vbool4_t _mask = __riscv_vmflt_vf_f32m8_b4(_p, 0.f, vl);
                        vfloat32m8_t _neg = __riscv_vfmul_vf_f32m8(_p, slope, vl);
                        vfloat32m8_t _res = __riscv_vmerge_vvm_f32m8(_p, _neg, _mask, vl);
                        __riscv_vse32_v_f32m8(ptr, _res, vl);
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
                else
                {
                    // elempack > 1, slope per lane
                    const float* slope_vec = (const float*)slope_data + q * elempack;
                    for (int i = 0; i < size; i++)
                    {
                        float s = slope_vec[i % elempack];
                        if (ptr[i] < 0)
                            ptr[i] *= s;
                    }
                }
            }
            else
            {
                const float slope = slope_data[0];
#if __riscv_vector
                int n = size;
                while (n > 0)
                {
                    size_t vl = __riscv_vsetvl_e32m8(n);
                    vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vl);
                    vbool4_t _mask = __riscv_vmflt_vf_f32m8_b4(_p, 0.f, vl);
                    vfloat32m8_t _neg = __riscv_vfmul_vf_f32m8(_p, slope, vl);
                    vfloat32m8_t _res = __riscv_vmerge_vvm_f32m8(_p, _neg, _mask, vl);
                    __riscv_vse32_v_f32m8(ptr, _res, vl);
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
    }

    return 0;
}

} // namespace ncnn
