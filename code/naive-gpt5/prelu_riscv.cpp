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

    int dims = bottom_top_blob.dims;
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;

    int channels = bottom_top_blob.c;
    int elempack = bottom_top_blob.elempack;

    if (dims == 1)
    {
        int size = w * elempack;
        float* ptr = bottom_top_blob;

        if (num_slope > 1)
        {
            const float* slope = slope_data;
#if __riscv_vector
            int n = size;
            int off = 0;
            while (n > 0)
            {
                size_t vl = __riscv_vsetvl_e32m8(n);
                vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr + off, vl);
                vfloat32m8_t _s = __riscv_vle32_v_f32m8(slope + off, vl);
                vfloat32m8_t _pos = __riscv_vfmax_vf_f32m8(_p, 0.f, vl);
                vfloat32m8_t _neg = __riscv_vfmin_vf_f32m8(_p, 0.f, vl);
                vfloat32m8_t _out = __riscv_vfadd_vv_f32m8(_pos, __riscv_vfmul_vv_f32m8(_s, _neg, vl), vl);
                __riscv_vse32_v_f32m8(ptr + off, _out, vl);
                off += vl;
                n -= vl;
            }
#else
            #pragma omp parallel for num_threads(opt.num_threads)
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
            int off = 0;
            while (n > 0)
            {
                size_t vl = __riscv_vsetvl_e32m8(n);
                vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr + off, vl);
                vfloat32m8_t _pos = __riscv_vfmax_vf_f32m8(_p, 0.f, vl);
                vfloat32m8_t _neg = __riscv_vfmin_vf_f32m8(_p, 0.f, vl);
                vfloat32m8_t _s = __riscv_vfmv_v_f_f32m8(slope, vl);
                vfloat32m8_t _out = __riscv_vfadd_vv_f32m8(_pos, __riscv_vfmul_vv_f32m8(_s, _neg, vl), vl);
                __riscv_vse32_v_f32m8(ptr + off, _out, vl);
                off += vl;
                n -= vl;
            }
#else
            #pragma omp parallel for num_threads(opt.num_threads)
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
        int size = w * elempack;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            float* ptr = bottom_top_blob.row(i);

            if (num_slope > 1)
            {
#if __riscv_vector
                if (elempack > 1)
                {
                    // per-pack slope vector constant across width
                    const float* srow = (const float*)slope_data + i * elempack;
                    int j = 0;
                    while (j + elempack - 1 < size)
                    {
                        size_t vl = elempack;
                        vfloat32m1_t _p = __riscv_vle32_v_f32m1(ptr + j, vl);
                        vfloat32m1_t _s = __riscv_vle32_v_f32m1(srow, vl);
                        vfloat32m1_t _pos = __riscv_vfmax_vf_f32m1(_p, 0.f, vl);
                        vfloat32m1_t _neg = __riscv_vfmin_vf_f32m1(_p, 0.f, vl);
                        vfloat32m1_t _out = __riscv_vfadd_vv_f32m1(_pos, __riscv_vfmul_vv_f32m1(_s, _neg, vl), vl);
                        __riscv_vse32_v_f32m1(ptr + j, _out, vl);
                        j += elempack;
                    }
                    for (; j < size; j++)
                    {
                        if (ptr[j] < 0)
                            ptr[j] *= srow[j % elempack];
                    }
                }
                else
                {
                    // elempack == 1, slope per-row scalar
                    float slope = ((const float*)slope_data)[i];
                    int n = size;
                    int off = 0;
                    while (n > 0)
                    {
                        size_t vl = __riscv_vsetvl_e32m8(n);
                        vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr + off, vl);
                        vfloat32m8_t _pos = __riscv_vfmax_vf_f32m8(_p, 0.f, vl);
                        vfloat32m8_t _neg = __riscv_vfmin_vf_f32m8(_p, 0.f, vl);
                        vfloat32m8_t _s = __riscv_vfmv_v_f_f32m8(slope, vl);
                        vfloat32m8_t _out = __riscv_vfadd_vv_f32m8(_pos, __riscv_vfmul_vv_f32m8(_s, _neg, vl), vl);
                        __riscv_vse32_v_f32m8(ptr + off, _out, vl);
                        off += vl;
                        n -= vl;
                    }
                }
#else
                float slope = ((const float*)slope_data)[i];
                for (int j = 0; j < size; j++)
                {
                    if (ptr[j] < 0)
                        ptr[j] *= slope;
                }
#endif // __riscv_vector
            }
            else
            {
                float slope = slope_data[0];
#if __riscv_vector
                int n = size;
                int off = 0;
                while (n > 0)
                {
                    size_t vl = __riscv_vsetvl_e32m8(n);
                    vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr + off, vl);
                    vfloat32m8_t _pos = __riscv_vfmax_vf_f32m8(_p, 0.f, vl);
                    vfloat32m8_t _neg = __riscv_vfmin_vf_f32m8(_p, 0.f, vl);
                    vfloat32m8_t _s = __riscv_vfmv_v_f_f32m8(slope, vl);
                    vfloat32m8_t _out = __riscv_vfadd_vv_f32m8(_pos, __riscv_vfmul_vv_f32m8(_s, _neg, vl), vl);
                    __riscv_vse32_v_f32m8(ptr + off, _out, vl);
                    off += vl;
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
        int size = w * h * elempack;
        const float* slope_ptr = slope_data;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            if (num_slope > 1)
            {
#if __riscv_vector
                if (elempack > 1)
                {
                    const float* sch = (const float*)slope_ptr + q * elempack;
                    int i = 0;
                    while (i + elempack - 1 < size)
                    {
                        size_t vl = elempack;
                        vfloat32m1_t _p = __riscv_vle32_v_f32m1(ptr + i, vl);
                        vfloat32m1_t _s = __riscv_vle32_v_f32m1(sch, vl);
                        vfloat32m1_t _pos = __riscv_vfmax_vf_f32m1(_p, 0.f, vl);
                        vfloat32m1_t _neg = __riscv_vfmin_vf_f32m1(_p, 0.f, vl);
                        vfloat32m1_t _out = __riscv_vfadd_vv_f32m1(_pos, __riscv_vfmul_vv_f32m1(_s, _neg, vl), vl);
                        __riscv_vse32_v_f32m1(ptr + i, _out, vl);
                        i += elempack;
                    }
                    for (; i < size; i++)
                    {
                        if (ptr[i] < 0)
                            ptr[i] *= sch[i % elempack];
                    }
                }
                else
                {
                    float slope = slope_ptr[q];
                    int n = size;
                    int off = 0;
                    while (n > 0)
                    {
                        size_t vl = __riscv_vsetvl_e32m8(n);
                        vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr + off, vl);
                        vfloat32m8_t _pos = __riscv_vfmax_vf_f32m8(_p, 0.f, vl);
                        vfloat32m8_t _neg = __riscv_vfmin_vf_f32m8(_p, 0.f, vl);
                        vfloat32m8_t _s = __riscv_vfmv_v_f_f32m8(slope, vl);
                        vfloat32m8_t _out = __riscv_vfadd_vv_f32m8(_pos, __riscv_vfmul_vv_f32m8(_s, _neg, vl), vl);
                        __riscv_vse32_v_f32m8(ptr + off, _out, vl);
                        off += vl;
                        n -= vl;
                    }
                }
#else
                float slope = slope_ptr[q];
                for (int i = 0; i < size; i++)
                {
                    if (ptr[i] < 0)
                        ptr[i] *= slope;
                }
#endif // __riscv_vector
            }
            else
            {
                float slope = slope_ptr[0];
#if __riscv_vector
                int n = size;
                int off = 0;
                while (n > 0)
                {
                    size_t vl = __riscv_vsetvl_e32m8(n);
                    vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr + off, vl);
                    vfloat32m8_t _pos = __riscv_vfmax_vf_f32m8(_p, 0.f, vl);
                    vfloat32m8_t _neg = __riscv_vfmin_vf_f32m8(_p, 0.f, vl);
                    vfloat32m8_t _s = __riscv_vfmv_v_f_f32m8(slope, vl);
                    vfloat32m8_t _out = __riscv_vfadd_vv_f32m8(_pos, __riscv_vfmul_vv_f32m8(_s, _neg, vl), vl);
                    __riscv_vse32_v_f32m8(ptr + off, _out, vl);
                    off += vl;
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
