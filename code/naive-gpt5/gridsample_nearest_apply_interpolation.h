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

#include "cpu.h"

#if __riscv_vector
#include <riscv_vector.h>
#endif // __riscv_vector

static void gridsample_nearest_apply_interpolation_p1(const Mat& src, Mat& dst, const Mat& offset_value, const Option& opt)
{
    const int channels = dst.c;
    const int outw = dst.w;
    const int outh = dst.h;
    const int outd = dst.d;
    const int grid_size = outw * outh * outd;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        const float* srcptr = src.channel(q);
        float* dstptr = dst.channel(q);

        const int* offset_ptr = (const int*)offset_value.channel(0);

        for (int i = 0; i < grid_size; i++)
        {
            float v = offset_ptr[0] >= 0 ? *(srcptr + offset_ptr[0]) : 0.f;
            offset_ptr++;
            *dstptr++ = v;
        }
    }
}

#if __riscv_vector
static inline size_t __ncnn_vl_from_elempack(const Mat& dst)
{
    return __riscv_vsetvl_e32m1(dst.elempack);
}

static void gridsample_nearest_apply_interpolation_p4(const Mat& src, Mat& dst, const Mat& offset_value, const Option& opt)
{
    const int channels = dst.c;
    const int outw = dst.w;
    const int outh = dst.h;
    const int outd = dst.d;
    const int grid_size = outw * outh * outd;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        const float* srcptr = src.channel(q);
        float* dstptr = dst.channel(q);
        const int* offset_ptr = (const int*)offset_value.channel(0);
        const size_t vl = __ncnn_vl_from_elempack(dst);

        for (int i = 0; i < grid_size; i++)
        {
            vfloat32m1_t v = offset_ptr[0] >= 0 ? __riscv_vle32_v_f32m1(srcptr + offset_ptr[0], vl) : __riscv_vfmv_v_f_f32m1(0.f, vl);
            offset_ptr++;
            __riscv_vse32_v_f32m1(dstptr, v, vl);
            dstptr += vl;
        }
    }
}

static void gridsample_nearest_apply_interpolation_p8(const Mat& src, Mat& dst, const Mat& offset_value, const Option& opt)
{
    gridsample_nearest_apply_interpolation_p4(src, dst, offset_value, opt);
}

static void gridsample_nearest_apply_interpolation_p16(const Mat& src, Mat& dst, const Mat& offset_value, const Option& opt)
{
    gridsample_nearest_apply_interpolation_p4(src, dst, offset_value, opt);
}

#endif // __riscv_vector
