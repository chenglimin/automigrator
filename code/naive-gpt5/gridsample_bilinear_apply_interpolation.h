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

static void gridsample_2d_bilinear_apply_interpolation_p1(const Mat& src, Mat& dst, const Mat& offset_value, const Option& opt)
{
    const int channels = dst.c;
    const int outw = dst.w;
    const int outh = dst.h;
    const int grid_size = outw * outh;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        const float* srcptr = src.channel(q);
        float* dstptr = dst.channel(q);
        const float* offset_value_ptr = offset_value.channel(0);

        for (int i = 0; i < grid_size; i++)
        {
            const int* offset_ptr = (const int*)offset_value_ptr;
            const float* value_ptr = offset_value_ptr + 4;

            float v00 = offset_ptr[0] >= 0 ? *(srcptr + offset_ptr[0]) : 0.f;
            float v01 = offset_ptr[1] >= 0 ? *(srcptr + offset_ptr[1]) : 0.f;
            float v10 = offset_ptr[2] >= 0 ? *(srcptr + offset_ptr[2]) : 0.f;
            float v11 = offset_ptr[3] >= 0 ? *(srcptr + offset_ptr[3]) : 0.f;

            float alpha = value_ptr[0];
            float beta = value_ptr[1];

            float v0 = v00 * (1.f - alpha) + v01 * alpha;
            float v1 = v10 * (1.f - alpha) + v11 * alpha;

            *dstptr = v0 * (1.f - beta) + v1 * beta;

            dstptr += 1;
            offset_value_ptr += 6;
        }
    }
}

#if __riscv_vector
static inline size_t __ncnn_vl_from_elempack(const Mat& dst)
{
    // use elempack as vector length
    return __riscv_vsetvl_e32m1(dst.elempack);
}

static void gridsample_2d_bilinear_apply_interpolation_p4(const Mat& src, Mat& dst, const Mat& offset_value, const Option& opt)
{
    const int channels = dst.c;
    const int outw = dst.w;
    const int outh = dst.h;
    const int grid_size = outw * outh;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        const float* srcptr = src.channel(q);
        float* dstptr = dst.channel(q);
        const float* offset_value_ptr = offset_value.channel(0);
        const size_t vl = __ncnn_vl_from_elempack(dst);

        for (int i = 0; i < grid_size; i++)
        {
            const int* offset_ptr = (const int*)offset_value_ptr;
            const float* value_ptr = offset_value_ptr + 4;

            vfloat32m1_t v00 = offset_ptr[0] >= 0 ? __riscv_vle32_v_f32m1(srcptr + offset_ptr[0], vl) : __riscv_vfmv_v_f_f32m1(0.f, vl);
            vfloat32m1_t v01 = offset_ptr[1] >= 0 ? __riscv_vle32_v_f32m1(srcptr + offset_ptr[1], vl) : __riscv_vfmv_v_f_f32m1(0.f, vl);
            vfloat32m1_t v10 = offset_ptr[2] >= 0 ? __riscv_vle32_v_f32m1(srcptr + offset_ptr[2], vl) : __riscv_vfmv_v_f_f32m1(0.f, vl);
            vfloat32m1_t v11 = offset_ptr[3] >= 0 ? __riscv_vle32_v_f32m1(srcptr + offset_ptr[3], vl) : __riscv_vfmv_v_f_f32m1(0.f, vl);

            float alpha = value_ptr[0];
            float beta = value_ptr[1];

            vfloat32m1_t v0 = __riscv_vfmacc_vf_f32m1(__riscv_vfmul_vf_f32m1(v00, 1.f - alpha, vl), alpha, v01, vl);
            vfloat32m1_t v1 = __riscv_vfmacc_vf_f32m1(__riscv_vfmul_vf_f32m1(v10, 1.f - alpha, vl), alpha, v11, vl);
            vfloat32m1_t outv = __riscv_vfmacc_vf_f32m1(__riscv_vfmul_vf_f32m1(v0, 1.f - beta, vl), beta, v1, vl);

            __riscv_vse32_v_f32m1(dstptr, outv, vl);
            dstptr += vl;
            offset_value_ptr += 6;
        }
    }
}

static void gridsample_2d_bilinear_apply_interpolation_p8(const Mat& src, Mat& dst, const Mat& offset_value, const Option& opt)
{
    gridsample_2d_bilinear_apply_interpolation_p4(src, dst, offset_value, opt);
}

static void gridsample_2d_bilinear_apply_interpolation_p16(const Mat& src, Mat& dst, const Mat& offset_value, const Option& opt)
{
    gridsample_2d_bilinear_apply_interpolation_p4(src, dst, offset_value, opt);
}

static void gridsample_3d_bilinear_apply_interpolation_p4(const Mat& src, Mat& dst, const Mat& offset_value, const Option& opt)
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
        const float* offset_value_ptr = offset_value.channel(0);
        const size_t vl = __ncnn_vl_from_elempack(dst);

        for (int i = 0; i < grid_size; i++)
        {
            const int* offset_ptr = (const int*)offset_value_ptr;
            const float* value_ptr = offset_value_ptr + 8;

            vfloat32m1_t v000 = offset_ptr[0] >= 0 ? __riscv_vle32_v_f32m1(srcptr + offset_ptr[0], vl) : __riscv_vfmv_v_f_f32m1(0.f, vl);
            vfloat32m1_t v001 = offset_ptr[1] >= 0 ? __riscv_vle32_v_f32m1(srcptr + offset_ptr[1], vl) : __riscv_vfmv_v_f_f32m1(0.f, vl);
            vfloat32m1_t v010 = offset_ptr[2] >= 0 ? __riscv_vle32_v_f32m1(srcptr + offset_ptr[2], vl) : __riscv_vfmv_v_f_f32m1(0.f, vl);
            vfloat32m1_t v011 = offset_ptr[3] >= 0 ? __riscv_vle32_v_f32m1(srcptr + offset_ptr[3], vl) : __riscv_vfmv_v_f_f32m1(0.f, vl);

            vfloat32m1_t v100 = offset_ptr[4] >= 0 ? __riscv_vle32_v_f32m1(srcptr + offset_ptr[4], vl) : __riscv_vfmv_v_f_f32m1(0.f, vl);
            vfloat32m1_t v101 = offset_ptr[5] >= 0 ? __riscv_vle32_v_f32m1(srcptr + offset_ptr[5], vl) : __riscv_vfmv_v_f_f32m1(0.f, vl);
            vfloat32m1_t v110 = offset_ptr[6] >= 0 ? __riscv_vle32_v_f32m1(srcptr + offset_ptr[6], vl) : __riscv_vfmv_v_f_f32m1(0.f, vl);
            vfloat32m1_t v111 = offset_ptr[7] >= 0 ? __riscv_vle32_v_f32m1(srcptr + offset_ptr[7], vl) : __riscv_vfmv_v_f_f32m1(0.f, vl);

            float alpha = value_ptr[0];
            float beta = value_ptr[1];
            float gamma = value_ptr[2];

            vfloat32m1_t v00 = __riscv_vfmacc_vf_f32m1(__riscv_vfmul_vf_f32m1(v000, 1.f - alpha, vl), alpha, v001, vl);
            vfloat32m1_t v01 = __riscv_vfmacc_vf_f32m1(__riscv_vfmul_vf_f32m1(v010, 1.f - alpha, vl), alpha, v011, vl);
            vfloat32m1_t v10 = __riscv_vfmacc_vf_f32m1(__riscv_vfmul_vf_f32m1(v100, 1.f - alpha, vl), alpha, v101, vl);
            vfloat32m1_t v11 = __riscv_vfmacc_vf_f32m1(__riscv_vfmul_vf_f32m1(v110, 1.f - alpha, vl), alpha, v111, vl);

            vfloat32m1_t v0 = __riscv_vfmacc_vf_f32m1(__riscv_vfmul_vf_f32m1(v00, 1.f - beta, vl), beta, v01, vl);
            vfloat32m1_t v1 = __riscv_vfmacc_vf_f32m1(__riscv_vfmul_vf_f32m1(v10, 1.f - beta, vl), beta, v11, vl);
            vfloat32m1_t outv = __riscv_vfmacc_vf_f32m1(__riscv_vfmul_vf_f32m1(v0, 1.f - gamma, vl), gamma, v1, vl);

            __riscv_vse32_v_f32m1(dstptr, outv, vl);
            dstptr += vl;
            offset_value_ptr += 11;
        }
    }
}

static void gridsample_3d_bilinear_apply_interpolation_p8(const Mat& src, Mat& dst, const Mat& offset_value, const Option& opt)
{
    gridsample_3d_bilinear_apply_interpolation_p4(src, dst, offset_value, opt);
}

static void gridsample_3d_bilinear_apply_interpolation_p16(const Mat& src, Mat& dst, const Mat& offset_value, const Option& opt)
{
    gridsample_3d_bilinear_apply_interpolation_p4(src, dst, offset_value, opt);
}

#endif // __riscv_vector
