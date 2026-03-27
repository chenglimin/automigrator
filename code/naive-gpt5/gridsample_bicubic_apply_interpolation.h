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

static inline void cubic_interp1d_scalar(float tx, float& c0, float& c1, float& c2, float& c3)
{
    const float A = -0.75f;
    float x0 = tx + 1.f;
    float x1 = tx;
    float x2 = 1.f - tx;
    c0 = ((A * x0 - 5.f * A) * x0 + 8.f * A) * x0 - 4.f * A;
    c1 = ((A + 2.f) * x1 - (A + 3.f)) * x1 * x1 + 1.f;
    c2 = ((A + 2.f) * x2 - (A + 3.f)) * x2 * x2 + 1.f;
    c3 = 1.f - c0 - c1 - c2;
}

static void gridsample_2d_bicubic_apply_interpolation_p1(const Mat& src, Mat& dst, Mat& offset_value, const Option& opt)
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
            float x_coeffs0, x_coeffs1, x_coeffs2, x_coeffs3;
            float y_coeffs0, y_coeffs1, y_coeffs2, y_coeffs3;
            cubic_interp1d_scalar(offset_value_ptr[0], x_coeffs0, x_coeffs1, x_coeffs2, x_coeffs3);
            cubic_interp1d_scalar(offset_value_ptr[1], y_coeffs0, y_coeffs1, y_coeffs2, y_coeffs3);

            const int* offset_ptr = (const int*)offset_value_ptr + 2;

            float value_f[4];
            for (int ii = 0; ii < 4; ii++)
            {
                float x0_val = offset_ptr[0] >= 0 ? *(srcptr + offset_ptr[0]) : 0.f;
                float x1_val = offset_ptr[1] >= 0 ? *(srcptr + offset_ptr[1]) : 0.f;
                float x2_val = offset_ptr[2] >= 0 ? *(srcptr + offset_ptr[2]) : 0.f;
                float x3_val = offset_ptr[3] >= 0 ? *(srcptr + offset_ptr[3]) : 0.f;

                value_f[ii] = x_coeffs0 * x0_val;
                value_f[ii] += x_coeffs1 * x1_val;
                value_f[ii] += x_coeffs2 * x2_val;
                value_f[ii] += x_coeffs3 * x3_val;

                offset_ptr += 4;
            }

            *dstptr = y_coeffs0 * value_f[0] + y_coeffs1 * value_f[1] + y_coeffs2 * value_f[2] + y_coeffs3 * value_f[3];

            dstptr += 1;
            offset_value_ptr += 18;
        }
    }
}

#if __riscv_vector
static inline size_t __ncnn_vl_from_elempack(const Mat& dst)
{
    return __riscv_vsetvl_e32m1(dst.elempack);
}

static void gridsample_2d_bicubic_apply_interpolation_p4(const Mat& src, Mat& dst, Mat& offset_value, const Option& opt)
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
            float x_coeffs0, x_coeffs1, x_coeffs2, x_coeffs3;
            float y_coeffs0, y_coeffs1, y_coeffs2, y_coeffs3;
            cubic_interp1d_scalar(offset_value_ptr[0], x_coeffs0, x_coeffs1, x_coeffs2, x_coeffs3);
            cubic_interp1d_scalar(offset_value_ptr[1], y_coeffs0, y_coeffs1, y_coeffs2, y_coeffs3);

            const int* offset_ptr = (const int*)offset_value_ptr + 2;

            vfloat32m1_t value_f0 = __riscv_vfmv_v_f_f32m1(0.f, vl);
            vfloat32m1_t value_f1 = __riscv_vfmv_v_f_f32m1(0.f, vl);
            vfloat32m1_t value_f2 = __riscv_vfmv_v_f_f32m1(0.f, vl);
            vfloat32m1_t value_f3 = __riscv_vfmv_v_f_f32m1(0.f, vl);
            for (int ii = 0; ii < 4; ii++)
            {
                vfloat32m1_t x0_val = offset_ptr[0] >= 0 ? __riscv_vle32_v_f32m1(srcptr + offset_ptr[0], vl) : __riscv_vfmv_v_f_f32m1(0.f, vl);
                vfloat32m1_t x1_val = offset_ptr[1] >= 0 ? __riscv_vle32_v_f32m1(srcptr + offset_ptr[1], vl) : __riscv_vfmv_v_f_f32m1(0.f, vl);
                vfloat32m1_t x2_val = offset_ptr[2] >= 0 ? __riscv_vle32_v_f32m1(srcptr + offset_ptr[2], vl) : __riscv_vfmv_v_f_f32m1(0.f, vl);
                vfloat32m1_t x3_val = offset_ptr[3] >= 0 ? __riscv_vle32_v_f32m1(srcptr + offset_ptr[3], vl) : __riscv_vfmv_v_f_f32m1(0.f, vl);

                vfloat32m1_t sum = __riscv_vfmul_vf_f32m1(x0_val, x_coeffs0, vl);
                sum = __riscv_vfmacc_vf_f32m1(sum, x_coeffs1, x1_val, vl);
                sum = __riscv_vfmacc_vf_f32m1(sum, x_coeffs2, x2_val, vl);
                sum = __riscv_vfmacc_vf_f32m1(sum, x_coeffs3, x3_val, vl);

                if (ii == 0) value_f0 = sum;
                else if (ii == 1) value_f1 = sum;
                else if (ii == 2) value_f2 = sum;
                else value_f3 = sum;

                offset_ptr += 4;
            }

            vfloat32m1_t outv = __riscv_vfmul_vf_f32m1(value_f0, y_coeffs0, vl);
            outv = __riscv_vfmacc_vf_f32m1(outv, y_coeffs1, value_f1, vl);
            outv = __riscv_vfmacc_vf_f32m1(outv, y_coeffs2, value_f2, vl);
            outv = __riscv_vfmacc_vf_f32m1(outv, y_coeffs3, value_f3, vl);

            __riscv_vse32_v_f32m1(dstptr, outv, vl);
            dstptr += vl;
            offset_value_ptr += 18;
        }
    }
}

static void gridsample_2d_bicubic_apply_interpolation_p8(const Mat& src, Mat& dst, Mat& offset_value, const Option& opt)
{
    gridsample_2d_bicubic_apply_interpolation_p4(src, dst, offset_value, opt);
}

static void gridsample_2d_bicubic_apply_interpolation_p16(const Mat& src, Mat& dst, Mat& offset_value, const Option& opt)
{
    gridsample_2d_bicubic_apply_interpolation_p4(src, dst, offset_value, opt);
}

#endif // __riscv_vector
