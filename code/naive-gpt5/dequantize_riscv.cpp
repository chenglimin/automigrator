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

#include "dequantize_riscv.h"

#if __riscv_vector
#include <riscv_vector.h>
#endif // __riscv_vector

#include "riscv_usability.h"

namespace ncnn {

Dequantize_riscv::Dequantize_riscv()
{
#if __riscv_vector
    support_packing = true;
#endif // __riscv_vector
}

static void dequantize(const int* intptr, float* ptr, const Mat& scale_data, const Mat& bias_data, int elemcount, int elempack)
{
    const int scale_data_size = scale_data.w;
    const int bias_data_size = bias_data.w;
    const int size = elemcount * elempack;

    float scale = scale_data[0];

#if __riscv_vector
    // when scale/bias are per-pack, process fixed-width chunks of elempack
    if (scale_data_size > 1 && elempack > 1)
    {
        size_t vl_fixed = __riscv_vsetvl_e32m1(elempack);
        vfloat32m1_t _scale = __riscv_vle32_v_f32m1((const float*)scale_data, vl_fixed);
        vfloat32m1_t _bias = __riscv_vfmv_v_f_f32m1(0.f, vl_fixed);
        bool has_bias = bias_data_size != 0;
        if (has_bias)
        {
            if (bias_data_size > 1)
                _bias = __riscv_vle32_v_f32m1((const float*)bias_data, vl_fixed);
            else
                _bias = __riscv_vfmv_v_f_f32m1(bias_data[0], vl_fixed);
        }

        int i = 0;
        for (; i + elempack - 1 < size; i += elempack)
        {
            vint32m1_t _vi = __riscv_vle32_v_i32m1(intptr, vl_fixed);
            vfloat32m1_t _vf = __riscv_vfcvt_f_x_v_f32m1(_vi, vl_fixed);
            if (has_bias)
            {
                vfloat32m1_t _res = __riscv_vfmacc_vv_f32m1(_bias, _vf, _scale, vl_fixed);
                __riscv_vse32_v_f32m1(ptr, _res, vl_fixed);
            }
            else
            {
                vfloat32m1_t _res = __riscv_vfmul_vv_f32m1(_vf, _scale, vl_fixed);
                __riscv_vse32_v_f32m1(ptr, _res, vl_fixed);
            }
            intptr += elempack;
            ptr += elempack;
        }
        // tail
        for (; i < size; i++)
        {
            *ptr = has_bias ? (*intptr * scale_data[i % elempack] + (bias_data_size > 1 ? bias_data[i % elempack] : bias_data[0])) : (*intptr * scale_data[i % elempack]);
            intptr++;
            ptr++;
        }
        return;
    }
#endif // __riscv_vector

    if (bias_data_size == 0)
    {
#if __riscv_vector
        int n = size;
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e32m8(n);
            vint32m8_t _vi = __riscv_vle32_v_i32m8(intptr, vl);
            vfloat32m8_t _vf = __riscv_vfcvt_f_x_v_f32m8(_vi, vl);
            vfloat32m8_t _scale = __riscv_vfmv_v_f_f32m8(scale, vl);
            vfloat32m8_t _res = __riscv_vfmul_vv_f32m8(_vf, _scale, vl);
            __riscv_vse32_v_f32m8(ptr, _res, vl);
            intptr += vl;
            ptr += vl;
            n -= vl;
        }
#else
        for (int i = 0; i < size; i++)
        {
            *ptr = *intptr * scale;
            intptr++;
            ptr++;
        }
#endif // __riscv_vector
    }
    else
    {
        float bias = bias_data[0];
#if __riscv_vector
        int n = size;
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e32m8(n);
            vint32m8_t _vi = __riscv_vle32_v_i32m8(intptr, vl);
            vfloat32m8_t _vf = __riscv_vfcvt_f_x_v_f32m8(_vi, vl);
            vfloat32m8_t _scale = __riscv_vfmv_v_f_f32m8(scale, vl);
            vfloat32m8_t _bias = __riscv_vfmv_v_f_f32m8(bias, vl);
            vfloat32m8_t _res = __riscv_vfmacc_vv_f32m8(_bias, _vf, _scale, vl);
            __riscv_vse32_v_f32m8(ptr, _res, vl);
            intptr += vl;
            ptr += vl;
            n -= vl;
        }
#else
        for (int i = 0; i < size; i++)
        {
            *ptr = *intptr * scale + bias;
            intptr++;
            ptr++;
        }
#endif // __riscv_vector
    }
}

int Dequantize_riscv::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    const int dims = bottom_blob.dims;
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;

    top_blob.create_like(bottom_blob, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    if (dims == 1)
    {
        const int wp = std::max(1, w / opt.num_threads);
        const int nn_w = (w + wp - 1) / wp;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < nn_w; ii++)
        {
            const int i = ii * wp;

            const int* intptr = (const int*)bottom_blob + i * elempack;
            float* ptr = (float*)top_blob + i * elempack;

            // assert scale_data_size == 1
            // assert bias_data_size == 0 || bias_data_size == 1

            const int size = std::min(w - i, wp) * elempack;

            dequantize(intptr, ptr, scale_data, bias_data, size, 1);
        }
    }

    if (dims == 2)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            const int* intptr = bottom_blob.row<const int>(i);
            float* ptr = top_blob.row(i);

            const Mat scale_data_i = scale_data_size > 1 ? scale_data.range(i * elempack, elempack) : scale_data;
            const Mat bias_data_i = bias_data_size > 1 ? bias_data.range(i * elempack, elempack) : bias_data;

            dequantize(intptr, ptr, scale_data_i, bias_data_i, w, elempack);
        }
    }

    if (dims == 3)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const int* intptr = bottom_blob.channel(q);
            float* ptr = top_blob.channel(q);

            const Mat scale_data_q = scale_data_size > 1 ? scale_data.range(q * elempack, elempack) : scale_data;
            const Mat bias_data_q = bias_data_size > 1 ? bias_data.range(q * elempack, elempack) : bias_data;

            dequantize(intptr, ptr, scale_data_q, bias_data_q, w * h, elempack);
        }
    }

    return 0;
}

} // namespace ncnn
