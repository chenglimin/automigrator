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
#endif
}

static inline void dequantize_kernel(const int* intptr, float* ptr, const Mat& scale_data, const Mat& bias_data, int elemcount, int elempack)
{
    const int scale_data_size = scale_data.w;
    const int bias_data_size = bias_data.w;
    const int size = elemcount * elempack;

#if __riscv_vector
    const int packn = (int)(csrr_vlenb() / 4);

    if (elempack == packn)
    {
        size_t vlp = __riscv_vsetvl_e32m8(packn);

        if (scale_data_size == 1 && bias_data_size == 0)
        {
            float scale = scale_data[0];
            vfloat32m8_t _scale = __riscv_vfmv_v_f_f32m8(scale, vlp);
            for (int i = 0; i < elemcount; i++)
            {
                vint32m8_t _vi = __riscv_vle32_v_i32m8(intptr, vlp);
                vfloat32m8_t _v = __riscv_vfcvt_f_x_v_f32m8(_vi, vlp);
                _v = __riscv_vfmul_vv_f32m8(_v, _scale, vlp);
                __riscv_vse32_v_f32m8(ptr, _v, vlp);
                intptr += packn;
                ptr += packn;
            }
            return;
        }
        if (scale_data_size == 1 && bias_data_size > 1)
        {
            float scale = scale_data[0];
            vfloat32m8_t _scale = __riscv_vfmv_v_f_f32m8(scale, vlp);
            vfloat32m8_t _bias_v = __riscv_vle32_v_f32m8((const float*)bias_data, vlp);
            for (int i = 0; i < elemcount; i++)
            {
                vint32m8_t _vi = __riscv_vle32_v_i32m8(intptr, vlp);
                vfloat32m8_t _v = __riscv_vfcvt_f_x_v_f32m8(_vi, vlp);
                _v = __riscv_vfmul_vv_f32m8(_v, _scale, vlp);
                _v = __riscv_vfadd_vv_f32m8(_v, _bias_v, vlp);
                __riscv_vse32_v_f32m8(ptr, _v, vlp);
                intptr += packn;
                ptr += packn;
            }
            return;
        }
        if (scale_data_size > 1 && bias_data_size == 0)
        {
            vfloat32m8_t _scale_v = __riscv_vle32_v_f32m8((const float*)scale_data, vlp);
            for (int i = 0; i < elemcount; i++)
            {
                vint32m8_t _vi = __riscv_vle32_v_i32m8(intptr, vlp);
                vfloat32m8_t _v = __riscv_vfcvt_f_x_v_f32m8(_vi, vlp);
                _v = __riscv_vfmul_vv_f32m8(_v, _scale_v, vlp);
                __riscv_vse32_v_f32m8(ptr, _v, vlp);
                intptr += packn;
                ptr += packn;
            }
            return;
        }
        if (scale_data_size > 1 && bias_data_size == 1)
        {
            vfloat32m8_t _scale_v = __riscv_vle32_v_f32m8((const float*)scale_data, vlp);
            float bias = bias_data[0];
            for (int i = 0; i < elemcount; i++)
            {
                vint32m8_t _vi = __riscv_vle32_v_i32m8(intptr, vlp);
                vfloat32m8_t _v = __riscv_vfcvt_f_x_v_f32m8(_vi, vlp);
                _v = __riscv_vfmul_vv_f32m8(_v, _scale_v, vlp);
                _v = __riscv_vfadd_vf_f32m8(_v, bias, vlp);
                __riscv_vse32_v_f32m8(ptr, _v, vlp);
                intptr += packn;
                ptr += packn;
            }
            return;
        }
        if (scale_data_size > 1 && bias_data_size > 1)
        {
            vfloat32m8_t _scale_v = __riscv_vle32_v_f32m8((const float*)scale_data, vlp);
            vfloat32m8_t _bias_v = __riscv_vle32_v_f32m8((const float*)bias_data, vlp);
            for (int i = 0; i < elemcount; i++)
            {
                vint32m8_t _vi = __riscv_vle32_v_i32m8(intptr, vlp);
                vfloat32m8_t _v = __riscv_vfcvt_f_x_v_f32m8(_vi, vlp);
                _v = __riscv_vfmul_vv_f32m8(_v, _scale_v, vlp);
                _v = __riscv_vfadd_vv_f32m8(_v, _bias_v, vlp);
                __riscv_vse32_v_f32m8(ptr, _v, vlp);
                intptr += packn;
                ptr += packn;
            }
            return;
        }
    }

    // generic path: scalar scale and optional scalar bias
    float scale = scale_data[0];
    int n = size;
    while (n > 0)
    {
        size_t vl = __riscv_vsetvl_e32m8(n);
        vint32m8_t _vi = __riscv_vle32_v_i32m8(intptr, vl);
        vfloat32m8_t _v = __riscv_vfcvt_f_x_v_f32m8(_vi, vl);
        vfloat32m8_t _scale = __riscv_vfmv_v_f_f32m8(scale, vl);
        _v = __riscv_vfmul_vv_f32m8(_v, _scale, vl);
        if (bias_data_size)
        {
            float bias = bias_data_size > 1 ? bias_data[0] : bias_data[0];
            vfloat32m8_t _bias = __riscv_vfmv_v_f_f32m8(bias, vl);
            _v = __riscv_vfadd_vv_f32m8(_v, _bias, vl);
        }
        __riscv_vse32_v_f32m8(ptr, _v, vl);
        intptr += vl;
        ptr += vl;
        n -= vl;
    }
#else
    float scale = scale_data[0];
    if (bias_data_size == 0)
    {
        for (int i = 0; i < size; i++)
        {
            *ptr = *intptr * scale;
            intptr++;
            ptr++;
        }
    }
    else
    {
        float bias = bias_data[0];
        for (int i = 0; i < size; i++)
        {
            *ptr = *intptr * scale + bias;
            intptr++;
            ptr++;
        }
    }
#endif
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

            const int size = std::min(w - i, wp) * elempack;
            dequantize_kernel(intptr, ptr, scale_data, bias_data, size, 1);
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

            dequantize_kernel(intptr, ptr, scale_data_i, bias_data_i, w, elempack);
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

            dequantize_kernel(intptr, ptr, scale_data_q, bias_data_q, w * h, elempack);
        }
    }

    return 0;
}

} // namespace ncnn
