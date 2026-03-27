// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "requantize_riscv.h"

#if __riscv_vector
#include <riscv_vector.h>
#endif // __riscv_vector

#include "riscv_activation.h"

namespace ncnn {

Requantize_riscv::Requantize_riscv()
{
#if __riscv_vector
    support_packing = true;
#endif // __riscv_vector
}

#if __riscv_vector
static inline void quantize_store_int8(signed char* ptr, vfloat32m8_t _v, size_t vl)
{
    // convert to int32 with default rounding, then saturate to [-127, 127], narrow and store
    vint32m8_t _vi32 = __riscv_vfcvt_x_f_v_i32m8(_v, vl);
    _vi32 = __riscv_vmax_vx_i32m8(_vi32, -127, vl);
    _vi32 = __riscv_vmin_vx_i32m8(_vi32, 127, vl);

    // narrow to int16 then int8 with shift 0
    vint16m4_t _vi16 = __riscv_vnclip_wx_i16m4(_vi32, 0, vl);
    vint8m2_t _vi8 = __riscv_vnclip_wx_i8m2(_vi16, 0, vl);
    __riscv_vse8_v_i8m2(ptr, _vi8, vl);
}
#endif // __riscv_vector

static void requantize_kernel(const int* intptr, signed char* ptr, const Mat& scale_in_data, const Mat& bias_data, const Mat& scale_out_data, int activation_type, const Mat& activation_params, int elemcount, int elempack)
{
    const int scale_in_data_size = scale_in_data.w;
    const int bias_data_size = bias_data.w;
    const int scale_out_data_size = scale_out_data.w;
    const int size = elemcount * elempack;

#if __riscv_vector
    // fast vector path only when per-pack scales/bias are scalar
    if (scale_in_data_size == 1 && scale_out_data_size == 1 && (bias_data_size == 0 || bias_data_size == 1))
    {
        float scale_in = scale_in_data[0];
        float scale_out = scale_out_data[0];
        float bias = bias_data_size ? (float)bias_data[0] : 0.f;

        int n = size;
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e32m8(n);
            vint32m8_t _vint = __riscv_vle32_v_i32m8(intptr, vl);
            vfloat32m8_t _vf = __riscv_vfcvt_f_x_v_f32m8(_vint, vl);

            _vf = __riscv_vfmul_vf_f32m8(_vf, scale_in, vl);
            if (bias_data_size)
                _vf = __riscv_vfadd_vf_f32m8(_vf, bias, vl);

            _vf = activation_ps(_vf, activation_type, activation_params, vl);
            _vf = __riscv_vfmul_vf_f32m8(_vf, scale_out, vl);

            quantize_store_int8(ptr, _vf, vl);

            intptr += vl;
            ptr += vl;
            n -= vl;
        }
        return;
    }
#endif // __riscv_vector

    // fallback scalar path
    float scale_in = scale_in_data[0];
    float scale_out = scale_out_data[0];
    float bias0 = bias_data_size ? (float)bias_data[0] : 0.f;

    for (int i = 0; i < size; i++)
    {
        float v = (float)intptr[i] * scale_in + bias0;
        v = activation_ss(v, activation_type, activation_params);
        ptr[i] = float2int8(v * scale_out);
    }
}

int Requantize_riscv::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    const int dims = bottom_blob.dims;
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;
    const size_t out_elemsize = elempack * 1u;

    if (dims == 1)
    {
        top_blob.create(w, out_elemsize, elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        const int wp = std::max(1, w / opt.num_threads);
        const int nn_w = (w + wp - 1) / wp;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < nn_w; ii++)
        {
            const int i = ii * wp;

            const int* intptr = (const int*)bottom_blob + i * elempack;
            signed char* ptr = (signed char*)top_blob + i * elempack;

            // assert scale_in_data_size == 1
            // assert bias_data_size == 0 || bias_data_size == 1
            // assert scale_out_data_size == 1

            const int size = std::min(w - i, wp) * elempack;

            requantize_kernel(intptr, ptr, scale_in_data, bias_data, scale_out_data, activation_type, activation_params, size, 1);
        }
    }

    if (dims == 2)
    {
        top_blob.create(w, h, out_elemsize, elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            const int* intptr = bottom_blob.row<const int>(i);
            signed char* ptr = top_blob.row<signed char>(i);

            const Mat scale_in_data_i = scale_in_data_size > 1 ? scale_in_data.range(i * elempack, elempack) : scale_in_data;
            const Mat bias_data_i = bias_data_size > 1 ? bias_data.range(i * elempack, elempack) : bias_data;
            const Mat scale_out_data_i = scale_out_data_size > 1 ? scale_out_data.range(i * elempack, elempack) : scale_out_data;

            requantize_kernel(intptr, ptr, scale_in_data_i, bias_data_i, scale_out_data_i, activation_type, activation_params, w, elempack);
        }
    }

    if (dims == 3)
    {
        top_blob.create(w, h, channels, out_elemsize, elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const int* intptr = bottom_blob.channel(q);
            signed char* ptr = top_blob.channel(q);

            const Mat scale_in_data_q = scale_in_data_size > 1 ? scale_in_data.range(q * elempack, elempack) : scale_in_data;
            const Mat bias_data_q = bias_data_size > 1 ? bias_data.range(q * elempack, elempack) : bias_data;
            const Mat scale_out_data_q = scale_out_data_size > 1 ? scale_out_data.range(q * elempack, elempack) : scale_out_data;

            requantize_kernel(intptr, ptr, scale_in_data_q, bias_data_q, scale_out_data_q, activation_type, activation_params, w * h, elempack);
        }
    }

    return 0;
}

} // namespace ncnn
