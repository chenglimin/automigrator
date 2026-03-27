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

#include "requantize_riscv.h"

#if __riscv_vector
#include <riscv_vector.h>
#endif // __riscv_vector

#include "riscv_activation.h"
#include "riscv_usability.h"

namespace ncnn {

Requantize_riscv::Requantize_riscv()
{
#if __riscv_vector
    support_packing = true;
#endif // __riscv_vector
}

#if __riscv_vector
static inline vint8m2_t float2int8_ps(vfloat32m8_t _v, size_t vl)
{
    // convert to int32
    vint32m8_t _i32 = __riscv_vfcvt_x_f_v_i32m8_rm(_v, __RISCV_FRM_RNE, vl);
    // clamp to [-127,127]
    _i32 = __riscv_vmin_vx_i32m8(_i32, 127, vl);
    _i32 = __riscv_vmax_vx_i32m8(_i32, -127, vl);
    // narrow i32m8 -> i16m4 -> i8m2 with shift=0
    vint16m4_t _i16 = __riscv_vnclip_wx_i16m4(_i32, 0, __RISCV_VXRM_RNU, vl);
    vint8m2_t _i8 = __riscv_vnclip_wx_i8m2(_i16, 0, __RISCV_VXRM_RNU, vl);
    return _i8;
}
#endif // __riscv_vector

static void requantize_scalar(const int* intptr, signed char* ptr, const Mat& scale_in_data, const Mat& bias_data, const Mat& scale_out_data, int activation_type, const Mat& activation_params, int elemcount, int elempack);

#if __riscv_vector
static void requantize_vec(const int* intptr, signed char* ptr, const Mat& scale_in_data, const Mat& bias_data, const Mat& scale_out_data, int activation_type, const Mat& activation_params, int elemcount, int elempack)
{
    const int scale_in_data_size = scale_in_data.w;
    const int bias_data_size = bias_data.w;
    const int scale_out_data_size = scale_out_data.w;
    const int size = elemcount * elempack;

    float scale_in = scale_in_data[0];
    float scale_out = scale_out_data[0];

    if (elempack > 1 && (scale_in_data_size > 1 || bias_data_size > 1 || scale_out_data_size > 1))
    {
        // fallback to scalar for per-pack params to ensure correctness
        requantize_scalar(intptr, ptr, scale_in_data, bias_data, scale_out_data, activation_type, activation_params, elemcount, elempack);
        return;
    }

    int n = size;
    while (n > 0)
    {
        size_t vl = __riscv_vsetvl_e32m8(n);
        vfloat32m8_t _v = __riscv_vfcvt_f_x_v_f32m8(__riscv_vle32_v_i32m8(intptr, vl), vl);
        _v = __riscv_vfmul_vf_f32m8(_v, scale_in, vl);
        if (bias_data_size)
        {
            float b = bias_data_size == 1 ? bias_data[0] : 0.f;
            _v = __riscv_vfadd_vf_f32m8(_v, b, vl);
        }
        _v = activation_ps(_v, activation_type, activation_params, vl);
        _v = __riscv_vfmul_vf_f32m8(_v, scale_out, vl);

        vint8m2_t _i8 = float2int8_ps(_v, vl);
        __riscv_vse8_v_i8m2(ptr, _i8, vl);

        intptr += vl;
        ptr += vl;
        n -= vl;
    }
}
#endif // __riscv_vector

static void requantize_scalar(const int* intptr, signed char* ptr, const Mat& scale_in_data, const Mat& bias_data, const Mat& scale_out_data, int activation_type, const Mat& activation_params, int elemcount, int elempack)
{
    const int scale_in_data_size = scale_in_data.w;
    const int bias_data_size = bias_data.w;
    const int scale_out_data_size = scale_out_data.w;
    const int size = elemcount * elempack;

    float scale_in = scale_in_data[0];
    float scale_out = scale_out_data[0];

    for (int i = 0; i < size; i++)
    {
        float v = (float)intptr[i] * (scale_in_data_size == 1 ? scale_in : ((const float*)scale_in_data)[i % elempack]);
        if (bias_data_size)
        {
            v += (bias_data_size == 1 ? bias_data[0] : ((const float*)bias_data)[i % elempack]);
        }
        v = activation_ss(v, activation_type, activation_params);
        v *= (scale_out_data_size == 1 ? scale_out : ((const float*)scale_out_data)[i % elempack]);
        // clamp per ncnn behavior
        int int32 = (int)roundf(v);
        if (int32 > 127) int32 = 127;
        if (int32 < -127) int32 = -127;
        ptr[i] = (signed char)int32;
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

            const int size = std::min(w - i, wp) * elempack;

#if __riscv_vector
            requantize_vec(intptr, ptr, scale_in_data, bias_data, scale_out_data, activation_type, activation_params, size, 1);
#else
            requantize_scalar(intptr, ptr, scale_in_data, bias_data, scale_out_data, activation_type, activation_params, size, 1);
#endif
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

#if __riscv_vector
            requantize_vec(intptr, ptr, scale_in_data_i, bias_data_i, scale_out_data_i, activation_type, activation_params, w, elempack);
#else
            requantize_scalar(intptr, ptr, scale_in_data_i, bias_data_i, scale_out_data_i, activation_type, activation_params, w, elempack);
#endif
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

#if __riscv_vector
            requantize_vec(intptr, ptr, scale_in_data_q, bias_data_q, scale_out_data_q, activation_type, activation_params, w * h, elempack);
#else
            requantize_scalar(intptr, ptr, scale_in_data_q, bias_data_q, scale_out_data_q, activation_type, activation_params, w * h, elempack);
#endif
        }
    }

    return 0;
}

} // namespace ncnn
