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
#include <algorithm>

namespace ncnn {

Requantize_riscv::Requantize_riscv()
{
#if __riscv_vector
    support_packing = true;
#endif // __riscv_vector
}

static void requantize_relu(const int* intptr, signed char* ptr, const Mat& scale_in_data, const Mat& bias_data, const Mat& scale_out_data, int elemcount, int elempack)
{
    const int scale_in_data_size = scale_in_data.w;
    const int bias_data_size = bias_data.w;
    const int scale_out_data_size = scale_out_data.w;
    const int size = elemcount * elempack;

    float scale_in = scale_in_data[0];
#if __riscv_vector
    vfloat32m1_t _scale_in0, _scale_in1;
    {
        size_t vl = __riscv_vsetvl_e32m1(4);
        _scale_in0 = __riscv_vfmv_v_f_f32m1(scale_in, vl);
        _scale_in1 = _scale_in0;
    }
    if (scale_in_data_size > 1)
    {
        if (elempack == 8)
        {
            size_t vl = __riscv_vsetvl_e32m1(4);
            _scale_in0 = __riscv_vle32_v_f32m1((const float*)scale_in_data, vl);
            _scale_in1 = __riscv_vle32_v_f32m1((const float*)scale_in_data + 4, vl);
        }
    }
#endif // __riscv_vector

    float scale_out = scale_out_data[0];
#if __riscv_vector
    vfloat32m1_t _scale_out0, _scale_out1;
    {
        size_t vl = __riscv_vsetvl_e32m1(4);
        _scale_out0 = __riscv_vfmv_v_f_f32m1(scale_out, vl);
        _scale_out1 = _scale_out0;
    }
    if (scale_out_data_size > 1)
    {
        if (elempack == 8)
        {
            size_t vl = __riscv_vsetvl_e32m1(4);
            _scale_out0 = __riscv_vle32_v_f32m1((const float*)scale_out_data, vl);
            _scale_out1 = __riscv_vle32_v_f32m1((const float*)scale_out_data + 4, vl);
        }
    }
#endif // __riscv_vector

    float scale = scale_in * scale_out;
#if __riscv_vector
    vfloat32m1_t _scale0 = __riscv_vfmul_vv_f32m1(_scale_in0, _scale_out0, __riscv_vsetvl_e32m1(4));
    vfloat32m1_t _scale1 = __riscv_vfmul_vv_f32m1(_scale_in1, _scale_out1, __riscv_vsetvl_e32m1(4));
#endif // __riscv_vector

    if (bias_data_size == 0)
    {
        int i = 0;
#if __riscv_vector
        for (; i + 7 < size; i += 8)
        {
            size_t vl = __riscv_vsetvl_e32m1(4);
            vfloat32m1_t _v0 = __riscv_vfcvt_f_x_v_f32m1(__riscv_vle32_v_i32m1(intptr, vl), vl);
            vfloat32m1_t _v1 = __riscv_vfcvt_f_x_v_f32m1(__riscv_vle32_v_i32m1(intptr + 4, vl), vl);
            _v0 = __riscv_vfmul_vv_f32m1(_v0, _scale0, vl);
            _v1 = __riscv_vfmul_vv_f32m1(_v1, _scale1, vl);
            // relu
            _v0 = __riscv_vfmax_vf_f32m1(_v0, 0.f, vl);
            _v1 = __riscv_vfmax_vf_f32m1(_v1, 0.f, vl);
            // convert and store
            float tmp0[4], tmp1[4];
            __riscv_vse32_v_f32m1(tmp0, _v0, vl);
            __riscv_vse32_v_f32m1(tmp1, _v1, vl);
            ptr[0] = float2int8(tmp0[0]);
            ptr[1] = float2int8(tmp0[1]);
            ptr[2] = float2int8(tmp0[2]);
            ptr[3] = float2int8(tmp0[3]);
            ptr[4] = float2int8(tmp1[0]);
            ptr[5] = float2int8(tmp1[1]);
            ptr[6] = float2int8(tmp1[2]);
            ptr[7] = float2int8(tmp1[3]);
            intptr += 8;
            ptr += 8;
        }
        for (; i + 3 < size; i += 4)
        {
            size_t vl = __riscv_vsetvl_e32m1(4);
            vfloat32m1_t _v = __riscv_vfcvt_f_x_v_f32m1(__riscv_vle32_v_i32m1(intptr, vl), vl);
            _v = __riscv_vfmul_vv_f32m1(_v, _scale0, vl);
            _v = __riscv_vfmax_vf_f32m1(_v, 0.f, vl);
            float tmp[4];
            __riscv_vse32_v_f32m1(tmp, _v, vl);
            ptr[0] = float2int8(tmp[0]);
            ptr[1] = float2int8(tmp[1]);
            ptr[2] = float2int8(tmp[2]);
            ptr[3] = float2int8(tmp[3]);
            intptr += 4;
            ptr += 4;
        }
#endif // __riscv_vector
        for (; i < size; i++)
        {
            float v = *intptr * scale;
            if (v < 0) v = 0;
            *ptr = float2int8(v);
            intptr++;
            ptr++;
        }
    }
    else
    {
        float bias = bias_data[0];
#if __riscv_vector
        vfloat32m1_t _bias0, _bias1;
        {
            size_t vl = __riscv_vsetvl_e32m1(4);
            _bias0 = __riscv_vfmv_v_f_f32m1(bias, vl);
            _bias1 = _bias0;
        }
        if (bias_data_size > 1)
        {
            if (elempack == 8)
            {
                size_t vl = __riscv_vsetvl_e32m1(4);
                _bias0 = __riscv_vle32_v_f32m1((const float*)bias_data, vl);
                _bias1 = __riscv_vle32_v_f32m1((const float*)bias_data + 4, vl);
            }
        }
        // scale bias by scale_out
        _bias0 = __riscv_vfmul_vv_f32m1(_bias0, _scale_out0, __riscv_vsetvl_e32m1(4));
        _bias1 = __riscv_vfmul_vv_f32m1(_bias1, _scale_out1, __riscv_vsetvl_e32m1(4));
#endif // __riscv_vector

        bias = bias * scale_out;

        int i = 0;
#if __riscv_vector
        for (; i + 7 < size; i += 8)
        {
            size_t vl = __riscv_vsetvl_e32m1(4);
            vfloat32m1_t _v0 = __riscv_vfcvt_f_x_v_f32m1(__riscv_vle32_v_i32m1(intptr, vl), vl);
            vfloat32m1_t _v1 = __riscv_vfcvt_f_x_v_f32m1(__riscv_vle32_v_i32m1(intptr + 4, vl), vl);
            _v0 = __riscv_vfmacc_vv_f32m1(_bias0, _v0, _scale0, vl);
            _v1 = __riscv_vfmacc_vv_f32m1(_bias1, _v1, _scale1, vl);
            _v0 = __riscv_vfmax_vf_f32m1(_v0, 0.f, vl);
            _v1 = __riscv_vfmax_vf_f32m1(_v1, 0.f, vl);
            float tmp0[4], tmp1[4];
            __riscv_vse32_v_f32m1(tmp0, _v0, vl);
            __riscv_vse32_v_f32m1(tmp1, _v1, vl);
            ptr[0] = float2int8(tmp0[0]);
            ptr[1] = float2int8(tmp0[1]);
            ptr[2] = float2int8(tmp0[2]);
            ptr[3] = float2int8(tmp0[3]);
            ptr[4] = float2int8(tmp1[0]);
            ptr[5] = float2int8(tmp1[1]);
            ptr[6] = float2int8(tmp1[2]);
            ptr[7] = float2int8(tmp1[3]);
            intptr += 8;
            ptr += 8;
        }
        for (; i + 3 < size; i += 4)
        {
            size_t vl = __riscv_vsetvl_e32m1(4);
            vfloat32m1_t _v = __riscv_vfcvt_f_x_v_f32m1(__riscv_vle32_v_i32m1(intptr, vl), vl);
            _v = __riscv_vfmacc_vv_f32m1(_bias0, _v, _scale0, vl);
            _v = __riscv_vfmax_vf_f32m1(_v, 0.f, vl);
            float tmp[4];
            __riscv_vse32_v_f32m1(tmp, _v, vl);
            ptr[0] = float2int8(tmp[0]);
            ptr[1] = float2int8(tmp[1]);
            ptr[2] = float2int8(tmp[2]);
            ptr[3] = float2int8(tmp[3]);
            intptr += 4;
            ptr += 4;
        }
#endif // __riscv_vector
        for (; i < size; i++)
        {
            float v = *intptr * scale + bias;
            if (v < 0) v = 0;
            *ptr = float2int8(v);
            intptr++;
            ptr++;
        }
    }
}

static void requantize_generic(const int* intptr, signed char* ptr, const Mat& scale_in_data, const Mat& bias_data, const Mat& scale_out_data, int activation_type, const Mat& activation_params, int elemcount, int elempack)
{
    if (activation_type == 1)
    {
        requantize_relu(intptr, ptr, scale_in_data, bias_data, scale_out_data, elemcount, elempack);
        return;
    }

    const int scale_in_data_size = scale_in_data.w;
    const int bias_data_size = bias_data.w;
    const int scale_out_data_size = scale_out_data.w;
    const int size = elemcount * elempack;

    float scale_in = scale_in_data[0];
    float scale_out = scale_out_data[0];

#if __riscv_vector
    vfloat32m1_t _scale_in0, _scale_in1;
    vfloat32m1_t _scale_out0, _scale_out1;
    {
        size_t vl = __riscv_vsetvl_e32m1(4);
        _scale_in0 = __riscv_vfmv_v_f_f32m1(scale_in, vl);
        _scale_in1 = _scale_in0;
        _scale_out0 = __riscv_vfmv_v_f_f32m1(scale_out, vl);
        _scale_out1 = _scale_out0;
    }
    if (scale_in_data_size > 1)
    {
        if (elempack == 8)
        {
            size_t vl = __riscv_vsetvl_e32m1(4);
            _scale_in0 = __riscv_vle32_v_f32m1((const float*)scale_in_data, vl);
            _scale_in1 = __riscv_vle32_v_f32m1((const float*)scale_in_data + 4, vl);
        }
    }
    if (scale_out_data_size > 1)
    {
        if (elempack == 8)
        {
            size_t vl = __riscv_vsetvl_e32m1(4);
            _scale_out0 = __riscv_vle32_v_f32m1((const float*)scale_out_data, vl);
            _scale_out1 = __riscv_vle32_v_f32m1((const float*)scale_out_data + 4, vl);
        }
    }
#endif // __riscv_vector

    if (bias_data_size == 0)
    {
        int i = 0;
#if __riscv_vector
        for (; i + 7 < size; i += 8)
        {
            size_t vl = __riscv_vsetvl_e32m1(4);
            vfloat32m1_t _v0 = __riscv_vfcvt_f_x_v_f32m1(__riscv_vle32_v_i32m1(intptr, vl), vl);
            vfloat32m1_t _v1 = __riscv_vfcvt_f_x_v_f32m1(__riscv_vle32_v_i32m1(intptr + 4, vl), vl);
            _v0 = __riscv_vfmul_vv_f32m1(_v0, _scale_in0, vl);
            _v1 = __riscv_vfmul_vv_f32m1(_v1, _scale_in1, vl);
            _v0 = activation_ps(_v0, activation_type, activation_params, vl);
            _v1 = activation_ps(_v1, activation_type, activation_params, vl);
            _v0 = __riscv_vfmul_vv_f32m1(_v0, _scale_out0, vl);
            _v1 = __riscv_vfmul_vv_f32m1(_v1, _scale_out1, vl);
            float tmp0[4], tmp1[4];
            __riscv_vse32_v_f32m1(tmp0, _v0, vl);
            __riscv_vse32_v_f32m1(tmp1, _v1, vl);
            ptr[0] = float2int8(tmp0[0]);
            ptr[1] = float2int8(tmp0[1]);
            ptr[2] = float2int8(tmp0[2]);
            ptr[3] = float2int8(tmp0[3]);
            ptr[4] = float2int8(tmp1[0]);
            ptr[5] = float2int8(tmp1[1]);
            ptr[6] = float2int8(tmp1[2]);
            ptr[7] = float2int8(tmp1[3]);
            intptr += 8;
            ptr += 8;
        }
        for (; i + 3 < size; i += 4)
        {
            size_t vl = __riscv_vsetvl_e32m1(4);
            vfloat32m1_t _v = __riscv_vfcvt_f_x_v_f32m1(__riscv_vle32_v_i32m1(intptr, vl), vl);
            _v = __riscv_vfmul_vv_f32m1(_v, _scale_in0, vl);
            _v = activation_ps(_v, activation_type, activation_params, vl);
            _v = __riscv_vfmul_vv_f32m1(_v, _scale_out0, vl);
            float tmp[4];
            __riscv_vse32_v_f32m1(tmp, _v, vl);
            ptr[0] = float2int8(tmp[0]);
            ptr[1] = float2int8(tmp[1]);
            ptr[2] = float2int8(tmp[2]);
            ptr[3] = float2int8(tmp[3]);
            intptr += 4;
            ptr += 4;
        }
#endif // __riscv_vector
        for (; i < size; i++)
        {
            float v = *intptr * scale_in;
            v = activation_ss(v, activation_type, activation_params);
            *ptr = float2int8(v * scale_out);
            intptr++;
            ptr++;
        }
    }
    else
    {
        float bias = bias_data[0];
#if __riscv_vector
        vfloat32m1_t _bias0, _bias1;
        {
            size_t vl = __riscv_vsetvl_e32m1(4);
            _bias0 = __riscv_vfmv_v_f_f32m1(bias, vl);
            _bias1 = _bias0;
        }
        if (bias_data_size > 1)
        {
            if (elempack == 8)
            {
                size_t vl = __riscv_vsetvl_e32m1(4);
                _bias0 = __riscv_vle32_v_f32m1((const float*)bias_data, vl);
                _bias1 = __riscv_vle32_v_f32m1((const float*)bias_data + 4, vl);
            }
        }
#endif // __riscv_vector

        int i = 0;
#if __riscv_vector
        for (; i + 7 < size; i += 8)
        {
            size_t vl = __riscv_vsetvl_e32m1(4);
            vfloat32m1_t _v0 = __riscv_vfcvt_f_x_v_f32m1(__riscv_vle32_v_i32m1(intptr, vl), vl);
            vfloat32m1_t _v1 = __riscv_vfcvt_f_x_v_f32m1(__riscv_vle32_v_i32m1(intptr + 4, vl), vl);
            _v0 = __riscv_vfmacc_vv_f32m1(_bias0, _v0, _scale_in0, vl);
            _v1 = __riscv_vfmacc_vv_f32m1(_bias1, _v1, _scale_in1, vl);
            _v0 = activation_ps(_v0, activation_type, activation_params, vl);
            _v1 = activation_ps(_v1, activation_type, activation_params, vl);
            _v0 = __riscv_vfmul_vv_f32m1(_v0, _scale_out0, vl);
            _v1 = __riscv_vfmul_vv_f32m1(_v1, _scale_out1, vl);
            float tmp0[4], tmp1[4];
            __riscv_vse32_v_f32m1(tmp0, _v0, vl);
            __riscv_vse32_v_f32m1(tmp1, _v1, vl);
            ptr[0] = float2int8(tmp0[0]);
            ptr[1] = float2int8(tmp0[1]);
            ptr[2] = float2int8(tmp0[2]);
            ptr[3] = float2int8(tmp0[3]);
            ptr[4] = float2int8(tmp1[0]);
            ptr[5] = float2int8(tmp1[1]);
            ptr[6] = float2int8(tmp1[2]);
            ptr[7] = float2int8(tmp1[3]);
            intptr += 8;
            ptr += 8;
        }
        for (; i + 3 < size; i += 4)
        {
            size_t vl = __riscv_vsetvl_e32m1(4);
            vfloat32m1_t _v = __riscv_vfcvt_f_x_v_f32m1(__riscv_vle32_v_i32m1(intptr, vl), vl);
            _v = __riscv_vfmacc_vv_f32m1(_bias0, _v, _scale_in0, vl);
            _v = activation_ps(_v, activation_type, activation_params, vl);
            _v = __riscv_vfmul_vv_f32m1(_v, _scale_out0, vl);
            float tmp[4];
            __riscv_vse32_v_f32m1(tmp, _v, vl);
            ptr[0] = float2int8(tmp[0]);
            ptr[1] = float2int8(tmp[1]);
            ptr[2] = float2int8(tmp[2]);
            ptr[3] = float2int8(tmp[3]);
            intptr += 4;
            ptr += 4;
        }
#endif // __riscv_vector
        for (; i < size; i++)
        {
            float v = *intptr * scale_in + bias;
            v = activation_ss(v, activation_type, activation_params);
            *ptr = float2int8(v * scale_out);
            intptr++;
            ptr++;
        }
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

            requantize_generic(intptr, ptr, scale_in_data, bias_data, scale_out_data, activation_type, activation_params, size, 1);
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

            requantize_generic(intptr, ptr, scale_in_data_i, bias_data_i, scale_out_data_i, activation_type, activation_params, w, elempack);
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

            requantize_generic(intptr, ptr, scale_in_data_q, bias_data_q, scale_out_data_q, activation_type, activation_params, w * h, elempack);
        }
    }

    return 0;
}

} // namespace ncnn
