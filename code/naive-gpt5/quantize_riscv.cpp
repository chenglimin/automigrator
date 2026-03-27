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

#include "quantize_riscv.h"

#if __riscv_vector
#include <riscv_vector.h>
#endif // __riscv_vector

#include "riscv_usability.h"
#include <vector>
#include <math.h>

namespace ncnn {

Quantize_riscv::Quantize_riscv()
{
#if __riscv_vector
    support_packing = true;
#endif // __riscv_vector
}

static inline signed char float2int8_scalar(float v)
{
    int int32 = static_cast<int>(round(v));
    if (int32 > 127) return 127;
    if (int32 < -127) return -127;
    return (signed char)int32;
}

static void quantize(const float* ptr, signed char* s8ptr, const Mat& scale_data, int elemcount, int elempack)
{
    const int scale_data_size = scale_data.w;
    const int size = elemcount * elempack;

    float scale = scale_data[0];

    int n = size;
#if __riscv_vector
    if (scale_data_size == 1)
    {
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e32m8(n);
            vfloat32m8_t _v = __riscv_vle32_v_f32m8(ptr, vl);
            _v = __riscv_vfmul_vf_f32m8(_v, scale, vl);

            // store to tmp and scalar convert
            std::vector<float> tmp(vl);
            __riscv_vse32_v_f32m8(tmp.data(), _v, vl);
            for (size_t i = 0; i < vl; i++)
            {
                s8ptr[i] = float2int8_scalar(tmp[i]);
            }

            ptr += vl;
            s8ptr += vl;
            n -= vl;
        }
        return;
    }
#endif // __riscv_vector

    // fallback scalar path, handles per-lane scale
    for (int i = 0; i < size; i++)
    {
        float s = scale_data_size == 1 ? scale : (float)scale_data[i % elempack];
        float v = ptr[i] * s;
        s8ptr[i] = float2int8_scalar(v);
    }
}

#if __riscv_vector
static void quantize_pack4to8(const float* ptr0, const float* ptr1, signed char* s8ptr, const Mat& scale_data, int elemcount)
{
    const int scale_data_size = scale_data.w;
    float scale0 = scale_data[0];
    float scale1 = scale0;
    if (scale_data_size > 1)
    {
        scale0 = ((const float*)scale_data)[0];
        scale1 = ((const float*)scale_data)[4];
    }

    int i = 0;
    // vectorize when scale is uniform per pack
    if (scale_data_size == 1)
    {
        for (; i + 1 < elemcount; i += 2)
        {
            size_t vl = __riscv_vsetvl_e32m1(4);
            vfloat32m1_t _v0 = __riscv_vle32_v_f32m1(ptr0, vl);
            vfloat32m1_t _v1 = __riscv_vle32_v_f32m1(ptr1, vl);
            vfloat32m1_t _v2 = __riscv_vle32_v_f32m1(ptr0 + 4, vl);
            vfloat32m1_t _v3 = __riscv_vle32_v_f32m1(ptr1 + 4, vl);
            _v0 = __riscv_vfmul_vf_f32m1(_v0, scale0, vl);
            _v1 = __riscv_vfmul_vf_f32m1(_v1, scale1, vl);
            _v2 = __riscv_vfmul_vf_f32m1(_v2, scale0, vl);
            _v3 = __riscv_vfmul_vf_f32m1(_v3, scale1, vl);

            float tmp0[4], tmp1[4], tmp2[4], tmp3[4];
            __riscv_vse32_v_f32m1(tmp0, _v0, vl);
            __riscv_vse32_v_f32m1(tmp1, _v1, vl);
            __riscv_vse32_v_f32m1(tmp2, _v2, vl);
            __riscv_vse32_v_f32m1(tmp3, _v3, vl);

            for (int k = 0; k < 4; k++) s8ptr[k] = float2int8_scalar(tmp0[k]);
            for (int k = 0; k < 4; k++) s8ptr[4 + k] = float2int8_scalar(tmp1[k]);
            for (int k = 0; k < 4; k++) s8ptr[8 + k] = float2int8_scalar(tmp2[k]);
            for (int k = 0; k < 4; k++) s8ptr[12 + k] = float2int8_scalar(tmp3[k]);

            ptr0 += 8;
            ptr1 += 8;
            s8ptr += 16;
        }
    }
    // scalar fallback for remaining
    for (; i < elemcount; i++)
    {
        for (int k = 0; k < 4; k++) s8ptr[k] = float2int8_scalar(ptr0[k] * scale0);
        for (int k = 0; k < 4; k++) s8ptr[4 + k] = float2int8_scalar(ptr1[k] * scale1);
        ptr0 += 4;
        ptr1 += 4;
        s8ptr += 8;
    }
}

static void quantize_pack4to1(const float* ptr, signed char* s8ptr0, signed char* s8ptr1, signed char* s8ptr2, signed char* s8ptr3, const Mat& scale_data, int elemcount)
{
    const int scale_data_size = scale_data.w;
    float scale = scale_data[0];

    int i = 0;
    if (scale_data_size == 1)
    {
        for (; i + 7 < elemcount; i += 8)
        {
            size_t vl = __riscv_vsetvl_e32m1(4);
            vfloat32m1_t _v0 = __riscv_vle32_v_f32m1(ptr, vl);
            vfloat32m1_t _v1 = __riscv_vle32_v_f32m1(ptr + 4, vl);
            vfloat32m1_t _v2 = __riscv_vle32_v_f32m1(ptr + 8, vl);
            vfloat32m1_t _v3 = __riscv_vle32_v_f32m1(ptr + 12, vl);
            vfloat32m1_t _v4 = __riscv_vle32_v_f32m1(ptr + 16, vl);
            vfloat32m1_t _v5 = __riscv_vle32_v_f32m1(ptr + 20, vl);
            vfloat32m1_t _v6 = __riscv_vle32_v_f32m1(ptr + 24, vl);
            vfloat32m1_t _v7 = __riscv_vle32_v_f32m1(ptr + 28, vl);
            _v0 = __riscv_vfmul_vf_f32m1(_v0, scale, vl);
            _v1 = __riscv_vfmul_vf_f32m1(_v1, scale, vl);
            _v2 = __riscv_vfmul_vf_f32m1(_v2, scale, vl);
            _v3 = __riscv_vfmul_vf_f32m1(_v3, scale, vl);
            _v4 = __riscv_vfmul_vf_f32m1(_v4, scale, vl);
            _v5 = __riscv_vfmul_vf_f32m1(_v5, scale, vl);
            _v6 = __riscv_vfmul_vf_f32m1(_v6, scale, vl);
            _v7 = __riscv_vfmul_vf_f32m1(_v7, scale, vl);

            float t0[4], t1[4], t2[4], t3[4], t4[4], t5[4], t6[4], t7[4];
            __riscv_vse32_v_f32m1(t0, _v0, vl);
            __riscv_vse32_v_f32m1(t1, _v1, vl);
            __riscv_vse32_v_f32m1(t2, _v2, vl);
            __riscv_vse32_v_f32m1(t3, _v3, vl);
            __riscv_vse32_v_f32m1(t4, _v4, vl);
            __riscv_vse32_v_f32m1(t5, _v5, vl);
            __riscv_vse32_v_f32m1(t6, _v6, vl);
            __riscv_vse32_v_f32m1(t7, _v7, vl);

            s8ptr0[0] = float2int8_scalar(t0[0]);
            s8ptr1[0] = float2int8_scalar(t0[1]);
            s8ptr2[0] = float2int8_scalar(t0[2]);
            s8ptr3[0] = float2int8_scalar(t0[3]);
            s8ptr0[1] = float2int8_scalar(t1[0]);
            s8ptr1[1] = float2int8_scalar(t1[1]);
            s8ptr2[1] = float2int8_scalar(t1[2]);
            s8ptr3[1] = float2int8_scalar(t1[3]);
            s8ptr0[2] = float2int8_scalar(t2[0]);
            s8ptr1[2] = float2int8_scalar(t2[1]);
            s8ptr2[2] = float2int8_scalar(t2[2]);
            s8ptr3[2] = float2int8_scalar(t2[3]);
            s8ptr0[3] = float2int8_scalar(t3[0]);
            s8ptr1[3] = float2int8_scalar(t3[1]);
            s8ptr2[3] = float2int8_scalar(t3[2]);
            s8ptr3[3] = float2int8_scalar(t3[3]);

            s8ptr0 += 4;
            s8ptr1 += 4;
            s8ptr2 += 4;
            s8ptr3 += 4;
            ptr += 32;
        }
    }
    for (; i < elemcount; i++)
    {
        s8ptr0[0] = float2int8_scalar(ptr[0] * scale);
        s8ptr1[0] = float2int8_scalar(ptr[1] * scale);
        s8ptr2[0] = float2int8_scalar(ptr[2] * scale);
        s8ptr3[0] = float2int8_scalar(ptr[3] * scale);
        ptr += 4;
        s8ptr0 += 1;
        s8ptr1 += 1;
        s8ptr2 += 1;
        s8ptr3 += 1;
    }
}
#endif // __riscv_vector

int Quantize_riscv::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    const int dims = bottom_blob.dims;
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;

    if (dims == 1)
    {
        int out_elempack = 1;
#if __riscv_vector
        if (opt.use_packing_layout)
        {
            out_elempack = w * elempack % 8 == 0 ? 8 : 1;
        }
#endif
        const int outw = w * elempack / out_elempack;
        const size_t out_elemsize = out_elempack * 1u;

        top_blob.create(outw, out_elemsize, out_elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        const int wp = std::max(1, w / opt.num_threads);
        const int nn_w = (w + wp - 1) / wp;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < nn_w; ii++)
        {
            const int i = ii * wp;

            const float* ptr = (const float*)bottom_blob + i * elempack;
            signed char* s8ptr = (signed char*)top_blob + i * elempack;

            const int size = std::min(w - i, wp) * elempack;

            quantize(ptr, s8ptr, scale_data, size, 1);
        }
    }

    if (dims == 2)
    {
        int out_elempack = 1;
#if __riscv_vector
        if (opt.use_packing_layout)
        {
            out_elempack = h * elempack % 8 == 0 ? 8 : 1;
        }
#endif
        const int outh = h * elempack / out_elempack;
        const size_t out_elemsize = out_elempack * 1u;

        top_blob.create(w, outh, out_elemsize, out_elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

#if __riscv_vector
        if (elempack == 4 && out_elempack == 8)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < outh; i++)
            {
                const float* ptr0 = bottom_blob.row(i * 2);
                const float* ptr1 = bottom_blob.row(i * 2 + 1);
                signed char* s8ptr = top_blob.row<signed char>(i);

                const Mat scale_data_i = scale_data_size > 1 ? scale_data.range(i * out_elempack, out_elempack) : scale_data;

                quantize_pack4to8(ptr0, ptr1, s8ptr, scale_data_i, w);
            }
        }
        if (elempack == 4 && out_elempack == 1)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                const float* ptr = bottom_blob.row(i);
                signed char* s8ptr0 = top_blob.row<signed char>(i * 4);
                signed char* s8ptr1 = top_blob.row<signed char>(i * 4 + 1);
                signed char* s8ptr2 = top_blob.row<signed char>(i * 4 + 2);
                signed char* s8ptr3 = top_blob.row<signed char>(i * 4 + 3);

                const Mat scale_data_i = scale_data_size > 1 ? scale_data.range(i * elempack, elempack) : scale_data;

                quantize_pack4to1(ptr, s8ptr0, s8ptr1, s8ptr2, s8ptr3, scale_data_i, w);
            }
        }
#endif // __riscv_vector
        if (elempack == out_elempack)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                const float* ptr = bottom_blob.row(i);
                signed char* s8ptr = top_blob.row<signed char>(i);

                const Mat scale_data_i = scale_data_size > 1 ? scale_data.range(i * elempack, elempack) : scale_data;

                quantize(ptr, s8ptr, scale_data_i, w, elempack);
            }
        }
    }

    if (dims == 3)
    {
        int out_elempack = 1;
#if __riscv_vector
        if (opt.use_packing_layout)
        {
            out_elempack = channels * elempack % 8 == 0 ? 8 : 1;
        }
#endif
        const int outc = channels * elempack / out_elempack;
        const size_t out_elemsize = out_elempack * 1u;

        top_blob.create(w, h, outc, out_elemsize, out_elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

#if __riscv_vector
        if (elempack == 4 && out_elempack == 8)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < outc; q++)
            {
                const float* ptr0 = bottom_blob.channel(q * 2);
                const float* ptr1 = bottom_blob.channel(q * 2 + 1);
                signed char* s8ptr = top_blob.channel(q);

                const Mat scale_data_q = scale_data_size > 1 ? scale_data.range(q * out_elempack, out_elempack) : scale_data;

                quantize_pack4to8(ptr0, ptr1, s8ptr, scale_data_q, w * h);
            }
        }
        if (elempack == 4 && out_elempack == 1)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = bottom_blob.channel(q);
                signed char* s8ptr0 = top_blob.channel(q * 4);
                signed char* s8ptr1 = top_blob.channel(q * 4 + 1);
                signed char* s8ptr2 = top_blob.channel(q * 4 + 2);
                signed char* s8ptr3 = top_blob.channel(q * 4 + 3);

                const Mat scale_data_q = scale_data_size > 1 ? scale_data.range(q * elempack, elempack) : scale_data;

                quantize_pack4to1(ptr, s8ptr0, s8ptr1, s8ptr2, s8ptr3, scale_data_q, w * h);
            }
        }
#endif // __riscv_vector
        if (elempack == out_elempack)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = bottom_blob.channel(q);
                signed char* s8ptr = top_blob.channel(q);

                const Mat scale_data_q = scale_data_size > 1 ? scale_data.range(q * elempack, elempack) : scale_data;

                quantize(ptr, s8ptr, scale_data_q, w * h, elempack);
            }
        }
    }

    return 0;
}

} // namespace ncnn
