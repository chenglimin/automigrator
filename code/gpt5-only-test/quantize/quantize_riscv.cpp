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

namespace ncnn {

Quantize_riscv::Quantize_riscv()
{
#if __riscv_vector
    support_packing = true;
#endif // __riscv_vector
}

static inline signed char float2int8_scalar(float v)
{
    int int32 = (int)round(v);
    if (int32 > 127) return 127;
    if (int32 < -127) return -127;
    return (signed char)int32;
}

static void quantize_scalar(const float* ptr, signed char* s8ptr, const Mat& scale_data, int elemcount, int elempack)
{
    const int size = elemcount * elempack;
    const float scale = scale_data[0];
    for (int i = 0; i < size; i++)
    {
        s8ptr[i] = float2int8_scalar(ptr[i] * scale);
    }
}

#if __riscv_vector
static void quantize_v(const float* ptr, signed char* s8ptr, const Mat& scale_data, int elemcount, int elempack)
{
    const int size = elemcount * elempack;
    const int scale_data_size = scale_data.w;

    float scale = scale_data[0];
    int n = size;
    if (scale_data_size == 1)
    {
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e32m8(n);
            vfloat32m8_t _v = __riscv_vle32_v_f32m8(ptr, vl);
            _v = __riscv_vfmul_vf_f32m8(_v, scale, vl);
            // store to temp then scalar convert as we lack int8 pack here
            std::vector<float> tmp(vl);
            __riscv_vse32_v_f32m8(tmp.data(), _v, vl);
            for (size_t i = 0; i < vl; i++)
                s8ptr[i] = float2int8_scalar(tmp[i]);
            ptr += vl;
            s8ptr += vl;
            n -= vl;
        }
    }
    else
    {
        // per-pack scales: load elempack scales once then loop
        // only elempack==4 or 8 need vectorized scale load; fall back to scalar compute for generality
        quantize_scalar(ptr, s8ptr, scale_data, elemcount, elempack);
    }
}
#endif // __riscv_vector

static void quantize_dispatch(const float* ptr, signed char* s8ptr, const Mat& scale_data, int elemcount, int elempack)
{
#if __riscv_vector
    quantize_v(ptr, s8ptr, scale_data, elemcount, elempack);
#else
    quantize_scalar(ptr, s8ptr, scale_data, elemcount, elempack);
#endif
}

int Quantize_riscv::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    const int dims = bottom_blob.dims;
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;
    const int scale_data_size = scale_data.w;

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
            const Mat scale_data_i = scale_data_size > 1 ? scale_data.range(i * elempack, elempack) : scale_data;
            quantize_dispatch(ptr, s8ptr, scale_data_i, size / elempack, elempack);
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

        if (elempack == out_elempack)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                const float* ptr = bottom_blob.row(i);
                signed char* s8ptr = top_blob.row<signed char>(i);
                const Mat scale_data_i = scale_data_size > 1 ? scale_data.range(i * elempack, elempack) : scale_data;
                quantize_dispatch(ptr, s8ptr, scale_data_i, w, elempack);
            }
        }
        else if (elempack == 4 && out_elempack == 1)
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
                for (int j = 0; j < w; j++)
                {
                    const float* p = ptr + j * 4;
                    signed char v0 = float2int8_scalar(p[0] * scale_data_i[0]);
                    signed char v1 = float2int8_scalar(p[1] * scale_data_i[1 % scale_data_i.w]);
                    signed char v2 = float2int8_scalar(p[2] * scale_data_i[2 % scale_data_i.w]);
                    signed char v3 = float2int8_scalar(p[3] * scale_data_i[3 % scale_data_i.w]);
                    s8ptr0[j] = v0;
                    s8ptr1[j] = v1;
                    s8ptr2[j] = v2;
                    s8ptr3[j] = v3;
                }
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

        if (elempack == out_elempack)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = bottom_blob.channel(q);
                signed char* s8ptr = top_blob.channel(q);
                const Mat scale_data_q = scale_data_size > 1 ? scale_data.range(q * elempack, elempack) : scale_data;
                quantize_dispatch(ptr, s8ptr, scale_data_q, w * h, elempack);
            }
        }
        else if (elempack == 4 && out_elempack == 1)
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
                for (int i = 0; i < w * h; i++)
                {
                    const float* p = ptr + i * 4;
                    s8ptr0[i] = float2int8_scalar(p[0] * scale_data_q[0]);
                    s8ptr1[i] = float2int8_scalar(p[1] * scale_data_q[1 % scale_data_q.w]);
                    s8ptr2[i] = float2int8_scalar(p[2] * scale_data_q[2 % scale_data_q.w]);
                    s8ptr3[i] = float2int8_scalar(p[3] * scale_data_q[3 % scale_data_q.w]);
                }
            }
        }
    }

    return 0;
}

} // namespace ncnn
