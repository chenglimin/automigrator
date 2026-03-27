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

#include "cast_riscv.h"

#if __riscv_vector
#include <riscv_vector.h>
#endif // __riscv_vector
#include "riscv_usability.h"

#include "cpu.h"

namespace ncnn {

Cast_riscv::Cast_riscv()
{
#if __riscv_vector
    support_packing = true;
#endif // __riscv_vector
#if NCNN_ZFH
#if __riscv_vector
    support_fp16_storage = cpu_support_riscv_zvfh();
#else
    support_fp16_storage = cpu_support_riscv_zfh();
#endif
#endif
}

static inline void cast_fp32_to_fp16_rvv(const Mat& bottom_blob, Mat& top_blob, const Option& opt)
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int d = bottom_blob.d;
    const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;
    const int size = w * h * d * elempack;

#pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        const float* ptr = bottom_blob.channel(q);
        unsigned short* outptr = top_blob.channel(q);
        int n = size;
#if __riscv_vector && __riscv_zvfh
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e32m8(n);
            vfloat32m8_t _fp32 = __riscv_vle32_v_f32m8(ptr, vl);
            // narrow to f16: convert fp32 to fp16 (store as uint16)
            vfloat16m8_t _fp16 = __riscv_vfncvt_f_f_v_f16m8(_fp32, vl);
            vuint16m8_t _u16 = __riscv_vreinterpret_v_f16m8_u16m8(_fp16);
            __riscv_vse16_v_u16m8(outptr, _u16, vl);
            ptr += vl;
            outptr += vl;
            n -= vl;
        }
#else
        for (int i = 0; i < n; i++)
        {
            outptr[i] = float32_to_float16(ptr[i]);
        }
#endif
    }
}

static inline void cast_fp16_to_fp32_rvv(const Mat& bottom_blob, Mat& top_blob, const Option& opt)
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int d = bottom_blob.d;
    const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;
    const int size = w * h * d * elempack;

#pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        const unsigned short* ptr = bottom_blob.channel(q);
        float* outptr = top_blob.channel(q);
        int n = size;
#if __riscv_vector && __riscv_zvfh
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e16m8(n);
            vuint16m8_t _u16 = __riscv_vle16_v_u16m8(ptr, vl);
            vfloat16m8_t _fp16 = __riscv_vreinterpret_v_u16m8_f16m8(_u16);
            vfloat32m8_t _fp32 = __riscv_vfwcvt_f_f_v_f32m8(_fp16, vl);
            __riscv_vse32_v_f32m8(outptr, _fp32, vl);
            ptr += vl;
            outptr += vl;
            n -= vl;
        }
#else
        for (int i = 0; i < n; i++)
        {
            outptr[i] = float16_to_float32(ptr[i]);
        }
#endif
    }
}

static inline void cast_int8_to_fp32_scalar(const Mat& bottom_blob, Mat& top_blob, const Option& opt)
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int d = bottom_blob.d;
    int channels = bottom_blob.c;
    int elempack = bottom_blob.elempack;
    int size = w * h * d * elempack;
#pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        const signed char* ptr = bottom_blob.channel(q);
        float* outptr = top_blob.channel(q);
        for (int i = 0; i < size; i++)
        {
            outptr[i] = (float)ptr[i];
        }
    }
}

static inline void cast_fp32_to_bf16_scalar(const Mat& bottom_blob, Mat& top_blob, const Option& opt)
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int d = bottom_blob.d;
    int channels = bottom_blob.c;
    int elempack = bottom_blob.elempack;
    int size = w * h * d * elempack;
#pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        const float* ptr = bottom_blob.channel(q);
        unsigned short* outptr = top_blob.channel(q);
        for (int i = 0; i < size; i++)
        {
            outptr[i] = float32_to_bfloat16(ptr[i]);
        }
    }
}

static inline void cast_bf16_to_fp32_scalar(const Mat& bottom_blob, Mat& top_blob, const Option& opt)
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int d = bottom_blob.d;
    int channels = bottom_blob.c;
    int elempack = bottom_blob.elempack;
    int size = w * h * d * elempack;
#pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        const unsigned short* ptr = bottom_blob.channel(q);
        float* outptr = top_blob.channel(q);
        for (int i = 0; i < size; i++)
        {
            outptr[i] = bfloat16_to_float32(ptr[i]);
        }
    }
}

int Cast_riscv::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    if (type_from == type_to)
    {
        top_blob = bottom_blob;
        return 0;
    }

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int d = bottom_blob.d;
    int channels = bottom_blob.c;
    int dims = bottom_blob.dims;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    size_t out_elemsize = elemsize;
    if (type_to == 1)
    {
        if (type_from == 3)
        {
            Cast::forward(bottom_blob, top_blob, opt);
        }
        // float32
        out_elemsize = 4 * elempack;
    }
    else if (type_to == 2)
    {
        // float16
        out_elemsize = 2 * elempack;
    }
    else if (type_to == 3)
    {
        // int8
        out_elemsize = elempack;
    }
    else if (type_to == 4)
    {
        // bfloat16
        out_elemsize = 2 * elempack;
    }

    if (dims == 1)
    {
        top_blob.create(w, out_elemsize, elempack, opt.blob_allocator);
    }
    else if (dims == 2)
    {
        top_blob.create(w, h, out_elemsize, elempack, opt.blob_allocator);
    }
    else if (dims == 3)
    {
        top_blob.create(w, h, channels, out_elemsize, elempack, opt.blob_allocator);
    }
    else if (dims == 4)
    {
        top_blob.create(w, h, d, channels, out_elemsize, elempack, opt.blob_allocator);
    }
    if (top_blob.empty())
        return -100;

    if (type_from == 1 && type_to == 2)
    {
        cast_fp32_to_fp16_rvv(bottom_blob, top_blob, opt);
    }

    if (type_from == 2 && type_to == 1)
    {
        cast_fp16_to_fp32_rvv(bottom_blob, top_blob, opt);
    }

    if (type_from == 3 && type_to == 1)
    {
        cast_int8_to_fp32_scalar(bottom_blob, top_blob, opt);
    }

    if (type_from == 1 && type_to == 4)
    {
        cast_fp32_to_bf16_scalar(bottom_blob, top_blob, opt);
    }

    if (type_from == 4 && type_to == 1)
    {
        cast_bf16_to_fp32_scalar(bottom_blob, top_blob, opt);
    }

    return 0;
}

} // namespace ncnn
