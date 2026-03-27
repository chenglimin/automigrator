// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2019 THL A29 Limited, a Tencent company. All rights reserved.
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

#include <riscv_vector.h>
#include "cpu.h"

namespace ncnn {

// RVV helper functions not needed here; rely on vsetvl for dynamic VLEN

static void cast_fp32_to_fp16_rvv(const Mat& bottom_blob, Mat& top_blob, const Option& opt)
{
    // ZVFH is disabled per build options; use scalar helper for IEEE754 half conversion
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

        for (int i = 0; i < size; i++)
        {
            outptr[i] = float32_to_float16(ptr[i]);
        }
    }
}

static void cast_fp16_to_fp32_rvv(const Mat& bottom_blob, Mat& top_blob, const Option& opt)
{
    // ZVFH disabled; use scalar helper conversion
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

        for (int i = 0; i < size; i++)
        {
            outptr[i] = float16_to_float32(ptr[i]);
        }
    }
}

static void cast_bf16_to_fp32_rvv(const Mat& bottom_blob, Mat& top_blob, const Option& opt)
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

        int i = 0;
        while (i < size)
        {
            int len = size - i;
            size_t vl = __riscv_vsetvl_e16m4(len);
            vuint16m4_t vbf16 = __riscv_vle16_v_u16m4(ptr, vl);
            vuint32m8_t v32 = __riscv_vzext_vf2_u32m8(vbf16, vl);
            vuint32m8_t vshift = __riscv_vsll_vx_u32m8(v32, 16, vl);
            vfloat32m8_t vfp32 = __riscv_vreinterpret_v_u32m8_f32m8(vshift);
            __riscv_vse32_v_f32m8(outptr, vfp32, vl);
            ptr += vl;
            outptr += vl;
            i += (int)vl;
        }
    }
}

static void cast_fp32_to_bf16_rvv(const Mat& bottom_blob, Mat& top_blob, const Option& opt)
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

        int i = 0;
        while (i < size)
        {
            int len = size - i;
            size_t vl = __riscv_vsetvl_e32m8(len);
            vfloat32m8_t vfp32 = __riscv_vle32_v_f32m8(ptr, vl);
            vuint32m8_t vi32 = __riscv_vreinterpret_v_f32m8_u32m8(vfp32);
            vuint16m4_t vbf16 = __riscv_vnsrl_wx_u16m4(vi32, 16, vl);
            __riscv_vse16_v_u16m4(outptr, vbf16, vl);
            ptr += vl;
            outptr += vl;
            i += (int)vl;
        }
    }
}

Cast_riscv::Cast_riscv()
{
    support_packing = true;
    support_bf16_storage = true;
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
            // fallback to generic for int8->fp32 to reuse scalar rounding path
            Cast::forward(bottom_blob, top_blob, opt);
            return 0;
        }
        out_elemsize = 4 * elempack;
    }
    else if (type_to == 2)
    {
        out_elemsize = 2 * elempack;
    }
    else if (type_to == 3)
    {
        out_elemsize = elempack;
    }
    else if (type_to == 4)
    {
        out_elemsize = 2 * elempack;
    }

    if (dims == 1)
        top_blob.create(w, out_elemsize, elempack, opt.blob_allocator);
    else if (dims == 2)
        top_blob.create(w, h, out_elemsize, elempack, opt.blob_allocator);
    else if (dims == 3)
        top_blob.create(w, h, channels, out_elemsize, elempack, opt.blob_allocator);
    else if (dims == 4)
        top_blob.create(w, h, d, channels, out_elemsize, elempack, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    if (type_from == 1 && type_to == 2)
    {
        cast_fp32_to_fp16_rvv(bottom_blob, top_blob, opt);
        return 0;
    }

    if (type_from == 2 && type_to == 1)
    {
        cast_fp16_to_fp32_rvv(bottom_blob, top_blob, opt);
        return 0;
    }

    if (type_from == 1 && type_to == 4)
    {
        cast_fp32_to_bf16_rvv(bottom_blob, top_blob, opt);
        return 0;
    }

    if (type_from == 4 && type_to == 1)
    {
        cast_bf16_to_fp32_rvv(bottom_blob, top_blob, opt);
        return 0;
    }

    // fallback to generic if not handled
    return Cast::forward(bottom_blob, top_blob, opt);
}

} // namespace ncnn
