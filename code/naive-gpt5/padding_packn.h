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

#ifndef NCNN_PADDING_PACKN_RVV_H
#define NCNN_PADDING_PACKN_RVV_H

#include "riscv_usability.h"

namespace ncnn {

// Fill with constant value vector v (packn floats)
static void padding_constant_packn_rvv(const Mat& src, Mat& dst, int top, int bottom, int left, int right, const float* v)
{
    const float* ptr = src;
    float* outptr = dst;
    const int packn = csrr_vlenb() / 4;

#if __riscv_vector
    size_t vl = __riscv_vsetvl_e32m1(packn);
    vfloat32m1_t _v = __riscv_vle32_v_f32m1(v, vl);
#endif
    int top_size = top * dst.w;
    int bottom_size = bottom * dst.w;

    // fill top
    for (int y = 0; y < top_size; y++)
    {
#if __riscv_vector
        __riscv_vse32_v_f32m1(outptr, _v, vl);
        outptr += packn;
#else
        for (int i = 0; i < packn; i++) outptr[i] = v[i];
        outptr += packn;
#endif
    }
    // fill center
    for (int y = 0; y < src.h; y++)
    {
        for (int x = 0; x < left; x++)
        {
#if __riscv_vector
            __riscv_vse32_v_f32m1(outptr, _v, vl);
            outptr += packn;
#else
            for (int i = 0; i < packn; i++) outptr[i] = v[i];
            outptr += packn;
#endif
        }
        for (int x = 0; x < src.w; x++)
        {
#if __riscv_vector
            vfloat32m1_t _p = __riscv_vle32_v_f32m1(ptr, vl);
            __riscv_vse32_v_f32m1(outptr, _p, vl);
            ptr += packn;
            outptr += packn;
#else
            for (int i = 0; i < packn; i++) outptr[i] = ptr[i];
            ptr += packn;
            outptr += packn;
#endif
        }
        for (int x = 0; x < right; x++)
        {
#if __riscv_vector
            __riscv_vse32_v_f32m1(outptr, _v, vl);
            outptr += packn;
#else
            for (int i = 0; i < packn; i++) outptr[i] = v[i];
            outptr += packn;
#endif
        }
    }
    // fill bottom
    for (int y = 0; y < bottom_size; y++)
    {
#if __riscv_vector
        __riscv_vse32_v_f32m1(outptr, _v, vl);
        outptr += packn;
#else
        for (int i = 0; i < packn; i++) outptr[i] = v[i];
        outptr += packn;
#endif
    }
}

// Replicate border values
static void padding_replicate_packn_rvv(const Mat& src, Mat& dst, int top, int bottom, int left, int right)
{
    const float* ptr = src;
    float* outptr = dst;
    const int packn = csrr_vlenb() / 4;
#if __riscv_vector
    size_t vl = __riscv_vsetvl_e32m1(packn);
#endif
    // fill top (replicate first row)
    for (int y = 0; y < top; y++)
    {
        const float* ptr0 = ptr;
#if __riscv_vector
        vfloat32m1_t _p = __riscv_vle32_v_f32m1(ptr0, vl);
#endif
        for (int x = 0; x < left; x++)
        {
#if __riscv_vector
            __riscv_vse32_v_f32m1(outptr, _p, vl);
            outptr += packn;
#else
            for (int i = 0; i < packn; i++) outptr[i] = ptr0[i];
            outptr += packn;
#endif
        }
        for (int x = 0; x < src.w; x++)
        {
#if __riscv_vector
            _p = __riscv_vle32_v_f32m1(ptr0, vl);
            __riscv_vse32_v_f32m1(outptr, _p, vl);
            ptr0 += packn;
            outptr += packn;
#else
            for (int i = 0; i < packn; i++) outptr[i] = ptr0[i];
            ptr0 += packn;
            outptr += packn;
#endif
        }
        for (int x = 0; x < right; x++)
        {
#if __riscv_vector
            __riscv_vse32_v_f32m1(outptr, _p, vl);
            outptr += packn;
#else
            for (int i = 0; i < packn; i++) outptr[i] = ptr0[i - packn];
            outptr += packn;
#endif
        }
    }
    // fill center
    for (int y = 0; y < src.h; y++)
    {
#if __riscv_vector
        vfloat32m1_t _p = __riscv_vle32_v_f32m1(ptr, vl);
#endif
        for (int x = 0; x < left; x++)
        {
#if __riscv_vector
            __riscv_vse32_v_f32m1(outptr, _p, vl);
            outptr += packn;
#else
            for (int i = 0; i < packn; i++) outptr[i] = ptr[i];
            outptr += packn;
#endif
        }
        for (int x = 0; x < src.w; x++)
        {
#if __riscv_vector
            _p = __riscv_vle32_v_f32m1(ptr, vl);
            __riscv_vse32_v_f32m1(outptr, _p, vl);
            ptr += packn;
            outptr += packn;
#else
            for (int i = 0; i < packn; i++) outptr[i] = ptr[i];
            ptr += packn;
            outptr += packn;
#endif
        }
        for (int x = 0; x < right; x++)
        {
#if __riscv_vector
            __riscv_vse32_v_f32m1(outptr, _p, vl);
            outptr += packn;
#else
            for (int i = 0; i < packn; i++) outptr[i] = ptr[i - packn];
            outptr += packn;
#endif
        }
    }
    // fill bottom (replicate last row)
    ptr -= src.w * packn;
    for (int y = 0; y < bottom; y++)
    {
        const float* ptr0 = ptr;
#if __riscv_vector
        vfloat32m1_t _p = __riscv_vle32_v_f32m1(ptr0, vl);
#endif
        for (int x = 0; x < left; x++)
        {
#if __riscv_vector
            __riscv_vse32_v_f32m1(outptr, _p, vl);
            outptr += packn;
#else
            for (int i = 0; i < packn; i++) outptr[i] = ptr0[i];
            outptr += packn;
#endif
        }
        for (int x = 0; x < src.w; x++)
        {
#if __riscv_vector
            _p = __riscv_vle32_v_f32m1(ptr0, vl);
            __riscv_vse32_v_f32m1(outptr, _p, vl);
            ptr0 += packn;
            outptr += packn;
#else
            for (int i = 0; i < packn; i++) outptr[i] = ptr0[i];
            ptr0 += packn;
            outptr += packn;
#endif
        }
        for (int x = 0; x < right; x++)
        {
#if __riscv_vector
            __riscv_vse32_v_f32m1(outptr, _p, vl);
            outptr += packn;
#else
            for (int i = 0; i < packn; i++) outptr[i] = ptr0[i - packn];
            outptr += packn;
#endif
        }
    }
}

// Reflect border values
static void padding_reflect_packn_rvv(const Mat& src, Mat& dst, int top, int bottom, int left, int right)
{
    const float* ptr = src;
    float* outptr = dst;
    const int packn = csrr_vlenb() / 4;
#if __riscv_vector
    size_t vl = __riscv_vsetvl_e32m1(packn);
#endif
    // fill top (reflect rows above)
    ptr += top * src.w * packn;
    for (int y = 0; y < top; y++)
    {
        const float* ptr0 = ptr;
        for (int x = 0; x < left; x++)
        {
            const float* p = ptr0 + (left - x) * packn;
#if __riscv_vector
            vfloat32m1_t _p = __riscv_vle32_v_f32m1(p, vl);
            __riscv_vse32_v_f32m1(outptr, _p, vl);
            outptr += packn;
#else
            for (int i = 0; i < packn; i++) outptr[i] = p[i];
            outptr += packn;
#endif
        }
        for (int x = 0; x < src.w; x++)
        {
#if __riscv_vector
            vfloat32m1_t _p = __riscv_vle32_v_f32m1(ptr0, vl);
            __riscv_vse32_v_f32m1(outptr, _p, vl);
            ptr0 += packn;
            outptr += packn;
#else
            for (int i = 0; i < packn; i++) outptr[i] = ptr0[i];
            ptr0 += packn;
            outptr += packn;
#endif
        }
        for (int x = 0; x < right; x++)
        {
            const float* p = ptr0 - 2 * packn - x * packn;
#if __riscv_vector
            vfloat32m1_t _p = __riscv_vle32_v_f32m1(p, vl);
            __riscv_vse32_v_f32m1(outptr, _p, vl);
            outptr += packn;
#else
            for (int i = 0; i < packn; i++) outptr[i] = p[i];
            outptr += packn;
#endif
        }
        ptr -= src.w * packn;
    }
    // fill center
    for (int y = 0; y < src.h; y++)
    {
        for (int x = 0; x < left; x++)
        {
            const float* p = ptr + (left - x) * packn;
#if __riscv_vector
            vfloat32m1_t _p = __riscv_vle32_v_f32m1(p, vl);
            __riscv_vse32_v_f32m1(outptr, _p, vl);
            outptr += packn;
#else
            for (int i = 0; i < packn; i++) outptr[i] = p[i];
            outptr += packn;
#endif
        }
        for (int x = 0; x < src.w; x++)
        {
#if __riscv_vector
            vfloat32m1_t _p = __riscv_vle32_v_f32m1(ptr, vl);
            __riscv_vse32_v_f32m1(outptr, _p, vl);
            ptr += packn;
            outptr += packn;
#else
            for (int i = 0; i < packn; i++) outptr[i] = ptr[i];
            ptr += packn;
            outptr += packn;
#endif
        }
        for (int x = 0; x < right; x++)
        {
            const float* p = ptr - 2 * packn - x * packn;
#if __riscv_vector
            vfloat32m1_t _p = __riscv_vle32_v_f32m1(p, vl);
            __riscv_vse32_v_f32m1(outptr, _p, vl);
            outptr += packn;
#else
            for (int i = 0; i < packn; i++) outptr[i] = p[i];
            outptr += packn;
#endif
        }
    }
    // fill bottom
    ptr -= 2 * src.w * packn;
    for (int y = 0; y < bottom; y++)
    {
        const float* ptr0 = ptr;
        for (int x = 0; x < left; x++)
        {
            const float* p = ptr0 + (left - x) * packn;
#if __riscv_vector
            vfloat32m1_t _p = __riscv_vle32_v_f32m1(p, vl);
            __riscv_vse32_v_f32m1(outptr, _p, vl);
            outptr += packn;
#else
            for (int i = 0; i < packn; i++) outptr[i] = p[i];
            outptr += packn;
#endif
        }
        for (int x = 0; x < src.w; x++)
        {
#if __riscv_vector
            vfloat32m1_t _p = __riscv_vle32_v_f32m1(ptr0, vl);
            __riscv_vse32_v_f32m1(outptr, _p, vl);
            ptr0 += packn;
            outptr += packn;
#else
            for (int i = 0; i < packn; i++) outptr[i] = ptr0[i];
            ptr0 += packn;
            outptr += packn;
#endif
        }
        for (int x = 0; x < right; x++)
        {
            const float* p = ptr0 - 2 * packn - x * packn;
#if __riscv_vector
            vfloat32m1_t _p = __riscv_vle32_v_f32m1(p, vl);
            __riscv_vse32_v_f32m1(outptr, _p, vl);
            outptr += packn;
#else
            for (int i = 0; i < packn; i++) outptr[i] = p[i];
            outptr += packn;
#endif
        }
        ptr -= src.w * packn;
    }
}

} // namespace ncnn

#endif // NCNN_PADDING_PACKN_RVV_H
