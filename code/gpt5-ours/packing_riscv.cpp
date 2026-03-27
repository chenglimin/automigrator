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

#include "packing_riscv.h"

#if __riscv_vector
#include <riscv_vector.h>
#endif // __riscv_vector

#include "riscv_usability.h"

namespace ncnn {

Packing_riscv::Packing_riscv()
{
#if __riscv_vector
    support_packing = true;
#endif // __riscv_vector
}

static inline void pack_transpose4(float* outptr, const float* r0, const float* r1, const float* r2, const float* r3, int w)
{
#if __riscv_vector
    int n = w;
    while (n > 0)
    {
        size_t vl = __riscv_vsetvl_e32m1(n);
        vfloat32m1_t _r0 = __riscv_vle32_v_f32m1(r0, vl);
        vfloat32m1_t _r1 = __riscv_vle32_v_f32m1(r1, vl);
        vfloat32m1_t _r2 = __riscv_vle32_v_f32m1(r2, vl);
        vfloat32m1_t _r3 = __riscv_vle32_v_f32m1(r3, vl);
        vfloat32m1x4_t _p = __riscv_vcreate_v_f32m1x4(_r0, _r1, _r2, _r3);
        __riscv_vsseg4e32_v_f32m1x4(outptr, _p, vl);
        r0 += vl;
        r1 += vl;
        r2 += vl;
        r3 += vl;
        outptr += 4 * vl;
        n -= vl;
    }
#else
    for (int j = 0; j < w; j++)
    {
        outptr[0] = r0[0];
        outptr[1] = r1[0];
        outptr[2] = r2[0];
        outptr[3] = r3[0];
        r0++;
        r1++;
        r2++;
        r3++;
        outptr += 4;
    }
#endif
}

int Packing_riscv::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int elembits = bottom_blob.elembits();

    if (elembits == 8)
        return forward_int8(bottom_blob, top_blob, opt);

    if (use_padding)
    {
        return Packing::forward(bottom_blob, top_blob, opt);
    }

    if (elembits != 32)
    {
        // non-fp32 type
        return Packing::forward(bottom_blob, top_blob, opt);
    }

    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    if (elempack == out_elempack)
    {
        top_blob = bottom_blob;
        return 0;
    }

    bool pack1toN = elempack == 1 && (out_elempack == 4 || out_elempack == 8 || out_elempack == 16);
    bool packNto1 = (elempack == 4 || elempack == 8 || elempack == 16) && out_elempack == 1;
    bool pack4to8 = elempack == 4 && out_elempack == 8;
    bool pack8to4 = elempack == 8 && out_elempack == 4;
    bool pack4to16 = elempack == 4 && out_elempack == 16;
    bool pack16to4 = elempack == 16 && out_elempack == 4;
    bool pack8to16 = elempack == 8 && out_elempack == 16;
    bool pack16to8 = elempack == 16 && out_elempack == 8;

    if (!pack1toN && !packNto1 && !pack4to8 && !pack8to4 && !pack4to16 && !pack16to4 && !pack8to16 && !pack16to8)
    {
        return Packing::forward(bottom_blob, top_blob, opt);
    }

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int d = bottom_blob.d;
    int channels = bottom_blob.c;
    int dims = bottom_blob.dims;

    if (!use_padding)
    {
        if (dims == 1 && w * elempack % out_elempack != 0)
        {
            top_blob = bottom_blob;
            return 0;
        }
        if (dims == 2 && h * elempack % out_elempack != 0)
        {
            top_blob = bottom_blob;
            return 0;
        }
        if ((dims == 3 || dims == 4) && channels * elempack % out_elempack != 0)
        {
            top_blob = bottom_blob;
            return 0;
        }
    }

    if (dims == 1)
    {
        top_blob = bottom_blob;
        top_blob.w = w * elempack / out_elempack;
        top_blob.cstep = w * elempack / out_elempack;
        top_blob.elemsize = elemsize / elempack * out_elempack;
        top_blob.elempack = out_elempack;
        return 0;
    }

    if (dims == 2)
    {
        int outh = h * elempack / out_elempack;
        size_t out_elemsize = elemsize / elempack * out_elempack;

        top_blob.create(w, outh, out_elemsize, out_elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        if (elempack == 1 && out_elempack == 4)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < outh; i++)
            {
                const float* r0 = bottom_blob.row(i * 4);
                const float* r1 = bottom_blob.row(i * 4 + 1);
                const float* r2 = bottom_blob.row(i * 4 + 2);
                const float* r3 = bottom_blob.row(i * 4 + 3);
                float* outptr = top_blob.row(i);
                pack_transpose4(outptr, r0, r1, r2, r3, w);
            }
        }
        else if (elempack == 4 && out_elempack == 1)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                const float* r0 = bottom_blob.row(i);
                float* outptr0 = top_blob.row(i * 4);
                float* outptr1 = top_blob.row(i * 4 + 1);
                float* outptr2 = top_blob.row(i * 4 + 2);
                float* outptr3 = top_blob.row(i * 4 + 3);
#if __riscv_vector
                int n = w;
                while (n > 0)
                {
                    size_t vl = __riscv_vsetvl_e32m1(n);
                    // segmented load 4 interleaved lanes
                    vfloat32m1x4_t _p = __riscv_vlseg4e32_v_f32m1x4(r0, vl);
                    vfloat32m1_t _r0 = __riscv_vget_v_f32m1x4_f32m1(_p, 0);
                    vfloat32m1_t _r1 = __riscv_vget_v_f32m1x4_f32m1(_p, 1);
                    vfloat32m1_t _r2 = __riscv_vget_v_f32m1x4_f32m1(_p, 2);
                    vfloat32m1_t _r3 = __riscv_vget_v_f32m1x4_f32m1(_p, 3);
                    __riscv_vse32_v_f32m1(outptr0, _r0, vl);
                    __riscv_vse32_v_f32m1(outptr1, _r1, vl);
                    __riscv_vse32_v_f32m1(outptr2, _r2, vl);
                    __riscv_vse32_v_f32m1(outptr3, _r3, vl);
                    r0 += 4 * vl;
                    outptr0 += vl;
                    outptr1 += vl;
                    outptr2 += vl;
                    outptr3 += vl;
                    n -= vl;
                }
#else
                for (int j = 0; j < w; j++)
                {
                    *outptr0++ = r0[0];
                    *outptr1++ = r0[1];
                    *outptr2++ = r0[2];
                    *outptr3++ = r0[3];
                    r0 += 4;
                }
#endif
            }
        }
        else
        {
            // fallback generic for other directions
            return Packing::forward(bottom_blob, top_blob, opt);
        }

        return 0;
    }

    if (dims == 3 || dims == 4)
    {
        int size = w * h * d;
        int outc = channels * elempack / out_elempack;
        size_t out_elemsize = elemsize / elempack * out_elempack;

        if (dims == 3)
            top_blob.create(w, h, outc, out_elemsize, out_elempack, opt.blob_allocator);
        else
            top_blob.create(w, h, d, outc, out_elemsize, out_elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        if (elempack == 1 && out_elempack == 4)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < outc; q++)
            {
                const float* r0 = bottom_blob.channel(q * 4);
                const float* r1 = bottom_blob.channel(q * 4 + 1);
                const float* r2 = bottom_blob.channel(q * 4 + 2);
                const float* r3 = bottom_blob.channel(q * 4 + 3);
                float* outptr = top_blob.channel(q);
                pack_transpose4(outptr, r0, r1, r2, r3, size);
            }
        }
        else if (elempack == 4 && out_elempack == 1)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* r0 = bottom_blob.channel(q);
                float* outptr0 = top_blob.channel(q * 4);
                float* outptr1 = top_blob.channel(q * 4 + 1);
                float* outptr2 = top_blob.channel(q * 4 + 2);
                float* outptr3 = top_blob.channel(q * 4 + 3);
#if __riscv_vector
                int n = size;
                while (n > 0)
                {
                    size_t vl = __riscv_vsetvl_e32m1(n);
                    vfloat32m1x4_t _p = __riscv_vlseg4e32_v_f32m1x4(r0, vl);
                    vfloat32m1_t _r0 = __riscv_vget_v_f32m1x4_f32m1(_p, 0);
                    vfloat32m1_t _r1 = __riscv_vget_v_f32m1x4_f32m1(_p, 1);
                    vfloat32m1_t _r2 = __riscv_vget_v_f32m1x4_f32m1(_p, 2);
                    vfloat32m1_t _r3 = __riscv_vget_v_f32m1x4_f32m1(_p, 3);
                    __riscv_vse32_v_f32m1(outptr0, _r0, vl);
                    __riscv_vse32_v_f32m1(outptr1, _r1, vl);
                    __riscv_vse32_v_f32m1(outptr2, _r2, vl);
                    __riscv_vse32_v_f32m1(outptr3, _r3, vl);
                    r0 += 4 * vl;
                    outptr0 += vl;
                    outptr1 += vl;
                    outptr2 += vl;
                    outptr3 += vl;
                    n -= vl;
                }
#else
                for (int i = 0; i < size; i++)
                {
                    *outptr0++ = r0[0];
                    *outptr1++ = r0[1];
                    *outptr2++ = r0[2];
                    *outptr3++ = r0[3];
                    r0 += 4;
                }
#endif
            }
        }
        else
        {
            return Packing::forward(bottom_blob, top_blob, opt);
        }

        return 0;
    }

    return Packing::forward(bottom_blob, top_blob, opt);
}

int Packing_riscv::forward_int8(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    // Use generic fallback for int8, per spec priority is fp32 paths
    return Packing::forward(bottom_blob, top_blob, opt);
}

} // namespace ncnn
