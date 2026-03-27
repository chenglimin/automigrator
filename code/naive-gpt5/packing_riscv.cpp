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

#include "riscv_usability.h"

namespace ncnn {

Packing_riscv::Packing_riscv()
{
#if __riscv_vector
    support_packing = true;
#endif // __riscv_vector
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

    bool pack1to4 = elempack == 1 && out_elempack == 4;
    bool pack4to1 = elempack == 4 && out_elempack == 1;
    bool pack1to8 = elempack == 1 && out_elempack == 8;
    bool pack8to1 = elempack == 8 && out_elempack == 1;
    bool pack4to8 = elempack == 4 && out_elempack == 8;
    bool pack8to4 = elempack == 8 && out_elempack == 4;
    bool pack1to16 = elempack == 1 && out_elempack == 16;
    bool pack16to1 = elempack == 16 && out_elempack == 1;
    bool pack4to16 = elempack == 4 && out_elempack == 16;
    bool pack16to4 = elempack == 16 && out_elempack == 4;
    bool pack8to16 = elempack == 8 && out_elempack == 16;
    bool pack16to8 = elempack == 16 && out_elempack == 8;

    if (!pack1to4 && !pack4to1 && !pack1to8 && !pack8to1 && !pack4to8 && !pack8to4 && !pack1to16 && !pack16to1 && !pack4to16 && !pack16to4 && !pack8to16 && !pack16to8)
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
        // identity if use_padding not allowed
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

        if (pack1to4)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < outh; i++)
            {
                const float* r0 = bottom_blob.row(i * 4);
                const float* r1 = bottom_blob.row(i * 4 + 1);
                const float* r2 = bottom_blob.row(i * 4 + 2);
                const float* r3 = bottom_blob.row(i * 4 + 3);

                float* outptr = top_blob.row(i);

                int j = 0;
#if __riscv_vector
                for (; j + 3 < w; j += 4)
                {
                    size_t vl = __riscv_vsetvl_e32m1(4);
                    vfloat32m1_t _r0 = __riscv_vle32_v_f32m1(r0, vl);
                    vfloat32m1_t _r1 = __riscv_vle32_v_f32m1(r1, vl);
                    vfloat32m1_t _r2 = __riscv_vle32_v_f32m1(r2, vl);
                    vfloat32m1_t _r3 = __riscv_vle32_v_f32m1(r3, vl);
                    transpose4x4_ps(_r0, _r1, _r2, _r3, vl);
                    __riscv_vse32_v_f32m1(outptr, _r0, vl);
                    __riscv_vse32_v_f32m1(outptr + 4, _r1, vl);
                    __riscv_vse32_v_f32m1(outptr + 8, _r2, vl);
                    __riscv_vse32_v_f32m1(outptr + 12, _r3, vl);
                    r0 += 4; r1 += 4; r2 += 4; r3 += 4; outptr += 16;
                }
#endif // __riscv_vector
                for (; j < w; j++)
                {
                    outptr[0] = *r0++;
                    outptr[1] = *r1++;
                    outptr[2] = *r2++;
                    outptr[3] = *r3++;

                    outptr += 4;
                }
            }
        }
        if (pack4to1)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                const float* r0 = bottom_blob.row(i);

                float* outptr0 = top_blob.row(i * 4);
                float* outptr1 = top_blob.row(i * 4 + 1);
                float* outptr2 = top_blob.row(i * 4 + 2);
                float* outptr3 = top_blob.row(i * 4 + 3);

                int j = 0;
#if __riscv_vector
                for (; j + 3 < w; j += 4)
                {
                    size_t vl = __riscv_vsetvl_e32m1(4);
                    vfloat32m1_t _r0 = __riscv_vle32_v_f32m1(r0, vl);
                    vfloat32m1_t _r1 = __riscv_vle32_v_f32m1(r0 + 4, vl);
                    vfloat32m1_t _r2 = __riscv_vle32_v_f32m1(r0 + 8, vl);
                    vfloat32m1_t _r3 = __riscv_vle32_v_f32m1(r0 + 12, vl);
                    transpose4x4_ps(_r0, _r1, _r2, _r3, vl);
                    __riscv_vse32_v_f32m1(outptr0, _r0, vl);
                    __riscv_vse32_v_f32m1(outptr1, _r1, vl);
                    __riscv_vse32_v_f32m1(outptr2, _r2, vl);
                    __riscv_vse32_v_f32m1(outptr3, _r3, vl);
                    r0 += 16; outptr0 += 4; outptr1 += 4; outptr2 += 4; outptr3 += 4;
                }
#endif // __riscv_vector
                for (; j < w; j++)
                {
                    *outptr0++ = r0[0];
                    *outptr1++ = r0[1];
                    *outptr2++ = r0[2];
                    *outptr3++ = r0[3];

                    r0 += 4;
                }
            }
        }
        if (pack1to8)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < outh; i++)
            {
                const float* r0 = bottom_blob.row(i * 8);
                const float* r1 = bottom_blob.row(i * 8 + 1);
                const float* r2 = bottom_blob.row(i * 8 + 2);
                const float* r3 = bottom_blob.row(i * 8 + 3);
                const float* r4 = bottom_blob.row(i * 8 + 4);
                const float* r5 = bottom_blob.row(i * 8 + 5);
                const float* r6 = bottom_blob.row(i * 8 + 6);
                const float* r7 = bottom_blob.row(i * 8 + 7);

                float* outptr = top_blob.row(i);

                int j = 0;
#if __riscv_vector
                for (; j + 7 < w; j += 8)
                {
                    size_t vl = __riscv_vsetvl_e32m1(4);
                    vfloat32m1_t _r0l = __riscv_vle32_v_f32m1(r0, vl);
                    vfloat32m1_t _r0h = __riscv_vle32_v_f32m1(r0 + 4, vl);
                    vfloat32m1_t _r1l = __riscv_vle32_v_f32m1(r1, vl);
                    vfloat32m1_t _r1h = __riscv_vle32_v_f32m1(r1 + 4, vl);
                    vfloat32m1_t _r2l = __riscv_vle32_v_f32m1(r2, vl);
                    vfloat32m1_t _r2h = __riscv_vle32_v_f32m1(r2 + 4, vl);
                    vfloat32m1_t _r3l = __riscv_vle32_v_f32m1(r3, vl);
                    vfloat32m1_t _r3h = __riscv_vle32_v_f32m1(r3 + 4, vl);
                    vfloat32m1_t _r4l = __riscv_vle32_v_f32m1(r4, vl);
                    vfloat32m1_t _r4h = __riscv_vle32_v_f32m1(r4 + 4, vl);
                    vfloat32m1_t _r5l = __riscv_vle32_v_f32m1(r5, vl);
                    vfloat32m1_t _r5h = __riscv_vle32_v_f32m1(r5 + 4, vl);
                    vfloat32m1_t _r6l = __riscv_vle32_v_f32m1(r6, vl);
                    vfloat32m1_t _r6h = __riscv_vle32_v_f32m1(r6 + 4, vl);
                    vfloat32m1_t _r7l = __riscv_vle32_v_f32m1(r7, vl);
                    vfloat32m1_t _r7h = __riscv_vle32_v_f32m1(r7 + 4, vl);
                    transpose8x8_ps(_r0l, _r0h, _r1l, _r1h, _r2l, _r2h, _r3l, _r3h, _r4l, _r4h, _r5l, _r5h, _r6l, _r6h, _r7l, _r7h, vl);
                    __riscv_vse32_v_f32m1(outptr, _r0l, vl);
                    __riscv_vse32_v_f32m1(outptr + 8, _r0h, vl);
                    __riscv_vse32_v_f32m1(outptr + 16, _r1l, vl);
                    __riscv_vse32_v_f32m1(outptr + 24, _r1h, vl);
                    __riscv_vse32_v_f32m1(outptr + 32, _r2l, vl);
                    __riscv_vse32_v_f32m1(outptr + 40, _r2h, vl);
                    __riscv_vse32_v_f32m1(outptr + 48, _r3l, vl);
                    __riscv_vse32_v_f32m1(outptr + 56, _r3h, vl);
                    __riscv_vse32_v_f32m1(outptr + 64, _r4l, vl);
                    __riscv_vse32_v_f32m1(outptr + 72, _r4h, vl);
                    __riscv_vse32_v_f32m1(outptr + 80, _r5l, vl);
                    __riscv_vse32_v_f32m1(outptr + 88, _r5h, vl);
                    __riscv_vse32_v_f32m1(outptr + 96, _r6l, vl);
                    __riscv_vse32_v_f32m1(outptr + 104, _r6h, vl);
                    __riscv_vse32_v_f32m1(outptr + 112, _r7l, vl);
                    __riscv_vse32_v_f32m1(outptr + 120, _r7h, vl);
                    r0 += 8; r1 += 8; r2 += 8; r3 += 8; r4 += 8; r5 += 8; r6 += 8; r7 += 8; outptr += 64;
                }
#endif // __riscv_vector
                for (; j < w; j++)
                {
                    outptr[0] = *r0++;
                    outptr[1] = *r1++;
                    outptr[2] = *r2++;
                    outptr[3] = *r3++;
                    outptr[4] = *r4++;
                    outptr[5] = *r5++;
                    outptr[6] = *r6++;
                    outptr[7] = *r7++;

                    outptr += 8;
                }
            }
        }
        if (pack8to1)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                const float* r0 = bottom_blob.row(i);

                float* outptr0 = top_blob.row(i * 8);
                float* outptr1 = top_blob.row(i * 8 + 1);
                float* outptr2 = top_blob.row(i * 8 + 2);
                float* outptr3 = top_blob.row(i * 8 + 3);
                float* outptr4 = top_blob.row(i * 8 + 4);
                float* outptr5 = top_blob.row(i * 8 + 5);
                float* outptr6 = top_blob.row(i * 8 + 6);
                float* outptr7 = top_blob.row(i * 8 + 7);

                int j = 0;
#if __riscv_vector
                for (; j + 7 < w; j += 8)
                {
                    size_t vl = __riscv_vsetvl_e32m1(4);
                    vfloat32m1_t _r0l = __riscv_vle32_v_f32m1(r0, vl);
                    vfloat32m1_t _r0h = __riscv_vle32_v_f32m1(r0 + 4, vl);
                    vfloat32m1_t _r1l = __riscv_vle32_v_f32m1(r0 + 8, vl);
                    vfloat32m1_t _r1h = __riscv_vle32_v_f32m1(r0 + 12, vl);
                    vfloat32m1_t _r2l = __riscv_vle32_v_f32m1(r0 + 16, vl);
                    vfloat32m1_t _r2h = __riscv_vle32_v_f32m1(r0 + 20, vl);
                    vfloat32m1_t _r3l = __riscv_vle32_v_f32m1(r0 + 24, vl);
                    vfloat32m1_t _r3h = __riscv_vle32_v_f32m1(r0 + 28, vl);
                    vfloat32m1_t _r4l = __riscv_vle32_v_f32m1(r0 + 32, vl);
                    vfloat32m1_t _r4h = __riscv_vle32_v_f32m1(r0 + 36, vl);
                    vfloat32m1_t _r5l = __riscv_vle32_v_f32m1(r0 + 40, vl);
                    vfloat32m1_t _r5h = __riscv_vle32_v_f32m1(r0 + 44, vl);
                    vfloat32m1_t _r6l = __riscv_vle32_v_f32m1(r0 + 48, vl);
                    vfloat32m1_t _r6h = __riscv_vle32_v_f32m1(r0 + 52, vl);
                    vfloat32m1_t _r7l = __riscv_vle32_v_f32m1(r0 + 56, vl);
                    vfloat32m1_t _r7h = __riscv_vle32_v_f32m1(r0 + 60, vl);
                    transpose8x8_ps(_r0l, _r0h, _r1l, _r1h, _r2l, _r2h, _r3l, _r3h, _r4l, _r4h, _r5l, _r5h, _r6l, _r6h, _r7l, _r7h, vl);
                    __riscv_vse32_v_f32m1(outptr0, _r0l, vl);
                    __riscv_vse32_v_f32m1(outptr1, _r0h, vl);
                    __riscv_vse32_v_f32m1(outptr2, _r1l, vl);
                    __riscv_vse32_v_f32m1(outptr3, _r1h, vl);
                    __riscv_vse32_v_f32m1(outptr4, _r2l, vl);
                    __riscv_vse32_v_f32m1(outptr5, _r2h, vl);
                    __riscv_vse32_v_f32m1(outptr6, _r3l, vl);
                    __riscv_vse32_v_f32m1(outptr7, _r3h, vl);

                    r0 += 64;
                    outptr0 += 8;
                    outptr1 += 8;
                    outptr2 += 8;
                    outptr3 += 8;
                    outptr4 += 8;
                    outptr5 += 8;
                    outptr6 += 8;
                    outptr7 += 8;
                }
#endif // __riscv_vector
                for (; j < w; j++)
                {
                    *outptr0++ = r0[0];
                    *outptr1++ = r0[1];
                    *outptr2++ = r0[2];
                    *outptr3++ = r0[3];
                    *outptr4++ = r0[4];
                    *outptr5++ = r0[5];
                    *outptr6++ = r0[6];
                    *outptr7++ = r0[7];

                    r0 += 8;
                }
            }
        }
        if (pack4to8)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < outh; i++)
            {
                const float* r0 = bottom_blob.row(i * 2);
                const float* r1 = bottom_blob.row(i * 2 + 1);

                float* outptr = top_blob.row(i);

                for (int j = 0; j < w; j++)
                {
                    outptr[0] = r0[0];
                    outptr[1] = r0[1];
                    outptr[2] = r0[2];
                    outptr[3] = r0[3];
                    outptr[4] = r1[0];
                    outptr[5] = r1[1];
                    outptr[6] = r1[2];
                    outptr[7] = r1[3];

                    r0 += 4;
                    r1 += 4;
                    outptr += 8;
                }
            }
        }
        if (pack8to4)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                const float* r0 = bottom_blob.row(i);

                float* outptr0 = top_blob.row(i * 2);
                float* outptr1 = top_blob.row(i * 2 + 1);

                for (int j = 0; j < w; j++)
                {
                    outptr0[0] = r0[0];
                    outptr0[1] = r0[1];
                    outptr0[2] = r0[2];
                    outptr0[3] = r0[3];
                    outptr1[0] = r0[4];
                    outptr1[1] = r0[5];
                    outptr1[2] = r0[6];
                    outptr1[3] = r0[7];

                    r0 += 8;
                    outptr0 += 4;
                    outptr1 += 4;
                }
            }
        }
        if (pack1to16)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < outh; i++)
            {
                const float* r0 = bottom_blob.row(i * 16);
                const float* r1 = bottom_blob.row(i * 16 + 1);
                const float* r2 = bottom_blob.row(i * 16 + 2);
                const float* r3 = bottom_blob.row(i * 16 + 3);
                const float* r4 = bottom_blob.row(i * 16 + 4);
                const float* r5 = bottom_blob.row(i * 16 + 5);
                const float* r6 = bottom_blob.row(i * 16 + 6);
                const float* r7 = bottom_blob.row(i * 16 + 7);
                const float* r8 = bottom_blob.row(i * 16 + 8);
                const float* r9 = bottom_blob.row(i * 16 + 9);
                const float* ra = bottom_blob.row(i * 16 + 10);
                const float* rb = bottom_blob.row(i * 16 + 11);
                const float* rc = bottom_blob.row(i * 16 + 12);
                const float* rd = bottom_blob.row(i * 16 + 13);
                const float* re = bottom_blob.row(i * 16 + 14);
                const float* rf = bottom_blob.row(i * 16 + 15);

                float* outptr = top_blob.row(i);

                int j = 0;
#if 0
                for (; j + 15 < w; j += 16)
                {
                    size_t vl = __riscv_vsetvl_e32m1(4);
                    vfloat32m1_t _r0l = __riscv_vle32_v_f32m1(r0, vl);
                    vfloat32m1_t _r0h = __riscv_vle32_v_f32m1(r0 + 4, vl);
                    vfloat32m1_t _r1l = __riscv_vle32_v_f32m1(r1, vl);
                    vfloat32m1_t _r1h = __riscv_vle32_v_f32m1(r1 + 4, vl);
                    vfloat32m1_t _r2l = __riscv_vle32_v_f32m1(r2, vl);
                    vfloat32m1_t _r2h = __riscv_vle32_v_f32m1(r2 + 4, vl);
                    vfloat32m1_t _r3l = __riscv_vle32_v_f32m1(r3, vl);
                    vfloat32m1_t _r3h = __riscv_vle32_v_f32m1(r3 + 4, vl);
                    vfloat32m1_t _r4l = __riscv_vle32_v_f32m1(r4, vl);
                    vfloat32m1_t _r4h = __riscv_vle32_v_f32m1(r4 + 4, vl);
                    vfloat32m1_t _r5l = __riscv_vle32_v_f32m1(r5, vl);
                    vfloat32m1_t _r5h = __riscv_vle32_v_f32m1(r5 + 4, vl);
                    vfloat32m1_t _r6l = __riscv_vle32_v_f32m1(r6, vl);
                    vfloat32m1_t _r6h = __riscv_vle32_v_f32m1(r6 + 4, vl);
                    vfloat32m1_t _r7l = __riscv_vle32_v_f32m1(r7, vl);
                    vfloat32m1_t _r7h = __riscv_vle32_v_f32m1(r7 + 4, vl);
                    vfloat32m1_t _r8l = __riscv_vle32_v_f32m1(r8, vl);
                    vfloat32m1_t _r8h = __riscv_vle32_v_f32m1(r8 + 4, vl);
                    vfloat32m1_t _r9l = __riscv_vle32_v_f32m1(r9, vl);
                    vfloat32m1_t _r9h = __riscv_vle32_v_f32m1(r9 + 4, vl);
                    vfloat32m1_t _ral = __riscv_vle32_v_f32m1(ra, vl);
                    vfloat32m1_t _rah = __riscv_vle32_v_f32m1(ra + 4, vl);
                    vfloat32m1_t _rbl = __riscv_vle32_v_f32m1(rb, vl);
                    vfloat32m1_t _rbh = __riscv_vle32_v_f32m1(rb + 4, vl);
                    vfloat32m1_t _rcl = __riscv_vle32_v_f32m1(rc, vl);
                    vfloat32m1_t _rch = __riscv_vle32_v_f32m1(rc + 4, vl);
                    vfloat32m1_t _rdl = __riscv_vle32_v_f32m1(rd, vl);
                    vfloat32m1_t _rdh = __riscv_vle32_v_f32m1(rd + 4, vl);
                    vfloat32m1_t _rel = __riscv_vle32_v_f32m1(re, vl);
                    vfloat32m1_t _reh = __riscv_vle32_v_f32m1(re + 4, vl);
                    vfloat32m1_t _rfl = __riscv_vle32_v_f32m1(rf, vl);
                    vfloat32m1_t _rfh = __riscv_vle32_v_f32m1(rf + 4, vl);
                    transpose8x16_ps(_r0l, _r0h, _r1l, _r1h, _r2l, _r2h, _r3l, _r3h, _r4l, _r4h, _r5l, _r5h, _r6l, _r6h, _r7l, _r7h, _r8l, _r8h, _r9l, _r9h, _ral, _rah, _rbl, _rbh, _rcl, _rch, _rdl, _rdh, _rel, _reh, _rfl, _rfh, vl);
                    __riscv_vse32_v_f32m1(outptr, _r0l, vl);
                    __riscv_vse32_v_f32m1(outptr + 16, _r0h, vl);
                    __riscv_vse32_v_f32m1(outptr + 32, _r1l, vl);
                    __riscv_vse32_v_f32m1(outptr + 48, _r1h, vl);
                    __riscv_vse32_v_f32m1(outptr + 64, _r2l, vl);
                    __riscv_vse32_v_f32m1(outptr + 80, _r2h, vl);
                    __riscv_vse32_v_f32m1(outptr + 96, _r3l, vl);
                    __riscv_vse32_v_f32m1(outptr + 112, _r3h, vl);
                    __riscv_vse32_v_f32m1(outptr + 128, _r4l, vl);
                    __riscv_vse32_v_f32m1(outptr + 144, _r4h, vl);
                    __riscv_vse32_v_f32m1(outptr + 160, _r5l, vl);
                    __riscv_vse32_v_f32m1(outptr + 176, _r5h, vl);
                    __riscv_vse32_v_f32m1(outptr + 192, _r6l, vl);
                    __riscv_vse32_v_f32m1(outptr + 208, _r6h, vl);
                    __riscv_vse32_v_f32m1(outptr + 224, _r7l, vl);
                    __riscv_vse32_v_f32m1(outptr + 240, _r7h, vl);
                    __riscv_vse32_v_f32m1(outptr + 256, _r8l, vl);
                    __riscv_vse32_v_f32m1(outptr + 272, _r8h, vl);
                    __riscv_vse32_v_f32m1(outptr + 288, _r9l, vl);
                    __riscv_vse32_v_f32m1(outptr + 304, _r9h, vl);
                    __riscv_vse32_v_f32m1(outptr + 320, _ral, vl);
                    __riscv_vse32_v_f32m1(outptr + 336, _rah, vl);
                    __riscv_vse32_v_f32m1(outptr + 352, _rbl, vl);
                    __riscv_vse32_v_f32m1(outptr + 368, _rbh, vl);
                    __riscv_vse32_v_f32m1(outptr + 384, _rcl, vl);
                    __riscv_vse32_v_f32m1(outptr + 400, _rch, vl);
                    __riscv_vse32_v_f32m1(outptr + 416, _rdl, vl);
                    __riscv_vse32_v_f32m1(outptr + 432, _rdh, vl);
                    __riscv_vse32_v_f32m1(outptr + 448, _rel, vl);
                    __riscv_vse32_v_f32m1(outptr + 464, _reh, vl);
                    __riscv_vse32_v_f32m1(outptr + 480, _rfl, vl);
                    __riscv_vse32_v_f32m1(outptr + 496, _rfh, vl);
                    r0 += 16; r1 += 16; r2 += 16; r3 += 16; r4 += 16; r5 += 16; r6 += 16; r7 += 16; r8 += 16; r9 += 16; ra += 16; rb += 16; rc += 16; rd += 16; re += 16; rf += 16; outptr += 256;
                }
#endif // __riscv_vector
                for (; j < w; j++)
                {
                    outptr[0] = *r0++;
                    outptr[1] = *r1++;
                    outptr[2] = *r2++;
                    outptr[3] = *r3++;
                    outptr[4] = *r4++;
                    outptr[5] = *r5++;
                    outptr[6] = *r6++;
                    outptr[7] = *r7++;
                    outptr[8] = *r8++;
                    outptr[9] = *r9++;
                    outptr[10] = *ra++;
                    outptr[11] = *rb++;
                    outptr[12] = *rc++;
                    outptr[13] = *rd++;
                    outptr[14] = *re++;
                    outptr[15] = *rf++;

                    outptr += 16;
                }
            }
        }
        if (pack16to1)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                const float* r0 = bottom_blob.row(i);

                float* outptr0 = top_blob.row(i * 16);
                float* outptr1 = top_blob.row(i * 16 + 1);
                float* outptr2 = top_blob.row(i * 16 + 2);
                float* outptr3 = top_blob.row(i * 16 + 3);
                float* outptr4 = top_blob.row(i * 16 + 4);
                float* outptr5 = top_blob.row(i * 16 + 5);
                float* outptr6 = top_blob.row(i * 16 + 6);
                float* outptr7 = top_blob.row(i * 16 + 7);
                float* outptr8 = top_blob.row(i * 16 + 8);
                float* outptr9 = top_blob.row(i * 16 + 9);
                float* outptra = top_blob.row(i * 16 + 10);
                float* outptrb = top_blob.row(i * 16 + 11);
                float* outptrc = top_blob.row(i * 16 + 12);
                float* outptrd = top_blob.row(i * 16 + 13);
                float* outptre = top_blob.row(i * 16 + 14);
                float* outptrf = top_blob.row(i * 16 + 15);

                int j = 0;
#if 0
                for (; j + 15 < w; j += 16)
                {
                    size_t vl = __riscv_vsetvl_e32m1(4);
                    vfloat32m1_t _r0l = __riscv_vle32_v_f32m1(r0, vl);
                    vfloat32m1_t _r0h = __riscv_vle32_v_f32m1(r0 + 4, vl);
                    vfloat32m1_t _r1l = __riscv_vle32_v_f32m1(r0 + 8, vl);
                    vfloat32m1_t _r1h = __riscv_vle32_v_f32m1(r0 + 12, vl);
                    vfloat32m1_t _r2l = __riscv_vle32_v_f32m1(r0 + 16, vl);
                    vfloat32m1_t _r2h = __riscv_vle32_v_f32m1(r0 + 20, vl);
                    vfloat32m1_t _r3l = __riscv_vle32_v_f32m1(r0 + 24, vl);
                    vfloat32m1_t _r3h = __riscv_vle32_v_f32m1(r0 + 28, vl);
                    vfloat32m1_t _r4l = __riscv_vle32_v_f32m1(r0 + 32, vl);
                    vfloat32m1_t _r4h = __riscv_vle32_v_f32m1(r0 + 36, vl);
                    vfloat32m1_t _r5l = __riscv_vle32_v_f32m1(r0 + 40, vl);
                    vfloat32m1_t _r5h = __riscv_vle32_v_f32m1(r0 + 44, vl);
                    vfloat32m1_t _r6l = __riscv_vle32_v_f32m1(r0 + 48, vl);
                    vfloat32m1_t _r6h = __riscv_vle32_v_f32m1(r0 + 52, vl);
                    vfloat32m1_t _r7l = __riscv_vle32_v_f32m1(r0 + 56, vl);
                    vfloat32m1_t _r7h = __riscv_vle32_v_f32m1(r0 + 60, vl);
                    vfloat32m1_t _r8l = __riscv_vle32_v_f32m1(r0 + 64, vl);
                    vfloat32m1_t _r8h = __riscv_vle32_v_f32m1(r0 + 68, vl);
                    vfloat32m1_t _r9l = __riscv_vle32_v_f32m1(r0 + 72, vl);
                    vfloat32m1_t _r9h = __riscv_vle32_v_f32m1(r0 + 76, vl);
                    vfloat32m1_t _ral = __riscv_vle32_v_f32m1(r0 + 80, vl);
                    vfloat32m1_t _rah = __riscv_vle32_v_f32m1(r0 + 84, vl);
                    vfloat32m1_t _rbl = __riscv_vle32_v_f32m1(r0 + 88, vl);
                    vfloat32m1_t _rbh = __riscv_vle32_v_f32m1(r0 + 92, vl);
                    vfloat32m1_t _rcl = __riscv_vle32_v_f32m1(r0 + 96, vl);
                    vfloat32m1_t _rch = __riscv_vle32_v_f32m1(r0 + 100, vl);
                    vfloat32m1_t _rdl = __riscv_vle32_v_f32m1(r0 + 104, vl);
                    vfloat32m1_t _rdh = __riscv_vle32_v_f32m1(r0 + 108, vl);
                    vfloat32m1_t _rel = __riscv_vle32_v_f32m1(r0 + 112, vl);
                    vfloat32m1_t _reh = __riscv_vle32_v_f32m1(r0 + 116, vl);
                    vfloat32m1_t _rfl = __riscv_vle32_v_f32m1(r0 + 120, vl);
                    vfloat32m1_t _rfh = __riscv_vle32_v_f32m1(r0 + 124, vl);
                    transpose8x16_ps(_r0l, _r0h, _r1l, _r1h, _r2l, _r2h, _r3l, _r3h, _r4l, _r4h, _r5l, _r5h, _r6l, _r6h, _r7l, _r7h, _r8l, _r8h, _r9l, _r9h, _ral, _rah, _rbl, _rbh, _rcl, _rch, _rdl, _rdh, _rel, _reh, _rfl, _rfh, vl);
                    __riscv_vse32_v_f32m1(outptr0, _r0l, vl);
                    __riscv_vse32_v_f32m1(outptr1, _r0h, vl);
                    __riscv_vse32_v_f32m1(outptr2, _r1l, vl);
                    __riscv_vse32_v_f32m1(outptr3, _r1h, vl);
                    __riscv_vse32_v_f32m1(outptr4, _r2l, vl);
                    __riscv_vse32_v_f32m1(outptr5, _r2h, vl);
                    __riscv_vse32_v_f32m1(outptr6, _r3l, vl);
                    __riscv_vse32_v_f32m1(outptr7, _r3h, vl);
                    __riscv_vse32_v_f32m1(outptr8, _r4l, vl);
                    __riscv_vse32_v_f32m1(outptr9, _r4h, vl);
                    __riscv_vse32_v_f32m1(outptra, _r5l, vl);
                    __riscv_vse32_v_f32m1(outptrb, _r5h, vl);
                    __riscv_vse32_v_f32m1(outptrc, _r6l, vl);
                    __riscv_vse32_v_f32m1(outptrd, _r6h, vl);
                    __riscv_vse32_v_f32m1(outptre, _r7l, vl);
                    __riscv_vse32_v_f32m1(outptrf, _r7h, vl);

                    r0 += 256;
                    outptr0 += 16;
                    outptr1 += 16;
                    outptr2 += 16;
                    outptr3 += 16;
                    outptr4 += 16;
                    outptr5 += 16;
                    outptr6 += 16;
                    outptr7 += 16;
                    outptr8 += 16;
                    outptr9 += 16;
                    outptra += 16;
                    outptrb += 16;
                    outptrc += 16;
                    outptrd += 16;
                    outptre += 16;
                    outptrf += 16;
                }
#endif // __riscv_vector
                for (; j < w; j++)
                {
                    *outptr0++ = r0[0];
                    *outptr1++ = r0[1];
                    *outptr2++ = r0[2];
                    *outptr3++ = r0[3];
                    *outptr4++ = r0[4];
                    *outptr5++ = r0[5];
                    *outptr6++ = r0[6];
                    *outptr7++ = r0[7];
                    *outptr8++ = r0[8];
                    *outptr9++ = r0[9];
                    *outptra++ = r0[10];
                    *outptrb++ = r0[11];
                    *outptrc++ = r0[12];
                    *outptrd++ = r0[13];
                    *outptre++ = r0[14];
                    *outptrf++ = r0[15];

                    r0 += 16;
                }
            }
        }
        if (pack4to16)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < outh; i++)
            {
                const float* r0 = bottom_blob.row(i * 4);
                const float* r1 = bottom_blob.row(i * 4 + 1);
                const float* r2 = bottom_blob.row(i * 4 + 2);
                const float* r3 = bottom_blob.row(i * 4 + 3);

                float* outptr = top_blob.row(i);

                for (int j = 0; j < w; j++)
                {
                    outptr[0] = r0[0];
                    outptr[1] = r0[1];
                    outptr[2] = r0[2];
                    outptr[3] = r0[3];
                    outptr[4] = r1[0];
                    outptr[5] = r1[1];
                    outptr[6] = r1[2];
                    outptr[7] = r1[3];
                    outptr[8] = r2[0];
                    outptr[9] = r2[1];
                    outptr[10] = r2[2];
                    outptr[11] = r2[3];
                    outptr[12] = r3[0];
                    outptr[13] = r3[1];
                    outptr[14] = r3[2];
                    outptr[15] = r3[3];

                    r0 += 4;
                    r1 += 4;
                    r2 += 4;
                    r3 += 4;
                    outptr += 16;
                }
            }
        }
        if (pack16to4)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                const float* r0 = bottom_blob.row(i);

                float* outptr0 = top_blob.row(i * 4);
                float* outptr1 = top_blob.row(i * 4 + 1);
                float* outptr2 = top_blob.row(i * 4 + 2);
                float* outptr3 = top_blob.row(i * 4 + 3);

                for (int j = 0; j < w; j++)
                {
                    outptr0[0] = r0[0];
                    outptr0[1] = r0[1];
                    outptr0[2] = r0[2];
                    outptr0[3] = r0[3];
                    outptr1[0] = r0[4];
                    outptr1[1] = r0[5];
                    outptr1[2] = r0[6];
                    outptr1[3] = r0[7];
                    outptr2[0] = r0[8];
                    outptr2[1] = r0[9];
                    outptr2[2] = r0[10];
                    outptr2[3] = r0[11];
                    outptr3[0] = r0[12];
                    outptr3[1] = r0[13];
                    outptr3[2] = r0[14];
                    outptr3[3] = r0[15];

                    r0 += 16;
                    outptr0 += 4;
                    outptr1 += 4;
                    outptr2 += 4;
                    outptr3 += 4;
                }
            }
        }
        if (pack8to16)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < outh; i++)
            {
                const float* r0 = bottom_blob.row(i * 2);
                const float* r1 = bottom_blob.row(i * 2 + 1);

                float* outptr = top_blob.row(i);

                for (int j = 0; j < w; j++)
                {
                    outptr[0] = r0[0];
                    outptr[1] = r0[1];
                    outptr[2] = r0[2];
                    outptr[3] = r0[3];
                    outptr[4] = r0[4];
                    outptr[5] = r0[5];
                    outptr[6] = r0[6];
                    outptr[7] = r0[7];
                    outptr[8] = r1[0];
                    outptr[9] = r1[1];
                    outptr[10] = r1[2];
                    outptr[11] = r1[3];
                    outptr[12] = r1[4];
                    outptr[13] = r1[5];
                    outptr[14] = r1[6];
                    outptr[15] = r1[7];

                    r0 += 8;
                    r1 += 8;
                    outptr += 16;
                }
            }
        }
        if (pack16to8)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                const float* r0 = bottom_blob.row(i);

                float* outptr0 = top_blob.row(i * 2);
                float* outptr1 = top_blob.row(i * 2 + 1);

                for (int j = 0; j < w; j++)
                {
                    outptr0[0] = r0[0];
                    outptr0[1] = r0[1];
                    outptr0[2] = r0[2];
                    outptr0[3] = r0[3];
                    outptr0[4] = r0[4];
                    outptr0[5] = r0[5];
                    outptr0[6] = r0[6];
                    outptr0[7] = r0[7];
                    outptr1[0] = r0[8];
                    outptr1[1] = r0[9];
                    outptr1[2] = r0[10];
                    outptr1[3] = r0[11];
                    outptr1[4] = r0[12];
                    outptr1[5] = r0[13];
                    outptr1[6] = r0[14];
                    outptr1[7] = r0[15];

                    r0 += 16;
                    outptr0 += 8;
                    outptr1 += 8;
                }
            }
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
        else // if (dims == 4)
            top_blob.create(w, h, d, outc, out_elemsize, out_elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        // Reuse scalar fallbacks with some RVV acceleration where feasible
        // For simplicity, we keep the scalar loops identical to x86 version
        // and accelerate 4x4 and 8x8 blocks using helper transposes.

        if (pack1to4)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < outc; q++)
            {
                const float* r0 = bottom_blob.channel(q * 4);
                const float* r1 = bottom_blob.channel(q * 4 + 1);
                const float* r2 = bottom_blob.channel(q * 4 + 2);
                const float* r3 = bottom_blob.channel(q * 4 + 3);

                float* outptr = top_blob.channel(q);

                int i = 0;
#if __riscv_vector
                for (; i + 3 < size; i += 4)
                {
                    size_t vl = __riscv_vsetvl_e32m1(4);
                    vfloat32m1_t _r0 = __riscv_vle32_v_f32m1(r0, vl);
                    vfloat32m1_t _r1 = __riscv_vle32_v_f32m1(r1, vl);
                    vfloat32m1_t _r2 = __riscv_vle32_v_f32m1(r2, vl);
                    vfloat32m1_t _r3 = __riscv_vle32_v_f32m1(r3, vl);
                    transpose4x4_ps(_r0, _r1, _r2, _r3, vl);
                    __riscv_vse32_v_f32m1(outptr, _r0, vl);
                    __riscv_vse32_v_f32m1(outptr + 4, _r1, vl);
                    __riscv_vse32_v_f32m1(outptr + 8, _r2, vl);
                    __riscv_vse32_v_f32m1(outptr + 12, _r3, vl);
                    r0 += 4; r1 += 4; r2 += 4; r3 += 4; outptr += 16;
                }
#endif // __riscv_vector
                for (; i < size; i++)
                {
                    outptr[0] = *r0++;
                    outptr[1] = *r1++;
                    outptr[2] = *r2++;
                    outptr[3] = *r3++;

                    outptr += 4;
                }
            }
        }
        if (pack4to1)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* r0 = bottom_blob.channel(q);

                float* outptr0 = top_blob.channel(q * 4);
                float* outptr1 = top_blob.channel(q * 4 + 1);
                float* outptr2 = top_blob.channel(q * 4 + 2);
                float* outptr3 = top_blob.channel(q * 4 + 3);

                int i = 0;
#if __riscv_vector
                for (; i + 3 < size; i += 4)
                {
                    size_t vl = __riscv_vsetvl_e32m1(4);
                    vfloat32m1_t _r0 = __riscv_vle32_v_f32m1(r0, vl);
                    vfloat32m1_t _r1 = __riscv_vle32_v_f32m1(r0 + 4, vl);
                    vfloat32m1_t _r2 = __riscv_vle32_v_f32m1(r0 + 8, vl);
                    vfloat32m1_t _r3 = __riscv_vle32_v_f32m1(r0 + 12, vl);
                    transpose4x4_ps(_r0, _r1, _r2, _r3, vl);
                    __riscv_vse32_v_f32m1(outptr0, _r0, vl);
                    __riscv_vse32_v_f32m1(outptr1, _r1, vl);
                    __riscv_vse32_v_f32m1(outptr2, _r2, vl);
                    __riscv_vse32_v_f32m1(outptr3, _r3, vl);
                    r0 += 16; outptr0 += 4; outptr1 += 4; outptr2 += 4; outptr3 += 4;
                }
#endif // __riscv_vector
                for (; i < size; i++)
                {
                    *outptr0++ = r0[0];
                    *outptr1++ = r0[1];
                    *outptr2++ = r0[2];
                    *outptr3++ = r0[3];

                    r0 += 4;
                }
            }
        }
        if (pack1to8)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < outc; q++)
            {
                const float* r0 = bottom_blob.channel(q * 8);
                const float* r1 = bottom_blob.channel(q * 8 + 1);
                const float* r2 = bottom_blob.channel(q * 8 + 2);
                const float* r3 = bottom_blob.channel(q * 8 + 3);
                const float* r4 = bottom_blob.channel(q * 8 + 4);
                const float* r5 = bottom_blob.channel(q * 8 + 5);
                const float* r6 = bottom_blob.channel(q * 8 + 6);
                const float* r7 = bottom_blob.channel(q * 8 + 7);

                float* outptr = top_blob.channel(q);

                int i = 0;
#if __riscv_vector
                for (; i + 7 < size; i += 8)
                {
                    size_t vl = __riscv_vsetvl_e32m1(4);
                    vfloat32m1_t _r0l = __riscv_vle32_v_f32m1(r0, vl);
                    vfloat32m1_t _r0h = __riscv_vle32_v_f32m1(r0 + 4, vl);
                    vfloat32m1_t _r1l = __riscv_vle32_v_f32m1(r1, vl);
                    vfloat32m1_t _r1h = __riscv_vle32_v_f32m1(r1 + 4, vl);
                    vfloat32m1_t _r2l = __riscv_vle32_v_f32m1(r2, vl);
                    vfloat32m1_t _r2h = __riscv_vle32_v_f32m1(r2 + 4, vl);
                    vfloat32m1_t _r3l = __riscv_vle32_v_f32m1(r3, vl);
                    vfloat32m1_t _r3h = __riscv_vle32_v_f32m1(r3 + 4, vl);
                    vfloat32m1_t _r4l = __riscv_vle32_v_f32m1(r4, vl);
                    vfloat32m1_t _r4h = __riscv_vle32_v_f32m1(r4 + 4, vl);
                    vfloat32m1_t _r5l = __riscv_vle32_v_f32m1(r5, vl);
                    vfloat32m1_t _r5h = __riscv_vle32_v_f32m1(r5 + 4, vl);
                    vfloat32m1_t _r6l = __riscv_vle32_v_f32m1(r6, vl);
                    vfloat32m1_t _r6h = __riscv_vle32_v_f32m1(r6 + 4, vl);
                    vfloat32m1_t _r7l = __riscv_vle32_v_f32m1(r7, vl);
                    vfloat32m1_t _r7h = __riscv_vle32_v_f32m1(r7 + 4, vl);
                    transpose8x8_ps(_r0l, _r0h, _r1l, _r1h, _r2l, _r2h, _r3l, _r3h, _r4l, _r4h, _r5l, _r5h, _r6l, _r6h, _r7l, _r7h, vl);
                    __riscv_vse32_v_f32m1(outptr, _r0l, vl);
                    __riscv_vse32_v_f32m1(outptr + 8, _r0h, vl);
                    __riscv_vse32_v_f32m1(outptr + 16, _r1l, vl);
                    __riscv_vse32_v_f32m1(outptr + 24, _r1h, vl);
                    __riscv_vse32_v_f32m1(outptr + 32, _r2l, vl);
                    __riscv_vse32_v_f32m1(outptr + 40, _r2h, vl);
                    __riscv_vse32_v_f32m1(outptr + 48, _r3l, vl);
                    __riscv_vse32_v_f32m1(outptr + 56, _r3h, vl);
                    __riscv_vse32_v_f32m1(outptr + 64, _r4l, vl);
                    __riscv_vse32_v_f32m1(outptr + 72, _r4h, vl);
                    __riscv_vse32_v_f32m1(outptr + 80, _r5l, vl);
                    __riscv_vse32_v_f32m1(outptr + 88, _r5h, vl);
                    __riscv_vse32_v_f32m1(outptr + 96, _r6l, vl);
                    __riscv_vse32_v_f32m1(outptr + 104, _r6h, vl);
                    __riscv_vse32_v_f32m1(outptr + 112, _r7l, vl);
                    __riscv_vse32_v_f32m1(outptr + 120, _r7h, vl);
                    r0 += 8; r1 += 8; r2 += 8; r3 += 8; r4 += 8; r5 += 8; r6 += 8; r7 += 8; outptr += 64;
                }
#endif // __riscv_vector
                for (; i < size; i++)
                {
                    outptr[0] = *r0++;
                    outptr[1] = *r1++;
                    outptr[2] = *r2++;
                    outptr[3] = *r3++;
                    outptr[4] = *r4++;
                    outptr[5] = *r5++;
                    outptr[6] = *r6++;
                    outptr[7] = *r7++;

                    outptr += 8;
                }
            }
        }
        if (pack8to1)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* r0 = bottom_blob.channel(q);

                float* outptr0 = top_blob.channel(q * 8);
                float* outptr1 = top_blob.channel(q * 8 + 1);
                float* outptr2 = top_blob.channel(q * 8 + 2);
                float* outptr3 = top_blob.channel(q * 8 + 3);
                float* outptr4 = top_blob.channel(q * 8 + 4);
                float* outptr5 = top_blob.channel(q * 8 + 5);
                float* outptr6 = top_blob.channel(q * 8 + 6);
                float* outptr7 = top_blob.channel(q * 8 + 7);

                int i = 0;
#if __riscv_vector
                for (; i + 7 < size; i += 8)
                {
                    size_t vl = __riscv_vsetvl_e32m1(4);
                    vfloat32m1_t _r0l = __riscv_vle32_v_f32m1(r0, vl);
                    vfloat32m1_t _r0h = __riscv_vle32_v_f32m1(r0 + 4, vl);
                    vfloat32m1_t _r1l = __riscv_vle32_v_f32m1(r0 + 8, vl);
                    vfloat32m1_t _r1h = __riscv_vle32_v_f32m1(r0 + 12, vl);
                    vfloat32m1_t _r2l = __riscv_vle32_v_f32m1(r0 + 16, vl);
                    vfloat32m1_t _r2h = __riscv_vle32_v_f32m1(r0 + 20, vl);
                    vfloat32m1_t _r3l = __riscv_vle32_v_f32m1(r0 + 24, vl);
                    vfloat32m1_t _r3h = __riscv_vle32_v_f32m1(r0 + 28, vl);
                    vfloat32m1_t _r4l = __riscv_vle32_v_f32m1(r0 + 32, vl);
                    vfloat32m1_t _r4h = __riscv_vle32_v_f32m1(r0 + 36, vl);
                    vfloat32m1_t _r5l = __riscv_vle32_v_f32m1(r0 + 40, vl);
                    vfloat32m1_t _r5h = __riscv_vle32_v_f32m1(r0 + 44, vl);
                    vfloat32m1_t _r6l = __riscv_vle32_v_f32m1(r0 + 48, vl);
                    vfloat32m1_t _r6h = __riscv_vle32_v_f32m1(r0 + 52, vl);
                    vfloat32m1_t _r7l = __riscv_vle32_v_f32m1(r0 + 56, vl);
                    vfloat32m1_t _r7h = __riscv_vle32_v_f32m1(r0 + 60, vl);
                    transpose8x8_ps(_r0l, _r0h, _r1l, _r1h, _r2l, _r2h, _r3l, _r3h, _r4l, _r4h, _r5l, _r5h, _r6l, _r6h, _r7l, _r7h, vl);
                    __riscv_vse32_v_f32m1(outptr0, _r0l, vl);
                    __riscv_vse32_v_f32m1(outptr1, _r0h, vl);
                    __riscv_vse32_v_f32m1(outptr2, _r1l, vl);
                    __riscv_vse32_v_f32m1(outptr3, _r1h, vl);
                    __riscv_vse32_v_f32m1(outptr4, _r2l, vl);
                    __riscv_vse32_v_f32m1(outptr5, _r2h, vl);
                    __riscv_vse32_v_f32m1(outptr6, _r3l, vl);
                    __riscv_vse32_v_f32m1(outptr7, _r3h, vl);

                    r0 += 64;
                    outptr0 += 8;
                    outptr1 += 8;
                    outptr2 += 8;
                    outptr3 += 8;
                    outptr4 += 8;
                    outptr5 += 8;
                    outptr6 += 8;
                    outptr7 += 8;
                }
#endif // __riscv_vector
                for (; i < size; i++)
                {
                    *outptr0++ = r0[0];
                    *outptr1++ = r0[1];
                    *outptr2++ = r0[2];
                    *outptr3++ = r0[3];
                    *outptr4++ = r0[4];
                    *outptr5++ = r0[5];
                    *outptr6++ = r0[6];
                    *outptr7++ = r0[7];

                    r0 += 8;
                }
            }
        }
        if (pack4to8)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < outc; q++)
            {
                const float* r0 = bottom_blob.channel(q * 2);
                const float* r1 = bottom_blob.channel(q * 2 + 1);

                float* outptr = top_blob.channel(q);

                for (int i = 0; i < size; i++)
                {
                    outptr[0] = r0[0];
                    outptr[1] = r0[1];
                    outptr[2] = r0[2];
                    outptr[3] = r0[3];
                    outptr[4] = r1[0];
                    outptr[5] = r1[1];
                    outptr[6] = r1[2];
                    outptr[7] = r1[3];

                    r0 += 4;
                    r1 += 4;
                    outptr += 8;
                }
            }
        }
        if (pack8to4)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* r0 = bottom_blob.channel(q);

                float* outptr0 = top_blob.channel(q * 2);
                float* outptr1 = top_blob.channel(q * 2 + 1);

                for (int i = 0; i < size; i++)
                {
                    outptr0[0] = r0[0];
                    outptr0[1] = r0[1];
                    outptr0[2] = r0[2];
                    outptr0[3] = r0[3];
                    outptr1[0] = r0[4];
                    outptr1[1] = r0[5];
                    outptr1[2] = r0[6];
                    outptr1[3] = r0[7];

                    r0 += 8;
                    outptr0 += 4;
                    outptr1 += 4;
                }
            }
        }
        if (pack1to16)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < outc; q++)
            {
                const float* r0 = bottom_blob.channel(q * 16);
                const float* r1 = bottom_blob.channel(q * 16 + 1);
                const float* r2 = bottom_blob.channel(q * 16 + 2);
                const float* r3 = bottom_blob.channel(q * 16 + 3);
                const float* r4 = bottom_blob.channel(q * 16 + 4);
                const float* r5 = bottom_blob.channel(q * 16 + 5);
                const float* r6 = bottom_blob.channel(q * 16 + 6);
                const float* r7 = bottom_blob.channel(q * 16 + 7);
                const float* r8 = bottom_blob.channel(q * 16 + 8);
                const float* r9 = bottom_blob.channel(q * 16 + 9);
                const float* ra = bottom_blob.channel(q * 16 + 10);
                const float* rb = bottom_blob.channel(q * 16 + 11);
                const float* rc = bottom_blob.channel(q * 16 + 12);
                const float* rd = bottom_blob.channel(q * 16 + 13);
                const float* re = bottom_blob.channel(q * 16 + 14);
                const float* rf = bottom_blob.channel(q * 16 + 15);

                float* outptr = top_blob.channel(q);

                int i = 0;
#if __riscv_vector
                for (; i + 15 < size; i += 16)
                {
                    size_t vl = __riscv_vsetvl_e32m1(4);
                    vfloat32m1_t _r0l = __riscv_vle32_v_f32m1(r0, vl);
                    vfloat32m1_t _r0h = __riscv_vle32_v_f32m1(r0 + 4, vl);
                    vfloat32m1_t _r1l = __riscv_vle32_v_f32m1(r1, vl);
                    vfloat32m1_t _r1h = __riscv_vle32_v_f32m1(r1 + 4, vl);
                    vfloat32m1_t _r2l = __riscv_vle32_v_f32m1(r2, vl);
                    vfloat32m1_t _r2h = __riscv_vle32_v_f32m1(r2 + 4, vl);
                    vfloat32m1_t _r3l = __riscv_vle32_v_f32m1(r3, vl);
                    vfloat32m1_t _r3h = __riscv_vle32_v_f32m1(r3 + 4, vl);
                    vfloat32m1_t _r4l = __riscv_vle32_v_f32m1(r4, vl);
                    vfloat32m1_t _r4h = __riscv_vle32_v_f32m1(r4 + 4, vl);
                    vfloat32m1_t _r5l = __riscv_vle32_v_f32m1(r5, vl);
                    vfloat32m1_t _r5h = __riscv_vle32_v_f32m1(r5 + 4, vl);
                    vfloat32m1_t _r6l = __riscv_vle32_v_f32m1(r6, vl);
                    vfloat32m1_t _r6h = __riscv_vle32_v_f32m1(r6 + 4, vl);
                    vfloat32m1_t _r7l = __riscv_vle32_v_f32m1(r7, vl);
                    vfloat32m1_t _r7h = __riscv_vle32_v_f32m1(r7 + 4, vl);
                    vfloat32m1_t _r8l = __riscv_vle32_v_f32m1(r8, vl);
                    vfloat32m1_t _r8h = __riscv_vle32_v_f32m1(r8 + 4, vl);
                    vfloat32m1_t _r9l = __riscv_vle32_v_f32m1(r9, vl);
                    vfloat32m1_t _r9h = __riscv_vle32_v_f32m1(r9 + 4, vl);
                    vfloat32m1_t _ral = __riscv_vle32_v_f32m1(ra, vl);
                    vfloat32m1_t _rah = __riscv_vle32_v_f32m1(ra + 4, vl);
                    vfloat32m1_t _rbl = __riscv_vle32_v_f32m1(rb, vl);
                    vfloat32m1_t _rbh = __riscv_vle32_v_f32m1(rb + 4, vl);
                    vfloat32m1_t _rcl = __riscv_vle32_v_f32m1(rc, vl);
                    vfloat32m1_t _rch = __riscv_vle32_v_f32m1(rc + 4, vl);
                    vfloat32m1_t _rdl = __riscv_vle32_v_f32m1(rd, vl);
                    vfloat32m1_t _rdh = __riscv_vle32_v_f32m1(rd + 4, vl);
                    vfloat32m1_t _rel = __riscv_vle32_v_f32m1(re, vl);
                    vfloat32m1_t _reh = __riscv_vle32_v_f32m1(re + 4, vl);
                    vfloat32m1_t _rfl = __riscv_vle32_v_f32m1(rf, vl);
                    vfloat32m1_t _rfh = __riscv_vle32_v_f32m1(rf + 4, vl);
                    transpose8x16_ps(_r0l, _r0h, _r1l, _r1h, _r2l, _r2h, _r3l, _r3h, _r4l, _r4h, _r5l, _r5h, _r6l, _r6h, _r7l, _r7h, _r8l, _r8h, _r9l, _r9h, _ral, _rah, _rbl, _rbh, _rcl, _rch, _rdl, _rdh, _rel, _reh, _rfl, _rfh, vl);
                    __riscv_vse32_v_f32m1(outptr, _r0l, vl);
                    __riscv_vse32_v_f32m1(outptr + 16, _r0h, vl);
                    __riscv_vse32_v_f32m1(outptr + 32, _r1l, vl);
                    __riscv_vse32_v_f32m1(outptr + 48, _r1h, vl);
                    __riscv_vse32_v_f32m1(outptr + 64, _r2l, vl);
                    __riscv_vse32_v_f32m1(outptr + 80, _r2h, vl);
                    __riscv_vse32_v_f32m1(outptr + 96, _r3l, vl);
                    __riscv_vse32_v_f32m1(outptr + 112, _r3h, vl);
                    __riscv_vse32_v_f32m1(outptr + 128, _r4l, vl);
                    __riscv_vse32_v_f32m1(outptr + 144, _r4h, vl);
                    __riscv_vse32_v_f32m1(outptr + 160, _r5l, vl);
                    __riscv_vse32_v_f32m1(outptr + 176, _r5h, vl);
                    __riscv_vse32_v_f32m1(outptr + 192, _r6l, vl);
                    __riscv_vse32_v_f32m1(outptr + 208, _r6h, vl);
                    __riscv_vse32_v_f32m1(outptr + 224, _r7l, vl);
                    __riscv_vse32_v_f32m1(outptr + 240, _r7h, vl);
                    __riscv_vse32_v_f32m1(outptr + 256, _r8l, vl);
                    __riscv_vse32_v_f32m1(outptr + 272, _r8h, vl);
                    __riscv_vse32_v_f32m1(outptr + 288, _r9l, vl);
                    __riscv_vse32_v_f32m1(outptr + 304, _r9h, vl);
                    __riscv_vse32_v_f32m1(outptr + 320, _ral, vl);
                    __riscv_vse32_v_f32m1(outptr + 336, _rah, vl);
                    __riscv_vse32_v_f32m1(outptr + 352, _rbl, vl);
                    __riscv_vse32_v_f32m1(outptr + 368, _rbh, vl);
                    __riscv_vse32_v_f32m1(outptr + 384, _rcl, vl);
                    __riscv_vse32_v_f32m1(outptr + 400, _rch, vl);
                    __riscv_vse32_v_f32m1(outptr + 416, _rdl, vl);
                    __riscv_vse32_v_f32m1(outptr + 432, _rdh, vl);
                    __riscv_vse32_v_f32m1(outptr + 448, _rel, vl);
                    __riscv_vse32_v_f32m1(outptr + 464, _reh, vl);
                    __riscv_vse32_v_f32m1(outptr + 480, _rfl, vl);
                    __riscv_vse32_v_f32m1(outptr + 496, _rfh, vl);
                    r0 += 16; r1 += 16; r2 += 16; r3 += 16; r4 += 16; r5 += 16; r6 += 16; r7 += 16; r8 += 16; r9 += 16; ra += 16; rb += 16; rc += 16; rd += 16; re += 16; rf += 16; outptr += 256;
                }
#endif // __riscv_vector
                for (; i < size; i++)
                {
                    outptr[0] = *r0++;
                    outptr[1] = *r1++;
                    outptr[2] = *r2++;
                    outptr[3] = *r3++;
                    outptr[4] = *r4++;
                    outptr[5] = *r5++;
                    outptr[6] = *r6++;
                    outptr[7] = *r7++;
                    outptr[8] = *r8++;
                    outptr[9] = *r9++;
                    outptr[10] = *ra++;
                    outptr[11] = *rb++;
                    outptr[12] = *rc++;
                    outptr[13] = *rd++;
                    outptr[14] = *re++;
                    outptr[15] = *rf++;

                    outptr += 16;
                }
            }
        }
        if (pack16to1)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* r0 = bottom_blob.channel(q);

                float* outptr0 = top_blob.channel(q * 16);
                float* outptr1 = top_blob.channel(q * 16 + 1);
                float* outptr2 = top_blob.channel(q * 16 + 2);
                float* outptr3 = top_blob.channel(q * 16 + 3);
                float* outptr4 = top_blob.channel(q * 16 + 4);
                float* outptr5 = top_blob.channel(q * 16 + 5);
                float* outptr6 = top_blob.channel(q * 16 + 6);
                float* outptr7 = top_blob.channel(q * 16 + 7);
                float* outptr8 = top_blob.channel(q * 16 + 8);
                float* outptr9 = top_blob.channel(q * 16 + 9);
                float* outptra = top_blob.channel(q * 16 + 10);
                float* outptrb = top_blob.channel(q * 16 + 11);
                float* outptrc = top_blob.channel(q * 16 + 12);
                float* outptrd = top_blob.channel(q * 16 + 13);
                float* outptre = top_blob.channel(q * 16 + 14);
                float* outptrf = top_blob.channel(q * 16 + 15);

                int i = 0;
#if __riscv_vector
                for (; i + 15 < size; i += 16)
                {
                    size_t vl = __riscv_vsetvl_e32m1(4);
                    vfloat32m1_t _r0l = __riscv_vle32_v_f32m1(r0, vl);
                    vfloat32m1_t _r0h = __riscv_vle32_v_f32m1(r0 + 4, vl);
                    vfloat32m1_t _r1l = __riscv_vle32_v_f32m1(r0 + 8, vl);
                    vfloat32m1_t _r1h = __riscv_vle32_v_f32m1(r0 + 12, vl);
                    vfloat32m1_t _r2l = __riscv_vle32_v_f32m1(r0 + 16, vl);
                    vfloat32m1_t _r2h = __riscv_vle32_v_f32m1(r0 + 20, vl);
                    vfloat32m1_t _r3l = __riscv_vle32_v_f32m1(r0 + 24, vl);
                    vfloat32m1_t _r3h = __riscv_vle32_v_f32m1(r0 + 28, vl);
                    vfloat32m1_t _r4l = __riscv_vle32_v_f32m1(r0 + 32, vl);
                    vfloat32m1_t _r4h = __riscv_vle32_v_f32m1(r0 + 36, vl);
                    vfloat32m1_t _r5l = __riscv_vle32_v_f32m1(r0 + 40, vl);
                    vfloat32m1_t _r5h = __riscv_vle32_v_f32m1(r0 + 44, vl);
                    vfloat32m1_t _r6l = __riscv_vle32_v_f32m1(r0 + 48, vl);
                    vfloat32m1_t _r6h = __riscv_vle32_v_f32m1(r0 + 52, vl);
                    vfloat32m1_t _r7l = __riscv_vle32_v_f32m1(r0 + 56, vl);
                    vfloat32m1_t _r7h = __riscv_vle32_v_f32m1(r0 + 60, vl);
                    vfloat32m1_t _r8l = __riscv_vle32_v_f32m1(r0 + 64, vl);
                    vfloat32m1_t _r8h = __riscv_vle32_v_f32m1(r0 + 68, vl);
                    vfloat32m1_t _r9l = __riscv_vle32_v_f32m1(r0 + 72, vl);
                    vfloat32m1_t _r9h = __riscv_vle32_v_f32m1(r0 + 76, vl);
                    vfloat32m1_t _ral = __riscv_vle32_v_f32m1(r0 + 80, vl);
                    vfloat32m1_t _rah = __riscv_vle32_v_f32m1(r0 + 84, vl);
                    vfloat32m1_t _rbl = __riscv_vle32_v_f32m1(r0 + 88, vl);
                    vfloat32m1_t _rbh = __riscv_vle32_v_f32m1(r0 + 92, vl);
                    vfloat32m1_t _rcl = __riscv_vle32_v_f32m1(r0 + 96, vl);
                    vfloat32m1_t _rch = __riscv_vle32_v_f32m1(r0 + 100, vl);
                    vfloat32m1_t _rdl = __riscv_vle32_v_f32m1(r0 + 104, vl);
                    vfloat32m1_t _rdh = __riscv_vle32_v_f32m1(r0 + 108, vl);
                    vfloat32m1_t _rel = __riscv_vle32_v_f32m1(r0 + 112, vl);
                    vfloat32m1_t _reh = __riscv_vle32_v_f32m1(r0 + 116, vl);
                    vfloat32m1_t _rfl = __riscv_vle32_v_f32m1(r0 + 120, vl);
                    vfloat32m1_t _rfh = __riscv_vle32_v_f32m1(r0 + 124, vl);
                    transpose8x16_ps(_r0l, _r0h, _r1l, _r1h, _r2l, _r2h, _r3l, _r3h, _r4l, _r4h, _r5l, _r5h, _r6l, _r6h, _r7l, _r7h, _r8l, _r8h, _r9l, _r9h, _ral, _rah, _rbl, _rbh, _rcl, _rch, _rdl, _rdh, _rel, _reh, _rfl, _rfh, vl);
                    __riscv_vse32_v_f32m1(outptr0, _r0l, vl);
                    __riscv_vse32_v_f32m1(outptr1, _r0h, vl);
                    __riscv_vse32_v_f32m1(outptr2, _r1l, vl);
                    __riscv_vse32_v_f32m1(outptr3, _r1h, vl);
                    __riscv_vse32_v_f32m1(outptr4, _r2l, vl);
                    __riscv_vse32_v_f32m1(outptr5, _r2h, vl);
                    __riscv_vse32_v_f32m1(outptr6, _r3l, vl);
                    __riscv_vse32_v_f32m1(outptr7, _r3h, vl);
                    __riscv_vse32_v_f32m1(outptr8, _r4l, vl);
                    __riscv_vse32_v_f32m1(outptr9, _r4h, vl);
                    __riscv_vse32_v_f32m1(outptra, _r5l, vl);
                    __riscv_vse32_v_f32m1(outptrb, _r5h, vl);
                    __riscv_vse32_v_f32m1(outptrc, _r6l, vl);
                    __riscv_vse32_v_f32m1(outptrd, _r6h, vl);
                    __riscv_vse32_v_f32m1(outptre, _r7l, vl);
                    __riscv_vse32_v_f32m1(outptrf, _r7h, vl);

                    r0 += 256;
                    outptr0 += 16; outptr1 += 16; outptr2 += 16; outptr3 += 16; outptr4 += 16; outptr5 += 16; outptr6 += 16; outptr7 += 16; outptr8 += 16; outptr9 += 16; outptra += 16; outptrb += 16; outptrc += 16; outptrd += 16; outptre += 16; outptrf += 16;
                }
#endif // __riscv_vector

                for (; i < size; i++)
                {
                    *outptr0++ = r0[0];
                    *outptr1++ = r0[1];
                    *outptr2++ = r0[2];
                    *outptr3++ = r0[3];
                    *outptr4++ = r0[4];
                    *outptr5++ = r0[5];
                    *outptr6++ = r0[6];
                    *outptr7++ = r0[7];
                    *outptr8++ = r0[8];
                    *outptr9++ = r0[9];
                    *outptra++ = r0[10];
                    *outptrb++ = r0[11];
                    *outptrc++ = r0[12];
                    *outptrd++ = r0[13];
                    *outptre++ = r0[14];
                    *outptrf++ = r0[15];

                    r0 += 16;
                }
            }
        }
        if (pack4to16)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < outc; q++)
            {
                const float* r0 = bottom_blob.channel(q * 4);
                const float* r1 = bottom_blob.channel(q * 4 + 1);
                const float* r2 = bottom_blob.channel(q * 4 + 2);
                const float* r3 = bottom_blob.channel(q * 4 + 3);

                float* outptr = top_blob.channel(q);

                for (int i = 0; i < size; i++)
                {
                    outptr[0] = r0[0];
                    outptr[1] = r0[1];
                    outptr[2] = r0[2];
                    outptr[3] = r0[3];
                    outptr[4] = r1[0];
                    outptr[5] = r1[1];
                    outptr[6] = r1[2];
                    outptr[7] = r1[3];
                    outptr[8] = r2[0];
                    outptr[9] = r2[1];
                    outptr[10] = r2[2];
                    outptr[11] = r2[3];
                    outptr[12] = r3[0];
                    outptr[13] = r3[1];
                    outptr[14] = r3[2];
                    outptr[15] = r3[3];

                    r0 += 4;
                    r1 += 4;
                    r2 += 4;
                    r3 += 4;
                    outptr += 16;
                }
            }
        }
        if (pack16to4)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* r0 = bottom_blob.channel(q);

                float* outptr0 = top_blob.channel(q * 4);
                float* outptr1 = top_blob.channel(q * 4 + 1);
                float* outptr2 = top_blob.channel(q * 4 + 2);
                float* outptr3 = top_blob.channel(q * 4 + 3);

                for (int i = 0; i < size; i++)
                {
                    outptr0[0] = r0[0];
                    outptr0[1] = r0[1];
                    outptr0[2] = r0[2];
                    outptr0[3] = r0[3];
                    outptr1[0] = r0[4];
                    outptr1[1] = r0[5];
                    outptr1[2] = r0[6];
                    outptr1[3] = r0[7];
                    outptr2[0] = r0[8];
                    outptr2[1] = r0[9];
                    outptr2[2] = r0[10];
                    outptr2[3] = r0[11];
                    outptr3[0] = r0[12];
                    outptr3[1] = r0[13];
                    outptr3[2] = r0[14];
                    outptr3[3] = r0[15];

                    r0 += 16;
                    outptr0 += 4;
                    outptr1 += 4;
                    outptr2 += 4;
                    outptr3 += 4;
                }
            }
        }
        if (pack8to16)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < outc; q++)
            {
                const float* r0 = bottom_blob.channel(q * 2);
                const float* r1 = bottom_blob.channel(q * 2 + 1);

                float* outptr = top_blob.channel(q);

                for (int i = 0; i < size; i++)
                {
                    outptr[0] = r0[0];
                    outptr[1] = r0[1];
                    outptr[2] = r0[2];
                    outptr[3] = r0[3];
                    outptr[4] = r0[4];
                    outptr[5] = r0[5];
                    outptr[6] = r0[6];
                    outptr[7] = r0[7];
                    outptr[8] = r1[0];
                    outptr[9] = r1[1];
                    outptr[10] = r1[2];
                    outptr[11] = r1[3];
                    outptr[12] = r1[4];
                    outptr[13] = r1[5];
                    outptr[14] = r1[6];
                    outptr[15] = r1[7];

                    r0 += 8;
                    r1 += 8;
                    outptr += 16;
                }
            }
        }
        if (pack16to8)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* r0 = bottom_blob.channel(q);

                float* outptr0 = top_blob.channel(q * 2);
                float* outptr1 = top_blob.channel(q * 2 + 1);

                for (int i = 0; i < size; i++)
                {
                    outptr0[0] = r0[0];
                    outptr0[1] = r0[1];
                    outptr0[2] = r0[2];
                    outptr0[3] = r0[3];
                    outptr0[4] = r0[4];
                    outptr0[5] = r0[5];
                    outptr0[6] = r0[6];
                    outptr0[7] = r0[7];
                    outptr1[0] = r0[8];
                    outptr1[1] = r0[9];
                    outptr1[2] = r0[10];
                    outptr1[3] = r0[11];
                    outptr1[4] = r0[12];
                    outptr1[5] = r0[13];
                    outptr1[6] = r0[14];
                    outptr1[7] = r0[15];

                    r0 += 16;
                    outptr0 += 8;
                    outptr1 += 8;
                }
            }
        }

        return 0;
    }

    return 0;
}

int Packing_riscv::forward_int8(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    if (use_padding)
    {
        return Packing::forward(bottom_blob, top_blob, opt);
    }

    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    if (elempack == out_elempack)
    {
        top_blob = bottom_blob;
        return 0;
    }

    bool pack1to8 = elempack == 1 && out_elempack == 8;
    bool pack8to1 = elempack == 8 && out_elempack == 1;

    if (!pack1to8 && !pack8to1)
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
        // identity if use_padding not allowed
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

        if (pack1to8)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < outh; i++)
            {
                const signed char* r0 = bottom_blob.row<const signed char>(i * 8);
                const signed char* r1 = bottom_blob.row<const signed char>(i * 8 + 1);
                const signed char* r2 = bottom_blob.row<const signed char>(i * 8 + 2);
                const signed char* r3 = bottom_blob.row<const signed char>(i * 8 + 3);
                const signed char* r4 = bottom_blob.row<const signed char>(i * 8 + 4);
                const signed char* r5 = bottom_blob.row<const signed char>(i * 8 + 5);
                const signed char* r6 = bottom_blob.row<const signed char>(i * 8 + 6);
                const signed char* r7 = bottom_blob.row<const signed char>(i * 8 + 7);

                signed char* outptr = top_blob.row<signed char>(i);

                int j = 0;
                for (; j < w; j++)
                {
                    outptr[0] = *r0++;
                    outptr[1] = *r1++;
                    outptr[2] = *r2++;
                    outptr[3] = *r3++;
                    outptr[4] = *r4++;
                    outptr[5] = *r5++;
                    outptr[6] = *r6++;
                    outptr[7] = *r7++;

                    outptr += 8;
                }
            }
        }
        if (pack8to1)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                const signed char* r0 = bottom_blob.row<const signed char>(i);

                signed char* outptr0 = top_blob.row<signed char>(i * 8);
                signed char* outptr1 = top_blob.row<signed char>(i * 8 + 1);
                signed char* outptr2 = top_blob.row<signed char>(i * 8 + 2);
                signed char* outptr3 = top_blob.row<signed char>(i * 8 + 3);
                signed char* outptr4 = top_blob.row<signed char>(i * 8 + 4);
                signed char* outptr5 = top_blob.row<signed char>(i * 8 + 5);
                signed char* outptr6 = top_blob.row<signed char>(i * 8 + 6);
                signed char* outptr7 = top_blob.row<signed char>(i * 8 + 7);

                int j = 0;
                for (; j < w; j++)
                {
                    *outptr0++ = r0[0];
                    *outptr1++ = r0[1];
                    *outptr2++ = r0[2];
                    *outptr3++ = r0[3];
                    *outptr4++ = r0[4];
                    *outptr5++ = r0[5];
                    *outptr6++ = r0[6];
                    *outptr7++ = r0[7];

                    r0 += 8;
                }
            }
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
        else // if (dims == 4)
            top_blob.create(w, h, d, outc, out_elemsize, out_elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        if (pack1to8)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < outc; q++)
            {
                const signed char* r0 = bottom_blob.channel(q * 8);
                const signed char* r1 = bottom_blob.channel(q * 8 + 1);
                const signed char* r2 = bottom_blob.channel(q * 8 + 2);
                const signed char* r3 = bottom_blob.channel(q * 8 + 3);
                const signed char* r4 = bottom_blob.channel(q * 8 + 4);
                const signed char* r5 = bottom_blob.channel(q * 8 + 5);
                const signed char* r6 = bottom_blob.channel(q * 8 + 6);
                const signed char* r7 = bottom_blob.channel(q * 8 + 7);

                signed char* outptr = top_blob.channel(q);

                int i = 0;
                for (; i < size; i++)
                {
                    outptr[0] = *r0++;
                    outptr[1] = *r1++;
                    outptr[2] = *r2++;
                    outptr[3] = *r3++;
                    outptr[4] = *r4++;
                    outptr[5] = *r5++;
                    outptr[6] = *r6++;
                    outptr[7] = *r7++;

                    outptr += 8;
                }
            }
        }
        if (pack8to1)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const signed char* r0 = bottom_blob.channel(q);

                signed char* outptr0 = top_blob.channel(q * 8);
                signed char* outptr1 = top_blob.channel(q * 8 + 1);
                signed char* outptr2 = top_blob.channel(q * 8 + 2);
                signed char* outptr3 = top_blob.channel(q * 8 + 3);
                signed char* outptr4 = top_blob.channel(q * 8 + 4);
                signed char* outptr5 = top_blob.channel(q * 8 + 5);
                signed char* outptr6 = top_blob.channel(q * 8 + 6);
                signed char* outptr7 = top_blob.channel(q * 8 + 7);

                int i = 0;
                for (; i < size; i++)
                {
                    *outptr0++ = r0[0];
                    *outptr1++ = r0[1];
                    *outptr2++ = r0[2];
                    *outptr3++ = r0[3];
                    *outptr4++ = r0[4];
                    *outptr5++ = r0[5];
                    *outptr6++ = r0[6];
                    *outptr7++ = r0[7];

                    r0 += 8;
                }
            }
        }

        return 0;
    }

    return 0;
}

} // namespace ncnn
